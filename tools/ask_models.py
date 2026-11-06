#!/usr/bin/env python3
"""Hỏi mô hình audio (Qwen3-Omni qua vLLM, API kiểu OpenAI) đúng bộ câu hỏi của trang gán nhãn.

Prompt và JSON schema sinh tự động từ data/config.json, nên luôn khớp câu hỏi trên trang.
Đầu ra: JSONL theo định dạng của audit_questions.py:
  {"sample_id": ..., "annotator": "<model>", "answers": {"<panel>.<qid>": "<code>"}, "parse_error": bool, "reasoning": "..."}

Chạy:
  python3 tools/ask_models.py --manifest manifests/10154_2.jsonl --clips clips --model qwen_instruct --out out/qwen_instruct.jsonl [models=local]
  python3 tools/ask_models.py --manifest manifests/10154_2.jsonl --dry-run          # chỉ in prompt + schema, không cần GPU
  python3 tools/ask_models.py --manifest manifests/10154_2.jsonl --fake --out out/fake.jsonl   # đáp án ngẫu nhiên để thử stats/notebook
"""
import argparse, base64, glob, json, os, random, re, sys, urllib.request

import ui

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QCFG = os.path.join(ROOT, "data", "config.json")

# Câu mức file không có trong bộ câu hỏi: dùng để xếp bucket (xem collect.py).
FILE_QS = ["tstyle.genre", "tstyle.accent", "tenv.channel", "tenv.noise_type", "tenv.distance"]


def load_jsonl(path):
    with open(path, encoding="utf8") as f:
        return [json.loads(l) for l in f if l.strip()]


def questions(tag="turntaking", keys=None, cfg_path=QCFG):
    """[(key 'panel.qid', q)] theo thứ tự trên trang. Chỉ lấy câu có codes (để ép enum)."""
    with open(cfg_path, encoding="utf8") as f:
        panels = json.load(f)["panels"][tag]
    out = []
    for p in panels:
        for q in p.get("qs", []):
            k = p["id"] + "." + q["id"]
            if q.get("codes") and (keys is None or k in keys):
                out.append((k, q))
    if keys:
        missing = set(keys) - {k for k, _ in out}
        assert not missing, f"không có trong config: {missing}"
    return out


def schema(qs):
    return {"type": "object", "additionalProperties": False, "required": [k for k, _ in qs],
            "properties": {k: {"type": "string", "enum": q["codes"]} for k, q in qs}}


SYSTEM = (
    "Bạn là người gán nhãn audio tiếng Việt cẩn thận. Bạn nghe audio và trả lời các câu hỏi trắc nghiệm, "
    "chọn theo những gì thực sự nghe được, không đoán theo nội dung. Nếu không nghe rõ hoặc không chắc, "
    "chọn mã UNK khi câu hỏi có mã đó. Chỉ trả về một JSON object, khoá là mã câu hỏi, giá trị là mã đáp án."
)


def question_block(qs):
    lines = []
    for i, (k, q) in enumerate(qs, 1):
        lines.append(f"{i}. [{k}] {q['title']}")
        if q.get("desc"):
            lines.append(f"   ({q['desc']})")
        for code, opt in zip(q["codes"], q["opts"]):
            lines.append(f"   - {code}: {opt[0]}" + (f" – {opt[1]}" if len(opt) > 1 else ""))
    return "\n".join(lines)


def segment_context(row):
    # chỉ đưa một mốc trung tính: tên trường trong `times` (t_pause_start, t_bc_start...) sẽ lộ đáp án
    t = next(iter((row.get("times") or {}).values()), None)
    if t is None:
        t = (row.get("marks_auto") or {}).get("turn_end_sec")
    s = (f"Đây là một đoạn hội thoại. “Người dùng” là {row.get('user', 'người nói chính')}. "
         f"Sự kiện cần xét nằm ở khoảng giây {t:.2f} tính từ đầu clip. " if t is not None else
         f"Đây là một đoạn hội thoại. “Người dùng” là {row.get('user', 'người nói chính')}. Sự kiện cần xét nằm gần cuối đoạn. ")
    s += ("Hãy nghe cả đoạn để hiểu ngữ cảnh. “Người dùng” là người đang nói (hoặc bắt đầu nói) ngay tại mốc đó; "
         "chỉ xét lượt nói của người dùng tại sự kiện đó.")
    if row.get("user_text"):
        s += f"\nLời của người dùng (máy chép, có thể sai): “{row['user_text']}”"
    return s


def wav_part(path):
    with open(path, "rb") as f:
        b = base64.b64encode(f.read()).decode()
    return {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64," + b}}


def parse(text, qs):
    """Lấy JSON cuối cùng trong câu trả lời; giá trị ngoài codes coi là thiếu."""
    reasoning = ""
    m = re.search(r"<think>(.*?)</think>", text, re.S)
    if m:
        reasoning, text = m.group(1).strip(), text[m.end():]
    objs = re.findall(r"\{.*\}", text, re.S)
    try:
        raw = json.loads(objs[-1]) if objs else {}
    except ValueError:
        raw = {}
    ans = {k: raw[k] for k, q in qs if raw.get(k) in q["codes"]}
    return ans, len(ans) < len(qs), reasoning


def ask(model, audios, context, qs, timeout=600):
    """Một lần gọi /chat/completions. model: một mục trong config models (base_url, served_name...)."""
    content = [wav_part(a) for a in audios]
    content.append({"type": "text", "text": context + "\n\nCâu hỏi:\n" + question_block(qs) + "\n\nTrả lời bằng JSON."})
    body = {"model": model["served_name"], "temperature": model.get("temperature", 0),
            "max_tokens": model.get("max_tokens", 4096),
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]}
    if model.get("json_schema", True):
        body["response_format"] = {"type": "json_schema", "json_schema": {"name": "answers", "schema": schema(qs)}}
    req = urllib.request.Request(model["base_url"].rstrip("/") + "/chat/completions",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        msg = json.load(r)["choices"][0]["message"]
    ans, err, reasoning = parse(msg.get("content") or "", qs)
    return ans, err, msg.get("reasoning_content") or reasoning


def clip_path(row, clips):
    if row.get("audio"):
        p = row["audio"]
        return p if os.path.isabs(p) else os.path.join(ROOT, p)
    return os.path.join(clips, row["event_id"] + ".wav")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", nargs="+", required=True, help="file hoặc glob manifests/*.jsonl")
    ap.add_argument("--model", help="khoá trong config/models, vd. qwen_instruct")
    ap.add_argument("--tag", default="turntaking")
    ap.add_argument("--out")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--fake", action="store_true", help="đáp án ngẫu nhiên, không gọi mô hình")
    import conf
    argv, overrides = conf.split_argv()
    a = ap.parse_args(argv)
    C = conf.load(overrides)
    a.clips = conf.path(C, "clips")

    rows = [r for m in a.manifest for p in sorted(glob.glob(m)) for r in load_jsonl(p)][: a.limit]
    qs = questions(a.tag)
    if a.dry_run:
        r = rows[0]
        print("SYSTEM:\n" + SYSTEM + "\n\nUSER:\n[audio] " + clip_path(r, a.clips) + "\n" + segment_context(r)
              + "\n\nCâu hỏi:\n" + question_block(qs))
        print("\nSCHEMA:\n" + json.dumps(schema(qs), ensure_ascii=False, indent=1))
        return
    if not a.out or not (a.model or a.fake):
        ap.error("cần --out và --model (hoặc --fake)")
    name = "fake" if a.fake else a.model
    model = None if a.fake else C["models"][a.model]
    done = {r["sample_id"] for r in load_jsonl(a.out)} if os.path.exists(a.out) else set()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    rnd = random.Random(0)
    todo = [r for r in rows if r["event_id"] not in done]
    st = {"đã có từ trước": len(rows) - len(todo), "mới": 0, "lỗi parse": 0, "lỗi gọi mô hình": 0}
    with open(a.out, "a", encoding="utf8") as f, ui.progress() as pg:
        task = pg.add_task(name, total=len(todo))
        for r in todo:
            sid = r["event_id"]
            pg.advance(task)
            if a.fake:
                ans, err, why = {k: rnd.choice(q["codes"]) for k, q in qs}, False, ""
            else:
                try:
                    ans, err, why = ask(model, [clip_path(r, a.clips)], segment_context(r), qs)
                except Exception as e:  # lỗi một mẫu không làm dừng cả lượt chạy
                    st["lỗi gọi mô hình"] += 1
                    pg.console.print(f"[red]lỗi[/] {sid}: {e}")
                    continue
            f.write(json.dumps({"sample_id": sid, "annotator": name, "answers": ans,
                                "parse_error": err, "reasoning": why}, ensure_ascii=False) + "\n")
            f.flush()
            st["mới"] += 1
            st["lỗi parse"] += err
    ui.table(f"{name} → {a.out}", ["", ">số mẫu"], list(st.items()))


if __name__ == "__main__":
    main()
