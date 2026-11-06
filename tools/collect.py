#!/usr/bin/env python3
"""Lấy audio từ các thư mục nguồn trên server cho đến khi đủ quota của từng bucket.

Domain và thuộc tính KHÔNG lấy từ đường dẫn: mỗi file được chuẩn hoá → segment.py (VAD, diarization, ASR)
→ Qwen3-Omni phân loại (genre, channel, noise...) → xếp vào bucket khớp `match` và còn thiếu quota
→ build_manifest tạo segment + clip.

Chạy (Hydra, config ở config/):
  python3 tools/collect.py dry_run=true                    # chỉ liệt kê file nguồn và quota còn thiếu
  python3 tools/collect.py models=local collect=local      # dùng config/models/local.yaml, config/collect/local.yaml
  python3 tools/collect.py collect.buckets.0.target_minutes=60
Chạy lại tiếp tục từ chỗ dừng (đọc raw_index.jsonl, classified.jsonl). Tăng target_minutes để lấy thêm.
"""
import hashlib, json, os, random, subprocess, sys

import hydra
from omegaconf import OmegaConf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import ask_models, build_manifest, conf, segment, ui  # noqa: E402

EXT = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".mp4", ".mkv", ".webm", ".mov", ".avi"}


def jl(path):
    return ask_models.load_jsonl(path) if os.path.exists(path) else []


def append(path, row):
    with open(path, "a", encoding="utf8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def fid(path):
    """id ổn định theo nội dung: hash 4 MB đầu + kích thước (đủ để bỏ trùng, không phải đọc hết file)."""
    h = hashlib.sha1(str(os.path.getsize(path)).encode())
    with open(path, "rb") as f:
        h.update(f.read(4 << 20))
    return h.hexdigest()[:16]


def files(sources, seed):
    out = []
    for s in sources:
        if os.path.isfile(s):
            out.append(s)
        for d, _, fs in os.walk(s):
            out += [os.path.join(d, f) for f in fs if os.path.splitext(f)[1].lower() in EXT]
    out = sorted(set(out))
    random.Random(seed).shuffle(out)
    return out


def normalize(src, dst, max_min):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-t", str(max_min * 60),
                    "-vn", "-ac", "1", "-ar", "16000", dst], check=True)


def excerpts(wav, work, tr, n, dur):
    """n đoạn ~dur giây rải đều trong file, bắt đầu tại đầu một lượt nói."""
    if not tr:
        return []
    picks = [tr[int(i * len(tr) / n)]["s"] for i in range(n)]
    out = []
    for i, s in enumerate(picks):
        p = os.path.join(work, f"_ex{i}.wav")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(s), "-t", str(dur), "-i", wav, p], check=True)
        out.append(p)
    return out


def classify(model, wav, work, tr, k):
    qs = ask_models.questions("turntaking", ask_models.FILE_QS)
    text = " / ".join(f"{t['spk']}: {t['text']}" for t in tr)[:k["transcript_chars"]]
    ctx = ("Bạn nghe vài đoạn trích từ cùng một file audio. Trả lời các câu hỏi cho cả file (không phải một lượt nói). "
           "Câu về giọng hoặc tuổi thì xét người nói chính.\nTranscript rút gọn (máy chép, có thể sai): " + text)
    ans, err, _ = ask_models.ask(model, excerpts(wav, work, tr, k["n_excerpts"], k["excerpt_sec"]), ctx, qs)
    return {k.split(".", 1)[1]: v for k, v in ans.items()}, err


def n_speakers(tr, min_share=0.1):
    tot = {}
    for t in tr:
        tot[t["spk"]] = tot.get(t["spk"], 0) + t["e"] - t["s"]
    s = sum(tot.values()) or 1
    return sum(v / s >= min_share for v in tot.values())


def matches(match, labels):
    for k, want in match.items():
        v = labels.get(k)
        if isinstance(want, list) and v not in want:
            return False
        if not isinstance(want, list) and v != want:
            return False
    return True


@hydra.main(config_path=conf.CONF_DIR, config_name="config", version_base=None)
def main(C):
    C = OmegaConf.to_container(C, resolve=True)
    cfg, models = C["collect"], C["models"]
    out = conf.path(C, "raw")
    work, fdir = os.path.join(out, "_work"), os.path.join(out, "files")
    clips, mdir = conf.path(C, "clips"), conf.path(C, "manifests")
    for d in (work, fdir, mdir):
        os.makedirs(d, exist_ok=True)
    idx_p, cls_p = os.path.join(out, "raw_index.jsonl"), os.path.join(out, "classified.jsonl")

    have = {b["name"]: 0.0 for b in cfg["buckets"]}
    for r in jl(idx_p):
        for b in r["buckets"]:
            have[b] = have.get(b, 0.0) + r["speech_min"]
    cls = {r["id"]: r for r in jl(cls_p)}
    need = lambda: {b["name"]: b["target_minutes"] - have[b["name"]] for b in cfg["buckets"] if have[b["name"]] < b["target_minutes"]}
    srcs = files(cfg["sources"], cfg["seed"])

    def bucket_table(title):
        ui.table(title, ["bucket", "mô tả", ">có (phút)", ">mục tiêu", ">còn thiếu", "đạt"],
                 [(b["name"], b["desc"], f"{have[b['name']]:.1f}", b["target_minutes"],
                   f"{max(0, b['target_minutes'] - have[b['name']]):.1f}", "✓" if have[b["name"]] >= b["target_minutes"] else "✗")
                  for b in cfg["buckets"]])

    ui.console.print(f"[bold]{len(srcs)}[/] file nguồn trong {len(cfg['sources'])} thư mục")
    bucket_table("Quota trước khi chạy")
    if C["dry_run"]:
        return
    max_min = cfg["max_minutes_per_file"]
    total_need = sum(need().values())
    st = {"lấy": 0, "bỏ (không bucket cần)": 0, "bỏ qua (đã xử lý)": 0, "lỗi": 0, "segment": 0}

    with ui.progress() as pg:
        t_files = pg.add_task("file nguồn", total=len(srcs))
        t_min = pg.add_task("phút còn thiếu", total=round(total_need, 1))
        for src in srcs:
            if not need():
                break
            pg.update(t_files, advance=1, description=f"file nguồn [dim]{os.path.basename(src)[:30]}[/]")
            i = fid(src)
            c = cls.get(i)
            if c and c.get("used"):
                st["bỏ qua (đã xử lý)"] += 1
                continue
            if c and not any(matches(b["match"], c["labels"]) for b in cfg["buckets"] if b["name"] in need()):
                st["bỏ qua (đã xử lý)"] += 1
                continue                       # đã phân loại, không bucket nào đang cần
            wav = os.path.join(fdir, i + ".wav")
            try:
                if not os.path.exists(wav):
                    normalize(src, wav, max_min)
                tr, speech = segment.run(wav, fdir, models, **C["segment"])
                speech_min = sum(e - s for s, e in speech) / 60
                if c is None:
                    labels, err = classify(models[cfg["classifier"]], wav, work, tr, cfg["classify"]) if tr else ({}, True)
                    labels["speakers"] = n_speakers(tr)
                    c = {"id": i, "src": src, "labels": labels, "parse_error": err, "speech_min": round(speech_min, 2)}
            except Exception as e:  # file hỏng không làm dừng cả lượt
                st["lỗi"] += 1
                pg.console.print(f"[red]lỗi[/] {src}: {e}")
                continue
            bs = [b["name"] for b in cfg["buckets"] if b["name"] in need() and matches(b["match"], c["labels"])]
            c["used"] = bool(bs and speech_min >= cfg["min_speech_minutes"])
            append(cls_p, c)
            cls[i] = c
            if not c["used"]:
                st["bỏ (không bucket cần)"] += 1
                os.remove(wav)
                continue
            rows = build_manifest.build(i, fdir, clips=clips, **C["manifest"],
                                        extra={"bucket": bs, "source_path": src, "file_labels": c["labels"]})
            with open(os.path.join(mdir, i + ".jsonl"), "w", encoding="utf8") as f:
                f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
            append(idx_p, {"id": i, "src": src, "buckets": bs, "speech_min": round(speech_min, 2),
                           "n_segments": len(rows), "labels": c["labels"]})
            before = sum(need().values())
            for b in bs:
                have[b] += speech_min
            pg.update(t_min, advance=round(before - sum(need().values()), 2))
            st["lấy"] += 1
            st["segment"] += len(rows)
            pg.console.print(f"[green]+[/] {os.path.basename(src)} → {', '.join(bs)} ({speech_min:.1f} phút, {len(rows)} segment)")

    ui.table("Kết quả lượt chạy", ["", ">số"], list(st.items()))
    bucket_table("Quota sau khi chạy")
    ui.console.print("[green]Đủ quota mọi bucket.[/]" if not need() else "[yellow]Hết nguồn, còn bucket chưa đủ quota.[/]")

if __name__ == "__main__":
    main()
