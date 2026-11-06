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
import glob, hashlib, json, os, random, subprocess, sys, time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import psutil

import hydra
from hydra.core.hydra_config import HydraConfig
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
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", src, "-t", str(max_min * 60),
                    "-vn", "-ac", "1", "-ar", "16000", dst], check=True)


def excerpts(wav, work, tr, n, dur):
    stem = os.path.splitext(os.path.basename(wav))[0]  # tên riêng theo file để các luồng không ghi đè nhau
    """n đoạn ~dur giây rải đều trong file, bắt đầu tại đầu một lượt nói."""
    if not tr:
        return []
    picks = [tr[int(i * len(tr) / n)]["s"] for i in range(n)]
    out = []
    for i, s in enumerate(picks):
        p = os.path.join(work, f"{stem}_ex{i}.wav")
        subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-ss", str(s), "-t", str(dur), "-i", wav, p], check=True)
        out.append(p)
    return out


def classify(model, wav, work, tr, k):
    qs = ask_models.questions("turntaking", ask_models.FILE_QS)
    text = " / ".join(f"{t['spk']}: {t['text']}" for t in tr)[:k["transcript_chars"]]
    ctx = ("Bạn nghe vài đoạn trích từ cùng một file audio. Trả lời các câu hỏi cho cả file (không phải một lượt nói). "
           "Câu về giọng hoặc tuổi thì xét người nói chính.\nTranscript rút gọn (máy chép, có thể sai): " + text)
    ans, err, _, raw = ask_models.ask(model, excerpts(wav, work, tr, k["n_excerpts"], k["excerpt_sec"]), ctx, qs)
    return {k.split(".", 1)[1]: v for k, v in ans.items()}, err, raw


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
    ui.table("Kết quả sẽ ghi vào", ["", "đường dẫn"], [
        ("quyết định LẤY / BỎ từng file", cls_p), ("file đã lấy (bucket, phút)", idx_p),
        ("wav, transcript, turnevent", fdir), ("segment (manifest)", mdir), ("clip", clips),
        ("log + config của lượt này", HydraConfig.get().runtime.output_dir)])
    bucket_table("Quota trước khi chạy")
    if C["dry_run"]:
        return
    m = models[cfg["classifier"]]
    try:  # vLLM chưa chạy thì dừng ngay, khỏi chạy ASR cho từng file rồi mới lỗi
        import urllib.request
        urllib.request.urlopen(m["base_url"].rstrip("/") + "/models", timeout=10).read()
    except Exception as e:
        ui.console.print(f"[red]Không gọi được {cfg['classifier']} ở {m['base_url']}: {e}[/]\n"
                         f"Kiểm tra vllm serve đã chạy và base_url trong config models đúng host:cổng.")
        sys.exit(1)
    max_min = cfg["max_minutes_per_file"]
    total_need = sum(need().values())
    survey = cfg["survey"]
    active = (lambda: True) if survey else (lambda: bool(need()))
    st = {"lấy": 0, "bỏ": 0, "bỏ qua (đã quyết định ở lượt trước)": 0, "lỗi": 0, "segment": 0}

    def decide(c, speech_min):
        """Luồng chính: LẤY hay BỎ file, kèm lý do đọc được. Trả (quyết định, lý do, bucket)."""
        if failed(c):
            return "BỎ", "mô hình không trả lời được (lỗi parse; xem: decisions.py --raw)", []
        if speech_min < cfg["min_speech_minutes"]:
            return "BỎ", f"ít tiếng nói ({speech_min:.1f} < {cfg['min_speech_minutes']} phút)", []
        fit = [b["name"] for b in cfg["buckets"] if matches(b["match"], c["labels"])]
        if survey:  # chỉ phân loại để thống kê nguồn, không lấy file nào
            return "KHẢO SÁT", (f"khớp {', '.join(fit)}" if fit else "không khớp bucket nào"), []
        if not fit:
            return "BỎ", "không khớp bucket nào", []
        bs = [b for b in fit if b in need()]
        if not bs:
            return "BỎ", f"khớp {', '.join(fit)} nhưng đã đủ quota", []
        return "LẤY", f"vào {', '.join(bs)} ({speech_min:.1f} phút)", bs

    def process(src, i, c):
        """Chạy trong luồng phụ: chuẩn hoá, VAD + diarization + ASR, phân loại (nếu chưa có). Không đụng trạng thái chung."""
        wav = os.path.join(fdir, i + ".wav")
        if not os.path.exists(wav):
            normalize(src, wav, max_min)
        tr, speech = segment.run(wav, fdir, models, **C["segment"])
        speech_min = sum(e - s for s, e in speech) / 60
        if c is None:
            labels, err, raw = classify(m, wav, work, tr, cfg["classify"]) if tr else ({}, True, "(không có transcript)")
            labels["speakers"] = n_speakers(tr)
            c = {"id": i, "src": src, "labels": labels, "parse_error": err, "speech_min": round(speech_min, 2)}
            if raw:
                c["raw"] = raw  # câu trả lời thô của mô hình khi lỗi parse
        for f in glob.glob(os.path.join(work, i + "_ex*.wav")):
            os.remove(f)
        return wav, speech_min, c

    def failed(c):
        return c.get("parse_error") and not c["labels"].get("genre")

    def wanted(c):
        """Luồng chính: file này còn cần xử lý không (theo cache phân loại và quota hiện tại)."""
        if c and c.get("used"):
            return False
        if survey:
            return not c  # khảo sát: chỉ cần phân loại file chưa có nhãn
        return not c or any(matches(b["match"], c["labels"]) for b in cfg["buckets"] if b["name"] in need())

    peak = [0]
    workers, cpu_max, cores = cfg["workers"], cfg["max_cpu_percent"], psutil.cpu_count() or 1
    # mỗi worker chạy zipformer với asr.threads luồng: tổng vượt số nhân thì CPU quá tải ngay từ đầu
    thr = models["asr"].get("threads", 1)
    if models["asr"].get("provider") == "cpu" and workers * thr > cores:
        models["asr"]["threads"] = max(1, cores // workers)
        ui.console.print(f"[yellow]{workers} worker × {thr} luồng ASR > {cores} nhân CPU: giảm asr.threads xuống {models['asr']['threads']}[/]")
    psutil.cpu_percent(None)  # mốc đầu cho lần đo sau
    with ui.progress() as pg, ThreadPoolExecutor(workers) as ex:
        t_files = pg.add_task("file nguồn", total=len(srcs))
        t_min = pg.add_task("phút còn thiếu", total=round(total_need, 1))
        it, running, seen, exhausted = iter(srcs), {}, set(), [False]

        per_file = max(1, models["asr"].get("threads", 1)) if models["asr"].get("provider") == "cpu" else 1

        def submit(cpu):
            """Nạp thêm file theo phần CPU còn trống: mỗi file ước tính chiếm `per_file` nhân (asr.threads khi ASR chạy CPU).
            Không vượt `workers`, dừng khi đủ quota; luôn giữ ít nhất 1 file chạy."""
            # tải ước tính (số nhân): lấy số lớn hơn giữa CPU đo được và phần các file đang chạy sẽ dùng,
            # vì file vừa nạp chưa kịp đẩy CPU lên
            load = max(cpu / 100 * cores, len(running) * per_file)
            room = int((cpu_max / 100 * cores - load) / per_file)
            n = max(room, 0 if running else 1)
            while n > 0 and len(running) < workers and active():
                src = next(it, None)
                if src is None:
                    exhausted[0] = True
                    return
                i = fid(src)
                if cls.get(i) and failed(cls[i]):
                    cls.pop(i)  # lượt trước mô hình không trả lời được: phân loại lại
                if i in seen or not wanted(cls.get(i)):
                    seen.add(i)
                    st["bỏ qua (đã quyết định ở lượt trước)"] += 1
                    pg.advance(t_files)
                    continue
                seen.add(i)
                running[ex.submit(process, src, i, cls.get(i))] = src
                n -= 1

        def handle(fut):
            src = running.pop(fut)
            pg.advance(t_files)
            try:
                wav, speech_min, c = fut.result()
            except Exception as e:  # file hỏng không làm dừng cả lượt
                st["lỗi"] += 1
                pg.console.print(f"[red]lỗi[/] {src}: {e}")
                return
            i = c["id"]
            c["decision"], c["reason"], bs = decide(c, speech_min)
            c["used"] = c["decision"] == "LẤY"
            if c["decision"] == "KHẢO SÁT":
                st["khảo sát"] = st.get("khảo sát", 0) + 1
                pg.console.print(f"[cyan]· KHẢO SÁT[/] {os.path.basename(src)}: {c['reason']} [dim]({' '.join(f'{k}={v}' for k, v in c['labels'].items())})[/]")
                append(cls_p, dict(c, decided_at=time.strftime("%Y-%m-%d %H:%M:%S")))
                cls[i] = c
                os.remove(wav)
                return
            c["decided_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            append(cls_p, c)
            cls[i] = c
            lab = " ".join(f"{k}={v}" for k, v in c["labels"].items())
            if not c["used"]:
                st["bỏ"] += 1
                pg.console.print(f"[yellow]- BỎ[/] {os.path.basename(src)}: {c['reason']} [dim]({lab})[/]")
                os.remove(wav)
                return
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
            pg.console.print(f"[green]+ LẤY[/] {os.path.basename(src)}: {c['reason']}, {len(rows)} segment [dim]({lab})[/]")

        while True:
            done = wait(running, timeout=cfg["cpu_check_sec"], return_when=FIRST_COMPLETED)[0] if running else set()
            for fut in done:
                handle(fut)
            cpu = psutil.cpu_percent(None)
            peak[0] = max(peak[0], len(running))
            submit(cpu)
            pg.update(t_files, description=f"file nguồn [dim]{len(running)}/{workers} luồng · CPU {cpu:.0f}%[/]")
            if not running and (exhausted[0] or not active()):
                break

    st["số file chạy cùng lúc tối đa"] = peak[0]
    ui.table("Kết quả lượt chạy", ["", ">số"], list(st.items()))
    bucket_table("Quota sau khi chạy")
    if survey:
        ui.console.print("[green]Khảo sát xong.[/] Xem phân bố: python3 tools/decisions.py --dist, hoặc notebooks/collect_results.ipynb")
    else:
        ui.console.print("[green]Đủ quota mọi bucket.[/]" if not need() else "[yellow]Hết nguồn, còn bucket chưa đủ quota.[/]")

if __name__ == "__main__":
    main()
