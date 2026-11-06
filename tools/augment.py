#!/usr/bin/env python3
"""Tạo segment tổng hợp cho điều kiện âm thanh còn thiếu: ồn, vang (mic xa), điện thoại, nói nhỏ.

Đầu vào là manifest có clip (trường `audio`). Mỗi segment sinh thêm một bản, đánh dấu synthetic: true,
nhãn âm thanh (file_labels) ghi đè theo phép biến đổi. Mốc thời gian giữ nguyên vì không đổi độ dài.
Chạy:
  python3 tools/augment.py --manifest manifests/abc.jsonl --op noise --noise-dir /data/musan/noise --snr 5 --out manifests/abc_noise.jsonl
  python3 tools/augment.py --manifest manifests/abc.jsonl --op phone --out manifests/abc_phone.jsonl
  python3 tools/augment.py --manifest ... --op reverb --rir-dir /data/rirs   |   --op quiet --gain-db -20
"""
import argparse, glob, json, os, random, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from ask_models import load_jsonl  # noqa: E402
import ui  # noqa: E402

LABELS = {"noise": {"noise_type": "MACHINE"}, "reverb": {"distance": "FAR"},
          "phone": {"channel": "PHONE"}, "quiet": {"distance": "FAR"}}


def rd(p):
    import soundfile as sf
    x, sr = sf.read(p, dtype="float32")
    return (x if x.ndim == 1 else x.mean(axis=1)), sr


def apply(op, src, dst, a, rnd):
    import numpy as np, soundfile as sf
    if op == "phone":  # dải 300–3400 Hz, 8 kHz, nén µ-law, rồi trả về 16 kHz
        subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", src, "-af", "highpass=f=300,lowpass=f=3400",
                        "-ar", "8000", "-acodec", "pcm_mulaw", "-f", "wav", dst + ".tmp"], check=True)
        subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", dst + ".tmp", "-ar", "16000", "-acodec", "pcm_s16le", dst], check=True)
        os.remove(dst + ".tmp")
        return {}
    x, sr = rd(src)
    if op == "quiet":
        y = x * 10 ** (a.gain_db / 20)
        info = {"gain_db": a.gain_db}
    elif op == "noise":
        p = rnd.choice(a._noises)
        n, _ = rd(p)
        n = np.resize(n, len(x))
        snr = a.snr if a.snr is not None else rnd.uniform(0, 20)
        k = np.sqrt((x ** 2).mean() / max((n ** 2).mean(), 1e-10) / 10 ** (snr / 10))
        y = x + k * n
        info = {"noise": p, "snr_db": round(snr, 1)}
    else:  # reverb
        from scipy.signal import fftconvolve
        p = rnd.choice(a._rirs)
        h, _ = rd(p)
        y = fftconvolve(x, h / (np.abs(h).max() or 1))[: len(x)]
        y *= np.abs(x).max() / max(np.abs(y).max(), 1e-9)
        info = {"rir": p}
    peak = np.abs(y).max()
    sf.write(dst, y / peak * 0.99 if peak > 1 else y, sr)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--op", required=True, choices=list(LABELS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--noise-dir")
    ap.add_argument("--rir-dir")
    ap.add_argument("--snr", type=float, help="dB; bỏ trống = ngẫu nhiên 0–20")
    ap.add_argument("--gain-db", type=float, default=-20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=os.cpu_count(), help="số segment xử lý cùng lúc")
    import conf
    argv, overrides = conf.split_argv()
    a = ap.parse_args(argv)
    a.clips = conf.path(conf.load(overrides), "clips")
    wavs = lambda d: sorted(glob.glob(os.path.join(d, "**", "*.wav"), recursive=True))
    if a.op == "noise":
        a._noises = wavs(a.noise_dir or ap.error("--op noise cần --noise-dir"))
    if a.op == "reverb":
        a._rirs = wavs(a.rir_dir or ap.error("--op reverb cần --rir-dir"))
    os.makedirs(a.clips, exist_ok=True)
    rows = [r for r in load_jsonl(a.manifest) if not r.get("synthetic") and r.get("audio")]

    def one(r):
        # seed riêng theo segment: kết quả như nhau dù chạy bao nhiêu luồng, theo thứ tự nào
        rnd = random.Random(f"{a.seed}-{r['event_id']}")
        eid = f"{r['event_id']}_{a.op}"
        dst = os.path.join(a.clips, eid + ".wav")
        info = apply(a.op, os.path.join(ROOT, r["audio"]), dst, a, rnd)
        return dict(r, event_id=eid, audio=os.path.relpath(dst, ROOT), synthetic=True,
                    augment={"op": a.op, "from": r["event_id"], **info},
                    file_labels={**(r.get("file_labels") or {}), **LABELS[a.op]})

    pg = ui.progress()
    pg.start()
    task = pg.add_task(f"augment {a.op} ×{a.workers}", total=len(rows))
    out = []
    with ThreadPoolExecutor(a.workers) as ex:
        for r in ex.map(one, rows):  # giữ thứ tự của manifest
            out.append(r)
            pg.advance(task)
    with open(a.out, "w", encoding="utf8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in out)
    pg.stop()
    ui.table("Kết quả", ["phép", ">segment tổng hợp", "manifest", "thư mục clip"], [(a.op, len(out), a.out, a.clips)])


if __name__ == "__main__":
    main()
