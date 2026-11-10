#!/usr/bin/env python3
"""Mẫu có ngữ cảnh từ cặp file 2 kênh (AGENT/CTM đã tách sẵn): VAD RIÊNG TỪNG KÊNH, mốc = điểm kết thúc âm thanh của kênh người dùng.
Dùng: python3 tools/make_context_2ch.py --root <thư mục pairs16k> --pick pair_002:CTM:28.3 pair_004:AGENT:49.5 ... [--first 16]
(pair:KÊNH_NGƯỜI_DÙNG:giây_kết_thúc_lượt). Ghi nối vào data/context.json + audio/ctx_N.mp3. Chưa có chữ (cần ASR) nên segs rỗng."""
import argparse, glob, json, subprocess, sys, numpy as np, librosa
sys.path.insert(0, "/tmp")
ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True); ap.add_argument("--pick", nargs="+", required=True)
ap.add_argument("--first", type=int, default=16); ap.add_argument("--out", default="data/context.json")
ap.add_argument("--pre", type=float, default=12); ap.add_argument("--post", type=float, default=3)
a = ap.parse_args()
FPS = 50
def vad(y):
    rms = librosa.amplitude_to_db(librosa.feature.rms(y=y, frame_length=640, hop_length=320)[0] + 1e-6)
    lo, hi = np.percentile(rms, 10), np.percentile(rms, 97)
    sp = rms > lo + 0.4 * (hi - lo); segs = []; i = 0
    while i < len(sp):
        if sp[i]:
            j = i
            while j < len(sp) and sp[j]: j += 1
            segs.append([i / FPS, j / FPS]); i = j
        else: i += 1
    m = []
    for s in segs:
        if m and s[0] - m[-1][1] < 0.3: m[-1][1] = s[1]
        else: m.append(s)
    return [s for s in m if s[1] - s[0] >= 0.25]
data = json.load(open(a.out)) if glob.glob(a.out) else []
data = [d for d in data if d["id"] < a.first]
for k, pk in enumerate(a.pick):
    pair, ch, end = pk.split(":"); end = float(end); n = a.first + k
    fs = {c: glob.glob(f"{a.root}/{pair}/*{c}.wav")[0] for c in ("AGENT", "CTM")}
    ys = {c: librosa.load(f, sr=16000)[0] for c, f in fs.items()}
    L = min(len(v) for v in ys.values()); mix = (ys["AGENT"][:L] + ys["CTM"][:L]) / 2
    t0 = max(0, end - a.pre); t1 = min(L / 16000, end + a.post)
    segs = vad(ys[ch][:L])
    ends = [round(e - t0, 2) for s, e in segs if t0 < e <= t1 and e - t0 > 0.5]
    y = mix[int(t0 * 16000):int(t1 * 16000)]
    pkk = np.abs(y[: len(y) // 150 * 150]).reshape(150, -1).max(1); pkk = (pkk / pkk.max()).round(3).tolist()
    sf = f"/tmp/mix_{n}.wav"
    import soundfile as sf_; sf_.write(sf, y, 16000)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", sf, "-ac", "1", "-b:a", "48k", f"audio/ctx_{n}.mp3"], check=True)
    cut = round(end - t0, 2)
    other = [[round(s - t0, 2), round(e - t0, 2)] for s, e in vad(ys["AGENT" if ch == "CTM" else "CTM"][:L]) if e > t0 and s < t1]
    data.append({"id": n, "src": f"audio/ctx_{n}.mp3", "dur": round(t1 - t0, 2), "cut": cut, "user": "S1" if ch == "CTM" else "S0",
                 "peaks": pkk, "vad": [e for e in ends if abs(e - cut) > 0.3], "segs": [], "other_vad": other,
                 "label": "Tổng đài: " + ("khách" if ch == "CTM" else "tổng đài viên"), "source": f"cskh {pair} {ch}",
                 "auto": {"event": None, "event_id": f"{pair}_{ch}_{end}"}})
    print(n, pair, ch, "mốc:", ends, "cut", cut)
json.dump(data, open(a.out, "w"), ensure_ascii=False, separators=(",", ":"))
