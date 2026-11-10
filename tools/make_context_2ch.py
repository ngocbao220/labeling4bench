#!/usr/bin/env python3
"""Mẫu có ngữ cảnh từ cặp file 2 kênh (AGENT/CTM đã tách sẵn): VAD RIÊNG TỪNG KÊNH, mốc = điểm kết thúc âm thanh của kênh người dùng.
Dùng: python3 tools/make_context_2ch.py --root <thư mục pairs16k> --pick pair_002:CTM:28.3 pair_004:AGENT:49.5 ... [--first 16]
(pair:KÊNH_NGƯỜI_DÙNG:giây_kết_thúc_lượt). Mỗi mẫu CHỈ gồm kênh của người dùng. Ghi nối vào data/context.json + audio/ctx_N.mp3. Chưa có chữ (cần ASR) nên segs rỗng."""
import argparse, glob, json, subprocess, sys, numpy as np, librosa
sys.path.insert(0, "/tmp")
ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True); ap.add_argument("--pick", nargs="+", required=True)
ap.add_argument("--first", type=int, default=16); ap.add_argument("--out", default="data/context.json")
ap.add_argument("--pre", type=float, default=12); ap.add_argument("--post", type=float, default=3)
a = ap.parse_args()
FPS = 50
def vad(y):
    """VAD năng lượng cho MỘT kênh. Ngưỡng bám theo mức tiếng nói của chính kênh (97% - 30 dB), bỏ qua nhiễu nền và tiếng lọt từ kênh bên kia."""
    rms = librosa.amplitude_to_db(librosa.feature.rms(y=y, frame_length=640, hop_length=320)[0] + 1e-6)
    hi = np.percentile(rms, 97); on, off = hi - 30, hi - 36
    sp = np.zeros(len(rms), bool); st = False
    for i, v in enumerate(rms):
        st = (v > off) if st else (v > on)
        sp[i] = st
    segs = []; i = 0
    while i < len(sp):
        if sp[i]:
            j = i
            while j < len(sp) and sp[j]: j += 1
            segs.append([i / FPS, j / FPS]); i = j
        else: i += 1
    m = []
    for s_ in segs:
        if m and s_[0] - m[-1][1] < 0.3: m[-1][1] = s_[1]
        else: m.append(s_)
    return [s_ for s_ in m if s_[1] - s_[0] >= 0.25], rms, on
data = json.load(open(a.out)) if glob.glob(a.out) else []
data = [d for d in data if d["id"] < a.first]
for k, pk in enumerate(a.pick):
    pair, ch, end = pk.split(":"); end = float(end); n = a.first + k
    f = glob.glob(f"{a.root}/{pair}/*{ch}.wav")[0]
    y_all = librosa.load(f, sr=16000)[0]          # CHỈ kênh của người dùng
    segs, rms, thr = vad(y_all)
    near = min((e for s_, e in segs), key=lambda e: abs(e - end)); end = near   # bám vào điểm kết thúc VAD gần nhất
    t0 = max(0, end - a.pre); t1 = min(len(y_all) / 16000, end + a.post)
    y = y_all[int(t0 * 16000):int(t1 * 16000)]
    def contrast(e):  # mức trước mốc trừ mức lớn nhất ngay sau mốc (dB); nhỏ = mốc nằm giữa tiếng
        i = int(e * FPS); return float(rms[max(0, i - 15):i].mean() - (rms[i + 1:i + 13].max() if i + 13 < len(rms) else -99))
    ends = [round(e - t0, 2) for s_, e in segs if t0 < e <= t1 and e - t0 > 2.0 and contrast(e) >= 20]
    pkk = (np.abs(y[: len(y) // 150 * 150]).reshape(150, -1).max(1) / max(np.abs(y).max(), 1e-6)).round(3).tolist()
    import soundfile as sf_; sf = f"/tmp/ch_{n}.wav"; sf_.write(sf, y, 16000)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", sf, "-ac", "1", "-b:a", "48k", f"audio/ctx_{n}.mp3"], check=True)
    cut = round(end - t0, 2)
    # kiểm tra: trước mốc 0,3 s phải có tiếng, sau mốc 0,25 s phải yên lặng
    chk = []
    for e in ends:
        i = int((e + t0) * FPS); chk.append((round(float(rms[max(0, i - 15):i].mean()), 0), round(float(rms[i + 1:i + 13].max()) if i + 13 < len(rms) else -99, 0)))
    data.append({"id": n, "src": f"audio/ctx_{n}.mp3", "dur": round(t1 - t0, 2), "cut": cut, "user": "S1" if ch == "CTM" else "S0",
                 "peaks": pkk, "vad": [e for e in ends if abs(e - cut) > 0.3], "segs": [],
                 "label": "Tổng đài: " + ("khách" if ch == "CTM" else "tổng đài viên") + " (1 kênh)", "source": f"cskh {pair} {ch}",
                 "auto": {"event": None, "event_id": f"{pair}_{ch}_{end}"}})
    print(n, pair, ch, "thr", round(thr, 1), "mốc:", ends, "cut", cut, "(dB trước, dB tối đa sau):", chk)
json.dump(data, open(a.out, "w"), ensure_ascii=False, separators=(",", ":"))
