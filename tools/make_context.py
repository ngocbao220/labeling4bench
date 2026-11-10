#!/usr/bin/env python3
"""Dựng mẫu 'có ngữ cảnh' cho UI gán nhãn turn-taking: cắt clip ≥12 giây trước điểm cắt,
tính đường cao độ (F0) và năng lượng khách quan, lấy lời các lượt trước. Ghi data/context.json + audio/ctx_N.mp3.
Mốc trong manifest là giây TƯƠNG ĐỐI so với clip. Dùng: python3 tools/make_context.py --dir <thư mục có 10154_2.wav,_transcript.txt> --ids 035 037 038 042 045"""
import argparse, json, subprocess, numpy as np, librosa, re, os
ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True); ap.add_argument("--conv", default="10154_2")
ap.add_argument("--manifest", default="manifests/10154_2.jsonl"); ap.add_argument("--ids", nargs="+", required=True)
ap.add_argument("--first", type=int, default=11); ap.add_argument("--out", default="data/context.json")
a = ap.parse_args()
y_all, sr = librosa.load(f"{a.dir}/{a.conv}.wav", sr=16000)
tr = []
for l in open(f"{a.dir}/{a.conv}_transcript.txt", encoding="utf-8"):
    p = l.rstrip("\n").split("\t")
    if len(p) < 3: continue
    m = re.match(r"SPEAKER_+(\d+):\s*(.*)", p[2])
    if m: tr.append((float(p[0]), float(p[1]), "S" + str(int(m.group(1))), m.group(2).strip()))
man = {json.loads(l)["event_id"]: json.loads(l) for l in open(a.manifest)}
FPS = 20
out = []
for k, eid in enumerate(a.ids):
    r = man[f"{a.conv}_{eid}"]; t0, t1 = r["clip"]; n = a.first + k
    t = r["times"]; ev = r["event_auto"]
    cut = t.get("t_pause_start") or t.get("t_end") or t.get("t_bc_end") or t.get("t_int_start") + 1.5
    if ev == "INT": cut = t["t_int_start"] + 1.5
    y = y_all[int(t0 * sr):int(t1 * sr)]
    f0, vo, _ = librosa.pyin(y, fmin=70, fmax=400, sr=sr, frame_length=1024, hop_length=sr // FPS)
    rms = librosa.amplitude_to_db(librosa.feature.rms(y=y, frame_length=640, hop_length=sr // FPS)[0] + 1e-6)
    user = "S" + str(int(r["user"].split("_")[-1]))
    uf = f0[(~np.isnan(f0)) & (np.arange(len(f0)) / FPS < cut)]
    med = float(np.median(uf)) if len(uf) else 150.0
    semi = [None if np.isnan(v) else round(12 * np.log2(v / med), 1) for v in f0]
    pk = np.abs(y[: len(y) // 150 * 150]).reshape(150, -1).max(1); pk = (pk / pk.max()).round(3).tolist()
    segs = []
    for s, e, spk, tx in tr:
        if e <= t0 or s >= t1: continue
        S, E = max(s, t0) - t0, min(e, t1) - t0
        hide = bool((S < cut and E > cut - 0.35) or (ev == "INT" and spk == user and cut - 1.6 <= S < cut))  # lượt đang diễn ra tại điểm cắt: che đến bước 2
        segs.append({"spk": spk, "s": round(S, 2), "e": round(E, 2), "text": ("… " if s < t0 else "") + tx, "hide": bool(hide), "after": S >= cut})
    last = [g for g in segs if g["spk"] == user and g["s"] < cut]
    if last: last[-1]["hide"] = True  # lời của chính người dùng ở điểm cắt luôn bị che
    open_ = f"audio/ctx_{n}.mp3"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(t0), "-t", str(t1 - t0), "-i", f"{a.dir}/{a.conv}.wav", "-ac", "1", "-b:a", "48k", open_], check=True)
    out.append({"id": n, "src": open_, "dur": round(t1 - t0, 2), "cut": round(cut, 2), "user": user, "peaks": pk,
                "f0": semi, "rms": [round(float(v), 1) for v in rms], "fps": FPS, "segs": segs,
                "auto": {"event": ev, "event_id": r["event_id"]}})
json.dump(out, open(a.out, "w"), ensure_ascii=False, separators=(",", ":"))
print([(o["id"], o["auto"]["event"], o["cut"], o["user"], len(o["segs"])) for o in out])
