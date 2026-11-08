#!/usr/bin/env python3
"""Audio → transcript + turn-event: VAD (Silero) → diarization (Nemotron) → ASR (zipformer, sherpa-onnx).

Ghi ra <dir>/<id>_transcript.txt  ("bắt đầu<TAB>kết thúc<TAB>SPEAKER_xx: lời")
     và <dir>/<id>_turnevent.txt   ("bắt đầu<TAB>kết thúc<TAB>EoT|MoT|BC|INT"), đúng định dạng build_manifest.py đọc.
Chạy:
  python3 tools/segment.py data/raw/files/abc.wav --out data/raw/files
  python3 tools/segment.py --events-only data/raw/files/abc     # chỉ gán lại sự kiện từ transcript có sẵn (không cần GPU)
  Ngưỡng gán sự kiện ở config/segment (Hydra), ghi đè bằng vd. segment.bc_max=1.5

Các mô hình load theo config/models (Hydra; khoá vad, diarization, asr), ghi đè bằng vd. models=local. Phần nặng chỉ import khi cần.
"""
import argparse, functools, json, os, re, sys, threading

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BC_WORDS = {"ừ", "ừm", "ờ", "à", "ạ", "dạ", "vâng", "ừ ừ", "dạ vâng", "đúng", "đúng rồi", "ok", "ừ đúng", "thế à", "vậy à", "à ra thế", "hả", "ồ"}
# từ cuối câu cho thấy câu chưa xong: người khác nói ngay sau đó thì coi là chen ngang
OPEN_END = {"mà", "thì", "là", "và", "nhưng", "với", "của", "cho", "để", "ờ", "ừm", "kiểu", "tại", "vì", "nên", "rồi thì"}


# ---------------- mô hình ----------------

# Mỗi mô hình chỉ load một lần cho cả lượt chạy (collect.py gọi transcribe cho từng file).
@functools.lru_cache(maxsize=None)
def _vad_model(onnx):
    from silero_vad import load_silero_vad
    return load_silero_vad(onnx=onnx)


@functools.lru_cache(maxsize=None)
def _diar_model(path, device):
    from nemo.collections.asr.models import SortformerEncLabelModel
    m = SortformerEncLabelModel.restore_from(path, map_location=device)
    m.eval()
    return m


# collect.py chạy nhiều tiến trình, mỗi tiến trình một bản mô hình. Khoá chỉ có tác dụng nếu gọi từ nhiều luồng
# trong cùng tiến trình (VAD, diarization dùng chung mô hình); ASR mỗi luồng một recognizer.
_VAD_LOCK, _DIAR_LOCK, _LOCAL = threading.Lock(), threading.Lock(), threading.local()


def _asr(cfg_json):
    if getattr(_LOCAL, "asr_key", None) != cfg_json:
        _LOCAL.asr, _LOCAL.asr_key = ASR(json.loads(cfg_json)), cfg_json
    return _LOCAL.asr


def vad(wav, cfg):
    from silero_vad import read_audio, get_speech_timestamps
    x = read_audio(wav, sampling_rate=16000)
    with _VAD_LOCK:
        m = _vad_model(cfg.get("onnx", False))
        ts = get_speech_timestamps(x, m, sampling_rate=16000,
                                   min_silence_duration_ms=cfg.get("min_silence_ms", 100), return_seconds=True)
    return [(t["start"], t["end"]) for t in ts]


def diarize(wav, cfg):
    """[(start, end, 'SPEAKER_xx')]. Dùng NeMo (Sortformer). shortcut: API diarize() theo NeMo 2.x, đổi tại đây nếu checkpoint khác."""
    with _DIAR_LOCK:
        m = _diar_model(cfg["path"], cfg.get("device", "cuda"))
        out = m.diarize(audio=[wav], batch_size=1)[0]
    segs = []
    for x in out:  # mỗi phần tử: "start end speaker_N"
        s, e, spk = (x.split() if isinstance(x, str) else x)[:3]
        n = re.sub(r"\D", "", str(spk)) or "0"
        segs.append((float(s), float(e), f"SPEAKER_{int(n):02d}"))
    return sorted(segs)


class ASR:
    def __init__(self, cfg):
        import sherpa_onnx
        self.r = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=cfg["encoder"], decoder=cfg["decoder"], joiner=cfg["joiner"], tokens=cfg["tokens"],
            num_threads=1, sample_rate=16000,  # 1 luồng: song song bằng nhiều tiến trình (collect.workers) decoding_method="greedy_search",
            provider=cfg.get("provider", "cpu"))

    def __call__(self, samples):
        st = self.r.create_stream()
        st.accept_waveform(16000, samples)
        self.r.decode_stream(st)
        return st.result.text.strip().lower()


def load_wav(path):
    import soundfile as sf
    x, sr = sf.read(path, dtype="float32")
    assert sr == 16000, f"{path}: cần 16 kHz (collect.py đã chuẩn hoá)"
    return x if x.ndim == 1 else x.mean(axis=1)


def clip_to_speech(segs, speech, min_dur=0.2):
    """Cắt đoạn diarization theo VAD để biên chính xác hơn."""
    out = []
    for s, e, spk in segs:
        for a, b in speech:
            lo, hi = max(s, a), min(e, b)
            if hi - lo >= min_dur:
                out.append((lo, hi, spk))
    return sorted(out)


def transcribe(wav, models):
    speech = vad(wav, models.get("vad", {}))
    segs = clip_to_speech(diarize(wav, models["diarization"]), speech)
    asr, x = _asr(json.dumps(models["asr"], sort_keys=True)), load_wav(wav)
    tr = []
    for s, e, spk in segs:
        txt = asr(x[int(s * 16000):int(e * 16000)])
        if txt:
            tr.append({"s": round(s, 3), "e": round(e, 3), "spk": spk, "text": txt})
    return tr, speech


# ---------------- gán sự kiện (quy tắc 3.1 / 3.2) ----------------

def is_bc(seg, max_dur):
    t = re.sub(r"[^\w\s]", "", seg["text"]).strip()
    return seg["e"] - seg["s"] <= max_dur and (t in BC_WORDS or len(t.split()) <= 2)


def events(tr, bc_max, int_gap):
    """tr: các đoạn (s, e, spk, text) đã sắp theo s. Trả [(start, end, code)].
    EoT/MoT ghi tại điểm kết thúc lượt (start = end), BC tại (bắt đầu, kết thúc) tiếng đệm, INT tại lúc chen vào."""
    ev, bcs = [], set()
    for i, b in enumerate(tr):
        # 3.2: b chồng lên lượt của a
        a = next((x for x in reversed(tr[:i]) if x["spk"] != b["spk"] and x["s"] < b["s"] < x["e"]), None)
        if a is None:
            continue
        if a["e"] > b["e"] and is_bc(b, bc_max):
            ev.append((b["s"], b["e"], "BC"))
            bcs.add(i)
        elif a["e"] <= b["e"]:
            ev.append((b["s"], a["e"], "INT"))
    for i, a in enumerate(tr):
        if i in bcs:
            continue
        nxt = next((x for j, x in enumerate(tr[i + 1:], i + 1) if j not in bcs and x["s"] >= a["e"] - 0.02), None)
        if nxt is None:
            continue
        gap = nxt["s"] - a["e"]
        if nxt["spk"] == a["spk"]:
            ev.append((a["e"], a["e"], "MoT"))                    # 3.1: cùng người nói tiếp
        elif gap < int_gap and a["text"].split()[-1:] and a["text"].split()[-1] in OPEN_END:
            ev.append((nxt["s"], nxt["s"], "INT"))                # chen vào khi câu còn bỏ ngỏ, không chồng lời
        else:
            ev.append((a["e"], a["e"], "EoT"))
    return sorted(set(ev))


def write(out, name, tr, ev):
    with open(os.path.join(out, name + "_transcript.txt"), "w", encoding="utf8") as f:
        for t in tr:
            f.write(f"{t['s']:.3f}\t{t['e']:.3f}\t{t['spk']}: {t['text']}\n")
    with open(os.path.join(out, name + "_turnevent.txt"), "w", encoding="utf8") as f:
        for s, e, c in ev:
            f.write(f"{s:.3f}\t{e:.3f}\t{c}\n")


def run(wav, out, models, bc_max, int_gap):
    """Dùng từ collect.py. Trả (transcript, speech) để tính số phút có tiếng nói."""
    name = os.path.splitext(os.path.basename(wav))[0]
    tr, speech = transcribe(wav, models)
    write(out, name, tr, events(tr, bc_max, int_gap))
    return tr, speech


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="file wav 16 kHz, hoặc <dir>/<id> khi --events-only")
    ap.add_argument("--out")
    ap.add_argument("--events-only", action="store_true")
    import conf
    argv, overrides = conf.split_argv()
    a = ap.parse_args(argv)
    C = conf.load(overrides)
    if a.events_only:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from build_manifest import load_tr
        d, name = os.path.split(a.input)
        tr = sorted(load_tr(a.input + "_transcript.txt"), key=lambda t: t["s"])
        ev = events(tr, **C["segment"])
        write(a.out or d, name, tr, ev)
    else:
        tr, _ = run(a.input, a.out or os.path.dirname(a.input), C["models"], **C["segment"])
        ev = events(tr, **C["segment"])
    from collections import Counter
    import ui
    cnt = Counter(c for _, _, c in ev)
    ui.table("Sự kiện gán được", ["loại", ">số", ">%"], [(k, v, f"{100 * v / len(ev):.1f}") for k, v in cnt.most_common()])


if __name__ == "__main__":
    main()
