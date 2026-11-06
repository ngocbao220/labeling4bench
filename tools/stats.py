#!/usr/bin/env python3
"""Thống kê độ phủ của bộ segment và liệt kê ô còn thiếu so với quota.

Nhãn của mỗi segment, theo thứ tự ưu tiên: đa số người gán (--humans) > mô hình đầu tiên (--answers)
> nhãn mức file (file_labels, từ collect.py) > event_auto.
Chạy:
  python3 tools/stats.py --manifest 'manifests/*.jsonl' --answers out/qwen_instruct.jsonl out/qwen_thinking.jsonl \
      [--humans humans.jsonl] [--csv out/stats] [collect=local]
Quota (config/quotas) và bucket (config/collect) đọc bằng Hydra; ghi đè như các tool khác.
"""
import argparse, collections, csv, glob, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ask_models import load_jsonl  # noqa: E402
from audit_questions import majority  # noqa: E402
import ui  # noqa: E402
from rich.console import Console  # noqa: E402

FIELDS = {"event": "tcontent.event", "genre": "tstyle.genre", "accent": "tstyle.accent", "age": "tstyle.age",
          "channel": "tenv.channel", "noise_type": "tenv.noise_type", "distance": "tenv.distance",
          "audibility": "tenv.audibility"}


def condition(l):
    if l.get("channel") == "PHONE":
        return "phone"
    if l.get("distance") == "FAR":
        return "far"
    if l.get("noise_type") not in (None, "QUIET"):
        return "noisy"
    return "clean" if l.get("noise_type") == "QUIET" else "unknown"


OUT = Console()  # báo cáo ra stdout (có thể chuyển hướng vào file)


def table(title, rows, cols=None):
    if cols is None:
        tot = sum(rows.values()) or 1
        ui.table(title, ["giá trị", ">số", ">%"], [(k, v, f"{100 * v / tot:.1f}") for k, v in sorted(rows.items(), key=lambda x: -x[1])], OUT)
    else:
        ui.table(title, [""] + [">" + c for c in cols], [[r] + [rows[r].get(c, 0) for c in cols] for r in sorted(rows)], OUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", help="glob; mặc định <paths.manifests>/*.jsonl")
    ap.add_argument("--answers", nargs="*", default=[])
    ap.add_argument("--humans")
    ap.add_argument("--csv", help="thư mục ghi bảng chéo dạng CSV")
    import conf
    argv, overrides = conf.split_argv()
    a = ap.parse_args(argv)
    C = conf.load(overrides)
    a.manifest = a.manifest or os.path.join(conf.path(C, "manifests"), "*.jsonl")

    segs = [r for p in sorted(glob.glob(a.manifest)) for r in load_jsonl(p)]
    models = [{r["sample_id"]: r for r in load_jsonl(p)} for p in a.answers]
    hum = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in load_jsonl(a.humans) if a.humans else []:
        for k, v in r["answers"].items():
            if v != "UNK":
                hum[r["sample_id"]][k].append(v)

    n_hum = len(hum)

    def label(s, name):
        key = FIELDS[name]
        h = majority(hum.get(s["event_id"], {}).get(key, []))
        if h:
            return h
        for m in models[:1]:
            v = m.get(s["event_id"], {}).get("answers", {}).get(key)
            if v and v != "UNK":
                return v
        v = (s.get("file_labels") or {}).get(name)
        return v or (s.get("event_auto") if name == "event" else None)

    L = [{n: label(s, n) for n in FIELDS} | {"bucket": s.get("bucket") or ["(không bucket)"],
                                             "synthetic": bool(s.get("synthetic"))} for s in segs]
    for l in L:
        l["condition"] = condition(l)

    OUT.rule("[bold]Thống kê độ phủ")
    OUT.print(f"{len(segs)} segment từ {len({s['conversation_id'] for s in segs})} file; "
              f"{sum(l['synthetic'] for l in L)} segment tổng hợp. Nhãn người cho {n_hum} segment.")

    cfg = C["collect"]
    idx = os.path.join(conf.path(C, "raw"), "raw_index.jsonl")
    if os.path.exists(idx):
        have = collections.Counter()
        for r in load_jsonl(idx) if os.path.exists(idx) else []:
            for b in r["buckets"]:
                have[b] += r["speech_min"]
        ui.table("Phút có tiếng nói theo bucket", ["bucket", "mô tả", ">có", ">mục tiêu", "đạt"],
                 [(b["name"], b["desc"], f"{have[b['name']]:.1f}", b["target_minutes"],
                   "✓" if have[b["name"]] >= b["target_minutes"] else "✗") for b in cfg["buckets"]], OUT)

    for n in list(FIELDS) + ["condition"]:
        table(f"Phân bố `{n}`", collections.Counter(str(l[n]) for l in L))

    evs = sorted({str(l["event"]) for l in L})
    cross = {}
    for dim in ("bucket", "genre", "condition", "accent"):
        t = collections.defaultdict(collections.Counter)
        for l in L:
            for v in (l[dim] if isinstance(l[dim], list) else [l[dim]]):
                t[str(v)][str(l["event"])] += 1
        cross[dim] = t
        table(f"`event` × `{dim}`", t, evs)

    if len(models) >= 2:
        agr = []
        keys = sorted({k for r in models[0].values() for k in r["answers"]})
        for k in keys:
            pairs = [(r["answers"].get(k), models[1][sid]["answers"].get(k)) for sid, r in models[0].items() if sid in models[1]]
            pairs = [p for p in pairs if None not in p]
            if pairs:
                agr.append((k, len(pairs), round(100 * sum(x == y for x, y in pairs) / len(pairs))))
        ui.table("Đồng thuận giữa hai mô hình đầu theo câu hỏi", ["câu", ">n", ">khớp %"], sorted(agr, key=lambda x: x[2]), OUT)

    q = C["quotas"]
    miss = []
    ev_cnt = collections.Counter(str(l["event"]) for l in L)
    miss += [f"event={e}: {ev_cnt[e]}/{q['event']}" for e in ("EOT", "MOT", "BC", "INT") if ev_cnt[e] < q.get("event", 0)]
    for dim, qk in (("bucket", "event x bucket"), ("condition", "event x condition"), ("accent", "event x accent")):
        for v, row in sorted(cross[dim].items()):
            if v in ("None", "unknown", "UNK", "(không bucket)"):
                continue
            miss += [f"event={e} × {dim}={v}: {row.get(e, 0)}/{q[qk]}" for e in ("EOT", "MOT", "BC", "INT") if row.get(e, 0) < q.get(qk, 0)]
    if miss:
        ui.table("Ô chưa đạt quota", ["ô", ">có", ">cần"], [m.rsplit(": ", 1)[0:1] + m.rsplit(": ", 1)[1].split("/") for m in miss], OUT)
    else:
        OUT.print("[green]Tất cả ô đã đạt quota.[/]")

    if a.csv:
        os.makedirs(a.csv, exist_ok=True)
        for dim, t in cross.items():
            with open(os.path.join(a.csv, f"event_x_{dim}.csv"), "w", newline="", encoding="utf8") as f:
                w = csv.writer(f)
                w.writerow([dim] + evs)
                w.writerows([[r] + [t[r].get(e, 0) for e in evs] for r in sorted(t)])


if __name__ == "__main__":
    main()
