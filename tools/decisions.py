#!/usr/bin/env python3
"""Xem quyết định LẤY / BỎ / KHẢO SÁT của collect.py cho từng file nguồn, kèm lý do và nhãn mô hình gán.

Đọc <paths.raw>/classified.jsonl (mỗi file lấy dòng mới nhất).
Chạy:
  python3 tools/decisions.py                      # tổng hợp + toàn bộ danh sách
  python3 tools/decisions.py --only BỎ --limit 50
  python3 tools/decisions.py --grep interview     # lọc theo đường dẫn, lý do hoặc nhãn
  python3 tools/decisions.py --csv out/decisions.csv collect=local
  python3 tools/decisions.py --dist               # phân bố mọi file đã phân loại theo nhãn (phút), không quan tâm quota
"""
import argparse, collections, csv, os, re

import conf
import ui
from ask_models import load_jsonl

COLS = ["genre", "channel", "noise_type", "distance", "accent", "speakers"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["LẤY", "BỎ", "KHẢO SÁT"])
    ap.add_argument("--grep")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--csv")
    ap.add_argument("--raw", action="store_true", help="in câu trả lời thô của mô hình cho các file lỗi parse")
    ap.add_argument("--dist", action="store_true", help="phân bố của mọi file đã phân loại (số file, số phút), không quan tâm quota")
    argv, overrides = conf.split_argv()
    a = ap.parse_args(argv)
    C = conf.load(overrides)
    p = os.path.join(conf.path(C, "raw"), "classified.jsonl")
    if not os.path.exists(p):
        ui.console.print(f"[red]Chưa có {p}[/]: chạy tools/collect.py trước.")
        return
    rows = list({r["id"]: r for r in load_jsonl(p)}.values())
    for r in rows:  # bản ghi cũ (trước khi có trường decision)
        r.setdefault("decision", "LẤY" if r.get("used") else "BỎ")
        r.setdefault("reason", "")

    if a.dist:
        dist(rows, C["collect"]["sources"])
        return
    by = collections.Counter((r["decision"], re.sub(r"\(.*?\)|[\d.]+", "", r["reason"]).strip()) for r in rows)
    ui.table(f"Quyết định cho {len(rows)} file", ["quyết định", "lý do", ">số file"],
             [(d, why, n) for (d, why), n in by.most_common()])

    if a.only:
        rows = [r for r in rows if r["decision"] == a.only]
    if a.grep:
        rows = [r for r in rows if a.grep.lower() in (r["src"] + r["reason"] + str(r["labels"])).lower()]
    rows.sort(key=lambda r: r.get("decided_at", ""), reverse=True)
    shown = rows[: a.limit]
    ui.table(f"Chi tiết ({len(shown)}/{len(rows)} file, mới nhất trước)",
             ["file", "quyết định", "lý do", "nhãn mô hình"],
             [(os.path.basename(r["src"]), {"LẤY": "[green]LẤY[/]", "BỎ": "[yellow]BỎ[/]"}.get(r["decision"], f"[cyan]{r['decision']}[/]"), r["reason"],
               " ".join(f"{k}={r['labels'][k]}" for k in COLS if r["labels"].get(k) is not None)) for r in shown])
    if a.raw:
        for r in shown:
            if r.get("raw"):
                ui.console.rule(os.path.basename(r["src"]))
                ui.console.print(r["raw"], markup=False, highlight=False)
        if not any(r.get("raw") for r in shown):
            ui.console.print("Không có file nào lưu câu trả lời thô (chỉ lưu khi lỗi parse, từ bản này trở đi).")
    if a.csv:
        os.makedirs(os.path.dirname(os.path.abspath(a.csv)), exist_ok=True)
        with open(a.csv, "w", newline="", encoding="utf8") as f:
            w = csv.writer(f)
            w.writerow(["src", "decision", "reason", "speech_min", "parse_error", "decided_at"] + COLS)
            w.writerows([[r["src"], r["decision"], r["reason"], r.get("speech_min"), r.get("parse_error"), r.get("decided_at")]
                         + [r["labels"].get(k) for k in COLS] for r in rows])
        ui.console.print(f"Đã ghi {a.csv}")


def source_group(src, roots):
    """Thư mục con đầu tiên dưới thư mục nguồn chứa file (vd. /data/youtube/podcast/x.wav → youtube/podcast)."""
    for r in sorted(roots, key=len, reverse=True):
        r = r.rstrip("/") + "/"
        if src.startswith(r):
            rest = src[len(r):].split("/")
            return os.path.basename(r.rstrip("/")) + ("/" + rest[0] if len(rest) > 1 else "")
    return os.path.dirname(src)


def dist(rows, roots):
    """Phân bố mọi file đã phân loại theo từng nhãn: số file, số phút có tiếng nói, % phút."""
    ok = [r for r in rows if r["labels"].get("genre")]
    tot = sum(r.get("speech_min") or 0 for r in ok) or 1
    ui.console.print(f"[bold]{len(ok)}[/] file có nhãn ({len(rows) - len(ok)} file lỗi parse không tính) · "
                     f"[bold]{tot:.1f}[/] phút có tiếng nói")
    for r in ok:
        r["labels"]["nguồn"] = source_group(r["src"], roots)
    for k in ["nguồn"] + COLS:
        n, m = collections.Counter(), collections.Counter()
        for r in ok:
            v = str(r["labels"].get(k))
            n[v] += 1
            m[v] += r.get("speech_min") or 0
        ui.table(f"Phân bố `{k}`", ["giá trị", ">số file", ">phút", ">% phút"],
                 [(v, n[v], f"{m[v]:.1f}", f"{100 * m[v] / tot:.1f}") for v, _ in m.most_common()])
    for a_, b_ in (("genre", "channel"), ("genre", "noise_type"), ("genre", "accent"), ("nguồn", "genre")):
        t = collections.defaultdict(collections.Counter)
        for r in ok:
            t[str(r["labels"].get(a_))][str(r["labels"].get(b_))] += r.get("speech_min") or 0
        cs = sorted({c for row in t.values() for c in row})
        ui.table(f"Số phút: `{a_}` × `{b_}`", [a_] + [">" + c for c in cs],
                 [[k] + [f"{t[k][c]:.0f}" if t[k][c] else "" for c in cs] for k in sorted(t)])


if __name__ == "__main__":
    main()
