#!/usr/bin/env python3
"""Xem quyết định LẤY / BỎ của collect.py cho từng file nguồn, kèm lý do và nhãn mô hình gán.

Đọc <paths.raw>/classified.jsonl (mỗi file lấy dòng mới nhất).
Chạy:
  python3 tools/decisions.py                      # tổng hợp + toàn bộ danh sách
  python3 tools/decisions.py --only BỎ --limit 50
  python3 tools/decisions.py --grep interview     # lọc theo đường dẫn, lý do hoặc nhãn
  python3 tools/decisions.py --csv out/decisions.csv collect=local
"""
import argparse, collections, csv, os, re

import conf
import ui
from ask_models import load_jsonl

COLS = ["genre", "channel", "noise_type", "distance", "accent", "speakers"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["LẤY", "BỎ"])
    ap.add_argument("--grep")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--csv")
    ap.add_argument("--raw", action="store_true", help="in câu trả lời thô của mô hình cho các file lỗi parse")
    argv, overrides = conf.split_argv()
    a = ap.parse_args(argv)
    p = os.path.join(conf.path(conf.load(overrides), "raw"), "classified.jsonl")
    if not os.path.exists(p):
        ui.console.print(f"[red]Chưa có {p}[/]: chạy tools/collect.py trước.")
        return
    rows = list({r["id"]: r for r in load_jsonl(p)}.values())
    for r in rows:  # bản ghi cũ (trước khi có trường decision)
        r.setdefault("decision", "LẤY" if r.get("used") else "BỎ")
        r.setdefault("reason", "")

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
             [(os.path.basename(r["src"]), "[green]LẤY[/]" if r["decision"] == "LẤY" else "[yellow]BỎ[/]", r["reason"],
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


if __name__ == "__main__":
    main()
