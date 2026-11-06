#!/usr/bin/env python3
"""Đánh giá chất lượng từng câu hỏi: giữ, viết lại, hay bỏ (để mô hình tự gán).

Đầu vào: hai file JSONL, mỗi dòng là một bản ghi như file JSON web xuất ra:
  {"sample_id": 1, "annotator": "an01", "answers": {"tcontent.event": "EOT", ...}}
  --humans  nhiều dòng cho mỗi sample_id (mỗi người gán nhãn một dòng)
  --model   mỗi sample_id một dòng, do Qwen thinker (hoặc mô hình khác) trả lời cùng bộ câu hỏi
Chạy:  python3 audit_questions.py --humans humans.jsonl --model qwen.jsonl [--model other.jsonl ...]

Với mỗi trường (câu hỏi), in:
  n          số mẫu có cả nhãn người và nhãn mô hình
  h_agree    độ đồng thuận giữa các người gán nhãn (tỉ lệ cặp cùng đáp án)
  maj        nhãn đa số của người (cần >= 2 phiếu giống nhau)
  m_acc      mô hình khớp nhãn đa số bao nhiêu %
  kappa      Cohen's kappa giữa mô hình và nhãn đa số
Kết luận:
  DROP     mô hình làm tốt (m_acc >= 0.90 và kappa >= 0.80): bỏ khỏi bộ câu hỏi người, để mô hình tự gán
  REWRITE  người cũng không đồng thuận (h_agree < 0.60): câu hỏi mơ hồ, viết lại hoặc bỏ
  KEEP     người đồng thuận nhưng mô hình sai nhiều: câu hỏi có giá trị, giữ lại
  WATCH    trường hợp còn lại
"""
import argparse, collections, itertools, json, os, sys


def load(path):
    out = []
    with open(path, encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def majority(votes):
    c = collections.Counter(votes).most_common()
    if not c or c[0][1] < 2:
        return None
    if len(c) > 1 and c[1][1] == c[0][1]:
        return None
    return c[0][0]


def pair_agree(votes):
    pairs = list(itertools.combinations(votes, 2))
    return sum(a == b for a, b in pairs) / len(pairs) if pairs else None


def kappa(a, b):
    n = len(a)
    if not n:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = collections.Counter(a), collections.Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--humans", required=True)
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--drop-acc", type=float, default=0.90)
    ap.add_argument("--drop-kappa", type=float, default=0.80)
    ap.add_argument("--min-human-agree", type=float, default=0.60)
    a = ap.parse_args()

    votes = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in load(a.humans):
        for k, v in r["answers"].items():
            if v not in ("UNK",):
                votes[k][r["sample_id"]].append(v)

    rows = {}
    for mp in a.model:
        ans = {r["sample_id"]: r["answers"] for r in load(mp)}
        for field, per in votes.items():
            h, m, ag = [], [], []
            for sid, vs in per.items():
                pa = pair_agree(vs)
                if pa is not None:
                    ag.append(pa)
                mj = majority(vs)
                mv = ans.get(sid, {}).get(field)
                if mj is not None and mv is not None:
                    h.append(mj); m.append(mv)
            rows.setdefault(field, []).append((mp, h, m, ag))

    out = []
    for field in sorted(rows):
        accs, kaps, hag = [], [], None
        for mp, h, m, ag in rows[field]:
            n = len(h)
            acc = sum(x == y for x, y in zip(h, m)) / n if n else None
            kp = kappa(h, m)
            hag = sum(ag) / len(ag) if ag else None
            accs.append(acc); kaps.append(kp)
            f = lambda x: "-" if x is None else f"{x:.2f}"
            out.append([field, os.path.basename(mp), n, f(hag), f(acc), f(kp), None])
        best_acc = max([x for x in accs if x is not None], default=None)
        best_kp = max([x for x in kaps if x is not None], default=None)
        if hag is not None and hag < a.min_human_agree:
            v = "REWRITE"
        elif best_acc is not None and best_acc >= a.drop_acc and (best_kp or 0) >= a.drop_kappa:
            v = "DROP"
        elif best_acc is not None and best_acc < 0.75:
            v = "KEEP"
        else:
            v = "WATCH"
        out[-1][-1] = {"DROP": "[green]DROP[/]", "REWRITE": "[red]REWRITE[/]", "KEEP": "[yellow]KEEP[/]"}.get(v, v)
    try:
        import ui
        ui.table("Audit câu hỏi", ["câu", "mô hình", ">n", ">h_agree", ">m_acc", ">kappa", "kết luận"], out, ui.Console())
    except ImportError:  # không có rich: in dạng cột
        for r in out:
            print("\t".join("" if x is None else str(x) for x in r))


if __name__ == "__main__":
    sys.exit(main())
