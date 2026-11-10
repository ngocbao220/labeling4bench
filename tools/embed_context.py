#!/usr/bin/env python3
"""Nhúng data/context.json vào index.html (biến CTX)."""
import json
H = open("index.html", encoding="utf-8").read()
c = {str(o["id"]): o for o in json.load(open("data/context.json", encoding="utf-8"))}
for o in c.values(): o.pop("other_vad", None)
a = H.index("var CTX = "); b = H.index(";\n", a)
open("index.html", "w", encoding="utf-8").write(H[:a] + "var CTX = " + json.dumps(c, ensure_ascii=False, separators=(",", ":")) + H[b:])
