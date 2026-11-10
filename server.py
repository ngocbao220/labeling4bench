#!/usr/bin/env python3
"""Backend đơn giản cho trang gán nhãn: phục vụ file tĩnh + lưu cấu hình câu hỏi.

Chạy:   python3 server.py                 -> http://127.0.0.1:8000
Đặt mật khẩu sửa:  EDIT_TOKEN=matkhau python3 server.py
Mở cho máy khác:   HOST=0.0.0.0 PORT=8000 EDIT_TOKEN=matkhau python3 server.py

API:
  GET    /api/config  -> {"config": {...}|null, "editable": true|false}
  PUT    /api/config  -> lưu cấu hình (cần header X-Token nếu có EDIT_TOKEN)
  DELETE /api/config  -> xoá cấu hình đã lưu, quay về mặc định trong index.html
  GET    /api/samples -> các segment trong manifests/*.jsonl có clip tồn tại (trang dùng thay danh sách mẫu cứng)
Cấu hình lưu ở data/config.json (bản cũ giữ ở data/config.json.bak).
"""
import re
import glob, hmac, json, os, tempfile
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
CFG = os.path.join(DATA, "config.json")
TOKEN = os.environ.get("EDIT_TOKEN", "")
MAX = 300_000
PUBLIC = {"/", "/index.html"}


def valid(cfg):
    """Kiểm tra cấu trúc: {"panels": {tag: [ {id,name,intro,qs|scale...} ]}}"""
    if not isinstance(cfg, dict) or not isinstance(cfg.get("panels"), dict):
        return False
    for tag, panels in cfg["panels"].items():
        if not isinstance(tag, str) or not isinstance(panels, list):
            return False
        for p in panels:
            if not isinstance(p, dict):
                return False
            for k in ("id", "name", "intro"):
                if not isinstance(p.get(k), str):
                    return False
            if not isinstance(p.get("qs", []), list):
                return False
            for q in p.get("qs", []):
                if not (isinstance(q, dict) and isinstance(q.get("id"), str)
                        and isinstance(q.get("title"), str)
                        and isinstance(q.get("opts"), list)
                        and isinstance(q.get("codes", []), list) and all(isinstance(x, str) for x in q.get("codes", []))
                        and all(isinstance(o, list) and o and all(isinstance(x, str) for x in o) for o in q["opts"])):
                    return False
            if p.get("type") == "score":
                sc = p.get("scale")
                if not (isinstance(sc, list) and sc and all(isinstance(o, list) and len(o) >= 2 and all(isinstance(x, str) for x in o) for o in sc)):
                    return False
                d = p.get("descs")
                if d is not None and not (isinstance(d, list) and d and all(isinstance(x, str) for x in d)):
                    return False
    return True


class H(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=ROOT, **k)

    def send_json(self, code, obj):
        b = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def authed(self):
        return not TOKEN or hmac.compare_digest(self.headers.get("X-Token", ""), TOKEN)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/api/config":
            cfg = None
            if os.path.exists(CFG):
                with open(CFG, encoding="utf8") as f:
                    cfg = json.load(f)
            return self.send_json(200, {"config": cfg, "editable": True, "needs_token": bool(TOKEN)})
        if path == "/api/samples":
            rows = []
            for p in sorted(glob.glob(os.path.join(ROOT, "manifests", "*.jsonl"))):
                with open(p, encoding="utf8") as f:
                    rows += [json.loads(l) for l in f if l.strip()]
            rows = [r for r in rows if r.get("audio") and os.path.exists(os.path.join(ROOT, r["audio"]))]
            return self.send_json(200, {"samples": rows})
        # chỉ phục vụ trang, audio ví dụ và clip segment; không lộ server.py hay data/
        if path.startswith("/audio/") or path.startswith("/clips/"):
            return self.send_media(path)
        if path in PUBLIC:
            return super().do_GET()
        self.send_error(404)

    def send_media(self, path):
        """Phục vụ audio có hỗ trợ Range (trình duyệt cần để tua/đặt currentTime)."""
        import mimetypes, posixpath
        from urllib.parse import unquote
        fp = os.path.normpath(os.path.join(ROOT, unquote(path).lstrip("/")))
        if not fp.startswith(ROOT + os.sep) or not os.path.isfile(fp):
            return self.send_error(404)
        size = os.path.getsize(fp)
        a, b, code = 0, size - 1, 200
        m = re.match(r"bytes=(\d*)-(\d*)$", self.headers.get("Range", "").strip())
        if m and (m.group(1) or m.group(2)):
            if m.group(1):
                a = int(m.group(1))
                b = int(m.group(2)) if m.group(2) else size - 1
            else:
                a = max(0, size - int(m.group(2)))
            b = min(b, size - 1)
            if a > b:
                self.send_response(416); self.send_header("Content-Range", "bytes */%d" % size); self.end_headers(); return
            code = 206
        self.send_response(code)
        self.send_header("Content-Type", mimetypes.guess_type(fp)[0] or "application/octet-stream")
        self.send_header("Accept-Ranges", "bytes")
        if code == 206:
            self.send_header("Content-Range", "bytes %d-%d/%d" % (a, b, size))
        self.send_header("Content-Length", str(b - a + 1))
        self.end_headers()
        with open(fp, "rb") as f:
            f.seek(a)
            left = b - a + 1
            while left > 0:
                chunk = f.read(min(65536, left))
                if not chunk:
                    break
                self.wfile.write(chunk); left -= len(chunk)

    def do_PUT(self):
        if self.path.split("?")[0] != "/api/config":
            return self.send_error(404)
        if not self.authed():
            return self.send_json(401, {"error": "Sai hoặc thiếu mật khẩu sửa"})
        n = int(self.headers.get("Content-Length", 0))
        if n <= 0 or n > MAX:
            return self.send_json(413, {"error": "Dữ liệu quá lớn hoặc rỗng"})
        try:
            cfg = json.loads(self.rfile.read(n).decode("utf8"))
        except ValueError:
            return self.send_json(400, {"error": "JSON không hợp lệ"})
        if not valid(cfg):
            return self.send_json(400, {"error": "Cấu trúc câu hỏi không hợp lệ"})
        os.makedirs(DATA, exist_ok=True)
        if os.path.exists(CFG):
            os.replace(CFG, CFG + ".bak")
        fd, tmp = tempfile.mkstemp(dir=DATA)
        with os.fdopen(fd, "w", encoding="utf8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=1)
        os.replace(tmp, CFG)
        self.send_json(200, {"ok": True})

    def do_DELETE(self):
        if self.path.split("?")[0] != "/api/config":
            return self.send_error(404)
        if not self.authed():
            return self.send_json(401, {"error": "Sai hoặc thiếu mật khẩu sửa"})
        if os.path.exists(CFG):
            os.replace(CFG, CFG + ".bak")
        self.send_json(200, {"ok": True})


if __name__ == "__main__":
    host, port = os.environ.get("HOST", "127.0.0.1"), int(os.environ.get("PORT", "8000"))
    print(f"Mở http://{host}:{port}  (mật khẩu sửa: {'có' if TOKEN else 'KHÔNG đặt'})")
    ThreadingHTTPServer((host, port), H).serve_forever()
