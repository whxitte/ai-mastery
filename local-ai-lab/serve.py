"""Local preview server that applies the same security headers as vercel.json.

Usage:  python3 serve.py [port]      (default port 8000, binds to 127.0.0.1 only)
"""
import http.server
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load_rules():
    config = json.loads((ROOT / "vercel.json").read_text(encoding="utf-8"))
    rules = []
    for rule in config.get("headers", []):
        # vercel "source" patterns used here are plain regexes once (.*) is honoured
        pattern = re.compile("^" + rule["source"].replace("\\\\", "\\") + "$")
        rules.append((pattern, [(h["key"], h["value"]) for h in rule["headers"]]))
    return rules


RULES = load_rules()


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map, ".md": "text/markdown; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".svg": "image/svg+xml"}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def end_headers(self):
        path = self.path.split("?", 1)[0]
        for pattern, headers in RULES:
            if pattern.match(path):
                for key, value in headers:
                    if key.lower() != "content-type":
                        self.send_header(key, value)
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    with http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler) as httpd:
        print(f"Local AI Lab running at http://127.0.0.1:{port}  (Ctrl+C to stop)")
        httpd.serve_forever()
