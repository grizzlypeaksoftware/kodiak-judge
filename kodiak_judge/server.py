"""Local Kodiak judge: a JSON API and a web page, Python standard library only (no web framework).

    uv run python -m kodiak_judge.server                       # http://127.0.0.1:8790  (model: Kodiak-v0.4-1B)
    uv run python -m kodiak_judge.server --model cortex-agent-llc/kodiak-v0.4-1b-accuracy --port 8790

API
    GET  /health                -> {"ok": true, "model": ...}
    GET  /v1/info               -> model, default threshold, measured trade-off curve (if a report is present)
    POST /v1/judge              {"question", "answer_1", "answer_2", "threshold"?}  -> one verdict
    POST /v1/judge/batch        {"items": [{"question", "answer_1", "answer_2"}, ...], "threshold"?} -> {"results": [...]}
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from kodiak_judge.judge import Judge

HERE = Path(__file__).resolve().parent
MAX_BATCH = 64


def make_handler(judge: Judge, model: str, curve: dict | None):
    lock = threading.Lock()  # one GPU: run requests one at a time

    class H(BaseHTTPRequestHandler):
        def _send(self, code: int, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, fmt, *args):  # quieter log
            print(time.strftime("%H:%M:%S"), self.command, self.path, fmt % args if "%" in fmt else "")

        def do_GET(self):
            self.path = urlsplit(self.path).path  # ignore ?query (the page uses ?ex=...)
            if self.path in ("/", "/index.html"):
                return self._send(200, (HERE / "static" / "index.html").read_bytes(), "text/html; charset=utf-8")
            if self.path == "/health":
                return self._send(200, {"ok": True, "model": model})
            if self.path == "/v1/info":
                return self._send(200, {"model": model, "threshold": judge.threshold, "labels": ["first", "second", "tie"],
                                        "both_orders": judge.both_orders, "escalate_on_disagreement": judge.escalate_on_disagreement,
                                        "measured": curve})
            self._send(404, {"error": "not found"})

        def do_POST(self):
            try:
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n) or b"{}")
            except (ValueError, json.JSONDecodeError):
                return self._send(400, {"error": "body must be JSON"})
            if not isinstance(body, dict):
                return self._send(400, {"error": "body must be a JSON object"})
            t = body.get("threshold")
            if t is not None and not (isinstance(t, (int, float)) and 0 <= t <= 1):
                return self._send(400, {"error": "threshold must be a number from 0 to 1"})
            if self.path == "/v1/judge":
                items = [body]
            elif self.path == "/v1/judge/batch":
                items = body.get("items")
                if not isinstance(items, list) or not items or len(items) > MAX_BATCH:
                    return self._send(400, {"error": f"items must be a list of 1 to {MAX_BATCH} pairs"})
            else:
                return self._send(404, {"error": "not found"})
            for it in items:
                if not all(isinstance(it.get(k), str) and it.get(k).strip() for k in ("question", "answer_1", "answer_2")):
                    return self._send(400, {"error": "each pair needs non-empty strings: question, answer_1, answer_2"})
            start = time.perf_counter()
            with lock:
                try:
                    res = judge.judge_many(items, threshold=t)
                except Exception as e:  # e.g. input longer than the model's ~8k-token limit
                    return self._send(422, {"error": f"{type(e).__name__}: {e}"})
            ms = round((time.perf_counter() - start) * 1000, 1)
            if self.path == "/v1/judge":
                return self._send(200, {**res[0], "model": model, "ms": ms})
            self._send(200, {"results": res, "model": model, "ms": ms})

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="cortex-agent-llc/kodiak-v0.4-1b")
    ap.add_argument("--host", default="127.0.0.1", help="127.0.0.1 = this machine only; 0.0.0.0 to reach it over your network")
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--threshold", type=float, default=0.7)
    ap.add_argument("--curve", default="", help="a reports/curve-*.json to show on the page")
    a = ap.parse_args()
    judge = Judge.load(a.model, threshold=a.threshold)
    curve = json.loads(Path(a.curve).read_text()) if a.curve else None
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(judge, a.model, curve))
    print(f"Kodiak judge on http://{a.host}:{a.port}  (model {a.model})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
