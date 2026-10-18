"""Local dashboard server: serves frontend/ and generates explanations on demand.

    GET  /api/data?sector=NIFTY50   prices, anomalies and explanations (built live)
    GET  /api/status                whether explanations can be generated
    POST /api/explain               {"date": "YYYY-MM-DD", "force": false}
                                    runs explain_anomalies.explain_date for that day
    GET  /<anything else>           static files from frontend/

Usage (PowerShell, with the venv active):
    $env:GEMINI_API_KEY = "your-key"
    python backend/src/server.py            # http://127.0.0.1:8000
    python backend/src/server.py --port 8080

Binds to 127.0.0.1 only: the API spends your Gemini quota, so it should not be
reachable from other machines.
"""
import argparse
import json
import os
import re
import sys
import threading
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR / "rag"))

import explain_anomalies  # noqa: E402
import export_frontend_data  # noqa: E402

FRONTEND_DIR = SRC_DIR.parents[1] / "frontend"
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
SECTOR_RE = re.compile(r"^[A-Z0-9]{2,20}$")
MAX_BODY_BYTES = 1024

# One explanation at a time: keeps the explanations file consistent and stays
# inside Gemini's rate limits.
_explain_lock = threading.Lock()
_clients = {}


def gemini_client():
    if "gemini" not in _clients:
        _clients["gemini"] = explain_anomalies.get_gemini_client()
    return _clients["gemini"]


def context_index():
    # Loading the vector store into memory takes a few seconds; do it once.
    if "index" not in _clients:
        _clients["index"] = explain_anomalies.get_context_index()
    return _clients["index"]


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(FRONTEND_DIR), **kwargs)

    # -- helpers --------------------------------------------------------------

    def send_json(self, status: HTTPStatus, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_error_json(self, status: HTTPStatus, message: str) -> None:
        self.send_json(status, {"error": message})

    def sector_param(self, query: dict) -> str | None:
        sector = query.get("sector", [explain_anomalies.SECTOR])[0]
        return sector if SECTOR_RE.match(sector) else None

    # -- routes ---------------------------------------------------------------

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/api/data":
            return self.handle_data(parse_qs(url.query))
        if url.path == "/api/status":
            return self.handle_status()
        if url.path.startswith("/api/"):
            return self.send_error_json(HTTPStatus.NOT_FOUND, "Unknown API endpoint.")
        return super().do_GET()

    def do_POST(self):
        url = urlparse(self.path)
        if url.path == "/api/explain":
            return self.handle_explain(parse_qs(url.query))
        return self.send_error_json(HTTPStatus.NOT_FOUND, "Unknown API endpoint.")

    def handle_data(self, query: dict):
        sector = self.sector_param(query)
        if not sector:
            return self.send_error_json(HTTPStatus.BAD_REQUEST, "Invalid sector.")
        try:
            data = export_frontend_data.build(sector)
        except FileNotFoundError as exc:
            return self.send_error_json(HTTPStatus.NOT_FOUND, str(exc))
        data["can_explain"] = True
        self.send_json(HTTPStatus.OK, data)

    def handle_status(self):
        has_key = bool(os.environ.get("GEMINI_API_KEY"))
        has_store = explain_anomalies.VECTORSTORE_DIR.exists()
        problems = []
        if not has_key:
            problems.append("GEMINI_API_KEY is not set in the terminal that started the server.")
        if not has_store:
            problems.append("The Chroma vector store is missing; run ingest_all_to_chroma.py.")
        self.send_json(HTTPStatus.OK, {"ready": not problems, "problems": problems,
                                       "model": explain_anomalies.GEMINI_MODEL})

    def handle_explain(self, query: dict):
        sector = self.sector_param(query)
        if not sector:
            return self.send_error_json(HTTPStatus.BAD_REQUEST, "Invalid sector.")

        length = int(self.headers.get("Content-Length") or 0)
        if not 0 < length <= MAX_BODY_BYTES:
            return self.send_error_json(HTTPStatus.BAD_REQUEST, "Expected a small JSON body.")
        try:
            body = json.loads(self.rfile.read(length))
        except json.JSONDecodeError:
            return self.send_error_json(HTTPStatus.BAD_REQUEST, "Body is not valid JSON.")

        date = str(body.get("date", ""))
        if not DATE_RE.match(date):
            return self.send_error_json(HTTPStatus.BAD_REQUEST, "date must be YYYY-MM-DD.")

        try:
            with _explain_lock:
                entry = explain_anomalies.explain_date(
                    date, sector, force=bool(body.get("force")),
                    get_client=gemini_client, get_index=context_index)
        except ValueError as exc:            # not a trading day / not flagged
            return self.send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
        except (RuntimeError, FileNotFoundError) as exc:   # missing key / vector store
            return self.send_error_json(HTTPStatus.SERVICE_UNAVAILABLE, str(exc))
        except Exception as exc:             # Gemini / Chroma failures
            self.log_error("explain %s failed: %r", date, exc)
            return self.send_error_json(HTTPStatus.BAD_GATEWAY, f"Explanation failed: {exc}")

        self.send_json(HTTPStatus.OK, {
            "date": entry["date"],
            "explanation": export_frontend_data.format_explanation(entry),
        })


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), DashboardHandler)
    print(f"Dashboard: http://127.0.0.1:{args.port}/")
    if not os.environ.get("GEMINI_API_KEY"):
        print("Note: GEMINI_API_KEY is not set -- the page will load, but "
              "'See explanation' will report that the key is missing.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
