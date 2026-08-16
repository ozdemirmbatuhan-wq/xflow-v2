#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import mimetypes
import threading
import uuid
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from aeropt.pipeline import DEFAULT_REQUEST, InputError, run_design
from aeropt.diagnostics import diagnose_runtime_failure
from aeropt.flow5 import Flow5CancelledError


ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
MAX_BODY_BYTES = 1_000_000


class OptimizationJobs:
    """Small in-process job registry for progress reporting and cancellation."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, dict[str, object]] = {}

    def create(self, payload: dict[str, object]) -> dict[str, object]:
        job_id = uuid.uuid4().hex
        cancel_event = threading.Event()
        job: dict[str, object] = {
            "id": job_id,
            "status": "queued",
            "progress": {
                "stage": "queued",
                "percent": 0.0,
                "current": 0,
                "total": 1,
                "message": "Optimizasyon sıraya alındı",
            },
            "cancel_event": cancel_event,
            "result": None,
            "best_so_far": None,
            "error": None,
        }
        with self._lock:
            self._jobs[job_id] = job

        def update_progress(progress: dict[str, object]) -> None:
            with self._lock:
                current = self._jobs.get(job_id)
                if current is not None:
                    progress_payload = dict(progress)
                    best_update = progress_payload.pop("best_so_far", None)
                    if isinstance(best_update, dict):
                        accumulated = dict(current.get("best_so_far") or {})
                        accumulated.update(
                            {
                                key: value
                                for key, value in best_update.items()
                                if value is not None
                            }
                        )
                        current["best_so_far"] = accumulated or None
                    current["progress"] = progress_payload

        def run() -> None:
            with self._lock:
                job["status"] = "running"
            try:
                result = run_design(
                    payload,
                    progress_callback=update_progress,
                    cancel_event=cancel_event,
                )
                with self._lock:
                    job["result"] = result
                    job["status"] = "completed"
            except Flow5CancelledError as exc:
                with self._lock:
                    job["status"] = "cancelled"
                    job["error"] = {"type": "cancelled", "message": str(exc)}
                    job["progress"] = {
                        **dict(job.get("progress") or {}),
                        "message": (
                            "Optimizasyon durduruldu; en iyi tamamlanan ara sonuç hazır"
                            if job.get("best_so_far")
                            else "Optimizasyon durduruldu; henüz gösterilebilir aday tamamlanmadı"
                        ),
                    }
            except InputError as exc:
                with self._lock:
                    job["status"] = "failed"
                    job["error"] = {"type": "input", "message": str(exc)}
            except Exception as exc:
                with self._lock:
                    job["status"] = "failed"
                    job["error"] = {
                        "type": "runtime",
                        "message": "Optimizasyon sırasında beklenmeyen bir hata oluştu.",
                        "detail": str(exc),
                        "diagnosis": diagnose_runtime_failure(exc),
                    }

        threading.Thread(target=run, name=f"aeropt-job-{job_id[:8]}", daemon=True).start()
        return self.snapshot(job_id) or {"id": job_id, "status": "queued"}

    def snapshot(self, job_id: str) -> dict[str, object] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            return {
                key: value
                for key, value in job.items()
                if key != "cancel_event" and (key != "result" or value is not None)
            }

    def cancel(self, job_id: str) -> dict[str, object] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            event = job["cancel_event"]
            if isinstance(event, threading.Event):
                event.set()
            if job["status"] == "queued":
                job["status"] = "cancelled"
            return {
                "id": job_id,
                "status": job["status"],
                "cancel_requested": True,
            }


JOBS = OptimizationJobs()


class AeroOptHandler(BaseHTTPRequestHandler):
    server_version = "AeroOpt/0.8.0"

    def _json(self, status: int, value: object) -> None:
        body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _static(self, filename: str) -> None:
        allowed = {"index.html", "app.js", "styles.css"}
        if filename not in allowed:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        path = STATIC / filename
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self'; script-src 'self'; img-src 'self' data:")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in {"/", "/index.html"}:
            self._static("index.html")
        elif path in {"/app.js", "/styles.css"}:
            self._static(path[1:])
        elif path == "/api/defaults":
            self._json(HTTPStatus.OK, DEFAULT_REQUEST)
        elif path == "/api/health":
            self._json(HTTPStatus.OK, {"ok": True, "service": "AeroOpt"})
        elif path.startswith("/api/jobs/"):
            job_id = path.removeprefix("/api/jobs/").strip("/")
            job = JOBS.snapshot(job_id)
            if job is None:
                self._json(HTTPStatus.NOT_FOUND, {"error": "Optimizasyon işi bulunamadı"})
            else:
                self._json(HTTPStatus.OK, job)
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path.startswith("/api/jobs/") and path.endswith("/cancel"):
            job_id = path.removeprefix("/api/jobs/").removesuffix("/cancel").strip("/")
            job = JOBS.cancel(job_id)
            if job is None:
                self._json(HTTPStatus.NOT_FOUND, {"error": "Optimizasyon işi bulunamadı"})
            else:
                self._json(HTTPStatus.ACCEPTED, job)
            return
        if path not in {"/api/optimize", "/api/jobs"}:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_BODY_BYTES:
                raise InputError("İstek boyutu geçersiz")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise InputError("İstek bir JSON nesnesi olmalı")
            if path == "/api/jobs":
                self._json(HTTPStatus.ACCEPTED, JOBS.create(payload))
                return
            result = run_design(payload)
            self._json(HTTPStatus.OK, result)
        except (InputError, json.JSONDecodeError) as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc), "type": "input"})
        except Exception as exc:
            self.log_error("optimization error: %r", exc)
            self._json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {
                    "error": "Optimizasyon sırasında beklenmeyen bir hata oluştu.",
                    "detail": str(exc),
                    "diagnosis": diagnose_runtime_failure(exc),
                },
            )

    def do_DELETE(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if not path.startswith("/api/jobs/"):
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        job_id = path.removeprefix("/api/jobs/").strip("/")
        job = JOBS.cancel(job_id)
        if job is None:
            self._json(HTTPStatus.NOT_FOUND, {"error": "Optimizasyon işi bulunamadı"})
        else:
            self._json(HTTPStatus.ACCEPTED, job)

    def log_message(self, format: str, *args: object) -> None:
        print(f"[{self.log_date_time_string()}] {format % args}")


def main() -> None:
    parser = argparse.ArgumentParser(description="AeroOpt yerel web arayüzü")
    parser.add_argument("--host", default="127.0.0.1", help="Dinlenecek adres (varsayılan: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="Port (varsayılan: 8765)")
    parser.add_argument("--no-browser", action="store_true", help="Tarayıcıyı otomatik açma")
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), AeroOptHandler)
    url_host = "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host
    url = f"http://{url_host}:{args.port}"
    print(f"AeroOpt hazır: {url}")
    print("Durdurmak için Ctrl+C")
    if not args.no_browser:
        threading.Timer(0.7, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nAeroOpt durduruldu.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
