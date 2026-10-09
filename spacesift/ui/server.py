"""Local web server for the SpaceSift UI (standard library only; charts load Plotly from a CDN)."""

from __future__ import annotations

import json
import mimetypes
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from .. import json_safe

STATIC = Path(__file__).resolve().parent / "static"
METHODS = ["none", "biweight-0.25", "biweight-0.5", "biweight-1.0", "biweight-2.0", "prewhiten+biweight-1.0"]


def list_experiments(roots: list[Path]) -> list[dict]:
    out = []
    for root in roots:
        for rec in sorted(root.glob("*/record.json")):
            try:
                r = json.loads(rec.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            out.append({"id": r.get("id", rec.parent.name), "question": r.get("question", ""),
                        "n_trials": r.get("n_trials"), "n_stars": r.get("n_stars"),
                        "finished": r.get("finished", ""), "root": str(root)})
    return sorted(out, key=lambda e: e["id"])


def experiment_detail(roots: list[Path], exp_id: str) -> dict | None:
    for root in roots:
        d = root / exp_id
        if (d / "record.json").exists():
            r = json.loads((d / "record.json").read_text())
            r.pop("config", None)
            for k in ("summary",):
                if k in r and len(json.dumps(r[k])) > 20000:
                    r[k] = "(large; see record.json)"
            images = sorted(str(p.relative_to(d)).replace("\\", "/") for p in d.rglob("*.png") if "shards" not in p.parts)
            notes = (d / "NOTES.md").read_text(encoding="utf-8") if (d / "NOTES.md").exists() else ""
            analyses = {}
            for a in sorted((d / "analysis").glob("*/analysis.json")) if (d / "analysis").exists() else []:
                try:
                    analyses[a.parent.name] = json.loads(a.read_text())
                except json.JSONDecodeError:
                    pass
            return {"id": exp_id, "root": str(root), "record": r, "images": images, "notes": notes,
                    "analyses": analyses}
    return None


def make_handler(roots: list[Path]):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quiet console
            pass

        def _send(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200):
            self._send(code, json.dumps(json_safe(obj)).encode(), "application/json")

        def do_GET(self):
            path = unquote(urlparse(self.path).path)
            if path in ("/", "/index.html"):
                return self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            if path == "/api/methods":
                return self._json(METHODS)
            if path == "/api/experiments":
                return self._json(list_experiments(roots))
            if path.startswith("/api/experiment/"):
                d = experiment_detail(roots, path.removeprefix("/api/experiment/"))
                return self._json(d) if d else self._json({"error": "not found"}, 404)
            if path.startswith("/files/"):
                # /files/<exp_id>/<relative path>, confined to that experiment's folder.
                _, _, exp_id, rel = path.split("/", 3)
                for root in roots:
                    base = (root / exp_id).resolve()
                    target = (base / rel).resolve()
                    if base in target.parents and target.is_file():
                        ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
                        return self._send(200, target.read_bytes(), ctype)
                return self._json({"error": "not found"}, 404)
            return self._json({"error": "not found"}, 404)

        def do_POST(self):
            if urlparse(self.path).path != "/api/trial":
                return self._json({"error": "not found"}, 404)
            try:
                length = int(self.headers.get("Content-Length", 0))
                params = json.loads(self.rfile.read(length) or b"{}")
                from .trial import run_ui_trial
                return self._json(run_ui_trial(params))
            except Exception as exc:  # report to the page instead of dropping the connection
                return self._json({"error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()}, 500)

    return Handler


def serve(port: int = 8765, roots: list[Path] | None = None, open_browser: bool = True) -> None:
    roots = [r.resolve() for r in (roots or [])]
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(roots))
    url = f"http://127.0.0.1:{port}/"
    print(f"SpaceSift UI at {url}  (experiments from: {', '.join(map(str, roots)) or 'none'})  Ctrl+C to stop")
    if open_browser:
        import webbrowser
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
