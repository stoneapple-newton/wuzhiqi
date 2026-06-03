"""Dependency-free local HTTP server for the Wuzhiqi dashboard."""

from __future__ import annotations

import json
import mimetypes
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from wuzhiqi.dashboard.games import HumanVsModelSession
from wuzhiqi.dashboard.jobs import CODEX_PROMPTS, JobManager, codex_available, run_codex_analysis
from wuzhiqi.dashboard.logs import discover_logs, parse_log, resolve_log, summarize_records
from wuzhiqi.evaluate import evaluate_checkpoint_match


ROOT = Path.cwd()
STATIC_DIR = Path(__file__).with_name("static")
JOBS = JobManager()
GAMES: dict[str, HumanVsModelSession] = {}


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "WuzhiqiDashboard/0.1"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                self._send_file(STATIC_DIR / "index.html")
            elif parsed.path.startswith("/static/"):
                self._send_file(STATIC_DIR / parsed.path.removeprefix("/static/"))
            elif parsed.path == "/api/health":
                self._json({"ok": True, "root": str(ROOT), "codex": codex_available(ROOT)})
            elif parsed.path == "/api/logs":
                self._json({"logs": [ref.__dict__ | {"path": str(ref.path)} for ref in discover_logs(ROOT)]})
            elif parsed.path == "/api/log-summary":
                query = parse_qs(parsed.query)
                log_id = query.get("id", [""])[0]
                path = resolve_log(ROOT, log_id)
                self._json(summarize_records(parse_log(path)) | {"log": log_id})
            elif parsed.path == "/api/checkpoints":
                self._json({"checkpoints": list_checkpoints()})
            elif parsed.path == "/api/prompts":
                self._json({"prompts": sorted(CODEX_PROMPTS)})
            elif parsed.path.startswith("/api/jobs/"):
                job_id = parsed.path.rsplit("/", maxsplit=1)[-1]
                job = JOBS.get(job_id)
                if job is None:
                    self._json({"error": "job not found"}, status=404)
                else:
                    self._json(job.__dict__)
            else:
                self._json({"error": "not found"}, status=404)
        except Exception as exc:
            self._json({"error": str(exc)}, status=500)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            body = self._read_json()
            if parsed.path == "/api/analysis/codex":
                log_path = resolve_log(ROOT, str(body["log_id"]))
                prompt_id = str(body["prompt_id"])
                if prompt_id not in CODEX_PROMPTS:
                    raise ValueError("unknown prompt_id")
                job = JOBS.submit("codex_analysis", lambda: run_codex_analysis(ROOT, log_path, prompt_id))
                self._json({"job_id": job.id, "status": job.status}, status=202)
            elif parsed.path == "/api/evaluate":
                job = JOBS.submit("evaluation", lambda: run_evaluation_job(body))
                self._json({"job_id": job.id, "status": job.status}, status=202)
            elif parsed.path == "/api/matches/model-vs-model":
                payload = dict(body)
                payload["opponent"] = "model"
                job = JOBS.submit("model_vs_model", lambda: run_evaluation_job(payload))
                self._json({"job_id": job.id, "status": job.status}, status=202)
            elif parsed.path == "/api/games/human-vs-model":
                game_id = uuid.uuid4().hex
                session = HumanVsModelSession(
                    config_path=ROOT / str(body.get("config", "config/default.yaml")),
                    model_path=ROOT / str(body["model"]) if body.get("model") else None,
                    playouts=int(body["playouts"]) if body.get("playouts") else None,
                    human_first=bool(body.get("human_first", True)),
                    use_heuristic=bool(body.get("use_heuristic", False)),
                )
                GAMES[game_id] = session
                self._json({"game_id": game_id, "state": session.snapshot()})
            elif parsed.path.startswith("/api/games/") and parsed.path.endswith("/move"):
                game_id = parsed.path.split("/")[3]
                session = GAMES[game_id]
                state = session.human_move(int(body["row"]), int(body["col"]))
                self._json({"game_id": game_id, "state": state})
            else:
                self._json({"error": "not found"}, status=404)
        except Exception as exc:
            self._json({"error": str(exc)}, status=500)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _json(self, payload: object, status: int = 200) -> None:
        data = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_file(self, path: Path) -> None:
        resolved = path.resolve()
        if STATIC_DIR.resolve() not in resolved.parents and resolved != STATIC_DIR.resolve():
            self._json({"error": "forbidden"}, status=403)
            return
        if not resolved.exists() or not resolved.is_file():
            self._json({"error": "not found"}, status=404)
            return
        data = resolved.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(resolved.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args) -> None:
        return None


def list_checkpoints() -> list[dict[str, object]]:
    folder = ROOT / "checkpoints"
    if not folder.exists():
        return []
    out = []
    for path in sorted(folder.glob("*.pt"), key=lambda item: item.stat().st_mtime, reverse=True):
        stat = path.stat()
        out.append(
            {
                "id": path.relative_to(ROOT).as_posix(),
                "name": path.name,
                "size": stat.st_size,
                "modified": stat.st_mtime,
            }
        )
    return out


def run_evaluation_job(payload: dict) -> dict[str, object]:
    model = ROOT / str(payload["model"])
    opponent = str(payload.get("opponent", "heuristic"))
    opponent_model = ROOT / str(payload["opponent_model"]) if payload.get("opponent_model") else None
    result = evaluate_checkpoint_match(
        model_file=model,
        opponent=opponent,
        opponent_model_file=opponent_model,
        config_path=ROOT / str(payload.get("config", "config/default.yaml")),
        games=int(payload["games"]) if payload.get("games") else None,
        playouts=int(payload["playouts"]) if payload.get("playouts") else None,
    ).as_dict()
    out_dir = ROOT / "experiments" / "evaluations"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{int(time.time())}_{opponent}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["path"] = out_path.relative_to(ROOT).as_posix()
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the local Wuzhiqi dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"Wuzhiqi dashboard: http://{args.host}:{args.port}")
    server.serve_forever()

