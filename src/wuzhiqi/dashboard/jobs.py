"""Small in-process job manager for dashboard actions."""

from __future__ import annotations

import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from shutil import which
from typing import Any, Callable


@dataclass
class Job:
    id: str
    kind: str
    status: str = "queued"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    result: Any = None
    error: str | None = None
    stdout: str = ""
    stderr: str = ""


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def submit(self, kind: str, func: Callable[[], Any]) -> Job:
        job = Job(id=uuid.uuid4().hex, kind=kind)
        with self._lock:
            self._jobs[job.id] = job
        thread = threading.Thread(target=self._run, args=(job.id, func), daemon=True)
        thread.start()
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def _run(self, job_id: str, func: Callable[[], Any]) -> None:
        self._update(job_id, status="running")
        try:
            result = func()
        except Exception as exc:  # pragma: no cover - defensive job boundary
            self._update(job_id, status="failed", error=str(exc))
        else:
            self._update(job_id, status="succeeded", result=result)

    def _update(self, job_id: str, **changes: Any) -> None:
        with self._lock:
            job = self._jobs[job_id]
            for key, value in changes.items():
                setattr(job, key, value)
            job.updated_at = time.time()


CODEX_PROMPTS = {
    "training_health": "Analyze this Wuzhiqi AlphaZero training log. Summarize current health, risks, and next checks.",
    "value_head": "Analyze the value-head learning signals in this Wuzhiqi training log. Focus on explained variance, value loss, and target balance.",
    "kl_lr": "Analyze KL divergence and adaptive learning-rate behavior in this Wuzhiqi training log. Identify instability or throttling.",
    "checkpoint_readiness": "Review whether this Wuzhiqi training run appears ready for checkpoint evaluation or promotion.",
    "next_experiment": "Recommend the next Wuzhiqi training or evaluation experiment based on this log.",
}


def codex_available(root: Path) -> dict[str, Any]:
    command = resolve_codex_command()
    if command is None:
        return {"available": False, "error": "codex command not found"}
    try:
        completed = subprocess.run(
            [command, "--version"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except OSError as exc:
        return {"available": False, "error": str(exc)}
    return {
        "available": completed.returncode == 0,
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def run_codex_analysis(root: Path, log_path: Path, prompt_id: str) -> dict[str, Any]:
    command = resolve_codex_command()
    if command is None:
        raise FileNotFoundError("codex command not found")
    prompt = CODEX_PROMPTS[prompt_id]
    log_excerpt = log_path.read_text(encoding="utf-8", errors="replace")[-60000:]
    full_prompt = f"{prompt}\n\nLog path: {log_path}\n\nRecent log excerpt:\n```text\n{log_excerpt}\n```"
    completed = subprocess.run(
        [command, "exec", "-"],
        cwd=root,
        input=full_prompt,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    output = completed.stdout.strip() or completed.stderr.strip()
    out_dir = root / "experiments" / "dashboard_analyses"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{int(time.time())}_{prompt_id}.txt"
    out_path.write_text(output, encoding="utf-8")
    return {
        "prompt_id": prompt_id,
        "command": command,
        "log": log_path.relative_to(root).as_posix(),
        "returncode": completed.returncode,
        "output": output,
        "path": out_path.relative_to(root).as_posix(),
    }


def resolve_codex_command() -> str | None:
    """Find Codex CLI across ordinary PATH and Windows npm shim locations."""

    for name in ("codex", "codex.cmd", "codex.ps1"):
        resolved = which(name)
        if resolved:
            return resolved

    home = Path.home()
    candidates = [
        home / "AppData" / "Roaming" / "npm" / "codex.cmd",
        home / "AppData" / "Roaming" / "npm" / "codex.ps1",
        home / "AppData" / "Roaming" / "npm" / "codex",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None
