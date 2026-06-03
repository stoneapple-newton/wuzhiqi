"""Training log discovery and parsing for the dashboard."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any


TEXT_ROW_RE = re.compile(
    r"batch i:(?P<batch>\d+), episode_len:(?P<episode_len>\d+)"
    r"(?:,start_player:(?P<start_player>-?\d+),winner:(?P<winner>-?\d+))?"
    r".*?kl:(?P<kl>[\d.\-eE]+),lr_multiplier:(?P<lr_multiplier>[\d.\-eE]+)"
    r"(?:,effective_lr:(?P<effective_lr>[\d.\-eE]+))?"
    r",loss:(?P<loss>[\d.\-eE]+)"
    r"(?:,policy_loss:(?P<policy_loss>[\d.\-eE]+),value_loss:(?P<value_loss>[\d.\-eE]+))?"
    r",entropy:(?P<entropy>[\d.\-eE]+).*?"
    r"explained_var_old:(?P<explained_var_old>[\d.\-eE]+),"
    r"explained_var_new:(?P<explained_var_new>[\d.\-eE]+)",
    re.DOTALL,
)


@dataclass(frozen=True)
class LogRef:
    id: str
    path: Path
    size: int
    modified: float


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def discover_logs(root: Path) -> list[LogRef]:
    candidates: list[Path] = []
    for folder in (root / "experiments", root / "data"):
        if folder.exists():
            candidates.extend(path for path in folder.iterdir() if path.suffix.lower() in {".log", ".jsonl", ".txt"})
    refs = []
    for path in sorted(candidates, key=lambda item: item.stat().st_mtime, reverse=True):
        stat = path.stat()
        refs.append(LogRef(id=path.relative_to(root).as_posix(), path=path, size=stat.st_size, modified=stat.st_mtime))
    return refs


def resolve_log(root: Path, log_id: str) -> Path:
    path = (root / log_id).resolve()
    allowed_roots = [(root / "experiments").resolve(), (root / "data").resolve()]
    if not any(path == allowed or allowed in path.parents for allowed in allowed_roots):
        raise ValueError("log path must be under experiments/ or data/")
    if not path.exists():
        raise FileNotFoundError(log_id)
    return path


def parse_log(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return _parse_jsonl(path)
    return _parse_text_log(path)


def _parse_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("event") != "batch":
            continue
        normalized = {
            "batch": _safe_int(record.get("batch")),
            "episode_len": _safe_int(record.get("episode_len")),
            "start_player": _safe_int(record.get("start_player")),
            "winner": _safe_int(record.get("winner")),
            "kl": _safe_float(record.get("kl")),
            "lr_multiplier": _safe_float(record.get("lr_multiplier")),
            "effective_lr": _safe_float(record.get("effective_lr")),
            "loss": _safe_float(record.get("loss")),
            "policy_loss": _safe_float(record.get("policy_loss")),
            "value_loss": _safe_float(record.get("value_loss")),
            "entropy": _safe_float(record.get("entropy")),
            "replay_buffer": _safe_int(record.get("replay_buffer")),
            "explained_var_old": _safe_float(record.get("explained_var_old")),
            "explained_var_new": _safe_float(record.get("explained_var_new")),
            "target_counts": record.get("target_counts"),
            "reserved_plane_target_counts": record.get("reserved_plane_target_counts"),
            "first_second_winner_counts": record.get("first_second_winner_counts"),
        }
        if normalized["batch"] is not None and normalized["episode_len"] is not None:
            records.append(normalized)
    return records


def _parse_text_log(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    records = []
    for match in TEXT_ROW_RE.finditer(text):
        row = {key: _safe_float(value) for key, value in match.groupdict().items()}
        for key in ("batch", "episode_len", "start_player", "winner"):
            row[key] = _safe_int(match.group(key))
        records.append(row)
    return records


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"records": [], "summary": {}, "windows": [], "alerts": ["No training rows parsed."]}
    numeric_keys = [
        "episode_len",
        "kl",
        "lr_multiplier",
        "effective_lr",
        "loss",
        "policy_loss",
        "value_loss",
        "entropy",
        "replay_buffer",
        "explained_var_old",
        "explained_var_new",
    ]
    first = records[0]
    last = records[-1]
    latest = records[-100:]
    summary: dict[str, Any] = {
        "rows": len(records),
        "first_batch": first["batch"],
        "latest_batch": last["batch"],
        "latest": last,
    }
    for key in numeric_keys:
        values = [float(row[key]) for row in records if row.get(key) is not None]
        if values:
            summary[key] = {
                "latest": last.get(key),
                "average": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "median": median(values),
            }
    alerts = training_alerts(records, latest)
    return {
        "records": records,
        "summary": summary,
        "windows": window_summary(records),
        "alerts": alerts,
        "current_read": current_read(records),
    }


def window_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest_batch = int(records[-1]["batch"])
    starts = [1, 101, 201, 301, 401, max(1, latest_batch - 99)]
    windows: list[tuple[int, int]] = []
    for start in starts:
        end = min(start + 99, latest_batch)
        if start <= latest_batch and (start, end) not in windows:
            windows.append((start, end))
    out = []
    for start, end in windows:
        chunk = [row for row in records if start <= int(row["batch"]) <= end]
        if not chunk:
            continue
        row: dict[str, Any] = {"label": f"{start}-{end}", "rows": len(chunk)}
        for key in ("episode_len", "kl", "lr_multiplier", "loss", "policy_loss", "value_loss", "entropy", "explained_var_new"):
            values = [float(item[key]) for item in chunk if item.get(key) is not None]
            row[key] = sum(values) / len(values) if values else None
        out.append(row)
    return out


def training_alerts(records: list[dict[str, Any]], latest: list[dict[str, Any]]) -> list[str]:
    alerts = []
    ev_zero = all(row.get("explained_var_new") == 0 and row.get("explained_var_old") == 0 for row in latest if row.get("explained_var_new") is not None)
    if ev_zero:
        alerts.append("Explained variance is zero across the latest parsed window; inspect value targets and value-head learning.")
    kl_spikes = sum(1 for row in latest if (row.get("kl") or 0) > 0.04)
    if kl_spikes:
        alerts.append(f"KL exceeded 0.04 in {kl_spikes} of the latest {len(latest)} rows.")
    if records[-1].get("loss") is not None and float(records[-1]["loss"]) != float(records[-1]["loss"]):
        alerts.append("Latest loss is NaN.")
    return alerts


def current_read(records: list[dict[str, Any]]) -> str:
    first = records[:100]
    latest = records[-100:]

    def average(rows: list[dict[str, Any]], key: str) -> float | None:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        return sum(values) / len(values) if values else None

    latest_batch = records[-1]["batch"]
    episode_now = average(latest, "episode_len")
    episode_then = average(first, "episode_len")
    loss_now = average(latest, "loss")
    loss_then = average(first, "loss")
    parts = [f"Latest parsed batch is {latest_batch}."]
    if episode_now is not None and episode_then is not None:
        parts.append(f"Episode length average moved from {episode_then:.1f} early to {episode_now:.1f} in the latest window.")
    if loss_now is not None and loss_then is not None:
        parts.append(f"Loss average moved from {loss_then:.3f} early to {loss_now:.3f} recently.")
    return " ".join(parts)
