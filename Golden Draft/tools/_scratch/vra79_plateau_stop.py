#!/usr/bin/env python3
"""Plateau gate watcher for VRA-79 eval streams.

Reads append-only eval rows and writes a stop sentinel when:
- hard max step is reached, or
- micro-curve gain is flat and medium-curve slope is non-positive.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def _log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Watch eval_stream and trigger a stop sentinel on plateau.")
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--stream-path", default="")
    ap.add_argument("--stop-file", default="")
    ap.add_argument("--poll-s", type=float, default=15.0)
    ap.add_argument("--once", type=int, default=0, choices=[0, 1])

    ap.add_argument("--split", default="disjoint")
    ap.add_argument("--micro-n", type=int, default=512)
    ap.add_argument("--medium-n", type=int, default=2048)

    ap.add_argument("--hard-max-step", type=int, default=10000)
    ap.add_argument("--warmup-step", type=int, default=2000)
    ap.add_argument("--window-checkpoints", type=int, default=20)
    ap.add_argument("--window-gain-threshold", type=float, default=0.0005)
    ap.add_argument("--medium-window", type=int, default=4)
    ap.add_argument("--medium-slope-threshold", type=float, default=0.0)
    return ap.parse_args()


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            rows.append(row)
        except Exception:
            continue
    return rows


def _dedupe_by_step(rows: list[dict[str, Any]], eval_n: int, split: str) -> list[dict[str, Any]]:
    by_step: dict[int, dict[str, Any]] = {}
    for row in rows:
        try:
            if int(row.get("eval_n", -1)) != int(eval_n):
                continue
            if str(row.get("split", "")) != str(split):
                continue
            step = int(row.get("step", -1))
            if step <= 0:
                continue
            by_step[step] = row
        except Exception:
            continue
    return [by_step[k] for k in sorted(by_step.keys())]


def _series_slope(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    sx = float(n - 1) * float(n) / 2.0
    sxx = float(n - 1) * float(n) * float(2 * n - 1) / 6.0
    sy = float(sum(values))
    sxy = float(sum(float(i) * float(v) for i, v in enumerate(values)))
    den = float(n) * sxx - sx * sx
    if den == 0.0:
        return 0.0
    return (float(n) * sxy - sx * sy) / den


def _acc_delta(row: dict[str, Any]) -> float | None:
    val = row.get("acc_delta")
    if isinstance(val, (int, float)):
        return float(val)
    return None


def _write_stop_file(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _evaluate(args: argparse.Namespace, rows: list[dict[str, Any]]) -> tuple[bool, dict[str, Any]]:
    micro = _dedupe_by_step(rows, int(args.micro_n), str(args.split))
    medium = _dedupe_by_step(rows, int(args.medium_n), str(args.split))
    latest_step = int(micro[-1]["step"]) if micro else 0

    if latest_step <= 0:
        return False, {"reason": "no_micro_rows", "latest_step": 0}
    if latest_step >= int(args.hard_max_step):
        return True, {"reason": "hard_max_step", "latest_step": latest_step}
    if latest_step < int(args.warmup_step):
        return False, {"reason": "warmup_not_reached", "latest_step": latest_step}

    micro_window = micro[-int(args.window_checkpoints) :]
    if len(micro_window) < int(args.window_checkpoints):
        return False, {"reason": "insufficient_micro_window", "latest_step": latest_step}

    micro_deltas = [val for val in (_acc_delta(row) for row in micro_window) if val is not None]
    if len(micro_deltas) < int(args.window_checkpoints):
        return False, {"reason": "missing_micro_acc_delta", "latest_step": latest_step}
    micro_gain = max(micro_deltas) - min(micro_deltas)

    medium_window = medium[-int(args.medium_window) :]
    if len(medium_window) < int(args.medium_window):
        return False, {"reason": "insufficient_medium_window", "latest_step": latest_step, "micro_gain": micro_gain}
    medium_deltas = [val for val in (_acc_delta(row) for row in medium_window) if val is not None]
    if len(medium_deltas) < int(args.medium_window):
        return False, {"reason": "missing_medium_acc_delta", "latest_step": latest_step, "micro_gain": micro_gain}
    medium_slope = _series_slope(medium_deltas)

    plateau = (micro_gain < float(args.window_gain_threshold)) and (
        medium_slope <= float(args.medium_slope_threshold)
    )
    meta = {
        "reason": "plateau" if plateau else "continue",
        "latest_step": latest_step,
        "micro_gain": micro_gain,
        "medium_slope": medium_slope,
        "window_checkpoints": int(args.window_checkpoints),
        "medium_window": int(args.medium_window),
    }
    return bool(plateau), meta


def main() -> int:
    args = _parse_args()
    run_root = Path(args.run_root).resolve()
    stream_path = Path(args.stream_path).resolve() if args.stream_path else (run_root / "eval_stream.jsonl")
    stop_file = Path(args.stop_file).resolve() if args.stop_file else (run_root / "stop_now.flag")
    _log(f"watching stream={stream_path}")
    _log(f"stop_file={stop_file}")

    while True:
        rows = _read_rows(stream_path)
        should_stop, meta = _evaluate(args, rows)
        _log(f"gate: {meta}")
        if should_stop:
            payload = {
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "gate": meta,
                "run_root": str(run_root),
                "stream_path": str(stream_path),
            }
            _write_stop_file(stop_file, payload)
            _log("stop sentinel written")
            return 0

        if int(args.once):
            return 0
        time.sleep(max(1.0, float(args.poll_s)))


if __name__ == "__main__":
    raise SystemExit(main())

