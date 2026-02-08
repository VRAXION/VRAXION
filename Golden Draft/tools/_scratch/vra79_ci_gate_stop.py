#!/usr/bin/env python3
"""CI-gated stop watcher for VRA-79 eval streams.

This watcher is intentionally simple and interpretable:
- It watches `eval_stream.jsonl` for a specific (split, eval_n) series.
- It writes a stop sentinel when the eval becomes statistically above chance
  for `consecutive` points:
    ci_low > chance_acc

This is meant for "atomic-friendly" runs where we want to stop early as soon
as the tiny model demonstrates a real learning gradient.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterable


def _log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Stop training when CI low exceeds chance.")
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--stream-path", default="")
    ap.add_argument("--stop-file", default="")
    ap.add_argument("--poll-s", type=float, default=10.0)

    ap.add_argument("--split", default="disjoint")
    ap.add_argument("--eval-n", type=int, default=2048)
    ap.add_argument("--ci", type=int, default=95, choices=[95, 99])
    ap.add_argument("--consecutive", type=int, default=3)
    ap.add_argument("--hard-max-step", type=int, default=10000)
    ap.add_argument("--once", type=int, default=0, choices=[0, 1])
    return ap.parse_args()


def _iter_new_lines(path: Path, offset: int) -> tuple[int, list[str]]:
    if not path.exists():
        return offset, []
    data = path.read_text(encoding="utf-8", errors="replace")
    if offset >= len(data):
        return offset, []
    chunk = data[offset:]
    new_offset = len(data)
    lines = [ln for ln in chunk.splitlines() if ln.strip()]
    return new_offset, lines


def _parse_rows(lines: Iterable[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ln in lines:
        try:
            row = json.loads(ln)
            if isinstance(row, dict):
                rows.append(row)
        except Exception:
            continue
    return rows


def _ci_low_key(ci: int) -> str:
    return "ci95_low" if int(ci) == 95 else "ci99_low"


def _write_stop_file(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    run_root = Path(args.run_root).resolve()
    stream_path = Path(args.stream_path).resolve() if args.stream_path else (run_root / "eval_stream.jsonl")
    stop_file = Path(args.stop_file).resolve() if args.stop_file else (run_root / "stop_now.flag")
    ci_key = _ci_low_key(int(args.ci))

    _log(f"watching stream={stream_path}")
    _log(f"stop_file={stop_file}")
    _log(
        "gate: {ci_key} > chance for {k} consecutive (split={split} eval_n={n})".format(
            ci_key=ci_key,
            k=int(args.consecutive),
            split=str(args.split),
            n=int(args.eval_n),
        )
    )

    offset = 0
    last_seen_step = 0
    consec = 0

    while True:
        offset, lines = _iter_new_lines(stream_path, offset)
        rows = _parse_rows(lines)
        # Process in chronological order.
        rows.sort(key=lambda r: int(r.get("step", 0) or 0))

        latest_step = last_seen_step
        for row in rows:
            try:
                step = int(row.get("step", 0) or 0)
                if step <= last_seen_step:
                    continue
                if int(row.get("eval_n", -1)) != int(args.eval_n):
                    continue
                if str(row.get("split", "")) != str(args.split):
                    continue
                chance = float(row.get("chance_acc"))
                ci_low = float(row.get(ci_key))
            except Exception:
                continue

            latest_step = max(latest_step, step)
            passed = bool(ci_low > chance)
            if passed:
                consec += 1
            else:
                consec = 0
            last_seen_step = step
            _log(
                "step={s} pass={p} consec={c} {ci_key}={cl:.6g} chance={ch:.6g}".format(
                    s=step,
                    p=int(passed),
                    c=consec,
                    ci_key=ci_key,
                    cl=ci_low,
                    ch=chance,
                )
            )

            if consec >= int(args.consecutive):
                payload = {
                    "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "reason": "ci_gate_pass",
                    "ci": int(args.ci),
                    "consecutive": int(args.consecutive),
                    "latest_step": int(step),
                    "split": str(args.split),
                    "eval_n": int(args.eval_n),
                    "gate": {"ci_low": float(ci_low), "chance": float(chance)},
                    "run_root": str(run_root),
                    "stream_path": str(stream_path),
                }
                _write_stop_file(stop_file, payload)
                _log("stop sentinel written (ci_gate_pass)")
                return 0

        if int(latest_step) >= int(args.hard_max_step):
            payload = {
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "reason": "hard_max_step",
                "latest_step": int(latest_step),
                "hard_max_step": int(args.hard_max_step),
                "run_root": str(run_root),
                "stream_path": str(stream_path),
            }
            _write_stop_file(stop_file, payload)
            _log("stop sentinel written (hard_max_step)")
            return 0

        if int(args.once):
            return 0
        time.sleep(max(1.0, float(args.poll_s)))


if __name__ == "__main__":
    raise SystemExit(main())

