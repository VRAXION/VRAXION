#!/usr/bin/env python3
"""Checker for E82 CALC-SCRIBE v003 floor division artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED = [
    "run_manifest.json",
    "progress.jsonl",
    "partial_aggregate_snapshot.json",
    "seed_results.json",
    "aggregate_metrics.json",
    "decision.json",
    "row_level_failure_examples.jsonl",
    "report.md",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="target/pilot_wave/e82_calc_scribe_v003_floor_division_confirm")
    parser.add_argument("--write-summary", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    failures: list[str] = []
    for name in REQUIRED:
        if not (out / name).exists():
            failures.append(f"missing artifact: {name}")
    if not failures:
        manifest = read_json(out / "run_manifest.json")
        aggregate = read_json(out / "aggregate_metrics.json")
        decision = read_json(out / "decision.json")
        progress = [line for line in (out / "progress.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        if manifest.get("artifact_contract") != "E82_CALC_SCRIBE_V003_FLOOR_DIVISION_CONFIRM":
            failures.append("artifact contract mismatch")
        if manifest.get("targeted_repair") != "allow_floor_division_operator":
            failures.append("targeted repair mismatch")
        if len(progress) < 3:
            failures.append("progress writeout too sparse")
        if not any('"event": "heartbeat"' in line or '"event":"heartbeat"' in line for line in progress):
            failures.append("missing heartbeat")
        if decision.get("failure_count") != 0:
            failures.append("decision failure_count != 0")
        validation = aggregate.get("validation", {})
        adversarial = aggregate.get("adversarial", {})
        if validation.get("marker_min", 0.0) < 0.999:
            failures.append("validation marker min below 0.999")
        if validation.get("floor_marker_min", 0.0) < 1.0:
            failures.append("floor marker min below 1.0")
        if adversarial.get("action_min", 0.0) < 1.0:
            failures.append("adversarial action min below 1.0")
    summary = {
        "checker": "E82_CALC_SCRIBE_V003_FLOOR_DIVISION_CONFIRM_CHECK",
        "out": str(out),
        "failure_count": len(failures),
        "failures": failures,
        "passed": not failures,
    }
    if args.write_summary:
        (out / "checker_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
