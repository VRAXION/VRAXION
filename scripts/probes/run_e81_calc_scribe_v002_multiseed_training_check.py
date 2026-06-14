#!/usr/bin/env python3
"""Checker for E81 CALC-SCRIBE v002 multi-seed training artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED = [
    "training_manifest.json",
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
    parser.add_argument("--out", default="target/pilot_wave/e81_calc_scribe_v002_multiseed_training")
    parser.add_argument("--write-summary", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    failures: list[str] = []

    for name in REQUIRED:
        if not (out / name).exists():
            failures.append(f"missing artifact: {name}")

    if not failures:
        manifest = read_json(out / "training_manifest.json")
        aggregate = read_json(out / "aggregate_metrics.json")
        decision = read_json(out / "decision.json")
        seed_results = read_json(out / "seed_results.json")
        progress_lines = [line for line in (out / "progress.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        seed_count = len(seed_results.get("seeds", []))

        if manifest.get("artifact_contract") != "E81_CALC_SCRIBE_V002_MULTISEED_TRAINING":
            failures.append("artifact contract mismatch")
        if seed_count < 1:
            failures.append("no seed results")
        if len(progress_lines) < 3:
            failures.append("progress writeout too sparse")
        if not any('"event": "heartbeat"' in line or '"event":"heartbeat"' in line for line in progress_lines):
            failures.append("missing heartbeat")
        if decision.get("failure_count") != 0:
            failures.append("decision failure_count != 0")
        if aggregate.get("action_accuracy_min", {}).get("adversarial", 0.0) < 1.0:
            failures.append("adversarial action minimum below 1.0")
        if aggregate.get("marker_validation_min", {}).get("validation", 0.0) < 0.99:
            failures.append("validation marker minimum below 0.99")
        for result in seed_results.get("seeds", []):
            genome = result.get("best_genome", {})
            if not genome.get("allow_multi_operator_ast"):
                failures.append(f"seed {result.get('seed')} best genome lacks multi-operator AST")

    summary = {
        "checker": "E81_CALC_SCRIBE_V002_MULTISEED_TRAINING_CHECK",
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
