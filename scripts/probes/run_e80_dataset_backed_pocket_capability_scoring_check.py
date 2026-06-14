#!/usr/bin/env python3
"""Checker for E80 dataset-backed Pocket capability scoring artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED = [
    "backend_manifest.json",
    "progress.jsonl",
    "partial_aggregate_snapshot.json",
    "system_results.json",
    "row_level_samples.jsonl",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="target/pilot_wave/e80_dataset_backed_pocket_capability_scoring")
    parser.add_argument("--write-summary", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    failures: list[str] = []

    for name in REQUIRED:
        if not (out / name).exists():
            failures.append(f"missing artifact: {name}")

    if not failures:
        manifest = read_json(out / "backend_manifest.json")
        aggregate = read_json(out / "aggregate_metrics.json")
        decision = read_json(out / "decision.json")
        systems = read_json(out / "system_results.json")
        progress_lines = [
            line for line in (out / "progress.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        sample_lines = [
            line for line in (out / "row_level_samples.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()
        ]

        if manifest.get("artifact_contract") != "E80_DATASET_BACKED_POCKET_CAPABILITY_SCORING":
            failures.append("backend_manifest artifact_contract mismatch")
        if len(manifest.get("seeds", [])) < 1:
            failures.append("no seeds recorded")
        if len(progress_lines) < 3:
            failures.append("progress writeout too sparse")
        if not any('"event": "heartbeat"' in line or '"event":"heartbeat"' in line for line in progress_lines):
            failures.append("missing heartbeat/partial writeout")
        if len(sample_lines) == 0:
            failures.append("missing row-level samples")
        if aggregate.get("bad_promotion_count") != 0:
            failures.append("bad promotion detected")
        if decision.get("failure_count") != 0:
            failures.append("decision failure_count is not zero")
        if not systems:
            failures.append("no system results")
        promoted = [name for name, item in systems.items() if item.get("promoted_candidate")]
        if not promoted:
            failures.append("no promoted dataset-backed candidate")
        for name, item in systems.items():
            rows = item.get("rows", 0)
            if rows <= 0:
                failures.append(f"system has no evaluated rows: {name}")
            if name.startswith("bad_") and item.get("promoted_candidate"):
                failures.append(f"bad control promoted: {name}")

    summary = {
        "checker": "E80_DATASET_BACKED_POCKET_CAPABILITY_SCORING_CHECK",
        "out": str(out),
        "failure_count": len(failures),
        "failures": failures,
        "passed": len(failures) == 0,
    }
    if args.write_summary:
        (out / "checker_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
    print(json.dumps(summary, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
