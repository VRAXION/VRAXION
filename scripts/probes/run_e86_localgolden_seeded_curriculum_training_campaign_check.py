#!/usr/bin/env python3
"""Checker for E86 LocalGolden-seeded curriculum training campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED = [
    "run_manifest.json",
    "progress.jsonl",
    "partial_aggregate_snapshot.json",
    "task_generation_report.json",
    "seed_results.json",
    "aggregate_metrics.json",
    "evolution_report.json",
    "mutation_summary.json",
    "decision.json",
    "report.md",
    "evolution_history.jsonl",
    "pocket_count_timeseries.jsonl",
    "promotion_ledger.jsonl",
    "row_level_samples.jsonl",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="target/pilot_wave/e86_localgolden_seeded_curriculum_training_campaign")
    parser.add_argument("--write-summary", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    failures: list[str] = []
    for name in REQUIRED:
        if not (out / name).exists():
            failures.append(f"missing artifact: {name}")
    if not failures:
        manifest = read_json(out / "run_manifest.json")
        generation = read_json(out / "task_generation_report.json")
        aggregate = read_json(out / "aggregate_metrics.json")
        evolution = read_json(out / "evolution_report.json")
        mutation = read_json(out / "mutation_summary.json")
        decision = read_json(out / "decision.json")
        progress = [line for line in (out / "progress.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        history = [line for line in (out / "evolution_history.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        timeseries = [line for line in (out / "pocket_count_timeseries.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        samples = [line for line in (out / "row_level_samples.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        seed_progress_dir = out / "seed_progress"
        seed_progress_files = list(seed_progress_dir.glob("seed_*.jsonl")) if seed_progress_dir.exists() else []
        if manifest.get("artifact_contract") != "E86_LOCALGOLDEN_SEEDED_CURRICULUM_TRAINING_CAMPAIGN":
            failures.append("artifact contract mismatch")
        if "not open-domain model training" not in manifest.get("boundary", ""):
            failures.append("boundary mismatch")
        if generation.get("case_count", 0) < 100_000:
            failures.append("too few generated cases")
        if len(progress) < 3:
            failures.append("progress writeout too sparse")
        if not any('"event": "heartbeat"' in line or '"event":"heartbeat"' in line for line in progress):
            failures.append("missing heartbeat")
        if not seed_progress_files:
            failures.append("missing per-seed progress files")
        if len(history) < manifest.get("generations", 0) * max(1, len(manifest.get("seeds", []))):
            failures.append("evolution history too sparse")
        if len(timeseries) != len(history):
            failures.append("pocket count timeseries/history length mismatch")
        if len(samples) < 100:
            failures.append("row-level samples too sparse")
        if decision.get("failure_count") != 0:
            failures.append("decision failure_count != 0")
        if aggregate.get("validation_action_min") != 1.0:
            failures.append("validation action min != 1.0")
        if aggregate.get("adversarial_action_min") != 1.0:
            failures.append("adversarial action min != 1.0")
        for key in [
            "validation_false_call_max",
            "adversarial_false_call_max",
            "validation_false_commit_max",
            "adversarial_false_commit_max",
        ]:
            if aggregate.get(key) != 0.0:
                failures.append(f"{key} != 0")
        if aggregate.get("local_golden_candidate_count") != len(manifest.get("seeds", [])):
            failures.append("not every seed reached local_golden_candidate")
        if aggregate.get("accepted_mutations_total", 0) <= 0:
            failures.append("no accepted mutations")
        if aggregate.get("rollback_count_total", 0) <= 0:
            failures.append("no rollback evidence")
        if mutation.get("rejected_mutations_total", 0) <= 0:
            failures.append("no rejected mutation evidence")
        if evolution.get("observed_count_max", 0) <= evolution.get("initial_pocket_count", 1):
            failures.append("pocket count never grew")
        if not evolution.get("plateau_detected"):
            failures.append("plateau not detected")
    summary = {
        "checker": "E86_LOCALGOLDEN_SEEDED_CURRICULUM_TRAINING_CAMPAIGN_CHECK",
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
