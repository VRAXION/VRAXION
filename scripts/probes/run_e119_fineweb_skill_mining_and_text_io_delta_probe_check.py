#!/usr/bin/env python3
"""Checker for E119 FineWeb skill mining and text-IO delta probe."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


CONTRACT = "E119_FINEWEB_SKILL_MINING_AND_TEXT_IO_DELTA_PROBE"
REQUIRED = [
    "run_manifest.json",
    "dataset_report.json",
    "operator_source_report.json",
    "skill_candidate_report.json",
    "text_io_delta_report.json",
    "generation_readiness_report.json",
    "row_level_samples.jsonl",
    "skill_candidate_examples.jsonl",
    "progress.jsonl",
    "partial_aggregate_snapshot.json",
    "aggregate_metrics.json",
    "deterministic_replay.json",
    "decision.json",
    "summary.json",
    "report.md",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def deterministic_hash(payload: Any) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def check(root: Path) -> list[str]:
    failures: list[str] = []
    for name in REQUIRED:
        if not (root / name).exists():
            failures.append(f"missing artifact: {name}")
    if failures:
        return failures

    manifest = read_json(root / "run_manifest.json")
    dataset = read_json(root / "dataset_report.json")
    operators = read_json(root / "operator_source_report.json")
    skills = read_json(root / "skill_candidate_report.json")["rows"]
    text_delta = read_json(root / "text_io_delta_report.json")
    generation = read_json(root / "generation_readiness_report.json")
    aggregate = read_json(root / "aggregate_metrics.json")
    replay = read_json(root / "deterministic_replay.json")
    decision = read_json(root / "decision.json")
    summary = read_json(root / "summary.json")
    progress = [line for line in (root / "progress.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    samples = [line for line in (root / "row_level_samples.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    candidate_examples = [line for line in (root / "skill_candidate_examples.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]

    if manifest.get("artifact_contract") != CONTRACT:
        failures.append("contract mismatch")
    boundary = manifest.get("boundary", "")
    for text in ["not Gemma-style language model", "not final training", "not PermaCore", "not TrueGolden"]:
        if text not in boundary:
            failures.append(f"boundary missing: {text}")
    if manifest.get("gradient_descent_used") or manifest.get("optimizer_used") or manifest.get("backprop_used"):
        failures.append("gradient/optimizer/backprop unexpectedly enabled")
    if dataset.get("rows_seen") != aggregate.get("rows_seen"):
        failures.append("dataset/aggregate row mismatch")
    if aggregate.get("rows_seen", 0) < 10_000:
        failures.append("too few rows scanned")
    if aggregate.get("operator_count") != operators.get("operator_count"):
        failures.append("operator count mismatch")
    if aggregate.get("current_hard_negative_count") != 0:
        failures.append("current hard negatives detected")
    if aggregate.get("current_text_io_accuracy", 0.0) < aggregate.get("legacy_text_io_accuracy", 0.0):
        failures.append("current library worse than legacy")
    if aggregate.get("current_minus_legacy_delta", 0.0) < 0.20:
        failures.append("text IO delta below threshold")
    if aggregate.get("farm_candidate_count", 0) < 3:
        failures.append("too few farm candidates")
    if aggregate.get("skill_candidate_count") != len(skills):
        failures.append("skill candidate count mismatch")
    if not generation.get("grounded_canonical_text_io_ready"):
        failures.append("grounded canonical text IO not ready")
    if generation.get("freeform_gemma_style_generation_ready"):
        failures.append("unexpected Gemma/freeform generation claim")
    if not samples:
        failures.append("row-level samples missing")
    if not candidate_examples:
        failures.append("candidate examples missing")
    if len(progress) < 2 or not any('"event": "complete"' in line or '"event":"complete"' in line for line in progress):
        failures.append("progress incomplete")
    replay_payload = {
        "aggregate": {key: aggregate[key] for key in aggregate if key != "seconds"},
        "skill_rows": skills,
        "text_delta": text_delta,
        "dataset": manifest.get("dataset"),
        "contract": CONTRACT,
    }
    if replay.get("hash") != deterministic_hash(replay_payload) or not replay.get("hash_match"):
        failures.append("deterministic replay mismatch")
    if decision.get("failure_count") != 0:
        failures.append("decision failure_count nonzero")
    if summary.get("artifact_contract") != CONTRACT:
        failures.append("summary contract mismatch")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="target/pilot_wave/e119_fineweb_skill_mining_and_text_io_delta_probe")
    parser.add_argument("--write-summary", action="store_true")
    args = parser.parse_args()
    root = Path(args.out)
    failures = check(root)
    payload = {
        "artifact_contract": CONTRACT,
        "failure_count": len(failures),
        "failures": failures,
        "target_checker_passed": not failures,
    }
    if args.write_summary:
        (root / "checker_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
