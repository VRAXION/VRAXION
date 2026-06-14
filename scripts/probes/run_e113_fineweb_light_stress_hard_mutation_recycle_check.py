#!/usr/bin/env python3
"""Checker for E113 FineWeb light stress hard mutation recycle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


CONTRACT = "E113_FINEWEB_LIGHT_STRESS_HARD_MUTATION_RECYCLE"
REQUIRED = [
    "run_manifest.json",
    "dataset_report.json",
    "operator_stress_results.json",
    "mutation_variant_report.json",
    "mutation_events.jsonl",
    "mutation_summary.json",
    "progress.jsonl",
    "partial_aggregate_snapshot.json",
    "aggregate_metrics.json",
    "deterministic_replay.json",
    "decision.json",
    "summary.json",
    "report.md",
    "row_level_samples.jsonl",
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
    operators = read_json(root / "operator_stress_results.json")["rows"]
    variants = read_json(root / "mutation_variant_report.json")["rows"]
    mutation = read_json(root / "mutation_summary.json")
    aggregate = read_json(root / "aggregate_metrics.json")
    replay = read_json(root / "deterministic_replay.json")
    decision = read_json(root / "decision.json")
    summary = read_json(root / "summary.json")
    progress = [line for line in (root / "progress.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    samples = [line for line in (root / "row_level_samples.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]

    if manifest.get("artifact_contract") != CONTRACT:
        failures.append("contract mismatch")
    boundary = manifest.get("boundary", "")
    for text in ["not PermaCore", "not TrueGolden", "not final training"]:
        if text not in boundary:
            failures.append(f"boundary missing: {text}")
    if manifest.get("gradient_descent_used") or manifest.get("optimizer_used") or manifest.get("backprop_used"):
        failures.append("gradient/optimizer/backprop unexpectedly enabled")
    if aggregate.get("rows_seen", 0) < 10_000:
        failures.append("too few FineWeb rows scanned")
    if aggregate.get("rows_seen") != dataset.get("rows_seen"):
        failures.append("dataset row count mismatch")
    if aggregate.get("operator_count") != len(operators):
        failures.append("operator count mismatch")
    if aggregate.get("variant_count") != len(variants):
        failures.append("variant count mismatch")
    if len(variants) < len(operators) * 4:
        failures.append("variant coverage too sparse")
    if aggregate.get("selected_hard_negative_total") != 0:
        failures.append("selected hard negatives remain")
    if aggregate.get("accepted_mutations_total", 0) <= 0:
        failures.append("no accepted mutation/recycle copies")
    if aggregate.get("rollback_count_total", 0) <= 0:
        failures.append("no rollback pressure recorded")
    if mutation.get("recycled_operator_count") != aggregate.get("recycled_operator_count"):
        failures.append("recycled operator count mismatch")
    if not samples:
        failures.append("row-level samples missing")
    if len(progress) < 2 or not any('"event": "complete"' in line or '"event":"complete"' in line for line in progress):
        failures.append("progress incomplete")
    selected = [row for row in variants if row.get("selected")]
    if len(selected) != len(operators):
        failures.append("selected variant count mismatch")
    for row in operators:
        if row.get("rank_before") != "CoreMemoryCandidate":
            failures.append(f"non-CoreMemoryCandidate input: {row.get('operator_id')}")
        if row.get("selected_hard_negative") != 0:
            failures.append(f"selected hard negative: {row.get('operator_id')}")
    replay_payload = {
        "aggregate": {key: aggregate[key] for key in aggregate if key != "seconds"},
        "operators": operators,
        "variants": variants,
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
    parser.add_argument("--out", default="target/pilot_wave/e113_fineweb_light_stress_hard_mutation_recycle")
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
