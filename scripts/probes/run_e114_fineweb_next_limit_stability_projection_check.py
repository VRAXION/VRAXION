#!/usr/bin/env python3
"""Checker for E114 FineWeb next-limit stability/projection stress."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


CONTRACT = "E114_FINEWEB_NEXT_LIMIT_STABILITY_PROJECTION"
REQUIRED = [
    "run_manifest.json",
    "source_inventory.json",
    "operator_projection_report.json",
    "stability_trend_report.json",
    "target_sufficiency_report.json",
    "chunk_trend.jsonl",
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
    inventory = read_json(root / "source_inventory.json")
    operators = read_json(root / "operator_projection_report.json")["rows"]
    trend = read_json(root / "stability_trend_report.json")
    target = read_json(root / "target_sufficiency_report.json")
    aggregate = read_json(root / "aggregate_metrics.json")
    replay = read_json(root / "deterministic_replay.json")
    decision = read_json(root / "decision.json")
    summary = read_json(root / "summary.json")
    chunks = [json.loads(line) for line in (root / "chunk_trend.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    progress = [line for line in (root / "progress.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]

    if manifest.get("artifact_contract") != CONTRACT:
        failures.append("contract mismatch")
    boundary = manifest.get("boundary", "")
    for text in ["not PermaCore", "not TrueGolden", "not final training"]:
        if text not in boundary:
            failures.append(f"boundary missing: {text}")
    if manifest.get("gradient_descent_used") or manifest.get("optimizer_used") or manifest.get("backprop_used"):
        failures.append("gradient/optimizer/backprop unexpectedly enabled")
    if inventory.get("total_rows", 0) < aggregate.get("rows_seen_source", 0):
        failures.append("source inventory smaller than rows_seen_source")
    if aggregate.get("rows_kept", 0) < 100_000:
        failures.append("too few rows kept for E114")
    if aggregate.get("operator_count") != len(operators):
        failures.append("operator count mismatch")
    if aggregate.get("selected_hard_negative_total") != 0:
        failures.append("selected hard negatives detected")
    if aggregate.get("selected_neutral_waste_total") != 0:
        failures.append("selected neutral waste detected")
    if aggregate.get("selected_call_total", 0) <= 0:
        failures.append("no selected calls")
    if not chunks or len(chunks) != trend.get("chunk_count"):
        failures.append("chunk trend mismatch")
    if trend.get("degradation_detected"):
        failures.append("degradation detected")
    if aggregate.get("chunk_count") != trend.get("chunk_count"):
        failures.append("aggregate/trend chunk count mismatch")
    if target.get("projected_reach_permacore_count") != aggregate.get("projected_reach_permacore_count"):
        failures.append("target projection mismatch")
    if len(progress) < 2 or not any('"event": "complete"' in line or '"event":"complete"' in line for line in progress):
        failures.append("progress incomplete")
    for row in operators:
        if row.get("current_run_hard_negative") != 0:
            failures.append(f"operator hard negative: {row.get('operator_id')}")
        if row.get("current_run_neutral_waste") != 0:
            failures.append(f"operator neutral waste: {row.get('operator_id')}")
        if row.get("projected_activation_after_full_fineweb", 0) < 0:
            failures.append(f"invalid projection: {row.get('operator_id')}")
    replay_payload = {
        "aggregate": {key: aggregate[key] for key in aggregate if key != "seconds"},
        "operators": operators,
        "chunks": chunks,
        "source": manifest.get("source"),
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
    parser.add_argument("--out", default="target/pilot_wave/e114_fineweb_next_limit_stability_projection")
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
