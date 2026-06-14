#!/usr/bin/env python3
"""Checker for E112 Gold to CoreMemoryCandidate prune-heavy probation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


CONTRACT = "E112_GOLD_TO_CORE_PRUNE_HEAVY_PROBATION_WAVE"
CORE_MIN = 100_000
CORE_FAMILY_MIN = 15
CORE_CAMPAIGN_MIN = 8
PRUNE_SELECTED_RATIO_MIN = 0.50

REQUIRED = [
    "run_manifest.json",
    "wave_manifest.json",
    "input_rank_report.json",
    "wave_results.json",
    "promotion_report.json",
    "operator_stats.json",
    "mutation_variant_report.json",
    "mutation_events.json",
    "mutation_summary.json",
    "duration_report.json",
    "progress.jsonl",
    "partial_aggregate_snapshot.json",
    "aggregate_metrics.json",
    "deterministic_replay.json",
    "decision.json",
    "summary.json",
    "report.md",
    "row_level_samples.jsonl",
]

SAMPLE_REQUIRED = [
    "sample_manifest.json",
    "wave_manifest.json",
    "input_rank_report.json",
    "wave_results.json",
    "promotion_report.json",
    "operator_stats.json",
    "mutation_variant_report.json",
    "mutation_summary.json",
    "duration_report.json",
    "aggregate_metrics.json",
    "deterministic_replay.json",
    "decision.json",
    "summary.json",
    "row_level_samples.jsonl",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def deterministic_hash(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def check_common(root: Path, sample_only: bool) -> list[str]:
    failures: list[str] = []
    for name in SAMPLE_REQUIRED if sample_only else REQUIRED:
        if not (root / name).exists():
            failures.append(f"missing artifact: {name}")
    if failures:
        return failures

    wave_manifest = read_json(root / "wave_manifest.json")
    input_report = read_json(root / "input_rank_report.json")
    wave_results = read_json(root / "wave_results.json")
    promotion = read_json(root / "promotion_report.json")
    stats = read_json(root / "operator_stats.json")
    variants = read_json(root / "mutation_variant_report.json")
    mutation_summary = read_json(root / "mutation_summary.json")
    duration = read_json(root / "duration_report.json")
    aggregate = read_json(root / "aggregate_metrics.json")
    replay = read_json(root / "deterministic_replay.json")
    decision = read_json(root / "decision.json")
    summary = read_json(root / "summary.json")

    if sample_only:
        manifest = read_json(root / "sample_manifest.json")
        if manifest.get("artifact_contract") != CONTRACT:
            failures.append("sample contract mismatch")
        samples = [line for line in (root / "row_level_samples.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(samples) < 256:
            failures.append("sample rows too sparse")
    else:
        manifest = read_json(root / "run_manifest.json")
        if manifest.get("artifact_contract") != CONTRACT:
            failures.append("artifact contract mismatch")
        boundary = manifest.get("boundary", "")
        for text in ["not PermaCore", "not TrueGolden", "not final training"]:
            if text not in boundary:
                failures.append(f"boundary caveat missing: {text}")
        if manifest.get("gradient_descent_used") or manifest.get("optimizer_used") or manifest.get("backprop_used"):
            failures.append("gradient/optimizer/backprop unexpectedly enabled")
        progress = [line for line in (root / "progress.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(progress) < 4 or not any('"event": "heartbeat"' in line or '"event":"heartbeat"' in line for line in progress):
            failures.append("missing/sparse progress heartbeat")

    if wave_manifest.get("artifact_contract") != CONTRACT:
        failures.append("wave contract mismatch")
    if wave_manifest.get("candidate_source_rank") != "Gold" or wave_manifest.get("target_rank") != "CoreMemoryCandidate":
        failures.append("wave source/target rank mismatch")
    boundary = wave_manifest.get("boundary", "")
    for text in ["not PermaCore", "not TrueGolden", "not final training"]:
        if text not in boundary:
            failures.append(f"wave boundary missing: {text}")
    if wave_manifest.get("prune_selected_ratio_min") != PRUNE_SELECTED_RATIO_MIN:
        failures.append("prune ratio gate mismatch")

    if input_report.get("e109_decision") != "e109_rank_ladder_and_golden_watch_confirmed":
        failures.append("E109 input decision mismatch")
    if input_report.get("e110_decision") != "e110_wave1_silver_to_gold_pressure_confirmed":
        failures.append("E110 input decision mismatch")
    if input_report.get("e111_decision") != "e111_bronze_mutation_prune_wave_gold_conversion_confirmed":
        failures.append("E111 input decision mismatch")
    if input_report.get("merged_gold_candidates") != aggregate.get("candidate_count"):
        failures.append("merged Gold candidate count mismatch")

    rows = wave_results.get("rows", [])
    stat_rows = stats.get("rows", [])
    variant_rows = variants.get("rows", [])
    if len(rows) != len(stat_rows) or len(rows) != aggregate.get("candidate_count"):
        failures.append("wave row count mismatch")
    if len(variant_rows) < len(rows) * 5:
        failures.append("variant rows too sparse")
    if aggregate.get("candidate_count", 0) <= 0:
        failures.append("no candidates")
    if aggregate.get("core_memory_candidate_count") != aggregate.get("candidate_count"):
        failures.append("not all candidates reached CoreMemoryCandidate")
    if aggregate.get("prune_heavy_selected_ratio", 0.0) < PRUNE_SELECTED_RATIO_MIN:
        failures.append("prune-heavy selected ratio below gate")
    if aggregate.get("hard_negative_total") != 0:
        failures.append("hard negative total nonzero")
    for key in ["wrong_scope_call_rate", "false_commit_rate", "unsupported_answer_rate", "negative_transfer_rate"]:
        if aggregate.get(key) != 0.0:
            failures.append(f"{key} nonzero")
    if aggregate.get("reload_match_rate") != 1.0:
        failures.append("reload match rate not clean")
    if aggregate.get("long_horizon_no_harm_rate") != 1.0 or aggregate.get("negative_scope_pass_rate") != 1.0:
        failures.append("no-harm/negative-scope rate not clean")
    if aggregate.get("qualified_activation_after_min", 0) < CORE_MIN:
        failures.append("Core activation threshold not met")
    if aggregate.get("family_coverage_after_min", 0) < CORE_FAMILY_MIN:
        failures.append("Core family coverage threshold not met")
    if aggregate.get("campaign_count_after_min", 0) < CORE_CAMPAIGN_MIN:
        failures.append("Core campaign threshold not met")
    if aggregate.get("mutation_attempts_total", 0) <= 0 or aggregate.get("prune_attempts_total", 0) <= 0:
        failures.append("mutation/prune attempts missing")
    if aggregate.get("rollback_count_total") != aggregate.get("rejected_mutations_total"):
        failures.append("rollback/rejected mismatch")
    if duration.get("measured_wall_seconds", 0) < 0:
        failures.append("invalid duration")
    if mutation_summary.get("prune_heavy_selected_ratio", 0) < PRUNE_SELECTED_RATIO_MIN:
        failures.append("mutation summary prune ratio below gate")

    selected_variants = promotion.get("selected_variants", {})
    if len(selected_variants) != len(rows):
        failures.append("selected variant count mismatch")
    if len(promotion.get("core_memory_candidates", [])) != aggregate.get("core_memory_candidate_count"):
        failures.append("promotion count mismatch")

    variant_by_id = {row.get("variant_id"): row for row in variant_rows}
    for row in rows:
        operator_id = row.get("operator_id")
        if row.get("rank_before") != "Gold":
            failures.append(f"non-Gold candidate in wave: {operator_id}")
        if row.get("rank_after") != "CoreMemoryCandidate":
            failures.append(f"row not CoreMemoryCandidate: {operator_id}")
        if row.get("rank_after") in {"PermaCore", "TrueGolden"}:
            failures.append(f"forbidden PermaCore/TrueGolden row: {operator_id}")
        if row.get("qualified_activation", 0) < CORE_MIN:
            failures.append(f"row below Core activation threshold: {operator_id}")
        if row.get("combined_family_coverage", 0) < CORE_FAMILY_MIN:
            failures.append(f"row below Core family coverage: {operator_id}")
        if row.get("campaign_count", 0) < CORE_CAMPAIGN_MIN:
            failures.append(f"row below Core campaign threshold: {operator_id}")
        if row.get("selected_prune_ratio", 0.0) < 0.50:
            failures.append(f"selected prune ratio below 50%: {operator_id}")
        if not row.get("reload_shadow_pass") or not row.get("challenger_pass") or not row.get("prune_pass"):
            failures.append(f"row missing reload/challenger/prune pass: {operator_id}")
        if not row.get("long_horizon_no_harm_pass") or not row.get("negative_scope_pass"):
            failures.append(f"row missing long-horizon/no-harm pass: {operator_id}")
        selected_id = row.get("selected_variant_id")
        variant = variant_by_id.get(selected_id)
        if not variant or not variant.get("selected"):
            failures.append(f"selected variant missing/not marked: {operator_id}")
        if row.get("mutation_attempts", 0) <= 0 or row.get("prune_attempts", 0) <= 0:
            failures.append(f"missing mutation/prune budget: {operator_id}")
        if row.get("rollback_count") != row.get("rejected_mutations"):
            failures.append(f"row rollback/rejected mismatch: {operator_id}")

    replay_payload = {
        "aggregate": {key: aggregate[key] for key in aggregate if key != "seconds"},
        "wave_results": rows,
        "promotion_report": promotion,
        "mutation_summary": mutation_summary,
        "duration": {key: value for key, value in duration.items() if key != "measured_wall_seconds"},
        "input_report": input_report,
        "wave_manifest": wave_manifest,
    }
    if replay.get("hash") != deterministic_hash(replay_payload):
        failures.append("deterministic replay hash mismatch")
    if decision.get("failure_count") != 0 or decision.get("decision") != "e112_gold_to_core_prune_heavy_probation_confirmed":
        failures.append("decision not confirmed")
    if summary.get("artifact_contract") != CONTRACT:
        failures.append("summary contract mismatch")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="target/pilot_wave/e112_gold_to_core_prune_heavy_probation_wave")
    parser.add_argument("--sample-only")
    parser.add_argument("--write-summary", action="store_true")
    args = parser.parse_args()
    root = Path(args.sample_only) if args.sample_only else Path(args.out)
    failures = check_common(root, sample_only=bool(args.sample_only))
    summary = {
        "checker": "E112_GOLD_TO_CORE_PRUNE_HEAVY_PROBATION_WAVE_CHECK",
        "root": str(root),
        "sample_only": bool(args.sample_only),
        "failure_count": len(failures),
        "failures": failures,
        "passed": not failures,
    }
    if args.write_summary and not args.sample_only:
        (Path(args.out) / "checker_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
