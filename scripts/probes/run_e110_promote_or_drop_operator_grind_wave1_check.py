#!/usr/bin/env python3
"""Checker for E110 Wave 1 promote-or-drop artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


CONTRACT = "E110_PROMOTE_OR_DROP_OPERATOR_GRIND_WAVE1"

REQUIRED = [
    "run_manifest.json",
    "wave_manifest.json",
    "input_rank_report.json",
    "wave_results.json",
    "promotion_report.json",
    "operator_stats.json",
    "challenger_prune_report.json",
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
    "challenger_prune_report.json",
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
    challenger = read_json(root / "challenger_prune_report.json")
    aggregate = read_json(root / "aggregate_metrics.json")
    replay = read_json(root / "deterministic_replay.json")
    decision = read_json(root / "decision.json")
    summary = read_json(root / "summary.json")

    if sample_only:
        manifest = read_json(root / "sample_manifest.json")
        if manifest.get("artifact_contract") != CONTRACT:
            failures.append("sample contract mismatch")
        samples = [line for line in (root / "row_level_samples.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(samples) < 64:
            failures.append("sample rows too sparse")
    else:
        manifest = read_json(root / "run_manifest.json")
        if manifest.get("artifact_contract") != CONTRACT:
            failures.append("artifact contract mismatch")
        boundary = manifest.get("boundary", "")
        for text in ["not Diamond promotion", "not Core promotion", "not final training"]:
            if text not in boundary:
                failures.append(f"boundary caveat missing: {text}")
        if manifest.get("gradient_descent_used") or manifest.get("optimizer_used") or manifest.get("backprop_used"):
            failures.append("gradient/optimizer/backprop unexpectedly enabled")
        progress = [line for line in (root / "progress.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(progress) < 3 or not any('"event": "heartbeat"' in line or '"event":"heartbeat"' in line for line in progress):
            failures.append("missing/sparse progress heartbeat")

    if wave_manifest.get("artifact_contract") != CONTRACT:
        failures.append("wave contract mismatch")
    if wave_manifest.get("candidate_source_rank") != "Silver" or wave_manifest.get("target_rank") != "Gold":
        failures.append("wave source/target rank mismatch")
    if "not Diamond/Core/PermaCore promotion" not in wave_manifest.get("boundary", ""):
        failures.append("wave boundary missing")
    if input_report.get("e109_decision") != "e109_rank_ladder_and_golden_watch_confirmed":
        failures.append("E109 input decision mismatch")
    if input_report.get("wave1_silver_candidates") != aggregate.get("candidate_count"):
        failures.append("Silver candidate count mismatch")
    rows = wave_results.get("rows", [])
    stat_rows = stats.get("rows", [])
    if len(rows) != len(stat_rows) or len(rows) != aggregate.get("candidate_count"):
        failures.append("wave row count mismatch")
    if aggregate.get("candidate_count", 0) <= 0:
        failures.append("no candidates")
    if aggregate.get("hard_negative_total") != 0:
        failures.append("hard negative total nonzero")
    for key in ["wrong_scope_call_rate", "false_commit_rate", "unsupported_answer_rate", "negative_transfer_rate"]:
        if aggregate.get(key) != 0.0:
            failures.append(f"{key} nonzero")
    if aggregate.get("neutral_waste_over_threshold_count") != 0:
        failures.append("neutral waste threshold exceeded")
    if aggregate.get("challenger_replacement_count") != 0 or aggregate.get("pruned_variant_replacement_count") != 0:
        failures.append("replacement unexpectedly won")
    if aggregate.get("reload_match_rate") != 1.0:
        failures.append("reload match rate not clean")
    promoted = promotion.get("promoted_to_gold", [])
    if len(promoted) != aggregate.get("promoted_to_gold_count"):
        failures.append("promotion count mismatch")
    if aggregate.get("promoted_to_gold_count", 0) <= 0:
        failures.append("nothing promoted")

    for row in rows:
        if row.get("rank_before") != "Silver":
            failures.append(f"non-Silver candidate in wave: {row.get('operator_id')}")
        if row.get("hard_negative_add") != 0 or row.get("hard_negative") != 0:
            failures.append(f"hard negative row: {row.get('operator_id')}")
        if row.get("wave1_outcome") == "PromotedToGold":
            if row.get("rank_after") != "Gold":
                failures.append(f"promoted row not Gold: {row.get('operator_id')}")
            if row.get("qualified_activation", 0) < 3000:
                failures.append(f"Gold below activation threshold: {row.get('operator_id')}")
            if row.get("combined_family_coverage", 0) < 5:
                failures.append(f"Gold below coverage threshold: {row.get('operator_id')}")
            if row.get("campaign_count", 0) < 3:
                failures.append(f"Gold below campaign threshold: {row.get('operator_id')}")
            if not row.get("reload_shadow_pass") or not row.get("challenger_pass") or not row.get("prune_pass"):
                failures.append(f"Gold missing reload/challenger/prune pass: {row.get('operator_id')}")
        if row.get("rank_after") in {"DiamondCandidate", "CoreMemoryCandidate", "PermaCore"}:
            failures.append(f"forbidden higher rank in E110 wave1: {row.get('operator_id')}")

    replay_payload = {
        "aggregate": {key: aggregate[key] for key in aggregate if key != "seconds"},
        "wave_results": rows,
        "promotion_report": promotion,
        "challenger": challenger,
        "input_report": input_report,
        "wave_manifest": wave_manifest,
    }
    if replay.get("hash") != deterministic_hash(replay_payload):
        failures.append("deterministic replay hash mismatch")
    if decision.get("failure_count") != 0 or decision.get("decision") != "e110_wave1_silver_to_gold_pressure_confirmed":
        failures.append("decision not confirmed")
    if summary.get("artifact_contract") != CONTRACT:
        failures.append("summary contract mismatch")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="target/pilot_wave/e110_promote_or_drop_operator_grind_wave1")
    parser.add_argument("--sample-only")
    parser.add_argument("--write-summary", action="store_true")
    args = parser.parse_args()
    root = Path(args.sample_only) if args.sample_only else Path(args.out)
    failures = check_common(root, sample_only=bool(args.sample_only))
    summary = {
        "checker": "E110_PROMOTE_OR_DROP_OPERATOR_GRIND_WAVE1_CHECK",
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
