#!/usr/bin/env python3
"""Checker for E109 rank ladder and GoldenWatch probation artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


CONTRACT = "E109_OPERATOR_RANK_LADDER_AND_GOLDEN_WATCH_PROBATION_MODE"

REQUIRED = [
    "run_manifest.json",
    "rank_policy_manifest.json",
    "input_artifact_report.json",
    "qualified_activation_ledger.json",
    "rank_results.json",
    "golden_watch_report.json",
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
    "rank_policy_manifest.json",
    "input_artifact_report.json",
    "qualified_activation_ledger.json",
    "rank_results.json",
    "golden_watch_report.json",
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

    policy = read_json(root / "rank_policy_manifest.json")
    input_report = read_json(root / "input_artifact_report.json")
    ledger = read_json(root / "qualified_activation_ledger.json")
    rank_results = read_json(root / "rank_results.json")
    watch = read_json(root / "golden_watch_report.json")
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
        if len(samples) < 100:
            failures.append("sample rows too sparse")
    else:
        manifest = read_json(root / "run_manifest.json")
        if manifest.get("artifact_contract") != CONTRACT:
            failures.append("artifact contract mismatch")
        boundary = manifest.get("boundary", "")
        for text in ["not Core promotion", "not TrueGolden promotion", "not final training"]:
            if text not in boundary:
                failures.append(f"boundary caveat missing: {text}")
        if manifest.get("gradient_descent_used") or manifest.get("optimizer_used") or manifest.get("backprop_used"):
            failures.append("gradient/optimizer/backprop unexpectedly enabled")
        progress = [line for line in (root / "progress.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(progress) < 3 or not any('"event": "heartbeat"' in line or '"event":"heartbeat"' in line for line in progress):
            failures.append("missing/sparse progress heartbeat")

    if policy.get("artifact_contract") != CONTRACT:
        failures.append("policy contract mismatch")
    if policy.get("rank_scope_required") is not True:
        failures.append("rank scope requirement not set")
    if policy.get("hard_negative_stops_promotion") is not True:
        failures.append("hard negative stop not set")
    if policy.get("silver", {}).get("min_qualified_activation") != 300:
        failures.append("Silver threshold mismatch")
    if policy.get("gold", {}).get("min_qualified_activation") != 3000:
        failures.append("Gold threshold mismatch")
    if policy.get("diamond", {}).get("min_qualified_activation") != 30000:
        failures.append("Diamond threshold mismatch")

    rows = ledger.get("rows", [])
    rank_rows = rank_results.get("rows", [])
    if len(rows) != len(rank_rows) or len(rows) != aggregate.get("operator_count"):
        failures.append("rank row count mismatch")
    if input_report.get("e108_decision") != "e108_external_transfer_no_harm_positive":
        failures.append("E108 positive input missing")
    if input_report.get("e108_negative_transfer_rate") != 0.0 or input_report.get("e108_no_harm_rate") != 1.0:
        failures.append("E108 no-harm input mismatch")

    hard_negative_rows = [row for row in rows if row.get("hard_negative", 0) > 0]
    if hard_negative_rows:
        failures.append("hard negative rows present")
    if aggregate.get("hard_negative_total") != 0 or aggregate.get("hard_negative_freeze_count") != 0:
        failures.append("hard negative aggregate nonzero")
    if aggregate.get("gold_count", 0) <= 0:
        failures.append("no Gold rank produced")
    if aggregate.get("silver_count", 0) <= 0:
        failures.append("no Silver rank produced")
    if aggregate.get("diamond_candidate_count") != 0:
        failures.append("DiamondCandidate should not be produced in E109")
    if aggregate.get("challenger_replacement_count") != 0 or aggregate.get("pruned_variant_replacement_count") != 0:
        failures.append("challenger/pruned replacement unexpectedly won")
    if watch.get("gold_watch_pass_count") != aggregate.get("gold_count"):
        failures.append("GoldWatch count mismatch")
    if watch.get("silver_watch_pass_count") != aggregate.get("silver_count"):
        failures.append("SilverWatch count mismatch")
    if challenger.get("challenger_replacement_count") != 0:
        failures.append("challenger replacement count nonzero")

    for row in rows:
        rank = row.get("rank")
        qa = int(row.get("qualified_activation", 0))
        if not row.get("scope"):
            failures.append(f"missing scope: {row.get('operator_id')}")
        if rank == "Silver" and qa < 300:
            failures.append(f"Silver below threshold: {row.get('operator_id')}")
        if rank == "Gold":
            if qa < 3000:
                failures.append(f"Gold below threshold: {row.get('operator_id')}")
            if row.get("combined_family_coverage", 0) < 5:
                failures.append(f"Gold family coverage below threshold: {row.get('operator_id')}")
            if row.get("campaign_count", 0) < 3:
                failures.append(f"Gold campaign count below threshold: {row.get('operator_id')}")
            if not row.get("challenger_pass") or not row.get("prune_pass") or not row.get("reload_shadow_pass"):
                failures.append(f"Gold missing challenger/prune/reload pass: {row.get('operator_id')}")
        if rank == "DiamondCandidate":
            failures.append(f"unexpected DiamondCandidate row: {row.get('operator_id')}")

    replay_payload = {
        "aggregate": {key: aggregate[key] for key in aggregate if key != "seconds"},
        "rank_results": rank_rows,
        "watch": watch,
        "challenger": challenger,
        "policy": policy,
        "input_report": input_report,
    }
    if replay.get("hash") != deterministic_hash(replay_payload):
        failures.append("deterministic replay hash mismatch")
    if decision.get("failure_count") != 0 or decision.get("decision") != "e109_rank_ladder_and_golden_watch_confirmed":
        failures.append("decision not confirmed")
    if summary.get("artifact_contract") != CONTRACT:
        failures.append("summary contract mismatch")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="target/pilot_wave/e109_operator_rank_ladder_and_golden_watch_probation_mode")
    parser.add_argument("--sample-only")
    parser.add_argument("--write-summary", action="store_true")
    args = parser.parse_args()
    root = Path(args.sample_only) if args.sample_only else Path(args.out)
    failures = check_common(root, sample_only=bool(args.sample_only))
    summary = {
        "checker": "E109_OPERATOR_RANK_LADDER_AND_GOLDEN_WATCH_PROBATION_MODE_CHECK",
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
