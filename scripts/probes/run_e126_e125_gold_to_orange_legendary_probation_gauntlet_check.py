#!/usr/bin/env python3
"""Checker for E126 E125 Gold to Orange/Legendary probation gauntlet."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


CONTRACT = "E126_E125_GOLD_TO_ORANGE_LEGENDARY_PROBATION_GAUNTLET"
ORANGE_TARGET = 300_000
EXPECTED_INPUT_COUNT = 20
REQUIRED = [
    "run_manifest.json",
    "input_gold_report.json",
    "probation_report.json",
    "operator_orange_results.json",
    "operator_cards.json",
    "variant_report.json",
    "mutation_summary.json",
    "row_level_samples.jsonl",
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
    input_report = read_json(root / "input_gold_report.json")
    probation = read_json(root / "probation_report.json")
    results = read_json(root / "operator_orange_results.json")["rows"]
    cards = read_json(root / "operator_cards.json")["rows"]
    variants = read_json(root / "variant_report.json")["rows"]
    mutation = read_json(root / "mutation_summary.json")
    aggregate = read_json(root / "aggregate_metrics.json")
    replay = read_json(root / "deterministic_replay.json")
    decision = read_json(root / "decision.json")
    summary = read_json(root / "summary.json")
    report_text = (root / "report.md").read_text(encoding="utf-8")
    progress = [line for line in (root / "progress.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    samples = [json.loads(line) for line in (root / "row_level_samples.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]

    if manifest.get("artifact_contract") != CONTRACT:
        failures.append("contract mismatch")
    boundary = manifest.get("boundary", "")
    for phrase in ["scope-limited Orange/LegendaryCandidate only", "not Core", "not PermaCore", "not TrueGolden", "not Gemma-style generation", "not final training"]:
        if phrase not in boundary:
            failures.append(f"boundary missing: {phrase}")
    forbidden_claims = ["PermaCore promotion confirmed", "TrueGolden promotion confirmed", "Core promotion confirmed", "Gemma-level"]
    for phrase in forbidden_claims:
        if phrase in report_text:
            failures.append(f"forbidden claim in report: {phrase}")
    if manifest.get("gradient_descent_used") or manifest.get("optimizer_used") or manifest.get("backprop_used"):
        failures.append("gradient/optimizer/backprop unexpectedly enabled")
    if len(input_report.get("rows", [])) != EXPECTED_INPUT_COUNT:
        failures.append("expected exactly twenty E125 Gold inputs")
    if probation.get("orange_target") != ORANGE_TARGET:
        failures.append("orange target mismatch")
    if aggregate.get("candidate_count") != EXPECTED_INPUT_COUNT:
        failures.append("candidate count mismatch")
    if aggregate.get("orange_legendary_candidate_count") != aggregate.get("candidate_count"):
        failures.append("not all candidates reached Orange/Legendary")
    for key in ["hard_negative_total", "false_commit_total", "wrong_scope_call_total", "unsupported_answer_total", "negative_transfer_total", "direct_flow_write_total"]:
        if aggregate.get(key) != 0:
            failures.append(f"{key} nonzero")
    if aggregate.get("qualified_activation_min", 0) < ORANGE_TARGET:
        failures.append("qualified activation min below Orange target")
    if aggregate.get("family_coverage_min", 0) < 12:
        failures.append("family coverage min below threshold")
    if aggregate.get("campaign_count_min", 0) < 8:
        failures.append("campaign count min below threshold")
    for key in ["reload_match_rate", "negative_scope_pass_rate", "challenger_pass_rate", "prune_pass_rate"]:
        if aggregate.get(key) != 1.0:
            failures.append(f"{key} not perfect")
    if mutation.get("accepted_mutations_total", 0) <= 0 or mutation.get("rollback_count_total", 0) <= 0 or mutation.get("prune_attempts_total", 0) <= 0:
        failures.append("mutation/prune/rollback evidence missing")
    if len(results) != EXPECTED_INPUT_COUNT or len(cards) != EXPECTED_INPUT_COUNT:
        failures.append("result/card count mismatch")
    selected = [row for row in variants if row.get("selected")]
    if len(selected) != len(results):
        failures.append("selected variant count mismatch")
    for row in results:
        oid = row.get("operator_id")
        if row.get("rank_after") != "OrangeLegendaryCandidate":
            failures.append(f"wrong rank_after for {oid}")
        if row.get("qualified_activation", 0) < ORANGE_TARGET:
            failures.append(f"under Orange target: {oid}")
        if row.get("e126_remaining_to_orange") != 0 or not row.get("e126_reaches_orange_legendary"):
            failures.append(f"Orange target not reached: {oid}")
        if row.get("hard_negative") or row.get("false_commit") or row.get("wrong_scope_call") or row.get("unsupported_answer") or row.get("direct_flow_write"):
            failures.append(f"unsafe row: {oid}")
        if not row.get("reload_shadow_pass") or not row.get("negative_scope_pass") or not row.get("challenger_pass") or not row.get("prune_pass"):
            failures.append(f"gate failed: {oid}")
        if float(row.get("selected_prune_ratio", 0.0)) < 0.60:
            failures.append(f"selected prune ratio too low: {oid}")
        reg = root / "operator_registry" / f"{oid}.json"
        if not reg.exists():
            failures.append(f"missing registry entry: {oid}")
        else:
            registry = read_json(reg)
            if registry.get("operator_id") != oid or registry.get("direct_flow_write_allowed") is not False:
                failures.append(f"registry guard mismatch: {oid}")
    if not samples:
        failures.append("missing row-level samples")
    for sample in samples:
        if sample.get("hard_negative") or sample.get("wrong_scope_call") or sample.get("false_commit") or sample.get("unsupported_answer") or sample.get("direct_flow_write"):
            failures.append(f"unsafe sample: {sample.get('sample_id')}")
    if len(progress) < 3 or not any('"event": "complete"' in line or '"event":"complete"' in line for line in progress):
        failures.append("progress incomplete")

    replay_payload = {
        "contract": CONTRACT,
        "source_e125_decision": read_json(Path(manifest["source_root"]) / "decision.json"),
        "results": results,
        "variants": variants,
        "aggregate": {key: value for key, value in aggregate.items() if key != "seconds"},
        "sample_hash": deterministic_hash(samples[:16]),
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
    parser.add_argument("--out", default="target/pilot_wave/e126_e125_gold_to_orange_legendary_probation_gauntlet")
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
