#!/usr/bin/env python3
"""Checker for E124 text-understanding mass mutation farm capacity probe."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


CONTRACT = "E124_TEXT_UNDERSTANDING_MASS_MUTATION_FARM_CAPACITY_PROBE"
REQUIRED = [
    "run_manifest.json",
    "candidate_pool_report.json",
    "mass_mutation_lane_report.json",
    "operator_cards.json",
    "operator_gold_results.json",
    "variant_report.json",
    "batch_capacity_report.json",
    "negative_card_interaction_report.json",
    "mutation_summary.json",
    "row_level_samples.jsonl",
    "candidate_examples.jsonl",
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
    candidates = read_json(root / "candidate_pool_report.json")["rows"]
    lanes = read_json(root / "mass_mutation_lane_report.json")
    cards = read_json(root / "operator_cards.json")["rows"]
    results = read_json(root / "operator_gold_results.json")["rows"]
    variants = read_json(root / "variant_report.json")["rows"]
    batches = read_json(root / "batch_capacity_report.json")["rows"]
    interactions = read_json(root / "negative_card_interaction_report.json")["rows"]
    mutation = read_json(root / "mutation_summary.json")
    aggregate = read_json(root / "aggregate_metrics.json")
    replay = read_json(root / "deterministic_replay.json")
    decision = read_json(root / "decision.json")
    summary = read_json(root / "summary.json")
    progress = [line for line in (root / "progress.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    samples = [line for line in (root / "row_level_samples.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    examples = [line for line in (root / "candidate_examples.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    report_text = (root / "report.md").read_text(encoding="utf-8")

    if manifest.get("artifact_contract") != CONTRACT:
        failures.append("contract mismatch")
    boundary = manifest.get("boundary", "")
    for text in ["scoped mass Operator farming only", "not Core", "not PermaCore", "not TrueGolden", "not final training", "not Gemma-style generation"]:
        if text not in boundary:
            failures.append(f"boundary missing: {text}")
    for phrase in ["PermaCore promotion confirmed", "TrueGolden promotion confirmed", "Core promotion confirmed", "Gemma-level"]:
        if phrase in report_text:
            failures.append(f"forbidden claim in report: {phrase}")
    if manifest.get("gradient_descent_used") or manifest.get("optimizer_used") or manifest.get("backprop_used"):
        failures.append("gradient/optimizer/backprop unexpectedly enabled")
    if aggregate.get("rows_seen", 0) < 10_000:
        failures.append("too few rows scanned")
    if aggregate.get("active_operator_count") != 144 or not aggregate.get("orange_only_confirmed"):
        failures.append("E122 orange-only baseline not confirmed")
    if aggregate.get("negative_card_count", 0) <= 0:
        failures.append("negative cards missing")
    if aggregate.get("candidate_pool_count") != len(candidates):
        failures.append("candidate pool count mismatch")
    if aggregate.get("selected_candidate_count") != len(cards) or len(cards) != len(results):
        failures.append("selected cards/results count mismatch")
    if aggregate.get("mutation_lane_count") != lanes.get("lane_count"):
        failures.append("mutation lane count mismatch")
    if lanes.get("lane_count", 0) < 4:
        failures.append("not enough parallel mutation lanes")
    if aggregate.get("promoted_to_gold_count") != len([row for row in results if row.get("rank_after") == "Gold"]):
        failures.append("gold count mismatch")
    if aggregate.get("promoted_to_gold_count", 0) < 2:
        failures.append("too few Gold promotions for mass farm")
    for key in ["hard_negative_total", "wrong_scope_call_total", "false_commit_total", "unsupported_answer_total", "negative_transfer_total"]:
        if aggregate.get(key) != 0:
            failures.append(f"{key} nonzero")
    if aggregate.get("negative_card_false_block_count") != 0 or aggregate.get("normal_router_callable_cards") != 0:
        failures.append("negative card false block or router leak")
    if aggregate.get("negative_card_blocked_variant_count", 0) <= 0:
        failures.append("negative cards blocked no unsafe variants")
    if mutation.get("accepted_mutations_total", 0) <= 0 or mutation.get("rollback_count_total", 0) <= 0:
        failures.append("mutation/rollback evidence missing")
    if len(variants) < len(results) * 4:
        failures.append("not enough variant lanes per candidate")
    selected = [row for row in variants if row.get("selected")]
    if len(selected) != len(results):
        failures.append("selected variant count mismatch")
    if not batches:
        failures.append("batch capacity rows missing")
    if aggregate.get("max_clean_batch_size", 0) <= 0:
        failures.append("no clean batch capacity detected")
    for row in batches:
        if row.get("hard_negative_total") or row.get("false_commit_total"):
            failures.append(f"unsafe batch row: {row.get('batch_size')}")
    for row in results:
        oid = row.get("operator_id")
        if row.get("rank_after") == "Gold":
            if row.get("qualified_activation", 0) < 3000:
                failures.append(f"Gold under activation threshold: {oid}")
            if row.get("family_coverage", 0) < 5:
                failures.append(f"Gold under family coverage threshold: {oid}")
            if row.get("campaign_count", 0) < 3:
                failures.append(f"Gold under campaign threshold: {oid}")
            if not row.get("reload_shadow_pass") or not row.get("negative_scope_pass") or not row.get("challenger_pass") or not row.get("prune_pass"):
                failures.append(f"Gold gate failed: {oid}")
            registry = root / "operator_registry" / f"{oid}.json"
            if not registry.exists():
                failures.append(f"missing registry entry: {oid}")
            else:
                payload = read_json(registry)
                if payload.get("direct_flow_write_allowed") is not False:
                    failures.append(f"registry direct write allowed: {oid}")
        if row.get("rank_after") in {"Core", "PermaCore", "TrueGolden"}:
            failures.append(f"invalid overpromotion: {oid}")
    if len(interactions) != len(results):
        failures.append("negative-card interaction count mismatch")
    for row in interactions:
        if row.get("false_block_count") or row.get("normal_router_callable_cards"):
            failures.append(f"bad negative-card interaction: {row.get('candidate_id')}")
    if not samples:
        failures.append("row-level samples missing")
    if not examples:
        failures.append("candidate examples missing")
    if len(progress) < 3 or not any('"event": "complete"' in line or '"event":"complete"' in line for line in progress):
        failures.append("progress incomplete")
    replay_payload = {
        "aggregate": {k: v for k, v in aggregate.items() if k != "seconds"},
        "candidates": candidates,
        "selected_candidates": cards,
        "results": results,
        "variants": variants,
        "batch_rows": batches,
        "interactions": interactions,
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
    parser.add_argument("--out", default="target/pilot_wave/e124_text_understanding_mass_mutation_farm_capacity_probe")
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
