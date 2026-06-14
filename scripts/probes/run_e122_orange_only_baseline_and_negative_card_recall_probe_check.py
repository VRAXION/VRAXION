#!/usr/bin/env python3
"""Checker for E122 orange-only baseline and negative-card recall probe."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


CONTRACT = "E122_ORANGE_ONLY_BASELINE_AND_NEGATIVE_CARD_RECALL_PROBE"
ORANGE_TARGET = 300_000
REQUIRED = [
    "run_manifest.json",
    "input_operator_report.json",
    "orange_only_results.json",
    "negative_knowledge_cards.json",
    "negative_card_usage_report.json",
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
    input_report = read_json(root / "input_operator_report.json")
    results = read_json(root / "orange_only_results.json")["rows"]
    cards = read_json(root / "negative_knowledge_cards.json")["rows"]
    usage = read_json(root / "negative_card_usage_report.json")
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
    for phrase in ["Orange-only active baseline", "mutation-planner negative cards only", "not Core", "not PermaCore", "not TrueGolden", "not final training", "not Gemma-style generation"]:
        if phrase not in boundary:
            failures.append(f"boundary missing: {phrase}")
    for phrase in ["PermaCore promotion confirmed", "TrueGolden promotion confirmed", "Core promotion confirmed", "Gemma-level"]:
        if phrase in report_text:
            failures.append(f"forbidden claim in report: {phrase}")
    if manifest.get("gradient_descent_used") or manifest.get("optimizer_used") or manifest.get("backprop_used"):
        failures.append("gradient/optimizer/backprop unexpectedly enabled")

    active_rows = input_report.get("active_rows", [])
    inactive_rows = input_report.get("inactive_rows", [])
    if aggregate.get("active_operator_count") != len(active_rows):
        failures.append("active operator count mismatch")
    if aggregate.get("inactive_operator_count") != len(inactive_rows):
        failures.append("inactive operator count mismatch")
    if aggregate.get("active_operator_count", 0) <= 0:
        failures.append("no active operators")
    if aggregate.get("orange_only_active_count") != aggregate.get("active_operator_count"):
        failures.append("active set is not orange-only")
    if aggregate.get("non_orange_active_count") != 0:
        failures.append("non-orange active count nonzero")
    if aggregate.get("deprecated_count") != 3:
        failures.append("expected three deprecated operators to remain inactive")
    if len(results) != aggregate.get("active_operator_count"):
        failures.append("result count mismatch")
    for key in [
        "hard_negative_total",
        "wrong_scope_call_total",
        "false_commit_total",
        "unsupported_answer_total",
        "negative_transfer_total",
        "direct_flow_write_total",
    ]:
        if aggregate.get(key) != 0:
            failures.append(f"{key} nonzero")
    if aggregate.get("qualified_activation_min", 0) < ORANGE_TARGET:
        failures.append("qualified activation min below orange target")
    for key in ["reload_match_rate", "negative_scope_pass_rate", "challenger_pass_rate", "prune_pass_rate"]:
        if aggregate.get(key) != 1.0:
            failures.append(f"{key} not perfect")

    card_ids = set()
    for card in cards:
        cid = card.get("negative_card_id")
        if not cid or cid in card_ids:
            failures.append(f"duplicate or missing card id: {cid}")
        card_ids.add(cid)
        if card.get("runtime_callable") is not False or card.get("visible_to_router") is not False or card.get("visible_to_mutation_planner") is not True:
            failures.append(f"negative card visibility/callability mismatch: {cid}")
        if card.get("status") != "active_negative_prior":
            failures.append(f"negative card status mismatch: {cid}")
        if card.get("false_block_count") != 0:
            failures.append(f"negative card false block nonzero: {cid}")
        registry = root / "negative_card_registry" / f"{cid}.json"
        if not registry.exists():
            failures.append(f"missing negative card registry file: {cid}")
    if len(cards) != aggregate.get("negative_card_count"):
        failures.append("negative card count mismatch")
    if aggregate.get("negative_card_count", 0) < aggregate.get("active_operator_count", 0):
        failures.append("too few negative cards")
    if usage.get("negative_card_count") != aggregate.get("negative_card_count"):
        failures.append("usage negative card count mismatch")
    if usage.get("negative_card_recall_event_count") != aggregate.get("negative_card_recall_event_count"):
        failures.append("negative card recall count mismatch")
    if usage.get("prevented_repeat_failure_count") != aggregate.get("prevented_repeat_failure_count"):
        failures.append("prevented repeat count mismatch")
    if aggregate.get("negative_card_recall_event_count", 0) <= 0:
        failures.append("negative cards were not recalled")
    if aggregate.get("prevented_repeat_failure_count", 0) <= 0:
        failures.append("negative cards prevented no repeated failures")
    if aggregate.get("false_block_count") != 0 or usage.get("false_block_count") != 0 or aggregate.get("false_block_rate") != 0.0:
        failures.append("negative card false block detected")
    if usage.get("normal_router_callable_cards") != 0 or usage.get("planner_only") is not True:
        failures.append("negative cards leaked into normal router")
    if usage.get("negative_card_precision") != 1.0:
        failures.append("negative card precision not perfect in this probe")

    for row in results:
        oid = row.get("operator_id")
        if row.get("rank_after") != "OrangeLegendaryCandidate":
            failures.append(f"not orange after E122: {oid}")
        if row.get("qualified_activation", 0) < ORANGE_TARGET:
            failures.append(f"under orange target: {oid}")
        if row.get("e122_remaining_to_orange") != 0 or not row.get("e122_orange_only_baseline"):
            failures.append(f"orange baseline not confirmed: {oid}")
        if row.get("hard_negative") or row.get("wrong_scope_call") or row.get("false_commit") or row.get("unsupported_answer") or row.get("direct_flow_write"):
            failures.append(f"unsafe row: {oid}")
        if not row.get("reload_shadow_pass") or not row.get("negative_scope_pass") or not row.get("challenger_pass") or not row.get("prune_pass"):
            failures.append(f"gate failed: {oid}")
        if row.get("negative_card_count", 0) <= 0 or row.get("negative_card_recall_count", 0) <= 0:
            failures.append(f"negative card evidence missing: {oid}")
    if mutation.get("accepted_mutations_total", 0) <= 0 or mutation.get("rollback_count_total", 0) <= 0:
        failures.append("mutation/rollback evidence missing")
    if mutation.get("negative_card_recall_event_count") != aggregate.get("negative_card_recall_event_count"):
        failures.append("mutation summary recall mismatch")
    if not samples:
        failures.append("row-level samples missing")
    for sample in samples:
        if sample.get("false_block") or sample.get("hard_negative") or sample.get("direct_flow_write"):
            failures.append(f"unsafe sample: {sample.get('sample_id')}")
    if len(progress) < 4 or not any('"event": "complete"' in line or '"event":"complete"' in line for line in progress):
        failures.append("progress incomplete")

    input_rows = active_rows + inactive_rows
    replay_payload = {
        "contract": CONTRACT,
        "input_row_hash": deterministic_hash([{key: row.get(key) for key in ("operator_id", "rank", "qualified_activation", "e117_activation_after_gauntlet")} for row in input_rows]),
        "results": results,
        "negative_cards": cards,
        "aggregate": {key: value for key, value in aggregate.items() if key != "seconds"},
        "sample_hash": deterministic_hash(samples[:32]),
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
    parser.add_argument("--out", default="target/pilot_wave/e122_orange_only_baseline_and_negative_card_recall_probe")
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
