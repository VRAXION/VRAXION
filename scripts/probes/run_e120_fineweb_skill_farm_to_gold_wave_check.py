#!/usr/bin/env python3
"""Checker for E120 FineWeb skill farm to scoped Gold wave."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


CONTRACT = "E120_FINEWEB_SKILL_FARM_TO_GOLD_WAVE"
REQUIRED = [
    "run_manifest.json",
    "input_candidate_report.json",
    "operator_library_manifest.json",
    "operator_cards.json",
    "operator_gold_results.json",
    "variant_report.json",
    "promotion_report.json",
    "negative_scope_report.json",
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
    input_report = read_json(root / "input_candidate_report.json")
    library = read_json(root / "operator_library_manifest.json")
    cards = read_json(root / "operator_cards.json")["rows"]
    results = read_json(root / "operator_gold_results.json")["rows"]
    variants = read_json(root / "variant_report.json")["rows"]
    promotion = read_json(root / "promotion_report.json")
    negative_scope = read_json(root / "negative_scope_report.json")
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
    for text in ["scoped Gold promotion only", "not Core", "not PermaCore", "not TrueGolden", "not Gemma-style generation"]:
        if text not in boundary:
            failures.append(f"boundary missing: {text}")
    if manifest.get("gradient_descent_used") or manifest.get("optimizer_used") or manifest.get("backprop_used"):
        failures.append("gradient/optimizer/backprop unexpectedly enabled")
    if input_report.get("input_candidate_count") != len(cards):
        failures.append("input/card count mismatch")
    if library.get("operator_count") != len(cards):
        failures.append("library/card count mismatch")
    if aggregate.get("saved_operator_count") != len(cards):
        failures.append("aggregate/card count mismatch")
    if aggregate.get("saved_operator_count") != len(results):
        failures.append("saved/results count mismatch")
    if aggregate.get("promoted_to_gold_count") != len(promotion.get("promoted_to_gold", [])):
        failures.append("promotion count mismatch")
    if aggregate.get("promoted_to_gold_count", 0) < 1:
        failures.append("no Gold promotions")
    for key in ["hard_negative_total", "wrong_scope_call_total", "false_commit_total", "unsupported_answer_total", "negative_transfer_total"]:
        if aggregate.get(key) != 0:
            failures.append(f"{key} nonzero")
    if not negative_scope.get("pass"):
        failures.append("negative scope failed")
    if aggregate.get("reload_match_rate") != 1.0:
        failures.append("reload match not perfect")
    if aggregate.get("negative_scope_pass_rate") != 1.0:
        failures.append("negative scope pass rate not perfect")
    if aggregate.get("challenger_pass_rate") != 1.0:
        failures.append("challenger pass rate not perfect")
    if aggregate.get("prune_pass_rate") != 1.0:
        failures.append("prune pass rate not perfect")
    if mutation.get("accepted_mutations_total", 0) <= 0 or mutation.get("rollback_count_total", 0) <= 0:
        failures.append("mutation/rollback pressure missing")
    if not samples:
        failures.append("row-level samples missing")
    registry_dir = root / "operator_registry"
    if not registry_dir.exists():
        failures.append("operator registry missing")
    for card in cards:
        oid = card["operator_id"]
        reg = registry_dir / f"{oid}.json"
        if not reg.exists():
            failures.append(f"missing registry entry: {oid}")
            continue
        payload = read_json(reg)
        if payload.get("operator_id") != oid:
            failures.append(f"registry id mismatch: {oid}")
        if payload.get("direct_flow_write_allowed") is not False:
            failures.append(f"direct flow write not blocked: {oid}")
        if "Guard" not in payload.get("load_policy", "") and "guard" not in payload.get("load_policy", ""):
            failures.append(f"load policy missing guard: {oid}")
    selected = [row for row in variants if row.get("selected")]
    if len(selected) != len(results):
        failures.append("selected variant count mismatch")
    for row in results:
        if row.get("rank_after") == "Gold":
            if row.get("qualified_activation", 0) < 3000:
                failures.append(f"Gold under activation threshold: {row.get('operator_id')}")
            if row.get("family_coverage", 0) < 5:
                failures.append(f"Gold under family coverage threshold: {row.get('operator_id')}")
            if row.get("campaign_count", 0) < 3:
                failures.append(f"Gold under campaign threshold: {row.get('operator_id')}")
            if not row.get("negative_scope_pass") or not row.get("challenger_pass") or not row.get("prune_pass"):
                failures.append(f"Gold gate failed: {row.get('operator_id')}")
        if row.get("rank_after") in {"Core", "PermaCore", "TrueGolden"}:
            failures.append(f"invalid overpromotion: {row.get('operator_id')}")
    replay_payload = {
        "aggregate": {key: aggregate[key] for key in aggregate if key != "seconds"},
        "results": results,
        "variants": variants,
        "cards": cards,
        "contract": CONTRACT,
    }
    if replay.get("hash") != deterministic_hash(replay_payload) or not replay.get("hash_match"):
        failures.append("deterministic replay mismatch")
    if decision.get("failure_count") != 0:
        failures.append("decision failure_count nonzero")
    if summary.get("artifact_contract") != CONTRACT:
        failures.append("summary contract mismatch")
    if len(progress) < 2 or not any('"event": "complete"' in line or '"event":"complete"' in line for line in progress):
        failures.append("progress incomplete")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="target/pilot_wave/e120_fineweb_skill_farm_to_gold_wave")
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
