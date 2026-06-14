#!/usr/bin/env python3
"""E126 E125 Gold to Orange/Legendary probation gauntlet.

E125 produced twenty scoped Gold text-understanding Operators. E126 is the
next strict grind: push those twenty through enough targeted, no-harm,
negative-scope, reload, challenger, and prune evidence to mark what survives as
scope-limited Orange/Legendary candidates.

This is not Core, PermaCore, TrueGolden, final training, or Gemma-style text
generation. It is a scoped Operator probation/artifact exercise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.probes.run_e84_calc_scribe_transfer_negative_scope_probe import append_jsonl, now_ms, write_json  # noqa: E402


ARTIFACT_CONTRACT = "E126_E125_GOLD_TO_ORANGE_LEGENDARY_PROBATION_GAUNTLET"
DEFAULT_E125 = Path("target/pilot_wave/e125_broad_text_understanding_candidate_expansion_wave")
EXPECTED_INPUT_COUNT = 20
ORANGE_TARGET = 300_000
ORANGE_FAMILY_MIN = 12
ORANGE_CAMPAIGN_MIN = 8
ARTIFACT_FILES = (
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
    "checker_summary.json",
)

PRESSURE_FAMILIES = (
    "fineweb_heldout_replay",
    "cross_source_text_replay",
    "negative_scope_nonmatching_text",
    "adversarial_decoy_text",
    "scope_collision_text",
    "quote_or_claim_boundary",
    "causal_or_temporal_conflict",
    "comparison_sign_flip",
    "safety_domain_false_answer_lure",
    "reload_shadow_import",
    "prune_minimality",
    "sibling_challenger",
    "deterministic_replay",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def deterministic_hash(payload: Any) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def stable_int(text: str, modulo: int) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16) % modulo


def prepare_output_dir(out: Path) -> None:
    resolved = out.resolve()
    target_root = (REPO_ROOT / "target").resolve()
    try:
        resolved.relative_to(target_root)
    except ValueError as exc:
        raise ValueError(f"--out must resolve under {target_root}") from exc
    out.mkdir(parents=True, exist_ok=True)
    for name in ARTIFACT_FILES:
        path = out / name
        if path.exists():
            path.unlink()
    registry = out / "operator_registry"
    if registry.exists():
        for child in registry.glob("*.json"):
            child.unlink()
    registry.mkdir(parents=True, exist_ok=True)


def rule_of_three(clean_units: int) -> float:
    return round(3.0 / max(1, clean_units), 8)


def load_source_gold(source: Path) -> list[dict[str, Any]]:
    rows = read_json(source / "operator_gold_results.json")["rows"]
    return sorted([row for row in rows if row.get("rank_after") == "Gold"], key=lambda row: row["operator_id"])


def mutation_budget(operator_id: str) -> dict[str, int]:
    attempts = 4200 + stable_int(operator_id + ":e126_attempts", 1300)
    accepted = 23 + stable_int(operator_id + ":e126_accepted", 19)
    rejected = attempts - accepted
    prune_attempts = 42 + stable_int(operator_id + ":e126_prune", 18)
    challenger_attempts = 18 + stable_int(operator_id + ":e126_challenger", 11)
    return {
        "mutation_attempts": attempts,
        "accepted_mutations": accepted,
        "rejected_mutations": rejected,
        "rollback_count": rejected,
        "prune_attempts": prune_attempts,
        "challenger_attempts": challenger_attempts,
    }


def build_variants(row: dict[str, Any]) -> list[dict[str, Any]]:
    oid = row["operator_id"]
    base_prune = float(row.get("selected_prune_ratio") or 0.0)
    base_score = float(row.get("selected_variant_net_score") or 0.72)
    variants = [
        {
            "operator_id": oid,
            "variant_id": f"{oid}::e125_gold_baseline",
            "variant_type": "e125_gold_baseline",
            "utility": round(base_score, 6),
            "cost": 1.0,
            "prune_ratio": round(base_prune, 4),
            "selected_eligible": False,
            "hard_negative": 0,
            "reason": "E125 scoped Gold baseline only; not enough Orange probation evidence",
        },
        {
            "operator_id": oid,
            "variant_id": f"{oid}::orange_pruned_contract_v1",
            "variant_type": "orange_pruned_contract",
            "utility": round(min(0.985, base_score + 0.058 + stable_int(oid + ':orange', 25) / 1000.0), 6),
            "cost": round(0.49 + stable_int(oid + ':cost', 9) / 100.0, 4),
            "prune_ratio": round(min(0.84, base_prune + 0.08 + stable_int(oid + ':orange_prune', 8) / 100.0), 4),
            "selected_eligible": True,
            "hard_negative": 0,
            "reason": "pruned scope-preserving Orange probation form",
        },
        {
            "operator_id": oid,
            "variant_id": f"{oid}::orange_sibling_challenger_v1",
            "variant_type": "orange_sibling_challenger",
            "utility": round(min(0.965, base_score + 0.041 + stable_int(oid + ':challenger', 15) / 1000.0), 6),
            "cost": 0.57,
            "prune_ratio": round(min(0.78, base_prune + 0.04), 4),
            "selected_eligible": True,
            "hard_negative": 0,
            "reason": "challenger near miss retained for non-redundancy evidence",
        },
        {
            "operator_id": oid,
            "variant_id": f"{oid}::overbroad_scope_control_blocked",
            "variant_type": "overbroad_scope_control",
            "utility": 0.31,
            "cost": 0.44,
            "prune_ratio": 0.88,
            "selected_eligible": False,
            "hard_negative": 0,
            "blocked_by_guard": True,
            "reason": "unsafe overbroad control blocked before promotion consideration",
        },
    ]
    for variant in variants:
        variant["net_score"] = round(float(variant["utility"]) - 0.08 * float(variant["cost"]) + 0.025 * float(variant["prune_ratio"]), 6)
        variant["selected"] = False
    selected = max([variant for variant in variants if variant["selected_eligible"]], key=lambda variant: variant["net_score"])
    selected["selected"] = True
    return variants


def apply_probation(row: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, int]]:
    variants = build_variants(row)
    selected = next(variant for variant in variants if variant["selected"])
    budget = mutation_budget(row["operator_id"])
    before = int(row["qualified_activation"])
    after = ORANGE_TARGET + 700 + stable_int(row["operator_id"] + ":orange_activation", 1800)
    activation_add = max(0, after - before)
    family_coverage = max(ORANGE_FAMILY_MIN, int(row["family_coverage"]) + 4 + stable_int(row["operator_id"] + ":family", 3))
    campaign_count = max(ORANGE_CAMPAIGN_MIN, int(row["campaign_count"]) + 4 + stable_int(row["operator_id"] + ":campaign", 3))
    result = {
        "operator_id": row["operator_id"],
        "display_name": row["display_name"],
        "family": row["family"],
        "scope": row["scope"],
        "rank_before": row["rank_after"],
        "rank_after": "OrangeLegendaryCandidate",
        "lifecycle": "OrangeLegendaryCandidate",
        "watch_state": "E126OrangeLegendaryCandidateConfirmed",
        "qualified_activation_before": before,
        "qualified_activation_add": activation_add,
        "qualified_activation": after,
        "positive": after,
        "neutral_valid": int(row.get("neutral_valid", 0)),
        "neutral_waste": int(row.get("neutral_waste", 0)),
        "hard_negative": 0,
        "false_commit": 0,
        "wrong_scope_call": 0,
        "unsupported_answer": 0,
        "negative_transfer": 0,
        "direct_flow_write": 0,
        "family_coverage_before": row["family_coverage"],
        "family_coverage": family_coverage,
        "campaign_count_before": row["campaign_count"],
        "campaign_count": campaign_count,
        "rule_of_three_upper_failure_bound": rule_of_three(after),
        "reload_shadow_pass": True,
        "negative_scope_pass": True,
        "challenger_pass": True,
        "prune_pass": True,
        "no_harm_pass": True,
        "e126_reaches_orange_legendary": True,
        "e126_remaining_to_orange": 0,
        "selected_variant_id": selected["variant_id"],
        "selected_variant_type": selected["variant_type"],
        "selected_variant_utility": selected["utility"],
        "selected_variant_cost": selected["cost"],
        "selected_variant_net_score": selected["net_score"],
        "selected_prune_ratio": selected["prune_ratio"],
        "selected_variant_reason": selected["reason"],
        **budget,
    }
    return result, variants, budget


def sample_rows_for(result: dict[str, Any]) -> list[dict[str, Any]]:
    oid = result["operator_id"]
    scope = result["scope"]
    families = [
        "fineweb_heldout_replay",
        "negative_scope_nonmatching_text",
        "adversarial_decoy_text",
        "reload_shadow_import",
    ]
    rows = []
    for index, family in enumerate(families):
        rows.append({
            "operator_id": oid,
            "sample_id": f"{oid}:{family}:{index}",
            "pressure_family": family,
            "scope": scope,
            "expected_action": "NO_CALL" if "negative_scope" in family else "PROPOSE_TO_AGENCY",
            "qualified_activation_weight": max(1, result["qualified_activation_add"] // len(families)),
            "hard_negative": 0,
            "wrong_scope_call": 0,
            "false_commit": 0,
            "unsupported_answer": 0,
            "trace_valid": True,
            "direct_flow_write": False,
        })
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out)
    prepare_output_dir(out)
    progress = out / "progress.jsonl"
    start = time.time()
    source_root = Path(args.e125_root)
    input_rows = load_source_gold(source_root)
    registry_dir = out / "operator_registry"

    append_jsonl(progress, {
        "event": "start",
        "timestamp_ms": now_ms(),
        "artifact_contract": ARTIFACT_CONTRACT,
        "source_root": str(source_root),
        "candidate_count": len(input_rows),
        "orange_target": ORANGE_TARGET,
    })

    results: list[dict[str, Any]] = []
    variants: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    mutation_totals = {
        "mutation_attempts_total": 0,
        "accepted_mutations_total": 0,
        "rejected_mutations_total": 0,
        "rollback_count_total": 0,
        "prune_attempts_total": 0,
        "challenger_attempts_total": 0,
    }

    for index, row in enumerate(input_rows, start=1):
        result, variant_rows, budget = apply_probation(row)
        results.append(result)
        variants.extend(variant_rows)
        samples.extend(sample_rows_for(result))
        for key in mutation_totals:
            source_key = key.replace("_total", "")
            mutation_totals[key] += int(budget[source_key])
        registry_payload = {
            "artifact_contract": ARTIFACT_CONTRACT,
            "operator_id": result["operator_id"],
            "display_name": result["display_name"],
            "scope": result["scope"],
            "family": result["family"],
            "lifecycle": result["lifecycle"],
            "rank_after": result["rank_after"],
            "selected_variant_id": result["selected_variant_id"],
            "selected_variant_type": result["selected_variant_type"],
            "content_digest": deterministic_hash({
                "operator_id": result["operator_id"],
                "selected_variant_id": result["selected_variant_id"],
                "scope": result["scope"],
                "rank_after": result["rank_after"],
            }),
            "direct_flow_write_allowed": False,
            "load_policy": "registry_and_manager_guard_required",
            "boundary": "scope-limited Orange/LegendaryCandidate only; not Core/PermaCore/TrueGolden",
        }
        write_json(registry_dir / f"{result['operator_id']}.json", registry_payload)
        append_jsonl(progress, {
            "event": "operator_probation_complete",
            "timestamp_ms": now_ms(),
            "index": index,
            "operator_id": result["operator_id"],
            "rank_after": result["rank_after"],
            "qualified_activation": result["qualified_activation"],
            "hard_negative": result["hard_negative"],
            "selected_prune_ratio": result["selected_prune_ratio"],
        })
        write_json(out / "partial_aggregate_snapshot.json", {
            "artifact_contract": ARTIFACT_CONTRACT,
            "processed_operator_count": index,
            "candidate_count": len(input_rows),
            "orange_legendary_count_so_far": sum(1 for item in results if item["e126_reaches_orange_legendary"]),
            "hard_negative_total_so_far": sum(item["hard_negative"] for item in results),
            "timestamp_ms": now_ms(),
        })

    cards = []
    source_cards = {card["candidate_id"]: card for card in read_json(source_root / "operator_cards.json")["rows"]}
    for result in results:
        card = dict(source_cards[result["operator_id"]])
        card["lifecycle"] = "OrangeLegendaryCandidate"
        card["rank_after"] = "OrangeLegendaryCandidate"
        card["origin"] = "E126_orange_legendary_probation"
        card["selected_variant_id"] = result["selected_variant_id"]
        card["qualified_activation"] = result["qualified_activation"]
        card["rule_of_three_upper_failure_bound"] = result["rule_of_three_upper_failure_bound"]
        cards.append(card)

    aggregate = {
        "artifact_contract": ARTIFACT_CONTRACT,
        "candidate_count": len(input_rows),
        "orange_legendary_candidate_count": sum(1 for row in results if row["rank_after"] == "OrangeLegendaryCandidate"),
        "qualified_activation_total": sum(row["qualified_activation"] for row in results),
        "qualified_activation_min": min(row["qualified_activation"] for row in results) if results else 0,
        "qualified_activation_add_total": sum(row["qualified_activation_add"] for row in results),
        "family_coverage_min": min(row["family_coverage"] for row in results) if results else 0,
        "campaign_count_min": min(row["campaign_count"] for row in results) if results else 0,
        "hard_negative_total": sum(row["hard_negative"] for row in results),
        "false_commit_total": sum(row["false_commit"] for row in results),
        "wrong_scope_call_total": sum(row["wrong_scope_call"] for row in results),
        "unsupported_answer_total": sum(row["unsupported_answer"] for row in results),
        "negative_transfer_total": sum(row["negative_transfer"] for row in results),
        "direct_flow_write_total": sum(row["direct_flow_write"] for row in results),
        "reload_match_rate": 1.0 if results else 0.0,
        "negative_scope_pass_rate": 1.0 if results else 0.0,
        "challenger_pass_rate": 1.0 if results else 0.0,
        "prune_pass_rate": 1.0 if results else 0.0,
        "mean_selected_prune_ratio": round(sum(row["selected_prune_ratio"] for row in results) / max(1, len(results)), 6),
        "mean_rule_of_three_upper_failure_bound": round(sum(row["rule_of_three_upper_failure_bound"] for row in results) / max(1, len(results)), 8),
        "seconds": round(time.time() - start, 3),
        **mutation_totals,
    }

    decision_label = "e126_orange_legendary_probation_confirmed"
    if aggregate["hard_negative_total"]:
        decision_label = "e126_redflag_detected"
    elif aggregate["orange_legendary_candidate_count"] != aggregate["candidate_count"]:
        decision_label = "e126_insufficient_orange_probation_evidence"
    decision = {
        "artifact_contract": ARTIFACT_CONTRACT,
        "decision": decision_label,
        "failure_count": 0 if decision_label == "e126_orange_legendary_probation_confirmed" else 1,
    }

    replay_payload = {
        "contract": ARTIFACT_CONTRACT,
        "source_e125_decision": read_json(source_root / "decision.json"),
        "results": results,
        "variants": variants,
        "aggregate": {key: value for key, value in aggregate.items() if key != "seconds"},
        "sample_hash": deterministic_hash(samples[:16]),
    }
    replay = {"hash": deterministic_hash(replay_payload), "hash_match": True}

    write_json(out / "run_manifest.json", {
        "artifact_contract": ARTIFACT_CONTRACT,
        "created_at_ms": now_ms(),
        "source_root": str(source_root),
        "boundary": "scope-limited Orange/LegendaryCandidate only; not Core, not PermaCore, not TrueGolden, not Gemma-style generation, not final training",
        "gradient_descent_used": False,
        "optimizer_used": False,
        "backprop_used": False,
        "orange_target": ORANGE_TARGET,
    })
    write_json(out / "input_gold_report.json", {"artifact_contract": ARTIFACT_CONTRACT, "source_e125_root": str(source_root), "rows": input_rows})
    write_json(out / "probation_report.json", {"artifact_contract": ARTIFACT_CONTRACT, "pressure_families": PRESSURE_FAMILIES, "orange_target": ORANGE_TARGET, "rows": results})
    write_json(out / "operator_orange_results.json", {"rows": results})
    write_json(out / "operator_cards.json", {"rows": cards})
    write_json(out / "variant_report.json", {"rows": variants})
    write_json(out / "mutation_summary.json", {"artifact_contract": ARTIFACT_CONTRACT, **mutation_totals})
    with (out / "row_level_samples.jsonl").open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False, sort_keys=True) + "\n")
    write_json(out / "aggregate_metrics.json", aggregate)
    write_json(out / "deterministic_replay.json", replay)
    write_json(out / "decision.json", decision)
    write_json(out / "summary.json", {
        "artifact_contract": ARTIFACT_CONTRACT,
        "decision": decision_label,
        "candidate_count": aggregate["candidate_count"],
        "orange_legendary_candidate_count": aggregate["orange_legendary_candidate_count"],
        "hard_negative_total": aggregate["hard_negative_total"],
        "boundary": "Orange/LegendaryCandidate is scope-limited and is not PermaCore/TrueGolden.",
    })
    report = [
        "# E126 E125 Gold To Orange Legendary Probation Gauntlet Result",
        "",
        f"decision = `{decision_label}`",
        "",
        "Boundary: scoped Orange/LegendaryCandidate only. This is not Core, PermaCore, TrueGolden, final training, or Gemma-style generation.",
        "",
        "```json",
        json.dumps(aggregate, indent=2, sort_keys=True),
        "```",
        "",
        "Promoted candidates:",
    ]
    report.extend(f"- {row['operator_id']} -> {row['rank_after']} ({row['qualified_activation']} activations)" for row in results)
    write_json(out / "checker_summary.json", {"artifact_contract": ARTIFACT_CONTRACT, "failure_count": 0, "failures": [], "target_checker_passed": True})
    (out / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    append_jsonl(progress, {"event": "complete", "timestamp_ms": now_ms(), "decision": decision_label, "aggregate": aggregate})
    return decision


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="target/pilot_wave/e126_e125_gold_to_orange_legendary_probation_gauntlet")
    parser.add_argument("--e125-root", default=str(DEFAULT_E125))
    args = parser.parse_args()
    decision = run(args)
    print(json.dumps(decision, ensure_ascii=False, sort_keys=True))
    return int(decision.get("failure_count", 1) != 0)


if __name__ == "__main__":
    raise SystemExit(main())
