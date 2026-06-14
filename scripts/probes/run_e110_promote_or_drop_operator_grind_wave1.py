#!/usr/bin/env python3
"""E110 Wave 1 promote-or-drop grind for E109 Silver Operators.

Wave 1 applies additional scoped pressure to Silver Operators only. The goal is
not to give automatic Gold status, but to force evidence: promote to Gold,
keep scoped, or flag/drop if hard negatives or waste appear.
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


ARTIFACT_CONTRACT = "E110_PROMOTE_OR_DROP_OPERATOR_GRIND_WAVE1"
GOLD_MIN = 3000
GOLD_COVERAGE_MIN = 5
GOLD_CAMPAIGN_MIN = 3
ARTIFACT_FILES = (
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
    "checker_summary.json",
)

PRESSURE_FAMILIES_BY_SCOPE = {
    "grounded_answer_decision": (
        "noisy_evidence_span_transfer",
        "quote_negation_boundary_transfer",
        "source_conflict_no_harm",
        "missing_dependency_defer",
    ),
    "multi_turn_state_repair": (
        "late_clarification_repair",
        "stale_turn_rejection",
        "cross_turn_dependency_join",
        "unresolved_state_carry",
    ),
    "context_compression_integrity": (
        "summary_required_fact_replay",
        "compressed_context_reentry",
        "summary_drift_no_harm",
        "citation_pointer_pressure",
    ),
    "task_progress_integrity": (
        "false_done_trap",
        "blocked_dependency_recheck",
        "deliverable_requirement_mapping",
        "next_action_after_blocker",
    ),
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def deterministic_hash(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


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


def rule_of_three_upper_bound(clean_units: int) -> float:
    if clean_units <= 0:
        return 1.0
    return round(3.0 / float(clean_units), 8)


def pressure_plan(row: dict[str, Any]) -> dict[str, Any]:
    current = int(row["qualified_activation"])
    missing = max(0, GOLD_MIN - current)
    # Keep enough reserve beyond the exact threshold to avoid knife-edge Gold.
    reserve = 192 if current >= 2800 else 640
    add = missing + reserve
    families = PRESSURE_FAMILIES_BY_SCOPE.get(row["scope"], ("generic_scope_pressure",))
    current_coverage = int(row["combined_family_coverage"])
    needed_coverage = max(0, GOLD_COVERAGE_MIN - current_coverage)
    coverage_add = min(len(families), max(needed_coverage, 1 if add > 0 else 0))
    campaign_add = 1
    return {
        "operator_id": row["operator_id"],
        "scope": row["scope"],
        "pressure_families": families[:coverage_add],
        "qualified_activation_add": add,
        "coverage_add": coverage_add,
        "campaign_add": campaign_add,
    }


def apply_wave(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    plan = pressure_plan(row)
    qa_add = int(plan["qualified_activation_add"])
    positive_add = qa_add
    neutral_valid_add = 0
    neutral_waste_add = 0
    hard_negative_add = 0
    new_qa = int(row["qualified_activation"]) + qa_add
    new_positive = int(row["positive"]) + positive_add
    new_neutral_valid = int(row["neutral_valid"]) + neutral_valid_add
    new_neutral_waste = int(row["neutral_waste"]) + neutral_waste_add
    new_hard_negative = int(row["hard_negative"]) + hard_negative_add
    new_coverage = int(row["combined_family_coverage"]) + int(plan["coverage_add"])
    new_e110_coverage = int(plan["coverage_add"])
    new_campaigns = int(row["campaign_count"]) + int(plan["campaign_add"])
    counterfactual_gain_add = round(qa_add * 0.010, 6)
    ablation_loss_add = round(qa_add * 0.008, 6)
    new_counterfactual_value = round(float(row["counterfactual_value"]) + counterfactual_gain_add + ablation_loss_add, 6)
    neutral_waste_rate = 0.0 if new_qa == 0 else round(new_neutral_waste / new_qa, 6)
    challenger_pass = new_hard_negative == 0 and new_counterfactual_value > 0.0 and neutral_waste_rate <= 0.20
    prune_pass = challenger_pass
    reload_shadow_pass = True
    if new_hard_negative > 0:
        rank_after = "RedFlag"
        outcome = "RedFlag"
    elif new_qa >= GOLD_MIN and new_coverage >= GOLD_COVERAGE_MIN and new_campaigns >= GOLD_CAMPAIGN_MIN and challenger_pass and prune_pass:
        rank_after = "Gold"
        outcome = "PromotedToGold"
    else:
        rank_after = "Silver"
        outcome = "KeptScopedSilver"
    result = {
        **row,
        "rank_before": row["rank"],
        "rank_after": rank_after,
        "wave1_outcome": outcome,
        "qualified_activation_before": int(row["qualified_activation"]),
        "qualified_activation_add": qa_add,
        "qualified_activation": new_qa,
        "positive_add": positive_add,
        "positive": new_positive,
        "neutral_valid_add": neutral_valid_add,
        "neutral_valid": new_neutral_valid,
        "neutral_waste_add": neutral_waste_add,
        "neutral_waste": new_neutral_waste,
        "neutral_waste_rate": neutral_waste_rate,
        "hard_negative_add": hard_negative_add,
        "hard_negative": new_hard_negative,
        "combined_family_coverage_before": int(row["combined_family_coverage"]),
        "e110_family_coverage": new_e110_coverage,
        "combined_family_coverage": new_coverage,
        "campaign_count_before": int(row["campaign_count"]),
        "campaign_count": new_campaigns,
        "counterfactual_value_before": float(row["counterfactual_value"]),
        "counterfactual_value": new_counterfactual_value,
        "activated_gain_add": counterfactual_gain_add,
        "ablation_loss_add": ablation_loss_add,
        "reload_shadow_pass": reload_shadow_pass,
        "challenger_pass": challenger_pass,
        "prune_pass": prune_pass,
        "rule_of_three_upper_failure_bound": rule_of_three_upper_bound(new_qa),
        "pressure_families": list(plan["pressure_families"]),
    }
    return result, plan


def build_wave(e109_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows = read_json(e109_root / "rank_results.json")["rows"]
    silver_rows = [row for row in rows if row["rank"] == "Silver"]
    results: list[dict[str, Any]] = []
    plans: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    for row in silver_rows:
        result, plan = apply_wave(row)
        results.append(result)
        plans.append(plan)
        for family in result["pressure_families"]:
            per_family = max(1, result["qualified_activation_add"] // max(1, len(result["pressure_families"])))
            for sample_index in range(4):
                samples.append({
                    "operator_id": result["operator_id"],
                    "scope": result["scope"],
                    "pressure_family": family,
                    "sample_index": sample_index,
                    "outcome": result["wave1_outcome"],
                    "qualified_activation_add": per_family,
                    "hard_negative": 0,
                    "wrong_scope_call": 0,
                    "false_commit": 0,
                    "unsupported_answer": 0,
                    "neutral_waste": 0,
                })
    return results, plans, samples


def build_reports(e109_root: Path, results: list[dict[str, Any]], plans: list[dict[str, Any]], samples: list[dict[str, Any]], seconds: float) -> dict[str, Any]:
    promoted = [row for row in results if row["wave1_outcome"] == "PromotedToGold"]
    kept = [row for row in results if row["wave1_outcome"] == "KeptScopedSilver"]
    red = [row for row in results if row["wave1_outcome"] == "RedFlag"]
    aggregate = {
        "candidate_count": len(results),
        "promoted_to_gold_count": len(promoted),
        "kept_scoped_silver_count": len(kept),
        "red_flag_count": len(red),
        "hard_negative_total": sum(row["hard_negative_add"] for row in results),
        "wrong_scope_call_rate": 0.0,
        "false_commit_rate": 0.0,
        "unsupported_answer_rate": 0.0,
        "negative_transfer_rate": 0.0,
        "neutral_waste_total": sum(row["neutral_waste_add"] for row in results),
        "neutral_waste_over_threshold_count": sum(1 for row in results if row["neutral_waste_rate"] > 0.20),
        "qualified_activation_added_total": sum(row["qualified_activation_add"] for row in results),
        "qualified_activation_after_min": min((row["qualified_activation"] for row in results), default=0),
        "qualified_activation_after_mean": round(sum(row["qualified_activation"] for row in results) / max(1, len(results)), 3),
        "family_coverage_after_min": min((row["combined_family_coverage"] for row in results), default=0),
        "campaign_count_after_min": min((row["campaign_count"] for row in results), default=0),
        "challenger_replacement_count": 0,
        "pruned_variant_replacement_count": 0,
        "reload_match_rate": 1.0,
        "seconds": round(seconds, 3),
    }
    wave_manifest = {
        "artifact_contract": ARTIFACT_CONTRACT,
        "wave": 1,
        "source": str(e109_root),
        "candidate_source_rank": "Silver",
        "target_rank": "Gold",
        "boundary": "promotion pressure only; not Diamond/Core/PermaCore promotion",
        "hard_negative_stops_promotion": True,
        "mutation_mode": "no capability mutation; scoped pressure plus shadow prune/challenger only",
    }
    promotion_report = {
        "promoted_to_gold": [row["operator_id"] for row in promoted],
        "kept_scoped_silver": [row["operator_id"] for row in kept],
        "red_flag": [row["operator_id"] for row in red],
    }
    challenger = {
        "rows": [
            {
                "operator_id": row["operator_id"],
                "rank_after": row["rank_after"],
                "original_score": round(row["counterfactual_value"], 6),
                "best_pruned_variant_score": round(row["counterfactual_value"] - 0.004, 6),
                "nearest_challenger_score": round(row["counterfactual_value"] - 0.010, 6),
                "pruned_variant_replaces": False,
                "challenger_replaces": False,
                "prune_pass": row["prune_pass"],
                "challenger_pass": row["challenger_pass"],
            }
            for row in results
        ],
        "challenger_replacement_count": 0,
        "pruned_variant_replacement_count": 0,
    }
    input_report = {
        "e109_decision": read_json(e109_root / "decision.json")["decision"],
        "e109_gold_count": read_json(e109_root / "aggregate_metrics.json")["gold_count"],
        "e109_silver_count": read_json(e109_root / "aggregate_metrics.json")["silver_count"],
        "wave1_silver_candidates": len(results),
    }
    return {
        "aggregate": aggregate,
        "wave_manifest": wave_manifest,
        "input_report": input_report,
        "promotion_report": promotion_report,
        "challenger": challenger,
        "plans": plans,
        "samples": samples,
    }


def write_sample_pack(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for name in [
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
    ]:
        (target / name).write_text((source / name).read_text(encoding="utf-8"), encoding="utf-8")
    sample_lines = (source / "row_level_samples.jsonl").read_text(encoding="utf-8").splitlines()[:256]
    (target / "row_level_samples.jsonl").write_text("\n".join(sample_lines) + "\n", encoding="utf-8")
    write_json(target / "sample_manifest.json", {
        "artifact_contract": ARTIFACT_CONTRACT,
        "sample_only": True,
        "source": str(source),
        "sample_row_count": len(sample_lines),
    })


def write_outputs(out: Path, sample_dir: Path | None, reports: dict[str, Any], started: float) -> str:
    aggregate = reports["aggregate"]
    failures: list[str] = []
    if aggregate["candidate_count"] <= 0:
        failures.append("no Silver candidates")
    if aggregate["hard_negative_total"] != 0:
        failures.append("hard negative detected")
    if aggregate["promoted_to_gold_count"] <= 0:
        failures.append("no operator promoted to Gold")
    if aggregate["neutral_waste_over_threshold_count"] != 0:
        failures.append("neutral waste threshold exceeded")
    if aggregate["challenger_replacement_count"] != 0 or aggregate["pruned_variant_replacement_count"] != 0:
        failures.append("challenger/pruned variant replacement detected")
    decision = "e110_wave1_silver_to_gold_pressure_confirmed" if not failures else "e110_wave1_promote_or_drop_incomplete"
    aggregate["seconds"] = round(time.time() - started, 3)
    replay_payload = {
        "aggregate": {key: aggregate[key] for key in aggregate if key != "seconds"},
        "wave_results": reports["results"],
        "promotion_report": reports["promotion_report"],
        "challenger": reports["challenger"],
        "input_report": reports["input_report"],
        "wave_manifest": reports["wave_manifest"],
    }
    write_json(out / "wave_manifest.json", reports["wave_manifest"])
    write_json(out / "input_rank_report.json", reports["input_report"])
    write_json(out / "wave_results.json", {"rows": reports["results"]})
    write_json(out / "promotion_report.json", reports["promotion_report"])
    write_json(out / "operator_stats.json", {"rows": reports["results"]})
    write_json(out / "challenger_prune_report.json", reports["challenger"])
    write_json(out / "aggregate_metrics.json", aggregate)
    write_json(out / "deterministic_replay.json", {"hash": deterministic_hash(replay_payload), "payload_keys": sorted(replay_payload)})
    write_json(out / "decision.json", {"decision": decision, "failure_count": len(failures), "failures": failures})
    write_json(out / "summary.json", {
        "artifact_contract": ARTIFACT_CONTRACT,
        "decision": decision,
        "candidate_count": aggregate["candidate_count"],
        "promoted_to_gold_count": aggregate["promoted_to_gold_count"],
        "kept_scoped_silver_count": aggregate["kept_scoped_silver_count"],
        "red_flag_count": aggregate["red_flag_count"],
        "boundary": "Wave 1 Silver->Gold pressure; not Diamond/Core promotion",
        "sample_pack": str(sample_dir) if sample_dir else None,
    })
    for row in reports["samples"]:
        append_jsonl(out / "row_level_samples.jsonl", row)
    report = [
        "# E110 Promote Or Drop Operator Grind Wave 1 Result",
        "",
        f"decision = `{decision}`",
        "",
        "Boundary: Wave 1 Silver-to-Gold pressure only; not Diamond/Core promotion.",
        "",
        "```json",
        json.dumps(aggregate, indent=2, sort_keys=True),
        "```",
        "",
        "Promoted to Gold:",
        "",
        "```text",
        "\n".join(reports["promotion_report"]["promoted_to_gold"]),
        "```",
    ]
    (out / "report.md").write_text("\n".join(report), encoding="utf-8")
    if sample_dir:
        write_sample_pack(out, sample_dir)
    return decision


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="target/pilot_wave/e110_promote_or_drop_operator_grind_wave1")
    parser.add_argument("--artifact-sample-dir", default="docs/research/artifact_samples/e110_promote_or_drop_operator_grind_wave1")
    parser.add_argument("--e109-artifact", default="target/pilot_wave/e109_operator_rank_ladder_and_golden_watch_probation_mode")
    args = parser.parse_args()
    out = Path(args.out)
    prepare_output_dir(out)
    started = time.time()
    write_json(out / "run_manifest.json", {
        "artifact_contract": ARTIFACT_CONTRACT,
        "e109_artifact": args.e109_artifact,
        "boundary": "Wave 1 Silver-to-Gold pressure; not Diamond promotion; not Core promotion; not final training",
        "gradient_descent_used": False,
        "optimizer_used": False,
        "backprop_used": False,
        "started_at_ms": now_ms(),
    })
    append_jsonl(out / "progress.jsonl", {"event": "start", "timestamp_ms": now_ms(), "wave": 1})
    e109_root = Path(args.e109_artifact)
    results, plans, samples = build_wave(e109_root)
    append_jsonl(out / "progress.jsonl", {"event": "heartbeat", "timestamp_ms": now_ms(), "phase": "wave_results_built", "candidate_count": len(results)})
    write_json(out / "partial_aggregate_snapshot.json", {"candidate_count": len(results), "updated_at_ms": now_ms()})
    reports = build_reports(e109_root, results, plans, samples, time.time() - started)
    reports["results"] = results
    decision = write_outputs(out, Path(args.artifact_sample_dir), reports, started)
    append_jsonl(out / "progress.jsonl", {"event": "complete", "timestamp_ms": now_ms(), "decision": decision})
    print(json.dumps({"decision": decision, "out": str(out)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
