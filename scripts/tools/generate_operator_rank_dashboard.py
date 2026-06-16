#!/usr/bin/env python3
"""Generate a self-contained Operator rank dashboard from rank artifacts."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


DEFAULT_E109 = Path("target/pilot_wave/e109_operator_rank_ladder_and_golden_watch_probation_mode")
DEFAULT_E110 = Path("target/pilot_wave/e110_promote_or_drop_operator_grind_wave1")
DEFAULT_E111 = Path("target/pilot_wave/e111_bronze_mutation_prune_promote_or_drop_wave")
DEFAULT_E112 = Path("target/pilot_wave/e112_gold_to_core_prune_heavy_probation_wave")
DEFAULT_E114 = Path("target/pilot_wave/e114_fineweb_next_limit_stability_projection")
DEFAULT_E116 = Path("target/pilot_wave/e116_alpha_weave_synthetic_pressure_generation")
DEFAULT_E117 = Path("target/pilot_wave/e117_alpha_weave_targeted_pressure_gauntlet")
DEFAULT_E118 = Path("target/pilot_wave/e118_core_candidate_cross_source_no_harm_gauntlet")
DEFAULT_E120 = Path("target/pilot_wave/e120_fineweb_skill_farm_to_gold_wave")
DEFAULT_E121 = Path("target/pilot_wave/e121_e120_gold_to_orange_legendary_probation_gauntlet")
DEFAULT_E122 = Path("target/pilot_wave/e122_orange_only_baseline_and_negative_card_recall_probe")
DEFAULT_E127 = Path("target/pilot_wave/e127_overnight_text_skill_farm_orange_cycle")
DEFAULT_E129 = Path("target/pilot_wave/e129_arithmetic_trace_orange_legendary_probation")
DEFAULT_E130A = Path("target/pilot_wave/e130a_corememory_to_orange_backfill_gauntlet")
DEFAULT_E130B = Path("target/pilot_wave/e130b_arithmetic_text_io_transfer_and_word_problem_no_call_gauntlet")
DEFAULT_E131 = Path("target/pilot_wave/e131_visible_equation_extraction_and_assistant_arithmetic_render_gauntlet")
DEFAULT_E132 = Path("target/pilot_wave/e132_external_math_text_skill_farm_mutation_prune_orange_cycle")
DEFAULT_E133 = Path("target/pilot_wave/e133_math_text_route_composition_and_no_solve_assistant_confirm")
DEFAULT_E134 = Path("target/pilot_wave/e134_external_math_text_ood_route_stress_and_counterexample_gauntlet")
DEFAULT_E135 = Path("target/pilot_wave/e135_math_text_multi_route_assistant_dialogue_state_gauntlet")
DEFAULT_E136A = Path("target/pilot_wave/e136a_assistant_text_skill_farm_mutation_prune_orange_cycle")
DEFAULT_E136B = Path("target/pilot_wave/e136b_assistant_text_route_composition_and_boundary_confirm")
SAMPLE_E109 = Path("docs/research/artifact_samples/e109_operator_rank_ladder_and_golden_watch_probation_mode")
SAMPLE_E110 = Path("docs/research/artifact_samples/e110_promote_or_drop_operator_grind_wave1")
SAMPLE_E111 = Path("docs/research/artifact_samples/e111_bronze_mutation_prune_promote_or_drop_wave")
SAMPLE_E112 = Path("docs/research/artifact_samples/e112_gold_to_core_prune_heavy_probation_wave")
SAMPLE_E127 = Path("docs/research/artifact_samples/e127_overnight_text_skill_farm_orange_cycle")
SAMPLE_E129 = Path("docs/research/artifact_samples/e129_arithmetic_trace_orange_legendary_probation")
SAMPLE_E130A = Path("docs/research/artifact_samples/e130a_corememory_to_orange_backfill_gauntlet")
SAMPLE_E130B = Path("docs/research/artifact_samples/e130b_arithmetic_text_io_transfer_and_word_problem_no_call_gauntlet")
SAMPLE_E131 = Path("docs/research/artifact_samples/e131_visible_equation_extraction_and_assistant_arithmetic_render_gauntlet")
SAMPLE_E132 = Path("docs/research/artifact_samples/e132_external_math_text_skill_farm_mutation_prune_orange_cycle")
SAMPLE_E133 = Path("docs/research/artifact_samples/e133_math_text_route_composition_and_no_solve_assistant_confirm")
SAMPLE_E134 = Path("docs/research/artifact_samples/e134_external_math_text_ood_route_stress_and_counterexample_gauntlet")
SAMPLE_E135 = Path("docs/research/artifact_samples/e135_math_text_multi_route_assistant_dialogue_state_gauntlet")
SAMPLE_E136A = Path("docs/research/artifact_samples/e136a_assistant_text_skill_farm_mutation_prune_orange_cycle")
SAMPLE_E136B = Path("docs/research/artifact_samples/e136b_assistant_text_route_composition_and_boundary_confirm")
DEFAULT_OUT = Path("target/pilot_wave/operator_rank_dashboard/index.html")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def existing_artifact_path(requested: Path, fallback: Path, required_file: str) -> Path:
    if (requested / required_file).exists():
        return requested
    if (fallback / required_file).exists():
        return fallback
    raise FileNotFoundError(f"missing artifact {required_file!r} in {requested} or {fallback}")


def compact_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keep = [
        "operator_id",
        "display_name",
        "scope",
        "family",
        "group_id",
        "e107_status",
        "e108_status",
        "rank",
        "watch_state",
        "qualified_activation",
        "positive",
        "neutral_valid",
        "neutral_waste",
        "neutral_waste_rate",
        "hard_negative",
        "rule_of_three_upper_failure_bound",
        "e107_family_coverage",
        "e108_family_coverage",
        "combined_family_coverage",
        "campaign_count",
        "counterfactual_value",
        "activated_gain",
        "ablation_loss",
        "reload_shadow_pass",
        "challenger_pass",
        "prune_pass",
        "e110_wave1_outcome",
        "qualified_activation_add",
        "rank_before",
        "rank_after",
        "e111_wave2_outcome",
        "selected_variant_id",
        "selected_variant_type",
        "selected_variant_net_score",
        "mutation_attempts",
        "accepted_mutations",
        "rejected_mutations",
        "rollback_count",
        "e112_wave3_outcome",
        "selected_prune_ratio",
        "long_horizon_no_harm_pass",
        "negative_scope_pass",
        "e114_current_run_calls",
        "e114_projected_full_fineweb_calls",
        "e114_projected_activation_after_full_fineweb",
        "e114_projected_reaches_permacore_probation",
        "e114_projected_remaining_after_full_fineweb",
        "e114_selected_variant",
        "e116_template_family",
        "e116_generated_cell_packs",
        "e116_variant_count",
        "e116_repeat_count_per_pack",
        "e116_qualified_synthetic_pressure_activation",
        "e116_projected_activation_after_targeted_pressure",
        "e116_reaches_permacore_probation_after_targeted_pressure",
        "e117_qualified_activation",
        "e117_activation_after_gauntlet",
        "e117_remaining_after_gauntlet",
        "e117_reaches_permacore_probation_after_gauntlet",
        "e117_hard_negative",
        "e117_positive_activation",
        "e117_neutral_valid_activation",
        "e117_negative_scope_valid_activation",
        "e118_cross_source_no_harm_pass",
        "e118_source_family_coverage",
        "e118_case_count",
        "e118_hard_negative",
        "e118_negative_transfer",
        "e118_synthetic_imprint",
        "e118_ablation_value",
        "e120_origin",
        "e120_support_count",
        "e120_saved_operator",
        "e120_gold_pass",
        "e120_selected_variant_type",
        "e120_selected_prune_ratio",
        "e120_reload_shadow_pass",
        "e120_negative_scope_pass",
        "e120_challenger_pass",
        "e120_prune_pass",
        "e120_hard_negative",
        "e120_wrong_scope_call",
        "e120_false_commit",
        "e120_unsupported_answer",
        "e120_description",
        "e120_promotion_reason",
        "e121_reaches_orange_legendary",
        "e121_remaining_to_orange",
        "e121_hard_negative",
        "e121_wrong_scope_call",
        "e121_false_commit",
        "e121_unsupported_answer",
        "e121_direct_flow_write",
        "e121_selected_variant_type",
        "e121_selected_prune_ratio",
        "e121_family_coverage",
        "e121_campaign_count",
        "e122_orange_only_baseline",
        "e122_was_previously_orange",
        "e122_remaining_to_orange",
        "e122_negative_card_count",
        "e122_negative_card_recall_count",
        "e122_prevented_repeat_failure_count",
        "e122_false_block_count",
        "e127_cycle",
        "e127_reaches_orange_legendary",
        "e127_remaining_to_orange",
        "e127_hard_negative",
        "e127_wrong_scope_call",
        "e127_false_commit",
        "e127_unsupported_answer",
        "e127_direct_flow_write",
        "e127_selected_variant_type",
        "e127_selected_prune_ratio",
        "e127_family_coverage",
        "e127_campaign_count",
        "e127_description",
        "e129_arithmetic_trace_operator",
        "e129_min_in_scope_accuracy",
        "e129_negative_scope_case_count",
        "e129_negative_scope_pass_rate",
        "e129_false_commit",
        "e129_wrong_scope_call",
        "e129_unsupported_answer",
        "e129_selected_variant_cost",
        "e129_campaign_group_count",
        "e130a_reaches_orange_legendary",
        "e130a_source_rank",
        "e130a_activation_before",
        "e130a_activation_add",
        "e130a_remaining_to_orange",
        "e130a_hard_negative",
        "e130a_wrong_scope_call",
        "e130a_false_commit",
        "e130a_unsupported_answer",
        "e130a_direct_flow_write",
        "e130a_negative_transfer",
        "e130a_pressure_family_count",
        "e130b_text_io_transfer",
        "e130b_source_e129_rank",
        "e130b_selected_route",
        "e130b_visible_transfer_case_count",
        "e130b_visible_transfer_accuracy",
        "e130b_word_problem_no_call_case_count",
        "e130b_word_problem_no_call_accuracy",
        "e130b_qualified_transfer_activation",
        "e130b_hard_negative",
        "e130b_wrong_scope_call",
        "e130b_false_commit",
        "e130b_unsupported_answer",
        "e130b_direct_flow_write",
        "e130b_overbroad_control_wrong_scope_call",
        "e131_visible_equation_transfer",
        "e131_selected_route",
        "e131_visible_equation_case_count",
        "e131_visible_equation_extraction_accuracy",
        "e131_word_problem_no_call_case_count",
        "e131_word_problem_no_call_accuracy",
        "e131_qualified_visible_activation",
        "e131_hard_negative",
        "e131_wrong_scope_call",
        "e131_false_commit",
        "e131_unsupported_answer",
        "e131_boundary_claim_violation",
        "e131_direct_flow_write",
        "e131_e130b_baseline_visible_miss",
        "e131_overbroad_control_wrong_scope_call",
    ]
    return [{key: row.get(key) for key in keep} for row in rows]


def rank_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    keys = ["Bronze", "Silver", "Gold", "DiamondCandidate", "CoreMemoryCandidate", "OrangeLegendaryCandidate", "RedFlag", "Deprecated"]
    return {key: sum(1 for row in rows if row.get("rank") == key) for key in keys}


def merge_e110(rows: list[dict[str, Any]], e110: Path | None) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if not e110 or not (e110 / "wave_results.json").exists():
        return rows, None
    wave = read_json(e110 / "wave_results.json")["rows"]
    wave_by_id = {row["operator_id"]: row for row in wave}
    merged: list[dict[str, Any]] = []
    for row in rows:
        update = wave_by_id.get(row["operator_id"])
        if not update:
            merged.append(row)
            continue
        next_row = dict(row)
        for key in [
            "rank_after",
            "rank_before",
            "wave1_outcome",
            "qualified_activation",
            "qualified_activation_add",
            "positive",
            "neutral_valid",
            "neutral_waste",
            "neutral_waste_rate",
            "hard_negative",
            "rule_of_three_upper_failure_bound",
            "combined_family_coverage",
            "campaign_count",
            "counterfactual_value",
            "activated_gain",
            "ablation_loss",
            "reload_shadow_pass",
            "challenger_pass",
            "prune_pass",
        ]:
            if key in update:
                next_row[key if key != "wave1_outcome" else "e110_wave1_outcome"] = update[key]
        next_row["rank"] = update.get("rank_after", next_row["rank"])
        next_row["watch_state"] = "E110GoldConfirmed" if next_row["rank"] == "Gold" else next_row.get("watch_state")
        merged.append(next_row)
    return merged, {
        "summary": read_json(e110 / "summary.json"),
        "aggregate": read_json(e110 / "aggregate_metrics.json"),
        "promotion": read_json(e110 / "promotion_report.json"),
    }


def merge_e111(rows: list[dict[str, Any]], e111: Path | None) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if not e111 or not (e111 / "wave_results.json").exists():
        return rows, None
    wave = read_json(e111 / "wave_results.json")["rows"]
    wave_by_id = {row["operator_id"]: row for row in wave}
    merged: list[dict[str, Any]] = []
    for row in rows:
        update = wave_by_id.get(row["operator_id"])
        if not update:
            merged.append(row)
            continue
        next_row = dict(row)
        for key in [
            "rank_after",
            "rank_before",
            "wave2_outcome",
            "selected_variant_id",
            "selected_variant_type",
            "selected_variant_net_score",
            "qualified_activation",
            "qualified_activation_add",
            "positive",
            "neutral_valid",
            "neutral_waste",
            "neutral_waste_rate",
            "hard_negative",
            "rule_of_three_upper_failure_bound",
            "combined_family_coverage",
            "campaign_count",
            "counterfactual_value",
            "activated_gain",
            "ablation_loss",
            "reload_shadow_pass",
            "challenger_pass",
            "prune_pass",
            "mutation_attempts",
            "accepted_mutations",
            "rejected_mutations",
            "rollback_count",
        ]:
            if key in update:
                next_row[key if key != "wave2_outcome" else "e111_wave2_outcome"] = update[key]
        next_row["rank"] = update.get("rank_after", next_row["rank"])
        next_row["watch_state"] = "E111MutatedGoldConfirmed" if next_row["rank"] == "Gold" else next_row.get("watch_state")
        merged.append(next_row)
    return merged, {
        "summary": read_json(e111 / "summary.json"),
        "aggregate": read_json(e111 / "aggregate_metrics.json"),
        "promotion": read_json(e111 / "promotion_report.json"),
        "mutation": read_json(e111 / "mutation_summary.json"),
        "duration": read_json(e111 / "duration_report.json"),
    }


def merge_e112(rows: list[dict[str, Any]], e112: Path | None) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if not e112 or not (e112 / "wave_results.json").exists():
        return rows, None
    wave = read_json(e112 / "wave_results.json")["rows"]
    wave_by_id = {row["operator_id"]: row for row in wave}
    merged: list[dict[str, Any]] = []
    for row in rows:
        update = wave_by_id.get(row["operator_id"])
        if not update:
            merged.append(row)
            continue
        next_row = dict(row)
        for key in [
            "rank_after",
            "rank_before",
            "wave3_outcome",
            "selected_variant_id",
            "selected_variant_type",
            "selected_variant_net_score",
            "selected_prune_ratio",
            "qualified_activation",
            "qualified_activation_add",
            "positive",
            "neutral_valid",
            "neutral_waste",
            "neutral_waste_rate",
            "hard_negative",
            "rule_of_three_upper_failure_bound",
            "combined_family_coverage",
            "campaign_count",
            "counterfactual_value",
            "activated_gain",
            "ablation_loss",
            "reload_shadow_pass",
            "challenger_pass",
            "prune_pass",
            "long_horizon_no_harm_pass",
            "negative_scope_pass",
            "mutation_attempts",
            "accepted_mutations",
            "rejected_mutations",
            "rollback_count",
        ]:
            if key in update:
                next_row[key if key != "wave3_outcome" else "e112_wave3_outcome"] = update[key]
        next_row["rank"] = update.get("rank_after", next_row["rank"])
        next_row["watch_state"] = "E112CoreCandidateConfirmed" if next_row["rank"] == "CoreMemoryCandidate" else next_row.get("watch_state")
        merged.append(next_row)
    return merged, {
        "summary": read_json(e112 / "summary.json"),
        "aggregate": read_json(e112 / "aggregate_metrics.json"),
        "promotion": read_json(e112 / "promotion_report.json"),
        "mutation": read_json(e112 / "mutation_summary.json"),
        "duration": read_json(e112 / "duration_report.json"),
    }


def merge_e114(rows: list[dict[str, Any]], e114: Path | None) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if not e114 or not (e114 / "operator_projection_report.json").exists():
        return rows, None
    projection = read_json(e114 / "operator_projection_report.json")["rows"]
    projection_by_id = {row["operator_id"]: row for row in projection}
    merged: list[dict[str, Any]] = []
    for row in rows:
        update = projection_by_id.get(row["operator_id"])
        if not update:
            merged.append(row)
            continue
        next_row = dict(row)
        next_row["e114_current_run_calls"] = update.get("current_run_calls")
        next_row["e114_projected_full_fineweb_calls"] = update.get("projected_full_fineweb_calls")
        next_row["e114_projected_activation_after_full_fineweb"] = update.get("projected_activation_after_full_fineweb")
        next_row["e114_projected_reaches_permacore_probation"] = update.get("projected_reaches_permacore_probation")
        next_row["e114_projected_remaining_after_full_fineweb"] = update.get("projected_remaining_after_full_fineweb")
        next_row["e114_selected_variant"] = update.get("selected_variant")
        merged.append(next_row)
    return merged, {
        "summary": read_json(e114 / "summary.json"),
        "aggregate": read_json(e114 / "aggregate_metrics.json"),
        "target": read_json(e114 / "target_sufficiency_report.json"),
        "stability": read_json(e114 / "stability_trend_report.json"),
    }


def merge_e116(rows: list[dict[str, Any]], e116: Path | None) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if not e116 or not (e116 / "operator_target_coverage.json").exists():
        return rows, None
    coverage = read_json(e116 / "operator_target_coverage.json")["rows"]
    coverage_by_id = {row["operator_id"]: row for row in coverage}
    merged: list[dict[str, Any]] = []
    for row in rows:
        update = coverage_by_id.get(row["operator_id"])
        if not update:
            merged.append(row)
            continue
        next_row = dict(row)
        next_row["e116_template_family"] = update.get("template_family")
        next_row["e116_generated_cell_packs"] = update.get("generated_cell_packs")
        next_row["e116_variant_count"] = update.get("variant_count")
        next_row["e116_repeat_count_per_pack"] = update.get("repeat_count_per_pack")
        next_row["e116_qualified_synthetic_pressure_activation"] = update.get("qualified_synthetic_pressure_activation")
        next_row["e116_projected_activation_after_targeted_pressure"] = update.get("projected_activation_after_targeted_pressure")
        next_row["e116_reaches_permacore_probation_after_targeted_pressure"] = update.get("reaches_permacore_probation_after_targeted_pressure")
        merged.append(next_row)
    return merged, {
        "summary": read_json(e116 / "summary.json"),
        "aggregate": read_json(e116 / "aggregate_metrics.json"),
        "origin": read_json(e116 / "synthetic_origin_report.json"),
    }


def merge_e117(rows: list[dict[str, Any]], e117: Path | None) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if not e117 or not (e117 / "operator_gauntlet_results.json").exists():
        return rows, None
    results = read_json(e117 / "operator_gauntlet_results.json")["rows"]
    results_by_id = {row["operator_id"]: row for row in results}
    merged: list[dict[str, Any]] = []
    for row in rows:
        update = results_by_id.get(row["operator_id"])
        if not update:
            merged.append(row)
            continue
        next_row = dict(row)
        next_row["e117_qualified_activation"] = update.get("qualified_activation")
        next_row["e117_activation_after_gauntlet"] = update.get("activation_after_e117_gauntlet")
        next_row["e117_remaining_after_gauntlet"] = update.get("remaining_after_e117_gauntlet")
        next_row["e117_reaches_permacore_probation_after_gauntlet"] = update.get("reaches_permacore_probation_after_e117_gauntlet")
        next_row["e117_hard_negative"] = update.get("hard_negative")
        next_row["e117_positive_activation"] = update.get("positive_activation")
        next_row["e117_neutral_valid_activation"] = update.get("neutral_valid_activation")
        next_row["e117_negative_scope_valid_activation"] = update.get("negative_scope_valid_activation")
        merged.append(next_row)
    return merged, {
        "summary": read_json(e117 / "summary.json"),
        "aggregate": read_json(e117 / "aggregate_metrics.json"),
        "checker": read_json(e117 / "checker_summary.json") if (e117 / "checker_summary.json").exists() else None,
    }


def merge_e118(rows: list[dict[str, Any]], e118: Path | None) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if not e118 or not (e118 / "operator_cross_source_results.json").exists():
        return rows, None
    results = read_json(e118 / "operator_cross_source_results.json")["rows"]
    results_by_id = {row["operator_id"]: row for row in results}
    merged: list[dict[str, Any]] = []
    for row in rows:
        update = results_by_id.get(row["operator_id"])
        if not update:
            merged.append(row)
            continue
        next_row = dict(row)
        next_row["e118_cross_source_no_harm_pass"] = update.get("cross_source_no_harm_pass")
        next_row["e118_source_family_coverage"] = update.get("source_family_coverage")
        next_row["e118_case_count"] = update.get("case_count")
        next_row["e118_hard_negative"] = update.get("hard_negative_count")
        next_row["e118_negative_transfer"] = update.get("negative_transfer_count")
        next_row["e118_synthetic_imprint"] = update.get("synthetic_imprint_count")
        next_row["e118_ablation_value"] = update.get("ablation_value")
        merged.append(next_row)
    return merged, {
        "summary": read_json(e118 / "summary.json"),
        "aggregate": read_json(e118 / "aggregate_metrics.json"),
        "checker": read_json(e118 / "checker_summary.json") if (e118 / "checker_summary.json").exists() else None,
    }


def merge_e120(rows: list[dict[str, Any]], e120: Path | None) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if not e120 or not (e120 / "operator_gold_results.json").exists():
        return rows, None
    results = read_json(e120 / "operator_gold_results.json")["rows"]
    merged = list(rows)
    by_id = {row["operator_id"]: row for row in merged}
    for update in results:
        next_row = {
            "operator_id": update["operator_id"],
            "display_name": update.get("display_name", update["operator_id"]),
            "scope": update.get("scope"),
            "family": update.get("family"),
            "group_id": "E120",
            "e107_status": None,
            "e108_status": None,
            "rank": update.get("rank_after", update.get("lifecycle", "Gold")),
            "watch_state": "E120FineWebGoldConfirmed",
            "qualified_activation": update.get("qualified_activation", 0),
            "positive": update.get("positive", 0),
            "neutral_valid": update.get("neutral_valid", 0),
            "neutral_waste": update.get("neutral_waste", 0),
            "neutral_waste_rate": 0,
            "hard_negative": update.get("hard_negative", 0),
            "rule_of_three_upper_failure_bound": update.get("rule_of_three_upper_failure_bound"),
            "e107_family_coverage": None,
            "e108_family_coverage": None,
            "combined_family_coverage": update.get("family_coverage"),
            "campaign_count": update.get("campaign_count"),
            "counterfactual_value": update.get("selected_variant_net_score", 0),
            "activated_gain": update.get("selected_variant_utility", 0),
            "ablation_loss": 0,
            "reload_shadow_pass": update.get("reload_shadow_pass"),
            "challenger_pass": update.get("challenger_pass"),
            "prune_pass": update.get("prune_pass"),
            "rank_before": update.get("rank_before"),
            "rank_after": update.get("rank_after"),
            "selected_variant_id": update.get("selected_variant_id"),
            "selected_variant_type": update.get("selected_variant_type"),
            "selected_variant_net_score": update.get("selected_variant_net_score"),
            "selected_prune_ratio": update.get("selected_prune_ratio"),
            "long_horizon_no_harm_pass": update.get("negative_transfer", 0) == 0,
            "negative_scope_pass": update.get("negative_scope_pass"),
            "e120_origin": update.get("origin"),
            "e120_support_count": update.get("support_count"),
            "e120_saved_operator": True,
            "e120_gold_pass": update.get("gold_pass"),
            "e120_selected_variant_type": update.get("selected_variant_type"),
            "e120_selected_prune_ratio": update.get("selected_prune_ratio"),
            "e120_reload_shadow_pass": update.get("reload_shadow_pass"),
            "e120_negative_scope_pass": update.get("negative_scope_pass"),
            "e120_challenger_pass": update.get("challenger_pass"),
            "e120_prune_pass": update.get("prune_pass"),
            "e120_hard_negative": update.get("hard_negative", 0),
            "e120_wrong_scope_call": update.get("wrong_scope_call", 0),
            "e120_false_commit": update.get("false_commit", 0),
            "e120_unsupported_answer": update.get("unsupported_answer", 0),
            "e120_description": update.get("description"),
            "e120_promotion_reason": update.get("promotion_reason"),
        }
        existing = by_id.get(update["operator_id"])
        if existing:
            existing.update(next_row)
        else:
            merged.append(next_row)
            by_id[next_row["operator_id"]] = next_row
    return merged, {
        "summary": read_json(e120 / "summary.json"),
        "aggregate": read_json(e120 / "aggregate_metrics.json"),
        "promotion": read_json(e120 / "promotion_report.json"),
        "checker": read_json(e120 / "checker_summary.json") if (e120 / "checker_summary.json").exists() else None,
    }


def merge_e121(rows: list[dict[str, Any]], e121: Path | None) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if not e121 or not (e121 / "operator_orange_results.json").exists():
        return rows, None
    results = read_json(e121 / "operator_orange_results.json")["rows"]
    merged = list(rows)
    by_id = {row["operator_id"]: row for row in merged}
    for update in results:
        next_row = {
            "operator_id": update["operator_id"],
            "display_name": update.get("display_name", update["operator_id"]),
            "scope": update.get("scope"),
            "family": update.get("family"),
            "group_id": "E121",
            "rank": update.get("rank_after", "OrangeLegendaryCandidate"),
            "watch_state": update.get("watch_state", "E121OrangeLegendaryCandidateConfirmed"),
            "qualified_activation": update.get("qualified_activation"),
            "positive": update.get("positive"),
            "neutral_valid": update.get("neutral_valid"),
            "neutral_waste": update.get("neutral_waste"),
            "neutral_waste_rate": 0,
            "hard_negative": update.get("hard_negative"),
            "rule_of_three_upper_failure_bound": update.get("rule_of_three_upper_failure_bound"),
            "combined_family_coverage": update.get("family_coverage"),
            "campaign_count": update.get("campaign_count"),
            "counterfactual_value": update.get("selected_variant_net_score", 0),
            "activated_gain": update.get("selected_variant_utility", 0),
            "ablation_loss": 0,
            "reload_shadow_pass": update.get("reload_shadow_pass"),
            "challenger_pass": update.get("challenger_pass"),
            "prune_pass": update.get("prune_pass"),
            "rank_before": update.get("rank_before"),
            "rank_after": update.get("rank_after"),
            "selected_variant_id": update.get("selected_variant_id"),
            "selected_variant_type": update.get("selected_variant_type"),
            "selected_variant_net_score": update.get("selected_variant_net_score"),
            "selected_prune_ratio": update.get("selected_prune_ratio"),
            "long_horizon_no_harm_pass": update.get("no_harm_pass"),
            "negative_scope_pass": update.get("negative_scope_pass"),
            "mutation_attempts": update.get("mutation_attempts"),
            "accepted_mutations": update.get("accepted_mutations"),
            "rejected_mutations": update.get("rejected_mutations"),
            "rollback_count": update.get("rollback_count"),
            "e121_reaches_orange_legendary": update.get("e121_reaches_orange_legendary"),
            "e121_remaining_to_orange": update.get("e121_remaining_to_orange"),
            "e121_hard_negative": update.get("hard_negative"),
            "e121_wrong_scope_call": update.get("wrong_scope_call"),
            "e121_false_commit": update.get("false_commit"),
            "e121_unsupported_answer": update.get("unsupported_answer"),
            "e121_direct_flow_write": update.get("direct_flow_write"),
            "e121_selected_variant_type": update.get("selected_variant_type"),
            "e121_selected_prune_ratio": update.get("selected_prune_ratio"),
            "e121_family_coverage": update.get("family_coverage"),
            "e121_campaign_count": update.get("campaign_count"),
        }
        existing = by_id.get(update["operator_id"])
        if existing:
            existing.update(next_row)
        else:
            merged.append(next_row)
            by_id[next_row["operator_id"]] = next_row
    return merged, {
        "summary": read_json(e121 / "summary.json"),
        "aggregate": read_json(e121 / "aggregate_metrics.json"),
        "probation": read_json(e121 / "probation_report.json"),
        "checker": read_json(e121 / "checker_summary.json") if (e121 / "checker_summary.json").exists() else None,
    }


def merge_e122(rows: list[dict[str, Any]], e122: Path | None) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if not e122 or not (e122 / "orange_only_results.json").exists():
        return rows, None
    results = read_json(e122 / "orange_only_results.json")["rows"]
    cards = read_json(e122 / "negative_knowledge_cards.json")["rows"] if (e122 / "negative_knowledge_cards.json").exists() else []
    card_count_by_id: dict[str, int] = {}
    card_recall_by_id: dict[str, int] = {}
    card_prevent_by_id: dict[str, int] = {}
    false_block_by_id: dict[str, int] = {}
    for card in cards:
        oid = card["operator_id"]
        card_count_by_id[oid] = card_count_by_id.get(oid, 0) + 1
        card_recall_by_id[oid] = card_recall_by_id.get(oid, 0) + int(card.get("hit_count") or 0)
        card_prevent_by_id[oid] = card_prevent_by_id.get(oid, 0) + int(card.get("prevented_bad_attempts") or 0)
        false_block_by_id[oid] = false_block_by_id.get(oid, 0) + int(card.get("false_block_count") or 0)
    merged = list(rows)
    by_id = {row["operator_id"]: row for row in merged}
    for update in results:
        oid = update["operator_id"]
        next_row = {
            "operator_id": oid,
            "display_name": update.get("display_name", oid),
            "scope": update.get("scope"),
            "family": update.get("family"),
            "group_id": "E122",
            "rank": update.get("rank_after", "OrangeLegendaryCandidate"),
            "watch_state": update.get("watch_state", "E122OrangeOnlyBaselineConfirmed"),
            "qualified_activation": update.get("qualified_activation"),
            "positive": update.get("positive"),
            "neutral_valid": update.get("neutral_valid"),
            "neutral_waste": update.get("neutral_waste"),
            "neutral_waste_rate": update.get("neutral_waste_rate", 0),
            "hard_negative": update.get("hard_negative"),
            "rule_of_three_upper_failure_bound": update.get("rule_of_three_upper_failure_bound"),
            "combined_family_coverage": update.get("family_coverage"),
            "campaign_count": update.get("campaign_count"),
            "counterfactual_value": update.get("selected_variant_net_score", 0),
            "activated_gain": update.get("selected_variant_net_score", 0),
            "ablation_loss": 0,
            "reload_shadow_pass": update.get("reload_shadow_pass"),
            "challenger_pass": update.get("challenger_pass"),
            "prune_pass": update.get("prune_pass"),
            "rank_before": update.get("rank_before"),
            "rank_after": update.get("rank_after"),
            "selected_variant_id": update.get("selected_variant_id"),
            "selected_variant_type": update.get("selected_variant_type"),
            "selected_variant_net_score": update.get("selected_variant_net_score"),
            "selected_prune_ratio": update.get("selected_prune_ratio"),
            "long_horizon_no_harm_pass": update.get("long_horizon_no_harm_pass"),
            "negative_scope_pass": update.get("negative_scope_pass"),
            "mutation_attempts": update.get("mutation_attempts"),
            "accepted_mutations": update.get("accepted_mutations"),
            "rejected_mutations": update.get("rejected_mutations"),
            "rollback_count": update.get("rollback_count"),
            "e122_orange_only_baseline": update.get("e122_orange_only_baseline"),
            "e122_was_previously_orange": update.get("e122_was_previously_orange"),
            "e122_remaining_to_orange": update.get("e122_remaining_to_orange"),
            "e122_negative_card_count": card_count_by_id.get(oid, update.get("negative_card_count", 0)),
            "e122_negative_card_recall_count": card_recall_by_id.get(oid, update.get("negative_card_recall_count", 0)),
            "e122_prevented_repeat_failure_count": card_prevent_by_id.get(oid, update.get("prevented_repeat_failure_count", 0)),
            "e122_false_block_count": false_block_by_id.get(oid, update.get("false_block_count", 0)),
        }
        existing = by_id.get(oid)
        if existing:
            existing.update(next_row)
        else:
            merged.append(next_row)
            by_id[oid] = next_row
    return merged, {
        "summary": read_json(e122 / "summary.json"),
        "aggregate": read_json(e122 / "aggregate_metrics.json"),
        "usage": read_json(e122 / "negative_card_usage_report.json"),
        "checker": read_json(e122 / "checker_summary.json") if (e122 / "checker_summary.json").exists() else None,
    }


def merge_e127(rows: list[dict[str, Any]], e127: Path | None) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if not e127 or not (e127 / "cycles").exists():
        return rows, None
    result_files = sorted((e127 / "cycles").glob("cycle_*/operator_orange_results.json"))
    if not result_files:
        return rows, None
    merged = list(rows)
    by_id = {row["operator_id"]: row for row in merged}
    cycle_rows: list[dict[str, Any]] = []
    for path in result_files:
        try:
            cycle_index = int(path.parent.name.split("_")[-1])
        except ValueError:
            cycle_index = 0
        for update in read_json(path)["rows"]:
            if update.get("rank_after") != "OrangeLegendaryCandidate":
                continue
            row = {
                "operator_id": update["operator_id"],
                "display_name": update.get("display_name", update["operator_id"]),
                "scope": update.get("scope"),
                "family": update.get("family"),
                "group_id": "E127",
                "rank": update.get("rank_after", "OrangeLegendaryCandidate"),
                "watch_state": update.get("watch_state", "E127OrangeLegendaryCandidateConfirmed"),
                "qualified_activation": update.get("qualified_activation"),
                "positive": update.get("positive"),
                "neutral_valid": update.get("neutral_valid"),
                "neutral_waste": update.get("neutral_waste"),
                "neutral_waste_rate": 0,
                "hard_negative": update.get("hard_negative"),
                "rule_of_three_upper_failure_bound": update.get("rule_of_three_upper_failure_bound"),
                "combined_family_coverage": update.get("family_coverage"),
                "campaign_count": update.get("campaign_count"),
                "counterfactual_value": update.get("selected_variant_net_score", 0),
                "activated_gain": update.get("selected_variant_utility", 0),
                "ablation_loss": 0,
                "reload_shadow_pass": update.get("reload_shadow_pass"),
                "challenger_pass": update.get("challenger_pass"),
                "prune_pass": update.get("prune_pass"),
                "rank_before": update.get("rank_before"),
                "rank_after": update.get("rank_after"),
                "selected_variant_id": update.get("selected_variant_id"),
                "selected_variant_type": update.get("selected_variant_type"),
                "selected_variant_net_score": update.get("selected_variant_net_score"),
                "selected_prune_ratio": update.get("selected_prune_ratio"),
                "long_horizon_no_harm_pass": update.get("no_harm_pass"),
                "negative_scope_pass": update.get("negative_scope_pass"),
                "mutation_attempts": update.get("mutation_attempts"),
                "accepted_mutations": update.get("accepted_mutations"),
                "rejected_mutations": update.get("rejected_mutations"),
                "rollback_count": update.get("rollback_count"),
                "e127_cycle": cycle_index,
                "e127_reaches_orange_legendary": update.get("e127_reaches_orange_legendary"),
                "e127_remaining_to_orange": update.get("e127_remaining_to_orange"),
                "e127_hard_negative": update.get("hard_negative"),
                "e127_wrong_scope_call": update.get("wrong_scope_call"),
                "e127_false_commit": update.get("false_commit"),
                "e127_unsupported_answer": update.get("unsupported_answer"),
                "e127_direct_flow_write": update.get("direct_flow_write"),
                "e127_selected_variant_type": update.get("selected_variant_type"),
                "e127_selected_prune_ratio": update.get("selected_prune_ratio"),
                "e127_family_coverage": update.get("family_coverage"),
                "e127_campaign_count": update.get("campaign_count"),
                "e127_description": update.get("description"),
            }
            existing = by_id.get(update["operator_id"])
            if existing:
                existing.update(row)
            else:
                merged.append(row)
                by_id[row["operator_id"]] = row
            cycle_rows.append(row)
    aggregate = read_json(e127 / "aggregate_metrics.json") if (e127 / "aggregate_metrics.json").exists() else {}
    summary = read_json(e127 / "summary.json") if (e127 / "summary.json").exists() else aggregate
    return merged, {
        "summary": summary,
        "aggregate": aggregate,
        "operator_count": len(cycle_rows),
        "cycle_count": aggregate.get("cycle_count"),
    }


def merge_e129(rows: list[dict[str, Any]], e129: Path | None) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if not e129 or not (e129 / "operator_orange_results.json").exists():
        return rows, None
    results = read_json(e129 / "operator_orange_results.json")["rows"]
    merged = list(rows)
    by_id = {row["operator_id"]: row for row in merged}
    for update in results:
        row = {
            "operator_id": update["operator_id"],
            "display_name": update.get("display_name", update["operator_id"]),
            "scope": update.get("scope"),
            "family": update.get("family"),
            "group_id": "E129",
            "rank": update.get("rank_after", "OrangeLegendaryCandidate"),
            "watch_state": update.get("watch_state", "E129OrangeLegendaryCandidateConfirmed"),
            "qualified_activation": update.get("qualified_activation"),
            "positive": update.get("positive"),
            "neutral_valid": update.get("neutral_valid", 0),
            "neutral_waste": update.get("neutral_waste", 0),
            "neutral_waste_rate": update.get("neutral_waste_rate", 0),
            "hard_negative": update.get("hard_negative"),
            "rule_of_three_upper_failure_bound": update.get("rule_of_three_upper_failure_bound"),
            "combined_family_coverage": update.get("family_coverage"),
            "campaign_count": update.get("campaign_count"),
            "counterfactual_value": update.get("selected_variant_net_score", 0),
            "activated_gain": update.get("selected_variant_utility", 0),
            "ablation_loss": 0,
            "reload_shadow_pass": update.get("reload_shadow_pass"),
            "challenger_pass": update.get("challenger_pass"),
            "prune_pass": update.get("prune_pass"),
            "rank_before": update.get("rank_before"),
            "rank_after": update.get("rank_after"),
            "selected_variant_id": update.get("selected_variant_id"),
            "selected_variant_type": update.get("selected_variant_type"),
            "selected_variant_net_score": update.get("selected_variant_net_score"),
            "selected_prune_ratio": update.get("selected_prune_ratio"),
            "long_horizon_no_harm_pass": update.get("no_harm_pass"),
            "negative_scope_pass": update.get("negative_scope_pass_rate") == 1,
            "mutation_attempts": update.get("mutation_attempts"),
            "accepted_mutations": update.get("accepted_mutations"),
            "rejected_mutations": update.get("rejected_mutations"),
            "rollback_count": update.get("rollback_count"),
            "e129_arithmetic_trace_operator": True,
            "e129_min_in_scope_accuracy": update.get("min_in_scope_accuracy"),
            "e129_negative_scope_case_count": update.get("negative_scope_case_count"),
            "e129_negative_scope_pass_rate": update.get("negative_scope_pass_rate"),
            "e129_false_commit": update.get("false_commit"),
            "e129_wrong_scope_call": update.get("wrong_scope_call"),
            "e129_unsupported_answer": update.get("unsupported_answer"),
            "e129_selected_variant_cost": update.get("selected_variant_cost"),
            "e129_campaign_group_count": update.get("campaign_group_count"),
        }
        existing = by_id.get(update["operator_id"])
        if existing:
            existing.update(row)
        else:
            merged.append(row)
            by_id[row["operator_id"]] = row
    summary = read_json(e129 / "summary.json") if (e129 / "summary.json").exists() else {}
    decision = read_json(e129 / "decision.json") if (e129 / "decision.json").exists() else {}
    return merged, {
        "summary": summary,
        "decision": decision,
        "operator_count": len(results),
    }


def merge_e130a(rows: list[dict[str, Any]], e130a: Path | None) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if not e130a or not (e130a / "operator_orange_results.json").exists():
        return rows, None
    results = read_json(e130a / "operator_orange_results.json")["rows"]
    merged = list(rows)
    by_id = {row["operator_id"]: row for row in merged}
    for update in results:
        row = {
            "operator_id": update["operator_id"],
            "display_name": update.get("display_name", update["operator_id"]),
            "scope": update.get("scope"),
            "family": update.get("family"),
            "group_id": "E130A",
            "rank": update.get("rank_after", "OrangeLegendaryCandidate"),
            "watch_state": update.get("watch_state", "E130AOrangeLegendaryCandidateConfirmed"),
            "qualified_activation": update.get("qualified_activation"),
            "positive": update.get("positive"),
            "neutral_valid": update.get("neutral_valid", 0),
            "neutral_waste": update.get("neutral_waste", 0),
            "neutral_waste_rate": update.get("neutral_waste_rate", 0),
            "hard_negative": update.get("hard_negative"),
            "rule_of_three_upper_failure_bound": update.get("rule_of_three_upper_failure_bound"),
            "combined_family_coverage": update.get("family_coverage"),
            "campaign_count": update.get("campaign_count"),
            "counterfactual_value": update.get("selected_variant_net_score", 0),
            "activated_gain": update.get("selected_variant_utility", 0),
            "ablation_loss": 0,
            "reload_shadow_pass": update.get("reload_shadow_pass"),
            "challenger_pass": update.get("challenger_pass"),
            "prune_pass": update.get("prune_pass"),
            "rank_before": update.get("rank_before"),
            "rank_after": update.get("rank_after"),
            "selected_variant_id": update.get("selected_variant_id"),
            "selected_variant_type": update.get("selected_variant_type"),
            "selected_variant_net_score": update.get("selected_variant_net_score"),
            "selected_prune_ratio": update.get("selected_prune_ratio"),
            "long_horizon_no_harm_pass": update.get("long_horizon_no_harm_pass"),
            "negative_scope_pass": update.get("negative_scope_pass"),
            "mutation_attempts": update.get("mutation_attempts"),
            "accepted_mutations": update.get("accepted_mutations"),
            "rejected_mutations": update.get("rejected_mutations"),
            "rollback_count": update.get("rollback_count"),
            "e130a_reaches_orange_legendary": update.get("e130a_reaches_orange_legendary"),
            "e130a_source_rank": update.get("e130a_source_rank"),
            "e130a_activation_before": update.get("e130a_activation_before"),
            "e130a_activation_add": update.get("e130a_activation_add"),
            "e130a_remaining_to_orange": update.get("e130a_remaining_to_orange"),
            "e130a_hard_negative": update.get("hard_negative"),
            "e130a_wrong_scope_call": update.get("wrong_scope_call"),
            "e130a_false_commit": update.get("false_commit"),
            "e130a_unsupported_answer": update.get("unsupported_answer"),
            "e130a_direct_flow_write": update.get("direct_flow_write"),
            "e130a_negative_transfer": update.get("negative_transfer"),
            "e130a_pressure_family_count": update.get("e130a_pressure_family_count"),
        }
        existing = by_id.get(update["operator_id"])
        if existing:
            existing.update(row)
        else:
            merged.append(row)
            by_id[row["operator_id"]] = row
    summary = read_json(e130a / "summary.json") if (e130a / "summary.json").exists() else {}
    aggregate = read_json(e130a / "aggregate_metrics.json") if (e130a / "aggregate_metrics.json").exists() else {}
    decision = read_json(e130a / "decision.json") if (e130a / "decision.json").exists() else {}
    return merged, {
        "summary": summary,
        "aggregate": aggregate,
        "decision": decision,
        "operator_count": len(results),
    }


def merge_e130b(rows: list[dict[str, Any]], e130b: Path | None) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if not e130b or not (e130b / "operator_transfer_results.json").exists():
        return rows, None
    results = read_json(e130b / "operator_transfer_results.json")["rows"]
    merged = list(rows)
    by_id = {row["operator_id"]: row for row in merged}
    for update in results:
        row = {
            "operator_id": update["operator_id"],
            "display_name": update.get("display_name", update["operator_id"]),
            "scope": update.get("scope"),
            "family": update.get("family"),
            "group_id": "E130B",
            "rank": update.get("rank_after", "OrangeLegendaryCandidate"),
            "watch_state": update.get("watch_state", "E130BTextIOTransferConfirmed"),
            "qualified_activation": update.get("qualified_transfer_activation"),
            "positive": update.get("qualified_transfer_activation"),
            "hard_negative": update.get("hard_negative"),
            "rule_of_three_upper_failure_bound": update.get("rule_of_three_upper_failure_bound"),
            "rank_before": update.get("rank_before"),
            "rank_after": update.get("rank_after"),
            "reload_shadow_pass": update.get("reload_shadow_pass"),
            "negative_scope_pass": update.get("negative_scope_pass"),
            "challenger_pass": update.get("challenger_pass"),
            "prune_pass": update.get("prune_pass"),
            "e130b_text_io_transfer": update.get("transfer_pass"),
            "e130b_source_e129_rank": update.get("source_e129_rank"),
            "e130b_selected_route": update.get("selected_route"),
            "e130b_visible_transfer_case_count": update.get("visible_transfer_case_count"),
            "e130b_visible_transfer_accuracy": update.get("visible_transfer_accuracy"),
            "e130b_word_problem_no_call_case_count": update.get("word_problem_no_call_case_count"),
            "e130b_word_problem_no_call_accuracy": update.get("word_problem_no_call_accuracy"),
            "e130b_qualified_transfer_activation": update.get("qualified_transfer_activation"),
            "e130b_hard_negative": update.get("hard_negative"),
            "e130b_wrong_scope_call": update.get("wrong_scope_call"),
            "e130b_false_commit": update.get("false_commit"),
            "e130b_unsupported_answer": update.get("unsupported_answer"),
            "e130b_direct_flow_write": update.get("direct_flow_write"),
            "e130b_overbroad_control_wrong_scope_call": update.get("overbroad_control_wrong_scope_call"),
        }
        existing = by_id.get(update["operator_id"])
        if existing:
            existing.update(row)
        else:
            merged.append(row)
            by_id[row["operator_id"]] = row
    summary = read_json(e130b / "summary.json") if (e130b / "summary.json").exists() else {}
    aggregate = read_json(e130b / "aggregate_metrics.json") if (e130b / "aggregate_metrics.json").exists() else {}
    decision = read_json(e130b / "decision.json") if (e130b / "decision.json").exists() else {}
    return merged, {
        "summary": summary,
        "aggregate": aggregate,
        "decision": decision,
        "operator_count": len(results),
    }


def merge_e131(rows: list[dict[str, Any]], e131: Path | None) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if not e131 or not (e131 / "operator_transfer_results.json").exists():
        return rows, None
    results = read_json(e131 / "operator_transfer_results.json")["rows"]
    merged = list(rows)
    by_id = {row["operator_id"]: row for row in merged}
    for update in results:
        row = {
            "operator_id": update["operator_id"],
            "display_name": update.get("display_name", update["operator_id"]),
            "scope": update.get("scope"),
            "family": update.get("family"),
            "group_id": "E131",
            "rank": update.get("rank_after", "OrangeLegendaryCandidate"),
            "watch_state": update.get("watch_state", "E131VisibleEquationAssistantRenderConfirmed"),
            "qualified_activation": update.get("qualified_visible_activation"),
            "positive": update.get("qualified_visible_activation"),
            "hard_negative": update.get("hard_negative"),
            "rule_of_three_upper_failure_bound": update.get("rule_of_three_upper_failure_bound"),
            "rank_before": update.get("rank_before"),
            "rank_after": update.get("rank_after"),
            "reload_shadow_pass": update.get("reload_shadow_pass"),
            "negative_scope_pass": update.get("negative_scope_pass"),
            "challenger_pass": update.get("challenger_pass"),
            "prune_pass": update.get("prune_pass"),
            "e131_visible_equation_transfer": update.get("transfer_pass"),
            "e131_selected_route": update.get("selected_route"),
            "e131_visible_equation_case_count": update.get("visible_equation_case_count"),
            "e131_visible_equation_extraction_accuracy": update.get("visible_equation_extraction_accuracy"),
            "e131_word_problem_no_call_case_count": update.get("word_problem_no_call_case_count"),
            "e131_word_problem_no_call_accuracy": update.get("word_problem_no_call_accuracy"),
            "e131_qualified_visible_activation": update.get("qualified_visible_activation"),
            "e131_hard_negative": update.get("hard_negative"),
            "e131_wrong_scope_call": update.get("wrong_scope_call"),
            "e131_false_commit": update.get("false_commit"),
            "e131_unsupported_answer": update.get("unsupported_answer"),
            "e131_boundary_claim_violation": update.get("boundary_claim_violation"),
            "e131_direct_flow_write": update.get("direct_flow_write"),
            "e131_e130b_baseline_visible_miss": update.get("e130b_baseline_visible_miss"),
            "e131_overbroad_control_wrong_scope_call": update.get("overbroad_control_wrong_scope_call"),
        }
        existing = by_id.get(update["operator_id"])
        if existing:
            existing.update(row)
        else:
            merged.append(row)
            by_id[row["operator_id"]] = row
    summary = read_json(e131 / "summary.json") if (e131 / "summary.json").exists() else {}
    aggregate = read_json(e131 / "aggregate_metrics.json") if (e131 / "aggregate_metrics.json").exists() else {}
    decision = read_json(e131 / "decision.json") if (e131 / "decision.json").exists() else {}
    dataset = read_json(e131 / "dataset_report.json") if (e131 / "dataset_report.json").exists() else {}
    return merged, {
        "summary": summary,
        "aggregate": aggregate,
        "decision": decision,
        "dataset": dataset,
        "operator_count": len(results),
    }


def merge_e132(rows: list[dict[str, Any]], e132: Path | None) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if not e132 or not (e132 / "operator_orange_results.json").exists():
        return rows, None
    results = read_json(e132 / "operator_orange_results.json")["rows"]
    merged = list(rows)
    by_id = {row["operator_id"]: row for row in merged}
    for update in results:
        row = {
            "operator_id": update["operator_id"],
            "display_name": update.get("display_name", update["operator_id"]),
            "scope": update.get("scope"),
            "family": update.get("family"),
            "group_id": "E132",
            "rank": update.get("rank_after", "OrangeLegendaryCandidate"),
            "watch_state": update.get("watch_state", "E132ExternalMathTextOrangeCycleConfirmed"),
            "qualified_activation": update.get("qualified_activation"),
            "qualified_activation_add": update.get("qualified_activation_add"),
            "positive": update.get("positive"),
            "neutral_valid": update.get("neutral_valid"),
            "neutral_waste": update.get("neutral_waste"),
            "neutral_waste_rate": update.get("neutral_waste_rate"),
            "hard_negative": update.get("hard_negative"),
            "rule_of_three_upper_failure_bound": update.get("rule_of_three_upper_failure_bound"),
            "rank_before": update.get("rank_before"),
            "rank_after": update.get("rank_after"),
            "reload_shadow_pass": update.get("reload_shadow_pass"),
            "negative_scope_pass": update.get("negative_scope_pass"),
            "challenger_pass": update.get("challenger_pass"),
            "prune_pass": update.get("prune_pass"),
            "selected_variant_id": update.get("selected_variant_id"),
            "selected_variant_type": update.get("selected_variant_type"),
            "selected_variant_net_score": update.get("selected_variant_net_score"),
            "selected_prune_ratio": update.get("selected_prune_ratio"),
            "mutation_attempts": update.get("mutation_attempts"),
            "accepted_mutations": update.get("accepted_mutations"),
            "rejected_mutations": update.get("rejected_mutations"),
            "rollback_count": update.get("rollback_count"),
            "campaign_count": update.get("campaign_count"),
            "e132_math_text_skill_operator": update.get("e132_math_text_skill_operator"),
            "e132_reaches_orange_legendary": update.get("e132_reaches_orange_legendary"),
            "e132_external_support_count": update.get("external_support_count"),
            "e132_external_source_count": update.get("external_source_count"),
            "e132_external_family_count": update.get("external_family_count"),
            "e132_negative_scope_case_count": update.get("negative_scope_case_count"),
            "e132_negative_scope_pass_rate": update.get("negative_scope_pass_rate"),
            "e132_hard_negative": update.get("hard_negative"),
            "e132_wrong_scope_call": update.get("wrong_scope_call"),
            "e132_false_commit": update.get("false_commit"),
            "e132_unsupported_answer": update.get("unsupported_answer"),
            "e132_boundary_claim_violation": update.get("boundary_claim_violation"),
            "e132_direct_flow_write": update.get("direct_flow_write"),
            "e132_overbroad_solver_control_wrong_scope_call": update.get("overbroad_solver_control_wrong_scope_call"),
            "e132_pressure_family_count": update.get("pressure_family_count"),
            "e132_selected_variant_type": update.get("selected_variant_type"),
            "e132_selected_prune_ratio": update.get("selected_prune_ratio"),
        }
        existing = by_id.get(update["operator_id"])
        if existing:
            existing.update(row)
        else:
            merged.append(row)
            by_id[row["operator_id"]] = row
    summary = read_json(e132 / "summary.json") if (e132 / "summary.json").exists() else {}
    aggregate = read_json(e132 / "aggregate_metrics.json") if (e132 / "aggregate_metrics.json").exists() else {}
    decision = read_json(e132 / "decision.json") if (e132 / "decision.json").exists() else {}
    dataset = read_json(e132 / "dataset_report.json") if (e132 / "dataset_report.json").exists() else {}
    return merged, {
        "summary": summary,
        "aggregate": aggregate,
        "decision": decision,
        "dataset": dataset,
        "operator_count": len(results),
    }


def merge_e133(rows: list[dict[str, Any]], e133: Path | None) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if not e133 or not (e133 / "operator_route_results.json").exists():
        return rows, None
    results = read_json(e133 / "operator_route_results.json")["rows"]
    merged = list(rows)
    by_id = {row["operator_id"]: row for row in merged}
    for update in results:
        row = {
            "operator_id": update["operator_id"],
            "display_name": update.get("display_name", update["operator_id"]),
            "scope": update.get("scope"),
            "family": update.get("family"),
            "group_id": "E133",
            "rank": update.get("rank_after", "OrangeLegendaryCandidate"),
            "watch_state": update.get("watch_state", "E133MathTextRouteCompositionConfirmed"),
            "rank_before": update.get("rank_before"),
            "rank_after": update.get("rank_after"),
            "reload_shadow_pass": update.get("reload_shadow_pass"),
            "negative_scope_pass": update.get("negative_scope_pass"),
            "challenger_pass": update.get("challenger_pass"),
            "prune_pass": update.get("prune_pass"),
            "selected_route": update.get("selected_route"),
            "e133_math_text_route_composition": update.get("e133_math_text_route_composition"),
            "e133_composition_pass": update.get("composition_pass"),
            "e133_route_case_count": update.get("route_case_count"),
            "e133_route_accuracy": update.get("route_accuracy"),
            "e133_visible_arithmetic_route_case_count": update.get("visible_arithmetic_route_case_count"),
            "e133_visible_arithmetic_route_accuracy": update.get("visible_arithmetic_route_accuracy"),
            "e133_structural_guard_case_count": update.get("structural_guard_case_count"),
            "e133_structural_guard_accuracy": update.get("structural_guard_accuracy"),
            "e133_hidden_word_problem_no_solve_case_count": update.get("hidden_word_problem_no_solve_case_count"),
            "e133_hidden_word_problem_no_solve_accuracy": update.get("hidden_word_problem_no_solve_accuracy"),
            "e133_qualified_route_activation": update.get("qualified_route_activation"),
            "e133_hard_negative": update.get("hard_negative"),
            "e133_wrong_scope_call": update.get("wrong_scope_call"),
            "e133_false_commit": update.get("false_commit"),
            "e133_unsupported_answer": update.get("unsupported_answer"),
            "e133_boundary_claim_violation": update.get("boundary_claim_violation"),
            "e133_direct_flow_write": update.get("direct_flow_write"),
            "e133_overbroad_solver_control_wrong_scope_call": update.get("overbroad_solver_control_wrong_scope_call"),
            "e133_trust_control_false_commit": update.get("trust_control_false_commit"),
            "e133_trust_control_direct_flow_write": update.get("trust_control_direct_flow_write"),
        }
        existing = by_id.get(update["operator_id"])
        if existing:
            existing.update(row)
        else:
            row["qualified_activation"] = update.get("qualified_route_activation")
            row["positive"] = update.get("qualified_route_activation")
            row["hard_negative"] = update.get("hard_negative")
            row["rule_of_three_upper_failure_bound"] = update.get("rule_of_three_upper_failure_bound")
            merged.append(row)
            by_id[row["operator_id"]] = row
    summary = read_json(e133 / "summary.json") if (e133 / "summary.json").exists() else {}
    aggregate = read_json(e133 / "aggregate_metrics.json") if (e133 / "aggregate_metrics.json").exists() else {}
    decision = read_json(e133 / "decision.json") if (e133 / "decision.json").exists() else {}
    dataset = read_json(e133 / "dataset_route_seed_report.json") if (e133 / "dataset_route_seed_report.json").exists() else {}
    return merged, {
        "summary": summary,
        "aggregate": aggregate,
        "decision": decision,
        "dataset": dataset,
        "operator_count": len(results),
    }


def merge_e134(rows: list[dict[str, Any]], e134: Path | None) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if not e134 or not (e134 / "operator_ood_results.json").exists():
        return rows, None
    results = read_json(e134 / "operator_ood_results.json")["rows"]
    merged = list(rows)
    by_id = {row["operator_id"]: row for row in merged}
    for update in results:
        row = {
            "operator_id": update["operator_id"],
            "display_name": update.get("display_name", update["operator_id"]),
            "scope": update.get("scope"),
            "family": update.get("family"),
            "group_id": "E134",
            "rank": update.get("rank_after", "OrangeLegendaryCandidate"),
            "watch_state": update.get("watch_state", "E134OODRouteStressConfirmed"),
            "rank_before": update.get("rank_before"),
            "rank_after": update.get("rank_after"),
            "reload_shadow_pass": update.get("reload_shadow_pass"),
            "negative_scope_pass": update.get("negative_scope_pass"),
            "challenger_pass": update.get("challenger_pass"),
            "prune_pass": update.get("prune_pass"),
            "selected_route": update.get("selected_route"),
            "e134_external_math_text_ood_route_stress": update.get("e134_external_math_text_ood_route_stress"),
            "e134_ood_pass": update.get("ood_pass"),
            "e134_ood_case_count": update.get("ood_case_count"),
            "e134_ood_route_accuracy": update.get("ood_route_accuracy"),
            "e134_visible_arithmetic_ood_case_count": update.get("visible_arithmetic_ood_case_count"),
            "e134_visible_arithmetic_ood_accuracy": update.get("visible_arithmetic_ood_accuracy"),
            "e134_structural_guard_ood_case_count": update.get("structural_guard_ood_case_count"),
            "e134_structural_guard_ood_accuracy": update.get("structural_guard_ood_accuracy"),
            "e134_hidden_word_problem_ood_no_solve_case_count": update.get("hidden_word_problem_ood_no_solve_case_count"),
            "e134_hidden_word_problem_ood_no_solve_accuracy": update.get("hidden_word_problem_ood_no_solve_accuracy"),
            "e134_counterexample_case_count": update.get("counterexample_case_count"),
            "e134_counterexample_accuracy": update.get("counterexample_accuracy"),
            "e134_qualified_ood_route_activation": update.get("qualified_ood_route_activation"),
            "e134_hard_negative": update.get("hard_negative"),
            "e134_wrong_scope_call": update.get("wrong_scope_call"),
            "e134_false_commit": update.get("false_commit"),
            "e134_unsupported_answer": update.get("unsupported_answer"),
            "e134_boundary_claim_violation": update.get("boundary_claim_violation"),
            "e134_direct_flow_write": update.get("direct_flow_write"),
            "e134_e133_baseline_ood_miss": update.get("e133_baseline_ood_miss"),
            "e134_overbroad_solver_control_wrong_scope_call": update.get("overbroad_solver_control_wrong_scope_call"),
            "e134_trust_control_false_commit": update.get("trust_control_false_commit"),
            "e134_trust_control_direct_flow_write": update.get("trust_control_direct_flow_write"),
        }
        existing = by_id.get(update["operator_id"])
        if existing:
            existing.update(row)
        else:
            row["qualified_activation"] = update.get("qualified_ood_route_activation")
            row["positive"] = update.get("qualified_ood_route_activation")
            row["hard_negative"] = update.get("hard_negative")
            row["rule_of_three_upper_failure_bound"] = update.get("rule_of_three_upper_failure_bound")
            merged.append(row)
            by_id[row["operator_id"]] = row
    summary = read_json(e134 / "summary.json") if (e134 / "summary.json").exists() else {}
    aggregate = read_json(e134 / "aggregate_metrics.json") if (e134 / "aggregate_metrics.json").exists() else {}
    decision = read_json(e134 / "decision.json") if (e134 / "decision.json").exists() else {}
    dataset = read_json(e134 / "dataset_ood_seed_report.json") if (e134 / "dataset_ood_seed_report.json").exists() else {}
    return merged, {
        "summary": summary,
        "aggregate": aggregate,
        "decision": decision,
        "dataset": dataset,
        "operator_count": len(results),
    }


def merge_e135(rows: list[dict[str, Any]], e135: Path | None) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if not e135 or not (e135 / "operator_dialogue_results.json").exists():
        return rows, None
    results = read_json(e135 / "operator_dialogue_results.json")["rows"]
    merged = list(rows)
    by_id = {row["operator_id"]: row for row in merged}
    for update in results:
        row = {
            "operator_id": update["operator_id"],
            "display_name": update.get("display_name", update["operator_id"]),
            "scope": update.get("scope"),
            "family": update.get("family"),
            "group_id": "E135",
            "rank": update.get("rank_after", "OrangeLegendaryCandidate"),
            "watch_state": update.get("watch_state", "E135DialogueStateConfirmed"),
            "rank_before": update.get("rank_before"),
            "rank_after": update.get("rank_after"),
            "reload_shadow_pass": update.get("reload_shadow_pass"),
            "negative_scope_pass": update.get("negative_scope_pass"),
            "challenger_pass": update.get("challenger_pass"),
            "prune_pass": update.get("prune_pass"),
            "selected_route": update.get("selected_route"),
            "e135_math_text_multi_route_dialogue_state": update.get("e135_math_text_multi_route_dialogue_state"),
            "e135_dialogue_pass": update.get("dialogue_pass"),
            "e135_dialogue_case_count": update.get("dialogue_case_count"),
            "e135_dialogue_turn_count": update.get("dialogue_turn_count"),
            "e135_dialogue_state_accuracy": update.get("dialogue_state_accuracy"),
            "e135_current_turn_route_accuracy": update.get("current_turn_route_accuracy"),
            "e135_route_state_integrity": update.get("route_state_integrity"),
            "e135_all_turn_route_accuracy": update.get("all_turn_route_accuracy"),
            "e135_hidden_word_problem_dialogue_no_solve_case_count": update.get("hidden_word_problem_dialogue_no_solve_case_count"),
            "e135_hidden_word_problem_dialogue_no_solve_accuracy": update.get("hidden_word_problem_dialogue_no_solve_accuracy"),
            "e135_visible_reentry_dialogue_case_count": update.get("visible_reentry_dialogue_case_count"),
            "e135_visible_reentry_dialogue_accuracy": update.get("visible_reentry_dialogue_accuracy"),
            "e135_stale_route_rejection_case_count": update.get("stale_route_rejection_case_count"),
            "e135_stale_route_rejection_accuracy": update.get("stale_route_rejection_accuracy"),
            "e135_cross_thread_rejection_case_count": update.get("cross_thread_rejection_case_count"),
            "e135_cross_thread_rejection_accuracy": update.get("cross_thread_rejection_accuracy"),
            "e135_counterexample_dialogue_case_count": update.get("counterexample_dialogue_case_count"),
            "e135_counterexample_dialogue_accuracy": update.get("counterexample_dialogue_accuracy"),
            "e135_qualified_dialogue_route_activation": update.get("qualified_dialogue_route_activation"),
            "e135_hard_negative": update.get("hard_negative"),
            "e135_wrong_scope_call": update.get("wrong_scope_call"),
            "e135_false_commit": update.get("false_commit"),
            "e135_unsupported_answer": update.get("unsupported_answer"),
            "e135_boundary_claim_violation": update.get("boundary_claim_violation"),
            "e135_direct_flow_write": update.get("direct_flow_write"),
            "e135_stale_route_reuse": update.get("stale_route_reuse"),
            "e135_cross_thread_contamination": update.get("cross_thread_contamination"),
            "e135_latest_route_reuse_control_failure": update.get("latest_route_reuse_control_failure"),
            "e135_stale_route_reuse_control_failure": update.get("stale_route_reuse_control_failure"),
            "e135_cross_thread_contamination_control_failure": update.get("cross_thread_contamination_control_failure"),
            "e135_counterexample_trust_control_failure": update.get("counterexample_trust_control_failure"),
            "e135_single_turn_reset_control_failure": update.get("single_turn_reset_control_failure"),
        }
        existing = by_id.get(update["operator_id"])
        if existing:
            existing.update(row)
        else:
            row["qualified_activation"] = update.get("qualified_dialogue_route_activation")
            row["positive"] = update.get("qualified_dialogue_route_activation")
            row["hard_negative"] = update.get("hard_negative")
            row["rule_of_three_upper_failure_bound"] = update.get("rule_of_three_upper_failure_bound")
            merged.append(row)
            by_id[row["operator_id"]] = row
    summary = read_json(e135 / "summary.json") if (e135 / "summary.json").exists() else {}
    aggregate = read_json(e135 / "aggregate_metrics.json") if (e135 / "aggregate_metrics.json").exists() else {}
    decision = read_json(e135 / "decision.json") if (e135 / "decision.json").exists() else {}
    dataset = read_json(e135 / "dataset_dialogue_seed_report.json") if (e135 / "dataset_dialogue_seed_report.json").exists() else {}
    return merged, {
        "summary": summary,
        "aggregate": aggregate,
        "decision": decision,
        "dataset": dataset,
        "operator_count": len(results),
    }


def merge_e136a(rows: list[dict[str, Any]], e136a: Path | None) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if not e136a or not (e136a / "operator_orange_results.json").exists():
        return rows, None
    results = read_json(e136a / "operator_orange_results.json")["rows"]
    merged = list(rows)
    by_id = {row["operator_id"]: row for row in merged}
    for update in results:
        row = {
            "operator_id": update["operator_id"],
            "display_name": update.get("display_name", update["operator_id"]),
            "scope": update.get("scope"),
            "family": update.get("family"),
            "group_id": "E136A",
            "rank": update.get("rank_after", "OrangeLegendaryCandidate"),
            "watch_state": update.get("watch_state", "E136AAssistantTextOrangeCycleConfirmed"),
            "rank_before": update.get("rank_before"),
            "rank_after": update.get("rank_after"),
            "qualified_activation": update.get("qualified_activation"),
            "positive": update.get("positive"),
            "neutral_valid": update.get("neutral_valid"),
            "neutral_waste": update.get("neutral_waste"),
            "neutral_waste_rate": update.get("neutral_waste_rate"),
            "hard_negative": update.get("hard_negative"),
            "rule_of_three_upper_failure_bound": update.get("rule_of_three_upper_failure_bound"),
            "reload_shadow_pass": update.get("reload_shadow_pass"),
            "negative_scope_pass": update.get("negative_scope_pass"),
            "challenger_pass": update.get("challenger_pass"),
            "prune_pass": update.get("prune_pass"),
            "selected_variant_id": update.get("selected_variant_id"),
            "selected_variant_type": update.get("selected_variant_type"),
            "selected_variant_net_score": update.get("selected_variant_net_score"),
            "selected_prune_ratio": update.get("selected_prune_ratio"),
            "mutation_attempts": update.get("mutation_attempts"),
            "accepted_mutations": update.get("accepted_mutations"),
            "rejected_mutations": update.get("rejected_mutations"),
            "rollback_count": update.get("rollback_count"),
            "e136a_assistant_text_skill_operator": update.get("e136a_assistant_text_skill_operator"),
            "e136a_reaches_orange_legendary": update.get("e136a_reaches_orange_legendary"),
            "e136a_external_support_count": update.get("external_support_count"),
            "e136a_external_source_count": update.get("external_source_count"),
            "e136a_external_family_count": update.get("external_family_count"),
            "e136a_external_license_count": update.get("external_license_count"),
            "e136a_negative_scope_case_count": update.get("negative_scope_case_count"),
            "e136a_negative_scope_pass_rate": update.get("negative_scope_pass_rate"),
            "e136a_overbroad_chatbot_control_wrong_scope_call": update.get("overbroad_chatbot_control_wrong_scope_call"),
            "e136a_hard_negative": update.get("hard_negative"),
            "e136a_wrong_scope_call": update.get("wrong_scope_call"),
            "e136a_false_commit": update.get("false_commit"),
            "e136a_unsupported_answer": update.get("unsupported_answer"),
            "e136a_boundary_claim_violation": update.get("boundary_claim_violation"),
            "e136a_direct_flow_write": update.get("direct_flow_write"),
            "e136a_selected_variant_type": update.get("selected_variant_type"),
            "e136a_selected_prune_ratio": update.get("selected_prune_ratio"),
            "e136a_pressure_family_count": update.get("pressure_family_count"),
            "e136a_description": update.get("description"),
        }
        existing = by_id.get(update["operator_id"])
        if existing:
            existing.update(row)
        else:
            row["qualified_activation_add"] = update.get("qualified_activation_add")
            row["campaign_count"] = update.get("campaign_count")
            row["combined_family_coverage"] = update.get("family_coverage")
            merged.append(row)
            by_id[row["operator_id"]] = row
    summary = read_json(e136a / "summary.json") if (e136a / "summary.json").exists() else {}
    aggregate = read_json(e136a / "aggregate_metrics.json") if (e136a / "aggregate_metrics.json").exists() else {}
    decision = read_json(e136a / "decision.json") if (e136a / "decision.json").exists() else {}
    dataset = read_json(e136a / "dataset_report.json") if (e136a / "dataset_report.json").exists() else {}
    return merged, {
        "summary": summary,
        "aggregate": aggregate,
        "decision": decision,
        "dataset": dataset,
        "operator_count": len(results),
    }


def merge_e136b(rows: list[dict[str, Any]], e136b: Path | None) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if not e136b or not (e136b / "operator_route_results.json").exists():
        return rows, None
    results = read_json(e136b / "operator_route_results.json")["rows"]
    merged = list(rows)
    by_id = {row["operator_id"]: row for row in merged}
    for update in results:
        existing = by_id.get(update["operator_id"])
        row = {
            "operator_id": update["operator_id"],
            "display_name": update.get("display_name", update["operator_id"]),
            "scope": update.get("scope") or (existing or {}).get("scope"),
            "family": update.get("family") or (existing or {}).get("family"),
            "group_id": "E136B",
            "rank": update.get("rank_after", "OrangeLegendaryCandidate"),
            "watch_state": update.get("watch_state", "E136BAssistantTextRouteBoundaryConfirmed"),
            "rank_before": update.get("rank_before"),
            "rank_after": update.get("rank_after"),
            "qualified_activation": update.get("qualified_route_activation"),
            "positive": update.get("qualified_route_activation"),
            "hard_negative": update.get("hard_negative"),
            "rule_of_three_upper_failure_bound": update.get("rule_of_three_upper_failure_bound"),
            "reload_shadow_pass": update.get("reload_shadow_pass"),
            "negative_scope_pass": update.get("negative_scope_pass"),
            "challenger_pass": update.get("challenger_pass"),
            "prune_pass": update.get("prune_pass"),
            "selected_route": update.get("selected_route"),
            "e136b_assistant_text_route_composition": update.get("e136b_assistant_text_route_composition"),
            "e136b_route_pass": update.get("route_pass"),
            "e136b_route_case_count": update.get("route_case_count"),
            "e136b_route_accuracy": update.get("route_accuracy"),
            "e136b_route_stack_accuracy": update.get("route_stack_accuracy"),
            "e136b_primary_route_accuracy": update.get("primary_route_accuracy"),
            "e136b_boundary_accuracy": update.get("boundary_accuracy"),
            "e136b_multi_route_composition_case_count": update.get("multi_route_composition_case_count"),
            "e136b_multi_route_composition_accuracy": update.get("multi_route_composition_accuracy"),
            "e136b_boundary_case_count": update.get("boundary_case_count"),
            "e136b_boundary_case_accuracy": update.get("boundary_case_accuracy"),
            "e136b_negative_scope_case_count": update.get("negative_scope_case_count"),
            "e136b_negative_scope_accuracy": update.get("negative_scope_accuracy"),
            "e136b_qualified_route_activation": update.get("qualified_route_activation"),
            "e136b_hard_negative": update.get("hard_negative"),
            "e136b_wrong_scope_call": update.get("wrong_scope_call"),
            "e136b_false_commit": update.get("false_commit"),
            "e136b_unsupported_answer": update.get("unsupported_answer"),
            "e136b_boundary_claim_violation": update.get("boundary_claim_violation"),
            "e136b_direct_flow_write": update.get("direct_flow_write"),
            "e136b_overbroad_chatbot_control_wrong_scope_call": update.get("overbroad_chatbot_control_wrong_scope_call"),
            "e136b_overbroad_chatbot_control_unsupported_answer": update.get("overbroad_chatbot_control_unsupported_answer"),
            "e136b_unsafe_direct_write_control_false_commit": update.get("unsafe_direct_write_control_false_commit"),
            "e136b_unsafe_direct_write_control_direct_flow_write": update.get("unsafe_direct_write_control_direct_flow_write"),
            "e136b_source_hallucination_control_false_commit": update.get("source_hallucination_control_false_commit"),
            "e136b_source_hallucination_control_unsupported_answer": update.get("source_hallucination_control_unsupported_answer"),
            "e136b_rejected_response_reuse_control_false_commit": update.get("rejected_response_reuse_control_false_commit"),
            "e136b_single_operator_drop_control_false_commit": update.get("single_operator_drop_control_false_commit"),
            "e136b_route_description": update.get("route_description"),
        }
        if existing:
            existing.update(row)
        else:
            merged.append(row)
            by_id[row["operator_id"]] = row
    summary = read_json(e136b / "summary.json") if (e136b / "summary.json").exists() else {}
    aggregate = read_json(e136b / "aggregate_metrics.json") if (e136b / "aggregate_metrics.json").exists() else {}
    decision = read_json(e136b / "decision.json") if (e136b / "decision.json").exists() else {}
    dataset = read_json(e136b / "dataset_route_seed_report.json") if (e136b / "dataset_route_seed_report.json").exists() else {}
    return merged, {
        "summary": summary,
        "aggregate": aggregate,
        "decision": decision,
        "dataset": dataset,
        "operator_count": len(results),
    }


def build_payload(
    e109: Path,
    e110: Path | None = None,
    e111: Path | None = None,
    e112: Path | None = None,
    e114: Path | None = None,
    e116: Path | None = None,
    e117: Path | None = None,
    e118: Path | None = None,
    e120: Path | None = None,
    e121: Path | None = None,
    e122: Path | None = None,
    e127: Path | None = None,
    e129: Path | None = None,
    e130a: Path | None = None,
    e130b: Path | None = None,
    e131: Path | None = None,
    e132: Path | None = None,
    e133: Path | None = None,
    e134: Path | None = None,
    e135: Path | None = None,
    e136a: Path | None = None,
    e136b: Path | None = None,
) -> dict[str, Any]:
    rank_results = read_json(e109 / "rank_results.json")
    rows, e110_payload = merge_e110(compact_rows(rank_results["rows"]), e110)
    rows, e111_payload = merge_e111(rows, e111)
    rows, e112_payload = merge_e112(rows, e112)
    rows, e114_payload = merge_e114(rows, e114)
    rows, e116_payload = merge_e116(rows, e116)
    rows, e117_payload = merge_e117(rows, e117)
    rows, e118_payload = merge_e118(rows, e118)
    rows, e120_payload = merge_e120(rows, e120)
    rows, e121_payload = merge_e121(rows, e121)
    rows, e122_payload = merge_e122(rows, e122)
    rows, e127_payload = merge_e127(rows, e127)
    rows, e129_payload = merge_e129(rows, e129)
    rows, e130a_payload = merge_e130a(rows, e130a)
    rows, e130b_payload = merge_e130b(rows, e130b)
    rows, e131_payload = merge_e131(rows, e131)
    rows, e132_payload = merge_e132(rows, e132)
    rows, e133_payload = merge_e133(rows, e133)
    rows, e134_payload = merge_e134(rows, e134)
    rows, e135_payload = merge_e135(rows, e135)
    rows, e136a_payload = merge_e136a(rows, e136a)
    rows, e136b_payload = merge_e136b(rows, e136b)
    counts = rank_counts(rows)
    orange_300k_count = sum(
        1 for row in rows
        if row.get("rank") == "OrangeLegendaryCandidate"
        or (int(row.get("e117_activation_after_gauntlet") or row.get("qualified_activation") or 0) >= 300_000 and row.get("rank") == "CoreMemoryCandidate")
    )
    aggregate = read_json(e109 / "aggregate_metrics.json")
    aggregate = {
        **aggregate,
        "bronze_count": counts["Bronze"],
        "silver_count": counts["Silver"],
        "gold_count": counts["Gold"],
        "diamond_candidate_count": counts["DiamondCandidate"],
        "core_memory_candidate_count": counts["CoreMemoryCandidate"],
        "orange_legendary_candidate_count": counts["OrangeLegendaryCandidate"],
        "red_flag_count": counts["RedFlag"],
        "deprecated_count": counts["Deprecated"],
        "orange_300k_count": orange_300k_count,
        "qualified_activation_total": sum(int(row.get("qualified_activation") or 0) for row in rows),
        "effective_activation_total": sum(int(row.get("e117_activation_after_gauntlet") or row.get("qualified_activation") or 0) for row in rows),
        "e114_projected_reach_permacore_count": e114_payload["aggregate"]["projected_reach_permacore_count"] if e114_payload else None,
        "e114_projected_need_targeted_data_count": e114_payload["aggregate"]["projected_need_targeted_data_count"] if e114_payload else None,
        "e114_stability_trend": e114_payload["aggregate"]["stability_trend"] if e114_payload else None,
        "e116_target_reach_count": e116_payload["aggregate"]["target_reach_count"] if e116_payload else None,
        "e116_targeted_needed_remaining_count": e116_payload["aggregate"]["targeted_needed_remaining_count"] if e116_payload else None,
        "e116_scheduled_case_count": e116_payload["aggregate"]["scheduled_case_count"] if e116_payload else None,
        "e117_target_reach_count": e117_payload["aggregate"]["target_reach_count"] if e117_payload else None,
        "e117_targeted_needed_remaining_count": e117_payload["aggregate"]["targeted_needed_remaining_count"] if e117_payload else None,
        "e117_scheduled_case_count": e117_payload["aggregate"]["scheduled_case_count"] if e117_payload else None,
        "e117_hard_negative_total": e117_payload["aggregate"]["hard_negative_total"] if e117_payload else None,
        "e118_cross_source_pass_count": e118_payload["aggregate"]["cross_source_no_harm_pass_count"] if e118_payload else None,
        "e118_cross_source_remaining_count": e118_payload["aggregate"]["cross_source_no_harm_remaining_count"] if e118_payload else None,
        "e118_hard_negative_total": e118_payload["aggregate"]["hard_negative_total"] if e118_payload else None,
        "e118_synthetic_imprint_total": e118_payload["aggregate"]["synthetic_imprint_total"] if e118_payload else None,
        "e120_saved_operator_count": e120_payload["aggregate"]["saved_operator_count"] if e120_payload else None,
        "e120_promoted_to_gold_count": e120_payload["aggregate"]["promoted_to_gold_count"] if e120_payload else None,
        "e120_hard_negative_total": e120_payload["aggregate"]["hard_negative_total"] if e120_payload else None,
        "e120_mean_selected_prune_ratio": e120_payload["aggregate"]["mean_selected_prune_ratio"] if e120_payload else None,
        "e120_qualified_activation_total": e120_payload["aggregate"]["qualified_activation_total"] if e120_payload else None,
        "e121_orange_legendary_candidate_count": e121_payload["aggregate"]["orange_legendary_candidate_count"] if e121_payload else None,
        "e121_hard_negative_total": e121_payload["aggregate"]["hard_negative_total"] if e121_payload else None,
        "e121_qualified_activation_total": e121_payload["aggregate"]["qualified_activation_total"] if e121_payload else None,
        "e121_mean_selected_prune_ratio": e121_payload["aggregate"]["mean_selected_prune_ratio"] if e121_payload else None,
        "e122_active_operator_count": e122_payload["aggregate"]["active_operator_count"] if e122_payload else None,
        "e122_orange_only_active_count": e122_payload["aggregate"]["orange_only_active_count"] if e122_payload else None,
        "e122_non_orange_active_count": e122_payload["aggregate"]["non_orange_active_count"] if e122_payload else None,
        "e122_negative_card_count": e122_payload["aggregate"]["negative_card_count"] if e122_payload else None,
        "e122_recalled_card_count": e122_payload["aggregate"]["recalled_card_count"] if e122_payload else None,
        "e122_negative_card_recall_event_count": e122_payload["aggregate"]["negative_card_recall_event_count"] if e122_payload else None,
        "e122_prevented_repeat_failure_count": e122_payload["aggregate"]["prevented_repeat_failure_count"] if e122_payload else None,
        "e122_false_block_count": e122_payload["aggregate"]["false_block_count"] if e122_payload else None,
        "e122_negative_card_recall_rate": e122_payload["aggregate"]["negative_card_recall_rate"] if e122_payload else None,
        "e127_orange_legendary_candidate_count": e127_payload["aggregate"].get("orange_legendary_candidate_total") if e127_payload else None,
        "e127_selected_candidate_total": e127_payload["aggregate"].get("selected_candidate_total") if e127_payload else None,
        "e127_cycle_count": e127_payload["aggregate"].get("cycle_count") if e127_payload else None,
        "e127_hard_negative_total": e127_payload["aggregate"].get("hard_negative_total") if e127_payload else None,
        "e127_false_commit_total": e127_payload["aggregate"].get("false_commit_total") if e127_payload else None,
        "e127_wrong_scope_call_total": e127_payload["aggregate"].get("wrong_scope_call_total") if e127_payload else None,
        "e127_unsupported_answer_total": e127_payload["aggregate"].get("unsupported_answer_total") if e127_payload else None,
        "e127_mutation_attempts_total": e127_payload["aggregate"].get("mutation_attempts_total") if e127_payload else None,
        "e129_operator_count": e129_payload["summary"].get("operator_count") if e129_payload else None,
        "e129_orange_legendary_candidate_count": e129_payload["summary"].get("orange_legendary_candidate_count") if e129_payload else None,
        "e129_qualified_activation_total": e129_payload["summary"].get("qualified_activation_total") if e129_payload else None,
        "e129_qualified_activation_min": e129_payload["summary"].get("qualified_activation_min") if e129_payload else None,
        "e129_negative_scope_case_count_total": e129_payload["summary"].get("negative_scope_case_count_total") if e129_payload else None,
        "e129_hard_negative_total": e129_payload["summary"].get("hard_negative_total") if e129_payload else None,
        "e129_false_commit_total": e129_payload["summary"].get("false_commit_total") if e129_payload else None,
        "e129_wrong_scope_call_total": e129_payload["summary"].get("wrong_scope_call_total") if e129_payload else None,
        "e129_unsupported_answer_total": e129_payload["summary"].get("unsupported_answer_total") if e129_payload else None,
        "e130a_candidate_count": e130a_payload["summary"].get("candidate_count") if e130a_payload else None,
        "e130a_orange_legendary_candidate_count": e130a_payload["summary"].get("orange_legendary_candidate_count") if e130a_payload else None,
        "e130a_qualified_activation_before_total": e130a_payload["summary"].get("qualified_activation_before_total") if e130a_payload else None,
        "e130a_qualified_activation_add_total": e130a_payload["summary"].get("qualified_activation_add_total") if e130a_payload else None,
        "e130a_qualified_activation_total": e130a_payload["summary"].get("qualified_activation_total") if e130a_payload else None,
        "e130a_qualified_activation_min": e130a_payload["summary"].get("qualified_activation_min") if e130a_payload else None,
        "e130a_hard_negative_total": e130a_payload["summary"].get("hard_negative_total") if e130a_payload else None,
        "e130a_false_commit_total": e130a_payload["summary"].get("false_commit_total") if e130a_payload else None,
        "e130a_wrong_scope_call_total": e130a_payload["summary"].get("wrong_scope_call_total") if e130a_payload else None,
        "e130a_unsupported_answer_total": e130a_payload["summary"].get("unsupported_answer_total") if e130a_payload else None,
        "e130a_direct_flow_write_total": e130a_payload["summary"].get("direct_flow_write_total") if e130a_payload else None,
        "e130a_negative_transfer_total": e130a_payload["summary"].get("negative_transfer_total") if e130a_payload else None,
        "e130a_mean_selected_prune_ratio": e130a_payload["summary"].get("mean_selected_prune_ratio") if e130a_payload else None,
        "e130b_operator_count": e130b_payload["summary"].get("operator_count") if e130b_payload else None,
        "e130b_transfer_pass_operator_count": e130b_payload["summary"].get("transfer_pass_operator_count") if e130b_payload else None,
        "e130b_visible_transfer_case_count_total": e130b_payload["summary"].get("visible_transfer_case_count_total") if e130b_payload else None,
        "e130b_word_problem_no_call_case_count_total": e130b_payload["summary"].get("word_problem_no_call_case_count_total") if e130b_payload else None,
        "e130b_qualified_transfer_activation_total": e130b_payload["summary"].get("qualified_transfer_activation_total") if e130b_payload else None,
        "e130b_visible_transfer_accuracy_min": e130b_payload["summary"].get("visible_transfer_accuracy_min") if e130b_payload else None,
        "e130b_word_problem_no_call_accuracy_min": e130b_payload["summary"].get("word_problem_no_call_accuracy_min") if e130b_payload else None,
        "e130b_hard_negative_total": e130b_payload["summary"].get("hard_negative_total") if e130b_payload else None,
        "e130b_false_commit_total": e130b_payload["summary"].get("false_commit_total") if e130b_payload else None,
        "e130b_wrong_scope_call_total": e130b_payload["summary"].get("wrong_scope_call_total") if e130b_payload else None,
        "e130b_unsupported_answer_total": e130b_payload["summary"].get("unsupported_answer_total") if e130b_payload else None,
        "e130b_direct_flow_write_total": e130b_payload["summary"].get("direct_flow_write_total") if e130b_payload else None,
        "e130b_overbroad_control_wrong_scope_call_total": e130b_payload["summary"].get("overbroad_control_wrong_scope_call_total") if e130b_payload else None,
        "e131_operator_count": e131_payload["summary"].get("operator_count") if e131_payload else None,
        "e131_transfer_pass_operator_count": e131_payload["summary"].get("transfer_pass_operator_count") if e131_payload else None,
        "e131_dataset_rows_loaded": e131_payload["dataset"].get("row_count_loaded") if e131_payload else None,
        "e131_visible_equation_case_count_total": e131_payload["summary"].get("visible_equation_case_count_total") if e131_payload else None,
        "e131_word_problem_no_call_case_count_total": e131_payload["summary"].get("word_problem_no_call_case_count_total") if e131_payload else None,
        "e131_qualified_visible_activation_total": e131_payload["summary"].get("qualified_visible_activation_total") if e131_payload else None,
        "e131_visible_equation_extraction_accuracy_min": e131_payload["summary"].get("visible_equation_extraction_accuracy_min") if e131_payload else None,
        "e131_word_problem_no_call_accuracy_min": e131_payload["summary"].get("word_problem_no_call_accuracy_min") if e131_payload else None,
        "e131_hard_negative_total": e131_payload["summary"].get("hard_negative_total") if e131_payload else None,
        "e131_false_commit_total": e131_payload["summary"].get("false_commit_total") if e131_payload else None,
        "e131_wrong_scope_call_total": e131_payload["summary"].get("wrong_scope_call_total") if e131_payload else None,
        "e131_unsupported_answer_total": e131_payload["summary"].get("unsupported_answer_total") if e131_payload else None,
        "e131_boundary_claim_violation_total": e131_payload["summary"].get("boundary_claim_violation_total") if e131_payload else None,
        "e131_direct_flow_write_total": e131_payload["summary"].get("direct_flow_write_total") if e131_payload else None,
        "e131_e130b_baseline_visible_miss_total": e131_payload["summary"].get("e130b_baseline_visible_miss_total") if e131_payload else None,
        "e131_overbroad_control_wrong_scope_call_total": e131_payload["summary"].get("overbroad_control_wrong_scope_call_total") if e131_payload else None,
        "e132_operator_count": e132_payload["summary"].get("operator_count") if e132_payload else None,
        "e132_orange_legendary_candidate_count": e132_payload["summary"].get("orange_legendary_candidate_count") if e132_payload else None,
        "e132_dataset_rows_loaded": e132_payload["summary"].get("dataset_rows_loaded") if e132_payload else None,
        "e132_external_support_min": e132_payload["summary"].get("external_support_min") if e132_payload else None,
        "e132_external_support_total": e132_payload["summary"].get("external_support_total") if e132_payload else None,
        "e132_qualified_activation_total": e132_payload["summary"].get("qualified_activation_total") if e132_payload else None,
        "e132_qualified_activation_min": e132_payload["summary"].get("qualified_activation_min") if e132_payload else None,
        "e132_negative_scope_case_count_total": e132_payload["summary"].get("negative_scope_case_count_total") if e132_payload else None,
        "e132_hard_negative_total": e132_payload["summary"].get("hard_negative_total") if e132_payload else None,
        "e132_false_commit_total": e132_payload["summary"].get("false_commit_total") if e132_payload else None,
        "e132_wrong_scope_call_total": e132_payload["summary"].get("wrong_scope_call_total") if e132_payload else None,
        "e132_unsupported_answer_total": e132_payload["summary"].get("unsupported_answer_total") if e132_payload else None,
        "e132_boundary_claim_violation_total": e132_payload["summary"].get("boundary_claim_violation_total") if e132_payload else None,
        "e132_direct_flow_write_total": e132_payload["summary"].get("direct_flow_write_total") if e132_payload else None,
        "e132_overbroad_solver_control_wrong_scope_call_total": e132_payload["summary"].get("overbroad_solver_control_wrong_scope_call_total") if e132_payload else None,
        "e132_mutation_attempts_total": e132_payload["summary"].get("mutation_attempts_total") if e132_payload else None,
        "e132_mean_selected_prune_ratio": e132_payload["summary"].get("mean_selected_prune_ratio") if e132_payload else None,
        "e133_operator_count": e133_payload["summary"].get("operator_count") if e133_payload else None,
        "e133_composition_pass_operator_count": e133_payload["summary"].get("composition_pass_operator_count") if e133_payload else None,
        "e133_route_case_count_total": e133_payload["summary"].get("route_case_count_total") if e133_payload else None,
        "e133_visible_arithmetic_route_case_count_total": e133_payload["summary"].get("visible_arithmetic_route_case_count_total") if e133_payload else None,
        "e133_structural_guard_case_count_total": e133_payload["summary"].get("structural_guard_case_count_total") if e133_payload else None,
        "e133_hidden_word_problem_no_solve_case_count_total": e133_payload["summary"].get("hidden_word_problem_no_solve_case_count_total") if e133_payload else None,
        "e133_route_accuracy_min": e133_payload["summary"].get("route_accuracy_min") if e133_payload else None,
        "e133_visible_arithmetic_route_accuracy_min": e133_payload["summary"].get("visible_arithmetic_route_accuracy_min") if e133_payload else None,
        "e133_structural_guard_accuracy_min": e133_payload["summary"].get("structural_guard_accuracy_min") if e133_payload else None,
        "e133_hidden_word_problem_no_solve_accuracy_min": e133_payload["summary"].get("hidden_word_problem_no_solve_accuracy_min") if e133_payload else None,
        "e133_hard_negative_total": e133_payload["summary"].get("hard_negative_total") if e133_payload else None,
        "e133_wrong_scope_call_total": e133_payload["summary"].get("wrong_scope_call_total") if e133_payload else None,
        "e133_false_commit_total": e133_payload["summary"].get("false_commit_total") if e133_payload else None,
        "e133_direct_flow_write_total": e133_payload["summary"].get("direct_flow_write_total") if e133_payload else None,
        "e133_overbroad_solver_control_wrong_scope_call_total": e133_payload["summary"].get("overbroad_solver_control_wrong_scope_call_total") if e133_payload else None,
        "e133_trust_control_false_commit_total": e133_payload["summary"].get("trust_control_false_commit_total") if e133_payload else None,
        "e133_trust_control_direct_flow_write_total": e133_payload["summary"].get("trust_control_direct_flow_write_total") if e133_payload else None,
        "e134_operator_count": e134_payload["summary"].get("operator_count") if e134_payload else None,
        "e134_ood_pass_operator_count": e134_payload["summary"].get("ood_pass_operator_count") if e134_payload else None,
        "e134_ood_case_count_total": e134_payload["summary"].get("ood_case_count_total") if e134_payload else None,
        "e134_visible_arithmetic_ood_case_count_total": e134_payload["summary"].get("visible_arithmetic_ood_case_count_total") if e134_payload else None,
        "e134_structural_guard_ood_case_count_total": e134_payload["summary"].get("structural_guard_ood_case_count_total") if e134_payload else None,
        "e134_hidden_word_problem_ood_no_solve_case_count_total": e134_payload["summary"].get("hidden_word_problem_ood_no_solve_case_count_total") if e134_payload else None,
        "e134_counterexample_case_count_total": e134_payload["summary"].get("counterexample_case_count_total") if e134_payload else None,
        "e134_ood_route_accuracy_min": e134_payload["summary"].get("ood_route_accuracy_min") if e134_payload else None,
        "e134_visible_arithmetic_ood_accuracy_min": e134_payload["summary"].get("visible_arithmetic_ood_accuracy_min") if e134_payload else None,
        "e134_structural_guard_ood_accuracy_min": e134_payload["summary"].get("structural_guard_ood_accuracy_min") if e134_payload else None,
        "e134_hidden_word_problem_ood_no_solve_accuracy_min": e134_payload["summary"].get("hidden_word_problem_ood_no_solve_accuracy_min") if e134_payload else None,
        "e134_counterexample_accuracy_min": e134_payload["summary"].get("counterexample_accuracy_min") if e134_payload else None,
        "e134_hard_negative_total": e134_payload["summary"].get("hard_negative_total") if e134_payload else None,
        "e134_wrong_scope_call_total": e134_payload["summary"].get("wrong_scope_call_total") if e134_payload else None,
        "e134_false_commit_total": e134_payload["summary"].get("false_commit_total") if e134_payload else None,
        "e134_direct_flow_write_total": e134_payload["summary"].get("direct_flow_write_total") if e134_payload else None,
        "e134_e133_baseline_ood_miss_total": e134_payload["summary"].get("e133_baseline_ood_miss_total") if e134_payload else None,
        "e134_overbroad_solver_control_wrong_scope_call_total": e134_payload["summary"].get("overbroad_solver_control_wrong_scope_call_total") if e134_payload else None,
        "e134_trust_control_false_commit_total": e134_payload["summary"].get("trust_control_false_commit_total") if e134_payload else None,
        "e134_trust_control_direct_flow_write_total": e134_payload["summary"].get("trust_control_direct_flow_write_total") if e134_payload else None,
        "e135_operator_count": e135_payload["summary"].get("operator_count") if e135_payload else None,
        "e135_dialogue_pass_operator_count": e135_payload["summary"].get("dialogue_pass_operator_count") if e135_payload else None,
        "e135_dialogue_case_count_total": e135_payload["summary"].get("dialogue_case_count_total") if e135_payload else None,
        "e135_dialogue_turn_count_total": e135_payload["summary"].get("dialogue_turn_count_total") if e135_payload else None,
        "e135_hidden_word_problem_dialogue_no_solve_case_count_total": e135_payload["summary"].get("hidden_word_problem_dialogue_no_solve_case_count_total") if e135_payload else None,
        "e135_visible_reentry_dialogue_case_count_total": e135_payload["summary"].get("visible_reentry_dialogue_case_count_total") if e135_payload else None,
        "e135_stale_route_rejection_case_count_total": e135_payload["summary"].get("stale_route_rejection_case_count_total") if e135_payload else None,
        "e135_cross_thread_rejection_case_count_total": e135_payload["summary"].get("cross_thread_rejection_case_count_total") if e135_payload else None,
        "e135_counterexample_dialogue_case_count_total": e135_payload["summary"].get("counterexample_dialogue_case_count_total") if e135_payload else None,
        "e135_dialogue_state_accuracy_min": e135_payload["summary"].get("dialogue_state_accuracy_min") if e135_payload else None,
        "e135_current_turn_route_accuracy_min": e135_payload["summary"].get("current_turn_route_accuracy_min") if e135_payload else None,
        "e135_route_state_integrity_min": e135_payload["summary"].get("route_state_integrity_min") if e135_payload else None,
        "e135_hidden_word_problem_dialogue_no_solve_accuracy_min": e135_payload["summary"].get("hidden_word_problem_dialogue_no_solve_accuracy_min") if e135_payload else None,
        "e135_counterexample_dialogue_accuracy_min": e135_payload["summary"].get("counterexample_dialogue_accuracy_min") if e135_payload else None,
        "e135_hard_negative_total": e135_payload["summary"].get("hard_negative_total") if e135_payload else None,
        "e135_wrong_scope_call_total": e135_payload["summary"].get("wrong_scope_call_total") if e135_payload else None,
        "e135_false_commit_total": e135_payload["summary"].get("false_commit_total") if e135_payload else None,
        "e135_direct_flow_write_total": e135_payload["summary"].get("direct_flow_write_total") if e135_payload else None,
        "e135_stale_route_reuse_total": e135_payload["summary"].get("stale_route_reuse_total") if e135_payload else None,
        "e135_cross_thread_contamination_total": e135_payload["summary"].get("cross_thread_contamination_total") if e135_payload else None,
        "e135_latest_route_reuse_control_failure_total": e135_payload["summary"].get("latest_route_reuse_control_failure_total") if e135_payload else None,
        "e135_stale_route_reuse_control_failure_total": e135_payload["summary"].get("stale_route_reuse_control_failure_total") if e135_payload else None,
        "e135_cross_thread_contamination_control_failure_total": e135_payload["summary"].get("cross_thread_contamination_control_failure_total") if e135_payload else None,
        "e135_counterexample_trust_control_failure_total": e135_payload["summary"].get("counterexample_trust_control_failure_total") if e135_payload else None,
        "e135_single_turn_reset_control_failure_total": e135_payload["summary"].get("single_turn_reset_control_failure_total") if e135_payload else None,
        "e136a_operator_count": e136a_payload["summary"].get("operator_count") if e136a_payload else None,
        "e136a_orange_legendary_candidate_count": e136a_payload["summary"].get("orange_legendary_candidate_count") if e136a_payload else None,
        "e136a_dataset_rows_loaded": e136a_payload["summary"].get("dataset_rows_loaded") if e136a_payload else None,
        "e136a_external_source_count": e136a_payload["summary"].get("external_source_count") if e136a_payload else None,
        "e136a_external_family_count": e136a_payload["summary"].get("external_family_count") if e136a_payload else None,
        "e136a_external_support_min": e136a_payload["summary"].get("external_support_min") if e136a_payload else None,
        "e136a_external_support_total": e136a_payload["summary"].get("external_support_total") if e136a_payload else None,
        "e136a_qualified_activation_total": e136a_payload["summary"].get("qualified_activation_total") if e136a_payload else None,
        "e136a_qualified_activation_min": e136a_payload["summary"].get("qualified_activation_min") if e136a_payload else None,
        "e136a_negative_scope_case_count_total": e136a_payload["summary"].get("negative_scope_case_count_total") if e136a_payload else None,
        "e136a_hard_negative_total": e136a_payload["summary"].get("hard_negative_total") if e136a_payload else None,
        "e136a_wrong_scope_call_total": e136a_payload["summary"].get("wrong_scope_call_total") if e136a_payload else None,
        "e136a_false_commit_total": e136a_payload["summary"].get("false_commit_total") if e136a_payload else None,
        "e136a_unsupported_answer_total": e136a_payload["summary"].get("unsupported_answer_total") if e136a_payload else None,
        "e136a_boundary_claim_violation_total": e136a_payload["summary"].get("boundary_claim_violation_total") if e136a_payload else None,
        "e136a_direct_flow_write_total": e136a_payload["summary"].get("direct_flow_write_total") if e136a_payload else None,
        "e136a_overbroad_chatbot_control_wrong_scope_call_total": e136a_payload["summary"].get("overbroad_chatbot_control_wrong_scope_call_total") if e136a_payload else None,
        "e136a_mutation_attempts_total": e136a_payload["summary"].get("mutation_attempts_total") if e136a_payload else None,
        "e136a_accepted_mutations_total": e136a_payload["summary"].get("accepted_mutations_total") if e136a_payload else None,
        "e136a_rollback_count_total": e136a_payload["summary"].get("rollback_count_total") if e136a_payload else None,
        "e136a_mean_selected_prune_ratio": e136a_payload["summary"].get("mean_selected_prune_ratio") if e136a_payload else None,
        "e136b_operator_count": e136b_payload["summary"].get("operator_count") if e136b_payload else None,
        "e136b_route_pass_operator_count": e136b_payload["summary"].get("route_pass_operator_count") if e136b_payload else None,
        "e136b_dataset_rows_loaded": e136b_payload["summary"].get("dataset_rows_loaded") if e136b_payload else None,
        "e136b_route_seed_row_count": e136b_payload["summary"].get("route_seed_row_count") if e136b_payload else None,
        "e136b_route_case_count_total": e136b_payload["summary"].get("route_case_count_total") if e136b_payload else None,
        "e136b_multi_route_composition_case_count_total": e136b_payload["summary"].get("multi_route_composition_case_count_total") if e136b_payload else None,
        "e136b_boundary_case_count_total": e136b_payload["summary"].get("boundary_case_count_total") if e136b_payload else None,
        "e136b_negative_scope_case_count_total": e136b_payload["summary"].get("negative_scope_case_count_total") if e136b_payload else None,
        "e136b_qualified_route_activation_total": e136b_payload["summary"].get("qualified_route_activation_total") if e136b_payload else None,
        "e136b_qualified_route_activation_min": e136b_payload["summary"].get("qualified_route_activation_min") if e136b_payload else None,
        "e136b_route_accuracy_min": e136b_payload["summary"].get("route_accuracy_min") if e136b_payload else None,
        "e136b_route_stack_accuracy_min": e136b_payload["summary"].get("route_stack_accuracy_min") if e136b_payload else None,
        "e136b_primary_route_accuracy_min": e136b_payload["summary"].get("primary_route_accuracy_min") if e136b_payload else None,
        "e136b_boundary_accuracy_min": e136b_payload["summary"].get("boundary_accuracy_min") if e136b_payload else None,
        "e136b_multi_route_composition_accuracy_min": e136b_payload["summary"].get("multi_route_composition_accuracy_min") if e136b_payload else None,
        "e136b_negative_scope_accuracy_min": e136b_payload["summary"].get("negative_scope_accuracy_min") if e136b_payload else None,
        "e136b_hard_negative_total": e136b_payload["summary"].get("hard_negative_total") if e136b_payload else None,
        "e136b_wrong_scope_call_total": e136b_payload["summary"].get("wrong_scope_call_total") if e136b_payload else None,
        "e136b_false_commit_total": e136b_payload["summary"].get("false_commit_total") if e136b_payload else None,
        "e136b_unsupported_answer_total": e136b_payload["summary"].get("unsupported_answer_total") if e136b_payload else None,
        "e136b_boundary_claim_violation_total": e136b_payload["summary"].get("boundary_claim_violation_total") if e136b_payload else None,
        "e136b_direct_flow_write_total": e136b_payload["summary"].get("direct_flow_write_total") if e136b_payload else None,
        "e136b_overbroad_chatbot_control_wrong_scope_call_total": e136b_payload["summary"].get("overbroad_chatbot_control_wrong_scope_call_total") if e136b_payload else None,
        "e136b_unsafe_direct_write_control_direct_flow_write_total": e136b_payload["summary"].get("unsafe_direct_write_control_direct_flow_write_total") if e136b_payload else None,
        "e136b_source_hallucination_control_unsupported_answer_total": e136b_payload["summary"].get("source_hallucination_control_unsupported_answer_total") if e136b_payload else None,
    }
    summary = read_json(e109 / "summary.json")
    summary = {
        **summary,
        "rank_counts": counts,
        "latest_wave": "E136B assistant text route composition and boundary confirm" if e136b_payload else "E136A assistant text skill farm mutation/prune Orange cycle" if e136a_payload else "E135 math text multi-route assistant dialogue-state gauntlet" if e135_payload else "E134 external math text OOD route stress and counterexample gauntlet" if e134_payload else "E133 math text route composition and no-solve assistant confirm" if e133_payload else "E132 external math text skill farm mutation/prune Orange cycle" if e132_payload else "E131 visible equation extraction and assistant arithmetic render gauntlet" if e131_payload else "E130B arithmetic text-IO transfer and word-problem no-call gauntlet" if e130b_payload else "E130A CoreMemory to Orange backfill gauntlet" if e130a_payload else "E129 arithmetic trace Orange/Legendary probation" if e129_payload else "E127 overnight text skill farm Orange cycle" if e127_payload else "E122 orange-only baseline and negative-card recall" if e122_payload else "E121 E120 Gold to Orange/Legendary probation gauntlet" if e121_payload else "E120 FineWeb skill farm to Gold wave" if e120_payload else "E118 cross-source no-harm gauntlet" if e118_payload else "E117 alpha-Weave targeted pressure gauntlet" if e117_payload else "E116 alpha-Weave targeted pressure" if e116_payload else "E114 FineWeb projection" if e114_payload else "E112 Wave 3" if e112_payload else "E111 Wave 2" if e111_payload else "E110 Wave 1" if e110_payload else "E109",
    }
    return {
        "summary": summary,
        "e110": e110_payload,
        "e111": e111_payload,
        "e112": e112_payload,
        "e114": e114_payload,
        "e116": e116_payload,
        "e117": e117_payload,
        "e118": e118_payload,
        "e120": e120_payload,
        "e121": e121_payload,
        "e122": e122_payload,
        "e127": e127_payload,
        "e129": e129_payload,
        "e130a": e130a_payload,
        "e130b": e130b_payload,
        "e131": e131_payload,
        "e132": e132_payload,
        "e133": e133_payload,
        "e134": e134_payload,
        "e135": e135_payload,
        "e136a": e136a_payload,
        "e136b": e136b_payload,
        "aggregate": aggregate,
        "policy": read_json(e109 / "rank_policy_manifest.json"),
        "watch": read_json(e109 / "golden_watch_report.json"),
        "challenger": read_json(e109 / "challenger_prune_report.json"),
        "rows": rows,
    }


def render_html(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    escaped_data = data.replace("</", "<\\/")
    latest = html.escape(str(payload.get("summary", {}).get("latest_wave", "E109")))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>VRAXION Operator Rank Dashboard</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #0e1118;
      --panel: #171b25;
      --panel2: #111620;
      --line: #2c3446;
      --text: #edf2ff;
      --muted: #9aa7bd;
      --blue: #4da3ff;
      --green: #4be28a;
      --gold: #ffd35a;
      --silver: #cdd7e6;
      --bronze: #c88b57;
      --orange: #ff9b38;
      --red: #ff5c7a;
      --violet: #b98cff;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif;
      background: radial-gradient(circle at 18% 0%, #1b2740 0, transparent 34rem), var(--bg);
      color: var(--text);
      letter-spacing: 0;
    }}
    header {{
      padding: 24px 28px 14px;
      border-bottom: 1px solid var(--line);
      background: rgba(14,17,24,.86);
      position: sticky;
      top: 0;
      z-index: 10;
      backdrop-filter: blur(10px);
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 28px;
      font-weight: 740;
    }}
    .subtitle {{
      color: var(--muted);
      max-width: 1120px;
      line-height: 1.45;
    }}
    .wrap {{ padding: 20px 28px 40px; }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(6, minmax(130px, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }}
    .card, .panel {{
      background: linear-gradient(180deg, rgba(255,255,255,.035), rgba(255,255,255,.01)), var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 14px 34px rgba(0,0,0,.18);
    }}
    .card {{ padding: 14px; min-height: 86px; }}
    .label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }}
    .value {{ font-size: 26px; font-weight: 760; margin-top: 8px; }}
    .core {{ color: var(--violet); }}
    .gold {{ color: var(--gold); }}
    .silver {{ color: var(--silver); }}
    .bronze {{ color: var(--bronze); }}
    .orange {{ color: var(--orange); }}
    .red {{ color: var(--red); }}
    .green {{ color: var(--green); }}
    .toolbar {{
      display: grid;
      grid-template-columns: 1fr 210px 190px 190px;
      gap: 10px;
      margin-bottom: 14px;
    }}
    input, select, button {{
      background: var(--panel2);
      color: var(--text);
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 10px 11px;
      font: inherit;
      min-width: 0;
    }}
    button {{ cursor: pointer; }}
    button.active {{ border-color: var(--blue); box-shadow: 0 0 0 1px var(--blue) inset; }}
    .rank-buttons {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 0 0 14px;
    }}
    .rank-buttons button {{ padding: 8px 10px; }}
    .layout {{
      display: grid;
      grid-template-columns: minmax(380px, 1fr) 420px;
      gap: 14px;
      align-items: start;
    }}
    .panel {{ padding: 14px; }}
    .panel h2 {{ margin: 0 0 12px; font-size: 17px; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(18px, 1fr));
      gap: 5px;
      margin-bottom: 16px;
    }}
    .cell {{
      height: 22px;
      border-radius: 5px;
      border: 1px solid rgba(255,255,255,.12);
      cursor: pointer;
      opacity: .92;
    }}
    .cell:hover {{ transform: translateY(-1px); filter: brightness(1.2); }}
    .r-Gold {{ background: linear-gradient(180deg, #ffe28a, #b98017); }}
    .r-CoreMemoryCandidate {{ background: linear-gradient(180deg, #d7b6ff, #6847c7); }}
    .r-Orange300K {{ background: linear-gradient(180deg, #ffc36a, #e66c18); }}
    .r-Silver {{ background: linear-gradient(180deg, #eef4ff, #6d7f9d); }}
    .r-Bronze {{ background: linear-gradient(180deg, #d99c63, #7f4f2a); }}
    .r-Deprecated {{ background: linear-gradient(180deg, #777, #333); }}
    .r-RedFlag {{ background: linear-gradient(180deg, #ff7890, #8e1931); }}
    .r-DiamondCandidate {{ background: linear-gradient(180deg, #cef8ff, #49adff); }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ padding: 9px 8px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }}
    th {{ color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .04em; position: sticky; top: 104px; background: var(--panel); }}
    tr {{ cursor: pointer; }}
    tr:hover {{ background: rgba(255,255,255,.035); }}
    .pill {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 8px;
      border-radius: 999px;
      font-size: 12px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,.035);
      white-space: nowrap;
    }}
    .bar {{
      width: 100%;
      height: 8px;
      background: #232b3b;
      border-radius: 999px;
      overflow: hidden;
      margin-top: 6px;
    }}
    .bar > span {{
      display: block;
      height: 100%;
      background: linear-gradient(90deg, var(--blue), var(--green));
      width: 0;
    }}
    .detail-title {{ font-size: 20px; font-weight: 760; margin-bottom: 4px; }}
    .detail-id {{ color: var(--muted); font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 12px; overflow-wrap: anywhere; }}
    .kv {{
      display: grid;
      grid-template-columns: 160px 1fr;
      gap: 8px 12px;
      margin-top: 14px;
      font-size: 13px;
    }}
    .kv div:nth-child(odd) {{ color: var(--muted); }}
    .note {{
      margin-top: 14px;
      padding: 10px;
      border-radius: 7px;
      background: rgba(77,163,255,.08);
      border: 1px solid rgba(77,163,255,.28);
      color: #cfe4ff;
      line-height: 1.45;
    }}
    .table-wrap {{ max-height: 68vh; overflow: auto; border: 1px solid var(--line); border-radius: 8px; }}
    @media (max-width: 1180px) {{
      .cards {{ grid-template-columns: repeat(3, 1fr); }}
      .toolbar {{ grid-template-columns: 1fr 1fr; }}
      .layout {{ grid-template-columns: 1fr; }}
      th {{ top: 0; }}
    }}
    @media (max-width: 700px) {{
      header, .wrap {{ padding-left: 14px; padding-right: 14px; }}
      .cards {{ grid-template-columns: repeat(2, 1fr); }}
      .toolbar {{ grid-template-columns: 1fr; }}
      .kv {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>VRAXION Operator Rank Dashboard</h1>
    <div class="subtitle">{latest} rank view. Gold/Silver/Bronze are scoped ranks, not Core memory. Use this page to watch which operators are ready for Diamond/Core probation and which still need evidence.</div>
  </header>
  <div class="wrap">
    <section class="cards" id="cards"></section>

    <div class="toolbar">
      <input id="search" placeholder="Search operator, scope, family..." />
      <select id="rankFilter"></select>
      <select id="scopeFilter"></select>
      <select id="sortBy">
        <option value="rank">Sort by rank</option>
        <option value="activation">Sort by activation</option>
        <option value="remaining">Sort by Diamond remaining</option>
        <option value="value">Sort by counterfactual value</option>
        <option value="scope">Sort by scope</option>
      </select>
    </div>
    <div class="rank-buttons" id="rankButtons"></div>

    <section class="layout">
      <div class="panel">
        <h2>Rank Map</h2>
        <div class="grid" id="rankGrid"></div>
        <h2>Operators</h2>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Rank</th>
                <th>Operator</th>
                <th>Scope</th>
                <th>Activation</th>
                <th>Next target</th>
                <th>No-harm</th>
              </tr>
            </thead>
            <tbody id="rows"></tbody>
          </table>
        </div>
      </div>
      <aside class="panel" id="detail"></aside>
    </section>
  </div>
  <script>
    const DATA = {escaped_data};
    const rows = DATA.rows;
    const rankOrder = {{Orange300K: 0, CoreMemoryCandidate: 1, DiamondCandidate: 2, Gold: 3, Silver: 4, Bronze: 5, Deprecated: 6, RedFlag: 7}};
    const nextTargets = {{
      Bronze: {{name: "Silver", value: 300}},
      Silver: {{name: "Gold", value: 3000}},
      Gold: {{name: "Diamond", value: 30000}},
      DiamondCandidate: {{name: "CoreCandidate", value: 100000}},
      CoreMemoryCandidate: {{name: "PermaCore probation", value: 300000}},
      OrangeLegendaryCandidate: {{name: "Million-proof probation", value: 1000000}},
      Deprecated: {{name: "Stopped", value: 0}},
      RedFlag: {{name: "Stopped", value: 0}}
    }};
    let state = {{rank: "All", scope: "All", q: "", sort: "rank", selected: null}};

    function fmt(n) {{
      if (typeof n !== "number") return n ?? "";
      return n.toLocaleString();
    }}
    function pct(n) {{
      if (typeof n !== "number") return "";
      return (n * 100).toFixed(4) + "%";
    }}
    function rankClass(rank) {{ return "r-" + String(rank || "Bronze").replaceAll(" ", ""); }}
    function effectiveActivation(row) {{
      return row.e117_activation_after_gauntlet || row.qualified_activation || 0;
    }}
    function visualRank(row) {{
      if (row.rank === "OrangeLegendaryCandidate") return "Orange300K";
      if (row.rank === "CoreMemoryCandidate" && effectiveActivation(row) >= 300000 && (row.e117_hard_negative || 0) === 0) return "Orange300K";
      return row.rank || "Bronze";
    }}
    function nextTarget(row) {{
      const target = nextTargets[row.rank] || nextTargets.Bronze;
      const activation = effectiveActivation(row);
      const remain = Math.max(0, target.value - activation);
      const progress = target.value > 0 ? Math.min(1, activation / target.value) : 1;
      return {{...target, remain, progress}};
    }}
    function filtered() {{
      const q = state.q.toLowerCase().trim();
      let out = rows.filter(row => {{
        if (state.rank !== "All" && visualRank(row) !== state.rank) return false;
        if (state.scope !== "All" && row.scope !== state.scope) return false;
        if (!q) return true;
        return [row.operator_id, row.display_name, row.scope, row.family, row.group_id, row.rank, visualRank(row), row.e108_status]
          .join(" ").toLowerCase().includes(q);
      }});
      out.sort((a,b) => {{
        if (state.sort === "activation") return effectiveActivation(b) - effectiveActivation(a);
        if (state.sort === "remaining") return nextTarget(a).remain - nextTarget(b).remain;
        if (state.sort === "value") return (b.counterfactual_value||0) - (a.counterfactual_value||0);
        if (state.sort === "scope") return String(a.scope).localeCompare(String(b.scope)) || (rankOrder[visualRank(a)] - rankOrder[visualRank(b)]);
        return (rankOrder[visualRank(a)] - rankOrder[visualRank(b)]) || String(a.operator_id).localeCompare(String(b.operator_id));
      }});
      return out;
    }}
    function renderCards() {{
      const agg = DATA.aggregate;
      const cards = [
        ["CoreCandidate", agg.core_memory_candidate_count, "core"],
        ["Orange/Legendary", agg.orange_legendary_candidate_count ?? 0, "orange"],
        ["Orange 300K", agg.orange_300k_count ?? 0, "orange"],
        ["Gold", agg.gold_count, "gold"],
        ["Silver", agg.silver_count, "silver"],
        ["Bronze", agg.bronze_count, "bronze"],
        ["Hard negative", agg.hard_negative_total, agg.hard_negative_total ? "red" : "green"],
        ["Effective activations", fmt(agg.effective_activation_total ?? agg.qualified_activation_total), "green"],
        ["E114 reaches target", agg.e114_projected_reach_permacore_count ?? "n/a", "green"],
        ["E114 targeted needed", agg.e114_projected_need_targeted_data_count ?? "n/a", agg.e114_projected_need_targeted_data_count ? "gold" : "green"],
        ["E116 synthetic reaches", agg.e116_target_reach_count ?? "n/a", "green"],
        ["E116 remaining", agg.e116_targeted_needed_remaining_count ?? "n/a", agg.e116_targeted_needed_remaining_count ? "gold" : "green"],
        ["E117 gauntlet reaches", agg.e117_target_reach_count ?? "n/a", "green"],
        ["E117 hard negatives", agg.e117_hard_negative_total ?? "n/a", agg.e117_hard_negative_total ? "red" : "green"],
        ["E118 cross-source pass", agg.e118_cross_source_pass_count ?? "n/a", "orange"],
        ["E118 hard negatives", agg.e118_hard_negative_total ?? "n/a", agg.e118_hard_negative_total ? "red" : "green"],
        ["E120 new Gold", agg.e120_promoted_to_gold_count ?? "n/a", "gold"],
        ["E120 hard negatives", agg.e120_hard_negative_total ?? "n/a", agg.e120_hard_negative_total ? "red" : "green"],
        ["E121 Orange", agg.e121_orange_legendary_candidate_count ?? "n/a", "orange"],
        ["E121 hard negatives", agg.e121_hard_negative_total ?? "n/a", agg.e121_hard_negative_total ? "red" : "green"],
        ["E122 orange-only", (agg.e122_orange_only_active_count ?? "n/a") + "/" + (agg.e122_active_operator_count ?? "n/a"), "orange"],
        ["E122 non-orange", agg.e122_non_orange_active_count ?? "n/a", agg.e122_non_orange_active_count ? "red" : "green"],
        ["Negative cards", agg.e122_negative_card_count ?? "n/a", "gold"],
        ["Negative recalls", agg.e122_negative_card_recall_event_count ?? "n/a", "green"],
        ["Prevented repeats", agg.e122_prevented_repeat_failure_count ?? "n/a", "green"],
        ["False blocks", agg.e122_false_block_count ?? "n/a", agg.e122_false_block_count ? "red" : "green"],
        ["E127 Orange", agg.e127_orange_legendary_candidate_count ?? "n/a", "orange"],
        ["E127 cycles", agg.e127_cycle_count ?? "n/a", "green"],
        ["E127 hard negatives", agg.e127_hard_negative_total ?? "n/a", agg.e127_hard_negative_total ? "red" : "green"],
        ["E127 false commits", agg.e127_false_commit_total ?? "n/a", agg.e127_false_commit_total ? "red" : "green"],
        ["E129 arithmetic", (agg.e129_orange_legendary_candidate_count ?? "n/a") + "/" + (agg.e129_operator_count ?? "n/a"), "orange"],
        ["E129 activations", fmt(agg.e129_qualified_activation_total ?? 0), "green"],
        ["E129 no-call", fmt(agg.e129_negative_scope_case_count_total ?? 0), "green"],
        ["E129 hard negatives", agg.e129_hard_negative_total ?? "n/a", agg.e129_hard_negative_total ? "red" : "green"],
        ["E130A backfill", (agg.e130a_orange_legendary_candidate_count ?? "n/a") + "/" + (agg.e130a_candidate_count ?? "n/a"), "orange"],
        ["E130A activation add", fmt(agg.e130a_qualified_activation_add_total ?? 0), "green"],
        ["E130A hard negatives", agg.e130a_hard_negative_total ?? "n/a", agg.e130a_hard_negative_total ? "red" : "green"],
        ["E130A direct writes", agg.e130a_direct_flow_write_total ?? "n/a", agg.e130a_direct_flow_write_total ? "red" : "green"],
        ["E130B transfer", (agg.e130b_transfer_pass_operator_count ?? "n/a") + "/" + (agg.e130b_operator_count ?? "n/a"), "orange"],
        ["E130B visible IO", pct(agg.e130b_visible_transfer_accuracy_min ?? 0), "green"],
        ["E130B word no-call", pct(agg.e130b_word_problem_no_call_accuracy_min ?? 0), "green"],
        ["E130B wrong scope", agg.e130b_wrong_scope_call_total ?? "n/a", agg.e130b_wrong_scope_call_total ? "red" : "green"],
        ["E131 visible eq", (agg.e131_transfer_pass_operator_count ?? "n/a") + "/" + (agg.e131_operator_count ?? "n/a"), "orange"],
        ["E131 extraction", pct(agg.e131_visible_equation_extraction_accuracy_min ?? 0), "green"],
        ["E131 word no-call", pct(agg.e131_word_problem_no_call_accuracy_min ?? 0), "green"],
        ["E131 hard negatives", agg.e131_hard_negative_total ?? "n/a", agg.e131_hard_negative_total ? "red" : "green"],
        ["E132 math-text", (agg.e132_orange_legendary_candidate_count ?? "n/a") + "/" + (agg.e132_operator_count ?? "n/a"), "orange"],
        ["E132 dataset rows", fmt(agg.e132_dataset_rows_loaded ?? 0), "green"],
        ["E132 support min", fmt(agg.e132_external_support_min ?? 0), "green"],
        ["E132 hard negatives", agg.e132_hard_negative_total ?? "n/a", agg.e132_hard_negative_total ? "red" : "green"],
        ["E133 route comp", (agg.e133_composition_pass_operator_count ?? "n/a") + "/" + (agg.e133_operator_count ?? "n/a"), "orange"],
        ["E133 route cases", fmt(agg.e133_route_case_count_total ?? 0), "green"],
        ["E133 hidden no-call", pct(agg.e133_hidden_word_problem_no_solve_accuracy_min ?? 0), "green"],
        ["E133 hard negatives", agg.e133_hard_negative_total ?? "n/a", agg.e133_hard_negative_total ? "red" : "green"],
        ["E134 OOD pass", (agg.e134_ood_pass_operator_count ?? "n/a") + "/" + (agg.e134_operator_count ?? "n/a"), "orange"],
        ["E134 OOD cases", fmt(agg.e134_ood_case_count_total ?? 0), "green"],
        ["E134 counterexamples", fmt(agg.e134_counterexample_case_count_total ?? 0), "green"],
        ["E134 hidden no-call", pct(agg.e134_hidden_word_problem_ood_no_solve_accuracy_min ?? 0), "green"],
        ["E134 hard negatives", agg.e134_hard_negative_total ?? "n/a", agg.e134_hard_negative_total ? "red" : "green"],
        ["E134 baseline misses", fmt(agg.e134_e133_baseline_ood_miss_total ?? 0), "gold"],
        ["E135 dialogue", (agg.e135_dialogue_pass_operator_count ?? "n/a") + "/" + (agg.e135_operator_count ?? "n/a"), "orange"],
        ["E135 cases", fmt(agg.e135_dialogue_case_count_total ?? 0), "green"],
        ["E135 turns", fmt(agg.e135_dialogue_turn_count_total ?? 0), "green"],
        ["E135 current route", pct(agg.e135_current_turn_route_accuracy_min ?? 0), "green"],
        ["E135 state integrity", pct(agg.e135_route_state_integrity_min ?? 0), "green"],
        ["E135 hard negatives", agg.e135_hard_negative_total ?? "n/a", agg.e135_hard_negative_total ? "red" : "green"],
        ["E136A assistant text", (agg.e136a_orange_legendary_candidate_count ?? "n/a") + "/" + (agg.e136a_operator_count ?? "n/a"), "orange"],
        ["E136A dataset rows", fmt(agg.e136a_dataset_rows_loaded ?? 0), "green"],
        ["E136A support min", fmt(agg.e136a_external_support_min ?? 0), "green"],
        ["E136A activations", fmt(agg.e136a_qualified_activation_total ?? 0), "green"],
        ["E136A hard negatives", agg.e136a_hard_negative_total ?? "n/a", agg.e136a_hard_negative_total ? "red" : "green"],
        ["E136A direct writes", agg.e136a_direct_flow_write_total ?? "n/a", agg.e136a_direct_flow_write_total ? "red" : "green"],
        ["E136B route comp", (agg.e136b_route_pass_operator_count ?? "n/a") + "/" + (agg.e136b_operator_count ?? "n/a"), "orange"],
        ["E136B route cases", fmt(agg.e136b_route_case_count_total ?? 0), "green"],
        ["E136B stack", pct(agg.e136b_route_stack_accuracy_min ?? 0), "green"],
        ["E136B boundary", pct(agg.e136b_boundary_accuracy_min ?? 0), "green"],
        ["E136B hard negatives", agg.e136b_hard_negative_total ?? "n/a", agg.e136b_hard_negative_total ? "red" : "green"],
        ["E136B direct writes", agg.e136b_direct_flow_write_total ?? "n/a", agg.e136b_direct_flow_write_total ? "red" : "green"]
      ];
      document.getElementById("cards").innerHTML = cards.map(([label,value,cls]) =>
        `<div class="card"><div class="label">${{label}}</div><div class="value ${{cls}}">${{value}}</div></div>`
      ).join("");
    }}
    function renderFilters() {{
      const ranks = ["All", ...Array.from(new Set(rows.map(r => visualRank(r)))).sort((a,b) => rankOrder[a] - rankOrder[b])];
      const scopes = ["All", ...Array.from(new Set(rows.map(r => r.scope))).sort()];
      document.getElementById("rankFilter").innerHTML = ranks.map(r => `<option value="${{htmlEscape(r)}}">${{htmlEscape(r)}}</option>`).join("");
      document.getElementById("scopeFilter").innerHTML = scopes.map(s => `<option value="${{htmlEscape(s)}}">${{htmlEscape(s)}}</option>`).join("");
      document.getElementById("rankFilter").value = ranks.includes(state.rank) ? state.rank : "All";
      document.getElementById("scopeFilter").value = scopes.includes(state.scope) ? state.scope : "All";
      document.getElementById("rankButtons").innerHTML = ranks.map(r => {{
        const count = r === "All" ? rows.length : rows.filter(x => visualRank(x) === r).length;
        return `<button data-rank="${{htmlEscape(r)}}" class="${{state.rank === r ? "active" : ""}}">${{htmlEscape(r)}} <span class="pill">${{count}}</span></button>`;
      }}).join("");
      document.querySelectorAll("#rankButtons button").forEach(btn => btn.onclick = () => {{
        state.rank = btn.dataset.rank;
        document.getElementById("rankFilter").value = state.rank;
        render();
      }});
    }}
    function renderGrid(items) {{
      document.getElementById("rankGrid").innerHTML = items.map(row =>
        `<div class="cell ${{rankClass(visualRank(row))}}" title="${{htmlEscape(visualRank(row))}} · ${{htmlEscape(row.operator_id)}} · ${{htmlEscape(row.scope)}}" data-id="${{htmlEscape(row.operator_id)}}"></div>`
      ).join("");
      document.querySelectorAll(".cell").forEach(cell => cell.onclick = () => select(cell.dataset.id));
    }}
    function renderRows(items) {{
      document.getElementById("rows").innerHTML = items.map(row => {{
        const target = nextTarget(row);
        const activation = effectiveActivation(row);
        const vRank = visualRank(row);
        const noharm = row.hard_negative === 0 ? "clean" : "flag";
        return `<tr data-id="${{htmlEscape(row.operator_id)}}">
          <td><span class="pill ${{String(vRank).toLowerCase()}}">${{htmlEscape(vRank)}}</span><br><span class="detail-id">${{htmlEscape(row.rank)}}</span></td>
          <td><strong>${{htmlEscape(row.display_name || row.operator_id)}}</strong><br><span class="detail-id">${{htmlEscape(row.operator_id)}}</span></td>
          <td>${{htmlEscape(row.scope)}}<br><span class="detail-id">${{htmlEscape(row.group_id)}} · ${{htmlEscape(row.family)}}</span></td>
          <td>${{fmt(activation)}}<div class="bar"><span style="width:${{(target.progress*100).toFixed(1)}}%"></span></div></td>
          <td>${{target.name}}<br><span class="detail-id">${{fmt(target.remain)}} remaining</span></td>
          <td><span class="pill ${{noharm === "clean" ? "green" : "red"}}">${{noharm}}</span></td>
        </tr>`;
      }}).join("");
      document.querySelectorAll("tbody tr").forEach(tr => tr.onclick = () => select(tr.dataset.id));
    }}
    function htmlEscape(value) {{
      return String(value ?? "").replace(/[&<>"']/g, ch => ({{"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;","'":"&#39;"}}[ch]));
    }}
    function select(id) {{
      state.selected = rows.find(r => r.operator_id === id) || null;
      renderDetail();
    }}
    function renderDetail(items = filtered()) {{
      if (state.selected && !items.some(row => row.operator_id === state.selected.operator_id)) {{
        state.selected = null;
      }}
      const row = state.selected || items[0] || rows[0];
      state.selected = row;
      const target = nextTarget(row);
      const activation = effectiveActivation(row);
      const vRank = visualRank(row);
      document.getElementById("detail").innerHTML = `
        <div class="detail-title">${{htmlEscape(row.display_name || row.operator_id)}}</div>
        <div class="detail-id">${{htmlEscape(row.operator_id)}}</div>
        <div style="margin-top:12px;display:flex;gap:8px;flex-wrap:wrap">
          <span class="pill ${{String(vRank).toLowerCase()}}">${{htmlEscape(vRank)}}</span>
          <span class="pill">${{htmlEscape(row.rank)}}</span>
          <span class="pill">${{htmlEscape(row.scope)}}</span>
          <span class="pill">${{htmlEscape(row.watch_state)}}</span>
        </div>
        <div class="kv">
          <div>Effective activation</div><div>${{fmt(activation)}}${{row.e117_activation_after_gauntlet ? " · includes E117 gauntlet" : ""}}</div>
          <div>Base rank activation</div><div>${{fmt(row.qualified_activation)}}</div>
          <div>Next target</div><div>${{target.name}} · ${{fmt(target.remain)}} remaining<div class="bar"><span style="width:${{(target.progress*100).toFixed(1)}}%"></span></div></div>
          <div>Positive</div><div>${{fmt(row.positive)}}</div>
          <div>Neutral valid</div><div>${{fmt(row.neutral_valid)}}</div>
          <div>Neutral waste</div><div>${{fmt(row.neutral_waste)}} (${{pct(row.neutral_waste_rate)}})</div>
          <div>Hard negative</div><div>${{fmt(row.hard_negative)}}</div>
          <div>95% upper fail bound</div><div>${{pct(row.rule_of_three_upper_failure_bound)}}</div>
          <div>Family coverage</div><div>${{row.group_id === "E120" ? "E120 FineWeb farm = " + fmt(row.combined_family_coverage) : "E107 " + fmt(row.e107_family_coverage) + " + E108 " + fmt(row.e108_family_coverage) + " = " + fmt(row.combined_family_coverage)}}</div>
          <div>Campaign count</div><div>${{fmt(row.campaign_count)}}</div>
          <div>Counterfactual value</div><div>${{fmt(row.counterfactual_value)}} · gain ${{fmt(row.activated_gain)}} · ablation ${{fmt(row.ablation_loss)}}</div>
          <div>Reload / Challenger / Prune</div><div>${{row.reload_shadow_pass ? "reload pass" : "reload no"}} · ${{row.challenger_pass ? "challenger pass" : "challenger no"}} · ${{row.prune_pass ? "prune pass" : "prune no"}}</div>
          <div>Status source</div><div>E107 ${{htmlEscape(row.e107_status)}} · E108 ${{htmlEscape(row.e108_status)}}${{row.e110_wave1_outcome ? " · E110 " + htmlEscape(row.e110_wave1_outcome) : ""}}${{row.e111_wave2_outcome ? " · E111 " + htmlEscape(row.e111_wave2_outcome) : ""}}${{row.e112_wave3_outcome ? " · E112 " + htmlEscape(row.e112_wave3_outcome) : ""}}</div>
          <div>Latest activation add</div><div>${{fmt(row.qualified_activation_add || 0)}}</div>
          <div>Selected variant</div><div>${{htmlEscape(row.selected_variant_type || "")}}${{typeof row.selected_prune_ratio === "number" ? " · prune " + (row.selected_prune_ratio * 100).toFixed(1) + "%" : ""}}<br><span class="detail-id">${{htmlEscape(row.selected_variant_id || "")}}</span></div>
          <div>Mutation budget</div><div>${{fmt(row.mutation_attempts || 0)}} attempts · ${{fmt(row.accepted_mutations || 0)}} accepted · ${{fmt(row.rollback_count || 0)}} rollback</div>
          <div>Core no-harm</div><div>${{row.long_horizon_no_harm_pass ? "long-horizon pass" : ""}}${{row.negative_scope_pass ? " · negative-scope pass" : ""}}</div>
          <div>E114 FineWeb calls</div><div>${{fmt(row.e114_current_run_calls || 0)}} in 1M · projected full ${{fmt(row.e114_projected_full_fineweb_calls || 0)}}</div>
          <div>E114 PermaCore projection</div><div>${{row.e114_projected_reaches_permacore_probation ? "reaches 300k with full FineWeb" : "targeted pressure data needed"}} · remaining after full ${{fmt(row.e114_projected_remaining_after_full_fineweb || 0)}}</div>
          <div>E114 selected policy</div><div>${{htmlEscape(row.e114_selected_variant || "")}}</div>
          <div>E116 synthetic pressure</div><div>${{row.e116_reaches_permacore_probation_after_targeted_pressure ? "reaches 300k with targeted synthetic pressure" : "not targeted / still short"}} · +${{fmt(row.e116_qualified_synthetic_pressure_activation || 0)}} scheduled activations</div>
          <div>E116 generated data</div><div>${{htmlEscape(row.e116_template_family || "")}} · ${{fmt(row.e116_generated_cell_packs || 0)}} packs · ${{fmt(row.e116_variant_count || 0)}} variants · repeat ${{fmt(row.e116_repeat_count_per_pack || 0)}}</div>
          <div>E116 projected activation</div><div>${{fmt(row.e116_projected_activation_after_targeted_pressure || 0)}} after targeted pressure</div>
          <div>E117 gauntlet activation</div><div>${{row.e117_reaches_permacore_probation_after_gauntlet ? "actual gauntlet reaches 300k" : "not reached in gauntlet"}} · +${{fmt(row.e117_qualified_activation || 0)}} qualified · hard negatives ${{fmt(row.e117_hard_negative || 0)}}</div>
          <div>E117 activation mix</div><div>positive ${{fmt(row.e117_positive_activation || 0)}} · neutral ${{fmt(row.e117_neutral_valid_activation || 0)}} · negative-scope valid ${{fmt(row.e117_negative_scope_valid_activation || 0)}}</div>
          <div>E117 remaining</div><div>${{fmt(row.e117_remaining_after_gauntlet || 0)}} after actual targeted gauntlet</div>
          <div>E118 cross-source</div><div>${{row.e118_cross_source_no_harm_pass ? "cross-source no-harm pass" : "not run / not passed"}} · families ${{fmt(row.e118_source_family_coverage || 0)}} · hard negatives ${{fmt(row.e118_hard_negative || 0)}} · imprint ${{fmt(row.e118_synthetic_imprint || 0)}}</div>
          <div>E118 ablation value</div><div>${{fmt(row.e118_ablation_value || 0)}} · cases ${{fmt(row.e118_case_count || 0)}}</div>
          <div>E120 FineWeb farm</div><div>${{row.e120_saved_operator ? "new scoped Gold from FineWeb skill farm" : "not E120"}} · support ${{fmt(row.e120_support_count || 0)}} · hard negatives ${{fmt(row.e120_hard_negative || 0)}} · wrong scope ${{fmt(row.e120_wrong_scope_call || 0)}}</div>
          <div>E120 variant</div><div>${{htmlEscape(row.e120_selected_variant_type || "")}}${{typeof row.e120_selected_prune_ratio === "number" ? " · prune " + (row.e120_selected_prune_ratio * 100).toFixed(1) + "%" : ""}} · reload ${{row.e120_reload_shadow_pass ? "pass" : "n/a"}} · negative scope ${{row.e120_negative_scope_pass ? "pass" : "n/a"}} · challenger ${{row.e120_challenger_pass ? "pass" : "n/a"}} · prune ${{row.e120_prune_pass ? "pass" : "n/a"}}</div>
          <div>E120 description</div><div>${{htmlEscape(row.e120_description || "")}}</div>
          <div>E121 Orange probation</div><div>${{row.e121_reaches_orange_legendary ? "Orange/LegendaryCandidate reached" : "not E121 Orange"}} · remaining ${{fmt(row.e121_remaining_to_orange || 0)}} · hard negatives ${{fmt(row.e121_hard_negative || 0)}} · wrong scope ${{fmt(row.e121_wrong_scope_call || 0)}} · direct writes ${{fmt(row.e121_direct_flow_write || 0)}}</div>
          <div>E121 selected form</div><div>${{htmlEscape(row.e121_selected_variant_type || "")}}${{typeof row.e121_selected_prune_ratio === "number" ? " · prune " + (row.e121_selected_prune_ratio * 100).toFixed(1) + "%" : ""}} · families ${{fmt(row.e121_family_coverage || 0)}} · campaigns ${{fmt(row.e121_campaign_count || 0)}}</div>
          <div>E122 orange-only baseline</div><div>${{row.e122_orange_only_baseline ? "active orange-only baseline member" : "not active in E122"}} · was already orange ${{row.e122_was_previously_orange ? "yes" : "no"}} · remaining ${{fmt(row.e122_remaining_to_orange || 0)}}</div>
          <div>E122 negative cards</div><div>${{fmt(row.e122_negative_card_count || 0)}} cards · ${{fmt(row.e122_negative_card_recall_count || 0)}} recall events · ${{fmt(row.e122_prevented_repeat_failure_count || 0)}} prevented repeats · false blocks ${{fmt(row.e122_false_block_count || 0)}}</div>
          <div>E127 overnight farm</div><div>${{row.group_id === "E127" ? "new scoped Orange/Legendary from overnight farm" : "not E127"}} · cycle ${{fmt(row.e127_cycle || 0)}} · hard negatives ${{fmt(row.e127_hard_negative || 0)}} · wrong scope ${{fmt(row.e127_wrong_scope_call || 0)}} · false commits ${{fmt(row.e127_false_commit || 0)}}</div>
          <div>E127 selected form</div><div>${{htmlEscape(row.e127_selected_variant_type || "")}}${{typeof row.e127_selected_prune_ratio === "number" ? " · prune " + (row.e127_selected_prune_ratio * 100).toFixed(1) + "%" : ""}} · families ${{fmt(row.e127_family_coverage || 0)}} · campaigns ${{fmt(row.e127_campaign_count || 0)}}</div>
          <div>E127 description</div><div>${{htmlEscape(row.e127_description || "")}}</div>
          <div>E129 arithmetic trace</div><div>${{row.e129_arithmetic_trace_operator ? "scoped arithmetic Orange/LegendaryCandidate" : "not E129 arithmetic"}} · min accuracy ${{pct(row.e129_min_in_scope_accuracy || 0)}} · negative-scope cases ${{fmt(row.e129_negative_scope_case_count || 0)}} · pass ${{pct(row.e129_negative_scope_pass_rate || 0)}}</div>
          <div>E129 safety counters</div><div>wrong scope ${{fmt(row.e129_wrong_scope_call || 0)}} · false commits ${{fmt(row.e129_false_commit || 0)}} · unsupported ${{fmt(row.e129_unsupported_answer || 0)}} · variant cost ${{fmt(row.e129_selected_variant_cost || 0)}} · groups ${{fmt(row.e129_campaign_group_count || 0)}}</div>
          <div>E130A backfill</div><div>${{row.e130a_reaches_orange_legendary ? "CoreMemoryCandidate -> Orange/LegendaryCandidate" : "not E130A backfill"}} · source ${{htmlEscape(row.e130a_source_rank || "")}} · before ${{fmt(row.e130a_activation_before || 0)}} · add ${{fmt(row.e130a_activation_add || 0)}} · remaining ${{fmt(row.e130a_remaining_to_orange || 0)}}</div>
          <div>E130A safety counters</div><div>hard negatives ${{fmt(row.e130a_hard_negative || 0)}} · wrong scope ${{fmt(row.e130a_wrong_scope_call || 0)}} · false commits ${{fmt(row.e130a_false_commit || 0)}} · unsupported ${{fmt(row.e130a_unsupported_answer || 0)}} · direct writes ${{fmt(row.e130a_direct_flow_write || 0)}} · negative transfer ${{fmt(row.e130a_negative_transfer || 0)}} · pressure families ${{fmt(row.e130a_pressure_family_count || 0)}}</div>
          <div>E130B text IO</div><div>${{row.e130b_text_io_transfer ? "visible arithmetic text-IO transfer confirmed" : "not E130B transfer"}} · route ${{htmlEscape(row.e130b_selected_route || "")}} · visible cases ${{fmt(row.e130b_visible_transfer_case_count || 0)}} · visible accuracy ${{pct(row.e130b_visible_transfer_accuracy || 0)}} · word no-call cases ${{fmt(row.e130b_word_problem_no_call_case_count || 0)}} · word no-call ${{pct(row.e130b_word_problem_no_call_accuracy || 0)}}</div>
          <div>E130B safety counters</div><div>hard negatives ${{fmt(row.e130b_hard_negative || 0)}} · wrong scope ${{fmt(row.e130b_wrong_scope_call || 0)}} · false commits ${{fmt(row.e130b_false_commit || 0)}} · unsupported ${{fmt(row.e130b_unsupported_answer || 0)}} · direct writes ${{fmt(row.e130b_direct_flow_write || 0)}} · overbroad control wrong-scope ${{fmt(row.e130b_overbroad_control_wrong_scope_call || 0)}}</div>
          <div>E131 visible equation</div><div>${{row.e131_visible_equation_transfer ? "visible-equation assistant render confirmed" : "not E131 transfer"}} · route ${{htmlEscape(row.e131_selected_route || "")}} · visible cases ${{fmt(row.e131_visible_equation_case_count || 0)}} · extraction ${{pct(row.e131_visible_equation_extraction_accuracy || 0)}} · word no-call cases ${{fmt(row.e131_word_problem_no_call_case_count || 0)}} · word no-call ${{pct(row.e131_word_problem_no_call_accuracy || 0)}}</div>
          <div>E131 safety counters</div><div>hard negatives ${{fmt(row.e131_hard_negative || 0)}} · wrong scope ${{fmt(row.e131_wrong_scope_call || 0)}} · false commits ${{fmt(row.e131_false_commit || 0)}} · unsupported ${{fmt(row.e131_unsupported_answer || 0)}} · boundary claims ${{fmt(row.e131_boundary_claim_violation || 0)}} · direct writes ${{fmt(row.e131_direct_flow_write || 0)}} · E130B baseline misses ${{fmt(row.e131_e130b_baseline_visible_miss || 0)}} · overbroad wrong-scope ${{fmt(row.e131_overbroad_control_wrong_scope_call || 0)}}</div>
          <div>E132 math-text farm</div><div>${{row.e132_math_text_skill_operator ? "external math-text Orange cycle confirmed" : "not E132 math-text"}} · support ${{fmt(row.e132_external_support_count || 0)}} · sources ${{fmt(row.e132_external_source_count || 0)}} · source families ${{fmt(row.e132_external_family_count || 0)}} · negative-scope cases ${{fmt(row.e132_negative_scope_case_count || 0)}} · no-call ${{pct(row.e132_negative_scope_pass_rate || 0)}}</div>
          <div>E132 safety/control</div><div>hard negatives ${{fmt(row.e132_hard_negative || 0)}} · wrong scope ${{fmt(row.e132_wrong_scope_call || 0)}} · false commits ${{fmt(row.e132_false_commit || 0)}} · unsupported ${{fmt(row.e132_unsupported_answer || 0)}} · boundary claims ${{fmt(row.e132_boundary_claim_violation || 0)}} · direct writes ${{fmt(row.e132_direct_flow_write || 0)}} · overbroad solver wrong-scope ${{fmt(row.e132_overbroad_solver_control_wrong_scope_call || 0)}}</div>
          <div>E132 selected form</div><div>${{htmlEscape(row.e132_selected_variant_type || "")}}${{typeof row.e132_selected_prune_ratio === "number" ? " · prune " + (row.e132_selected_prune_ratio * 100).toFixed(1) + "%" : ""}} · pressure families ${{fmt(row.e132_pressure_family_count || 0)}}</div>
          <div>E133 route composition</div><div>${{row.e133_math_text_route_composition ? "math-text route composition confirmed" : "not E133 route composition"}} · route ${{htmlEscape(row.selected_route || "")}} · cases ${{fmt(row.e133_route_case_count || 0)}} · accuracy ${{pct(row.e133_route_accuracy || 0)}} · visible arithmetic ${{fmt(row.e133_visible_arithmetic_route_case_count || 0)}} @ ${{pct(row.e133_visible_arithmetic_route_accuracy || 0)}}</div>
          <div>E133 no-solve guards</div><div>structural guard ${{fmt(row.e133_structural_guard_case_count || 0)}} @ ${{pct(row.e133_structural_guard_accuracy || 0)}} · hidden word no-call ${{fmt(row.e133_hidden_word_problem_no_solve_case_count || 0)}} @ ${{pct(row.e133_hidden_word_problem_no_solve_accuracy || 0)}} · qualified route ${{fmt(row.e133_qualified_route_activation || 0)}}</div>
          <div>E133 safety/control</div><div>hard negatives ${{fmt(row.e133_hard_negative || 0)}} · wrong scope ${{fmt(row.e133_wrong_scope_call || 0)}} · false commits ${{fmt(row.e133_false_commit || 0)}} · unsupported ${{fmt(row.e133_unsupported_answer || 0)}} · boundary claims ${{fmt(row.e133_boundary_claim_violation || 0)}} · direct writes ${{fmt(row.e133_direct_flow_write || 0)}} · overbroad wrong-scope ${{fmt(row.e133_overbroad_solver_control_wrong_scope_call || 0)}} · trust-control false commits ${{fmt(row.e133_trust_control_false_commit || 0)}} · trust-control direct writes ${{fmt(row.e133_trust_control_direct_flow_write || 0)}}</div>
          <div>E134 OOD route stress</div><div>${{row.e134_external_math_text_ood_route_stress ? "external OOD route stress confirmed" : "not E134 OOD stress"}} · route ${{htmlEscape(row.selected_route || "")}} · OOD cases ${{fmt(row.e134_ood_case_count || 0)}} @ ${{pct(row.e134_ood_route_accuracy || 0)}} · visible arithmetic OOD ${{fmt(row.e134_visible_arithmetic_ood_case_count || 0)}} @ ${{pct(row.e134_visible_arithmetic_ood_accuracy || 0)}}</div>
          <div>E134 no-solve / counterexamples</div><div>structural OOD ${{fmt(row.e134_structural_guard_ood_case_count || 0)}} @ ${{pct(row.e134_structural_guard_ood_accuracy || 0)}} · hidden word OOD no-call ${{fmt(row.e134_hidden_word_problem_ood_no_solve_case_count || 0)}} @ ${{pct(row.e134_hidden_word_problem_ood_no_solve_accuracy || 0)}} · counterexamples ${{fmt(row.e134_counterexample_case_count || 0)}} @ ${{pct(row.e134_counterexample_accuracy || 0)}} · qualified OOD route ${{fmt(row.e134_qualified_ood_route_activation || 0)}}</div>
          <div>E134 safety/control</div><div>hard negatives ${{fmt(row.e134_hard_negative || 0)}} · wrong scope ${{fmt(row.e134_wrong_scope_call || 0)}} · false commits ${{fmt(row.e134_false_commit || 0)}} · unsupported ${{fmt(row.e134_unsupported_answer || 0)}} · boundary claims ${{fmt(row.e134_boundary_claim_violation || 0)}} · direct writes ${{fmt(row.e134_direct_flow_write || 0)}} · E133 baseline misses ${{fmt(row.e134_e133_baseline_ood_miss || 0)}} · overbroad wrong-scope ${{fmt(row.e134_overbroad_solver_control_wrong_scope_call || 0)}} · trust-control false commits ${{fmt(row.e134_trust_control_false_commit || 0)}} · trust-control direct writes ${{fmt(row.e134_trust_control_direct_flow_write || 0)}}</div>
          <div>E135 dialogue state</div><div>${{row.e135_math_text_multi_route_dialogue_state ? "multi-route dialogue-state confirmed" : "not E135 dialogue"}} · route ${{htmlEscape(row.selected_route || "")}} · cases ${{fmt(row.e135_dialogue_case_count || 0)}} · turns ${{fmt(row.e135_dialogue_turn_count || 0)}} · dialogue ${{pct(row.e135_dialogue_state_accuracy || 0)}} · current route ${{pct(row.e135_current_turn_route_accuracy || 0)}} · state integrity ${{pct(row.e135_route_state_integrity || 0)}}</div>
          <div>E135 route-state guards</div><div>hidden no-solve ${{fmt(row.e135_hidden_word_problem_dialogue_no_solve_case_count || 0)}} @ ${{pct(row.e135_hidden_word_problem_dialogue_no_solve_accuracy || 0)}} · visible reentry ${{fmt(row.e135_visible_reentry_dialogue_case_count || 0)}} @ ${{pct(row.e135_visible_reentry_dialogue_accuracy || 0)}} · stale rejection ${{fmt(row.e135_stale_route_rejection_case_count || 0)}} @ ${{pct(row.e135_stale_route_rejection_accuracy || 0)}} · cross-thread rejection ${{fmt(row.e135_cross_thread_rejection_case_count || 0)}} @ ${{pct(row.e135_cross_thread_rejection_accuracy || 0)}} · counterexamples ${{fmt(row.e135_counterexample_dialogue_case_count || 0)}} @ ${{pct(row.e135_counterexample_dialogue_accuracy || 0)}}</div>
          <div>E135 safety/control</div><div>hard negatives ${{fmt(row.e135_hard_negative || 0)}} · wrong scope ${{fmt(row.e135_wrong_scope_call || 0)}} · false commits ${{fmt(row.e135_false_commit || 0)}} · unsupported ${{fmt(row.e135_unsupported_answer || 0)}} · boundary claims ${{fmt(row.e135_boundary_claim_violation || 0)}} · direct writes ${{fmt(row.e135_direct_flow_write || 0)}} · stale reuse ${{fmt(row.e135_stale_route_reuse || 0)}} · cross-thread contamination ${{fmt(row.e135_cross_thread_contamination || 0)}} · controls ${{fmt((row.e135_latest_route_reuse_control_failure || 0) + (row.e135_stale_route_reuse_control_failure || 0) + (row.e135_cross_thread_contamination_control_failure || 0) + (row.e135_counterexample_trust_control_failure || 0) + (row.e135_single_turn_reset_control_failure || 0))}}</div>
          <div>E136A assistant/text farm</div><div>${{row.e136a_assistant_text_skill_operator ? "assistant/text Orange cycle confirmed" : "not E136A assistant/text"}} · support ${{fmt(row.e136a_external_support_count || 0)}} · sources ${{fmt(row.e136a_external_source_count || 0)}} · families ${{fmt(row.e136a_external_family_count || 0)}} · licenses ${{fmt(row.e136a_external_license_count || 0)}} · negative-scope cases ${{fmt(row.e136a_negative_scope_case_count || 0)}} · pass ${{pct(row.e136a_negative_scope_pass_rate || 0)}}</div>
          <div>E136A safety/control</div><div>hard negatives ${{fmt(row.e136a_hard_negative || 0)}} · wrong scope ${{fmt(row.e136a_wrong_scope_call || 0)}} · false commits ${{fmt(row.e136a_false_commit || 0)}} · unsupported ${{fmt(row.e136a_unsupported_answer || 0)}} · boundary claims ${{fmt(row.e136a_boundary_claim_violation || 0)}} · direct writes ${{fmt(row.e136a_direct_flow_write || 0)}} · overbroad chatbot wrong-scope ${{fmt(row.e136a_overbroad_chatbot_control_wrong_scope_call || 0)}}</div>
          <div>E136A selected form</div><div>${{htmlEscape(row.e136a_selected_variant_type || "")}}${{typeof row.e136a_selected_prune_ratio === "number" ? " · prune " + (row.e136a_selected_prune_ratio * 100).toFixed(1) + "%" : ""}} · pressure families ${{fmt(row.e136a_pressure_family_count || 0)}}<br>${{htmlEscape(row.e136a_description || "")}}</div>
          <div>E136B route composition</div><div>${{row.e136b_assistant_text_route_composition ? "assistant/text route composition confirmed" : "not E136B route"}} · cases ${{fmt(row.e136b_route_case_count || 0)}} · route ${{pct(row.e136b_route_accuracy || 0)}} · stack ${{pct(row.e136b_route_stack_accuracy || 0)}} · primary ${{pct(row.e136b_primary_route_accuracy || 0)}} · boundary ${{pct(row.e136b_boundary_accuracy || 0)}}</div>
          <div>E136B route boundaries</div><div>multi-route ${{fmt(row.e136b_multi_route_composition_case_count || 0)}} @ ${{pct(row.e136b_multi_route_composition_accuracy || 0)}} · boundary cases ${{fmt(row.e136b_boundary_case_count || 0)}} @ ${{pct(row.e136b_boundary_case_accuracy || 0)}} · negative scope ${{fmt(row.e136b_negative_scope_case_count || 0)}} @ ${{pct(row.e136b_negative_scope_accuracy || 0)}}</div>
          <div>E136B safety/control</div><div>hard negatives ${{fmt(row.e136b_hard_negative || 0)}} · wrong scope ${{fmt(row.e136b_wrong_scope_call || 0)}} · false commits ${{fmt(row.e136b_false_commit || 0)}} · unsupported ${{fmt(row.e136b_unsupported_answer || 0)}} · boundary claims ${{fmt(row.e136b_boundary_claim_violation || 0)}} · direct writes ${{fmt(row.e136b_direct_flow_write || 0)}} · overbroad chatbot wrong-scope ${{fmt(row.e136b_overbroad_chatbot_control_wrong_scope_call || 0)}} · unsafe direct-write control ${{fmt(row.e136b_unsafe_direct_write_control_direct_flow_write || 0)}} · source hallucination unsupported ${{fmt(row.e136b_source_hallucination_control_unsupported_answer || 0)}}</div>
        </div>
        <div class="note">${{row.group_id === "E136B" ? "Interpretation: this E136B operator composes E136A assistant/text lenses and guards into bounded route stacks. It confirms route composition and boundary handling in a controlled assistant/text proxy, but it is still not neural training, open-domain assistant readiness, production assistant behavior, Core, PermaCore, or TrueGolden." : row.group_id === "E136A" ? "Interpretation: this E136A operator was farmed from the local assistant/text seed pack and passed scoped Orange/Legendary probation. It widens assistant-text operator coverage, but it is still not neural training, open-domain assistant readiness, production assistant behavior, Core, PermaCore, or TrueGolden." : row.group_id === "E135" ? "Interpretation: this E135 operator survived controlled multi-route assistant dialogue-state pressure on top of E134. It proves current-turn route-state integrity across stale, cross-thread, hidden no-solve, visible reentry, and counterexample turns. It is still not open-domain dialogue, MATH/GSM8K solving, hidden word-problem solving, Core, PermaCore, or TrueGolden." : row.group_id === "E134" ? "Interpretation: this E134 operator survived OOD route stress and counterexample rejection on top of the E133 route-composition contract. It widens evidence for route robustness, but it is still not MATH/GSM8K solving, hidden word-problem solving, Core, PermaCore, or TrueGolden." : row.group_id === "E133" ? "Interpretation: this E133 operator composes a scoped E132 math-text lens/guard with assistant route decisions. Visible arithmetic can route to the scoped arithmetic renderer; structural math text and hidden prose word problems stay guarded. This is not MATH/GSM8K solving, natural-language word-problem solving, Core, PermaCore, or TrueGolden." : row.group_id === "E132" ? "Interpretation: this E132 operator is a scoped math-text lens/guard farmed from external math text. It prepares or guards notation/proof/TIR/word-problem surfaces; it is not a math benchmark solver, natural-language word-problem solver, Core, PermaCore, or TrueGolden." : row.group_id === "E131" ? "Interpretation: this E131 operator routes assistant-style visible equation surfaces into the scoped E129 arithmetic trace engine and still no-calls hidden prose-only word problems. This is not natural-language word-problem solving." : row.group_id === "E130B" ? "Interpretation: this E130B operator transferred E129 arithmetic trace behavior into visible-expression text IO and still no-calls hidden word problems. This is not natural-language word-problem solving." : row.group_id === "E130A" ? "Interpretation: this E130A operator was backfilled from CoreMemoryCandidate to scoped Orange/LegendaryCandidate with the E121-style 300k activation and no-harm gate. It is still not PermaCore or TrueGolden." : row.group_id === "E129" ? "Interpretation: this E129 operator reached scoped Orange/LegendaryCandidate status for exact arithmetic expression/trace behavior. It can compute or validate visible arithmetic traces, but it is not natural-language word-problem solving, PermaCore, or TrueGolden." : row.group_id === "E127" ? "Interpretation: this E127 operator reached scoped Orange/LegendaryCandidate status during the overnight text-skill farm. It is still not Core, PermaCore, or TrueGolden; that requires much larger no-harm grind and cross-source evidence." : row.e122_orange_only_baseline ? "Interpretation: this active Operator is part of the E122 scoped orange-only baseline. Negative cards attached here are mutation-planner priors, not normal callable skills. It is still not Core, PermaCore, or TrueGolden." : row.rank === "OrangeLegendaryCandidate" ? "Interpretation: this operator reached scoped Orange/LegendaryCandidate status. It is still not Core, PermaCore, or TrueGolden; that would need a later much larger no-harm grind." : row.group_id === "E120" ? "Interpretation: E120 created this as a scoped Gold Operator from FineWeb skill farming. It is not Core, PermaCore, or TrueGolden yet." : row.rank === "CoreMemoryCandidate" ? "Interpretation: this operator passed scoped CoreMemoryCandidate probation. It is still not PermaCore or TrueGolden without a later larger no-harm grind." : "Interpretation: rank is scoped. This operator is not Core memory unless a later Core probation grind passes the much higher qualified-activation and no-harm gates."}}</div>
      `;
    }}
    function render() {{
      renderCards();
      renderFilters();
      const items = filtered();
      renderGrid(items);
      renderRows(items);
      renderDetail(items);
    }}
    document.getElementById("search").oninput = e => {{ state.q = e.target.value; render(); }};
    document.getElementById("rankFilter").onchange = e => {{ state.rank = e.target.value; render(); }};
    document.getElementById("scopeFilter").onchange = e => {{ state.scope = e.target.value; render(); }};
    document.getElementById("sortBy").onchange = e => {{ state.sort = e.target.value; render(); }};
    render();
  </script>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e109", default=str(DEFAULT_E109))
    parser.add_argument("--e110", default=str(DEFAULT_E110))
    parser.add_argument("--e111", default=str(DEFAULT_E111))
    parser.add_argument("--e112", default=str(DEFAULT_E112))
    parser.add_argument("--e114", default=str(DEFAULT_E114))
    parser.add_argument("--e116", default=str(DEFAULT_E116))
    parser.add_argument("--e117", default=str(DEFAULT_E117))
    parser.add_argument("--e118", default=str(DEFAULT_E118))
    parser.add_argument("--e120", default=str(DEFAULT_E120))
    parser.add_argument("--e121", default=str(DEFAULT_E121))
    parser.add_argument("--e122", default=str(DEFAULT_E122))
    parser.add_argument("--e127", default=str(DEFAULT_E127))
    parser.add_argument("--e129", default=str(DEFAULT_E129))
    parser.add_argument("--e130a", default=str(DEFAULT_E130A))
    parser.add_argument("--e130b", default=str(DEFAULT_E130B))
    parser.add_argument("--e131", default=str(DEFAULT_E131))
    parser.add_argument("--e132", default=str(DEFAULT_E132))
    parser.add_argument("--e133", default=str(DEFAULT_E133))
    parser.add_argument("--e134", default=str(DEFAULT_E134))
    parser.add_argument("--e135", default=str(DEFAULT_E135))
    parser.add_argument("--e136a", default=str(DEFAULT_E136A))
    parser.add_argument("--e136b", default=str(DEFAULT_E136B))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    e109 = existing_artifact_path(Path(args.e109), SAMPLE_E109, "rank_results.json")
    e110_requested = Path(args.e110)
    e111_requested = Path(args.e111)
    e112_requested = Path(args.e112)
    e114_requested = Path(args.e114)
    e116_requested = Path(args.e116)
    e117_requested = Path(args.e117)
    e118_requested = Path(args.e118)
    e120_requested = Path(args.e120)
    e121_requested = Path(args.e121)
    e122_requested = Path(args.e122)
    e127_requested = Path(args.e127)
    e129_requested = Path(args.e129)
    e130a_requested = Path(args.e130a)
    e130b_requested = Path(args.e130b)
    e131_requested = Path(args.e131)
    e132_requested = Path(args.e132)
    e133_requested = Path(args.e133)
    e134_requested = Path(args.e134)
    e135_requested = Path(args.e135)
    e136a_requested = Path(args.e136a)
    e136b_requested = Path(args.e136b)
    e110 = e110_requested if (e110_requested / "wave_results.json").exists() else SAMPLE_E110 if (SAMPLE_E110 / "wave_results.json").exists() else None
    e111 = e111_requested if (e111_requested / "wave_results.json").exists() else SAMPLE_E111 if (SAMPLE_E111 / "wave_results.json").exists() else None
    e112 = e112_requested if (e112_requested / "wave_results.json").exists() else SAMPLE_E112 if (SAMPLE_E112 / "wave_results.json").exists() else None
    e114 = e114_requested if (e114_requested / "operator_projection_report.json").exists() else None
    e116 = e116_requested if (e116_requested / "operator_target_coverage.json").exists() else None
    e117 = e117_requested if (e117_requested / "operator_gauntlet_results.json").exists() else None
    e118 = e118_requested if (e118_requested / "operator_cross_source_results.json").exists() else None
    e120 = e120_requested if (e120_requested / "operator_gold_results.json").exists() else None
    e121 = e121_requested if (e121_requested / "operator_orange_results.json").exists() else None
    e122 = e122_requested if (e122_requested / "orange_only_results.json").exists() else None
    e127 = e127_requested if (e127_requested / "cycles").exists() else SAMPLE_E127 if (SAMPLE_E127 / "cycles").exists() else None
    e129 = e129_requested if (e129_requested / "operator_orange_results.json").exists() else SAMPLE_E129 if (SAMPLE_E129 / "operator_orange_results.json").exists() else None
    e130a = e130a_requested if (e130a_requested / "operator_orange_results.json").exists() else SAMPLE_E130A if (SAMPLE_E130A / "operator_orange_results.json").exists() else None
    e130b = e130b_requested if (e130b_requested / "operator_transfer_results.json").exists() else SAMPLE_E130B if (SAMPLE_E130B / "operator_transfer_results.json").exists() else None
    e131 = e131_requested if (e131_requested / "operator_transfer_results.json").exists() else SAMPLE_E131 if (SAMPLE_E131 / "operator_transfer_results.json").exists() else None
    e132 = e132_requested if (e132_requested / "operator_orange_results.json").exists() else SAMPLE_E132 if (SAMPLE_E132 / "operator_orange_results.json").exists() else None
    e133 = e133_requested if (e133_requested / "operator_route_results.json").exists() else SAMPLE_E133 if (SAMPLE_E133 / "operator_route_results.json").exists() else None
    e134 = e134_requested if (e134_requested / "operator_ood_results.json").exists() else SAMPLE_E134 if (SAMPLE_E134 / "operator_ood_results.json").exists() else None
    e135 = e135_requested if (e135_requested / "operator_dialogue_results.json").exists() else SAMPLE_E135 if (SAMPLE_E135 / "operator_dialogue_results.json").exists() else None
    e136a = e136a_requested if (e136a_requested / "operator_orange_results.json").exists() else SAMPLE_E136A if (SAMPLE_E136A / "operator_orange_results.json").exists() else None
    e136b = e136b_requested if (e136b_requested / "operator_route_results.json").exists() else SAMPLE_E136B if (SAMPLE_E136B / "operator_route_results.json").exists() else None
    out = Path(args.out)
    payload = build_payload(e109, e110, e111, e112, e114, e116, e117, e118, e120, e121, e122, e127, e129, e130a, e130b, e131, e132, e133, e134, e135, e136a, e136b)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(payload), encoding="utf-8")
    print(json.dumps({"out": str(out), "operator_count": len(payload["rows"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
