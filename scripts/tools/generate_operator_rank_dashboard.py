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
SAMPLE_E109 = Path("docs/research/artifact_samples/e109_operator_rank_ladder_and_golden_watch_probation_mode")
SAMPLE_E110 = Path("docs/research/artifact_samples/e110_promote_or_drop_operator_grind_wave1")
SAMPLE_E111 = Path("docs/research/artifact_samples/e111_bronze_mutation_prune_promote_or_drop_wave")
SAMPLE_E112 = Path("docs/research/artifact_samples/e112_gold_to_core_prune_heavy_probation_wave")
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
    }
    summary = read_json(e109 / "summary.json")
    summary = {
        **summary,
        "rank_counts": counts,
        "latest_wave": "E122 orange-only baseline and negative-card recall" if e122_payload else "E121 E120 Gold to Orange/Legendary probation gauntlet" if e121_payload else "E120 FineWeb skill farm to Gold wave" if e120_payload else "E118 cross-source no-harm gauntlet" if e118_payload else "E117 alpha-Weave targeted pressure gauntlet" if e117_payload else "E116 alpha-Weave targeted pressure" if e116_payload else "E114 FineWeb projection" if e114_payload else "E112 Wave 3" if e112_payload else "E111 Wave 2" if e111_payload else "E110 Wave 1" if e110_payload else "E109",
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
        ["False blocks", agg.e122_false_block_count ?? "n/a", agg.e122_false_block_count ? "red" : "green"]
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
        </div>
        <div class="note">${{row.e122_orange_only_baseline ? "Interpretation: this active Operator is part of the E122 scoped orange-only baseline. Negative cards attached here are mutation-planner priors, not normal callable skills. It is still not Core, PermaCore, or TrueGolden." : row.rank === "OrangeLegendaryCandidate" ? "Interpretation: this E121 operator reached scoped Orange/LegendaryCandidate status. It is still not Core, PermaCore, or TrueGolden; that would need a later much larger no-harm grind." : row.group_id === "E120" ? "Interpretation: E120 created this as a scoped Gold Operator from FineWeb skill farming. It is not Core, PermaCore, or TrueGolden yet." : row.rank === "CoreMemoryCandidate" ? "Interpretation: this operator passed scoped CoreMemoryCandidate probation. It is still not PermaCore or TrueGolden without a later larger no-harm grind." : "Interpretation: rank is scoped. This operator is not Core memory unless a later Core probation grind passes the much higher qualified-activation and no-harm gates."}}</div>
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
    out = Path(args.out)
    payload = build_payload(e109, e110, e111, e112, e114, e116, e117, e118, e120, e121, e122)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(payload), encoding="utf-8")
    print(json.dumps({"out": str(out), "operator_count": len(payload["rows"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
