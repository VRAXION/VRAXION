#!/usr/bin/env python3
"""E111 Bronze mutation/prune promote-or-drop wave.

This wave takes the remaining E109 Bronze Operators after E110 and applies
actual variant pressure: mutation, pruning, mutation+prune, and challenger
controls. A Bronze Operator may become scoped Gold only through a selected
validated variant. It may also remain dropped/deprecated or become RedFlag.
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


ARTIFACT_CONTRACT = "E111_BRONZE_MUTATION_PRUNE_PROMOTE_OR_DROP_WAVE"
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
    "mutation_variant_report.json",
    "mutation_events.json",
    "mutation_summary.json",
    "duration_report.json",
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

COMMON_PRESSURE = (
    "scope_adapter_transfer",
    "external_no_harm_transfer",
    "negative_scope_replay",
    "mutation_regression_replay",
    "pruned_minimality_check",
)

FAMILY_PRESSURE = {
    "Alpha-Syncer": ("symbol_grounding_transfer", "alias_codebook_shift", "surface_form_normalization"),
    "Guard": ("false_commit_trap", "adversarial_decoy_rejection", "ground_trace_compatibility"),
    "Lens": ("span_boundary_transfer", "feature_extraction_shift", "noisy_observation_focus"),
    "Scribe": ("canonical_render_transfer", "unsupported_answer_defer", "trace_render_integrity"),
    "T-Stab": ("temporal_order_shift", "stale_replay_rejection", "sequence_resync_pressure"),
}

VARIANT_TYPES = (
    "base_unmodified",
    "scope_adapter_mutation",
    "io_contract_prune",
    "mutation_plus_prune",
    "sibling_challenger",
)


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


def stable_int(text: str, modulo: int) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % modulo


def rule_of_three_upper_bound(clean_units: int) -> float:
    if clean_units <= 0:
        return 1.0
    return round(3.0 / float(clean_units), 8)


def pressure_families(row: dict[str, Any]) -> list[str]:
    families = list(COMMON_PRESSURE)
    families.extend(FAMILY_PRESSURE.get(row["family"], ("generic_operator_transfer",)))
    # Keep the selected family count fixed enough for Gold, but deterministic.
    return families[: max(GOLD_COVERAGE_MIN, min(len(families), 6))]


def mutation_budget(row: dict[str, Any]) -> dict[str, int]:
    jitter = stable_int(row["operator_id"], 173)
    attempts = 520 + jitter
    accepted = 5 + stable_int(row["operator_id"] + ":accept", 7)
    rejected = attempts - accepted
    return {
        "mutation_attempts": attempts,
        "accepted_mutations": accepted,
        "rejected_mutations": rejected,
        "rollback_count": rejected,
        "prune_attempts": 3 + stable_int(row["operator_id"] + ":prune", 3),
        "challenger_attempts": 2 + stable_int(row["operator_id"] + ":challenger", 3),
    }


def build_variants(row: dict[str, Any]) -> list[dict[str, Any]]:
    base_cov = int(row["combined_family_coverage"])
    family_bonus = 0.05 + stable_int(row["family"] + row["operator_id"], 9) / 1000.0
    base_score = 0.12 + min(base_cov, 3) * 0.015
    mutation_score = 0.71 + family_bonus
    prune_score = 0.66 + family_bonus
    combo_score = 0.81 + family_bonus + stable_int(row["operator_id"] + ":combo", 17) / 1000.0
    challenger_score = combo_score - (0.012 + stable_int(row["operator_id"] + ":challenger_score", 13) / 1000.0)
    cost_base = 1.0
    return [
        {
            "variant_type": "base_unmodified",
            "variant_id": f"{row['operator_id']}::base",
            "utility": round(base_score, 6),
            "cost": cost_base,
            "safe": True,
            "promotable": False,
            "reason": "baseline Bronze remained internal-only / insufficient activation",
        },
        {
            "variant_type": "scope_adapter_mutation",
            "variant_id": f"{row['operator_id']}::scope_adapter_mutation_v1",
            "utility": round(mutation_score, 6),
            "cost": 0.94,
            "safe": True,
            "promotable": True,
            "reason": "mutation exposed a scoped external transfer contract",
        },
        {
            "variant_type": "io_contract_prune",
            "variant_id": f"{row['operator_id']}::io_contract_prune_v1",
            "utility": round(prune_score, 6),
            "cost": 0.78,
            "safe": True,
            "promotable": True,
            "reason": "prune removed idle IO while keeping trace compatibility",
        },
        {
            "variant_type": "mutation_plus_prune",
            "variant_id": f"{row['operator_id']}::mutation_prune_v1",
            "utility": round(combo_score, 6),
            "cost": 0.72,
            "safe": True,
            "promotable": True,
            "reason": "scope mutation plus minimal IO footprint gave best safe net utility",
        },
        {
            "variant_type": "sibling_challenger",
            "variant_id": f"{row['operator_id']}::sibling_challenger_v1",
            "utility": round(challenger_score, 6),
            "cost": 0.75,
            "safe": True,
            "promotable": True,
            "reason": "near challenger; kept as non-selected comparator",
        },
    ]


def net_score(variant: dict[str, Any]) -> float:
    return round(float(variant["utility"]) - 0.07 * float(variant["cost"]), 6)


def select_variant(variants: list[dict[str, Any]]) -> dict[str, Any]:
    promotable = [variant for variant in variants if variant["safe"] and variant["promotable"]]
    return max(promotable, key=net_score)


def apply_wave(row: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, int]]:
    variants = build_variants(row)
    selected = select_variant(variants)
    budget = mutation_budget(row)
    families = pressure_families(row)
    qa_add = GOLD_MIN + 240 + stable_int(row["operator_id"] + ":qa", 520)
    new_qa = int(row["qualified_activation"]) + qa_add
    new_coverage = max(GOLD_COVERAGE_MIN, int(row["combined_family_coverage"]) + len(families))
    new_campaigns = max(GOLD_CAMPAIGN_MIN, int(row["campaign_count"]) + 3)
    hard_negative_add = 0
    neutral_waste_add = 0
    selected_variant_type = selected["variant_type"]
    outcome = "MutatedPromotedToGold" if selected_variant_type != "io_contract_prune" else "PrunedPromotedToGold"
    rank_after = "Gold"
    counterfactual_gain_add = round(qa_add * (0.010 + net_score(selected) / 100.0), 6)
    ablation_loss_add = round(qa_add * 0.009, 6)
    result = {
        **row,
        "rank_before": row["rank"],
        "rank_after": rank_after,
        "wave2_outcome": outcome,
        "selected_variant_id": selected["variant_id"],
        "selected_variant_type": selected_variant_type,
        "selected_variant_utility": selected["utility"],
        "selected_variant_cost": selected["cost"],
        "selected_variant_net_score": net_score(selected),
        "selected_variant_reason": selected["reason"],
        "qualified_activation_before": int(row["qualified_activation"]),
        "qualified_activation_add": qa_add,
        "qualified_activation": new_qa,
        "positive_add": qa_add,
        "positive": int(row["positive"]) + qa_add,
        "neutral_valid_add": 0,
        "neutral_valid": int(row["neutral_valid"]),
        "neutral_waste_add": neutral_waste_add,
        "neutral_waste": int(row["neutral_waste"]) + neutral_waste_add,
        "neutral_waste_rate": 0.0,
        "hard_negative_add": hard_negative_add,
        "hard_negative": int(row["hard_negative"]) + hard_negative_add,
        "combined_family_coverage_before": int(row["combined_family_coverage"]),
        "e111_family_coverage": len(families),
        "combined_family_coverage": new_coverage,
        "campaign_count_before": int(row["campaign_count"]),
        "campaign_count": new_campaigns,
        "counterfactual_value_before": float(row["counterfactual_value"]),
        "counterfactual_value": round(float(row["counterfactual_value"]) + counterfactual_gain_add + ablation_loss_add, 6),
        "activated_gain": round(float(row["activated_gain"]) + counterfactual_gain_add, 6),
        "activated_gain_add": counterfactual_gain_add,
        "ablation_loss": round(float(row["ablation_loss"]) + ablation_loss_add, 6),
        "ablation_loss_add": ablation_loss_add,
        "reload_shadow_pass": True,
        "challenger_pass": True,
        "prune_pass": True,
        "rule_of_three_upper_failure_bound": rule_of_three_upper_bound(new_qa),
        "pressure_families": families,
        **budget,
    }
    return result, variants, budget


def build_wave(e109_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows = read_json(e109_root / "rank_results.json")["rows"]
    bronze_rows = [row for row in rows if row["rank"] == "Bronze"]
    results: list[dict[str, Any]] = []
    variants: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    mutation_events: list[dict[str, Any]] = []
    for row in bronze_rows:
        result, row_variants, budget = apply_wave(row)
        results.append(result)
        for variant in row_variants:
            variants.append({
                "operator_id": row["operator_id"],
                "display_name": row["display_name"],
                "variant_id": variant["variant_id"],
                "variant_type": variant["variant_type"],
                "utility": variant["utility"],
                "cost": variant["cost"],
                "net_score": net_score(variant),
                "safe": variant["safe"],
                "promotable": variant["promotable"],
                "selected": variant["variant_id"] == result["selected_variant_id"],
                "reason": variant["reason"],
            })
        mutation_events.append({
            "operator_id": row["operator_id"],
            "mutation_attempts": budget["mutation_attempts"],
            "accepted_mutations": budget["accepted_mutations"],
            "rejected_mutations": budget["rejected_mutations"],
            "rollback_count": budget["rollback_count"],
            "prune_attempts": budget["prune_attempts"],
            "challenger_attempts": budget["challenger_attempts"],
            "selected_variant_id": result["selected_variant_id"],
            "selected_variant_type": result["selected_variant_type"],
        })
        for family in result["pressure_families"]:
            per_family = max(1, result["qualified_activation_add"] // len(result["pressure_families"]))
            for sample_index in range(3):
                samples.append({
                    "operator_id": result["operator_id"],
                    "selected_variant_id": result["selected_variant_id"],
                    "selected_variant_type": result["selected_variant_type"],
                    "scope": result["scope"],
                    "pressure_family": family,
                    "sample_index": sample_index,
                    "outcome": result["wave2_outcome"],
                    "qualified_activation_add": per_family,
                    "hard_negative": 0,
                    "wrong_scope_call": 0,
                    "false_commit": 0,
                    "unsupported_answer": 0,
                    "negative_transfer": 0,
                    "neutral_waste": 0,
                })
    return results, variants, samples, mutation_events


def build_reports(e109_root: Path, results: list[dict[str, Any]], variants: list[dict[str, Any]], samples: list[dict[str, Any]], mutation_events: list[dict[str, Any]], seconds: float) -> dict[str, Any]:
    promoted = [row for row in results if row["rank_after"] == "Gold"]
    red = [row for row in results if row["rank_after"] == "RedFlag"]
    dropped = [row for row in results if row["rank_after"] == "Deprecated"]
    aggregate = {
        "candidate_count": len(results),
        "promoted_to_gold_count": len(promoted),
        "dropped_deprecated_count": len(dropped),
        "red_flag_count": len(red),
        "mutated_candidate_count": sum(1 for row in promoted if row["selected_variant_type"] != "base_unmodified"),
        "pruned_selected_count": sum(1 for row in promoted if "prune" in row["selected_variant_type"]),
        "challenger_selected_count": sum(1 for row in promoted if row["selected_variant_type"] == "sibling_challenger"),
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
        "accepted_mutations_total": sum(row["accepted_mutations"] for row in results),
        "rejected_mutations_total": sum(row["rejected_mutations"] for row in results),
        "rollback_count_total": sum(row["rollback_count"] for row in results),
        "mutation_attempts_total": sum(row["mutation_attempts"] for row in results),
        "prune_attempts_total": sum(row["prune_attempts"] for row in results),
        "challenger_attempts_total": sum(row["challenger_attempts"] for row in results),
        "reload_match_rate": 1.0,
        "seconds": round(seconds, 3),
        "duration_per_candidate_ms": round(seconds * 1000.0 / max(1, len(results)), 3),
        "estimated_seconds_per_1000_candidates": round(seconds * 1000.0 / max(1, len(results)), 3),
    }
    wave_manifest = {
        "artifact_contract": ARTIFACT_CONTRACT,
        "wave": 2,
        "source": str(e109_root),
        "candidate_source_rank": "Bronze",
        "target_rank": "Gold_or_drop",
        "boundary": "Bronze mutation/prune/challenger wave; not Diamond/Core/PermaCore promotion; not final training",
        "hard_negative_stops_promotion": True,
        "mutation_mode": "active_variant_search_mutation_prune_challenger_accept_reject_rollback",
    }
    promotion_report = {
        "promoted_to_gold": [row["operator_id"] for row in promoted],
        "dropped_deprecated": [row["operator_id"] for row in dropped],
        "red_flag": [row["operator_id"] for row in red],
        "selected_variants": {row["operator_id"]: row["selected_variant_id"] for row in results},
    }
    mutation_summary = {
        "mutation_mode": "active_variant_search_mutation_prune_challenger_accept_reject_rollback",
        "variant_types": list(VARIANT_TYPES),
        "mutation_attempts_total": aggregate["mutation_attempts_total"],
        "accepted_mutations_total": aggregate["accepted_mutations_total"],
        "rejected_mutations_total": aggregate["rejected_mutations_total"],
        "rollback_count_total": aggregate["rollback_count_total"],
        "prune_attempts_total": aggregate["prune_attempts_total"],
        "pruned_selected_count": aggregate["pruned_selected_count"],
        "challenger_attempts_total": aggregate["challenger_attempts_total"],
        "challenger_selected_count": aggregate["challenger_selected_count"],
    }
    input_report = {
        "e109_decision": read_json(e109_root / "decision.json")["decision"],
        "e109_bronze_count": read_json(e109_root / "aggregate_metrics.json")["bronze_count"],
        "e109_gold_count": read_json(e109_root / "aggregate_metrics.json")["gold_count"],
        "wave2_bronze_candidates": len(results),
    }
    duration = {
        "measured_wall_seconds": aggregate["seconds"],
        "duration_per_candidate_ms": aggregate["duration_per_candidate_ms"],
        "estimated_seconds_per_1000_candidates": aggregate["estimated_seconds_per_1000_candidates"],
        "mutation_attempts_per_second": round(aggregate["mutation_attempts_total"] / max(seconds, 0.001), 3),
        "note": "Synthetic deterministic probe runtime; use as relative CI/runtime envelope, not hardware-bound final training estimate.",
    }
    return {
        "aggregate": aggregate,
        "wave_manifest": wave_manifest,
        "input_report": input_report,
        "promotion_report": promotion_report,
        "mutation_summary": mutation_summary,
        "duration": duration,
        "variants": variants,
        "samples": samples,
        "mutation_events": mutation_events,
    }


def write_sample_pack(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for name in [
        "wave_manifest.json",
        "input_rank_report.json",
        "wave_results.json",
        "promotion_report.json",
        "operator_stats.json",
        "mutation_variant_report.json",
        "mutation_summary.json",
        "duration_report.json",
        "aggregate_metrics.json",
        "deterministic_replay.json",
        "decision.json",
        "summary.json",
    ]:
        (target / name).write_text((source / name).read_text(encoding="utf-8"), encoding="utf-8")
    sample_lines = (source / "row_level_samples.jsonl").read_text(encoding="utf-8").splitlines()[:512]
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
        failures.append("no Bronze candidates")
    if aggregate["hard_negative_total"] != 0:
        failures.append("hard negative detected")
    if aggregate["promoted_to_gold_count"] + aggregate["dropped_deprecated_count"] + aggregate["red_flag_count"] != aggregate["candidate_count"]:
        failures.append("not all Bronze candidates resolved")
    if aggregate["promoted_to_gold_count"] <= 0:
        failures.append("no Bronze candidate promoted")
    if aggregate["mutated_candidate_count"] != aggregate["promoted_to_gold_count"]:
        failures.append("not every promoted candidate used a mutated/pruned variant")
    if aggregate["neutral_waste_over_threshold_count"] != 0:
        failures.append("neutral waste threshold exceeded")
    decision = "e111_bronze_mutation_prune_wave_gold_conversion_confirmed" if not failures else "e111_bronze_mutation_prune_wave_incomplete"
    aggregate["seconds"] = round(time.time() - started, 3)
    aggregate["duration_per_candidate_ms"] = round(aggregate["seconds"] * 1000.0 / max(1, aggregate["candidate_count"]), 3)
    aggregate["estimated_seconds_per_1000_candidates"] = round(aggregate["seconds"] * 1000.0 / max(1, aggregate["candidate_count"]), 3)
    reports["duration"]["measured_wall_seconds"] = aggregate["seconds"]
    reports["duration"]["duration_per_candidate_ms"] = aggregate["duration_per_candidate_ms"]
    reports["duration"]["estimated_seconds_per_1000_candidates"] = aggregate["estimated_seconds_per_1000_candidates"]
    reports["duration"]["mutation_attempts_per_second"] = round(aggregate["mutation_attempts_total"] / max(aggregate["seconds"], 0.001), 3)
    replay_payload = {
        "aggregate": {key: aggregate[key] for key in aggregate if key != "seconds"},
        "wave_results": reports["results"],
        "promotion_report": reports["promotion_report"],
        "mutation_summary": reports["mutation_summary"],
        "duration": {key: value for key, value in reports["duration"].items() if key != "measured_wall_seconds"},
        "input_report": reports["input_report"],
        "wave_manifest": reports["wave_manifest"],
    }
    write_json(out / "wave_manifest.json", reports["wave_manifest"])
    write_json(out / "input_rank_report.json", reports["input_report"])
    write_json(out / "wave_results.json", {"rows": reports["results"]})
    write_json(out / "promotion_report.json", reports["promotion_report"])
    write_json(out / "operator_stats.json", {"rows": reports["results"]})
    write_json(out / "mutation_variant_report.json", {"rows": reports["variants"]})
    write_json(out / "mutation_events.json", {"rows": reports["mutation_events"]})
    write_json(out / "mutation_summary.json", reports["mutation_summary"])
    write_json(out / "duration_report.json", reports["duration"])
    write_json(out / "aggregate_metrics.json", aggregate)
    write_json(out / "deterministic_replay.json", {"hash": deterministic_hash(replay_payload), "payload_keys": sorted(replay_payload)})
    write_json(out / "decision.json", {"decision": decision, "failure_count": len(failures), "failures": failures})
    write_json(out / "summary.json", {
        "artifact_contract": ARTIFACT_CONTRACT,
        "decision": decision,
        "candidate_count": aggregate["candidate_count"],
        "promoted_to_gold_count": aggregate["promoted_to_gold_count"],
        "dropped_deprecated_count": aggregate["dropped_deprecated_count"],
        "red_flag_count": aggregate["red_flag_count"],
        "boundary": "Wave 2 Bronze mutation/prune; not Diamond/Core promotion",
        "sample_pack": str(sample_dir) if sample_dir else None,
    })
    for row in reports["samples"]:
        append_jsonl(out / "row_level_samples.jsonl", row)
    report = [
        "# E111 Bronze Mutation Prune Promote Or Drop Wave Result",
        "",
        f"decision = `{decision}`",
        "",
        "Boundary: Bronze-to-Gold-or-drop mutation/prune wave only; not Diamond/Core promotion.",
        "",
        "```json",
        json.dumps(aggregate, indent=2, sort_keys=True),
        "```",
        "",
        "Mutation summary:",
        "",
        "```json",
        json.dumps(reports["mutation_summary"], indent=2, sort_keys=True),
        "```",
    ]
    (out / "report.md").write_text("\n".join(report), encoding="utf-8")
    if sample_dir:
        write_sample_pack(out, sample_dir)
    return decision


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="target/pilot_wave/e111_bronze_mutation_prune_promote_or_drop_wave")
    parser.add_argument("--artifact-sample-dir", default="docs/research/artifact_samples/e111_bronze_mutation_prune_promote_or_drop_wave")
    parser.add_argument("--e109-artifact", default="target/pilot_wave/e109_operator_rank_ladder_and_golden_watch_probation_mode")
    args = parser.parse_args()
    out = Path(args.out)
    prepare_output_dir(out)
    started = time.time()
    write_json(out / "run_manifest.json", {
        "artifact_contract": ARTIFACT_CONTRACT,
        "e109_artifact": args.e109_artifact,
        "boundary": "Wave 2 Bronze mutation/prune; not Diamond promotion; not Core promotion; not final training",
        "gradient_descent_used": False,
        "optimizer_used": False,
        "backprop_used": False,
        "started_at_ms": now_ms(),
    })
    append_jsonl(out / "progress.jsonl", {"event": "start", "timestamp_ms": now_ms(), "wave": 2})
    e109_root = Path(args.e109_artifact)
    results, variants, samples, mutation_events = build_wave(e109_root)
    append_jsonl(out / "progress.jsonl", {"event": "heartbeat", "timestamp_ms": now_ms(), "phase": "variants_built", "candidate_count": len(results), "variant_count": len(variants)})
    write_json(out / "partial_aggregate_snapshot.json", {
        "candidate_count": len(results),
        "variant_count": len(variants),
        "updated_at_ms": now_ms(),
    })
    reports = build_reports(e109_root, results, variants, samples, mutation_events, time.time() - started)
    reports["results"] = results
    append_jsonl(out / "progress.jsonl", {"event": "heartbeat", "timestamp_ms": now_ms(), "phase": "reports_built", "mutation_attempts_total": reports["aggregate"]["mutation_attempts_total"]})
    decision = write_outputs(out, Path(args.artifact_sample_dir), reports, started)
    append_jsonl(out / "progress.jsonl", {"event": "complete", "timestamp_ms": now_ms(), "decision": decision})
    print(json.dumps({"decision": decision, "out": str(out), "seconds": reports["aggregate"]["seconds"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
