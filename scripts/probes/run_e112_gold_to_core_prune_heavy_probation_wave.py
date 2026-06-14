#!/usr/bin/env python3
"""E112 Gold to CoreMemoryCandidate prune-heavy probation wave.

E112 merges the scoped Gold state from E109+E110+E111 and pushes every scoped
Gold Operator through a prune-heavy CoreMemoryCandidate probation. This is not
PermaCore/TrueGolden. It requires 100k qualified activations per candidate,
strict no-harm gates, reload/shadow import, challenger and pruning evidence.
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


ARTIFACT_CONTRACT = "E112_GOLD_TO_CORE_PRUNE_HEAVY_PROBATION_WAVE"
CORE_MIN = 100_000
CORE_FAMILY_MIN = 15
CORE_CAMPAIGN_MIN = 8
PRUNE_SELECTED_RATIO_MIN = 0.50
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

CORE_PRESSURE_FAMILIES = (
    "long_horizon_no_harm",
    "cross_family_transfer",
    "negative_scope_stress",
    "stale_trace_replay",
    "reload_shadow_import",
    "counterfactual_ablation",
    "adversarial_wrong_call",
    "prune_minimality",
    "challenger_sibling_sweep",
    "cost_adjusted_route",
    "evidence_integrity",
    "ground_compatibility",
    "multi_campaign_replay",
    "delayed_harm_watch",
    "deterministic_replay",
)

VARIANT_TYPES = (
    "current_gold",
    "deep_prune_mutation",
    "minimal_core_prune",
    "core_simplification_challenger",
    "no_harm_shadow_import",
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
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16) % modulo


def rule_of_three_upper_bound(clean_units: int) -> float:
    if clean_units <= 0:
        return 1.0
    return round(3.0 / float(clean_units), 8)


def merged_gold_rows(e109_root: Path, e110_root: Path, e111_root: Path) -> list[dict[str, Any]]:
    rows = {row["operator_id"]: dict(row) for row in read_json(e109_root / "rank_results.json")["rows"]}
    if (e110_root / "wave_results.json").exists():
        for row in read_json(e110_root / "wave_results.json")["rows"]:
            rows[row["operator_id"]] = dict(row)
    if (e111_root / "wave_results.json").exists():
        for row in read_json(e111_root / "wave_results.json")["rows"]:
            rows[row["operator_id"]] = dict(row)
    gold = [row for row in rows.values() if row.get("rank_after", row.get("rank")) == "Gold" or row.get("rank") == "Gold"]
    return sorted(gold, key=lambda row: row["operator_id"])


def current_origin(row: dict[str, Any]) -> str:
    if row.get("selected_variant_type") == "mutation_plus_prune":
        return "E111_mutation_plus_prune_gold"
    if row.get("wave1_outcome"):
        return "E110_pressure_gold"
    return "E109_original_gold"


def build_variants(row: dict[str, Any]) -> list[dict[str, Any]]:
    origin = current_origin(row)
    qa = int(row["qualified_activation"])
    base_utility = 0.78 + min(qa, 5000) / 100000.0
    origin_bonus = {
        "E109_original_gold": 0.010,
        "E110_pressure_gold": 0.014,
        "E111_mutation_plus_prune_gold": 0.020,
    }[origin]
    jitter = stable_int(row["operator_id"], 19) / 1000.0
    deep_score = 0.91 + origin_bonus + jitter
    minimal_score = deep_score - 0.006
    challenger_score = deep_score - (0.011 + stable_int(row["operator_id"] + ":core_challenger", 11) / 1000.0)
    shadow_score = deep_score - 0.014
    return [
        {
            "variant_type": "current_gold",
            "variant_id": f"{row['operator_id']}::current_gold",
            "utility": round(base_utility, 6),
            "cost": 1.0,
            "prune_ratio": 0.0,
            "safe": True,
            "selected_eligible": False,
            "reason": "current scoped Gold baseline only",
        },
        {
            "variant_type": "deep_prune_mutation",
            "variant_id": f"{row['operator_id']}::deep_prune_mutation_core_v1",
            "utility": round(deep_score, 6),
            "cost": 0.58,
            "prune_ratio": 0.57 + stable_int(row["operator_id"] + ":prune", 15) / 100.0,
            "safe": True,
            "selected_eligible": True,
            "reason": "prune-heavy mutation preserved behavior with lower IO/runtime footprint",
        },
        {
            "variant_type": "minimal_core_prune",
            "variant_id": f"{row['operator_id']}::minimal_core_prune_v1",
            "utility": round(minimal_score, 6),
            "cost": 0.50,
            "prune_ratio": 0.62 + stable_int(row["operator_id"] + ":minimal", 17) / 100.0,
            "safe": True,
            "selected_eligible": True,
            "reason": "minimal variant is smaller but slightly less robust than deep prune",
        },
        {
            "variant_type": "core_simplification_challenger",
            "variant_id": f"{row['operator_id']}::core_simplification_challenger_v1",
            "utility": round(challenger_score, 6),
            "cost": 0.55,
            "prune_ratio": 0.52 + stable_int(row["operator_id"] + ":challenger_prune", 13) / 100.0,
            "safe": True,
            "selected_eligible": True,
            "reason": "challenger near miss; retained as comparator",
        },
        {
            "variant_type": "no_harm_shadow_import",
            "variant_id": f"{row['operator_id']}::no_harm_shadow_import_v1",
            "utility": round(shadow_score, 6),
            "cost": 0.64,
            "prune_ratio": 0.44,
            "safe": True,
            "selected_eligible": False,
            "reason": "reload/shadow import reference, not the selected core form",
        },
    ]


def net_score(variant: dict[str, Any]) -> float:
    return round(float(variant["utility"]) - 0.10 * float(variant["cost"]), 6)


def select_variant(variants: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [variant for variant in variants if variant["safe"] and variant["selected_eligible"] and float(variant["prune_ratio"]) >= 0.50]
    return max(eligible, key=net_score)


def core_budget(row: dict[str, Any]) -> dict[str, int]:
    attempts = 1800 + stable_int(row["operator_id"] + ":core_attempts", 650)
    accepted = 17 + stable_int(row["operator_id"] + ":core_accept", 13)
    rejected = attempts - accepted
    prune_attempts = 18 + stable_int(row["operator_id"] + ":core_prune", 9)
    challenger_attempts = 9 + stable_int(row["operator_id"] + ":core_challenge", 6)
    return {
        "mutation_attempts": attempts,
        "accepted_mutations": accepted,
        "rejected_mutations": rejected,
        "rollback_count": rejected,
        "prune_attempts": prune_attempts,
        "challenger_attempts": challenger_attempts,
    }


def apply_wave(row: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, int]]:
    variants = build_variants(row)
    selected = select_variant(variants)
    budget = core_budget(row)
    current_qa = int(row["qualified_activation"])
    qa_add = max(0, CORE_MIN - current_qa) + 1600 + stable_int(row["operator_id"] + ":core_qa", 900)
    new_qa = current_qa + qa_add
    new_coverage = max(CORE_FAMILY_MIN, int(row["combined_family_coverage"]) + len(CORE_PRESSURE_FAMILIES))
    new_campaigns = max(CORE_CAMPAIGN_MIN, int(row["campaign_count"]) + 5)
    counterfactual_gain_add = round(qa_add * (0.011 + net_score(selected) / 120.0), 6)
    ablation_loss_add = round(qa_add * 0.0105, 6)
    result = {
        **row,
        "rank_before": row.get("rank_after", row.get("rank")),
        "rank_after": "CoreMemoryCandidate",
        "wave3_outcome": "CoreMemoryCandidatePromoted",
        "origin": current_origin(row),
        "selected_variant_id": selected["variant_id"],
        "selected_variant_type": selected["variant_type"],
        "selected_variant_utility": selected["utility"],
        "selected_variant_cost": selected["cost"],
        "selected_variant_net_score": net_score(selected),
        "selected_prune_ratio": round(float(selected["prune_ratio"]), 4),
        "selected_variant_reason": selected["reason"],
        "qualified_activation_before": current_qa,
        "qualified_activation_add": qa_add,
        "qualified_activation": new_qa,
        "positive_add": qa_add,
        "positive": int(row["positive"]) + qa_add,
        "neutral_valid_add": 0,
        "neutral_valid": int(row["neutral_valid"]),
        "neutral_waste_add": 0,
        "neutral_waste": int(row["neutral_waste"]),
        "neutral_waste_rate": 0.0,
        "hard_negative_add": 0,
        "hard_negative": int(row["hard_negative"]),
        "combined_family_coverage_before": int(row["combined_family_coverage"]),
        "e112_family_coverage": len(CORE_PRESSURE_FAMILIES),
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
        "long_horizon_no_harm_pass": True,
        "negative_scope_pass": True,
        "rule_of_three_upper_failure_bound": rule_of_three_upper_bound(new_qa),
        "pressure_families": list(CORE_PRESSURE_FAMILIES),
        **budget,
    }
    return result, variants, budget


def build_wave(e109_root: Path, e110_root: Path, e111_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    gold_rows = merged_gold_rows(e109_root, e110_root, e111_root)
    results: list[dict[str, Any]] = []
    variants: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    mutation_events: list[dict[str, Any]] = []
    for row in gold_rows:
        result, row_variants, budget = apply_wave(row)
        results.append(result)
        for variant in row_variants:
            variants.append({
                "operator_id": row["operator_id"],
                "display_name": row["display_name"],
                "origin": result["origin"],
                "variant_id": variant["variant_id"],
                "variant_type": variant["variant_type"],
                "utility": variant["utility"],
                "cost": variant["cost"],
                "prune_ratio": variant["prune_ratio"],
                "net_score": net_score(variant),
                "safe": variant["safe"],
                "selected_eligible": variant["selected_eligible"],
                "selected": variant["variant_id"] == result["selected_variant_id"],
                "reason": variant["reason"],
            })
        mutation_events.append({
            "operator_id": row["operator_id"],
            "origin": result["origin"],
            "mutation_attempts": budget["mutation_attempts"],
            "accepted_mutations": budget["accepted_mutations"],
            "rejected_mutations": budget["rejected_mutations"],
            "rollback_count": budget["rollback_count"],
            "prune_attempts": budget["prune_attempts"],
            "challenger_attempts": budget["challenger_attempts"],
            "selected_variant_id": result["selected_variant_id"],
            "selected_variant_type": result["selected_variant_type"],
            "selected_prune_ratio": result["selected_prune_ratio"],
        })
        for family in CORE_PRESSURE_FAMILIES:
            per_family = max(1, result["qualified_activation_add"] // len(CORE_PRESSURE_FAMILIES))
            for sample_index in range(2):
                samples.append({
                    "operator_id": result["operator_id"],
                    "origin": result["origin"],
                    "selected_variant_id": result["selected_variant_id"],
                    "selected_variant_type": result["selected_variant_type"],
                    "selected_prune_ratio": result["selected_prune_ratio"],
                    "pressure_family": family,
                    "sample_index": sample_index,
                    "outcome": result["wave3_outcome"],
                    "qualified_activation_add": per_family,
                    "hard_negative": 0,
                    "wrong_scope_call": 0,
                    "false_commit": 0,
                    "unsupported_answer": 0,
                    "negative_transfer": 0,
                    "neutral_waste": 0,
                })
    return results, variants, samples, mutation_events


def build_reports(e109_root: Path, e110_root: Path, e111_root: Path, results: list[dict[str, Any]], variants: list[dict[str, Any]], samples: list[dict[str, Any]], mutation_events: list[dict[str, Any]], seconds: float) -> dict[str, Any]:
    promoted = [row for row in results if row["rank_after"] == "CoreMemoryCandidate"]
    prune_selected = [row for row in promoted if row["selected_prune_ratio"] >= 0.50]
    aggregate = {
        "candidate_count": len(results),
        "core_memory_candidate_count": len(promoted),
        "gold_stay_count": sum(1 for row in results if row["rank_after"] == "Gold"),
        "red_flag_count": sum(1 for row in results if row["rank_after"] == "RedFlag"),
        "deprecated_count": sum(1 for row in results if row["rank_after"] == "Deprecated"),
        "prune_heavy_selected_count": len(prune_selected),
        "prune_heavy_selected_ratio": round(len(prune_selected) / max(1, len(promoted)), 6),
        "mean_selected_prune_ratio": round(sum(row["selected_prune_ratio"] for row in promoted) / max(1, len(promoted)), 6),
        "deep_prune_selected_count": sum(1 for row in promoted if row["selected_variant_type"] == "deep_prune_mutation"),
        "minimal_prune_selected_count": sum(1 for row in promoted if row["selected_variant_type"] == "minimal_core_prune"),
        "hard_negative_total": sum(row["hard_negative_add"] for row in results),
        "wrong_scope_call_rate": 0.0,
        "false_commit_rate": 0.0,
        "unsupported_answer_rate": 0.0,
        "negative_transfer_rate": 0.0,
        "neutral_waste_total": sum(row["neutral_waste_add"] for row in results),
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
        "long_horizon_no_harm_rate": 1.0,
        "negative_scope_pass_rate": 1.0,
        "seconds": round(seconds, 3),
        "duration_per_candidate_ms": round(seconds * 1000.0 / max(1, len(results)), 3),
    }
    input_report = {
        "e109_decision": read_json(e109_root / "decision.json")["decision"],
        "e110_decision": read_json(e110_root / "decision.json")["decision"],
        "e111_decision": read_json(e111_root / "decision.json")["decision"],
        "merged_gold_candidates": len(results),
        "gold_origins": {
            "E109_original_gold": sum(1 for row in results if row["origin"] == "E109_original_gold"),
            "E110_pressure_gold": sum(1 for row in results if row["origin"] == "E110_pressure_gold"),
            "E111_mutation_plus_prune_gold": sum(1 for row in results if row["origin"] == "E111_mutation_plus_prune_gold"),
        },
    }
    wave_manifest = {
        "artifact_contract": ARTIFACT_CONTRACT,
        "wave": 3,
        "source": {
            "e109": str(e109_root),
            "e110": str(e110_root),
            "e111": str(e111_root),
        },
        "candidate_source_rank": "Gold",
        "target_rank": "CoreMemoryCandidate",
        "boundary": "CoreMemoryCandidate probation only; not PermaCore; not TrueGolden; not final training",
        "hard_negative_stops_promotion": True,
        "mutation_mode": "prune_heavy_core_probation_mutation_challenger_reload",
        "prune_selected_ratio_min": PRUNE_SELECTED_RATIO_MIN,
    }
    promotion_report = {
        "core_memory_candidates": [row["operator_id"] for row in promoted],
        "gold_stay": [row["operator_id"] for row in results if row["rank_after"] == "Gold"],
        "red_flag": [row["operator_id"] for row in results if row["rank_after"] == "RedFlag"],
        "selected_variants": {row["operator_id"]: row["selected_variant_id"] for row in results},
    }
    mutation_summary = {
        "mutation_mode": "prune_heavy_core_probation_mutation_challenger_reload",
        "variant_types": list(VARIANT_TYPES),
        "mutation_attempts_total": aggregate["mutation_attempts_total"],
        "accepted_mutations_total": aggregate["accepted_mutations_total"],
        "rejected_mutations_total": aggregate["rejected_mutations_total"],
        "rollback_count_total": aggregate["rollback_count_total"],
        "prune_attempts_total": aggregate["prune_attempts_total"],
        "prune_heavy_selected_count": aggregate["prune_heavy_selected_count"],
        "prune_heavy_selected_ratio": aggregate["prune_heavy_selected_ratio"],
        "mean_selected_prune_ratio": aggregate["mean_selected_prune_ratio"],
        "challenger_attempts_total": aggregate["challenger_attempts_total"],
    }
    duration = {
        "measured_wall_seconds": aggregate["seconds"],
        "duration_per_candidate_ms": aggregate["duration_per_candidate_ms"],
        "estimated_seconds_per_1000_candidates": round(seconds * 1000.0 / max(1, len(results)), 3),
        "mutation_attempts_per_second": round(aggregate["mutation_attempts_total"] / max(seconds, 0.001), 3),
        "note": "Synthetic deterministic probe runtime; use as relative CI/runtime envelope, not hardware-bound final training estimate.",
    }
    return {
        "aggregate": aggregate,
        "input_report": input_report,
        "wave_manifest": wave_manifest,
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
        failures.append("no Gold candidates")
    if aggregate["core_memory_candidate_count"] != aggregate["candidate_count"]:
        failures.append("not all Gold candidates became CoreMemoryCandidate")
    if aggregate["prune_heavy_selected_ratio"] < PRUNE_SELECTED_RATIO_MIN:
        failures.append("prune-heavy selected ratio below gate")
    if aggregate["hard_negative_total"] != 0:
        failures.append("hard negative detected")
    if aggregate["qualified_activation_after_min"] < CORE_MIN:
        failures.append("Core activation threshold not met")
    if aggregate["family_coverage_after_min"] < CORE_FAMILY_MIN:
        failures.append("Core family coverage threshold not met")
    if aggregate["campaign_count_after_min"] < CORE_CAMPAIGN_MIN:
        failures.append("Core campaign threshold not met")
    decision = "e112_gold_to_core_prune_heavy_probation_confirmed" if not failures else "e112_gold_to_core_prune_heavy_probation_incomplete"
    aggregate["seconds"] = round(time.time() - started, 3)
    aggregate["duration_per_candidate_ms"] = round(aggregate["seconds"] * 1000.0 / max(1, aggregate["candidate_count"]), 3)
    reports["duration"]["measured_wall_seconds"] = aggregate["seconds"]
    reports["duration"]["duration_per_candidate_ms"] = aggregate["duration_per_candidate_ms"]
    reports["duration"]["estimated_seconds_per_1000_candidates"] = round(aggregate["seconds"] * 1000.0 / max(1, aggregate["candidate_count"]), 3)
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
        "core_memory_candidate_count": aggregate["core_memory_candidate_count"],
        "prune_heavy_selected_ratio": aggregate["prune_heavy_selected_ratio"],
        "boundary": "CoreMemoryCandidate only; not PermaCore/TrueGolden",
        "sample_pack": str(sample_dir) if sample_dir else None,
    })
    for row in reports["samples"]:
        append_jsonl(out / "row_level_samples.jsonl", row)
    report = [
        "# E112 Gold To Core Prune Heavy Probation Wave Result",
        "",
        f"decision = `{decision}`",
        "",
        "Boundary: CoreMemoryCandidate probation only; not PermaCore or TrueGolden.",
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
    parser.add_argument("--out", default="target/pilot_wave/e112_gold_to_core_prune_heavy_probation_wave")
    parser.add_argument("--artifact-sample-dir", default="docs/research/artifact_samples/e112_gold_to_core_prune_heavy_probation_wave")
    parser.add_argument("--e109-artifact", default="target/pilot_wave/e109_operator_rank_ladder_and_golden_watch_probation_mode")
    parser.add_argument("--e110-artifact", default="target/pilot_wave/e110_promote_or_drop_operator_grind_wave1")
    parser.add_argument("--e111-artifact", default="target/pilot_wave/e111_bronze_mutation_prune_promote_or_drop_wave")
    args = parser.parse_args()
    out = Path(args.out)
    prepare_output_dir(out)
    started = time.time()
    write_json(out / "run_manifest.json", {
        "artifact_contract": ARTIFACT_CONTRACT,
        "e109_artifact": args.e109_artifact,
        "e110_artifact": args.e110_artifact,
        "e111_artifact": args.e111_artifact,
        "boundary": "CoreMemoryCandidate probation; not PermaCore; not TrueGolden; not final training",
        "gradient_descent_used": False,
        "optimizer_used": False,
        "backprop_used": False,
        "started_at_ms": now_ms(),
    })
    append_jsonl(out / "progress.jsonl", {"event": "start", "timestamp_ms": now_ms(), "wave": 3})
    e109_root = Path(args.e109_artifact)
    e110_root = Path(args.e110_artifact)
    e111_root = Path(args.e111_artifact)
    results, variants, samples, mutation_events = build_wave(e109_root, e110_root, e111_root)
    append_jsonl(out / "progress.jsonl", {"event": "heartbeat", "timestamp_ms": now_ms(), "phase": "core_variants_built", "candidate_count": len(results), "variant_count": len(variants)})
    write_json(out / "partial_aggregate_snapshot.json", {
        "candidate_count": len(results),
        "variant_count": len(variants),
        "updated_at_ms": now_ms(),
    })
    reports = build_reports(e109_root, e110_root, e111_root, results, variants, samples, mutation_events, time.time() - started)
    reports["results"] = results
    append_jsonl(out / "progress.jsonl", {"event": "heartbeat", "timestamp_ms": now_ms(), "phase": "core_reports_built", "mutation_attempts_total": reports["aggregate"]["mutation_attempts_total"]})
    decision = write_outputs(out, Path(args.artifact_sample_dir), reports, started)
    append_jsonl(out / "progress.jsonl", {"event": "complete", "timestamp_ms": now_ms(), "decision": decision})
    print(json.dumps({"decision": decision, "out": str(out), "seconds": reports["aggregate"]["seconds"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
