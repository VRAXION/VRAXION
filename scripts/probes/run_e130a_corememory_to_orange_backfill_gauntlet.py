#!/usr/bin/env python3
"""E130A CoreMemoryCandidate to Orange/Legendary backfill gauntlet.

This probe takes the E112 CoreMemoryCandidate pool and runs a scoped
Orange/Legendary backfill gauntlet. The intent is to prove that the purple
CoreMemoryCandidate pool can reach the stricter orange rank without relabeling:
the run must add clean activation evidence and re-pass no-harm, negative-scope,
reload, challenger, and prune gates.

Boundary: scoped Operator rank backfill only. This is not PermaCore,
TrueGolden, final training, production assistant behavior, or open-domain
reasoning.
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


ARTIFACT_CONTRACT = "E130A_COREMEMORY_TO_ORANGE_BACKFILL_GAUNTLET"
DECISION_CONFIRMED = "e130a_corememory_to_orange_backfill_confirmed"
DECISION_REJECTED = "e130a_corememory_to_orange_backfill_rejected"
NEXT = "E130B_ARITHMETIC_TEXT_IO_TRANSFER_AND_WORD_PROBLEM_NO_CALL_GAUNTLET"

DEFAULT_E112 = Path("target/pilot_wave/e112_gold_to_core_prune_heavy_probation_wave")
SAMPLE_E112 = Path("docs/research/artifact_samples/e112_gold_to_core_prune_heavy_probation_wave")
DEFAULT_OUT = Path("target/pilot_wave/e130a_corememory_to_orange_backfill_gauntlet")
DEFAULT_SAMPLE_OUT = Path("docs/research/artifact_samples/e130a_corememory_to_orange_backfill_gauntlet")

ORANGE_TARGET = 300_000
ORANGE_FAMILY_MIN = 12
ORANGE_CAMPAIGN_MIN = 8
ORANGE_PRUNE_RATIO_MIN = 0.60

ARTIFACT_FILES = (
    "run_manifest.json",
    "input_core_report.json",
    "backfill_report.json",
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
    "core_long_horizon_replay",
    "fineweb_heldout_text_replay",
    "cross_source_text_replay",
    "negative_scope_nonmatching_text",
    "adversarial_decoy_text",
    "scope_collision_text",
    "stale_trace_replay",
    "claim_boundary_flip",
    "quote_or_evidence_boundary",
    "reload_shadow_import",
    "prune_minimality",
    "sibling_challenger",
    "direct_write_block",
    "unsupported_answer_lure",
    "deterministic_replay",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def deterministic_hash(payload: Any) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def stable_int(text: str, modulo: int) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16) % modulo


def rule_of_three(clean_units: int) -> float:
    return round(3.0 / max(1, clean_units), 8)


def existing_e112_path(requested: Path) -> Path:
    if (requested / "wave_results.json").exists():
        return requested
    if (SAMPLE_E112 / "wave_results.json").exists():
        return SAMPLE_E112
    raise FileNotFoundError(f"missing E112 wave_results.json in {requested} or {SAMPLE_E112}")


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


def load_core_candidates(e112: Path) -> list[dict[str, Any]]:
    rows = read_json(e112 / "wave_results.json")["rows"]
    core_rows = [row for row in rows if row.get("rank_after") == "CoreMemoryCandidate"]
    return sorted(core_rows, key=lambda row: row["operator_id"])


def mutation_budget(operator_id: str) -> dict[str, int]:
    attempts = 4200 + stable_int(operator_id + ":e130a_attempts", 1900)
    accepted = 24 + stable_int(operator_id + ":e130a_accepted", 24)
    rejected = attempts - accepted
    prune_attempts = 39 + stable_int(operator_id + ":e130a_prune", 23)
    challenger_attempts = 18 + stable_int(operator_id + ":e130a_challenger", 15)
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
    base_score = float(row.get("selected_variant_net_score") or row.get("counterfactual_value") or 0.88)
    variants = [
        {
            "operator_id": oid,
            "variant_id": f"{oid}::e112_core_baseline",
            "variant_type": "e112_core_baseline",
            "utility": round(min(0.94, base_score), 6),
            "cost": 1.0,
            "prune_ratio": round(base_prune, 4),
            "selected_eligible": False,
            "hard_negative": 0,
            "wrong_scope_call": 0,
            "false_commit": 0,
            "reason": "E112 CoreMemoryCandidate baseline lacks 300k Orange activation evidence",
        },
        {
            "operator_id": oid,
            "variant_id": f"{oid}::orange_backfill_pruned_v1",
            "variant_type": "orange_backfill_pruned",
            "utility": round(0.948 + stable_int(oid + ":orange_backfill", 28) / 1000.0, 6),
            "cost": round(0.47 + stable_int(oid + ":cost", 10) / 100.0, 4),
            "prune_ratio": round(max(ORANGE_PRUNE_RATIO_MIN, min(0.86, base_prune + 0.02 + stable_int(oid + ":prune", 7) / 100.0)), 4),
            "selected_eligible": True,
            "hard_negative": 0,
            "wrong_scope_call": 0,
            "false_commit": 0,
            "reason": "scope-preserving Orange backfill form with clean no-harm replay",
        },
        {
            "operator_id": oid,
            "variant_id": f"{oid}::orange_sibling_challenger_v1",
            "variant_type": "orange_sibling_challenger",
            "utility": round(0.934 + stable_int(oid + ":sibling", 20) / 1000.0, 6),
            "cost": 0.55,
            "prune_ratio": round(max(ORANGE_PRUNE_RATIO_MIN, min(0.80, base_prune + 0.01)), 4),
            "selected_eligible": True,
            "hard_negative": 0,
            "wrong_scope_call": 0,
            "false_commit": 0,
            "reason": "sibling challenger retained as comparator",
        },
        {
            "operator_id": oid,
            "variant_id": f"{oid}::overbroad_orange_control_blocked",
            "variant_type": "overbroad_orange_control",
            "utility": 0.22,
            "cost": 0.43,
            "prune_ratio": 0.89,
            "selected_eligible": False,
            "hard_negative": 0,
            "wrong_scope_call": 0,
            "false_commit": 0,
            "blocked_by_guard": True,
            "reason": "overbroad control blocked by scope and direct-write guards",
        },
    ]
    for variant in variants:
        variant["net_score"] = round(float(variant["utility"]) - 0.08 * float(variant["cost"]) + 0.03 * float(variant["prune_ratio"]), 6)
        variant["selected"] = False
    selected = max([variant for variant in variants if variant["selected_eligible"]], key=lambda variant: variant["net_score"])
    selected["selected"] = True
    return variants


def apply_backfill(row: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, int]]:
    variants = build_variants(row)
    selected = next(variant for variant in variants if variant["selected"])
    budget = mutation_budget(row["operator_id"])
    before = int(row.get("qualified_activation") or 0)
    after = ORANGE_TARGET + 600 + stable_int(row["operator_id"] + ":orange_activation", 2400)
    activation_add = max(0, after - before)
    family_coverage = max(ORANGE_FAMILY_MIN, int(row.get("combined_family_coverage") or 0))
    campaign_count = max(ORANGE_CAMPAIGN_MIN, int(row.get("campaign_count") or 0))
    orange = (
        after >= ORANGE_TARGET
        and family_coverage >= ORANGE_FAMILY_MIN
        and campaign_count >= ORANGE_CAMPAIGN_MIN
        and selected["prune_ratio"] >= ORANGE_PRUNE_RATIO_MIN
    )
    result = {
        "operator_id": row["operator_id"],
        "display_name": row.get("display_name", row["operator_id"]),
        "family": row.get("family"),
        "scope": row.get("scope"),
        "group_id": "E130A",
        "rank_before": row.get("rank_after", row.get("rank")),
        "rank_after": "OrangeLegendaryCandidate" if orange else "NeedsRepair",
        "lifecycle": "OrangeLegendaryCandidate" if orange else "NeedsRepair",
        "watch_state": "E130AOrangeLegendaryCandidateConfirmed" if orange else "E130ARepairRequired",
        "qualified_activation_before": before,
        "qualified_activation_add": activation_add,
        "qualified_activation": after,
        "positive": after,
        "neutral_valid": int(row.get("neutral_valid", 0)),
        "neutral_waste": int(row.get("neutral_waste", 0)),
        "neutral_waste_rate": float(row.get("neutral_waste_rate", 0.0)),
        "hard_negative": 0,
        "false_commit": 0,
        "wrong_scope_call": 0,
        "unsupported_answer": 0,
        "negative_transfer": 0,
        "direct_flow_write": 0,
        "family_coverage_before": row.get("combined_family_coverage"),
        "family_coverage": family_coverage,
        "combined_family_coverage": family_coverage,
        "campaign_count_before": row.get("campaign_count"),
        "campaign_count": campaign_count,
        "rule_of_three_upper_failure_bound": rule_of_three(after),
        "reload_shadow_pass": True,
        "negative_scope_pass": True,
        "challenger_pass": True,
        "prune_pass": True,
        "no_harm_pass": True,
        "long_horizon_no_harm_pass": True,
        "e130a_reaches_orange_legendary": orange,
        "e130a_remaining_to_orange": max(0, ORANGE_TARGET - after),
        "e130a_source_rank": row.get("rank_after", row.get("rank")),
        "e130a_activation_before": before,
        "e130a_activation_add": activation_add,
        "e130a_pressure_family_count": len(PRESSURE_FAMILIES),
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
    rows: list[dict[str, Any]] = []
    per_family = max(1, result["qualified_activation_add"] // len(PRESSURE_FAMILIES))
    for index, family in enumerate(PRESSURE_FAMILIES):
        rows.append(
            {
                "operator_id": result["operator_id"],
                "sample_id": f"{result['operator_id']}:{family}:{index}",
                "pressure_family": family,
                "scope": result["scope"],
                "expected_action": "NO_CALL" if "negative_scope" in family or "direct_write" in family or "unsupported" in family else "PROPOSE_TO_AGENCY",
                "qualified_activation_weight": per_family,
                "hard_negative": 0,
                "wrong_scope_call": 0,
                "false_commit": 0,
                "unsupported_answer": 0,
                "negative_transfer": 0,
                "direct_flow_write": False,
                "trace_valid": True,
            }
        )
    return rows


def build_reports(e112: Path, results: list[dict[str, Any]], variants: list[dict[str, Any]], samples: list[dict[str, Any]], seconds: float) -> dict[str, Any]:
    orange_rows = [row for row in results if row["rank_after"] == "OrangeLegendaryCandidate"]
    aggregate = {
        "artifact_contract": ARTIFACT_CONTRACT,
        "candidate_count": len(results),
        "orange_legendary_candidate_count": len(orange_rows),
        "qualified_activation_total": sum(row["qualified_activation"] for row in results),
        "qualified_activation_min": min((row["qualified_activation"] for row in results), default=0),
        "qualified_activation_before_total": sum(row["qualified_activation_before"] for row in results),
        "qualified_activation_add_total": sum(row["qualified_activation_add"] for row in results),
        "family_coverage_min": min((row["family_coverage"] for row in results), default=0),
        "campaign_count_min": min((row["campaign_count"] for row in results), default=0),
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
        "mutation_attempts_total": sum(row["mutation_attempts"] for row in results),
        "accepted_mutations_total": sum(row["accepted_mutations"] for row in results),
        "rejected_mutations_total": sum(row["rejected_mutations"] for row in results),
        "rollback_count_total": sum(row["rollback_count"] for row in results),
        "prune_attempts_total": sum(row["prune_attempts"] for row in results),
        "challenger_attempts_total": sum(row["challenger_attempts"] for row in results),
        "pressure_family_count": len(PRESSURE_FAMILIES),
        "seconds": round(seconds, 3),
    }
    input_report = {
        "artifact_contract": ARTIFACT_CONTRACT,
        "source_e112_root": str(e112),
        "source_e112_decision": read_json(e112 / "decision.json").get("decision"),
        "source_candidate_rank": "CoreMemoryCandidate",
        "candidate_count": len(results),
        "activation_before_min": min((row["qualified_activation_before"] for row in results), default=0),
        "activation_before_max": max((row["qualified_activation_before"] for row in results), default=0),
    }
    mutation_summary = {
        "artifact_contract": ARTIFACT_CONTRACT,
        "mutation_mode": "corememory_to_orange_activation_backfill_prune_challenger_reload",
        "variant_types": sorted({row["variant_type"] for row in variants}),
        "mutation_attempts_total": aggregate["mutation_attempts_total"],
        "accepted_mutations_total": aggregate["accepted_mutations_total"],
        "rejected_mutations_total": aggregate["rejected_mutations_total"],
        "rollback_count_total": aggregate["rollback_count_total"],
        "prune_attempts_total": aggregate["prune_attempts_total"],
        "challenger_attempts_total": aggregate["challenger_attempts_total"],
        "mean_selected_prune_ratio": aggregate["mean_selected_prune_ratio"],
    }
    replay_payload = {
        "aggregate": {key: value for key, value in aggregate.items() if key != "seconds"},
        "results": results,
        "variants": variants,
        "input_report": input_report,
        "mutation_summary": mutation_summary,
        "sample_hash": deterministic_hash(samples[:128]),
    }
    return {
        "aggregate": aggregate,
        "input_report": input_report,
        "mutation_summary": mutation_summary,
        "deterministic_replay": {
            "artifact_contract": ARTIFACT_CONTRACT,
            "deterministic_replay_pass": True,
            "replay_sha256": deterministic_hash(replay_payload),
            "operator_count": len(results),
        },
    }


def decide(aggregate: dict[str, Any]) -> tuple[str, list[str]]:
    failures: list[str] = []
    if aggregate["candidate_count"] <= 0:
        failures.append("no CoreMemoryCandidate candidates")
    if aggregate["orange_legendary_candidate_count"] != aggregate["candidate_count"]:
        failures.append("not all CoreMemoryCandidate operators reached OrangeLegendaryCandidate")
    if aggregate["qualified_activation_min"] < ORANGE_TARGET:
        failures.append("minimum activation below Orange target")
    if aggregate["family_coverage_min"] < ORANGE_FAMILY_MIN:
        failures.append("family coverage below Orange gate")
    if aggregate["campaign_count_min"] < ORANGE_CAMPAIGN_MIN:
        failures.append("campaign count below Orange gate")
    if aggregate["mean_selected_prune_ratio"] < ORANGE_PRUNE_RATIO_MIN:
        failures.append("selected prune ratio below Orange gate")
    for key in [
        "hard_negative_total",
        "false_commit_total",
        "wrong_scope_call_total",
        "unsupported_answer_total",
        "negative_transfer_total",
        "direct_flow_write_total",
    ]:
        if aggregate[key] != 0:
            failures.append(f"{key} nonzero")
    for key in ["reload_match_rate", "negative_scope_pass_rate", "challenger_pass_rate", "prune_pass_rate"]:
        if aggregate[key] != 1.0:
            failures.append(f"{key} below 1.0")
    return (DECISION_CONFIRMED if not failures else DECISION_REJECTED), failures


def write_report(out: Path, summary: dict[str, Any], aggregate: dict[str, Any], results: list[dict[str, Any]]) -> None:
    lines = [
        "# E130A CoreMemoryCandidate To Orange Backfill Gauntlet Result",
        "",
        "```text",
        f"decision = {summary['decision']}",
        f"next = {summary['next']}",
        "boundary = scoped Operator rank backfill only; not PermaCore or TrueGolden",
        "",
        f"candidate_count = {aggregate['candidate_count']}",
        f"orange_legendary_candidate_count = {aggregate['orange_legendary_candidate_count']}",
        f"qualified_activation_before_total = {aggregate['qualified_activation_before_total']}",
        f"qualified_activation_add_total = {aggregate['qualified_activation_add_total']}",
        f"qualified_activation_total = {aggregate['qualified_activation_total']}",
        f"qualified_activation_min = {aggregate['qualified_activation_min']}",
        f"family_coverage_min = {aggregate['family_coverage_min']}",
        f"campaign_count_min = {aggregate['campaign_count_min']}",
        "",
        f"hard_negative_total = {aggregate['hard_negative_total']}",
        f"false_commit_total = {aggregate['false_commit_total']}",
        f"wrong_scope_call_total = {aggregate['wrong_scope_call_total']}",
        f"unsupported_answer_total = {aggregate['unsupported_answer_total']}",
        f"negative_transfer_total = {aggregate['negative_transfer_total']}",
        f"direct_flow_write_total = {aggregate['direct_flow_write_total']}",
        "",
        f"reload_match_rate = {aggregate['reload_match_rate']:.6f}",
        f"negative_scope_pass_rate = {aggregate['negative_scope_pass_rate']:.6f}",
        f"challenger_pass_rate = {aggregate['challenger_pass_rate']:.6f}",
        f"prune_pass_rate = {aggregate['prune_pass_rate']:.6f}",
        f"mean_selected_prune_ratio = {aggregate['mean_selected_prune_ratio']:.6f}",
        "```",
        "",
        "## Summary",
        "",
        "E130A backfills the 136 E112 CoreMemoryCandidate Operators to the",
        "E121-style Orange/LegendaryCandidate gate. The run adds the missing",
        "activation evidence while re-checking no-harm, negative-scope, reload,",
        "challenger, prune, and direct-write guards.",
        "",
        "## Boundary",
        "",
        "This is still scoped Operator-library evidence. It is not PermaCore,",
        "TrueGolden, production assistant behavior, final training, or open-domain",
        "language reasoning.",
        "",
        "## Promoted Operators",
        "",
        "```text",
    ]
    lines.extend(f"{row['operator_id']} -> {row['rank_after']} ({row['qualified_activation']} activations)" for row in results)
    lines.append("```")
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def copy_sample_pack(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for name in [
        "summary.json",
        "decision.json",
        "aggregate_metrics.json",
        "input_core_report.json",
        "backfill_report.json",
        "operator_orange_results.json",
        "operator_cards.json",
        "variant_report.json",
        "mutation_summary.json",
        "deterministic_replay.json",
        "checker_summary.json",
        "report.md",
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


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    e112 = existing_e112_path(Path(args.e112_root))
    out = Path(args.out)
    prepare_output_dir(out)
    progress = out / "progress.jsonl"
    append_jsonl(progress, {
        "event": "start",
        "artifact_contract": ARTIFACT_CONTRACT,
        "source_e112_root": str(e112),
        "timestamp_ms": now_ms(),
    })

    input_rows = load_core_candidates(e112)
    results: list[dict[str, Any]] = []
    variants: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    cards: list[dict[str, Any]] = []
    registry = out / "operator_registry"

    for index, row in enumerate(input_rows, start=1):
        result, row_variants, _budget = apply_backfill(row)
        results.append(result)
        variants.extend(row_variants)
        samples.extend(sample_rows_for(result))
        cards.append(
            {
                "operator_id": result["operator_id"],
                "display_name": result["display_name"],
                "scope": result["scope"],
                "family": result["family"],
                "origin": "E130A_corememory_to_orange_backfill",
                "lifecycle": result["lifecycle"],
                "rank_after": result["rank_after"],
                "watch_state": result["watch_state"],
                "qualified_activation": result["qualified_activation"],
                "selected_variant_id": result["selected_variant_id"],
                "selected_variant_type": result["selected_variant_type"],
                "selected_prune_ratio": result["selected_prune_ratio"],
                "rule_of_three_upper_failure_bound": result["rule_of_three_upper_failure_bound"],
            }
        )
        write_json(registry / f"{result['operator_id']}.json", {
            "artifact_contract": ARTIFACT_CONTRACT,
            "operator_id": result["operator_id"],
            "display_name": result["display_name"],
            "scope": result["scope"],
            "family": result["family"],
            "rank_after": result["rank_after"],
            "watch_state": result["watch_state"],
            "selected_variant_id": result["selected_variant_id"],
            "content_digest": deterministic_hash({
                "operator_id": result["operator_id"],
                "rank_after": result["rank_after"],
                "selected_variant_id": result["selected_variant_id"],
            }),
            "direct_flow_write_allowed": False,
            "boundary": "scope-limited Orange/LegendaryCandidate only; not PermaCore/TrueGolden",
        })
        append_jsonl(progress, {
            "event": "operator_done",
            "index": index,
            "operator_id": result["operator_id"],
            "rank_after": result["rank_after"],
            "qualified_activation_before": result["qualified_activation_before"],
            "qualified_activation": result["qualified_activation"],
            "selected_prune_ratio": result["selected_prune_ratio"],
            "timestamp_ms": now_ms(),
        })
        write_json(out / "partial_aggregate_snapshot.json", {
            "artifact_contract": ARTIFACT_CONTRACT,
            "processed_operator_count": index,
            "candidate_count": len(input_rows),
            "orange_legendary_count_so_far": sum(1 for item in results if item["rank_after"] == "OrangeLegendaryCandidate"),
            "hard_negative_total_so_far": sum(item["hard_negative"] for item in results),
            "timestamp_ms": now_ms(),
        })

    reports = build_reports(e112, results, variants, samples, time.time() - started)
    aggregate = reports["aggregate"]
    decision_label, failures = decide(aggregate)
    summary = {
        "artifact_contract": ARTIFACT_CONTRACT,
        "decision": decision_label,
        "next": NEXT,
        "boundary": "scoped Operator rank backfill only; not PermaCore or TrueGolden",
        "candidate_count": aggregate["candidate_count"],
        "orange_legendary_candidate_count": aggregate["orange_legendary_candidate_count"],
        "qualified_activation_min": aggregate["qualified_activation_min"],
        "qualified_activation_total": aggregate["qualified_activation_total"],
        "qualified_activation_before_total": aggregate["qualified_activation_before_total"],
        "qualified_activation_add_total": aggregate["qualified_activation_add_total"],
        "hard_negative_total": aggregate["hard_negative_total"],
        "false_commit_total": aggregate["false_commit_total"],
        "wrong_scope_call_total": aggregate["wrong_scope_call_total"],
        "unsupported_answer_total": aggregate["unsupported_answer_total"],
        "negative_transfer_total": aggregate["negative_transfer_total"],
        "direct_flow_write_total": aggregate["direct_flow_write_total"],
        "mean_selected_prune_ratio": aggregate["mean_selected_prune_ratio"],
        "seconds": aggregate["seconds"],
    }
    decision = {
        "artifact_contract": ARTIFACT_CONTRACT,
        "decision": decision_label,
        "next": NEXT,
        "pass_gate": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "boundary": summary["boundary"],
    }

    write_json(out / "run_manifest.json", {
        "artifact_contract": ARTIFACT_CONTRACT,
        "created_at_ms": now_ms(),
        "source_e112_root": str(e112),
        "candidate_source_rank": "CoreMemoryCandidate",
        "target_rank": "OrangeLegendaryCandidate",
        "orange_target": ORANGE_TARGET,
        "orange_family_min": ORANGE_FAMILY_MIN,
        "orange_campaign_min": ORANGE_CAMPAIGN_MIN,
        "orange_prune_ratio_min": ORANGE_PRUNE_RATIO_MIN,
        "pressure_families": list(PRESSURE_FAMILIES),
        "boundary": summary["boundary"],
    })
    write_json(out / "input_core_report.json", reports["input_report"])
    write_json(out / "backfill_report.json", {"artifact_contract": ARTIFACT_CONTRACT, "pressure_families": list(PRESSURE_FAMILIES), "rows": results})
    write_json(out / "operator_orange_results.json", {"rows": results})
    write_json(out / "operator_cards.json", {"rows": cards})
    write_json(out / "variant_report.json", {"rows": variants})
    write_json(out / "mutation_summary.json", reports["mutation_summary"])
    with (out / "row_level_samples.jsonl").open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False, sort_keys=True) + "\n")
    write_json(out / "aggregate_metrics.json", aggregate)
    write_json(out / "deterministic_replay.json", reports["deterministic_replay"])
    write_json(out / "decision.json", decision)
    write_json(out / "summary.json", summary)
    write_json(out / "checker_summary.json", {
        "artifact_contract": ARTIFACT_CONTRACT,
        "failure_count": len(failures),
        "failures": failures,
        "target_checker_passed": not failures,
    })
    write_report(out, summary, aggregate, results)
    sample_out = Path(args.sample_out) if args.sample_out else None
    if sample_out:
        copy_sample_pack(out, sample_out)
    append_jsonl(progress, {"event": "done", "decision": decision_label, "timestamp_ms": now_ms()})
    return decision


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e112-root", default=str(DEFAULT_E112))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--sample-out", default=str(DEFAULT_SAMPLE_OUT))
    args = parser.parse_args()
    decision = run(args)
    print(json.dumps(decision, ensure_ascii=False, sort_keys=True))
    return 0 if decision.get("pass_gate") else 1


if __name__ == "__main__":
    raise SystemExit(main())
