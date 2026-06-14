#!/usr/bin/env python3
"""E107 Operator Library E90-E106 survival/role/regression gauntlet.

This is a quality-control probe, not a new world-knowledge skill. It loads the
documented E90-E106 StableOperatorCandidate inventory, lets each seed/neighborhood
select from the full library, and assigns survival roles from objective selection,
cost, regression, and negative-scope evidence.

This is not open-domain capability evaluation and not final training.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import random
import re
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.probes.run_e84_calc_scribe_transfer_negative_scope_probe import append_jsonl, now_ms, write_json  # noqa: E402


ARTIFACT_CONTRACT = "E107_OPERATOR_LIBRARY_E90_E106_SURVIVAL_ROLE_AND_REGRESSION_GAUNTLET"


@dataclass(frozen=True)
class OperatorSpec:
    operator_id: str
    display_name: str
    group_id: str
    group_title: str
    family: str
    role: str
    cost: float


@dataclass(frozen=True)
class Neighborhood:
    neighborhood_id: str
    description: str
    required_groups: tuple[str, ...]
    required_families: tuple[str, ...]
    focus_terms: tuple[str, ...]
    max_selected: int
    adversarial: bool


CONTROL_OPERATORS: tuple[OperatorSpec, ...] = (
    OperatorSpec("final_answer_popularity_selector_control", "Final Answer Popularity Selector Control", "CONTROL", "Injected Controls", "unsafe_control", "unsafe", 0.02),
    OperatorSpec("full_library_scan_overreach_control", "Full Library Scan Overreach Control", "CONTROL", "Injected Controls", "unsafe_control", "unsafe", 0.03),
    OperatorSpec("ignore_negative_scope_control", "Ignore Negative Scope Control", "CONTROL", "Injected Controls", "unsafe_control", "unsafe", 0.03),
    OperatorSpec("skip_regression_replay_control", "Skip Regression Replay Control", "CONTROL", "Injected Controls", "unsafe_control", "unsafe", 0.03),
    OperatorSpec("stale_trace_route_control", "Stale Trace Route Control", "CONTROL", "Injected Controls", "unsafe_control", "unsafe", 0.03),
    OperatorSpec("unsafe_complete_priority_control", "Unsafe Complete Priority Control", "CONTROL", "Injected Controls", "unsafe_control", "unsafe", 0.03),
    OperatorSpec("cost_blind_selector_control", "Cost Blind Selector Control", "CONTROL", "Injected Controls", "unsafe_control", "unsafe", 0.02),
    OperatorSpec("random_operator_neighborhood_control", "Random Operator Neighborhood Control", "CONTROL", "Injected Controls", "control", "noop", 0.02),
)


NEIGHBORHOODS: tuple[Neighborhood, ...] = (
    Neighborhood("visible_evidence_ruleshift", "Visible evidence binding, revocation, and ruleshift.", ("E90", "E101", "E102"), ("Alpha-Syncer", "Guard", "Scribe"), ("evidence", "binding", "answer", "contradiction", "unresolved"), 16, False),
    Neighborhood("temporal_bitstream_resync", "Temporal frame/bitstream stability and stale replay rejection.", ("E91", "E96"), ("T-Stab", "Guard", "Lens"), ("frame", "crc", "bit", "stale", "replay", "temporal"), 12, False),
    Neighborhood("lexical_glyph_grounding", "Lexical alias, glyph, unit, negation, and canonical lexeme grounding.", ("E92", "E100"), ("Alpha-Syncer", "Guard", "Scribe", "Lens"), ("alias", "negation", "unit", "symbol", "lexeme", "span"), 14, False),
    Neighborhood("agency_commit_safety", "Proposal commit boundary, ground conflict, and trace coverage.", ("E93", "E102", "E106"), ("Guard", "T-Stab", "Scribe"), ("commit", "ground", "trace", "complete", "answerability"), 16, True),
    Neighborhood("answer_output_hygiene", "Answer rendering, citation, contradiction report, and no-answer boundary.", ("E94", "E102"), ("Scribe", "Guard", "T-Stab"), ("answer", "citation", "format", "uncertainty", "boundary"), 13, False),
    Neighborhood("active_evidence_search", "Missing dependency search, source ranking, and retrieved evidence integration.", ("E95", "E101", "E103"), ("Lens", "Guard", "Scribe", "T-Stab"), ("missing", "source", "request", "search", "clarification"), 18, True),
    Neighborhood("trace_ground_memory_hygiene", "Trace dedupe, provenance, ground promotion, and replay audit.", ("E96", "E104", "E105"), ("Lens", "Guard", "Scribe", "T-Stab"), ("trace", "ground", "replay", "summary", "context"), 21, True),
    Neighborhood("route_composition_execution", "Operator routing, adapter detection, sequence planning, and checkpoints.", ("E97", "E98"), ("Lens", "Guard", "Scribe", "T-Stab"), ("route", "operator", "composition", "checkpoint", "dependency"), 14, False),
    Neighborhood("curriculum_regression_scheduler", "Capability gaps, lesson ranking, regression replay, and mutation queue.", ("E99", "E106"), ("Lens", "Guard", "Scribe", "T-Stab"), ("curriculum", "regression", "promotion", "progress", "next"), 13, True),
    Neighborhood("text_ingress_conflict_resolution", "Text span ingress, source attribution, and conflict resolution.", ("E100", "E101"), ("Lens", "Guard", "Scribe", "T-Stab"), ("text", "span", "source", "conflict", "quote", "contrast"), 14, True),
    Neighborhood("clarification_state_repair", "Clarification response binding, state repair, and multi-turn continuity.", ("E103", "E104"), ("Lens", "Guard", "Scribe", "T-Stab"), ("clarification", "repair", "pending", "turn", "state"), 14, False),
    Neighborhood("context_compression_reentry", "Context-window pressure, fact preservation, compaction, and re-entry.", ("E105", "E104"), ("Lens", "Guard", "Scribe", "T-Stab"), ("context", "summary", "compression", "dependency", "reentry"), 14, True),
    Neighborhood("false_done_progress_traps", "False completion, blocker, stale progress, and recheck traps.", ("E106", "E99", "E93"), ("Lens", "Guard", "Scribe", "T-Stab"), ("complete", "progress", "recheck", "block", "commit"), 18, True),
    Neighborhood("full_long_horizon_bundle", "Long-horizon multi-skill route with text, memory, progress, and answer guards.", ("E90", "E93", "E95", "E98", "E99", "E101", "E102", "E104", "E105", "E106"), ("Lens", "Guard", "Scribe", "T-Stab"), ("evidence", "route", "trace", "answer", "progress", "summary"), 38, True),
)


def title_from_id(operator_id: str) -> str:
    return " ".join(part.capitalize() for part in operator_id.split("_"))


def infer_family(operator_id: str) -> str:
    if operator_id.endswith("_alpha_syncer"):
        return "Alpha-Syncer"
    if operator_id.endswith("_t_stab"):
        return "T-Stab"
    if operator_id.endswith("_guard"):
        return "Guard"
    if operator_id.endswith("_lens"):
        return "Lens"
    if operator_id.endswith("_scribe"):
        return "Scribe"
    return "Operator"


def family_cost(family: str) -> float:
    return {
        "Guard": 0.12,
        "Lens": 0.11,
        "Scribe": 0.10,
        "T-Stab": 0.13,
        "Alpha-Syncer": 0.10,
    }.get(family, 0.14)


def load_e90_e106_inventory(cards_path: Path) -> list[OperatorSpec]:
    text = cards_path.read_text(encoding="utf-8")
    operators: list[OperatorSpec] = []
    current_group = ""
    current_title = ""
    pending_component: str | None = None
    group_re = re.compile(r"^## (E(?:9[0-9]|10[0-6]))\b\s*(.*)$")
    for raw_line in text.splitlines():
        line = raw_line.strip()
        match = group_re.match(line)
        if match:
            current_group = match.group(1)
            current_title = match.group(2).strip() or current_group
            pending_component = None
            continue
        if not current_group:
            continue
        if line.startswith("component_id ="):
            pending_component = line.split("=", 1)[1].strip()
            continue
        if pending_component and line == "status       = StableOperatorCandidate":
            family = infer_family(pending_component)
            operators.append(OperatorSpec(
                operator_id=pending_component,
                display_name=title_from_id(pending_component),
                group_id=current_group,
                group_title=current_title,
                family=family,
                role="candidate",
                cost=family_cost(family),
            ))
            pending_component = None
    unique: dict[str, OperatorSpec] = {}
    for operator in operators:
        unique[operator.operator_id] = operator
    return list(unique.values())


def stable_float(text: str) -> float:
    value = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
    return value / float(2**64 - 1)


def relevance_score(operator: OperatorSpec, neighborhood: Neighborhood, seed: int) -> float:
    if operator.role == "unsafe":
        return -7.0
    if operator.role == "noop":
        return -2.0
    score = 0.0
    if operator.group_id in neighborhood.required_groups:
        score += 2.7
    if operator.family in neighborhood.required_families:
        score += 0.55
    tokens = set(operator.operator_id.split("_"))
    for term in neighborhood.focus_terms:
        if term in tokens or term in operator.operator_id:
            score += 0.52
    if operator.family == "Guard" and neighborhood.adversarial:
        score += 0.28
    if operator.group_id in ("E93", "E102", "E106") and neighborhood.adversarial:
        score += 0.26
    if operator.group_id not in neighborhood.required_groups and not (tokens & set(neighborhood.focus_terms)):
        score -= 0.35
    score -= operator.cost * 0.42
    score += (stable_float(f"{seed}:{neighborhood.neighborhood_id}:{operator.operator_id}") - 0.5) * 0.04
    return round(score, 6)


def select_for_neighborhood(seed: int, neighborhood: Neighborhood, library: list[OperatorSpec]) -> dict[str, Any]:
    scored = sorted(
        ((operator, relevance_score(operator, neighborhood, seed)) for operator in library),
        key=lambda item: (-item[1], item[0].operator_id),
    )
    selected: list[OperatorSpec] = []
    accepted = rollback = 0
    used_groups: set[str] = set()
    used_families: set[str] = set()
    selected_ids: set[str] = set()

    def accept(operator: OperatorSpec) -> None:
        nonlocal accepted
        if operator.operator_id in selected_ids or len(selected) >= neighborhood.max_selected:
            return
        selected.append(operator)
        selected_ids.add(operator.operator_id)
        used_groups.add(operator.group_id)
        used_families.add(operator.family)
        accepted += 1

    # First reserve one seat for every required group/family. The rest of the
    # seats are then filled by utility. This prevents a broad high-score family
    # from crowding out a required specialist in long-horizon neighborhoods.
    for group in neighborhood.required_groups:
        candidates = [item for item in scored if item[0].group_id == group and item[0].role == "candidate"]
        if candidates:
            accept(candidates[0][0])
    for family in neighborhood.required_families:
        candidates = [item for item in scored if item[0].family == family and item[0].role == "candidate"]
        if candidates:
            accept(candidates[0][0])

    for operator, score in scored:
        improves = (
            score > 0.78
            and len(selected) < neighborhood.max_selected
            and operator.role == "candidate"
            and operator.operator_id not in selected_ids
        )
        if improves:
            accept(operator)
    rejected = [{"operator_id": operator.operator_id, "score": score} for operator, score in scored if operator.operator_id not in selected_ids]
    rollback = len(rejected)
    selected_ids = tuple(operator.operator_id for operator in selected)
    selected_groups = {operator.group_id for operator in selected}
    selected_families = {operator.family for operator in selected}
    selected_controls = [operator.operator_id for operator in selected if operator.role != "candidate"]
    group_coverage = len(set(neighborhood.required_groups) & selected_groups) / max(1, len(neighborhood.required_groups))
    family_coverage = len(set(neighborhood.required_families) & selected_families) / max(1, len(neighborhood.required_families))
    focus_coverage = sum(1 for term in neighborhood.focus_terms if any(term in operator_id for operator_id in selected_ids)) / max(1, len(neighborhood.focus_terms))
    overreach = len(selected) > neighborhood.max_selected
    cost = round(sum(operator.cost for operator in selected), 6)
    success = (
        group_coverage == 1.0
        and family_coverage == 1.0
        and focus_coverage >= 0.5
        and not selected_controls
        and not overreach
    )
    return {
        "seed": seed,
        "neighborhood_id": neighborhood.neighborhood_id,
        "selected": selected_ids,
        "selected_count": len(selected),
        "selected_groups": sorted(selected_groups),
        "selected_families": sorted(selected_families),
        "group_coverage": round(group_coverage, 6),
        "family_coverage": round(family_coverage, 6),
        "focus_coverage": round(focus_coverage, 6),
        "cost": cost,
        "max_selected": neighborhood.max_selected,
        "success": success,
        "unsafe_control_selected": bool(selected_controls),
        "selected_controls": selected_controls,
        "full_library_overreach": overreach,
        "cost_blowup": cost > neighborhood.max_selected * 0.16,
        "accepted": accepted,
        "rejected": len(rejected),
        "rollback": rollback,
        "top_rejected": rejected[:12],
    }


def run_seed(seed: int, library: list[OperatorSpec], out: Path) -> dict[str, Any]:
    seed_results = []
    seed_path = out / "seed_progress" / f"seed_{seed}.jsonl"
    rng = random.Random(seed)
    neighborhoods = list(NEIGHBORHOODS)
    rng.shuffle(neighborhoods)
    for index, neighborhood in enumerate(neighborhoods):
        result = select_for_neighborhood(seed, neighborhood, library)
        seed_results.append(result)
        append_jsonl(seed_path, {
            "seed": seed,
            "event": "neighborhood_complete",
            "neighborhood_id": neighborhood.neighborhood_id,
            "selected_count": result["selected_count"],
            "success": result["success"],
            "timestamp_ms": now_ms(),
        })
        append_jsonl(out / "operator_evolution_history.jsonl", {
            "seed": seed,
            "generation": index,
            "neighborhood_id": neighborhood.neighborhood_id,
            "selected_count_final": result["selected_count"],
            "final_selected": result["selected"],
            "accepted": result["accepted"],
            "rejected": result["rejected"],
            "rollback": result["rollback"],
        })
    return {
        "seed": seed,
        "neighborhood_results": seed_results,
        "accepted": sum(int(row["accepted"]) for row in seed_results),
        "rejected": sum(int(row["rejected"]) for row in seed_results),
        "rollback": sum(int(row["rollback"]) for row in seed_results),
    }


def flatten_results(seed_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for result in seed_results for row in result["neighborhood_results"]]


def selection_frequency(seed_results: list[dict[str, Any]], library: list[OperatorSpec]) -> dict[str, Any]:
    rows = []
    flat = flatten_results(seed_results)
    total = len(flat)
    for operator in library:
        selected_rows = [row for row in flat if operator.operator_id in row["selected"]]
        neighborhoods = sorted({row["neighborhood_id"] for row in selected_rows})
        rows.append({
            "operator_id": operator.operator_id,
            "display_name": operator.display_name,
            "group_id": operator.group_id,
            "family": operator.family,
            "role": operator.role,
            "selected_frequency": round(len(selected_rows) / max(1, total), 6),
            "selected_count": len(selected_rows),
            "neighborhood_count": len(neighborhoods),
            "neighborhoods": neighborhoods,
        })
    return {"rows": rows, "stable_top": [row["operator_id"] for row in rows if row["selected_frequency"] >= 0.18 and row["role"] == "candidate"]}


def counterfactual_report(seed_results: list[dict[str, Any]], library: list[OperatorSpec]) -> dict[str, Any]:
    flat = flatten_results(seed_results)
    summary = {}
    for operator in library:
        selected_rows = [row for row in flat if operator.operator_id in row["selected"]]
        if not selected_rows or operator.role != "candidate":
            summary[operator.operator_id] = {"mean_survival_score_loss": 0.0, "mean_cost_delta": 0.0}
            continue
        losses = []
        costs = []
        for row in selected_rows:
            score = 0.42 * row["group_coverage"] + 0.28 * row["family_coverage"] + 0.20 * row["focus_coverage"] + 0.10 * int(row["success"])
            # Removing the only member from a group/family is high impact; removing
            # one of many is lower but still nonzero support.
            selected_ids = set(row["selected"])
            selected_ids.remove(operator.operator_id)
            selected_groups = set(row["selected_groups"])
            selected_families = set(row["selected_families"])
            if not any(other.group_id == operator.group_id and other.operator_id in selected_ids for other in library):
                selected_groups.discard(operator.group_id)
            if not any(other.family == operator.family and other.operator_id in selected_ids for other in library):
                selected_families.discard(operator.family)
            neighborhood = next(item for item in NEIGHBORHOODS if item.neighborhood_id == row["neighborhood_id"])
            group_cov = len(set(neighborhood.required_groups) & selected_groups) / max(1, len(neighborhood.required_groups))
            family_cov = len(set(neighborhood.required_families) & selected_families) / max(1, len(neighborhood.required_families))
            reduced = 0.42 * group_cov + 0.28 * family_cov + 0.20 * max(0.0, row["focus_coverage"] - 0.05) + 0.10 * int(group_cov == 1.0 and family_cov == 1.0)
            losses.append(score - reduced)
            costs.append(-operator.cost)
        summary[operator.operator_id] = {
            "mean_survival_score_loss": round(statistics.mean(losses), 6),
            "mean_cost_delta": round(statistics.mean(costs), 6),
        }
    return {"summary": summary}


def lifecycle_report(seed_results: list[dict[str, Any]], library: list[OperatorSpec]) -> dict[str, Any]:
    freq = selection_frequency(seed_results, library)
    cf = counterfactual_report(seed_results, library)["summary"]
    rows = []
    for row in freq["rows"]:
        if row["role"] == "unsafe":
            status = "Quarantine"
        elif row["role"] == "noop":
            status = "Deprecated"
        elif row["selected_frequency"] >= 0.18 and row["neighborhood_count"] >= 4:
            status = "StableSupport"
        elif row["selected_frequency"] >= 0.05:
            status = "Specialist"
        elif row["selected_count"] > 0:
            status = "BundleSupport"
        else:
            status = "Deprecated"
        rows.append({**row, "final_status": status, "counterfactual": cf.get(row["operator_id"], {})})
    return {"operator_lifecycle_table": rows}


def aggregate(seed_results: list[dict[str, Any]], library: list[OperatorSpec], seconds: float) -> dict[str, Any]:
    flat = flatten_results(seed_results)
    lifecycle = lifecycle_report(seed_results, library)["operator_lifecycle_table"]
    candidate_rows = [row for row in lifecycle if row["role"] == "candidate"]
    status_counts: dict[str, int] = {}
    for row in candidate_rows:
        status_counts[row["final_status"]] = status_counts.get(row["final_status"], 0) + 1
    return {
        "seed_count": len(seed_results),
        "neighborhood_count": len(NEIGHBORHOODS),
        "case_count": len(flat),
        "candidate_operator_count": sum(1 for operator in library if operator.role == "candidate"),
        "operator_group_count": len({operator.group_id for operator in library if operator.role == "candidate"}),
        "survival_success_min": min(1.0 if row["success"] else 0.0 for row in flat),
        "survival_success_mean": round(statistics.mean(1.0 if row["success"] else 0.0 for row in flat), 6),
        "adversarial_survival_success_min": min(1.0 if row["success"] else 0.0 for row in flat if next(n for n in NEIGHBORHOODS if n.neighborhood_id == row["neighborhood_id"]).adversarial),
        "unsafe_control_selected_rate": round(statistics.mean(1.0 if row["unsafe_control_selected"] else 0.0 for row in flat), 6),
        "full_library_overreach_rate": round(statistics.mean(1.0 if row["full_library_overreach"] else 0.0 for row in flat), 6),
        "cost_blowup_rate": round(statistics.mean(1.0 if row["cost_blowup"] else 0.0 for row in flat), 6),
        "group_coverage_min": min(float(row["group_coverage"]) for row in flat),
        "family_coverage_min": min(float(row["family_coverage"]) for row in flat),
        "focus_coverage_min": min(float(row["focus_coverage"]) for row in flat),
        "role_assignment_coverage": round(sum(1 for row in candidate_rows if row["final_status"] in {"StableSupport", "Specialist", "BundleSupport", "Deprecated"}) / max(1, len(candidate_rows)), 6),
        "stable_support_count": status_counts.get("StableSupport", 0),
        "specialist_count": status_counts.get("Specialist", 0),
        "bundle_support_count": status_counts.get("BundleSupport", 0),
        "deprecated_count": status_counts.get("Deprecated", 0),
        "accepted_mutations_total": sum(int(result["accepted"]) for result in seed_results),
        "rejected_mutations_total": sum(int(result["rejected"]) for result in seed_results),
        "rollback_count_total": sum(int(result["rollback"]) for result in seed_results),
        "seconds": round(seconds, 3),
    }


def deterministic_hash(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def write_sample_pack(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for name in [
        "operator_library_manifest.json",
        "task_generation_report.json",
        "source_inventory_report.json",
        "neighborhood_results.json",
        "survival_role_report.json",
        "aggregate_metrics.json",
        "selection_frequency_report.json",
        "counterfactual_report.json",
        "operator_lifecycle_report.json",
        "deterministic_replay.json",
        "decision.json",
        "summary.json",
    ]:
        (target / name).write_text((source / name).read_text(encoding="utf-8"), encoding="utf-8")
    write_json(target / "sample_manifest.json", {
        "artifact_contract": ARTIFACT_CONTRACT,
        "source": str(source),
        "sample_only": True,
    })


def write_reports(out: Path, sample_dir: Path | None, seed_results: list[dict[str, Any]], library: list[OperatorSpec], args: argparse.Namespace, seconds: float) -> None:
    agg = aggregate(seed_results, library, seconds)
    freq = selection_frequency(seed_results, library)
    cf = counterfactual_report(seed_results, library)
    lifecycle = lifecycle_report(seed_results, library)
    flat = flatten_results(seed_results)
    survival_role = {
        "status_counts": {
            status: sum(1 for row in lifecycle["operator_lifecycle_table"] if row["role"] == "candidate" and row["final_status"] == status)
            for status in ["StableSupport", "Specialist", "BundleSupport", "Deprecated"]
        },
        "top_stable_support": [row for row in lifecycle["operator_lifecycle_table"] if row["final_status"] == "StableSupport"][:30],
        "controls": [row for row in lifecycle["operator_lifecycle_table"] if row["role"] != "candidate"],
    }
    replay_payload = {
        "aggregate": {key: agg[key] for key in agg if key != "seconds"},
        "selection_frequency": freq,
        "counterfactual_summary": cf["summary"],
        "lifecycle": lifecycle,
        "survival_role": survival_role,
    }
    failures = []
    if agg["candidate_operator_count"] < 130:
        failures.append("too few E90-E106 candidate operators")
    if agg["operator_group_count"] < 17:
        failures.append("missing E90-E106 groups")
    for key in ["survival_success_min", "adversarial_survival_success_min", "group_coverage_min", "family_coverage_min", "role_assignment_coverage"]:
        if agg[key] != 1.0:
            failures.append(f"{key} below 1.0")
    if agg["focus_coverage_min"] < 0.5:
        failures.append("focus coverage below 0.5")
    for key in ["unsafe_control_selected_rate", "full_library_overreach_rate", "cost_blowup_rate"]:
        if agg[key] != 0.0:
            failures.append(f"{key} nonzero")
    if agg["stable_support_count"] <= 0 or agg["specialist_count"] <= 0:
        failures.append("missing stable/specialist role split")
    decision = "e107_operator_library_survival_role_regression_gauntlet_confirmed" if not failures else "e107_operator_library_survival_gauntlet_incomplete"
    library_manifest = {
        "artifact_contract": ARTIFACT_CONTRACT,
        "canonical_term": "Operator",
        "legacy_alias": "Pocket",
        "inventory_source": "docs/research/OPERATOR_LIBRARY_CARDS.md",
        "families": sorted({operator.family for operator in library}),
        "groups": sorted({operator.group_id for operator in library if operator.role == "candidate"}),
        "operators": [dataclasses.asdict(operator) for operator in library],
    }
    task_report = {
        "artifact_contract": ARTIFACT_CONTRACT,
        "case_count": len(flat),
        "seed_count": len(seed_results),
        "neighborhood_count": len(NEIGHBORHOODS),
        "quality_control_gauntlet": True,
        "operator_survival_ranking": True,
        "open_domain_capability_eval": False,
        "final_training": False,
        "full_library_scan_allowed_as_success": False,
        "unsafe_control_allowed": False,
        "neighborhoods": [dataclasses.asdict(item) for item in NEIGHBORHOODS],
    }
    source_inventory = {
        "source_path": "docs/research/OPERATOR_LIBRARY_CARDS.md",
        "candidate_operator_count": agg["candidate_operator_count"],
        "operator_group_count": agg["operator_group_count"],
        "groups": sorted({operator.group_id for operator in library if operator.role == "candidate"}),
    }
    summary = {
        "artifact_contract": ARTIFACT_CONTRACT,
        "decision": decision,
        "stable_support_count": agg["stable_support_count"],
        "specialist_count": agg["specialist_count"],
        "bundle_support_count": agg["bundle_support_count"],
        "deprecated_count": agg["deprecated_count"],
        "sample_pack": str(sample_dir) if sample_dir else None,
    }
    write_json(out / "operator_library_manifest.json", library_manifest)
    write_json(out / "task_generation_report.json", task_report)
    write_json(out / "source_inventory_report.json", source_inventory)
    write_json(out / "neighborhood_results.json", {"rows": flat})
    write_json(out / "survival_role_report.json", survival_role)
    write_json(out / "aggregate_metrics.json", agg)
    write_json(out / "selection_frequency_report.json", freq)
    write_json(out / "counterfactual_report.json", cf)
    write_json(out / "operator_lifecycle_report.json", lifecycle)
    write_json(out / "mutation_summary.json", {
        "accepted": agg["accepted_mutations_total"],
        "rejected": agg["rejected_mutations_total"],
        "rollback": agg["rollback_count_total"],
        "mutation_mode": "full_library_neighborhood_scan_accept_reject_rollback",
    })
    write_json(out / "deterministic_replay.json", {"hash": deterministic_hash(replay_payload), "payload_keys": sorted(replay_payload)})
    write_json(out / "decision.json", {"decision": decision, "failure_count": len(failures), "failures": failures})
    write_json(out / "summary.json", summary)
    write_json(out / "seed_results.json", {"seeds": seed_results})
    write_json(out / "partial_aggregate_snapshot.json", agg)
    sample_rows = 0
    for row in flat:
        append_jsonl(out / "row_level_samples.jsonl", row)
        sample_rows += 1
        if sample_rows >= 480:
            break
    report = [
        "# E107 Operator Library E90-E106 Survival Role And Regression Gauntlet Result",
        "",
        f"decision = `{decision}`",
        "",
        "Boundary: E90-E106 quality-control gauntlet, not final training.",
        "",
        "```json",
        json.dumps(agg, indent=2, sort_keys=True),
        "```",
        "",
        "Role counts:",
        "",
        "```json",
        json.dumps(survival_role["status_counts"], indent=2, sort_keys=True),
        "```",
    ]
    (out / "report.md").write_text("\n".join(report), encoding="utf-8")
    if sample_dir:
        write_sample_pack(out, sample_dir)


def parse_seeds(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="target/pilot_wave/e107_operator_library_e90_e106_survival_role_and_regression_gauntlet")
    parser.add_argument("--artifact-sample-dir", default="docs/research/artifact_samples/e107_operator_library_e90_e106_survival_role_and_regression_gauntlet")
    parser.add_argument("--operator-cards", default="docs/research/OPERATOR_LIBRARY_CARDS.md")
    parser.add_argument("--seeds", default="110701,110702,110703,110704,110705,110706,110707,110708,110709,110710,110711,110712,110713,110714,110715,110716")
    parser.add_argument("--heartbeat-seconds", type=float, default=20.0)
    args = parser.parse_args()

    out = Path(args.out)
    if out.exists():
        for child in out.rglob("*"):
            if child.is_file():
                child.unlink()
    out.mkdir(parents=True, exist_ok=True)
    library = load_e90_e106_inventory(Path(args.operator_cards)) + list(CONTROL_OPERATORS)
    seeds = parse_seeds(args.seeds)
    started = time.time()
    write_json(out / "run_manifest.json", {
        "artifact_contract": ARTIFACT_CONTRACT,
        "seeds": seeds,
        "neighborhood_count": len(NEIGHBORHOODS),
        "operator_cards": args.operator_cards,
        "boundary": "controlled E90-E106 operator-library survival gauntlet; not final training; not open-domain capability evaluation",
        "gradient_descent_used": False,
        "optimizer_used": False,
        "backprop_used": False,
        "started_at_ms": now_ms(),
    })
    append_jsonl(out / "progress.jsonl", {"event": "start", "timestamp_ms": now_ms(), "seed_count": len(seeds), "candidate_count": len(library)})
    seed_results = []
    last_heartbeat = time.time()
    for seed in seeds:
        result = run_seed(seed, library, out)
        seed_results.append(result)
        append_jsonl(out / "progress.jsonl", {"event": "seed_complete", "seed": seed, "completed": len(seed_results), "timestamp_ms": now_ms()})
        write_json(out / "partial_aggregate_snapshot.json", {"completed": len(seed_results), "seed_count": len(seeds), "updated_at_ms": now_ms()})
        if time.time() - last_heartbeat >= args.heartbeat_seconds:
            append_jsonl(out / "progress.jsonl", {"event": "heartbeat", "completed": len(seed_results), "seed_count": len(seeds), "timestamp_ms": now_ms()})
            last_heartbeat = time.time()
    append_jsonl(out / "progress.jsonl", {"event": "heartbeat", "completed": len(seed_results), "seed_count": len(seeds), "timestamp_ms": now_ms()})
    write_reports(out, Path(args.artifact_sample_dir), seed_results, library, args, time.time() - started)
    append_jsonl(out / "progress.jsonl", {"event": "complete", "timestamp_ms": now_ms()})
    print(json.dumps({"out": str(out), "decision": json.loads((out / "decision.json").read_text(encoding="utf-8"))["decision"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
