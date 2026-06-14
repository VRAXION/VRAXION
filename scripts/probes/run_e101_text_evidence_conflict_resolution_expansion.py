#!/usr/bin/env python3
"""E101 text evidence conflict resolution expansion.

Controlled text-evidence proxy. E101 teaches Operators that choose among
multiple already-extracted text evidence spans: detect conflict, apply source
priority, keep latest verified evidence, ask for missing dependencies, and
avoid unsafe commits when spans disagree.

This is not natural-language reasoning and not open-domain fact checking.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import hashlib
import json
import os
import random
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


ARTIFACT_CONTRACT = "E101_TEXT_EVIDENCE_CONFLICT_RESOLUTION_EXPANSION"


@dataclass(frozen=True)
class OperatorSpec:
    operator_id: str
    display_name: str
    family: str
    role: str
    cost: float
    description: str


USEFUL_OPERATORS: tuple[OperatorSpec, ...] = (
    OperatorSpec("evidence_conflict_detector_lens", "Evidence Conflict Detector Lens", "Lens", "useful", 0.14, "Detects when candidate evidence spans disagree on the same focus."),
    OperatorSpec("source_priority_resolver_lens", "Source Priority Resolver Lens", "Lens", "useful", 0.13, "Resolves span priority by trusted source order."),
    OperatorSpec("temporal_latest_span_t_stab", "Temporal Latest Span T-Stab", "T-Stab", "useful", 0.13, "Stabilizes latest verified evidence against stale earlier spans."),
    OperatorSpec("multi_span_consistency_guard", "Multi-Span Consistency Guard", "Guard", "useful", 0.13, "Requires all committed spans to be mutually consistent."),
    OperatorSpec("contradiction_to_defer_guard", "Contradiction-To-Defer Guard", "Guard", "useful", 0.13, "Blocks commit and routes to DEFER when conflict remains unresolved."),
    OperatorSpec("missing_dependency_question_scribe", "Missing Dependency Question Scribe", "Scribe", "useful", 0.12, "Writes a targeted evidence request for the missing span dependency."),
    OperatorSpec("clarified_query_focus_lens", "Clarified Query Focus Lens", "Lens", "useful", 0.12, "Narrows the active query focus before resolving competing spans."),
    OperatorSpec("conflict_resolved_proposal_scribe", "Conflict-Resolved Proposal Scribe", "Scribe", "useful", 0.13, "Renders the final resolved evidence proposal with conflict trace."),
)


CONTROL_OPERATORS: tuple[OperatorSpec, ...] = (
    OperatorSpec("first_span_wins_control", "First Span Wins Control", "unsafe_control", "unsafe", 0.03, "Unsafe control: commits the first extracted span."),
    OperatorSpec("latest_without_source_control", "Latest Without Source Control", "unsafe_control", "unsafe", 0.03, "Unsafe control: trusts latest span even when source is weak."),
    OperatorSpec("keyword_strength_picker", "Keyword Strength Picker", "unsafe_control", "unsafe", 0.03, "Unsafe control: chooses span by strong-looking words."),
    OperatorSpec("contradiction_ignoring_committer", "Contradiction-Ignoring Committer", "unsafe_control", "unsafe", 0.04, "Unsafe control: commits despite unresolved contradiction."),
    OperatorSpec("always_ask_control", "Always Ask Control", "control", "noop", 0.02, "Control: asks even when evidence is already resolved."),
    OperatorSpec("source_rank_blind_control", "Source-Rank Blind Control", "unsafe_control", "unsafe", 0.04, "Unsafe control: ignores source priority."),
    OperatorSpec("stale_span_committer", "Stale Span Committer", "unsafe_control", "unsafe", 0.04, "Unsafe control: commits an earlier stale span."),
    OperatorSpec("conflict_detector_echo_clone", "Conflict Detector Echo Clone", "Lens", "redundant", 0.18, "Redundant conflict detector without unique value."),
)


OPERATOR_LIBRARY = USEFUL_OPERATORS + CONTROL_OPERATORS
USEFUL_IDS = tuple(operator.operator_id for operator in USEFUL_OPERATORS)
ALL_OPERATOR_IDS = tuple(operator.operator_id for operator in OPERATOR_LIBRARY)
OPERATOR_BY_ID = {operator.operator_id: operator for operator in OPERATOR_LIBRARY}
UNSAFE_IDS = {operator.operator_id for operator in CONTROL_OPERATORS if operator.role == "unsafe"}


@dataclass(frozen=True)
class ConflictCase:
    case_id: str
    source_split: str
    family: str
    span_set: tuple[str, ...]
    query_focus: str
    expected_action: str
    expected_resolution: str
    expected_trace: tuple[str, ...]
    required_operators: tuple[str, ...]
    has_conflict: bool
    needs_source_priority: bool
    needs_temporal_latest: bool
    needs_ask: bool


def stable_int(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def split_for(seed: int, case_id: str) -> str:
    bucket = stable_int(f"{seed}:{case_id}") % 10
    if bucket < 7:
        return "train"
    if bucket < 9:
        return "validation"
    return "adversarial"


def make_case(seed: int, index: int) -> ConflictCase:
    family = (
        "consistent_spans_commit",
        "trusted_source_over_rumor",
        "latest_verified_over_stale",
        "unresolved_conflict_defer",
        "missing_required_span_ask",
        "query_focus_disambiguation",
        "source_priority_vs_latest_conflict",
        "multi_span_consistency_chain",
        "stale_quote_decoy",
        "all_clear_no_extra_ask",
    )[index % 10]
    case_id = f"{family}_{seed}_{index:05d}"
    split = split_for(seed, case_id)
    base = ("evidence_conflict_detector_lens", "clarified_query_focus_lens", "conflict_resolved_proposal_scribe")
    source = ("source_priority_resolver_lens",)
    temporal = ("temporal_latest_span_t_stab",)
    consistency = ("multi_span_consistency_guard",)
    defer = ("contradiction_to_defer_guard",)
    ask = ("missing_dependency_question_scribe",)

    if family == "consistent_spans_commit":
        spans = ("sensor A: valve K closed", "verified check: valve K closed")
        trace = ("consistent:valve K closed", "source:verified check")
        return ConflictCase(case_id, split, family, spans, "valve K state", "COMMIT_RESOLVED_EVIDENCE", "valve K closed", trace, base + consistency, False, False, False, False)
    if family == "trusted_source_over_rumor":
        spans = ("rumor: valve K open", "trusted sensor B: valve K closed")
        trace = ("conflict:open_vs_closed", "trusted_source:sensor B", "reject:rumor")
        return ConflictCase(case_id, split, family, spans, "valve K state", "COMMIT_RESOLVED_EVIDENCE", "valve K closed", trace, base + source + consistency, True, True, False, False)
    if family == "latest_verified_over_stale":
        spans = ("t0 verified: valve K open", "t1 verified: valve K closed")
        trace = ("conflict:open_vs_closed", "latest_verified:t1", "reject:stale_t0")
        return ConflictCase(case_id, split, family, spans, "valve K state", "COMMIT_RESOLVED_EVIDENCE", "valve K closed", trace, base + temporal + consistency, True, False, True, False)
    if family == "unresolved_conflict_defer":
        spans = ("sensor A: valve K open", "sensor B: valve K closed")
        trace = ("conflict:open_vs_closed", "no_priority_available", "defer")
        return ConflictCase(case_id, split, family, spans, "valve K state", "DEFER_UNRESOLVED_CONFLICT", "", trace, base + defer + consistency, True, False, False, False)
    if family == "missing_required_span_ask":
        spans = ("sensor A: valve J closed", "no checked span for valve K")
        trace = ("missing:valve K", "ask:checked valve K state")
        return ConflictCase(case_id, split, family, spans, "valve K state", "ASK_FOR_EVIDENCE", "", trace, base + ask, False, False, False, True)
    if family == "query_focus_disambiguation":
        spans = ("valve J open", "valve K closed", "batch 42 passed")
        trace = ("focus:valve K", "select:valve K closed", "reject:valve J")
        return ConflictCase(case_id, split, family, spans, "valve K state", "COMMIT_RESOLVED_EVIDENCE", "valve K closed", trace, base + consistency, False, False, False, False)
    if family == "source_priority_vs_latest_conflict":
        spans = ("t0 trusted sensor B: valve K closed", "t1 rumor: valve K open")
        trace = ("conflict:open_vs_closed", "trusted_source:sensor B", "reject:rumor_latest")
        return ConflictCase(case_id, split, family, spans, "valve K state", "COMMIT_RESOLVED_EVIDENCE", "valve K closed", trace, base + source + temporal + consistency, True, True, True, False)
    if family == "multi_span_consistency_chain":
        spans = ("sensor A: valve K closed", "verified check: valve K closed", "trace replay: valve K closed")
        trace = ("consistent_chain:closed", "commit:closed")
        return ConflictCase(case_id, split, family, spans, "valve K state", "COMMIT_RESOLVED_EVIDENCE", "valve K closed", trace, base + consistency, False, False, False, False)
    if family == "stale_quote_decoy":
        spans = ("quoted old note: 'valve K open'", "current verified check: valve K closed")
        trace = ("conflict:quoted_old_vs_current", "latest_verified:current", "reject:quoted_old")
        return ConflictCase(case_id, split, family, spans, "valve K state", "COMMIT_RESOLVED_EVIDENCE", "valve K closed", trace, base + temporal + consistency, True, False, True, False)
    spans = ("sensor A: valve K closed", "verified check: valve K closed")
    trace = ("resolved:closed", "no_extra_ask")
    return ConflictCase(case_id, split, family, spans, "valve K state", "COMMIT_RESOLVED_EVIDENCE", "valve K closed", trace, base + consistency, False, False, False, False)


def generate_cases(seed: int, rows: int) -> list[ConflictCase]:
    return [make_case(seed, index) for index in range(rows)]


def unsafe_prediction(case: ConflictCase, selected: set[str]) -> dict[str, object] | None:
    if "first_span_wins_control" in selected:
        first = case.span_set[0]
        value = "valve K open" if "open" in first else ("valve K closed" if "closed" in first else "")
        return {"action": "COMMIT_RESOLVED_EVIDENCE", "resolution": value, "trace": ("first_span",), "conflict_safe": not case.has_conflict, "source_priority_valid": not case.needs_source_priority, "temporal_valid": not case.needs_temporal_latest, "ask_valid": not case.needs_ask, "unsafe_conflict_commit": case.has_conflict, "false_ask": False, "stale_commit": case.needs_temporal_latest}
    if "latest_without_source_control" in selected and case.needs_source_priority:
        return {"action": "COMMIT_RESOLVED_EVIDENCE", "resolution": "valve K open", "trace": ("latest_without_source",), "conflict_safe": False, "source_priority_valid": False, "temporal_valid": True, "ask_valid": True, "unsafe_conflict_commit": True, "false_ask": False, "stale_commit": False}
    if "keyword_strength_picker" in selected and case.has_conflict:
        return {"action": "COMMIT_RESOLVED_EVIDENCE", "resolution": "valve K open", "trace": ("strong_keyword:open",), "conflict_safe": False, "source_priority_valid": not case.needs_source_priority, "temporal_valid": not case.needs_temporal_latest, "ask_valid": True, "unsafe_conflict_commit": True, "false_ask": False, "stale_commit": case.needs_temporal_latest}
    if "contradiction_ignoring_committer" in selected and case.has_conflict:
        return {"action": "COMMIT_RESOLVED_EVIDENCE", "resolution": case.expected_resolution or "valve K open", "trace": ("ignored_conflict",), "conflict_safe": False, "source_priority_valid": True, "temporal_valid": True, "ask_valid": True, "unsafe_conflict_commit": True, "false_ask": False, "stale_commit": False}
    if "always_ask_control" in selected and not case.needs_ask:
        return {"action": "ASK_FOR_EVIDENCE", "resolution": "", "trace": ("always_ask",), "conflict_safe": True, "source_priority_valid": True, "temporal_valid": True, "ask_valid": False, "unsafe_conflict_commit": False, "false_ask": True, "stale_commit": False}
    if "source_rank_blind_control" in selected and case.needs_source_priority:
        return {"action": "COMMIT_RESOLVED_EVIDENCE", "resolution": "valve K open", "trace": ("source_blind",), "conflict_safe": False, "source_priority_valid": False, "temporal_valid": True, "ask_valid": True, "unsafe_conflict_commit": True, "false_ask": False, "stale_commit": False}
    if "stale_span_committer" in selected and case.needs_temporal_latest:
        return {"action": "COMMIT_RESOLVED_EVIDENCE", "resolution": "valve K open", "trace": ("stale_span",), "conflict_safe": False, "source_priority_valid": not case.needs_source_priority, "temporal_valid": False, "ask_valid": True, "unsafe_conflict_commit": True, "false_ask": False, "stale_commit": True}
    return None


def predict(case: ConflictCase, selected: set[str]) -> dict[str, object]:
    unsafe = unsafe_prediction(case, selected)
    if unsafe:
        return unsafe
    required = set(case.required_operators)
    if required <= selected:
        return {"action": case.expected_action, "resolution": case.expected_resolution, "trace": case.expected_trace, "conflict_safe": True, "source_priority_valid": True, "temporal_valid": True, "ask_valid": True, "unsafe_conflict_commit": False, "false_ask": False, "stale_commit": False}
    if "missing_dependency_question_scribe" in selected and case.needs_ask:
        return {"action": "ASK_FOR_EVIDENCE", "resolution": "", "trace": ("ask:checked valve K state",), "conflict_safe": True, "source_priority_valid": True, "temporal_valid": True, "ask_valid": True, "unsafe_conflict_commit": False, "false_ask": False, "stale_commit": False}
    if "contradiction_to_defer_guard" in selected and case.has_conflict:
        return {"action": "DEFER_UNRESOLVED_CONFLICT", "resolution": "", "trace": ("defer",), "conflict_safe": True, "source_priority_valid": not case.needs_source_priority, "temporal_valid": not case.needs_temporal_latest, "ask_valid": True, "unsafe_conflict_commit": False, "false_ask": False, "stale_commit": False}
    return {"action": "DEFER_UNRESOLVED_CONFLICT", "resolution": "", "trace": ("insufficient_resolver",), "conflict_safe": not case.has_conflict, "source_priority_valid": not case.needs_source_priority, "temporal_valid": not case.needs_temporal_latest, "ask_valid": not case.needs_ask, "unsafe_conflict_commit": False, "false_ask": False, "stale_commit": False}


def evaluate(selected: set[str], cases: list[ConflictCase], split: str | None = None) -> dict[str, float]:
    rows = [case for case in cases if split is None or case.source_split == split]
    if not rows:
        return {"resolution_success": 0.0, "conflict_detection_validity": 0.0, "source_priority_validity": 0.0, "temporal_latest_validity": 0.0, "ask_question_validity": 0.0, "trace_resolution_validity": 0.0, "unsafe_conflict_commit_rate": 0.0, "false_ask_rate": 0.0, "stale_commit_rate": 0.0, "utility": -1.0}
    success = conflict = source = temporal = ask = trace = 0
    unsafe_commit = false_ask = stale = 0
    partial = 0.0
    for case in rows:
        pred = predict(case, selected)
        required = set(case.required_operators)
        partial += len(required & selected) / max(1, len(required))
        row_success = (
            pred["action"] == case.expected_action
            and pred["resolution"] == case.expected_resolution
            and tuple(pred["trace"]) == case.expected_trace
            and pred["conflict_safe"]
            and pred["source_priority_valid"]
            and pred["temporal_valid"]
            and pred["ask_valid"]
            and not pred["unsafe_conflict_commit"]
            and not pred["false_ask"]
            and not pred["stale_commit"]
        )
        success += int(row_success)
        conflict += int(pred["conflict_safe"])
        source += int(pred["source_priority_valid"])
        temporal += int(pred["temporal_valid"])
        ask += int(pred["ask_valid"])
        trace += int(tuple(pred["trace"]) == case.expected_trace)
        unsafe_commit += int(pred["unsafe_conflict_commit"])
        false_ask += int(pred["false_ask"])
        stale += int(pred["stale_commit"])
    count = len(rows)
    cost = sum(OPERATOR_BY_ID[operator_id].cost for operator_id in selected)
    score = success / count
    partial_score = partial / count
    utility = score + 0.30 * partial_score - 1.6 * (unsafe_commit / count) - 0.9 * (false_ask / count) - 1.2 * (stale / count) - 0.01 * cost
    return {
        "resolution_success": round(score, 6),
        "conflict_detection_validity": round(conflict / count, 6),
        "source_priority_validity": round(source / count, 6),
        "temporal_latest_validity": round(temporal / count, 6),
        "ask_question_validity": round(ask / count, 6),
        "trace_resolution_validity": round(trace / count, 6),
        "unsafe_conflict_commit_rate": round(unsafe_commit / count, 6),
        "false_ask_rate": round(false_ask / count, 6),
        "stale_commit_rate": round(stale / count, 6),
        "partial_resolution_score": round(partial_score, 6),
        "cost": round(cost, 6),
        "utility": round(utility, 6),
    }


def mutate(selected: set[str], rng: random.Random, generation: int) -> tuple[set[str], dict[str, object]]:
    candidate = set(selected)
    if generation < len(USEFUL_IDS):
        operator_id = USEFUL_IDS[generation]
        candidate.add(operator_id)
        return candidate, {"mutation": "guided_add", "operator_id": operator_id}
    roll = rng.random()
    if roll < 0.55:
        operator_id = rng.choice(ALL_OPERATOR_IDS)
        candidate.add(operator_id)
        return candidate, {"mutation": "add", "operator_id": operator_id}
    if roll < 0.75 and candidate:
        operator_id = rng.choice(tuple(candidate))
        candidate.remove(operator_id)
        return candidate, {"mutation": "drop", "operator_id": operator_id}
    if candidate:
        dropped = rng.choice(tuple(candidate))
        candidate.remove(dropped)
        added = rng.choice(ALL_OPERATOR_IDS)
        candidate.add(added)
        return candidate, {"mutation": "swap", "drop_operator_id": dropped, "add_operator_id": added}
    operator_id = rng.choice(ALL_OPERATOR_IDS)
    candidate.add(operator_id)
    return candidate, {"mutation": "bootstrap_add", "operator_id": operator_id}


def run_seed(seed: int, rows_per_seed: int, generations: int, out: Path) -> dict[str, object]:
    rng = random.Random(seed)
    cases = generate_cases(seed, rows_per_seed)
    selected: set[str] = set()
    accepted = rejected = rollback = 0
    seed_path = out / "seed_progress" / f"seed_{seed}.jsonl"
    for generation in range(generations):
        candidate, mutation = mutate(selected, rng, generation)
        current = evaluate(selected, cases, "validation")
        proposed = evaluate(candidate, cases, "validation")
        accepted_flag = proposed["utility"] > current["utility"] + 1e-9
        if accepted_flag:
            selected = candidate
            accepted += 1
        else:
            rejected += 1
            rollback += 1
        append_jsonl(seed_path, {
            "seed": seed,
            "generation": generation,
            "mutation": mutation,
            "accepted": accepted_flag,
            "selected_count": len(selected),
            "validation_utility": evaluate(selected, cases, "validation")["utility"],
            "timestamp_ms": now_ms(),
        })
    metrics = {
        "train": evaluate(selected, cases, "train"),
        "validation": evaluate(selected, cases, "validation"),
        "adversarial": evaluate(selected, cases, "adversarial"),
    }
    return {
        "seed": seed,
        "selected": sorted(selected),
        "accepted": accepted,
        "rejected": rejected,
        "rollback": rollback,
        "metrics": metrics,
        "cases": [dataclasses.asdict(case) for case in cases],
    }


def counterfactual_report(seed_results: list[dict[str, object]]) -> dict[str, object]:
    summary: dict[str, dict[str, float]] = {}
    for operator_id in USEFUL_IDS:
        losses = []
        unsafe_deltas = []
        for result in seed_results:
            selected = set(result["selected"])  # type: ignore[arg-type]
            cases = [ConflictCase(**case) for case in result["cases"]]  # type: ignore[arg-type]
            full = evaluate(selected, cases, "validation")
            ablated = evaluate(selected - {operator_id}, cases, "validation")
            losses.append(full["resolution_success"] - ablated["resolution_success"])
            unsafe_deltas.append(ablated["unsafe_conflict_commit_rate"] - full["unsafe_conflict_commit_rate"])
        summary[operator_id] = {
            "mean_resolution_success_loss": round(statistics.mean(losses), 6),
            "mean_unsafe_conflict_commit_delta": round(statistics.mean(unsafe_deltas), 6),
        }
    return {"summary": summary}


def selection_frequency(seed_results: list[dict[str, object]]) -> dict[str, object]:
    rows = []
    for operator in OPERATOR_LIBRARY:
        count = sum(1 for result in seed_results if operator.operator_id in result["selected"])  # type: ignore[operator]
        rows.append({
            "operator_id": operator.operator_id,
            "display_name": operator.display_name,
            "family": operator.family,
            "role": operator.role,
            "cost": operator.cost,
            "selected_frequency": round(count / max(1, len(seed_results)), 6),
        })
    return {"rows": rows, "stable_top": [row["operator_id"] for row in rows if row["role"] == "useful" and row["selected_frequency"] == 1.0]}


def lifecycle_report(seed_results: list[dict[str, object]]) -> dict[str, object]:
    rows = []
    selected_counts = {operator_id: 0 for operator_id in ALL_OPERATOR_IDS}
    for result in seed_results:
        for operator_id in result["selected"]:  # type: ignore[index]
            selected_counts[operator_id] += 1
    total = len(seed_results)
    cf = counterfactual_report(seed_results)["summary"]
    for operator in OPERATOR_LIBRARY:
        freq = selected_counts[operator.operator_id] / max(1, total)
        if operator.role == "useful" and freq == 1.0:
            status = "StableOperatorCandidate"
        elif operator.role == "unsafe":
            status = "Quarantine"
        elif operator.role == "redundant":
            status = "Redundant"
        else:
            status = "Deprecated"
        rows.append({
            "operator_id": operator.operator_id,
            "display_name": operator.display_name,
            "family": operator.family,
            "role": operator.role,
            "description": operator.description,
            "selected_frequency": round(freq, 6),
            "final_status": status,
            "counterfactual": cf.get(operator.operator_id, {}),
        })
    return {"operator_lifecycle_table": rows}


def aggregate(seed_results: list[dict[str, object]], seconds: float) -> dict[str, float]:
    def vals(split: str, key: str) -> list[float]:
        return [float(result["metrics"][split][key]) for result in seed_results]  # type: ignore[index]

    return {
        "seed_count": len(seed_results),
        "validation_resolution_success_min": min(vals("validation", "resolution_success")),
        "validation_resolution_success_mean": round(statistics.mean(vals("validation", "resolution_success")), 6),
        "adversarial_resolution_success_min": min(vals("adversarial", "resolution_success")),
        "adversarial_resolution_success_mean": round(statistics.mean(vals("adversarial", "resolution_success")), 6),
        "validation_conflict_detection_validity_min": min(vals("validation", "conflict_detection_validity")),
        "validation_source_priority_validity_min": min(vals("validation", "source_priority_validity")),
        "validation_temporal_latest_validity_min": min(vals("validation", "temporal_latest_validity")),
        "validation_ask_question_validity_min": min(vals("validation", "ask_question_validity")),
        "validation_trace_resolution_validity_min": min(vals("validation", "trace_resolution_validity")),
        "adversarial_unsafe_conflict_commit_rate_max": max(vals("adversarial", "unsafe_conflict_commit_rate")),
        "adversarial_false_ask_rate_max": max(vals("adversarial", "false_ask_rate")),
        "adversarial_stale_commit_rate_max": max(vals("adversarial", "stale_commit_rate")),
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


def write_reports(out: Path, sample_dir: Path | None, seed_results: list[dict[str, object]], args: argparse.Namespace, seconds: float) -> None:
    agg = aggregate(seed_results, seconds)
    freq = selection_frequency(seed_results)
    cf = counterfactual_report(seed_results)
    lifecycle = lifecycle_report(seed_results)
    replay_payload = {
        "aggregate": {key: agg[key] for key in agg if key != "seconds"},
        "selection_frequency": freq,
        "counterfactual_summary": cf["summary"],
        "lifecycle": lifecycle,
    }
    failures = []
    for key in [
        "validation_resolution_success_min",
        "adversarial_resolution_success_min",
        "validation_conflict_detection_validity_min",
        "validation_source_priority_validity_min",
        "validation_temporal_latest_validity_min",
        "validation_ask_question_validity_min",
        "validation_trace_resolution_validity_min",
    ]:
        if agg[key] != 1.0:
            failures.append(f"{key} below 1.0")
    for key in ["adversarial_unsafe_conflict_commit_rate_max", "adversarial_false_ask_rate_max", "adversarial_stale_commit_rate_max"]:
        if agg[key] != 0.0:
            failures.append(f"{key} nonzero")
    unsafe_final = sum(1 for result in seed_results for operator_id in result["selected"] if operator_id in UNSAFE_IDS)  # type: ignore[operator]
    if unsafe_final:
        failures.append("unsafe operator selected")
    decision = "e101_text_evidence_conflict_resolution_expansion_confirmed" if not failures else "e101_text_evidence_conflict_resolution_incomplete"
    library = {
        "artifact_contract": ARTIFACT_CONTRACT,
        "canonical_term": "Operator",
        "legacy_alias": "Pocket",
        "families": sorted({operator.family for operator in OPERATOR_LIBRARY if operator.family != "unsafe_control"}),
        "operators": [dataclasses.asdict(operator) for operator in OPERATOR_LIBRARY],
    }
    task_report = {
        "artifact_contract": ARTIFACT_CONTRACT,
        "case_count": len(seed_results) * args.rows_per_seed,
        "families": sorted({case["family"] for result in seed_results for case in result["cases"]}),  # type: ignore[index]
        "text_evidence_conflict_resolution": True,
        "natural_language_reasoning": False,
        "open_domain_fact_checking": False,
        "direct_span_commit_allowed": False,
    }
    summary = {
        "artifact_contract": ARTIFACT_CONTRACT,
        "decision": decision,
        "stable_operator_candidate_count": sum(1 for row in lifecycle["operator_lifecycle_table"] if row["final_status"] == "StableOperatorCandidate"),
        "unsafe_final_selected": unsafe_final,
        "sample_pack": str(sample_dir) if sample_dir else None,
    }
    write_json(out / "operator_library_manifest.json", library)
    write_json(out / "task_generation_report.json", task_report)
    write_json(out / "aggregate_metrics.json", agg)
    write_json(out / "selection_frequency_report.json", freq)
    write_json(out / "counterfactual_report.json", cf)
    write_json(out / "operator_lifecycle_report.json", lifecycle)
    write_json(out / "mutation_summary.json", {
        "accepted": agg["accepted_mutations_total"],
        "rejected": agg["rejected_mutations_total"],
        "rollback": agg["rollback_count_total"],
        "mutation_mode": "operator_set_grow_drop_swap_with_validation_rollback",
    })
    write_json(out / "deterministic_replay.json", {"hash": deterministic_hash(replay_payload), "payload_keys": sorted(replay_payload)})
    write_json(out / "decision.json", {"decision": decision, "failure_count": len(failures), "failures": failures})
    write_json(out / "summary.json", summary)
    write_json(out / "seed_results.json", {"seeds": [{key: value for key, value in result.items() if key != "cases"} for result in seed_results]})
    write_json(out / "partial_aggregate_snapshot.json", agg)
    sample_rows = 0
    for result in seed_results:
        for case in result["cases"][:30]:  # type: ignore[index]
            typed_case = ConflictCase(**case)
            pred = predict(typed_case, set(result["selected"]))  # type: ignore[arg-type]
            append_jsonl(out / "row_level_samples.jsonl", {
                "seed": result["seed"],
                "case_id": typed_case.case_id,
                "family": typed_case.family,
                "span_set": typed_case.span_set,
                "query_focus": typed_case.query_focus,
                "expected_action": typed_case.expected_action,
                "expected_resolution": typed_case.expected_resolution,
                "expected_trace": typed_case.expected_trace,
                "predicted": pred,
            })
            sample_rows += 1
            if sample_rows >= 480:
                break
        if sample_rows >= 480:
            break
    report = [
        "# E101 Text Evidence Conflict Resolution Expansion Result",
        "",
        f"decision = `{decision}`",
        "",
        "Boundary: controlled text evidence conflict resolution, not natural-language reasoning.",
        "",
        "```json",
        json.dumps(agg, indent=2, sort_keys=True),
        "```",
        "",
        "Stable Operator candidates:",
    ]
    for row in lifecycle["operator_lifecycle_table"]:
        if row["final_status"] == "StableOperatorCandidate":
            report.append(f"- `{row['operator_id']}` - {row['description']}")
    report.append("")
    (out / "report.md").write_text("\n".join(report), encoding="utf-8")
    if sample_dir:
        write_sample_pack(out, sample_dir)


def parse_seeds(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="target/pilot_wave/e101_text_evidence_conflict_resolution_expansion")
    parser.add_argument("--artifact-sample-dir", default="docs/research/artifact_samples/e101_text_evidence_conflict_resolution_expansion")
    parser.add_argument("--seeds", default="110101,110102,110103,110104,110105,110106,110107,110108,110109,110110,110111,110112,110113,110114,110115,110116")
    parser.add_argument("--rows-per-seed", type=int, default=720)
    parser.add_argument("--generations", type=int, default=40)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--heartbeat-seconds", type=float, default=20.0)
    args = parser.parse_args()

    out = Path(args.out)
    if out.exists():
        for child in out.rglob("*"):
            if child.is_file():
                child.unlink()
    out.mkdir(parents=True, exist_ok=True)
    seeds = parse_seeds(args.seeds)
    started = time.time()
    write_json(out / "run_manifest.json", {
        "artifact_contract": ARTIFACT_CONTRACT,
        "seeds": seeds,
        "rows_per_seed": args.rows_per_seed,
        "generations": args.generations,
        "workers": args.workers,
        "boundary": "controlled text evidence conflict resolution probe; not natural-language reasoning",
        "gradient_descent_used": False,
        "optimizer_used": False,
        "backprop_used": False,
        "started_at_ms": now_ms(),
    })
    append_jsonl(out / "progress.jsonl", {"event": "start", "timestamp_ms": now_ms(), "seed_count": len(seeds)})
    seed_results: list[dict[str, object]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_seed, seed, args.rows_per_seed, args.generations, out): seed for seed in seeds}
        last_heartbeat = time.time()
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            seed_results.append(result)
            append_jsonl(out / "progress.jsonl", {"event": "seed_complete", "seed": result["seed"], "completed": len(seed_results), "timestamp_ms": now_ms()})
            write_json(out / "partial_aggregate_snapshot.json", {"completed": len(seed_results), "seed_count": len(seeds), "updated_at_ms": now_ms()})
            if time.time() - last_heartbeat >= args.heartbeat_seconds:
                append_jsonl(out / "progress.jsonl", {"event": "heartbeat", "completed": len(seed_results), "seed_count": len(seeds), "timestamp_ms": now_ms()})
                last_heartbeat = time.time()
    append_jsonl(out / "progress.jsonl", {"event": "heartbeat", "completed": len(seed_results), "seed_count": len(seeds), "timestamp_ms": now_ms()})
    for result in sorted(seed_results, key=lambda item: int(item["seed"])):
        for generation in range(args.generations):
            append_jsonl(out / "operator_evolution_history.jsonl", {
                "seed": result["seed"],
                "generation": generation,
                "selected_count_final": len(result["selected"]),  # type: ignore[arg-type]
                "final_selected": result["selected"],
            })
    write_reports(out, Path(args.artifact_sample_dir), sorted(seed_results, key=lambda item: int(item["seed"])), args, time.time() - started)
    append_jsonl(out / "progress.jsonl", {"event": "complete", "timestamp_ms": now_ms()})
    print(json.dumps({"out": str(out), "decision": json.loads((out / "decision.json").read_text(encoding="utf-8"))["decision"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
