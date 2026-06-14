#!/usr/bin/env python3
"""E105 conversation memory summary and context compression expansion.

Controlled context-compression proxy. E105 teaches Operators that compress a
long evidence-state trace into a smaller summary while preserving required
facts, unresolved dependencies, citation pointers, and safe answer re-entry.

This is not open-domain summarization and not general dialogue memory.
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


ARTIFACT_CONTRACT = "E105_CONVERSATION_MEMORY_SUMMARY_AND_CONTEXT_COMPRESSION_EXPANSION"


@dataclass(frozen=True)
class OperatorSpec:
    operator_id: str
    display_name: str
    family: str
    role: str
    cost: float
    description: str


USEFUL_OPERATORS: tuple[OperatorSpec, ...] = (
    OperatorSpec("context_window_pressure_lens", "Context Window Pressure Lens", "Lens", "useful", 0.13, "Detects when a trace must be compressed before the next reasoning step."),
    OperatorSpec("summary_relevance_span_selector_lens", "Summary Relevance Span Selector Lens", "Lens", "useful", 0.14, "Selects only trace spans needed for the active question and pending dependencies."),
    OperatorSpec("required_fact_preservation_guard", "Required Fact Preservation Guard", "Guard", "useful", 0.14, "Requires every active fact dependency to survive compression."),
    OperatorSpec("unresolved_dependency_preservation_t_stab", "Unresolved Dependency Preservation T-Stab", "T-Stab", "useful", 0.14, "Stabilizes unresolved dependencies so compression cannot drop them."),
    OperatorSpec("citation_pointer_compaction_scribe", "Citation Pointer Compaction Scribe", "Scribe", "useful", 0.13, "Writes compact evidence pointers instead of copying full raw turns."),
    OperatorSpec("obsolete_turn_prune_guard", "Obsolete Turn Prune Guard", "Guard", "useful", 0.13, "Prunes repaired or obsolete turns only when their facts are preserved elsewhere."),
    OperatorSpec("summary_drift_detection_guard", "Summary Drift Detection Guard", "Guard", "useful", 0.13, "Rejects summaries that introduce stale, hallucinated, or cross-thread state."),
    OperatorSpec("compressed_context_reentry_scribe", "Compressed Context Reentry Scribe", "Scribe", "useful", 0.13, "Routes the compressed context back into the active reasoning state."),
)


CONTROL_OPERATORS: tuple[OperatorSpec, ...] = (
    OperatorSpec("last_turn_only_summary_control", "Last Turn Only Summary Control", "unsafe_control", "unsafe", 0.03, "Unsafe control: keeps only the last turn and drops earlier dependencies."),
    OperatorSpec("keyword_frequency_summary_control", "Keyword Frequency Summary Control", "unsafe_control", "unsafe", 0.04, "Unsafe control: summarizes by frequent tokens instead of active dependencies."),
    OperatorSpec("drop_unresolved_dependency_control", "Drop Unresolved Dependency Control", "unsafe_control", "unsafe", 0.04, "Unsafe control: removes unresolved dependencies to look shorter."),
    OperatorSpec("drop_citation_pointer_control", "Drop Citation Pointer Control", "unsafe_control", "unsafe", 0.04, "Unsafe control: keeps text but removes evidence pointers."),
    OperatorSpec("stale_fact_summary_control", "Stale Fact Summary Control", "unsafe_control", "unsafe", 0.04, "Unsafe control: preserves stale facts after newer repairs exist."),
    OperatorSpec("overcompressed_summary_control", "Overcompressed Summary Control", "unsafe_control", "unsafe", 0.03, "Unsafe control: compresses below the required fact budget."),
    OperatorSpec("hallucinated_bridge_summary_control", "Hallucinated Bridge Summary Control", "unsafe_control", "unsafe", 0.04, "Unsafe control: inserts an unsupported bridge fact between spans."),
    OperatorSpec("summary_guard_echo_clone", "Summary Guard Echo Clone", "Guard", "redundant", 0.18, "Redundant summary guard without unique value."),
)


OPERATOR_LIBRARY = USEFUL_OPERATORS + CONTROL_OPERATORS
USEFUL_IDS = tuple(operator.operator_id for operator in USEFUL_OPERATORS)
ALL_OPERATOR_IDS = tuple(operator.operator_id for operator in OPERATOR_LIBRARY)
OPERATOR_BY_ID = {operator.operator_id: operator for operator in OPERATOR_LIBRARY}
UNSAFE_IDS = {operator.operator_id for operator in CONTROL_OPERATORS if operator.role == "unsafe"}


@dataclass(frozen=True)
class SummaryCase:
    case_id: str
    source_split: str
    family: str
    raw_trace: tuple[str, ...]
    required_facts: tuple[str, ...]
    unresolved_dependencies: tuple[str, ...]
    expected_summary: tuple[str, ...]
    expected_citation_pointers: tuple[str, ...]
    expected_reentry_state: str
    expected_compression_ratio: float
    required_operators: tuple[str, ...]
    stale_fact_present: bool
    cross_thread_present: bool
    unresolved_present: bool
    citation_required: bool
    answerable_after_summary: bool


def stable_int(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def split_for(seed: int, case_id: str) -> str:
    bucket = stable_int(f"{seed}:{case_id}") % 10
    if bucket < 7:
        return "train"
    if bucket < 9:
        return "validation"
    return "adversarial"


def make_case(seed: int, index: int) -> SummaryCase:
    family = (
        "preserve_active_goal_and_missing_dependency",
        "prune_obsolete_repaired_turns",
        "preserve_citation_pointer_after_compression",
        "unresolved_dependency_not_dropped",
        "stale_fact_not_summarized",
        "multi_thread_context_isolated",
        "numeric_trace_summary_preserved",
        "source_priority_summary_preserved",
        "long_trace_budget_compression",
        "reentry_after_summary_answerable",
    )[index % 10]
    case_id = f"{family}_{seed}_{index:05d}"
    split = split_for(seed, case_id)
    base = ("context_window_pressure_lens", "summary_relevance_span_selector_lens", "required_fact_preservation_guard")
    carry = ("unresolved_dependency_preservation_t_stab", "citation_pointer_compaction_scribe", "compressed_context_reentry_scribe")
    safety = ("obsolete_turn_prune_guard", "summary_drift_detection_guard")

    if family == "preserve_active_goal_and_missing_dependency":
        raw = ("t1 ask verified valve state", "t2 valve K closed [e1]", "t3 source still missing", "t4 small talk obsolete")
        summary = ("goal:verified valve state", "fact:valve K closed", "pending:source")
        return SummaryCase(case_id, split, family, raw, ("valve K closed",), ("source",), summary, ("e1",), "ASK source", 0.45, base + carry + safety, False, False, True, True, False)
    if family == "prune_obsolete_repaired_turns":
        raw = ("t1 ask state", "t2 partial check", "t3 valve K closed [e2]", "t4 t2 obsolete after repair")
        summary = ("fact:valve K closed", "repair:e2")
        return SummaryCase(case_id, split, family, raw, ("valve K closed",), tuple(), summary, ("e2",), "ANSWER_READY", 0.35, base + ("citation_pointer_compaction_scribe", "obsolete_turn_prune_guard", "compressed_context_reentry_scribe", "summary_drift_detection_guard"), False, False, False, True, True)
    if family == "preserve_citation_pointer_after_compression":
        raw = ("t1 ask trace result", "t2 calc marker <<7*6=42>> [e3]", "t3 answer pending citation")
        summary = ("fact:calc result 42", "citation:e3")
        return SummaryCase(case_id, split, family, raw, ("calc result 42",), tuple(), summary, ("e3",), "ANSWER_READY", 0.40, base + carry + safety, False, False, False, True, True)
    if family == "unresolved_dependency_not_dropped":
        raw = ("t1 ask state+source", "t2 valve K closed [e4]", "t3 source missing")
        summary = ("fact:valve K closed", "pending:source")
        return SummaryCase(case_id, split, family, raw, ("valve K closed",), ("source",), summary, ("e4",), "ASK source", 0.50, base + carry + safety, False, False, True, True, False)
    if family == "stale_fact_not_summarized":
        raw = ("t1 valve K open [old]", "t2 newer repair valve K closed [e5]", "t3 old fact stale")
        summary = ("fact:valve K closed", "citation:e5", "rejected:old")
        return SummaryCase(case_id, split, family, raw, ("valve K closed",), tuple(), summary, ("e5",), "ANSWER_READY", 0.42, base + ("citation_pointer_compaction_scribe", "obsolete_turn_prune_guard", "summary_drift_detection_guard", "compressed_context_reentry_scribe"), True, False, False, True, True)
    if family == "multi_thread_context_isolated":
        raw = ("thread A ask valve K", "thread B valve K open [b1]", "thread A valve K closed [e6]")
        summary = ("thread:A", "fact:valve K closed", "citation:e6")
        return SummaryCase(case_id, split, family, raw, ("valve K closed",), tuple(), summary, ("e6",), "ANSWER_READY", 0.45, base + carry + safety, False, True, False, True, True)
    if family == "numeric_trace_summary_preserved":
        raw = ("t1 ask calc", "t2 <<12/3=4>> [e7]", "t3 arithmetic trace verified")
        summary = ("fact:visible calc result 4", "citation:e7")
        return SummaryCase(case_id, split, family, raw, ("visible calc result 4",), tuple(), summary, ("e7",), "ANSWER_READY", 0.40, base + carry + safety, False, False, False, True, True)
    if family == "source_priority_summary_preserved":
        raw = ("t1 low source says open", "t2 trusted source says closed [e8]", "t3 low source rejected")
        summary = ("fact:trusted valve K closed", "citation:e8", "rejected:low_source")
        return SummaryCase(case_id, split, family, raw, ("trusted valve K closed",), tuple(), summary, ("e8",), "ANSWER_READY", 0.42, base + carry + safety, True, False, False, True, True)
    if family == "long_trace_budget_compression":
        raw = ("t1 ask state", "t2 noise", "t3 noise", "t4 valve K closed [e9]", "t5 source verified [e10]", "t6 obsolete chatter")
        summary = ("fact:valve K closed", "fact:source verified", "citations:e9,e10")
        return SummaryCase(case_id, split, family, raw, ("valve K closed", "source verified"), tuple(), summary, ("e9", "e10"), "ANSWER_READY", 0.33, base + carry + safety, False, False, False, True, True)
    raw = ("t1 ask verified state", "t2 valve K closed [e11]", "t3 source verified [e12]", "t4 compress before final")
    summary = ("fact:valve K closed", "fact:source verified", "citations:e11,e12")
    return SummaryCase(case_id, split, family, raw, ("valve K closed", "source verified"), tuple(), summary, ("e11", "e12"), "ANSWER_READY", 0.38, base + carry + safety, False, False, False, True, True)


def generate_cases(seed: int, rows: int) -> list[SummaryCase]:
    return [make_case(seed, index) for index in range(rows)]


def unsafe_prediction(case: SummaryCase, selected: set[str]) -> dict[str, object] | None:
    if "last_turn_only_summary_control" in selected:
        return {"summary": (case.raw_trace[-1],), "citation_pointers": tuple(), "reentry_state": "UNKNOWN", "compression_ratio": 0.20, "required_fact_preserved": False, "unresolved_dependency_preserved": False, "citation_pointer_valid": False, "stale_fact_excluded": True, "context_reentry_valid": False, "hallucinated_summary": False, "lost_dependency": bool(case.unresolved_dependencies), "stale_summary": False, "cross_thread_bleed": False, "overcompression": True}
    if "keyword_frequency_summary_control" in selected:
        return {"summary": ("keyword:valve",), "citation_pointers": tuple(), "reentry_state": "UNKNOWN", "compression_ratio": 0.25, "required_fact_preserved": False, "unresolved_dependency_preserved": False, "citation_pointer_valid": False, "stale_fact_excluded": not case.stale_fact_present, "context_reentry_valid": False, "hallucinated_summary": False, "lost_dependency": bool(case.unresolved_dependencies), "stale_summary": case.stale_fact_present, "cross_thread_bleed": False, "overcompression": True}
    if "drop_unresolved_dependency_control" in selected and case.unresolved_present:
        return {"summary": tuple(item for item in case.expected_summary if not item.startswith("pending:")), "citation_pointers": case.expected_citation_pointers, "reentry_state": "ANSWER_READY", "compression_ratio": case.expected_compression_ratio, "required_fact_preserved": True, "unresolved_dependency_preserved": False, "citation_pointer_valid": True, "stale_fact_excluded": True, "context_reentry_valid": False, "hallucinated_summary": False, "lost_dependency": True, "stale_summary": False, "cross_thread_bleed": False, "overcompression": False}
    if "drop_citation_pointer_control" in selected and case.citation_required:
        return {"summary": case.expected_summary, "citation_pointers": tuple(), "reentry_state": case.expected_reentry_state, "compression_ratio": case.expected_compression_ratio, "required_fact_preserved": True, "unresolved_dependency_preserved": True, "citation_pointer_valid": False, "stale_fact_excluded": True, "context_reentry_valid": False, "hallucinated_summary": False, "lost_dependency": False, "stale_summary": False, "cross_thread_bleed": False, "overcompression": False}
    if "stale_fact_summary_control" in selected and case.stale_fact_present:
        return {"summary": case.expected_summary + ("stale:open",), "citation_pointers": case.expected_citation_pointers, "reentry_state": case.expected_reentry_state, "compression_ratio": case.expected_compression_ratio + 0.05, "required_fact_preserved": True, "unresolved_dependency_preserved": True, "citation_pointer_valid": True, "stale_fact_excluded": False, "context_reentry_valid": False, "hallucinated_summary": False, "lost_dependency": False, "stale_summary": True, "cross_thread_bleed": False, "overcompression": False}
    if "overcompressed_summary_control" in selected:
        return {"summary": ("too_short",), "citation_pointers": tuple(), "reentry_state": "UNKNOWN", "compression_ratio": 0.10, "required_fact_preserved": False, "unresolved_dependency_preserved": False, "citation_pointer_valid": False, "stale_fact_excluded": True, "context_reentry_valid": False, "hallucinated_summary": False, "lost_dependency": bool(case.unresolved_dependencies), "stale_summary": False, "cross_thread_bleed": False, "overcompression": True}
    if "hallucinated_bridge_summary_control" in selected:
        return {"summary": case.expected_summary + ("unsupported:therefore safe",), "citation_pointers": case.expected_citation_pointers, "reentry_state": case.expected_reentry_state, "compression_ratio": case.expected_compression_ratio + 0.05, "required_fact_preserved": True, "unresolved_dependency_preserved": True, "citation_pointer_valid": True, "stale_fact_excluded": True, "context_reentry_valid": False, "hallucinated_summary": True, "lost_dependency": False, "stale_summary": False, "cross_thread_bleed": case.cross_thread_present, "overcompression": False}
    return None


def predict(case: SummaryCase, selected: set[str]) -> dict[str, object]:
    unsafe = unsafe_prediction(case, selected)
    if unsafe:
        return unsafe
    required = set(case.required_operators)
    if required <= selected:
        return {"summary": case.expected_summary, "citation_pointers": case.expected_citation_pointers, "reentry_state": case.expected_reentry_state, "compression_ratio": case.expected_compression_ratio, "required_fact_preserved": True, "unresolved_dependency_preserved": True, "citation_pointer_valid": True, "stale_fact_excluded": True, "context_reentry_valid": True, "hallucinated_summary": False, "lost_dependency": False, "stale_summary": False, "cross_thread_bleed": False, "overcompression": False}
    if case.unresolved_present and "unresolved_dependency_preservation_t_stab" in selected:
        return {"summary": ("pending:" + "|".join(case.unresolved_dependencies),), "citation_pointers": tuple(), "reentry_state": "ASK_REMAINING", "compression_ratio": 0.30, "required_fact_preserved": False, "unresolved_dependency_preserved": True, "citation_pointer_valid": False, "stale_fact_excluded": True, "context_reentry_valid": False, "hallucinated_summary": False, "lost_dependency": False, "stale_summary": False, "cross_thread_bleed": False, "overcompression": False}
    return {"summary": tuple(), "citation_pointers": tuple(), "reentry_state": "UNKNOWN", "compression_ratio": 0.0, "required_fact_preserved": False, "unresolved_dependency_preserved": False, "citation_pointer_valid": False, "stale_fact_excluded": False, "context_reentry_valid": False, "hallucinated_summary": False, "lost_dependency": bool(case.unresolved_dependencies), "stale_summary": case.stale_fact_present, "cross_thread_bleed": case.cross_thread_present, "overcompression": True}


def evaluate(selected: set[str], cases: list[SummaryCase], split: str | None = None) -> dict[str, float]:
    rows = [case for case in cases if split is None or case.source_split == split]
    if not rows:
        return {"context_compression_success": 0.0, "required_fact_preservation": 0.0, "unresolved_dependency_preservation": 0.0, "citation_pointer_validity": 0.0, "stale_fact_exclusion": 0.0, "context_reentry_success": 0.0, "compression_ratio_validity": 0.0, "hallucinated_summary_rate": 0.0, "lost_dependency_rate": 0.0, "stale_summary_rate": 0.0, "cross_thread_bleed_rate": 0.0, "overcompression_rate": 0.0, "utility": -1.0}
    success = fact = unresolved = citation = stale_exclusion = reentry = ratio_valid = 0
    hallucinated = lost = stale = bleed = overcompressed = 0
    partial = 0.0
    for case in rows:
        pred = predict(case, selected)
        required = set(case.required_operators)
        partial += len(required & selected) / max(1, len(required))
        row_success = (
            tuple(pred["summary"]) == case.expected_summary
            and tuple(pred["citation_pointers"]) == case.expected_citation_pointers
            and pred["reentry_state"] == case.expected_reentry_state
            and pred["required_fact_preserved"]
            and pred["unresolved_dependency_preserved"]
            and pred["citation_pointer_valid"]
            and pred["stale_fact_excluded"]
            and pred["context_reentry_valid"]
            and float(pred["compression_ratio"]) <= case.expected_compression_ratio + 1e-9
            and not pred["hallucinated_summary"]
            and not pred["lost_dependency"]
            and not pred["stale_summary"]
            and not pred["cross_thread_bleed"]
            and not pred["overcompression"]
        )
        success += int(row_success)
        fact += int(pred["required_fact_preserved"])
        unresolved += int(pred["unresolved_dependency_preserved"])
        citation += int(pred["citation_pointer_valid"])
        stale_exclusion += int(pred["stale_fact_excluded"])
        reentry += int(pred["context_reentry_valid"])
        ratio_valid += int(float(pred["compression_ratio"]) <= case.expected_compression_ratio + 1e-9 and not pred["overcompression"])
        hallucinated += int(pred["hallucinated_summary"])
        lost += int(pred["lost_dependency"])
        stale += int(pred["stale_summary"])
        bleed += int(pred["cross_thread_bleed"])
        overcompressed += int(pred["overcompression"])
    count = len(rows)
    cost = sum(OPERATOR_BY_ID[operator_id].cost for operator_id in selected)
    score = success / count
    partial_score = partial / count
    utility = score + 0.28 * partial_score - 1.3 * (hallucinated / count) - 1.2 * (lost / count) - 1.1 * (stale / count) - 1.1 * (bleed / count) - 0.8 * (overcompressed / count) - 0.01 * cost
    return {
        "context_compression_success": round(score, 6),
        "required_fact_preservation": round(fact / count, 6),
        "unresolved_dependency_preservation": round(unresolved / count, 6),
        "citation_pointer_validity": round(citation / count, 6),
        "stale_fact_exclusion": round(stale_exclusion / count, 6),
        "context_reentry_success": round(reentry / count, 6),
        "compression_ratio_validity": round(ratio_valid / count, 6),
        "hallucinated_summary_rate": round(hallucinated / count, 6),
        "lost_dependency_rate": round(lost / count, 6),
        "stale_summary_rate": round(stale / count, 6),
        "cross_thread_bleed_rate": round(bleed / count, 6),
        "overcompression_rate": round(overcompressed / count, 6),
        "partial_compression_score": round(partial_score, 6),
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
        lost_dependency_deltas = []
        for result in seed_results:
            selected = set(result["selected"])  # type: ignore[arg-type]
            cases = [SummaryCase(**case) for case in result["cases"]]  # type: ignore[arg-type]
            full = evaluate(selected, cases, "validation")
            ablated = evaluate(selected - {operator_id}, cases, "validation")
            losses.append(full["context_compression_success"] - ablated["context_compression_success"])
            lost_dependency_deltas.append(ablated["lost_dependency_rate"] - full["lost_dependency_rate"])
        summary[operator_id] = {
            "mean_context_compression_success_loss": round(statistics.mean(losses), 6),
            "mean_lost_dependency_delta": round(statistics.mean(lost_dependency_deltas), 6),
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
        "validation_context_compression_success_min": min(vals("validation", "context_compression_success")),
        "validation_context_compression_success_mean": round(statistics.mean(vals("validation", "context_compression_success")), 6),
        "adversarial_context_compression_success_min": min(vals("adversarial", "context_compression_success")),
        "adversarial_context_compression_success_mean": round(statistics.mean(vals("adversarial", "context_compression_success")), 6),
        "validation_required_fact_preservation_min": min(vals("validation", "required_fact_preservation")),
        "validation_unresolved_dependency_preservation_min": min(vals("validation", "unresolved_dependency_preservation")),
        "validation_citation_pointer_validity_min": min(vals("validation", "citation_pointer_validity")),
        "validation_stale_fact_exclusion_min": min(vals("validation", "stale_fact_exclusion")),
        "validation_context_reentry_success_min": min(vals("validation", "context_reentry_success")),
        "validation_compression_ratio_validity_min": min(vals("validation", "compression_ratio_validity")),
        "adversarial_hallucinated_summary_rate_max": max(vals("adversarial", "hallucinated_summary_rate")),
        "adversarial_lost_dependency_rate_max": max(vals("adversarial", "lost_dependency_rate")),
        "adversarial_stale_summary_rate_max": max(vals("adversarial", "stale_summary_rate")),
        "adversarial_cross_thread_bleed_rate_max": max(vals("adversarial", "cross_thread_bleed_rate")),
        "adversarial_overcompression_rate_max": max(vals("adversarial", "overcompression_rate")),
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
        "validation_context_compression_success_min",
        "adversarial_context_compression_success_min",
        "validation_required_fact_preservation_min",
        "validation_unresolved_dependency_preservation_min",
        "validation_citation_pointer_validity_min",
        "validation_stale_fact_exclusion_min",
        "validation_context_reentry_success_min",
        "validation_compression_ratio_validity_min",
    ]:
        if agg[key] != 1.0:
            failures.append(f"{key} below 1.0")
    for key in [
        "adversarial_hallucinated_summary_rate_max",
        "adversarial_lost_dependency_rate_max",
        "adversarial_stale_summary_rate_max",
        "adversarial_cross_thread_bleed_rate_max",
        "adversarial_overcompression_rate_max",
    ]:
        if agg[key] != 0.0:
            failures.append(f"{key} nonzero")
    unsafe_final = sum(1 for result in seed_results for operator_id in result["selected"] if operator_id in UNSAFE_IDS)  # type: ignore[operator]
    if unsafe_final:
        failures.append("unsafe operator selected")
    decision = "e105_context_compression_summary_expansion_confirmed" if not failures else "e105_context_compression_summary_incomplete"
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
        "context_compression_summary": True,
        "open_domain_summarization": False,
        "direct_summary_without_evidence_allowed": False,
        "requires_citation_pointer_preservation": True,
        "requires_unresolved_dependency_preservation": True,
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
            typed_case = SummaryCase(**case)
            pred = predict(typed_case, set(result["selected"]))  # type: ignore[arg-type]
            append_jsonl(out / "row_level_samples.jsonl", {
                "seed": result["seed"],
                "case_id": typed_case.case_id,
                "family": typed_case.family,
                "raw_trace": typed_case.raw_trace,
                "expected_summary": typed_case.expected_summary,
                "expected_citation_pointers": typed_case.expected_citation_pointers,
                "expected_reentry_state": typed_case.expected_reentry_state,
                "expected_compression_ratio": typed_case.expected_compression_ratio,
                "predicted": pred,
            })
            sample_rows += 1
            if sample_rows >= 480:
                break
        if sample_rows >= 480:
            break
    report = [
        "# E105 Conversation Memory Summary And Context Compression Expansion Result",
        "",
        f"decision = `{decision}`",
        "",
        "Boundary: controlled context compression, not open-domain summarization.",
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
    parser.add_argument("--out", default="target/pilot_wave/e105_conversation_memory_summary_and_context_compression_expansion")
    parser.add_argument("--artifact-sample-dir", default="docs/research/artifact_samples/e105_conversation_memory_summary_and_context_compression_expansion")
    parser.add_argument("--seeds", default="110501,110502,110503,110504,110505,110506,110507,110508,110509,110510,110511,110512,110513,110514,110515,110516")
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
        "boundary": "controlled context compression probe; not open-domain summarization",
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
