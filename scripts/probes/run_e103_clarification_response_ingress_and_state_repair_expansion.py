#!/usr/bin/env python3
"""E103 clarification response ingress and state repair expansion.

Controlled clarification-response proxy. E103 teaches Operators that ingest a
response to an earlier ASK/DEFER action, bind it to the pending dependency,
write a safe repair patch, and re-enter grounded answer rendering only after the
state is repaired.

This is not open-domain dialogue and not general language understanding.
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


ARTIFACT_CONTRACT = "E103_CLARIFICATION_RESPONSE_INGRESS_AND_STATE_REPAIR_EXPANSION"


@dataclass(frozen=True)
class OperatorSpec:
    operator_id: str
    display_name: str
    family: str
    role: str
    cost: float
    description: str


USEFUL_OPERATORS: tuple[OperatorSpec, ...] = (
    OperatorSpec("pending_question_trace_lens", "Pending Question Trace Lens", "Lens", "useful", 0.13, "Reads the previous ASK/DEFER trace and recovers the pending dependency."),
    OperatorSpec("clarification_span_locator_lens", "Clarification Span Locator Lens", "Lens", "useful", 0.13, "Locates the mechanical answer span inside a clarification response."),
    OperatorSpec("clarification_dependency_binder_guard", "Clarification Dependency Binder Guard", "Guard", "useful", 0.14, "Requires the clarification span to answer the active pending dependency."),
    OperatorSpec("state_repair_patch_scribe", "State Repair Patch Scribe", "Scribe", "useful", 0.14, "Writes a repair proposal that updates Flow/Ground from the clarification."),
    OperatorSpec("stale_pending_question_guard", "Stale Pending Question Guard", "Guard", "useful", 0.13, "Rejects clarification responses that target an old or already closed question."),
    OperatorSpec("irrelevant_clarification_filter_guard", "Irrelevant Clarification Filter Guard", "Guard", "useful", 0.13, "Rejects clarification text that does not cover the pending dependency."),
    OperatorSpec("repaired_answer_reentry_scribe", "Repaired Answer Reentry Scribe", "Scribe", "useful", 0.13, "Routes the repaired state back into grounded answer decision."),
    OperatorSpec("repair_trace_integrity_t_stab", "Repair Trace Integrity T-Stab", "T-Stab", "useful", 0.13, "Stabilizes the ask -> clarification -> repair -> answer trace chain."),
)


CONTROL_OPERATORS: tuple[OperatorSpec, ...] = (
    OperatorSpec("any_clarification_committer", "Any Clarification Committer", "unsafe_control", "unsafe", 0.03, "Unsafe control: commits any clarification response as repair."),
    OperatorSpec("stale_question_reopener", "Stale Question Reopener", "unsafe_control", "unsafe", 0.04, "Unsafe control: reopens closed pending questions and commits stale answers."),
    OperatorSpec("irrelevant_answer_binder", "Irrelevant Answer Binder", "unsafe_control", "unsafe", 0.04, "Unsafe control: binds unrelated clarification text to the active dependency."),
    OperatorSpec("conflicting_clarification_overwriter", "Conflicting Clarification Overwriter", "unsafe_control", "unsafe", 0.04, "Unsafe control: overwrites Ground when clarification conflicts with it."),
    OperatorSpec("answer_without_reentry_control", "Answer Without Reentry Control", "unsafe_control", "unsafe", 0.03, "Unsafe control: emits final answer without state-repair reentry."),
    OperatorSpec("always_reask_control", "Always Reask Control", "control", "noop", 0.02, "Control: asks again even when a clarification is sufficient."),
    OperatorSpec("latest_text_blind_binder", "Latest Text Blind Binder", "unsafe_control", "unsafe", 0.04, "Unsafe control: treats the latest text span as correct regardless of pending trace."),
    OperatorSpec("repair_trace_echo_clone", "Repair Trace Echo Clone", "T-Stab", "redundant", 0.18, "Redundant repair trace stabilizer without unique value."),
)


OPERATOR_LIBRARY = USEFUL_OPERATORS + CONTROL_OPERATORS
USEFUL_IDS = tuple(operator.operator_id for operator in USEFUL_OPERATORS)
ALL_OPERATOR_IDS = tuple(operator.operator_id for operator in OPERATOR_LIBRARY)
OPERATOR_BY_ID = {operator.operator_id: operator for operator in OPERATOR_LIBRARY}
UNSAFE_IDS = {operator.operator_id for operator in CONTROL_OPERATORS if operator.role == "unsafe"}


@dataclass(frozen=True)
class ClarificationCase:
    case_id: str
    source_split: str
    family: str
    pending_question_id: str
    pending_dependency: str
    clarification_text: str
    existing_ground: tuple[str, ...]
    expected_action: str
    expected_final_answer: str
    expected_repaired_dependency: str
    expected_trace: tuple[str, ...]
    required_operators: tuple[str, ...]
    should_commit_repair: bool
    stale_pending: bool
    irrelevant_response: bool
    conflict_response: bool
    partial_response: bool
    answerable_after_repair: bool


def stable_int(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def split_for(seed: int, case_id: str) -> str:
    bucket = stable_int(f"{seed}:{case_id}") % 10
    if bucket < 7:
        return "train"
    if bucket < 9:
        return "validation"
    return "adversarial"


def make_case(seed: int, index: int) -> ClarificationCase:
    family = (
        "missing_dependency_resolved",
        "irrelevant_clarification_rejected",
        "stale_pending_question_rejected",
        "conflicting_clarification_defer",
        "partial_clarification_reask",
        "multi_dependency_incremental_repair",
        "source_attributed_clarification_commit",
        "no_answer_hold_unresolved",
        "numeric_marker_clarification_repair",
        "latest_cycle_clarification_commit",
    )[index % 10]
    case_id = f"{family}_{seed}_{index:05d}"
    split = split_for(seed, case_id)
    base = ("pending_question_trace_lens", "clarification_span_locator_lens", "clarification_dependency_binder_guard")
    repair = ("state_repair_patch_scribe", "repaired_answer_reentry_scribe", "repair_trace_integrity_t_stab")
    stale = ("stale_pending_question_guard",)
    irrelevant = ("irrelevant_clarification_filter_guard",)

    if family == "missing_dependency_resolved":
        trace = ("pending:q1:valve K state", "clarification:c1 covers valve K state", "repair:valve K closed", "reenter_answer")
        return ClarificationCase(case_id, split, family, "q1", "valve K state", "c1: valve K is closed", tuple(), "COMMIT_REPAIR_ANSWER", "valve K is closed", "valve K state=closed", trace, base + repair, True, False, False, False, False, True)
    if family == "irrelevant_clarification_rejected":
        trace = ("pending:q2:valve K state", "clarification:c2 covers valve J state", "reject_irrelevant", "ask_again")
        return ClarificationCase(case_id, split, family, "q2", "valve K state", "c2: valve J is closed", tuple(), "ASK_FOR_EVIDENCE", "", "", trace, base + irrelevant + ("repair_trace_integrity_t_stab",), False, False, True, False, False, False)
    if family == "stale_pending_question_rejected":
        trace = ("pending:q3 already closed", "clarification:c3 targets stale q3", "reject_stale", "keep_current_ground")
        return ClarificationCase(case_id, split, family, "q3", "valve K state", "c3: valve K is open", ("current: valve K closed",), "REJECT_STALE_CLARIFICATION", "", "", trace, ("pending_question_trace_lens",) + stale + ("repair_trace_integrity_t_stab",), False, True, False, False, False, False)
    if family == "conflicting_clarification_defer":
        trace = ("pending:q4:valve K state", "ground:valve K closed", "clarification:c4 says open", "defer_conflict")
        return ClarificationCase(case_id, split, family, "q4", "valve K state", "c4: valve K is open", ("ground: valve K closed",), "DEFER_CONFLICT", "", "", trace, base + ("stale_pending_question_guard", "irrelevant_clarification_filter_guard", "repair_trace_integrity_t_stab"), False, False, False, True, False, False)
    if family == "partial_clarification_reask":
        trace = ("pending:q5:valve K state", "clarification:c5 partial", "dependency_not_resolved", "ask_again")
        return ClarificationCase(case_id, split, family, "q5", "valve K state", "c5: valve K was checked", tuple(), "ASK_FOR_EVIDENCE", "", "", trace, base + irrelevant + ("repair_trace_integrity_t_stab",), False, False, False, False, True, False)
    if family == "multi_dependency_incremental_repair":
        trace = ("pending:q6:state+source", "clarification:c6 covers state only", "partial_repair:state", "ask_missing_source")
        return ClarificationCase(case_id, split, family, "q6", "valve K state + source", "c6: valve K is closed", tuple(), "PARTIAL_REPAIR_ASK", "", "valve K state=closed", trace, base + ("state_repair_patch_scribe", "repair_trace_integrity_t_stab"), True, False, False, False, True, False)
    if family == "source_attributed_clarification_commit":
        trace = ("pending:q7:verified valve K state", "clarification:c7 source verified", "repair:verified closed", "reenter_answer")
        return ClarificationCase(case_id, split, family, "q7", "verified valve K state", "c7: verified source says valve K is closed", tuple(), "COMMIT_REPAIR_ANSWER", "verified valve K state is closed", "verified valve K state=closed", trace, base + repair, True, False, False, False, False, True)
    if family == "no_answer_hold_unresolved":
        trace = ("pending:q8:valve K state", "clarification:c8 no evidence", "hold_unresolved")
        return ClarificationCase(case_id, split, family, "q8", "valve K state", "c8: I do not know", tuple(), "HOLD_UNRESOLVED", "", "", trace, base + irrelevant + ("repair_trace_integrity_t_stab",), False, False, True, False, False, False)
    if family == "numeric_marker_clarification_repair":
        trace = ("pending:q9:visible calc result", "clarification:c9 marker valid", "repair:42", "reenter_answer")
        return ClarificationCase(case_id, split, family, "q9", "visible calc result", "c9: <<7*6=42>>", tuple(), "COMMIT_REPAIR_ANSWER", "the visible calc trace shows 42", "visible calc result=42", trace, base + repair, True, False, False, False, False, True)
    trace = ("pending:q10 latest cycle", "ignore_old:c10_old", "clarification:c10_new covers valve K state", "repair:closed", "reenter_answer")
    return ClarificationCase(case_id, split, family, "q10", "latest valve K state", "c10_old: valve K open; c10_new: valve K closed", ("old: valve K open",), "COMMIT_REPAIR_ANSWER", "latest valve K state is closed", "latest valve K state=closed", trace, base + repair + stale, True, False, False, False, False, True)


def generate_cases(seed: int, rows: int) -> list[ClarificationCase]:
    return [make_case(seed, index) for index in range(rows)]


def unsafe_prediction(case: ClarificationCase, selected: set[str]) -> dict[str, object] | None:
    if "any_clarification_committer" in selected:
        return {"action": "COMMIT_REPAIR_ANSWER", "final_answer": "valve K is closed", "repaired_dependency": f"{case.pending_dependency}=committed", "commit_repair": True, "dependency_binding_valid": False, "state_repair_valid": False, "answer_reentry_valid": True, "trace": ("commit_any",), "trace_valid": False, "unsafe_repair": True, "stale_repair": case.stale_pending, "irrelevant_repair": case.irrelevant_response, "false_reask": False}
    if "stale_question_reopener" in selected and case.stale_pending:
        return {"action": "COMMIT_REPAIR_ANSWER", "final_answer": "valve K is open", "repaired_dependency": "valve K state=open", "commit_repair": True, "dependency_binding_valid": False, "state_repair_valid": False, "answer_reentry_valid": True, "trace": ("reopened_stale",), "trace_valid": False, "unsafe_repair": True, "stale_repair": True, "irrelevant_repair": False, "false_reask": False}
    if "irrelevant_answer_binder" in selected and case.irrelevant_response:
        return {"action": "COMMIT_REPAIR_ANSWER", "final_answer": "valve K is closed", "repaired_dependency": "valve K state=from_irrelevant", "commit_repair": True, "dependency_binding_valid": False, "state_repair_valid": False, "answer_reentry_valid": True, "trace": ("bound_irrelevant",), "trace_valid": False, "unsafe_repair": True, "stale_repair": False, "irrelevant_repair": True, "false_reask": False}
    if "conflicting_clarification_overwriter" in selected and case.conflict_response:
        return {"action": "COMMIT_REPAIR_ANSWER", "final_answer": "valve K is open", "repaired_dependency": "valve K state=open", "commit_repair": True, "dependency_binding_valid": True, "state_repair_valid": False, "answer_reentry_valid": True, "trace": ("overwrite_conflict",), "trace_valid": False, "unsafe_repair": True, "stale_repair": False, "irrelevant_repair": False, "false_reask": False}
    if "answer_without_reentry_control" in selected and case.answerable_after_repair:
        return {"action": "COMMIT_REPAIR_ANSWER", "final_answer": case.expected_final_answer, "repaired_dependency": case.expected_repaired_dependency, "commit_repair": case.should_commit_repair, "dependency_binding_valid": True, "state_repair_valid": True, "answer_reentry_valid": False, "trace": ("answer_without_reentry",), "trace_valid": False, "unsafe_repair": True, "stale_repair": False, "irrelevant_repair": False, "false_reask": False}
    if "always_reask_control" in selected and case.answerable_after_repair:
        return {"action": "ASK_FOR_EVIDENCE", "final_answer": "", "repaired_dependency": "", "commit_repair": False, "dependency_binding_valid": True, "state_repair_valid": False, "answer_reentry_valid": False, "trace": ("always_reask",), "trace_valid": False, "unsafe_repair": False, "stale_repair": False, "irrelevant_repair": False, "false_reask": True}
    if "latest_text_blind_binder" in selected and (case.stale_pending or case.conflict_response):
        return {"action": "COMMIT_REPAIR_ANSWER", "final_answer": "latest text says open", "repaired_dependency": f"{case.pending_dependency}=latest_text", "commit_repair": True, "dependency_binding_valid": False, "state_repair_valid": False, "answer_reentry_valid": True, "trace": ("latest_text_blind",), "trace_valid": False, "unsafe_repair": True, "stale_repair": case.stale_pending, "irrelevant_repair": False, "false_reask": False}
    return None


def predict(case: ClarificationCase, selected: set[str]) -> dict[str, object]:
    unsafe = unsafe_prediction(case, selected)
    if unsafe:
        return unsafe
    required = set(case.required_operators)
    if required <= selected:
        return {"action": case.expected_action, "final_answer": case.expected_final_answer, "repaired_dependency": case.expected_repaired_dependency, "commit_repair": case.should_commit_repair, "dependency_binding_valid": True, "state_repair_valid": True, "answer_reentry_valid": True, "trace": case.expected_trace, "trace_valid": True, "unsafe_repair": False, "stale_repair": False, "irrelevant_repair": False, "false_reask": False}
    if case.stale_pending and "stale_pending_question_guard" in selected:
        return {"action": "REJECT_STALE_CLARIFICATION", "final_answer": "", "repaired_dependency": "", "commit_repair": False, "dependency_binding_valid": True, "state_repair_valid": True, "answer_reentry_valid": False, "trace": ("reject_stale",), "trace_valid": False, "unsafe_repair": False, "stale_repair": False, "irrelevant_repair": False, "false_reask": False}
    if case.irrelevant_response and "irrelevant_clarification_filter_guard" in selected:
        return {"action": "ASK_FOR_EVIDENCE", "final_answer": "", "repaired_dependency": "", "commit_repair": False, "dependency_binding_valid": True, "state_repair_valid": True, "answer_reentry_valid": False, "trace": ("reject_irrelevant",), "trace_valid": False, "unsafe_repair": False, "stale_repair": False, "irrelevant_repair": False, "false_reask": False}
    if case.partial_response and "clarification_dependency_binder_guard" in selected:
        return {"action": "ASK_FOR_EVIDENCE", "final_answer": "", "repaired_dependency": "", "commit_repair": False, "dependency_binding_valid": True, "state_repair_valid": False, "answer_reentry_valid": False, "trace": ("partial_reask",), "trace_valid": False, "unsafe_repair": False, "stale_repair": False, "irrelevant_repair": False, "false_reask": False}
    return {"action": "ASK_FOR_EVIDENCE", "final_answer": "", "repaired_dependency": "", "commit_repair": False, "dependency_binding_valid": False, "state_repair_valid": False, "answer_reentry_valid": False, "trace": ("unrepaired",), "trace_valid": False, "unsafe_repair": False, "stale_repair": False, "irrelevant_repair": False, "false_reask": case.answerable_after_repair}


def evaluate(selected: set[str], cases: list[ClarificationCase], split: str | None = None) -> dict[str, float]:
    rows = [case for case in cases if split is None or case.source_split == split]
    if not rows:
        return {"clarification_repair_success": 0.0, "final_answer_after_repair_accuracy": 0.0, "dependency_binding_validity": 0.0, "state_repair_validity": 0.0, "answer_reentry_success": 0.0, "trace_integrity": 0.0, "unsafe_repair_rate": 0.0, "stale_repair_rate": 0.0, "irrelevant_repair_rate": 0.0, "false_reask_rate": 0.0, "utility": -1.0}
    success = answer = binding = repair = reentry = trace = 0
    unsafe = stale = irrelevant = false_reask = 0
    partial = 0.0
    for case in rows:
        pred = predict(case, selected)
        required = set(case.required_operators)
        partial += len(required & selected) / max(1, len(required))
        row_success = (
            pred["action"] == case.expected_action
            and pred["final_answer"] == case.expected_final_answer
            and pred["repaired_dependency"] == case.expected_repaired_dependency
            and bool(pred["commit_repair"]) == case.should_commit_repair
            and tuple(pred["trace"]) == case.expected_trace
            and pred["dependency_binding_valid"]
            and pred["state_repair_valid"]
            and (pred["answer_reentry_valid"] or not case.answerable_after_repair)
            and pred["trace_valid"]
            and not pred["unsafe_repair"]
            and not pred["stale_repair"]
            and not pred["irrelevant_repair"]
            and not pred["false_reask"]
        )
        success += int(row_success)
        answer += int(pred["final_answer"] == case.expected_final_answer and pred["action"] == case.expected_action)
        binding += int(pred["dependency_binding_valid"])
        repair += int(pred["state_repair_valid"] and bool(pred["commit_repair"]) == case.should_commit_repair)
        reentry += int(pred["answer_reentry_valid"] or not case.answerable_after_repair)
        trace += int(pred["trace_valid"] and tuple(pred["trace"]) == case.expected_trace)
        unsafe += int(pred["unsafe_repair"])
        stale += int(pred["stale_repair"])
        irrelevant += int(pred["irrelevant_repair"])
        false_reask += int(pred["false_reask"])
    count = len(rows)
    cost = sum(OPERATOR_BY_ID[operator_id].cost for operator_id in selected)
    score = success / count
    partial_score = partial / count
    utility = score + 0.28 * partial_score - 1.8 * (unsafe / count) - 1.2 * (stale / count) - 1.2 * (irrelevant / count) - 0.8 * (false_reask / count) - 0.01 * cost
    return {
        "clarification_repair_success": round(score, 6),
        "final_answer_after_repair_accuracy": round(answer / count, 6),
        "dependency_binding_validity": round(binding / count, 6),
        "state_repair_validity": round(repair / count, 6),
        "answer_reentry_success": round(reentry / count, 6),
        "trace_integrity": round(trace / count, 6),
        "unsafe_repair_rate": round(unsafe / count, 6),
        "stale_repair_rate": round(stale / count, 6),
        "irrelevant_repair_rate": round(irrelevant / count, 6),
        "false_reask_rate": round(false_reask / count, 6),
        "partial_repair_score": round(partial_score, 6),
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
            cases = [ClarificationCase(**case) for case in result["cases"]]  # type: ignore[arg-type]
            full = evaluate(selected, cases, "validation")
            ablated = evaluate(selected - {operator_id}, cases, "validation")
            losses.append(full["clarification_repair_success"] - ablated["clarification_repair_success"])
            unsafe_deltas.append(ablated["unsafe_repair_rate"] - full["unsafe_repair_rate"])
        summary[operator_id] = {
            "mean_clarification_repair_success_loss": round(statistics.mean(losses), 6),
            "mean_unsafe_repair_delta": round(statistics.mean(unsafe_deltas), 6),
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
        "validation_clarification_repair_success_min": min(vals("validation", "clarification_repair_success")),
        "validation_clarification_repair_success_mean": round(statistics.mean(vals("validation", "clarification_repair_success")), 6),
        "adversarial_clarification_repair_success_min": min(vals("adversarial", "clarification_repair_success")),
        "adversarial_clarification_repair_success_mean": round(statistics.mean(vals("adversarial", "clarification_repair_success")), 6),
        "validation_final_answer_after_repair_accuracy_min": min(vals("validation", "final_answer_after_repair_accuracy")),
        "validation_dependency_binding_validity_min": min(vals("validation", "dependency_binding_validity")),
        "validation_state_repair_validity_min": min(vals("validation", "state_repair_validity")),
        "validation_answer_reentry_success_min": min(vals("validation", "answer_reentry_success")),
        "validation_trace_integrity_min": min(vals("validation", "trace_integrity")),
        "adversarial_unsafe_repair_rate_max": max(vals("adversarial", "unsafe_repair_rate")),
        "adversarial_stale_repair_rate_max": max(vals("adversarial", "stale_repair_rate")),
        "adversarial_irrelevant_repair_rate_max": max(vals("adversarial", "irrelevant_repair_rate")),
        "adversarial_false_reask_rate_max": max(vals("adversarial", "false_reask_rate")),
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
        "validation_clarification_repair_success_min",
        "adversarial_clarification_repair_success_min",
        "validation_final_answer_after_repair_accuracy_min",
        "validation_dependency_binding_validity_min",
        "validation_state_repair_validity_min",
        "validation_answer_reentry_success_min",
        "validation_trace_integrity_min",
    ]:
        if agg[key] != 1.0:
            failures.append(f"{key} below 1.0")
    for key in ["adversarial_unsafe_repair_rate_max", "adversarial_stale_repair_rate_max", "adversarial_irrelevant_repair_rate_max", "adversarial_false_reask_rate_max"]:
        if agg[key] != 0.0:
            failures.append(f"{key} nonzero")
    unsafe_final = sum(1 for result in seed_results for operator_id in result["selected"] if operator_id in UNSAFE_IDS)  # type: ignore[operator]
    if unsafe_final:
        failures.append("unsafe operator selected")
    decision = "e103_clarification_response_state_repair_expansion_confirmed" if not failures else "e103_clarification_response_state_repair_incomplete"
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
        "clarification_response_ingress": True,
        "state_repair_after_ask": True,
        "open_domain_dialogue": False,
        "direct_repair_without_pending_question_allowed": False,
        "requires_repair_trace": True,
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
            typed_case = ClarificationCase(**case)
            pred = predict(typed_case, set(result["selected"]))  # type: ignore[arg-type]
            append_jsonl(out / "row_level_samples.jsonl", {
                "seed": result["seed"],
                "case_id": typed_case.case_id,
                "family": typed_case.family,
                "pending_question_id": typed_case.pending_question_id,
                "pending_dependency": typed_case.pending_dependency,
                "clarification_text": typed_case.clarification_text,
                "expected_action": typed_case.expected_action,
                "expected_final_answer": typed_case.expected_final_answer,
                "expected_repaired_dependency": typed_case.expected_repaired_dependency,
                "expected_trace": typed_case.expected_trace,
                "predicted": pred,
            })
            sample_rows += 1
            if sample_rows >= 480:
                break
        if sample_rows >= 480:
            break
    report = [
        "# E103 Clarification Response Ingress And State Repair Expansion Result",
        "",
        f"decision = `{decision}`",
        "",
        "Boundary: controlled clarification-response state repair, not open-domain dialogue.",
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
    parser.add_argument("--out", default="target/pilot_wave/e103_clarification_response_ingress_and_state_repair_expansion")
    parser.add_argument("--artifact-sample-dir", default="docs/research/artifact_samples/e103_clarification_response_ingress_and_state_repair_expansion")
    parser.add_argument("--seeds", default="110301,110302,110303,110304,110305,110306,110307,110308,110309,110310,110311,110312,110313,110314,110315,110316")
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
        "boundary": "controlled clarification-response state repair probe; not open-domain dialogue",
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
