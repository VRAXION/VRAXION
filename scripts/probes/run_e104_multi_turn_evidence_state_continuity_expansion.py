#!/usr/bin/env python3
"""E104 multi-turn evidence state continuity expansion.

Controlled multi-turn state proxy. E104 teaches Operators that preserve pending
dependencies, repaired Ground/Flow deltas, and answer trace continuity across
several ASK/clarification cycles.

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


ARTIFACT_CONTRACT = "E104_MULTI_TURN_EVIDENCE_STATE_CONTINUITY_EXPANSION"


@dataclass(frozen=True)
class OperatorSpec:
    operator_id: str
    display_name: str
    family: str
    role: str
    cost: float
    description: str


USEFUL_OPERATORS: tuple[OperatorSpec, ...] = (
    OperatorSpec("turn_boundary_cycle_lens", "Turn Boundary Cycle Lens", "Lens", "useful", 0.13, "Separates incoming clarification text by turn and cycle id."),
    OperatorSpec("pending_dependency_stack_t_stab", "Pending Dependency Stack T-Stab", "T-Stab", "useful", 0.14, "Stabilizes unresolved dependencies across multiple turns."),
    OperatorSpec("active_turn_state_router_guard", "Active Turn State Router Guard", "Guard", "useful", 0.14, "Routes each clarification to the currently active pending question."),
    OperatorSpec("multi_turn_ground_delta_scribe", "Multi-Turn Ground Delta Scribe", "Scribe", "useful", 0.14, "Writes ordered Ground/Flow deltas after each accepted clarification."),
    OperatorSpec("clarification_chain_join_lens", "Clarification Chain Join Lens", "Lens", "useful", 0.13, "Joins several clarification spans into one answerable evidence chain."),
    OperatorSpec("cross_turn_stale_context_guard", "Cross-Turn Stale Context Guard", "Guard", "useful", 0.13, "Blocks stale or previous-turn state from contaminating the active cycle."),
    OperatorSpec("unresolved_state_carry_scribe", "Unresolved State Carry Scribe", "Scribe", "useful", 0.13, "Carries unresolved dependencies forward without pretending they are answered."),
    OperatorSpec("final_turn_answer_continuity_guard", "Final Turn Answer Continuity Guard", "Guard", "useful", 0.13, "Allows final answer only when all turn-linked dependencies are resolved."),
)


CONTROL_OPERATORS: tuple[OperatorSpec, ...] = (
    OperatorSpec("single_turn_memory_reset_control", "Single-Turn Memory Reset Control", "unsafe_control", "unsafe", 0.03, "Unsafe control: forgets earlier pending dependencies each turn."),
    OperatorSpec("latest_turn_only_binder", "Latest Turn Only Binder", "unsafe_control", "unsafe", 0.04, "Unsafe control: binds every answer to the latest turn only."),
    OperatorSpec("stale_dependency_reuse_control", "Stale Dependency Reuse Control", "unsafe_control", "unsafe", 0.04, "Unsafe control: reuses old resolved dependencies after a newer cycle supersedes them."),
    OperatorSpec("cross_thread_contamination_control", "Cross-Thread Contamination Control", "unsafe_control", "unsafe", 0.04, "Unsafe control: mixes state from another pending question chain."),
    OperatorSpec("answer_before_all_turns_resolved", "Answer Before All Turns Resolved", "unsafe_control", "unsafe", 0.03, "Unsafe control: answers as soon as one dependency is repaired."),
    OperatorSpec("always_restart_dialogue_control", "Always Restart Dialogue Control", "control", "noop", 0.02, "Control: restarts the state chain instead of preserving continuity."),
    OperatorSpec("turn_order_shuffle_committer", "Turn Order Shuffle Committer", "unsafe_control", "unsafe", 0.04, "Unsafe control: commits correct spans in the wrong turn order."),
    OperatorSpec("continuity_echo_clone", "Continuity Echo Clone", "T-Stab", "redundant", 0.18, "Redundant continuity stabilizer without unique value."),
)


OPERATOR_LIBRARY = USEFUL_OPERATORS + CONTROL_OPERATORS
USEFUL_IDS = tuple(operator.operator_id for operator in USEFUL_OPERATORS)
ALL_OPERATOR_IDS = tuple(operator.operator_id for operator in OPERATOR_LIBRARY)
OPERATOR_BY_ID = {operator.operator_id: operator for operator in OPERATOR_LIBRARY}
UNSAFE_IDS = {operator.operator_id for operator in CONTROL_OPERATORS if operator.role == "unsafe"}


@dataclass(frozen=True)
class MultiTurnCase:
    case_id: str
    source_split: str
    family: str
    turns: tuple[str, ...]
    pending_dependencies: tuple[str, ...]
    expected_action: str
    expected_final_answer: str
    expected_ground_deltas: tuple[str, ...]
    expected_trace: tuple[str, ...]
    required_operators: tuple[str, ...]
    answerable_final: bool
    stale_turn_present: bool
    cross_thread_present: bool
    unresolved_remaining: bool
    order_sensitive: bool


def stable_int(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def split_for(seed: int, case_id: str) -> str:
    bucket = stable_int(f"{seed}:{case_id}") % 10
    if bucket < 7:
        return "train"
    if bucket < 9:
        return "validation"
    return "adversarial"


def make_case(seed: int, index: int) -> MultiTurnCase:
    family = (
        "two_step_dependency_chain",
        "three_step_source_then_state",
        "interleaved_irrelevant_turn",
        "stale_cycle_update_rejected",
        "partial_chain_hold_unresolved",
        "cross_thread_contamination_rejected",
        "numeric_then_source_chain",
        "repeated_question_new_cycle",
        "out_of_order_clarification_reorder",
        "final_answer_after_all_repaired",
    )[index % 10]
    case_id = f"{family}_{seed}_{index:05d}"
    split = split_for(seed, case_id)
    base = ("turn_boundary_cycle_lens", "pending_dependency_stack_t_stab", "active_turn_state_router_guard")
    delta = ("multi_turn_ground_delta_scribe", "clarification_chain_join_lens", "unresolved_state_carry_scribe")
    final = ("cross_turn_stale_context_guard", "final_turn_answer_continuity_guard")

    if family == "two_step_dependency_chain":
        turns = ("t1 ASK valve K state", "t2 c1 valve K closed", "t3 c2 source verified")
        trace = ("cycle:t1", "carry:valve K state+source", "repair:t2:state", "repair:t3:source", "answer")
        return MultiTurnCase(case_id, split, family, turns, ("valve K state", "source"), "ANSWER", "verified valve K state is closed", ("state=closed", "source=verified"), trace, base + delta + final, True, False, False, False, True)
    if family == "three_step_source_then_state":
        turns = ("t1 ASK verified state", "t2 c1 source verified", "t3 c2 valve K closed")
        trace = ("cycle:t1", "repair:t2:source", "carry:state", "repair:t3:state", "answer")
        return MultiTurnCase(case_id, split, family, turns, ("source", "valve K state"), "ANSWER", "verified valve K state is closed", ("source=verified", "state=closed"), trace, base + delta + final, True, False, False, False, True)
    if family == "interleaved_irrelevant_turn":
        turns = ("t1 ASK valve K state", "t2 c1 valve J closed", "t3 c2 valve K closed")
        trace = ("cycle:t1", "reject:t2:irrelevant", "carry:valve K state", "repair:t3:state", "answer")
        return MultiTurnCase(case_id, split, family, turns, ("valve K state",), "ANSWER", "valve K is closed", ("state=closed",), trace, base + delta + final, True, False, False, False, True)
    if family == "stale_cycle_update_rejected":
        turns = ("t1 ASK valve K state", "t2 c1 valve K closed", "t3 stale previous cycle says open")
        trace = ("cycle:t1", "repair:t2:state", "reject:t3:stale", "answer")
        return MultiTurnCase(case_id, split, family, turns, ("valve K state",), "ANSWER", "valve K is closed", ("state=closed",), trace, base + ("multi_turn_ground_delta_scribe", "cross_turn_stale_context_guard", "final_turn_answer_continuity_guard"), True, True, False, False, True)
    if family == "partial_chain_hold_unresolved":
        turns = ("t1 ASK state+source", "t2 c1 valve K closed")
        trace = ("cycle:t1", "repair:t2:state", "carry:source", "ask_remaining")
        return MultiTurnCase(case_id, split, family, turns, ("valve K state", "source"), "ASK_FOR_EVIDENCE", "", ("state=closed",), trace, base + ("multi_turn_ground_delta_scribe", "unresolved_state_carry_scribe", "final_turn_answer_continuity_guard"), False, False, False, True, True)
    if family == "cross_thread_contamination_rejected":
        turns = ("t1 ASK valve K state", "t2 other thread says valve K open", "t3 c1 valve K closed")
        trace = ("cycle:t1", "reject:t2:cross_thread", "repair:t3:state", "answer")
        return MultiTurnCase(case_id, split, family, turns, ("valve K state",), "ANSWER", "valve K is closed", ("state=closed",), trace, base + delta + final, True, False, True, False, True)
    if family == "numeric_then_source_chain":
        turns = ("t1 ASK visible calc+source", "t2 c1 <<7*6=42>>", "t3 c2 source verified")
        trace = ("cycle:t1", "repair:t2:calc", "repair:t3:source", "answer")
        return MultiTurnCase(case_id, split, family, turns, ("visible calc result", "source"), "ANSWER", "verified visible calc trace shows 42", ("calc=42", "source=verified"), trace, base + delta + final, True, False, False, False, True)
    if family == "repeated_question_new_cycle":
        turns = ("t1 ASK valve K state", "t2 c1 valve K closed", "t3 NEW ASK valve K state", "t4 c2 valve K open")
        trace = ("cycle:t1", "repair:t2:closed", "new_cycle:t3", "repair:t4:open", "answer_new_cycle")
        return MultiTurnCase(case_id, split, family, turns, ("latest valve K state",), "ANSWER", "latest valve K state is open", ("state=open",), trace, base + delta + final, True, True, False, False, True)
    if family == "out_of_order_clarification_reorder":
        turns = ("t1 ASK source+state", "t3 c2 valve K closed arrives", "t2 c1 source verified arrives")
        trace = ("cycle:t1", "repair:state", "repair:source", "ordered_join", "answer")
        return MultiTurnCase(case_id, split, family, turns, ("source", "valve K state"), "ANSWER", "verified valve K state is closed", ("state=closed", "source=verified"), trace, base + delta + final, True, False, False, False, True)
    turns = ("t1 ASK verified state", "t2 c1 valve K closed", "t3 c2 source verified", "t4 final check no change")
    trace = ("cycle:t1", "repair:t2:state", "repair:t3:source", "stable:t4", "answer")
    return MultiTurnCase(case_id, split, family, turns, ("valve K state", "source"), "ANSWER", "verified valve K state is closed", ("state=closed", "source=verified"), trace, base + delta + final, True, False, False, False, True)


def generate_cases(seed: int, rows: int) -> list[MultiTurnCase]:
    return [make_case(seed, index) for index in range(rows)]


def unsafe_prediction(case: MultiTurnCase, selected: set[str]) -> dict[str, object] | None:
    if "single_turn_memory_reset_control" in selected and len(case.turns) > 1:
        return {"action": "ASK_FOR_EVIDENCE", "final_answer": "", "ground_deltas": tuple(), "trace": ("reset_each_turn",), "pending_stack_valid": False, "turn_order_valid": True, "ground_continuity_valid": False, "trace_valid": False, "cross_turn_contamination": False, "stale_dependency_reuse": False, "premature_answer": False, "false_restart": True}
    if "latest_turn_only_binder" in selected and case.answerable_final:
        return {"action": "ANSWER", "final_answer": "latest turn says closed", "ground_deltas": ("latest_only",), "trace": ("latest_only",), "pending_stack_valid": False, "turn_order_valid": False, "ground_continuity_valid": False, "trace_valid": False, "cross_turn_contamination": False, "stale_dependency_reuse": False, "premature_answer": False, "false_restart": False}
    if "stale_dependency_reuse_control" in selected and case.stale_turn_present:
        return {"action": "ANSWER", "final_answer": "stale valve K state reused", "ground_deltas": ("stale_state",), "trace": ("stale_reuse",), "pending_stack_valid": False, "turn_order_valid": False, "ground_continuity_valid": False, "trace_valid": False, "cross_turn_contamination": False, "stale_dependency_reuse": True, "premature_answer": False, "false_restart": False}
    if "cross_thread_contamination_control" in selected and case.cross_thread_present:
        return {"action": "ANSWER", "final_answer": "other thread state committed", "ground_deltas": ("cross_thread",), "trace": ("cross_thread",), "pending_stack_valid": False, "turn_order_valid": True, "ground_continuity_valid": False, "trace_valid": False, "cross_turn_contamination": True, "stale_dependency_reuse": False, "premature_answer": False, "false_restart": False}
    if "answer_before_all_turns_resolved" in selected and case.unresolved_remaining:
        return {"action": "ANSWER", "final_answer": "partial answer", "ground_deltas": case.expected_ground_deltas, "trace": ("premature_answer",), "pending_stack_valid": False, "turn_order_valid": True, "ground_continuity_valid": False, "trace_valid": False, "cross_turn_contamination": False, "stale_dependency_reuse": False, "premature_answer": True, "false_restart": False}
    if "always_restart_dialogue_control" in selected:
        return {"action": "ASK_FOR_EVIDENCE", "final_answer": "", "ground_deltas": tuple(), "trace": ("restart",), "pending_stack_valid": False, "turn_order_valid": True, "ground_continuity_valid": False, "trace_valid": False, "cross_turn_contamination": False, "stale_dependency_reuse": False, "premature_answer": False, "false_restart": True}
    if "turn_order_shuffle_committer" in selected and case.order_sensitive:
        return {"action": case.expected_action, "final_answer": case.expected_final_answer, "ground_deltas": tuple(reversed(case.expected_ground_deltas)), "trace": tuple(reversed(case.expected_trace)), "pending_stack_valid": True, "turn_order_valid": False, "ground_continuity_valid": False, "trace_valid": False, "cross_turn_contamination": False, "stale_dependency_reuse": False, "premature_answer": False, "false_restart": False}
    return None


def predict(case: MultiTurnCase, selected: set[str]) -> dict[str, object]:
    unsafe = unsafe_prediction(case, selected)
    if unsafe:
        return unsafe
    required = set(case.required_operators)
    if required <= selected:
        return {"action": case.expected_action, "final_answer": case.expected_final_answer, "ground_deltas": case.expected_ground_deltas, "trace": case.expected_trace, "pending_stack_valid": True, "turn_order_valid": True, "ground_continuity_valid": True, "trace_valid": True, "cross_turn_contamination": False, "stale_dependency_reuse": False, "premature_answer": False, "false_restart": False}
    if case.unresolved_remaining and "unresolved_state_carry_scribe" in selected:
        return {"action": "ASK_FOR_EVIDENCE", "final_answer": "", "ground_deltas": case.expected_ground_deltas, "trace": ("carry_unresolved",), "pending_stack_valid": True, "turn_order_valid": True, "ground_continuity_valid": True, "trace_valid": False, "cross_turn_contamination": False, "stale_dependency_reuse": False, "premature_answer": False, "false_restart": False}
    if case.stale_turn_present and "cross_turn_stale_context_guard" in selected:
        return {"action": case.expected_action, "final_answer": case.expected_final_answer, "ground_deltas": case.expected_ground_deltas, "trace": ("stale_blocked_partial_trace",), "pending_stack_valid": True, "turn_order_valid": True, "ground_continuity_valid": True, "trace_valid": False, "cross_turn_contamination": False, "stale_dependency_reuse": False, "premature_answer": False, "false_restart": False}
    return {"action": "ASK_FOR_EVIDENCE", "final_answer": "", "ground_deltas": tuple(), "trace": ("continuity_incomplete",), "pending_stack_valid": False, "turn_order_valid": False, "ground_continuity_valid": False, "trace_valid": False, "cross_turn_contamination": False, "stale_dependency_reuse": False, "premature_answer": False, "false_restart": case.answerable_final}


def evaluate(selected: set[str], cases: list[MultiTurnCase], split: str | None = None) -> dict[str, float]:
    rows = [case for case in cases if split is None or case.source_split == split]
    if not rows:
        return {"multi_turn_continuity_success": 0.0, "final_answer_accuracy": 0.0, "pending_stack_integrity": 0.0, "turn_order_validity": 0.0, "ground_continuity_validity": 0.0, "trace_chain_integrity": 0.0, "cross_turn_contamination_rate": 0.0, "stale_dependency_reuse_rate": 0.0, "premature_answer_rate": 0.0, "false_restart_rate": 0.0, "utility": -1.0}
    success = answer = stack = order = ground = trace = 0
    contamination = stale = premature = false_restart = 0
    partial = 0.0
    for case in rows:
        pred = predict(case, selected)
        required = set(case.required_operators)
        partial += len(required & selected) / max(1, len(required))
        row_success = (
            pred["action"] == case.expected_action
            and pred["final_answer"] == case.expected_final_answer
            and tuple(pred["ground_deltas"]) == case.expected_ground_deltas
            and tuple(pred["trace"]) == case.expected_trace
            and pred["pending_stack_valid"]
            and pred["turn_order_valid"]
            and pred["ground_continuity_valid"]
            and pred["trace_valid"]
            and not pred["cross_turn_contamination"]
            and not pred["stale_dependency_reuse"]
            and not pred["premature_answer"]
            and not pred["false_restart"]
        )
        success += int(row_success)
        answer += int(pred["final_answer"] == case.expected_final_answer and pred["action"] == case.expected_action)
        stack += int(pred["pending_stack_valid"])
        order += int(pred["turn_order_valid"])
        ground += int(pred["ground_continuity_valid"] and tuple(pred["ground_deltas"]) == case.expected_ground_deltas)
        trace += int(pred["trace_valid"] and tuple(pred["trace"]) == case.expected_trace)
        contamination += int(pred["cross_turn_contamination"])
        stale += int(pred["stale_dependency_reuse"])
        premature += int(pred["premature_answer"])
        false_restart += int(pred["false_restart"])
    count = len(rows)
    cost = sum(OPERATOR_BY_ID[operator_id].cost for operator_id in selected)
    score = success / count
    partial_score = partial / count
    utility = score + 0.28 * partial_score - 1.5 * (contamination / count) - 1.3 * (stale / count) - 1.0 * (premature / count) - 0.8 * (false_restart / count) - 0.01 * cost
    return {
        "multi_turn_continuity_success": round(score, 6),
        "final_answer_accuracy": round(answer / count, 6),
        "pending_stack_integrity": round(stack / count, 6),
        "turn_order_validity": round(order / count, 6),
        "ground_continuity_validity": round(ground / count, 6),
        "trace_chain_integrity": round(trace / count, 6),
        "cross_turn_contamination_rate": round(contamination / count, 6),
        "stale_dependency_reuse_rate": round(stale / count, 6),
        "premature_answer_rate": round(premature / count, 6),
        "false_restart_rate": round(false_restart / count, 6),
        "partial_continuity_score": round(partial_score, 6),
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
        contamination_deltas = []
        for result in seed_results:
            selected = set(result["selected"])  # type: ignore[arg-type]
            cases = [MultiTurnCase(**case) for case in result["cases"]]  # type: ignore[arg-type]
            full = evaluate(selected, cases, "validation")
            ablated = evaluate(selected - {operator_id}, cases, "validation")
            losses.append(full["multi_turn_continuity_success"] - ablated["multi_turn_continuity_success"])
            contamination_deltas.append(ablated["cross_turn_contamination_rate"] - full["cross_turn_contamination_rate"])
        summary[operator_id] = {
            "mean_multi_turn_continuity_success_loss": round(statistics.mean(losses), 6),
            "mean_cross_turn_contamination_delta": round(statistics.mean(contamination_deltas), 6),
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
        "validation_multi_turn_continuity_success_min": min(vals("validation", "multi_turn_continuity_success")),
        "validation_multi_turn_continuity_success_mean": round(statistics.mean(vals("validation", "multi_turn_continuity_success")), 6),
        "adversarial_multi_turn_continuity_success_min": min(vals("adversarial", "multi_turn_continuity_success")),
        "adversarial_multi_turn_continuity_success_mean": round(statistics.mean(vals("adversarial", "multi_turn_continuity_success")), 6),
        "validation_final_answer_accuracy_min": min(vals("validation", "final_answer_accuracy")),
        "validation_pending_stack_integrity_min": min(vals("validation", "pending_stack_integrity")),
        "validation_turn_order_validity_min": min(vals("validation", "turn_order_validity")),
        "validation_ground_continuity_validity_min": min(vals("validation", "ground_continuity_validity")),
        "validation_trace_chain_integrity_min": min(vals("validation", "trace_chain_integrity")),
        "adversarial_cross_turn_contamination_rate_max": max(vals("adversarial", "cross_turn_contamination_rate")),
        "adversarial_stale_dependency_reuse_rate_max": max(vals("adversarial", "stale_dependency_reuse_rate")),
        "adversarial_premature_answer_rate_max": max(vals("adversarial", "premature_answer_rate")),
        "adversarial_false_restart_rate_max": max(vals("adversarial", "false_restart_rate")),
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
        "validation_multi_turn_continuity_success_min",
        "adversarial_multi_turn_continuity_success_min",
        "validation_final_answer_accuracy_min",
        "validation_pending_stack_integrity_min",
        "validation_turn_order_validity_min",
        "validation_ground_continuity_validity_min",
        "validation_trace_chain_integrity_min",
    ]:
        if agg[key] != 1.0:
            failures.append(f"{key} below 1.0")
    for key in [
        "adversarial_cross_turn_contamination_rate_max",
        "adversarial_stale_dependency_reuse_rate_max",
        "adversarial_premature_answer_rate_max",
        "adversarial_false_restart_rate_max",
    ]:
        if agg[key] != 0.0:
            failures.append(f"{key} nonzero")
    unsafe_final = sum(1 for result in seed_results for operator_id in result["selected"] if operator_id in UNSAFE_IDS)  # type: ignore[operator]
    if unsafe_final:
        failures.append("unsafe operator selected")
    decision = "e104_multi_turn_evidence_state_continuity_confirmed" if not failures else "e104_multi_turn_evidence_state_continuity_incomplete"
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
        "multi_turn_evidence_state_continuity": True,
        "stateful_dialogue_proxy": True,
        "open_domain_dialogue": False,
        "direct_answer_without_continuity_allowed": False,
        "requires_turn_trace_chain": True,
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
            typed_case = MultiTurnCase(**case)
            pred = predict(typed_case, set(result["selected"]))  # type: ignore[arg-type]
            append_jsonl(out / "row_level_samples.jsonl", {
                "seed": result["seed"],
                "case_id": typed_case.case_id,
                "family": typed_case.family,
                "turns": typed_case.turns,
                "pending_dependencies": typed_case.pending_dependencies,
                "expected_action": typed_case.expected_action,
                "expected_final_answer": typed_case.expected_final_answer,
                "expected_ground_deltas": typed_case.expected_ground_deltas,
                "expected_trace": typed_case.expected_trace,
                "predicted": pred,
            })
            sample_rows += 1
            if sample_rows >= 480:
                break
        if sample_rows >= 480:
            break
    report = [
        "# E104 Multi-Turn Evidence State Continuity Expansion Result",
        "",
        f"decision = `{decision}`",
        "",
        "Boundary: controlled multi-turn evidence-state continuity, not open-domain dialogue.",
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
    parser.add_argument("--out", default="target/pilot_wave/e104_multi_turn_evidence_state_continuity_expansion")
    parser.add_argument("--artifact-sample-dir", default="docs/research/artifact_samples/e104_multi_turn_evidence_state_continuity_expansion")
    parser.add_argument("--seeds", default="110401,110402,110403,110404,110405,110406,110407,110408,110409,110410,110411,110412,110413,110414,110415,110416")
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
        "boundary": "controlled multi-turn evidence-state continuity probe; not open-domain dialogue",
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
