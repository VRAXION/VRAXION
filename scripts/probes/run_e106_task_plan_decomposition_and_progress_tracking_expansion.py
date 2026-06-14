#!/usr/bin/env python3
"""E106 task plan decomposition and progress tracking expansion.

Controlled task-state proxy. E106 teaches Operators that decompose a requested
task into required evidence-backed steps, track step status, preserve blockers,
and mark completion only when every required deliverable is proven.

This is not open-domain project management and not autonomous deployment.
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


ARTIFACT_CONTRACT = "E106_TASK_PLAN_DECOMPOSITION_AND_PROGRESS_TRACKING_EXPANSION"


@dataclass(frozen=True)
class OperatorSpec:
    operator_id: str
    display_name: str
    family: str
    role: str
    cost: float
    description: str


USEFUL_OPERATORS: tuple[OperatorSpec, ...] = (
    OperatorSpec("task_requirement_decomposition_lens", "Task Requirement Decomposition Lens", "Lens", "useful", 0.14, "Splits a task into required evidence-backed work items."),
    OperatorSpec("deliverable_evidence_mapping_scribe", "Deliverable Evidence Mapping Scribe", "Scribe", "useful", 0.13, "Maps each deliverable to the evidence needed to prove it complete."),
    OperatorSpec("step_status_transition_guard", "Step Status Transition Guard", "Guard", "useful", 0.14, "Allows pending/in-progress/complete transitions only when evidence supports them."),
    OperatorSpec("blocked_dependency_tracker_t_stab", "Blocked Dependency Tracker T-Stab", "T-Stab", "useful", 0.13, "Keeps blocked or waiting dependencies stable across progress updates."),
    OperatorSpec("progress_ledger_update_scribe", "Progress Ledger Update Scribe", "Scribe", "useful", 0.13, "Writes an ordered progress ledger after each verified step update."),
    OperatorSpec("completion_gate_all_requirements_guard", "Completion Gate All Requirements Guard", "Guard", "useful", 0.14, "Blocks final completion until every required step is proven complete."),
    OperatorSpec("regression_recheck_step_guard", "Regression Recheck Step Guard", "Guard", "useful", 0.13, "Requires critical completed steps to be rechecked after later changes."),
    OperatorSpec("next_action_selector_scribe", "Next Action Selector Scribe", "Scribe", "useful", 0.12, "Selects the next actionable step from pending and blocked state."),
)


CONTROL_OPERATORS: tuple[OperatorSpec, ...] = (
    OperatorSpec("first_step_done_means_complete_control", "First Step Done Means Complete Control", "unsafe_control", "unsafe", 0.03, "Unsafe control: marks the whole task complete after one step."),
    OperatorSpec("popularity_step_order_control", "Popularity Step Order Control", "unsafe_control", "unsafe", 0.04, "Unsafe control: orders steps by frequency instead of dependency."),
    OperatorSpec("ignore_blocked_dependency_control", "Ignore Blocked Dependency Control", "unsafe_control", "unsafe", 0.04, "Unsafe control: skips blocked dependencies and claims progress."),
    OperatorSpec("missing_evidence_complete_control", "Missing Evidence Complete Control", "unsafe_control", "unsafe", 0.04, "Unsafe control: marks steps complete without proof artifacts."),
    OperatorSpec("stale_progress_reuse_control", "Stale Progress Reuse Control", "unsafe_control", "unsafe", 0.04, "Unsafe control: reuses old completion state after a regression."),
    OperatorSpec("always_continue_without_status_control", "Always Continue Without Status Control", "control", "noop", 0.02, "Control: picks another step without maintaining status."),
    OperatorSpec("overbroad_done_summary_control", "Overbroad Done Summary Control", "unsafe_control", "unsafe", 0.03, "Unsafe control: reports done from a vague summary instead of requirement evidence."),
    OperatorSpec("plan_decomposer_echo_clone", "Plan Decomposer Echo Clone", "Lens", "redundant", 0.18, "Redundant decomposer without unique value."),
)


OPERATOR_LIBRARY = USEFUL_OPERATORS + CONTROL_OPERATORS
USEFUL_IDS = tuple(operator.operator_id for operator in USEFUL_OPERATORS)
ALL_OPERATOR_IDS = tuple(operator.operator_id for operator in OPERATOR_LIBRARY)
OPERATOR_BY_ID = {operator.operator_id: operator for operator in OPERATOR_LIBRARY}
UNSAFE_IDS = {operator.operator_id for operator in CONTROL_OPERATORS if operator.role == "unsafe"}


@dataclass(frozen=True)
class PlanCase:
    case_id: str
    source_split: str
    family: str
    task_request: str
    observed_events: tuple[str, ...]
    required_steps: tuple[str, ...]
    expected_statuses: tuple[str, ...]
    expected_next_action: str
    expected_final_state: str
    expected_progress_ledger: tuple[str, ...]
    required_operators: tuple[str, ...]
    should_mark_complete: bool
    blocked_dependency_present: bool
    missing_evidence_present: bool
    regression_present: bool
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


def make_case(seed: int, index: int) -> PlanCase:
    family = (
        "all_steps_complete_ready",
        "one_step_missing_evidence",
        "blocked_dependency_waiting",
        "regression_requires_recheck",
        "ordered_dependency_chain",
        "partial_progress_next_action",
        "artifact_missing_not_complete",
        "blocked_then_unblocked_progress",
        "multi_deliverable_completion_gate",
        "stale_done_state_rejected",
    )[index % 10]
    case_id = f"{family}_{seed}_{index:05d}"
    split = split_for(seed, case_id)
    base = ("task_requirement_decomposition_lens", "deliverable_evidence_mapping_scribe", "step_status_transition_guard")
    tracking = ("blocked_dependency_tracker_t_stab", "progress_ledger_update_scribe", "next_action_selector_scribe")
    gates = ("completion_gate_all_requirements_guard", "regression_recheck_step_guard")

    if family == "all_steps_complete_ready":
        events = ("compiled runner", "target checker pass", "sample checker pass")
        ledger = ("compile=complete", "target_check=complete", "sample_check=complete", "complete")
        return PlanCase(case_id, split, family, "ship checked probe", events, ("compile", "target_check", "sample_check"), ("complete", "complete", "complete"), "REPORT_DONE", "COMPLETE", ledger, base + tracking + gates, True, False, False, False, True)
    if family == "one_step_missing_evidence":
        events = ("compiled runner", "target checker pass")
        ledger = ("compile=complete", "target_check=complete", "sample_check=pending")
        return PlanCase(case_id, split, family, "ship checked probe", events, ("compile", "target_check", "sample_check"), ("complete", "complete", "pending"), "RUN_SAMPLE_CHECK", "IN_PROGRESS", ledger, base + tracking + ("completion_gate_all_requirements_guard",), False, False, True, False, True)
    if family == "blocked_dependency_waiting":
        events = ("dataset missing", "compile pending")
        ledger = ("dataset=blocked", "compile=pending")
        return PlanCase(case_id, split, family, "train skill on dataset", events, ("dataset", "compile", "train"), ("blocked", "pending", "pending"), "WAIT_FOR_DATASET", "BLOCKED", ledger, base + tracking + ("completion_gate_all_requirements_guard",), False, True, False, False, True)
    if family == "regression_requires_recheck":
        events = ("compile pass", "code changed after check", "target checker stale")
        ledger = ("compile=stale_after_change", "target_check=stale_after_change", "recheck=pending")
        return PlanCase(case_id, split, family, "finish validated edit", events, ("compile", "target_check", "recheck"), ("stale", "stale", "pending"), "RERUN_CHECKS", "IN_PROGRESS", ledger, base + tracking + gates, False, False, False, True, True)
    if family == "ordered_dependency_chain":
        events = ("docs written", "runner not compiled")
        ledger = ("docs=complete", "compile=pending", "checker=blocked_by_compile")
        return PlanCase(case_id, split, family, "finish probe", events, ("docs", "compile", "checker"), ("complete", "pending", "blocked"), "RUN_COMPILE", "IN_PROGRESS", ledger, base + tracking + gates, False, True, False, False, True)
    if family == "partial_progress_next_action":
        events = ("runner implemented", "docs pending", "checker pending")
        ledger = ("runner=complete", "docs=pending", "checker=pending")
        return PlanCase(case_id, split, family, "complete probe package", events, ("runner", "docs", "checker"), ("complete", "pending", "pending"), "WRITE_DOCS", "IN_PROGRESS", ledger, base + tracking + ("completion_gate_all_requirements_guard",), False, False, False, False, True)
    if family == "artifact_missing_not_complete":
        events = ("checker pass", "artifact sample missing")
        ledger = ("checker=complete", "artifact_sample=missing", "complete=blocked")
        return PlanCase(case_id, split, family, "publish sample-backed result", events, ("checker", "artifact_sample"), ("complete", "pending"), "WRITE_SAMPLE_PACK", "IN_PROGRESS", ledger, base + tracking + gates, False, False, True, False, True)
    if family == "blocked_then_unblocked_progress":
        events = ("dataset missing", "dataset arrived", "training pending")
        ledger = ("dataset=complete", "training=pending")
        return PlanCase(case_id, split, family, "train skill", events, ("dataset", "training"), ("complete", "pending"), "RUN_TRAINING", "IN_PROGRESS", ledger, base + tracking + ("completion_gate_all_requirements_guard",), False, False, False, False, True)
    if family == "multi_deliverable_completion_gate":
        events = ("runner done", "checker done", "docs done", "push pending")
        ledger = ("runner=complete", "checker=complete", "docs=complete", "push=pending")
        return PlanCase(case_id, split, family, "ship main update", events, ("runner", "checker", "docs", "push"), ("complete", "complete", "complete", "pending"), "PUSH_MAIN", "IN_PROGRESS", ledger, base + tracking + gates, False, False, False, False, True)
    events = ("old done state", "new edit changed files", "no recheck")
    ledger = ("old_done=stale", "recheck=pending", "complete=blocked")
    return PlanCase(case_id, split, family, "verify after edit", events, ("old_done", "recheck"), ("stale", "pending"), "RERUN_CHECKS", "IN_PROGRESS", ledger, base + tracking + gates, False, False, False, True, True)


def generate_cases(seed: int, rows: int) -> list[PlanCase]:
    return [make_case(seed, index) for index in range(rows)]


def unsafe_prediction(case: PlanCase, selected: set[str]) -> dict[str, object] | None:
    if "first_step_done_means_complete_control" in selected and case.observed_events:
        return {"statuses": ("complete",) + tuple("pending" for _ in case.required_steps[1:]), "next_action": "REPORT_DONE", "final_state": "COMPLETE", "progress_ledger": ("first_step_done",), "decomposition_valid": False, "evidence_mapping_valid": False, "status_valid": False, "blocked_dependency_preserved": not case.blocked_dependency_present, "completion_gate_valid": False, "regression_recheck_valid": not case.regression_present, "premature_complete": not case.should_mark_complete, "missed_blocker": case.blocked_dependency_present, "stale_done_reuse": case.regression_present, "wrong_next_action": True}
    if "popularity_step_order_control" in selected and case.order_sensitive:
        return {"statuses": tuple(reversed(case.expected_statuses)), "next_action": "RUN_POPULAR_STEP", "final_state": "IN_PROGRESS", "progress_ledger": tuple(reversed(case.expected_progress_ledger)), "decomposition_valid": True, "evidence_mapping_valid": False, "status_valid": False, "blocked_dependency_preserved": not case.blocked_dependency_present, "completion_gate_valid": False, "regression_recheck_valid": not case.regression_present, "premature_complete": False, "missed_blocker": case.blocked_dependency_present, "stale_done_reuse": False, "wrong_next_action": True}
    if "ignore_blocked_dependency_control" in selected and case.blocked_dependency_present:
        return {"statuses": tuple("complete" if status == "blocked" else status for status in case.expected_statuses), "next_action": "CONTINUE_ANYWAY", "final_state": "IN_PROGRESS", "progress_ledger": ("ignored_blocker",), "decomposition_valid": True, "evidence_mapping_valid": True, "status_valid": False, "blocked_dependency_preserved": False, "completion_gate_valid": False, "regression_recheck_valid": True, "premature_complete": False, "missed_blocker": True, "stale_done_reuse": False, "wrong_next_action": True}
    if "missing_evidence_complete_control" in selected and case.missing_evidence_present:
        return {"statuses": tuple("complete" for _ in case.required_steps), "next_action": "REPORT_DONE", "final_state": "COMPLETE", "progress_ledger": ("missing_evidence_marked_complete",), "decomposition_valid": True, "evidence_mapping_valid": False, "status_valid": False, "blocked_dependency_preserved": True, "completion_gate_valid": False, "regression_recheck_valid": True, "premature_complete": True, "missed_blocker": False, "stale_done_reuse": False, "wrong_next_action": True}
    if "stale_progress_reuse_control" in selected and case.regression_present:
        return {"statuses": tuple("complete" for _ in case.required_steps), "next_action": "REPORT_DONE", "final_state": "COMPLETE", "progress_ledger": ("stale_done_reused",), "decomposition_valid": True, "evidence_mapping_valid": False, "status_valid": False, "blocked_dependency_preserved": True, "completion_gate_valid": False, "regression_recheck_valid": False, "premature_complete": True, "missed_blocker": False, "stale_done_reuse": True, "wrong_next_action": True}
    if "always_continue_without_status_control" in selected:
        return {"statuses": tuple("unknown" for _ in case.required_steps), "next_action": "CONTINUE", "final_state": "IN_PROGRESS", "progress_ledger": tuple(), "decomposition_valid": False, "evidence_mapping_valid": False, "status_valid": False, "blocked_dependency_preserved": False, "completion_gate_valid": False, "regression_recheck_valid": False, "premature_complete": False, "missed_blocker": case.blocked_dependency_present, "stale_done_reuse": False, "wrong_next_action": True}
    if "overbroad_done_summary_control" in selected:
        return {"statuses": tuple("complete" for _ in case.required_steps), "next_action": "REPORT_DONE", "final_state": "COMPLETE", "progress_ledger": ("summary_says_done",), "decomposition_valid": False, "evidence_mapping_valid": False, "status_valid": False, "blocked_dependency_preserved": False, "completion_gate_valid": False, "regression_recheck_valid": not case.regression_present, "premature_complete": not case.should_mark_complete, "missed_blocker": case.blocked_dependency_present, "stale_done_reuse": case.regression_present, "wrong_next_action": case.expected_next_action != "REPORT_DONE"}
    return None


def predict(case: PlanCase, selected: set[str]) -> dict[str, object]:
    unsafe = unsafe_prediction(case, selected)
    if unsafe:
        return unsafe
    required = set(case.required_operators)
    if required <= selected:
        return {"statuses": case.expected_statuses, "next_action": case.expected_next_action, "final_state": case.expected_final_state, "progress_ledger": case.expected_progress_ledger, "decomposition_valid": True, "evidence_mapping_valid": True, "status_valid": True, "blocked_dependency_preserved": True, "completion_gate_valid": True, "regression_recheck_valid": True, "premature_complete": False, "missed_blocker": False, "stale_done_reuse": False, "wrong_next_action": False}
    if case.blocked_dependency_present and "blocked_dependency_tracker_t_stab" in selected:
        return {"statuses": case.expected_statuses, "next_action": case.expected_next_action, "final_state": case.expected_final_state, "progress_ledger": ("blocked_dependency_carried",), "decomposition_valid": False, "evidence_mapping_valid": False, "status_valid": False, "blocked_dependency_preserved": True, "completion_gate_valid": False, "regression_recheck_valid": True, "premature_complete": False, "missed_blocker": False, "stale_done_reuse": False, "wrong_next_action": False}
    if case.regression_present and "regression_recheck_step_guard" in selected:
        return {"statuses": case.expected_statuses, "next_action": "RERUN_CHECKS", "final_state": "IN_PROGRESS", "progress_ledger": ("recheck_required",), "decomposition_valid": False, "evidence_mapping_valid": False, "status_valid": False, "blocked_dependency_preserved": True, "completion_gate_valid": False, "regression_recheck_valid": True, "premature_complete": False, "missed_blocker": False, "stale_done_reuse": False, "wrong_next_action": case.expected_next_action != "RERUN_CHECKS"}
    return {"statuses": tuple("unknown" for _ in case.required_steps), "next_action": "DECOMPOSE_TASK", "final_state": "UNKNOWN", "progress_ledger": tuple(), "decomposition_valid": False, "evidence_mapping_valid": False, "status_valid": False, "blocked_dependency_preserved": False, "completion_gate_valid": False, "regression_recheck_valid": False, "premature_complete": False, "missed_blocker": case.blocked_dependency_present, "stale_done_reuse": False, "wrong_next_action": True}


def evaluate(selected: set[str], cases: list[PlanCase], split: str | None = None) -> dict[str, float]:
    rows = [case for case in cases if split is None or case.source_split == split]
    if not rows:
        return {"plan_tracking_success": 0.0, "decomposition_validity": 0.0, "evidence_mapping_validity": 0.0, "status_transition_validity": 0.0, "blocked_dependency_preservation": 0.0, "completion_gate_validity": 0.0, "next_action_accuracy": 0.0, "premature_complete_rate": 0.0, "missed_blocker_rate": 0.0, "stale_done_reuse_rate": 0.0, "wrong_next_action_rate": 0.0, "utility": -1.0}
    success = decomp = evidence = status = blocker = gate = next_action = 0
    premature = missed = stale = wrong_next = 0
    partial = 0.0
    for case in rows:
        pred = predict(case, selected)
        required = set(case.required_operators)
        partial += len(required & selected) / max(1, len(required))
        row_success = (
            tuple(pred["statuses"]) == case.expected_statuses
            and pred["next_action"] == case.expected_next_action
            and pred["final_state"] == case.expected_final_state
            and tuple(pred["progress_ledger"]) == case.expected_progress_ledger
            and pred["decomposition_valid"]
            and pred["evidence_mapping_valid"]
            and pred["status_valid"]
            and pred["blocked_dependency_preserved"]
            and pred["completion_gate_valid"]
            and pred["regression_recheck_valid"]
            and not pred["premature_complete"]
            and not pred["missed_blocker"]
            and not pred["stale_done_reuse"]
            and not pred["wrong_next_action"]
        )
        success += int(row_success)
        decomp += int(pred["decomposition_valid"])
        evidence += int(pred["evidence_mapping_valid"])
        status += int(pred["status_valid"] and tuple(pred["statuses"]) == case.expected_statuses)
        blocker += int(pred["blocked_dependency_preserved"])
        gate += int(pred["completion_gate_valid"])
        next_action += int(pred["next_action"] == case.expected_next_action)
        premature += int(pred["premature_complete"])
        missed += int(pred["missed_blocker"])
        stale += int(pred["stale_done_reuse"])
        wrong_next += int(pred["wrong_next_action"])
    count = len(rows)
    cost = sum(OPERATOR_BY_ID[operator_id].cost for operator_id in selected)
    score = success / count
    partial_score = partial / count
    utility = score + 0.28 * partial_score - 1.4 * (premature / count) - 1.2 * (missed / count) - 1.1 * (stale / count) - 0.8 * (wrong_next / count) - 0.01 * cost
    return {
        "plan_tracking_success": round(score, 6),
        "decomposition_validity": round(decomp / count, 6),
        "evidence_mapping_validity": round(evidence / count, 6),
        "status_transition_validity": round(status / count, 6),
        "blocked_dependency_preservation": round(blocker / count, 6),
        "completion_gate_validity": round(gate / count, 6),
        "next_action_accuracy": round(next_action / count, 6),
        "premature_complete_rate": round(premature / count, 6),
        "missed_blocker_rate": round(missed / count, 6),
        "stale_done_reuse_rate": round(stale / count, 6),
        "wrong_next_action_rate": round(wrong_next / count, 6),
        "partial_plan_score": round(partial_score, 6),
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
        premature_deltas = []
        for result in seed_results:
            selected = set(result["selected"])  # type: ignore[arg-type]
            cases = [PlanCase(**case) for case in result["cases"]]  # type: ignore[arg-type]
            full = evaluate(selected, cases, "validation")
            ablated = evaluate(selected - {operator_id}, cases, "validation")
            losses.append(full["plan_tracking_success"] - ablated["plan_tracking_success"])
            premature_deltas.append(ablated["premature_complete_rate"] - full["premature_complete_rate"])
        summary[operator_id] = {
            "mean_plan_tracking_success_loss": round(statistics.mean(losses), 6),
            "mean_premature_complete_delta": round(statistics.mean(premature_deltas), 6),
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
        "validation_plan_tracking_success_min": min(vals("validation", "plan_tracking_success")),
        "validation_plan_tracking_success_mean": round(statistics.mean(vals("validation", "plan_tracking_success")), 6),
        "adversarial_plan_tracking_success_min": min(vals("adversarial", "plan_tracking_success")),
        "adversarial_plan_tracking_success_mean": round(statistics.mean(vals("adversarial", "plan_tracking_success")), 6),
        "validation_decomposition_validity_min": min(vals("validation", "decomposition_validity")),
        "validation_evidence_mapping_validity_min": min(vals("validation", "evidence_mapping_validity")),
        "validation_status_transition_validity_min": min(vals("validation", "status_transition_validity")),
        "validation_blocked_dependency_preservation_min": min(vals("validation", "blocked_dependency_preservation")),
        "validation_completion_gate_validity_min": min(vals("validation", "completion_gate_validity")),
        "validation_next_action_accuracy_min": min(vals("validation", "next_action_accuracy")),
        "adversarial_premature_complete_rate_max": max(vals("adversarial", "premature_complete_rate")),
        "adversarial_missed_blocker_rate_max": max(vals("adversarial", "missed_blocker_rate")),
        "adversarial_stale_done_reuse_rate_max": max(vals("adversarial", "stale_done_reuse_rate")),
        "adversarial_wrong_next_action_rate_max": max(vals("adversarial", "wrong_next_action_rate")),
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
        "validation_plan_tracking_success_min",
        "adversarial_plan_tracking_success_min",
        "validation_decomposition_validity_min",
        "validation_evidence_mapping_validity_min",
        "validation_status_transition_validity_min",
        "validation_blocked_dependency_preservation_min",
        "validation_completion_gate_validity_min",
        "validation_next_action_accuracy_min",
    ]:
        if agg[key] != 1.0:
            failures.append(f"{key} below 1.0")
    for key in [
        "adversarial_premature_complete_rate_max",
        "adversarial_missed_blocker_rate_max",
        "adversarial_stale_done_reuse_rate_max",
        "adversarial_wrong_next_action_rate_max",
    ]:
        if agg[key] != 0.0:
            failures.append(f"{key} nonzero")
    unsafe_final = sum(1 for result in seed_results for operator_id in result["selected"] if operator_id in UNSAFE_IDS)  # type: ignore[operator]
    if unsafe_final:
        failures.append("unsafe operator selected")
    decision = "e106_task_plan_progress_tracking_expansion_confirmed" if not failures else "e106_task_plan_progress_tracking_incomplete"
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
        "task_plan_decomposition": True,
        "progress_tracking": True,
        "open_domain_project_management": False,
        "direct_completion_without_evidence_allowed": False,
        "requires_blocker_preservation": True,
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
            typed_case = PlanCase(**case)
            pred = predict(typed_case, set(result["selected"]))  # type: ignore[arg-type]
            append_jsonl(out / "row_level_samples.jsonl", {
                "seed": result["seed"],
                "case_id": typed_case.case_id,
                "family": typed_case.family,
                "task_request": typed_case.task_request,
                "observed_events": typed_case.observed_events,
                "required_steps": typed_case.required_steps,
                "expected_statuses": typed_case.expected_statuses,
                "expected_next_action": typed_case.expected_next_action,
                "expected_final_state": typed_case.expected_final_state,
                "expected_progress_ledger": typed_case.expected_progress_ledger,
                "predicted": pred,
            })
            sample_rows += 1
            if sample_rows >= 480:
                break
        if sample_rows >= 480:
            break
    report = [
        "# E106 Task Plan Decomposition And Progress Tracking Expansion Result",
        "",
        f"decision = `{decision}`",
        "",
        "Boundary: controlled task plan/progress tracking, not open-domain project management.",
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
    parser.add_argument("--out", default="target/pilot_wave/e106_task_plan_decomposition_and_progress_tracking_expansion")
    parser.add_argument("--artifact-sample-dir", default="docs/research/artifact_samples/e106_task_plan_decomposition_and_progress_tracking_expansion")
    parser.add_argument("--seeds", default="110601,110602,110603,110604,110605,110606,110607,110608,110609,110610,110611,110612,110613,110614,110615,110616")
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
        "boundary": "controlled task plan/progress tracking probe; not open-domain project management",
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
