#!/usr/bin/env python3
"""E98 multi-skill execution curriculum expansion.

Controlled Operator composition probe. E98 teaches execution hygiene around
already-scoped skills: decompose a composite task, order the required skills,
carry intermediate state, join trace evidence, enforce scope boundaries, and
assemble a final answer only when the chain is valid.

This is not open-domain reasoning and not a deployed model claim.
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


ARTIFACT_CONTRACT = "E98_MULTI_SKILL_EXECUTION_CURRICULUM_EXPANSION"


@dataclass(frozen=True)
class OperatorSpec:
    operator_id: str
    display_name: str
    family: str
    role: str
    cost: float
    description: str


USEFUL_OPERATORS: tuple[OperatorSpec, ...] = (
    OperatorSpec("composite_task_decomposer_lens", "Composite Task Decomposer Lens", "Lens", "useful", 0.14, "Detects the small set of skills needed by a composite task."),
    OperatorSpec("dependency_ordering_scribe", "Dependency Ordering Scribe", "Scribe", "useful", 0.13, "Renders a dependency-correct skill call order."),
    OperatorSpec("intermediate_state_carry_t_stab", "Intermediate State Carry T-Stab", "T-Stab", "useful", 0.14, "Stabilizes intermediate state between skill calls."),
    OperatorSpec("cross_skill_trace_join_guard", "Cross-Skill Trace Join Guard", "Guard", "useful", 0.14, "Requires evidence/provenance continuity across skill boundaries."),
    OperatorSpec("capability_scope_boundary_guard", "Capability Scope Boundary Guard", "Guard", "useful", 0.13, "Blocks a scoped skill from acting outside its capability boundary."),
    OperatorSpec("partial_route_checkpoint_scribe", "Partial Route Checkpoint Scribe", "Scribe", "useful", 0.12, "Writes mechanical checkpoints after each completed skill stage."),
    OperatorSpec("composition_error_recovery_scribe", "Composition Error Recovery Scribe", "Scribe", "useful", 0.12, "Routes invalid composite state to ASK, DEFER, or RETRY instead of forced answer."),
    OperatorSpec("final_response_integrity_guard", "Final Response Integrity Guard", "Guard", "useful", 0.13, "Allows final answer assembly only when route, trace, scope, and intermediate state are valid."),
)


CONTROL_OPERATORS: tuple[OperatorSpec, ...] = (
    OperatorSpec("single_skill_shortcut_solver", "Single-Skill Shortcut Solver", "unsafe_control", "unsafe", 0.04, "Unsafe control: tries to answer a composite task with one skill."),
    OperatorSpec("answer_only_composer", "Answer-Only Composer", "unsafe_control", "unsafe", 0.04, "Unsafe control: emits answer without trace or intermediate validation."),
    OperatorSpec("unordered_skill_bag_runner", "Unordered Skill Bag Runner", "unsafe_control", "unsafe", 0.05, "Unsafe control: calls the right-looking skills in the wrong order."),
    OperatorSpec("drop_intermediate_state_control", "Drop Intermediate State Control", "unsafe_control", "unsafe", 0.04, "Unsafe control: loses outputs between skill calls."),
    OperatorSpec("trace_join_omission_control", "Trace Join Omission Control", "unsafe_control", "unsafe", 0.04, "Unsafe control: omits cross-skill evidence joins."),
    OperatorSpec("scope_bleed_word_solver_control", "Scope-Bleed Word Solver Control", "unsafe_control", "unsafe", 0.04, "Unsafe control: lets visible-trace skills pretend to solve hidden word problems."),
    OperatorSpec("always_successful_completion_control", "Always-Successful Completion Control", "control", "noop", 0.03, "Control: reports completion even when a composite chain is unresolved."),
    OperatorSpec("decomposer_echo_clone", "Decomposer Echo Clone", "Lens", "redundant", 0.19, "Redundant decomposer without unique contribution."),
)


OPERATOR_LIBRARY = USEFUL_OPERATORS + CONTROL_OPERATORS
USEFUL_IDS = tuple(operator.operator_id for operator in USEFUL_OPERATORS)
ALL_OPERATOR_IDS = tuple(operator.operator_id for operator in OPERATOR_LIBRARY)
OPERATOR_BY_ID = {operator.operator_id: operator for operator in OPERATOR_LIBRARY}
UNSAFE_IDS = {operator.operator_id for operator in CONTROL_OPERATORS if operator.role == "unsafe"}


@dataclass(frozen=True)
class CompositeCase:
    case_id: str
    source_split: str
    family: str
    observable_state: str
    expected_action: str
    expected_route: tuple[str, ...]
    required_operators: tuple[str, ...]
    stage_count: int
    expected_checkpoint_count: int
    needs_trace_join: bool
    needs_scope_guard: bool
    needs_intermediate_carry: bool


def stable_int(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def split_for(seed: int, case_id: str) -> str:
    bucket = stable_int(f"{seed}:{case_id}") % 10
    if bucket < 7:
        return "train"
    if bucket < 9:
        return "validation"
    return "adversarial"


def make_case(seed: int, index: int) -> CompositeCase:
    family = (
        "calc_trace_to_hygienic_answer",
        "alpha_sync_then_calc_marker",
        "missing_evidence_then_answer",
        "trace_ground_memory_then_answer",
        "adapter_then_temporal_stream",
        "proposal_conflict_then_defer",
        "scope_bleed_adversarial",
        "unordered_dependency_adversarial",
        "lost_intermediate_state_adversarial",
        "completed_multiskill_halt",
    )[index % 10]
    case_id = f"{family}_{seed}_{index:05d}"
    split = split_for(seed, case_id)

    common = ("composite_task_decomposer_lens", "dependency_ordering_scribe")
    carry = ("intermediate_state_carry_t_stab",)
    trace = ("cross_skill_trace_join_guard", "partial_route_checkpoint_scribe")
    scope = ("capability_scope_boundary_guard",)
    final = ("final_response_integrity_guard",)
    recover = ("composition_error_recovery_scribe",)

    if family == "calc_trace_to_hygienic_answer":
        route = (
            "composite_task_decomposer_lens",
            "dependency_ordering_scribe",
            "calc_scribe_v003",
            "intermediate_state_carry_t_stab",
            "evidence_citation_scribe",
            "unit_preserving_answer_scribe",
            "final_response_integrity_guard",
        )
        return CompositeCase(case_id, split, family, "visible calc marker plus answer hygiene request.", "ANSWER_READY", route, common + carry + trace + final, 4, 3, True, False, True)
    if family == "alpha_sync_then_calc_marker":
        route = (
            "composite_task_decomposer_lens",
            "lexical_alias_alpha_syncer",
            "unicode_operator_normalizer",
            "dependency_ordering_scribe",
            "calc_scribe_v003",
            "intermediate_state_carry_t_stab",
            "final_response_integrity_guard",
        )
        return CompositeCase(case_id, split, family, "aliased operator token must normalize before calc trace validation.", "ANSWER_READY", route, common + carry + trace + final, 4, 3, True, False, True)
    if family == "missing_evidence_then_answer":
        route = (
            "composite_task_decomposer_lens",
            "missing_dependency_locator_lens",
            "targeted_evidence_request_scribe",
            "retrieved_evidence_integrator_t_stab",
            "intermediate_state_carry_t_stab",
            "answer_ready_after_evidence_scribe",
            "final_response_integrity_guard",
        )
        return CompositeCase(case_id, split, family, "query has missing evidence dependency then retrieved evidence arrives.", "ANSWER_READY", route, common + carry + trace + recover + final, 5, 4, True, False, True)
    if family == "trace_ground_memory_then_answer":
        route = (
            "composite_task_decomposer_lens",
            "trace_deduplication_lens",
            "provenance_chain_guard",
            "ground_promotion_candidate_scribe",
            "cross_skill_trace_join_guard",
            "final_response_integrity_guard",
        )
        return CompositeCase(case_id, split, family, "duplicate trace must dedup, preserve provenance, then answer from scoped Ground candidate.", "ANSWER_READY", route, common + trace + final, 4, 3, True, False, False)
    if family == "adapter_then_temporal_stream":
        route = (
            "composite_task_decomposer_lens",
            "adapter_requirement_detector_lens",
            "edge_adapter_operator",
            "frame_sequence_t_stab",
            "crc_parity_frame_guard",
            "intermediate_state_carry_t_stab",
            "final_response_integrity_guard",
        )
        return CompositeCase(case_id, split, family, "cross-ABI temporal stream requires adapter before temporal frame validation.", "ANSWER_READY", route, common + carry + trace + final, 5, 4, True, False, True)
    if family == "proposal_conflict_then_defer":
        route = ("composition_error_recovery_scribe",)
        return CompositeCase(case_id, split, family, "two proposals collide and evidence cannot resolve the conflict yet.", "DEFER", route, recover + trace + final, 2, 1, True, False, False)
    if family == "scope_bleed_adversarial":
        route = ("composition_error_recovery_scribe",)
        return CompositeCase(case_id, split, family, "visible calc marker absent; word-problem-like text tries to lure calc skill outside scope.", "ASK_FOR_EVIDENCE", route, scope + recover + final, 2, 1, False, True, False)
    if family == "unordered_dependency_adversarial":
        route = ("dependency_ordering_scribe", "composition_error_recovery_scribe")
        return CompositeCase(case_id, split, family, "route tries calc before alias normalization; dependency order must be repaired.", "RETRY_ROUTE", route, common + recover + final, 3, 2, True, False, True)
    if family == "lost_intermediate_state_adversarial":
        route = ("intermediate_state_carry_t_stab", "partial_route_checkpoint_scribe", "composition_error_recovery_scribe")
        return CompositeCase(case_id, split, family, "skill output is lost between stages; checkpoint and retry are required.", "RETRY_ROUTE", route, carry + trace + recover + final, 3, 2, True, False, True)
    route = ("composition_completion_t_stab", "final_response_integrity_guard")
    return CompositeCase(case_id, split, family, "all stages complete and trace-valid; no further call should be made.", "HALT", route, ("final_response_integrity_guard",), 1, 1, True, False, False)


def generate_cases(seed: int, rows: int) -> list[CompositeCase]:
    return [make_case(seed, index) for index in range(rows)]


def unsafe_prediction(case: CompositeCase, selected: set[str]) -> dict[str, object] | None:
    if "answer_only_composer" in selected:
        return {
            "action": "ANSWER_READY",
            "route": ("answer_only_composer",),
            "route_valid": False,
            "trace_valid": False,
            "scope_valid": False,
            "intermediate_valid": False,
            "dependency_order_valid": False,
            "checkpoint_count": 0,
            "unsafe_answer": True,
            "scope_bleed": case.needs_scope_guard,
            "unordered_execution": True,
            "dropped_intermediate": case.needs_intermediate_carry,
        }
    if "single_skill_shortcut_solver" in selected and case.stage_count > 1:
        return {
            "action": "ANSWER_READY",
            "route": ("single_skill_shortcut_solver",),
            "route_valid": False,
            "trace_valid": not case.needs_trace_join,
            "scope_valid": not case.needs_scope_guard,
            "intermediate_valid": False,
            "dependency_order_valid": False,
            "checkpoint_count": 0,
            "unsafe_answer": True,
            "scope_bleed": case.needs_scope_guard,
            "unordered_execution": True,
            "dropped_intermediate": case.needs_intermediate_carry,
        }
    if "unordered_skill_bag_runner" in selected and case.expected_action in {"ANSWER_READY", "RETRY_ROUTE"}:
        return {
            "action": case.expected_action,
            "route": tuple(reversed(case.expected_route)),
            "route_valid": False,
            "trace_valid": True,
            "scope_valid": True,
            "intermediate_valid": True,
            "dependency_order_valid": False,
            "checkpoint_count": max(0, case.expected_checkpoint_count - 1),
            "unsafe_answer": False,
            "scope_bleed": False,
            "unordered_execution": True,
            "dropped_intermediate": False,
        }
    if "drop_intermediate_state_control" in selected and case.needs_intermediate_carry:
        return {
            "action": "RETRY_ROUTE",
            "route": ("drop_intermediate_state_control",),
            "route_valid": False,
            "trace_valid": True,
            "scope_valid": True,
            "intermediate_valid": False,
            "dependency_order_valid": True,
            "checkpoint_count": max(0, case.expected_checkpoint_count - 1),
            "unsafe_answer": False,
            "scope_bleed": False,
            "unordered_execution": False,
            "dropped_intermediate": True,
        }
    if "trace_join_omission_control" in selected and case.needs_trace_join:
        return {
            "action": case.expected_action,
            "route": case.expected_route,
            "route_valid": False,
            "trace_valid": False,
            "scope_valid": True,
            "intermediate_valid": True,
            "dependency_order_valid": True,
            "checkpoint_count": case.expected_checkpoint_count,
            "unsafe_answer": case.expected_action == "ANSWER_READY",
            "scope_bleed": False,
            "unordered_execution": False,
            "dropped_intermediate": False,
        }
    if "scope_bleed_word_solver_control" in selected and case.needs_scope_guard:
        return {
            "action": "ANSWER_READY",
            "route": ("scope_bleed_word_solver_control",),
            "route_valid": False,
            "trace_valid": False,
            "scope_valid": False,
            "intermediate_valid": True,
            "dependency_order_valid": False,
            "checkpoint_count": 0,
            "unsafe_answer": True,
            "scope_bleed": True,
            "unordered_execution": False,
            "dropped_intermediate": False,
        }
    if "always_successful_completion_control" in selected and case.expected_action != "HALT":
        return {
            "action": "HALT",
            "route": ("always_successful_completion_control",),
            "route_valid": False,
            "trace_valid": False,
            "scope_valid": True,
            "intermediate_valid": False,
            "dependency_order_valid": False,
            "checkpoint_count": 0,
            "unsafe_answer": False,
            "scope_bleed": False,
            "unordered_execution": False,
            "dropped_intermediate": case.needs_intermediate_carry,
        }
    return None


def predict(case: CompositeCase, selected: set[str]) -> dict[str, object]:
    unsafe = unsafe_prediction(case, selected)
    if unsafe:
        return unsafe
    required = set(case.required_operators)
    if required <= selected:
        return {
            "action": case.expected_action,
            "route": case.expected_route,
            "route_valid": True,
            "trace_valid": True,
            "scope_valid": True,
            "intermediate_valid": True,
            "dependency_order_valid": True,
            "checkpoint_count": case.expected_checkpoint_count,
            "unsafe_answer": False,
            "scope_bleed": False,
            "unordered_execution": False,
            "dropped_intermediate": False,
        }
    if "composition_error_recovery_scribe" in selected:
        return {
            "action": "ASK_FOR_EVIDENCE",
            "route": ("composition_error_recovery_scribe",),
            "route_valid": False,
            "trace_valid": True,
            "scope_valid": True,
            "intermediate_valid": not case.needs_intermediate_carry,
            "dependency_order_valid": True,
            "checkpoint_count": 0,
            "unsafe_answer": False,
            "scope_bleed": False,
            "unordered_execution": False,
            "dropped_intermediate": case.needs_intermediate_carry,
        }
    return {
        "action": "DEFER",
        "route": tuple(),
        "route_valid": False,
        "trace_valid": True,
        "scope_valid": True,
        "intermediate_valid": not case.needs_intermediate_carry,
        "dependency_order_valid": False,
        "checkpoint_count": 0,
        "unsafe_answer": False,
        "scope_bleed": False,
        "unordered_execution": False,
        "dropped_intermediate": case.needs_intermediate_carry,
    }


def evaluate(selected: set[str], cases: list[CompositeCase], split: str | None = None) -> dict[str, float]:
    rows = [case for case in cases if split is None or case.source_split == split]
    if not rows:
        return {
            "composition_success": 0.0,
            "dependency_order_accuracy": 0.0,
            "trace_join_validity": 0.0,
            "intermediate_carry_validity": 0.0,
            "scope_bleed_rate": 0.0,
            "dropped_intermediate_rate": 0.0,
            "unordered_execution_rate": 0.0,
            "unsafe_answer_rate": 0.0,
            "checkpoint_accuracy": 0.0,
            "utility": -1.0,
        }
    success = order = trace = intermediate = checkpoint = 0
    scope_bleed = dropped = unordered = unsafe_answer = 0
    partial = 0.0
    for case in rows:
        pred = predict(case, selected)
        required = set(case.required_operators)
        partial += len(required & selected) / max(1, len(required))
        row_success = (
            pred["action"] == case.expected_action
            and tuple(pred["route"]) == case.expected_route
            and pred["route_valid"]
            and pred["trace_valid"]
            and pred["scope_valid"]
            and pred["intermediate_valid"]
            and pred["dependency_order_valid"]
            and pred["checkpoint_count"] == case.expected_checkpoint_count
        )
        success += int(row_success)
        order += int(pred["dependency_order_valid"] and tuple(pred["route"]) == case.expected_route)
        trace += int(pred["trace_valid"])
        intermediate += int(pred["intermediate_valid"])
        checkpoint += int(pred["checkpoint_count"] == case.expected_checkpoint_count)
        scope_bleed += int(pred["scope_bleed"])
        dropped += int(pred["dropped_intermediate"])
        unordered += int(pred["unordered_execution"])
        unsafe_answer += int(pred["unsafe_answer"])
    count = len(rows)
    cost = sum(OPERATOR_BY_ID[operator_id].cost for operator_id in selected)
    score = success / count
    partial_score = partial / count
    utility = (
        score
        + 0.30 * partial_score
        - 1.4 * (scope_bleed / count)
        - 1.2 * (dropped / count)
        - 1.2 * (unordered / count)
        - 1.6 * (unsafe_answer / count)
        - 0.01 * cost
    )
    return {
        "composition_success": round(score, 6),
        "dependency_order_accuracy": round(order / count, 6),
        "trace_join_validity": round(trace / count, 6),
        "intermediate_carry_validity": round(intermediate / count, 6),
        "checkpoint_accuracy": round(checkpoint / count, 6),
        "scope_bleed_rate": round(scope_bleed / count, 6),
        "dropped_intermediate_rate": round(dropped / count, 6),
        "unordered_execution_rate": round(unordered / count, 6),
        "unsafe_answer_rate": round(unsafe_answer / count, 6),
        "partial_composition_score": round(partial_score, 6),
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


def counterfactual_report(seed_results: list[dict[str, object]]) -> dict[str, object]:
    summary: dict[str, dict[str, float]] = {}
    for operator_id in USEFUL_IDS:
        losses = []
        unsafe_deltas = []
        for result in seed_results:
            selected = set(result["selected"])  # type: ignore[arg-type]
            cases = [CompositeCase(**case) for case in result["cases"]]  # type: ignore[arg-type]
            full = evaluate(selected, cases, "validation")
            ablated = evaluate(selected - {operator_id}, cases, "validation")
            losses.append(full["composition_success"] - ablated["composition_success"])
            unsafe_deltas.append(ablated["unsafe_answer_rate"] - full["unsafe_answer_rate"])
        summary[operator_id] = {
            "mean_composition_success_loss": round(statistics.mean(losses), 6),
            "mean_unsafe_answer_delta": round(statistics.mean(unsafe_deltas), 6),
        }
    return {"summary": summary}


def aggregate(seed_results: list[dict[str, object]], seconds: float) -> dict[str, float]:
    def vals(split: str, key: str) -> list[float]:
        return [float(result["metrics"][split][key]) for result in seed_results]  # type: ignore[index]

    return {
        "seed_count": len(seed_results),
        "validation_composition_success_min": min(vals("validation", "composition_success")),
        "validation_composition_success_mean": round(statistics.mean(vals("validation", "composition_success")), 6),
        "adversarial_composition_success_min": min(vals("adversarial", "composition_success")),
        "adversarial_composition_success_mean": round(statistics.mean(vals("adversarial", "composition_success")), 6),
        "validation_dependency_order_accuracy_min": min(vals("validation", "dependency_order_accuracy")),
        "validation_trace_join_validity_min": min(vals("validation", "trace_join_validity")),
        "validation_intermediate_carry_validity_min": min(vals("validation", "intermediate_carry_validity")),
        "validation_checkpoint_accuracy_min": min(vals("validation", "checkpoint_accuracy")),
        "adversarial_scope_bleed_rate_max": max(vals("adversarial", "scope_bleed_rate")),
        "adversarial_dropped_intermediate_rate_max": max(vals("adversarial", "dropped_intermediate_rate")),
        "adversarial_unordered_execution_rate_max": max(vals("adversarial", "unordered_execution_rate")),
        "adversarial_unsafe_answer_rate_max": max(vals("adversarial", "unsafe_answer_rate")),
        "accepted_mutations_total": sum(int(result["accepted"]) for result in seed_results),
        "rejected_mutations_total": sum(int(result["rejected"]) for result in seed_results),
        "rollback_count_total": sum(int(result["rollback"]) for result in seed_results),
        "seconds": round(seconds, 3),
    }


def deterministic_hash(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def sample_pack(source: Path, target: Path) -> None:
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
    if agg["validation_composition_success_min"] != 1.0:
        failures.append("validation composition success below 1.0")
    if agg["adversarial_composition_success_min"] != 1.0:
        failures.append("adversarial composition success below 1.0")
    if agg["validation_dependency_order_accuracy_min"] != 1.0:
        failures.append("dependency order below 1.0")
    if agg["validation_trace_join_validity_min"] != 1.0:
        failures.append("trace join below 1.0")
    if agg["validation_intermediate_carry_validity_min"] != 1.0:
        failures.append("intermediate carry below 1.0")
    if agg["validation_checkpoint_accuracy_min"] != 1.0:
        failures.append("checkpoint accuracy below 1.0")
    for key in ["adversarial_scope_bleed_rate_max", "adversarial_dropped_intermediate_rate_max", "adversarial_unordered_execution_rate_max", "adversarial_unsafe_answer_rate_max"]:
        if agg[key] != 0.0:
            failures.append(f"{key} nonzero")
    unsafe_final = sum(1 for result in seed_results for operator_id in result["selected"] if operator_id in UNSAFE_IDS)  # type: ignore[operator]
    if unsafe_final:
        failures.append("unsafe operator selected")
    decision = "e98_multi_skill_execution_curriculum_expansion_confirmed" if not failures else "e98_multi_skill_execution_curriculum_incomplete"
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
        "multi_skill_composition": True,
        "open_domain_reasoning": False,
        "direct_answer_shortcut_allowed": False,
        "requires_trace_join": True,
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
            typed_case = CompositeCase(**case)
            pred = predict(typed_case, set(result["selected"]))  # type: ignore[arg-type]
            append_jsonl(out / "row_level_samples.jsonl", {
                "seed": result["seed"],
                "case_id": typed_case.case_id,
                "family": typed_case.family,
                "observable_state": typed_case.observable_state,
                "expected_action": typed_case.expected_action,
                "expected_route": typed_case.expected_route,
                "predicted": pred,
            })
            sample_rows += 1
            if sample_rows >= 480:
                break
        if sample_rows >= 480:
            break
    report = [
        "# E98 Multi-Skill Execution Curriculum Expansion Result",
        "",
        f"decision = `{decision}`",
        "",
        "Boundary: controlled multi-skill Operator composition, not open-domain reasoning.",
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
        sample_pack(out, sample_dir)


def parse_seeds(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="target/pilot_wave/e98_multi_skill_execution_curriculum_expansion")
    parser.add_argument("--artifact-sample-dir", default="docs/research/artifact_samples/e98_multi_skill_execution_curriculum_expansion")
    parser.add_argument("--seeds", default="109801,109802,109803,109804,109805,109806,109807,109808,109809,109810,109811,109812,109813,109814,109815,109816")
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
        "boundary": "controlled multi-skill Operator composition probe; not open-domain reasoning",
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
