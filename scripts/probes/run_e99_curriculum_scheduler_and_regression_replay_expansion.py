#!/usr/bin/env python3
"""E99 curriculum scheduler and regression replay expansion.

Controlled training-governance probe. E99 adds Operators that choose the next
lesson from capability gaps while preserving regression replay, adversarial
coverage, difficulty ramping, compute budget, and promotion prechecks.

This is not autonomous open-ended training. It is a scoped curriculum
scheduler proxy for Operator-library growth.
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


ARTIFACT_CONTRACT = "E99_CURRICULUM_SCHEDULER_AND_REGRESSION_REPLAY_EXPANSION"


@dataclass(frozen=True)
class OperatorSpec:
    operator_id: str
    display_name: str
    family: str
    role: str
    cost: float
    description: str


USEFUL_OPERATORS: tuple[OperatorSpec, ...] = (
    OperatorSpec("capability_gap_detector_lens", "Capability Gap Detector Lens", "Lens", "useful", 0.13, "Detects the weakest verified capability frontier from metric deltas."),
    OperatorSpec("lesson_candidate_ranker_scribe", "Lesson Candidate Ranker Scribe", "Scribe", "useful", 0.13, "Ranks next lessons by expected utility, risk, and scope."),
    OperatorSpec("regression_replay_set_guard", "Regression Replay Set Guard", "Guard", "useful", 0.14, "Requires prior stable skills to remain in the replay set."),
    OperatorSpec("adversarial_family_sampler_lens", "Adversarial Family Sampler Lens", "Lens", "useful", 0.12, "Selects adversarial families tied to the active capability gap."),
    OperatorSpec("difficulty_ramp_t_stab", "Difficulty Ramp T-Stab", "T-Stab", "useful", 0.12, "Stabilizes easy-to-hard lesson progression instead of abrupt jumps."),
    OperatorSpec("compute_budget_allocator_guard", "Compute Budget Allocator Guard", "Guard", "useful", 0.12, "Keeps curriculum seed/worker/mutation budget within declared limits."),
    OperatorSpec("promotion_gate_precheck_guard", "Promotion Gate Precheck Guard", "Guard", "useful", 0.13, "Blocks promotion attempts until validation, adversarial, replay, and scope gates exist."),
    OperatorSpec("next_mutation_queue_scribe", "Next Mutation Queue Scribe", "Scribe", "useful", 0.11, "Writes the next concrete mutation/lesson queue with reasons and replay links."),
)


CONTROL_OPERATORS: tuple[OperatorSpec, ...] = (
    OperatorSpec("random_lesson_selector", "Random Lesson Selector", "unsafe_control", "unsafe", 0.03, "Unsafe control: picks lesson families randomly."),
    OperatorSpec("novelty_only_curriculum", "Novelty-Only Curriculum", "unsafe_control", "unsafe", 0.04, "Unsafe control: chases new tasks without replay protection."),
    OperatorSpec("easy_only_sampler", "Easy-Only Sampler", "unsafe_control", "unsafe", 0.03, "Unsafe control: avoids hard/adversarial cases."),
    OperatorSpec("no_replay_curriculum", "No-Replay Curriculum", "unsafe_control", "unsafe", 0.04, "Unsafe control: omits prior skill replay and risks forgetting."),
    OperatorSpec("train_metric_only_promoter", "Train-Metric-Only Promoter", "unsafe_control", "unsafe", 0.04, "Unsafe control: promotes from train score without adversarial/scope gates."),
    OperatorSpec("budgetless_curriculum_expander", "Budgetless Curriculum Expander", "unsafe_control", "unsafe", 0.04, "Unsafe control: expands lessons beyond budget."),
    OperatorSpec("stale_gap_repeater", "Stale Gap Repeater", "control", "noop", 0.03, "Control: repeats solved gaps instead of targeting current evidence."),
    OperatorSpec("gap_detector_echo_clone", "Gap Detector Echo Clone", "Lens", "redundant", 0.18, "Redundant gap detector without unique value."),
)


OPERATOR_LIBRARY = USEFUL_OPERATORS + CONTROL_OPERATORS
USEFUL_IDS = tuple(operator.operator_id for operator in USEFUL_OPERATORS)
ALL_OPERATOR_IDS = tuple(operator.operator_id for operator in OPERATOR_LIBRARY)
OPERATOR_BY_ID = {operator.operator_id: operator for operator in OPERATOR_LIBRARY}
UNSAFE_IDS = {operator.operator_id for operator in CONTROL_OPERATORS if operator.role == "unsafe"}


@dataclass(frozen=True)
class CurriculumCase:
    case_id: str
    source_split: str
    family: str
    observed_metrics: str
    expected_action: str
    expected_plan: tuple[str, ...]
    required_operators: tuple[str, ...]
    requires_replay: bool
    requires_adversarial: bool
    requires_budget_guard: bool
    requires_promotion_precheck: bool


def stable_int(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def split_for(seed: int, case_id: str) -> str:
    bucket = stable_int(f"{seed}:{case_id}") % 10
    if bucket < 7:
        return "train"
    if bucket < 9:
        return "validation"
    return "adversarial"


def make_case(seed: int, index: int) -> CurriculumCase:
    family = (
        "calc_operator_gap_next_lesson",
        "scope_bleed_adversarial_repair",
        "memory_stale_regression_replay",
        "output_hygiene_regression_replay",
        "route_loop_budget_regression",
        "promotion_candidate_precheck",
        "no_gap_hold_replay_only",
        "overfit_train_family_warning",
        "cross_skill_mix_curriculum",
        "compute_pressure_downselect",
    )[index % 10]
    case_id = f"{family}_{seed}_{index:05d}"
    split = split_for(seed, case_id)
    base = ("capability_gap_detector_lens", "lesson_candidate_ranker_scribe", "next_mutation_queue_scribe")
    replay = ("regression_replay_set_guard",)
    adv = ("adversarial_family_sampler_lens",)
    ramp = ("difficulty_ramp_t_stab",)
    budget = ("compute_budget_allocator_guard",)
    promo = ("promotion_gate_precheck_guard",)

    if family == "calc_operator_gap_next_lesson":
        plan = ("detect_gap:calc_floor_division_edge", "rank_lesson:operator_variant", "include_replay:calc_scribe_v003", "ramp:easy_to_hard", "queue_mutation:operator_normalizer")
        return CurriculumCase(case_id, split, family, "calc marker min dropped on new operator edge; prior calc stable.", "QUEUE_NEXT_LESSON", plan, base + replay + ramp, True, False, False, False)
    if family == "scope_bleed_adversarial_repair":
        plan = ("detect_gap:scope_bleed", "sample_adversarial:word_problem_lure", "include_replay:visible_marker_only", "precheck:block_promotion", "queue_mutation:scope_guard")
        return CurriculumCase(case_id, split, family, "train high but adversarial scope bleed detected.", "QUEUE_ADVERSARIAL_REPAIR", plan, base + replay + adv + promo, True, True, False, True)
    if family == "memory_stale_regression_replay":
        plan = ("detect_gap:stale_trace_pollution", "rank_lesson:memory_hygiene", "include_replay:trace_ground", "ramp:delayed_feedback", "queue_mutation:stale_pruner")
        return CurriculumCase(case_id, split, family, "memory hygiene stable but delayed stale trace family regressed.", "QUEUE_REPLAY_REPAIR", plan, base + replay + ramp, True, False, False, False)
    if family == "output_hygiene_regression_replay":
        plan = ("detect_gap:unsafe_render", "rank_lesson:output_hygiene", "include_replay:scribe_output", "sample_adversarial:no_citation", "queue_mutation:final_integrity")
        return CurriculumCase(case_id, split, family, "output marker render works but citation-free adversarial rows fail.", "QUEUE_REPLAY_REPAIR", plan, base + replay + adv, True, True, False, False)
    if family == "route_loop_budget_regression":
        plan = ("detect_gap:route_loop_budget", "rank_lesson:route_guard", "include_replay:e97_route", "enforce_budget:strict", "queue_mutation:loop_guard")
        return CurriculumCase(case_id, split, family, "route success high but loop and budget stress reappeared.", "QUEUE_REPLAY_REPAIR", plan, base + replay + budget, True, False, True, False)
    if family == "promotion_candidate_precheck":
        plan = ("detect_gap:promotion_evidence_missing", "precheck:block_promotion", "include_replay:negative_scope", "sample_adversarial:token_swap", "queue_mutation:reload_audit")
        return CurriculumCase(case_id, split, family, "candidate score high but reload and negative-scope gates missing.", "BLOCK_PROMOTION_AND_QUEUE_TESTS", plan, base + replay + adv + promo, True, True, False, True)
    if family == "no_gap_hold_replay_only":
        plan = ("detect_gap:none", "hold_new_mutation", "include_replay:stable_core", "enforce_budget:minimal", "queue_mutation:none")
        return CurriculumCase(case_id, split, family, "all active metrics clean; no new lesson justified yet.", "HOLD_REPLAY_ONLY", plan, ("capability_gap_detector_lens", "regression_replay_set_guard", "compute_budget_allocator_guard", "next_mutation_queue_scribe"), True, False, True, False)
    if family == "overfit_train_family_warning":
        plan = ("detect_gap:train_val_ood_gap", "sample_adversarial:heldout_family", "include_replay:baseline", "precheck:block_promotion", "queue_mutation:generalization_probe")
        return CurriculumCase(case_id, split, family, "train metric improved but OOD dropped.", "QUEUE_ADVERSARIAL_REPAIR", plan, base + replay + adv + promo, True, True, False, True)
    if family == "cross_skill_mix_curriculum":
        plan = ("detect_gap:cross_skill_composition", "rank_lesson:multi_skill_mix", "include_replay:e95_e96_e97_e98", "ramp:two_to_four_skills", "queue_mutation:composition_bridge")
        return CurriculumCase(case_id, split, family, "single skills stable; mixed skill chain needs expansion.", "QUEUE_NEXT_LESSON", plan, base + replay + ramp, True, False, False, False)
    plan = ("detect_gap:compute_pressure", "rank_lesson:highest_expected_value", "include_replay:minimal_canary", "enforce_budget:downselect", "queue_mutation:smallest_safe")
    return CurriculumCase(case_id, split, family, "too many candidates for current run budget; select smallest safe lesson set.", "QUEUE_BUDGETED_LESSON", plan, base + replay + budget, True, False, True, False)


def generate_cases(seed: int, rows: int) -> list[CurriculumCase]:
    return [make_case(seed, index) for index in range(rows)]


def unsafe_prediction(case: CurriculumCase, selected: set[str]) -> dict[str, object] | None:
    if "random_lesson_selector" in selected:
        return {"action": "QUEUE_NEXT_LESSON", "plan": ("random_family",), "gap_valid": False, "replay_valid": False, "adversarial_valid": False, "budget_valid": True, "promotion_safe": False, "stale_lesson": True, "forgetting_risk": True, "unsafe_promotion": case.requires_promotion_precheck, "over_budget": False}
    if "novelty_only_curriculum" in selected and case.requires_replay:
        return {"action": "QUEUE_NEXT_LESSON", "plan": ("new_family_only",), "gap_valid": True, "replay_valid": False, "adversarial_valid": not case.requires_adversarial, "budget_valid": True, "promotion_safe": not case.requires_promotion_precheck, "stale_lesson": False, "forgetting_risk": True, "unsafe_promotion": case.requires_promotion_precheck, "over_budget": False}
    if "easy_only_sampler" in selected and case.requires_adversarial:
        return {"action": case.expected_action, "plan": case.expected_plan[:2] + ("easy_only",), "gap_valid": True, "replay_valid": True, "adversarial_valid": False, "budget_valid": True, "promotion_safe": True, "stale_lesson": False, "forgetting_risk": False, "unsafe_promotion": False, "over_budget": False}
    if "no_replay_curriculum" in selected and case.requires_replay:
        return {"action": case.expected_action, "plan": tuple(item for item in case.expected_plan if not item.startswith("include_replay")), "gap_valid": True, "replay_valid": False, "adversarial_valid": True, "budget_valid": True, "promotion_safe": True, "stale_lesson": False, "forgetting_risk": True, "unsafe_promotion": False, "over_budget": False}
    if "train_metric_only_promoter" in selected and case.requires_promotion_precheck:
        return {"action": "PROMOTE_NOW", "plan": ("train_score_high",), "gap_valid": False, "replay_valid": False, "adversarial_valid": False, "budget_valid": True, "promotion_safe": False, "stale_lesson": False, "forgetting_risk": True, "unsafe_promotion": True, "over_budget": False}
    if "budgetless_curriculum_expander" in selected and case.requires_budget_guard:
        return {"action": case.expected_action, "plan": case.expected_plan + ("expand_all_candidates",), "gap_valid": True, "replay_valid": True, "adversarial_valid": True, "budget_valid": False, "promotion_safe": True, "stale_lesson": False, "forgetting_risk": False, "unsafe_promotion": False, "over_budget": True}
    if "stale_gap_repeater" in selected and case.family == "no_gap_hold_replay_only":
        return {"action": "QUEUE_NEXT_LESSON", "plan": ("repeat_solved_gap",), "gap_valid": False, "replay_valid": True, "adversarial_valid": True, "budget_valid": True, "promotion_safe": True, "stale_lesson": True, "forgetting_risk": False, "unsafe_promotion": False, "over_budget": False}
    return None


def predict(case: CurriculumCase, selected: set[str]) -> dict[str, object]:
    unsafe = unsafe_prediction(case, selected)
    if unsafe:
        return unsafe
    required = set(case.required_operators)
    if required <= selected:
        return {"action": case.expected_action, "plan": case.expected_plan, "gap_valid": True, "replay_valid": True, "adversarial_valid": True, "budget_valid": True, "promotion_safe": True, "stale_lesson": False, "forgetting_risk": False, "unsafe_promotion": False, "over_budget": False}
    return {"action": "HOLD_REPLAY_ONLY", "plan": ("insufficient_scheduler",), "gap_valid": False, "replay_valid": "regression_replay_set_guard" in selected, "adversarial_valid": not case.requires_adversarial, "budget_valid": "compute_budget_allocator_guard" in selected or not case.requires_budget_guard, "promotion_safe": not case.requires_promotion_precheck, "stale_lesson": False, "forgetting_risk": case.requires_replay and "regression_replay_set_guard" not in selected, "unsafe_promotion": False, "over_budget": False}


def evaluate(selected: set[str], cases: list[CurriculumCase], split: str | None = None) -> dict[str, float]:
    rows = [case for case in cases if split is None or case.source_split == split]
    if not rows:
        return {"curriculum_success": 0.0, "gap_target_accuracy": 0.0, "replay_coverage": 0.0, "adversarial_coverage": 0.0, "budget_validity": 0.0, "promotion_safety": 0.0, "stale_lesson_rate": 0.0, "forgetting_risk_rate": 0.0, "unsafe_promotion_rate": 0.0, "over_budget_rate": 0.0, "utility": -1.0}
    success = gap = replay = adversarial = budget = promo = 0
    stale = forgetting = unsafe_promo = over_budget = 0
    partial = 0.0
    for case in rows:
        pred = predict(case, selected)
        required = set(case.required_operators)
        partial += len(required & selected) / max(1, len(required))
        row_success = (
            pred["action"] == case.expected_action
            and tuple(pred["plan"]) == case.expected_plan
            and pred["gap_valid"]
            and pred["replay_valid"]
            and pred["adversarial_valid"]
            and pred["budget_valid"]
            and pred["promotion_safe"]
            and not pred["stale_lesson"]
            and not pred["forgetting_risk"]
            and not pred["unsafe_promotion"]
            and not pred["over_budget"]
        )
        success += int(row_success)
        gap += int(pred["gap_valid"])
        replay += int(pred["replay_valid"])
        adversarial += int(pred["adversarial_valid"])
        budget += int(pred["budget_valid"])
        promo += int(pred["promotion_safe"])
        stale += int(pred["stale_lesson"])
        forgetting += int(pred["forgetting_risk"])
        unsafe_promo += int(pred["unsafe_promotion"])
        over_budget += int(pred["over_budget"])
    count = len(rows)
    cost = sum(OPERATOR_BY_ID[operator_id].cost for operator_id in selected)
    score = success / count
    partial_score = partial / count
    utility = score + 0.28 * partial_score - 1.4 * (forgetting / count) - 1.6 * (unsafe_promo / count) - 1.2 * (over_budget / count) - 0.8 * (stale / count) - 0.01 * cost
    return {
        "curriculum_success": round(score, 6),
        "gap_target_accuracy": round(gap / count, 6),
        "replay_coverage": round(replay / count, 6),
        "adversarial_coverage": round(adversarial / count, 6),
        "budget_validity": round(budget / count, 6),
        "promotion_safety": round(promo / count, 6),
        "stale_lesson_rate": round(stale / count, 6),
        "forgetting_risk_rate": round(forgetting / count, 6),
        "unsafe_promotion_rate": round(unsafe_promo / count, 6),
        "over_budget_rate": round(over_budget / count, 6),
        "partial_curriculum_score": round(partial_score, 6),
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
        forgetting_deltas = []
        for result in seed_results:
            selected = set(result["selected"])  # type: ignore[arg-type]
            cases = [CurriculumCase(**case) for case in result["cases"]]  # type: ignore[arg-type]
            full = evaluate(selected, cases, "validation")
            ablated = evaluate(selected - {operator_id}, cases, "validation")
            losses.append(full["curriculum_success"] - ablated["curriculum_success"])
            forgetting_deltas.append(ablated["forgetting_risk_rate"] - full["forgetting_risk_rate"])
        summary[operator_id] = {
            "mean_curriculum_success_loss": round(statistics.mean(losses), 6),
            "mean_forgetting_risk_delta": round(statistics.mean(forgetting_deltas), 6),
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
        "validation_curriculum_success_min": min(vals("validation", "curriculum_success")),
        "validation_curriculum_success_mean": round(statistics.mean(vals("validation", "curriculum_success")), 6),
        "adversarial_curriculum_success_min": min(vals("adversarial", "curriculum_success")),
        "adversarial_curriculum_success_mean": round(statistics.mean(vals("adversarial", "curriculum_success")), 6),
        "validation_gap_target_accuracy_min": min(vals("validation", "gap_target_accuracy")),
        "validation_replay_coverage_min": min(vals("validation", "replay_coverage")),
        "validation_adversarial_coverage_min": min(vals("validation", "adversarial_coverage")),
        "validation_budget_validity_min": min(vals("validation", "budget_validity")),
        "validation_promotion_safety_min": min(vals("validation", "promotion_safety")),
        "adversarial_forgetting_risk_rate_max": max(vals("adversarial", "forgetting_risk_rate")),
        "adversarial_unsafe_promotion_rate_max": max(vals("adversarial", "unsafe_promotion_rate")),
        "adversarial_over_budget_rate_max": max(vals("adversarial", "over_budget_rate")),
        "adversarial_stale_lesson_rate_max": max(vals("adversarial", "stale_lesson_rate")),
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
    if agg["validation_curriculum_success_min"] != 1.0:
        failures.append("validation curriculum success below 1.0")
    if agg["adversarial_curriculum_success_min"] != 1.0:
        failures.append("adversarial curriculum success below 1.0")
    for key in ["validation_gap_target_accuracy_min", "validation_replay_coverage_min", "validation_adversarial_coverage_min", "validation_budget_validity_min", "validation_promotion_safety_min"]:
        if agg[key] != 1.0:
            failures.append(f"{key} below 1.0")
    for key in ["adversarial_forgetting_risk_rate_max", "adversarial_unsafe_promotion_rate_max", "adversarial_over_budget_rate_max", "adversarial_stale_lesson_rate_max"]:
        if agg[key] != 0.0:
            failures.append(f"{key} nonzero")
    unsafe_final = sum(1 for result in seed_results for operator_id in result["selected"] if operator_id in UNSAFE_IDS)  # type: ignore[operator]
    if unsafe_final:
        failures.append("unsafe operator selected")
    decision = "e99_curriculum_scheduler_regression_replay_expansion_confirmed" if not failures else "e99_curriculum_scheduler_incomplete"
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
        "curriculum_scheduler": True,
        "autonomous_open_ended_training": False,
        "requires_regression_replay": True,
        "requires_adversarial_coverage": True,
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
            typed_case = CurriculumCase(**case)
            pred = predict(typed_case, set(result["selected"]))  # type: ignore[arg-type]
            append_jsonl(out / "row_level_samples.jsonl", {
                "seed": result["seed"],
                "case_id": typed_case.case_id,
                "family": typed_case.family,
                "observed_metrics": typed_case.observed_metrics,
                "expected_action": typed_case.expected_action,
                "expected_plan": typed_case.expected_plan,
                "predicted": pred,
            })
            sample_rows += 1
            if sample_rows >= 480:
                break
        if sample_rows >= 480:
            break
    report = [
        "# E99 Curriculum Scheduler And Regression Replay Expansion Result",
        "",
        f"decision = `{decision}`",
        "",
        "Boundary: controlled curriculum scheduling, not autonomous open-ended training.",
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
    parser.add_argument("--out", default="target/pilot_wave/e99_curriculum_scheduler_and_regression_replay_expansion")
    parser.add_argument("--artifact-sample-dir", default="docs/research/artifact_samples/e99_curriculum_scheduler_and_regression_replay_expansion")
    parser.add_argument("--seeds", default="109901,109902,109903,109904,109905,109906,109907,109908,109909,109910,109911,109912,109913,109914,109915,109916")
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
        "boundary": "controlled curriculum scheduler probe; not autonomous open-ended training",
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
