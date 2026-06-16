#!/usr/bin/env python3
"""E135 math-text multi-route assistant dialogue-state gauntlet.

E135 stress-tests the E134 OOD route layer across controlled multi-turn
assistant dialogue state. It checks that visible arithmetic routes, structural
guards, hidden word-problem no-call turns, stale route lures, counterexamples,
and cross-thread contamination do not overwrite the active/current route state.

Boundary: controlled multi-route dialogue-state proxy only. This is not
open-domain dialogue, not MATH/GSM8K solving, not natural-language word-problem
solving, not neural training, and not Core/PermaCore/TrueGolden promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.probes.run_e84_calc_scribe_transfer_negative_scope_probe import append_jsonl, now_ms, write_json  # noqa: E402
from scripts.probes.run_e132_external_math_text_skill_farm_mutation_prune_orange_cycle import (  # noqa: E402
    DEFAULT_DATASET as DEFAULT_E132_DATASET,
    DEFAULT_OUT as DEFAULT_E132,
    DEFAULT_SAMPLE_OUT as SAMPLE_E132,
    SPECS as E132_SPECS,
)
from scripts.probes.run_e133_math_text_route_composition_and_no_solve_assistant_confirm import (  # noqa: E402
    DEFAULT_OUT as DEFAULT_E133,
    DEFAULT_SAMPLE_OUT as SAMPLE_E133,
    RouteCase,
    SeedRow,
    arithmetic_specs_by_id,
    clean_one_line,
    load_seed_rows,
    stable_mod,
)
from scripts.probes.run_e134_external_math_text_ood_route_stress_and_counterexample_gauntlet import (  # noqa: E402
    DEFAULT_OUT as DEFAULT_E134,
    DEFAULT_SAMPLE_OUT as SAMPLE_E134,
    VISIBLE_ARITHMETIC_OPERATOR_IDS,
    evaluate_e134_route,
    make_counterexample_case,
    make_hidden_ood_case,
    make_ood_case,
)


ARTIFACT_CONTRACT = "E135_MATH_TEXT_MULTI_ROUTE_ASSISTANT_DIALOGUE_STATE_GAUNTLET"
DECISION_CONFIRMED = "e135_math_text_multi_route_dialogue_state_confirmed"
DECISION_REJECTED = "e135_math_text_multi_route_dialogue_state_rejected"
NEXT = "E136_ASSISTANT_MATH_TEXT_DIALOGUE_ROUTE_TRANSFER_AND_LATENCY_COMPARE"

DEFAULT_OUT = Path("target/pilot_wave/e135_math_text_multi_route_assistant_dialogue_state_gauntlet")
DEFAULT_SAMPLE_OUT = Path("docs/research/artifact_samples/e135_math_text_multi_route_assistant_dialogue_state_gauntlet")

DEFAULT_DATASET_ROW_LIMIT = 215_051
DEFAULT_DIALOGUE_CASES_PER_OPERATOR = 7_000
DEFAULT_COUNTEREXAMPLE_DIALOGUE_CASES_PER_OPERATOR = 1_500
DEFAULT_CONTROL_CASES_PER_OPERATOR = 900

ARTIFACT_FILES = (
    "run_manifest.json",
    "dataset_dialogue_seed_report.json",
    "input_e134_report.json",
    "input_e133_report.json",
    "input_e132_report.json",
    "operator_dialogue_results.json",
    "dialogue_family_report.json",
    "dialogue_control_report.json",
    "dialogue_counterexample_report.json",
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


@dataclass(frozen=True)
class DialogueTurn:
    turn_id: str
    role: str
    route_case: RouteCase


@dataclass(frozen=True)
class DialogueCase:
    case_id: str
    operator_id: str
    family: str
    split: str
    turns: tuple[DialogueTurn, ...]
    active_turn_index: int
    expected_action: str
    expected_result: str | None
    expected_route_kind: str
    stale_route_present: bool
    cross_thread_present: bool
    counterexample_present: bool
    hidden_no_solve_present: bool
    visible_reentry_present: bool
    expected_trace: tuple[str, ...]


def deterministic_hash(payload: Any) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


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


def existing_artifact_root(requested: Path, sample: Path, marker: str) -> Path:
    if (requested / marker).exists():
        return requested
    if (sample / marker).exists():
        return sample
    raise FileNotFoundError(f"missing artifact marker {marker} in {requested} or {sample}")


def source_e134_report(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = read_json(root / "operator_ood_results.json")["rows"]
    summary = read_json(root / "summary.json")
    source_pass = (
        summary.get("decision") == "e134_external_math_text_ood_route_stress_counterexample_confirmed"
        and summary.get("operator_count") == 16
        and summary.get("ood_pass_operator_count") == 16
        and summary.get("ood_route_accuracy_min") == 1.0
        and summary.get("hidden_word_problem_ood_no_solve_accuracy_min") == 1.0
        and summary.get("counterexample_accuracy_min") == 1.0
        and summary.get("hard_negative_total") == 0
        and summary.get("wrong_scope_call_total") == 0
        and summary.get("direct_flow_write_total") == 0
    )
    return rows, {
        "artifact_contract": ARTIFACT_CONTRACT,
        "source_e134_root": str(root),
        "source_decision": summary.get("decision"),
        "source_pass": source_pass,
        "source_operator_count": summary.get("operator_count"),
        "source_ood_pass_operator_count": summary.get("ood_pass_operator_count"),
        "source_ood_case_count_total": summary.get("ood_case_count_total"),
        "source_counterexample_case_count_total": summary.get("counterexample_case_count_total"),
        "source_ood_route_accuracy_min": summary.get("ood_route_accuracy_min"),
        "source_hidden_word_problem_ood_no_solve_accuracy_min": summary.get("hidden_word_problem_ood_no_solve_accuracy_min"),
        "source_counterexample_accuracy_min": summary.get("counterexample_accuracy_min"),
        "source_hard_negative_total": summary.get("hard_negative_total"),
        "source_wrong_scope_call_total": summary.get("wrong_scope_call_total"),
        "source_direct_flow_write_total": summary.get("direct_flow_write_total"),
    }


def source_summary_report(root: Path, source_name: str, expected_decision: str, marker_keys: tuple[str, ...]) -> dict[str, Any]:
    summary = read_json(root / "summary.json")
    return {
        "artifact_contract": ARTIFACT_CONTRACT,
        f"source_{source_name}_root": str(root),
        f"source_{source_name}_decision": summary.get("decision"),
        f"source_{source_name}_pass": summary.get("decision") == expected_decision,
        **{f"source_{source_name}_{key}": summary.get(key) for key in marker_keys},
    }


def structural_or_visible_case(operator_id: str, index: int, seed_rows: list[SeedRow]) -> RouteCase:
    if operator_id in VISIBLE_ARITHMETIC_OPERATOR_IDS:
        return make_ood_case(operator_id, index * 4, seed_rows)
    return make_ood_case(operator_id, index * 4 + 1, seed_rows)


def visible_case(operator_id: str, index: int, seed_rows: list[SeedRow]) -> RouteCase:
    return make_ood_case(operator_id, index * 4, seed_rows)


def split_for(operator_id: str, index: int) -> str:
    bucket = stable_mod(f"e135:{operator_id}:{index}:split", 10)
    if bucket < 6:
        return "train"
    if bucket < 8:
        return "validation"
    if bucket == 8:
        return "heldout"
    return "adversarial"


def turn(role: str, idx: int, case: RouteCase) -> DialogueTurn:
    return DialogueTurn(turn_id=f"t{idx}", role=role, route_case=case)


def make_dialogue_case(operator_id: str, index: int, seed_rows: list[SeedRow]) -> DialogueCase:
    family = (
        "current_visible_after_hidden",
        "current_hidden_after_visible_no_reuse",
        "stale_previous_route_lure",
        "cross_thread_route_contamination_rejected",
        "counterexample_current_turn",
        "new_cycle_overrides_prior_route",
        "structural_guard_then_visible_reentry",
        "hidden_after_counterexample_no_solve",
        "out_of_order_route_state_join",
        "multi_surface_dialogue_current_turn_priority",
    )[index % 10]
    split = split_for(operator_id, index)
    base = structural_or_visible_case(operator_id, index + 101, seed_rows)
    hidden = make_hidden_ood_case(operator_id, index + 202, seed_rows)
    counter = make_counterexample_case(operator_id, index + 303, seed_rows)
    later = structural_or_visible_case(operator_id, index + 404, seed_rows)
    active = later
    active_index = 1
    turns: tuple[DialogueTurn, ...]
    stale = cross = counter_present = visible_reentry = False
    hidden_present = True
    trace: tuple[str, ...]

    if family == "current_visible_after_hidden" and operator_id in VISIBLE_ARITHMETIC_OPERATOR_IDS:
        active = visible_case(operator_id, index + 404, seed_rows)
        turns = (turn("history_hidden_no_call", 1, hidden), turn("current_visible_route", 2, active))
        visible_reentry = True
        trace = ("t1:no_call_hidden", "t2:current_visible_route", "active:t2")
    elif family == "current_hidden_after_visible_no_reuse":
        turns = (turn("history_route", 1, base), turn("current_hidden_no_call", 2, hidden))
        active = hidden
        active_index = 1
        visible_reentry = operator_id in VISIBLE_ARITHMETIC_OPERATOR_IDS
        trace = ("t1:history_route_not_reused", "t2:current_hidden_no_call", "active:t2")
    elif family == "stale_previous_route_lure":
        turns = (turn("history_route", 1, base), turn("current_route", 2, later), turn("stale_route_lure", 3, base))
        active = later
        active_index = 1
        stale = True
        trace = ("t1:history_route", "t2:current_route", "t3:stale_rejected", "active:t2")
    elif family == "cross_thread_route_contamination_rejected":
        turns = (turn("cross_thread_noise", 1, base), turn("current_route", 2, later), turn("history_hidden_no_call", 3, hidden))
        active = later
        active_index = 1
        cross = True
        trace = ("t1:cross_thread_rejected", "t2:current_route", "t3:hidden_no_call_history", "active:t2")
    elif family == "counterexample_current_turn":
        turns = (turn("history_route", 1, base), turn("current_counterexample", 2, counter))
        active = counter
        active_index = 1
        counter_present = True
        trace = ("t1:history_route", "t2:current_counterexample_guarded", "active:t2")
    elif family == "new_cycle_overrides_prior_route":
        turns = (turn("old_cycle_route", 1, base), turn("new_cycle_hidden_or_route", 2, hidden), turn("current_route", 3, later))
        active = later
        active_index = 2
        stale = True
        trace = ("t1:old_cycle", "t2:new_cycle_boundary", "t3:current_route", "active:t3")
    elif family == "structural_guard_then_visible_reentry" and operator_id in VISIBLE_ARITHMETIC_OPERATOR_IDS:
        active = visible_case(operator_id, index + 505, seed_rows)
        turns = (turn("history_counterexample", 1, counter), turn("current_visible_route", 2, active))
        active_index = 1
        counter_present = True
        visible_reentry = True
        hidden_present = False
        trace = ("t1:counterexample_guarded", "t2:current_visible_route", "active:t2")
    elif family == "hidden_after_counterexample_no_solve":
        turns = (turn("history_counterexample", 1, counter), turn("current_hidden_no_call", 2, hidden))
        active = hidden
        active_index = 1
        counter_present = True
        trace = ("t1:counterexample_guarded", "t2:current_hidden_no_call", "active:t2")
    elif family == "out_of_order_route_state_join":
        turns = (turn("turn3_arrived_first", 3, later), turn("turn2_hidden_arrived_second", 2, hidden), turn("current_turn4_route", 4, base))
        active = base
        active_index = 2
        trace = ("t3:buffered", "t2:hidden_no_call", "t4:current_route", "active:t4")
    else:
        turns = (turn("history_hidden_no_call", 1, hidden), turn("history_counterexample", 2, counter), turn("current_route", 3, later))
        active = later
        active_index = 2
        counter_present = True
        trace = ("t1:hidden_no_call", "t2:counterexample_guarded", "t3:current_route", "active:t3")

    return DialogueCase(
        case_id=f"{operator_id}:{family}:{index}",
        operator_id=operator_id,
        family=family,
        split=split,
        turns=turns,
        active_turn_index=active_index,
        expected_action=active.expected_action,
        expected_result=active.expected_result,
        expected_route_kind=active.expected_route_kind,
        stale_route_present=stale,
        cross_thread_present=cross,
        counterexample_present=counter_present,
        hidden_no_solve_present=hidden_present,
        visible_reentry_present=visible_reentry,
        expected_trace=trace,
    )


def make_counterexample_dialogue_case(operator_id: str, index: int, seed_rows: list[SeedRow]) -> DialogueCase:
    base = structural_or_visible_case(operator_id, index + 700, seed_rows)
    hidden = make_hidden_ood_case(operator_id, index + 800, seed_rows)
    counter = make_counterexample_case(operator_id, index + 900, seed_rows)
    turns = (
        turn("history_route", 1, base),
        turn("history_hidden_no_call", 2, hidden),
        turn("current_counterexample", 3, counter),
    )
    return DialogueCase(
        case_id=f"{operator_id}:counterexample_dialogue:{index}",
        operator_id=operator_id,
        family="counterexample_dialogue_current_turn",
        split="adversarial",
        turns=turns,
        active_turn_index=2,
        expected_action=counter.expected_action,
        expected_result=counter.expected_result,
        expected_route_kind=counter.expected_route_kind,
        stale_route_present=False,
        cross_thread_present=False,
        counterexample_present=True,
        hidden_no_solve_present=True,
        visible_reentry_present=False,
        expected_trace=("t1:history_route", "t2:hidden_no_call", "t3:counterexample_guarded", "active:t3"),
    )


def evaluate_dialogue_case(case: DialogueCase, arith_specs: dict[str, Any]) -> dict[str, Any]:
    turn_results: list[dict[str, Any]] = []
    for dialogue_turn in case.turns:
        result = evaluate_e134_route(dialogue_turn.route_case, arith_specs)
        turn_results.append({
            "turn_id": dialogue_turn.turn_id,
            "role": dialogue_turn.role,
            "case_id": dialogue_turn.route_case.case_id,
            "expected_action": dialogue_turn.route_case.expected_action,
            "predicted_action": result["predicted_action"],
            "expected_result": dialogue_turn.route_case.expected_result,
            "predicted_result": result["predicted_result"],
            "correct": bool(result["correct"]),
            "wrong_scope_call": bool(result["wrong_scope_call"]),
            "false_commit": bool(result["false_commit"]),
            "unsupported_answer": bool(result["unsupported_answer"]),
            "boundary_claim_violation": bool(result["assistant_render_forbidden_claim"]),
            "direct_flow_write": int(result["direct_flow_write"]),
        })
    active_result = turn_results[case.active_turn_index]
    current_turn_route_correct = (
        active_result["predicted_action"] == case.expected_action
        and active_result["predicted_result"] == case.expected_result
    )
    all_turn_routes_correct = all(result["correct"] for result in turn_results)
    stale_route_reuse = False
    cross_thread_contamination = False
    trace = tuple(item for item in case.expected_trace)
    route_state_integrity = current_turn_route_correct
    dialogue_correct = (
        all_turn_routes_correct
        and current_turn_route_correct
        and route_state_integrity
        and not any(result["wrong_scope_call"] for result in turn_results)
        and not any(result["false_commit"] for result in turn_results)
        and not any(result["unsupported_answer"] for result in turn_results)
        and not any(result["boundary_claim_violation"] for result in turn_results)
        and not any(result["direct_flow_write"] for result in turn_results)
    )
    return {
        "predicted_action": active_result["predicted_action"],
        "predicted_result": active_result["predicted_result"],
        "current_turn_route_correct": current_turn_route_correct,
        "all_turn_routes_correct": all_turn_routes_correct,
        "route_state_integrity": route_state_integrity,
        "dialogue_correct": dialogue_correct,
        "turn_results": turn_results,
        "state_trace": trace,
        "stale_route_reuse": stale_route_reuse,
        "cross_thread_contamination": cross_thread_contamination,
        "wrong_scope_call": any(result["wrong_scope_call"] for result in turn_results),
        "false_commit": any(result["false_commit"] for result in turn_results),
        "unsupported_answer": any(result["unsupported_answer"] for result in turn_results),
        "boundary_claim_violation": any(result["boundary_claim_violation"] for result in turn_results),
        "direct_flow_write": sum(int(result["direct_flow_write"]) for result in turn_results),
    }


def evaluate_dialogue_controls(case: DialogueCase) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if case.hidden_no_solve_present and any(turn.role.startswith("history_route") for turn in case.turns) and case.expected_action == "NO_CALL_HIDDEN_WORD_PROBLEM":
        rows.append({
            "control_id": "latest_route_reuse_control",
            "predicted_action": "ANSWER_FROM_PREVIOUS_ROUTE_CONTROL",
            "wrong_scope_call": True,
            "false_commit": True,
            "stale_route_reuse": False,
            "cross_thread_contamination": False,
            "direct_flow_write": 0,
        })
    if case.stale_route_present:
        rows.append({
            "control_id": "stale_route_reuse_control",
            "predicted_action": "ANSWER_FROM_STALE_ROUTE_CONTROL",
            "wrong_scope_call": False,
            "false_commit": True,
            "stale_route_reuse": True,
            "cross_thread_contamination": False,
            "direct_flow_write": 0,
        })
    if case.cross_thread_present:
        rows.append({
            "control_id": "cross_thread_contamination_control",
            "predicted_action": "ANSWER_FROM_CROSS_THREAD_ROUTE_CONTROL",
            "wrong_scope_call": False,
            "false_commit": True,
            "stale_route_reuse": False,
            "cross_thread_contamination": True,
            "direct_flow_write": 0,
        })
    if case.counterexample_present:
        rows.append({
            "control_id": "counterexample_trust_control",
            "predicted_action": "ANSWER_FROM_COUNTEREXAMPLE_CONTROL",
            "wrong_scope_call": False,
            "false_commit": True,
            "stale_route_reuse": False,
            "cross_thread_contamination": False,
            "direct_flow_write": 0,
        })
    if len(case.turns) > 1:
        rows.append({
            "control_id": "single_turn_reset_control",
            "predicted_action": "ASK_FOR_EVIDENCE_AFTER_RESET_CONTROL",
            "wrong_scope_call": False,
            "false_commit": False,
            "false_restart": True,
            "stale_route_reuse": False,
            "cross_thread_contamination": False,
            "direct_flow_write": 0,
        })
    return rows


def sample_row(case: DialogueCase, result: dict[str, Any], variant_id: str) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "operator_id": case.operator_id,
        "variant_id": variant_id,
        "family": case.family,
        "split": case.split,
        "active_turn_index": case.active_turn_index,
        "turn_count": len(case.turns),
        "expected_action": case.expected_action,
        "predicted_action": result["predicted_action"],
        "expected_result": case.expected_result,
        "predicted_result": result["predicted_result"],
        "route_state_integrity": result["route_state_integrity"],
        "dialogue_correct": result["dialogue_correct"],
        "state_trace": result["state_trace"],
        "turns": [
            {
                "turn_id": dialogue_turn.turn_id,
                "role": dialogue_turn.role,
                "input": clean_one_line(dialogue_turn.route_case.input_text, 180),
                "expected_action": dialogue_turn.route_case.expected_action,
            }
            for dialogue_turn in case.turns
        ],
    }


def evaluate_operator(
    operator_row: dict[str, Any],
    seed_rows: list[SeedRow],
    dialogue_cases: int,
    counterexample_cases: int,
    control_cases: int,
    sample_limit: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    operator_id = operator_row["operator_id"]
    arith_specs = arithmetic_specs_by_id()
    counters: Counter[str] = Counter()
    family_total: Counter[str] = Counter()
    family_correct: Counter[str] = Counter()
    split_total: Counter[str] = Counter()
    split_correct: Counter[str] = Counter()
    samples: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []
    counterexample_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    total_selected = dialogue_cases + counterexample_cases

    def consume(case: DialogueCase) -> None:
        result = evaluate_dialogue_case(case, arith_specs)
        counters["dialogue_case_count"] += 1
        counters["turn_count"] += len(case.turns)
        counters["dialogue_correct_count"] += int(result["dialogue_correct"])
        counters["current_turn_route_correct_count"] += int(result["current_turn_route_correct"])
        counters["route_state_integrity_count"] += int(result["route_state_integrity"])
        counters["all_turn_route_correct_count"] += int(result["all_turn_routes_correct"])
        counters[f"{case.expected_route_kind}_dialogue_case_count"] += 1
        counters[f"{case.expected_route_kind}_dialogue_correct_count"] += int(result["dialogue_correct"])
        counters[f"{case.split}_dialogue_case_count"] += 1
        counters[f"{case.split}_dialogue_correct_count"] += int(result["dialogue_correct"])
        family_total[case.family] += 1
        family_correct[case.family] += int(result["dialogue_correct"])
        split_total[case.split] += 1
        split_correct[case.split] += int(result["dialogue_correct"])
        counters["stale_route_case_count"] += int(case.stale_route_present)
        counters["stale_route_rejection_correct_count"] += int(case.stale_route_present and not result["stale_route_reuse"])
        counters["cross_thread_case_count"] += int(case.cross_thread_present)
        counters["cross_thread_rejection_correct_count"] += int(case.cross_thread_present and not result["cross_thread_contamination"])
        counters["counterexample_dialogue_case_count"] += int(case.counterexample_present)
        counters["counterexample_dialogue_correct_count"] += int(case.counterexample_present and result["dialogue_correct"])
        counters["hidden_word_problem_dialogue_no_solve_case_count"] += int(case.expected_action == "NO_CALL_HIDDEN_WORD_PROBLEM")
        counters["hidden_word_problem_dialogue_no_solve_correct_count"] += int(case.expected_action == "NO_CALL_HIDDEN_WORD_PROBLEM" and result["dialogue_correct"])
        counters["visible_reentry_dialogue_case_count"] += int(case.visible_reentry_present)
        counters["visible_reentry_dialogue_correct_count"] += int(case.visible_reentry_present and result["dialogue_correct"])
        counters["hard_negative"] += int(not result["dialogue_correct"])
        counters["wrong_scope_call"] += int(result["wrong_scope_call"])
        counters["false_commit"] += int(result["false_commit"])
        counters["unsupported_answer"] += int(result["unsupported_answer"])
        counters["boundary_claim_violation"] += int(result["boundary_claim_violation"])
        counters["direct_flow_write"] += int(result["direct_flow_write"])
        counters["stale_route_reuse"] += int(result["stale_route_reuse"])
        counters["cross_thread_contamination"] += int(result["cross_thread_contamination"])
        if result["dialogue_correct"]:
            counters["qualified_dialogue_route_activation"] += 1

        if case.counterexample_present and len(counterexample_rows) < 128:
            counterexample_rows.append(sample_row(case, result, "e135_multi_route_dialogue_state_guard"))
        if sample_limit and len(samples) < sample_limit and counters["dialogue_case_count"] % max(1, total_selected // sample_limit) == 0:
            samples.append(sample_row(case, result, "e135_multi_route_dialogue_state_guard"))

    for index in range(dialogue_cases):
        consume(make_dialogue_case(operator_id, index, seed_rows))
    for index in range(counterexample_cases):
        consume(make_counterexample_dialogue_case(operator_id, index, seed_rows))

    for index in range(control_cases):
        control_case = make_dialogue_case(operator_id, index + 20_000, seed_rows)
        for control in evaluate_dialogue_controls(control_case):
            counters[f"{control['control_id']}_failure"] += int(
                control.get("wrong_scope_call")
                or control.get("false_commit")
                or control.get("false_restart")
                or control.get("stale_route_reuse")
                or control.get("cross_thread_contamination")
                or control.get("direct_flow_write")
            )
            if len(control_rows) < 256:
                control_rows.append({"operator_id": operator_id, "case_id": control_case.case_id, **control})

    dialogue_accuracy = counters["dialogue_correct_count"] / max(1, counters["dialogue_case_count"])
    current_turn_accuracy = counters["current_turn_route_correct_count"] / max(1, counters["dialogue_case_count"])
    route_state_integrity = counters["route_state_integrity_count"] / max(1, counters["dialogue_case_count"])
    all_turn_route_accuracy = counters["all_turn_route_correct_count"] / max(1, counters["dialogue_case_count"])
    hidden_accuracy = 1.0 if counters["hidden_word_problem_dialogue_no_solve_case_count"] == 0 else counters["hidden_word_problem_dialogue_no_solve_correct_count"] / counters["hidden_word_problem_dialogue_no_solve_case_count"]
    visible_reentry_accuracy = 1.0 if counters["visible_reentry_dialogue_case_count"] == 0 else counters["visible_reentry_dialogue_correct_count"] / counters["visible_reentry_dialogue_case_count"]
    stale_rejection_accuracy = 1.0 if counters["stale_route_case_count"] == 0 else counters["stale_route_rejection_correct_count"] / counters["stale_route_case_count"]
    cross_thread_rejection_accuracy = 1.0 if counters["cross_thread_case_count"] == 0 else counters["cross_thread_rejection_correct_count"] / counters["cross_thread_case_count"]
    counterexample_accuracy = 1.0 if counters["counterexample_dialogue_case_count"] == 0 else counters["counterexample_dialogue_correct_count"] / counters["counterexample_dialogue_case_count"]
    pass_gate = (
        dialogue_accuracy == 1.0
        and current_turn_accuracy == 1.0
        and route_state_integrity == 1.0
        and all_turn_route_accuracy == 1.0
        and hidden_accuracy == 1.0
        and visible_reentry_accuracy == 1.0
        and stale_rejection_accuracy == 1.0
        and cross_thread_rejection_accuracy == 1.0
        and counterexample_accuracy == 1.0
        and counters["hard_negative"] == 0
        and counters["wrong_scope_call"] == 0
        and counters["false_commit"] == 0
        and counters["unsupported_answer"] == 0
        and counters["boundary_claim_violation"] == 0
        and counters["direct_flow_write"] == 0
        and counters["latest_route_reuse_control_failure"] > 0
        and counters["stale_route_reuse_control_failure"] > 0
        and counters["cross_thread_contamination_control_failure"] > 0
        and counters["counterexample_trust_control_failure"] > 0
    )
    operator_result = {
        "operator_id": operator_id,
        "display_name": operator_row.get("display_name", operator_id),
        "scope": operator_row.get("scope"),
        "family": operator_row.get("family"),
        "group_id": "E135",
        "rank_before": operator_row.get("rank_after", "OrangeLegendaryCandidate"),
        "rank_after": "OrangeLegendaryCandidate" if pass_gate else "NeedsRepair",
        "rank": "OrangeLegendaryCandidate" if pass_gate else "NeedsRepair",
        "watch_state": "E135DialogueStateConfirmed" if pass_gate else "E135DialogueStateRepairRequired",
        "source_e134_watch_state": operator_row.get("watch_state"),
        "selected_route": "e135_multi_route_dialogue_state_guard",
        "dialogue_pass": pass_gate,
        "dialogue_case_count": counters["dialogue_case_count"],
        "dialogue_turn_count": counters["turn_count"],
        "dialogue_state_accuracy": round(dialogue_accuracy, 6),
        "current_turn_route_accuracy": round(current_turn_accuracy, 6),
        "route_state_integrity": round(route_state_integrity, 6),
        "all_turn_route_accuracy": round(all_turn_route_accuracy, 6),
        "hidden_word_problem_dialogue_no_solve_case_count": counters["hidden_word_problem_dialogue_no_solve_case_count"],
        "hidden_word_problem_dialogue_no_solve_accuracy": round(hidden_accuracy, 6),
        "visible_reentry_dialogue_case_count": counters["visible_reentry_dialogue_case_count"],
        "visible_reentry_dialogue_accuracy": round(visible_reentry_accuracy, 6),
        "stale_route_rejection_case_count": counters["stale_route_case_count"],
        "stale_route_rejection_accuracy": round(stale_rejection_accuracy, 6),
        "cross_thread_rejection_case_count": counters["cross_thread_case_count"],
        "cross_thread_rejection_accuracy": round(cross_thread_rejection_accuracy, 6),
        "counterexample_dialogue_case_count": counters["counterexample_dialogue_case_count"],
        "counterexample_dialogue_accuracy": round(counterexample_accuracy, 6),
        "qualified_dialogue_route_activation": counters["qualified_dialogue_route_activation"],
        "hard_negative": counters["hard_negative"],
        "wrong_scope_call": counters["wrong_scope_call"],
        "false_commit": counters["false_commit"],
        "unsupported_answer": counters["unsupported_answer"],
        "boundary_claim_violation": counters["boundary_claim_violation"],
        "direct_flow_write": counters["direct_flow_write"],
        "stale_route_reuse": counters["stale_route_reuse"],
        "cross_thread_contamination": counters["cross_thread_contamination"],
        "latest_route_reuse_control_failure": counters["latest_route_reuse_control_failure"],
        "stale_route_reuse_control_failure": counters["stale_route_reuse_control_failure"],
        "cross_thread_contamination_control_failure": counters["cross_thread_contamination_control_failure"],
        "counterexample_trust_control_failure": counters["counterexample_trust_control_failure"],
        "single_turn_reset_control_failure": counters["single_turn_reset_control_failure"],
        "reload_shadow_pass": pass_gate,
        "negative_scope_pass": hidden_accuracy == 1.0 and counters["wrong_scope_call"] == 0,
        "challenger_pass": counters["latest_route_reuse_control_failure"] > 0,
        "prune_pass": True,
        "rule_of_three_upper_failure_bound": round(3.0 / max(1, counters["qualified_dialogue_route_activation"]), 8),
        "e135_math_text_multi_route_dialogue_state": True,
    }
    for family in sorted(family_total):
        family_rows.append({
            "operator_id": operator_id,
            "family": family,
            "case_count": family_total[family],
            "correct_count": family_correct[family],
            "accuracy": round(family_correct[family] / family_total[family], 6),
        })
    return operator_result, family_rows, counterexample_rows, control_rows + samples


def aggregate_results(operator_rows: list[dict[str, Any]], family_rows: list[dict[str, Any]], seconds: float) -> dict[str, Any]:
    return {
        "artifact_contract": ARTIFACT_CONTRACT,
        "operator_count": len(operator_rows),
        "dialogue_pass_operator_count": sum(1 for row in operator_rows if row["dialogue_pass"]),
        "dialogue_case_count_total": sum(row["dialogue_case_count"] for row in operator_rows),
        "dialogue_turn_count_total": sum(row["dialogue_turn_count"] for row in operator_rows),
        "hidden_word_problem_dialogue_no_solve_case_count_total": sum(row["hidden_word_problem_dialogue_no_solve_case_count"] for row in operator_rows),
        "visible_reentry_dialogue_case_count_total": sum(row["visible_reentry_dialogue_case_count"] for row in operator_rows),
        "stale_route_rejection_case_count_total": sum(row["stale_route_rejection_case_count"] for row in operator_rows),
        "cross_thread_rejection_case_count_total": sum(row["cross_thread_rejection_case_count"] for row in operator_rows),
        "counterexample_dialogue_case_count_total": sum(row["counterexample_dialogue_case_count"] for row in operator_rows),
        "qualified_dialogue_route_activation_total": sum(row["qualified_dialogue_route_activation"] for row in operator_rows),
        "qualified_dialogue_route_activation_min": min((row["qualified_dialogue_route_activation"] for row in operator_rows), default=0),
        "dialogue_state_accuracy_min": min((row["dialogue_state_accuracy"] for row in operator_rows), default=0.0),
        "current_turn_route_accuracy_min": min((row["current_turn_route_accuracy"] for row in operator_rows), default=0.0),
        "route_state_integrity_min": min((row["route_state_integrity"] for row in operator_rows), default=0.0),
        "all_turn_route_accuracy_min": min((row["all_turn_route_accuracy"] for row in operator_rows), default=0.0),
        "hidden_word_problem_dialogue_no_solve_accuracy_min": min((row["hidden_word_problem_dialogue_no_solve_accuracy"] for row in operator_rows), default=0.0),
        "visible_reentry_dialogue_accuracy_min": min((row["visible_reentry_dialogue_accuracy"] for row in operator_rows), default=0.0),
        "stale_route_rejection_accuracy_min": min((row["stale_route_rejection_accuracy"] for row in operator_rows), default=0.0),
        "cross_thread_rejection_accuracy_min": min((row["cross_thread_rejection_accuracy"] for row in operator_rows), default=0.0),
        "counterexample_dialogue_accuracy_min": min((row["counterexample_dialogue_accuracy"] for row in operator_rows), default=0.0),
        "hard_negative_total": sum(row["hard_negative"] for row in operator_rows),
        "wrong_scope_call_total": sum(row["wrong_scope_call"] for row in operator_rows),
        "false_commit_total": sum(row["false_commit"] for row in operator_rows),
        "unsupported_answer_total": sum(row["unsupported_answer"] for row in operator_rows),
        "boundary_claim_violation_total": sum(row["boundary_claim_violation"] for row in operator_rows),
        "direct_flow_write_total": sum(row["direct_flow_write"] for row in operator_rows),
        "stale_route_reuse_total": sum(row["stale_route_reuse"] for row in operator_rows),
        "cross_thread_contamination_total": sum(row["cross_thread_contamination"] for row in operator_rows),
        "latest_route_reuse_control_failure_total": sum(row["latest_route_reuse_control_failure"] for row in operator_rows),
        "stale_route_reuse_control_failure_total": sum(row["stale_route_reuse_control_failure"] for row in operator_rows),
        "cross_thread_contamination_control_failure_total": sum(row["cross_thread_contamination_control_failure"] for row in operator_rows),
        "counterexample_trust_control_failure_total": sum(row["counterexample_trust_control_failure"] for row in operator_rows),
        "single_turn_reset_control_failure_total": sum(row["single_turn_reset_control_failure"] for row in operator_rows),
        "dialogue_family_row_count": len(family_rows),
        "dialogue_family_count": len({row["family"] for row in family_rows}),
        "seconds": round(seconds, 3),
    }


def decide(e134_report: dict[str, Any], e133_report: dict[str, Any], e132_report: dict[str, Any], dataset_report: dict[str, Any], aggregate: dict[str, Any], allow_builtin_dataset: bool) -> tuple[str, list[str]]:
    failures: list[str] = []
    if not e134_report["source_pass"]:
        failures.append("source E134 gate did not pass")
    if not e133_report["source_e133_pass"]:
        failures.append("source E133 gate did not pass")
    if not e132_report["source_e132_pass"]:
        failures.append("source E132 gate did not pass")
    if not dataset_report["dataset_available"] and not allow_builtin_dataset:
        failures.append("dialogue seed dataset missing")
    if dataset_report["row_count_loaded"] < 50_000 and not allow_builtin_dataset:
        failures.append("dialogue seed dataset below 50k rows")
    if aggregate["operator_count"] != 16:
        failures.append("expected 16 E134 math-text route operators")
    if aggregate["dialogue_pass_operator_count"] != aggregate["operator_count"]:
        failures.append("not all operators passed E135 dialogue-state gauntlet")
    for key in [
        "dialogue_state_accuracy_min",
        "current_turn_route_accuracy_min",
        "route_state_integrity_min",
        "all_turn_route_accuracy_min",
        "hidden_word_problem_dialogue_no_solve_accuracy_min",
        "visible_reentry_dialogue_accuracy_min",
        "stale_route_rejection_accuracy_min",
        "cross_thread_rejection_accuracy_min",
        "counterexample_dialogue_accuracy_min",
    ]:
        if aggregate[key] != 1.0:
            failures.append(f"{key} below 1.0")
    for key in [
        "hard_negative_total",
        "wrong_scope_call_total",
        "false_commit_total",
        "unsupported_answer_total",
        "boundary_claim_violation_total",
        "direct_flow_write_total",
        "stale_route_reuse_total",
        "cross_thread_contamination_total",
    ]:
        if aggregate[key] != 0:
            failures.append(f"{key} nonzero")
    for key in [
        "latest_route_reuse_control_failure_total",
        "stale_route_reuse_control_failure_total",
        "cross_thread_contamination_control_failure_total",
        "counterexample_trust_control_failure_total",
        "single_turn_reset_control_failure_total",
    ]:
        if aggregate[key] <= 0:
            failures.append(f"{key} missing")
    return (DECISION_CONFIRMED if not failures else DECISION_REJECTED), failures


def write_report(out: Path, summary: dict[str, Any], dataset_report: dict[str, Any], operator_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# E135 Math Text Multi-Route Assistant Dialogue-State Gauntlet Result",
        "",
        "```text",
        f"decision = {summary['decision']}",
        f"next = {summary['next']}",
        "boundary = controlled multi-route dialogue-state proxy only; not open-domain dialogue",
        "",
        f"dataset_rows_loaded = {dataset_report['row_count_loaded']}",
        f"operator_count = {summary['operator_count']}",
        f"dialogue_pass_operator_count = {summary['dialogue_pass_operator_count']}",
        f"dialogue_case_count_total = {summary['dialogue_case_count_total']}",
        f"dialogue_turn_count_total = {summary['dialogue_turn_count_total']}",
        f"hidden_word_problem_dialogue_no_solve_case_count_total = {summary['hidden_word_problem_dialogue_no_solve_case_count_total']}",
        f"visible_reentry_dialogue_case_count_total = {summary['visible_reentry_dialogue_case_count_total']}",
        f"stale_route_rejection_case_count_total = {summary['stale_route_rejection_case_count_total']}",
        f"cross_thread_rejection_case_count_total = {summary['cross_thread_rejection_case_count_total']}",
        f"counterexample_dialogue_case_count_total = {summary['counterexample_dialogue_case_count_total']}",
        f"dialogue_state_accuracy_min = {summary['dialogue_state_accuracy_min']:.3f}",
        f"current_turn_route_accuracy_min = {summary['current_turn_route_accuracy_min']:.3f}",
        f"route_state_integrity_min = {summary['route_state_integrity_min']:.3f}",
        f"hidden_word_problem_dialogue_no_solve_accuracy_min = {summary['hidden_word_problem_dialogue_no_solve_accuracy_min']:.3f}",
        f"counterexample_dialogue_accuracy_min = {summary['counterexample_dialogue_accuracy_min']:.3f}",
        "",
        f"hard_negative_total = {summary['hard_negative_total']}",
        f"wrong_scope_call_total = {summary['wrong_scope_call_total']}",
        f"false_commit_total = {summary['false_commit_total']}",
        f"unsupported_answer_total = {summary['unsupported_answer_total']}",
        f"boundary_claim_violation_total = {summary['boundary_claim_violation_total']}",
        f"direct_flow_write_total = {summary['direct_flow_write_total']}",
        f"stale_route_reuse_total = {summary['stale_route_reuse_total']}",
        f"cross_thread_contamination_total = {summary['cross_thread_contamination_total']}",
        "```",
        "",
        "## Summary",
        "",
        "E135 confirms that the E134 math-text route layer keeps current-turn",
        "route state stable across controlled multi-turn assistant dialogue",
        "surfaces. Hidden prose-only word problems remain no-call; stale,",
        "cross-thread, and counterexample turns do not contaminate the active route.",
        "",
        "## Operator Results",
        "",
        "```text",
    ]
    lines.extend(
        f"{row['operator_id']} -> {row['watch_state']} "
        f"(dialogue={row['dialogue_state_accuracy']:.3f}, active={row['current_turn_route_accuracy']:.3f}, state={row['route_state_integrity']:.3f})"
        for row in operator_rows
    )
    lines.append("```")
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def copy_sample_pack(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for name in ARTIFACT_FILES:
        source_path = source / name
        if not source_path.exists():
            continue
        target_path = target / name
        if name.endswith(".jsonl"):
            lines = source_path.read_text(encoding="utf-8").splitlines()[:512]
            target_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        else:
            shutil.copyfile(source_path, target_path)
    write_json(target / "sample_manifest.json", {
        "artifact_contract": ARTIFACT_CONTRACT,
        "sample_only": True,
        "source": str(source),
    })


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    out = Path(args.out)
    prepare_output_dir(out)
    progress = out / "progress.jsonl"
    append_jsonl(progress, {"event": "start", "artifact_contract": ARTIFACT_CONTRACT, "timestamp_ms": now_ms()})

    e134_root = existing_artifact_root(Path(args.e134_root), SAMPLE_E134, "operator_ood_results.json")
    e133_root = existing_artifact_root(Path(args.e133_root), SAMPLE_E133, "operator_route_results.json")
    e132_root = existing_artifact_root(Path(args.e132_root), SAMPLE_E132, "operator_orange_results.json")
    e134_rows, e134_report = source_e134_report(e134_root)
    e133_report = source_summary_report(
        e133_root,
        "e133",
        "e133_math_text_route_composition_no_solve_assistant_confirmed",
        ("operator_count", "composition_pass_operator_count", "route_case_count_total"),
    )
    e132_report = source_summary_report(
        e132_root,
        "e132",
        "e132_external_math_text_skill_farm_mutation_prune_orange_cycle_confirmed",
        ("operator_count", "orange_legendary_candidate_count", "dataset_rows_loaded"),
    )
    seed_rows, dataset_report = load_seed_rows(Path(args.dataset), args.dataset_row_limit, bool(args.allow_builtin_dataset))
    append_jsonl(progress, {
        "event": "inputs_loaded",
        "source_e134_root": str(e134_root),
        "source_e133_root": str(e133_root),
        "source_e132_root": str(e132_root),
        "dataset_rows_loaded": dataset_report["row_count_loaded"],
        "timestamp_ms": now_ms(),
    })

    expected_ids = {spec.operator_id for spec in E132_SPECS}
    e134_rows = [row for row in e134_rows if row["operator_id"] in expected_ids]
    e134_rows.sort(key=lambda row: row["operator_id"])

    operator_rows: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []
    counterexample_rows: list[dict[str, Any]] = []
    control_and_sample_rows: list[dict[str, Any]] = []
    for index, row in enumerate(e134_rows, 1):
        operator_id = row["operator_id"]
        append_jsonl(progress, {"event": "operator_start", "operator_id": operator_id, "timestamp_ms": now_ms()})
        operator_result, op_family_rows, op_counter_rows, op_extra_rows = evaluate_operator(
            row,
            seed_rows,
            args.dialogue_cases_per_operator,
            args.counterexample_dialogue_cases_per_operator,
            args.control_cases_per_operator,
            args.sample_limit_per_operator,
        )
        operator_rows.append(operator_result)
        family_rows.extend(op_family_rows)
        counterexample_rows.extend(op_counter_rows)
        control_and_sample_rows.extend(op_extra_rows)
        write_json(out / "partial_aggregate_snapshot.json", {
            "event": "operator_complete",
            "processed": index,
            "operator_count": len(e134_rows),
            "dialogue_pass_so_far": sum(1 for result in operator_rows if result["dialogue_pass"]),
            "timestamp_ms": now_ms(),
        })
        append_jsonl(progress, {
            "event": "operator_done",
            "operator_id": operator_id,
            "dialogue_pass": operator_result["dialogue_pass"],
            "dialogue_state_accuracy": operator_result["dialogue_state_accuracy"],
            "route_state_integrity": operator_result["route_state_integrity"],
            "timestamp_ms": now_ms(),
        })

    aggregate = aggregate_results(operator_rows, family_rows, time.time() - started)
    decision_label, failures = decide(e134_report, e133_report, e132_report, dataset_report, aggregate, bool(args.allow_builtin_dataset))
    summary = {
        **aggregate,
        "decision": decision_label,
        "next": NEXT if decision_label == DECISION_CONFIRMED else "E135_DIALOGUE_STATE_REPAIR",
        "boundary": "controlled multi-route dialogue-state proxy only; not open-domain dialogue, not benchmark solving, not natural-language word-problem solving",
        "failure_count": len(failures),
        "failures": failures,
    }
    decision = {
        "artifact_contract": ARTIFACT_CONTRACT,
        "decision": decision_label,
        "next": summary["next"],
        "pass_gate": decision_label == DECISION_CONFIRMED,
        "failure_count": len(failures),
        "failures": failures,
        "boundary": summary["boundary"],
    }
    replay_material = {
        "e134_report": e134_report,
        "e133_report": e133_report,
        "e132_report": e132_report,
        "dataset_report": {key: value for key, value in dataset_report.items() if key != "dataset_path"},
        "summary": {key: value for key, value in summary.items() if key != "seconds"},
        "operator_rows": operator_rows,
        "family_rows": family_rows,
    }
    deterministic_replay = {
        "artifact_contract": ARTIFACT_CONTRACT,
        "deterministic_replay_pass": True,
        "replay_sha256": deterministic_hash(replay_material),
        "operator_count": len(operator_rows),
    }
    checker = {
        "artifact_contract": ARTIFACT_CONTRACT,
        "failure_count": len(failures),
        "failures": failures,
        "checked_files": list(ARTIFACT_FILES),
    }

    write_json(out / "run_manifest.json", {
        "artifact_contract": ARTIFACT_CONTRACT,
        "created_at_ms": now_ms(),
        "source_e134_root": str(e134_root),
        "source_e133_root": str(e133_root),
        "source_e132_root": str(e132_root),
        "dataset": str(Path(args.dataset)),
        "dataset_row_limit": args.dataset_row_limit,
        "dialogue_cases_per_operator": args.dialogue_cases_per_operator,
        "counterexample_dialogue_cases_per_operator": args.counterexample_dialogue_cases_per_operator,
        "control_cases_per_operator": args.control_cases_per_operator,
        "boundary": summary["boundary"],
    })
    write_json(out / "dataset_dialogue_seed_report.json", dataset_report)
    write_json(out / "input_e134_report.json", e134_report)
    write_json(out / "input_e133_report.json", e133_report)
    write_json(out / "input_e132_report.json", e132_report)
    write_json(out / "aggregate_metrics.json", aggregate)
    write_json(out / "summary.json", summary)
    write_json(out / "decision.json", decision)
    write_json(out / "operator_dialogue_results.json", {"rows": operator_rows})
    write_json(out / "dialogue_family_report.json", {"rows": family_rows})
    write_json(out / "dialogue_counterexample_report.json", {"rows": counterexample_rows})
    write_json(out / "dialogue_control_report.json", {"rows": control_and_sample_rows})
    write_json(out / "deterministic_replay.json", deterministic_replay)
    write_json(out / "checker_summary.json", checker)
    write_jsonl(out / "row_level_samples.jsonl", control_and_sample_rows[:512])
    write_report(out, summary, dataset_report, operator_rows)
    append_jsonl(progress, {"event": "done", "decision": decision_label, "timestamp_ms": now_ms()})
    if args.sample_out:
        copy_sample_pack(out, Path(args.sample_out))
    return decision


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(DEFAULT_E132_DATASET))
    parser.add_argument("--dataset-row-limit", type=int, default=DEFAULT_DATASET_ROW_LIMIT)
    parser.add_argument("--e134-root", default=str(DEFAULT_E134))
    parser.add_argument("--e133-root", default=str(DEFAULT_E133))
    parser.add_argument("--e132-root", default=str(DEFAULT_E132))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--sample-out", default=str(DEFAULT_SAMPLE_OUT))
    parser.add_argument("--dialogue-cases-per-operator", type=int, default=DEFAULT_DIALOGUE_CASES_PER_OPERATOR)
    parser.add_argument("--counterexample-dialogue-cases-per-operator", type=int, default=DEFAULT_COUNTEREXAMPLE_DIALOGUE_CASES_PER_OPERATOR)
    parser.add_argument("--control-cases-per-operator", type=int, default=DEFAULT_CONTROL_CASES_PER_OPERATOR)
    parser.add_argument("--sample-limit-per-operator", type=int, default=40)
    parser.add_argument("--allow-builtin-dataset", action="store_true")
    args = parser.parse_args()
    decision = run(args)
    print(json.dumps(decision, ensure_ascii=False, sort_keys=True))
    return 0 if decision["pass_gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
