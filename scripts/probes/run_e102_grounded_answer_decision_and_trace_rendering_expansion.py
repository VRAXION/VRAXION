#!/usr/bin/env python3
"""E102 grounded answer decision and trace rendering expansion.

Controlled answer-boundary proxy. E102 teaches Operators that decide whether a
question is answerable from resolved evidence, render a scoped grounded answer,
attach citations, and ASK/DEFER when dependencies are missing, partial, stale,
or unsupported.

This is not open-domain question answering and not general language reasoning.
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


ARTIFACT_CONTRACT = "E102_GROUNDED_ANSWER_DECISION_AND_TRACE_RENDERING_EXPANSION"


@dataclass(frozen=True)
class OperatorSpec:
    operator_id: str
    display_name: str
    family: str
    role: str
    cost: float
    description: str


USEFUL_OPERATORS: tuple[OperatorSpec, ...] = (
    OperatorSpec("query_requirement_mapper_lens", "Query Requirement Mapper Lens", "Lens", "useful", 0.14, "Maps a query to the mechanical evidence requirements needed to answer it."),
    OperatorSpec("resolved_evidence_coverage_guard", "Resolved Evidence Coverage Guard", "Guard", "useful", 0.14, "Requires resolved evidence to cover every query dependency before answering."),
    OperatorSpec("answerability_decision_guard", "Answerability Decision Guard", "Guard", "useful", 0.13, "Chooses ANSWER, ASK, or DEFER from coverage and conflict state."),
    OperatorSpec("grounded_answer_template_scribe", "Grounded Answer Template Scribe", "Scribe", "useful", 0.13, "Renders the answer from resolved evidence without adding unsupported content."),
    OperatorSpec("evidence_citation_link_scribe", "Evidence Citation Link Scribe", "Scribe", "useful", 0.12, "Links every answer claim to a resolved evidence span identifier."),
    OperatorSpec("unsupported_answer_defer_guard", "Unsupported Answer Defer Guard", "Guard", "useful", 0.13, "Blocks answers when any required dependency is unsupported or contradictory."),
    OperatorSpec("ask_when_dependency_missing_scribe", "Ask-When-Dependency-Missing Scribe", "Scribe", "useful", 0.12, "Writes a targeted question for the missing evidence dependency."),
    OperatorSpec("answer_trace_integrity_t_stab", "Answer Trace Integrity T-Stab", "T-Stab", "useful", 0.13, "Stabilizes answer trace, citation, and route history before final output."),
)


CONTROL_OPERATORS: tuple[OperatorSpec, ...] = (
    OperatorSpec("answer_from_memory_guess_control", "Answer From Memory Guess Control", "unsafe_control", "unsafe", 0.03, "Unsafe control: answers without visible resolved evidence."),
    OperatorSpec("answer_without_citation_control", "Answer Without Citation Control", "unsafe_control", "unsafe", 0.03, "Unsafe control: emits a correct-looking answer without citation links."),
    OperatorSpec("ignore_missing_dependency_control", "Ignore Missing Dependency Control", "unsafe_control", "unsafe", 0.04, "Unsafe control: answers despite missing evidence dependencies."),
    OperatorSpec("overconfident_partial_evidence_control", "Overconfident Partial Evidence Control", "unsafe_control", "unsafe", 0.04, "Unsafe control: treats partial evidence as complete."),
    OperatorSpec("always_defer_control", "Always Defer Control", "control", "noop", 0.02, "Control: defers even when evidence fully supports an answer."),
    OperatorSpec("template_only_answer_control", "Template-Only Answer Control", "unsafe_control", "unsafe", 0.03, "Unsafe control: renders a template answer without evidence coverage."),
    OperatorSpec("stale_evidence_answerer", "Stale Evidence Answerer", "unsafe_control", "unsafe", 0.04, "Unsafe control: answers from stale evidence after a newer resolved span exists."),
    OperatorSpec("answerability_echo_clone", "Answerability Echo Clone", "Guard", "redundant", 0.18, "Redundant answerability guard without unique value."),
)


OPERATOR_LIBRARY = USEFUL_OPERATORS + CONTROL_OPERATORS
USEFUL_IDS = tuple(operator.operator_id for operator in USEFUL_OPERATORS)
ALL_OPERATOR_IDS = tuple(operator.operator_id for operator in OPERATOR_LIBRARY)
OPERATOR_BY_ID = {operator.operator_id: operator for operator in OPERATOR_LIBRARY}
UNSAFE_IDS = {operator.operator_id for operator in CONTROL_OPERATORS if operator.role == "unsafe"}


@dataclass(frozen=True)
class AnswerCase:
    case_id: str
    source_split: str
    family: str
    query: str
    resolved_evidence: tuple[str, ...]
    expected_action: str
    expected_answer: str
    expected_citations: tuple[str, ...]
    expected_trace: tuple[str, ...]
    required_operators: tuple[str, ...]
    missing_dependency: bool
    partial_evidence: bool
    stale_evidence_present: bool
    unsupported_scope: bool


def stable_int(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def split_for(seed: int, case_id: str) -> str:
    bucket = stable_int(f"{seed}:{case_id}") % 10
    if bucket < 7:
        return "train"
    if bucket < 9:
        return "validation"
    return "adversarial"


def make_case(seed: int, index: int) -> AnswerCase:
    family = (
        "single_resolved_state_answer",
        "conflict_resolved_with_citation",
        "missing_dependency_ask",
        "partial_evidence_defer",
        "stale_evidence_rejected_answer",
        "multi_evidence_join_answer",
        "scope_unsupported_defer",
        "numeric_trace_answer",
        "latest_source_answer",
        "answerable_no_extra_ask",
    )[index % 10]
    case_id = f"{family}_{seed}_{index:05d}"
    split = split_for(seed, case_id)
    base = ("query_requirement_mapper_lens", "resolved_evidence_coverage_guard", "answerability_decision_guard")
    render = ("grounded_answer_template_scribe", "evidence_citation_link_scribe", "answer_trace_integrity_t_stab")
    unsupported = ("unsupported_answer_defer_guard",)
    ask = ("ask_when_dependency_missing_scribe",)

    if family == "single_resolved_state_answer":
        ev = ("e1: valve K closed",)
        trace = ("requirements:valve K state", "covered:e1", "answer")
        return AnswerCase(case_id, split, family, "What is valve K state?", ev, "ANSWER", "valve K is closed", ("e1",), trace, base + render, False, False, False, False)
    if family == "conflict_resolved_with_citation":
        ev = ("e2: resolved conflict -> valve K closed", "e3: rejected rumor open")
        trace = ("requirements:valve K state", "covered:e2", "rejected:e3", "answer")
        return AnswerCase(case_id, split, family, "What is valve K state?", ev, "ANSWER", "valve K is closed", ("e2",), trace, base + render, False, False, False, False)
    if family == "missing_dependency_ask":
        ev = ("e4: valve J closed",)
        trace = ("requirements:valve K state", "missing:valve K state", "ask")
        return AnswerCase(case_id, split, family, "What is valve K state?", ev, "ASK_FOR_EVIDENCE", "", tuple(), trace, base + ask + unsupported, True, False, False, False)
    if family == "partial_evidence_defer":
        ev = ("e5: valve K was checked",)
        trace = ("requirements:valve K state", "partial:e5", "defer")
        return AnswerCase(case_id, split, family, "What is valve K state?", ev, "DEFER_UNSUPPORTED", "", tuple(), trace, base + unsupported, False, True, False, False)
    if family == "stale_evidence_rejected_answer":
        ev = ("e6_stale: valve K open", "e7_current: valve K closed")
        trace = ("requirements:valve K state", "reject_stale:e6_stale", "covered:e7_current", "answer")
        return AnswerCase(case_id, split, family, "What is valve K state?", ev, "ANSWER", "valve K is closed", ("e7_current",), trace, base + render + unsupported, False, False, True, False)
    if family == "multi_evidence_join_answer":
        ev = ("e8: valve K closed", "e9: source verified")
        trace = ("requirements:state+verification", "covered:e8", "covered:e9", "answer")
        return AnswerCase(case_id, split, family, "What is the verified valve K state?", ev, "ANSWER", "verified valve K state is closed", ("e8", "e9"), trace, base + render, False, False, False, False)
    if family == "scope_unsupported_defer":
        ev = ("e10: visible calc marker valid",)
        trace = ("requirements:word problem solution", "scope_unsupported:e10", "defer")
        return AnswerCase(case_id, split, family, "Solve the hidden word problem.", ev, "DEFER_UNSUPPORTED", "", tuple(), trace, base + unsupported, False, False, False, True)
    if family == "numeric_trace_answer":
        ev = ("e11: <<7*6=42>>",)
        trace = ("requirements:visible calc trace", "covered:e11", "answer")
        return AnswerCase(case_id, split, family, "What does the visible calc trace show?", ev, "ANSWER", "the visible calc trace shows 42", ("e11",), trace, base + render, False, False, False, False)
    if family == "latest_source_answer":
        ev = ("e12_old: trusted source said open", "e13_new: trusted source says closed")
        trace = ("requirements:latest trusted state", "reject_stale:e12_old", "covered:e13_new", "answer")
        return AnswerCase(case_id, split, family, "What is the latest trusted valve K state?", ev, "ANSWER", "latest trusted valve K state is closed", ("e13_new",), trace, base + render + unsupported, False, False, True, False)
    ev = ("e14: valve K closed",)
    trace = ("requirements:valve K state", "covered:e14", "answer")
    return AnswerCase(case_id, split, family, "What is valve K state?", ev, "ANSWER", "valve K is closed", ("e14",), trace, base + render, False, False, False, False)


def generate_cases(seed: int, rows: int) -> list[AnswerCase]:
    return [make_case(seed, index) for index in range(rows)]


def unsafe_prediction(case: AnswerCase, selected: set[str]) -> dict[str, object] | None:
    if "answer_from_memory_guess_control" in selected:
        return {"action": "ANSWER", "answer": "valve K is open", "citations": tuple(), "trace": ("memory_guess",), "coverage_valid": False, "citation_valid": False, "trace_valid": False, "unsupported_answer": True, "false_defer": False, "stale_answer": case.stale_evidence_present}
    if "answer_without_citation_control" in selected and case.expected_action == "ANSWER":
        return {"action": "ANSWER", "answer": case.expected_answer, "citations": tuple(), "trace": case.expected_trace, "coverage_valid": True, "citation_valid": False, "trace_valid": True, "unsupported_answer": True, "false_defer": False, "stale_answer": False}
    if "ignore_missing_dependency_control" in selected and case.missing_dependency:
        return {"action": "ANSWER", "answer": "valve K is closed", "citations": ("guessed",), "trace": ("ignored_missing",), "coverage_valid": False, "citation_valid": False, "trace_valid": False, "unsupported_answer": True, "false_defer": False, "stale_answer": False}
    if "overconfident_partial_evidence_control" in selected and case.partial_evidence:
        return {"action": "ANSWER", "answer": "valve K is closed", "citations": ("partial",), "trace": ("partial_as_complete",), "coverage_valid": False, "citation_valid": False, "trace_valid": False, "unsupported_answer": True, "false_defer": False, "stale_answer": False}
    if "always_defer_control" in selected and case.expected_action == "ANSWER":
        return {"action": "DEFER_UNSUPPORTED", "answer": "", "citations": tuple(), "trace": ("always_defer",), "coverage_valid": True, "citation_valid": True, "trace_valid": False, "unsupported_answer": False, "false_defer": True, "stale_answer": False}
    if "template_only_answer_control" in selected:
        return {"action": "ANSWER", "answer": "the answer is available", "citations": tuple(), "trace": ("template",), "coverage_valid": False, "citation_valid": False, "trace_valid": False, "unsupported_answer": True, "false_defer": False, "stale_answer": False}
    if "stale_evidence_answerer" in selected and case.stale_evidence_present:
        return {"action": "ANSWER", "answer": "valve K is open", "citations": ("stale",), "trace": ("stale_answer",), "coverage_valid": False, "citation_valid": False, "trace_valid": False, "unsupported_answer": True, "false_defer": False, "stale_answer": True}
    return None


def predict(case: AnswerCase, selected: set[str]) -> dict[str, object]:
    unsafe = unsafe_prediction(case, selected)
    if unsafe:
        return unsafe
    required = set(case.required_operators)
    if required <= selected:
        return {"action": case.expected_action, "answer": case.expected_answer, "citations": case.expected_citations, "trace": case.expected_trace, "coverage_valid": True, "citation_valid": True, "trace_valid": True, "unsupported_answer": False, "false_defer": False, "stale_answer": False}
    if "ask_when_dependency_missing_scribe" in selected and case.missing_dependency:
        return {"action": "ASK_FOR_EVIDENCE", "answer": "", "citations": tuple(), "trace": ("missing:valve K state", "ask"), "coverage_valid": False, "citation_valid": True, "trace_valid": False, "unsupported_answer": False, "false_defer": False, "stale_answer": False}
    if "unsupported_answer_defer_guard" in selected and (case.partial_evidence or case.unsupported_scope):
        return {"action": "DEFER_UNSUPPORTED", "answer": "", "citations": tuple(), "trace": ("defer_unsupported",), "coverage_valid": False, "citation_valid": True, "trace_valid": False, "unsupported_answer": False, "false_defer": False, "stale_answer": False}
    return {"action": "DEFER_UNSUPPORTED", "answer": "", "citations": tuple(), "trace": ("insufficient_answerer",), "coverage_valid": False, "citation_valid": False, "trace_valid": False, "unsupported_answer": False, "false_defer": case.expected_action == "ANSWER", "stale_answer": False}


def evaluate(selected: set[str], cases: list[AnswerCase], split: str | None = None) -> dict[str, float]:
    rows = [case for case in cases if split is None or case.source_split == split]
    if not rows:
        return {"answer_decision_success": 0.0, "answer_accuracy": 0.0, "requirement_coverage_validity": 0.0, "citation_validity": 0.0, "trace_integrity": 0.0, "unsupported_answer_rate": 0.0, "false_defer_rate": 0.0, "stale_answer_rate": 0.0, "utility": -1.0}
    success = answer_acc = coverage = citation = trace = 0
    unsupported = false_defer = stale = 0
    partial = 0.0
    for case in rows:
        pred = predict(case, selected)
        required = set(case.required_operators)
        partial += len(required & selected) / max(1, len(required))
        row_success = (
            pred["action"] == case.expected_action
            and pred["answer"] == case.expected_answer
            and tuple(pred["citations"]) == case.expected_citations
            and tuple(pred["trace"]) == case.expected_trace
            and pred["coverage_valid"]
            and pred["citation_valid"]
            and pred["trace_valid"]
            and not pred["unsupported_answer"]
            and not pred["false_defer"]
            and not pred["stale_answer"]
        )
        success += int(row_success)
        answer_acc += int(pred["answer"] == case.expected_answer and pred["action"] == case.expected_action)
        coverage += int(pred["coverage_valid"])
        citation += int(pred["citation_valid"] and tuple(pred["citations"]) == case.expected_citations)
        trace += int(pred["trace_valid"] and tuple(pred["trace"]) == case.expected_trace)
        unsupported += int(pred["unsupported_answer"])
        false_defer += int(pred["false_defer"])
        stale += int(pred["stale_answer"])
    count = len(rows)
    cost = sum(OPERATOR_BY_ID[operator_id].cost for operator_id in selected)
    score = success / count
    partial_score = partial / count
    utility = score + 0.30 * partial_score - 1.7 * (unsupported / count) - 0.9 * (false_defer / count) - 1.3 * (stale / count) - 0.01 * cost
    return {
        "answer_decision_success": round(score, 6),
        "answer_accuracy": round(answer_acc / count, 6),
        "requirement_coverage_validity": round(coverage / count, 6),
        "citation_validity": round(citation / count, 6),
        "trace_integrity": round(trace / count, 6),
        "unsupported_answer_rate": round(unsupported / count, 6),
        "false_defer_rate": round(false_defer / count, 6),
        "stale_answer_rate": round(stale / count, 6),
        "partial_answer_score": round(partial_score, 6),
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
        unsupported_deltas = []
        for result in seed_results:
            selected = set(result["selected"])  # type: ignore[arg-type]
            cases = [AnswerCase(**case) for case in result["cases"]]  # type: ignore[arg-type]
            full = evaluate(selected, cases, "validation")
            ablated = evaluate(selected - {operator_id}, cases, "validation")
            losses.append(full["answer_decision_success"] - ablated["answer_decision_success"])
            unsupported_deltas.append(ablated["unsupported_answer_rate"] - full["unsupported_answer_rate"])
        summary[operator_id] = {
            "mean_answer_decision_success_loss": round(statistics.mean(losses), 6),
            "mean_unsupported_answer_delta": round(statistics.mean(unsupported_deltas), 6),
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
        "validation_answer_decision_success_min": min(vals("validation", "answer_decision_success")),
        "validation_answer_decision_success_mean": round(statistics.mean(vals("validation", "answer_decision_success")), 6),
        "adversarial_answer_decision_success_min": min(vals("adversarial", "answer_decision_success")),
        "adversarial_answer_decision_success_mean": round(statistics.mean(vals("adversarial", "answer_decision_success")), 6),
        "validation_answer_accuracy_min": min(vals("validation", "answer_accuracy")),
        "validation_requirement_coverage_validity_min": min(vals("validation", "requirement_coverage_validity")),
        "validation_citation_validity_min": min(vals("validation", "citation_validity")),
        "validation_trace_integrity_min": min(vals("validation", "trace_integrity")),
        "adversarial_unsupported_answer_rate_max": max(vals("adversarial", "unsupported_answer_rate")),
        "adversarial_false_defer_rate_max": max(vals("adversarial", "false_defer_rate")),
        "adversarial_stale_answer_rate_max": max(vals("adversarial", "stale_answer_rate")),
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
        "validation_answer_decision_success_min",
        "adversarial_answer_decision_success_min",
        "validation_answer_accuracy_min",
        "validation_requirement_coverage_validity_min",
        "validation_citation_validity_min",
        "validation_trace_integrity_min",
    ]:
        if agg[key] != 1.0:
            failures.append(f"{key} below 1.0")
    for key in ["adversarial_unsupported_answer_rate_max", "adversarial_false_defer_rate_max", "adversarial_stale_answer_rate_max"]:
        if agg[key] != 0.0:
            failures.append(f"{key} nonzero")
    unsafe_final = sum(1 for result in seed_results for operator_id in result["selected"] if operator_id in UNSAFE_IDS)  # type: ignore[operator]
    if unsafe_final:
        failures.append("unsafe operator selected")
    decision = "e102_grounded_answer_decision_expansion_confirmed" if not failures else "e102_grounded_answer_decision_incomplete"
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
        "grounded_answer_decision": True,
        "open_domain_question_answering": False,
        "direct_answer_without_evidence_allowed": False,
        "requires_citation_trace": True,
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
            typed_case = AnswerCase(**case)
            pred = predict(typed_case, set(result["selected"]))  # type: ignore[arg-type]
            append_jsonl(out / "row_level_samples.jsonl", {
                "seed": result["seed"],
                "case_id": typed_case.case_id,
                "family": typed_case.family,
                "query": typed_case.query,
                "resolved_evidence": typed_case.resolved_evidence,
                "expected_action": typed_case.expected_action,
                "expected_answer": typed_case.expected_answer,
                "expected_citations": typed_case.expected_citations,
                "expected_trace": typed_case.expected_trace,
                "predicted": pred,
            })
            sample_rows += 1
            if sample_rows >= 480:
                break
        if sample_rows >= 480:
            break
    report = [
        "# E102 Grounded Answer Decision And Trace Rendering Expansion Result",
        "",
        f"decision = `{decision}`",
        "",
        "Boundary: controlled grounded answer decision, not open-domain question answering.",
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
    parser.add_argument("--out", default="target/pilot_wave/e102_grounded_answer_decision_and_trace_rendering_expansion")
    parser.add_argument("--artifact-sample-dir", default="docs/research/artifact_samples/e102_grounded_answer_decision_and_trace_rendering_expansion")
    parser.add_argument("--seeds", default="110201,110202,110203,110204,110205,110206,110207,110208,110209,110210,110211,110212,110213,110214,110215,110216")
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
        "boundary": "controlled grounded answer decision probe; not open-domain question answering",
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
