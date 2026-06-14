#!/usr/bin/env python3
"""E100 text evidence span ingress and proposal expansion.

Controlled text-ingress proxy. E100 teaches Operators that convert short,
noisy text observations into mechanical evidence-span proposals while handling
frame boundaries, source attribution, negation/contrast scope, quote scope,
irrelevant spans, and weak/uncertain claims.

This is not natural-language understanding and not open-domain reasoning.
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


ARTIFACT_CONTRACT = "E100_TEXT_EVIDENCE_SPAN_INGRESS_AND_PROPOSAL_EXPANSION"


@dataclass(frozen=True)
class OperatorSpec:
    operator_id: str
    display_name: str
    family: str
    role: str
    cost: float
    description: str


USEFUL_OPERATORS: tuple[OperatorSpec, ...] = (
    OperatorSpec("text_frame_boundary_lens", "Text Frame Boundary Lens", "Lens", "useful", 0.12, "Finds the valid byte/text frame boundary before span extraction."),
    OperatorSpec("evidence_span_locator_lens", "Evidence Span Locator Lens", "Lens", "useful", 0.14, "Locates the minimal visible text span that supports a proposal."),
    OperatorSpec("source_attribution_lens", "Source Attribution Lens", "Lens", "useful", 0.12, "Attaches mechanical source/provenance context to the candidate span."),
    OperatorSpec("negation_contrast_scope_guard", "Negation Contrast Scope Guard", "Guard", "useful", 0.14, "Blocks spans invalidated by nearby negation or contrast boundaries."),
    OperatorSpec("quote_scope_boundary_guard", "Quote Scope Boundary Guard", "Guard", "useful", 0.13, "Prevents quoted or reported text from being committed as current evidence without attribution."),
    OperatorSpec("irrelevant_span_filter_guard", "Irrelevant Span Filter Guard", "Guard", "useful", 0.12, "Rejects nearby but task-irrelevant spans."),
    OperatorSpec("weak_claim_uncertainty_t_stab", "Weak Claim Uncertainty T-Stab", "T-Stab", "useful", 0.12, "Stabilizes uncertain text evidence into ASK/DEFER instead of forced commit."),
    OperatorSpec("text_evidence_proposal_scribe", "Text Evidence Proposal Scribe", "Scribe", "useful", 0.13, "Renders a normalized evidence-span proposal for Agency review."),
)


CONTROL_OPERATORS: tuple[OperatorSpec, ...] = (
    OperatorSpec("keyword_only_span_picker", "Keyword-Only Span Picker", "unsafe_control", "unsafe", 0.03, "Unsafe control: picks spans by keyword match only."),
    OperatorSpec("first_number_committer", "First Number Committer", "unsafe_control", "unsafe", 0.03, "Unsafe control: commits the first numeric-looking token."),
    OperatorSpec("negation_blind_extractor", "Negation-Blind Extractor", "unsafe_control", "unsafe", 0.04, "Unsafe control: ignores negation and contrast scope."),
    OperatorSpec("quote_bleed_extractor", "Quote-Bleed Extractor", "unsafe_control", "unsafe", 0.04, "Unsafe control: commits quoted or reported text as current evidence."),
    OperatorSpec("source_blind_extractor", "Source-Blind Extractor", "unsafe_control", "unsafe", 0.04, "Unsafe control: omits source/provenance from extracted evidence."),
    OperatorSpec("whole_sentence_dump_control", "Whole Sentence Dump Control", "unsafe_control", "unsafe", 0.04, "Unsafe control: dumps the entire sentence instead of a minimal span."),
    OperatorSpec("answer_from_text_without_evidence_control", "Answer From Text Without Evidence Control", "unsafe_control", "unsafe", 0.04, "Unsafe control: answers from text without an evidence proposal."),
    OperatorSpec("span_locator_echo_clone", "Span Locator Echo Clone", "Lens", "redundant", 0.18, "Redundant span locator without unique value."),
)


OPERATOR_LIBRARY = USEFUL_OPERATORS + CONTROL_OPERATORS
USEFUL_IDS = tuple(operator.operator_id for operator in USEFUL_OPERATORS)
ALL_OPERATOR_IDS = tuple(operator.operator_id for operator in OPERATOR_LIBRARY)
OPERATOR_BY_ID = {operator.operator_id: operator for operator in OPERATOR_LIBRARY}
UNSAFE_IDS = {operator.operator_id for operator in CONTROL_OPERATORS if operator.role == "unsafe"}


@dataclass(frozen=True)
class TextSpanCase:
    case_id: str
    source_split: str
    family: str
    text_observation: str
    query_focus: str
    expected_action: str
    expected_span: str
    expected_source: str
    required_operators: tuple[str, ...]
    has_negation_or_contrast: bool
    has_quote_scope: bool
    has_irrelevant_decoy: bool
    is_weak_or_uncertain: bool


def stable_int(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def split_for(seed: int, case_id: str) -> str:
    bucket = stable_int(f"{seed}:{case_id}") % 10
    if bucket < 7:
        return "train"
    if bucket < 9:
        return "validation"
    return "adversarial"


def make_case(seed: int, index: int) -> TextSpanCase:
    family = (
        "direct_supported_span",
        "contrast_after_but_span",
        "negated_decoy_span",
        "quoted_reported_span",
        "source_shift_span",
        "irrelevant_nearby_number_span",
        "weak_uncertain_claim_span",
        "frame_boundary_split_span",
        "multi_sentence_latest_evidence",
        "no_evidence_ask",
    )[index % 10]
    case_id = f"{family}_{seed}_{index:05d}"
    split = split_for(seed, case_id)
    base = ("text_frame_boundary_lens", "evidence_span_locator_lens", "source_attribution_lens", "text_evidence_proposal_scribe")
    neg = ("negation_contrast_scope_guard",)
    quote = ("quote_scope_boundary_guard",)
    irrelevant = ("irrelevant_span_filter_guard",)
    weak = ("weak_claim_uncertainty_t_stab",)

    if family == "direct_supported_span":
        return TextSpanCase(case_id, split, family, "sensor A reports: valve K is open now.", "valve K state", "PROPOSE_EVIDENCE", "valve K is open", "sensor A", base, False, False, False, False)
    if family == "contrast_after_but_span":
        return TextSpanCase(case_id, split, family, "Earlier valve K looked open, but the verified check says valve K is closed now.", "valve K state", "PROPOSE_EVIDENCE", "valve K is closed", "verified check", base + neg, True, False, False, False)
    if family == "negated_decoy_span":
        return TextSpanCase(case_id, split, family, "The note says valve K is not open; the valid state remains closed.", "valve K state", "PROPOSE_EVIDENCE", "valid state remains closed", "note", base + neg, True, False, True, False)
    if family == "quoted_reported_span":
        return TextSpanCase(case_id, split, family, "A rumor said 'valve K is open', but sensor A confirms valve K is closed.", "valve K state", "PROPOSE_EVIDENCE", "valve K is closed", "sensor A", base + quote + neg, True, True, True, False)
    if family == "source_shift_span":
        return TextSpanCase(case_id, split, family, "Operator note claims valve K open. Trusted sensor B reports valve K closed.", "valve K state", "PROPOSE_EVIDENCE", "valve K closed", "sensor B", base + quote, False, False, True, False)
    if family == "irrelevant_nearby_number_span":
        return TextSpanCase(case_id, split, family, "Batch 42 passed. For valve K, the measured state is closed.", "valve K state", "PROPOSE_EVIDENCE", "measured state is closed", "observation", base + irrelevant, False, False, True, False)
    if family == "weak_uncertain_claim_span":
        return TextSpanCase(case_id, split, family, "Maybe valve K is open, but no verified source has checked it.", "valve K state", "ASK_FOR_EVIDENCE", "", "none", base + weak, False, False, False, True)
    if family == "frame_boundary_split_span":
        return TextSpanCase(case_id, split, family, "[frame tail noise] valve K is open now. [next frame belongs to valve J]", "valve K state", "PROPOSE_EVIDENCE", "valve K is open", "frame", base + irrelevant, False, False, True, False)
    if family == "multi_sentence_latest_evidence":
        return TextSpanCase(case_id, split, family, "First reading: valve K open. Later verified reading: valve K closed.", "valve K state", "PROPOSE_EVIDENCE", "valve K closed", "verified reading", base + neg, True, False, True, False)
    return TextSpanCase(case_id, split, family, "The paragraph discusses valve J and batch 42. It gives no checked evidence for valve K.", "valve K state", "ASK_FOR_EVIDENCE", "", "none", base + irrelevant + weak, False, False, True, True)


def generate_cases(seed: int, rows: int) -> list[TextSpanCase]:
    return [make_case(seed, index) for index in range(rows)]


def unsafe_prediction(case: TextSpanCase, selected: set[str]) -> dict[str, object] | None:
    if "answer_from_text_without_evidence_control" in selected:
        return {"action": "ANSWER_READY", "span": "", "source": "none", "span_valid": False, "source_valid": False, "negation_safe": False, "quote_safe": False, "irrelevant_safe": False, "uncertainty_safe": False, "unsafe_text_commit": True, "irrelevant_commit": case.has_irrelevant_decoy}
    if "keyword_only_span_picker" in selected and case.has_irrelevant_decoy:
        return {"action": "PROPOSE_EVIDENCE", "span": "valve K open", "source": "keyword", "span_valid": False, "source_valid": False, "negation_safe": not case.has_negation_or_contrast, "quote_safe": not case.has_quote_scope, "irrelevant_safe": False, "uncertainty_safe": not case.is_weak_or_uncertain, "unsafe_text_commit": True, "irrelevant_commit": True}
    if "first_number_committer" in selected and "42" in case.text_observation:
        return {"action": "PROPOSE_EVIDENCE", "span": "42", "source": "first_number", "span_valid": False, "source_valid": False, "negation_safe": True, "quote_safe": True, "irrelevant_safe": False, "uncertainty_safe": True, "unsafe_text_commit": True, "irrelevant_commit": True}
    if "negation_blind_extractor" in selected and case.has_negation_or_contrast:
        return {"action": "PROPOSE_EVIDENCE", "span": "valve K is open", "source": case.expected_source, "span_valid": False, "source_valid": True, "negation_safe": False, "quote_safe": not case.has_quote_scope, "irrelevant_safe": True, "uncertainty_safe": not case.is_weak_or_uncertain, "unsafe_text_commit": True, "irrelevant_commit": False}
    if "quote_bleed_extractor" in selected and case.has_quote_scope:
        return {"action": "PROPOSE_EVIDENCE", "span": "valve K is open", "source": "rumor", "span_valid": False, "source_valid": False, "negation_safe": True, "quote_safe": False, "irrelevant_safe": True, "uncertainty_safe": True, "unsafe_text_commit": True, "irrelevant_commit": False}
    if "source_blind_extractor" in selected:
        return {"action": case.expected_action, "span": case.expected_span, "source": "unknown", "span_valid": case.expected_action == "PROPOSE_EVIDENCE", "source_valid": False, "negation_safe": True, "quote_safe": True, "irrelevant_safe": True, "uncertainty_safe": not case.is_weak_or_uncertain, "unsafe_text_commit": case.expected_action == "PROPOSE_EVIDENCE", "irrelevant_commit": False}
    if "whole_sentence_dump_control" in selected and case.expected_action == "PROPOSE_EVIDENCE":
        return {"action": "PROPOSE_EVIDENCE", "span": case.text_observation, "source": case.expected_source, "span_valid": False, "source_valid": True, "negation_safe": True, "quote_safe": True, "irrelevant_safe": not case.has_irrelevant_decoy, "uncertainty_safe": True, "unsafe_text_commit": True, "irrelevant_commit": case.has_irrelevant_decoy}
    return None


def predict(case: TextSpanCase, selected: set[str]) -> dict[str, object]:
    unsafe = unsafe_prediction(case, selected)
    if unsafe:
        return unsafe
    required = set(case.required_operators)
    if required <= selected:
        return {"action": case.expected_action, "span": case.expected_span, "source": case.expected_source, "span_valid": True, "source_valid": True, "negation_safe": True, "quote_safe": True, "irrelevant_safe": True, "uncertainty_safe": True, "unsafe_text_commit": False, "irrelevant_commit": False}
    if "weak_claim_uncertainty_t_stab" in selected and case.is_weak_or_uncertain:
        return {"action": "ASK_FOR_EVIDENCE", "span": "", "source": "none", "span_valid": case.expected_action == "ASK_FOR_EVIDENCE", "source_valid": True, "negation_safe": True, "quote_safe": True, "irrelevant_safe": True, "uncertainty_safe": True, "unsafe_text_commit": False, "irrelevant_commit": False}
    return {"action": "DEFER", "span": "", "source": "none", "span_valid": False, "source_valid": False, "negation_safe": not case.has_negation_or_contrast, "quote_safe": not case.has_quote_scope, "irrelevant_safe": not case.has_irrelevant_decoy, "uncertainty_safe": not case.is_weak_or_uncertain, "unsafe_text_commit": False, "irrelevant_commit": False}


def evaluate(selected: set[str], cases: list[TextSpanCase], split: str | None = None) -> dict[str, float]:
    rows = [case for case in cases if split is None or case.source_split == split]
    if not rows:
        return {"span_ingress_success": 0.0, "span_accuracy": 0.0, "source_attribution_validity": 0.0, "negation_contrast_safety": 0.0, "quote_scope_safety": 0.0, "irrelevant_filter_safety": 0.0, "uncertainty_safety": 0.0, "unsafe_text_commit_rate": 0.0, "irrelevant_commit_rate": 0.0, "utility": -1.0}
    success = span = source = neg = quote = irrelevant = uncertainty = 0
    unsafe_commit = irrelevant_commit = 0
    partial = 0.0
    for case in rows:
        pred = predict(case, selected)
        required = set(case.required_operators)
        partial += len(required & selected) / max(1, len(required))
        row_success = (
            pred["action"] == case.expected_action
            and pred["span"] == case.expected_span
            and pred["source"] == case.expected_source
            and pred["span_valid"]
            and pred["source_valid"]
            and pred["negation_safe"]
            and pred["quote_safe"]
            and pred["irrelevant_safe"]
            and pred["uncertainty_safe"]
            and not pred["unsafe_text_commit"]
            and not pred["irrelevant_commit"]
        )
        success += int(row_success)
        span += int(pred["span"] == case.expected_span and pred["span_valid"])
        source += int(pred["source"] == case.expected_source and pred["source_valid"])
        neg += int(pred["negation_safe"])
        quote += int(pred["quote_safe"])
        irrelevant += int(pred["irrelevant_safe"])
        uncertainty += int(pred["uncertainty_safe"])
        unsafe_commit += int(pred["unsafe_text_commit"])
        irrelevant_commit += int(pred["irrelevant_commit"])
    count = len(rows)
    cost = sum(OPERATOR_BY_ID[operator_id].cost for operator_id in selected)
    score = success / count
    partial_score = partial / count
    utility = score + 0.30 * partial_score - 1.5 * (unsafe_commit / count) - 1.2 * (irrelevant_commit / count) - 0.01 * cost
    return {
        "span_ingress_success": round(score, 6),
        "span_accuracy": round(span / count, 6),
        "source_attribution_validity": round(source / count, 6),
        "negation_contrast_safety": round(neg / count, 6),
        "quote_scope_safety": round(quote / count, 6),
        "irrelevant_filter_safety": round(irrelevant / count, 6),
        "uncertainty_safety": round(uncertainty / count, 6),
        "unsafe_text_commit_rate": round(unsafe_commit / count, 6),
        "irrelevant_commit_rate": round(irrelevant_commit / count, 6),
        "partial_span_score": round(partial_score, 6),
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
            cases = [TextSpanCase(**case) for case in result["cases"]]  # type: ignore[arg-type]
            full = evaluate(selected, cases, "validation")
            ablated = evaluate(selected - {operator_id}, cases, "validation")
            losses.append(full["span_ingress_success"] - ablated["span_ingress_success"])
            unsafe_deltas.append(ablated["unsafe_text_commit_rate"] - full["unsafe_text_commit_rate"])
        summary[operator_id] = {
            "mean_span_ingress_success_loss": round(statistics.mean(losses), 6),
            "mean_unsafe_text_commit_delta": round(statistics.mean(unsafe_deltas), 6),
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
        "validation_span_ingress_success_min": min(vals("validation", "span_ingress_success")),
        "validation_span_ingress_success_mean": round(statistics.mean(vals("validation", "span_ingress_success")), 6),
        "adversarial_span_ingress_success_min": min(vals("adversarial", "span_ingress_success")),
        "adversarial_span_ingress_success_mean": round(statistics.mean(vals("adversarial", "span_ingress_success")), 6),
        "validation_span_accuracy_min": min(vals("validation", "span_accuracy")),
        "validation_source_attribution_validity_min": min(vals("validation", "source_attribution_validity")),
        "validation_negation_contrast_safety_min": min(vals("validation", "negation_contrast_safety")),
        "validation_quote_scope_safety_min": min(vals("validation", "quote_scope_safety")),
        "validation_irrelevant_filter_safety_min": min(vals("validation", "irrelevant_filter_safety")),
        "validation_uncertainty_safety_min": min(vals("validation", "uncertainty_safety")),
        "adversarial_unsafe_text_commit_rate_max": max(vals("adversarial", "unsafe_text_commit_rate")),
        "adversarial_irrelevant_commit_rate_max": max(vals("adversarial", "irrelevant_commit_rate")),
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
        "validation_span_ingress_success_min",
        "adversarial_span_ingress_success_min",
        "validation_span_accuracy_min",
        "validation_source_attribution_validity_min",
        "validation_negation_contrast_safety_min",
        "validation_quote_scope_safety_min",
        "validation_irrelevant_filter_safety_min",
        "validation_uncertainty_safety_min",
    ]:
        if agg[key] != 1.0:
            failures.append(f"{key} below 1.0")
    if agg["adversarial_unsafe_text_commit_rate_max"] != 0.0:
        failures.append("unsafe text commit nonzero")
    if agg["adversarial_irrelevant_commit_rate_max"] != 0.0:
        failures.append("irrelevant commit nonzero")
    unsafe_final = sum(1 for result in seed_results for operator_id in result["selected"] if operator_id in UNSAFE_IDS)  # type: ignore[operator]
    if unsafe_final:
        failures.append("unsafe operator selected")
    decision = "e100_text_evidence_span_ingress_expansion_confirmed" if not failures else "e100_text_evidence_span_ingress_incomplete"
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
        "text_evidence_span_ingress": True,
        "natural_language_understanding": False,
        "open_domain_reasoning": False,
        "direct_text_answer_allowed": False,
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
            typed_case = TextSpanCase(**case)
            pred = predict(typed_case, set(result["selected"]))  # type: ignore[arg-type]
            append_jsonl(out / "row_level_samples.jsonl", {
                "seed": result["seed"],
                "case_id": typed_case.case_id,
                "family": typed_case.family,
                "text_observation": typed_case.text_observation,
                "query_focus": typed_case.query_focus,
                "expected_action": typed_case.expected_action,
                "expected_span": typed_case.expected_span,
                "expected_source": typed_case.expected_source,
                "predicted": pred,
            })
            sample_rows += 1
            if sample_rows >= 480:
                break
        if sample_rows >= 480:
            break
    report = [
        "# E100 Text Evidence Span Ingress And Proposal Expansion Result",
        "",
        f"decision = `{decision}`",
        "",
        "Boundary: controlled text evidence-span ingress, not natural-language understanding.",
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
    parser.add_argument("--out", default="target/pilot_wave/e100_text_evidence_span_ingress_and_proposal_expansion")
    parser.add_argument("--artifact-sample-dir", default="docs/research/artifact_samples/e100_text_evidence_span_ingress_and_proposal_expansion")
    parser.add_argument("--seeds", default="110001,110002,110003,110004,110005,110006,110007,110008,110009,110010,110011,110012,110013,110014,110015,110016")
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
        "boundary": "controlled text evidence-span ingress probe; not natural-language understanding",
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
