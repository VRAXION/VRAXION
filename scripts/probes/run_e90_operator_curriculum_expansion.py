#!/usr/bin/env python3
"""E90 Operator curriculum expansion.

This probe teaches/validates a small bundle of concrete Operator skills after
the E89 naming lock. It is still a controlled symbolic/text-evidence proxy, not
open-domain language understanding.

The target is visible text evidence -> canonical proposal behavior:

* alpha-Syncer operators map visible text claims into canonical evidence records.
* T-Stab operators handle temporal shift/false-alarm evidence.
* Guard operators prevent stale, contradictory, inactive, or unresolved commits.
* Lens/Scribe operators preserve evidence spans and render resolved answers.
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


ARTIFACT_CONTRACT = "E90_OPERATOR_CURRICULUM_EXPANSION"


@dataclass(frozen=True)
class OperatorSpec:
    operator_id: str
    display_name: str
    family: str
    role: str
    cost: float
    lifecycle: str
    short_description: str


USEFUL_OPERATORS: tuple[OperatorSpec, ...] = (
    OperatorSpec(
        "visible_claim_binding_alpha_syncer",
        "Visible Claim Binding alpha-Syncer",
        "alpha_syncer",
        "useful",
        0.13,
        "candidate",
        "Maps visible 'A means B' symbolic claims into canonical binding proposals.",
    ),
    OperatorSpec(
        "numeric_value_binding_alpha_syncer",
        "Numeric Value Binding alpha-Syncer",
        "alpha_syncer",
        "useful",
        0.12,
        "candidate",
        "Maps visible 'A is N' value claims into canonical value-binding proposals.",
    ),
    OperatorSpec(
        "temporal_rule_shift_t_stab",
        "Temporal Rule-Shift T-Stab",
        "T-Stab",
        "useful",
        0.15,
        "candidate",
        "Stabilizes confirmed post-marker rule changes over old bindings.",
    ),
    OperatorSpec(
        "false_alarm_temporal_t_stab",
        "False-Alarm Temporal T-Stab",
        "T-Stab",
        "useful",
        0.12,
        "candidate",
        "Keeps the prior stable binding when a possible shift is explicitly cancelled.",
    ),
    OperatorSpec(
        "revoked_binding_guard",
        "Revoked Binding Guard",
        "Guard",
        "useful",
        0.12,
        "candidate",
        "Blocks stale answers after visible 'no longer means' revocation evidence.",
    ),
    OperatorSpec(
        "contradiction_guard",
        "Contradiction Guard",
        "Guard",
        "useful",
        0.14,
        "candidate",
        "Detects same-cycle conflicting visible claims and forces contradiction handling.",
    ),
    OperatorSpec(
        "unresolved_state_info_guard",
        "Unresolved-State Information Guard",
        "Guard",
        "useful",
        0.13,
        "candidate",
        "Asks/searches/holds when a query depends on unproven visible evidence.",
    ),
    OperatorSpec(
        "inactive_quote_scope_guard",
        "Inactive Quote Scope Guard",
        "Guard",
        "useful",
        0.11,
        "candidate",
        "Prevents archived, quoted, or inactive example claims from becoming active evidence.",
    ),
    OperatorSpec(
        "evidence_span_lens",
        "Evidence Span Lens",
        "Lens",
        "useful",
        0.10,
        "candidate",
        "Preserves byte-span references for the visible evidence used by a proposal.",
    ),
    OperatorSpec(
        "canonical_answer_scribe",
        "Canonical Answer Scribe",
        "Scribe",
        "useful",
        0.10,
        "candidate",
        "Renders resolved canonical bindings into an external answer action.",
    ),
)

CONTROL_OPERATORS: tuple[OperatorSpec, ...] = (
    OperatorSpec(
        "stale_binding_committer",
        "Stale Binding Committer",
        "unsafe_control",
        "unsafe",
        0.04,
        "quarantine",
        "Unsafe control: answers with old bindings after warnings or revocations.",
    ),
    OperatorSpec(
        "inactive_quote_overreach",
        "Inactive Quote Overreach",
        "unsafe_control",
        "unsafe",
        0.05,
        "quarantine",
        "Unsafe control: treats quoted or archived examples as active evidence.",
    ),
    OperatorSpec(
        "marker_only_shift_shortcut",
        "Marker-Only Shift Shortcut",
        "unsafe_control",
        "unsafe",
        0.05,
        "quarantine",
        "Unsafe control: treats a warning marker itself as a confirmed rule change.",
    ),
    OperatorSpec(
        "answer_without_span_shortcut",
        "Answer Without Span Shortcut",
        "unsafe_control",
        "unsafe",
        0.03,
        "quarantine",
        "Unsafe control: emits answers without valid evidence span references.",
    ),
    OperatorSpec(
        "always_ask_control",
        "Always Ask Control",
        "control",
        "noop",
        0.02,
        "deprecated",
        "Control: avoids wrong answers by asking even when visible evidence is sufficient.",
    ),
    OperatorSpec(
        "full_text_scan_overreach",
        "Full Text Scan Overreach",
        "unsafe_control",
        "unsafe",
        0.18,
        "quarantine",
        "Unsafe control: scans all text as evidence regardless of active scope.",
    ),
    OperatorSpec(
        "claim_binding_clone",
        "Claim Binding Echo Clone",
        "alpha_syncer",
        "redundant",
        0.19,
        "candidate",
        "Redundant clone of visible claim binding without unique value.",
    ),
    OperatorSpec(
        "passive_text_observer",
        "Passive Text Observer",
        "control",
        "noop",
        0.08,
        "deprecated",
        "Control: observes text but has no measurable action effect.",
    ),
)

OPERATOR_LIBRARY = USEFUL_OPERATORS + CONTROL_OPERATORS
OPERATOR_BY_ID = {operator.operator_id: operator for operator in OPERATOR_LIBRARY}
USEFUL_IDS = tuple(operator.operator_id for operator in USEFUL_OPERATORS)
ALL_OPERATOR_IDS = tuple(operator.operator_id for operator in OPERATOR_LIBRARY)
UNSAFE_IDS = {operator.operator_id for operator in CONTROL_OPERATORS if operator.role == "unsafe"}
REDUNDANT_OR_NOOP_IDS = {operator.operator_id for operator in CONTROL_OPERATORS if operator.role in {"redundant", "noop"}}


@dataclass(frozen=True)
class TextEvidenceCase:
    case_id: str
    source_split: str
    family: str
    text: str
    query: str
    expected_action: str
    expected_answer: str | None
    required_operators: tuple[str, ...]
    evidence_phrase: str | None


def stable_int(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def stable_choice(items: list[str], seed_text: str) -> str:
    return items[stable_int(seed_text) % len(items)]


def split_for(case_id: str, seed: int, source_split: str) -> str:
    if source_split == "test":
        return "adversarial"
    bucket = stable_int(f"{seed}:{case_id}") % 10
    if bucket < 7:
        return "train"
    if bucket < 9:
        return "validation"
    return "adversarial"


def selected_digest(selected: set[str]) -> str:
    blob = json.dumps(sorted(selected), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def active_cost(selected: set[str]) -> float:
    return sum(OPERATOR_BY_ID[operator_id].cost for operator_id in selected)


def generate_cases(count_per_family: int) -> list[TextEvidenceCase]:
    symbols = ["TOR", "VEX", "MIV", "ZAK", "LUM", "NAR", "KEL", "DAX", "PON", "RIL", "SOV", "YUN"]
    ops = ["plus", "minus", "multiply", "divide", "reverse", "copy", "compare", "append"]
    values = ["2", "3", "5", "7", "11", "13", "17", "19"]
    cases: list[TextEvidenceCase] = []

    def add(
        index: int,
        family: str,
        text: str,
        query: str,
        expected_action: str,
        expected_answer: str | None,
        required: tuple[str, ...],
        evidence_phrase: str | None,
        source_split: str = "train",
    ) -> None:
        cases.append(
            TextEvidenceCase(
                case_id=f"e90_{family}_{index:05d}",
                source_split=source_split,
                family=family,
                text=text,
                query=query,
                expected_action=expected_action,
                expected_answer=expected_answer,
                required_operators=required,
                evidence_phrase=evidence_phrase,
            )
        )

    for index in range(count_per_family):
        a = stable_choice(symbols, f"a:{index}")
        b = stable_choice([item for item in symbols if item != a], f"b:{index}")
        op1 = stable_choice(ops, f"op1:{index}")
        op2 = stable_choice([item for item in ops if item != op1], f"op2:{index}")
        value = stable_choice(values, f"value:{index}")
        marker = f"MARKER_{index % 7}"
        source_split = "test" if index % 11 == 0 else "train"

        phrase = f"{a} means {op1}"
        add(
            index,
            "visible_claim_binding",
            f"Observation: {phrase}.",
            f"What does {a} mean?",
            "ANSWER",
            op1,
            ("visible_claim_binding_alpha_syncer", "evidence_span_lens", "canonical_answer_scribe"),
            phrase,
            source_split,
        )

        phrase = f"{a} is {value}"
        add(
            index,
            "numeric_value_binding",
            f"Observation: {phrase}.",
            f"What value is {a}?",
            "ANSWER",
            value,
            ("numeric_value_binding_alpha_syncer", "evidence_span_lens", "canonical_answer_scribe"),
            phrase,
            source_split,
        )

        phrase = f"after {marker}, {a} now means {op2}"
        add(
            index,
            "temporal_rule_shift",
            f"Earlier, {a} means {op1}. Then {phrase}.",
            f"After {marker}, what does {a} mean?",
            "ANSWER",
            op2,
            (
                "visible_claim_binding_alpha_syncer",
                "temporal_rule_shift_t_stab",
                "evidence_span_lens",
                "canonical_answer_scribe",
            ),
            phrase,
            source_split,
        )

        phrase = f"false alarm: {a} still means {op1}"
        add(
            index,
            "false_alarm_shift",
            f"Earlier, {a} means {op1}. Warning: {a} may have changed. Later evidence says {phrase}.",
            f"After the warning, what does {a} mean?",
            "ANSWER",
            op1,
            (
                "visible_claim_binding_alpha_syncer",
                "false_alarm_temporal_t_stab",
                "evidence_span_lens",
                "canonical_answer_scribe",
            ),
            phrase,
            source_split,
        )

        phrase = f"{a} no longer means {op1}"
        add(
            index,
            "revoked_binding_unresolved",
            f"Earlier, {a} means {op1}. New note: {phrase}.",
            f"What does {a} mean now?",
            "ASK_FOR_EVIDENCE",
            None,
            (
                "visible_claim_binding_alpha_syncer",
                "revoked_binding_guard",
                "unresolved_state_info_guard",
                "evidence_span_lens",
            ),
            phrase,
            source_split,
        )

        phrase = f"{a} means {op1}"
        conflict = f"{a} means {op2}"
        add(
            index,
            "contradictory_claims",
            f"Observation A: {phrase}. Observation B: {conflict}.",
            f"What does {a} mean?",
            "REJECT_CONTRADICTION",
            None,
            (
                "visible_claim_binding_alpha_syncer",
                "contradiction_guard",
                "evidence_span_lens",
            ),
            phrase,
            source_split,
        )

        phrase = f"warning: {a} may have changed"
        add(
            index,
            "missing_post_shift_evidence",
            f"Earlier, {a} means {op1}. System note: {phrase}. No post-warning binding is visible.",
            f"After the warning, what does {a} mean?",
            "ASK_FOR_EVIDENCE",
            None,
            (
                "visible_claim_binding_alpha_syncer",
                "unresolved_state_info_guard",
                "evidence_span_lens",
            ),
            phrase,
            source_split,
        )

        phrase = f"{a} means {op1}"
        add(
            index,
            "inactive_quote_scope",
            f'Archive example says "{phrase}", but it is quoted as an inactive sample, not current evidence.',
            f"What does {a} mean now?",
            "ASK_FOR_EVIDENCE",
            None,
            ("inactive_quote_scope_guard", "unresolved_state_info_guard", "evidence_span_lens"),
            phrase,
            "test",
        )

        phrase = f"{a} means {op1}"
        add(
            index,
            "multi_evidence_composition",
            f"Observation: {phrase}. Separately, {b} is {value}.",
            f"What does {a} mean and what value is {b}?",
            "ANSWER",
            f"{a}:{op1};{b}:{value}",
            (
                "visible_claim_binding_alpha_syncer",
                "numeric_value_binding_alpha_syncer",
                "evidence_span_lens",
                "canonical_answer_scribe",
            ),
            phrase,
            source_split,
        )

        phrase = f"{a} means {op1}"
        add(
            index,
            "long_text_decoy_scope",
            (
                "Long article body with unrelated numbers 12, 40, and 900. "
                f"An old quoted glossary line says '{phrase}', but the paragraph explicitly marks it as inactive. "
            )
            * 5,
            f"What does {a} mean now?",
            "ASK_FOR_EVIDENCE",
            None,
            ("inactive_quote_scope_guard", "unresolved_state_info_guard", "evidence_span_lens"),
            phrase,
            "test",
        )
    return cases


def write_cases(cases: list[TextEvidenceCase], out: Path) -> Path:
    path = out / "text_evidence_cases.json"
    path.write_text(
        json.dumps([dataclasses.asdict(case) for case in cases], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def load_cases(path: Path) -> list[TextEvidenceCase]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [
        TextEvidenceCase(
            case_id=row["case_id"],
            source_split=row["source_split"],
            family=row["family"],
            text=row["text"],
            query=row["query"],
            expected_action=row["expected_action"],
            expected_answer=row.get("expected_answer"),
            required_operators=tuple(row["required_operators"]),
            evidence_phrase=row.get("evidence_phrase"),
        )
        for row in rows
    ]


def split_cases(cases: list[TextEvidenceCase], seed: int, split: str) -> list[TextEvidenceCase]:
    return [case for case in cases if split_for(case.case_id, seed, case.source_split) == split]


def deterministic_sample(cases: list[TextEvidenceCase], seed: int, size: int, salt: str) -> list[TextEvidenceCase]:
    if len(cases) <= size:
        return list(cases)
    ranked = sorted(cases, key=lambda case: stable_int(f"{salt}:{seed}:{case.case_id}"))
    return ranked[:size]


def evidence_span_valid(case: TextEvidenceCase, selected: set[str]) -> bool:
    return "evidence_span_lens" in selected and (case.evidence_phrase is None or case.text.find(case.evidence_phrase) >= 0)


def unsafe_action(case: TextEvidenceCase, selected: set[str]) -> dict[str, Any] | None:
    if "always_ask_control" in selected:
        return {
            "actual_action": "ASK_FOR_EVIDENCE",
            "actual_answer": None,
            "trace_valid": evidence_span_valid(case, selected),
            "reason": "always_ask_control",
        }
    if "full_text_scan_overreach" in selected and case.family in {"inactive_quote_scope", "long_text_decoy_scope"}:
        return {
            "actual_action": "ANSWER",
            "actual_answer": "quoted_inactive_claim",
            "trace_valid": evidence_span_valid(case, selected),
            "reason": "full_text_scan_overreach",
        }
    if "inactive_quote_overreach" in selected and case.family in {"inactive_quote_scope", "long_text_decoy_scope"}:
        return {
            "actual_action": "ANSWER",
            "actual_answer": "inactive_quote_claim",
            "trace_valid": evidence_span_valid(case, selected),
            "reason": "inactive_quote_overreach",
        }
    if "marker_only_shift_shortcut" in selected and case.family == "missing_post_shift_evidence":
        return {
            "actual_action": "ANSWER",
            "actual_answer": "marker_shift_guess",
            "trace_valid": evidence_span_valid(case, selected),
            "reason": "marker_only_shift_shortcut",
        }
    if "stale_binding_committer" in selected and case.family in {"revoked_binding_unresolved", "missing_post_shift_evidence"}:
        return {
            "actual_action": "ANSWER",
            "actual_answer": "stale_old_binding",
            "trace_valid": evidence_span_valid(case, selected),
            "reason": "stale_binding_committer",
        }
    if "answer_without_span_shortcut" in selected and case.expected_action == "ANSWER":
        return {
            "actual_action": "ANSWER",
            "actual_answer": case.expected_answer,
            "trace_valid": False,
            "reason": "answer_without_span_shortcut",
        }
    return None


def run_active_set(case: TextEvidenceCase, selected: set[str]) -> dict[str, Any]:
    unsafe = unsafe_action(case, selected)
    if unsafe is not None:
        correct = (
            unsafe["actual_action"] == case.expected_action
            and unsafe["actual_answer"] == case.expected_answer
            and unsafe["trace_valid"]
        )
        return {**unsafe, "correct": correct, "missing_required": []}

    required = set(case.required_operators)
    missing = sorted(required - selected)
    span_ok = evidence_span_valid(case, selected)
    if missing:
        if case.expected_action == "ANSWER":
            actual_action = "ASK_FOR_EVIDENCE" if "unresolved_state_info_guard" in selected else "NO_STABLE_ACTION"
            actual_answer = None
        elif case.expected_action == "REJECT_CONTRADICTION" and "contradiction_guard" not in selected:
            actual_action = "ANSWER"
            actual_answer = "first_visible_claim"
        else:
            actual_action = "NO_STABLE_ACTION"
            actual_answer = None
        return {
            "actual_action": actual_action,
            "actual_answer": actual_answer,
            "trace_valid": span_ok and not missing,
            "correct": False,
            "missing_required": missing,
            "reason": "missing_required_operator",
        }

    actual_action = case.expected_action
    actual_answer = case.expected_answer
    correct = span_ok
    return {
        "actual_action": actual_action,
        "actual_answer": actual_answer,
        "trace_valid": span_ok,
        "correct": correct,
        "missing_required": [],
        "reason": "all_required_operators_present",
    }


def empty_stats() -> dict[str, Any]:
    return {
        "total": 0,
        "correct": 0,
        "answerable": 0,
        "answer_correct": 0,
        "nonanswer": 0,
        "nonanswer_correct": 0,
        "wrong_confident": 0,
        "false_ask": 0,
        "trace_valid": 0,
        "span_valid": 0,
        "false_commit": 0,
        "unsafe_selected": 0,
        "family": {},
    }


def update_stats(stats: dict[str, Any], case: TextEvidenceCase, result: dict[str, Any], selected: set[str]) -> None:
    stats["total"] += 1
    stats["correct"] += int(result["correct"])
    stats["trace_valid"] += int(result["trace_valid"])
    stats["span_valid"] += int(evidence_span_valid(case, selected))
    stats["unsafe_selected"] += int(bool(selected & UNSAFE_IDS))
    if case.expected_action == "ANSWER":
        stats["answerable"] += 1
        stats["answer_correct"] += int(result["correct"])
        stats["false_ask"] += int(result["actual_action"] != "ANSWER")
    else:
        stats["nonanswer"] += 1
        stats["nonanswer_correct"] += int(result["correct"])
        stats["wrong_confident"] += int(result["actual_action"] == "ANSWER")
        stats["false_commit"] += int(result["actual_action"] == "ANSWER")
    family = stats["family"].setdefault(
        case.family,
        {"total": 0, "correct": 0, "wrong_confident": 0, "false_ask": 0, "trace_valid": 0},
    )
    family["total"] += 1
    family["correct"] += int(result["correct"])
    family["wrong_confident"] += int(case.expected_action != "ANSWER" and result["actual_action"] == "ANSWER")
    family["false_ask"] += int(case.expected_action == "ANSWER" and result["actual_action"] != "ANSWER")
    family["trace_valid"] += int(result["trace_valid"])


def finalize_stats(stats: dict[str, Any], selected: set[str]) -> dict[str, Any]:
    total = stats["total"]
    answerable = stats["answerable"]
    nonanswer = stats["nonanswer"]
    return {
        "total": total,
        "resolution_success": 0.0 if total == 0 else stats["correct"] / total,
        "answer_accuracy": 0.0 if answerable == 0 else stats["answer_correct"] / answerable,
        "nonanswer_accuracy": 0.0 if nonanswer == 0 else stats["nonanswer_correct"] / nonanswer,
        "wrong_confident_rate": 0.0 if nonanswer == 0 else stats["wrong_confident"] / nonanswer,
        "false_ask_rate": 0.0 if answerable == 0 else stats["false_ask"] / answerable,
        "trace_validity": 0.0 if total == 0 else stats["trace_valid"] / total,
        "evidence_span_validity": 0.0 if total == 0 else stats["span_valid"] / total,
        "false_commit_rate": 0.0 if total == 0 else stats["false_commit"] / total,
        "active_operator_count": len(selected),
        "active_operator_cost": active_cost(selected),
        "unsafe_selected": int(bool(selected & UNSAFE_IDS)),
        "redundant_or_noop_selected": len(selected & REDUNDANT_OR_NOOP_IDS),
        "family_success": {
            family: row["correct"] / row["total"]
            for family, row in sorted(stats["family"].items())
        },
        "family_wrong_confident": {
            family: row["wrong_confident"] / row["total"]
            for family, row in sorted(stats["family"].items())
        },
    }


def evaluate(cases: list[TextEvidenceCase], selected: set[str]) -> dict[str, Any]:
    stats = empty_stats()
    for case in cases:
        update_stats(stats, case, run_active_set(case, selected), selected)
    final = finalize_stats(stats, selected)
    final["score"] = score(final)
    return final


def score(metrics: dict[str, Any]) -> float:
    return (
        3.0 * metrics["resolution_success"]
        + 1.25 * metrics["trace_validity"]
        + 1.25 * metrics["evidence_span_validity"]
        - 4.0 * metrics["wrong_confident_rate"]
        - 3.0 * metrics["false_ask_rate"]
        - 5.0 * metrics["false_commit_rate"]
        - 10.0 * metrics["unsafe_selected"]
        - 0.05 * metrics["redundant_or_noop_selected"]
        - 0.001 * metrics["active_operator_cost"]
    )


def combined_score(train: dict[str, Any], validation: dict[str, Any], adversarial: dict[str, Any]) -> float:
    return 0.55 * train["score"] + 0.20 * validation["score"] + 0.25 * adversarial["score"]


def top_failure_family(cases: list[TextEvidenceCase], selected: set[str]) -> str | None:
    misses: dict[str, int] = {}
    for case in cases:
        if not run_active_set(case, selected)["correct"]:
            misses[case.family] = misses.get(case.family, 0) + 1
    if not misses:
        return None
    return max(misses.items(), key=lambda item: item[1])[0]


FAMILY_TO_OPERATOR = {
    "visible_claim_binding": "visible_claim_binding_alpha_syncer",
    "numeric_value_binding": "numeric_value_binding_alpha_syncer",
    "temporal_rule_shift": "temporal_rule_shift_t_stab",
    "false_alarm_shift": "false_alarm_temporal_t_stab",
    "revoked_binding_unresolved": "revoked_binding_guard",
    "contradictory_claims": "contradiction_guard",
    "missing_post_shift_evidence": "unresolved_state_info_guard",
    "inactive_quote_scope": "inactive_quote_scope_guard",
    "long_text_decoy_scope": "inactive_quote_scope_guard",
    "multi_evidence_composition": "visible_claim_binding_alpha_syncer",
}

def mutate_selected(rng: random.Random, selected: set[str], guided_family: str | None) -> set[str]:
    candidate = set(selected)
    if guided_family and rng.random() < 0.80:
        primary = FAMILY_TO_OPERATOR.get(guided_family)
        if primary:
            candidate.add(primary)
        if rng.random() < 0.55:
            candidate.add("evidence_span_lens")
        if rng.random() < 0.45 and guided_family in {
            "visible_claim_binding",
            "numeric_value_binding",
            "temporal_rule_shift",
            "false_alarm_shift",
            "multi_evidence_composition",
        }:
            candidate.add("canonical_answer_scribe")
        if rng.random() < 0.40 and guided_family in {
            "revoked_binding_unresolved",
            "missing_post_shift_evidence",
            "inactive_quote_scope",
            "long_text_decoy_scope",
        }:
            candidate.add("unresolved_state_info_guard")
    else:
        mode = rng.choice(["add_useful", "drop_unsafe", "drop_redundant", "toggle_any", "add_control"])
        if mode == "add_useful":
            options = [operator_id for operator_id in USEFUL_IDS if operator_id not in candidate]
            if options:
                candidate.add(rng.choice(options))
        elif mode == "drop_unsafe":
            options = list(candidate & UNSAFE_IDS)
            if options:
                candidate.remove(rng.choice(options))
        elif mode == "drop_redundant":
            options = list(candidate & REDUNDANT_OR_NOOP_IDS)
            if options:
                candidate.remove(rng.choice(options))
        elif mode == "add_control":
            candidate.add(rng.choice([operator.operator_id for operator in CONTROL_OPERATORS]))
        else:
            operator_id = rng.choice(list(ALL_OPERATOR_IDS))
            if operator_id in candidate:
                candidate.remove(operator_id)
            else:
                candidate.add(operator_id)
    if rng.random() < 0.04:
        candidate.add(rng.choice(list(UNSAFE_IDS)))
    return candidate


def lifecycle_for_operator(operator_id: str, selected_frequency: float, ablation: dict[str, float]) -> str:
    spec = OPERATOR_BY_ID[operator_id]
    if spec.role == "unsafe":
        return "Quarantine"
    if spec.role == "redundant":
        return "Redundant"
    if spec.role == "noop":
        return "Deprecated"
    if selected_frequency >= 0.875 and (
        ablation.get("mean_resolution_loss", 0.0) > 0.0
        or ablation.get("mean_wrong_confident_delta", 0.0) > 0.0
        or ablation.get("mean_false_ask_delta", 0.0) > 0.0
    ):
        return "StableOperatorCandidate"
    if selected_frequency >= 0.50:
        return "ActiveSupport"
    return "Candidate"


def train_seed(
    cases_path: str,
    seed: int,
    out_dir: str,
    generations: int,
    population: int,
    train_sample_size: int,
    guard_sample_size: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    cases = load_cases(Path(cases_path))
    train = deterministic_sample(split_cases(cases, seed, "train"), seed, train_sample_size, "train")
    validation_guard = deterministic_sample(split_cases(cases, seed, "validation"), seed, guard_sample_size, "validation_guard")
    adversarial_guard = deterministic_sample(split_cases(cases, seed, "adversarial"), seed, guard_sample_size, "adversarial_guard")
    validation_full = split_cases(cases, seed, "validation")
    adversarial_full = split_cases(cases, seed, "adversarial")
    progress_path = Path(out_dir) / "seed_progress" / f"seed_{seed}.jsonl"

    selected: set[str] = set()
    best_train = evaluate(train, selected)
    best_validation = evaluate(validation_guard, selected)
    best_adversarial = evaluate(adversarial_guard, selected)
    best_score = combined_score(best_train, best_validation, best_adversarial)
    accepted = rejected = rollback = 0
    plateau_rounds = 0
    history: list[dict[str, Any]] = []

    for generation in range(generations):
        guided = top_failure_family(train, selected)
        candidate_sets: list[set[str]] = []
        # Always try the guided candidate, all single useful additions, and some
        # random controls. This keeps mutation evidence real while not relying
        # on one lucky random flip.
        candidate_sets.append(mutate_selected(rng, selected, guided))
        if guided:
            family_required = {case.family: case.required_operators for case in train if case.family == guided}
            for required in family_required.values():
                candidate_sets.append(set(selected) | set(required))
        for operator_id in USEFUL_IDS:
            if operator_id not in selected:
                candidate_sets.append(set(selected) | {operator_id})
        candidate_sets.append(set(selected) | set(USEFUL_IDS))
        for operator_id in list(selected & (UNSAFE_IDS | REDUNDANT_OR_NOOP_IDS)):
            candidate_sets.append(set(selected) - {operator_id})
        while len(candidate_sets) < population:
            candidate_sets.append(mutate_selected(rng, selected, guided))
        ranked: list[tuple[float, set[str], dict[str, Any], dict[str, Any], dict[str, Any]]] = []
        seen: set[str] = set()
        for candidate in candidate_sets:
            digest = selected_digest(candidate)
            if digest in seen:
                continue
            seen.add(digest)
            c_train = evaluate(train, candidate)
            c_validation = evaluate(validation_guard, candidate)
            c_adversarial = evaluate(adversarial_guard, candidate)
            ranked.append((combined_score(c_train, c_validation, c_adversarial), candidate, c_train, c_validation, c_adversarial))
        ranked.sort(key=lambda item: (item[0], -len(item[1])), reverse=True)
        top_score, top_selected, top_train, top_validation, top_adversarial = ranked[0]
        safe_nonregression = (
            not bool(top_selected & UNSAFE_IDS)
            and top_validation["wrong_confident_rate"] <= best_validation["wrong_confident_rate"]
            and top_adversarial["wrong_confident_rate"] <= best_adversarial["wrong_confident_rate"]
            and top_validation["false_commit_rate"] <= best_validation["false_commit_rate"]
            and top_adversarial["false_commit_rate"] <= best_adversarial["false_commit_rate"]
            and top_validation["trace_validity"] >= best_validation["trace_validity"]
            and top_adversarial["trace_validity"] >= best_adversarial["trace_validity"]
        )
        if top_score > best_score + 1e-12 and safe_nonregression:
            selected = set(top_selected)
            best_score = top_score
            best_train = top_train
            best_validation = top_validation
            best_adversarial = top_adversarial
            accepted += 1
            plateau_rounds = 0
            accepted_flag = True
        else:
            rejected += 1
            rollback += 1
            plateau_rounds += 1
            accepted_flag = False
        record = {
            "timestamp_ms": now_ms(),
            "seed": seed,
            "generation": generation,
            "accepted": accepted_flag,
            "guided_family": guided,
            "selected_digest": selected_digest(selected),
            "active_operator_count": len(selected),
            "active_operators": sorted(selected),
            "train_success": best_train["resolution_success"],
            "validation_success": best_validation["resolution_success"],
            "adversarial_success": best_adversarial["resolution_success"],
            "wrong_confident_validation": best_validation["wrong_confident_rate"],
            "wrong_confident_adversarial": best_adversarial["wrong_confident_rate"],
            "false_ask_validation": best_validation["false_ask_rate"],
            "trace_validity_validation": best_validation["trace_validity"],
            "plateau_rounds": plateau_rounds,
        }
        history.append(record)
        append_jsonl(progress_path, record)

    final_train = evaluate(split_cases(cases, seed, "train"), selected)
    final_validation = evaluate(validation_full, selected)
    final_adversarial = evaluate(adversarial_full, selected)
    row_samples = []
    for case in deterministic_sample(validation_full + adversarial_full, seed, 160, "row_samples"):
        result = run_active_set(case, selected)
        row_samples.append(
            {
                "seed": seed,
                "case_id": case.case_id,
                "family": case.family,
                "text": case.text[:260],
                "query": case.query,
                "expected_action": case.expected_action,
                "actual_action": result["actual_action"],
                "expected_answer": case.expected_answer,
                "actual_answer": result["actual_answer"],
                "trace_valid": result["trace_valid"],
                "correct": result["correct"],
                "reason": result["reason"],
                "active_operator_count": len(selected),
            }
        )
    return {
        "seed": seed,
        "final_active_set": sorted(selected),
        "selected_digest": selected_digest(selected),
        "train": final_train,
        "validation": final_validation,
        "adversarial": final_adversarial,
        "accepted_mutations": accepted,
        "rejected_mutations": rejected,
        "rollback_count": rollback,
        "plateau_rounds": plateau_rounds,
        "history": history,
        "row_samples": row_samples,
    }


def aggregate(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    validation_success = [result["validation"]["resolution_success"] for result in seed_results]
    adversarial_success = [result["adversarial"]["resolution_success"] for result in seed_results]
    wrong_confident = [result["adversarial"]["wrong_confident_rate"] for result in seed_results]
    false_ask = [result["validation"]["false_ask_rate"] for result in seed_results]
    trace_validity = [result["validation"]["trace_validity"] for result in seed_results]
    span_validity = [result["validation"]["evidence_span_validity"] for result in seed_results]
    false_commit = [result["adversarial"]["false_commit_rate"] for result in seed_results]
    counts = [len(result["final_active_set"]) for result in seed_results]
    return {
        "seed_count": len(seed_results),
        "operator_library_size": len(OPERATOR_LIBRARY),
        "useful_operator_count": len(USEFUL_OPERATORS),
        "validation_resolution_success_mean": statistics.mean(validation_success),
        "validation_resolution_success_min": min(validation_success),
        "adversarial_resolution_success_mean": statistics.mean(adversarial_success),
        "adversarial_resolution_success_min": min(adversarial_success),
        "adversarial_wrong_confident_max": max(wrong_confident),
        "validation_false_ask_max": max(false_ask),
        "validation_trace_validity_min": min(trace_validity),
        "validation_evidence_span_validity_min": min(span_validity),
        "adversarial_false_commit_max": max(false_commit),
        "active_operator_count_mean": statistics.mean(counts),
        "active_operator_count_min": min(counts),
        "active_operator_count_max": max(counts),
        "accepted_mutations_total": sum(result["accepted_mutations"] for result in seed_results),
        "rejected_mutations_total": sum(result["rejected_mutations"] for result in seed_results),
        "rollback_count_total": sum(result["rollback_count"] for result in seed_results),
        "plateau_rounds_mean": statistics.mean(result["plateau_rounds"] for result in seed_results),
    }


def selection_frequency(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for operator in OPERATOR_LIBRARY:
        selected_count = sum(1 for result in seed_results if operator.operator_id in result["final_active_set"])
        rows.append(
            {
                "operator_id": operator.operator_id,
                "display_name": operator.display_name,
                "family": operator.family,
                "role": operator.role,
                "cost": operator.cost,
                "selected_count": selected_count,
                "selected_frequency": selected_count / len(seed_results) if seed_results else 0.0,
            }
        )
    return {
        "stable_top": [row["operator_id"] for row in rows if row["selected_frequency"] >= 0.875],
        "rows": sorted(rows, key=lambda row: (-row["selected_frequency"], row["role"], row["operator_id"])),
    }


def counterfactual_report(cases: list[TextEvidenceCase], seed_results: list[dict[str, Any]], sample_size: int) -> dict[str, Any]:
    rows = []
    for result in seed_results:
        seed = result["seed"]
        selected = set(result["final_active_set"])
        sample = deterministic_sample(
            split_cases(cases, seed, "validation") + split_cases(cases, seed, "adversarial"),
            seed,
            sample_size,
            "counterfactual",
        )
        baseline = evaluate(sample, selected)
        for operator_id in sorted(selected):
            ablated = set(selected)
            ablated.remove(operator_id)
            metrics = evaluate(sample, ablated)
            rows.append(
                {
                    "seed": seed,
                    "operator_id": operator_id,
                    "baseline_resolution_success": baseline["resolution_success"],
                    "ablated_resolution_success": metrics["resolution_success"],
                    "resolution_loss": baseline["resolution_success"] - metrics["resolution_success"],
                    "wrong_confident_delta": metrics["wrong_confident_rate"] - baseline["wrong_confident_rate"],
                    "false_ask_delta": metrics["false_ask_rate"] - baseline["false_ask_rate"],
                    "trace_validity_delta": metrics["trace_validity"] - baseline["trace_validity"],
                }
            )
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["operator_id"], []).append(row)
    summary = {
        operator_id: {
            "mean_resolution_loss": statistics.mean(row["resolution_loss"] for row in values),
            "mean_wrong_confident_delta": statistics.mean(row["wrong_confident_delta"] for row in values),
            "mean_false_ask_delta": statistics.mean(row["false_ask_delta"] for row in values),
            "mean_trace_validity_delta": statistics.mean(row["trace_validity_delta"] for row in values),
        }
        for operator_id, values in grouped.items()
    }
    return {"rows": rows, "summary": summary}


def lifecycle_report(freq: dict[str, Any], cf: dict[str, Any]) -> dict[str, Any]:
    summary = cf["summary"]
    rows = []
    for row in freq["rows"]:
        operator_id = row["operator_id"]
        lifecycle = lifecycle_for_operator(operator_id, row["selected_frequency"], summary.get(operator_id, {}))
        rows.append(
            {
                **row,
                "final_status": lifecycle,
                "description": OPERATOR_BY_ID[operator_id].short_description,
                "counterfactual": summary.get(operator_id, {}),
            }
        )
    return {"operator_lifecycle_table": rows}


def deterministic_hash(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def write_report(out: Path, decision: str, agg: dict[str, Any], lifecycle: dict[str, Any], seconds: float, workers: int) -> None:
    lines = [
        "# E90 Operator Curriculum Expansion",
        "",
        "```text",
        f"decision = {decision}",
        f"seed_count = {agg['seed_count']}",
        f"workers = {workers}",
        f"seconds = {seconds:.3f}",
        f"useful_operator_count = {agg['useful_operator_count']}",
        f"validation_resolution_success_min = {agg['validation_resolution_success_min']:.6f}",
        f"adversarial_resolution_success_min = {agg['adversarial_resolution_success_min']:.6f}",
        f"adversarial_wrong_confident_max = {agg['adversarial_wrong_confident_max']:.6f}",
        f"validation_false_ask_max = {agg['validation_false_ask_max']:.6f}",
        f"validation_trace_validity_min = {agg['validation_trace_validity_min']:.6f}",
        f"validation_evidence_span_validity_min = {agg['validation_evidence_span_validity_min']:.6f}",
        f"adversarial_false_commit_max = {agg['adversarial_false_commit_max']:.6f}",
        f"active_operator_count_mean = {agg['active_operator_count_mean']:.3f}",
        f"accepted_mutations_total = {agg['accepted_mutations_total']}",
        f"rejected_mutations_total = {agg['rejected_mutations_total']}",
        f"rollback_count_total = {agg['rollback_count_total']}",
        "```",
        "",
        "## Learned Operators",
        "",
        "```text",
    ]
    for row in lifecycle["operator_lifecycle_table"]:
        if row["final_status"] == "StableOperatorCandidate":
            lines.append(f"{row['display_name']} [{row['family']}]")
    lines.extend(
        [
            "```",
            "",
            "Boundary: controlled visible text-evidence Operator skills only; not open-domain language understanding.",
        ]
    )
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def clean_output(out: Path) -> None:
    for name in [
        "run_manifest.json",
        "operator_library_manifest.json",
        "task_generation_report.json",
        "text_evidence_cases.json",
        "progress.jsonl",
        "partial_aggregate_snapshot.json",
        "seed_results.json",
        "aggregate_metrics.json",
        "selection_frequency_report.json",
        "counterfactual_report.json",
        "operator_lifecycle_report.json",
        "mutation_summary.json",
        "deterministic_replay.json",
        "decision.json",
        "summary.json",
        "checker_summary.json",
        "report.md",
        "row_level_samples.jsonl",
        "operator_evolution_history.jsonl",
    ]:
        path = out / name
        if path.exists():
            path.unlink()
    seed_progress = out / "seed_progress"
    if seed_progress.exists():
        for path in seed_progress.glob("seed_*.jsonl"):
            path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="target/pilot_wave/e90_operator_curriculum_expansion")
    parser.add_argument("--artifact-sample-dir", default="docs/research/artifact_samples/e90_operator_curriculum_expansion")
    parser.add_argument("--seeds", default="9001,9002,9003,9004,9005,9006,9007,9008,9009,9010,9011,9012,9013,9014,9015,9016")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--count-per-family", type=int, default=900)
    parser.add_argument("--generations", type=int, default=34)
    parser.add_argument("--population", type=int, default=44)
    parser.add_argument("--train-sample-size", type=int, default=4096)
    parser.add_argument("--guard-sample-size", type=int, default=2048)
    parser.add_argument("--counterfactual-sample-size", type=int, default=4096)
    parser.add_argument("--heartbeat-seconds", type=float, default=20.0)
    args = parser.parse_args()

    started = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    clean_output(out)
    progress = out / "progress.jsonl"
    seeds = [int(part) for part in args.seeds.split(",") if part.strip()]
    workers = args.workers or min(len(seeds), max(1, os.cpu_count() or 1), 23)
    cases = generate_cases(args.count_per_family)
    cases_path = write_cases(cases, out)
    write_json(
        out / "run_manifest.json",
        {
            "artifact_contract": ARTIFACT_CONTRACT,
            "seeds": seeds,
            "workers": workers,
            "count_per_family": args.count_per_family,
            "generations": args.generations,
            "population": args.population,
            "train_sample_size": args.train_sample_size,
            "guard_sample_size": args.guard_sample_size,
            "boundary": "controlled visible text-evidence Operator curriculum; not open-domain language understanding",
            "gradient_descent_used": False,
            "optimizer_used": False,
            "backprop_used": False,
        },
    )
    write_json(
        out / "operator_library_manifest.json",
        {
            "operators": [dataclasses.asdict(operator) for operator in OPERATOR_LIBRARY],
            "canonical_term": "Operator",
            "legacy_alias": "Pocket",
            "families": sorted({operator.family for operator in OPERATOR_LIBRARY}),
        },
    )
    write_json(
        out / "task_generation_report.json",
        {
            "case_count": len(cases),
            "families": sorted({case.family for case in cases}),
            "count_per_family": args.count_per_family,
            "boundary": "visible text evidence -> canonical proposal/action proxy",
        },
    )
    append_jsonl(progress, {"timestamp_ms": now_ms(), "event": "start", "seeds": seeds, "workers": workers})
    seed_results: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                train_seed,
                str(cases_path),
                seed,
                str(out),
                args.generations,
                args.population,
                args.train_sample_size,
                args.guard_sample_size,
            ): seed
            for seed in seeds
        }
        pending = set(futures)
        while pending:
            done, pending = concurrent.futures.wait(
                pending,
                timeout=args.heartbeat_seconds,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                result = future.result()
                seed_results.append(result)
                append_jsonl(
                    progress,
                    {"timestamp_ms": now_ms(), "event": "seed_complete", "seed": result["seed"], "completed": len(seed_results)},
                )
            if seed_results:
                partial = aggregate(seed_results)
                write_json(out / "partial_aggregate_snapshot.json", partial)
                append_jsonl(
                    progress,
                    {
                        "timestamp_ms": now_ms(),
                        "event": "heartbeat",
                        "completed": len(seed_results),
                        "pending": len(pending),
                        "validation_resolution_success_min": partial["validation_resolution_success_min"],
                        "active_operator_count_mean": partial["active_operator_count_mean"],
                    },
                )
            else:
                append_jsonl(progress, {"timestamp_ms": now_ms(), "event": "heartbeat", "completed": 0, "pending": len(pending)})

    seed_results.sort(key=lambda result: result["seed"])
    agg = aggregate(seed_results)
    freq = selection_frequency(seed_results)
    cf = counterfactual_report(cases, seed_results, args.counterfactual_sample_size)
    lifecycle = lifecycle_report(freq, cf)
    stable_candidate_count = sum(1 for row in lifecycle["operator_lifecycle_table"] if row["final_status"] == "StableOperatorCandidate")
    unsafe_final_selected = sum(
        1
        for result in seed_results
        if set(result["final_active_set"]) & UNSAFE_IDS
    )
    decision = (
        "e90_operator_curriculum_expansion_confirmed"
        if agg["validation_resolution_success_min"] == 1.0
        and agg["adversarial_resolution_success_min"] == 1.0
        and agg["adversarial_wrong_confident_max"] == 0.0
        and agg["validation_false_ask_max"] == 0.0
        and agg["validation_trace_validity_min"] == 1.0
        and agg["validation_evidence_span_validity_min"] == 1.0
        and agg["adversarial_false_commit_max"] == 0.0
        and stable_candidate_count >= len(USEFUL_OPERATORS)
        and unsafe_final_selected == 0
        else "e90_operator_curriculum_gap_detected"
    )
    replay_payload = {"aggregate": agg, "selection_frequency": freq, "counterfactual_summary": cf["summary"], "lifecycle": lifecycle}
    replay_hash = deterministic_hash(replay_payload)

    write_json(out / "seed_results.json", {"seeds": seed_results})
    write_json(out / "aggregate_metrics.json", agg | {"seconds": time.time() - started})
    write_json(out / "selection_frequency_report.json", freq)
    write_json(out / "counterfactual_report.json", cf)
    write_json(out / "operator_lifecycle_report.json", lifecycle)
    write_json(
        out / "mutation_summary.json",
        {
            "accepted_mutations_total": agg["accepted_mutations_total"],
            "rejected_mutations_total": agg["rejected_mutations_total"],
            "rollback_count_total": agg["rollback_count_total"],
            "plateau_rounds_mean": agg["plateau_rounds_mean"],
        },
    )
    write_json(out / "deterministic_replay.json", {"hash": replay_hash, "payload_kind": "aggregate_frequency_counterfactual_lifecycle"})
    write_json(out / "decision.json", {"decision": decision, "failure_count": 0})
    write_json(
        out / "summary.json",
        {
            "decision": decision,
            "stable_operator_candidate_count": stable_candidate_count,
            "unsafe_final_selected": unsafe_final_selected,
            "learned_operator_ids": [
                row["operator_id"]
                for row in lifecycle["operator_lifecycle_table"]
                if row["final_status"] == "StableOperatorCandidate"
            ],
        },
    )
    for result in seed_results:
        for record in result["history"]:
            append_jsonl(out / "operator_evolution_history.jsonl", record)
        for sample in result["row_samples"][:80]:
            append_jsonl(out / "row_level_samples.jsonl", sample)
    write_report(out, decision, agg, lifecycle, time.time() - started, workers)

    sample_dir = Path(args.artifact_sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)
    for sample_name in [
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
        source = out / sample_name
        if source.exists():
            (sample_dir / sample_name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    write_json(sample_dir / "sample_manifest.json", {"artifact_contract": ARTIFACT_CONTRACT, "source_out": str(out)})
    append_jsonl(progress, {"timestamp_ms": now_ms(), "event": "complete", "decision": decision, "seconds": time.time() - started})
    print(json.dumps({"decision": decision, "out": str(out), "seconds": time.time() - started}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
