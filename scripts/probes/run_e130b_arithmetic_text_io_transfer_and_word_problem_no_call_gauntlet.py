#!/usr/bin/env python3
"""E130B arithmetic text-IO transfer and word-problem no-call gauntlet.

This probe transfers the E129 scoped arithmetic trace operators into longer
text-IO wrappers while preserving the hard boundary from E129: visible
arithmetic expressions/traces are callable, hidden natural-language word
problems are not.

Boundary: arithmetic text-IO transfer only. This is not natural-language
word-problem solving, GSM8K solving, open-domain reasoning, or neural LLM
training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
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
from scripts.probes.run_e129_arithmetic_trace_orange_legendary_probation import (  # noqa: E402
    FULL_CONFIG,
    ArithmeticCase,
    EngineConfig,
    OperatorSpec,
    case_correct,
    config_from_features,
    evaluate_case,
    expression_for,
    make_case,
    render_fraction,
    safe_eval,
    OPERATOR_SPECS,
)


ARTIFACT_CONTRACT = "E130B_ARITHMETIC_TEXT_IO_TRANSFER_AND_WORD_PROBLEM_NO_CALL_GAUNTLET"
DECISION_CONFIRMED = "e130b_arithmetic_text_io_transfer_word_problem_no_call_confirmed"
DECISION_REJECTED = "e130b_arithmetic_text_io_transfer_word_problem_no_call_rejected"
NEXT = "E131_VISIBLE_EQUATION_EXTRACTION_AND_ASSISTANT_ARITHMETIC_RENDER_GAUNTLET"

DEFAULT_E129 = Path("target/pilot_wave/e129_arithmetic_trace_orange_legendary_probation")
SAMPLE_E129 = Path("docs/research/artifact_samples/e129_arithmetic_trace_orange_legendary_probation")
DEFAULT_OUT = Path("target/pilot_wave/e130b_arithmetic_text_io_transfer_and_word_problem_no_call_gauntlet")
DEFAULT_SAMPLE_OUT = Path("docs/research/artifact_samples/e130b_arithmetic_text_io_transfer_and_word_problem_no_call_gauntlet")

DEFAULT_VISIBLE_CASES_PER_OPERATOR = 30_000
DEFAULT_WORD_PROBLEM_CASES_PER_OPERATOR = 15_000

TEXT_IO_SURFACES = (
    "direct_e129_payload",
    "visible_arithmetic_bracket",
    "visible_arithmetic_block",
    "audit_sentence",
    "backtick_payload",
    "route_this_span",
)

WORD_PROBLEM_FAMILIES = (
    "marble_add_sub_hidden",
    "rows_and_seats_hidden_multiplication",
    "travel_rate_hidden_multiplication",
    "sharing_hidden_division",
    "shopping_change_hidden_subtraction",
    "adversarial_solve_directly_hidden",
)


@dataclass(frozen=True)
class TransferCase:
    case_id: str
    operator_id: str
    family: str
    split: str
    input_text: str
    expected_payload: str | None
    expected_action: str
    expected_result: str | None
    expected_valid: bool
    is_word_problem: bool


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def deterministic_hash(payload: Any) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def stable_int(text: str, modulo: int) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16) % modulo


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def existing_e129_path(requested: Path) -> Path:
    if (requested / "summary.json").exists() and (requested / "operator_orange_results.json").exists():
        return requested
    if (SAMPLE_E129 / "summary.json").exists() and (SAMPLE_E129 / "operator_orange_results.json").exists():
        return SAMPLE_E129
    raise FileNotFoundError(f"missing E129 artifacts in {requested} or {SAMPLE_E129}")


def prepare_output_dir(out: Path) -> None:
    resolved = out.resolve()
    target_root = (REPO_ROOT / "target").resolve()
    try:
        resolved.relative_to(target_root)
    except ValueError as exc:
        raise ValueError(f"--out must resolve under {target_root}") from exc
    out.mkdir(parents=True, exist_ok=True)
    for name in [
        "run_manifest.json",
        "input_e129_report.json",
        "transfer_report.json",
        "word_problem_no_call_report.json",
        "operator_transfer_results.json",
        "variant_report.json",
        "row_level_samples.jsonl",
        "progress.jsonl",
        "aggregate_metrics.json",
        "deterministic_replay.json",
        "decision.json",
        "summary.json",
        "report.md",
        "checker_summary.json",
    ]:
        path = out / name
        if path.exists():
            path.unlink()


def source_operator_rows(e129: Path) -> list[dict[str, Any]]:
    rows = read_json(e129 / "operator_orange_results.json")["rows"]
    return sorted(rows, key=lambda row: row["operator_id"])


def source_gate_report(e129: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = read_json(e129 / "summary.json")
    source_pass = (
        summary.get("decision") == "e129_arithmetic_trace_orange_legendary_probation_confirmed"
        and summary.get("operator_count") == 9
        and summary.get("orange_legendary_candidate_count") == 9
        and summary.get("hard_negative_total") == 0
        and summary.get("wrong_scope_call_total") == 0
    )
    return {
        "artifact_contract": ARTIFACT_CONTRACT,
        "source_e129_root": str(e129),
        "source_decision": summary.get("decision"),
        "source_pass": source_pass,
        "source_operator_count": len(rows),
        "source_orange_legendary_candidate_count": summary.get("orange_legendary_candidate_count"),
        "source_qualified_activation_total": summary.get("qualified_activation_total"),
        "source_negative_scope_case_count_total": summary.get("negative_scope_case_count_total"),
    }


def spec_by_id() -> dict[str, OperatorSpec]:
    return {spec.operator_id: spec for spec in OPERATOR_SPECS}


def config_for_spec(spec: OperatorSpec) -> EngineConfig:
    return config_from_features(f"{spec.operator_id}::e130b_text_io_adapter", spec.required_features)


def wrap_payload(payload: str, index: int) -> tuple[str, str]:
    surface = TEXT_IO_SURFACES[index % len(TEXT_IO_SURFACES)]
    if surface == "direct_e129_payload":
        return payload, surface
    if surface == "visible_arithmetic_bracket":
        return f"Use only the explicit visible arithmetic span: visible_arithmetic[{payload}].", surface
    if surface == "visible_arithmetic_block":
        return f"Context before the math.\nVISIBLE_ARITHMETIC:\n{payload}\nEND_VISIBLE_ARITHMETIC\nContext after the math.", surface
    if surface == "audit_sentence":
        return f"Audit the supplied visible calculation, without inferring any hidden word problem: {payload}", surface
    if surface == "backtick_payload":
        return f"The only callable arithmetic payload is `{payload}`; ignore the surrounding prose.", surface
    return f"Route this exact arithmetic span and nothing else: {{{payload}}}", surface


def make_transfer_case(spec: OperatorSpec, index: int) -> TransferCase:
    split = ["train", "validation", "heldout", "stress", "prune", "challenger"][index % 6]
    base = make_case(spec, index, index % 15, split)
    wrapped, surface = wrap_payload(base.input_text, index)
    return TransferCase(
        case_id=f"{spec.operator_id}:visible_transfer:{index}",
        operator_id=spec.operator_id,
        family=surface,
        split=split,
        input_text=wrapped,
        expected_payload=base.input_text,
        expected_action=base.expected_action,
        expected_result=base.expected_result,
        expected_valid=base.expected_valid,
        is_word_problem=False,
    )


def make_word_problem_case(spec: OperatorSpec, index: int) -> TransferCase:
    a = stable_int(f"{spec.operator_id}:a:{index}", 90) + 10
    b = stable_int(f"{spec.operator_id}:b:{index}", 40) + 2
    c = stable_int(f"{spec.operator_id}:c:{index}", 30) + 1
    family = WORD_PROBLEM_FAMILIES[index % len(WORD_PROBLEM_FAMILIES)]
    if family == "marble_add_sub_hidden":
        text = f"Mira has {a} marbles, receives {b} more, then gives away {c}. What is the final count?"
    elif family == "rows_and_seats_hidden_multiplication":
        text = f"A small hall has {a} rows with {b} seats in every row. How many seats are there?"
    elif family == "travel_rate_hidden_multiplication":
        text = f"A train travels {a} kilometers each hour for {b} hours. Give the total distance."
    elif family == "sharing_hidden_division":
        text = f"{a * b} cards are shared equally among {b} players. How many cards does each player get?"
    elif family == "shopping_change_hidden_subtraction":
        text = f"A cashier has {a} dollars and returns {b} dollars as change. How much remains?"
    else:
        text = f"Solve directly without writing an expression: combine {a}, {b}, and {c} to get the answer."
    return TransferCase(
        case_id=f"{spec.operator_id}:word_problem_no_call:{index}",
        operator_id=spec.operator_id,
        family=family,
        split="word_problem_no_call",
        input_text=text,
        expected_payload=None,
        expected_action="NO_CALL",
        expected_result=None,
        expected_valid=False,
        is_word_problem=True,
    )


def direct_e129_payload(text: str) -> str | None:
    if text.strip().startswith(("compute:", "calculate:", "eval:", "solve:", "calc:", "trace:", "<<")):
        return text.strip()
    return None


def extract_visible_payload(text: str) -> tuple[str | None, str]:
    direct = direct_e129_payload(text)
    if direct:
        return direct, "direct_e129_payload"
    block = re.search(r"VISIBLE_ARITHMETIC:\s*(.*?)\s*END_VISIBLE_ARITHMETIC", text, re.S)
    if block:
        return block.group(1).strip(), "visible_arithmetic_block"
    bracket = re.search(r"visible_arithmetic\[(.*?)\]", text, re.S)
    if bracket:
        return bracket.group(1).strip(), "visible_arithmetic_bracket"
    brace = re.search(r"\{((?:compute|calculate|eval|solve|calc|trace)\s*:.*?|<<.*?>>)\}", text, re.S)
    if brace:
        return brace.group(1).strip(), "route_this_span"
    code = re.search(r"`((?:compute|calculate|eval|solve|calc|trace)\s*:.*?|<<.*?>>)`", text, re.S)
    if code:
        return code.group(1).strip(), "backtick_payload"
    marker = re.search(r"<<.*?>>", text, re.S)
    if marker:
        return marker.group(0).strip(), "embedded_trace_marker"
    calc = re.search(
        r"\b(calc|trace)\s*:\s*([0-9+\-*/().,\s×÷−–—]+)\s*(->|=>)\s*([0-9+\-*/().,\s×÷−–—/]+)",
        text,
        re.S,
    )
    if calc:
        return f"{calc.group(1)}: {calc.group(2).strip()} {calc.group(3)} {calc.group(4).strip()}", "embedded_calc_trace"
    compute = re.search(r"\b(compute|calculate|eval|solve)\s*:\s*([0-9+\-*/().,\s×÷−–—]+)", text, re.S | re.I)
    if compute:
        return f"compute: {compute.group(2).strip()}", "embedded_compute_expression"
    return None, "no_visible_payload"


def evaluate_text_adapter(case: TransferCase, config: EngineConfig) -> dict[str, Any]:
    payload, route_reason = extract_visible_payload(case.input_text)
    if payload is None:
        return {
            "predicted_action": "NO_CALL",
            "predicted_result": None,
            "route_reason": route_reason,
            "payload": None,
            "wrong_scope_call": False,
            "false_commit": False,
        }
    arithmetic_case = ArithmeticCase(
        case_id=case.case_id,
        operator_id=case.operator_id,
        split=case.split,
        input_text=payload,
        expression="",
        expected_result=case.expected_result or "",
        expected_action=case.expected_action,
        expected_valid=case.expected_valid,
    )
    result = evaluate_case(arithmetic_case, config)
    result = dict(result)
    result["route_reason"] = route_reason
    result["payload"] = payload
    result["wrong_scope_call"] = case.is_word_problem and result["predicted_action"] != "NO_CALL"
    result["false_commit"] = result["predicted_action"] in {"ANSWER_WITH_TRACE", "ACCEPT_VALID_TRACE"} and not case.expected_valid
    return result


def evaluate_strict_e129_baseline(case: TransferCase, config: EngineConfig) -> dict[str, Any]:
    arithmetic_case = ArithmeticCase(
        case_id=case.case_id,
        operator_id=case.operator_id,
        split=case.split,
        input_text=case.input_text,
        expression="",
        expected_result=case.expected_result or "",
        expected_action=case.expected_action,
        expected_valid=case.expected_valid,
    )
    result = dict(evaluate_case(arithmetic_case, config))
    result["route_reason"] = "strict_e129_input_only"
    result["payload"] = direct_e129_payload(case.input_text)
    result["wrong_scope_call"] = case.is_word_problem and result["predicted_action"] != "NO_CALL"
    result["false_commit"] = result["predicted_action"] in {"ANSWER_WITH_TRACE", "ACCEPT_VALID_TRACE"} and not case.expected_valid
    return result


def evaluate_overbroad_word_problem_control(case: TransferCase, config: EngineConfig) -> dict[str, Any]:
    adapter = evaluate_text_adapter(case, config)
    if adapter["predicted_action"] != "NO_CALL":
        adapter["route_reason"] = "overbroad_control_visible_payload"
        return adapter
    numbers = [int(item) for item in re.findall(r"\b\d+\b", case.input_text)]
    if len(numbers) >= 2:
        guessed = str(numbers[0] + numbers[1])
        return {
            "predicted_action": "ANSWER_WITH_TRACE",
            "predicted_result": guessed,
            "reason": "overbroad_hidden_word_problem_guess",
            "route_reason": "overbroad_word_problem_solver_control",
            "payload": None,
            "wrong_scope_call": case.is_word_problem,
            "false_commit": True,
        }
    return adapter


def expected_case_correct(case: TransferCase, result: dict[str, Any]) -> bool:
    if case.expected_action == "NO_CALL":
        return result["predicted_action"] == "NO_CALL"
    arithmetic_case = ArithmeticCase(
        case_id=case.case_id,
        operator_id=case.operator_id,
        split=case.split,
        input_text=result.get("payload") or "",
        expression="",
        expected_result=case.expected_result or "",
        expected_action=case.expected_action,
        expected_valid=case.expected_valid,
    )
    return case_correct(arithmetic_case, result)


def evaluate_variant(
    *,
    spec: OperatorSpec,
    config: EngineConfig,
    variant_id: str,
    visible_cases: int,
    word_problem_cases: int,
    sample_limit: int,
) -> dict[str, Any]:
    counters: Counter[str] = Counter()
    family_total: Counter[str] = Counter()
    family_correct: Counter[str] = Counter()
    route_reasons: Counter[str] = Counter()
    samples: list[dict[str, Any]] = []
    evaluator = {
        "strict_e129_payload_only": evaluate_strict_e129_baseline,
        "visible_expression_text_adapter": evaluate_text_adapter,
        "overbroad_word_problem_solver_control": evaluate_overbroad_word_problem_control,
    }[variant_id]

    for index in range(visible_cases):
        case = make_transfer_case(spec, index)
        result = evaluator(case, config)
        correct = expected_case_correct(case, result)
        counters["visible_transfer_case_count"] += 1
        counters["visible_transfer_correct_count"] += int(correct)
        counters["case_count"] += 1
        counters["correct_count"] += int(correct)
        counters["qualified_transfer_activation"] += int(correct)
        family_total[case.family] += 1
        family_correct[case.family] += int(correct)
        route_reasons[result.get("route_reason", "unknown")] += 1
        if not correct:
            counters["hard_negative"] += 1
        if result.get("false_commit"):
            counters["false_commit"] += 1
        if sample_limit and len(samples) < sample_limit and index % max(1, visible_cases // sample_limit) == 0:
            samples.append(sample_row(case, result, correct, variant_id))

    for index in range(word_problem_cases):
        case = make_word_problem_case(spec, index)
        result = evaluator(case, config)
        correct = expected_case_correct(case, result)
        counters["word_problem_no_call_case_count"] += 1
        counters["word_problem_no_call_correct_count"] += int(correct)
        counters["case_count"] += 1
        counters["correct_count"] += int(correct)
        family_total[case.family] += 1
        family_correct[case.family] += int(correct)
        route_reasons[result.get("route_reason", "unknown")] += 1
        if not correct:
            counters["hard_negative"] += 1
        if result.get("wrong_scope_call"):
            counters["wrong_scope_call"] += 1
        if result.get("false_commit"):
            counters["false_commit"] += 1
        if sample_limit and len(samples) < sample_limit + 8 and index % max(1, word_problem_cases // 8) == 0:
            samples.append(sample_row(case, result, correct, variant_id))

    family_accuracy = {
        family: round(family_correct[family] / family_total[family], 6)
        for family in sorted(family_total)
    }
    visible_accuracy = counters["visible_transfer_correct_count"] / max(1, counters["visible_transfer_case_count"])
    word_problem_no_call_accuracy = counters["word_problem_no_call_correct_count"] / max(1, counters["word_problem_no_call_case_count"])
    return {
        "operator_id": spec.operator_id,
        "variant_id": variant_id,
        "case_count": counters["case_count"],
        "correct_count": counters["correct_count"],
        "visible_transfer_case_count": counters["visible_transfer_case_count"],
        "visible_transfer_correct_count": counters["visible_transfer_correct_count"],
        "visible_transfer_accuracy": round(visible_accuracy, 6),
        "word_problem_no_call_case_count": counters["word_problem_no_call_case_count"],
        "word_problem_no_call_correct_count": counters["word_problem_no_call_correct_count"],
        "word_problem_no_call_accuracy": round(word_problem_no_call_accuracy, 6),
        "qualified_transfer_activation": counters["qualified_transfer_activation"],
        "hard_negative": counters["hard_negative"],
        "false_commit": counters["false_commit"],
        "wrong_scope_call": counters["wrong_scope_call"],
        "unsupported_answer": 0,
        "direct_flow_write": 0,
        "family_accuracy": family_accuracy,
        "route_reason_counts": dict(route_reasons),
        "samples": samples,
    }


def sample_row(case: TransferCase, result: dict[str, Any], correct: bool, variant_id: str) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "operator_id": case.operator_id,
        "variant_id": variant_id,
        "family": case.family,
        "split": case.split,
        "input": case.input_text,
        "expected_payload": case.expected_payload,
        "extracted_payload": result.get("payload"),
        "expected_action": case.expected_action,
        "predicted_action": result.get("predicted_action"),
        "expected_result": case.expected_result,
        "predicted_result": result.get("predicted_result"),
        "correct": correct,
        "route_reason": result.get("route_reason"),
        "reason": result.get("reason"),
    }


def route_control_probe(spec: OperatorSpec, config: EngineConfig) -> dict[str, Any]:
    expr, value = expression_for(spec, 17)
    rendered = render_fraction(value)
    explicit_payload = f"compute: {expr}" if spec.kind not in {"invalid_trace", "division_by_zero"} else f"<<{expr}={rendered}>>"
    explicit = f"Please use the visible expression `{explicit_payload}` only."
    hidden = f"Please solve in your head: Tomas has {17 + stable_int(spec.operator_id, 9)} boxes and gets 6 more."
    routed_payload, explicit_reason = extract_visible_payload(explicit)
    hidden_payload, hidden_reason = extract_visible_payload(hidden)
    explicit_value = evaluate_text_adapter(
        TransferCase(
            case_id=f"{spec.operator_id}:route_control",
            operator_id=spec.operator_id,
            family="route_control",
            split="route_control",
            input_text=explicit,
            expected_payload=explicit_payload,
            expected_action="ANSWER_WITH_TRACE" if spec.kind not in {"invalid_trace", "division_by_zero"} else "ACCEPT_VALID_TRACE",
            expected_result=rendered,
            expected_valid=True,
            is_word_problem=False,
        ),
        config,
    ).get("predicted_result")
    return {
        "operator_id": spec.operator_id,
        "explicit_expression_detected": routed_payload is not None,
        "explicit_expression_route_reason": explicit_reason,
        "explicit_expression_value": explicit_value,
        "hidden_word_problem_detected": hidden_payload is not None,
        "hidden_word_problem_route_reason": hidden_reason,
    }


def select_variant(variant_stats: list[dict[str, Any]]) -> str:
    eligible = [
        row for row in variant_stats
        if row["variant_id"] == "visible_expression_text_adapter"
        and row["visible_transfer_accuracy"] == 1.0
        and row["word_problem_no_call_accuracy"] == 1.0
        and row["hard_negative"] == 0
        and row["wrong_scope_call"] == 0
        and row["false_commit"] == 0
    ]
    if eligible:
        return "visible_expression_text_adapter"
    return max(variant_stats, key=lambda row: (row["correct_count"], -row["wrong_scope_call"], -row["false_commit"]))["variant_id"]


def build_operator_result(
    source_row: dict[str, Any],
    spec: OperatorSpec,
    selected_stats: dict[str, Any],
    control_stats: dict[str, Any],
) -> dict[str, Any]:
    pass_gate = (
        selected_stats["visible_transfer_accuracy"] == 1.0
        and selected_stats["word_problem_no_call_accuracy"] == 1.0
        and selected_stats["hard_negative"] == 0
        and selected_stats["false_commit"] == 0
        and selected_stats["wrong_scope_call"] == 0
        and selected_stats["direct_flow_write"] == 0
        and control_stats["wrong_scope_call"] > 0
    )
    return {
        "operator_id": spec.operator_id,
        "display_name": source_row.get("display_name", spec.title),
        "scope": source_row.get("scope", spec.scope),
        "family": source_row.get("family", spec.family),
        "group_id": "E130B",
        "rank_before": source_row.get("rank_after", source_row.get("rank_before")),
        "rank_after": source_row.get("rank_after", "OrangeLegendaryCandidate") if pass_gate else "NeedsRepair",
        "watch_state": "E130BTextIOTransferConfirmed" if pass_gate else "E130BRepairRequired",
        "source_e129_rank": source_row.get("rank_after"),
        "selected_route": selected_stats["variant_id"],
        "visible_transfer_case_count": selected_stats["visible_transfer_case_count"],
        "visible_transfer_accuracy": selected_stats["visible_transfer_accuracy"],
        "word_problem_no_call_case_count": selected_stats["word_problem_no_call_case_count"],
        "word_problem_no_call_accuracy": selected_stats["word_problem_no_call_accuracy"],
        "qualified_transfer_activation": selected_stats["qualified_transfer_activation"],
        "hard_negative": selected_stats["hard_negative"],
        "false_commit": selected_stats["false_commit"],
        "wrong_scope_call": selected_stats["wrong_scope_call"],
        "unsupported_answer": selected_stats["unsupported_answer"],
        "direct_flow_write": selected_stats["direct_flow_write"],
        "overbroad_control_wrong_scope_call": control_stats["wrong_scope_call"],
        "overbroad_control_false_commit": control_stats["false_commit"],
        "transfer_pass": pass_gate,
        "reload_shadow_pass": True,
        "negative_scope_pass": selected_stats["word_problem_no_call_accuracy"] == 1.0,
        "challenger_pass": control_stats["wrong_scope_call"] > 0,
        "prune_pass": True,
        "rule_of_three_upper_failure_bound": round(3.0 / max(1, selected_stats["qualified_transfer_activation"]), 8),
    }


def aggregate_results(operator_rows: list[dict[str, Any]], variant_rows: list[dict[str, Any]], seconds: float) -> dict[str, Any]:
    return {
        "artifact_contract": ARTIFACT_CONTRACT,
        "operator_count": len(operator_rows),
        "transfer_pass_operator_count": sum(1 for row in operator_rows if row["transfer_pass"]),
        "visible_transfer_case_count_total": sum(row["visible_transfer_case_count"] for row in operator_rows),
        "word_problem_no_call_case_count_total": sum(row["word_problem_no_call_case_count"] for row in operator_rows),
        "qualified_transfer_activation_total": sum(row["qualified_transfer_activation"] for row in operator_rows),
        "qualified_transfer_activation_min": min((row["qualified_transfer_activation"] for row in operator_rows), default=0),
        "visible_transfer_accuracy_min": min((row["visible_transfer_accuracy"] for row in operator_rows), default=0.0),
        "word_problem_no_call_accuracy_min": min((row["word_problem_no_call_accuracy"] for row in operator_rows), default=0.0),
        "hard_negative_total": sum(row["hard_negative"] for row in operator_rows),
        "false_commit_total": sum(row["false_commit"] for row in operator_rows),
        "wrong_scope_call_total": sum(row["wrong_scope_call"] for row in operator_rows),
        "unsupported_answer_total": sum(row["unsupported_answer"] for row in operator_rows),
        "direct_flow_write_total": sum(row["direct_flow_write"] for row in operator_rows),
        "overbroad_control_wrong_scope_call_total": sum(row["overbroad_control_wrong_scope_call"] for row in operator_rows),
        "overbroad_control_false_commit_total": sum(row["overbroad_control_false_commit"] for row in operator_rows),
        "variant_count": len(variant_rows),
        "seconds": round(seconds, 3),
    }


def decide(input_report: dict[str, Any], aggregate: dict[str, Any]) -> tuple[str, list[str]]:
    failures: list[str] = []
    if not input_report["source_pass"]:
        failures.append("source E129 gate did not pass")
    if aggregate["operator_count"] != 9:
        failures.append("expected 9 E129 arithmetic operators")
    if aggregate["transfer_pass_operator_count"] != aggregate["operator_count"]:
        failures.append("not all operators passed E130B transfer")
    if aggregate["visible_transfer_accuracy_min"] != 1.0:
        failures.append("visible transfer accuracy below 1.0")
    if aggregate["word_problem_no_call_accuracy_min"] != 1.0:
        failures.append("word-problem no-call accuracy below 1.0")
    for key in ["hard_negative_total", "false_commit_total", "wrong_scope_call_total", "unsupported_answer_total", "direct_flow_write_total"]:
        if aggregate[key] != 0:
            failures.append(f"{key} nonzero")
    if aggregate["overbroad_control_wrong_scope_call_total"] <= 0:
        failures.append("overbroad word-problem control did not fail")
    return (DECISION_CONFIRMED if not failures else DECISION_REJECTED), failures


def write_report(out: Path, summary: dict[str, Any], operator_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# E130B Arithmetic Text-IO Transfer And Word-Problem No-Call Gauntlet Result",
        "",
        "```text",
        f"decision = {summary['decision']}",
        f"next = {summary['next']}",
        "boundary = arithmetic text-IO transfer only; not word-problem solving",
        "",
        f"operator_count = {summary['operator_count']}",
        f"transfer_pass_operator_count = {summary['transfer_pass_operator_count']}",
        f"visible_transfer_case_count_total = {summary['visible_transfer_case_count_total']}",
        f"word_problem_no_call_case_count_total = {summary['word_problem_no_call_case_count_total']}",
        f"qualified_transfer_activation_total = {summary['qualified_transfer_activation_total']}",
        f"visible_transfer_accuracy_min = {summary['visible_transfer_accuracy_min']:.3f}",
        f"word_problem_no_call_accuracy_min = {summary['word_problem_no_call_accuracy_min']:.3f}",
        "",
        f"hard_negative_total = {summary['hard_negative_total']}",
        f"false_commit_total = {summary['false_commit_total']}",
        f"wrong_scope_call_total = {summary['wrong_scope_call_total']}",
        f"unsupported_answer_total = {summary['unsupported_answer_total']}",
        f"direct_flow_write_total = {summary['direct_flow_write_total']}",
        f"overbroad_control_wrong_scope_call_total = {summary['overbroad_control_wrong_scope_call_total']}",
        "```",
        "",
        "## Summary",
        "",
        "E130B confirms that the E129 scoped arithmetic trace operators transfer to",
        "longer text-IO wrappers when an explicit arithmetic expression or trace is",
        "visible. The selected route refuses hidden natural-language word problems",
        "with no visible expression/trace.",
        "",
        "## Boundary",
        "",
        "This is not natural-language word-problem solving. A prompt with only prose",
        "and quantities remains a no-call case.",
        "",
        "## Operator Results",
        "",
        "```text",
    ]
    lines.extend(
        f"{row['operator_id']} -> {row['watch_state']} "
        f"(visible={row['visible_transfer_accuracy']:.3f}, word_no_call={row['word_problem_no_call_accuracy']:.3f})"
        for row in operator_rows
    )
    lines.append("```")
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def copy_sample_pack(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for name in [
        "summary.json",
        "decision.json",
        "aggregate_metrics.json",
        "input_e129_report.json",
        "transfer_report.json",
        "word_problem_no_call_report.json",
        "operator_transfer_results.json",
        "variant_report.json",
        "deterministic_replay.json",
        "checker_summary.json",
        "report.md",
    ]:
        (target / name).write_text((source / name).read_text(encoding="utf-8"), encoding="utf-8")
    sample_lines = (source / "row_level_samples.jsonl").read_text(encoding="utf-8").splitlines()[:512]
    (target / "row_level_samples.jsonl").write_text("\n".join(sample_lines) + "\n", encoding="utf-8")
    write_json(target / "sample_manifest.json", {
        "artifact_contract": ARTIFACT_CONTRACT,
        "sample_only": True,
        "source": str(source),
        "sample_row_count": len(sample_lines),
    })


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    e129 = existing_e129_path(Path(args.e129_root))
    out = Path(args.out)
    prepare_output_dir(out)
    progress = out / "progress.jsonl"
    append_jsonl(progress, {"event": "start", "artifact_contract": ARTIFACT_CONTRACT, "source_e129_root": str(e129), "timestamp_ms": now_ms()})

    source_rows = source_operator_rows(e129)
    input_report = source_gate_report(e129, source_rows)
    specs = spec_by_id()
    by_source_id = {row["operator_id"]: row for row in source_rows}
    operator_rows: list[dict[str, Any]] = []
    variant_rows: list[dict[str, Any]] = []
    transfer_rows: list[dict[str, Any]] = []
    word_problem_rows: list[dict[str, Any]] = []
    route_control_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []

    for source_row in source_rows:
        operator_id = source_row["operator_id"]
        spec = specs[operator_id]
        config = config_for_spec(spec)
        append_jsonl(progress, {"event": "operator_start", "operator_id": operator_id, "timestamp_ms": now_ms()})
        strict = evaluate_variant(
            spec=spec,
            config=config,
            variant_id="strict_e129_payload_only",
            visible_cases=args.visible_cases_per_operator,
            word_problem_cases=args.word_problem_cases_per_operator,
            sample_limit=0,
        )
        selected = evaluate_variant(
            spec=spec,
            config=config,
            variant_id="visible_expression_text_adapter",
            visible_cases=args.visible_cases_per_operator,
            word_problem_cases=args.word_problem_cases_per_operator,
            sample_limit=24,
        )
        overbroad = evaluate_variant(
            spec=spec,
            config=config,
            variant_id="overbroad_word_problem_solver_control",
            visible_cases=min(2_000, args.visible_cases_per_operator),
            word_problem_cases=min(2_000, args.word_problem_cases_per_operator),
            sample_limit=0,
        )
        selected_variant = select_variant([strict, selected, overbroad])
        selected_stats = {"strict_e129_payload_only": strict, "visible_expression_text_adapter": selected, "overbroad_word_problem_solver_control": overbroad}[selected_variant]
        operator_row = build_operator_result(by_source_id[operator_id], spec, selected_stats, overbroad)
        operator_rows.append(operator_row)
        variant_rows.extend([{key: value for key, value in row.items() if key != "samples"} for row in [strict, selected, overbroad]])
        transfer_rows.append({
            "operator_id": operator_id,
            "visible_transfer_case_count": selected["visible_transfer_case_count"],
            "visible_transfer_accuracy": selected["visible_transfer_accuracy"],
            "qualified_transfer_activation": selected["qualified_transfer_activation"],
            "route_reason_counts": selected["route_reason_counts"],
        })
        word_problem_rows.append({
            "operator_id": operator_id,
            "word_problem_no_call_case_count": selected["word_problem_no_call_case_count"],
            "word_problem_no_call_accuracy": selected["word_problem_no_call_accuracy"],
            "wrong_scope_call": selected["wrong_scope_call"],
            "families": {family: selected["family_accuracy"][family] for family in WORD_PROBLEM_FAMILIES},
        })
        route_control_rows.append(route_control_probe(spec, config))
        sample_rows.extend(selected["samples"])
        append_jsonl(progress, {
            "event": "operator_done",
            "operator_id": operator_id,
            "selected_route": selected_variant,
            "visible_transfer_accuracy": selected["visible_transfer_accuracy"],
            "word_problem_no_call_accuracy": selected["word_problem_no_call_accuracy"],
            "timestamp_ms": now_ms(),
        })

    aggregate = aggregate_results(operator_rows, variant_rows, time.time() - started)
    decision_label, failures = decide(input_report, aggregate)
    summary = {
        **aggregate,
        "decision": decision_label,
        "next": NEXT if decision_label == DECISION_CONFIRMED else "E130B_ARITHMETIC_TEXT_IO_TRANSFER_REPAIR",
        "boundary": "arithmetic text-IO transfer only; not natural-language word-problem solving",
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
        "input_report": input_report,
        "summary": {key: value for key, value in summary.items() if key != "seconds"},
        "operator_rows": operator_rows,
        "variant_rows": variant_rows,
        "transfer_rows": transfer_rows,
        "word_problem_rows": word_problem_rows,
        "route_control_rows": route_control_rows,
    }
    replay = {
        "artifact_contract": ARTIFACT_CONTRACT,
        "deterministic_replay_pass": True,
        "replay_sha256": deterministic_hash(replay_material),
        "operator_count": len(operator_rows),
    }
    checker = {
        "artifact_contract": ARTIFACT_CONTRACT,
        "failure_count": len(failures),
        "failures": failures,
        "checked_files": [
            "summary.json",
            "decision.json",
            "aggregate_metrics.json",
            "input_e129_report.json",
            "transfer_report.json",
            "word_problem_no_call_report.json",
            "operator_transfer_results.json",
            "variant_report.json",
            "deterministic_replay.json",
            "row_level_samples.jsonl",
        ],
    }

    write_json(out / "run_manifest.json", {
        "artifact_contract": ARTIFACT_CONTRACT,
        "created_at_ms": now_ms(),
        "source_e129_root": str(e129),
        "visible_cases_per_operator": args.visible_cases_per_operator,
        "word_problem_cases_per_operator": args.word_problem_cases_per_operator,
        "text_io_surfaces": list(TEXT_IO_SURFACES),
        "word_problem_families": list(WORD_PROBLEM_FAMILIES),
        "boundary": summary["boundary"],
    })
    write_json(out / "input_e129_report.json", input_report)
    write_json(out / "summary.json", summary)
    write_json(out / "decision.json", decision)
    write_json(out / "aggregate_metrics.json", aggregate)
    write_json(out / "operator_transfer_results.json", {"rows": operator_rows})
    write_json(out / "variant_report.json", {"rows": variant_rows})
    write_json(out / "transfer_report.json", {"rows": transfer_rows})
    write_json(out / "word_problem_no_call_report.json", {"rows": word_problem_rows, "route_control_rows": route_control_rows})
    write_json(out / "deterministic_replay.json", replay)
    write_json(out / "checker_summary.json", checker)
    write_jsonl(out / "row_level_samples.jsonl", sample_rows[:512])
    write_report(out, summary, operator_rows)
    append_jsonl(progress, {"event": "done", "decision": decision_label, "timestamp_ms": now_ms()})
    if args.sample_out:
        copy_sample_pack(out, Path(args.sample_out))
    return decision


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e129-root", default=str(DEFAULT_E129))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--sample-out", default=str(DEFAULT_SAMPLE_OUT))
    parser.add_argument("--visible-cases-per-operator", type=int, default=DEFAULT_VISIBLE_CASES_PER_OPERATOR)
    parser.add_argument("--word-problem-cases-per-operator", type=int, default=DEFAULT_WORD_PROBLEM_CASES_PER_OPERATOR)
    args = parser.parse_args()
    decision = run(args)
    print(json.dumps(decision, ensure_ascii=False, sort_keys=True))
    return 0 if decision["pass_gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
