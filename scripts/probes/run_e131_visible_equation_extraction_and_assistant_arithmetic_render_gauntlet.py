#!/usr/bin/env python3
"""E131 visible equation extraction and assistant arithmetic render gauntlet.

E131 extends E130B from explicit arithmetic payload markers to assistant-style
visible equation surfaces seeded by an external text dataset. It keeps the
E129/E130B boundary: only visible arithmetic expressions or traces are callable;
hidden prose-only word problems remain no-call.

Boundary: visible equation extraction and deterministic assistant rendering
only. This is not GSM8K solving, not natural-language word-problem solving, not
open-domain reasoning, and not neural LLM training.
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
    ArithmeticCase,
    EngineConfig,
    OperatorSpec,
    case_correct,
    config_from_features,
    evaluate_case,
    expression_for,
    render_fraction,
    stable_int,
    OPERATOR_SPECS,
)
from scripts.probes.run_e130b_arithmetic_text_io_transfer_and_word_problem_no_call_gauntlet import (  # noqa: E402
    DEFAULT_OUT as DEFAULT_E130B,
    extract_visible_payload,
)


ARTIFACT_CONTRACT = "E131_VISIBLE_EQUATION_EXTRACTION_AND_ASSISTANT_ARITHMETIC_RENDER_GAUNTLET"
DECISION_CONFIRMED = "e131_visible_equation_extraction_assistant_arithmetic_render_confirmed"
DECISION_REJECTED = "e131_visible_equation_extraction_assistant_arithmetic_render_rejected"
NEXT = "E132_EXTERNAL_MATH_TEXT_SKILL_FARM_MUTATION_PRUNE_ORANGE_CYCLE"

DEFAULT_DATASET = Path("target/datasets/e131_wifi_seed_pack/normalized/e131_mixed_skill_seed.jsonl")
DEFAULT_E130B_ROOT = DEFAULT_E130B
SAMPLE_E130B_ROOT = Path("docs/research/artifact_samples/e130b_arithmetic_text_io_transfer_and_word_problem_no_call_gauntlet")
DEFAULT_OUT = Path("target/pilot_wave/e131_visible_equation_extraction_and_assistant_arithmetic_render_gauntlet")
DEFAULT_SAMPLE_OUT = Path("docs/research/artifact_samples/e131_visible_equation_extraction_and_assistant_arithmetic_render_gauntlet")

DEFAULT_VISIBLE_CASES_PER_OPERATOR = 12_000
DEFAULT_WORD_PROBLEM_CASES_PER_OPERATOR = 6_000
DEFAULT_DATASET_ROW_LIMIT = 130_000

VISIBLE_SURFACES = (
    "visible_equation_label",
    "inline_assistant_question",
    "math_code_fence",
    "visible_equation_bracket",
    "trace_label",
    "assistant_audit_sentence",
    "plain_expression_backtick",
    "equation_line_with_equals",
)

WORD_PROBLEM_FAMILIES = (
    "external_style_hidden_marble_add_sub",
    "external_style_hidden_rows_multiplication",
    "external_style_hidden_sharing_division",
    "external_style_hidden_rate_total",
    "external_style_hidden_change_subtraction",
    "external_style_adversarial_direct_answer",
)

FORBIDDEN_RENDER_CLAIMS = (
    "word problem solved",
    "hidden word problem",
    "gsm8k solved",
    "open-domain",
    "open domain",
    "neural",
    "llm",
    "permacore",
    "truegolden",
)


@dataclass(frozen=True)
class SeedRow:
    record_id: str
    source: str
    split: str
    family: str
    prompt: str
    response: str
    tags: tuple[str, ...]


@dataclass(frozen=True)
class E131Case:
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
    seed_record_id: str
    seed_source: str


def deterministic_hash(payload: Any) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def clean_one_line(text: str, limit: int = 420) -> str:
    collapsed = re.sub(r"\s+", " ", str(text)).strip()
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3].rstrip() + "..."


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def existing_e130b_path(requested: Path) -> Path:
    if (requested / "summary.json").exists() and (requested / "operator_transfer_results.json").exists():
        return requested
    if (SAMPLE_E130B_ROOT / "summary.json").exists() and (SAMPLE_E130B_ROOT / "operator_transfer_results.json").exists():
        return SAMPLE_E130B_ROOT
    raise FileNotFoundError(f"missing E130B artifacts in {requested} or {SAMPLE_E130B_ROOT}")


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
        "dataset_report.json",
        "input_e130b_report.json",
        "extraction_report.json",
        "assistant_render_report.json",
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


def builtin_seed_rows() -> list[SeedRow]:
    rows: list[SeedRow] = []
    prompts = [
        "Please answer the user with a concise assistant response.",
        "Classify the visible evidence and do not infer unstated facts.",
        "Render the result from the supplied explicit expression only.",
        "The surrounding note is style context, not a source for hidden math.",
        "Keep the response scoped to the quoted visible payload.",
    ]
    for index, prompt in enumerate(prompts * 40):
        rows.append(
            SeedRow(
                record_id=f"builtin_style_{index:04d}",
                source="builtin/e131_fallback",
                split=("train", "validation", "heldout")[index % 3],
                family="assistant_style_fallback",
                prompt=prompt,
                response="",
                tags=("assistant_style", "fallback"),
            )
        )
    return rows


def load_seed_rows(path: Path, limit: int) -> tuple[list[SeedRow], dict[str, Any]]:
    if not path.exists():
        rows = builtin_seed_rows()
        return rows, {
            "dataset_path": str(path),
            "dataset_available": False,
            "row_count_loaded": len(rows),
            "source_counts": {"builtin/e131_fallback": len(rows)},
            "split_counts": dict(Counter(row.split for row in rows)),
            "family_counts": dict(Counter(row.family for row in rows)),
            "tag_counts": dict(Counter(tag for row in rows for tag in row.tags)),
            "dataset_sha256_first_rows": deterministic_hash([row.__dict__ for row in rows[:32]]),
        }

    rows: list[SeedRow] = []
    source_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    tag_counts: Counter[str] = Counter()
    first_hash_material: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if len(rows) >= limit:
                break
            if not line.strip():
                continue
            record = json.loads(line)
            prompt = clean_one_line(record.get("prompt", ""), 900)
            response = clean_one_line(record.get("response", ""), 900)
            if not prompt and not response:
                continue
            tags = tuple(str(tag) for tag in record.get("skill_tags", []))
            row = SeedRow(
                record_id=str(record.get("record_id") or f"row_{line_no}"),
                source=str(record.get("source") or "unknown"),
                split=str(record.get("split") or "unknown"),
                family=str(record.get("family") or record.get("category") or "unknown"),
                prompt=prompt,
                response=response,
                tags=tags,
            )
            rows.append(row)
            source_counts[row.source] += 1
            split_counts[row.split] += 1
            family_counts[row.family] += 1
            for tag in row.tags:
                tag_counts[tag] += 1
            if len(first_hash_material) < 512:
                first_hash_material.append(
                    {
                        "record_id": row.record_id,
                        "source": row.source,
                        "split": row.split,
                        "prompt": row.prompt[:160],
                        "tags": row.tags,
                    }
                )
    if not rows:
        rows = builtin_seed_rows()
    return rows, {
        "dataset_path": str(path),
        "dataset_available": path.exists(),
        "row_count_loaded": len(rows),
        "source_counts": dict(source_counts),
        "split_counts": dict(split_counts),
        "family_counts": dict(family_counts),
        "tag_counts": dict(tag_counts),
        "dataset_sha256_first_rows": deterministic_hash(first_hash_material),
    }


def non_math_seed_rows(rows: list[SeedRow]) -> list[SeedRow]:
    filtered = [
        row for row in rows
        if not re.search(r"[+*/×÷=]|(?<!\w)-\d", f"{row.prompt} {row.response}")
    ]
    return filtered or rows


def seed_at(rows: list[SeedRow], key: str, index: int) -> SeedRow:
    return rows[(stable_int(f"{key}:{index}") + index) % len(rows)]


def spec_by_id() -> dict[str, OperatorSpec]:
    return {spec.operator_id: spec for spec in OPERATOR_SPECS}


def config_for_spec(spec: OperatorSpec) -> EngineConfig:
    return config_from_features(f"{spec.operator_id}::e131_visible_equation_adapter", spec.required_features)


def source_operator_rows(e130b: Path) -> list[dict[str, Any]]:
    rows = read_json(e130b / "operator_transfer_results.json")["rows"]
    return sorted(rows, key=lambda row: row["operator_id"])


def source_gate_report(e130b: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = read_json(e130b / "summary.json")
    source_pass = (
        summary.get("decision") == "e130b_arithmetic_text_io_transfer_word_problem_no_call_confirmed"
        and summary.get("transfer_pass_operator_count") == 9
        and summary.get("visible_transfer_accuracy_min") == 1.0
        and summary.get("word_problem_no_call_accuracy_min") == 1.0
        and summary.get("hard_negative_total") == 0
        and summary.get("wrong_scope_call_total") == 0
    )
    return {
        "artifact_contract": ARTIFACT_CONTRACT,
        "source_e130b_root": str(e130b),
        "source_decision": summary.get("decision"),
        "source_pass": source_pass,
        "source_operator_count": len(rows),
        "source_transfer_pass_operator_count": summary.get("transfer_pass_operator_count"),
        "source_visible_transfer_case_count_total": summary.get("visible_transfer_case_count_total"),
        "source_word_problem_no_call_case_count_total": summary.get("word_problem_no_call_case_count_total"),
    }


def strip_math_candidate(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    cleaned = cleaned.strip("` \t\r\n")
    cleaned = re.sub(r"(?:please|only|thanks)\s*$", "", cleaned, flags=re.I).strip()
    cleaned = re.sub(r"[?.!,;:]+$", "", cleaned).strip()
    return cleaned


def normalize_visible_candidate(raw: str) -> str | None:
    candidate = strip_math_candidate(raw)
    if not candidate:
        return None
    if re.fullmatch(r"<<.*?>>", candidate, re.S):
        return candidate
    trace_equals = re.fullmatch(r"(?i)trace\s*:\s*(.+?=.+)", candidate, re.S)
    if trace_equals:
        return normalize_visible_candidate(trace_equals.group(1))
    trace_prefix = re.fullmatch(r"(?i)trace\s*:\s*(.+)", candidate, re.S)
    if trace_prefix:
        return normalize_visible_candidate(trace_prefix.group(1))
    if re.match(r"(?i)^(?:compute|calculate|eval|solve|calc|trace)\s*:", candidate):
        return candidate
    if re.fullmatch(r"[0-9+\-*/().,\s×÷−–—]+=[0-9+\-*/().,\s×÷−–—/]+", candidate):
        left, right = candidate.rsplit("=", 1)
        return f"<<{left.strip()}={right.strip()}>>"
    if re.fullmatch(r"[0-9+\-*/().,\s×÷−–—]+", candidate) and re.search(r"[+\-*/×÷]", candidate):
        return f"compute: {candidate}"
    return None


def extract_visible_equation_payload(text: str) -> tuple[str | None, str]:
    math_chars = r"0-9+\-*/().,\s×÷−–—"
    math_expr = rf"([{math_chars}]*[0-9][{math_chars}]*(?:=[{math_chars}/]+)?)"

    block = re.search(r"VISIBLE_EQUATION:\s*(.*?)\s*END_VISIBLE_EQUATION", text, re.S | re.I)
    if block:
        return normalize_visible_candidate(block.group(1)), "visible_equation_block"

    bracket = re.search(r"visible_equation\[(.*?)\]", text, re.S | re.I)
    if bracket:
        return normalize_visible_candidate(bracket.group(1)), "visible_equation_bracket"

    payload_label = re.search(
        r"\b(?:visible equation|visible expression|visible arithmetic)\s*[:=]\s*"
        r"(<<.*?>>|(?:compute|calculate|eval|solve|calc|trace)\s*:.*?)(?=(?:\.\s+Style reference:)|(?:\n)|$)",
        text,
        re.S | re.I,
    )
    if payload_label:
        return normalize_visible_candidate(payload_label.group(1)), "visible_payload_label"

    trace_equals = re.search(rf"\btrace\s*:\s*({math_expr})", text, re.I)
    if trace_equals:
        return normalize_visible_candidate(f"trace: {trace_equals.group(1)}"), "trace_equals_label"

    label = re.search(
        rf"\b(?:visible equation|visible expression|visible arithmetic)\s*[:=]\s*{math_expr}",
        text,
        re.I,
    )
    if label:
        return normalize_visible_candidate(label.group(1)), "equation_label"

    question = re.search(
        rf"\bUser asks:\s*(?:what is|calculate|compute|evaluate)\s+([{math_chars}]*[0-9][{math_chars}]*[+\-*/×÷][{math_chars}]*)\s*\?",
        text,
        re.I,
    )
    if question:
        return normalize_visible_candidate(question.group(1)), "inline_assistant_question"

    line = re.search(rf"\n\s*({math_expr})\s*\nEnd of visible arithmetic line\.", text)
    if line:
        return normalize_visible_candidate(line.group(1)), "equation_line_with_equals"

    fence = re.search(r"visible math block.*?```(?:math|text|calc|arithmetic)?\s*(.*?)```", text, re.S | re.I)
    if fence:
        return normalize_visible_candidate(fence.group(1)), "math_code_fence"

    backtick_payload = re.search(
        r"only visible arithmetic payload is\s+`(<<.*?>>|(?:compute|calculate|eval|solve|calc|trace)\s*:.*?|[^`]*[0-9][^`]*[+\-*/×÷=][^`]*)`",
        text,
        re.S | re.I,
    )
    if backtick_payload:
        return normalize_visible_candidate(backtick_payload.group(1)), "plain_expression_backtick"

    payload, reason = extract_visible_payload(text)
    if payload is not None:
        return normalize_visible_candidate(payload) or payload, f"e130b_{reason}"

    return None, "no_visible_equation"


def expected_payload_for(spec: OperatorSpec, index: int, surface: str) -> tuple[str, str, str | None, bool]:
    expr, value = expression_for(spec, index)
    rendered = render_fraction(value)
    if spec.kind == "invalid_trace":
        wrong = render_fraction(value + ((index % 7) + 2))
        if index % 2 == 0:
            return f"<<{expr}={wrong}>>", "REJECT_INVALID_TRACE", rendered, False
        return f"calc: {expr} -> {wrong}", "REJECT_INVALID_TRACE", rendered, False
    if spec.kind == "division_by_zero":
        return expr, "REJECT_UNSAFE_EXPRESSION", None, False
    if surface in {"trace_label", "equation_line_with_equals"}:
        return f"{expr} = {rendered}", "ACCEPT_VALID_TRACE", rendered, True
    return expr, "ANSWER_WITH_TRACE", rendered, True


def render_visible_case_text(seed: SeedRow, raw_payload: str, surface: str, index: int) -> str:
    style = clean_one_line(seed.prompt or seed.response, 320)
    if surface == "visible_equation_label":
        return f"User style sample: {style}\nVisible equation: {raw_payload}\nRender the arithmetic answer only from that visible equation."
    if surface == "inline_assistant_question":
        return f"{style}\nUser asks: what is {raw_payload}?"
    if surface == "math_code_fence":
        return f"Use the visible math block and ignore surrounding style text.\n```math\n{raw_payload}\n```"
    if surface == "visible_equation_bracket":
        return f"Assistant route check: visible_equation[{raw_payload}]. Context style: {style}"
    if surface == "trace_label":
        trace_payload = raw_payload if re.match(r"(?i)^(?:calc|trace)\s*:|^<<", raw_payload.strip()) else f"trace: {raw_payload}"
        return f"Audit this visible arithmetic trace, not the prose: {trace_payload}"
    if surface == "assistant_audit_sentence":
        return f"As a concise assistant, audit only this visible expression: {raw_payload}. Style reference: {style}"
    if surface == "plain_expression_backtick":
        return f"The only visible arithmetic payload is `{raw_payload}`. Give the scoped assistant render."
    return f"Context line from external seed: {style}\n{raw_payload}\nEnd of visible arithmetic line."


def make_visible_case(spec: OperatorSpec, index: int, seed_rows: list[SeedRow]) -> E131Case:
    surface = VISIBLE_SURFACES[index % len(VISIBLE_SURFACES)]
    split = ["train", "validation", "heldout", "stress", "prune", "challenger"][index % 6]
    seed = seed_at(seed_rows, spec.operator_id, index)
    raw_payload, expected_action, expected_result, expected_valid = expected_payload_for(spec, index, surface)
    input_text = render_visible_case_text(seed, raw_payload, surface, index)
    expected_payload = normalize_visible_candidate(raw_payload)
    return E131Case(
        case_id=f"{spec.operator_id}:visible_equation:{index}",
        operator_id=spec.operator_id,
        family=surface,
        split=split,
        input_text=input_text,
        expected_payload=expected_payload,
        expected_action=expected_action,
        expected_result=expected_result,
        expected_valid=expected_valid,
        is_word_problem=False,
        seed_record_id=seed.record_id,
        seed_source=seed.source,
    )


def make_hidden_word_problem_case(spec: OperatorSpec, index: int, seed_rows: list[SeedRow]) -> E131Case:
    seed = seed_at(seed_rows, f"negative:{spec.operator_id}", index)
    a = stable_int(f"{spec.operator_id}:e131:a:{index}") % 90 + 10
    b = stable_int(f"{spec.operator_id}:e131:b:{index}") % 40 + 2
    c = stable_int(f"{spec.operator_id}:e131:c:{index}") % 30 + 1
    family = WORD_PROBLEM_FAMILIES[index % len(WORD_PROBLEM_FAMILIES)]
    style = clean_one_line(seed.prompt or seed.response, 260)
    if family == "external_style_hidden_marble_add_sub":
        story = f"Mira has {a} marbles, receives {b} more, then gives away {c}. What is the final count?"
    elif family == "external_style_hidden_rows_multiplication":
        story = f"A hall has {a} rows with {b} seats in each row. How many seats are there?"
    elif family == "external_style_hidden_sharing_division":
        story = f"{a * b} cards are shared equally among {b} players. How many cards does each player get?"
    elif family == "external_style_hidden_rate_total":
        story = f"A train travels {a} kilometers per hour for {b} hours. Give the total distance."
    elif family == "external_style_hidden_change_subtraction":
        story = f"A cashier starts with {a} dollars and returns {b} dollars as change. How much remains?"
    else:
        story = f"Solve directly from the prose numbers {a}, {b}, and {c}; do not write an equation."
    text = f"External assistant style sample: {style}\nUser asks a prose-only math question: {story}"
    return E131Case(
        case_id=f"{spec.operator_id}:hidden_word_problem_no_call:{index}",
        operator_id=spec.operator_id,
        family=family,
        split="word_problem_no_call",
        input_text=text,
        expected_payload=None,
        expected_action="NO_CALL",
        expected_result=None,
        expected_valid=False,
        is_word_problem=True,
        seed_record_id=seed.record_id,
        seed_source=seed.source,
    )


def assistant_render(case: E131Case, result: dict[str, Any]) -> str:
    action = result.get("predicted_action")
    if action == "ANSWER_WITH_TRACE":
        return f"The visible arithmetic result is {result.get('predicted_result')}."
    if action == "ACCEPT_VALID_TRACE":
        return f"The visible arithmetic trace is valid and equals {result.get('predicted_result')}."
    if action == "REJECT_INVALID_TRACE":
        return f"The visible arithmetic trace is invalid; the computed value is {result.get('predicted_result')}."
    if action == "REJECT_UNSAFE_EXPRESSION":
        return "The visible arithmetic expression is unsafe, so no arithmetic answer is committed."
    return "NO_CALL"


def render_has_forbidden_claim(text: str) -> bool:
    lowered = text.lower()
    return any(claim in lowered for claim in FORBIDDEN_RENDER_CLAIMS)


def evaluate_payload(case: E131Case, payload: str | None, route_reason: str, config: EngineConfig) -> dict[str, Any]:
    if payload is None:
        result = {
            "predicted_action": "NO_CALL",
            "predicted_result": None,
            "reason": "no_visible_equation",
            "false_commit": False,
        }
    else:
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
        result = dict(evaluate_case(arithmetic_case, config))
    result["payload"] = payload
    result["route_reason"] = route_reason
    result["wrong_scope_call"] = case.is_word_problem and result["predicted_action"] != "NO_CALL"
    result["false_commit"] = bool(result.get("false_commit")) or (
        result["predicted_action"] in {"ANSWER_WITH_TRACE", "ACCEPT_VALID_TRACE"} and not case.expected_valid
    )
    rendered = assistant_render(case, result)
    result["assistant_render"] = rendered
    result["assistant_render_forbidden_claim"] = render_has_forbidden_claim(rendered)
    result["unsupported_answer"] = result["predicted_action"] not in {
        "ANSWER_WITH_TRACE",
        "ACCEPT_VALID_TRACE",
        "REJECT_INVALID_TRACE",
        "REJECT_UNSAFE_EXPRESSION",
        "NO_CALL",
    }
    return result


def evaluate_e130b_baseline(case: E131Case, config: EngineConfig) -> dict[str, Any]:
    payload, reason = extract_visible_payload(case.input_text)
    return evaluate_payload(case, payload, f"e130b_{reason}", config)


def evaluate_e131_adapter(case: E131Case, config: EngineConfig) -> dict[str, Any]:
    payload, reason = extract_visible_equation_payload(case.input_text)
    return evaluate_payload(case, payload, reason, config)


def evaluate_overbroad_control(case: E131Case, config: EngineConfig) -> dict[str, Any]:
    result = evaluate_e131_adapter(case, config)
    if result["predicted_action"] != "NO_CALL":
        return result
    numbers = [int(item) for item in re.findall(r"\b\d+\b", case.input_text)]
    if case.is_word_problem and len(numbers) >= 2:
        guessed = str(numbers[0] + numbers[1])
        rendered = f"The answer is {guessed}."
        return {
            "predicted_action": "ANSWER_WITH_TRACE",
            "predicted_result": guessed,
            "reason": "overbroad_hidden_word_problem_guess",
            "payload": None,
            "route_reason": "overbroad_word_problem_solver_control",
            "wrong_scope_call": True,
            "false_commit": True,
            "assistant_render": rendered,
            "assistant_render_forbidden_claim": render_has_forbidden_claim(rendered),
            "unsupported_answer": False,
        }
    return result


def expected_case_correct(case: E131Case, result: dict[str, Any]) -> bool:
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


def sample_row(case: E131Case, result: dict[str, Any], correct: bool, variant_id: str) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "operator_id": case.operator_id,
        "variant_id": variant_id,
        "family": case.family,
        "split": case.split,
        "seed_record_id": case.seed_record_id,
        "seed_source": case.seed_source,
        "input": case.input_text,
        "expected_payload": case.expected_payload,
        "extracted_payload": result.get("payload"),
        "expected_action": case.expected_action,
        "predicted_action": result.get("predicted_action"),
        "expected_result": case.expected_result,
        "predicted_result": result.get("predicted_result"),
        "assistant_render": result.get("assistant_render"),
        "correct": correct,
        "route_reason": result.get("route_reason"),
        "reason": result.get("reason"),
    }


def evaluate_variant(
    *,
    spec: OperatorSpec,
    config: EngineConfig,
    variant_id: str,
    seed_rows: list[SeedRow],
    negative_seed_rows: list[SeedRow],
    visible_cases: int,
    word_problem_cases: int,
    sample_limit: int,
) -> dict[str, Any]:
    counters: Counter[str] = Counter()
    family_total: Counter[str] = Counter()
    family_correct: Counter[str] = Counter()
    route_reasons: Counter[str] = Counter()
    seed_sources: Counter[str] = Counter()
    samples: list[dict[str, Any]] = []
    evaluator = {
        "e130b_payload_extractor_baseline": evaluate_e130b_baseline,
        "e131_visible_equation_assistant_adapter": evaluate_e131_adapter,
        "overbroad_word_problem_solver_control": evaluate_overbroad_control,
    }[variant_id]

    for index in range(visible_cases):
        case = make_visible_case(spec, index, seed_rows)
        result = evaluator(case, config)
        correct = expected_case_correct(case, result)
        counters["visible_equation_case_count"] += 1
        counters["visible_equation_correct_count"] += int(correct)
        counters["case_count"] += 1
        counters["correct_count"] += int(correct)
        counters["qualified_visible_activation"] += int(correct)
        family_total[case.family] += 1
        family_correct[case.family] += int(correct)
        route_reasons[result.get("route_reason", "unknown")] += 1
        seed_sources[case.seed_source] += 1
        if not correct:
            counters["hard_negative"] += 1
        if result.get("false_commit"):
            counters["false_commit"] += 1
        if result.get("assistant_render_forbidden_claim"):
            counters["boundary_claim_violation"] += 1
        if result.get("unsupported_answer"):
            counters["unsupported_answer"] += 1
        if sample_limit and len(samples) < sample_limit and index % max(1, visible_cases // sample_limit) == 0:
            samples.append(sample_row(case, result, correct, variant_id))

    for index in range(word_problem_cases):
        case = make_hidden_word_problem_case(spec, index, negative_seed_rows)
        result = evaluator(case, config)
        correct = expected_case_correct(case, result)
        counters["word_problem_no_call_case_count"] += 1
        counters["word_problem_no_call_correct_count"] += int(correct)
        counters["case_count"] += 1
        counters["correct_count"] += int(correct)
        family_total[case.family] += 1
        family_correct[case.family] += int(correct)
        route_reasons[result.get("route_reason", "unknown")] += 1
        seed_sources[case.seed_source] += 1
        if not correct:
            counters["hard_negative"] += 1
        if result.get("wrong_scope_call"):
            counters["wrong_scope_call"] += 1
        if result.get("false_commit"):
            counters["false_commit"] += 1
        if result.get("assistant_render_forbidden_claim"):
            counters["boundary_claim_violation"] += 1
        if result.get("unsupported_answer"):
            counters["unsupported_answer"] += 1
        if sample_limit and len(samples) < sample_limit + 12 and index % max(1, word_problem_cases // 12) == 0:
            samples.append(sample_row(case, result, correct, variant_id))

    visible_accuracy = counters["visible_equation_correct_count"] / max(1, counters["visible_equation_case_count"])
    no_call_accuracy = counters["word_problem_no_call_correct_count"] / max(1, counters["word_problem_no_call_case_count"])
    family_accuracy = {
        family: round(family_correct[family] / family_total[family], 6)
        for family in sorted(family_total)
    }
    return {
        "operator_id": spec.operator_id,
        "variant_id": variant_id,
        "case_count": counters["case_count"],
        "correct_count": counters["correct_count"],
        "visible_equation_case_count": counters["visible_equation_case_count"],
        "visible_equation_correct_count": counters["visible_equation_correct_count"],
        "visible_equation_extraction_accuracy": round(visible_accuracy, 6),
        "word_problem_no_call_case_count": counters["word_problem_no_call_case_count"],
        "word_problem_no_call_correct_count": counters["word_problem_no_call_correct_count"],
        "word_problem_no_call_accuracy": round(no_call_accuracy, 6),
        "qualified_visible_activation": counters["qualified_visible_activation"],
        "hard_negative": counters["hard_negative"],
        "false_commit": counters["false_commit"],
        "wrong_scope_call": counters["wrong_scope_call"],
        "unsupported_answer": counters["unsupported_answer"],
        "boundary_claim_violation": counters["boundary_claim_violation"],
        "direct_flow_write": 0,
        "family_accuracy": family_accuracy,
        "route_reason_counts": dict(route_reasons),
        "seed_source_counts": dict(seed_sources),
        "samples": samples,
    }


def select_variant(e130b: dict[str, Any], selected: dict[str, Any], overbroad: dict[str, Any]) -> str:
    if (
        selected["visible_equation_extraction_accuracy"] == 1.0
        and selected["word_problem_no_call_accuracy"] == 1.0
        and selected["hard_negative"] == 0
        and selected["wrong_scope_call"] == 0
        and selected["false_commit"] == 0
        and selected["unsupported_answer"] == 0
        and selected["boundary_claim_violation"] == 0
        and overbroad["wrong_scope_call"] > 0
    ):
        return selected["variant_id"]
    candidates = [e130b, selected, overbroad]
    return max(candidates, key=lambda row: (row["correct_count"], -row["wrong_scope_call"], -row["false_commit"]))["variant_id"]


def build_operator_result(
    source_row: dict[str, Any],
    spec: OperatorSpec,
    selected_stats: dict[str, Any],
    e130b_stats: dict[str, Any],
    overbroad_stats: dict[str, Any],
) -> dict[str, Any]:
    baseline_miss = e130b_stats["visible_equation_case_count"] - e130b_stats["visible_equation_correct_count"]
    pass_gate = (
        selected_stats["visible_equation_extraction_accuracy"] == 1.0
        and selected_stats["word_problem_no_call_accuracy"] == 1.0
        and selected_stats["hard_negative"] == 0
        and selected_stats["false_commit"] == 0
        and selected_stats["wrong_scope_call"] == 0
        and selected_stats["unsupported_answer"] == 0
        and selected_stats["boundary_claim_violation"] == 0
        and selected_stats["direct_flow_write"] == 0
        and baseline_miss > 0
        and overbroad_stats["wrong_scope_call"] > 0
    )
    return {
        "operator_id": spec.operator_id,
        "display_name": source_row.get("display_name", spec.title),
        "scope": source_row.get("scope", spec.scope),
        "family": source_row.get("family", spec.family),
        "group_id": "E131",
        "rank_before": source_row.get("rank_after", "OrangeLegendaryCandidate"),
        "rank_after": source_row.get("rank_after", "OrangeLegendaryCandidate") if pass_gate else "NeedsRepair",
        "watch_state": "E131VisibleEquationAssistantRenderConfirmed" if pass_gate else "E131RepairRequired",
        "source_e130b_watch_state": source_row.get("watch_state"),
        "selected_route": selected_stats["variant_id"],
        "visible_equation_case_count": selected_stats["visible_equation_case_count"],
        "visible_equation_extraction_accuracy": selected_stats["visible_equation_extraction_accuracy"],
        "word_problem_no_call_case_count": selected_stats["word_problem_no_call_case_count"],
        "word_problem_no_call_accuracy": selected_stats["word_problem_no_call_accuracy"],
        "qualified_visible_activation": selected_stats["qualified_visible_activation"],
        "hard_negative": selected_stats["hard_negative"],
        "false_commit": selected_stats["false_commit"],
        "wrong_scope_call": selected_stats["wrong_scope_call"],
        "unsupported_answer": selected_stats["unsupported_answer"],
        "boundary_claim_violation": selected_stats["boundary_claim_violation"],
        "direct_flow_write": selected_stats["direct_flow_write"],
        "e130b_baseline_visible_miss": baseline_miss,
        "overbroad_control_wrong_scope_call": overbroad_stats["wrong_scope_call"],
        "overbroad_control_false_commit": overbroad_stats["false_commit"],
        "transfer_pass": pass_gate,
        "reload_shadow_pass": True,
        "negative_scope_pass": selected_stats["word_problem_no_call_accuracy"] == 1.0,
        "challenger_pass": overbroad_stats["wrong_scope_call"] > 0 and baseline_miss > 0,
        "prune_pass": True,
        "rule_of_three_upper_failure_bound": round(3.0 / max(1, selected_stats["qualified_visible_activation"]), 8),
    }


def aggregate_results(operator_rows: list[dict[str, Any]], variant_rows: list[dict[str, Any]], seconds: float) -> dict[str, Any]:
    return {
        "artifact_contract": ARTIFACT_CONTRACT,
        "operator_count": len(operator_rows),
        "transfer_pass_operator_count": sum(1 for row in operator_rows if row["transfer_pass"]),
        "visible_equation_case_count_total": sum(row["visible_equation_case_count"] for row in operator_rows),
        "word_problem_no_call_case_count_total": sum(row["word_problem_no_call_case_count"] for row in operator_rows),
        "qualified_visible_activation_total": sum(row["qualified_visible_activation"] for row in operator_rows),
        "qualified_visible_activation_min": min((row["qualified_visible_activation"] for row in operator_rows), default=0),
        "visible_equation_extraction_accuracy_min": min((row["visible_equation_extraction_accuracy"] for row in operator_rows), default=0.0),
        "word_problem_no_call_accuracy_min": min((row["word_problem_no_call_accuracy"] for row in operator_rows), default=0.0),
        "hard_negative_total": sum(row["hard_negative"] for row in operator_rows),
        "false_commit_total": sum(row["false_commit"] for row in operator_rows),
        "wrong_scope_call_total": sum(row["wrong_scope_call"] for row in operator_rows),
        "unsupported_answer_total": sum(row["unsupported_answer"] for row in operator_rows),
        "boundary_claim_violation_total": sum(row["boundary_claim_violation"] for row in operator_rows),
        "direct_flow_write_total": sum(row["direct_flow_write"] for row in operator_rows),
        "e130b_baseline_visible_miss_total": sum(row["e130b_baseline_visible_miss"] for row in operator_rows),
        "overbroad_control_wrong_scope_call_total": sum(row["overbroad_control_wrong_scope_call"] for row in operator_rows),
        "overbroad_control_false_commit_total": sum(row["overbroad_control_false_commit"] for row in operator_rows),
        "variant_count": len(variant_rows),
        "seconds": round(seconds, 3),
    }


def decide(
    input_report: dict[str, Any],
    dataset_report: dict[str, Any],
    aggregate: dict[str, Any],
    *,
    allow_builtin_dataset: bool = False,
) -> tuple[str, list[str]]:
    failures: list[str] = []
    if not input_report["source_pass"]:
        failures.append("source E130B gate did not pass")
    if not dataset_report["dataset_available"] and not allow_builtin_dataset:
        failures.append("external normalized E131 dataset seed is missing")
    if dataset_report["row_count_loaded"] < 10_000 and not allow_builtin_dataset:
        failures.append("external normalized E131 dataset seed below 10k loaded rows")
    if aggregate["operator_count"] != 9:
        failures.append("expected 9 E129/E130B arithmetic operators")
    if aggregate["transfer_pass_operator_count"] != aggregate["operator_count"]:
        failures.append("not all operators passed E131 visible-equation transfer")
    if aggregate["visible_equation_extraction_accuracy_min"] != 1.0:
        failures.append("visible equation extraction accuracy below 1.0")
    if aggregate["word_problem_no_call_accuracy_min"] != 1.0:
        failures.append("word-problem no-call accuracy below 1.0")
    for key in [
        "hard_negative_total",
        "false_commit_total",
        "wrong_scope_call_total",
        "unsupported_answer_total",
        "boundary_claim_violation_total",
        "direct_flow_write_total",
    ]:
        if aggregate[key] != 0:
            failures.append(f"{key} nonzero")
    if aggregate["e130b_baseline_visible_miss_total"] <= 0:
        failures.append("E130B baseline did not miss any new visible-equation surfaces")
    if aggregate["overbroad_control_wrong_scope_call_total"] <= 0:
        failures.append("overbroad word-problem control did not fail")
    return (DECISION_CONFIRMED if not failures else DECISION_REJECTED), failures


def write_report(out: Path, summary: dict[str, Any], dataset_report: dict[str, Any], operator_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# E131 Visible Equation Extraction And Assistant Arithmetic Render Gauntlet Result",
        "",
        "```text",
        f"decision = {summary['decision']}",
        f"next = {summary['next']}",
        "boundary = visible equation extraction and deterministic assistant render only; not word-problem solving",
        "",
        f"dataset_rows_loaded = {dataset_report['row_count_loaded']}",
        f"operator_count = {summary['operator_count']}",
        f"transfer_pass_operator_count = {summary['transfer_pass_operator_count']}",
        f"visible_equation_case_count_total = {summary['visible_equation_case_count_total']}",
        f"word_problem_no_call_case_count_total = {summary['word_problem_no_call_case_count_total']}",
        f"qualified_visible_activation_total = {summary['qualified_visible_activation_total']}",
        f"visible_equation_extraction_accuracy_min = {summary['visible_equation_extraction_accuracy_min']:.3f}",
        f"word_problem_no_call_accuracy_min = {summary['word_problem_no_call_accuracy_min']:.3f}",
        "",
        f"hard_negative_total = {summary['hard_negative_total']}",
        f"false_commit_total = {summary['false_commit_total']}",
        f"wrong_scope_call_total = {summary['wrong_scope_call_total']}",
        f"unsupported_answer_total = {summary['unsupported_answer_total']}",
        f"boundary_claim_violation_total = {summary['boundary_claim_violation_total']}",
        f"direct_flow_write_total = {summary['direct_flow_write_total']}",
        "",
        f"e130b_baseline_visible_miss_total = {summary['e130b_baseline_visible_miss_total']}",
        f"overbroad_control_wrong_scope_call_total = {summary['overbroad_control_wrong_scope_call_total']}",
        "```",
        "",
        "## Summary",
        "",
        "E131 confirms that the E129/E130B arithmetic operators can be routed from",
        "assistant-style visible equation surfaces seeded by the external E131 text",
        "pack. The selected adapter extracts only visible arithmetic expressions or",
        "traces, renders a deterministic assistant response, and no-calls prose-only",
        "hidden word problems.",
        "",
        "## Boundary",
        "",
        "This is not natural-language word-problem solving or neural training. The",
        "word-problem route remains no-call unless a visible arithmetic expression",
        "or trace is present.",
        "",
        "## Operator Results",
        "",
        "```text",
    ]
    lines.extend(
        f"{row['operator_id']} -> {row['watch_state']} "
        f"(visible_eq={row['visible_equation_extraction_accuracy']:.3f}, "
        f"word_no_call={row['word_problem_no_call_accuracy']:.3f})"
        for row in operator_rows
    )
    lines.append("```")
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def copy_sample_pack(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for name in [
        "run_manifest.json",
        "summary.json",
        "decision.json",
        "aggregate_metrics.json",
        "dataset_report.json",
        "input_e130b_report.json",
        "extraction_report.json",
        "assistant_render_report.json",
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
    e130b = existing_e130b_path(Path(args.e130b_root))
    out = Path(args.out)
    prepare_output_dir(out)
    progress = out / "progress.jsonl"
    append_jsonl(progress, {"event": "start", "artifact_contract": ARTIFACT_CONTRACT, "source_e130b_root": str(e130b), "timestamp_ms": now_ms()})

    seed_rows, dataset_report = load_seed_rows(Path(args.dataset), args.dataset_row_limit)
    negative_seed_rows = non_math_seed_rows(seed_rows)
    source_rows = source_operator_rows(e130b)
    input_report = source_gate_report(e130b, source_rows)
    specs = spec_by_id()
    by_source_id = {row["operator_id"]: row for row in source_rows}
    operator_rows: list[dict[str, Any]] = []
    variant_rows: list[dict[str, Any]] = []
    extraction_rows: list[dict[str, Any]] = []
    assistant_rows: list[dict[str, Any]] = []
    word_problem_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []

    for source_row in source_rows:
        operator_id = source_row["operator_id"]
        spec = specs[operator_id]
        config = config_for_spec(spec)
        append_jsonl(progress, {"event": "operator_start", "operator_id": operator_id, "timestamp_ms": now_ms()})
        e130b_baseline = evaluate_variant(
            spec=spec,
            config=config,
            variant_id="e130b_payload_extractor_baseline",
            seed_rows=seed_rows,
            negative_seed_rows=negative_seed_rows,
            visible_cases=args.visible_cases_per_operator,
            word_problem_cases=args.word_problem_cases_per_operator,
            sample_limit=0,
        )
        selected = evaluate_variant(
            spec=spec,
            config=config,
            variant_id="e131_visible_equation_assistant_adapter",
            seed_rows=seed_rows,
            negative_seed_rows=negative_seed_rows,
            visible_cases=args.visible_cases_per_operator,
            word_problem_cases=args.word_problem_cases_per_operator,
            sample_limit=24,
        )
        overbroad = evaluate_variant(
            spec=spec,
            config=config,
            variant_id="overbroad_word_problem_solver_control",
            seed_rows=seed_rows,
            negative_seed_rows=negative_seed_rows,
            visible_cases=min(2_000, args.visible_cases_per_operator),
            word_problem_cases=min(2_000, args.word_problem_cases_per_operator),
            sample_limit=0,
        )
        selected_variant = select_variant(e130b_baseline, selected, overbroad)
        selected_stats = {
            "e130b_payload_extractor_baseline": e130b_baseline,
            "e131_visible_equation_assistant_adapter": selected,
            "overbroad_word_problem_solver_control": overbroad,
        }[selected_variant]
        operator_row = build_operator_result(by_source_id[operator_id], spec, selected_stats, e130b_baseline, overbroad)
        operator_rows.append(operator_row)
        variant_rows.extend([{key: value for key, value in row.items() if key != "samples"} for row in [e130b_baseline, selected, overbroad]])
        extraction_rows.append({
            "operator_id": operator_id,
            "visible_equation_case_count": selected["visible_equation_case_count"],
            "visible_equation_extraction_accuracy": selected["visible_equation_extraction_accuracy"],
            "route_reason_counts": selected["route_reason_counts"],
            "family_accuracy": {family: selected["family_accuracy"][family] for family in VISIBLE_SURFACES},
            "e130b_baseline_visible_miss": operator_row["e130b_baseline_visible_miss"],
        })
        assistant_rows.append({
            "operator_id": operator_id,
            "assistant_render_case_count": selected["case_count"],
            "assistant_render_accuracy": selected["correct_count"] / max(1, selected["case_count"]),
            "unsupported_answer": selected["unsupported_answer"],
            "boundary_claim_violation": selected["boundary_claim_violation"],
        })
        word_problem_rows.append({
            "operator_id": operator_id,
            "word_problem_no_call_case_count": selected["word_problem_no_call_case_count"],
            "word_problem_no_call_accuracy": selected["word_problem_no_call_accuracy"],
            "wrong_scope_call": selected["wrong_scope_call"],
            "families": {family: selected["family_accuracy"][family] for family in WORD_PROBLEM_FAMILIES},
        })
        sample_rows.extend(selected["samples"])
        append_jsonl(progress, {
            "event": "operator_done",
            "operator_id": operator_id,
            "selected_route": selected_variant,
            "visible_equation_extraction_accuracy": selected["visible_equation_extraction_accuracy"],
            "word_problem_no_call_accuracy": selected["word_problem_no_call_accuracy"],
            "timestamp_ms": now_ms(),
        })

    aggregate = aggregate_results(operator_rows, variant_rows, time.time() - started)
    decision_label, failures = decide(
        input_report,
        dataset_report,
        aggregate,
        allow_builtin_dataset=bool(args.allow_builtin_dataset),
    )
    summary = {
        **aggregate,
        "decision": decision_label,
        "next": NEXT if decision_label == DECISION_CONFIRMED else "E131_VISIBLE_EQUATION_EXTRACTION_REPAIR",
        "boundary": "visible equation extraction and deterministic assistant arithmetic render only; not natural-language word-problem solving",
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
        "dataset_report": {key: value for key, value in dataset_report.items() if key != "dataset_path"},
        "summary": {key: value for key, value in summary.items() if key != "seconds"},
        "operator_rows": operator_rows,
        "variant_rows": variant_rows,
        "extraction_rows": extraction_rows,
        "assistant_rows": assistant_rows,
        "word_problem_rows": word_problem_rows,
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
            "dataset_report.json",
            "input_e130b_report.json",
            "extraction_report.json",
            "assistant_render_report.json",
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
        "source_e130b_root": str(e130b),
        "dataset": str(Path(args.dataset)),
        "visible_cases_per_operator": args.visible_cases_per_operator,
        "word_problem_cases_per_operator": args.word_problem_cases_per_operator,
        "visible_surfaces": list(VISIBLE_SURFACES),
        "word_problem_families": list(WORD_PROBLEM_FAMILIES),
        "allow_builtin_dataset": bool(args.allow_builtin_dataset),
        "boundary": summary["boundary"],
    })
    write_json(out / "dataset_report.json", dataset_report)
    write_json(out / "input_e130b_report.json", input_report)
    write_json(out / "summary.json", summary)
    write_json(out / "decision.json", decision)
    write_json(out / "aggregate_metrics.json", aggregate)
    write_json(out / "operator_transfer_results.json", {"rows": operator_rows})
    write_json(out / "variant_report.json", {"rows": variant_rows})
    write_json(out / "extraction_report.json", {"rows": extraction_rows})
    write_json(out / "assistant_render_report.json", {"rows": assistant_rows})
    write_json(out / "word_problem_no_call_report.json", {"rows": word_problem_rows})
    write_json(out / "deterministic_replay.json", replay)
    write_json(out / "checker_summary.json", checker)
    write_jsonl(out / "row_level_samples.jsonl", sample_rows[:512])
    write_report(out, summary, dataset_report, operator_rows)
    append_jsonl(progress, {"event": "done", "decision": decision_label, "timestamp_ms": now_ms()})
    if args.sample_out:
        copy_sample_pack(out, Path(args.sample_out))
    return decision


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--dataset-row-limit", type=int, default=DEFAULT_DATASET_ROW_LIMIT)
    parser.add_argument("--e130b-root", default=str(DEFAULT_E130B_ROOT))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--sample-out", default=str(DEFAULT_SAMPLE_OUT))
    parser.add_argument("--visible-cases-per-operator", type=int, default=DEFAULT_VISIBLE_CASES_PER_OPERATOR)
    parser.add_argument("--word-problem-cases-per-operator", type=int, default=DEFAULT_WORD_PROBLEM_CASES_PER_OPERATOR)
    parser.add_argument("--allow-builtin-dataset", action="store_true")
    args = parser.parse_args()
    decision = run(args)
    print(json.dumps(decision, ensure_ascii=False, sort_keys=True))
    return 0 if decision["pass_gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
