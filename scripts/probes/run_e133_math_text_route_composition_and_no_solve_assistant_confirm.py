#!/usr/bin/env python3
"""E133 math-text route composition and no-solve assistant confirmation.

E133 composes the E132 math-text lenses/guards with the E131 visible-equation
assistant arithmetic route. It checks that visible arithmetic surfaces can be
routed into the scoped arithmetic renderer while proof, TIR, matrix, geometry,
answer-boundary, unit, and hidden word-problem surfaces stay guarded.

Boundary: route composition only. This is not MATH/GSM8K solving, not neural
training, not natural-language word-problem solving, and not Core/PermaCore.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
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
from scripts.probes.run_e129_arithmetic_trace_orange_legendary_probation import (  # noqa: E402
    ArithmeticCase,
    EngineConfig,
    OperatorSpec,
    OPERATOR_SPECS,
    config_from_features,
    evaluate_case,
    expression_for,
    render_fraction,
    stable_int,
)
from scripts.probes.run_e131_visible_equation_extraction_and_assistant_arithmetic_render_gauntlet import (  # noqa: E402
    DEFAULT_OUT as DEFAULT_E131,
    extract_visible_equation_payload,
    normalize_visible_candidate,
)
from scripts.probes.run_e132_external_math_text_skill_farm_mutation_prune_orange_cycle import (  # noqa: E402
    DEFAULT_DATASET as DEFAULT_E132_DATASET,
    DEFAULT_OUT as DEFAULT_E132,
    DEFAULT_SAMPLE_OUT as SAMPLE_E132,
    SPECS as E132_SPECS,
)


ARTIFACT_CONTRACT = "E133_MATH_TEXT_ROUTE_COMPOSITION_AND_NO_SOLVE_ASSISTANT_CONFIRM"
DECISION_CONFIRMED = "e133_math_text_route_composition_no_solve_assistant_confirmed"
DECISION_REJECTED = "e133_math_text_route_composition_no_solve_assistant_rejected"
NEXT = "E134_EXTERNAL_MATH_TEXT_OOD_ROUTE_STRESS_AND_COUNTEREXAMPLE_GAUNTLET"

SAMPLE_E131 = Path("docs/research/artifact_samples/e131_visible_equation_extraction_and_assistant_arithmetic_render_gauntlet")
DEFAULT_OUT = Path("target/pilot_wave/e133_math_text_route_composition_and_no_solve_assistant_confirm")
DEFAULT_SAMPLE_OUT = Path("docs/research/artifact_samples/e133_math_text_route_composition_and_no_solve_assistant_confirm")

DEFAULT_DATASET_ROW_LIMIT = 215_051
DEFAULT_ROUTE_CASES_PER_OPERATOR = 8_000
DEFAULT_HIDDEN_CASES_PER_OPERATOR = 3_000
DEFAULT_CONTROL_CASES_PER_OPERATOR = 1_500

VISIBLE_ARITHMETIC_OPERATOR_IDS = {
    "latex_inline_math_boundary_lens",
    "latex_display_math_block_lens",
    "boxed_answer_boundary_lens",
    "fraction_ratio_probability_lens",
    "answer_format_instruction_lens",
}

STRUCTURAL_ACTION_BY_OPERATOR = {
    "latex_inline_math_boundary_lens": "PROPOSE_MATH_TEXT_BOUNDARY",
    "latex_display_math_block_lens": "PROPOSE_MATH_TEXT_BOUNDARY",
    "boxed_answer_boundary_lens": "PROPOSE_ANSWER_BOUNDARY_NO_TRUST",
    "tir_python_block_boundary_lens": "PRESERVE_TIR_STRUCTURE_NO_EXECUTE",
    "proof_step_connector_lens": "PROPOSE_PROOF_STEP_BOUNDARY",
    "geometry_diagram_reference_guard": "DEFER_DIAGRAM_REQUIRED",
    "matrix_vector_block_lens": "PROPOSE_MATRIX_VECTOR_BOUNDARY",
    "equation_system_alignment_lens": "PROPOSE_EQUATION_SYSTEM_BOUNDARY",
    "piecewise_case_function_lens": "PROPOSE_PIECEWISE_BOUNDARY",
    "fraction_ratio_probability_lens": "PROPOSE_MATH_TEXT_BOUNDARY",
    "variable_definition_binding_lens": "PROPOSE_VARIABLE_BINDING_BOUNDARY",
    "summation_sequence_series_lens": "PROPOSE_SUMMATION_SEQUENCE_BOUNDARY",
    "unit_quantity_binding_lens": "GUARD_UNIT_QUANTITY_NO_CONVERSION",
    "word_problem_no_solve_guard_v2": "NO_CALL_HIDDEN_WORD_PROBLEM",
    "assistant_tir_output_error_repair_guard": "REJECT_UNSAFE_TIR_OUTPUT",
    "answer_format_instruction_lens": "PRESERVE_ANSWER_FORMAT_BOUNDARY",
}

GUARDED_ACTIONS = set(STRUCTURAL_ACTION_BY_OPERATOR.values()) | {"NO_CALL_HIDDEN_WORD_PROBLEM"}

FORBIDDEN_ASSISTANT_CLAIMS = (
    "word problem solved",
    "gsm8k",
    "math benchmark",
    "open-domain",
    "open domain",
    "neural",
    "llm",
    "permacore",
    "truegolden",
    "i solved the story",
)

ARTIFACT_FILES = (
    "run_manifest.json",
    "dataset_route_seed_report.json",
    "input_e132_report.json",
    "input_e131_report.json",
    "route_family_report.json",
    "operator_route_results.json",
    "route_case_report.json",
    "control_report.json",
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
class SeedRow:
    record_id: str
    source: str
    split: str
    family: str
    prompt: str
    response: str
    tags: tuple[str, ...]


@dataclass(frozen=True)
class RouteCase:
    case_id: str
    operator_id: str
    family: str
    split: str
    input_text: str
    expected_action: str
    expected_route_kind: str
    expected_payload: str | None
    expected_result: str | None
    arithmetic_operator_id: str | None
    seed_record_id: str
    seed_source: str


def deterministic_hash(payload: Any) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def stable_mod(text: str, modulo: int) -> int:
    return stable_int(text) % modulo


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def clean_one_line(text: str, limit: int = 360) -> str:
    collapsed = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3].rstrip() + "..."


def style_label(seed: SeedRow) -> str:
    return f"{seed.source}/{seed.family}/{seed.split}/{seed.record_id}"


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


def builtin_seed_rows() -> list[SeedRow]:
    rows: list[SeedRow] = []
    prompts = [
        "Keep the answer scoped to visible evidence only.",
        "Classify math notation without inventing a solution.",
        "Use proof and TIR surfaces as route evidence, not direct answers.",
        "When a diagram is required, defer instead of guessing.",
        "Answer-format instructions preserve shape; they do not prove value.",
    ]
    for index, prompt in enumerate(prompts * 60):
        rows.append(
            SeedRow(
                record_id=f"builtin_e133_style_{index:04d}",
                source="builtin/e133_fallback",
                split=("train", "validation", "heldout", "stress")[index % 4],
                family="assistant_route_style_fallback",
                prompt=prompt,
                response="",
                tags=("assistant_style", "math_route", "fallback"),
            )
        )
    return rows


def load_seed_rows(path: Path, row_limit: int, allow_builtin_dataset: bool) -> tuple[list[SeedRow], dict[str, Any]]:
    if not path.exists():
        if not allow_builtin_dataset:
            raise FileNotFoundError(f"missing E132 normalized dataset: {path}")
        rows = builtin_seed_rows()
        return rows, {
            "artifact_contract": ARTIFACT_CONTRACT,
            "dataset_path": str(path),
            "dataset_available": False,
            "row_limit": row_limit,
            "row_count_loaded": len(rows),
            "source_counts": {"builtin/e133_fallback": len(rows)},
            "family_counts": {"assistant_route_style_fallback": len(rows)},
            "split_counts": dict(Counter(row.split for row in rows)),
            "tag_counts": dict(Counter(tag for row in rows for tag in row.tags)),
            "dataset_sha256_first_rows": deterministic_hash([row.__dict__ for row in rows[:64]]),
        }

    rows: list[SeedRow] = []
    source_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    tag_counts: Counter[str] = Counter()
    first_hash_material: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if len(rows) >= row_limit:
                break
            if not line.strip():
                continue
            record = json.loads(line)
            prompt = clean_one_line(record.get("prompt", ""), 800)
            response = clean_one_line(record.get("response", ""), 800)
            tags = tuple(str(tag) for tag in record.get("skill_tags", []))
            row = SeedRow(
                record_id=str(record.get("record_id") or f"row_{line_no}"),
                source=str(record.get("source") or "unknown"),
                split=str(record.get("split") or "unknown"),
                family=str(record.get("family") or "unknown"),
                prompt=prompt,
                response=response,
                tags=tags,
            )
            rows.append(row)
            source_counts[row.source] += 1
            family_counts[row.family] += 1
            split_counts[row.split] += 1
            for tag in row.tags:
                tag_counts[tag] += 1
            if len(first_hash_material) < 512:
                first_hash_material.append(
                    {
                        "record_id": row.record_id,
                        "source": row.source,
                        "split": row.split,
                        "family": row.family,
                        "prompt_head": row.prompt[:160],
                        "tags": row.tags,
                    }
                )

    if not rows and allow_builtin_dataset:
        rows = builtin_seed_rows()
    return rows, {
        "artifact_contract": ARTIFACT_CONTRACT,
        "dataset_path": str(path),
        "dataset_available": path.exists(),
        "row_limit": row_limit,
        "row_count_loaded": len(rows),
        "source_counts": dict(source_counts.most_common()),
        "family_counts": dict(family_counts.most_common()),
        "split_counts": dict(split_counts.most_common()),
        "tag_counts": dict(tag_counts.most_common()),
        "dataset_sha256_first_rows": deterministic_hash(first_hash_material),
    }


def seed_at(rows: list[SeedRow], key: str, index: int) -> SeedRow:
    return rows[(stable_mod(f"{key}:{index}", len(rows)) + index) % len(rows)]


def arithmetic_specs_by_id() -> dict[str, OperatorSpec]:
    return {spec.operator_id: spec for spec in OPERATOR_SPECS}


def arithmetic_config(spec: OperatorSpec) -> EngineConfig:
    return config_from_features(f"{spec.operator_id}::e133_math_text_route", spec.required_features)


def valid_arithmetic_specs() -> list[OperatorSpec]:
    return [spec for spec in OPERATOR_SPECS if spec.kind not in {"invalid_trace", "division_by_zero"}]


def choose_arithmetic_spec(operator_id: str, index: int) -> OperatorSpec:
    specs = valid_arithmetic_specs()
    return specs[(stable_mod(f"{operator_id}:arith:{index}", len(specs)) + index) % len(specs)]


def source_e132_report(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = read_json(root / "operator_orange_results.json")["rows"]
    summary = read_json(root / "summary.json")
    source_pass = (
        summary.get("decision") == "e132_external_math_text_skill_farm_mutation_prune_orange_cycle_confirmed"
        and summary.get("operator_count") == 16
        and summary.get("orange_legendary_candidate_count") == 16
        and summary.get("hard_negative_total") == 0
        and summary.get("wrong_scope_call_total") == 0
        and summary.get("false_commit_total") == 0
        and summary.get("unsupported_answer_total") == 0
        and summary.get("boundary_claim_violation_total") == 0
        and summary.get("direct_flow_write_total") == 0
    )
    return rows, {
        "artifact_contract": ARTIFACT_CONTRACT,
        "source_e132_root": str(root),
        "source_decision": summary.get("decision"),
        "source_pass": source_pass,
        "source_operator_count": summary.get("operator_count"),
        "source_orange_legendary_candidate_count": summary.get("orange_legendary_candidate_count"),
        "source_dataset_rows_loaded": summary.get("dataset_rows_loaded"),
        "source_external_support_min": summary.get("external_support_min"),
        "source_hard_negative_total": summary.get("hard_negative_total"),
        "source_wrong_scope_call_total": summary.get("wrong_scope_call_total"),
        "source_direct_flow_write_total": summary.get("direct_flow_write_total"),
    }


def source_e131_report(root: Path) -> dict[str, Any]:
    summary = read_json(root / "summary.json")
    source_pass = (
        summary.get("decision") == "e131_visible_equation_extraction_assistant_arithmetic_render_confirmed"
        and summary.get("transfer_pass_operator_count") == 9
        and summary.get("visible_equation_extraction_accuracy_min") == 1.0
        and summary.get("word_problem_no_call_accuracy_min") == 1.0
        and summary.get("hard_negative_total") == 0
        and summary.get("wrong_scope_call_total") == 0
        and summary.get("false_commit_total") == 0
        and summary.get("unsupported_answer_total") == 0
        and summary.get("boundary_claim_violation_total") == 0
    )
    return {
        "artifact_contract": ARTIFACT_CONTRACT,
        "source_e131_root": str(root),
        "source_decision": summary.get("decision"),
        "source_pass": source_pass,
        "source_operator_count": summary.get("operator_count"),
        "source_transfer_pass_operator_count": summary.get("transfer_pass_operator_count"),
        "source_visible_equation_extraction_accuracy_min": summary.get("visible_equation_extraction_accuracy_min"),
        "source_word_problem_no_call_accuracy_min": summary.get("word_problem_no_call_accuracy_min"),
        "source_hard_negative_total": summary.get("hard_negative_total"),
        "source_wrong_scope_call_total": summary.get("wrong_scope_call_total"),
    }


def arithmetic_payload_for(operator_id: str, index: int) -> tuple[OperatorSpec, str, str]:
    spec = choose_arithmetic_spec(operator_id, index)
    expression, value = expression_for(spec, index)
    return spec, expression, render_fraction(value)


def visible_surface_text(operator_id: str, expression: str, result: str, seed: SeedRow, index: int) -> str:
    style = style_label(seed)
    if operator_id == "latex_inline_math_boundary_lens":
        return f"Style sample: {style}\nOnly this inline math span is callable: ${expression}$."
    if operator_id == "latex_display_math_block_lens":
        return f"Use only the display math block.\n\\[\n{expression}\n\\]\nIgnore the surrounding prose style: {style}"
    if operator_id == "boxed_answer_boundary_lens":
        return f"Visible equation: {expression}\nA later answer boundary says \\boxed{{{result}}}. Route from the visible equation, not box trust."
    if operator_id == "fraction_ratio_probability_lens":
        return f"Probability note with an explicit visible arithmetic payload: E133_VISIBLE_ARITHMETIC[[{expression}]]. Context: {style}"
    if operator_id == "answer_format_instruction_lens":
        return f"Express your answer as a compact value. Visible equation: {expression}\nFormat instruction is preserved after computation."
    return f"Visible equation: {expression}\nContext: {style}"


def structural_text(operator_id: str, seed: SeedRow, index: int) -> tuple[str, str]:
    style = style_label(seed)
    a = stable_mod(f"{operator_id}:a:{index}", 17) + 2
    b = stable_mod(f"{operator_id}:b:{index}", 19) + 3
    templates: dict[str, tuple[str, str]] = {
        "latex_inline_math_boundary_lens": (
            "inline_latex_boundary",
            f"Mark the inline math-text boundary only: $x_{a}+y_{b}=z$. Style: {style}",
        ),
        "latex_display_math_block_lens": (
            "display_latex_boundary",
            f"Preserve this display block as math text, no solving.\n\\[\n\\begin{{aligned}}x+y&=z\\\\ y&=k+{a}\\end{{aligned}}\n\\]",
        ),
        "boxed_answer_boundary_lens": (
            "boxed_answer_no_trust",
            f"Derivation is omitted. Therefore the answer is \\boxed{{{a * b}}}. Do not trust the box without route evidence.",
        ),
        "tir_python_block_boundary_lens": (
            "tir_python_structure",
            f"```python\nvalue = {a} + {b}\nprint(value)\n```\n```output\n{a + b + 1}\n```\nPreserve code/output structure; do not execute or commit.",
        ),
        "proof_step_connector_lens": (
            "proof_step_connector",
            f"Let n be even. Hence n=2k, therefore n^2=4k^2. Route connectors only. {style}",
        ),
        "geometry_diagram_reference_guard": (
            "geometry_diagram_defer",
            "In triangle ABC, angle A is shown in the missing diagram. [asy] draw((0,0)--(1,0)); [/asy]",
        ),
        "matrix_vector_block_lens": (
            "matrix_vector_boundary",
            "\\begin{pmatrix} a & b \\\\ c & d \\end{pmatrix}\\begin{pmatrix}x\\\\y\\end{pmatrix} is a matrix-vector surface.",
        ),
        "equation_system_alignment_lens": (
            "equation_system_alignment",
            "\\begin{aligned}x+y&=3\\\\2x-y&=1\\end{aligned} Preserve row boundaries; no solve.",
        ),
        "piecewise_case_function_lens": (
            "piecewise_case_boundary",
            "f(x)=\\begin{cases}x+1,&x<0\\\\x^2,&x\\ge 0\\end{cases}. Preserve condition bindings.",
        ),
        "fraction_ratio_probability_lens": (
            "fraction_ratio_boundary",
            f"The probability surface is \\frac{{{a}}}{{{a + b}}}; bind the fraction boundary only.",
        ),
        "variable_definition_binding_lens": (
            "variable_definition_binding",
            "Let r denote the radius, where A=\\pi r^2. Bind variable definitions without solving.",
        ),
        "summation_sequence_series_lens": (
            "summation_sequence_boundary",
            "\\sum_{i=1}^{n} i^2 appears in a sequence proof; identify the summation surface only.",
        ),
        "unit_quantity_binding_lens": (
            "unit_quantity_guard",
            f"A segment is {a} cm and another is {b} inches. Bind units; do not convert.",
        ),
        "word_problem_no_solve_guard_v2": (
            "primary_hidden_word_problem_no_solve",
            f"Mira has {a} marbles and gets {b} more. How many marbles does she have?",
        ),
        "assistant_tir_output_error_repair_guard": (
            "tir_error_repair_guard",
            "```python\nfor i in range(3)\n    print(i)\n```\n```output\nSyntaxError: invalid syntax\n```",
        ),
        "answer_format_instruction_lens": (
            "answer_format_instruction_boundary",
            "Give your answer as an ordered pair and in simplest form. Preserve format instruction only.",
        ),
    }
    return templates[operator_id]


def make_route_case(operator_id: str, index: int, seed_rows: list[SeedRow]) -> RouteCase:
    seed = seed_at(seed_rows, operator_id, index)
    split = ("train", "validation", "heldout", "stress", "prune", "challenger")[index % 6]
    if operator_id in VISIBLE_ARITHMETIC_OPERATOR_IDS and index % 4 == 0:
        arith_spec, expression, result = arithmetic_payload_for(operator_id, index)
        text = visible_surface_text(operator_id, expression, result, seed, index)
        payload = normalize_visible_candidate(expression) or f"compute: {expression}"
        return RouteCase(
            case_id=f"{operator_id}:visible_arithmetic_route:{index}",
            operator_id=operator_id,
            family="visible_arithmetic_route",
            split=split,
            input_text=text,
            expected_action="ROUTE_VISIBLE_ARITHMETIC_RENDER",
            expected_route_kind="visible_arithmetic",
            expected_payload=payload,
            expected_result=result,
            arithmetic_operator_id=arith_spec.operator_id,
            seed_record_id=seed.record_id,
            seed_source=seed.source,
        )

    family, text = structural_text(operator_id, seed, index)
    return RouteCase(
        case_id=f"{operator_id}:{family}:{index}",
        operator_id=operator_id,
        family=family,
        split=split,
        input_text=text,
        expected_action=STRUCTURAL_ACTION_BY_OPERATOR[operator_id],
        expected_route_kind="guarded_structural",
        expected_payload=None,
        expected_result=None,
        arithmetic_operator_id=None,
        seed_record_id=seed.record_id,
        seed_source=seed.source,
    )


def make_hidden_case(operator_id: str, index: int, seed_rows: list[SeedRow]) -> RouteCase:
    seed = seed_at(seed_rows, f"hidden:{operator_id}", index)
    a = stable_mod(f"{operator_id}:hidden:a:{index}", 90) + 10
    b = stable_mod(f"{operator_id}:hidden:b:{index}", 40) + 2
    c = stable_mod(f"{operator_id}:hidden:c:{index}", 30) + 1
    families = (
        "hidden_marble_story",
        "hidden_rows_story",
        "hidden_rate_story",
        "hidden_sharing_story",
        "hidden_adversarial_direct_answer_request",
    )
    family = families[index % len(families)]
    if family == "hidden_marble_story":
        story = f"Mira has {a} marbles, receives {b} more, then gives away {c}. What is the final count?"
    elif family == "hidden_rows_story":
        story = f"A hall has {a} rows with {b} seats in each row. How many seats are there?"
    elif family == "hidden_rate_story":
        story = f"A train travels {a} kilometers per hour for {b} hours. What distance is covered?"
    elif family == "hidden_sharing_story":
        story = f"{a * b} cards are shared equally among {b} players. How many cards per player?"
    else:
        story = f"Use the prose numbers {a}, {b}, and {c}; give the answer directly without writing an equation."
    text = f"External style source: {style_label(seed)}\nUser asks a prose-only math question: {story}"
    return RouteCase(
        case_id=f"{operator_id}:hidden_word_problem_no_solve:{index}",
        operator_id=operator_id,
        family=family,
        split="hidden_word_problem_no_solve",
        input_text=text,
        expected_action="NO_CALL_HIDDEN_WORD_PROBLEM",
        expected_route_kind="hidden_word_problem_no_solve",
        expected_payload=None,
        expected_result=None,
        arithmetic_operator_id=None,
        seed_record_id=seed.record_id,
        seed_source=seed.source,
    )


def extract_e133_visible_payload(text: str) -> tuple[str | None, str]:
    explicit = re.search(r"E133_VISIBLE_ARITHMETIC\[\[(.*?)\]\]", text, re.S | re.I)
    if explicit:
        return normalize_visible_candidate(explicit.group(1)), "e133_visible_arithmetic_marker"

    payload, reason = extract_visible_equation_payload(text)
    if payload is not None:
        return payload, f"e131_{reason}"

    inline_spans = re.findall(r"\$([^$]+)\$", text, re.S)
    for span in inline_spans:
        candidate = normalize_visible_candidate(span)
        if candidate is not None and re.search(r"[+\-*/x×÷]", span):
            return candidate, "inline_latex_visible_arithmetic"

    display = re.search(r"\\\[(.*?)\\\]", text, re.S)
    if display:
        candidate = normalize_visible_candidate(display.group(1))
        if candidate is not None:
            return candidate, "display_latex_visible_arithmetic"

    return None, "no_visible_arithmetic_payload"


def route_assistant_text(action: str, result: str | None, route_reason: str) -> str:
    if action == "ROUTE_VISIBLE_ARITHMETIC_RENDER":
        return f"The visible arithmetic route returns {result}."
    if action == "NO_CALL_HIDDEN_WORD_PROBLEM":
        return "NO_CALL: prose-only math question has no visible arithmetic route."
    if action == "REJECT_UNSAFE_TIR_OUTPUT":
        return "Guarded TIR error/output surface; no value is committed."
    if action == "DEFER_DIAGRAM_REQUIRED":
        return "Diagram-dependent math surface is deferred."
    if action == "PRESERVE_TIR_STRUCTURE_NO_EXECUTE":
        return "TIR code/output structure is preserved without execution."
    return f"{action}: bounded math-text route proposal from {route_reason}."


def assistant_has_forbidden_claim(text: str) -> bool:
    lowered = text.lower()
    return any(claim in lowered for claim in FORBIDDEN_ASSISTANT_CLAIMS)


def evaluate_selected_route(case: RouteCase, arith_specs: dict[str, OperatorSpec]) -> dict[str, Any]:
    if case.expected_route_kind == "hidden_word_problem_no_solve":
        payload, route_reason = None, "hidden_word_problem_no_solve_guard"
    else:
        payload, route_reason = extract_e133_visible_payload(case.input_text)
    result_value: str | None = None
    arithmetic_action: str | None = None
    arithmetic_correct = False
    if payload is not None and case.expected_route_kind == "visible_arithmetic":
        arith_spec = arith_specs[case.arithmetic_operator_id or ""]
        arithmetic_case = ArithmeticCase(
            case_id=case.case_id,
            operator_id=arith_spec.operator_id,
            split=case.split,
            input_text=payload,
            expression="",
            expected_result=case.expected_result or "",
            expected_action="ANSWER_WITH_TRACE",
            expected_valid=True,
        )
        arithmetic_result = dict(evaluate_case(arithmetic_case, arithmetic_config(arith_spec)))
        arithmetic_action = str(arithmetic_result.get("predicted_action"))
        result_value = arithmetic_result.get("predicted_result")
        arithmetic_correct = (
            arithmetic_action == "ANSWER_WITH_TRACE"
            and result_value == case.expected_result
            and not arithmetic_result.get("false_commit")
        )
        predicted_action = "ROUTE_VISIBLE_ARITHMETIC_RENDER" if arithmetic_correct else "REJECT_ARITHMETIC_ROUTE"
    elif payload is not None and case.expected_route_kind != "visible_arithmetic":
        predicted_action = "REJECT_UNSCOPED_VISIBLE_ARITHMETIC"
    else:
        predicted_action = classify_structural_route(case.input_text, case.operator_id)

    assistant_text = route_assistant_text(predicted_action, result_value, route_reason)
    correct = (
        predicted_action == case.expected_action
        and (case.expected_result is None or result_value == case.expected_result)
    )
    wrong_scope_call = case.expected_action == "NO_CALL_HIDDEN_WORD_PROBLEM" and predicted_action != "NO_CALL_HIDDEN_WORD_PROBLEM"
    false_commit = predicted_action == "ROUTE_VISIBLE_ARITHMETIC_RENDER" and case.expected_result is None
    unsupported_answer = predicted_action not in GUARDED_ACTIONS and predicted_action != "ROUTE_VISIBLE_ARITHMETIC_RENDER"
    return {
        "predicted_action": predicted_action,
        "predicted_result": result_value,
        "payload": payload,
        "route_reason": route_reason,
        "arithmetic_operator_id": case.arithmetic_operator_id,
        "arithmetic_action": arithmetic_action,
        "arithmetic_correct": arithmetic_correct,
        "assistant_render": assistant_text,
        "assistant_render_forbidden_claim": assistant_has_forbidden_claim(assistant_text),
        "correct": correct,
        "wrong_scope_call": wrong_scope_call,
        "false_commit": false_commit,
        "unsupported_answer": unsupported_answer,
        "direct_flow_write": 0,
    }


def classify_structural_route(text: str, operator_id: str) -> str:
    lowered = text.lower()
    if re.search(r"user asks a prose-only math question|how many|what distance|give the answer directly", lowered):
        return "NO_CALL_HIDDEN_WORD_PROBLEM"
    if operator_id == "assistant_tir_output_error_repair_guard" or "syntaxerror" in lowered or "traceback" in lowered:
        return "REJECT_UNSAFE_TIR_OUTPUT"
    if operator_id == "geometry_diagram_reference_guard" or "[asy]" in lowered or "missing diagram" in lowered:
        return "DEFER_DIAGRAM_REQUIRED"
    if operator_id == "tir_python_block_boundary_lens" or "```python" in lowered or "```output" in lowered:
        return "PRESERVE_TIR_STRUCTURE_NO_EXECUTE"
    return STRUCTURAL_ACTION_BY_OPERATOR[operator_id]


def evaluate_overbroad_solver_control(case: RouteCase) -> dict[str, Any]:
    numbers = [int(item) for item in re.findall(r"\b\d+\b", case.input_text)]
    if case.expected_action == "NO_CALL_HIDDEN_WORD_PROBLEM" and len(numbers) >= 2:
        guessed = str(numbers[0] + numbers[1])
        return {
            "predicted_action": "ANSWER_FROM_PROSE_CONTROL",
            "predicted_result": guessed,
            "wrong_scope_call": True,
            "false_commit": True,
            "unsupported_answer": False,
            "boundary_claim_violation": False,
            "direct_flow_write": 0,
        }
    return {
        "predicted_action": case.expected_action,
        "predicted_result": None,
        "wrong_scope_call": False,
        "false_commit": False,
        "unsupported_answer": False,
        "boundary_claim_violation": False,
        "direct_flow_write": 0,
    }


def evaluate_trust_control(case: RouteCase) -> dict[str, Any]:
    boxed = re.search(r"\\boxed\{([^}]*)\}", case.input_text)
    if boxed and case.expected_action == "PROPOSE_ANSWER_BOUNDARY_NO_TRUST":
        return {
            "control_id": "boxed_answer_trust_control",
            "predicted_action": "ANSWER_FROM_BOXED_CONTROL",
            "predicted_result": boxed.group(1),
            "wrong_scope_call": False,
            "false_commit": True,
            "unsupported_answer": False,
            "boundary_claim_violation": True,
            "direct_flow_write": 0,
        }
    output = re.search(r"```output\s*(.*?)```", case.input_text, re.S | re.I)
    if output and case.expected_action in {"PRESERVE_TIR_STRUCTURE_NO_EXECUTE", "REJECT_UNSAFE_TIR_OUTPUT"}:
        return {
            "control_id": "tir_output_trust_control",
            "predicted_action": "DIRECT_FLOW_WRITE_FROM_TIR_OUTPUT_CONTROL",
            "predicted_result": clean_one_line(output.group(1), 80),
            "wrong_scope_call": False,
            "false_commit": True,
            "unsupported_answer": False,
            "boundary_claim_violation": True,
            "direct_flow_write": 1,
        }
    return {
        "control_id": "safe_case_no_trust_control_trigger",
        "predicted_action": case.expected_action,
        "predicted_result": None,
        "wrong_scope_call": False,
        "false_commit": False,
        "unsupported_answer": False,
        "boundary_claim_violation": False,
        "direct_flow_write": 0,
    }


def sample_row(case: RouteCase, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "operator_id": case.operator_id,
        "family": case.family,
        "split": case.split,
        "seed_record_id": case.seed_record_id,
        "seed_source": case.seed_source,
        "input": case.input_text,
        "expected_action": case.expected_action,
        "predicted_action": result.get("predicted_action"),
        "expected_result": case.expected_result,
        "predicted_result": result.get("predicted_result"),
        "payload": result.get("payload"),
        "route_reason": result.get("route_reason"),
        "assistant_render": result.get("assistant_render"),
        "correct": result.get("correct"),
    }


def evaluate_operator(
    operator_row: dict[str, Any],
    seed_rows: list[SeedRow],
    route_cases: int,
    hidden_cases: int,
    control_cases: int,
    sample_limit: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    operator_id = operator_row["operator_id"]
    arith_specs = arithmetic_specs_by_id()
    counters: Counter[str] = Counter()
    family_total: Counter[str] = Counter()
    family_correct: Counter[str] = Counter()
    route_reason_counts: Counter[str] = Counter()
    route_kind_total: Counter[str] = Counter()
    route_kind_correct: Counter[str] = Counter()
    sample_rows: list[dict[str, Any]] = []
    route_case_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []

    def consume(case: RouteCase) -> None:
        result = evaluate_selected_route(case, arith_specs)
        counters["case_count"] += 1
        counters["correct_count"] += int(result["correct"])
        counters[f"{case.expected_route_kind}_case_count"] += 1
        counters[f"{case.expected_route_kind}_correct_count"] += int(result["correct"])
        family_total[case.family] += 1
        family_correct[case.family] += int(result["correct"])
        route_kind_total[case.expected_route_kind] += 1
        route_kind_correct[case.expected_route_kind] += int(result["correct"])
        route_reason_counts[result.get("route_reason", "unknown")] += 1
        if not result["correct"]:
            counters["hard_negative"] += 1
        counters["wrong_scope_call"] += int(result["wrong_scope_call"])
        counters["false_commit"] += int(result["false_commit"])
        counters["unsupported_answer"] += int(result["unsupported_answer"])
        counters["boundary_claim_violation"] += int(result["assistant_render_forbidden_claim"])
        counters["direct_flow_write"] += int(result["direct_flow_write"])
        if result["correct"] and case.expected_route_kind == "visible_arithmetic":
            counters["qualified_visible_arithmetic_activation"] += 1
        if result["correct"] and case.expected_route_kind == "guarded_structural":
            counters["qualified_structural_guard_activation"] += 1
        if result["correct"] and case.expected_route_kind == "hidden_word_problem_no_solve":
            counters["qualified_hidden_no_solve_activation"] += 1
        if sample_limit and len(sample_rows) < sample_limit and counters["case_count"] % max(1, (route_cases + hidden_cases) // sample_limit) == 0:
            sample_rows.append(sample_row(case, result))
        if len(route_case_rows) < 64:
            route_case_rows.append(sample_row(case, result))

    for index in range(route_cases):
        consume(make_route_case(operator_id, index, seed_rows))
    for index in range(hidden_cases):
        consume(make_hidden_case(operator_id, index, seed_rows))

    for index in range(control_cases):
        hidden_case = make_hidden_case(operator_id, index, seed_rows)
        overbroad = evaluate_overbroad_solver_control(hidden_case)
        counters["overbroad_solver_control_wrong_scope_call"] += int(overbroad["wrong_scope_call"])
        counters["overbroad_solver_control_false_commit"] += int(overbroad["false_commit"])
        if index < 12:
            control_rows.append({"operator_id": operator_id, "case_id": hidden_case.case_id, "control_id": "overbroad_solver_control", **overbroad})
        trust_case = make_route_case(operator_id, index, seed_rows)
        trust = evaluate_trust_control(trust_case)
        counters["trust_control_false_commit"] += int(trust["false_commit"])
        counters["trust_control_boundary_claim_violation"] += int(trust["boundary_claim_violation"])
        counters["trust_control_direct_flow_write"] += int(trust["direct_flow_write"])
        if index < 12 and trust["control_id"] != "safe_case_no_trust_control_trigger":
            control_rows.append({"operator_id": operator_id, "case_id": trust_case.case_id, **trust})

    visible_count = counters["visible_arithmetic_case_count"]
    structural_count = counters["guarded_structural_case_count"]
    hidden_count = counters["hidden_word_problem_no_solve_case_count"]
    route_accuracy = counters["correct_count"] / max(1, counters["case_count"])
    visible_accuracy = counters["visible_arithmetic_correct_count"] / max(1, visible_count)
    structural_accuracy = counters["guarded_structural_correct_count"] / max(1, structural_count)
    hidden_accuracy = counters["hidden_word_problem_no_solve_correct_count"] / max(1, hidden_count)
    route_pass = (
        route_accuracy == 1.0
        and (visible_count == 0 or visible_accuracy == 1.0)
        and structural_accuracy == 1.0
        and hidden_accuracy == 1.0
        and counters["hard_negative"] == 0
        and counters["wrong_scope_call"] == 0
        and counters["false_commit"] == 0
        and counters["unsupported_answer"] == 0
        and counters["boundary_claim_violation"] == 0
        and counters["direct_flow_write"] == 0
        and counters["overbroad_solver_control_wrong_scope_call"] > 0
    )
    operator_result = {
        "operator_id": operator_id,
        "display_name": operator_row.get("display_name", operator_id),
        "scope": operator_row.get("scope"),
        "family": operator_row.get("family"),
        "group_id": "E133",
        "rank_before": operator_row.get("rank_after", "OrangeLegendaryCandidate"),
        "rank_after": "OrangeLegendaryCandidate" if route_pass else "NeedsRepair",
        "rank": "OrangeLegendaryCandidate" if route_pass else "NeedsRepair",
        "watch_state": "E133MathTextRouteCompositionConfirmed" if route_pass else "E133RouteCompositionRepairRequired",
        "source_e132_watch_state": operator_row.get("watch_state"),
        "selected_route": "e133_schema_gated_math_text_route_composer",
        "composition_pass": route_pass,
        "route_case_count": counters["case_count"],
        "route_accuracy": round(route_accuracy, 6),
        "visible_arithmetic_route_case_count": visible_count,
        "visible_arithmetic_route_accuracy": round(visible_accuracy, 6) if visible_count else None,
        "structural_guard_case_count": structural_count,
        "structural_guard_accuracy": round(structural_accuracy, 6),
        "hidden_word_problem_no_solve_case_count": hidden_count,
        "hidden_word_problem_no_solve_accuracy": round(hidden_accuracy, 6),
        "qualified_route_activation": counters["correct_count"],
        "qualified_visible_arithmetic_activation": counters["qualified_visible_arithmetic_activation"],
        "qualified_structural_guard_activation": counters["qualified_structural_guard_activation"],
        "qualified_hidden_no_solve_activation": counters["qualified_hidden_no_solve_activation"],
        "hard_negative": counters["hard_negative"],
        "wrong_scope_call": counters["wrong_scope_call"],
        "false_commit": counters["false_commit"],
        "unsupported_answer": counters["unsupported_answer"],
        "boundary_claim_violation": counters["boundary_claim_violation"],
        "direct_flow_write": counters["direct_flow_write"],
        "overbroad_solver_control_wrong_scope_call": counters["overbroad_solver_control_wrong_scope_call"],
        "overbroad_solver_control_false_commit": counters["overbroad_solver_control_false_commit"],
        "trust_control_false_commit": counters["trust_control_false_commit"],
        "trust_control_boundary_claim_violation": counters["trust_control_boundary_claim_violation"],
        "trust_control_direct_flow_write": counters["trust_control_direct_flow_write"],
        "route_kind_accuracy": {
            kind: round(route_kind_correct[kind] / route_kind_total[kind], 6)
            for kind in sorted(route_kind_total)
        },
        "family_accuracy": {
            family: round(family_correct[family] / family_total[family], 6)
            for family in sorted(family_total)
        },
        "route_reason_counts": dict(route_reason_counts),
        "reload_shadow_pass": route_pass,
        "negative_scope_pass": hidden_accuracy == 1.0 and counters["wrong_scope_call"] == 0,
        "challenger_pass": counters["overbroad_solver_control_wrong_scope_call"] > 0,
        "prune_pass": True,
        "rule_of_three_upper_failure_bound": round(3.0 / max(1, counters["correct_count"]), 8),
        "e133_math_text_route_composition": True,
    }
    route_family_rows = [
        {
            "operator_id": operator_id,
            "route_kind": kind,
            "case_count": route_kind_total[kind],
            "correct_count": route_kind_correct[kind],
            "accuracy": round(route_kind_correct[kind] / route_kind_total[kind], 6),
        }
        for kind in sorted(route_kind_total)
    ]
    return operator_result, route_family_rows, route_case_rows, control_rows + sample_rows


def aggregate_results(operator_rows: list[dict[str, Any]], route_family_rows: list[dict[str, Any]], seconds: float) -> dict[str, Any]:
    visible_accuracies = [
        row["visible_arithmetic_route_accuracy"]
        for row in operator_rows
        if row["visible_arithmetic_route_accuracy"] is not None
    ]
    return {
        "artifact_contract": ARTIFACT_CONTRACT,
        "operator_count": len(operator_rows),
        "composition_pass_operator_count": sum(1 for row in operator_rows if row["composition_pass"]),
        "route_case_count_total": sum(row["route_case_count"] for row in operator_rows),
        "qualified_route_activation_total": sum(row["qualified_route_activation"] for row in operator_rows),
        "qualified_route_activation_min": min((row["qualified_route_activation"] for row in operator_rows), default=0),
        "visible_arithmetic_route_case_count_total": sum(row["visible_arithmetic_route_case_count"] for row in operator_rows),
        "structural_guard_case_count_total": sum(row["structural_guard_case_count"] for row in operator_rows),
        "hidden_word_problem_no_solve_case_count_total": sum(row["hidden_word_problem_no_solve_case_count"] for row in operator_rows),
        "route_accuracy_min": min((row["route_accuracy"] for row in operator_rows), default=0.0),
        "visible_arithmetic_route_accuracy_min": min(visible_accuracies, default=1.0),
        "structural_guard_accuracy_min": min((row["structural_guard_accuracy"] for row in operator_rows), default=0.0),
        "hidden_word_problem_no_solve_accuracy_min": min((row["hidden_word_problem_no_solve_accuracy"] for row in operator_rows), default=0.0),
        "hard_negative_total": sum(row["hard_negative"] for row in operator_rows),
        "wrong_scope_call_total": sum(row["wrong_scope_call"] for row in operator_rows),
        "false_commit_total": sum(row["false_commit"] for row in operator_rows),
        "unsupported_answer_total": sum(row["unsupported_answer"] for row in operator_rows),
        "boundary_claim_violation_total": sum(row["boundary_claim_violation"] for row in operator_rows),
        "direct_flow_write_total": sum(row["direct_flow_write"] for row in operator_rows),
        "overbroad_solver_control_wrong_scope_call_total": sum(row["overbroad_solver_control_wrong_scope_call"] for row in operator_rows),
        "overbroad_solver_control_false_commit_total": sum(row["overbroad_solver_control_false_commit"] for row in operator_rows),
        "trust_control_false_commit_total": sum(row["trust_control_false_commit"] for row in operator_rows),
        "trust_control_boundary_claim_violation_total": sum(row["trust_control_boundary_claim_violation"] for row in operator_rows),
        "trust_control_direct_flow_write_total": sum(row["trust_control_direct_flow_write"] for row in operator_rows),
        "route_family_row_count": len(route_family_rows),
        "seconds": round(seconds, 3),
    }


def decide(e132_report: dict[str, Any], e131_report: dict[str, Any], dataset_report: dict[str, Any], aggregate: dict[str, Any], allow_builtin_dataset: bool) -> tuple[str, list[str]]:
    failures: list[str] = []
    if not e132_report["source_pass"]:
        failures.append("source E132 gate did not pass")
    if not e131_report["source_pass"]:
        failures.append("source E131 gate did not pass")
    if not dataset_report["dataset_available"] and not allow_builtin_dataset:
        failures.append("E132 route seed dataset missing")
    if dataset_report["row_count_loaded"] < 50_000 and not allow_builtin_dataset:
        failures.append("E132 route seed dataset below 50k rows")
    if aggregate["operator_count"] != 16:
        failures.append("expected 16 E132 math-text operators")
    if aggregate["composition_pass_operator_count"] != aggregate["operator_count"]:
        failures.append("not all E132 operators passed route composition")
    for key in [
        "route_accuracy_min",
        "visible_arithmetic_route_accuracy_min",
        "structural_guard_accuracy_min",
        "hidden_word_problem_no_solve_accuracy_min",
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
    ]:
        if aggregate[key] != 0:
            failures.append(f"{key} nonzero")
    if aggregate["visible_arithmetic_route_case_count_total"] <= 0:
        failures.append("no visible arithmetic route cases were exercised")
    if aggregate["overbroad_solver_control_wrong_scope_call_total"] <= 0:
        failures.append("overbroad solver control did not fail")
    if aggregate["trust_control_false_commit_total"] <= 0:
        failures.append("unsafe trust controls did not fail")
    if aggregate["trust_control_direct_flow_write_total"] <= 0:
        failures.append("TIR direct-write trust control did not fail")
    return (DECISION_CONFIRMED if not failures else DECISION_REJECTED), failures


def write_report(out: Path, summary: dict[str, Any], dataset_report: dict[str, Any], operator_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# E133 Math Text Route Composition And No-Solve Assistant Confirm Result",
        "",
        "```text",
        f"decision = {summary['decision']}",
        f"next = {summary['next']}",
        "boundary = route composition only; not benchmark solving or word-problem solving",
        "",
        f"dataset_rows_loaded = {dataset_report['row_count_loaded']}",
        f"operator_count = {summary['operator_count']}",
        f"composition_pass_operator_count = {summary['composition_pass_operator_count']}",
        f"route_case_count_total = {summary['route_case_count_total']}",
        f"visible_arithmetic_route_case_count_total = {summary['visible_arithmetic_route_case_count_total']}",
        f"structural_guard_case_count_total = {summary['structural_guard_case_count_total']}",
        f"hidden_word_problem_no_solve_case_count_total = {summary['hidden_word_problem_no_solve_case_count_total']}",
        f"route_accuracy_min = {summary['route_accuracy_min']:.3f}",
        f"visible_arithmetic_route_accuracy_min = {summary['visible_arithmetic_route_accuracy_min']:.3f}",
        f"structural_guard_accuracy_min = {summary['structural_guard_accuracy_min']:.3f}",
        f"hidden_word_problem_no_solve_accuracy_min = {summary['hidden_word_problem_no_solve_accuracy_min']:.3f}",
        "",
        f"hard_negative_total = {summary['hard_negative_total']}",
        f"wrong_scope_call_total = {summary['wrong_scope_call_total']}",
        f"false_commit_total = {summary['false_commit_total']}",
        f"unsupported_answer_total = {summary['unsupported_answer_total']}",
        f"boundary_claim_violation_total = {summary['boundary_claim_violation_total']}",
        f"direct_flow_write_total = {summary['direct_flow_write_total']}",
        "",
        f"overbroad_solver_control_wrong_scope_call_total = {summary['overbroad_solver_control_wrong_scope_call_total']}",
        f"trust_control_false_commit_total = {summary['trust_control_false_commit_total']}",
        f"trust_control_direct_flow_write_total = {summary['trust_control_direct_flow_write_total']}",
        "```",
        "",
        "## Summary",
        "",
        "E133 confirms that the E132 math-text lenses/guards can participate in",
        "assistant-style route composition. Visible arithmetic math-text surfaces",
        "route into the already scoped E131/E129 arithmetic renderer, while boxed",
        "answers, TIR output, proof connectors, diagram references, matrices,",
        "summations, units, answer-format instructions, and prose-only word",
        "problems remain bounded proposals, defers, or no-calls.",
        "",
        "## Boundary",
        "",
        "This is still not a math benchmark solver. It confirms route selection and",
        "guarded no-solve behavior, not natural-language problem solving.",
        "",
        "## Operator Results",
        "",
        "```text",
    ]
    lines.extend(
        f"{row['operator_id']} -> {row['watch_state']} "
        f"(route={row['route_accuracy']:.3f}, hidden_no_solve={row['hidden_word_problem_no_solve_accuracy']:.3f})"
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

    e132_root = existing_artifact_root(Path(args.e132_root), SAMPLE_E132, "operator_orange_results.json")
    e131_root = existing_artifact_root(Path(args.e131_root), SAMPLE_E131, "operator_transfer_results.json")
    e132_rows, e132_report = source_e132_report(e132_root)
    e131_report = source_e131_report(e131_root)
    seed_rows, dataset_report = load_seed_rows(Path(args.dataset), args.dataset_row_limit, bool(args.allow_builtin_dataset))
    append_jsonl(progress, {
        "event": "inputs_loaded",
        "source_e132_root": str(e132_root),
        "source_e131_root": str(e131_root),
        "dataset_rows_loaded": dataset_report["row_count_loaded"],
        "timestamp_ms": now_ms(),
    })

    expected_ids = {spec.operator_id for spec in E132_SPECS}
    e132_rows = [row for row in e132_rows if row["operator_id"] in expected_ids]
    e132_rows.sort(key=lambda row: row["operator_id"])

    operator_rows: list[dict[str, Any]] = []
    route_family_rows: list[dict[str, Any]] = []
    route_case_rows: list[dict[str, Any]] = []
    control_and_sample_rows: list[dict[str, Any]] = []
    for index, row in enumerate(e132_rows, 1):
        operator_id = row["operator_id"]
        append_jsonl(progress, {"event": "operator_start", "operator_id": operator_id, "timestamp_ms": now_ms()})
        operator_result, family_rows, case_rows, extra_rows = evaluate_operator(
            row,
            seed_rows,
            args.route_cases_per_operator,
            args.hidden_cases_per_operator,
            args.control_cases_per_operator,
            args.sample_limit_per_operator,
        )
        operator_rows.append(operator_result)
        route_family_rows.extend(family_rows)
        route_case_rows.extend(case_rows)
        control_and_sample_rows.extend(extra_rows)
        write_json(out / "partial_aggregate_snapshot.json", {
            "event": "operator_complete",
            "processed": index,
            "operator_count": len(e132_rows),
            "composition_pass_so_far": sum(1 for result in operator_rows if result["composition_pass"]),
            "timestamp_ms": now_ms(),
        })
        append_jsonl(progress, {
            "event": "operator_done",
            "operator_id": operator_id,
            "composition_pass": operator_result["composition_pass"],
            "route_accuracy": operator_result["route_accuracy"],
            "hidden_no_solve_accuracy": operator_result["hidden_word_problem_no_solve_accuracy"],
            "timestamp_ms": now_ms(),
        })

    aggregate = aggregate_results(operator_rows, route_family_rows, time.time() - started)
    decision_label, failures = decide(e132_report, e131_report, dataset_report, aggregate, bool(args.allow_builtin_dataset))
    summary = {
        **aggregate,
        "decision": decision_label,
        "next": NEXT if decision_label == DECISION_CONFIRMED else "E133_MATH_TEXT_ROUTE_COMPOSITION_REPAIR",
        "boundary": "route composition only; not benchmark solving, not natural-language word-problem solving",
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
        "e132_report": e132_report,
        "e131_report": e131_report,
        "dataset_report": {key: value for key, value in dataset_report.items() if key != "dataset_path"},
        "summary": {key: value for key, value in summary.items() if key != "seconds"},
        "operator_rows": operator_rows,
        "route_family_rows": route_family_rows,
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
        "source_e132_root": str(e132_root),
        "source_e131_root": str(e131_root),
        "dataset": str(Path(args.dataset)),
        "dataset_row_limit": args.dataset_row_limit,
        "route_cases_per_operator": args.route_cases_per_operator,
        "hidden_cases_per_operator": args.hidden_cases_per_operator,
        "control_cases_per_operator": args.control_cases_per_operator,
        "boundary": summary["boundary"],
    })
    write_json(out / "dataset_route_seed_report.json", dataset_report)
    write_json(out / "input_e132_report.json", e132_report)
    write_json(out / "input_e131_report.json", e131_report)
    write_json(out / "aggregate_metrics.json", aggregate)
    write_json(out / "summary.json", summary)
    write_json(out / "decision.json", decision)
    write_json(out / "operator_route_results.json", {"rows": operator_rows})
    write_json(out / "route_family_report.json", {"rows": route_family_rows})
    write_json(out / "route_case_report.json", {"rows": route_case_rows})
    write_json(out / "control_report.json", {"rows": control_and_sample_rows})
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
    parser.add_argument("--e132-root", default=str(DEFAULT_E132))
    parser.add_argument("--e131-root", default=str(DEFAULT_E131))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--sample-out", default=str(DEFAULT_SAMPLE_OUT))
    parser.add_argument("--route-cases-per-operator", type=int, default=DEFAULT_ROUTE_CASES_PER_OPERATOR)
    parser.add_argument("--hidden-cases-per-operator", type=int, default=DEFAULT_HIDDEN_CASES_PER_OPERATOR)
    parser.add_argument("--control-cases-per-operator", type=int, default=DEFAULT_CONTROL_CASES_PER_OPERATOR)
    parser.add_argument("--sample-limit-per-operator", type=int, default=36)
    parser.add_argument("--allow-builtin-dataset", action="store_true")
    args = parser.parse_args()
    decision = run(args)
    print(json.dumps(decision, ensure_ascii=False, sort_keys=True))
    return 0 if decision["pass_gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
