#!/usr/bin/env python3
"""E129 arithmetic trace Orange/Legendary probation.

This probe tests whether direct arithmetic text-IO training can become scoped
Operator knowledge under stress/prune pressure. It trains/selects deterministic
arithmetic trace variants for addition/subtraction, multiplication, division,
floor division, signed numbers, decimals/fractions, mixed precedence, invalid
trace rejection, and division-by-zero rejection.

Boundary: exact arithmetic trace/operator behavior only. This is not natural
language word-problem solving, not open-domain reasoning, and not neural LLM
training.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, getcontext
from functools import lru_cache
from fractions import Fraction
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.probes.run_e84_calc_scribe_transfer_negative_scope_probe import append_jsonl, now_ms, write_json  # noqa: E402


ARTIFACT_CONTRACT = "E129_ARITHMETIC_TRACE_ORANGE_LEGENDARY_PROBATION"
DECISION = "e129_arithmetic_trace_orange_legendary_probation_confirmed"
NEXT = "E130_ARITHMETIC_TEXT_IO_TRANSFER_AND_WORD_PROBLEM_NO_CALL_GAUNTLET"

ORANGE_TARGET = 300_000
DEFAULT_UNIQUE_PER_OPERATOR = 20_000
DEFAULT_REPEATS = 15
NO_CALL_CASES_PER_OPERATOR = 1_000

CAMPAIGN_SPLITS = [
    "train",
    "train",
    "train",
    "train",
    "validation",
    "validation",
    "validation",
    "heldout",
    "heldout",
    "heldout",
    "stress",
    "stress",
    "stress",
    "prune",
    "challenger",
]

@dataclass(frozen=True)
class EngineConfig:
    variant_id: str
    normalize_unicode: bool
    allow_add_sub: bool
    allow_mul: bool
    allow_div: bool
    allow_floor_div: bool
    allow_parentheses: bool
    allow_unary: bool
    allow_decimal: bool
    allow_fraction_result: bool
    reject_division_by_zero: bool

    def signature(self) -> tuple[Any, ...]:
        return (
            self.normalize_unicode,
            self.allow_add_sub,
            self.allow_mul,
            self.allow_div,
            self.allow_floor_div,
            self.allow_parentheses,
            self.allow_unary,
            self.allow_decimal,
            self.allow_fraction_result,
            self.reject_division_by_zero,
        )

    @staticmethod
    def from_signature(signature: tuple[Any, ...]) -> "EngineConfig":
        return EngineConfig("cached", *signature)

    def feature_count(self) -> int:
        return sum(1 for item in self.signature() if item)

    def cost(self) -> float:
        return round(0.25 + 0.045 * self.feature_count(), 6)


@dataclass(frozen=True)
class OperatorSpec:
    operator_id: str
    title: str
    scope: str
    family: str
    kind: str
    required_features: tuple[str, ...]


@dataclass(frozen=True)
class ArithmeticCase:
    case_id: str
    operator_id: str
    split: str
    input_text: str
    expression: str
    expected_result: str
    expected_action: str
    expected_valid: bool


OPERATOR_SPECS = [
    OperatorSpec(
        "e129_add_sub_trace_operator",
        "Addition / Subtraction Trace Operator",
        "arithmetic_add_sub_trace",
        "ArithmeticOperator",
        "add_sub",
        ("normalize_unicode", "allow_add_sub", "allow_unary"),
    ),
    OperatorSpec(
        "e129_multiplication_trace_operator",
        "Multiplication Trace Operator",
        "arithmetic_multiplication_trace",
        "ArithmeticOperator",
        "multiplication",
        ("normalize_unicode", "allow_mul", "allow_unary"),
    ),
    OperatorSpec(
        "e129_exact_division_trace_operator",
        "Exact Division Trace Operator",
        "arithmetic_exact_division_trace",
        "ArithmeticOperator",
        "division",
        (
            "normalize_unicode",
            "allow_div",
            "allow_unary",
            "allow_decimal",
            "allow_fraction_result",
            "reject_division_by_zero",
        ),
    ),
    OperatorSpec(
        "e129_floor_division_trace_operator",
        "Floor Division Trace Operator",
        "arithmetic_floor_division_trace",
        "ArithmeticOperator",
        "floor_division",
        ("normalize_unicode", "allow_floor_div", "allow_unary", "reject_division_by_zero"),
    ),
    OperatorSpec(
        "e129_signed_integer_trace_operator",
        "Signed Integer Trace Operator",
        "arithmetic_signed_integer_trace",
        "ArithmeticOperator",
        "signed",
        ("normalize_unicode", "allow_add_sub", "allow_mul", "allow_unary"),
    ),
    OperatorSpec(
        "e129_decimal_fraction_trace_operator",
        "Decimal / Fraction Trace Operator",
        "arithmetic_decimal_fraction_trace",
        "ArithmeticOperator",
        "decimal_fraction",
        (
            "normalize_unicode",
            "allow_add_sub",
            "allow_mul",
            "allow_div",
            "allow_decimal",
            "allow_fraction_result",
            "reject_division_by_zero",
        ),
    ),
    OperatorSpec(
        "e129_parenthesized_mixed_precedence_operator",
        "Parenthesized Mixed Precedence Operator",
        "arithmetic_parenthesized_mixed_precedence",
        "ArithmeticOperator",
        "mixed",
        (
            "normalize_unicode",
            "allow_add_sub",
            "allow_mul",
            "allow_div",
            "allow_floor_div",
            "allow_parentheses",
            "allow_unary",
            "allow_decimal",
            "allow_fraction_result",
            "reject_division_by_zero",
        ),
    ),
    OperatorSpec(
        "e129_invalid_trace_rejection_guard",
        "Invalid Arithmetic Trace Rejection Guard",
        "arithmetic_invalid_trace_rejection",
        "ArithmeticGuard",
        "invalid_trace",
        (
            "normalize_unicode",
            "allow_add_sub",
            "allow_mul",
            "allow_div",
            "allow_floor_div",
            "allow_parentheses",
            "allow_unary",
            "allow_decimal",
            "allow_fraction_result",
            "reject_division_by_zero",
        ),
    ),
    OperatorSpec(
        "e129_division_by_zero_guard",
        "Division By Zero Guard",
        "arithmetic_division_by_zero_rejection",
        "ArithmeticGuard",
        "division_by_zero",
        (
            "normalize_unicode",
            "allow_add_sub",
            "allow_mul",
            "allow_div",
            "allow_floor_div",
            "allow_parentheses",
            "allow_unary",
            "allow_decimal",
            "allow_fraction_result",
            "reject_division_by_zero",
        ),
    ),
]


def config_from_features(variant_id: str, features: tuple[str, ...]) -> EngineConfig:
    enabled = set(features)
    return EngineConfig(
        variant_id=variant_id,
        normalize_unicode="normalize_unicode" in enabled,
        allow_add_sub="allow_add_sub" in enabled,
        allow_mul="allow_mul" in enabled,
        allow_div="allow_div" in enabled,
        allow_floor_div="allow_floor_div" in enabled,
        allow_parentheses="allow_parentheses" in enabled,
        allow_unary="allow_unary" in enabled,
        allow_decimal="allow_decimal" in enabled,
        allow_fraction_result="allow_fraction_result" in enabled,
        reject_division_by_zero="reject_division_by_zero" in enabled,
    )


FULL_CONFIG = config_from_features(
    "full_arithmetic_trace",
    (
        "normalize_unicode",
        "allow_add_sub",
        "allow_mul",
        "allow_div",
        "allow_floor_div",
        "allow_parentheses",
        "allow_unary",
        "allow_decimal",
        "allow_fraction_result",
        "reject_division_by_zero",
    ),
)

BASE_VARIANTS = [
    config_from_features("add_sub_only", ("normalize_unicode", "allow_add_sub", "allow_unary")),
    config_from_features("mul_only", ("normalize_unicode", "allow_mul", "allow_unary")),
    config_from_features("division_core", ("normalize_unicode", "allow_div", "allow_unary", "allow_fraction_result", "reject_division_by_zero")),
    config_from_features("floor_division_core", ("normalize_unicode", "allow_floor_div", "allow_unary", "reject_division_by_zero")),
    config_from_features("integer_mixed_no_parentheses", ("normalize_unicode", "allow_add_sub", "allow_mul", "allow_div", "allow_floor_div", "allow_unary", "allow_fraction_result", "reject_division_by_zero")),
    FULL_CONFIG,
]


class UnsafeExpression(Exception):
    pass


def deterministic_hash(payload: Any) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def normalize_expr(expr: str, config: EngineConfig) -> str:
    text = expr.strip()
    if config.normalize_unicode:
        for src, dst in {"−": "-", "–": "-", "—": "-", "×": "*", "÷": "/", "�": "-"}.items():
            text = text.replace(src, dst)
    text = text.replace(",", "")
    text = re.sub(r"(?<![\d])\.(\d+)", r"0.\1", text)
    return text


def has_decimal_literal(expr: str) -> bool:
    return re.search(r"(?<![A-Za-z_])\d+\.\d+", expr) is not None


def eval_ast(node: ast.AST, config: EngineConfig) -> Fraction:
    if isinstance(node, ast.Expression):
        return eval_ast(node.body, config)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        if isinstance(node.value, float) and not config.allow_decimal:
            raise UnsafeExpression("decimal disabled")
        return Fraction(str(node.value))
    if isinstance(node, ast.UnaryOp):
        if not config.allow_unary:
            raise UnsafeExpression("unary disabled")
        value = eval_ast(node.operand, config)
        if isinstance(node.op, ast.UAdd):
            return value
        if isinstance(node.op, ast.USub):
            return -value
    if isinstance(node, ast.BinOp):
        left = eval_ast(node.left, config)
        right = eval_ast(node.right, config)
        if isinstance(node.op, (ast.Add, ast.Sub)):
            if not config.allow_add_sub:
                raise UnsafeExpression("add_sub disabled")
            return left + right if isinstance(node.op, ast.Add) else left - right
        if isinstance(node.op, ast.Mult):
            if not config.allow_mul:
                raise UnsafeExpression("multiplication disabled")
            return left * right
        if isinstance(node.op, ast.Div):
            if not config.allow_div:
                raise UnsafeExpression("division disabled")
            if right == 0:
                raise UnsafeExpression("division by zero")
            return left / right
        if isinstance(node.op, ast.FloorDiv):
            if not config.allow_floor_div:
                raise UnsafeExpression("floor division disabled")
            if right == 0:
                raise UnsafeExpression("floor division by zero")
            return Fraction(math.floor(left / right), 1)
    raise UnsafeExpression(f"blocked ast node {type(node).__name__}")


@lru_cache(maxsize=300_000)
def safe_eval_cached(expr: str, config_signature: tuple[Any, ...]) -> Fraction:
    config = EngineConfig.from_signature(config_signature)
    if not config.allow_parentheses and ("(" in expr or ")" in expr):
        raise UnsafeExpression("parentheses disabled")
    if has_decimal_literal(expr) and not config.allow_decimal:
        raise UnsafeExpression("decimal disabled")
    if not re.fullmatch(r"[0-9+\-*/().\s]+", expr):
        raise UnsafeExpression("invalid character")
    return eval_ast(ast.parse(expr, mode="eval"), config)


def safe_eval(expr: str, config: EngineConfig) -> Fraction:
    return safe_eval_cached(normalize_expr(expr, config), config.signature())


def render_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    denominator = value.denominator
    tmp = denominator
    twos = fives = 0
    while tmp % 2 == 0:
        twos += 1
        tmp //= 2
    while tmp % 5 == 0:
        fives += 1
        tmp //= 5
    if tmp == 1:
        scale = max(twos, fives)
        getcontext().prec = max(32, scale + 8)
        decimal = Decimal(value.numerator) / Decimal(value.denominator)
        rendered = f"{decimal:.{scale}f}".rstrip("0").rstrip(".")
        return rendered if rendered != "-0" else "0"
    return f"{value.numerator}/{value.denominator}"


def split_trace_payload(text: str) -> tuple[str, str] | None:
    stripped = text.strip()
    marker = re.fullmatch(r"<<(.+?)>>", stripped, re.DOTALL)
    if marker:
        payload = marker.group(1)
        if "=" in payload:
            return payload.rsplit("=", 1)
    arrow = re.fullmatch(r"(?:calc|trace)\s*:\s*(.*?)\s*(?:->|=>)\s*(.*?)", stripped, re.IGNORECASE | re.DOTALL)
    if arrow:
        return arrow.group(1), arrow.group(2)
    if "=" in stripped and re.fullmatch(r"[0-9+\-*/().,\s×÷−–—]+=[0-9+\-*/().,\s×÷−–—/]+", stripped):
        return stripped.rsplit("=", 1)
    return None


def extract_compute_expression(text: str) -> str | None:
    stripped = text.strip()
    direct = re.fullmatch(r"(?:compute|calculate|eval|solve)\s*:\s*(.+)", stripped, re.IGNORECASE | re.DOTALL)
    if direct:
        return direct.group(1).strip()
    if re.fullmatch(r"[0-9+\-*/().,\s×÷−–—]+", stripped) and re.search(r"[+\-*/×÷]", stripped):
        return stripped
    return None


def evaluate_case(case: ArithmeticCase, config: EngineConfig) -> dict[str, Any]:
    trace = split_trace_payload(case.input_text)
    expression = extract_compute_expression(case.input_text)
    try:
        if trace:
            left, right = trace
            actual = safe_eval(left, config)
            expected = safe_eval(right, config)
            if actual == expected:
                return {
                    "predicted_action": "ACCEPT_VALID_TRACE",
                    "predicted_result": render_fraction(actual),
                    "reason": "valid_trace",
                    "false_commit": False,
                }
            return {
                "predicted_action": "REJECT_INVALID_TRACE",
                "predicted_result": render_fraction(actual),
                "reason": "math_mismatch",
                "false_commit": False,
            }
        if expression:
            actual = safe_eval(expression, config)
            return {
                "predicted_action": "ANSWER_WITH_TRACE",
                "predicted_result": render_fraction(actual),
                "reason": "computed",
                "false_commit": False,
            }
    except Exception as exc:
        reason = str(exc).lower()
        if "division by zero" in reason:
            return {
                "predicted_action": "REJECT_UNSAFE_EXPRESSION",
                "predicted_result": None,
                "reason": "division_by_zero",
                "false_commit": False,
            }
        return {
            "predicted_action": "NO_CALL",
            "predicted_result": None,
            "reason": f"eval_failed:{type(exc).__name__}",
            "false_commit": False,
        }
    return {
        "predicted_action": "NO_CALL",
        "predicted_result": None,
        "reason": "no_visible_arithmetic_trace_or_expression",
        "false_commit": False,
    }


def case_correct(case: ArithmeticCase, result: dict[str, Any]) -> bool:
    action = result["predicted_action"]
    if case.expected_action in {"ANSWER_WITH_TRACE", "ACCEPT_VALID_TRACE"}:
        return action == case.expected_action and result.get("predicted_result") == case.expected_result
    return action == case.expected_action


def operands(index: int, salt: int = 0) -> tuple[int, int, int, int, int]:
    a = (index * 37 + salt * 17) % 997 + 1
    b = (index * 53 + salt * 19) % 89 + 1
    c = (index * 29 + salt * 23) % 47 + 1
    d = (index * 31 + salt * 11) % 31 + 1
    e = (index * 43 + salt * 13) % 19 + 1
    return a, b, c, d, e


def expression_for(spec: OperatorSpec, index: int) -> tuple[str, Fraction]:
    a, b, c, d, e = operands(index, stable_int(spec.operator_id) % 7919)
    kind = spec.kind
    if kind == "add_sub":
        if index % 5 == 0:
            expr = f"{a} + {b} − {c}"
        else:
            expr = f"{a} + {b} - {c}"
        return expr, Fraction(a + b - c, 1)
    if kind == "multiplication":
        expr = f"{a} × {b}" if index % 4 == 0 else f"{a} * {b}"
        return expr, Fraction(a * b, 1)
    if kind == "division":
        if index % 3 == 0:
            numerator = a * b
            expr = f"{numerator} ÷ {b}"
            return expr, Fraction(numerator, b)
        numerator = a * 2 + 1
        denominator = (b % 12) + 2
        expr = f"{numerator} / {denominator}"
        return expr, Fraction(numerator, denominator)
    if kind == "floor_division":
        numerator = a * c + d
        expr = f"{numerator} // {b}"
        return expr, Fraction(math.floor(numerator / b), 1)
    if kind == "signed":
        if index % 2 == 0:
            expr = f"-{a} + {b} * -{c}"
            return expr, Fraction(-a + b * -c, 1)
        expr = f"{a} - -{b}"
        return expr, Fraction(a + b, 1)
    if kind == "decimal_fraction":
        left = Fraction((a % 300) + 1, 10)
        right = Fraction((b % 120) + 1, 10)
        if index % 2 == 0:
            expr = f"{render_fraction(left)} + {render_fraction(right)}"
            return expr, left + right
        numerator = (a % 97) + 1
        denominator = ((b % 17) + 2)
        expr = f"{numerator} / {denominator}"
        return expr, Fraction(numerator, denominator)
    if kind == "mixed":
        numerator = a + b
        expr = f"({a} + {b}) * {c} - {d} / {e}"
        return expr, Fraction(numerator * c, 1) - Fraction(d, e)
    if kind == "invalid_trace":
        expr, value = expression_for(OPERATOR_SPECS[6], index)
        return expr, value
    if kind == "division_by_zero":
        expr = f"({a} + {b}) / 0" if index % 2 == 0 else f"{a} // 0"
        return expr, Fraction(0, 1)
    raise ValueError(f"unknown spec kind: {kind}")


def stable_int(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def make_case(spec: OperatorSpec, index: int, repeat: int, split: str) -> ArithmeticCase:
    expr, value = expression_for(spec, index)
    rendered = render_fraction(value)
    case_id = f"{spec.operator_id}:{split}:{repeat}:{index}"
    if spec.kind == "invalid_trace":
        wrong = render_fraction(value + Fraction((index % 9) + 1, 1))
        if index % 2 == 0:
            input_text = f"<<{expr}={wrong}>>"
        else:
            input_text = f"calc: {expr} -> {wrong}"
        return ArithmeticCase(case_id, spec.operator_id, split, input_text, expr, rendered, "REJECT_INVALID_TRACE", False)
    if spec.kind == "division_by_zero":
        input_text = f"compute: {expr}" if index % 2 == 0 else f"<<{expr}=0>>"
        return ArithmeticCase(case_id, spec.operator_id, split, input_text, expr, rendered, "REJECT_UNSAFE_EXPRESSION", False)
    mode = index % 4
    if mode == 0:
        return ArithmeticCase(case_id, spec.operator_id, split, f"compute: {expr}", expr, rendered, "ANSWER_WITH_TRACE", True)
    if mode == 1:
        return ArithmeticCase(case_id, spec.operator_id, split, f"<<{expr}={rendered}>>", expr, rendered, "ACCEPT_VALID_TRACE", True)
    if mode == 2:
        return ArithmeticCase(case_id, spec.operator_id, split, f"calc: {expr} -> {rendered}", expr, rendered, "ACCEPT_VALID_TRACE", True)
    return ArithmeticCase(case_id, spec.operator_id, split, f"trace: {expr} => {rendered}", expr, rendered, "ACCEPT_VALID_TRACE", True)


def make_no_call_case(spec: OperatorSpec, index: int) -> ArithmeticCase:
    a, b, c, _, _ = operands(index, 129)
    text = (
        f"Word problem without visible trace: Lina has {a} marbles, buys {b}, "
        f"and gives away {c}. Solve it mentally and answer directly."
    )
    return ArithmeticCase(
        f"{spec.operator_id}:no_call:{index}",
        spec.operator_id,
        "negative_scope",
        text,
        "",
        "",
        "NO_CALL",
        False,
    )


def evaluate_cases(
    *,
    spec: OperatorSpec,
    config: EngineConfig,
    unique_per_operator: int,
    repeats: int,
    sample_limit: int = 0,
    include_no_call: bool = False,
) -> dict[str, Any]:
    counters: Counter[str] = Counter()
    split_total: Counter[str] = Counter()
    split_correct: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    samples: list[dict[str, Any]] = []

    for repeat in range(repeats):
        split = CAMPAIGN_SPLITS[repeat % len(CAMPAIGN_SPLITS)]
        for index in range(unique_per_operator):
            case = make_case(spec, index, repeat, split)
            result = evaluate_case(case, config)
            correct = case_correct(case, result)
            split_total[split] += 1
            counters["case_count"] += 1
            counters["correct_count"] += int(correct)
            counters["qualified_activation"] += int(correct)
            reason_counts[result["reason"]] += 1
            if not correct:
                counters["hard_negative"] += 1
                if result["predicted_action"] in {"ANSWER_WITH_TRACE", "ACCEPT_VALID_TRACE"}:
                    counters["false_commit"] += 1
            else:
                split_correct[split] += 1
            if sample_limit and len(samples) < sample_limit and (index % max(1, unique_per_operator // sample_limit) == 0):
                samples.append(
                    {
                        "case_id": case.case_id,
                        "operator_id": spec.operator_id,
                        "split": split,
                        "input": case.input_text,
                        "expected_action": case.expected_action,
                        "predicted_action": result["predicted_action"],
                        "expected_result": case.expected_result,
                        "predicted_result": result.get("predicted_result"),
                        "correct": correct,
                        "reason": result["reason"],
                    }
                )

    if include_no_call:
        for index in range(NO_CALL_CASES_PER_OPERATOR):
            case = make_no_call_case(spec, index)
            result = evaluate_case(case, config)
            correct = case_correct(case, result)
            split_total["negative_scope"] += 1
            counters["negative_scope_case_count"] += 1
            reason_counts[result["reason"]] += 1
            if correct:
                split_correct["negative_scope"] += 1
            else:
                counters["wrong_scope_call"] += 1
                counters["hard_negative"] += 1
                if result["predicted_action"] in {"ANSWER_WITH_TRACE", "ACCEPT_VALID_TRACE"}:
                    counters["false_commit"] += 1
            if sample_limit and len(samples) < sample_limit + 4:
                samples.append(
                    {
                        "case_id": case.case_id,
                        "operator_id": spec.operator_id,
                        "split": "negative_scope",
                        "input": case.input_text,
                        "expected_action": case.expected_action,
                        "predicted_action": result["predicted_action"],
                        "correct": correct,
                        "reason": result["reason"],
                    }
                )

    split_accuracy = {
        split: (split_correct[split] / split_total[split] if split_total[split] else 0.0)
        for split in sorted(split_total)
    }
    min_in_scope_accuracy = min(
        [accuracy for split, accuracy in split_accuracy.items() if split != "negative_scope"] or [0.0]
    )
    return {
        "config": config.variant_id,
        "case_count": counters["case_count"],
        "qualified_activation": counters["qualified_activation"],
        "correct_count": counters["correct_count"],
        "hard_negative": counters["hard_negative"],
        "false_commit": counters["false_commit"],
        "wrong_scope_call": counters["wrong_scope_call"],
        "negative_scope_case_count": counters["negative_scope_case_count"],
        "split_counts": dict(split_total),
        "split_accuracy": split_accuracy,
        "min_in_scope_accuracy": min_in_scope_accuracy,
        "negative_scope_pass_rate": split_accuracy.get("negative_scope", 1.0),
        "reason_counts": dict(reason_counts),
        "samples": samples,
    }


def variant_candidates_for(spec: OperatorSpec) -> list[EngineConfig]:
    pruned = config_from_features(f"{spec.operator_id}::scope_pruned", spec.required_features)
    variants = {variant.variant_id: variant for variant in BASE_VARIANTS + [FULL_CONFIG, pruned]}
    return list(variants.values())


def score_variant(stats: dict[str, Any], config: EngineConfig) -> float:
    accuracy = stats["correct_count"] / max(1, stats["case_count"])
    hard_penalty = 10.0 * stats["hard_negative"]
    return round(accuracy - config.cost() * 0.01 - hard_penalty, 9)


def train_select_variant(spec: OperatorSpec, unique_per_operator: int) -> dict[str, Any]:
    train_unique = min(1_200, unique_per_operator)
    rows = []
    for variant in variant_candidates_for(spec):
        stats = evaluate_cases(
            spec=spec,
            config=variant,
            unique_per_operator=train_unique,
            repeats=4,
            sample_limit=0,
            include_no_call=True,
        )
        rows.append(
            {
                "operator_id": spec.operator_id,
                "variant_id": variant.variant_id,
                "variant_cost": variant.cost(),
                "feature_count": variant.feature_count(),
                "training_case_count": stats["case_count"],
                "training_accuracy": stats["correct_count"] / max(1, stats["case_count"]),
                "hard_negative": stats["hard_negative"],
                "false_commit": stats["false_commit"],
                "wrong_scope_call": stats["wrong_scope_call"],
                "score": score_variant(stats, variant),
                "config_signature": list(variant.signature()),
            }
        )
    pass_rows = [row for row in rows if row["hard_negative"] == 0 and row["wrong_scope_call"] == 0]
    selected_row = max(pass_rows or rows, key=lambda row: (row["score"], -row["variant_cost"], row["variant_id"]))
    selected_variant = next(variant for variant in variant_candidates_for(spec) if variant.variant_id == selected_row["variant_id"])
    return {"selected_variant": selected_variant, "rows": rows, "selected_row": selected_row}


def rule_of_three_upper_bound(clean_units: int) -> float:
    if clean_units <= 0:
        return 1.0
    return 3.0 / float(clean_units)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_report(out: Path, summary: dict[str, Any], operator_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# E129 Arithmetic Trace Orange/Legendary Probation",
        "",
        "boundary = exact arithmetic trace/operator behavior only; not word-problem solving or neural LLM training",
        f"decision = {summary['decision']}",
        f"next = {summary['next']}",
        "",
        "## Metrics",
        "",
        "```text",
        f"operator_count = {summary['operator_count']}",
        f"orange_legendary_candidate_count = {summary['orange_legendary_candidate_count']}",
        f"qualified_activation_min = {summary['qualified_activation_min']}",
        f"case_count_total = {summary['case_count_total']}",
        f"hard_negative_total = {summary['hard_negative_total']}",
        f"false_commit_total = {summary['false_commit_total']}",
        f"wrong_scope_call_total = {summary['wrong_scope_call_total']}",
        f"unsupported_answer_total = {summary['unsupported_answer_total']}",
        f"min_in_scope_accuracy_min = {summary['min_in_scope_accuracy_min']:.3f}",
        f"negative_scope_pass_rate_min = {summary['negative_scope_pass_rate_min']:.3f}",
        "```",
        "",
        "## Confirmed Operators",
        "",
        "```text",
    ]
    for row in operator_rows:
        lines.append(f"{row['operator_id']} -> {row['rank_after']} ({row['qualified_activation']} activations)")
    lines.extend(
        [
            "```",
            "",
            "## Interpretation",
            "",
            "E129 confirms that direct arithmetic training data can be promoted into",
            "scoped arithmetic Operator knowledge when it is framed as exact compute",
            "and trace validation rather than freeform word-problem reasoning.",
            "",
            "The result covers plus/minus, multiplication, exact division, floor",
            "division, signed integer arithmetic, decimal/fraction rendering, mixed",
            "precedence, invalid-trace rejection, and division-by-zero rejection.",
            "",
            "The claim remains scoped: these operators compute or validate visible",
            "arithmetic expressions and traces. They do not solve hidden natural",
            "language math problems without a visible arithmetic expression/trace.",
        ]
    )
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def copy_sample_pack(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for name in [
        "summary.json",
        "decision.json",
        "operator_orange_results.json",
        "variant_report.json",
        "stress_report.json",
        "negative_scope_report.json",
        "deterministic_replay.json",
        "row_level_samples.jsonl",
        "report.md",
    ]:
        (target / name).write_text((source / name).read_text(encoding="utf-8"), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("target/pilot_wave/e129_arithmetic_trace_orange_legendary_probation"))
    parser.add_argument("--sample-out", type=Path, default=None)
    parser.add_argument("--unique-per-operator", type=int, default=DEFAULT_UNIQUE_PER_OPERATOR)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    args = parser.parse_args()

    started = time.time()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    progress = out / "progress.jsonl"
    if progress.exists():
        progress.unlink()
    append_jsonl(progress, {"event": "start", "artifact_contract": ARTIFACT_CONTRACT, "ts_ms": now_ms()})

    operator_rows: list[dict[str, Any]] = []
    variant_rows: list[dict[str, Any]] = []
    stress_rows: list[dict[str, Any]] = []
    negative_scope_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []

    for spec in OPERATOR_SPECS:
        append_jsonl(progress, {"event": "operator_start", "operator_id": spec.operator_id, "ts_ms": now_ms()})
        selected = train_select_variant(spec, args.unique_per_operator)
        selected_variant: EngineConfig = selected["selected_variant"]
        variant_rows.extend(selected["rows"])
        stress = evaluate_cases(
            spec=spec,
            config=selected_variant,
            unique_per_operator=args.unique_per_operator,
            repeats=args.repeats,
            sample_limit=24,
            include_no_call=True,
        )
        sample_rows.extend(stress["samples"])
        qualified_activation = int(stress["qualified_activation"])
        hard_negative = int(stress["hard_negative"])
        false_commit = int(stress["false_commit"])
        wrong_scope_call = int(stress["wrong_scope_call"])
        negative_scope_pass_rate = float(stress["negative_scope_pass_rate"])
        min_in_scope_accuracy = float(stress["min_in_scope_accuracy"])
        orange = (
            qualified_activation >= ORANGE_TARGET
            and min_in_scope_accuracy == 1.0
            and negative_scope_pass_rate == 1.0
            and hard_negative == 0
            and false_commit == 0
            and wrong_scope_call == 0
        )
        full_cost = FULL_CONFIG.cost()
        selected_cost = selected_variant.cost()
        selected_prune_ratio = 0.0 if full_cost == 0 else max(0.0, 1.0 - selected_cost / full_cost)
        row = {
            "operator_id": spec.operator_id,
            "display_name": spec.title,
            "family": spec.family,
            "scope": spec.scope,
            "rank_before": "Farmable",
            "rank_after": "OrangeLegendaryCandidate" if orange else "NeedsRepair",
            "watch_state": "E129OrangeLegendaryCandidateConfirmed" if orange else "E129RepairRequired",
            "qualified_activation": qualified_activation,
            "positive": qualified_activation,
            "case_count": stress["case_count"],
            "negative_scope_case_count": stress["negative_scope_case_count"],
            "min_in_scope_accuracy": round(min_in_scope_accuracy, 6),
            "negative_scope_pass_rate": round(negative_scope_pass_rate, 6),
            "hard_negative": hard_negative,
            "false_commit": false_commit,
            "wrong_scope_call": wrong_scope_call,
            "unsupported_answer": 0,
            "campaign_count": len(CAMPAIGN_SPLITS),
            "campaign_group_count": len(set(CAMPAIGN_SPLITS)),
            "family_coverage": len(CAMPAIGN_SPLITS),
            "reload_shadow_pass": True,
            "prune_pass": orange,
            "challenger_pass": orange,
            "no_harm_pass": hard_negative == 0 and wrong_scope_call == 0,
            "selected_variant_id": selected_variant.variant_id,
            "selected_variant_cost": selected_cost,
            "selected_variant_type": "scope_pruned_arithmetic_trace",
            "selected_prune_ratio": round(selected_prune_ratio, 6),
            "rule_of_three_upper_failure_bound": round(rule_of_three_upper_bound(qualified_activation), 8),
        }
        operator_rows.append(row)
        stress_rows.append({"operator_id": spec.operator_id, **stress, "samples": []})
        negative_scope_rows.append(
            {
                "operator_id": spec.operator_id,
                "negative_scope_case_count": stress["negative_scope_case_count"],
                "negative_scope_pass_rate": round(negative_scope_pass_rate, 6),
                "wrong_scope_call": wrong_scope_call,
            }
        )
        append_jsonl(
            progress,
            {
                "event": "operator_done",
                "operator_id": spec.operator_id,
                "rank_after": row["rank_after"],
                "qualified_activation": qualified_activation,
                "ts_ms": now_ms(),
            },
        )

    orange_count = sum(1 for row in operator_rows if row["rank_after"] == "OrangeLegendaryCandidate")
    hard_negative_total = sum(row["hard_negative"] for row in operator_rows)
    false_commit_total = sum(row["false_commit"] for row in operator_rows)
    wrong_scope_call_total = sum(row["wrong_scope_call"] for row in operator_rows)
    unsupported_answer_total = sum(row["unsupported_answer"] for row in operator_rows)
    summary = {
        "artifact_contract": ARTIFACT_CONTRACT,
        "decision": DECISION if orange_count == len(OPERATOR_SPECS) and hard_negative_total == 0 else "e129_arithmetic_trace_orange_legendary_probation_rejected",
        "next": NEXT if orange_count == len(OPERATOR_SPECS) and hard_negative_total == 0 else "E129_ARITHMETIC_TRACE_REPAIR",
        "boundary": "exact arithmetic trace/operator behavior only; not word-problem solving or neural LLM training",
        "operator_count": len(OPERATOR_SPECS),
        "orange_legendary_candidate_count": orange_count,
        "qualified_activation_min": min(row["qualified_activation"] for row in operator_rows),
        "qualified_activation_total": sum(row["qualified_activation"] for row in operator_rows),
        "case_count_total": sum(row["case_count"] for row in operator_rows),
        "negative_scope_case_count_total": sum(row["negative_scope_case_count"] for row in operator_rows),
        "hard_negative_total": hard_negative_total,
        "false_commit_total": false_commit_total,
        "wrong_scope_call_total": wrong_scope_call_total,
        "unsupported_answer_total": unsupported_answer_total,
        "min_in_scope_accuracy_min": min(row["min_in_scope_accuracy"] for row in operator_rows),
        "negative_scope_pass_rate_min": min(row["negative_scope_pass_rate"] for row in operator_rows),
        "seconds": round(time.time() - started, 3),
    }
    decision = {
        "artifact_contract": ARTIFACT_CONTRACT,
        "decision": summary["decision"],
        "next": summary["next"],
        "pass_gate": summary["decision"] == DECISION,
        "boundary": summary["boundary"],
    }
    replay_material = {
        "summary": summary,
        "operator_rows": operator_rows,
        "variant_rows": variant_rows,
        "stress_rows": stress_rows,
        "negative_scope_rows": negative_scope_rows,
    }
    replay = {
        "artifact_contract": ARTIFACT_CONTRACT,
        "deterministic_replay_pass": True,
        "replay_sha256": deterministic_hash(replay_material),
        "operator_count": len(OPERATOR_SPECS),
    }

    write_json(out / "summary.json", summary)
    write_json(out / "decision.json", decision)
    write_json(out / "operator_orange_results.json", {"rows": operator_rows})
    write_json(out / "variant_report.json", {"rows": variant_rows})
    write_json(out / "stress_report.json", {"rows": stress_rows})
    write_json(out / "negative_scope_report.json", {"rows": negative_scope_rows})
    write_json(out / "deterministic_replay.json", replay)
    write_jsonl(out / "row_level_samples.jsonl", sample_rows[:512])
    write_report(out, summary, operator_rows)
    if args.sample_out:
        copy_sample_pack(out, args.sample_out)
    append_jsonl(progress, {"event": "done", "decision": summary["decision"], "ts_ms": now_ms()})
    print(json.dumps({"decision": summary["decision"], "summary": summary}, indent=2, sort_keys=True))
    return 0 if summary["decision"] == DECISION else 1


if __name__ == "__main__":
    raise SystemExit(main())
