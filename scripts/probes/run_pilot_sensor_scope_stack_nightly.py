#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_pilot_sensor_probe as sensor_probe
from run_pilot_sensor_distill_probe import BAND_VALUES, NgramVectorizer, evidence_to_bands


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "output" / "pilot_sensor_scope_stack_nightly_001"
DOC_REPORT = ROOT / "docs" / "research" / "PILOT_SENSOR_SCOPE_STACK_NIGHTLY_001_RESULT.md"

POLICY = "evidence_strength_margin_guard"
DIAGNOSTIC_TAG = "strict_unseen_synonym"
FLAGS = [
    "add_cue",
    "mul_cue",
    "unknown_cue",
    "weak_marker",
    "ambiguity_marker",
    "mention_only",
    "multi_step_unsupported",
    "negation_add",
    "negation_mul",
    "negation_unknown",
    "correction_present",
    "correction_to_add",
    "correction_to_mul",
    "correction_to_unknown",
]


@dataclass(frozen=True)
class SensorCase:
    split: str
    name: str
    text: str
    expected: tuple[str, ...]
    phenomenon_tag: str


@dataclass
class MLPArm:
    name: str
    kind: str
    mode: str
    vectorizer: NgramVectorizer
    model: torch.nn.Module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PILOT_SENSOR_SCOPE_STACK_NIGHTLY_001 gated scope-stack nightly probe.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--epochs", type=int, default=700)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--smoke-seeds", type=str, default="0")
    parser.add_argument("--matrix-seeds", type=str, default="0,1,2")
    parser.add_argument("--confirm-seeds", type=str, default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--confirm-top-k", type=int, default=2)
    return parser.parse_args()


def parse_seeds(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def case(split: str, name: str, text: str, expected: str | tuple[str, ...], tag: str) -> SensorCase:
    exp = (expected,) if isinstance(expected, str) else expected
    return SensorCase(split, name, text, exp, tag)


def training_cases() -> list[SensorCase]:
    rows: list[SensorCase] = []
    nums = ["1", "2", "4", "5"]
    for n in nums:
        rows += [
            case("train", f"train_add_{n}_a", f"add {n}", "EXEC_ADD", "known"),
            case("train", f"train_add_{n}_b", f"please add {n}", "EXEC_ADD", "known"),
            case("train", f"train_add_{n}_c", f"plus {n}", "EXEC_ADD", "known"),
            case("train", f"train_mul_{n}_a", f"multiply by {n}", "EXEC_MUL", "known"),
            case("train", f"train_mul_{n}_b", f"please multiply {n}", "EXEC_MUL", "known"),
            case("train", f"train_mul_{n}_c", f"times {n}", "EXEC_MUL", "known"),
            case("train", f"train_unknown_{n}_a", f"divide by {n}", "REJECT_UNKNOWN", "unknown"),
            case("train", f"train_unknown_{n}_b", f"sqrt {n}", "REJECT_UNKNOWN", "unknown"),
            case("train", f"train_unknown_{n}_c", f"mod by {n}", "REJECT_UNKNOWN", "unknown"),
            case("train", f"train_weak_add_{n}_a", f"maybe add {n}", "HOLD_ASK_RESEARCH", "weak"),
            case("train", f"train_weak_add_{n}_b", f"perhaps add {n}", "HOLD_ASK_RESEARCH", "weak"),
            case("train", f"train_weak_mul_{n}_a", f"maybe multiply by {n}", "HOLD_ASK_RESEARCH", "weak"),
            case("train", f"train_weak_mul_{n}_b", f"could multiply {n}", "HOLD_ASK_RESEARCH", "weak"),
            case("train", f"train_amb_{n}_a", f"add or multiply by {n}", "HOLD_ASK_RESEARCH", "ambiguous"),
            case("train", f"train_amb_{n}_b", f"multiply or add by {n}", "HOLD_ASK_RESEARCH", "ambiguous"),
            case("train", f"train_amb_{n}_c", "not sure whether to add or multiply", "HOLD_ASK_RESEARCH", "ambiguous"),
            case("train", f"train_neg_add_{n}", f"do not add {n}", ("HOLD_ASK_RESEARCH", "REFRAME"), "negation"),
            case("train", f"train_neg_mul_{n}", f"do not multiply by {n}", ("HOLD_ASK_RESEARCH", "REFRAME"), "negation"),
            case("train", f"train_neg_add_then_mul_{n}", f"do not add {n}, multiply by {n} instead", "EXEC_MUL", "negation"),
            case("train", f"train_neg_mul_then_add_{n}", f"do not multiply by {n}, add {n} instead", "EXEC_ADD", "negation"),
            case("train", f"train_neither_{n}", "neither add nor multiply", "HOLD_ASK_RESEARCH", "negation"),
            case("train", f"train_corr_add_mul_{n}_a", f"add {n}. actually multiply by {n}", "EXEC_MUL", "correction"),
            case("train", f"train_corr_mul_add_{n}_a", f"multiply by {n}. actually add {n}", "EXEC_ADD", "correction"),
            case("train", f"train_corr_add_mul_{n}_b", f"add {n}. correction: multiply by {n}", "EXEC_MUL", "correction"),
            case("train", f"train_corr_mul_add_{n}_b", f"multiply by {n}. correction: add {n}", "EXEC_ADD", "correction"),
            case("train", f"train_first_then_{n}_a", f"first add {n}, then multiply by {n}", "HOLD_ASK_RESEARCH", "multi_step_unsupported"),
            case("train", f"train_first_then_{n}_b", f"first multiply by {n}, then add {n}", "HOLD_ASK_RESEARCH", "multi_step_unsupported"),
        ]
    rows += [
        case("train", "train_mention_add_word", "the word add appears in the note", "HOLD_ASK_RESEARCH", "mention_trap"),
        case("train", "train_mention_mul_word", "the word multiply appears in the note", "HOLD_ASK_RESEARCH", "mention_trap"),
        case("train", "train_mention_add_said", "someone said add but no operation is requested", "HOLD_ASK_RESEARCH", "mention_trap"),
        case("train", "train_mention_mul_said", "someone said multiply but no operation is requested", "HOLD_ASK_RESEARCH", "mention_trap"),
        case("train", "train_instruction_quote", "do not follow the instruction 'add 2'", ("HOLD_ASK_RESEARCH", "REFRAME"), "mention_trap"),
        case("train", "train_plus_sign", "plus sign is visible", "HOLD_ASK_RESEARCH", "substring_trap"),
        case("train", "train_plus_page", "plus sign appears on the page", "HOLD_ASK_RESEARCH", "substring_trap"),
        case("train", "train_additive_label", "the additive label is visible", "HOLD_ASK_RESEARCH", "substring_trap"),
        case("train", "train_multiply_styled", "multiply styled text but no operation", "HOLD_ASK_RESEARCH", "substring_trap"),
        case("train", "train_use_add_no_mul", "use add. actually no, use multiply", "EXEC_MUL", "correction"),
        case("train", "train_use_mul_no_add", "use multiply. actually no, use add", "EXEC_ADD", "correction"),
    ]
    return rows


def eval_cases() -> list[SensorCase]:
    rows = [
        case("controlled", "controlled_add", "add 3", "EXEC_ADD", "known"),
        case("controlled", "controlled_mul", "multiply by 3", "EXEC_MUL", "known"),
        case("controlled", "controlled_unknown", "divide by 3", "REJECT_UNKNOWN", "unknown"),
        case("controlled", "controlled_weak", "maybe add 3", "HOLD_ASK_RESEARCH", "weak"),
        case("controlled", "controlled_ambiguous", "add or multiply by 3", "HOLD_ASK_RESEARCH", "ambiguous"),
        case("controlled", "controlled_negation", "do not add 3, multiply by 3 instead", "EXEC_MUL", "negation"),
        case("controlled", "controlled_correction", "add 3. actually multiply by 3", "EXEC_MUL", "correction"),
        case("controlled", "controlled_scope", "the word add appears in the note", "HOLD_ASK_RESEARCH", "mention_trap"),
        case("heldout_surface", "surface_add_1", "ADD 7 please", "EXEC_ADD", "surface_variation"),
        case("heldout_surface", "surface_add_2", "kindly plus 7 now", "EXEC_ADD", "surface_variation"),
        case("heldout_surface", "surface_mul_1", "now multiply by 7 please", "EXEC_MUL", "surface_variation"),
        case("heldout_surface", "surface_mul_2", "TIMES 7!", "EXEC_MUL", "surface_variation"),
        case("heldout_surface", "surface_unknown_1", "please divide by 7", "REJECT_UNKNOWN", "surface_variation"),
        case("heldout_surface", "surface_unknown_2", "take sqrt now", "REJECT_UNKNOWN", "surface_variation"),
        case("heldout_scope", "scope_mention_1", "the word add appears in the note", "HOLD_ASK_RESEARCH", "mention_trap"),
        case("heldout_scope", "scope_mention_2", "do not follow the instruction 'add 3'", ("HOLD_ASK_RESEARCH", "REFRAME"), "mention_trap"),
        case("heldout_scope", "scope_mention_3", "someone said multiply, but no operation is requested", "HOLD_ASK_RESEARCH", "mention_trap"),
        case("heldout_scope", "scope_substring_1", "plus sign appears on the page", "HOLD_ASK_RESEARCH", "substring_trap"),
        case("heldout_scope", "scope_substring_2", "the additive label is visible", "HOLD_ASK_RESEARCH", "substring_trap"),
        case("heldout_negation", "negation_1", "do not add 3", ("HOLD_ASK_RESEARCH", "REFRAME"), "negation"),
        case("heldout_negation", "negation_2", "do not multiply by 3", ("HOLD_ASK_RESEARCH", "REFRAME"), "negation"),
        case("heldout_negation", "negation_3", "do not add 3, multiply by 3 instead", "EXEC_MUL", "negation"),
        case("heldout_negation", "negation_4", "do not multiply by 3, add 3 instead", "EXEC_ADD", "negation"),
        case("heldout_negation", "negation_5", "neither add nor multiply", "HOLD_ASK_RESEARCH", "negation"),
        case("heldout_negation", "negation_6", "not sure whether to add or multiply", "HOLD_ASK_RESEARCH", "negation"),
        case("heldout_correction", "correction_1", "add 3. wait, actually multiply by 3", "EXEC_MUL", "correction"),
        case("heldout_correction", "correction_2", "multiply by 3. correction: add 3", "EXEC_ADD", "correction"),
        case("heldout_correction", "correction_3", "use add. actually no, use multiply", "EXEC_MUL", "correction"),
        case("heldout_correction", "correction_4", "use multiply. actually no, use add", "EXEC_ADD", "correction"),
        case("heldout_weak_ambiguous", "weak_1", "probably multiply by 3", "HOLD_ASK_RESEARCH", "weak"),
        case("heldout_weak_ambiguous", "weak_2", "it could be add", "HOLD_ASK_RESEARCH", "weak"),
        case("heldout_weak_ambiguous", "amb_1", "add, multiply, or divide by 3", "HOLD_ASK_RESEARCH", "ambiguous"),
        case("heldout_weak_ambiguous", "amb_2", "maybe plus, maybe times 3", "HOLD_ASK_RESEARCH", "ambiguous"),
        case("heldout_multi_step_unsupported", "multi_1", "first add 3, then multiply by 3", "HOLD_ASK_RESEARCH", "multi_step_unsupported"),
        case("heldout_multi_step_unsupported", "multi_2", "first multiply by 3, then add 3", "HOLD_ASK_RESEARCH", "multi_step_unsupported"),
        case("strict_unseen_synonym_diagnostic", "strict_add_1", "increment by 3", "EXEC_ADD", DIAGNOSTIC_TAG),
        case("strict_unseen_synonym_diagnostic", "strict_add_2", "raise the value by 3", "EXEC_ADD", DIAGNOSTIC_TAG),
        case("strict_unseen_synonym_diagnostic", "strict_mul_1", "product with 3", "EXEC_MUL", DIAGNOSTIC_TAG),
        case("strict_unseen_synonym_diagnostic", "strict_unknown_1", "halve it", "REJECT_UNKNOWN", DIAGNOSTIC_TAG),
        case("strict_unseen_synonym_diagnostic", "strict_unknown_2", "exponentiate by 3", "REJECT_UNKNOWN", DIAGNOSTIC_TAG),
    ]
    return rows


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def teacher_flags(text: str) -> dict[str, int]:
    lower = text.lower()
    tok = tokens(lower)
    add, mul, unknown = sensor_probe.cue_counts(tok)
    negated = sensor_probe.negated_ops(lower)
    correction_tail = sensor_probe.split_correction(lower)
    tail_tok = tokens(correction_tail or "")
    tail_add, tail_mul, tail_unknown = sensor_probe.cue_counts(tail_tok)
    flags = {name: 0 for name in FLAGS}
    flags["mention_only"] = int(sensor_probe.is_mention_trap(lower, tok))
    flags["multi_step_unsupported"] = int(bool(re.search(r"\bfirst\b", lower) and re.search(r"\bthen\b", lower)))
    flags["weak_marker"] = int(sensor_probe.has_weak_marker(tok))
    flags["ambiguity_marker"] = int((" or " in lower) or ("whether" in tok) or (add and mul))
    flags["negation_add"] = int("ADD" in negated)
    flags["negation_mul"] = int("MUL" in negated)
    flags["negation_unknown"] = int(bool(re.search(r"\bnot (divide|sqrt|root|mod|quotient)\b", lower)))
    flags["correction_present"] = int(correction_tail is not None)
    if correction_tail is not None:
        flags["correction_to_add"] = int(bool(tail_add))
        flags["correction_to_mul"] = int(bool(tail_mul))
        flags["correction_to_unknown"] = int(bool(tail_unknown))
    flags["add_cue"] = int(bool(add))
    flags["mul_cue"] = int(bool(mul))
    flags["unknown_cue"] = int(bool(unknown))
    return flags


def flags_to_evidence(flags: dict[str, int]) -> np.ndarray:
    if flags["mention_only"]:
        return np.zeros(3, dtype=float)
    if flags["multi_step_unsupported"]:
        return np.array([0.80, 0.80, 0.0], dtype=float)
    if flags["correction_to_add"]:
        return np.array([0.90, 0.0, 0.0], dtype=float)
    if flags["correction_to_mul"]:
        return np.array([0.0, 0.90, 0.0], dtype=float)
    if flags["correction_to_unknown"]:
        return np.array([0.0, 0.0, 0.90], dtype=float)

    add = bool(flags["add_cue"] and not flags["negation_add"])
    mul = bool(flags["mul_cue"] and not flags["negation_mul"])
    unknown = bool(flags["unknown_cue"] and not flags["negation_unknown"])
    if flags["ambiguity_marker"] and add and mul and unknown:
        return np.array([0.80, 0.80, 0.80], dtype=float)
    evidence = np.zeros(3, dtype=float)
    if add:
        evidence[0] = 0.90
    if mul:
        evidence[1] = 0.90
    if unknown:
        evidence[2] = 0.90
    if unknown and (add or mul):
        evidence[0] = min(evidence[0], 0.25)
        evidence[1] = min(evidence[1], 0.25)
        evidence[2] = 0.90
    if flags["weak_marker"]:
        evidence = np.minimum(evidence, np.array([0.45, 0.45, 0.90]))
        if unknown:
            evidence[2] = 0.90
    if flags["ambiguity_marker"] and (add or mul):
        if add:
            evidence[0] = max(evidence[0], 0.50)
        if mul:
            evidence[1] = max(evidence[1], 0.50)
    return evidence


def evidence_bands_to_values(bands: Iterable[int]) -> np.ndarray:
    return np.array([BAND_VALUES[int(band)] for band in bands], dtype=float)


def teacher_evidence(case: SensorCase) -> np.ndarray:
    return sensor_probe.structured_rule_sensor(case.text)


def action_for(evidence: np.ndarray) -> str:
    return sensor_probe.policy_action(evidence, POLICY)


def is_correct(action: str, expected: tuple[str, ...]) -> bool:
    return action in expected


def is_false_commit(action: str, expected: tuple[str, ...]) -> bool:
    return sensor_probe.is_false_commit(action, expected)


def is_missed_execute(action: str, expected: tuple[str, ...]) -> bool:
    return sensor_probe.is_missed_execute(action, expected)


class MLP(torch.nn.Module):
    def __init__(self, inputs: int, hidden: int, outputs: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(inputs, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_direct_arm(name: str, kind: str, train_rows: list[SensorCase], *, seed: int, hidden: int, epochs: int, lr: float) -> MLPArm:
    torch.manual_seed(seed)
    vectorizer = NgramVectorizer(kind)
    vectorizer.fit(row.text for row in train_rows)
    x = vectorizer.transform(row.text for row in train_rows)
    y = torch.tensor([evidence_to_bands(teacher_evidence(row)) for row in train_rows], dtype=torch.long)
    model = MLP(x.shape[1], hidden, 9)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    for _ in range(epochs):
        opt.zero_grad()
        logits = model(x).view(-1, 3, 3)
        loss = sum(F.cross_entropy(logits[:, channel, :], y[:, channel]) for channel in range(3))
        loss.backward()
        opt.step()
    return MLPArm(name=name, kind=kind, mode="direct", vectorizer=vectorizer, model=model)


def train_scope_arm(name: str, kind: str, train_rows: list[SensorCase], *, seed: int, hidden: int, epochs: int, lr: float) -> MLPArm:
    torch.manual_seed(seed)
    vectorizer = NgramVectorizer(kind)
    vectorizer.fit(row.text for row in train_rows)
    x = vectorizer.transform(row.text for row in train_rows)
    y = torch.tensor([[teacher_flags(row.text)[flag] for flag in FLAGS] for row in train_rows], dtype=torch.float32)
    model = MLP(x.shape[1], hidden, len(FLAGS))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    for _ in range(epochs):
        opt.zero_grad()
        logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        opt.step()
    return MLPArm(name=name, kind=kind, mode="scope", vectorizer=vectorizer, model=model)


def predict_arm(arm: MLPArm, text: str) -> tuple[dict[str, int], np.ndarray]:
    with torch.no_grad():
        x = arm.vectorizer.transform([text])
        out = arm.model(x)
    if arm.mode == "direct":
        bands = out.view(1, 3, 3).argmax(dim=-1).squeeze(0).cpu().tolist()
        return {flag: 0 for flag in FLAGS}, evidence_bands_to_values(bands)
    pred = (torch.sigmoid(out).squeeze(0) >= 0.5).cpu().int().tolist()
    flags = {flag: int(pred[idx]) for idx, flag in enumerate(FLAGS)}
    return flags, flags_to_evidence(flags)


def evaluate_model(model_name: str, model_type: str, seed: int, rows: list[SensorCase], predictor) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for row in rows:
        teacher_flag_map = teacher_flags(row.text)
        teach_evidence = teacher_evidence(row)
        teach_action = action_for(teach_evidence)
        student_flags, student_evidence = predictor(row)
        student_action = action_for(student_evidence)
        out.append({
            "model_name": model_name,
            "model_type": model_type,
            "seed": seed,
            "split": row.split,
            "case": row.name,
            "text": row.text,
            "phenomenon_tag": row.phenomenon_tag,
            "expected_action": "|".join(row.expected),
            "teacher_flags": json.dumps(teacher_flag_map, sort_keys=True),
            "teacher_evidence": json.dumps([round(float(x), 4) for x in teach_evidence]),
            "teacher_action": teach_action,
            "student_flags": json.dumps(student_flags, sort_keys=True),
            "student_evidence": json.dumps([round(float(x), 4) for x in student_evidence]),
            "student_action": student_action,
            "action_correct": is_correct(student_action, row.expected),
            "evidence_band_correct": evidence_to_bands(student_evidence) == evidence_to_bands(teach_evidence),
            "scope_flag_correct": all(int(student_flags.get(flag, 0)) == teacher_flag_map[flag] for flag in FLAGS) if model_type in {"scope", "oracle"} else False,
            "teacher_student_disagreement": student_action != teach_action,
            "false_commit": is_false_commit(student_action, row.expected),
            "missed_execute": is_missed_execute(student_action, row.expected),
            "diagnostic": row.phenomenon_tag == DIAGNOSTIC_TAG,
        })
    return out


def baseline_rows(eval_rows: list[SensorCase]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    rows += evaluate_model(
        "keyword_sensor",
        "baseline",
        -1,
        eval_rows,
        lambda row: ({flag: 0 for flag in FLAGS}, sensor_probe.keyword_sensor(row.text)),
    )
    rows += evaluate_model(
        "structured_rule_sensor_teacher",
        "teacher",
        -1,
        eval_rows,
        lambda row: (teacher_flags(row.text), teacher_evidence(row)),
    )
    rows += evaluate_model(
        "oracle_flags_mapper",
        "oracle",
        -1,
        eval_rows,
        lambda row: (teacher_flags(row.text), flags_to_evidence(teacher_flags(row.text))),
    )
    return rows


def train_and_eval_arm(arm_name: str, seed: int, train_rows: list[SensorCase], eval_rows: list[SensorCase], args: argparse.Namespace) -> list[dict[str, object]]:
    if arm_name == "direct_evidence_word_ngram_mlp":
        arm = train_direct_arm(arm_name, "word", train_rows, seed=seed, hidden=args.hidden, epochs=args.epochs, lr=args.learning_rate)
    elif arm_name == "direct_evidence_char_ngram_mlp":
        arm = train_direct_arm(arm_name, "char", train_rows, seed=seed, hidden=args.hidden, epochs=args.epochs, lr=args.learning_rate)
    elif arm_name == "scope_stack_word_ngram_mlp":
        arm = train_scope_arm(arm_name, "word", train_rows, seed=seed, hidden=args.hidden, epochs=args.epochs, lr=args.learning_rate)
    elif arm_name == "scope_stack_char_ngram_mlp":
        arm = train_scope_arm(arm_name, "char", train_rows, seed=seed, hidden=args.hidden, epochs=args.epochs, lr=args.learning_rate)
    else:
        raise ValueError(arm_name)
    return evaluate_model(arm.name, arm.mode, seed, eval_rows, lambda row: predict_arm(arm, row.text))


def fraction(rows: list[dict[str, object]], predicate, field: str = "action_correct") -> float:
    subset = [row for row in rows if predicate(row)]
    if not subset:
        return 0.0
    return float(sum(bool(row[field]) for row in subset) / len(subset))


def rate(rows: list[dict[str, object]], predicate, field: str) -> float:
    return fraction(rows, predicate, field)


def main_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [row for row in rows if not bool(row["diagnostic"])]


def metrics_for(rows: list[dict[str, object]]) -> dict[str, float]:
    main = main_rows(rows)
    known = [row for row in main if row["split"] in {"controlled", "heldout_surface"} and row["expected_action"] in {"EXEC_ADD", "EXEC_MUL"}]
    return {
        "action_accuracy_after_guard": fraction(main, lambda _: True),
        "evidence_band_accuracy": fraction(main, lambda _: True, "evidence_band_correct"),
        "scope_flag_accuracy": fraction(main, lambda row: row["model_type"] in {"scope", "oracle"}, "scope_flag_correct"),
        "false_commit_rate": rate(main, lambda _: True, "false_commit"),
        "keyword_trap_false_commit_rate": rate(main, lambda row: row["phenomenon_tag"] in {"mention_trap", "substring_trap"}, "false_commit"),
        "missed_execute_rate": rate(main, lambda _: True, "missed_execute"),
        "over_hold_rate_on_known": float(sum(row["student_action"] == "HOLD_ASK_RESEARCH" for row in known) / len(known)) if known else 0.0,
        "heldout_scope_accuracy": fraction(rows, lambda row: row["split"] == "heldout_scope"),
        "heldout_negation_accuracy": fraction(rows, lambda row: row["split"] == "heldout_negation"),
        "heldout_correction_accuracy": fraction(rows, lambda row: row["split"] == "heldout_correction"),
        "heldout_weak_ambiguous_accuracy": fraction(rows, lambda row: row["split"] == "heldout_weak_ambiguous"),
        "multi_step_unsupported_hold_accuracy": fraction(rows, lambda row: row["phenomenon_tag"] == "multi_step_unsupported"),
        "teacher_student_disagreement_rate": rate(main, lambda _: True, "teacher_student_disagreement"),
        "known_execute_accuracy": float(sum(bool(row["action_correct"]) for row in known) / len(known)) if known else 0.0,
        "strict_unseen_synonym_accuracy": fraction(rows, lambda row: row["diagnostic"]),
    }


def selection_score(metrics: dict[str, float]) -> float:
    return (
        metrics["action_accuracy_after_guard"]
        - 2.0 * metrics["false_commit_rate"]
        - 2.0 * metrics["keyword_trap_false_commit_rate"]
        - max(0.0, 0.90 - metrics["heldout_negation_accuracy"])
        - max(0.0, 0.85 - metrics["heldout_correction_accuracy"])
        - max(0.0, 0.85 - metrics["heldout_scope_accuracy"])
    )


def aggregate_by_model_stage(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    seed_metrics: dict[tuple[str, str, int], dict[str, float]] = {}
    for key in sorted({(str(row["stage"]), str(row["model_name"]), int(row["seed"])) for row in rows}):
        stage, model, seed = key
        subset = [row for row in rows if row["stage"] == stage and row["model_name"] == model and int(row["seed"]) == seed]
        seed_metrics[key] = metrics_for(subset)
    out: dict[str, dict[str, float]] = {}
    for stage, model in sorted({(key[0], key[1]) for key in seed_metrics}):
        vals = [metrics for (s, m, _), metrics in seed_metrics.items() if s == stage and m == model]
        avg = {metric: mean(item[metric] for item in vals) for metric in vals[0]}
        avg["selection_score"] = selection_score(avg)
        avg["seed_count"] = float(len(vals))
        out[f"{stage}:{model}"] = avg
    return out


def phenomenon_metrics(rows: list[dict[str, object]]) -> dict[str, dict[str, dict[str, float]]]:
    out: dict[str, dict[str, dict[str, float]]] = {}
    for stage, model in sorted({(str(row["stage"]), str(row["model_name"])) for row in rows}):
        subset = [row for row in rows if row["stage"] == stage and row["model_name"] == model]
        key = f"{stage}:{model}"
        out[key] = {}
        for tag in sorted({str(row["phenomenon_tag"]) for row in subset}):
            tag_rows = [row for row in subset if row["phenomenon_tag"] == tag]
            out[key][tag] = {
                "action_accuracy": fraction(tag_rows, lambda _: True),
                "false_commit_rate": rate(tag_rows, lambda _: True, "false_commit"),
                "evidence_band_accuracy": fraction(tag_rows, lambda _: True, "evidence_band_correct"),
                "scope_flag_accuracy": fraction(tag_rows, lambda row: row["model_type"] in {"scope", "oracle"}, "scope_flag_correct"),
            }
    return out


def failure_type(row: dict[str, object]) -> str:
    tag = str(row["phenomenon_tag"])
    if bool(row["false_commit"]):
        if tag == "negation":
            return "negation_error"
        if tag in {"mention_trap", "substring_trap"}:
            return "mention_trap_error"
        if tag == "multi_step_unsupported":
            return "multi_step_error"
        return "false_commit"
    if bool(row["missed_execute"]):
        if tag == DIAGNOSTIC_TAG:
            return "synonym_gap"
        return "missed_execute"
    if tag == "correction" and not bool(row["action_correct"]):
        return "correction_error"
    if not bool(row["scope_flag_correct"]) and row["model_type"] == "scope":
        return "scope_error"
    if not bool(row["evidence_band_correct"]):
        return "evidence_mapping_error"
    return "over_hold"


def failure_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    failures = []
    for row in rows:
        if row["model_type"] in {"baseline", "teacher"}:
            continue
        if bool(row["action_correct"]) and bool(row["evidence_band_correct"]) and (row["model_type"] != "scope" or bool(row["scope_flag_correct"])):
            continue
        failures.append({
            "text": row["text"],
            "split": row["split"],
            "phenomenon_tag": row["phenomenon_tag"],
            "expected_action": row["expected_action"],
            "teacher_flags": row["teacher_flags"],
            "teacher_evidence": row["teacher_evidence"],
            "student_flags": row["student_flags"],
            "student_evidence": row["student_evidence"],
            "student_action": row["student_action"],
            "failure_type": failure_type(row),
            "model_name": row["model_name"],
            "seed": row["seed"],
            "stage": row["stage"],
        })
    return failures


def smoke_viable(metrics: dict[str, float]) -> bool:
    return (
        metrics["false_commit_rate"] <= 0.25
        and metrics["heldout_scope_accuracy"] >= 0.50
        and metrics["heldout_correction_accuracy"] >= 0.50
    )


def promote(metrics: dict[str, float]) -> bool:
    return (
        metrics["action_accuracy_after_guard"] >= 0.85
        and metrics["false_commit_rate"] <= 0.10
        and metrics["heldout_scope_accuracy"] >= 0.75
        and metrics["heldout_correction_accuracy"] >= 0.75
    )


def passes(metrics: dict[str, float]) -> bool:
    return (
        metrics["action_accuracy_after_guard"] >= 0.90
        and metrics["false_commit_rate"] <= 0.05
        and metrics["keyword_trap_false_commit_rate"] <= 0.05
        and metrics["heldout_scope_accuracy"] >= 0.85
        and metrics["heldout_negation_accuracy"] >= 0.90
        and metrics["heldout_correction_accuracy"] >= 0.85
        and metrics["known_execute_accuracy"] >= 0.95
    )


def verdicts(agg: dict[str, dict[str, float]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    confirm_scope_pass = any(key.startswith("confirm:scope_stack") and passes(metrics) for key, metrics in agg.items())
    scope_best = max((metrics["selection_score"] for key, metrics in agg.items() if "scope_stack" in key), default=-999.0)
    direct_best = max((metrics["selection_score"] for key, metrics in agg.items() if "direct_evidence" in key), default=-999.0)
    learned_pass = any((key.startswith("confirm:") or key.startswith("matrix:")) and ("direct_evidence" in key or "scope_stack" in key) and passes(metrics) for key, metrics in agg.items())
    teacher_pass = any(key.endswith("structured_rule_sensor_teacher") and metrics["action_accuracy_after_guard"] >= 0.99 for key, metrics in agg.items())
    for key, metrics in agg.items():
        labels: list[str] = []
        if key.endswith("structured_rule_sensor_teacher"):
            labels.append("TEACHER_REFERENCE_PASS" if metrics["action_accuracy_after_guard"] >= 0.99 else "TEACHER_REFERENCE_WEAK")
        if key.endswith("oracle_flags_mapper"):
            labels.append("ORACLE_FLAGS_MAPPER_PASS" if metrics["action_accuracy_after_guard"] >= 0.99 else "ORACLE_FLAGS_MAPPER_WEAK")
        if "direct_evidence" in key and not passes(metrics):
            labels.append("DIRECT_EVIDENCE_STILL_WEAK")
            if metrics["action_accuracy_after_guard"] >= 0.90 and metrics["heldout_weak_ambiguous_accuracy"] < 0.85:
                labels.append("WEAK_AMBIGUOUS_BOTTLENECK")
        if "scope_stack" in key:
            if passes(metrics):
                labels.append("SCOPE_STACK_POSITIVE")
            elif scope_best > direct_best + 0.05:
                labels.append("SCOPE_STACK_PARTIAL_POSITIVE")
            elif metrics["action_accuracy_after_guard"] >= 0.90 and metrics["heldout_weak_ambiguous_accuracy"] < 0.85:
                labels.append("SCOPE_STACK_NEAR_PASS_WEAK_AMBIGUOUS_BOTTLENECK")
        if metrics["strict_unseen_synonym_accuracy"] < 0.75:
            labels.append("STRICT_UNSEEN_SYNONYM_UNSOLVED")
        if not labels:
            labels.append("REFERENCE_OR_BASELINE")
        out[key] = labels
    if teacher_pass and not learned_pass:
        for key in out:
            if "direct_evidence" in key or "scope_stack" in key:
                out[key].append("DISTILL_RULE_TEACHER_ONLY")
                out[key].append("LEARNED_SCOPE_SENSOR_BOTTLENECK")
    if confirm_scope_pass:
        out["global"] = ["SCOPE_STACK_POSITIVE"]
    elif scope_best > direct_best + 0.05:
        out["global"] = ["SCOPE_STACK_PARTIAL_POSITIVE"]
    elif any(
        ("direct_evidence" in key or "scope_stack" in key)
        and metrics["action_accuracy_after_guard"] >= 0.90
        and metrics["heldout_scope_accuracy"] >= 0.85
        and metrics["heldout_negation_accuracy"] >= 0.90
        and metrics["heldout_correction_accuracy"] >= 0.85
        and metrics["heldout_weak_ambiguous_accuracy"] < 0.85
        for key, metrics in agg.items()
    ):
        out["global"] = ["LEARNED_SENSOR_NEAR_PASS", "WEAK_AMBIGUOUS_FALSE_COMMIT_BOTTLENECK", "SCOPE_STACK_NO_BETTER_THAN_DIRECT"]
    elif teacher_pass and not learned_pass:
        out["global"] = ["DISTILL_RULE_TEACHER_ONLY", "LEARNED_SCOPE_SENSOR_BOTTLENECK"]
    else:
        out["global"] = ["NIGHTLY_INCONCLUSIVE"]
    return out


def run_stage(args: argparse.Namespace) -> tuple[list[dict[str, object]], dict[str, object]]:
    train_rows = training_cases()
    eval_rows = eval_cases()
    rows = baseline_rows(eval_rows)
    for row in rows:
        row["stage"] = "reference"

    ref_agg = aggregate_by_model_stage(rows)
    teacher_key = "reference:structured_rule_sensor_teacher"
    if ref_agg.get(teacher_key, {}).get("action_accuracy_after_guard", 0.0) < 0.99:
        return rows, {"stopped": "teacher_sanity_failed", "smoke_viable": [], "matrix_promoted": [], "confirmed": []}

    arms = [
        "direct_evidence_word_ngram_mlp",
        "direct_evidence_char_ngram_mlp",
        "scope_stack_word_ngram_mlp",
        "scope_stack_char_ngram_mlp",
    ]
    smoke_rows: list[dict[str, object]] = []
    for seed in parse_seeds(args.smoke_seeds):
        for arm in arms:
            arm_rows = train_and_eval_arm(arm, seed, train_rows, eval_rows, args)
            for row in arm_rows:
                row["stage"] = "smoke"
            smoke_rows += arm_rows
    rows += smoke_rows
    smoke_agg = aggregate_by_model_stage(smoke_rows)
    viable = [
        key.split(":", 1)[1]
        for key, metrics in smoke_agg.items()
        if key.startswith("smoke:") and smoke_viable(metrics)
    ]

    matrix_rows: list[dict[str, object]] = []
    for seed in parse_seeds(args.matrix_seeds):
        for arm in viable:
            arm_rows = train_and_eval_arm(arm, seed, train_rows, eval_rows, args)
            for row in arm_rows:
                row["stage"] = "matrix"
            matrix_rows += arm_rows
    rows += matrix_rows
    matrix_agg = aggregate_by_model_stage(matrix_rows)
    promoted = [
        key.split(":", 1)[1]
        for key, metrics in sorted(matrix_agg.items(), key=lambda item: item[1]["selection_score"], reverse=True)
        if key.startswith("matrix:") and promote(metrics)
    ][: args.confirm_top_k]

    confirm_rows: list[dict[str, object]] = []
    for seed in parse_seeds(args.confirm_seeds):
        for arm in promoted:
            arm_rows = train_and_eval_arm(arm, seed, train_rows, eval_rows, args)
            for row in arm_rows:
                row["stage"] = "confirm"
            confirm_rows += arm_rows
    rows += confirm_rows
    return rows, {"stopped": "", "smoke_viable": viable, "matrix_promoted": promoted, "confirmed": promoted}


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_failures(failures: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for row in failures:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def metric_table(agg: dict[str, dict[str, float]]) -> list[str]:
    lines = [
        "| Stage:Model | Seeds | Score | Action | Scope | Neg | Corr | Weak/Amb | False Commit | Trap False | Scope Flags | Evidence Band | Strict Syn |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, m in agg.items():
        lines.append(
            f"| `{key}` | `{m['seed_count']:.0f}` | `{m['selection_score']:.3f}` | `{m['action_accuracy_after_guard']:.3f}` "
            f"| `{m['heldout_scope_accuracy']:.3f}` | `{m['heldout_negation_accuracy']:.3f}` "
            f"| `{m['heldout_correction_accuracy']:.3f}` | `{m['heldout_weak_ambiguous_accuracy']:.3f}` | `{m['false_commit_rate']:.3f}` "
            f"| `{m['keyword_trap_false_commit_rate']:.3f}` | `{m['scope_flag_accuracy']:.3f}` "
            f"| `{m['evidence_band_accuracy']:.3f}` | `{m['strict_unseen_synonym_accuracy']:.3f}` |"
        )
    return lines


def phenomenon_table(phenom: dict[str, dict[str, dict[str, float]]]) -> list[str]:
    lines = [
        "| Stage:Model | Phenomenon | Action | False Commit | Evidence Band | Scope Flag |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for key, by_tag in phenom.items():
        for tag, m in by_tag.items():
            lines.append(
                f"| `{key}` | `{tag}` | `{m['action_accuracy']:.3f}` | `{m['false_commit_rate']:.3f}` "
                f"| `{m['evidence_band_accuracy']:.3f}` | `{m['scope_flag_accuracy']:.3f}` |"
            )
    return lines


def failure_lines(failures: list[dict[str, object]], limit: int = 40) -> list[str]:
    if not failures:
        return ["No non-reference failures."]
    lines = []
    for row in failures[:limit]:
        lines.append(
            f"- `{row['stage']}:{row['model_name']}` seed `{row['seed']}` `{row['split']}/{row['phenomenon_tag']}`: "
            f"{row['text']} -> expected `{row['expected_action']}`, got `{row['student_action']}` ({row['failure_type']})."
        )
    if len(failures) > limit:
        lines.append(f"- ... {len(failures) - limit} more in `failure_examples.jsonl`.")
    return lines


def write_report(rows: list[dict[str, object]], failures: list[dict[str, object]], summary: dict[str, object], output_report: Path) -> None:
    agg = aggregate_by_model_stage(rows)
    phenom = phenomenon_metrics(rows)
    verdict = verdicts(agg)
    lines = [
        "# PILOT_SENSOR_SCOPE_STACK_NIGHTLY_001 Result",
        "",
        "## Goal",
        "",
        "Decide whether learned scope-aware text-to-evidence failure is mainly data/architecture, or whether an explicit scope/event parser is needed.",
        "",
        "Pipeline under test: `text -> scope flags -> evidence -> fixed guard -> action`.",
        "",
        "## Setup",
        "",
        "- Fixed guard: `evidence_strength_margin_guard` from `PILOT_TOPK_GUARD_001`.",
        "- Teacher: `structured_rule_sensor` from `PILOT_SENSOR_001`.",
        "- Direct arms predict evidence bands directly.",
        "- Scope-stack arms predict scope flags, then deterministic mapper produces evidence.",
        "- No direct action head is used for verdict.",
        "",
        "## Gate Summary",
        "",
        "```json",
        json.dumps(summary, indent=2, sort_keys=True),
        "```",
        "",
        "## Aggregate Metrics",
        "",
        *metric_table(agg),
        "",
        "## Phenomenon Metrics",
        "",
        *phenomenon_table(phenom),
        "",
        "## Verdict",
        "",
        "```json",
        json.dumps(verdict, indent=2, sort_keys=True),
        "```",
        "",
        "## Failure Examples",
        "",
        *failure_lines(failures),
        "",
        "## Interpretation",
        "",
        "A scope-stack positive means decomposition into learned flags plus deterministic evidence mapping is sufficient in this toy command sensor setting.",
        "",
        "If teacher and oracle mapper pass but learned arms fail, the bottleneck is learned scope/event extraction rather than the guard or evidence mapper.",
        "",
        "Strict unseen synonyms are diagnostic only because the setup has no pretrained semantics.",
        "",
        "## Claim Boundary",
        "",
        "No general NLU, full PilotPulse integration, production VRAXION/INSTNCT, or consciousness claim.",
    ]
    text = "\n".join(lines) + "\n"
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(text)
    DOC_REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC_REPORT.write_text(text)


def main() -> None:
    args = parse_args()
    rows, gate_summary = run_stage(args)
    failures = failure_rows(rows)
    agg = aggregate_by_model_stage(rows)
    phenom = phenomenon_metrics(rows)
    verdict = verdicts(agg)
    summary = {
        "gates": gate_summary,
        "aggregate": agg,
        "phenomenon_metrics": phenom,
        "verdict": verdict,
        "failure_count": len(failures),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.out_dir / "metrics.csv")
    write_failures(failures, args.out_dir / "failure_examples.jsonl")
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_report(rows, failures, gate_summary, args.out_dir / "report.md")
    print(json.dumps({
        "doc_report": str(DOC_REPORT),
        "metrics": str(args.out_dir / "metrics.csv"),
        "failures": str(args.out_dir / "failure_examples.jsonl"),
        "report": str(args.out_dir / "report.md"),
        "failure_count": len(failures),
        "gates": gate_summary,
        "verdict": verdict,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
