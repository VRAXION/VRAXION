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
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_pilot_sensor_probe as sensor_probe


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "output" / "pilot_sensor_distill_001"
DOC_REPORT = ROOT / "docs" / "research" / "PILOT_SENSOR_DISTILL_001_RESULT.md"

BAND_NAMES = ["NONE", "WEAK", "STRONG"]
BAND_VALUES = np.array([0.0, 0.45, 0.90], dtype=float)
POLICIES = ["evidence_strength_margin_guard", "topK2_guard"]
DIAGNOSTIC_TAG = "strict_unseen_synonym"


@dataclass(frozen=True)
class DistillCase:
    split: str
    name: str
    text: str
    expected: tuple[str, ...]
    phenomenon_tag: str


@dataclass
class LinearStudent:
    name: str
    vectorizer: "NgramVectorizer"
    model: torch.nn.Linear


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PILOT_SENSOR_DISTILL_001 learned evidence sensor distillation probe.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=23)
    return parser.parse_args()


def cases() -> list[DistillCase]:
    rows = [
        # Train-simple: enough teacher coverage to make scope learning possible, but heldout text differs.
        DistillCase("train_simple", "train_known_add_1", "add 2", ("EXEC_ADD",), "known"),
        DistillCase("train_simple", "train_known_add_2", "plus 2", ("EXEC_ADD",), "known"),
        DistillCase("train_simple", "train_known_add_3", "please add 2", ("EXEC_ADD",), "known"),
        DistillCase("train_simple", "train_known_mul_1", "multiply by 2", ("EXEC_MUL",), "known"),
        DistillCase("train_simple", "train_known_mul_2", "times 2", ("EXEC_MUL",), "known"),
        DistillCase("train_simple", "train_known_mul_3", "please multiply 2", ("EXEC_MUL",), "known"),
        DistillCase("train_simple", "train_unknown_1", "divide by 2", ("REJECT_UNKNOWN",), "unknown"),
        DistillCase("train_simple", "train_unknown_2", "sqrt 2", ("REJECT_UNKNOWN",), "unknown"),
        DistillCase("train_simple", "train_unknown_3", "mod by 2", ("REJECT_UNKNOWN",), "unknown"),
        DistillCase("train_simple", "train_weak_add_1", "maybe add 2", ("HOLD_ASK_RESEARCH",), "weak"),
        DistillCase("train_simple", "train_weak_add_2", "perhaps add 2", ("HOLD_ASK_RESEARCH",), "weak"),
        DistillCase("train_simple", "train_weak_mul_1", "maybe multiply by 2", ("HOLD_ASK_RESEARCH",), "weak"),
        DistillCase("train_simple", "train_weak_mul_2", "could multiply 2", ("HOLD_ASK_RESEARCH",), "weak"),
        DistillCase("train_simple", "train_ambiguous_1", "add or multiply by 2", ("HOLD_ASK_RESEARCH",), "ambiguous"),
        DistillCase("train_simple", "train_ambiguous_2", "multiply or add by 2", ("HOLD_ASK_RESEARCH",), "ambiguous"),
        DistillCase("train_simple", "train_ambiguous_3", "not sure whether to add or multiply", ("HOLD_ASK_RESEARCH",), "ambiguous"),
        DistillCase("train_simple", "train_negation_1", "do not add 2", ("HOLD_ASK_RESEARCH", "REFRAME"), "negation"),
        DistillCase("train_simple", "train_negation_2", "do not multiply by 2", ("HOLD_ASK_RESEARCH", "REFRAME"), "negation"),
        DistillCase("train_simple", "train_negation_3", "do not add 2, multiply by 2 instead", ("EXEC_MUL",), "negation"),
        DistillCase("train_simple", "train_negation_4", "do not multiply by 2, add 2 instead", ("EXEC_ADD",), "negation"),
        DistillCase("train_simple", "train_negation_5", "neither add nor multiply", ("HOLD_ASK_RESEARCH",), "negation"),
        DistillCase("train_simple", "train_correction_1", "add 2. actually multiply by 2", ("EXEC_MUL",), "correction"),
        DistillCase("train_simple", "train_correction_2", "multiply by 2. actually add 2", ("EXEC_ADD",), "correction"),
        DistillCase("train_simple", "train_correction_3", "add 2. correction: multiply by 2", ("EXEC_MUL",), "correction"),
        DistillCase("train_simple", "train_correction_4", "multiply by 2. correction: add 2", ("EXEC_ADD",), "correction"),
        DistillCase("train_simple", "train_mention_1", "the word multiply appears in the note", ("HOLD_ASK_RESEARCH",), "mention_trap"),
        DistillCase("train_simple", "train_mention_2", "someone said add but no operation is requested", ("HOLD_ASK_RESEARCH",), "mention_trap"),
        DistillCase("train_simple", "train_substring_1", "plus sign is visible", ("HOLD_ASK_RESEARCH",), "substring_trap"),
        DistillCase("train_simple", "train_substring_2", "multiply styled text but no operation", ("HOLD_ASK_RESEARCH",), "substring_trap"),
        DistillCase("train_simple", "train_multistep_1", "first multiply by 2, then add 2", ("HOLD_ASK_RESEARCH",), "multi_step_unsupported"),
        # Heldout surface variations with known cue atoms.
        DistillCase("heldout_surface", "surface_add_1", "ADD 3 please", ("EXEC_ADD",), "surface_variation"),
        DistillCase("heldout_surface", "surface_add_2", "kindly plus 3 now", ("EXEC_ADD",), "surface_variation"),
        DistillCase("heldout_surface", "surface_add_3", "please, add 3.", ("EXEC_ADD",), "surface_variation"),
        DistillCase("heldout_surface", "surface_mul_1", "now multiply by 3 please", ("EXEC_MUL",), "surface_variation"),
        DistillCase("heldout_surface", "surface_mul_2", "TIMES 3!", ("EXEC_MUL",), "surface_variation"),
        DistillCase("heldout_surface", "surface_unknown_1", "please divide by 3", ("REJECT_UNKNOWN",), "surface_variation"),
        DistillCase("heldout_surface", "surface_unknown_2", "take sqrt now", ("REJECT_UNKNOWN",), "surface_variation"),
        # Heldout scope traps.
        DistillCase("heldout_scope", "scope_mention_1", "the word add appears in the note", ("HOLD_ASK_RESEARCH",), "mention_trap"),
        DistillCase("heldout_scope", "scope_mention_2", "do not follow the instruction 'add 3'", ("HOLD_ASK_RESEARCH", "REFRAME"), "mention_trap"),
        DistillCase("heldout_scope", "scope_mention_3", "someone said multiply, but no operation is requested", ("HOLD_ASK_RESEARCH",), "mention_trap"),
        DistillCase("heldout_scope", "scope_substring_1", "plus sign appears on the page", ("HOLD_ASK_RESEARCH",), "substring_trap"),
        DistillCase("heldout_scope", "scope_substring_2", "the additive label is visible", ("HOLD_ASK_RESEARCH",), "substring_trap"),
        # Heldout negation.
        DistillCase("heldout_negation", "negation_1", "do not add 3", ("HOLD_ASK_RESEARCH", "REFRAME"), "negation"),
        DistillCase("heldout_negation", "negation_2", "do not multiply by 3", ("HOLD_ASK_RESEARCH", "REFRAME"), "negation"),
        DistillCase("heldout_negation", "negation_3", "do not add 3, multiply by 3 instead", ("EXEC_MUL",), "negation"),
        DistillCase("heldout_negation", "negation_4", "do not multiply by 3, add 3 instead", ("EXEC_ADD",), "negation"),
        DistillCase("heldout_negation", "negation_5", "neither add nor multiply", ("HOLD_ASK_RESEARCH",), "negation"),
        DistillCase("heldout_negation", "negation_6", "not sure whether to add or multiply", ("HOLD_ASK_RESEARCH",), "negation"),
        # Heldout correction.
        DistillCase("heldout_correction", "correction_1", "add 3. wait, actually multiply by 3", ("EXEC_MUL",), "correction"),
        DistillCase("heldout_correction", "correction_2", "multiply by 3. correction: add 3", ("EXEC_ADD",), "correction"),
        DistillCase("heldout_correction", "correction_3", "first add 3, then multiply by 3", ("HOLD_ASK_RESEARCH",), "multi_step_unsupported"),
        DistillCase("heldout_correction", "correction_4", "use add. actually no, use multiply", ("EXEC_MUL",), "correction"),
        # Strict unseen synonym diagnostic. These are deliberately excluded from main pass/fail.
        DistillCase("strict_unseen_synonym_diagnostic", "strict_add_1", "increment by 3", ("EXEC_ADD",), "strict_unseen_synonym"),
        DistillCase("strict_unseen_synonym_diagnostic", "strict_add_2", "raise the value by 3", ("EXEC_ADD",), "strict_unseen_synonym"),
        DistillCase("strict_unseen_synonym_diagnostic", "strict_mul_1", "product with 3", ("EXEC_MUL",), "strict_unseen_synonym"),
        DistillCase("strict_unseen_synonym_diagnostic", "strict_unknown_1", "halve it", ("REJECT_UNKNOWN",), "strict_unseen_synonym"),
        DistillCase("strict_unseen_synonym_diagnostic", "strict_unknown_2", "exponentiate by 3", ("REJECT_UNKNOWN",), "strict_unseen_synonym"),
    ]
    return rows


def normalize_text(text: str) -> str:
    return " ".join(re.findall(r"[a-z]+", text.lower()))


class NgramVectorizer:
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.vocab: dict[str, int] = {}

    def _features(self, text: str) -> list[str]:
        if self.kind == "word":
            words = re.findall(r"[a-z]+", text.lower())
            feats = [f"w={word}" for word in words]
            feats += [f"w2={words[i]}_{words[i + 1]}" for i in range(max(0, len(words) - 1))]
            return feats
        if self.kind == "char":
            cleaned = f" {normalize_text(text)} "
            feats: list[str] = []
            for n in (3, 4, 5):
                feats += [f"c{n}={cleaned[i:i + n]}" for i in range(max(0, len(cleaned) - n + 1))]
            return feats
        raise ValueError(self.kind)

    def fit(self, texts: Iterable[str]) -> None:
        vocab = sorted({feat for text in texts for feat in self._features(text)})
        self.vocab = {feat: idx for idx, feat in enumerate(vocab)}

    def transform(self, texts: Iterable[str]) -> torch.Tensor:
        text_list = list(texts)
        x = torch.zeros((len(text_list), len(self.vocab)), dtype=torch.float32)
        for row, text in enumerate(text_list):
            for feat in self._features(text):
                idx = self.vocab.get(feat)
                if idx is not None:
                    x[row, idx] = 1.0
        return x


def evidence_to_bands(evidence: np.ndarray) -> tuple[int, int, int]:
    bands = []
    for value in evidence:
        if float(value) >= 0.75:
            bands.append(2)
        elif float(value) > 1e-9:
            bands.append(1)
        else:
            bands.append(0)
    return tuple(bands)  # type: ignore[return-value]


def bands_to_evidence(bands: Iterable[int]) -> np.ndarray:
    return np.array([BAND_VALUES[int(band)] for band in bands], dtype=float)


def teacher_evidence(case: DistillCase) -> np.ndarray:
    return sensor_probe.structured_rule_sensor(case.text)


def keyword_evidence(case: DistillCase) -> np.ndarray:
    return sensor_probe.keyword_sensor(case.text)


def expected_band_ok(expected: tuple[str, ...], evidence: np.ndarray) -> bool:
    return sensor_probe.band_correct(expected, evidence)


def action_for(evidence: np.ndarray, policy: str) -> str:
    return sensor_probe.policy_action(evidence, policy)


def train_student(name: str, kind: str, train_cases: list[DistillCase], *, epochs: int, learning_rate: float, seed: int) -> LinearStudent:
    torch.manual_seed(seed)
    vectorizer = NgramVectorizer(kind)
    vectorizer.fit(case.text for case in train_cases)
    x = vectorizer.transform(case.text for case in train_cases)
    y = torch.tensor([evidence_to_bands(teacher_evidence(case)) for case in train_cases], dtype=torch.long)
    model = torch.nn.Linear(x.shape[1], 9)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    for _ in range(epochs):
        opt.zero_grad()
        logits = model(x).view(-1, 3, 3)
        loss = sum(F.cross_entropy(logits[:, channel, :], y[:, channel]) for channel in range(3))
        loss.backward()
        opt.step()
    return LinearStudent(name=name, vectorizer=vectorizer, model=model)


def predict_student(student: LinearStudent, text: str) -> np.ndarray:
    with torch.no_grad():
        x = student.vectorizer.transform([text])
        logits = student.model(x).view(1, 3, 3)
        bands = logits.argmax(dim=-1).squeeze(0).cpu().tolist()
    return bands_to_evidence(bands)


def is_correct(action: str, expected: tuple[str, ...]) -> bool:
    return action in expected


def is_false_commit(action: str, expected: tuple[str, ...]) -> bool:
    return sensor_probe.is_false_commit(action, expected)


def is_missed_execute(action: str, expected: tuple[str, ...]) -> bool:
    return sensor_probe.is_missed_execute(action, expected)


def failure_type(case: DistillCase, action: str, expected: tuple[str, ...], evidence_band_correct: bool) -> str:
    if is_false_commit(action, expected):
        if case.phenomenon_tag == "negation":
            return "negation_scope_error"
        if case.phenomenon_tag in {"mention_trap", "substring_trap"}:
            return "mention_trap_error"
        return "false_commit"
    if is_missed_execute(action, expected):
        if case.phenomenon_tag == DIAGNOSTIC_TAG:
            return "synonym_gap"
        return "missed_execute"
    if "REJECT_UNKNOWN" in expected and action != "REJECT_UNKNOWN":
        return "unknown_missed"
    if action == "HOLD_ASK_RESEARCH" and any(item in {"EXEC_ADD", "EXEC_MUL"} for item in expected):
        return "over_hold"
    if case.phenomenon_tag == "correction":
        return "correction_scope_error"
    if not evidence_band_correct:
        return "evidence_band_error"
    return "action_error"


def evaluate(args: argparse.Namespace) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    all_cases = cases()
    train_cases = [case for case in all_cases if case.split == "train_simple"]
    students = [
        train_student("word_ngram_linear_student", "word", train_cases, epochs=args.epochs, learning_rate=args.learning_rate, seed=args.seed),
        train_student("char_ngram_linear_student", "char", train_cases, epochs=args.epochs, learning_rate=args.learning_rate, seed=args.seed + 1),
    ]

    models: list[tuple[str, str, object]] = [
        ("keyword_sensor", "baseline", keyword_evidence),
        ("structured_rule_sensor_teacher", "teacher", teacher_evidence),
    ]
    models.extend((student.name, "learned", student) for student in students)

    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for model_name, model_type, model_obj in models:
        for policy in POLICIES:
            for case in all_cases:
                teach_evidence = teacher_evidence(case)
                teach_action = action_for(teach_evidence, policy)
                if model_type == "learned":
                    student_evidence = predict_student(model_obj, case.text)  # type: ignore[arg-type]
                else:
                    student_evidence = model_obj(case)  # type: ignore[operator]
                student_action = action_for(student_evidence, policy)
                band_match_teacher = evidence_to_bands(student_evidence) == evidence_to_bands(teach_evidence)
                correct = is_correct(student_action, case.expected)
                row = {
                    "model": model_name,
                    "model_type": model_type,
                    "guard": policy,
                    "split": case.split,
                    "case": case.name,
                    "text": case.text,
                    "phenomenon_tag": case.phenomenon_tag,
                    "expected_action": "|".join(case.expected),
                    "teacher_evidence": json.dumps([round(float(x), 4) for x in teach_evidence]),
                    "teacher_action": teach_action,
                    "student_evidence": json.dumps([round(float(x), 4) for x in student_evidence]),
                    "student_action": student_action,
                    "action_correct": correct,
                    "evidence_band_correct": band_match_teacher,
                    "guard_compatible_band": expected_band_ok(case.expected, student_evidence),
                    "teacher_student_disagreement": student_action != teach_action,
                    "false_commit": is_false_commit(student_action, case.expected),
                    "missed_execute": is_missed_execute(student_action, case.expected),
                    "diagnostic": case.phenomenon_tag == DIAGNOSTIC_TAG,
                }
                rows.append(row)
                if model_type == "learned" and (not correct or not band_match_teacher):
                    failures.append({
                        "model": model_name,
                        "guard": policy,
                        "text": case.text,
                        "split": case.split,
                        "phenomenon_tag": case.phenomenon_tag,
                        "expected_action": "|".join(case.expected),
                        "teacher_evidence": row["teacher_evidence"],
                        "teacher_action": teach_action,
                        "student_evidence": row["student_evidence"],
                        "student_action": student_action,
                        "failure_type": failure_type(case, student_action, case.expected, band_match_teacher),
                    })
    return rows, failures


def fraction(rows: list[dict[str, object]], predicate, field: str = "action_correct") -> float:
    subset = [row for row in rows if predicate(row)]
    if not subset:
        return 0.0
    return float(sum(bool(row[field]) for row in subset) / len(subset))


def rate(rows: list[dict[str, object]], predicate, field: str) -> float:
    subset = [row for row in rows if predicate(row)]
    if not subset:
        return 0.0
    return float(sum(bool(row[field]) for row in subset) / len(subset))


def main_eval_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [row for row in rows if row["split"] != "train_simple" and not bool(row["diagnostic"])]


def aggregate(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    keys = sorted({f"{row['model']}+{row['guard']}" for row in rows})
    for key in keys:
        model, policy = key.split("+", 1)
        subset = [row for row in rows if row["model"] == model and row["guard"] == policy]
        main_rows = main_eval_rows(subset)
        known_rows = [
            row for row in main_rows
            if row["split"] == "heldout_surface" and row["expected_action"] in {"EXEC_ADD", "EXEC_MUL"}
        ]
        metrics = {
            "train_action_accuracy": fraction(subset, lambda row: row["split"] == "train_simple"),
            "action_accuracy_after_guard": fraction(main_rows, lambda _: True),
            "evidence_band_accuracy": fraction(main_rows, lambda _: True, "evidence_band_correct"),
            "teacher_student_disagreement_rate": rate(main_rows, lambda _: True, "teacher_student_disagreement"),
            "heldout_surface_accuracy": fraction(subset, lambda row: row["split"] == "heldout_surface"),
            "heldout_scope_accuracy": fraction(subset, lambda row: row["split"] == "heldout_scope"),
            "heldout_negation_accuracy": fraction(subset, lambda row: row["split"] == "heldout_negation"),
            "heldout_correction_accuracy": fraction(subset, lambda row: row["split"] == "heldout_correction"),
            "strict_unseen_synonym_accuracy": fraction(subset, lambda row: row["split"] == "strict_unseen_synonym_diagnostic"),
            "false_commit_rate": rate(main_rows, lambda _: True, "false_commit"),
            "keyword_trap_false_commit_rate": rate(main_rows, lambda row: row["phenomenon_tag"] in {"mention_trap", "substring_trap"}, "false_commit"),
            "known_execute_accuracy": float(sum(bool(row["action_correct"]) for row in known_rows) / len(known_rows)) if known_rows else 0.0,
            "unknown_reject_accuracy": fraction(main_rows, lambda row: row["phenomenon_tag"] == "unknown"),
            "weak_hold_accuracy": fraction(main_rows, lambda row: row["phenomenon_tag"] == "weak"),
            "ambiguous_hold_accuracy": fraction(main_rows, lambda row: row["phenomenon_tag"] == "ambiguous"),
            "missed_execute_rate": rate(main_rows, lambda _: True, "missed_execute"),
            "over_hold_rate_on_known": float(sum(row["student_action"] == "HOLD_ASK_RESEARCH" for row in known_rows) / len(known_rows)) if known_rows else 0.0,
        }
        out[key] = metrics
    return out


def phenomenon_metrics(rows: list[dict[str, object]]) -> dict[str, dict[str, dict[str, float]]]:
    out: dict[str, dict[str, dict[str, float]]] = {}
    for key in sorted({f"{row['model']}+{row['guard']}" for row in rows}):
        model, policy = key.split("+", 1)
        subset = [row for row in rows if row["model"] == model and row["guard"] == policy and row["split"] != "train_simple"]
        out[key] = {}
        for tag in sorted({str(row["phenomenon_tag"]) for row in subset}):
            tag_rows = [row for row in subset if row["phenomenon_tag"] == tag]
            out[key][tag] = {
                "action_accuracy": fraction(tag_rows, lambda _: True),
                "false_commit_rate": rate(tag_rows, lambda _: True, "false_commit"),
                "evidence_band_accuracy": fraction(tag_rows, lambda _: True, "evidence_band_correct"),
                "teacher_student_disagreement_rate": rate(tag_rows, lambda _: True, "teacher_student_disagreement"),
            }
    return out


def catastrophic_tags(phenom: dict[str, dict[str, float]]) -> list[str]:
    bad: list[str] = []
    for tag, metrics in phenom.items():
        if tag == DIAGNOSTIC_TAG:
            continue
        if metrics["action_accuracy"] < 0.75 or metrics["false_commit_rate"] > 0.10:
            bad.append(tag)
    return bad


def learned_passes(metrics: dict[str, float], phenom: dict[str, dict[str, float]]) -> bool:
    return (
        metrics["action_accuracy_after_guard"] >= 0.90
        and metrics["heldout_scope_accuracy"] >= 0.85
        and metrics["heldout_negation_accuracy"] >= 0.90
        and metrics["heldout_correction_accuracy"] >= 0.85
        and metrics["false_commit_rate"] <= 0.05
        and metrics["keyword_trap_false_commit_rate"] <= 0.05
        and metrics["known_execute_accuracy"] >= 0.95
        and not catastrophic_tags(phenom)
    )


def final_verdicts(rows: list[dict[str, object]], agg: dict[str, dict[str, float]]) -> dict[str, list[str]]:
    by_tag = phenomenon_metrics(rows)
    out: dict[str, list[str]] = {}
    learned_positive = False
    teacher_pass = False
    for key, metrics in agg.items():
        model = key.split("+", 1)[0]
        phenom = by_tag[key]
        verdict: list[str] = []
        if model == "structured_rule_sensor_teacher":
            teacher_pass = learned_passes(metrics, phenom)
            verdict.append("TEACHER_REFERENCE_PASS" if teacher_pass else "TEACHER_REFERENCE_WEAK")
        elif model == "keyword_sensor":
            if metrics["keyword_trap_false_commit_rate"] > 0.05:
                verdict.append("DISTILL_SCOPE_WEAK")
            if metrics["false_commit_rate"] > 0.05:
                verdict.append("DISTILL_FALSE_COMMIT_HIGH")
            if metrics["heldout_negation_accuracy"] < 0.90:
                verdict.append("DISTILL_NEGATION_WEAK")
            if metrics["heldout_correction_accuracy"] < 0.85:
                verdict.append("DISTILL_CORRECTION_WEAK")
        else:
            if learned_passes(metrics, phenom):
                verdict.append("DISTILL_POSITIVE")
                learned_positive = True
            if metrics["heldout_surface_accuracy"] < 0.85:
                verdict.append("DISTILL_SURFACE_WEAK")
            if metrics["heldout_scope_accuracy"] < 0.85 or any(tag in catastrophic_tags(phenom) for tag in ["mention_trap", "substring_trap"]):
                verdict.append("DISTILL_SCOPE_WEAK")
            if metrics["heldout_negation_accuracy"] < 0.90 or "negation" in catastrophic_tags(phenom):
                verdict.append("DISTILL_NEGATION_WEAK")
            if metrics["heldout_correction_accuracy"] < 0.85 or "correction" in catastrophic_tags(phenom):
                verdict.append("DISTILL_CORRECTION_WEAK")
            if metrics["false_commit_rate"] > 0.05:
                verdict.append("DISTILL_FALSE_COMMIT_HIGH")
            if metrics["over_hold_rate_on_known"] > 0.05:
                verdict.append("DISTILL_OVERHOLD_HIGH")
            if metrics["strict_unseen_synonym_accuracy"] < 0.75:
                verdict.append("STRICT_UNSEEN_SYNONYM_UNSOLVED")
        out[key] = verdict or ["DISTILL_INCONCLUSIVE"]
    if teacher_pass and not learned_positive:
        for key in out:
            if key.startswith("word_ngram_linear_student+") or key.startswith("char_ngram_linear_student+"):
                if "DISTILL_RULE_TEACHER_ONLY" not in out[key]:
                    out[key].append("DISTILL_RULE_TEACHER_ONLY")
    return out


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_failures(failures: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for failure in failures:
            fh.write(json.dumps(failure, sort_keys=True) + "\n")


def metric_table(agg: dict[str, dict[str, float]]) -> list[str]:
    lines = [
        "| Model+Guard | Train | Main Action | Evidence Band | Disagree | Surface | Scope | Negation | Correction | False Commit | Trap False Commit | Strict Synonym |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, m in agg.items():
        lines.append(
            f"| `{key}` | `{m['train_action_accuracy']:.3f}` | `{m['action_accuracy_after_guard']:.3f}` "
            f"| `{m['evidence_band_accuracy']:.3f}` | `{m['teacher_student_disagreement_rate']:.3f}` "
            f"| `{m['heldout_surface_accuracy']:.3f}` | `{m['heldout_scope_accuracy']:.3f}` "
            f"| `{m['heldout_negation_accuracy']:.3f}` | `{m['heldout_correction_accuracy']:.3f}` "
            f"| `{m['false_commit_rate']:.3f}` | `{m['keyword_trap_false_commit_rate']:.3f}` "
            f"| `{m['strict_unseen_synonym_accuracy']:.3f}` |"
        )
    return lines


def phenomenon_table(phenom: dict[str, dict[str, dict[str, float]]]) -> list[str]:
    lines = [
        "| Model+Guard | Phenomenon | Action | False Commit | Evidence Band | Disagree |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for key, tag_map in phenom.items():
        for tag, m in tag_map.items():
            lines.append(
                f"| `{key}` | `{tag}` | `{m['action_accuracy']:.3f}` | `{m['false_commit_rate']:.3f}` "
                f"| `{m['evidence_band_accuracy']:.3f}` | `{m['teacher_student_disagreement_rate']:.3f}` |"
            )
    return lines


def minimal_pair_lines(rows: list[dict[str, object]]) -> list[str]:
    probes = [
        "add 3",
        "do not add 3",
        "do not add 3, multiply by 3 instead",
        "the word add appears in the note",
        "add 3. wait, actually multiply by 3",
    ]
    lines = [
        "| Probe | Model+Guard | Teacher Evidence | Student Evidence | Student Action |",
        "|---|---|---|---|---|",
    ]
    for text in probes:
        case = DistillCase("manual_minimal_pair", "manual", text, ("HOLD_ASK_RESEARCH",), "manual")
        teach = teacher_evidence(case)
        for model_name, evidence in [
            ("structured_rule_sensor_teacher+evidence_strength_margin_guard", teach),
            ("keyword_sensor+evidence_strength_margin_guard", keyword_evidence(case)),
        ]:
            lines.append(
                f"| `{text}` | `{model_name}` | `{[round(float(x), 3) for x in teach]}` "
                f"| `{[round(float(x), 3) for x in evidence]}` | `{action_for(evidence, 'evidence_strength_margin_guard')}` |"
            )
        for row in rows:
            if row["text"] == text and row["guard"] == "evidence_strength_margin_guard" and row["model_type"] == "learned":
                lines.append(
                    f"| `{text}` | `{row['model']}+{row['guard']}` | `{row['teacher_evidence']}` "
                    f"| `{row['student_evidence']}` | `{row['student_action']}` |"
                )
    return lines


def failure_summary_lines(failures: list[dict[str, object]], limit: int = 30) -> list[str]:
    if not failures:
        return ["No learned-student failures."]
    lines = []
    for failure in failures[:limit]:
        lines.append(
            f"- `{failure['model']}+{failure['guard']}` `{failure['split']}/{failure['phenomenon_tag']}`: "
            f"{failure['text']} -> expected `{failure['expected_action']}`, got `{failure['student_action']}` "
            f"({failure['failure_type']})."
        )
    if len(failures) > limit:
        lines.append(f"- ... {len(failures) - limit} more failures in `failure_examples.jsonl`.")
    return lines


def write_report(rows: list[dict[str, object]], failures: list[dict[str, object]], agg: dict[str, dict[str, float]], verdict: dict[str, list[str]], output_report: Path) -> None:
    phenom = phenomenon_metrics(rows)
    lines = [
        "# PILOT_SENSOR_DISTILL_001 Result",
        "",
        "## Goal",
        "",
        "Distill the hand-auditable `structured_rule_sensor` into learned text-to-evidence sensors while preserving the evidence bottleneck.",
        "",
        "This is not a natural-language-understanding claim. The downstream action is always produced by the fixed guard from predicted evidence.",
        "",
        "## Setup",
        "",
        "- Teacher: `structured_rule_sensor` from `PILOT_SENSOR_001`.",
        "- Baselines: `keyword_sensor`, `structured_rule_sensor_teacher`.",
        "- Students: `word_ngram_linear_student`, `char_ngram_linear_student`.",
        "- Targets: per-channel evidence bands `NONE`, `WEAK`, `STRONG`; no direct action head is used for verdict.",
        "- Guards: `evidence_strength_margin_guard`, `topK2_guard`; thresholds are unchanged from `PILOT_TOPK_GUARD_001`.",
        "",
        "## Context",
        "",
        "This probe uses an auditable evidence bottleneck: text is translated into intermediate evidence concepts, then a deterministic guard produces the action.",
        "",
        "Related reference points: concept bottleneck models, faithful translation-to-solver reasoning, and classification with reject/abstain options.",
        "",
        "- Concept Bottleneck Models: https://www.microsoft.com/en-us/research/publication/concept-bottleneck-models/",
        "- Faithful Chain-of-Thought Reasoning: https://huggingface.co/papers/2301.13379",
        "- Classification with reject option: https://www.sciencedirect.com/science/article/pii/S2666827025000477",
        "",
        "## Aggregate Metrics",
        "",
        *metric_table(agg),
        "",
        "## Phenomenon-Tagged Metrics",
        "",
        *phenomenon_table(phenom),
        "",
        "## Verdict",
        "",
        "```json",
        json.dumps(verdict, indent=2, sort_keys=True),
        "```",
        "",
        "## Minimal Pairs",
        "",
        *minimal_pair_lines(rows),
        "",
        "## Failure Examples",
        "",
        *failure_summary_lines(failures),
        "",
        "## Interpretation",
        "",
        "A positive learned-student result means the rule teacher's scope-aware evidence state is learnable on this controlled toy command language.",
        "",
        "If the teacher passes but learned students fail, the bottleneck remains learned scope-aware text-to-evidence extraction.",
        "",
        "The strict unseen synonym split is diagnostic only because this setup has no pretrained semantic knowledge.",
        "",
        "## Claim Boundary",
        "",
        "No general NLU, full PilotPulse learning, full VRAXION/INSTNCT proof, production architecture, or consciousness claim.",
    ]
    text = "\n".join(lines) + "\n"
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(text)
    DOC_REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC_REPORT.write_text(text)


def main() -> None:
    args = parse_args()
    rows, failures = evaluate(args)
    agg = aggregate(rows)
    verdict = final_verdicts(rows, agg)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.out_dir / "metrics.csv")
    write_failures(failures, args.out_dir / "failure_examples.jsonl")
    summary = {
        "aggregate": agg,
        "phenomenon_metrics": phenomenon_metrics(rows),
        "verdict": verdict,
        "failure_count": len(failures),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_report(rows, failures, agg, verdict, args.out_dir / "report.md")
    print(json.dumps({
        "verdict": verdict,
        "failure_count": len(failures),
        "metrics": str(args.out_dir / "metrics.csv"),
        "failures": str(args.out_dir / "failure_examples.jsonl"),
        "report": str(args.out_dir / "report.md"),
        "doc_report": str(DOC_REPORT),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
