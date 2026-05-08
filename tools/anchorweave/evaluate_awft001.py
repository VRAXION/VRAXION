#!/usr/bin/env python3
"""Evaluate AWFT-001 predictions against labels."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any


REQUIRED_FIELDS = [
    "scenario_id",
    "first_action",
    "high_salience",
    "low_salience",
    "symbol_attach",
    "symbol_reject",
    "counterfactual_answers",
    "commitment_level",
    "claim_boundary",
]

OVERCOMMIT_VALUES = {"confirmed_same_place", "premature_commit"}


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate AWFT-001 predictions.")
    parser.add_argument("--labels", required=True, type=Path, help="Path to eval_labels.jsonl.")
    parser.add_argument("--predictions", required=True, type=Path, help="Path to predictions JSONL.")
    return parser.parse_args(argv)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: row must be a JSON object")
            rows.append(row)
    return rows


def index_rows(rows: list[dict[str, Any]], label: str) -> tuple[dict[str, dict[str, Any]], list[str]]:
    indexed: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for row in rows:
        scenario_id = row.get("scenario_id")
        if not isinstance(scenario_id, str) or not scenario_id:
            raise ValueError(f"{label}: row missing non-empty scenario_id")
        if scenario_id in indexed:
            duplicates.append(scenario_id)
            continue
        indexed[scenario_id] = row
    return indexed, duplicates


def as_set(row: dict[str, Any] | None, field: str) -> set[str]:
    if row is None:
        return set()
    value = row.get(field, [])
    if not isinstance(value, list):
        return set()
    return {item for item in value if isinstance(item, str)}


def f1_from_counts(tp: int, pred_total: int, gold_total: int) -> float:
    if pred_total == 0 and gold_total == 0:
        return 1.0
    if pred_total == 0 or gold_total == 0 or tp == 0:
        return 0.0
    precision = tp / pred_total
    recall = tp / gold_total
    return 2 * precision * recall / (precision + recall)


def micro_f1(labels: list[dict[str, Any]], predictions: dict[str, dict[str, Any]], field: str) -> float:
    true_positive = 0
    pred_total = 0
    gold_total = 0
    for label in labels:
        scenario_id = label["scenario_id"]
        pred = predictions.get(scenario_id)
        gold_set = as_set(label, field)
        pred_set = as_set(pred, field)
        true_positive += len(gold_set & pred_set)
        pred_total += len(pred_set)
        gold_total += len(gold_set)
    return f1_from_counts(true_positive, pred_total, gold_total)


def accuracy(labels: list[dict[str, Any]], predictions: dict[str, dict[str, Any]], field: str) -> float:
    if not labels:
        return 0.0
    correct = 0
    for label in labels:
        scenario_id = label["scenario_id"]
        pred = predictions.get(scenario_id)
        if pred is not None and pred.get(field) == label.get(field):
            correct += 1
    return correct / len(labels)


def counterfactual_accuracy(
    labels: list[dict[str, Any]], predictions: dict[str, dict[str, Any]]
) -> float:
    total = 0
    correct = 0
    for label in labels:
        scenario_id = label["scenario_id"]
        gold_answers = label.get("counterfactual_answers", {})
        pred = predictions.get(scenario_id)
        pred_answers = pred.get("counterfactual_answers", {}) if isinstance(pred, dict) else {}
        if not isinstance(gold_answers, dict):
            continue
        if not isinstance(pred_answers, dict):
            pred_answers = {}
        for cf_id, gold_value in gold_answers.items():
            total += 1
            if pred_answers.get(cf_id) == gold_value:
                correct += 1
    return correct / total if total else 0.0


def overcommitment_rate(
    labels: list[dict[str, Any]], predictions: dict[str, dict[str, Any]]
) -> float:
    if not labels:
        return 0.0
    overcommit = 0
    for label in labels:
        scenario_id = label["scenario_id"]
        pred = predictions.get(scenario_id)
        pred_commitment = pred.get("commitment_level") if isinstance(pred, dict) else None
        gold_commitment = label.get("commitment_level")
        if pred_commitment in OVERCOMMIT_VALUES and gold_commitment != "confirmed_same_place":
            overcommit += 1
    return overcommit / len(labels)


def validate_label_rows(labels: list[dict[str, Any]]) -> None:
    for row in labels:
        missing = [field for field in REQUIRED_FIELDS if field not in row]
        if missing:
            raise ValueError(f"label {row.get('scenario_id', '<unknown>')} missing fields: {missing}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    try:
        labels = read_jsonl(args.labels)
        prediction_rows = read_jsonl(args.predictions)
        validate_label_rows(labels)
        label_index, label_duplicates = index_rows(labels, "labels")
        if label_duplicates:
            raise ValueError(f"duplicate scenario_id in labels: {sorted(set(label_duplicates))}")
        predictions, prediction_duplicates = index_rows(prediction_rows, "predictions")
        if prediction_duplicates:
            raise ValueError(
                f"duplicate scenario_id in predictions: {sorted(set(prediction_duplicates))}"
            )

        label_ids = set(label_index)
        prediction_ids = set(predictions)
        missing_ids = sorted(label_ids - prediction_ids)
        extra_ids = sorted(prediction_ids - label_ids)

        ordered_labels = [label_index[scenario_id] for scenario_id in sorted(label_index)]
        results = {
            "first_action_accuracy": accuracy(ordered_labels, predictions, "first_action"),
            "salience_high_f1": micro_f1(ordered_labels, predictions, "high_salience"),
            "salience_low_f1": micro_f1(ordered_labels, predictions, "low_salience"),
            "symbol_attach_f1": micro_f1(ordered_labels, predictions, "symbol_attach"),
            "symbol_reject_f1": micro_f1(ordered_labels, predictions, "symbol_reject"),
            "counterfactual_accuracy": counterfactual_accuracy(ordered_labels, predictions),
            "commitment_accuracy": accuracy(ordered_labels, predictions, "commitment_level"),
            "overcommitment_rate": overcommitment_rate(ordered_labels, predictions),
            "counts": {
                "evaluated": len(ordered_labels),
                "missing_predictions": len(missing_ids),
                "extra_predictions": len(extra_ids),
                "duplicate_predictions": len(prediction_duplicates),
            },
        }

        if missing_ids:
            print(f"warning: missing predictions: {', '.join(missing_ids)}", file=sys.stderr)
        if extra_ids:
            print(f"warning: extra predictions ignored: {', '.join(extra_ids)}", file=sys.stderr)

        print(json.dumps(results, ensure_ascii=True, indent=2, sort_keys=True))
        print("AWFT-001 evaluation summary")
        for key, value in results.items():
            if key == "counts":
                continue
            print(f"{key}: {value:.6f}")
        counts = results["counts"]
        print(
            "counts: "
            f"evaluated={counts['evaluated']} "
            f"missing={counts['missing_predictions']} "
            f"extra={counts['extra_predictions']} "
            f"duplicate={counts['duplicate_predictions']}"
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
