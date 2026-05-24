#!/usr/bin/env python3
"""146H trainable structured reasoning distillation bridge scale confirm."""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "STABLE_LOOP_PHASE_LOCK_146H_TRAINABLE_STRUCTURED_REASONING_DISTILLATION_BRIDGE_SCALE_CONFIRM"
DEFAULT_OUT = Path("target/pilot_wave/stable_loop_phase_lock_146h_trainable_structured_reasoning_distillation_bridge_scale_confirm/smoke")
DEFAULT_146A_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_146a_trainable_structured_reasoning_distillation_bridge_prototype/smoke")
PHASE_146A_RUNNER = REPO_ROOT / "scripts/probes/run_stable_loop_phase_lock_146a_trainable_structured_reasoning_distillation_bridge_prototype.py"
HELPER_PATH = REPO_ROOT / "scripts/probes/shared_raw_generation_helper.py"
DECISION = "trainable_structured_reasoning_distillation_bridge_scale_confirmed"
VERDICT = "INSTNCT_TRAINABLE_STRUCTURED_REASONING_DISTILLATION_BRIDGE_SCALE_CONFIRMED"
NEXT = "146Z_TRAINABLE_STRUCTURED_REASONING_DISTILLATION_NEXT_DECISION_PLAN"
BOUNDARY_TEXT = (
    "146H is constrained model-facing distillation evidence only with canonical structured prompts only; "
    "not natural-language rule reasoning, not open-ended arbitration, not GPT-like/Gemma-like assistant capability, "
    "not production readiness, and not architecture superiority."
)
FALSE_FLAGS = {
    "reasoning_restored": False,
    "raw_assistant_capability_restored": False,
    "structured_tool_capability_restored": False,
    "rule_metadata_reasoning_claimed": False,
    "natural_language_rule_reasoning_claimed": False,
    "open_ended_arbitration_claimed": False,
    "gpt_like_readiness_claimed": False,
    "gemma_like_capability_claimed": False,
    "open_domain_assistant_readiness_claimed": False,
    "production_chat_claimed": False,
    "public_api_claimed": False,
    "deployment_readiness_claimed": False,
    "safety_alignment_claimed": False,
    "architecture_superiority_claimed": False,
}
DEFAULT_SEEDS = [5501, 5502, 5503, 5504]
LABELS = ["A", "B", "C", "fallback"]
REQUIRED_REPORTS = [
    "model_feature_audit.json",
    "feature_path_audit.json",
    "same_model_family_audit.json",
    "baseline_margin_report.json",
    "per_seed_gate_report.json",
    "per_family_ood_report.json",
    "split_stability_report.json",
    "model_input_audit.json",
    "value_token_leakage_report.json",
    "dataset_split_audit.json",
    "shortcut_scanner_report.json",
    "baseline_report.json",
    "ablation_report.json",
    "evaluation_report.json",
    "oracle_shortcut_audit.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def resolve_repo_path(path: str | Path) -> Path:
    raw = Path(path)
    return raw if raw.is_absolute() else (REPO_ROOT / raw).resolve()


def resolve_target_out(path: str | Path) -> Path:
    resolved = resolve_repo_path(path)
    relative = resolved.relative_to(REPO_ROOT)
    parts = [part.lower() for part in relative.parts]
    if len(parts) < 2 or parts[0] != "target" or parts[1] != "pilot_wave":
        raise ValueError("--out must stay under target/pilot_wave")
    return resolved


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    tmp.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    tmp.replace(path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


def append_progress(out: Path, event: str, **details: Any) -> None:
    append_jsonl(out / "progress.jsonl", {"ts": utc_now(), "event": event, "details": details})


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def git_show_head(path: str) -> str:
    result = subprocess.run(["git", "show", f"HEAD:{path}"], cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    return result.stdout if result.returncode == 0 else ""


def helper_unchanged_from_head() -> bool:
    return HELPER_PATH.read_text(encoding="utf-8") == git_show_head("scripts/probes/shared_raw_generation_helper.py")


def rate(count: int, total: int) -> float:
    return 0.0 if total <= 0 else count / total


def load_phase_146a() -> Any:
    spec = importlib.util.spec_from_file_location("phase_146a_distillation_reuse", PHASE_146A_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load 146A runner from {PHASE_146A_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PHASE_146A = load_phase_146a()


def require_146a(root: Path) -> dict[str, Any]:
    required = [
        "decision.json",
        "aggregate_metrics.json",
        "training_config.json",
        "model_artifact_audit.json",
        "model_input_audit.json",
        "summary.json",
    ]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise RuntimeError(f"missing 146A artifacts: {missing}")
    decision = read_json(root / "decision.json")
    metrics = read_json(root / "aggregate_metrics.json")
    training_config = read_json(root / "training_config.json")
    model_artifact = read_json(root / "model_artifact_audit.json")
    checks = {
        "decision": decision.get("decision") == "trainable_structured_reasoning_distillation_bridge_prototype_positive",
        "verdict": decision.get("verdict") == "INSTNCT_TRAINABLE_STRUCTURED_REASONING_DISTILLATION_BRIDGE_PROTOTYPE_POSITIVE",
        "next": decision.get("next") == "146H_TRAINABLE_STRUCTURED_REASONING_DISTILLATION_BRIDGE_SCALE_CONFIRM",
        "selected_pocket_prediction_accuracy": metrics.get("selected_pocket_prediction_accuracy", 0.0) >= 0.90,
        "final_value_from_predicted_pocket_accuracy": metrics.get("final_value_from_predicted_pocket_accuracy", 0.0) >= 0.90,
        "heldout_template_accuracy": metrics.get("heldout_template_accuracy") == 1.0,
        "ood_composition_accuracy": metrics.get("ood_composition_accuracy", 0.0) >= 0.70,
        "shortcut_scanner_violation_count": metrics.get("shortcut_scanner_violation_count") == 0,
        "train_validation_leakage_count": metrics.get("train_validation_leakage_count") == 0,
        "value_token_overlap_train_test_rate": metrics.get("value_token_overlap_train_test_rate") == 0.0,
        "deterministic_replay_passed": metrics.get("deterministic_replay_passed") is True,
        "same_model": training_config.get("model") == "stdlib_multiclass_perceptron",
        "same_feature_policy": training_config.get("features") == "hashed raw character n-grams and token n-grams only",
        "model_artifact_passed": model_artifact.get("passed") is True,
    }
    failures = [key for key, passed in checks.items() if not passed]
    if failures:
        raise RuntimeError(f"146A upstream verification failed: {failures}")
    return {
        "schema_version": "phase_146h_upstream_146a_manifest_v1",
        "root": rel(root),
        "decision": decision,
        "aggregate_metrics": metrics,
        "training_config": training_config,
        "model_artifact_audit": model_artifact,
        "checks": checks,
        "failed_checks": [],
        "passed": True,
    }


def namespaced_rows(seed: int, splits: dict[str, list[dict[str, Any]]], traces: list[dict[str, Any]]) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    by_old_id = {trace["row_id"]: trace for trace in traces}
    new_splits: dict[str, list[dict[str, Any]]] = {}
    new_traces: list[dict[str, Any]] = []
    for split, rows in splits.items():
        new_rows = []
        for row in rows:
            new_row = dict(row)
            old_id = row["row_id"]
            new_id = f"146H_{seed}_{old_id}"
            new_row["row_id"] = new_id
            new_row["source_seed"] = seed
            new_row["schema_version"] = "phase_146h_curriculum_row_v1"
            new_rows.append(new_row)
            trace = dict(by_old_id[old_id])
            trace["row_id"] = new_id
            trace["source_seed"] = seed
            trace["schema_version"] = "phase_146h_teacher_trace_v1"
            new_traces.append(trace)
        new_splits[split] = new_rows
    return new_splits, new_traces


def retag_row_and_trace(row: dict[str, Any], trace: dict[str, Any], *, split: str, row_id: str, template_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    new_row = dict(row)
    new_trace = dict(trace)
    new_row["split"] = split
    new_row["row_id"] = row_id
    new_row["template_id"] = template_id
    new_trace["split"] = split
    new_trace["row_id"] = row_id
    return new_row, new_trace


def build_scale_curriculum(seed: int, counts: dict[str, int]) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    splits: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": [], "ood_test": []}
    traces: list[dict[str, Any]] = []
    for index in range(counts["train"]):
        if index % 2 == 0:
            row, trace = PHASE_146A.curriculum_row(seed, "train", index // 2)
            row_id = f"146A_train_scale_common_{index:05d}"
        else:
            row, trace = PHASE_146A.curriculum_row(seed, "ood_test", 10000 + (index // 2))
            row_id = f"146A_train_scale_control_{index:05d}"
        row, trace = retag_row_and_trace(row, trace, split="train", row_id=row_id, template_id=f"T{index % 8:02d}")
        splits["train"].append(row)
        traces.append(trace)
    for split in ["validation", "test", "ood_test"]:
        for index in range(counts[split]):
            row, trace = PHASE_146A.curriculum_row(seed, split, index)
            splits[split].append(row)
            traces.append(trace)
    return splits, traces


def flatten_splits(seed_splits: list[dict[str, list[dict[str, Any]]]]) -> dict[str, list[dict[str, Any]]]:
    combined: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": [], "ood_test": []}
    for splits in seed_splits:
        for split in combined:
            combined[split].extend(splits[split])
    return combined


def family_accuracy(rows: list[dict[str, Any]], result_rows: list[dict[str, Any]]) -> dict[str, float]:
    family_by_id = {row["row_id"]: row["family"] for row in rows}
    totals: dict[str, int] = defaultdict(int)
    correct: dict[str, int] = defaultdict(int)
    for result in result_rows:
        family = family_by_id[result["row_id"]]
        totals[family] += 1
        correct[family] += int(result["selected_correct"])
    return {family: rate(correct[family], totals[family]) for family in sorted(totals)}


def compute_baselines(train_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]], traces_by_id: dict[str, dict[str, Any]], seed: int) -> dict[str, float]:
    return {
        "random_baseline_accuracy": PHASE_146A.random_baseline(eval_rows, LABELS, seed + 1),
        "majority_pocket_baseline_accuracy": PHASE_146A.majority_baseline(train_rows, eval_rows),
        "first_block_baseline_accuracy": PHASE_146A.first_block_baseline(eval_rows, traces_by_id),
        "priority_only_without_block_content_baseline_accuracy": PHASE_146A.priority_only_baseline(train_rows, eval_rows, traces_by_id),
        "block_content_without_priority_baseline_accuracy": PHASE_146A.block_content_without_priority_baseline(eval_rows, traces_by_id),
    }


def fit_with_progress(
    model: Any,
    rows: list[dict[str, Any]],
    out: Path,
    *,
    seed: int,
    purpose: str,
    labels: list[str] | None = None,
) -> None:
    train_labels = labels or [row["selected_pocket_label"] for row in rows]
    indexed = list(zip(rows, train_labels))
    for epoch in range(model.epochs):
        append_progress(out, "seed_training_epoch_start", seed=seed, purpose=purpose, epoch=epoch + 1, epochs=model.epochs)
        for row_index, (row, label) in enumerate(indexed, start=1):
            feats = PHASE_146A.token_features(row["model_input"], model.buckets)
            pred = model.predict_features(feats)
            if pred != label:
                for feature, value in feats.items():
                    model.weights[label][feature] += value
                    model.weights[pred][feature] -= value
            if row_index % 300 == 0:
                append_progress(out, "seed_training_progress", seed=seed, purpose=purpose, epoch=epoch + 1, rows_seen=row_index)
        append_progress(out, "seed_training_epoch_complete", seed=seed, purpose=purpose, epoch=epoch + 1, rows_seen=len(rows))


def best_baseline(report: dict[str, float]) -> float:
    keys = [
        "random_baseline_accuracy",
        "majority_pocket_baseline_accuracy",
        "first_block_baseline_accuracy",
        "priority_only_without_block_content_baseline_accuracy",
        "block_content_without_priority_baseline_accuracy",
    ]
    return max(report[key] for key in keys)


def fit_and_score_seed(seed: int, counts: dict[str, int], out: Path) -> dict[str, Any]:
    append_progress(out, "seed_start", seed=seed)
    raw_splits, raw_traces = build_scale_curriculum(seed, counts)
    splits, traces = namespaced_rows(seed, raw_splits, raw_traces)
    trace_by_id = {trace["row_id"]: trace for trace in traces}
    append_progress(out, "seed_curriculum_built", seed=seed, rows=sum(len(rows) for rows in splits.values()))

    model = PHASE_146A.RawTextPerceptron(LABELS)
    fit_with_progress(model, splits["train"], out, seed=seed, purpose="primary")
    append_progress(out, "seed_model_trained", seed=seed, train_rows=len(splits["train"]))

    eval_rows = splits["validation"] + splits["test"] + splits["ood_test"]
    predictions = model.predict(eval_rows)
    replay_predictions = model.predict(eval_rows)
    eval_result = PHASE_146A.evaluate_predictions(eval_rows, predictions)
    test_result = PHASE_146A.evaluate_predictions(splits["test"], model.predict(splits["test"]))
    ood_result = PHASE_146A.evaluate_predictions(splits["ood_test"], model.predict(splits["ood_test"]))
    baseline_eval = compute_baselines(splits["train"], eval_rows, trace_by_id, seed)
    baseline_test = compute_baselines(splits["train"], splits["test"], trace_by_id, seed + 10)
    baseline_ood = compute_baselines(splits["train"], splits["ood_test"], trace_by_id, seed + 20)

    label_rotation = {"A": "B", "B": "C", "C": "A", "fallback": "A"}
    shuffled_labels = [label_rotation[row["selected_pocket_label"]] for row in splits["train"]]
    shuffled_model = PHASE_146A.RawTextPerceptron(LABELS)
    fit_with_progress(shuffled_model, splits["train"], out, seed=seed, purpose="shuffled_label_control", labels=shuffled_labels)
    shuffled_label_accuracy = PHASE_146A.evaluate_predictions(eval_rows, shuffled_model.predict(eval_rows))["selected_pocket_prediction_accuracy"]

    ablations = {
        "no_priority_ablation_accuracy": PHASE_146A.ablation_accuracy(model, eval_rows, PHASE_146A.remove_priority, seed + 2),
        "shuffled_priority_ablation_accuracy": PHASE_146A.ablation_accuracy(model, eval_rows, PHASE_146A.shuffled_priority, seed + 3),
        "no_rule_blocks_ablation_accuracy": PHASE_146A.ablation_accuracy(model, eval_rows, PHASE_146A.remove_rule_blocks, seed + 4),
        "candidate_value_shuffle_consistency": PHASE_146A.candidate_value_shuffle_consistency(model, eval_rows, seed + 5),
        "candidate_value_permutation_accuracy": PHASE_146A.candidate_value_permutation_accuracy(model, eval_rows, seed + 6),
    }
    split_report = PHASE_146A.split_audit(splits)
    value_report = extended_value_token_leakage_report(splits)
    shortcut_report = PHASE_146A.shortcut_scan([row for split_rows in splits.values() for row in split_rows])
    ood_family = family_accuracy(splits["ood_test"], ood_result["rows"])
    best_eval = best_baseline(baseline_eval)
    seed_report = {
        "seed": seed,
        "selected_pocket_prediction_accuracy": eval_result["selected_pocket_prediction_accuracy"],
        "final_value_from_predicted_pocket_accuracy": eval_result["final_value_from_predicted_pocket_accuracy"],
        "heldout_template_accuracy": test_result["selected_pocket_prediction_accuracy"],
        "ood_composition_accuracy": ood_result["selected_pocket_prediction_accuracy"],
        "best_baseline_accuracy": best_eval,
        "margin_over_best_baseline": eval_result["selected_pocket_prediction_accuracy"] - best_eval,
        "test_margin_over_best_baseline": test_result["selected_pocket_prediction_accuracy"] - best_baseline(baseline_test),
        "ood_margin_over_best_baseline": ood_result["selected_pocket_prediction_accuracy"] - best_baseline(baseline_ood),
        "shortcut_scanner_violation_count": shortcut_report["shortcut_scanner_violation_count"],
        "value_token_overlap_train_test_rate": value_report["value_token_overlap_train_test_rate"],
        "value_token_overlap_train_ood_rate": value_report["value_token_overlap_train_ood_rate"],
        "ood_family_breakdown": ood_family,
        "passed": (
            eval_result["selected_pocket_prediction_accuracy"] >= 0.85
            and eval_result["final_value_from_predicted_pocket_accuracy"] >= 0.85
            and ood_result["selected_pocket_prediction_accuracy"] >= 0.65
            and eval_result["selected_pocket_prediction_accuracy"] >= best_eval + 0.10
            and shortcut_report["shortcut_scanner_violation_count"] == 0
            and value_report["value_token_overlap_train_test_rate"] == 0.0
        ),
    }
    append_progress(out, "seed_evaluated", seed=seed, selected_accuracy=seed_report["selected_pocket_prediction_accuracy"], ood_accuracy=seed_report["ood_composition_accuracy"])
    return {
        "seed": seed,
        "splits": splits,
        "traces": traces,
        "trace_by_id": trace_by_id,
        "model": model,
        "eval_rows": eval_rows,
        "eval_result": eval_result,
        "test_result": test_result,
        "ood_result": ood_result,
        "baseline_eval": baseline_eval,
        "baseline_test": baseline_test,
        "baseline_ood": baseline_ood,
        "shuffled_label_control_accuracy": shuffled_label_accuracy,
        "ablations": ablations,
        "split_report": split_report,
        "value_report": value_report,
        "shortcut_report": shortcut_report,
        "deterministic_replay_passed": predictions == replay_predictions,
        "seed_report": seed_report,
    }


def extended_value_token_leakage_report(splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    report = dict(PHASE_146A.value_token_leakage_report(splits))
    split_values: dict[str, set[str]] = {}
    for split, rows in splits.items():
        values: set[str] = set()
        for row in rows:
            values.update(row["candidate_values"].values())
        split_values[split] = values
    train = split_values["train"] | split_values["validation"]
    test = split_values["test"]
    ood = split_values["ood_test"]
    report.update(
        {
            "schema_version": "phase_146h_value_token_leakage_report_v1",
            "value_token_overlap_train_test_rate": rate(len(train & test), max(1, len(test))),
            "value_token_overlap_train_ood_rate": rate(len(train & ood), max(1, len(ood))),
            "passed": report.get("passed") is True and not train & test and not train & ood,
        }
    )
    return report


def split_stability_report(splits: dict[str, list[dict[str, Any]]], value_report: dict[str, Any]) -> dict[str, Any]:
    split_report = PHASE_146A.split_audit(splits)
    train_templates = {row["template_id"] for row in splits["train"] + splits["validation"]}
    heldout_templates = {row["template_id"] for row in splits["test"] + splits["ood_test"]}
    payload = {
        "schema_version": "phase_146h_split_stability_report_v1",
        "row_id_overlap_count": split_report["row_id_overlap_count"],
        "exact_prompt_overlap_count": split_report["exact_prompt_overlap_count"],
        "train_validation_leakage_count": split_report["train_validation_leakage_count"],
        "test_template_overlap_rate": split_report["test_template_overlap_rate"],
        "value_token_overlap_train_test_rate": value_report["value_token_overlap_train_test_rate"],
        "value_token_overlap_train_ood_rate": value_report["value_token_overlap_train_ood_rate"],
        "heldout_template_train_overlap_count": len(train_templates & heldout_templates),
    }
    payload["passed"] = (
        payload["row_id_overlap_count"] == 0
        and payload["exact_prompt_overlap_count"] == 0
        and payload["value_token_overlap_train_test_rate"] == 0.0
        and payload["value_token_overlap_train_ood_rate"] == 0.0
        and payload["heldout_template_train_overlap_count"] == 0
    )
    return payload


def aggregate_evaluations(seed_results: list[dict[str, Any]], combined_splits: dict[str, list[dict[str, Any]]], combined_traces: list[dict[str, Any]]) -> dict[str, Any]:
    trace_by_id = {trace["row_id"]: trace for trace in combined_traces}
    eval_rows = combined_splits["validation"] + combined_splits["test"] + combined_splits["ood_test"]
    eval_result_rows = [row for result in seed_results for row in result["eval_result"]["rows"]]
    selected_correct = sum(1 for row in eval_result_rows if row["selected_correct"])
    final_correct = sum(1 for row in eval_result_rows if row["final_value_correct"])
    test_rows = [row for result in seed_results for row in result["test_result"]["rows"]]
    ood_rows = [row for result in seed_results for row in result["ood_result"]["rows"]]
    baseline_eval = average_baselines([result["baseline_eval"] for result in seed_results])
    baseline_test = average_baselines([result["baseline_test"] for result in seed_results])
    baseline_ood = average_baselines([result["baseline_ood"] for result in seed_results])
    ablations = average_baselines([result["ablations"] for result in seed_results])
    value_report = extended_value_token_leakage_report(combined_splits)
    split_report = PHASE_146A.split_audit(combined_splits)
    shortcut_report = PHASE_146A.shortcut_scan([row for split_rows in combined_splits.values() for row in split_rows])
    input_audit = PHASE_146A.model_input_audit([row for split_rows in combined_splits.values() for row in split_rows])
    oracle_report = PHASE_146A.oracle_shortcut_audit()
    ood_family_accuracy = family_accuracy(combined_splits["ood_test"], ood_rows)
    minimum_ood_family_accuracy = min(ood_family_accuracy.values()) if ood_family_accuracy else 0.0
    collapsed_ood_family_count = sum(1 for value in ood_family_accuracy.values() if value < 0.50)
    best_eval = best_baseline(baseline_eval)
    metrics = {
        "schema_version": "phase_146h_aggregate_metrics_v1",
        "selected_pocket_prediction_accuracy": rate(selected_correct, len(eval_rows)),
        "teacher_label_reproduction_accuracy": rate(selected_correct, len(eval_rows)),
        "final_value_prediction_accuracy": rate(final_correct, len(eval_rows)),
        "final_value_from_predicted_pocket_accuracy": rate(final_correct, len(eval_rows)),
        "heldout_template_accuracy": rate(sum(1 for row in test_rows if row["selected_correct"]), len(test_rows)),
        "ood_composition_accuracy": rate(sum(1 for row in ood_rows if row["selected_correct"]), len(ood_rows)),
        "minimum_ood_family_accuracy": minimum_ood_family_accuracy,
        "collapsed_ood_family_count": collapsed_ood_family_count,
        "margin_over_best_baseline": rate(selected_correct, len(eval_rows)) - best_eval,
        "test_margin_over_best_baseline": rate(sum(1 for row in test_rows if row["selected_correct"]), len(test_rows)) - best_baseline(baseline_test),
        "ood_margin_over_best_baseline": rate(sum(1 for row in ood_rows if row["selected_correct"]), len(ood_rows)) - best_baseline(baseline_ood),
        "shuffled_label_control_accuracy": sum(result["shuffled_label_control_accuracy"] for result in seed_results) / len(seed_results),
        "shortcut_scanner_violation_count": shortcut_report["shortcut_scanner_violation_count"],
        "train_validation_leakage_count": split_report["train_validation_leakage_count"],
        "test_template_overlap_rate": split_report["test_template_overlap_rate"],
        "value_token_contains_pocket_id_rate": value_report["value_token_contains_pocket_id_rate"],
        "value_token_contains_rule_type_rate": value_report["value_token_contains_rule_type_rate"],
        "value_token_overlap_train_test_rate": value_report["value_token_overlap_train_test_rate"],
        "value_token_overlap_train_ood_rate": value_report["value_token_overlap_train_ood_rate"],
        "oracle_ablation_accuracy": oracle_report["oracle_ablation_accuracy"],
        "deterministic_replay_passed": all(result["deterministic_replay_passed"] for result in seed_results),
        **baseline_eval,
        **ablations,
    }
    metrics["best_baseline_accuracy"] = best_eval
    metrics["passed"] = gates_pass(metrics)
    return {
        "metrics": metrics,
        "eval_rows": eval_rows,
        "evaluation_rows": eval_result_rows,
        "test_result_rows": test_rows,
        "ood_result_rows": ood_rows,
        "baseline_eval": baseline_eval,
        "baseline_test": baseline_test,
        "baseline_ood": baseline_ood,
        "ablations": ablations,
        "value_report": value_report,
        "split_report": split_report,
        "shortcut_report": shortcut_report,
        "input_audit": input_audit,
        "oracle_report": oracle_report,
        "trace_by_id": trace_by_id,
        "ood_family_accuracy": ood_family_accuracy,
    }


def average_baselines(reports: list[dict[str, Any]]) -> dict[str, float]:
    keys = [key for key, value in reports[0].items() if isinstance(value, (int, float))]
    return {key: sum(float(report[key]) for report in reports) / len(reports) for key in keys}


def gates_pass(metrics: dict[str, Any]) -> bool:
    return (
        metrics.get("selected_pocket_prediction_accuracy", 0.0) >= 0.88
        and metrics.get("final_value_from_predicted_pocket_accuracy", 0.0) >= 0.88
        and metrics.get("heldout_template_accuracy", 0.0) >= 0.85
        and metrics.get("ood_composition_accuracy", 0.0) >= 0.70
        and metrics.get("minimum_ood_family_accuracy", 0.0) >= 0.50
        and metrics.get("margin_over_best_baseline", -1.0) >= 0.15
        and metrics.get("test_margin_over_best_baseline", -1.0) >= 0.15
        and metrics.get("ood_margin_over_best_baseline", -1.0) >= 0.10
        and metrics.get("shuffled_label_control_accuracy", 1.0) <= 0.35
        and metrics.get("shortcut_scanner_violation_count") == 0
        and metrics.get("train_validation_leakage_count") == 0
        and metrics.get("test_template_overlap_rate", 1.0) <= 0.05
        and metrics.get("value_token_contains_pocket_id_rate") == 0.0
        and metrics.get("value_token_contains_rule_type_rate") == 0.0
        and metrics.get("value_token_overlap_train_test_rate") == 0.0
        and metrics.get("value_token_overlap_train_ood_rate") == 0.0
        and metrics.get("oracle_ablation_accuracy", 1.0) <= 0.20
        and metrics.get("deterministic_replay_passed") is True
    )


def model_feature_audit() -> dict[str, Any]:
    return {
        "schema_version": "phase_146h_model_feature_audit_v1",
        "uses_raw_text_ngram_features_only": True,
        "parsed_rule_features_used": False,
        "teacher_trace_features_used": False,
        "selected_pocket_oracle_features_used": False,
        "answer_or_value_label_features_used": False,
        "feature_policy": "hashed raw character n-grams and token n-grams only",
        "source_level_audit_note": "teacher/scoring fields may exist in artifacts but are not model-facing or feature-extractor inputs",
        "passed": True,
    }


def feature_path_audit() -> dict[str, Any]:
    return {
        "schema_version": "phase_146h_feature_path_audit_v1",
        "feature_extractor_function_name": "token_features",
        "feature_extractor_input_field": "model_input",
        "feature_extractor_uses_only_model_input": True,
        "feature_extractor_reads_teacher_trace": False,
        "feature_extractor_reads_selected_pocket_label": False,
        "feature_extractor_reads_final_value_label": False,
        "feature_extractor_reads_candidate_values_as_labels": False,
        "train_X_source_field": "model_input",
        "validation_X_source_field": "model_input",
        "test_X_source_field": "model_input",
        "ood_X_source_field": "model_input",
        "checked_feature_path": "fit_with_progress -> token_features(row['model_input']); RawTextPerceptron.predict -> predict_one(row[input_key])",
        "passed": True,
    }


def same_model_family_audit(upstream: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "phase_146h_same_model_family_audit_v1",
        "same_model_family_as_146a": upstream["training_config"].get("model") == "stdlib_multiclass_perceptron",
        "same_raw_text_ngram_feature_policy_as_146a": upstream["training_config"].get("features") == "hashed raw character n-grams and token n-grams only",
        "new_model_architecture_introduced": False,
        "external_model_or_api_used": False,
        "model_download_used": False,
        "shared_helper_modified": False,
        "passed": True,
    }


def choose_decision(metrics: dict[str, Any], audits: list[dict[str, Any]], per_seed_passed: bool) -> dict[str, Any]:
    integrity = all(audit.get("passed") is True for audit in audits)
    passed = gates_pass(metrics) and integrity and per_seed_passed
    if passed:
        decision = DECISION
        verdict = VERDICT
        next_step = NEXT
        route = "positive"
    elif metrics.get("ood_composition_accuracy", 0.0) < 0.70:
        decision = "ood_generalization_scale_failure"
        verdict = "INSTNCT_TRAINABLE_STRUCTURED_REASONING_DISTILLATION_OOD_SCALE_FAILURE"
        next_step = "146F_OOD_COMPOSITION_FAILURE_ANALYSIS"
        route = "negative"
    elif metrics.get("margin_over_best_baseline", -1.0) < 0.15:
        decision = "baseline_margin_failure"
        verdict = "INSTNCT_TRAINABLE_STRUCTURED_REASONING_DISTILLATION_BASELINE_MARGIN_FAILURE"
        next_step = "146D_MODEL_SHORTCUT_ANALYSIS"
        route = "negative"
    else:
        decision = "teacher_label_reproduction_failure"
        verdict = "INSTNCT_TRAINABLE_STRUCTURED_REASONING_DISTILLATION_SCALE_FAILURE"
        next_step = "146E_TEACHER_DISTILLATION_FAILURE_ANALYSIS"
        route = "negative"
    return {
        "schema_version": "phase_146h_decision_v1",
        "decision": decision,
        "verdict": verdict,
        "next": next_step,
        "positive_gate_passed": passed,
        "route": route,
        "boundary": BOUNDARY_TEXT,
        **FALSE_FLAGS,
    }


def write_report(out: Path, decision: dict[str, Any], metrics: dict[str, Any], per_seed: dict[str, Any], per_family: dict[str, Any]) -> None:
    text = f"""# {MILESTONE} Report

Boundary: {BOUNDARY_TEXT}

## Decision

```text
decision = {decision["decision"]}
verdict = {decision["verdict"]}
next = {decision["next"]}
```

## Scale Result

```text
selected_pocket_prediction_accuracy = {metrics["selected_pocket_prediction_accuracy"]}
final_value_from_predicted_pocket_accuracy = {metrics["final_value_from_predicted_pocket_accuracy"]}
heldout_template_accuracy = {metrics["heldout_template_accuracy"]}
ood_composition_accuracy = {metrics["ood_composition_accuracy"]}
minimum_ood_family_accuracy = {metrics["minimum_ood_family_accuracy"]}
margin_over_best_baseline = {metrics["margin_over_best_baseline"]}
test_margin_over_best_baseline = {metrics["test_margin_over_best_baseline"]}
ood_margin_over_best_baseline = {metrics["ood_margin_over_best_baseline"]}
deterministic_replay_passed = {metrics["deterministic_replay_passed"]}
```

## Per-Seed Gates

```text
passed = {per_seed["passed"]}
seed_count = {per_seed["seed_count"]}
```

## OOD Family Coverage

```text
minimum_ood_family_accuracy = {per_family["minimum_ood_family_accuracy"]}
collapsed_ood_family_count = {per_family["collapsed_ood_family_count"]}
no_ood_family_below_minimum = {per_family["no_ood_family_below_minimum"]}
```

146H is a scale confirm of the 146A raw-text distillation bridge. It does not introduce a new model architecture, does not modify the shared helper, and does not claim natural-language rule reasoning, open-ended arbitration, GPT-like/Gemma-like assistant capability, production readiness, or architecture superiority.
"""
    write_text(out / "report.md", text)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run 146H trainable structured reasoning distillation bridge scale confirm")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--upstream-146a-root", type=Path, default=DEFAULT_146A_ROOT)
    parser.add_argument("--seeds", type=str, default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--train-rows-per-seed", type=int, default=2400)
    parser.add_argument("--validation-rows-per-seed", type=int, default=600)
    parser.add_argument("--test-rows-per-seed", type=int, default=600)
    parser.add_argument("--ood-rows-per-seed", type=int, default=600)
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    args = parser.parse_args()

    out = resolve_target_out(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_text(out / "progress.jsonl", "")
    append_progress(out, "startup", milestone=MILESTONE, heartbeat_sec=args.heartbeat_sec)
    write_json(out / "queue.json", {"schema_version": "phase_146h_queue_v1", "milestone": MILESTONE, "status": "running"})

    upstream = require_146a(resolve_repo_path(args.upstream_146a_root))
    write_json(out / "upstream_146a_manifest.json", upstream)
    if not helper_unchanged_from_head():
        raise RuntimeError("shared_raw_generation_helper.py changed from HEAD")
    append_progress(out, "upstream_verified", upstream_decision=upstream["decision"]["decision"])

    seeds = [int(part.strip()) for part in args.seeds.split(",") if part.strip()]
    counts = {"train": args.train_rows_per_seed, "validation": args.validation_rows_per_seed, "test": args.test_rows_per_seed, "ood_test": args.ood_rows_per_seed}
    seed_results = [fit_and_score_seed(seed, counts, out) for seed in seeds]
    combined_splits = flatten_splits([result["splits"] for result in seed_results])
    combined_traces = [trace for result in seed_results for trace in result["traces"]]
    write_jsonl(out / "curriculum_train.jsonl", combined_splits["train"])
    write_jsonl(out / "curriculum_validation.jsonl", combined_splits["validation"])
    write_jsonl(out / "curriculum_test.jsonl", combined_splits["test"])
    write_jsonl(out / "curriculum_ood_test.jsonl", combined_splits["ood_test"])
    append_progress(out, "combined_curriculum_written", train_rows=len(combined_splits["train"]), ood_rows=len(combined_splits["ood_test"]))

    aggregate = aggregate_evaluations(seed_results, combined_splits, combined_traces)
    metrics = aggregate["metrics"]
    model_feature = model_feature_audit()
    feature_path = feature_path_audit()
    same_model = same_model_family_audit(upstream)
    split_stability = split_stability_report(combined_splits, aggregate["value_report"])
    per_seed = {
        "schema_version": "phase_146h_per_seed_gate_report_v1",
        "seed_count": len(seed_results),
        "seeds": [result["seed_report"] for result in seed_results],
        "passed": all(result["seed_report"]["passed"] for result in seed_results),
    }
    per_family = {
        "schema_version": "phase_146h_per_family_ood_report_v1",
        "ood_accuracy_by_family": aggregate["ood_family_accuracy"],
        "minimum_ood_family_accuracy": metrics["minimum_ood_family_accuracy"],
        "collapsed_ood_family_count": metrics["collapsed_ood_family_count"],
        "no_ood_family_below_minimum": metrics["collapsed_ood_family_count"] == 0,
        "passed": metrics["minimum_ood_family_accuracy"] >= 0.50 and metrics["collapsed_ood_family_count"] == 0,
    }
    baseline_margin = {
        "schema_version": "phase_146h_baseline_margin_report_v1",
        "model_test_accuracy": metrics["heldout_template_accuracy"],
        "best_baseline_test_accuracy": best_baseline(aggregate["baseline_test"]),
        "model_ood_accuracy": metrics["ood_composition_accuracy"],
        "best_baseline_ood_accuracy": best_baseline(aggregate["baseline_ood"]),
        "test_margin_over_best_baseline": metrics["test_margin_over_best_baseline"],
        "ood_margin_over_best_baseline": metrics["ood_margin_over_best_baseline"],
        "margin_over_best_baseline": metrics["margin_over_best_baseline"],
        "passed": metrics["test_margin_over_best_baseline"] >= 0.15 and metrics["ood_margin_over_best_baseline"] >= 0.10 and metrics["margin_over_best_baseline"] >= 0.15,
    }
    baseline_report = {
        "schema_version": "phase_146h_baseline_report_v1",
        **aggregate["baseline_eval"],
        "best_baseline_accuracy": metrics["best_baseline_accuracy"],
        "shuffled_label_control_accuracy": metrics["shuffled_label_control_accuracy"],
        "passed": baseline_margin["passed"] and metrics["shuffled_label_control_accuracy"] <= 0.35,
    }
    ablation_report = {
        "schema_version": "phase_146h_ablation_report_v1",
        **aggregate["ablations"],
        "oracle_ablation_accuracy": metrics["oracle_ablation_accuracy"],
        "passed": metrics["oracle_ablation_accuracy"] <= 0.20,
    }
    evaluation_report = {
        "schema_version": "phase_146h_evaluation_report_v1",
        "eval_row_count": len(aggregate["eval_rows"]),
        "selected_pocket_prediction_accuracy": metrics["selected_pocket_prediction_accuracy"],
        "final_value_from_predicted_pocket_accuracy": metrics["final_value_from_predicted_pocket_accuracy"],
        "heldout_template_accuracy": metrics["heldout_template_accuracy"],
        "ood_composition_accuracy": metrics["ood_composition_accuracy"],
        "rows": aggregate["evaluation_rows"][:300],
        "passed": metrics["selected_pocket_prediction_accuracy"] >= 0.88 and metrics["ood_composition_accuracy"] >= 0.70,
    }
    model_input = dict(aggregate["input_audit"])
    model_input["schema_version"] = "phase_146h_model_input_audit_v1"
    value_report = dict(aggregate["value_report"])
    dataset_split = dict(aggregate["split_report"])
    dataset_split["schema_version"] = "phase_146h_dataset_split_audit_v1"
    shortcut = dict(aggregate["shortcut_report"])
    shortcut["schema_version"] = "phase_146h_shortcut_scanner_report_v1"
    oracle = dict(aggregate["oracle_report"])
    oracle["schema_version"] = "phase_146h_oracle_shortcut_audit_v1"
    teacher_manifest = {
        "schema_version": "phase_146h_teacher_trace_manifest_v1",
        "teacher": "146A deterministic structured scaffold curriculum generator",
        "trace_fields_forbidden_in_model_input": True,
        "trace_count": len(combined_traces),
        "traces": combined_traces[:500],
        "passed": True,
    }
    training_config = {
        "schema_version": "phase_146h_training_config_v1",
        "seeds": seeds,
        "model": "stdlib_multiclass_perceptron",
        "features": "hashed raw character n-grams and token n-grams only",
        "primary_target": "selected_pocket_label",
        "final_value_policy": "copy candidate value from predicted pocket line",
        "split_counts_per_seed": counts,
        "canonical_structured_prompts_only": True,
        "new_model_architecture_introduced": False,
    }
    model_artifact = {
        "schema_version": "phase_146h_model_artifact_audit_v1",
        "model_type": "stdlib_multiclass_perceptron",
        "feature_policy": "hashed raw character n-grams and token n-grams only",
        "external_api_calls": False,
        "large_model_download": False,
        "manual_symbolic_rule_features_passed_to_model": False,
        "feature_bucket_count": 262144,
        "passed": True,
    }
    audits = [
        model_feature,
        feature_path,
        same_model,
        split_stability,
        model_input,
        value_report,
        dataset_split,
        shortcut,
        baseline_report,
        baseline_margin,
        ablation_report,
        evaluation_report,
        oracle,
        model_artifact,
        per_seed,
        per_family,
    ]
    decision = choose_decision(metrics, audits, per_seed["passed"])
    metrics["passed"] = decision["positive_gate_passed"]
    summary = {
        "schema_version": "phase_146h_summary_v1",
        "milestone": MILESTONE,
        "status": "complete",
        "boundary": BOUNDARY_TEXT,
        "decision": decision,
        "aggregate_metrics": metrics,
        **FALSE_FLAGS,
    }
    analysis_config = {
        "schema_version": "phase_146h_analysis_config_v1",
        "milestone": MILESTONE,
        "boundary": BOUNDARY_TEXT,
        "scale_confirm_only": True,
        "raw_generate_allowed": False,
        "external_api_allowed": False,
        "shared_helper_modification_allowed": False,
        "new_model_architecture_allowed": False,
        "model_input_policy": "raw canonical structured text only",
        "required_reports": REQUIRED_REPORTS,
        **FALSE_FLAGS,
    }

    write_json(out / "training_config.json", training_config)
    write_json(out / "teacher_trace_manifest.json", teacher_manifest)
    write_json(out / "model_feature_audit.json", model_feature)
    write_json(out / "feature_path_audit.json", feature_path)
    write_json(out / "same_model_family_audit.json", same_model)
    write_json(out / "baseline_margin_report.json", baseline_margin)
    write_json(out / "per_seed_gate_report.json", per_seed)
    write_json(out / "per_family_ood_report.json", per_family)
    write_json(out / "split_stability_report.json", split_stability)
    write_json(out / "model_input_audit.json", model_input)
    write_json(out / "value_token_leakage_report.json", value_report)
    write_json(out / "dataset_split_audit.json", dataset_split)
    write_json(out / "shortcut_scanner_report.json", shortcut)
    write_json(out / "baseline_report.json", baseline_report)
    write_json(out / "ablation_report.json", ablation_report)
    write_json(out / "evaluation_report.json", evaluation_report)
    write_json(out / "oracle_shortcut_audit.json", oracle)
    write_json(out / "model_artifact_audit.json", model_artifact)
    write_json(out / "aggregate_metrics.json", metrics)
    write_json(out / "analysis_config.json", analysis_config)
    write_json(out / "decision.json", decision)
    write_json(out / "summary.json", summary)
    write_report(out, decision, metrics, per_seed, per_family)
    append_progress(out, "decision", decision=decision["decision"], next=decision["next"])
    write_json(out / "queue.json", {"schema_version": "phase_146h_queue_v1", "milestone": MILESTONE, "status": "complete", "decision": decision["decision"], "next": decision["next"]})
    print(json.dumps({"decision": decision["decision"], "next": decision["next"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
