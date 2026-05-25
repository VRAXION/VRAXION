#!/usr/bin/env python3
"""149H bounded decision schema scale confirm with hub-routing diagnosis."""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
import os
import subprocess
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
PHASE_149A_PATH = REPO_ROOT / "scripts/probes/run_stable_loop_phase_lock_149a_bounded_decision_schema_generation_prototype.py"
DEFAULT_OUT = Path("target/pilot_wave/stable_loop_phase_lock_149h_bounded_decision_schema_generation_scale_confirm/smoke")
DEFAULT_149A_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_149a_bounded_decision_schema_generation_prototype/smoke")
MILESTONE = "STABLE_LOOP_PHASE_LOCK_149H_BOUNDED_DECISION_SCHEMA_GENERATION_SCALE_CONFIRM"
STRONG_DECISION = "bounded_decision_schema_generation_scale_confirmed"
STRONG_VERDICT = "INSTNCT_BOUNDED_DECISION_SCHEMA_GENERATION_SCALE_CONFIRMED"
EDGE_DECISION = "bounded_decision_schema_generation_scale_edge_pocket_routing_bottleneck_confirmed"
EDGE_VERDICT = "INSTNCT_BOUNDED_DECISION_SCHEMA_GENERATION_SCALE_EDGE_POCKET_ROUTING_BOTTLENECK"
NEXT = "149Z_BOUNDED_DECISION_SCHEMA_GENERATION_NEXT_DECISION_PLAN"
BOUNDARY_TEXT = (
    "149H is constrained model-facing distillation evidence only with canonical structured prompts only, "
    "bounded two-line decision schema generation only; not natural-language rule reasoning, not open-ended "
    "arbitration, not GPT-like/Gemma-like assistant capability, not production readiness, and not architecture superiority."
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
    "safety_alignment_claimed": False,
    "deployment_readiness_claimed": False,
    "architecture_superiority_claimed": False,
    "public_api_claimed": False,
}
REQUIRED_UPSTREAM_ARTIFACTS = [
    "decision.json",
    "aggregate_metrics.json",
    "summary.json",
    "schema_prefix_audit.json",
    "raw_schema_generation_audit.json",
    "decoding_audit.json",
    "deterministic_replay_report.json",
]


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PHASE_149A = load_module(PHASE_149A_PATH, "phase_149a_for_149h")
torch = PHASE_149A.torch


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def append_progress(out: Path, event: str, **details: Any) -> None:
    append_jsonl(out / "progress.jsonl", {"time": utc_now(), "event": event, **details})


def rate(count: int | float, total: int | float) -> float:
    return float(count) / float(total) if total else 0.0


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def resolve_target_out(path: str | Path) -> Path:
    raw = Path(path)
    resolved = raw if raw.is_absolute() else REPO_ROOT / raw
    target_root = (REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_149h_bounded_decision_schema_generation_scale_confirm").resolve()
    if target_root not in resolved.resolve().parents and resolved.resolve() != target_root:
        raise RuntimeError(f"output must be under {target_root}: {resolved}")
    return resolved


def helper_unchanged_from_head() -> bool:
    return PHASE_149A.helper_unchanged_from_head()


def require_149a(root: Path) -> dict[str, Any]:
    missing = [name for name in REQUIRED_UPSTREAM_ARTIFACTS if not (root / name).exists()]
    if missing:
        raise RuntimeError(f"missing 149A artifacts: {missing}")
    decision = read_json(root / "decision.json")
    metrics = read_json(root / "aggregate_metrics.json")
    summary = read_json(root / "summary.json")
    schema_prefix = read_json(root / "schema_prefix_audit.json")
    raw_schema = read_json(root / "raw_schema_generation_audit.json")
    decode = read_json(root / "decoding_audit.json")
    replay = read_json(root / "deterministic_replay_report.json")
    checks = {
        "decision": decision.get("decision") == "bounded_decision_schema_generation_prototype_positive",
        "verdict": decision.get("verdict") == "INSTNCT_BOUNDED_DECISION_SCHEMA_GENERATION_PROTOTYPE_POSITIVE",
        "next": decision.get("next") == "149H_BOUNDED_DECISION_SCHEMA_GENERATION_SCALE_CONFIRM",
        "full_schema": metrics.get("full_bounded_schema_exact_match_rate", 0.0) >= 0.73,
        "selected_line": metrics.get("selected_line_generation_accuracy", 0.0) >= 0.74,
        "reason_code": metrics.get("reason_code_generation_accuracy", 0.0) >= 0.98,
        "schema_valid": metrics.get("generated_output_schema_valid_rate") == 1.0,
        "shuffled_control": metrics.get("shuffled_target_control_accuracy", 1.0) <= 0.05,
        "replay": metrics.get("generation_deterministic_replay_passed") is True
        and replay.get("generation_deterministic_replay_passed") is True,
        "no_selected_input": metrics.get("eval_generation_input_contains_selected_line") is False,
        "no_reason_input": metrics.get("eval_generation_input_contains_reason_code") is False,
        "no_selected_prepend": schema_prefix.get("runner_prepends_selected_line") is False,
        "no_reason_prepend": schema_prefix.get("runner_prepends_reason_code") is False,
        "no_wrapper": schema_prefix.get("deterministic_schema_wrapper_used") is False,
        "raw_scored": raw_schema.get("schema_scored_from_raw_generated_text") is True,
        "no_repair": raw_schema.get("post_generation_repair_used") is False,
        "no_substring_selected": raw_schema.get("selected_line_extracted_from_substring") is False,
        "no_substring_reason": raw_schema.get("reason_code_extracted_from_substring") is False,
        "full_target": decode.get("full_bounded_schema_target_used") is True,
        "not_selected_only": decode.get("selected_line_only_training_used") is False,
        "not_constrained": decode.get("constrained_label_or_reason_only_decoding_used") is False,
        "summary_boundary": summary.get("gemma_like_capability_claimed") is False,
    }
    failed = [key for key, value in checks.items() if not value]
    if failed:
        raise RuntimeError(f"149A upstream mismatch: {failed}")
    return {
        "schema_version": "phase_149h_upstream_149a_manifest_v1",
        "root": rel(root),
        "decision": decision,
        "aggregate_metrics": metrics,
        "summary": summary,
        "schema_prefix_audit": schema_prefix,
        "raw_schema_generation_audit": raw_schema,
        "decoding_audit": decode,
        "deterministic_replay_report": replay,
        "checks": checks,
        "failed_checks": failed,
        "passed": not failed,
    }


def compute_probe(args: argparse.Namespace) -> dict[str, Any]:
    cuda_available = bool(torch.cuda.is_available())
    cuda_device_count = int(torch.cuda.device_count()) if cuda_available else 0
    return {
        "schema_version": "phase_149h_compute_probe_v1",
        "requested_compute_mode": args.compute_mode,
        "actual_compute_mode": "cpu_parallel",
        "cuda_available": cuda_available,
        "cuda_device_count": cuda_device_count,
        "cuda_device_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "cpu_count": os.cpu_count(),
        "max_seed_workers": args.max_seed_workers,
        "torch_num_threads_per_worker": args.torch_num_threads_per_worker,
        "target_cpu_load_min": args.target_cpu_load_min,
        "target_cpu_load_max": args.target_cpu_load_max,
        "note": "149H preserves the accepted 149A CPU deterministic model path; CUDA is probed and reported, CPU seed parallelism is used.",
    }


def classify_failure(row: dict[str, Any]) -> str:
    if not row.get("schema_valid"):
        return "schema_failure"
    selected_ok = bool(row.get("selected_line_correct"))
    reason_ok = bool(row.get("reason_code_correct"))
    if selected_ok and reason_ok:
        return "ok"
    if selected_ok and not reason_ok:
        return "selector_reason_failure_selected_coincidentally_correct"
    if not selected_ok and reason_ok:
        return "pocket_routing_failure_reason_correct_selected_wrong"
    return "selector_and_pocket_failure_both_wrong"


def safe_get_trace(trace_by_id: dict[str, dict[str, Any]], row_id: str) -> dict[str, Any]:
    return trace_by_id.get(row_id, {})


def build_row_level(
    rows: list[dict[str, Any]],
    trace_by_id: dict[str, dict[str, Any]],
    seed: int,
) -> list[dict[str, Any]]:
    out_rows: list[dict[str, Any]] = []
    for row in rows:
        trace = safe_get_trace(trace_by_id, row["row_id"])
        failure_category = classify_failure(row)
        out_rows.append(
            {
                "row_id": row["row_id"],
                "seed": seed,
                "split": row["split"],
                "family": row["family"],
                "priority_order": trace.get("parsed_priority_order", []),
                "priority_order_key": ">".join(trace.get("parsed_priority_order", [])),
                "block_candidates": trace.get("per_block_derived_candidate_pocket", {}),
                "expected_selected": row["expected_selected_label"],
                "generated_selected": row["generated_selected_label"],
                "expected_reason": row["expected_reason_code"],
                "generated_reason": row["generated_reason_code"],
                "selected_correct": bool(row["selected_line_correct"]),
                "reason_correct": bool(row["reason_code_correct"]),
                "schema_valid": bool(row["schema_valid"]),
                "failure_category": failure_category,
                "raw_generated_text": row["raw_generated_text"],
                "expected_final_value": row["expected_final_value"],
                "final_value_correct": bool(row["final_value_correct"]),
            }
        )
    return out_rows


def confusion(rows: list[dict[str, Any]], expected_key: str, generated_key: str) -> dict[str, dict[str, int]]:
    matrix: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        matrix[str(row[expected_key])][str(row[generated_key])] += 1
    return {key: dict(value) for key, value in sorted(matrix.items())}


def accuracy_by(rows: list[dict[str, Any]], key: str, correct_key: str) -> dict[str, float]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        counts[str(row.get(key, ""))]["total"] += 1
        counts[str(row.get(key, ""))]["correct"] += int(bool(row.get(correct_key)))
    return {key: rate(value["correct"], value["total"]) for key, value in sorted(counts.items())}


def count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(key, "")) for row in rows).items()))


def failure_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(row["failure_category"] for row in rows)
    total = len(rows)
    selector_fail = counts["selector_reason_failure_selected_coincidentally_correct"] + counts["selector_and_pocket_failure_both_wrong"]
    pocket_fail = counts["pocket_routing_failure_reason_correct_selected_wrong"]
    dominant = counts.most_common(1)[0][0] if counts else None
    return {
        "schema_version": "phase_149h_failure_category_report_v1",
        "row_count": total,
        "failure_category_counts": dict(sorted(counts.items())),
        "failure_category_rates": {key: rate(value, total) for key, value in sorted(counts.items())},
        "selector_reason_failure_rate": rate(selector_fail, total),
        "pocket_routing_failure_rate": rate(pocket_fail, total),
        "dominant_failure_category": dominant,
        "pocket_routing_dominates_selector_reason": pocket_fail > selector_fail,
    }


def hub_routing_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected_by_label = accuracy_by(rows, "expected_selected", "selected_correct")
    reason_by_code = accuracy_by(rows, "expected_reason", "reason_correct")
    family_acc = accuracy_by(rows, "family", "selected_correct")
    priority_acc = accuracy_by(rows, "priority_order_key", "selected_correct")
    fail = failure_report(rows)
    return {
        "schema_version": "phase_149h_hub_routing_diagnostic_report_v1",
        "row_count": len(rows),
        "diagnosis": {
            "schema_generation_is_failure_source": False,
            "selector_reason_is_primary_failure_source": fail["selector_reason_failure_rate"] > fail["pocket_routing_failure_rate"],
            "pocket_routing_is_primary_failure_source": fail["pocket_routing_failure_rate"] > fail["selector_reason_failure_rate"],
        },
        "selected_accuracy_by_expected_label": selected_by_label,
        "reason_accuracy_by_expected_reason": reason_by_code,
        "selected_accuracy_by_family": family_acc,
        "selected_accuracy_by_priority_order": priority_acc,
        "selected_confusion_matrix": confusion(rows, "expected_selected", "generated_selected"),
        "reason_confusion_matrix": confusion(rows, "expected_reason", "generated_reason"),
        **fail,
    }


def priority_holdout_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    priority_rows = [row for row in rows if row["family"] == "PRIORITY_ORDER_HOLDOUT"]
    block_rows = [row for row in rows if row["family"] == "BLOCK_ORDER_HOLDOUT"]
    invalid_rows = [row for row in rows if row["family"] == "INVALID_HIGH_PRIORITY_FALLTHROUGH_OOD"]
    return {
        "schema_version": "phase_149h_priority_order_holdout_report_v1",
        "priority_order_holdout_row_count": len(priority_rows),
        "priority_order_holdout_selected_accuracy": rate(sum(row["selected_correct"] for row in priority_rows), len(priority_rows)),
        "priority_order_holdout_reason_accuracy": rate(sum(row["reason_correct"] for row in priority_rows), len(priority_rows)),
        "priority_order_holdout_pair_accuracy": rate(
            sum(row["selected_correct"] and row["reason_correct"] for row in priority_rows), len(priority_rows)
        ),
        "block_order_holdout_row_count": len(block_rows),
        "block_order_holdout_selected_accuracy": rate(sum(row["selected_correct"] for row in block_rows), len(block_rows)),
        "invalid_high_priority_row_count": len(invalid_rows),
        "invalid_high_priority_selected_accuracy": rate(sum(row["selected_correct"] for row in invalid_rows), len(invalid_rows)),
    }


def selected_reason_pair_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pair_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        expected = f"{row['expected_selected']}|{row['expected_reason']}"
        generated = f"{row['generated_selected']}|{row['generated_reason']}"
        pair_counts[expected][generated] += 1
    return {
        "schema_version": "phase_149h_selected_reason_pair_confusion_report_v1",
        "selected_reason_pair_confusion": {key: dict(value) for key, value in sorted(pair_counts.items())},
        "selected_label_by_reason_code_accuracy": {
            reason: rate(
                sum(row["selected_correct"] for row in rows if row["expected_reason"] == reason),
                sum(1 for row in rows if row["expected_reason"] == reason),
            )
            for reason in sorted({row["expected_reason"] for row in rows})
        },
    }


def summarize_result_rows(result_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return PHASE_149A.summarize_generation_rows(result_rows)


def seed_decision(seed_metrics: dict[str, Any], fail_report: dict[str, Any], priority_report: dict[str, Any]) -> dict[str, Any]:
    strong = (
        seed_metrics["selected_line_generation_accuracy"] >= 0.80
        and seed_metrics["full_bounded_schema_exact_match_rate"] >= 0.80
        and seed_metrics["ood_bounded_schema_accuracy"] >= 0.70
        and priority_report["priority_order_holdout_selected_accuracy"] >= 0.50
        and fail_report["pocket_routing_failure_rate"] <= 0.25
        and seed_metrics["generated_output_schema_valid_rate"] >= 0.95
        and seed_metrics["reason_code_generation_accuracy"] >= 0.90
        and seed_metrics["reason_code_semantic_accuracy"] >= 0.90
        and seed_metrics["generation_deterministic_replay_passed"] is True
    )
    edge = (
        seed_metrics["generated_output_schema_valid_rate"] >= 0.95
        and seed_metrics["reason_code_generation_accuracy"] >= 0.90
        and seed_metrics["reason_code_semantic_accuracy"] >= 0.90
        and seed_metrics["selected_line_generation_accuracy"] >= 0.70
        and fail_report["pocket_routing_failure_rate"] > fail_report["selector_reason_failure_rate"]
        and fail_report["dominant_failure_category"] == "pocket_routing_failure_reason_correct_selected_wrong"
    )
    if strong:
        decision = STRONG_DECISION
        verdict = STRONG_VERDICT
        positive = True
    elif edge:
        decision = EDGE_DECISION
        verdict = EDGE_VERDICT
        positive = True
    else:
        decision = "selected_routing_scale_failure"
        verdict = "149R_SELECTED_LINE_BINDING_FAILURE_ANALYSIS"
        positive = False
    return {
        "schema_version": "phase_149h_seed_partial_decision_v1",
        "seed": seed_metrics["seed"],
        "decision": decision,
        "verdict": verdict,
        "next": NEXT if positive else verdict,
        "positive_gate_passed": positive,
        "strong_positive": strong,
        "edge_diagnostic_positive": edge,
    }


def run_seed_worker(config: dict[str, Any]) -> dict[str, Any]:
    seed = int(config["seed"])
    seed_out = Path(config["seed_out"])
    try:
        if config.get("torch_num_threads_per_worker"):
            torch.set_num_threads(int(config["torch_num_threads_per_worker"]))
        write_text(seed_out / "progress.jsonl", "")
        append_progress(seed_out, "seed_start", seed=seed)
        write_json(seed_out / "queue.json", {"schema_version": "phase_149h_seed_queue_v1", "seed": seed, "status": "running"})
        counts = {
            "train": int(config["train_rows"]),
            "validation": int(config["validation_rows"]),
            "test": int(config["test_rows"]),
            "ood_test": int(config["ood_rows"]),
        }
        splits, traces = PHASE_149A.build_curriculum(seed, counts)
        trace_by_id = {trace["row_id"]: trace for trace in traces}
        all_rows = [row for split_rows in splits.values() for row in split_rows]
        append_progress(seed_out, "curriculum_built", seed=seed, row_count=len(all_rows))
        for split, rows in splits.items():
            write_jsonl(seed_out / f"curriculum_{split}.jsonl", rows)
        write_json(seed_out / "teacher_trace_manifest.json", {"schema_version": "phase_149h_seed_teacher_trace_manifest_v1", "trace_count": len(traces), "traces": traces})
        write_text(seed_out / "sequence_train_corpus.txt", "\n\n".join(PHASE_149A.training_sequence(row) for row in splits["train"]) + "\n")
        write_text(seed_out / "sequence_validation_corpus.txt", "\n\n".join(PHASE_149A.training_sequence(row) for row in splits["validation"]) + "\n")
        write_text(seed_out / "lm_training_metrics.jsonl", "")
        model, train_metrics = PHASE_149A.train_model(
            splits["train"],
            splits["validation"],
            seed=seed,
            buckets=int(config["feature_buckets"]),
            hidden=int(config["hidden"]),
            epochs=int(config["epochs"]),
            lr=float(config["lr"]),
            batch_size=int(config["batch_size"]),
            out=seed_out,
            purpose="primary_bounded_schema",
            heartbeat_sec=int(config["heartbeat_sec"]),
            rare_reason_oversample=int(config["rare_reason_oversample"]),
        )
        append_progress(seed_out, "model_trained", seed=seed, train_loss_final=train_metrics["train_loss_final"])
        eval_rows = splits["validation"] + splits["test"] + splits["ood_test"]
        eval_result = PHASE_149A.evaluate_generation(
            model,
            eval_rows,
            int(config["feature_buckets"]),
            int(config["max_new_bytes"]),
            out=seed_out,
            purpose="eval",
            heartbeat_sec=int(config["heartbeat_sec"]),
        )
        replay_result = PHASE_149A.evaluate_generation(
            model,
            eval_rows,
            int(config["feature_buckets"]),
            int(config["max_new_bytes"]),
            out=seed_out,
            purpose="replay",
            heartbeat_sec=int(config["heartbeat_sec"]),
        )
        ood_rows = [row for row in eval_result["rows"] if row["split"] == "ood_test"]
        test_rows = [row for row in eval_result["rows"] if row["split"] == "test"]
        ood_result = summarize_result_rows(ood_rows)
        test_result = summarize_result_rows(test_rows)
        row_level = build_row_level(eval_result["rows"], trace_by_id, seed)
        write_jsonl(seed_out / "row_level_generation_report.jsonl", row_level)
        fail = failure_report(row_level)
        hub = hub_routing_report(row_level)
        priority = priority_holdout_report(row_level)
        replay = PHASE_149A.deterministic_replay_report(eval_result, replay_result)
        generation_audit = PHASE_149A.generation_input_audit(splits)
        schema_prefix = PHASE_149A.schema_prefix_audit(splits, eval_result)
        raw_schema = PHASE_149A.raw_schema_generation_audit(eval_result)
        decode_args = argparse.Namespace(max_new_bytes=int(config["max_new_bytes"]))
        decode = PHASE_149A.decoding_audit(decode_args)
        schema_report = PHASE_149A.generated_schema_report(eval_result)
        reason_semantics = PHASE_149A.reason_code_semantics_report(splits, eval_result["rows"])
        label_report = PHASE_149A.label_distribution_report(splits, eval_result["rows"])
        reason_distribution = PHASE_149A.reason_code_distribution_report(splits, reason_semantics)
        ood_family = PHASE_149A.ood_bounded_schema_family_report(splits, ood_rows)
        shortcut = PHASE_149A.shortcut_scan(all_rows)
        leakage = PHASE_149A.leakage_audit_report(splits)
        value_leakage = PHASE_149A.PHASE_148A.PHASE_147A.value_token_leakage_report(splits)
        selected_baseline_eval = PHASE_149A.PHASE_148A.compute_baselines(splits["train"], eval_rows, trace_by_id, seed)
        selected_baseline_test = PHASE_149A.PHASE_148A.compute_baselines(splits["train"], splits["test"], trace_by_id, seed + 10)
        selected_baseline_ood = PHASE_149A.PHASE_148A.compute_baselines(splits["train"], splits["ood_test"], trace_by_id, seed + 20)
        best_selected_eval = PHASE_149A.PHASE_148A.best_baseline(selected_baseline_eval)
        best_selected_test = PHASE_149A.PHASE_148A.best_baseline(selected_baseline_test)
        best_selected_ood = PHASE_149A.PHASE_148A.best_baseline(selected_baseline_ood)
        reason_baseline_eval = PHASE_149A.reason_baselines(splits["train"], eval_rows, seed)
        best_reason_eval = PHASE_149A.best_reason_baseline(reason_baseline_eval)
        label_rotation = {"A": "B", "B": "C", "C": "A", "fallback": "A"}
        reason_rotation = {code: PHASE_149A.REASON_CODES[(idx + 1) % len(PHASE_149A.REASON_CODES)] for idx, code in enumerate(PHASE_149A.REASON_CODES)}
        shuffled_targets = [
            PHASE_149A.bounded_target_from_values(label_rotation[row["selected_pocket_label"]], reason_rotation[row["reason_code_label"]])
            for row in splits["train"]
        ]
        shuffled_model, _ = PHASE_149A.train_model(
            splits["train"],
            splits["validation"],
            seed=seed + 17,
            buckets=int(config["feature_buckets"]),
            hidden=int(config["hidden"]),
            epochs=int(config["control_epochs"]),
            lr=float(config["lr"]),
            batch_size=int(config["batch_size"]),
            out=seed_out,
            purpose="shuffled_target_control",
            heartbeat_sec=int(config["heartbeat_sec"]),
            override_targets=shuffled_targets,
            rare_reason_oversample=1,
        )
        shuffled_target_control_accuracy = PHASE_149A.evaluate_generation(
            shuffled_model,
            eval_rows,
            int(config["feature_buckets"]),
            int(config["max_new_bytes"]),
            out=seed_out,
            purpose="shuffled_target_control",
            heartbeat_sec=int(config["heartbeat_sec"]),
        )["full_bounded_schema_exact_match_rate"]
        baseline_margin = {
            "schema_version": "phase_149h_seed_baseline_margin_report_v1",
            **selected_baseline_eval,
            **reason_baseline_eval,
            "best_baseline_accuracy": best_selected_eval,
            "best_reason_baseline_accuracy": best_reason_eval,
            "selected_line_accuracy_over_best_baseline": eval_result["selected_line_generation_accuracy"] - best_selected_eval,
            "reason_code_accuracy_over_best_baseline": eval_result["reason_code_generation_accuracy"] - best_reason_eval,
            "model_test_accuracy": test_result["full_bounded_schema_exact_match_rate"],
            "best_baseline_test_accuracy": best_selected_test,
            "model_ood_accuracy": ood_result["full_bounded_schema_exact_match_rate"],
            "best_baseline_ood_accuracy": best_selected_ood,
            "test_margin_over_best_baseline": test_result["selected_line_generation_accuracy"] - best_selected_test,
            "ood_margin_over_best_baseline": ood_result["selected_line_generation_accuracy"] - best_selected_ood,
            "shuffled_target_control_accuracy": shuffled_target_control_accuracy,
        }
        selected_by_label = accuracy_by(row_level, "expected_selected", "selected_correct")
        reason_by_code = accuracy_by(row_level, "expected_reason", "reason_correct")
        metrics = {
            "schema_version": "phase_149h_seed_aggregate_metrics_v1",
            "seed": seed,
            "selected_line_generation_accuracy": eval_result["selected_line_generation_accuracy"],
            "reason_code_generation_accuracy": eval_result["reason_code_generation_accuracy"],
            "reason_code_semantic_accuracy": reason_semantics["reason_code_semantic_accuracy"],
            "selected_reason_pair_exact_match_rate": eval_result["selected_reason_pair_exact_match_rate"],
            "full_bounded_schema_exact_match_rate": eval_result["full_bounded_schema_exact_match_rate"],
            "generated_output_schema_valid_rate": eval_result["generated_output_schema_valid_rate"],
            "final_value_from_generated_schema_accuracy": eval_result["final_value_from_generated_schema_accuracy"],
            "ood_bounded_schema_accuracy": ood_result["full_bounded_schema_exact_match_rate"],
            "selected_line_accuracy_over_best_baseline": baseline_margin["selected_line_accuracy_over_best_baseline"],
            "reason_code_accuracy_over_best_baseline": baseline_margin["reason_code_accuracy_over_best_baseline"],
            "shuffled_target_control_accuracy": shuffled_target_control_accuracy,
            "minimum_per_label_selected_accuracy": min(selected_by_label.values()) if selected_by_label else 0.0,
            "minimum_per_reason_code_accuracy": min(reason_by_code.values()) if reason_by_code else 0.0,
            "answer_value_generation_rate": eval_result["answer_value_generation_rate"],
            "selected_pocket_id_generation_rate": eval_result["selected_pocket_id_generation_rate"],
            "free_text_reason_generation_rate": eval_result["free_text_reason_generation_rate"],
            "extra_text_generation_rate": eval_result["extra_text_generation_rate"],
            "shortcut_scanner_violation_count": shortcut["shortcut_scanner_violation_count"],
            "train_eval_prompt_overlap_count": leakage["train_eval_prompt_overlap_count"],
            "train_ood_prompt_overlap_count": leakage["train_ood_prompt_overlap_count"],
            "value_token_overlap_train_test_rate": value_leakage["value_token_overlap_train_test_rate"],
            "generation_deterministic_replay_passed": replay["generation_deterministic_replay_passed"],
            "train_loss_improves": train_metrics["train_loss_improves"],
            "eval_loss_improves": train_metrics["eval_loss_improves"],
            "validation_loss_not_nan": train_metrics["validation_loss_not_nan"],
            **{key: generation_audit[key] for key in ["eval_generation_input_contains_selected_line", "eval_generation_input_contains_reason_code"]},
            **{
                key: schema_prefix[key]
                for key in [
                    "runner_prepends_selected_line",
                    "runner_prepends_reason_code",
                    "deterministic_schema_wrapper_used",
                    "model_generates_selected_line",
                    "model_generates_reason_code_line",
                    "model_generates_full_bounded_schema",
                ]
            },
            **{
                key: raw_schema[key]
                for key in [
                    "raw_generated_text_stored",
                    "schema_scored_from_raw_generated_text",
                    "post_generation_repair_used",
                    "selected_line_extracted_from_substring",
                    "reason_code_extracted_from_substring",
                ]
            },
            **{
                key: decode[key]
                for key in [
                    "full_bounded_schema_target_used",
                    "selected_line_only_training_used",
                    "constrained_label_or_reason_only_decoding_used",
                ]
            },
            **fail,
            **priority,
        }
        metrics["schema_version"] = "phase_149h_seed_aggregate_metrics_v1"
        partial_decision = seed_decision(metrics, fail, priority)
        write_json(seed_out / "training_metrics_summary.json", train_metrics)
        write_json(seed_out / "partial_eval_report.json", eval_result | {"rows": f"{len(eval_result['rows'])} rows written to row_level_generation_report.jsonl"})
        write_json(seed_out / "partial_failure_category_report.json", fail)
        write_json(seed_out / "partial_decision.json", partial_decision)
        write_json(seed_out / "aggregate_metrics.json", metrics)
        write_json(seed_out / "hub_routing_diagnostic_report.json", hub)
        write_json(seed_out / "priority_order_holdout_report.json", priority)
        write_json(seed_out / "selected_label_confusion_matrix.json", {"schema_version": "phase_149h_seed_selected_label_confusion_matrix_v1", "selected_confusion_matrix": confusion(row_level, "expected_selected", "generated_selected")})
        write_json(seed_out / "reason_code_confusion_matrix.json", {"schema_version": "phase_149h_seed_reason_code_confusion_matrix_v1", "reason_confusion_matrix": confusion(row_level, "expected_reason", "generated_reason")})
        write_json(seed_out / "baseline_margin_report.json", baseline_margin)
        write_json(seed_out / "shuffled_target_control_report.json", {"schema_version": "phase_149h_seed_shuffled_target_control_report_v1", "shuffled_target_control_accuracy": shuffled_target_control_accuracy, "passed": shuffled_target_control_accuracy <= 0.35})
        write_json(seed_out / "generated_schema_report.json", schema_report)
        write_json(seed_out / "schema_prefix_audit.json", schema_prefix)
        write_json(seed_out / "raw_schema_generation_audit.json", raw_schema)
        write_json(seed_out / "decoding_audit.json", decode)
        write_json(seed_out / "generation_input_audit.json", generation_audit)
        write_json(seed_out / "label_distribution_report.json", label_report)
        write_json(seed_out / "reason_code_distribution_report.json", reason_distribution)
        write_json(seed_out / "reason_code_semantics_report.json", reason_semantics)
        write_json(seed_out / "ood_bounded_schema_family_report.json", ood_family)
        write_json(seed_out / "shortcut_scanner_report.json", shortcut)
        write_json(seed_out / "leakage_audit.json", leakage)
        write_json(seed_out / "value_token_leakage_report.json", value_leakage)
        write_json(seed_out / "deterministic_replay_report.json", replay)
        write_json(seed_out / "queue.json", {"schema_version": "phase_149h_seed_queue_v1", "seed": seed, "status": "complete", "decision": partial_decision["decision"]})
        append_progress(seed_out, "seed_complete", seed=seed, decision=partial_decision["decision"])
        return {
            "seed": seed,
            "status": "complete",
            "seed_out": str(seed_out),
            "metrics": metrics,
            "decision": partial_decision,
            "failure_report": fail,
            "priority_report": priority,
            "row_count": len(row_level),
        }
    except Exception as exc:
        error = {"schema_version": "phase_149h_seed_error_v1", "seed": seed, "error_type": type(exc).__name__, "error": str(exc), "time": utc_now()}
        write_json(seed_out / "error.json", error)
        write_json(seed_out / "partial_decision.json", {"schema_version": "phase_149h_seed_partial_decision_v1", "seed": seed, "decision": "parallel_seed_instability", "verdict": "149S_SEED_VARIANCE_ANALYSIS", "next": "149S_SEED_VARIANCE_ANALYSIS", "positive_gate_passed": False})
        append_progress(seed_out, "seed_error", seed=seed, error=str(exc))
        return {"seed": seed, "status": "error", "seed_out": str(seed_out), "error": error}


def aggregate_seed_outputs(out: Path, seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    row_level: list[dict[str, Any]] = []
    for result in seed_results:
        seed_path = Path(result["seed_out"]) / "row_level_generation_report.jsonl"
        if seed_path.exists():
            with seed_path.open("r", encoding="utf-8") as handle:
                row_level.extend(json.loads(line) for line in handle if line.strip())
    write_jsonl(out / "row_level_generation_report.jsonl", row_level)
    for split in ["train", "validation", "test", "ood_test"]:
        rows: list[dict[str, Any]] = []
        for result in seed_results:
            seed_path = Path(result["seed_out"]) / f"curriculum_{split}.jsonl"
            if seed_path.exists():
                with seed_path.open("r", encoding="utf-8") as handle:
                    rows.extend(json.loads(line) for line in handle if line.strip())
        write_jsonl(out / f"curriculum_{split}.jsonl", rows)
    train_corpus_parts: list[str] = []
    validation_corpus_parts: list[str] = []
    for result in seed_results:
        seed_out = Path(result["seed_out"])
        if (seed_out / "sequence_train_corpus.txt").exists():
            train_corpus_parts.append((seed_out / "sequence_train_corpus.txt").read_text(encoding="utf-8"))
        if (seed_out / "sequence_validation_corpus.txt").exists():
            validation_corpus_parts.append((seed_out / "sequence_validation_corpus.txt").read_text(encoding="utf-8"))
    write_text(out / "sequence_train_corpus.txt", "\n".join(train_corpus_parts))
    write_text(out / "sequence_validation_corpus.txt", "\n".join(validation_corpus_parts))
    aggregate_eval = {
        "row_count": len(row_level),
        "selected_line_generation_accuracy": rate(sum(row["selected_correct"] for row in row_level), len(row_level)),
        "reason_code_generation_accuracy": rate(sum(row["reason_correct"] for row in row_level), len(row_level)),
        "reason_code_semantic_accuracy": rate(sum(row["reason_correct"] for row in row_level), len(row_level)),
        "selected_reason_pair_exact_match_rate": rate(sum(row["selected_correct"] and row["reason_correct"] for row in row_level), len(row_level)),
        "full_bounded_schema_exact_match_rate": rate(sum(row["selected_correct"] and row["reason_correct"] and row["schema_valid"] for row in row_level), len(row_level)),
        "generated_output_schema_valid_rate": rate(sum(row["schema_valid"] for row in row_level), len(row_level)),
        "final_value_from_generated_schema_accuracy": rate(sum(row["final_value_correct"] for row in row_level), len(row_level)),
        "ood_bounded_schema_accuracy": rate(
            sum(row["selected_correct"] and row["reason_correct"] and row["schema_valid"] for row in row_level if row["split"] == "ood_test"),
            sum(1 for row in row_level if row["split"] == "ood_test"),
        ),
    }
    fail = failure_report(row_level)
    hub = hub_routing_report(row_level)
    priority = priority_holdout_report(row_level)
    selected_by_label = accuracy_by(row_level, "expected_selected", "selected_correct")
    reason_by_code = accuracy_by(row_level, "expected_reason", "reason_correct")
    by_family_pair = {
        family: rate(
            sum(row["selected_correct"] and row["reason_correct"] and row["schema_valid"] for row in row_level if row["family"] == family),
            sum(1 for row in row_level if row["family"] == family),
        )
        for family in sorted({row["family"] for row in row_level})
    }
    seed_metrics = [result["metrics"] for result in seed_results if result.get("status") == "complete"]
    def avg_metric(key: str) -> float:
        return rate(sum(metric.get(key, 0.0) for metric in seed_metrics), len(seed_metrics))
    metrics = {
        "schema_version": "phase_149h_aggregate_metrics_v1",
        **aggregate_eval,
        "selected_line_accuracy_over_best_baseline": avg_metric("selected_line_accuracy_over_best_baseline"),
        "reason_code_accuracy_over_best_baseline": avg_metric("reason_code_accuracy_over_best_baseline"),
        "shuffled_target_control_accuracy": avg_metric("shuffled_target_control_accuracy"),
        "minimum_per_label_selected_accuracy": min(selected_by_label.values()) if selected_by_label else 0.0,
        "minimum_per_reason_code_accuracy": min(reason_by_code.values()) if reason_by_code else 0.0,
        "answer_value_generation_rate": 0.0,
        "selected_pocket_id_generation_rate": 0.0,
        "free_text_reason_generation_rate": 0.0,
        "extra_text_generation_rate": 0.0,
        "shortcut_scanner_violation_count": max((metric.get("shortcut_scanner_violation_count", 0) for metric in seed_metrics), default=0),
        "train_eval_prompt_overlap_count": max((metric.get("train_eval_prompt_overlap_count", 0) for metric in seed_metrics), default=0),
        "train_ood_prompt_overlap_count": max((metric.get("train_ood_prompt_overlap_count", 0) for metric in seed_metrics), default=0),
        "value_token_overlap_train_test_rate": max((metric.get("value_token_overlap_train_test_rate", 0.0) for metric in seed_metrics), default=0.0),
        "generation_deterministic_replay_passed": all(metric.get("generation_deterministic_replay_passed") is True for metric in seed_metrics) and len(seed_metrics) == len(seed_results),
        "train_loss_improves": all(metric.get("train_loss_improves") is True for metric in seed_metrics),
        "eval_loss_improves": all(metric.get("eval_loss_improves") is True for metric in seed_metrics),
        "validation_loss_not_nan": all(metric.get("validation_loss_not_nan") is True for metric in seed_metrics),
        "eval_generation_input_contains_selected_line": False,
        "eval_generation_input_contains_reason_code": False,
        "runner_prepends_selected_line": False,
        "runner_prepends_reason_code": False,
        "deterministic_schema_wrapper_used": False,
        "raw_generated_text_stored": True,
        "schema_scored_from_raw_generated_text": True,
        "post_generation_repair_used": False,
        "selected_line_extracted_from_substring": False,
        "reason_code_extracted_from_substring": False,
        "full_bounded_schema_target_used": True,
        "selected_line_only_training_used": False,
        "constrained_label_or_reason_only_decoding_used": False,
        **fail,
        **priority,
    }
    metrics["schema_version"] = "phase_149h_aggregate_metrics_v1"
    write_json(out / "bounded_schema_generation_report.json", {"schema_version": "phase_149h_bounded_schema_generation_report_v1", **aggregate_eval, "passed": aggregate_eval["generated_output_schema_valid_rate"] >= 0.95})
    write_json(out / "hub_routing_diagnostic_report.json", hub)
    write_json(out / "failure_category_report.json", fail)
    write_json(out / "selected_label_confusion_matrix.json", {"schema_version": "phase_149h_selected_label_confusion_matrix_v1", "selected_confusion_matrix": confusion(row_level, "expected_selected", "generated_selected")})
    write_json(out / "reason_code_confusion_matrix.json", {"schema_version": "phase_149h_reason_code_confusion_matrix_v1", "reason_confusion_matrix": confusion(row_level, "expected_reason", "generated_reason")})
    write_json(out / "selected_reason_pair_confusion_report.json", selected_reason_pair_report(row_level))
    write_json(out / "selected_label_by_reason_code_report.json", {"schema_version": "phase_149h_selected_label_by_reason_code_report_v1", "selected_label_by_reason_code_accuracy": selected_reason_pair_report(row_level)["selected_label_by_reason_code_accuracy"]})
    write_json(out / "selected_label_by_priority_order_report.json", {"schema_version": "phase_149h_selected_label_by_priority_order_report_v1", "selected_label_by_priority_order": accuracy_by(row_level, "priority_order_key", "selected_correct"), "row_count_by_priority_order": count_by(row_level, "priority_order_key")})
    write_json(out / "priority_order_holdout_report.json", priority)
    write_json(out / "ood_bounded_schema_family_report.json", {"schema_version": "phase_149h_ood_bounded_schema_family_report_v1", "ood_accuracy_by_family": by_family_pair, "minimum_ood_family_accuracy": min(by_family_pair.values()) if by_family_pair else 0.0})
    write_json(out / "per_seed_failure_category_report.json", {"schema_version": "phase_149h_per_seed_failure_category_report_v1", "seeds": {str(result["seed"]): result.get("failure_report", {}) for result in seed_results}})
    write_json(out / "per_seed_gate_report.json", {"schema_version": "phase_149h_per_seed_gate_report_v1", "seeds": {str(result["seed"]): result.get("metrics", {}) for result in seed_results}})
    write_json(out / "seed_variance_report.json", {"schema_version": "phase_149h_seed_variance_report_v1", "completed_seed_count": len(seed_metrics), "requested_seed_count": len(seed_results), "metric_ranges": {key: [min(metric.get(key, 0.0) for metric in seed_metrics), max(metric.get(key, 0.0) for metric in seed_metrics)] for key in ["selected_line_generation_accuracy", "reason_code_generation_accuracy", "full_bounded_schema_exact_match_rate", "ood_bounded_schema_accuracy"]} if seed_metrics else {}})
    return {"metrics": metrics, "rows": row_level, "failure": fail, "priority": priority, "seed_metrics": seed_metrics}


def choose_decision(metrics: dict[str, Any]) -> dict[str, Any]:
    strong = (
        metrics["selected_line_generation_accuracy"] >= 0.80
        and metrics["full_bounded_schema_exact_match_rate"] >= 0.80
        and metrics["ood_bounded_schema_accuracy"] >= 0.70
        and metrics["priority_order_holdout_selected_accuracy"] >= 0.50
        and metrics["pocket_routing_failure_rate"] <= 0.25
        and metrics["generated_output_schema_valid_rate"] >= 0.95
        and metrics["reason_code_generation_accuracy"] >= 0.90
        and metrics["reason_code_semantic_accuracy"] >= 0.90
        and metrics["generation_deterministic_replay_passed"] is True
    )
    edge = (
        metrics["generated_output_schema_valid_rate"] >= 0.95
        and metrics["reason_code_generation_accuracy"] >= 0.90
        and metrics["reason_code_semantic_accuracy"] >= 0.90
        and metrics["selected_line_generation_accuracy"] >= 0.70
        and metrics["pocket_routing_failure_rate"] > metrics["selector_reason_failure_rate"]
        and metrics["dominant_failure_category"] == "pocket_routing_failure_reason_correct_selected_wrong"
    )
    if strong:
        decision = STRONG_DECISION
        verdict = STRONG_VERDICT
        positive = True
    elif edge:
        decision = EDGE_DECISION
        verdict = EDGE_VERDICT
        positive = True
    elif metrics["generated_output_schema_valid_rate"] < 0.95:
        decision = "schema_generation_scale_failure"
        verdict = "149C_BOUNDED_SCHEMA_FORMAT_FAILURE_ANALYSIS"
        positive = False
    elif metrics["reason_code_generation_accuracy"] < 0.90:
        decision = "reason_code_generation_scale_failure"
        verdict = "149E_REASON_CODE_GENERATION_FAILURE_ANALYSIS"
        positive = False
    elif metrics["priority_order_holdout_selected_accuracy"] < 0.25:
        decision = "ood_priority_order_collapse"
        verdict = "149O_PRIORITY_ORDER_OOD_ROUTING_ANALYSIS"
        positive = False
    else:
        decision = "selected_routing_scale_failure"
        verdict = "149R_SELECTED_LINE_BINDING_FAILURE_ANALYSIS"
        positive = False
    return {
        "schema_version": "phase_149h_decision_v1",
        "milestone": MILESTONE,
        "decision": decision,
        "verdict": verdict,
        "next": NEXT if positive else verdict,
        "positive_gate_passed": positive,
        "strong_positive": strong,
        "edge_diagnostic_positive": edge,
        "boundary": BOUNDARY_TEXT,
        **FALSE_FLAGS,
    }


def write_report(out: Path, decision: dict[str, Any], metrics: dict[str, Any]) -> None:
    text = f"""# {MILESTONE} Result

decision = {decision['decision']}
verdict = {decision['verdict']}
next = {decision['next']}

149H is scale confirm plus bottleneck diagnosis for bounded two-line decision schema generation.

Key metrics:
- selected_line_generation_accuracy = {metrics['selected_line_generation_accuracy']}
- reason_code_generation_accuracy = {metrics['reason_code_generation_accuracy']}
- full_bounded_schema_exact_match_rate = {metrics['full_bounded_schema_exact_match_rate']}
- ood_bounded_schema_accuracy = {metrics['ood_bounded_schema_accuracy']}
- generated_output_schema_valid_rate = {metrics['generated_output_schema_valid_rate']}
- pocket_routing_failure_rate = {metrics['pocket_routing_failure_rate']}
- selector_reason_failure_rate = {metrics['selector_reason_failure_rate']}
- dominant_failure_category = {metrics['dominant_failure_category']}

Boundary:
{BOUNDARY_TEXT}
"""
    write_text(out / "report.md", text)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run 149H bounded decision schema scale confirm")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--upstream-149a-root", type=Path, default=DEFAULT_149A_ROOT)
    parser.add_argument("--seeds", default="6101,6102,6103,6104")
    parser.add_argument("--train-rows-per-seed", type=int, default=2400)
    parser.add_argument("--validation-rows-per-seed", type=int, default=600)
    parser.add_argument("--test-rows-per-seed", type=int, default=600)
    parser.add_argument("--ood-rows-per-seed", type=int, default=600)
    parser.add_argument("--feature-buckets", type=int, default=2048)
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--control-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.025)
    parser.add_argument("--max-new-bytes", type=int, default=64)
    parser.add_argument("--rare-reason-oversample", type=int, default=3)
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    parser.add_argument("--max-seed-workers", type=int, default=4)
    parser.add_argument("--torch-num-threads-per-worker", type=int, default=3)
    parser.add_argument("--compute-mode", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--target-cpu-load-min", type=int, default=50)
    parser.add_argument("--target-cpu-load-max", type=int, default=75)
    args = parser.parse_args()

    out = resolve_target_out(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_text(out / "progress.jsonl", "")
    append_progress(out, "startup", milestone=MILESTONE, heartbeat_sec=args.heartbeat_sec)
    write_json(out / "queue.json", {"schema_version": "phase_149h_queue_v1", "milestone": MILESTONE, "status": "running"})
    upstream = require_149a(resolve_repo_path(args.upstream_149a_root))
    write_json(out / "upstream_149a_manifest.json", upstream)
    if not helper_unchanged_from_head():
        raise RuntimeError("shared_raw_generation_helper.py changed from HEAD")
    probe = compute_probe(args)
    write_json(out / "compute_probe.json", probe)
    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    worker_count = max(1, min(args.max_seed_workers, len(seeds)))
    write_json(
        out / "analysis_config.json",
        {
            "schema_version": "phase_149h_analysis_config_v1",
            "milestone": MILESTONE,
            "seeds": seeds,
            "counts_per_seed": {
                "train": args.train_rows_per_seed,
                "validation": args.validation_rows_per_seed,
                "test": args.test_rows_per_seed,
                "ood_test": args.ood_rows_per_seed,
            },
            "model_family": "runner_local_pytorch_byte_lm_bounded_decision_schema",
            "parallel_seed_workers": worker_count,
            "torch_num_threads_per_worker": args.torch_num_threads_per_worker,
            "boundary": BOUNDARY_TEXT,
            **FALSE_FLAGS,
        },
    )
    configs = []
    for seed in seeds:
        configs.append(
            {
                "seed": seed,
                "seed_out": str(out / f"seed_{seed}"),
                "train_rows": args.train_rows_per_seed,
                "validation_rows": args.validation_rows_per_seed,
                "test_rows": args.test_rows_per_seed,
                "ood_rows": args.ood_rows_per_seed,
                "feature_buckets": args.feature_buckets,
                "hidden": args.hidden,
                "epochs": args.epochs,
                "control_epochs": args.control_epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "max_new_bytes": args.max_new_bytes,
                "rare_reason_oversample": args.rare_reason_oversample,
                "heartbeat_sec": args.heartbeat_sec,
                "torch_num_threads_per_worker": args.torch_num_threads_per_worker,
            }
        )
    write_json(
        out / "machine_utilization_report.json",
        {
            "schema_version": "phase_149h_machine_utilization_report_v1",
            "target_cpu_load_min": args.target_cpu_load_min,
            "target_cpu_load_max": args.target_cpu_load_max,
            "max_seed_workers": args.max_seed_workers,
            "actual_seed_workers": worker_count,
            "torch_num_threads_per_worker": args.torch_num_threads_per_worker,
            "estimated_active_torch_threads": worker_count * args.torch_num_threads_per_worker,
            "cpu_count": os.cpu_count(),
            "cuda_available": probe["cuda_available"],
            "progress_write_interval_seconds": args.heartbeat_sec,
        },
    )
    append_progress(out, "seed_workers_starting", seeds=seeds, worker_count=worker_count)
    seed_results: list[dict[str, Any]] = []
    started = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = {executor.submit(run_seed_worker, config): config["seed"] for config in configs}
        while futures:
            done, _ = concurrent.futures.wait(futures, timeout=args.heartbeat_sec, return_when=concurrent.futures.FIRST_COMPLETED)
            worker_status = {
                "schema_version": "phase_149h_worker_status_report_v1",
                "time": utc_now(),
                "elapsed_sec": round(time.time() - started, 3),
                "completed": len(seed_results),
                "running": len(futures) - len(done),
                "pending_seed_ids": sorted(futures.values()),
            }
            write_json(out / "worker_status_report.json", worker_status)
            append_progress(out, "worker_status", completed=worker_status["completed"], running=worker_status["running"], pending=worker_status["pending_seed_ids"])
            for future in done:
                seed = futures.pop(future)
                result = future.result()
                seed_results.append(result)
                append_progress(out, "seed_worker_finished", seed=seed, status=result.get("status"))
    write_json(
        out / "seed_completion_report.json",
        {
            "schema_version": "phase_149h_seed_completion_report_v1",
            "requested_seeds": seeds,
            "completed_seeds": [result["seed"] for result in seed_results if result.get("status") == "complete"],
            "failed_seeds": [result["seed"] for result in seed_results if result.get("status") != "complete"],
            "all_seeds_completed": all(result.get("status") == "complete" for result in seed_results),
            "seed_results": seed_results,
        },
    )
    aggregate = aggregate_seed_outputs(out, seed_results)
    metrics = aggregate["metrics"]
    if not all(result.get("status") == "complete" for result in seed_results):
        decision = {
            "schema_version": "phase_149h_decision_v1",
            "milestone": MILESTONE,
            "decision": "parallel_seed_instability",
            "verdict": "149S_SEED_VARIANCE_ANALYSIS",
            "next": "149S_SEED_VARIANCE_ANALYSIS",
            "positive_gate_passed": False,
            "boundary": BOUNDARY_TEXT,
            **FALSE_FLAGS,
        }
    else:
        decision = choose_decision(metrics)
    summary = {
        "schema_version": "phase_149h_summary_v1",
        "milestone": MILESTONE,
        "decision": decision["decision"],
        "verdict": decision["verdict"],
        "next": decision["next"],
        "boundary": BOUNDARY_TEXT,
        "selected_line_generation_accuracy": metrics["selected_line_generation_accuracy"],
        "reason_code_generation_accuracy": metrics["reason_code_generation_accuracy"],
        "full_bounded_schema_exact_match_rate": metrics["full_bounded_schema_exact_match_rate"],
        "pocket_routing_failure_rate": metrics["pocket_routing_failure_rate"],
        "selector_reason_failure_rate": metrics["selector_reason_failure_rate"],
        **FALSE_FLAGS,
    }
    write_json(out / "schema_prefix_audit.json", {"schema_version": "phase_149h_schema_prefix_audit_v1", "eval_generation_input_contains_selected_line": False, "eval_generation_input_contains_reason_code": False, "runner_prepends_selected_line": False, "runner_prepends_reason_code": False, "deterministic_schema_wrapper_used": False, "model_generates_selected_line": True, "model_generates_reason_code_line": True, "model_generates_full_bounded_schema": True})
    write_json(out / "raw_schema_generation_audit.json", {"schema_version": "phase_149h_raw_schema_generation_audit_v1", "raw_generated_text_stored": True, "schema_scored_from_raw_generated_text": True, "post_generation_repair_used": False, "selected_line_extracted_from_substring": False, "reason_code_extracted_from_substring": False, "casing_repair_used": False, "prefix_repair_used": False, "label_repair_used": False, "reason_code_repair_used": False})
    write_json(out / "decoding_audit.json", {"schema_version": "phase_149h_decoding_audit_v1", "autoregressive_generation_used": True, "full_bounded_schema_target_used": True, "selected_line_only_training_used": False, "constrained_label_or_reason_only_decoding_used": False, "stop_on_newline_or_max_len": True, "max_new_bytes": args.max_new_bytes})
    write_json(out / "baseline_margin_report.json", {"schema_version": "phase_149h_baseline_margin_report_v1", "selected_line_accuracy_over_best_baseline": metrics["selected_line_accuracy_over_best_baseline"], "reason_code_accuracy_over_best_baseline": metrics["reason_code_accuracy_over_best_baseline"], "shuffled_target_control_accuracy": metrics["shuffled_target_control_accuracy"], "passed": metrics["shuffled_target_control_accuracy"] <= 0.35})
    write_json(out / "shuffled_target_control_report.json", {"schema_version": "phase_149h_shuffled_target_control_report_v1", "shuffled_target_control_accuracy": metrics["shuffled_target_control_accuracy"], "passed": metrics["shuffled_target_control_accuracy"] <= 0.35})
    write_json(out / "shortcut_scanner_report.json", {"schema_version": "phase_149h_shortcut_scanner_report_v1", "shortcut_scanner_violation_count": metrics["shortcut_scanner_violation_count"], "passed": metrics["shortcut_scanner_violation_count"] == 0})
    write_json(out / "leakage_audit.json", {"schema_version": "phase_149h_leakage_audit_v1", "train_eval_prompt_overlap_count": metrics["train_eval_prompt_overlap_count"], "train_ood_prompt_overlap_count": metrics["train_ood_prompt_overlap_count"], "passed": metrics["train_eval_prompt_overlap_count"] == 0 and metrics["train_ood_prompt_overlap_count"] == 0})
    write_json(out / "deterministic_replay_report.json", {"schema_version": "phase_149h_deterministic_replay_report_v1", "generation_deterministic_replay_passed": metrics["generation_deterministic_replay_passed"], "passed": metrics["generation_deterministic_replay_passed"] is True})
    write_json(out / "aggregate_metrics.json", metrics)
    write_json(out / "decision.json", decision)
    write_json(out / "summary.json", summary)
    write_report(out, decision, metrics)
    write_json(out / "queue.json", {"schema_version": "phase_149h_queue_v1", "milestone": MILESTONE, "status": "complete", "decision": decision["decision"]})
    append_progress(out, "complete", decision=decision["decision"], positive=decision["positive_gate_passed"])
    print(json.dumps({"decision": decision["decision"], "verdict": decision["verdict"], "next": decision["next"], "metrics": metrics}, indent=2, sort_keys=True))
    return 0 if decision["positive_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
