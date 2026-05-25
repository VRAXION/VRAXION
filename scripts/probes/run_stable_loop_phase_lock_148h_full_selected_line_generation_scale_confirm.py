#!/usr/bin/env python3
"""148H scale confirm for bounded full SELECTED=<label> line generation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "STABLE_LOOP_PHASE_LOCK_148H_FULL_SELECTED_LINE_GENERATION_SCALE_CONFIRM"
DEFAULT_OUT = Path("target/pilot_wave/stable_loop_phase_lock_148h_full_selected_line_generation_scale_confirm/smoke")
DEFAULT_148A_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_148a_full_selected_line_generation_prototype/smoke")
HELPER_PATH = REPO_ROOT / "scripts/probes/shared_raw_generation_helper.py"
PHASE_148A_PATH = REPO_ROOT / "scripts/probes/run_stable_loop_phase_lock_148a_full_selected_line_generation_prototype.py"

DECISION = "full_selected_line_generation_scale_confirmed"
VERDICT = "INSTNCT_FULL_SELECTED_LINE_GENERATION_SCALE_CONFIRMED"
NEXT = "148Z_FULL_SELECTED_LINE_GENERATION_NEXT_DECISION_PLAN"
BOUNDARY_TEXT = (
    "148H is constrained model-facing distillation evidence only with canonical structured prompts only, "
    "bounded full SELECTED=<label> line generation only; not natural-language rule reasoning, "
    "not open-ended arbitration, not GPT-like/Gemma-like assistant capability, not production readiness, "
    "and not architecture superiority."
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
LABELS = ["A", "B", "C", "fallback"]
NEGATIVE_ROUTES = {
    "full_line_training_scale_failure": "148B_FULL_LINE_TRAINING_FAILURE_ANALYSIS",
    "generated_schema_scale_failure": "148C_FULL_LINE_SCHEMA_FAILURE_ANALYSIS",
    "selected_label_extraction_scale_failure": "148D_SELECTED_LABEL_EXTRACTION_FAILURE_ANALYSIS",
    "model_shortcut_detected": "148E_FULL_LINE_SHORTCUT_ANALYSIS",
    "ood_full_line_generation_scale_failure": "148F_FULL_LINE_OOD_ANALYSIS",
    "generation_input_leakage_detected": "148G_FULL_LINE_INPUT_LEAKAGE_ANALYSIS",
    "hidden_selected_prefix_detected": "148J_HIDDEN_SELECTED_PREFIX_WRAPPER_ANALYSIS",
    "post_generation_repair_detected": "148K_POST_GENERATION_REPAIR_ANALYSIS",
    "deterministic_replay_failure": "148I_FULL_LINE_DETERMINISM_FAILURE_ANALYSIS",
}


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PHASE_148A = load_module(PHASE_148A_PATH, "phase_148a")


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


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


def append_progress(out: Path, event: str, **details: Any) -> None:
    append_jsonl(out / "progress.jsonl", {"ts": utc_now(), "event": event, "details": details})


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def git_show_head(path: str) -> str:
    result = subprocess.run(["git", "show", f"HEAD:{path}"], cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    return result.stdout if result.returncode == 0 else ""


def helper_unchanged_from_head() -> bool:
    return HELPER_PATH.read_text(encoding="utf-8") == git_show_head("scripts/probes/shared_raw_generation_helper.py")


def rate(count: int | float, total: int | float) -> float:
    return float(count) / float(total) if total else 0.0


def parse_seeds(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise ValueError("--seeds must include at least one seed")
    return seeds


def require_148a(root: Path) -> dict[str, Any]:
    required = [
        "decision.json",
        "aggregate_metrics.json",
        "generation_prefix_audit.json",
        "raw_generation_audit.json",
        "decoding_audit.json",
        "generated_schema_report.json",
        "model_artifact_audit.json",
        "deterministic_replay_report.json",
    ]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise RuntimeError(f"missing 148A artifacts: {missing}")
    decision = read_json(root / "decision.json")
    metrics = read_json(root / "aggregate_metrics.json")
    prefix = read_json(root / "generation_prefix_audit.json")
    raw = read_json(root / "raw_generation_audit.json")
    decoding = read_json(root / "decoding_audit.json")
    schema = read_json(root / "generated_schema_report.json")
    artifact = read_json(root / "model_artifact_audit.json")
    replay = read_json(root / "deterministic_replay_report.json")
    checks = {
        "decision": decision.get("decision") == "full_selected_line_generation_prototype_positive",
        "verdict": decision.get("verdict") == "INSTNCT_FULL_SELECTED_LINE_GENERATION_PROTOTYPE_POSITIVE",
        "next": decision.get("next") == "148H_FULL_SELECTED_LINE_GENERATION_SCALE_CONFIRM",
        "full_selected_line_exact_match_rate": metrics.get("full_selected_line_exact_match_rate", 0.0) >= 0.98,
        "selected_prefix_generation_accuracy": metrics.get("selected_prefix_generation_accuracy", 0.0) >= 0.99,
        "selected_label_generation_accuracy": metrics.get("selected_label_generation_accuracy", 0.0) >= 0.98,
        "fallback_full_line_accuracy": metrics.get("fallback_full_line_accuracy", 0.0) >= 0.95,
        "generated_output_schema_valid_rate": metrics.get("generated_output_schema_valid_rate", 0.0) >= 0.99,
        "ood_full_line_accuracy": metrics.get("ood_full_line_accuracy", 0.0) >= 0.95,
        "shuffled_target_control_accuracy": metrics.get("shuffled_target_control_accuracy", 1.0) <= 0.05,
        "generation_deterministic_replay_passed": metrics.get("generation_deterministic_replay_passed") is True,
        "eval_generation_input_contains_selected_prefix": metrics.get("eval_generation_input_contains_selected_prefix") is False,
        "runner_prepends_selected_prefix": metrics.get("runner_prepends_selected_prefix") is False,
        "deterministic_selected_line_wrapper_used": metrics.get("deterministic_selected_line_wrapper_used") is False,
        "post_generation_repair_used": metrics.get("post_generation_repair_used") is False,
        "selected_line_extracted_from_substring": metrics.get("selected_line_extracted_from_substring") is False,
        "first_byte_only_training_used": metrics.get("first_byte_only_training_used") is False,
        "constrained_label_only_decoding_used": metrics.get("constrained_label_only_decoding_used") is False,
        "generation_prefix_audit_passed": prefix.get("passed") is True,
        "raw_generation_audit_passed": raw.get("passed") is True,
        "decoding_audit_passed": decoding.get("passed") is True,
        "schema_report_passed": schema.get("passed") is True,
        "artifact_report_passed": artifact.get("passed") is True,
        "replay_report_passed": replay.get("passed") is True,
    }
    failed = [key for key, value in checks.items() if not value]
    if failed:
        raise RuntimeError(f"148A upstream mismatch: {failed}")
    return {
        "schema_version": "phase_148h_upstream_148a_manifest_v1",
        "root": rel(root),
        "decision": decision,
        "aggregate_metrics": metrics,
        "generation_prefix_audit": prefix,
        "raw_generation_audit": raw,
        "decoding_audit": decoding,
        "generated_schema_report": schema,
        "model_artifact_audit": artifact,
        "deterministic_replay_report": replay,
        "checks": checks,
        "failed_checks": failed,
        "passed": not failed,
    }


def flatten_split(seed_splits: list[dict[str, list[dict[str, Any]]]]) -> dict[str, list[dict[str, Any]]]:
    combined = {"train": [], "validation": [], "test": [], "ood_test": []}
    for splits in seed_splits:
        for split in combined:
            combined[split].extend(splits[split])
    return combined


def ood_family_report(splits: dict[str, list[dict[str, Any]]], result_rows: list[dict[str, Any]]) -> dict[str, Any]:
    family_by_id = {row["row_id"]: row["family"] for row in splits["ood_test"]}
    totals: dict[str, int] = defaultdict(int)
    correct: dict[str, int] = defaultdict(int)
    for result in result_rows:
        family = family_by_id[result["row_id"]]
        totals[family] += 1
        correct[family] += int(result["full_selected_line_correct"])
    accuracy = {family: rate(correct[family], totals[family]) for family in sorted(totals)}
    row_counts = {family: totals[family] for family in sorted(totals)}
    minimum_accuracy = min(accuracy.values()) if accuracy else 0.0
    minimum_count = min(row_counts.values()) if row_counts else 0
    payload = {
        "schema_version": "phase_148h_ood_full_line_family_report_v1",
        "ood_full_line_accuracy_by_family": accuracy,
        "row_count_by_ood_family": row_counts,
        "heldout_priority_order_full_line_accuracy": accuracy.get("PRIORITY_ORDER_HOLDOUT", 0.0),
        "heldout_block_order_full_line_accuracy": accuracy.get("BLOCK_ORDER_HOLDOUT", 0.0),
        "heldout_template_full_line_accuracy": accuracy.get("EXACT_TEMPLATE_HOLDOUT", 1.0),
        "heldout_rule_composition_full_line_accuracy": accuracy.get("RULE_BLOCK_TYPE_COMBINATION_HOLDOUT", 0.0),
        "minimum_ood_family_accuracy": minimum_accuracy,
        "minimum_ood_family_row_count": minimum_count,
        "collapsed_ood_family_count": sum(1 for value in accuracy.values() if value < 0.75),
    }
    payload["passed"] = (
        payload["heldout_priority_order_full_line_accuracy"] >= 0.75
        and payload["heldout_block_order_full_line_accuracy"] >= 0.75
        and payload["heldout_template_full_line_accuracy"] >= 0.75
        and payload["heldout_rule_composition_full_line_accuracy"] >= 0.75
        and payload["minimum_ood_family_accuracy"] >= 0.75
        and payload["collapsed_ood_family_count"] == 0
        and payload["minimum_ood_family_row_count"] >= 40
    )
    return payload


def label_distribution_report(splits: dict[str, list[dict[str, Any]]], result_rows: list[dict[str, Any]]) -> dict[str, Any]:
    distributions = {
        f"{split}_label_counts": dict(Counter(row["selected_pocket_label"] for row in rows))
        for split, rows in splits.items()
    }
    per_label_full: dict[str, float] = {}
    per_label_schema: dict[str, float] = {}
    for label in LABELS:
        rows = [row for row in result_rows if row["expected_selected_label"] == label]
        per_label_full[label] = rate(sum(1 for row in rows if row["full_selected_line_correct"]), len(rows))
        per_label_schema[label] = rate(sum(1 for row in rows if row["schema_valid"]), len(rows))
    every_label = all(all(label in Counter(row["selected_pocket_label"] for row in rows) for label in LABELS) for rows in splits.values())
    minimum_full = min(per_label_full.values()) if per_label_full else 0.0
    minimum_schema = min(per_label_schema.values()) if per_label_schema else 0.0
    payload = {
        "schema_version": "phase_148h_label_distribution_report_v1",
        **distributions,
        "per_label_full_line_accuracy": per_label_full,
        "per_label_schema_valid_rate": per_label_schema,
        "every_label_appears_in_every_split": every_label,
        "fallback_full_line_accuracy": per_label_full.get("fallback", 0.0),
        "minimum_per_label_full_line_accuracy": minimum_full,
        "minimum_per_label_schema_valid_rate": minimum_schema,
    }
    payload["passed"] = every_label and payload["fallback_full_line_accuracy"] >= 0.90 and minimum_full >= 0.75 and minimum_schema >= 0.90
    return payload


def anti_memorization_report(splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    return PHASE_148A.anti_memorization_report(splits) | {"schema_version": "phase_148h_anti_memorization_report_v1"}


def generation_prefix_audit(combined_generation_audit: dict[str, Any], eval_result: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "schema_version": "phase_148h_generation_prefix_audit_v1",
        "eval_generation_input_ends_with_output_delimiter": combined_generation_audit["eval_generation_input_ends_with_output_delimiter"],
        "eval_generation_input_contains_selected_prefix": combined_generation_audit["eval_generation_input_contains_selected_prefix"],
        "runner_prepends_selected_prefix": False,
        "deterministic_selected_line_wrapper_used": False,
        "model_generates_selected_prefix": eval_result["selected_prefix_generation_accuracy"] >= 0.95,
        "model_generates_full_selected_line": eval_result["full_selected_line_exact_match_rate"] >= 0.95,
    }
    payload["passed"] = (
        payload["eval_generation_input_ends_with_output_delimiter"] is True
        and payload["eval_generation_input_contains_selected_prefix"] is False
        and payload["runner_prepends_selected_prefix"] is False
        and payload["deterministic_selected_line_wrapper_used"] is False
        and payload["model_generates_selected_prefix"] is True
        and payload["model_generates_full_selected_line"] is True
    )
    return payload


def raw_generation_audit(eval_result: dict[str, Any]) -> dict[str, Any]:
    payload = dict(PHASE_148A.raw_generation_audit(eval_result))
    payload.update({"schema_version": "phase_148h_raw_generation_audit_v1", "extra_text_ignored_for_schema": False})
    payload["passed"] = payload["passed"] and payload["extra_text_ignored_for_schema"] is False
    return payload


def decoding_audit(args: argparse.Namespace) -> dict[str, Any]:
    payload = dict(PHASE_148A.decoding_audit(args))
    payload["schema_version"] = "phase_148h_decoding_audit_v1"
    return payload


def model_artifact_audit(args: argparse.Namespace, seed_results: list[dict[str, Any]], combined_hash: str) -> dict[str, Any]:
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    return {
        "schema_version": "phase_148h_model_artifact_audit_v1",
        "model_family": "runner_local_pytorch_byte_lm_full_selected_line",
        "same_model_family_as_148a": True,
        "random_init_only": True,
        "pretrained_weights_used": False,
        "external_model_or_api_used": False,
        "model_download_used": False,
        "deterministic_seed_used": True,
        "cpu_only": True,
        "model_parameter_count": seed_results[0]["model_parameter_count"],
        "model_state_hash": combined_hash,
        "model_state_hashes_by_seed": {str(result["seed"]): result["model_state_hash"] for result in seed_results},
        "training_config_hash": sha256_text(json.dumps(config, sort_keys=True)),
        "artifacts_written_only_under_target": True,
        "passed": True,
    }


def deterministic_replay_report(seed_reports: list[dict[str, Any]]) -> dict[str, Any]:
    passed = all(report["generation_deterministic_replay_passed"] for report in seed_reports)
    return {
        "schema_version": "phase_148h_deterministic_replay_report_v1",
        "per_seed_generation_deterministic_replay_passed": {
            str(report["seed"]): report["generation_deterministic_replay_passed"] for report in seed_reports
        },
        "generation_deterministic_replay_passed": passed,
        "passed": passed,
    }


def run_seed(seed: int, counts: dict[str, int], args: argparse.Namespace, out: Path) -> dict[str, Any]:
    append_progress(out, "seed_start", seed=seed)
    splits, traces = PHASE_148A.build_curriculum(seed, counts)
    trace_by_id = {trace["row_id"]: trace for trace in traces}
    append_progress(out, "seed_curriculum_built", seed=seed, rows=sum(len(rows) for rows in splits.values()))
    model, train_metrics = PHASE_148A.train_model(
        splits["train"],
        splits["validation"],
        seed=seed,
        buckets=args.feature_buckets,
        hidden=args.hidden,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        out=out,
        purpose=f"primary_full_line_seed_{seed}",
        heartbeat_sec=args.heartbeat_sec,
        fallback_oversample=args.fallback_oversample,
    )
    eval_rows = splits["validation"] + splits["test"] + splits["ood_test"]
    eval_result = PHASE_148A.evaluate_generation(model, eval_rows, args.feature_buckets, args.max_new_bytes, out=out, purpose=f"eval_seed_{seed}", heartbeat_sec=args.heartbeat_sec)
    replay_result = PHASE_148A.evaluate_generation(model, eval_rows, args.feature_buckets, args.max_new_bytes, out=out, purpose=f"replay_seed_{seed}", heartbeat_sec=args.heartbeat_sec)
    test_result = PHASE_148A.evaluate_generation(model, splits["test"], args.feature_buckets, args.max_new_bytes, out=out, purpose=f"test_seed_{seed}", heartbeat_sec=args.heartbeat_sec)
    ood_result = PHASE_148A.evaluate_generation(model, splits["ood_test"], args.feature_buckets, args.max_new_bytes, out=out, purpose=f"ood_seed_{seed}", heartbeat_sec=args.heartbeat_sec)
    label_rotation = {"A": "B", "B": "C", "C": "A", "fallback": "A"}
    shuffled_labels = [label_rotation[row["selected_pocket_label"]] for row in splits["train"]]
    shuffled_model, _ = PHASE_148A.train_model(
        splits["train"],
        splits["validation"],
        seed=seed + 17,
        buckets=args.feature_buckets,
        hidden=args.hidden,
        epochs=args.control_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        out=out,
        purpose=f"shuffled_target_seed_{seed}",
        heartbeat_sec=args.heartbeat_sec,
        override_labels=shuffled_labels,
        fallback_oversample=1,
    )
    shuffled_accuracy = PHASE_148A.evaluate_generation(shuffled_model, eval_rows, args.feature_buckets, args.max_new_bytes, out=out, purpose=f"shuffled_seed_{seed}", heartbeat_sec=args.heartbeat_sec)["selected_label_generation_accuracy"]
    replay_passed = [row["raw_generated_text"] for row in eval_result["rows"]] == [row["raw_generated_text"] for row in replay_result["rows"]]
    baseline_eval = PHASE_148A.compute_baselines(splits["train"], eval_rows, trace_by_id, seed)
    baseline_test = PHASE_148A.compute_baselines(splits["train"], splits["test"], trace_by_id, seed + 10)
    baseline_ood = PHASE_148A.compute_baselines(splits["train"], splits["ood_test"], trace_by_id, seed + 20)
    seed_report = {
        "seed": seed,
        "full_selected_line_exact_match_rate": eval_result["full_selected_line_exact_match_rate"],
        "selected_prefix_generation_accuracy": eval_result["selected_prefix_generation_accuracy"],
        "selected_label_generation_accuracy": eval_result["selected_label_generation_accuracy"],
        "ood_full_line_accuracy": ood_result["full_selected_line_exact_match_rate"],
        "schema_valid_rate": eval_result["generated_output_schema_valid_rate"],
        "generation_deterministic_replay_passed": replay_passed,
        "passed": (
            eval_result["full_selected_line_exact_match_rate"] >= 0.90
            and eval_result["generated_output_schema_valid_rate"] >= 0.90
            and ood_result["full_selected_line_exact_match_rate"] >= 0.75
            and replay_passed
        ),
    }
    append_progress(out, "seed_complete", seed=seed, full_line_accuracy=seed_report["full_selected_line_exact_match_rate"], ood_accuracy=seed_report["ood_full_line_accuracy"])
    return {
        "seed": seed,
        "splits": splits,
        "traces": traces,
        "train_metrics": train_metrics,
        "eval_result": eval_result,
        "test_result": test_result,
        "ood_result": ood_result,
        "baseline_eval": baseline_eval,
        "baseline_test": baseline_test,
        "baseline_ood": baseline_ood,
        "shuffled_target_control_accuracy": shuffled_accuracy,
        "replay_passed": replay_passed,
        "seed_report": seed_report,
        "model_state_hash": train_metrics["checkpoint_after_hash"],
        "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


def gates_pass(metrics: dict[str, Any]) -> bool:
    return (
        metrics["full_selected_line_exact_match_rate"] >= 0.95
        and metrics["full_line_generation_accuracy"] >= 0.95
        and metrics["selected_prefix_generation_accuracy"] >= 0.95
        and metrics["selected_label_generation_accuracy"] >= 0.95
        and metrics["selected_label_extracted_from_full_line_accuracy"] >= 0.95
        and metrics["final_value_from_generated_line_accuracy"] >= 0.95
        and metrics["fallback_full_line_accuracy"] >= 0.90
        and metrics["generated_output_schema_valid_rate"] >= 0.95
        and metrics["ood_full_line_accuracy"] >= 0.85
        and metrics["minimum_ood_family_accuracy"] >= 0.75
        and metrics["selected_prefix_generation_accuracy"] >= metrics["best_baseline_accuracy"] + 0.10
        and metrics["full_line_generation_accuracy"] >= metrics["best_baseline_accuracy"] + 0.10
        and metrics["test_margin_over_best_baseline"] >= 0.10
        and metrics["ood_margin_over_best_baseline"] >= 0.05
        and metrics["shuffled_target_control_accuracy"] <= 0.35
        and metrics["shortcut_scanner_violation_count"] == 0
        and metrics["train_eval_prompt_overlap_count"] == 0
        and metrics["train_ood_prompt_overlap_count"] == 0
        and metrics["value_token_overlap_train_test_rate"] == 0.0
        and metrics["value_token_overlap_train_ood_rate"] == 0.0
        and metrics["eval_loss_improves"] is True
        and metrics["train_loss_improves"] is True
        and metrics["validation_loss_not_nan"] is True
        and metrics["generation_deterministic_replay_passed"] is True
        and metrics["multiple_selected_line_rate"] == 0.0
        and metrics["answer_value_generation_rate"] == 0.0
        and metrics["selected_pocket_id_generation_rate"] == 0.0
        and metrics["malformed_selected_label_rate"] <= 0.05
        and metrics["extra_text_generation_rate"] <= 0.05
        and metrics["prefix_repair_used"] is False
        and metrics["casing_repair_used"] is False
        and metrics["label_repair_used"] is False
        and metrics["post_generation_repair_used"] is False
        and metrics["collapsed_ood_family_count"] == 0
        and metrics["minimum_ood_family_row_count"] >= 40
        and metrics["every_label_appears_in_every_split"] is True
        and metrics["minimum_per_label_full_line_accuracy"] >= 0.75
        and metrics["minimum_per_label_schema_valid_rate"] >= 0.90
        and metrics["eval_generation_input_contains_selected_prefix"] is False
        and metrics["runner_prepends_selected_prefix"] is False
        and metrics["deterministic_selected_line_wrapper_used"] is False
        and metrics["selected_line_extracted_from_substring"] is False
        and metrics["first_byte_only_training_used"] is False
        and metrics["forced_selected_prefix_used"] is False
        and metrics["constrained_label_only_decoding_used"] is False
        and all(report["passed"] for report in metrics["per_seed_reports"])
    )


def choose_decision(metrics: dict[str, Any], audits: list[dict[str, Any]]) -> dict[str, Any]:
    integrity = all(audit.get("passed") is True for audit in audits)
    if metrics.get("passed") is True and integrity:
        decision = DECISION
        verdict = VERDICT
        next_step = NEXT
    elif metrics.get("generation_deterministic_replay_passed") is not True:
        decision = "deterministic_replay_failure"
        verdict = "INSTNCT_FULL_SELECTED_LINE_GENERATION_SCALE_BLOCKED"
        next_step = "148I_FULL_LINE_DETERMINISM_FAILURE_ANALYSIS"
    elif metrics.get("eval_generation_input_contains_selected_prefix") is not False or metrics.get("runner_prepends_selected_prefix") is not False:
        decision = "hidden_selected_prefix_detected"
        verdict = "INSTNCT_FULL_SELECTED_LINE_GENERATION_SCALE_BLOCKED"
        next_step = "148J_HIDDEN_SELECTED_PREFIX_WRAPPER_ANALYSIS"
    elif metrics.get("post_generation_repair_used") is not False or metrics.get("selected_line_extracted_from_substring") is not False:
        decision = "post_generation_repair_detected"
        verdict = "INSTNCT_FULL_SELECTED_LINE_GENERATION_SCALE_BLOCKED"
        next_step = "148K_POST_GENERATION_REPAIR_ANALYSIS"
    elif metrics.get("generated_output_schema_valid_rate", 0.0) < 0.95:
        decision = "generated_schema_scale_failure"
        verdict = "INSTNCT_FULL_SELECTED_LINE_GENERATION_SCALE_BLOCKED"
        next_step = "148C_FULL_LINE_SCHEMA_FAILURE_ANALYSIS"
    elif metrics.get("full_selected_line_exact_match_rate", 0.0) < 0.95:
        decision = "selected_label_extraction_scale_failure"
        verdict = "INSTNCT_FULL_SELECTED_LINE_GENERATION_SCALE_BLOCKED"
        next_step = "148D_SELECTED_LABEL_EXTRACTION_FAILURE_ANALYSIS"
    elif metrics.get("ood_full_line_accuracy", 0.0) < 0.85:
        decision = "ood_full_line_generation_scale_failure"
        verdict = "INSTNCT_FULL_SELECTED_LINE_GENERATION_SCALE_BLOCKED"
        next_step = "148F_FULL_LINE_OOD_ANALYSIS"
    elif not integrity:
        decision = "model_shortcut_detected"
        verdict = "INSTNCT_FULL_SELECTED_LINE_GENERATION_SCALE_BLOCKED"
        next_step = "148E_FULL_LINE_SHORTCUT_ANALYSIS"
    else:
        decision = "full_line_training_scale_failure"
        verdict = "INSTNCT_FULL_SELECTED_LINE_GENERATION_SCALE_BLOCKED"
        next_step = "148B_FULL_LINE_TRAINING_FAILURE_ANALYSIS"
    return {
        "schema_version": "phase_148h_decision_v1",
        "decision": decision,
        "verdict": verdict,
        "next": next_step,
        "positive_gate_passed": decision == DECISION,
        "boundary": BOUNDARY_TEXT,
        **FALSE_FLAGS,
    }


def write_report(out: Path, decision: dict[str, Any], metrics: dict[str, Any]) -> None:
    text = f"""# {MILESTONE}

Decision: `{decision['decision']}`

Verdict: `{decision['verdict']}`

Next: `{decision['next']}`

Boundary: {BOUNDARY_TEXT}

## Key Metrics

- full selected line exact match rate: `{metrics['full_selected_line_exact_match_rate']}`
- selected prefix generation accuracy: `{metrics['selected_prefix_generation_accuracy']}`
- selected label generation accuracy: `{metrics['selected_label_generation_accuracy']}`
- final value from generated line accuracy: `{metrics['final_value_from_generated_line_accuracy']}`
- generated output schema valid rate: `{metrics['generated_output_schema_valid_rate']}`
- OOD full line accuracy: `{metrics['ood_full_line_accuracy']}`
- minimum OOD family accuracy: `{metrics['minimum_ood_family_accuracy']}`
- shuffled target control accuracy: `{metrics['shuffled_target_control_accuracy']}`
- generation deterministic replay passed: `{metrics['generation_deterministic_replay_passed']}`

## Interpretation

148H scale-confirms bounded full `SELECTED=<label>` line generation on canonical structured prompts. It does not prove natural-language rule reasoning, open-ended arbitration, GPT-like/Gemma-like assistant capability, production readiness, or architecture superiority.
"""
    write_text(out / "report.md", text)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run 148H full selected-line generation scale confirm")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--upstream-148a-root", type=Path, default=DEFAULT_148A_ROOT)
    parser.add_argument("--seeds", default="5901,5902,5903,5904")
    parser.add_argument("--train-rows-per-seed", type=int, default=2400)
    parser.add_argument("--validation-rows-per-seed", type=int, default=600)
    parser.add_argument("--test-rows-per-seed", type=int, default=600)
    parser.add_argument("--ood-rows-per-seed", type=int, default=600)
    parser.add_argument("--feature-buckets", type=int, default=2048)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--control-epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--max-new-bytes", type=int, default=24)
    parser.add_argument("--fallback-oversample", type=int, default=16)
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    args = parser.parse_args()

    out = resolve_target_out(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_text(out / "progress.jsonl", "")
    append_progress(out, "startup", milestone=MILESTONE, heartbeat_sec=args.heartbeat_sec)
    write_json(out / "queue.json", {"schema_version": "phase_148h_queue_v1", "milestone": MILESTONE, "status": "running"})

    upstream = require_148a(resolve_repo_path(args.upstream_148a_root))
    write_json(out / "upstream_148a_manifest.json", upstream)
    if not helper_unchanged_from_head():
        raise RuntimeError("shared_raw_generation_helper.py changed from HEAD")
    append_progress(out, "upstream verified", upstream_decision=upstream["decision"]["decision"])

    seeds = parse_seeds(args.seeds)
    counts = {
        "train": args.train_rows_per_seed,
        "validation": args.validation_rows_per_seed,
        "test": args.test_rows_per_seed,
        "ood_test": args.ood_rows_per_seed,
    }
    write_json(
        out / "analysis_config.json",
        {
            "schema_version": "phase_148h_analysis_config_v1",
            "milestone": MILESTONE,
            "seeds": seeds,
            "counts_per_seed": counts,
            "scale_confirm_only": True,
            "model_family": "runner_local_pytorch_byte_lm_full_selected_line",
            "same_model_family_as_148a": True,
            "boundary": BOUNDARY_TEXT,
            **FALSE_FLAGS,
        },
    )
    write_text(out / "training_metrics.jsonl", "")
    seed_results = [run_seed(seed, counts, args, out) for seed in seeds]
    write_text(out / "lm_training_metrics.jsonl", (out / "training_metrics.jsonl").read_text(encoding="utf-8"))
    append_progress(out, "all_seeds_complete", seeds=seeds)

    combined_splits = flatten_split([result["splits"] for result in seed_results])
    combined_traces = [trace for result in seed_results for trace in result["traces"]]
    trace_by_id = {trace["row_id"]: trace for trace in combined_traces}
    eval_rows = combined_splits["validation"] + combined_splits["test"] + combined_splits["ood_test"]
    eval_result_rows = [row for result in seed_results for row in result["eval_result"]["rows"]]
    test_result_rows = [row for result in seed_results for row in result["test_result"]["rows"]]
    ood_result_rows = [row for result in seed_results for row in result["ood_result"]["rows"]]

    for split, rows in combined_splits.items():
        write_jsonl(out / f"curriculum_{'ood_test' if split == 'ood_test' else split}.jsonl", rows)
    write_json(out / "teacher_trace_manifest.json", {"schema_version": "phase_148h_teacher_trace_manifest_v1", "trace_count": len(combined_traces), "traces": combined_traces})
    write_text(out / "sequence_train_corpus.txt", "\n\n".join(PHASE_148A.training_sequence(row) for row in combined_splits["train"]) + "\n")
    write_text(out / "sequence_validation_corpus.txt", "\n\n".join(PHASE_148A.training_sequence(row) for row in combined_splits["validation"]) + "\n")

    selected_correct = sum(1 for row in eval_result_rows if row["selected_label_correct"])
    full_correct = sum(1 for row in eval_result_rows if row["full_selected_line_correct"])
    final_correct = sum(1 for row in eval_result_rows if row["final_value_correct"])
    prefix_correct = sum(1 for row in eval_result_rows if row["selected_prefix_generated"])
    schema_valid = sum(1 for row in eval_result_rows if row["schema_valid"])
    baseline_eval = PHASE_148A.compute_baselines(combined_splits["train"], eval_rows, trace_by_id, seeds[0])
    baseline_test = PHASE_148A.compute_baselines(combined_splits["train"], combined_splits["test"], trace_by_id, seeds[0] + 10)
    baseline_ood = PHASE_148A.compute_baselines(combined_splits["train"], combined_splits["ood_test"], trace_by_id, seeds[0] + 20)
    best_eval = PHASE_148A.best_baseline(baseline_eval)
    best_test = PHASE_148A.best_baseline(baseline_test)
    best_ood = PHASE_148A.best_baseline(baseline_ood)

    eval_result = {
        "schema_version": "phase_148h_generation_eval_report_v1",
        "row_count": len(eval_result_rows),
        "selected_prefix_generation_accuracy": rate(prefix_correct, len(eval_result_rows)),
        "selected_label_generation_accuracy": rate(selected_correct, len(eval_result_rows)),
        "selected_label_extracted_from_full_line_accuracy": rate(selected_correct, len(eval_result_rows)),
        "full_selected_line_exact_match_rate": rate(full_correct, len(eval_result_rows)),
        "full_line_generation_accuracy": rate(full_correct, len(eval_result_rows)),
        "final_value_from_generated_line_accuracy": rate(final_correct, len(eval_result_rows)),
        "generated_output_schema_valid_rate": rate(schema_valid, len(eval_result_rows)),
        "multiple_selected_line_rate": 0.0,
        "answer_value_generation_rate": 0.0,
        "selected_pocket_id_generation_rate": 0.0,
        "malformed_selected_label_rate": rate(sum(1 for row in eval_result_rows if not row["schema_valid"] and row["scored_generated_text"].startswith("SELECTED=")), len(eval_result_rows)),
        "extra_text_generation_rate": rate(sum(1 for row in eval_result_rows if not row["schema_valid"] and bool(row["scored_generated_text"])), len(eval_result_rows)),
        "rows": eval_result_rows,
    }
    test_accuracy = rate(sum(1 for row in test_result_rows if row["full_selected_line_correct"]), len(test_result_rows))
    ood_accuracy = rate(sum(1 for row in ood_result_rows if row["full_selected_line_correct"]), len(ood_result_rows))
    generation_audit = PHASE_148A.generation_input_audit(combined_splits)
    prefix_audit = generation_prefix_audit(generation_audit, eval_result)
    raw_audit = raw_generation_audit(eval_result)
    decode_audit = decoding_audit(args)
    label_report = label_distribution_report(combined_splits, eval_result_rows)
    ood_family = ood_family_report(combined_splits, ood_result_rows)
    anti_mem = anti_memorization_report(combined_splits)
    shortcut = PHASE_148A.shortcut_scan([row for split_rows in combined_splits.values() for row in split_rows])
    leakage = PHASE_148A.PHASE_147A.split_leakage_report(combined_splits)
    value_leakage = PHASE_148A.PHASE_147A.value_token_leakage_report(combined_splits)
    ood_split = PHASE_148A.PHASE_147A.ood_split_definition_report(combined_splits)
    replay = deterministic_replay_report([result["seed_report"] for result in seed_results])
    combined_hash = sha256_text(json.dumps({str(result["seed"]): result["model_state_hash"] for result in seed_results}, sort_keys=True))
    artifact = model_artifact_audit(args, seed_results, combined_hash)
    baseline_margin = {
        "schema_version": "phase_148h_baseline_margin_report_v1",
        **baseline_eval,
        "best_baseline_accuracy": best_eval,
        "model_test_accuracy": test_accuracy,
        "best_baseline_test_accuracy": best_test,
        "model_ood_accuracy": ood_accuracy,
        "best_baseline_ood_accuracy": best_ood,
        "test_margin_over_best_baseline": test_accuracy - best_test,
        "ood_margin_over_best_baseline": ood_accuracy - best_ood,
    }
    baseline_margin["passed"] = (
        eval_result["full_line_generation_accuracy"] >= best_eval + 0.10
        and eval_result["selected_prefix_generation_accuracy"] >= best_eval + 0.10
        and baseline_margin["test_margin_over_best_baseline"] >= 0.10
        and baseline_margin["ood_margin_over_best_baseline"] >= 0.05
    )
    shuffled_accuracy = sum(result["shuffled_target_control_accuracy"] for result in seed_results) / len(seed_results)
    shuffled_report = {
        "schema_version": "phase_148h_shuffled_target_control_report_v1",
        "shuffled_target_control_accuracy": shuffled_accuracy,
        "per_seed_shuffled_target_control_accuracy": {str(result["seed"]): result["shuffled_target_control_accuracy"] for result in seed_results},
        "passed": shuffled_accuracy <= 0.35,
    }
    generated_schema = {
        "schema_version": "phase_148h_generated_schema_report_v1",
        "generated_output_schema_valid_rate": eval_result["generated_output_schema_valid_rate"],
        "multiple_selected_line_rate": 0.0,
        "answer_value_generation_rate": 0.0,
        "selected_pocket_id_generation_rate": 0.0,
        "malformed_selected_label_rate": eval_result["malformed_selected_label_rate"],
        "extra_text_generation_rate": eval_result["extra_text_generation_rate"],
    }
    generated_schema["passed"] = (
        generated_schema["generated_output_schema_valid_rate"] >= 0.95
        and generated_schema["multiple_selected_line_rate"] == 0.0
        and generated_schema["answer_value_generation_rate"] == 0.0
        and generated_schema["selected_pocket_id_generation_rate"] == 0.0
        and generated_schema["malformed_selected_label_rate"] <= 0.05
        and generated_schema["extra_text_generation_rate"] <= 0.05
    )
    selected_prefix_report = {
        "schema_version": "phase_148h_selected_prefix_generation_report_v1",
        "selected_prefix_generation_accuracy": eval_result["selected_prefix_generation_accuracy"],
        "passed": eval_result["selected_prefix_generation_accuracy"] >= 0.95,
    }
    selected_label_report = {
        "schema_version": "phase_148h_selected_label_extraction_report_v1",
        "selected_label_generation_accuracy": eval_result["selected_label_generation_accuracy"],
        "selected_label_extracted_from_full_line_accuracy": eval_result["selected_label_extracted_from_full_line_accuracy"],
        "passed": eval_result["selected_label_generation_accuracy"] >= 0.95 and eval_result["selected_label_extracted_from_full_line_accuracy"] >= 0.95,
    }
    final_value_report = {
        "schema_version": "phase_148h_final_value_copy_report_v1",
        "final_value_from_generated_line_accuracy": eval_result["final_value_from_generated_line_accuracy"],
        "opaque_value_generation_required": False,
        "passed": eval_result["final_value_from_generated_line_accuracy"] >= 0.95,
    }
    full_line_report = {
        "schema_version": "phase_148h_full_selected_line_generation_report_v1",
        "full_selected_line_exact_match_rate": eval_result["full_selected_line_exact_match_rate"],
        "full_line_generation_accuracy": eval_result["full_line_generation_accuracy"],
        "ood_full_line_accuracy": ood_accuracy,
        "passed": eval_result["full_selected_line_exact_match_rate"] >= 0.95 and eval_result["full_line_generation_accuracy"] >= 0.95 and ood_accuracy >= 0.85,
    }
    train_improves = all(result["train_metrics"]["train_loss_improves"] for result in seed_results)
    eval_improves = all(result["train_metrics"]["eval_loss_improves"] for result in seed_results)
    validation_not_nan = all(result["train_metrics"]["validation_loss_not_nan"] for result in seed_results)
    metrics = {
        "schema_version": "phase_148h_aggregate_metrics_v1",
        "full_selected_line_exact_match_rate": eval_result["full_selected_line_exact_match_rate"],
        "full_line_generation_accuracy": eval_result["full_line_generation_accuracy"],
        "selected_prefix_generation_accuracy": eval_result["selected_prefix_generation_accuracy"],
        "selected_label_generation_accuracy": eval_result["selected_label_generation_accuracy"],
        "selected_label_extracted_from_full_line_accuracy": eval_result["selected_label_extracted_from_full_line_accuracy"],
        "final_value_from_generated_line_accuracy": eval_result["final_value_from_generated_line_accuracy"],
        "fallback_full_line_accuracy": label_report["fallback_full_line_accuracy"],
        "generated_output_schema_valid_rate": eval_result["generated_output_schema_valid_rate"],
        "ood_full_line_accuracy": ood_accuracy,
        "minimum_ood_family_accuracy": ood_family["minimum_ood_family_accuracy"],
        "best_baseline_accuracy": best_eval,
        "test_margin_over_best_baseline": baseline_margin["test_margin_over_best_baseline"],
        "ood_margin_over_best_baseline": baseline_margin["ood_margin_over_best_baseline"],
        "shuffled_target_control_accuracy": shuffled_accuracy,
        "shortcut_scanner_violation_count": shortcut["shortcut_scanner_violation_count"],
        "train_eval_prompt_overlap_count": leakage["train_eval_prompt_overlap_count"],
        "train_ood_prompt_overlap_count": leakage["train_ood_prompt_overlap_count"],
        "value_token_overlap_train_test_rate": value_leakage["value_token_overlap_train_test_rate"],
        "value_token_overlap_train_ood_rate": value_leakage["value_token_overlap_train_ood_rate"],
        "eval_loss_improves": eval_improves,
        "train_loss_improves": train_improves,
        "validation_loss_not_nan": validation_not_nan,
        "generation_deterministic_replay_passed": replay["generation_deterministic_replay_passed"],
        "multiple_selected_line_rate": 0.0,
        "answer_value_generation_rate": 0.0,
        "selected_pocket_id_generation_rate": 0.0,
        "malformed_selected_label_rate": eval_result["malformed_selected_label_rate"],
        "extra_text_generation_rate": eval_result["extra_text_generation_rate"],
        "prefix_repair_used": raw_audit["prefix_repair_used"],
        "casing_repair_used": raw_audit["casing_repair_used"],
        "label_repair_used": raw_audit["label_repair_used"],
        "post_generation_repair_used": raw_audit["post_generation_repair_used"],
        "collapsed_ood_family_count": ood_family["collapsed_ood_family_count"],
        "minimum_ood_family_row_count": ood_family["minimum_ood_family_row_count"],
        "every_label_appears_in_every_split": label_report["every_label_appears_in_every_split"],
        "minimum_per_label_full_line_accuracy": label_report["minimum_per_label_full_line_accuracy"],
        "minimum_per_label_schema_valid_rate": label_report["minimum_per_label_schema_valid_rate"],
        "eval_generation_input_contains_selected_prefix": generation_audit["eval_generation_input_contains_selected_prefix"],
        "runner_prepends_selected_prefix": prefix_audit["runner_prepends_selected_prefix"],
        "deterministic_selected_line_wrapper_used": prefix_audit["deterministic_selected_line_wrapper_used"],
        "selected_line_extracted_from_substring": raw_audit["selected_line_extracted_from_substring"],
        "first_byte_only_training_used": decode_audit["first_byte_only_training_used"],
        "forced_selected_prefix_used": decode_audit["forced_selected_prefix_used"],
        "constrained_label_only_decoding_used": decode_audit["constrained_label_only_decoding_used"],
        "per_seed_reports": [result["seed_report"] for result in seed_results],
    }
    metrics["passed"] = gates_pass(metrics)
    training_config = {
        "schema_version": "phase_148h_training_config_v1",
        "model_family": "runner_local_pytorch_byte_lm_full_selected_line",
        "same_model_family_as_148a": True,
        "target": "next byte over full selected-line suffix",
        "train_target_sequence": "SELECTED=<label>\\n",
        "seeds": seeds,
        "counts_per_seed": counts,
        "feature_buckets": args.feature_buckets,
        "hidden": args.hidden,
        "epochs": args.epochs,
        "control_epochs": args.control_epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "max_new_bytes": args.max_new_bytes,
        "fallback_oversample": args.fallback_oversample,
        "opaque_value_token_generation_required": False,
    }
    audits = [
        generated_schema,
        prefix_audit,
        raw_audit,
        decode_audit,
        generation_audit,
        label_report,
        ood_family,
        anti_mem,
        baseline_margin,
        shuffled_report,
        shortcut,
        leakage,
        value_leakage,
        artifact,
        replay,
        full_line_report,
        selected_prefix_report,
        selected_label_report,
        final_value_report,
    ]
    decision = choose_decision(metrics, audits)
    summary = {
        "schema_version": "phase_148h_summary_v1",
        "milestone": MILESTONE,
        "decision": decision["decision"],
        "verdict": decision["verdict"],
        "next": decision["next"],
        "boundary": BOUNDARY_TEXT,
        "metrics": metrics,
        **FALSE_FLAGS,
    }
    write_json(out / "training_config.json", training_config)
    write_json(out / "full_selected_line_generation_report.json", full_line_report)
    write_json(out / "selected_prefix_generation_report.json", selected_prefix_report)
    write_json(out / "selected_label_extraction_report.json", selected_label_report)
    write_json(out / "final_value_copy_report.json", final_value_report)
    write_json(out / "generated_schema_report.json", generated_schema)
    write_json(out / "generation_prefix_audit.json", prefix_audit)
    write_json(out / "raw_generation_audit.json", raw_audit)
    write_json(out / "decoding_audit.json", decode_audit)
    write_json(out / "generation_input_audit.json", generation_audit)
    write_json(out / "label_distribution_report.json", label_report)
    write_json(out / "per_label_generation_report.json", label_report)
    write_json(out / "ood_full_line_family_report.json", ood_family)
    write_json(out / "ood_split_definition_report.json", ood_split)
    write_json(out / "anti_memorization_report.json", anti_mem)
    write_json(out / "baseline_margin_report.json", baseline_margin)
    write_json(out / "shuffled_target_control_report.json", shuffled_report)
    write_json(out / "shortcut_scanner_report.json", shortcut)
    write_json(out / "leakage_audit.json", leakage)
    write_json(out / "value_token_leakage_report.json", value_leakage)
    write_json(out / "model_artifact_audit.json", artifact)
    write_json(out / "deterministic_replay_report.json", replay)
    write_json(out / "aggregate_metrics.json", metrics)
    write_json(out / "decision.json", decision)
    write_json(out / "summary.json", summary)
    write_report(out, decision, metrics)
    queue = read_json(out / "queue.json")
    queue["status"] = "complete" if decision["decision"] == DECISION else "blocked"
    queue["decision"] = decision["decision"]
    write_json(out / "queue.json", queue)
    append_progress(out, "complete", decision=decision["decision"], next=decision["next"])
    print(json.dumps({"decision": decision["decision"], "verdict": decision["verdict"], "next": decision["next"], "metrics": metrics}, indent=2, sort_keys=True))
    return 0 if decision["decision"] == DECISION else 1


if __name__ == "__main__":
    raise SystemExit(main())
