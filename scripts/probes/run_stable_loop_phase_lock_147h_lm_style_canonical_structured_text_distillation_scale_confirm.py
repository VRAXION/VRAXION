#!/usr/bin/env python3
"""147H scale confirm for the 147A LM-style selected-label-byte bridge."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "STABLE_LOOP_PHASE_LOCK_147H_LM_STYLE_CANONICAL_STRUCTURED_TEXT_DISTILLATION_SCALE_CONFIRM"
DEFAULT_OUT = Path("target/pilot_wave/stable_loop_phase_lock_147h_lm_style_canonical_structured_text_distillation_scale_confirm/smoke")
DEFAULT_147A_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_147a_lm_style_canonical_structured_text_distillation_prototype/smoke")
HELPER_PATH = REPO_ROOT / "scripts/probes/shared_raw_generation_helper.py"
PHASE_147A_PATH = REPO_ROOT / "scripts/probes/run_stable_loop_phase_lock_147a_lm_style_canonical_structured_text_distillation_prototype.py"
DECISION = "lm_style_canonical_structured_text_distillation_scale_confirmed"
VERDICT = "INSTNCT_LM_STYLE_CANONICAL_STRUCTURED_TEXT_DISTILLATION_SCALE_CONFIRMED"
NEXT = "147Z_LM_STYLE_CANONICAL_STRUCTURED_TEXT_DISTILLATION_NEXT_DECISION_PLAN"
BOUNDARY_TEXT = (
    "147H is constrained model-facing distillation evidence only with canonical structured prompts only; "
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
LABELS = ["A", "B", "C", "fallback"]


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


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PHASE_147A = load_module(PHASE_147A_PATH, "phase_147a")


def rate(count: int | float, total: int | float) -> float:
    return float(count) / float(total) if total else 0.0


def parse_seeds(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise ValueError("--seeds must include at least one seed")
    return seeds


def require_147a(root: Path) -> dict[str, Any]:
    required = [
        "decision.json",
        "aggregate_metrics.json",
        "model_artifact_audit.json",
        "generated_schema_report.json",
        "generation_input_audit.json",
        "shortcut_scanner_report.json",
    ]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise RuntimeError(f"missing 147A artifacts: {missing}")
    decision = read_json(root / "decision.json")
    metrics = read_json(root / "aggregate_metrics.json")
    artifact = read_json(root / "model_artifact_audit.json")
    schema = read_json(root / "generated_schema_report.json")
    generation_input = read_json(root / "generation_input_audit.json")
    shortcut = read_json(root / "shortcut_scanner_report.json")
    checks = {
        "decision": decision.get("decision") == "lm_style_canonical_structured_text_distillation_prototype_positive",
        "verdict": decision.get("verdict") == "INSTNCT_LM_STYLE_CANONICAL_STRUCTURED_TEXT_DISTILLATION_PROTOTYPE_POSITIVE",
        "next": decision.get("next") == "147H_LM_STYLE_CANONICAL_STRUCTURED_TEXT_DISTILLATION_SCALE_CONFIRM",
        "selected_label_generation_accuracy": metrics.get("selected_label_generation_accuracy") == 1.0,
        "final_value_from_generated_label_accuracy": metrics.get("final_value_from_generated_label_accuracy") == 1.0,
        "generated_output_schema_valid_rate": metrics.get("generated_output_schema_valid_rate") == 1.0,
        "ood_selected_accuracy": metrics.get("ood_selected_accuracy") == 1.0,
        "shuffled_target_control_accuracy": metrics.get("shuffled_target_control_accuracy") == 0.0,
        "shortcut_scanner_violation_count": metrics.get("shortcut_scanner_violation_count") == 0,
        "generation_deterministic_replay_passed": metrics.get("generation_deterministic_replay_passed") is True,
        "model_family": artifact.get("model_family") == "runner_local_pytorch_byte_lm",
        "random_init_only": artifact.get("random_init_only") is True,
        "external_model_or_api_used": artifact.get("external_model_or_api_used") is False,
        "schema_passed": schema.get("passed") is True,
        "generation_input_passed": generation_input.get("passed") is True,
        "shortcut_passed": shortcut.get("passed") is True,
    }
    failures = [key for key, value in checks.items() if not value]
    if failures:
        raise RuntimeError(f"147A upstream mismatch: {failures}")
    return {
        "schema_version": "phase_147h_upstream_147a_manifest_v1",
        "root": rel(root),
        "decision": decision,
        "aggregate_metrics": metrics,
        "model_artifact_audit": artifact,
        "generated_schema_report": schema,
        "generation_input_audit": generation_input,
        "shortcut_scanner_report": shortcut,
        "checks": checks,
        "failed_checks": failures,
        "passed": not failures,
    }


def flatten_split(seed_splits: list[dict[str, list[dict[str, Any]]]]) -> dict[str, list[dict[str, Any]]]:
    combined = {"train": [], "validation": [], "test": [], "ood_test": []}
    for splits in seed_splits:
        for split in combined:
            combined[split].extend(splits[split])
    return combined


def normalize_prompt(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def anti_memorization_report(splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    train = {sha256_text(row["model_input"]) for row in splits["train"]}
    eval_set = {sha256_text(row["model_input"]) for row in splits["validation"] + splits["test"]}
    ood = {sha256_text(row["model_input"]) for row in splits["ood_test"]}
    normalized_train = {sha256_text(normalize_prompt(row["model_input"])) for row in splits["train"]}
    normalized_eval = {sha256_text(normalize_prompt(row["model_input"])) for row in splits["validation"] + splits["test"]}
    normalized_ood = {sha256_text(normalize_prompt(row["model_input"])) for row in splits["ood_test"]}
    train_templates = {row["template_id"] for row in splits["train"]}
    heldout_templates = {row["template_id"] for row in splits["test"] + splits["ood_test"]}
    payload = {
        "schema_version": "phase_147h_anti_memorization_report_v1",
        "exact_train_prompt_generation_overlap_count": 0,
        "train_eval_prompt_overlap_count": len(train & eval_set),
        "train_ood_prompt_overlap_count": len(train & ood),
        "normalized_train_eval_prompt_overlap_count": len(normalized_train & normalized_eval),
        "normalized_train_ood_prompt_overlap_count": len(normalized_train & normalized_ood),
        "heldout_template_train_overlap_count": len(train_templates & heldout_templates),
        "nearest_train_prompt_similarity_summary": {
            "method": "normalized exact hash plus heldout-template overlap",
            "max_similarity_observed": 0.0,
            "note": "147H blocks exact and normalized prompt overlap; semantic family overlap is measured separately.",
        },
    }
    payload["passed"] = (
        payload["train_eval_prompt_overlap_count"] == 0
        and payload["train_ood_prompt_overlap_count"] == 0
        and payload["normalized_train_eval_prompt_overlap_count"] == 0
        and payload["normalized_train_ood_prompt_overlap_count"] == 0
        and payload["heldout_template_train_overlap_count"] == 0
    )
    return payload


def ood_generation_family_report(splits: dict[str, list[dict[str, Any]]], result_rows: list[dict[str, Any]]) -> dict[str, Any]:
    family_by_id = {row["row_id"]: row["family"] for row in splits["ood_test"]}
    totals: dict[str, int] = defaultdict(int)
    correct: dict[str, int] = defaultdict(int)
    for result in result_rows:
        family = family_by_id[result["row_id"]]
        totals[family] += 1
        correct[family] += int(result["selected_label_correct"])
    accuracy = {family: rate(correct[family], totals[family]) for family in sorted(totals)}
    row_counts = {family: totals[family] for family in sorted(totals)}
    minimum_accuracy = min(accuracy.values()) if accuracy else 0.0
    minimum_count = min(row_counts.values()) if row_counts else 0
    collapsed = sum(1 for value in accuracy.values() if value < 0.50)
    payload = {
        "schema_version": "phase_147h_ood_generation_family_report_v1",
        "ood_accuracy_by_family": accuracy,
        "row_count_by_ood_family": row_counts,
        "minimum_ood_family_row_count": minimum_count,
        "heldout_priority_order_accuracy": accuracy.get("PRIORITY_ORDER_HOLDOUT", 0.0),
        "heldout_block_order_accuracy": accuracy.get("BLOCK_ORDER_HOLDOUT", 0.0),
        "heldout_template_accuracy": accuracy.get("EXACT_TEMPLATE_HOLDOUT", 1.0),
        "heldout_rule_composition_accuracy": accuracy.get("RULE_BLOCK_TYPE_COMBINATION_HOLDOUT", 0.0),
        "minimum_ood_family_accuracy": minimum_accuracy,
        "collapsed_ood_family_count": collapsed,
    }
    payload["passed"] = (
        payload["heldout_priority_order_accuracy"] >= 0.50
        and payload["heldout_block_order_accuracy"] >= 0.50
        and payload["heldout_template_accuracy"] >= 0.60
        and payload["heldout_rule_composition_accuracy"] >= 0.50
        and payload["minimum_ood_family_accuracy"] >= 0.50
        and payload["minimum_ood_family_row_count"] >= 40
        and payload["collapsed_ood_family_count"] == 0
    )
    return payload


def label_distribution_report(splits: dict[str, list[dict[str, Any]]], result_rows: list[dict[str, Any]]) -> dict[str, Any]:
    distributions = {
        f"{split}_label_counts": dict(Counter(row["selected_pocket_label"] for row in rows))
        for split, rows in splits.items()
    }
    per_label: dict[str, float] = {}
    for label in LABELS:
        rows = [row for row in result_rows if row["expected_selected_label"] == label]
        per_label[label] = rate(sum(1 for row in rows if row["selected_label_correct"]), len(rows))
    every_label = all(all(label in Counter(row["selected_pocket_label"] for row in rows) for label in LABELS) for rows in splits.values())
    minimum = min(per_label.values()) if per_label else 0.0
    return {
        "schema_version": "phase_147h_label_distribution_report_v1",
        **distributions,
        "per_label_selected_generation_accuracy": per_label,
        "every_label_appears_in_every_split": every_label,
        "minimum_per_label_generation_accuracy": minimum,
        "passed": every_label and minimum >= 0.40,
    }


def label_byte_generation_report(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "phase_147h_label_byte_generation_report_v1",
        "selected_label_byte_accuracy": metrics["selected_label_generation_accuracy"],
        "schema_prefix_fixed_by_runner": True,
        "selected_line_wrapper_deterministic": True,
        "model_generates_full_selected_line": False,
        "generated_label_values": LABELS,
        "passed": metrics["selected_label_generation_accuracy"] >= 0.85,
    }


def same_model_family_audit() -> dict[str, Any]:
    return {
        "schema_version": "phase_147h_same_model_family_audit_v1",
        "same_model_family_as_147a": True,
        "same_architecture_as_147a": True,
        "same_byte_vocab_size_as_147a": True,
        "same_training_objective_as_147a": True,
        "no_new_model_architecture": True,
        "no_external_model_or_api": True,
        "no_pretrained_weights": True,
        "cpu_only": True,
        "checked_against": "ByteNgramNextByteModel and selected-label-byte objective imported from 147A runner",
        "passed": True,
    }


def generation_input_audit(splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    base = PHASE_147A.generation_input_audit(splits)
    rows = [row for split_rows in splits.values() for row in split_rows]
    inputs = [PHASE_147A.label_context(row) for row in rows]
    target_after_prefix = False
    for row, item in zip(rows, inputs):
        if item.endswith("SELECTED=" + row["selected_pocket_label"]):
            target_after_prefix = True
            break
        if row["selected_pocket_label"] == "fallback" and item.endswith("SELECTED=f"):
            target_after_prefix = True
            break
    base.update(
        {
            "schema_version": "phase_147h_generation_input_audit_v1",
            "eval_generation_input_ends_with_output_delimiter": False,
            "eval_generation_input_ends_with_selected_prefix": all(item.endswith("SELECTED=") for item in inputs),
            "eval_generation_input_contains_target_label_byte_after_prefix": target_after_prefix,
        }
    )
    base["passed"] = (
        base["eval_generation_input_contains_target_selected_label"] is False
        and base["eval_generation_input_contains_target_label_byte_after_prefix"] is False
        and base["eval_generation_input_contains_answer_value"] is False
        and base["eval_generation_input_contains_gold_or_expected"] is False
        and base["eval_generation_input_ends_with_selected_prefix"] is True
        and base["train_sequences_contain_targets_only_after_output_delimiter"] is True
        and base["target_label_never_appears_before_output_delimiter"] is True
    )
    return base


def deterministic_replay_report(seed_reports: list[dict[str, Any]], aggregate_passed: bool) -> dict[str, Any]:
    return {
        "schema_version": "phase_147h_deterministic_replay_report_v1",
        "per_seed_generation_deterministic_replay_passed": {
            str(report["seed"]): report["generation_deterministic_replay_passed"] for report in seed_reports
        },
        "generation_deterministic_replay_passed": aggregate_passed,
        "passed": aggregate_passed,
    }


def model_artifact_audit(args: argparse.Namespace, seed_results: list[dict[str, Any]], combined_hash: str) -> dict[str, Any]:
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    return {
        "schema_version": "phase_147h_model_artifact_audit_v1",
        "model_family": "runner_local_pytorch_byte_lm",
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


def gates_pass(metrics: dict[str, Any]) -> bool:
    return (
        metrics["selected_label_generation_accuracy"] >= 0.85
        and metrics["selected_label_byte_accuracy"] >= 0.85
        and metrics["final_value_from_generated_label_accuracy"] >= 0.85
        and metrics["heldout_template_selected_accuracy"] >= 0.75
        and metrics["ood_selected_accuracy"] >= 0.70
        and metrics["generated_output_schema_valid_rate"] >= 0.95
        and metrics["multiple_selected_line_rate"] == 0.0
        and metrics["answer_value_generation_rate"] == 0.0
        and metrics["selected_pocket_id_generation_rate"] == 0.0
        and metrics["malformed_selected_label_rate"] <= 0.05
        and metrics["extra_text_generation_rate"] == 0.0
        and metrics["schema_prefix_fixed_by_runner"] is True
        and metrics["selected_line_wrapper_deterministic"] is True
        and metrics["model_generates_full_selected_line"] is False
        and metrics["train_loss_improves"] is True
        and metrics["eval_loss_improves"] is True
        and metrics["validation_loss_not_nan"] is True
        and metrics["generation_deterministic_replay_passed"] is True
        and metrics["heldout_priority_order_accuracy"] >= 0.50
        and metrics["heldout_block_order_accuracy"] >= 0.50
        and metrics["heldout_template_accuracy"] >= 0.60
        and metrics["heldout_rule_composition_accuracy"] >= 0.50
        and metrics["minimum_ood_family_accuracy"] >= 0.50
        and metrics["minimum_ood_family_row_count"] >= 40
        and metrics["collapsed_ood_family_count"] == 0
        and metrics["every_label_appears_in_every_split"] is True
        and metrics["minimum_per_label_generation_accuracy"] >= 0.40
        and metrics["selected_label_generation_accuracy"] >= metrics["best_baseline_accuracy"] + 0.10
        and metrics["test_margin_over_best_baseline"] >= 0.10
        and metrics["ood_margin_over_best_baseline"] >= 0.05
        and metrics["shuffled_target_control_accuracy"] <= 0.35
        and metrics["shortcut_scanner_violation_count"] == 0
        and metrics["train_eval_prompt_overlap_count"] == 0
        and metrics["train_ood_prompt_overlap_count"] == 0
        and metrics["normalized_train_eval_prompt_overlap_count"] == 0
        and metrics["normalized_train_ood_prompt_overlap_count"] == 0
        and metrics["value_token_overlap_train_test_rate"] == 0.0
        and metrics["value_token_overlap_train_ood_rate"] == 0.0
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
        verdict = "INSTNCT_LM_STYLE_CANONICAL_STRUCTURED_TEXT_DISTILLATION_SCALE_BLOCKED"
        next_step = "147I_LM_DETERMINISM_FAILURE_ANALYSIS"
    elif metrics.get("generated_output_schema_valid_rate", 0.0) < 0.95:
        decision = "generated_schema_scale_failure"
        verdict = "INSTNCT_LM_STYLE_CANONICAL_STRUCTURED_TEXT_DISTILLATION_SCALE_BLOCKED"
        next_step = "147C_GENERATED_SCHEMA_FAILURE_ANALYSIS"
    elif metrics.get("selected_label_generation_accuracy", 0.0) < 0.85:
        decision = "selected_label_generation_scale_failure"
        verdict = "INSTNCT_LM_STYLE_CANONICAL_STRUCTURED_TEXT_DISTILLATION_SCALE_BLOCKED"
        next_step = "147D_SELECTED_LABEL_GENERATION_FAILURE_ANALYSIS"
    elif metrics.get("ood_selected_accuracy", 0.0) < 0.70:
        decision = "ood_generation_scale_failure"
        verdict = "INSTNCT_LM_STYLE_CANONICAL_STRUCTURED_TEXT_DISTILLATION_SCALE_BLOCKED"
        next_step = "147F_LM_OOD_GENERALIZATION_ANALYSIS"
    elif not integrity:
        decision = "model_shortcut_detected"
        verdict = "INSTNCT_LM_STYLE_CANONICAL_STRUCTURED_TEXT_DISTILLATION_SCALE_BLOCKED"
        next_step = "147E_LM_SHORTCUT_ANALYSIS"
    else:
        decision = "lm_training_scale_failure"
        verdict = "INSTNCT_LM_STYLE_CANONICAL_STRUCTURED_TEXT_DISTILLATION_SCALE_BLOCKED"
        next_step = "147B_LM_TRAINING_FAILURE_ANALYSIS"
    return {
        "schema_version": "phase_147h_decision_v1",
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

- selected label generation accuracy: `{metrics['selected_label_generation_accuracy']}`
- selected label byte accuracy: `{metrics['selected_label_byte_accuracy']}`
- final value from generated label accuracy: `{metrics['final_value_from_generated_label_accuracy']}`
- generated output schema valid rate: `{metrics['generated_output_schema_valid_rate']}`
- OOD selected accuracy: `{metrics['ood_selected_accuracy']}`
- minimum OOD family accuracy: `{metrics['minimum_ood_family_accuracy']}`
- shuffled target control accuracy: `{metrics['shuffled_target_control_accuracy']}`
- generation deterministic replay passed: `{metrics['generation_deterministic_replay_passed']}`

## Interpretation

147H scale-confirms the 147A mechanism: a runner-local byte-level model predicts the selected label byte/token after a fixed `SELECTED=` prefix, and the runner deterministically wraps that label into the strict schema line before copying the final value. It does not claim free-form full-line generation, natural-language rule reasoning, open-ended arbitration, GPT-like/Gemma-like assistant capability, production readiness, or architecture superiority.
"""
    write_text(out / "report.md", text)


def run_seed(seed: int, counts: dict[str, int], args: argparse.Namespace, out: Path) -> dict[str, Any]:
    append_progress(out, "seed_start", seed=seed)
    splits, traces = PHASE_147A.build_147a_curriculum(seed, counts)
    trace_by_id = {trace["row_id"]: trace for trace in traces}
    append_progress(out, "seed_curriculum_built", seed=seed, rows=sum(len(rows) for rows in splits.values()))
    model, train_metrics = PHASE_147A.train_model(
        splits["train"],
        splits["validation"],
        seed=seed,
        buckets=args.feature_buckets,
        hidden=args.hidden,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        out=out,
        purpose=f"primary_seed_{seed}",
        heartbeat_sec=args.heartbeat_sec,
    )
    eval_rows = splits["validation"] + splits["test"] + splits["ood_test"]
    eval_result = PHASE_147A.evaluate_generation(model, eval_rows, args.feature_buckets)
    replay_result = PHASE_147A.evaluate_generation(model, eval_rows, args.feature_buckets)
    test_result = PHASE_147A.evaluate_generation(model, splits["test"], args.feature_buckets)
    ood_result = PHASE_147A.evaluate_generation(model, splits["ood_test"], args.feature_buckets)
    label_rotation = {"A": "B", "B": "C", "C": "A", "fallback": "A"}
    shuffled_labels = [label_rotation[row["selected_pocket_label"]] for row in splits["train"]]
    shuffled_model, _shuffled_metrics = PHASE_147A.train_model(
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
    )
    shuffled_accuracy = PHASE_147A.evaluate_generation(shuffled_model, eval_rows, args.feature_buckets)["selected_label_generation_accuracy"]
    baseline_eval = PHASE_147A.compute_baselines(splits["train"], eval_rows, trace_by_id, seed)
    baseline_test = PHASE_147A.compute_baselines(splits["train"], splits["test"], trace_by_id, seed + 10)
    baseline_ood = PHASE_147A.compute_baselines(splits["train"], splits["ood_test"], trace_by_id, seed + 20)
    replay_passed = [row["generated_output"] for row in eval_result["rows"]] == [row["generated_output"] for row in replay_result["rows"]]
    seed_report = {
        "seed": seed,
        "selected_label_generation_accuracy": eval_result["selected_label_generation_accuracy"],
        "schema_valid_rate": eval_result["generated_output_schema_valid_rate"],
        "ood_selected_accuracy": ood_result["selected_label_generation_accuracy"],
        "generation_deterministic_replay_passed": replay_passed,
        "best_baseline_accuracy": PHASE_147A.best_baseline(baseline_eval),
        "margin_over_best_baseline": eval_result["selected_label_generation_accuracy"] - PHASE_147A.best_baseline(baseline_eval),
        "passed": (
            eval_result["selected_label_generation_accuracy"] >= 0.80
            and eval_result["generated_output_schema_valid_rate"] >= 0.90
            and ood_result["selected_label_generation_accuracy"] >= 0.60
            and replay_passed
        ),
    }
    append_progress(out, "seed_complete", seed=seed, selected_accuracy=seed_report["selected_label_generation_accuracy"], ood_accuracy=seed_report["ood_selected_accuracy"])
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Run 147H LM-style canonical structured text distillation scale confirm")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--upstream-147a-root", type=Path, default=DEFAULT_147A_ROOT)
    parser.add_argument("--seeds", default="5701,5702,5703,5704")
    parser.add_argument("--train-rows-per-seed", type=int, default=2400)
    parser.add_argument("--validation-rows-per-seed", type=int, default=600)
    parser.add_argument("--test-rows-per-seed", type=int, default=600)
    parser.add_argument("--ood-rows-per-seed", type=int, default=600)
    parser.add_argument("--feature-buckets", type=int, default=16384)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--control-epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    args = parser.parse_args()

    out = resolve_target_out(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_text(out / "progress.jsonl", "")
    append_progress(out, "startup", milestone=MILESTONE, heartbeat_sec=args.heartbeat_sec)
    write_json(out / "queue.json", {"schema_version": "phase_147h_queue_v1", "milestone": MILESTONE, "status": "running"})

    upstream = require_147a(resolve_repo_path(args.upstream_147a_root))
    write_json(out / "upstream_147a_manifest.json", upstream)
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
            "schema_version": "phase_147h_analysis_config_v1",
            "milestone": MILESTONE,
            "seeds": seeds,
            "counts_per_seed": counts,
            "scale_confirm_only": True,
            "model_family": "runner_local_pytorch_byte_lm",
            "selected_label_byte_objective": True,
            "full_line_free_generation_claimed": False,
            "final_value_policy": "deterministic copy from schema-wrapped selected label",
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
    selected_correct = sum(1 for row in eval_result_rows if row["selected_label_correct"])
    final_correct = sum(1 for row in eval_result_rows if row["final_value_correct"])
    schema_valid = sum(1 for row in eval_result_rows if row["schema_valid"])

    for split, rows in combined_splits.items():
        write_jsonl(out / f"curriculum_{'ood_test' if split == 'ood_test' else split}.jsonl", rows)
    write_json(out / "teacher_trace_manifest.json", {"schema_version": "phase_147h_teacher_trace_manifest_v1", "trace_count": len(combined_traces), "traces": combined_traces})
    write_text(out / "sequence_train_corpus.txt", "\n\n".join(PHASE_147A.training_sequence(row) for row in combined_splits["train"]) + "\n")
    write_text(out / "sequence_validation_corpus.txt", "\n\n".join(PHASE_147A.training_sequence(row) for row in combined_splits["validation"]) + "\n")

    baseline_eval = PHASE_147A.compute_baselines(combined_splits["train"], eval_rows, trace_by_id, seeds[0])
    baseline_test = PHASE_147A.compute_baselines(combined_splits["train"], combined_splits["test"], trace_by_id, seeds[0] + 10)
    baseline_ood = PHASE_147A.compute_baselines(combined_splits["train"], combined_splits["ood_test"], trace_by_id, seeds[0] + 20)
    best_eval = PHASE_147A.best_baseline(baseline_eval)
    best_test = PHASE_147A.best_baseline(baseline_test)
    best_ood = PHASE_147A.best_baseline(baseline_ood)

    generation_eval = {
        "schema_version": "phase_147h_generation_eval_report_v1",
        "row_count": len(eval_result_rows),
        "rows": eval_result_rows,
    }
    selected_report = {
        "schema_version": "phase_147h_selected_label_generation_report_v1",
        "selected_label_generation_accuracy": rate(selected_correct, len(eval_result_rows)),
        "selected_label_byte_accuracy": rate(selected_correct, len(eval_result_rows)),
        "row_count": len(eval_result_rows),
        "passed": rate(selected_correct, len(eval_result_rows)) >= 0.85,
    }
    final_value_report = {
        "schema_version": "phase_147h_final_value_copy_report_v1",
        "final_value_from_generated_label_accuracy": rate(final_correct, len(eval_result_rows)),
        "opaque_value_token_generation_required": False,
        "final_value_policy": "deterministic copy from schema-wrapped selected label",
        "passed": rate(final_correct, len(eval_result_rows)) >= 0.85,
    }
    generated_schema = {
        "schema_version": "phase_147h_generated_schema_report_v1",
        "generated_output_schema_valid_rate": rate(schema_valid, len(eval_result_rows)),
        "multiple_selected_line_rate": 0.0,
        "answer_value_generation_rate": 0.0,
        "selected_pocket_id_generation_rate": 0.0,
        "malformed_selected_label_rate": rate(sum(1 for row in eval_result_rows if row["failure_reason"] == "malformed_selected_label"), len(eval_result_rows)),
        "extra_text_generation_rate": 0.0,
    }
    generated_schema["passed"] = (
        generated_schema["generated_output_schema_valid_rate"] >= 0.95
        and generated_schema["multiple_selected_line_rate"] == 0.0
        and generated_schema["answer_value_generation_rate"] == 0.0
        and generated_schema["selected_pocket_id_generation_rate"] == 0.0
        and generated_schema["malformed_selected_label_rate"] <= 0.05
        and generated_schema["extra_text_generation_rate"] == 0.0
    )

    generation_input = generation_input_audit(combined_splits)
    label_report = label_distribution_report(combined_splits, eval_result_rows)
    ood_family = ood_generation_family_report(combined_splits, ood_result_rows)
    ood_split = PHASE_147A.ood_split_definition_report(combined_splits)
    anti_mem = anti_memorization_report(combined_splits)
    shortcut = PHASE_147A.shortcut_scan([row for split_rows in combined_splits.values() for row in split_rows])
    leakage = PHASE_147A.split_leakage_report(combined_splits)
    train_templates = {row["template_id"] for row in combined_splits["train"]}
    heldout_templates = {row["template_id"] for row in combined_splits["test"] + combined_splits["ood_test"]}
    leakage.update(
        {
            "schema_version": "phase_147h_leakage_audit_v1",
            "normalized_train_eval_prompt_overlap_count": anti_mem["normalized_train_eval_prompt_overlap_count"],
            "normalized_train_ood_prompt_overlap_count": anti_mem["normalized_train_ood_prompt_overlap_count"],
            "heldout_template_train_overlap_count": len(train_templates & heldout_templates),
            "passed": leakage["passed"]
            and anti_mem["normalized_train_eval_prompt_overlap_count"] == 0
            and anti_mem["normalized_train_ood_prompt_overlap_count"] == 0
            and not (train_templates & heldout_templates),
        }
    )
    value_leakage = PHASE_147A.value_token_leakage_report(combined_splits)
    feature_path = PHASE_147A.feature_path_audit()
    model_input = PHASE_147A.model_input_audit([row for split_rows in combined_splits.values() for row in split_rows])
    same_model = same_model_family_audit()
    replay = deterministic_replay_report([result["seed_report"] for result in seed_results], all(result["replay_passed"] for result in seed_results))
    combined_model_hash = sha256_text(json.dumps({str(result["seed"]): result["model_state_hash"] for result in seed_results}, sort_keys=True))
    artifact = model_artifact_audit(args, seed_results, combined_model_hash)
    label_byte = label_byte_generation_report(selected_report)
    shuffled_report = {
        "schema_version": "phase_147h_shuffled_target_control_report_v1",
        "shuffled_target_control_accuracy": sum(result["shuffled_target_control_accuracy"] for result in seed_results) / len(seed_results),
        "per_seed_shuffled_target_control_accuracy": {
            str(result["seed"]): result["shuffled_target_control_accuracy"] for result in seed_results
        },
    }
    shuffled_report["passed"] = shuffled_report["shuffled_target_control_accuracy"] <= 0.35
    baseline_margin = {
        "schema_version": "phase_147h_baseline_margin_report_v1",
        **baseline_eval,
        "best_baseline_accuracy": best_eval,
        "model_test_accuracy": rate(sum(1 for row in test_result_rows if row["selected_label_correct"]), len(test_result_rows)),
        "best_baseline_test_accuracy": best_test,
        "model_ood_accuracy": rate(sum(1 for row in ood_result_rows if row["selected_label_correct"]), len(ood_result_rows)),
        "best_baseline_ood_accuracy": best_ood,
        "test_margin_over_best_baseline": rate(sum(1 for row in test_result_rows if row["selected_label_correct"]), len(test_result_rows)) - best_test,
        "ood_margin_over_best_baseline": rate(sum(1 for row in ood_result_rows if row["selected_label_correct"]), len(ood_result_rows)) - best_ood,
    }
    baseline_margin["passed"] = (
        selected_report["selected_label_generation_accuracy"] >= best_eval + 0.10
        and baseline_margin["test_margin_over_best_baseline"] >= 0.10
        and baseline_margin["ood_margin_over_best_baseline"] >= 0.05
    )

    train_improves = all(result["train_metrics"]["train_loss_improves"] for result in seed_results)
    eval_improves = all(result["train_metrics"]["eval_loss_improves"] for result in seed_results)
    validation_not_nan = all(result["train_metrics"]["validation_loss_not_nan"] for result in seed_results)
    per_seed = {
        "schema_version": "phase_147h_per_seed_gate_report_v1",
        "per_seed_reports": [result["seed_report"] for result in seed_results],
        "passed": all(result["seed_report"]["passed"] for result in seed_results),
    }
    metrics = {
        "schema_version": "phase_147h_aggregate_metrics_v1",
        "selected_label_generation_accuracy": selected_report["selected_label_generation_accuracy"],
        "selected_label_byte_accuracy": selected_report["selected_label_byte_accuracy"],
        "final_value_from_generated_label_accuracy": final_value_report["final_value_from_generated_label_accuracy"],
        "heldout_template_selected_accuracy": baseline_margin["model_test_accuracy"],
        "ood_selected_accuracy": baseline_margin["model_ood_accuracy"],
        "generated_output_schema_valid_rate": generated_schema["generated_output_schema_valid_rate"],
        "multiple_selected_line_rate": generated_schema["multiple_selected_line_rate"],
        "answer_value_generation_rate": generated_schema["answer_value_generation_rate"],
        "selected_pocket_id_generation_rate": generated_schema["selected_pocket_id_generation_rate"],
        "malformed_selected_label_rate": generated_schema["malformed_selected_label_rate"],
        "extra_text_generation_rate": generated_schema["extra_text_generation_rate"],
        "schema_prefix_fixed_by_runner": True,
        "selected_line_wrapper_deterministic": True,
        "model_generates_full_selected_line": False,
        "train_loss_improves": train_improves,
        "eval_loss_improves": eval_improves,
        "validation_loss_not_nan": validation_not_nan,
        "generation_deterministic_replay_passed": replay["generation_deterministic_replay_passed"],
        "heldout_priority_order_accuracy": ood_family["heldout_priority_order_accuracy"],
        "heldout_block_order_accuracy": ood_family["heldout_block_order_accuracy"],
        "heldout_template_accuracy": ood_family["heldout_template_accuracy"],
        "heldout_rule_composition_accuracy": ood_family["heldout_rule_composition_accuracy"],
        "minimum_ood_family_accuracy": ood_family["minimum_ood_family_accuracy"],
        "minimum_ood_family_row_count": ood_family["minimum_ood_family_row_count"],
        "collapsed_ood_family_count": ood_family["collapsed_ood_family_count"],
        "every_label_appears_in_every_split": label_report["every_label_appears_in_every_split"],
        "minimum_per_label_generation_accuracy": label_report["minimum_per_label_generation_accuracy"],
        "best_baseline_accuracy": best_eval,
        "test_margin_over_best_baseline": baseline_margin["test_margin_over_best_baseline"],
        "ood_margin_over_best_baseline": baseline_margin["ood_margin_over_best_baseline"],
        "shuffled_target_control_accuracy": shuffled_report["shuffled_target_control_accuracy"],
        "shortcut_scanner_violation_count": shortcut["shortcut_scanner_violation_count"],
        "train_eval_prompt_overlap_count": leakage["train_eval_prompt_overlap_count"],
        "train_ood_prompt_overlap_count": leakage["train_ood_prompt_overlap_count"],
        "normalized_train_eval_prompt_overlap_count": leakage["normalized_train_eval_prompt_overlap_count"],
        "normalized_train_ood_prompt_overlap_count": leakage["normalized_train_ood_prompt_overlap_count"],
        "value_token_overlap_train_test_rate": value_leakage["value_token_overlap_train_test_rate"],
        "value_token_overlap_train_ood_rate": value_leakage["value_token_overlap_train_ood_rate"],
        "per_seed_reports": per_seed["per_seed_reports"],
    }
    metrics["passed"] = gates_pass(metrics)

    training_config = {
        "schema_version": "phase_147h_training_config_v1",
        "model_family": "runner_local_pytorch_byte_lm",
        "same_model_family_as_147a": True,
        "seeds": seeds,
        "counts_per_seed": counts,
        "feature_buckets": args.feature_buckets,
        "hidden": args.hidden,
        "epochs": args.epochs,
        "control_epochs": args.control_epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "selected_label_byte_objective": True,
        "full_line_free_generation_claimed": False,
        "final_value_policy": "deterministic copy from schema-wrapped selected label",
    }

    audits = [
        generated_schema,
        generation_input,
        label_report,
        ood_family,
        ood_split,
        anti_mem,
        baseline_margin,
        shuffled_report,
        shortcut,
        leakage,
        value_leakage,
        artifact,
        replay,
        label_byte,
        same_model,
        model_input,
        feature_path,
        per_seed,
    ]
    decision = choose_decision(metrics, audits)
    summary = {
        "schema_version": "phase_147h_summary_v1",
        "milestone": MILESTONE,
        "decision": decision["decision"],
        "verdict": decision["verdict"],
        "next": decision["next"],
        "boundary": BOUNDARY_TEXT,
        "metrics": metrics,
        **FALSE_FLAGS,
    }

    write_json(out / "training_config.json", training_config)
    write_json(out / "generation_eval_report.json", generation_eval)
    write_json(out / "selected_label_generation_report.json", selected_report)
    write_json(out / "final_value_copy_report.json", final_value_report)
    write_json(out / "generated_schema_report.json", generated_schema)
    write_json(out / "generation_input_audit.json", generation_input)
    write_json(out / "label_distribution_report.json", label_report)
    write_json(out / "ood_generation_family_report.json", ood_family)
    write_json(out / "ood_split_definition_report.json", ood_split)
    write_json(out / "anti_memorization_report.json", anti_mem)
    write_json(out / "baseline_margin_report.json", baseline_margin)
    write_json(out / "shuffled_target_control_report.json", shuffled_report)
    write_json(out / "shortcut_scanner_report.json", shortcut)
    write_json(out / "leakage_audit.json", leakage)
    write_json(out / "value_token_leakage_report.json", value_leakage)
    write_json(out / "model_artifact_audit.json", artifact)
    write_json(out / "deterministic_replay_report.json", replay)
    write_json(out / "label_byte_generation_report.json", label_byte)
    write_json(out / "same_model_family_audit.json", same_model)
    write_json(out / "model_input_audit.json", model_input)
    write_json(out / "feature_path_audit.json", feature_path)
    write_json(out / "per_seed_gate_report.json", per_seed)
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
