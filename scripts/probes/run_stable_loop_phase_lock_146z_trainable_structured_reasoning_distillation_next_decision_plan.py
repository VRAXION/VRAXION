#!/usr/bin/env python3
"""146Z planning-only next decision after trainable distillation scale confirm."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "STABLE_LOOP_PHASE_LOCK_146Z_TRAINABLE_STRUCTURED_REASONING_DISTILLATION_NEXT_DECISION_PLAN"
DEFAULT_OUT = Path("target/pilot_wave/stable_loop_phase_lock_146z_trainable_structured_reasoning_distillation_next_decision_plan/smoke")
DEFAULT_146H_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_146h_trainable_structured_reasoning_distillation_bridge_scale_confirm/smoke")
RUNNER_PATH = Path(__file__).resolve()
CHECKER_PATH = REPO_ROOT / "scripts/probes/run_stable_loop_phase_lock_146z_trainable_structured_reasoning_distillation_next_decision_plan_check.py"
HELPER_PATH = REPO_ROOT / "scripts/probes/shared_raw_generation_helper.py"
DECISION = "lm_style_canonical_structured_text_distillation_prototype_plan_recommended"
SELECTED_OPTION = "lm_style_canonical_structured_text_distillation_prototype"
NEXT = "147A_LM_STYLE_CANONICAL_STRUCTURED_TEXT_DISTILLATION_PROTOTYPE"
BOUNDARY_TEXT = (
    "146Z is constrained model-facing distillation evidence only with canonical structured prompts only; "
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
REQUIRED_147A_AUDITS = [
    "feature_path_audit.json",
    "model_artifact_audit.json",
    "model_input_audit.json",
    "ood_split_definition_report.json",
    "generated_schema_report.json",
    "anti_memorization_report.json",
    "baseline_margin_report.json",
    "shortcut_scanner_report.json",
    "leakage_audit.json",
]
VALID_SELECTED_LINES = ["SELECTED=A", "SELECTED=B", "SELECTED=C", "SELECTED=fallback"]
INVALID_SELECTED_OUTPUTS = [
    "SELECTED=pocket_a",
    "ANSWER=...",
    "selected_pocket_id=...",
    "winner=pocket_*",
    "extra text before/after the SELECTED line",
    "malformed labels",
    "multiple SELECTED lines",
]
NEGATIVE_ROUTES_147A = {
    "lm_training_failure": "147B_LM_TRAINING_FAILURE_ANALYSIS",
    "generated_schema_failure": "147C_GENERATED_SCHEMA_FAILURE_ANALYSIS",
    "selected_label_generation_failure": "147D_SELECTED_LABEL_GENERATION_FAILURE_ANALYSIS",
    "model_shortcut_detected": "147E_LM_SHORTCUT_ANALYSIS",
    "ood_generation_failure": "147F_LM_OOD_GENERALIZATION_ANALYSIS",
    "ood_split_not_actually_heldout": "147G_LM_OOD_SPLIT_LEAKAGE_ANALYSIS",
    "distillation_regression": "146D_MODEL_SHORTCUT_ANALYSIS",
}


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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def git_show_head(path: str) -> str:
    result = subprocess.run(["git", "show", f"HEAD:{path}"], cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    return result.stdout if result.returncode == 0 else ""


def helper_unchanged_from_head() -> bool:
    return HELPER_PATH.read_text(encoding="utf-8") == git_show_head("scripts/probes/shared_raw_generation_helper.py")


def scan_ast() -> dict[str, Any]:
    failures: list[str] = []
    for path in [RUNNER_PATH, CHECKER_PATH]:
        if not path.exists():
            failures.append(f"missing:{rel(path)}")
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module == "shared_raw_generation_helper":
                    failures.append(f"forbidden_import:{rel(path)}:{module}")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in {"torch", "tensorflow", "shared_raw_generation_helper", "requests", "socket", "urllib", "http.client"}:
                        failures.append(f"forbidden_import:{rel(path)}:{alias.name}")
            if isinstance(node, ast.Call):
                name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ""
                if name in {"train", "fit", "forward", "backward", "step", "raw_generate", "load_checkpoint", "save_checkpoint"}:
                    failures.append(f"forbidden_call:{rel(path)}:{name}")
    return {"schema_version": "phase_146z_ast_scan_v1", "passed": not failures, "failures": failures}


def require_146h(root: Path) -> tuple[dict[str, Any], list[str]]:
    required = [
        "decision.json",
        "aggregate_metrics.json",
        "summary.json",
        "model_feature_audit.json",
        "feature_path_audit.json",
        "same_model_family_audit.json",
        "model_artifact_audit.json",
        "per_family_ood_report.json",
        "baseline_margin_report.json",
        "shortcut_scanner_report.json",
        "value_token_leakage_report.json",
    ]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        return {"root": rel(root), "missing_artifacts": missing}, [f"missing:{name}" for name in missing]
    decision = read_json(root / "decision.json")
    metrics = read_json(root / "aggregate_metrics.json")
    summary = read_json(root / "summary.json")
    model_feature = read_json(root / "model_feature_audit.json")
    feature_path = read_json(root / "feature_path_audit.json")
    same_model = read_json(root / "same_model_family_audit.json")
    model_artifact = read_json(root / "model_artifact_audit.json")
    per_family = read_json(root / "per_family_ood_report.json")
    baseline_margin = read_json(root / "baseline_margin_report.json")
    shortcut = read_json(root / "shortcut_scanner_report.json")
    value = read_json(root / "value_token_leakage_report.json")
    checks = {
        "decision": decision.get("decision") == "trainable_structured_reasoning_distillation_bridge_scale_confirmed",
        "verdict": decision.get("verdict") == "INSTNCT_TRAINABLE_STRUCTURED_REASONING_DISTILLATION_BRIDGE_SCALE_CONFIRMED",
        "next": decision.get("next") == "146Z_TRAINABLE_STRUCTURED_REASONING_DISTILLATION_NEXT_DECISION_PLAN",
        "selected_pocket_prediction_accuracy": metrics.get("selected_pocket_prediction_accuracy") == 1.0,
        "final_value_from_predicted_pocket_accuracy": metrics.get("final_value_from_predicted_pocket_accuracy") == 1.0,
        "heldout_template_accuracy": metrics.get("heldout_template_accuracy") == 1.0,
        "ood_composition_accuracy": metrics.get("ood_composition_accuracy") == 1.0,
        "minimum_ood_family_accuracy": metrics.get("minimum_ood_family_accuracy") == 1.0,
        "margin_over_best_baseline": metrics.get("margin_over_best_baseline", 0.0) >= 0.58,
        "shortcut_scanner_violation_count": metrics.get("shortcut_scanner_violation_count") == 0,
        "value_token_overlap_train_test_rate": metrics.get("value_token_overlap_train_test_rate") == 0.0,
        "value_token_overlap_train_ood_rate": metrics.get("value_token_overlap_train_ood_rate") == 0.0,
        "deterministic_replay_passed": metrics.get("deterministic_replay_passed") is True,
        "model_feature_audit_passed": model_feature.get("passed") is True,
        "feature_path_audit_passed": feature_path.get("passed") is True,
        "same_model_family_audit_passed": same_model.get("passed") is True,
        "model_artifact_audit_passed": model_artifact.get("passed") is True,
        "per_family_ood_passed": per_family.get("passed") is True and per_family.get("collapsed_ood_family_count") == 0,
        "baseline_margin_passed": baseline_margin.get("passed") is True,
        "shortcut_report_passed": shortcut.get("passed") is True,
        "value_leakage_passed": value.get("passed") is True,
    }
    failures = [key for key, passed in checks.items() if not passed]
    return {
        "schema_version": "phase_146z_upstream_146h_manifest_v1",
        "root": rel(root),
        "decision": decision,
        "aggregate_metrics": metrics,
        "summary": summary,
        "model_feature_audit": model_feature,
        "feature_path_audit": feature_path,
        "same_model_family_audit": same_model,
        "model_artifact_audit": model_artifact,
        "per_family_ood_report": per_family,
        "baseline_margin_report": baseline_margin,
        "shortcut_scanner_report": shortcut,
        "value_token_leakage_report": value,
        "checks": checks,
        "failed_checks": failures,
        "passed": not failures,
    }, failures


def build_evidence_chain_summary(upstream: dict[str, Any]) -> dict[str, Any]:
    metrics = upstream["aggregate_metrics"]
    return {
        "schema_version": "phase_146z_evidence_chain_summary_v1",
        "145h_mixed_structured_rule_helper_scale_confirmed": True,
        "146a_trainable_distillation_prototype_positive": True,
        "146h_trainable_distillation_scale_confirmed": True,
        "146h_selected_pocket_prediction_accuracy": metrics["selected_pocket_prediction_accuracy"],
        "146h_final_value_from_predicted_pocket_accuracy": metrics["final_value_from_predicted_pocket_accuracy"],
        "146h_ood_composition_accuracy": metrics["ood_composition_accuracy"],
        "146h_margin_over_best_baseline": metrics["margin_over_best_baseline"],
        "remaining_gap": "LM-style sequence generation of selected labels from canonical structured text is untested",
        "passed": upstream["passed"],
    }


def build_trainable_distillation_state_report(upstream: dict[str, Any]) -> dict[str, Any]:
    metrics = upstream["aggregate_metrics"]
    return {
        "schema_version": "phase_146z_trainable_distillation_state_report_v1",
        "raw_text_perceptron_bridge_scale_confirmed": True,
        "same_146a_model_family_confirmed": True,
        "feature_path_audit_passed": upstream["feature_path_audit"]["passed"],
        "model_artifact_audit_passed": upstream["model_artifact_audit"]["passed"],
        "selected_pocket_prediction_accuracy": metrics["selected_pocket_prediction_accuracy"],
        "ood_composition_accuracy": metrics["ood_composition_accuracy"],
        "minimum_ood_family_accuracy": metrics["minimum_ood_family_accuracy"],
        "shortcut_scanner_violation_count": metrics["shortcut_scanner_violation_count"],
        "value_token_overlap_train_test_rate": metrics["value_token_overlap_train_test_rate"],
        "next_gap": "runner-local LM-style selected-label generation from canonical structured text",
        "passed": True,
    }


def build_model_architecture_gap_analysis() -> dict[str, Any]:
    return {
        "schema_version": "phase_146z_model_architecture_gap_analysis_v1",
        "raw_text_classifier_distillation_scale_confirmed": True,
        "lm_style_sequence_generation_untested": True,
        "natural_language_rule_reasoning_untested": True,
        "open_ended_arbitration_claimed": False,
        "gpt_like_or_gemma_like_capability_claimed": False,
        "recommended_architecture_step": "runner-local PyTorch byte-level causal next-byte model on canonical structured prompt/output pairs",
        "why_not_more_perceptron_scale": "146H already scale-confirmed the raw-text perceptron bridge under OOD, leakage, and baseline audits.",
        "why_not_natural_language_yet": "selected-label sequence generation should be tested before free-form natural-language inputs.",
        "passed": True,
    }


def build_next_decision_matrix() -> dict[str, Any]:
    options = [
        {
            "option": "lm_style_canonical_structured_text_distillation_prototype",
            "recommended": True,
            "score": 1.0,
            "rationale": "Tests the next model-facing bridge: sequence generation of SELECTED labels from canonical structured text.",
        },
        {
            "option": "scale_raw_text_perceptron_curriculum_further",
            "recommended": False,
            "score": 0.56,
            "rationale": "Would add confidence to an already scale-confirmed bridge but not test LM-style generation.",
        },
        {
            "option": "natural_language_wrapper_before_sequence_model",
            "recommended": False,
            "score": 0.38,
            "rationale": "Introduces natural-language ambiguity before selected-label sequence generation is proven.",
        },
        {
            "option": "helper_engine_extension_after_distillation",
            "recommended": False,
            "score": 0.34,
            "rationale": "Returns to deterministic helper feature growth instead of moving toward model-facing sequence behavior.",
        },
        {
            "option": "stop_at_raw_text_distillation_bridge",
            "recommended": False,
            "score": 0.20,
            "rationale": "Avoids overclaim but leaves the LM-style bridge untested.",
        },
    ]
    return {
        "schema_version": "phase_146z_next_decision_matrix_v1",
        "options": options,
        "selected_option": SELECTED_OPTION,
        "decision": DECISION,
        "passed": True,
    }


def build_target_147a_milestone_plan() -> dict[str, Any]:
    return {
        "schema_version": "phase_146z_target_147a_milestone_plan_v1",
        "milestone": NEXT,
        "implementation_ready": True,
        "planning_only": False,
        "intended_primitive": [
            "confirmed structured teacher curriculum",
            "canonical structured prompt/output text pairs",
            "runner-local PyTorch byte-level causal next-byte model",
            "generated SELECTED=<label>",
            "deterministic candidate-value copy from generated selected label",
            "ANSWER=<value>",
        ],
        "model_policy": {
            "runner_local_pytorch_only": True,
            "byte_level_causal_next_byte_model": True,
            "external_api_used": False,
            "external_model_download_used": False,
            "shared_helper_modification_allowed": False,
            "natural_language_input_allowed": False,
            "deterministic_cpu_settings_required": True,
            "heartbeat_progress_required": True,
        },
        "valid_generated_schema": VALID_SELECTED_LINES,
        "invalid_generated_outputs": INVALID_SELECTED_OUTPUTS,
        "primary_target": "SELECTED=<A|B|C|fallback>",
        "final_value_policy": "copy candidate value from input using generated selected label",
        "opaque_value_token_generation_required": False,
        "required_audits": REQUIRED_147A_AUDITS,
        "feature_path_audit_requirements": {
            "feature_or_model_input_source": "raw canonical sequence text only",
            "teacher_trace_fields_allowed_in_teacher_or_scoring_artifacts": True,
            "teacher_trace_fields_forbidden_in_model_input": True,
            "selected_pocket_id_forbidden_in_model_input": True,
            "expected_gold_target_answer_labels_forbidden_in_model_input": True,
        },
        "ood_split_definition_requirements": {
            "ood_templates_held_out_from_train": True,
            "ood_priority_orders_held_out_from_train_where_applicable": True,
            "ood_block_order_patterns_held_out_from_train_where_applicable": True,
            "ood_rule_block_combinations_held_out_from_train_where_applicable": True,
            "train_ood_template_overlap_count": 0,
            "ood_family_row_counts_present": True,
        },
        "model_artifact_audit_requirements": {
            "model_family": "runner_local_pytorch_byte_lm",
            "new_external_model_used": False,
            "external_model_or_api_used": False,
            "model_download_used": False,
            "model_artifacts_written_only_under_target": True,
            "deterministic_seed_used": True,
        },
        "positive_gates": {
            "selected_label_generation_accuracy": ">= 0.70",
            "final_value_from_generated_label_accuracy": ">= 0.70",
            "heldout_template_selected_accuracy": ">= 0.60",
            "ood_selected_accuracy": ">= 0.50",
            "generated_output_schema_valid_rate": ">= 0.80",
            "multiple_selected_line_rate": "= 0.0",
            "answer_value_generation_rate": "= 0.0",
            "malformed_selected_label_rate": "<= 0.20",
            "eval_loss_improves": "= true",
            "train_loss_improves": "= true",
            "validation_loss_not_nan": "= true",
            "selected_label_generation_accuracy_vs_best_baseline": ">= +0.10",
            "test_margin_over_best_baseline": ">= 0.10",
            "ood_margin_over_best_baseline": ">= 0.05",
            "shuffled_target_control_accuracy": "<= 0.35",
            "shortcut_scanner_violation_count": "= 0",
            "train_eval_prompt_overlap_count": "= 0",
            "train_ood_prompt_overlap_count": "= 0",
            "value_token_overlap_train_test_rate": "= 0.0",
            "generation_deterministic_replay_passed": "= true",
        },
        "ood_controls": {
            "heldout_priority_order_accuracy": ">= 0.50",
            "heldout_block_order_accuracy": ">= 0.50",
            "heldout_template_accuracy": ">= 0.60",
            "heldout_rule_composition_accuracy": ">= 0.50",
        },
        "expected_positive_route": {
            "decision": "lm_style_canonical_structured_text_distillation_prototype_positive",
            "verdict": "INSTNCT_LM_STYLE_CANONICAL_STRUCTURED_TEXT_DISTILLATION_PROTOTYPE_POSITIVE",
            "next": "147H_LM_STYLE_CANONICAL_STRUCTURED_TEXT_DISTILLATION_SCALE_CONFIRM",
        },
        "clean_negative_routes": NEGATIVE_ROUTES_147A,
        "claim_limit": "A positive 147A proves only runner-local byte-level LM-style selected-label generation on canonical structured prompts.",
        "not_claimed": [
            "natural-language rule reasoning",
            "open-ended arbitration",
            "GPT-like/Gemma-like assistant capability",
            "production readiness",
            "architecture superiority",
        ],
    }


def build_anti_oracle_requirements() -> dict[str, Any]:
    return {
        "schema_version": "phase_146z_anti_oracle_requirements_v1",
        "model_facing_inputs_must_forbid": [
            "teacher trace fields",
            "selected_pocket_id",
            "winner=pocket_*",
            "ANSWER=<value>",
            "gold value",
            "target value",
            "resolved output",
            "expected output",
            "per-row oracle metadata",
        ],
        "generated_output_must_not_include": [
            "ANSWER=<value>",
            "selected_pocket_id",
            "winner=pocket_*",
            "gold/target/resolved shortcuts",
        ],
        "strict_selected_schema_required": VALID_SELECTED_LINES,
        "anti_memorization_report_required": True,
        "ood_split_definition_report_required": True,
        "shortcut_scanner_required": True,
        "leakage_audit_required": True,
        "passed": True,
    }


def build_risk_register() -> dict[str, Any]:
    return {
        "schema_version": "phase_146z_risk_register_v1",
        "risks": [
            {
                "risk": "LM-style output schema drift",
                "mitigation": "strict generated_schema_report with exact SELECTED line validation",
            },
            {
                "risk": "prompt or template memorization",
                "mitigation": "anti_memorization_report plus heldout template, priority order, block order, and composition controls",
            },
            {
                "risk": "hidden teacher/oracle fields in model input",
                "mitigation": "model_input_audit and shortcut scanner forbid teacher trace, selected_pocket_id, answer/gold/target fields",
            },
            {
                "risk": "opaque value generation confused with selected-label reasoning",
                "mitigation": "147A learns SELECTED labels only; final value is deterministic copy from the input",
            },
            {
                "risk": "overclaiming 147A as natural-language or Gemma-like capability",
                "mitigation": "canonical structured prompts only and broad capability flags remain false",
            },
        ],
        "passed": True,
    }


def build_decision() -> dict[str, Any]:
    return {
        "schema_version": "phase_146z_decision_v1",
        "decision": DECISION,
        "selected_option": SELECTED_OPTION,
        "next": NEXT,
        "positive_gate_passed": True,
        "boundary": BOUNDARY_TEXT,
        **FALSE_FLAGS,
    }


def write_report(out: Path, decision: dict[str, Any]) -> None:
    text = f"""# {MILESTONE}

Decision: `{decision['decision']}`

Selected option: `{decision['selected_option']}`

Next: `{decision['next']}`

Boundary: {BOUNDARY_TEXT}

## Summary

146H scale-confirmed the raw-text trainable distillation bridge. The selected next direction is a runner-local LM-style canonical structured text distillation prototype, not another helper extension and not natural-language input.

The target 147A milestone is implementation-ready: canonical structured prompt/output pairs train a local byte-level causal next-byte model to generate exactly one `SELECTED=<label>` line. Final value scoring copies the candidate value from the input after selected-label generation; direct generation of unseen opaque value tokens is out of scope.

## Guardrails

147A carries forward the 146H hardening: model artifact audit, feature/input audit, OOD split definition, generated schema report, anti-memorization report, baseline margins, shortcut scanner, and leakage audit.

## Claim Limit

146Z is planning-only. A future positive 147A would prove only runner-local byte-level LM-style selected-label generation on canonical structured prompts. It would not prove natural-language rule reasoning, open-ended arbitration, GPT-like/Gemma-like assistant capability, production readiness, or architecture superiority.
"""
    write_text(out / "report.md", text)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run 146Z trainable distillation next decision plan")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--upstream-146h-root", type=Path, default=DEFAULT_146H_ROOT)
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    args = parser.parse_args()

    out = resolve_target_out(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_text(out / "progress.jsonl", "")
    append_progress(out, "startup", milestone=MILESTONE, heartbeat_sec=args.heartbeat_sec)
    write_json(out / "queue.json", {"schema_version": "phase_146z_queue_v1", "milestone": MILESTONE, "status": "running"})

    upstream, upstream_failures = require_146h(resolve_repo_path(args.upstream_146h_root))
    if upstream_failures:
        raise RuntimeError(f"146H upstream verification failed: {upstream_failures}")
    write_json(out / "upstream_146h_manifest.json", upstream)
    append_progress(out, "upstream_verified", upstream_decision=upstream["decision"]["decision"])

    static = scan_ast()
    if not static["passed"]:
        raise RuntimeError(f"146Z static scan failed: {static['failures']}")
    if not helper_unchanged_from_head():
        raise RuntimeError("shared_raw_generation_helper.py changed from HEAD")
    append_progress(out, "static_integrity_verified", helper_unchanged=True)

    decision = build_decision()
    evidence = build_evidence_chain_summary(upstream)
    state = build_trainable_distillation_state_report(upstream)
    gap = build_model_architecture_gap_analysis()
    matrix = build_next_decision_matrix()
    target = build_target_147a_milestone_plan()
    anti = build_anti_oracle_requirements()
    risk = build_risk_register()
    config = {
        "schema_version": "phase_146z_analysis_config_v1",
        "milestone": MILESTONE,
        "boundary": BOUNDARY_TEXT,
        "planning_only": True,
        "artifact_only": True,
        "raw_generate_allowed": False,
        "shared_helper_import_allowed": False,
        "helper_modification_allowed": False,
        "training_allowed": False,
        "torch_forward_pass_allowed": False,
        "checkpoint_mutation_allowed": False,
        "selected_option": SELECTED_OPTION,
        "next": NEXT,
        "runner_sha256": sha256_text(RUNNER_PATH.read_text(encoding="utf-8")),
        **FALSE_FLAGS,
    }
    summary = {
        "schema_version": "phase_146z_summary_v1",
        "milestone": MILESTONE,
        "status": "complete",
        "boundary": BOUNDARY_TEXT,
        "decision": decision,
        "selected_option": SELECTED_OPTION,
        "next": NEXT,
        "upstream_146h_verified": True,
        "target_147a_implementation_ready": True,
        **FALSE_FLAGS,
    }

    write_json(out / "analysis_config.json", config)
    write_json(out / "evidence_chain_summary.json", evidence)
    write_json(out / "trainable_distillation_state_report.json", state)
    write_json(out / "model_architecture_gap_analysis.json", gap)
    write_json(out / "next_decision_matrix.json", matrix)
    write_json(out / "target_147a_milestone_plan.json", target)
    write_json(out / "anti_oracle_requirements.json", anti)
    write_json(out / "risk_register.json", risk)
    write_json(out / "decision.json", decision)
    write_json(out / "summary.json", summary)
    write_report(out, decision)
    append_progress(out, "decision", decision=decision["decision"], selected_option=decision["selected_option"], next=decision["next"])
    write_json(out / "queue.json", {"schema_version": "phase_146z_queue_v1", "milestone": MILESTONE, "status": "complete", "decision": decision["decision"], "next": decision["next"]})
    print(json.dumps({"decision": decision["decision"], "selected_option": decision["selected_option"], "next": decision["next"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
