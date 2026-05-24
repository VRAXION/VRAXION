#!/usr/bin/env python3
"""Checker for 146Z trainable distillation next-decision planning milestone."""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_146z_trainable_structured_reasoning_distillation_next_decision_plan/smoke"
HELPER = "scripts/probes/shared_raw_generation_helper.py"
RUNNER = "scripts/probes/run_stable_loop_phase_lock_146z_trainable_structured_reasoning_distillation_next_decision_plan.py"
CHECKER = "scripts/probes/run_stable_loop_phase_lock_146z_trainable_structured_reasoning_distillation_next_decision_plan_check.py"
DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_146Z_TRAINABLE_STRUCTURED_REASONING_DISTILLATION_NEXT_DECISION_PLAN_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_146Z_TRAINABLE_STRUCTURED_REASONING_DISTILLATION_NEXT_DECISION_PLAN_RESULT.md",
]
ALLOWED_MUTATIONS = {RUNNER, CHECKER, *DOCS}
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "analysis_config.json",
    "upstream_146h_manifest.json",
    "evidence_chain_summary.json",
    "trainable_distillation_state_report.json",
    "model_architecture_gap_analysis.json",
    "next_decision_matrix.json",
    "target_147a_milestone_plan.json",
    "anti_oracle_requirements.json",
    "risk_register.json",
    "decision.json",
    "summary.json",
    "report.md",
]
FALSE_FLAGS = [
    "reasoning_restored",
    "raw_assistant_capability_restored",
    "structured_tool_capability_restored",
    "rule_metadata_reasoning_claimed",
    "natural_language_rule_reasoning_claimed",
    "open_ended_arbitration_claimed",
    "gpt_like_readiness_claimed",
    "gemma_like_capability_claimed",
    "open_domain_assistant_readiness_claimed",
    "production_chat_claimed",
    "public_api_claimed",
    "deployment_readiness_claimed",
    "safety_alignment_claimed",
    "architecture_superiority_claimed",
]
BOUNDARY_PHRASES = [
    "constrained model-facing distillation evidence only",
    "canonical structured prompts only",
    "not natural-language rule reasoning",
    "not open-ended arbitration",
    "not GPT-like/Gemma-like assistant capability",
    "not production readiness",
    "not architecture superiority",
]
EXPECTED_DECISION = "lm_style_canonical_structured_text_distillation_prototype_plan_recommended"
EXPECTED_OPTION = "lm_style_canonical_structured_text_distillation_prototype"
EXPECTED_NEXT = "147A_LM_STYLE_CANONICAL_STRUCTURED_TEXT_DISTILLATION_PROTOTYPE"
REQUIRED_147A_AUDITS = {
    "feature_path_audit.json",
    "model_artifact_audit.json",
    "model_input_audit.json",
    "ood_split_definition_report.json",
    "generated_schema_report.json",
    "anti_memorization_report.json",
    "baseline_margin_report.json",
    "shortcut_scanner_report.json",
    "leakage_audit.json",
}
VALID_SELECTED_LINES = {"SELECTED=A", "SELECTED=B", "SELECTED=C", "SELECTED=fallback"}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def git_status() -> str:
    result = subprocess.run(["git", "status", "--short"], cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    return result.stdout.rstrip("\n")


def changed_paths() -> list[str]:
    return [line[3:].replace("\\", "/") for line in git_status().splitlines() if line.strip()]


def git_show_head(path: str) -> str:
    result = subprocess.run(["git", "show", f"HEAD:{path}"], cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    return result.stdout if result.returncode == 0 else ""


def require_changed_files(failures: list[str]) -> None:
    for path in changed_paths():
        if path.startswith("target/"):
            continue
        if path not in ALLOWED_MUTATIONS:
            failures.append(f"UNEXPECTED_CHANGED_FILE:{path}")


def require_false_flags(payload: dict[str, Any], failures: list[str], prefix: str) -> None:
    for key in FALSE_FLAGS:
        if payload.get(key) is not False:
            failures.append(f"{prefix}_BOUNDARY_FLAG_NOT_FALSE:{key}:{payload.get(key)}")


def scan_python(path: Path) -> list[str]:
    failures: list[str] = []
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "shared_raw_generation_helper":
                failures.append(f"FORBIDDEN_IMPORT:{path.name}:{module}")
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in {"torch", "tensorflow", "shared_raw_generation_helper", "requests", "socket", "urllib", "http.client"}:
                    failures.append(f"FORBIDDEN_IMPORT:{path.name}:{alias.name}")
        if isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ""
            if name in {"train", "fit", "forward", "backward", "step", "raw_generate", "load_checkpoint", "save_checkpoint"}:
                failures.append(f"FORBIDDEN_CALL:{path.name}:{name}")
    return failures


def require_static_files(failures: list[str]) -> None:
    for rel_path in [RUNNER, CHECKER, *DOCS]:
        path = REPO_ROOT / rel_path
        if not path.exists():
            failures.append(f"MISSING_TRACKED_FILE:{rel_path}")
            continue
        text = path.read_text(encoding="utf-8")
        for term in ["146Z", "planning-only", "147A", EXPECTED_OPTION]:
            if term not in text:
                failures.append(f"TERM_MISSING:{rel_path}:{term}")
        for phrase in BOUNDARY_PHRASES:
            if phrase not in text:
                failures.append(f"BOUNDARY_TERM_MISSING:{rel_path}:{phrase}")
        if path.suffix == ".py":
            failures.extend(scan_python(path))
    helper_source = (REPO_ROOT / HELPER).read_text(encoding="utf-8")
    if helper_source != git_show_head(HELPER):
        failures.append("SHARED_HELPER_CHANGED_FROM_HEAD")


def require_artifacts(root: Path, failures: list[str]) -> None:
    for name in REQUIRED_ARTIFACTS:
        if not (root / name).exists():
            failures.append(f"MISSING_ARTIFACT:{name}")
    progress = root / "progress.jsonl"
    if progress.exists() and progress.stat().st_size == 0:
        failures.append("PROGRESS_JSONL_EMPTY")


def require_upstream(root: Path, failures: list[str]) -> None:
    upstream = load_json(root / "upstream_146h_manifest.json")
    decision = upstream.get("decision", {})
    metrics = upstream.get("aggregate_metrics", {})
    if decision.get("decision") != "trainable_structured_reasoning_distillation_bridge_scale_confirmed":
        failures.append(f"BAD_146H_DECISION:{decision.get('decision')}")
    if decision.get("verdict") != "INSTNCT_TRAINABLE_STRUCTURED_REASONING_DISTILLATION_BRIDGE_SCALE_CONFIRMED":
        failures.append(f"BAD_146H_VERDICT:{decision.get('verdict')}")
    if decision.get("next") != "146Z_TRAINABLE_STRUCTURED_REASONING_DISTILLATION_NEXT_DECISION_PLAN":
        failures.append(f"BAD_146H_NEXT:{decision.get('next')}")
    exact = {
        "selected_pocket_prediction_accuracy": 1.0,
        "final_value_from_predicted_pocket_accuracy": 1.0,
        "heldout_template_accuracy": 1.0,
        "ood_composition_accuracy": 1.0,
        "minimum_ood_family_accuracy": 1.0,
        "shortcut_scanner_violation_count": 0,
        "value_token_overlap_train_test_rate": 0.0,
        "value_token_overlap_train_ood_rate": 0.0,
        "deterministic_replay_passed": True,
    }
    for key, expected in exact.items():
        if metrics.get(key) != expected:
            failures.append(f"UPSTREAM_146H_METRIC_MISMATCH:{key}:{metrics.get(key)}")
    if metrics.get("margin_over_best_baseline", 0.0) < 0.58:
        failures.append(f"UPSTREAM_146H_MARGIN_TOO_LOW:{metrics.get('margin_over_best_baseline')}")
    if upstream.get("failed_checks") != [] or upstream.get("passed") is not True:
        failures.append(f"UPSTREAM_146H_FAILED_CHECKS:{upstream.get('failed_checks')}")
    for key in [
        "model_feature_audit_passed",
        "feature_path_audit_passed",
        "same_model_family_audit_passed",
        "model_artifact_audit_passed",
        "per_family_ood_passed",
        "baseline_margin_passed",
        "shortcut_report_passed",
        "value_leakage_passed",
    ]:
        if upstream.get("checks", {}).get(key) is not True:
            failures.append(f"UPSTREAM_146H_CHECK_NOT_TRUE:{key}")


def require_decision(root: Path, failures: list[str]) -> None:
    decision = load_json(root / "decision.json")
    summary = load_json(root / "summary.json")
    config = load_json(root / "analysis_config.json")
    report = (root / "report.md").read_text(encoding="utf-8")
    if decision.get("decision") != EXPECTED_DECISION:
        failures.append(f"BAD_DECISION:{decision.get('decision')}")
    if decision.get("selected_option") != EXPECTED_OPTION:
        failures.append(f"BAD_SELECTED_OPTION:{decision.get('selected_option')}")
    if decision.get("next") != EXPECTED_NEXT:
        failures.append(f"BAD_NEXT:{decision.get('next')}")
    if config.get("planning_only") is not True or config.get("artifact_only") is not True:
        failures.append("CONFIG_NOT_PLANNING_ARTIFACT_ONLY")
    for key in ["raw_generate_allowed", "shared_helper_import_allowed", "helper_modification_allowed", "training_allowed", "torch_forward_pass_allowed", "checkpoint_mutation_allowed"]:
        if config.get(key) is not False:
            failures.append(f"CONFIG_FORBIDDEN_CAPABILITY_NOT_FALSE:{key}:{config.get(key)}")
    for payload_name, payload in [("decision", decision), ("summary", summary), ("config", config)]:
        require_false_flags(payload, failures, payload_name)
        text = json.dumps(payload)
        for phrase in BOUNDARY_PHRASES:
            if phrase not in text:
                failures.append(f"BOUNDARY_MISSING:{payload_name}:{phrase}")
    for phrase in BOUNDARY_PHRASES:
        if phrase not in report:
            failures.append(f"REPORT_BOUNDARY_MISSING:{phrase}")


def require_target_147a(root: Path, failures: list[str]) -> None:
    target = load_json(root / "target_147a_milestone_plan.json")
    if target.get("milestone") != EXPECTED_NEXT:
        failures.append(f"TARGET_BAD_MILESTONE:{target.get('milestone')}")
    if target.get("implementation_ready") is not True:
        failures.append("TARGET_NOT_IMPLEMENTATION_READY")
    model_policy = target.get("model_policy", {})
    required_policy = {
        "runner_local_pytorch_only": True,
        "byte_level_causal_next_byte_model": True,
        "external_api_used": False,
        "external_model_download_used": False,
        "shared_helper_modification_allowed": False,
        "natural_language_input_allowed": False,
        "deterministic_cpu_settings_required": True,
        "heartbeat_progress_required": True,
    }
    for key, expected in required_policy.items():
        if model_policy.get(key) != expected:
            failures.append(f"TARGET_MODEL_POLICY_BAD:{key}:{model_policy.get(key)}")
    if set(target.get("valid_generated_schema", [])) != VALID_SELECTED_LINES:
        failures.append(f"TARGET_BAD_VALID_SCHEMA:{target.get('valid_generated_schema')}")
    for invalid in ["SELECTED=pocket_a", "ANSWER=...", "selected_pocket_id=...", "winner=pocket_*", "multiple SELECTED lines"]:
        if invalid not in target.get("invalid_generated_outputs", []):
            failures.append(f"TARGET_INVALID_OUTPUT_MISSING:{invalid}")
    if set(target.get("required_audits", [])) != REQUIRED_147A_AUDITS:
        failures.append("TARGET_REQUIRED_AUDITS_MISMATCH")
    if target.get("opaque_value_token_generation_required") is not False:
        failures.append("TARGET_REQUIRES_OPAQUE_VALUE_GENERATION")
    feature = target.get("feature_path_audit_requirements", {})
    if feature.get("teacher_trace_fields_forbidden_in_model_input") is not True:
        failures.append("TARGET_TEACHER_TRACE_NOT_FORBIDDEN_IN_INPUT")
    if feature.get("selected_pocket_id_forbidden_in_model_input") is not True:
        failures.append("TARGET_SELECTED_POCKET_ID_NOT_FORBIDDEN")
    ood = target.get("ood_split_definition_requirements", {})
    if ood.get("train_ood_template_overlap_count") != 0:
        failures.append("TARGET_OOD_TEMPLATE_OVERLAP_NOT_ZERO")
    model_artifact = target.get("model_artifact_audit_requirements", {})
    if model_artifact.get("model_family") != "runner_local_pytorch_byte_lm":
        failures.append(f"TARGET_BAD_MODEL_FAMILY:{model_artifact.get('model_family')}")
    if model_artifact.get("external_model_or_api_used") is not False or model_artifact.get("model_download_used") is not False:
        failures.append("TARGET_EXTERNAL_MODEL_OR_DOWNLOAD_ALLOWED")
    gates = target.get("positive_gates", {})
    for key in [
        "selected_label_generation_accuracy",
        "final_value_from_generated_label_accuracy",
        "generated_output_schema_valid_rate",
        "multiple_selected_line_rate",
        "answer_value_generation_rate",
        "generation_deterministic_replay_passed",
    ]:
        if key not in gates:
            failures.append(f"TARGET_GATE_MISSING:{key}")
    controls = target.get("ood_controls", {})
    for key in ["heldout_priority_order_accuracy", "heldout_block_order_accuracy", "heldout_template_accuracy", "heldout_rule_composition_accuracy"]:
        if key not in controls:
            failures.append(f"TARGET_OOD_CONTROL_MISSING:{key}")
    expected_route = target.get("expected_positive_route", {})
    if expected_route.get("decision") != "lm_style_canonical_structured_text_distillation_prototype_positive":
        failures.append(f"TARGET_BAD_147A_DECISION:{expected_route.get('decision')}")
    if expected_route.get("next") != "147H_LM_STYLE_CANONICAL_STRUCTURED_TEXT_DISTILLATION_SCALE_CONFIRM":
        failures.append(f"TARGET_BAD_147A_NEXT:{expected_route.get('next')}")


def require_artifact_payloads(root: Path, failures: list[str]) -> None:
    matrix = load_json(root / "next_decision_matrix.json")
    options = {item.get("option"): item for item in matrix.get("options", [])}
    for option in [
        "lm_style_canonical_structured_text_distillation_prototype",
        "scale_raw_text_perceptron_curriculum_further",
        "natural_language_wrapper_before_sequence_model",
        "helper_engine_extension_after_distillation",
        "stop_at_raw_text_distillation_bridge",
    ]:
        if option not in options:
            failures.append(f"DECISION_OPTION_MISSING:{option}")
    if matrix.get("selected_option") != EXPECTED_OPTION:
        failures.append(f"MATRIX_BAD_SELECTED_OPTION:{matrix.get('selected_option')}")
    anti = load_json(root / "anti_oracle_requirements.json")
    for key in ["teacher trace fields", "selected_pocket_id", "ANSWER=<value>", "winner=pocket_*"]:
        if key not in anti.get("model_facing_inputs_must_forbid", []):
            failures.append(f"ANTI_ORACLE_INPUT_FORBID_MISSING:{key}")
    if anti.get("anti_memorization_report_required") is not True or anti.get("ood_split_definition_report_required") is not True:
        failures.append("ANTI_ORACLE_REQUIRED_REPORT_FLAG_MISSING")
    for name in [
        "evidence_chain_summary.json",
        "trainable_distillation_state_report.json",
        "model_architecture_gap_analysis.json",
        "next_decision_matrix.json",
        "anti_oracle_requirements.json",
        "risk_register.json",
    ]:
        payload = load_json(root / name)
        if payload.get("passed") is not True:
            failures.append(f"ARTIFACT_NOT_PASSED:{name}")


def require_docs(failures: list[str]) -> None:
    for doc in DOCS:
        path = REPO_ROOT / doc
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        if len(text.strip()) < 500:
            failures.append(f"DOC_TOO_SHORT:{doc}")
        for phrase in BOUNDARY_PHRASES:
            if phrase not in text:
                failures.append(f"DOC_BOUNDARY_MISSING:{doc}:{phrase}")
        for term in [EXPECTED_DECISION, EXPECTED_OPTION, EXPECTED_NEXT, "SELECTED=A", "generated_schema_report.json"]:
            if term not in text:
                failures.append(f"DOC_TERM_MISSING:{doc}:{term}")


def run_checks(root: Path, check_changed_files: bool) -> list[str]:
    failures: list[str] = []
    if check_changed_files:
        require_changed_files(failures)
    require_static_files(failures)
    require_artifacts(root, failures)
    if failures:
        return failures
    require_upstream(root, failures)
    require_decision(root, failures)
    require_target_147a(root, failures)
    require_artifact_payloads(root, failures)
    require_docs(failures)
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Check 146Z trainable distillation next decision plan")
    parser.add_argument("--root", type=Path, default=SMOKE_ROOT)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--skip-changed-files-check", action="store_true")
    args = parser.parse_args()
    root = args.root if args.root.is_absolute() else (REPO_ROOT / args.root).resolve()
    failures = run_checks(root, check_changed_files=not args.skip_changed_files_check)
    if failures:
        print("146Z CHECK FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("146Z CHECK PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
