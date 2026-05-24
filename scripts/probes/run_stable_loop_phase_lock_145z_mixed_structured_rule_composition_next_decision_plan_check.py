#!/usr/bin/env python3
"""Checker for 145Z mixed structured-rule composition next-decision planning milestone."""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_145z_mixed_structured_rule_composition_next_decision_plan/smoke"
HELPER = "scripts/probes/shared_raw_generation_helper.py"
RUNNER = "scripts/probes/run_stable_loop_phase_lock_145z_mixed_structured_rule_composition_next_decision_plan.py"
CHECKER = "scripts/probes/run_stable_loop_phase_lock_145z_mixed_structured_rule_composition_next_decision_plan_check.py"
DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_145Z_MIXED_STRUCTURED_RULE_COMPOSITION_NEXT_DECISION_PLAN_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_145Z_MIXED_STRUCTURED_RULE_COMPOSITION_NEXT_DECISION_PLAN_RESULT.md",
]
ALLOWED_MUTATIONS = {RUNNER, CHECKER, *DOCS}
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "analysis_config.json",
    "upstream_145h_manifest.json",
    "evidence_chain_summary.json",
    "structured_helper_stack_state_report.json",
    "model_facing_bridge_gap_analysis.json",
    "next_decision_matrix.json",
    "target_146a_milestone_plan.json",
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
    "constrained helper/backend evidence only",
    "does not prove natural-language reasoning",
    "not open-ended arbitration",
    "not GPT-like/Gemma-like capability",
    "not production readiness",
    "not architecture superiority",
]
EXPECTED_DECISION = "trainable_structured_reasoning_distillation_bridge_plan_recommended"
EXPECTED_OPTION = "trainable_structured_reasoning_distillation_bridge_plan"
EXPECTED_NEXT = "146A_TRAINABLE_STRUCTURED_REASONING_DISTILLATION_BRIDGE_PROTOTYPE"
EXPECTED_145H_METRICS = {
    "main_eval_rows": 9600,
    "end_to_end_answer_accuracy": 1.0,
    "positive_composition_subset_accuracy": 1.0,
    "fallback_control_subset_accuracy": 1.0,
    "final_selected_pocket_derivation_accuracy": 1.0,
    "selected_pocket_to_marker_binding_accuracy": 1.0,
    "priority_pocket_oracle_rejection_rate": 1.0,
    "rule_composition_ablation_accuracy": 0.0,
    "shared_helper_no_change_since_145a": True,
    "legacy_structured_rule_metadata_regression_passed": True,
    "legacy_selected_pocket_binding_regression_passed": True,
    "deterministic_replay_passed": True,
}
REQUIRED_146A_OUTPUTS = {
    "curriculum_train.jsonl",
    "curriculum_validation.jsonl",
    "curriculum_test.jsonl",
    "curriculum_ood_test.jsonl",
    "teacher_trace_manifest.json",
    "training_config.json",
    "evaluation_report.json",
    "oracle_shortcut_audit.json",
    "decision.json",
    "summary.json",
    "report.md",
}
REQUIRED_146A_METRICS = {
    "teacher_label_reproduction_accuracy",
    "selected_pocket_prediction_accuracy",
    "final_value_prediction_accuracy",
    "heldout_template_accuracy",
    "ood_composition_accuracy",
    "oracle_ablation_accuracy",
    "shortcut_scanner_violation_count",
    "train_validation_leakage_count",
    "test_template_overlap_rate",
    "deterministic_replay_passed",
}
REQUIRED_FORBIDDEN = {
    "selected_pocket_id",
    "winner=pocket_*",
    "final_selected",
    "derived_selected",
    "answer value",
    "gold value",
    "target value",
    "resolved output",
    "expected output",
    "teacher trace fields in input",
    "per-row oracle metadata",
}


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
            if module.startswith("run_stable_loop_phase_lock_") or module == "shared_raw_generation_helper":
                failures.append(f"FORBIDDEN_IMPORT:{path.name}:{module}")
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in {"torch", "tensorflow", "shared_raw_generation_helper"}:
                    failures.append(f"FORBIDDEN_IMPORT:{path.name}:{alias.name}")
        if isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ""
            if name in {"train", "fit", "backward", "step", "forward", "raw_generate", "load_checkpoint", "save_checkpoint"}:
                failures.append(f"FORBIDDEN_CALL:{path.name}:{name}")
    return failures


def require_static_files(failures: list[str]) -> None:
    for rel_path in [RUNNER, CHECKER, *DOCS]:
        path = REPO_ROOT / rel_path
        if not path.exists():
            failures.append(f"MISSING_TRACKED_FILE:{rel_path}")
            continue
        text = path.read_text(encoding="utf-8")
        for term in ["145Z", "planning-only", "146A", EXPECTED_OPTION]:
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


def require_upstream(upstream: dict[str, Any], failures: list[str]) -> None:
    decision = upstream.get("decision", {})
    metrics = upstream.get("aggregate_metrics", {})
    if decision.get("decision") != "mixed_structured_rule_composition_priority_binding_scale_confirmed":
        failures.append(f"BAD_145H_DECISION:{decision.get('decision')}")
    if decision.get("verdict") != "INSTNCT_MIXED_STRUCTURED_RULE_COMPOSITION_PRIORITY_BINDING_SCALE_CONFIRMED":
        failures.append(f"BAD_145H_VERDICT:{decision.get('verdict')}")
    if decision.get("next") != "145Z_MIXED_STRUCTURED_RULE_COMPOSITION_NEXT_DECISION_PLAN":
        failures.append(f"BAD_145H_NEXT:{decision.get('next')}")
    if upstream.get("failed_checks") != []:
        failures.append(f"UPSTREAM_145H_FAILED_CHECKS:{upstream.get('failed_checks')}")
    for key, expected in EXPECTED_145H_METRICS.items():
        if metrics.get(key) != expected:
            failures.append(f"UPSTREAM_145H_METRIC_MISMATCH:{key}:{metrics.get(key)}")
    for key in [
        "helper_no_change_audit_passed",
        "denominator_report_passed",
        "priority_order_coverage_passed",
        "block_type_candidate_coverage_passed",
        "request_audit_passed",
        "static_manifest_integrity_passed",
    ]:
        if upstream.get("checks", {}).get(key) is not True:
            failures.append(f"UPSTREAM_145H_GATE_NOT_TRUE:{key}:{upstream.get('checks', {}).get(key)}")


def require_gap_and_state(root: Path, failures: list[str]) -> None:
    evidence = load_json(root / "evidence_chain_summary.json")
    state = load_json(root / "structured_helper_stack_state_report.json")
    gap = load_json(root / "model_facing_bridge_gap_analysis.json")
    for payload_name, payload in [("evidence", evidence), ("state", state), ("gap", gap)]:
        if payload.get("passed") is not True:
            failures.append(f"{payload_name.upper()}_REPORT_NOT_PASSED")
    if gap.get("structured_helper_stack_scale_confirmed") is not True:
        failures.append("GAP_STACK_NOT_CONFIRMED")
    if gap.get("trainable_model_internalization_untested") is not True:
        failures.append("GAP_MODEL_INTERNALIZATION_NOT_MARKED_UNTESTED")
    if gap.get("natural_language_rule_reasoning_untested") is not True:
        failures.append("GAP_NL_REASONING_NOT_MARKED_UNTESTED")
    if gap.get("open_ended_arbitration_claimed") is not False:
        failures.append("GAP_OPEN_ENDED_ARBITRATION_CLAIMED")
    if gap.get("gpt_like_or_gemma_like_capability_claimed") is not False:
        failures.append("GAP_GPT_GEMMA_CLAIMED")


def require_decision_matrix_and_target(root: Path, failures: list[str]) -> None:
    matrix = load_json(root / "next_decision_matrix.json")
    target = load_json(root / "target_146a_milestone_plan.json")
    anti = load_json(root / "anti_oracle_requirements.json")
    risk = load_json(root / "risk_register.json")
    if matrix.get("selected_option") != EXPECTED_OPTION or matrix.get("decision") != EXPECTED_DECISION:
        failures.append("DECISION_MATRIX_BAD_SELECTION")
    options = {item.get("option"): item for item in matrix.get("options", [])}
    for option in [
        "trainable_structured_reasoning_distillation_bridge_plan",
        "structured_helper_engine_extension_plan",
        "language_interface_wrapper_around_helper_plan",
        "stop_at_mixed_structured_rule_helper_primitive",
    ]:
        if option not in options:
            failures.append(f"DECISION_OPTION_MISSING:{option}")
    if options.get(EXPECTED_OPTION, {}).get("recommended") is not True:
        failures.append("EXPECTED_OPTION_NOT_RECOMMENDED")
    if target.get("milestone") != EXPECTED_NEXT or target.get("implementation_ready") is not True:
        failures.append("TARGET_146A_NOT_IMPLEMENTATION_READY")
    if target.get("first_model_facing_bridge_prototype") is not True:
        failures.append("TARGET_146A_NOT_MODEL_FACING_BRIDGE")
    if target.get("input_policy", {}).get("canonical_structured_prompts_only") is not True:
        failures.append("TARGET_146A_NOT_CANONICAL_STRUCTURED_ONLY")
    if target.get("input_policy", {}).get("free_form_natural_language_rule_parsing") is not False:
        failures.append("TARGET_146A_ALLOWS_FREE_FORM_NL")
    if not REQUIRED_146A_OUTPUTS.issubset(set(target.get("required_outputs", []))):
        failures.append("TARGET_146A_OUTPUTS_INCOMPLETE")
    if not REQUIRED_146A_METRICS.issubset(set(target.get("required_metrics", []))):
        failures.append("TARGET_146A_METRICS_INCOMPLETE")
    if not REQUIRED_FORBIDDEN.issubset(set(target.get("model_facing_input_forbidden_shortcuts", []))):
        failures.append("TARGET_146A_FORBIDDEN_SHORTCUTS_INCOMPLETE")
    if set(target.get("clean_negative_routes", {})) != {
        "curriculum_generation_failure",
        "train_eval_leakage_detected",
        "model_shortcut_detected",
        "teacher_label_reproduction_failure",
        "ood_generalization_failure",
        "helper_stack_regression",
    }:
        failures.append("TARGET_146A_NEGATIVE_ROUTES_INCOMPLETE")
    if anti.get("passed") is not True or not REQUIRED_FORBIDDEN.issubset(set(anti.get("model_facing_inputs_must_forbid", []))):
        failures.append("ANTI_ORACLE_REQUIREMENTS_FAIL")
    if anti.get("oracle_ablation_required") is not True or anti.get("shortcut_scanner_required") is not True:
        failures.append("ANTI_ORACLE_MISSING_ABLATION_OR_SCANNER")
    if risk.get("passed") is not True or len(risk.get("risks", [])) < 3:
        failures.append("RISK_REGISTER_INCOMPLETE")


def require_decision_and_boundary(root: Path, failures: list[str]) -> None:
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
    if summary.get("target_146a_implementation_ready") is not True:
        failures.append("SUMMARY_TARGET_146A_NOT_READY")
    if config.get("raw_generate_allowed") is not False or config.get("shared_helper_import_allowed") is not False:
        failures.append("CONFIG_FORBIDDEN_EXECUTION_ALLOWED")
    if config.get("training_or_checkpoint_mutation_allowed") is not False:
        failures.append("CONFIG_TRAINING_OR_CHECKPOINT_MUTATION_ALLOWED")
    for payload_name, payload in [("decision", decision), ("summary", summary), ("config", config)]:
        require_false_flags(payload, failures, payload_name)
        text = json.dumps(payload)
        for phrase in BOUNDARY_PHRASES:
            if phrase not in text:
                failures.append(f"BOUNDARY_MISSING:{payload_name}:{phrase}")
    for phrase in BOUNDARY_PHRASES:
        if phrase not in report:
            failures.append(f"REPORT_BOUNDARY_MISSING:{phrase}")
    for doc in DOCS:
        text = (REPO_ROOT / doc).read_text(encoding="utf-8")
        if len(text.strip()) < 500:
            failures.append(f"DOC_TOO_SHORT:{doc}")
        for phrase in BOUNDARY_PHRASES:
            if phrase not in text:
                failures.append(f"DOC_BOUNDARY_MISSING:{doc}:{phrase}")


def run_checks(root: Path, check_changed_files: bool) -> list[str]:
    failures: list[str] = []
    if check_changed_files:
        require_changed_files(failures)
    require_static_files(failures)
    require_artifacts(root, failures)
    if failures:
        return failures
    upstream = load_json(root / "upstream_145h_manifest.json")
    require_upstream(upstream, failures)
    require_gap_and_state(root, failures)
    require_decision_matrix_and_target(root, failures)
    require_decision_and_boundary(root, failures)
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Check 145Z mixed structured-rule composition next-decision plan")
    parser.add_argument("--root", type=Path, default=SMOKE_ROOT)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--skip-changed-files-check", action="store_true")
    args = parser.parse_args()
    root = args.root if args.root.is_absolute() else (REPO_ROOT / args.root).resolve()
    failures = run_checks(root, check_changed_files=not args.skip_changed_files_check)
    if failures:
        print("145Z CHECK FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("145Z CHECK PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
