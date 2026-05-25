#!/usr/bin/env python3
"""Checker for 148Z full selected-line next-decision planning milestone."""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_148z_full_selected_line_generation_next_decision_plan/smoke"
HELPER = "scripts/probes/shared_raw_generation_helper.py"
RUNNER = "scripts/probes/run_stable_loop_phase_lock_148z_full_selected_line_generation_next_decision_plan.py"
CHECKER = "scripts/probes/run_stable_loop_phase_lock_148z_full_selected_line_generation_next_decision_plan_check.py"
DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_148Z_FULL_SELECTED_LINE_GENERATION_NEXT_DECISION_PLAN_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_148Z_FULL_SELECTED_LINE_GENERATION_NEXT_DECISION_PLAN_RESULT.md",
]
ALLOWED_MUTATIONS = {RUNNER, CHECKER, *DOCS}
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "analysis_config.json",
    "upstream_148h_manifest.json",
    "evidence_chain_summary.json",
    "full_selected_line_generation_state_report.json",
    "bounded_decision_schema_gap_analysis.json",
    "next_decision_matrix.json",
    "target_149a_milestone_plan.json",
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
    "bounded decision schema",
    "not natural-language rule reasoning",
    "not open-ended arbitration",
    "not GPT-like/Gemma-like assistant capability",
    "not production readiness",
    "not architecture superiority",
]
EXPECTED_DECISION = "bounded_decision_schema_generation_prototype_plan_recommended"
EXPECTED_OPTION = "bounded_decision_schema_generation_prototype"
EXPECTED_NEXT = "149A_BOUNDED_DECISION_SCHEMA_GENERATION_PROTOTYPE"
VALID_SELECTED_LINES = {"SELECTED=A", "SELECTED=B", "SELECTED=C", "SELECTED=fallback"}
VALID_REASON_CODES = {
    "priority_quorum",
    "priority_recency",
    "priority_validity",
    "fallback_invalid_high_priority",
    "structural_invalid_fallback",
}
REQUIRED_149A_REPORTS = {
    "queue.json",
    "progress.jsonl",
    "upstream_148z_manifest.json",
    "curriculum_train.jsonl",
    "curriculum_validation.jsonl",
    "curriculum_test.jsonl",
    "curriculum_ood_test.jsonl",
    "sequence_train_corpus.txt",
    "sequence_validation_corpus.txt",
    "training_config.json",
    "lm_training_metrics.jsonl",
    "bounded_decision_schema_report.json",
    "selected_line_generation_report.json",
    "reason_code_generation_report.json",
    "generated_schema_report.json",
    "generation_input_audit.json",
    "raw_generation_audit.json",
    "decoding_audit.json",
    "final_value_copy_report.json",
    "label_distribution_report.json",
    "reason_code_distribution_report.json",
    "ood_bounded_schema_family_report.json",
    "anti_memorization_report.json",
    "baseline_margin_report.json",
    "shuffled_target_control_report.json",
    "shortcut_scanner_report.json",
    "leakage_audit.json",
    "value_token_leakage_report.json",
    "model_artifact_audit.json",
    "deterministic_replay_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
}
REQUIRED_CONTROLS = {
    "SAME_PROMPT_STRUCTURE_DIFFERENT_SELECTED_AND_REASON",
    "SAME_RULE_BLOCKS_DIFFERENT_PRIORITY",
    "SAME_PRIORITY_DIFFERENT_BLOCK_VALUES",
    "SAME_TEMPLATE_OPPOSITE_SELECTED_LINE",
    "SAME_SELECTED_DIFFERENT_REASON_CODE",
    "REASON_CODE_SHUFFLED_TARGET_CONTROL",
    "OOD_BOUNDED_SCHEMA_GENERATION_CONTROL",
    "GENERATION_INPUT_TARGET_LEAKAGE_CONTROL",
    "RAW_GENERATION_SCHEMA_DRIFT_CONTROL",
    "NATURAL_LANGUAGE_REASON_LEAKAGE_CONTROL",
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
        if isinstance(node, ast.ImportFrom) and (node.module or "") == "shared_raw_generation_helper":
            failures.append(f"FORBIDDEN_IMPORT:{path.name}:shared_raw_generation_helper")
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in {"torch", "tensorflow", "shared_raw_generation_helper", "requests", "socket", "urllib", "http.client"}:
                    failures.append(f"FORBIDDEN_IMPORT:{path.name}:{alias.name}")
        if isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ""
            if name in {"train", "fit", "forward", "backward", "step", "raw_generate", "load_checkpoint", "save_checkpoint", "urlopen"}:
                failures.append(f"FORBIDDEN_CALL:{path.name}:{name}")
    return failures


def require_static_files(failures: list[str]) -> None:
    for rel_path in [RUNNER, CHECKER, *DOCS]:
        path = REPO_ROOT / rel_path
        if not path.exists():
            failures.append(f"MISSING_TRACKED_FILE:{rel_path}")
            continue
        text = path.read_text(encoding="utf-8")
        for term in ["148Z", "planning-only", "149A", EXPECTED_OPTION]:
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
    upstream = load_json(root / "upstream_148h_manifest.json")
    decision = upstream.get("decision", {})
    metrics = upstream.get("aggregate_metrics", {})
    if decision.get("decision") != "full_selected_line_generation_scale_confirmed":
        failures.append(f"BAD_148H_DECISION:{decision.get('decision')}")
    if decision.get("verdict") != "INSTNCT_FULL_SELECTED_LINE_GENERATION_SCALE_CONFIRMED":
        failures.append(f"BAD_148H_VERDICT:{decision.get('verdict')}")
    if decision.get("next") != "148Z_FULL_SELECTED_LINE_GENERATION_NEXT_DECISION_PLAN":
        failures.append(f"BAD_148H_NEXT:{decision.get('next')}")
    thresholds = {
        "full_selected_line_exact_match_rate": 0.998,
        "full_line_generation_accuracy": 0.998,
        "selected_label_generation_accuracy": 0.998,
        "final_value_from_generated_line_accuracy": 0.998,
        "ood_full_line_accuracy": 0.995,
        "minimum_ood_family_accuracy": 0.94,
    }
    for key, threshold in thresholds.items():
        if metrics.get(key, 0.0) < threshold:
            failures.append(f"UPSTREAM_148H_METRIC_LOW:{key}:{metrics.get(key)}")
    exact = {
        "selected_prefix_generation_accuracy": 1.0,
        "generated_output_schema_valid_rate": 1.0,
        "minimum_ood_family_row_count": 140,
        "generation_deterministic_replay_passed": True,
        "eval_generation_input_contains_selected_prefix": False,
        "runner_prepends_selected_prefix": False,
        "deterministic_selected_line_wrapper_used": False,
        "post_generation_repair_used": False,
        "selected_line_extracted_from_substring": False,
        "first_byte_only_training_used": False,
        "constrained_label_only_decoding_used": False,
    }
    for key, expected in exact.items():
        if metrics.get(key) != expected:
            failures.append(f"UPSTREAM_148H_METRIC_MISMATCH:{key}:{metrics.get(key)}")
    if metrics.get("shuffled_target_control_accuracy", 1.0) > 0.01:
        failures.append(f"UPSTREAM_148H_SHUFFLED_TOO_HIGH:{metrics.get('shuffled_target_control_accuracy')}")
    if upstream.get("failed_checks") != [] or upstream.get("passed") is not True:
        failures.append(f"UPSTREAM_148H_FAILED_CHECKS:{upstream.get('failed_checks')}")
    for key in [
        "prefix_audit_passed",
        "model_generates_full_selected_line",
        "raw_audit_passed",
        "decoding_audit_passed",
        "schema_report_passed",
        "ood_report_passed",
        "model_artifact_report_passed",
        "same_model_family_as_148a",
        "deterministic_replay_report_passed",
    ]:
        if upstream.get("checks", {}).get(key) is not True:
            failures.append(f"UPSTREAM_148H_CHECK_NOT_TRUE:{key}")


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
    for key in [
        "raw_generate_allowed",
        "shared_helper_import_allowed",
        "helper_modification_allowed",
        "training_allowed",
        "torch_forward_pass_allowed",
        "checkpoint_mutation_allowed",
        "external_api_allowed",
        "external_model_download_allowed",
        "natural_language_input_claimed",
    ]:
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


def require_gap_and_matrix(root: Path, failures: list[str]) -> None:
    state = load_json(root / "full_selected_line_generation_state_report.json")
    gap = load_json(root / "bounded_decision_schema_gap_analysis.json")
    matrix = load_json(root / "next_decision_matrix.json")
    if state.get("full_selected_line_scale_confirmed") is not True:
        failures.append("STATE_FULL_LINE_NOT_CONFIRMED")
    if state.get("bounded_reason_code_generation_untested") is not True:
        failures.append("STATE_REASON_CODE_GAP_MISSING")
    if gap.get("answer_plus_bounded_reason_bridge_missing") is not True:
        failures.append("GAP_BOUNDED_REASON_BRIDGE_MISSING")
    if gap.get("controlled_natural_language_still_too_early") is not True:
        failures.append("GAP_NL_TOO_EARLY_MISSING")
    if matrix.get("selected_option") != EXPECTED_OPTION:
        failures.append(f"MATRIX_BAD_SELECTED_OPTION:{matrix.get('selected_option')}")
    options = {row.get("option"): row for row in matrix.get("options", [])}
    for option in [
        "full_selected_line_plus_explanation_stub",
        EXPECTED_OPTION,
        "controlled_natural_language_wrapper_later_plan",
        "scale_current_full_line_bridge_further",
        "stop_at_full_selected_line_generation",
    ]:
        if option not in options:
            failures.append(f"MATRIX_OPTION_MISSING:{option}")
    if options.get(EXPECTED_OPTION, {}).get("recommended") is not True:
        failures.append("MATRIX_EXPECTED_OPTION_NOT_RECOMMENDED")


def require_target_149a(root: Path, failures: list[str]) -> None:
    target = load_json(root / "target_149a_milestone_plan.json")
    if target.get("milestone") != EXPECTED_NEXT:
        failures.append(f"TARGET_BAD_MILESTONE:{target.get('milestone')}")
    if target.get("implementation_ready") is not True:
        failures.append("TARGET_NOT_IMPLEMENTATION_READY")
    if target.get("expected_positive_route", {}).get("decision") != "bounded_decision_schema_generation_prototype_positive":
        failures.append("TARGET_BAD_EXPECTED_POSITIVE_DECISION")
    if set(target.get("required_reports", [])) != REQUIRED_149A_REPORTS:
        failures.append("TARGET_REQUIRED_REPORTS_MISMATCH")
    schema = target.get("valid_schema", {})
    if set(schema.get("line_1", [])) != VALID_SELECTED_LINES:
        failures.append("TARGET_SELECTED_LINES_MISMATCH")
    if {item.replace("REASON_CODE=", "") for item in schema.get("line_2", [])} != VALID_REASON_CODES:
        failures.append("TARGET_REASON_CODES_MISMATCH")
    if schema.get("line_count") != 2 or schema.get("extra_text_allowed") is not False:
        failures.append("TARGET_SCHEMA_SHAPE_BAD")
    generation_input = target.get("generation_input_policy", {})
    required_generation_input = {
        "eval_generation_input_must_end_exactly_with": "<OUTPUT>\\n",
        "eval_generation_input_contains_selected_line": False,
        "eval_generation_input_contains_reason_code": False,
        "runner_prepends_selected_line": False,
        "runner_prepends_reason_code": False,
        "deterministic_schema_wrapper_used": False,
        "model_generates_selected_line": True,
        "model_generates_reason_code_line": True,
        "model_generates_full_bounded_schema": True,
    }
    for key, expected in required_generation_input.items():
        if generation_input.get(key) != expected:
            failures.append(f"TARGET_GENERATION_INPUT_BAD:{key}:{generation_input.get(key)}")
    raw = target.get("raw_generation_policy", {})
    required_raw = {
        "raw_generated_text_stored": True,
        "schema_scored_from_raw_generated_text": True,
        "post_generation_repair_used": False,
        "selected_line_extracted_from_substring": False,
        "reason_code_extracted_from_substring": False,
        "casing_repair_used": False,
        "prefix_repair_used": False,
        "label_repair_used": False,
        "reason_code_repair_used": False,
    }
    for key, expected in required_raw.items():
        if raw.get(key) != expected:
            failures.append(f"TARGET_RAW_GENERATION_BAD:{key}:{raw.get(key)}")
    decoding = target.get("decoding_policy", {})
    required_decoding = {
        "autoregressive_generation_used": True,
        "full_bounded_schema_target_used": True,
        "selected_line_only_training_used": False,
        "forced_selected_prefix_used": False,
        "forced_reason_code_prefix_used": False,
        "constrained_label_or_reason_only_decoding_used": False,
        "stop_on_double_newline_or_max_len": True,
        "max_new_bytes_required": True,
    }
    for key, expected in required_decoding.items():
        if decoding.get(key) != expected:
            failures.append(f"TARGET_DECODING_BAD:{key}:{decoding.get(key)}")
    if target.get("final_value_policy", {}).get("direct_opaque_value_token_generation_required") is not False:
        failures.append("TARGET_OPAQUE_VALUE_GENERATION_REQUIRED")
    gates = target.get("required_gates", {})
    expected_gate_checks = {
        "selected_line_generation_accuracy": (">=", 0.70),
        "reason_code_generation_accuracy": (">=", 0.60),
        "full_bounded_schema_exact_match_rate": (">=", 0.60),
        "generated_output_schema_valid_rate": (">=", 0.75),
        "final_value_from_generated_schema_accuracy": (">=", 0.70),
        "ood_bounded_schema_accuracy": (">=", 0.45),
        "selected_line_accuracy_over_best_baseline": (">=", 0.10),
        "reason_code_accuracy_over_best_baseline": (">=", 0.05),
        "shuffled_target_control_accuracy": ("<=", 0.35),
        "answer_value_generation_rate": ("==", 0.0),
        "selected_pocket_id_generation_rate": ("==", 0.0),
        "free_text_reason_generation_rate": ("==", 0.0),
        "extra_text_generation_rate": ("<=", 0.20),
        "shortcut_scanner_violation_count": ("==", 0),
        "train_eval_prompt_overlap_count": ("==", 0),
        "train_ood_prompt_overlap_count": ("==", 0),
        "value_token_overlap_train_test_rate": ("==", 0.0),
        "generation_deterministic_replay_passed": ("==", True),
    }
    for key, (op, value) in expected_gate_checks.items():
        if gates.get(key, {}).get(op) != value:
            failures.append(f"TARGET_GATE_BAD:{key}:{gates.get(key)}")
    audit = target.get("audit_hard_gates", {})
    for key, expected in target.get("generation_input_policy", {}).items():
        if key.startswith("eval_") or key.startswith("runner_") or key in {"deterministic_schema_wrapper_used", "model_generates_full_bounded_schema"}:
            if audit.get(key) != expected and key in audit:
                failures.append(f"TARGET_AUDIT_INCONSISTENT:{key}:{audit.get(key)}")
    for key in [
        "raw_generated_text_stored",
        "schema_scored_from_raw_generated_text",
        "post_generation_repair_used",
        "selected_line_extracted_from_substring",
        "reason_code_extracted_from_substring",
        "forced_reason_code_prefix_used",
        "constrained_label_or_reason_only_decoding_used",
    ]:
        if key not in audit:
            failures.append(f"TARGET_AUDIT_MISSING:{key}")
    if set(target.get("required_control_families", [])) != REQUIRED_CONTROLS:
        failures.append("TARGET_CONTROL_FAMILIES_MISMATCH")
    routes = target.get("clean_negative_routes", {})
    if routes.get("reason_code_generation_failure") != "149E_REASON_CODE_GENERATION_FAILURE_ANALYSIS":
        failures.append("TARGET_REASON_CODE_ROUTE_MISSING")
    if routes.get("natural_language_overclaim_detected") != "149J_NATURAL_LANGUAGE_OVERCLAIM_ANALYSIS":
        failures.append("TARGET_NL_OVERCLAIM_ROUTE_MISSING")


def require_anti_oracle(root: Path, failures: list[str]) -> None:
    anti = load_json(root / "anti_oracle_requirements.json")
    if anti.get("hidden_schema_wrapper_forbidden") is not True:
        failures.append("ANTI_HIDDEN_SCHEMA_WRAPPER_NOT_FORBIDDEN")
    if anti.get("runner_prepends_selected_line") is not False:
        failures.append("ANTI_SELECTED_LINE_PREPEND_NOT_FALSE")
    if anti.get("runner_prepends_reason_code") is not False:
        failures.append("ANTI_REASON_CODE_PREPEND_NOT_FALSE")
    if anti.get("schema_scored_from_raw_generated_text_required") is not True:
        failures.append("ANTI_RAW_SCHEMA_SCORING_NOT_REQUIRED")
    if anti.get("natural_language_reason_generation_forbidden") is not True:
        failures.append("ANTI_NL_REASON_NOT_FORBIDDEN")
    for forbidden in ["selected_pocket_id", "ANSWER=", "GOLD=", "TARGET=", "EXPECTED=", "REASON_CODE="]:
        if forbidden not in anti.get("model_input_forbidden_fields", []):
            failures.append(f"ANTI_FORBIDDEN_FIELD_MISSING:{forbidden}")


def check(root: Path, skip_changed_files_check: bool) -> list[str]:
    failures: list[str] = []
    if not skip_changed_files_check:
        require_changed_files(failures)
    require_static_files(failures)
    require_artifacts(root, failures)
    if failures:
        return failures
    require_upstream(root, failures)
    require_decision(root, failures)
    require_gap_and_matrix(root, failures)
    require_target_149a(root, failures)
    require_anti_oracle(root, failures)
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(SMOKE_ROOT), help="148Z smoke artifact root")
    parser.add_argument("--check-only", action="store_true", help="Compatibility flag; checker always checks only")
    parser.add_argument("--skip-changed-files-check", action="store_true", help="Allow running while later milestone files are present")
    args = parser.parse_args()
    failures = check(Path(args.root), args.skip_changed_files_check)
    if failures:
        print("148Z CHECK FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("148Z CHECK PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
