#!/usr/bin/env python3
"""Checker for 147Z LM-style distillation next-decision planning milestone."""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_147z_lm_style_canonical_structured_text_distillation_next_decision_plan/smoke"
HELPER = "scripts/probes/shared_raw_generation_helper.py"
RUNNER = "scripts/probes/run_stable_loop_phase_lock_147z_lm_style_canonical_structured_text_distillation_next_decision_plan.py"
CHECKER = "scripts/probes/run_stable_loop_phase_lock_147z_lm_style_canonical_structured_text_distillation_next_decision_plan_check.py"
DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_147Z_LM_STYLE_CANONICAL_STRUCTURED_TEXT_DISTILLATION_NEXT_DECISION_PLAN_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_147Z_LM_STYLE_CANONICAL_STRUCTURED_TEXT_DISTILLATION_NEXT_DECISION_PLAN_RESULT.md",
]
ALLOWED_MUTATIONS = {RUNNER, CHECKER, *DOCS}
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "analysis_config.json",
    "upstream_147h_manifest.json",
    "evidence_chain_summary.json",
    "lm_style_distillation_state_report.json",
    "full_line_generation_gap_analysis.json",
    "next_decision_matrix.json",
    "target_148a_milestone_plan.json",
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
EXPECTED_DECISION = "full_selected_line_generation_prototype_plan_recommended"
EXPECTED_OPTION = "full_selected_line_generation_prototype"
EXPECTED_NEXT = "148A_FULL_SELECTED_LINE_GENERATION_PROTOTYPE"
VALID_RAW_LINES = {"SELECTED=A", "SELECTED=B", "SELECTED=C", "SELECTED=fallback"}
REQUIRED_148A_REPORTS = {
    "generation_prefix_audit.json",
    "raw_generation_audit.json",
    "decoding_audit.json",
    "full_line_generation_report.json",
    "generated_schema_report.json",
    "generation_input_audit.json",
    "anti_memorization_report.json",
    "ood_generation_family_report.json",
    "baseline_margin_report.json",
    "shortcut_scanner_report.json",
    "leakage_audit.json",
    "model_artifact_audit.json",
    "deterministic_replay_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
}
REQUIRED_CONTROLS = {
    "SAME_BLOCKS_DIFFERENT_PRIORITY",
    "SAME_PRIORITY_DIFFERENT_BLOCK_VALUES",
    "SAME_TEMPLATE_OPPOSITE_PRIORITY_WINNER",
    "PRIORITY_ORDER_HOLDOUT",
    "BLOCK_ORDER_HOLDOUT",
    "RULE_BLOCK_TYPE_COMBINATION_HOLDOUT",
    "INVALID_HIGH_PRIORITY_FALLTHROUGH_OOD",
    "STRUCTURAL_INVALID_PROMPT_OOD",
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
            if module == "shared_raw_generation_helper":
                failures.append(f"FORBIDDEN_IMPORT:{path.name}:{module}")
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
        for term in ["147Z", "planning-only", "148A", EXPECTED_OPTION]:
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
    upstream = load_json(root / "upstream_147h_manifest.json")
    decision = upstream.get("decision", {})
    metrics = upstream.get("aggregate_metrics", {})
    if decision.get("decision") != "lm_style_canonical_structured_text_distillation_scale_confirmed":
        failures.append(f"BAD_147H_DECISION:{decision.get('decision')}")
    if decision.get("verdict") != "INSTNCT_LM_STYLE_CANONICAL_STRUCTURED_TEXT_DISTILLATION_SCALE_CONFIRMED":
        failures.append(f"BAD_147H_VERDICT:{decision.get('verdict')}")
    if decision.get("next") != "147Z_LM_STYLE_CANONICAL_STRUCTURED_TEXT_DISTILLATION_NEXT_DECISION_PLAN":
        failures.append(f"BAD_147H_NEXT:{decision.get('next')}")
    exact = {
        "selected_label_generation_accuracy": 1.0,
        "selected_label_byte_accuracy": 1.0,
        "final_value_from_generated_label_accuracy": 1.0,
        "generated_output_schema_valid_rate": 1.0,
        "ood_selected_accuracy": 1.0,
        "minimum_ood_family_accuracy": 1.0,
        "minimum_ood_family_row_count": 140,
        "shuffled_target_control_accuracy": 0.0,
        "generation_deterministic_replay_passed": True,
        "model_generates_full_selected_line": False,
        "schema_prefix_fixed_by_runner": True,
        "selected_line_wrapper_deterministic": True,
    }
    for key, expected in exact.items():
        if metrics.get(key) != expected:
            failures.append(f"UPSTREAM_147H_METRIC_MISMATCH:{key}:{metrics.get(key)}")
    if upstream.get("failed_checks") != [] or upstream.get("passed") is not True:
        failures.append(f"UPSTREAM_147H_FAILED_CHECKS:{upstream.get('failed_checks')}")
    for key in [
        "label_byte_report_passed",
        "label_byte_report_full_line_false",
        "label_byte_report_prefix_fixed",
        "same_model_family_audit_passed",
        "generation_input_audit_passed",
        "generation_input_ends_with_selected_prefix",
        "schema_report_passed",
        "ood_report_passed",
        "deterministic_replay_report_passed",
    ]:
        if upstream.get("checks", {}).get(key) is not True:
            failures.append(f"UPSTREAM_147H_CHECK_NOT_TRUE:{key}")


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
    gap = load_json(root / "full_line_generation_gap_analysis.json")
    matrix = load_json(root / "next_decision_matrix.json")
    if gap.get("selected_label_byte_generation_scale_confirmed") is not True:
        failures.append("GAP_SELECTED_BYTE_NOT_CONFIRMED")
    if gap.get("fixed_selected_prefix_used_by_147h") is not True:
        failures.append("GAP_FIXED_PREFIX_NOT_RECORDED")
    if gap.get("model_generates_full_selected_line") is not False:
        failures.append("GAP_FULL_LINE_NOT_MARKED_FALSE")
    if gap.get("full_selected_line_generation_untested") is not True:
        failures.append("GAP_FULL_LINE_NOT_MARKED_UNTESTED")
    if gap.get("hidden_wrapper_risk_for_148a") is not True:
        failures.append("GAP_HIDDEN_WRAPPER_RISK_MISSING")
    if matrix.get("selected_option") != EXPECTED_OPTION:
        failures.append(f"MATRIX_BAD_SELECTED_OPTION:{matrix.get('selected_option')}")
    options = {row.get("option"): row for row in matrix.get("options", [])}
    for option in [
        "full_selected_line_generation_prototype",
        "multi_token_schema_generation_prototype",
        "controlled_natural_language_wrapper_later_plan",
        "scale_selected_byte_bridge_further",
        "stop_at_selected_label_byte_generation",
    ]:
        if option not in options:
            failures.append(f"MATRIX_OPTION_MISSING:{option}")
    if options.get(EXPECTED_OPTION, {}).get("recommended") is not True:
        failures.append("MATRIX_EXPECTED_OPTION_NOT_RECOMMENDED")


def require_target_148a(root: Path, failures: list[str]) -> None:
    target = load_json(root / "target_148a_milestone_plan.json")
    if target.get("milestone") != EXPECTED_NEXT:
        failures.append(f"TARGET_BAD_MILESTONE:{target.get('milestone')}")
    if target.get("implementation_ready") is not True:
        failures.append("TARGET_NOT_IMPLEMENTATION_READY")
    if target.get("expected_positive_route", {}).get("decision") != "full_selected_line_generation_prototype_positive":
        failures.append("TARGET_BAD_EXPECTED_POSITIVE_DECISION")
    if set(target.get("required_reports", [])) != REQUIRED_148A_REPORTS:
        failures.append("TARGET_REQUIRED_REPORTS_MISMATCH")
    if set(target.get("valid_raw_generated_outputs", [])) != VALID_RAW_LINES:
        failures.append(f"TARGET_BAD_VALID_RAW_LINES:{target.get('valid_raw_generated_outputs')}")
    generation_input = target.get("generation_input_policy", {})
    required_generation_input = {
        "eval_generation_input_must_end_exactly_with": "<OUTPUT>\\n",
        "forbidden_eval_generation_input_suffix": "<OUTPUT>\\nSELECTED=",
        "eval_generation_input_ends_with_output_delimiter": True,
        "eval_generation_input_contains_selected_prefix": False,
        "runner_prepends_selected_prefix": False,
        "runner_wraps_label_byte": False,
        "deterministic_selected_line_wrapper_used": False,
        "model_generates_selected_prefix": True,
        "model_generates_full_selected_line": True,
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
        "casing_repair_used": False,
        "prefix_repair_used": False,
    }
    for key, expected in required_raw.items():
        if raw.get(key) != expected:
            failures.append(f"TARGET_RAW_GENERATION_BAD:{key}:{raw.get(key)}")
    decoding = target.get("decoding_policy", {})
    required_decoding = {
        "autoregressive_generation_used": True,
        "forced_selected_prefix_used": False,
        "constrained_label_only_decoding_used": False,
        "stop_on_newline_or_max_len": True,
        "max_new_bytes_required": True,
    }
    for key, expected in required_decoding.items():
        if decoding.get(key) != expected:
            failures.append(f"TARGET_DECODING_BAD:{key}:{decoding.get(key)}")
    final_value = target.get("final_value_policy", {})
    if final_value.get("direct_opaque_value_token_generation_required") is not False:
        failures.append("TARGET_OPAQUE_VALUE_GENERATION_REQUIRED")
    gates = target.get("required_gates", {})
    expected_gate_checks = {
        "selected_prefix_generation_accuracy": (">=", 0.70),
        "selected_label_generation_accuracy": (">=", 0.70),
        "full_selected_line_exact_match_rate": (">=", 0.70),
        "generated_output_schema_valid_rate": (">=", 0.80),
        "final_value_from_generated_line_accuracy": (">=", 0.70),
        "extra_text_generation_rate": ("<=", 0.20),
        "multiple_selected_line_rate": ("==", 0.0),
        "answer_value_generation_rate": ("==", 0.0),
        "selected_pocket_id_generation_rate": ("==", 0.0),
        "ood_full_line_accuracy": (">=", 0.50),
        "full_line_generation_accuracy_over_best_baseline": (">=", 0.10),
        "shuffled_target_control_accuracy": ("<=", 0.35),
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
    expected_audit = {
        "eval_generation_input_ends_with_output_delimiter": True,
        "eval_generation_input_contains_selected_prefix": False,
        "runner_prepends_selected_prefix": False,
        "model_generates_selected_prefix": True,
        "model_generates_full_selected_line": True,
        "deterministic_selected_line_wrapper_used": False,
        "autoregressive_generation_used": True,
        "forced_selected_prefix_used": False,
        "constrained_label_only_decoding_used": False,
        "stop_on_newline_or_max_len": True,
        "raw_generated_text_stored": True,
        "schema_scored_from_raw_generated_text": True,
        "post_generation_repair_used": False,
        "selected_line_extracted_from_substring": False,
        "casing_repair_used": False,
        "prefix_repair_used": False,
    }
    for key, expected in expected_audit.items():
        if audit.get(key) != expected:
            failures.append(f"TARGET_AUDIT_BAD:{key}:{audit.get(key)}")
    if set(target.get("required_control_families", [])) != REQUIRED_CONTROLS:
        failures.append("TARGET_CONTROL_FAMILIES_MISMATCH")
    routes = target.get("clean_negative_routes", {})
    if routes.get("hidden_wrapper_detected") != "148J_HIDDEN_SELECTED_PREFIX_WRAPPER_ANALYSIS":
        failures.append("TARGET_HIDDEN_WRAPPER_ROUTE_MISSING")
    if routes.get("generation_input_leakage_detected") != "148G_FULL_LINE_INPUT_LEAKAGE_ANALYSIS":
        failures.append("TARGET_INPUT_LEAKAGE_ROUTE_MISSING")


def require_anti_oracle(root: Path, failures: list[str]) -> None:
    anti = load_json(root / "anti_oracle_requirements.json")
    if anti.get("hidden_wrapper_forbidden") is not True:
        failures.append("ANTI_HIDDEN_WRAPPER_NOT_FORBIDDEN")
    if anti.get("runner_prepends_selected_prefix") is not False:
        failures.append("ANTI_RUNNER_PREFIX_NOT_FALSE")
    if anti.get("deterministic_selected_line_wrapper_used") is not False:
        failures.append("ANTI_WRAPPER_NOT_FALSE")
    if anti.get("schema_scored_from_raw_generated_text_required") is not True:
        failures.append("ANTI_RAW_SCHEMA_SCORING_NOT_REQUIRED")
    for forbidden in ["selected_pocket_id", "ANSWER=", "GOLD=", "TARGET=", "EXPECTED="]:
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
    require_target_148a(root, failures)
    require_anti_oracle(root, failures)
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(SMOKE_ROOT), help="147Z smoke artifact root")
    parser.add_argument("--check-only", action="store_true", help="Compatibility flag; checker always checks only")
    parser.add_argument("--skip-changed-files-check", action="store_true", help="Allow running while later milestone files are present")
    args = parser.parse_args()
    failures = check(Path(args.root), args.skip_changed_files_check)
    if failures:
        print("147Z CHECK FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("147Z CHECK PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
