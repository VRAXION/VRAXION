#!/usr/bin/env python3
"""Checker for 143U duplicate selected marker conflict rejection primitive plan."""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_143u_duplicate_selected_marker_conflict_rejection_helper_primitive_plan/smoke"
RUNNER = "scripts/probes/run_stable_loop_phase_lock_143u_duplicate_selected_marker_conflict_rejection_helper_primitive_plan.py"
CHECKER = "scripts/probes/run_stable_loop_phase_lock_143u_duplicate_selected_marker_conflict_rejection_helper_primitive_plan_check.py"
DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_143U_DUPLICATE_SELECTED_MARKER_CONFLICT_REJECTION_HELPER_PRIMITIVE_PLAN_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_143U_DUPLICATE_SELECTED_MARKER_CONFLICT_REJECTION_HELPER_PRIMITIVE_PLAN_RESULT.md",
]
ALLOWED_MUTATIONS = {RUNNER, CHECKER, *DOCS}
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "analysis_config.json",
    "upstream_143r_manifest.json",
    "primitive_repair_requirements.json",
    "selected_marker_occurrence_policy_matrix.json",
    "repair_options_matrix.json",
    "selected_repair_recommendation.json",
    "anti_oracle_requirements.json",
    "target_143v_milestone_plan.json",
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
    "open_ended_arbitration_claimed",
    "gpt_like_readiness_claimed",
    "open_domain_assistant_readiness_claimed",
    "production_chat_claimed",
    "public_api_claimed",
    "deployment_readiness_claimed",
    "safety_alignment_claimed",
    "architecture_superiority_claimed",
]
BOUNDARY_PHRASES = [
    "planning-only",
    "constrained helper/backend evidence only",
    "prompt-visible selected-pocket binding only",
    "not rule metadata reasoning",
    "not open-ended arbitration",
    "not GPT-like/open-domain/broad assistant capability",
    "not production/public API/deployment/safety readiness",
    "not architecture superiority",
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def git_status() -> str:
    result = subprocess.run(["git", "status", "--short"], cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    return result.stdout.rstrip("\n")


def changed_paths() -> list[str]:
    paths: list[str] = []
    for line in git_status().splitlines():
        if line.strip():
            paths.append(line[3:].replace("\\", "/"))
    return paths


def require_changed_files(failures: list[str]) -> None:
    for path in changed_paths():
        if path.startswith("target/"):
            continue
        if path not in ALLOWED_MUTATIONS:
            failures.append(f"UNEXPECTED_CHANGED_FILE:{path}")


def require_artifacts(root: Path, failures: list[str]) -> None:
    for name in REQUIRED_ARTIFACTS:
        if not (root / name).exists():
            failures.append(f"MISSING_ARTIFACT:{name}")


def require_false_flags(payload: dict[str, Any], failures: list[str], prefix: str) -> None:
    for key in FALSE_FLAGS:
        if payload.get(key) is not False:
            failures.append(f"{prefix}_BOUNDARY_FLAG_NOT_FALSE:{key}:{payload.get(key)}")


def ast_scan(failures: list[str]) -> None:
    for rel_path in [RUNNER, CHECKER]:
        path = REPO_ROOT / rel_path
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) and any(alias.name in {"torch", "shared_raw_generation_helper"} for alias in node.names):
                failures.append(f"FORBIDDEN_IMPORT:{rel_path}")
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module.startswith("run_stable_loop_phase_lock_") or module == "shared_raw_generation_helper":
                    failures.append(f"FORBIDDEN_IMPORT:{rel_path}:{module}")
            if isinstance(node, ast.Call):
                name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ""
                if name in {"raw_generate", "load_checkpoint", "train", "fit", "backward", "step", "forward"}:
                    failures.append(f"FORBIDDEN_CALL:{rel_path}:{name}")


def require_config(root: Path, failures: list[str]) -> None:
    config = load_json(root / "analysis_config.json")
    expected_false = [
        "helper_repair_implemented",
        "new_helper_generation_run",
        "shared_helper_imported",
        "shared_helper_called",
        "training_performed",
        "torch_forward_pass_run",
        "checkpoint_mutated",
        "helper_backend_modified",
        "request_key_change_allowed",
        "runtime_surface_mutated",
        "product_surface_mutated",
        "release_surface_mutated",
    ]
    if config.get("planning_only") is not True or config.get("artifact_only") is not True:
        failures.append("CONFIG_NOT_PLANNING_ONLY")
    for key in expected_false:
        if config.get(key) is not False:
            failures.append(f"CONFIG_BOUNDARY_NOT_FALSE:{key}:{config.get(key)}")
    require_false_flags(config, failures, "config")


def require_upstream(root: Path, failures: list[str]) -> None:
    upstream = load_json(root / "upstream_143r_manifest.json")
    decision = upstream.get("decision", {})
    trace = upstream.get("duplicate_conflict_trace_report", {})
    helper = upstream.get("helper_duplicate_marker_semantics_audit", {})
    if decision.get("decision") != "duplicate_selected_marker_conflict_analysis_complete":
        failures.append(f"BAD_143R_DECISION:{decision.get('decision')}")
    if decision.get("root_cause_id") != "selected_marker_duplicate_conflict_uses_first_occurrence_without_occurrence_count_policy":
        failures.append(f"BAD_143R_ROOT:{decision.get('root_cause_id')}")
    if decision.get("next") != "143U_DUPLICATE_SELECTED_MARKER_CONFLICT_REJECTION_HELPER_PRIMITIVE_PLAN":
        failures.append(f"BAD_143R_NEXT:{decision.get('next')}")
    if decision.get("recommended_policy") != "selected_marker_occurrence_count_must_equal_one":
        failures.append(f"BAD_143R_POLICY:{decision.get('recommended_policy')}")
    if decision.get("target_repair_prototype") != "143V_SELECTED_MARKER_OCCURRENCE_COUNT_REJECTION_PROTOTYPE":
        failures.append(f"BAD_143R_TARGET:{decision.get('target_repair_prototype')}")
    for key, expected in {
        "generated_equals_first_duplicate_value_rate": 1.0,
        "generated_equals_last_duplicate_value_rate": 0.0,
        "generated_equals_fallback_rate": 0.0,
    }.items():
        if trace.get(key) != expected:
            failures.append(f"BAD_143R_TRACE:{key}:{trace.get(key)}")
    for key, expected in {
        "prompt_find_selected_marker_found": True,
        "selected_marker_first_occurrence_offset_used": True,
        "selected_marker_count_variable_found": False,
        "fallback_on_duplicate_conflict_found": False,
    }.items():
        if helper.get(key) != expected:
            failures.append(f"BAD_143R_HELPER_AUDIT:{key}:{helper.get(key)}")


def require_requirements(root: Path, failures: list[str]) -> None:
    requirements = load_json(root / "primitive_repair_requirements.json")
    anti = load_json(root / "anti_oracle_requirements.json")
    if requirements.get("selected_option") != "selected_marker_occurrence_count_must_equal_one":
        failures.append(f"BAD_REQUIREMENT_OPTION:{requirements.get('selected_option')}")
    for key in [
        "planning_only",
        "must_not_modify_shared_helper",
        "must_not_call_helper_generation",
        "candidate_line_counting_required",
        "count_only_actual_selected_marker_candidate_lines",
        "do_not_count_selected_marker_mentions_in_prose_or_instructions",
        "policy_applies_only_to_selected_marker",
        "non_selected_marker_duplicates_out_of_scope",
        "same_value_duplicate_acceptance_deferred",
    ]:
        if requirements.get(key) is not True:
            failures.append(f"REQUIREMENT_NOT_TRUE:{key}:{requirements.get(key)}")
    if requirements.get("recommended_occurrence_count_method") != "line.strip().startswith(selected_marker)":
        failures.append(f"BAD_OCCURRENCE_METHOD:{requirements.get('recommended_occurrence_count_method')}")
    for key in [
        "forbid_request_key_changes",
        "forbid_per_row_selected_pocket_metadata",
        "forbid_per_row_manifest_switching",
        "forbid_payload_marker_list_narrowed_to_correct_pocket",
        "forbid_hidden_final_winner_value_gold_answer_markers",
        "forbid_post_generation_repair",
        "forbid_broad_capability_claims",
    ]:
        if anti.get(key) is not True:
            failures.append(f"ANTI_ORACLE_NOT_TRUE:{key}:{anti.get(key)}")


def require_policy_and_options(root: Path, failures: list[str]) -> None:
    policy = load_json(root / "selected_marker_occurrence_policy_matrix.json")
    options = load_json(root / "repair_options_matrix.json")
    recommendation = load_json(root / "selected_repair_recommendation.json")
    if policy.get("recommended_policy") != "selected_marker_occurrence_count_must_equal_one":
        failures.append(f"BAD_POLICY:{policy.get('recommended_policy')}")
    if policy.get("candidate_line_definition") != "An actual candidate marker line is counted only when line.strip().startswith(selected_marker).":
        failures.append("BAD_CANDIDATE_LINE_DEFINITION")
    for key in [
        "selected_marker_occurrence_policy_applies_only_to_the_selected_marker",
        "non_selected_marker_duplicates_out_of_scope",
        "same_value_duplicate_acceptance_deferred",
    ]:
        if policy.get(key) is not True:
            failures.append(f"POLICY_SCOPE_BAD:{key}:{policy.get(key)}")
    rows = {row.get("case_id"): row.get("expected_behavior") for row in policy.get("rows", [])}
    expected_rows = {
        "zero_selected_marker_candidate_lines": "fallback",
        "one_selected_marker_candidate_line": "extract_value_from_that_line",
        "two_or_more_selected_marker_candidate_lines_same_value": "fallback_for_now",
        "two_or_more_selected_marker_candidate_lines_conflicting_values": "fallback",
    }
    if rows != expected_rows:
        failures.append(f"BAD_POLICY_ROWS:{rows}")
    if options.get("selected_option") != "selected_marker_occurrence_count_must_equal_one":
        failures.append(f"BAD_OPTIONS_SELECTION:{options.get('selected_option')}")
    option_ids = {option.get("option_id") for option in options.get("options", [])}
    expected_options = {
        "selected_marker_occurrence_count_must_equal_one",
        "duplicate_same_value_allowed_conflicting_rejected",
        "keep_first_occurrence_policy",
    }
    if option_ids != expected_options:
        failures.append(f"BAD_OPTIONS:{sorted(option_ids)}")
    if recommendation.get("selected_option") != "selected_marker_occurrence_count_must_equal_one":
        failures.append(f"BAD_RECOMMENDATION:{recommendation.get('selected_option')}")
    if recommendation.get("next") != "143V_SELECTED_MARKER_OCCURRENCE_COUNT_REJECTION_PROTOTYPE":
        failures.append(f"BAD_RECOMMENDATION_NEXT:{recommendation.get('next')}")
    if recommendation.get("same_value_duplicate_acceptance_claimed") is not False:
        failures.append("SAME_VALUE_DUPLICATE_ACCEPTANCE_CLAIMED")


def require_target_143v(root: Path, failures: list[str]) -> None:
    plan = load_json(root / "target_143v_milestone_plan.json")
    if plan.get("milestone") != "143V_SELECTED_MARKER_OCCURRENCE_COUNT_REJECTION_PROTOTYPE":
        failures.append(f"BAD_143V_MILESTONE:{plan.get('milestone')}")
    if plan.get("helper_changing_prototype") is not True:
        failures.append("143V_NOT_HELPER_CHANGING_PROTOTYPE")
    scope = plan.get("allowed_helper_change_scope", {})
    if scope.get("only_function_may_change") != "_instnct_select_rule_selected_pocket_value":
        failures.append(f"BAD_143V_HELPER_SCOPE:{scope.get('only_function_may_change')}")
    for key in [
        "request_validation_must_not_change",
        "allowed_request_keys_must_not_change",
        "forbidden_request_keys_must_not_loosen",
        "old_decoder_path_must_remain_unchanged",
        "non_instnct_raw_generation_path_must_remain_unchanged",
    ]:
        if scope.get(key) is not True:
            failures.append(f"143V_SCOPE_NOT_TRUE:{key}:{scope.get(key)}")
    occurrence = plan.get("selected_marker_occurrence_counting_policy", {})
    if occurrence.get("method") != "line.strip().startswith(selected_marker)":
        failures.append(f"BAD_143V_OCCURRENCE_METHOD:{occurrence.get('method')}")
    for key in [
        "count_only_actual_selected_marker_candidate_lines",
        "do_not_count_selected_marker_mentions_in_prose_or_instructions",
        "policy_applies_only_to_selected_marker",
    ]:
        if occurrence.get(key) is not True:
            failures.append(f"143V_OCCURRENCE_SCOPE_BAD:{key}:{occurrence.get(key)}")
    required_controls = {
        "DUPLICATE_SELECTED_MARKER_CONFLICT_CONTROL",
        "DUPLICATE_SELECTED_MARKER_SAME_VALUE_CONTROL",
        "DUPLICATE_NON_SELECTED_MARKER_SCOPE_CONTROL",
        "SELECTED_MARKER_MENTION_IN_PROSE_CONTROL",
        "ZERO_SELECTED_MARKER_FALLBACK_CONTROL",
        "SINGLE_SELECTED_MARKER_POSITIVE_CONTROL",
        "SELECTED_MARKER_VALUE_MISSING_CONTROL",
        "WINNER_LABEL_MISSING_AMBIGUOUS_CONTROL",
        "POCKET_MARKER_ORDER_PERMUTATION_CONTROL",
        "LEGACY_MANIFEST_REGRESSION_CONTROL",
        "STATIC_MANIFEST_INTEGRITY_CONTROL",
        "HELPER_REQUEST_AUDIT_CONTROL",
    }
    if set(plan.get("required_controls", [])) != required_controls:
        failures.append("BAD_143V_REQUIRED_CONTROLS")
    gates = plan.get("positive_gates", {})
    for key in [
        "single_selected_marker_binding_accuracy",
        "selected_marker_line_occurrence_count_accuracy",
        "selected_marker_prose_mention_false_positive_rate",
        "duplicate_selected_marker_conflict_rejection_rate",
        "duplicate_selected_marker_same_value_rejection_rate",
        "duplicate_non_selected_marker_binding_accuracy",
        "duplicate_non_selected_marker_regression_rate",
        "zero_selected_marker_fallback_rate",
        "selected_marker_value_missing_fallback_rate",
        "pocket_marker_order_permutation_accuracy",
        "per_row_manifest_switch_rate",
        "per_row_payload_marker_switch_rate",
        "helper_request_forbidden_metadata_count",
        "legacy_manifest_regression_passed",
        "deterministic_replay_passed",
    ]:
        if key not in gates:
            failures.append(f"143V_GATE_MISSING:{key}")
    routes = plan.get("clean_negative_routes", {})
    for key in [
        "duplicate_conflict_still_not_rejected",
        "overbroad_duplicate_rejection_detected",
        "single_marker_binding_regression",
        "helper_integrity_failure",
    ]:
        if key not in routes:
            failures.append(f"143V_ROUTE_MISSING:{key}")


def require_decision_and_boundary(root: Path, failures: list[str]) -> None:
    decision = load_json(root / "decision.json")
    summary = load_json(root / "summary.json")
    report = (root / "report.md").read_text(encoding="utf-8")
    if decision.get("decision") != "duplicate_selected_marker_conflict_rejection_primitive_plan_recommended":
        failures.append(f"BAD_DECISION:{decision.get('decision')}")
    if decision.get("selected_option") != "selected_marker_occurrence_count_must_equal_one":
        failures.append(f"BAD_SELECTED_OPTION:{decision.get('selected_option')}")
    if decision.get("next") != "143V_SELECTED_MARKER_OCCURRENCE_COUNT_REJECTION_PROTOTYPE":
        failures.append(f"BAD_NEXT:{decision.get('next')}")
    if decision.get("helper_repair_implemented") is not False or decision.get("positive_capability_claimed") is not False:
        failures.append("DECISION_OVERCLAIM")
    for payload_name, payload in [("decision", decision), ("summary", summary)]:
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
    ast_scan(failures)
    require_artifacts(root, failures)
    if failures:
        return failures
    require_config(root, failures)
    require_upstream(root, failures)
    require_requirements(root, failures)
    require_policy_and_options(root, failures)
    require_target_143v(root, failures)
    require_decision_and_boundary(root, failures)
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Check 143U duplicate selected marker conflict rejection primitive plan")
    parser.add_argument("--root", type=Path, default=SMOKE_ROOT)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--skip-changed-files-check", action="store_true")
    args = parser.parse_args()
    root = args.root if args.root.is_absolute() else (REPO_ROOT / args.root).resolve()
    failures = run_checks(root, check_changed_files=not args.skip_changed_files_check)
    if failures:
        print("143U CHECK FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("143U CHECK PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
