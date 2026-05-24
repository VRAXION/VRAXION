#!/usr/bin/env python3
"""Checker for 143R duplicate selected marker conflict analysis."""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_143r_duplicate_selected_marker_conflict_analysis/smoke"
RUNNER = "scripts/probes/run_stable_loop_phase_lock_143r_duplicate_selected_marker_conflict_analysis.py"
CHECKER = "scripts/probes/run_stable_loop_phase_lock_143r_duplicate_selected_marker_conflict_analysis_check.py"
DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_143R_DUPLICATE_SELECTED_MARKER_CONFLICT_ANALYSIS_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_143R_DUPLICATE_SELECTED_MARKER_CONFLICT_ANALYSIS_RESULT.md",
]
ALLOWED_MUTATIONS = {RUNNER, CHECKER, *DOCS}
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "analysis_config.json",
    "upstream_143p_manifest.json",
    "duplicate_conflict_trace_report.json",
    "duplicate_conflict_failure_mode_report.json",
    "helper_duplicate_marker_semantics_audit.json",
    "selected_marker_occurrence_policy_matrix.json",
    "alternative_hypothesis_matrix.json",
    "root_cause_report.json",
    "repair_options_matrix.json",
    "risk_register.json",
    "target_143u_milestone_plan.json",
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
    "artifact-only duplicate selected marker conflict analysis",
    "constrained helper/backend evidence",
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
    if config.get("artifact_only") is not True:
        failures.append("CONFIG_NOT_ARTIFACT_ONLY")
    for key in expected_false:
        if config.get(key) is not False:
            failures.append(f"CONFIG_BOUNDARY_NOT_FALSE:{key}:{config.get(key)}")
    require_false_flags(config, failures, "config")


def require_upstream(root: Path, failures: list[str]) -> None:
    upstream = load_json(root / "upstream_143p_manifest.json")
    decision = upstream.get("decision", {})
    metrics = upstream.get("aggregate_metrics", {})
    duplicate = upstream.get("duplicate_selected_marker_conflict_report", {})
    if decision.get("decision") != "duplicate_selected_marker_conflict_not_rejected":
        failures.append(f"BAD_143P_DECISION:{decision.get('decision')}")
    if decision.get("next") != "143R_DUPLICATE_SELECTED_MARKER_CONFLICT_ANALYSIS":
        failures.append(f"BAD_143P_NEXT:{decision.get('next')}")
    expected = {
        "winner_label_parse_accuracy": 1.0,
        "selected_pocket_to_marker_binding_accuracy": 1.0,
        "pocket_marker_order_permutation_accuracy": 1.0,
        "main_pocket_writeback_rate": 1.0,
        "duplicate_selected_marker_conflict_rejection_rate": 0.0,
        "duplicate_selected_marker_first_value_rate": 1.0,
        "duplicate_selected_marker_last_value_rate": 0.0,
        "legacy_manifest_regression_passed": True,
        "deterministic_replay_passed": True,
    }
    for key, expected_value in expected.items():
        if metrics.get(key) != expected_value:
            failures.append(f"BAD_143P_METRIC:{key}:{metrics.get(key)}")
    if duplicate.get("row_count", 0) <= 0:
        failures.append("BAD_DUPLICATE_ROW_COUNT")
    if duplicate.get("fallback_rate") != 0.0:
        failures.append(f"BAD_DUPLICATE_FALLBACK_RATE:{duplicate.get('fallback_rate')}")
    if duplicate.get("duplicate_selected_marker_first_value_rate") != 1.0:
        failures.append("BAD_DUPLICATE_FIRST_VALUE_RATE")
    if duplicate.get("duplicate_selected_marker_last_value_rate") != 0.0:
        failures.append("BAD_DUPLICATE_LAST_VALUE_RATE")


def require_trace_and_failure(root: Path, failures: list[str]) -> None:
    trace = load_json(root / "duplicate_conflict_trace_report.json")
    failure = load_json(root / "duplicate_conflict_failure_mode_report.json")
    expected = {
        "generated_equals_first_duplicate_value_rate": 1.0,
        "generated_equals_last_duplicate_value_rate": 0.0,
        "generated_equals_fallback_rate": 0.0,
        "generated_equals_unexpected_value_rate": 0.0,
    }
    if trace.get("duplicate_rows_count", 0) <= 0:
        failures.append("TRACE_DUPLICATE_ROWS_NOT_POSITIVE")
    if trace.get("selected_marker_occurrence_count_min", 0) < 2 or trace.get("selected_marker_occurrence_count_max", 0) < 2:
        failures.append("TRACE_OCCURRENCE_COUNTS_TOO_LOW")
    for key, expected_value in expected.items():
        if trace.get(key) != expected_value:
            failures.append(f"TRACE_METRIC_BAD:{key}:{trace.get(key)}")
    if trace.get("passed") is not True:
        failures.append("TRACE_NOT_PASSED")
    if failure.get("failure_mode_id") != "duplicate_selected_marker_conflict_first_occurrence_selected":
        failures.append(f"BAD_FAILURE_MODE:{failure.get('failure_mode_id')}")
    if failure.get("single_selected_marker_binding_still_works") is not True:
        failures.append("SINGLE_MARKER_BINDING_NOT_SUPPORTED")
    if failure.get("first_occurrence_selected") is not True:
        failures.append("FIRST_OCCURRENCE_NOT_CONFIRMED")


def require_helper_audit(root: Path, failures: list[str]) -> None:
    audit = load_json(root / "helper_duplicate_marker_semantics_audit.json")
    expected = {
        "selected_function_name": "_instnct_select_rule_selected_pocket_value",
        "function_found": True,
        "prompt_find_selected_marker_found": True,
        "selected_marker_first_occurrence_offset_used": True,
        "all_occurrence_scan_found": False,
        "selected_marker_count_variable_found": False,
        "duplicate_selected_marker_conflict_rejection_found": False,
        "fallback_on_duplicate_conflict_found": False,
        "selected_marker_missing_fallback_found": True,
        "selected_marker_value_missing_fallback_found": True,
        "winner_label_count_policy_found": True,
    }
    for key, expected_value in expected.items():
        if audit.get(key) != expected_value:
            failures.append(f"BAD_HELPER_AUDIT:{key}:{audit.get(key)}")
    if not audit.get("helper_source_sha256"):
        failures.append("HELPER_SHA_MISSING")


def require_hypotheses(root: Path, failures: list[str]) -> None:
    matrix = load_json(root / "alternative_hypothesis_matrix.json")
    rows = {row.get("hypothesis_id"): row for row in matrix.get("hypotheses", [])}
    expected = {
        "winner_label_parser_failure": "rejected",
        "static_marker_map_failure": "rejected",
        "marker_order_shortcut": "rejected",
        "request_metadata_oracle": "rejected",
        "per_row_manifest_switching": "rejected",
        "legacy_regression": "rejected",
        "selected_marker_missing_or_value_missing_gap": "rejected",
        "duplicate_selected_marker_first_occurrence_semantics": "supported",
    }
    for hypothesis_id, status in expected.items():
        row = rows.get(hypothesis_id)
        if not row:
            failures.append(f"MISSING_HYPOTHESIS:{hypothesis_id}")
            continue
        if row.get("status") != status:
            failures.append(f"BAD_HYPOTHESIS_STATUS:{hypothesis_id}:{row.get('status')}")
        if not row.get("evidence_artifacts") or not row.get("metrics") or not row.get("explanation"):
            failures.append(f"HYPOTHESIS_NOT_AUDITABLE:{hypothesis_id}")


def require_policy_and_root(root: Path, failures: list[str]) -> None:
    policy = load_json(root / "selected_marker_occurrence_policy_matrix.json")
    root_report = load_json(root / "root_cause_report.json")
    if policy.get("recommended_policy") != "selected_marker_occurrence_count_must_equal_one":
        failures.append(f"BAD_POLICY:{policy.get('recommended_policy')}")
    for key in [
        "selected_marker_occurrence_policy_applies_only_to_the_selected_marker",
        "non_selected_marker_duplicates_out_of_scope",
        "same_value_duplicate_acceptance_deferred",
    ]:
        if policy.get(key) is not True:
            failures.append(f"POLICY_SCOPE_BAD:{key}:{policy.get(key)}")
    rows = {row.get("case_id"): row.get("recommended_policy") for row in policy.get("rows", [])}
    expected_rows = {
        "zero_selected_marker_occurrences": "fallback",
        "one_selected_marker_occurrence": "extract_value",
        "two_or_more_selected_marker_occurrences_same_value": "fallback_for_now",
        "two_or_more_selected_marker_occurrences_conflicting_values": "fallback",
    }
    if rows != expected_rows:
        failures.append(f"BAD_POLICY_ROWS:{rows}")
    if root_report.get("root_cause_id") != "selected_marker_duplicate_conflict_uses_first_occurrence_without_occurrence_count_policy":
        failures.append(f"BAD_ROOT_CAUSE:{root_report.get('root_cause_id')}")
    for key in [
        "supported_by_143p_duplicate_conflict",
        "supported_by_helper_source_audit",
        "winner_label_parser_failure_rejected",
        "static_marker_map_failure_rejected",
        "metadata_oracle_rejected",
        "per_row_manifest_switching_rejected",
        "legacy_regression_rejected",
        "selected_marker_missing_or_value_missing_gap_rejected",
        "duplicate_first_occurrence_behavior_confirmed",
        "selected_marker_occurrence_count_policy_missing",
    ]:
        if root_report.get(key) is not True:
            failures.append(f"ROOT_EVIDENCE_BAD:{key}:{root_report.get(key)}")
    if root_report.get("architecture_failure_claimed") is not False or root_report.get("reasoning_failure_claimed") is not False:
        failures.append("ROOT_OVERCLAIM")


def require_options_and_target(root: Path, failures: list[str]) -> None:
    options = load_json(root / "repair_options_matrix.json")
    plan = load_json(root / "target_143u_milestone_plan.json")
    option_ids = {option.get("option_id") for option in options.get("options", [])}
    expected_options = {
        "selected_marker_occurrence_count_must_equal_one",
        "duplicate_same_value_allowed_conflicting_rejected",
        "keep_first_occurrence_policy",
    }
    if option_ids != expected_options:
        failures.append(f"BAD_REPAIR_OPTIONS:{sorted(option_ids)}")
    if options.get("recommended_option") != "selected_marker_occurrence_count_must_equal_one":
        failures.append(f"BAD_RECOMMENDED_OPTION:{options.get('recommended_option')}")
    for option in options.get("options", []):
        for key in ["mechanism", "scope_cost", "oracle_risk", "shortcut_risk", "helper_change_required", "request_key_change_required", "recommendation"]:
            if key not in option:
                failures.append(f"OPTION_FIELD_MISSING:{option.get('option_id')}:{key}")
    if plan.get("milestone") != "143U_DUPLICATE_SELECTED_MARKER_CONFLICT_REJECTION_HELPER_PRIMITIVE_PLAN":
        failures.append(f"BAD_143U_MILESTONE:{plan.get('milestone')}")
    if plan.get("planning_only") is not True or plan.get("must_not_modify_shared_helper") is not True:
        failures.append("143U_NOT_PLANNING_ONLY")
    if plan.get("recommended_next_prototype") != "143V_SELECTED_MARKER_OCCURRENCE_COUNT_REJECTION_PROTOTYPE":
        failures.append(f"BAD_143U_PROTOTYPE:{plan.get('recommended_next_prototype')}")
    forbidden = " ".join(plan.get("must_forbid", []))
    for phrase in ["helper request key changes", "per-row selected_pocket_id metadata", "per-row manifest switching", "payload marker list narrowed", "hidden final/winner-value/gold/answer", "broad architecture claims"]:
        if phrase not in forbidden:
            failures.append(f"143U_FORBID_MISSING:{phrase}")


def require_decision_and_boundary(root: Path, failures: list[str]) -> None:
    decision = load_json(root / "decision.json")
    summary = load_json(root / "summary.json")
    report = (root / "report.md").read_text(encoding="utf-8")
    if decision.get("decision") != "duplicate_selected_marker_conflict_analysis_complete":
        failures.append(f"BAD_DECISION:{decision.get('decision')}")
    if decision.get("root_cause_id") != "selected_marker_duplicate_conflict_uses_first_occurrence_without_occurrence_count_policy":
        failures.append(f"BAD_DECISION_ROOT:{decision.get('root_cause_id')}")
    if decision.get("next") != "143U_DUPLICATE_SELECTED_MARKER_CONFLICT_REJECTION_HELPER_PRIMITIVE_PLAN":
        failures.append(f"BAD_NEXT:{decision.get('next')}")
    if decision.get("target_repair_prototype") != "143V_SELECTED_MARKER_OCCURRENCE_COUNT_REJECTION_PROTOTYPE":
        failures.append(f"BAD_TARGET_PROTOTYPE:{decision.get('target_repair_prototype')}")
    if decision.get("positive_capability_claimed") is not False:
        failures.append("POSITIVE_CAPABILITY_OVERCLAIM")
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
    require_trace_and_failure(root, failures)
    require_helper_audit(root, failures)
    require_hypotheses(root, failures)
    require_policy_and_root(root, failures)
    require_options_and_target(root, failures)
    require_decision_and_boundary(root, failures)
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Check 143R duplicate selected marker conflict analysis")
    parser.add_argument("--root", type=Path, default=SMOKE_ROOT)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--skip-changed-files-check", action="store_true")
    args = parser.parse_args()
    root = args.root if args.root.is_absolute() else (REPO_ROOT / args.root).resolve()
    failures = run_checks(root, check_changed_files=not args.skip_changed_files_check)
    if failures:
        print("143R CHECK FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("143R CHECK PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
