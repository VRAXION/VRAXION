#!/usr/bin/env python3
"""Checker for 145H mixed structured-rule composition priority binding scale confirm."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_145h_mixed_structured_rule_composition_priority_binding_scale_confirm/smoke"
HELPER = "scripts/probes/shared_raw_generation_helper.py"
RUNNER = "scripts/probes/run_stable_loop_phase_lock_145h_mixed_structured_rule_composition_priority_binding_scale_confirm.py"
CHECKER = "scripts/probes/run_stable_loop_phase_lock_145h_mixed_structured_rule_composition_priority_binding_scale_confirm_check.py"
DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_145H_MIXED_STRUCTURED_RULE_COMPOSITION_PRIORITY_BINDING_SCALE_CONFIRM_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_145H_MIXED_STRUCTURED_RULE_COMPOSITION_PRIORITY_BINDING_SCALE_CONFIRM_RESULT.md",
]
ALLOWED_MUTATIONS = {RUNNER, CHECKER, *DOCS}
NEW_DECODER = "deterministic_pocket_gated_mixed_structured_rule_composition_binding_decoder"
OLD_SELECTED_DECODER = "deterministic_pocket_gated_rule_selected_pocket_binding_decoder"
OLD_STRUCTURED_DECODER = "deterministic_pocket_gated_structured_rule_metadata_binding_decoder"
ALLOWED_HELPER_KEYS = {"prompt", "checkpoint_path", "checkpoint_hash", "seed", "max_new_tokens", "generation_config"}
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "analysis_config.json",
    "upstream_145a_manifest.json",
    "shared_helper_no_change_audit.json",
    "helper_mixed_rule_semantics_audit.json",
    "shared_helper_diff_audit.json",
    "mixed_rule_block_parser_report.json",
    "per_block_candidate_derivation_report.json",
    "priority_policy_report.json",
    "final_selected_pocket_derivation_report.json",
    "selected_pocket_binding_report.json",
    "same_line_value_extraction_report.json",
    "invalid_high_priority_fallthrough_report.json",
    "semantic_invalid_high_priority_fallthrough_report.json",
    "structural_invalid_prompt_fallback_report.json",
    "all_blocks_invalid_fallback_report.json",
    "missing_priority_report.json",
    "duplicate_priority_entry_report.json",
    "unknown_priority_entry_report.json",
    "missing_block_reference_report.json",
    "multiple_priority_lines_report.json",
    "duplicate_rule_block_type_report.json",
    "malformed_block_boundary_report.json",
    "metadata_outside_block_report.json",
    "nested_block_boundary_report.json",
    "empty_rule_block_report.json",
    "priority_pocket_oracle_report.json",
    "same_blocks_different_priority_report.json",
    "same_priority_different_block_values_report.json",
    "same_template_opposite_priority_winner_report.json",
    "rule_composition_ablation_report.json",
    "rule_composition_oracle_shortcut_report.json",
    "legacy_structured_rule_metadata_regression_report.json",
    "legacy_selected_pocket_binding_regression_report.json",
    "static_manifest_integrity_report.json",
    "prompt_scanner_report.json",
    "helper_request_audit.json",
    "positive_vs_fallback_denominator_report.json",
    "priority_order_coverage_report.json",
    "block_type_candidate_coverage_report.json",
    "per_seed_gate_report.json",
    "per_family_gate_report.json",
    "aggregate_metrics.json",
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
    "open_domain_assistant_readiness_claimed",
    "production_chat_claimed",
    "public_api_claimed",
    "deployment_readiness_claimed",
    "safety_alignment_claimed",
    "architecture_superiority_claimed",
]
BOUNDARY_PHRASES = [
    "constrained helper/backend evidence only",
    "mixed structured-rule composition with explicit priority over block types only",
    "not natural-language rule reasoning",
    "not open-ended arbitration",
    "not GPT-like/open-domain capability",
    "not production readiness",
    "not architecture superiority",
]


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


def ast_scan(failures: list[str]) -> None:
    for rel_path in [RUNNER, CHECKER]:
        tree = ast.parse((REPO_ROOT / rel_path).read_text(encoding="utf-8"), filename=rel_path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ""
                if rel_path == CHECKER and name == "raw_generate":
                    failures.append("CHECKER_RAW_GENERATE_NOT_ALLOWED")
                if name in {"train", "fit", "backward", "step", "forward"}:
                    failures.append(f"TRAINING_CALL_NOT_ALLOWED:{rel_path}:{name}")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "shared_raw_generation_helper":
                        failures.append(f"SHARED_HELPER_IMPORT_NOT_ALLOWED:{rel_path}")
                    if alias.name in {"torch", "tensorflow"}:
                        failures.append(f"TRAINING_IMPORT_NOT_ALLOWED:{rel_path}:{alias.name}")
            if isinstance(node, ast.ImportFrom):
                if node.module == "shared_raw_generation_helper":
                    failures.append(f"SHARED_HELPER_IMPORT_NOT_ALLOWED:{rel_path}")
                if node.module in {"torch", "tensorflow"}:
                    failures.append(f"TRAINING_IMPORT_NOT_ALLOWED:{rel_path}:{node.module}")


def require_static_files(failures: list[str]) -> None:
    for rel_path in [RUNNER, CHECKER, *DOCS]:
        path = REPO_ROOT / rel_path
        if not path.exists():
            failures.append(f"MISSING_TRACKED_FILE:{rel_path}")
            continue
        text = path.read_text(encoding="utf-8")
        for term in ["145H", NEW_DECODER, "mixed structured-rule composition"]:
            if term not in text:
                failures.append(f"TERM_MISSING:{rel_path}:{term}")
        for phrase in BOUNDARY_PHRASES:
            if phrase not in text:
                failures.append(f"BOUNDARY_TERM_MISSING:{rel_path}:{phrase}")
    helper_source = (REPO_ROOT / HELPER).read_text(encoding="utf-8")
    head_source = git_show_head(HELPER)
    if helper_source != head_source:
        failures.append("SHARED_HELPER_CHANGED_FROM_HEAD")
    for term in [
        NEW_DECODER,
        OLD_SELECTED_DECODER,
        OLD_STRUCTURED_DECODER,
        "_instnct_select_mixed_structured_rule_composition_value",
        "_instnct_parse_mixed_rule_composition",
        "_instnct_parse_mixed_priority",
        "priority_pocket_oracle",
        "nested_rule_block_before_block_end",
        "metadata_outside_block",
    ]:
        if term not in helper_source:
            failures.append(f"HELPER_TERM_MISSING:{term}")


def require_artifacts(root: Path, failures: list[str]) -> None:
    for name in REQUIRED_ARTIFACTS:
        if not (root / name).exists():
            failures.append(f"MISSING_ARTIFACT:{name}")


def require_upstream(root: Path, failures: list[str]) -> None:
    upstream = load_json(root / "upstream_145a_manifest.json")
    decision = upstream.get("decision", {})
    metrics = upstream.get("aggregate_metrics", {})
    if decision.get("decision") != "mixed_structured_rule_composition_priority_binding_prototype_positive":
        failures.append(f"BAD_145A_DECISION:{decision.get('decision')}")
    if decision.get("verdict") != "INSTNCT_MIXED_STRUCTURED_RULE_COMPOSITION_PRIORITY_BINDING_PROTOTYPE_POSITIVE":
        failures.append(f"BAD_145A_VERDICT:{decision.get('verdict')}")
    if decision.get("next") != "145H_MIXED_STRUCTURED_RULE_COMPOSITION_PRIORITY_BINDING_SCALE_CONFIRM":
        failures.append(f"BAD_145A_NEXT:{decision.get('next')}")
    exact_metrics = {
        "end_to_end_answer_accuracy": 1.0,
        "final_selected_pocket_derivation_accuracy": 1.0,
        "selected_pocket_to_marker_binding_accuracy": 1.0,
        "semantic_invalid_high_priority_fallthrough_accuracy": 1.0,
        "structural_invalid_prompt_fallback_rate": 1.0,
        "priority_pocket_oracle_rejection_rate": 1.0,
        "rule_composition_ablation_accuracy": 0.0,
        "distinct_block_candidate_coverage": True,
        "legacy_structured_rule_metadata_regression_passed": True,
        "legacy_selected_pocket_binding_regression_passed": True,
        "deterministic_replay_passed": True,
    }
    for key, expected in exact_metrics.items():
        if metrics.get(key) != expected:
            failures.append(f"BAD_145A_METRIC:{key}:{metrics.get(key)}")


def require_audits(root: Path, failures: list[str]) -> None:
    no_change = load_json(root / "shared_helper_no_change_audit.json")
    semantics = load_json(root / "helper_mixed_rule_semantics_audit.json")
    request = load_json(root / "helper_request_audit.json")
    static = load_json(root / "static_manifest_integrity_report.json")
    prompt = load_json(root / "prompt_scanner_report.json")
    current_hash = hashlib.sha256((REPO_ROOT / HELPER).read_text(encoding="utf-8", errors="replace").encode("utf-8", errors="replace")).hexdigest()
    if no_change.get("passed") is not True:
        failures.append("SHARED_HELPER_NO_CHANGE_AUDIT_FAIL")
    if no_change.get("current_shared_helper_sha256") != current_hash:
        failures.append("SHARED_HELPER_HASH_MISMATCH")
    if no_change.get("shared_helper_no_change_since_145a") is not True or no_change.get("shared_helper_modified_by_145h") is not False:
        failures.append("SHARED_HELPER_NO_CHANGE_FLAGS_BAD")
    if semantics.get("passed") is not True:
        failures.append("HELPER_MIXED_RULE_SEMANTICS_AUDIT_FAIL")
    if set(request.get("allowed_helper_keys", [])) != ALLOWED_HELPER_KEYS:
        failures.append(f"BAD_ALLOWED_HELPER_KEYS:{request.get('allowed_helper_keys')}")
    for key in ["all_requests_allowed_keys_only", "selected_pocket_id_not_in_request_metadata", "winner_label_not_in_request_metadata", "passed"]:
        if request.get(key) is not True:
            failures.append(f"REQUEST_AUDIT_FAIL:{key}:{request.get(key)}")
    if request.get("helper_request_forbidden_metadata_count") != 0:
        failures.append(f"FORBIDDEN_METADATA:{request.get('helper_request_forbidden_metadata_count')}")
    if request.get("per_row_checkpoint_path_switch_rate") != 0.0 or request.get("per_row_checkpoint_hash_switch_rate") != 0.0:
        failures.append("CHECKPOINT_SWITCHING_DETECTED")
    if static.get("passed") is not True or static.get("per_row_manifest_switch_rate") != 0.0 or static.get("per_row_payload_marker_switch_rate") != 0.0:
        failures.append("STATIC_MANIFEST_INTEGRITY_FAIL")
    if static.get("payload_marker_list_narrowed_to_correct_pocket") is not False:
        failures.append("PAYLOAD_MARKER_LIST_NARROWED")
    if prompt.get("passed") is not True or prompt.get("forbidden_violation_count") != 0:
        failures.append("PROMPT_SCANNER_FAIL")
    if prompt.get("priority_pocket_oracle_control_rows_rejected") != prompt.get("priority_pocket_oracle_control_rows_detected"):
        failures.append("PRIORITY_POCKET_ORACLE_CONTROL_NOT_REJECTED")


def require_reports(root: Path, failures: list[str]) -> None:
    report_names = [
        name
        for name in REQUIRED_ARTIFACTS
        if name.endswith("_report.json") or name.endswith("_audit.json") or name in {"helper_request_audit.json", "static_manifest_integrity_report.json", "prompt_scanner_report.json"}
    ]
    for name in report_names:
        payload = load_json(root / name)
        if payload.get("passed") is not True:
            failures.append(f"REPORT_NOT_PASSED:{name}")


def require_denominator_and_coverage(root: Path, failures: list[str]) -> None:
    denominator = load_json(root / "positive_vs_fallback_denominator_report.json")
    priority = load_json(root / "priority_order_coverage_report.json")
    block = load_json(root / "block_type_candidate_coverage_report.json")
    if denominator.get("passed") is not True:
        failures.append("DENOMINATOR_REPORT_FAIL")
    if denominator.get("fallback_rows_in_positive_denominator") != 0:
        failures.append("END_TO_END_DENOMINATOR_INCLUDES_FALLBACK_ROWS")
    if denominator.get("expected_fallback_rows_excluded_from_end_to_end_answer_accuracy") is not True:
        failures.append("FALLBACK_ROWS_NOT_EXCLUDED_FROM_POSITIVE_DENOMINATOR")
    for key in ["positive_composition_subset_accuracy", "fallback_control_subset_accuracy"]:
        if denominator.get(key, 0.0) < 0.98:
            failures.append(f"DENOMINATOR_METRIC_BELOW_GATE:{key}:{denominator.get(key)}")
    for payload, name in [(priority, "PRIORITY_COVERAGE"), (block, "BLOCK_CANDIDATE_COVERAGE")]:
        if payload.get("passed") is not True:
            failures.append(f"{name}_REPORT_FAIL")
    if priority.get("all_priority_orders_covered") is not True:
        failures.append("ALL_PRIORITY_ORDERS_NOT_COVERED")
    if priority.get("all_block_types_win_under_priority") is not True:
        failures.append("ALL_BLOCK_TYPES_DO_NOT_WIN")
    if priority.get("all_pockets_win_under_each_supported_block_type") is not True:
        failures.append("ALL_POCKETS_DO_NOT_WIN_UNDER_EACH_BLOCK_TYPE")
    if block.get("distinct_block_candidate_coverage") is not True:
        failures.append("DISTINCT_BLOCK_CANDIDATE_COVERAGE_FAIL")


def require_metrics(root: Path, failures: list[str]) -> None:
    metrics = load_json(root / "aggregate_metrics.json")
    thresholds = {
        "mixed_rule_block_parse_accuracy": 0.98,
        "per_block_candidate_derivation_accuracy": 0.98,
        "priority_policy_parse_accuracy": 0.98,
        "final_selected_pocket_derivation_accuracy": 0.98,
        "selected_pocket_to_marker_binding_accuracy": 0.98,
        "same_line_value_extraction_accuracy": 0.98,
        "end_to_end_answer_accuracy": 0.98,
        "positive_composition_subset_accuracy": 0.98,
        "fallback_control_subset_accuracy": 0.98,
        "invalid_high_priority_fallthrough_accuracy": 0.98,
        "semantic_invalid_high_priority_fallthrough_accuracy": 0.98,
        "structural_invalid_prompt_fallback_rate": 0.98,
        "all_blocks_invalid_fallback_rate": 0.98,
        "missing_priority_fallback_rate": 0.98,
        "duplicate_priority_rejection_rate": 0.98,
        "unknown_priority_rejection_rate": 0.98,
        "missing_block_reference_rejection_rate": 0.98,
        "multiple_priority_lines_rejection_rate": 0.98,
        "duplicate_rule_block_type_rejection_rate": 0.98,
        "malformed_block_boundary_rejection_rate": 0.98,
        "metadata_outside_block_rejection_rate": 0.98,
        "nested_block_boundary_rejection_rate": 0.98,
        "empty_rule_block_rejection_rate": 0.98,
        "priority_pocket_oracle_rejection_rate": 0.98,
        "same_blocks_different_priority_accuracy": 0.98,
        "priority_only_changes_winner_accuracy": 0.98,
        "same_priority_different_block_values_accuracy": 0.98,
        "same_template_opposite_priority_winner_accuracy": 0.98,
    }
    for key, threshold in thresholds.items():
        if metrics.get(key, 0.0) < threshold:
            failures.append(f"METRIC_BELOW_GATE:{key}:{metrics.get(key)}")
    boolean_true = [
        "all_priority_orders_covered",
        "all_block_types_win_under_priority",
        "all_pockets_win_under_each_supported_block_type",
        "distinct_block_candidate_coverage",
        "shared_helper_no_change_since_145a",
        "legacy_structured_rule_metadata_regression_passed",
        "legacy_selected_pocket_binding_regression_passed",
        "deterministic_replay_passed",
    ]
    for key in boolean_true:
        if metrics.get(key) is not True:
            failures.append(f"METRIC_NOT_TRUE:{key}:{metrics.get(key)}")
    if metrics.get("rule_composition_ablation_accuracy", 1.0) > 0.05:
        failures.append(f"RULE_COMPOSITION_ABLATION_TOO_HIGH:{metrics.get('rule_composition_ablation_accuracy')}")
    for key in ["helper_request_forbidden_metadata_count", "per_row_manifest_switch_rate", "per_row_payload_marker_switch_rate"]:
        if metrics.get(key) not in {0, 0.0}:
            failures.append(f"METRIC_NOT_ZERO:{key}:{metrics.get(key)}")
    if metrics.get("main_eval_rows") != 9600:
        failures.append(f"BAD_MAIN_EVAL_ROWS:{metrics.get('main_eval_rows')}")


def require_decision_and_boundary(root: Path, failures: list[str]) -> None:
    decision = load_json(root / "decision.json")
    summary = load_json(root / "summary.json")
    report = (root / "report.md").read_text(encoding="utf-8")
    if decision.get("decision") != "mixed_structured_rule_composition_priority_binding_scale_confirmed":
        failures.append(f"BAD_DECISION:{decision.get('decision')}")
    if decision.get("verdict") != "INSTNCT_MIXED_STRUCTURED_RULE_COMPOSITION_PRIORITY_BINDING_SCALE_CONFIRMED":
        failures.append(f"BAD_VERDICT:{decision.get('verdict')}")
    if decision.get("next") != "145Z_MIXED_STRUCTURED_RULE_COMPOSITION_NEXT_DECISION_PLAN":
        failures.append(f"BAD_NEXT:{decision.get('next')}")
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
    require_static_files(failures)
    ast_scan(failures)
    require_artifacts(root, failures)
    if failures:
        return failures
    require_upstream(root, failures)
    require_audits(root, failures)
    require_reports(root, failures)
    require_denominator_and_coverage(root, failures)
    require_metrics(root, failures)
    require_decision_and_boundary(root, failures)
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Check 145H mixed structured-rule composition scale confirm")
    parser.add_argument("--root", type=Path, default=SMOKE_ROOT)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--skip-changed-files-check", action="store_true")
    args = parser.parse_args()
    root = args.root if args.root.is_absolute() else (REPO_ROOT / args.root).resolve()
    failures = run_checks(root, check_changed_files=not args.skip_changed_files_check)
    if failures:
        print("145H CHECK FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("145H CHECK PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
