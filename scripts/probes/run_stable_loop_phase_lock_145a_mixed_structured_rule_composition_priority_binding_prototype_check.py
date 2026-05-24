#!/usr/bin/env python3
"""Checker for 145A mixed structured-rule composition priority binding prototype."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_145a_mixed_structured_rule_composition_priority_binding_prototype/smoke"
HELPER = "scripts/probes/shared_raw_generation_helper.py"
RUNNER = "scripts/probes/run_stable_loop_phase_lock_145a_mixed_structured_rule_composition_priority_binding_prototype.py"
CHECKER = "scripts/probes/run_stable_loop_phase_lock_145a_mixed_structured_rule_composition_priority_binding_prototype_check.py"
DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_145A_MIXED_STRUCTURED_RULE_COMPOSITION_PRIORITY_BINDING_PROTOTYPE_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_145A_MIXED_STRUCTURED_RULE_COMPOSITION_PRIORITY_BINDING_PROTOTYPE_RESULT.md",
]
ALLOWED_MUTATIONS = {HELPER, RUNNER, CHECKER, *DOCS}
ALLOWED_HELPER_KEYS = {"prompt", "checkpoint_path", "checkpoint_hash", "seed", "max_new_tokens", "generation_config"}
NEW_DECODER = "deterministic_pocket_gated_mixed_structured_rule_composition_binding_decoder"
OLD_SELECTED_DECODER = "deterministic_pocket_gated_rule_selected_pocket_binding_decoder"
OLD_STRUCTURED_DECODER = "deterministic_pocket_gated_structured_rule_metadata_binding_decoder"
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "analysis_config.json",
    "upstream_144z_manifest.json",
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
    return [line[3:].replace("\\", "/") for line in git_status().splitlines() if line.strip()]


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
        path = REPO_ROOT / rel_path
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if rel_path == CHECKER and isinstance(node, ast.Call):
                name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ""
                if name == "raw_generate":
                    failures.append("CHECKER_RAW_GENERATE_NOT_ALLOWED")
            if isinstance(node, ast.Call):
                name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ""
                if name in {"train", "fit", "backward", "step", "forward"}:
                    failures.append(f"TRAINING_CALL_NOT_ALLOWED:{rel_path}:{name}")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if rel_path == CHECKER and alias.name == "shared_raw_generation_helper":
                        failures.append("CHECKER_IMPORTS_SHARED_HELPER")


def require_static_files(failures: list[str]) -> None:
    for rel_path in [RUNNER, CHECKER, *DOCS]:
        path = REPO_ROOT / rel_path
        if not path.exists():
            failures.append(f"MISSING_TRACKED_FILE:{rel_path}")
            continue
        text = path.read_text(encoding="utf-8")
        for term in ["145A", NEW_DECODER, "mixed structured-rule composition"]:
            if term not in text:
                failures.append(f"TERM_MISSING:{rel_path}:{term}")
        for phrase in BOUNDARY_PHRASES:
            if phrase not in text:
                failures.append(f"BOUNDARY_TERM_MISSING:{rel_path}:{phrase}")
    helper_source = (REPO_ROOT / HELPER).read_text(encoding="utf-8")
    for term in [
        NEW_DECODER,
        OLD_SELECTED_DECODER,
        OLD_STRUCTURED_DECODER,
        "_instnct_select_mixed_structured_rule_composition_value",
        "_instnct_parse_mixed_rule_composition",
        "_instnct_parse_mixed_priority",
    ]:
        if term not in helper_source:
            failures.append(f"HELPER_TERM_MISSING:{term}")
    if "priority_pocket_oracle" not in helper_source or "nested_rule_block_before_block_end" not in helper_source:
        failures.append("HELPER_STRICT_POLICY_TERMS_MISSING")


def require_artifacts(root: Path, failures: list[str]) -> None:
    for name in REQUIRED_ARTIFACTS:
        if not (root / name).exists():
            failures.append(f"MISSING_ARTIFACT:{name}")


def require_upstream(root: Path, failures: list[str]) -> None:
    upstream = load_json(root / "upstream_144z_manifest.json")
    decision = upstream.get("decision", {})
    target = upstream.get("target_145a_milestone_plan", {})
    if decision.get("decision") != "structured_rule_composition_priority_binding_prototype_plan_recommended":
        failures.append(f"BAD_144Z_DECISION:{decision.get('decision')}")
    if decision.get("selected_option") != "mixed_structured_rule_composition_priority_binding_prototype":
        failures.append(f"BAD_144Z_OPTION:{decision.get('selected_option')}")
    if decision.get("next") != "145A_MIXED_STRUCTURED_RULE_COMPOSITION_PRIORITY_BINDING_PROTOTYPE":
        failures.append(f"BAD_144Z_NEXT:{decision.get('next')}")
    if target.get("decoder_name") != NEW_DECODER or target.get("implementation_ready") is not True:
        failures.append("144Z_TARGET_145A_NOT_IMPLEMENTATION_READY")
    if NEW_DECODER not in json.dumps(target):
        failures.append("144Z_TARGET_MISSING_NEW_DECODER")
    for decoder in [OLD_SELECTED_DECODER, OLD_STRUCTURED_DECODER]:
        if decoder not in json.dumps(target):
            failures.append(f"144Z_TARGET_OLD_DECODER_MISSING:{decoder}")


def require_helper_diff(root: Path, failures: list[str]) -> None:
    audit = load_json(root / "shared_helper_diff_audit.json")
    expected_true = [
        "source_changed",
        "new_mixed_decoder_string_present",
        "new_mixed_selection_function_present",
        "new_mixed_parser_helpers_present",
        "new_behavior_manifest_gated",
        "old_selected_pocket_decoder_still_present",
        "old_structured_metadata_decoder_still_present",
        "old_selected_pocket_binding_function_unchanged",
        "old_structured_metadata_function_unchanged",
        "validate_request_unchanged",
        "allowed_request_keys_unchanged",
        "forbidden_request_keys_not_loosened",
        "no_training_import_added",
        "no_network_or_io_added",
        "passed",
    ]
    for key in expected_true:
        if audit.get(key) is not True:
            failures.append(f"HELPER_DIFF_FAIL:{key}:{audit.get(key)}")
    current = (REPO_ROOT / HELPER).read_text(encoding="utf-8")
    current_hash = hashlib.sha256(current.encode("utf-8", errors="replace")).hexdigest()
    if audit.get("helper_source_sha256_after") != current_hash:
        failures.append("HELPER_SOURCE_DOES_NOT_MATCH_AUDIT_AFTER_HASH")


def require_audits(root: Path, failures: list[str]) -> None:
    request = load_json(root / "helper_request_audit.json")
    static = load_json(root / "static_manifest_integrity_report.json")
    prompt = load_json(root / "prompt_scanner_report.json")
    if set(request.get("allowed_helper_keys", [])) != ALLOWED_HELPER_KEYS:
        failures.append(f"BAD_ALLOWED_HELPER_KEYS:{request.get('allowed_helper_keys')}")
    for key in ["all_requests_allowed_keys_only", "selected_pocket_id_not_in_request_metadata", "winner_label_not_in_request_metadata", "passed"]:
        if request.get(key) is not True:
            failures.append(f"REQUEST_AUDIT_FAIL:{key}:{request.get(key)}")
    if request.get("helper_request_forbidden_metadata_count") != 0:
        failures.append(f"FORBIDDEN_METADATA:{request.get('helper_request_forbidden_metadata_count')}")
    if request.get("per_row_checkpoint_path_switch_rate") != 0.0 or request.get("per_row_checkpoint_hash_switch_rate") != 0.0:
        failures.append("CHECKPOINT_SWITCHING_DETECTED")
    if request.get("raw_generate_allowed_in_runner") is not True or request.get("raw_generate_allowed_in_checker") is not False:
        failures.append("RAW_GENERATE_BOUNDARY_BAD")
    if static.get("passed") is not True or static.get("per_row_manifest_switch_rate") != 0.0 or static.get("per_row_payload_marker_switch_rate") != 0.0:
        failures.append("STATIC_MANIFEST_INTEGRITY_FAIL")
    if static.get("payload_marker_list_narrowed_to_correct_pocket") is not False:
        failures.append("PAYLOAD_MARKER_LIST_NARROWED")
    if prompt.get("passed") is not True or prompt.get("forbidden_violation_count") != 0:
        failures.append("PROMPT_SCANNER_FAIL")
    if prompt.get("priority_pocket_oracle_control_rows_detected", 0) <= 0:
        failures.append("PRIORITY_POCKET_ORACLE_CONTROL_NOT_SCANNED")
    if prompt.get("priority_pocket_oracle_control_rows_rejected") != prompt.get("priority_pocket_oracle_control_rows_detected"):
        failures.append("PRIORITY_POCKET_ORACLE_CONTROL_NOT_REJECTED")


def require_reports(root: Path, failures: list[str]) -> None:
    for name in REQUIRED_ARTIFACTS:
        if name.endswith("_report.json") or name in {"shared_helper_diff_audit.json", "static_manifest_integrity_report.json", "prompt_scanner_report.json", "helper_request_audit.json"}:
            payload = load_json(root / name)
            if payload.get("passed") is not True:
                failures.append(f"REPORT_NOT_PASSED:{name}")


def require_metrics(root: Path, failures: list[str]) -> None:
    metrics = load_json(root / "aggregate_metrics.json")
    thresholds = {
        "mixed_rule_block_parse_accuracy": 0.90,
        "per_block_candidate_derivation_accuracy": 0.90,
        "priority_policy_parse_accuracy": 0.90,
        "final_selected_pocket_derivation_accuracy": 0.90,
        "selected_pocket_to_marker_binding_accuracy": 0.95,
        "same_line_value_extraction_accuracy": 0.95,
        "end_to_end_answer_accuracy": 0.90,
        "invalid_high_priority_fallthrough_accuracy": 0.90,
        "semantic_invalid_high_priority_fallthrough_accuracy": 0.90,
        "structural_invalid_prompt_fallback_rate": 0.90,
        "all_blocks_invalid_fallback_rate": 0.90,
        "missing_priority_fallback_rate": 0.90,
        "duplicate_priority_rejection_rate": 0.90,
        "unknown_priority_rejection_rate": 0.90,
        "missing_block_reference_rejection_rate": 0.90,
        "multiple_priority_lines_rejection_rate": 0.90,
        "duplicate_rule_block_type_rejection_rate": 0.90,
        "malformed_block_boundary_rejection_rate": 0.90,
        "metadata_outside_block_rejection_rate": 0.90,
        "nested_block_boundary_rejection_rate": 0.90,
        "empty_rule_block_rejection_rate": 0.90,
        "priority_pocket_oracle_rejection_rate": 0.90,
        "same_blocks_different_priority_accuracy": 0.90,
        "priority_only_changes_winner_accuracy": 0.90,
        "same_priority_different_block_values_accuracy": 0.90,
        "same_template_opposite_priority_winner_accuracy": 0.90,
    }
    for key, threshold in thresholds.items():
        if metrics.get(key, 0.0) < threshold:
            failures.append(f"METRIC_BELOW_GATE:{key}:{metrics.get(key)}")
    if metrics.get("distinct_block_candidate_coverage") is not True:
        failures.append("DISTINCT_BLOCK_CANDIDATE_COVERAGE_FAIL")
    if metrics.get("rule_composition_ablation_accuracy", 1.0) > 0.15:
        failures.append(f"RULE_COMPOSITION_ABLATION_TOO_HIGH:{metrics.get('rule_composition_ablation_accuracy')}")
    for key in ["helper_request_forbidden_metadata_count", "per_row_manifest_switch_rate", "per_row_payload_marker_switch_rate"]:
        if metrics.get(key) not in {0, 0.0}:
            failures.append(f"METRIC_NOT_ZERO:{key}:{metrics.get(key)}")
    for key in ["legacy_structured_rule_metadata_regression_passed", "legacy_selected_pocket_binding_regression_passed", "deterministic_replay_passed"]:
        if metrics.get(key) is not True:
            failures.append(f"METRIC_NOT_TRUE:{key}:{metrics.get(key)}")


def require_decision_and_boundary(root: Path, failures: list[str]) -> None:
    decision = load_json(root / "decision.json")
    summary = load_json(root / "summary.json")
    report = (root / "report.md").read_text(encoding="utf-8")
    if decision.get("decision") != "mixed_structured_rule_composition_priority_binding_prototype_positive":
        failures.append(f"BAD_DECISION:{decision.get('decision')}")
    if decision.get("verdict") != "INSTNCT_MIXED_STRUCTURED_RULE_COMPOSITION_PRIORITY_BINDING_PROTOTYPE_POSITIVE":
        failures.append(f"BAD_VERDICT:{decision.get('verdict')}")
    if decision.get("next") != "145H_MIXED_STRUCTURED_RULE_COMPOSITION_PRIORITY_BINDING_SCALE_CONFIRM":
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
    require_helper_diff(root, failures)
    require_audits(root, failures)
    require_reports(root, failures)
    require_metrics(root, failures)
    require_decision_and_boundary(root, failures)
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Check 145A mixed structured-rule composition prototype")
    parser.add_argument("--root", type=Path, default=SMOKE_ROOT)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--skip-changed-files-check", action="store_true")
    args = parser.parse_args()
    root = args.root if args.root.is_absolute() else (REPO_ROOT / args.root).resolve()
    failures = run_checks(root, check_changed_files=not args.skip_changed_files_check)
    if failures:
        print("145A CHECK FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("145A CHECK PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
