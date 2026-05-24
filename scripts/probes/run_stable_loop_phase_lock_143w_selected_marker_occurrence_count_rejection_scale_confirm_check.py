#!/usr/bin/env python3
"""Checker for 143W selected marker occurrence count rejection scale confirm."""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_143w_selected_marker_occurrence_count_rejection_scale_confirm/smoke"
HELPER = "scripts/probes/shared_raw_generation_helper.py"
RUNNER = "scripts/probes/run_stable_loop_phase_lock_143w_selected_marker_occurrence_count_rejection_scale_confirm.py"
CHECKER = "scripts/probes/run_stable_loop_phase_lock_143w_selected_marker_occurrence_count_rejection_scale_confirm_check.py"
DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_143W_SELECTED_MARKER_OCCURRENCE_COUNT_REJECTION_SCALE_CONFIRM_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_143W_SELECTED_MARKER_OCCURRENCE_COUNT_REJECTION_SCALE_CONFIRM_RESULT.md",
]
ALLOWED_MUTATIONS = {RUNNER, CHECKER, *DOCS}
ALLOWED_HELPER_KEYS = {"prompt", "checkpoint_path", "checkpoint_hash", "seed", "max_new_tokens", "generation_config"}
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "upstream_143v_manifest.json",
    "analysis_config.json",
    "shared_helper_no_change_audit.json",
    "helper_repair_semantics_audit.json",
    "selected_marker_occurrence_count_report.json",
    "selected_marker_candidate_line_parser_report.json",
    "duplicate_selected_marker_conflict_report.json",
    "duplicate_selected_marker_same_value_report.json",
    "duplicate_non_selected_marker_scope_report.json",
    "duplicate_non_selected_marker_conflict_report.json",
    "selected_marker_prose_mention_report.json",
    "selected_marker_prose_line_start_report.json",
    "selected_marker_prose_plus_one_valid_line_report.json",
    "selected_marker_invalid_value_report.json",
    "selected_marker_multi_value_same_line_report.json",
    "following_line_value_leak_report.json",
    "per_seed_gate_report.json",
    "per_family_gate_report.json",
    "legacy_manifest_regression_report.json",
    "static_manifest_integrity_report.json",
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
            if rel_path == CHECKER and isinstance(node, ast.Call):
                name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ""
                if name == "raw_generate":
                    failures.append("CHECKER_RAW_GENERATE_NOT_ALLOWED")
            if isinstance(node, ast.Call):
                name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ""
                if name in {"train", "fit", "backward", "step", "forward"}:
                    failures.append(f"TRAINING_CALL_NOT_ALLOWED:{rel_path}:{name}")


def require_upstream(root: Path, failures: list[str]) -> None:
    upstream = load_json(root / "upstream_143v_manifest.json")
    decision = upstream.get("decision", {})
    metrics = upstream.get("aggregate_metrics", {})
    if decision.get("decision") != "selected_marker_occurrence_count_rejection_prototype_positive":
        failures.append(f"BAD_143V_DECISION:{decision.get('decision')}")
    if decision.get("next") != "143W_SELECTED_MARKER_OCCURRENCE_COUNT_REJECTION_SCALE_CONFIRM":
        failures.append(f"BAD_143V_NEXT:{decision.get('next')}")
    for key, expected in {
        "single_selected_marker_binding_accuracy": 1.0,
        "duplicate_selected_marker_conflict_rejection_rate": 1.0,
        "duplicate_selected_marker_same_value_rejection_rate": 1.0,
        "duplicate_non_selected_marker_conflict_binding_accuracy": 1.0,
        "selected_marker_candidate_line_parse_accuracy": 1.0,
        "selected_marker_prose_line_false_positive_rate": 0.0,
        "following_line_value_leak_rate": 0.0,
        "legacy_manifest_regression_passed": True,
        "deterministic_replay_passed": True,
    }.items():
        if metrics.get(key) != expected:
            failures.append(f"UPSTREAM_143V_METRIC_MISMATCH:{key}:{metrics.get(key)}")


def require_helper_no_change(root: Path, failures: list[str]) -> None:
    audit = load_json(root / "shared_helper_no_change_audit.json")
    expected_true = [
        "shared_helper_no_change_since_143v",
        "shared_raw_generation_helper_unchanged_from_head",
        "new_decoder_string_present",
    ]
    for key in expected_true:
        if audit.get(key) is not True:
            failures.append(f"HELPER_NO_CHANGE_FAIL:{key}:{audit.get(key)}")
    if audit.get("shared_helper_modified_by_143w") is not False:
        failures.append(f"HELPER_MODIFIED_BY_143W:{audit.get('shared_helper_modified_by_143w')}")
    helper_source = (REPO_ROOT / HELPER).read_text(encoding="utf-8")
    head = subprocess.run(["git", "show", f"HEAD:{HELPER}"], cwd=REPO_ROOT, text=True, capture_output=True, check=False).stdout
    if helper_source != head:
        failures.append("HELPER_CHANGED_FROM_HEAD")
    if "prompt.find(selected_marker)" in helper_source:
        failures.append("HELPER_STILL_USES_PROMPT_FIND_SELECTED_MARKER")
    if "candidate_line_re" not in helper_source or "re.escape(selected_marker)" not in helper_source:
        failures.append("HELPER_CANDIDATE_LINE_REGEX_MISSING")


def require_helper_semantics(root: Path, failures: list[str]) -> None:
    audit = load_json(root / "helper_repair_semantics_audit.json")
    for key in [
        "function_found",
        "candidate_line_re_found",
        "prompt_splitlines_used",
        "prompt_find_selected_marker_not_used",
        "same_line_extraction_used",
        "fallback_if_candidate_line_count_not_one",
        "fallback_on_selected_marker_value_missing",
        "old_decoder_path_still_exists",
        "new_behavior_manifest_gated",
        "passed",
    ]:
        if audit.get(key) is not True:
            failures.append(f"HELPER_REPAIR_SEMANTICS_FAIL:{key}:{audit.get(key)}")


def require_audits(root: Path, failures: list[str]) -> None:
    request = load_json(root / "helper_request_audit.json")
    static = load_json(root / "static_manifest_integrity_report.json")
    legacy = load_json(root / "legacy_manifest_regression_report.json")
    if request.get("helper_request_forbidden_metadata_count") != 0:
        failures.append(f"FORBIDDEN_METADATA:{request.get('helper_request_forbidden_metadata_count')}")
    if set(request.get("allowed_helper_keys", [])) != ALLOWED_HELPER_KEYS:
        failures.append(f"BAD_ALLOWED_HELPER_KEYS:{request.get('allowed_helper_keys')}")
    for key in ["all_requests_allowed_keys_only", "selected_pocket_id_not_in_request_metadata", "winner_label_not_in_request_metadata"]:
        if request.get(key) is not True:
            failures.append(f"REQUEST_AUDIT_FAIL:{key}:{request.get(key)}")
    if request.get("raw_generate_allowed_in_runner") is not True or request.get("raw_generate_allowed_in_checker") is not False:
        failures.append("RAW_GENERATE_BOUNDARY_BAD")
    if static.get("passed") is not True or static.get("per_row_manifest_switch_rate") != 0.0 or static.get("per_row_payload_marker_switch_rate") != 0.0:
        failures.append("STATIC_MANIFEST_INTEGRITY_FAIL")
    for key, expected in {
        "legacy_final_markers_present_accuracy": 1.0,
        "legacy_no_resolved_final_markers_fallback_rate": 1.0,
        "legacy_abc_static_first_marker_rate": 1.0,
        "legacy_old_decoder_binding_activation_rate": 0.0,
        "legacy_manifest_regression_passed": True,
    }.items():
        if legacy.get(key) != expected:
            failures.append(f"LEGACY_FAIL:{key}:{legacy.get(key)}")


def require_reports(root: Path, failures: list[str]) -> None:
    for name in [
        "selected_marker_occurrence_count_report.json",
        "selected_marker_candidate_line_parser_report.json",
        "duplicate_selected_marker_conflict_report.json",
        "duplicate_selected_marker_same_value_report.json",
        "duplicate_non_selected_marker_scope_report.json",
        "duplicate_non_selected_marker_conflict_report.json",
        "selected_marker_prose_mention_report.json",
        "selected_marker_prose_line_start_report.json",
        "selected_marker_prose_plus_one_valid_line_report.json",
        "selected_marker_invalid_value_report.json",
        "selected_marker_multi_value_same_line_report.json",
        "following_line_value_leak_report.json",
    ]:
        report = load_json(root / name)
        if report.get("passed") is not True:
            failures.append(f"REPORT_NOT_PASSED:{name}")
    for gate_name in ["per_seed_gate_report.json", "per_family_gate_report.json"]:
        gate = load_json(root / gate_name)
        if gate.get("passed") is not True:
            failures.append(f"GATE_REPORT_NOT_PASSED:{gate_name}")


def require_metrics(root: Path, failures: list[str]) -> None:
    metrics = load_json(root / "aggregate_metrics.json")
    for key, threshold in {
        "single_selected_marker_binding_accuracy": 0.98,
        "positive_binding_subset_writeback_rate": 0.98,
        "selected_marker_line_occurrence_count_accuracy": 0.98,
        "selected_marker_candidate_line_parse_accuracy": 0.98,
        "selected_marker_prose_plus_one_valid_line_accuracy": 0.98,
        "selected_marker_invalid_value_fallback_rate": 0.98,
        "selected_marker_multi_value_same_line_fallback_rate": 0.98,
        "duplicate_selected_marker_conflict_rejection_rate": 0.98,
        "duplicate_selected_marker_same_value_rejection_rate": 0.98,
        "duplicate_non_selected_marker_binding_accuracy": 0.98,
        "duplicate_non_selected_marker_conflict_binding_accuracy": 0.98,
        "zero_selected_marker_fallback_rate": 0.98,
        "selected_marker_value_missing_fallback_rate": 0.98,
        "pocket_marker_order_permutation_accuracy": 0.98,
    }.items():
        if metrics.get(key, 0.0) < threshold:
            failures.append(f"METRIC_BELOW_GATE:{key}:{metrics.get(key)}")
    for key in [
        "selected_marker_prose_mention_false_positive_rate",
        "selected_marker_prose_line_false_positive_rate",
        "following_line_value_leak_rate",
        "duplicate_non_selected_marker_regression_rate",
        "per_row_manifest_switch_rate",
        "per_row_payload_marker_switch_rate",
        "helper_request_forbidden_metadata_count",
    ]:
        if metrics.get(key) != 0.0 and metrics.get(key) != 0:
            failures.append(f"METRIC_NOT_ZERO:{key}:{metrics.get(key)}")
    if metrics.get("legacy_manifest_regression_passed") is not True or metrics.get("deterministic_replay_passed") is not True:
        failures.append("DETERMINISM_OR_LEGACY_FAIL")
    if metrics.get("shared_helper_no_change_since_143v") is not True:
        failures.append("SHARED_HELPER_NO_CHANGE_METRIC_FAIL")
    if metrics.get("closed_pocket_ablation_accuracy", 1.0) > 0.05:
        failures.append(f"CLOSED_POCKET_ABLATION_TOO_HIGH:{metrics.get('closed_pocket_ablation_accuracy')}")


def require_decision_and_boundary(root: Path, failures: list[str]) -> None:
    decision = load_json(root / "decision.json")
    summary = load_json(root / "summary.json")
    report = (root / "report.md").read_text(encoding="utf-8")
    if decision.get("decision") != "selected_marker_occurrence_count_rejection_scale_confirmed":
        failures.append(f"BAD_DECISION:{decision.get('decision')}")
    if decision.get("verdict") != "INSTNCT_SELECTED_MARKER_OCCURRENCE_COUNT_REJECTION_SCALE_CONFIRMED":
        failures.append(f"BAD_VERDICT:{decision.get('verdict')}")
    if decision.get("next") != "143Z_RULE_SELECTED_POCKET_BINDING_NEXT_DECISION_PLAN":
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
    ast_scan(failures)
    require_artifacts(root, failures)
    if failures:
        return failures
    require_upstream(root, failures)
    require_helper_no_change(root, failures)
    require_helper_semantics(root, failures)
    require_audits(root, failures)
    require_reports(root, failures)
    require_metrics(root, failures)
    require_decision_and_boundary(root, failures)
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Check 143W selected marker occurrence count rejection scale confirm")
    parser.add_argument("--root", type=Path, default=SMOKE_ROOT)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--skip-changed-files-check", action="store_true")
    args = parser.parse_args()
    root = args.root if args.root.is_absolute() else (REPO_ROOT / args.root).resolve()
    failures = run_checks(root, check_changed_files=not args.skip_changed_files_check)
    if failures:
        print("143W CHECK FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("143W CHECK PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
