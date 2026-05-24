#!/usr/bin/env python3
"""Checker for 143P rule-selected pocket payload binding scale confirm."""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_143p_rule_selected_pocket_payload_binding_scale_confirm/smoke"
RUNNER = "scripts/probes/run_stable_loop_phase_lock_143p_rule_selected_pocket_payload_binding_scale_confirm.py"
CHECKER = "scripts/probes/run_stable_loop_phase_lock_143p_rule_selected_pocket_payload_binding_scale_confirm_check.py"
DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_143P_RULE_SELECTED_POCKET_PAYLOAD_BINDING_SCALE_CONFIRM_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_143P_RULE_SELECTED_POCKET_PAYLOAD_BINDING_SCALE_CONFIRM_RESULT.md",
]
ALLOWED_MUTATIONS = {RUNNER, CHECKER, *DOCS}
ALLOWED_HELPER_KEYS = {"prompt", "checkpoint_path", "checkpoint_hash", "seed", "max_new_tokens", "generation_config"}
ACCEPTED_DECISIONS = {
    "rule_selected_pocket_payload_binding_scale_confirmed",
    "duplicate_selected_marker_conflict_not_rejected",
    "malformed_winner_label_not_rejected",
    "selected_marker_missing_not_rejected",
}
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "upstream_143k_manifest.json",
    "analysis_config.json",
    "duplicate_selected_marker_conflict_report.json",
    "malformed_winner_label_report.json",
    "duplicate_same_winner_label_report.json",
    "selected_marker_missing_report.json",
    "selected_marker_value_missing_report.json",
    "legacy_manifest_regression_report.json",
    "shared_helper_no_change_audit.json",
    "helper_request_audit.json",
    "static_manifest_integrity_report.json",
    "pocket_marker_order_permutation_report.json",
    "prompt_scanner_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
]
FALSE_FLAGS = [
    "reasoning_restored",
    "raw_assistant_capability_restored",
    "structured_tool_capability_restored",
    "gpt_like_readiness_claimed",
    "open_domain_assistant_readiness_claimed",
    "production_chat_claimed",
    "public_api_claimed",
    "deployment_readiness_claimed",
    "safety_alignment_claimed",
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
    for artifact in REQUIRED_ARTIFACTS:
        if not (root / artifact).exists():
            failures.append(f"MISSING_ARTIFACT:{artifact}")


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
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("run_stable_loop_phase_lock_"):
                failures.append(f"OLD_RUNNER_IMPORT:{rel_path}")


def require_upstream(root: Path, failures: list[str]) -> None:
    upstream = load_json(root / "upstream_143k_manifest.json")
    decision = upstream.get("decision", {})
    metrics = upstream.get("aggregate_metrics", {})
    if decision.get("decision") != "rule_selected_pocket_payload_binding_prototype_positive":
        failures.append(f"BAD_143K_DECISION:{decision.get('decision')}")
    if decision.get("next") != "143P_RULE_SELECTED_POCKET_PAYLOAD_BINDING_SCALE_CONFIRM":
        failures.append(f"BAD_143K_NEXT:{decision.get('next')}")
    for key, expected in {
        "winner_label_parse_accuracy": 1.0,
        "selected_pocket_to_marker_binding_accuracy": 1.0,
        "pocket_marker_order_permutation_accuracy": 1.0,
        "first_prompt_marker_shortcut_rate": 0.0,
        "missing_winner_label_fallback_rate": 1.0,
        "ambiguous_winner_label_rejection_rate": 1.0,
        "closed_pocket_ablation_accuracy": 0.0,
        "legacy_manifest_regression_passed": True,
        "deterministic_replay_passed": True,
    }.items():
        if metrics.get(key) != expected:
            failures.append(f"UPSTREAM_143K_METRIC_MISMATCH:{key}:{metrics.get(key)}")


def require_audits(root: Path, failures: list[str]) -> None:
    helper = load_json(root / "shared_helper_no_change_audit.json")
    if helper.get("shared_helper_no_change_since_143k") is not True:
        failures.append("SHARED_HELPER_CHANGED")
    if helper.get("new_decoder_string_present") is not True:
        failures.append("NEW_DECODER_STRING_NOT_PRESENT")
    request = load_json(root / "helper_request_audit.json")
    if set(request.get("allowed_helper_keys", [])) != ALLOWED_HELPER_KEYS:
        failures.append(f"BAD_ALLOWED_HELPER_KEYS:{request.get('allowed_helper_keys')}")
    for key in ["all_requests_allowed_keys_only", "selected_pocket_id_not_in_request_metadata", "winner_label_not_in_request_metadata"]:
        if request.get(key) is not True:
            failures.append(f"REQUEST_AUDIT_FAIL:{key}:{request.get(key)}")
    if request.get("helper_request_forbidden_metadata_count") != 0:
        failures.append(f"FORBIDDEN_METADATA_COUNT:{request.get('helper_request_forbidden_metadata_count')}")
    if request.get("raw_generate_allowed_in_runner") is not True or request.get("raw_generate_allowed_in_checker") is not False:
        failures.append("RAW_GENERATE_BOUNDARY_BAD")


def require_reports(root: Path, failures: list[str]) -> None:
    prompt = load_json(root / "prompt_scanner_report.json")
    static = load_json(root / "static_manifest_integrity_report.json")
    legacy = load_json(root / "legacy_manifest_regression_report.json")
    order = load_json(root / "pocket_marker_order_permutation_report.json")
    if prompt.get("passed") is not True or prompt.get("case_insensitive_regex") is not True:
        failures.append("PROMPT_SCAN_FAIL")
    if static.get("passed") is not True:
        failures.append("STATIC_MANIFEST_INTEGRITY_FAIL")
    if static.get("per_row_manifest_switch_rate") != 0.0 or static.get("per_row_payload_marker_switch_rate") != 0.0:
        failures.append("PER_ROW_MANIFEST_OR_PAYLOAD_SWITCH")
    for key, expected in {
        "legacy_final_markers_present_accuracy": 1.0,
        "legacy_no_resolved_final_markers_fallback_rate": 1.0,
        "legacy_abc_static_first_marker_rate": 1.0,
        "legacy_old_decoder_binding_activation_rate": 0.0,
        "legacy_manifest_regression_passed": True,
    }.items():
        if legacy.get(key) != expected:
            failures.append(f"LEGACY_METRIC_FAIL:{key}:{legacy.get(key)}")
    for key in ["all_6_marker_orders_covered", "all_3_winner_labels_covered", "all_winner_positions_covered"]:
        if order.get(key) is not True:
            failures.append(f"ORDER_COVERAGE_FAIL:{key}")


def require_metrics(root: Path, failures: list[str]) -> None:
    metrics = load_json(root / "aggregate_metrics.json")
    for key, threshold in {
        "winner_label_parse_accuracy": 0.95,
        "selected_pocket_to_marker_binding_accuracy": 0.95,
        "pocket_marker_order_permutation_accuracy": 0.95,
        "same_values_different_winner_accuracy": 0.95,
        "same_winner_different_values_accuracy": 0.95,
        "main_pocket_writeback_rate": 0.95,
        "missing_winner_label_fallback_rate": 0.95,
        "ambiguous_winner_label_rejection_rate": 0.95,
        "duplicate_same_winner_label_rejection_rate": 0.95,
        "malformed_winner_label_rejection_rate": 0.95,
        "selected_marker_missing_fallback_rate": 0.95,
        "selected_marker_value_missing_fallback_rate": 0.95,
    }.items():
        if metrics.get(key, 0.0) < threshold:
            failures.append(f"METRIC_BELOW_GATE:{key}:{metrics.get(key)}")
    for key in [
        "all_6_marker_orders_covered",
        "all_3_winner_labels_covered",
        "all_winner_positions_covered",
        "selected_pocket_id_not_in_request_metadata",
        "winner_label_not_in_request_metadata",
        "deterministic_replay_passed",
        "legacy_manifest_regression_passed",
        "shared_helper_no_change_since_143k",
    ]:
        if metrics.get(key) is not True:
            failures.append(f"METRIC_NOT_TRUE:{key}:{metrics.get(key)}")
    for key in [
        "first_prompt_marker_shortcut_rate",
        "visible_bypass_violation_rate",
        "noisy_distractor_violation_rate",
        "direct_pocket_value_marker_rate",
        "resolved_final_marker_rate",
        "per_row_manifest_switch_rate",
        "per_row_payload_marker_switch_rate",
        "helper_request_forbidden_metadata_count",
    ]:
        if metrics.get(key) != 0.0 and metrics.get(key) != 0:
            failures.append(f"METRIC_NOT_ZERO:{key}:{metrics.get(key)}")
    if metrics.get("closed_pocket_ablation_accuracy", 1.0) > 0.05:
        failures.append(f"ABLATION_TOO_HIGH:{metrics.get('closed_pocket_ablation_accuracy')}")


def require_decision(root: Path, failures: list[str]) -> None:
    decision = load_json(root / "decision.json")
    summary = load_json(root / "summary.json")
    metrics = load_json(root / "aggregate_metrics.json")
    report = (root / "report.md").read_text(encoding="utf-8")
    if decision.get("decision") not in ACCEPTED_DECISIONS:
        failures.append(f"BAD_DECISION:{decision.get('decision')}")
    if decision.get("decision") == "rule_selected_pocket_payload_binding_scale_confirmed":
        if decision.get("next") != "143Z_RULE_SELECTED_POCKET_BINDING_NEXT_DECISION_PLAN":
            failures.append(f"BAD_POSITIVE_NEXT:{decision.get('next')}")
        if metrics.get("duplicate_selected_marker_conflict_rejection_rate", 0.0) < 0.95:
            failures.append("POSITIVE_WITH_DUPLICATE_CONFLICT_GAP")
    if decision.get("decision") == "duplicate_selected_marker_conflict_not_rejected":
        if decision.get("next") != "143R_DUPLICATE_SELECTED_MARKER_CONFLICT_ANALYSIS":
            failures.append(f"BAD_DUPLICATE_NEXT:{decision.get('next')}")
        if metrics.get("duplicate_selected_marker_conflict_rejection_rate", 1.0) >= 0.95:
            failures.append("DUPLICATE_NEGATIVE_WITHOUT_GAP")
        if metrics.get("duplicate_selected_marker_first_value_rate", 0.0) == 0.0 and metrics.get("duplicate_selected_marker_last_value_rate", 0.0) == 0.0:
            failures.append("DUPLICATE_NEGATIVE_WITHOUT_FIRST_OR_LAST_SELECTION")
    for key in ["rule_metadata_reasoning_claimed", "open_ended_arbitration_claimed", "architecture_superiority_claimed"]:
        if decision.get(key) is not False:
            failures.append(f"DECISION_BOUNDARY_NOT_FALSE:{key}:{decision.get(key)}")
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
    require_audits(root, failures)
    require_reports(root, failures)
    require_metrics(root, failures)
    require_decision(root, failures)
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Check 143P rule-selected pocket payload binding scale confirm")
    parser.add_argument("--root", type=Path, default=SMOKE_ROOT)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--skip-changed-files-check", action="store_true")
    args = parser.parse_args()
    root = args.root if args.root.is_absolute() else (REPO_ROOT / args.root).resolve()
    failures = run_checks(root, check_changed_files=not args.skip_changed_files_check)
    if failures:
        print("143P CHECK FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("143P CHECK PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
