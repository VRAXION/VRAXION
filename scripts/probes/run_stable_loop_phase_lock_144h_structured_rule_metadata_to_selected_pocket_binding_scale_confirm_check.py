#!/usr/bin/env python3
"""Checker for 144H structured rule metadata binding scale confirm."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_144h_structured_rule_metadata_to_selected_pocket_binding_scale_confirm/smoke"
HELPER = "scripts/probes/shared_raw_generation_helper.py"
RUNNER = "scripts/probes/run_stable_loop_phase_lock_144h_structured_rule_metadata_to_selected_pocket_binding_scale_confirm.py"
CHECKER = "scripts/probes/run_stable_loop_phase_lock_144h_structured_rule_metadata_to_selected_pocket_binding_scale_confirm_check.py"
DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_144H_STRUCTURED_RULE_METADATA_TO_SELECTED_POCKET_BINDING_SCALE_CONFIRM_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_144H_STRUCTURED_RULE_METADATA_TO_SELECTED_POCKET_BINDING_SCALE_CONFIRM_RESULT.md",
]
ALLOWED_MUTATIONS = {RUNNER, CHECKER, *DOCS}
ALLOWED_HELPER_KEYS = {"prompt", "checkpoint_path", "checkpoint_hash", "seed", "max_new_tokens", "generation_config"}
NEW_DECODER = "deterministic_pocket_gated_structured_rule_metadata_binding_decoder"
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "analysis_config.json",
    "upstream_144b_manifest.json",
    "shared_helper_no_change_audit.json",
    "helper_structured_rule_semantics_audit.json",
    "structured_rule_metadata_parser_report.json",
    "derived_selected_pocket_report.json",
    "selected_pocket_binding_report.json",
    "quorum_rule_report.json",
    "quorum_tie_break_report.json",
    "quorum_clear_winner_ignores_tie_break_report.json",
    "recency_rule_report.json",
    "wrong_family_extra_key_rejection_report.json",
    "tie_break_rule_report.json",
    "hierarchy_policy_report.json",
    "same_values_different_rule_report.json",
    "same_rule_different_values_report.json",
    "same_template_opposite_rule_winner_report.json",
    "rule_metadata_corruption_matrix_report.json",
    "missing_rule_metadata_matrix_report.json",
    "ambiguous_rule_metadata_matrix_report.json",
    "rule_metadata_ablation_report.json",
    "explicit_winner_baseline_report.json",
    "legacy_selected_pocket_binding_regression_report.json",
    "legacy_143w_binding_regression_report.json",
    "static_manifest_integrity_report.json",
    "helper_request_audit.json",
    "prompt_scanner_report.json",
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
    "structured rule metadata to selected-pocket binding only",
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


def require_static_files(failures: list[str]) -> None:
    for rel_path in [RUNNER, CHECKER, *DOCS]:
        path = REPO_ROOT / rel_path
        if not path.exists():
            failures.append(f"MISSING_TRACKED_FILE:{rel_path}")
            continue
        text = path.read_text(encoding="utf-8")
        for term in ["144H", NEW_DECODER, "structured rule metadata"]:
            if term not in text:
                failures.append(f"TERM_MISSING:{rel_path}:{term}")
        for phrase in BOUNDARY_PHRASES:
            if phrase not in text:
                failures.append(f"BOUNDARY_TERM_MISSING:{rel_path}:{phrase}")
    helper_source = (REPO_ROOT / HELPER).read_text(encoding="utf-8")
    head = subprocess.run(["git", "show", f"HEAD:{HELPER}"], cwd=REPO_ROOT, text=True, capture_output=True, check=False).stdout
    if helper_source != head:
        failures.append("SHARED_HELPER_CHANGED_FROM_HEAD")


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


def require_artifacts(root: Path, failures: list[str]) -> None:
    for name in REQUIRED_ARTIFACTS:
        if not (root / name).exists():
            failures.append(f"MISSING_ARTIFACT:{name}")


def require_upstream(root: Path, failures: list[str]) -> None:
    upstream = load_json(root / "upstream_144b_manifest.json")
    decision = upstream.get("decision", {})
    metrics = upstream.get("aggregate_metrics", {})
    if decision.get("decision") != "structured_rule_metadata_to_selected_pocket_binding_prototype_positive":
        failures.append(f"BAD_144B_DECISION:{decision.get('decision')}")
    if decision.get("next") != "144H_STRUCTURED_RULE_METADATA_TO_SELECTED_POCKET_BINDING_SCALE_CONFIRM":
        failures.append(f"BAD_144B_NEXT:{decision.get('next')}")
    expected = {
        "rule_metadata_parse_accuracy": 1.0,
        "derived_selected_pocket_accuracy": 1.0,
        "selected_pocket_to_marker_binding_accuracy": 1.0,
        "same_line_value_extraction_accuracy": 1.0,
        "end_to_end_answer_accuracy": 1.0,
        "rule_metadata_ablation_accuracy": 0.0,
        "corrupt_rule_metadata_rejection_rate": 1.0,
        "missing_rule_metadata_fallback_rate": 1.0,
        "ambiguous_rule_metadata_rejection_rate": 1.0,
        "legacy_selected_pocket_binding_regression_passed": True,
        "deterministic_replay_passed": True,
    }
    for key, value in expected.items():
        if metrics.get(key) != value:
            failures.append(f"UPSTREAM_144B_METRIC_MISMATCH:{key}:{metrics.get(key)}")


def require_helper_audits(root: Path, failures: list[str]) -> None:
    no_change = load_json(root / "shared_helper_no_change_audit.json")
    semantics = load_json(root / "helper_structured_rule_semantics_audit.json")
    helper_source = (REPO_ROOT / HELPER).read_text(encoding="utf-8")
    current_hash = hashlib.sha256(helper_source.encode("utf-8", errors="replace")).hexdigest()
    if no_change.get("current_shared_helper_sha256") != current_hash:
        failures.append("HELPER_HASH_MISMATCH")
    for key in ["shared_helper_no_change_since_144b", "shared_helper_modified_by_144h", "shared_raw_generation_helper_unchanged_from_head", "new_decoder_string_present", "passed"]:
        expected = False if key == "shared_helper_modified_by_144h" else True
        if no_change.get(key) is not expected:
            failures.append(f"HELPER_NO_CHANGE_FAIL:{key}:{no_change.get(key)}")
    for key in [
        "structured_function_found",
        "parser_function_found",
        "derive_function_found",
        "new_decoder_manifest_gated",
        "old_selected_pocket_decoder_present",
        "candidate_line_re_present",
        "same_line_extraction_present",
        "rule_type_quorum_present",
        "rule_type_recency_present",
        "rule_type_tie_break_present",
        "rule_type_hierarchy_present",
        "hierarchy_combiner_only_wording_present",
        "passed",
    ]:
        if semantics.get(key) is not True:
            failures.append(f"HELPER_SEMANTICS_FAIL:{key}:{semantics.get(key)}")


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
    if request.get("raw_generate_allowed_in_runner") is not True or request.get("raw_generate_allowed_in_checker") is not False:
        failures.append("RAW_GENERATE_BOUNDARY_BAD")
    if static.get("passed") is not True or static.get("per_row_manifest_switch_rate") != 0.0 or static.get("per_row_payload_marker_switch_rate") != 0.0:
        failures.append("STATIC_MANIFEST_INTEGRITY_FAIL")
    if static.get("payload_marker_list_narrowed_to_correct_pocket") is not False:
        failures.append("PAYLOAD_MARKER_LIST_NARROWED")
    if prompt.get("passed") is not True or prompt.get("forbidden_violation_count") != 0:
        failures.append("PROMPT_SCANNER_FAIL")


def require_reports(root: Path, failures: list[str]) -> None:
    for name in [
        "structured_rule_metadata_parser_report.json",
        "derived_selected_pocket_report.json",
        "selected_pocket_binding_report.json",
        "quorum_rule_report.json",
        "quorum_tie_break_report.json",
        "quorum_clear_winner_ignores_tie_break_report.json",
        "recency_rule_report.json",
        "wrong_family_extra_key_rejection_report.json",
        "tie_break_rule_report.json",
        "hierarchy_policy_report.json",
        "same_values_different_rule_report.json",
        "same_rule_different_values_report.json",
        "same_template_opposite_rule_winner_report.json",
        "rule_metadata_corruption_matrix_report.json",
        "missing_rule_metadata_matrix_report.json",
        "ambiguous_rule_metadata_matrix_report.json",
        "rule_metadata_ablation_report.json",
        "explicit_winner_baseline_report.json",
        "legacy_selected_pocket_binding_regression_report.json",
        "legacy_143w_binding_regression_report.json",
    ]:
        report = load_json(root / name)
        if report.get("passed") is not True:
            failures.append(f"REPORT_NOT_PASSED:{name}")
    for name in ["per_seed_gate_report.json", "per_family_gate_report.json"]:
        report = load_json(root / name)
        if report.get("passed") is not True:
            failures.append(f"GATE_REPORT_NOT_PASSED:{name}")


def require_metrics(root: Path, failures: list[str]) -> None:
    metrics = load_json(root / "aggregate_metrics.json")
    for key, threshold in {
        "rule_metadata_parse_accuracy": 0.98,
        "derived_selected_pocket_accuracy": 0.98,
        "selected_pocket_to_marker_binding_accuracy": 0.98,
        "same_line_value_extraction_accuracy": 0.98,
        "end_to_end_answer_accuracy": 0.98,
        "rule_derived_no_winner_label_accuracy": 0.98,
        "explicit_winner_baseline_accuracy": 0.98,
        "corrupt_rule_metadata_rejection_rate": 0.98,
        "missing_rule_metadata_fallback_rate": 0.98,
        "ambiguous_rule_metadata_rejection_rate": 0.98,
        "quorum_tie_break_accuracy": 0.98,
        "quorum_clear_winner_ignores_tie_break_accuracy": 0.98,
        "wrong_family_extra_key_rejection_rate": 0.98,
        "hierarchy_policy_accuracy": 0.98,
        "same_template_opposite_rule_winner_accuracy": 0.98,
    }.items():
        if metrics.get(key, 0.0) < threshold:
            failures.append(f"METRIC_BELOW_GATE:{key}:{metrics.get(key)}")
    if metrics.get("rule_metadata_ablation_accuracy", 1.0) > 0.05:
        failures.append(f"RULE_METADATA_ABLATION_TOO_HIGH:{metrics.get('rule_metadata_ablation_accuracy')}")
    for key in ["helper_request_forbidden_metadata_count", "per_row_manifest_switch_rate", "per_row_payload_marker_switch_rate"]:
        if metrics.get(key) not in {0, 0.0}:
            failures.append(f"METRIC_NOT_ZERO:{key}:{metrics.get(key)}")
    for key in ["legacy_selected_pocket_binding_regression_passed", "legacy_143w_binding_regression_passed", "shared_helper_no_change_since_144b", "deterministic_replay_passed"]:
        if metrics.get(key) is not True:
            failures.append(f"METRIC_NOT_TRUE:{key}:{metrics.get(key)}")


def require_decision_and_boundary(root: Path, failures: list[str]) -> None:
    decision = load_json(root / "decision.json")
    summary = load_json(root / "summary.json")
    report = (root / "report.md").read_text(encoding="utf-8")
    if decision.get("decision") != "structured_rule_metadata_to_selected_pocket_binding_scale_confirmed":
        failures.append(f"BAD_DECISION:{decision.get('decision')}")
    if decision.get("verdict") != "INSTNCT_STRUCTURED_RULE_METADATA_TO_SELECTED_POCKET_BINDING_SCALE_CONFIRMED":
        failures.append(f"BAD_VERDICT:{decision.get('verdict')}")
    if decision.get("next") != "144Z_STRUCTURED_RULE_METADATA_BINDING_NEXT_DECISION_PLAN":
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
    require_helper_audits(root, failures)
    require_audits(root, failures)
    require_reports(root, failures)
    require_metrics(root, failures)
    require_decision_and_boundary(root, failures)
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Check 144H structured rule metadata binding scale confirm")
    parser.add_argument("--root", type=Path, default=SMOKE_ROOT)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--skip-changed-files-check", action="store_true")
    args = parser.parse_args()
    root = args.root if args.root.is_absolute() else (REPO_ROOT / args.root).resolve()
    failures = run_checks(root, check_changed_files=not args.skip_changed_files_check)
    if failures:
        print("144H CHECK FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("144H CHECK PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
