#!/usr/bin/env python3
"""Checker for 143I no-resolved-final-marker bridge failure analysis."""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_143i_no_resolved_final_marker_bridge_failure_analysis/smoke"
RUNNER = "scripts/probes/run_stable_loop_phase_lock_143i_no_resolved_final_marker_bridge_failure_analysis.py"
CHECKER = "scripts/probes/run_stable_loop_phase_lock_143i_no_resolved_final_marker_bridge_failure_analysis_check.py"
DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_143I_NO_RESOLVED_FINAL_MARKER_BRIDGE_FAILURE_ANALYSIS_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_143I_NO_RESOLVED_FINAL_MARKER_BRIDGE_FAILURE_ANALYSIS_RESULT.md",
]
ALLOWED_MUTATIONS = {RUNNER, CHECKER, *DOCS}
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "analysis_config.json",
    "upstream_143f_manifest.json",
    "helper_selection_semantics_audit.json",
    "no_resolved_failure_mode_report.json",
    "abc_static_marker_diagnostic_report.json",
    "explicit_vs_rule_derived_winner_report.json",
    "static_manifest_integrity_report.json",
    "prompt_scan_integrity_report.json",
    "alternative_hypothesis_matrix.json",
    "root_cause_report.json",
    "bridge_options_matrix.json",
    "risk_register.json",
    "target_143j_milestone_plan.json",
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
    "artifact-only failure analysis",
    "constrained helper/backend evidence",
    "not architecture failure",
    "not open-ended reasoning",
    "not general composition",
    "not GPT-like",
    "not production/public API/deployment/safety readiness",
    "not architecture superiority",
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def git_status() -> str:
    result = subprocess.run(["git", "status", "--short"], cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    return result.stdout.rstrip("\n")


def changed_paths() -> list[str]:
    paths: list[str] = []
    for line in git_status().splitlines():
        if line.strip():
            paths.append(line[3:].replace("\\", "/"))
    return paths


def require_false_flags(payload: dict[str, Any], failures: list[str]) -> None:
    for key in FALSE_FLAGS:
        if payload.get(key) is not False:
            failures.append(f"BOUNDARY_FLAG_NOT_FALSE:{key}")


def ast_scan(path: Path) -> list[str]:
    failures: list[str] = []
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("run_stable_loop_phase_lock_"):
            failures.append(f"OLD_RUNNER_IMPORT:{path.name}")
        if isinstance(node, ast.Import) and any(alias.name in {"torch", "shared_raw_generation_helper"} for alias in node.names):
            failures.append(f"FORBIDDEN_IMPORT:{path.name}")
        if isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ""
            if name in {"raw_generate", "load_checkpoint", "train", "fit", "backward", "step", "forward"}:
                failures.append(f"FORBIDDEN_CALL:{path.name}:{name}")
    return failures


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


def require_config(root: Path, failures: list[str]) -> None:
    config = load_json(root / "analysis_config.json")
    expected_false = [
        "training_performed",
        "new_helper_generation_run",
        "shared_helper_imported",
        "shared_helper_called",
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
    require_false_flags(config, failures)


def require_upstream(root: Path, failures: list[str]) -> None:
    upstream = load_json(root / "upstream_143f_manifest.json")
    decision = upstream.get("decision", {})
    metrics = upstream.get("metrics", {})
    if decision.get("decision") != "resolved_final_marker_dependency_confirmed":
        failures.append(f"BAD_143F_DECISION:{decision.get('decision')}")
    if decision.get("next") != "143I_NO_RESOLVED_FINAL_MARKER_BRIDGE_FAILURE_ANALYSIS":
        failures.append(f"BAD_143F_NEXT:{decision.get('next')}")
    if decision.get("clean_negative_valid") is not True:
        failures.append("143F_CLEAN_NEGATIVE_NOT_TRUE")
    expected = {
        "resolved_marker_present_subset_accuracy": 1.0,
        "no_resolved_final_marker_subset_accuracy": 0.0,
        "no_resolved_final_marker_subset_fallback_rate": 1.0,
        "no_resolved_final_marker_subset_shortcut_rate": 0.0,
        "no_resolved_final_marker_subset_unexpected_value_rate": 0.0,
        "no_resolved_final_marker_subset_visible_rate": 0.0,
        "no_resolved_final_marker_subset_noisy_rate": 0.0,
        "no_resolved_final_marker_subset_train_namespace_rate": 0.0,
        "no_resolved_abc_static_first_pocket_rate": 1.0,
        "resolved_marker_dependency_delta": 1.0,
    }
    for key, expected_value in expected.items():
        if metrics.get(key) != expected_value:
            failures.append(f"BAD_143F_METRIC:{key}:{metrics.get(key)}")


def require_helper_audit(root: Path, failures: list[str]) -> None:
    audit = load_json(root / "helper_selection_semantics_audit.json")
    expected = {
        "selected_function_name": "_instnct_select_open_pocket_value",
        "function_found": True,
        "open_gate_check_found": True,
        "payload_marker_loop_found": True,
        "first_present_marker_return_found": True,
        "fallback_if_no_marker_found": True,
        "rule_selected_pocket_binding_found": False,
        "prompt_level_rule_parser_found": False,
        "per_row_manifest_binding_found": False,
    }
    for key, expected_value in expected.items():
        if audit.get(key) != expected_value:
            failures.append(f"BAD_HELPER_AUDIT:{key}:{audit.get(key)}")
    if not audit.get("helper_source_sha256"):
        failures.append("HELPER_SHA_MISSING")


def require_hypotheses(root: Path, failures: list[str]) -> None:
    matrix = load_json(root / "alternative_hypothesis_matrix.json")
    statuses = {row.get("hypothesis_id"): row for row in matrix.get("hypotheses", [])}
    expected_status = {
        "hidden_final_marker_leak": "rejected",
        "per_row_manifest_oracle": "rejected",
        "first_pocket_shortcut": "rejected",
        "default_pocket_shortcut": "rejected",
        "stale_visible_noisy_shortcut": "rejected",
        "random_unexpected_value": "rejected",
        "closed_pocket_fallback_due_to_missing_configured_marker": "supported",
        "abc_static_first_marker_scan": "supported",
        "explicit_winner_label_ignored_by_current_helper": "supported",
        "rule_derived_winner_text_ignored_by_current_helper": "supported",
    }
    for hypothesis_id, status in expected_status.items():
        row = statuses.get(hypothesis_id)
        if not row:
            failures.append(f"MISSING_HYPOTHESIS:{hypothesis_id}")
            continue
        if row.get("status") != status:
            failures.append(f"BAD_HYPOTHESIS_STATUS:{hypothesis_id}:{row.get('status')}")
        if not row.get("evidence_artifacts") or not row.get("metrics") or not row.get("explanation"):
            failures.append(f"HYPOTHESIS_NOT_AUDITABLE:{hypothesis_id}")


def require_root_and_options(root: Path, failures: list[str]) -> None:
    report = load_json(root / "root_cause_report.json")
    options = load_json(root / "bridge_options_matrix.json")
    plan = load_json(root / "target_143j_milestone_plan.json")
    required_true = [
        "supported_by_143f_clean_dependency",
        "supported_by_helper_source_audit",
        "hidden_marker_leak_rejected",
        "per_row_manifest_oracle_rejected",
        "shortcut_failure_rejected",
        "abc_static_first_marker_behavior_confirmed",
    ]
    if report.get("root_cause_id") != "helper_payload_marker_selection_lacks_rule_selected_pocket_binding":
        failures.append(f"BAD_ROOT_CAUSE:{report.get('root_cause_id')}")
    for key in required_true:
        if report.get(key) is not True:
            failures.append(f"ROOT_CAUSE_EVIDENCE_NOT_TRUE:{key}:{report.get(key)}")
    if report.get("architecture_failure_claimed") is not False:
        failures.append("ARCHITECTURE_FAILURE_CLAIMED")
    option_ids = {option.get("option_id") for option in options.get("options", [])}
    required_options = {
        "manifest_level_selected_pocket_binding_primitive",
        "prompt_level_explicit_winner_label_parser",
        "rule_metadata_parser",
        "keep_resolved_final_marker_and_stop_bridge_expansion",
    }
    if option_ids != required_options:
        failures.append(f"BAD_OPTIONS:{sorted(option_ids)}")
    for option in options.get("options", []):
        for key in ["mechanism", "expected_benefit", "oracle_risk", "shortcut_risk", "helper_change_required", "request_key_change_required", "required_controls", "recommendation"]:
            if key not in option:
                failures.append(f"OPTION_FIELD_MISSING:{option.get('option_id')}:{key}")
    if plan.get("milestone") != "143J_RULE_SELECTED_POCKET_PAYLOAD_BINDING_HELPER_PRIMITIVE_PLAN":
        failures.append(f"BAD_143J_MILESTONE:{plan.get('milestone')}")
    if plan.get("planning_only") is not True:
        failures.append("143J_NOT_PLANNING_ONLY")
    if plan.get("recommended_next") != "143K_RULE_SELECTED_POCKET_PAYLOAD_BINDING_PROTOTYPE_PROBE":
        failures.append(f"BAD_143J_NEXT:{plan.get('recommended_next')}")
    forbidden = " ".join(plan.get("must_forbid", []))
    for phrase in ["per-row oracle", "per-row manifest", "payload marker list", "helper request key changes", "broad architecture claims"]:
        if phrase not in forbidden:
            failures.append(f"143J_FORBID_MISSING:{phrase}")


def require_reports(root: Path, failures: list[str]) -> None:
    failure = load_json(root / "no_resolved_failure_mode_report.json")
    abc = load_json(root / "abc_static_marker_diagnostic_report.json")
    explicit = load_json(root / "explicit_vs_rule_derived_winner_report.json")
    static = load_json(root / "static_manifest_integrity_report.json")
    prompt = load_json(root / "prompt_scan_integrity_report.json")
    if failure.get("clean_failure") is not True or failure.get("failure_mode") != "closed_pocket_fallback_due_to_missing_configured_marker":
        failures.append("NO_RESOLVED_FAILURE_MODE_BAD")
    if abc.get("first_marker_scan_behavior_confirmed") is not True:
        failures.append("ABC_FIRST_MARKER_NOT_CONFIRMED")
    if explicit.get("explicit_winner_label_ignored_by_current_helper") is not True:
        failures.append("EXPLICIT_WINNER_LABEL_NOT_MARKED_IGNORED")
    if explicit.get("rule_derived_winner_text_ignored_by_current_helper") is not True:
        failures.append("RULE_DERIVED_TEXT_NOT_MARKED_IGNORED")
    if static.get("passed") is not True:
        failures.append("STATIC_MANIFEST_INTEGRITY_NOT_PASS")
    if prompt.get("passed") is not True:
        failures.append("PROMPT_SCAN_NOT_PASS")


def require_decision_summary_docs(root: Path, failures: list[str]) -> None:
    decision = load_json(root / "decision.json")
    summary = load_json(root / "summary.json")
    report_text = (root / "report.md").read_text(encoding="utf-8")
    if decision.get("decision") != "no_resolved_final_marker_bridge_failure_analysis_complete":
        failures.append(f"BAD_DECISION:{decision.get('decision')}")
    if decision.get("root_cause_id") != "helper_payload_marker_selection_lacks_rule_selected_pocket_binding":
        failures.append(f"BAD_DECISION_ROOT_CAUSE:{decision.get('root_cause_id')}")
    if decision.get("next") != "143J_RULE_SELECTED_POCKET_PAYLOAD_BINDING_HELPER_PRIMITIVE_PLAN":
        failures.append(f"BAD_DECISION_NEXT:{decision.get('next')}")
    if decision.get("artifact_only") is not True or decision.get("architecture_failure_claimed") is not False:
        failures.append("DECISION_BOUNDARY_BAD")
    for payload in [decision, summary]:
        require_false_flags(payload, failures)
        text = json.dumps(payload)
        for phrase in BOUNDARY_PHRASES:
            if phrase not in text:
                failures.append(f"BOUNDARY_PHRASE_MISSING_JSON:{phrase}")
    for phrase in BOUNDARY_PHRASES + ["helper-semantics bottleneck", "rule-selected pocket payload binding"]:
        if phrase not in report_text:
            failures.append(f"REPORT_PHRASE_MISSING:{phrase}")
    for rel_path in DOCS:
        path = REPO_ROOT / rel_path
        text = path.read_text(encoding="utf-8")
        if len(text.strip()) < 500:
            failures.append(f"DOC_TOO_SHORT:{rel_path}")
        for phrase in BOUNDARY_PHRASES:
            if phrase not in text:
                failures.append(f"DOC_BOUNDARY_PHRASE_MISSING:{rel_path}:{phrase}")


def run_checks(root: Path, check_changed_files: bool) -> list[str]:
    failures: list[str] = []
    if check_changed_files:
        require_changed_files(failures)
    for path in [REPO_ROOT / RUNNER, REPO_ROOT / CHECKER]:
        failures.extend(ast_scan(path))
    require_artifacts(root, failures)
    if failures:
        return failures
    require_config(root, failures)
    require_upstream(root, failures)
    require_helper_audit(root, failures)
    require_hypotheses(root, failures)
    require_root_and_options(root, failures)
    require_reports(root, failures)
    require_decision_summary_docs(root, failures)
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Check 143I no-resolved final marker bridge failure analysis")
    parser.add_argument("--root", type=Path, default=SMOKE_ROOT)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--skip-changed-files-check", action="store_true")
    args = parser.parse_args()
    root = args.root if args.root.is_absolute() else (REPO_ROOT / args.root).resolve()
    failures = run_checks(root, check_changed_files=not args.skip_changed_files_check)
    if failures:
        print("143I CHECK FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("143I CHECK PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
