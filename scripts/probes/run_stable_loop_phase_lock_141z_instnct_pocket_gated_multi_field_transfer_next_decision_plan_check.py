#!/usr/bin/env python3
"""Checker for 141Z next-decision planning milestone."""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_141z_instnct_pocket_gated_multi_field_transfer_next_decision_plan/smoke"
RUNNER = "scripts/probes/run_stable_loop_phase_lock_141z_instnct_pocket_gated_multi_field_transfer_next_decision_plan.py"
CHECKER = "scripts/probes/run_stable_loop_phase_lock_141z_instnct_pocket_gated_multi_field_transfer_next_decision_plan_check.py"
DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_141Z_INSTNCT_POCKET_GATED_MULTI_FIELD_TRANSFER_NEXT_DECISION_PLAN_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_141Z_INSTNCT_POCKET_GATED_MULTI_FIELD_TRANSFER_NEXT_DECISION_PLAN_RESULT.md",
]
ALLOWED_MUTATIONS = {RUNNER, CHECKER, *DOCS}
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "upstream_141f_manifest.json",
    "analysis_config.json",
    "ast_shortcut_scan_report.json",
    "evidence_chain_summary.json",
    "multi_field_to_conflict_priority_gap_analysis.json",
    "next_decision_matrix.json",
    "conflict_priority_transfer_requirements.json",
    "anti_shortcut_requirements.json",
    "target_142a_milestone_plan.json",
    "risk_register.json",
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


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def git_status() -> str:
    result = subprocess.run(["git", "status", "--short"], cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    return result.stdout.rstrip("\n")


def changed_paths() -> list[str]:
    return [line[3:].replace("\\", "/") for line in git_status().splitlines() if line.strip()]


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
        if isinstance(node, ast.Import) and any(alias.name == "torch" for alias in node.names):
            failures.append(f"TORCH_IMPORT_NOT_ALLOWED:{path.name}")
        if isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ""
            if name in {"train", "fit", "backward", "step", "raw_generate"}:
                failures.append(f"FORBIDDEN_CALL:{path.name}:{name}")
    return failures


def require_changed_files(failures: list[str]) -> None:
    for path in changed_paths():
        if path.startswith("target/"):
            continue
        if path not in ALLOWED_MUTATIONS:
            failures.append(f"UNEXPECTED_CHANGED_FILE:{path}")


def require_static_files(failures: list[str]) -> None:
    for rel in [RUNNER, CHECKER, *DOCS]:
        path = REPO_ROOT / rel
        if not path.exists():
            failures.append(f"MISSING_TRACKED_FILE:{rel}")
            continue
        text = path.read_text(encoding="utf-8")
        for term in ["141Z", "planning", "conflict", "priority", "142A"]:
            if term not in text:
                failures.append(f"TERM_MISSING:{rel}:{term}")
        if rel in DOCS:
            for phrase in ["not GPT-like readiness", "not open-ended reasoning", "not broad assistant capability"]:
                if phrase not in text:
                    failures.append(f"DOC_BOUNDARY_TERM_MISSING:{rel}:{phrase}")
        if path.suffix == ".py":
            failures.extend(ast_scan(path))


def require_artifacts(root: Path, failures: list[str]) -> None:
    for name in REQUIRED_ARTIFACTS:
        if not (root / name).exists():
            failures.append(f"MISSING_ARTIFACT:{name}")


def require_upstream(upstream: dict[str, Any], failures: list[str]) -> None:
    checks = upstream.get("gate_checks", {})
    if upstream.get("decision") != "instnct_pocket_gated_multi_field_transfer_scale_confirmed":
        failures.append("BAD_141F_DECISION")
    if upstream.get("next") != "141Z_INSTNCT_POCKET_GATED_MULTI_FIELD_TRANSFER_NEXT_DECISION_PLAN":
        failures.append("BAD_141F_NEXT")
    if upstream.get("failed_gate_checks") != []:
        failures.append(f"FAILED_141F_GATES:{upstream.get('failed_gate_checks')}")
    for key in [
        "main_final_answer_accuracy",
        "main_multi_field_binding_accuracy",
        "main_pocket_writeback_rate",
        "main_contrast_group_accuracy",
        "ablation_final_answer_accuracy",
        "pocket_ablation_delta_final_answer_accuracy",
        "single_field_shortcut_rate",
        "field_a_shortcut_rate",
        "field_b_shortcut_rate",
        "intermediate_copy_shortcut_rate",
        "priority_conflict_wrong_field_rate",
        "visible_bypass_violation_rate",
        "noisy_distractor_violation_rate",
        "direct_pocket_value_marker_rate",
        "deterministic_replay_passed",
        "request_allowed_keys",
        "request_forbidden_key_count",
        "request_no_forbidden_metadata",
        "request_runner_generation_allowed",
        "request_checker_generation_forbidden",
        "seed_gate_report",
        "family_gate_report",
    ]:
        if checks.get(key) is not True:
            failures.append(f"BAD_141F_GATE:{key}")
    audit = upstream.get("helper_request_audit", {})
    if audit.get("all_requests_allowed_keys_only") is not True:
        failures.append("BAD_HELPER_REQUEST_AUDIT_ALLOWED_KEYS")
    if audit.get("forbidden_keys_present_count") != 0:
        failures.append("BAD_HELPER_REQUEST_AUDIT_FORBIDDEN_COUNT")
    if audit.get("raw_generate_allowed_in_runner") is not True:
        failures.append("BAD_HELPER_REQUEST_AUDIT_RUNNER_RAW_GENERATE")
    if audit.get("raw_generate_allowed_in_checker") is not False:
        failures.append("BAD_HELPER_REQUEST_AUDIT_CHECKER_RAW_GENERATE")


def require_target_142a(target: dict[str, Any], reqs: dict[str, Any], failures: list[str]) -> None:
    if target.get("milestone") != "142A_INSTNCT_POCKET_GATED_CONFLICT_PRIORITY_TRANSFER_PROBE":
        failures.append("BAD_TARGET_142A_MILESTONE")
    if target.get("helper_only_final_eval") is not True or target.get("training_allowed") is not False:
        failures.append("BAD_TARGET_EXECUTION_POLICY")
    for row_type in [
        "A wins rows",
        "B wins rows",
        "table override wins rows",
        "rule override wins rows",
        "visible value loses rows",
        "noisy distractor loses rows",
        "same-template different-priority contrast rows",
        "priority inversion pairs",
    ]:
        if row_type not in target.get("row_design", []):
            failures.append(f"TARGET_ROW_TYPE_MISSING:{row_type}")
    for metric in [
        "priority_rule_accuracy",
        "conflict_resolution_accuracy",
        "wrong_priority_field_rate",
        "priority_default_shortcut_rate",
        "priority_inversion_accuracy",
        "same_template_opposite_winner_accuracy",
    ]:
        if metric not in target.get("required_metrics", []):
            failures.append(f"TARGET_METRIC_MISSING:{metric}")
    for artifact in [
        "priority_rule_manifest.json",
        "conflict_pair_manifest.json",
        "priority_inversion_report.json",
        "wrong_priority_field_report.json",
        "priority_control_report.json",
        "helper_request_audit.json",
        "canonical_metric_alias_report.json",
        "per_seed_gate_report.json",
        "per_family_gate_report.json",
    ]:
        if artifact not in target.get("required_artifacts", []):
            failures.append(f"TARGET_ARTIFACT_MISSING:{artifact}")
    for route in [
        "wrong_priority_field_selected",
        "single_field_shortcut_detected",
        "conflict_resolution_failure",
        "priority_default_shortcut_detected",
        "priority_inversion_failure",
        "pocket_ablation_not_decision_critical",
        "helper_integrity_failure",
    ]:
        if route not in target.get("failure_routes", {}):
            failures.append(f"TARGET_ROUTE_MISSING:{route}")
    gates = reqs.get("positive_gates", {})
    for key in [
        "priority_inversion_accuracy_min",
        "same_template_opposite_winner_accuracy_min",
        "priority_default_shortcut_rate",
        "wrong_priority_field_rate",
    ]:
        if key not in gates:
            failures.append(f"REQ_GATE_MISSING:{key}")


def require_artifact_content(root: Path, failures: list[str]) -> None:
    upstream = load_json(root / "upstream_141f_manifest.json")
    config = load_json(root / "analysis_config.json")
    ast_report = load_json(root / "ast_shortcut_scan_report.json")
    evidence = load_json(root / "evidence_chain_summary.json")
    gaps = load_json(root / "multi_field_to_conflict_priority_gap_analysis.json")
    matrix = load_json(root / "next_decision_matrix.json")
    reqs = load_json(root / "conflict_priority_transfer_requirements.json")
    shortcuts = load_json(root / "anti_shortcut_requirements.json")
    target = load_json(root / "target_142a_milestone_plan.json")
    risks = load_json(root / "risk_register.json")
    decision = load_json(root / "decision.json")
    summary = load_json(root / "summary.json")
    report = (root / "report.md").read_text(encoding="utf-8")
    progress = read_jsonl(root / "progress.jsonl")

    require_upstream(upstream, failures)
    for key in [
        "training_performed",
        "new_model_inference_run",
        "shared_helper_called",
        "helper_generation_called",
        "torch_forward_pass_run",
        "checkpoint_mutated",
        "helper_modified",
        "backend_modified",
        "public_request_key_change",
        "runtime_surface_mutated",
        "release_surface_mutated",
        "product_surface_mutated",
        "root_license_changed",
    ]:
        if config.get(key) is not False:
            failures.append(f"BOUNDARY_CONFIG_BAD:{key}")
    if config.get("planning_only") is not True or config.get("artifact_only") is not True:
        failures.append("CONFIG_NOT_PLANNING_ARTIFACT_ONLY")
    require_false_flags(config, failures)
    if ast_report.get("passed") is not True:
        failures.append(f"AST_SCAN_FAILED:{ast_report.get('failures')}")
    if evidence.get("current_state") != "hardened multi-field transfer scale confirmed":
        failures.append("BAD_EVIDENCE_STATE")
    if gaps.get("next_gap_to_test") != "conflict and priority transfer with inversion pairs":
        failures.append("BAD_NEXT_GAP")
    if matrix.get("selected_option") != "conflict_priority_transfer" or matrix.get("recommended_next") != "142A_INSTNCT_POCKET_GATED_CONFLICT_PRIORITY_TRANSFER_PROBE":
        failures.append("BAD_DECISION_MATRIX")
    if reqs.get("must_not_be_always_b_wins") is not True:
        failures.append("ALWAYS_B_GUARD_MISSING")
    for item in ["always B wins", "priority-default shortcut", "same-template priority inversion failure"]:
        if item not in shortcuts.get("reject", []):
            failures.append(f"SHORTCUT_REJECTION_MISSING:{item}")
    require_target_142a(target, reqs, failures)
    if not risks.get("risks"):
        failures.append("RISK_REGISTER_EMPTY")
    if decision.get("decision") != "conflict_priority_transfer_probe_recommended":
        failures.append(f"BAD_DECISION:{decision.get('decision')}")
    if decision.get("selected_option") != "conflict_priority_transfer":
        failures.append(f"BAD_SELECTED_OPTION:{decision.get('selected_option')}")
    if decision.get("next") != "142A_INSTNCT_POCKET_GATED_CONFLICT_PRIORITY_TRANSFER_PROBE":
        failures.append(f"BAD_NEXT:{decision.get('next')}")
    if decision.get("planning_only") is not True or decision.get("artifact_only") is not True:
        failures.append("DECISION_NOT_PLANNING_ARTIFACT_ONLY")
    require_false_flags(decision, failures)
    require_false_flags(summary, failures)
    for phrase in [
        "constrained helper/backend evidence only",
        "not open-ended reasoning",
        "not general composition",
        "not GPT-like readiness",
        "not open-domain reasoning",
        "not broad assistant capability",
        "not production/public API/deployment/safety readiness",
        "not architecture superiority",
    ]:
        if phrase not in report:
            failures.append(f"REPORT_BOUNDARY_MISSING:{phrase}")
    if len(progress) < 5:
        failures.append("PROGRESS_TOO_SHORT")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check 141Z next decision plan artifacts.")
    parser.add_argument("--root", type=Path, default=SMOKE_ROOT)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    root = args.root if args.root.is_absolute() else (REPO_ROOT / args.root).resolve()

    failures: list[str] = []
    require_changed_files(failures)
    require_static_files(failures)
    require_artifacts(root, failures)
    if not failures:
        require_artifact_content(root, failures)
    if failures:
        print("141Z CHECK FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("141Z CHECK PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
