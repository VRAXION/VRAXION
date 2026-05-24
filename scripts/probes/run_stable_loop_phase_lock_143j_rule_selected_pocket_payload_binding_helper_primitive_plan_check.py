#!/usr/bin/env python3
"""Checker for 143J rule-selected pocket payload binding primitive plan."""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_143j_rule_selected_pocket_payload_binding_helper_primitive_plan/smoke"
RUNNER = "scripts/probes/run_stable_loop_phase_lock_143j_rule_selected_pocket_payload_binding_helper_primitive_plan.py"
CHECKER = "scripts/probes/run_stable_loop_phase_lock_143j_rule_selected_pocket_payload_binding_helper_primitive_plan_check.py"
DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_143J_RULE_SELECTED_POCKET_PAYLOAD_BINDING_HELPER_PRIMITIVE_PLAN_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_143J_RULE_SELECTED_POCKET_PAYLOAD_BINDING_HELPER_PRIMITIVE_PLAN_RESULT.md",
]
ALLOWED_MUTATIONS = {RUNNER, CHECKER, *DOCS}
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "analysis_config.json",
    "upstream_143i_manifest.json",
    "primitive_design_requirements.json",
    "primitive_options_matrix.json",
    "oracle_shortcut_risk_register.json",
    "selected_primitive_recommendation.json",
    "target_143k_milestone_plan.json",
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
    "planning-only",
    "constrained helper/backend",
    "does not implement helper/backend behavior",
    "not rule metadata reasoning",
    "not open-domain",
    "not production/public API/deployment/safety readiness",
    "not architecture superiority",
]
SELECTED_OPTION = "prompt_level_explicit_winner_label_parser_plus_static_marker_map"


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
    if config.get("planning_only") is not True or config.get("artifact_only") is not True:
        failures.append("CONFIG_NOT_PLANNING_ARTIFACT_ONLY")
    for key in [
        "helper_generation_run",
        "shared_helper_imported",
        "shared_helper_called",
        "training_performed",
        "checkpoint_mutated",
        "helper_backend_modified",
        "helper_request_key_change",
        "runtime_surface_mutated",
        "product_surface_mutated",
        "release_surface_mutated",
    ]:
        if config.get(key) is not False:
            failures.append(f"CONFIG_BOUNDARY_NOT_FALSE:{key}:{config.get(key)}")
    require_false_flags(config, failures)


def require_upstream(root: Path, failures: list[str]) -> None:
    upstream = load_json(root / "upstream_143i_manifest.json")
    decision = upstream.get("decision", {})
    root_report = upstream.get("root_cause_report", {})
    if decision.get("decision") != "no_resolved_final_marker_bridge_failure_analysis_complete":
        failures.append(f"BAD_143I_DECISION:{decision.get('decision')}")
    if decision.get("root_cause_id") != "helper_payload_marker_selection_lacks_rule_selected_pocket_binding":
        failures.append(f"BAD_143I_ROOT:{decision.get('root_cause_id')}")
    if decision.get("next") != "143J_RULE_SELECTED_POCKET_PAYLOAD_BINDING_HELPER_PRIMITIVE_PLAN":
        failures.append(f"BAD_143I_NEXT:{decision.get('next')}")
    for key in [
        "supported_by_143f_clean_dependency",
        "supported_by_helper_source_audit",
        "hidden_marker_leak_rejected",
        "per_row_manifest_oracle_rejected",
        "shortcut_failure_rejected",
        "abc_static_first_marker_behavior_confirmed",
    ]:
        if root_report.get(key) is not True:
            failures.append(f"BAD_143I_ROOT_EVIDENCE:{key}:{root_report.get(key)}")


def require_options(root: Path, failures: list[str]) -> None:
    requirements = load_json(root / "primitive_design_requirements.json")
    options = load_json(root / "primitive_options_matrix.json")
    risks = load_json(root / "oracle_shortcut_risk_register.json")
    recommendation = load_json(root / "selected_primitive_recommendation.json")
    if requirements.get("main_selected_option") != SELECTED_OPTION:
        failures.append(f"BAD_REQUIREMENTS_SELECTED_OPTION:{requirements.get('main_selected_option')}")
    if requirements.get("manifest_must_not_carry_per_row_selected_pocket_id") is not True:
        failures.append("MANIFEST_SELECTED_POCKET_ORACLE_NOT_FORBIDDEN")
    if "143K positive proves prompt-visible selected-pocket binding only" not in requirements.get("143k_positive_claim_limit", ""):
        failures.append("CLAIM_LIMIT_MISSING_IN_REQUIREMENTS")
    if options.get("selected_option") != SELECTED_OPTION:
        failures.append(f"BAD_OPTIONS_SELECTED:{options.get('selected_option')}")
    expected_order = {
        "prompt_level_explicit_winner_label_parser_plus_static_marker_map": 1,
        "static_pocket_marker_map_plus_prompt_selected_pocket_binding": 2,
        "rule_metadata_parser": 3,
        "keep_resolved_final_marker_only": 4,
    }
    seen = {option.get("option_id"): option for option in options.get("options", [])}
    if set(seen) != set(expected_order):
        failures.append(f"BAD_OPTION_IDS:{sorted(seen)}")
    for option_id, order in expected_order.items():
        option = seen.get(option_id, {})
        for key in ["mechanism", "scope_cost", "oracle_risk", "shortcut_risk", "diagnostic_value", "implementation_risk", "helper_change_required", "request_key_change_required", "recommended_order", "recommendation"]:
            if key not in option:
                failures.append(f"OPTION_FIELD_MISSING:{option_id}:{key}")
        if option.get("recommended_order") != order:
            failures.append(f"BAD_OPTION_ORDER:{option_id}:{option.get('recommended_order')}")
    if recommendation.get("selected_option") != SELECTED_OPTION or recommendation.get("next") != "143K_RULE_SELECTED_POCKET_PAYLOAD_BINDING_PROTOTYPE_PROBE":
        failures.append("BAD_SELECTED_RECOMMENDATION")
    if "It would not prove rule metadata reasoning" not in recommendation.get("claim_limit", ""):
        failures.append("RECOMMENDATION_CLAIM_LIMIT_MISSING")
    risk_text = json.dumps(risks)
    for phrase in ["per-row selected_pocket_id", "per-row manifest", "payload marker list narrowed", "hidden winner-value", "first prompt marker", "overclaimed"]:
        if phrase not in risk_text:
            failures.append(f"RISK_REGISTER_MISSING:{phrase}")


def require_target_143k(root: Path, failures: list[str]) -> None:
    plan = load_json(root / "target_143k_milestone_plan.json")
    if plan.get("milestone") != "143K_RULE_SELECTED_POCKET_PAYLOAD_BINDING_PROTOTYPE_PROBE":
        failures.append(f"BAD_143K_MILESTONE:{plan.get('milestone')}")
    if plan.get("executable_probe") is not True:
        failures.append("143K_NOT_EXECUTABLE_PROBE")
    if "parse winner=pocket_a|pocket_b|pocket_c" not in plan.get("intended_primitive", ""):
        failures.append("143K_INTENDED_PRIMITIVE_MISSING")
    if plan.get("helper_request_keys") != ["prompt", "checkpoint_path", "checkpoint_hash", "seed", "max_new_tokens", "generation_config"]:
        failures.append(f"BAD_143K_HELPER_KEYS:{plan.get('helper_request_keys')}")
    required_controls = {
        "WINNER_LABEL_WRONG_POCKET_CONTROL",
        "WINNER_LABEL_MISSING_CONTROL",
        "WINNER_LABEL_AMBIGUOUS_CONTROL",
        "WINNER_LABEL_POSITION_INVARIANCE_CONTROL",
        "POCKET_MARKER_ORDER_PERMUTATION_CONTROL",
        "SAME_VALUES_DIFFERENT_WINNER_CONTROL",
        "SAME_WINNER_DIFFERENT_VALUES_CONTROL",
        "FIRST_PROMPT_MARKER_SHORTCUT_CONTROL",
        "FIRST_POCKET_CONTROL",
        "DEFAULT_POCKET_CONTROL",
        "LAST_POCKET_CONTROL",
        "STALE_POCKET_CONTROL",
        "VISIBLE_BYPASS_CONTROL",
        "NOISY_DISTRACTOR_CONTROL",
        "CLOSED_POCKET_ABLATION_CONTROL",
        "STATIC_MANIFEST_INTEGRITY_CONTROL",
    }
    if set(plan.get("required_controls", [])) != required_controls:
        failures.append(f"BAD_143K_CONTROLS:{plan.get('required_controls')}")
    scanner = plan.get("prompt_scanner", {})
    if scanner.get("allowed") != ["winner=pocket_a", "winner=pocket_b", "winner=pocket_c"]:
        failures.append(f"BAD_143K_ALLOWED_SCANNER:{scanner.get('allowed')}")
    for phrase in ["winner value", "selected value", "answer value", "gold value", "target value", "arbitrated final", "selected final"]:
        if phrase not in scanner.get("forbidden", []):
            failures.append(f"143K_FORBIDDEN_SCANNER_MISSING:{phrase}")
    for metric in [
        "winner_label_parse_accuracy",
        "selected_pocket_to_marker_binding_accuracy",
        "pocket_marker_order_permutation_accuracy",
        "first_prompt_marker_shortcut_rate",
        "ambiguous_winner_label_rejection_rate",
        "missing_winner_label_fallback_rate",
        "same_values_different_winner_accuracy",
        "same_winner_different_values_accuracy",
    ]:
        if metric not in plan.get("required_metrics", []):
            failures.append(f"143K_METRIC_MISSING:{metric}")
        if metric not in plan.get("positive_gates", {}):
            failures.append(f"143K_GATE_MISSING:{metric}")
    for key in ["winner_label_binding_failure", "oracle_manifest_shortcut_detected", "positional_pocket_shortcut_detected", "ambiguous_winner_not_rejected", "missing_winner_not_rejected", "helper_integrity_failure"]:
        if key not in plan.get("clean_negative_routes", {}):
            failures.append(f"143K_CLEAN_ROUTE_MISSING:{key}")
    must_forbid = " ".join(plan.get("must_forbid", []))
    for phrase in ["per-row selected_pocket_id", "per-row manifest", "payload marker list narrowed", "hidden final/winner-value/gold/answer", "helper request keys", "broad architecture claims"]:
        if phrase not in must_forbid:
            failures.append(f"143K_FORBID_MISSING:{phrase}")


def require_decision_summary_docs(root: Path, failures: list[str]) -> None:
    decision = load_json(root / "decision.json")
    summary = load_json(root / "summary.json")
    report_text = (root / "report.md").read_text(encoding="utf-8")
    if decision.get("decision") != "rule_selected_pocket_payload_binding_primitive_plan_recommended":
        failures.append(f"BAD_DECISION:{decision.get('decision')}")
    if decision.get("selected_option") != SELECTED_OPTION:
        failures.append(f"BAD_DECISION_SELECTED:{decision.get('selected_option')}")
    if decision.get("next") != "143K_RULE_SELECTED_POCKET_PAYLOAD_BINDING_PROTOTYPE_PROBE":
        failures.append(f"BAD_DECISION_NEXT:{decision.get('next')}")
    for key in ["planning_only", "artifact_only"]:
        if decision.get(key) is not True:
            failures.append(f"DECISION_NOT_TRUE:{key}")
    for key in ["helper_backend_modified", "helper_request_key_change", "rule_metadata_reasoning_claimed", "open_ended_arbitration_claimed", "architecture_superiority_claimed"]:
        if decision.get(key) is not False:
            failures.append(f"DECISION_NOT_FALSE:{key}:{decision.get(key)}")
    for payload in [decision, summary]:
        require_false_flags(payload, failures)
        text = json.dumps(payload)
        for phrase in BOUNDARY_PHRASES:
            if phrase not in text:
                failures.append(f"BOUNDARY_PHRASE_MISSING_JSON:{phrase}")
    for phrase in BOUNDARY_PHRASES + ["prompt-visible selected-pocket binding only", "not prove rule metadata reasoning", "No per-row manifest"]:
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
        if "143K positive would prove prompt-visible selected-pocket binding only" not in text:
            failures.append(f"DOC_CLAIM_LIMIT_MISSING:{rel_path}")


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
    require_options(root, failures)
    require_target_143k(root, failures)
    require_decision_summary_docs(root, failures)
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Check 143J rule-selected pocket payload binding primitive plan")
    parser.add_argument("--root", type=Path, default=SMOKE_ROOT)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--skip-changed-files-check", action="store_true")
    args = parser.parse_args()
    root = args.root if args.root.is_absolute() else (REPO_ROOT / args.root).resolve()
    failures = run_checks(root, check_changed_files=not args.skip_changed_files_check)
    if failures:
        print("143J CHECK FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("143J CHECK PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
