#!/usr/bin/env python3
"""Checker for 140Z multi-field transfer decision plan."""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_140z_instnct_pocket_gated_multi_field_transfer_decision_plan/smoke"
RUNNER = "scripts/probes/run_stable_loop_phase_lock_140z_instnct_pocket_gated_multi_field_transfer_decision_plan.py"
CHECKER = "scripts/probes/run_stable_loop_phase_lock_140z_instnct_pocket_gated_multi_field_transfer_decision_plan_check.py"
DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_140Z_INSTNCT_POCKET_GATED_MULTI_FIELD_TRANSFER_DECISION_PLAN_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_140Z_INSTNCT_POCKET_GATED_MULTI_FIELD_TRANSFER_DECISION_PLAN_RESULT.md",
]
ALLOWED_MUTATIONS = {RUNNER, CHECKER, *DOCS}
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "upstream_140y_manifest.json",
    "analysis_config.json",
    "ast_shortcut_scan_report.json",
    "evidence_chain_summary.json",
    "multi_step_to_multi_field_gap_analysis.json",
    "transfer_decision_matrix.json",
    "multi_field_transfer_requirements.json",
    "anti_shortcut_requirements.json",
    "target_141a_milestone_plan.json",
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


def check_changed_files() -> list[str]:
    failures: list[str] = []
    for path in changed_paths():
        if path in ALLOWED_MUTATIONS or path.startswith("target/"):
            continue
        failures.append(f"UNEXPECTED_CHANGED_FILE:{path}")
    return failures


def check_static_files() -> list[str]:
    failures: list[str] = []
    for rel in [RUNNER, CHECKER, *DOCS]:
        path = REPO_ROOT / rel
        if not path.exists():
            failures.append(f"MISSING_TRACKED_FILE:{rel}")
            continue
        text = path.read_text(encoding="utf-8")
        for term in ["140Z", "planning", "multi-field", "141A", "transfer"]:
            if term not in text:
                failures.append(f"TERM_MISSING:{rel}:{term}")
        if rel in DOCS and "not GPT-like readiness" not in text:
            failures.append(f"DOC_BOUNDARY_TERM_MISSING:{rel}")
        if path.suffix == ".py":
            failures.extend(ast_scan(path))
    return sorted(set(failures))


def check_artifacts() -> list[str]:
    failures: list[str] = []
    for name in REQUIRED_ARTIFACTS:
        if not (SMOKE_ROOT / name).exists():
            failures.append(f"MISSING_ARTIFACT:{name}")
    if failures:
        return failures

    upstream = load_json(SMOKE_ROOT / "upstream_140y_manifest.json")
    config = load_json(SMOKE_ROOT / "analysis_config.json")
    ast_report = load_json(SMOKE_ROOT / "ast_shortcut_scan_report.json")
    evidence = load_json(SMOKE_ROOT / "evidence_chain_summary.json")
    gaps = load_json(SMOKE_ROOT / "multi_step_to_multi_field_gap_analysis.json")
    matrix = load_json(SMOKE_ROOT / "transfer_decision_matrix.json")
    reqs = load_json(SMOKE_ROOT / "multi_field_transfer_requirements.json")
    shortcuts = load_json(SMOKE_ROOT / "anti_shortcut_requirements.json")
    target = load_json(SMOKE_ROOT / "target_141a_milestone_plan.json")
    risks = load_json(SMOKE_ROOT / "risk_register.json")
    decision = load_json(SMOKE_ROOT / "decision.json")
    summary = load_json(SMOKE_ROOT / "summary.json")
    report = (SMOKE_ROOT / "report.md").read_text(encoding="utf-8")
    progress = read_jsonl(SMOKE_ROOT / "progress.jsonl")

    if upstream.get("decision") != "instnct_pocket_gated_multi_step_transfer_scale_confirmed":
        failures.append("BAD_140Y_DECISION")
    if upstream.get("next") != "140Z_INSTNCT_POCKET_GATED_MULTI_FIELD_TRANSFER_DECISION_PLAN":
        failures.append("BAD_140Y_NEXT")
    exact = {
        "eval_row_count": upstream.get("eval_row_count") == 2880,
        "family_count": upstream.get("family_count") == 6,
        "scaffold_variant_count": upstream.get("scaffold_variant_count") == 48,
        "main_final_answer_accuracy": upstream.get("main_final_answer_accuracy") == 1.0,
        "main_step1_intermediate_accuracy": upstream.get("main_step1_intermediate_accuracy") == 1.0,
        "main_step2_final_accuracy": upstream.get("main_step2_final_accuracy") == 1.0,
        "main_pocket_writeback_rate": upstream.get("main_pocket_writeback_rate") == 1.0,
        "main_contrast_group_accuracy": upstream.get("main_contrast_group_accuracy") == 1.0,
        "ablation_final_answer_accuracy": upstream.get("ablation_final_answer_accuracy") == 0.0,
        "ablation_delta": upstream.get("pocket_ablation_delta_final_answer_accuracy") == 1.0,
        "source_copy_shortcut_rate": upstream.get("source_copy_shortcut_rate") == 0.0,
        "intermediate_copy_shortcut_rate": upstream.get("intermediate_copy_shortcut_rate") == 0.0,
        "visible_bypass_violation_rate": upstream.get("visible_bypass_violation_rate") == 0.0,
        "noisy_distractor_violation_rate": upstream.get("noisy_distractor_violation_rate") == 0.0,
        "direct_pocket_value_marker_rate": upstream.get("direct_pocket_value_marker_rate") == 0.0,
        "deterministic_replay_passed": upstream.get("deterministic_replay_passed") is True,
    }
    for key, passed in exact.items():
        if not passed:
            failures.append(f"BAD_140Y_PROFILE:{key}")
    for key in [
        "training_performed",
        "new_model_inference_run",
        "shared_helper_called",
        "helper_generation_called",
        "torch_forward_pass_run",
        "checkpoint_mutated",
        "helper_modified",
        "backend_modified",
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
        failures.append("AST_SCAN_FAILED")
    if evidence.get("current_state") != "multi-step transfer scale confirmed":
        failures.append("BAD_EVIDENCE_STATE")
    if gaps.get("next_gap_to_test") != "multi-field and multi-pocket transfer bridge":
        failures.append("BAD_NEXT_GAP")
    if matrix.get("selected_option") != "multi_field_multi_pocket_transfer" or matrix.get("recommended_next") != "141A_INSTNCT_POCKET_GATED_MULTI_FIELD_TRANSFER_PROBE":
        failures.append("BAD_DECISION_MATRIX")
    if reqs.get("must_test_multi_field_not_single_field") is not True:
        failures.append("MULTI_FIELD_REQUIREMENT_MISSING")
    if reqs.get("helper_only_final_eval") is not True or reqs.get("training_allowed") is not False:
        failures.append("BAD_141A_EXECUTION_POLICY")
    for family in [
        "FIELD_A_PLUS_FIELD_B_TO_FINAL",
        "POCKET_SOURCE_TABLE_RULE_FIELD",
        "DUAL_POCKET_PRIORITY_CONFLICT",
        "MULTI_FIELD_SAME_TEMPLATE_CONTRAST",
        "DISTRACTOR_FIELD_MIX",
        "INTERMEDIATE_FIELD_CHAIN",
    ]:
        if family not in reqs.get("families", []):
            failures.append(f"FAMILY_MISSING:{family}")
    for key in ["main_final_answer_accuracy_min", "multi_field_binding_accuracy_min", "single_field_shortcut_rate"]:
        if key not in reqs.get("positive_gates", {}):
            failures.append(f"POSITIVE_GATE_MISSING:{key}")
    for item in ["field A only", "field B only", "single-field shortcut success"]:
        if item not in shortcuts.get("reject", []):
            failures.append(f"SHORTCUT_REJECTION_MISSING:{item}")
    if target.get("milestone") != "141A_INSTNCT_POCKET_GATED_MULTI_FIELD_TRANSFER_PROBE":
        failures.append("BAD_TARGET_141A")
    if target.get("selected_candidate") != "open_multi_field_final_all_markers":
        failures.append("BAD_TARGET_SELECTED_CANDIDATE")
    for route in ["single_field_shortcut_detected", "multi_field_binding_failure", "pocket_ablation_not_decision_critical", "priority_conflict_failure", "helper_integrity_failure"]:
        if route not in target.get("failure_routes", {}):
            failures.append(f"TARGET_FAILURE_ROUTE_MISSING:{route}")
    if not risks.get("risks"):
        failures.append("RISK_REGISTER_EMPTY")
    if decision.get("decision") != "multi_field_transfer_probe_recommended":
        failures.append("BAD_DECISION")
    if decision.get("next") != "141A_INSTNCT_POCKET_GATED_MULTI_FIELD_TRANSFER_PROBE":
        failures.append("BAD_NEXT")
    if decision.get("planning_only") is not True or decision.get("artifact_only") is not True:
        failures.append("DECISION_NOT_PLANNING_ARTIFACT_ONLY")
    for payload in [decision, summary]:
        require_false_flags(payload, failures)
    if summary.get("decision") != decision.get("decision") or summary.get("next") != decision.get("next"):
        failures.append("SUMMARY_DECISION_MISMATCH")
    for term in ["not GPT-like readiness", "not broad assistant capability", "not production readiness", "not public API"]:
        if term not in report:
            failures.append(f"REPORT_BOUNDARY_TERM_MISSING:{term}")
    events = {row.get("event") for row in progress}
    for event in ["startup", "upstream verification", "artifact loading", "decision matrix", "decision", "final verdict"]:
        if event not in events:
            failures.append(f"PROGRESS_EVENT_MISSING:{event}")
    return sorted(set(failures))


def main() -> int:
    parser = argparse.ArgumentParser(description="Check 140Z multi-field transfer decision plan")
    parser.add_argument("--check-only", action="store_true")
    _args = parser.parse_args()
    failures: list[str] = []
    failures.extend(check_changed_files())
    failures.extend(check_static_files())
    failures.extend(check_artifacts())
    payload = {"schema_version": "phase_140z_checker_result_v1", "status": "pass" if not failures else "fail", "failures": failures}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
