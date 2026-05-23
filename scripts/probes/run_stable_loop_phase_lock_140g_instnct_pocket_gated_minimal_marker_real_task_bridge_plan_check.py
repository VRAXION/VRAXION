#!/usr/bin/env python3
"""Checker for 140G minimal-marker real-task bridge plan."""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_140g_instnct_pocket_gated_minimal_marker_real_task_bridge_plan/smoke"
RUNNER = "scripts/probes/run_stable_loop_phase_lock_140g_instnct_pocket_gated_minimal_marker_real_task_bridge_plan.py"
CHECKER = "scripts/probes/run_stable_loop_phase_lock_140g_instnct_pocket_gated_minimal_marker_real_task_bridge_plan_check.py"
DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_140G_INSTNCT_POCKET_GATED_MINIMAL_MARKER_REAL_TASK_BRIDGE_PLAN_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_140G_INSTNCT_POCKET_GATED_MINIMAL_MARKER_REAL_TASK_BRIDGE_PLAN_RESULT.md",
]
ALLOWED_MUTATIONS = {RUNNER, CHECKER, *DOCS}
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "analysis_config.json",
    "ast_shortcut_scan_report.json",
    "upstream_140f_manifest.json",
    "scale_confirm_evidence_summary.json",
    "minimal_marker_gap_analysis.json",
    "real_task_bridge_requirements.json",
    "marker_reduction_policy.json",
    "implicit_gate_policy.json",
    "target_140h_milestone_plan.json",
    "target_140h_failure_routes.json",
    "diagnostic_gap_register.json",
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
        if isinstance(node, ast.Import) and any(alias.name == "importlib" or alias.name.startswith("importlib.") for alias in node.names):
            failures.append(f"HELPER_DYNAMIC_IMPORT_RISK:{path.name}")
        if isinstance(node, ast.Call):
            name = ""
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            if name in {"train", "fit", "backward", "step"}:
                failures.append(f"TRAINING_CALL_NOT_ALLOWED:{path.name}:{name}")
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
        for term in ["140G", "minimal", "marker", "planning-only", "bridge"]:
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

    config = load_json(SMOKE_ROOT / "analysis_config.json")
    ast_report = load_json(SMOKE_ROOT / "ast_shortcut_scan_report.json")
    upstream = load_json(SMOKE_ROOT / "upstream_140f_manifest.json")
    evidence = load_json(SMOKE_ROOT / "scale_confirm_evidence_summary.json")
    gap = load_json(SMOKE_ROOT / "minimal_marker_gap_analysis.json")
    requirements = load_json(SMOKE_ROOT / "real_task_bridge_requirements.json")
    marker_policy = load_json(SMOKE_ROOT / "marker_reduction_policy.json")
    implicit_gate = load_json(SMOKE_ROOT / "implicit_gate_policy.json")
    target = load_json(SMOKE_ROOT / "target_140h_milestone_plan.json")
    routes = load_json(SMOKE_ROOT / "target_140h_failure_routes.json")
    gaps = load_json(SMOKE_ROOT / "diagnostic_gap_register.json")
    risks = load_json(SMOKE_ROOT / "risk_register.json")
    decision = load_json(SMOKE_ROOT / "decision.json")
    summary = load_json(SMOKE_ROOT / "summary.json")
    report = (SMOKE_ROOT / "report.md").read_text(encoding="utf-8")
    progress = read_jsonl(SMOKE_ROOT / "progress.jsonl")

    for key in ["artifact_only", "planning_only"]:
        if config.get(key) is not True:
            failures.append(f"CONFIG_NOT_PLANNING_ONLY:{key}")
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
    require_false_flags(config, failures)
    if ast_report.get("passed") is not True:
        failures.append("AST_SCAN_FAILED")

    if upstream.get("decision") != "instnct_pocket_gated_noisy_marker_bridge_scale_confirmed":
        failures.append("BAD_140F_DECISION")
    if upstream.get("next") != "140G_INSTNCT_POCKET_GATED_MINIMAL_MARKER_REAL_TASK_BRIDGE_PLAN":
        failures.append("BAD_140F_NEXT")
    if upstream.get("eval_row_count", 0) < 2000:
        failures.append("BAD_140F_ROW_COUNT")
    if upstream.get("main_answer_value_accuracy", 0.0) < 0.95:
        failures.append("BAD_140F_MAIN_ACCURACY")
    if upstream.get("main_pocket_writeback_rate", 0.0) < 0.95:
        failures.append("BAD_140F_WRITEBACK")
    if upstream.get("ablation_answer_value_accuracy", 1.0) > 0.05:
        failures.append("BAD_140F_ABLATION")
    if upstream.get("direct_pocket_value_marker_rate", 1.0) > 0.15:
        failures.append("BAD_140F_DIRECT_MARKER_RATE")
    if upstream.get("every_seed_passed") is not True or upstream.get("deterministic_replay_passed") is not True:
        failures.append("BAD_140F_REPLAY_OR_SEED_GATE")

    if evidence.get("scale_confirmed") is not True:
        failures.append("EVIDENCE_NOT_SCALE_CONFIRMED")
    if "GATE:POCKET_OPEN remains explicit." not in gap.get("remaining_scaffolding", []):
        failures.append("GAP_DOES_NOT_IDENTIFY_EXPLICIT_GATE")
    if "POCKET_VALUE=" not in marker_policy.get("disallow_as_main_path", []):
        failures.append("MARKER_POLICY_DOES_NOT_DISALLOW_POCKET_VALUE")
    if target.get("milestone") != "140H_INSTNCT_POCKET_GATED_MINIMAL_MARKER_REAL_TASK_BRIDGE_PROBE":
        failures.append("BAD_TARGET_140H")
    if target.get("train_allowed") is not False or target.get("helper_backend_modification_allowed") is not False:
        failures.append("TARGET_140H_BOUNDARY_BAD")
    design = target.get("required_design", {})
    for key in [
        "minimal_marker_or_implicit_gate",
        "natural_ish_task_text_primary",
        "explicit_pocket_value_marker_forbidden_in_main_eval",
        "explicit_pocket_tokens_strongly_reduced",
        "visible_wrong_value_present",
        "noisy_distractors_present",
        "closed_pocket_ablation_must_fail",
        "mutation_selection_must_prefer_correct_open_pocket_config",
    ]:
        if design.get(key) is not True:
            failures.append(f"TARGET_DESIGN_MISSING:{key}")
    gates = target.get("positive_gates", {})
    if gates.get("explicit_pocket_token_row_rate_max", 1.0) > 0.20:
        failures.append("TARGET_EXPLICIT_TOKEN_GATE_TOO_WEAK")
    if gates.get("direct_pocket_value_marker_rate_max", 1.0) > 0.02:
        failures.append("TARGET_DIRECT_MARKER_GATE_TOO_WEAK")
    if gates.get("implicit_or_minimal_gate_row_rate_min", 0.0) < 0.70:
        failures.append("TARGET_IMPLICIT_GATE_RATE_TOO_WEAK")
    infra = target.get("infrastructure_gates", {})
    for key in [
        "helper_only_final_eval",
        "expected_output_canary_passed",
        "ast_shortcut_scan_passed",
        "generated_text_before_scoring",
        "no_expected_or_scorer_metadata_in_helper_requests",
        "controls_failed",
        "deterministic_replay_passed",
    ]:
        if infra.get(key) is not True:
            failures.append(f"TARGET_INFRA_GATE_MISSING:{key}")
    if implicit_gate.get("ablation_arm", {}).get("post_generation_repair_allowed") is not False:
        failures.append("IMPLICIT_GATE_ABLATION_ALLOWS_REPAIR")
    if requirements.get("positive_gate_recommendations", {}).get("direct_pocket_value_marker_rate_max", 1.0) > 0.02:
        failures.append("REQUIREMENT_DIRECT_MARKER_GATE_TOO_WEAK")
    for route in [
        "minimal_marker_dependency_too_strong",
        "implicit_gate_not_decision_critical",
        "real_task_text_breaks_value_binding",
        "visible_value_bypass_returns",
        "noisy_distractor_copy_returns",
        "mutation_search_fails_to_select_open_pocket",
        "helper_integrity_failure",
    ]:
        if route not in routes.get("routes", {}):
            failures.append(f"FAILURE_ROUTE_MISSING:{route}")
    if not gaps.get("gaps"):
        failures.append("DIAGNOSTIC_GAPS_MISSING")
    if not risks.get("risks"):
        failures.append("RISKS_MISSING")

    if decision.get("decision") != "minimal_marker_real_task_bridge_plan_complete":
        failures.append("BAD_DECISION")
    if decision.get("next") != "140H_INSTNCT_POCKET_GATED_MINIMAL_MARKER_REAL_TASK_BRIDGE_PROBE":
        failures.append("BAD_NEXT")
    if decision.get("planning_only") is not True or decision.get("artifact_only") is not True:
        failures.append("DECISION_NOT_PLANNING_ONLY")
    if decision.get("pocket_mechanism_claimed") is not False:
        failures.append("PLANNING_OVERCLAIMS_POCKET_MECHANISM")
    if decision.get("architecture_superiority_claimed") is not False or decision.get("value_grounding_claimed") is not False:
        failures.append("OVERCLAIM_IN_DECISION")
    for payload in [decision, summary]:
        require_false_flags(payload, failures)
    if summary.get("decision") != decision.get("decision") or summary.get("next") != decision.get("next"):
        failures.append("SUMMARY_DECISION_MISMATCH")
    for term in ["planning-only", "does not train", "not GPT-like readiness", "not broad assistant capability"]:
        if term not in report:
            failures.append(f"REPORT_BOUNDARY_TERM_MISSING:{term}")

    events = {row.get("event") for row in progress}
    for event in [
        "startup",
        "upstream verification",
        "artifact loading",
        "scale confirm evidence summary",
        "minimal marker gap analysis",
        "target 140H plan writing",
        "decision",
        "final verdict",
    ]:
        if event not in events:
            failures.append(f"PROGRESS_EVENT_MISSING:{event}")
    return sorted(set(failures))


def main() -> int:
    parser = argparse.ArgumentParser(description="Check 140G minimal-marker real-task bridge plan")
    parser.add_argument("--check-only", action="store_true")
    _args = parser.parse_args()
    failures: list[str] = []
    failures.extend(check_changed_files())
    failures.extend(check_static_files())
    failures.extend(check_artifacts())
    payload = {"schema_version": "phase_140g_checker_result_v1", "status": "pass" if not failures else "fail", "failures": failures}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
