#!/usr/bin/env python3
"""Checker for 139YS pocket-gated scale-or-bridge decision plan."""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_139ys_instnct_pocket_gated_mutation_search_scale_or_real_task_bridge/smoke"
RUNNER = "scripts/probes/run_stable_loop_phase_lock_139ys_instnct_pocket_gated_mutation_search_scale_or_real_task_bridge.py"
CHECKER = "scripts/probes/run_stable_loop_phase_lock_139ys_instnct_pocket_gated_mutation_search_scale_or_real_task_bridge_check.py"
DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_139YS_INSTNCT_POCKET_GATED_MUTATION_SEARCH_SCALE_OR_REAL_TASK_BRIDGE_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_139YS_INSTNCT_POCKET_GATED_MUTATION_SEARCH_SCALE_OR_REAL_TASK_BRIDGE_RESULT.md",
]
ALLOWED_MUTATIONS = {RUNNER, CHECKER, *DOCS}
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "analysis_config.json",
    "ast_shortcut_scan_report.json",
    "upstream_139yr_manifest.json",
    "upstream_139yq_manifest.json",
    "upstream_138yo_manifest.json",
    "upstream_138yk_manifest.json",
    "evidence_chain_summary.json",
    "scale_vs_bridge_decision_matrix.json",
    "risk_register.json",
    "next_milestone_recommendation.json",
    "target_140a_milestone_plan.json",
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
        for term in ["139YS", "pocket", "bridge", "planning-only"]:
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
    upstream_139yr = load_json(SMOKE_ROOT / "upstream_139yr_manifest.json")
    upstream_139yq = load_json(SMOKE_ROOT / "upstream_139yq_manifest.json")
    upstream_138yo = load_json(SMOKE_ROOT / "upstream_138yo_manifest.json")
    upstream_138yk = load_json(SMOKE_ROOT / "upstream_138yk_manifest.json")
    chain = load_json(SMOKE_ROOT / "evidence_chain_summary.json")
    matrix = load_json(SMOKE_ROOT / "scale_vs_bridge_decision_matrix.json")
    recommendation = load_json(SMOKE_ROOT / "next_milestone_recommendation.json")
    plan = load_json(SMOKE_ROOT / "target_140a_milestone_plan.json")
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

    if upstream_139yr.get("decision") != "instnct_pocket_gated_mutation_search_confirmed":
        failures.append("BAD_139YR_DECISION")
    if upstream_139yr.get("verdict") != "INSTNCT_POCKET_GATED_MUTATION_SEARCH_CONFIRMED":
        failures.append("BAD_139YR_VERDICT")
    if upstream_139yr.get("gradient_used") is not False:
        failures.append("139YR_GRADIENT_USED")
    if upstream_139yr.get("selected") != "open_pocket_all_payload_markers":
        failures.append("BAD_139YR_SELECTED")
    if upstream_139yr.get("fitness_margin", 0.0) < 0.40:
        failures.append("139YR_FITNESS_MARGIN_TOO_LOW")
    if upstream_139yr.get("next") != "139YS_INSTNCT_POCKET_GATED_MUTATION_SEARCH_SCALE_OR_REAL_TASK_BRIDGE":
        failures.append("BAD_139YR_NEXT")

    if upstream_139yq.get("main_answer_value_accuracy") != 1.0:
        failures.append("BAD_139YQ_MAIN_ACCURACY")
    if upstream_139yq.get("main_pocket_writeback_rate") != 1.0:
        failures.append("BAD_139YQ_WRITEBACK")
    if upstream_139yq.get("main_phase_transport_success_rate") != 1.0:
        failures.append("BAD_139YQ_PHASE_TRANSPORT")
    if upstream_139yq.get("ablation_answer_value_accuracy") != 0.0:
        failures.append("BAD_139YQ_ABLATION_ACCURACY")
    if upstream_139yq.get("pocket_ablation_delta_answer_value_accuracy") != 1.0:
        failures.append("BAD_139YQ_ABLATION_DELTA")
    if upstream_139yq.get("deterministic_replay_passed") is not True:
        failures.append("BAD_139YQ_DETERMINISM")

    if upstream_138yo.get("instnct_answer_value_accuracy", 0.0) <= upstream_138yo.get("byte_gru_answer_value_accuracy", 1.0):
        failures.append("138YO_ADAPTER_DID_NOT_BEAT_BYTE_GRU")
    if upstream_138yo.get("instnct_pocket_writeback_rate") != 0.0:
        failures.append("BAD_138YO_POCKET_WRITEBACK")
    if upstream_138yo.get("pocket_ablation_delta_answer_value_accuracy") != 0.0:
        failures.append("BAD_138YO_ABLATION_DELTA")

    if upstream_138yk.get("answer_value_accuracy") != 0.0:
        failures.append("BAD_138YK_VALUE_ACCURACY")
    if upstream_138yk.get("family_default_shortcut_detected") is not True:
        failures.append("BAD_138YK_FAMILY_DEFAULT_PROFILE")
    if upstream_138yk.get("determinism_replay_passed") is not True:
        failures.append("BAD_138YK_DETERMINISM")

    if chain.get("chain_complete") is not True or len(chain.get("stages", [])) != 4:
        failures.append("EVIDENCE_CHAIN_INCOMPLETE")
    if matrix.get("decision") != "real_task_bridge_recommended":
        failures.append("BAD_MATRIX_DECISION")
    if matrix.get("selected_option") != "real_task_bridge_reduced_marker_probe":
        failures.append("BAD_MATRIX_SELECTION")
    scale = next((item for item in matrix.get("options", []) if item.get("option") == "scale_current_marker_bound_proof"), None)
    bridge = next((item for item in matrix.get("options", []) if item.get("option") == "real_task_bridge_reduced_marker_probe"), None)
    if not scale or scale.get("selected") is not False:
        failures.append("SCALE_OPTION_NOT_REJECTED")
    if not bridge or bridge.get("selected") is not True:
        failures.append("BRIDGE_OPTION_NOT_SELECTED")

    if recommendation.get("decision") != "real_task_bridge_recommended":
        failures.append("BAD_RECOMMENDATION_DECISION")
    if recommendation.get("recommended_next") != "140A_INSTNCT_POCKET_GATED_NOISY_MARKER_BRIDGE_PROBE":
        failures.append("BAD_RECOMMENDED_NEXT")

    if plan.get("milestone") != "140A_INSTNCT_POCKET_GATED_NOISY_MARKER_BRIDGE_PROBE":
        failures.append("BAD_140A_MILESTONE")
    bridge_design = plan.get("bridge_design", {})
    for key in [
        "explicit_pocket_value_markers_reduced",
        "noisy_prompt_distractors_added",
        "value_hidden_behind_natural_task_text",
        "pocket_gate_still_required",
        "visible_value_bypass_forbidden",
        "closed_pocket_ablation_must_fail",
        "mutation_selection_must_prefer_correct_open_pocket_config",
    ]:
        if bridge_design.get(key) is not True:
            failures.append(f"140A_BRIDGE_REQUIREMENT_MISSING:{key}")
    gates = plan.get("positive_gates", {})
    if gates.get("main_answer_value_accuracy_min", 0.0) < 0.80:
        failures.append("140A_MAIN_GATE_TOO_WEAK")
    if gates.get("pocket_writeback_rate_min", 0.0) < 0.90:
        failures.append("140A_WRITEBACK_GATE_TOO_WEAK")
    if gates.get("ablation_answer_value_accuracy_max", 1.0) > 0.10:
        failures.append("140A_ABLATION_GATE_TOO_WEAK")
    if gates.get("pocket_ablation_delta_min", 0.0) < 0.50:
        failures.append("140A_DELTA_GATE_TOO_WEAK")
    routes = plan.get("clean_negative_routes", {})
    for route in [
        "marker_dependency_too_strong",
        "pocket_ablation_not_decision_critical",
        "noisy_prompt_breaks_value_binding",
        "mutation_search_fails_to_select_open_pocket",
        "helper_integrity_failure",
    ]:
        if route not in routes:
            failures.append(f"140A_ROUTE_MISSING:{route}")
    controls = set(plan.get("required_controls", []))
    for control in ["VISIBLE_VALUE_BYPASS_CONTROL", "NOISY_DISTRACTOR_CONTROL", "CLOSED_POCKET_ABLATION_CONTROL"]:
        if control not in controls:
            failures.append(f"140A_CONTROL_MISSING:{control}")
    infra = plan.get("infrastructure_gates", {})
    for key in [
        "helper_only_final_eval",
        "expected_output_canary_passed",
        "ast_shortcut_scan_passed",
        "leakage_rejected",
        "controls_failed",
        "deterministic_replay_passed",
        "generated_text_before_scoring",
        "no_expected_or_scorer_metadata_in_helper_requests",
    ]:
        if infra.get(key) is not True:
            failures.append(f"140A_INFRA_GATE_MISSING:{key}")

    if decision.get("decision") != "real_task_bridge_recommended":
        failures.append("BAD_DECISION")
    if decision.get("next") != "140A_INSTNCT_POCKET_GATED_NOISY_MARKER_BRIDGE_PROBE":
        failures.append("BAD_NEXT")
    if decision.get("planning_only") is not True or decision.get("artifact_only") is not True:
        failures.append("DECISION_NOT_PLANNING_ONLY")
    if summary.get("decision") != decision.get("decision") or summary.get("next") != decision.get("next"):
        failures.append("SUMMARY_DECISION_MISMATCH")
    for payload in [decision, summary]:
        require_false_flags(payload, failures)
        if payload.get("architecture_superiority_claimed") is not False or payload.get("value_grounding_claimed") is not False:
            failures.append("OVERCLAIM_IN_DECISION_OR_SUMMARY")
    for term in ["planning-only", "does not run training", "not GPT-like readiness", "not broad assistant capability"]:
        if term not in report:
            failures.append(f"REPORT_BOUNDARY_TERM_MISSING:{term}")

    events = {row.get("event") for row in progress}
    for event in [
        "startup",
        "upstream verification",
        "artifact loading",
        "evidence chain summary",
        "scale vs bridge decision matrix",
        "target 140A plan writing",
        "recommendation",
        "decision",
        "final verdict",
    ]:
        if event not in events:
            failures.append(f"PROGRESS_EVENT_MISSING:{event}")
    return sorted(set(failures))


def main() -> int:
    parser = argparse.ArgumentParser(description="Check 139YS scale-or-bridge decision plan")
    parser.add_argument("--check-only", action="store_true")
    _args = parser.parse_args()
    failures: list[str] = []
    failures.extend(check_changed_files())
    failures.extend(check_static_files())
    failures.extend(check_artifacts())
    payload = {"schema_version": "phase_139ys_checker_result_v1", "status": "pass" if not failures else "fail", "failures": failures}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
