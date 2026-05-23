#!/usr/bin/env python3
"""Checker for 138YP INSTNCT pocket-gated value-grounding plan."""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_138yp_instnct_mutation_pocket_gated_value_grounding_plan/smoke"
RUNNER = "scripts/probes/run_stable_loop_phase_lock_138yp_instnct_mutation_pocket_gated_value_grounding_plan.py"
CHECKER = "scripts/probes/run_stable_loop_phase_lock_138yp_instnct_mutation_pocket_gated_value_grounding_plan_check.py"
DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_138YP_INSTNCT_MUTATION_POCKET_GATED_VALUE_GROUNDING_PLAN_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_138YP_INSTNCT_MUTATION_POCKET_GATED_VALUE_GROUNDING_PLAN_RESULT.md",
]
ALLOWED_MUTATIONS = {RUNNER, CHECKER, *DOCS}
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "upstream_138yo_manifest.json",
    "analysis_config.json",
    "ast_shortcut_scan_report.json",
    "pocket_bypass_diagnosis.json",
    "pocket_gating_requirements.json",
    "target_138yq_backend_contract.json",
    "target_138yq_eval_design.json",
    "pocket_ablation_gate_spec.json",
    "mutation_credit_assignment_bridge.json",
    "anti_shortcut_requirements.json",
    "target_138yq_failure_routes.json",
    "next_138yq_milestone_plan.json",
    "diagnostic_gap_register.json",
    "risk_register.json",
    "decision.json",
    "summary.json",
    "report.md",
]
FALSE_FLAGS = [
    "reasoning_restored",
    "reasoning_subtrack_real_raw_evidence_partially_restored",
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
        for term in ["138YP", "pocket", "writeback", "helper"]:
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

    upstream = load_json(SMOKE_ROOT / "upstream_138yo_manifest.json")
    config = load_json(SMOKE_ROOT / "analysis_config.json")
    ast_report = load_json(SMOKE_ROOT / "ast_shortcut_scan_report.json")
    bypass = load_json(SMOKE_ROOT / "pocket_bypass_diagnosis.json")
    gating = load_json(SMOKE_ROOT / "pocket_gating_requirements.json")
    backend = load_json(SMOKE_ROOT / "target_138yq_backend_contract.json")
    ablation = load_json(SMOKE_ROOT / "pocket_ablation_gate_spec.json")
    bridge = load_json(SMOKE_ROOT / "mutation_credit_assignment_bridge.json")
    plan = load_json(SMOKE_ROOT / "next_138yq_milestone_plan.json")
    gaps = load_json(SMOKE_ROOT / "diagnostic_gap_register.json")
    decision = load_json(SMOKE_ROOT / "decision.json")
    summary = load_json(SMOKE_ROOT / "summary.json")
    progress = read_jsonl(SMOKE_ROOT / "progress.jsonl")

    if upstream.get("decision") != "instnct_adapter_prompt_bound_value_grounding_improves":
        failures.append("BAD_138YO_DECISION")
    if upstream.get("instnct_pocket_writeback_rate") != 0.0:
        failures.append("BAD_138YO_POCKET_WRITEBACK_PROFILE")
    if upstream.get("pocket_ablation_delta_answer_value_accuracy") != 0.0:
        failures.append("BAD_138YO_ABLATION_PROFILE")
    for key in ["artifact_only", "planning_only"]:
        if config.get(key) is not True:
            failures.append(f"CONFIG_NOT_PLANNING_ONLY:{key}")
    for key in ["training_performed", "new_model_inference_run", "shared_helper_called", "checkpoint_mutated", "helper_modified"]:
        if config.get(key) is not False:
            failures.append(f"BOUNDARY_CONFIG_BAD:{key}")
    if ast_report.get("passed") is not True:
        failures.append("AST_SCAN_FAILED")
    if bypass.get("root_cause") != "prompt_bound_value_extraction_bypasses_pocket_writeback":
        failures.append("BAD_BYPASS_ROOT")
    if gating.get("minimum_positive_gates", {}).get("pocket_writeback_rate") < 0.95:
        failures.append("POCKET_WRITEBACK_GATE_TOO_WEAK")
    if gating.get("minimum_positive_gates", {}).get("pocket_ablation_delta_answer_value_accuracy") < 0.20:
        failures.append("ABLATION_DELTA_GATE_TOO_WEAK")
    if backend.get("new_manifest_fields_allowed_in_138yq", {}).get("value_selection_requires_open_pocket") is not True:
        failures.append("BACKEND_CONTRACT_DOES_NOT_REQUIRE_OPEN_POCKET")
    if backend.get("public_api_change_allowed", True) is not False:
        failures.append("PUBLIC_API_CHANGE_ALLOWED")
    if "decision-critical" not in ablation.get("pass_condition", ""):
        failures.append("ABLATION_PASS_CONDITION_NOT_DECISION_CRITICAL")
    if bridge.get("gradient_used") is not False:
        failures.append("MUTATION_BRIDGE_USES_GRADIENT")
    if plan.get("milestone") != "138YQ_INSTNCT_POCKET_GATED_VALUE_GROUNDING_PROBE":
        failures.append("BAD_TARGET_MILESTONE")
    if plan.get("helper_backend_modification_allowed") is not True:
        failures.append("TARGET_PLAN_MISSING_HELPER_BACKEND_MOD_ALLOWED")
    if plan.get("mutation_allowed") is not True:
        failures.append("TARGET_PLAN_MISSING_MUTATION_ALLOWED")
    if not gaps.get("gaps"):
        failures.append("DIAGNOSTIC_GAPS_MISSING")
    if decision.get("decision") != "instnct_mutation_pocket_gated_value_grounding_plan_complete":
        failures.append("BAD_DECISION")
    if decision.get("next") != "138YQ_INSTNCT_POCKET_GATED_VALUE_GROUNDING_PROBE":
        failures.append("BAD_NEXT")
    if decision.get("architecture_superiority_claimed") is not False or decision.get("pocket_mechanism_claimed") is not False:
        failures.append("OVERCLAIM_IN_DECISION")
    require_false_flags(decision, failures)
    require_false_flags(summary, failures)
    events = {row.get("event") for row in progress}
    for event in [
        "startup",
        "upstream verification",
        "artifact loading",
        "pocket bypass diagnosis",
        "pocket gating requirements",
        "target 138YQ plan writing",
        "decision",
        "final verdict",
    ]:
        if event not in events:
            failures.append(f"PROGRESS_EVENT_MISSING:{event}")
    return sorted(set(failures))


def main() -> int:
    parser = argparse.ArgumentParser(description="Check 138YP plan")
    parser.add_argument("--check-only", action="store_true")
    _args = parser.parse_args()
    failures: list[str] = []
    failures.extend(check_changed_files())
    failures.extend(check_static_files())
    failures.extend(check_artifacts())
    payload = {"schema_version": "phase_138yp_checker_result_v1", "status": "pass" if not failures else "fail", "failures": failures}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
