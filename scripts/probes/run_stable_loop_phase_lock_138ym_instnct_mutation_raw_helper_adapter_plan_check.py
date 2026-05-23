#!/usr/bin/env python3
"""Checker for 138YM INSTNCT mutation raw-helper adapter plan."""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_138ym_instnct_mutation_raw_helper_adapter_plan/smoke"
RUNNER = "scripts/probes/run_stable_loop_phase_lock_138ym_instnct_mutation_raw_helper_adapter_plan.py"
CHECKER = "scripts/probes/run_stable_loop_phase_lock_138ym_instnct_mutation_raw_helper_adapter_plan_check.py"
DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_138YM_INSTNCT_MUTATION_RAW_HELPER_ADAPTER_PLAN_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_138YM_INSTNCT_MUTATION_RAW_HELPER_ADAPTER_PLAN_RESULT.md",
]
ALLOWED_MUTATIONS = {RUNNER, CHECKER, *DOCS}
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "upstream_138yl_manifest.json",
    "analysis_config.json",
    "adapter_contract.json",
    "helper_surface_change_plan.json",
    "instnct_checkpoint_contract.json",
    "prompt_encoder_contract.json",
    "iterative_propagation_schedule.json",
    "output_decoder_contract.json",
    "forbidden_metadata_policy.json",
    "canary_and_ast_gate_plan.json",
    "determinism_plan.json",
    "comparison_eval_plan.json",
    "target_138yn_milestone_plan.json",
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
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in {"torch", "shared_raw_generation_helper"}:
                    failures.append(f"FORBIDDEN_IMPORT:{alias.name}:{path.name}")
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module in {"torch", "shared_raw_generation_helper"} or module.startswith("run_stable_loop_phase_lock_"):
                failures.append(f"FORBIDDEN_IMPORT_FROM:{module}:{path.name}")
        if isinstance(node, ast.Call):
            name = ast.unparse(node.func)
            if any(token in name for token in ["raw_generate", "load_checkpoint", "backward", "optimizer", "manual_seed", "forward"]):
                failures.append(f"FORBIDDEN_CALL:{name}:{path.name}")
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
        if rel in DOCS and len(text.strip()) < 300:
            failures.append(f"DOC_TOO_SHORT:{rel}")
        for term in ["artifact-only", "INSTNCT", "helper", "not GPT-like readiness"]:
            if rel in DOCS and term not in text:
                failures.append(f"DOC_TERM_MISSING:{rel}:{term}")
    failures.extend(ast_scan(REPO_ROOT / RUNNER))
    failures.extend(ast_scan(REPO_ROOT / CHECKER))
    return sorted(set(failures))


def check_artifacts() -> list[str]:
    failures: list[str] = []
    for name in REQUIRED_ARTIFACTS:
        if not (SMOKE_ROOT / name).exists():
            failures.append(f"MISSING_ARTIFACT:{name}")
    if failures:
        return failures

    config = load_json(SMOKE_ROOT / "analysis_config.json")
    upstream = load_json(SMOKE_ROOT / "upstream_138yl_manifest.json")
    adapter = load_json(SMOKE_ROOT / "adapter_contract.json")
    helper_plan = load_json(SMOKE_ROOT / "helper_surface_change_plan.json")
    propagation = load_json(SMOKE_ROOT / "iterative_propagation_schedule.json")
    decoder = load_json(SMOKE_ROOT / "output_decoder_contract.json")
    canary = load_json(SMOKE_ROOT / "canary_and_ast_gate_plan.json")
    determinism = load_json(SMOKE_ROOT / "determinism_plan.json")
    comparison = load_json(SMOKE_ROOT / "comparison_eval_plan.json")
    target = load_json(SMOKE_ROOT / "target_138yn_milestone_plan.json")
    decision = load_json(SMOKE_ROOT / "decision.json")
    summary = load_json(SMOKE_ROOT / "summary.json")
    progress = read_jsonl(SMOKE_ROOT / "progress.jsonl")

    for key in ["artifact_only", "planning_only"]:
        if config.get(key) is not True:
            failures.append(f"CONFIG_NOT_TRUE:{key}")
    for key in ["training_performed", "new_helper_inference_run", "shared_helper_called", "torch_forward_pass_run", "checkpoint_mutated", "helper_backend_modified"]:
        if config.get(key) is not False:
            failures.append(f"BOUNDARY_FAILURE:{key}")

    if upstream.get("decision") != "instnct_mutation_helper_integration_analysis_complete":
        failures.append("UPSTREAM_138YL_BAD_DECISION")
    if adapter.get("backend_name") != "repo_local_instnct_mutation_graph":
        failures.append("BAD_ADAPTER_BACKEND")
    if "expected_output" not in adapter.get("forbidden_request_material", []):
        failures.append("FORBIDDEN_METADATA_POLICY_TOO_WEAK")
    if helper_plan.get("selected_strategy") != "backend_dispatch_extension_after_contract":
        failures.append("BAD_HELPER_STRATEGY")
    if propagation.get("supports_iterative_refinement") is not True:
        failures.append("ITERATIVE_REFINEMENT_NOT_PLANNED")
    if decoder.get("may_not_use_expected_output") is not True:
        failures.append("DECODER_EXPECTED_OUTPUT_GAP")
    if "expected-output canary" not in canary.get("required_gates", []):
        failures.append("CANARY_GATE_MISSING")
    if "generated_text_hashes" not in determinism.get("replay_must_match", []):
        failures.append("DETERMINISM_TOO_WEAK")
    if "same eval rows" not in comparison.get("fair_comparison_requirements", []):
        failures.append("COMPARISON_NOT_FAIR")
    if target.get("milestone") != "138YN_INSTNCT_MUTATION_RAW_HELPER_ADAPTER_PROBE":
        failures.append("BAD_TARGET_MILESTONE")
    if target.get("helper_backend_modification_allowed") is not True:
        failures.append("TARGET_DOES_NOT_ALLOW_ADAPTER_PROBE")
    if target.get("train_allowed") is not False:
        failures.append("TARGET_TRAINING_SHOULD_BE_FALSE")

    if decision.get("decision") != "instnct_mutation_raw_helper_adapter_plan_complete":
        failures.append("BAD_DECISION")
    if decision.get("next") != "138YN_INSTNCT_MUTATION_RAW_HELPER_ADAPTER_PROBE":
        failures.append("BAD_NEXT")
    if decision.get("helper_modified_in_138ym") is not False:
        failures.append("HELPER_MODIFIED_IN_PLAN")
    require_false_flags(decision, failures)
    require_false_flags(summary, failures)

    events = {row.get("event") for row in progress}
    for event in [
        "startup",
        "upstream verification",
        "adapter contract",
        "helper surface change plan",
        "iterative propagation schedule",
        "target 138yn milestone plan",
        "decision",
        "final verdict",
    ]:
        if event not in events:
            failures.append(f"PROGRESS_EVENT_MISSING:{event}")
    return sorted(set(failures))


def main() -> int:
    parser = argparse.ArgumentParser(description="Check 138YM adapter plan")
    parser.add_argument("--check-only", action="store_true")
    _args = parser.parse_args()
    failures = []
    failures.extend(check_changed_files())
    failures.extend(check_static_files())
    failures.extend(check_artifacts())
    payload = {"schema_version": "phase_138ym_checker_result_v1", "status": "pass" if not failures else "fail", "failures": failures}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
