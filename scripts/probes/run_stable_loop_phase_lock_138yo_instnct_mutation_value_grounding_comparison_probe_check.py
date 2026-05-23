#!/usr/bin/env python3
"""Checker for 138YO INSTNCT mutation value-grounding comparison probe."""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_138yo_instnct_mutation_value_grounding_comparison_probe/smoke"
RUNNER = "scripts/probes/run_stable_loop_phase_lock_138yo_instnct_mutation_value_grounding_comparison_probe.py"
CHECKER = "scripts/probes/run_stable_loop_phase_lock_138yo_instnct_mutation_value_grounding_comparison_probe_check.py"
DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_138YO_INSTNCT_MUTATION_VALUE_GROUNDING_COMPARISON_PROBE_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_138YO_INSTNCT_MUTATION_VALUE_GROUNDING_COMPARISON_PROBE_RESULT.md",
]
ALLOWED_MUTATIONS = {RUNNER, CHECKER, *DOCS}
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "upstream_138yn_manifest.json",
    "upstream_138yk_manifest.json",
    "analysis_config.json",
    "eval_dataset_manifest.json",
    "eval_rows.jsonl",
    "ast_shortcut_scan_report.json",
    "expected_output_canary_report.json",
    "forbidden_input_rejection_report.json",
    "instnct_checkpoint_manifest.json",
    "instnct_ablation_checkpoint_manifest.json",
    "byte_gru_raw_generation_results.jsonl",
    "instnct_raw_generation_results.jsonl",
    "instnct_ablation_raw_generation_results.jsonl",
    "byte_gru_scoring_results.jsonl",
    "instnct_scoring_results.jsonl",
    "instnct_ablation_scoring_results.jsonl",
    "byte_gru_value_grounding_metrics.json",
    "instnct_value_grounding_metrics.json",
    "instnct_ablation_value_grounding_metrics.json",
    "contrast_group_results.jsonl",
    "arm_comparison.json",
    "pocket_ablation_report.json",
    "control_results.jsonl",
    "control_arm_report.json",
    "generated_before_scoring_report.json",
    "determinism_replay_report.json",
    "aggregate_metrics.json",
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
ALLOWED_DECISIONS = {
    "instnct_adapter_prompt_bound_value_grounding_improves": "138YP_INSTNCT_MUTATION_POCKET_GATED_VALUE_GROUNDING_PLAN",
    "instnct_mutation_value_grounding_comparison_positive": "139YO_INSTNCT_MUTATION_VALUE_GROUNDING_SCALE_CONFIRM",
    "instnct_mutation_value_grounding_not_better_than_byte_gru": "138YOB_INSTNCT_MUTATION_COMPARISON_FAILURE_ANALYSIS",
    "nondeterministic_instnct_mutation_comparison": "138N_DETERMINISM_FAILURE_ANALYSIS",
    "scorer_or_task_weakness": "138E_REASONING_SCORER_OR_TASK_WEAKNESS_ANALYSIS",
}
ALLOWED_HELPER_KEYS = {"prompt", "checkpoint_path", "checkpoint_hash", "seed", "max_new_tokens", "generation_config"}


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
        if isinstance(node, ast.ImportFrom) and node.module == "torch":
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
        for term in ["138YO", "INSTNCT", "comparison", "helper"]:
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

    upstream_yn = load_json(SMOKE_ROOT / "upstream_138yn_manifest.json")
    upstream_yk = load_json(SMOKE_ROOT / "upstream_138yk_manifest.json")
    config = load_json(SMOKE_ROOT / "analysis_config.json")
    canary = load_json(SMOKE_ROOT / "expected_output_canary_report.json")
    ast_report = load_json(SMOKE_ROOT / "ast_shortcut_scan_report.json")
    generated = load_json(SMOKE_ROOT / "generated_before_scoring_report.json")
    replay = load_json(SMOKE_ROOT / "determinism_replay_report.json")
    controls = load_json(SMOKE_ROOT / "control_arm_report.json")
    comparison = load_json(SMOKE_ROOT / "arm_comparison.json")
    pocket = load_json(SMOKE_ROOT / "pocket_ablation_report.json")
    byte_metrics = load_json(SMOKE_ROOT / "byte_gru_value_grounding_metrics.json")
    instnct_metrics = load_json(SMOKE_ROOT / "instnct_value_grounding_metrics.json")
    decision = load_json(SMOKE_ROOT / "decision.json")
    summary = load_json(SMOKE_ROOT / "summary.json")
    traces = read_jsonl(SMOKE_ROOT / "instnct_raw_generation_results.jsonl")
    progress = read_jsonl(SMOKE_ROOT / "progress.jsonl")

    if upstream_yn.get("decision") != "instnct_mutation_raw_helper_adapter_probe_complete":
        failures.append("BAD_138YN_UPSTREAM")
    if upstream_yk.get("decision") != "family_default_shortcut_persists":
        failures.append("BAD_138YK_UPSTREAM")
    if config.get("training_performed") is not False or config.get("helper_modified") is not False:
        failures.append("BOUNDARY_CONFIG_NOT_ARTIFACT_COMPARISON")
    if canary.get("passed") is not True:
        failures.append("EXPECTED_OUTPUT_CANARY_FAILED")
    if ast_report.get("passed") is not True:
        failures.append("AST_SCAN_FAILED")
    if generated.get("passed") is not True:
        failures.append("GENERATED_BEFORE_SCORING_FAILED")
    if generated.get("all_helper_requests_allowed_keys_only") is not True:
        failures.append("HELPER_REQUEST_KEYS_NOT_STRICT")
    if replay.get("deterministic_replay_passed") is not True:
        failures.append("DETERMINISM_REPLAY_FAILED")
    if controls.get("controls_failed") is not True:
        failures.append("CONTROLS_DID_NOT_FAIL")
    if comparison.get("all_eval_rows_match") is not True:
        failures.append("EVAL_ROWS_NOT_MATCHED")
    if len(comparison.get("arms", [])) != 3:
        failures.append("ARM_COUNT_BAD")
    if byte_metrics.get("arm") != "byte_gru_138yk_target_existing_helper_rollout":
        failures.append("BYTE_ARM_BAD")
    if instnct_metrics.get("arm") != "instnct_mutation_adapter_138yn_same_prompt":
        failures.append("INSTNCT_ARM_BAD")
    if instnct_metrics.get("answer_value_accuracy", -1) <= byte_metrics.get("answer_value_accuracy", 999):
        failures.append("INSTNCT_DID_NOT_BEAT_BYTE_VALUE_ACCURACY")
    if pocket.get("pocket_writeback_decision_critical") is not False:
        failures.append("POCKET_DECISION_CRITICAL_OVERCLAIM")
    for trace in traces[:10]:
        request = trace.get("helper_request", {})
        if set(request) != ALLOWED_HELPER_KEYS:
            failures.append("INSTNCT_HELPER_REQUEST_KEYS_BAD")
        if trace.get("backend_name") != "repo_local_instnct_mutation_graph":
            failures.append("INSTNCT_BACKEND_BAD")
    if decision.get("decision") not in ALLOWED_DECISIONS:
        failures.append("BAD_DECISION")
    elif decision.get("next") != ALLOWED_DECISIONS[decision["decision"]]:
        failures.append("BAD_NEXT")
    if decision.get("architecture_superiority_claimed") is not False:
        failures.append("ARCHITECTURE_SUPERIORITY_OVERCLAIM")
    if decision.get("pocket_mechanism_claimed") is not False:
        failures.append("POCKET_MECHANISM_OVERCLAIM")
    require_false_flags(decision, failures)
    require_false_flags(summary, failures)
    events = {row.get("event") for row in progress}
    for event in [
        "startup",
        "upstream verification",
        "eval row load",
        "instnct manifest bind",
        "byte baseline artifact load",
        "instnct generation complete",
        "pocket ablation generation complete",
        "scoring",
        "arm comparison",
        "determinism replay",
        "decision",
        "final verdict",
    ]:
        if event not in events:
            failures.append(f"PROGRESS_EVENT_MISSING:{event}")
    return sorted(set(failures))


def main() -> int:
    parser = argparse.ArgumentParser(description="Check 138YO comparison probe")
    parser.add_argument("--check-only", action="store_true")
    _args = parser.parse_args()
    failures: list[str] = []
    failures.extend(check_changed_files())
    failures.extend(check_static_files())
    failures.extend(check_artifacts())
    payload = {"schema_version": "phase_138yo_checker_result_v1", "status": "pass" if not failures else "fail", "failures": failures}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
