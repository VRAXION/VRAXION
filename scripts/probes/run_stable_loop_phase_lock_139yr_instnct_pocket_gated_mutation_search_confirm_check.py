#!/usr/bin/env python3
"""Checker for 139YR deterministic mutation-search confirm."""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_139yr_instnct_pocket_gated_mutation_search_confirm/smoke"
RUNNER = "scripts/probes/run_stable_loop_phase_lock_139yr_instnct_pocket_gated_mutation_search_confirm.py"
CHECKER = "scripts/probes/run_stable_loop_phase_lock_139yr_instnct_pocket_gated_mutation_search_confirm_check.py"
DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_139YR_INSTNCT_POCKET_GATED_MUTATION_SEARCH_CONFIRM_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_139YR_INSTNCT_POCKET_GATED_MUTATION_SEARCH_CONFIRM_RESULT.md",
]
ALLOWED_MUTATIONS = {RUNNER, CHECKER, *DOCS}
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "upstream_139yq_manifest.json",
    "helper_provenance_verification.json",
    "ast_shortcut_scan_report.json",
    "eval_dataset_manifest.json",
    "eval_rows.jsonl",
    "mutation_candidate_results.jsonl",
    "mutation_search_trace.jsonl",
    "raw_generation_results.jsonl",
    "pocket_trace.jsonl",
    "selection_report.json",
    "fitness_landscape.json",
    "control_results.jsonl",
    "control_arm_report.json",
    "expected_output_canary_report.json",
    "forbidden_input_rejection_report.json",
    "generated_before_scoring_report.json",
    "determinism_replay_report.json",
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
        for term in ["139YR", "mutation", "pocket", "helper"]:
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

    upstream = load_json(SMOKE_ROOT / "upstream_139yq_manifest.json")
    helper = load_json(SMOKE_ROOT / "helper_provenance_verification.json")
    dataset = load_json(SMOKE_ROOT / "eval_dataset_manifest.json")
    ast_report = load_json(SMOKE_ROOT / "ast_shortcut_scan_report.json")
    candidates = read_jsonl(SMOKE_ROOT / "mutation_candidate_results.jsonl")
    selection = load_json(SMOKE_ROOT / "selection_report.json")
    controls = load_json(SMOKE_ROOT / "control_arm_report.json")
    canary = load_json(SMOKE_ROOT / "expected_output_canary_report.json")
    generated = load_json(SMOKE_ROOT / "generated_before_scoring_report.json")
    replay = load_json(SMOKE_ROOT / "determinism_replay_report.json")
    decision = load_json(SMOKE_ROOT / "decision.json")
    summary = load_json(SMOKE_ROOT / "summary.json")
    traces = read_jsonl(SMOKE_ROOT / "mutation_search_trace.jsonl")
    progress = read_jsonl(SMOKE_ROOT / "progress.jsonl")

    if upstream.get("decision") != "instnct_pocket_gated_value_grounding_scale_confirmed":
        failures.append("BAD_139YQ_UPSTREAM")
    if helper.get("strict_pocket_gated_symbols_present") is not True:
        failures.append("HELPER_STRICT_POCKET_SYMBOLS_MISSING")
    if ast_report.get("passed") is not True:
        failures.append("AST_SCAN_FAILED")
    if dataset.get("row_count", 0) < 300:
        failures.append("MUTATION_EVAL_TOO_SMALL")
    if len(candidates) < 5:
        failures.append("TOO_FEW_CANDIDATES")
    selected = next((row for row in candidates if row.get("candidate") == selection.get("selected_candidate")), None)
    if not selected:
        failures.append("SELECTED_CANDIDATE_MISSING")
    else:
        if selected.get("candidate") != "open_pocket_all_payload_markers":
            failures.append("WRONG_SELECTED_CANDIDATE")
        if selected.get("answer_value_accuracy", 0.0) < 0.95:
            failures.append("SELECTED_ACCURACY_TOO_LOW")
        if selected.get("pocket_writeback_rate", 0.0) < 0.95:
            failures.append("SELECTED_WRITEBACK_TOO_LOW")
    if selection.get("gradient_used") is not False:
        failures.append("GRADIENT_USED")
    if selection.get("fitness_margin", 0.0) < 0.40:
        failures.append("FITNESS_MARGIN_TOO_LOW")
    if controls.get("controls_failed") is not True:
        failures.append("CONTROLS_DID_NOT_FAIL")
    if canary.get("passed") is not True:
        failures.append("CANARY_FAILED")
    if generated.get("passed") is not True or generated.get("all_helper_requests_allowed_keys_only") is not True:
        failures.append("GENERATED_BEFORE_SCORING_FAILED")
    if replay.get("deterministic_replay_passed") is not True:
        failures.append("DETERMINISM_FAILED")
    for trace in traces[:10]:
        request = trace.get("helper_request", {})
        if set(request) != ALLOWED_HELPER_KEYS:
            failures.append("HELPER_REQUEST_KEYS_BAD")
    if decision.get("decision") != "instnct_pocket_gated_mutation_search_confirmed":
        failures.append("BAD_DECISION")
    if decision.get("next") != "139YS_INSTNCT_POCKET_GATED_MUTATION_SEARCH_SCALE_OR_REAL_TASK_BRIDGE":
        failures.append("BAD_NEXT")
    if decision.get("gradient_used") is not False or decision.get("architecture_superiority_claimed") is not False or decision.get("value_grounding_claimed") is not False:
        failures.append("BROAD_OR_GRADIENT_OVERCLAIM")
    require_false_flags(decision, failures)
    require_false_flags(summary, failures)
    events = {row.get("event") for row in progress}
    for event in [
        "startup",
        "upstream verification",
        "helper provenance and ast",
        "eval row build",
        "candidate evaluated",
        "selection",
        "controls and canary",
        "decision",
        "final verdict",
    ]:
        if event not in events:
            failures.append(f"PROGRESS_EVENT_MISSING:{event}")
    return sorted(set(failures))


def main() -> int:
    parser = argparse.ArgumentParser(description="Check 139YR mutation-search confirm")
    parser.add_argument("--check-only", action="store_true")
    _args = parser.parse_args()
    failures: list[str] = []
    failures.extend(check_changed_files())
    failures.extend(check_static_files())
    failures.extend(check_artifacts())
    payload = {"schema_version": "phase_139yr_checker_result_v1", "status": "pass" if not failures else "fail", "failures": failures}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
