#!/usr/bin/env python3
"""Checker for 138YN INSTNCT mutation raw-helper adapter probe."""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_138yn_instnct_mutation_raw_helper_adapter_probe/smoke"
HELPER = "scripts/probes/shared_raw_generation_helper.py"
RUNNER = "scripts/probes/run_stable_loop_phase_lock_138yn_instnct_mutation_raw_helper_adapter_probe.py"
CHECKER = "scripts/probes/run_stable_loop_phase_lock_138yn_instnct_mutation_raw_helper_adapter_probe_check.py"
DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_138YN_INSTNCT_MUTATION_RAW_HELPER_ADAPTER_PROBE_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_138YN_INSTNCT_MUTATION_RAW_HELPER_ADAPTER_PROBE_RESULT.md",
]
ALLOWED_MUTATIONS = {HELPER, RUNNER, CHECKER, *DOCS}
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "upstream_138ym_manifest.json",
    "adapter_contract.json",
    "helper_provenance_verification.json",
    "forbidden_input_rejection_report.json",
    "expected_output_canary_report.json",
    "ast_shortcut_scan_report.json",
    "instnct_checkpoint_manifest.json",
    "prompt_encoder_trace.jsonl",
    "iterative_propagation_trace.jsonl",
    "raw_generation_trace.jsonl",
    "raw_generation_results.jsonl",
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
    for rel in [HELPER, RUNNER, CHECKER, *DOCS]:
        path = REPO_ROOT / rel
        if not path.exists():
            failures.append(f"MISSING_TRACKED_FILE:{rel}")
            continue
        text = path.read_text(encoding="utf-8")
        if rel == HELPER:
            for term in ["INSTNCT_MUTATION_BACKEND", "repo_local_instnct_mutation_graph", "load_instnct_checkpoint_manifest", "instnct_raw_generate"]:
                if term not in text:
                    failures.append(f"HELPER_TERM_MISSING:{term}")
        if rel in DOCS:
            for term in ["INSTNCT", "adapter", "helper", "not GPT-like readiness"]:
                if term not in text:
                    failures.append(f"DOC_TERM_MISSING:{rel}:{term}")
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

    upstream = load_json(SMOKE_ROOT / "upstream_138ym_manifest.json")
    helper = load_json(SMOKE_ROOT / "helper_provenance_verification.json")
    forbidden = load_json(SMOKE_ROOT / "forbidden_input_rejection_report.json")
    canary = load_json(SMOKE_ROOT / "expected_output_canary_report.json")
    ast_report = load_json(SMOKE_ROOT / "ast_shortcut_scan_report.json")
    checkpoint = load_json(SMOKE_ROOT / "instnct_checkpoint_manifest.json")
    generated = load_json(SMOKE_ROOT / "generated_before_scoring_report.json")
    replay = load_json(SMOKE_ROOT / "determinism_replay_report.json")
    decision = load_json(SMOKE_ROOT / "decision.json")
    summary = load_json(SMOKE_ROOT / "summary.json")
    traces = read_jsonl(SMOKE_ROOT / "raw_generation_trace.jsonl")
    progress = read_jsonl(SMOKE_ROOT / "progress.jsonl")

    if upstream.get("decision") != "instnct_mutation_raw_helper_adapter_plan_complete":
        failures.append("UPSTREAM_138YM_BAD_DECISION")
    if helper.get("adapter_backend_available") is not True:
        failures.append("ADAPTER_BACKEND_NOT_AVAILABLE")
    if forbidden.get("passed") is not True:
        failures.append("FORBIDDEN_INPUT_REJECTION_FAILED")
    if canary.get("passed") is not True:
        failures.append("EXPECTED_OUTPUT_CANARY_FAILED")
    if ast_report.get("passed") is not True:
        failures.append("AST_SCAN_FAILED")
    if checkpoint.get("backend_name") != "repo_local_instnct_mutation_graph":
        failures.append("BAD_CHECKPOINT_BACKEND")
    if generated.get("passed") is not True:
        failures.append("GENERATED_BEFORE_SCORING_FAILED")
    if replay.get("deterministic_replay_passed") is not True:
        failures.append("DETERMINISM_REPLAY_FAILED")
    for trace in traces:
        request = trace.get("helper_request", {})
        if set(request) != ALLOWED_HELPER_KEYS:
            failures.append("HELPER_REQUEST_KEYS_NOT_STRICT")
        response = trace.get("helper_response", {})
        if response.get("backend_name") != "repo_local_instnct_mutation_graph":
            failures.append("TRACE_BACKEND_NOT_INSTNCT")
        if "generated_text" not in trace:
            failures.append("TRACE_MISSING_GENERATED_TEXT")
    if decision.get("decision") != "instnct_mutation_raw_helper_adapter_probe_complete":
        failures.append("BAD_DECISION")
    if decision.get("next") != "138YO_INSTNCT_MUTATION_VALUE_GROUNDING_COMPARISON_PROBE":
        failures.append("BAD_NEXT")
    if decision.get("value_grounding_claimed") is not False:
        failures.append("VALUE_GROUNDING_CLAIMED_TOO_EARLY")
    require_false_flags(decision, failures)
    require_false_flags(summary, failures)
    events = {row.get("event") for row in progress}
    for event in [
        "startup",
        "upstream verification",
        "adapter manifest build",
        "helper provenance",
        "forbidden input rejection",
        "expected output canary",
        "ast shortcut scan",
        "final eval generation",
        "scoring",
        "determinism replay",
        "decision",
        "final verdict",
    ]:
        if event not in events:
            failures.append(f"PROGRESS_EVENT_MISSING:{event}")
    return sorted(set(failures))


def main() -> int:
    parser = argparse.ArgumentParser(description="Check 138YN adapter probe")
    parser.add_argument("--check-only", action="store_true")
    _args = parser.parse_args()
    failures = []
    failures.extend(check_changed_files())
    failures.extend(check_static_files())
    failures.extend(check_artifacts())
    payload = {"schema_version": "phase_138yn_checker_result_v1", "status": "pass" if not failures else "fail", "failures": failures}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
