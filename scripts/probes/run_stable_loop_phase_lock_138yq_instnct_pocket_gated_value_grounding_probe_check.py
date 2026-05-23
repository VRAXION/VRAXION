#!/usr/bin/env python3
"""Checker for 138YQ pocket-gated INSTNCT value-grounding probe."""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_138yq_instnct_pocket_gated_value_grounding_probe/smoke"
HELPER = "scripts/probes/shared_raw_generation_helper.py"
RUNNER = "scripts/probes/run_stable_loop_phase_lock_138yq_instnct_pocket_gated_value_grounding_probe.py"
CHECKER = "scripts/probes/run_stable_loop_phase_lock_138yq_instnct_pocket_gated_value_grounding_probe_check.py"
DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_138YQ_INSTNCT_POCKET_GATED_VALUE_GROUNDING_PROBE_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_138YQ_INSTNCT_POCKET_GATED_VALUE_GROUNDING_PROBE_RESULT.md",
]
ALLOWED_MUTATIONS = {HELPER, RUNNER, CHECKER, *DOCS}
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "upstream_138yp_manifest.json",
    "adapter_contract.json",
    "helper_provenance_verification.json",
    "instnct_pocket_gated_manifest.json",
    "instnct_pocket_gated_ablation_manifest.json",
    "eval_rows.jsonl",
    "pocket_payload_manifest.json",
    "raw_generation_trace.jsonl",
    "raw_generation_results.jsonl",
    "pocket_trace.jsonl",
    "pocket_ablation_results.jsonl",
    "scoring_results.jsonl",
    "contrast_group_results.jsonl",
    "pocket_gating_metrics.json",
    "arm_comparison.json",
    "control_results.jsonl",
    "control_arm_report.json",
    "mutation_candidate_results.jsonl",
    "expected_output_canary_report.json",
    "forbidden_input_rejection_report.json",
    "ast_shortcut_scan_report.json",
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


def ast_scan(path: Path, allow_torch: bool = False) -> list[str]:
    failures: list[str] = []
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("run_stable_loop_phase_lock_"):
            failures.append(f"OLD_RUNNER_IMPORT:{path.name}")
        if not allow_torch and isinstance(node, ast.Import) and any(alias.name == "torch" for alias in node.names):
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
    for rel in [HELPER, RUNNER, CHECKER, *DOCS]:
        path = REPO_ROOT / rel
        if not path.exists():
            failures.append(f"MISSING_TRACKED_FILE:{rel}")
            continue
        text = path.read_text(encoding="utf-8")
        if rel == HELPER:
            for term in ["value_selection_requires_open_pocket", "_instnct_select_open_pocket_value", "closed_pocket_fallback"]:
                if term not in text:
                    failures.append(f"HELPER_TERM_MISSING:{term}")
        if rel in DOCS and "not GPT-like readiness" not in text:
            failures.append(f"DOC_BOUNDARY_TERM_MISSING:{rel}")
        if path.suffix == ".py":
            failures.extend(ast_scan(path, allow_torch=(rel == HELPER)))
    return sorted(set(failures))


def check_artifacts() -> list[str]:
    failures: list[str] = []
    for name in REQUIRED_ARTIFACTS:
        if not (SMOKE_ROOT / name).exists():
            failures.append(f"MISSING_ARTIFACT:{name}")
    if failures:
        return failures

    upstream = load_json(SMOKE_ROOT / "upstream_138yp_manifest.json")
    helper = load_json(SMOKE_ROOT / "helper_provenance_verification.json")
    main_manifest = load_json(SMOKE_ROOT / "instnct_pocket_gated_manifest.json")
    ablation_manifest = load_json(SMOKE_ROOT / "instnct_pocket_gated_ablation_manifest.json")
    canary = load_json(SMOKE_ROOT / "expected_output_canary_report.json")
    ast_report = load_json(SMOKE_ROOT / "ast_shortcut_scan_report.json")
    generated = load_json(SMOKE_ROOT / "generated_before_scoring_report.json")
    replay = load_json(SMOKE_ROOT / "determinism_replay_report.json")
    controls = load_json(SMOKE_ROOT / "control_arm_report.json")
    metrics = load_json(SMOKE_ROOT / "pocket_gating_metrics.json")
    comparison = load_json(SMOKE_ROOT / "arm_comparison.json")
    decision = load_json(SMOKE_ROOT / "decision.json")
    summary = load_json(SMOKE_ROOT / "summary.json")
    traces = read_jsonl(SMOKE_ROOT / "raw_generation_trace.jsonl")
    pocket_traces = read_jsonl(SMOKE_ROOT / "pocket_trace.jsonl")
    mutation = read_jsonl(SMOKE_ROOT / "mutation_candidate_results.jsonl")
    progress = read_jsonl(SMOKE_ROOT / "progress.jsonl")

    if upstream.get("decision") != "instnct_mutation_pocket_gated_value_grounding_plan_complete":
        failures.append("BAD_138YP_UPSTREAM")
    if helper.get("strict_pocket_gated_symbols_present") is not True:
        failures.append("HELPER_STRICT_POCKET_SYMBOLS_MISSING")
    if main_manifest.get("value_selection_requires_open_pocket") is not True:
        failures.append("MAIN_MANIFEST_NOT_POCKET_GATED")
    if main_manifest.get("visible_value_bypass_forbidden") is not True:
        failures.append("VISIBLE_BYPASS_NOT_FORBIDDEN")
    if ablation_manifest.get("pockets", [{}])[0].get("gate_marker") != "GATE:NEVER_OPEN":
        failures.append("ABLATION_GATE_NOT_NEVER_OPEN")
    if canary.get("passed") is not True:
        failures.append("EXPECTED_OUTPUT_CANARY_FAILED")
    if ast_report.get("passed") is not True:
        failures.append("AST_SCAN_FAILED")
    if generated.get("passed") is not True or generated.get("all_helper_requests_allowed_keys_only") is not True:
        failures.append("GENERATED_BEFORE_SCORING_FAILED")
    if replay.get("deterministic_replay_passed") is not True:
        failures.append("DETERMINISM_REPLAY_FAILED")
    if controls.get("controls_failed") is not True:
        failures.append("CONTROLS_DID_NOT_FAIL")
    main = metrics.get("main", {})
    ablation = metrics.get("ablation", {})
    if main.get("answer_value_accuracy", 0.0) < 0.25:
        failures.append("MAIN_ACCURACY_TOO_LOW")
    if main.get("pocket_writeback_rate", 0.0) < 0.95:
        failures.append("MAIN_POCKET_WRITEBACK_TOO_LOW")
    if main.get("phase_transport_success_rate", 0.0) < 0.95:
        failures.append("MAIN_PHASE_TRANSPORT_TOO_LOW")
    if ablation.get("answer_value_accuracy", 1.0) > 0.05:
        failures.append("ABLATION_ACCURACY_TOO_HIGH")
    if comparison.get("pocket_ablation_delta_answer_value_accuracy", 0.0) < 0.20:
        failures.append("ABLATION_DELTA_TOO_LOW")
    if comparison.get("pocket_ablation_decision_critical") is not True:
        failures.append("ABLATION_NOT_DECISION_CRITICAL")
    for trace in traces[:10]:
        request = trace.get("helper_request", {})
        if set(request) != ALLOWED_HELPER_KEYS:
            failures.append("HELPER_REQUEST_KEYS_BAD")
        if trace.get("backend_name") != "repo_local_instnct_mutation_graph":
            failures.append("TRACE_BACKEND_BAD")
    if not pocket_traces or not all(row.get("highway_retained") is True for row in pocket_traces):
        failures.append("HIGHWAY_RETENTION_BAD")
    selected = [row for row in mutation if row.get("selected") is True]
    if len(selected) != 1 or selected[0].get("candidate") != "instnct_pocket_gated_main":
        failures.append("MUTATION_SELECTION_BAD")
    if decision.get("decision") != "instnct_pocket_gated_value_grounding_probe_positive":
        failures.append("BAD_DECISION")
    if decision.get("next") != "139YQ_INSTNCT_POCKET_GATED_VALUE_GROUNDING_SCALE_CONFIRM":
        failures.append("BAD_NEXT")
    if decision.get("pocket_mechanism_claimed") is not True:
        failures.append("POCKET_MECHANISM_NOT_CLAIMED_FOR_CONSTRAINED_PROBE")
    if decision.get("architecture_superiority_claimed") is not False or decision.get("value_grounding_claimed") is not False:
        failures.append("BROAD_OVERCLAIM")
    require_false_flags(decision, failures)
    require_false_flags(summary, failures)
    events = {row.get("event") for row in progress}
    for event in [
        "startup",
        "upstream verification",
        "helper provenance",
        "manifest build",
        "canary and ast",
        "eval row build",
        "generation",
        "scoring and comparison",
        "controls",
        "determinism replay",
        "decision",
        "final verdict",
    ]:
        if event not in events:
            failures.append(f"PROGRESS_EVENT_MISSING:{event}")
    return sorted(set(failures))


def main() -> int:
    parser = argparse.ArgumentParser(description="Check 138YQ pocket-gated probe")
    parser.add_argument("--check-only", action="store_true")
    _args = parser.parse_args()
    failures: list[str] = []
    failures.extend(check_changed_files())
    failures.extend(check_static_files())
    failures.extend(check_artifacts())
    payload = {"schema_version": "phase_138yq_checker_result_v1", "status": "pass" if not failures else "fail", "failures": failures}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
