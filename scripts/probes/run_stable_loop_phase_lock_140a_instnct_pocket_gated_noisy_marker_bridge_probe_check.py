#!/usr/bin/env python3
"""Checker for 140A noisy-marker bridge probe."""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_140a_instnct_pocket_gated_noisy_marker_bridge_probe/smoke"
RUNNER = "scripts/probes/run_stable_loop_phase_lock_140a_instnct_pocket_gated_noisy_marker_bridge_probe.py"
CHECKER = "scripts/probes/run_stable_loop_phase_lock_140a_instnct_pocket_gated_noisy_marker_bridge_probe_check.py"
DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_140A_INSTNCT_POCKET_GATED_NOISY_MARKER_BRIDGE_PROBE_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_140A_INSTNCT_POCKET_GATED_NOISY_MARKER_BRIDGE_PROBE_RESULT.md",
]
ALLOWED_MUTATIONS = {RUNNER, CHECKER, *DOCS}
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "upstream_139ys_manifest.json",
    "eval_config.json",
    "helper_provenance_verification.json",
    "ast_shortcut_scan_report.json",
    "expected_output_canary_report.json",
    "forbidden_input_rejection_report.json",
    "noisy_bridge_eval_manifest.json",
    "bridge_prompt_scaffold_manifest.json",
    "eval_rows.jsonl",
    "mutation_candidate_results.jsonl",
    "mutation_search_trace.jsonl",
    "selection_report.json",
    "fitness_landscape.json",
    "raw_generation_trace.jsonl",
    "raw_generation_results.jsonl",
    "pocket_trace.jsonl",
    "pocket_ablation_results.jsonl",
    "scoring_results.jsonl",
    "contrast_group_results.jsonl",
    "control_results.jsonl",
    "control_arm_report.json",
    "visible_bypass_control_report.json",
    "noisy_distractor_control_report.json",
    "generated_before_scoring_report.json",
    "noisy_marker_bridge_metrics.json",
    "per_seed_metrics.json",
    "per_family_metrics.json",
    "arm_comparison.json",
    "determinism_replay_report.json",
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
        for term in ["140A", "noisy", "bridge", "helper", "pocket"]:
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

    upstream = load_json(SMOKE_ROOT / "upstream_139ys_manifest.json")
    config = load_json(SMOKE_ROOT / "eval_config.json")
    helper = load_json(SMOKE_ROOT / "helper_provenance_verification.json")
    ast_report = load_json(SMOKE_ROOT / "ast_shortcut_scan_report.json")
    canary = load_json(SMOKE_ROOT / "expected_output_canary_report.json")
    forbidden = load_json(SMOKE_ROOT / "forbidden_input_rejection_report.json")
    eval_manifest = load_json(SMOKE_ROOT / "noisy_bridge_eval_manifest.json")
    scaffold = load_json(SMOKE_ROOT / "bridge_prompt_scaffold_manifest.json")
    candidates = read_jsonl(SMOKE_ROOT / "mutation_candidate_results.jsonl")
    selection = load_json(SMOKE_ROOT / "selection_report.json")
    raw_trace = read_jsonl(SMOKE_ROOT / "raw_generation_trace.jsonl")
    controls = load_json(SMOKE_ROOT / "control_arm_report.json")
    visible = load_json(SMOKE_ROOT / "visible_bypass_control_report.json")
    noisy = load_json(SMOKE_ROOT / "noisy_distractor_control_report.json")
    generated = load_json(SMOKE_ROOT / "generated_before_scoring_report.json")
    metrics = load_json(SMOKE_ROOT / "noisy_marker_bridge_metrics.json")
    comparison = load_json(SMOKE_ROOT / "arm_comparison.json")
    replay = load_json(SMOKE_ROOT / "determinism_replay_report.json")
    decision = load_json(SMOKE_ROOT / "decision.json")
    summary = load_json(SMOKE_ROOT / "summary.json")
    report = (SMOKE_ROOT / "report.md").read_text(encoding="utf-8")
    progress = read_jsonl(SMOKE_ROOT / "progress.jsonl")

    if upstream.get("decision") != "real_task_bridge_recommended":
        failures.append("BAD_139YS_DECISION")
    if upstream.get("target_milestone") != "140A_INSTNCT_POCKET_GATED_NOISY_MARKER_BRIDGE_PROBE":
        failures.append("BAD_139YS_TARGET")
    if upstream.get("train_allowed") is not False:
        failures.append("UPSTREAM_ALLOWED_TRAINING")
    if upstream.get("helper_backend_modification_allowed") is not False:
        failures.append("UPSTREAM_ALLOWED_HELPER_BACKEND_CHANGE")

    for key in ["training_performed", "helper_backend_modification_allowed", "public_api_change_allowed", "source_checkpoint_mutation_allowed", "runtime_surface_mutated", "release_surface_mutated", "product_surface_mutated", "root_license_changed"]:
        if config.get(key) is not False:
            failures.append(f"BOUNDARY_CONFIG_BAD:{key}")
    require_false_flags(config, failures)
    if helper.get("strict_pocket_gated_symbols_present") is not True:
        failures.append("HELPER_STRICT_POCKET_SYMBOLS_MISSING")
    if ast_report.get("passed") is not True:
        failures.append("AST_SCAN_FAILED")
    if canary.get("passed") is not True or forbidden.get("passed") is not True:
        failures.append("CANARY_OR_FORBIDDEN_REJECTION_FAILED")

    if eval_manifest.get("row_count", 0) < 200:
        failures.append("EVAL_TOO_SMALL")
    if scaffold.get("explicit_pocket_value_markers_reduced") is not True:
        failures.append("SCAFFOLD_NOT_REDUCED")
    if scaffold.get("noisy_prompt_distractors_added") is not True:
        failures.append("NOISY_DISTRACTORS_NOT_DECLARED")
    if scaffold.get("value_hidden_behind_natural_task_text") is not True:
        failures.append("NATURAL_TEXT_BRIDGE_NOT_DECLARED")
    if scaffold.get("direct_pocket_value_marker_rate", 1.0) > 0.40:
        failures.append("POCKET_VALUE_MARKER_RATE_TOO_HIGH")
    if scaffold.get("reduced_marker_row_rate", 0.0) < 0.60:
        failures.append("REDUCED_MARKER_RATE_TOO_LOW")

    if len(candidates) < 5:
        failures.append("TOO_FEW_MUTATION_CANDIDATES")
    selected_rows = [item for item in candidates if item.get("selected") is True]
    if len(selected_rows) != 1:
        failures.append("BAD_SELECTED_CANDIDATE_COUNT")
    else:
        selected = selected_rows[0]
        if selected.get("candidate") != "open_pocket_all_payload_markers_noisy_bridge":
            failures.append("BAD_SELECTED_CANDIDATE")
        if selected.get("answer_value_accuracy", 0.0) < 0.80:
            failures.append("SELECTED_ACCURACY_TOO_LOW")
        if selected.get("pocket_writeback_rate", 0.0) < 0.90:
            failures.append("SELECTED_WRITEBACK_TOO_LOW")
    if selection.get("gradient_used") is not False:
        failures.append("GRADIENT_USED")
    if selection.get("selected_candidate") != "open_pocket_all_payload_markers_noisy_bridge":
        failures.append("BAD_SELECTION_REPORT")
    if selection.get("fitness_margin", 0.0) <= 0.0:
        failures.append("FITNESS_MARGIN_NOT_POSITIVE")

    for trace in raw_trace[:20]:
        request = trace.get("helper_request", {})
        if set(request) != ALLOWED_HELPER_KEYS:
            failures.append("HELPER_REQUEST_KEYS_BAD")
        if trace.get("backend_name") != "repo_local_instnct_mutation_graph":
            failures.append("TRACE_BACKEND_BAD")
    if controls.get("controls_failed") is not True:
        failures.append("CONTROLS_DID_NOT_FAIL")
    required_controls = set(controls.get("required_controls_present", []))
    for control in ["VISIBLE_VALUE_BYPASS_CONTROL", "NOISY_DISTRACTOR_CONTROL", "CLOSED_POCKET_ABLATION_CONTROL", "STATIC_OUTPUT_CONTROL", "COPY_PROMPT_CONTROL", "TRAIN_NAMESPACE_REPLAY_CONTROL", "PREFIX_ONLY_CONTROL"]:
        if control not in required_controls:
            failures.append(f"CONTROL_MISSING:{control}")
    if visible.get("visible_bypass_control_failed") is not True:
        failures.append("VISIBLE_BYPASS_CONTROL_PASSED")
    if noisy.get("noisy_distractor_control_failed") is not True:
        failures.append("NOISY_DISTRACTOR_CONTROL_PASSED")
    if generated.get("passed") is not True or generated.get("all_helper_requests_allowed_keys_only") is not True:
        failures.append("GENERATED_BEFORE_SCORING_FAILED")
    if replay.get("deterministic_replay_passed") is not True:
        failures.append("DETERMINISM_FAILED")

    main = metrics.get("main", {})
    ablation = metrics.get("ablation", {})
    if main.get("answer_value_accuracy", 0.0) < 0.80:
        failures.append("MAIN_ACCURACY_TOO_LOW")
    if main.get("pocket_writeback_rate", 0.0) < 0.90:
        failures.append("MAIN_WRITEBACK_TOO_LOW")
    if main.get("contrast_group_accuracy", 0.0) < 0.80:
        failures.append("CONTRAST_GROUP_ACCURACY_TOO_LOW")
    if main.get("visible_bypass_violation_rate", 1.0) > 0.0:
        failures.append("VISIBLE_BYPASS_VIOLATION")
    if main.get("noisy_distractor_violation_rate", 1.0) > 0.0:
        failures.append("NOISY_DISTRACTOR_VIOLATION")
    if ablation.get("answer_value_accuracy", 1.0) > 0.10:
        failures.append("ABLATION_ACCURACY_TOO_HIGH")
    if comparison.get("pocket_ablation_delta_answer_value_accuracy", 0.0) < 0.50:
        failures.append("ABLATION_DELTA_TOO_LOW")

    if decision.get("decision") != "instnct_pocket_gated_noisy_marker_bridge_probe_positive":
        failures.append("BAD_DECISION")
    if decision.get("next") != "140F_INSTNCT_POCKET_GATED_NOISY_MARKER_BRIDGE_SCALE_CONFIRM":
        failures.append("BAD_NEXT")
    if decision.get("pocket_mechanism_claimed") is not True:
        failures.append("POCKET_MECHANISM_NOT_CLAIMED_FOR_POSITIVE")
    if decision.get("architecture_superiority_claimed") is not False or decision.get("value_grounding_claimed") is not False:
        failures.append("OVERCLAIM_IN_DECISION")
    for payload in [decision, summary]:
        require_false_flags(payload, failures)
    if summary.get("decision") != decision.get("decision") or summary.get("next") != decision.get("next"):
        failures.append("SUMMARY_DECISION_MISMATCH")
    for term in ["not GPT-like readiness", "not broad assistant capability", "not production readiness", "not public API"]:
        if term not in report:
            failures.append(f"REPORT_BOUNDARY_TERM_MISSING:{term}")

    events = {row.get("event") for row in progress}
    for event in [
        "startup",
        "upstream verification",
        "helper provenance",
        "noisy eval row build",
        "candidate evaluated",
        "mutation selection",
        "canary and ast",
        "final eval generation",
        "scoring",
        "controls",
        "determinism replay",
        "aggregate analysis",
        "decision",
        "final verdict",
    ]:
        if event not in events:
            failures.append(f"PROGRESS_EVENT_MISSING:{event}")
    return sorted(set(failures))


def main() -> int:
    parser = argparse.ArgumentParser(description="Check 140A noisy-marker bridge probe")
    parser.add_argument("--check-only", action="store_true")
    _args = parser.parse_args()
    failures: list[str] = []
    failures.extend(check_changed_files())
    failures.extend(check_static_files())
    failures.extend(check_artifacts())
    payload = {"schema_version": "phase_140a_checker_result_v1", "status": "pass" if not failures else "fail", "failures": failures}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
