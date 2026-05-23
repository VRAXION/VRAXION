#!/usr/bin/env python3
"""Checker for 140U real-task transform bridge probe."""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_140u_instnct_pocket_gated_real_task_transform_bridge_probe/smoke"
RUNNER = "scripts/probes/run_stable_loop_phase_lock_140u_instnct_pocket_gated_real_task_transform_bridge_probe.py"
CHECKER = "scripts/probes/run_stable_loop_phase_lock_140u_instnct_pocket_gated_real_task_transform_bridge_probe_check.py"
DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_140U_INSTNCT_POCKET_GATED_REAL_TASK_TRANSFORM_BRIDGE_PROBE_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_140U_INSTNCT_POCKET_GATED_REAL_TASK_TRANSFORM_BRIDGE_PROBE_RESULT.md",
]
ALLOWED_MUTATIONS = {RUNNER, CHECKER, *DOCS}
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "upstream_140ht_manifest.json",
    "eval_config.json",
    "helper_provenance_verification.json",
    "ast_shortcut_scan_report.json",
    "expected_output_canary_report.json",
    "forbidden_input_rejection_report.json",
    "transform_eval_manifest.json",
    "real_task_transform_prompt_manifest.json",
    "transform_binding_manifest.json",
    "explicit_marker_audit.json",
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
    "copy_only_shortcut_report.json",
    "generated_before_scoring_report.json",
    "freshness_leakage_audit.json",
    "transform_bridge_metrics.json",
    "transform_binding_metrics.json",
    "per_seed_metrics.json",
    "per_family_metrics.json",
    "per_scaffold_metrics.json",
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
            name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ""
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
        for term in ["140U", "transform", "copy", "pocket", "helper-only"]:
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
    upstream = load_json(SMOKE_ROOT / "upstream_140ht_manifest.json")
    config = load_json(SMOKE_ROOT / "eval_config.json")
    helper = load_json(SMOKE_ROOT / "helper_provenance_verification.json")
    ast_report = load_json(SMOKE_ROOT / "ast_shortcut_scan_report.json")
    canary = load_json(SMOKE_ROOT / "expected_output_canary_report.json")
    forbidden = load_json(SMOKE_ROOT / "forbidden_input_rejection_report.json")
    eval_manifest = load_json(SMOKE_ROOT / "transform_eval_manifest.json")
    prompt_manifest = load_json(SMOKE_ROOT / "real_task_transform_prompt_manifest.json")
    binding_manifest = load_json(SMOKE_ROOT / "transform_binding_manifest.json")
    audit = load_json(SMOKE_ROOT / "explicit_marker_audit.json")
    candidates = read_jsonl(SMOKE_ROOT / "mutation_candidate_results.jsonl")
    selection = load_json(SMOKE_ROOT / "selection_report.json")
    raw_trace = read_jsonl(SMOKE_ROOT / "raw_generation_trace.jsonl")
    controls = load_json(SMOKE_ROOT / "control_arm_report.json")
    visible = load_json(SMOKE_ROOT / "visible_bypass_control_report.json")
    noisy = load_json(SMOKE_ROOT / "noisy_distractor_control_report.json")
    copy_report = load_json(SMOKE_ROOT / "copy_only_shortcut_report.json")
    generated = load_json(SMOKE_ROOT / "generated_before_scoring_report.json")
    leakage = load_json(SMOKE_ROOT / "freshness_leakage_audit.json")
    metrics = load_json(SMOKE_ROOT / "transform_bridge_metrics.json")
    transform_metrics = load_json(SMOKE_ROOT / "transform_binding_metrics.json")
    comparison = load_json(SMOKE_ROOT / "arm_comparison.json")
    replay = load_json(SMOKE_ROOT / "determinism_replay_report.json")
    decision = load_json(SMOKE_ROOT / "decision.json")
    summary = load_json(SMOKE_ROOT / "summary.json")
    report = (SMOKE_ROOT / "report.md").read_text(encoding="utf-8")
    progress = read_jsonl(SMOKE_ROOT / "progress.jsonl")

    if upstream.get("decision") != "real_task_transform_bridge_recommended":
        failures.append("BAD_140HT_DECISION")
    if upstream.get("next") != "140U_INSTNCT_POCKET_GATED_REAL_TASK_TRANSFORM_BRIDGE_PROBE":
        failures.append("BAD_140HT_NEXT")
    if upstream.get("target_milestone") != "140U_INSTNCT_POCKET_GATED_REAL_TASK_TRANSFORM_BRIDGE_PROBE":
        failures.append("BAD_140HT_TARGET")

    for key in ["training_performed", "helper_backend_modification_allowed", "public_request_key_change_allowed", "source_checkpoint_mutation_allowed", "runtime_surface_mutated", "release_surface_mutated", "product_surface_mutated", "root_license_changed"]:
        if config.get(key) is not False:
            failures.append(f"BOUNDARY_CONFIG_BAD:{key}")
    require_false_flags(config, failures)
    if helper.get("strict_pocket_gated_symbols_present") is not True:
        failures.append("HELPER_STRICT_POCKET_SYMBOLS_MISSING")
    if ast_report.get("passed") is not True:
        failures.append("AST_SCAN_FAILED")
    if canary.get("passed") is not True or forbidden.get("passed") is not True:
        failures.append("CANARY_OR_FORBIDDEN_REJECTION_FAILED")

    if eval_manifest.get("row_count", 0) < 500:
        failures.append("TRANSFORM_EVAL_TOO_SMALL")
    if eval_manifest.get("family_count", 0) < 5:
        failures.append("TOO_FEW_FAMILIES")
    if eval_manifest.get("scaffold_variant_count", 0) < 20:
        failures.append("TOO_FEW_SCAFFOLDS")
    for key in ["natural_ish_task_text_primary", "direct_pocket_value_marker_forbidden_in_main_eval", "visible_wrong_value_present", "noisy_distractors_present"]:
        if prompt_manifest.get(key) is not True:
            failures.append(f"PROMPT_REQUIREMENT_MISSING:{key}")
    if binding_manifest.get("target_differs_from_source_rate", 0.0) < 1.0:
        failures.append("TARGET_SOURCE_NOT_DISTINCT")
    if len(binding_manifest.get("rule_ids", [])) < 4:
        failures.append("TOO_FEW_TRANSFORM_RULES")
    if audit.get("direct_pocket_value_marker_rate", 1.0) != 0.0:
        failures.append("DIRECT_POCKET_VALUE_MARKER_PRESENT")
    if audit.get("explicit_pocket_token_row_rate", 1.0) > 0.10:
        failures.append("EXPLICIT_POCKET_TOKEN_RATE_TOO_HIGH")
    if audit.get("implicit_or_minimal_gate_row_rate", 0.0) < 0.70:
        failures.append("IMPLICIT_GATE_RATE_TOO_LOW")

    if len(candidates) < 7:
        failures.append("TOO_FEW_MUTATION_CANDIDATES")
    selected_rows = [item for item in candidates if item.get("selected") is True]
    if len(selected_rows) != 1 or selected_rows[0].get("candidate") != "open_transform_target_all_markers":
        failures.append("BAD_SELECTED_CANDIDATE")
    if selection.get("gradient_used") is not False or selection.get("selected_candidate") != "open_transform_target_all_markers":
        failures.append("BAD_SELECTION_REPORT")
    if selection.get("fitness_margin", 0.0) <= 0.0:
        failures.append("FITNESS_MARGIN_NOT_POSITIVE")

    for trace in raw_trace[:25]:
        request = trace.get("helper_request", {})
        if set(request) != ALLOWED_HELPER_KEYS:
            failures.append("HELPER_REQUEST_KEYS_BAD")
        if trace.get("backend_name") != "repo_local_instnct_mutation_graph":
            failures.append("TRACE_BACKEND_BAD")
    if controls.get("controls_failed") is not True:
        failures.append("CONTROLS_DID_NOT_FAIL")
    for control in ["COPY_ONLY_SOURCE_CONTROL", "COPY_CANDIDATE_SOURCE_CONTROL", "VISIBLE_TARGET_BYPASS_CONTROL", "NOISY_DISTRACTOR_CONTROL", "CLOSED_POCKET_ABLATION_CONTROL", "STATIC_OUTPUT_CONTROL", "PREFIX_ONLY_CONTROL", "TRAIN_NAMESPACE_REPLAY_CONTROL"]:
        if control not in set(controls.get("required_controls_present", [])):
            failures.append(f"CONTROL_MISSING:{control}")
    if controls.get("copy_only_source_control_failed") is not True or copy_report.get("copy_only_control_failed") is not True:
        failures.append("COPY_ONLY_CONTROL_DID_NOT_FAIL")
    if visible.get("visible_bypass_control_failed") is not True or noisy.get("noisy_distractor_control_failed") is not True:
        failures.append("BYPASS_OR_DISTRACTOR_CONTROL_PASSED")
    if generated.get("passed") is not True or generated.get("all_helper_requests_allowed_keys_only") is not True:
        failures.append("GENERATED_BEFORE_SCORING_FAILED")
    if leakage.get("leakage_rejected") is not True or leakage.get("expected_or_scorer_metadata_in_helper_requests") is not False:
        failures.append("LEAKAGE_AUDIT_FAILED")
    if replay.get("deterministic_replay_passed") is not True:
        failures.append("DETERMINISM_FAILED")

    main = metrics.get("main", {})
    ablation = metrics.get("ablation", {})
    if main.get("answer_value_accuracy", 0.0) < 0.60:
        failures.append("MAIN_ACCURACY_TOO_LOW")
    if main.get("transform_accuracy", 0.0) < 0.60:
        failures.append("TRANSFORM_ACCURACY_TOO_LOW")
    if main.get("pocket_writeback_rate", 0.0) < 0.75:
        failures.append("MAIN_WRITEBACK_TOO_LOW")
    if main.get("source_copy_shortcut_rate", 1.0) > 0.0:
        failures.append("SOURCE_COPY_SHORTCUT")
    if main.get("visible_bypass_violation_rate", 1.0) > 0.0 or main.get("noisy_distractor_violation_rate", 1.0) > 0.0:
        failures.append("BYPASS_OR_DISTRACTOR_VIOLATION")
    if ablation.get("answer_value_accuracy", 1.0) > 0.15:
        failures.append("ABLATION_ACCURACY_TOO_HIGH")
    if comparison.get("pocket_ablation_delta_answer_value_accuracy", 0.0) < 0.35:
        failures.append("ABLATION_DELTA_TOO_LOW")
    if transform_metrics.get("copy_only_shortcut_detected") is not False or copy_report.get("copy_only_shortcut_detected") is not False:
        failures.append("COPY_ONLY_SHORTCUT_DETECTED")

    if decision.get("decision") != "instnct_pocket_gated_real_task_transform_bridge_probe_positive":
        failures.append("BAD_DECISION")
    if decision.get("next") != "140V_INSTNCT_POCKET_GATED_REAL_TASK_TRANSFORM_BRIDGE_SCALE_CONFIRM":
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
    for event in ["startup", "upstream verification", "helper and ast verification", "transform eval row build", "candidate evaluated", "mutation selection", "canary", "final eval generation", "scoring", "controls", "determinism replay", "aggregate analysis", "decision", "final verdict"]:
        if event not in events:
            failures.append(f"PROGRESS_EVENT_MISSING:{event}")
    return sorted(set(failures))


def main() -> int:
    parser = argparse.ArgumentParser(description="Check 140U real-task transform bridge probe")
    parser.add_argument("--check-only", action="store_true")
    _args = parser.parse_args()
    failures: list[str] = []
    failures.extend(check_changed_files())
    failures.extend(check_static_files())
    failures.extend(check_artifacts())
    payload = {"schema_version": "phase_140u_checker_result_v1", "status": "pass" if not failures else "fail", "failures": failures}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
