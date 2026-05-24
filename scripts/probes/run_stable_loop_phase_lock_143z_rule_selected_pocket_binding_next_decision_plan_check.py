#!/usr/bin/env python3
"""Checker for 143Z selected-pocket binding next-decision planning milestone."""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_143z_rule_selected_pocket_binding_next_decision_plan/smoke"
HELPER = "scripts/probes/shared_raw_generation_helper.py"
RUNNER = "scripts/probes/run_stable_loop_phase_lock_143z_rule_selected_pocket_binding_next_decision_plan.py"
CHECKER = "scripts/probes/run_stable_loop_phase_lock_143z_rule_selected_pocket_binding_next_decision_plan_check.py"
DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_143Z_RULE_SELECTED_POCKET_BINDING_NEXT_DECISION_PLAN_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_143Z_RULE_SELECTED_POCKET_BINDING_NEXT_DECISION_PLAN_RESULT.md",
]
ALLOWED_MUTATIONS = {RUNNER, CHECKER, *DOCS}
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "analysis_config.json",
    "upstream_143w_manifest.json",
    "evidence_chain_summary.json",
    "selected_pocket_binding_state_report.json",
    "rule_metadata_bridge_gap_analysis.json",
    "next_decision_matrix.json",
    "anti_oracle_requirements.json",
    "risk_register.json",
    "target_144a_milestone_plan.json",
    "decision.json",
    "summary.json",
    "report.md",
]
FALSE_FLAGS = [
    "reasoning_restored",
    "raw_assistant_capability_restored",
    "structured_tool_capability_restored",
    "rule_metadata_reasoning_claimed",
    "natural_language_rule_reasoning_claimed",
    "open_ended_arbitration_claimed",
    "gpt_like_readiness_claimed",
    "open_domain_assistant_readiness_claimed",
    "production_chat_claimed",
    "public_api_claimed",
    "deployment_readiness_claimed",
    "safety_alignment_claimed",
    "architecture_superiority_claimed",
]
BOUNDARY_PHRASES = [
    "constrained helper/backend evidence only",
    "prompt-visible selected-pocket binding only",
    "not rule metadata reasoning yet",
    "not natural-language rule reasoning",
    "not open-ended arbitration",
    "not GPT-like/open-domain/broad assistant capability",
    "not production/public API/deployment/safety readiness",
    "not architecture superiority",
]
EXPECTED_143W_METRICS = {
    "main_eval_rows": 3072,
    "single_selected_marker_binding_accuracy": 1.0,
    "positive_binding_subset_writeback_rate": 1.0,
    "duplicate_selected_marker_conflict_rejection_rate": 1.0,
    "duplicate_selected_marker_same_value_rejection_rate": 1.0,
    "duplicate_non_selected_marker_conflict_binding_accuracy": 1.0,
    "selected_marker_invalid_value_fallback_rate": 1.0,
    "selected_marker_multi_value_same_line_fallback_rate": 1.0,
    "selected_marker_prose_plus_one_valid_line_accuracy": 1.0,
    "following_line_value_leak_rate": 0.0,
    "legacy_manifest_regression_passed": True,
    "shared_helper_no_change_since_143v": True,
    "deterministic_replay_passed": True,
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def git_status() -> str:
    result = subprocess.run(["git", "status", "--short"], cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    return result.stdout.rstrip("\n")


def changed_paths() -> list[str]:
    return [line[3:].replace("\\", "/") for line in git_status().splitlines() if line.strip()]


def require_false_flags(payload: dict[str, Any], failures: list[str], prefix: str) -> None:
    for key in FALSE_FLAGS:
        if payload.get(key) is not False:
            failures.append(f"{prefix}_BOUNDARY_FLAG_NOT_FALSE:{key}:{payload.get(key)}")


def scan_python(path: Path) -> list[str]:
    failures: list[str] = []
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.startswith("run_stable_loop_phase_lock_") or module == "shared_raw_generation_helper":
                failures.append(f"FORBIDDEN_IMPORT:{path.name}:{module}")
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in {"torch", "shared_raw_generation_helper"}:
                    failures.append(f"FORBIDDEN_IMPORT:{path.name}:{alias.name}")
        if isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ""
            if name in {"train", "fit", "backward", "step", "forward", "raw_generate", "load_checkpoint"}:
                failures.append(f"FORBIDDEN_CALL:{path.name}:{name}")
    return failures


def require_changed_files(failures: list[str]) -> None:
    for path in changed_paths():
        if path.startswith("target/"):
            continue
        if path not in ALLOWED_MUTATIONS:
            failures.append(f"UNEXPECTED_CHANGED_FILE:{path}")


def require_static_files(failures: list[str]) -> None:
    for rel in [RUNNER, CHECKER, *DOCS]:
        path = REPO_ROOT / rel
        if not path.exists():
            failures.append(f"MISSING_TRACKED_FILE:{rel}")
            continue
        text = path.read_text(encoding="utf-8")
        for term in ["143Z", "planning-only", "structured rule metadata", "144A"]:
            if term not in text:
                failures.append(f"TERM_MISSING:{rel}:{term}")
        for phrase in BOUNDARY_PHRASES:
            if phrase not in text:
                failures.append(f"BOUNDARY_TERM_MISSING:{rel}:{phrase}")
        if path.suffix == ".py":
            failures.extend(scan_python(path))
    helper_source = (REPO_ROOT / HELPER).read_text(encoding="utf-8")
    head = subprocess.run(["git", "show", f"HEAD:{HELPER}"], cwd=REPO_ROOT, text=True, capture_output=True, check=False).stdout
    if helper_source != head:
        failures.append("SHARED_HELPER_CHANGED_FROM_HEAD")


def require_artifacts(root: Path, failures: list[str]) -> None:
    for name in REQUIRED_ARTIFACTS:
        if not (root / name).exists():
            failures.append(f"MISSING_ARTIFACT:{name}")


def require_content(root: Path, failures: list[str]) -> None:
    config = load_json(root / "analysis_config.json")
    upstream = load_json(root / "upstream_143w_manifest.json")
    evidence = load_json(root / "evidence_chain_summary.json")
    state = load_json(root / "selected_pocket_binding_state_report.json")
    gap = load_json(root / "rule_metadata_bridge_gap_analysis.json")
    matrix = load_json(root / "next_decision_matrix.json")
    anti = load_json(root / "anti_oracle_requirements.json")
    target = load_json(root / "target_144a_milestone_plan.json")
    decision = load_json(root / "decision.json")
    summary = load_json(root / "summary.json")
    report = (root / "report.md").read_text(encoding="utf-8")

    upstream_decision = upstream.get("decision", {})
    if upstream_decision.get("decision") != "selected_marker_occurrence_count_rejection_scale_confirmed":
        failures.append(f"BAD_143W_DECISION:{upstream_decision.get('decision')}")
    if upstream_decision.get("verdict") != "INSTNCT_SELECTED_MARKER_OCCURRENCE_COUNT_REJECTION_SCALE_CONFIRMED":
        failures.append(f"BAD_143W_VERDICT:{upstream_decision.get('verdict')}")
    if upstream_decision.get("next") != "143Z_RULE_SELECTED_POCKET_BINDING_NEXT_DECISION_PLAN":
        failures.append(f"BAD_143W_NEXT:{upstream_decision.get('next')}")
    if upstream.get("failed_gate_checks") != []:
        failures.append(f"UPSTREAM_143W_FAILED_GATES:{upstream.get('failed_gate_checks')}")
    metrics = upstream.get("aggregate_metrics", {})
    for key, expected in EXPECTED_143W_METRICS.items():
        if metrics.get(key) != expected:
            failures.append(f"UPSTREAM_143W_METRIC_MISMATCH:{key}:{metrics.get(key)}")

    for key in ["planning_only", "artifact_only", "shared_helper_unchanged_from_head"]:
        if config.get(key) is not True:
            failures.append(f"CONFIG_NOT_TRUE:{key}:{config.get(key)}")
    for key in [
        "training_performed",
        "new_model_inference_run",
        "shared_helper_called",
        "helper_generation_called",
        "raw_generate_called",
        "torch_forward_pass_run",
        "checkpoint_mutated",
        "helper_modified",
        "backend_modified",
        "public_request_key_change",
        "runtime_surface_mutated",
        "release_surface_mutated",
        "product_surface_mutated",
        "root_license_changed",
    ]:
        if config.get(key) is not False:
            failures.append(f"CONFIG_NOT_FALSE:{key}:{config.get(key)}")
    require_false_flags(config, failures, "CONFIG")

    if evidence.get("current_state") != "selected-pocket binding and selected-marker occurrence rejection scale confirmed":
        failures.append("BAD_EVIDENCE_STATE")
    if state.get("selected_pocket_binding_scale_confirmed") is not True:
        failures.append("STATE_BINDING_NOT_CONFIRMED")
    if state.get("selected_marker_occurrence_rejection_scale_confirmed") is not True:
        failures.append("STATE_OCCURRENCE_REJECTION_NOT_CONFIRMED")
    if gap.get("selected_pocket_binding_scale_confirmed") is not True:
        failures.append("GAP_BINDING_NOT_CONFIRMED")
    if gap.get("selected_marker_occurrence_rejection_scale_confirmed") is not True:
        failures.append("GAP_OCCURRENCE_NOT_CONFIRMED")
    if gap.get("rule_metadata_to_selected_pocket_identity_untested") is not True:
        failures.append("GAP_RULE_METADATA_NOT_UNTESTED")
    if gap.get("natural_language_rule_reasoning_untested") is not True:
        failures.append("GAP_NL_REASONING_NOT_UNTESTED")
    if gap.get("open_ended_arbitration_claimed") is not False:
        failures.append("GAP_OPEN_ENDED_CLAIM_NOT_FALSE")

    if matrix.get("selected_option") != "structured_rule_metadata_to_selected_pocket_binding_plan":
        failures.append(f"BAD_SELECTED_OPTION:{matrix.get('selected_option')}")
    if matrix.get("recommended_next") != "144A_STRUCTURED_RULE_METADATA_TO_SELECTED_POCKET_BINDING_PLAN":
        failures.append(f"BAD_MATRIX_NEXT:{matrix.get('recommended_next')}")
    option_ids = {row.get("option_id") for row in matrix.get("options", [])}
    for option in [
        "structured_rule_metadata_to_selected_pocket_binding_plan",
        "binding_robustness_extension_plan",
        "integrate_selected_pocket_binding_into_broader_arbitration_suite",
        "keep_prompt_visible_winner_label_only_and_stop_bridge_expansion",
    ]:
        if option not in option_ids:
            failures.append(f"MATRIX_OPTION_MISSING:{option}")

    for forbidden in [
        "winner=pocket_* in rule-derived subsets",
        "selected_pocket_id in prompt or request metadata",
        "per-row selected pocket request metadata",
        "per-row manifest switching",
        "payload marker list narrowed to the correct pocket",
    ]:
        if forbidden not in anti.get("forbid", []):
            failures.append(f"ANTI_ORACLE_FORBID_MISSING:{forbidden}")

    if target.get("milestone") != "144A_STRUCTURED_RULE_METADATA_TO_SELECTED_POCKET_BINDING_PLAN":
        failures.append(f"BAD_TARGET_144A:{target.get('milestone')}")
    if target.get("planning_only") is not True or target.get("artifact_only") is not True:
        failures.append("TARGET_144A_NOT_PLANNING_ONLY")
    if target.get("natural_language_rule_parsing_allowed") is not False:
        failures.append("TARGET_144A_ALLOWS_NL_RULE_PARSING")
    if target.get("first_executable_prototype") != "144B_STRUCTURED_RULE_METADATA_TO_SELECTED_POCKET_BINDING_PROTOTYPE":
        failures.append(f"BAD_144B_TARGET:{target.get('first_executable_prototype')}")
    grammars = target.get("candidate_rule_metadata_grammars", {})
    for grammar in ["quorum", "recency", "tie_break", "hierarchy"]:
        if grammar not in grammars:
            failures.append(f"RULE_GRAMMAR_MISSING:{grammar}")
    for metric in [
        "rule_metadata_parse_accuracy",
        "derived_selected_pocket_accuracy",
        "selected_pocket_to_marker_binding_accuracy",
        "end_to_end_answer_accuracy",
        "rule_metadata_ablation_accuracy",
        "explicit_winner_baseline_accuracy",
    ]:
        if metric not in target.get("target_144b_required_metrics", []):
            failures.append(f"TARGET_144B_METRIC_MISSING:{metric}")
    for subset in [
        "EXPLICIT_WINNER_LABEL_BASELINE",
        "RULE_METADATA_DERIVED_NO_WINNER_LABEL",
        "SAME_VALUES_DIFFERENT_RULE",
        "SAME_RULE_DIFFERENT_VALUES",
        "SAME_TEMPLATE_OPPOSITE_RULE_WINNER",
        "RULE_METADATA_CORRUPTION_CONTROL",
    ]:
        if subset not in target.get("target_144b_required_subsets", []):
            failures.append(f"TARGET_144B_SUBSET_MISSING:{subset}")

    if decision.get("decision") != "rule_metadata_to_selected_pocket_binding_plan_recommended":
        failures.append(f"BAD_DECISION:{decision.get('decision')}")
    if decision.get("selected_option") != "structured_rule_metadata_to_selected_pocket_binding_plan":
        failures.append(f"BAD_DECISION_OPTION:{decision.get('selected_option')}")
    if decision.get("next") != "144A_STRUCTURED_RULE_METADATA_TO_SELECTED_POCKET_BINDING_PLAN":
        failures.append(f"BAD_DECISION_NEXT:{decision.get('next')}")
    require_false_flags(decision, failures, "DECISION")
    require_false_flags(summary, failures, "SUMMARY")
    for phrase in BOUNDARY_PHRASES:
        if phrase not in json.dumps(decision):
            failures.append(f"DECISION_BOUNDARY_MISSING:{phrase}")
        if phrase not in json.dumps(summary):
            failures.append(f"SUMMARY_BOUNDARY_MISSING:{phrase}")
        if phrase not in report:
            failures.append(f"REPORT_BOUNDARY_MISSING:{phrase}")


def run_checks(root: Path, check_changed_files: bool) -> list[str]:
    failures: list[str] = []
    if check_changed_files:
        require_changed_files(failures)
    require_static_files(failures)
    require_artifacts(root, failures)
    if not failures:
        require_content(root, failures)
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Check 143Z selected-pocket binding next decision artifacts")
    parser.add_argument("--root", type=Path, default=SMOKE_ROOT)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--skip-changed-files-check", action="store_true")
    args = parser.parse_args()
    root = args.root if args.root.is_absolute() else (REPO_ROOT / args.root).resolve()
    failures = run_checks(root, check_changed_files=not args.skip_changed_files_check)
    if failures:
        print("143Z CHECK FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("143Z CHECK PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
