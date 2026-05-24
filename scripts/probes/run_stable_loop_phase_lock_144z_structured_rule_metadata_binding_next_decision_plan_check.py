#!/usr/bin/env python3
"""Checker for 144Z structured rule metadata binding next-decision planning milestone."""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_144z_structured_rule_metadata_binding_next_decision_plan/smoke"
HELPER = "scripts/probes/shared_raw_generation_helper.py"
RUNNER = "scripts/probes/run_stable_loop_phase_lock_144z_structured_rule_metadata_binding_next_decision_plan.py"
CHECKER = "scripts/probes/run_stable_loop_phase_lock_144z_structured_rule_metadata_binding_next_decision_plan_check.py"
DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_144Z_STRUCTURED_RULE_METADATA_BINDING_NEXT_DECISION_PLAN_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_144Z_STRUCTURED_RULE_METADATA_BINDING_NEXT_DECISION_PLAN_RESULT.md",
]
ALLOWED_MUTATIONS = {RUNNER, CHECKER, *DOCS}
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "analysis_config.json",
    "upstream_144h_manifest.json",
    "evidence_chain_summary.json",
    "structured_rule_binding_state_report.json",
    "next_decision_matrix.json",
    "mixed_rule_composition_gap_analysis.json",
    "target_145a_milestone_plan.json",
    "anti_oracle_requirements.json",
    "risk_register.json",
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
    "not natural-language rule reasoning",
    "not open-ended arbitration",
    "not GPT-like/open-domain/broad assistant capability",
    "not production/public API/deployment/safety readiness",
    "not architecture superiority",
]
EXPECTED_144H_METRICS = {
    "main_eval_rows": 6528,
    "rule_metadata_parse_accuracy": 1.0,
    "derived_selected_pocket_accuracy": 1.0,
    "selected_pocket_to_marker_binding_accuracy": 1.0,
    "same_line_value_extraction_accuracy": 1.0,
    "end_to_end_answer_accuracy": 1.0,
    "rule_metadata_ablation_accuracy": 0.0,
    "wrong_family_extra_key_rejection_rate": 1.0,
    "quorum_clear_winner_ignores_tie_break_accuracy": 1.0,
    "legacy_143w_binding_regression_passed": True,
    "shared_helper_no_change_since_144b": True,
    "deterministic_replay_passed": True,
}
NEW_DECODER = "deterministic_pocket_gated_mixed_structured_rule_composition_binding_decoder"
OLD_DECODERS = [
    "deterministic_pocket_gated_rule_selected_pocket_binding_decoder",
    "deterministic_pocket_gated_structured_rule_metadata_binding_decoder",
]
REQUIRED_SUBSETS = [
    "SINGLE_VALID_BLOCK_BASELINE",
    "MULTI_BLOCK_PRIORITY_RECENCY_WINS",
    "MULTI_BLOCK_PRIORITY_QUORUM_WINS",
    "MULTI_BLOCK_PRIORITY_TIE_BREAK_WINS",
    "INVALID_HIGH_PRIORITY_FALLS_THROUGH_TO_LOWER_PRIORITY",
    "ALL_BLOCKS_INVALID_FALLBACK",
    "MISSING_PRIORITY_CONTROL",
    "DUPLICATE_PRIORITY_ENTRY_CONTROL",
    "UNKNOWN_PRIORITY_ENTRY_CONTROL",
    "PRIORITY_REFERENCES_MISSING_BLOCK_CONTROL",
    "MULTIPLE_PRIORITY_LINES_CONTROL",
    "DUPLICATE_RULE_BLOCK_TYPE_CONTROL",
    "MALFORMED_BLOCK_BOUNDARY_CONTROL",
    "METADATA_OUTSIDE_BLOCK_CONTROL",
    "NESTED_BLOCK_BOUNDARY_CONTROL",
    "EMPTY_RULE_BLOCK_CONTROL",
    "PRIORITY_POCKET_ORACLE_CONTROL",
    "SAME_BLOCKS_DIFFERENT_PRIORITY",
    "SAME_PRIORITY_DIFFERENT_BLOCK_VALUES",
    "SAME_TEMPLATE_OPPOSITE_PRIORITY_WINNER",
    "RULE_COMPOSITION_CORRUPTION_CONTROL",
    "LEGACY_STRUCTURED_RULE_METADATA_REGRESSION_CONTROL",
    "LEGACY_SELECTED_POCKET_BINDING_REGRESSION_CONTROL",
]
REQUIRED_METRICS = [
    "mixed_rule_block_parse_accuracy",
    "per_block_candidate_derivation_accuracy",
    "priority_policy_parse_accuracy",
    "final_selected_pocket_derivation_accuracy",
    "selected_pocket_to_marker_binding_accuracy",
    "same_line_value_extraction_accuracy",
    "end_to_end_answer_accuracy",
    "invalid_high_priority_fallthrough_accuracy",
    "all_blocks_invalid_fallback_rate",
    "missing_priority_fallback_rate",
    "duplicate_priority_rejection_rate",
    "unknown_priority_rejection_rate",
    "missing_block_reference_rejection_rate",
    "multiple_priority_lines_rejection_rate",
    "duplicate_rule_block_type_rejection_rate",
    "malformed_block_boundary_rejection_rate",
    "metadata_outside_block_rejection_rate",
    "nested_block_boundary_rejection_rate",
    "empty_rule_block_rejection_rate",
    "priority_pocket_oracle_rejection_rate",
    "same_blocks_different_priority_accuracy",
    "same_priority_different_block_values_accuracy",
    "same_template_opposite_priority_winner_accuracy",
    "rule_composition_ablation_accuracy",
    "helper_request_forbidden_metadata_count",
    "per_row_manifest_switch_rate",
    "per_row_payload_marker_switch_rate",
    "legacy_structured_rule_metadata_regression_passed",
    "legacy_selected_pocket_binding_regression_passed",
    "deterministic_replay_passed",
]
TRACE_FIELDS = [
    "parsed_rule_blocks",
    "parsed_priority_order",
    "per_block_parse_success",
    "per_block_derived_candidate_pocket",
    "final_selected_pocket_id",
    "binding_marker",
    "extracted_value",
    "generated_answer",
    "failure_reason",
]
PROMPT_SCANNER_FORBIDDEN = [
    "final_selected",
    "derived_selected",
    "selected_pocket",
    "selected_pocket_id",
    "winner=pocket_*",
    "winner value",
    "selected value",
    "answer value",
    "target value",
    "resolved output",
    "expected output",
    "gold output",
]


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
        for term in ["144Z", "planning-only", "145A", NEW_DECODER, "mixed structured rule composition"]:
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


def require_upstream(upstream: dict[str, Any], failures: list[str]) -> None:
    decision = upstream.get("decision", {})
    metrics = upstream.get("aggregate_metrics", {})
    if decision.get("decision") != "structured_rule_metadata_to_selected_pocket_binding_scale_confirmed":
        failures.append(f"BAD_144H_DECISION:{decision.get('decision')}")
    if decision.get("verdict") != "INSTNCT_STRUCTURED_RULE_METADATA_TO_SELECTED_POCKET_BINDING_SCALE_CONFIRMED":
        failures.append(f"BAD_144H_VERDICT:{decision.get('verdict')}")
    if decision.get("next") != "144Z_STRUCTURED_RULE_METADATA_BINDING_NEXT_DECISION_PLAN":
        failures.append(f"BAD_144H_NEXT:{decision.get('next')}")
    if upstream.get("failed_gate_checks") != []:
        failures.append(f"UPSTREAM_144H_FAILED_GATES:{upstream.get('failed_gate_checks')}")
    for key, expected in EXPECTED_144H_METRICS.items():
        if metrics.get(key) != expected:
            failures.append(f"UPSTREAM_144H_METRIC_MISMATCH:{key}:{metrics.get(key)}")
    for key in ["per_seed_gate_passed", "per_family_gate_passed", "static_manifest_integrity_passed"]:
        if upstream.get(key) is not True:
            failures.append(f"UPSTREAM_144H_GATE_NOT_TRUE:{key}:{upstream.get(key)}")
    request = upstream.get("helper_request_audit", {})
    if request.get("all_requests_allowed_keys_only") is not True or request.get("helper_request_forbidden_metadata_count") != 0:
        failures.append("UPSTREAM_144H_REQUEST_AUDIT_FAIL")


def require_target_145a(target: dict[str, Any], failures: list[str]) -> None:
    if target.get("milestone") != "145A_MIXED_STRUCTURED_RULE_COMPOSITION_PRIORITY_BINDING_PROTOTYPE":
        failures.append(f"BAD_TARGET_145A:{target.get('milestone')}")
    if target.get("implementation_ready") is not True or target.get("planning_only") is not False:
        failures.append("TARGET_145A_NOT_EXECUTABLE_READY")
    if target.get("decoder_name") != NEW_DECODER:
        failures.append(f"BAD_145A_DECODER:{target.get('decoder_name')}")
    for decoder in OLD_DECODERS:
        if decoder not in target.get("old_decoders_must_remain_unchanged", []):
            failures.append(f"OLD_DECODER_NOT_PRESERVED:{decoder}")
    if target.get("request_key_change_allowed") is not False:
        failures.append("TARGET_145A_ALLOWS_REQUEST_KEY_CHANGE")
    if target.get("natural_language_rule_parsing_allowed") is not False or target.get("free_form_arbitration_allowed") is not False:
        failures.append("TARGET_145A_ALLOWS_BROAD_REASONING")
    if target.get("post_generation_repair_allowed") is not False:
        failures.append("TARGET_145A_ALLOWS_POST_GENERATION_REPAIR")
    for subset in REQUIRED_SUBSETS:
        if subset not in target.get("required_subsets", []):
            failures.append(f"TARGET_145A_SUBSET_MISSING:{subset}")
    for metric in REQUIRED_METRICS:
        if metric not in target.get("required_metrics", []):
            failures.append(f"TARGET_145A_METRIC_MISSING:{metric}")
        if metric not in target.get("positive_gates", {}):
            failures.append(f"TARGET_145A_GATE_MISSING:{metric}")
    for field in TRACE_FIELDS:
        if field not in target.get("required_trace_fields", []):
            failures.append(f"TRACE_FIELD_MISSING:{field}")
    for forbidden in PROMPT_SCANNER_FORBIDDEN:
        if forbidden not in target.get("prompt_scanner_forbidden", []):
            failures.append(f"PROMPT_SCANNER_FORBID_MISSING:{forbidden}")
    format_spec = target.get("mixed_rule_block_format_spec", {})
    policy = format_spec.get("block_boundary_policy", {})
    for key in [
        "missing_block_end",
        "nested_rule_block_before_block_end",
        "metadata_outside_rule_block_except_priority",
        "duplicate_rule_block_types",
        "unknown_rule_block_types",
        "empty_rule_blocks",
    ]:
        if key not in policy:
            failures.append(f"BLOCK_POLICY_MISSING:{key}")
    priority = format_spec.get("priority_policy", {})
    for key in [
        "priority_line_required",
        "multiple_priority_lines",
        "duplicate_priority_entries",
        "unknown_priority_entries",
        "priority_references_missing_block",
        "malformed_priority_separators",
        "priority_pocket_oracle_entries",
        "invalid_high_priority_block",
    ]:
        if key not in priority:
            failures.append(f"PRIORITY_POLICY_MISSING:{key}")
    if priority.get("priority_entries_are_block_types_not_pockets", format_spec.get("priority_entries_are_block_types_not_pockets")) is not True:
        failures.append("PRIORITY_BLOCK_TYPE_GUARD_MISSING")
    for route in [
        "mixed_rule_block_parse_failure",
        "priority_policy_parse_failure",
        "final_selected_pocket_derivation_failure",
        "rule_composition_oracle_shortcut_detected",
        "priority_ambiguity_not_rejected",
        "invalid_block_fallthrough_failure",
        "legacy_structured_rule_metadata_regression",
        "selected_pocket_binding_regression",
        "helper_integrity_failure",
    ]:
        if route not in target.get("clean_negative_routes", {}):
            failures.append(f"TARGET_145A_ROUTE_MISSING:{route}")


def require_content(root: Path, failures: list[str]) -> None:
    config = load_json(root / "analysis_config.json")
    upstream = load_json(root / "upstream_144h_manifest.json")
    evidence = load_json(root / "evidence_chain_summary.json")
    state = load_json(root / "structured_rule_binding_state_report.json")
    matrix = load_json(root / "next_decision_matrix.json")
    gap = load_json(root / "mixed_rule_composition_gap_analysis.json")
    target = load_json(root / "target_145a_milestone_plan.json")
    anti = load_json(root / "anti_oracle_requirements.json")
    decision = load_json(root / "decision.json")
    summary = load_json(root / "summary.json")
    report = (root / "report.md").read_text(encoding="utf-8")

    require_upstream(upstream, failures)
    require_target_145a(target, failures)

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
        "request_key_change",
        "public_request_key_change",
        "runtime_surface_mutated",
        "release_surface_mutated",
        "product_surface_mutated",
        "root_license_changed",
    ]:
        if config.get(key) is not False:
            failures.append(f"CONFIG_NOT_FALSE:{key}:{config.get(key)}")
    require_false_flags(config, failures, "CONFIG")

    if evidence.get("current_state") != "single structured rule metadata binding scale confirmed":
        failures.append("BAD_EVIDENCE_STATE")
    if state.get("single_structured_rule_binding_scale_confirmed") is not True:
        failures.append("STATE_SINGLE_RULE_NOT_CONFIRMED")
    if state.get("legacy_selected_pocket_binding_preserved") is not True:
        failures.append("STATE_LEGACY_BINDING_NOT_PRESERVED")
    for key in [
        "single_rule_structured_metadata_scale_confirmed",
        "mixed_structured_rule_composition_untested",
        "priority_policy_final_selection_untested",
        "invalid_high_priority_fallthrough_untested",
        "block_boundary_parser_untested",
    ]:
        if gap.get(key) is not True:
            failures.append(f"GAP_NOT_TRUE:{key}:{gap.get(key)}")
    if gap.get("priority_oracle_shortcut_claimed") is not False or gap.get("natural_language_rule_reasoning_claimed") is not False:
        failures.append("GAP_FALSE_CLAIM_BAD")

    if matrix.get("selected_option") != "mixed_structured_rule_composition_priority_binding_prototype":
        failures.append(f"BAD_SELECTED_OPTION:{matrix.get('selected_option')}")
    if matrix.get("recommended_next") != "145A_MIXED_STRUCTURED_RULE_COMPOSITION_PRIORITY_BINDING_PROTOTYPE":
        failures.append(f"BAD_MATRIX_NEXT:{matrix.get('recommended_next')}")
    option_ids = {row.get("option_id") for row in matrix.get("options", [])}
    for option in [
        "mixed_structured_rule_composition_priority_binding_prototype",
        "structured_rule_metadata_robustness_extension",
        "integration_into_broader_multi_pocket_arbitration_suite",
        "stop_at_single_rule_structured_metadata_binding",
    ]:
        if option not in option_ids:
            failures.append(f"MATRIX_OPTION_MISSING:{option}")

    for forbidden in [
        "final_selected",
        "derived_selected",
        "selected_pocket",
        "selected_pocket_id",
        "winner=pocket_*",
        "per-row selected pocket request metadata",
        "per-row manifest switching",
        "payload marker list narrowed to correct pocket",
        "post-generation repair",
        "priority=pocket_*",
    ]:
        if forbidden not in anti.get("rule_composition_subsets_forbid", []):
            failures.append(f"ANTI_ORACLE_MISSING:{forbidden}")

    if decision.get("decision") != "structured_rule_composition_priority_binding_prototype_plan_recommended":
        failures.append(f"BAD_DECISION:{decision.get('decision')}")
    if decision.get("selected_option") != "mixed_structured_rule_composition_priority_binding_prototype":
        failures.append(f"BAD_DECISION_OPTION:{decision.get('selected_option')}")
    if decision.get("next") != "145A_MIXED_STRUCTURED_RULE_COMPOSITION_PRIORITY_BINDING_PROTOTYPE":
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
    parser = argparse.ArgumentParser(description="Check 144Z structured rule metadata binding next-decision artifacts")
    parser.add_argument("--root", type=Path, default=SMOKE_ROOT)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--skip-changed-files-check", action="store_true")
    args = parser.parse_args()
    root = args.root if args.root.is_absolute() else (REPO_ROOT / args.root).resolve()
    failures = run_checks(root, check_changed_files=not args.skip_changed_files_check)
    if failures:
        print("144Z CHECK FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("144Z CHECK PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
