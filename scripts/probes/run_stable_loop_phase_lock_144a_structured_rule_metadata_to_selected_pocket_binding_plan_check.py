#!/usr/bin/env python3
"""Checker for 144A structured rule metadata planning milestone."""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_144a_structured_rule_metadata_to_selected_pocket_binding_plan/smoke"
HELPER = "scripts/probes/shared_raw_generation_helper.py"
RUNNER = "scripts/probes/run_stable_loop_phase_lock_144a_structured_rule_metadata_to_selected_pocket_binding_plan.py"
CHECKER = "scripts/probes/run_stable_loop_phase_lock_144a_structured_rule_metadata_to_selected_pocket_binding_plan_check.py"
DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_144A_STRUCTURED_RULE_METADATA_TO_SELECTED_POCKET_BINDING_PLAN_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_144A_STRUCTURED_RULE_METADATA_TO_SELECTED_POCKET_BINDING_PLAN_RESULT.md",
]
ALLOWED_MUTATIONS = {RUNNER, CHECKER, *DOCS}
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "analysis_config.json",
    "upstream_143z_manifest.json",
    "prototype_design_requirements.json",
    "structured_rule_metadata_grammar_spec.json",
    "rule_derivation_policy_matrix.json",
    "prototype_options_matrix.json",
    "selected_prototype_recommendation.json",
    "target_144b_milestone_plan.json",
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
    "structured rule metadata to selected-pocket binding only",
    "not natural-language rule reasoning",
    "not open-ended arbitration",
    "not GPT-like/open-domain/broad assistant capability",
    "not production/public API/deployment/safety readiness",
    "not architecture superiority",
]
NEW_DECODER = "deterministic_pocket_gated_structured_rule_metadata_binding_decoder"
OLD_DECODER = "deterministic_pocket_gated_rule_selected_pocket_binding_decoder"
REQUIRED_METRICS = [
    "rule_metadata_parse_accuracy",
    "derived_selected_pocket_accuracy",
    "selected_pocket_to_marker_binding_accuracy",
    "same_line_value_extraction_accuracy",
    "end_to_end_answer_accuracy",
    "rule_derived_no_winner_label_accuracy",
    "explicit_winner_baseline_accuracy",
    "rule_metadata_ablation_accuracy",
    "corrupt_rule_metadata_rejection_rate",
    "missing_rule_metadata_fallback_rate",
    "ambiguous_rule_metadata_rejection_rate",
    "helper_request_forbidden_metadata_count",
    "per_row_manifest_switch_rate",
    "per_row_payload_marker_switch_rate",
    "legacy_selected_pocket_binding_regression_passed",
    "deterministic_replay_passed",
]
REQUIRED_SUBSETS = [
    "EXPLICIT_WINNER_LABEL_BASELINE",
    "RULE_METADATA_DERIVED_NO_WINNER_LABEL",
    "QUORUM_RULE_DERIVED",
    "RECENCY_RULE_DERIVED",
    "TIE_BREAK_RULE_DERIVED",
    "HIERARCHY_RULE_DERIVED",
    "SAME_VALUES_DIFFERENT_RULE",
    "SAME_RULE_DIFFERENT_VALUES",
    "SAME_TEMPLATE_OPPOSITE_RULE_WINNER",
    "RULE_METADATA_CORRUPTION_CONTROL",
    "MISSING_RULE_METADATA_CONTROL",
    "AMBIGUOUS_RULE_METADATA_CONTROL",
    "LEGACY_SELECTED_POCKET_BINDING_REGRESSION_CONTROL",
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
        for term in ["144A", "planning-only", "structured rule metadata", "144B", NEW_DECODER]:
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
    gap = upstream.get("rule_metadata_bridge_gap_analysis", {})
    state = upstream.get("selected_pocket_binding_state_report", {})
    target = upstream.get("target_144a_milestone_plan", {})
    if decision.get("decision") != "rule_metadata_to_selected_pocket_binding_plan_recommended":
        failures.append(f"BAD_143Z_DECISION:{decision.get('decision')}")
    if decision.get("selected_option") != "structured_rule_metadata_to_selected_pocket_binding_plan":
        failures.append(f"BAD_143Z_OPTION:{decision.get('selected_option')}")
    if decision.get("next") != "144A_STRUCTURED_RULE_METADATA_TO_SELECTED_POCKET_BINDING_PLAN":
        failures.append(f"BAD_143Z_NEXT:{decision.get('next')}")
    for key in [
        "selected_pocket_binding_scale_confirmed",
        "selected_marker_occurrence_rejection_scale_confirmed",
        "rule_metadata_to_selected_pocket_identity_untested",
        "natural_language_rule_reasoning_untested",
    ]:
        if gap.get(key) is not True:
            failures.append(f"BAD_143Z_GAP:{key}:{gap.get(key)}")
    if gap.get("open_ended_arbitration_claimed") is not False:
        failures.append("143Z_OPEN_ENDED_CLAIM_NOT_FALSE")
    if state.get("selected_pocket_binding_scale_confirmed") is not True:
        failures.append("143Z_STATE_BINDING_NOT_CONFIRMED")
    if target.get("planning_only") is not True:
        failures.append("143Z_TARGET_144A_NOT_PLANNING_ONLY")
    if target.get("first_executable_prototype") != "144B_STRUCTURED_RULE_METADATA_TO_SELECTED_POCKET_BINDING_PROTOTYPE":
        failures.append(f"143Z_BAD_144B_TARGET:{target.get('first_executable_prototype')}")
    if upstream.get("failed_gate_checks") != []:
        failures.append(f"143Z_FAILED_GATES:{upstream.get('failed_gate_checks')}")


def require_grammar(grammar: dict[str, Any], failures: list[str]) -> None:
    if grammar.get("free_form_natural_language_rule_parsing_allowed") is not False:
        failures.append("GRAMMAR_ALLOWS_FREE_FORM_NL")
    general = grammar.get("general_rules", {})
    for key in [
        "exact_keys_only_per_rule_family",
        "rule_derived_subsets_forbid_winner_labels",
        "rule_derived_subsets_forbid_selected_pocket_id",
    ]:
        if general.get(key) is not True:
            failures.append(f"GRAMMAR_GENERAL_NOT_TRUE:{key}:{general.get(key)}")
    for key in [
        "duplicate_keys_policy",
        "unknown_keys_policy",
        "missing_required_keys_policy",
        "multiple_rule_type_lines_policy",
        "invalid_pocket_ids_policy",
        "malformed_separators_policy",
    ]:
        if general.get(key) != "fallback":
            failures.append(f"GRAMMAR_GENERAL_POLICY_BAD:{key}:{general.get(key)}")
    families = grammar.get("families", {})
    required = {
        "quorum": ["rule_type", "votes"],
        "recency": ["rule_type", "recency_order"],
        "tie_break": ["rule_type", "tied", "tie_break_order"],
        "hierarchy": ["rule_type", "hierarchy", "stale", "recency_winner", "quorum_winner", "tie_break_winner"],
    }
    for family, keys in required.items():
        spec = families.get(family)
        if not spec:
            failures.append(f"GRAMMAR_FAMILY_MISSING:{family}")
            continue
        if spec.get("required_keys") != keys:
            failures.append(f"GRAMMAR_REQUIRED_KEYS_BAD:{family}:{spec.get('required_keys')}")
        policy_text = " ".join(spec.get("policy", []))
        if "fallback" not in policy_text:
            failures.append(f"GRAMMAR_FALLBACK_POLICY_MISSING:{family}")
    hierarchy_policy = " ".join(families.get("hierarchy", {}).get("policy", []))
    if "does not claim nested derivation" not in hierarchy_policy:
        failures.append("HIERARCHY_OVERCLAIM_GUARD_MISSING")


def require_target_144b(target: dict[str, Any], failures: list[str]) -> None:
    if target.get("milestone") != "144B_STRUCTURED_RULE_METADATA_TO_SELECTED_POCKET_BINDING_PROTOTYPE":
        failures.append(f"BAD_TARGET_144B:{target.get('milestone')}")
    if target.get("decoder_name") != NEW_DECODER:
        failures.append(f"BAD_144B_DECODER:{target.get('decoder_name')}")
    if target.get("old_selected_pocket_binding_decoder_must_remain_unchanged") is not True:
        failures.append("OLD_SELECTED_POCKET_DECODER_NOT_PRESERVED")
    if target.get("existing_selected_pocket_decoder") != OLD_DECODER:
        failures.append(f"BAD_OLD_DECODER:{target.get('existing_selected_pocket_decoder')}")
    if target.get("implementation_ready") is not True or target.get("no_additional_planning_milestone_expected") is not True:
        failures.append("TARGET_144B_NOT_IMPLEMENTATION_READY")
    if target.get("request_key_change_allowed") is not False or target.get("post_generation_repair_allowed") is not False:
        failures.append("TARGET_144B_ALLOWS_FORBIDDEN_SURFACE")
    for subset in REQUIRED_SUBSETS:
        if subset not in target.get("required_subsets", []):
            failures.append(f"TARGET_144B_SUBSET_MISSING:{subset}")
    for metric in REQUIRED_METRICS:
        if metric not in target.get("required_metrics", []):
            failures.append(f"TARGET_144B_METRIC_MISSING:{metric}")
    for key in [
        "rule_metadata_parse_accuracy",
        "derived_selected_pocket_accuracy",
        "selected_pocket_to_marker_binding_accuracy",
        "same_line_value_extraction_accuracy",
        "end_to_end_answer_accuracy",
        "rule_metadata_ablation_accuracy",
        "corrupt_rule_metadata_rejection_rate",
        "missing_rule_metadata_fallback_rate",
        "ambiguous_rule_metadata_rejection_rate",
    ]:
        if key not in target.get("positive_gates", {}):
            failures.append(f"TARGET_144B_GATE_MISSING:{key}")
    for route in [
        "structured_rule_metadata_parse_failure",
        "derived_selected_pocket_failure",
        "rule_metadata_oracle_shortcut_detected",
        "rule_metadata_ambiguity_not_rejected",
        "hierarchy_priority_policy_failure",
        "selected_pocket_binding_regression",
        "helper_integrity_failure",
    ]:
        if route not in target.get("clean_negative_routes", {}):
            failures.append(f"TARGET_144B_ROUTE_MISSING:{route}")


def require_content(root: Path, failures: list[str]) -> None:
    config = load_json(root / "analysis_config.json")
    upstream = load_json(root / "upstream_143z_manifest.json")
    requirements = load_json(root / "prototype_design_requirements.json")
    grammar = load_json(root / "structured_rule_metadata_grammar_spec.json")
    policies = load_json(root / "rule_derivation_policy_matrix.json")
    options = load_json(root / "prototype_options_matrix.json")
    recommendation = load_json(root / "selected_prototype_recommendation.json")
    target = load_json(root / "target_144b_milestone_plan.json")
    anti = load_json(root / "anti_oracle_requirements.json")
    decision = load_json(root / "decision.json")
    summary = load_json(root / "summary.json")
    report = (root / "report.md").read_text(encoding="utf-8")

    require_upstream(upstream, failures)
    require_grammar(grammar, failures)
    require_target_144b(target, failures)

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

    if requirements.get("decoder_name") != NEW_DECODER:
        failures.append("REQUIREMENTS_BAD_DECODER")
    if requirements.get("existing_selected_pocket_decoder_to_preserve") != OLD_DECODER:
        failures.append("REQUIREMENTS_BAD_OLD_DECODER")
    if requirements.get("request_key_change_allowed") is not False:
        failures.append("REQUIREMENTS_ALLOW_REQUEST_KEY_CHANGE")
    for field in [
        "parsed_rule_type",
        "parsed_rule_fields",
        "parse_success",
        "derived_selected_pocket_id",
        "binding_marker",
        "extracted_value",
        "generated_answer",
        "failure_reason",
    ]:
        if field not in requirements.get("trace_fields_required", []):
            failures.append(f"TRACE_FIELD_MISSING:{field}")
    if policies.get("hierarchy_is_combiner_fixture_not_nested_derivation") is not True:
        failures.append("POLICY_HIERARCHY_GUARD_MISSING")
    if options.get("selected_option") != "canonical_structured_rule_metadata_parser_plus_existing_selected_pocket_binding":
        failures.append(f"BAD_OPTIONS_SELECTED:{options.get('selected_option')}")
    if recommendation.get("selected_option") != "canonical_structured_rule_metadata_parser_plus_existing_selected_pocket_binding":
        failures.append(f"BAD_RECOMMENDATION:{recommendation.get('selected_option')}")
    if recommendation.get("no_intermediate_micro_plan_after_144a") is not True:
        failures.append("MICRO_PLAN_GUARD_MISSING")
    for forbidden in [
        "winner=pocket_*",
        "selected_pocket_id",
        "per-row selected pocket request metadata",
        "per-row manifest switching",
        "payload marker list narrowed to correct pocket",
    ]:
        if forbidden not in anti.get("rule_derived_subsets_forbid", []):
            failures.append(f"ANTI_ORACLE_MISSING:{forbidden}")
    if decision.get("decision") != "structured_rule_metadata_to_selected_pocket_binding_prototype_plan_recommended":
        failures.append(f"BAD_DECISION:{decision.get('decision')}")
    if decision.get("selected_option") != "canonical_structured_rule_metadata_parser_plus_existing_selected_pocket_binding":
        failures.append(f"BAD_DECISION_OPTION:{decision.get('selected_option')}")
    if decision.get("next") != "144B_STRUCTURED_RULE_METADATA_TO_SELECTED_POCKET_BINDING_PROTOTYPE":
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
    parser = argparse.ArgumentParser(description="Check 144A structured rule metadata planning artifacts")
    parser.add_argument("--root", type=Path, default=SMOKE_ROOT)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--skip-changed-files-check", action="store_true")
    args = parser.parse_args()
    root = args.root if args.root.is_absolute() else (REPO_ROOT / args.root).resolve()
    failures = run_checks(root, check_changed_files=not args.skip_changed_files_check)
    if failures:
        print("144A CHECK FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("144A CHECK PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
