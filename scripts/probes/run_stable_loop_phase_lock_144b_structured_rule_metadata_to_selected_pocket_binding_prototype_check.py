#!/usr/bin/env python3
"""Checker for 144B structured rule metadata to selected-pocket binding prototype."""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_144b_structured_rule_metadata_to_selected_pocket_binding_prototype/smoke"
HELPER = "scripts/probes/shared_raw_generation_helper.py"
RUNNER = "scripts/probes/run_stable_loop_phase_lock_144b_structured_rule_metadata_to_selected_pocket_binding_prototype.py"
CHECKER = "scripts/probes/run_stable_loop_phase_lock_144b_structured_rule_metadata_to_selected_pocket_binding_prototype_check.py"
DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_144B_STRUCTURED_RULE_METADATA_TO_SELECTED_POCKET_BINDING_PROTOTYPE_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_144B_STRUCTURED_RULE_METADATA_TO_SELECTED_POCKET_BINDING_PROTOTYPE_RESULT.md",
]
ALLOWED_MUTATIONS = {HELPER, RUNNER, CHECKER, *DOCS}
ALLOWED_HELPER_KEYS = {"prompt", "checkpoint_path", "checkpoint_hash", "seed", "max_new_tokens", "generation_config"}
NEW_DECODER = "deterministic_pocket_gated_structured_rule_metadata_binding_decoder"
OLD_DECODER = "deterministic_pocket_gated_rule_selected_pocket_binding_decoder"
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "analysis_config.json",
    "upstream_144a_manifest.json",
    "shared_helper_diff_audit.json",
    "structured_rule_metadata_parser_report.json",
    "derived_selected_pocket_report.json",
    "selected_pocket_binding_report.json",
    "rule_metadata_ablation_report.json",
    "explicit_winner_baseline_report.json",
    "rule_metadata_corruption_report.json",
    "missing_rule_metadata_report.json",
    "ambiguous_rule_metadata_report.json",
    "hierarchy_policy_report.json",
    "legacy_selected_pocket_binding_regression_report.json",
    "static_manifest_integrity_report.json",
    "helper_request_audit.json",
    "prompt_scanner_report.json",
    "aggregate_metrics.json",
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


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def git_status() -> str:
    result = subprocess.run(["git", "status", "--short"], cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    return result.stdout.rstrip("\n")


def changed_paths() -> list[str]:
    paths: list[str] = []
    for line in git_status().splitlines():
        if line.strip():
            paths.append(line[3:].replace("\\", "/"))
    return paths


def require_changed_files(failures: list[str]) -> None:
    for path in changed_paths():
        if path.startswith("target/"):
            continue
        if path not in ALLOWED_MUTATIONS:
            failures.append(f"UNEXPECTED_CHANGED_FILE:{path}")


def require_artifacts(root: Path, failures: list[str]) -> None:
    for name in REQUIRED_ARTIFACTS:
        if not (root / name).exists():
            failures.append(f"MISSING_ARTIFACT:{name}")


def require_false_flags(payload: dict[str, Any], failures: list[str], prefix: str) -> None:
    for key in FALSE_FLAGS:
        if payload.get(key) is not False:
            failures.append(f"{prefix}_BOUNDARY_FLAG_NOT_FALSE:{key}:{payload.get(key)}")


def ast_scan(failures: list[str]) -> None:
    for rel_path in [RUNNER, CHECKER]:
        path = REPO_ROOT / rel_path
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if rel_path == CHECKER and isinstance(node, ast.Call):
                name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ""
                if name == "raw_generate":
                    failures.append("CHECKER_RAW_GENERATE_NOT_ALLOWED")
            if isinstance(node, ast.Call):
                name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ""
                if name in {"train", "fit", "backward", "step", "forward"}:
                    failures.append(f"TRAINING_CALL_NOT_ALLOWED:{rel_path}:{name}")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if rel_path == CHECKER and alias.name == "shared_raw_generation_helper":
                        failures.append("CHECKER_IMPORTS_SHARED_HELPER")


def require_static_files(failures: list[str]) -> None:
    for rel_path in [RUNNER, CHECKER, *DOCS]:
        path = REPO_ROOT / rel_path
        if not path.exists():
            failures.append(f"MISSING_TRACKED_FILE:{rel_path}")
            continue
        text = path.read_text(encoding="utf-8")
        for term in ["144B", NEW_DECODER, "structured rule metadata"]:
            if term not in text:
                failures.append(f"TERM_MISSING:{rel_path}:{term}")
        for phrase in BOUNDARY_PHRASES:
            if phrase not in text:
                failures.append(f"BOUNDARY_TERM_MISSING:{rel_path}:{phrase}")
    helper_source = (REPO_ROOT / HELPER).read_text(encoding="utf-8")
    for term in [NEW_DECODER, OLD_DECODER, "_instnct_select_structured_rule_metadata_value", "_instnct_parse_rule_metadata", "_instnct_derive_selected_pocket"]:
        if term not in helper_source:
            failures.append(f"HELPER_TERM_MISSING:{term}")
    if "prompt.find(selected_marker)" in helper_source:
        failures.append("HELPER_REGRESSED_TO_PROMPT_FIND_SELECTED_MARKER")
    if "candidate_line_re" not in helper_source or "re.escape(selected_marker)" not in helper_source:
        failures.append("HELPER_CANDIDATE_LINE_REGEX_MISSING")


def require_upstream(root: Path, failures: list[str]) -> None:
    upstream = load_json(root / "upstream_144a_manifest.json")
    decision = upstream.get("decision", {})
    target = upstream.get("target_144b_milestone_plan", {})
    grammar = upstream.get("structured_rule_metadata_grammar_spec", {})
    if decision.get("decision") != "structured_rule_metadata_to_selected_pocket_binding_prototype_plan_recommended":
        failures.append(f"BAD_144A_DECISION:{decision.get('decision')}")
    if decision.get("selected_option") != "canonical_structured_rule_metadata_parser_plus_existing_selected_pocket_binding":
        failures.append(f"BAD_144A_OPTION:{decision.get('selected_option')}")
    if decision.get("next") != "144B_STRUCTURED_RULE_METADATA_TO_SELECTED_POCKET_BINDING_PROTOTYPE":
        failures.append(f"BAD_144A_NEXT:{decision.get('next')}")
    if target.get("decoder_name") != NEW_DECODER or target.get("implementation_ready") is not True:
        failures.append("144A_TARGET_NOT_IMPLEMENTATION_READY")
    if target.get("existing_selected_pocket_decoder") != OLD_DECODER:
        failures.append(f"144A_BAD_OLD_DECODER:{target.get('existing_selected_pocket_decoder')}")
    if grammar.get("free_form_natural_language_rule_parsing_allowed") is not False:
        failures.append("144A_GRAMMAR_ALLOWS_FREE_FORM_NL")


def require_helper_diff(root: Path, failures: list[str]) -> None:
    audit = load_json(root / "shared_helper_diff_audit.json")
    expected_true = [
        "source_changed",
        "new_decoder_string_present",
        "new_binding_function_present",
        "new_parser_helpers_present",
        "new_behavior_manifest_gated",
        "old_selected_pocket_binding_decoder_present",
        "old_selected_pocket_binding_function_unchanged",
        "validate_request_unchanged",
        "allowed_request_keys_unchanged",
        "forbidden_request_keys_not_loosened",
        "raw_generate_unchanged",
        "non_instnct_generation_path_unchanged",
        "no_training_import_added",
        "no_network_or_io_added",
        "passed",
    ]
    for key in expected_true:
        if audit.get(key) is not True:
            failures.append(f"HELPER_DIFF_FAIL:{key}:{audit.get(key)}")
    current = (REPO_ROOT / HELPER).read_text(encoding="utf-8")
    if audit.get("helper_source_sha256_after") and audit["helper_source_sha256_after"] != __import__("hashlib").sha256(current.encode("utf-8", errors="replace")).hexdigest():
        failures.append("HELPER_SOURCE_DOES_NOT_MATCH_AUDIT_AFTER_HASH")


def require_audits(root: Path, failures: list[str]) -> None:
    request = load_json(root / "helper_request_audit.json")
    static = load_json(root / "static_manifest_integrity_report.json")
    prompt = load_json(root / "prompt_scanner_report.json")
    legacy = load_json(root / "legacy_selected_pocket_binding_regression_report.json")
    if set(request.get("allowed_helper_keys", [])) != ALLOWED_HELPER_KEYS:
        failures.append(f"BAD_ALLOWED_HELPER_KEYS:{request.get('allowed_helper_keys')}")
    for key in ["all_requests_allowed_keys_only", "selected_pocket_id_not_in_request_metadata", "winner_label_not_in_request_metadata", "passed"]:
        if request.get(key) is not True:
            failures.append(f"REQUEST_AUDIT_FAIL:{key}:{request.get(key)}")
    if request.get("helper_request_forbidden_metadata_count") != 0:
        failures.append(f"FORBIDDEN_METADATA:{request.get('helper_request_forbidden_metadata_count')}")
    if request.get("raw_generate_allowed_in_runner") is not True or request.get("raw_generate_allowed_in_checker") is not False:
        failures.append("RAW_GENERATE_BOUNDARY_BAD")
    if static.get("passed") is not True or static.get("per_row_manifest_switch_rate") != 0.0 or static.get("per_row_payload_marker_switch_rate") != 0.0:
        failures.append("STATIC_MANIFEST_INTEGRITY_FAIL")
    if static.get("payload_marker_list_narrowed_to_correct_pocket") is not False:
        failures.append("PAYLOAD_MARKER_LIST_NARROWED")
    if prompt.get("passed") is not True or prompt.get("forbidden_violation_count") != 0:
        failures.append("PROMPT_SCANNER_FAIL")
    if legacy.get("legacy_selected_pocket_binding_regression_passed") is not True:
        failures.append("LEGACY_SELECTED_POCKET_BINDING_REGRESSION_FAIL")
    if legacy.get("new_decoder_activation_rate_under_old_decoder") != 0.0:
        failures.append(f"NEW_DECODER_ACTIVATED_UNDER_OLD:{legacy.get('new_decoder_activation_rate_under_old_decoder')}")


def require_reports(root: Path, failures: list[str]) -> None:
    for name in [
        "structured_rule_metadata_parser_report.json",
        "derived_selected_pocket_report.json",
        "selected_pocket_binding_report.json",
        "rule_metadata_ablation_report.json",
        "explicit_winner_baseline_report.json",
        "rule_metadata_corruption_report.json",
        "missing_rule_metadata_report.json",
        "ambiguous_rule_metadata_report.json",
        "hierarchy_policy_report.json",
    ]:
        report = load_json(root / name)
        if report.get("passed") is not True:
            failures.append(f"REPORT_NOT_PASSED:{name}")


def require_metrics(root: Path, failures: list[str]) -> None:
    metrics = load_json(root / "aggregate_metrics.json")
    for key, threshold in {
        "rule_metadata_parse_accuracy": 0.90,
        "derived_selected_pocket_accuracy": 0.90,
        "selected_pocket_to_marker_binding_accuracy": 0.95,
        "same_line_value_extraction_accuracy": 0.95,
        "end_to_end_answer_accuracy": 0.90,
        "rule_derived_no_winner_label_accuracy": 0.90,
        "explicit_winner_baseline_accuracy": 0.95,
        "corrupt_rule_metadata_rejection_rate": 0.90,
        "missing_rule_metadata_fallback_rate": 0.90,
        "ambiguous_rule_metadata_rejection_rate": 0.90,
    }.items():
        if metrics.get(key, 0.0) < threshold:
            failures.append(f"METRIC_BELOW_GATE:{key}:{metrics.get(key)}")
    if metrics.get("rule_metadata_ablation_accuracy", 1.0) > 0.15:
        failures.append(f"RULE_METADATA_ABLATION_TOO_HIGH:{metrics.get('rule_metadata_ablation_accuracy')}")
    for key in ["helper_request_forbidden_metadata_count", "per_row_manifest_switch_rate", "per_row_payload_marker_switch_rate"]:
        if metrics.get(key) != 0 and metrics.get(key) != 0.0:
            failures.append(f"METRIC_NOT_ZERO:{key}:{metrics.get(key)}")
    if metrics.get("legacy_selected_pocket_binding_regression_passed") is not True or metrics.get("deterministic_replay_passed") is not True:
        failures.append("LEGACY_OR_DETERMINISM_FAIL")


def require_decision_and_boundary(root: Path, failures: list[str]) -> None:
    decision = load_json(root / "decision.json")
    summary = load_json(root / "summary.json")
    report = (root / "report.md").read_text(encoding="utf-8")
    if decision.get("decision") != "structured_rule_metadata_to_selected_pocket_binding_prototype_positive":
        failures.append(f"BAD_DECISION:{decision.get('decision')}")
    if decision.get("verdict") != "INSTNCT_STRUCTURED_RULE_METADATA_TO_SELECTED_POCKET_BINDING_PROTOTYPE_POSITIVE":
        failures.append(f"BAD_VERDICT:{decision.get('verdict')}")
    if decision.get("next") != "144H_STRUCTURED_RULE_METADATA_TO_SELECTED_POCKET_BINDING_SCALE_CONFIRM":
        failures.append(f"BAD_NEXT:{decision.get('next')}")
    for payload_name, payload in [("decision", decision), ("summary", summary)]:
        require_false_flags(payload, failures, payload_name)
        text = json.dumps(payload)
        for phrase in BOUNDARY_PHRASES:
            if phrase not in text:
                failures.append(f"BOUNDARY_MISSING:{payload_name}:{phrase}")
    for phrase in BOUNDARY_PHRASES:
        if phrase not in report:
            failures.append(f"REPORT_BOUNDARY_MISSING:{phrase}")
    for doc in DOCS:
        text = (REPO_ROOT / doc).read_text(encoding="utf-8")
        if len(text.strip()) < 500:
            failures.append(f"DOC_TOO_SHORT:{doc}")
        for phrase in BOUNDARY_PHRASES:
            if phrase not in text:
                failures.append(f"DOC_BOUNDARY_MISSING:{doc}:{phrase}")


def run_checks(root: Path, check_changed_files: bool) -> list[str]:
    failures: list[str] = []
    if check_changed_files:
        require_changed_files(failures)
    require_static_files(failures)
    ast_scan(failures)
    require_artifacts(root, failures)
    if failures:
        return failures
    require_upstream(root, failures)
    require_helper_diff(root, failures)
    require_audits(root, failures)
    require_reports(root, failures)
    require_metrics(root, failures)
    require_decision_and_boundary(root, failures)
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Check 144B structured rule metadata to selected-pocket binding prototype")
    parser.add_argument("--root", type=Path, default=SMOKE_ROOT)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--skip-changed-files-check", action="store_true")
    args = parser.parse_args()
    root = args.root if args.root.is_absolute() else (REPO_ROOT / args.root).resolve()
    failures = run_checks(root, check_changed_files=not args.skip_changed_files_check)
    if failures:
        print("144B CHECK FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("144B CHECK PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
