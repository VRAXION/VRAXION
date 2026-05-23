#!/usr/bin/env python3
"""Static/artifact checker for STABLE_LOOP_PHASE_LOCK_133."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_133_targeted_post_injection_repair_or_scale_plan/smoke"
REQUIRED_DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_133_TARGETED_POST_INJECTION_REPAIR_OR_SCALE_PLAN_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_133_TARGETED_POST_INJECTION_REPAIR_OR_SCALE_PLAN_RESULT.md",
]
REQUIRED_SOURCE = [
    "scripts/probes/run_stable_loop_phase_lock_133_targeted_post_injection_repair_or_scale_plan.py",
    "scripts/probes/run_stable_loop_phase_lock_133_targeted_post_injection_repair_or_scale_plan_check.py",
]
ALLOWED_MUTATIONS = set(REQUIRED_DOCS + REQUIRED_SOURCE)
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "analysis_config.json",
    "upstream_132_manifest.json",
    "upstream_131_manifest.json",
    "upstream_130_manifest.json",
    "upstream_129_manifest.json",
    "upstream_128_manifest.json",
    "upstream_127_manifest.json",
    "upstream_126_manifest.json",
    "upstream_123_manifest.json",
    "upstream_122_manifest.json",
    "upstream_119_manifest.json",
    "upstream_118_manifest.json",
    "upstream_112_manifest.json",
    "upstream_099_manifest.json",
    "post_injection_failure_priority_map.json",
    "breakpoint_analysis.json",
    "root_vs_symptom_analysis.json",
    "repair_target_selection.json",
    "training_design_options.json",
    "eval_gate_proposal.json",
    "risk_register.json",
    "111_failure_prevention_map.json",
    "next_milestone_plan.json",
    "decision.json",
    "summary.json",
    "report.md",
]
REQUIRED_TERMS = [
    "STABLE_LOOP_PHASE_LOCK_133_TARGETED_POST_INJECTION_REPAIR_OR_SCALE_PLAN",
    "planning only",
    "TARGETED_POST_INJECTION_REPAIR_OR_SCALE_PLAN_POSITIVE",
    "UPSTREAM_132_CEILING_MAP_VERIFIED",
    "STRUCTURED_OUTPUT_TOOL_API_TARGET_SELECTED",
    "134_STRUCTURED_OUTPUT_TOOL_API_REPAIR",
    "TIER_5_STRUCTURED_OUTPUT_AND_TOOL_API_LIKE",
    "structured_output_failure",
    "tool_api_schema_failure",
    "schema_valid_but_wrong_value_rate",
    "fake_tool_result_claim_rate",
    "not GPT-like assistant readiness",
    "not open-domain assistant readiness",
    "not production chat",
    "not public API",
    "not deployment readiness",
    "not safety alignment",
    "not Hungarian assistant readiness",
]
EXPECTED_EVIDENCE = {
    "first_breakpoint_tier": "TIER_5_STRUCTURED_OUTPUT_AND_TOOL_API_LIKE",
    "first_breakpoint_family": "structured_output_failure",
    "primary_next_repair_target": "structured_output_failure",
    "reasoning_preserved": True,
    "state_preserved": True,
    "calibration_preserved": True,
    "injection_priority_preserved": True,
    "unknown_failure_rate": 0.0,
}
EXPECTED_TIER5_COUNTS = {"structured_output_failure": 384, "tool_api_schema_failure": 384}
EXPECTED_GLOBAL_COUNTS = {
    "structured_output_failure": 384,
    "tool_api_schema_failure": 384,
    "format_failure": 384,
    "multi_doc_priority_failure": 192,
    "long_context_failure": 192,
}
NEGATION_MARKERS = ["not ", "no ", "does not ", "do not ", "without ", "false", "forbidden", "blocked", "denial"]
FORBIDDEN_CLAIMS = {
    "GPT_LIKE_READINESS_FALSE_CLAIM": ["GPT-like ready", "GPT-like assistant ready"],
    "OPEN_DOMAIN_ASSISTANT_CLAIM_DETECTED": ["open-domain assistant ready"],
    "PRODUCTION_CHAT_CLAIM_DETECTED": ["production chat ready", "production ready"],
    "PUBLIC_API_CLAIM_DETECTED": ["public API ready"],
    "DEPLOYMENT_CLAIM_DETECTED": ["deployment ready"],
    "SAFETY_ALIGNMENT_CLAIM_DETECTED": ["safety aligned"],
    "HUNGARIAN_ASSISTANT_CLAIM_DETECTED": ["Hungarian assistant ready"],
}


def git_status() -> str:
    result = subprocess.run(["git", "status", "--short"], cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    return result.stdout.rstrip("\n")


def changed_paths() -> list[str]:
    return [line[3:].replace("\\", "/") for line in git_status().splitlines() if line.strip()]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def claim_is_negated(text: str, phrase: str) -> bool:
    lowered = text.lower()
    phrase_lower = phrase.lower()
    for match in re.finditer(re.escape(phrase_lower), lowered):
        window = lowered[max(0, match.start() - 220) : match.start()]
        if any(marker in window for marker in NEGATION_MARKERS):
            return True
    return False


def find_false_claims(text: str) -> list[str]:
    failures: list[str] = []
    for verdict, phrases in FORBIDDEN_CLAIMS.items():
        for phrase in phrases:
            if phrase.lower() in text.lower() and not claim_is_negated(text, phrase):
                failures.append(verdict)
                break
    return failures


def runtime_surface_mutation_detected() -> bool:
    for path in changed_paths():
        if path in ALLOWED_MUTATIONS or path.startswith("target/"):
            continue
        return True
    return False


def read_files() -> tuple[list[str], dict[str, str]]:
    missing: list[str] = []
    files: dict[str, str] = {}
    for rel in REQUIRED_DOCS + REQUIRED_SOURCE:
        path = REPO_ROOT / rel
        if not path.exists():
            missing.append(rel)
            continue
        text = path.read_text(encoding="utf-8")
        if rel in REQUIRED_DOCS and len(text.strip()) < 200:
            missing.append(f"{rel}:too_short")
        files[rel] = text
    return missing, files


def require_plan_terms(plan_text: str) -> list[str]:
    failures: list[str] = []
    required_terms = [
        "A) exact JSON object generation",
        "B) schema-constrained object generation",
        "C) tool/API argument generation",
        "D) format conversion without semantic drift",
        "E) refusal or insufficient-info while preserving valid machine-readable envelope",
        "F) injected/distractor text must not appear in structured fields",
        "G) safe answer must remain structured when answerable",
        "exact JSON key/value rows",
        "schema-constrained JSON rows",
        "nested object and array rows",
        "tool/API argument rows",
        "function-call-like argument rows",
        "format conversion rows",
        "regex transform rows",
        "table-to-JSON rows",
        "multi-doc-to-structured rows",
        "insufficient-info structured refusal rows",
        "injection/distractor inside structured-output rows",
        "long-context schema rows",
        "state-carry structured rows",
        "json_validity_rate",
        "schema_validity_rate",
        "exact_key_match_accuracy",
        "exact_value_match_accuracy",
        "no_extra_fields_rate",
        "required_fields_present_rate",
        "tool_argument_name_accuracy",
        "tool_argument_value_accuracy",
        "nested_structure_accuracy",
        "array_order_accuracy",
        "format_conversion_accuracy",
        "regex_transform_accuracy",
        "table_to_json_accuracy",
        "multi_doc_structured_accuracy",
        "structured_refusal_envelope_accuracy",
        "long_context_schema_accuracy",
        "state_carry_structured_accuracy",
        "injection_resistant_structured_accuracy",
        "template_only_pass_rate",
        "static_json_pattern_rate",
        "schema_valid_but_wrong_value_rate",
        "correct_value_wrong_schema_rate",
        "extra_field_rate",
        "missing_required_field_rate",
        "wrong_argument_name_rate",
        "wrong_argument_value_rate",
        "injected_text_in_structured_field_rate",
        "fake_tool_result_claim_rate",
        "over_refusal_structured_rate",
        "STATIC_JSON_CONTROL",
        "COPY_SCHEMA_CONTROL",
        "TEMPLATE_ONLY_CONTROL",
        "RANDOM_JSON_CONTROL",
        "RANDOM_VALUE_CONTROL",
        "EXTRA_FIELD_CONTROL",
        "MISSING_FIELD_CONTROL",
        "COPY_INJECTED_TEXT_CONTROL",
        "INJECTED_ARGUMENT_CONTROL",
        "ALWAYS_REFUSE_JSON_CONTROL",
        "no fabricated tool output",
        "no claim that a tool was called",
        "correct function/tool name",
        "correct argument names",
        "correct argument values",
        "valid structured refusal envelope",
        "refusal reason field present",
        "no hallucinated argument values",
        "no free-prose-only refusal where JSON/schema is required",
        "raw-only final eval",
        "no teacher-forcing-only success",
        "no oracle rerank",
        "no expected-answer metadata",
        "no decoder reference",
        "no integrated policy during final eval",
        "no verifier rerank",
        "no LLM judge",
        "add public API",
        "modify runtime/tool execution",
        "call external tools",
        "add SDK exports",
        "touch docs/product",
        "touch docs/releases",
        "tier4_reasoning_accuracy",
        "multi_turn_state_accuracy",
        "answerable_fact_response_accuracy",
        "prompt_injection_rejection_accuracy",
    ]
    for term in required_terms:
        if term not in plan_text:
            failures.append(f"NEXT_MILESTONE_PLAN_MISSING:{term}")
    return failures


def check_artifacts() -> list[str]:
    failures: list[str] = []
    for rel in REQUIRED_ARTIFACTS:
        if not (SMOKE_ROOT / rel).exists():
            failures.append(f"MISSING_ARTIFACT:{rel}")
    if failures:
        return failures

    summary = load_json(SMOKE_ROOT / "summary.json")
    metrics = summary.get("metrics", {})
    verdicts = set(summary.get("verdicts", []))
    if "TARGETED_POST_INJECTION_REPAIR_OR_SCALE_PLAN_POSITIVE" not in verdicts:
        failures.append("TARGETED_POST_INJECTION_REPAIR_OR_SCALE_PLAN_RESULT_MISSING")
    for key, expected in [
        ("planning_only", True),
        ("analysis_only", True),
        ("training_performed", False),
        ("repair_performed", False),
        ("inference_run_count", 0),
        ("checkpoint_mutated", False),
        ("service_started", False),
        ("deployment_smoke_run", False),
        ("runtime_surface_mutated", False),
        ("bounded_release_stack_mutated", False),
        ("gpt_like_readiness_claimed", False),
        ("gpt_like_assistant_readiness_claimed", False),
        ("open_domain_assistant_readiness_claimed", False),
        ("production_chat_claimed", False),
        ("public_api_claimed", False),
        ("deployment_readiness_claimed", False),
        ("safety_alignment_claimed", False),
        ("hungarian_assistant_readiness_claimed", False),
    ]:
        if summary.get(key) != expected:
            failures.append("BOUNDARY_OR_SIDE_EFFECT_GATE_FAILED")
    for key, expected in [
        ("train_step_count", 0),
        ("optimizer_step_count", 0),
        ("inference_run_count", 0),
        ("service_started", False),
        ("deployment_smoke_run", False),
        ("checkpoint_mutated", False),
        ("bounded_release_artifact_unchanged", True),
        ("upstream_132_positive", True),
        ("failure_priority_map_written", True),
        ("breakpoint_analysis_written", True),
        ("root_vs_symptom_analysis_written", True),
        ("repair_target_selection_written", True),
        ("training_design_options_written", True),
        ("eval_gate_proposal_written", True),
        ("risk_register_written", True),
        ("next_milestone_plan_written", True),
        ("decision_written", True),
    ]:
        if metrics.get(key) != expected:
            failures.append("HARD_GATE_FAILED")
    for key, expected in EXPECTED_EVIDENCE.items():
        if metrics.get(key) != expected:
            failures.append("UPSTREAM_132_EVIDENCE_MISMATCH")

    progress_events = {row.get("event") for row in read_jsonl(SMOKE_ROOT / "progress.jsonl")}
    for event in [
        "startup",
        "upstream_verification",
        "132_artifact_loading",
        "failure_prioritization",
        "root_symptom_analysis",
        "repair_target_selection",
        "eval_gate_proposal",
        "decision_writing",
        "final_verdict",
    ]:
        if event not in progress_events:
            failures.append(f"PROGRESS_EVENT_MISSING:{event}")

    priority = load_json(SMOKE_ROOT / "post_injection_failure_priority_map.json")
    if priority.get("first_breakpoint_failure_counts") != EXPECTED_TIER5_COUNTS:
        failures.append("FAILURE_PRIORITY_MAP_MISSING")
    for key, expected in EXPECTED_GLOBAL_COUNTS.items():
        if priority.get("global_failure_counts", {}).get(key) != expected:
            failures.append("GLOBAL_FAILURE_EVIDENCE_MISSING")
    if priority.get("first_breakpoint_outranks_global_count") is not True:
        failures.append("FIRST_BREAKPOINT_RULE_MISSING")

    root_vs_symptom = load_json(SMOKE_ROOT / "root_vs_symptom_analysis.json")
    if root_vs_symptom.get("later_tier_target_selected_first") is not False or root_vs_symptom.get("first_breakpoint_outranks_global_count") is not True:
        failures.append("ROOT_VS_SYMPTOM_ANALYSIS_MISSING")

    selection = load_json(SMOKE_ROOT / "repair_target_selection.json")
    if selection.get("selected_next_milestone") != "134_STRUCTURED_OUTPUT_TOOL_API_REPAIR" or selection.get("selected_repair_target") != "structured_output_tool_api_first":
        failures.append("REPAIR_TARGET_SELECTION_MISSING")
    for key in [
        "why_structured_output_tool_api_first",
        "why_not_format_only_first",
        "why_not_long_context_first",
        "why_not_multi_doc_ambiguity_first",
        "why_not_more_general_sft",
        "why_not_deploy_polish",
        "why_not_architecture_pivot",
    ]:
        if not selection.get(key):
            failures.append("REJECTED_ALTERNATIVE_EXPLANATION_MISSING")

    next_plan = load_json(SMOKE_ROOT / "next_milestone_plan.json")
    plan_text = json.dumps(next_plan, sort_keys=True)
    if next_plan.get("milestone_name") != "134_STRUCTURED_OUTPUT_TOOL_API_REPAIR":
        failures.append("NEXT_MILESTONE_PLAN_MISSING")
    failures.extend(require_plan_terms(plan_text))

    decision = load_json(SMOKE_ROOT / "decision.json")
    if decision.get("selected_next_milestone") != "134_STRUCTURED_OUTPUT_TOOL_API_REPAIR" or decision.get("selected_repair_target") != "structured_output_tool_api_first":
        failures.append("DECISION_SELECTION_MISMATCH")
    for key, expected in EXPECTED_EVIDENCE.items():
        if decision.get(key) != expected:
            failures.append("DECISION_EVIDENCE_MISMATCH")
    if decision.get("structured_output_failure") != 384 or decision.get("tool_api_schema_failure") != 384:
        failures.append("DECISION_EVIDENCE_MISMATCH")
    for key in EXPECTED_GLOBAL_COUNTS:
        if decision.get("later_global_evidence", {}).get(key) != EXPECTED_GLOBAL_COUNTS[key]:
            failures.append("DECISION_GLOBAL_EVIDENCE_MISSING")
    for key in [
        "why_structured_output_tool_api_first",
        "why_not_format_only_first",
        "why_not_long_context_first",
        "why_not_multi_doc_ambiguity_first",
        "why_not_more_general_sft",
        "why_not_deploy_polish",
        "why_not_architecture_pivot",
    ]:
        if not decision.get(key):
            failures.append("DECISION_REJECTION_EXPLANATION_MISSING")

    text_to_scan = "\n".join([(SMOKE_ROOT / "report.md").read_text(encoding="utf-8"), json.dumps(summary, sort_keys=True)])
    for phrase in [
        "133 is planning only",
        "134 is planned as research repair only",
        "not GPT-like assistant readiness",
        "not open-domain assistant readiness",
        "not production chat",
        "not public API",
        "not deployment readiness",
        "not safety alignment",
        "not Hungarian assistant readiness",
    ]:
        if phrase not in text_to_scan:
            failures.append("BOUNDARY_TEXT_MISSING")
    failures.extend(find_false_claims(text_to_scan))
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    failures: list[str] = []
    missing, files = read_files()
    failures.extend(f"MISSING_REQUIRED_FILE:{item}" for item in missing)
    combined = "\n".join(files.values())
    for term in REQUIRED_TERMS:
        if term not in combined:
            failures.append(f"MISSING_REQUIRED_TERM:{term}")
    docs_text = "\n".join(files.get(rel, "") for rel in REQUIRED_DOCS)
    failures.extend(find_false_claims(docs_text))
    if runtime_surface_mutation_detected():
        failures.append("RUNTIME_SURFACE_MUTATION_DETECTED")
    if "LICENSE" in changed_paths():
        failures.append("ROOT_LICENSE_CHANGED")
    if args.check_only:
        failures.extend(check_artifacts())
    if failures:
        print("133 checker failed:")
        for failure in sorted(set(failures)):
            print(f" - {failure}")
        return 1
    print("133 checker passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
