#!/usr/bin/env python3
"""Checker for STABLE_LOOP_PHASE_LOCK_138WV wrapper/value decoupling analysis."""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_138wv_wrapper_value_decoupling_failure_analysis/smoke"
RUNNER = "scripts/probes/run_stable_loop_phase_lock_138wv_wrapper_value_decoupling_failure_analysis.py"
CHECKER = "scripts/probes/run_stable_loop_phase_lock_138wv_wrapper_value_decoupling_failure_analysis_check.py"
DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_138WV_WRAPPER_VALUE_DECOUPLING_FAILURE_ANALYSIS_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_138WV_WRAPPER_VALUE_DECOUPLING_FAILURE_ANALYSIS_RESULT.md",
]
ALLOWED_MUTATIONS = {RUNNER, CHECKER, *DOCS}
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "upstream_138w_manifest.json",
    "analysis_config.json",
    "post_wrapper_value_anatomy_report.json",
    "silence_taxonomy_report.json",
    "attractor_distribution_report.json",
    "value_candidate_report.json",
    "parrot_and_derivation_recheck.json",
    "wrapper_value_decoupling_root_cause.json",
    "next_repair_recommendation.json",
    "diagnostic_gap_register.json",
    "risk_register.json",
    "decision.json",
    "summary.json",
    "report.md",
]
REQUIRED_TERMS = [
    "STABLE_LOOP_PHASE_LOCK_138WV_WRAPPER_VALUE_DECOUPLING_FAILURE_ANALYSIS",
    "immediate_termination_proxy",
    "literal EOS",
    "stop_reason = max_new_tokens",
    "topological inhibition",
    "diagnostic_gap",
    "wrong_specific_value_attractor_dominant",
    "138U_WRONG_VALUE_ATTRACTOR_ANALYSIS",
    "artifact-only",
    "Raw assistant capability remains quarantined",
    "Structured/tool capability remains invalidated",
    "not GPT-like readiness",
    "not open-domain assistant readiness",
    "not production chat",
    "not public API",
    "not deployment readiness",
    "not safety alignment",
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
PRIMARY_LABELS = {
    "immediate_termination_proxy",
    "empty_or_whitespace_after_wrapper",
    "default_neutral_attractor",
    "structural_format_echo",
    "generic_wrong_value",
    "repeated_symbol_or_punctuation",
    "wrong_specific_value",
    "delayed_correct_value_wrong_position",
    "garbled_after_wrapper",
    "unknown_post_wrapper_behavior",
}
KNOWN_ROOTS = {
    "wrapper_termination_proxy_dominant",
    "default_neutral_attractor_dominant",
    "structural_format_echo_dominant",
    "wrong_specific_value_attractor_dominant",
    "mixed_post_wrapper_attractors",
    "wrapper_value_decoupling_ambiguous",
}
NEGATION_MARKERS = ["not ", "no ", "does not ", "do not ", "without ", "false", "forbidden", "remains quarantined", "remains invalidated"]
FORBIDDEN_CLAIMS = {
    "REASONING_RESTORED_FALSE_CLAIM": ["reasoning restored", "reasoning is restored"],
    "RAW_ASSISTANT_CAPABILITY_RESTORED_FALSE_CLAIM": ["raw assistant capability restored"],
    "STRUCTURED_TOOL_CAPABILITY_RESTORED_FALSE_CLAIM": ["structured/tool capability restored", "structured tool capability restored"],
    "GPT_LIKE_READINESS_FALSE_CLAIM": ["GPT-like ready", "GPT-like readiness improved"],
    "OPEN_DOMAIN_ASSISTANT_CLAIM_DETECTED": ["open-domain assistant ready"],
    "PRODUCTION_CHAT_CLAIM_DETECTED": ["production chat ready", "production ready"],
    "PUBLIC_API_CLAIM_DETECTED": ["public API ready"],
    "DEPLOYMENT_CLAIM_DETECTED": ["deployment ready"],
    "SAFETY_ALIGNMENT_CLAIM_DETECTED": ["safety aligned"],
    "LITERAL_EOS_CLAIM_DETECTED": ["literal EOS occurred", "true EOS behavior", "EOS-token evidence"],
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
        window = lowered[max(0, match.start() - 260) : match.start()]
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
    for rel in [RUNNER, CHECKER, *DOCS]:
        path = REPO_ROOT / rel
        if not path.exists():
            missing.append(rel)
            continue
        text = path.read_text(encoding="utf-8")
        if rel in DOCS and len(text.strip()) < 200:
            missing.append(f"{rel}:too_short")
        files[rel] = text
    return missing, files


def ast_scan(paths: list[Path]) -> list[str]:
    failures: list[str] = []
    forbidden_imports = {"torch", "shared_raw_generation_helper"}
    forbidden_calls = {"raw_generate", "load_checkpoint", "manual_seed", "forward", "backward", "optimizer", "train_target_model"}
    for path in paths:
        if not path.exists():
            failures.append(f"MISSING_AST_SOURCE:{path}")
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in forbidden_imports:
                        failures.append("INFERENCE_SIDE_EFFECT_DETECTED")
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module in forbidden_imports or module.startswith("run_stable_loop_phase_lock_"):
                    failures.append("INFERENCE_SIDE_EFFECT_DETECTED")
            if isinstance(node, ast.Call):
                name = ast.unparse(node.func)
                if any(token in name for token in forbidden_calls):
                    failures.append("INFERENCE_SIDE_EFFECT_DETECTED")
    return failures


def require_false_flags(payload: dict[str, Any], failures: list[str]) -> None:
    for key in FALSE_FLAGS:
        if payload.get(key) is not False:
            failures.append("BOUNDARY_CLAIM_FAILURE")


def check_artifacts() -> list[str]:
    failures: list[str] = []
    for rel in REQUIRED_ARTIFACTS:
        if not (SMOKE_ROOT / rel).exists():
            failures.append(f"MISSING_ARTIFACT:{rel}")
    if failures:
        return failures

    progress = read_jsonl(SMOKE_ROOT / "progress.jsonl")
    if len(progress) < 10:
        failures.append("PROGRESS_NOT_REFRESHED")

    manifest = load_json(SMOKE_ROOT / "upstream_138w_manifest.json")
    config = load_json(SMOKE_ROOT / "analysis_config.json")
    anatomy = load_json(SMOKE_ROOT / "post_wrapper_value_anatomy_report.json")
    taxonomy = load_json(SMOKE_ROOT / "silence_taxonomy_report.json")
    distribution = load_json(SMOKE_ROOT / "attractor_distribution_report.json")
    candidates = load_json(SMOKE_ROOT / "value_candidate_report.json")
    parrot = load_json(SMOKE_ROOT / "parrot_and_derivation_recheck.json")
    root = load_json(SMOKE_ROOT / "wrapper_value_decoupling_root_cause.json")
    recommendation = load_json(SMOKE_ROOT / "next_repair_recommendation.json")
    gaps = load_json(SMOKE_ROOT / "diagnostic_gap_register.json")
    decision = load_json(SMOKE_ROOT / "decision.json")
    summary = load_json(SMOKE_ROOT / "summary.json")
    report = (SMOKE_ROOT / "report.md").read_text(encoding="utf-8")

    if manifest.get("decision") != "wrapper_success_without_value_grounding_persists" or manifest.get("next") != "138WV_WRAPPER_VALUE_DECOUPLING_FAILURE_ANALYSIS":
        failures.append("UPSTREAM_138W_ROUTE_MISMATCH")
    for key, expected in {
        "answer_prefix_accuracy": 1.0,
        "eval_namespace_emission_accuracy": 1.0,
        "answer_value_accuracy": 0.0,
        "exact_answer_accuracy": 0.0,
        "value_after_prefix_accuracy": 0.0,
        "post_wrapper_garbage_token_rate": 0.0,
        "stale_chat_fragment_rate": 0.0,
        "train_namespace_leak_rate": 0.0,
    }.items():
        if manifest.get(key) != expected:
            failures.append("UPSTREAM_138W_PROFILE_MISMATCH")
    if manifest.get("stop_reason_is_max_new_tokens") is not True or manifest.get("literal_eos_artifact_present") is not False:
        failures.append("LITERAL_EOS_GUARDRAIL_FAILURE")
    if manifest.get("helper_integrity_passed") is not True or manifest.get("determinism_replay_passed") is not True:
        failures.append("RAW_HELPER_INTEGRITY_FAILURE")

    if config.get("artifact_only") is not True or config.get("new_inference_run") is not False or config.get("shared_helper_called") is not False:
        failures.append("ARTIFACT_ONLY_BOUNDARY_FAILURE")
    if config.get("literal_eos_claimed") is not False or config.get("immediate_termination_proxy_not_literal_eos") is not True:
        failures.append("LITERAL_EOS_GUARDRAIL_FAILURE")

    row_count = anatomy.get("row_count")
    anatomy_rows = anatomy.get("rows", [])
    taxonomy_rows = taxonomy.get("rows", [])
    if row_count != 768 or len(anatomy_rows) != row_count or len(taxonomy_rows) != row_count:
        failures.append("ROW_COUNT_MISMATCH")
    required_anatomy_fields = {
        "row_id",
        "family",
        "seed",
        "generated_text",
        "expected_output",
        "post_wrapper_text",
        "post_wrapper_first_token",
        "post_wrapper_first_nonspace_token",
        "post_wrapper_length_chars",
        "post_wrapper_length_tokens_or_bytes",
        "correct_value_present_after_wrapper",
        "expected_value_position",
        "generated_value_candidate",
        "value_candidate_source_label",
    }
    if any(not required_anatomy_fields.issubset(row) for row in anatomy_rows):
        failures.append("POST_WRAPPER_ANATOMY_FIELD_MISSING")
    labels = [row.get("primary_post_wrapper_class") for row in taxonomy_rows]
    if any(label not in PRIMARY_LABELS for label in labels) or taxonomy.get("exactly_one_primary_label_per_row") is not True:
        failures.append("TAXONOMY_PRIMARY_LABEL_FAILURE")
    if sum(taxonomy.get("primary_label_counts", {}).values()) != row_count:
        failures.append("TAXONOMY_COUNT_MISMATCH")
    for label in PRIMARY_LABELS:
        if f"{label}_rate" not in distribution:
            failures.append("ATTRACTOR_RATE_MISSING")
    if distribution.get("row_count") != row_count:
        failures.append("ATTRACTOR_ROW_COUNT_MISMATCH")
    if root.get("root_cause") not in KNOWN_ROOTS or recommendation.get("root_cause") != root.get("root_cause"):
        failures.append("ROOT_CAUSE_MISMATCH")
    if root.get("literal_eos_claimed") is not False or root.get("immediate_eos_rate") != "not_computed_literal_eos_not_available":
        failures.append("LITERAL_EOS_GUARDRAIL_FAILURE")
    if root.get("topological_inhibition_claim_status") != "hypothesis_only_diagnostic_gap_without_instrumentation":
        failures.append("TOPOLOGICAL_OVERCLAIM")
    expected_next_by_root = {
        "wrapper_termination_proxy_dominant": "138X_WRAPPER_TERMINATION_VALUE_PATH_REPAIR_PLAN",
        "default_neutral_attractor_dominant": "138Y_VALUE_ACTIVATION_CARRIER_REDESIGN_PLAN",
        "structural_format_echo_dominant": "138Z_SEMANTIC_VS_FORMAT_OBJECTIVE_REBALANCE_PLAN",
        "wrong_specific_value_attractor_dominant": "138U_WRONG_VALUE_ATTRACTOR_ANALYSIS",
        "mixed_post_wrapper_attractors": "138WB_WRAPPER_VALUE_DECOUPLING_MANUAL_REVIEW_PACKET",
        "wrapper_value_decoupling_ambiguous": "138WB_WRAPPER_VALUE_DECOUPLING_MANUAL_REVIEW_PACKET",
    }
    if recommendation.get("recommended_next") != expected_next_by_root.get(root.get("root_cause")):
        failures.append("RECOMMENDATION_ROUTE_MISMATCH")
    if root.get("root_cause") == "wrapper_value_decoupling_ambiguous":
        if decision.get("decision") != "wrapper_value_decoupling_ambiguous":
            failures.append("DECISION_MISMATCH")
    else:
        if decision.get("decision") != "wrapper_value_decoupling_failure_analysis_complete":
            failures.append("DECISION_MISMATCH")
    if decision.get("next") != recommendation.get("recommended_next") or decision.get("literal_eos_claimed") is not False:
        failures.append("DECISION_MISMATCH")

    for key in [
        "no_candidate_rate",
        "neutral_default_rate",
        "generic_value_rate",
        "train_seen_value_rate",
        "wrong_specific_value_rate",
        "expected_value_candidate_rate",
    ]:
        if key not in candidates:
            failures.append("VALUE_CANDIDATE_RATE_MISSING")
    if parrot.get("value_grounding_absent_confirmed") is not True:
        failures.append("PARROT_DERIVATION_RECHECK_FAILURE")
    if not any(gap.get("field") == "literal_eos_evidence" and gap.get("status") == "diagnostic_gap" for gap in gaps.get("gaps", [])):
        failures.append("DIAGNOSTIC_GAP_MISSING")

    require_false_flags(decision, failures)
    require_false_flags(summary, failures)
    failures.extend(find_false_claims(json.dumps(summary) + "\n" + report))
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    _args = parser.parse_args()
    failures: list[str] = []
    missing, files = read_files()
    failures.extend(f"MISSING_FILE:{item}" for item in missing)
    combined_text = "\n".join(files.values())
    docs_text = "\n".join(files.get(rel, "") for rel in DOCS)
    for term in REQUIRED_TERMS:
        if term not in combined_text:
            failures.append(f"MISSING_REQUIRED_TERM:{term}")
    failures.extend(find_false_claims(docs_text))
    if runtime_surface_mutation_detected():
        failures.append("UNAUTHORIZED_REPO_MUTATION_DETECTED")
    failures.extend(ast_scan([REPO_ROOT / RUNNER, REPO_ROOT / CHECKER]))
    failures.extend(check_artifacts())
    if failures:
        print("138WV checker failed:")
        for failure in sorted(set(failures)):
            print(f" - {failure}")
        return 1
    print("138WV checker passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
