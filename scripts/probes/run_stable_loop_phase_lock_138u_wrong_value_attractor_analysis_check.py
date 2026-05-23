#!/usr/bin/env python3
"""Checker for STABLE_LOOP_PHASE_LOCK_138U wrong-value attractor analysis."""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_138u_wrong_value_attractor_analysis/smoke"
RUNNER = "scripts/probes/run_stable_loop_phase_lock_138u_wrong_value_attractor_analysis.py"
CHECKER = "scripts/probes/run_stable_loop_phase_lock_138u_wrong_value_attractor_analysis_check.py"
DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_138U_WRONG_VALUE_ATTRACTOR_ANALYSIS_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_138U_WRONG_VALUE_ATTRACTOR_ANALYSIS_RESULT.md",
]
ALLOWED_MUTATIONS = {RUNNER, CHECKER, *DOCS}
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "upstream_138wv_manifest.json",
    "upstream_138w_manifest.json",
    "analysis_config.json",
    "wrong_value_distribution_report.json",
    "train_value_attractor_report.json",
    "eval_value_miss_report.json",
    "wrong_value_vs_prompt_report.json",
    "value_source_family_failure_report.json",
    "attractor_root_cause.json",
    "next_repair_recommendation.json",
    "diagnostic_gap_register.json",
    "risk_register.json",
    "decision.json",
    "summary.json",
    "report.md",
]
REQUIRED_TERMS = [
    "STABLE_LOOP_PHASE_LOCK_138U_WRONG_VALUE_ATTRACTOR_ANALYSIS",
    "wrong specific train-seen value",
    "global_train_value_prior_attractor",
    "high_frequency_train_value_attractor",
    "family_specific_train_value_attractor",
    "distractor_value_attractor",
    "wrong_table_entry_attractor",
    "output_head_value_prior",
    "diagnostic_gap",
    "wrong_value_attractor_analysis_complete",
    "138Y_VALUE_PRIOR_SUPPRESSION_AND_GROUNDING_OBJECTIVE_PLAN",
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
ATTRACTOR_SHAPES = {
    "global_single_wrong_value_attractor",
    "small_set_wrong_value_attractor",
    "family_specific_wrong_value_attractor",
    "seed_specific_wrong_value_attractor",
    "high_entropy_wrong_value_attractor",
}
KNOWN_ROOTS = {
    "global_train_value_prior_attractor",
    "family_specific_train_value_attractor",
    "high_frequency_train_value_attractor",
    "distractor_value_attractor",
    "wrong_table_entry_attractor",
    "prompt_copy_wrong_value_attractor",
    "output_head_value_prior",
    "mixed_wrong_value_attractors",
    "wrong_value_attractor_ambiguous",
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

    manifest_138wv = load_json(SMOKE_ROOT / "upstream_138wv_manifest.json")
    manifest_138w = load_json(SMOKE_ROOT / "upstream_138w_manifest.json")
    config = load_json(SMOKE_ROOT / "analysis_config.json")
    distribution = load_json(SMOKE_ROOT / "wrong_value_distribution_report.json")
    train_report = load_json(SMOKE_ROOT / "train_value_attractor_report.json")
    miss = load_json(SMOKE_ROOT / "eval_value_miss_report.json")
    prompt = load_json(SMOKE_ROOT / "wrong_value_vs_prompt_report.json")
    family = load_json(SMOKE_ROOT / "value_source_family_failure_report.json")
    root = load_json(SMOKE_ROOT / "attractor_root_cause.json")
    recommendation = load_json(SMOKE_ROOT / "next_repair_recommendation.json")
    gaps = load_json(SMOKE_ROOT / "diagnostic_gap_register.json")
    decision = load_json(SMOKE_ROOT / "decision.json")
    summary = load_json(SMOKE_ROOT / "summary.json")
    report = (SMOKE_ROOT / "report.md").read_text(encoding="utf-8")

    if manifest_138wv.get("decision") != "wrapper_value_decoupling_failure_analysis_complete" or manifest_138wv.get("next") != "138U_WRONG_VALUE_ATTRACTOR_ANALYSIS":
        failures.append("UPSTREAM_138WV_ROUTE_MISMATCH")
    for key, expected in {
        "wrong_specific_value_rate": 1.0,
        "train_seen_value_rate": 1.0,
        "expected_value_candidate_rate": 0.0,
        "immediate_termination_proxy_rate": 0.0,
        "default_neutral_attractor_rate": 0.0,
        "structural_format_echo_rate": 0.0,
        "unknown_post_wrapper_behavior_rate": 0.0,
    }.items():
        if manifest_138wv.get(key) != expected:
            failures.append("UPSTREAM_138WV_PROFILE_MISMATCH")
    if manifest_138wv.get("literal_eos_claimed") is not False or manifest_138wv.get("topological_inhibition_claim_status") != "hypothesis_only_diagnostic_gap_without_instrumentation":
        failures.append("UPSTREAM_138WV_GUARDRAIL_MISMATCH")
    if manifest_138w.get("helper_integrity_passed") is not True or manifest_138w.get("determinism_replay_passed") is not True:
        failures.append("RAW_HELPER_INTEGRITY_FAILURE")
    if manifest_138w.get("parrot_trap_detected") is not False or manifest_138w.get("stale_chat_fragment_rate") != 0.0 or manifest_138w.get("train_namespace_leak_rate") != 0.0:
        failures.append("UPSTREAM_138W_PROFILE_MISMATCH")

    if config.get("artifact_only") is not True or config.get("new_inference_run") is not False or config.get("shared_helper_called") is not False:
        failures.append("ARTIFACT_ONLY_BOUNDARY_FAILURE")

    row_count = distribution.get("row_count")
    rows = distribution.get("rows", [])
    if row_count != 768 or len(rows) != row_count:
        failures.append("ROW_COUNT_MISMATCH")
    if distribution.get("unique_wrong_value_count", 0) < 1 or distribution.get("attractor_shape") not in ATTRACTOR_SHAPES:
        failures.append("WRONG_VALUE_DISTRIBUTION_FAILURE")
    if distribution.get("most_common_wrong_value_rate", 0.0) < 0.50 and distribution.get("attractor_shape") != "family_specific_wrong_value_attractor":
        failures.append("DOMINANT_WRONG_VALUE_NOT_PROVEN")
    required_row_fields = {
        "row_id",
        "family",
        "seed",
        "expected_value",
        "generated_value_candidate",
        "generated_value_candidate_source_label",
        "prompt",
        "generated_text",
        "helper_trace_hash",
    }
    if any(not required_row_fields.issubset(row) for row in rows):
        failures.append("WRONG_VALUE_ROW_FIELD_MISSING")

    if not isinstance(train_report.get("generated_values_seen_in_train_rate"), (int, float)):
        failures.append("TRAIN_SEEN_VALUE_RATE_MISSING")
    for key in [
        "generated_values_seen_in_eval_expected_rate",
        "generated_values_seen_in_train_expected_rate",
        "generated_values_seen_in_train_prompt_rate",
        "train_value_frequency_rank_for_generated_values",
        "most_frequent_train_values",
        "generated_value_matches_most_frequent_train_value_rate",
    ]:
        if key not in train_report:
            failures.append("TRAIN_VALUE_ATTRACTOR_FIELD_MISSING")
    if miss.get("expected_value_candidate_rate") != 0.0 or miss.get("expected_value_never_emitted_rate") != 1.0:
        failures.append("EVAL_VALUE_MISS_PROFILE_MISMATCH")
    if prompt.get("wrong_value_unrelated_to_prompt_rate") is None or prompt.get("wrong_value_prompt_copy_rate") is None:
        failures.append("PROMPT_RELATION_RATE_MISSING")
    if len(family.get("families", {})) != 8:
        failures.append("FAMILY_BREAKDOWN_INCOMPLETE")
    if root.get("root_cause") not in KNOWN_ROOTS or recommendation.get("root_cause") != root.get("root_cause"):
        failures.append("ROOT_CAUSE_MISMATCH")
    expected_next_by_root = {
        "global_train_value_prior_attractor": "138Y_VALUE_PRIOR_SUPPRESSION_AND_GROUNDING_OBJECTIVE_PLAN",
        "high_frequency_train_value_attractor": "138Y_VALUE_PRIOR_SUPPRESSION_AND_GROUNDING_OBJECTIVE_PLAN",
        "family_specific_train_value_attractor": "138YF_FAMILY_SPECIFIC_VALUE_ATTRACTOR_REPAIR_PLAN",
        "distractor_value_attractor": "138YD_DISTRACTOR_AND_TABLE_SELECTION_REPAIR_PLAN",
        "wrong_table_entry_attractor": "138YD_DISTRACTOR_AND_TABLE_SELECTION_REPAIR_PLAN",
        "prompt_copy_wrong_value_attractor": "138P_PARROT_TRAP_VALUE_COPY_ANALYSIS",
        "output_head_value_prior": "138Y_VALUE_ACTIVATION_CARRIER_REDESIGN_PLAN",
        "mixed_wrong_value_attractors": "138Y_VALUE_ACTIVATION_CARRIER_REDESIGN_PLAN",
        "wrong_value_attractor_ambiguous": "138UB_WRONG_VALUE_ATTRACTOR_MANUAL_REVIEW_PACKET",
    }
    if recommendation.get("recommended_next") != expected_next_by_root.get(root.get("root_cause")):
        failures.append("RECOMMENDATION_ROUTE_MISMATCH")
    if root.get("root_cause") == "wrong_value_attractor_ambiguous":
        if decision.get("decision") != "wrong_value_attractor_ambiguous":
            failures.append("DECISION_MISMATCH")
    else:
        if decision.get("decision") != "wrong_value_attractor_analysis_complete":
            failures.append("DECISION_MISMATCH")
    if decision.get("next") != recommendation.get("recommended_next"):
        failures.append("DECISION_NEXT_MISMATCH")
    if not any(gap.get("field") == "output_head_value_prior" and gap.get("status") == "diagnostic_gap" for gap in gaps.get("gaps", [])):
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
        print("138U checker failed:")
        for failure in sorted(set(failures)):
            print(f" - {failure}")
        return 1
    print("138U checker passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
