#!/usr/bin/env python3
"""Checker for STABLE_LOOP_PHASE_LOCK_138S stale-chat rollout analysis."""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_138s_stale_chat_rollout_failure_analysis/smoke"
RUNNER = "scripts/probes/run_stable_loop_phase_lock_138s_stale_chat_rollout_failure_analysis.py"
CHECKER = "scripts/probes/run_stable_loop_phase_lock_138s_stale_chat_rollout_failure_analysis_check.py"
DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_138S_STALE_CHAT_ROLLOUT_FAILURE_ANALYSIS_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_138S_STALE_CHAT_ROLLOUT_FAILURE_ANALYSIS_RESULT.md",
]
ALLOWED_MUTATIONS = {RUNNER, CHECKER, *DOCS}
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "upstream_138i_manifest.json",
    "analysis_config.json",
    "stale_chat_distribution_report.json",
    "value_grounding_failure_report.json",
    "prefix_vs_value_decoupling_report.json",
    "source_prior_vs_training_objective_report.json",
    "output_pattern_taxonomy.json",
    "stale_chat_value_coupling_report.json",
    "diagnostic_gap_register.json",
    "next_repair_recommendation.json",
    "risk_register.json",
    "decision.json",
    "summary.json",
    "report.md",
]
REQUIRED_TERMS = [
    "STABLE_LOOP_PHASE_LOCK_138S_STALE_CHAT_ROLLOUT_FAILURE_ANALYSIS",
    "artifact-only stale-chat rollout failure analysis",
    "stale_chat_rollout_failure",
    "post_answer_value_accuracy",
    "value_grounding_fails_without_stale_fragments",
    "prefix_success_value_failure_rate",
    "138V_ANSWER_VALUE_GROUNDING_OBJECTIVE_REDESIGN_PLAN",
    "138T_STALE_CHAT_SUPPRESSION_AND_VALUE_GROUNDING_REPAIR_PLAN",
    "raw assistant capability remains quarantined",
    "structured/tool capability remains invalidated",
    "Not GPT-like readiness",
    "Not open-domain assistant readiness",
    "Not production chat",
    "Not public API",
    "Not deployment readiness",
    "Not safety alignment",
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
NEGATION_MARKERS = ["not ", "no ", "does not ", "do not ", "without ", "false", "remains quarantined", "remains invalidated", "artifact-only"]
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
    paths: list[str] = []
    for line in git_status().splitlines():
        if line.strip():
            paths.append(line[3:].replace("\\", "/"))
    return paths


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def claim_is_negated(text: str, phrase: str) -> bool:
    lowered = text.lower()
    phrase_lower = phrase.lower()
    for match in re.finditer(re.escape(phrase_lower), lowered):
        window = lowered[max(0, match.start() - 240) : match.start()]
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


def ast_scan_source(paths: list[Path]) -> list[str]:
    failures: list[str] = []
    forbidden_imports = {"torch", "shared_raw_generation_helper"}
    old_runner_re = re.compile(r"^run_stable_loop_phase_lock_")
    forbidden_calls = {"raw_generate", "manual_seed", "forward", "backward", "optimizer", "load_checkpoint"}
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in forbidden_imports or old_runner_re.match(alias.name):
                        failures.append("INFERENCE_SIDE_EFFECT_DETECTED")
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module in forbidden_imports or old_runner_re.match(module):
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

    progress = load_jsonl(SMOKE_ROOT / "progress.jsonl")
    if len(progress) < 8:
        failures.append("PROGRESS_NOT_REFRESHED")

    manifest = load_json(SMOKE_ROOT / "upstream_138i_manifest.json")
    if manifest.get("upstream_138i_verified") is not True:
        failures.append("UPSTREAM_138I_ARTIFACT_MISSING")
    if manifest.get("decision") != "stale_chat_rollout_failure" or manifest.get("next") != "138S_STALE_CHAT_ROLLOUT_FAILURE_ANALYSIS":
        failures.append("UPSTREAM_138I_ROUTE_MISMATCH")
    if manifest.get("post_train_namespace_leak_rate") != 0.0 or manifest.get("post_eval_namespace_emission_accuracy") != 1.0:
        failures.append("UPSTREAM_138I_METRIC_PROFILE_MISMATCH")
    if manifest.get("post_answer_prefix_accuracy") != 1.0 or manifest.get("post_answer_value_accuracy") != 0.0:
        failures.append("UPSTREAM_138I_METRIC_PROFILE_MISMATCH")
    if manifest.get("stale_chat_fragment_rate", 0.0) <= 0.10:
        failures.append("UPSTREAM_138I_STALE_RATE_NOT_ABOVE_GATE")
    if manifest.get("helper_eval_integrity_passed") is not True or manifest.get("no_expected_or_scorer_metadata_reached_helper_requests") is not True:
        failures.append("RAW_HELPER_INTEGRITY_FAILURE")

    config = load_json(SMOKE_ROOT / "analysis_config.json")
    if config.get("artifact_only_analysis") is not True or config.get("new_model_inference_run") is not False:
        failures.append("ARTIFACT_ONLY_BOUNDARY_FAILURE")
    if config.get("shared_helper_called") is not False or config.get("torch_forward_pass_run") is not False:
        failures.append("ARTIFACT_ONLY_BOUNDARY_FAILURE")

    stale = load_json(SMOKE_ROOT / "stale_chat_distribution_report.json")
    value = load_json(SMOKE_ROOT / "value_grounding_failure_report.json")
    prefix = load_json(SMOKE_ROOT / "prefix_vs_value_decoupling_report.json")
    source = load_json(SMOKE_ROOT / "source_prior_vs_training_objective_report.json")
    taxonomy = load_json(SMOKE_ROOT / "output_pattern_taxonomy.json")
    coupling = load_json(SMOKE_ROOT / "stale_chat_value_coupling_report.json")
    gaps = load_json(SMOKE_ROOT / "diagnostic_gap_register.json")
    recommendation = load_json(SMOKE_ROOT / "next_repair_recommendation.json")
    decision = load_json(SMOKE_ROOT / "decision.json")

    if stale.get("row_count") != 960 or stale.get("stale_chat_fragment_count") != 209:
        failures.append("STALE_DISTRIBUTION_MISMATCH")
    if abs(stale.get("stale_chat_fragment_rate", 0.0) - 0.21770833333333334) > 1e-12:
        failures.append("STALE_DISTRIBUTION_MISMATCH")
    if value.get("answer_prefix_accuracy") != 1.0 or value.get("eval_namespace_emission_accuracy") != 1.0:
        failures.append("VALUE_GROUNDING_PROFILE_MISMATCH")
    if value.get("answer_value_accuracy") != 0.0 or value.get("exact_answer_accuracy") != 0.0:
        failures.append("VALUE_GROUNDING_PROFILE_MISMATCH")
    if value.get("rows_with_no_stale_fragment_but_wrong_value", 0) <= 0:
        failures.append("VALUE_FAILURE_WITHOUT_STALE_NOT_PROVEN")
    if prefix.get("prefix_success_value_failure_rate") != 1.0 or prefix.get("wrapper_prefix_learned_without_value_grounding") is not True:
        failures.append("PREFIX_VALUE_DECOUPLING_NOT_PROVEN")
    if taxonomy.get("failed_row_count") != 960 or taxonomy.get("exactly_one_primary_label_per_failed_row") is not True:
        failures.append("OUTPUT_TAXONOMY_INVALID")
    labels = taxonomy.get("primary_failure_label_counts", {})
    if labels.get("stale_chat_with_wrong_value") != 209 or labels.get("eval_namespace_but_wrong_value") != 751:
        failures.append("OUTPUT_TAXONOMY_MISMATCH")
    if coupling.get("P_wrong_value_given_stale_chat") != 1.0 or coupling.get("P_wrong_value_given_no_stale_chat") != 1.0:
        failures.append("COUPLING_REPORT_MISMATCH")
    if coupling.get("value_failure_occurs_without_stale_chat") is not True:
        failures.append("COUPLING_REPORT_MISMATCH")
    if source.get("train_expected_outputs_with_user_or_assistant") != 0 or source.get("eval_expected_outputs_with_user_or_assistant") != 0:
        failures.append("SOURCE_PRIOR_REPORT_MISMATCH")
    if gaps.get("gap_count", 0) < 2:
        failures.append("DIAGNOSTIC_GAP_REGISTER_INCOMPLETE")
    if recommendation.get("recommended_next") != "138V_ANSWER_VALUE_GROUNDING_OBJECTIVE_REDESIGN_PLAN":
        failures.append("NEXT_RECOMMENDATION_MISMATCH")
    if recommendation.get("value_failure_occurs_without_stale_chat") is not True:
        failures.append("NEXT_RECOMMENDATION_MISMATCH")

    if decision.get("decision") != "stale_chat_rollout_failure_analysis_complete":
        failures.append("DECISION_MISMATCH")
    if decision.get("next") != recommendation.get("recommended_next"):
        failures.append("DECISION_MISMATCH")
    if decision.get("analysis_artifact_only") is not True or decision.get("new_model_inference_run") is not False:
        failures.append("ARTIFACT_ONLY_BOUNDARY_FAILURE")
    require_false_flags(decision, failures)

    summary = load_json(SMOKE_ROOT / "summary.json")
    report = (SMOKE_ROOT / "report.md").read_text(encoding="utf-8")
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
    failures.extend(ast_scan_source([REPO_ROOT / RUNNER, REPO_ROOT / CHECKER]))
    failures.extend(check_artifacts())
    if failures:
        print("138S checker failed:")
        for failure in sorted(set(failures)):
            print(f" - {failure}")
        return 1
    print("138S checker passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
