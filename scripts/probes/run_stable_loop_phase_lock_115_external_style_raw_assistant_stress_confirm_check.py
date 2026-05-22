#!/usr/bin/env python3
"""Static/artifact checker for STABLE_LOOP_PHASE_LOCK_115."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_115_external_style_raw_assistant_stress_confirm/smoke"
REQUIRED_DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_115_EXTERNAL_STYLE_RAW_ASSISTANT_STRESS_CONFIRM_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_115_EXTERNAL_STYLE_RAW_ASSISTANT_STRESS_CONFIRM_RESULT.md",
]
REQUIRED_SOURCE = [
    "scripts/probes/run_stable_loop_phase_lock_115_external_style_raw_assistant_stress_confirm.py",
    "scripts/probes/run_stable_loop_phase_lock_115_external_style_raw_assistant_stress_confirm_check.py",
]
ALLOWED_MUTATIONS = set(REQUIRED_DOCS + REQUIRED_SOURCE)
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "stress_config.json",
    "upstream_114_manifest.json",
    "upstream_113_manifest.json",
    "upstream_112_manifest.json",
    "upstream_111x_manifest.json",
    "upstream_099_manifest.json",
    "checkpoint_integrity_manifest.json",
    "bounded_release_integrity_manifest.json",
    "external_stress_dataset.jsonl",
    "eval_row_hashes.json",
    "freshness_leakage_audit.json",
    "raw_generation_results.jsonl",
    "diagnostic_helper_results.jsonl",
    "control_results.jsonl",
    "per_family_metrics.json",
    "per_seed_metrics.jsonl",
    "aggregate_metrics.json",
    "namespace_audit.json",
    "retention_report.json",
    "collapse_metrics.json",
    "overclaim_exfiltration_report.json",
    "control_arm_report.json",
    "human_readable_samples.jsonl",
    "failure_case_samples.jsonl",
    "decision.json",
    "summary.json",
    "report.md",
]
REQUIRED_TERMS = [
    "STABLE_LOOP_PHASE_LOCK_115_EXTERNAL_STYLE_RAW_ASSISTANT_STRESS_CONFIRM",
    "eval-only",
    "deterministic rubric-bounded",
    "EXTERNAL_STYLE_RAW_ASSISTANT_STRESS_CONFIRM_POSITIVE",
    "UPSTREAM_114_BRIDGE_VERIFIED",
    "RAW_EXTERNAL_STRESS_CONFIRMED",
    "CONTROLS_FAILED",
    "LEAKAGE_REJECTED",
    "NAMESPACE_MEMORIZATION_REJECTED",
    "116_RAW_ASSISTANT_CAPABILITY_CEILING_AND_GAP_MAP",
    "not GPT-like assistant readiness",
    "not open-domain assistant readiness",
    "not production chat",
    "not public API",
    "not deployment readiness",
    "not safety alignment",
]
NEGATION_MARKERS = ["not ", "no ", "does not ", "do not ", "without ", "false", "forbidden", "blocked", "denial"]
FORBIDDEN_CLAIMS = {
    "GPT_LIKE_READINESS_FALSE_CLAIM": ["GPT-like ready", "GPT-like assistant ready"],
    "OPEN_DOMAIN_ASSISTANT_CLAIM_DETECTED": ["open-domain assistant ready"],
    "PRODUCTION_CHAT_CLAIM_DETECTED": ["production chat ready", "production ready"],
    "PUBLIC_API_CLAIM_DETECTED": ["public API ready"],
    "DEPLOYMENT_CLAIM_DETECTED": ["deployment ready"],
    "SAFETY_ALIGNMENT_CLAIM_DETECTED": ["safety aligned"],
}
MAIN_ARM = "POST_112_RAW_CURRENT_CHASSIS_EXTERNAL_STRESS"
ARMS = {
    MAIN_ARM,
    "CURRENT_RAW_BASELINE",
    "INTEGRATED_DECODER_POLICY_REFERENCE_DIAGNOSTIC",
    "STATIC_OUTPUT_CONTROL",
    "COPY_PROMPT_CONTROL",
    "RANDOM_SLOT_CONTROL",
    "RANDOM_FACT_CONTROL",
}
CONTROL_ARMS = {"STATIC_OUTPUT_CONTROL", "COPY_PROMPT_CONTROL", "RANDOM_SLOT_CONTROL", "RANDOM_FACT_CONTROL"}


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
    if "EXTERNAL_STYLE_RAW_ASSISTANT_STRESS_CONFIRM_POSITIVE" not in verdicts:
        failures.append("EXTERNAL_STYLE_RAW_ASSISTANT_STRESS_CONFIRM_RESULT_MISSING")
    for key, expected in [
        ("eval_only", True),
        ("deterministic_rubric_bounded_scoring", True),
        ("training_performed", False),
        ("checkpoint_mutated", False),
        ("service_started", False),
        ("deployment_smoke_run", False),
        ("runtime_surface_mutated", False),
        ("bounded_release_stack_mutated", False),
        ("gpt_like_assistant_readiness_claimed", False),
        ("open_domain_assistant_readiness_claimed", False),
        ("production_chat_claimed", False),
        ("public_api_claimed", False),
        ("deployment_readiness_claimed", False),
        ("safety_alignment_claimed", False),
    ]:
        if summary.get(key) is not expected:
            failures.append("GPT_LIKE_READINESS_FALSE_CLAIM" if "claimed" in key else "RUNTIME_SURFACE_MUTATION_DETECTED")
    for key, expected in [
        ("train_step_count", 0),
        ("optimizer_step_count", 0),
        ("checkpoint_mutated", False),
        ("service_started", False),
        ("deployment_smoke_run", False),
        ("helper_metrics_used_for_positive_score", False),
        ("raw_helper_metrics_merged", False),
        ("llm_judge_used", False),
        ("subjective_scoring_used", False),
        ("current_world_fact_scoring_used", False),
    ]:
        if metrics.get(key) != expected:
            failures.append("TRAINING_SIDE_EFFECT_DETECTED" if "step" in key else "ORACLE_SHORTCUT_DETECTED")

    config = load_json(SMOKE_ROOT / "stress_config.json")
    if config.get("seeds") != [2101, 2102, 2103, 2104, 2105] or config.get("rows_per_family") != 96:
        failures.append("FULL_CONFIGURED_RUN_NOT_USED")
    for key, expected in [("long_context_chars", 8192), ("noise_blocks", 16), ("format_variants", 8), ("table_rows", 32), ("multi_doc_count", 5)]:
        if config.get(key) != expected:
            failures.append("FULL_CONFIGURED_RUN_NOT_USED")
    if config.get("main_positive_arm") != MAIN_ARM or set(config.get("arms", [])) != ARMS:
        failures.append("HELPER_PATH_USED_FOR_POSITIVE_SCORE")
    if config.get("current_world_fact_scoring_used") is not False:
        failures.append("UNBOUNDED_WORLD_KNOWLEDGE_SCORING_DETECTED")

    leakage = load_json(SMOKE_ROOT / "freshness_leakage_audit.json")
    if leakage.get("leakage_detected") or leakage.get("exact_prompt_overlap") != 0 or leakage.get("exact_expected_output_overlap") != 0 or leakage.get("near_duplicate_prompt_count") != 0:
        failures.append("BENCHMARK_LEAKAGE_DETECTED")

    row_hashes = load_json(SMOKE_ROOT / "eval_row_hashes.json")
    if len({payload.get("eval_row_hash") for payload in row_hashes.get("arms", {}).values()}) != 1:
        failures.append("BASELINE_EVAL_MISMATCH")

    aggregate = load_json(SMOKE_ROOT / "aggregate_metrics.json")
    if aggregate.get("all_seeds_passed_independently") is not True:
        failures.append("MULTI_SEED_EXTERNAL_STYLE_INSTABILITY_DETECTED")
    if aggregate.get("min_external_style_raw_accuracy", 0.0) < 0.90 or aggregate.get("mean_external_style_raw_accuracy", 0.0) < 0.93:
        failures.append("EXTERNAL_STYLE_RAW_GATE_FAILED")
    if aggregate.get("min_per_family_accuracy", 0.0) < 0.80:
        failures.append("FAMILY_SPECIFIC_EXTERNAL_STYLE_REGRESSION_DETECTED")
    if aggregate.get("controls_failed") is not True:
        failures.append("CONTROL_UNEXPECTED_PASS")
    if aggregate.get("benchmark_leakage_detected") is not False:
        failures.append("BENCHMARK_LEAKAGE_DETECTED")
    for key, limit in [("max_namespace_leak_rate", 0.03), ("max_teacher_namespace_copy_rate", 0.03), ("max_case_id_drift_rate", 0.05)]:
        if aggregate.get(key, 1.0) > limit:
            failures.append("NAMESPACE_MEMORIZATION_DETECTED")

    comparison = load_json(SMOKE_ROOT / "arm_comparison.json")
    arms = {row["arm"]: row for row in comparison.get("arms", [])}
    if set(arms) != ARMS:
        failures.append("BASELINE_EVAL_MISMATCH")
    if comparison.get("helper_metrics_used_for_positive_score") is not False or comparison.get("raw_helper_metrics_merged") is not False:
        failures.append("RAW_HELPER_METRICS_MERGED")
    for arm in CONTROL_ARMS:
        if arms.get(arm, {}).get("metrics", {}).get("external_style_raw_accuracy", 1.0) >= 0.70:
            failures.append("TASK_TOO_EASY_OR_SCORER_WEAK")

    seed_metrics = read_jsonl(SMOKE_ROOT / "per_seed_metrics.jsonl")
    if len(seed_metrics) != 5 or not all(seed.get("seed_passed") is True for seed in seed_metrics):
        failures.append("MULTI_SEED_EXTERNAL_STYLE_INSTABILITY_DETECTED")

    decision = load_json(SMOKE_ROOT / "decision.json")
    if decision.get("next") != "116_RAW_ASSISTANT_CAPABILITY_CEILING_AND_GAP_MAP":
        failures.append("DECISION_NEXT_MISMATCH")
    if decision.get("helper_metrics_used_for_positive_score") is not False or decision.get("raw_helper_metrics_merged") is not False:
        failures.append("HELPER_PATH_USED_FOR_POSITIVE_SCORE")

    text_to_scan = "\n".join([(SMOKE_ROOT / "report.md").read_text(encoding="utf-8"), json.dumps(summary, sort_keys=True)])
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
        print("115 checker failed:")
        for failure in sorted(set(failures)):
            print(f" - {failure}")
        return 1
    print("115 checker passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
