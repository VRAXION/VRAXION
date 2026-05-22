#!/usr/bin/env python3
"""Static/artifact checker for STABLE_LOOP_PHASE_LOCK_118."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_118_reasoning_first_raw_assistant_repair/smoke"
REQUIRED_DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_118_REASONING_FIRST_RAW_ASSISTANT_REPAIR_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_118_REASONING_FIRST_RAW_ASSISTANT_REPAIR_RESULT.md",
]
REQUIRED_SOURCE = [
    "scripts/probes/run_stable_loop_phase_lock_118_reasoning_first_raw_assistant_repair.py",
    "scripts/probes/run_stable_loop_phase_lock_118_reasoning_first_raw_assistant_repair_check.py",
]
ALLOWED_MUTATIONS = set(REQUIRED_DOCS + REQUIRED_SOURCE)
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "repair_config.json",
    "upstream_117_manifest.json",
    "upstream_116_manifest.json",
    "upstream_115_manifest.json",
    "upstream_112_manifest.json",
    "upstream_099_manifest.json",
    "train_dataset_manifest.json",
    "eval_dataset_manifest.json",
    "freshness_leakage_audit_start.json",
    "freshness_leakage_audit.json",
    "eval_dataset.jsonl",
    "train_examples_sample.jsonl",
    "training_metrics.jsonl",
    "rollout_metrics.jsonl",
    "arm_training_metrics.jsonl",
    "checkpoint_integrity_manifest.json",
    "target_118_checkpoint_manifest.json",
    "bounded_release_integrity_manifest.json",
    "raw_generation_results_pre.jsonl",
    "raw_generation_results_post.jsonl",
    "control_results.jsonl",
    "control_reasoning_results.jsonl",
    "eval_row_hashes.json",
    "arm_comparison.json",
    "reasoning_metrics.json",
    "retention_report.json",
    "collapse_metrics.json",
    "namespace_audit.json",
    "overclaim_exfiltration_report.json",
    "fineweb_retention_report.json",
    "aggregate_metrics.json",
    "human_readable_samples.jsonl",
    "failure_case_samples.jsonl",
    "decision.json",
    "summary.json",
    "report.md",
]
REQUIRED_TERMS = [
    "STABLE_LOOP_PHASE_LOCK_118_REASONING_FIRST_RAW_ASSISTANT_REPAIR",
    "targeted research repair",
    "raw-only final evaluation",
    "REASONING_FIRST_RAW_ASSISTANT_REPAIR_POSITIVE",
    "RAW_REASONING_BREAKPOINT_IMPROVED",
    "RAW_ROLLOUT_METRICS_IMPROVED",
    "SCHEDULED_SAMPLING_OR_ROLLOUT_USED",
    "RETENTION_PRESERVED",
    "NAMESPACE_MEMORIZATION_REJECTED",
    "119_REASONING_REPAIR_SCALE_CONFIRM",
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
MAIN_ARM = "POST_118_REASONING_REPAIRED_RAW"
PRE_ARM = "PRE_118_RAW_BASELINE"
ARMS = {
    MAIN_ARM,
    PRE_ARM,
    "NO_ROLLOUT_OBJECTIVE_CONTROL",
    "GENERAL_SFT_ONLY_CONTROL",
    "STATIC_OUTPUT_CONTROL",
    "COPY_PROMPT_CONTROL",
    "RANDOM_REASONING_CONTROL",
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
    if "REASONING_FIRST_RAW_ASSISTANT_REPAIR_POSITIVE" not in verdicts:
        failures.append("REASONING_FIRST_RAW_ASSISTANT_REPAIR_RESULT_MISSING")
    for key, expected in [
        ("targeted_research_repair", True),
        ("general_training", False),
        ("runtime_surface_mutated", False),
        ("bounded_release_stack_mutated", False),
        ("existing_checkpoint_mutated", False),
        ("service_started", False),
        ("deployment_smoke_run", False),
        ("gpt_like_assistant_readiness_claimed", False),
        ("open_domain_assistant_readiness_claimed", False),
        ("production_chat_claimed", False),
        ("public_api_claimed", False),
        ("deployment_readiness_claimed", False),
        ("safety_alignment_claimed", False),
    ]:
        if summary.get(key) != expected:
            failures.append("GPT_LIKE_READINESS_FALSE_CLAIM" if "claimed" in key else "RUNTIME_SURFACE_MUTATION_DETECTED")

    config = load_json(SMOKE_ROOT / "repair_config.json")
    expected_config = {
        "seeds": [2121, 2122, 2123],
        "steps": 12000,
        "batch_size": 64,
        "seq_len": 256,
        "train_examples": 120000,
        "eval_rows_per_family": 64,
        "fineweb_replay_tokens": 1000000,
        "rollout_eval_every": 50,
    }
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            failures.append("FULL_CONFIGURED_RUN_NOT_USED")
    if config.get("full_configured_run_used") is not True or config.get("positive_scored_arm") != MAIN_ARM or set(config.get("arms", [])) != ARMS:
        failures.append("FULL_CONFIGURED_RUN_NOT_USED")
    for key in [
        "llm_judge_used",
        "subjective_scoring_used",
        "current_world_fact_scoring_used",
        "integrated_policy_used_during_final_eval",
        "decoder_reference_used_during_final_eval",
        "oracle_rerank_used",
        "expected_answer_metadata_used",
        "teacher_forcing_used_during_final_eval",
        "verifier_rerank_used",
    ]:
        if config.get(key) is not False:
            failures.append("ORACLE_SHORTCUT_DETECTED")

    eval_rows = read_jsonl(SMOKE_ROOT / "eval_dataset.jsonl")
    if len(eval_rows) != 2112:
        failures.append("FULL_CONFIGURED_RUN_NOT_USED")
    progress_events = {row.get("event") for row in read_jsonl(SMOKE_ROOT / "progress.jsonl")}
    for event in ["start", "upstream_verification", "dataset_build", "freshness_leakage_audit_start", "freshness_leakage_audit", "training_start", "training_heartbeat", "raw_final_eval", "decision_writing", "final_verdict"]:
        if event not in progress_events:
            failures.append(f"PROGRESS_EVENT_MISSING:{event}")

    leakage = load_json(SMOKE_ROOT / "freshness_leakage_audit.json")
    if leakage.get("leakage_detected") or leakage.get("exact_prompt_overlap") != 0 or leakage.get("exact_expected_output_overlap") != 0 or leakage.get("near_duplicate_prompt_count") != 0:
        failures.append("TRAIN_EVAL_LEAKAGE_DETECTED")
    row_hashes = load_json(SMOKE_ROOT / "eval_row_hashes.json")
    if len({payload.get("eval_row_hash") for payload in row_hashes.get("arms", {}).values()}) != 1:
        failures.append("BASELINE_EVAL_MISMATCH")

    train_manifest = load_json(SMOKE_ROOT / "train_dataset_manifest.json")
    if train_manifest.get("train_examples") != 120000 or train_manifest.get("train_eval_namespace_disjoint") is not True:
        failures.append("TRAINING_HELPER_MISSING")
    if train_manifest.get("training_helper_safe") is not True or train_manifest.get("runner_local_training_helper") != "phase_118_runner_local_target_only_repair_harness":
        failures.append("TRAINING_HELPER_MISSING")
    mix = train_manifest.get("training_mix", {})
    if abs(sum(float(value) for value in mix.values()) - 1.0) > 0.0001:
        failures.append("TRAINING_MIX_INVALID")

    training_reports = {row["arm"]: row for row in read_jsonl(SMOKE_ROOT / "arm_training_metrics.jsonl")}
    main_training = training_reports.get(MAIN_ARM, {})
    if main_training.get("train_step_count") != 12000 or main_training.get("optimizer_step_count") != 12000:
        failures.append("TRAINING_SIDE_EFFECT_DETECTED")
    if main_training.get("target_118_checkpoint_changed") is not True:
        failures.append("TARGET_CHECKPOINT_NOT_CHANGED")
    if not (main_training.get("train_loss_final", 1.0) < main_training.get("train_loss_initial", 0.0)):
        failures.append("TRAIN_LOSS_NOT_IMPROVED")
    if main_training.get("scheduled_sampling_batch_count", 0) <= 0 and main_training.get("rollout_loss_batch_count", 0) <= 0:
        failures.append("RAW_OBJECTIVE_REDESIGN_INCOMPLETE")

    checkpoint = load_json(SMOKE_ROOT / "checkpoint_integrity_manifest.json")
    if checkpoint.get("source_100_checkpoint_unchanged") is not True or checkpoint.get("source_102_checkpoint_unchanged") is not True or checkpoint.get("existing_checkpoint_mutated") is not False:
        failures.append("CHECKPOINT_MUTATION_DETECTED")
    bounded = load_json(SMOKE_ROOT / "bounded_release_integrity_manifest.json")
    if bounded.get("bounded_release_artifact_unchanged") is not True:
        failures.append("BOUNDED_RELEASE_MUTATION_DETECTED")

    reasoning = load_json(SMOKE_ROOT / "reasoning_metrics.json")
    if reasoning.get("post_tier4_reasoning_accuracy", 0.0) < 0.995:
        failures.append("REASONING_REPAIR_PARTIAL")
    if reasoning.get("post_tier4_reasoning_accuracy", 0.0) < reasoning.get("pre_tier4_reasoning_accuracy", 1.0) + 0.005:
        failures.append("RAW_ROLLOUT_METRICS_NOT_IMPROVED")
    if reasoning.get("post_tier8_reasoning_combo_accuracy", 0.0) < 0.93:
        failures.append("REASONING_REPAIR_PARTIAL")
    if reasoning.get("post_tier8_reasoning_combo_accuracy", 0.0) < reasoning.get("pre_tier8_reasoning_combo_accuracy", 1.0) + 0.05:
        failures.append("RAW_ROLLOUT_METRICS_NOT_IMPROVED")
    if reasoning.get("reasoning_failure_count_post_ratio", 1.0) > 0.25:
        failures.append("RAW_REASONING_BREAKPOINT_NOT_IMPROVED")
    for key, threshold in [
        ("table_rule_reasoning_accuracy", 0.95),
        ("small_arithmetic_accuracy", 0.95),
        ("rule_chaining_accuracy", 0.95),
        ("contradiction_resolution_accuracy", 0.90),
    ]:
        if reasoning.get(key, 0.0) < threshold:
            failures.append("REASONING_FAMILY_GATE_FAILED")
    if reasoning.get("no_rollout_objective_raw_rollout_improved") is not False:
        failures.append("TEACHER_FORCING_ONLY_SUCCESS_DETECTED")

    retention = load_json(SMOKE_ROOT / "retention_report.json")
    if retention.get("retention_preserved") is not True:
        failures.append("RETENTION_REGRESSION_DETECTED")
    collapse = load_json(SMOKE_ROOT / "collapse_metrics.json")
    if collapse.get("collapse_rejected") is not True:
        failures.append("COLLAPSE_DETECTED")
    namespace = load_json(SMOKE_ROOT / "namespace_audit.json")
    if namespace.get("namespace_memorization_detected") is not False:
        failures.append("NAMESPACE_MEMORIZATION_DETECTED")
    overclaim = load_json(SMOKE_ROOT / "overclaim_exfiltration_report.json")
    if overclaim.get("overclaim_or_exfiltration_detected") is not False:
        failures.append("OVERCLAIM_DETECTED")

    comparison = load_json(SMOKE_ROOT / "arm_comparison.json")
    if comparison.get("positive_scored_arm") != MAIN_ARM or comparison.get("helper_or_decoder_metrics_merged") is not False:
        failures.append("RAW_HELPER_METRICS_MERGED")
    if comparison.get("controls_failed") is not True:
        failures.append("CONTROL_UNEXPECTED_PASS")

    aggregate = load_json(SMOKE_ROOT / "aggregate_metrics.json")
    for key, expected in [
        ("full_configured_run_used", True),
        ("target_118_checkpoint_changed", True),
        ("source_100_checkpoint_unchanged", True),
        ("source_102_checkpoint_unchanged", True),
        ("bounded_release_artifact_unchanged", True),
        ("packaged_winner_hash_unchanged", True),
        ("raw_rollout_reasoning_metrics_improved", True),
        ("retention_preserved", True),
        ("collapse_rejected", True),
        ("controls_failed", True),
        ("benchmark_leakage_detected", False),
        ("integrated_policy_used_during_final_eval", False),
        ("decoder_reference_used_during_final_eval", False),
        ("expected_answer_metadata_used", False),
        ("teacher_forcing_used_during_final_eval", False),
        ("oracle_rerank_used", False),
        ("llm_judge_used", False),
        ("verifier_rerank_used", False),
    ]:
        if aggregate.get(key) != expected:
            failures.append("ORACLE_SHORTCUT_DETECTED" if "eval" in key or "judge" in key or "rerank" in key else "AGGREGATE_GATE_FAILED")
    if aggregate.get("train_step_count") != 12000 or aggregate.get("optimizer_step_count") != 12000:
        failures.append("TRAINING_SIDE_EFFECT_DETECTED")
    for key in ["namespace_leak_rate", "teacher_namespace_copy_rate"]:
        if aggregate.get(key, 1.0) > 0.03:
            failures.append("NAMESPACE_MEMORIZATION_DETECTED")
    if aggregate.get("case_id_drift_rate", 1.0) > 0.05:
        failures.append("NAMESPACE_MEMORIZATION_DETECTED")
    for key in ["artifact_exfiltration_count", "gpt_like_claim_count", "production_chat_claim_count", "public_api_claim_count", "deployment_readiness_claim_count", "safety_alignment_claim_count"]:
        if aggregate.get(key) != 0:
            failures.append("OVERCLAIM_DETECTED")

    decision = load_json(SMOKE_ROOT / "decision.json")
    if decision.get("decision") != "reasoning_first_raw_assistant_repair_positive" or decision.get("next") != "119_REASONING_REPAIR_SCALE_CONFIRM":
        failures.append("DECISION_NEXT_MISMATCH")
    if decision.get("raw_only_final_eval") is not True:
        failures.append("ORACLE_SHORTCUT_DETECTED")

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
        print("118 checker failed:")
        for failure in sorted(set(failures)):
            print(f" - {failure}")
        return 1
    print("118 checker passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
