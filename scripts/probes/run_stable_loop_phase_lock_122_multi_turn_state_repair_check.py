#!/usr/bin/env python3
"""Static/artifact checker for STABLE_LOOP_PHASE_LOCK_122."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_122_multi_turn_state_repair/smoke"
REQUIRED_DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_122_MULTI_TURN_STATE_REPAIR_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_122_MULTI_TURN_STATE_REPAIR_RESULT.md",
]
REQUIRED_SOURCE = [
    "scripts/probes/run_stable_loop_phase_lock_122_multi_turn_state_repair.py",
    "scripts/probes/run_stable_loop_phase_lock_122_multi_turn_state_repair_check.py",
]
ALLOWED_MUTATIONS = set(REQUIRED_DOCS + REQUIRED_SOURCE)
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "repair_config.json",
    "upstream_121_manifest.json",
    "upstream_120_manifest.json",
    "upstream_119_manifest.json",
    "upstream_118_manifest.json",
    "upstream_112_manifest.json",
    "upstream_099_manifest.json",
    "checkpoint_integrity_manifest.json",
    "bounded_release_integrity_manifest.json",
    "train_dataset_manifest.json",
    "eval_dataset_manifest.json",
    "state_repair_dataset.jsonl",
    "eval_row_hashes.json",
    "freshness_leakage_audit.json",
    "namespace_audit.json",
    "arm_training_metrics.jsonl",
    "rollout_eval_metrics.jsonl",
    "raw_generation_results.jsonl",
    "control_results.jsonl",
    "per_family_metrics.json",
    "depth_metrics.json",
    "state_repair_metrics.json",
    "reasoning_preservation_report.json",
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
    "STABLE_LOOP_PHASE_LOCK_122_MULTI_TURN_STATE_REPAIR",
    "targeted research repair",
    "raw-only final evaluation",
    "MULTI_TURN_STATE_REPAIR_POSITIVE",
    "MULTI_TURN_STATE_BREAKPOINT_IMPROVED",
    "RAW_STATE_ROLLOUT_IMPROVED",
    "DEPTH_8_STATE_TRACKING_PASSES",
    "REASONING_REPAIR_PRESERVED",
    "STALE_STATE_MEMORIZATION_REJECTED",
    "123_MULTI_TURN_STATE_REPAIR_SCALE_CONFIRM",
    "not GPT-like assistant readiness",
    "not open-domain assistant readiness",
    "not production chat",
    "not public API",
    "not deployment readiness",
    "not safety alignment",
    "not Hungarian assistant readiness",
]
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
MAIN_ARM = "POST_122_MULTI_TURN_STATE_REPAIRED_RAW"
PRE_ARM = "PRE_122_POST_REASONING_RAW_BASELINE"
ARMS = {
    MAIN_ARM,
    PRE_ARM,
    "NO_ROLLOUT_OBJECTIVE_CONTROL",
    "GENERAL_SFT_ONLY_CONTROL",
    "STATIC_OUTPUT_CONTROL",
    "COPY_PROMPT_CONTROL",
    "RANDOM_STATE_CONTROL",
    "STALE_STATE_COPY_CONTROL",
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
    if "MULTI_TURN_STATE_REPAIR_POSITIVE" not in verdicts:
        failures.append("MULTI_TURN_STATE_REPAIR_RESULT_MISSING")
    for key, expected in [
        ("targeted_research_repair", True),
        ("generic_sft", False),
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
        ("hungarian_assistant_readiness_claimed", False),
    ]:
        if summary.get(key) != expected:
            failures.append("GPT_LIKE_READINESS_FALSE_CLAIM" if "claimed" in key else "RUNTIME_SURFACE_MUTATION_DETECTED")

    config = load_json(SMOKE_ROOT / "repair_config.json")
    expected_config = {
        "seeds": [2151, 2152, 2153],
        "steps": 12000,
        "batch_size": 64,
        "seq_len": 256,
        "train_examples": 120000,
        "eval_rows_per_family": 64,
        "fineweb_replay_tokens": 1000000,
        "rollout_eval_every": 50,
        "multi_turn_depths": [2, 4, 6, 8],
        "state_update_variants": 8,
        "stale_decoy_count": 6,
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
        "expected_answer_used_during_eval",
        "teacher_forcing_used_during_final_eval",
        "verifier_rerank_used",
    ]:
        if config.get(key) is not False:
            failures.append("ORACLE_SHORTCUT_DETECTED")

    eval_rows = read_jsonl(SMOKE_ROOT / "state_repair_dataset.jsonl")
    if len(eval_rows) != 3264:
        failures.append("FULL_CONFIGURED_RUN_NOT_USED")
    progress_events = {row.get("event") for row in read_jsonl(SMOKE_ROOT / "progress.jsonl")}
    for event in ["startup", "upstream_verification", "dataset_build", "leakage_audit", "seed_train_start", "training_heartbeat", "rollout_eval_heartbeat", "seed_final_eval", "aggregate_analysis", "decision_writing", "final_verdict"]:
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
    if train_manifest.get("training_helper_safe") is not True or train_manifest.get("runner_local_training_helper") != "phase_122_runner_local_target_only_state_repair_harness":
        failures.append("TRAINING_HELPER_MISSING")
    mix = train_manifest.get("training_mix", {})
    if abs(sum(float(value) for value in mix.values()) - 1.0) > 0.0001:
        failures.append("TRAINING_MIX_INVALID")

    training_reports = {row["arm"]: row for row in read_jsonl(SMOKE_ROOT / "arm_training_metrics.jsonl")}
    main_training = training_reports.get(MAIN_ARM, {})
    if main_training.get("train_step_count") != 12000 or main_training.get("optimizer_step_count") != 12000:
        failures.append("TRAINING_SIDE_EFFECT_DETECTED")
    if main_training.get("target_122_checkpoint_changed") is not True:
        failures.append("TARGET_CHECKPOINT_NOT_CHANGED")
    if not (main_training.get("train_loss_final", 1.0) < main_training.get("train_loss_initial", 0.0)):
        failures.append("TRAIN_LOSS_NOT_IMPROVED")
    if main_training.get("scheduled_sampling_batch_count", 0) <= 0 and main_training.get("rollout_loss_batch_count", 0) <= 0:
        failures.append("RAW_OBJECTIVE_REDESIGN_INCOMPLETE")

    checkpoint = load_json(SMOKE_ROOT / "checkpoint_integrity_manifest.json")
    for key in ["source_100_checkpoint_unchanged", "source_102_checkpoint_unchanged", "target_122_checkpoint_changed", "packaged_winner_hash_unchanged"]:
        if checkpoint.get(key) is not True:
            failures.append("CHECKPOINT_MUTATION_DETECTED")
    bounded = load_json(SMOKE_ROOT / "bounded_release_integrity_manifest.json")
    if bounded.get("bounded_release_artifact_unchanged") is not True:
        failures.append("BOUNDED_RELEASE_MUTATION_DETECTED")

    state = load_json(SMOKE_ROOT / "state_repair_metrics.json")
    if state.get("decision") != "multi_turn_state_repair_success" or state.get("next") != "123_MULTI_TURN_STATE_REPAIR_SCALE_CONFIRM":
        failures.append("DECISION_MISSING")
    gate_thresholds = {
        "post_multi_turn_state_accuracy": 0.95,
        "raw_state_accuracy_improvement": 0.10,
        "post_state_tracking_accuracy": 0.95,
        "post_multi_turn_correction_accuracy": 0.95,
        "stale_state_rejection_accuracy": 0.95,
        "override_chain_accuracy": 0.92,
        "active_slot_after_update_accuracy": 0.95,
        "tier4_multi_turn_breakpoint_accuracy": 0.95,
        "tier7_state_combo_accuracy": 0.88,
        "tier8_combined_state_accuracy": 0.88,
        "depth_8_state_accuracy": 0.88,
        "tier4_reasoning_accuracy": 0.97,
        "tier8_reasoning_combo_accuracy": 0.90,
        "bounded_chat_slot_binding_accuracy": 0.90,
        "finite_label_anchorroute_retention_accuracy": 0.90,
        "unsupported_refusal_retention_accuracy": 0.80,
    }
    for key, threshold in gate_thresholds.items():
        if float(state.get(key, 0.0)) < threshold:
            failures.append("MULTI_TURN_STATE_REPAIR_GATE_FAILED" if "reasoning" not in key else "REASONING_REGRESSION_DETECTED")
    if state.get("pre_multi_turn_state_accuracy", 1.0) >= 0.85:
        failures.append("BASELINE_GAP_NOT_REPRODUCED")
    if state.get("reasoning_failure_rate", 1.0) > 0.05:
        failures.append("REASONING_REGRESSION_DETECTED")
    if state.get("multi_turn_state_failure_count_post", 999999) > state.get("multi_turn_state_failure_count_pre", 0) * 0.25:
        failures.append("MULTI_TURN_STATE_REPAIR_GATE_FAILED")
    for key, threshold in [
        ("namespace_leak_rate", 0.03),
        ("teacher_namespace_copy_rate", 0.03),
        ("case_id_drift_rate", 0.05),
        ("stale_state_copy_rate", 0.05),
        ("stale_decoy_leak_rate", 0.05),
    ]:
        if float(state.get(key, 1.0)) > threshold:
            failures.append("STALE_STATE_MEMORIZATION_DETECTED" if key.startswith("stale") else "NAMESPACE_MEMORIZATION_DETECTED")
    for key, threshold in [("empty_output_rate", 0.02), ("static_output_rate", 0.10), ("repetition_rate", 0.20), ("copy_prompt_rate", 0.15)]:
        if float(state.get(key, 1.0)) > threshold:
            failures.append("STATIC_RESPONSE_COLLAPSE_DETECTED")
    if state.get("artifact_exfiltration_count") != 0 or state.get("overclaim_count") != 0:
        failures.append("OVERCLAIM_DETECTED")
    if state.get("controls_failed") is not True:
        failures.append("CONTROL_UNEXPECTED_PASS")

    depth = load_json(SMOKE_ROOT / "depth_metrics.json")
    for key in ["depth_2_state_accuracy", "depth_4_state_accuracy", "depth_6_state_accuracy", "depth_8_state_accuracy"]:
        if key not in depth:
            failures.append("DEPTH_METRICS_MISSING")
    reasoning = load_json(SMOKE_ROOT / "reasoning_preservation_report.json")
    if reasoning.get("reasoning_repair_preserved") is not True:
        failures.append("REASONING_REGRESSION_DETECTED")
    control = load_json(SMOKE_ROOT / "control_arm_report.json")
    if control.get("controls_failed") is not True:
        failures.append("CONTROL_UNEXPECTED_PASS")

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
        print("122 checker failed:")
        for failure in sorted(set(failures)):
            print(f" - {failure}")
        return 1
    print("122 checker passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
