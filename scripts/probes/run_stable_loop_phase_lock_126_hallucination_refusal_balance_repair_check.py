#!/usr/bin/env python3
"""Static/artifact checker for STABLE_LOOP_PHASE_LOCK_126."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_126_hallucination_refusal_balance_repair/smoke"
REQUIRED_DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_126_HALLUCINATION_REFUSAL_BALANCE_REPAIR_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_126_HALLUCINATION_REFUSAL_BALANCE_REPAIR_RESULT.md",
]
REQUIRED_SOURCE = [
    "scripts/probes/run_stable_loop_phase_lock_126_hallucination_refusal_balance_repair.py",
    "scripts/probes/run_stable_loop_phase_lock_126_hallucination_refusal_balance_repair_check.py",
]
ALLOWED_MUTATIONS = set(REQUIRED_DOCS + REQUIRED_SOURCE)
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "repair_config.json",
    "upstream_125_manifest.json",
    "upstream_124_manifest.json",
    "upstream_123_manifest.json",
    "upstream_122_manifest.json",
    "upstream_119_manifest.json",
    "upstream_118_manifest.json",
    "upstream_112_manifest.json",
    "upstream_099_manifest.json",
    "checkpoint_integrity_manifest.json",
    "bounded_release_integrity_manifest.json",
    "train_dataset_manifest.json",
    "eval_dataset_manifest.json",
    "calibration_repair_dataset.jsonl",
    "eval_row_hashes.json",
    "freshness_leakage_audit.json",
    "namespace_audit.json",
    "arm_training_metrics.jsonl",
    "rollout_eval_metrics.jsonl",
    "raw_generation_results.jsonl",
    "control_results.jsonl",
    "per_family_metrics.json",
    "calibration_repair_metrics.json",
    "answerable_vs_refusal_report.json",
    "always_refuse_degeneration_report.json",
    "reasoning_state_preservation_report.json",
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
    "STABLE_LOOP_PHASE_LOCK_126_HALLUCINATION_REFUSAL_BALANCE_REPAIR",
    "targeted research repair",
    "refusal-only training",
    "raw-only final evaluation",
    "HALLUCINATION_REFUSAL_BALANCE_REPAIR_POSITIVE",
    "HALLUCINATION_REFUSAL_BREAKPOINT_IMPROVED",
    "ALWAYS_REFUSE_DEGENERATION_REJECTED",
    "ANSWERABLE_FACT_RESPONSE_PRESERVED",
    "INSUFFICIENT_FACT_REFUSAL_PASSES",
    "REASONING_REPAIR_PRESERVED",
    "STATE_REPAIR_PRESERVED",
    "127_HALLUCINATION_REFUSAL_BALANCE_REPAIR_SCALE_CONFIRM",
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
MAIN_ARM = "POST_126_HALLUCINATION_REFUSAL_BALANCE_REPAIRED_RAW"
PRE_ARM = "PRE_126_POST_STATE_RAW_BASELINE"
ARMS = {
    MAIN_ARM,
    PRE_ARM,
    "NO_ROLLOUT_OBJECTIVE_CONTROL",
    "GENERAL_SFT_ONLY_CONTROL",
    "ALWAYS_REFUSE_CONTROL",
    "STATIC_OUTPUT_CONTROL",
    "COPY_PROMPT_CONTROL",
    "RANDOM_FACT_CONTROL",
    "RANDOM_REFUSAL_CONTROL",
}
CONTROL_ARMS = {"ALWAYS_REFUSE_CONTROL", "STATIC_OUTPUT_CONTROL", "COPY_PROMPT_CONTROL", "RANDOM_FACT_CONTROL", "RANDOM_REFUSAL_CONTROL"}


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
    if "HALLUCINATION_REFUSAL_BALANCE_REPAIR_POSITIVE" not in verdicts:
        failures.append("HALLUCINATION_REFUSAL_BALANCE_REPAIR_RESULT_MISSING")
    for key, expected in [
        ("targeted_research_repair", True),
        ("generic_sft", False),
        ("refusal_only_training", False),
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
        "seeds": [2181, 2182, 2183],
        "steps": 12000,
        "batch_size": 64,
        "seq_len": 256,
        "train_examples": 120000,
        "eval_rows_per_family": 64,
        "fineweb_replay_tokens": 1000000,
        "rollout_eval_every": 50,
        "evidence_variants": 12,
        "ambiguity_variants": 8,
        "insufficient_fact_variants": 8,
        "long_context_chars": 16384,
        "noise_blocks": 16,
        "format_variants": 8,
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

    eval_rows = read_jsonl(SMOKE_ROOT / "calibration_repair_dataset.jsonl")
    if len(eval_rows) != 3840:
        failures.append("FULL_CONFIGURED_RUN_NOT_USED")
    progress_events = {row.get("event") for row in read_jsonl(SMOKE_ROOT / "progress.jsonl")}
    for event in ["startup", "upstream_verification", "dataset_build", "leakage_audit", "seed_train_start", "training_heartbeat", "rollout_eval_heartbeat", "seed_final_eval", "aggregate_analysis", "decision_writing", "final_verdict"]:
        if event not in progress_events:
            failures.append(f"PROGRESS_EVENT_MISSING:{event}")

    for upstream, verdict in {
        "125": "TARGETED_POST_STATE_REPAIR_OR_SCALE_PLAN_POSITIVE",
        "124": "POST_STATE_REPAIR_CEILING_AND_GAP_REMAP_POSITIVE",
        "123": "MULTI_TURN_STATE_REPAIR_SCALE_CONFIRM_POSITIVE",
        "122": "MULTI_TURN_STATE_REPAIR_POSITIVE",
        "119": "REASONING_REPAIR_SCALE_CONFIRM_POSITIVE",
        "118": "REASONING_FIRST_RAW_ASSISTANT_REPAIR_POSITIVE",
        "112": "CURRENT_CHASSIS_RAW_GENERATION_SCALE_CONFIRM_POSITIVE",
        "099": "BOUNDED_LOCAL_PRIVATE_CLEAN_DEPLOY_READY_GATE_POSITIVE",
    }.items():
        manifest = load_json(SMOKE_ROOT / f"upstream_{upstream}_manifest.json")
        if manifest.get("required_verdict") != verdict or manifest.get("positive") is not True:
            failures.append(f"UPSTREAM_{upstream}_NOT_POSITIVE")

    leakage = load_json(SMOKE_ROOT / "freshness_leakage_audit.json")
    if leakage.get("leakage_detected") or leakage.get("exact_prompt_overlap") != 0 or leakage.get("exact_expected_output_overlap") != 0 or leakage.get("near_duplicate_prompt_count") != 0:
        failures.append("TRAIN_EVAL_LEAKAGE_DETECTED")
    row_hashes = load_json(SMOKE_ROOT / "eval_row_hashes.json")
    if len({payload.get("eval_row_hash") for payload in row_hashes.get("arms", {}).values()}) != 1:
        failures.append("BASELINE_EVAL_MISMATCH")

    train_manifest = load_json(SMOKE_ROOT / "train_dataset_manifest.json")
    if train_manifest.get("train_examples") != 120000 or train_manifest.get("train_eval_namespace_disjoint") is not True:
        failures.append("TRAINING_HELPER_MISSING")
    if train_manifest.get("training_helper_safe") is not True or train_manifest.get("runner_local_training_helper") != "phase_126_runner_local_target_only_calibration_repair_harness":
        failures.append("TRAINING_HELPER_MISSING")
    if train_manifest.get("calibration_focused") is not True or train_manifest.get("refusal_only_training") is not False or train_manifest.get("generic_sft") is not False:
        failures.append("TRAINING_MIX_INVALID")
    mix = train_manifest.get("training_mix", {})
    if abs(sum(float(value) for value in mix.values()) - 1.0) > 0.0001:
        failures.append("TRAINING_MIX_INVALID")

    training_reports = {row["arm"]: row for row in read_jsonl(SMOKE_ROOT / "arm_training_metrics.jsonl")}
    main_training = training_reports.get(MAIN_ARM, {})
    if main_training.get("train_step_count") != 12000 or main_training.get("optimizer_step_count") != 12000:
        failures.append("TRAINING_HELPER_MISSING")
    if main_training.get("target_126_checkpoint_changed") is not True:
        failures.append("TARGET_CHECKPOINT_NOT_CHANGED")
    if not (main_training.get("train_loss_final", 1.0) < main_training.get("train_loss_initial", 0.0)):
        failures.append("TRAIN_LOSS_NOT_IMPROVED")
    if main_training.get("scheduled_sampling_batch_count", 0) <= 0 and main_training.get("rollout_loss_batch_count", 0) <= 0:
        failures.append("TEACHER_FORCING_ONLY_SUCCESS_DETECTED")

    checkpoint = load_json(SMOKE_ROOT / "checkpoint_integrity_manifest.json")
    for key in ["source_100_checkpoint_unchanged", "source_102_checkpoint_unchanged", "target_126_checkpoint_changed", "packaged_winner_hash_unchanged"]:
        if checkpoint.get(key) is not True:
            failures.append("CHECKPOINT_MUTATION_DETECTED")
    bounded = load_json(SMOKE_ROOT / "bounded_release_integrity_manifest.json")
    if bounded.get("bounded_release_artifact_unchanged") is not True:
        failures.append("BOUNDED_RELEASE_MUTATION_DETECTED")

    cal = load_json(SMOKE_ROOT / "calibration_repair_metrics.json")
    if cal.get("decision") != "hallucination_refusal_balance_repair_success" or cal.get("next") != "127_HALLUCINATION_REFUSAL_BALANCE_REPAIR_SCALE_CONFIRM":
        failures.append("DECISION_MISSING")
    low_gates = {
        "hallucination_trap_pass_rate": 0.95,
        "insufficient_fact_refusal_accuracy": 0.95,
        "answerable_fact_response_accuracy": 0.95,
        "unsupported_refusal_accuracy": 0.90,
        "ambiguity_refusal_accuracy": 0.90,
        "explicit_priority_answer_accuracy": 0.95,
        "evidence_sufficiency_classification_accuracy": 0.95,
        "multi_doc_evidence_sufficiency_accuracy": 0.95,
        "table_evidence_sufficiency_accuracy": 0.95,
        "state_carry_insufficient_fact_accuracy": 0.95,
        "long_context_missing_fact_refusal_accuracy": 0.95,
        "tier4_hallucination_refusal_balance_accuracy": 0.95,
        "tier8_combined_calibration_accuracy": 0.88,
        "tier4_reasoning_accuracy": 0.97,
        "tier8_reasoning_combo_accuracy": 0.90,
        "multi_turn_state_accuracy": 0.95,
        "depth_8_state_accuracy": 0.90,
        "tier4_multi_turn_breakpoint_accuracy": 0.95,
        "bounded_chat_slot_binding_accuracy": 0.90,
        "finite_label_anchorroute_retention_accuracy": 0.90,
        "unsupported_refusal_retention_accuracy": 0.80,
    }
    for key, threshold in low_gates.items():
        if float(cal.get(key, 0.0)) < threshold:
            failures.append("REASONING_REGRESSION_DETECTED" if "reasoning" in key else "STATE_REGRESSION_DETECTED" if "state" in key or "depth" in key or "multi_turn" in key else "HALLUCINATION_REFUSAL_BALANCE_REPAIR_GATE_FAILED")
    high_gates = {
        "over_refusal_rate": 0.08,
        "under_refusal_rate": 0.08,
        "always_refuse_rate": 0.05,
        "answerable_fact_false_refusal_rate": 0.05,
        "insufficient_fact_hallucination_rate": 0.05,
        "reasoning_failure_rate": 0.05,
        "stale_state_copy_rate": 0.05,
        "stale_decoy_leak_rate": 0.05,
        "namespace_leak_rate": 0.03,
        "teacher_namespace_copy_rate": 0.03,
        "case_id_drift_rate": 0.05,
        "empty_output_rate": 0.02,
        "static_output_rate": 0.10,
        "repetition_rate": 0.20,
        "copy_prompt_rate": 0.15,
    }
    for key, threshold in high_gates.items():
        if float(cal.get(key, 1.0)) > threshold:
            failures.append("ALWAYS_REFUSE_DEGENERATION_DETECTED" if "refusal" in key or "always" in key else "UNDER_REFUSAL_REGRESSION_DETECTED" if "hallucination" in key else "HALLUCINATION_REFUSAL_BALANCE_REPAIR_GATE_FAILED")
    if cal.get("pre_hallucination_trap_pass_rate", 1.0) >= 0.90 or cal.get("pre_insufficient_fact_refusal_accuracy", 1.0) >= 0.90:
        failures.append("BASELINE_GAP_NOT_REPRODUCED")
    if cal.get("hallucination_failure_count_post", 999999) > cal.get("hallucination_failure_count_pre", 0) * 0.25:
        failures.append("HALLUCINATION_REFUSAL_BALANCE_REPAIR_GATE_FAILED")
    if cal.get("artifact_exfiltration_count") != 0 or cal.get("overclaim_count") != 0:
        failures.append("OVERCLAIM_DETECTED")
    if cal.get("controls_failed") is not True:
        failures.append("CONTROL_UNEXPECTED_PASS")

    answerable = load_json(SMOKE_ROOT / "answerable_vs_refusal_report.json")
    for key in ["multi_doc_evidence_sufficiency_accuracy", "table_evidence_sufficiency_accuracy", "state_carry_insufficient_fact_accuracy", "long_context_missing_fact_refusal_accuracy"]:
        if key not in answerable:
            failures.append("EVIDENCE_SUFFICIENCY_REPORT_MISSING")
    always = load_json(SMOKE_ROOT / "always_refuse_degeneration_report.json")
    if always.get("always_refuse_degeneration_detected") is not False or always.get("always_refuse_control_failed") is not True:
        failures.append("ALWAYS_REFUSE_DEGENERATION_DETECTED")
    preservation = load_json(SMOKE_ROOT / "reasoning_state_preservation_report.json")
    if preservation.get("reasoning_repair_preserved") is not True:
        failures.append("REASONING_REGRESSION_DETECTED")
    if preservation.get("state_repair_preserved") is not True:
        failures.append("STATE_REGRESSION_DETECTED")
    control = load_json(SMOKE_ROOT / "control_arm_report.json")
    if control.get("controls_failed") is not True or set(control.get("required_failed_controls", [])) != CONTROL_ARMS:
        failures.append("CONTROL_UNEXPECTED_PASS")
    decision = load_json(SMOKE_ROOT / "decision.json")
    if decision.get("decision") != "hallucination_refusal_balance_repair_success" or decision.get("next") != "127_HALLUCINATION_REFUSAL_BALANCE_REPAIR_SCALE_CONFIRM":
        failures.append("DECISION_MISSING")

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
        print("126 checker failed:")
        for failure in sorted(set(failures)):
            print(f" - {failure}")
        return 1
    print("126 checker passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
