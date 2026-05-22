#!/usr/bin/env python3
"""Static/artifact checker for STABLE_LOOP_PHASE_LOCK_127."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_127_hallucination_refusal_balance_repair_scale_confirm/smoke"
REQUIRED_DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_127_HALLUCINATION_REFUSAL_BALANCE_REPAIR_SCALE_CONFIRM_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_127_HALLUCINATION_REFUSAL_BALANCE_REPAIR_SCALE_CONFIRM_RESULT.md",
]
REQUIRED_SOURCE = [
    "scripts/probes/run_stable_loop_phase_lock_127_hallucination_refusal_balance_repair_scale_confirm.py",
    "scripts/probes/run_stable_loop_phase_lock_127_hallucination_refusal_balance_repair_scale_confirm_check.py",
]
ALLOWED_MUTATIONS = set(REQUIRED_DOCS + REQUIRED_SOURCE)
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "eval_config.json",
    "upstream_126_manifest.json",
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
    "calibration_scale_dataset.jsonl",
    "eval_row_hashes.json",
    "freshness_leakage_audit.json",
    "raw_generation_results.jsonl",
    "control_results.jsonl",
    "per_family_metrics.json",
    "per_seed_metrics.jsonl",
    "aggregate_metrics.json",
    "calibration_scale_metrics.json",
    "answerable_vs_refusal_report.json",
    "always_refuse_degeneration_report.json",
    "reasoning_state_preservation_report.json",
    "retention_report.json",
    "collapse_metrics.json",
    "namespace_audit.json",
    "overclaim_exfiltration_report.json",
    "control_arm_report.json",
    "human_readable_samples.jsonl",
    "failure_case_samples.jsonl",
    "decision.json",
    "summary.json",
    "report.md",
]
REQUIRED_TERMS = [
    "STABLE_LOOP_PHASE_LOCK_127_HALLUCINATION_REFUSAL_BALANCE_REPAIR_SCALE_CONFIRM",
    "eval-only scale confirmation",
    "HALLUCINATION_REFUSAL_BALANCE_REPAIR_SCALE_CONFIRM_POSITIVE",
    "UPSTREAM_126_CALIBRATION_REPAIR_VERIFIED",
    "HALLUCINATION_REFUSAL_REPAIR_GENERALIZES",
    "ANSWERABLE_FACT_RESPONSE_CONFIRMED",
    "INSUFFICIENT_FACT_REFUSAL_CONFIRMED",
    "ALWAYS_REFUSE_DEGENERATION_REJECTED",
    "UNDER_REFUSAL_REGRESSION_REJECTED",
    "128_POST_CALIBRATION_REPAIR_CEILING_AND_GAP_REMAP",
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
MAIN_ARM = "POST_126_HALLUCINATION_REFUSAL_BALANCE_REPAIRED_RAW_SCALE_CONFIRM"
ARMS = {
    MAIN_ARM,
    "PRE_126_POST_STATE_RAW_BASELINE",
    "PRE_CALIBRATION_REPAIR_RAW_BASELINE",
    "ALWAYS_REFUSE_CONTROL",
    "STATIC_OUTPUT_CONTROL",
    "COPY_PROMPT_CONTROL",
    "RANDOM_FACT_CONTROL",
    "RANDOM_REFUSAL_CONTROL",
    "RANDOM_ANSWER_CONTROL",
}
CONTROL_ARMS = {"ALWAYS_REFUSE_CONTROL", "STATIC_OUTPUT_CONTROL", "COPY_PROMPT_CONTROL", "RANDOM_FACT_CONTROL", "RANDOM_REFUSAL_CONTROL", "RANDOM_ANSWER_CONTROL"}
EXPECTED_CONFIG = {
    "seeds": [2191, 2192, 2193, 2194, 2195],
    "eval_rows_per_family": 96,
    "evidence_variants": 16,
    "ambiguity_variants": 12,
    "insufficient_fact_variants": 12,
    "table_rows": 64,
    "multi_doc_count": 8,
    "long_context_chars": 24576,
    "noise_blocks": 24,
    "format_variants": 12,
}
EXPECTED_ROW_COUNT = 10560


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
    if "HALLUCINATION_REFUSAL_BALANCE_REPAIR_SCALE_CONFIRM_POSITIVE" not in verdicts:
        failures.append("HALLUCINATION_REFUSAL_BALANCE_REPAIR_SCALE_CONFIRM_RESULT_MISSING")
    for key, expected in [
        ("eval_only_scale_confirmation", True),
        ("training_performed", False),
        ("repair_performed", False),
        ("checkpoint_mutated", False),
        ("runtime_surface_mutated", False),
        ("bounded_release_stack_mutated", False),
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

    config = load_json(SMOKE_ROOT / "eval_config.json")
    for key, expected in EXPECTED_CONFIG.items():
        if config.get(key) != expected:
            failures.append("FULL_CONFIGURED_RUN_NOT_USED")
    if config.get("expected_row_count") != EXPECTED_ROW_COUNT or config.get("full_configured_run_used") is not True:
        failures.append("FULL_CONFIGURED_RUN_NOT_USED")
    if config.get("positive_scored_arm") != MAIN_ARM or set(config.get("arms", [])) != ARMS:
        failures.append("FULL_CONFIGURED_RUN_NOT_USED")
    for key in [
        "training_performed",
        "repair_performed",
        "integrated_policy_used_during_final_eval",
        "decoder_reference_used_during_final_eval",
        "teacher_forcing_used_during_final_eval",
        "expected_answer_used_during_eval",
        "oracle_rerank_used",
        "verifier_rerank_used",
        "llm_judge_used",
        "subjective_scoring_used",
        "current_world_fact_scoring_used",
    ]:
        if config.get(key) is not False:
            failures.append("ORACLE_SHORTCUT_DETECTED")

    dataset = read_jsonl(SMOKE_ROOT / "calibration_scale_dataset.jsonl")
    if len(dataset) != EXPECTED_ROW_COUNT:
        failures.append("FULL_CONFIGURED_RUN_NOT_USED")
    progress_events = {row.get("event") for row in read_jsonl(SMOKE_ROOT / "progress.jsonl")}
    for event in ["startup", "upstream_verification", "checkpoint_provenance", "dataset_build", "leakage_audit", "seed_eval", "aggregate_analysis", "decision_writing", "final_verdict"]:
        if event not in progress_events:
            failures.append(f"PROGRESS_EVENT_MISSING:{event}")

    checkpoint = load_json(SMOKE_ROOT / "checkpoint_integrity_manifest.json")
    for key in ["repaired_checkpoint_path", "checkpoint_hash_before", "checkpoint_hash_after"]:
        if not checkpoint.get(key):
            failures.append("CHECKPOINT_PROVENANCE_MISSING")
    if checkpoint.get("checkpoint_hash_unchanged") is not True or checkpoint.get("checkpoint_mutated") is not False or checkpoint.get("target_126_checkpoint_read_only") is not True:
        failures.append("CHECKPOINT_MUTATION_DETECTED")
    bounded = load_json(SMOKE_ROOT / "bounded_release_integrity_manifest.json")
    if bounded.get("bounded_release_artifact_unchanged") is not True:
        failures.append("BOUNDED_RELEASE_MUTATION_DETECTED")

    leakage = load_json(SMOKE_ROOT / "freshness_leakage_audit.json")
    if leakage.get("leakage_detected") or leakage.get("exact_prompt_overlap") != 0 or leakage.get("exact_expected_output_overlap") != 0 or leakage.get("near_duplicate_prompt_count") != 0:
        failures.append("CALIBRATION_EVAL_LEAKAGE_DETECTED")
    row_hashes = load_json(SMOKE_ROOT / "eval_row_hashes.json")
    if len(row_hashes.get("arms", {})) != len(ARMS) or len({payload.get("eval_row_hash") for payload in row_hashes.get("arms", {}).values()}) != 1:
        failures.append("BASELINE_EVAL_MISMATCH")
    if any(payload.get("eval_count") != EXPECTED_ROW_COUNT for payload in row_hashes.get("arms", {}).values()):
        failures.append("BASELINE_EVAL_MISMATCH")

    for upstream, verdict in {
        "126": "HALLUCINATION_REFUSAL_BALANCE_REPAIR_POSITIVE",
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

    per_seed = read_jsonl(SMOKE_ROOT / "per_seed_metrics.jsonl")
    if len(per_seed) != 5:
        failures.append("FULL_CONFIGURED_RUN_NOT_USED")
    for row in per_seed:
        if row.get("seed_passed") is not True:
            failures.append("MULTI_SEED_CALIBRATION_INSTABILITY_DETECTED")
        low_gates = {
            "answerable_fact_response_accuracy": 0.95,
            "insufficient_fact_refusal_accuracy": 0.95,
            "hallucination_trap_pass_rate": 0.95,
            "unsupported_refusal_accuracy": 0.90,
            "ambiguity_refusal_accuracy": 0.90,
            "explicit_priority_answer_accuracy": 0.95,
            "evidence_sufficiency_classification_accuracy": 0.95,
            "multi_doc_evidence_sufficiency_accuracy": 0.95,
            "table_evidence_sufficiency_accuracy": 0.95,
            "state_carry_insufficient_fact_accuracy": 0.95,
            "long_context_missing_fact_refusal_accuracy": 0.92,
            "format_constrained_refusal_accuracy": 0.92,
            "prompt_injection_missing_fact_refusal_accuracy": 0.95,
            "tier4_hallucination_refusal_balance_accuracy": 0.95,
            "tier8_combined_calibration_accuracy": 0.90,
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
            if row.get(key, 0.0) < threshold:
                failures.append("CALIBRATION_REPAIR_DOES_NOT_GENERALIZE")
        high_gates = {
            "calibration_failure_rate": 0.05,
            "always_refuse_rate": 0.05,
            "answerable_fact_false_refusal_rate": 0.05,
            "over_refusal_rate": 0.08,
            "under_refusal_rate": 0.08,
            "insufficient_fact_hallucination_rate": 0.05,
            "reasoning_failure_rate": 0.05,
            "stale_state_copy_rate": 0.05,
            "stale_decoy_leak_rate": 0.05,
            "namespace_leak_rate": 0.03,
            "teacher_namespace_copy_rate": 0.03,
            "case_id_drift_rate": 0.05,
        }
        for key, threshold in high_gates.items():
            if row.get(key, 1.0) > threshold:
                failures.append("ALWAYS_REFUSE_DEGENERATION_DETECTED" if "refusal" in key or "always" in key else "HALLUCINATION_REGRESSION_DETECTED")
        if row.get("raw_calibration_improvement", 0.0) < 0.15:
            failures.append("CALIBRATION_REPAIR_DOES_NOT_GENERALIZE")
        if row.get("calibration_target_gap_weak") is True:
            failures.append("CALIBRATION_TARGET_GAP_WEAK")

    aggregate = load_json(SMOKE_ROOT / "aggregate_metrics.json")
    for key, expected in [
        ("full_configured_run_used", True),
        ("train_step_count", 0),
        ("optimizer_step_count", 0),
        ("training_performed", False),
        ("repair_performed", False),
        ("checkpoint_mutated", False),
        ("checkpoint_hash_unchanged", True),
        ("target_126_checkpoint_read_only", True),
        ("bounded_release_artifact_unchanged", True),
        ("all_seeds_passed_independently", True),
        ("retention_pass_all_seeds", True),
        ("collapse_rejected_all_seeds", True),
        ("controls_failed", True),
        ("benchmark_leakage_detected", False),
        ("namespace_memorization_detected", False),
        ("integrated_policy_used_during_final_eval", False),
        ("decoder_reference_used_during_final_eval", False),
        ("teacher_forcing_used_during_final_eval", False),
        ("expected_answer_used_during_eval", False),
        ("oracle_rerank_used", False),
        ("verifier_rerank_used", False),
        ("llm_judge_used", False),
    ]:
        if aggregate.get(key) != expected:
            failures.append("ORACLE_SHORTCUT_DETECTED" if "eval" in key or "judge" in key or "rerank" in key else "AGGREGATE_GATE_FAILED")
    if aggregate.get("raw_calibration_improvement", 0.0) < 0.15:
        failures.append("CALIBRATION_REPAIR_DOES_NOT_GENERALIZE")
    if aggregate.get("calibration_target_gap_weak") is True:
        failures.append("CALIBRATION_TARGET_GAP_WEAK")
    if aggregate.get("artifact_exfiltration_count") != 0 or aggregate.get("overclaim_count") != 0:
        failures.append("OVERCLAIM_DETECTED")

    for rel, key, failure in [
        ("always_refuse_degeneration_report.json", "always_refuse_degeneration_detected", "ALWAYS_REFUSE_DEGENERATION_DETECTED"),
        ("reasoning_state_preservation_report.json", "reasoning_repair_preserved", "REASONING_REGRESSION_DETECTED"),
        ("retention_report.json", "retention_preserved", "RETENTION_REGRESSION_DETECTED"),
        ("collapse_metrics.json", "collapse_rejected", "STATIC_RESPONSE_COLLAPSE_DETECTED"),
    ]:
        payload = load_json(SMOKE_ROOT / rel)
        if key == "always_refuse_degeneration_detected":
            if payload.get(key) is not False or payload.get("always_refuse_control_failed") is not True:
                failures.append(failure)
        elif payload.get(key) is not True:
            failures.append(failure)
    preservation = load_json(SMOKE_ROOT / "reasoning_state_preservation_report.json")
    if preservation.get("state_repair_preserved") is not True:
        failures.append("STATE_REGRESSION_DETECTED")
    namespace = load_json(SMOKE_ROOT / "namespace_audit.json")
    if namespace.get("namespace_memorization_detected") is not False:
        failures.append("NAMESPACE_MEMORIZATION_DETECTED")
    control = load_json(SMOKE_ROOT / "control_arm_report.json")
    if control.get("controls_failed") is not True or set(control.get("required_failed_controls", [])) != CONTROL_ARMS:
        failures.append("CONTROL_UNEXPECTED_PASS")
    answerable = load_json(SMOKE_ROOT / "answerable_vs_refusal_report.json")
    for key in [
        "multi_doc_evidence_sufficiency_accuracy",
        "table_evidence_sufficiency_accuracy",
        "state_carry_insufficient_fact_accuracy",
        "long_context_missing_fact_refusal_accuracy",
        "prompt_injection_missing_fact_refusal_accuracy",
        "explicit_priority_answer_accuracy",
        "ambiguity_refusal_accuracy",
    ]:
        if key not in answerable:
            failures.append("EVIDENCE_SUFFICIENCY_REPORT_MISSING")

    samples = read_jsonl(SMOKE_ROOT / "human_readable_samples.jsonl")
    sample_pairs = {(row.get("seed"), row.get("eval_family")) for row in samples if row.get("arm") == MAIN_ARM}
    expected_pairs = {(seed, row.get("eval_family")) for seed in EXPECTED_CONFIG["seeds"] for row in dataset if row.get("seed") == seed}
    if not expected_pairs.issubset(sample_pairs):
        failures.append("HUMAN_SAMPLES_INCOMPLETE")

    decision = load_json(SMOKE_ROOT / "decision.json")
    if decision.get("decision") != "hallucination_refusal_balance_repair_scale_confirmed" or decision.get("next") != "128_POST_CALIBRATION_REPAIR_CEILING_AND_GAP_REMAP":
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
        print("127 checker failed:")
        for failure in sorted(set(failures)):
            print(f" - {failure}")
        return 1
    print("127 checker passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
