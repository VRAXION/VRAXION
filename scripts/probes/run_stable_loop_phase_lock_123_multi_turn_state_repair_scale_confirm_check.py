#!/usr/bin/env python3
"""Static/artifact checker for STABLE_LOOP_PHASE_LOCK_123."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_123_multi_turn_state_repair_scale_confirm/smoke"
REQUIRED_DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_123_MULTI_TURN_STATE_REPAIR_SCALE_CONFIRM_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_123_MULTI_TURN_STATE_REPAIR_SCALE_CONFIRM_RESULT.md",
]
REQUIRED_SOURCE = [
    "scripts/probes/run_stable_loop_phase_lock_123_multi_turn_state_repair_scale_confirm.py",
    "scripts/probes/run_stable_loop_phase_lock_123_multi_turn_state_repair_scale_confirm_check.py",
]
ALLOWED_MUTATIONS = set(REQUIRED_DOCS + REQUIRED_SOURCE)
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "eval_config.json",
    "upstream_122_manifest.json",
    "upstream_121_manifest.json",
    "upstream_120_manifest.json",
    "upstream_119_manifest.json",
    "upstream_118_manifest.json",
    "upstream_112_manifest.json",
    "upstream_099_manifest.json",
    "checkpoint_integrity_manifest.json",
    "bounded_release_integrity_manifest.json",
    "state_scale_dataset.jsonl",
    "eval_row_hashes.json",
    "freshness_leakage_audit.json",
    "raw_generation_results.jsonl",
    "control_results.jsonl",
    "per_family_metrics.json",
    "per_seed_metrics.jsonl",
    "depth_metrics.json",
    "aggregate_metrics.json",
    "state_scale_metrics.json",
    "state_depth_report.json",
    "reasoning_preservation_report.json",
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
    "STABLE_LOOP_PHASE_LOCK_123_MULTI_TURN_STATE_REPAIR_SCALE_CONFIRM",
    "eval-only scale confirmation",
    "MULTI_TURN_STATE_REPAIR_SCALE_CONFIRM_POSITIVE",
    "UPSTREAM_122_STATE_REPAIR_VERIFIED",
    "MULTI_TURN_STATE_REPAIR_GENERALIZES",
    "DEPTH_8_STATE_TRACKING_CONFIRMED",
    "REASONING_REPAIR_PRESERVED",
    "STALE_STATE_MEMORIZATION_REJECTED",
    "124_POST_STATE_REPAIR_CEILING_AND_GAP_REMAP",
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
MAIN_ARM = "POST_122_MULTI_TURN_STATE_REPAIRED_RAW_SCALE_CONFIRM"
ARMS = {
    MAIN_ARM,
    "PRE_122_POST_REASONING_RAW_BASELINE",
    "PRE_STATE_REPAIR_RAW_BASELINE",
    "STATIC_OUTPUT_CONTROL",
    "COPY_PROMPT_CONTROL",
    "RANDOM_STATE_CONTROL",
    "STALE_STATE_COPY_CONTROL",
    "RANDOM_SLOT_CONTROL",
}
EXPECTED_CONFIG = {
    "seeds": [2161, 2162, 2163, 2164, 2165],
    "eval_rows_per_family": 96,
    "multi_turn_depths": [2, 4, 6, 8],
    "diagnostic_depths": [10, 12],
    "state_update_variants": 12,
    "stale_decoy_count": 8,
    "table_rows": 48,
    "multi_doc_count": 6,
    "long_context_chars": 16384,
    "noise_blocks": 16,
    "format_variants": 8,
}
EXPECTED_ROW_COUNT = 8640


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
    if "MULTI_TURN_STATE_REPAIR_SCALE_CONFIRM_POSITIVE" not in verdicts:
        failures.append("MULTI_TURN_STATE_REPAIR_SCALE_CONFIRM_RESULT_MISSING")
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

    dataset = read_jsonl(SMOKE_ROOT / "state_scale_dataset.jsonl")
    if len(dataset) != EXPECTED_ROW_COUNT:
        failures.append("FULL_CONFIGURED_RUN_NOT_USED")
    progress_events = {row.get("event") for row in read_jsonl(SMOKE_ROOT / "progress.jsonl")}
    for event in [
        "startup",
        "upstream_verification",
        "checkpoint_provenance",
        "dataset_build",
        "leakage_audit",
        "seed_eval",
        "aggregate_analysis",
        "decision_writing",
        "final_verdict",
    ]:
        if event not in progress_events:
            failures.append(f"PROGRESS_EVENT_MISSING:{event}")

    checkpoint = load_json(SMOKE_ROOT / "checkpoint_integrity_manifest.json")
    for key in ["repaired_checkpoint_path", "checkpoint_hash_before", "checkpoint_hash_after"]:
        if not checkpoint.get(key):
            failures.append("CHECKPOINT_PROVENANCE_MISSING")
    if checkpoint.get("checkpoint_hash_unchanged") is not True or checkpoint.get("checkpoint_mutated") is not False or checkpoint.get("target_122_checkpoint_read_only") is not True:
        failures.append("CHECKPOINT_MUTATION_DETECTED")
    bounded = load_json(SMOKE_ROOT / "bounded_release_integrity_manifest.json")
    if bounded.get("bounded_release_artifact_unchanged") is not True:
        failures.append("BOUNDED_RELEASE_MUTATION_DETECTED")

    leakage = load_json(SMOKE_ROOT / "freshness_leakage_audit.json")
    if leakage.get("leakage_detected") or leakage.get("exact_prompt_overlap") != 0 or leakage.get("exact_expected_output_overlap") != 0 or leakage.get("near_duplicate_prompt_count") != 0:
        failures.append("STATE_EVAL_LEAKAGE_DETECTED")
    row_hashes = load_json(SMOKE_ROOT / "eval_row_hashes.json")
    if len(row_hashes.get("arms", {})) != len(ARMS) or len({payload.get("eval_row_hash") for payload in row_hashes.get("arms", {}).values()}) != 1:
        failures.append("BASELINE_EVAL_MISMATCH")
    if any(payload.get("eval_count") != EXPECTED_ROW_COUNT for payload in row_hashes.get("arms", {}).values()):
        failures.append("BASELINE_EVAL_MISMATCH")

    per_seed = read_jsonl(SMOKE_ROOT / "per_seed_metrics.jsonl")
    if len(per_seed) != 5:
        failures.append("FULL_CONFIGURED_RUN_NOT_USED")
    for row in per_seed:
        if row.get("seed_passed") is not True:
            failures.append("MULTI_SEED_STATE_INSTABILITY_DETECTED")
        for key, threshold in [
            ("multi_turn_state_accuracy", 0.95),
            ("state_tracking_accuracy", 0.95),
            ("multi_turn_correction_accuracy", 0.95),
            ("active_vs_stale_tracking_accuracy", 0.95),
            ("override_chain_accuracy", 0.92),
            ("slot_update_sequence_accuracy", 0.95),
            ("stale_state_rejection_accuracy", 0.95),
            ("active_slot_after_update_accuracy", 0.95),
            ("tier4_multi_turn_breakpoint_accuracy", 0.95),
            ("tier7_state_combo_accuracy", 0.88),
            ("tier8_combined_state_accuracy", 0.88),
            ("depth_2_state_accuracy", 0.95),
            ("depth_4_state_accuracy", 0.95),
            ("depth_6_state_accuracy", 0.95),
            ("depth_8_state_accuracy", 0.90),
            ("tier4_reasoning_accuracy", 0.97),
            ("tier8_reasoning_combo_accuracy", 0.90),
            ("bounded_chat_slot_binding_accuracy", 0.90),
            ("finite_label_anchorroute_retention_accuracy", 0.90),
            ("unsupported_refusal_retention_accuracy", 0.80),
        ]:
            if row.get(key, 0.0) < threshold:
                failures.append("STATE_REPAIR_DOES_NOT_GENERALIZE")
        if row.get("state_accuracy_margin_vs_baseline", 0.0) < 0.20 or row.get("state_failure_rate", 1.0) > 0.05:
            failures.append("STATE_REPAIR_DOES_NOT_GENERALIZE")
        if row.get("reasoning_failure_rate", 1.0) > 0.05:
            failures.append("REASONING_REGRESSION_DETECTED")
        for key in ["namespace_leak_rate", "teacher_namespace_copy_rate"]:
            if row.get(key, 1.0) > 0.03:
                failures.append("NAMESPACE_MEMORIZATION_DETECTED")
        for key in ["case_id_drift_rate", "stale_state_copy_rate", "stale_decoy_leak_rate"]:
            if row.get(key, 1.0) > 0.05:
                failures.append("STALE_STATE_MEMORIZATION_DETECTED")

    depth = load_json(SMOKE_ROOT / "depth_metrics.json")
    for key, threshold in [
        ("depth_2_state_accuracy", 0.95),
        ("depth_4_state_accuracy", 0.95),
        ("depth_6_state_accuracy", 0.95),
        ("depth_8_state_accuracy", 0.90),
    ]:
        if depth.get(key, 0.0) < threshold:
            failures.append("DEPTH_8_STATE_REGRESSION_DETECTED")
    if "diagnostic_depth_10_state_accuracy" not in depth or "diagnostic_depth_12_state_accuracy" not in depth:
        failures.append("DIAGNOSTIC_DEPTH_MISSING")

    aggregate = load_json(SMOKE_ROOT / "aggregate_metrics.json")
    for key, expected in [
        ("full_configured_run_used", True),
        ("train_step_count", 0),
        ("optimizer_step_count", 0),
        ("training_performed", False),
        ("repair_performed", False),
        ("checkpoint_mutated", False),
        ("checkpoint_hash_unchanged", True),
        ("bounded_release_artifact_unchanged", True),
        ("all_seeds_passed_independently", True),
        ("retention_pass_all_seeds", True),
        ("collapse_rejected_all_seeds", True),
        ("controls_failed", True),
        ("benchmark_leakage_detected", False),
        ("namespace_memorization_detected", False),
        ("stale_state_memorization_detected", False),
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
    for key, threshold in [
        ("min_multi_turn_state_accuracy", 0.95),
        ("min_depth_8_state_accuracy", 0.90),
        ("tier4_reasoning_accuracy", 0.97),
        ("tier8_reasoning_combo_accuracy", 0.90),
    ]:
        if aggregate.get(key, 0.0) < threshold:
            failures.append("STATE_REPAIR_DOES_NOT_GENERALIZE")
    if aggregate.get("reasoning_failure_rate", 1.0) > 0.05:
        failures.append("REASONING_REGRESSION_DETECTED")
    if aggregate.get("stale_state_copy_rate", 1.0) > 0.05 or aggregate.get("stale_decoy_leak_rate", 1.0) > 0.05:
        failures.append("STALE_STATE_MEMORIZATION_DETECTED")
    if aggregate.get("artifact_exfiltration_count") != 0 or aggregate.get("overclaim_count") != 0:
        failures.append("OVERCLAIM_DETECTED")

    for rel, verdict, expected_bool in [
        ("reasoning_preservation_report.json", "REASONING_REGRESSION_DETECTED", "reasoning_repair_preserved"),
        ("retention_report.json", "RETENTION_REGRESSION_DETECTED", "retention_preserved"),
        ("collapse_metrics.json", "STATIC_RESPONSE_COLLAPSE_DETECTED", "collapse_rejected"),
    ]:
        payload = load_json(SMOKE_ROOT / rel)
        if payload.get(expected_bool) is not True:
            failures.append(verdict)
    namespace = load_json(SMOKE_ROOT / "namespace_audit.json")
    if namespace.get("namespace_memorization_detected") is not False or namespace.get("stale_state_memorization_detected") is not False:
        failures.append("NAMESPACE_MEMORIZATION_DETECTED")
    control = load_json(SMOKE_ROOT / "control_arm_report.json")
    if control.get("controls_failed") is not True:
        failures.append("CONTROL_UNEXPECTED_PASS")

    samples = read_jsonl(SMOKE_ROOT / "human_readable_samples.jsonl")
    sample_pairs = {(row.get("seed"), row.get("eval_family")) for row in samples if row.get("arm") == MAIN_ARM}
    expected_pairs = {(seed, row.get("eval_family")) for seed in EXPECTED_CONFIG["seeds"] for row in dataset if row.get("seed") == seed}
    if not expected_pairs.issubset(sample_pairs):
        failures.append("HUMAN_SAMPLES_INCOMPLETE")

    decision = load_json(SMOKE_ROOT / "decision.json")
    if decision.get("decision") != "multi_turn_state_repair_scale_confirmed" or decision.get("next") != "124_POST_STATE_REPAIR_CEILING_AND_GAP_REMAP":
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
        print("123 checker failed:")
        for failure in sorted(set(failures)):
            print(f" - {failure}")
        return 1
    print("123 checker passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
