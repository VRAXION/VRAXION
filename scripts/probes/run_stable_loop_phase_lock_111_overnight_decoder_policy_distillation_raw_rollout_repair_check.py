#!/usr/bin/env python3
"""Static/artifact checker for STABLE_LOOP_PHASE_LOCK_111."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_111_overnight_decoder_policy_distillation_raw_rollout_repair/smoke"
REQUIRED_DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_111_OVERNIGHT_DECODER_POLICY_DISTILLATION_RAW_ROLLOUT_REPAIR_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_111_OVERNIGHT_DECODER_POLICY_DISTILLATION_RAW_ROLLOUT_REPAIR_RESULT.md",
]
REQUIRED_SOURCE = [
    "scripts/probes/run_stable_loop_phase_lock_111_overnight_decoder_policy_distillation_raw_rollout_repair.py",
    "scripts/probes/run_stable_loop_phase_lock_111_overnight_decoder_policy_distillation_raw_rollout_repair_check.py",
]
ALLOWED_MUTATIONS = set(REQUIRED_DOCS + REQUIRED_SOURCE)
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "overnight_config.json",
    "upstream_110_manifest.json",
    "upstream_109_manifest.json",
    "upstream_108a_manifest.json",
    "upstream_100_manifest.json",
    "upstream_099_manifest.json",
    "bounded_release_integrity_manifest.json",
    "source_checkpoint_manifest.json",
    "teacher_dataset_manifest.json",
    "train_dataset_manifest.json",
    "eval_dataset_manifest.json",
    "train_examples_sample.jsonl",
    "eval_examples_sample.jsonl",
    "training_metrics.jsonl",
    "resource_metrics.jsonl",
    "checkpoint_manifest.json",
    "checkpoint_hashes.json",
    "generation_results_pre_raw.jsonl",
    "generation_results_post_raw.jsonl",
    "generation_results_integrated_teacher.jsonl",
    "arm_comparison.json",
    "fineweb_retention_metrics.json",
    "bounded_retention_metrics.json",
    "collapse_metrics.json",
    "overclaim_metrics.json",
    "resource_report.json",
    "human_readable_samples.jsonl",
    "failure_case_samples.jsonl",
    "decision.json",
    "summary.json",
    "report.md",
]
REQUIRED_TERMS = [
    "STABLE_LOOP_PHASE_LOCK_111_OVERNIGHT_DECODER_POLICY_DISTILLATION_RAW_ROLLOUT_REPAIR",
    "target-only overnight research training",
    "OVERNIGHT_DECODER_POLICY_DISTILLATION_RAW_ROLLOUT_REPAIR_POSITIVE",
    "OVERNIGHT_RUNTIME_UTILIZED",
    "TEACHER_DATASET_BUILT",
    "TARGET_RAW_DISTILLATION_TRAINING_COMPLETED",
    "RAW_OOD_ACCURACY_IMPROVES",
    "RAW_TO_INTEGRATED_GAP_REDUCED",
    "FINEWEB_RETENTION_WITHIN_LIMITS",
    "BOUNDED_RETENTION_PASSES",
    "CUDA_AVAILABLE_BUT_NOT_USED",
    "RESOURCE_UNDERUTILIZATION_DETECTED",
    "OVERNIGHT_RUNTIME_UNDERUSED",
    "INTEGRATED_POLICY_USED_DURING_RAW_EVAL",
    "ORACLE_SHORTCUT_DETECTED",
    "TEACHER_DATASET_LEAKAGE_DETECTED",
    "TRAIN_EVAL_LEAKAGE_DETECTED",
    "BASELINE_EVAL_MISMATCH",
    "RAW_OOD_ACCURACY_NOT_IMPROVED",
    "RAW_TO_INTEGRATED_GAP_REMAINS_HIGH",
    "FINEWEB_RETENTION_REGRESSION_DETECTED",
    "BOUNDED_RETENTION_REGRESSION_DETECTED",
    "SOURCE_CHECKPOINT_MUTATION_DETECTED",
    "PACKAGED_CHECKPOINT_MUTATION_DETECTED",
    "BOUNDED_RELEASE_MUTATION_DETECTED",
    "RUNTIME_SURFACE_MUTATION_DETECTED",
    "112_RAW_ASSISTANT_MULTI_SEED_OOD_CONFIRM",
    "111B_DISTILLATION_PARTIAL_FAILURE_ANALYSIS",
    "111R_RETENTION_OR_LM_REGRESSION_ANALYSIS",
    "111H_OVERNIGHT_HARNESS_UTILIZATION_FIX",
    "111C_BOUNDARY_FAILURE_ANALYSIS",
    "not GPT-like assistant readiness",
    "not open-domain assistant readiness",
    "not production chat",
    "not public API",
    "not deployment readiness",
    "not safety alignment",
]
NEGATION_MARKERS = ["not ", "no ", "does not ", "do not ", "without ", "false", "forbidden", "forbid", "reject", "rejected", "only", "cannot"]
FORBIDDEN_CLAIMS = {
    "GPT_LIKE_READINESS_FALSE_CLAIM": ["GPT-like assistant readiness"],
    "OPEN_DOMAIN_ASSISTANT_CLAIM_DETECTED": ["open-domain assistant readiness"],
    "PRODUCTION_CHAT_CLAIM_DETECTED": ["production chat"],
    "PUBLIC_API_CLAIM_DETECTED": ["public API"],
    "DEPLOYMENT_CLAIM_DETECTED": ["deployment readiness"],
    "SAFETY_ALIGNMENT_CLAIM_DETECTED": ["safety alignment"],
}
EXPECTED_ARMS = {
    "PRE_111_RAW_BASELINE",
    "POST_111_RAW_DISTILLED",
    "INTEGRATED_TEACHER_REFERENCE",
    "NO_FINEWEB_REPLAY_CONTROL",
    "NO_RETENTION_MIX_CONTROL",
    "SFT_ONLY_NO_TEACHER_CONTROL",
    "STATIC_OUTPUT_CONTROL",
    "COPY_PROMPT_CONTROL",
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
        window = lowered[max(0, match.start() - 180) : match.start()]
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
    protected_prefixes = ("instnct-core/", "tools/instnct_service_alpha/", "tools/instnct_deploy/", "sdk/", "packages/", "docs/product/", "docs/releases/")
    for path in changed_paths():
        if path in ALLOWED_MUTATIONS or path.startswith("target/"):
            continue
        if path == "LICENSE" or path.startswith(protected_prefixes):
            return True
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
    if "OVERNIGHT_DECODER_POLICY_DISTILLATION_RAW_ROLLOUT_REPAIR_POSITIVE" not in verdicts:
        failures.append("OVERNIGHT_DECODER_POLICY_DISTILLATION_RESULT_MISSING")
    if metrics.get("min_runtime_minutes", 0) < 360 or metrics.get("wall_clock_minutes", 0) < metrics.get("min_runtime_minutes", 360):
        failures.append("OVERNIGHT_RUNTIME_UNDERUSED")
    if metrics.get("early_finish_prevented") is True and metrics.get("extra_batches_launched_if_needed") is not True:
        failures.append("OVERNIGHT_RUNTIME_UNDERUSED")
    if metrics.get("cuda_available") is True and metrics.get("selected_device") != "cuda" and not metrics.get("cpu_only_fallback_declared"):
        failures.append("CUDA_AVAILABLE_BUT_NOT_USED")
    resource = load_json(SMOKE_ROOT / "resource_report.json")
    for key in ["gpu_utilization_samples", "gpu_memory_samples", "median_gpu_utilization", "p75_gpu_utilization", "p95_gpu_utilization", "gpu_idle_fraction"]:
        if key not in resource:
            failures.append("RESOURCE_REPORT_MISSING")
    if metrics.get("resource_underutilization_detected"):
        failures.append("RESOURCE_UNDERUTILIZATION_DETECTED")

    if metrics.get("target_111_checkpoint_changed") is not True or metrics.get("train_step_count", 0) <= 0 or metrics.get("optimizer_step_count", 0) <= 0:
        failures.append("NO_ACTUAL_TRAINING_UPDATE_DETECTED")
    if not metrics.get("train_loss_final", 1) < metrics.get("train_loss_initial", 0):
        failures.append("NO_ACTUAL_TRAINING_UPDATE_DETECTED")
    if metrics.get("post_111_raw_ood_accuracy", 0) < 0.80 or metrics.get("raw_accuracy_improvement", 0) < 0.20:
        failures.append("RAW_OOD_ACCURACY_NOT_IMPROVED")
    if metrics.get("post_111_raw_accuracy_gap_to_integrated_teacher", 1) > 0.15:
        failures.append("RAW_TO_INTEGRATED_GAP_REMAINS_HIGH")
    if metrics.get("post_111_raw_per_family_min_accuracy", 0) < 0.65:
        failures.append("RAW_OOD_ACCURACY_NOT_IMPROVED")
    if metrics.get("bounded_chat_slot_binding_accuracy", 0) < 0.90 or metrics.get("finite_label_anchorroute_retention_accuracy", 0) < 0.90 or metrics.get("unsupported_refusal_retention_accuracy", 0) < 0.80:
        failures.append("BOUNDED_RETENTION_REGRESSION_DETECTED")
    if metrics.get("fineweb_eval_loss_regression", 999) > 0.50 or metrics.get("fineweb_next_byte_accuracy_drop", 999) > 0.10:
        failures.append("FINEWEB_RETENTION_REGRESSION_DETECTED")
    for key in ["artifact_exfiltration_count", "gpt_like_claim_count", "production_chat_claim_count", "public_api_claim_count", "safety_alignment_claim_count"]:
        if metrics.get(key, 1) != 0:
            failures.append("ARTIFACT_EXFILTRATION_DETECTED" if key == "artifact_exfiltration_count" else "OVERCLAIM_DETECTED")
    if metrics.get("empty_output_rate", 1) > 0.02:
        failures.append("EMPTY_OUTPUT_COLLAPSE_DETECTED")
    if metrics.get("static_output_rate", 1) > 0.15:
        failures.append("STATIC_RESPONSE_COLLAPSE_DETECTED")
    if metrics.get("repetition_rate", 1) > 0.25 or metrics.get("copy_prompt_rate", 1) > 0.20:
        failures.append("REPETITION_COLLAPSE_DETECTED")
    if metrics.get("source_102_checkpoint_unchanged") is not True or metrics.get("source_100_checkpoint_unchanged") is not True:
        failures.append("SOURCE_CHECKPOINT_MUTATION_DETECTED")
    if metrics.get("packaged_winner_hash_unchanged") is not True:
        failures.append("PACKAGED_CHECKPOINT_MUTATION_DETECTED")
    if metrics.get("bounded_release_artifact_unchanged") is not True:
        failures.append("BOUNDED_RELEASE_MUTATION_DETECTED")
    for key in ["integrated_policy_used_during_final_raw_eval", "decoder_reference_used_during_final_raw_eval", "expected_answer_used_during_eval"]:
        if metrics.get(key) is not False:
            failures.append("INTEGRATED_POLICY_USED_DURING_RAW_EVAL" if "integrated" in key else "ORACLE_SHORTCUT_DETECTED")
    for key in ["teacher_eval_exact_prompt_overlap_count", "train_eval_exact_prompt_overlap_count", "train_eval_exact_response_overlap_count"]:
        if metrics.get(key, 1) != 0:
            failures.append("TRAIN_EVAL_LEAKAGE_DETECTED")
    if metrics.get("max_train_eval_prompt_jaccard", 1) >= 0.90:
        failures.append("TRAIN_EVAL_LEAKAGE_DETECTED")
    if metrics.get("max_teacher_eval_prompt_jaccard", 1) >= 0.90:
        failures.append("TEACHER_DATASET_LEAKAGE_DETECTED")

    arm = load_json(SMOKE_ROOT / "arm_comparison.json")
    arms = {row.get("arm") for row in arm.get("arms", [])}
    if arms != EXPECTED_ARMS or arm.get("all_eval_rows_match") is not True:
        failures.append("BASELINE_EVAL_MISMATCH")
    hashes = {row.get("eval_row_hash") for row in arm.get("arms", [])}
    counts = {row.get("eval_row_count") for row in arm.get("arms", [])}
    if len(hashes) != 1 or len(counts) != 1:
        failures.append("BASELINE_EVAL_MISMATCH")
    samples = read_jsonl(SMOKE_ROOT / "human_readable_samples.jsonl")
    if not samples:
        failures.append("HUMAN_SAMPLE_REPORT_MISSING")
    sample_arms = {row.get("arm") for row in samples}
    if not {"PRE_111_RAW_BASELINE", "POST_111_RAW_DISTILLED", "INTEGRATED_TEACHER_REFERENCE"}.issubset(sample_arms):
        failures.append("HUMAN_SAMPLE_REPORT_MISSING")
    progress = read_jsonl(SMOKE_ROOT / "progress.jsonl")
    if not progress or not any(row.get("event") == "training heartbeat" for row in progress):
        failures.append("RESOURCE_REPORT_MISSING")
    if not read_jsonl(SMOKE_ROOT / "training_metrics.jsonl") or not read_jsonl(SMOKE_ROOT / "resource_metrics.jsonl"):
        failures.append("RESOURCE_REPORT_MISSING")

    decision = load_json(SMOKE_ROOT / "decision.json")
    if decision.get("next") not in {
        "112_RAW_ASSISTANT_MULTI_SEED_OOD_CONFIRM",
        "111B_DISTILLATION_PARTIAL_FAILURE_ANALYSIS",
        "111R_RETENTION_OR_LM_REGRESSION_ANALYSIS",
        "111H_OVERNIGHT_HARNESS_UTILIZATION_FIX",
        "111C_BOUNDARY_FAILURE_ANALYSIS",
    }:
        failures.append("DECISION_RECOMMENDATION_MISSING")
    report = (SMOKE_ROOT / "report.md").read_text(encoding="utf-8")
    failures.extend(find_false_claims(report))
    for path in changed_paths():
        if path.startswith("target/"):
            failures.append("GENERATED_ARTIFACT_STAGED")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Check STABLE_LOOP_PHASE_LOCK_111 artifacts")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    failures: list[str] = []
    missing, files = read_files()
    failures.extend(f"MISSING_FILE:{item}" for item in missing)
    for term in REQUIRED_TERMS:
        if not any(term in text for text in files.values()):
            failures.append(f"MISSING_TERM:{term}")
    combined = "\n".join(files.values())
    failures.extend(find_false_claims(combined))
    if runtime_surface_mutation_detected():
        failures.append("RUNTIME_SURFACE_MUTATION_DETECTED")
    if args.check_only:
        failures.extend(check_artifacts())
    if failures:
        print("STABLE_LOOP_PHASE_LOCK_111_CHECK_FAIL")
        for failure in sorted(set(failures)):
            print(failure)
        return 1
    print("STABLE_LOOP_PHASE_LOCK_111_CHECK_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
