#!/usr/bin/env python3
"""Static/artifact checker for STABLE_LOOP_PHASE_LOCK_102."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_102_decoder_policy_and_rollout_repair/smoke"
REQUIRED_DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_102_DECODER_POLICY_AND_ROLLOUT_REPAIR_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_102_DECODER_POLICY_AND_ROLLOUT_REPAIR_RESULT.md",
]
REQUIRED_SOURCE = [
    "scripts/probes/run_stable_loop_phase_lock_102_decoder_policy_and_rollout_repair.py",
    "scripts/probes/run_stable_loop_phase_lock_102_decoder_policy_and_rollout_repair_check.py",
]
ALLOWED_MUTATIONS = set(REQUIRED_DOCS + REQUIRED_SOURCE)
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "repair_config.json",
    "upstream_101_manifest.json",
    "source_checkpoint_manifest.json",
    "repair_dataset_manifest.json",
    "repair_train_examples_sample.jsonl",
    "repair_eval_examples_sample.jsonl",
    "training_metrics.jsonl",
    "checkpoint_manifest.json",
    "checkpoint_hashes.json",
    "pre_repair_eval_metrics.json",
    "post_repair_eval_metrics.json",
    "arm_comparison.json",
    "control_delta_report.json",
    "raw_rollout_drift_report.json",
    "case_id_anchor_report.json",
    "slot_pinning_report.json",
    "language_guard_report.json",
    "lm_retention_report.json",
    "retention_report.json",
    "collapse_metrics.json",
    "generation_samples.jsonl",
    "human_readable_samples.jsonl",
    "failure_case_samples.jsonl",
    "summary.json",
    "report.md",
]
REQUIRED_TERMS = [
    "STABLE_LOOP_PHASE_LOCK_102_DECODER_POLICY_AND_ROLLOUT_REPAIR",
    "target-only research repair",
    "raw generation repair only",
    "not GPT-like assistant readiness",
    "not open-domain assistant readiness",
    "not production chat",
    "not public API",
    "not deployment readiness",
    "not public release",
    "not safety alignment",
    "DECODER_POLICY_AND_ROLLOUT_REPAIR_POSITIVE",
    "UPSTREAM_101_FRONTIER_MAP_VERIFIED",
    "RAW_GENERATION_IMPROVES",
    "CASE_ID_DRIFT_REDUCED",
    "SOURCE_100_CHECKPOINT_LOADED_READ_ONLY",
    "RAW_GENERATION_PATH_CONTAMINATED",
    "SOURCE_100_CHECKPOINT_MUTATION_DETECTED",
    "BOUNDED_RELEASE_MUTATION_DETECTED",
    "PACKAGED_CHECKPOINT_MUTATION_DETECTED",
    "NO_ACTUAL_TRAINING_UPDATE_DETECTED",
    "TOKEN_OBJECTIVE_NOT_LEARNED",
    "TRAIN_EVAL_LEAKAGE_DETECTED",
    "EVAL_ROW_MISMATCH",
    "CASE_ID_ANCHOR_REPAIR_INSUFFICIENT",
    "DISTRACTOR_NUMBER_COPY_DETECTED",
    "SLOT_DRIFT_REGRESSION_DETECTED",
    "CONTROL_DELTA_INSUFFICIENT",
    "DECODER_ASSISTED_REGRESSION_DETECTED",
    "LM_RETENTION_REGRESSION_DETECTED",
    "RETENTION_REGRESSION_DETECTED",
    "103_FRESH_RAW_GENERATION_CONFIRM",
    "102B_CASE_ID_ANCHOR_FAILURE_ANALYSIS",
]
NEGATION_MARKERS = ["not ", "no ", "does not ", "do not ", "without ", "false", "forbidden", "forbid", "reject", "rejected", "only", "cannot"]
FORBIDDEN_CLAIMS = {
    "GPT_LIKE_READINESS_FALSE_CLAIM": ["GPT-like assistant readiness"],
    "OPEN_DOMAIN_ASSISTANT_CLAIM_DETECTED": ["open-domain assistant readiness"],
    "PRODUCTION_CHAT_CLAIM_DETECTED": ["production chat"],
    "PUBLIC_API_CLAIM_DETECTED": ["public API"],
    "DEPLOYMENT_CLAIM_DETECTED": ["deployment readiness"],
    "PUBLIC_RELEASE_CLAIM_DETECTED": ["public release"],
    "SAFETY_ALIGNMENT_CLAIM_DETECTED": ["safety alignment"],
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
        window = lowered[max(0, match.start() - 160) : match.start()]
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
    if "DECODER_POLICY_AND_ROLLOUT_REPAIR_POSITIVE" not in verdicts:
        failures.append("DECODER_POLICY_AND_ROLLOUT_REPAIR_POSITIVE_MISSING")
    expected = {
        "source_100_checkpoint_unchanged": True,
        "bounded_release_artifact_unchanged": True,
        "packaged_winner_hash_unchanged": True,
        "target_102_checkpoint_changed": True,
        "decoder_assisted_used_for_raw": False,
        "ranked_scoring_used_for_raw": False,
        "response_table_used_for_main_prediction": False,
        "prediction_oracle_used": False,
        "llm_judge_used": False,
    }
    for key, value in expected.items():
        if metrics.get(key) != value:
            failures.append(f"GATE_VALUE_FAIL:{key}")
    if metrics.get("raw_generation_path") != "autoregressive":
        failures.append("RAW_GENERATION_PATH_CONTAMINATED")
    if metrics.get("train_step_count", 0) <= 0 or not metrics.get("train_loss_final", 999.0) < metrics.get("train_loss_initial", -999.0):
        failures.append("TOKEN_OBJECTIVE_NOT_LEARNED")
    if metrics.get("raw_free_generation_accuracy", 0.0) < metrics.get("upstream_101_raw_free_generation_accuracy", 0.0) + 0.25 or metrics.get("raw_free_generation_accuracy", 0.0) < 0.50:
        failures.append("RAW_GENERATION_NOT_IMPROVED")
    if metrics.get("case_id_drift_rate", 1.0) > metrics.get("upstream_101_case_id_drift_rate", 1.0) * 0.50:
        failures.append("CASE_ID_ANCHOR_REPAIR_INSUFFICIENT")
    if metrics.get("distractor_number_copy_rate", 1.0) > 0.10:
        failures.append("DISTRACTOR_NUMBER_COPY_DETECTED")
    if metrics.get("slot_drift_rate", 1.0) > metrics.get("upstream_101_slot_drift_rate", 0.0) + 0.02:
        failures.append("SLOT_DRIFT_REGRESSION_DETECTED")
    if metrics.get("decoder_assisted_accuracy", 0.0) < 0.90 or metrics.get("decoder_assisted_accuracy_delta", -1.0) < -0.03:
        failures.append("DECODER_ASSISTED_REGRESSION_DETECTED")
    for key, threshold in {
        "bounded_chat_slot_binding_accuracy": 0.90,
        "finite_label_anchorroute_retention_accuracy": 0.90,
        "unsupported_refusal_accuracy": 0.80,
        "utf8_valid_generation_rate": 0.80,
        "nonempty_generation_rate": 0.98,
    }.items():
        if metrics.get(key, 0.0) < threshold:
            failures.append(f"METRIC_GATE_FAIL:{key}")
    if metrics.get("empty_output_rate", 1.0) > 0.02 or metrics.get("static_output_rate", 1.0) > 0.15 or metrics.get("repetition_rate", 1.0) > 0.25 or metrics.get("copy_prompt_rate", 1.0) > 0.20:
        failures.append("COLLAPSE_GATE_FAIL")
    if metrics.get("fineweb_eval_loss_regression", 999.0) > 0.50 or metrics.get("next_byte_accuracy_drop", 999.0) > 0.12:
        failures.append("LM_RETENTION_REGRESSION_DETECTED")
    dataset = load_json(SMOKE_ROOT / "repair_dataset_manifest.json")
    if dataset.get("overlap_with_101_eval_count") != 0 or dataset.get("train_eval_exact_prompt_overlap_count") != 0 or dataset.get("train_eval_exact_response_overlap_count") != 0 or dataset.get("max_train_eval_prompt_jaccard", 1.0) >= 0.90:
        failures.append("TRAIN_EVAL_LEAKAGE_DETECTED")
    arms = load_json(SMOKE_ROOT / "arm_comparison.json")
    arm_hashes = {arm.get("eval_row_hash") for arm in arms.get("arms", [])}
    arm_counts = {arm.get("eval_count") for arm in arms.get("arms", [])}
    if arms.get("all_eval_rows_match") is not True or len(arm_hashes) != 1 or len(arm_counts) != 1:
        failures.append("EVAL_ROW_MISMATCH")
    deltas = load_json(SMOKE_ROOT / "control_delta_report.json")
    if deltas.get("case_id_drift_delta_vs_no_case_id_anchor_loss_control", 0.0) < 0.25 or deltas.get("raw_accuracy_delta_vs_no_rollout_consistency_control", 0.0) < 0.15 or deltas.get("wrong_language_delta_vs_no_language_guard_control", -1.0) < 0.0:
        failures.append("CONTROL_DELTA_INSUFFICIENT")
    if deltas.get("main_beats_copy_prompt_control") is not True or deltas.get("main_beats_static_output_control") is not True:
        failures.append("COPY_OR_STATIC_CONTROL_UNEXPECTED_PASS")
    if len(read_jsonl(SMOKE_ROOT / "human_readable_samples.jsonl")) < 40:
        failures.append("HUMAN_SAMPLE_REPORT_MISSING")
    if len(read_jsonl(SMOKE_ROOT / "training_metrics.jsonl")) < 5:
        failures.append("TRAINING_METRICS_INSUFFICIENT")
    failures.extend(find_false_claims((SMOKE_ROOT / "report.md").read_text(encoding="utf-8")))
    for path in changed_paths():
        if path.startswith("target/"):
            failures.append("GENERATED_ARTIFACT_STAGED")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Check STABLE_LOOP_PHASE_LOCK_102 artifacts")
    parser.add_argument("--check-only", action="store_true")
    _args = parser.parse_args()
    failures: list[str] = []
    missing, files = read_files()
    failures.extend(f"MISSING_FILE:{item}" for item in missing)
    combined = "\n".join(files.values())
    for term in REQUIRED_TERMS:
        if term not in combined:
            failures.append(f"TERM_MISSING:{term}")
    failures.extend(find_false_claims(combined))
    if runtime_surface_mutation_detected():
        failures.append("RUNTIME_SURFACE_MUTATION_DETECTED")
    if "LICENSE" in changed_paths():
        failures.append("ROOT_LICENSE_CHANGED")
    if SMOKE_ROOT.exists():
        failures.extend(check_artifacts())
    else:
        failures.append("SMOKE_ROOT_MISSING")
    if failures:
        for failure in failures:
            print(failure)
        return 1
    print("STABLE_LOOP_PHASE_LOCK_102_DECODER_POLICY_AND_ROLLOUT_REPAIR_CHECK_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
