#!/usr/bin/env python3
"""Static/artifact checker for STABLE_LOOP_PHASE_LOCK_103."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_103_fresh_raw_generation_confirm/smoke"
REQUIRED_DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_103_FRESH_RAW_GENERATION_CONFIRM_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_103_FRESH_RAW_GENERATION_CONFIRM_RESULT.md",
]
REQUIRED_SOURCE = [
    "scripts/probes/run_stable_loop_phase_lock_103_fresh_raw_generation_confirm.py",
    "scripts/probes/run_stable_loop_phase_lock_103_fresh_raw_generation_confirm_check.py",
]
ALLOWED_MUTATIONS = set(REQUIRED_DOCS + REQUIRED_SOURCE)
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "eval_config.json",
    "upstream_102_manifest.json",
    "checkpoint_manifest.json",
    "fresh_raw_eval_dataset.jsonl",
    "raw_generation_results.jsonl",
    "decoder_assisted_results.jsonl",
    "pre_repair_baseline_results.jsonl",
    "control_results.jsonl",
    "family_metrics.json",
    "mode_comparison.json",
    "case_id_anchor_report.json",
    "slot_pinning_report.json",
    "refusal_boundary_report.json",
    "language_diagnostic_report.json",
    "retention_report.json",
    "collapse_metrics.json",
    "raw_vs_decoder_gap.json",
    "failure_case_samples.jsonl",
    "human_readable_samples.jsonl",
    "decision_recommendation.json",
    "summary.json",
    "report.md",
]
REQUIRED_TERMS = [
    "STABLE_LOOP_PHASE_LOCK_103_FRESH_RAW_GENERATION_CONFIRM",
    "fresh raw-generation confirmation only",
    "eval-only",
    "no model capability improved by 103",
    "not GPT-like assistant readiness",
    "not open-domain assistant readiness",
    "not production chat",
    "not public API",
    "not deployment readiness",
    "not public release",
    "not safety alignment",
    "FRESH_RAW_GENERATION_CONFIRM_POSITIVE",
    "UPSTREAM_102_REPAIR_VERIFIED",
    "RAW_GENERATION_GENERALIZES",
    "CASE_ID_ANCHOR_GENERALIZES",
    "SLOT_PINNING_GENERALIZES",
    "RAW_GENERATION_PATH_CONTAMINATED",
    "ORACLE_SHORTCUT_DETECTED",
    "TRAIN_EVAL_LEAKAGE_DETECTED",
    "EVAL_ROW_MISMATCH",
    "CASE_ID_ANCHOR_GENERALIZATION_FAILS",
    "DISTRACTOR_NUMBER_COPY_DETECTED",
    "SLOT_PINNING_GENERALIZATION_FAILS",
    "DISTRACTOR_SUPPRESSION_REGRESSION_DETECTED",
    "CHECKPOINT_MUTATION_DETECTED",
    "BOUNDED_RELEASE_MUTATION_DETECTED",
    "TRAINING_SIDE_EFFECT_DETECTED",
    "RETENTION_REGRESSION_DETECTED",
    "104_MULTI_SEED_RAW_GENERATION_CONFIRM",
    "103B_CASE_ID_ANCHOR_GENERALIZATION_FAILURE_ANALYSIS",
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
    paths: list[str] = []
    for line in git_status().splitlines():
        if not line.strip():
            continue
        paths.append(line[3:].replace("\\", "/"))
    return paths


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
    if "FRESH_RAW_GENERATION_CONFIRM_POSITIVE" not in verdicts:
        failures.append("FRESH_RAW_GENERATION_CONFIRM_POSITIVE_MISSING")
    expected_false = [
        "ranked_scoring_used_for_raw",
        "prefix_forcing_used_for_raw",
        "decoder_assisted_used_for_raw",
        "response_table_used_for_raw",
        "prediction_oracle_used",
        "llm_judge_used",
    ]
    for key in expected_false:
        if metrics.get(key) is not False:
            failures.append(f"RAW_GENERATION_PATH_CONTAMINATED:{key}")
    expected_true = [
        "checkpoint_hash_unchanged",
        "bounded_release_artifact_unchanged",
        "source_100_checkpoint_unchanged",
    ]
    for key in expected_true:
        if metrics.get(key) is not True:
            failures.append(f"GATE_VALUE_FAIL:{key}")
    if metrics.get("raw_generation_path") != "autoregressive":
        failures.append("RAW_GENERATION_PATH_CONTAMINATED")
    if metrics.get("train_step_count") != 0 or metrics.get("optimizer_step_count") != 0:
        failures.append("TRAINING_SIDE_EFFECT_DETECTED")
    if metrics.get("raw_free_generation_accuracy", 0.0) < 0.85:
        failures.append("RAW_GENERATION_GENERALIZATION_FAILS")
    if metrics.get("raw_free_generation_accuracy", 0.0) < metrics.get("upstream_101_raw_free_generation_accuracy", 0.0) + 0.50:
        failures.append("RAW_GENERATION_GENERALIZATION_FAILS")
    for key, threshold in {
        "case_id_drift_rate": 0.10,
        "distractor_number_copy_rate": 0.10,
        "slot_drift_rate": 0.05,
        "distractor_leak_rate": 0.10,
        "empty_output_rate": 0.02,
        "static_output_rate": 0.15,
        "repetition_rate": 0.25,
        "copy_prompt_rate": 0.20,
    }.items():
        if metrics.get(key, 1.0) > threshold:
            failures.append(f"METRIC_GATE_FAIL:{key}")
    if metrics.get("decoder_assisted_accuracy", 0.0) < 0.90 or metrics.get("decoder_assisted_accuracy_delta_vs_102", -1.0) < -0.05:
        failures.append("DECODER_ASSISTED_REGRESSION_DETECTED")
    for key, threshold in {
        "bounded_chat_slot_binding_accuracy": 0.90,
        "finite_label_anchorroute_retention_accuracy": 0.90,
        "unsupported_refusal_accuracy": 0.80,
        "nonempty_generation_rate": 0.98,
        "utf8_valid_generation_rate": 0.80,
    }.items():
        if metrics.get(key, 0.0) < threshold:
            failures.append(f"METRIC_GATE_FAIL:{key}")
    for key in [
        "overlap_with_101_eval_count",
        "overlap_with_102_train_count",
        "overlap_with_102_eval_count",
        "overlap_with_100_train_eval_count",
    ]:
        if metrics.get(key) != 0:
            failures.append("TRAIN_EVAL_LEAKAGE_DETECTED")
    if metrics.get("max_prompt_jaccard_vs_102_train", 1.0) >= 0.90 or metrics.get("max_prompt_jaccard_vs_102_eval", 1.0) >= 0.90:
        failures.append("TRAIN_EVAL_LEAKAGE_DETECTED")
    mode_comparison = load_json(SMOKE_ROOT / "mode_comparison.json")
    if mode_comparison.get("all_eval_rows_match") is not True:
        failures.append("EVAL_ROW_MISMATCH")
    mode_hashes = {mode.get("eval_row_hash") for mode in mode_comparison.get("modes", [])}
    mode_counts = {mode.get("eval_count") for mode in mode_comparison.get("modes", [])}
    if len(mode_hashes) != 1 or len(mode_counts) != 1:
        failures.append("EVAL_ROW_MISMATCH")
    family_metrics = load_json(SMOKE_ROOT / "family_metrics.json")
    if family_metrics.get("all_families_reported") is not True:
        failures.append("FAMILY_METRICS_INCOMPLETE")
    decision = load_json(SMOKE_ROOT / "decision_recommendation.json")
    if decision.get("primary_next_milestone") != "104_MULTI_SEED_RAW_GENERATION_CONFIRM" or decision.get("mechanically_derived") is not True:
        failures.append("DECISION_RECOMMENDATION_MISSING")
    if len(read_jsonl(SMOKE_ROOT / "human_readable_samples.jsonl")) < 40:
        failures.append("HUMAN_SAMPLE_REPORT_MISSING")
    if len(read_jsonl(SMOKE_ROOT / "fresh_raw_eval_dataset.jsonl")) < 100:
        failures.append("EVAL_DATASET_BUILD_FAILS")
    failures.extend(find_false_claims((SMOKE_ROOT / "report.md").read_text(encoding="utf-8")))
    for path in changed_paths():
        if path.startswith("target/"):
            failures.append("GENERATED_ARTIFACT_STAGED")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Check STABLE_LOOP_PHASE_LOCK_103 artifacts")
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
    print("STABLE_LOOP_PHASE_LOCK_103_FRESH_RAW_GENERATION_CONFIRM_CHECK_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
