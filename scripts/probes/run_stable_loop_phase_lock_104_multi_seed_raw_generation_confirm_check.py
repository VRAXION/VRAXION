#!/usr/bin/env python3
"""Static/artifact checker for STABLE_LOOP_PHASE_LOCK_104."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_104_multi_seed_raw_generation_confirm/smoke"
REQUIRED_DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_104_MULTI_SEED_RAW_GENERATION_CONFIRM_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_104_MULTI_SEED_RAW_GENERATION_CONFIRM_RESULT.md",
]
REQUIRED_SOURCE = [
    "scripts/probes/run_stable_loop_phase_lock_104_multi_seed_raw_generation_confirm.py",
    "scripts/probes/run_stable_loop_phase_lock_104_multi_seed_raw_generation_confirm_check.py",
]
ALLOWED_MUTATIONS = set(REQUIRED_DOCS + REQUIRED_SOURCE)
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "eval_config.json",
    "upstream_103_manifest.json",
    "checkpoint_manifest.json",
    "seed_run_manifest.json",
    "multi_seed_aggregate.json",
    "family_metrics.json",
    "mode_comparison.json",
    "case_id_anchor_report.json",
    "slot_pinning_report.json",
    "retention_report.json",
    "collapse_metrics.json",
    "raw_vs_decoder_gap.json",
    "human_readable_samples.jsonl",
    "failure_case_samples.jsonl",
    "decision_recommendation.json",
    "summary.json",
    "report.md",
]
REQUIRED_TERMS = [
    "STABLE_LOOP_PHASE_LOCK_104_MULTI_SEED_RAW_GENERATION_CONFIRM",
    "multi-seed raw-generation confirmation only",
    "eval-only",
    "no model capability improved by 104",
    "not GPT-like assistant readiness",
    "not open-domain assistant readiness",
    "not production chat",
    "not public API",
    "not deployment readiness",
    "not public release",
    "not safety alignment",
    "Hungarian capability not claimed",
    "MULTI_SEED_RAW_GENERATION_CONFIRM_POSITIVE",
    "RAW_GENERATION_GENERALIZES_ALL_SEEDS",
    "CASE_ID_ANCHOR_GENERALIZES_ALL_SEEDS",
    "MULTI_SEED_RAW_GENERATION_INSTABILITY_DETECTED",
    "RAW_GENERATION_PATH_CONTAMINATED",
    "ORACLE_SHORTCUT_DETECTED",
    "STALE_SEED_ARTIFACT_USED",
    "TRAIN_EVAL_LEAKAGE_DETECTED",
    "EVAL_ROW_MISMATCH",
    "RETENTION_REGRESSION_DETECTED",
    "105_RAW_GENERATION_OOD_AND_BOUNDARY_CONFIRM",
    "104B_CASE_ID_MULTI_SEED_FAILURE_ANALYSIS",
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
    "HUNGARIAN_CAPABILITY_CLAIM_DETECTED": ["Hungarian assistant capability"],
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


def check_seed(seed_item: dict[str, Any], aggregate: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    seed = seed_item["seed"]
    seed_root = SMOKE_ROOT / f"seed_{seed}"
    for rel in ["summary.json", "report.md", "eval_config.json", "fresh_raw_eval_dataset.jsonl", "mode_comparison.json", "human_readable_samples.jsonl", "checkpoint_manifest.json"]:
        if not (seed_root / rel).exists():
            failures.append(f"MISSING_SEED_ARTIFACT:{seed}:{rel}")
    if failures:
        return failures
    if seed_item.get("seed_run_started") is not True or seed_item.get("seed_run_completed") is not True:
        failures.append(f"STALE_SEED_ARTIFACT_USED:{seed}")
    if seed_item.get("seed_summary_newer_than_104_start") is not True or seed_item.get("seed_report_newer_than_104_start") is not True:
        failures.append(f"STALE_SEED_ARTIFACT_USED:{seed}")
    if not seed_item.get("seed_command"):
        failures.append(f"STALE_SEED_ARTIFACT_USED:{seed}:missing_command")
    summary = load_json(seed_root / "summary.json")
    metrics = summary.get("metrics", {})
    if "SEED_RAW_GENERATION_CONFIRM_POSITIVE" not in set(summary.get("verdicts", [])):
        failures.append(f"MULTI_SEED_RAW_GENERATION_INSTABILITY_DETECTED:{seed}")
    expected_false = ["decoder_assisted_used_for_raw", "ranked_scoring_used_for_raw", "prefix_forcing_used_for_raw", "response_table_used_for_raw", "prediction_oracle_used", "llm_judge_used"]
    for key in expected_false:
        if metrics.get(key) is not False:
            failures.append(f"RAW_GENERATION_PATH_CONTAMINATED:{seed}:{key}")
    if metrics.get("raw_generation_path") != "autoregressive":
        failures.append(f"RAW_GENERATION_PATH_CONTAMINATED:{seed}")
    for key in ["checkpoint_hash_unchanged", "source_100_checkpoint_unchanged", "bounded_release_artifact_unchanged"]:
        if metrics.get(key) is not True:
            failures.append(f"CHECKPOINT_OR_RELEASE_GATE_FAIL:{seed}:{key}")
    if metrics.get("train_step_count") != 0 or metrics.get("optimizer_step_count") != 0:
        failures.append(f"TRAINING_SIDE_EFFECT_DETECTED:{seed}")
    for key in ["overlap_with_101_eval_count", "overlap_with_102_train_count", "overlap_with_102_eval_count", "overlap_with_103_eval_count", "overlap_with_100_train_eval_count"]:
        if metrics.get(key) != 0:
            failures.append(f"TRAIN_EVAL_LEAKAGE_DETECTED:{seed}:{key}")
    if metrics.get("max_prompt_jaccard_vs_102_train", 1.0) >= 0.90 or metrics.get("max_prompt_jaccard_vs_103_eval", 1.0) >= 0.90:
        failures.append(f"TRAIN_EVAL_LEAKAGE_DETECTED:{seed}:jaccard")
    if metrics.get("raw_free_generation_accuracy", 0.0) < 0.85:
        failures.append(f"RAW_GENERATION_GENERALIZATION_FAILS:{seed}")
    if metrics.get("case_id_drift_rate", 1.0) > 0.10 or metrics.get("distractor_number_copy_rate", 1.0) > 0.10:
        failures.append(f"CASE_ID_ANCHOR_GENERALIZATION_FAILS:{seed}")
    if metrics.get("slot_drift_rate", 1.0) > 0.05 or metrics.get("distractor_leak_rate", 1.0) > 0.10:
        failures.append(f"SLOT_OR_DISTRACTOR_GATE_FAIL:{seed}")
    if metrics.get("decoder_assisted_accuracy", 0.0) < 0.90 or metrics.get("decoder_assisted_accuracy_delta_vs_102", -1.0) < -0.05:
        failures.append(f"DECODER_ASSISTED_REGRESSION_DETECTED:{seed}")
    for key, threshold in {"bounded_chat_slot_binding_accuracy": 0.90, "finite_label_anchorroute_retention_accuracy": 0.90, "unsupported_refusal_accuracy": 0.80}.items():
        if metrics.get(key, 0.0) < threshold:
            failures.append(f"RETENTION_REGRESSION_DETECTED:{seed}:{key}")
    for key, threshold in {"empty_output_rate": 0.02, "static_output_rate": 0.15, "repetition_rate": 0.25, "copy_prompt_rate": 0.20}.items():
        if metrics.get(key, 1.0) > threshold:
            failures.append(f"COLLAPSE_GATE_FAIL:{seed}:{key}")
    mode_comparison = load_json(seed_root / "mode_comparison.json")
    mode_hashes = {item.get("eval_row_hash") for item in mode_comparison.get("modes", [])}
    mode_counts = {item.get("eval_count") for item in mode_comparison.get("modes", [])}
    if mode_comparison.get("all_eval_rows_match") is not True or len(mode_hashes) != 1 or len(mode_counts) != 1:
        failures.append(f"EVAL_ROW_MISMATCH:{seed}")
    return failures


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
    if "MULTI_SEED_RAW_GENERATION_CONFIRM_POSITIVE" not in verdicts:
        failures.append("MULTI_SEED_RAW_GENERATION_CONFIRM_POSITIVE_MISSING")
    aggregate = load_json(SMOKE_ROOT / "multi_seed_aggregate.json")
    seed_manifest = load_json(SMOKE_ROOT / "seed_run_manifest.json")
    seeds = seed_manifest.get("seeds", [])
    if [item.get("seed") for item in seeds] != [2027, 2028, 2029]:
        failures.append("SEED_SET_MISMATCH")
    if aggregate.get("all_seeds_passed_independently") is not True:
        failures.append("MULTI_SEED_RAW_GENERATION_INSTABILITY_DETECTED")
    for key in ["mean_only_pass_rejected", "best_seed_pass_rejected", "two_of_three_pass_rejected"]:
        if aggregate.get(key) is not True:
            failures.append(f"MEAN_OR_BEST_SEED_PASS_NOT_REJECTED:{key}")
    gates = {
        "min_raw_free_generation_accuracy": (0.85, "min"),
        "min_decoder_assisted_accuracy": (0.90, "min"),
        "min_bounded_chat_slot_binding_accuracy": (0.90, "min"),
        "min_finite_label_anchorroute_retention_accuracy": (0.90, "min"),
        "max_case_id_drift_rate": (0.10, "max"),
        "max_slot_drift_rate": (0.05, "max"),
        "max_distractor_leak_rate": (0.10, "max"),
    }
    for key, (threshold, kind) in gates.items():
        value = aggregate.get(key)
        if value is None or (kind == "min" and value < threshold) or (kind == "max" and value > threshold):
            failures.append(f"AGGREGATE_GATE_FAIL:{key}")
    for key in ["stddev_raw_free_generation_accuracy", "stddev_case_id_drift_rate"]:
        if key not in aggregate:
            failures.append(f"AGGREGATE_STDDEV_MISSING:{key}")
    if aggregate.get("checkpoint_hash_unchanged_all_seeds") is not True or aggregate.get("source_100_checkpoint_unchanged_all_seeds") is not True:
        failures.append("CHECKPOINT_MUTATION_DETECTED")
    if aggregate.get("bounded_release_artifact_unchanged_all_seeds") is not True:
        failures.append("BOUNDED_RELEASE_MUTATION_DETECTED")
    if aggregate.get("train_step_count") != 0 or aggregate.get("optimizer_step_count") != 0:
        failures.append("TRAINING_SIDE_EFFECT_DETECTED")
    for seed in seeds:
        failures.extend(check_seed(seed, aggregate))
    decision = load_json(SMOKE_ROOT / "decision_recommendation.json")
    if decision.get("primary_next_milestone") != "105_RAW_GENERATION_OOD_AND_BOUNDARY_CONFIRM" or decision.get("mechanically_derived") is not True:
        failures.append("DECISION_RECOMMENDATION_MISSING")
    if len(read_jsonl(SMOKE_ROOT / "human_readable_samples.jsonl")) < 120:
        failures.append("HUMAN_SAMPLE_REPORT_MISSING")
    failures.extend(find_false_claims((SMOKE_ROOT / "report.md").read_text(encoding="utf-8")))
    for path in changed_paths():
        if path.startswith("target/"):
            failures.append("GENERATED_ARTIFACT_STAGED")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Check STABLE_LOOP_PHASE_LOCK_104 artifacts")
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
    print("STABLE_LOOP_PHASE_LOCK_104_MULTI_SEED_RAW_GENERATION_CONFIRM_CHECK_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
