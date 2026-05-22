#!/usr/bin/env python3
"""Static/artifact checker for STABLE_LOOP_PHASE_LOCK_111R."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_111r_retention_or_lm_regression_analysis/smoke"
REQUIRED_DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_111R_RETENTION_OR_LM_REGRESSION_ANALYSIS_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_111R_RETENTION_OR_LM_REGRESSION_ANALYSIS_RESULT.md",
]
REQUIRED_SOURCE = [
    "scripts/probes/run_stable_loop_phase_lock_111r_retention_or_lm_regression_analysis.py",
    "scripts/probes/run_stable_loop_phase_lock_111r_retention_or_lm_regression_analysis_check.py",
]
ALLOWED_MUTATIONS = set(REQUIRED_DOCS + REQUIRED_SOURCE)
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "analysis_config.json",
    "upstream_111_manifest.json",
    "upstream_110_manifest.json",
    "upstream_109_manifest.json",
    "upstream_108a_manifest.json",
    "eval_path_compatibility_report.json",
    "namespace_leakage_report.json",
    "rollout_gap_report.json",
    "retention_regression_report.json",
    "data_balance_report.json",
    "output_collapse_report.json",
    "root_cause_classification.json",
    "recommended_next_plan.json",
    "human_readable_failure_samples.jsonl",
    "summary.json",
    "report.md",
]
REQUIRED_TERMS = [
    "STABLE_LOOP_PHASE_LOCK_111R_RETENTION_OR_LM_REGRESSION_ANALYSIS",
    "analysis only",
    "RETENTION_OR_LM_REGRESSION_ANALYSIS_POSITIVE",
    "UPSTREAM_111_FAILURE_VERIFIED",
    "EVAL_PATH_COMPATIBILITY_ANALYZED",
    "NAMESPACE_LEAKAGE_ANALYZED",
    "ROLLOUT_GAP_ANALYZED",
    "RETENTION_REGRESSION_ANALYZED",
    "DATA_BALANCE_ANALYZED",
    "ROOT_CAUSE_CLASSIFIED",
    "RECOMMENDED_NEXT_PLAN_WRITTEN",
    "NO_TRAINING_PERFORMED",
    "CHECKPOINTS_UNCHANGED",
    "111X_COMBINED_RAW_DISTILLATION_REDESIGN",
    "111E_EVAL_PATH_COMPATIBILITY_FIX",
    "111N_NAMESPACE_RANDOMIZATION_AND_ANTI_MEMORIZATION_REPAIR",
    "111G_SCHEDULED_SAMPLING_OR_ROLLOUT_LOSS_REPAIR",
    "111M_RETENTION_MIX_REBALANCE_REPAIR",
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
ALLOWED_NEXT = {
    "111E_EVAL_PATH_COMPATIBILITY_FIX",
    "111N_NAMESPACE_RANDOMIZATION_AND_ANTI_MEMORIZATION_REPAIR",
    "111G_SCHEDULED_SAMPLING_OR_ROLLOUT_LOSS_REPAIR",
    "111M_RETENTION_MIX_REBALANCE_REPAIR",
    "111X_COMBINED_RAW_DISTILLATION_REDESIGN",
}
ALLOWED_ROOT_CAUSES = {
    "EVAL_PATH_MISMATCH",
    "NAMESPACE_MEMORIZATION",
    "TEACHER_FORCING_ROLLOUT_GAP",
    "RETENTION_MIX_UNDERPOWERED",
    "DATA_BALANCE_FAILURE",
    "STOP_CONDITION_FAILURE",
    "TARGET_CHECKPOINT_COLLAPSE",
    "SCORER_FORMAT_MISMATCH",
    "MIXED_CAUSE",
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
    if "RETENTION_OR_LM_REGRESSION_ANALYSIS_POSITIVE" not in verdicts:
        failures.append("RETENTION_OR_LM_REGRESSION_ANALYSIS_RESULT_MISSING")
    if metrics.get("train_step_count") != 0 or metrics.get("optimizer_step_count") != 0:
        failures.append("TRAINING_SIDE_EFFECT_DETECTED")
    if metrics.get("checkpoints_unchanged") is not True:
        failures.append("CHECKPOINT_MUTATION_DETECTED")
    if metrics.get("primary_root_cause") not in ALLOWED_ROOT_CAUSES:
        failures.append("ROOT_CAUSE_CLASSIFICATION_MISSING")
    if metrics.get("recommended_next") not in ALLOWED_NEXT:
        failures.append("RECOMMENDED_NEXT_PLAN_MISSING")
    upstream111 = load_json(SMOKE_ROOT / "upstream_111_manifest.json")
    upstream_summary = upstream111.get("summary", {})
    upstream_verdicts = set(upstream_summary.get("verdicts", []))
    if upstream_summary.get("status") != "failed" or "RAW_OOD_ACCURACY_NOT_IMPROVED" not in upstream_verdicts:
        failures.append("UPSTREAM_111_FAILURE_NOT_FOUND")
    eval_path = load_json(SMOKE_ROOT / "eval_path_compatibility_report.json")
    if eval_path.get("pre_111_raw_ood_accuracy") != 0.0 or eval_path.get("prior_raw_ood_accuracy", 0.0) < 0.50:
        failures.append("EVAL_PATH_ANALYSIS_MISSING")
    namespace = load_json(SMOKE_ROOT / "namespace_leakage_report.json")
    for key in ["prompt_namespace_prefixes", "teacher_train_namespace_prefixes", "generated_namespace_prefixes", "namespace_leak_rate", "teacher_namespace_copy_rate", "case_id_drift_rate"]:
        if key not in namespace:
            failures.append("NAMESPACE_LEAKAGE_ANALYSIS_MISSING")
    rollout = load_json(SMOKE_ROOT / "rollout_gap_report.json")
    for key in ["teacher_forced_loss_final", "rollout_accuracy", "rollout_drift_rate", "first_wrong_token_position_mean", "prefix_survival_rate_mean", "repetition_rate", "static_output_rate"]:
        if key not in rollout:
            failures.append("ROLLOUT_GAP_ANALYSIS_MISSING")
    retention = load_json(SMOKE_ROOT / "retention_regression_report.json")
    for key in ["retention_rows_in_train", "retention_rows_in_eval", "retention_label_format", "retention_generated_outputs", "retention_failure_type_counts"]:
        if key not in retention:
            failures.append("RETENTION_ANALYSIS_MISSING")
    data_balance = load_json(SMOKE_ROOT / "data_balance_report.json")
    for key in ["teacher_distill_percentage_actual", "fineweb_replay_percentage_actual", "bounded_retention_percentage_actual", "finite_label_retention_percentage_actual", "refusal_boundary_percentage_actual"]:
        if key not in data_balance:
            failures.append("DATA_BALANCE_ANALYSIS_MISSING")
    root = load_json(SMOKE_ROOT / "root_cause_classification.json")
    if root.get("primary_root_cause") not in ALLOWED_ROOT_CAUSES or not root.get("secondary_root_causes"):
        failures.append("ROOT_CAUSE_CLASSIFICATION_MISSING")
    recommended = load_json(SMOKE_ROOT / "recommended_next_plan.json")
    for key in ["next", "primary_root_cause", "secondary_root_causes", "evidence_counts", "evidence_rates", "why_not_more_steps", "why_not_repeat_111_standard"]:
        if key not in recommended:
            failures.append("RECOMMENDED_NEXT_PLAN_MISSING")
    if recommended.get("next") == "MORE_STEPS":
        failures.append("MORE_STEPS_MISCLASSIFICATION")
    samples = read_jsonl(SMOKE_ROOT / "human_readable_failure_samples.jsonl")
    required_sample_types = {"911 prompt -> 711 output", "retention failure", "repetition/static output", "pre-baseline mismatch", "rollout drift"}
    if not required_sample_types.issubset({row.get("sample_type") for row in samples}):
        failures.append("HUMAN_SAMPLE_REPORT_MISSING")
    combined_reports = "\n".join((SMOKE_ROOT / rel).read_text(encoding="utf-8") for rel in ["summary.json", "report.md", "recommended_next_plan.json"])
    failures.extend(find_false_claims(combined_reports))
    for path in changed_paths():
        if path.startswith("target/"):
            failures.append("GENERATED_ARTIFACT_STAGED")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Check STABLE_LOOP_PHASE_LOCK_111R artifacts")
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
        print("STABLE_LOOP_PHASE_LOCK_111R_CHECK_FAIL")
        for failure in sorted(set(failures)):
            print(failure)
        return 1
    print("STABLE_LOOP_PHASE_LOCK_111R_CHECK_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
