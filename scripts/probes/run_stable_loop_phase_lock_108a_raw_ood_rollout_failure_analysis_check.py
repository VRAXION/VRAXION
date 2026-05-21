#!/usr/bin/env python3
"""Static/artifact checker for STABLE_LOOP_PHASE_LOCK_108A."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_108a_raw_ood_rollout_failure_analysis/smoke"
REQUIRED_DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_108A_RAW_OOD_ROLLOUT_FAILURE_ANALYSIS_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_108A_RAW_OOD_ROLLOUT_FAILURE_ANALYSIS_RESULT.md",
]
REQUIRED_SOURCE = [
    "scripts/probes/run_stable_loop_phase_lock_108a_raw_ood_rollout_failure_analysis.py",
    "scripts/probes/run_stable_loop_phase_lock_108a_raw_ood_rollout_failure_analysis_check.py",
]
ALLOWED_MUTATIONS = set(REQUIRED_DOCS + REQUIRED_SOURCE)
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "analysis_config.json",
    "upstream_108_manifest.json",
    "raw_decoder_pair_manifest.json",
    "raw_failure_attribution.json",
    "raw_failure_cases.jsonl",
    "raw_decoder_disagreement.jsonl",
    "first_error_position_report.json",
    "prefix_survival_report.json",
    "rollout_drift_report.json",
    "stop_condition_report.json",
    "family_failure_breakdown.json",
    "recommended_repair_plan.json",
    "human_readable_samples.jsonl",
    "summary.json",
    "report.md",
]
REQUIRED_TERMS = [
    "STABLE_LOOP_PHASE_LOCK_108A_RAW_OOD_ROLLOUT_FAILURE_ANALYSIS",
    "analysis-only",
    "RAW_OOD_ROLLOUT_FAILURE_ANALYSIS_POSITIVE",
    "RAW_DECODER_PAIRING_FAILS",
    "EXPECTED_RESPONSE_MISSING",
    "UNKNOWN_RAW_FAILURE_RATE_TOO_HIGH",
    "RAW_FAILURE_ATTRIBUTION_INCOMPLETE",
    "REPAIR_PLAN_MISSING",
    "109_DECODER_POLICY_INTEGRATION",
    "109_RAW_ROLLOUT_REPAIR",
    "109_SFT_ROLLOUT_DATA_REPAIR",
    "not GPT-like assistant readiness",
    "not open-domain assistant readiness",
    "not production chat",
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
    "109_DECODER_POLICY_INTEGRATION",
    "109_RAW_ROLLOUT_REPAIR",
    "109_SFT_ROLLOUT_DATA_REPAIR",
    "109_STOP_CONDITION_REPAIR",
    "109_PROMPT_FORMAT_REPAIR",
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
    if "RAW_OOD_ROLLOUT_FAILURE_ANALYSIS_POSITIVE" not in verdicts:
        failures.append("RAW_OOD_ROLLOUT_FAILURE_ANALYSIS_RESULT_MISSING")
    if metrics.get("train_step_count") != 0 or metrics.get("optimizer_step_count") != 0:
        failures.append("TRAINING_SIDE_EFFECT_DETECTED")
    if metrics.get("checkpoint_hash_unchanged") is not True:
        failures.append("CHECKPOINT_MUTATION_DETECTED")
    if metrics.get("bounded_release_artifact_unchanged") is not True:
        failures.append("CHECKPOINT_MUTATION_DETECTED")
    pair = load_json(SMOKE_ROOT / "raw_decoder_pair_manifest.json")
    if pair.get("raw_pair_count") != pair.get("decoder_pair_count") or pair.get("paired_row_count", 0) <= 0:
        failures.append("RAW_DECODER_PAIRING_FAILS")
    if pair.get("raw_decoder_disagreement_count", 0) <= 0:
        failures.append("RAW_DECODER_PAIRING_FAILS")
    attribution = load_json(SMOKE_ROOT / "raw_failure_attribution.json")
    if attribution.get("all_raw_fail_decoder_pass_rows_attributed") is not True:
        failures.append("RAW_FAILURE_ATTRIBUTION_INCOMPLETE")
    if attribution.get("unknown_raw_failure_rate", 1.0) > 0.10:
        failures.append("UNKNOWN_RAW_FAILURE_RATE_TOO_HIGH")
    cases = read_jsonl(SMOKE_ROOT / "raw_failure_cases.jsonl")
    if len(cases) != pair.get("raw_decoder_disagreement_count"):
        failures.append("RAW_FAILURE_ATTRIBUTION_INCOMPLETE")
    for row in cases:
        for key in ["expected_response_source", "expected_response", "decoder_output", "raw_output", "primary_surface_failure_label", "likely_mechanism_label", "first_wrong_token_position", "gold_prefix_survival_rate"]:
            if key not in row:
                failures.append("EXPECTED_RESPONSE_MISSING" if key.startswith("expected") else "RAW_FAILURE_ATTRIBUTION_INCOMPLETE")
                break
        if row.get("llm_judge_used") is not False:
            failures.append("LLM_JUDGE_USED")
        if row.get("prediction_oracle_used") is not False:
            failures.append("ORACLE_SHORTCUT_DETECTED")
    first_error = load_json(SMOKE_ROOT / "first_error_position_report.json")
    if "first_wrong_token_position_mean" not in first_error or "first_wrong_token_position_median" not in first_error:
        failures.append("FIRST_ERROR_REPORT_MISSING")
    prefix = load_json(SMOKE_ROOT / "prefix_survival_report.json")
    if "gold_prefix_survival_rate_mean" not in prefix or "gold_prefix_survival_rate_min" not in prefix:
        failures.append("PREFIX_SURVIVAL_REPORT_MISSING")
    repair = load_json(SMOKE_ROOT / "recommended_repair_plan.json")
    if repair.get("next") not in ALLOWED_NEXT:
        failures.append("REPAIR_PLAN_MISSING")
    for key in ["secondary_next_if_decoder_integration_fails", "primary_failure_mechanism", "secondary_failure_mechanisms", "evidence_counts", "evidence_rates", "recommended_scope_for_109"]:
        if key not in repair:
            failures.append(f"REPAIR_PLAN_FIELD_MISSING:{key}")
    samples = read_jsonl(SMOKE_ROOT / "human_readable_samples.jsonl")
    if not samples:
        failures.append("HUMAN_SAMPLE_REPORT_MISSING")
    sample_families = {row.get("eval_family") for row in samples}
    for family in ["OOD_LONG_NOISY_CONTEXT", "OOD_PROVIDED_FACT_DISTRACTOR_TRAP", "OOD_MULTI_TURN_STALE_OVERRIDE", "OOD_ADVERSARIAL_FORMATTING"]:
        if family not in sample_families:
            failures.append(f"HUMAN_SAMPLE_REPORT_MISSING:{family}")
    report = (SMOKE_ROOT / "report.md").read_text(encoding="utf-8")
    failures.extend(find_false_claims(report))
    for path in changed_paths():
        if path.startswith("target/"):
            failures.append("GENERATED_ARTIFACT_STAGED")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Check STABLE_LOOP_PHASE_LOCK_108A artifacts")
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
        print("STABLE_LOOP_PHASE_LOCK_108A_CHECK_FAIL")
        for failure in sorted(set(failures)):
            print(failure)
        return 1
    print("STABLE_LOOP_PHASE_LOCK_108A_CHECK_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
