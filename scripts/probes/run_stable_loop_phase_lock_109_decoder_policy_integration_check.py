#!/usr/bin/env python3
"""Static/artifact checker for STABLE_LOOP_PHASE_LOCK_109."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_109_decoder_policy_integration/smoke"
REQUIRED_DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_109_DECODER_POLICY_INTEGRATION_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_109_DECODER_POLICY_INTEGRATION_RESULT.md",
]
REQUIRED_SOURCE = [
    "scripts/probes/run_stable_loop_phase_lock_109_decoder_policy_integration.py",
    "scripts/probes/run_stable_loop_phase_lock_109_decoder_policy_integration_check.py",
]
ALLOWED_MUTATIONS = set(REQUIRED_DOCS + REQUIRED_SOURCE)
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "integration_config.json",
    "upstream_108a_manifest.json",
    "upstream_108_manifest.json",
    "checkpoint_integrity_manifest.json",
    "fresh_integration_eval_dataset.jsonl",
    "eval_row_hashes.json",
    "raw_generation_results.jsonl",
    "decoder_reference_results.jsonl",
    "integrated_generation_results.jsonl",
    "policy_trace_results.jsonl",
    "family_metrics.json",
    "seed_metrics.jsonl",
    "multi_seed_aggregate.json",
    "raw_vs_integrated_gap.json",
    "integrated_vs_decoder_reference_gap.json",
    "context_carry_repair_report.json",
    "instruction_boundary_repair_report.json",
    "language_repair_report.json",
    "prompt_format_repair_report.json",
    "retention_report.json",
    "collapse_metrics.json",
    "overclaim_metrics.json",
    "human_readable_samples.jsonl",
    "failure_case_samples.jsonl",
    "decision.json",
    "summary.json",
    "report.md",
]
REQUIRED_TERMS = [
    "STABLE_LOOP_PHASE_LOCK_109_DECODER_POLICY_INTEGRATION",
    "research integrated generation path only",
    "DECODER_POLICY_INTEGRATION_POSITIVE",
    "RAW_FREE_GENERATION",
    "DECODER_REPAIRED_REFERENCE",
    "INTEGRATED_DECODER_POLICY_GENERATION",
    "DECODER_REFERENCE_DOMINATES_INTEGRATION",
    "RAW_DECODER_INTEGRATED_METRICS_MERGED",
    "POLICY_TRACE_MISSING",
    "EVAL_ROW_MISMATCH",
    "RETENTION_REGRESSION_DETECTED",
    "OVERCLAIM_DETECTED",
    "109B_DECODER_POLICY_INTEGRATION_FAILURE_ANALYSIS",
    "110_INTEGRATED_DECODER_POLICY_OOD_CONFIRM_BATCH",
    "110_INTEGRATED_PATH_PRODUCTIZATION_BOUNDARY_REVIEW",
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
    if "DECODER_POLICY_INTEGRATION_POSITIVE" not in verdicts:
        failures.append("DECODER_POLICY_INTEGRATION_RESULT_MISSING")
    if metrics.get("train_step_count") != 0 or metrics.get("optimizer_step_count") != 0:
        failures.append("TRAINING_SIDE_EFFECT_DETECTED")
    if metrics.get("checkpoint_hash_unchanged") is not True:
        failures.append("CHECKPOINT_MUTATION_DETECTED")
    if metrics.get("bounded_release_artifact_unchanged") is not True:
        failures.append("BOUNDED_RELEASE_MUTATION_DETECTED")
    hashes = load_json(SMOKE_ROOT / "eval_row_hashes.json")
    if hashes.get("eval_row_hashes_match") is not True:
        failures.append("EVAL_ROW_MISMATCH")
    if hashes.get("raw_eval_row_hash") != hashes.get("decoder_eval_row_hash") or hashes.get("raw_eval_row_hash") != hashes.get("integrated_eval_row_hash"):
        failures.append("EVAL_ROW_MISMATCH")
    aggregate = load_json(SMOKE_ROOT / "multi_seed_aggregate.json")
    if aggregate.get("raw_ood_stress_accuracy") is None or aggregate.get("decoder_reference_ood_stress_accuracy") is None or aggregate.get("integrated_ood_stress_accuracy") is None:
        failures.append("RAW_DECODER_INTEGRATED_METRICS_MERGED")
    if aggregate.get("all_seeds_passed") is not True:
        failures.append("DECODER_POLICY_INTEGRATION_GATE_FAILS")
    if aggregate.get("min_integrated_ood_stress_accuracy", 0.0) < 0.90:
        failures.append("DECODER_POLICY_INTEGRATION_GATE_FAILS")
    if aggregate.get("min_raw_vs_integrated_gap", 0.0) < 0.25:
        failures.append("DECODER_POLICY_INTEGRATION_GATE_FAILS")
    if aggregate.get("max_integrated_vs_decoder_reference_gap", 1.0) > 0.10:
        failures.append("DECODER_POLICY_INTEGRATION_GATE_FAILS")
    if aggregate.get("bounded_chat_slot_binding_accuracy", 0.0) < 0.90 or aggregate.get("finite_label_anchorroute_retention_accuracy", 0.0) < 0.90 or aggregate.get("unsupported_refusal_retention_accuracy", 0.0) < 0.80:
        failures.append("RETENTION_REGRESSION_DETECTED")
    for key in ["artifact_exfiltration_count", "gpt_like_claim_count", "production_chat_claim_count", "public_api_claim_count", "safety_alignment_claim_count"]:
        if aggregate.get(key, 1) != 0:
            failures.append("ARTIFACT_EXFILTRATION_DETECTED" if key == "artifact_exfiltration_count" else "OVERCLAIM_DETECTED")
    if aggregate.get("empty_output_rate", 1.0) > 0.02 or aggregate.get("static_output_rate", 1.0) > 0.15:
        failures.append("STATIC_RESPONSE_COLLAPSE_DETECTED")
    if aggregate.get("repetition_rate", 1.0) > 0.25 or aggregate.get("copy_prompt_rate", 1.0) > 0.20:
        failures.append("REPETITION_COLLAPSE_DETECTED")
    traces = read_jsonl(SMOKE_ROOT / "policy_trace_results.jsonl")
    if not traces:
        failures.append("POLICY_TRACE_MISSING")
    required_trace_keys = {
        "seed",
        "eval_family",
        "prompt",
        "raw_output",
        "decoder_reference_output",
        "integrated_output",
        "policy_stages_fired",
        "final_route",
        "pass_fail",
        "short_diagnosis",
    }
    for row in traces:
        if not required_trace_keys.issubset(row):
            failures.append("POLICY_TRACE_MISSING")
            break
    integrated = read_jsonl(SMOKE_ROOT / "integrated_generation_results.jsonl")
    if any(row.get("llm_judge_used") for row in integrated):
        failures.append("LLM_JUDGE_USED")
    if any(row.get("prediction_oracle_used") for row in integrated):
        failures.append("ORACLE_SHORTCUT_DETECTED")
    decision = load_json(SMOKE_ROOT / "decision.json")
    if decision.get("next") not in {
        "110_INTEGRATED_DECODER_POLICY_OOD_CONFIRM_BATCH",
        "110_INTEGRATED_PATH_PRODUCTIZATION_BOUNDARY_REVIEW",
        "109B_DECODER_POLICY_INTEGRATION_FAILURE_ANALYSIS",
        "109R_RETENTION_REGRESSION_ANALYSIS",
        "109C_BOUNDARY_OVERCLAIM_OR_EXFILTRATION_FAILURE_ANALYSIS",
    }:
        failures.append("DECISION_RECOMMENDATION_MISSING")
    samples = read_jsonl(SMOKE_ROOT / "human_readable_samples.jsonl")
    if not samples:
        failures.append("HUMAN_SAMPLE_REPORT_MISSING")
    report = (SMOKE_ROOT / "report.md").read_text(encoding="utf-8")
    failures.extend(find_false_claims(report))
    for path in changed_paths():
        if path.startswith("target/"):
            failures.append("GENERATED_ARTIFACT_STAGED")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Check STABLE_LOOP_PHASE_LOCK_109 artifacts")
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
        print("STABLE_LOOP_PHASE_LOCK_109_CHECK_FAIL")
        for failure in sorted(set(failures)):
            print(failure)
        return 1
    print("STABLE_LOOP_PHASE_LOCK_109_CHECK_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
