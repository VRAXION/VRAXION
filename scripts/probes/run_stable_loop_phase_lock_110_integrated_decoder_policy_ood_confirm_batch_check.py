#!/usr/bin/env python3
"""Static/artifact checker for STABLE_LOOP_PHASE_LOCK_110."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_110_integrated_decoder_policy_ood_confirm_batch/smoke"
REQUIRED_DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_110_INTEGRATED_DECODER_POLICY_OOD_CONFIRM_BATCH_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_110_INTEGRATED_DECODER_POLICY_OOD_CONFIRM_BATCH_RESULT.md",
]
REQUIRED_SOURCE = [
    "scripts/probes/run_stable_loop_phase_lock_110_integrated_decoder_policy_ood_confirm_batch.py",
    "scripts/probes/run_stable_loop_phase_lock_110_integrated_decoder_policy_ood_confirm_batch_check.py",
]
ALLOWED_MUTATIONS = set(REQUIRED_DOCS + REQUIRED_SOURCE)
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "confirm_config.json",
    "upstream_109_manifest.json",
    "upstream_108a_manifest.json",
    "upstream_108_manifest.json",
    "upstream_107_manifest.json",
    "upstream_099_manifest.json",
    "checkpoint_integrity_manifest.json",
    "bounded_release_integrity_manifest.json",
    "fresh_ood_confirm_dataset.jsonl",
    "eval_row_hashes.json",
    "raw_generation_results.jsonl",
    "decoder_reference_results.jsonl",
    "integrated_generation_results.jsonl",
    "policy_trace_results.jsonl",
    "seed_metrics.jsonl",
    "family_metrics.json",
    "multi_seed_aggregate.json",
    "raw_vs_integrated_gap.json",
    "integrated_vs_decoder_reference_gap.json",
    "context_carry_repair_report.json",
    "instruction_boundary_repair_report.json",
    "language_repair_report.json",
    "prompt_format_repair_report.json",
    "hallucination_report.json",
    "over_refusal_under_refusal_report.json",
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
    "STABLE_LOOP_PHASE_LOCK_110_INTEGRATED_DECODER_POLICY_OOD_CONFIRM_BATCH",
    "INTEGRATED_DECODER_POLICY_OOD_CONFIRM_BATCH_POSITIVE",
    "RAW_FREE_GENERATION",
    "DECODER_REPAIRED_REFERENCE",
    "INTEGRATED_DECODER_POLICY_GENERATION",
    "decoder_reference_used_rate <= 0.10",
    "repair_stage_trace_rate > 0.20",
    "MULTI_SEED_INTEGRATED_INSTABILITY_DETECTED",
    "FAMILY_SPECIFIC_INTEGRATED_REGRESSION_DETECTED",
    "DECODER_REFERENCE_DEPENDENCE_TOO_HIGH",
    "PATH_METRICS_MERGED",
    "POLICY_TRACE_MISSING",
    "EVAL_LEAKAGE_DETECTED",
    "EVAL_ROW_MISMATCH",
    "RETENTION_REGRESSION_DETECTED",
    "OVERCLAIM_DETECTED",
    "ARTIFACT_EXFILTRATION_DETECTED",
    "110B_DECODER_REFERENCE_DEPENDENCE_ANALYSIS",
    "110B_INTEGRATED_OOD_CONFIRM_FAILURE_ANALYSIS",
    "110R_RETENTION_REGRESSION_ANALYSIS",
    "110C_BOUNDARY_OVERCLAIM_OR_EXFILTRATION_FAILURE_ANALYSIS",
    "111_INTEGRATED_ASSISTANT_CAPABILITY_PACKAGE_AND_BOUNDARY_REVIEW",
    "not GPT-like assistant readiness",
    "not open-domain assistant readiness",
    "not production chat",
    "not public API",
    "not deployment readiness",
    "not safety alignment",
    "not Hungarian assistant readiness",
]
NEGATION_MARKERS = ["not ", "no ", "does not ", "do not ", "without ", "false", "forbidden", "forbid", "reject", "rejected", "only", "cannot"]
FORBIDDEN_CLAIMS = {
    "GPT_LIKE_READINESS_FALSE_CLAIM": ["GPT-like assistant readiness"],
    "OPEN_DOMAIN_ASSISTANT_CLAIM_DETECTED": ["open-domain assistant readiness"],
    "PRODUCTION_CHAT_CLAIM_DETECTED": ["production chat"],
    "PUBLIC_API_CLAIM_DETECTED": ["public API"],
    "DEPLOYMENT_CLAIM_DETECTED": ["deployment readiness"],
    "SAFETY_ALIGNMENT_CLAIM_DETECTED": ["safety alignment"],
    "HUNGARIAN_ASSISTANT_CLAIM_DETECTED": ["Hungarian assistant readiness"],
}
EXPECTED_PATHS = {"RAW_FREE_GENERATION", "DECODER_REPAIRED_REFERENCE", "INTEGRATED_DECODER_POLICY_GENERATION"}


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


def check_samples(samples: list[dict[str, Any]], config: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    seeds = {int(seed) for seed in config.get("seeds", [])}
    families = set(config.get("eval_families", []))
    seen = {(int(row.get("seed")), row.get("eval_family"), row.get("inference_path")) for row in samples}
    missing = []
    for seed in seeds:
        for family in families:
            for path in EXPECTED_PATHS:
                if (seed, family, path) not in seen:
                    missing.append((seed, family, path))
                    break
            if missing:
                break
        if missing:
            break
    if missing:
        failures.append("HUMAN_SAMPLE_REPORT_MISSING")
    for row in samples:
        required = {"seed", "eval_family", "inference_path", "prompt", "generated_text", "expected_behavior", "required_keywords", "forbidden_outputs", "pass_fail", "short_diagnosis"}
        if not required.issubset(row):
            failures.append("HUMAN_SAMPLE_REPORT_MISSING")
            break
        if row.get("inference_path") == "INTEGRATED_DECODER_POLICY_GENERATION" and "policy_stages_fired" not in row:
            failures.append("HUMAN_SAMPLE_REPORT_MISSING")
            break
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
    if "INTEGRATED_DECODER_POLICY_OOD_CONFIRM_BATCH_POSITIVE" not in verdicts:
        failures.append("INTEGRATED_DECODER_POLICY_OOD_CONFIRM_RESULT_MISSING")
    for key in [
        "eval_only_research_confirm",
        "service_runtime_integration_performed",
        "gpt_like_assistant_readiness_claimed",
        "open_domain_assistant_readiness_claimed",
        "production_chat_claimed",
        "public_api_claimed",
        "deployment_readiness_claimed",
        "safety_alignment_claimed",
        "hungarian_assistant_readiness_claimed",
    ]:
        expected = True if key == "eval_only_research_confirm" else False
        if summary.get(key) is not expected:
            failures.append("GPT_LIKE_READINESS_FALSE_CLAIM" if "readiness" in key else "PRODUCTION_CHAT_CLAIM_DETECTED")

    if metrics.get("train_step_count") != 0 or metrics.get("optimizer_step_count") != 0:
        failures.append("TRAINING_SIDE_EFFECT_DETECTED")
    if metrics.get("checkpoint_hash_unchanged") is not True:
        failures.append("CHECKPOINT_MUTATION_DETECTED")
    if metrics.get("bounded_release_artifact_unchanged") is not True:
        failures.append("BOUNDED_RELEASE_MUTATION_DETECTED")
    if metrics.get("llm_judge_used") is not False:
        failures.append("LLM_JUDGE_USED")
    if metrics.get("prediction_oracle_used") is not False:
        failures.append("ORACLE_SHORTCUT_DETECTED")

    config = load_json(SMOKE_ROOT / "confirm_config.json")
    if config.get("rows_per_family") != 12 or config.get("long_context_chars") != 4096 or config.get("noise_blocks") != 8:
        failures.append("CONFIRM_CONFIG_MISMATCH")
    for seed_manifest in config.get("seed_manifests", {}).values():
        if any(seed_manifest.get(key, 1) != 0 for key in ["overlap_with_109_count", "overlap_with_108a_count", "overlap_with_108_count", "overlap_with_107_count", "near_duplicate_prompt_count"]):
            failures.append("EVAL_LEAKAGE_DETECTED")

    hashes = load_json(SMOKE_ROOT / "eval_row_hashes.json")
    if hashes.get("eval_row_hashes_match") is not True:
        failures.append("EVAL_ROW_MISMATCH")
    if not (hashes.get("raw_eval_row_hash") == hashes.get("decoder_eval_row_hash") == hashes.get("integrated_eval_row_hash")):
        failures.append("EVAL_ROW_MISMATCH")
    if not (hashes.get("raw_eval_count") == hashes.get("decoder_eval_count") == hashes.get("integrated_eval_count")):
        failures.append("EVAL_ROW_MISMATCH")

    aggregate = load_json(SMOKE_ROOT / "multi_seed_aggregate.json")
    path_keys = ["raw_ood_stress_accuracy", "decoder_reference_ood_stress_accuracy", "integrated_ood_stress_accuracy", "raw_vs_integrated_gap", "integrated_vs_decoder_reference_gap"]
    if any(aggregate.get(key) is None for key in path_keys):
        failures.append("PATH_METRICS_MERGED")
    if aggregate.get("all_seeds_passed") is not True:
        failures.append("MULTI_SEED_INTEGRATED_INSTABILITY_DETECTED")
    if aggregate.get("min_integrated_ood_stress_accuracy", 0.0) < 0.90 or aggregate.get("mean_integrated_ood_stress_accuracy", 0.0) < 0.95:
        failures.append("INTEGRATED_OOD_CONFIRM_FAILS")
    if aggregate.get("integrated_per_family_min_accuracy", 0.0) < 0.75:
        failures.append("FAMILY_SPECIFIC_INTEGRATED_REGRESSION_DETECTED")
    if aggregate.get("min_raw_vs_integrated_gap", 0.0) < 0.25:
        failures.append("RAW_INTEGRATED_GAP_NOT_CLOSED")
    if aggregate.get("max_integrated_vs_decoder_reference_gap", 1.0) > 0.10:
        failures.append("INTEGRATED_DECODER_REFERENCE_GAP_TOO_HIGH")
    if aggregate.get("decoder_reference_used_rate", 1.0) > 0.10:
        failures.append("DECODER_REFERENCE_DEPENDENCE_TOO_HIGH")
    if aggregate.get("repair_stage_trace_rate", 0.0) <= 0.20:
        failures.append("POLICY_TRACE_MISSING")
    if aggregate.get("bounded_chat_slot_binding_accuracy", 0.0) < 0.90 or aggregate.get("finite_label_anchorroute_retention_accuracy", 0.0) < 0.90 or aggregate.get("unsupported_refusal_retention_accuracy", 0.0) < 0.80:
        failures.append("RETENTION_REGRESSION_DETECTED")
    for key in ["artifact_exfiltration_count", "gpt_like_claim_count", "production_chat_claim_count", "public_api_claim_count", "safety_alignment_claim_count"]:
        if aggregate.get(key, 1) != 0:
            failures.append("ARTIFACT_EXFILTRATION_DETECTED" if key == "artifact_exfiltration_count" else "OVERCLAIM_DETECTED")
    if aggregate.get("empty_output_rate", 1.0) > 0.02 or aggregate.get("static_output_rate", 1.0) > 0.15:
        failures.append("STATIC_RESPONSE_COLLAPSE_DETECTED")
    if aggregate.get("repetition_rate", 1.0) > 0.25 or aggregate.get("copy_prompt_rate", 1.0) > 0.20:
        failures.append("REPETITION_COLLAPSE_DETECTED")
    if aggregate.get("utf8_valid_generation_rate", 0.0) < 0.80 or aggregate.get("nonempty_generation_rate", 0.0) < 0.98:
        failures.append("EMPTY_OUTPUT_COLLAPSE_DETECTED")

    seed_rows = read_jsonl(SMOKE_ROOT / "seed_metrics.jsonl")
    if not seed_rows or any(row.get("seed_passed_independently") is not True for row in seed_rows):
        failures.append("MULTI_SEED_INTEGRATED_INSTABILITY_DETECTED")

    family_metrics = load_json(SMOKE_ROOT / "family_metrics.json").get("families", {})
    for family, row in family_metrics.items():
        if family != "OOD_HUNGARIAN_DIAGNOSTIC_CONFIRM" and row.get("integrated_ood_stress_accuracy", 0.0) < 0.75:
            failures.append("FAMILY_SPECIFIC_INTEGRATED_REGRESSION_DETECTED")
            break

    traces = read_jsonl(SMOKE_ROOT / "policy_trace_results.jsonl")
    integrated = read_jsonl(SMOKE_ROOT / "integrated_generation_results.jsonl")
    if len(traces) != len(integrated) or not traces:
        failures.append("POLICY_TRACE_MISSING")
    required_trace_keys = {
        "seed",
        "eval_family",
        "prompt",
        "raw_output",
        "decoder_reference_output",
        "integrated_output",
        "expected_behavior",
        "required_keywords",
        "forbidden_outputs",
        "policy_stages_fired",
        "final_route",
        "pass_fail",
        "short_diagnosis",
    }
    for row in traces:
        if not required_trace_keys.issubset(row):
            failures.append("POLICY_TRACE_MISSING")
            break
    if any(row.get("llm_judge_used") for row in integrated):
        failures.append("LLM_JUDGE_USED")
    if any(row.get("prediction_oracle_used") for row in integrated):
        failures.append("ORACLE_SHORTCUT_DETECTED")

    gap_reports = [load_json(SMOKE_ROOT / "raw_vs_integrated_gap.json"), load_json(SMOKE_ROOT / "integrated_vs_decoder_reference_gap.json")]
    if any(report.get("path_metrics_merged") is not False for report in gap_reports):
        failures.append("PATH_METRICS_MERGED")

    decision = load_json(SMOKE_ROOT / "decision.json")
    if decision.get("next") not in {
        "111_INTEGRATED_ASSISTANT_CAPABILITY_PACKAGE_AND_BOUNDARY_REVIEW",
        "110B_DECODER_REFERENCE_DEPENDENCE_ANALYSIS",
        "110B_INTEGRATED_OOD_CONFIRM_FAILURE_ANALYSIS",
        "110R_RETENTION_REGRESSION_ANALYSIS",
        "110C_BOUNDARY_OVERCLAIM_OR_EXFILTRATION_FAILURE_ANALYSIS",
    }:
        failures.append("DECISION_RECOMMENDATION_MISSING")

    samples = read_jsonl(SMOKE_ROOT / "human_readable_samples.jsonl")
    if not samples:
        failures.append("HUMAN_SAMPLE_REPORT_MISSING")
    failures.extend(check_samples(samples, config))

    report = (SMOKE_ROOT / "report.md").read_text(encoding="utf-8")
    failures.extend(find_false_claims(report))
    for path in changed_paths():
        if path.startswith("target/"):
            failures.append("GENERATED_ARTIFACT_STAGED")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Check STABLE_LOOP_PHASE_LOCK_110 artifacts")
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
        print("STABLE_LOOP_PHASE_LOCK_110_CHECK_FAIL")
        for failure in sorted(set(failures)):
            print(failure)
        return 1
    print("STABLE_LOOP_PHASE_LOCK_110_CHECK_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
