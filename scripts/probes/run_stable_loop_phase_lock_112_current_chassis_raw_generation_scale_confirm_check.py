#!/usr/bin/env python3
"""Static/artifact checker for STABLE_LOOP_PHASE_LOCK_112."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_112_current_chassis_raw_generation_scale_confirm/smoke"
REQUIRED_DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_112_CURRENT_CHASSIS_RAW_GENERATION_SCALE_CONFIRM_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_112_CURRENT_CHASSIS_RAW_GENERATION_SCALE_CONFIRM_RESULT.md",
]
REQUIRED_SOURCE = [
    "scripts/probes/run_stable_loop_phase_lock_112_current_chassis_raw_generation_scale_confirm.py",
    "scripts/probes/run_stable_loop_phase_lock_112_current_chassis_raw_generation_scale_confirm_check.py",
]
ALLOWED_MUTATIONS = set(REQUIRED_DOCS + REQUIRED_SOURCE)
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "scale_config.json",
    "upstream_111x_manifest.json",
    "upstream_111r_manifest.json",
    "upstream_110_manifest.json",
    "upstream_100_manifest.json",
    "upstream_099_manifest.json",
    "train_dataset_manifest.json",
    "eval_dataset_manifest.json",
    "namespace_audit.json",
    "arm_training_metrics.jsonl",
    "arm_eval_results.jsonl",
    "arm_comparison.json",
    "transformer_fairness_report.json",
    "scale_aggregate.json",
    "retention_report.json",
    "collapse_metrics.json",
    "fineweb_retention_report.json",
    "overclaim_metrics.json",
    "checkpoint_integrity_manifest.json",
    "bounded_release_integrity_manifest.json",
    "human_readable_samples.jsonl",
    "failure_case_samples.jsonl",
    "decision.json",
    "summary.json",
    "report.md",
]
REQUIRED_TERMS = [
    "STABLE_LOOP_PHASE_LOCK_112_CURRENT_CHASSIS_RAW_GENERATION_SCALE_CONFIRM",
    "scale-confirm research gate",
    "CURRENT_CHASSIS_RAW_GENERATION_SCALE_CONFIRM_POSITIVE",
    "UPSTREAM_111X_CHASSIS_DECISION_VERIFIED",
    "RAW_OBJECTIVE_REDESIGN_SCALES",
    "NAMESPACE_MEMORIZATION_REJECTED",
    "RETENTION_PASSES_ALL_SEEDS",
    "COLLAPSE_REJECTED_ALL_SEEDS",
    "FINEWEB_RETENTION_WITHIN_LIMITS",
    "TRANSFORMER_BASELINE_RECORDED",
    "current_chassis_scale_confirmed",
    "current_chassis_viable_but_architecture_comparison_needed",
    "architecture_pivot_recommended",
    "raw_redesign_scale_regression",
    "no_viable_scale_path",
    "not GPT-like assistant readiness",
    "not open-domain assistant readiness",
    "not production chat",
    "not public API",
    "not deployment readiness",
    "not safety alignment",
]
NEGATION_MARKERS = ["not ", "no ", "does not ", "do not ", "without ", "false", "forbidden", "only"]
FORBIDDEN_CLAIMS = {
    "GPT_LIKE_READINESS_FALSE_CLAIM": ["GPT-like assistant readiness"],
    "OPEN_DOMAIN_ASSISTANT_CLAIM_DETECTED": ["open-domain assistant readiness"],
    "PRODUCTION_CHAT_CLAIM_DETECTED": ["production chat"],
    "PUBLIC_API_CLAIM_DETECTED": ["public API"],
    "DEPLOYMENT_CLAIM_DETECTED": ["deployment readiness"],
    "SAFETY_ALIGNMENT_CLAIM_DETECTED": ["safety alignment"],
}
ARMS = {
    "CURRENT_RAW_BASELINE",
    "REDESIGNED_RAW_OBJECTIVE_CURRENT_CHASSIS_SCALE",
    "SMALL_CAUSAL_TRANSFORMER_BASELINE_SCALE",
    "POLICY_TRACE_DISTILLATION_SCALE_DIAGNOSTIC",
    "STATIC_OUTPUT_CONTROL",
    "COPY_PROMPT_CONTROL",
}
ALLOWED_DECISIONS = {
    "current_chassis_scale_confirmed",
    "current_chassis_viable_but_architecture_comparison_needed",
    "architecture_pivot_recommended",
    "raw_redesign_scale_regression",
    "no_viable_scale_path",
}
ALLOWED_NEXT = {
    "113_RAW_ASSISTANT_CAPABILITY_PACKAGE_AND_BOUNDARY_REVIEW",
    "113_ARCHITECTURE_COMPARISON_SCALE_REVIEW",
    "113_ARCHITECTURE_PIVOT_EVALUATION",
    "112B_RAW_SCALE_REGRESSION_ANALYSIS",
    "112Y_FOUNDATION_OBJECTIVE_FAILURE_ANALYSIS",
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
    if "CURRENT_CHASSIS_RAW_GENERATION_SCALE_CONFIRM_POSITIVE" not in verdicts:
        failures.append("CURRENT_CHASSIS_RAW_GENERATION_SCALE_CONFIRM_RESULT_MISSING")
    for key in [
        "scale_confirm_research_gate",
        "service_runtime_integration_performed",
        "runtime_surface_mutated",
        "bounded_release_stack_mutated",
        "gpt_like_assistant_readiness_claimed",
        "open_domain_assistant_readiness_claimed",
        "production_chat_claimed",
        "public_api_claimed",
        "deployment_readiness_claimed",
        "safety_alignment_claimed",
    ]:
        expected = True if key == "scale_confirm_research_gate" else False
        if summary.get(key) is not expected:
            failures.append("GPT_LIKE_READINESS_FALSE_CLAIM" if "readiness" in key else "PRODUCTION_CHAT_CLAIM_DETECTED")

    config = load_json(SMOKE_ROOT / "scale_config.json")
    if config.get("seeds") != [2081, 2082, 2083] or config.get("steps") != 16000 or config.get("train_examples") != 180000 or config.get("eval_rows_per_family") != 48:
        failures.append("FULL_112_RUN_NOT_USED")
    if config.get("main_positive_arm") != "REDESIGNED_RAW_OBJECTIVE_CURRENT_CHASSIS_SCALE":
        failures.append("RAW_OBJECTIVE_REDESIGN_SCALE_FAILS")

    upstream_111x = load_json(SMOKE_ROOT / "upstream_111x_manifest.json")
    upstream_111x_summary = upstream_111x.get("summary", {})
    if "CHASSIS_DECISION_RAW_GENERATION_REDESIGN_GATE_POSITIVE" not in set(upstream_111x_summary.get("verdicts", [])):
        failures.append("UPSTREAM_111X_NOT_POSITIVE")
    upstream_decision = load_json(SMOKE_ROOT / "upstream_111x_architecture_decision.json")
    if upstream_decision.get("decision") != "current_chassis_remains_viable" or upstream_decision.get("winning_arm") != "REDESIGNED_RAW_OBJECTIVE_CURRENT_CHASSIS":
        failures.append("UPSTREAM_111X_NOT_POSITIVE")
    for rel, verdict in [
        ("upstream_111r_manifest.json", "UPSTREAM_STACK_NOT_POSITIVE"),
        ("upstream_110_manifest.json", "UPSTREAM_STACK_NOT_POSITIVE"),
        ("upstream_100_manifest.json", "UPSTREAM_STACK_NOT_POSITIVE"),
        ("upstream_099_manifest.json", "UPSTREAM_STACK_NOT_POSITIVE"),
    ]:
        if "POSITIVE" not in json.dumps(load_json(SMOKE_ROOT / rel).get("summary", {}).get("verdicts", [])):
            failures.append(verdict)

    for rel in ["checkpoint_integrity_manifest.json", "bounded_release_integrity_manifest.json"]:
        integrity = load_json(SMOKE_ROOT / rel)
        for key in ["bounded_release_artifact_unchanged", "source_100_checkpoint_unchanged", "source_102_checkpoint_unchanged", "packaged_winner_hash_unchanged"]:
            if integrity.get(key) is not True or metrics.get(key) is not True:
                failures.append("BOUNDED_RELEASE_MUTATION_DETECTED" if "bounded" in key else "SOURCE_CHECKPOINT_MUTATION_DETECTED")

    comparison = load_json(SMOKE_ROOT / "arm_comparison.json")
    if comparison.get("all_eval_rows_match") is not True:
        failures.append("BASELINE_EVAL_MISMATCH")
    arm_rows = comparison.get("arms", [])
    if {row.get("arm") for row in arm_rows} != ARMS:
        failures.append("BASELINE_EVAL_MISMATCH")
    hashes = {(row.get("eval_row_hash"), row.get("eval_prompt_hash"), row.get("eval_count")) for row in arm_rows}
    if len(hashes) != 1:
        failures.append("BASELINE_EVAL_MISMATCH")
    arms = {row["arm"]: row for row in arm_rows}
    static_acc = arms.get("STATIC_OUTPUT_CONTROL", {}).get("metrics", {}).get("raw_ood_accuracy", 1.0)
    copy_acc = arms.get("COPY_PROMPT_CONTROL", {}).get("metrics", {}).get("raw_ood_accuracy", 1.0)
    if static_acc >= 0.70 or copy_acc >= 0.70:
        failures.append("STATIC_OR_COPY_CONTROL_UNEXPECTED_PASS")

    eval_results = read_jsonl(SMOKE_ROOT / "arm_eval_results.jsonl")
    main_rows = [row for row in eval_results if row.get("arm") == "REDESIGNED_RAW_OBJECTIVE_CURRENT_CHASSIS_SCALE"]
    if not main_rows:
        failures.append("RAW_OBJECTIVE_REDESIGN_SCALE_FAILS")
    if any(row.get("integrated_policy_used_during_raw_eval") or row.get("decoder_reference_used_during_raw_eval") or row.get("expected_answer_used_during_eval") or row.get("policy_trace_used_during_final_eval") for row in main_rows):
        failures.append("INTEGRATED_POLICY_USED_DURING_RAW_EVAL")

    namespace = load_json(SMOKE_ROOT / "namespace_audit.json")
    per_arm_ns = namespace.get("per_arm", {}).get("REDESIGNED_RAW_OBJECTIVE_CURRENT_CHASSIS_SCALE", {})
    if per_arm_ns.get("namespace_leak_rate", 1.0) > 0.03 or per_arm_ns.get("teacher_namespace_copy_rate", 1.0) > 0.03 or per_arm_ns.get("case_id_drift_rate", 1.0) > 0.05:
        failures.append("NAMESPACE_MEMORIZATION_DETECTED")

    aggregate = load_json(SMOKE_ROOT / "scale_aggregate.json")
    if aggregate.get("all_seeds_passed_independently") is not True:
        failures.append("MULTI_SEED_SCALE_INSTABILITY_DETECTED")
    if aggregate.get("min_raw_ood_accuracy", 0.0) < 0.80 or aggregate.get("mean_raw_ood_accuracy", 0.0) < 0.85:
        failures.append("RAW_OBJECTIVE_REDESIGN_SCALE_FAILS")
    if aggregate.get("max_namespace_leak_rate", 1.0) > 0.03 or aggregate.get("max_teacher_namespace_copy_rate", 1.0) > 0.03 or aggregate.get("max_case_id_drift_rate", 1.0) > 0.05:
        failures.append("NAMESPACE_MEMORIZATION_DETECTED")
    if aggregate.get("retention_pass_all_seeds") is not True:
        failures.append("RETENTION_REGRESSION_DETECTED")
    if aggregate.get("collapse_rejected_all_seeds") is not True:
        failures.append("STATIC_RESPONSE_COLLAPSE_DETECTED")
    if aggregate.get("fineweb_eval_loss_regression", 999.0) > 0.50 or aggregate.get("fineweb_next_byte_accuracy_drop", 999.0) > 0.10:
        failures.append("LM_RETENTION_REGRESSION_DETECTED")
    for key in ["artifact_exfiltration_count", "gpt_like_claim_count", "production_chat_claim_count", "public_api_claim_count", "safety_alignment_claim_count"]:
        if aggregate.get(key, 1) != 0:
            failures.append("ARTIFACT_EXFILTRATION_DETECTED" if key == "artifact_exfiltration_count" else "OVERCLAIM_DETECTED")

    seed_rows = read_jsonl(SMOKE_ROOT / "seed_metrics.jsonl")
    if not seed_rows or any(row.get("seed_passed_independently") is not True for row in seed_rows):
        failures.append("MULTI_SEED_SCALE_INSTABILITY_DETECTED")

    fairness = load_json(SMOKE_ROOT / "transformer_fairness_report.json")
    for key in ["transformer_parameter_count", "current_chassis_parameter_count", "training_tokens_seen", "train_steps", "optimizer", "learning_rate", "batch_size", "seq_len", "wall_clock_sec", "device"]:
        if key not in fairness:
            failures.append("TRANSFORMER_BASELINE_FAIRNESS_MISSING")
    if fairness.get("parameter_count_ratio", 999.0) > 2.0 and fairness.get("normalized_comparison_required") is not True:
        failures.append("TRANSFORMER_BASELINE_FAIRNESS_MISSING")
    if fairness.get("architecture_superiority_claimed_from_raw_accuracy_alone") is True:
        failures.append("TRANSFORMER_BASELINE_FAIRNESS_MISSING")

    decision = load_json(SMOKE_ROOT / "decision.json")
    if decision.get("decision") not in ALLOWED_DECISIONS or decision.get("next") not in {
        "113_RAW_ASSISTANT_CAPABILITY_PACKAGE_AND_BOUNDARY_REVIEW",
        "113_ARCHITECTURE_COMPARISON_SCALE_REVIEW",
        "113_ARCHITECTURE_PIVOT_EVALUATION",
        "112B_RAW_SCALE_REGRESSION_ANALYSIS",
        "112Y_FOUNDATION_OBJECTIVE_FAILURE_ANALYSIS",
    }:
        failures.append("DECISION_MISSING")

    training_rows = read_jsonl(SMOKE_ROOT / "arm_training_metrics.jsonl")
    main_training = next((row for row in training_rows if row.get("arm") == "REDESIGNED_RAW_OBJECTIVE_CURRENT_CHASSIS_SCALE"), {})
    if main_training.get("train_step_count", 0) <= 0 or main_training.get("optimizer_step_count", 0) <= 0 or main_training.get("target_checkpoint_changed") is not True:
        failures.append("RAW_OBJECTIVE_REDESIGN_SCALE_FAILS")

    samples = read_jsonl(SMOKE_ROOT / "human_readable_samples.jsonl")
    required_sample_arms = {"CURRENT_RAW_BASELINE", "REDESIGNED_RAW_OBJECTIVE_CURRENT_CHASSIS_SCALE", "SMALL_CAUSAL_TRANSFORMER_BASELINE_SCALE", "STATIC_OUTPUT_CONTROL", "COPY_PROMPT_CONTROL"}
    if not samples or not required_sample_arms.issubset({row.get("arm") for row in samples}):
        failures.append("HUMAN_SAMPLE_REPORT_MISSING")
    for row in samples[:50]:
        required = {"seed", "arm", "prompt", "generated_text", "expected_behavior", "required_keywords", "forbidden_outputs", "pass_fail", "namespace_detected", "short_diagnosis"}
        if not required.issubset(row):
            failures.append("HUMAN_SAMPLE_REPORT_MISSING")
            break

    combined_reports = (SMOKE_ROOT / "summary.json").read_text(encoding="utf-8") + "\n" + (SMOKE_ROOT / "report.md").read_text(encoding="utf-8")
    failures.extend(find_false_claims(combined_reports))
    for path in changed_paths():
        if path.startswith("target/"):
            failures.append("GENERATED_ARTIFACT_STAGED")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Check STABLE_LOOP_PHASE_LOCK_112 artifacts")
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
        print("STABLE_LOOP_PHASE_LOCK_112_CHECK_FAIL")
        for failure in sorted(set(failures)):
            print(failure)
        return 1
    print("STABLE_LOOP_PHASE_LOCK_112_CHECK_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
