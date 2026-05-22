#!/usr/bin/env python3
"""Static/artifact checker for STABLE_LOOP_PHASE_LOCK_111X."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_111x_chassis_decision_raw_generation_redesign_gate/smoke"
REQUIRED_DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_111X_CHASSIS_DECISION_RAW_GENERATION_REDESIGN_GATE_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_111X_CHASSIS_DECISION_RAW_GENERATION_REDESIGN_GATE_RESULT.md",
]
REQUIRED_SOURCE = [
    "scripts/probes/run_stable_loop_phase_lock_111x_chassis_decision_raw_generation_redesign_gate.py",
    "scripts/probes/run_stable_loop_phase_lock_111x_chassis_decision_raw_generation_redesign_gate_check.py",
]
ALLOWED_MUTATIONS = set(REQUIRED_DOCS + REQUIRED_SOURCE)
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "decision_config.json",
    "upstream_111r_manifest.json",
    "upstream_111_manifest.json",
    "upstream_110_manifest.json",
    "upstream_109_manifest.json",
    "upstream_100_manifest.json",
    "upstream_099_manifest.json",
    "bounded_release_integrity_manifest.json",
    "train_dataset_manifest.json",
    "eval_dataset_manifest.json",
    "namespace_audit.json",
    "arm_training_metrics.jsonl",
    "arm_eval_results.jsonl",
    "arm_comparison.json",
    "architecture_decision.json",
    "retention_report.json",
    "collapse_metrics.json",
    "fineweb_retention_report.json",
    "transformer_baseline_fairness.json",
    "human_readable_samples.jsonl",
    "failure_case_samples.jsonl",
    "summary.json",
    "report.md",
]
REQUIRED_TERMS = [
    "STABLE_LOOP_PHASE_LOCK_111X_CHASSIS_DECISION_RAW_GENERATION_REDESIGN_GATE",
    "chassis decision gate",
    "CHASSIS_DECISION_RAW_GENERATION_REDESIGN_GATE_POSITIVE",
    "UPSTREAM_111R_ANALYSIS_VERIFIED",
    "NAMESPACE_MEMORIZATION_REJECTED",
    "RAW_OBJECTIVE_REDESIGN_EVALUATED",
    "POLICY_TRACE_DISTILLATION_EVALUATED",
    "SMALL_TRANSFORMER_BASELINE_EVALUATED",
    "ARCHITECTURE_DECISION_WRITTEN",
    "current_chassis_remains_viable",
    "current_chassis_viable_only_with_policy_trace",
    "architecture_comparison_needed_before_scaling",
    "architecture_pivot_recommended",
    "no_viable_raw_chassis_found",
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
ALLOWED_DECISIONS = {
    "current_chassis_remains_viable",
    "current_chassis_viable_only_with_policy_trace",
    "architecture_comparison_needed_before_scaling",
    "architecture_pivot_recommended",
    "no_viable_raw_chassis_found",
}
ALLOWED_NEXT = {
    "112_CURRENT_CHASSIS_RAW_GENERATION_SCALE_CONFIRM",
    "112_ARCHITECTURE_BASELINE_COMPARISON_SCALE",
    "112_POLICY_TRACE_DISTILLATION_SCALE_CONFIRM",
    "112_ARCHITECTURE_PIVOT_EVALUATION",
    "111Y_FOUNDATION_OBJECTIVE_FAILURE_ANALYSIS",
}
ARMS = {
    "CURRENT_RAW_BASELINE",
    "FAILED_111_STANDARD_REPLAY_DIAGNOSTIC",
    "REDESIGNED_RAW_OBJECTIVE_CURRENT_CHASSIS",
    "DECODER_POLICY_TRACE_DISTILLATION",
    "SMALL_CAUSAL_TRANSFORMER_BASELINE",
    "INTEGRATED_DECODER_POLICY_REFERENCE",
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


def arm_map() -> dict[str, dict[str, Any]]:
    comparison = load_json(SMOKE_ROOT / "arm_comparison.json")
    return {row["arm"]: row for row in comparison.get("arms", [])}


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
    if "CHASSIS_DECISION_RAW_GENERATION_REDESIGN_GATE_POSITIVE" not in verdicts:
        failures.append("CHASSIS_DECISION_RAW_GENERATION_REDESIGN_GATE_RESULT_MISSING")
    for key in [
        "chassis_decision_gate",
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
        expected = True if key == "chassis_decision_gate" else False
        if summary.get(key) is not expected:
            failures.append("GPT_LIKE_READINESS_FALSE_CLAIM" if "readiness" in key else "PRODUCTION_CHAT_CLAIM_DETECTED")

    upstream_111r = load_json(SMOKE_ROOT / "upstream_111r_manifest.json")
    if "RETENTION_OR_LM_REGRESSION_ANALYSIS_POSITIVE" not in set(upstream_111r.get("summary", {}).get("verdicts", [])):
        failures.append("UPSTREAM_111R_ARTIFACT_MISSING")
    upstream_111 = load_json(SMOKE_ROOT / "upstream_111_manifest.json")
    if upstream_111.get("summary", {}).get("status") != "failed":
        failures.append("UPSTREAM_111_FAILURE_NOT_FOUND")
    upstream_110 = load_json(SMOKE_ROOT / "upstream_110_manifest.json")
    if "INTEGRATED_DECODER_POLICY_OOD_CONFIRM_BATCH_POSITIVE" not in set(upstream_110.get("summary", {}).get("verdicts", [])):
        failures.append("UPSTREAM_STACK_NOT_POSITIVE")

    integrity = load_json(SMOKE_ROOT / "bounded_release_integrity_manifest.json")
    for key in ["bounded_release_artifact_unchanged", "packaged_winner_hash_unchanged", "source_102_checkpoint_unchanged", "source_100_checkpoint_unchanged"]:
        if integrity.get(key) is not True or metrics.get(key) is not True:
            failures.append("BOUNDED_RELEASE_MUTATION_DETECTED" if "bounded" in key else "SOURCE_CHECKPOINT_MUTATION_DETECTED")

    comparison = load_json(SMOKE_ROOT / "arm_comparison.json")
    if comparison.get("all_eval_rows_match") is not True:
        failures.append("BASELINE_EVAL_MISMATCH")
    arms = arm_map()
    if set(arms) != ARMS:
        failures.append("BASELINE_EVAL_MISMATCH")
    hashes = {(row.get("eval_row_hash"), row.get("eval_prompt_hash"), row.get("eval_count")) for row in comparison.get("arms", [])}
    if len(hashes) != 1:
        failures.append("BASELINE_EVAL_MISMATCH")

    decision = load_json(SMOKE_ROOT / "architecture_decision.json")
    for key in [
        "decision",
        "winning_arm",
        "current_chassis_viable",
        "transformer_baseline_stronger",
        "policy_trace_required",
        "raw_generation_scaling_recommended",
        "architecture_pivot_recommended",
        "next_milestone",
        "evidence_summary",
    ]:
        if key not in decision:
            failures.append("ARCHITECTURE_DECISION_MISSING")
    if decision.get("decision") not in ALLOWED_DECISIONS or decision.get("next_milestone") not in ALLOWED_NEXT:
        failures.append("ARCHITECTURE_DECISION_MISSING")

    namespace = load_json(SMOKE_ROOT / "namespace_audit.json")
    for key in ["train_namespace_prefixes", "eval_namespace_prefixes", "generated_namespace_prefixes", "candidate_namespace_leak_rate", "candidate_teacher_namespace_copy_rate", "candidate_case_id_drift_rate"]:
        if key not in namespace:
            failures.append("NAMESPACE_MEMORIZATION_DETECTED")
    if namespace.get("candidate_namespace_leak_rate", 1.0) > 0.05 or namespace.get("candidate_teacher_namespace_copy_rate", 1.0) > 0.05 or namespace.get("candidate_case_id_drift_rate", 1.0) > 0.10:
        failures.append("NAMESPACE_MEMORIZATION_DETECTED")

    fairness = load_json(SMOKE_ROOT / "transformer_baseline_fairness.json")
    for key in ["transformer_parameter_count", "current_chassis_parameter_count", "training_tokens_seen", "train_steps", "optimizer", "learning_rate", "batch_size", "seq_len", "wall_clock_sec", "device"]:
        if key not in fairness:
            failures.append("ARCHITECTURE_BASELINE_FAIRNESS_MISSING")
    ratio = fairness.get("parameter_count_ratio", 999.0)
    if ratio > 2.0 and fairness.get("normalized_comparison_required") is not True:
        failures.append("ARCHITECTURE_BASELINE_FAIRNESS_MISSING")
    if fairness.get("architecture_superiority_claimed_from_raw_accuracy_alone") is True:
        failures.append("ARCHITECTURE_BASELINE_FAIRNESS_MISSING")

    eval_results = read_jsonl(SMOKE_ROOT / "arm_eval_results.jsonl")
    if any(row.get("integrated_policy_used_during_raw_eval") or row.get("decoder_reference_used_during_raw_eval") or row.get("expected_answer_used_during_eval") for row in eval_results):
        failures.append("ORACLE_SHORTCUT_DETECTED")
    trace_rows = [row for row in eval_results if row.get("arm") == "DECODER_POLICY_TRACE_DISTILLATION"]
    if not trace_rows or any(row.get("policy_trace_used_during_final_eval") is not False for row in trace_rows):
        failures.append("POLICY_TRACE_USED_AS_INFERENCE_ORACLE")

    training_rows = read_jsonl(SMOKE_ROOT / "arm_training_metrics.jsonl")
    training_by_arm = {row.get("arm"): row for row in training_rows}
    redesigned = training_by_arm.get("REDESIGNED_RAW_OBJECTIVE_CURRENT_CHASSIS", {})
    for key in ["scheduled_sampling_batch_count", "rollout_loss_batch_count", "rollout_loss_weight", "prompt_binding_loss_weight", "retention_loss_weight"]:
        if redesigned.get(key, 0) <= 0:
            failures.append("RAW_OBJECTIVE_REDESIGN_INCOMPLETE")
    trace_training = training_by_arm.get("DECODER_POLICY_TRACE_DISTILLATION", {})
    if trace_training.get("policy_trace_labels_used_in_training") is not True or trace_training.get("policy_trace_used_during_final_eval") is not False:
        failures.append("POLICY_TRACE_USED_AS_INFERENCE_ORACLE")

    static_acc = arms.get("STATIC_OUTPUT_CONTROL", {}).get("metrics", {}).get("raw_ood_accuracy", 1.0)
    copy_acc = arms.get("COPY_PROMPT_CONTROL", {}).get("metrics", {}).get("raw_ood_accuracy", 1.0)
    if static_acc >= 0.70 or copy_acc >= 0.70:
        failures.append("STATIC_OR_COPY_CONTROL_UNEXPECTED_PASS")

    candidate = namespace.get("candidate_arm")
    if candidate in arms:
        candidate_metrics = arms[candidate].get("metrics", {})
        if candidate_metrics.get("bounded_chat_slot_binding_accuracy", 0.0) < 0.90 or candidate_metrics.get("finite_label_anchorroute_retention_accuracy", 0.0) < 0.90 or candidate_metrics.get("unsupported_refusal_accuracy", 0.0) < 0.80:
            failures.append("RETENTION_REGRESSION_DETECTED")
        candidate_training = training_by_arm.get(candidate, {})
        if candidate_training.get("fineweb_eval_loss_regression", 999.0) > 0.50 or candidate_training.get("fineweb_next_byte_accuracy_drop", 999.0) > 0.10:
            failures.append("LM_RETENTION_REGRESSION_DETECTED")

    samples = read_jsonl(SMOKE_ROOT / "human_readable_samples.jsonl")
    if not samples or not ARMS.issubset({row.get("arm") for row in samples}):
        failures.append("HUMAN_SAMPLE_REPORT_MISSING")
    for row in samples:
        required = {"arm", "eval_family", "prompt", "generated_text", "expected_behavior", "required_keywords", "forbidden_outputs", "pass_fail", "short_diagnosis"}
        if not required.issubset(row):
            failures.append("HUMAN_SAMPLE_REPORT_MISSING")
            break

    report_text = (SMOKE_ROOT / "report.md").read_text(encoding="utf-8")
    summary_text = (SMOKE_ROOT / "summary.json").read_text(encoding="utf-8")
    failures.extend(find_false_claims(report_text + "\n" + summary_text))
    for path in changed_paths():
        if path.startswith("target/"):
            failures.append("GENERATED_ARTIFACT_STAGED")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Check STABLE_LOOP_PHASE_LOCK_111X artifacts")
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
        print("STABLE_LOOP_PHASE_LOCK_111X_CHECK_FAIL")
        for failure in sorted(set(failures)):
            print(failure)
        return 1
    print("STABLE_LOOP_PHASE_LOCK_111X_CHECK_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
