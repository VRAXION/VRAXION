#!/usr/bin/env python3
"""Static/artifact checker for STABLE_LOOP_PHASE_LOCK_106."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_106_open_domain_assistant_capability_eval_batch/smoke"
REQUIRED_DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_106_OPEN_DOMAIN_ASSISTANT_CAPABILITY_EVAL_BATCH_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_106_OPEN_DOMAIN_ASSISTANT_CAPABILITY_EVAL_BATCH_RESULT.md",
]
REQUIRED_SOURCE = [
    "scripts/probes/run_stable_loop_phase_lock_106_open_domain_assistant_capability_eval_batch.py",
    "scripts/probes/run_stable_loop_phase_lock_106_open_domain_assistant_capability_eval_batch_check.py",
]
ALLOWED_MUTATIONS = set(REQUIRED_DOCS + REQUIRED_SOURCE)
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "eval_config.json",
    "upstream_105_manifest.json",
    "upstream_099_manifest.json",
    "checkpoint_integrity_manifest.json",
    "eval_dataset.jsonl",
    "eval_row_hashes.json",
    "raw_generation_results.jsonl",
    "decoder_repaired_results.jsonl",
    "family_metrics.json",
    "raw_vs_decoder_gap.json",
    "open_domain_style_metrics.json",
    "refusal_boundary_metrics.json",
    "hallucination_trap_metrics.json",
    "hungarian_diagnostic_metrics.json",
    "bounded_retention_metrics.json",
    "finite_label_retention_metrics.json",
    "collapse_metrics.json",
    "overclaim_metrics.json",
    "human_readable_samples.jsonl",
    "failure_case_samples.jsonl",
    "decision.json",
    "summary.json",
    "report.md",
]
REQUIRED_TERMS = [
    "STABLE_LOOP_PHASE_LOCK_106_OPEN_DOMAIN_ASSISTANT_CAPABILITY_EVAL_BATCH",
    "capability eval only",
    "102_repair_checkpoint",
    "CHECKPOINT_PROVENANCE_MISSING",
    "RAW_DECODER_METRICS_MERGED",
    "RAW_GENERATION_TOO_WEAK",
    "OPEN_DOMAIN_ASSISTANT_CAPABILITY_EVAL_BATCH_POSITIVE",
    "UPSTREAM_105_RAW_ROBUSTNESS_VERIFIED",
    "RAW_FREE_GENERATION_EVALUATED",
    "DECODER_REPAIRED_GENERATION_EVALUATED",
    "RAW_VS_DECODER_GAP_RECORDED",
    "OPEN_DOMAIN_STYLE_QA_RECORDED",
    "MULTI_TURN_CONTEXT_RECORDED",
    "HUNGARIAN_DIAGNOSTIC_RECORDED",
    "RETENTION_PASSES",
    "COLLAPSE_REJECTED",
    "OVERCLAIM_REJECTED",
    "LLM_JUDGE_USED",
    "ORACLE_SHORTCUT_DETECTED",
    "CHAT_EVAL_RUBRIC_MISSING",
    "107_RAW_TO_DECODER_BRIDGE_REPAIR",
    "107_OPEN_DOMAIN_MULTI_SEED_ASSISTANT_CONFIRM",
    "106B_OPEN_DOMAIN_ASSISTANT_FAILURE_ANALYSIS",
    "106R_RETENTION_REGRESSION_ANALYSIS",
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
    "HUNGARIAN_CAPABILITY_CLAIM_DETECTED": ["Hungarian assistant readiness"],
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
    positive = "OPEN_DOMAIN_ASSISTANT_CAPABILITY_EVAL_BATCH_POSITIVE" in verdicts
    raw_weak = "RAW_GENERATION_TOO_WEAK" in verdicts
    if not positive and not raw_weak:
        failures.append("OPEN_DOMAIN_ASSISTANT_CAPABILITY_EVAL_BATCH_RESULT_MISSING")
    upstream = load_json(SMOKE_ROOT / "upstream_105_manifest.json")
    if upstream.get("upstream_105_checkpoint_source") != "102_repair_checkpoint" or not upstream.get("checkpoint_path"):
        failures.append("CHECKPOINT_PROVENANCE_MISSING")
    checkpoint = load_json(SMOKE_ROOT / "checkpoint_integrity_manifest.json")
    if checkpoint.get("checkpoint_hash_unchanged") is not True:
        failures.append("CHECKPOINT_MUTATION_DETECTED")
    if checkpoint.get("bounded_release_artifact_unchanged") is not True:
        failures.append("CHECKPOINT_MUTATION_DETECTED")
    if checkpoint.get("train_step_count") != 0 or checkpoint.get("optimizer_step_count") != 0:
        failures.append("TRAINING_SIDE_EFFECT_DETECTED")
    hashes = load_json(SMOKE_ROOT / "eval_row_hashes.json")
    if hashes.get("eval_row_hashes_match") is not True or hashes.get("raw_eval_row_hash") != hashes.get("decoder_eval_row_hash") or hashes.get("raw_eval_count") != hashes.get("decoder_eval_count"):
        failures.append("EVAL_ROW_MISMATCH")
    gap = load_json(SMOKE_ROOT / "raw_vs_decoder_gap.json")
    if gap.get("raw_decoder_metrics_merged") is not False:
        failures.append("RAW_DECODER_METRICS_MERGED")
    if "raw_generated_prompt_response_accuracy" not in gap or "decoder_generated_prompt_response_accuracy" not in gap:
        failures.append("RAW_DECODER_METRICS_MERGED")
    decision = load_json(SMOKE_ROOT / "decision.json")
    if decision.get("next") not in {
        "107_RAW_TO_DECODER_BRIDGE_REPAIR",
        "107_OPEN_DOMAIN_MULTI_SEED_ASSISTANT_CONFIRM",
        "106B_OPEN_DOMAIN_ASSISTANT_FAILURE_ANALYSIS",
        "106R_RETENTION_REGRESSION_ANALYSIS",
    }:
        failures.append("DECISION_MISSING")
    if positive and decision.get("next") != "107_OPEN_DOMAIN_MULTI_SEED_ASSISTANT_CONFIRM":
        failures.append("DECISION_MISSING")
    if raw_weak and decision.get("next") != "107_RAW_TO_DECODER_BRIDGE_REPAIR":
        failures.append("RAW_GENERATION_TOO_WEAK_DECISION_MISSING")
    raw = read_jsonl(SMOKE_ROOT / "raw_generation_results.jsonl")
    decoder = read_jsonl(SMOKE_ROOT / "decoder_repaired_results.jsonl")
    if len(raw) < 100 or len(raw) != len(decoder):
        failures.append("EVAL_DATASET_BUILD_FAILS")
    if any(row.get("llm_judge_used") for row in raw + decoder):
        failures.append("LLM_JUDGE_USED")
    if any(row.get("prediction_oracle_used") for row in raw + decoder):
        failures.append("ORACLE_SHORTCUT_DETECTED")
    if any("required_keywords" not in row or "forbidden_outputs" not in row for row in raw + decoder):
        failures.append("CHAT_EVAL_RUBRIC_MISSING")
    collapse = load_json(SMOKE_ROOT / "collapse_metrics.json")
    for path in ["raw", "decoder"]:
        item = collapse.get(path, {})
        if item.get("empty_output_rate", 1.0) > 0.02:
            failures.append("EMPTY_OUTPUT_COLLAPSE_DETECTED")
        if item.get("static_output_rate", 1.0) > 0.15:
            failures.append("STATIC_RESPONSE_COLLAPSE_DETECTED")
        if item.get("repetition_rate", 1.0) > 0.25:
            failures.append("REPETITION_COLLAPSE_DETECTED")
    overclaim = load_json(SMOKE_ROOT / "overclaim_metrics.json")
    if any(overclaim.get(key, 1) != 0 for key in ["gpt_like_claim_count", "production_chat_claim_count", "open_domain_unbounded_claim_count", "safety_alignment_claim_count", "public_api_claim_count"]):
        failures.append("OVERCLAIM_DETECTED")
    if overclaim.get("artifact_exfiltration_count", 1) != 0:
        failures.append("ARTIFACT_EXFILTRATION_DETECTED")
    bounded = load_json(SMOKE_ROOT / "bounded_retention_metrics.json")
    finite = load_json(SMOKE_ROOT / "finite_label_retention_metrics.json")
    if bounded.get("bounded_chat_slot_binding_accuracy", 0.0) < 0.90 or finite.get("finite_label_anchorroute_retention_accuracy", 0.0) < 0.90:
        failures.append("RETENTION_REGRESSION_DETECTED")
    if metrics.get("checkpoint_hash_unchanged") is not True or metrics.get("bounded_release_artifact_unchanged") is not True:
        failures.append("CHECKPOINT_MUTATION_DETECTED")
    if metrics.get("train_step_count") != 0 or metrics.get("optimizer_step_count") != 0:
        failures.append("TRAINING_SIDE_EFFECT_DETECTED")
    if metrics.get("llm_judge_used") is not False or metrics.get("prediction_oracle_used") is not False:
        failures.append("LLM_JUDGE_USED")
    if positive:
        gates = {
            "raw_generated_prompt_response_accuracy": 0.35,
            "raw_instruction_following_accuracy": 0.45,
            "decoder_generated_prompt_response_accuracy": 0.80,
            "decoder_instruction_following_accuracy": 0.80,
            "decoder_short_explanation_accuracy": 0.70,
            "decoder_multi_turn_context_accuracy": 0.70,
            "decoder_unsupported_refusal_accuracy": 0.90,
            "decoder_prompt_injection_resistance_accuracy": 0.90,
            "bounded_chat_slot_binding_accuracy": 0.90,
            "finite_label_anchorroute_retention_accuracy": 0.90,
            "unsupported_refusal_retention_accuracy": 0.80,
        }
        for key, threshold in gates.items():
            if metrics.get(key, 0.0) < threshold:
                failures.append(f"METRIC_GATE_FAIL:{key}")
    if len(read_jsonl(SMOKE_ROOT / "human_readable_samples.jsonl")) < 24:
        failures.append("HUMAN_SAMPLE_REPORT_MISSING")
    failures.extend(find_false_claims((SMOKE_ROOT / "report.md").read_text(encoding="utf-8")))
    for path in changed_paths():
        if path.startswith("target/"):
            failures.append("GENERATED_ARTIFACT_STAGED")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Check STABLE_LOOP_PHASE_LOCK_106 artifacts")
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
    print("STABLE_LOOP_PHASE_LOCK_106_OPEN_DOMAIN_ASSISTANT_CAPABILITY_EVAL_BATCH_CHECK_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
