#!/usr/bin/env python3
"""Static/artifact checker for STABLE_LOOP_PHASE_LOCK_100."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_100_open_vocab_assistant_capability_scale/smoke"
REQUIRED_DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_100_OPEN_VOCAB_ASSISTANT_CAPABILITY_SCALE_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_100_OPEN_VOCAB_ASSISTANT_CAPABILITY_SCALE_RESULT.md",
]
REQUIRED_SOURCE = [
    "scripts/probes/run_stable_loop_phase_lock_100_open_vocab_assistant_capability_scale.py",
    "scripts/probes/run_stable_loop_phase_lock_100_open_vocab_assistant_capability_scale_check.py",
]
ALLOWED_MUTATIONS = set(REQUIRED_DOCS + REQUIRED_SOURCE)
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "training_config.json",
    "upstream_manifest.json",
    "bounded_release_freeze_manifest.json",
    "fineweb_source_manifest.json",
    "fineweb_replay_manifest.json",
    "dataset_manifest.json",
    "train_examples_sample.jsonl",
    "eval_examples_sample.jsonl",
    "training_metrics.jsonl",
    "checkpoint_manifest.json",
    "checkpoint_hashes.json",
    "lm_metrics.json",
    "assistant_generation_metrics.json",
    "retention_metrics.json",
    "collapse_metrics.json",
    "control_comparison.json",
    "generation_samples.jsonl",
    "human_readable_samples.jsonl",
    "failure_case_samples.jsonl",
    "summary.json",
    "report.md",
]
REQUIRED_TERMS = [
    "STABLE_LOOP_PHASE_LOCK_100_OPEN_VOCAB_ASSISTANT_CAPABILITY_SCALE",
    "runner-local open-vocab assistant capability scale",
    "bounded release stack frozen",
    "not GPT-like assistant readiness",
    "not open-domain assistant readiness",
    "not production chat",
    "not public API",
    "not hosted SaaS",
    "not deployment readiness",
    "not safety alignment",
    "OPEN_VOCAB_ASSISTANT_CAPABILITY_SCALE_POSITIVE",
    "BOUNDED_RELEASE_BASELINE_FROZEN",
    "OPEN_VOCAB_TRAINING_COMPLETED",
    "ASSISTANT_GENERATION_IMPROVES",
    "MULTI_TURN_SMOKE_RECORDED",
    "HUNGARIAN_ENGLISH_SMOKE_RECORDED",
    "RETENTION_PASSES",
    "COLLAPSE_REJECTED",
    "OPEN_VOCAB_ASSISTANT_CAPABILITY_SCALE_FAILS",
    "BOUNDED_RELEASE_MUTATION_DETECTED",
    "PACKAGED_CHECKPOINT_MUTATION_DETECTED",
    "TOKEN_OBJECTIVE_NOT_LEARNED",
    "ASSISTANT_GENERATION_NOT_IMPROVED",
    "MULTI_TURN_CONTEXT_FAILS",
    "HUNGARIAN_BASIC_FAILS",
    "RETENTION_REGRESSION_DETECTED",
    "STATIC_RESPONSE_COLLAPSE_DETECTED",
    "REPETITION_COLLAPSE_DETECTED",
    "101_FRESH_ASSISTANT_EVAL_AND_FAILURE_MAP",
]
NEGATION_MARKERS = ["not ", "no ", "does not ", "do not ", "without ", "false", "forbidden", "forbid", "reject", "rejected", "only"]
FORBIDDEN_CLAIMS = {
    "GPT_LIKE_READINESS_FALSE_CLAIM": ["GPT-like assistant readiness"],
    "OPEN_DOMAIN_ASSISTANT_CLAIM_DETECTED": ["open-domain assistant readiness"],
    "PRODUCTION_CHAT_CLAIM_DETECTED": ["production chat"],
    "PUBLIC_API_CLAIM_DETECTED": ["public API"],
    "HOSTED_SAAS_CLAIM_DETECTED": ["hosted SaaS"],
    "SAFETY_ALIGNMENT_CLAIM_DETECTED": ["safety alignment"],
    "DEPLOYMENT_CLAIM_DETECTED": ["deployment readiness"],
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
        window = lowered[max(0, match.start() - 140) : match.start()]
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
        if path in ALLOWED_MUTATIONS:
            continue
        if path == "LICENSE" or path.startswith("instnct-core/") or path.startswith("tools/instnct_") or path.startswith("docs/product/") or path.startswith("docs/releases/") or path.startswith("sdk/") or path.startswith("packages/"):
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
    verdicts = set(summary.get("verdicts", []))
    metrics = summary.get("metrics", {})
    if "OPEN_VOCAB_ASSISTANT_CAPABILITY_SCALE_POSITIVE" not in verdicts:
        failures.append("OPEN_VOCAB_ASSISTANT_CAPABILITY_SCALE_POSITIVE_MISSING")
    for key in ["bounded_release_artifact_unchanged", "packaged_winner_hash_unchanged", "no_training_on_bounded_release", "target_100_checkpoint_changed"]:
        if metrics.get(key) is not True:
            failures.append(f"FREEZE_OR_CHECKPOINT_GATE_FAIL:{key}")
    if metrics.get("train_step_count", 0) <= 0 or not metrics.get("train_loss_final", 999.0) < metrics.get("train_loss_initial", -999.0):
        failures.append("TOKEN_OBJECTIVE_NOT_LEARNED")
    if not metrics.get("eval_loss_after", 999.0) < metrics.get("eval_loss_before", -999.0):
        failures.append("EVAL_LOSS_NOT_IMPROVED")
    for key, threshold in {
        "generated_prompt_response_accuracy": 0.25,
        "instruction_following_accuracy": 0.50,
        "short_explanation_accuracy": 0.50,
        "multi_turn_context_accuracy": 0.40,
        "unsupported_refusal_accuracy": 0.80,
        "bounded_chat_slot_binding_accuracy": 0.80,
        "finite_label_anchorroute_retention_accuracy": 0.90,
        "nonempty_generation_rate": 0.98,
        "utf8_valid_generation_rate": 0.80,
    }.items():
        if metrics.get(key, 0.0) < threshold:
            failures.append(f"METRIC_GATE_FAIL:{key}")
    if metrics.get("empty_output_rate", 1.0) > 0.02 or metrics.get("static_output_rate", 1.0) > 0.15 or metrics.get("repetition_rate", 1.0) > 0.25 or metrics.get("copy_prompt_rate", 1.0) > 0.20:
        failures.append("COLLAPSE_GATE_FAIL")
    if metrics.get("llm_judge_used") is not False or metrics.get("prediction_oracle_used") is not False or metrics.get("response_table_used_for_main_prediction") is not False:
        failures.append("ORACLE_OR_JUDGE_USED")
    control = load_json(SMOKE_ROOT / "control_comparison.json")
    if control.get("all_eval_rows_match") is not True or len(control.get("arms", [])) < 8:
        failures.append("CONTROL_COMPARISON_INCOMPLETE")
    if len(read_jsonl(SMOKE_ROOT / "human_readable_samples.jsonl")) < 20:
        failures.append("HUMAN_SAMPLE_REPORT_MISSING")
    failures.extend(find_false_claims((SMOKE_ROOT / "report.md").read_text(encoding="utf-8")))
    for path in changed_paths():
        if path.startswith("target/"):
            failures.append("GENERATED_ARTIFACT_STAGED")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Check STABLE_LOOP_PHASE_LOCK_100 artifacts")
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
    print("STABLE_LOOP_PHASE_LOCK_100_OPEN_VOCAB_ASSISTANT_CAPABILITY_SCALE_CHECK_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
