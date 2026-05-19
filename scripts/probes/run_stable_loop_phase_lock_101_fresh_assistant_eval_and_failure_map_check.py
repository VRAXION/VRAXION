#!/usr/bin/env python3
"""Static/artifact checker for STABLE_LOOP_PHASE_LOCK_101."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_101_fresh_assistant_eval_and_failure_map/smoke"
REQUIRED_DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_101_FRESH_ASSISTANT_EVAL_AND_FAILURE_MAP_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_101_FRESH_ASSISTANT_EVAL_AND_FAILURE_MAP_RESULT.md",
]
REQUIRED_SOURCE = [
    "scripts/probes/run_stable_loop_phase_lock_101_fresh_assistant_eval_and_failure_map.py",
    "scripts/probes/run_stable_loop_phase_lock_101_fresh_assistant_eval_and_failure_map_check.py",
]
ALLOWED_MUTATIONS = set(REQUIRED_DOCS + REQUIRED_SOURCE)
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "eval_config.json",
    "upstream_manifest.json",
    "bounded_release_freeze_manifest.json",
    "eval_dataset.jsonl",
    "generation_results.jsonl",
    "family_metrics.json",
    "failure_map.json",
    "collapse_metrics.json",
    "retention_metrics.json",
    "human_readable_samples.jsonl",
    "failure_case_samples.jsonl",
    "summary.json",
    "report.md",
]
REQUIRED_TERMS = [
    "STABLE_LOOP_PHASE_LOCK_101_FRESH_ASSISTANT_EVAL_AND_FAILURE_MAP",
    "fresh assistant eval and failure-map",
    "099 bounded local/private release baseline frozen",
    "no training",
    "not GPT-like assistant readiness",
    "not open-domain assistant readiness",
    "not public API",
    "not hosted SaaS",
    "not production chat",
    "not deployment readiness",
    "not safety alignment",
    "100 OPEN_VOCAB_ASSISTANT_CAPABILITY_SCALE_POSITIVE",
    "099 BOUNDED_LOCAL_PRIVATE_CLEAN_DEPLOY_READY_GATE_POSITIVE",
    "FRESH_ASSISTANT_EVAL_AND_FAILURE_MAP_POSITIVE",
    "FRESH_ASSISTANT_FAILURE_MAP_RECORDED",
    "BOUNDED_RELEASE_BASELINE_FROZEN",
    "RETENTION_PASSES",
    "COLLAPSE_REJECTED",
    "GPT_LIKE_READINESS_NOT_CLAIMED",
    "PRODUCTION_CHAT_NOT_CLAIMED",
    "FRESH_ASSISTANT_EVAL_AND_FAILURE_MAP_FAILS",
    "UPSTREAM_100_NOT_POSITIVE",
    "BOUNDED_RELEASE_MUTATION_DETECTED",
    "PACKAGED_CHECKPOINT_MUTATION_DETECTED",
    "ASSISTANT_FRESH_EVAL_WEAK",
    "MULTI_TURN_CONTEXT_FAILS",
    "HUNGARIAN_BASIC_FAILS",
    "REFUSAL_REGRESSION_DETECTED",
    "RETENTION_REGRESSION_DETECTED",
    "STATIC_RESPONSE_COLLAPSE_DETECTED",
    "REPETITION_COLLAPSE_DETECTED",
    "GPT_LIKE_READINESS_FALSE_CLAIM",
    "PRODUCTION_CHAT_CLAIM_DETECTED",
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
        if path.startswith("target/"):
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
    if summary.get("status") != "positive":
        failures.append(f"SUMMARY_NOT_POSITIVE:{','.join(sorted(verdicts))}")
        return failures
    required_verdicts = {
        "FRESH_ASSISTANT_EVAL_AND_FAILURE_MAP_POSITIVE",
        "FRESH_ASSISTANT_FAILURE_MAP_RECORDED",
        "BOUNDED_RELEASE_BASELINE_FROZEN",
        "RETENTION_PASSES",
        "COLLAPSE_REJECTED",
        "GPT_LIKE_READINESS_NOT_CLAIMED",
        "PRODUCTION_CHAT_NOT_CLAIMED",
    }
    for verdict in required_verdicts:
        if verdict not in verdicts:
            failures.append(f"VERDICT_MISSING:{verdict}")
    for key in ["target_100_checkpoint_unchanged", "bounded_release_artifact_unchanged", "no_training_performed"]:
        if metrics.get(key) is not True:
            failures.append(f"FREEZE_GATE_FAIL:{key}")
    thresholds = {
        "fresh_eval_row_count": 400,
        "overall_generated_accuracy": 0.30,
        "instruction_following_accuracy": 0.50,
        "short_explanation_accuracy": 0.50,
        "multi_turn_context_accuracy": 0.40,
        "hungarian_basic_accuracy": 0.40,
        "english_basic_accuracy": 0.50,
        "unsupported_refusal_accuracy": 0.80,
        "boundary_refusal_accuracy": 0.90,
        "bounded_chat_slot_binding_accuracy": 0.80,
        "finite_label_anchorroute_retention_accuracy": 0.90,
        "nonempty_generation_rate": 0.98,
        "utf8_valid_generation_rate": 0.80,
    }
    for key, threshold in thresholds.items():
        if metrics.get(key, 0.0) < threshold:
            failures.append(f"METRIC_GATE_FAIL:{key}")
    if metrics.get("empty_output_rate", 1.0) > 0.02 or metrics.get("static_output_rate", 1.0) > 0.15 or metrics.get("repetition_rate", 1.0) > 0.25 or metrics.get("copy_prompt_rate", 1.0) > 0.20:
        failures.append("COLLAPSE_GATE_FAIL")
    if metrics.get("llm_judge_used") is not False or metrics.get("prediction_oracle_used") is not False or metrics.get("response_table_used_for_main_prediction") is not False:
        failures.append("ORACLE_OR_JUDGE_USED")
    if len(read_jsonl(SMOKE_ROOT / "eval_dataset.jsonl")) < 400:
        failures.append("EVAL_DATASET_TOO_SMALL")
    if len(read_jsonl(SMOKE_ROOT / "generation_results.jsonl")) < 400:
        failures.append("GENERATION_RESULTS_TOO_SMALL")
    failures.extend(find_false_claims((SMOKE_ROOT / "report.md").read_text(encoding="utf-8")))
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Check STABLE_LOOP_PHASE_LOCK_101 artifacts")
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
    if SMOKE_ROOT.exists():
        failures.extend(check_artifacts())
    else:
        failures.append("SMOKE_ROOT_MISSING")
    if failures:
        print("STABLE_LOOP_PHASE_LOCK_101_CHECK_FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("STABLE_LOOP_PHASE_LOCK_101_CHECK_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
