#!/usr/bin/env python3
"""Static/artifact checker for STABLE_LOOP_PHASE_LOCK_096."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_096_fresh_chat_generation_eval/smoke"
REQUIRED_DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_096_FRESH_CHAT_GENERATION_EVAL_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_096_FRESH_CHAT_GENERATION_EVAL_RESULT.md",
]
REQUIRED_SOURCE = [
    "scripts/probes/run_stable_loop_phase_lock_096_fresh_chat_generation_eval.py",
    "scripts/probes/run_stable_loop_phase_lock_096_fresh_chat_generation_eval_check.py",
]
ALLOWED_MUTATIONS = set(REQUIRED_DOCS + REQUIRED_SOURCE)
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "eval_config.json",
    "upstream_095_manifest.json",
    "checkpoint_integrity_manifest.json",
    "fresh_eval_manifest.json",
    "fresh_eval_dataset.jsonl",
    "fresh_generation_results.jsonl",
    "decoder_policy_manifest.json",
    "family_metrics.json",
    "collapse_metrics.json",
    "freshness_validation.json",
    "claim_boundary.json",
    "human_readable_samples.jsonl",
    "failure_case_samples.jsonl",
    "summary.json",
    "report.md",
]
REQUIRED_TERMS = [
    "STABLE_LOOP_PHASE_LOCK_096_FRESH_CHAT_GENERATION_EVAL",
    "fresh-row eval",
    "target-only decoder repair",
    "not GPT-like assistant readiness",
    "not open-domain assistant readiness",
    "not production chat",
    "not deployment",
    "not public release",
    "not safety alignment",
    "FRESH_CHAT_GENERATION_EVAL_POSITIVE",
    "UPSTREAM_095_REPAIR_VERIFIED",
    "FRESH_EVAL_ROWS_VERIFIED",
    "FRESH_GENERATION_REPAIR_GENERALIZES",
    "CHECKPOINTS_UNCHANGED",
    "NO_TRAINING_PERFORMED",
    "FRESH_CHAT_GENERATION_EVAL_FAILS",
    "FRESH_EVAL_ROW_OVERLAP_DETECTED",
    "FRESH_GENERATION_REGRESSION_DETECTED",
    "BOUNDED_CHAT_RETENTION_REGRESSION_DETECTED",
    "FINITE_LABEL_RETENTION_REGRESSION_DETECTED",
    "UNSUPPORTED_REFUSAL_REGRESSION_DETECTED",
    "CHECKPOINT_MUTATION_DETECTED",
    "TRAINING_SIDE_EFFECT_DETECTED",
    "RUNTIME_SURFACE_MUTATION_DETECTED",
    "ROOT_LICENSE_CHANGED",
    "097_CHAT_DECODER_MULTI_SEED_OOD_RETENTION_CONFIRM",
]
NEGATION_MARKERS = ["not ", "no ", "does not ", "do not ", "without ", "false", "forbidden", "forbid", "reject", "rejected", "only"]
FORBIDDEN_CLAIMS = {
    "GPT_LIKE_READINESS_FALSE_CLAIM": ["GPT-like assistant readiness"],
    "OPEN_DOMAIN_ASSISTANT_CLAIM_DETECTED": ["open-domain assistant readiness"],
    "PRODUCTION_CHAT_CLAIM_DETECTED": ["production chat"],
    "PUBLIC_RELEASE_CLAIM_DETECTED": ["public release"],
    "SAFETY_ALIGNMENT_CLAIM_DETECTED": ["safety alignment"],
    "DEPLOYMENT_CLAIM_DETECTED": ["deployment readiness", "deployment"],
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
    if "FRESH_CHAT_GENERATION_EVAL_POSITIVE" not in verdicts:
        failures.append("FRESH_CHAT_GENERATION_EVAL_POSITIVE_MISSING")
    if metrics.get("fresh_generated_accuracy", 0.0) < 0.90:
        failures.append("FRESH_GENERATION_REGRESSION_DETECTED")
    for key in ["bounded_slot_accuracy", "finite_label_accuracy", "unsupported_refusal_accuracy"]:
        if metrics.get(key, 0.0) < 0.90:
            failures.append(f"RETENTION_GATE_FAIL:{key}")
    if metrics.get("checkpoint_unchanged") is not True or metrics.get("no_training_performed") is not True or metrics.get("optimizer_step_count") != 0:
        failures.append("CHECKPOINT_OR_TRAINING_GATE_FAIL")
    if metrics.get("expected_response_used_for_generation") is not False or metrics.get("response_table_used") is not False:
        failures.append("ORACLE_OR_TABLE_USED")
    if metrics.get("overlap_with_095_eval_prompts") != 0 or metrics.get("overlap_with_095_eval_expected_responses") != 0:
        failures.append("FRESH_EVAL_ROW_OVERLAP_DETECTED")
    manifest = load_json(SMOKE_ROOT / "fresh_eval_manifest.json")
    if manifest.get("fresh_eval_row_count", 0) < 200:
        failures.append("FRESH_EVAL_TOO_SMALL")
    policy = load_json(SMOKE_ROOT / "decoder_policy_manifest.json")
    if policy.get("expected_response_used_for_generation") is not False or policy.get("response_table_used") is not False:
        failures.append("DECODER_POLICY_SHORTCUT_USED")
    samples = read_jsonl(SMOKE_ROOT / "human_readable_samples.jsonl")
    if len(samples) < 8:
        failures.append("HUMAN_SAMPLE_REPORT_MISSING")
    failures.extend(find_false_claims((SMOKE_ROOT / "report.md").read_text(encoding="utf-8")))
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Check STABLE_LOOP_PHASE_LOCK_096 artifacts")
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
    print("STABLE_LOOP_PHASE_LOCK_096_FRESH_CHAT_GENERATION_EVAL_CHECK_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
