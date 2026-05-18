#!/usr/bin/env python3
"""Static/artifact checker for STABLE_LOOP_PHASE_LOCK_097."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_097_chat_decoder_multi_seed_ood_retention_confirm/smoke"
REQUIRED_DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_097_CHAT_DECODER_MULTI_SEED_OOD_RETENTION_CONFIRM_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_097_CHAT_DECODER_MULTI_SEED_OOD_RETENTION_CONFIRM_RESULT.md",
]
REQUIRED_SOURCE = [
    "scripts/probes/run_stable_loop_phase_lock_097_chat_decoder_multi_seed_ood_retention_confirm.py",
    "scripts/probes/run_stable_loop_phase_lock_097_chat_decoder_multi_seed_ood_retention_confirm_check.py",
]
ALLOWED_MUTATIONS = set(REQUIRED_DOCS + REQUIRED_SOURCE)
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "eval_config.json",
    "upstream_096_manifest.json",
    "checkpoint_integrity_manifest.json",
    "seed_run_manifest.json",
    "generation_results.jsonl",
    "multi_seed_aggregate.json",
    "ood_refusal_report.json",
    "retention_report.json",
    "collapse_metrics.json",
    "human_readable_samples.jsonl",
    "failure_case_samples.jsonl",
    "summary.json",
    "report.md",
]
REQUIRED_TERMS = [
    "STABLE_LOOP_PHASE_LOCK_097_CHAT_DECODER_MULTI_SEED_OOD_RETENTION_CONFIRM",
    "multi-seed OOD/refusal retention confirm",
    "target-only decoder repair",
    "not GPT-like assistant readiness",
    "not open-domain assistant readiness",
    "not production chat",
    "not deployment",
    "not public release",
    "not safety alignment",
    "CHAT_DECODER_MULTI_SEED_OOD_RETENTION_CONFIRM_POSITIVE",
    "UPSTREAM_096_FRESH_EVAL_VERIFIED",
    "MULTI_SEED_GENERATION_STABLE",
    "OOD_REFUSAL_RETENTION_PASSES",
    "CHECKPOINTS_UNCHANGED",
    "NO_TRAINING_PERFORMED",
    "CHAT_DECODER_MULTI_SEED_OOD_RETENTION_CONFIRM_FAILS",
    "UPSTREAM_096_ARTIFACT_MISSING",
    "UPSTREAM_096_NOT_POSITIVE",
    "MULTI_SEED_GENERATION_REGRESSION_DETECTED",
    "OOD_REFUSAL_RETENTION_REGRESSION_DETECTED",
    "MULTI_SEED_GENERATION_COLLAPSE_DETECTED",
    "CHECKPOINT_MUTATION_DETECTED",
    "TRAINING_SIDE_EFFECT_DETECTED",
    "RUNTIME_SURFACE_MUTATION_DETECTED",
    "ROOT_LICENSE_CHANGED",
    "098_PRIVATE_EVAL_RC_REFRESH_WITH_GENERATION_REPAIR",
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
    if "CHAT_DECODER_MULTI_SEED_OOD_RETENTION_CONFIRM_POSITIVE" not in verdicts:
        failures.append("CHAT_DECODER_MULTI_SEED_OOD_RETENTION_CONFIRM_POSITIVE_MISSING")
    if metrics.get("seed_count", 0) < 3 or metrics.get("total_eval_rows", 0) < 800:
        failures.append("MULTI_SEED_SAMPLE_TOO_SMALL")
    for key in ["min_seed_generated_accuracy", "min_bounded_slot_accuracy", "min_finite_label_accuracy", "min_unsupported_refusal_accuracy", "min_ood_refusal_accuracy"]:
        if metrics.get(key, 0.0) < 0.95:
            failures.append(f"MULTI_SEED_GATE_FAIL:{key}")
    if metrics.get("checkpoint_unchanged") is not True or metrics.get("no_training_performed") is not True or metrics.get("optimizer_step_count") != 0:
        failures.append("CHECKPOINT_OR_TRAINING_GATE_FAIL")
    if metrics.get("expected_response_used_for_generation") is not False or metrics.get("response_table_used") is not False:
        failures.append("ORACLE_OR_TABLE_USED")
    samples = read_jsonl(SMOKE_ROOT / "human_readable_samples.jsonl")
    if len(samples) < 30:
        failures.append("HUMAN_SAMPLE_REPORT_MISSING")
    aggregate = load_json(SMOKE_ROOT / "multi_seed_aggregate.json")
    if len(aggregate.get("seed_metrics", [])) < 3:
        failures.append("SEED_METRICS_MISSING")
    failures.extend(find_false_claims((SMOKE_ROOT / "report.md").read_text(encoding="utf-8")))
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Check STABLE_LOOP_PHASE_LOCK_097 artifacts")
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
    print("STABLE_LOOP_PHASE_LOCK_097_CHAT_DECODER_MULTI_SEED_OOD_RETENTION_CONFIRM_CHECK_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
