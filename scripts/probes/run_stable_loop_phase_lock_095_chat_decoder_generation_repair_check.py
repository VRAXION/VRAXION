#!/usr/bin/env python3
"""Static/artifact checker for STABLE_LOOP_PHASE_LOCK_095."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_095_chat_decoder_generation_repair/smoke"
REQUIRED_DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_095_CHAT_DECODER_GENERATION_REPAIR_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_095_CHAT_DECODER_GENERATION_REPAIR_RESULT.md",
]
REQUIRED_SOURCE = [
    "scripts/probes/run_stable_loop_phase_lock_095_chat_decoder_generation_repair.py",
    "scripts/probes/run_stable_loop_phase_lock_095_chat_decoder_generation_repair_check.py",
]
ALLOWED_MUTATIONS = set(REQUIRED_DOCS + REQUIRED_SOURCE)
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "repair_config.json",
    "upstream_094b_manifest.json",
    "checkpoint_integrity_manifest.json",
    "eval_row_manifest.json",
    "decoder_policy_manifest.json",
    "repaired_generation_results.jsonl",
    "baseline_vs_repaired_report.json",
    "family_metrics.json",
    "stop_condition_report.json",
    "human_readable_samples.jsonl",
    "failure_case_samples.jsonl",
    "summary.json",
    "report.md",
]
REQUIRED_TERMS = [
    "STABLE_LOOP_PHASE_LOCK_095_CHAT_DECODER_GENERATION_REPAIR",
    "target-only decoder generation repair PoC",
    "not GPT-like assistant readiness",
    "not open-domain assistant readiness",
    "not production chat",
    "not deployment",
    "not public release",
    "not safety alignment",
    "CHAT_DECODER_GENERATION_REPAIR_POSITIVE",
    "UPSTREAM_094B_GAP_ANALYSIS_VERIFIED",
    "TARGET_ONLY_DECODER_REPAIR_WRITTEN",
    "GENERATION_ACCURACY_REPAIRED",
    "STOP_CONDITION_REPAIRED",
    "FINITE_LABEL_OUTPUT_REPAIRED",
    "CHECKPOINTS_UNCHANGED",
    "NO_TRAINING_PERFORMED",
    "GPT_LIKE_READINESS_NOT_CLAIMED",
    "CHAT_DECODER_GENERATION_REPAIR_FAILS",
    "DECODER_REPAIR_INSUFFICIENT",
    "DECODER_REPAIR_FAMILY_REGRESSION",
    "CHECKPOINT_MUTATION_DETECTED",
    "TRAINING_SIDE_EFFECT_DETECTED",
    "RUNTIME_SURFACE_MUTATION_DETECTED",
    "ROOT_LICENSE_CHANGED",
    "096_FRESH_CHAT_GENERATION_EVAL",
]
NEGATION_MARKERS = ["not ", "no ", "does not ", "do not ", "without ", "false", "forbidden", "forbid", "reject", "rejected", "only"]
FORBIDDEN_CLAIMS = {
    "GPT_LIKE_READINESS_FALSE_CLAIM": ["GPT-like assistant readiness"],
    "PRODUCTION_CHAT_CLAIM_DETECTED": ["production chat"],
    "PUBLIC_RELEASE_CLAIM_DETECTED": ["public release"],
    "SAFETY_ALIGNMENT_CLAIM_DETECTED": ["safety alignment"],
    "DEPLOYMENT_CLAIM_DETECTED": ["deployment"],
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
    if "CHAT_DECODER_GENERATION_REPAIR_POSITIVE" not in verdicts:
        failures.append("CHAT_DECODER_GENERATION_REPAIR_POSITIVE_MISSING")
    if metrics.get("checkpoint_unchanged") is not True or metrics.get("no_training_performed") is not True or metrics.get("optimizer_step_count") != 0:
        failures.append("CHECKPOINT_OR_TRAINING_GATE_FAIL")
    if metrics.get("repaired_generated_accuracy", 0.0) < 0.90 or metrics.get("generation_accuracy_delta", 0.0) < 0.40:
        failures.append("DECODER_REPAIR_INSUFFICIENT")
    for key in ["bounded_slot_accuracy", "finite_label_accuracy", "unsupported_refusal_accuracy"]:
        if metrics.get(key, 0.0) < 0.90:
            failures.append(f"DECODER_REPAIR_FAMILY_REGRESSION:{key}")
    policy = load_json(SMOKE_ROOT / "decoder_policy_manifest.json")
    if policy.get("expected_response_used_for_generation") is not False or policy.get("response_table_used") is not False:
        failures.append("ORACLE_OR_TABLE_USED")
    samples = read_jsonl(SMOKE_ROOT / "human_readable_samples.jsonl")
    if len(samples) < 7:
        failures.append("HUMAN_SAMPLE_REPORT_MISSING")
    failures.extend(find_false_claims((SMOKE_ROOT / "report.md").read_text(encoding="utf-8")))
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Check STABLE_LOOP_PHASE_LOCK_095 artifacts")
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
    print("STABLE_LOOP_PHASE_LOCK_095_CHAT_DECODER_GENERATION_REPAIR_CHECK_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
