#!/usr/bin/env python3
"""Static/artifact checker for STABLE_LOOP_PHASE_LOCK_099."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_099_bounded_local_private_clean_deploy_ready_gate/smoke"
REQUIRED_DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_099_BOUNDED_LOCAL_PRIVATE_CLEAN_DEPLOY_READY_GATE_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_099_BOUNDED_LOCAL_PRIVATE_CLEAN_DEPLOY_READY_GATE_RESULT.md",
]
REQUIRED_SOURCE = [
    "scripts/probes/run_stable_loop_phase_lock_099_bounded_local_private_clean_deploy_ready_gate.py",
    "scripts/probes/run_stable_loop_phase_lock_099_bounded_local_private_clean_deploy_ready_gate_check.py",
]
ALLOWED_MUTATIONS = set(REQUIRED_DOCS + REQUIRED_SOURCE)
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "generated_clean_local_private_deploy_config.json",
    "clean_deploy_config_manifest.json",
    "upstream_release_manifest.json",
    "fresh_harness_child_manifest.json",
    "fresh_harness_validation.json",
    "release_readiness_evidence_chain.json",
    "claim_boundary.json",
    "fresh_harness_stdout.txt",
    "fresh_harness_stderr.txt",
    "summary.json",
    "report.md",
]
REQUIRED_TERMS = [
    "STABLE_LOOP_PHASE_LOCK_099_BOUNDED_LOCAL_PRIVATE_CLEAN_DEPLOY_READY_GATE",
    "clean local/private bounded deploy-readiness gate",
    "local/private release-readiness",
    "not production deployment",
    "not public API",
    "not hosted SaaS",
    "not GPT-like assistant readiness",
    "not open-domain chat",
    "not production chat",
    "not safety alignment",
    "BOUNDED_LOCAL_PRIVATE_CLEAN_DEPLOY_READY_GATE_POSITIVE",
    "UPSTREAM_RELEASE_EVIDENCE_VERIFIED",
    "CLEAN_LOCAL_PRIVATE_CONFIG_GENERATED",
    "FRESH_DEPLOYMENT_HARNESS_SMOKE_PASSES",
    "SDK_SMOKE_STILL_PASSES",
    "BOUNDED_CHAT_SERVICE_SMOKE_PASSES",
    "LOCAL_PRIVATE_RELEASE_READY",
    "BOUNDED_LOCAL_PRIVATE_CLEAN_DEPLOY_READY_GATE_FAILS",
    "FRESH_HARNESS_SMOKE_FAILS",
    "FRESH_HARNESS_GATE_FAILS",
    "TRAINING_SIDE_EFFECT_DETECTED",
    "RUNTIME_SURFACE_MUTATION_DETECTED",
    "ROOT_LICENSE_CHANGED",
]
NEGATION_MARKERS = ["not ", "no ", "does not ", "do not ", "without ", "false", "forbidden", "forbid", "reject", "rejected", "only"]
FORBIDDEN_CLAIMS = {
    "GPT_LIKE_READINESS_FALSE_CLAIM": ["GPT-like assistant readiness"],
    "OPEN_DOMAIN_CHAT_CLAIM_DETECTED": ["open-domain chat"],
    "PRODUCTION_CHAT_CLAIM_DETECTED": ["production chat"],
    "PUBLIC_API_CLAIM_DETECTED": ["public API"],
    "HOSTED_SAAS_CLAIM_DETECTED": ["hosted SaaS"],
    "SAFETY_ALIGNMENT_CLAIM_DETECTED": ["safety alignment"],
    "PRODUCTION_DEPLOYMENT_CLAIM_DETECTED": ["production deployment"],
}


def git_status() -> str:
    result = subprocess.run(["git", "status", "--short"], cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    return result.stdout.rstrip("\n")


def changed_paths() -> list[str]:
    return [line[3:].replace("\\", "/") for line in git_status().splitlines() if line.strip()]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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
    if "BOUNDED_LOCAL_PRIVATE_CLEAN_DEPLOY_READY_GATE_POSITIVE" not in verdicts:
        failures.append("BOUNDED_LOCAL_PRIVATE_CLEAN_DEPLOY_READY_GATE_POSITIVE_MISSING")
    for key in ["deployment_harness_gate_pass", "sdk_smoke_still_passes", "bounded_chat_service_smoke_pass", "artifact_hash_verified", "checkpoint_hash_unchanged", "rollback_pointer_written", "upstream_098_positive", "upstream_089b_positive", "upstream_088_positive"]:
        if metrics.get(key) is not True:
            failures.append(f"RELEASE_GATE_METRIC_FAIL:{key}")
    if metrics.get("fresh_harness_smoke_exit_code") != 0 or metrics.get("train_step_count") != 0:
        failures.append("FRESH_HARNESS_EXIT_OR_TRAIN_GATE_FAIL")
    boundary = load_json(SMOKE_ROOT / "claim_boundary.json")
    if boundary.get("local_private_release_ready_claimed") is not True:
        failures.append("LOCAL_PRIVATE_RELEASE_READY_NOT_CLAIMED")
    for key in ["production_deployment_claimed", "public_api_claimed", "hosted_saas_claimed", "gpt_like_assistant_readiness_claimed", "open_domain_chat_claimed", "production_chat_claimed", "safety_alignment_claimed"]:
        if boundary.get(key) is not False:
            failures.append(f"CLAIM_BOUNDARY_FAIL:{key}")
    harness_summary = SMOKE_ROOT / "deployment_harness_smoke" / "summary.json"
    if not harness_summary.exists():
        failures.append("FRESH_HARNESS_SUMMARY_MISSING")
    else:
        h = load_json(harness_summary)
        if "BOUNDED_CHAT_DEPLOYMENT_HARNESS_INTEGRATION_POSITIVE" not in set(h.get("verdicts", [])):
            failures.append("FRESH_HARNESS_POSITIVE_VERDICT_MISSING")
    failures.extend(find_false_claims((SMOKE_ROOT / "report.md").read_text(encoding="utf-8")))
    for path in changed_paths():
        if path.startswith("target/"):
            failures.append("GENERATED_ARTIFACT_STAGED")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Check STABLE_LOOP_PHASE_LOCK_099 artifacts")
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
    print("STABLE_LOOP_PHASE_LOCK_099_BOUNDED_LOCAL_PRIVATE_CLEAN_DEPLOY_READY_GATE_CHECK_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
