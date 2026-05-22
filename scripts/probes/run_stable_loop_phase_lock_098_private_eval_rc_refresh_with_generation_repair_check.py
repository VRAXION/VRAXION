#!/usr/bin/env python3
"""Static/artifact checker for STABLE_LOOP_PHASE_LOCK_098."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_098_private_eval_rc_refresh_with_generation_repair/smoke"
REQUIRED_DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_098_PRIVATE_EVAL_RC_REFRESH_WITH_GENERATION_REPAIR_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_098_PRIVATE_EVAL_RC_REFRESH_WITH_GENERATION_REPAIR_RESULT.md",
]
REQUIRED_SOURCE = [
    "scripts/probes/run_stable_loop_phase_lock_098_private_eval_rc_refresh_with_generation_repair.py",
    "scripts/probes/run_stable_loop_phase_lock_098_private_eval_rc_refresh_with_generation_repair_check.py",
]
ALLOWED_MUTATIONS = set(REQUIRED_DOCS + REQUIRED_SOURCE)
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "refresh_config.json",
    "upstream_refresh_manifest.json",
    "generation_repair_evidence_manifest.json",
    "artifact_hash_manifest.json",
    "refreshed_claim_boundary.json",
    "operator_generation_repair_delta.md",
    "acceptance_delta_checklist.md",
    "rollback_pointer.json",
    "rc_refresh_index.json",
    "private_evaluation_rc_generation_repair_refresh.zip",
    "summary.json",
    "report.md",
]
REQUIRED_TERMS = [
    "STABLE_LOOP_PHASE_LOCK_098_PRIVATE_EVAL_RC_REFRESH_WITH_GENERATION_REPAIR",
    "packaging-only private evaluation RC refresh",
    "not clean deploy proof",
    "not GPT-like assistant readiness",
    "not open-domain chat",
    "not production chat",
    "not public API",
    "not hosted SaaS",
    "not safety alignment",
    "PRIVATE_EVAL_RC_REFRESH_WITH_GENERATION_REPAIR_POSITIVE",
    "UPSTREAM_089_PACKAGE_VERIFIED",
    "GENERATION_REPAIR_PROVENANCE_INCLUDED",
    "FRESH_GENERATION_EVAL_PROVENANCE_INCLUDED",
    "MULTI_SEED_OOD_RETENTION_PROVENANCE_INCLUDED",
    "RC_REFRESH_ZIP_WRITTEN",
    "NO_TRAINING_PERFORMED",
    "NO_INFERENCE_PERFORMED",
    "PRIVATE_EVAL_RC_REFRESH_WITH_GENERATION_REPAIR_FAILS",
    "UPSTREAM_089_ARTIFACT_MISSING",
    "UPSTREAM_STACK_NOT_POSITIVE",
    "UPSTREAM_GENERATION_REPAIR_ARTIFACT_MISSING",
    "UPSTREAM_GENERATION_REPAIR_NOT_POSITIVE",
    "ARTIFACT_HASH_MISMATCH",
    "RUNTIME_SURFACE_MUTATION_DETECTED",
    "ROOT_LICENSE_CHANGED",
    "099_BOUNDED_LOCAL_PRIVATE_CLEAN_DEPLOY_READY_GATE",
]
NEGATION_MARKERS = ["not ", "no ", "does not ", "do not ", "without ", "false", "forbidden", "forbid", "reject", "rejected", "only"]
FORBIDDEN_CLAIMS = {
    "GPT_LIKE_READINESS_FALSE_CLAIM": ["GPT-like assistant readiness"],
    "OPEN_DOMAIN_CHAT_CLAIM_DETECTED": ["open-domain chat"],
    "PRODUCTION_CHAT_CLAIM_DETECTED": ["production chat"],
    "PUBLIC_API_CLAIM_DETECTED": ["public API"],
    "HOSTED_SAAS_CLAIM_DETECTED": ["hosted SaaS"],
    "SAFETY_ALIGNMENT_CLAIM_DETECTED": ["safety alignment"],
    "DEPLOYMENT_CLAIM_DETECTED": ["clean deploy proof", "production deployment"],
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
    if "PRIVATE_EVAL_RC_REFRESH_WITH_GENERATION_REPAIR_POSITIVE" not in verdicts:
        failures.append("PRIVATE_EVAL_RC_REFRESH_WITH_GENERATION_REPAIR_POSITIVE_MISSING")
    if metrics.get("packaging_only") is not True or metrics.get("train_step_count") != 0 or metrics.get("inference_run_count") != 0:
        failures.append("PACKAGING_ONLY_GATE_FAIL")
    if metrics.get("service_started") is not False or metrics.get("deployment_smoke_run") is not False:
        failures.append("SIDE_EFFECT_GATE_FAIL")
    if metrics.get("generation_repair_evidence_bound") is not True or metrics.get("refresh_zip_written") is not True:
        failures.append("REFRESH_EVIDENCE_OR_ZIP_MISSING")
    boundary = load_json(SMOKE_ROOT / "refreshed_claim_boundary.json")
    for key in ["production_deployment_claimed", "public_api_claimed", "hosted_saas_claimed", "gpt_like_assistant_readiness_claimed", "open_domain_chat_claimed", "production_chat_claimed", "safety_alignment_claimed"]:
        if boundary.get(key) is not False:
            failures.append(f"CLAIM_BOUNDARY_FAIL:{key}")
    failures.extend(find_false_claims((SMOKE_ROOT / "report.md").read_text(encoding="utf-8")))
    for path in changed_paths():
        if path.startswith("target/"):
            failures.append("GENERATED_ARTIFACT_STAGED")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Check STABLE_LOOP_PHASE_LOCK_098 artifacts")
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
    print("STABLE_LOOP_PHASE_LOCK_098_PRIVATE_EVAL_RC_REFRESH_WITH_GENERATION_REPAIR_CHECK_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
