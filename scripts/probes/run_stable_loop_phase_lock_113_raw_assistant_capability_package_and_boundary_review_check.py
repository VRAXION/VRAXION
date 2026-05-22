#!/usr/bin/env python3
"""Static/artifact checker for STABLE_LOOP_PHASE_LOCK_113."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_113_raw_assistant_capability_package_and_boundary_review/smoke"
REQUIRED_DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_113_RAW_ASSISTANT_CAPABILITY_PACKAGE_AND_BOUNDARY_REVIEW_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_113_RAW_ASSISTANT_CAPABILITY_PACKAGE_AND_BOUNDARY_REVIEW_RESULT.md",
]
REQUIRED_SOURCE = [
    "scripts/probes/run_stable_loop_phase_lock_113_raw_assistant_capability_package_and_boundary_review.py",
    "scripts/probes/run_stable_loop_phase_lock_113_raw_assistant_capability_package_and_boundary_review_check.py",
]
ALLOWED_MUTATIONS = set(REQUIRED_DOCS + REQUIRED_SOURCE)
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "package_config.json",
    "upstream_099_manifest.json",
    "upstream_100_manifest.json",
    "upstream_110_manifest.json",
    "upstream_111r_manifest.json",
    "upstream_111x_manifest.json",
    "upstream_112_manifest.json",
    "evidence_index.json",
    "capability_package_manifest.json",
    "raw_generation_capability_summary.json",
    "boundary_review.json",
    "claim_boundary.json",
    "readiness_denial_matrix.json",
    "release_vs_capability_separation.json",
    "validated_findings_delta.json",
    "integrity_manifest.json",
    "retention_and_lm_summary.json",
    "sample_index.json",
    "limitation_register.json",
    "human_readable_summary.md",
    "decision.json",
    "summary.json",
    "report.md",
]
REQUIRED_TERMS = [
    "STABLE_LOOP_PHASE_LOCK_113_RAW_ASSISTANT_CAPABILITY_PACKAGE_AND_BOUNDARY_REVIEW",
    "evidence package and boundary-review milestone",
    "RAW_ASSISTANT_CAPABILITY_PACKAGE_AND_BOUNDARY_REVIEW_POSITIVE",
    "UPSTREAM_112_SCALE_CONFIRM_VERIFIED",
    "EVIDENCE_CHAIN_PACKAGED",
    "CLAIM_BOUNDARY_VERIFIED",
    "RELEASE_CAPABILITY_BOUNDARY_SEPARATED",
    "VALIDATED_FINDINGS_DELTA_WRITTEN",
    "NO_TRAINING_PERFORMED",
    "NO_INFERENCE_PERFORMED",
    "114_RAW_ASSISTANT_EXTERNAL_STRESS_BENCHMARK_BRIDGE",
    "claim_boundary.json",
    "release_vs_capability_separation.json",
    "validated_findings_delta.json",
    "not GPT-like assistant readiness",
    "not open-domain assistant readiness",
    "not production chat",
    "not public API",
    "not deployment readiness",
    "not safety alignment",
]
NEGATION_MARKERS = ["not ", "no ", "does not ", "do not ", "without ", "false", "forbidden", "blocked", "denial"]
FORBIDDEN_CLAIMS = {
    "GPT_LIKE_READINESS_FALSE_CLAIM": ["GPT-like ready", "GPT-like assistant ready"],
    "OPEN_DOMAIN_ASSISTANT_CLAIM_DETECTED": ["open-domain assistant ready"],
    "PRODUCTION_CHAT_CLAIM_DETECTED": ["production chat ready", "production ready"],
    "PUBLIC_API_CLAIM_DETECTED": ["public API ready"],
    "DEPLOYMENT_CLAIM_DETECTED": ["deployment ready"],
    "SAFETY_ALIGNMENT_CLAIM_DETECTED": ["safety aligned"],
}
CLAIM_FLAGS = [
    "gpt_like_readiness_claimed",
    "open_domain_assistant_readiness_claimed",
    "production_chat_claimed",
    "public_api_claimed",
    "deployment_readiness_claimed",
    "safety_alignment_claimed",
    "hungarian_assistant_readiness_claimed",
]


def git_status() -> str:
    result = subprocess.run(["git", "status", "--short"], cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    return result.stdout.rstrip("\n")


def changed_paths() -> list[str]:
    paths: list[str] = []
    for line in git_status().splitlines():
        if not line.strip():
            continue
        path = line[3:].replace("\\", "/")
        paths.append(path)
    return paths


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def claim_is_negated(text: str, phrase: str) -> bool:
    lowered = text.lower()
    phrase_lower = phrase.lower()
    for match in re.finditer(re.escape(phrase_lower), lowered):
        window = lowered[max(0, match.start() - 220) : match.start()]
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


def check_upstream_manifest(path: Path, expected_verdict: str) -> list[str]:
    failures: list[str] = []
    manifest = load_json(path)
    if manifest.get("positive_verdict") != expected_verdict:
        failures.append(f"UPSTREAM_VERDICT_MISMATCH:{path.name}")
    for key in ["root_path", "summary_hash", "key_metrics", "boundary_flags"]:
        if key not in manifest:
            failures.append(f"UPSTREAM_EVIDENCE_INCOMPLETE:{path.name}:{key}")
    if not manifest.get("key_metrics"):
        failures.append(f"UPSTREAM_EVIDENCE_INCOMPLETE:{path.name}:key_metrics_empty")
    if not manifest.get("boundary_flags"):
        failures.append(f"UPSTREAM_EVIDENCE_INCOMPLETE:{path.name}:boundary_flags_empty")
    return failures


def check_artifacts() -> list[str]:
    failures: list[str] = []
    for rel in REQUIRED_ARTIFACTS:
        if not (SMOKE_ROOT / rel).exists():
            failures.append(f"MISSING_ARTIFACT:{rel}")
    if failures:
        return failures

    expected_upstreams = {
        "upstream_099_manifest.json": "BOUNDED_LOCAL_PRIVATE_CLEAN_DEPLOY_READY_GATE_POSITIVE",
        "upstream_100_manifest.json": "OPEN_VOCAB_ASSISTANT_CAPABILITY_SCALE_POSITIVE",
        "upstream_110_manifest.json": "INTEGRATED_DECODER_POLICY_OOD_CONFIRM_BATCH_POSITIVE",
        "upstream_111r_manifest.json": "RETENTION_OR_LM_REGRESSION_ANALYSIS_POSITIVE",
        "upstream_111x_manifest.json": "CHASSIS_DECISION_RAW_GENERATION_REDESIGN_GATE_POSITIVE",
        "upstream_112_manifest.json": "CURRENT_CHASSIS_RAW_GENERATION_SCALE_CONFIRM_POSITIVE",
    }
    for rel, verdict in expected_upstreams.items():
        failures.extend(check_upstream_manifest(SMOKE_ROOT / rel, verdict))

    summary = load_json(SMOKE_ROOT / "summary.json")
    metrics = summary.get("metrics", {})
    verdicts = set(summary.get("verdicts", []))
    if "RAW_ASSISTANT_CAPABILITY_PACKAGE_AND_BOUNDARY_REVIEW_POSITIVE" not in verdicts:
        failures.append("RAW_ASSISTANT_CAPABILITY_PACKAGE_AND_BOUNDARY_REVIEW_RESULT_MISSING")
    for key, expected in [
        ("package_and_boundary_review_only", True),
        ("training_performed", False),
        ("inference_performed", False),
        ("service_started", False),
        ("deployment_smoke_run", False),
        ("runtime_surface_mutated", False),
        ("bounded_release_stack_mutated", False),
        ("gpt_like_assistant_readiness_claimed", False),
        ("open_domain_assistant_readiness_claimed", False),
        ("production_chat_claimed", False),
        ("public_api_claimed", False),
        ("deployment_readiness_claimed", False),
        ("safety_alignment_claimed", False),
        ("hungarian_assistant_readiness_claimed", False),
    ]:
        if summary.get(key) is not expected:
            failures.append("GPT_LIKE_READINESS_FALSE_CLAIM")
    for key, expected in [
        ("train_step_count", 0),
        ("optimizer_step_count", 0),
        ("inference_run_count", 0),
        ("service_started", False),
        ("deployment_smoke_run", False),
    ]:
        if metrics.get(key) != expected:
            failures.append("TRAINING_SIDE_EFFECT_DETECTED" if "step" in key else "INFERENCE_SIDE_EFFECT_DETECTED")

    claim_boundary = load_json(SMOKE_ROOT / "claim_boundary.json")
    for key in CLAIM_FLAGS:
        if claim_boundary.get(key) is not False:
            failures.append("CLAIM_BOUNDARY_MISSING")

    release_sep_text = json.dumps(load_json(SMOKE_ROOT / "release_vs_capability_separation.json"), sort_keys=True)
    for phrase in [
        "099 proves local/private bounded release readiness",
        "112 proves raw assistant capability scale on rubric-bounded eval",
        "113 does not merge these into production/public/GPT-like readiness",
    ]:
        if phrase not in release_sep_text:
            failures.append("RELEASE_CAPABILITY_BOUNDARY_MISSING")

    delta_text = json.dumps(load_json(SMOKE_ROOT / "validated_findings_delta.json"), sort_keys=True)
    for phrase in [
        "111 standard failed",
        "111R classified mixed cause",
        "111X decided current chassis viable",
        "112 scale-confirmed current chassis raw generation",
    ]:
        if phrase not in delta_text:
            failures.append("VALIDATED_FINDINGS_DELTA_MISSING")

    decision = load_json(SMOKE_ROOT / "decision.json")
    if decision.get("next") != "114_RAW_ASSISTANT_EXTERNAL_STRESS_BENCHMARK_BRIDGE":
        failures.append("DECISION_NEXT_MISMATCH")
    for key in ["reason", "prerequisites_satisfied", "blocked_claims", "recommended_scope"]:
        if not decision.get(key):
            failures.append("DECISION_MISSING_FIELD")

    human = (SMOKE_ROOT / "human_readable_summary.md").read_text(encoding="utf-8")
    for heading in ["What is proven", "What is not proven", "What remains risky", "Why 114 is next"]:
        if heading not in human:
            failures.append("HUMAN_READABLE_SUMMARY_INCOMPLETE")

    text_to_scan = "\n".join(
        [
            human,
            (SMOKE_ROOT / "report.md").read_text(encoding="utf-8"),
            json.dumps(summary, sort_keys=True),
            json.dumps(claim_boundary, sort_keys=True),
            release_sep_text,
        ]
    )
    failures.extend(find_false_claims(text_to_scan))
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    failures: list[str] = []

    missing, files = read_files()
    failures.extend(f"MISSING_REQUIRED_FILE:{item}" for item in missing)
    combined = "\n".join(files.values())
    for term in REQUIRED_TERMS:
        if term not in combined:
            failures.append(f"MISSING_REQUIRED_TERM:{term}")
    docs_text = "\n".join(files.get(rel, "") for rel in REQUIRED_DOCS)
    failures.extend(find_false_claims(docs_text))
    if runtime_surface_mutation_detected():
        failures.append("RUNTIME_SURFACE_MUTATION_DETECTED")
    if (REPO_ROOT / "LICENSE").exists() and "LICENSE" in changed_paths():
        failures.append("ROOT_LICENSE_CHANGED")
    if args.check_only:
        failures.extend(check_artifacts())

    if failures:
        print("113 checker failed:")
        for failure in sorted(set(failures)):
            print(f" - {failure}")
        return 1
    print("113 checker passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
