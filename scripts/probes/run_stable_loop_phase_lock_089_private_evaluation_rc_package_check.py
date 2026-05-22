#!/usr/bin/env python3
"""Static/artifact checker for STABLE_LOOP_PHASE_LOCK_089 private evaluation RC package."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import hashlib
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_089_private_evaluation_rc_package/smoke"

REQUIRED_DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_089_PRIVATE_EVALUATION_RC_PACKAGE_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_089_PRIVATE_EVALUATION_RC_PACKAGE_RESULT.md",
]

REQUIRED_SOURCE = [
    "scripts/probes/run_stable_loop_phase_lock_089_private_evaluation_rc_package.py",
    "scripts/probes/run_stable_loop_phase_lock_089_private_evaluation_rc_package_check.py",
]

ALLOWED_MUTATIONS = set(REQUIRED_DOCS + REQUIRED_SOURCE)

REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "package_config.json",
    "upstream_stack_manifest.json",
    "artifact_hash_manifest.json",
    "private_eval_capability_surface.json",
    "private_eval_known_limitations.json",
    "claim_boundary.json",
    "operator_quickstart.md",
    "operator_runbook.md",
    "one_command_smoke.ps1",
    "sample_prompts_expected_outputs.jsonl",
    "audit_and_log_locations.json",
    "rollback_pointer.json",
    "troubleshooting.md",
    "acceptance_checklist.md",
    "rc_package_index.json",
    "private_evaluation_rc_package.zip",
    "summary.json",
    "report.md",
]

EXACT_COMMANDS = [
    "python -m py_compile scripts/probes/run_stable_loop_phase_lock_089_private_evaluation_rc_package.py",
    "python -m py_compile scripts/probes/run_stable_loop_phase_lock_089_private_evaluation_rc_package_check.py",
    "python scripts/probes/run_stable_loop_phase_lock_089_private_evaluation_rc_package.py --out target/pilot_wave/stable_loop_phase_lock_089_private_evaluation_rc_package/smoke --upstream-083-root target/pilot_wave/stable_loop_phase_lock_083_chat_model_artifact_rc_package/smoke --upstream-084-root target/pilot_wave/stable_loop_phase_lock_084_bounded_chat_inference_runtime/smoke --upstream-085-root target/pilot_wave/stable_loop_phase_lock_085_bounded_chat_service_api_alpha/smoke --upstream-086-root target/pilot_wave/stable_loop_phase_lock_086_bounded_chat_deployment_harness_integration/smoke --upstream-087-root target/pilot_wave/stable_loop_phase_lock_087_bounded_chat_ood_red_team_eval/smoke --upstream-088-root target/pilot_wave/stable_loop_phase_lock_088_bounded_chat_long_run_concurrency_resource_stability/smoke --heartbeat-sec 20",
    "python scripts/probes/run_stable_loop_phase_lock_089_private_evaluation_rc_package_check.py --check-only",
    "python scripts/probes/run_stable_loop_phase_lock_088_bounded_chat_long_run_concurrency_resource_stability_check.py --check-only",
    "python scripts/probes/run_stable_loop_phase_lock_087_bounded_chat_ood_red_team_eval_check.py --check-only",
    "git diff --check",
]

REQUIRED_TERMS = [
    "STABLE_LOOP_PHASE_LOCK_089_PRIVATE_EVALUATION_RC_PACKAGE",
    "083 artifact RC",
    "084 local runtime",
    "085 localhost API alpha",
    "086 harness",
    "087 OOD/red-team",
    "088 long-run stability",
    "packaging-only",
    "target/pilot_wave/stable_loop_phase_lock_089_private_evaluation_rc_package/",
    "CHAT_MODEL_ARTIFACT_RC_PACKAGE_POSITIVE",
    "BOUNDED_CHAT_INFERENCE_RUNTIME_POSITIVE",
    "BOUNDED_CHAT_SERVICE_API_ALPHA_POSITIVE",
    "BOUNDED_CHAT_DEPLOYMENT_HARNESS_INTEGRATION_POSITIVE",
    "BOUNDED_CHAT_OOD_RED_TEAM_EVAL_POSITIVE",
    "BOUNDED_CHAT_LONG_RUN_CONCURRENCY_RESOURCE_STABILITY_POSITIVE",
    "total_requests = 240",
    "completed_requests = 240",
    "audit_log_coverage_rate = 1.0",
    "child_job_orphan_count = 0",
    "checkpoint_hash_unchanged = true",
    "direct_model_runner_used = false",
    "train_step_count = 0",
    "source_083_artifact_zip_sha256",
    "packaged_083_artifact_zip_sha256",
    "private_evaluation_rc_package_zip_sha256",
    "queue.json",
    "progress.jsonl",
    "package_config.json",
    "upstream_stack_manifest.json",
    "artifact_hash_manifest.json",
    "private_eval_capability_surface.json",
    "private_eval_known_limitations.json",
    "claim_boundary.json",
    "operator_quickstart.md",
    "operator_runbook.md",
    "one_command_smoke.ps1",
    "sample_prompts_expected_outputs.jsonl",
    "audit_and_log_locations.json",
    "rollback_pointer.json",
    "troubleshooting.md",
    "acceptance_checklist.md",
    "rc_package_index.json",
    "private_evaluation_rc_package.zip",
    "PRIVATE_EVALUATION_RC_PACKAGE_POSITIVE",
    "UPSTREAM_STACK_PROVENANCE_VERIFIED",
    "MODEL_ARTIFACT_HASH_BOUND",
    "LOCAL_RUNTIME_PROVENANCE_INCLUDED",
    "SERVICE_API_ALPHA_PROVENANCE_INCLUDED",
    "HARNESS_PROVENANCE_INCLUDED",
    "OOD_RED_TEAM_PROVENANCE_INCLUDED",
    "LONG_RUN_STABILITY_PROVENANCE_INCLUDED",
    "OPERATOR_RUNBOOK_WRITTEN",
    "ONE_COMMAND_SMOKE_WRITTEN",
    "ROLLBACK_POINTER_WRITTEN",
    "RC_ZIP_WRITTEN",
    "NO_TRAINING_PERFORMED",
    "NO_INFERENCE_PERFORMED",
    "PRODUCTION_CHAT_NOT_CLAIMED",
    "PRIVATE_EVALUATION_RC_PACKAGE_FAILS",
    "UPSTREAM_STACK_NOT_POSITIVE",
    "ARTIFACT_HASH_MISMATCH",
    "CHECKPOINT_MUTATION_DETECTED",
    "TRAINING_SIDE_EFFECT_DETECTED",
    "INFERENCE_SIDE_EFFECT_DETECTED",
    "SERVICE_STARTED_UNEXPECTEDLY",
    "OPERATOR_RUNBOOK_MISSING",
    "ONE_COMMAND_SMOKE_MISSING",
    "ROLLBACK_POINTER_MISSING",
    "KNOWN_LIMITATIONS_MISSING",
    "CLAIM_BOUNDARY_MISSING",
    "RC_ZIP_MISSING",
    "RUNTIME_SURFACE_MUTATION_DETECTED",
    "ROOT_LICENSE_CHANGED",
    "GPT_LIKE_READINESS_FALSE_CLAIM",
    "PRODUCTION_CHAT_CLAIM_DETECTED",
    "PUBLIC_API_CLAIM_DETECTED",
    "HOSTED_SAAS_CLAIM_DETECTED",
    "090_BOUNDED_LOCAL_PRIVATE_DEPLOY_READY_GATE",
]

BOUNDARY_TOKENS = [
    "not clean deploy proof",
    "not production deployment",
    "not a public API",
    "not hosted SaaS",
    "not GPT-like assistant readiness",
    "not open-domain chat",
    "not safety alignment",
]

FORBIDDEN_CLAIMS = {
    "GPT_LIKE_READINESS_FALSE_CLAIM": ["GPT-like assistant", "GPT-like assistant readiness"],
    "OPEN_DOMAIN_CHAT_CLAIM_DETECTED": ["open-domain chat"],
    "PRODUCTION_CHAT_CLAIM_DETECTED": ["production chat"],
    "PUBLIC_API_CLAIM_DETECTED": ["public API"],
    "PRODUCTION_DEPLOYMENT_CLAIM_DETECTED": ["production deployment"],
    "HOSTED_SAAS_CLAIM_DETECTED": ["hosted SaaS"],
    "SAFETY_ALIGNMENT_CLAIM_DETECTED": ["safety alignment"],
}
NEGATION_MARKERS = ["not ", "no ", "does not ", "do not ", "without ", "false", "reject", "rejected", "only"]
PLACEHOLDERS = ["TODO", "TBD", "placeholder", "pending measured", "coming soon"]
GENERATED_PATH_PARTS = ["target/", "node_modules/", ".svelte-kit/"]
GENERATED_SUFFIXES = [".exe", ".dll", ".so", ".dylib", ".zip", ".tar", ".tar.gz", ".tgz", ".7z", ".ckpt", ".bin"]


def git_status(paths: list[str] | None = None) -> str:
    cmd = ["git", "status", "--short"]
    if paths:
        cmd.extend(["--", *paths])
    result = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    return result.stdout.rstrip("\n")


def changed_paths() -> list[str]:
    paths: list[str] = []
    for raw in git_status().splitlines():
        if raw.strip():
            paths.append(raw[3:].replace("\\", "/"))
    return paths


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


def claim_is_negated(text: str, phrase: str) -> bool:
    lowered = text.lower()
    phrase_lower = phrase.lower()
    for match in re.finditer(re.escape(phrase_lower), lowered):
        window = lowered[max(0, match.start() - 90) : match.start()]
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


def root_license_changed() -> bool:
    return bool(git_status(["LICENSE"]))


def runtime_surface_mutation_detected() -> bool:
    for path in changed_paths():
        if path in ALLOWED_MUTATIONS:
            continue
        if path == "LICENSE":
            return True
        if path.startswith("instnct-core/"):
            return True
        if path.startswith("tools/instnct_service_alpha/") or path.startswith("tools/instnct_deploy/"):
            return True
        if path.startswith("docs/product/") or path.startswith("docs/releases/"):
            return True
        if path.startswith("sdk/") or path.startswith("packages/"):
            return True
    return False


def generated_artifact_staged() -> bool:
    for path in changed_paths():
        if any(part in path for part in GENERATED_PATH_PARTS):
            return True
        if any(path.endswith(suffix) for suffix in GENERATED_SUFFIXES):
            return True
    return False


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def artifact_checks() -> list[str]:
    failures: list[str] = []
    for rel in REQUIRED_ARTIFACTS:
        if not (SMOKE_ROOT / rel).exists():
            failures.append(f"SMOKE_ARTIFACT_MISSING:{rel}")
    if failures:
        return failures
    summary = load_json(SMOKE_ROOT / "summary.json")
    metrics = summary.get("metrics", {})
    if summary.get("status") != "positive":
        failures.append("SMOKE_NOT_POSITIVE")
    if "PRIVATE_EVALUATION_RC_PACKAGE_POSITIVE" not in summary.get("verdicts", []):
        failures.append("POSITIVE_VERDICT_MISSING")
    expected_true = [
        "packaging_only",
        "checkpoint_hash_unchanged",
        "artifact_hash_verified",
        "all_upstreams_positive",
        "operator_runbook_present",
        "one_command_smoke_present",
        "rollback_pointer_present",
        "known_limitations_present",
        "claim_boundary_present",
        "sample_prompts_outputs_present",
        "private_eval_rc_zip_written",
    ]
    for key in expected_true:
        value = metrics.get(key, summary.get(key))
        if value is not True:
            failures.append(f"METRIC_NOT_TRUE:{key}")
    for key in ["train_step_count", "inference_run_count"]:
        if summary.get(key, metrics.get(key)) != 0:
            failures.append(f"METRIC_NOT_ZERO:{key}")
    for key in ["service_started", "deployment_smoke_run", "production_deployment_claimed", "public_api_claimed", "hosted_saas_claimed", "gpt_like_assistant_readiness_claimed", "open_domain_chat_claimed", "safety_alignment_claimed"]:
        if summary.get(key, metrics.get(key)) is not False:
            failures.append(f"METRIC_NOT_FALSE:{key}")
    hashes = load_json(SMOKE_ROOT / "artifact_hash_manifest.json")
    if hashes.get("source_083_artifact_zip_sha256") != hashes.get("packaged_083_artifact_zip_sha256"):
        failures.append("ARTIFACT_HASH_MISMATCH")
    if not hashes.get("private_evaluation_rc_package_zip_sha256"):
        failures.append("RC_ZIP_HASH_MISSING")
    elif sha256_file(SMOKE_ROOT / "private_evaluation_rc_package.zip") != hashes.get("private_evaluation_rc_package_zip_sha256"):
        failures.append("RC_ZIP_HASH_MISMATCH")
    claim_boundary = load_json(SMOKE_ROOT / "claim_boundary.json")
    for key in ["production_deployment_claimed", "public_api_claimed", "hosted_saas_claimed", "gpt_like_assistant_readiness_claimed", "open_domain_chat_claimed", "safety_alignment_claimed", "clinical_high_stakes_claimed", "deploy_ready_by_itself"]:
        if claim_boundary.get(key) is not False:
            failures.append(f"CLAIM_BOUNDARY_FLAG_NOT_FALSE:{key}")
    limitations = " ".join(load_json(SMOKE_ROOT / "private_eval_known_limitations.json").get("limitations", []))
    for phrase in ["bounded domain only", "local/private only", "no open-domain chat", "no GPT-like assistant readiness", "no Hungarian chat proof", "no long multi-turn proof", "no production safety alignment", "no public API", "no hosted SaaS", "no clinical/high-stakes use", "current latency is not production-throughput evidence"]:
        if phrase not in limitations:
            failures.append(f"LIMITATION_MISSING:{phrase}")
    sample_rows = read_jsonl(SMOKE_ROOT / "sample_prompts_expected_outputs.jsonl")
    if len(sample_rows) < 6:
        failures.append("SAMPLE_PROMPTS_OUTPUTS_MISSING")
    for field in ["prompt", "expected_status", "expected_behavior", "example_output_or_pattern", "source_upstream", "claim_boundary_note"]:
        if not all(field in row for row in sample_rows):
            failures.append(f"SAMPLE_FIELD_MISSING:{field}")
    for rel in ["operator_quickstart.md", "operator_runbook.md", "one_command_smoke.ps1", "acceptance_checklist.md", "rollback_pointer.json"]:
        text = (SMOKE_ROOT / rel).read_text(encoding="utf-8")
        if len(text.strip()) < 200:
            failures.append(f"OPERATOR_ARTIFACT_TOO_SHORT:{rel}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    failures: list[str] = []
    missing, files = read_files()
    failures.extend([f"MISSING:{item}" for item in missing])
    joined = "\n".join(files.values())
    audited_text = "\n".join(text for rel, text in files.items() if rel in REQUIRED_DOCS)
    for term in REQUIRED_TERMS + BOUNDARY_TOKENS + EXACT_COMMANDS:
        if term not in joined:
            failures.append(f"MISSING_TERM:{term}")
    source = files.get("scripts/probes/run_stable_loop_phase_lock_089_private_evaluation_rc_package.py", "")
    for forbidden in ["subprocess.run", "Popen", "phase_lane_bounded_chat_inference_runtime", "cargo run"]:
        if forbidden in source:
            failures.append(f"PACKAGING_ONLY_FORBIDDEN_SOURCE_TERM:{forbidden}")
    for marker in PLACEHOLDERS:
        if marker.lower() in audited_text.lower():
            failures.append(f"PLACEHOLDER_PRESENT:{marker}")
    failures.extend(find_false_claims(audited_text))
    failures.extend(artifact_checks())
    if root_license_changed():
        failures.append("ROOT_LICENSE_CHANGED")
    if runtime_surface_mutation_detected():
        failures.append("RUNTIME_SURFACE_MUTATION_DETECTED")
    if generated_artifact_staged():
        failures.append("GENERATED_ARTIFACT_STAGED")

    payload = {
        "schema_version": "private_evaluation_rc_package_check_v1",
        "status": "passed" if not failures else "failed",
        "check_only": args.check_only,
        "verdicts": [
            "PRIVATE_EVALUATION_RC_PACKAGE_STATIC_CHECK_POSITIVE",
            "PRIVATE_EVALUATION_RC_PACKAGE_FILES_WRITTEN",
            "PACKAGING_ONLY_HARD_WALL_CHECKED",
            "UPSTREAM_STACK_PROVENANCE_CHECKED",
            "MODEL_ARTIFACT_HASH_BOUND_CHECKED",
            "OPERATOR_MATERIALS_CHECKED",
            "CLAIM_BOUNDARY_CHECKED",
            "ROOT_LICENSE_UNCHANGED",
            "PRODUCTION_CHAT_NOT_CLAIMED",
        ]
        if not failures
        else ["PRIVATE_EVALUATION_RC_PACKAGE_STATIC_CHECK_FAILS", *failures],
        "checked_files": REQUIRED_DOCS + REQUIRED_SOURCE,
        "allowed_mutations": sorted(ALLOWED_MUTATIONS),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
