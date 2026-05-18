#!/usr/bin/env python3
"""Static checker for STABLE_LOOP_PHASE_LOCK_086 bounded chat deployment harness integration."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

REQUIRED_DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_086_BOUNDED_CHAT_DEPLOYMENT_HARNESS_INTEGRATION_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_086_BOUNDED_CHAT_DEPLOYMENT_HARNESS_INTEGRATION_RESULT.md",
    "tools/instnct_deploy/README.md",
]

REQUIRED_SOURCE = [
    "tools/instnct_deploy/instnct_deploy.py",
    "tools/instnct_deploy/config/example.local.json",
    "scripts/probes/run_stable_loop_phase_lock_086_bounded_chat_deployment_harness_integration_check.py",
]

ALLOWED_MUTATIONS = set(REQUIRED_DOCS + REQUIRED_SOURCE)

EXACT_COMMANDS = [
    "python -m py_compile tools/instnct_deploy/instnct_deploy.py",
    "python -m py_compile scripts/probes/run_stable_loop_phase_lock_086_bounded_chat_deployment_harness_integration_check.py",
    "python tools/instnct_deploy/instnct_deploy.py validate-config --config tools/instnct_deploy/config/example.local.json",
    "python tools/instnct_deploy/instnct_deploy.py healthcheck --config tools/instnct_deploy/config/example.local.json",
    "python tools/instnct_deploy/instnct_deploy.py smoke --config tools/instnct_deploy/config/example.local.json --out target/pilot_wave/stable_loop_phase_lock_086_bounded_chat_deployment_harness_integration/smoke",
    "python scripts/probes/run_stable_loop_phase_lock_086_bounded_chat_deployment_harness_integration_check.py --check-only",
    "python scripts/probes/run_stable_loop_phase_lock_085_bounded_chat_service_api_alpha_check.py --check-only",
    "python scripts/probes/run_stable_loop_phase_lock_084_bounded_chat_inference_runtime_check.py --check-only",
    "python scripts/probes/run_stable_loop_phase_lock_083_chat_model_artifact_rc_package_check.py --check-only",
    "git diff --check",
]

BOUNDARY_TOKENS = [
    "local/private deployment harness integration only",
    "not production deployment",
    "not hosted SaaS",
    "not public beta",
    "not public API",
    "not SDK release",
    "not GPT-like assistant",
    "not open-domain chat",
    "not production chat",
    "not safety alignment",
]

REQUIRED_TERMS = [
    "STABLE_LOOP_PHASE_LOCK_086_BOUNDED_CHAT_DEPLOYMENT_HARNESS_INTEGRATION",
    "bounded_chat_service_alpha_enabled",
    "bounded_chat_service_config_path",
    "bounded_chat_service_smoke_out",
    "bounded_chat_require_085_positive",
    "target/pilot_wave/stable_loop_phase_lock_086_bounded_chat_deployment_harness_integration/",
    "tools/instnct_service_alpha/instnct_service_alpha.py",
    "python tools/instnct_service_alpha/instnct_service_alpha.py smoke --config tools/instnct_service_alpha/config/example.local.json --out",
    "queue.json",
    "progress.jsonl",
    "resolved_config.json",
    "healthcheck.json",
    "sdk_smoke_manifest.json",
    "bounded_chat_service_manifest.json",
    "bounded_chat_service_metrics.json",
    "bounded_chat_request_response.json",
    "artifact_validation.json",
    "rollback_pointer.json",
    "audit_log.jsonl",
    "summary.json",
    "report.md",
    "config -> healthcheck -> existing SDK smoke -> 085 service smoke -> artifact provenance -> audit -> rollback pointer",
    "BOUNDED_CHAT_SERVICE_API_ALPHA_POSITIVE",
    "bounded_chat_route_registered",
    "bounded_chat_child_084_positive",
    "artifact_hash_verified",
    "checkpoint_hash_unchanged",
    "train_step_count",
    "auth_required",
    "auth_rejection_has_no_child_side_effect",
    "policy_rejection_has_no_child_side_effect",
    "rate_limit_metadata_present",
    "bad_input_handled",
    "unsupported_input_handled",
    "audit_log_written",
    "child_runtime_artifacts_preserved",
    "083_artifact_package_zip_hash",
    "084_child_checkpoint_hash",
    "085_service_child_job_path",
    "086_harness_smoke_path",
    "Disable bounded_chat_service_alpha_enabled",
    "BOUNDED_CHAT_DEPLOYMENT_HARNESS_INTEGRATION_POSITIVE",
    "DEPLOYMENT_HARNESS_CONFIG_VALID",
    "DEPLOYMENT_HEALTHCHECK_PASSES",
    "SDK_SMOKE_THROUGH_HARNESS_STILL_PASSES",
    "BOUNDED_CHAT_SERVICE_SMOKE_THROUGH_HARNESS_PASSES",
    "BOUNDED_CHAT_ARTIFACT_PROVENANCE_VERIFIED",
    "CHECKPOINT_UNCHANGED_THROUGH_HARNESS",
    "AUTH_POLICY_RATE_LIMIT_THROUGH_HARNESS_PASSES",
    "ROLLBACK_POINTER_WRITTEN",
    "AUDIT_LOGGING_POSITIVE",
    "PRODUCTION_DEPLOYMENT_NOT_CLAIMED",
    "BOUNDED_CHAT_DEPLOYMENT_HARNESS_INTEGRATION_FAILS",
    "CONFIG_SCHEMA_INVALID",
    "POLICY_GUARD_REJECTS_REGULATED_DEPLOYMENT",
    "HEALTHCHECK_FAILS",
    "SDK_SMOKE_THROUGH_HARNESS_FAILS",
    "BOUNDED_CHAT_SERVICE_SMOKE_THROUGH_HARNESS_FAILS",
    "UPSTREAM_085_ARTIFACT_MISSING",
    "STALE_SERVICE_SMOKE_ARTIFACT_USED",
    "ARTIFACT_HASH_MISMATCH",
    "CHECKPOINT_MUTATION_DETECTED",
    "TRAINING_SIDE_EFFECT_DETECTED",
    "AUTH_POLICY_RATE_LIMIT_REGRESSION_DETECTED",
    "BAD_INPUT_REGRESSION_DETECTED",
    "UNSUPPORTED_INPUT_REGRESSION_DETECTED",
    "AUDIT_LOG_MISSING",
    "ROLLBACK_POINTER_MISSING",
    "PUBLIC_BIND_DETECTED",
    "PRODUCTION_DEPLOYMENT_CLAIM_DETECTED",
    "PUBLIC_API_CLAIM_DETECTED",
    "GPT_LIKE_READINESS_FALSE_CLAIM",
    "ROOT_LICENSE_CHANGED",
    "087_BOUNDED_CHAT_OOD_RED_TEAM_EVAL",
    "086B_BOUNDED_CHAT_DEPLOYMENT_HARNESS_FAILURE_ANALYSIS",
]

SOURCE_TERMS = [
    "REQUIRED_085_ARTIFACTS",
    "validate_bounded_chat_config_fields",
    "run_bounded_chat_service_smoke",
    "validate_bounded_chat_service_artifacts",
    "validate_artifacts",
    "write_rollback_pointer",
    "bounded_chat_service_child_command",
    "subprocess.run(self.bounded_chat_service_child_command",
    "BOUNDED_CHAT_SERVICE_API_ALPHA_POSITIVE",
    "STALE_SERVICE_SMOKE_ARTIFACT_USED",
    "AUTH_POLICY_RATE_LIMIT_REGRESSION_DETECTED",
    "BAD_INPUT_REGRESSION_DETECTED",
    "UNSUPPORTED_INPUT_REGRESSION_DETECTED",
    "PUBLIC_BIND_DETECTED",
    "PRODUCTION_DEPLOYMENT_CLAIM_DETECTED",
]

PLACEHOLDERS = ["TODO", "TBD", "placeholder", "pending measured", "coming soon"]

FORBIDDEN_CLAIMS = {
    "GPT_LIKE_READINESS_FALSE_CLAIM": ["GPT-like assistant", "GPT-like assistant readiness"],
    "OPEN_DOMAIN_CHAT_CLAIM_DETECTED": ["open-domain chat"],
    "PRODUCTION_CHAT_CLAIM_DETECTED": ["production chat"],
    "PUBLIC_API_CLAIM_DETECTED": ["public API"],
    "PUBLIC_BETA_CLAIM_DETECTED": ["public beta"],
    "HOSTED_SAAS_CLAIM_DETECTED": ["hosted SaaS"],
    "PRODUCTION_DEPLOYMENT_CLAIM_DETECTED": ["production deployment"],
    "SDK_RELEASE_CLAIM_DETECTED": ["SDK release"],
    "SAFETY_ALIGNMENT_CLAIM_DETECTED": ["safety alignment"],
}

NEGATION_MARKERS = ["not ", "no ", "does not ", "do not ", "without ", "false", "reject", "rejected", "only"]
GENERATED_PATH_PARTS = ["target/", "node_modules/", ".svelte-kit/"]
GENERATED_SUFFIXES = [".exe", ".dll", ".so", ".dylib", ".zip", ".tar", ".tar.gz", ".tgz", ".7z", ".ckpt", ".bin"]

POSITIVE_VERDICTS = [
    "BOUNDED_CHAT_DEPLOYMENT_HARNESS_INTEGRATION_STATIC_CHECK_POSITIVE",
    "DEPLOYMENT_HARNESS_086_FILES_WRITTEN",
    "BOUNDED_CHAT_SERVICE_CHILD_ORCHESTRATION_REQUIRED",
    "SDK_SMOKE_PRESERVATION_REQUIRED",
    "ARTIFACT_PROVENANCE_REQUIRED",
    "ROLLBACK_POINTER_REQUIRED",
    "AUDIT_LOGGING_REQUIRED",
    "NO_RUNTIME_OR_INFERENCE_DUPLICATION",
    "ROOT_LICENSE_UNCHANGED",
    "PRODUCTION_DEPLOYMENT_NOT_CLAIMED",
]


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
        if path.startswith("tools/instnct_service_alpha/"):
            return True
        if path.startswith("sdk/") or path.startswith("packages/"):
            return True
        if path.startswith("docs/releases/") or path.startswith("docs/product/"):
            return True
    return False


def generated_artifact_staged() -> bool:
    for path in changed_paths():
        if any(part in path for part in GENERATED_PATH_PARTS):
            return True
        if any(path.endswith(suffix) for suffix in GENERATED_SUFFIXES):
            return True
    return False


def config_checks(text: str) -> list[str]:
    failures: list[str] = []
    try:
        config = json.loads(text)
    except json.JSONDecodeError:
        return ["CONFIG_JSON_INVALID"]
    expected = {
        "bounded_chat_service_alpha_enabled": True,
        "bounded_chat_service_config_path": "tools/instnct_service_alpha/config/example.local.json",
        "bounded_chat_service_smoke_out": "target/pilot_wave/stable_loop_phase_lock_086_bounded_chat_deployment_harness_integration/service_smoke",
        "bounded_chat_require_085_positive": True,
        "production_default_training_enabled": False,
        "public_beta_promoted": False,
        "production_api_ready": False,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            failures.append(f"CONFIG_MISMATCH:{key}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    failures: list[str] = []
    missing, files = read_files()
    if missing:
        failures.extend([f"MISSING:{item}" for item in missing])

    joined = "\n".join(files.values())
    audited_text = "\n".join(text for rel, text in files.items() if not rel.endswith("_check.py"))
    for term in REQUIRED_TERMS + BOUNDARY_TOKENS + EXACT_COMMANDS:
        if term not in joined:
            failures.append(f"MISSING_TERM:{term}")
    source = files.get("tools/instnct_deploy/instnct_deploy.py", "")
    for term in SOURCE_TERMS:
        if term not in source:
            failures.append(f"MISSING_SOURCE_TERM:{term}")
    if "phase_lane_bounded_chat_inference_runtime" in source:
        failures.append("DIRECT_084_RUNTIME_CALL_DETECTED")
    if "shell=True" in source or "shell = True" in source:
        failures.append("COMMAND_ARGUMENT_UNSAFE")
    for marker in PLACEHOLDERS:
        if marker.lower() in audited_text.lower():
            failures.append(f"PLACEHOLDER_PRESENT:{marker}")
    failures.extend(find_false_claims(audited_text))
    failures.extend(config_checks(files.get("tools/instnct_deploy/config/example.local.json", "{}")))

    if root_license_changed():
        failures.append("ROOT_LICENSE_CHANGED")
    if runtime_surface_mutation_detected():
        failures.append("RUNTIME_SURFACE_MUTATION_DETECTED")
    if generated_artifact_staged():
        failures.append("GENERATED_ARTIFACT_STAGED")

    payload = {
        "schema_version": "bounded_chat_deployment_harness_integration_static_check_v1",
        "status": "passed" if not failures else "failed",
        "check_only": args.check_only,
        "verdicts": POSITIVE_VERDICTS if not failures else ["BOUNDED_CHAT_DEPLOYMENT_HARNESS_INTEGRATION_STATIC_CHECK_FAILS", *failures],
        "checked_files": REQUIRED_DOCS + REQUIRED_SOURCE,
        "allowed_mutations": sorted(ALLOWED_MUTATIONS),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
