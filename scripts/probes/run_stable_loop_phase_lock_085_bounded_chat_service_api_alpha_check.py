#!/usr/bin/env python3
"""Static checker for STABLE_LOOP_PHASE_LOCK_085 bounded chat service API alpha."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

REQUIRED_DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_085_BOUNDED_CHAT_SERVICE_API_ALPHA_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_085_BOUNDED_CHAT_SERVICE_API_ALPHA_RESULT.md",
    "tools/instnct_service_alpha/README.md",
]

REQUIRED_SOURCE = [
    "tools/instnct_service_alpha/instnct_service_alpha.py",
    "tools/instnct_service_alpha/config/example.local.json",
    "scripts/probes/run_stable_loop_phase_lock_085_bounded_chat_service_api_alpha_check.py",
]

ALLOWED_MUTATIONS = set(REQUIRED_DOCS + REQUIRED_SOURCE)

EXACT_COMMANDS = [
    "python -m py_compile tools/instnct_service_alpha/instnct_service_alpha.py",
    "python -m py_compile scripts/probes/run_stable_loop_phase_lock_085_bounded_chat_service_api_alpha_check.py",
    "python tools/instnct_service_alpha/instnct_service_alpha.py healthcheck --config tools/instnct_service_alpha/config/example.local.json",
    "python tools/instnct_service_alpha/instnct_service_alpha.py smoke --config tools/instnct_service_alpha/config/example.local.json --out target/pilot_wave/stable_loop_phase_lock_085_bounded_chat_service_api_alpha/smoke",
    "python scripts/probes/run_stable_loop_phase_lock_085_bounded_chat_service_api_alpha_check.py --check-only",
    "python scripts/probes/run_stable_loop_phase_lock_084_bounded_chat_inference_runtime_check.py --check-only",
    "python scripts/probes/run_stable_loop_phase_lock_083_chat_model_artifact_rc_package_check.py --check-only",
    "python scripts/probes/run_stable_loop_phase_lock_062_service_api_alpha_check.py --check-only",
    "git diff --check",
]

BOUNDARY_TOKENS = [
    "service API alpha only",
    "localhost/private only",
    "not deploy-ready service",
    "not public API",
    "not SDK surface",
    "not GPT-like assistant",
    "not open-domain chat",
    "not production chat",
    "not safety alignment",
    "not public beta / GA / hosted SaaS",
    "127.0.0.1",
    "POST /v1/bounded-chat/infer",
]

REQUIRED_TERMS = [
    "STABLE_LOOP_PHASE_LOCK_085_BOUNDED_CHAT_SERVICE_API_ALPHA",
    "POST /v1/bounded-chat/infer",
    "phase_lane_bounded_chat_inference_runtime",
    "cargo run -p instnct-core --example phase_lane_bounded_chat_inference_runtime",
    "single_inference.json",
    "runtime_metrics.json",
    "summary.json",
    "report.md",
    "audit_log.jsonl",
    "bounded_chat_artifact_root",
    "bounded_chat_runtime_out_root",
    "bounded_chat_max_input_chars",
    "bounded_chat_max_response_tokens",
    "bounded_chat_timeout_ms",
    "target/pilot_wave/stable_loop_phase_lock_083_chat_model_artifact_rc_package/smoke",
    "target/pilot_wave/stable_loop_phase_lock_085_bounded_chat_service_api_alpha/",
    "queue.json",
    "progress.jsonl",
    "service_config_resolved.json",
    "route_manifest.json",
    "bounded_chat_request_response.json",
    "bad_input_results.jsonl",
    "unsupported_input_results.jsonl",
    "auth_policy_results.jsonl",
    "rate_limit_report.json",
    "child_runtime_manifest.json",
    "service_metrics.json",
    "ok",
    "value",
    "error",
    "request_id",
    "idempotency_key",
    "route",
    "rate_limit",
    "artifact_hash",
    "child_job_path",
    "value.inference",
    "auth_result",
    "policy_result",
    "prompt_sha256",
    "checkpoint_sha256",
    "artifact_package_zip_sha256",
    "missing prompt",
    "non-string prompt",
    "empty prompt",
    "whitespace prompt",
    "oversized prompt",
    "malformed JSON",
    "invalid max_response_tokens",
    "unsupported topic",
    "status = unsupported",
    "artifact_hash_verified = true",
    "checkpoint_hash_unchanged = true",
    "train_step_count = 0",
    "localhost_bind_only = true",
    "public_bind_rejected = true",
    "auth_required = true",
    "auth_rejection_has_no_child_side_effect = true",
    "policy_rejection_has_no_child_side_effect = true",
    "rate_limit_metadata_present = true",
    "bounded_chat_route_registered = true",
    "bounded_chat_single_prompt_pass = true",
    "bounded_chat_json_envelope_pass = true",
    "bounded_chat_child_084_positive = true",
    "unsupported_input_handled = true",
    "bad_input_handled = true",
    "timeout_guard_pass = true",
    "idempotency_reuse_pass = true",
    "idempotency_conflict_pass = true",
    "audit_log_written = true",
    "child_runtime_artifacts_preserved = true",
    "existing_062_routes_preserved = true",
    "service_api_alpha_only = true",
    "BOUNDED_CHAT_SERVICE_API_ALPHA_POSITIVE",
    "LOCALHOST_BIND_RESTRICTED",
    "AUTH_GUARD_PASSES",
    "POLICY_GUARD_PASSES",
    "RATE_LIMIT_METADATA_PASSES",
    "BOUNDED_CHAT_ROUTE_REGISTERED",
    "BOUNDED_CHAT_INFERENCE_CHILD_RUNTIME_PASSES",
    "ARTIFACT_PACKAGE_VERIFIED_BY_CHILD",
    "CHECKPOINT_UNCHANGED",
    "JSON_RESPONSE_ENVELOPE_PASSES",
    "BAD_INPUT_HANDLED",
    "UNSUPPORTED_INPUT_HANDLED",
    "AUDIT_LOG_WRITTEN",
    "NO_TRAINING_PERFORMED",
    "PRODUCTION_CHAT_NOT_CLAIMED",
    "BOUNDED_CHAT_SERVICE_API_ALPHA_FAILS",
    "UPSTREAM_083_ARTIFACT_MISSING",
    "UPSTREAM_084_RUNTIME_MISSING",
    "PUBLIC_BIND_DETECTED",
    "SERVICE_API_PUBLIC_EXPOSURE_DETECTED",
    "AUTH_GUARD_MISSING",
    "AUTHZ_SIDE_EFFECT_LEAK",
    "POLICY_REJECTION_SIDE_EFFECT_LEAK",
    "RATE_LIMIT_BOUNDARY_MISSING",
    "BOUNDED_CHAT_ROUTE_MISSING",
    "BOUNDED_CHAT_INFERENCE_CHILD_RUNTIME_FAILS",
    "TIMEOUT_GUARD_FAILS",
    "ARTIFACT_HASH_MISMATCH",
    "CHECKPOINT_MUTATION_DETECTED",
    "JSON_RESPONSE_ENVELOPE_MISSING",
    "BAD_INPUT_NOT_HANDLED",
    "UNSUPPORTED_INPUT_NOT_HANDLED",
    "AUDIT_LOG_MISSING",
    "TRAINING_SIDE_EFFECT_DETECTED",
    "ORACLE_SHORTCUT_DETECTED",
    "SDK_PUBLIC_EXPORT_MUTATION_DETECTED",
    "DEPLOYMENT_HARNESS_MUTATION_DETECTED",
    "RUNTIME_SURFACE_MUTATION_DETECTED",
    "GPT_LIKE_READINESS_FALSE_CLAIM",
    "PRODUCTION_CHAT_CLAIM_DETECTED",
    "ROOT_LICENSE_CHANGED",
    "086_BOUNDED_CHAT_DEPLOYMENT_HARNESS_INTEGRATION",
    "085B_BOUNDED_CHAT_SERVICE_API_ALPHA_FAILURE_ANALYSIS",
]

SOURCE_TERMS = [
    '"POST /v1/bounded-chat/infer"',
    "route_bounded_chat_infer",
    "create_bounded_chat_job",
    "run_bounded_chat_job",
    "subprocess.run(",
    "timeout=process_timeout_sec",
    "BOUNDED_CHAT_INFERENCE_CHILD_RUNTIME_FAILS",
    "artifact_hash_verified",
    "checkpoint_hash_unchanged",
    "train_step_count",
    "service_audit",
    "service_progress",
    "validate_bind_host",
    "AUTH_REQUIRED",
    "POLICY_GUARD_REJECTED",
    "IDEMPOTENCY_CONFLICT",
    "RATE_LIMIT_EXCEEDED",
]

PLACEHOLDERS = ["TODO", "TBD", "placeholder", "pending measured", "coming soon"]

FORBIDDEN_CLAIMS = {
    "GPT_LIKE_READINESS_FALSE_CLAIM": ["GPT-like assistant", "GPT-like assistant readiness"],
    "OPEN_DOMAIN_CHAT_CLAIM_DETECTED": ["open-domain chat"],
    "PRODUCTION_CHAT_CLAIM_DETECTED": ["production chat"],
    "SAFETY_ALIGNMENT_CLAIM_DETECTED": ["safety alignment"],
    "PUBLIC_API_CLAIM_DETECTED": ["public API"],
    "PUBLIC_BETA_CLAIM_DETECTED": ["public beta"],
    "HOSTED_SAAS_CLAIM_DETECTED": ["hosted SaaS"],
    "DEPLOY_READY_SERVICE_CLAIM_DETECTED": ["deploy-ready service"],
    "SDK_PUBLIC_EXPORT_MUTATION_DETECTED": ["SDK surface"],
}

NEGATION_MARKERS = ["not ", "no ", "does not ", "do not ", "without ", "false", "reject", "rejected", "only"]
GENERATED_PATH_PARTS = ["target/", "node_modules/", ".svelte-kit/"]
GENERATED_SUFFIXES = [".exe", ".dll", ".so", ".dylib", ".zip", ".tar", ".tar.gz", ".tgz", ".7z", ".ckpt", ".bin"]

POSITIVE_VERDICTS = [
    "BOUNDED_CHAT_SERVICE_API_ALPHA_STATIC_CHECK_POSITIVE",
    "BOUNDED_CHAT_SERVICE_API_ALPHA_FILES_WRITTEN",
    "LOCALHOST_ONLY_SERVICE_GUARD_WRITTEN",
    "AUTH_POLICY_RATE_LIMIT_REQUIRED",
    "CHILD_084_RUNTIME_REQUIRED",
    "STRICT_SERVICE_ENVELOPE_REQUIRED",
    "BAD_AND_UNSUPPORTED_INPUT_REQUIRED",
    "AUDIT_AND_CHILD_ARTIFACTS_REQUIRED",
    "EXISTING_062_ROUTES_PRESERVED",
    "ROOT_LICENSE_UNCHANGED",
    "PRODUCTION_CHAT_NOT_CLAIMED",
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
        window = lowered[max(0, match.start() - 80) : match.start()]
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
        if path.startswith("tools/instnct_deploy/"):
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
    if config.get("bind_host") != "127.0.0.1":
        failures.append("PUBLIC_BIND_DETECTED")
    for key in [
        "bounded_chat_artifact_root",
        "bounded_chat_runtime_out_root",
        "bounded_chat_max_input_chars",
        "bounded_chat_max_response_tokens",
        "bounded_chat_timeout_ms",
    ]:
        if key not in config:
            failures.append(f"CONFIG_MISSING:{key}")
    for key in ["production_default_training_enabled", "public_beta_promoted", "production_api_ready"]:
        if config.get(key) is not False:
            failures.append(f"PRODUCTION_FLAG_NOT_FALSE:{key}")
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
    source = files.get("tools/instnct_service_alpha/instnct_service_alpha.py", "")
    for term in SOURCE_TERMS:
        if term not in source:
            failures.append(f"MISSING_SOURCE_TERM:{term}")
    if "shell=True" in source or "shell = True" in source:
        failures.append("COMMAND_ARGUMENT_UNSAFE")
    for marker in PLACEHOLDERS:
        if marker.lower() in audited_text.lower():
            failures.append(f"PLACEHOLDER_PRESENT:{marker}")
    failures.extend(find_false_claims(audited_text))
    failures.extend(config_checks(files.get("tools/instnct_service_alpha/config/example.local.json", "{}")))

    if root_license_changed():
        failures.append("ROOT_LICENSE_CHANGED")
    if runtime_surface_mutation_detected():
        failures.append("RUNTIME_SURFACE_MUTATION_DETECTED")
    if generated_artifact_staged():
        failures.append("GENERATED_ARTIFACT_STAGED")

    payload = {
        "schema_version": "bounded_chat_service_api_alpha_static_check_v1",
        "status": "passed" if not failures else "failed",
        "check_only": args.check_only,
        "verdicts": POSITIVE_VERDICTS if not failures else ["BOUNDED_CHAT_SERVICE_API_ALPHA_STATIC_CHECK_FAILS", *failures],
        "checked_files": REQUIRED_DOCS + REQUIRED_SOURCE,
        "allowed_mutations": sorted(ALLOWED_MUTATIONS),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
