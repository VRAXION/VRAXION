#!/usr/bin/env python3
"""Static checker for STABLE_LOOP_PHASE_LOCK_084 bounded chat inference runtime."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

REQUIRED_DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_084_BOUNDED_CHAT_INFERENCE_RUNTIME_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_084_BOUNDED_CHAT_INFERENCE_RUNTIME_RESULT.md",
]

REQUIRED_SOURCE = [
    "instnct-core/examples/phase_lane_bounded_chat_inference_runtime.rs",
    "scripts/probes/run_stable_loop_phase_lock_084_bounded_chat_inference_runtime_check.py",
]

ALLOWED_MUTATIONS = set(REQUIRED_DOCS + REQUIRED_SOURCE)

EXACT_COMMANDS = [
    "cargo check -p instnct-core --example phase_lane_bounded_chat_inference_runtime",
    "cargo run -p instnct-core --example phase_lane_bounded_chat_inference_runtime -- --out target/pilot_wave/stable_loop_phase_lock_084_bounded_chat_inference_runtime/smoke --artifact-root target/pilot_wave/stable_loop_phase_lock_083_chat_model_artifact_rc_package/smoke --max-input-chars 512 --max-response-tokens 64 --timeout-ms 1000 --heartbeat-sec 20",
    "python -m py_compile scripts/probes/run_stable_loop_phase_lock_084_bounded_chat_inference_runtime_check.py",
    "python scripts/probes/run_stable_loop_phase_lock_084_bounded_chat_inference_runtime_check.py --check-only",
    "python scripts/probes/run_stable_loop_phase_lock_083_chat_model_artifact_rc_package_check.py --check-only",
    "python scripts/probes/run_stable_loop_phase_lock_082_chat_diversity_multi_seed_confirm_check.py --check-only",
    "git diff --check",
]

BOUNDARY_TOKENS = [
    "bounded local inference runtime only",
    "not deploy-ready service",
    "not service API",
    "not SDK surface",
    "not GPT-like assistant",
    "not open-domain chat",
    "not production chat",
    "not safety alignment",
    "not public beta / GA / hosted SaaS",
    "no service API change",
    "no network listener",
    "no deployment harness change",
    "no SDK/public export change",
    "no release docs change",
    "no root LICENSE change",
    "no checkpoint mutation",
]

REQUIRED_TERMS = [
    "STABLE_LOOP_PHASE_LOCK_084_BOUNDED_CHAT_INFERENCE_RUNTIME",
    "phase_lane_bounded_chat_inference_runtime.rs",
    "run_stable_loop_phase_lock_084_bounded_chat_inference_runtime_check.py",
    "target/pilot_wave/stable_loop_phase_lock_084_bounded_chat_inference_runtime/",
    "target/pilot_wave/stable_loop_phase_lock_083_chat_model_artifact_rc_package/smoke",
    "--out",
    "--artifact-root",
    "--prompt",
    "--batch-in",
    "--max-input-chars",
    "--max-response-tokens",
    "--timeout-ms",
    "--json",
    "--heartbeat-sec",
    "artifact_index.json present",
    "integrity_hashes.json present",
    "capability_surface.json present",
    "claim_boundary.json present",
    "packaged checkpoint exists",
    "checkpoint sha256 matches integrity_hashes.json",
    "checkpoint size matches integrity_hashes.json",
    "083 summary contains CHAT_MODEL_ARTIFACT_RC_PACKAGE_POSITIVE",
    "checkpoint_hash_before",
    "checkpoint_hash_after",
    "checkpoint_hash_unchanged = true",
    "train_step_count = 0",
    "prediction_oracle_used = false",
    "llm_judge_used = false",
    "service_api_exposed = false",
    "deployment_harness_exposed = false",
    "sdk_surface_exposed = false",
    "route explanation",
    "stale/old packet explanation",
    "active/distractor/old/stale/inactive slot binding",
    "two-turn active-code carry",
    "boundary mini refusal",
    "anti-template-copy explanation",
    "finite-label AnchorRoute retention",
    "status = unsupported",
    "request_id",
    "prompt_sha256",
    "status",
    "output_text",
    "output_classification",
    "supported_family",
    "required_slot",
    "emitted_slot",
    "checkpoint_sha256",
    "artifact_package_zip_sha256",
    "latency_ms",
    "max_response_tokens",
    "truncated",
    "diagnosis",
    "empty prompt",
    "whitespace prompt",
    "oversized prompt",
    "invalid batch row",
    "unsupported topic",
    "same output_text",
    "same status",
    "same supported_family",
    "timeout guard exercised",
    "audit_log.jsonl",
    "timestamp",
    "output_sha256",
    "queue.json",
    "progress.jsonl",
    "runtime_config.json",
    "artifact_manifest.json",
    "checkpoint_manifest.json",
    "single_inference.json",
    "batch_inference.jsonl",
    "bad_input_results.jsonl",
    "unsupported_input_results.jsonl",
    "determinism_report.json",
    "timeout_report.json",
    "runtime_metrics.json",
    "summary.json",
    "report.md",
    "artifact_hash_verified = true",
    "single_prompt_pass = true",
    "batch_prompt_pass = true",
    "json_output_envelope_pass = true",
    "human_readable_output_pass = true",
    "deterministic_repeated_output_pass = true",
    "bad_input_handled = true",
    "unsupported_input_handled = true",
    "timeout_guard_pass = true",
    "audit_log_written = true",
    "BOUNDED_CHAT_INFERENCE_RUNTIME_POSITIVE",
    "ARTIFACT_PACKAGE_VERIFIED",
    "CHECKPOINT_LOADED_READ_ONLY",
    "SINGLE_PROMPT_INFERENCE_PASSES",
    "BATCH_INFERENCE_PASSES",
    "JSON_OUTPUT_ENVELOPE_PASSES",
    "HUMAN_READABLE_OUTPUT_WRITTEN",
    "DETERMINISTIC_OUTPUT_CONFIRMED",
    "BAD_INPUT_HANDLED",
    "UNSUPPORTED_INPUT_HANDLED",
    "TIMEOUT_GUARD_PASSES",
    "AUDIT_LOG_WRITTEN",
    "NO_TRAINING_PERFORMED",
    "RUNTIME_LOCAL_ONLY",
    "PRODUCTION_CHAT_NOT_CLAIMED",
    "BOUNDED_CHAT_INFERENCE_RUNTIME_FAILS",
    "UPSTREAM_083_ARTIFACT_MISSING",
    "ARTIFACT_HASH_MISMATCH",
    "CHECKPOINT_LOAD_FAILS",
    "CHECKPOINT_MUTATION_DETECTED",
    "SINGLE_PROMPT_INFERENCE_FAILS",
    "BATCH_INFERENCE_FAILS",
    "JSON_OUTPUT_ENVELOPE_MISSING",
    "HUMAN_READABLE_OUTPUT_MISSING",
    "DETERMINISM_FAILS",
    "BAD_INPUT_NOT_HANDLED",
    "UNSUPPORTED_INPUT_NOT_HANDLED",
    "TIMEOUT_GUARD_FAILS",
    "AUDIT_LOG_MISSING",
    "TRAINING_SIDE_EFFECT_DETECTED",
    "ORACLE_SHORTCUT_DETECTED",
    "LLM_JUDGE_USED",
    "SERVICE_API_SURFACE_DETECTED",
    "DEPLOYMENT_HARNESS_MUTATION_DETECTED",
    "SDK_PUBLIC_EXPORT_MUTATION_DETECTED",
    "GPT_LIKE_READINESS_FALSE_CLAIM",
    "PRODUCTION_CHAT_CLAIM_DETECTED",
    "RUNTIME_SURFACE_MUTATION_DETECTED",
    "ROOT_LICENSE_CHANGED",
    "085_BOUNDED_CHAT_SERVICE_API_ALPHA",
    "084B_BOUNDED_CHAT_INFERENCE_RUNTIME_FAILURE_ANALYSIS",
]

PLACEHOLDERS = ["TODO", "TBD", "placeholder", "pending measured", "coming soon"]

FORBIDDEN_CLAIMS = {
    "GPT_LIKE_READINESS_FALSE_CLAIM": ["GPT-like assistant", "GPT-like assistant readiness"],
    "OPEN_DOMAIN_CHAT_CLAIM_DETECTED": ["open-domain chat"],
    "PRODUCTION_CHAT_CLAIM_DETECTED": ["production chat"],
    "SAFETY_ALIGNMENT_CLAIM_DETECTED": ["safety alignment"],
    "PUBLIC_BETA_CLAIM_DETECTED": ["public beta"],
    "GA_CLAIM_DETECTED": ["GA release", "generally available"],
    "HOSTED_SAAS_CLAIM_DETECTED": ["hosted SaaS"],
    "SERVICE_API_SURFACE_DETECTED": ["service API"],
    "DEPLOYMENT_HARNESS_MUTATION_DETECTED": ["deployment harness"],
    "SDK_PUBLIC_EXPORT_MUTATION_DETECTED": ["SDK surface", "SDK/public export"],
    "ROOT_LICENSE_CHANGED": ["root LICENSE changed"],
}

NEGATION_MARKERS = ["not ", "no ", "does not ", "do not ", "without ", "false", "reject", "rejected", "only"]
GENERATED_PATH_PARTS = ["target/", "node_modules/", ".svelte-kit/"]
GENERATED_SUFFIXES = [".exe", ".dll", ".so", ".dylib", ".zip", ".tar", ".tar.gz", ".tgz", ".7z", ".ckpt", ".bin"]

POSITIVE_VERDICTS = [
    "BOUNDED_CHAT_INFERENCE_RUNTIME_STATIC_CHECK_POSITIVE",
    "BOUNDED_CHAT_INFERENCE_RUNTIME_FILES_WRITTEN",
    "LOCAL_ONLY_RUNTIME_GUARD_WRITTEN",
    "ARTIFACT_HASH_VERIFICATION_REQUIRED",
    "STRICT_JSON_ENVELOPE_REQUIRED",
    "BAD_AND_UNSUPPORTED_INPUT_REQUIRED",
    "DETERMINISM_AND_TIMEOUT_REQUIRED",
    "AUDIT_LOG_REQUIRED",
    "RUNTIME_SURFACE_UNCHANGED",
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
    for marker in PLACEHOLDERS:
        if marker.lower() in audited_text.lower():
            failures.append(f"PLACEHOLDER_PRESENT:{marker}")
    failures.extend(find_false_claims(audited_text))

    if root_license_changed():
        failures.append("ROOT_LICENSE_CHANGED")
    if runtime_surface_mutation_detected():
        failures.append("RUNTIME_SURFACE_MUTATION_DETECTED")
    if generated_artifact_staged():
        failures.append("GENERATED_ARTIFACT_STAGED")

    payload = {
        "schema_version": "bounded_chat_inference_runtime_static_check_v1",
        "status": "passed" if not failures else "failed",
        "check_only": args.check_only,
        "verdicts": POSITIVE_VERDICTS if not failures else ["BOUNDED_CHAT_INFERENCE_RUNTIME_STATIC_CHECK_FAILS", *failures],
        "checked_files": REQUIRED_DOCS + REQUIRED_SOURCE,
        "allowed_mutations": sorted(ALLOWED_MUTATIONS),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
