#!/usr/bin/env python3
"""Static checker for STABLE_LOOP_PHASE_LOCK_083 chat model artifact RC package."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

REQUIRED_DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_083_CHAT_MODEL_ARTIFACT_RC_PACKAGE_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_083_CHAT_MODEL_ARTIFACT_RC_PACKAGE_RESULT.md",
]

REQUIRED_SOURCE = [
    "scripts/probes/run_stable_loop_phase_lock_083_chat_model_artifact_rc_package.py",
    "scripts/probes/run_stable_loop_phase_lock_083_chat_model_artifact_rc_package_check.py",
]

ALLOWED_MUTATIONS = set(REQUIRED_DOCS + REQUIRED_SOURCE)

EXACT_COMMANDS = [
    "python -m py_compile scripts/probes/run_stable_loop_phase_lock_083_chat_model_artifact_rc_package.py",
    "python -m py_compile scripts/probes/run_stable_loop_phase_lock_083_chat_model_artifact_rc_package_check.py",
    "python scripts/probes/run_stable_loop_phase_lock_083_chat_model_artifact_rc_package.py --out target/pilot_wave/stable_loop_phase_lock_083_chat_model_artifact_rc_package/smoke --upstream-082-root target/pilot_wave/stable_loop_phase_lock_082_chat_diversity_multi_seed_confirm/smoke --upstream-081-root target/pilot_wave/stable_loop_phase_lock_081_chat_diversity_fresh_confirm/smoke --upstream-080-root target/pilot_wave/stable_loop_phase_lock_080_chat_composition_diversity_repair/smoke --upstream-074-root target/pilot_wave/stable_loop_phase_lock_074_multi_seed_scenario_gated_repair_confirm/smoke --heartbeat-sec 20",
    "python scripts/probes/run_stable_loop_phase_lock_083_chat_model_artifact_rc_package_check.py --check-only",
    "python scripts/probes/run_stable_loop_phase_lock_082_chat_diversity_multi_seed_confirm_check.py --check-only",
    "python scripts/probes/run_stable_loop_phase_lock_080_chat_composition_diversity_repair_check.py --check-only",
    "git diff --check",
]

BOUNDARY_TOKENS = [
    "private bounded model artifact RC only",
    "not deploy-ready by itself",
    "not inference runtime",
    "not service/API integration",
    "not GPT-like assistant",
    "not production chat",
    "not full English LM",
    "not language grounding",
    "not safety alignment",
    "not public beta / GA / hosted SaaS",
    "no instnct-core runtime change",
    "no service API change",
    "no deployment harness change",
    "no SDK/public export change",
    "no release docs change",
    "no root LICENSE change",
    "no upstream checkpoint mutation",
]

REQUIRED_TERMS = [
    "STABLE_LOOP_PHASE_LOCK_083_CHAT_MODEL_ARTIFACT_RC_PACKAGE",
    "run_stable_loop_phase_lock_083_chat_model_artifact_rc_package.py",
    "run_stable_loop_phase_lock_083_chat_model_artifact_rc_package_check.py",
    "copy existing 080 checkpoint into package dir",
    "hash files",
    "collect eval provenance",
    "write package manifests",
    "write artifact zip under target/",
    "target/pilot_wave/stable_loop_phase_lock_083_chat_model_artifact_rc_package/",
    "target/pilot_wave/stable_loop_phase_lock_080_chat_composition_diversity_repair/smoke/checkpoints/chat_composition_diversity_repair/model_checkpoint.json",
    "target/pilot_wave/stable_loop_phase_lock_082_chat_diversity_multi_seed_confirm/smoke",
    "target/pilot_wave/stable_loop_phase_lock_081_chat_diversity_fresh_confirm/smoke",
    "target/pilot_wave/stable_loop_phase_lock_080_chat_composition_diversity_repair/smoke",
    "target/pilot_wave/stable_loop_phase_lock_074_multi_seed_scenario_gated_repair_confirm/smoke",
    "CHAT_DIVERSITY_MULTI_SEED_CONFIRM_POSITIVE",
    "all_seed_pass = true",
    "all child_exit_code = 0",
    "all child_recheck_pass = true",
    "all checkpoint_hash_unchanged = true",
    "all train_step_count = 0",
    "source_checkpoint_sha256",
    "packaged_checkpoint_sha256",
    "source_checkpoint_size_bytes",
    "packaged_checkpoint_size_bytes",
    "packaged_checkpoint_sha256 == source_checkpoint_sha256",
    "packaged_checkpoint_size_bytes == source_checkpoint_size_bytes",
    "artifact_package_zip_sha256",
    "queue.json",
    "progress.jsonl",
    "package_config.json",
    "source_checkpoint_manifest.json",
    "packaged_checkpoint_manifest.json",
    "integrity_hashes.json",
    "upstream_082_manifest.json",
    "eval_provenance_manifest.json",
    "capability_surface.json",
    "known_limitations.json",
    "claim_boundary.json",
    "sample_prompts_outputs.jsonl",
    "repro_commands.ps1",
    "rollback_pointer.json",
    "artifact_index.json",
    "artifact_package.zip",
    "summary.json",
    "report.md",
    "bounded_domain_chat_composition = true",
    "finite_label_anchorroute_retention = true",
    "context_slot_binding = true",
    "multi_seed_chat_diversity_confirmed = true",
    "open_domain_chat_supported = false",
    "gpt_like_assistant_readiness_claimed = false",
    "full_English_LM_supported = false",
    "language_grounding_claimed = false",
    "production_chat_claimed = false",
    "safety_alignment_claimed = false",
    "public_beta_claimed = false",
    "GA_claimed = false",
    "hosted_SaaS_claimed = false",
    "deploy_ready_by_itself = false",
    "bounded English domain only",
    "no open-domain chat",
    "no Hungarian chat proof",
    "no long multi-turn proof",
    "no production safety alignment",
    "no service/API runtime",
    "no deployment harness integration",
    "no public beta / GA",
    "no hosted SaaS",
    "no clinical/high-stakes use",
    "fresh instruction",
    "short explanation",
    "context slot",
    "two-turn carry",
    "boundary mini",
    "anti-template-copy",
    "finite-label retention",
    "source_seed",
    "eval_family",
    "prompt",
    "model_output",
    "expected_behavior",
    "pass_fail",
    "novelty_flag",
    "template_copy_flag",
    "skeleton_reuse_flag",
    "slot_binding_diagnosis",
    "082 checker",
    "081 checker",
    "080 checker",
    "package hash verification",
    "no training command as required step",
    "previous checkpoint path/hash if available",
    "source 080 checkpoint path/hash",
    "artifact package path",
    "rollback instruction",
    "no automatic production rollback claim",
    "packaged checkpoint",
    "CHAT_MODEL_ARTIFACT_RC_PACKAGE_POSITIVE",
    "UPSTREAM_082_MULTI_SEED_PROOF_VERIFIED",
    "SOURCE_CHECKPOINT_VERIFIED",
    "PACKAGED_CHECKPOINT_HASH_MATCHES_SOURCE",
    "EVAL_PROVENANCE_MANIFEST_WRITTEN",
    "CAPABILITY_SURFACE_RECORDED",
    "KNOWN_LIMITATIONS_RECORDED",
    "SAMPLE_PROMPTS_OUTPUTS_WRITTEN",
    "ROLLBACK_POINTER_WRITTEN",
    "REPRO_COMMANDS_WRITTEN",
    "NO_TRAINING_PERFORMED",
    "RUNTIME_SURFACE_UNCHANGED",
    "PRODUCTION_CHAT_NOT_CLAIMED",
    "CHAT_MODEL_ARTIFACT_RC_PACKAGE_FAILS",
    "UPSTREAM_082_ARTIFACT_MISSING",
    "UPSTREAM_080_ARTIFACT_MISSING",
    "UPSTREAM_082_NOT_POSITIVE",
    "SOURCE_CHECKPOINT_MISSING",
    "CHECKPOINT_COPY_HASH_MISMATCH",
    "CHECKPOINT_MUTATION_DETECTED",
    "EVAL_PROVENANCE_INCOMPLETE",
    "CAPABILITY_SURFACE_MISSING",
    "KNOWN_LIMITATIONS_MISSING",
    "SAMPLE_PROMPTS_OUTPUTS_MISSING",
    "ROLLBACK_POINTER_MISSING",
    "REPRO_COMMANDS_MISSING",
    "TRAINING_SIDE_EFFECT_DETECTED",
    "ORACLE_SHORTCUT_DETECTED",
    "LLM_JUDGE_USED",
    "GPT_LIKE_READINESS_FALSE_CLAIM",
    "PRODUCTION_CHAT_CLAIM_DETECTED",
    "RUNTIME_SURFACE_MUTATION_DETECTED",
    "ROOT_LICENSE_CHANGED",
    "084_BOUNDED_CHAT_INFERENCE_RUNTIME",
    "083B_CHAT_MODEL_ARTIFACT_PACKAGE_FAILURE_ANALYSIS",
]

PLACEHOLDERS = ["TODO", "TBD", "placeholder", "pending measured", "coming soon"]

FORBIDDEN_CLAIMS = {
    "GPT_LIKE_READINESS_FALSE_CLAIM": ["GPT-like assistant", "GPT-like assistant readiness"],
    "FULL_ENGLISH_LM_CLAIM_DETECTED": ["full English LM"],
    "LANGUAGE_GROUNDING_CLAIM_DETECTED": ["language grounding"],
    "PRODUCTION_CHAT_CLAIM_DETECTED": ["production chat"],
    "SAFETY_ALIGNMENT_CLAIM_DETECTED": ["safety alignment"],
    "PUBLIC_BETA_CLAIM_DETECTED": ["public beta"],
    "GA_CLAIM_DETECTED": ["GA release", "generally available"],
    "HOSTED_SAAS_CLAIM_DETECTED": ["hosted SaaS"],
    "SERVICE_API_CLAIM_DETECTED": ["service API integration", "service API change"],
    "DEPLOYMENT_HARNESS_CLAIM_DETECTED": ["deployment harness integration", "deployment harness change"],
    "SDK_EXPORT_CLAIM_DETECTED": ["SDK/public export change"],
    "ROOT_LICENSE_CHANGED": ["root LICENSE changed"],
}

NEGATION_MARKERS = ["not ", "no ", "does not ", "do not ", "without ", "false", "reject", "rejected", "only"]
GENERATED_PATH_PARTS = ["target/", "node_modules/", ".svelte-kit/"]
GENERATED_SUFFIXES = [".exe", ".dll", ".so", ".dylib", ".zip", ".tar", ".tar.gz", ".tgz", ".7z", ".ckpt", ".bin"]

POSITIVE_VERDICTS = [
    "CHAT_MODEL_ARTIFACT_RC_PACKAGE_STATIC_CHECK_POSITIVE",
    "CHAT_MODEL_ARTIFACT_RC_PACKAGE_FILES_WRITTEN",
    "PACKAGING_ONLY_GUARD_WRITTEN",
    "CHECKPOINT_COPY_INTEGRITY_REQUIRED",
    "UPSTREAM_082_PROOF_REQUIRED",
    "CAPABILITY_SURFACE_REQUIRED",
    "KNOWN_LIMITATIONS_REQUIRED",
    "SAMPLES_REPRO_ROLLBACK_REQUIRED",
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
        "schema_version": "chat_model_artifact_rc_package_static_check_v1",
        "status": "passed" if not failures else "failed",
        "check_only": args.check_only,
        "verdicts": POSITIVE_VERDICTS if not failures else ["CHAT_MODEL_ARTIFACT_RC_PACKAGE_STATIC_CHECK_FAILS", *failures],
        "checked_files": REQUIRED_DOCS + REQUIRED_SOURCE,
        "allowed_mutations": sorted(ALLOWED_MUTATIONS),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
