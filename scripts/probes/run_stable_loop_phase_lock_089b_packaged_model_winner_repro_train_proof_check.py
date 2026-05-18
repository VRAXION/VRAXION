#!/usr/bin/env python3
"""Static/artifact checker for STABLE_LOOP_PHASE_LOCK_089B."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_089b_packaged_model_winner_repro_train_proof/smoke"

REQUIRED_DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_089B_PACKAGED_MODEL_WINNER_REPRO_TRAIN_PROOF_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_089B_PACKAGED_MODEL_WINNER_REPRO_TRAIN_PROOF_RESULT.md",
]
REQUIRED_SOURCE = [
    "scripts/probes/run_stable_loop_phase_lock_089b_packaged_model_winner_repro_train_proof.py",
    "scripts/probes/run_stable_loop_phase_lock_089b_packaged_model_winner_repro_train_proof_check.py",
]
ALLOWED_MUTATIONS = set(REQUIRED_DOCS + REQUIRED_SOURCE)

REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "winner_proof_config.json",
    "upstream_manifest.json",
    "package_hash_binding.json",
    "packaged_checkpoint_eval.json",
    "repro_training_manifest.json",
    "repro_training_metrics.jsonl",
    "deterministic_mismatch_analysis.json",
    "arm_comparison.json",
    "control_delta_report.json",
    "tamper_control_report.json",
    "leakage_control_report.json",
    "eval_row_hashes.json",
    "human_readable_samples.jsonl",
    "failure_case_samples.jsonl",
    "summary.json",
    "report.md",
]

EXACT_COMMANDS = [
    "python -m py_compile scripts/probes/run_stable_loop_phase_lock_089b_packaged_model_winner_repro_train_proof.py",
    "python scripts/probes/run_stable_loop_phase_lock_089b_packaged_model_winner_repro_train_proof.py --out target/pilot_wave/stable_loop_phase_lock_089b_packaged_model_winner_repro_train_proof/smoke --upstream-078-root target/pilot_wave/stable_loop_phase_lock_078_chat_composition_repair/smoke --upstream-080-root target/pilot_wave/stable_loop_phase_lock_080_chat_composition_diversity_repair/smoke --upstream-081-root target/pilot_wave/stable_loop_phase_lock_081_chat_diversity_fresh_confirm/smoke --upstream-082-root target/pilot_wave/stable_loop_phase_lock_082_chat_diversity_multi_seed_confirm/smoke --upstream-083-root target/pilot_wave/stable_loop_phase_lock_083_chat_model_artifact_rc_package/smoke --upstream-089-root target/pilot_wave/stable_loop_phase_lock_089_private_evaluation_rc_package/smoke --seed 2026 --chat-examples 80000 --heartbeat-sec 20",
    "python -m py_compile scripts/probes/run_stable_loop_phase_lock_089b_packaged_model_winner_repro_train_proof_check.py",
    "python scripts/probes/run_stable_loop_phase_lock_089b_packaged_model_winner_repro_train_proof_check.py --check-only",
    "python scripts/probes/run_stable_loop_phase_lock_089_private_evaluation_rc_package_check.py --check-only",
    "python scripts/probes/run_stable_loop_phase_lock_088_bounded_chat_long_run_concurrency_resource_stability_check.py --check-only",
    "git diff --check",
]

REQUIRED_TERMS = [
    "STABLE_LOOP_PHASE_LOCK_089B_PACKAGED_MODEL_WINNER_REPRO_AND_ADVERSARIAL_TRAIN_PROOF",
    "PACKAGED_MODEL_WINNER_REPRO_TRAIN_PROOF_POSITIVE",
    "PACKAGE_HASH_BINDING_VERIFIED",
    "PACKAGED_CHECKPOINT_FRESH_EVAL_PASSES",
    "DETERMINISTIC_REPRO_TRAINING_PASSES",
    "TOKEN_OBJECTIVE_LEARNED",
    "WINNER_BEATS_CONTROLS",
    "BASELINE_EVAL_ROWS_MATCH",
    "TAMPER_CONTROLS_FAIL_AS_EXPECTED",
    "LEAKAGE_CONTROLS_FAIL_AS_EXPECTED",
    "FINITE_LABEL_ANCHORROUTE_RETENTION_PASSES",
    "NO_TRAINING_ON_PACKAGED_CHECKPOINT",
    "CHECKPOINT_PIPELINE_PASSES",
    "PRODUCTION_CHAT_NOT_CLAIMED",
    "PACKAGED_MODEL_WINNER_REPRO_TRAIN_PROOF_FAILS",
    "UPSTREAM_ARTIFACT_MISSING",
    "UPSTREAM_STACK_NOT_POSITIVE",
    "PACKAGE_CHECKPOINT_HASH_MISMATCH",
    "PACKAGED_CHECKPOINT_FRESH_EVAL_FAILS",
    "REPRO_TRAINING_FAILS",
    "TOKEN_OBJECTIVE_NOT_LEARNED",
    "WINNER_NOT_REPRODUCED",
    "STALE_REPRO_ARTIFACT_USED",
    "BASELINE_EVAL_MISMATCH",
    "CONTROL_DELTA_INSUFFICIENT",
    "RANDOM_OR_COPY_CONTROL_UNEXPECTED_PASS",
    "CORRUPTED_CHECKPOINT_UNEXPECTEDLY_ACCEPTED",
    "WRONG_HASH_UNEXPECTEDLY_ACCEPTED",
    "LEAKAGE_CONTROL_UNEXPECTEDLY_ACCEPTED",
    "RESPONSE_TABLE_SHORTCUT_UNEXPECTEDLY_ACCEPTED",
    "FINITE_LABEL_RETENTION_REGRESSION_DETECTED",
    "CHECKPOINT_MUTATION_DETECTED",
    "TRAINING_SIDE_EFFECT_ON_PACKAGED_CHECKPOINT",
    "RUNTIME_SURFACE_MUTATION_DETECTED",
    "ROOT_LICENSE_CHANGED",
    "GPT_LIKE_READINESS_FALSE_CLAIM",
    "PRODUCTION_CHAT_CLAIM_DETECTED",
    "deterministic_mismatch_analysis.json",
    "full_checkpoint_file_sha256",
    "normalized_model_payload_sha256",
    "metadata_only_mismatch",
    "payload_hash_matches",
    "PACKAGED_089_RC_CHECKPOINT",
    "REPRODUCED_080_DIVERSITY_REPAIR",
    "NO_REPAIR_078_BASELINE",
    "RESPONSE_TABLE_ONLY_CONTROL",
    "ONE_TARGET_PER_PROMPT_CONTROL",
    "NO_SKELETON_DROPOUT_CONTROL",
    "NO_LEXICAL_DROPOUT_CONTROL",
    "NO_CLAUSE_RANDOMIZATION_CONTROL",
    "RANDOM_LABEL_CONTROL",
    "COPY_PROMPT_CONTROL",
    "eval_row_hash",
    "eval_row_count",
    "eval_dataset_sha256",
    "bounded winner reproducibility proof only",
    "not GPT-like assistant readiness",
    "not open-domain chat",
    "not full English LM",
    "not production deployment",
    "not safety alignment",
    "not public release",
    "091_OPEN_VOCAB_CHAT_LM_FOUNDATION",
    "090_BOUNDED_LOCAL_PRIVATE_DEPLOY_READY_GATE",
    "089C_WINNER_PROOF_FAILURE_ANALYSIS",
]

PLACEHOLDERS = ["TODO", "TBD", "placeholder", "pending measured", "coming soon"]
NEGATION_MARKERS = ["not ", "no ", "does not ", "do not ", "without ", "false", "reject", "rejected", "only"]
FORBIDDEN_CLAIMS = {
    "GPT_LIKE_READINESS_FALSE_CLAIM": ["GPT-like assistant readiness"],
    "OPEN_DOMAIN_CHAT_CLAIM_DETECTED": ["open-domain chat"],
    "FULL_ENGLISH_LM_CLAIM_DETECTED": ["full English LM"],
    "PRODUCTION_CHAT_CLAIM_DETECTED": ["production chat"],
    "PRODUCTION_DEPLOYMENT_CLAIM_DETECTED": ["production deployment"],
    "SAFETY_ALIGNMENT_CLAIM_DETECTED": ["safety alignment"],
    "PUBLIC_RELEASE_CLAIM_DETECTED": ["public release"],
}
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
        window = lowered[max(0, match.start() - 100) : match.start()]
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
    if "PACKAGED_MODEL_WINNER_REPRO_TRAIN_PROOF_POSITIVE" not in summary.get("verdicts", []):
        failures.append("POSITIVE_VERDICT_MISSING")

    expected_true = [
        "package_hash_binding_pass",
        "packaged_checkpoint_eval_pass",
        "repro_training_pass",
        "winner_beats_controls",
        "tamper_controls_pass",
        "leakage_controls_pass",
        "packaged_checkpoint_hash_unchanged",
        "finite_label_retention_pass",
    ]
    for key in expected_true:
        if metrics.get(key) is not True:
            failures.append(f"METRIC_NOT_TRUE:{key}")

    if summary.get("packaged_train_step_count") != 0:
        failures.append("TRAINING_SIDE_EFFECT_ON_PACKAGED_CHECKPOINT")
    if summary.get("training_side_effect_on_packaged_checkpoint") is not False:
        failures.append("TRAINING_SIDE_EFFECT_ON_PACKAGED_CHECKPOINT")
    for key in [
        "gpt_like_assistant_readiness_claimed",
        "open_domain_chat_claimed",
        "full_English_LM_claimed",
        "production_deployment_claimed",
        "production_chat_claimed",
        "safety_alignment_claimed",
        "public_release_claimed",
    ]:
        if summary.get(key) is not False:
            failures.append(f"OVERCLAIM_FLAG_TRUE:{key}")

    binding = load_json(SMOKE_ROOT / "package_hash_binding.json")
    if not binding.get("package_hash_binding_pass"):
        failures.append("PACKAGE_CHECKPOINT_HASH_MISMATCH")
    if not binding.get("source_equals_packaged_checkpoint_file"):
        failures.append("PACKAGE_CHECKPOINT_HASH_MISMATCH")
    if not binding.get("packaged_checkpoint_file_matches_080_file"):
        failures.append("PACKAGE_CHECKPOINT_HASH_MISMATCH")

    packaged_eval = load_json(SMOKE_ROOT / "packaged_checkpoint_eval.json")
    if packaged_eval.get("packaged_train_step_count") != 0 or not packaged_eval.get("packaged_checkpoint_hash_unchanged"):
        failures.append("TRAINING_SIDE_EFFECT_ON_PACKAGED_CHECKPOINT")
    if not packaged_eval.get("packaged_checkpoint_eval_pass"):
        failures.append("PACKAGED_CHECKPOINT_FRESH_EVAL_FAILS")

    repro = load_json(SMOKE_ROOT / "repro_training_manifest.json")
    if repro.get("child_train_step_count", 0) <= 0 or repro.get("child_token_train_step_count", 0) <= 0:
        failures.append("REPRO_TRAINING_FAILS")
    if not repro.get("repro_child_started_after_089b_start"):
        failures.append("STALE_REPRO_ARTIFACT_USED")
    if not repro.get("repro_child_summary_newer_than_089b_start") or not repro.get("repro_child_report_newer_than_089b_start"):
        failures.append("STALE_REPRO_ARTIFACT_USED")
    if not (repro.get("token_loss_final", 1e9) < repro.get("token_loss_initial", -1e9)):
        failures.append("TOKEN_OBJECTIVE_NOT_LEARNED")
    if repro.get("child_checkpoint_save_load_pass") is not True or repro.get("child_resume_from_checkpoint_pass") is not True:
        failures.append("CHECKPOINT_PIPELINE_FAILS")

    mismatch = load_json(SMOKE_ROOT / "deterministic_mismatch_analysis.json")
    if mismatch.get("payload_hash_matches") is not True or mismatch.get("file_hash_matches") is not True:
        failures.append("WINNER_NOT_REPRODUCED")

    hashes = load_json(SMOKE_ROOT / "eval_row_hashes.json")
    if not hashes.get("all_eval_row_hash_identical") or not hashes.get("all_eval_row_count_identical"):
        failures.append("BASELINE_EVAL_MISMATCH")

    arms = load_json(SMOKE_ROOT / "arm_comparison.json")
    if not arms.get("winner_beats_controls"):
        failures.append("CONTROL_DELTA_INSUFFICIENT")
    required_arms = {
        "PACKAGED_089_RC_CHECKPOINT",
        "REPRODUCED_080_DIVERSITY_REPAIR",
        "NO_REPAIR_078_BASELINE",
        "RESPONSE_TABLE_ONLY_CONTROL",
        "ONE_TARGET_PER_PROMPT_CONTROL",
        "NO_SKELETON_DROPOUT_CONTROL",
        "NO_LEXICAL_DROPOUT_CONTROL",
        "NO_CLAUSE_RANDOMIZATION_CONTROL",
        "RANDOM_LABEL_CONTROL",
        "COPY_PROMPT_CONTROL",
    }
    present = {row.get("arm") for row in arms.get("arms", [])}
    missing_arms = sorted(required_arms - present)
    if missing_arms:
        failures.append(f"ARM_MISSING:{','.join(missing_arms)}")
    for row in arms.get("arms", []):
        for field in ["eval_row_hash", "eval_row_count", "eval_dataset_sha256"]:
            if field not in row:
                failures.append(f"ARM_FIELD_MISSING:{row.get('arm')}:{field}")
    for control in arms.get("control_arms", []):
        for field in ["control_name", "expected_failure_mode", "actual_metrics", "passed_unexpectedly"]:
            if field not in control:
                failures.append(f"CONTROL_FIELD_MISSING:{field}")
        if control.get("control_name") in {"RANDOM_LABEL_CONTROL", "COPY_PROMPT_CONTROL"} and control.get("passed_unexpectedly"):
            failures.append("RANDOM_OR_COPY_CONTROL_UNEXPECTED_PASS")

    tamper = load_json(SMOKE_ROOT / "tamper_control_report.json")
    if not tamper.get("all_tamper_controls_failed_as_expected"):
        failures.append("TAMPER_CONTROL_NOT_FAILED")
    for control in tamper.get("controls", []):
        for field in ["artifact_path", "mutation_type", "expected_failure", "observed_failure", "detector_used"]:
            if field not in control:
                failures.append(f"TAMPER_FIELD_MISSING:{field}")
        if control.get("observed_failure") is not True:
            failures.append(control.get("failure_verdict", "TAMPER_CONTROL_NOT_FAILED"))

    leakage = load_json(SMOKE_ROOT / "leakage_control_report.json")
    if leakage.get("leakage_controls_fail_as_expected") is not True:
        failures.append("LEAKAGE_CONTROL_UNEXPECTEDLY_ACCEPTED")

    samples = read_jsonl(SMOKE_ROOT / "human_readable_samples.jsonl")
    sample_arms = {row.get("arm") for row in samples}
    for arm in ["PACKAGED_089_RC_CHECKPOINT", "REPRODUCED_080_DIVERSITY_REPAIR", "NO_REPAIR_078_BASELINE", "RESPONSE_TABLE_ONLY_CONTROL", "COPY_PROMPT_CONTROL", "RANDOM_LABEL_CONTROL"]:
        if arm not in sample_arms:
            failures.append(f"HUMAN_SAMPLE_ARM_MISSING:{arm}")
    for field in ["arm", "prompt", "output", "expected_behavior", "pass_fail", "novelty_flag", "template_copy_flag", "skeleton_reuse_flag", "slot_binding_diagnosis"]:
        if not all(field in row for row in samples):
            failures.append(f"HUMAN_SAMPLE_FIELD_MISSING:{field}")
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
    for term in REQUIRED_TERMS + EXACT_COMMANDS:
        if term not in joined:
            failures.append(f"MISSING_TERM:{term}")
    source = files.get("scripts/probes/run_stable_loop_phase_lock_089b_packaged_model_winner_repro_train_proof.py", "")
    for required in ["subprocess.Popen", "phase_lane_chat_composition_diversity_repair", "phase_lane_chat_diversity_fresh_confirm"]:
        if required not in source:
            failures.append(f"REQUIRED_CHILD_ORCHESTRATION_MISSING:{required}")
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
        "schema_version": "packaged_model_winner_repro_train_proof_check_v1",
        "status": "passed" if not failures else "failed",
        "check_only": args.check_only,
        "verdicts": [
            "PACKAGED_MODEL_WINNER_REPRO_TRAIN_PROOF_STATIC_CHECK_POSITIVE",
            "WINNER_PROOF_FILES_WRITTEN",
            "PACKAGE_HASH_BINDING_CHECKED",
            "PACKAGED_EVAL_CHECKED",
            "REPRO_TRAINING_CHECKED",
            "CONTROL_DELTA_CHECKED",
            "TAMPER_AND_LEAKAGE_CONTROLS_CHECKED",
            "ROOT_LICENSE_UNCHANGED",
            "PRODUCTION_CHAT_NOT_CLAIMED",
        ]
        if not failures
        else ["PACKAGED_MODEL_WINNER_REPRO_TRAIN_PROOF_STATIC_CHECK_FAILS", *failures],
        "checked_files": REQUIRED_DOCS + REQUIRED_SOURCE,
        "allowed_mutations": sorted(ALLOWED_MUTATIONS),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
