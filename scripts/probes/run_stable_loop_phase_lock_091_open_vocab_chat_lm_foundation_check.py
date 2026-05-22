#!/usr/bin/env python3
"""Static/artifact checker for STABLE_LOOP_PHASE_LOCK_091."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_091_open_vocab_chat_lm_foundation/smoke"

REQUIRED_DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_091_OPEN_VOCAB_CHAT_LM_FOUNDATION_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_091_OPEN_VOCAB_CHAT_LM_FOUNDATION_RESULT.md",
]
REQUIRED_SOURCE = [
    "scripts/probes/run_stable_loop_phase_lock_091_open_vocab_chat_lm_foundation.py",
    "scripts/probes/run_stable_loop_phase_lock_091_open_vocab_chat_lm_foundation_check.py",
]
ALLOWED_MUTATIONS = set(REQUIRED_DOCS + REQUIRED_SOURCE)

REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "training_config.json",
    "upstream_089b_manifest.json",
    "dataset_manifest.json",
    "tokenizer_manifest.json",
    "train_examples_sample.jsonl",
    "eval_examples_sample.jsonl",
    "training_metrics.jsonl",
    "checkpoint_manifest.json",
    "checkpoint_hashes.json",
    "generation_samples.jsonl",
    "human_readable_samples.jsonl",
    "lm_metrics.json",
    "open_vocab_generation_metrics.json",
    "bounded_retention_metrics.json",
    "collapse_metrics.json",
    "leakage_metrics.json",
    "arm_comparison.json",
    "control_delta_report.json",
    "failure_case_samples.jsonl",
    "summary.json",
    "report.md",
]

EXACT_COMMANDS = [
    "python -m py_compile scripts/probes/run_stable_loop_phase_lock_091_open_vocab_chat_lm_foundation.py",
    "python scripts/probes/run_stable_loop_phase_lock_091_open_vocab_chat_lm_foundation.py --out target/pilot_wave/stable_loop_phase_lock_091_open_vocab_chat_lm_foundation/smoke --upstream-089b-root target/pilot_wave/stable_loop_phase_lock_089b_packaged_model_winner_repro_train_proof/smoke --seed 2026 --train-tokens 250000 --eval-tokens 50000 --seq-len 128 --batch-size 32 --steps 1200 --heartbeat-sec 20",
    "python -m py_compile scripts/probes/run_stable_loop_phase_lock_091_open_vocab_chat_lm_foundation_check.py",
    "python scripts/probes/run_stable_loop_phase_lock_091_open_vocab_chat_lm_foundation_check.py --check-only",
    "python scripts/probes/run_stable_loop_phase_lock_089b_packaged_model_winner_repro_train_proof_check.py --check-only",
    "python scripts/probes/run_stable_loop_phase_lock_089_private_evaluation_rc_package_check.py --check-only",
    "git diff --check",
]

REQUIRED_TERMS = [
    "STABLE_LOOP_PHASE_LOCK_091_OPEN_VOCAB_CHAT_LM_FOUNDATION",
    "runner-local open-vocab next-byte LM foundation sanity passed",
    "packaged bounded winner itself is now GPT-like",
    "INSTNCT is proven as open-domain LM",
    "production/open-domain assistant readiness",
    "runner_local_pytorch_lm",
    "packaged_winner_checkpoint_trained",
    "packaged_winner_used_for_retention_reference",
    "architecture_winner_for_open_vocab_claimed",
    "byte_level",
    "causal_next_byte",
    "OPEN_VOCAB_BYTE_LM_MAIN",
    "CHAR_BIGRAM_BASELINE",
    "RANDOM_BYTE_CONTROL",
    "SHUFFLED_TARGET_CONTROL",
    "BOUNDED_ONLY_TRAIN_CONTROL",
    "NO_ANCHOR_RETENTION_MIX_CONTROL",
    "STATIC_OUTPUT_CONTROL",
    "COPY_PROMPT_CONTROL",
    "corpus_sha256",
    "train_split_sha256",
    "eval_split_sha256",
    "train_eval_exact_text_overlap_count",
    "max_train_eval_jaccard",
    "eval_row_hash",
    "eval_token_hash",
    "eval_token_count",
    "delta_vs_char_bigram_loss",
    "delta_vs_shuffled_target_loss",
    "delta_vs_random_accuracy",
    "packaged_winner_hash_before",
    "packaged_winner_hash_after",
    "packaged_winner_hash_unchanged",
    "OPEN_VOCAB_CHAT_LM_FOUNDATION_POSITIVE",
    "UPSTREAM_089B_WINNER_PROOF_VERIFIED",
    "BYTE_LEVEL_TOKENIZER_BUILT",
    "OPEN_VOCAB_NEXT_BYTE_TRAINING_COMPLETED",
    "TOKEN_OBJECTIVE_LEARNED",
    "MAIN_BEATS_CONTROLS",
    "OPEN_VOCAB_GENERATION_SMOKE_PASSES",
    "BOUNDED_CHAT_RETENTION_PASSES",
    "FINITE_LABEL_ANCHORROUTE_RETENTION_PASSES",
    "LEAKAGE_AUDIT_PASSES",
    "COLLAPSE_REJECTED",
    "NO_TRAINING_ON_PACKAGED_CHECKPOINT",
    "PRODUCTION_CHAT_NOT_CLAIMED",
    "GPT_LIKE_READINESS_NOT_CLAIMED",
    "OPEN_VOCAB_LM_FOUNDATION_FAILS",
    "UPSTREAM_089B_ARTIFACT_MISSING",
    "UPSTREAM_089B_NOT_POSITIVE",
    "TOKENIZER_BUILD_FAILS",
    "DATASET_BUILD_FAILS",
    "BASELINE_EVAL_MISMATCH",
    "TRAIN_EVAL_LEAKAGE_DETECTED",
    "NO_ACTUAL_TRAINING_UPDATE_DETECTED",
    "TOKEN_OBJECTIVE_NOT_LEARNED",
    "CONTROL_DELTA_INSUFFICIENT",
    "OPEN_VOCAB_GENERATION_SMOKE_FAILS",
    "BOUNDED_CHAT_RETENTION_REGRESSION_DETECTED",
    "FINITE_LABEL_RETENTION_REGRESSION_DETECTED",
    "EMPTY_OUTPUT_COLLAPSE_DETECTED",
    "STATIC_RESPONSE_COLLAPSE_DETECTED",
    "REPETITION_COLLAPSE_DETECTED",
    "PACKAGED_CHECKPOINT_MUTATION_DETECTED",
    "TRAINING_SIDE_EFFECT_ON_PACKAGED_CHECKPOINT",
    "ORACLE_SHORTCUT_DETECTED",
    "LLM_JUDGE_USED",
    "ARCHITECTURE_WINNER_FALSE_CLAIM",
    "RUNTIME_SURFACE_MUTATION_DETECTED",
    "ROOT_LICENSE_CHANGED",
    "GPT_LIKE_READINESS_FALSE_CLAIM",
    "PRODUCTION_CHAT_CLAIM_DETECTED",
    "PUBLIC_RELEASE_CLAIM_DETECTED",
    "092_OPEN_VOCAB_FINEWEB_SLICE_CONFIRM",
    "091B_OPEN_VOCAB_LM_FAILURE_ANALYSIS",
]

NEGATION_MARKERS = ["not ", "no ", "does not ", "do not ", "without ", "false", "forbidden", "forbid", "reject", "rejected", "only"]
FORBIDDEN_CLAIMS = {
    "GPT_LIKE_READINESS_FALSE_CLAIM": ["GPT-like assistant readiness", "itself is now GPT-like"],
    "OPEN_DOMAIN_CLAIM_DETECTED": ["open-domain assistant readiness", "proven as open-domain LM"],
    "PRODUCTION_CHAT_CLAIM_DETECTED": ["production chat"],
    "PUBLIC_RELEASE_CLAIM_DETECTED": ["public release"],
    "SAFETY_ALIGNMENT_CLAIM_DETECTED": ["safety alignment"],
}
PLACEHOLDERS = ["TODO", "TBD", "placeholder", "pending measured", "coming soon"]
GENERATED_PATH_PARTS = ["target/", "node_modules/", ".svelte-kit/"]
GENERATED_SUFFIXES = [".exe", ".dll", ".so", ".dylib", ".zip", ".tar", ".tar.gz", ".tgz", ".7z", ".ckpt", ".bin", ".pt"]


def git_status(paths: list[str] | None = None) -> str:
    cmd = ["git", "status", "--short"]
    if paths:
        cmd.extend(["--", *paths])
    result = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    return result.stdout.rstrip("\n")


def changed_paths() -> list[str]:
    return [line[3:].replace("\\", "/") for line in git_status().splitlines() if line.strip()]


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
        window = lowered[max(0, match.start() - 120) : match.start()]
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
    if "OPEN_VOCAB_CHAT_LM_FOUNDATION_POSITIVE" not in summary.get("verdicts", []):
        failures.append("POSITIVE_VERDICT_MISSING")
    for key in ["runner_local_pytorch_lm", "packaged_winner_used_for_retention_reference"]:
        if summary.get(key) is not True:
            failures.append(f"SUMMARY_FLAG_NOT_TRUE:{key}")
    for key in ["packaged_winner_checkpoint_trained", "architecture_winner_for_open_vocab_claimed", "prediction_oracle_used", "llm_judge_used", "production_chat_claimed", "public_release_claimed", "safety_alignment_claimed"]:
        if summary.get(key) is not False:
            failures.append(f"SUMMARY_FLAG_NOT_FALSE:{key}")

    dataset = load_json(SMOKE_ROOT / "dataset_manifest.json")
    if dataset.get("train_eval_exact_text_overlap_count") != 0:
        failures.append("TRAIN_EVAL_LEAKAGE_DETECTED")
    if dataset.get("max_train_eval_jaccard", 1.0) >= 0.90:
        failures.append("TRAIN_EVAL_LEAKAGE_DETECTED")

    tokenizer = load_json(SMOKE_ROOT / "tokenizer_manifest.json")
    if tokenizer.get("tokenizer_type") != "byte_level" or tokenizer.get("vocab_size", 0) < 259:
        failures.append("TOKENIZER_BUILD_FAILS")

    checkpoint = load_json(SMOKE_ROOT / "checkpoint_manifest.json")
    if checkpoint.get("train_step_count", 0) <= 0:
        failures.append("NO_ACTUAL_TRAINING_UPDATE_DETECTED")
    if checkpoint.get("checkpoint_after_hash") == checkpoint.get("checkpoint_before_hash"):
        failures.append("NO_ACTUAL_TRAINING_UPDATE_DETECTED")
    if not checkpoint.get("train_loss_final", 1e9) < checkpoint.get("train_loss_initial", -1e9):
        failures.append("TOKEN_OBJECTIVE_NOT_LEARNED")

    deltas = load_json(SMOKE_ROOT / "control_delta_report.json")
    if not deltas.get("main_beats_controls"):
        failures.append("CONTROL_DELTA_INSUFFICIENT")
    if deltas.get("delta_vs_char_bigram_loss", -1.0) <= 0:
        failures.append("CONTROL_DELTA_INSUFFICIENT")
    if deltas.get("delta_vs_shuffled_target_loss", -1.0) < 0.25:
        failures.append("CONTROL_DELTA_INSUFFICIENT")
    if deltas.get("delta_vs_random_accuracy", -1.0) < 0.10:
        failures.append("CONTROL_DELTA_INSUFFICIENT")

    arms = load_json(SMOKE_ROOT / "arm_comparison.json")
    required_arms = {"OPEN_VOCAB_BYTE_LM_MAIN", "CHAR_BIGRAM_BASELINE", "RANDOM_BYTE_CONTROL", "SHUFFLED_TARGET_CONTROL", "BOUNDED_ONLY_TRAIN_CONTROL", "NO_ANCHOR_RETENTION_MIX_CONTROL", "STATIC_OUTPUT_CONTROL", "COPY_PROMPT_CONTROL"}
    present = {row.get("arm") for row in arms.get("arms", [])}
    if required_arms - present:
        failures.append(f"ARM_MISSING:{','.join(sorted(required_arms - present))}")
    hashes = {(row.get("eval_row_hash"), row.get("eval_token_hash"), row.get("eval_token_count")) for row in arms.get("arms", [])}
    if len(hashes) != 1:
        failures.append("BASELINE_EVAL_MISMATCH")

    generation = load_json(SMOKE_ROOT / "open_vocab_generation_metrics.json")
    if generation.get("nonempty_generation_rate", 0.0) < 0.98:
        failures.append("OPEN_VOCAB_GENERATION_SMOKE_FAILS")
    if generation.get("utf8_valid_generation_rate", 0.0) < 0.80:
        failures.append("OPEN_VOCAB_GENERATION_SMOKE_FAILS")
    collapse = load_json(SMOKE_ROOT / "collapse_metrics.json")
    if collapse.get("empty_output_rate", 1.0) > 0.02:
        failures.append("EMPTY_OUTPUT_COLLAPSE_DETECTED")
    if collapse.get("static_output_rate", 1.0) > 0.15:
        failures.append("STATIC_RESPONSE_COLLAPSE_DETECTED")
    if collapse.get("repetition_rate", 1.0) > 0.25:
        failures.append("REPETITION_COLLAPSE_DETECTED")

    retention = load_json(SMOKE_ROOT / "bounded_retention_metrics.json")
    if not retention.get("packaged_winner_hash_unchanged") or not retention.get("no_training_on_packaged_checkpoint"):
        failures.append("PACKAGED_CHECKPOINT_MUTATION_DETECTED")
    if retention.get("bounded_chat_slot_binding_accuracy", 0.0) < 0.80 or retention.get("unsupported_refusal_accuracy", 0.0) < 0.80:
        failures.append("BOUNDED_CHAT_RETENTION_REGRESSION_DETECTED")
    if retention.get("finite_label_anchorroute_retention_accuracy", 0.0) < 0.90:
        failures.append("FINITE_LABEL_RETENTION_REGRESSION_DETECTED")

    samples = read_jsonl(SMOKE_ROOT / "human_readable_samples.jsonl")
    families = {row.get("eval_family") for row in samples}
    for family in ["short English continuation", "unseen word continuation", "mixed punctuation continuation", "number/text continuation", "simple dialogue continuation", "unsupported-domain refusal continuation"]:
        if family not in families:
            failures.append(f"HUMAN_SAMPLE_FAMILY_MISSING:{family}")
    for field in ["arm", "prompt", "generated_text", "expected_behavior", "pass_fail", "utf8_valid", "nonempty", "repetition_flag", "copy_prompt_flag", "bounded_retention_flag", "short_diagnosis"]:
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
    for marker in PLACEHOLDERS:
        if marker.lower() in audited_text.lower():
            failures.append(f"PLACEHOLDER_PRESENT:{marker}")
    failures.extend(find_false_claims(audited_text))
    failures.extend(artifact_checks())
    if git_status(["LICENSE"]):
        failures.append("ROOT_LICENSE_CHANGED")
    if runtime_surface_mutation_detected():
        failures.append("RUNTIME_SURFACE_MUTATION_DETECTED")
    if generated_artifact_staged():
        failures.append("GENERATED_ARTIFACT_STAGED")
    payload = {
        "schema_version": "open_vocab_chat_lm_foundation_check_v1",
        "status": "passed" if not failures else "failed",
        "check_only": args.check_only,
        "verdicts": [
            "OPEN_VOCAB_CHAT_LM_FOUNDATION_STATIC_CHECK_POSITIVE",
            "OPEN_VOCAB_LM_FILES_WRITTEN",
            "ARCHITECTURE_CLAIM_BOUNDARY_CHECKED",
            "DATASET_SPLIT_CHECKED",
            "BASELINE_CONTROLS_CHECKED",
            "GENERATION_COLLAPSE_CHECKED",
            "BOUNDED_RETENTION_CHECKED",
            "ROOT_LICENSE_UNCHANGED",
            "PRODUCTION_CHAT_NOT_CLAIMED",
        ]
        if not failures
        else ["OPEN_VOCAB_CHAT_LM_FOUNDATION_STATIC_CHECK_FAILS", *failures],
        "checked_files": REQUIRED_DOCS + REQUIRED_SOURCE,
        "allowed_mutations": sorted(ALLOWED_MUTATIONS),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
