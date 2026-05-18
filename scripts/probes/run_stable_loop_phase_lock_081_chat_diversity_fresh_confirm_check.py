#!/usr/bin/env python3
"""Static checker for STABLE_LOOP_PHASE_LOCK_081 chat diversity fresh confirm."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

REQUIRED_DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_081_CHAT_DIVERSITY_FRESH_CONFIRM_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_081_CHAT_DIVERSITY_FRESH_CONFIRM_RESULT.md",
]

REQUIRED_SOURCE = [
    "instnct-core/examples/phase_lane_chat_diversity_fresh_confirm.rs",
    "scripts/probes/run_stable_loop_phase_lock_081_chat_diversity_fresh_confirm_check.py",
]

ALLOWED_MUTATIONS = set(REQUIRED_DOCS + REQUIRED_SOURCE)

EXACT_COMMANDS = [
    "cargo check -p instnct-core --example phase_lane_chat_diversity_fresh_confirm",
    "cargo run -p instnct-core --example phase_lane_chat_diversity_fresh_confirm -- --out target/pilot_wave/stable_loop_phase_lock_081_chat_diversity_fresh_confirm/smoke --upstream-080-root target/pilot_wave/stable_loop_phase_lock_080_chat_composition_diversity_repair/smoke --upstream-079b-root target/pilot_wave/stable_loop_phase_lock_079b_chat_composition_fresh_failure_analysis/smoke --upstream-079-root target/pilot_wave/stable_loop_phase_lock_079_chat_composition_fresh_confirm/smoke --upstream-078-root target/pilot_wave/stable_loop_phase_lock_078_chat_composition_repair/smoke --upstream-074-root target/pilot_wave/stable_loop_phase_lock_074_multi_seed_scenario_gated_repair_confirm/smoke --seed 2027 --heartbeat-sec 20",
    "python -m py_compile scripts/probes/run_stable_loop_phase_lock_081_chat_diversity_fresh_confirm_check.py",
    "python scripts/probes/run_stable_loop_phase_lock_081_chat_diversity_fresh_confirm_check.py --check-only",
    "python scripts/probes/run_stable_loop_phase_lock_080_chat_composition_diversity_repair_check.py --check-only",
    "python scripts/probes/run_stable_loop_phase_lock_079b_chat_composition_fresh_failure_analysis_check.py --check-only",
    "git diff --check",
]

BOUNDARY_TOKENS = [
    "bounded fresh chat diversity confirmation only",
    "not GPT-like assistant readiness",
    "not full English LM",
    "not language grounding",
    "not production chat",
    "not safety alignment",
    "not public beta / GA / hosted SaaS",
    "no service API change",
    "no deployment harness change",
    "no SDK/public export change",
    "no release docs change",
    "no root LICENSE change",
    "no upstream checkpoint mutation",
]

REQUIRED_TERMS = [
    "STABLE_LOOP_PHASE_LOCK_081_CHAT_DIVERSITY_FRESH_CONFIRM",
    "phase_lane_chat_diversity_fresh_confirm.rs",
    "run_stable_loop_phase_lock_081_chat_diversity_fresh_confirm_check.py",
    "UPSTREAM_080_ARTIFACT_MISSING",
    "TRAINING_SIDE_EFFECT_DETECTED",
    "CHECKPOINT_MUTATION_DETECTED",
    "FRESH_PROMPT_LEAKAGE_DETECTED",
    "TEMPLATE_COPY_DETECTED",
    "SKELETON_REUSE_REGRESSION_DETECTED",
    "VOCAB_DIVERSITY_REGRESSION_DETECTED",
    "CONTEXT_SLOT_BINDING_FAILS",
    "FINITE_LABEL_RETENTION_REGRESSION_DETECTED",
    "STATIC_RESPONSE_COLLAPSE_DETECTED",
    "REPETITION_COLLAPSE_DETECTED",
    "EMPTY_OUTPUT_COLLAPSE_DETECTED",
    "HUMAN_SAMPLE_REPORT_MISSING",
    "GPT_LIKE_READINESS_FALSE_CLAIM",
    "PRODUCTION_CHAT_CLAIM_DETECTED",
    "RUNTIME_SURFACE_MUTATION_DETECTED",
    "ROOT_LICENSE_CHANGED",
    "CHAT_DIVERSITY_FRESH_CONFIRM_POSITIVE",
    "NO_TRAINING_PERFORMED",
    "CHECKPOINT_UNCHANGED",
    "FRESH_DIVERSITY_MULTI_TOKEN_RESPONSES_PASS",
    "FRESH_DIVERSITY_INSTRUCTION_PASSES",
    "FRESH_DIVERSITY_CONTEXT_SLOT_BINDING_PASSES",
    "FRESH_DIVERSITY_TEMPLATE_COPY_REJECTED",
    "FRESH_DIVERSITY_SKELETON_REUSE_REJECTED",
    "FRESH_DIVERSITY_VOCAB_ENTROPY_PASSES",
    "FINITE_LABEL_ANCHORROUTE_RETENTION_PASSES",
    "STATIC_RESPONSE_COLLAPSE_REJECTED",
    "PRODUCTION_CHAT_NOT_CLAIMED",
    "queue.json",
    "progress.jsonl",
    "benchmark_config.json",
    "upstream_080_manifest.json",
    "checkpoint_manifest.json",
    "fresh_chat_eval_dataset.jsonl",
    "generation_samples.jsonl",
    "human_readable_samples.jsonl",
    "composition_metrics.json",
    "novelty_metrics.json",
    "skeleton_diversity_metrics.json",
    "vocabulary_entropy_metrics.json",
    "context_slot_metrics.json",
    "finite_label_retention_metrics.json",
    "collapse_metrics.json",
    "summary.json",
    "report.md",
    "train_step_count = 0",
    "checkpoint_hash_before",
    "checkpoint_hash_after",
    "checkpoint_hash_unchanged = true",
    "prediction_oracle_used = false",
    "llm_judge_used = false",
    "decoder_path = token_level_next_token",
    "response_table_used_for_main_prediction = false",
    "eval_started_after_081_start = true",
    "FRESH_DIVERSITY_SIMPLE_INSTRUCTION",
    "FRESH_DIVERSITY_SHORT_EXPLANATION",
    "FRESH_DIVERSITY_CONTEXT_SLOT",
    "FRESH_DIVERSITY_TWO_TURN",
    "FRESH_DIVERSITY_BOUNDARY_MINI",
    "FRESH_DIVERSITY_SEMANTIC_RECOMBINATION",
    "FRESH_ANTI_TEMPLATE_COPY",
    "FRESH_ANTI_SKELETON_REUSE",
    "FRESH_ANTI_REPETITION",
    "FINITE_LABEL_ANCHORROUTE_RETENTION",
    "multi_token_response_rate",
    "non_empty_response_rate",
    "fresh_instruction_accuracy",
    "fresh_context_carry_accuracy",
    "slot_binding_accuracy",
    "two_turn_dialogue_accuracy",
    "boundary_refusal_accuracy",
    "novel_response_rate",
    "template_copy_rate",
    "exact_train_response_copy_rate",
    "exact_eval_response_copy_rate",
    "response_table_copy_rate",
    "semantic_template_overlap_rate",
    "slot_only_skeleton_reuse_rate",
    "response_skeleton_reuse_rate",
    "top_skeleton_rate",
    "response_skeleton_diversity",
    "generated_to_train_vocab_ratio",
    "unique_bigram_count",
    "unique_trigram_count",
    "token_entropy",
    "response_entropy",
    "label_only_response_rate",
    "empty_output_rate",
    "space_output_rate",
    "static_response_rate",
    "repetition_rate",
    "copy_prompt_rate",
    "finite_label_retention_accuracy",
    "overlap_with_080_train_prompt_count",
    "overlap_with_080_eval_prompt_count",
    "overlap_with_079_prompt_count",
    "overlap_with_078_prompt_count",
    "overlap_with_076_prompt_count",
    "near_duplicate_prompt_count",
    "slot_value_expected",
    "slot_value_emitted",
    "wrong_slot_rate",
    "missing_slot_rate",
    "stale_slot_rate",
    "novelty_flag",
    "template_copy_flag",
    "skeleton_reuse_flag",
    "semantic_template_overlap_score",
    "slot_binding_diagnosis",
    "082_CHAT_DIVERSITY_MULTI_SEED_CONFIRM",
    "081B_CHAT_DIVERSITY_FRESH_FAILURE_ANALYSIS",
]

PLACEHOLDERS = ["TODO", "TBD", "placeholder", "pending measured", "coming soon"]

FORBIDDEN_CLAIMS = {
    "GPT_LIKE_READINESS_FALSE_CLAIM": ["GPT-like assistant readiness"],
    "FULL_ENGLISH_LM_CLAIM_DETECTED": ["full English LM"],
    "LANGUAGE_GROUNDING_CLAIM_DETECTED": ["language grounding"],
    "PRODUCTION_CHAT_CLAIM_DETECTED": ["production chat"],
    "SAFETY_ALIGNMENT_CLAIM_DETECTED": ["safety alignment"],
    "PUBLIC_BETA_CLAIM_DETECTED": ["public beta"],
    "GA_CLAIM_DETECTED": ["GA release", "generally available"],
    "HOSTED_SAAS_CLAIM_DETECTED": ["hosted SaaS"],
    "SERVICE_API_CLAIM_DETECTED": ["service API change"],
    "DEPLOYMENT_HARNESS_CLAIM_DETECTED": ["deployment harness change"],
    "SDK_EXPORT_CLAIM_DETECTED": ["SDK/public export change"],
    "ROOT_LICENSE_CHANGED": ["root LICENSE changed"],
}

NEGATION_MARKERS = ["not ", "no ", "does not ", "do not ", "without ", "false", "reject", "rejected", "bounded"]
GENERATED_PATH_PARTS = ["target/", "node_modules/", ".svelte-kit/"]
GENERATED_SUFFIXES = [".exe", ".dll", ".so", ".dylib", ".zip", ".tar", ".tar.gz", ".tgz", ".7z", ".ckpt", ".bin"]

POSITIVE_VERDICTS = [
    "CHAT_DIVERSITY_FRESH_CONFIRM_STATIC_CHECK_POSITIVE",
    "CHAT_DIVERSITY_FRESH_CONFIRM_PACKAGE_WRITTEN",
    "EVAL_ONLY_GUARD_WRITTEN",
    "FRESH_PROMPT_LEAKAGE_GUARD_WRITTEN",
    "TEMPLATE_AND_SKELETON_AUDIT_REQUIRED",
    "VOCAB_ENTROPY_AUDIT_REQUIRED",
    "FINITE_LABEL_RETENTION_REQUIRED",
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
        "schema_version": "chat_diversity_fresh_confirm_static_check_v1",
        "status": "passed" if not failures else "failed",
        "check_only": args.check_only,
        "verdicts": POSITIVE_VERDICTS if not failures else ["CHAT_DIVERSITY_FRESH_CONFIRM_STATIC_CHECK_FAILS", *failures],
        "checked_files": REQUIRED_DOCS + REQUIRED_SOURCE,
        "allowed_mutations": sorted(ALLOWED_MUTATIONS),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
