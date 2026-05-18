#!/usr/bin/env python3
"""Static checker for STABLE_LOOP_PHASE_LOCK_080 chat composition diversity repair."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

REQUIRED_DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_080_CHAT_COMPOSITION_DIVERSITY_REPAIR_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_080_CHAT_COMPOSITION_DIVERSITY_REPAIR_RESULT.md",
]

REQUIRED_SOURCE = [
    "instnct-core/examples/phase_lane_chat_composition_diversity_repair.rs",
    "scripts/probes/run_stable_loop_phase_lock_080_chat_composition_diversity_repair_check.py",
]

ALLOWED_MUTATIONS = set(REQUIRED_DOCS + REQUIRED_SOURCE)

EXACT_COMMANDS = [
    "cargo check -p instnct-core --example phase_lane_chat_composition_diversity_repair",
    "cargo run -p instnct-core --example phase_lane_chat_composition_diversity_repair -- --out target/pilot_wave/stable_loop_phase_lock_080_chat_composition_diversity_repair/smoke --upstream-078-root target/pilot_wave/stable_loop_phase_lock_078_chat_composition_repair/smoke --upstream-079-root target/pilot_wave/stable_loop_phase_lock_079_chat_composition_fresh_confirm/smoke --upstream-079b-root target/pilot_wave/stable_loop_phase_lock_079b_chat_composition_fresh_failure_analysis/smoke --upstream-074-root target/pilot_wave/stable_loop_phase_lock_074_multi_seed_scenario_gated_repair_confirm/smoke --chat-examples 80000 --seed 2026 --heartbeat-sec 20",
    "python -m py_compile scripts/probes/run_stable_loop_phase_lock_080_chat_composition_diversity_repair_check.py",
    "python scripts/probes/run_stable_loop_phase_lock_080_chat_composition_diversity_repair_check.py --check-only",
    "python scripts/probes/run_stable_loop_phase_lock_079b_chat_composition_fresh_failure_analysis_check.py --check-only",
    "python scripts/probes/run_stable_loop_phase_lock_079_chat_composition_fresh_confirm_check.py --check-only",
    "git diff --check",
]

BOUNDARY_TOKENS = [
    "bounded runner-local chat composition diversity repair only",
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
    "STABLE_LOOP_PHASE_LOCK_080_CHAT_COMPOSITION_DIVERSITY_REPAIR",
    "phase_lane_chat_composition_diversity_repair.rs",
    "run_stable_loop_phase_lock_080_chat_composition_diversity_repair_check.py",
    "TOKEN_COMPOSITION_DIVERSITY_REPAIR",
    "NO_REPAIR_078_BASELINE",
    "NO_SKELETON_DROPOUT_CONTROL",
    "NO_LEXICAL_DROPOUT_CONTROL",
    "NO_CLAUSE_RANDOMIZATION_CONTROL",
    "ONE_TARGET_PER_PROMPT_CONTROL",
    "RESPONSE_TABLE_ONLY_CONTROL",
    "FINITE_LABEL_RETENTION_CONTROL",
    "CHECKPOINT_RELOAD_EVAL",
    "RESUME_FROM_CHECKPOINT",
    "UPSTREAM_078_ARTIFACT_MISSING",
    "UPSTREAM_079B_ARTIFACT_MISSING",
    "NO_ACTUAL_TRAINING_UPDATE_DETECTED",
    "TOKEN_OBJECTIVE_NOT_LEARNED",
    "RESPONSE_TABLE_DEPENDENCE_STILL_HIGH",
    "TEMPLATE_COPY_STILL_HIGH",
    "SKELETON_REUSE_STILL_HIGH",
    "VOCAB_DIVERSITY_TOO_LOW",
    "CONTEXT_SLOT_BINDING_STILL_FAILS",
    "BOUNDARY_REFUSAL_MINI_STILL_FAILS",
    "FINITE_LABEL_RETENTION_REGRESSION_DETECTED",
    "STATIC_RESPONSE_COLLAPSE_DETECTED",
    "EMPTY_OUTPUT_COLLAPSE_DETECTED",
    "REPETITION_COLLAPSE_DETECTED",
    "TRAIN_EVAL_LEAKAGE_DETECTED",
    "CONTROL_DELTA_INSUFFICIENT",
    "MANY_TARGET_DIVERSITY_TOO_LOW",
    "CHAT_EVAL_RUBRIC_MISSING",
    "HUMAN_SAMPLE_REPORT_MISSING",
    "CHECKPOINT_RELOAD_FAILS",
    "RESUME_FROM_CHECKPOINT_FAILS",
    "EVAL_AFTER_RELOAD_MISMATCH",
    "RUNTIME_SURFACE_MUTATION_DETECTED",
    "ROOT_LICENSE_CHANGED",
    "UPSTREAM_CHECKPOINT_MUTATION_DETECTED",
    "GPT_LIKE_READINESS_FALSE_CLAIM",
    "PRODUCTION_CHAT_CLAIM_DETECTED",
    "CHAT_COMPOSITION_DIVERSITY_REPAIR_POSITIVE",
    "TOKEN_LEVEL_DIVERSITY_TRAINING_COMPLETED",
    "TOKEN_OBJECTIVE_LEARNED",
    "RESPONSE_TABLE_DEPENDENCE_REDUCED",
    "TEMPLATE_COPY_REJECTED",
    "SKELETON_REUSE_REDUCED",
    "VOCAB_DIVERSITY_IMPROVED",
    "CONTEXT_SLOT_BINDING_RETAINED",
    "BOUNDARY_REFUSAL_MINI_RETAINED",
    "FINITE_LABEL_ANCHORROUTE_RETENTION_PASSES",
    "CONTROL_DELTA_PASSES",
    "CHECKPOINT_PIPELINE_PASSES",
    "UPSTREAM_CHECKPOINT_UNCHANGED",
    "PRODUCTION_CHAT_NOT_CLAIMED",
    "queue.json",
    "progress.jsonl",
    "training_config.json",
    "upstream_manifest.json",
    "diversity_dataset_manifest.json",
    "train_examples_sample.jsonl",
    "eval_examples_sample.jsonl",
    "training_metrics.jsonl",
    "checkpoint_manifest.json",
    "checkpoint_hashes.json",
    "generation_samples.jsonl",
    "human_readable_samples.jsonl",
    "composition_metrics.json",
    "novelty_metrics.json",
    "skeleton_diversity_metrics.json",
    "vocabulary_entropy_metrics.json",
    "context_slot_metrics.json",
    "finite_label_retention_metrics.json",
    "collapse_metrics.json",
    "arm_comparison.json",
    "summary.json",
    "report.md",
    "decoder_path = token_level_next_token",
    "response_table_used_for_main_prediction = false",
    "response_table_path_available_but_disabled = true",
    "skeleton_dropout_enabled = true",
    "lexical_dropout_enabled = true",
    "clause_order_randomization_enabled = true",
    "many_valid_continuation_enabled = true",
    "valid_target_count_per_prompt",
    "mean_valid_targets_per_prompt",
    "min_valid_targets_per_prompt",
    "token_train_step_count",
    "token_loss_initial",
    "token_loss_final",
    "teacher_forced_next_token_accuracy",
    "exact_copy_rate",
    "semantic_template_overlap_rate",
    "slot_only_skeleton_reuse_rate",
    "genuinely_novel_response_rate",
    "response_skeleton_reuse_rate",
    "top_skeleton_rate",
    "response_skeleton_diversity",
    "generated_to_train_vocab_ratio",
    "unique_bigram_count",
    "unique_trigram_count",
    "token_entropy",
    "response_entropy",
    "slot_value_expected",
    "slot_value_emitted",
    "wrong_slot_rate",
    "missing_slot_rate",
    "stale_slot_rate",
    "train_eval_exact_prompt_overlap_count",
    "train_eval_exact_response_overlap_count",
    "train_eval_template_overlap_count",
    "max_train_eval_prompt_jaccard",
    "max_train_eval_response_jaccard",
    "prediction_oracle_used = false",
    "llm_judge_used = false",
    "checkpoint_save_load_pass = true",
    "resume_from_checkpoint_pass = true",
    "eval_after_reload_matches_before = true",
    "upstream_checkpoint_unchanged = true",
    "active scenario binding",
    "distractor scenario rejection",
    "old/stale/inactive suppression",
    "answer-only scenario binding",
    "before_078_style_response",
    "080_diversity_response",
    "why_080_is_or_is_not_more compositional",
    "081_CHAT_DIVERSITY_FRESH_CONFIRM",
    "080B_CHAT_DIVERSITY_REPAIR_FAILURE_ANALYSIS",
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
    "CHAT_COMPOSITION_DIVERSITY_REPAIR_STATIC_CHECK_POSITIVE",
    "CHAT_COMPOSITION_DIVERSITY_REPAIR_PACKAGE_WRITTEN",
    "RUNNER_LOCAL_DIVERSITY_REPAIR_GUARD_WRITTEN",
    "MANY_VALID_CONTINUATION_REQUIRED",
    "SKELETON_DROPOUT_AND_DIVERSITY_REQUIRED",
    "CONTROL_DELTA_REQUIRED",
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
    audited_text = "\n".join(
        text
        for rel, text in files.items()
        if not rel.endswith("_check.py")
    )
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
        "schema_version": "chat_composition_diversity_repair_static_check_v1",
        "status": "passed" if not failures else "failed",
        "check_only": args.check_only,
        "verdicts": POSITIVE_VERDICTS if not failures else ["CHAT_COMPOSITION_DIVERSITY_REPAIR_STATIC_CHECK_FAILS", *failures],
        "checked_files": REQUIRED_DOCS + REQUIRED_SOURCE,
        "allowed_mutations": sorted(ALLOWED_MUTATIONS),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
