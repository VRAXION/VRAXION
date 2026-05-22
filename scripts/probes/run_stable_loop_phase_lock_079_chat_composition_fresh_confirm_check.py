#!/usr/bin/env python3
"""Static checker for STABLE_LOOP_PHASE_LOCK_079 chat composition fresh confirm."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]

REQUIRED_DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_079_CHAT_COMPOSITION_FRESH_CONFIRM_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_079_CHAT_COMPOSITION_FRESH_CONFIRM_RESULT.md",
]

REQUIRED_SOURCE = [
    "instnct-core/examples/phase_lane_chat_composition_fresh_confirm.rs",
    "scripts/probes/run_stable_loop_phase_lock_079_chat_composition_fresh_confirm_check.py",
]

ALLOWED_MUTATIONS = set(REQUIRED_DOCS + REQUIRED_SOURCE)

EXACT_COMMANDS = [
    "cargo check -p instnct-core --example phase_lane_chat_composition_fresh_confirm",
    "cargo run -p instnct-core --example phase_lane_chat_composition_fresh_confirm -- --out target/pilot_wave/stable_loop_phase_lock_079_chat_composition_fresh_confirm/smoke --upstream-078-root target/pilot_wave/stable_loop_phase_lock_078_chat_composition_repair/smoke --upstream-077b-root target/pilot_wave/stable_loop_phase_lock_077b_chat_generation_failure_analysis/smoke --upstream-076-root target/pilot_wave/stable_loop_phase_lock_076_chat_generation_poc/smoke --upstream-074-root target/pilot_wave/stable_loop_phase_lock_074_multi_seed_scenario_gated_repair_confirm/smoke --seed 2027 --heartbeat-sec 20",
    "python -m py_compile scripts/probes/run_stable_loop_phase_lock_079_chat_composition_fresh_confirm_check.py",
    "python scripts/probes/run_stable_loop_phase_lock_079_chat_composition_fresh_confirm_check.py --check-only",
    "python scripts/probes/run_stable_loop_phase_lock_078_chat_composition_repair_check.py --check-only",
    "python scripts/probes/run_stable_loop_phase_lock_077b_chat_generation_failure_analysis_check.py --check-only",
    "git diff --check",
]

BOUNDARY_TOKENS = [
    "bounded fresh chat composition confirm only",
    "no training",
    "no resume",
    "no checkpoint repair",
    "no checkpoint mutation",
    "no replacement checkpoint",
    "no decoder weight update",
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
    "STABLE_LOOP_PHASE_LOCK_079_CHAT_COMPOSITION_FRESH_CONFIRM",
    "phase_lane_chat_composition_fresh_confirm.rs",
    "run_stable_loop_phase_lock_079_chat_composition_fresh_confirm_check.py",
    "UPSTREAM_078_ARTIFACT_MISSING",
    "STALE_CHECKPOINT_ARTIFACT_USED",
    "TRAINING_SIDE_EFFECT_DETECTED",
    "CHECKPOINT_MUTATION_DETECTED",
    "ORACLE_SHORTCUT_DETECTED",
    "FRESH_PROMPT_LEAKAGE_DETECTED",
    "TEMPLATE_COPY_DETECTED",
    "RESPONSE_TABLE_PATH_USED_IN_EVAL",
    "LABEL_ONLY_RESPONSE_COLLAPSE_DETECTED",
    "FRESH_INSTRUCTION_COMPOSITION_FAILS",
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
    "CHAT_COMPOSITION_FRESH_CONFIRM_POSITIVE",
    "NO_TRAINING_PERFORMED",
    "CHECKPOINT_UNCHANGED",
    "FRESH_MULTI_TOKEN_RESPONSES_PASS",
    "FRESH_INSTRUCTION_COMPOSITION_PASSES",
    "FRESH_CONTEXT_SLOT_BINDING_PASSES",
    "FRESH_TEMPLATE_COPY_REJECTED",
    "FRESH_NOVEL_RESPONSES_PASS",
    "FINITE_LABEL_ANCHORROUTE_RETENTION_PASSES",
    "STATIC_RESPONSE_COLLAPSE_REJECTED",
    "PRODUCTION_CHAT_NOT_CLAIMED",
    "queue.json",
    "progress.jsonl",
    "benchmark_config.json",
    "upstream_078_manifest.json",
    "checkpoint_manifest.json",
    "fresh_chat_eval_dataset.jsonl",
    "generation_samples.jsonl",
    "human_readable_samples.jsonl",
    "composition_metrics.json",
    "novelty_metrics.json",
    "context_slot_metrics.json",
    "finite_label_retention_metrics.json",
    "collapse_metrics.json",
    "summary.json",
    "report.md",
    "upstream_078_summary_present = true",
    "upstream_078_positive = true",
    "checkpoint_exists = true",
    "eval_started_after_079_start = true",
    "checkpoint_hash_before",
    "checkpoint_hash_after",
    "checkpoint_hash_unchanged = true",
    "train_step_count = 0",
    "prediction_oracle_used = false",
    "llm_judge_used = false",
    "decoder_path = token_level_next_token",
    "response_table_used_for_main_prediction = false",
    "overlap_with_078_train_prompt_count",
    "overlap_with_078_eval_prompt_count",
    "overlap_with_077_prompt_count",
    "overlap_with_076_prompt_count",
    "max_prompt_token_jaccard_vs_078_train",
    "max_prompt_token_jaccard_vs_078_eval",
    "max_prompt_token_jaccard_vs_077",
    "max_prompt_token_jaccard_vs_076",
    "near_duplicate_prompt_count",
    "exact_train_response_copy_rate",
    "exact_eval_response_copy_rate",
    "response_table_copy_rate",
    "semantic_template_overlap_rate",
    "template_copy_rate",
    "novel_response_rate",
    "multi_token_response_rate",
    "non_empty_response_rate",
    "fresh_instruction_accuracy",
    "fresh_context_carry_accuracy",
    "slot_binding_accuracy",
    "two_turn_dialogue_accuracy",
    "boundary_refusal_accuracy",
    "label_only_response_rate",
    "generated_token_count_mean",
    "generated_token_count_min",
    "sentence_like_response_rate",
    "empty_output_rate",
    "space_output_rate",
    "static_response_rate",
    "repetition_rate",
    "copy_prompt_rate",
    "finite_label_retention_accuracy",
    "slot_value_expected",
    "slot_value_emitted",
    "wrong_slot_rate",
    "missing_slot_rate",
    "stale_slot_rate",
    "semantic_template_overlap_score",
    "novelty_flag",
    "template_copy_flag",
    "slot_binding_diagnosis",
    "active scenario binding",
    "distractor scenario rejection",
    "old/stale/inactive suppression",
    "answer-only scenario binding",
    "080_CHAT_COMPOSITION_MULTI_SEED_CONFIRM",
    "079B_CHAT_COMPOSITION_FRESH_FAILURE_ANALYSIS",
]

PLACEHOLDERS = ["TODO", "TBD", "placeholder", "pending measured", "coming soon"]

FORBIDDEN_CLAIMS = {
    "GPT_LIKE_READINESS_FALSE_CLAIM": ["GPT-like assistant readiness"],
    "FULL_ENGLISH_LM_CLAIM_DETECTED": ["full English LM"],
    "LANGUAGE_GROUNDING_CLAIM_DETECTED": ["language grounding"],
    "PRODUCTION_CHAT_CLAIM_DETECTED": ["production chat"],
    "SAFETY_ALIGNMENT_CLAIM_DETECTED": ["safety alignment"],
    "GA_CLAIM_DETECTED": ["GA release", "GA launched", "generally available"],
    "PUBLIC_BETA_CLAIM_DETECTED": ["public beta"],
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
    "CHAT_COMPOSITION_FRESH_CONFIRM_STATIC_CHECK_POSITIVE",
    "CHAT_COMPOSITION_FRESH_CONFIRM_PACKAGE_WRITTEN",
    "EVAL_ONLY_GUARD_WRITTEN",
    "FRESH_PROMPT_LEAKAGE_GUARD_WRITTEN",
    "SEMANTIC_TEMPLATE_COPY_AUDIT_REQUIRED",
    "TOKEN_LEVEL_EVAL_PATH_REQUIRED",
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
        if not raw.strip():
            continue
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


def placeholder_hits(files: dict[str, str]) -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    for rel, text in files.items():
        if rel.endswith((".py", ".rs")):
            continue
        for marker in PLACEHOLDERS:
            if re.search(rf"\b{re.escape(marker)}\b", text, flags=re.IGNORECASE):
                hits.append({"file": rel, "marker": marker})
    return hits


def line_is_negated(line: str, phrase: str) -> bool:
    phrase_start = line.find(phrase.lower())
    if phrase_start < 0:
        return False
    return any(marker in line[:phrase_start] for marker in NEGATION_MARKERS)


def forbidden_claim_hits(files: dict[str, str]) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for rel, text in files.items():
        if rel.endswith((".py", ".rs")):
            continue
        for idx, raw_line in enumerate(text.splitlines(), start=1):
            line = raw_line.lower()
            for verdict, phrases in FORBIDDEN_CLAIMS.items():
                for phrase in phrases:
                    phrase_lower = phrase.lower()
                    if phrase_lower in line and not line_is_negated(line, phrase_lower):
                        hits.append({"file": rel, "line": idx, "verdict": verdict, "phrase": phrase})
    return hits


def missing_commands(files: dict[str, str]) -> list[str]:
    text = "\n".join(files.values())
    return [command for command in EXACT_COMMANDS if command not in text]


def missing_boundary_tokens(files: dict[str, str]) -> list[str]:
    text = "\n".join(files.values())
    return [token for token in BOUNDARY_TOKENS if token not in text]


def missing_required_terms(files: dict[str, str]) -> list[str]:
    text = "\n".join(files.values())
    return [term for term in REQUIRED_TERMS if term not in text]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    missing_docs, files = read_files()
    placeholders = placeholder_hits(files)
    command_misses = missing_commands(files)
    boundary_misses = missing_boundary_tokens(files)
    forbidden = forbidden_claim_hits(files)
    required_misses = missing_required_terms(files)
    license_changed = root_license_changed()
    runtime_changed = runtime_surface_mutation_detected()
    generated_staged = generated_artifact_staged()

    check_pass = not any(
        [
            missing_docs,
            placeholders,
            command_misses,
            boundary_misses,
            forbidden,
            required_misses,
            license_changed,
            runtime_changed,
            generated_staged,
        ]
    )
    verdicts = POSITIVE_VERDICTS.copy() if check_pass else ["CHAT_COMPOSITION_FRESH_CONFIRM_STATIC_CHECK_FAILS"]
    if runtime_changed:
        verdicts.append("RUNTIME_SURFACE_MUTATION_DETECTED")
    if license_changed:
        verdicts.append("ROOT_LICENSE_CHANGED")
    if forbidden:
        verdicts.extend(sorted({hit["verdict"] for hit in forbidden}))

    payload = {
        "check_pass": check_pass,
        "missing_docs": missing_docs,
        "placeholder_hits": placeholders,
        "missing_commands": command_misses,
        "missing_boundary_tokens": boundary_misses,
        "forbidden_claim_hits": forbidden,
        "generated_artifact_staged": generated_staged,
        "root_license_changed": license_changed,
        "runtime_surface_mutation_detected": runtime_changed,
        "missing_required_terms": required_misses,
        "verdicts": verdicts,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.check_only and not check_pass:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
