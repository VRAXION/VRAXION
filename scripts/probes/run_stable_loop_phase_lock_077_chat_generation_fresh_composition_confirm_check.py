#!/usr/bin/env python3
"""Static checker for STABLE_LOOP_PHASE_LOCK_077 fresh chat composition confirm."""

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
    "docs/research/STABLE_LOOP_PHASE_LOCK_077_CHAT_GENERATION_FRESH_COMPOSITION_CONFIRM_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_077_CHAT_GENERATION_FRESH_COMPOSITION_CONFIRM_RESULT.md",
]

REQUIRED_SOURCE = [
    "instnct-core/examples/phase_lane_chat_generation_fresh_composition_confirm.rs",
    "scripts/probes/run_stable_loop_phase_lock_077_chat_generation_fresh_composition_confirm_check.py",
]

ALLOWED_MUTATIONS = set(REQUIRED_DOCS + REQUIRED_SOURCE)

EXACT_COMMANDS = [
    "cargo check -p instnct-core --example phase_lane_chat_generation_fresh_composition_confirm",
    "cargo run -p instnct-core --example phase_lane_chat_generation_fresh_composition_confirm -- --out target/pilot_wave/stable_loop_phase_lock_077_chat_generation_fresh_composition_confirm/smoke --checkpoint target/pilot_wave/stable_loop_phase_lock_076_chat_generation_poc/smoke/checkpoints/chat_generation_poc/model_checkpoint.json --upstream-076-root target/pilot_wave/stable_loop_phase_lock_076_chat_generation_poc/smoke --upstream-074-root target/pilot_wave/stable_loop_phase_lock_074_multi_seed_scenario_gated_repair_confirm/smoke --seed 2027 --heartbeat-sec 20",
    "python -m py_compile scripts/probes/run_stable_loop_phase_lock_077_chat_generation_fresh_composition_confirm_check.py",
    "python scripts/probes/run_stable_loop_phase_lock_077_chat_generation_fresh_composition_confirm_check.py --check-only",
    "python scripts/probes/run_stable_loop_phase_lock_076_chat_generation_poc_check.py --check-only",
    "python scripts/probes/run_stable_loop_phase_lock_074_multi_seed_scenario_gated_repair_confirm_check.py --check-only",
    "git diff --check",
]

BOUNDARY_TOKENS = [
    "bounded fresh composition confirm only",
    "no training",
    "no resume",
    "no checkpoint repair",
    "no checkpoint mutation",
    "no replacement checkpoint",
    "not GPT-like assistant readiness",
    "not full English LM",
    "not language grounding",
    "not production chat",
    "not public beta / GA / hosted SaaS",
    "no service API change",
    "no deployment harness change",
    "no SDK/public export change",
    "no release docs change",
    "no root LICENSE change",
]

REQUIRED_TERMS = [
    "STABLE_LOOP_PHASE_LOCK_077_CHAT_GENERATION_FRESH_COMPOSITION_CONFIRM",
    "phase_lane_chat_generation_fresh_composition_confirm.rs",
    "upstream_076_summary_present = true",
    "upstream_076_positive = true",
    "checkpoint_exists = true",
    "child_eval_started_after_077_start = true",
    "checkpoint_hash_before",
    "checkpoint_hash_after",
    "checkpoint_hash_unchanged = true",
    "train_step_count = 0",
    "prediction_oracle_used = false",
    "response_uses_decoder_loop = true",
    "UPSTREAM_076_ARTIFACT_MISSING",
    "STALE_CHECKPOINT_ARTIFACT_USED",
    "CHECKPOINT_MUTATION_DETECTED",
    "TRAINING_SIDE_EFFECT_DETECTED",
    "ORACLE_SHORTCUT_DETECTED",
    "FRESH_SIMPLE_INSTRUCTION",
    "FRESH_SHORT_EXPLANATION",
    "FRESH_CONTEXT_CARRY_CHAT",
    "FRESH_TWO_TURN_DIALOGUE",
    "FRESH_BOUNDARY_REFUSAL_MINI",
    "FRESH_COMPOSITION_NOVELTY",
    "ANTI_TEMPLATE_COPY",
    "ANTI_REPETITION",
    "FINITE_LABEL_ANCHORROUTE_RETENTION",
    "new wording",
    "new entities",
    "new instruction shapes",
    "new context-carry variants",
    "new refusal/boundary phrasing",
    "no exact prompt overlap with 076 train/eval prompts",
    "queue.json",
    "progress.jsonl",
    "benchmark_config.json",
    "upstream_076_manifest.json",
    "upstream_074_manifest.json",
    "checkpoint_manifest.json",
    "fresh_chat_eval_dataset.jsonl",
    "generation_samples.jsonl",
    "human_readable_samples.jsonl",
    "composition_metrics.json",
    "novelty_metrics.json",
    "collapse_metrics.json",
    "finite_label_retention_metrics.json",
    "summary.json",
    "report.md",
    "exact_train_response_copy_rate",
    "exact_eval_response_copy_rate",
    "response_table_copy_rate",
    "template_copy_rate",
    "train_response_ngram_overlap",
    "novel_response_rate",
    "prompt_ngram_overlap_stats",
    "train_eval_exact_prompt_overlap_count = 0",
    "multi_token_response_rate",
    "non_empty_response_rate",
    "fresh_instruction_accuracy",
    "fresh_context_carry_accuracy",
    "two_turn_dialogue_accuracy",
    "boundary_refusal_accuracy",
    "label_only_response_rate",
    "generated_token_count_mean",
    "generated_token_count_min",
    "unique_response_count",
    "empty_output_rate",
    "space_output_rate",
    "top_response_rate",
    "static_response_rate",
    "repetition_rate",
    "copy_prompt_rate",
    "eval_family",
    "prompt",
    "expected_behavior",
    "required_keywords",
    "forbidden_outputs",
    "model_output",
    "output_classification",
    "pass_fail",
    "short_diagnosis",
    "template_copy_flag",
    "novelty_flag",
    "active scenario binding",
    "distractor scenario rejection",
    "old/stale/inactive suppression",
    "answer-only scenario binding",
    "multi_token_response_rate >= 0.90",
    "non_empty_response_rate >= 0.98",
    "fresh_instruction_accuracy >= 0.70",
    "fresh_context_carry_accuracy >= 0.65",
    "novel_response_rate >= 0.60",
    "template_copy_rate <= 0.30",
    "label_only_response_rate <= 0.20",
    "generated_token_count_min >= 2",
    "empty_output_rate <= 0.02",
    "space_output_rate <= 0.02",
    "static_response_rate <= 0.15",
    "repetition_rate <= 0.20",
    "finite_label_retention_accuracy >= 0.90",
    "CHAT_GENERATION_FRESH_COMPOSITION_CONFIRM_POSITIVE",
    "FRESH_MULTI_TOKEN_RESPONSES_PASS",
    "FRESH_INSTRUCTION_FOLLOWING_PASSES",
    "FRESH_CONTEXT_CARRY_CHAT_PASSES",
    "TEMPLATE_COPY_REJECTED",
    "STATIC_RESPONSE_COLLAPSE_REJECTED",
    "FINITE_LABEL_RETENTION_PASSES",
    "NO_TRAINING_PERFORMED",
    "PRODUCTION_CHAT_NOT_CLAIMED",
    "CHAT_GENERATION_FRESH_COMPOSITION_CONFIRM_FAILS",
    "FRESH_INSTRUCTION_FOLLOWING_FAILS",
    "FRESH_CONTEXT_CARRY_CHAT_FAILS",
    "TEMPLATE_COPY_DETECTED",
    "LABEL_ONLY_RESPONSE_COLLAPSE_DETECTED",
    "CHAT_GENERATION_SURFACE_STILL_TOO_TABLE_LIKE",
    "STATIC_RESPONSE_COLLAPSE_DETECTED",
    "REPETITION_COLLAPSE_DETECTED",
    "EMPTY_OUTPUT_COLLAPSE_DETECTED",
    "FINITE_LABEL_RETENTION_REGRESSION_DETECTED",
    "TRAIN_EVAL_LEAKAGE_DETECTED",
    "HUMAN_SAMPLE_REPORT_MISSING",
    "OPEN_ENDED_ASSISTANT_CLAIM_DETECTED",
    "GPT_LIKE_READINESS_FALSE_CLAIM",
    "PRODUCTION_CHAT_CLAIM_DETECTED",
    "RUNTIME_SURFACE_MUTATION_DETECTED",
    "ROOT_LICENSE_CHANGED",
    "078_CHAT_GENERATION_MULTI_SEED_CONFIRM",
    "077B_CHAT_GENERATION_FAILURE_ANALYSIS",
]

PLACEHOLDERS = ["TODO", "TBD", "placeholder", "pending measured", "coming soon"]

FORBIDDEN_CLAIMS = {
    "OPEN_ENDED_ASSISTANT_CLAIM_DETECTED": ["open-ended assistant"],
    "GPT_LIKE_READINESS_FALSE_CLAIM": ["GPT-like assistant readiness"],
    "FULL_ENGLISH_LM_CLAIM_DETECTED": ["full English LM"],
    "LANGUAGE_GROUNDING_CLAIM_DETECTED": ["language grounding"],
    "PRODUCTION_CHAT_CLAIM_DETECTED": ["production chat"],
    "GA_CLAIM_DETECTED": ["GA release", "GA launched", "generally available"],
    "PUBLIC_BETA_CLAIM_DETECTED": ["public beta"],
    "HOSTED_SAAS_CLAIM_DETECTED": ["hosted SaaS"],
    "SERVICE_API_CLAIM_DETECTED": ["service API change"],
    "DEPLOYMENT_HARNESS_CLAIM_DETECTED": ["deployment harness change"],
    "SDK_EXPORT_CLAIM_DETECTED": ["SDK/public export change"],
    "ROOT_LICENSE_CHANGED": ["root LICENSE changed"],
}

NEGATION_MARKERS = ["not ", "no ", "does not ", "do not ", "without ", "unsupported", "false", "reject", "rejected", "bounded"]
GENERATED_PATH_PARTS = ["target/", "node_modules/", ".svelte-kit/"]
GENERATED_SUFFIXES = [".exe", ".dll", ".so", ".dylib", ".zip", ".tar", ".tar.gz", ".tgz", ".7z", ".ckpt", ".bin"]
GENERATED_NAME_PARTS = ["checkpoint", "smoke/", "service job artifacts", "release artifacts"]

POSITIVE_VERDICTS = [
    "CHAT_GENERATION_FRESH_COMPOSITION_STATIC_CHECK_POSITIVE",
    "CHAT_GENERATION_FRESH_COMPOSITION_PACKAGE_WRITTEN",
    "EVAL_ONLY_GUARD_WRITTEN",
    "TEMPLATE_COPY_DETECTION_REQUIRED",
    "FRESH_PROMPT_LEAKAGE_GUARD_WRITTEN",
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


def changed_paths() -> list[str]:
    paths: list[str] = []
    for raw in git_status().splitlines():
        if not raw.strip():
            continue
        paths.append(raw[3:].replace("\\", "/"))
    return paths


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
        if path.startswith("target/") and any(part in path for part in GENERATED_NAME_PARTS):
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
    prefix = line[:phrase_start]
    return any(marker in prefix for marker in NEGATION_MARKERS)


def forbidden_claim_hits(files: dict[str, str]) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for rel, text in files.items():
        if rel.endswith((".py", ".rs")):
            continue
        for idx, raw_line in enumerate(text.splitlines(), start=1):
            line = raw_line.lower()
            if not line.strip():
                continue
            for verdict, phrases in FORBIDDEN_CLAIMS.items():
                for phrase in phrases:
                    lowered = phrase.lower()
                    if lowered in line and not line_is_negated(line, lowered):
                        hits.append({"file": rel, "line": idx, "phrase": phrase, "verdict": verdict})
    return hits


def missing_commands(files: dict[str, str]) -> list[str]:
    bundle = "\n".join(files.values())
    return [command for command in EXACT_COMMANDS if command not in bundle]


def missing_boundary_tokens(files: dict[str, str]) -> list[dict[str, str]]:
    missing: list[dict[str, str]] = []
    for rel in REQUIRED_DOCS:
        text = files.get(rel, "")
        for token in BOUNDARY_TOKENS:
            if token not in text:
                missing.append({"file": rel, "token": token})
    return missing


def missing_required_terms(files: dict[str, str]) -> list[str]:
    bundle = "\n".join(files.values())
    return [term for term in REQUIRED_TERMS if term not in bundle]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if not args.check_only:
        parser.error("Only --check-only is supported; this checker validates committed files only.")

    missing_docs, files = read_files()
    placeholders = placeholder_hits(files)
    commands = missing_commands(files)
    boundaries = missing_boundary_tokens(files)
    forbidden = forbidden_claim_hits(files)
    generated = generated_artifact_staged()
    license_changed = root_license_changed()
    runtime_mutation = runtime_surface_mutation_detected()
    required = missing_required_terms(files)

    check_pass = not any([
        missing_docs,
        placeholders,
        commands,
        boundaries,
        forbidden,
        generated,
        license_changed,
        runtime_mutation,
        required,
    ])

    verdicts = POSITIVE_VERDICTS.copy() if check_pass else ["CHAT_GENERATION_FRESH_COMPOSITION_STATIC_CHECK_FAILS"]
    if generated:
        verdicts.append("GENERATED_ARTIFACT_STAGED")
    if license_changed:
        verdicts.append("ROOT_LICENSE_CHANGED")
    if runtime_mutation:
        verdicts.append("RUNTIME_SURFACE_MUTATION_DETECTED")
    for hit in forbidden:
        verdicts.append(hit["verdict"])

    payload = {
        "check_pass": check_pass,
        "missing_docs": missing_docs,
        "placeholder_hits": placeholders,
        "missing_commands": commands,
        "missing_boundary_tokens": boundaries,
        "forbidden_claim_hits": forbidden,
        "generated_artifact_staged": generated,
        "root_license_changed": license_changed,
        "runtime_surface_mutation_detected": runtime_mutation,
        "missing_required_terms": required,
        "verdicts": verdicts,
    }
    print(json.dumps(payload, separators=(",", ":")))
    return 0 if check_pass else 1


if __name__ == "__main__":
    sys.exit(main())
