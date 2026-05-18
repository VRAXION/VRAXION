#!/usr/bin/env python3
"""Static checker for STABLE_LOOP_PHASE_LOCK_076 chat generation PoC."""

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
    "docs/research/STABLE_LOOP_PHASE_LOCK_076_CHAT_GENERATION_POC_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_076_CHAT_GENERATION_POC_RESULT.md",
]

REQUIRED_SOURCE = [
    "instnct-core/examples/phase_lane_chat_generation_poc.rs",
    "scripts/probes/run_stable_loop_phase_lock_076_chat_generation_poc_check.py",
]

ALLOWED_MUTATIONS = set(REQUIRED_DOCS + REQUIRED_SOURCE)

EXACT_COMMANDS = [
    "cargo check -p instnct-core --example phase_lane_chat_generation_poc",
    "cargo run -p instnct-core --example phase_lane_chat_generation_poc -- --out target/pilot_wave/stable_loop_phase_lock_076_chat_generation_poc/smoke --upstream-checkpoint target/pilot_wave/stable_loop_phase_lock_072_counterfactual_scenario_binding_repair/smoke/checkpoints/scenario_gated_sidepacket_repair/model_checkpoint.json --upstream-074-root target/pilot_wave/stable_loop_phase_lock_074_multi_seed_scenario_gated_repair_confirm/smoke --upstream-075-root target/pilot_wave/stable_loop_phase_lock_075_chat_surface_baseline_gap_analysis/smoke --seed 2026 --chat-examples 20000 --heartbeat-sec 20",
    "python -m py_compile scripts/probes/run_stable_loop_phase_lock_076_chat_generation_poc_check.py",
    "python scripts/probes/run_stable_loop_phase_lock_076_chat_generation_poc_check.py --check-only",
    "python scripts/probes/run_stable_loop_phase_lock_075_chat_surface_baseline_gap_analysis_check.py --check-only",
    "python scripts/probes/run_stable_loop_phase_lock_074_multi_seed_scenario_gated_repair_confirm_check.py --check-only",
    "git diff --check",
]

BOUNDARY_TOKENS = [
    "runner-local chat generation PoC only",
    "no product API",
    "no SDK surface",
    "no service API change",
    "no deployment harness change",
    "no release docs change",
    "no public crate export change",
    "no root LICENSE change",
    "no full English LM training",
    "no production chat",
    "no ChatGPT-like assistant readiness",
    "no language grounding",
    "no safety alignment",
    "no public beta",
    "no GA",
    "no hosted SaaS",
    "boundary_refusal_accuracy is a controlled mini-eval only",
    "no production safety claim",
    "no clinical/high-stakes readiness",
]

REQUIRED_TERMS = [
    "STABLE_LOOP_PHASE_LOCK_076_CHAT_GENERATION_POC",
    "phase_lane_chat_generation_poc.rs",
    "runner-local decoder",
    "fixed vocabulary from controlled chat SFT examples",
    "simple token-level decoder/generation loop",
    "deterministic greedy decode",
    "seeded sampling allowed only as an audit mode",
    "default max response length 64 tokens",
    "stop token support",
    "upstream_checkpoint_hash_before",
    "upstream_checkpoint_hash_after",
    "upstream_checkpoint_unchanged = true",
    "train_step_count > 0",
    "checkpoint_before_hash",
    "checkpoint_after_hash",
    "checkpoint_after_hash != checkpoint_before_hash",
    "checkpoint_save_load_pass = true",
    "resume_from_checkpoint_pass = true",
    "eval_after_reload_matches_before = true",
    "prediction_oracle_used = false",
    "response_uses_decoder_loop = true",
    "SIMPLE_INSTRUCTION_CHAT",
    "SHORT_ANSWER_EXPLANATION",
    "CONTEXT_CARRY_CHAT",
    "ANCHORROUTE_RETENTION",
    "BOUNDARY_REFUSAL",
    "queue.json",
    "progress.jsonl",
    "training_config.json",
    "upstream_manifest.json",
    "chat_sft_dataset_manifest.json",
    "train_examples_sample.jsonl",
    "eval_examples_sample.jsonl",
    "training_metrics.jsonl",
    "checkpoint_manifest.json",
    "checkpoint_hashes.json",
    "generation_samples.jsonl",
    "human_readable_samples.jsonl",
    "chat_eval_metrics.json",
    "finite_label_retention_metrics.json",
    "collapse_metrics.json",
    "summary.json",
    "report.md",
    "prompt",
    "expected_behavior",
    "required_keywords",
    "forbidden_outputs",
    "model_output",
    "pass_fail",
    "diagnosis",
    "eval_family",
    "output_classification",
    "short_diagnosis",
    "chat_generation_supported = true",
    "free_form_answering_supported = true",
    "multi_token_response_rate >= 0.80",
    "label_only_response_rate <= 0.20",
    "generated_token_count_mean > 3.0",
    "generated_token_count_min >= 2",
    "non_empty_response_rate >= 0.95",
    "instruction_following_accuracy >= 0.65",
    "context_carry_chat_accuracy >= 0.60",
    "boundary_refusal_accuracy >= 0.70",
    "finite_label_retention_accuracy >= 0.90",
    "empty_output_rate <= 0.02",
    "space_output_rate <= 0.02",
    "top_response_rate <= 0.35",
    "static_response_rate <= 0.20",
    "repetition_rate <= 0.20",
    "train_eval_exact_prompt_overlap_count = 0",
    "train_eval_exact_response_overlap_count",
    "train_eval_template_overlap_count",
    "eval_prompt_hash",
    "train_prompt_hash",
    "active scenario binding",
    "distractor scenario rejection",
    "inactive/stale pocket suppression",
    "answer-only scenario binding",
    "empty_output_rate",
    "space_output_rate",
    "top_response_rate",
    "static_response_rate",
    "repetition_rate",
    "copy_prompt_rate",
    "unique_response_count",
    "CHAT_GENERATION_POC_POSITIVE",
    "RUNNER_LOCAL_DECODER_LOOP_CREATED",
    "RUNNER_LOCAL_DECODER_SURFACE_CONFIRMED",
    "CONTROLLED_CHAT_SFT_COMPLETED",
    "MULTI_TOKEN_CHAT_OUTPUT_PRODUCED",
    "LABEL_ONLY_CHAT_REJECTED",
    "INSTRUCTION_FOLLOWING_CHAT_BASELINE_PASSES",
    "CONTEXT_CARRY_CHAT_BASELINE_PASSES",
    "RUBRIC_BOUNDED_CHAT_EVAL_RECORDED",
    "FINITE_LABEL_ANCHORROUTE_RETENTION_PASSES",
    "STATIC_RESPONSE_COLLAPSE_REJECTED",
    "TRAIN_EVAL_LEAKAGE_REJECTED",
    "UPSTREAM_CHECKPOINT_UNCHANGED",
    "CHECKPOINT_PIPELINE_PASSES",
    "PRODUCTION_TRAINING_NOT_CLAIMED",
    "CHAT_GENERATION_POC_FAILS",
    "CHAT_GENERATION_SURFACE_STILL_UNSUPPORTED",
    "NO_ACTUAL_TRAINING_UPDATE_DETECTED",
    "UPSTREAM_CHECKPOINT_MUTATION_DETECTED",
    "ORACLE_SHORTCUT_DETECTED",
    "TRAIN_EVAL_LEAKAGE_DETECTED",
    "LABEL_ONLY_RESPONSE_COLLAPSE_DETECTED",
    "STATIC_RESPONSE_COLLAPSE_DETECTED",
    "EMPTY_OUTPUT_COLLAPSE_DETECTED",
    "INSTRUCTION_FOLLOWING_CHAT_FAILS",
    "CONTEXT_CARRY_CHAT_FAILS",
    "FINITE_LABEL_RETENTION_REGRESSION_DETECTED",
    "CHAT_EVAL_RUBRIC_MISSING",
    "CHECKPOINT_RELOAD_FAILS",
    "RESUME_FROM_CHECKPOINT_FAILS",
    "EVAL_AFTER_RELOAD_MISMATCH",
    "RUNTIME_SURFACE_MUTATION_DETECTED",
    "ROOT_LICENSE_CHANGED",
]

PLACEHOLDERS = ["TODO", "TBD", "placeholder", "pending measured", "coming soon"]

FORBIDDEN_CLAIMS = {
    "OPEN_ENDED_CHAT_READY_CLAIM_DETECTED": ["open-ended assistant", "ChatGPT-like assistant readiness"],
    "FREE_FORM_GENERATION_FALSE_CLAIM": ["full free-form generation", "production free-form generation"],
    "PERPLEXITY_CLAIM_DETECTED": ["perplexity benchmark", "perplexity support"],
    "FULL_ENGLISH_LM_CLAIM_DETECTED": ["full English LM"],
    "LANGUAGE_GROUNDING_CLAIM_DETECTED": ["language grounding"],
    "SAFETY_ALIGNMENT_CLAIM_DETECTED": ["safety alignment"],
    "PRODUCTION_TRAINING_CLAIM_DETECTED": ["production training", "production chat"],
    "GA_CLAIM_DETECTED": ["GA release", "GA launched", "generally available"],
    "PUBLIC_BETA_CLAIM_DETECTED": ["public beta"],
    "HOSTED_SAAS_CLAIM_DETECTED": ["hosted SaaS"],
    "SERVICE_API_CLAIM_DETECTED": ["service API change"],
    "DEPLOYMENT_HARNESS_CLAIM_DETECTED": ["deployment harness change"],
    "ROOT_LICENSE_CHANGED": ["root LICENSE changed"],
}

NEGATION_MARKERS = ["not ", "no ", "does not ", "do not ", "without ", "unsupported", "false", "reject", "rejected", "controlled"]
GENERATED_PATH_PARTS = ["target/", "node_modules/", ".svelte-kit/"]
GENERATED_SUFFIXES = [".exe", ".dll", ".so", ".dylib", ".zip", ".tar", ".tar.gz", ".tgz", ".7z", ".ckpt", ".bin"]
GENERATED_NAME_PARTS = ["checkpoint", "smoke/", "service job artifacts", "release artifacts"]

POSITIVE_VERDICTS = [
    "CHAT_GENERATION_POC_STATIC_CHECK_POSITIVE",
    "CHAT_GENERATION_POC_PACKAGE_WRITTEN",
    "RUNNER_LOCAL_DECODER_GUARD_WRITTEN",
    "UPSTREAM_CHECKPOINT_READ_ONLY_GUARD_WRITTEN",
    "RUBRIC_BOUNDED_CHAT_EVAL_REQUIRED",
    "FINITE_LABEL_RETENTION_REQUIRED",
    "RUNTIME_SURFACE_UNCHANGED",
    "ROOT_LICENSE_UNCHANGED",
    "PRODUCTION_TRAINING_NOT_CLAIMED",
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

    verdicts = POSITIVE_VERDICTS.copy() if check_pass else ["CHAT_GENERATION_POC_STATIC_CHECK_FAILS"]
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
