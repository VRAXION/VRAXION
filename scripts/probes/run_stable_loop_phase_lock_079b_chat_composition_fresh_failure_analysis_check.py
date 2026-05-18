#!/usr/bin/env python3
"""Static checker for STABLE_LOOP_PHASE_LOCK_079B chat composition failure analysis."""

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
    "docs/research/STABLE_LOOP_PHASE_LOCK_079B_CHAT_COMPOSITION_FRESH_FAILURE_ANALYSIS_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_079B_CHAT_COMPOSITION_FRESH_FAILURE_ANALYSIS_RESULT.md",
]

REQUIRED_SOURCE = [
    "scripts/probes/run_stable_loop_phase_lock_079b_chat_composition_fresh_failure_analysis.py",
    "scripts/probes/run_stable_loop_phase_lock_079b_chat_composition_fresh_failure_analysis_check.py",
]

ALLOWED_MUTATIONS = set(REQUIRED_DOCS + REQUIRED_SOURCE)

EXACT_COMMANDS = [
    "python -m py_compile scripts/probes/run_stable_loop_phase_lock_079b_chat_composition_fresh_failure_analysis.py",
    "python scripts/probes/run_stable_loop_phase_lock_079b_chat_composition_fresh_failure_analysis.py --out target/pilot_wave/stable_loop_phase_lock_079b_chat_composition_fresh_failure_analysis/smoke --upstream-079-root target/pilot_wave/stable_loop_phase_lock_079_chat_composition_fresh_confirm/smoke --upstream-078-root target/pilot_wave/stable_loop_phase_lock_078_chat_composition_repair/smoke --upstream-077b-root target/pilot_wave/stable_loop_phase_lock_077b_chat_generation_failure_analysis/smoke --heartbeat-sec 20",
    "python -m py_compile scripts/probes/run_stable_loop_phase_lock_079b_chat_composition_fresh_failure_analysis_check.py",
    "python scripts/probes/run_stable_loop_phase_lock_079b_chat_composition_fresh_failure_analysis_check.py --check-only",
    "python scripts/probes/run_stable_loop_phase_lock_079_chat_composition_fresh_confirm_check.py --check-only",
    "python scripts/probes/run_stable_loop_phase_lock_078_chat_composition_repair_check.py --check-only",
    "git diff --check",
]

BOUNDARY_TOKENS = [
    "analysis-only",
    "no training",
    "no new inference",
    "no 078 rerun",
    "no 079 rerun",
    "no checkpoint mutation",
    "no replacement checkpoint",
    "no product/API/SDK surface changes",
    "no service API change",
    "no deployment harness change",
    "no SDK/public export change",
    "no release docs change",
    "no root LICENSE change",
    "no GPT-like readiness claim",
    "no production chat",
]

REQUIRED_TERMS = [
    "STABLE_LOOP_PHASE_LOCK_079B_CHAT_COMPOSITION_FRESH_FAILURE_ANALYSIS",
    "run_stable_loop_phase_lock_079b_chat_composition_fresh_failure_analysis.py",
    "run_stable_loop_phase_lock_079b_chat_composition_fresh_failure_analysis_check.py",
    "UPSTREAM_079_ARTIFACT_MISSING",
    "UPSTREAM_078_ARTIFACT_MISSING",
    "UPSTREAM_077B_ARTIFACT_MISSING",
    "summary.json",
    "generation_samples.jsonl",
    "human_readable_samples.jsonl",
    "composition_metrics.json",
    "novelty_metrics.json",
    "context_slot_metrics.json",
    "finite_label_retention_metrics.json",
    "collapse_metrics.json",
    "fresh_chat_eval_dataset.jsonl",
    "train_examples_sample.jsonl",
    "eval_examples_sample.jsonl",
    "repair_dataset_manifest.json",
    "checkpoints/chat_composition_repair/model_checkpoint.json",
    "repair_recommendation.json",
    "exact_078_train_response_copy",
    "exact_078_eval_response_copy",
    "exact_078_generated_output_copy",
    "exact_076_response_table_copy",
    "semantic_078_template_overlap",
    "response_skeleton_reuse",
    "low_vocab_recombination",
    "greedy_decoder_reused_high_prior_template",
    "finite_label_retention_label",
    "genuinely_novel_response",
    "unknown_source",
    "unknown_source_rate <= 0.10",
    "template_copy_attribution.json",
    "semantic_overlap_report.json",
    "response_skeleton_report.json",
    "vocabulary_entropy_report.json",
    "decoder_prior_report.json",
    "context_carry_composition_report.json",
    "retention_non_regression_report.json",
    "row_level_attribution.jsonl",
    "human_failure_digest.jsonl",
    "queue.json",
    "progress.jsonl",
    "analysis_config.json",
    "upstream_079_manifest.json",
    "upstream_078_manifest.json",
    "upstream_077b_manifest.json",
    "exact_078_train_response_copy_rate",
    "exact_078_eval_response_copy_rate",
    "exact_078_generated_output_copy_rate",
    "exact_076_response_table_copy_rate",
    "semantic_078_template_overlap_rate",
    "response_skeleton_reuse_rate",
    "genuinely_novel_response_rate",
    "max_token_jaccard_to_078_train_response",
    "max_token_jaccard_to_078_eval_output",
    "max_token_jaccard_to_078_generated_output",
    "max_token_jaccard_to_076_response_table",
    "mean_max_template_overlap",
    "rows_above_0_80_overlap",
    "rows_above_0_90_overlap",
    "skeleton_template",
    "skeleton_count",
    "skeleton_reuse_rate",
    "top_reused_skeletons",
    "skeleton_by_eval_family",
    "generated_vocab_size",
    "train_vocab_size",
    "eval_vocab_size",
    "generated_to_train_vocab_ratio",
    "unique_response_count",
    "unique_bigram_count",
    "unique_trigram_count",
    "response_entropy",
    "token_entropy",
    "top_response_rate",
    "top_skeleton_rate",
    "high_prior_template_selection_rate",
    "greedy_decode_reuse_rate",
    "repeated_prefix_rate",
    "context_slot_binding_accuracy",
    "slot_inserted_into_template",
    "slot_only_changed_with_same_skeleton_rate",
    "context_composition_novelty_rate",
    "finite_label_retention_accuracy",
    "retention_fail_count",
    "retention_template_copy_relevance",
    "next_milestone = 080_CHAT_COMPOSITION_DIVERSITY_REPAIR",
    "reduce exact response target reuse",
    "replace one-label-one-response training with many-valid-continuation training",
    "use token-level continuation objective over multiple paraphrase targets",
    "add response skeleton dropout",
    "add lexical dropout / synonym slots",
    "add randomized clause order",
    "add fresh heldout paraphrase families",
    "add semantic slot recombination",
    "add entropy regularization or diversity penalty if available",
    "keep context slot binding objective",
    "keep finite-label AnchorRoute retention",
    "keep no product API / no SDK / no service surface",
    "keep no GPT-like readiness claim",
    "adding more response-table entries alone is not enough",
    "exact-response supervised templates are the current failure source",
    "next repair should target composition diversity and template abstraction, not only more data volume",
    "CHAT_COMPOSITION_FRESH_FAILURE_ANALYSIS_POSITIVE",
    "UPSTREAM_079_FAILURE_PROFILE_LOADED",
    "TEMPLATE_COPY_ATTRIBUTION_WRITTEN",
    "SEMANTIC_TEMPLATE_OVERLAP_ANALYZED",
    "RESPONSE_SKELETON_REUSE_ANALYZED",
    "VOCAB_ENTROPY_REPORT_WRITTEN",
    "DECODER_PRIOR_REPORT_WRITTEN",
    "CONTEXT_CARRY_COMPOSITION_ANALYZED",
    "RETENTION_NON_REGRESSION_CONFIRMED",
    "REPAIR_RECOMMENDATION_WRITTEN",
    "NO_TRAINING_PERFORMED",
    "PRODUCTION_CHAT_NOT_CLAIMED",
    "CHAT_COMPOSITION_FRESH_FAILURE_ANALYSIS_FAILS",
    "FAILURE_CASE_INPUT_MISSING",
    "TEMPLATE_COPY_ATTRIBUTION_INCOMPLETE",
    "SEMANTIC_OVERLAP_ANALYSIS_INCOMPLETE",
    "RESPONSE_SKELETON_ANALYSIS_INCOMPLETE",
    "VOCAB_ENTROPY_ANALYSIS_INCOMPLETE",
    "DECODER_PRIOR_ANALYSIS_INCOMPLETE",
    "UNKNOWN_SOURCE_RATE_TOO_HIGH",
    "REPAIR_RECOMMENDATION_MISSING",
    "TRAINING_SIDE_EFFECT_DETECTED",
    "CHECKPOINT_MUTATION_DETECTED",
    "GPT_LIKE_READINESS_FALSE_CLAIM",
    "PRODUCTION_CHAT_CLAIM_DETECTED",
    "RUNTIME_SURFACE_MUTATION_DETECTED",
    "ROOT_LICENSE_CHANGED",
]

PLACEHOLDERS = ["TODO", "TBD", "placeholder", "pending measured", "coming soon"]

FORBIDDEN_CLAIMS = {
    "TRAINING_SIDE_EFFECT_DETECTED": ["training performed", "trained model", "model trained"],
    "CHECKPOINT_MUTATION_DETECTED": ["checkpoint mutated", "checkpoint updated"],
    "CAPABILITY_IMPROVEMENT_CLAIM_DETECTED": ["model improved", "capability improved"],
    "GPT_LIKE_READINESS_FALSE_CLAIM": ["GPT-like readiness", "GPT-like assistant readiness"],
    "PRODUCTION_CHAT_CLAIM_DETECTED": ["production chat"],
    "SERVICE_API_CLAIM_DETECTED": ["service API change"],
    "DEPLOYMENT_HARNESS_CLAIM_DETECTED": ["deployment harness change"],
    "SDK_EXPORT_CLAIM_DETECTED": ["SDK/public export change"],
    "ROOT_LICENSE_CHANGED": ["root LICENSE changed"],
}

NEGATION_MARKERS = ["not ", "no ", "does not ", "do not ", "without ", "false", "reject", "rejected", "analysis-only", "keep no "]
GENERATED_PATH_PARTS = ["target/", "node_modules/", ".svelte-kit/"]
GENERATED_SUFFIXES = [".exe", ".dll", ".so", ".dylib", ".zip", ".tar", ".tar.gz", ".tgz", ".7z", ".ckpt", ".bin"]

POSITIVE_VERDICTS = [
    "CHAT_COMPOSITION_FRESH_FAILURE_ANALYSIS_STATIC_CHECK_POSITIVE",
    "CHAT_COMPOSITION_FRESH_FAILURE_ANALYSIS_PACKAGE_WRITTEN",
    "ANALYSIS_ONLY_HARD_WALL_WRITTEN",
    "TEMPLATE_COPY_ATTRIBUTION_REQUIRED",
    "SEMANTIC_AND_SKELETON_ANALYSIS_REQUIRED",
    "VOCAB_AND_DECODER_PRIOR_ANALYSIS_REQUIRED",
    "REPAIR_RECOMMENDATION_REQUIRED",
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
        if rel.endswith(".py"):
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
        if rel.endswith(".py"):
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

    check_pass = not any([
        missing_docs,
        placeholders,
        command_misses,
        boundary_misses,
        forbidden,
        required_misses,
        license_changed,
        runtime_changed,
        generated_staged,
    ])
    verdicts = POSITIVE_VERDICTS.copy() if check_pass else ["CHAT_COMPOSITION_FRESH_FAILURE_ANALYSIS_STATIC_CHECK_FAILS"]
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
