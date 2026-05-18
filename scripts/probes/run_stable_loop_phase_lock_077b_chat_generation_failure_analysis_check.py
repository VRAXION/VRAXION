#!/usr/bin/env python3
"""Static checker for STABLE_LOOP_PHASE_LOCK_077B chat generation failure analysis."""

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
    "docs/research/STABLE_LOOP_PHASE_LOCK_077B_CHAT_GENERATION_FAILURE_ANALYSIS_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_077B_CHAT_GENERATION_FAILURE_ANALYSIS_RESULT.md",
]

REQUIRED_SOURCE = [
    "scripts/probes/run_stable_loop_phase_lock_077b_chat_generation_failure_analysis.py",
    "scripts/probes/run_stable_loop_phase_lock_077b_chat_generation_failure_analysis_check.py",
]

ALLOWED_MUTATIONS = set(REQUIRED_DOCS + REQUIRED_SOURCE)

EXACT_COMMANDS = [
    "python -m py_compile scripts/probes/run_stable_loop_phase_lock_077b_chat_generation_failure_analysis.py",
    "python scripts/probes/run_stable_loop_phase_lock_077b_chat_generation_failure_analysis.py --out target/pilot_wave/stable_loop_phase_lock_077b_chat_generation_failure_analysis/smoke --upstream-077-root target/pilot_wave/stable_loop_phase_lock_077_chat_generation_fresh_composition_confirm/smoke --upstream-076-root target/pilot_wave/stable_loop_phase_lock_076_chat_generation_poc/smoke --heartbeat-sec 20",
    "python -m py_compile scripts/probes/run_stable_loop_phase_lock_077b_chat_generation_failure_analysis_check.py",
    "python scripts/probes/run_stable_loop_phase_lock_077b_chat_generation_failure_analysis_check.py --check-only",
    "python scripts/probes/run_stable_loop_phase_lock_077_chat_generation_fresh_composition_confirm_check.py --check-only",
    "python scripts/probes/run_stable_loop_phase_lock_076_chat_generation_poc_check.py --check-only",
    "git diff --check",
]

BOUNDARY_TOKENS = [
    "analysis-only",
    "no training",
    "no new inference",
    "no 076 rerun",
    "no 077 rerun",
    "no checkpoint mutation",
    "no replacement checkpoint",
    "no checkpoint repair",
    "no model capability improved",
    "no production chat",
    "no GPT-like assistant readiness",
    "no service API change",
    "no deployment harness change",
    "no SDK/public export change",
    "no release docs change",
    "no root LICENSE change",
]

REQUIRED_TERMS = [
    "STABLE_LOOP_PHASE_LOCK_077B_CHAT_GENERATION_FAILURE_ANALYSIS",
    "run_stable_loop_phase_lock_077b_chat_generation_failure_analysis.py",
    "UPSTREAM_077_ARTIFACT_MISSING",
    "UPSTREAM_076_ARTIFACT_MISSING",
    "summary.json",
    "generation_samples.jsonl",
    "human_readable_samples.jsonl",
    "composition_metrics.json",
    "novelty_metrics.json",
    "collapse_metrics.json",
    "finite_label_retention_metrics.json",
    "chat_sft_dataset_manifest.json",
    "checkpoint_manifest.json",
    "model_checkpoint.json",
    "exact_response_table_copy",
    "exact_train_response_copy",
    "exact_eval_response_copy",
    "semantic_template_copy",
    "finite_label_retention_label",
    "context_slot_not_bound",
    "boundary_refusal_not_selected",
    "wrong_template_family_selected",
    "prompt_copy",
    "unknown_source",
    "unknown_source_rate <= 0.10",
    "template_copy_source_coverage >= 0.90",
    "human_failure_digest rows > 0",
    "queue.json",
    "progress.jsonl",
    "analysis_config.json",
    "upstream_077_manifest.json",
    "upstream_076_manifest.json",
    "template_copy_source_report.json",
    "fresh_context_carry_failure_report.json",
    "boundary_refusal_failure_report.json",
    "response_table_dependence_report.json",
    "prompt_to_template_mapping.jsonl",
    "failure_cluster_report.json",
    "repair_recommendation.json",
    "human_failure_digest.jsonl",
    "context_carry_failure_count",
    "context_slot_expected",
    "context_slot_model_output",
    "selected_template_label_or_response",
    "slot_binding_miss_rate",
    "wrong_template_family_rate",
    "boundary_failure_count",
    "expected_refusal_keywords",
    "boundary_template_selection_rate",
    "exact_train_response_copy_rate",
    "exact_eval_response_copy_rate",
    "response_table_copy_rate",
    "template_copy_rate",
    "novel_response_rate",
    "train_response_ngram_overlap",
    "top_copied_templates",
    "copy_rate_by_eval_family",
    "eval_family",
    "prompt",
    "model_output",
    "expected_behavior",
    "classified_source",
    "copied_template_if_any",
    "required_keywords",
    "missing_keywords",
    "short_diagnosis",
    "next_milestone = 078_CHAT_COMPOSITION_REPAIR",
    "use token-level next-token objective",
    "reduce response_table dependence",
    "add paraphrase / many-target variants",
    "add fresh composition curriculum",
    "add context carry variable-slot training",
    "add boundary refusal paraphrase variants",
    "add template dropout",
    "add semantic slot recombination",
    "retain finite-label AnchorRoute scenario-state eval",
    "keep no product API / no SDK / no service surface",
    "do not claim GPT-like assistant readiness",
    "adding more table responses alone is not enough",
    "CHAT_GENERATION_FAILURE_ANALYSIS_POSITIVE",
    "UPSTREAM_077_FAILURE_PROFILE_LOADED",
    "TEMPLATE_COPY_SOURCE_ATTRIBUTION_WRITTEN",
    "FRESH_CONTEXT_CARRY_FAILURE_ANALYZED",
    "BOUNDARY_REFUSAL_FAILURE_ANALYZED",
    "RESPONSE_TABLE_DEPENDENCE_CONFIRMED",
    "REPAIR_RECOMMENDATION_WRITTEN",
    "NO_TRAINING_PERFORMED",
    "PRODUCTION_CHAT_NOT_CLAIMED",
    "CHAT_GENERATION_FAILURE_ANALYSIS_FAILS",
    "FAILURE_CASE_INPUT_MISSING",
    "TEMPLATE_COPY_ANALYSIS_INCOMPLETE",
    "CONTEXT_CARRY_ANALYSIS_INCOMPLETE",
    "BOUNDARY_REFUSAL_ANALYSIS_INCOMPLETE",
    "UNKNOWN_SOURCE_RATE_TOO_HIGH",
    "REPAIR_RECOMMENDATION_MISSING",
    "TRAINING_SIDE_EFFECT_DETECTED",
    "CHECKPOINT_MUTATION_DETECTED",
    "PRODUCTION_CHAT_CLAIM_DETECTED",
    "GPT_LIKE_READINESS_FALSE_CLAIM",
    "RUNTIME_SURFACE_MUTATION_DETECTED",
    "ROOT_LICENSE_CHANGED",
]

PLACEHOLDERS = ["TODO", "TBD", "placeholder", "pending measured", "coming soon"]

FORBIDDEN_CLAIMS = {
    "TRAINING_SIDE_EFFECT_DETECTED": ["training performed", "trained model", "model trained"],
    "CHECKPOINT_MUTATION_DETECTED": ["checkpoint mutated", "checkpoint repaired", "checkpoint updated"],
    "CAPABILITY_IMPROVEMENT_CLAIM_DETECTED": ["model improved", "capability improved"],
    "PRODUCTION_CHAT_CLAIM_DETECTED": ["production chat"],
    "GPT_LIKE_READINESS_FALSE_CLAIM": ["GPT-like assistant readiness"],
    "FULL_ENGLISH_LM_CLAIM_DETECTED": ["full English LM"],
    "LANGUAGE_GROUNDING_CLAIM_DETECTED": ["language grounding"],
    "GA_CLAIM_DETECTED": ["GA release", "GA launched", "generally available"],
    "PUBLIC_BETA_CLAIM_DETECTED": ["public beta"],
    "HOSTED_SAAS_CLAIM_DETECTED": ["hosted SaaS"],
    "SERVICE_API_CLAIM_DETECTED": ["service API change"],
    "DEPLOYMENT_HARNESS_CLAIM_DETECTED": ["deployment harness change"],
    "SDK_EXPORT_CLAIM_DETECTED": ["SDK/public export change"],
    "ROOT_LICENSE_CHANGED": ["root LICENSE changed"],
}

NEGATION_MARKERS = ["not ", "no ", "does not ", "do not ", "without ", "unsupported", "false", "reject", "rejected", "analysis-only"]
GENERATED_PATH_PARTS = ["target/", "node_modules/", ".svelte-kit/"]
GENERATED_SUFFIXES = [".exe", ".dll", ".so", ".dylib", ".zip", ".tar", ".tar.gz", ".tgz", ".7z", ".ckpt", ".bin"]
GENERATED_NAME_PARTS = ["checkpoint", "smoke/", "service job artifacts", "release artifacts"]

POSITIVE_VERDICTS = [
    "CHAT_GENERATION_FAILURE_ANALYSIS_STATIC_CHECK_POSITIVE",
    "CHAT_GENERATION_FAILURE_ANALYSIS_PACKAGE_WRITTEN",
    "ANALYSIS_ONLY_HARD_WALL_WRITTEN",
    "TEMPLATE_COPY_SOURCE_ATTRIBUTION_REQUIRED",
    "CONTEXT_AND_BOUNDARY_FAILURE_ANALYSIS_REQUIRED",
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

    verdicts = POSITIVE_VERDICTS.copy() if check_pass else ["CHAT_GENERATION_FAILURE_ANALYSIS_STATIC_CHECK_FAILS"]
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
