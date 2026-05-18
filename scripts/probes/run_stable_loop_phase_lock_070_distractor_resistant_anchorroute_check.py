#!/usr/bin/env python3
"""Static checker for STABLE_LOOP_PHASE_LOCK_070 distractor-resistant training."""

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
    "docs/research/STABLE_LOOP_PHASE_LOCK_070_DISTRACTOR_RESISTANT_ANCHORROUTE_TRAINING_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_070_DISTRACTOR_RESISTANT_ANCHORROUTE_TRAINING_RESULT.md",
]

REQUIRED_SOURCE = [
    "instnct-core/examples/phase_lane_distractor_resistant_anchorroute_training.rs",
    "scripts/probes/run_stable_loop_phase_lock_070_distractor_resistant_anchorroute_check.py",
]

ALLOWED_MUTATIONS = set(REQUIRED_DOCS + REQUIRED_SOURCE)

EXACT_COMMANDS = [
    "cargo check -p instnct-core --example phase_lane_distractor_resistant_anchorroute_training",
    "cargo run -p instnct-core --example phase_lane_distractor_resistant_anchorroute_training -- --out target/pilot_wave/stable_loop_phase_lock_070_distractor_resistant_anchorroute_training/smoke --upstream-checkpoint target/pilot_wave/stable_loop_phase_lock_068_real_text_anchorcell_confirm_scale/confirm/seed_2028/checkpoints/MIXED_WITH_ROUTE_GRAMMAR_ON/model_checkpoint.json --upstream-summary target/pilot_wave/stable_loop_phase_lock_068_real_text_anchorcell_confirm_scale/confirm/summary.json --benchmark-069-root target/pilot_wave/stable_loop_phase_lock_069_model_capability_benchmark_gate/smoke --targeted-examples 120000 --seed 2026 --heartbeat-sec 20",
    "python -m py_compile scripts/probes/run_stable_loop_phase_lock_070_distractor_resistant_anchorroute_check.py",
    "python scripts/probes/run_stable_loop_phase_lock_070_distractor_resistant_anchorroute_check.py --check-only",
    "python scripts/probes/run_stable_loop_phase_lock_069_model_capability_benchmark_check.py --check-only",
    "python scripts/probes/run_stable_loop_phase_lock_068_real_text_anchorcell_confirm_scale_check.py --check-only",
    "cargo test -p instnct-core sdk_candidate",
    "git diff --check",
]

BOUNDARY_TOKENS = [
    "finite-label repair training only",
    "no open-ended assistant",
    "no free-form generation",
    "no perplexity",
    "no full English LM",
    "no language grounding",
    "no production training",
    "no GA",
    "no public beta",
    "no hosted SaaS",
    "no clinical use",
    "no high-stakes education use",
    "no service API change",
    "no deployment harness change",
    "no public crate export change",
    "no root LICENSE change",
]

REQUIRED_TERMS = [
    "FRESH_TARGETED_MIX_TRAINING",
    "FINETUNE_068_TARGETED_REPAIR",
    "NO_TRAIN_069_BASELINE",
    "FRESH_SHUFFLED_LABEL_CONTROL",
    "FINETUNE_SHUFFLED_LABEL_CONTROL",
    "NO_ROUTE_FEATURE_CONTROL",
    "CHECKPOINT_RELOAD_EVAL",
    "ROLLBACK_REHEARSAL",
    "RESUME_FROM_CHECKPOINT",
    "HARD_DISTRACTOR_ANCHOR_BINDING",
    "NEAR_MISS_ANCHOR_SELECTION",
    "SAME_KEY_DIFFERENT_CONTEXT",
    "LONG_CONTEXT_NEEDLE_RETRIEVAL",
    "IRRELEVANT_POCKET_SUPPRESSION",
    "NEGATIVE_ROUTE_REJECTION",
    "ANSWER_ONLY_HARD_BINDING",
    "TRACE_MIXED_HARD_BINDING",
    "RETENTION_FINEWEB_CONTINUATION",
    "RETENTION_NON_ROUTE_CONTROL",
    "CHECKPOINT_MUTATION_DETECTED",
    "NO_ACTUAL_TRAINING_UPDATE_DETECTED",
    "TRAIN_BENCHMARK_LEAKAGE_DETECTED",
    "RETENTION_REGRESSION_DETECTED",
    "ARM_COMPARISON_MISSING",
    "OPEN_ENDED_CLAIM_DETECTED",
    "FAILURE_CASE_REPORT_MISSING",
    "RUNTIME_SURFACE_MUTATION_DETECTED",
    "ROOT_LICENSE_CHANGED",
    "overlap_with_069_eval_count",
    "best_arm",
    "fresh_pass",
    "finetune_pass",
    "fresh_vs_finetune_delta",
    "recommended_next_strategy",
    "warm_start_hash",
    "initialized_hash",
    "final_hash",
    "prediction_oracle_used = false",
    "queue.json",
    "progress.jsonl",
    "training_config.json",
    "upstream_068_manifest.json",
    "baseline_069_reference.json",
    "targeted_dataset_manifest.json",
    "train_examples_sample.jsonl",
    "heldout_examples_sample.jsonl",
    "ood_examples_sample.jsonl",
    "training_metrics.jsonl",
    "checkpoint_manifest.json",
    "checkpoint_hashes.json",
    "post_training_capability_metrics.json",
    "per_family_metrics.json",
    "retention_metrics.json",
    "regression_report.json",
    "baseline_knockout_report.json",
    "arm_comparison.json",
    "human_readable_samples.jsonl",
    "failure_case_samples.jsonl",
    "collapse_metrics.json",
    "summary.json",
    "report.md",
    "DISTRACTOR_RESISTANT_ANCHORROUTE_TRAINING_POSITIVE",
    "DISTRACTOR_RESISTANT_ANCHORROUTE_TRAINING_FAILS",
    "UPSTREAM_068_CHECKPOINT_UNCHANGED",
    "TRAIN_BENCHMARK_LEAKAGE_REJECTED",
    "RETENTION_GATE_PASSES",
    "ARM_COMPARISON_WRITTEN",
    "BEST_ARM_SELECTED",
    "PRODUCTION_TRAINING_NOT_CLAIMED",
]

PLACEHOLDERS = ["TODO", "TBD", "placeholder", "pending measured", "coming soon"]

FORBIDDEN_CLAIMS = {
    "OPEN_ENDED_CLAIM_DETECTED": ["open-ended assistant", "free-form generation", "free form generation"],
    "PERPLEXITY_CLAIM_DETECTED": ["perplexity support", "perplexity LM"],
    "FULL_ENGLISH_MODEL_CLAIM_DETECTED": ["full English LM", "full English model"],
    "LANGUAGE_GROUNDING_CLAIM_DETECTED": ["language grounding"],
    "PRODUCTION_TRAINING_CLAIM_DETECTED": ["production training", "production-scale training"],
    "GA_CLAIM_DETECTED": ["GA release", "GA launched", "generally available"],
    "PUBLIC_BETA_CLAIM_DETECTED": ["public beta"],
    "HOSTED_SAAS_CLAIM_DETECTED": ["hosted SaaS"],
    "CLINICAL_READY_CLAIM_DETECTED": ["clinical readiness", "clinical ready"],
    "HIGH_STAKES_EDUCATION_READY_CLAIM_DETECTED": ["high-stakes education readiness", "high-stakes education ready"],
}

NEGATION_MARKERS = ["not ", "no ", "does not ", "do not ", "without ", "unsupported", "false", "reject", "rejected"]
GENERATED_PATH_PARTS = ["target/", "node_modules/", ".svelte-kit/"]
GENERATED_SUFFIXES = [".exe", ".dll", ".so", ".dylib", ".zip", ".tar", ".tar.gz", ".tgz", ".7z", ".ckpt", ".bin"]
GENERATED_NAME_PARTS = ["checkpoint", "smoke/", "service job artifacts", "release artifacts"]
MUTATION_PATHS = ["LICENSE", "instnct-core", "tools/instnct_service_alpha", "tools/instnct_deploy", "docs/releases"]

POSITIVE_VERDICTS = [
    "DISTRACTOR_RESISTANT_ANCHORROUTE_STATIC_CHECK_POSITIVE",
    "DISTRACTOR_RESISTANT_ANCHORROUTE_PACKAGE_WRITTEN",
    "FRESH_AND_FINETUNE_ARMS_REQUIRED",
    "RETENTION_GATE_REQUIRED",
    "ARM_COMPARISON_REQUIRED",
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
        if path.startswith("instnct-core/") and path not in ALLOWED_MUTATIONS:
            return True
        if path.startswith("tools/instnct_service_alpha/") or path.startswith("tools/instnct_deploy/"):
            return True
        if path.startswith("docs/releases/"):
            return True
    return False


def generated_artifact_staged() -> bool:
    for path in changed_paths():
        if any(part in path for part in GENERATED_PATH_PARTS):
            return True
        if any(path.endswith(suffix) for suffix in GENERATED_SUFFIXES):
            return True
        if any(part in path for part in GENERATED_NAME_PARTS) and path.startswith("target/"):
            return True
    return False


def placeholder_hits(files: dict[str, str]) -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    for rel, text in files.items():
        if rel.endswith(".py") or rel.endswith(".rs"):
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
        if rel.endswith(".py") or rel.endswith(".rs"):
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

    verdicts = POSITIVE_VERDICTS.copy() if check_pass else ["DISTRACTOR_RESISTANT_ANCHORROUTE_STATIC_CHECK_FAILS"]
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
        "missing_source": [rel for rel in REQUIRED_SOURCE if rel not in files],
        "placeholder_hits": placeholders,
        "missing_commands": commands,
        "missing_doc_references": [],
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
