#!/usr/bin/env python3
"""Static checker for STABLE_LOOP_PHASE_LOCK_071 repaired checkpoint confirm."""

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
    "docs/research/STABLE_LOOP_PHASE_LOCK_071_REPAIRED_CHECKPOINT_CAPABILITY_CONFIRM_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_071_REPAIRED_CHECKPOINT_CAPABILITY_CONFIRM_RESULT.md",
]

REQUIRED_SOURCE = [
    "instnct-core/examples/phase_lane_repaired_checkpoint_capability_confirm.rs",
    "scripts/probes/run_stable_loop_phase_lock_071_repaired_checkpoint_capability_confirm_check.py",
]

ALLOWED_MUTATIONS = set(REQUIRED_DOCS + REQUIRED_SOURCE)

EXACT_COMMANDS = [
    "cargo check -p instnct-core --example phase_lane_repaired_checkpoint_capability_confirm",
    "cargo run -p instnct-core --example phase_lane_repaired_checkpoint_capability_confirm -- --out target/pilot_wave/stable_loop_phase_lock_071_repaired_checkpoint_capability_confirm/smoke --checkpoint target/pilot_wave/stable_loop_phase_lock_070_distractor_resistant_anchorroute_training/smoke/checkpoints/finetune_068_targeted_repair/model_checkpoint.json --upstream-070-root target/pilot_wave/stable_loop_phase_lock_070_distractor_resistant_anchorroute_training/smoke --benchmark-069-root target/pilot_wave/stable_loop_phase_lock_069_model_capability_benchmark_gate/smoke --seed 2027 --heartbeat-sec 20",
    "python -m py_compile scripts/probes/run_stable_loop_phase_lock_071_repaired_checkpoint_capability_confirm_check.py",
    "python scripts/probes/run_stable_loop_phase_lock_071_repaired_checkpoint_capability_confirm_check.py --check-only",
    "python scripts/probes/run_stable_loop_phase_lock_070_distractor_resistant_anchorroute_check.py --check-only",
    "python scripts/probes/run_stable_loop_phase_lock_069_model_capability_benchmark_check.py --check-only",
    "cargo test -p instnct-core sdk_candidate",
    "git diff --check",
]

BOUNDARY_TOKENS = [
    "eval-only",
    "no training",
    "no checkpoint repair",
    "no open-ended assistant",
    "no free-form generation",
    "no perplexity",
    "no full English LM",
    "no language grounding",
    "no production training",
    "no GA",
    "no public beta",
    "no hosted SaaS",
    "no service API change",
    "no deployment harness change",
    "no release docs change",
    "no public crate export change",
    "no root LICENSE change",
]

REQUIRED_TERMS = [
    "UPSTREAM_070_ARTIFACT_MISSING",
    "CHECKPOINT_MUTATION_DETECTED",
    "TRAINING_SIDE_EFFECT_DETECTED",
    "ORACLE_SHORTCUT_DETECTED",
    "BENCHMARK_LEAKAGE_DETECTED",
    "BASELINE_EVAL_MISMATCH",
    "NO_ROUTE_CONTROL_MISSING",
    "FRESH_HARD_DISTRACTOR_GENERALIZATION_FAILS",
    "FRESH_COUNTERFACTUAL_GENERALIZATION_FAILS",
    "FRESH_LONG_CONTEXT_GENERALIZATION_FAILS",
    "RETENTION_CONFIRM_FAILS",
    "CAPABILITY_FAMILY_GATE_FAILS",
    "STATIC_OUTPUT_COLLAPSE_DETECTED",
    "HUMAN_SAMPLE_REPORT_MISSING",
    "OPEN_ENDED_CLAIM_DETECTED",
    "PERPLEXITY_CLAIM_DETECTED",
    "RUNTIME_SURFACE_MUTATION_DETECTED",
    "ROOT_LICENSE_CHANGED",
    "FRESH_CONTEXT_ENTITY_EXTRACTION",
    "FRESH_COUNTERFACTUAL_BINDING",
    "FRESH_DISTRACTOR_RESISTANCE",
    "FRESH_LONG_CONTEXT_NEEDLE_BINDING",
    "FRESH_NEAR_MISS_ANCHOR_SELECTION",
    "FRESH_IRRELEVANT_POCKET_SUPPRESSION",
    "FRESH_NEGATIVE_ROUTE_REJECTION",
    "RETENTION_INSTRUCTION_FOLLOWING_CLOSED",
    "RETENTION_MULTI_HOP_KEY_VALUE_BINDING",
    "RETENTION_SYMBOLIC_RULE_CLOSED_CHOICE",
    "RETENTION_NON_ROUTE_TEXT_CONTROL",
    "OPEN_ENDED_INTERFACE_LIMITATION",
    "train_step_count = 0",
    "checkpoint_hash_before",
    "checkpoint_hash_after",
    "checkpoint_hash_unchanged = true",
    "prediction_oracle_used = false",
    "open_ended_generation_supported = false",
    "free_form_answering_supported = false",
    "perplexity_supported = false",
    "finite_label_surface = true",
    "closed-label success does not imply language grounding",
    "this is not an open-ended assistant",
    "overlap_with_069_samples_count",
    "overlap_with_070_samples_count",
    "upstream_exact_overlap_audit_limited",
    "eval_row_hash_model",
    "eval_row_hash_baselines",
    "eval_row_hash_no_route_control",
    "baseline_eval_mismatch = false",
    "delta_vs_no_route_control",
    "queue.json",
    "progress.jsonl",
    "benchmark_config.json",
    "upstream_070_manifest.json",
    "checkpoint_manifest.json",
    "capability_dataset_manifest.json",
    "benchmark_examples_sample.jsonl",
    "baseline_metrics.json",
    "no_route_feature_control_metrics.json",
    "capability_metrics.json",
    "per_family_metrics.json",
    "limitation_report.json",
    "human_readable_samples.jsonl",
    "failure_case_samples.jsonl",
    "collapse_metrics.json",
    "summary.json",
    "report.md",
    "REPAIRED_CHECKPOINT_CAPABILITY_CONFIRM_POSITIVE",
    "REPAIRED_CHECKPOINT_CAPABILITY_CONFIRM_FAILS",
    "UPSTREAM_070_CHECKPOINT_VERIFIED",
    "NO_TRAINING_PERFORMED",
    "CHECKPOINT_UNCHANGED",
    "NO_ROUTE_CONTROL_RECORDED",
    "BASELINE_COMPARISON_RECORDED",
    "HUMAN_READABLE_SAMPLES_WRITTEN",
    "OPEN_ENDED_LIMITATION_RECORDED",
    "PRODUCTION_TRAINING_NOT_CLAIMED",
]

PLACEHOLDERS = ["TODO", "TBD", "placeholder", "pending measured", "coming soon"]

FORBIDDEN_CLAIMS = {
    "OPEN_ENDED_CLAIM_DETECTED": ["open-ended assistant", "free-form generation", "free form generation"],
    "PERPLEXITY_CLAIM_DETECTED": ["perplexity support", "perplexity benchmark", "perplexity LM"],
    "FULL_ENGLISH_MODEL_CLAIM_DETECTED": ["full English LM", "full English model"],
    "LANGUAGE_GROUNDING_CLAIM_DETECTED": ["language grounding"],
    "PRODUCTION_TRAINING_CLAIM_DETECTED": ["production training", "production model"],
    "GA_CLAIM_DETECTED": ["GA release", "GA launched", "generally available"],
    "PUBLIC_BETA_CLAIM_DETECTED": ["public beta"],
    "HOSTED_SAAS_CLAIM_DETECTED": ["hosted SaaS"],
}

NEGATION_MARKERS = ["not ", "no ", "does not ", "do not ", "without ", "unsupported", "false", "reject", "rejected"]
GENERATED_PATH_PARTS = ["target/", "node_modules/", ".svelte-kit/"]
GENERATED_SUFFIXES = [".exe", ".dll", ".so", ".dylib", ".zip", ".tar", ".tar.gz", ".tgz", ".7z", ".ckpt", ".bin"]
GENERATED_NAME_PARTS = ["checkpoint", "smoke/", "service job artifacts", "release artifacts"]

POSITIVE_VERDICTS = [
    "REPAIRED_CHECKPOINT_CAPABILITY_CONFIRM_STATIC_CHECK_POSITIVE",
    "REPAIRED_CHECKPOINT_CAPABILITY_CONFIRM_PACKAGE_WRITTEN",
    "EVAL_ONLY_REPAIRED_CHECKPOINT_GATE_WRITTEN",
    "NO_ROUTE_CONTROL_REQUIRED",
    "FRESH_BENCHMARK_LEAKAGE_GUARD_WRITTEN",
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

    verdicts = POSITIVE_VERDICTS.copy() if check_pass else ["REPAIRED_CHECKPOINT_CAPABILITY_CONFIRM_STATIC_CHECK_FAILS"]
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
