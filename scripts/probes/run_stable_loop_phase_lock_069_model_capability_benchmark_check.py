#!/usr/bin/env python3
"""Static checker for STABLE_LOOP_PHASE_LOCK_069 model capability benchmark gate."""

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
    "docs/research/STABLE_LOOP_PHASE_LOCK_069_MODEL_CAPABILITY_BENCHMARK_GATE_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_069_MODEL_CAPABILITY_BENCHMARK_GATE_RESULT.md",
]

REQUIRED_SOURCE = [
    "instnct-core/examples/phase_lane_model_capability_benchmark_gate.rs",
    "scripts/probes/run_stable_loop_phase_lock_069_model_capability_benchmark_check.py",
]

EXACT_COMMANDS = [
    "cargo check -p instnct-core --example phase_lane_model_capability_benchmark_gate",
    "cargo run -p instnct-core --example phase_lane_model_capability_benchmark_gate -- --out target/pilot_wave/stable_loop_phase_lock_069_model_capability_benchmark_gate/smoke --checkpoint target/pilot_wave/stable_loop_phase_lock_068_real_text_anchorcell_confirm_scale/confirm/seed_2028/checkpoints/MIXED_WITH_ROUTE_GRAMMAR_ON/model_checkpoint.json --upstream-summary target/pilot_wave/stable_loop_phase_lock_068_real_text_anchorcell_confirm_scale/confirm/summary.json --seed 2026 --heartbeat-sec 20",
    "python -m py_compile scripts/probes/run_stable_loop_phase_lock_069_model_capability_benchmark_check.py",
    "python scripts/probes/run_stable_loop_phase_lock_069_model_capability_benchmark_check.py --check-only",
    "python scripts/probes/run_stable_loop_phase_lock_068_real_text_anchorcell_confirm_scale_check.py --check-only",
    "python scripts/probes/run_stable_loop_phase_lock_067_real_text_anchorcell_training_poc_check.py --check-only",
    "cargo test -p instnct-core sdk_candidate",
    "git diff --check",
]

BOUNDARY_TOKENS = [
    "no retraining",
    "no production training",
    "no full English LM capability",
    "no perplexity support",
    "no free-form generation",
    "no language grounding",
    "no GA",
    "no public beta",
    "no hosted SaaS",
    "no clinical use",
    "no high-stakes education use",
    "no full VRAXION",
    "no consciousness",
    "no biological/FlyWire equivalence",
    "no physical quantum behavior",
]

REQUIRED_TERMS = [
    "UPSTREAM_068_ARTIFACT_MISSING",
    "CHECKPOINT_MUTATION_DETECTED",
    "TRAINING_SIDE_EFFECT_DETECTED",
    "ORACLE_SHORTCUT_DETECTED",
    "BASELINE_EVAL_MISMATCH",
    "CAPABILITY_FAMILY_GATE_FAILS",
    "OPEN_ENDED_CLAIM_DETECTED",
    "PERPLEXITY_CLAIM_DETECTED",
    "STATIC_OUTPUT_COLLAPSE_DETECTED",
    "HUMAN_SAMPLE_REPORT_MISSING",
    "RUNTIME_SURFACE_MUTATION_DETECTED",
    "ROOT_LICENSE_CHANGED",
    "prediction_oracle_used = false",
    "checkpoint_label_count",
    "benchmark_label_count",
    "labels_not_in_checkpoint_count",
    "unsupported_label_cases",
    "open_ended_generation_supported = false",
    "perplexity_supported = false",
    "free_form_answering_supported = false",
    "train_step_count = 0",
    "checkpoint_hash_before",
    "checkpoint_hash_after",
    "checkpoint_hash_unchanged = true",
    "eval_row_hash_model",
    "eval_row_hash_baselines",
    "baseline_eval_mismatch = false",
    "MODEL_CAPABILITY_BENCHMARK_GATE_POSITIVE",
    "MODEL_CAPABILITY_BENCHMARK_GATE_FAILS",
    "CURRENT_CHECKPOINT_CAPABILITY_PROFILE_WRITTEN",
    "UPSTREAM_068_CHECKPOINT_VERIFIED",
    "FINITE_LABEL_SURFACE_MEASURED",
    "OPEN_ENDED_LIMITATION_RECORDED",
    "HUMAN_READABLE_SAMPLES_WRITTEN",
    "NO_TRAINING_PERFORMED",
    "PRODUCTION_TRAINING_NOT_CLAIMED",
    "queue.json",
    "progress.jsonl",
    "benchmark_config.json",
    "upstream_068_manifest.json",
    "checkpoint_manifest.json",
    "capability_dataset_manifest.json",
    "benchmark_examples_sample.jsonl",
    "baseline_metrics.json",
    "capability_metrics.json",
    "per_family_metrics.json",
    "limitation_report.json",
    "human_readable_samples.jsonl",
    "failure_case_samples.jsonl",
    "collapse_metrics.json",
    "summary.json",
    "report.md",
]

TASK_FAMILIES = [
    "FINEWEB_CLOSED_CONTINUATION_SELECTION",
    "CONTEXT_ENTITY_EXTRACTION",
    "INSTRUCTION_FOLLOWING_CLOSED",
    "MULTI_HOP_KEY_VALUE_BINDING",
    "COUNTERFACTUAL_BINDING",
    "DISTRACTOR_RESISTANCE",
    "LONG_CONTEXT_NEEDLE_BINDING",
    "SYMBOLIC_RULE_CLOSED_CHOICE",
    "NON_ROUTE_TEXT_CONTROL",
    "OPEN_ENDED_INTERFACE_LIMITATION",
]

PLACEHOLDERS = ["TODO", "TBD", "placeholder", "pending measured", "coming soon"]

FORBIDDEN_CLAIMS = {
    "RETRAINING_CLAIM_DETECTED": ["rerun 067", "rerun 068", "train a replacement model"],
    "OPEN_ENDED_CLAIM_DETECTED": ["open-ended assistant", "free-form answer generation", "free form assistant"],
    "PERPLEXITY_CLAIM_DETECTED": ["perplexity benchmark", "perplexity support"],
    "FULL_ENGLISH_MODEL_CLAIM_DETECTED": ["full English LM", "full English model"],
    "LANGUAGE_GROUNDING_CLAIM_DETECTED": ["language grounding"],
    "PRODUCTION_TRAINING_CLAIM_DETECTED": ["production training", "production model readiness"],
    "GA_CLAIM_DETECTED": ["GA release", "GA launched", "generally available"],
    "PUBLIC_BETA_CLAIM_DETECTED": ["public beta"],
    "HOSTED_SAAS_CLAIM_DETECTED": ["hosted SaaS"],
    "CLINICAL_READY_CLAIM_DETECTED": ["clinical readiness", "clinical ready"],
    "HIGH_STAKES_EDUCATION_READY_CLAIM_DETECTED": ["high-stakes education readiness", "high-stakes education ready"],
}

NEGATION_MARKERS = ["not ", "no ", "does not ", "do not ", "without ", "unsupported", "false", "reject", "rejected", "missing"]
GENERATED_PATH_PARTS = ["target/", "node_modules/", ".svelte-kit/"]
GENERATED_SUFFIXES = [".exe", ".dll", ".so", ".dylib", ".zip", ".tar", ".tar.gz", ".tgz", ".7z", ".ckpt", ".bin"]
GENERATED_NAME_PARTS = ["checkpoint", "smoke/", "service job artifacts", "release artifacts", "confirm_snapshot/"]
MUTATION_PATHS = ["LICENSE", "instnct-core", "tools/instnct_service_alpha", "tools/instnct_deploy", "docs/releases"]

ALLOWED_MUTATIONS = {
    "instnct-core/examples/phase_lane_model_capability_benchmark_gate.rs",
    "scripts/probes/run_stable_loop_phase_lock_069_model_capability_benchmark_check.py",
    "docs/research/STABLE_LOOP_PHASE_LOCK_069_MODEL_CAPABILITY_BENCHMARK_GATE_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_069_MODEL_CAPABILITY_BENCHMARK_GATE_RESULT.md",
}

POSITIVE_VERDICTS = [
    "MODEL_CAPABILITY_BENCHMARK_STATIC_CHECK_POSITIVE",
    "MODEL_CAPABILITY_BENCHMARK_PACKAGE_WRITTEN",
    "EVAL_ONLY_CHECKPOINT_GATE_WRITTEN",
    "FINITE_LABEL_SURFACE_BOUNDARY_WRITTEN",
    "OPEN_ENDED_LIMITATION_REQUIRED",
    "HUMAN_SAMPLE_REPORT_REQUIRED",
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


def root_license_changed() -> bool:
    return bool(git_status(["LICENSE"]))


def runtime_surface_mutation_detected() -> bool:
    status = git_status(MUTATION_PATHS)
    for raw in status.splitlines():
        path = raw[3:].replace("\\", "/")
        if path in ALLOWED_MUTATIONS:
            continue
        return True
    return False


def generated_artifact_staged() -> bool:
    status = git_status()
    for raw in status.splitlines():
        path = raw[3:].replace("\\", "/")
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
    terms = REQUIRED_TERMS + TASK_FAMILIES
    return [term for term in terms if term not in bundle]


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

    verdicts = POSITIVE_VERDICTS.copy() if check_pass else ["MODEL_CAPABILITY_BENCHMARK_STATIC_CHECK_FAILS"]
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
