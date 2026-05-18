#!/usr/bin/env python3
"""Static checker for STABLE_LOOP_PHASE_LOCK_075 chat-surface gap analysis."""

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
    "docs/research/STABLE_LOOP_PHASE_LOCK_075_CHAT_SURFACE_BASELINE_AND_GAP_ANALYSIS_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_075_CHAT_SURFACE_BASELINE_AND_GAP_ANALYSIS_RESULT.md",
]

REQUIRED_SOURCE = [
    "instnct-core/examples/phase_lane_chat_surface_baseline_gap_analysis.rs",
    "scripts/probes/run_stable_loop_phase_lock_075_chat_surface_baseline_gap_analysis_check.py",
]

ALLOWED_MUTATIONS = set(REQUIRED_DOCS + REQUIRED_SOURCE)

EXACT_COMMANDS = [
    "cargo check -p instnct-core --example phase_lane_chat_surface_baseline_gap_analysis",
    "cargo run -p instnct-core --example phase_lane_chat_surface_baseline_gap_analysis -- --out target/pilot_wave/stable_loop_phase_lock_075_chat_surface_baseline_gap_analysis/smoke --checkpoint target/pilot_wave/stable_loop_phase_lock_072_counterfactual_scenario_binding_repair/smoke/checkpoints/scenario_gated_sidepacket_repair/model_checkpoint.json --upstream-074-root target/pilot_wave/stable_loop_phase_lock_074_multi_seed_scenario_gated_repair_confirm/smoke --seed 2026 --heartbeat-sec 20",
    "python -m py_compile scripts/probes/run_stable_loop_phase_lock_075_chat_surface_baseline_gap_analysis_check.py",
    "python scripts/probes/run_stable_loop_phase_lock_075_chat_surface_baseline_gap_analysis_check.py --check-only",
    "python scripts/probes/run_stable_loop_phase_lock_074_multi_seed_scenario_gated_repair_confirm_check.py --check-only",
    "git diff --check",
]

BOUNDARY_TOKENS = [
    "eval-only chat-surface baseline/gap analysis",
    "no training",
    "no checkpoint repair",
    "no checkpoint mutation",
    "no decoder behavior",
    "no open-ended assistant capability proven",
    "no free-form generation proven unless directly measured",
    "no perplexity support",
    "no full English LM",
    "no language grounding",
    "no production training",
    "no chat release readiness",
    "no service API change",
    "no deployment harness change",
    "no release docs change",
    "no public crate export change",
    "no root LICENSE change",
]

REQUIRED_TERMS = [
    "STABLE_LOOP_PHASE_LOCK_075_CHAT_SURFACE_BASELINE_AND_GAP_ANALYSIS",
    "phase_lane_chat_surface_baseline_gap_analysis.rs",
    "UPSTREAM_074_ARTIFACT_MISSING",
    "train_step_count = 0",
    "checkpoint_hash_before",
    "checkpoint_hash_after",
    "checkpoint_hash_unchanged = true",
    "prediction_oracle_used = false",
    "decoder_generation_loop_available",
    "chat_generation_supported",
    "free_form_answering_supported",
    "multi_turn_dialogue_supported",
    "perplexity_supported = false",
    "finite_label_surface = true",
    "chat_release_readiness_proven = false",
    "TRAINING_SIDE_EFFECT_DETECTED",
    "CHECKPOINT_MUTATION_DETECTED",
    "ORACLE_SHORTCUT_DETECTED",
    "RUNTIME_SURFACE_MUTATION_DETECTED",
    "ROOT_LICENSE_CHANGED",
    "FREE_FORM_RESPONSE_PROBE",
    "MULTI_TOKEN_CONTINUATION_PROBE",
    "SINGLE_TURN_INSTRUCTION_PROBE",
    "TWO_TURN_DIALOGUE_PROBE",
    "CONTEXT_CARRY_CHAT_PROBE",
    "BOUNDARY_REFUSAL_PROBE",
    "DEGENERATION_PROBE",
    "FINITE_LABEL_CONTROL_PROBE",
    "finite_label",
    "empty",
    "space_only",
    "copied_prompt_fragment",
    "static_repeated_output",
    "unsupported",
    "free_form_candidate",
    "queue.json",
    "progress.jsonl",
    "chat_probe_config.json",
    "upstream_074_manifest.json",
    "checkpoint_manifest.json",
    "chat_probe_dataset.jsonl",
    "chat_probe_outputs.jsonl",
    "human_readable_samples.jsonl",
    "gap_analysis.json",
    "degeneration_metrics.json",
    "finite_label_control_metrics.json",
    "summary.json",
    "report.md",
    "probe_family",
    "prompt",
    "expected_behavior",
    "raw_model_output",
    "output_classification",
    "pass_fail_or_unsupported",
    "short_diagnosis",
    "empty_output_rate",
    "space_output_rate",
    "repeated_output_rate",
    "static_output_rate",
    "label_only_rate",
    "copy_prompt_rate",
    "unique_output_count",
    "CHAT_SURFACE_BASELINE_GAP_ANALYSIS_FAILS",
    "CHAT_GENERATION_SURFACE_UNSUPPORTED",
    "FREE_FORM_ANSWERING_UNSUPPORTED",
    "MULTI_TURN_DIALOGUE_UNSUPPORTED",
    "PERPLEXITY_UNSUPPORTED",
    "OPEN_ENDED_CHAT_READY_CLAIM_REJECTED",
    "UPSTREAM_074_CHECKPOINT_VERIFIED",
    "NO_TRAINING_PERFORMED",
    "CHECKPOINT_UNCHANGED",
    "FINITE_LABEL_SURFACE_CONFIRMED",
    "CHAT_GAP_ANALYSIS_WRITTEN",
    "HUMAN_READABLE_SAMPLES_WRITTEN",
    "PRODUCTION_TRAINING_NOT_CLAIMED",
    "OPEN_ENDED_CHAT_READY_CLAIM_DETECTED",
    "FREE_FORM_GENERATION_FALSE_CLAIM",
    "PERPLEXITY_CLAIM_DETECTED",
    "HUMAN_SAMPLE_REPORT_MISSING",
    "076_CHAT_GENERATION_POC",
]

PLACEHOLDERS = ["TODO", "TBD", "placeholder", "pending measured", "coming soon"]

FORBIDDEN_CLAIMS = {
    "OPEN_ENDED_CHAT_READY_CLAIM_DETECTED": [
        "open-ended assistant capability proven",
        "chat release readiness proven",
        "chat-ready",
    ],
    "FREE_FORM_GENERATION_FALSE_CLAIM": [
        "free-form generation proven",
        "free form generation proven",
    ],
    "PERPLEXITY_CLAIM_DETECTED": ["perplexity benchmark", "perplexity support"],
    "FULL_ENGLISH_LM_CLAIM_DETECTED": ["full English LM capability", "full English model"],
    "LANGUAGE_GROUNDING_CLAIM_DETECTED": ["language grounding proven", "language grounding claim"],
    "PRODUCTION_TRAINING_CLAIM_DETECTED": ["production training claim", "production model"],
    "GA_CLAIM_DETECTED": ["GA release", "GA launched", "generally available"],
    "PUBLIC_BETA_CLAIM_DETECTED": ["public beta"],
    "HOSTED_SAAS_CLAIM_DETECTED": ["hosted SaaS"],
    "SERVICE_API_CLAIM_DETECTED": ["service API change"],
    "DEPLOYMENT_HARNESS_CLAIM_DETECTED": ["deployment harness change"],
    "ROOT_LICENSE_CHANGED": ["root LICENSE changed"],
}

NEGATION_MARKERS = ["not ", "no ", "does not ", "do not ", "without ", "unsupported", "false", "reject", "rejected", "unless directly measured"]
GENERATED_PATH_PARTS = ["target/", "node_modules/", ".svelte-kit/"]
GENERATED_SUFFIXES = [".exe", ".dll", ".so", ".dylib", ".zip", ".tar", ".tar.gz", ".tgz", ".7z", ".ckpt", ".bin"]
GENERATED_NAME_PARTS = ["checkpoint", "smoke/", "service job artifacts", "release artifacts"]

POSITIVE_VERDICTS = [
    "CHAT_SURFACE_BASELINE_GAP_STATIC_CHECK_POSITIVE",
    "CHAT_SURFACE_BASELINE_GAP_PACKAGE_WRITTEN",
    "EVAL_ONLY_CHAT_GAP_GUARD_WRITTEN",
    "NO_DECODER_ADDITION_GUARD_WRITTEN",
    "UNSUPPORTED_CHAT_OUTCOME_ALLOWED",
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

    verdicts = POSITIVE_VERDICTS.copy() if check_pass else ["CHAT_SURFACE_BASELINE_GAP_STATIC_CHECK_FAILS"]
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
