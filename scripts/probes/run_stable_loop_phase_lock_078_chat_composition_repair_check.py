#!/usr/bin/env python3
"""Static checker for STABLE_LOOP_PHASE_LOCK_078 chat composition repair."""

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
    "docs/research/STABLE_LOOP_PHASE_LOCK_078_CHAT_COMPOSITION_REPAIR_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_078_CHAT_COMPOSITION_REPAIR_RESULT.md",
]

REQUIRED_SOURCE = [
    "instnct-core/examples/phase_lane_chat_composition_repair.rs",
    "scripts/probes/run_stable_loop_phase_lock_078_chat_composition_repair_check.py",
]

ALLOWED_MUTATIONS = set(REQUIRED_DOCS + REQUIRED_SOURCE)

EXACT_COMMANDS = [
    "cargo check -p instnct-core --example phase_lane_chat_composition_repair",
    "cargo run -p instnct-core --example phase_lane_chat_composition_repair -- --out target/pilot_wave/stable_loop_phase_lock_078_chat_composition_repair/smoke --upstream-076-root target/pilot_wave/stable_loop_phase_lock_076_chat_generation_poc/smoke --upstream-077-root target/pilot_wave/stable_loop_phase_lock_077_chat_generation_fresh_composition_confirm/smoke --upstream-077b-root target/pilot_wave/stable_loop_phase_lock_077b_chat_generation_failure_analysis/smoke --upstream-074-root target/pilot_wave/stable_loop_phase_lock_074_multi_seed_scenario_gated_repair_confirm/smoke --chat-examples 60000 --seed 2026 --heartbeat-sec 20",
    "python -m py_compile scripts/probes/run_stable_loop_phase_lock_078_chat_composition_repair_check.py",
    "python scripts/probes/run_stable_loop_phase_lock_078_chat_composition_repair_check.py --check-only",
    "python scripts/probes/run_stable_loop_phase_lock_077b_chat_generation_failure_analysis_check.py --check-only",
    "python scripts/probes/run_stable_loop_phase_lock_077_chat_generation_fresh_composition_confirm_check.py --check-only",
    "python scripts/probes/run_stable_loop_phase_lock_076_chat_generation_poc_check.py --check-only",
    "git diff --check",
]

BOUNDARY_TOKENS = [
    "runner-local",
    "token-level next-token",
    "response_table_used_for_main_prediction = false",
    "response_table_path_available_but_disabled = true",
    "llm_judge_used = false",
    "no service API change",
    "no deployment harness change",
    "no SDK/public export change",
    "no release docs change",
    "no root LICENSE change",
    "no upstream checkpoint mutation",
    "not GPT-like assistant readiness",
    "not full English LM",
    "not production chat",
    "not safety alignment",
    "not public beta",
    "not GA",
    "not hosted SaaS",
]

REQUIRED_TERMS = [
    "STABLE_LOOP_PHASE_LOCK_078_CHAT_COMPOSITION_REPAIR",
    "phase_lane_chat_composition_repair.rs",
    "run_stable_loop_phase_lock_078_chat_composition_repair_check.py",
    "UPSTREAM_076_ARTIFACT_MISSING",
    "UPSTREAM_077B_ARTIFACT_MISSING",
    "NO_ACTUAL_TRAINING_UPDATE_DETECTED",
    "TOKEN_OBJECTIVE_NOT_LEARNED",
    "RESPONSE_TABLE_DEPENDENCE_STILL_HIGH",
    "TEMPLATE_COPY_STILL_HIGH",
    "CONTEXT_SLOT_BINDING_STILL_FAILS",
    "BOUNDARY_REFUSAL_MINI_STILL_FAILS",
    "CONTROL_DELTA_INSUFFICIENT",
    "FINITE_LABEL_RETENTION_REGRESSION_DETECTED",
    "TRAIN_EVAL_LEAKAGE_DETECTED",
    "ORACLE_SHORTCUT_DETECTED",
    "CHAT_EVAL_RUBRIC_MISSING",
    "HUMAN_SAMPLE_REPORT_MISSING",
    "RUNTIME_SURFACE_MUTATION_DETECTED",
    "ROOT_LICENSE_CHANGED",
    "UPSTREAM_CHECKPOINT_MUTATION_DETECTED",
    "CHAT_COMPOSITION_REPAIR_POSITIVE",
    "TOKEN_LEVEL_COMPOSITION_TRAINING_COMPLETED",
    "TOKEN_OBJECTIVE_LEARNED",
    "RESPONSE_TABLE_DEPENDENCE_REDUCED",
    "TEMPLATE_COPY_REJECTED",
    "CONTEXT_SLOT_BINDING_REPAIRED",
    "BOUNDARY_REFUSAL_MINI_REPAIRED",
    "FINITE_LABEL_ANCHORROUTE_RETENTION_PASSES",
    "CONTROL_DELTA_PASSES",
    "CHECKPOINT_PIPELINE_PASSES",
    "UPSTREAM_CHECKPOINT_UNCHANGED",
    "PRODUCTION_CHAT_NOT_CLAIMED",
    "queue.json",
    "progress.jsonl",
    "training_config.json",
    "upstream_manifest.json",
    "repair_dataset_manifest.json",
    "train_examples_sample.jsonl",
    "eval_examples_sample.jsonl",
    "training_metrics.jsonl",
    "checkpoint_manifest.json",
    "checkpoint_hashes.json",
    "generation_samples.jsonl",
    "human_readable_samples.jsonl",
    "composition_metrics.json",
    "novelty_metrics.json",
    "context_slot_metrics.json",
    "finite_label_retention_metrics.json",
    "collapse_metrics.json",
    "arm_comparison.json",
    "summary.json",
    "report.md",
    "token_train_step_count",
    "token_loss_initial",
    "token_loss_final",
    "teacher_forced_next_token_accuracy",
    "response_table_used_for_main_prediction = false",
    "decoder_path = token_level_next_token",
    "response_table_path_available_but_disabled = true",
    "llm_judge_used = false",
    "slot_value_expected",
    "slot_value_emitted",
    "slot_binding_accuracy",
    "wrong_slot_rate",
    "missing_slot_rate",
    "stale_slot_rate",
    "exact_template_copy_rate",
    "semantic_template_overlap_rate",
    "response_table_copy_rate",
    "novel_response_rate",
    "train_response_ngram_overlap",
    "delta_vs_response_table_only_control",
    "delta_vs_no_context_slot_control",
    "delta_vs_no_dropout_control",
    "boundary_refusal_accuracy is not safety alignment",
    "no production safety claim",
    "no clinical/high-stakes readiness",
    "079_CHAT_COMPOSITION_FRESH_CONFIRM",
    "078B_CHAT_COMPOSITION_REPAIR_FAILURE_ANALYSIS",
]

PLACEHOLDERS = ["TODO", "TBD", "placeholder", "pending measured", "coming soon"]

FORBIDDEN_CLAIMS = {
    "GPT_LIKE_READINESS_FALSE_CLAIM": ["GPT-like assistant readiness"],
    "FULL_ENGLISH_LM_CLAIM_DETECTED": ["full English LM"],
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

NEGATION_MARKERS = ["not ", "no ", "does not ", "do not ", "without ", "false", "reject", "rejected"]
GENERATED_PATH_PARTS = ["target/", "node_modules/", ".svelte-kit/"]
GENERATED_SUFFIXES = [".exe", ".dll", ".so", ".dylib", ".zip", ".tar", ".tar.gz", ".tgz", ".7z", ".ckpt", ".bin"]

POSITIVE_VERDICTS = [
    "CHAT_COMPOSITION_REPAIR_STATIC_CHECK_POSITIVE",
    "CHAT_COMPOSITION_REPAIR_PACKAGE_WRITTEN",
    "RUNNER_LOCAL_TOKEN_DECODER_GUARD_WRITTEN",
    "RESPONSE_TABLE_BYPASS_REQUIRED",
    "TOKEN_OBJECTIVE_PROOF_REQUIRED",
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

    verdicts = POSITIVE_VERDICTS.copy() if check_pass else ["CHAT_COMPOSITION_REPAIR_STATIC_CHECK_FAILS"]
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
