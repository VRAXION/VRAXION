#!/usr/bin/env python3
"""Static checker for STABLE_LOOP_PHASE_LOCK_074 multi-seed scenario confirm."""

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
    "docs/research/STABLE_LOOP_PHASE_LOCK_074_MULTI_SEED_SCENARIO_GATED_REPAIR_CONFIRM_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_074_MULTI_SEED_SCENARIO_GATED_REPAIR_CONFIRM_RESULT.md",
]

REQUIRED_SOURCE = [
    "scripts/probes/run_stable_loop_phase_lock_074_multi_seed_scenario_gated_repair_confirm.py",
    "scripts/probes/run_stable_loop_phase_lock_074_multi_seed_scenario_gated_repair_confirm_check.py",
]

ALLOWED_MUTATIONS = set(REQUIRED_DOCS + REQUIRED_SOURCE)

EXACT_COMMANDS = [
    "cargo check -p instnct-core --example phase_lane_scenario_gated_repair_fresh_confirm",
    "python -m py_compile scripts/probes/run_stable_loop_phase_lock_074_multi_seed_scenario_gated_repair_confirm.py",
    "python -m py_compile scripts/probes/run_stable_loop_phase_lock_074_multi_seed_scenario_gated_repair_confirm_check.py",
    "python scripts/probes/run_stable_loop_phase_lock_074_multi_seed_scenario_gated_repair_confirm.py --out target/pilot_wave/stable_loop_phase_lock_074_multi_seed_scenario_gated_repair_confirm/smoke --checkpoint target/pilot_wave/stable_loop_phase_lock_072_counterfactual_scenario_binding_repair/smoke/checkpoints/scenario_gated_sidepacket_repair/model_checkpoint.json --upstream-072-root target/pilot_wave/stable_loop_phase_lock_072_counterfactual_scenario_binding_repair/smoke --upstream-071b-root target/pilot_wave/stable_loop_phase_lock_071b_repair_overfit_failure_analysis/smoke --upstream-071-root target/pilot_wave/stable_loop_phase_lock_071_repaired_checkpoint_capability_confirm/smoke --upstream-070-root target/pilot_wave/stable_loop_phase_lock_070_distractor_resistant_anchorroute_training/smoke --seeds 2027,2028,2029 --heartbeat-sec 20",
    "python scripts/probes/run_stable_loop_phase_lock_074_multi_seed_scenario_gated_repair_confirm_check.py --check-only",
    "python scripts/probes/run_stable_loop_phase_lock_073_scenario_gated_repair_fresh_confirm_check.py --check-only",
    "python scripts/probes/run_stable_loop_phase_lock_072_counterfactual_scenario_binding_repair_check.py --check-only",
    "git diff --check",
]

BOUNDARY_TOKENS = [
    "multi-seed eval only",
    "no training",
    "no checkpoint repair",
    "no checkpoint mutation",
    "no stale child artifacts",
    "no mean-only pass",
    "no open-ended assistant",
    "no free-form generation",
    "no perplexity",
    "no full English LM",
    "no language grounding",
    "no production training",
    "no release readiness by itself",
    "no service API change",
    "no deployment harness change",
    "no release docs change",
    "no public crate export change",
    "no root LICENSE change",
]

REQUIRED_TERMS = [
    "STABLE_LOOP_PHASE_LOCK_074_MULTI_SEED_SCENARIO_GATED_REPAIR_CONFIRM",
    "phase_lane_scenario_gated_repair_fresh_confirm",
    "2027,2028,2029",
    "UPSTREAM_072_ARTIFACT_MISSING",
    "child_run_started = true",
    "child_run_completed = true",
    "child_exit_code = 0",
    "child_summary_newer_than_074_start = true",
    "child_report_newer_than_074_start = true",
    "child_command recorded exactly",
    "STALE_CHILD_ARTIFACT_USED",
    "no mean-only pass",
    "MULTI_SEED_SCENARIO_INSTABILITY_DETECTED",
    "SCENARIO_GATED_REPAIR_FRESH_CONFIRM_POSITIVE",
    "train_step_count = 0",
    "checkpoint_hash_unchanged = true",
    "prediction_oracle_used = false",
    "baseline_eval_mismatch = false",
    "all overlap counts = 0",
    "collapse_detected = false",
    "CHILD_073_GATE_RECHECK_FAILS",
    "fresh_active_scenario_binding_accuracy >= 0.90",
    "fresh_counterfactual_scenario_switch_accuracy >= 0.85",
    "fresh_distractor_scenario_rejection_accuracy >= 0.90",
    "fresh_old_scenario_suppression_accuracy >= 0.90",
    "fresh_inactive_pocket_suppression_accuracy >= 0.85",
    "fresh_stale_pocket_suppression_accuracy >= 0.85",
    "fresh_answer_only_scenario_binding_accuracy >= 0.85",
    "family_min_accuracy >= 0.85",
    "supported_accuracy >= 0.88",
    "active_scenario_selection_accuracy >= 0.95",
    "distractor_scenario_selection_rate <= 0.05",
    "old_scenario_selection_rate <= 0.05",
    "delta_vs_no_route_control > 0.10",
    "delta_vs_ungated_sidepacket_control > 0.03",
    "delta_vs_copy_first_match > 0.10",
    "shuffled_scenario_label_control_accuracy < 0.70",
    "GATED_ADVANTAGE_REGRESSION_DETECTED",
    "SHUFFLED_SCENARIO_CONTROL_UNEXPECTED_PASS",
    "active_scenario_selection_accuracy",
    "distractor_scenario_selection_rate",
    "old_scenario_selection_rate",
    "inactive_pocket_selection_rate",
    "stale_pocket_selection_rate",
    "first_ledger_bias_rate",
    "side_note_leak_rate",
    "queue.json",
    "progress.jsonl",
    "multi_seed_config.json",
    "upstream_072_manifest.json",
    "child_run_manifest.json",
    "seed_metrics.jsonl",
    "aggregate_metrics.json",
    "multi_seed_stability.json",
    "baseline_knockout_aggregate.json",
    "scenario_source_attribution_aggregate.json",
    "retention_aggregate.json",
    "failure_case_samples.jsonl",
    "summary.json",
    "report.md",
    "seed",
    "task_family",
    "input",
    "expected_output",
    "model_output",
    "baseline_outputs",
    "no_route_output",
    "ungated_control_output",
    "shuffled_control_output",
    "wrong_answer_source",
    "short_diagnosis",
    "multi_seed_eval_only = true",
    "open_ended_generation_supported = false",
    "free_form_answering_supported = false",
    "perplexity_supported = false",
    "full_English_LM_supported = false",
    "language_grounding_claimed = false",
    "production_training_claimed = false",
    "MULTI_SEED_SCENARIO_GATED_REPAIR_CONFIRM_POSITIVE",
    "FRESH_CHILD_RUNS_CONFIRMED",
    "CHILD_073_GATES_RECHECKED",
    "MULTI_SEED_MIN_GATE_PASSES",
    "SCENARIO_GATED_REPAIR_STABLE_ACROSS_SEEDS",
    "FRESH_GATED_ADVANTAGE_STABLE",
    "SCENARIO_SOURCE_ATTRIBUTION_AGGREGATED",
    "SHUFFLED_SCENARIO_CONTROL_FAILS_ALL_SEEDS",
    "CHECKPOINT_UNCHANGED_ALL_SEEDS",
    "NO_TRAINING_PERFORMED",
    "FRESH_EVAL_LEAKAGE_REJECTED_ALL_SEEDS",
    "BASELINE_COMPARISON_RECORDED",
    "FAILURE_CASE_REPORT_WRITTEN",
    "OPEN_ENDED_LIMITATION_RECORDED",
    "PRODUCTION_TRAINING_NOT_CLAIMED",
    "MULTI_SEED_SCENARIO_GATED_REPAIR_CONFIRM_FAILS",
    "CHECKPOINT_MUTATION_DETECTED",
    "TRAINING_SIDE_EFFECT_DETECTED",
    "ORACLE_SHORTCUT_DETECTED",
    "BENCHMARK_LEAKAGE_DETECTED",
    "BASELINE_EVAL_MISMATCH",
    "STATIC_OUTPUT_COLLAPSE_DETECTED",
    "FAILURE_CASE_REPORT_MISSING",
    "OPEN_ENDED_CLAIM_DETECTED",
    "PERPLEXITY_CLAIM_DETECTED",
    "RUNTIME_SURFACE_MUTATION_DETECTED",
    "ROOT_LICENSE_CHANGED",
]

PLACEHOLDERS = ["TODO", "TBD", "placeholder", "pending measured", "coming soon"]

FORBIDDEN_CLAIMS = {
    "OPEN_ENDED_CLAIM_DETECTED": ["open-ended assistant", "free-form generation", "free form generation"],
    "PERPLEXITY_CLAIM_DETECTED": ["perplexity benchmark", "perplexity training"],
    "FULL_ENGLISH_LM_CLAIM_DETECTED": ["full English LM"],
    "LANGUAGE_GROUNDING_CLAIM_DETECTED": ["language grounding"],
    "PRODUCTION_TRAINING_CLAIM_DETECTED": ["production training", "production model"],
    "RELEASE_READINESS_CLAIM_DETECTED": ["release readiness by itself"],
    "GA_CLAIM_DETECTED": ["GA release", "GA launched", "generally available"],
    "PUBLIC_BETA_CLAIM_DETECTED": ["public beta"],
    "HOSTED_SAAS_CLAIM_DETECTED": ["hosted SaaS"],
    "SERVICE_API_CLAIM_DETECTED": ["service API change"],
    "DEPLOYMENT_HARNESS_CLAIM_DETECTED": ["deployment harness change"],
    "ROOT_LICENSE_CHANGED": ["root LICENSE changed"],
}

NEGATION_MARKERS = ["not ", "no ", "does not ", "do not ", "without ", "unsupported", "false", "reject", "rejected"]
GENERATED_PATH_PARTS = ["target/", "node_modules/", ".svelte-kit/"]
GENERATED_SUFFIXES = [".exe", ".dll", ".so", ".dylib", ".zip", ".tar", ".tar.gz", ".tgz", ".7z", ".ckpt", ".bin"]
GENERATED_NAME_PARTS = ["checkpoint", "smoke/", "service job artifacts", "release artifacts"]

POSITIVE_VERDICTS = [
    "MULTI_SEED_SCENARIO_GATED_REPAIR_CONFIRM_STATIC_CHECK_POSITIVE",
    "MULTI_SEED_SCENARIO_GATED_REPAIR_CONFIRM_PACKAGE_WRITTEN",
    "FRESH_CHILD_RUN_GUARDS_WRITTEN",
    "NO_MEAN_ONLY_PASS_GUARD_WRITTEN",
    "CHILD_073_RECHECK_REQUIRED",
    "SCENARIO_SOURCE_ATTRIBUTION_AGGREGATE_REQUIRED",
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
    prefix = line[:phrase_start]
    return any(marker in prefix for marker in NEGATION_MARKERS)


def forbidden_claim_hits(files: dict[str, str]) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for rel, text in files.items():
        if rel.endswith(".py"):
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

    verdicts = POSITIVE_VERDICTS.copy() if check_pass else ["MULTI_SEED_SCENARIO_GATED_REPAIR_CONFIRM_STATIC_CHECK_FAILS"]
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
