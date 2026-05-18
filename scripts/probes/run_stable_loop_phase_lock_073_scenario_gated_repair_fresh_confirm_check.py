#!/usr/bin/env python3
"""Static checker for STABLE_LOOP_PHASE_LOCK_073 scenario gated fresh confirm."""

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
    "docs/research/STABLE_LOOP_PHASE_LOCK_073_SCENARIO_GATED_REPAIR_FRESH_CONFIRM_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_073_SCENARIO_GATED_REPAIR_FRESH_CONFIRM_RESULT.md",
]

REQUIRED_SOURCE = [
    "instnct-core/examples/phase_lane_scenario_gated_repair_fresh_confirm.rs",
    "scripts/probes/run_stable_loop_phase_lock_073_scenario_gated_repair_fresh_confirm_check.py",
]

ALLOWED_MUTATIONS = set(REQUIRED_DOCS + REQUIRED_SOURCE)

EXACT_COMMANDS = [
    "cargo check -p instnct-core --example phase_lane_scenario_gated_repair_fresh_confirm",
    "cargo run -p instnct-core --example phase_lane_scenario_gated_repair_fresh_confirm -- --out target/pilot_wave/stable_loop_phase_lock_073_scenario_gated_repair_fresh_confirm/smoke --checkpoint target/pilot_wave/stable_loop_phase_lock_072_counterfactual_scenario_binding_repair/smoke/checkpoints/scenario_gated_sidepacket_repair/model_checkpoint.json --upstream-072-root target/pilot_wave/stable_loop_phase_lock_072_counterfactual_scenario_binding_repair/smoke --upstream-071b-root target/pilot_wave/stable_loop_phase_lock_071b_repair_overfit_failure_analysis/smoke --upstream-071-root target/pilot_wave/stable_loop_phase_lock_071_repaired_checkpoint_capability_confirm/smoke --upstream-070-root target/pilot_wave/stable_loop_phase_lock_070_distractor_resistant_anchorroute_training/smoke --seed 2027 --heartbeat-sec 20",
    "python -m py_compile scripts/probes/run_stable_loop_phase_lock_073_scenario_gated_repair_fresh_confirm_check.py",
    "python scripts/probes/run_stable_loop_phase_lock_073_scenario_gated_repair_fresh_confirm_check.py --check-only",
    "python scripts/probes/run_stable_loop_phase_lock_072_counterfactual_scenario_binding_repair_check.py --check-only",
    "python scripts/probes/run_stable_loop_phase_lock_071b_repair_overfit_failure_analysis_check.py --check-only",
    "git diff --check",
]

BOUNDARY_TOKENS = [
    "finite-label scenario-state confirmation only",
    "no training",
    "no checkpoint repair",
    "no checkpoint mutation",
    "no upstream rerun",
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
    "phase_lane_scenario_gated_repair_fresh_confirm.rs",
    "UPSTREAM_072_ARTIFACT_MISSING",
    "train_step_count = 0",
    "checkpoint_hash_before",
    "checkpoint_hash_after",
    "checkpoint_hash_unchanged = true",
    "prediction_oracle_used = false",
    "TRAINING_SIDE_EFFECT_DETECTED",
    "CHECKPOINT_MUTATION_DETECTED",
    "ORACLE_SHORTCUT_DETECTED",
    "overlap_with_070_eval_count = 0",
    "overlap_with_071_eval_count = 0",
    "overlap_with_071b_failure_digest_count = 0",
    "overlap_with_072_train_count = 0",
    "overlap_with_072_eval_count = 0",
    "BENCHMARK_LEAKAGE_DETECTED",
    "FRESH_EVAL_LEAKAGE_DETECTED",
    "eval_row_hash_model",
    "eval_row_hash_baselines",
    "eval_row_hash_no_route_control",
    "eval_row_hash_ungated_control",
    "eval_row_hash_shuffled_control",
    "baseline_eval_mismatch = false",
    "BASELINE_EVAL_MISMATCH",
    "FRESH_ACTIVE_SCENARIO_BINDING",
    "FRESH_COUNTERFACTUAL_SCENARIO_SWITCH",
    "FRESH_DISTRACTOR_SCENARIO_REJECTION",
    "FRESH_OLD_SCENARIO_SUPPRESSION",
    "FRESH_INACTIVE_POCKET_SUPPRESSION",
    "FRESH_STALE_POCKET_SUPPRESSION",
    "FRESH_FIRST_LEDGER_BIAS_SUPPRESSION",
    "FRESH_SIDE_NOTE_SUPPRESSION",
    "FRESH_ANSWER_ONLY_SCENARIO_BINDING",
    "FRESH_TRACE_MIXED_SCENARIO_BINDING",
    "RETENTION_INSTRUCTION_FOLLOWING_CLOSED",
    "RETENTION_MULTI_HOP_KEY_VALUE_BINDING",
    "RETENTION_SYMBOLIC_RULE_CLOSED_CHOICE",
    "RETENTION_NON_ROUTE_TEXT_CONTROL",
    "OPEN_ENDED_INTERFACE_LIMITATION",
    "MAJORITY_LABEL",
    "COPY_FIRST_MATCH",
    "COPY_LAST_TOKEN",
    "NO_ROUTE_FEATURE_CONTROL",
    "UNGATED_SIDEPACKET_SIMULATED_CONTROL",
    "SHUFFLED_SCENARIO_LABEL_CONTROL",
    "SHUFFLED_SCENARIO_CONTROL_UNEXPECTED_PASS",
    "active_scenario_selection_accuracy",
    "distractor_scenario_selection_rate",
    "old_scenario_selection_rate",
    "inactive_pocket_selection_rate",
    "stale_pocket_selection_rate",
    "first_ledger_bias_rate",
    "side_note_leak_rate",
    "SCENARIO_SOURCE_ATTRIBUTION_MISSING",
    "fresh_active_scenario_binding_accuracy >= 0.90",
    "fresh_counterfactual_scenario_switch_accuracy >= 0.85",
    "fresh_distractor_scenario_rejection_accuracy >= 0.90",
    "fresh_old_scenario_suppression_accuracy >= 0.90",
    "fresh_inactive_pocket_suppression_accuracy >= 0.85",
    "fresh_stale_pocket_suppression_accuracy >= 0.85",
    "fresh_first_ledger_bias_suppression_accuracy >= 0.85",
    "fresh_side_note_suppression_accuracy >= 0.85",
    "fresh_answer_only_scenario_binding_accuracy >= 0.85",
    "family_min_accuracy >= 0.75",
    "supported_accuracy >= 0.88",
    "delta_vs_no_route_control > 0.10",
    "delta_vs_ungated_sidepacket_control > 0.03",
    "delta_vs_copy_first_match > 0.10",
    "shuffled_scenario_label_control_accuracy < 0.70",
    "retention families pass",
    "top_output_rate <= 0.45",
    "space_output_rate <= 0.02",
    "empty_output_rate <= 0.02",
    "collapse_detected = false",
    "GATED_WRITEBACK_NOT_UNIQUELY_CONFIRMED_ON_FRESH_EVAL",
    "TRACE_DEPENDENCE_DETECTED",
    "RETENTION_CONFIRM_FAILS",
    "queue.json",
    "progress.jsonl",
    "benchmark_config.json",
    "upstream_072_manifest.json",
    "checkpoint_manifest.json",
    "capability_dataset_manifest.json",
    "benchmark_examples_sample.jsonl",
    "baseline_metrics.json",
    "no_route_feature_control_metrics.json",
    "ungated_sidepacket_control_metrics.json",
    "shuffled_scenario_control_metrics.json",
    "capability_metrics.json",
    "per_family_metrics.json",
    "scenario_selection_metrics.json",
    "pocket_suppression_metrics.json",
    "retention_metrics.json",
    "limitation_report.json",
    "human_readable_samples.jsonl",
    "failure_case_samples.jsonl",
    "collapse_metrics.json",
    "summary.json",
    "report.md",
    "task_family",
    "input",
    "expected_output",
    "model_output",
    "baseline_outputs",
    "no_route_output",
    "ungated_control_output",
    "shuffled_control_output",
    "pass_fail",
    "limitation_flag",
    "SCENARIO_GATED_REPAIR_FRESH_CONFIRM_POSITIVE",
    "UPSTREAM_072_CHECKPOINT_VERIFIED",
    "NO_TRAINING_PERFORMED",
    "CHECKPOINT_UNCHANGED",
    "FRESH_ACTIVE_SCENARIO_BINDING_PASSES",
    "FRESH_COUNTERFACTUAL_GENERALIZATION_PASSES",
    "FRESH_POCKET_SUPPRESSION_PASSES",
    "FRESH_GATED_ADVANTAGE_CONFIRMED",
    "SCENARIO_SOURCE_ATTRIBUTION_RECORDED",
    "ANSWER_ONLY_FRESH_SCENARIO_BINDING_PASSES",
    "FRESH_EVAL_LEAKAGE_REJECTED",
    "NO_ROUTE_CONTROL_RECORDED",
    "UNGATED_CONTROL_RECORDED",
    "SHUFFLED_SCENARIO_CONTROL_FAILS",
    "RETENTION_CONFIRM_PASSES",
    "BASELINE_COMPARISON_RECORDED",
    "HUMAN_READABLE_SAMPLES_WRITTEN",
    "OPEN_ENDED_LIMITATION_RECORDED",
    "PRODUCTION_TRAINING_NOT_CLAIMED",
    "SCENARIO_GATED_REPAIR_FRESH_CONFIRM_FAILS",
    "NO_ROUTE_CONTROL_MISSING",
    "UNGATED_CONTROL_MISSING",
    "FRESH_ACTIVE_SCENARIO_BINDING_FAILS",
    "FRESH_COUNTERFACTUAL_GENERALIZATION_FAILS",
    "FRESH_POCKET_SUPPRESSION_FAILS",
    "CAPABILITY_FAMILY_GATE_FAILS",
    "STATIC_OUTPUT_COLLAPSE_DETECTED",
    "HUMAN_SAMPLE_REPORT_MISSING",
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
    "SCENARIO_GATED_REPAIR_FRESH_CONFIRM_STATIC_CHECK_POSITIVE",
    "SCENARIO_GATED_REPAIR_FRESH_CONFIRM_PACKAGE_WRITTEN",
    "EVAL_ONLY_HARD_WALL_WRITTEN",
    "FRESH_EVAL_LEAKAGE_GUARD_WRITTEN",
    "SAME_ROW_CONTROLS_REQUIRED",
    "SCENARIO_SOURCE_ATTRIBUTION_REQUIRED",
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

    verdicts = POSITIVE_VERDICTS.copy() if check_pass else ["SCENARIO_GATED_REPAIR_FRESH_CONFIRM_STATIC_CHECK_FAILS"]
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
