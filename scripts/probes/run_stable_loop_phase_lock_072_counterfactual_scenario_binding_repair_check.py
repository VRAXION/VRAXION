#!/usr/bin/env python3
"""Static checker for STABLE_LOOP_PHASE_LOCK_072 scenario binding repair."""

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
    "docs/research/STABLE_LOOP_PHASE_LOCK_072_COUNTERFACTUAL_SCENARIO_BINDING_REPAIR_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_072_COUNTERFACTUAL_SCENARIO_BINDING_REPAIR_RESULT.md",
]

REQUIRED_SOURCE = [
    "instnct-core/examples/phase_lane_counterfactual_scenario_binding_repair.rs",
    "scripts/probes/run_stable_loop_phase_lock_072_counterfactual_scenario_binding_repair_check.py",
]

ALLOWED_MUTATIONS = set(REQUIRED_DOCS + REQUIRED_SOURCE)

EXACT_COMMANDS = [
    "cargo check -p instnct-core --example phase_lane_counterfactual_scenario_binding_repair",
    "cargo run -p instnct-core --example phase_lane_counterfactual_scenario_binding_repair -- --out target/pilot_wave/stable_loop_phase_lock_072_counterfactual_scenario_binding_repair/smoke --upstream-checkpoint target/pilot_wave/stable_loop_phase_lock_070_distractor_resistant_anchorroute_training/smoke/checkpoints/finetune_068_targeted_repair/model_checkpoint.json --upstream-071b-root target/pilot_wave/stable_loop_phase_lock_071b_repair_overfit_failure_analysis/smoke --targeted-examples 120000 --seed 2026 --heartbeat-sec 20",
    "python -m py_compile scripts/probes/run_stable_loop_phase_lock_072_counterfactual_scenario_binding_repair_check.py",
    "python scripts/probes/run_stable_loop_phase_lock_072_counterfactual_scenario_binding_repair_check.py --check-only",
    "python scripts/probes/run_stable_loop_phase_lock_071b_repair_overfit_failure_analysis_check.py --check-only",
    "python scripts/probes/run_stable_loop_phase_lock_071_repaired_checkpoint_capability_confirm_check.py --check-only",
    "git diff --check",
]

BOUNDARY_TOKENS = [
    "finite-label scenario-state repair only",
    "no open-ended assistant",
    "no free-form generation",
    "no perplexity",
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
    "protected highway plus gated sidepocket",
    "base route stays protected",
    "only active scenario sidepacket may write back to readout",
    "NO_TRAIN_071_BASELINE",
    "STANDARD_TARGETED_REPAIR_BASELINE",
    "SCENARIO_GATED_SIDEPACKET_REPAIR",
    "UNGATED_SIDEPACKET_REPAIR_CONTROL",
    "NO_ROUTE_FEATURE_CONTROL",
    "SHUFFLED_SCENARIO_LABEL_CONTROL",
    "CHECKPOINT_RELOAD_EVAL",
    "ROLLBACK_REHEARSAL",
    "RESUME_FROM_CHECKPOINT",
    "ACTIVE_SCENARIO_MARKER_BINDING",
    "SAME_KEY_DIFFERENT_SCENARIO_SWITCH",
    "DISTRACTOR_SCENARIO_REJECTION",
    "STALE_SCENARIO_SUPPRESSION",
    "INACTIVE_POCKET_NEGATIVE_ROUTE",
    "FIRST_LEDGER_BIAS_SUPPRESSION",
    "SIDE_NOTE_SUPPRESSION",
    "ANSWER_ONLY_SCENARIO_BINDING",
    "TRACE_MIXED_SCENARIO_BINDING",
    "RETENTION_INSTRUCTION_FOLLOWING_CLOSED",
    "RETENTION_MULTI_HOP_KEY_VALUE_BINDING",
    "RETENTION_SYMBOLIC_RULE_CLOSED_CHOICE",
    "RETENTION_NON_ROUTE_TEXT_CONTROL",
    "scenario:active",
    "scenario:old",
    "scenario:distractor",
    "pocket:active",
    "pocket:inactive",
    "writeback:active_only",
    "trace_mixed_accuracy",
    "answer_only_accuracy",
    "answer_only_active_scenario_accuracy",
    "fresh_counterfactual_binding_accuracy >= 0.85",
    "active_scenario_selection_accuracy >= 0.90",
    "distractor_scenario_selection_rate <= 0.10",
    "old_scenario_selection_rate <= 0.10",
    "fresh_context_entity_extraction_accuracy >= 0.85",
    "fresh_irrelevant_pocket_suppression_accuracy >= 0.80",
    "answer_only_active_scenario_accuracy >= 0.85",
    "family_min_accuracy >= 0.70",
    "collapse_detected = false",
    "delta_vs_standard_targeted > 0.03",
    "delta_vs_ungated_sidepacket > 0.03",
    "delta_vs_no_route > 0.10",
    "GATED_WRITEBACK_NOT_UNIQUELY_PROVEN",
    "base_route_retention_accuracy",
    "protected_highway_regression",
    "sidepocket_writeback_accuracy",
    "inactive_sidepocket_not_readout_rate",
    "PROTECTED_HIGHWAY_REGRESSION_DETECTED",
    "RETENTION_REGRESSION_DETECTED",
    "overlap_with_071_eval_count = 0",
    "overlap_with_071b_failure_digest_count = 0",
    "overlap_with_070_eval_count = 0",
    "baseline_eval_mismatch = false",
    "prediction_oracle_used = false",
    "TRAIN_BENCHMARK_LEAKAGE_DETECTED",
    "BASELINE_EVAL_MISMATCH",
    "SHUFFLED_SCENARIO_CONTROL_UNEXPECTED_PASS",
    "TRACE_DEPENDENCE_DETECTED",
    "queue.json",
    "progress.jsonl",
    "training_config.json",
    "upstream_checkpoint_manifest.json",
    "upstream_071b_manifest.json",
    "targeted_dataset_manifest.json",
    "train_examples_sample.jsonl",
    "eval_examples_sample.jsonl",
    "training_metrics.jsonl",
    "checkpoint_manifest.json",
    "checkpoint_hashes.json",
    "arm_comparison.json",
    "scenario_selection_metrics.json",
    "pocket_writeback_metrics.json",
    "protected_highway_metrics.json",
    "retention_metrics.json",
    "wrong_answer_source_after_repair.json",
    "baseline_knockout_report.json",
    "failure_case_samples.jsonl",
    "collapse_metrics.json",
    "summary.json",
    "report.md",
    "COUNTERFACTUAL_SCENARIO_BINDING_REPAIR_POSITIVE",
    "SCENARIO_GATED_SIDEPACKET_REPAIR_COMPLETED",
    "ACTIVE_SCENARIO_SELECTION_IMPROVED",
    "DISTRACTOR_SCENARIO_REJECTED",
    "STALE_SCENARIO_SUPPRESSED",
    "INACTIVE_POCKET_SUPPRESSED",
    "GATED_WRITEBACK_ADVANTAGE_SHOWN",
    "PROTECTED_HIGHWAY_PRESERVED",
    "ANSWER_ONLY_SCENARIO_BINDING_PASSES",
    "SHUFFLED_SCENARIO_CONTROL_FAILS",
    "TRAIN_BENCHMARK_LEAKAGE_REJECTED",
    "RETENTION_GATE_PASSES",
    "BEST_ARM_SELECTED",
    "UPSTREAM_CHECKPOINT_UNCHANGED",
    "ORACLE_SHORTCUT_REJECTED",
    "PRODUCTION_TRAINING_NOT_CLAIMED",
    "COUNTERFACTUAL_SCENARIO_BINDING_REPAIR_FAILS",
    "ACTIVE_SCENARIO_SELECTION_STILL_FAILS",
    "DISTRACTOR_SCENARIO_STILL_SELECTED",
    "STALE_SCENARIO_STILL_SELECTED",
    "INACTIVE_POCKET_STILL_SELECTED",
    "FAILURE_CASE_REPORT_MISSING",
    "CHECKPOINT_MUTATION_DETECTED",
    "ORACLE_SHORTCUT_DETECTED",
    "OPEN_ENDED_CLAIM_DETECTED",
    "RUNTIME_SURFACE_MUTATION_DETECTED",
    "ROOT_LICENSE_CHANGED",
]

PLACEHOLDERS = ["TODO", "TBD", "placeholder", "pending measured", "coming soon"]

FORBIDDEN_CLAIMS = {
    "OPEN_ENDED_CLAIM_DETECTED": [
        "open-ended assistant",
        "free-form generation",
        "free form generation",
    ],
    "PERPLEXITY_CLAIM_DETECTED": ["perplexity benchmark", "perplexity training"],
    "LANGUAGE_GROUNDING_CLAIM_DETECTED": ["language grounding"],
    "PRODUCTION_TRAINING_CLAIM_DETECTED": ["production training", "production model"],
    "GA_CLAIM_DETECTED": ["GA release", "GA launched", "generally available"],
    "PUBLIC_BETA_CLAIM_DETECTED": ["public beta"],
    "HOSTED_SAAS_CLAIM_DETECTED": ["hosted SaaS"],
    "SERVICE_API_CLAIM_DETECTED": ["service API change"],
    "DEPLOYMENT_HARNESS_CLAIM_DETECTED": ["deployment harness change"],
    "ROOT_LICENSE_CHANGED": ["root LICENSE changed"],
}

NEGATION_MARKERS = [
    "not ",
    "no ",
    "does not ",
    "do not ",
    "without ",
    "unsupported",
    "false",
    "reject",
    "rejected",
]
GENERATED_PATH_PARTS = ["target/", "node_modules/", ".svelte-kit/"]
GENERATED_SUFFIXES = [
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".zip",
    ".tar",
    ".tar.gz",
    ".tgz",
    ".7z",
    ".ckpt",
    ".bin",
]
GENERATED_NAME_PARTS = ["checkpoint", "smoke/", "service job artifacts", "release artifacts"]

POSITIVE_VERDICTS = [
    "COUNTERFACTUAL_SCENARIO_BINDING_REPAIR_STATIC_CHECK_POSITIVE",
    "SCENARIO_BINDING_REPAIR_PACKAGE_WRITTEN",
    "GATED_SIDEPACKET_COMPARISON_REQUIRED",
    "PROTECTED_HIGHWAY_METRICS_REQUIRED",
    "ANSWER_ONLY_GATE_REQUIRED",
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
        if path.startswith("docs/releases/"):
            return True
        if path.startswith("docs/product/"):
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

    verdicts = POSITIVE_VERDICTS.copy() if check_pass else ["COUNTERFACTUAL_SCENARIO_BINDING_REPAIR_STATIC_CHECK_FAILS"]
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
