#!/usr/bin/env python3
"""Static checker for STABLE_LOOP_PHASE_LOCK_082 chat diversity multi-seed confirm."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

REQUIRED_DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_082_CHAT_DIVERSITY_MULTI_SEED_CONFIRM_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_082_CHAT_DIVERSITY_MULTI_SEED_CONFIRM_RESULT.md",
]

REQUIRED_SOURCE = [
    "scripts/probes/run_stable_loop_phase_lock_082_chat_diversity_multi_seed_confirm.py",
    "scripts/probes/run_stable_loop_phase_lock_082_chat_diversity_multi_seed_confirm_check.py",
]

ALLOWED_MUTATIONS = set(REQUIRED_DOCS + REQUIRED_SOURCE)

EXACT_COMMANDS = [
    "cargo check -p instnct-core --example phase_lane_chat_diversity_fresh_confirm",
    "python -m py_compile scripts/probes/run_stable_loop_phase_lock_082_chat_diversity_multi_seed_confirm.py",
    "python -m py_compile scripts/probes/run_stable_loop_phase_lock_082_chat_diversity_multi_seed_confirm_check.py",
    "python scripts/probes/run_stable_loop_phase_lock_082_chat_diversity_multi_seed_confirm.py --out target/pilot_wave/stable_loop_phase_lock_082_chat_diversity_multi_seed_confirm/smoke --upstream-080-root target/pilot_wave/stable_loop_phase_lock_080_chat_composition_diversity_repair/smoke --upstream-079b-root target/pilot_wave/stable_loop_phase_lock_079b_chat_composition_fresh_failure_analysis/smoke --upstream-079-root target/pilot_wave/stable_loop_phase_lock_079_chat_composition_fresh_confirm/smoke --upstream-078-root target/pilot_wave/stable_loop_phase_lock_078_chat_composition_repair/smoke --upstream-074-root target/pilot_wave/stable_loop_phase_lock_074_multi_seed_scenario_gated_repair_confirm/smoke --seeds 2027,2028,2029 --heartbeat-sec 20",
    "python scripts/probes/run_stable_loop_phase_lock_082_chat_diversity_multi_seed_confirm_check.py --check-only",
    "python scripts/probes/run_stable_loop_phase_lock_081_chat_diversity_fresh_confirm_check.py --check-only",
    "python scripts/probes/run_stable_loop_phase_lock_080_chat_composition_diversity_repair_check.py --check-only",
    "git diff --check",
]

BOUNDARY_TOKENS = [
    "bounded multi-seed chat diversity confirmation only",
    "not GPT-like assistant readiness",
    "not full English LM",
    "not language grounding",
    "not production chat",
    "not safety alignment",
    "not public beta / GA / hosted SaaS",
    "no service API change",
    "no deployment harness change",
    "no SDK/public export change",
    "no release docs change",
    "no root LICENSE change",
    "no upstream checkpoint mutation",
]

REQUIRED_TERMS = [
    "STABLE_LOOP_PHASE_LOCK_082_CHAT_DIVERSITY_MULTI_SEED_CONFIRM",
    "run_stable_loop_phase_lock_082_chat_diversity_multi_seed_confirm.py",
    "run_stable_loop_phase_lock_082_chat_diversity_multi_seed_confirm_check.py",
    "phase_lane_chat_diversity_fresh_confirm",
    "2027,2028,2029",
    "UPSTREAM_080_ARTIFACT_MISSING",
    "STALE_CHILD_ARTIFACT_USED",
    "child_run_started = true",
    "child_run_completed = true",
    "child_exit_code = 0",
    "child_summary_newer_than_082_start = true",
    "child_report_newer_than_082_start = true",
    "child_command recorded exactly",
    "no mean-only pass",
    "MULTI_SEED_CHAT_DIVERSITY_INSTABILITY_DETECTED",
    "CHILD_081_GATE_RECHECK_FAILS",
    "CHAT_DIVERSITY_FRESH_CONFIRM_POSITIVE",
    "train_step_count = 0",
    "checkpoint_hash_unchanged = true",
    "prediction_oracle_used = false",
    "llm_judge_used = false",
    "response_table_used_for_main_prediction = false",
    "template_copy_rate <= 0.25",
    "response_table_copy_rate <= 0.20",
    "semantic_template_overlap_rate <= 0.50",
    "slot_only_skeleton_reuse_rate <= 0.25",
    "response_skeleton_reuse_rate <= 0.50",
    "top_skeleton_rate <= 0.35",
    "response_skeleton_diversity >= 0.50",
    "slot_binding_accuracy >= 0.75",
    "finite_label_retention_accuracy >= 0.90",
    "empty_output_rate <= 0.02",
    "space_output_rate <= 0.02",
    "static_response_rate <= 0.15",
    "repetition_rate <= 0.20",
    "copy_prompt_rate <= 0.15",
    "all prompt overlap counts = 0",
    "near_duplicate_prompt_count = 0",
    "FRESH_PROMPT_LEAKAGE_DETECTED",
    "CHECKPOINT_MUTATION_DETECTED",
    "TEMPLATE_COPY_REGRESSION_DETECTED",
    "SKELETON_REUSE_REGRESSION_DETECTED",
    "VOCAB_DIVERSITY_REGRESSION_DETECTED",
    "CONTEXT_SLOT_BINDING_REGRESSION_DETECTED",
    "FINITE_LABEL_RETENTION_REGRESSION_DETECTED",
    "STATIC_RESPONSE_COLLAPSE_DETECTED",
    "FAILURE_CASE_REPORT_MISSING",
    "GPT_LIKE_READINESS_FALSE_CLAIM",
    "PRODUCTION_CHAT_CLAIM_DETECTED",
    "RUNTIME_SURFACE_MUTATION_DETECTED",
    "ROOT_LICENSE_CHANGED",
    "min_novel_response_rate",
    "max_template_copy_rate",
    "min_response_skeleton_diversity",
    "max_response_skeleton_reuse_rate",
    "min_slot_binding_accuracy",
    "min_finite_label_retention_accuracy",
    "stddev_novel_response_rate",
    "stddev_template_copy_rate",
    "stddev_response_entropy",
    "queue.json",
    "progress.jsonl",
    "multi_seed_config.json",
    "upstream_080_manifest.json",
    "child_run_manifest.json",
    "seed_metrics.jsonl",
    "aggregate_metrics.json",
    "multi_seed_stability.json",
    "novelty_aggregate.json",
    "skeleton_diversity_aggregate.json",
    "vocabulary_entropy_aggregate.json",
    "context_slot_aggregate.json",
    "finite_label_retention_aggregate.json",
    "failure_case_samples.jsonl",
    "summary.json",
    "report.md",
    "seed",
    "eval_family",
    "prompt",
    "model_output",
    "expected_behavior",
    "pass_fail",
    "novelty_flag",
    "template_copy_flag",
    "skeleton_reuse_flag",
    "semantic_template_overlap_score",
    "slot_binding_diagnosis",
    "short_diagnosis",
    "CHAT_DIVERSITY_MULTI_SEED_CONFIRM_POSITIVE",
    "FRESH_CHILD_RUNS_CONFIRMED",
    "CHILD_081_GATES_RECHECKED",
    "MULTI_SEED_MIN_GATE_PASSES",
    "CHAT_DIVERSITY_STABLE_ACROSS_SEEDS",
    "TEMPLATE_COPY_REJECTED_ALL_SEEDS",
    "SKELETON_REUSE_REJECTED_ALL_SEEDS",
    "VOCAB_ENTROPY_PASSES_ALL_SEEDS",
    "CONTEXT_SLOT_BINDING_PASSES_ALL_SEEDS",
    "FINITE_LABEL_ANCHORROUTE_RETENTION_PASSES_ALL_SEEDS",
    "CHECKPOINT_UNCHANGED_ALL_SEEDS",
    "NO_TRAINING_PERFORMED",
    "FAILURE_CASE_REPORT_WRITTEN",
    "PRODUCTION_CHAT_NOT_CLAIMED",
    "CHAT_DIVERSITY_MULTI_SEED_CONFIRM_FAILS",
    "083_CHAT_MODEL_ARTIFACT_RC_PACKAGE",
    "082B_CHAT_DIVERSITY_MULTI_SEED_FAILURE_ANALYSIS",
]

PLACEHOLDERS = ["TODO", "TBD", "placeholder", "pending measured", "coming soon"]

FORBIDDEN_CLAIMS = {
    "GPT_LIKE_READINESS_FALSE_CLAIM": ["GPT-like assistant readiness"],
    "FULL_ENGLISH_LM_CLAIM_DETECTED": ["full English LM"],
    "LANGUAGE_GROUNDING_CLAIM_DETECTED": ["language grounding"],
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

NEGATION_MARKERS = ["not ", "no ", "does not ", "do not ", "without ", "false", "reject", "rejected", "bounded"]
GENERATED_PATH_PARTS = ["target/", "node_modules/", ".svelte-kit/"]
GENERATED_SUFFIXES = [".exe", ".dll", ".so", ".dylib", ".zip", ".tar", ".tar.gz", ".tgz", ".7z", ".ckpt", ".bin"]

POSITIVE_VERDICTS = [
    "CHAT_DIVERSITY_MULTI_SEED_CONFIRM_STATIC_CHECK_POSITIVE",
    "CHAT_DIVERSITY_MULTI_SEED_CONFIRM_PACKAGE_WRITTEN",
    "FRESH_CHILD_RUN_GUARDS_WRITTEN",
    "NO_MEAN_ONLY_PASS_GUARD_WRITTEN",
    "CHILD_081_RECHECK_REQUIRED",
    "MULTI_SEED_MIN_STDDEV_AGGREGATES_REQUIRED",
    "FAILURE_CASE_REPORT_REQUIRED",
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
        if raw.strip():
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


def claim_is_negated(text: str, phrase: str) -> bool:
    lowered = text.lower()
    phrase_lower = phrase.lower()
    for match in re.finditer(re.escape(phrase_lower), lowered):
        window = lowered[max(0, match.start() - 80) : match.start()]
        if any(marker in window for marker in NEGATION_MARKERS):
            return True
    return False


def find_false_claims(text: str) -> list[str]:
    failures: list[str] = []
    for verdict, phrases in FORBIDDEN_CLAIMS.items():
        for phrase in phrases:
            if phrase.lower() in text.lower() and not claim_is_negated(text, phrase):
                failures.append(verdict)
                break
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    failures: list[str] = []
    missing, files = read_files()
    if missing:
        failures.extend([f"MISSING:{item}" for item in missing])

    joined = "\n".join(files.values())
    audited_text = "\n".join(text for rel, text in files.items() if not rel.endswith("_check.py"))
    for term in REQUIRED_TERMS + BOUNDARY_TOKENS + EXACT_COMMANDS:
        if term not in joined:
            failures.append(f"MISSING_TERM:{term}")
    for marker in PLACEHOLDERS:
        if marker.lower() in audited_text.lower():
            failures.append(f"PLACEHOLDER_PRESENT:{marker}")
    failures.extend(find_false_claims(audited_text))

    if root_license_changed():
        failures.append("ROOT_LICENSE_CHANGED")
    if runtime_surface_mutation_detected():
        failures.append("RUNTIME_SURFACE_MUTATION_DETECTED")
    if generated_artifact_staged():
        failures.append("GENERATED_ARTIFACT_STAGED")

    payload = {
        "schema_version": "chat_diversity_multi_seed_confirm_static_check_v1",
        "status": "passed" if not failures else "failed",
        "check_only": args.check_only,
        "verdicts": POSITIVE_VERDICTS if not failures else ["CHAT_DIVERSITY_MULTI_SEED_CONFIRM_STATIC_CHECK_FAILS", *failures],
        "checked_files": REQUIRED_DOCS + REQUIRED_SOURCE,
        "allowed_mutations": sorted(ALLOWED_MUTATIONS),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
