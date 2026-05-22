#!/usr/bin/env python3
"""Static/artifact checker for STABLE_LOOP_PHASE_LOCK_116."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_116_raw_assistant_capability_ceiling_and_gap_map/smoke"
REQUIRED_DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_116_RAW_ASSISTANT_CAPABILITY_CEILING_AND_GAP_MAP_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_116_RAW_ASSISTANT_CAPABILITY_CEILING_AND_GAP_MAP_RESULT.md",
]
REQUIRED_SOURCE = [
    "scripts/probes/run_stable_loop_phase_lock_116_raw_assistant_capability_ceiling_and_gap_map.py",
    "scripts/probes/run_stable_loop_phase_lock_116_raw_assistant_capability_ceiling_and_gap_map_check.py",
]
ALLOWED_MUTATIONS = set(REQUIRED_DOCS + REQUIRED_SOURCE)
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "eval_config.json",
    "upstream_115_manifest.json",
    "upstream_114_manifest.json",
    "upstream_113_manifest.json",
    "upstream_112_manifest.json",
    "upstream_099_manifest.json",
    "checkpoint_integrity_manifest.json",
    "bounded_release_integrity_manifest.json",
    "ceiling_dataset.jsonl",
    "eval_row_hashes.json",
    "freshness_leakage_audit.json",
    "raw_generation_results.jsonl",
    "control_results.jsonl",
    "tier_metrics.json",
    "family_metrics.json",
    "ceiling_by_tier.json",
    "failure_mode_map.json",
    "capability_gap_map.json",
    "retention_report.json",
    "collapse_metrics.json",
    "overclaim_exfiltration_report.json",
    "next_training_targets.json",
    "actual_inference_diagnostic_report.json",
    "human_readable_samples.jsonl",
    "failure_case_samples.jsonl",
    "decision.json",
    "summary.json",
    "report.md",
]
REQUIRED_TERMS = [
    "STABLE_LOOP_PHASE_LOCK_116_RAW_ASSISTANT_CAPABILITY_CEILING_AND_GAP_MAP",
    "eval-only",
    "deterministic rubric-bounded",
    "RAW_ASSISTANT_CAPABILITY_CEILING_AND_GAP_MAP_POSITIVE",
    "UPSTREAM_115_STRESS_CONFIRM_VERIFIED",
    "CEILING_AND_GAP_MAP_COMPLETE",
    "FAILURE_MODE_MAP_WRITTEN",
    "NEXT_TRAINING_TARGETS_WRITTEN",
    "117_TARGETED_CAPABILITY_REPAIR_OR_SCALE_PLAN",
    "not GPT-like assistant readiness",
    "not open-domain assistant readiness",
    "not production chat",
    "not public API",
    "not deployment readiness",
    "not safety alignment",
]
NEGATION_MARKERS = ["not ", "no ", "does not ", "do not ", "without ", "false", "forbidden", "blocked", "denial"]
FORBIDDEN_CLAIMS = {
    "GPT_LIKE_READINESS_FALSE_CLAIM": ["GPT-like ready", "GPT-like assistant ready"],
    "OPEN_DOMAIN_ASSISTANT_CLAIM_DETECTED": ["open-domain assistant ready"],
    "PRODUCTION_CHAT_CLAIM_DETECTED": ["production chat ready", "production ready"],
    "PUBLIC_API_CLAIM_DETECTED": ["public API ready"],
    "DEPLOYMENT_CLAIM_DETECTED": ["deployment ready"],
    "SAFETY_ALIGNMENT_CLAIM_DETECTED": ["safety aligned"],
}
MAIN_ARM = "POST_112_RAW_CURRENT_CHASSIS_CEILING_MAP"
ARMS = {MAIN_ARM, "CURRENT_RAW_BASELINE", "STATIC_OUTPUT_CONTROL", "COPY_PROMPT_CONTROL", "RANDOM_FACT_CONTROL", "RANDOM_SLOT_CONTROL"}


def git_status() -> str:
    result = subprocess.run(["git", "status", "--short"], cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    return result.stdout.rstrip("\n")


def changed_paths() -> list[str]:
    return [line[3:].replace("\\", "/") for line in git_status().splitlines() if line.strip()]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def claim_is_negated(text: str, phrase: str) -> bool:
    lowered = text.lower()
    phrase_lower = phrase.lower()
    for match in re.finditer(re.escape(phrase_lower), lowered):
        window = lowered[max(0, match.start() - 220) : match.start()]
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


def runtime_surface_mutation_detected() -> bool:
    for path in changed_paths():
        if path in ALLOWED_MUTATIONS or path.startswith("target/"):
            continue
        return True
    return False


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


def check_artifacts() -> list[str]:
    failures: list[str] = []
    for rel in REQUIRED_ARTIFACTS:
        if not (SMOKE_ROOT / rel).exists():
            failures.append(f"MISSING_ARTIFACT:{rel}")
    if failures:
        return failures

    summary = load_json(SMOKE_ROOT / "summary.json")
    metrics = summary.get("metrics", {})
    verdicts = set(summary.get("verdicts", []))
    if "RAW_ASSISTANT_CAPABILITY_CEILING_AND_GAP_MAP_POSITIVE" not in verdicts:
        failures.append("RAW_ASSISTANT_CAPABILITY_CEILING_AND_GAP_MAP_RESULT_MISSING")
    for key, expected in [
        ("eval_only", True),
        ("training_performed", False),
        ("checkpoint_mutated", False),
        ("service_started", False),
        ("deployment_smoke_run", False),
        ("runtime_surface_mutated", False),
        ("bounded_release_stack_mutated", False),
        ("gpt_like_assistant_readiness_claimed", False),
        ("open_domain_assistant_readiness_claimed", False),
        ("production_chat_claimed", False),
        ("public_api_claimed", False),
        ("deployment_readiness_claimed", False),
        ("safety_alignment_claimed", False),
    ]:
        if summary.get(key) is not expected:
            failures.append("GPT_LIKE_READINESS_FALSE_CLAIM" if "claimed" in key else "RUNTIME_SURFACE_MUTATION_DETECTED")
    for key, expected in [
        ("train_step_count", 0),
        ("optimizer_step_count", 0),
        ("checkpoint_mutated", False),
        ("service_started", False),
        ("deployment_smoke_run", False),
        ("llm_judge_used", False),
        ("subjective_scoring_used", False),
        ("current_world_fact_scoring_used", False),
    ]:
        if metrics.get(key) != expected:
            failures.append("TRAINING_SIDE_EFFECT_DETECTED" if "step" in key else "ORACLE_SHORTCUT_DETECTED")

    config = load_json(SMOKE_ROOT / "eval_config.json")
    if config.get("seeds") != [2111, 2112, 2113] or config.get("rows_per_family_per_tier") != 32:
        failures.append("FULL_CONFIGURED_RUN_NOT_USED")
    for key, expected in [("max_context_chars", 32768), ("noise_blocks", 32), ("format_variants", 12), ("table_rows", 64), ("multi_doc_count", 8), ("multi_turn_depth", 6)]:
        if config.get(key) != expected:
            failures.append("FULL_CONFIGURED_RUN_NOT_USED")
    if config.get("expected_row_count") != 13056 or config.get("main_scored_arm") != MAIN_ARM or set(config.get("arms", [])) != ARMS:
        failures.append("FULL_CONFIGURED_RUN_NOT_USED")
    if config.get("current_world_fact_scoring_used") is not False or config.get("llm_judge_used") is not False:
        failures.append("UNBOUNDED_WORLD_KNOWLEDGE_SCORING_DETECTED")

    dataset = read_jsonl(SMOKE_ROOT / "ceiling_dataset.jsonl")
    if len(dataset) != 13056:
        failures.append("FULL_CONFIGURED_RUN_NOT_USED")
    progress = read_jsonl(SMOKE_ROOT / "progress.jsonl")
    events = {row.get("event") for row in progress}
    for event in ["start", "upstream_verification", "dataset_build", "freshness_leakage_audit_start", "freshness_leakage_audit", "aggregate_analysis", "decision_writing", "final_verdict"]:
        if event not in events:
            failures.append(f"PROGRESS_EVENT_MISSING:{event}")

    leakage = load_json(SMOKE_ROOT / "freshness_leakage_audit.json")
    if leakage.get("leakage_detected") or leakage.get("exact_prompt_overlap") != 0 or leakage.get("exact_expected_output_overlap") != 0 or leakage.get("near_duplicate_prompt_count") != 0:
        failures.append("BENCHMARK_LEAKAGE_DETECTED")

    row_hashes = load_json(SMOKE_ROOT / "eval_row_hashes.json")
    if len({payload.get("eval_row_hash") for payload in row_hashes.get("arms", {}).values()}) != 1:
        failures.append("BASELINE_EVAL_MISMATCH")

    ceiling = load_json(SMOKE_ROOT / "ceiling_by_tier.json")
    if len(ceiling.get("tiers", {})) != 8 or ceiling.get("ceiling_status") not in {"breakpoint_found", "ceiling_not_reached_within_config"}:
        failures.append("CEILING_MAP_INCOMPLETE")
    failure_map = load_json(SMOKE_ROOT / "failure_mode_map.json")
    if failure_map.get("map_complete") is not True or failure_map.get("unknown_failure_rate", 1.0) > 0.10:
        failures.append("FAILURE_MAP_INCOMPLETE")
    gap_map = load_json(SMOKE_ROOT / "capability_gap_map.json")
    if gap_map.get("ceiling_status") != ceiling.get("ceiling_status"):
        failures.append("CAPABILITY_GAP_MAP_INCOMPLETE")
    next_targets = load_json(SMOKE_ROOT / "next_training_targets.json")
    if next_targets.get("recommended_next") != "117_TARGETED_CAPABILITY_REPAIR_OR_SCALE_PLAN":
        failures.append("NEXT_TRAINING_TARGETS_MISSING")

    retention = load_json(SMOKE_ROOT / "retention_report.json")
    if retention.get("retention_preserved") is not True:
        failures.append("RETENTION_REGRESSION_DETECTED")
    collapse = load_json(SMOKE_ROOT / "collapse_metrics.json")
    if collapse.get("collapse_rejected") is not True:
        failures.append("COLLAPSE_DETECTED")
    overclaim = load_json(SMOKE_ROOT / "overclaim_exfiltration_report.json")
    if overclaim.get("overclaim_or_exfiltration_detected") is not False:
        failures.append("OVERCLAIM_DETECTED")

    aggregate = load_json(SMOKE_ROOT / "aggregate_metrics.json")
    if aggregate.get("ceiling_map_complete") is not True or aggregate.get("unknown_failure_rate", 1.0) > 0.10:
        failures.append("CEILING_MAP_INCOMPLETE")
    if aggregate.get("controls_failed") is not True:
        failures.append("CONTROL_UNEXPECTED_PASS")
    if aggregate.get("benchmark_leakage_detected") is not False:
        failures.append("BENCHMARK_LEAKAGE_DETECTED")
    if aggregate.get("retention_preserved") is not True:
        failures.append("RETENTION_REGRESSION_DETECTED")
    if aggregate.get("collapse_rejected") is not True:
        failures.append("COLLAPSE_DETECTED")

    decision = load_json(SMOKE_ROOT / "decision.json")
    if decision.get("next") not in {"117_TARGETED_CAPABILITY_REPAIR_OR_SCALE_PLAN", "117I_ACTUAL_INFERENCE_INTEGRATION_GAP_ANALYSIS"}:
        failures.append("DECISION_NEXT_MISMATCH")
    diagnostic = load_json(SMOKE_ROOT / "actual_inference_diagnostic_report.json")
    if diagnostic.get("diagnostic_only") is not True or diagnostic.get("checkpoint_mutated") is not False:
        failures.append("ACTUAL_INFERENCE_DIAGNOSTIC_BOUNDARY_FAILURE")

    text_to_scan = "\n".join([(SMOKE_ROOT / "report.md").read_text(encoding="utf-8"), json.dumps(summary, sort_keys=True)])
    failures.extend(find_false_claims(text_to_scan))
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    failures: list[str] = []
    missing, files = read_files()
    failures.extend(f"MISSING_REQUIRED_FILE:{item}" for item in missing)
    combined = "\n".join(files.values())
    for term in REQUIRED_TERMS:
        if term not in combined:
            failures.append(f"MISSING_REQUIRED_TERM:{term}")
    docs_text = "\n".join(files.get(rel, "") for rel in REQUIRED_DOCS)
    failures.extend(find_false_claims(docs_text))
    if runtime_surface_mutation_detected():
        failures.append("RUNTIME_SURFACE_MUTATION_DETECTED")
    if "LICENSE" in changed_paths():
        failures.append("ROOT_LICENSE_CHANGED")
    if args.check_only:
        failures.extend(check_artifacts())
    if failures:
        print("116 checker failed:")
        for failure in sorted(set(failures)):
            print(f" - {failure}")
        return 1
    print("116 checker passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
