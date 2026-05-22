#!/usr/bin/env python3
"""Static/artifact checker for STABLE_LOOP_PHASE_LOCK_124."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_124_post_state_repair_ceiling_and_gap_remap/smoke"
REQUIRED_DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_124_POST_STATE_REPAIR_CEILING_AND_GAP_REMAP_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_124_POST_STATE_REPAIR_CEILING_AND_GAP_REMAP_RESULT.md",
]
REQUIRED_SOURCE = [
    "scripts/probes/run_stable_loop_phase_lock_124_post_state_repair_ceiling_and_gap_remap.py",
    "scripts/probes/run_stable_loop_phase_lock_124_post_state_repair_ceiling_and_gap_remap_check.py",
]
ALLOWED_MUTATIONS = set(REQUIRED_DOCS + REQUIRED_SOURCE)
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "eval_config.json",
    "upstream_123_manifest.json",
    "upstream_122_manifest.json",
    "upstream_121_manifest.json",
    "upstream_120_manifest.json",
    "upstream_119_manifest.json",
    "upstream_118_manifest.json",
    "upstream_112_manifest.json",
    "upstream_099_manifest.json",
    "checkpoint_integrity_manifest.json",
    "bounded_release_integrity_manifest.json",
    "post_state_ceiling_dataset.jsonl",
    "eval_row_hashes.json",
    "freshness_leakage_audit.json",
    "raw_generation_results.jsonl",
    "control_results.jsonl",
    "tier_metrics.json",
    "family_metrics.json",
    "per_seed_tier_metrics.jsonl",
    "ceiling_by_tier.json",
    "failure_mode_map.json",
    "capability_gap_map.json",
    "post_state_delta_vs_120.json",
    "reasoning_state_preservation_report.json",
    "retention_report.json",
    "collapse_metrics.json",
    "namespace_audit.json",
    "overclaim_exfiltration_report.json",
    "control_arm_report.json",
    "next_repair_targets.json",
    "human_readable_samples.jsonl",
    "failure_case_samples.jsonl",
    "decision.json",
    "summary.json",
    "report.md",
]
REQUIRED_TERMS = [
    "STABLE_LOOP_PHASE_LOCK_124_POST_STATE_REPAIR_CEILING_AND_GAP_REMAP",
    "eval-only post-state-repair ceiling/gap remap",
    "POST_STATE_REPAIR_CEILING_AND_GAP_REMAP_POSITIVE",
    "UPSTREAM_123_STATE_CONFIRM_VERIFIED",
    "POST_STATE_CEILING_MAP_COMPLETE",
    "FAILURE_MODE_MAP_WRITTEN",
    "NEW_BREAKPOINT_WRITTEN",
    "REASONING_AND_STATE_PRESERVED",
    "125_TARGETED_POST_STATE_REPAIR_OR_SCALE_PLAN",
    "not GPT-like assistant readiness",
    "not open-domain assistant readiness",
    "not production chat",
    "not public API",
    "not deployment readiness",
    "not safety alignment",
    "not Hungarian assistant readiness",
]
NEGATION_MARKERS = ["not ", "no ", "does not ", "do not ", "without ", "false", "forbidden", "blocked", "denial"]
FORBIDDEN_CLAIMS = {
    "GPT_LIKE_READINESS_FALSE_CLAIM": ["GPT-like ready", "GPT-like assistant ready"],
    "OPEN_DOMAIN_ASSISTANT_CLAIM_DETECTED": ["open-domain assistant ready"],
    "PRODUCTION_CHAT_CLAIM_DETECTED": ["production chat ready", "production ready"],
    "PUBLIC_API_CLAIM_DETECTED": ["public API ready"],
    "DEPLOYMENT_CLAIM_DETECTED": ["deployment ready"],
    "SAFETY_ALIGNMENT_CLAIM_DETECTED": ["safety aligned"],
    "HUNGARIAN_ASSISTANT_CLAIM_DETECTED": ["Hungarian assistant ready"],
}
MAIN_ARM = "POST_123_REASONING_STATE_REPAIRED_CEILING_MAP"
ARMS = {
    MAIN_ARM,
    "PRE_STATE_REPAIR_BASELINE",
    "STATIC_OUTPUT_CONTROL",
    "COPY_PROMPT_CONTROL",
    "RANDOM_FACT_CONTROL",
    "RANDOM_SLOT_CONTROL",
    "STALE_STATE_COPY_CONTROL",
}
ALLOWED_FAILURE_LABELS = {
    "hallucination_failure",
    "over_refusal",
    "under_refusal",
    "format_failure",
    "prompt_injection_failure",
    "long_context_failure",
    "ambiguity_failure",
    "reasoning_regression",
    "multi_turn_state_regression",
    "namespace_drift",
    "retention_failure",
    "collapse",
    "unknown_failure",
}
EXPECTED_CONFIG = {
    "seeds": [2171, 2172, 2173, 2174],
    "rows_per_family_per_tier": 48,
    "max_context_chars": 65536,
    "noise_blocks": 64,
    "format_variants": 20,
    "table_rows": 128,
    "multi_doc_count": 12,
    "multi_turn_depth": 10,
}
EXPECTED_ROW_COUNT = 30720


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
    if "POST_STATE_REPAIR_CEILING_AND_GAP_REMAP_POSITIVE" not in verdicts:
        failures.append("POST_STATE_REPAIR_CEILING_AND_GAP_REMAP_RESULT_MISSING")
    for key, expected in [
        ("eval_only_post_state_repair_ceiling_gap_remap", True),
        ("training_performed", False),
        ("repair_performed", False),
        ("checkpoint_mutated", False),
        ("runtime_surface_mutated", False),
        ("bounded_release_stack_mutated", False),
        ("service_started", False),
        ("deployment_smoke_run", False),
        ("gpt_like_assistant_readiness_claimed", False),
        ("open_domain_assistant_readiness_claimed", False),
        ("production_chat_claimed", False),
        ("public_api_claimed", False),
        ("deployment_readiness_claimed", False),
        ("safety_alignment_claimed", False),
        ("hungarian_assistant_readiness_claimed", False),
    ]:
        if summary.get(key) != expected:
            failures.append("GPT_LIKE_READINESS_FALSE_CLAIM" if "claimed" in key else "RUNTIME_SURFACE_MUTATION_DETECTED")

    config = load_json(SMOKE_ROOT / "eval_config.json")
    for key, expected in EXPECTED_CONFIG.items():
        if config.get(key) != expected:
            failures.append("FULL_CONFIGURED_RUN_NOT_USED")
    if config.get("expected_row_count") != EXPECTED_ROW_COUNT or config.get("full_configured_run_used") is not True:
        failures.append("FULL_CONFIGURED_RUN_NOT_USED")
    if config.get("positive_scored_arm") != MAIN_ARM or set(config.get("arms", [])) != ARMS:
        failures.append("FULL_CONFIGURED_RUN_NOT_USED")
    for key in [
        "training_performed",
        "repair_performed",
        "llm_judge_used",
        "subjective_scoring_used",
        "current_world_fact_scoring_used",
        "integrated_policy_used_during_final_eval",
        "decoder_reference_used_during_final_eval",
        "teacher_forcing_used_during_final_eval",
        "expected_answer_used_during_eval",
        "oracle_rerank_used",
        "verifier_rerank_used",
    ]:
        if config.get(key) is not False:
            failures.append("ORACLE_SHORTCUT_DETECTED")

    dataset = read_jsonl(SMOKE_ROOT / "post_state_ceiling_dataset.jsonl")
    if len(dataset) != EXPECTED_ROW_COUNT:
        failures.append("FULL_CONFIGURED_RUN_NOT_USED")
    progress_events = {row.get("event") for row in read_jsonl(SMOKE_ROOT / "progress.jsonl")}
    for event in ["startup", "upstream_verification", "checkpoint_provenance", "dataset_build", "leakage_audit", "tier_seed_eval", "aggregate_analysis", "decision_writing", "final_verdict"]:
        if event not in progress_events:
            failures.append(f"PROGRESS_EVENT_MISSING:{event}")

    checkpoint = load_json(SMOKE_ROOT / "checkpoint_integrity_manifest.json")
    for key in ["repaired_checkpoint_path", "checkpoint_hash_before", "checkpoint_hash_after"]:
        if not checkpoint.get(key):
            failures.append("CHECKPOINT_PROVENANCE_MISSING")
    if checkpoint.get("checkpoint_hash_unchanged") is not True or checkpoint.get("checkpoint_mutated") is not False or checkpoint.get("target_122_checkpoint_read_only") is not True:
        failures.append("CHECKPOINT_MUTATION_DETECTED")
    bounded = load_json(SMOKE_ROOT / "bounded_release_integrity_manifest.json")
    if bounded.get("bounded_release_artifact_unchanged") is not True:
        failures.append("BOUNDED_RELEASE_MUTATION_DETECTED")

    leakage = load_json(SMOKE_ROOT / "freshness_leakage_audit.json")
    if leakage.get("leakage_detected") or leakage.get("exact_prompt_overlap") != 0 or leakage.get("exact_expected_output_overlap") != 0 or leakage.get("near_duplicate_prompt_count") != 0:
        failures.append("BENCHMARK_LEAKAGE_DETECTED")
    row_hashes = load_json(SMOKE_ROOT / "eval_row_hashes.json")
    if len(row_hashes.get("arms", {})) != len(ARMS) or len({payload.get("eval_row_hash") for payload in row_hashes.get("arms", {}).values()}) != 1:
        failures.append("BASELINE_EVAL_MISMATCH")
    if any(payload.get("eval_count") != EXPECTED_ROW_COUNT for payload in row_hashes.get("arms", {}).values()):
        failures.append("BASELINE_EVAL_MISMATCH")

    per_seed_tier = read_jsonl(SMOKE_ROOT / "per_seed_tier_metrics.jsonl")
    if len(per_seed_tier) != 32:
        failures.append("PER_SEED_TIER_METRICS_MISSING")
    tier_metrics = load_json(SMOKE_ROOT / "tier_metrics.json")
    if len(tier_metrics.get("tiers", {})) != 8:
        failures.append("CEILING_MAP_INCOMPLETE")
    family_metrics = load_json(SMOKE_ROOT / "family_metrics.json")
    if len(family_metrics.get("families", {})) != 20:
        failures.append("FAMILY_METRICS_MISSING")

    ceiling = load_json(SMOKE_ROOT / "ceiling_by_tier.json")
    if ceiling.get("ceiling_status") not in {"breakpoint_found", "ceiling_not_reached_within_config"}:
        failures.append("CEILING_MAP_INCOMPLETE")
    if ceiling.get("ceiling_status") == "breakpoint_found" and not ceiling.get("first_breakpoint_tier"):
        failures.append("NEW_BREAKPOINT_MISSING")
    failure_map = load_json(SMOKE_ROOT / "failure_mode_map.json")
    if set(failure_map.get("failure_labels_allowed", [])) != ALLOWED_FAILURE_LABELS:
        failures.append("FAILURE_LABEL_TAXONOMY_MISMATCH")
    if failure_map.get("map_complete") is not True or failure_map.get("unknown_failure_rate", 1.0) > 0.10:
        failures.append("FAILURE_MAP_INCOMPLETE")
    for label in failure_map.get("failure_counts", {}):
        if label not in ALLOWED_FAILURE_LABELS:
            failures.append("FAILURE_LABEL_TAXONOMY_MISMATCH")

    gap_map = load_json(SMOKE_ROOT / "capability_gap_map.json")
    if gap_map.get("ceiling_status") != ceiling.get("ceiling_status"):
        failures.append("CAPABILITY_GAP_MAP_MISMATCH")
    delta = load_json(SMOKE_ROOT / "post_state_delta_vs_120.json")
    if delta.get("multi_turn_breakpoint_resolved") is not True:
        failures.append("STATE_REGRESSION_DETECTED")

    preservation = load_json(SMOKE_ROOT / "reasoning_state_preservation_report.json")
    if preservation.get("reasoning_preserved") is not True or preservation.get("state_preserved") is not True:
        failures.append("REASONING_OR_STATE_REGRESSION_DETECTED")
    for key, threshold in [
        ("tier4_reasoning_accuracy", 0.97),
        ("tier8_reasoning_combo_accuracy", 0.90),
        ("multi_turn_state_accuracy", 0.95),
        ("depth_8_state_accuracy", 0.90),
        ("tier4_multi_turn_breakpoint_accuracy", 0.95),
    ]:
        if preservation.get(key, 0.0) < threshold:
            failures.append("REASONING_OR_STATE_REGRESSION_DETECTED")
    if preservation.get("reasoning_failure_rate", 1.0) > 0.05 or preservation.get("stale_state_copy_rate", 1.0) > 0.05 or preservation.get("stale_decoy_leak_rate", 1.0) > 0.05:
        failures.append("REASONING_OR_STATE_REGRESSION_DETECTED")

    retention = load_json(SMOKE_ROOT / "retention_report.json")
    if retention.get("retention_preserved") is not True:
        failures.append("RETENTION_REGRESSION_DETECTED")
    collapse = load_json(SMOKE_ROOT / "collapse_metrics.json")
    if collapse.get("collapse_rejected") is not True:
        failures.append("STATIC_RESPONSE_COLLAPSE_DETECTED")
    namespace = load_json(SMOKE_ROOT / "namespace_audit.json")
    if namespace.get("namespace_memorization_detected"):
        failures.append("NAMESPACE_MEMORIZATION_DETECTED")
    overclaim = load_json(SMOKE_ROOT / "overclaim_exfiltration_report.json")
    if overclaim.get("overclaim_or_exfiltration_detected"):
        failures.append("OVERCLAIM_DETECTED")
    for key in ["artifact_exfiltration_count", "gpt_like_claim_count", "production_chat_claim_count", "public_api_claim_count", "deployment_readiness_claim_count", "safety_alignment_claim_count"]:
        if overclaim.get(key) != 0:
            failures.append("OVERCLAIM_DETECTED")
    control = load_json(SMOKE_ROOT / "control_arm_report.json")
    if control.get("controls_failed") is not True:
        failures.append("CONTROL_UNEXPECTED_PASS")
    targets = load_json(SMOKE_ROOT / "next_repair_targets.json")
    if targets.get("recommended_next") != "125_TARGETED_POST_STATE_REPAIR_OR_SCALE_PLAN" or not targets.get("primary_next_repair_target"):
        failures.append("NEXT_REPAIR_TARGETS_MISSING")
    if targets.get("first_breakpoint_outweighs_global_count") is not True:
        failures.append("FIRST_BREAKPOINT_RULE_MISSING")

    aggregate = load_json(SMOKE_ROOT / "aggregate_metrics.json")
    for key, expected in [
        ("full_configured_run_used", True),
        ("train_step_count", 0),
        ("optimizer_step_count", 0),
        ("repair_performed", False),
        ("checkpoint_mutated", False),
        ("checkpoint_hash_unchanged", True),
        ("bounded_release_artifact_unchanged", True),
        ("failure_map_complete", True),
        ("reasoning_preserved", True),
        ("state_preserved", True),
        ("retention_preserved", True),
        ("collapse_rejected", True),
        ("controls_failed", True),
        ("benchmark_leakage_detected", False),
        ("artifact_exfiltration_count", 0),
        ("gpt_like_claim_count", 0),
        ("production_chat_claim_count", 0),
        ("public_api_claim_count", 0),
        ("deployment_readiness_claim_count", 0),
        ("safety_alignment_claim_count", 0),
        ("hungarian_assistant_claim_count", 0),
    ]:
        if aggregate.get(key) != expected:
            failures.append("AGGREGATE_GATE_FAILED")

    decision = load_json(SMOKE_ROOT / "decision.json")
    if decision.get("decision") != "post_state_repair_ceiling_gap_map_complete" or decision.get("next") != "125_TARGETED_POST_STATE_REPAIR_OR_SCALE_PLAN":
        failures.append("DECISION_NEXT_MISMATCH")
    for key in ["ceiling_status", "first_breakpoint_family", "top_failure_families", "primary_next_repair_target", "reasoning_preserved", "state_preserved"]:
        if key not in decision:
            failures.append("DECISION_MISSING_FIELD")

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
        print("124 checker failed:")
        for failure in sorted(set(failures)):
            print(f" - {failure}")
        return 1
    print("124 checker passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
