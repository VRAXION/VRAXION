#!/usr/bin/env python3
"""Static/artifact checker for STABLE_LOOP_PHASE_LOCK_101."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_101_fresh_assistant_eval_and_raw_decoder_frontier_map/smoke"
REQUIRED_DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_101_FRESH_ASSISTANT_EVAL_AND_RAW_DECODER_FRONTIER_MAP_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_101_FRESH_ASSISTANT_EVAL_AND_RAW_DECODER_FRONTIER_MAP_RESULT.md",
]
REQUIRED_SOURCE = [
    "scripts/probes/run_stable_loop_phase_lock_101_fresh_assistant_eval_and_raw_decoder_frontier_map.py",
    "scripts/probes/run_stable_loop_phase_lock_101_fresh_assistant_eval_and_raw_decoder_frontier_map_check.py",
]
ALLOWED_MUTATIONS = set(REQUIRED_DOCS + REQUIRED_SOURCE)
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "eval_config.json",
    "upstream_100_manifest.json",
    "checkpoint_integrity_manifest.json",
    "eval_row_manifest.json",
    "decode_policy_matrix.json",
    "fresh_assistant_eval_dataset.jsonl",
    "raw_generation_results.jsonl",
    "decoder_assisted_results.jsonl",
    "ranked_scoring_results.jsonl",
    "prefix_forcing_diagnostics.jsonl",
    "family_metrics.json",
    "mode_comparison.json",
    "raw_vs_decoder_gap.json",
    "drift_analysis.json",
    "hungarian_english_report.json",
    "multi_turn_report.json",
    "refusal_boundary_report.json",
    "retention_report.json",
    "collapse_metrics.json",
    "decision_recommendation.json",
    "failure_case_samples.jsonl",
    "human_readable_samples.jsonl",
    "summary.json",
    "report.md",
]
REQUIRED_TERMS = [
    "STABLE_LOOP_PHASE_LOCK_101_FRESH_ASSISTANT_EVAL_AND_RAW_DECODER_FRONTIER_MAP",
    "fresh assistant frontier mapping",
    "eval-only",
    "no model capability improved",
    "not GPT-like assistant readiness",
    "not open-domain assistant readiness",
    "not production chat",
    "not public API",
    "not deployment",
    "not safety alignment",
    "FRESH_ASSISTANT_FRONTIER_MAP_POSITIVE",
    "UPSTREAM_100_CAPABILITY_SCALE_VERIFIED",
    "RAW_VS_DECODER_GAP_RECORDED",
    "FAMILY_FAILURE_MAP_WRITTEN",
    "DECISION_RECOMMENDATION_WRITTEN",
    "TRAINING_SIDE_EFFECT_DETECTED",
    "CHECKPOINT_MUTATION_DETECTED",
    "BOUNDED_RELEASE_MUTATION_DETECTED",
    "EVAL_ROW_MISMATCH",
    "DIAGNOSTIC_MODE_MISCOUNTED_AS_FREE_GENERATION",
    "DECODE_POLICY_CHERRY_PICKING_DETECTED",
    "FAMILY_FAILURE_MAP_INCOMPLETE",
    "RETENTION_REGRESSION_DETECTED",
    "GPT_LIKE_READINESS_FALSE_CLAIM",
    "PRODUCTION_CHAT_CLAIM_DETECTED",
    "102_DECODER_POLICY_AND_ROLLOUT_REPAIR",
    "102B_ASSISTANT_REPRESENTATION_OR_SFT_REPAIR",
    "HUNGARIAN_SFT_AND_EVAL_TRACK_LATER",
]
EVAL_MODES = [
    "RAW_GREEDY_GENERATION",
    "RAW_SAMPLED_GENERATION_LOW_TEMP",
    "DECODER_ASSISTED_GENERATION",
    "DECODER_ASSISTED_STRICT_BOUNDARY",
    "PREFIX_FORCED_DIAGNOSTIC",
    "RANKED_RESPONSE_SCORING",
]
EVAL_FAMILIES = [
    "FRESH_SHORT_INSTRUCTION",
    "FRESH_SHORT_EXPLANATION",
    "FRESH_OPEN_DOMAIN_SIMPLE_QA",
    "FRESH_OPEN_DOMAIN_UNSUPPORTED",
    "FRESH_MULTI_TURN_CONTEXT_CARRY",
    "FRESH_BOUNDARY_REFUSAL",
    "FRESH_PROMPT_INJECTION",
    "FRESH_HUNGARIAN_BASIC_CHAT",
    "FRESH_ENGLISH_BASIC_CHAT",
    "FRESH_ACTIVE_SLOT_BINDING",
    "FRESH_STALE_DISTRACTOR_SUPPRESSION",
    "FRESH_ANTI_REPETITION",
    "FINITE_LABEL_ANCHORROUTE_RETENTION",
    "BOUNDED_RELEASE_RETENTION",
]
NEGATION_MARKERS = ["not ", "no ", "does not ", "do not ", "without ", "false", "forbidden", "forbid", "reject", "rejected", "only", "cannot"]
FORBIDDEN_CLAIMS = {
    "GPT_LIKE_READINESS_FALSE_CLAIM": ["GPT-like assistant readiness"],
    "OPEN_DOMAIN_ASSISTANT_CLAIM_DETECTED": ["open-domain assistant readiness"],
    "PRODUCTION_CHAT_CLAIM_DETECTED": ["production chat"],
    "PUBLIC_API_CLAIM_DETECTED": ["public API"],
    "DEPLOYMENT_CLAIM_DETECTED": ["deployment readiness"],
    "PUBLIC_RELEASE_CLAIM_DETECTED": ["public release"],
    "SAFETY_ALIGNMENT_CLAIM_DETECTED": ["safety alignment"],
}


def git_status() -> str:
    result = subprocess.run(["git", "status", "--short"], cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    return result.stdout.rstrip("\n")


def changed_paths() -> list[str]:
    paths: list[str] = []
    for line in git_status().splitlines():
        if not line.strip():
            continue
        path = line[3:].replace("\\", "/")
        paths.append(path)
    return paths


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def claim_is_negated(text: str, phrase: str) -> bool:
    lowered = text.lower()
    phrase_lower = phrase.lower()
    for match in re.finditer(re.escape(phrase_lower), lowered):
        window = lowered[max(0, match.start() - 160) : match.start()]
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
        if path in ALLOWED_MUTATIONS:
            continue
        if path.startswith("target/"):
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


def check_same_rows() -> list[str]:
    failures: list[str] = []
    manifest = load_json(SMOKE_ROOT / "eval_row_manifest.json")
    modes = manifest.get("modes", [])
    if len(modes) != len(EVAL_MODES):
        failures.append("DECODE_POLICY_MATRIX_MISSING")
        return failures
    if sorted(item.get("mode") for item in modes) != sorted(EVAL_MODES):
        failures.append("DECODE_POLICY_MATRIX_MISSING")
    row_hashes = {item.get("eval_row_hash") for item in modes}
    prompt_hashes = {item.get("eval_prompt_hash") for item in modes}
    counts = {item.get("eval_count") for item in modes}
    dataset_hashes = {item.get("eval_dataset_sha256") for item in modes}
    if len(row_hashes) != 1 or len(prompt_hashes) != 1 or len(counts) != 1 or len(dataset_hashes) != 1:
        failures.append("EVAL_ROW_MISMATCH")
    matrix = load_json(SMOKE_ROOT / "decode_policy_matrix.json")
    if matrix.get("fixed_before_eval") is not True or matrix.get("post_hoc_tuning_used") is not False:
        failures.append("DECODE_POLICY_CHERRY_PICKING_DETECTED")
    return failures


def check_artifacts() -> list[str]:
    failures: list[str] = []
    for rel in REQUIRED_ARTIFACTS:
        if not (SMOKE_ROOT / rel).exists():
            failures.append(f"MISSING_ARTIFACT:{rel}")
    if failures:
        return failures
    failures.extend(check_same_rows())
    summary = load_json(SMOKE_ROOT / "summary.json")
    verdicts = set(summary.get("verdicts", []))
    metrics = summary.get("metrics", {})
    if "FRESH_ASSISTANT_FRONTIER_MAP_POSITIVE" not in verdicts:
        failures.append("FRESH_ASSISTANT_FRONTIER_MAP_POSITIVE_MISSING")
    for key, expected in {
        "train_step_count": 0,
        "optimizer_step_count": 0,
        "checkpoint_hash_unchanged": True,
        "bounded_release_artifact_unchanged": True,
        "no_training_performed": True,
        "raw_vs_decoder_gap_recorded": True,
        "failure_modes_classified": True,
        "decision_recommendation_written": True,
        "all_decode_policies_same_eval_rows": True,
    }.items():
        if metrics.get(key) != expected:
            failures.append(f"EVAL_ONLY_OR_MAPPING_GATE_FAIL:{key}")
    if metrics.get("bounded_chat_slot_binding_accuracy", 0.0) < 0.80 or metrics.get("finite_label_anchorroute_retention_accuracy", 0.0) < 0.90 or metrics.get("bounded_release_retention_pass") is not True:
        failures.append("RETENTION_REGRESSION_DETECTED")
    for key in ["gpt_like_claim_count", "production_chat_claim_count", "public_api_claim_count", "safety_alignment_claim_count", "open_domain_answer_leak_count"]:
        if metrics.get(key, 0) != 0:
            failures.append("OPEN_DOMAIN_ANSWER_LEAK_DETECTED" if key == "open_domain_answer_leak_count" else "GPT_LIKE_READINESS_FALSE_CLAIM")
    family_metrics = load_json(SMOKE_ROOT / "family_metrics.json")
    families = family_metrics.get("families", {})
    if sorted(families) != sorted(EVAL_FAMILIES):
        failures.append("FAMILY_FAILURE_MAP_INCOMPLETE")
    for family in EVAL_FAMILIES:
        item = families.get(family, {})
        for key in ["raw_accuracy", "decoder_assisted_accuracy", "ranked_accuracy", "gap_raw_to_decoder", "gap_raw_to_ranked"]:
            if key not in item:
                failures.append(f"FAMILY_METRIC_MISSING:{family}:{key}")
    gap = load_json(SMOKE_ROOT / "raw_vs_decoder_gap.json")
    if gap.get("diagnostic_modes_counted_as_free_generation") is not False:
        failures.append("DIAGNOSTIC_MODE_MISCOUNTED_AS_FREE_GENERATION")
    decision = load_json(SMOKE_ROOT / "decision_recommendation.json")
    if not decision.get("primary_next_milestone") or decision.get("mechanically_derived") is not True:
        failures.append("DECISION_RECOMMENDATION_MISSING")
    if len(read_jsonl(SMOKE_ROOT / "human_readable_samples.jsonl")) < len(EVAL_MODES) * 6:
        failures.append("HUMAN_SAMPLE_REPORT_MISSING")
    raw_rows = read_jsonl(SMOKE_ROOT / "raw_generation_results.jsonl")
    if raw_rows and any("failure_label" not in row for row in raw_rows):
        failures.append("FAILURE_MODE_CLASSIFICATION_MISSING")
    if len(read_jsonl(SMOKE_ROOT / "progress.jsonl")) < 8:
        failures.append("PARTIAL_WRITEOUT_INSUFFICIENT")
    failures.extend(find_false_claims((SMOKE_ROOT / "report.md").read_text(encoding="utf-8")))
    for path in changed_paths():
        if path.startswith("target/"):
            failures.append("GENERATED_ARTIFACT_STAGED")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Check STABLE_LOOP_PHASE_LOCK_101 artifacts")
    parser.add_argument("--check-only", action="store_true")
    _args = parser.parse_args()
    failures: list[str] = []
    missing, files = read_files()
    failures.extend(f"MISSING_FILE:{item}" for item in missing)
    combined = "\n".join(files.values())
    for term in REQUIRED_TERMS:
        if term not in combined:
            failures.append(f"TERM_MISSING:{term}")
    failures.extend(find_false_claims(combined))
    if runtime_surface_mutation_detected():
        failures.append("RUNTIME_SURFACE_MUTATION_DETECTED")
    if "LICENSE" in changed_paths():
        failures.append("ROOT_LICENSE_CHANGED")
    if SMOKE_ROOT.exists():
        failures.extend(check_artifacts())
    else:
        failures.append("SMOKE_ROOT_MISSING")
    if failures:
        for failure in failures:
            print(failure)
        return 1
    print("STABLE_LOOP_PHASE_LOCK_101_FRESH_ASSISTANT_EVAL_AND_RAW_DECODER_FRONTIER_MAP_CHECK_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
