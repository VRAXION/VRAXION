#!/usr/bin/env python3
"""Static/artifact checker for STABLE_LOOP_PHASE_LOCK_094B."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_094b_chat_sft_free_generation_gap_analysis/smoke"

REQUIRED_DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_094B_CHAT_SFT_FREE_GENERATION_GAP_ANALYSIS_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_094B_CHAT_SFT_FREE_GENERATION_GAP_ANALYSIS_RESULT.md",
]
REQUIRED_SOURCE = [
    "scripts/probes/run_stable_loop_phase_lock_094b_chat_sft_free_generation_gap_analysis.py",
    "scripts/probes/run_stable_loop_phase_lock_094b_chat_sft_free_generation_gap_analysis_check.py",
]
ALLOWED_MUTATIONS = set(REQUIRED_DOCS + REQUIRED_SOURCE)

REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "analysis_config.json",
    "upstream_094_manifest.json",
    "checkpoint_integrity_manifest.json",
    "eval_row_manifest.json",
    "decode_policy_matrix.json",
    "decode_policy_results.jsonl",
    "ranked_vs_generated_gap.json",
    "rollout_drift_analysis.json",
    "prefix_forcing_analysis.json",
    "stop_condition_analysis.json",
    "prompt_format_analysis.json",
    "failure_mode_classification.json",
    "human_readable_samples.jsonl",
    "failure_case_samples.jsonl",
    "summary.json",
    "report.md",
]

REQUIRED_TERMS = [
    "STABLE_LOOP_PHASE_LOCK_094B_CHAT_SFT_FREE_GENERATION_GAP_ANALYSIS",
    "analysis only",
    "no model capability improved",
    "not GPT-like assistant readiness",
    "not open-domain assistant",
    "not production chat",
    "not deployment",
    "not public release",
    "not safety alignment",
    "CHAT_SFT_FREE_GENERATION_GAP_ANALYSIS_POSITIVE",
    "UPSTREAM_094_SFT_POC_VERIFIED",
    "RANKED_GENERATED_GAP_CONFIRMED",
    "DECODE_POLICY_MATRIX_RECORDED",
    "ROLLOUT_DRIFT_ANALYZED",
    "PREFIX_FORCING_ANALYZED",
    "STOP_CONDITION_ANALYZED",
    "FAILURE_MODE_CLASSIFIED",
    "NO_TRAINING_PERFORMED",
    "CHECKPOINTS_UNCHANGED",
    "GPT_LIKE_READINESS_NOT_CLAIMED",
    "TRAINING_SIDE_EFFECT_DETECTED",
    "CHECKPOINT_MUTATION_DETECTED",
    "RUNTIME_SURFACE_MUTATION_DETECTED",
    "UPSTREAM_094_ARTIFACT_MISSING",
    "UPSTREAM_094_NOT_POSITIVE",
    "DECODE_POLICY_EVAL_ROW_MISMATCH",
    "DECODE_POLICY_MATRIX_MISSING",
    "RANKED_GENERATED_GAP_MISSING",
    "FAILURE_MODE_CLASSIFICATION_MISSING",
    "HUMAN_SAMPLE_REPORT_MISSING",
    "LLM_JUDGE_USED",
    "ORACLE_SHORTCUT_DETECTED",
    "ROOT_LICENSE_CHANGED",
    "GPT_LIKE_READINESS_FALSE_CLAIM",
    "PRODUCTION_CHAT_CLAIM_DETECTED",
    "PUBLIC_RELEASE_CLAIM_DETECTED",
    "095_CHAT_DECODER_GENERATION_REPAIR",
    "095B_CHAT_SFT_DATA_AND_ROLLOUT_REPAIR",
]

REQUIRED_POLICIES = [
    "greedy",
    "top_k_1",
    "top_k_4_temp_0.4",
    "top_k_8_temp_0.6",
    "top_k_24_temp_0.7",
    "top_k_24_temp_0.85",
    "nucleus_p_0.9_temp_0.7",
    "expected-prefix-forced first 8 bytes",
    "expected-prefix-forced first 16 bytes",
    "expected-prefix-forced first 32 bytes",
]

NEGATION_MARKERS = ["not ", "no ", "does not ", "do not ", "without ", "false", "forbidden", "forbid", "reject", "rejected", "only"]
FORBIDDEN_CLAIMS = {
    "GPT_LIKE_READINESS_FALSE_CLAIM": ["GPT-like assistant readiness"],
    "OPEN_DOMAIN_CLAIM_DETECTED": ["open-domain assistant"],
    "PRODUCTION_CHAT_CLAIM_DETECTED": ["production chat"],
    "PUBLIC_RELEASE_CLAIM_DETECTED": ["public release"],
    "SAFETY_ALIGNMENT_CLAIM_DETECTED": ["safety alignment"],
    "DEPLOYMENT_CLAIM_DETECTED": ["deployment"],
}
GENERATED_PATH_PARTS = ["target/", "node_modules/", ".svelte-kit/"]
GENERATED_SUFFIXES = [".exe", ".dll", ".so", ".dylib", ".zip", ".tar", ".tar.gz", ".tgz", ".7z", ".ckpt", ".bin", ".pt"]


def git_status(paths: list[str] | None = None) -> str:
    cmd = ["git", "status", "--short"]
    if paths:
        cmd.extend(["--", *paths])
    result = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
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
        window = lowered[max(0, match.start() - 140) : match.start()]
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
        if path == "LICENSE":
            return True
        if path.startswith("instnct-core/"):
            return True
        if path.startswith("tools/instnct_service_alpha/") or path.startswith("tools/instnct_deploy/"):
            return True
        if path.startswith("docs/product/") or path.startswith("docs/releases/"):
            return True
        if path.startswith("sdk/") or path.startswith("packages/"):
            return True
    return False


def generated_artifact_staged() -> bool:
    for path in changed_paths():
        if path in ALLOWED_MUTATIONS:
            continue
        if any(part in path for part in GENERATED_PATH_PARTS):
            return True
        if any(path.endswith(suffix) for suffix in GENERATED_SUFFIXES):
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
    verdicts = set(summary.get("verdicts", []))
    metrics = summary.get("metrics", {})
    if "CHAT_SFT_FREE_GENERATION_GAP_ANALYSIS_POSITIVE" not in verdicts:
        failures.append("CHAT_SFT_FREE_GENERATION_GAP_ANALYSIS_POSITIVE_MISSING")
    flags = {
        "analysis_completed": True,
        "no_training_performed": True,
        "source_093_checkpoint_unchanged": True,
        "target_094_checkpoint_unchanged": True,
        "all_decode_policies_same_eval_rows": True,
        "failure_mode_classification_present": True,
        "human_samples_present": True,
        "llm_judge_used": False,
        "prediction_oracle_used": False,
    }
    for key, expected in flags.items():
        if metrics.get(key) is not expected:
            failures.append(f"METRIC_FLAG_MISMATCH:{key}")
    if metrics.get("optimizer_step_count") != 0:
        failures.append("TRAINING_SIDE_EFFECT_DETECTED")
    matrix = load_json(SMOKE_ROOT / "decode_policy_matrix.json")
    names = [row["name"] for row in matrix.get("policies", [])]
    for policy in REQUIRED_POLICIES:
        if policy not in names:
            failures.append(f"DECODE_POLICY_MATRIX_MISSING:{policy}")
    decode_rows = read_jsonl(SMOKE_ROOT / "decode_policy_results.jsonl")
    if len(decode_rows) != len(REQUIRED_POLICIES):
        failures.append("DECODE_POLICY_RESULTS_COUNT_MISMATCH")
    hashes = {row.get("eval_row_hash") for row in decode_rows}
    counts = {row.get("eval_row_count") for row in decode_rows}
    if len(hashes) != 1 or len(counts) != 1:
        failures.append("DECODE_POLICY_EVAL_ROW_MISMATCH")
    gap = load_json(SMOKE_ROOT / "ranked_vs_generated_gap.json")
    for key in ["expected_response_loss", "generated_response_loss", "best_non_expected_response_loss", "rank_margin", "gold_prefix_survival_rate", "free_rollout_drift_rate", "gap_after_prefix_forcing"]:
        if key not in gap:
            failures.append(f"RANKED_GENERATED_GAP_MISSING:{key}")
    classification = load_json(SMOKE_ROOT / "failure_mode_classification.json")
    if not classification.get("primary_failure_mode") or not classification.get("recommended_next_milestone"):
        failures.append("FAILURE_MODE_CLASSIFICATION_MISSING")
    if classification.get("recommended_next_milestone") not in {"095_CHAT_DECODER_GENERATION_REPAIR", "095B_CHAT_SFT_DATA_AND_ROLLOUT_REPAIR"}:
        failures.append("FAILURE_MODE_CLASSIFICATION_MISSING:next")
    samples = read_jsonl(SMOKE_ROOT / "human_readable_samples.jsonl")
    kinds = {row.get("sample_kind") for row in samples}
    for kind in ["ranked expected response", "baseline 094 generation", "best decode policy generation", "prefix-forced generation"]:
        if kind not in kinds:
            failures.append(f"HUMAN_SAMPLE_REPORT_MISSING:{kind}")
    report_text = (SMOKE_ROOT / "report.md").read_text(encoding="utf-8")
    failures.extend(find_false_claims(report_text))
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Check STABLE_LOOP_PHASE_LOCK_094B artifacts")
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
    if generated_artifact_staged():
        failures.append("GENERATED_ARTIFACT_STAGED")
    if SMOKE_ROOT.exists():
        failures.extend(check_artifacts())
    else:
        failures.append("SMOKE_ROOT_MISSING")
    if failures:
        for failure in failures:
            print(failure)
        return 1
    print("STABLE_LOOP_PHASE_LOCK_094B_CHAT_SFT_FREE_GENERATION_GAP_ANALYSIS_CHECK_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
