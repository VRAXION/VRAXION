#!/usr/bin/env python3
"""Static/artifact checker for STABLE_LOOP_PHASE_LOCK_094."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_094_open_vocab_chat_sft_mix_poc/smoke"

REQUIRED_DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_094_OPEN_VOCAB_CHAT_SFT_MIX_POC_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_094_OPEN_VOCAB_CHAT_SFT_MIX_POC_RESULT.md",
]
REQUIRED_SOURCE = [
    "scripts/probes/run_stable_loop_phase_lock_094_open_vocab_chat_sft_mix_poc.py",
    "scripts/probes/run_stable_loop_phase_lock_094_open_vocab_chat_sft_mix_poc_check.py",
]
ALLOWED_MUTATIONS = set(REQUIRED_DOCS + REQUIRED_SOURCE)

REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "training_config.json",
    "upstream_093_manifest.json",
    "source_checkpoint_manifest.json",
    "sft_dataset_manifest.json",
    "fineweb_replay_manifest.json",
    "sft_train_examples_sample.jsonl",
    "sft_eval_examples_sample.jsonl",
    "training_metrics.jsonl",
    "checkpoint_manifest.json",
    "checkpoint_hashes.json",
    "pre_sft_eval_metrics.json",
    "post_sft_eval_metrics.json",
    "chat_sft_metrics.json",
    "fineweb_retention_metrics.json",
    "bounded_retention_metrics.json",
    "collapse_metrics.json",
    "control_comparison.json",
    "generation_samples.jsonl",
    "human_readable_samples.jsonl",
    "failure_case_samples.jsonl",
    "summary.json",
    "report.md",
]

REQUIRED_TERMS = [
    "STABLE_LOOP_PHASE_LOCK_094_OPEN_VOCAB_CHAT_SFT_MIX_POC",
    "runner-local PyTorch byte-LM",
    "Chat SFT mix PoC",
    "not GPT-like assistant readiness",
    "not open-domain assistant readiness",
    "not production chat",
    "not public release",
    "not deployment",
    "not safety alignment",
    "not proof INSTNCT/AnchorRoute is an open-domain LM winner",
    "OPEN_VOCAB_CHAT_SFT_MIX_POC_POSITIVE",
    "UPSTREAM_093_FINEWEB_MARGIN_VERIFIED",
    "BEST_093_CHECKPOINT_LOADED_READ_ONLY",
    "CHAT_SFT_DATASET_BUILT",
    "CHAT_SFT_TRAINING_COMPLETED",
    "SFT_OBJECTIVE_LEARNED",
    "CHAT_FORMAT_SIGNAL_IMPROVES",
    "FINEWEB_RETENTION_WITHIN_LIMITS",
    "BOUNDED_CHAT_RETENTION_PASSES",
    "FINITE_LABEL_ANCHORROUTE_RETENTION_PASSES",
    "COLLAPSE_REJECTED",
    "SOURCE_093_CHECKPOINT_UNCHANGED",
    "PRODUCTION_CHAT_NOT_CLAIMED",
    "GPT_LIKE_READINESS_NOT_CLAIMED",
    "OPEN_VOCAB_CHAT_SFT_MIX_POC_FAILS",
    "BEST_SEED_SELECTION_UNDOCUMENTED",
    "SOURCE_093_CHECKPOINT_MUTATION_DETECTED",
    "NO_ACTUAL_SFT_UPDATE_DETECTED",
    "TRAIN_EVAL_LEAKAGE_DETECTED",
    "DATASET_MIX_CONTAMINATION_DETECTED",
    "BASELINE_EVAL_MISMATCH",
    "CHAT_FORMAT_SIGNAL_NOT_IMPROVED",
    "COPY_OR_STATIC_CONTROL_UNEXPECTED_PASS",
    "CHAT_SFT_TEMPLATE_COPY_REGRESSION_DETECTED",
    "FINEWEB_RETENTION_REGRESSION_DETECTED",
    "BOUNDED_CHAT_RETENTION_REGRESSION_DETECTED",
    "FINITE_LABEL_RETENTION_REGRESSION_DETECTED",
    "UNSUPPORTED_REFUSAL_REGRESSION_DETECTED",
    "EMPTY_OUTPUT_COLLAPSE_DETECTED",
    "STATIC_RESPONSE_COLLAPSE_DETECTED",
    "REPETITION_COLLAPSE_DETECTED",
    "HUMAN_SAMPLE_REPORT_MISSING",
    "LLM_JUDGE_USED",
    "CHAT_EVAL_RUBRIC_MISSING",
    "FINEWEB_SOURCE_MUTATION_DETECTED",
    "RUNTIME_SURFACE_MUTATION_DETECTED",
    "ROOT_LICENSE_CHANGED",
    "GPT_LIKE_READINESS_FALSE_CLAIM",
    "PRODUCTION_CHAT_CLAIM_DETECTED",
    "PUBLIC_RELEASE_CLAIM_DETECTED",
    "ARCHITECTURE_WINNER_FALSE_CLAIM",
    "095_FRESH_OPEN_DOMAIN_CHAT_EVAL",
    "094B_CHAT_SFT_RETENTION_FAILURE_ANALYSIS",
    "094B_CHAT_SFT_SIGNAL_FAILURE_ANALYSIS",
    "094B_CHAT_SFT_GENERATION_FAILURE_ANALYSIS",
]

FORBIDDEN_CLAIMS = {
    "GPT_LIKE_READINESS_FALSE_CLAIM": ["GPT-like assistant readiness"],
    "OPEN_DOMAIN_CLAIM_DETECTED": ["open-domain assistant readiness", "proven as open-domain LM winner"],
    "PRODUCTION_CHAT_CLAIM_DETECTED": ["production chat"],
    "PUBLIC_RELEASE_CLAIM_DETECTED": ["public release"],
    "SAFETY_ALIGNMENT_CLAIM_DETECTED": ["safety alignment"],
    "DEPLOYMENT_CLAIM_DETECTED": ["deployment"],
}
NEGATION_MARKERS = ["not ", "no ", "does not ", "do not ", "without ", "false", "forbidden", "forbid", "reject", "rejected", "only"]
GENERATED_PATH_PARTS = ["target/", "node_modules/", ".svelte-kit/"]
GENERATED_SUFFIXES = [".exe", ".dll", ".so", ".dylib", ".zip", ".tar", ".tar.gz", ".tgz", ".7z", ".ckpt", ".bin", ".pt"]


def git_status(paths: list[str] | None = None) -> str:
    cmd = ["git", "status", "--short"]
    if paths:
        cmd.extend(["--", *paths])
    result = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    return result.stdout.rstrip("\n")


def changed_paths() -> list[str]:
    paths = []
    for line in git_status().splitlines():
        if not line.strip():
            continue
        paths.append(line[3:].replace("\\", "/"))
    return paths


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
    if "OPEN_VOCAB_CHAT_SFT_MIX_POC_POSITIVE" not in verdicts:
        failures.append("OPEN_VOCAB_CHAT_SFT_MIX_POC_POSITIVE_MISSING")
    required_flags = {
        "source_093_checkpoint_unchanged": True,
        "target_sft_checkpoint_changed": True,
        "fineweb_source_hash_unchanged": True,
        "all_control_eval_rows_match": True,
        "chat_eval_rubric_present": True,
        "llm_judge_used": False,
        "architecture_winner_for_open_vocab_claimed": False,
        "packaged_bounded_winner_trained": False,
    }
    for key, expected in required_flags.items():
        if metrics.get(key) is not expected:
            failures.append(f"METRIC_FLAG_MISMATCH:{key}")
    numeric_gates = [
        ("sft_train_step_count", 1, "gte"),
        ("sft_eval_loss_delta", 0.15, "gte"),
        ("post_sft_prompt_response_accuracy", 0.70, "gte"),
        ("exact_sft_train_response_copy_rate", 0.25, "lte"),
        ("response_skeleton_reuse_rate", 0.50, "lte"),
        ("novel_response_rate", 0.50, "gte"),
        ("fineweb_eval_loss_regression", 0.35, "lte"),
        ("fineweb_next_byte_accuracy_drop", 0.08, "lte"),
        ("bounded_chat_slot_binding_accuracy", 0.80, "gte"),
        ("finite_label_anchorroute_retention_accuracy", 0.90, "gte"),
        ("unsupported_refusal_accuracy", 0.80, "gte"),
        ("nonempty_generation_rate", 0.98, "gte"),
        ("utf8_valid_generation_rate", 0.80, "gte"),
        ("empty_output_rate", 0.02, "lte"),
        ("static_output_rate", 0.15, "lte"),
        ("repetition_rate", 0.25, "lte"),
        ("copy_prompt_rate", 0.20, "lte"),
    ]
    for key, threshold, op in numeric_gates:
        value = metrics.get(key)
        if value is None:
            failures.append(f"METRIC_MISSING:{key}")
            continue
        if op == "gte" and not float(value) >= threshold:
            failures.append(f"METRIC_GATE_FAIL:{key}")
        if op == "lte" and not float(value) <= threshold:
            failures.append(f"METRIC_GATE_FAIL:{key}")
    dataset = load_json(SMOKE_ROOT / "sft_dataset_manifest.json")
    if dataset.get("sft_train_eval_exact_prompt_overlap_count") != 0 or dataset.get("sft_train_eval_exact_response_overlap_count") != 0 or dataset.get("max_sft_train_eval_prompt_jaccard", 1.0) >= 0.90:
        failures.append("TRAIN_EVAL_LEAKAGE_DETECTED")
    replay = load_json(SMOKE_ROOT / "fineweb_replay_manifest.json")
    if replay.get("fineweb_rows_in_sft_eval") != 0 or replay.get("bounded_rows_in_fineweb_eval") != 0:
        failures.append("DATASET_MIX_CONTAMINATION_DETECTED")
    source = load_json(SMOKE_ROOT / "source_checkpoint_manifest.json")
    if source.get("source_093_checkpoint_unchanged") is not True:
        failures.append("SOURCE_093_CHECKPOINT_MUTATION_DETECTED")
    controls = load_json(SMOKE_ROOT / "control_comparison.json")
    if controls.get("all_control_eval_rows_match") is not True:
        failures.append("BASELINE_EVAL_MISMATCH")
    sample_rows = read_jsonl(SMOKE_ROOT / "human_readable_samples.jsonl")
    families = {row.get("eval_family") for row in sample_rows}
    arms = {row.get("arm") for row in sample_rows}
    for family in ["short instruction", "simple dialogue", "bounded active slot", "context carry", "unsupported open-domain refusal", "boundary/injection refusal"]:
        if family not in families:
            failures.append(f"HUMAN_SAMPLE_REPORT_MISSING:{family}")
    if "PRE_SFT_093_BEST_CHECKPOINT" not in arms or "POST_SFT_MIX_CHECKPOINT" not in arms:
        failures.append("HUMAN_SAMPLE_REPORT_MISSING:paired_arms")
    report_text = (SMOKE_ROOT / "report.md").read_text(encoding="utf-8")
    failures.extend(find_false_claims(report_text))
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Check STABLE_LOOP_PHASE_LOCK_094 artifacts")
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
    print("STABLE_LOOP_PHASE_LOCK_094_OPEN_VOCAB_CHAT_SFT_MIX_POC_CHECK_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
