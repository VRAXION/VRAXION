#!/usr/bin/env python3
"""Static/artifact checker for STABLE_LOOP_PHASE_LOCK_093."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_093_open_vocab_fineweb_margin_scale_confirm/smoke"
DEFAULT_FINEWEB_SOURCE = Path(r"S:\AI\MESSY TRAINING DATA - INPUT ONLY\Fineweb edu 10B\fineweb_edu_30m.txt")

REQUIRED_DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_093_OPEN_VOCAB_FINEWEB_MARGIN_SCALE_CONFIRM_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_093_OPEN_VOCAB_FINEWEB_MARGIN_SCALE_CONFIRM_RESULT.md",
]
REQUIRED_SOURCE = [
    "scripts/probes/run_stable_loop_phase_lock_093_open_vocab_fineweb_margin_scale_confirm.py",
    "scripts/probes/run_stable_loop_phase_lock_093_open_vocab_fineweb_margin_scale_confirm_check.py",
]
ALLOWED_MUTATIONS = set(REQUIRED_DOCS + REQUIRED_SOURCE)

REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "training_config.json",
    "upstream_092_manifest.json",
    "fineweb_source_manifest.json",
    "dataset_manifest.json",
    "tokenizer_manifest.json",
    "seed_run_manifest.json",
    "train_examples_sample.jsonl",
    "eval_examples_sample.jsonl",
    "training_metrics.jsonl",
    "checkpoint_manifest.json",
    "checkpoint_hashes.json",
    "generation_samples.jsonl",
    "human_readable_samples.jsonl",
    "lm_metrics.json",
    "fineweb_generation_metrics.json",
    "bounded_retention_metrics.json",
    "collapse_metrics.json",
    "leakage_metrics.json",
    "arm_comparison.json",
    "control_delta_report.json",
    "multi_seed_aggregate.json",
    "failure_case_samples.jsonl",
    "summary.json",
    "report.md",
]

EXACT_COMMANDS = [
    "python -m py_compile scripts/probes/run_stable_loop_phase_lock_093_open_vocab_fineweb_margin_scale_confirm.py",
    'python scripts/probes/run_stable_loop_phase_lock_093_open_vocab_fineweb_margin_scale_confirm.py --out target/pilot_wave/stable_loop_phase_lock_093_open_vocab_fineweb_margin_scale_confirm/smoke --upstream-092-root target/pilot_wave/stable_loop_phase_lock_092_open_vocab_fineweb_slice_confirm/smoke --fineweb-source "S:\\AI\\MESSY TRAINING DATA - INPUT ONLY\\Fineweb edu 10B\\fineweb_edu_30m.txt" --seeds 2026,2027,2028 --train-tokens 1000000 --eval-tokens 200000 --seq-len 128 --batch-size 32 --steps 3000 --heartbeat-sec 20',
    "python -m py_compile scripts/probes/run_stable_loop_phase_lock_093_open_vocab_fineweb_margin_scale_confirm_check.py",
    "python scripts/probes/run_stable_loop_phase_lock_093_open_vocab_fineweb_margin_scale_confirm_check.py --check-only",
    "python scripts/probes/run_stable_loop_phase_lock_092_open_vocab_fineweb_slice_confirm_check.py --check-only",
    "python scripts/probes/run_stable_loop_phase_lock_091_open_vocab_chat_lm_foundation_check.py --check-only",
    "git diff --check",
]

REQUIRED_TERMS = [
    "STABLE_LOOP_PHASE_LOCK_093_OPEN_VOCAB_FINEWEB_MARGIN_SCALE_CONFIRM",
    "runner-local PyTorch byte-LM",
    "FineWeb margin/scale confirmation only",
    "not GPT-like assistant readiness",
    "not open-domain assistant readiness",
    "not production chat",
    "not public release",
    "not deployment",
    "not safety alignment",
    "not proof INSTNCT/AnchorRoute is open-domain LM winner",
    "OPEN_VOCAB_FINEWEB_BYTE_LM_MAIN",
    "CHAR_UNIGRAM_BASELINE",
    "CHAR_BIGRAM_BASELINE",
    "CHAR_TRIGRAM_BASELINE",
    "RANDOM_BYTE_CONTROL",
    "SHUFFLED_TARGET_CONTROL",
    "STATIC_OUTPUT_CONTROL",
    "COPY_PROMPT_CONTROL",
    "delta_vs_char_bigram_loss >= 0.03 on at least 2/3 seeds",
    "min_delta_vs_char_bigram_loss > 0.00",
    "mean_delta_vs_char_bigram_loss >= 0.03",
    "OPEN_VOCAB_FINEWEB_MARGIN_SCALE_CONFIRM_POSITIVE",
    "UPSTREAM_092_FINEWEB_CONFIRM_VERIFIED",
    "FINEWEB_SOURCE_IMMUTABILITY_PASSES",
    "BYTE_LEVEL_TOKENIZER_REUSED",
    "OPEN_VOCAB_NEXT_BYTE_TRAINING_MULTI_SEED_COMPLETED",
    "TOKEN_OBJECTIVE_LEARNED",
    "MAIN_BEATS_RANDOM_AND_SHUFFLED_CONTROLS",
    "CHAR_BIGRAM_MARGIN_PASSES",
    "COLLAPSE_REJECTED_ALL_SEEDS",
    "BOUNDED_CHAT_RETENTION_PASSES_ALL_SEEDS",
    "FINITE_LABEL_ANCHORROUTE_RETENTION_PASSES_ALL_SEEDS",
    "PACKAGED_WINNER_UNCHANGED",
    "PRODUCTION_CHAT_NOT_CLAIMED",
    "GPT_LIKE_READINESS_NOT_CLAIMED",
    "OPEN_VOCAB_FINEWEB_MARGIN_SCALE_CONFIRM_FAILS",
    "UPSTREAM_092_ARTIFACT_MISSING",
    "UPSTREAM_092_NOT_POSITIVE",
    "FINEWEB_SLICE_MISSING",
    "FINEWEB_SOURCE_MUTATION_DETECTED",
    "DATASET_MIX_CONTAMINATION_DETECTED",
    "TRAIN_EVAL_LEAKAGE_DETECTED",
    "BASELINE_EVAL_MISMATCH",
    "NO_ACTUAL_TRAINING_UPDATE_DETECTED",
    "TOKEN_OBJECTIVE_NOT_LEARNED",
    "CONTROL_DELTA_INSUFFICIENT",
    "OPEN_VOCAB_MARGIN_TOO_WEAK",
    "MULTI_SEED_OPEN_VOCAB_INSTABILITY_DETECTED",
    "OPEN_VOCAB_GENERATION_SMOKE_FAILS",
    "BOUNDED_CHAT_RETENTION_REGRESSION_DETECTED",
    "FINITE_LABEL_RETENTION_REGRESSION_DETECTED",
    "PACKAGED_CHECKPOINT_MUTATION_DETECTED",
    "TRAINING_SIDE_EFFECT_ON_PACKAGED_CHECKPOINT",
    "ORACLE_SHORTCUT_DETECTED",
    "LLM_JUDGE_USED",
    "RUNTIME_SURFACE_MUTATION_DETECTED",
    "ROOT_LICENSE_CHANGED",
    "GPT_LIKE_READINESS_FALSE_CLAIM",
    "PRODUCTION_CHAT_CLAIM_DETECTED",
    "PUBLIC_RELEASE_CLAIM_DETECTED",
    "ARCHITECTURE_WINNER_FALSE_CLAIM",
    "094_OPEN_VOCAB_CHAT_SFT_MIX_POC",
    "093B_OPEN_VOCAB_MARGIN_FAILURE_ANALYSIS",
]

NEGATION_MARKERS = ["not ", "no ", "does not ", "do not ", "without ", "false", "forbidden", "forbid", "reject", "rejected", "only"]
FORBIDDEN_CLAIMS = {
    "GPT_LIKE_READINESS_FALSE_CLAIM": ["GPT-like assistant readiness"],
    "OPEN_DOMAIN_CLAIM_DETECTED": ["open-domain assistant readiness", "proven as open-domain LM winner"],
    "PRODUCTION_CHAT_CLAIM_DETECTED": ["production chat"],
    "PUBLIC_RELEASE_CLAIM_DETECTED": ["public release"],
    "SAFETY_ALIGNMENT_CLAIM_DETECTED": ["safety alignment"],
    "DEPLOYMENT_CLAIM_DETECTED": ["deployment"],
}
PLACEHOLDERS = ["TODO", "TBD", "placeholder", "pending measured", "coming soon"]
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
        if any(part in path for part in GENERATED_PATH_PARTS):
            return True
        if any(path.endswith(suffix) for suffix in GENERATED_SUFFIXES):
            return True
    return False


def check_seed(seed: int) -> list[str]:
    failures: list[str] = []
    seed_root = SMOKE_ROOT / f"seed_{seed}"
    if not seed_root.exists():
        return [f"SEED_ARTIFACT_MISSING:{seed}"]
    for rel in [
        "dataset_manifest.json",
        "checkpoint_manifest.json",
        "checkpoint_hashes.json",
        "generation_samples.jsonl",
        "human_readable_samples.jsonl",
        "lm_metrics.json",
        "fineweb_generation_metrics.json",
        "bounded_retention_metrics.json",
        "collapse_metrics.json",
        "leakage_metrics.json",
        "arm_comparison.json",
        "control_delta_report.json",
    ]:
        if not (seed_root / rel).exists():
            failures.append(f"SEED_ARTIFACT_MISSING:{seed}:{rel}")
    if failures:
        return failures
    dataset = load_json(seed_root / "dataset_manifest.json")
    if dataset.get("bounded_retention_rows_in_lm_train") != 0 or dataset.get("bounded_retention_rows_in_lm_eval") != 0:
        failures.append(f"DATASET_MIX_CONTAMINATION_DETECTED:{seed}")
    if dataset.get("train_eval_exact_text_overlap_count") != 0 or dataset.get("max_train_eval_jaccard", 1.0) >= 0.90:
        failures.append(f"TRAIN_EVAL_LEAKAGE_DETECTED:{seed}")
    checkpoint = load_json(seed_root / "checkpoint_manifest.json")
    if checkpoint.get("train_step_count", 0) <= 0 or checkpoint.get("checkpoint_after_hash") == checkpoint.get("checkpoint_before_hash"):
        failures.append(f"NO_ACTUAL_TRAINING_UPDATE_DETECTED:{seed}")
    if not checkpoint.get("train_loss_final", 1e9) < checkpoint.get("train_loss_initial", -1e9):
        failures.append(f"TOKEN_OBJECTIVE_NOT_LEARNED:{seed}")
    arms = load_json(seed_root / "arm_comparison.json")
    expected = {"OPEN_VOCAB_FINEWEB_BYTE_LM_MAIN", "CHAR_UNIGRAM_BASELINE", "CHAR_BIGRAM_BASELINE", "CHAR_TRIGRAM_BASELINE", "RANDOM_BYTE_CONTROL", "SHUFFLED_TARGET_CONTROL", "STATIC_OUTPUT_CONTROL", "COPY_PROMPT_CONTROL"}
    present = {row.get("arm") for row in arms.get("arms", [])}
    if expected - present:
        failures.append(f"ARM_MISSING:{seed}:{','.join(sorted(expected - present))}")
    hashes = {(row.get("eval_row_hash"), row.get("eval_token_hash"), row.get("eval_token_count")) for row in arms.get("arms", [])}
    if len(hashes) != 1:
        failures.append(f"BASELINE_EVAL_MISMATCH:{seed}")
    deltas = load_json(seed_root / "control_delta_report.json")
    if deltas.get("delta_vs_shuffled_target_loss", -1.0) < 0.25 or deltas.get("delta_vs_random_accuracy", -1.0) < 0.10:
        failures.append(f"CONTROL_DELTA_INSUFFICIENT:{seed}")
    if deltas.get("delta_vs_char_bigram_loss", -1.0) <= 0.0:
        failures.append(f"OPEN_VOCAB_MARGIN_TOO_WEAK:{seed}")
    generation = load_json(seed_root / "fineweb_generation_metrics.json")
    if generation.get("nonempty_generation_rate", 0.0) < 0.98 or generation.get("utf8_valid_generation_rate", 0.0) < 0.80:
        failures.append(f"OPEN_VOCAB_GENERATION_SMOKE_FAILS:{seed}")
    collapse = load_json(seed_root / "collapse_metrics.json")
    if collapse.get("empty_output_rate", 1.0) > 0.02:
        failures.append(f"EMPTY_OUTPUT_COLLAPSE_DETECTED:{seed}")
    if collapse.get("static_output_rate", 1.0) > 0.15:
        failures.append(f"STATIC_RESPONSE_COLLAPSE_DETECTED:{seed}")
    if collapse.get("repetition_rate", 1.0) > 0.25:
        failures.append(f"REPETITION_COLLAPSE_DETECTED:{seed}")
    retention = load_json(seed_root / "bounded_retention_metrics.json")
    if not retention.get("packaged_winner_hash_unchanged") or not retention.get("no_training_on_packaged_checkpoint"):
        failures.append(f"PACKAGED_CHECKPOINT_MUTATION_DETECTED:{seed}")
    if retention.get("bounded_chat_slot_binding_accuracy", 0.0) < 0.80 or retention.get("unsupported_refusal_accuracy", 0.0) < 0.80:
        failures.append(f"BOUNDED_CHAT_RETENTION_REGRESSION_DETECTED:{seed}")
    if retention.get("finite_label_anchorroute_retention_accuracy", 0.0) < 0.90:
        failures.append(f"FINITE_LABEL_RETENTION_REGRESSION_DETECTED:{seed}")
    samples = read_jsonl(seed_root / "human_readable_samples.jsonl")
    families = {row.get("eval_family") for row in samples}
    for family in ["short English continuation", "unseen word continuation", "mixed punctuation continuation", "number/text continuation", "simple dialogue continuation", "unsupported-domain refusal continuation"]:
        if family not in families:
            failures.append(f"HUMAN_SAMPLE_REPORT_MISSING:{seed}:{family}")
    for field in ["seed", "arm", "prompt", "generated_text", "expected_behavior", "pass_fail", "utf8_valid", "nonempty", "repetition_flag", "copy_prompt_flag", "bounded_retention_flag", "short_diagnosis"]:
        if not all(field in row for row in samples):
            failures.append(f"HUMAN_SAMPLE_FIELD_MISSING:{seed}:{field}")
    return failures


def artifact_checks() -> list[str]:
    failures: list[str] = []
    for rel in REQUIRED_ARTIFACTS:
        if not (SMOKE_ROOT / rel).exists():
            failures.append(f"SMOKE_ARTIFACT_MISSING:{rel}")
    if failures:
        return failures
    summary = load_json(SMOKE_ROOT / "summary.json")
    metrics = summary.get("metrics", {})
    if summary.get("status") != "positive":
        failures.append("SMOKE_NOT_POSITIVE")
    if "OPEN_VOCAB_FINEWEB_MARGIN_SCALE_CONFIRM_POSITIVE" not in summary.get("verdicts", []):
        failures.append("POSITIVE_VERDICT_MISSING")
    for key in ["runner_local_pytorch_lm"]:
        if summary.get(key) is not True:
            failures.append(f"SUMMARY_FLAG_NOT_TRUE:{key}")
    for key in ["packaged_bounded_winner_trained", "architecture_winner_for_open_vocab_claimed", "prediction_oracle_used", "llm_judge_used", "production_chat_claimed", "public_release_claimed", "deployment_claimed", "safety_alignment_claimed"]:
        if summary.get(key) is not False:
            failures.append(f"SUMMARY_FLAG_NOT_FALSE:{key}")
    fineweb = load_json(SMOKE_ROOT / "fineweb_source_manifest.json")
    if not DEFAULT_FINEWEB_SOURCE.exists():
        failures.append("FINEWEB_SLICE_MISSING")
    if fineweb.get("fineweb_source_path") != str(DEFAULT_FINEWEB_SOURCE):
        failures.append("FINEWEB_SOURCE_PATH_MISMATCH")
    if not fineweb.get("fineweb_source_hash_unchanged"):
        failures.append("FINEWEB_SOURCE_MUTATION_DETECTED")
    aggregate = load_json(SMOKE_ROOT / "multi_seed_aggregate.json")
    if aggregate.get("seed_count") != 3:
        failures.append("SEED_COUNT_MISMATCH")
    if not aggregate.get("all_seed_base_gates_pass"):
        failures.append("MULTI_SEED_OPEN_VOCAB_INSTABILITY_DETECTED")
    if aggregate.get("char_bigram_margin_seed_pass_count", 0) < 2:
        failures.append("OPEN_VOCAB_MARGIN_TOO_WEAK")
    if aggregate.get("min_delta_vs_char_bigram_loss", -1.0) <= 0.0:
        failures.append("OPEN_VOCAB_MARGIN_TOO_WEAK")
    if aggregate.get("mean_delta_vs_char_bigram_loss", -1.0) < 0.03:
        failures.append("OPEN_VOCAB_MARGIN_TOO_WEAK")
    if not aggregate.get("collapse_false_all_seeds"):
        failures.append("OPEN_VOCAB_GENERATION_SMOKE_FAILS")
    if not aggregate.get("retention_pass_all_seeds"):
        failures.append("BOUNDED_CHAT_RETENTION_REGRESSION_DETECTED")
    if not aggregate.get("packaged_winner_hash_unchanged_all_seeds"):
        failures.append("PACKAGED_CHECKPOINT_MUTATION_DETECTED")
    if not aggregate.get("fineweb_source_hash_unchanged_all_seeds"):
        failures.append("FINEWEB_SOURCE_MUTATION_DETECTED")
    seed_rows = read_jsonl(SMOKE_ROOT / "seed_run_manifest.json")
    seeds = [int(row["seed"]) for row in seed_rows]
    for seed in seeds:
        failures.extend(check_seed(seed))
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    failures: list[str] = []
    missing, files = read_files()
    failures.extend([f"MISSING:{item}" for item in missing])
    joined = "\n".join(files.values())
    audited_text = "\n".join(text for rel, text in files.items() if rel in REQUIRED_DOCS)
    for term in REQUIRED_TERMS + EXACT_COMMANDS:
        if term not in joined:
            failures.append(f"MISSING_TERM:{term}")
    for marker in PLACEHOLDERS:
        if marker.lower() in audited_text.lower():
            failures.append(f"PLACEHOLDER_PRESENT:{marker}")
    failures.extend(find_false_claims(audited_text))
    failures.extend(artifact_checks())
    if git_status(["LICENSE"]):
        failures.append("ROOT_LICENSE_CHANGED")
    if runtime_surface_mutation_detected():
        failures.append("RUNTIME_SURFACE_MUTATION_DETECTED")
    if generated_artifact_staged():
        failures.append("GENERATED_ARTIFACT_STAGED")
    payload = {
        "schema_version": "open_vocab_fineweb_margin_scale_confirm_check_v1",
        "status": "passed" if not failures else "failed",
        "check_only": args.check_only,
        "verdicts": [
            "OPEN_VOCAB_FINEWEB_MARGIN_SCALE_CONFIRM_STATIC_CHECK_POSITIVE",
            "OPEN_VOCAB_FINEWEB_MARGIN_FILES_WRITTEN",
            "FINEWEB_SOURCE_IMMUTABILITY_CHECKED",
            "MULTI_SEED_BASE_GATES_CHECKED",
            "CHAR_BIGRAM_MARGIN_CHECKED",
            "BASELINE_SPLIT_EQUALITY_CHECKED",
            "GENERATION_COLLAPSE_CHECKED",
            "BOUNDED_RETENTION_CHECKED",
            "ARCHITECTURE_CLAIM_BOUNDARY_CHECKED",
            "ROOT_LICENSE_UNCHANGED",
        ]
        if not failures
        else ["OPEN_VOCAB_FINEWEB_MARGIN_SCALE_CONFIRM_STATIC_CHECK_FAILS", *failures],
        "checked_files": REQUIRED_DOCS + REQUIRED_SOURCE,
        "allowed_mutations": sorted(ALLOWED_MUTATIONS),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
