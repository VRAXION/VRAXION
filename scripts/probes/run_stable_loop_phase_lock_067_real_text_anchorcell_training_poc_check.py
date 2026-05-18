#!/usr/bin/env python3
"""Static checker for STABLE_LOOP_PHASE_LOCK_067 real-text AnchorCell training PoC."""

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
    "docs/research/STABLE_LOOP_PHASE_LOCK_067_REAL_TEXT_ANCHORCELL_TRAINING_POC_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_067_REAL_TEXT_ANCHORCELL_TRAINING_POC_RESULT.md",
]

REQUIRED_SOURCE = [
    "instnct-core/examples/phase_lane_real_text_anchorcell_training_poc.rs",
    "scripts/probes/run_stable_loop_phase_lock_067_real_text_anchorcell_training_poc_check.py",
]

EXACT_COMMANDS = [
    "cargo check -p instnct-core --example phase_lane_real_text_anchorcell_training_poc",
    'cargo run -p instnct-core --example phase_lane_real_text_anchorcell_training_poc -- --out target/pilot_wave/stable_loop_phase_lock_067_real_text_anchorcell_training_poc/smoke --fineweb-root "S:\\AI\\MESSY TRAINING DATA - INPUT ONLY\\Fineweb edu 10B" --mode smoke --seed 2026 --heartbeat-sec 20',
    "python -m py_compile scripts/probes/run_stable_loop_phase_lock_067_real_text_anchorcell_training_poc_check.py",
    "python scripts/probes/run_stable_loop_phase_lock_067_real_text_anchorcell_training_poc_check.py --check-only",
    "cargo test -p instnct-core sdk_candidate",
    "python scripts/probes/run_stable_loop_phase_lock_066_core_ga_private_readiness_check.py --check-only",
    "git diff --check",
]

BOUNDARY_TOKENS = [
    "no production training",
    "no full-corpus training",
    "no GA",
    "no public beta",
    "no hosted SaaS",
    "no clinical use",
    "no high-stakes education use",
    "no full VRAXION",
    "no language grounding",
    "no consciousness",
    "no biological/FlyWire equivalence",
    "no physical quantum behavior",
]

REQUIRED_TERMS = [
    "fineweb_edu_30m.txt",
    "FINEWEB_SMOKE_SOURCE_MISSING",
    "FINEWEB_INPUT_MUTATION_DETECTED",
    "FULL_CORPUS_TRAINING_ATTEMPTED",
    "NO_ACTUAL_TRAINING_UPDATE_DETECTED",
    "ORACLE_SHORTCUT_DETECTED",
    "TRAIN_EVAL_LEAKAGE_DETECTED",
    "BASELINE_EVAL_MISMATCH",
    "FAMILY_MIN_GATE_FAILS",
    "STATIC_OUTPUT_COLLAPSE_DETECTED",
    "COPY_SHORTCUT_DETECTED",
    "CHECKPOINT_RELOAD_FAILS",
    "ROLLBACK_REHEARSAL_FAILS",
    "RESUME_FROM_CHECKPOINT_FAILS",
    "prediction_oracle_used = false",
    "checkpoint_after_hash != checkpoint_before_hash",
    "train_eval_exact_input_overlap_count",
    "train_ood_exact_input_overlap_count",
    "baseline_eval_mismatch = false",
    "progress.jsonl",
    "dataset_manifest.json",
    "fineweb_file_manifest.json",
    "fineweb_sample_offsets.jsonl",
    "baseline_metrics.json",
    "training_metrics.jsonl",
    "checkpoint_manifest.json",
    "checkpoint_hashes.json",
    "reload_eval_report.json",
    "rollback_report.json",
    "resume_report.json",
    "collapse_metrics.json",
    "baseline_knockout_report.json",
    "per_family_metrics.json",
    "summary.json",
    "report.md",
    "REAL_TEXT_ANCHORCELL_TRAINING_POC_POSITIVE",
    "FINEWEB_INPUT_IMMUTABILITY_PASSES",
    "MIXED_DATASET_BEATS_BASELINES",
    "CHECKPOINT_PIPELINE_STRICT_PASS",
    "PRODUCTION_TRAINING_NOT_CLAIMED",
]

PLACEHOLDERS = ["TODO", "TBD", "placeholder", "pending measured", "coming soon"]

FORBIDDEN_CLAIMS = {
    "PRODUCTION_TRAINING_CLAIM_DETECTED": ["production training", "production trained"],
    "FULL_CORPUS_TRAINING_CLAIM_DETECTED": ["full-corpus training", "full 10b training"],
    "GA_CLAIM_DETECTED": ["GA release", "GA launched", "generally available"],
    "PUBLIC_BETA_CLAIM_DETECTED": ["public beta"],
    "HOSTED_SAAS_CLAIM_DETECTED": ["hosted SaaS"],
    "CLINICAL_READY_CLAIM_DETECTED": ["clinical readiness", "clinical ready"],
    "HIGH_STAKES_EDUCATION_READY_CLAIM_DETECTED": [
        "high-stakes education readiness",
        "high-stakes education ready",
    ],
    "LANGUAGE_GROUNDING_CLAIM_DETECTED": ["language grounding"],
    "CONSCIOUSNESS_CLAIM_DETECTED": ["consciousness"],
}

NEGATION_MARKERS = ["not ", "no ", "does not ", "do not ", "without ", "false", "reject"]
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
GENERATED_NAMES = [
    "progress.jsonl",
    "dataset_manifest.json",
    "fineweb_file_manifest.json",
    "training_metrics.jsonl",
    "checkpoint_manifest.json",
    "checkpoint_hashes.json",
    "reload_eval_report.json",
    "rollback_report.json",
    "resume_report.json",
    "collapse_metrics.json",
    "baseline_knockout_report.json",
]

PUBLIC_SURFACE_PATHS = [
    "LICENSE",
    "instnct-core/src/lib.rs",
    "instnct-core/Cargo.toml",
    "tools/instnct_service_alpha",
    "tools/instnct_deploy",
]

POSITIVE_VERDICTS = [
    "REAL_TEXT_ANCHORCELL_TRAINING_POC_POSITIVE",
    "FINEWEB_INPUT_IMMUTABILITY_PASSES",
    "FINEWEB_CARRIER_TRAINING_WORKS",
    "ANCHORCELL_TRACE_SUPERVISION_WORKS",
    "MIXED_DATASET_BEATS_BASELINES",
    "FREQUENCY_BASELINE_REJECTED",
    "BIGRAM_TRIGRAM_BASELINE_REJECTED",
    "STATIC_OUTPUT_COLLAPSE_REJECTED",
    "COPY_SHORTCUT_REJECTED",
    "TRAIN_EVAL_LEAKAGE_REJECTED",
    "ORACLE_SHORTCUT_REJECTED",
    "PER_FAMILY_GATES_PASS",
    "CHECKPOINT_PIPELINE_STRICT_PASS",
    "PRODUCTION_TRAINING_NOT_CLAIMED",
]


def git_status(paths: list[str] | None = None) -> str:
    cmd = ["git", "status", "--short"]
    if paths:
        cmd.extend(["--", *paths])
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip()


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


def public_surface_mutation_detected() -> bool:
    status = git_status(PUBLIC_SURFACE_PATHS)
    for line in status.splitlines():
        if "instnct-core/examples/phase_lane_real_text_anchorcell_training_poc.rs" in line:
            continue
        if "scripts/probes/run_stable_loop_phase_lock_067_real_text_anchorcell_training_poc_check.py" in line:
            continue
        return True
    return False


def generated_artifact_staged() -> bool:
    status = git_status()
    for raw in status.splitlines():
        path = raw[3:].replace("\\", "/")
        if any(part in path for part in GENERATED_PATH_PARTS):
            return True
        if any(path.endswith(suffix) for suffix in GENERATED_SUFFIXES):
            return True
        if any(name in path for name in GENERATED_NAMES) and path.startswith("target/"):
            return True
    return False


def placeholder_hits(files: dict[str, str]) -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    for rel, text in files.items():
        if rel.endswith(".py") or rel.endswith(".rs"):
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
        if rel.endswith(".py") or rel.endswith(".rs"):
            continue
        for idx, raw_line in enumerate(text.splitlines(), start=1):
            line = raw_line.lower()
            if not line.strip():
                continue
            for verdict, phrases in FORBIDDEN_CLAIMS.items():
                for phrase in phrases:
                    lowered = phrase.lower()
                    if lowered in line and not line_is_negated(line, lowered):
                        hits.append(
                            {
                                "file": rel,
                                "line": idx,
                                "phrase": phrase,
                                "verdict": verdict,
                            }
                        )
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
        parser.error("067 checker is static-only; pass --check-only")

    missing_files, files = read_files()
    placeholders = placeholder_hits(files)
    commands = missing_commands(files)
    boundary = missing_boundary_tokens(files)
    forbidden = forbidden_claim_hits(files)
    required = missing_required_terms(files)
    root_changed = root_license_changed()
    public_surface_changed = public_surface_mutation_detected()
    generated = generated_artifact_staged()

    check_pass = not any(
        [
            missing_files,
            placeholders,
            commands,
            boundary,
            forbidden,
            required,
            root_changed,
            public_surface_changed,
            generated,
        ]
    )

    verdicts = POSITIVE_VERDICTS if check_pass else ["REAL_TEXT_ANCHORCELL_TRAINING_POC_STATIC_CHECK_FAILS"]
    if root_changed:
        verdicts.append("ROOT_LICENSE_CHANGED")
    if public_surface_changed:
        verdicts.append("PUBLIC_SURFACE_MUTATION_DETECTED")
    if generated:
        verdicts.append("GENERATED_ARTIFACT_STAGED")
    if forbidden:
        verdicts.extend(sorted({hit["verdict"] for hit in forbidden}))

    print(
        json.dumps(
            {
                "check_pass": check_pass,
                "missing_docs": [p for p in missing_files if p.startswith("docs/")],
                "missing_source": [p for p in missing_files if not p.startswith("docs/")],
                "placeholder_hits": placeholders,
                "missing_commands": commands,
                "missing_boundary_tokens": boundary,
                "forbidden_claim_hits": forbidden,
                "generated_artifact_staged": generated,
                "root_license_changed": root_changed,
                "public_surface_mutation_detected": public_surface_changed,
                "missing_required_terms": required,
                "verdicts": verdicts,
            },
            separators=(",", ":"),
        )
    )
    return 0 if check_pass else 1


if __name__ == "__main__":
    sys.exit(main())
