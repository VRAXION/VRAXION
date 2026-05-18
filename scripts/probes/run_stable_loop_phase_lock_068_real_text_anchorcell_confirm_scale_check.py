#!/usr/bin/env python3
"""Static checker for STABLE_LOOP_PHASE_LOCK_068 real-text AnchorCell confirm scale."""

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
    "docs/research/STABLE_LOOP_PHASE_LOCK_068_REAL_TEXT_ANCHORCELL_CONFIRM_SCALE_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_068_REAL_TEXT_ANCHORCELL_CONFIRM_SCALE_RESULT.md",
]

REQUIRED_SOURCE = [
    "instnct-core/examples/phase_lane_real_text_anchorcell_training_poc.rs",
    "scripts/probes/run_stable_loop_phase_lock_068_real_text_anchorcell_confirm_scale.py",
    "scripts/probes/run_stable_loop_phase_lock_068_real_text_anchorcell_confirm_scale_check.py",
]

EXACT_COMMANDS = [
    "cargo check -p instnct-core --example phase_lane_real_text_anchorcell_training_poc",
    "python -m py_compile scripts/probes/run_stable_loop_phase_lock_068_real_text_anchorcell_confirm_scale.py",
    "python -m py_compile scripts/probes/run_stable_loop_phase_lock_068_real_text_anchorcell_confirm_scale_check.py",
    'python scripts/probes/run_stable_loop_phase_lock_068_real_text_anchorcell_confirm_scale.py --out target/pilot_wave/stable_loop_phase_lock_068_real_text_anchorcell_confirm_scale/confirm --fineweb-root "S:\\AI\\MESSY TRAINING DATA - INPUT ONLY\\Fineweb edu 10B" --fineweb-bytes 268435456 --seeds 2026,2027,2028 --anchorcell-examples 100000 --heartbeat-sec 20',
    "python scripts/probes/run_stable_loop_phase_lock_068_real_text_anchorcell_confirm_scale_check.py --check-only",
    "python scripts/probes/run_stable_loop_phase_lock_067_real_text_anchorcell_training_poc_check.py --check-only",
    "cargo test -p instnct-core sdk_candidate",
    "git diff --check",
]

DOC_REFERENCES = [
    "067 real-text AnchorCell training PoC",
    "068 real-text AnchorCell confirm scale",
    "FineWeb confirm snapshot",
    "MIXED_WITH_ROUTE_GRAMMAR_ON",
]

BOUNDARY_TOKENS = [
    "no full 10B training",
    "no production training",
    "no full English model",
    "no language grounding",
    "no GA",
    "no public beta",
    "no hosted SaaS",
    "no clinical use",
    "no high-stakes education use",
    "no full VRAXION",
    "no consciousness",
    "no biological/FlyWire equivalence",
    "no physical quantum behavior",
]

REQUIRED_TERMS = [
    "--mode confirm",
    "--fineweb-source",
    "--anchorcell-examples",
    "fineweb_bytes <= 1 GiB",
    "anchorcell_examples <= 250000",
    "child_summary_newer_than_068_start",
    "child_report_newer_than_068_start",
    "STALE_CHILD_ARTIFACT_USED",
    "CONFIRM_SNAPSHOT_MUTATION_DETECTED",
    "FINEWEB_INPUT_MUTATION_DETECTED",
    "FULL_CORPUS_TRAINING_ATTEMPTED",
    "CONFIRM_SCALE_LIMIT_EXCEEDED",
    "MULTI_SEED_INSTABILITY_DETECTED",
    "CHILD_GATE_RECHECK_FAILS",
    "BASELINE_KNOCKOUT_REGRESSION",
    "FAILURE_CASE_REPORT_MISSING",
    "PERFORMANCE_CLAIM_OVERREACH",
    "RUNTIME_SURFACE_MUTATION_DETECTED",
    "ROOT_LICENSE_CHANGED",
    "queue.json",
    "progress.jsonl",
    "confirm_config.json",
    "fineweb_confirm_source_manifest.json",
    "fineweb_extraction_manifest.json",
    "seed_metrics.jsonl",
    "aggregate_metrics.json",
    "multi_seed_stability.json",
    "training_curve_report.json",
    "baseline_knockout_aggregate.json",
    "failure_case_samples.jsonl",
    "checkpoint_pipeline_report.json",
    "summary.json",
    "report.md",
    "REAL_TEXT_ANCHORCELL_CONFIRM_SCALE_POSITIVE",
    "FRESH_CHILD_RUNS_CONFIRMED",
    "CONFIRM_SNAPSHOT_IMMUTABILITY_PASSES",
    "CHILD_067_GATES_RECHECKED",
    "MULTI_SEED_MIN_GATE_PASSES",
    "CONFIRM_SCALE_LIMIT_ENFORCED",
    "FAILURE_CASE_REPORT_WRITTEN",
    "BASELINE_KNOCKOUT_STABLE",
    "CHECKPOINT_PIPELINE_MULTI_SEED_PASS",
    "PRODUCTION_TRAINING_NOT_CLAIMED",
]

PLACEHOLDERS = ["TODO", "TBD", "placeholder", "pending measured", "coming soon"]

FORBIDDEN_CLAIMS = {
    "FULL_CORPUS_TRAINING_CLAIM_DETECTED": ["full 10B training", "full-corpus training", "all-shard training"],
    "PRODUCTION_TRAINING_CLAIM_DETECTED": ["production training", "production trained"],
    "FULL_ENGLISH_MODEL_CLAIM_DETECTED": ["full English model"],
    "LANGUAGE_GROUNDING_CLAIM_DETECTED": ["language grounding"],
    "GA_CLAIM_DETECTED": ["GA release", "GA launched", "generally available"],
    "PUBLIC_BETA_CLAIM_DETECTED": ["public beta"],
    "HOSTED_SAAS_CLAIM_DETECTED": ["hosted SaaS"],
    "CLINICAL_READY_CLAIM_DETECTED": ["clinical readiness", "clinical ready"],
    "HIGH_STAKES_EDUCATION_READY_CLAIM_DETECTED": ["high-stakes education readiness", "high-stakes education ready"],
    "PERFORMANCE_CLAIM_OVERREACH": ["production performance claim", "training throughput claim"],
}

NEGATION_MARKERS = ["not ", "no ", "does not ", "do not ", "without ", "false", "reject", "rejected"]
GENERATED_PATH_PARTS = ["target/", "node_modules/", ".svelte-kit/"]
GENERATED_SUFFIXES = [".exe", ".dll", ".so", ".dylib", ".zip", ".tar", ".tar.gz", ".tgz", ".7z", ".ckpt", ".bin"]
GENERATED_NAME_PARTS = ["checkpoint", "smoke/", "service job artifacts", "release artifacts", "confirm_snapshot/"]
MUTATION_PATHS = ["LICENSE", "instnct-core", "tools/instnct_service_alpha", "tools/instnct_deploy", "docs/releases"]

ALLOWED_MUTATIONS = {
    "instnct-core/examples/phase_lane_real_text_anchorcell_training_poc.rs",
    "scripts/probes/run_stable_loop_phase_lock_068_real_text_anchorcell_confirm_scale.py",
    "scripts/probes/run_stable_loop_phase_lock_068_real_text_anchorcell_confirm_scale_check.py",
    "docs/research/STABLE_LOOP_PHASE_LOCK_068_REAL_TEXT_ANCHORCELL_CONFIRM_SCALE_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_068_REAL_TEXT_ANCHORCELL_CONFIRM_SCALE_RESULT.md",
}

POSITIVE_VERDICTS = [
    "REAL_TEXT_ANCHORCELL_CONFIRM_SCALE_STATIC_CHECK_POSITIVE",
    "REAL_TEXT_ANCHORCELL_CONFIRM_SCALE_PACKAGE_WRITTEN",
    "CONFIRM_SCALE_LIMIT_ENFORCED",
    "FRESH_CHILD_RUN_VALIDATION_WRITTEN",
    "CONFIRM_SNAPSHOT_IMMUTABILITY_WRITTEN",
    "CHILD_067_GATE_RECHECK_WRITTEN",
    "MULTI_SEED_MIN_GATE_WRITTEN",
    "FAILURE_CASE_REPORT_WRITTEN",
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


def root_license_changed() -> bool:
    return bool(git_status(["LICENSE"]))


def runtime_surface_mutation_detected() -> bool:
    status = git_status(MUTATION_PATHS)
    for raw in status.splitlines():
        path = raw[3:].replace("\\", "/")
        if path in ALLOWED_MUTATIONS:
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
        if any(part in path for part in GENERATED_NAME_PARTS) and path.startswith("target/"):
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
                        hits.append({"file": rel, "line": idx, "phrase": phrase, "verdict": verdict})
    return hits


def missing_commands(files: dict[str, str]) -> list[str]:
    bundle = "\n".join(files.values())
    return [command for command in EXACT_COMMANDS if command not in bundle]


def missing_doc_references(files: dict[str, str]) -> list[str]:
    bundle = "\n".join(files.get(rel, "") for rel in REQUIRED_DOCS)
    return [ref for ref in DOC_REFERENCES if ref not in bundle]


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
        parser.error("068 checker is static-only; pass --check-only")

    missing_files, files = read_files()
    placeholders = placeholder_hits(files)
    commands = missing_commands(files)
    doc_refs = missing_doc_references(files)
    boundary = missing_boundary_tokens(files)
    forbidden = forbidden_claim_hits(files)
    required = missing_required_terms(files)
    generated = generated_artifact_staged()
    root_changed = root_license_changed()
    runtime_mutation = runtime_surface_mutation_detected()

    check_pass = not any(
        [
            missing_files,
            placeholders,
            commands,
            doc_refs,
            boundary,
            forbidden,
            required,
            generated,
            root_changed,
            runtime_mutation,
        ]
    )
    verdicts = POSITIVE_VERDICTS if check_pass else ["REAL_TEXT_ANCHORCELL_CONFIRM_SCALE_STATIC_CHECK_FAILS"]
    if generated:
        verdicts.append("GENERATED_ARTIFACT_STAGED")
    if root_changed:
        verdicts.append("ROOT_LICENSE_CHANGED")
    if runtime_mutation:
        verdicts.append("RUNTIME_SURFACE_MUTATION_DETECTED")
    if forbidden:
        verdicts.extend(sorted({hit["verdict"] for hit in forbidden}))

    print(
        json.dumps(
            {
                "check_pass": check_pass,
                "missing_docs": [p for p in missing_files if p.startswith("docs/")],
                "placeholder_hits": placeholders,
                "missing_commands": commands,
                "missing_doc_references": doc_refs,
                "missing_boundary_tokens": boundary,
                "forbidden_claim_hits": forbidden,
                "generated_artifact_staged": generated,
                "root_license_changed": root_changed,
                "runtime_surface_mutation_detected": runtime_mutation,
                "missing_required_terms": required,
                "verdicts": verdicts,
            },
            separators=(",", ":"),
        )
    )
    return 0 if check_pass else 1


if __name__ == "__main__":
    sys.exit(main())
