#!/usr/bin/env python3
"""Static checker for STABLE_LOOP_PHASE_LOCK_071B repair overfit analysis."""

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
    "docs/research/STABLE_LOOP_PHASE_LOCK_071B_REPAIR_OVERFIT_FAILURE_ANALYSIS_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_071B_REPAIR_OVERFIT_FAILURE_ANALYSIS_RESULT.md",
]

REQUIRED_SOURCE = [
    "scripts/probes/run_stable_loop_phase_lock_071b_repair_overfit_failure_analysis.py",
    "scripts/probes/run_stable_loop_phase_lock_071b_repair_overfit_failure_analysis_check.py",
]

ALLOWED_MUTATIONS = set(REQUIRED_DOCS + REQUIRED_SOURCE)

EXACT_COMMANDS = [
    "python -m py_compile scripts/probes/run_stable_loop_phase_lock_071b_repair_overfit_failure_analysis.py",
    "python scripts/probes/run_stable_loop_phase_lock_071b_repair_overfit_failure_analysis.py --out target/pilot_wave/stable_loop_phase_lock_071b_repair_overfit_failure_analysis/smoke --upstream-071-root target/pilot_wave/stable_loop_phase_lock_071_repaired_checkpoint_capability_confirm/smoke --upstream-070-root target/pilot_wave/stable_loop_phase_lock_070_distractor_resistant_anchorroute_training/smoke --benchmark-069-root target/pilot_wave/stable_loop_phase_lock_069_model_capability_benchmark_gate/smoke --heartbeat-sec 20",
    "python -m py_compile scripts/probes/run_stable_loop_phase_lock_071b_repair_overfit_failure_analysis_check.py",
    "python scripts/probes/run_stable_loop_phase_lock_071b_repair_overfit_failure_analysis_check.py --check-only",
    "python scripts/probes/run_stable_loop_phase_lock_071_repaired_checkpoint_capability_confirm_check.py --check-only",
    "python scripts/probes/run_stable_loop_phase_lock_070_distractor_resistant_anchorroute_check.py --check-only",
    "git diff --check",
]

BOUNDARY_TOKENS = [
    "analysis-only",
    "no training",
    "no inference",
    "no checkpoint repair",
    "no checkpoint mutation",
    "no 069/070/071 rerun",
    "no model capability improved",
    "no production training",
    "no open-ended assistant",
    "no free-form generation",
    "no language grounding",
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
    "UPSTREAM_071_ARTIFACT_MISSING",
    "FAILURE_CASE_INPUT_MISSING",
    "WRONG_ANSWER_SOURCE_UNCLASSIFIED_TOO_HIGH",
    "COUNTERFACTUAL_ANALYSIS_INCOMPLETE",
    "CONTEXT_ANALYSIS_INCOMPLETE",
    "POCKET_ANALYSIS_INCOMPLETE",
    "CURRICULUM_PATCH_MISSING",
    "TRAINING_SIDE_EFFECT_DETECTED",
    "CHECKPOINT_MUTATION_DETECTED",
    "RUNTIME_SURFACE_MUTATION_DETECTED",
    "ROOT_LICENSE_CHANGED",
    "FRESH_COUNTERFACTUAL_BINDING",
    "FRESH_CONTEXT_ENTITY_EXTRACTION",
    "FRESH_IRRELEVANT_POCKET_SUPPRESSION",
    "old_scenario_value",
    "distractor_scenario_value",
    "first_ledger_value",
    "side_note_value",
    "inactive_pocket_value",
    "stale_pocket_value",
    "copy_first_match_value",
    "no_route_control_value",
    "unknown_label",
    "unknown_label <= 20%",
    "active_scenario_miss_rate",
    "old_scenario_selection_rate",
    "distractor_scenario_selection_rate",
    "first_ledger_value_selection_rate",
    "inactive_pocket_selection_rate",
    "stale_pocket_selection_rate",
    "exact_anchor_success_rate",
    "key_collision_rate",
    "side_note_value_selection_rate",
    "copy_first_match_agreement_rate",
    "no_route_agreement_rate",
    "irrelevant_pocket_selection_rate",
    "human_failure_digest.jsonl",
    "task_family",
    "classified_wrong_source",
    "short_diagnosis",
    "active scenario marker strengthening",
    "same key / different scenario training",
    "stale scenario suppression",
    "inactive pocket negative examples",
    "scenario:active",
    "scenario:old",
    "scenario:distractor",
    "answer-only plus trace-mixed variants",
    "no-route and copy-first controls retained",
    "no FineWeb scale-up as immediate fix",
    "queue.json",
    "progress.jsonl",
    "analysis_config.json",
    "upstream_071_manifest.json",
    "failure_cluster_report.json",
    "counterfactual_source_attribution.json",
    "context_extraction_source_attribution.json",
    "pocket_suppression_source_attribution.json",
    "template_failure_matrix.json",
    "wrong_answer_source_matrix.json",
    "active_scenario_miss_report.json",
    "distractor_scenario_selection_report.json",
    "stale_pocket_selection_report.json",
    "key_collision_report.json",
    "no_route_control_comparison.json",
    "recommended_curriculum_patch.json",
    "summary.json",
    "report.md",
    "REPAIR_OVERFIT_FAILURE_ANALYSIS_POSITIVE",
    "UPSTREAM_071_FAILURE_PROFILE_LOADED",
    "COUNTERFACTUAL_FAILURE_CLUSTERS_WRITTEN",
    "CONTEXT_EXTRACTION_FAILURE_CLUSTERS_WRITTEN",
    "POCKET_SUPPRESSION_FAILURE_CLUSTERS_WRITTEN",
    "WRONG_ANSWER_SOURCE_ATTRIBUTION_WRITTEN",
    "ACTIVE_SCENARIO_MISS_RATE_RECORDED",
    "DISTRACTOR_SCENARIO_SELECTION_RECORDED",
    "KEY_COLLISION_REPORT_WRITTEN",
    "CURRICULUM_PATCH_RECOMMENDED",
    "NO_TRAINING_PERFORMED",
    "PRODUCTION_TRAINING_NOT_CLAIMED",
]

PLACEHOLDERS = ["TODO", "TBD", "placeholder", "pending measured", "coming soon"]

FORBIDDEN_CLAIMS = {
    "TRAINING_SIDE_EFFECT_DETECTED": ["training performed", "training happened", "retrained", "rerun 071"],
    "CHECKPOINT_MUTATION_DETECTED": ["checkpoint mutated", "checkpoint repaired", "checkpoint updated"],
    "CAPABILITY_IMPROVEMENT_CLAIM_DETECTED": ["capability improved", "model improved", "capability gate passed"],
    "OPEN_ENDED_CLAIM_DETECTED": ["open-ended assistant", "free-form generation", "free form generation"],
    "LANGUAGE_GROUNDING_CLAIM_DETECTED": ["language grounding"],
    "PRODUCTION_TRAINING_CLAIM_DETECTED": ["production training", "production model"],
    "GA_CLAIM_DETECTED": ["GA release", "GA launched", "generally available"],
    "PUBLIC_BETA_CLAIM_DETECTED": ["public beta"],
    "HOSTED_SAAS_CLAIM_DETECTED": ["hosted SaaS"],
}

NEGATION_MARKERS = ["not ", "no ", "does not ", "do not ", "without ", "unsupported", "false", "reject", "rejected"]
GENERATED_PATH_PARTS = ["target/", "node_modules/", ".svelte-kit/"]
GENERATED_SUFFIXES = [".exe", ".dll", ".so", ".dylib", ".zip", ".tar", ".tar.gz", ".tgz", ".7z", ".ckpt", ".bin"]
GENERATED_NAME_PARTS = ["checkpoint", "smoke/", "service job artifacts", "release artifacts"]

POSITIVE_VERDICTS = [
    "REPAIR_OVERFIT_FAILURE_ANALYSIS_STATIC_CHECK_POSITIVE",
    "REPAIR_OVERFIT_FAILURE_ANALYSIS_PACKAGE_WRITTEN",
    "ANALYSIS_ONLY_HARD_WALL_WRITTEN",
    "WRONG_ANSWER_SOURCE_CLASSIFICATION_REQUIRED",
    "CURRICULUM_PATCH_REQUIRED",
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
    return False


def generated_artifact_staged() -> bool:
    for path in changed_paths():
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
        if rel.endswith(".py"):
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
        if rel.endswith(".py"):
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

    verdicts = POSITIVE_VERDICTS.copy() if check_pass else ["REPAIR_OVERFIT_FAILURE_ANALYSIS_STATIC_CHECK_FAILS"]
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
        "missing_doc_references": [],
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
