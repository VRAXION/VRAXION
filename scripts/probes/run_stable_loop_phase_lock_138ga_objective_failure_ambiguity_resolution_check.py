#!/usr/bin/env python3
"""Checker for STABLE_LOOP_PHASE_LOCK_138GA ambiguity resolution."""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_138ga_objective_failure_ambiguity_resolution/smoke"
RUNNER = "scripts/probes/run_stable_loop_phase_lock_138ga_objective_failure_ambiguity_resolution.py"
CHECKER = "scripts/probes/run_stable_loop_phase_lock_138ga_objective_failure_ambiguity_resolution_check.py"
DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_138GA_OBJECTIVE_FAILURE_AMBIGUITY_RESOLUTION_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_138GA_OBJECTIVE_FAILURE_AMBIGUITY_RESOLUTION_RESULT.md",
]
ALLOWED_MUTATIONS = {RUNNER, CHECKER, *DOCS}
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "upstream_138g_manifest.json",
    "upstream_138r_manifest.json",
    "near_match_rows.jsonl",
    "near_match_classification_report.json",
    "meaningful_partial_answer_report.json",
    "scorer_eval_weakness_report.json",
    "objective_failure_disambiguation.json",
    "human_readable_near_match_samples.jsonl",
    "decision.json",
    "summary.json",
    "report.md",
]
REQUIRED_TERMS = [
    "STABLE_LOOP_PHASE_LOCK_138GA_OBJECTIVE_FAILURE_AMBIGUITY_RESOLUTION",
    "artifact-only near-match ambiguity resolution",
    "objective_failure_ambiguous",
    "near_match_artifact_inconsistency",
    "meaningful_partial_answer",
    "train_namespace_overlap",
    "unknown_near_match",
    "138H_REAL_RAW_REASONING_ROLLOUT_ALIGNED_OBJECTIVE_REDESIGN_PLAN",
    "138E_REASONING_SCORER_OR_TASK_WEAKNESS_ANALYSIS",
    "138GB_NEAR_MATCH_MANUAL_REVIEW_PACKET",
    "shared_raw_generation_helper.py",
    "raw assistant capability remains quarantined",
    "structured/tool capability remains invalidated",
    "not GPT-like readiness",
    "not open-domain assistant readiness",
    "not production chat",
    "not public API",
    "not deployment readiness",
    "not safety alignment",
]
FALSE_FLAGS = [
    "reasoning_restored",
    "reasoning_subtrack_real_raw_evidence_partially_restored",
    "raw_assistant_capability_restored",
    "structured_tool_capability_restored",
    "gpt_like_readiness_claimed",
    "open_domain_assistant_readiness_claimed",
    "production_chat_claimed",
    "public_api_claimed",
    "deployment_readiness_claimed",
    "safety_alignment_claimed",
]
PRIMARY_LABELS = {
    "meaningful_partial_answer",
    "numeric_partial_match",
    "formatting_or_wrapper_mismatch",
    "train_namespace_overlap",
    "stale_chat_overlap",
    "prompt_copy_overlap",
    "distractor_overlap",
    "common_token_overlap",
    "scorer_false_near_match",
    "unknown_near_match",
}
NEGATION_MARKERS = ["not ", "no ", "does not ", "do not ", "without ", "false", "remains quarantined", "remains invalidated", "artifact-only"]
FORBIDDEN_CLAIMS = {
    "REASONING_RESTORED_FALSE_CLAIM": ["reasoning restored", "reasoning is restored"],
    "RAW_ASSISTANT_CAPABILITY_RESTORED_FALSE_CLAIM": ["raw assistant capability restored"],
    "STRUCTURED_TOOL_CAPABILITY_RESTORED_FALSE_CLAIM": ["structured/tool capability restored", "structured tool capability restored"],
    "GPT_LIKE_READINESS_FALSE_CLAIM": ["GPT-like ready", "GPT-like readiness improved"],
    "OPEN_DOMAIN_ASSISTANT_CLAIM_DETECTED": ["open-domain assistant ready"],
    "PRODUCTION_CHAT_CLAIM_DETECTED": ["production chat ready", "production ready"],
    "PUBLIC_API_CLAIM_DETECTED": ["public API ready"],
    "DEPLOYMENT_CLAIM_DETECTED": ["deployment ready"],
    "SAFETY_ALIGNMENT_CLAIM_DETECTED": ["safety aligned"],
}


def git_status() -> str:
    result = subprocess.run(["git", "status", "--short"], cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    return result.stdout.rstrip("\n")


def changed_paths() -> list[str]:
    paths: list[str] = []
    for line in git_status().splitlines():
        if not line.strip():
            continue
        paths.append(line[3:].replace("\\", "/"))
    return paths


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def claim_is_negated(text: str, phrase: str) -> bool:
    lowered = text.lower()
    phrase_lower = phrase.lower()
    for match in re.finditer(re.escape(phrase_lower), lowered):
        window = lowered[max(0, match.start() - 240) : match.start()]
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
    for rel in [RUNNER, CHECKER, *DOCS]:
        path = REPO_ROOT / rel
        if not path.exists():
            missing.append(rel)
            continue
        text = path.read_text(encoding="utf-8")
        if rel in DOCS and len(text.strip()) < 200:
            missing.append(f"{rel}:too_short")
        files[rel] = text
    return missing, files


def ast_scan_source(paths: list[Path]) -> list[str]:
    failures: list[str] = []
    forbidden_imports = {"torch", "shared_raw_generation_helper"}
    old_runner_re = re.compile(r"^run_stable_loop_phase_lock_")
    forbidden_calls = {
        "raw_generate",
        "manual_seed",
        "forward",
        "backward",
        "optimizer",
        "Start-Process",
    }
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in forbidden_imports or old_runner_re.match(alias.name):
                        failures.append("INFERENCE_SIDE_EFFECT_DETECTED")
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module in forbidden_imports or old_runner_re.match(module):
                    failures.append("INFERENCE_SIDE_EFFECT_DETECTED")
            if isinstance(node, ast.Call):
                name = ast.unparse(node.func)
                if any(token in name for token in forbidden_calls):
                    failures.append("INFERENCE_SIDE_EFFECT_DETECTED")
    return failures


def check_artifacts() -> list[str]:
    failures: list[str] = []
    for rel in REQUIRED_ARTIFACTS:
        if not (SMOKE_ROOT / rel).exists():
            failures.append(f"MISSING_ARTIFACT:{rel}")
    if failures:
        return failures

    manifest_138g = load_json(SMOKE_ROOT / "upstream_138g_manifest.json")
    if manifest_138g.get("upstream_138g_verified") is not True:
        failures.append("UPSTREAM_138G_ARTIFACT_MISSING")
    if manifest_138g.get("decision") != "objective_failure_ambiguous":
        failures.append("UPSTREAM_138G_ARTIFACT_MISSING")
    if manifest_138g.get("next") != "138GA_OBJECTIVE_FAILURE_AMBIGUITY_RESOLUTION":
        failures.append("UPSTREAM_138G_ARTIFACT_MISSING")
    if manifest_138g.get("near_match_rate", 0.0) <= 0.0:
        failures.append("UPSTREAM_138G_ARTIFACT_MISSING")
    if manifest_138g.get("expected_token_inclusion_rate") != 0.0:
        failures.append("UPSTREAM_138G_ARTIFACT_MISSING")
    if manifest_138g.get("teacher_forced_loss_fields_diagnostic_gap") is not True:
        failures.append("UPSTREAM_138G_ARTIFACT_MISSING")

    manifest_138r = load_json(SMOKE_ROOT / "upstream_138r_manifest.json")
    if manifest_138r.get("upstream_138r_verified") is not True:
        failures.append("UPSTREAM_138R_ARTIFACT_MISSING")
    if manifest_138r.get("helper_canary_ast_leakage_controls_passed") is not True:
        failures.append("RAW_HELPER_INTEGRITY_FAILURE")

    rows = load_jsonl(SMOKE_ROOT / "near_match_rows.jsonl")
    samples = load_jsonl(SMOKE_ROOT / "human_readable_near_match_samples.jsonl")
    classification = load_json(SMOKE_ROOT / "near_match_classification_report.json")
    meaningful = load_json(SMOKE_ROOT / "meaningful_partial_answer_report.json")
    scorer = load_json(SMOKE_ROOT / "scorer_eval_weakness_report.json")
    disambiguation = load_json(SMOKE_ROOT / "objective_failure_disambiguation.json")
    decision = load_json(SMOKE_ROOT / "decision.json")
    summary = load_json(SMOKE_ROOT / "summary.json")
    report_text = (SMOKE_ROOT / "report.md").read_text(encoding="utf-8")

    if classification.get("total_scored_row_count") != 960:
        failures.append("NEAR_MATCH_EXTRACTION_INVALID")
    if classification.get("near_match_row_count") != len(rows):
        failures.append("NEAR_MATCH_EXTRACTION_INVALID")
    computed_rate = len(rows) / classification.get("total_scored_row_count", 1)
    if abs(classification.get("near_match_rate", -1.0) - computed_rate) > 1e-12:
        failures.append("NEAR_MATCH_ARTIFACT_INCONSISTENCY")
    if classification.get("near_match_rate_matches_138g") is not True:
        failures.append("NEAR_MATCH_ARTIFACT_INCONSISTENCY")
    if len(samples) != len(rows):
        failures.append("HUMAN_READABLE_NEAR_MATCH_SAMPLES_INCOMPLETE")

    for row in rows:
        label = row.get("primary_label")
        if label not in PRIMARY_LABELS:
            failures.append("NEAR_MATCH_CLASSIFICATION_INCOMPLETE")
        if row.get("primary_classification") != label:
            failures.append("NEAR_MATCH_CLASSIFICATION_INCOMPLETE")
        if not isinstance(row.get("secondary_labels"), list):
            failures.append("NEAR_MATCH_CLASSIFICATION_INCOMPLETE")
        for key in [
            "row_id",
            "family",
            "prompt",
            "generated_text",
            "expected_output",
            "near_match_score/source",
            "primary_classification",
            "deterministic_reason",
            "why_meaningful_or_not",
        ]:
            if key not in row:
                failures.append("HUMAN_READABLE_NEAR_MATCH_SAMPLES_INCOMPLETE")
        if row.get("expected_output", "").startswith("ANSWER=E") and "ANSWER=T" in row.get("generated_text", ""):
            if label != "train_namespace_overlap":
                failures.append("TRAIN_NAMESPACE_OVERLAP_NOT_PRIMARY")
        if label == "meaningful_partial_answer" and row.get("exact_expected_present") is not True:
            failures.append("MEANINGFUL_PARTIAL_OVERCLAIM")

    primary_counts = classification.get("primary_label_counts", {})
    if sum(primary_counts.values()) != len(rows):
        failures.append("NEAR_MATCH_CLASSIFICATION_INCOMPLETE")
    if classification.get("exactly_one_primary_label_per_row") is not True:
        failures.append("NEAR_MATCH_CLASSIFICATION_INCOMPLETE")
    if meaningful.get("generic_overlap_rejected") is not True:
        failures.append("MEANINGFUL_PARTIAL_OVERCLAIM")
    if meaningful.get("train_namespace_prefix_rejected") is not True:
        failures.append("MEANINGFUL_PARTIAL_OVERCLAIM")
    if scorer.get("expected_token_inclusion_rate") != 0.0:
        failures.append("SCORER_EVAL_WEAKNESS_REPORT_INVALID")

    valid_routes = {
        ("objective_failure_disambiguated", "138H_REAL_RAW_REASONING_ROLLOUT_ALIGNED_OBJECTIVE_REDESIGN_PLAN"),
        ("scorer_or_eval_design_contributes", "138E_REASONING_SCORER_OR_TASK_WEAKNESS_ANALYSIS"),
        ("ambiguity_requires_manual_sample_review", "138GB_NEAR_MATCH_MANUAL_REVIEW_PACKET"),
    }
    if (decision.get("decision"), decision.get("next")) not in valid_routes:
        failures.append("DECISION_MISMATCH")
    if decision.get("decision") == "objective_failure_disambiguated" and disambiguation.get("objective_failure_disambiguated") is not True:
        failures.append("DECISION_MISMATCH")
    if decision.get("decision") == "scorer_or_eval_design_contributes" and scorer.get("scorer_or_eval_design_contributes") is not True:
        failures.append("DECISION_MISMATCH")
    if decision.get("decision") == "ambiguity_requires_manual_sample_review" and decision.get("unknown_near_match_rate", 0.0) <= 0.10:
        failures.append("DECISION_MISMATCH")

    if decision.get("artifact_only_analysis") is not True:
        failures.append("ARTIFACT_ONLY_BOUNDARY_MISSING")
    if decision.get("new_model_inference_run") is not False or decision.get("shared_helper_called") is not False:
        failures.append("ARTIFACT_ONLY_BOUNDARY_MISSING")
    for key in FALSE_FLAGS:
        if decision.get(key) is not False or summary.get(key) is not False:
            failures.append("BOUNDARY_CLAIM_FAILURE")
    failures.extend(find_false_claims(json.dumps(summary) + "\n" + report_text))
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    _args = parser.parse_args()
    failures: list[str] = []
    missing, files = read_files()
    failures.extend(f"MISSING_FILE:{item}" for item in missing)
    combined_text = "\n".join(files.values())
    docs_text = "\n".join(files.get(rel, "") for rel in DOCS)
    for term in REQUIRED_TERMS:
        if term not in combined_text:
            failures.append(f"MISSING_REQUIRED_TERM:{term}")
    failures.extend(find_false_claims(docs_text))
    if runtime_surface_mutation_detected():
        failures.append("UNAUTHORIZED_REPO_MUTATION_DETECTED")
    failures.extend(ast_scan_source([REPO_ROOT / RUNNER, REPO_ROOT / CHECKER]))
    failures.extend(check_artifacts())
    if failures:
        print("138GA checker failed:")
        for failure in sorted(set(failures)):
            print(f" - {failure}")
        return 1
    print("138GA checker passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
