#!/usr/bin/env python3
"""Checker for STABLE_LOOP_PHASE_LOCK_137B real-raw reasoning repair plan."""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_137b_real_raw_reasoning_repair_plan/smoke"
RUNNER = "scripts/probes/run_stable_loop_phase_lock_137b_real_raw_reasoning_repair_plan.py"
CHECKER = "scripts/probes/run_stable_loop_phase_lock_137b_real_raw_reasoning_repair_plan_check.py"
REQUIRED_SOURCE = [RUNNER, CHECKER]
REQUIRED_DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_137B_REAL_RAW_REASONING_REPAIR_PLAN_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_137B_REAL_RAW_REASONING_REPAIR_PLAN_RESULT.md",
]
ALLOWED_MUTATIONS = set(REQUIRED_SOURCE + REQUIRED_DOCS)
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "upstream_137r_manifest.json",
    "upstream_136r_manifest.json",
    "reasoning_failure_diagnosis.json",
    "generation_quality_report.json",
    "scoring_mismatch_report.json",
    "checkpoint_capability_gap_report.json",
    "repair_option_matrix.json",
    "recommended_repair_target.json",
    "next_milestone_plan.json",
    "risk_register.json",
    "anti_shortcut_requirements.json",
    "decision.json",
    "summary.json",
    "report.md",
]
REQUIRED_TERMS = [
    "STABLE_LOOP_PHASE_LOCK_137B_REAL_RAW_REASONING_REPAIR_PLAN",
    "REAL_RAW_REASONING_REPAIR_PLAN_COMPLETE",
    "137R clean negative",
    "138R_REAL_RAW_REASONING_REPAIR_TRAINING_PLAN_OR_PROBE",
    "planning only",
    "no new inference",
    "reasoning is not restored",
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
    "raw_assistant_capability_restored",
    "structured_tool_capability_restored",
    "gpt_like_readiness_claimed",
    "open_domain_assistant_readiness_claimed",
    "production_chat_claimed",
    "public_api_claimed",
    "deployment_readiness_claimed",
    "safety_alignment_claimed",
]
NEGATION_MARKERS = ["not ", "no ", "does not ", "do not ", "without ", "false", "remains quarantined", "remains invalidated", "planning only"]
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
    return [line[3:].replace("\\", "/") for line in git_status().splitlines() if line.strip()]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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
    for rel in REQUIRED_SOURCE + REQUIRED_DOCS:
        path = REPO_ROOT / rel
        if not path.exists():
            missing.append(rel)
            continue
        text = path.read_text(encoding="utf-8")
        if rel in REQUIRED_DOCS and len(text.strip()) < 200:
            missing.append(f"{rel}:too_short")
        files[rel] = text
    return missing, files


def ast_scan_source(paths: list[Path]) -> list[str]:
    failures: list[str] = []
    forbidden_calls = {"raw_generate", "discover_backend", "load_checkpoint", "import_helper"}
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

        class Scanner(ast.NodeVisitor):
            def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
                module = node.module or ""
                if "shared_raw_generation_helper" in module:
                    failures.append("INFERENCE_SIDE_EFFECT_DETECTED")
                self.generic_visit(node)

            def visit_Import(self, node: ast.Import) -> None:
                for alias in node.names:
                    if "shared_raw_generation_helper" in alias.name:
                        failures.append("INFERENCE_SIDE_EFFECT_DETECTED")
                self.generic_visit(node)

            def visit_Call(self, node: ast.Call) -> None:
                name = ast.unparse(node.func)
                if any(token in name for token in forbidden_calls):
                    failures.append("INFERENCE_SIDE_EFFECT_DETECTED")
                self.generic_visit(node)

        Scanner().visit(tree)
    return failures


def check_artifacts() -> list[str]:
    failures: list[str] = []
    for rel in REQUIRED_ARTIFACTS:
        if not (SMOKE_ROOT / rel).exists():
            failures.append(f"MISSING_ARTIFACT:{rel}")
    if failures:
        return failures

    upstream_137r = load_json(SMOKE_ROOT / "upstream_137r_manifest.json")
    if upstream_137r.get("upstream_137r_clean_negative_verified") is not True:
        failures.append("UPSTREAM_137R_NOT_CLEAN_NEGATIVE")
    if upstream_137r.get("mean_real_raw_reasoning_accuracy") != 0.0:
        failures.append("UPSTREAM_137R_NOT_CLEAN_NEGATIVE")
    for key in ["canary_passed", "ast_scan_passed", "controls_failed", "leakage_rejected", "checkpoint_hash_unchanged", "no_expected_or_scorer_metadata_reached_generation"]:
        if upstream_137r.get(key) is not True:
            failures.append("UPSTREAM_137R_ARTIFACT_INCOMPLETE")

    upstream_136r = load_json(SMOKE_ROOT / "upstream_136r_manifest.json")
    if upstream_136r.get("upstream_136r_verified") is not True or upstream_136r.get("upstream_135e_verified") is not True or upstream_136r.get("upstream_135d_verified") is not True:
        failures.append("UPSTREAM_CHAIN_NOT_VERIFIED")

    quality = load_json(SMOKE_ROOT / "generation_quality_report.json")
    required_quality = [
        "row_count",
        "generated_text_exists_rate",
        "nonempty_rate",
        "token_count_min",
        "token_count_mean",
        "token_count_max",
        "unique_output_hash_count",
        "repeated_output_rate",
        "stale_user_assistant_fragment_rate",
        "expected_token_inclusion_rate",
        "numeric_expected_token_inclusion_rate",
        "prompt_copy_rate",
        "distractor_copy_rate",
        "refusal_fragment_rate",
        "policy_fragment_rate",
        "utf8_replacement_rate",
        "off_prompt_output_rate",
    ]
    for key in required_quality:
        if key not in quality:
            failures.append("GENERATION_QUALITY_REPORT_INCOMPLETE")
    if quality.get("row_count") != 960 or quality.get("artifact_derived_only") is not True:
        failures.append("GENERATION_QUALITY_REPORT_INVALID")

    scoring = load_json(SMOKE_ROOT / "scoring_mismatch_report.json")
    if scoring.get("controls_failed") is not True or scoring.get("leakage_rejected") is not True:
        failures.append("SCORING_MISMATCH_REPORT_INVALID")
    if scoring.get("expected_token_inclusion_rate", 1.0) > 0.01:
        failures.append("SCORING_MISMATCH_REPORT_INVALID")

    diagnosis = load_json(SMOKE_ROOT / "reasoning_failure_diagnosis.json")
    if diagnosis.get("helper_integrity_failure") is not False:
        failures.append("HELPER_INTEGRITY_MISDIAGNOSED")
    if diagnosis.get("leakage_eval_contamination") is not False:
        failures.append("LEAKAGE_MISDIAGNOSED")
    if diagnosis.get("checkpoint_model_capability_gap") is not True:
        failures.append("PRIMARY_DIAGNOSIS_MISSING")

    gap = load_json(SMOKE_ROOT / "checkpoint_capability_gap_report.json")
    if gap.get("checkpoint_capability_gap_likelihood") != "high":
        failures.append("CHECKPOINT_CAPABILITY_GAP_NOT_SELECTED")

    recommendation = load_json(SMOKE_ROOT / "recommended_repair_target.json")
    if recommendation.get("recommended_next") != "138R_REAL_RAW_REASONING_REPAIR_TRAINING_PLAN_OR_PROBE":
        failures.append("NEXT_MILESTONE_MISMATCH")

    next_plan = load_json(SMOKE_ROOT / "next_milestone_plan.json")
    if next_plan.get("milestone") != "138R_REAL_RAW_REASONING_REPAIR_TRAINING_PLAN_OR_PROBE":
        failures.append("NEXT_MILESTONE_MISMATCH")
    for key in ["raw_helper_final_eval", "generated_before_scoring_proof", "anti_oracle_canary", "ast_shortcut_scan", "leakage_audit", "clean_negative_accepted", "no_threshold_weakening_to_force_positive", "no_helper_alteration_to_improve_score"]:
        if next_plan.get(key) is not True:
            failures.append("NEXT_MILESTONE_GATES_INCOMPLETE")

    anti = load_json(SMOKE_ROOT / "anti_shortcut_requirements.json")
    for term in ["shared_raw_generation_helper.py only", "generated_text before scoring", "no LLM judge", "controls must fail", "clean negative accepted"]:
        if term not in anti.get("requirements", []):
            failures.append("ANTI_SHORTCUT_REQUIREMENTS_INCOMPLETE")

    decision = load_json(SMOKE_ROOT / "decision.json")
    if decision.get("decision") != "real_raw_reasoning_repair_plan_complete":
        failures.append("DECISION_MISMATCH")
    if decision.get("next") != "138R_REAL_RAW_REASONING_REPAIR_TRAINING_PLAN_OR_PROBE":
        failures.append("NEXT_MILESTONE_MISMATCH")
    for key in ["primary_diagnosis", "helper_integrity_status", "scorer_status", "leakage_status", "generation_quality_summary", "checkpoint_capability_gap_likelihood", "rejected_alternatives", "required_gates_for_next"]:
        if key not in decision:
            failures.append("DECISION_INCOMPLETE")
    if decision.get("inference_run_count") != 0 or decision.get("train_step_count") != 0 or decision.get("shared_helper_called_for_new_generation") is not False:
        failures.append("INFERENCE_SIDE_EFFECT_DETECTED")
    for key in FALSE_FLAGS:
        if decision.get(key) is not False:
            failures.append("BOUNDARY_CLAIM_FAILURE")

    summary = load_json(SMOKE_ROOT / "summary.json")
    if summary.get("planning_only") is not True or summary.get("new_model_inference_run") is not False:
        failures.append("PLANNING_ONLY_BOUNDARY_MISSING")
    report = (SMOKE_ROOT / "report.md").read_text(encoding="utf-8")
    failures.extend(find_false_claims(json.dumps(summary) + "\n" + report))
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    _args = parser.parse_args()
    failures: list[str] = []
    missing, files = read_files()
    failures.extend(f"MISSING_FILE:{item}" for item in missing)
    combined_text = "\n".join(files.values())
    docs_text = "\n".join(files.get(rel, "") for rel in REQUIRED_DOCS)
    for term in REQUIRED_TERMS:
        if term not in combined_text:
            failures.append(f"MISSING_REQUIRED_TERM:{term}")
    failures.extend(find_false_claims(docs_text))
    if runtime_surface_mutation_detected():
        failures.append("RUNTIME_SURFACE_MUTATION_DETECTED")
    failures.extend(ast_scan_source([REPO_ROOT / RUNNER, REPO_ROOT / CHECKER]))
    failures.extend(check_artifacts())
    if failures:
        print("137B checker failed:")
        for failure in sorted(set(failures)):
            print(f" - {failure}")
        return 1
    print("137B checker passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
