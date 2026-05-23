#!/usr/bin/env python3
"""Static/artifact checker for STABLE_LOOP_PHASE_LOCK_135D."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_135d_global_raw_evidence_rebuild_plan/smoke"
REQUIRED_DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_135D_GLOBAL_RAW_EVIDENCE_REBUILD_PLAN_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_135D_GLOBAL_RAW_EVIDENCE_REBUILD_PLAN_RESULT.md",
]
REQUIRED_SOURCE = [
    "scripts/probes/run_stable_loop_phase_lock_135d_global_raw_evidence_rebuild_plan.py",
    "scripts/probes/run_stable_loop_phase_lock_135d_global_raw_evidence_rebuild_plan_check.py",
]
ALLOWED_MUTATIONS = set(REQUIRED_DOCS + REQUIRED_SOURCE)
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "upstream_135b_manifest.json",
    "phase_rebuild_matrix.json",
    "claim_quarantine_map.json",
    "raw_generation_helper_requirements.json",
    "expected_output_canary_spec.json",
    "rebuild_sequence.json",
    "manual_review_plan.json",
    "future_checker_requirements.json",
    "evidence_recovery_risk_register.json",
    "decision.json",
    "summary.json",
    "report.md",
]
EXPECTED_COUNTS = {
    "ORACLE_SHORTCUT_DETECTED": 11,
    "DETERMINISTIC_HARNESS_ONLY": 3,
    "NEEDS_MANUAL_REVIEW": 17,
    "REAL_RAW_GENERATION_EVIDENCE": 4,
    "NOT_RAW_EVIDENCE_PHASE": 6,
}
ALLOWED_ACTIONS = {
    "keep_as_non_raw_evidence",
    "keep_as_harness_only",
    "manual_review_required",
    "rebuild_with_real_raw_generation",
    "invalidated_until_rebuilt",
    "retain_as_valid_raw_evidence",
}
REQUIRED_TERMS = [
    "STABLE_LOOP_PHASE_LOCK_135D_GLOBAL_RAW_EVIDENCE_REBUILD_PLAN",
    "planning-only",
    "GLOBAL_RAW_EVIDENCE_REBUILD_PLAN_COMPLETE",
    "PHASE_REBUILD_MATRIX_WRITTEN",
    "CLAIM_QUARANTINE_MAP_WRITTEN",
    "RAW_GENERATION_HELPER_REQUIREMENTS_WRITTEN",
    "EXPECTED_OUTPUT_CANARY_SPEC_WRITTEN",
    "135E_SHARED_RAW_GENERATION_HELPER_AND_CANARY_GATE",
    "136R_REAL_RAW_CORE_CAPABILITY_MINIMAL_REBUILD",
    "142R_REAL_RAW_CEILING_AND_GAP_REMAP",
    "raw assistant capability remains quarantined",
    "structured/tool capability remains invalidated as model evidence",
    "not GPT-like readiness",
    "not open-domain assistant readiness",
    "not production chat",
    "not public API",
    "not deployment readiness",
    "not safety alignment",
]
NEGATION_MARKERS = ["not ", "no ", "does not ", "do not ", "without ", "false", "forbidden", "blocked", "denial", "remains quarantined"]
FORBIDDEN_CLAIMS = {
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
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

    upstream = load_json(SMOKE_ROOT / "upstream_135b_manifest.json")
    if upstream.get("upstream_135b_verified") is not True:
        failures.append("UPSTREAM_135B_NOT_VERIFIED")
    if upstream.get("phase_count") != 41 or upstream.get("classification_counts") != EXPECTED_COUNTS:
        failures.append("UPSTREAM_135B_RECLASSIFICATION_MISMATCH")
    if upstream.get("stage_b_status") != "not_attempted_due_to_global_shortcut_audit":
        failures.append("UPSTREAM_135B_STAGE_B_MISMATCH")

    matrix = load_json(SMOKE_ROOT / "phase_rebuild_matrix.json")
    phases = matrix.get("phases", [])
    if matrix.get("source_of_truth") != "135B phase_evidence_reclassification.json":
        failures.append("UPSTREAM_135B_RECLASSIFICATION_MISMATCH")
    if matrix.get("phase_count") != 41 or len(phases) != 41:
        failures.append("PHASE_MATRIX_COUNT_MISMATCH")
    if Counter(row.get("current_classification") for row in phases) != EXPECTED_COUNTS:
        failures.append("UPSTREAM_135B_RECLASSIFICATION_MISMATCH")
    required_row_keys = {"phase", "file", "current_classification", "raw_model_evidence_status", "action_required", "rebuild_priority", "can_be_used_for_claims", "notes", "evidence_basis"}
    for row in phases:
        if set(row) != required_row_keys:
            failures.append("PHASE_REBUILD_MATRIX_INCOMPLETE")
        if row.get("action_required") not in ALLOWED_ACTIONS:
            failures.append("PHASE_REBUILD_MATRIX_INCOMPLETE")
        if not row.get("file") or not row.get("evidence_basis"):
            failures.append("PHASE_REBUILD_MATRIX_INCOMPLETE")
        if row.get("current_classification") in {"ORACLE_SHORTCUT_DETECTED", "DETERMINISTIC_HARNESS_ONLY", "NEEDS_MANUAL_REVIEW"} and row.get("can_be_used_for_claims") is not False:
            failures.append("RAW_CLAIM_NOT_QUARANTINED")

    claim_map = load_json(SMOKE_ROOT / "claim_quarantine_map.json")
    for key in ["bounded_local_private_release_stack", "raw_assistant_capability_track", "structured_tool_output_track", "gpt_like_open_domain_readiness"]:
        if key not in claim_map:
            failures.append("RELEASE_RAW_CLAIM_BOUNDARY_MISSING")
    if claim_map.get("raw_assistant_capability_track", {}).get("status") != "quarantined_pending_rebuild":
        failures.append("RAW_CLAIM_NOT_QUARANTINED")
    if claim_map.get("structured_tool_output_track", {}).get("status") != "invalidated_as_model_evidence":
        failures.append("STRUCTURED_TOOL_NOT_INVALIDATED")
    if claim_map.get("bounded_local_private_release_stack", {}).get("status") != "unaffected_unless_135b_directly_implicates_it":
        failures.append("RELEASE_RAW_CLAIM_BOUNDARY_MISSING")

    helper = load_json(SMOKE_ROOT / "raw_generation_helper_requirements.json")
    for key in ["expected_output", "expected_payload", "expected_answer", "required_keys", "scorer metadata", "labels", "oracle data"]:
        if key not in helper.get("forbidden_generation_inputs", []):
            failures.append("RAW_HELPER_REQUIREMENTS_INCOMPLETE")
    for key in ["prompt", "checkpoint_path", "checkpoint_hash", "seed", "max_new_tokens", "generation_config"]:
        if key not in helper.get("allowed_generation_inputs", []):
            failures.append("RAW_HELPER_REQUIREMENTS_INCOMPLETE")

    canary = load_json(SMOKE_ROOT / "expected_output_canary_spec.json")
    if canary.get("failure_verdict") != "ORACLE_SHORTCUT_DETECTED":
        failures.append("CANARY_SPEC_INCOMPLETE")
    if canary.get("acceptance", {}).get("generated_text_must_be_identical") is not True:
        failures.append("CANARY_SPEC_INCOMPLETE")

    sequence = load_json(SMOKE_ROOT / "rebuild_sequence.json")
    if sequence.get("do_not_continue_to_136_post_structured_tool_repair_ceiling_and_gap_remap") is not True:
        failures.append("136_NOT_BLOCKED")
    if sequence.get("correct_next") != "135E_SHARED_RAW_GENERATION_HELPER_AND_CANARY_GATE":
        failures.append("NEXT_MISMATCH")
    expected_sequence = [
        "135E_SHARED_RAW_GENERATION_HELPER_AND_CANARY_GATE",
        "136R_REAL_RAW_CORE_CAPABILITY_MINIMAL_REBUILD",
        "137R_REAL_RAW_REASONING_REBUILD",
        "138R_REAL_RAW_MULTI_TURN_STATE_REBUILD",
        "139R_REAL_RAW_HALLUCINATION_REFUSAL_REBUILD",
        "140R_REAL_RAW_INJECTION_PRIORITY_REBUILD",
        "141R_REAL_RAW_STRUCTURED_TOOL_REBUILD",
        "142R_REAL_RAW_CEILING_AND_GAP_REMAP",
    ]
    if sequence.get("sequence") != expected_sequence:
        failures.append("REBUILD_SEQUENCE_MISMATCH")

    manual = load_json(SMOKE_ROOT / "manual_review_plan.json")
    if manual.get("manual_review_phase_count") != EXPECTED_COUNTS["NEEDS_MANUAL_REVIEW"]:
        failures.append("MANUAL_REVIEW_PLAN_INCOMPLETE")
    if manual.get("manual_review_phases_can_be_used_for_claims") is not False:
        failures.append("MANUAL_REVIEW_PHASE_TRUSTED")
    for review in manual.get("reviews", []):
        for key in ["file", "reason_for_manual_review", "exact_inspection_checklist", "raw_generation_path_questions", "expected_output_oracle_shortcut_questions", "allowed_final_classifications", "required_reviewer_output_artifact"]:
            if not review.get(key):
                failures.append("MANUAL_REVIEW_PLAN_INCOMPLETE")

    future = load_json(SMOKE_ROOT / "future_checker_requirements.json")
    required_future_terms = [
        "AST scan for row[\"expected_output\"] in positive arm",
        "AST scan for expected_payload used in generation path",
        "AST scan for generated_text assigned from expected material",
        "checker verifies raw_generation_helper_provenance",
        "checker verifies expected-output canary pass",
        "checker verifies raw final eval flags",
        "checker rejects deterministic positive-arm construction",
    ]
    for term in required_future_terms:
        if term not in future.get("required_checks", []):
            failures.append("FUTURE_CHECKER_REQUIREMENTS_INCOMPLETE")
    if future.get("non_negotiable") is not True:
        failures.append("FUTURE_CHECKER_REQUIREMENTS_INCOMPLETE")

    decision = load_json(SMOKE_ROOT / "decision.json")
    if decision.get("decision") != "global_raw_evidence_rebuild_plan_complete":
        failures.append("DECISION_MISMATCH")
    if decision.get("next") != "135E_SHARED_RAW_GENERATION_HELPER_AND_CANARY_GATE":
        failures.append("NEXT_MISMATCH")
    for key in [
        "upstream_135b_verified",
        "phase_rebuild_matrix_written",
        "claim_quarantine_map_written",
        "raw_generation_helper_requirements_written",
        "expected_output_canary_spec_written",
        "rebuild_sequence_written",
        "manual_review_plan_written",
        "future_checker_requirements_written",
        "decision_written",
    ]:
        if decision.get(key) is not True:
            failures.append("HARD_GATE_NOT_WRITTEN")
    for key, expected in [
        ("train_step_count", 0),
        ("optimizer_step_count", 0),
        ("inference_run_count", 0),
        ("service_started", False),
        ("deployment_smoke_run", False),
        ("checkpoint_mutated", False),
        ("bounded_release_artifact_unchanged", True),
        ("runtime_surface_mutated", False),
        ("root_license_changed", False),
        ("repo_cleanup_performed", False),
        ("raw_assistant_capability_restored", False),
        ("structured_tool_capability_restored", False),
        ("gpt_like_readiness_claimed", False),
        ("open_domain_assistant_readiness_claimed", False),
        ("production_chat_claimed", False),
        ("public_api_claimed", False),
        ("deployment_readiness_claimed", False),
        ("safety_alignment_claimed", False),
    ]:
        if decision.get(key) != expected:
            failures.append("SIDE_EFFECT_OR_OVERCLAIM_DETECTED")

    summary = load_json(SMOKE_ROOT / "summary.json")
    verdicts = set(summary.get("verdicts", []))
    for verdict in [
        "GLOBAL_RAW_EVIDENCE_REBUILD_PLAN_COMPLETE",
        "UPSTREAM_135B_RECLASSIFICATION_VERIFIED",
        "RAW_ASSISTANT_CAPABILITY_REMAINS_QUARANTINED",
        "STRUCTURED_TOOL_CAPABILITY_REMAINS_INVALIDATED_AS_MODEL_EVIDENCE",
        "NO_MODEL_INFERENCE_PERFORMED",
    ]:
        if verdict not in verdicts:
            failures.append(f"VERDICT_MISSING:{verdict}")
    for key, expected in [
        ("planning_only", True),
        ("training_performed", False),
        ("repair_performed", False),
        ("model_inference_performed", False),
        ("checkpoint_mutated", False),
        ("runtime_surface_mutated", False),
        ("root_license_changed", False),
        ("repo_cleanup_performed", False),
        ("raw_assistant_capability_restored", False),
        ("structured_tool_capability_restored", False),
        ("gpt_like_readiness_claimed", False),
        ("open_domain_assistant_readiness_claimed", False),
        ("production_chat_claimed", False),
        ("public_api_claimed", False),
        ("deployment_readiness_claimed", False),
        ("safety_alignment_claimed", False),
    ]:
        if summary.get(key) != expected:
            failures.append("SIDE_EFFECT_OR_OVERCLAIM_DETECTED")

    progress_events = {row.get("event") for row in read_jsonl(SMOKE_ROOT / "progress.jsonl")}
    for event in [
        "startup",
        "upstream_verification",
        "phase_rebuild_matrix",
        "claim_quarantine_map",
        "raw_generation_helper_requirements",
        "expected_output_canary_spec",
        "rebuild_sequence",
        "manual_review_plan",
        "future_checker_requirements",
        "plan_validation",
        "decision_writing",
        "final_verdict",
    ]:
        if event not in progress_events:
            failures.append(f"PROGRESS_EVENT_MISSING:{event}")

    text_to_scan = "\n".join([(SMOKE_ROOT / "report.md").read_text(encoding="utf-8"), json.dumps(summary, sort_keys=True)])
    for phrase in [
        "135D is planning only",
        "raw assistant capability remains quarantined",
        "structured/tool capability remains invalidated as model evidence",
        "bounded local/private release remains separate",
        "no raw capability restored",
        "not GPT-like readiness",
        "not open-domain assistant readiness",
        "not production chat",
        "not public API",
        "not deployment readiness",
        "not safety alignment",
    ]:
        if phrase not in text_to_scan:
            failures.append("BOUNDARY_TEXT_MISSING")
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
        print("135D checker failed:")
        for failure in sorted(set(failures)):
            print(f" - {failure}")
        return 1
    print("135D checker passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
