#!/usr/bin/env python3
"""Static/artifact checker for STABLE_LOOP_PHASE_LOCK_135A."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_135a_structured_tool_raw_generation_audit/smoke"
REQUIRED_DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_135A_STRUCTURED_TOOL_RAW_GENERATION_AUDIT_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_135A_STRUCTURED_TOOL_RAW_GENERATION_AUDIT_RESULT.md",
]
REQUIRED_SOURCE = [
    "scripts/probes/run_stable_loop_phase_lock_135a_structured_tool_raw_generation_audit.py",
    "scripts/probes/run_stable_loop_phase_lock_135a_structured_tool_raw_generation_audit_check.py",
]
ALLOWED_MUTATIONS = set(REQUIRED_DOCS + REQUIRED_SOURCE)
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "audit_config.json",
    "source_code_audit.json",
    "positive_arm_generation_path_report.json",
    "artifact_trace_report.json",
    "oracle_shortcut_report.json",
    "evidence_reclassification.json",
    "decision.json",
    "summary.json",
    "report.md",
]
REQUIRED_TERMS = [
    "STABLE_LOOP_PHASE_LOCK_135A_STRUCTURED_TOOL_RAW_GENERATION_AUDIT",
    "audit-only",
    "STRUCTURED_TOOL_ORACLE_SHORTCUT_DETECTED",
    "STRUCTURED_OUTPUT_SCALE_CONFIRM_INVALID_AS_MODEL_EVIDENCE",
    "135B_STRUCTURED_TOOL_REAL_RAW_EVAL_REBUILD",
    "not GPT-like assistant readiness",
    "not open-domain assistant readiness",
    "not production chat",
    "not public API",
    "not deployment readiness",
    "not safety alignment",
    "not Hungarian assistant readiness",
]
NEGATION_MARKERS = ["not ", "no ", "does not ", "do not ", "without ", "false", "forbidden", "blocked", "denial"]
FORBIDDEN_CLAIMS = {
    "GPT_LIKE_READINESS_FALSE_CLAIM": ["GPT-like ready", "GPT-like assistant ready"],
    "OPEN_DOMAIN_ASSISTANT_CLAIM_DETECTED": ["open-domain assistant ready"],
    "PRODUCTION_CHAT_CLAIM_DETECTED": ["production chat ready", "production ready"],
    "PUBLIC_API_CLAIM_DETECTED": ["public API ready"],
    "DEPLOYMENT_CLAIM_DETECTED": ["deployment ready"],
    "SAFETY_ALIGNMENT_CLAIM_DETECTED": ["safety aligned"],
    "HUNGARIAN_ASSISTANT_CLAIM_DETECTED": ["Hungarian assistant ready"],
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
        window = lowered[max(0, match.start() - 220) : match.start()]
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

    config = load_json(SMOKE_ROOT / "audit_config.json")
    for key, expected in [
        ("audit_only", True),
        ("training_performed", False),
        ("repair_performed", False),
        ("inference_run_count", 0),
        ("checkpoint_mutated", False),
        ("runtime_surface_mutated", False),
    ]:
        if config.get(key) != expected:
            failures.append("AUDIT_SIDE_EFFECT_DETECTED")

    source = load_json(SMOKE_ROOT / "source_code_audit.json")
    if source.get("direct_expected_output_shortcut_detected") is not True:
        failures.append("EXPECTED_OUTPUT_SHORTCUT_NOT_DETECTED")
    findings = source.get("shortcut_findings", [])
    files = {finding.get("file") for finding in findings}
    if "scripts/probes/run_stable_loop_phase_lock_134_structured_output_tool_api_repair.py" not in files:
        failures.append("PHASE_134_SHORTCUT_NOT_DETECTED")
    if "scripts/probes/run_stable_loop_phase_lock_135_structured_output_tool_api_repair_scale_confirm.py" not in files:
        failures.append("PHASE_135_SHORTCUT_NOT_DETECTED")

    path_report = load_json(SMOKE_ROOT / "positive_arm_generation_path_report.json")
    for key, expected in [
        ("positive_arm_returns_expected_output_directly", True),
        ("expected_payload_used_in_generation_path", True),
        ("deterministic_answer_construction_instead_of_model_output", True),
        ("generated_text_exists_independently", False),
        ("generated_text_produced_by_model_raw_generation_function", False),
        ("expected_output_used_only_for_scoring", False),
    ]:
        if path_report.get(key) != expected:
            failures.append("POSITIVE_ARM_GENERATION_PATH_MISCLASSIFIED")

    oracle = load_json(SMOKE_ROOT / "oracle_shortcut_report.json")
    for key, expected in [
        ("structured_tool_oracle_shortcut_detected", True),
        ("direct_expected_output_return_in_positive_arm", True),
        ("oracle_metadata_used_during_final_eval", True),
        ("declared_raw_only_flags_conflict_with_source_path", True),
    ]:
        if oracle.get(key) != expected:
            failures.append("ORACLE_SHORTCUT_REPORT_MISCLASSIFIED")

    reclass = load_json(SMOKE_ROOT / "evidence_reclassification.json")
    decision = load_json(SMOKE_ROOT / "decision.json")
    summary = load_json(SMOKE_ROOT / "summary.json")
    for payload in [reclass, decision]:
        if payload.get("decision") != "structured_tool_oracle_shortcut_detected":
            failures.append("DECISION_MISCLASSIFIED")
        if payload.get("next") != "135B_STRUCTURED_TOOL_REAL_RAW_EVAL_REBUILD":
            failures.append("NEXT_MISMATCH")
        if payload.get("structured_output_scale_confirm_invalid_as_model_evidence") is not True:
            failures.append("EVIDENCE_NOT_INVALIDATED")
    if "STRUCTURED_TOOL_ORACLE_SHORTCUT_DETECTED" not in set(summary.get("verdicts", [])):
        failures.append("SHORTCUT_VERDICT_MISSING")
    if "STRUCTURED_OUTPUT_SCALE_CONFIRM_INVALID_AS_MODEL_EVIDENCE" not in set(summary.get("verdicts", [])):
        failures.append("INVALIDATION_VERDICT_MISSING")
    for key, expected in [
        ("audit_only", True),
        ("training_performed", False),
        ("repair_performed", False),
        ("inference_run_count", 0),
        ("checkpoint_mutated", False),
        ("runtime_surface_mutated", False),
        ("service_started", False),
        ("deployment_smoke_run", False),
        ("public_api_changed", False),
        ("gpt_like_assistant_readiness_claimed", False),
        ("open_domain_assistant_readiness_claimed", False),
        ("production_chat_claimed", False),
        ("public_api_claimed", False),
        ("deployment_readiness_claimed", False),
        ("safety_alignment_claimed", False),
        ("hungarian_assistant_readiness_claimed", False),
    ]:
        if summary.get(key) != expected:
            failures.append("AUDIT_SIDE_EFFECT_DETECTED")

    trace = load_json(SMOKE_ROOT / "artifact_trace_report.json")
    if len(trace.get("traces", [])) != 2:
        failures.append("ARTIFACT_TRACE_INCOMPLETE")
    progress_events = {row.get("event") for row in read_jsonl(SMOKE_ROOT / "progress.jsonl")}
    for event in ["startup", "source_code_audit_start", "source_code_audit_complete", "artifact_trace_start", "artifact_trace_complete", "decision_writing", "final_verdict"]:
        if event not in progress_events:
            failures.append(f"PROGRESS_EVENT_MISSING:{event}")

    text_to_scan = "\n".join([(SMOKE_ROOT / "report.md").read_text(encoding="utf-8"), json.dumps(summary, sort_keys=True)])
    for phrase in [
        "135A is audit-only",
        "not GPT-like assistant readiness",
        "not open-domain assistant readiness",
        "not production chat",
        "not public API",
        "not deployment readiness",
        "not safety alignment",
        "not Hungarian assistant readiness",
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
        print("135A checker failed:")
        for failure in sorted(set(failures)):
            print(f" - {failure}")
        return 1
    print("135A checker passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
