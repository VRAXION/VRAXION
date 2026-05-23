#!/usr/bin/env python3
"""Static/artifact checker for STABLE_LOOP_PHASE_LOCK_135B."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = REPO_ROOT / "target/pilot_wave/stable_loop_phase_lock_135b_global_raw_evidence_audit_and_structured_rebuild/smoke"
REQUIRED_DOCS = [
    "docs/research/STABLE_LOOP_PHASE_LOCK_135B_GLOBAL_RAW_EVIDENCE_AUDIT_AND_STRUCTURED_REBUILD_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_135B_GLOBAL_RAW_EVIDENCE_AUDIT_AND_STRUCTURED_REBUILD_RESULT.md",
]
REQUIRED_SOURCE = [
    "scripts/probes/run_stable_loop_phase_lock_135b_global_raw_evidence_audit_and_structured_rebuild.py",
    "scripts/probes/run_stable_loop_phase_lock_135b_global_raw_evidence_audit_and_structured_rebuild_check.py",
]
ALLOWED_MUTATIONS = set(REQUIRED_DOCS + REQUIRED_SOURCE)
REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "audit_rebuild_config.json",
    "upstream_135a_manifest.json",
    "quarantined_134_135_evidence_manifest.json",
    "source_code_shortcut_scan.json",
    "global_raw_evidence_audit.json",
    "phase_evidence_reclassification.json",
    "positive_arm_generation_path_report.json",
    "evidence_chain_impact_report.json",
    "raw_generation_helper_provenance.json",
    "expected_output_canary_report.json",
    "structured_tool_real_raw_dataset.jsonl",
    "real_raw_generation_trace.jsonl",
    "raw_generation_results.jsonl",
    "structured_tool_real_raw_rebuild_report.json",
    "oracle_shortcut_guard_report.json",
    "decision.json",
    "summary.json",
    "report.md",
]
REQUIRED_TERMS = [
    "STABLE_LOOP_PHASE_LOCK_135B_GLOBAL_RAW_EVIDENCE_AUDIT_AND_STRUCTURED_REBUILD",
    "Stage A",
    "Stage B",
    "GLOBAL_RAW_EVIDENCE_SHORTCUT_AUDIT_COMPLETED",
    "RAW_EVIDENCE_CHAIN_PARTIALLY_INVALIDATED",
    "135D_GLOBAL_RAW_EVIDENCE_REBUILD_PLAN",
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

    config = load_json(SMOKE_ROOT / "audit_rebuild_config.json")
    for key, expected in [
        ("stage_a_global_audit", True),
        ("training_performed", False),
        ("repair_performed", False),
        ("checkpoint_mutated", False),
        ("runtime_surface_mutated", False),
        ("public_api_changed", False),
    ]:
        if config.get(key) != expected:
            failures.append("SIDE_EFFECT_DETECTED")
    if config.get("seeds") != [2261, 2262, 2263, 2264, 2265] or config.get("eval_rows_per_family") != 96:
        failures.append("FULL_CONFIGURED_RUN_NOT_USED")

    upstream = load_json(SMOKE_ROOT / "upstream_135a_manifest.json")
    if upstream.get("positive_audit") is not True:
        failures.append("UPSTREAM_135A_NOT_VERIFIED")
    quarantine = load_json(SMOKE_ROOT / "quarantined_134_135_evidence_manifest.json")
    if quarantine.get("accepted_as_model_raw_generation_evidence") is not False:
        failures.append("QUARANTINE_NOT_ENFORCED")

    global_audit = load_json(SMOKE_ROOT / "global_raw_evidence_audit.json")
    if global_audit.get("broader_shortcuts_found") is not True:
        failures.append("GLOBAL_SHORTCUTS_NOT_DETECTED")
    if global_audit.get("oracle_shortcut_phase_count", 0) <= 0:
        failures.append("ORACLE_SHORTCUT_PHASES_MISSING")

    reclass = load_json(SMOKE_ROOT / "phase_evidence_reclassification.json")
    phases = reclass.get("phases", [])
    if not phases:
        failures.append("PHASE_RECLASSIFICATION_EMPTY")
    if any(row.get("classification") == "REAL_RAW_GENERATION_EVIDENCE" and row.get("invalid_as_raw_model_evidence") for row in phases):
        failures.append("REAL_RAW_PHASE_MARKED_INVALID")
    if not any(row.get("invalid_as_raw_model_evidence") for row in phases):
        failures.append("NO_PHASE_INVALIDATED")

    source = load_json(SMOKE_ROOT / "source_code_shortcut_scan.json")
    if source.get("scanned_files", 0) < 10:
        failures.append("SOURCE_SCAN_TOO_SMALL")
    if "ORACLE_SHORTCUT_DETECTED" not in source.get("classifications", {}):
        failures.append("ORACLE_SHORTCUT_CLASSIFICATION_MISSING")

    impact = load_json(SMOKE_ROOT / "evidence_chain_impact_report.json")
    if impact.get("global_rebuild_plan_required") is not True:
        failures.append("GLOBAL_REBUILD_NOT_REQUIRED")
    helper = load_json(SMOKE_ROOT / "raw_generation_helper_provenance.json")
    if helper.get("fake_helper_used") is not False or helper.get("simulated_model_output_used") is not False:
        failures.append("FAKE_HELPER_USED")
    if helper.get("stage_b_status") != "not_attempted_due_to_global_shortcut_audit":
        failures.append("STAGE_B_STATUS_MISMATCH")
    rebuild = load_json(SMOKE_ROOT / "structured_tool_real_raw_rebuild_report.json")
    if rebuild.get("attempted") is not False or rebuild.get("real_raw_generation_used") is not False:
        failures.append("STRUCTURED_REBUILD_SHOULD_NOT_RUN")
    canary = load_json(SMOKE_ROOT / "expected_output_canary_report.json")
    if canary.get("canary_built") is not True or canary.get("canary_run") is not False:
        failures.append("CANARY_STATUS_MISMATCH")
    guard = load_json(SMOKE_ROOT / "oracle_shortcut_guard_report.json")
    for key in ["oracle_rerank_used", "verifier_used", "llm_judge_used", "teacher_forcing_used", "constrained_decoding_used", "json_mode_used", "retry_loop_used", "best_of_n_used", "actual_tool_execution_used", "runtime_tool_call_used"]:
        if guard.get(key) is not False:
            failures.append("FORBIDDEN_GENERATION_PATH_USED")

    if read_jsonl(SMOKE_ROOT / "real_raw_generation_trace.jsonl"):
        failures.append("REAL_RAW_TRACE_SHOULD_BE_EMPTY")
    if read_jsonl(SMOKE_ROOT / "raw_generation_results.jsonl"):
        failures.append("RAW_RESULTS_SHOULD_BE_EMPTY")
    if not read_jsonl(SMOKE_ROOT / "structured_tool_real_raw_dataset.jsonl"):
        failures.append("STRUCTURED_DATASET_MISSING")

    decision = load_json(SMOKE_ROOT / "decision.json")
    if decision.get("decision") != "raw_evidence_chain_partially_invalidated":
        failures.append("DECISION_MISMATCH")
    if decision.get("next") != "135D_GLOBAL_RAW_EVIDENCE_REBUILD_PLAN":
        failures.append("NEXT_MISMATCH")
    if decision.get("stage_b_status") != "not_attempted_due_to_global_shortcut_audit":
        failures.append("STAGE_B_DECISION_MISMATCH")

    summary = load_json(SMOKE_ROOT / "summary.json")
    verdicts = set(summary.get("verdicts", []))
    for verdict in ["GLOBAL_RAW_EVIDENCE_SHORTCUT_AUDIT_COMPLETED", "RAW_EVIDENCE_CHAIN_PARTIALLY_INVALIDATED", "STRUCTURED_REBUILD_NOT_ATTEMPTED_DUE_TO_GLOBAL_AUDIT"]:
        if verdict not in verdicts:
            failures.append(f"VERDICT_MISSING:{verdict}")
    for key, expected in [
        ("audit_rebuild_only", True),
        ("training_performed", False),
        ("repair_performed", False),
        ("checkpoint_mutated", False),
        ("runtime_surface_mutated", False),
        ("service_started", False),
        ("deployment_smoke_run", False),
        ("public_api_changed", False),
        ("root_license_changed", False),
        ("fake_helper_used", False),
        ("simulated_model_output_used", False),
        ("actual_tool_execution_used", False),
        ("runtime_tool_call_used", False),
        ("gpt_like_assistant_readiness_claimed", False),
        ("open_domain_assistant_readiness_claimed", False),
        ("production_chat_claimed", False),
        ("public_api_claimed", False),
        ("deployment_readiness_claimed", False),
        ("safety_alignment_claimed", False),
        ("hungarian_assistant_readiness_claimed", False),
    ]:
        if summary.get(key) != expected:
            failures.append("SIDE_EFFECT_OR_OVERCLAIM_DETECTED")

    progress_events = {row.get("event") for row in read_jsonl(SMOKE_ROOT / "progress.jsonl")}
    for event in ["startup", "upstream_verification", "source_scan_start", "source_scan_complete", "phase_classification", "evidence_reclassification", "helper_provenance", "canary_setup", "raw_generation_block", "scoring", "aggregate_analysis", "decision_writing", "final_verdict"]:
        if event not in progress_events:
            failures.append(f"PROGRESS_EVENT_MISSING:{event}")

    text_to_scan = "\n".join([(SMOKE_ROOT / "report.md").read_text(encoding="utf-8"), json.dumps(summary, sort_keys=True)])
    for phrase in ["135B is audit/rebuild only", "not GPT-like assistant readiness", "not open-domain assistant readiness", "not production chat", "not public API", "not deployment readiness", "not safety alignment", "not Hungarian assistant readiness"]:
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
        print("135B checker failed:")
        for failure in sorted(set(failures)):
            print(f" - {failure}")
        return 1
    print("135B checker passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
