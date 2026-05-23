#!/usr/bin/env python3
"""135D global raw-evidence rebuild plan.

This planning-only milestone consumes the 135B global raw-evidence audit and
writes the phase-by-phase rebuild plan that must precede any renewed raw
assistant capability claims. It does not train, run model inference, repair,
mutate checkpoints, start services, deploy, delete old evidence, or consolidate
phase runners.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "STABLE_LOOP_PHASE_LOCK_135D_GLOBAL_RAW_EVIDENCE_REBUILD_PLAN"
DEFAULT_OUT = Path("target/pilot_wave/stable_loop_phase_lock_135d_global_raw_evidence_rebuild_plan/smoke")
DEFAULT_UPSTREAM_135B_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_135b_global_raw_evidence_audit_and_structured_rebuild/smoke")

EXPECTED_COUNTS = {
    "ORACLE_SHORTCUT_DETECTED": 11,
    "DETERMINISTIC_HARNESS_ONLY": 3,
    "NEEDS_MANUAL_REVIEW": 17,
    "REAL_RAW_GENERATION_EVIDENCE": 4,
    "NOT_RAW_EVIDENCE_PHASE": 6,
}
EXPECTED_PHASE_COUNT = 41
SUCCESS_DECISION = "global_raw_evidence_rebuild_plan_complete"
SUCCESS_NEXT = "135E_SHARED_RAW_GENERATION_HELPER_AND_CANARY_GATE"
BOUNDARY_TEXT = (
    "135D is planning only. Raw assistant capability remains quarantined. "
    "Structured/tool capability remains invalidated as model evidence. The "
    "bounded local/private release stack remains separate unless directly "
    "implicated by 135B. 135D restores no raw capability, performs no training, "
    "runs no model inference, performs no repair, mutates no checkpoints, starts "
    "no services, deploys nothing, deletes nothing, consolidates nothing, and "
    "does not modify runtime or release surfaces. It is not GPT-like readiness, "
    "not open-domain assistant readiness, not production chat, not public API, "
    "not deployment readiness, and not safety alignment."
)

ACTION_BY_CLASSIFICATION = {
    "ORACLE_SHORTCUT_DETECTED": "invalidated_until_rebuilt",
    "DETERMINISTIC_HARNESS_ONLY": "keep_as_harness_only",
    "NEEDS_MANUAL_REVIEW": "manual_review_required",
    "REAL_RAW_GENERATION_EVIDENCE": "retain_as_valid_raw_evidence",
    "NOT_RAW_EVIDENCE_PHASE": "keep_as_non_raw_evidence",
}
ALLOWED_ACTIONS = sorted(set(ACTION_BY_CLASSIFICATION.values()) | {"rebuild_with_real_raw_generation"})


class GateError(Exception):
    def __init__(self, verdict: str, message: str):
        super().__init__(message)
        self.verdict = verdict
        self.message = message


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    tmp.replace(path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def resolve_target_out(text: str) -> Path:
    path = Path(text)
    if path.is_absolute() or any(part == ".." for part in path.parts):
        raise GateError("GLOBAL_RAW_EVIDENCE_REBUILD_PLAN_FAILS", "--out must be repo-relative")
    parts = [part.lower() for part in path.parts]
    if len(parts) < 2 or parts[0] != "target" or parts[1] != "pilot_wave":
        raise GateError("GLOBAL_RAW_EVIDENCE_REBUILD_PLAN_FAILS", "--out must stay under target/pilot_wave")
    return (REPO_ROOT / path).resolve()


def resolve_path(text: str) -> Path:
    path = Path(text)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def append_progress(out: Path, event: str, status: str = "completed", **details: Any) -> None:
    append_jsonl(out / "progress.jsonl", {"ts": utc_now(), "event": event, "status": status, "details": details})


def write_summary(out: Path, status: str, verdicts: list[str], decision: dict[str, Any], failure: str | None = None) -> None:
    write_json(
        out / "summary.json",
        {
            "schema_version": "phase_135d_summary_v1",
            "milestone": MILESTONE,
            "status": status,
            "verdicts": verdicts,
            "decision": decision.get("decision"),
            "next": decision.get("next"),
            "failure": failure,
            "planning_only": True,
            "training_performed": False,
            "repair_performed": False,
            "model_inference_performed": False,
            "train_step_count": 0,
            "optimizer_step_count": 0,
            "inference_run_count": 0,
            "service_started": False,
            "deployment_smoke_run": False,
            "checkpoint_mutated": False,
            "bounded_release_artifact_unchanged": True,
            "runtime_surface_mutated": False,
            "root_license_changed": False,
            "repo_cleanup_performed": False,
            "raw_assistant_capability_restored": False,
            "structured_tool_capability_restored": False,
            "gpt_like_readiness_claimed": False,
            "open_domain_assistant_readiness_claimed": False,
            "production_chat_claimed": False,
            "public_api_claimed": False,
            "deployment_readiness_claimed": False,
            "safety_alignment_claimed": False,
            "boundary": BOUNDARY_TEXT,
            "metrics": decision,
        },
    )


def write_report(out: Path, verdicts: list[str], decision: dict[str, Any]) -> None:
    lines = [
        f"# {MILESTONE}",
        "",
        "## Boundary",
        "",
        BOUNDARY_TEXT,
        "",
        "## Verdicts",
        "",
    ]
    lines.extend(f"- `{verdict}`" for verdict in verdicts)
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- `decision`: `{decision.get('decision')}`",
            f"- `next`: `{decision.get('next')}`",
            f"- `phase_count`: `{decision.get('phase_count')}`",
            f"- `raw_assistant_capability_restored`: `false`",
            f"- `structured_tool_capability_restored`: `false`",
            "",
            "## Claims",
            "",
            "- raw assistant capability remains quarantined",
            "- structured/tool capability remains invalidated as model evidence",
            "- bounded local/private release remains separate",
            "- no raw capability restored",
            "- not GPT-like readiness",
            "- not open-domain assistant readiness",
            "- not production chat",
            "- not public API",
            "- not deployment readiness",
            "- not safety alignment",
        ]
    )
    write_text(out / "report.md", "\n".join(lines) + "\n")


def refresh_status(out: Path, status: str, verdicts: list[str], decision: dict[str, Any]) -> None:
    write_summary(out, status, verdicts, decision)
    write_report(out, verdicts, decision)


def phase_id_from_file(path_text: str) -> str:
    match = re.search(r"phase_lock_(\d{3}[a-z]*)_", path_text)
    return match.group(1).upper() if match else "UNKNOWN"


def load_required_json(root: Path, name: str) -> dict[str, Any]:
    path = root / name
    if not path.exists():
        raise GateError("UPSTREAM_135B_ARTIFACT_MISSING", f"missing {rel(path)}")
    return read_json(path)


def verify_upstream_135b(out: Path, root: Path) -> dict[str, Any]:
    decision = load_required_json(root, "decision.json")
    global_audit = load_required_json(root, "global_raw_evidence_audit.json")
    reclass = load_required_json(root, "phase_evidence_reclassification.json")
    source_scan = load_required_json(root, "source_code_shortcut_scan.json")
    path_report = load_required_json(root, "positive_arm_generation_path_report.json")
    impact = load_required_json(root, "evidence_chain_impact_report.json")
    summary = load_required_json(root, "summary.json")

    if decision.get("decision") != "raw_evidence_chain_partially_invalidated":
        raise GateError("UPSTREAM_135B_RECLASSIFICATION_MISMATCH", "135B decision is not the required partial invalidation")
    if decision.get("next") != "135D_GLOBAL_RAW_EVIDENCE_REBUILD_PLAN":
        raise GateError("UPSTREAM_135B_RECLASSIFICATION_MISMATCH", "135B next does not route to 135D")
    if decision.get("stage_b_status") != "not_attempted_due_to_global_shortcut_audit":
        raise GateError("UPSTREAM_135B_RECLASSIFICATION_MISMATCH", "135B Stage B status is not the required global-audit block")

    phases = reclass.get("phases", [])
    counts = Counter(row.get("classification") for row in phases)
    if len(phases) != EXPECTED_PHASE_COUNT:
        raise GateError("PHASE_MATRIX_COUNT_MISMATCH", f"expected {EXPECTED_PHASE_COUNT} phases, found {len(phases)}")
    if dict(counts) != EXPECTED_COUNTS:
        raise GateError("UPSTREAM_135B_RECLASSIFICATION_MISMATCH", f"classification counts mismatch: {dict(counts)}")
    if global_audit.get("classification_counts") != EXPECTED_COUNTS:
        raise GateError("UPSTREAM_135B_RECLASSIFICATION_MISMATCH", "global audit classification counts mismatch")
    if global_audit.get("scanned_phase_count") != EXPECTED_PHASE_COUNT:
        raise GateError("PHASE_MATRIX_COUNT_MISMATCH", "global audit phase count mismatch")
    if impact.get("structured_output_tool_api_status") != "INVALIDATED_AS_MODEL_EVIDENCE_BY_135A":
        raise GateError("UPSTREAM_135B_RECLASSIFICATION_MISMATCH", "structured/tool status was not invalidated by 135A")

    manifest = {
        "schema_version": "phase_135d_upstream_135b_manifest_v1",
        "root": rel(root),
        "decision_sha256": file_hash(root / "decision.json"),
        "global_raw_evidence_audit_sha256": file_hash(root / "global_raw_evidence_audit.json"),
        "phase_evidence_reclassification_sha256": file_hash(root / "phase_evidence_reclassification.json"),
        "summary_sha256": file_hash(root / "summary.json"),
        "upstream_135b_verified": True,
        "decision": decision.get("decision"),
        "next": decision.get("next"),
        "stage_b_status": decision.get("stage_b_status"),
        "phase_count": len(phases),
        "classification_counts": dict(counts),
        "expected_counts": EXPECTED_COUNTS,
    }
    write_json(out / "upstream_135b_manifest.json", manifest)
    return {
        "decision": decision,
        "global_audit": global_audit,
        "reclass": reclass,
        "source_scan": source_scan,
        "path_report": path_report,
        "impact": impact,
        "summary": summary,
        "manifest": manifest,
    }


def evidence_basis_for(phase_file: str, source_scan: dict[str, Any], path_report: dict[str, Any]) -> dict[str, Any]:
    source_rows = {row.get("phase_file"): row for row in source_scan.get("files", [])}
    path_rows = {row.get("phase_file"): row for row in path_report.get("paths", [])}
    source = source_rows.get(phase_file, {})
    path = path_rows.get(phase_file, {})
    return {
        "source_code_shortcut_scan_classification": source.get("classification"),
        "finding_count": source.get("finding_count"),
        "declared_raw_only_flags_conflict_with_source_path": source.get("declared_raw_only_flags_conflict_with_source_path"),
        "positive_arm_path_reported": bool(path),
        "critical_finding_count": len(path.get("critical_findings", [])) if path else 0,
    }


def build_phase_matrix(reclass: dict[str, Any], source_scan: dict[str, Any], path_report: dict[str, Any]) -> list[dict[str, Any]]:
    matrix: list[dict[str, Any]] = []
    for row in reclass.get("phases", []):
        classification = row.get("classification")
        phase_file = row.get("phase_file")
        action = ACTION_BY_CLASSIFICATION.get(classification)
        if action is None:
            raise GateError("PHASE_REBUILD_MATRIX_INCOMPLETE", f"unknown classification {classification!r}")
        if classification == "ORACLE_SHORTCUT_DETECTED":
            raw_status = "invalidated_as_raw_model_evidence"
            priority = "P0"
            can_claim = False
            notes = "Expected-output/oracle shortcut detected; rebuild with shared real raw helper before any raw claim."
        elif classification == "DETERMINISTIC_HARNESS_ONLY":
            raw_status = "harness_only_not_raw_model_evidence"
            priority = "P1"
            can_claim = False
            notes = "Deterministic harness evidence may remain as harness evidence only."
        elif classification == "NEEDS_MANUAL_REVIEW":
            raw_status = "quarantined_pending_manual_review"
            priority = "P1"
            can_claim = False
            notes = "Manual inspection required before any raw model evidence claim."
        elif classification == "REAL_RAW_GENERATION_EVIDENCE":
            raw_status = "retained_as_existing_raw_evidence_subject_to_new_helper_canary_consistency"
            priority = "P2"
            can_claim = True
            notes = "135B found no shortcut, but future rebuild still standardizes helper provenance and canary gates."
        else:
            raw_status = "not_raw_evidence_phase"
            priority = "P3"
            can_claim = False
            notes = "Keep separate from raw assistant capability claims."
        matrix.append(
            {
                "phase": phase_id_from_file(str(phase_file)),
                "file": phase_file,
                "current_classification": classification,
                "raw_model_evidence_status": raw_status,
                "action_required": action,
                "rebuild_priority": priority,
                "can_be_used_for_claims": can_claim,
                "notes": notes,
                "evidence_basis": {
                    "phase_evidence_reclassification": {
                        "finding_count": row.get("finding_count"),
                        "invalid_as_raw_model_evidence": row.get("invalid_as_raw_model_evidence"),
                        "declared_raw_only_flags_conflict_with_source_path": row.get("declared_raw_only_flags_conflict_with_source_path"),
                    },
                    **evidence_basis_for(str(phase_file), source_scan, path_report),
                },
            }
        )
    return matrix


def build_claim_quarantine_map() -> dict[str, Any]:
    return {
        "schema_version": "phase_135d_claim_quarantine_map_v1",
        "bounded_local_private_release_stack": {
            "status": "unaffected_unless_135b_directly_implicates_it",
            "can_be_used_for_claims": True,
            "notes": "Release-stack evidence remains separated from raw assistant capability evidence.",
        },
        "bounded_release_stack": {
            "status": "unaffected_unless_135b_directly_implicates_it",
            "can_be_used_for_claims": True,
            "notes": "No 135D artifact restores or weakens bounded release claims.",
        },
        "raw_assistant_capability_track": {
            "status": "quarantined_pending_rebuild",
            "can_be_used_for_claims": False,
            "next_required_step": "135E_SHARED_RAW_GENERATION_HELPER_AND_CANARY_GATE",
        },
        "structured_tool_output_track": {
            "status": "invalidated_as_model_evidence",
            "can_be_used_for_claims": False,
            "reason": "135A/135B detected expected-output shortcut contamination.",
        },
        "gpt_like_open_domain_readiness": {
            "status": "not_claimable_pending_rebuild",
            "can_be_used_for_claims": False,
        },
        "valid_claims": [
            "bounded local/private release stack remains separately tracked",
            "135D rebuild plan completeness if all plan artifacts pass",
        ],
        "quarantined_claims": [
            "raw assistant capability track",
            "post-114 raw capability chain pending rebuild",
            "GPT-like/open-domain readiness pending rebuild",
        ],
        "invalidated_claims": [
            "134 structured/tool repair as raw model evidence",
            "135 structured/tool scale confirm as raw model evidence",
        ],
        "unaffected_release_claims": [
            "bounded local/private release stack unless directly implicated by future audit evidence",
        ],
        "needs_manual_review_claims": [
            "any phase classified NEEDS_MANUAL_REVIEW in 135B",
        ],
    }


def build_raw_helper_requirements() -> dict[str, Any]:
    return {
        "schema_version": "phase_135d_raw_generation_helper_requirements_v1",
        "mandatory_for_future_raw_capability_milestones": True,
        "allowed_generation_inputs": [
            "prompt",
            "checkpoint_path",
            "checkpoint_hash",
            "seed",
            "max_new_tokens",
            "generation_config",
        ],
        "forbidden_generation_inputs": [
            "expected_output",
            "expected_payload",
            "expected_answer",
            "required_keys",
            "required_keywords",
            "forbidden_outputs",
            "schema answer object",
            "scorer metadata",
            "labels",
            "oracle data",
        ],
        "required_outputs": [
            "generated_text",
            "token_count",
            "stop_reason",
            "generation_trace_hash",
            "model_checkpoint_hash",
            "generation_config_hash",
        ],
        "forbidden_generation_paths": [
            "expected_output return",
            "expected_payload construction",
            "deterministic positive-arm answer construction",
            "oracle rerank",
            "verifier rerank",
            "LLM judge",
            "teacher forcing",
            "constrained decoding",
            "JSON mode",
            "grammar decoder",
            "regex fixer",
            "retry loop",
            "best-of-n",
        ],
    }


def build_canary_spec() -> dict[str, Any]:
    return {
        "schema_version": "phase_135d_expected_output_canary_spec_v1",
        "required": True,
        "procedure": [
            "duplicate eval row",
            "corrupt expected_output",
            "corrupt expected_payload",
            "keep prompt identical",
            "run generation through same helper",
            "compare generated_text hashes",
        ],
        "acceptance": {
            "generated_text_must_be_identical": True,
            "scoring_may_change": True,
            "generation_may_not_change": True,
        },
        "failure_verdict": "ORACLE_SHORTCUT_DETECTED",
    }


def build_manual_review_plan(matrix: list[dict[str, Any]]) -> dict[str, Any]:
    reviews = []
    for row in matrix:
        if row["current_classification"] != "NEEDS_MANUAL_REVIEW":
            continue
        reviews.append(
            {
                "file": row["file"],
                "reason_for_manual_review": "135B static audit found no direct shortcut but could not prove shared real raw generation provenance.",
                "exact_inspection_checklist": [
                    "trace positive-arm generation path from prompt to generated_text",
                    "confirm expected_output, expected_payload, expected_answer, labels, and scorer metadata are unavailable before generation",
                    "confirm no deterministic row-label answer construction",
                    "confirm raw final eval flags match the source path",
                    "confirm helper provenance and checkpoint hash are recorded",
                    "confirm expected-output canary exists and passes",
                ],
                "raw_generation_path_questions": [
                    "which function produces generated_text",
                    "what inputs are passed into that function",
                    "is checkpoint-backed autoregressive generation actually invoked",
                    "is the output post-processed before scoring",
                ],
                "expected_output_oracle_shortcut_questions": [
                    "can changing expected_output change generated_text",
                    "can expected_payload influence generation",
                    "are required keys or schema answers available to the generator",
                    "does any positive arm return or format expected material directly",
                ],
                "allowed_final_classifications": [
                    "REAL_RAW_GENERATION_EVIDENCE",
                    "DETERMINISTIC_HARNESS_ONLY",
                    "ORACLE_SHORTCUT_DETECTED",
                    "NOT_RAW_EVIDENCE_PHASE",
                ],
                "required_reviewer_output_artifact": f"manual_review_{row['phase'].lower()}_raw_generation_path.json",
                "can_be_used_for_raw_claims_until_reviewed": False,
            }
        )
    return {
        "schema_version": "phase_135d_manual_review_plan_v1",
        "manual_review_phase_count": len(reviews),
        "manual_review_phases_can_be_used_for_claims": False,
        "reviews": reviews,
    }


def build_future_checker_requirements() -> dict[str, Any]:
    return {
        "schema_version": "phase_135d_future_checker_requirements_v1",
        "mandatory_for_future_raw_capability_milestones": True,
        "required_checks": [
            "AST scan for row[\"expected_output\"] in positive arm",
            "AST scan for expected_payload used in generation path",
            "AST scan for generated_text assigned from expected material",
            "AST scan for positive-arm deterministic construction",
            "checker verifies raw_generation_helper_provenance",
            "checker verifies expected-output canary pass",
            "checker verifies raw final eval flags",
            "checker rejects deterministic positive-arm construction",
        ],
        "future_raw_milestone_must_have": [
            "raw_generation_helper_provenance exists",
            "expected_output_canary passes",
            "AST scan passes",
            "raw final eval flags pass",
            "positive-arm deterministic construction rejected",
        ],
        "non_negotiable": True,
    }


def build_rebuild_sequence() -> dict[str, Any]:
    sequence = [
        "135E_SHARED_RAW_GENERATION_HELPER_AND_CANARY_GATE",
        "136R_REAL_RAW_CORE_CAPABILITY_MINIMAL_REBUILD",
        "137R_REAL_RAW_REASONING_REBUILD",
        "138R_REAL_RAW_MULTI_TURN_STATE_REBUILD",
        "139R_REAL_RAW_HALLUCINATION_REFUSAL_REBUILD",
        "140R_REAL_RAW_INJECTION_PRIORITY_REBUILD",
        "141R_REAL_RAW_STRUCTURED_TOOL_REBUILD",
        "142R_REAL_RAW_CEILING_AND_GAP_REMAP",
    ]
    return {
        "schema_version": "phase_135d_rebuild_sequence_v1",
        "blocked_milestone": "136_POST_STRUCTURED_TOOL_REPAIR_CEILING_AND_GAP_REMAP",
        "do_not_continue_to_136_post_structured_tool_repair_ceiling_and_gap_remap": True,
        "correct_next": sequence[0],
        "sequence": sequence,
        "reason": "Raw evidence chain must be rebuilt through shared helper and canary gates before new ceiling remaps.",
    }


def build_risk_register() -> dict[str, Any]:
    return {
        "schema_version": "phase_135d_evidence_recovery_risk_register_v1",
        "risks": [
            {
                "risk": "future milestone reintroduces expected-output shortcut",
                "mitigation": "AST scanner and expected-output canary are mandatory gates",
                "severity": "high",
            },
            {
                "risk": "manual review phases are treated as real raw evidence too early",
                "mitigation": "manual review phases cannot support claims until reviewer artifacts exist",
                "severity": "high",
            },
            {
                "risk": "bounded release and raw assistant claims are conflated",
                "mitigation": "claim quarantine map separates release stack from raw capability evidence",
                "severity": "medium",
            },
            {
                "risk": "old evidence is deleted before rebuild is complete",
                "mitigation": "135D forbids cleanup, deletion, and consolidation",
                "severity": "medium",
            },
        ],
    }


def validate_plan_artifacts(out: Path, matrix: list[dict[str, Any]], claim_map: dict[str, Any], manual: dict[str, Any]) -> None:
    if len(matrix) != EXPECTED_PHASE_COUNT:
        raise GateError("PHASE_MATRIX_COUNT_MISMATCH", "phase rebuild matrix row count mismatch")
    if Counter(row["current_classification"] for row in matrix) != EXPECTED_COUNTS:
        raise GateError("UPSTREAM_135B_RECLASSIFICATION_MISMATCH", "phase rebuild matrix classification counts mismatch")
    required_keys = {"phase", "file", "current_classification", "raw_model_evidence_status", "action_required", "rebuild_priority", "can_be_used_for_claims", "notes", "evidence_basis"}
    for row in matrix:
        if set(row) != required_keys or any(row[key] in (None, "", []) for key in required_keys if key != "can_be_used_for_claims"):
            raise GateError("PHASE_REBUILD_MATRIX_INCOMPLETE", f"incomplete matrix row for {row.get('file')}")
        if row["action_required"] not in ALLOWED_ACTIONS:
            raise GateError("PHASE_REBUILD_MATRIX_INCOMPLETE", f"invalid action {row['action_required']}")
    for key in ["bounded_local_private_release_stack", "raw_assistant_capability_track", "structured_tool_output_track", "gpt_like_open_domain_readiness"]:
        if key not in claim_map:
            raise GateError("RELEASE_RAW_CLAIM_BOUNDARY_MISSING", f"claim map missing {key}")
    if manual.get("manual_review_phase_count") != EXPECTED_COUNTS["NEEDS_MANUAL_REVIEW"]:
        raise GateError("PHASE_REBUILD_MATRIX_INCOMPLETE", "manual review phase count mismatch")
    forbidden_cleanup_targets = ["scripts/probes", "docs/research", "target/pilot_wave"]
    cleanup_plan_text = json.dumps(build_risk_register(), sort_keys=True)
    if "delete old runners" in cleanup_plan_text.lower():
        raise GateError("UNAUTHORIZED_REPO_CLEANUP_DETECTED", "cleanup/deletion detected")
    for artifact in [
        "phase_rebuild_matrix.json",
        "claim_quarantine_map.json",
        "raw_generation_helper_requirements.json",
        "expected_output_canary_spec.json",
        "rebuild_sequence.json",
        "manual_review_plan.json",
        "future_checker_requirements.json",
        "evidence_recovery_risk_register.json",
    ]:
        if not (out / artifact).exists():
            raise GateError("REBUILD_PLAN_INCOMPLETE", f"missing {artifact}")


def run(args: argparse.Namespace) -> None:
    out = resolve_target_out(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    heartbeat = args.heartbeat_sec
    write_json(
        out / "queue.json",
        {
            "schema_version": "phase_135d_queue_v1",
            "milestone": MILESTONE,
            "status": "started",
            "started_at": utc_now(),
            "heartbeat_sec": heartbeat,
        },
    )
    append_progress(out, "startup", heartbeat_sec=heartbeat)
    running_decision = {"decision": "pending", "next": "pending", "phase_count": 0}
    refresh_status(out, "running", ["GLOBAL_RAW_EVIDENCE_REBUILD_PLAN_RUNNING"], running_decision)

    upstream_root = resolve_path(args.upstream_135b_root)
    upstream = verify_upstream_135b(out, upstream_root)
    append_progress(out, "upstream_verification", **upstream["manifest"])
    refresh_status(out, "running", ["UPSTREAM_135B_VERIFIED"], {"decision": "pending", "next": "pending", "phase_count": EXPECTED_PHASE_COUNT})

    matrix = build_phase_matrix(upstream["reclass"], upstream["source_scan"], upstream["path_report"])
    matrix_payload = {
        "schema_version": "phase_135d_phase_rebuild_matrix_v1",
        "source_of_truth": "135B phase_evidence_reclassification.json",
        "phase_count": len(matrix),
        "classification_counts": dict(Counter(row["current_classification"] for row in matrix)),
        "allowed_action_required": ALLOWED_ACTIONS,
        "phases": matrix,
    }
    write_json(out / "phase_rebuild_matrix.json", matrix_payload)
    append_progress(out, "phase_rebuild_matrix", phase_count=len(matrix))

    claim_map = build_claim_quarantine_map()
    write_json(out / "claim_quarantine_map.json", claim_map)
    append_progress(out, "claim_quarantine_map")

    helper = build_raw_helper_requirements()
    canary = build_canary_spec()
    sequence = build_rebuild_sequence()
    manual = build_manual_review_plan(matrix)
    future = build_future_checker_requirements()
    risk = build_risk_register()
    write_json(out / "raw_generation_helper_requirements.json", helper)
    append_progress(out, "raw_generation_helper_requirements")
    write_json(out / "expected_output_canary_spec.json", canary)
    append_progress(out, "expected_output_canary_spec")
    write_json(out / "rebuild_sequence.json", sequence)
    append_progress(out, "rebuild_sequence")
    write_json(out / "manual_review_plan.json", manual)
    append_progress(out, "manual_review_plan")
    write_json(out / "future_checker_requirements.json", future)
    append_progress(out, "future_checker_requirements")
    write_json(out / "evidence_recovery_risk_register.json", risk)
    append_progress(out, "evidence_recovery_risk_register")

    validate_plan_artifacts(out, matrix, claim_map, manual)
    append_progress(out, "plan_validation")

    decision = {
        "schema_version": "phase_135d_decision_v1",
        "decision": SUCCESS_DECISION,
        "next": SUCCESS_NEXT,
        "upstream_135b_verified": True,
        "phase_count": len(matrix),
        "classification_counts": dict(Counter(row["current_classification"] for row in matrix)),
        "phase_rebuild_matrix_written": True,
        "claim_quarantine_map_written": True,
        "raw_generation_helper_requirements_written": True,
        "expected_output_canary_spec_written": True,
        "rebuild_sequence_written": True,
        "manual_review_plan_written": True,
        "future_checker_requirements_written": True,
        "decision_written": True,
        "train_step_count": 0,
        "optimizer_step_count": 0,
        "inference_run_count": 0,
        "service_started": False,
        "deployment_smoke_run": False,
        "checkpoint_mutated": False,
        "bounded_release_artifact_unchanged": True,
        "runtime_surface_mutated": False,
        "root_license_changed": False,
        "repo_cleanup_performed": False,
        "raw_assistant_capability_restored": False,
        "structured_tool_capability_restored": False,
        "gpt_like_readiness_claimed": False,
        "open_domain_assistant_readiness_claimed": False,
        "production_chat_claimed": False,
        "public_api_claimed": False,
        "deployment_readiness_claimed": False,
        "safety_alignment_claimed": False,
    }
    write_json(out / "decision.json", decision)
    append_progress(out, "decision_writing", decision=SUCCESS_DECISION)
    verdicts = [
        "GLOBAL_RAW_EVIDENCE_REBUILD_PLAN_COMPLETE",
        "UPSTREAM_135B_RECLASSIFICATION_VERIFIED",
        "PHASE_REBUILD_MATRIX_WRITTEN",
        "CLAIM_QUARANTINE_MAP_WRITTEN",
        "RAW_GENERATION_HELPER_REQUIREMENTS_WRITTEN",
        "EXPECTED_OUTPUT_CANARY_SPEC_WRITTEN",
        "MANUAL_REVIEW_PLAN_WRITTEN",
        "FUTURE_CHECKER_REQUIREMENTS_WRITTEN",
        "RAW_ASSISTANT_CAPABILITY_REMAINS_QUARANTINED",
        "STRUCTURED_TOOL_CAPABILITY_REMAINS_INVALIDATED_AS_MODEL_EVIDENCE",
        "NO_TRAINING_PERFORMED",
        "NO_MODEL_INFERENCE_PERFORMED",
        "NO_CHECKPOINT_MUTATION",
        "NO_RUNTIME_OR_RELEASE_MUTATION",
    ]
    refresh_status(out, "positive", verdicts, decision)
    append_progress(out, "final_verdict", verdicts=verdicts)
    write_json(
        out / "queue.json",
        {
            "schema_version": "phase_135d_queue_v1",
            "milestone": MILESTONE,
            "status": "completed",
            "completed_at": utc_now(),
            "heartbeat_sec": heartbeat,
        },
    )


def write_failure(args: argparse.Namespace, error: GateError) -> None:
    try:
        out = resolve_target_out(args.out)
    except GateError:
        out = (REPO_ROOT / DEFAULT_OUT).resolve()
    out.mkdir(parents=True, exist_ok=True)
    decision = {
        "schema_version": "phase_135d_failure_decision_v1",
        "decision": "upstream_135b_artifact_missing" if error.verdict == "UPSTREAM_135B_ARTIFACT_MISSING" else "rebuild_plan_incomplete",
        "next": "135D_UPSTREAM_AUDIT_ARTIFACT_MISSING" if error.verdict == "UPSTREAM_135B_ARTIFACT_MISSING" else "135D_REBUILD_PLAN_INCOMPLETE_ANALYSIS",
        "failure_verdict": error.verdict,
        "failure_message": error.message,
        "raw_assistant_capability_restored": False,
        "structured_tool_capability_restored": False,
    }
    write_json(out / "decision.json", decision)
    append_progress(out, "failure", status="failed", verdict=error.verdict, message=error.message)
    write_summary(out, "failure", ["GLOBAL_RAW_EVIDENCE_REBUILD_PLAN_FAILS", error.verdict], decision, error.message)
    write_report(out, ["GLOBAL_RAW_EVIDENCE_REBUILD_PLAN_FAILS", error.verdict], decision)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--upstream-135b-root", default=str(DEFAULT_UPSTREAM_135B_ROOT))
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        run(args)
    except GateError as error:
        write_failure(args, error)
        print(f"{error.verdict}: {error.message}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
