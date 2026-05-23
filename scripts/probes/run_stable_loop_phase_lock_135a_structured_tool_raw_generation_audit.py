#!/usr/bin/env python3
"""135A structured/tool raw-generation audit.

This audit-only milestone inspects the 134/135 structured-output harnesses and
artifacts for oracle/expected-output shortcuts. It performs no training, no
repair, no checkpoint mutation, no inference, no service startup, no
deployment, and no runtime/product/release/API mutation.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "STABLE_LOOP_PHASE_LOCK_135A_STRUCTURED_TOOL_RAW_GENERATION_AUDIT"
DEFAULT_OUT = Path("target/pilot_wave/stable_loop_phase_lock_135a_structured_tool_raw_generation_audit/smoke")
DEFAULT_PHASE_134_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_134_structured_output_tool_api_repair/smoke")
DEFAULT_PHASE_135_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_135_structured_output_tool_api_repair_scale_confirm/smoke")
PHASE_134_RUNNER = Path("scripts/probes/run_stable_loop_phase_lock_134_structured_output_tool_api_repair.py")
PHASE_135_RUNNER = Path("scripts/probes/run_stable_loop_phase_lock_135_structured_output_tool_api_repair_scale_confirm.py")
BOUNDARY_TEXT = (
    "135A is audit-only. It does not train, repair, run model inference, mutate "
    "checkpoints, start services, deploy, add public APIs, or modify runtime, "
    "service, product, release, SDK, or root LICENSE surfaces. It audits whether "
    "134/135 structured-output/tool-API-like results are real raw generation "
    "evidence or expected-output/oracle shortcut evidence. It is not GPT-like "
    "assistant readiness, not open-domain assistant readiness, not production "
    "chat, not public API, not deployment readiness, not safety alignment, and "
    "not Hungarian assistant readiness."
)


class AuditError(Exception):
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


def read_jsonl_sample(path: Path, limit: int = 64) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    return rows


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def resolve_target_out(text: str) -> Path:
    path = Path(text)
    if path.is_absolute() or any(part == ".." for part in path.parts):
        raise AuditError("STRUCTURED_TOOL_RAW_GENERATION_AUDIT_FAILS", "--out must be repo-relative")
    parts = [part.lower() for part in path.parts]
    if len(parts) < 2 or parts[0] != "target" or parts[1] != "pilot_wave":
        raise AuditError("STRUCTURED_TOOL_RAW_GENERATION_AUDIT_FAILS", "--out must stay under target/pilot_wave")
    return (REPO_ROOT / path).resolve()


def resolve_path(text: str) -> Path:
    path = Path(text)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def append_progress(out: Path, event: str, status: str = "completed", **details: Any) -> None:
    append_jsonl(out / "progress.jsonl", {"ts": utc_now(), "event": event, "status": status, "details": details})


def source_segment(lines: list[str], lineno: int, radius: int = 3) -> list[str]:
    start = max(1, lineno - radius)
    end = min(len(lines), lineno + radius)
    return [f"{idx}: {lines[idx - 1]}" for idx in range(start, end + 1)]


def find_expected_output_shortcuts(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))
    lines = text.splitlines()
    findings: list[dict[str, Any]] = []

    def is_name(node: ast.AST, name: str) -> bool:
        return isinstance(node, ast.Name) and node.id == name

    def is_arm_main_compare(node: ast.AST) -> bool:
        if not isinstance(node, ast.Compare) or len(node.ops) != 1 or not isinstance(node.ops[0], ast.Eq):
            return False
        operands = [node.left, *node.comparators]
        return any(is_name(item, "arm") for item in operands) and any(is_name(item, "MAIN_ARM") for item in operands)

    def contains_arm_main_compare(node: ast.AST) -> bool:
        return any(is_arm_main_compare(child) for child in ast.walk(node))

    def is_row_expected_output(node: ast.AST | None) -> bool:
        if not isinstance(node, ast.Subscript) or not is_name(node.value, "row"):
            return False
        subscript = node.slice
        return isinstance(subscript, ast.Constant) and subscript.value == "expected_output"

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.function_stack: list[str] = []
            self.main_arm_context: list[int] = []

        def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
            self.function_stack.append(node.name)
            self.generic_visit(node)
            self.function_stack.pop()

        def visit_If(self, node: ast.If) -> Any:
            in_main_arm = contains_arm_main_compare(node.test)
            if in_main_arm:
                self.main_arm_context.append(node.lineno)
            self.generic_visit(node)
            if in_main_arm:
                self.main_arm_context.pop()

        def visit_Return(self, node: ast.Return) -> Any:
            expr = ast.unparse(node.value) if node.value is not None else ""
            if self.main_arm_context and is_row_expected_output(node.value):
                findings.append(
                    {
                        "file": rel(path),
                        "function": self.function_stack[-1] if self.function_stack else None,
                        "line": node.lineno,
                        "pattern": "main_arm_returns_expected_output",
                        "expression": expr,
                        "context": source_segment(lines, node.lineno),
                        "severity": "critical",
                    }
                )
            self.generic_visit(node)

    Visitor().visit(tree)
    return findings


def source_code_audit() -> dict[str, Any]:
    files = [REPO_ROOT / PHASE_134_RUNNER, REPO_ROOT / PHASE_135_RUNNER]
    audits: list[dict[str, Any]] = []
    shortcut_findings: list[dict[str, Any]] = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        findings = find_expected_output_shortcuts(path)
        shortcut_findings.extend(findings)
        audits.append(
            {
                "file": rel(path),
                "sha256": file_hash(path),
                "line_count": len(text.splitlines()),
                "direct_expected_output_return_count": len(re.findall(r"return\s+row\[\"expected_output\"\]", text)),
                "expected_payload_reference_count": len(re.findall(r"expected_payload", text)),
                "main_arm_expected_output_findings": findings,
            }
        )
    return {
        "schema_version": "phase_135a_source_code_audit_v1",
        "audited_files": audits,
        "direct_expected_output_shortcut_detected": bool(shortcut_findings),
        "shortcut_findings": shortcut_findings,
    }


def artifact_trace(root: Path, phase: str) -> dict[str, Any]:
    summary_path = root / "summary.json"
    decision_path = root / "decision.json"
    raw_path = root / "raw_generation_results.jsonl"
    dataset_candidates = [
        root / "structured_tool_repair_dataset.jsonl",
        root / "structured_tool_scale_dataset.jsonl",
    ]
    dataset_path = next((path for path in dataset_candidates if path.exists()), dataset_candidates[-1])
    summary = read_json(summary_path) if summary_path.exists() else {}
    decision = read_json(decision_path) if decision_path.exists() else {}
    raw_sample = read_jsonl_sample(raw_path)
    dataset_sample = read_jsonl_sample(dataset_path)
    sample_equal_count = 0
    for raw, data in zip(raw_sample, dataset_sample):
        generated = raw.get("generated_text", raw.get("output"))
        expected = raw.get("expected_output", data.get("expected_output"))
        if generated == expected:
            sample_equal_count += 1
    flags = {
        "integrated_policy_used_during_final_eval": summary.get("metrics", {}).get("integrated_policy_used_during_final_eval", summary.get("integrated_policy_used_during_final_eval")),
        "expected_answer_used_during_eval": summary.get("metrics", {}).get("expected_answer_used_during_eval", summary.get("expected_answer_used_during_eval")),
        "oracle_rerank_used": summary.get("metrics", {}).get("oracle_rerank_used", summary.get("oracle_rerank_used")),
        "llm_judge_used": summary.get("metrics", {}).get("llm_judge_used", summary.get("llm_judge_used")),
    }
    return {
        "phase": phase,
        "root": rel(root),
        "summary_exists": summary_path.exists(),
        "decision_exists": decision_path.exists(),
        "raw_generation_results_exists": raw_path.exists(),
        "dataset_exists": dataset_path.exists(),
        "summary_status": summary.get("status"),
        "decision": decision.get("decision"),
        "next": decision.get("next"),
        "declared_raw_eval_flags": flags,
        "raw_sample_count": len(raw_sample),
        "dataset_sample_count": len(dataset_sample),
        "positive_arm_sample_generated_equals_expected_count": sample_equal_count,
        "positive_arm_sample_generated_equals_expected_rate": sample_equal_count / len(raw_sample) if raw_sample else None,
        "raw_sample_sha256": hashlib.sha256(json.dumps(raw_sample, sort_keys=True).encode("utf-8")).hexdigest() if raw_sample else None,
        "dataset_sample_sha256": hashlib.sha256(json.dumps(dataset_sample, sort_keys=True).encode("utf-8")).hexdigest() if dataset_sample else None,
    }


def write_summary(out: Path, status: str, verdicts: list[str], decision: dict[str, Any]) -> None:
    write_json(
        out / "summary.json",
        {
            "schema_version": "phase_135a_summary_v1",
            "milestone": MILESTONE,
            "status": status,
            "verdicts": verdicts,
            "decision": decision.get("decision"),
            "next": decision.get("next"),
            "audit_only": True,
            "training_performed": False,
            "repair_performed": False,
            "inference_run_count": 0,
            "checkpoint_mutated": False,
            "runtime_surface_mutated": False,
            "service_started": False,
            "deployment_smoke_run": False,
            "public_api_changed": False,
            "gpt_like_assistant_readiness_claimed": False,
            "open_domain_assistant_readiness_claimed": False,
            "production_chat_claimed": False,
            "public_api_claimed": False,
            "deployment_readiness_claimed": False,
            "safety_alignment_claimed": False,
            "hungarian_assistant_readiness_claimed": False,
            "boundary": BOUNDARY_TEXT,
            "metrics": decision,
        },
    )


def write_report(out: Path, verdicts: list[str], decision: dict[str, Any], source: dict[str, Any]) -> None:
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
            f"- `structured_output_scale_confirm_invalid_as_model_evidence`: `{decision.get('structured_output_scale_confirm_invalid_as_model_evidence')}`",
            f"- `structured_tool_oracle_shortcut_detected`: `{decision.get('structured_tool_oracle_shortcut_detected')}`",
            "",
            "## Source Findings",
            "",
        ]
    )
    for finding in source.get("shortcut_findings", []):
        lines.append(f"- `{finding['file']}:{finding['line']}` `{finding['pattern']}`")
    write_text(out / "report.md", "\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> None:
    out = resolve_target_out(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "queue.json", {"schema_version": "phase_135a_queue_v1", "milestone": MILESTONE, "status": "started", "started_at": utc_now(), "heartbeat_sec": args.heartbeat_sec})
    append_progress(out, "startup")
    write_json(
        out / "audit_config.json",
        {
            "schema_version": "phase_135a_audit_config_v1",
            "milestone": MILESTONE,
            "phase_134_runner": rel(REPO_ROOT / PHASE_134_RUNNER),
            "phase_135_runner": rel(REPO_ROOT / PHASE_135_RUNNER),
            "phase_134_root": args.phase_134_root,
            "phase_135_root": args.phase_135_root,
            "audit_only": True,
            "training_performed": False,
            "repair_performed": False,
            "inference_run_count": 0,
            "checkpoint_mutated": False,
            "runtime_surface_mutated": False,
        },
    )

    append_progress(out, "source_code_audit_start")
    source = source_code_audit()
    write_json(out / "source_code_audit.json", source)
    append_progress(out, "source_code_audit_complete", direct_expected_output_shortcut_detected=source["direct_expected_output_shortcut_detected"])

    phase_134_root = resolve_path(args.phase_134_root)
    phase_135_root = resolve_path(args.phase_135_root)
    append_progress(out, "artifact_trace_start")
    traces = [artifact_trace(phase_134_root, "134"), artifact_trace(phase_135_root, "135")]
    trace_report = {"schema_version": "phase_135a_artifact_trace_report_v1", "traces": traces}
    write_json(out / "artifact_trace_report.json", trace_report)
    append_progress(out, "artifact_trace_complete")

    source_shortcut = source["direct_expected_output_shortcut_detected"]
    artifact_expected_match = any((trace.get("positive_arm_sample_generated_equals_expected_rate") or 0.0) >= 0.95 for trace in traces)
    shortcut_detected = source_shortcut
    generation_path = {
        "schema_version": "phase_135a_positive_arm_generation_path_report_v1",
        "positive_arm_returns_expected_output_directly": source_shortcut,
        "expected_payload_used_in_generation_path": any(audit["expected_payload_reference_count"] > 0 for audit in source["audited_files"]),
        "deterministic_answer_construction_instead_of_model_output": source_shortcut,
        "generated_text_exists_independently": False,
        "generated_text_produced_by_model_raw_generation_function": False,
        "expected_output_used_only_for_scoring": not source_shortcut,
        "artifact_generated_equals_expected_high_rate": artifact_expected_match,
        "finding": "MAIN_ARM direct expected_output return detected" if shortcut_detected else "no direct shortcut detected",
    }
    write_json(out / "positive_arm_generation_path_report.json", generation_path)

    oracle_report = {
        "schema_version": "phase_135a_oracle_shortcut_report_v1",
        "structured_tool_oracle_shortcut_detected": shortcut_detected,
        "direct_expected_output_return_in_positive_arm": source_shortcut,
        "expected_payload_used_in_generation_path": generation_path["expected_payload_used_in_generation_path"],
        "oracle_metadata_used_during_final_eval": shortcut_detected,
        "declared_oracle_flags_were_sufficient": False,
        "declared_raw_only_flags_conflict_with_source_path": shortcut_detected,
        "verdict": "STRUCTURED_TOOL_ORACLE_SHORTCUT_DETECTED" if shortcut_detected else "STRUCTURED_TOOL_RAW_GENERATION_AUDIT_POSITIVE",
    }
    write_json(out / "oracle_shortcut_report.json", oracle_report)

    if shortcut_detected:
        decision = {
            "schema_version": "phase_135a_decision_v1",
            "decision": "structured_tool_oracle_shortcut_detected",
            "next": "135B_STRUCTURED_TOOL_REAL_RAW_EVAL_REBUILD",
            "structured_tool_oracle_shortcut_detected": True,
            "structured_output_scale_confirm_invalid_as_model_evidence": True,
            "phase_134_invalid_as_model_evidence": True,
            "phase_135_invalid_as_model_evidence": True,
            "raw_generation_evidence_status": "invalidated_by_expected_output_shortcut",
            "recommended_action": "rebuild structured/tool eval with genuine raw model generation path before 136",
        }
        verdicts = [
            "STRUCTURED_TOOL_ORACLE_SHORTCUT_DETECTED",
            "STRUCTURED_OUTPUT_SCALE_CONFIRM_INVALID_AS_MODEL_EVIDENCE",
            "STRUCTURED_TOOL_RAW_GENERATION_AUDIT_COMPLETED",
            "NO_TRAINING_PERFORMED",
            "NO_REPAIR_PERFORMED",
            "NO_CHECKPOINT_MUTATION",
            "NO_RUNTIME_OR_API_MUTATION",
        ]
        status = "audit_complete_shortcut_detected"
    else:
        decision = {
            "schema_version": "phase_135a_decision_v1",
            "decision": "structured_tool_raw_generation_audit_positive",
            "next": "136_POST_STRUCTURED_TOOL_REPAIR_CEILING_AND_GAP_REMAP",
            "structured_tool_oracle_shortcut_detected": False,
            "structured_output_scale_confirm_invalid_as_model_evidence": False,
            "raw_generation_evidence_status": "source_path_clear",
        }
        verdicts = [
            "STRUCTURED_TOOL_RAW_GENERATION_AUDIT_POSITIVE",
            "NO_TRAINING_PERFORMED",
            "NO_REPAIR_PERFORMED",
            "NO_CHECKPOINT_MUTATION",
            "NO_RUNTIME_OR_API_MUTATION",
        ]
        status = "positive"

    write_json(out / "evidence_reclassification.json", {"schema_version": "phase_135a_evidence_reclassification_v1", **decision})
    write_json(out / "decision.json", decision)
    append_progress(out, "decision_writing", decision=decision["decision"])
    write_summary(out, status, verdicts, decision)
    write_report(out, verdicts, decision, source)
    append_progress(out, "final_verdict", verdicts=verdicts)
    write_json(out / "queue.json", {"schema_version": "phase_135a_queue_v1", "milestone": MILESTONE, "status": "completed", "completed_at": utc_now(), "heartbeat_sec": args.heartbeat_sec})


def write_failure(args: argparse.Namespace, error: AuditError) -> None:
    try:
        out = resolve_target_out(args.out)
    except AuditError:
        out = (REPO_ROOT / DEFAULT_OUT).resolve()
    out.mkdir(parents=True, exist_ok=True)
    decision = {"decision": "structured_tool_raw_generation_audit_failed", "next": "135A_STRUCTURED_TOOL_RAW_GENERATION_AUDIT_RETRY", "failure_verdict": error.verdict, "failure_message": error.message}
    write_json(out / "decision.json", {"schema_version": "phase_135a_failure_decision_v1", **decision})
    append_progress(out, "failure", status="failed", verdict=error.verdict, message=error.message)
    write_summary(out, "failure", ["STRUCTURED_TOOL_RAW_GENERATION_AUDIT_FAILS", error.verdict], decision, error.verdict)
    write_report(out, ["STRUCTURED_TOOL_RAW_GENERATION_AUDIT_FAILS", error.verdict], decision, {"shortcut_findings": []})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--phase-134-root", default=str(DEFAULT_PHASE_134_ROOT))
    parser.add_argument("--phase-135-root", default=str(DEFAULT_PHASE_135_ROOT))
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        run(args)
    except AuditError as error:
        write_failure(args, error)
        print(f"{error.verdict}: {error.message}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
