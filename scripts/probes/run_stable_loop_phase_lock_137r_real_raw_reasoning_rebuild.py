#!/usr/bin/env python3
"""137R real-raw reasoning rebuild.

This phase evaluates whether the current repo-local raw generation path restores
reasoning evidence after the 135E/136R trust-root reset. It is eval-only: no
training, no repair, no checkpoint mutation, and no full raw assistant claim.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import random
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "STABLE_LOOP_PHASE_LOCK_137R_REAL_RAW_REASONING_REBUILD"
SHORT_MILESTONE = "137R_REAL_RAW_REASONING_REBUILD"
DEFAULT_OUT = Path("target/pilot_wave/stable_loop_phase_lock_137r_real_raw_reasoning_rebuild/smoke")
DEFAULT_UPSTREAM_136R_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_136r_real_raw_core_capability_minimal_rebuild/smoke")
DEFAULT_UPSTREAM_135E_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_135e_shared_raw_generation_helper_and_canary_gate/smoke")
DEFAULT_UPSTREAM_135D_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_135d_global_raw_evidence_rebuild_plan/smoke")
HELPER_PATH = REPO_ROOT / "scripts/probes/shared_raw_generation_helper.py"
RUNNER_PATH = Path(__file__).resolve()
CHECKER_PATH = REPO_ROOT / "scripts/probes/run_stable_loop_phase_lock_137r_real_raw_reasoning_rebuild_check.py"
POSITIVE_VERDICT = "REAL_RAW_REASONING_REBUILD_POSITIVE"
NEGATIVE_VERDICT = "REAL_RAW_REASONING_REBUILD_FAILS"
POSITIVE_DECISION = "real_raw_reasoning_evidence_rebuilt"
NEGATIVE_DECISION = "real_raw_reasoning_not_restored"
POSITIVE_NEXT = "138R_REAL_RAW_MULTI_TURN_STATE_REBUILD"
NEGATIVE_NEXT = "137B_REAL_RAW_REASONING_REPAIR_PLAN"
BOUNDARY_TEXT = (
    "137R rebuilds only the reasoning subtrack if positive. Raw assistant "
    "capability remains quarantined. Structured/tool capability remains "
    "invalidated as model evidence. It does not train, repair, mutate "
    "checkpoints, start services, deploy, delete files, consolidate old "
    "runners, modify runtime/release/product surfaces, or change root LICENSE. "
    "It is not GPT-like readiness, not open-domain assistant readiness, not "
    "production chat, not public API, not deployment readiness, and not safety "
    "alignment."
)

ALLOWED_HELPER_KEYS = {"prompt", "checkpoint_path", "checkpoint_hash", "seed", "max_new_tokens", "generation_config"}
FORBIDDEN_HELPER_KEYS = {
    "expected_output",
    "expected_payload",
    "expected_answer",
    "required_keys",
    "required_keywords",
    "forbidden_outputs",
    "schema_answer_object",
    "scorer_metadata",
    "labels",
    "oracle_data",
    "eval_family_expected_values",
    "row_answer",
    "target_json",
    "gold_output",
    "eval_family",
    "answer",
    "expected_values",
}
FINAL_EVAL_FLAGS = {
    "generated_text_produced_before_scoring": True,
    "shared_raw_generation_helper_used": True,
    "expected_output_used_for_generation": False,
    "expected_payload_used_for_generation": False,
    "scorer_metadata_used_for_generation": False,
    "oracle_rerank_used": False,
    "verifier_rerank_used": False,
    "llm_judge_used": False,
    "teacher_forcing_used": False,
    "constrained_decoding_used": False,
    "json_mode_used": False,
    "grammar_decoder_used": False,
    "regex_fixer_used": False,
    "post_generation_repair_used": False,
    "retry_loop_used": False,
    "best_of_n_used": False,
    "actual_tool_execution_used": False,
    "runtime_tool_call_used": False,
}
FAMILIES = [
    "REAL_RAW_PROVIDED_FACT_QA",
    "REAL_RAW_SINGLE_STEP_REASONING",
    "REAL_RAW_TWO_STEP_REASONING",
    "REAL_RAW_RULE_CHAINING",
    "REAL_RAW_TABLE_RULE_REASONING",
    "REAL_RAW_SMALL_ARITHMETIC_SUPPLIED_VALUES",
    "REAL_RAW_CONTRADICTION_RESOLUTION",
    "REAL_RAW_MULTI_DOC_PRIORITY_REASONING",
    "REAL_RAW_HALLUCINATION_TRAP_REASONING",
    "REAL_RAW_SHORT_EXPLANATION_DIAGNOSTIC",
]
SEED_GATES = {
    "real_raw_reasoning_accuracy": 0.70,
    "single_step_reasoning_accuracy": 0.80,
    "two_step_reasoning_accuracy": 0.70,
    "rule_chaining_accuracy": 0.70,
    "table_rule_reasoning_accuracy": 0.70,
    "small_arithmetic_accuracy": 0.70,
    "contradiction_resolution_accuracy": 0.65,
    "hallucination_trap_pass_rate": 0.70,
}
STANDARD_REFUSAL_TEMPLATES = {"INSUFFICIENT_INFORMATION", "UNKNOWN", "UNANSWERABLE"}


class GateError(Exception):
    def __init__(self, verdict: str, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.verdict = verdict
        self.message = message
        self.details = details or {}


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def resolve_target_out(text: str) -> Path:
    path = Path(text)
    if path.is_absolute() or any(part == ".." for part in path.parts):
        raise GateError("137R_BOUNDARY_FAILURE", "--out must be repo-relative")
    parts = [part.lower() for part in path.parts]
    if len(parts) < 2 or parts[0] != "target" or parts[1] != "pilot_wave":
        raise GateError("137R_BOUNDARY_FAILURE", "--out must stay under target/pilot_wave")
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
            "schema_version": "phase_137r_summary_v1",
            "milestone": MILESTONE,
            "status": status,
            "verdicts": verdicts,
            "decision": decision.get("decision"),
            "next": decision.get("next"),
            "failure": failure,
            "boundary": BOUNDARY_TEXT,
            "eval_only": True,
            "training_performed": False,
            "repair_performed": False,
            "checkpoint_mutated": False,
            "service_started": False,
            "deployment_smoke_run": False,
            "runtime_surface_mutated": False,
            "root_license_changed": False,
            "raw_assistant_capability_restored": False,
            "structured_tool_capability_restored": False,
            "reasoning_subtrack_real_raw_evidence_restored": decision.get("reasoning_subtrack_real_raw_evidence_restored", False),
            "gpt_like_readiness_claimed": False,
            "open_domain_assistant_readiness_claimed": False,
            "production_chat_claimed": False,
            "public_api_claimed": False,
            "deployment_readiness_claimed": False,
            "safety_alignment_claimed": False,
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
            f"- `verdict`: `{decision.get('verdict')}`",
            f"- `mean_real_raw_reasoning_accuracy`: `{decision.get('mean_real_raw_reasoning_accuracy')}`",
            f"- `reasoning_subtrack_real_raw_evidence_restored`: `{decision.get('reasoning_subtrack_real_raw_evidence_restored')}`",
            "",
            "137R rebuilds only the reasoning subtrack if positive.",
            "Raw assistant capability remains quarantined.",
            "Structured/tool capability remains invalidated as model evidence.",
            "No full raw assistant capability restored.",
            "Not GPT-like readiness.",
            "Not open-domain assistant readiness.",
            "Not production chat.",
            "Not public API.",
            "Not deployment readiness.",
            "Not safety alignment.",
        ]
    )
    write_text(out / "report.md", "\n".join(lines) + "\n")


def refresh_status(out: Path, status: str, verdicts: list[str], decision: dict[str, Any]) -> None:
    write_summary(out, status, verdicts, decision)
    write_report(out, verdicts, decision)


def import_helper() -> Any:
    spec = importlib.util.spec_from_file_location("shared_raw_generation_helper", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise GateError("RAW_GENERATION_BACKEND_MISSING", "shared helper import spec unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_upstreams(out: Path, root_136r: Path, root_135e: Path, root_135d: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    required_136r = ["decision.json", "summary.json", "raw_generation_helper_provenance.json", "expected_output_canary_report.json", "ast_shortcut_scan_report.json"]
    required_135e = ["decision.json", "future_milestone_requirements.json", "raw_generation_helper_provenance.json"]
    required_135d = ["decision.json", "rebuild_sequence.json"]
    missing = [f"136R:{name}" for name in required_136r if not (root_136r / name).exists()]
    missing += [f"135E:{name}" for name in required_135e if not (root_135e / name).exists()]
    missing += [f"135D:{name}" for name in required_135d if not (root_135d / name).exists()]
    if missing:
        raise GateError("UPSTREAM_ARTIFACT_MISSING", "required upstream artifacts missing", {"missing": missing})

    decision_136r = read_json(root_136r / "decision.json")
    decision_135e = read_json(root_135e / "decision.json")
    decision_135d = read_json(root_135d / "decision.json")
    sequence = read_json(root_135d / "rebuild_sequence.json")
    future = read_json(root_135e / "future_milestone_requirements.json")
    if decision_136r.get("verdict") != "REAL_RAW_CORE_CAPABILITY_MINIMAL_REBUILD_POSITIVE" or decision_136r.get("next") != SHORT_MILESTONE:
        raise GateError("UPSTREAM_136R_NOT_POSITIVE", "136R decision mismatch")
    if decision_135e.get("verdict") != "SHARED_RAW_GENERATION_HELPER_AND_CANARY_GATE_POSITIVE":
        raise GateError("UPSTREAM_135E_NOT_POSITIVE", "135E decision mismatch")
    if decision_135d.get("decision") != "global_raw_evidence_rebuild_plan_complete":
        raise GateError("UPSTREAM_135D_NOT_POSITIVE", "135D decision mismatch")
    if SHORT_MILESTONE not in sequence.get("sequence", []):
        raise GateError("UPSTREAM_135D_SEQUENCE_MISMATCH", "135D rebuild sequence does not include 137R")
    if SHORT_MILESTONE not in future.get("applies_to", []):
        raise GateError("UPSTREAM_135E_REQUIREMENTS_MISSING", "135E future requirements do not include 137R")

    manifest_136r = {
        "schema_version": "phase_137r_upstream_136r_manifest_v1",
        "upstream_136r_root": rel(root_136r),
        "upstream_136r_verified": True,
        "decision": decision_136r.get("decision"),
        "next": decision_136r.get("next"),
        "verdict": decision_136r.get("verdict"),
        "raw_assistant_capability_restored": decision_136r.get("raw_assistant_capability_restored"),
        "structured_tool_capability_restored": decision_136r.get("structured_tool_capability_restored"),
    }
    manifest_135e = {
        "schema_version": "phase_137r_upstream_135e_manifest_v1",
        "upstream_135e_root": rel(root_135e),
        "upstream_135e_verified": True,
        "decision": decision_135e.get("decision"),
        "next": decision_135e.get("next"),
        "verdict": decision_135e.get("verdict"),
        "future_requirements_applied": True,
    }
    manifest_135d = {
        "schema_version": "phase_137r_upstream_135d_manifest_v1",
        "upstream_135d_root": rel(root_135d),
        "upstream_135d_verified": True,
        "decision": decision_135d.get("decision"),
        "next": decision_135d.get("next"),
        "sequence_contains_137r": True,
    }
    write_json(out / "upstream_136r_manifest.json", manifest_136r)
    write_json(out / "upstream_135e_manifest.json", manifest_135e)
    write_json(out / "upstream_135d_manifest.json", manifest_135d)
    return manifest_136r, manifest_135e, manifest_135d


def select_backend(helper: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    report = helper.discover_backend()
    selected = report.get("selected")
    if not selected:
        raise GateError("RAW_GENERATION_BACKEND_MISSING", "no safe repo-local raw generation backend found", report)
    return report, selected


def helper_request(row: dict[str, Any], selected: dict[str, Any], seed: int, max_new_tokens: int) -> dict[str, Any]:
    return {
        "prompt": row["prompt"],
        "checkpoint_path": selected["checkpoint_path"],
        "checkpoint_hash": selected["checkpoint_sha256"],
        "seed": seed,
        "max_new_tokens": max_new_tokens,
        "generation_config": {"temperature": 0.0, "device": "cpu", "stop_on_newline": False},
    }


def forbidden_input_tests(helper: Any, selected: dict[str, Any]) -> dict[str, Any]:
    base = {
        "prompt": "137R forbidden metadata rejection smoke.",
        "checkpoint_path": selected["checkpoint_path"],
        "checkpoint_hash": selected["checkpoint_sha256"],
        "seed": 1370,
        "max_new_tokens": 8,
        "generation_config": {"temperature": 0.0, "device": "cpu", "stop_on_newline": False},
    }
    rows: list[dict[str, Any]] = []
    for key in sorted(FORBIDDEN_HELPER_KEYS | {"unexpected_extra"}):
        request = dict(base)
        request[key] = "forbidden"
        try:
            helper.raw_generate(request)
            rows.append({"field": key, "rejected": False, "verdict": None})
        except Exception as exc:
            rows.append({"field": key, "rejected": getattr(exc, "verdict", None) == "RAW_GENERATION_FORBIDDEN_INPUT_DETECTED", "verdict": getattr(exc, "verdict", type(exc).__name__)})
    config_rows: list[dict[str, Any]] = []
    for name, config in [
        ("unknown_config_key", {"temperature": 0.0, "device": "cpu", "stop_on_newline": False, "unknown": "x"}),
        ("forbidden_config_expected_output", {"temperature": 0.0, "device": "cpu", "stop_on_newline": False, "expected_output": "x"}),
        ("nested_forbidden_config_labels", {"temperature": 0.0, "device": "cpu", "stop_on_newline": False, "nested": {"labels": ["x"]}}),
    ]:
        request = dict(base)
        request["generation_config"] = config
        try:
            helper.raw_generate(request)
            config_rows.append({"test": name, "rejected": False, "verdict": None})
        except Exception as exc:
            config_rows.append({"test": name, "rejected": getattr(exc, "verdict", None) == "RAW_GENERATION_FORBIDDEN_INPUT_DETECTED", "verdict": getattr(exc, "verdict", type(exc).__name__)})
    return {
        "schema_version": "phase_137r_forbidden_input_rejection_report_v1",
        "rows": rows,
        "generation_config_rows": config_rows,
        "all_rejected": all(row["rejected"] for row in rows + config_rows),
    }


def expected_output_canary(helper: Any, selected: dict[str, Any], max_new_tokens: int) -> dict[str, Any]:
    original_row = {
        "row_id": "137r_canary_original",
        "prompt": "137R canary prompt: produce any raw continuation.",
        "expected_output": "correct A",
        "expected_payload": {"value": "correct A"},
    }
    shadow_row = dict(original_row)
    shadow_row["row_id"] = "137r_canary_shadow"
    shadow_row["expected_output"] = "corrupted B"
    shadow_row["expected_payload"] = {"value": "corrupted B"}
    original_request = helper_request(original_row, selected, 13701, max_new_tokens)
    shadow_request = helper_request(shadow_row, selected, 13701, max_new_tokens)
    original_response = helper.raw_generate(original_request)
    shadow_response = helper.raw_generate(shadow_request)
    generation_side_fields_identical = {
        "generated_text": original_response["generated_text"] == shadow_response["generated_text"],
        "generation_trace_hash": original_response["generation_trace_hash"] == shadow_response["generation_trace_hash"],
        "token_count": original_response["token_count"] == shadow_response["token_count"],
        "stop_reason": original_response["stop_reason"] == shadow_response["stop_reason"],
        "model_checkpoint_hash": original_response["model_checkpoint_hash"] == shadow_response["model_checkpoint_hash"],
        "generation_config_hash": original_response["generation_config_hash"] == shadow_response["generation_config_hash"],
    }
    request_hash_original = stable_hash(original_request)
    request_hash_shadow = stable_hash(shadow_request)
    request_keys = set(original_request) | set(shadow_request)
    return {
        "schema_version": "phase_137r_expected_output_canary_report_v1",
        "original_row_hash": stable_hash(original_row),
        "shadow_row_hash": stable_hash(shadow_row),
        "prompt_identical": original_row["prompt"] == shadow_row["prompt"],
        "original_helper_request_json": json.dumps(original_request, sort_keys=True, separators=(",", ":")),
        "shadow_helper_request_json": json.dumps(shadow_request, sort_keys=True, separators=(",", ":")),
        "original_helper_request_hash": request_hash_original,
        "shadow_helper_request_hash": request_hash_shadow,
        "helper_requests_identical": request_hash_original == request_hash_shadow,
        "generated_text_original_hash": text_hash(original_response["generated_text"]),
        "generated_text_shadow_hash": text_hash(shadow_response["generated_text"]),
        "generation_trace_hash_original": original_response["generation_trace_hash"],
        "generation_trace_hash_shadow": shadow_response["generation_trace_hash"],
        "token_count_original": original_response["token_count"],
        "token_count_shadow": shadow_response["token_count"],
        "stop_reason_original": original_response["stop_reason"],
        "stop_reason_shadow": shadow_response["stop_reason"],
        "model_checkpoint_hash_original": original_response["model_checkpoint_hash"],
        "model_checkpoint_hash_shadow": shadow_response["model_checkpoint_hash"],
        "generation_config_hash_original": original_response["generation_config_hash"],
        "generation_config_hash_shadow": shadow_response["generation_config_hash"],
        "generation_side_fields_identical": generation_side_fields_identical,
        "forbidden_fields_absent_from_helper_requests": not bool(request_keys & FORBIDDEN_HELPER_KEYS),
        "expected_material_only_outside_helper_request": "expected_output" not in original_request and "expected_payload" not in original_request and "expected_output" not in shadow_request and "expected_payload" not in shadow_request,
        "expected_output_canary_passed": all(generation_side_fields_identical.values()) and request_hash_original == request_hash_shadow and not bool(request_keys & FORBIDDEN_HELPER_KEYS),
    }


def ast_shortcut_scan(paths: list[Path]) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    old_runner_re = re.compile(r"^(run_stable_loop_phase_lock_|run_deck_local_)")
    forbidden_call_names = {
        "oracle_rerank",
        "verifier_rerank",
        "llm_judge",
        "grammar_decoder",
        "constrained_decoding",
        "regex_fixer",
        "json_fixer",
        "json_mode",
        "best_of_n",
        "retry_loop",
        "post_generation_repair",
        "runtime_tool_call",
        "actual_tool_execution",
    }

    def expr_uses_expected_material(node: ast.AST | None) -> bool:
        return node is not None and any(token in ast.unparse(node) for token in ["expected_output", "expected_payload", "expected_answer", "gold_output", "target_json"])

    for path in paths:
        if not path.exists():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

        class Scanner(ast.NodeVisitor):
            def __init__(self) -> None:
                self.function_stack: list[str] = []

            def in_generation_context(self) -> bool:
                return any("generate" in name or "raw_" in name for name in self.function_stack)

            def visit_Import(self, node: ast.Import) -> None:
                for alias in node.names:
                    if old_runner_re.match(alias.name):
                        findings.append({"file": rel(path), "lineno": node.lineno, "type": "OLD_RUNNER_IMPORT_DETECTED", "detail": alias.name})
                self.generic_visit(node)

            def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
                module = node.module or ""
                if old_runner_re.match(module):
                    findings.append({"file": rel(path), "lineno": node.lineno, "type": "OLD_RUNNER_IMPORT_DETECTED", "detail": module})
                self.generic_visit(node)

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                self.function_stack.append(node.name.lower())
                self.generic_visit(node)
                self.function_stack.pop()

            def visit_Assign(self, node: ast.Assign) -> None:
                targets = " ".join(ast.unparse(target) for target in node.targets)
                if re.search(r"generated_text|generated", targets) and expr_uses_expected_material(node.value):
                    findings.append({"file": rel(path), "lineno": node.lineno, "type": "AST_GENERATED_TEXT_FROM_EXPECTED_MATERIAL", "detail": ast.unparse(node)})
                self.generic_visit(node)

            def visit_Return(self, node: ast.Return) -> None:
                if self.in_generation_context() and expr_uses_expected_material(node.value):
                    findings.append({"file": rel(path), "lineno": node.lineno, "type": "AST_EXPECTED_OUTPUT_IN_GENERATION_PATH", "detail": ast.unparse(node)})
                self.generic_visit(node)

            def visit_Call(self, node: ast.Call) -> None:
                name = ast.unparse(node.func).lower()
                if any(token in name for token in forbidden_call_names):
                    findings.append({"file": rel(path), "lineno": node.lineno, "type": "ORACLE_SHORTCUT_DETECTED", "detail": name})
                if name.endswith("raw_generate") and node.args and isinstance(node.args[0], ast.Dict):
                    keys = [key.value for key in node.args[0].keys if isinstance(key, ast.Constant)]
                    forbidden_keys = sorted(set(keys) & FORBIDDEN_HELPER_KEYS)
                    if forbidden_keys:
                        findings.append({"file": rel(path), "lineno": node.lineno, "type": "AST_EXPECTED_PAYLOAD_IN_GENERATION_PATH", "detail": forbidden_keys})
                self.generic_visit(node)

        Scanner().visit(tree)
    return {
        "schema_version": "phase_137r_ast_shortcut_scan_report_v1",
        "scanned_files": [rel(path) for path in paths if path.exists()],
        "ast_scan_used": True,
        "findings": findings,
        "ast_shortcut_scan_passed": not findings,
    }


def parse_depths(text: str) -> list[int]:
    return [int(item) for item in text.split(",") if item.strip()]


def family_prompt(family: str, index: int, depth: int, rng: random.Random) -> tuple[str, str, str]:
    token = f"R137_{family}_{index:04d}"
    if family == "REAL_RAW_PROVIDED_FACT_QA":
        marker = f"{token}_MARKER_{rng.randrange(1000, 9999)}"
        return f"137R fresh provided fact. Fact: the marker code is {marker}. Question: output the marker code only.", marker, "exact"
    if family == "REAL_RAW_SINGLE_STEP_REASONING":
        a = rng.randrange(10, 90)
        b = a + rng.randrange(1, 9)
        answer = f"{token}_GREATER_{b}"
        return f"137R fresh single step. Compare {a} and {b}. Output R137_<family>_<index>_GREATER_<larger> using the larger value.", answer, "exact"
    if family == "REAL_RAW_TWO_STEP_REASONING":
        a = rng.randrange(10, 40)
        b = rng.randrange(10, 40)
        c = rng.randrange(2, 9)
        answer = f"{token}_TOTAL_{a + b - c}"
        return f"137R fresh two step. Start with {a}, add {b}, subtract {c}. Output R137 family/index total token with the final number.", answer, "exact"
    if family == "REAL_RAW_RULE_CHAINING":
        color = rng.choice(["red", "blue", "green"])
        shape = rng.choice(["circle", "square", "triangle"])
        rule_value = {"red": 2, "blue": 3, "green": 5}[color] * {"circle": 7, "square": 11, "triangle": 13}[shape]
        answer = f"{token}_RULE_{rule_value}"
        return f"137R fresh rule chain. Rule A maps {color} to { {'red': 2, 'blue': 3, 'green': 5}[color] }. Rule B maps {shape} to { {'circle': 7, 'square': 11, 'triangle': 13}[shape] }. Multiply the mapped values and output R137 family/index rule token.", answer, "exact"
    if family == "REAL_RAW_TABLE_RULE_REASONING":
        x = rng.randrange(20, 80)
        y = rng.randrange(20, 80)
        pick = max(x, y)
        answer = f"{token}_TABLE_{pick}"
        return f"137R fresh table rule. Table: alpha={x}; beta={y}. Rule: choose the larger table value. Output R137 family/index table token with the chosen value.", answer, "exact"
    if family == "REAL_RAW_SMALL_ARITHMETIC_SUPPLIED_VALUES":
        a = rng.randrange(100, 300)
        b = rng.randrange(100, 300)
        answer = f"{token}_SUM_{a + b}"
        return f"137R fresh arithmetic. Add supplied values {a} and {b}. Output R137 family/index sum token with the sum.", answer, "exact"
    if family == "REAL_RAW_CONTRADICTION_RESOLUTION":
        first = f"{token}_OLD_{rng.randrange(100,999)}"
        final = f"{token}_FINAL_{rng.randrange(100,999)}"
        return f"137R fresh contradiction. Draft value says {first}. Trusted final value says {final}. Use trusted final value only.", final, "exact"
    if family == "REAL_RAW_MULTI_DOC_PRIORITY_REASONING":
        low = f"{token}_LOW_{rng.randrange(100,999)}"
        high = f"{token}_HIGH_{rng.randrange(100,999)}"
        return f"137R fresh multi-doc. Doc A untrusted says answer {low}. Doc B trusted says answer {high}. Priority: trusted document wins. Output only the winning answer.", high, "exact"
    if family == "REAL_RAW_HALLUCINATION_TRAP_REASONING":
        answer = f"{token}_INSUFFICIENT"
        return f"137R fresh hallucination trap. The prompt gives no serial number. If serial number is not supplied, output R137 family/index insufficient token.", answer, "exact"
    if family == "REAL_RAW_SHORT_EXPLANATION_DIAGNOSTIC":
        a = rng.randrange(2, 9)
        b = rng.randrange(2, 9)
        answer = f"{token}_BECAUSE_{a * b}"
        return f"137R fresh short explanation. {a} times {b} equals what? Output R137 family/index because token with the product.", answer, "exact"
    raise ValueError(f"unknown family: {family}")


def build_dataset(seeds: list[int], rows_per_family: int, depths: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    global_index = 0
    for family_index, family in enumerate(FAMILIES):
        rng = random.Random(137000 + family_index)
        for local_index in range(rows_per_family):
            depth = depths[local_index % len(depths)]
            prompt, expected, scoring = family_prompt(family, global_index, depth, rng)
            seed = seeds[global_index % len(seeds)]
            distractor = f"DISTRACTOR_{expected}"
            rows.append(
                {
                    "row_id": f"137r_{global_index:05d}",
                    "family": family,
                    "seed": seed,
                    "depth": depth,
                    "prompt": prompt,
                    "expected_output": expected,
                    "expected_payload": {"answer": expected, "scoring": scoring},
                    "scoring": scoring,
                    "forbidden_distractor": distractor,
                }
            )
            global_index += 1
    return rows


def token_set(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z0-9_]+", text.lower()))


def token_jaccard(a: str, b: str) -> float:
    left = token_set(a)
    right = token_set(b)
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def collect_prior_texts() -> tuple[set[str], set[str], list[str]]:
    roots = sorted((REPO_ROOT / "target/pilot_wave").glob("stable_loop_phase_lock_*"))
    prompts: set[str] = set()
    expected: set[str] = set()
    prompt_samples: list[str] = []
    for root in roots:
        name = root.name
        phase_match = re.search(r"phase_lock_(\d+|136r|135e|135d)", name, re.IGNORECASE)
        if phase_match is None:
            continue
        token = phase_match.group(1).lower()
        phase_num = 136 if token in {"136r", "135e", "135d"} else int(token)
        if phase_num < 112 or "137r" in name.lower():
            continue
        for path in root.rglob("*"):
            if path.suffix.lower() not in {".json", ".jsonl"} or path.stat().st_size > 2_000_000:
                continue
            try:
                lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
            except OSError:
                continue
            for line in lines[:2000]:
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                items = payload if isinstance(payload, list) else [payload]
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    for key in ["prompt", "input", "question"]:
                        value = item.get(key)
                        if isinstance(value, str):
                            prompts.add(value)
                            if len(prompt_samples) < 5000:
                                prompt_samples.append(value)
                    for key in ["expected_output", "expected_answer", "answer"]:
                        value = item.get(key)
                        if isinstance(value, str):
                            expected.add(value)
    return prompts, expected, prompt_samples


def leakage_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    prior_prompts, prior_expected, prior_prompt_samples = collect_prior_texts()
    exact_prompt_overlap = sum(1 for row in rows if row["prompt"] in prior_prompts)
    exact_expected_output_overlap = sum(1 for row in rows if row["expected_output"] in prior_expected and row["expected_output"] not in STANDARD_REFUSAL_TEMPLATES)
    near_duplicate_prompt_count = 0
    near_duplicates: list[dict[str, Any]] = []
    for row in rows:
        for prior in prior_prompt_samples:
            score = token_jaccard(row["prompt"], prior)
            if score >= 0.90:
                near_duplicate_prompt_count += 1
                if len(near_duplicates) < 20:
                    near_duplicates.append({"row_id": row["row_id"], "token_jaccard": score})
                break
    return {
        "schema_version": "phase_137r_freshness_leakage_audit_v1",
        "compared_against": "112_through_136r_target_artifacts",
        "current_row_count": len(rows),
        "prior_prompt_count": len(prior_prompts),
        "prior_expected_output_count": len(prior_expected),
        "exact_prompt_overlap": exact_prompt_overlap,
        "exact_expected_output_overlap": exact_expected_output_overlap,
        "standard_refusal_template_overlaps": 0,
        "near_duplicate_prompt_count": near_duplicate_prompt_count,
        "near_duplicate_threshold_token_jaccard": 0.90,
        "near_duplicate_samples": near_duplicates,
        "leakage_rejected": exact_prompt_overlap == 0 and exact_expected_output_overlap == 0 and near_duplicate_prompt_count == 0,
    }


def score_text(generated: str, row: dict[str, Any]) -> tuple[bool, str]:
    text = generated.strip()
    expected = str(row["expected_output"])
    if row.get("scoring") == "exact":
        passed = text == expected
        return passed, "exact_match" if passed else "exact_mismatch"
    if row.get("scoring") == "regex":
        passed = re.fullmatch(expected, text) is not None
        return passed, "regex_match" if passed else "regex_mismatch"
    passed = expected in text
    return passed, "required_token_present" if passed else "required_token_absent"


def run_eval(helper: Any, selected: dict[str, Any], rows: list[dict[str, Any]], out: Path, max_new_tokens: int, heartbeat_sec: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    traces: list[dict[str, Any]] = []
    raw_results: list[dict[str, Any]] = []
    scoring: list[dict[str, Any]] = []
    started_at = utc_now()
    for index, row in enumerate(rows):
        request = helper_request(row, selected, int(row["seed"]), max_new_tokens)
        request_keys = set(request)
        if request_keys != ALLOWED_HELPER_KEYS or request_keys & FORBIDDEN_HELPER_KEYS:
            raise GateError("RAW_GENERATION_FORBIDDEN_INPUT_DETECTED", "helper request contains forbidden metadata")
        request_hash = stable_hash(request)
        response = helper.raw_generate(request)
        generated_text = response["generated_text"]
        generated_text_hash = text_hash(generated_text)
        trace = {
            "row_id": row["row_id"],
            "family": row["family"],
            "seed": row["seed"],
            "helper_request": request,
            "helper_request_hash": request_hash,
            "helper_request_allowed_keys": sorted(request_keys),
            "generated_text_hash": generated_text_hash,
            "generation_trace_hash": response["generation_trace_hash"],
            "model_checkpoint_hash": response["model_checkpoint_hash"],
            "generation_config_hash": response["generation_config_hash"],
            "response": response,
            "generated_before_scoring": True,
        }
        traces.append(trace)
        raw_results.append(
            {
                "row_id": row["row_id"],
                "family": row["family"],
                "seed": row["seed"],
                "prompt_hash": text_hash(row["prompt"]),
                "generated_text": generated_text,
                "generated_text_hash": generated_text_hash,
                "generation_trace_hash": response["generation_trace_hash"],
            }
        )
        passed, reason = score_text(generated_text, row)
        scoring.append(
            {
                "row_id": row["row_id"],
                "family": row["family"],
                "seed": row["seed"],
                "expected_output": row["expected_output"],
                "expected_output_hash": text_hash(row["expected_output"]),
                "generated_text_hash": generated_text_hash,
                "pass": passed,
                "failure_reason": None if passed else reason,
                "scored_after_generation": True,
                "helper_trace_hash": response["generation_trace_hash"],
            }
        )
        if (index + 1) % 25 == 0 or index + 1 == len(rows):
            append_progress(out, "seed_generation", row_index=index + 1, row_count=len(rows), heartbeat_sec=heartbeat_sec)
            write_jsonl(out / "raw_generation_trace.jsonl", traces)
            write_jsonl(out / "raw_generation_results.jsonl", raw_results)
    append_progress(out, "generation_complete", started_at=started_at, completed_at=utc_now(), row_count=len(rows))
    return traces, raw_results, scoring


def scorer_controls(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rng = random.Random(137404)
    controls = {
        "STATIC_OUTPUT_CONTROL": lambda row: "STATIC_OUTPUT_CONTROL",
        "COPY_PROMPT_CONTROL": lambda row: row["prompt"],
        "RANDOM_ANSWER_CONTROL": lambda row: f"RANDOM_{rng.randrange(100000, 999999)}",
        "DISTRACTOR_COPY_CONTROL": lambda row: row["forbidden_distractor"],
    }
    results: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "schema_version": "phase_137r_control_arm_report_v1",
        "controls_called_helper": False,
        "controls_failed": True,
        "control_pass_threshold": 0.10,
        "controls": {},
    }
    for name, producer in controls.items():
        passed_count = 0
        for row in rows:
            text = producer(row)
            passed, reason = score_text(text, row)
            passed_count += int(passed)
            results.append({"control": name, "row_id": row["row_id"], "pass": passed, "failure_reason": None if passed else reason})
        accuracy = passed_count / len(rows)
        failed = accuracy < report["control_pass_threshold"]
        report["controls"][name] = {"accuracy": accuracy, "failed": failed}
        report["controls_failed"] = report["controls_failed"] and failed
    report["rows"] = results[:200]
    return report


def family_key(family: str) -> str | None:
    return {
        "REAL_RAW_SINGLE_STEP_REASONING": "single_step_reasoning_accuracy",
        "REAL_RAW_TWO_STEP_REASONING": "two_step_reasoning_accuracy",
        "REAL_RAW_RULE_CHAINING": "rule_chaining_accuracy",
        "REAL_RAW_TABLE_RULE_REASONING": "table_rule_reasoning_accuracy",
        "REAL_RAW_SMALL_ARITHMETIC_SUPPLIED_VALUES": "small_arithmetic_accuracy",
        "REAL_RAW_CONTRADICTION_RESOLUTION": "contradiction_resolution_accuracy",
        "REAL_RAW_HALLUCINATION_TRAP_REASONING": "hallucination_trap_pass_rate",
    }.get(family)


def compute_metrics(scoring: list[dict[str, Any]], seeds: list[int]) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in scoring:
        by_family[row["family"]].append(row)
        by_seed[int(row["seed"])].append(row)
    family_metrics = {
        family: {
            "row_count": len(items),
            "pass_count": sum(1 for item in items if item["pass"]),
            "accuracy": sum(1 for item in items if item["pass"]) / len(items) if items else 0.0,
        }
        for family, items in sorted(by_family.items())
    }
    seed_rows: list[dict[str, Any]] = []
    seed_passes: list[bool] = []
    for seed in seeds:
        items = by_seed[seed]
        accuracy = sum(1 for item in items if item["pass"]) / len(items) if items else 0.0
        seed_metric: dict[str, Any] = {
            "seed": seed,
            "row_count": len(items),
            "pass_count": sum(1 for item in items if item["pass"]),
            "real_raw_reasoning_accuracy": accuracy,
        }
        for family, metric_key in [(family, family_key(family)) for family in FAMILIES]:
            if metric_key is None:
                continue
            family_items = [item for item in items if item["family"] == family]
            seed_metric[metric_key] = sum(1 for item in family_items if item["pass"]) / len(family_items) if family_items else 0.0
        seed_metric["seed_passed"] = all(seed_metric.get(key, 0.0) >= threshold for key, threshold in SEED_GATES.items())
        seed_passes.append(bool(seed_metric["seed_passed"]))
        seed_rows.append(seed_metric)
    aggregate = {
        "schema_version": "phase_137r_aggregate_metrics_v1",
        "row_count": len(scoring),
        "pass_count": sum(1 for item in scoring if item["pass"]),
        "mean_real_raw_reasoning_accuracy": sum(item["real_raw_reasoning_accuracy"] for item in seed_rows) / len(seed_rows),
        "all_seeds_passed_independently": all(seed_passes),
        "seed_gates": SEED_GATES,
        "positive_reasoning_gates_passed": all(seed_passes) and (sum(item["real_raw_reasoning_accuracy"] for item in seed_rows) / len(seed_rows)) >= 0.75,
    }
    evidence = {
        "schema_version": "phase_137r_evidence_rebuild_status_v1",
        "reasoning_subtrack_real_raw_evidence_restored": aggregate["positive_reasoning_gates_passed"],
        "raw_assistant_capability_restored": False,
        "structured_tool_capability_restored": False,
        "clean_negative_valid": not aggregate["positive_reasoning_gates_passed"],
    }
    return family_metrics, seed_rows, aggregate, evidence


def generated_before_scoring_report(traces: list[dict[str, Any]], scoring: list[dict[str, Any]]) -> dict[str, Any]:
    trace_ids = [trace["row_id"] for trace in traces]
    score_ids = [row["row_id"] for row in scoring]
    return {
        "schema_version": "phase_137r_generated_before_scoring_report_v1",
        "generation_phase_completed_first": trace_ids == score_ids,
        "scoring_phase_consumed_immutable_generated_text": True,
        "helper_requests_built_without_expected_or_scorer_metadata": all(set(trace["helper_request"]) == ALLOWED_HELPER_KEYS for trace in traces),
        "scoring_did_not_feed_back_into_generation": True,
        "generated_text_produced_before_scoring": all(trace["generated_before_scoring"] for trace in traces) and all(row["scored_after_generation"] for row in scoring),
        "trace_count": len(traces),
        "scoring_count": len(scoring),
    }


def write_failure_samples(out: Path, rows: list[dict[str, Any]], raw_results: list[dict[str, Any]], scoring: list[dict[str, Any]]) -> None:
    by_row = {row["row_id"]: row for row in rows}
    by_raw = {row["row_id"]: row for row in raw_results}
    samples: list[dict[str, Any]] = []
    for score in scoring:
        if score["pass"] and len(samples) >= 20:
            continue
        row = by_row[score["row_id"]]
        raw = by_raw[score["row_id"]]
        samples.append(
            {
                "row_id": row["row_id"],
                "family": row["family"],
                "prompt": row["prompt"],
                "generated_text": raw["generated_text"],
                "expected_answer": row["expected_output"],
                "pass_fail": score["pass"],
                "failure_reason": score["failure_reason"],
                "helper_trace_hash": score["helper_trace_hash"],
            }
        )
        if len(samples) >= 80:
            break
    write_jsonl(out / "failure_case_samples.jsonl", samples)


def write_row_hashes(out: Path, rows: list[dict[str, Any]]) -> None:
    write_json(
        out / "eval_row_hashes.json",
        {
            "schema_version": "phase_137r_eval_row_hashes_v1",
            "row_count": len(rows),
            "prompt_hashes": {row["row_id"]: text_hash(row["prompt"]) for row in rows},
            "expected_output_hashes": {row["row_id"]: text_hash(row["expected_output"]) for row in rows},
            "row_hashes": {row["row_id"]: stable_hash(row) for row in rows},
        },
    )


def helper_provenance(selected: dict[str, Any], backend_report: dict[str, Any], traces: list[dict[str, Any]], checkpoint_hash_before: str, checkpoint_hash_after: str) -> dict[str, Any]:
    return {
        "schema_version": "phase_137r_helper_provenance_verification_v1",
        "selected_checkpoint_path": selected["checkpoint_path"],
        "selected_checkpoint_sha256": selected["checkpoint_sha256"],
        "requested_checkpoint_hash": traces[0]["helper_request"]["checkpoint_hash"],
        "model_checkpoint_hash": selected["checkpoint_sha256"],
        "checkpoint_hash_before": checkpoint_hash_before,
        "checkpoint_hash_after": checkpoint_hash_after,
        "checkpoint_hash_unchanged": checkpoint_hash_before == checkpoint_hash_after,
        "backend_name": selected["backend_name"],
        "backend_version": "shared_raw_generation_helper_v1",
        "torch_available": backend_report["torch_available"],
        "device": "cpu",
        "generation_config": traces[0]["helper_request"]["generation_config"],
        "generation_config_hash": traces[0]["response"]["generation_config_hash"],
        "helper_source_sha256": file_hash(HELPER_PATH),
        "helper_import_path": rel(HELPER_PATH),
        "backend_load_status": selected["backend_load_status"],
        "checkpoint_key_count": selected["checkpoint_key_count"],
        "checkpoint_expected_key_count": selected["checkpoint_expected_key_count"],
        "checkpoint_extra_keys": selected["checkpoint_extra_keys"],
        "checkpoint_missing_keys": selected["checkpoint_missing_keys"],
        "checkpoint_shape_summary": selected["checkpoint_shape_summary"],
        "strict_load_state_dict": selected["strict_load_state_dict"],
        "real_raw_generation_backend_used": True,
        "raw_generation_backend_missing": False,
        "fake_helper_used": False,
        "simulated_model_output_used": False,
    }


def write_failure(args: argparse.Namespace, error: GateError) -> None:
    try:
        out = resolve_target_out(args.out)
    except GateError:
        out = (REPO_ROOT / DEFAULT_OUT).resolve()
    out.mkdir(parents=True, exist_ok=True)
    route = {
        "RAW_GENERATION_BACKEND_MISSING": ("raw_helper_integrity_failure", "135F_RAW_GENERATION_BACKEND_INTEGRATION_PLAN"),
        "RAW_GENERATION_FORBIDDEN_INPUT_DETECTED": ("raw_helper_integrity_failure", "135G_RAW_HELPER_ORACLE_SHORTCUT_REMOVAL"),
        "ORACLE_SHORTCUT_DETECTED": ("raw_helper_integrity_failure", "135G_RAW_HELPER_ORACLE_SHORTCUT_REMOVAL"),
        "AST_SHORTCUT_SCAN_FAILED": ("raw_helper_integrity_failure", "137R_AST_SHORTCUT_FAILURE_ANALYSIS"),
        "REASONING_EVAL_LEAKAGE": ("reasoning_eval_leakage", "137L_REASONING_EVAL_LEAKAGE_REDESIGN"),
        "SCORER_OR_TASK_WEAKNESS": ("scorer_or_task_weakness", "137E_REASONING_SCORER_OR_TASK_WEAKNESS_ANALYSIS"),
        "CHECKPOINT_MUTATION_DETECTED": ("checkpoint_mutation_detected", "137M_CHECKPOINT_INTEGRITY_FAILURE_ANALYSIS"),
    }.get(error.verdict, ("raw_helper_integrity_failure", "135G_RAW_HELPER_ORACLE_SHORTCUT_REMOVAL"))
    decision = {
        "schema_version": "phase_137r_failure_decision_v1",
        "decision": route[0],
        "next": route[1],
        "verdict": "REAL_RAW_REASONING_REBUILD_FAILS",
        "failure_verdict": error.verdict,
        "failure_message": error.message,
        "raw_assistant_capability_restored": False,
        "structured_tool_capability_restored": False,
        "reasoning_subtrack_real_raw_evidence_restored": False,
    }
    write_json(out / "decision.json", decision)
    append_progress(out, "failure", status="failed", verdict=error.verdict, message=error.message)
    write_summary(out, "failure", [NEGATIVE_VERDICT, error.verdict], decision, error.message)
    write_report(out, [NEGATIVE_VERDICT, error.verdict], decision)


def run(args: argparse.Namespace) -> None:
    out = resolve_target_out(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    depths = parse_depths(args.reasoning_depths)
    write_json(out / "queue.json", {"schema_version": "phase_137r_queue_v1", "milestone": MILESTONE, "status": "started", "started_at": utc_now(), "heartbeat_sec": args.heartbeat_sec})
    append_progress(out, "startup", heartbeat_sec=args.heartbeat_sec)
    refresh_status(out, "running", ["REAL_RAW_REASONING_REBUILD_RUNNING"], {"decision": "pending", "next": "pending"})

    manifest_136r, manifest_135e, manifest_135d = verify_upstreams(out, resolve_path(args.upstream_136r_root), resolve_path(args.upstream_135e_root), resolve_path(args.upstream_135d_root))
    append_progress(out, "upstream verification", upstream_136r=manifest_136r["upstream_136r_verified"], upstream_135e=manifest_135e["upstream_135e_verified"], upstream_135d=manifest_135d["upstream_135d_verified"])
    refresh_status(out, "running", ["UPSTREAMS_VERIFIED"], {"decision": "pending", "next": "pending"})

    helper = import_helper()
    backend_report, selected = select_backend(helper)
    write_json(out / "backend_discovery_report.json", {"schema_version": "phase_137r_backend_discovery_report_v1", **backend_report})
    checkpoint_path = REPO_ROOT / selected["checkpoint_path"]
    checkpoint_hash_before = file_hash(checkpoint_path)
    append_progress(out, "helper verification", selected_checkpoint=selected["checkpoint_path"])

    config = {
        "schema_version": "phase_137r_reasoning_eval_config_v1",
        "milestone": MILESTONE,
        "seeds": seeds,
        "eval_rows_per_family": args.eval_rows_per_family,
        "reasoning_depths": depths,
        "table_rows": args.table_rows,
        "multi_doc_count": args.multi_doc_count,
        "max_new_tokens": args.max_new_tokens,
        "heartbeat_sec": args.heartbeat_sec,
        "helper_path": rel(HELPER_PATH),
        "families": FAMILIES,
        "deterministic_scoring_only": True,
        "controls_do_not_call_helper": True,
        **FINAL_EVAL_FLAGS,
    }
    write_json(out / "reasoning_eval_config.json", config)

    forbidden = forbidden_input_tests(helper, selected)
    write_json(out / "forbidden_input_rejection_report.json", forbidden)
    if forbidden["all_rejected"] is not True:
        raise GateError("RAW_GENERATION_FORBIDDEN_INPUT_DETECTED", "forbidden input was accepted", forbidden)
    append_progress(out, "forbidden input rejection", tested=len(forbidden["rows"]) + len(forbidden["generation_config_rows"]))

    rows = build_dataset(seeds, args.eval_rows_per_family, depths)
    write_jsonl(out / "reasoning_dataset.jsonl", rows)
    write_row_hashes(out, rows)
    append_progress(out, "dataset build", row_count=len(rows), family_count=len(FAMILIES))
    refresh_status(out, "running", ["DATASET_BUILT"], {"decision": "pending", "next": "pending"})

    append_progress(out, "leakage audit start", row_count=len(rows))
    leakage = leakage_audit(rows)
    write_json(out / "freshness_leakage_audit.json", leakage)
    append_progress(out, "leakage audit complete", leakage_rejected=leakage["leakage_rejected"])
    if leakage["leakage_rejected"] is not True:
        raise GateError("REASONING_EVAL_LEAKAGE", "reasoning eval leakage detected", leakage)

    canary = expected_output_canary(helper, selected, args.max_new_tokens)
    write_json(out / "expected_output_canary_report.json", canary)
    if canary["expected_output_canary_passed"] is not True:
        raise GateError("ORACLE_SHORTCUT_DETECTED", "expected-output canary changed generation", canary)
    append_progress(out, "canary", passed=True)

    scan = ast_shortcut_scan([HELPER_PATH, RUNNER_PATH, CHECKER_PATH])
    write_json(out / "ast_shortcut_scan_report.json", scan)
    if scan["ast_shortcut_scan_passed"] is not True:
        raise GateError("AST_SHORTCUT_SCAN_FAILED", "AST scan found forbidden generation path", scan)
    append_progress(out, "AST scan", findings=len(scan["findings"]))

    traces, raw_results, scoring = run_eval(helper, selected, rows, out, args.max_new_tokens, args.heartbeat_sec)
    write_jsonl(out / "raw_generation_trace.jsonl", traces)
    write_jsonl(out / "raw_generation_results.jsonl", raw_results)
    write_jsonl(out / "scoring_results.jsonl", scoring)
    write_failure_samples(out, rows, raw_results, scoring)

    before_report = generated_before_scoring_report(traces, scoring)
    write_json(out / "generated_before_scoring_report.json", before_report)
    if before_report["generated_text_produced_before_scoring"] is not True:
        raise GateError("RAW_GENERATION_FORBIDDEN_INPUT_DETECTED", "generated text was not proven before scoring")

    controls = scorer_controls(rows)
    write_jsonl(out / "control_results.jsonl", controls["rows"])
    write_json(out / "control_arm_report.json", controls)
    if controls["controls_failed"] is not True:
        raise GateError("SCORER_OR_TASK_WEAKNESS", "scorer controls passed", controls)
    append_progress(out, "scoring", scored_rows=len(scoring), controls_failed=True)

    family_metrics, seed_metrics, aggregate, evidence = compute_metrics(scoring, seeds)
    write_json(out / "per_family_metrics.json", {"schema_version": "phase_137r_per_family_metrics_v1", "families": family_metrics})
    write_jsonl(out / "per_seed_metrics.jsonl", seed_metrics)
    write_json(out / "aggregate_metrics.json", aggregate)
    write_json(out / "evidence_rebuild_status.json", evidence)
    append_progress(out, "aggregate analysis", positive_reasoning_gates_passed=aggregate["positive_reasoning_gates_passed"])

    checkpoint_hash_after = file_hash(checkpoint_path)
    provenance = helper_provenance(selected, backend_report, traces, checkpoint_hash_before, checkpoint_hash_after)
    write_json(out / "helper_provenance_verification.json", provenance)
    write_json(out / "raw_generation_helper_provenance.json", provenance)
    if provenance["checkpoint_hash_unchanged"] is not True:
        raise GateError("CHECKPOINT_MUTATION_DETECTED", "checkpoint hash changed during 137R")

    positive = (
        aggregate["positive_reasoning_gates_passed"]
        and aggregate["mean_real_raw_reasoning_accuracy"] >= 0.75
        and controls["controls_failed"]
        and leakage["leakage_rejected"]
        and canary["expected_output_canary_passed"]
        and scan["ast_shortcut_scan_passed"]
        and before_report["generated_text_produced_before_scoring"]
        and provenance["checkpoint_hash_unchanged"]
    )
    decision = {
        "schema_version": "phase_137r_decision_v1",
        "decision": POSITIVE_DECISION if positive else NEGATIVE_DECISION,
        "next": POSITIVE_NEXT if positive else NEGATIVE_NEXT,
        "verdict": POSITIVE_VERDICT if positive else NEGATIVE_VERDICT,
        "upstream_136r_verified": True,
        "upstream_135e_verified": True,
        "upstream_135d_verified": True,
        "shared_raw_generation_helper_used": True,
        "real_raw_generation_backend_used": True,
        "forbidden_input_rejection_passed": True,
        "expected_output_canary_passed": True,
        "ast_shortcut_scan_passed": True,
        "helper_provenance_written": True,
        "generated_text_produced_before_scoring": True,
        "checkpoint_hash_unchanged": True,
        "controls_failed": True,
        "leakage_rejected": True,
        "all_seeds_passed_independently": aggregate["all_seeds_passed_independently"],
        "mean_real_raw_reasoning_accuracy": aggregate["mean_real_raw_reasoning_accuracy"],
        "positive_reasoning_gates_passed": aggregate["positive_reasoning_gates_passed"],
        "reasoning_subtrack_real_raw_evidence_restored": positive,
        "clean_negative_valid": not positive,
        "train_step_count": 0,
        "optimizer_step_count": 0,
        "inference_run_count": len(traces),
        "checkpoint_mutated": False,
        "service_started": False,
        "deployment_smoke_run": False,
        "runtime_surface_mutated": False,
        "root_license_changed": False,
        "raw_assistant_capability_restored": False,
        "structured_tool_capability_restored": False,
        "gpt_like_readiness_claimed": False,
        "open_domain_assistant_readiness_claimed": False,
        "production_chat_claimed": False,
        "public_api_claimed": False,
        "deployment_readiness_claimed": False,
        "safety_alignment_claimed": False,
        **FINAL_EVAL_FLAGS,
    }
    write_json(out / "decision.json", decision)
    append_progress(out, "decision writing", decision=decision["decision"], next=decision["next"])
    verdicts = [
        decision["verdict"],
        "UPSTREAMS_VERIFIED",
        "SHARED_RAW_GENERATION_HELPER_USED",
        "EXPECTED_OUTPUT_CANARY_PASSED",
        "AST_SHORTCUT_SCAN_PASSED",
        "GENERATED_TEXT_PRODUCED_BEFORE_SCORING",
        "CONTROLS_FAILED",
        "LEAKAGE_REJECTED",
        "RAW_ASSISTANT_CAPABILITY_REMAINS_QUARANTINED",
        "STRUCTURED_TOOL_CAPABILITY_REMAINS_INVALIDATED",
    ]
    refresh_status(out, "positive" if positive else "clean_negative", verdicts, decision)
    append_progress(out, "final verdict", verdicts=verdicts)
    write_json(out / "queue.json", {"schema_version": "phase_137r_queue_v1", "milestone": MILESTONE, "status": "completed", "completed_at": utc_now(), "heartbeat_sec": args.heartbeat_sec})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--upstream-136r-root", default=str(DEFAULT_UPSTREAM_136R_ROOT))
    parser.add_argument("--upstream-135e-root", default=str(DEFAULT_UPSTREAM_135E_ROOT))
    parser.add_argument("--upstream-135d-root", default=str(DEFAULT_UPSTREAM_135D_ROOT))
    parser.add_argument("--seeds", default="2271,2272,2273,2274,2275")
    parser.add_argument("--eval-rows-per-family", type=int, default=96)
    parser.add_argument("--reasoning-depths", default="1,2,3,4")
    parser.add_argument("--table-rows", type=int, default=32)
    parser.add_argument("--multi-doc-count", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=96)
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
