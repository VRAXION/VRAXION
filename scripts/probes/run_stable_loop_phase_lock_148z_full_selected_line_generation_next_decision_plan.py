#!/usr/bin/env python3
"""148Z planning-only next decision after full selected-line scale confirm."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "STABLE_LOOP_PHASE_LOCK_148Z_FULL_SELECTED_LINE_GENERATION_NEXT_DECISION_PLAN"
DEFAULT_OUT = Path("target/pilot_wave/stable_loop_phase_lock_148z_full_selected_line_generation_next_decision_plan/smoke")
DEFAULT_148H_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_148h_full_selected_line_generation_scale_confirm/smoke")
RUNNER_PATH = Path(__file__).resolve()
CHECKER_PATH = REPO_ROOT / "scripts/probes/run_stable_loop_phase_lock_148z_full_selected_line_generation_next_decision_plan_check.py"
HELPER_PATH = REPO_ROOT / "scripts/probes/shared_raw_generation_helper.py"

DECISION = "bounded_decision_schema_generation_prototype_plan_recommended"
SELECTED_OPTION = "bounded_decision_schema_generation_prototype"
NEXT = "149A_BOUNDED_DECISION_SCHEMA_GENERATION_PROTOTYPE"
BOUNDARY_TEXT = (
    "148Z is constrained model-facing distillation evidence only with canonical structured prompts only, "
    "bounded decision schema planning only; not natural-language rule reasoning, not open-ended arbitration, "
    "not GPT-like/Gemma-like assistant capability, not production readiness, and not architecture superiority."
)
FALSE_FLAGS = {
    "reasoning_restored": False,
    "raw_assistant_capability_restored": False,
    "structured_tool_capability_restored": False,
    "rule_metadata_reasoning_claimed": False,
    "natural_language_rule_reasoning_claimed": False,
    "open_ended_arbitration_claimed": False,
    "gpt_like_readiness_claimed": False,
    "gemma_like_capability_claimed": False,
    "open_domain_assistant_readiness_claimed": False,
    "production_chat_claimed": False,
    "public_api_claimed": False,
    "deployment_readiness_claimed": False,
    "safety_alignment_claimed": False,
    "architecture_superiority_claimed": False,
}

VALID_149A_SELECTED_LINES = ["SELECTED=A", "SELECTED=B", "SELECTED=C", "SELECTED=fallback"]
VALID_149A_REASON_CODES = [
    "priority_quorum",
    "priority_recency",
    "priority_validity",
    "fallback_invalid_high_priority",
    "structural_invalid_fallback",
]
REQUIRED_149A_REPORTS = [
    "queue.json",
    "progress.jsonl",
    "upstream_148z_manifest.json",
    "curriculum_train.jsonl",
    "curriculum_validation.jsonl",
    "curriculum_test.jsonl",
    "curriculum_ood_test.jsonl",
    "sequence_train_corpus.txt",
    "sequence_validation_corpus.txt",
    "training_config.json",
    "lm_training_metrics.jsonl",
    "bounded_decision_schema_report.json",
    "selected_line_generation_report.json",
    "reason_code_generation_report.json",
    "generated_schema_report.json",
    "generation_input_audit.json",
    "raw_generation_audit.json",
    "decoding_audit.json",
    "final_value_copy_report.json",
    "label_distribution_report.json",
    "reason_code_distribution_report.json",
    "ood_bounded_schema_family_report.json",
    "anti_memorization_report.json",
    "baseline_margin_report.json",
    "shuffled_target_control_report.json",
    "shortcut_scanner_report.json",
    "leakage_audit.json",
    "value_token_leakage_report.json",
    "model_artifact_audit.json",
    "deterministic_replay_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
]
CONTROL_FAMILIES_149A = [
    "SAME_PROMPT_STRUCTURE_DIFFERENT_SELECTED_AND_REASON",
    "SAME_RULE_BLOCKS_DIFFERENT_PRIORITY",
    "SAME_PRIORITY_DIFFERENT_BLOCK_VALUES",
    "SAME_TEMPLATE_OPPOSITE_SELECTED_LINE",
    "SAME_SELECTED_DIFFERENT_REASON_CODE",
    "REASON_CODE_SHUFFLED_TARGET_CONTROL",
    "OOD_BOUNDED_SCHEMA_GENERATION_CONTROL",
    "GENERATION_INPUT_TARGET_LEAKAGE_CONTROL",
    "RAW_GENERATION_SCHEMA_DRIFT_CONTROL",
    "NATURAL_LANGUAGE_REASON_LEAKAGE_CONTROL",
]
NEGATIVE_ROUTES_149A = {
    "bounded_schema_training_failure": "149B_BOUNDED_SCHEMA_TRAINING_FAILURE_ANALYSIS",
    "generated_schema_failure": "149C_BOUNDED_SCHEMA_FORMAT_FAILURE_ANALYSIS",
    "selected_line_regression_failure": "149D_SELECTED_LINE_REGRESSION_ANALYSIS",
    "reason_code_generation_failure": "149E_REASON_CODE_GENERATION_FAILURE_ANALYSIS",
    "model_shortcut_detected": "149F_BOUNDED_SCHEMA_SHORTCUT_ANALYSIS",
    "ood_bounded_schema_failure": "149G_BOUNDED_SCHEMA_OOD_ANALYSIS",
    "generation_input_leakage_detected": "149H_BOUNDED_SCHEMA_INPUT_LEAKAGE_ANALYSIS",
    "deterministic_replay_failure": "149I_BOUNDED_SCHEMA_DETERMINISM_FAILURE_ANALYSIS",
    "natural_language_overclaim_detected": "149J_NATURAL_LANGUAGE_OVERCLAIM_ANALYSIS",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def resolve_repo_path(path: str | Path) -> Path:
    raw = Path(path)
    return raw if raw.is_absolute() else (REPO_ROOT / raw).resolve()


def resolve_target_out(path: str | Path) -> Path:
    resolved = resolve_repo_path(path)
    relative = resolved.relative_to(REPO_ROOT)
    parts = [part.lower() for part in relative.parts]
    if len(parts) < 2 or parts[0] != "target" or parts[1] != "pilot_wave":
        raise ValueError("--out must stay under target/pilot_wave")
    return resolved


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


def append_progress(out: Path, event: str, **details: Any) -> None:
    append_jsonl(out / "progress.jsonl", {"ts": utc_now(), "event": event, "details": details})


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def git_show_head(path: str) -> str:
    result = subprocess.run(["git", "show", f"HEAD:{path}"], cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    return result.stdout if result.returncode == 0 else ""


def helper_unchanged_from_head() -> bool:
    return HELPER_PATH.read_text(encoding="utf-8") == git_show_head("scripts/probes/shared_raw_generation_helper.py")


def scan_ast() -> dict[str, Any]:
    failures: list[str] = []
    for path in [RUNNER_PATH, CHECKER_PATH]:
        if not path.exists():
            failures.append(f"missing:{rel(path)}")
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "") == "shared_raw_generation_helper":
                failures.append(f"forbidden_import:{rel(path)}:shared_raw_generation_helper")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in {"torch", "tensorflow", "shared_raw_generation_helper", "requests", "socket", "urllib", "http.client"}:
                        failures.append(f"forbidden_import:{rel(path)}:{alias.name}")
            if isinstance(node, ast.Call):
                name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ""
                if name in {"train", "fit", "forward", "backward", "step", "raw_generate", "load_checkpoint", "save_checkpoint", "urlopen"}:
                    failures.append(f"forbidden_call:{rel(path)}:{name}")
    return {"schema_version": "phase_148z_ast_scan_v1", "passed": not failures, "failures": failures}


def require_148h(root: Path) -> tuple[dict[str, Any], list[str]]:
    required = [
        "decision.json",
        "aggregate_metrics.json",
        "summary.json",
        "generation_prefix_audit.json",
        "raw_generation_audit.json",
        "decoding_audit.json",
        "generated_schema_report.json",
        "ood_full_line_family_report.json",
        "model_artifact_audit.json",
        "deterministic_replay_report.json",
    ]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        return {"root": rel(root), "missing_artifacts": missing}, [f"missing:{name}" for name in missing]

    decision = read_json(root / "decision.json")
    metrics = read_json(root / "aggregate_metrics.json")
    summary = read_json(root / "summary.json")
    prefix = read_json(root / "generation_prefix_audit.json")
    raw = read_json(root / "raw_generation_audit.json")
    decoding = read_json(root / "decoding_audit.json")
    schema = read_json(root / "generated_schema_report.json")
    ood = read_json(root / "ood_full_line_family_report.json")
    artifact = read_json(root / "model_artifact_audit.json")
    replay = read_json(root / "deterministic_replay_report.json")

    checks = {
        "decision": decision.get("decision") == "full_selected_line_generation_scale_confirmed",
        "verdict": decision.get("verdict") == "INSTNCT_FULL_SELECTED_LINE_GENERATION_SCALE_CONFIRMED",
        "next": decision.get("next") == "148Z_FULL_SELECTED_LINE_GENERATION_NEXT_DECISION_PLAN",
        "full_selected_line_exact_match_rate": metrics.get("full_selected_line_exact_match_rate", 0.0) >= 0.998,
        "full_line_generation_accuracy": metrics.get("full_line_generation_accuracy", 0.0) >= 0.998,
        "selected_prefix_generation_accuracy": metrics.get("selected_prefix_generation_accuracy") == 1.0,
        "selected_label_generation_accuracy": metrics.get("selected_label_generation_accuracy", 0.0) >= 0.998,
        "final_value_from_generated_line_accuracy": metrics.get("final_value_from_generated_line_accuracy", 0.0) >= 0.998,
        "generated_output_schema_valid_rate": metrics.get("generated_output_schema_valid_rate") == 1.0,
        "ood_full_line_accuracy": metrics.get("ood_full_line_accuracy", 0.0) >= 0.995,
        "minimum_ood_family_accuracy": metrics.get("minimum_ood_family_accuracy", 0.0) >= 0.94,
        "minimum_ood_family_row_count": metrics.get("minimum_ood_family_row_count") == 140,
        "shuffled_target_control_accuracy": metrics.get("shuffled_target_control_accuracy", 1.0) <= 0.01,
        "generation_deterministic_replay_passed": metrics.get("generation_deterministic_replay_passed") is True,
        "eval_generation_input_contains_selected_prefix": metrics.get("eval_generation_input_contains_selected_prefix") is False,
        "runner_prepends_selected_prefix": metrics.get("runner_prepends_selected_prefix") is False,
        "deterministic_selected_line_wrapper_used": metrics.get("deterministic_selected_line_wrapper_used") is False,
        "post_generation_repair_used": metrics.get("post_generation_repair_used") is False,
        "selected_line_extracted_from_substring": metrics.get("selected_line_extracted_from_substring") is False,
        "first_byte_only_training_used": metrics.get("first_byte_only_training_used") is False,
        "constrained_label_only_decoding_used": metrics.get("constrained_label_only_decoding_used") is False,
        "prefix_audit_passed": prefix.get("passed") is True,
        "model_generates_full_selected_line": prefix.get("model_generates_full_selected_line") is True,
        "raw_audit_passed": raw.get("passed") is True,
        "decoding_audit_passed": decoding.get("passed") is True,
        "schema_report_passed": schema.get("passed") is True,
        "ood_report_passed": ood.get("passed") is True,
        "model_artifact_report_passed": artifact.get("passed") is True,
        "same_model_family_as_148a": artifact.get("same_model_family_as_148a") is True,
        "deterministic_replay_report_passed": replay.get("passed") is True,
        "summary_route": (
            summary.get("decision") == "full_selected_line_generation_scale_confirmed"
            and summary.get("verdict") == "INSTNCT_FULL_SELECTED_LINE_GENERATION_SCALE_CONFIRMED"
            and summary.get("next") == "148Z_FULL_SELECTED_LINE_GENERATION_NEXT_DECISION_PLAN"
        ),
    }
    failures = [key for key, passed in checks.items() if not passed]
    return {
        "schema_version": "phase_148z_upstream_148h_manifest_v1",
        "root": rel(root),
        "decision": decision,
        "aggregate_metrics": metrics,
        "summary": summary,
        "generation_prefix_audit": prefix,
        "raw_generation_audit": raw,
        "decoding_audit": decoding,
        "generated_schema_report": schema,
        "ood_full_line_family_report": ood,
        "model_artifact_audit": artifact,
        "deterministic_replay_report": replay,
        "checks": checks,
        "failed_checks": failures,
        "passed": not failures,
    }, failures


def build_evidence_chain_summary(upstream: dict[str, Any]) -> dict[str, Any]:
    metrics = upstream["aggregate_metrics"]
    return {
        "schema_version": "phase_148z_evidence_chain_summary_v1",
        "145h_mixed_structured_rule_composition_helper_scale_confirmed": True,
        "146h_raw_text_trainable_distillation_bridge_scale_confirmed": True,
        "147h_lm_style_selected_label_byte_generation_scale_confirmed": True,
        "148h_full_selected_line_generation_scale_confirmed": True,
        "148h_full_selected_line_exact_match_rate": metrics["full_selected_line_exact_match_rate"],
        "148h_ood_full_line_accuracy": metrics["ood_full_line_accuracy"],
        "148h_generated_output_schema_valid_rate": metrics["generated_output_schema_valid_rate"],
        "148h_no_hidden_prefix_or_wrapper": (
            metrics["eval_generation_input_contains_selected_prefix"] is False
            and metrics["runner_prepends_selected_prefix"] is False
            and metrics["deterministic_selected_line_wrapper_used"] is False
        ),
        "remaining_gap": "bounded multi-line decision schema generation with a controlled REASON_CODE is untested",
        "passed": upstream["passed"],
    }


def build_state_report(upstream: dict[str, Any]) -> dict[str, Any]:
    metrics = upstream["aggregate_metrics"]
    return {
        "schema_version": "phase_148z_full_selected_line_generation_state_report_v1",
        "full_selected_line_scale_confirmed": True,
        "model_generates_full_selected_line": True,
        "selected_line_wrapper_removed": True,
        "post_generation_repair_forbidden_and_absent": True,
        "bounded_reason_code_generation_untested": True,
        "multi_line_decision_schema_generation_untested": True,
        "natural_language_rule_reasoning_untested": True,
        "open_ended_arbitration_claimed": False,
        "gpt_like_or_gemma_like_capability_claimed": False,
        "production_readiness_claimed": False,
        "architecture_superiority_claimed": False,
        "source_metrics": {
            "full_selected_line_exact_match_rate": metrics["full_selected_line_exact_match_rate"],
            "selected_label_generation_accuracy": metrics["selected_label_generation_accuracy"],
            "final_value_from_generated_line_accuracy": metrics["final_value_from_generated_line_accuracy"],
            "ood_full_line_accuracy": metrics["ood_full_line_accuracy"],
        },
        "passed": True,
    }


def build_gap_analysis(upstream: dict[str, Any]) -> dict[str, Any]:
    metrics = upstream["aggregate_metrics"]
    return {
        "schema_version": "phase_148z_bounded_decision_schema_gap_analysis_v1",
        "full_selected_line_generation_scale_confirmed": metrics["full_selected_line_exact_match_rate"] >= 0.998,
        "full_selected_line_is_single_line_only": True,
        "bounded_reason_code_generation_untested": True,
        "answer_plus_bounded_reason_bridge_missing": True,
        "controlled_natural_language_still_too_early": True,
        "scale_current_full_line_bridge_further_lower_priority": True,
        "opaque_value_generation_remains_out_of_scope": True,
        "natural_language_rule_reasoning_untested": True,
        "open_ended_arbitration_claimed": False,
        "gpt_like_or_gemma_like_capability_claimed": False,
        "recommended_gap_closure": "149A should generate SELECTED=<label> plus bounded REASON_CODE=<code> under strict raw multi-line schema validation.",
        "passed": True,
    }


def build_decision_matrix() -> dict[str, Any]:
    rows = [
        {
            "option": "full_selected_line_plus_explanation_stub",
            "recommended": False,
            "score": 0.74,
            "reason": "A BECAUSE stub is useful later, but the wording risks natural-language overclaim unless reason values are first bounded.",
        },
        {
            "option": SELECTED_OPTION,
            "recommended": True,
            "score": 1.0,
            "reason": "Adds a controlled answer-plus-reason bridge with strict multi-line schema and bounded REASON_CODE values, while staying canonical and auditable.",
            "target_milestone": NEXT,
        },
        {
            "option": "controlled_natural_language_wrapper_later_plan",
            "recommended": False,
            "score": 0.35,
            "reason": "Too early; bounded schema reasons should be tested before any natural-language wrapper.",
        },
        {
            "option": "scale_current_full_line_bridge_further",
            "recommended": False,
            "score": 0.42,
            "reason": "148H already scale-confirmed full selected-line generation strongly under current controls.",
        },
        {
            "option": "stop_at_full_selected_line_generation",
            "recommended": False,
            "score": 0.1,
            "reason": "Stops before the next model-facing explanation-schema gap is tested.",
        },
    ]
    return {
        "schema_version": "phase_148z_next_decision_matrix_v1",
        "selected_option": SELECTED_OPTION,
        "options": rows,
        "recommendation": {
            "decision": DECISION,
            "selected_option": SELECTED_OPTION,
            "next": NEXT,
        },
        "passed": True,
    }


def build_target_149a_plan() -> dict[str, Any]:
    return {
        "schema_version": "phase_148z_target_149a_milestone_plan_v1",
        "milestone": NEXT,
        "implementation_ready": True,
        "milestone_type": "executable_prototype",
        "expected_positive_route": {
            "decision": "bounded_decision_schema_generation_prototype_positive",
            "verdict": "INSTNCT_BOUNDED_DECISION_SCHEMA_GENERATION_PROTOTYPE_POSITIVE",
            "next": "149H_BOUNDED_DECISION_SCHEMA_GENERATION_SCALE_CONFIRM",
        },
        "boundary": (
            "149A remains constrained model-facing distillation evidence only with canonical structured prompts only, "
            "bounded multi-line decision schema generation only; not natural-language rule reasoning, not open-ended arbitration, "
            "not GPT-like/Gemma-like assistant capability, not production readiness, and not architecture superiority."
        ),
        "intended_primitive": [
            "canonical structured prompt",
            "runner-local PyTorch byte-level autoregressive model",
            "raw model continuation generates SELECTED=<label>",
            "raw model continuation generates REASON_CODE=<bounded_code>",
            "strict schema validation from raw generated text",
            "deterministic candidate-value copy from generated selected line",
        ],
        "valid_schema": {
            "line_1": VALID_149A_SELECTED_LINES,
            "line_2": [f"REASON_CODE={code}" for code in VALID_149A_REASON_CODES],
            "line_count": 2,
            "extra_text_allowed": False,
        },
        "generation_input_policy": {
            "eval_generation_input_must_end_exactly_with": "<OUTPUT>\\n",
            "eval_generation_input_contains_selected_line": False,
            "eval_generation_input_contains_reason_code": False,
            "runner_prepends_selected_line": False,
            "runner_prepends_reason_code": False,
            "deterministic_schema_wrapper_used": False,
            "model_generates_selected_line": True,
            "model_generates_reason_code_line": True,
            "model_generates_full_bounded_schema": True,
        },
        "raw_generation_policy": {
            "raw_generated_text_stored": True,
            "schema_scored_from_raw_generated_text": True,
            "only_allowed_postprocess": "strip trailing newline",
            "post_generation_repair_used": False,
            "selected_line_extracted_from_substring": False,
            "reason_code_extracted_from_substring": False,
            "casing_repair_used": False,
            "prefix_repair_used": False,
            "label_repair_used": False,
            "reason_code_repair_used": False,
        },
        "decoding_policy": {
            "autoregressive_generation_used": True,
            "full_bounded_schema_target_used": True,
            "selected_line_only_training_used": False,
            "forced_selected_prefix_used": False,
            "forced_reason_code_prefix_used": False,
            "constrained_label_or_reason_only_decoding_used": False,
            "stop_on_double_newline_or_max_len": True,
            "max_new_bytes_required": True,
        },
        "final_value_policy": {
            "direct_opaque_value_token_generation_required": False,
            "final_value_from_generated_schema_accuracy": "computed by deterministic copy from the candidate line selected by raw generated SELECTED=<label>",
        },
        "required_reports": REQUIRED_149A_REPORTS,
        "required_gates": {
            "selected_line_generation_accuracy": {">=": 0.70},
            "reason_code_generation_accuracy": {">=": 0.60},
            "full_bounded_schema_exact_match_rate": {">=": 0.60},
            "generated_output_schema_valid_rate": {">=": 0.75},
            "final_value_from_generated_schema_accuracy": {">=": 0.70},
            "ood_bounded_schema_accuracy": {">=": 0.45},
            "selected_line_accuracy_over_best_baseline": {">=": 0.10},
            "reason_code_accuracy_over_best_baseline": {">=": 0.05},
            "shuffled_target_control_accuracy": {"<=": 0.35},
            "answer_value_generation_rate": {"==": 0.0},
            "selected_pocket_id_generation_rate": {"==": 0.0},
            "free_text_reason_generation_rate": {"==": 0.0},
            "extra_text_generation_rate": {"<=": 0.20},
            "shortcut_scanner_violation_count": {"==": 0},
            "train_eval_prompt_overlap_count": {"==": 0},
            "train_ood_prompt_overlap_count": {"==": 0},
            "value_token_overlap_train_test_rate": {"==": 0.0},
            "generation_deterministic_replay_passed": {"==": True},
        },
        "audit_hard_gates": {
            "eval_generation_input_contains_selected_line": False,
            "eval_generation_input_contains_reason_code": False,
            "runner_prepends_selected_line": False,
            "runner_prepends_reason_code": False,
            "deterministic_schema_wrapper_used": False,
            "model_generates_full_bounded_schema": True,
            "autoregressive_generation_used": True,
            "full_bounded_schema_target_used": True,
            "selected_line_only_training_used": False,
            "forced_selected_prefix_used": False,
            "forced_reason_code_prefix_used": False,
            "constrained_label_or_reason_only_decoding_used": False,
            "raw_generated_text_stored": True,
            "schema_scored_from_raw_generated_text": True,
            "post_generation_repair_used": False,
            "selected_line_extracted_from_substring": False,
            "reason_code_extracted_from_substring": False,
            "casing_repair_used": False,
            "prefix_repair_used": False,
            "label_repair_used": False,
            "reason_code_repair_used": False,
        },
        "required_control_families": CONTROL_FAMILIES_149A,
        "clean_negative_routes": NEGATIVE_ROUTES_149A,
    }


def build_anti_oracle_requirements() -> dict[str, Any]:
    return {
        "schema_version": "phase_148z_anti_oracle_requirements_v1",
        "model_input_forbidden_fields": [
            "selected_pocket_id",
            "winner=pocket_*",
            "final_selected",
            "derived_selected",
            "answer value",
            "gold value",
            "target value",
            "resolved output",
            "expected output",
            "teacher trace fields",
            "per-row oracle metadata",
            "ANSWER=",
            "GOLD=",
            "TARGET=",
            "EXPECTED=",
            "REASON_CODE=",
        ],
        "generation_input_forbidden_fields": ["SELECTED=", "REASON_CODE=", "ANSWER=", "GOLD=", "TARGET=", "EXPECTED="],
        "hidden_schema_wrapper_forbidden": True,
        "runner_prepends_selected_line": False,
        "runner_prepends_reason_code": False,
        "deterministic_schema_wrapper_used": False,
        "schema_scored_from_raw_generated_text_required": True,
        "natural_language_reason_generation_forbidden": True,
        "external_api_forbidden": True,
        "pretrained_model_forbidden": True,
        "model_download_forbidden": True,
        "passed": True,
    }


def build_risk_register() -> dict[str, Any]:
    return {
        "schema_version": "phase_148z_risk_register_v1",
        "risks": [
            {
                "risk": "bounded reason code becomes hidden oracle label",
                "severity": "high",
                "mitigation": "149A must generate REASON_CODE from prompt only; no reason code in generation input or model-facing prompt.",
                "negative_route": "149F_BOUNDED_SCHEMA_SHORTCUT_ANALYSIS",
            },
            {
                "risk": "runner repairs malformed multi-line schema",
                "severity": "high",
                "mitigation": "Score strict schema from raw generated text; allow only trailing newline stripping.",
                "negative_route": "149C_BOUNDED_SCHEMA_FORMAT_FAILURE_ANALYSIS",
            },
            {
                "risk": "natural-language overclaim via BECAUSE/free text",
                "severity": "high",
                "mitigation": "149A uses bounded REASON_CODE values only; free text reason generation rate must be zero.",
                "negative_route": "149J_NATURAL_LANGUAGE_OVERCLAIM_ANALYSIS",
            },
            {
                "risk": "selected-line capability regresses when adding reason code",
                "severity": "medium",
                "mitigation": "Gate selected_line_generation_accuracy separately from reason_code_generation_accuracy.",
                "negative_route": "149D_SELECTED_LINE_REGRESSION_ANALYSIS",
            },
            {
                "risk": "OOD schema family collapse",
                "severity": "medium",
                "mitigation": "Gate ood_bounded_schema_accuracy and per-family OOD report.",
                "negative_route": "149G_BOUNDED_SCHEMA_OOD_ANALYSIS",
            },
        ],
        "passed": True,
    }


def build_report(upstream: dict[str, Any]) -> str:
    metrics = upstream["aggregate_metrics"]
    return f"""# {MILESTONE}

## Decision

decision = {DECISION}
selected_option = {SELECTED_OPTION}
next = {NEXT}

## Upstream 148H Evidence

148H scale-confirmed bounded full `SELECTED=<label>` line generation on canonical structured prompts.

- full_selected_line_exact_match_rate = {metrics['full_selected_line_exact_match_rate']}
- selected_prefix_generation_accuracy = {metrics['selected_prefix_generation_accuracy']}
- selected_label_generation_accuracy = {metrics['selected_label_generation_accuracy']}
- final_value_from_generated_line_accuracy = {metrics['final_value_from_generated_line_accuracy']}
- generated_output_schema_valid_rate = {metrics['generated_output_schema_valid_rate']}
- ood_full_line_accuracy = {metrics['ood_full_line_accuracy']}
- minimum_ood_family_accuracy = {metrics['minimum_ood_family_accuracy']}
- shuffled_target_control_accuracy = {metrics['shuffled_target_control_accuracy']}
- generation_deterministic_replay_passed = {str(metrics['generation_deterministic_replay_passed']).lower()}

## Recommendation

The next bridge is `149A_BOUNDED_DECISION_SCHEMA_GENERATION_PROTOTYPE`: a canonical structured prompt should produce a strict bounded multi-line schema:

```text
SELECTED=<label>
REASON_CODE=<bounded_code>
```

This is not natural-language reasoning. `REASON_CODE` is a bounded audit tag, and final answer value remains deterministic copy from the generated selected line.

Boundary: {BOUNDARY_TEXT}
"""


def build_decision(upstream_failures: list[str]) -> dict[str, Any]:
    positive = not upstream_failures
    return {
        "schema_version": "phase_148z_decision_v1",
        "decision": DECISION if positive else "upstream_148h_evidence_mismatch",
        "selected_option": SELECTED_OPTION if positive else "blocked_by_upstream_148h_mismatch",
        "next": NEXT if positive else "148H_FULL_SELECTED_LINE_GENERATION_SCALE_CONFIRM_RECHECK",
        "positive_gate_passed": positive,
        "boundary": BOUNDARY_TEXT,
        **FALSE_FLAGS,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--upstream-148h-root", type=Path, default=DEFAULT_148H_ROOT)
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    args = parser.parse_args()

    out = resolve_target_out(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_text(out / "progress.jsonl", "")
    append_progress(out, "startup", milestone=MILESTONE, heartbeat_sec=args.heartbeat_sec)
    write_json(out / "queue.json", {"schema_version": "phase_148z_queue_v1", "milestone": MILESTONE, "status": "running"})

    upstream, upstream_failures = require_148h(resolve_repo_path(args.upstream_148h_root))
    write_json(out / "upstream_148h_manifest.json", upstream)
    append_progress(out, "upstream_checked", passed=not upstream_failures, failures=upstream_failures)
    if not helper_unchanged_from_head():
        raise RuntimeError("shared_raw_generation_helper.py changed from HEAD")

    ast_scan = scan_ast()
    write_json(
        out / "analysis_config.json",
        {
            "schema_version": "phase_148z_analysis_config_v1",
            "milestone": MILESTONE,
            "planning_only": True,
            "artifact_only": True,
            "training_allowed": False,
            "raw_generate_allowed": False,
            "shared_helper_import_allowed": False,
            "helper_modification_allowed": False,
            "torch_forward_pass_allowed": False,
            "checkpoint_mutation_allowed": False,
            "external_api_allowed": False,
            "external_model_download_allowed": False,
            "natural_language_input_claimed": False,
            "selected_option": SELECTED_OPTION,
            "next": NEXT,
            "boundary": BOUNDARY_TEXT,
            "ast_scan": ast_scan,
            **FALSE_FLAGS,
        },
    )
    append_progress(out, "analysis_config_written")

    evidence = build_evidence_chain_summary(upstream)
    state = build_state_report(upstream)
    gap = build_gap_analysis(upstream)
    matrix = build_decision_matrix()
    target = build_target_149a_plan()
    anti = build_anti_oracle_requirements()
    risks = build_risk_register()
    decision = build_decision(upstream_failures)
    summary = {
        "schema_version": "phase_148z_summary_v1",
        "milestone": MILESTONE,
        "decision": decision["decision"],
        "selected_option": decision["selected_option"],
        "next": decision["next"],
        "boundary": BOUNDARY_TEXT,
        "upstream_148h_passed": upstream["passed"],
        "target_149a_milestone_plan_implementation_ready": target["implementation_ready"],
        **FALSE_FLAGS,
    }

    artifacts = {
        "evidence_chain_summary.json": evidence,
        "full_selected_line_generation_state_report.json": state,
        "bounded_decision_schema_gap_analysis.json": gap,
        "next_decision_matrix.json": matrix,
        "target_149a_milestone_plan.json": target,
        "anti_oracle_requirements.json": anti,
        "risk_register.json": risks,
        "decision.json": decision,
        "summary.json": summary,
    }
    for name, payload in artifacts.items():
        write_json(out / name, payload)
        append_progress(out, "artifact_written", artifact=name)
    write_text(out / "report.md", build_report(upstream))

    queue = read_json(out / "queue.json")
    queue["status"] = "complete" if decision["positive_gate_passed"] else "blocked"
    queue["decision"] = decision["decision"]
    write_json(out / "queue.json", queue)
    append_progress(out, "complete", decision=decision["decision"], next=decision["next"])
    print(json.dumps({"decision": decision["decision"], "selected_option": decision["selected_option"], "next": decision["next"]}, indent=2, sort_keys=True))
    return 0 if decision["positive_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
