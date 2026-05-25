#!/usr/bin/env python3
"""147Z planning-only next decision after LM-style selected-byte scale confirm."""

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
MILESTONE = "STABLE_LOOP_PHASE_LOCK_147Z_LM_STYLE_CANONICAL_STRUCTURED_TEXT_DISTILLATION_NEXT_DECISION_PLAN"
DEFAULT_OUT = Path("target/pilot_wave/stable_loop_phase_lock_147z_lm_style_canonical_structured_text_distillation_next_decision_plan/smoke")
DEFAULT_147H_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_147h_lm_style_canonical_structured_text_distillation_scale_confirm/smoke")
RUNNER_PATH = Path(__file__).resolve()
CHECKER_PATH = REPO_ROOT / "scripts/probes/run_stable_loop_phase_lock_147z_lm_style_canonical_structured_text_distillation_next_decision_plan_check.py"
HELPER_PATH = REPO_ROOT / "scripts/probes/shared_raw_generation_helper.py"
DECISION = "full_selected_line_generation_prototype_plan_recommended"
SELECTED_OPTION = "full_selected_line_generation_prototype"
NEXT = "148A_FULL_SELECTED_LINE_GENERATION_PROTOTYPE"
BOUNDARY_TEXT = (
    "147Z is constrained model-facing distillation evidence only with canonical structured prompts only; "
    "not natural-language rule reasoning, not open-ended arbitration, not GPT-like/Gemma-like assistant capability, "
    "not production readiness, and not architecture superiority."
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
REQUIRED_148A_REPORTS = [
    "generation_prefix_audit.json",
    "raw_generation_audit.json",
    "decoding_audit.json",
    "full_line_generation_report.json",
    "generated_schema_report.json",
    "generation_input_audit.json",
    "anti_memorization_report.json",
    "ood_generation_family_report.json",
    "baseline_margin_report.json",
    "shortcut_scanner_report.json",
    "leakage_audit.json",
    "model_artifact_audit.json",
    "deterministic_replay_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
]
CONTROL_FAMILIES_148A = [
    "SAME_BLOCKS_DIFFERENT_PRIORITY",
    "SAME_PRIORITY_DIFFERENT_BLOCK_VALUES",
    "SAME_TEMPLATE_OPPOSITE_PRIORITY_WINNER",
    "PRIORITY_ORDER_HOLDOUT",
    "BLOCK_ORDER_HOLDOUT",
    "RULE_BLOCK_TYPE_COMBINATION_HOLDOUT",
    "INVALID_HIGH_PRIORITY_FALLTHROUGH_OOD",
    "STRUCTURAL_INVALID_PROMPT_OOD",
]
VALID_RAW_SELECTED_LINES = ["SELECTED=A", "SELECTED=B", "SELECTED=C", "SELECTED=fallback"]
NEGATIVE_ROUTES_148A = {
    "full_line_training_failure": "148B_FULL_LINE_TRAINING_FAILURE_ANALYSIS",
    "generated_schema_failure": "148C_FULL_LINE_SCHEMA_FAILURE_ANALYSIS",
    "selected_label_extraction_failure": "148D_SELECTED_LABEL_EXTRACTION_FAILURE_ANALYSIS",
    "model_shortcut_detected": "148E_FULL_LINE_SHORTCUT_ANALYSIS",
    "ood_full_line_generation_failure": "148F_FULL_LINE_OOD_ANALYSIS",
    "generation_input_leakage_detected": "148G_FULL_LINE_INPUT_LEAKAGE_ANALYSIS",
    "deterministic_replay_failure": "148I_FULL_LINE_DETERMINISM_FAILURE_ANALYSIS",
    "hidden_wrapper_detected": "148J_HIDDEN_SELECTED_PREFIX_WRAPPER_ANALYSIS",
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
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module == "shared_raw_generation_helper":
                    failures.append(f"forbidden_import:{rel(path)}:{module}")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in {"torch", "tensorflow", "shared_raw_generation_helper", "requests", "socket", "urllib", "http.client"}:
                        failures.append(f"forbidden_import:{rel(path)}:{alias.name}")
            if isinstance(node, ast.Call):
                name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ""
                if name in {"train", "fit", "forward", "backward", "step", "raw_generate", "load_checkpoint", "save_checkpoint", "urlopen"}:
                    failures.append(f"forbidden_call:{rel(path)}:{name}")
    return {"schema_version": "phase_147z_ast_scan_v1", "passed": not failures, "failures": failures}


def require_147h(root: Path) -> tuple[dict[str, Any], list[str]]:
    required = [
        "decision.json",
        "aggregate_metrics.json",
        "summary.json",
        "label_byte_generation_report.json",
        "same_model_family_audit.json",
        "generation_input_audit.json",
        "generated_schema_report.json",
        "ood_generation_family_report.json",
        "deterministic_replay_report.json",
    ]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        return {"root": rel(root), "missing_artifacts": missing}, [f"missing:{name}" for name in missing]

    decision = read_json(root / "decision.json")
    metrics = read_json(root / "aggregate_metrics.json")
    summary = read_json(root / "summary.json")
    label_byte = read_json(root / "label_byte_generation_report.json")
    same_model = read_json(root / "same_model_family_audit.json")
    generation_input = read_json(root / "generation_input_audit.json")
    schema = read_json(root / "generated_schema_report.json")
    ood = read_json(root / "ood_generation_family_report.json")
    replay = read_json(root / "deterministic_replay_report.json")

    checks = {
        "decision": decision.get("decision") == "lm_style_canonical_structured_text_distillation_scale_confirmed",
        "verdict": decision.get("verdict") == "INSTNCT_LM_STYLE_CANONICAL_STRUCTURED_TEXT_DISTILLATION_SCALE_CONFIRMED",
        "next": decision.get("next") == "147Z_LM_STYLE_CANONICAL_STRUCTURED_TEXT_DISTILLATION_NEXT_DECISION_PLAN",
        "selected_label_generation_accuracy": metrics.get("selected_label_generation_accuracy") == 1.0,
        "selected_label_byte_accuracy": metrics.get("selected_label_byte_accuracy") == 1.0,
        "final_value_from_generated_label_accuracy": metrics.get("final_value_from_generated_label_accuracy") == 1.0,
        "generated_output_schema_valid_rate": metrics.get("generated_output_schema_valid_rate") == 1.0,
        "ood_selected_accuracy": metrics.get("ood_selected_accuracy") == 1.0,
        "minimum_ood_family_accuracy": metrics.get("minimum_ood_family_accuracy") == 1.0,
        "minimum_ood_family_row_count": metrics.get("minimum_ood_family_row_count") == 140,
        "shuffled_target_control_accuracy": metrics.get("shuffled_target_control_accuracy") == 0.0,
        "generation_deterministic_replay_passed": metrics.get("generation_deterministic_replay_passed") is True,
        "model_generates_full_selected_line": metrics.get("model_generates_full_selected_line") is False,
        "schema_prefix_fixed_by_runner": metrics.get("schema_prefix_fixed_by_runner") is True,
        "selected_line_wrapper_deterministic": metrics.get("selected_line_wrapper_deterministic") is True,
        "label_byte_report_passed": label_byte.get("passed") is True,
        "label_byte_report_full_line_false": label_byte.get("model_generates_full_selected_line") is False,
        "label_byte_report_prefix_fixed": label_byte.get("schema_prefix_fixed_by_runner") is True,
        "same_model_family_audit_passed": same_model.get("passed") is True,
        "generation_input_audit_passed": generation_input.get("passed") is True,
        "generation_input_ends_with_selected_prefix": generation_input.get("eval_generation_input_ends_with_selected_prefix") is True,
        "schema_report_passed": schema.get("passed") is True,
        "ood_report_passed": ood.get("passed") is True and ood.get("minimum_ood_family_row_count") == 140,
        "deterministic_replay_report_passed": replay.get("passed") is True,
        "summary_passed": (
            summary.get("decision") == "lm_style_canonical_structured_text_distillation_scale_confirmed"
            and summary.get("verdict") == "INSTNCT_LM_STYLE_CANONICAL_STRUCTURED_TEXT_DISTILLATION_SCALE_CONFIRMED"
            and summary.get("next") == "147Z_LM_STYLE_CANONICAL_STRUCTURED_TEXT_DISTILLATION_NEXT_DECISION_PLAN"
        ),
    }
    failures = [key for key, passed in checks.items() if not passed]
    return {
        "schema_version": "phase_147z_upstream_147h_manifest_v1",
        "root": rel(root),
        "decision": decision,
        "aggregate_metrics": metrics,
        "summary": summary,
        "label_byte_generation_report": label_byte,
        "same_model_family_audit": same_model,
        "generation_input_audit": generation_input,
        "generated_schema_report": schema,
        "ood_generation_family_report": ood,
        "deterministic_replay_report": replay,
        "checks": checks,
        "failed_checks": failures,
        "passed": not failures,
    }, failures


def build_evidence_chain_summary(upstream: dict[str, Any]) -> dict[str, Any]:
    metrics = upstream["aggregate_metrics"]
    return {
        "schema_version": "phase_147z_evidence_chain_summary_v1",
        "143w_explicit_selected_pocket_binding_scale_confirmed": True,
        "144h_single_structured_rule_metadata_binding_scale_confirmed": True,
        "145h_mixed_structured_rule_composition_helper_scale_confirmed": True,
        "146h_raw_text_trainable_distillation_bridge_scale_confirmed": True,
        "147h_lm_style_selected_label_byte_generation_scale_confirmed": True,
        "147h_selected_label_generation_accuracy": metrics["selected_label_generation_accuracy"],
        "147h_selected_label_byte_accuracy": metrics["selected_label_byte_accuracy"],
        "147h_generated_output_schema_valid_rate": metrics["generated_output_schema_valid_rate"],
        "147h_ood_selected_accuracy": metrics["ood_selected_accuracy"],
        "147h_model_generates_full_selected_line": metrics["model_generates_full_selected_line"],
        "147h_schema_prefix_fixed_by_runner": metrics["schema_prefix_fixed_by_runner"],
        "remaining_gap": "bounded raw full SELECTED=<label> line generation without a hidden SELECTED= wrapper is untested",
        "passed": upstream["passed"],
    }


def build_lm_style_state_report(upstream: dict[str, Any]) -> dict[str, Any]:
    metrics = upstream["aggregate_metrics"]
    return {
        "schema_version": "phase_147z_lm_style_distillation_state_report_v1",
        "selected_label_byte_scale_confirmed": True,
        "fixed_selected_prefix_used_by_147h": True,
        "selected_line_wrapper_deterministic_in_147h": True,
        "full_selected_line_generation_untested": True,
        "natural_language_rule_reasoning_untested": True,
        "open_ended_arbitration_claimed": False,
        "gpt_like_or_gemma_like_capability_claimed": False,
        "production_readiness_claimed": False,
        "architecture_superiority_claimed": False,
        "source_metrics": {
            "selected_label_generation_accuracy": metrics["selected_label_generation_accuracy"],
            "selected_label_byte_accuracy": metrics["selected_label_byte_accuracy"],
            "final_value_from_generated_label_accuracy": metrics["final_value_from_generated_label_accuracy"],
            "ood_selected_accuracy": metrics["ood_selected_accuracy"],
        },
        "passed": True,
    }


def build_gap_analysis(upstream: dict[str, Any]) -> dict[str, Any]:
    metrics = upstream["aggregate_metrics"]
    return {
        "schema_version": "phase_147z_full_line_generation_gap_analysis_v1",
        "selected_label_byte_generation_scale_confirmed": metrics["selected_label_byte_accuracy"] == 1.0,
        "fixed_selected_prefix_used_by_147h": metrics["schema_prefix_fixed_by_runner"] is True,
        "model_generates_full_selected_line": metrics["model_generates_full_selected_line"],
        "full_selected_line_generation_untested": True,
        "hidden_wrapper_risk_for_148a": True,
        "post_generation_repair_risk_for_148a": True,
        "schema_memorization_risk_for_148a": True,
        "opaque_value_generation_remains_out_of_scope": True,
        "natural_language_rule_reasoning_untested": True,
        "open_ended_arbitration_claimed": False,
        "gpt_like_or_gemma_like_capability_claimed": False,
        "recommended_gap_closure": "148A must remove the fixed SELECTED= prefix and score strict schema from raw model continuation.",
        "passed": True,
    }


def build_decision_matrix() -> dict[str, Any]:
    rows = [
        {
            "option": "full_selected_line_generation_prototype",
            "recommended": True,
            "score": 1.0,
            "reason": "Closes the exact 147H gap by requiring raw SELECTED=<label> line generation from <OUTPUT> without a hidden prefix wrapper.",
            "primary_risks": ["hidden_wrapper_detected", "post_generation_repair", "schema_memorization"],
        },
        {
            "option": "multi_token_schema_generation_prototype",
            "recommended": False,
            "score": 0.74,
            "reason": "Useful later, but full selected-line generation is the smaller bounded bridge after selected-byte generation.",
        },
        {
            "option": "controlled_natural_language_wrapper_later_plan",
            "recommended": False,
            "score": 0.36,
            "reason": "Too early; full selected-line schema generation is still untested under canonical structured prompts.",
        },
        {
            "option": "scale_selected_byte_bridge_further",
            "recommended": False,
            "score": 0.42,
            "reason": "147H already scale-confirmed selected-byte generation at perfect measured accuracy under the current controls.",
        },
        {
            "option": "stop_at_selected_label_byte_generation",
            "recommended": False,
            "score": 0.1,
            "reason": "Stops before the next model-facing sequence generation gap is tested.",
        },
    ]
    return {
        "schema_version": "phase_147z_next_decision_matrix_v1",
        "selected_option": SELECTED_OPTION,
        "options": rows,
        "recommendation": {
            "decision": DECISION,
            "selected_option": SELECTED_OPTION,
            "next": NEXT,
        },
        "passed": True,
    }


def build_target_148a_plan() -> dict[str, Any]:
    return {
        "schema_version": "phase_147z_target_148a_milestone_plan_v1",
        "milestone": NEXT,
        "implementation_ready": True,
        "milestone_type": "executable_prototype",
        "expected_positive_route": {
            "decision": "full_selected_line_generation_prototype_positive",
            "verdict": "INSTNCT_FULL_SELECTED_LINE_GENERATION_PROTOTYPE_POSITIVE",
            "next": "148H_FULL_SELECTED_LINE_GENERATION_SCALE_CONFIRM",
        },
        "boundary": (
            "148A remains constrained model-facing distillation evidence only with canonical structured prompts only; "
            "not natural-language rule reasoning, not open-ended arbitration, not GPT-like/Gemma-like assistant capability, "
            "not production readiness, and not architecture superiority."
        ),
        "intended_primitive": [
            "canonical structured prompt",
            "runner-local PyTorch byte-level autoregressive model",
            "raw model continuation generates full SELECTED=<label> line",
            "strict schema validation from raw generated text",
            "deterministic candidate-value copy",
        ],
        "generation_input_policy": {
            "eval_generation_input_must_end_exactly_with": "<OUTPUT>\\n",
            "forbidden_eval_generation_input_suffix": "<OUTPUT>\\nSELECTED=",
            "eval_generation_input_ends_with_output_delimiter": True,
            "eval_generation_input_contains_selected_prefix": False,
            "runner_prepends_selected_prefix": False,
            "runner_wraps_label_byte": False,
            "deterministic_selected_line_wrapper_used": False,
            "model_generates_selected_prefix": True,
            "model_generates_full_selected_line": True,
        },
        "valid_raw_generated_outputs": VALID_RAW_SELECTED_LINES,
        "invalid_raw_generated_outputs": [
            "SELECTED=pocket_a",
            "ANSWER=...",
            "winner=pocket_*",
            "selected_pocket_id=...",
            "malformed labels",
            "multiple SELECTED lines",
            "extra text before or after selected line",
        ],
        "raw_generation_policy": {
            "raw_generated_text_stored": True,
            "schema_scored_from_raw_generated_text": True,
            "only_allowed_postprocess": "strip trailing newline",
            "post_generation_repair_used": False,
            "selected_line_extracted_from_substring": False,
            "casing_repair_used": False,
            "prefix_repair_used": False,
        },
        "decoding_policy": {
            "autoregressive_generation_used": True,
            "forced_selected_prefix_used": False,
            "constrained_label_only_decoding_used": False,
            "stop_on_newline_or_max_len": True,
            "max_new_bytes_required": True,
        },
        "final_value_policy": {
            "direct_opaque_value_token_generation_required": False,
            "final_value_from_generated_line_accuracy": "computed by deterministic copy from the candidate line selected by raw generated SELECTED=<label>",
        },
        "required_reports": REQUIRED_148A_REPORTS,
        "required_gates": {
            "selected_prefix_generation_accuracy": {">=": 0.70},
            "selected_label_generation_accuracy": {">=": 0.70},
            "full_selected_line_exact_match_rate": {">=": 0.70},
            "generated_output_schema_valid_rate": {">=": 0.80},
            "final_value_from_generated_line_accuracy": {">=": 0.70},
            "extra_text_generation_rate": {"<=": 0.20},
            "multiple_selected_line_rate": {"==": 0.0},
            "answer_value_generation_rate": {"==": 0.0},
            "selected_pocket_id_generation_rate": {"==": 0.0},
            "ood_full_line_accuracy": {">=": 0.50},
            "full_line_generation_accuracy_over_best_baseline": {">=": 0.10},
            "shuffled_target_control_accuracy": {"<=": 0.35},
            "shortcut_scanner_violation_count": {"==": 0},
            "train_eval_prompt_overlap_count": {"==": 0},
            "train_ood_prompt_overlap_count": {"==": 0},
            "value_token_overlap_train_test_rate": {"==": 0.0},
            "generation_deterministic_replay_passed": {"==": True},
        },
        "audit_hard_gates": {
            "eval_generation_input_ends_with_output_delimiter": True,
            "eval_generation_input_contains_selected_prefix": False,
            "runner_prepends_selected_prefix": False,
            "model_generates_selected_prefix": True,
            "model_generates_full_selected_line": True,
            "deterministic_selected_line_wrapper_used": False,
            "autoregressive_generation_used": True,
            "forced_selected_prefix_used": False,
            "constrained_label_only_decoding_used": False,
            "stop_on_newline_or_max_len": True,
            "raw_generated_text_stored": True,
            "schema_scored_from_raw_generated_text": True,
            "post_generation_repair_used": False,
            "selected_line_extracted_from_substring": False,
            "casing_repair_used": False,
            "prefix_repair_used": False,
        },
        "required_control_families": CONTROL_FAMILIES_148A,
        "clean_negative_routes": NEGATIVE_ROUTES_148A,
    }


def build_anti_oracle_requirements() -> dict[str, Any]:
    return {
        "schema_version": "phase_147z_anti_oracle_requirements_v1",
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
        ],
        "generation_input_forbidden_suffixes": ["<OUTPUT>\\nSELECTED="],
        "hidden_wrapper_forbidden": True,
        "runner_prepends_selected_prefix": False,
        "deterministic_selected_line_wrapper_used": False,
        "prefix_repair_used": False,
        "casing_repair_used": False,
        "selected_line_extracted_from_substring": False,
        "schema_scored_from_raw_generated_text_required": True,
        "external_api_forbidden": True,
        "pretrained_model_forbidden": True,
        "model_download_forbidden": True,
        "passed": True,
    }


def build_risk_register() -> dict[str, Any]:
    return {
        "schema_version": "phase_147z_risk_register_v1",
        "risks": [
            {
                "risk": "hidden_selected_prefix_wrapper",
                "severity": "high",
                "mitigation": "148A must prove eval generation input contains no SELECTED= prefix and runner_prepends_selected_prefix=false.",
                "negative_route": "148J_HIDDEN_SELECTED_PREFIX_WRAPPER_ANALYSIS",
            },
            {
                "risk": "post_generation_repair_masks_invalid_raw_output",
                "severity": "high",
                "mitigation": "Score schema from raw generated text only; allow only trailing newline stripping.",
                "negative_route": "148C_FULL_LINE_SCHEMA_FAILURE_ANALYSIS",
            },
            {
                "risk": "schema_memorization_without structured decision sensitivity",
                "severity": "medium",
                "mitigation": "Carry forward same-template opposite-priority and same-blocks-different-priority controls.",
                "negative_route": "148E_FULL_LINE_SHORTCUT_ANALYSIS",
            },
            {
                "risk": "OOD full-line generation collapse",
                "severity": "medium",
                "mitigation": "Gate ood_full_line_accuracy and required holdout control families.",
                "negative_route": "148F_FULL_LINE_OOD_ANALYSIS",
            },
            {
                "risk": "overclaiming beyond canonical structured prompts",
                "severity": "medium",
                "mitigation": "Boundary wording is required in docs, decision, summary, and report.",
                "negative_route": "148E_FULL_LINE_SHORTCUT_ANALYSIS",
            },
        ],
        "passed": True,
    }


def build_report(out: Path, upstream: dict[str, Any]) -> str:
    metrics = upstream["aggregate_metrics"]
    return f"""# {MILESTONE}

## Decision

decision = {DECISION}
selected_option = {SELECTED_OPTION}
next = {NEXT}

## Upstream 147H Evidence

147H scale-confirmed runner-local LM-style selected-label byte/token prediction after a fixed `SELECTED=` prefix.

- selected_label_generation_accuracy = {metrics['selected_label_generation_accuracy']}
- selected_label_byte_accuracy = {metrics['selected_label_byte_accuracy']}
- final_value_from_generated_label_accuracy = {metrics['final_value_from_generated_label_accuracy']}
- generated_output_schema_valid_rate = {metrics['generated_output_schema_valid_rate']}
- ood_selected_accuracy = {metrics['ood_selected_accuracy']}
- minimum_ood_family_accuracy = {metrics['minimum_ood_family_accuracy']}
- minimum_ood_family_row_count = {metrics['minimum_ood_family_row_count']}
- shuffled_target_control_accuracy = {metrics['shuffled_target_control_accuracy']}
- generation_deterministic_replay_passed = {str(metrics['generation_deterministic_replay_passed']).lower()}
- model_generates_full_selected_line = {str(metrics['model_generates_full_selected_line']).lower()}
- schema_prefix_fixed_by_runner = {str(metrics['schema_prefix_fixed_by_runner']).lower()}
- selected_line_wrapper_deterministic = {str(metrics['selected_line_wrapper_deterministic']).lower()}

## Recommendation

The next honest bridge is bounded full `SELECTED=<label>` line generation. 148A must remove the fixed `SELECTED=` prefix from generation input, forbid runner-side selected-line wrapping, and score strict schema validity from raw generated continuation.

## Target 148A Guardrails

- Generation input ends with `<OUTPUT>` only.
- The model must generate the `SELECTED=` prefix and label in the raw continuation.
- The runner must not prepend `SELECTED=`.
- The runner must not repair malformed output or extract a valid substring from junk.
- Final value remains deterministic copy from the generated selected line.
- Direct opaque value-token generation remains out of scope.

## Boundary

{BOUNDARY_TEXT}

Generated artifacts are under `{rel(out)}`.
"""


def run(args: argparse.Namespace) -> int:
    out = resolve_target_out(args.out)
    upstream_root = resolve_repo_path(args.upstream_147h_root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "progress.jsonl").write_text("", encoding="utf-8")
    append_progress(out, "started", milestone=MILESTONE, output_root=rel(out), upstream_147h_root=rel(upstream_root))

    queue = {
        "schema_version": "phase_147z_queue_v1",
        "milestone": MILESTONE,
        "queued_at": utc_now(),
        "planning_only": True,
        "artifact_only": True,
        "out": rel(out),
        "upstream_147h_root": rel(upstream_root),
        "heartbeat_sec": args.heartbeat_sec,
        "steps": [
            "verify upstream 147H evidence",
            "summarize LM-style selected-byte state",
            "compare next decision options",
            "write implementation-ready 148A full-line generation plan",
            "write decision artifacts",
        ],
    }
    write_json(out / "queue.json", queue)
    append_progress(out, "queue_written", queue_path=rel(out / "queue.json"))

    upstream, upstream_failures = require_147h(upstream_root)
    write_json(out / "upstream_147h_manifest.json", upstream)
    append_progress(out, "upstream_147h_verified", passed=not upstream_failures, failed_checks=upstream_failures)
    if upstream_failures:
        decision = {
            "schema_version": "phase_147z_decision_v1",
            "decision": "upstream_147h_evidence_failure",
            "selected_option": None,
            "next": "147H_LM_STYLE_CANONICAL_STRUCTURED_TEXT_DISTILLATION_SCALE_CONFIRM",
            "positive_gate_passed": False,
            "failed_checks": upstream_failures,
            "boundary": BOUNDARY_TEXT,
            **FALSE_FLAGS,
        }
        write_json(out / "decision.json", decision)
        append_progress(out, "failed", reason="upstream_147h_evidence_failure")
        return 1

    ast_scan = scan_ast()
    helper_ok = helper_unchanged_from_head()
    append_progress(out, "static_integrity_checked", ast_scan_passed=ast_scan["passed"], helper_unchanged_from_head=helper_ok)

    analysis_config = {
        "schema_version": "phase_147z_analysis_config_v1",
        "milestone": MILESTONE,
        "planning_only": True,
        "artifact_only": True,
        "raw_generate_allowed": False,
        "shared_helper_import_allowed": False,
        "helper_modification_allowed": False,
        "training_allowed": False,
        "torch_forward_pass_allowed": False,
        "checkpoint_mutation_allowed": False,
        "external_api_allowed": False,
        "external_model_download_allowed": False,
        "natural_language_input_claimed": False,
        "boundary": BOUNDARY_TEXT,
        "static_integrity": ast_scan,
        "shared_raw_generation_helper_unchanged_from_head": helper_ok,
        "source_sha256": {
            "runner": sha256_text(RUNNER_PATH.read_text(encoding="utf-8")),
            "checker": sha256_text(CHECKER_PATH.read_text(encoding="utf-8")) if CHECKER_PATH.exists() else "",
        },
        **FALSE_FLAGS,
    }
    write_json(out / "analysis_config.json", analysis_config)

    evidence = build_evidence_chain_summary(upstream)
    state = build_lm_style_state_report(upstream)
    gap = build_gap_analysis(upstream)
    decision_matrix = build_decision_matrix()
    target_148a = build_target_148a_plan()
    anti_oracle = build_anti_oracle_requirements()
    risk = build_risk_register()

    write_json(out / "evidence_chain_summary.json", evidence)
    write_json(out / "lm_style_distillation_state_report.json", state)
    write_json(out / "full_line_generation_gap_analysis.json", gap)
    write_json(out / "next_decision_matrix.json", decision_matrix)
    write_json(out / "target_148a_milestone_plan.json", target_148a)
    write_json(out / "anti_oracle_requirements.json", anti_oracle)
    write_json(out / "risk_register.json", risk)
    append_progress(out, "planning_artifacts_written", artifact_count=7)

    positive_gate_passed = (
        ast_scan["passed"]
        and helper_ok
        and evidence["passed"]
        and state["selected_label_byte_scale_confirmed"]
        and gap["full_selected_line_generation_untested"]
        and target_148a["implementation_ready"]
        and anti_oracle["passed"]
        and risk["passed"]
    )
    decision = {
        "schema_version": "phase_147z_decision_v1",
        "decision": DECISION if positive_gate_passed else "full_selected_line_generation_plan_blocked",
        "selected_option": SELECTED_OPTION if positive_gate_passed else None,
        "next": NEXT if positive_gate_passed else "147H_LM_STYLE_CANONICAL_STRUCTURED_TEXT_DISTILLATION_SCALE_CONFIRM",
        "verdict": "INSTNCT_FULL_SELECTED_LINE_GENERATION_PROTOTYPE_PLAN_RECOMMENDED" if positive_gate_passed else "INSTNCT_147Z_PLAN_BLOCKED",
        "positive_gate_passed": positive_gate_passed,
        "planning_only": True,
        "artifact_only": True,
        "boundary": BOUNDARY_TEXT,
        "target_148a_implementation_ready": target_148a["implementation_ready"],
        "selected_byte_bridge_scale_confirmed": True,
        "full_selected_line_generation_selected": True,
        "fixed_selected_prefix_removed_in_148a_plan": True,
        "hidden_wrapper_forbidden_in_148a_plan": True,
        "broad_capability_claimed": False,
        **FALSE_FLAGS,
    }
    summary = {
        "schema_version": "phase_147z_summary_v1",
        "milestone": MILESTONE,
        "decision": decision["decision"],
        "selected_option": decision["selected_option"],
        "next": decision["next"],
        "positive_gate_passed": positive_gate_passed,
        "upstream_147h_decision": upstream["decision"]["decision"],
        "upstream_147h_selected_label_byte_accuracy": upstream["aggregate_metrics"]["selected_label_byte_accuracy"],
        "upstream_147h_model_generates_full_selected_line": upstream["aggregate_metrics"]["model_generates_full_selected_line"],
        "recommended_148a_claim": "bounded raw full SELECTED=<label> line generation on canonical structured prompts",
        "boundary": BOUNDARY_TEXT,
        **FALSE_FLAGS,
    }
    write_json(out / "decision.json", decision)
    write_json(out / "summary.json", summary)
    write_text(out / "report.md", build_report(out, upstream))
    append_progress(out, "decision_written", decision=decision["decision"], positive_gate_passed=positive_gate_passed)
    append_progress(out, "completed", passed=positive_gate_passed)
    return 0 if positive_gate_passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output directory under target/pilot_wave")
    parser.add_argument("--upstream-147h-root", default=str(DEFAULT_147H_ROOT), help="Accepted 147H smoke artifact root")
    parser.add_argument("--heartbeat-sec", type=int, default=20, help="Progress heartbeat target in seconds")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
