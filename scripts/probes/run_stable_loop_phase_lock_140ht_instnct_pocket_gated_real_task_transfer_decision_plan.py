#!/usr/bin/env python3
"""140HT planning-only transfer decision after minimal-marker bridge scale confirm."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "STABLE_LOOP_PHASE_LOCK_140HT_INSTNCT_POCKET_GATED_REAL_TASK_TRANSFER_DECISION_PLAN"
DEFAULT_OUT = Path("target/pilot_wave/stable_loop_phase_lock_140ht_instnct_pocket_gated_real_task_transfer_decision_plan/smoke")
DEFAULT_140HS_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_140hs_instnct_pocket_gated_minimal_marker_real_task_bridge_scale_confirm/smoke")
RUNNER_PATH = Path(__file__).resolve()
CHECKER_PATH = REPO_ROOT / "scripts/probes/run_stable_loop_phase_lock_140ht_instnct_pocket_gated_real_task_transfer_decision_plan_check.py"
POSITIVE_NEXT = "140U_INSTNCT_POCKET_GATED_REAL_TASK_TRANSFORM_BRIDGE_PROBE"
FALSE_FLAGS = {
    "reasoning_restored": False,
    "raw_assistant_capability_restored": False,
    "structured_tool_capability_restored": False,
    "gpt_like_readiness_claimed": False,
    "open_domain_assistant_readiness_claimed": False,
    "production_chat_claimed": False,
    "public_api_claimed": False,
    "deployment_readiness_claimed": False,
    "safety_alignment_claimed": False,
}
BOUNDARY_TEXT = (
    "140HT is planning-only and artifact-only after the positive 140HS minimal-marker "
    "real-task bridge scale confirm. It does not train, run inference, call helper "
    "generation, mutate checkpoints, modify helper/backend/runtime/release/product "
    "surfaces, start services, deploy, change root LICENSE, or claim GPT-like or "
    "broad assistant readiness."
)


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
    try:
        relative = resolved.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ValueError("--out must stay inside repo") from exc
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


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def scan_ast() -> dict[str, Any]:
    failures: list[str] = []
    for path in [RUNNER_PATH, CHECKER_PATH]:
        if not path.exists():
            failures.append(f"missing:{rel(path)}")
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("run_stable_loop_phase_lock_"):
                failures.append(f"old_runner_import:{rel(path)}:{node.module}")
            if isinstance(node, ast.Import) and any(alias.name == "torch" for alias in node.names):
                failures.append(f"torch_import:{rel(path)}")
            if isinstance(node, ast.Call):
                name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ""
                if name in {"train", "fit", "backward", "step", "raw_generate"}:
                    failures.append(f"forbidden_call:{rel(path)}:{name}")
    return {"schema_version": "phase_140ht_ast_scan_v1", "passed": not failures, "failures": failures}


def require_140hs(root: Path) -> dict[str, Any]:
    required = [
        "decision.json",
        "arm_comparison.json",
        "selection_report.json",
        "control_arm_report.json",
        "determinism_replay_report.json",
        "generated_before_scoring_report.json",
        "expected_output_canary_report.json",
        "explicit_marker_audit.json",
    ]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise RuntimeError(f"missing 140HS artifacts: {missing}")
    decision = read_json(root / "decision.json")
    comparison = read_json(root / "arm_comparison.json")
    selection = read_json(root / "selection_report.json")
    controls = read_json(root / "control_arm_report.json")
    replay = read_json(root / "determinism_replay_report.json")
    generated = read_json(root / "generated_before_scoring_report.json")
    canary = read_json(root / "expected_output_canary_report.json")
    audit = read_json(root / "explicit_marker_audit.json")
    if decision.get("decision") != "instnct_pocket_gated_minimal_marker_real_task_bridge_scale_confirmed":
        raise RuntimeError(f"bad 140HS decision: {decision.get('decision')}")
    if decision.get("next") != "140HT_INSTNCT_POCKET_GATED_REAL_TASK_TRANSFER_DECISION_PLAN":
        raise RuntimeError(f"bad 140HS next: {decision.get('next')}")
    gates = {
        "row_count": comparison.get("eval_row_count", 0) >= 2000,
        "family_count": comparison.get("family_count", 0) >= 6,
        "scaffold_count": comparison.get("scaffold_variant_count", 0) >= 40,
        "main_answer_value_accuracy": comparison.get("main_answer_value_accuracy", 0.0) >= 0.95,
        "main_pocket_writeback_rate": comparison.get("main_pocket_writeback_rate", 0.0) >= 0.95,
        "main_phase_transport_success_rate": comparison.get("main_phase_transport_success_rate", 0.0) >= 0.95,
        "main_contrast_group_accuracy": comparison.get("main_contrast_group_accuracy", 0.0) >= 0.95,
        "ablation_answer_value_accuracy": comparison.get("ablation_answer_value_accuracy", 1.0) <= 0.05,
        "direct_pocket_value_marker_rate": audit.get("direct_pocket_value_marker_rate", 1.0) == 0.0,
        "explicit_pocket_token_row_rate": audit.get("explicit_pocket_token_row_rate", 1.0) <= 0.10,
        "implicit_or_minimal_gate_row_rate": audit.get("implicit_or_minimal_gate_row_rate", 0.0) >= 0.90,
        "controls_failed": controls.get("controls_failed") is True,
        "deterministic_replay_passed": replay.get("deterministic_replay_passed") is True,
        "generated_before_scoring": generated.get("passed") is True,
        "canary_passed": canary.get("passed") is True,
        "mutation_selected_open_pocket": selection.get("selected_candidate") == "open_minimal_marker_all_payloads_scale",
        "gradient_used_false": selection.get("gradient_used") is False,
    }
    failed = [key for key, passed in gates.items() if not passed]
    if failed:
        raise RuntimeError(f"140HS gates failed: {failed}")
    return {
        "root": rel(root),
        "decision": decision.get("decision"),
        "verdict": decision.get("verdict"),
        "next": decision.get("next"),
        "eval_row_count": comparison.get("eval_row_count"),
        "family_count": comparison.get("family_count"),
        "scaffold_variant_count": comparison.get("scaffold_variant_count"),
        "main_answer_value_accuracy": comparison.get("main_answer_value_accuracy"),
        "main_exact_answer_accuracy": comparison.get("main_exact_answer_accuracy"),
        "main_pocket_writeback_rate": comparison.get("main_pocket_writeback_rate"),
        "main_phase_transport_success_rate": comparison.get("main_phase_transport_success_rate"),
        "main_contrast_group_accuracy": comparison.get("main_contrast_group_accuracy"),
        "ablation_answer_value_accuracy": comparison.get("ablation_answer_value_accuracy"),
        "ablation_pocket_writeback_rate": comparison.get("ablation_pocket_writeback_rate"),
        "pocket_ablation_delta_answer_value_accuracy": comparison.get("pocket_ablation_delta_answer_value_accuracy"),
        "direct_pocket_value_marker_rate": audit.get("direct_pocket_value_marker_rate"),
        "explicit_pocket_token_row_rate": audit.get("explicit_pocket_token_row_rate"),
        "explicit_gate_pocket_open_row_rate": audit.get("explicit_gate_pocket_open_row_rate"),
        "implicit_or_minimal_gate_row_rate": audit.get("implicit_or_minimal_gate_row_rate"),
        "visible_bypass_control_failed": comparison.get("visible_bypass_control_failed"),
        "noisy_distractor_control_failed": comparison.get("noisy_distractor_control_failed"),
        "every_seed_passed": comparison.get("every_seed_passed"),
        "deterministic_replay_passed": replay.get("deterministic_replay_passed"),
        "selected_candidate": selection.get("selected_candidate"),
        "fitness_margin": selection.get("fitness_margin"),
        "gradient_used": selection.get("gradient_used"),
        "gate_checks": gates,
    }


def evidence_chain(upstream: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "phase_140ht_evidence_chain_summary_v1",
        "current_state": "minimal-marker real-task bridge scale confirmed",
        "upstream_milestone": "140HS_INSTNCT_POCKET_GATED_MINIMAL_MARKER_REAL_TASK_BRIDGE_SCALE_CONFIRM",
        "evidence": {
            "eval_row_count": upstream["eval_row_count"],
            "family_count": upstream["family_count"],
            "scaffold_variant_count": upstream["scaffold_variant_count"],
            "main_answer_value_accuracy": upstream["main_answer_value_accuracy"],
            "main_pocket_writeback_rate": upstream["main_pocket_writeback_rate"],
            "main_phase_transport_success_rate": upstream["main_phase_transport_success_rate"],
            "main_contrast_group_accuracy": upstream["main_contrast_group_accuracy"],
            "ablation_answer_value_accuracy": upstream["ablation_answer_value_accuracy"],
            "pocket_ablation_delta_answer_value_accuracy": upstream["pocket_ablation_delta_answer_value_accuracy"],
            "direct_pocket_value_marker_rate": upstream["direct_pocket_value_marker_rate"],
            "explicit_pocket_token_row_rate": upstream["explicit_pocket_token_row_rate"],
            "implicit_or_minimal_gate_row_rate": upstream["implicit_or_minimal_gate_row_rate"],
            "every_seed_passed": upstream["every_seed_passed"],
            "deterministic_replay_passed": upstream["deterministic_replay_passed"],
            "selected_candidate": upstream["selected_candidate"],
            "gradient_used": upstream["gradient_used"],
        },
        "interpretation": [
            "Minimal-marker pocket copy/writeback survives scale.",
            "Closed-pocket ablation remains decision-critical.",
            "Visible wrong values and noisy distractors did not produce bypass.",
            "The next useful falsification is transfer: transform a pocket-carried source into a distinct target value.",
        ],
        "not_claimed": [
            "GPT-like readiness",
            "broad assistant capability",
            "open-domain value grounding",
            "architecture superiority",
            "production readiness",
        ],
    }


def bridge_gap_analysis(upstream: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "phase_140ht_bridge_gap_analysis_v1",
        "closed_gaps": [
            "Direct POCKET_VALUE main path removed.",
            "Literal GATE:POCKET_OPEN removed from main path.",
            "Minimal/natural gate survived scale.",
            "Closed-pocket ablation remained causal.",
            "Visible and noisy distractor controls failed.",
        ],
        "remaining_gaps": [
            "Current bridge primarily validates pocket value copy/writeback, not transform.",
            "No evidence yet that a pocket source can be mapped through a rule/table/composition into a distinct target.",
            "No evidence yet for multi-field or multi-pocket binding.",
            "No evidence yet for fully hidden gate or non-visible carrier extraction.",
        ],
        "why_not_more_scale": "140HS already confirmed the same minimal-marker copy bridge at scale; more same-shape scale has lower diagnostic value than transfer.",
        "next_gap_to_test": "real-task transform bridge",
        "supporting_metrics": {
            "main_answer_value_accuracy": upstream["main_answer_value_accuracy"],
            "direct_pocket_value_marker_rate": upstream["direct_pocket_value_marker_rate"],
            "explicit_pocket_token_row_rate": upstream["explicit_pocket_token_row_rate"],
            "implicit_or_minimal_gate_row_rate": upstream["implicit_or_minimal_gate_row_rate"],
        },
    }


def decision_matrix() -> dict[str, Any]:
    options = [
        {
            "option": "A_more_minimal_marker_scale",
            "recommendation": "reject_for_now",
            "pros": ["more rows/seeds could further stress stability"],
            "cons": ["same-shape copy bridge already scale-confirmed", "low new falsification value"],
            "diagnostic_value": "medium_low",
        },
        {
            "option": "B_real_task_transform_bridge",
            "recommendation": "select",
            "pros": ["tests transform rather than copy", "preserves pocket causality and helper-only gates", "directly addresses the next untested bridge gap"],
            "cons": ["higher clean-negative risk because transform may reveal copy-only shortcut"],
            "diagnostic_value": "high",
            "target_milestone": POSITIVE_NEXT,
        },
        {
            "option": "C_multi_field_multi_pocket_binding_bridge",
            "recommendation": "defer",
            "pros": ["tests richer binding"],
            "cons": ["should follow single-source transform evidence", "more failure modes make interpretation harder"],
            "diagnostic_value": "high_later",
        },
        {
            "option": "D_hidden_gate_implicit_carrier_bridge",
            "recommendation": "defer",
            "pros": ["reduces scaffolding further"],
            "cons": ["gate visibility is already minimal; transform gap is more immediate"],
            "diagnostic_value": "medium_later",
        },
        {
            "option": "E_integration_blocker_analysis",
            "recommendation": "not_selected",
            "pros": ["useful if upstream evidence were inconsistent"],
            "cons": ["140HS evidence is complete and positive"],
            "diagnostic_value": "low_now",
        },
    ]
    return {
        "schema_version": "phase_140ht_transfer_decision_matrix_v1",
        "selected_option": "B_real_task_transform_bridge",
        "recommended_next": POSITIVE_NEXT,
        "options": options,
    }


def transfer_requirements() -> dict[str, Any]:
    return {
        "schema_version": "phase_140ht_real_task_transfer_requirements_v1",
        "target_milestone": POSITIVE_NEXT,
        "must_test_transform_not_copy": True,
        "required_task_shapes": [
            "pocket carries source value and task requires distinct mapped target value",
            "table transform from pocket source",
            "rule transform from pocket source",
            "two-step pocket payload plus rule transform",
            "same source form can map to different target values across contrast groups",
        ],
        "required_controls": [
            "copy-only source value shortcut control",
            "visible target bypass control",
            "noisy distractor copy control",
            "closed-pocket ablation control",
            "prefix-only control",
            "train namespace replay control",
        ],
        "positive_gates": {
            "main_answer_value_accuracy_min": 0.60,
            "main_transform_accuracy_min": 0.60,
            "main_pocket_writeback_rate_min": 0.75,
            "ablation_answer_value_accuracy_max": 0.15,
            "pocket_ablation_delta_min": 0.35,
            "direct_pocket_value_marker_rate": 0.0,
            "visible_bypass_violation_rate": 0.0,
            "noisy_distractor_violation_rate": 0.0,
            "copy_only_shortcut_detected": False,
            "deterministic_replay_passed": True,
        },
    }


def anti_shortcuts() -> dict[str, Any]:
    return {
        "schema_version": "phase_140ht_anti_shortcut_requirements_v1",
        "reject": [
            "direct POCKET_VALUE main path",
            "literal source copy as target",
            "visible target bypass",
            "noisy distractor copy",
            "prefix-only success",
            "namespace-only success",
            "closed-pocket success",
            "post-generation repair",
            "oracle/rerank/verifier/LLM judge",
            "expected/scorer metadata in helper request",
            "determinism mismatch",
        ],
        "required_artifacts": [
            "copy_only_shortcut_report.json",
            "transform_binding_metrics.json",
            "pocket_ablation_results.jsonl",
            "visible_bypass_control_report.json",
            "noisy_distractor_control_report.json",
            "determinism_replay_report.json",
        ],
    }


def target_140u_plan() -> dict[str, Any]:
    return {
        "schema_version": "phase_140ht_target_140u_milestone_plan_v1",
        "milestone": POSITIVE_NEXT,
        "type": "executable helper-only transform bridge probe",
        "train_allowed": False,
        "helper_generation_allowed": True,
        "helper_backend_modification_allowed": False,
        "public_request_key_change_allowed": False,
        "source_checkpoint_mutation_allowed": False,
        "clean_negative_accepted": True,
        "required_design": {
            "natural_ish_task_text_primary": True,
            "direct_pocket_value_main_path_forbidden": True,
            "minimal_or_implicit_gate": True,
            "visible_wrong_values_present": True,
            "noisy_distractors_present": True,
            "closed_pocket_ablation_must_fail": True,
            "mutation_selection_must_prefer_correct_open_pocket_config": True,
            "transform_target_must_differ_from_pocket_source": True,
            "copy_only_shortcut_must_fail": True,
        },
        "required_metrics": [
            "main_answer_value_accuracy",
            "main_transform_accuracy",
            "main_pocket_writeback_rate",
            "ablation_answer_value_accuracy",
            "pocket_ablation_delta",
            "direct_pocket_value_marker_rate",
            "visible_bypass_violation_rate",
            "noisy_distractor_violation_rate",
            "copy_only_shortcut_detected",
            "deterministic_replay_passed",
        ],
        "failure_routes": {
            "copy_only_shortcut_detected": "140UC_COPY_ONLY_TRANSFER_FAILURE_ANALYSIS",
            "transform_binding_failure": "140UT_TRANSFORM_BINDING_FAILURE_ANALYSIS",
            "pocket_ablation_not_decision_critical": "140UJ_POCKET_CAUSALITY_FAILURE_ANALYSIS",
            "implicit_gate_failure": "140UI_IMPLICIT_GATE_FAILURE_ANALYSIS",
            "helper_integrity_failure": "135G_RAW_HELPER_ORACLE_SHORTCUT_REMOVAL",
        },
    }


def write_report(out: Path, decision: dict[str, Any], evidence: dict[str, Any], matrix: dict[str, Any]) -> None:
    e = evidence["evidence"]
    text = f"""# {MILESTONE}

Decision: `{decision['decision']}`

Verdict: `{decision['verdict']}`

Next: `{decision['next']}`

Boundary: {BOUNDARY_TEXT}

140HS evidence:

- eval rows: `{e['eval_row_count']}`
- families: `{e['family_count']}`
- scaffold variants: `{e['scaffold_variant_count']}`
- main answer value accuracy: `{e['main_answer_value_accuracy']}`
- main pocket writeback rate: `{e['main_pocket_writeback_rate']}`
- main phase transport success rate: `{e['main_phase_transport_success_rate']}`
- main contrast group accuracy: `{e['main_contrast_group_accuracy']}`
- ablation answer value accuracy: `{e['ablation_answer_value_accuracy']}`
- direct `POCKET_VALUE=` marker rate: `{e['direct_pocket_value_marker_rate']}`
- explicit `POCKET_` token row rate: `{e['explicit_pocket_token_row_rate']}`
- implicit/minimal gate row rate: `{e['implicit_or_minimal_gate_row_rate']}`
- deterministic replay passed: `{e['deterministic_replay_passed']}`

Selected next-step option: `{matrix['selected_option']}`

Rationale: same-shape minimal-marker copy/writeback has already been
scale-confirmed. The next useful falsification is a real-task transform bridge,
where the pocket source must map to a distinct target and copy-only source
reuse is rejected.

This remains constrained pocket-gated helper evidence, not GPT-like readiness,
not broad assistant capability, not production readiness, not public API
readiness, not deployment readiness, and not safety alignment.
"""
    write_text(out / "report.md", text)


def main() -> int:
    parser = argparse.ArgumentParser(description=MILESTONE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--upstream-140hs-root", type=Path, default=DEFAULT_140HS_ROOT)
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    args = parser.parse_args()

    out = resolve_target_out(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_text(out / "progress.jsonl", "")
    append_progress(out, "startup", milestone=MILESTONE, heartbeat_sec=args.heartbeat_sec)
    write_json(out / "queue.json", {"schema_version": "phase_140ht_queue_v1", "milestone": MILESTONE, "status": "running"})

    config = {
        "schema_version": "phase_140ht_analysis_config_v1",
        "milestone": MILESTONE,
        "boundary": BOUNDARY_TEXT,
        "planning_only": True,
        "artifact_only": True,
        "training_performed": False,
        "new_model_inference_run": False,
        "shared_helper_called": False,
        "helper_generation_called": False,
        "torch_forward_pass_run": False,
        "checkpoint_mutated": False,
        "helper_modified": False,
        "backend_modified": False,
        "runtime_surface_mutated": False,
        "release_surface_mutated": False,
        "product_surface_mutated": False,
        "root_license_changed": False,
        **FALSE_FLAGS,
    }
    write_json(out / "analysis_config.json", config)

    upstream = require_140hs(resolve_repo_path(args.upstream_140hs_root))
    write_json(out / "upstream_140hs_manifest.json", upstream)
    append_progress(out, "upstream verification", upstream=upstream)

    ast_report = scan_ast()
    write_json(out / "ast_shortcut_scan_report.json", ast_report)
    append_progress(out, "artifact loading", config_hash=stable_hash(config), ast_passed=ast_report["passed"])

    evidence = evidence_chain(upstream)
    gap = bridge_gap_analysis(upstream)
    matrix = decision_matrix()
    requirements = transfer_requirements()
    shortcuts = anti_shortcuts()
    target_plan = target_140u_plan()
    write_json(out / "evidence_chain_summary.json", evidence)
    append_progress(out, "evidence chain summary", row_count=upstream["eval_row_count"])
    write_json(out / "bridge_gap_analysis.json", gap)
    append_progress(out, "bridge gap analysis", next_gap=gap["next_gap_to_test"])
    write_json(out / "transfer_decision_matrix.json", matrix)
    append_progress(out, "decision matrix", selected=matrix["selected_option"])
    write_json(out / "real_task_transfer_requirements.json", requirements)
    write_json(out / "anti_shortcut_requirements.json", shortcuts)
    write_json(out / "target_140u_milestone_plan.json", target_plan)
    append_progress(out, "target 140U plan writing", next=target_plan["milestone"])

    write_json(out / "diagnostic_gap_register.json", {"schema_version": "phase_140ht_diagnostic_gap_register_v1", "gaps": ["No transform bridge measurement exists yet; 140HT only plans 140U.", "No broad assistant, hidden-state, or architecture superiority evidence is claimed."]})
    write_json(out / "risk_register.json", {"schema_version": "phase_140ht_risk_register_v1", "risks": [{"risk": "140U may expose copy-only shortcut rather than transform", "mitigation": "require copy-only shortcut report and transform accuracy"}, {"risk": "transform target may weaken pocket causality", "mitigation": "require closed-pocket ablation failure and deterministic replay"}]})

    decision = {
        "schema_version": "phase_140ht_decision_v1",
        "decision": "real_task_transform_bridge_recommended",
        "verdict": "INSTNCT_POCKET_GATED_REAL_TASK_TRANSFER_DECISION_COMPLETE",
        "next": POSITIVE_NEXT,
        "boundary": BOUNDARY_TEXT,
        "planning_only": True,
        "artifact_only": True,
        "clean_negative_valid": True,
        "selected_option": matrix["selected_option"],
        "architecture_superiority_claimed": False,
        "value_grounding_claimed": False,
        "pocket_mechanism_claimed": False,
        **FALSE_FLAGS,
    }
    write_json(out / "decision.json", decision)
    summary = {
        "schema_version": "phase_140ht_summary_v1",
        "milestone": MILESTONE,
        "status": "complete",
        "boundary": BOUNDARY_TEXT,
        "upstream": upstream,
        "selected_option": matrix["selected_option"],
        "recommended_next": POSITIVE_NEXT,
        "analysis_config_hash": stable_hash(config),
        **decision,
    }
    write_json(out / "summary.json", summary)
    write_report(out, decision, evidence, matrix)
    append_progress(out, "decision", decision=decision["decision"], next=decision["next"])
    append_progress(out, "final verdict", verdict=decision["verdict"])
    write_json(out / "queue.json", {"schema_version": "phase_140ht_queue_v1", "milestone": MILESTONE, "status": "complete", "decision": decision["decision"], "next": decision["next"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
