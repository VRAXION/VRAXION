#!/usr/bin/env python3
"""140W planning-only decision after transform bridge scale confirm."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "STABLE_LOOP_PHASE_LOCK_140W_INSTNCT_POCKET_GATED_MULTI_STEP_TRANSFER_DECISION_PLAN"
DEFAULT_OUT = Path("target/pilot_wave/stable_loop_phase_lock_140w_instnct_pocket_gated_multi_step_transfer_decision_plan/smoke")
DEFAULT_140V_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_140v_instnct_pocket_gated_real_task_transform_bridge_scale_confirm/smoke")
RUNNER_PATH = Path(__file__).resolve()
CHECKER_PATH = REPO_ROOT / "scripts/probes/run_stable_loop_phase_lock_140w_instnct_pocket_gated_multi_step_transfer_decision_plan_check.py"
POSITIVE_NEXT = "140X_INSTNCT_POCKET_GATED_MULTI_STEP_TRANSFER_PROBE"
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
    "140W is planning-only and artifact-only after the positive 140V real-task "
    "transform bridge scale confirm. It does not train, run inference, call helper "
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
    return {"schema_version": "phase_140w_ast_scan_v1", "passed": not failures, "failures": failures}


def require_140v(root: Path) -> dict[str, Any]:
    required = ["decision.json", "arm_comparison.json", "copy_only_shortcut_report.json", "selection_report.json", "determinism_replay_report.json"]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise RuntimeError(f"missing 140V artifacts: {missing}")
    decision = read_json(root / "decision.json")
    comparison = read_json(root / "arm_comparison.json")
    copy_report = read_json(root / "copy_only_shortcut_report.json")
    selection = read_json(root / "selection_report.json")
    replay = read_json(root / "determinism_replay_report.json")
    if decision.get("decision") != "instnct_pocket_gated_real_task_transform_bridge_scale_confirmed":
        raise RuntimeError(f"bad 140V decision: {decision.get('decision')}")
    if decision.get("next") != "140W_INSTNCT_POCKET_GATED_MULTI_STEP_TRANSFER_DECISION_PLAN":
        raise RuntimeError(f"bad 140V next: {decision.get('next')}")
    gates = {
        "row_count": comparison.get("eval_row_count", 0) >= 2500,
        "family_count": comparison.get("family_count", 0) >= 6,
        "scaffold_count": comparison.get("scaffold_variant_count", 0) >= 40,
        "main_answer_value_accuracy": comparison.get("main_answer_value_accuracy", 0.0) >= 0.85,
        "main_transform_accuracy": comparison.get("main_transform_accuracy", 0.0) >= 0.85,
        "main_pocket_writeback_rate": comparison.get("main_pocket_writeback_rate", 0.0) >= 0.85,
        "main_contrast_group_accuracy": comparison.get("main_contrast_group_accuracy", 0.0) >= 0.85,
        "ablation_answer_value_accuracy": comparison.get("ablation_answer_value_accuracy", 1.0) <= 0.10,
        "ablation_delta": comparison.get("pocket_ablation_delta_answer_value_accuracy", 0.0) > 0.60,
        "source_copy_shortcut_rate": comparison.get("source_copy_shortcut_rate", 1.0) == 0.0,
        "copy_only_shortcut_detected": copy_report.get("copy_only_shortcut_detected") is False,
        "direct_pocket_value_marker_rate": comparison.get("direct_pocket_value_marker_rate", 1.0) == 0.0,
        "visible_bypass_violation_rate": comparison.get("visible_bypass_violation_rate", 1.0) == 0.0,
        "noisy_distractor_violation_rate": comparison.get("noisy_distractor_violation_rate", 1.0) == 0.0,
        "every_seed_passed": comparison.get("every_seed_passed") is True,
        "deterministic_replay_passed": replay.get("deterministic_replay_passed") is True,
        "selected_candidate": selection.get("selected_candidate") == "open_transform_target_all_markers_scale",
    }
    failed = [key for key, passed in gates.items() if not passed]
    if failed:
        raise RuntimeError(f"140V gates failed: {failed}")
    return {
        "root": rel(root),
        "decision": decision.get("decision"),
        "verdict": decision.get("verdict"),
        "next": decision.get("next"),
        "eval_row_count": comparison.get("eval_row_count"),
        "family_count": comparison.get("family_count"),
        "scaffold_variant_count": comparison.get("scaffold_variant_count"),
        "main_answer_value_accuracy": comparison.get("main_answer_value_accuracy"),
        "main_transform_accuracy": comparison.get("main_transform_accuracy"),
        "main_pocket_writeback_rate": comparison.get("main_pocket_writeback_rate"),
        "main_contrast_group_accuracy": comparison.get("main_contrast_group_accuracy"),
        "ablation_answer_value_accuracy": comparison.get("ablation_answer_value_accuracy"),
        "pocket_ablation_delta_answer_value_accuracy": comparison.get("pocket_ablation_delta_answer_value_accuracy"),
        "source_copy_shortcut_rate": comparison.get("source_copy_shortcut_rate"),
        "copy_only_shortcut_detected": copy_report.get("copy_only_shortcut_detected"),
        "direct_pocket_value_marker_rate": comparison.get("direct_pocket_value_marker_rate"),
        "explicit_pocket_token_row_rate": comparison.get("explicit_pocket_token_row_rate"),
        "implicit_or_minimal_gate_row_rate": comparison.get("implicit_or_minimal_gate_row_rate"),
        "visible_bypass_violation_rate": comparison.get("visible_bypass_violation_rate"),
        "noisy_distractor_violation_rate": comparison.get("noisy_distractor_violation_rate"),
        "every_seed_passed": comparison.get("every_seed_passed"),
        "deterministic_replay_passed": replay.get("deterministic_replay_passed"),
        "selected_candidate": selection.get("selected_candidate"),
        "fitness_margin": selection.get("fitness_margin"),
        "gate_checks": gates,
    }


def evidence_summary(upstream: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "phase_140w_evidence_chain_summary_v1",
        "current_state": "real-task transform bridge scale confirmed",
        "evidence": {key: upstream[key] for key in [
            "eval_row_count",
            "family_count",
            "scaffold_variant_count",
            "main_answer_value_accuracy",
            "main_transform_accuracy",
            "main_pocket_writeback_rate",
            "main_contrast_group_accuracy",
            "ablation_answer_value_accuracy",
            "pocket_ablation_delta_answer_value_accuracy",
            "source_copy_shortcut_rate",
            "copy_only_shortcut_detected",
            "direct_pocket_value_marker_rate",
            "every_seed_passed",
            "deterministic_replay_passed",
        ]},
        "interpretation": [
            "Single-step transform, not source copy, survived scale.",
            "Closed-pocket ablation remained decision-critical.",
            "The next useful falsification is multi-step transfer: source plus multiple transform operations before target.",
        ],
        "not_claimed": ["GPT-like readiness", "broad assistant capability", "open-domain reasoning", "architecture superiority"],
    }


def gap_analysis() -> dict[str, Any]:
    return {
        "schema_version": "phase_140w_multi_step_gap_analysis_v1",
        "closed_gaps": [
            "pocket source to transformed target works at scale",
            "source-copy shortcut rejected at scale",
            "visible/noisy bypass rejected at scale",
            "closed-pocket ablation remains causal at scale",
        ],
        "remaining_gaps": [
            "no measured multi-step transform chain",
            "no measured intermediate state consistency",
            "no measured multi-field or multi-pocket composition",
            "no hidden-gate transfer evidence beyond minimal visible gate",
        ],
        "next_gap_to_test": "multi-step transfer bridge",
    }


def decision_matrix() -> dict[str, Any]:
    return {
        "schema_version": "phase_140w_transfer_decision_matrix_v1",
        "selected_option": "multi_step_transfer_probe",
        "recommended_next": POSITIVE_NEXT,
        "options": [
            {"option": "more_single_step_transform_scale", "recommendation": "reject_for_now", "diagnostic_value": "medium_low", "reason": "140V already scale-confirmed single-step transform"},
            {"option": "multi_step_transfer_probe", "recommendation": "select", "diagnostic_value": "high", "reason": "tests chained transform beyond source->target copy-safe mapping"},
            {"option": "multi_field_multi_pocket_binding_bridge", "recommendation": "defer", "diagnostic_value": "high_later", "reason": "composition should follow multi-step single-pocket transfer"},
            {"option": "hidden_gate_implicit_carrier_bridge", "recommendation": "defer", "diagnostic_value": "medium_later", "reason": "gate reduction can be tested after transfer complexity"},
            {"option": "integration_blocker_analysis", "recommendation": "not_selected", "diagnostic_value": "low_now", "reason": "upstream 140V evidence is complete and positive"},
        ],
    }


def requirements() -> dict[str, Any]:
    return {
        "schema_version": "phase_140w_multi_step_transfer_requirements_v1",
        "target_milestone": POSITIVE_NEXT,
        "must_test_multi_step_not_single_step": True,
        "required_task_shapes": [
            "pocket source -> intermediate A -> final target",
            "table lookup plus rule recode",
            "rule composition with held-out intermediate value",
            "contrast group where same source form has distinct multi-step routes",
        ],
        "positive_gates": {
            "main_answer_value_accuracy_min": 0.50,
            "main_multi_step_accuracy_min": 0.50,
            "main_intermediate_consistency_min": 0.50,
            "main_pocket_writeback_rate_min": 0.70,
            "ablation_answer_value_accuracy_max": 0.15,
            "pocket_ablation_delta_min": 0.30,
            "single_step_shortcut_detected": False,
            "source_copy_shortcut_detected": False,
            "deterministic_replay_passed": True,
        },
    }


def anti_shortcuts() -> dict[str, Any]:
    return {
        "schema_version": "phase_140w_anti_shortcut_requirements_v1",
        "reject": [
            "source copy as target",
            "single-step target lookup without intermediate",
            "visible target bypass",
            "noisy distractor copy",
            "closed-pocket success",
            "prefix-only success",
            "oracle/rerank/verifier/LLM judge",
            "expected/scorer metadata in helper request",
        ],
    }


def target_plan() -> dict[str, Any]:
    return {
        "schema_version": "phase_140w_target_140x_milestone_plan_v1",
        "milestone": POSITIVE_NEXT,
        "type": "executable helper-only multi-step transfer probe",
        "train_allowed": False,
        "helper_generation_allowed": True,
        "helper_backend_modification_allowed": False,
        "public_request_key_change_allowed": False,
        "source_checkpoint_mutation_allowed": False,
        "clean_negative_accepted": True,
        "required_design": {
            "natural_ish_task_text_primary": True,
            "minimal_or_implicit_gate": True,
            "pocket_source_value_present": True,
            "intermediate_value_required": True,
            "final_target_differs_from_source_and_intermediate": True,
            "single_step_shortcut_must_fail": True,
            "source_copy_shortcut_must_fail": True,
            "closed_pocket_ablation_must_fail": True,
            "deterministic_replay_required": True,
        },
        "failure_routes": {
            "single_step_shortcut_detected": "140XS_SINGLE_STEP_SHORTCUT_FAILURE_ANALYSIS",
            "multi_step_binding_failure": "140XT_MULTI_STEP_BINDING_FAILURE_ANALYSIS",
            "pocket_ablation_not_decision_critical": "140XJ_POCKET_CAUSALITY_FAILURE_ANALYSIS",
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

140V evidence:

- eval rows: `{e['eval_row_count']}`
- families: `{e['family_count']}`
- scaffold variants: `{e['scaffold_variant_count']}`
- main answer value accuracy: `{e['main_answer_value_accuracy']}`
- main transform accuracy: `{e['main_transform_accuracy']}`
- main pocket writeback rate: `{e['main_pocket_writeback_rate']}`
- main contrast group accuracy: `{e['main_contrast_group_accuracy']}`
- ablation answer value accuracy: `{e['ablation_answer_value_accuracy']}`
- source copy shortcut rate: `{e['source_copy_shortcut_rate']}`
- copy-only shortcut detected: `{e['copy_only_shortcut_detected']}`
- direct `POCKET_VALUE=` marker rate: `{e['direct_pocket_value_marker_rate']}`
- deterministic replay passed: `{e['deterministic_replay_passed']}`

Selected option: `{matrix['selected_option']}`.

This remains constrained pocket-gated helper planning: not GPT-like readiness,
not broad assistant capability, not production readiness, not public API
readiness, not deployment readiness, and not safety alignment.
"""
    write_text(out / "report.md", text)


def main() -> int:
    parser = argparse.ArgumentParser(description=MILESTONE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--upstream-140v-root", type=Path, default=DEFAULT_140V_ROOT)
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    args = parser.parse_args()

    out = resolve_target_out(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_text(out / "progress.jsonl", "")
    append_progress(out, "startup", milestone=MILESTONE, heartbeat_sec=args.heartbeat_sec)
    write_json(out / "queue.json", {"schema_version": "phase_140w_queue_v1", "milestone": MILESTONE, "status": "running"})

    config = {
        "schema_version": "phase_140w_analysis_config_v1",
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

    upstream = require_140v(resolve_repo_path(args.upstream_140v_root))
    write_json(out / "upstream_140v_manifest.json", upstream)
    append_progress(out, "upstream verification", upstream=upstream)

    ast_report = scan_ast()
    write_json(out / "ast_shortcut_scan_report.json", ast_report)
    append_progress(out, "artifact loading", ast_passed=ast_report["passed"])

    evidence = evidence_summary(upstream)
    gaps = gap_analysis()
    matrix = decision_matrix()
    reqs = requirements()
    shortcuts = anti_shortcuts()
    target = target_plan()
    write_json(out / "evidence_chain_summary.json", evidence)
    write_json(out / "multi_step_gap_analysis.json", gaps)
    write_json(out / "transfer_decision_matrix.json", matrix)
    write_json(out / "multi_step_transfer_requirements.json", reqs)
    write_json(out / "anti_shortcut_requirements.json", shortcuts)
    write_json(out / "target_140x_milestone_plan.json", target)
    append_progress(out, "decision matrix", selected=matrix["selected_option"], next=matrix["recommended_next"])

    write_json(out / "diagnostic_gap_register.json", {"schema_version": "phase_140w_diagnostic_gap_register_v1", "gaps": ["No multi-step transfer measurement exists yet; 140W only plans 140X.", "No broad assistant or architecture superiority evidence is claimed."]})
    write_json(out / "risk_register.json", {"schema_version": "phase_140w_risk_register_v1", "risks": [{"risk": "140X may collapse to single-step shortcut", "mitigation": "require single-step shortcut report"}, {"risk": "intermediate consistency may fail", "mitigation": "route to multi-step binding failure analysis"}]})

    decision = {
        "schema_version": "phase_140w_decision_v1",
        "decision": "multi_step_transfer_probe_recommended",
        "verdict": "INSTNCT_POCKET_GATED_MULTI_STEP_TRANSFER_DECISION_COMPLETE",
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
    summary = {"schema_version": "phase_140w_summary_v1", "milestone": MILESTONE, "status": "complete", "boundary": BOUNDARY_TEXT, "upstream": upstream, "selected_option": matrix["selected_option"], "recommended_next": POSITIVE_NEXT, "analysis_config_hash": stable_hash(config), **decision}
    write_json(out / "summary.json", summary)
    write_report(out, decision, evidence, matrix)
    append_progress(out, "decision", decision=decision["decision"], next=decision["next"])
    append_progress(out, "final verdict", verdict=decision["verdict"])
    write_json(out / "queue.json", {"schema_version": "phase_140w_queue_v1", "milestone": MILESTONE, "status": "complete", "decision": decision["decision"], "next": decision["next"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
