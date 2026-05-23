#!/usr/bin/env python3
"""140G artifact-only plan for minimal-marker real-task pocket bridge."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "STABLE_LOOP_PHASE_LOCK_140G_INSTNCT_POCKET_GATED_MINIMAL_MARKER_REAL_TASK_BRIDGE_PLAN"
DEFAULT_OUT = Path("target/pilot_wave/stable_loop_phase_lock_140g_instnct_pocket_gated_minimal_marker_real_task_bridge_plan/smoke")
DEFAULT_140F_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_140f_instnct_pocket_gated_noisy_marker_bridge_scale_confirm/smoke")
RUNNER_PATH = Path(__file__).resolve()
CHECKER_PATH = REPO_ROOT / "scripts/probes/run_stable_loop_phase_lock_140g_instnct_pocket_gated_minimal_marker_real_task_bridge_plan_check.py"

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
    "140G is planning-only and artifact-only after the positive 140F noisy-marker bridge "
    "scale confirm. It does not train, run new inference, call shared_raw_generation_helper.py "
    "for generation, mutate checkpoints, modify helper/backend/runtime/release/product surfaces, "
    "import old phase runners, start services, deploy, change root LICENSE, or claim GPT-like "
    "or broad assistant readiness."
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
    return {"schema_version": "phase_140g_ast_scan_v1", "passed": not failures, "failures": failures}


def require_140f(root: Path) -> dict[str, Any]:
    required = [
        "decision.json",
        "arm_comparison.json",
        "selection_report.json",
        "control_arm_report.json",
        "determinism_replay_report.json",
        "expected_output_canary_report.json",
        "generated_before_scoring_report.json",
    ]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise RuntimeError(f"missing 140F artifacts: {missing}")
    decision = read_json(root / "decision.json")
    comparison = read_json(root / "arm_comparison.json")
    selection = read_json(root / "selection_report.json")
    controls = read_json(root / "control_arm_report.json")
    replay = read_json(root / "determinism_replay_report.json")
    canary = read_json(root / "expected_output_canary_report.json")
    generated = read_json(root / "generated_before_scoring_report.json")
    if decision.get("decision") != "instnct_pocket_gated_noisy_marker_bridge_scale_confirmed":
        raise RuntimeError(f"bad 140F decision: {decision.get('decision')}")
    if decision.get("next") != "140G_INSTNCT_POCKET_GATED_MINIMAL_MARKER_REAL_TASK_BRIDGE_PLAN":
        raise RuntimeError(f"bad 140F next: {decision.get('next')}")
    gate_checks = {
        "eval_row_count": comparison.get("eval_row_count", 0) >= 2000,
        "family_count": comparison.get("family_count", 0) >= 6,
        "scaffold_variant_count": comparison.get("scaffold_variant_count", 0) >= 20,
        "main_answer_value_accuracy": comparison.get("main_answer_value_accuracy", 0.0) >= 0.95,
        "main_pocket_writeback_rate": comparison.get("main_pocket_writeback_rate", 0.0) >= 0.95,
        "ablation_answer_value_accuracy": comparison.get("ablation_answer_value_accuracy", 1.0) <= 0.05,
        "pocket_ablation_delta": comparison.get("pocket_ablation_delta_answer_value_accuracy", 0.0) >= 0.90,
        "reduced_marker_row_rate": comparison.get("reduced_marker_row_rate", 0.0) >= 0.85,
        "direct_pocket_value_marker_rate": comparison.get("direct_pocket_value_marker_rate", 1.0) <= 0.15,
        "visible_bypass_control_failed": comparison.get("visible_bypass_control_failed") is True,
        "noisy_distractor_control_failed": comparison.get("noisy_distractor_control_failed") is True,
        "every_seed_passed": comparison.get("every_seed_passed") is True,
        "deterministic_replay_passed": replay.get("deterministic_replay_passed") is True,
        "canary_passed": canary.get("passed") is True,
        "generated_before_scoring": generated.get("passed") is True,
        "controls_failed": controls.get("controls_failed") is True,
        "mutation_selected_open_pocket": selection.get("selected_candidate") == "open_pocket_all_payload_markers_noisy_bridge_scale",
        "gradient_used_false": selection.get("gradient_used") is False,
    }
    failed = [key for key, passed in gate_checks.items() if not passed]
    if failed:
        raise RuntimeError(f"140F scale gates failed: {failed}")
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
        "main_contrast_group_accuracy": comparison.get("main_contrast_group_accuracy"),
        "ablation_answer_value_accuracy": comparison.get("ablation_answer_value_accuracy"),
        "pocket_ablation_delta_answer_value_accuracy": comparison.get("pocket_ablation_delta_answer_value_accuracy"),
        "reduced_marker_row_rate": comparison.get("reduced_marker_row_rate"),
        "direct_pocket_value_marker_rate": comparison.get("direct_pocket_value_marker_rate"),
        "visible_bypass_control_failed": comparison.get("visible_bypass_control_failed"),
        "noisy_distractor_control_failed": comparison.get("noisy_distractor_control_failed"),
        "every_seed_passed": comparison.get("every_seed_passed"),
        "deterministic_replay_passed": replay.get("deterministic_replay_passed"),
        "selected_candidate": selection.get("selected_candidate"),
        "fitness_margin": selection.get("fitness_margin"),
        "gradient_used": selection.get("gradient_used"),
        "gate_checks": gate_checks,
    }


def build_evidence_summary(upstream: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "phase_140g_scale_confirm_evidence_summary_v1",
        "source_milestone": "140F_INSTNCT_POCKET_GATED_NOISY_MARKER_BRIDGE_SCALE_CONFIRM",
        "scale_confirmed": True,
        "evidence": {
            "eval_row_count": upstream["eval_row_count"],
            "family_count": upstream["family_count"],
            "scaffold_variant_count": upstream["scaffold_variant_count"],
            "main_answer_value_accuracy": upstream["main_answer_value_accuracy"],
            "main_pocket_writeback_rate": upstream["main_pocket_writeback_rate"],
            "ablation_answer_value_accuracy": upstream["ablation_answer_value_accuracy"],
            "pocket_ablation_delta_answer_value_accuracy": upstream["pocket_ablation_delta_answer_value_accuracy"],
            "reduced_marker_row_rate": upstream["reduced_marker_row_rate"],
            "direct_pocket_value_marker_rate": upstream["direct_pocket_value_marker_rate"],
            "every_seed_passed": upstream["every_seed_passed"],
            "deterministic_replay_passed": upstream["deterministic_replay_passed"],
            "selected_candidate": upstream["selected_candidate"],
            "fitness_margin": upstream["fitness_margin"],
            "gradient_used": upstream["gradient_used"],
        },
        "interpretation": [
            "Noisy/reduced-marker bridge behavior is stable at 140F scale.",
            "Another same-shape scale pass has lower diagnostic value than removing more scaffolding.",
            "The next useful falsification is minimal-marker or implicit-gate real-task-style bridge planning.",
        ],
    }


def build_gap_analysis(upstream: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "phase_140g_minimal_marker_gap_analysis_v1",
        "current_direct_pocket_value_marker_rate": upstream["direct_pocket_value_marker_rate"],
        "current_reduced_marker_row_rate": upstream["reduced_marker_row_rate"],
        "remaining_scaffolding": [
            "GATE:POCKET_OPEN remains explicit.",
            "POCKET_BIND= and POCKET_TABLE_ROW= markers remain explicit in most rows.",
            "Prompt still tells the model to return the open-pocket value.",
            "The helper manifest still performs deterministic marker selection.",
        ],
        "next_unknowns": [
            "Can the bridge survive without visible POCKET_* payload tokens in most rows?",
            "Can an implicit or natural-language gate substitute for GATE:POCKET_OPEN while preserving closed-pocket ablation?",
            "Can real-task-like carrier text preserve pocket causality without becoming visible-value bypass?",
            "Does mutation selection still prefer the correct open-pocket config when markers are minimal?",
        ],
        "not_claimed": [
            "measured hidden-state mechanism",
            "general value grounding",
            "GPT-like readiness",
            "open-domain assistant readiness",
            "architecture superiority",
        ],
    }


def build_real_task_requirements() -> dict[str, Any]:
    return {
        "schema_version": "phase_140g_real_task_bridge_requirements_v1",
        "required_prompt_properties": [
            "natural-ish task text is the primary carrier",
            "explicit POCKET_VALUE rows are forbidden except controls",
            "explicit POCKET_BIND and POCKET_TABLE_ROW rows are sharply reduced",
            "implicit gate or minimal marker replaces most GATE:POCKET_OPEN usage",
            "visible wrong value remains present",
            "multiple distractor values remain present",
            "same-family contrast groups keep distinct expected values",
            "closed-pocket ablation must still fail",
            "visible bypass and noisy distractor controls must fail",
            "mutation selection must still choose the correct open-pocket config",
        ],
        "positive_gate_recommendations": {
            "main_answer_value_accuracy_min": 0.70,
            "main_pocket_writeback_rate_min": 0.80,
            "ablation_answer_value_accuracy_max": 0.15,
            "pocket_ablation_delta_min": 0.45,
            "explicit_pocket_token_row_rate_max": 0.20,
            "direct_pocket_value_marker_rate_max": 0.02,
            "implicit_or_minimal_gate_row_rate_min": 0.70,
            "visible_bypass_violation_rate_max": 0.0,
            "noisy_distractor_violation_rate_max": 0.0,
            "deterministic_replay_passed": True,
        },
    }


def build_marker_policy() -> dict[str, Any]:
    return {
        "schema_version": "phase_140g_marker_reduction_policy_v1",
        "disallow_as_main_path": ["POCKET_VALUE="],
        "reduce_as_main_path": ["POCKET_BIND=", "POCKET_TABLE_ROW=", "GATE:POCKET_OPEN"],
        "allow_as_controls_only": ["POCKET_VALUE=", "GATE:NEVER_OPEN", "GATE:POCKET_OPEN with explicit payload"],
        "new_allowed_minimal_markers": [
            "short side-note cue without POCKET_* prefix",
            "natural-language gate phrase",
            "structured but non-POCKET task footnote",
        ],
        "required_audits": [
            "explicit_pocket_token_row_rate",
            "direct_pocket_value_marker_rate",
            "implicit_or_minimal_gate_row_rate",
            "visible_value_bypass_rate",
            "distractor_copy_rate",
        ],
    }


def build_implicit_gate_policy() -> dict[str, Any]:
    return {
        "schema_version": "phase_140g_implicit_gate_policy_v1",
        "goal": "Replace most literal GATE:POCKET_OPEN markers with minimal or natural-language gate cues while keeping a closed-pocket ablation arm.",
        "main_arm": {
            "gate_style": "minimal_or_natural_language_gate",
            "explicit_gate_token_rate_max": 0.30,
            "pocket_causality_required": True,
        },
        "ablation_arm": {
            "gate_style": "gate phrase removed or semantically negated",
            "expected_result": "value path fails or emits closed-pocket fallback",
            "post_generation_repair_allowed": False,
        },
        "invalid_positive_conditions": [
            "positive from visible value copy",
            "positive from distractor copy",
            "positive with high explicit POCKET_* token rate",
            "positive without closed-pocket ablation delta",
            "positive without deterministic replay",
        ],
    }


def build_target_140h_plan() -> dict[str, Any]:
    return {
        "schema_version": "phase_140g_target_140h_milestone_plan_v1",
        "milestone": "140H_INSTNCT_POCKET_GATED_MINIMAL_MARKER_REAL_TASK_BRIDGE_PROBE",
        "type": "targeted executable bridge probe",
        "train_allowed": False,
        "helper_generation_allowed": True,
        "helper_backend_modification_allowed": False,
        "public_api_change_allowed": False,
        "source_checkpoint_mutation_allowed": False,
        "clean_negative_accepted": True,
        "required_design": {
            "minimal_marker_or_implicit_gate": True,
            "natural_ish_task_text_primary": True,
            "explicit_pocket_value_marker_forbidden_in_main_eval": True,
            "explicit_pocket_tokens_strongly_reduced": True,
            "visible_wrong_value_present": True,
            "noisy_distractors_present": True,
            "closed_pocket_ablation_must_fail": True,
            "mutation_selection_must_prefer_correct_open_pocket_config": True,
        },
        "positive_gates": {
            "main_answer_value_accuracy_min": 0.70,
            "main_pocket_writeback_rate_min": 0.80,
            "ablation_answer_value_accuracy_max": 0.15,
            "pocket_ablation_delta_min": 0.45,
            "explicit_pocket_token_row_rate_max": 0.20,
            "direct_pocket_value_marker_rate_max": 0.02,
            "implicit_or_minimal_gate_row_rate_min": 0.70,
            "visible_bypass_control_must_fail": True,
            "noisy_distractor_control_must_fail": True,
            "deterministic_replay_passed": True,
        },
        "infrastructure_gates": {
            "helper_only_final_eval": True,
            "expected_output_canary_passed": True,
            "ast_shortcut_scan_passed": True,
            "generated_text_before_scoring": True,
            "no_expected_or_scorer_metadata_in_helper_requests": True,
            "controls_failed": True,
            "deterministic_replay_passed": True,
        },
        "required_artifacts": [
            "queue.json",
            "progress.jsonl",
            "upstream_140g_manifest.json",
            "minimal_marker_eval_manifest.json",
            "real_task_bridge_prompt_manifest.json",
            "implicit_gate_policy.json",
            "mutation_candidate_results.jsonl",
            "raw_generation_trace.jsonl",
            "raw_generation_results.jsonl",
            "pocket_trace.jsonl",
            "pocket_ablation_results.jsonl",
            "scoring_results.jsonl",
            "contrast_group_results.jsonl",
            "control_results.jsonl",
            "control_arm_report.json",
            "explicit_marker_audit.json",
            "visible_bypass_control_report.json",
            "noisy_distractor_control_report.json",
            "determinism_replay_report.json",
            "decision.json",
            "summary.json",
            "report.md",
        ],
    }


def build_failure_routes() -> dict[str, Any]:
    return {
        "schema_version": "phase_140g_target_140h_failure_routes_v1",
        "routes": {
            "minimal_marker_dependency_too_strong": "140I_MINIMAL_MARKER_DEPENDENCY_ANALYSIS",
            "implicit_gate_not_decision_critical": "140J_IMPLICIT_GATE_CAUSALITY_ANALYSIS",
            "real_task_text_breaks_value_binding": "140K_REAL_TASK_TEXT_VALUE_BINDING_ANALYSIS",
            "visible_value_bypass_returns": "140L_VISIBLE_VALUE_BYPASS_REGRESSION_ANALYSIS",
            "noisy_distractor_copy_returns": "140M_NOISY_DISTRACTOR_COPY_REGRESSION_ANALYSIS",
            "mutation_search_fails_to_select_open_pocket": "140E_MUTATION_SELECTION_FAILURE_ANALYSIS",
            "helper_integrity_failure": "135G_RAW_HELPER_ORACLE_SHORTCUT_REMOVAL",
        },
    }


def write_report(out: Path, decision: dict[str, Any], evidence: dict[str, Any], target_plan: dict[str, Any]) -> None:
    e = evidence["evidence"]
    text = f"""# {MILESTONE}

Decision: `{decision['decision']}`

Verdict: `{decision['verdict']}`

Next: `{decision['next']}`

Boundary: {BOUNDARY_TEXT}

140F evidence:

- eval rows: `{e['eval_row_count']}`
- families: `{e['family_count']}`
- scaffold variants: `{e['scaffold_variant_count']}`
- main answer value accuracy: `{e['main_answer_value_accuracy']}`
- main pocket writeback rate: `{e['main_pocket_writeback_rate']}`
- ablation answer value accuracy: `{e['ablation_answer_value_accuracy']}`
- ablation delta: `{e['pocket_ablation_delta_answer_value_accuracy']}`
- reduced marker row rate: `{e['reduced_marker_row_rate']}`
- direct `POCKET_VALUE=` marker rate: `{e['direct_pocket_value_marker_rate']}`
- every seed passed: `{e['every_seed_passed']}`
- deterministic replay passed: `{e['deterministic_replay_passed']}`

Target milestone: `{target_plan['milestone']}`

140G does not create a new capability proof. It plans the next executable probe:
minimal-marker or implicit-gate real-task-style bridge prompts, with closed-pocket
ablation, visible bypass controls, noisy distractor controls, deterministic replay,
and strict helper-only evidence.

This remains constrained pocket-gated helper evidence, not GPT-like readiness,
not broad assistant capability, not production readiness, not public API readiness,
not deployment readiness, and not safety alignment.
"""
    write_text(out / "report.md", text)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run 140G minimal-marker real-task bridge plan")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--upstream-140f-root", type=Path, default=DEFAULT_140F_ROOT)
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    args = parser.parse_args()

    out = resolve_target_out(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_text(out / "progress.jsonl", "")
    append_progress(out, "startup", milestone=MILESTONE, heartbeat_sec=args.heartbeat_sec)
    write_json(out / "queue.json", {"schema_version": "phase_140g_queue_v1", "milestone": MILESTONE, "status": "running"})

    config = {
        "schema_version": "phase_140g_analysis_config_v1",
        "milestone": MILESTONE,
        "boundary": BOUNDARY_TEXT,
        "artifact_only": True,
        "planning_only": True,
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

    upstream = require_140f(resolve_repo_path(args.upstream_140f_root))
    write_json(out / "upstream_140f_manifest.json", upstream)
    append_progress(out, "upstream verification", upstream=upstream)

    ast_report = scan_ast()
    write_json(out / "ast_shortcut_scan_report.json", ast_report)
    append_progress(out, "artifact loading", config_hash=stable_hash(config), ast_passed=ast_report["passed"])

    evidence = build_evidence_summary(upstream)
    write_json(out / "scale_confirm_evidence_summary.json", evidence)
    append_progress(out, "scale confirm evidence summary", row_count=upstream["eval_row_count"])

    gap = build_gap_analysis(upstream)
    write_json(out / "minimal_marker_gap_analysis.json", gap)
    append_progress(out, "minimal marker gap analysis", remaining_scaffolding=len(gap["remaining_scaffolding"]))

    requirements = build_real_task_requirements()
    marker_policy = build_marker_policy()
    implicit_gate = build_implicit_gate_policy()
    target_plan = build_target_140h_plan()
    failure_routes = build_failure_routes()
    write_json(out / "real_task_bridge_requirements.json", requirements)
    write_json(out / "marker_reduction_policy.json", marker_policy)
    write_json(out / "implicit_gate_policy.json", implicit_gate)
    write_json(out / "target_140h_milestone_plan.json", target_plan)
    write_json(out / "target_140h_failure_routes.json", failure_routes)
    append_progress(out, "target 140H plan writing", next=target_plan["milestone"])

    gaps = {
        "schema_version": "phase_140g_diagnostic_gap_register_v1",
        "gaps": [
            "No measured hidden-state, logit, grower, scout, or topological internal mechanism artifact exists.",
            "140F evidence is helper-backend mechanism evidence, not architecture-superiority evidence.",
            "Minimal-marker implicit gate behavior is not yet measured; 140G only plans 140H.",
        ],
    }
    risks = {
        "schema_version": "phase_140g_risk_register_v1",
        "risks": [
            {"risk": "removing explicit gate tokens may collapse pocket causality", "mitigation": "require closed-pocket ablation and implicit-gate causality route"},
            {"risk": "natural task text may reintroduce visible-value bypass", "mitigation": "keep visible wrong values and visible bypass controls"},
            {"risk": "minimal markers may be too weak for current deterministic helper manifest path", "mitigation": "accept clean negative and route to marker dependency analysis"},
            {"risk": "longer 140H run loses partial results on crash", "mitigation": "require heartbeat progress and partial artifact refreshes"},
        ],
    }
    write_json(out / "diagnostic_gap_register.json", gaps)
    write_json(out / "risk_register.json", risks)

    decision = {
        "schema_version": "phase_140g_decision_v1",
        "decision": "minimal_marker_real_task_bridge_plan_complete",
        "verdict": "INSTNCT_POCKET_GATED_MINIMAL_MARKER_REAL_TASK_BRIDGE_PLAN_COMPLETE",
        "next": "140H_INSTNCT_POCKET_GATED_MINIMAL_MARKER_REAL_TASK_BRIDGE_PROBE",
        "boundary": BOUNDARY_TEXT,
        "planning_only": True,
        "artifact_only": True,
        "clean_negative_valid": True,
        "architecture_superiority_claimed": False,
        "value_grounding_claimed": False,
        "pocket_mechanism_claimed": False,
        **FALSE_FLAGS,
    }
    write_json(out / "decision.json", decision)
    summary = {
        "schema_version": "phase_140g_summary_v1",
        "milestone": MILESTONE,
        "status": "complete",
        "boundary": BOUNDARY_TEXT,
        "upstream": upstream,
        "target_140h_milestone": target_plan["milestone"],
        "analysis_config_hash": stable_hash(config),
        **decision,
    }
    write_json(out / "summary.json", summary)
    write_report(out, decision, evidence, target_plan)
    append_progress(out, "decision", decision=decision["decision"], next=decision["next"])
    append_progress(out, "final verdict", verdict=decision["verdict"])
    write_json(out / "queue.json", {"schema_version": "phase_140g_queue_v1", "milestone": MILESTONE, "status": "complete", "decision": decision["decision"], "next": decision["next"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
