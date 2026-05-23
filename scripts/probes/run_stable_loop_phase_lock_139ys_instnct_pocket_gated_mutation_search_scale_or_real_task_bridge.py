#!/usr/bin/env python3
"""139YS artifact-only decision plan for pocket-gated scale vs bridge."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "STABLE_LOOP_PHASE_LOCK_139YS_INSTNCT_POCKET_GATED_MUTATION_SEARCH_SCALE_OR_REAL_TASK_BRIDGE"
DEFAULT_OUT = Path("target/pilot_wave/stable_loop_phase_lock_139ys_instnct_pocket_gated_mutation_search_scale_or_real_task_bridge/smoke")
DEFAULT_139YR_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_139yr_instnct_pocket_gated_mutation_search_confirm/smoke")
DEFAULT_139YQ_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_139yq_instnct_pocket_gated_value_grounding_scale_confirm/smoke")
DEFAULT_138YO_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_138yo_instnct_mutation_value_grounding_comparison_probe/smoke")
DEFAULT_138YK_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_138yk_family_default_suppressed_contrastive_repair_probe/smoke")
RUNNER_PATH = Path(__file__).resolve()
CHECKER_PATH = REPO_ROOT / "scripts/probes/run_stable_loop_phase_lock_139ys_instnct_pocket_gated_mutation_search_scale_or_real_task_bridge_check.py"

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
    "139YS is planning-only artifact analysis after 139YR. It does not train, "
    "run new inference, call shared_raw_generation_helper.py for generation, "
    "mutate checkpoints, modify helper/backend/runtime/release/product surfaces, "
    "import old phase runners, start services, deploy, change root LICENSE, or "
    "claim GPT-like or broad assistant readiness."
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
            if isinstance(node, ast.Import) and any(alias.name == "importlib" or alias.name.startswith("importlib.") for alias in node.names):
                failures.append(f"helper_dynamic_import_risk:{rel(path)}")
    return {"schema_version": "phase_139ys_ast_scan_v1", "passed": not failures, "failures": failures}


def require_139yr(root: Path) -> dict[str, Any]:
    required = ["decision.json", "selection_report.json"]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise RuntimeError(f"missing 139YR artifacts: {missing}")
    decision = read_json(root / "decision.json")
    selection = read_json(root / "selection_report.json")
    if decision.get("decision") != "instnct_pocket_gated_mutation_search_confirmed":
        raise RuntimeError(f"bad 139YR decision: {decision.get('decision')}")
    if decision.get("verdict") != "INSTNCT_POCKET_GATED_MUTATION_SEARCH_CONFIRMED":
        raise RuntimeError(f"bad 139YR verdict: {decision.get('verdict')}")
    if decision.get("next") != "139YS_INSTNCT_POCKET_GATED_MUTATION_SEARCH_SCALE_OR_REAL_TASK_BRIDGE":
        raise RuntimeError(f"bad 139YR next: {decision.get('next')}")
    if decision.get("gradient_used") is not False or selection.get("gradient_used") is not False:
        raise RuntimeError("139YR unexpectedly used gradient")
    if selection.get("selected_candidate") != "open_pocket_all_payload_markers":
        raise RuntimeError(f"bad 139YR selected candidate: {selection.get('selected_candidate')}")
    if selection.get("fitness_margin", 0.0) < 0.40:
        raise RuntimeError(f"139YR fitness margin too low: {selection.get('fitness_margin')}")
    return {
        "root": rel(root),
        "decision": decision.get("decision"),
        "verdict": decision.get("verdict"),
        "next": decision.get("next"),
        "gradient_used": decision.get("gradient_used"),
        "selected": selection.get("selected_candidate"),
        "selected_accuracy": selection.get("selected_accuracy"),
        "selected_pocket_writeback_rate": selection.get("selected_pocket_writeback_rate"),
        "selected_fitness": selection.get("selected_fitness"),
        "runner_up_candidate": selection.get("runner_up_candidate"),
        "runner_up_fitness": selection.get("runner_up_fitness"),
        "fitness_margin": selection.get("fitness_margin"),
    }


def require_139yq(root: Path) -> dict[str, Any]:
    required = ["decision.json", "arm_comparison.json", "determinism_replay_report.json"]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise RuntimeError(f"missing 139YQ artifacts: {missing}")
    decision = read_json(root / "decision.json")
    comparison = read_json(root / "arm_comparison.json")
    replay = read_json(root / "determinism_replay_report.json")
    if decision.get("decision") != "instnct_pocket_gated_value_grounding_scale_confirmed":
        raise RuntimeError(f"bad 139YQ decision: {decision.get('decision')}")
    if comparison.get("main_answer_value_accuracy") != 1.0:
        raise RuntimeError("139YQ main accuracy is not 1.0")
    if comparison.get("main_pocket_writeback_rate") != 1.0:
        raise RuntimeError("139YQ pocket writeback is not 1.0")
    if comparison.get("main_phase_transport_success_rate") != 1.0:
        raise RuntimeError("139YQ phase transport is not 1.0")
    if comparison.get("ablation_answer_value_accuracy") != 0.0:
        raise RuntimeError("139YQ ablation accuracy is not 0.0")
    if comparison.get("pocket_ablation_delta_answer_value_accuracy") != 1.0:
        raise RuntimeError("139YQ ablation delta is not 1.0")
    if replay.get("deterministic_replay_passed") is not True:
        raise RuntimeError("139YQ determinism failed")
    return {
        "root": rel(root),
        "decision": decision.get("decision"),
        "verdict": decision.get("verdict"),
        "next": decision.get("next"),
        "eval_row_count": comparison.get("eval_row_count"),
        "main_answer_value_accuracy": comparison.get("main_answer_value_accuracy"),
        "main_pocket_writeback_rate": comparison.get("main_pocket_writeback_rate"),
        "main_phase_transport_success_rate": comparison.get("main_phase_transport_success_rate"),
        "ablation_answer_value_accuracy": comparison.get("ablation_answer_value_accuracy"),
        "ablation_pocket_writeback_rate": comparison.get("ablation_pocket_writeback_rate"),
        "pocket_ablation_delta_answer_value_accuracy": comparison.get("pocket_ablation_delta_answer_value_accuracy"),
        "deterministic_replay_passed": replay.get("deterministic_replay_passed"),
    }


def require_138yo(root: Path) -> dict[str, Any]:
    required = ["decision.json", "aggregate_metrics.json", "pocket_ablation_report.json", "determinism_replay_report.json"]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise RuntimeError(f"missing 138YO artifacts: {missing}")
    decision = read_json(root / "decision.json")
    aggregate = read_json(root / "aggregate_metrics.json")
    pocket = read_json(root / "pocket_ablation_report.json")
    replay = read_json(root / "determinism_replay_report.json")
    if decision.get("decision") != "instnct_adapter_prompt_bound_value_grounding_improves":
        raise RuntimeError(f"bad 138YO decision: {decision.get('decision')}")
    if aggregate.get("instnct_answer_value_accuracy", 0.0) <= aggregate.get("byte_gru_answer_value_accuracy", 1.0):
        raise RuntimeError("138YO adapter did not beat byte-GRU")
    if aggregate.get("instnct_pocket_writeback_rate") != 0.0:
        raise RuntimeError("138YO pocket writeback profile changed")
    if pocket.get("answer_value_accuracy_delta") != 0.0:
        raise RuntimeError("138YO pocket ablation profile changed")
    if replay.get("deterministic_replay_passed") is not True:
        raise RuntimeError("138YO determinism failed")
    return {
        "root": rel(root),
        "decision": decision.get("decision"),
        "verdict": decision.get("verdict"),
        "next": decision.get("next"),
        "byte_gru_answer_value_accuracy": aggregate.get("byte_gru_answer_value_accuracy"),
        "instnct_answer_value_accuracy": aggregate.get("instnct_answer_value_accuracy"),
        "instnct_intra_family_contrastive_accuracy": aggregate.get("instnct_intra_family_contrastive_accuracy"),
        "instnct_minus_byte_answer_value_accuracy": aggregate.get("instnct_minus_byte_answer_value_accuracy"),
        "instnct_pocket_writeback_rate": aggregate.get("instnct_pocket_writeback_rate"),
        "pocket_ablation_delta_answer_value_accuracy": pocket.get("answer_value_accuracy_delta"),
        "pocket_writeback_decision_critical": pocket.get("pocket_writeback_decision_critical"),
        "deterministic_replay_passed": replay.get("deterministic_replay_passed"),
    }


def require_138yk(root: Path) -> dict[str, Any]:
    required = ["decision.json", "aggregate_metrics.json"]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise RuntimeError(f"missing 138YK artifacts: {missing}")
    decision = read_json(root / "decision.json")
    aggregate = read_json(root / "aggregate_metrics.json")
    if aggregate.get("answer_value_accuracy") != 0.0:
        raise RuntimeError("138YK byte-GRU value accuracy profile changed")
    if aggregate.get("exact_answer_accuracy") != 0.0:
        raise RuntimeError("138YK exact accuracy profile changed")
    if aggregate.get("intra_family_contrastive_accuracy") != 0.0:
        raise RuntimeError("138YK contrastive profile changed")
    if aggregate.get("family_default_shortcut_detected") is not True:
        raise RuntimeError("138YK family-default shortcut no longer detected")
    if decision.get("determinism_replay_passed") is not True:
        raise RuntimeError("138YK determinism failed")
    if aggregate.get("stale_chat_fragment_rate") != 0.0 or aggregate.get("train_namespace_leak_rate") != 0.0:
        raise RuntimeError("138YK integrity profile changed")
    return {
        "root": rel(root),
        "decision": decision.get("decision"),
        "verdict": decision.get("verdict"),
        "next": decision.get("next"),
        "answer_value_accuracy": aggregate.get("answer_value_accuracy"),
        "exact_answer_accuracy": aggregate.get("exact_answer_accuracy"),
        "intra_family_contrastive_accuracy": aggregate.get("intra_family_contrastive_accuracy"),
        "family_default_attractor_rate": aggregate.get("family_default_attractor_rate"),
        "family_default_shortcut_detected": aggregate.get("family_default_shortcut_detected"),
        "high_frequency_train_value_replay_detected": aggregate.get("high_frequency_train_value_replay_detected"),
        "intra_family_mode_collapse_rate": aggregate.get("intra_family_mode_collapse_rate"),
        "determinism_replay_passed": decision.get("determinism_replay_passed"),
        "stale_chat_fragment_rate": aggregate.get("stale_chat_fragment_rate"),
        "train_namespace_leak_rate": aggregate.get("train_namespace_leak_rate"),
    }


def build_evidence_chain(upstreams: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "phase_139ys_evidence_chain_summary_v1",
        "chain_complete": True,
        "gradient_used": False,
        "stages": [
            {
                "milestone": "138YK_FAMILY_DEFAULT_SUPPRESSED_CONTRASTIVE_REPAIR_PROBE",
                "finding": "byte-GRU repair/probe did not improve prompt-specific value binding",
                "answer_value_accuracy": upstreams["138yk"]["answer_value_accuracy"],
                "family_default_shortcut_detected": upstreams["138yk"]["family_default_shortcut_detected"],
            },
            {
                "milestone": "138YO_INSTNCT_MUTATION_VALUE_GROUNDING_COMPARISON_PROBE",
                "finding": "INSTNCT adapter improved over byte-GRU but did not use pocket writeback",
                "byte_gru_answer_value_accuracy": upstreams["138yo"]["byte_gru_answer_value_accuracy"],
                "instnct_answer_value_accuracy": upstreams["138yo"]["instnct_answer_value_accuracy"],
                "instnct_pocket_writeback_rate": upstreams["138yo"]["instnct_pocket_writeback_rate"],
                "pocket_ablation_delta": upstreams["138yo"]["pocket_ablation_delta_answer_value_accuracy"],
            },
            {
                "milestone": "139YQ_INSTNCT_POCKET_GATED_VALUE_GROUNDING_SCALE_CONFIRM",
                "finding": "strict pocket-gated proof scaled with decision-critical closed-pocket ablation",
                "main_answer_value_accuracy": upstreams["139yq"]["main_answer_value_accuracy"],
                "main_pocket_writeback_rate": upstreams["139yq"]["main_pocket_writeback_rate"],
                "ablation_answer_value_accuracy": upstreams["139yq"]["ablation_answer_value_accuracy"],
                "pocket_ablation_delta": upstreams["139yq"]["pocket_ablation_delta_answer_value_accuracy"],
            },
            {
                "milestone": "139YR_INSTNCT_POCKET_GATED_MUTATION_SEARCH_CONFIRM",
                "finding": "non-gradient mutation search selected the correct open-pocket config",
                "selected": upstreams["139yr"]["selected"],
                "fitness_margin": upstreams["139yr"]["fitness_margin"],
                "gradient_used": upstreams["139yr"]["gradient_used"],
            },
        ],
        "interpretation": [
            "The current marker-bound pocket mechanism is stable enough for a decision gate.",
            "Repeating the same marker-bound proof has diminishing diagnostic value.",
            "The next falsification should reduce marker scaffolding while preserving pocket causality gates.",
        ],
    }


def build_decision_matrix() -> dict[str, Any]:
    return {
        "schema_version": "phase_139ys_scale_vs_bridge_decision_matrix_v1",
        "options": [
            {
                "option": "scale_current_marker_bound_proof",
                "selected": False,
                "pros": [
                    "Confirms stability over more rows, families, seeds, and pocket variants.",
                    "Lower execution risk because 139YQ and 139YR already passed.",
                ],
                "cons": [
                    "May overfit confidence to explicit POCKET_VALUE marker scaffolding.",
                    "Does not test whether value binding survives more task-like prompts.",
                ],
                "diagnostic_value": "medium",
                "recommendation": "defer unless 140A fails from insufficient statistical power",
            },
            {
                "option": "real_task_bridge_reduced_marker_probe",
                "selected": True,
                "pros": [
                    "Tests whether pocket-gated value binding survives reduced explicit markers.",
                    "Directly targets the next uncertainty after mutation search confirmation.",
                    "Retains closed-pocket ablation, visible bypass, controls, and deterministic replay gates.",
                ],
                "cons": [
                    "Higher clean-negative risk.",
                    "May require a new analysis route if noisy prompts break value binding.",
                ],
                "diagnostic_value": "high",
                "recommendation": "select as 140A",
            },
        ],
        "selected_option": "real_task_bridge_reduced_marker_probe",
        "decision": "real_task_bridge_recommended",
        "next": "140A_INSTNCT_POCKET_GATED_NOISY_MARKER_BRIDGE_PROBE",
    }


def build_target_140a_plan() -> dict[str, Any]:
    return {
        "schema_version": "phase_139ys_target_140a_milestone_plan_v1",
        "milestone": "140A_INSTNCT_POCKET_GATED_NOISY_MARKER_BRIDGE_PROBE",
        "type": "targeted bridge probe",
        "train_allowed": False,
        "mutation_search_allowed": True,
        "helper_generation_allowed_in_140a": True,
        "helper_backend_modification_allowed": False,
        "public_api_change_allowed": False,
        "source_checkpoint_mutation_allowed": False,
        "clean_negative_accepted": True,
        "bridge_design": {
            "explicit_pocket_value_markers_reduced": True,
            "noisy_prompt_distractors_added": True,
            "value_hidden_behind_natural_task_text": True,
            "pocket_gate_still_required": True,
            "visible_value_bypass_forbidden": True,
            "closed_pocket_ablation_must_fail": True,
            "mutation_selection_must_prefer_correct_open_pocket_config": True,
        },
        "required_controls": [
            "VISIBLE_VALUE_BYPASS_CONTROL",
            "NOISY_DISTRACTOR_CONTROL",
            "CLOSED_POCKET_ABLATION_CONTROL",
            "STATIC_OUTPUT_CONTROL",
            "COPY_PROMPT_CONTROL",
            "TRAIN_NAMESPACE_REPLAY_CONTROL",
            "PREFIX_ONLY_CONTROL",
        ],
        "infrastructure_gates": {
            "helper_only_final_eval": True,
            "expected_output_canary_passed": True,
            "ast_shortcut_scan_passed": True,
            "leakage_rejected": True,
            "controls_failed": True,
            "deterministic_replay_passed": True,
            "generated_text_before_scoring": True,
            "no_expected_or_scorer_metadata_in_helper_requests": True,
        },
        "positive_gates": {
            "main_answer_value_accuracy_min": 0.80,
            "pocket_writeback_rate_min": 0.90,
            "ablation_answer_value_accuracy_max": 0.10,
            "pocket_ablation_delta_min": 0.50,
            "visible_bypass_control_must_fail": True,
            "noisy_distractor_control_must_fail": True,
            "deterministic_replay_passed": True,
        },
        "clean_negative_routes": {
            "marker_dependency_too_strong": "140B_MARKER_DEPENDENCY_ANALYSIS",
            "pocket_ablation_not_decision_critical": "140C_POCKET_CAUSALITY_FAILURE_ANALYSIS",
            "noisy_prompt_breaks_value_binding": "140D_NOISY_PROMPT_VALUE_BINDING_ANALYSIS",
            "mutation_search_fails_to_select_open_pocket": "140E_MUTATION_SELECTION_FAILURE_ANALYSIS",
            "helper_integrity_failure": "135G_RAW_HELPER_ORACLE_SHORTCUT_REMOVAL",
        },
        "capability_boundary": {
            "pocket_mechanism_evidence_allowed": True,
            "value_grounding_claim_allowed": False,
            "architecture_superiority_claim_allowed": False,
            **FALSE_FLAGS,
        },
        "required_artifacts": [
            "queue.json",
            "progress.jsonl",
            "upstream_139ys_manifest.json",
            "noisy_bridge_eval_manifest.json",
            "bridge_prompt_scaffold_manifest.json",
            "mutation_candidate_results.jsonl",
            "raw_generation_trace.jsonl",
            "raw_generation_results.jsonl",
            "pocket_trace.jsonl",
            "pocket_ablation_results.jsonl",
            "visible_bypass_control_report.json",
            "noisy_distractor_control_report.json",
            "arm_comparison.json",
            "determinism_replay_report.json",
            "decision.json",
            "summary.json",
            "report.md",
        ],
    }


def write_report(out: Path, decision: dict[str, Any], chain: dict[str, Any], recommendation: dict[str, Any], target_plan: dict[str, Any]) -> None:
    text = f"""# {MILESTONE}

Decision: `{decision['decision']}`

Next: `{decision['next']}`

Boundary: {BOUNDARY_TEXT}

This is planning-only. It reads existing artifacts and writes a bridge decision. It does not run training, inference, helper generation, checkpoint mutation, service deployment, runtime changes, release changes, product changes, or public API changes.

Evidence chain:

- 138YK byte-GRU value accuracy: `{chain['stages'][0]['answer_value_accuracy']}`
- 138YO INSTNCT adapter value accuracy: `{chain['stages'][1]['instnct_answer_value_accuracy']}`
- 138YO pocket writeback rate: `{chain['stages'][1]['instnct_pocket_writeback_rate']}`
- 139YQ main/ablation value accuracy: `{chain['stages'][2]['main_answer_value_accuracy']}` / `{chain['stages'][2]['ablation_answer_value_accuracy']}`
- 139YR selected mutation config: `{chain['stages'][3]['selected']}`
- 139YR fitness margin: `{chain['stages'][3]['fitness_margin']}`

Recommendation: `{recommendation['recommended_next']}`

Rationale: the marker-bound proof and non-gradient mutation selection are stable enough. The next useful falsification is whether pocket-gated value binding survives weaker explicit markers, noisy distractors, and more task-like prompt text while closed-pocket ablation still fails.

Target milestone: `{target_plan['milestone']}`

Capability boundary: not GPT-like readiness, not broad assistant capability, not production readiness, not public API readiness, not deployment readiness, and not safety alignment.
"""
    write_text(out / "report.md", text)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run 139YS pocket-gated scale-or-bridge decision plan")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--upstream-139yr-root", type=Path, default=DEFAULT_139YR_ROOT)
    parser.add_argument("--upstream-139yq-root", type=Path, default=DEFAULT_139YQ_ROOT)
    parser.add_argument("--upstream-138yo-root", type=Path, default=DEFAULT_138YO_ROOT)
    parser.add_argument("--upstream-138yk-root", type=Path, default=DEFAULT_138YK_ROOT)
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    args = parser.parse_args()

    out = resolve_target_out(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_text(out / "progress.jsonl", "")
    append_progress(out, "startup", milestone=MILESTONE, heartbeat_sec=args.heartbeat_sec)
    write_json(out / "queue.json", {"schema_version": "phase_139ys_queue_v1", "milestone": MILESTONE, "status": "running"})

    config = {
        "schema_version": "phase_139ys_analysis_config_v1",
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

    try:
        upstreams = {
            "139yr": require_139yr(resolve_repo_path(args.upstream_139yr_root)),
            "139yq": require_139yq(resolve_repo_path(args.upstream_139yq_root)),
            "138yo": require_138yo(resolve_repo_path(args.upstream_138yo_root)),
            "138yk": require_138yk(resolve_repo_path(args.upstream_138yk_root)),
        }
        upstream_complete = True
    except RuntimeError as exc:
        upstreams = {}
        upstream_complete = False
        append_progress(out, "upstream verification", passed=False, error=str(exc))
        decision = {
            "schema_version": "phase_139ys_decision_v1",
            "decision": "upstream_evidence_incomplete",
            "next": "139YS_UPSTREAM_EVIDENCE_REPAIR",
            "verdict": "INSTNCT_POCKET_GATED_SCALE_OR_BRIDGE_UPSTREAM_INCOMPLETE",
            "boundary": BOUNDARY_TEXT,
            "clean_negative_valid": True,
            "planning_only": True,
            "artifact_only": True,
            "upstream_error": str(exc),
            "architecture_superiority_claimed": False,
            "value_grounding_claimed": False,
            "pocket_mechanism_claimed": False,
            **FALSE_FLAGS,
        }
        write_json(out / "decision.json", decision)
        write_json(out / "summary.json", {"schema_version": "phase_139ys_summary_v1", "milestone": MILESTONE, "status": "blocked", "boundary": BOUNDARY_TEXT, **decision})
        write_text(out / "report.md", f"# {MILESTONE}\n\nDecision: `{decision['decision']}`\n\nNext: `{decision['next']}`\n\nBoundary: {BOUNDARY_TEXT}\n\nUpstream evidence is incomplete: `{str(exc)}`\n")
        write_json(out / "queue.json", {"schema_version": "phase_139ys_queue_v1", "milestone": MILESTONE, "status": "complete", "decision": decision["decision"], "next": decision["next"]})
        append_progress(out, "decision", decision=decision["decision"], next=decision["next"])
        append_progress(out, "final verdict", verdict=decision["verdict"])
        return 0

    if not upstream_complete:
        raise AssertionError("unreachable")

    write_json(out / "upstream_139yr_manifest.json", upstreams["139yr"])
    write_json(out / "upstream_139yq_manifest.json", upstreams["139yq"])
    write_json(out / "upstream_138yo_manifest.json", upstreams["138yo"])
    write_json(out / "upstream_138yk_manifest.json", upstreams["138yk"])
    append_progress(out, "upstream verification", passed=True, upstream_hash=stable_hash(upstreams))

    ast_report = scan_ast()
    write_json(out / "ast_shortcut_scan_report.json", ast_report)
    append_progress(out, "artifact loading", config_hash=stable_hash(config), ast_passed=ast_report["passed"])

    chain = build_evidence_chain(upstreams)
    write_json(out / "evidence_chain_summary.json", chain)
    append_progress(out, "evidence chain summary", chain_complete=chain["chain_complete"])

    matrix = build_decision_matrix()
    write_json(out / "scale_vs_bridge_decision_matrix.json", matrix)
    append_progress(out, "scale vs bridge decision matrix", selected=matrix["selected_option"])

    risks = {
        "schema_version": "phase_139ys_risk_register_v1",
        "risks": [
            {
                "risk": "140A fails because marker dependency is stronger than current evidence reveals",
                "mitigation": "route clean negative to 140B_MARKER_DEPENDENCY_ANALYSIS",
            },
            {
                "risk": "noisy prompts break value binding through distraction rather than pocket causality",
                "mitigation": "add noisy distractor controls and route to 140D_NOISY_PROMPT_VALUE_BINDING_ANALYSIS",
            },
            {
                "risk": "visible prompt value bypass recreates 138YO-style evidence",
                "mitigation": "require visible bypass control failure and closed-pocket ablation failure",
            },
            {
                "risk": "longer 140A execution loses partial evidence on crash",
                "mitigation": "require heartbeat progress, summary, and report refreshes every few minutes or less",
            },
        ],
    }
    write_json(out / "risk_register.json", risks)

    target_plan = build_target_140a_plan()
    write_json(out / "target_140a_milestone_plan.json", target_plan)
    append_progress(out, "target 140A plan writing", next=target_plan["milestone"])

    recommendation = {
        "schema_version": "phase_139ys_next_milestone_recommendation_v1",
        "decision": "real_task_bridge_recommended",
        "recommended_next": "140A_INSTNCT_POCKET_GATED_NOISY_MARKER_BRIDGE_PROBE",
        "rejected_next": "139YT_MARKER_BOUND_POCKET_SCALE_EXTENSION",
        "reason": "Current marker-bound pocket proof, scale confirm, and non-gradient mutation selection are stable; the next useful test is reduced-scaffold bridge behavior.",
        "evidence_hash": stable_hash(chain),
    }
    write_json(out / "next_milestone_recommendation.json", recommendation)
    append_progress(out, "recommendation", next=recommendation["recommended_next"])

    decision = {
        "schema_version": "phase_139ys_decision_v1",
        "decision": "real_task_bridge_recommended",
        "next": "140A_INSTNCT_POCKET_GATED_NOISY_MARKER_BRIDGE_PROBE",
        "verdict": "INSTNCT_POCKET_GATED_SCALE_OR_REAL_TASK_BRIDGE_RECOMMENDED",
        "boundary": BOUNDARY_TEXT,
        "planning_only": True,
        "artifact_only": True,
        "clean_negative_valid": True,
        "recommended_option": matrix["selected_option"],
        "gradient_used": False,
        "architecture_superiority_claimed": False,
        "value_grounding_claimed": False,
        "pocket_mechanism_claimed": False,
        **FALSE_FLAGS,
    }
    write_json(out / "decision.json", decision)

    summary = {
        "schema_version": "phase_139ys_summary_v1",
        "milestone": MILESTONE,
        "status": "complete",
        "boundary": BOUNDARY_TEXT,
        "analysis_config_hash": stable_hash(config),
        "upstreams": upstreams,
        "decision_matrix": {
            "selected_option": matrix["selected_option"],
            "decision": matrix["decision"],
            "next": matrix["next"],
        },
        "target_140a_milestone": target_plan["milestone"],
        **decision,
    }
    write_json(out / "summary.json", summary)
    write_report(out, decision, chain, recommendation, target_plan)
    append_progress(out, "decision", decision=decision["decision"], next=decision["next"])
    append_progress(out, "final verdict", verdict=decision["verdict"])
    write_json(out / "queue.json", {"schema_version": "phase_139ys_queue_v1", "milestone": MILESTONE, "status": "complete", "decision": decision["decision"], "next": decision["next"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
