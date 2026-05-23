#!/usr/bin/env python3
"""138YP artifact-only plan for pocket-gated INSTNCT value grounding."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "STABLE_LOOP_PHASE_LOCK_138YP_INSTNCT_MUTATION_POCKET_GATED_VALUE_GROUNDING_PLAN"
DEFAULT_OUT = Path("target/pilot_wave/stable_loop_phase_lock_138yp_instnct_mutation_pocket_gated_value_grounding_plan/smoke")
DEFAULT_138YO_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_138yo_instnct_mutation_value_grounding_comparison_probe/smoke")
RUNNER_PATH = Path(__file__).resolve()
CHECKER_PATH = REPO_ROOT / "scripts/probes/run_stable_loop_phase_lock_138yp_instnct_mutation_pocket_gated_value_grounding_plan_check.py"
HELPER_PATH = REPO_ROOT / "scripts/probes/shared_raw_generation_helper.py"

FALSE_FLAGS = {
    "reasoning_restored": False,
    "reasoning_subtrack_real_raw_evidence_partially_restored": False,
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
    "138YP is artifact-only planning after 138YO. It does not train, infer, "
    "call the helper, mutate checkpoints, modify helper/backend/runtime/release "
    "surfaces, import old phase runners, start services, deploy, or claim broad "
    "assistant capability. It designs the next probe where value grounding must "
    "be pocket-gated and ablation-sensitive."
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
    return {"schema_version": "phase_138yp_ast_scan_v1", "passed": not failures, "failures": failures}


def require_138yo(root: Path) -> dict[str, Any]:
    required = ["decision.json", "aggregate_metrics.json", "pocket_ablation_report.json", "arm_comparison.json", "generated_before_scoring_report.json", "determinism_replay_report.json"]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise RuntimeError(f"missing 138YO artifacts: {missing}")
    decision = read_json(root / "decision.json")
    aggregate = read_json(root / "aggregate_metrics.json")
    pocket = read_json(root / "pocket_ablation_report.json")
    replay = read_json(root / "determinism_replay_report.json")
    if decision.get("decision") != "instnct_adapter_prompt_bound_value_grounding_improves":
        raise RuntimeError(f"bad 138YO decision: {decision.get('decision')}")
    if decision.get("next") != "138YP_INSTNCT_MUTATION_POCKET_GATED_VALUE_GROUNDING_PLAN":
        raise RuntimeError(f"bad 138YO next: {decision.get('next')}")
    if decision.get("pocket_mechanism_claimed") is not False:
        raise RuntimeError("138YO overclaimed pocket mechanism")
    if aggregate.get("instnct_answer_value_accuracy", 0.0) <= aggregate.get("byte_gru_answer_value_accuracy", 1.0):
        raise RuntimeError("138YO did not show adapter improvement")
    if aggregate.get("instnct_pocket_writeback_rate") != 0.0:
        raise RuntimeError("138YO pocket writeback evidence changed; recheck required")
    if pocket.get("answer_value_accuracy_delta") != 0.0:
        raise RuntimeError("138YO ablation delta changed; recheck required")
    if replay.get("deterministic_replay_passed") is not True:
        raise RuntimeError("138YO determinism did not pass")
    return {
        "root": rel(root),
        "decision": decision.get("decision"),
        "next": decision.get("next"),
        "verdict": decision.get("verdict"),
        "instnct_answer_value_accuracy": aggregate.get("instnct_answer_value_accuracy"),
        "byte_gru_answer_value_accuracy": aggregate.get("byte_gru_answer_value_accuracy"),
        "instnct_minus_byte_answer_value_accuracy": aggregate.get("instnct_minus_byte_answer_value_accuracy"),
        "instnct_pocket_writeback_rate": aggregate.get("instnct_pocket_writeback_rate"),
        "pocket_ablation_delta_answer_value_accuracy": pocket.get("answer_value_accuracy_delta"),
        "pocket_writeback_decision_critical": pocket.get("pocket_writeback_decision_critical"),
        "deterministic_replay_passed": replay.get("deterministic_replay_passed"),
    }


def write_report(out: Path, decision: dict[str, Any], bypass: dict[str, Any], plan: dict[str, Any]) -> None:
    text = f"""# {MILESTONE}

Decision: `{decision['decision']}`

Next: `{decision['next']}`

Boundary: {BOUNDARY_TEXT}

138YO showed real adapter-path improvement, but not pocket-mechanism evidence:

- INSTNCT adapter answer value accuracy: `{bypass['instnct_answer_value_accuracy']}`
- Byte-GRU answer value accuracy: `{bypass['byte_gru_answer_value_accuracy']}`
- Pocket writeback rate: `{bypass['instnct_pocket_writeback_rate']}`
- Pocket ablation delta: `{bypass['pocket_ablation_delta_answer_value_accuracy']}`

138YQ must therefore make value emission depend on threshold-open pocket writeback.
If pocket ablation does not reduce value accuracy, 138YQ must fail.

Target milestone: `{plan['milestone']}`
"""
    write_text(out / "report.md", text)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run 138YP pocket-gated value-grounding plan")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--upstream-138yo-root", type=Path, default=DEFAULT_138YO_ROOT)
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    args = parser.parse_args()

    out = resolve_target_out(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    append_progress(out, "startup", milestone=MILESTONE, heartbeat_sec=args.heartbeat_sec)
    write_json(out / "queue.json", {"schema_version": "phase_138yp_queue_v1", "milestone": MILESTONE, "status": "running"})

    upstream = require_138yo(resolve_repo_path(args.upstream_138yo_root))
    write_json(out / "upstream_138yo_manifest.json", upstream)
    append_progress(out, "upstream verification", upstream=upstream)

    config = {
        "schema_version": "phase_138yp_analysis_config_v1",
        "milestone": MILESTONE,
        "boundary": BOUNDARY_TEXT,
        "artifact_only": True,
        "planning_only": True,
        "training_performed": False,
        "new_model_inference_run": False,
        "shared_helper_called": False,
        "checkpoint_mutated": False,
        "helper_modified": False,
        "runtime_surface_mutated": False,
        "root_license_changed": False,
        **FALSE_FLAGS,
    }
    write_json(out / "analysis_config.json", config)
    append_progress(out, "artifact loading", config_hash=stable_hash(config))

    ast_report = scan_ast()
    write_json(out / "ast_shortcut_scan_report.json", ast_report)

    bypass = {
        "schema_version": "phase_138yp_pocket_bypass_diagnosis_v1",
        "root_cause": "prompt_bound_value_extraction_bypasses_pocket_writeback",
        "adapter_path_improves_over_byte_gru": True,
        "pocket_mechanism_supported": False,
        "byte_gru_answer_value_accuracy": upstream["byte_gru_answer_value_accuracy"],
        "instnct_answer_value_accuracy": upstream["instnct_answer_value_accuracy"],
        "instnct_minus_byte_answer_value_accuracy": upstream["instnct_minus_byte_answer_value_accuracy"],
        "instnct_pocket_writeback_rate": upstream["instnct_pocket_writeback_rate"],
        "pocket_ablation_delta_answer_value_accuracy": upstream["pocket_ablation_delta_answer_value_accuracy"],
        "diagnosis": [
            "138YO used the same 138YK prompts and improved value output.",
            "The current prompt-bound adapter could extract visible EV/VAL/SYM tokens without pocket writeback.",
            "Pocket ablation did not reduce accuracy.",
            "The next probe must make pocket writeback necessary for any positive value-grounding claim.",
        ],
    }
    write_json(out / "pocket_bypass_diagnosis.json", bypass)
    append_progress(out, "pocket bypass diagnosis", root=bypass["root_cause"])

    gating_requirements = {
        "schema_version": "phase_138yp_pocket_gating_requirements_v1",
        "required_mechanisms": [
            "value_selection_requires_open_pocket",
            "threshold_tick_gate_must_fire_before_value_decode",
            "closed_pocket_or_ablation_must_emit_fallback_or_wrong_value",
            "pocket_writeback_count_must_be_positive_for_positive_rows",
            "pocket_ablation_delta_must_be_decision_critical",
            "highway_retention_must_remain_true",
            "no expected/scorer/oracle metadata in helper requests",
        ],
        "minimum_positive_gates": {
            "instnct_answer_value_accuracy": 0.25,
            "pocket_writeback_rate": 0.95,
            "phase_transport_success_rate": 0.95,
            "pocket_ablation_delta_answer_value_accuracy": 0.20,
            "pocket_ablation_answer_value_accuracy_max": 0.05,
            "deterministic_replay_passed": True,
        },
        "overclaim_rejections": [
            "positive without nonzero pocket writeback",
            "positive without ablation delta",
            "positive from visible prompt-value extraction only",
            "architecture superiority claim from adapter smoke alone",
        ],
    }
    write_json(out / "pocket_gating_requirements.json", gating_requirements)
    append_progress(out, "pocket gating requirements", positive_gates=gating_requirements["minimum_positive_gates"])

    backend_contract = {
        "schema_version": "phase_138yp_target_138yq_backend_contract_v1",
        "backend_name": "repo_local_instnct_mutation_graph",
        "allowed_helper_request_keys": ["prompt", "checkpoint_path", "checkpoint_hash", "seed", "max_new_tokens", "generation_config"],
        "new_manifest_fields_allowed_in_138yq": {
            "value_selection_requires_open_pocket": True,
            "pocket_payload_markers": ["POCKET_VALUE=", "POCKET_BIND=", "POCKET_TABLE_ROW="],
            "visible_value_bypass_forbidden": True,
            "closed_pocket_fallback_value": "SYM_POCKET_CLOSED",
            "ablation_gate_marker": "GATE:NEVER_OPEN",
        },
        "helper_change_allowed_in_138yq": "strict backend dispatch semantics only; no public API change",
        "public_api_change_allowed": False,
        "source_checkpoint_mutation_allowed": False,
        "post_generation_repair_allowed": False,
        "oracle_metadata_allowed": False,
    }
    write_json(out / "target_138yq_backend_contract.json", backend_contract)

    eval_design = {
        "schema_version": "phase_138yp_target_138yq_eval_design_v1",
        "families": [
            "POCKET_DIRECT_BINDING",
            "POCKET_TABLE_BINDING",
            "POCKET_RULE_DERIVED",
            "POCKET_OOD_SYMBOL_BINDING",
            "POCKET_CONTRAST_SAME_TEMPLATE_DISTINCT_VALUES",
            "POCKET_CLOSED_NEGATIVE_CONTROL",
        ],
        "row_requirements": [
            "same helper request keys as raw helper",
            "ANSWER=E required",
            "expected value may appear only in pocket payload syntax accepted by gated backend",
            "same prompt with gate ablated must fail",
            "same-family contrast groups require distinct correct values",
            "no train namespace leak",
            "no stale chat",
        ],
        "comparison_arms": [
            "byte_gru_138yk_or_current_byte_lm_baseline",
            "instnct_pocket_gated_main",
            "instnct_pocket_gate_ablation",
            "instnct_visible_value_bypass_control",
        ],
    }
    write_json(out / "target_138yq_eval_design.json", eval_design)

    ablation_spec = {
        "schema_version": "phase_138yp_pocket_ablation_gate_spec_v1",
        "main_arm": "gate marker present; threshold pocket opens; writeback supplies value to decoder",
        "ablation_arm": "gate marker removed or replaced; pocket never opens; decoder cannot read pocket payload",
        "pass_condition": "decision-critical: main value accuracy exceeds ablation by at least 0.20 and ablation accuracy remains <= 0.05",
        "decision_critical_metrics": [
            "pocket_writeback_rate",
            "phase_transport_success_rate",
            "pocket_ablation_delta_answer_value_accuracy",
            "pocket_ablation_delta_contrastive_accuracy",
        ],
    }
    write_json(out / "pocket_ablation_gate_spec.json", ablation_spec)

    mutation_bridge = {
        "schema_version": "phase_138yp_mutation_credit_assignment_bridge_v1",
        "gradient_used": False,
        "credit_signal": "fitness difference between pocket-gated main and ablated/closed-pocket controls",
        "mutation_probe_requirement": "138YQ must record whether candidate pocket rules improve decision-critical helper eval metrics without backprop.",
        "black_box_run_rejection": "progress.jsonl, summary.json, and report.md must be refreshed during long mutation probes at heartbeat cadence.",
    }
    write_json(out / "mutation_credit_assignment_bridge.json", mutation_bridge)

    anti_shortcut = {
        "schema_version": "phase_138yp_anti_shortcut_requirements_v1",
        "reject": [
            "prompt-bound visible value extraction without pocket writeback",
            "ANSWER=E prefix-only success",
            "family-default shortcut",
            "high-frequency train-value replay",
            "post-generation repair",
            "regex fixer",
            "oracle/rerank/verifier/LLM judge",
            "expected-output construction",
            "best-of-n",
            "retry loop",
            "threshold weakening",
        ],
    }
    write_json(out / "anti_shortcut_requirements.json", anti_shortcut)

    failure_routes = {
        "schema_version": "phase_138yp_target_138yq_failure_routes_v1",
        "routes": {
            "pocket_writeback_not_used": "138YO_INSTNCT_MUTATION_VALUE_GROUNDING_COMPARISON_PROBE",
            "pocket_ablation_not_decision_critical": "138YP_POCKET_GATING_DESIGN_RECHECK",
            "visible_value_bypass_detected": "138YP_PROMPT_BOUND_BYPASS_ANALYSIS",
            "no_value_improvement": "138YQ_FAILURE_ANALYSIS",
            "nondeterministic_probe": "138N_DETERMINISM_FAILURE_ANALYSIS",
            "raw_helper_integrity_failure": "135G_RAW_HELPER_ORACLE_SHORTCUT_REMOVAL",
            "scorer_or_task_weakness": "138E_REASONING_SCORER_OR_TASK_WEAKNESS_ANALYSIS",
        },
    }
    write_json(out / "target_138yq_failure_routes.json", failure_routes)

    milestone_plan = {
        "schema_version": "phase_138yp_next_138yq_milestone_plan_v1",
        "milestone": "138YQ_INSTNCT_POCKET_GATED_VALUE_GROUNDING_PROBE",
        "type": "targeted helper/backend probe",
        "train_allowed": False,
        "mutation_allowed": True,
        "helper_backend_modification_allowed": True,
        "public_api_change_allowed": False,
        "source_checkpoint_mutation_allowed": False,
        "clean_negative_accepted": True,
        "required_artifacts": [
            "queue.json",
            "progress.jsonl",
            "upstream_138yp_manifest.json",
            "adapter_contract.json",
            "instnct_pocket_gated_manifest.json",
            "eval_rows.jsonl",
            "pocket_payload_manifest.json",
            "raw_generation_trace.jsonl",
            "raw_generation_results.jsonl",
            "pocket_trace.jsonl",
            "pocket_ablation_results.jsonl",
            "scoring_results.jsonl",
            "pocket_gating_metrics.json",
            "arm_comparison.json",
            "determinism_replay_report.json",
            "decision.json",
            "summary.json",
            "report.md",
        ],
        "success_route": {
            "decision": "instnct_pocket_gated_value_grounding_probe_positive",
            "next": "139YQ_INSTNCT_POCKET_GATED_VALUE_GROUNDING_SCALE_CONFIRM",
        },
    }
    write_json(out / "next_138yq_milestone_plan.json", milestone_plan)
    append_progress(out, "target 138YQ plan writing", next=milestone_plan["milestone"])

    gaps = {
        "schema_version": "phase_138yp_diagnostic_gap_register_v1",
        "gaps": [
            "No measured hidden-state, logit, grower, or scout internals in 138YO.",
            "138YO adapter improvement is not architecture-superiority evidence.",
            "138YO did not prove pocket writeback caused value grounding.",
        ],
    }
    risks = {
        "schema_version": "phase_138yp_risk_register_v1",
        "risks": [
            {"risk": "helper_adapter_overfits_to pocket syntax", "mitigation": "include closed-pocket and visible-bypass controls"},
            {"risk": "ablation control too weak", "mitigation": "require decision-critical ablation delta"},
            {"risk": "mutation loop black-box loss", "mitigation": "heartbeat progress and partial outcome artifacts"},
        ],
    }
    write_json(out / "diagnostic_gap_register.json", gaps)
    write_json(out / "risk_register.json", risks)

    decision = {
        "schema_version": "phase_138yp_decision_v1",
        "decision": "instnct_mutation_pocket_gated_value_grounding_plan_complete",
        "next": "138YQ_INSTNCT_POCKET_GATED_VALUE_GROUNDING_PROBE",
        "verdict": "INSTNCT_POCKET_GATED_VALUE_GROUNDING_PLAN_COMPLETE",
        "clean_negative_valid": True,
        "architecture_superiority_claimed": False,
        "pocket_mechanism_claimed": False,
        "value_grounding_claimed": False,
        **FALSE_FLAGS,
    }
    write_json(out / "decision.json", decision)
    summary = {
        "schema_version": "phase_138yp_summary_v1",
        "milestone": MILESTONE,
        "status": "complete",
        "boundary": BOUNDARY_TEXT,
        "upstream": upstream,
        "pocket_bypass_root_cause": bypass["root_cause"],
        "next_plan": milestone_plan,
        **decision,
    }
    write_json(out / "summary.json", summary)
    write_report(out, decision, bypass, milestone_plan)
    append_progress(out, "decision", decision=decision["decision"], next=decision["next"])
    append_progress(out, "final verdict", verdict=decision["verdict"])
    write_json(out / "queue.json", {"schema_version": "phase_138yp_queue_v1", "milestone": MILESTONE, "status": "complete", "decision": decision["decision"], "next": decision["next"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
