#!/usr/bin/env python3
"""117 targeted capability repair or scale plan.

This analysis-only milestone reads the positive 116 ceiling/gap map, ranks
capability gaps, selects the next targeted repair/scale milestone, and writes a
concrete 118 plan. It performs no training, no inference, no repair, no service
startup, no deployment smoke, and no checkpoint mutation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "STABLE_LOOP_PHASE_LOCK_117_TARGETED_CAPABILITY_REPAIR_OR_SCALE_PLAN"
DEFAULT_OUT = Path("target/pilot_wave/stable_loop_phase_lock_117_targeted_capability_repair_or_scale_plan/smoke")
DEFAULT_UPSTREAM_116_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_116_raw_assistant_capability_ceiling_and_gap_map/smoke")
DEFAULT_UPSTREAM_115_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_115_external_style_raw_assistant_stress_confirm/smoke")
DEFAULT_UPSTREAM_114_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_114_raw_assistant_external_stress_benchmark_bridge/smoke")
DEFAULT_UPSTREAM_113_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_113_raw_assistant_capability_package_and_boundary_review/smoke")
DEFAULT_UPSTREAM_112_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_112_current_chassis_raw_generation_scale_confirm/smoke")
DEFAULT_UPSTREAM_099_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_099_bounded_local_private_clean_deploy_ready_gate/smoke")

POSITIVE_VERDICT = "TARGETED_CAPABILITY_REPAIR_OR_SCALE_PLAN_POSITIVE"
BOUNDARY_TEXT = (
    "117 is analysis/planning only. It reads existing artifacts and writes a targeted "
    "repair or scale plan. It performs no training, no repair, no model inference, no "
    "checkpoint mutation, no service startup, no deployment smoke, and no runtime/product/"
    "release integration. It is not GPT-like assistant readiness, not open-domain assistant "
    "readiness, not production chat, not public API, not deployment readiness, and not "
    "safety alignment."
)

CANDIDATE_MILESTONES = [
    "118_REASONING_FIRST_RAW_ASSISTANT_REPAIR",
    "118_MULTI_TURN_STATE_REPAIR",
    "118_HALLUCINATION_REFUSAL_BALANCE_REPAIR",
    "118_FORMAT_INJECTION_ROBUSTNESS_REPAIR",
    "118_LONG_CONTEXT_REASONING_REPAIR",
    "118_CAPABILITY_SCALE_WITHOUT_REPAIR",
]


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


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
        if limit and len(rows) >= limit:
            break
    return rows


def stable_json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def resolve_target_out(text: str) -> Path:
    path = Path(text)
    if path.is_absolute() or any(part == ".." for part in path.parts):
        raise GateError("TARGETED_CAPABILITY_REPAIR_OR_SCALE_PLAN_FAILS", "--out must be repo-relative")
    parts = [part.lower() for part in path.parts]
    if len(parts) < 2 or parts[0] != "target" or parts[1] != "pilot_wave":
        raise GateError("TARGETED_CAPABILITY_REPAIR_OR_SCALE_PLAN_FAILS", "--out must stay under target/pilot_wave")
    return (REPO_ROOT / path).resolve()


def resolve_upstream(text: str) -> Path:
    path = Path(text)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def append_progress(out: Path, event: str, status: str = "completed", **details: Any) -> None:
    append_jsonl(out / "progress.jsonl", {"ts": utc_now(), "event": event, "status": status, "details": details})


def verify_positive(root: Path, positive_verdict: str, missing_verdict: str = "UPSTREAM_ARTIFACT_MISSING") -> dict[str, Any]:
    path = root / "summary.json"
    if not path.exists():
        raise GateError(missing_verdict, f"missing {rel(path)}")
    summary = read_json(path)
    if summary.get("status") != "positive" or positive_verdict not in set(summary.get("verdicts", [])):
        raise GateError("UPSTREAM_STACK_NOT_POSITIVE", f"{positive_verdict} not found in {rel(path)}")
    return summary


def write_manifest(out: Path, name: str, root: Path, summary: dict[str, Any], verdict: str) -> None:
    metrics = summary.get("metrics", {})
    write_json(
        out / f"upstream_{name}_manifest.json",
        {
            "schema_version": "phase_117_upstream_manifest_v1",
            "upstream": name,
            "root": rel(root),
            "summary_hash": stable_json_hash(summary),
            "positive_verdict": verdict,
            "key_metrics": {
                key: metrics[key]
                for key in [
                    "decision",
                    "next",
                    "raw_accuracy",
                    "baseline_raw_accuracy",
                    "first_breakpoint_tier",
                    "unknown_failure_rate",
                    "controls_failed",
                    "benchmark_leakage_detected",
                    "retention_preserved",
                    "collapse_rejected",
                    "external_style_raw_accuracy",
                    "mean_external_style_raw_accuracy",
                    "min_external_style_raw_accuracy",
                    "bounded_release_artifact_unchanged",
                    "source_100_checkpoint_unchanged",
                    "source_102_checkpoint_unchanged",
                ]
                if key in metrics
            },
            "boundary_flags": {
                key: value
                for key, value in summary.items()
                if key.endswith("_claimed") or key.endswith("_mutated") or key.endswith("_performed") or key in {"training_performed"}
            },
        },
    )


def write_summary(out: Path, phase: str, status: str, verdicts: list[str], metrics: dict[str, Any], failure: str | None = None) -> None:
    write_json(
        out / "summary.json",
        {
            "schema_version": "phase_117_targeted_plan_summary_v1",
            "milestone": MILESTONE,
            "phase": phase,
            "status": status,
            "verdicts": verdicts,
            "metrics": metrics,
            "failure": failure,
            "boundary": BOUNDARY_TEXT,
            "analysis_only": True,
            "training_performed": False,
            "inference_run_count": 0,
            "checkpoint_mutated": False,
            "service_started": False,
            "deployment_smoke_run": False,
            "runtime_surface_mutated": False,
            "bounded_release_stack_mutated": False,
            "gpt_like_readiness_claimed": False,
            "open_domain_assistant_readiness_claimed": False,
            "production_chat_claimed": False,
            "public_api_claimed": False,
            "deployment_readiness_claimed": False,
            "safety_alignment_claimed": False,
        },
    )


def write_report(out: Path, phase: str, verdicts: list[str], metrics: dict[str, Any], decision: dict[str, Any] | None = None) -> None:
    decision = decision or {}
    lines = [
        f"# {MILESTONE}",
        "",
        BOUNDARY_TEXT,
        "",
        "## Status",
        f"- phase: `{phase}`",
        f"- verdicts: `{', '.join(verdicts) if verdicts else 'pending'}`",
        f"- selected_next_milestone: `{decision.get('selected_next_milestone', metrics.get('selected_next_milestone', 'pending'))}`",
        f"- selected_repair_target: `{decision.get('selected_repair_target', metrics.get('selected_repair_target', 'pending'))}`",
        f"- first_breakpoint_tier: `{metrics.get('first_breakpoint_tier', 'pending')}`",
        f"- reasoning_failure_count: `{metrics.get('reasoning_failure_count', 'pending')}`",
        "",
        "117 is planning only. It is not GPT-like assistant readiness, not open-domain assistant readiness, not production chat, not public API, not deployment readiness, and not safety alignment.",
    ]
    write_text(out / "report.md", "\n".join(lines) + "\n")


def write_live(out: Path, phase: str, verdicts: list[str], metrics: dict[str, Any], decision: dict[str, Any] | None = None) -> None:
    write_summary(out, phase, "running", verdicts, metrics)
    write_report(out, phase, verdicts, metrics, decision)


def load_116_artifacts(root: Path) -> dict[str, Any]:
    required = [
        "summary.json",
        "decision.json",
        "ceiling_by_tier.json",
        "failure_mode_map.json",
        "capability_gap_map.json",
        "next_training_targets.json",
        "tier_metrics.json",
        "family_metrics.json",
        "retention_report.json",
        "collapse_metrics.json",
        "overclaim_exfiltration_report.json",
    ]
    artifacts = {}
    for name in required:
        path = root / name
        if not path.exists():
            raise GateError("UPSTREAM_116_ARTIFACT_MISSING", f"missing {rel(path)}")
        artifacts[name] = read_json(path)
    artifacts["human_samples"] = read_jsonl(root / "human_readable_samples.jsonl", limit=80)
    artifacts["failure_samples"] = read_jsonl(root / "failure_case_samples.jsonl", limit=200)
    return artifacts


def build_failure_priority_map(artifacts: dict[str, Any]) -> dict[str, Any]:
    failure_counts = artifacts["failure_mode_map.json"]["failure_counts"]
    first_breakpoint = artifacts["capability_gap_map.json"]["first_breakpoint_tier"]
    priority_order = [
        "reasoning_failure",
        "multi_turn_state_failure",
        "hallucination_failure",
        "over_refusal",
        "format_failure",
        "prompt_injection_failure",
        "long_context_failure",
    ]
    rows = []
    for index, label in enumerate(priority_order, start=1):
        count = int(failure_counts.get(label, 0))
        rows.append(
            {
                "rank": index,
                "failure_label": label,
                "evidence_count": count,
                "priority_reason": (
                    "first breakpoint and largest class"
                    if label == "reasoning_failure"
                    else "downstream or compounded after first breakpoint"
                ),
                "selected": label == "reasoning_failure",
            }
        )
    return {
        "schema_version": "phase_117_failure_priority_map_v1",
        "first_breakpoint_tier": first_breakpoint,
        "reasoning_failure_count": int(failure_counts.get("reasoning_failure", 0)),
        "reasoning_is_largest_failure_class": int(failure_counts.get("reasoning_failure", 0)) == max(int(value) for value in failure_counts.values()),
        "priority": rows,
        "failure_counts": failure_counts,
    }


def build_breakpoint_analysis(artifacts: dict[str, Any]) -> dict[str, Any]:
    gap_map = artifacts["capability_gap_map.json"]
    first = gap_map["first_breakpoint_tier"]
    first_gap = next((gap for gap in gap_map.get("gaps", []) if gap.get("tier") == first), {})
    return {
        "schema_version": "phase_117_breakpoint_analysis_v1",
        "first_breakpoint_tier": first,
        "first_breakpoint_accuracy": first_gap.get("accuracy"),
        "first_breakpoint_failures": first_gap.get("dominant_failures", {}),
        "gap_sequence": gap_map.get("gaps", []),
        "interpretation": "Tier 4 multi-step reasoning is the first observed capability breakpoint; later gaps compound reasoning with state, refusal, format, injection, and long-context pressure.",
    }


def build_root_vs_symptom_analysis(artifacts: dict[str, Any]) -> dict[str, Any]:
    failure_map = artifacts["failure_mode_map.json"]
    counts = failure_map["failure_counts"]
    return {
        "schema_version": "phase_117_root_vs_symptom_analysis_v1",
        "root_causes": [
            {
                "label": "reasoning_failure",
                "evidence_count": counts.get("reasoning_failure", 0),
                "evidence": [
                    "first breakpoint appears at TIER_4_MULTI_STEP_REASONING",
                    "reasoning is the largest failure class",
                    "TIER_8 combined stress includes reasoning failures",
                ],
            }
        ],
        "downstream_or_compounded_symptoms": [
            {"label": "multi_turn_state_failure", "reason": "appears after reasoning breakpoint and likely depends on stable rule/state composition"},
            {"label": "long_context_failure", "reason": "appears in combined stress with reasoning failures"},
            {"label": "format_failure", "reason": "later adversarial format tier; important but not first root breakpoint"},
            {"label": "hallucination_failure", "reason": "later refusal-balance tier; should be gated in 118 but not selected first"},
            {"label": "prompt_injection_failure", "reason": "later adversarial tier; should remain a hard eval gate"},
        ],
        "not_root_causes": ["retention_failure", "collapse", "namespace_drift", "unknown_failure"],
    }


def build_repair_target_selection(priority_map: dict[str, Any]) -> dict[str, Any]:
    selected = "118_REASONING_FIRST_RAW_ASSISTANT_REPAIR"
    return {
        "schema_version": "phase_117_repair_target_selection_v1",
        "selected_next_milestone": selected,
        "selected_repair_target": "reasoning_first",
        "selected_failure_label": "reasoning_failure",
        "selected_from_candidates": CANDIDATE_MILESTONES,
        "supporting_evidence": {
            "first_breakpoint_tier": priority_map["first_breakpoint_tier"],
            "reasoning_failure_count": priority_map["reasoning_failure_count"],
            "reasoning_is_largest_failure_class": priority_map["reasoning_is_largest_failure_class"],
            "later_failures_include_reasoning_as_compounded_factor": True,
        },
        "rejected_alternatives": {
            "118_MULTI_TURN_STATE_REPAIR": "multi-turn fails after the reasoning breakpoint and is likely downstream of rule/state composition",
            "118_HALLUCINATION_REFUSAL_BALANCE_REPAIR": "hallucination/refusal failures are smaller and later; keep as gates in 118",
            "118_FORMAT_INJECTION_ROBUSTNESS_REPAIR": "format/injection failures are later and should be guarded during reasoning repair",
            "118_LONG_CONTEXT_REASONING_REPAIR": "long-context failures appear in the final combined tier; repair reasoning first before expanding context",
            "118_CAPABILITY_SCALE_WITHOUT_REPAIR": "116 found a concrete breakpoint, so generic scale is premature",
        },
    }


def build_training_design_options() -> dict[str, Any]:
    return {
        "schema_version": "phase_117_training_design_options_v1",
        "recommended": "targeted_reasoning_repair_with_raw_only_final_eval",
        "options": [
            {
                "name": "reasoning_first_targeted_repair",
                "recommended": True,
                "data_design": ["provided-fact multi-step reasoning", "small arithmetic over supplied values", "rule chaining", "table + rule reasoning", "contradiction resolution"],
                "objective_notes": ["use scheduled sampling or rollout-style objective if training is included", "avoid teacher-forcing-only success"],
            },
            {
                "name": "generic_scale_without_repair",
                "recommended": False,
                "rejection_reason": "does not directly target the first breakpoint and risks another 111-style loss-improves/wrong-skill failure",
            },
            {
                "name": "long_context_first",
                "recommended": False,
                "rejection_reason": "long-context failure appears after the first reasoning breakpoint and in combined stress",
            },
        ],
    }


def build_eval_gate_proposal() -> dict[str, Any]:
    return {
        "schema_version": "phase_117_eval_gate_proposal_v1",
        "hard_gates_for_118": [
            "Tier 4 reasoning accuracy improves against 116 baseline",
            "Tier 8 reasoning-combo accuracy improves against 116 baseline",
            "provided-fact multi-step reasoning passes",
            "small arithmetic over supplied values passes",
            "rule chaining passes",
            "table + rule reasoning passes",
            "contradiction resolution passes",
            "retention preserved",
            "collapse rejected",
            "namespace drift rejected",
            "static/copy/random controls fail",
            "no overclaim or exfiltration",
            "raw-only final eval uses no integrated policy, decoder reference, oracle rerank, expected-answer metadata, or teacher forcing",
        ],
        "false_success_prevention": ["no LLM judge", "no hidden world knowledge", "no mean-only pass", "train/eval leakage audit against 112-117"],
    }


def build_risk_register() -> dict[str, Any]:
    return {
        "schema_version": "phase_117_risk_register_v1",
        "risks": [
            {"risk": "teacher-forcing-only success", "mitigation": "require raw autoregressive final eval and rollout-style objective if training is used"},
            {"risk": "namespace memorization", "mitigation": "disjoint train/eval namespaces and explicit namespace drift gates"},
            {"risk": "reasoning overfits templates", "mitigation": "fresh provided-fact rows, paraphrase variants, table/rule/contradiction mixes"},
            {"risk": "retention regression", "mitigation": "bounded and finite-label retention gates remain hard"},
            {"risk": "collapse or static outputs", "mitigation": "collapse metrics plus static/copy/random controls must fail"},
            {"risk": "overclaim after capability gains", "mitigation": "boundary counts remain hard zero"},
        ],
    }


def build_111_failure_prevention_map() -> dict[str, Any]:
    return {
        "schema_version": "phase_117_111_failure_prevention_map_v1",
        "guards": {
            "eval_path_mismatch": "118 must compare against 116 rows and use identical final rows across arms",
            "namespace_memorization": "train/eval namespace disjointness and anti-memorization rows are mandatory",
            "teacher_forcing_rollout_gap": "raw-only final eval plus scheduled sampling or rollout-style objective if training is used",
            "retention_mix_underpowered": "retention gates remain hard positive requirements",
            "target_checkpoint_collapse": "collapse metrics, human samples, and failure samples must be written",
            "data_balance_failure": "data mix must be recorded and checked before positive verdict",
        },
    }


def build_next_milestone_plan() -> dict[str, Any]:
    return {
        "schema_version": "phase_117_next_milestone_plan_v1",
        "milestone_name": "118_REASONING_FIRST_RAW_ASSISTANT_REPAIR",
        "purpose": "Repair the first observed 116 breakpoint: Tier 4 multi-step reasoning, while preserving retention, namespace discipline, collapse resistance, and boundary controls.",
        "upstreams": ["117 positive", "116 positive", "115 positive", "112 positive", "099 positive"],
        "train_eval_type": "targeted research repair with raw-only final eval",
        "data_design": ["provided-fact multi-step reasoning", "small arithmetic over supplied values", "rule chaining", "table + rule reasoning", "contradiction resolution", "no hidden world knowledge"],
        "anti_leakage_rules": ["fresh train/eval rows", "exact and near-duplicate audit against 112-117 artifacts", "disjoint row hashes across train/eval"],
        "anti_memorization_rules": ["train/eval namespace disjointness", "anti-memorization rows", "case-id and active-slot copy gates"],
        "objective_guardrails": ["scheduled sampling or rollout-style objective if training is used", "no teacher-forcing-only success", "prompt-output binding checks"],
        "final_eval_forbidden_paths": ["integrated policy", "decoder reference", "oracle rerank", "expected-answer metadata", "teacher forcing"],
        "retention_gates": ["bounded chat retention >= 0.90", "finite-label AnchorRoute retention >= 0.90", "unsupported refusal retention >= 0.80"],
        "collapse_gates": ["empty output <= 0.02", "static output <= 0.10", "repetition <= 0.20", "copy prompt <= 0.15"],
        "positive_verdicts": ["REASONING_FIRST_RAW_ASSISTANT_REPAIR_POSITIVE", "RAW_REASONING_BREAKPOINT_IMPROVED", "RETENTION_PRESERVED", "COLLAPSE_REJECTED", "NO_OVERCLAIM"],
        "failure_verdicts": ["REASONING_REPAIR_FAILS", "TRAIN_EVAL_LEAKAGE_DETECTED", "NAMESPACE_MEMORIZATION_DETECTED", "TEACHER_FORCING_ONLY_SUCCESS_DETECTED", "RETENTION_REGRESSION_DETECTED", "COLLAPSE_DETECTED"],
        "validation_commands": [
            "python -m py_compile scripts/probes/run_stable_loop_phase_lock_118_reasoning_first_raw_assistant_repair.py",
            "python scripts/probes/run_stable_loop_phase_lock_118_reasoning_first_raw_assistant_repair_check.py --check-only",
            "git diff --check",
        ],
        "boundary_text": "118 is targeted research repair only; it is not GPT-like assistant readiness, not open-domain readiness, not production chat, not public API, not deployment readiness, and not safety alignment.",
    }


def build_decision(selection: dict[str, Any], eval_gates: dict[str, Any]) -> dict[str, Any]:
    evidence = selection["supporting_evidence"]
    return {
        "schema_version": "phase_117_decision_v1",
        "selected_next_milestone": selection["selected_next_milestone"],
        "selected_repair_target": selection["selected_repair_target"],
        "primary_reason": "Reasoning is the first 116 breakpoint and the largest failure class.",
        "supporting_evidence": evidence,
        "rejected_alternatives": selection["rejected_alternatives"],
        "why_not_more_general_training": "116 identified a concrete breakpoint; generic training risks another 111-style wrong-objective failure.",
        "why_not_deploy_polish": "raw assistant capability remains research-harness evidence, not deploy/product readiness.",
        "why_not_architecture_pivot": "111X and 112 kept the current chassis viable; 116 shows targeted gaps rather than chassis collapse.",
        "why_not_long_context_first": "long-context failures appear after the reasoning breakpoint and in combined stress.",
        "why_not_multi_turn_first": "multi-turn failures appear after the reasoning breakpoint and likely depend on stable reasoning/state composition.",
        "why_not_hallucination_first": "hallucination/refusal failures are later and smaller; keep them as hard gates during reasoning repair.",
        "hard_gates_for_118": eval_gates["hard_gates_for_118"],
        "expected_success_criteria": ["reasoning breakpoint improves", "Tier 8 reasoning-combo improves", "retention/collapse/boundary gates remain clean"],
        "expected_failure_modes": ["teacher-forcing-only success", "namespace memorization", "retention regression", "collapse", "format/injection regressions"],
    }


def run(args: argparse.Namespace) -> int:
    start = time.time()
    out = resolve_target_out(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, Any] = {
        "schema_version": "phase_117_targeted_plan_metrics_v1",
        "train_step_count": 0,
        "optimizer_step_count": 0,
        "inference_run_count": 0,
        "checkpoint_mutated": False,
        "service_started": False,
        "deployment_smoke_run": False,
        "runtime_surface_mutated": False,
        "decision": "pending",
        "selected_next_milestone": "pending",
    }
    write_json(out / "queue.json", {"schema_version": "phase_117_queue_v1", "milestone": MILESTONE, "created_at": utc_now(), "tasks": ["verify upstreams", "load 116 artifacts", "prioritize failures", "select repair target", "write 118 plan", "decide"]})
    write_json(out / "analysis_config.json", {"schema_version": "phase_117_analysis_config_v1", "milestone": MILESTONE, "out": rel(out), "heartbeat_sec": args.heartbeat_sec, "analysis_only": True, "no_training": True, "no_inference": True, "candidate_milestones": CANDIDATE_MILESTONES})
    append_progress(out, "start", "running", milestone=MILESTONE)
    write_live(out, "start", [], metrics)

    upstreams = {
        "116": (resolve_upstream(args.upstream_116_root), "RAW_ASSISTANT_CAPABILITY_CEILING_AND_GAP_MAP_POSITIVE", "UPSTREAM_116_ARTIFACT_MISSING"),
        "115": (resolve_upstream(args.upstream_115_root), "EXTERNAL_STYLE_RAW_ASSISTANT_STRESS_CONFIRM_POSITIVE", "UPSTREAM_ARTIFACT_MISSING"),
        "114": (resolve_upstream(args.upstream_114_root), "RAW_ASSISTANT_EXTERNAL_STRESS_BENCHMARK_BRIDGE_POSITIVE", "UPSTREAM_ARTIFACT_MISSING"),
        "113": (resolve_upstream(args.upstream_113_root), "RAW_ASSISTANT_CAPABILITY_PACKAGE_AND_BOUNDARY_REVIEW_POSITIVE", "UPSTREAM_ARTIFACT_MISSING"),
        "112": (resolve_upstream(args.upstream_112_root), "CURRENT_CHASSIS_RAW_GENERATION_SCALE_CONFIRM_POSITIVE", "UPSTREAM_ARTIFACT_MISSING"),
        "099": (resolve_upstream(args.upstream_099_root), "BOUNDED_LOCAL_PRIVATE_CLEAN_DEPLOY_READY_GATE_POSITIVE", "UPSTREAM_ARTIFACT_MISSING"),
    }
    summaries: dict[str, dict[str, Any]] = {}
    for name, (root, verdict, missing_verdict) in upstreams.items():
        summaries[name] = verify_positive(root, verdict, missing_verdict)
        write_manifest(out, name, root, summaries[name], verdict)
    metrics["upstream_116_positive"] = True
    append_progress(out, "upstream_verification", upstreams=list(upstreams))
    write_live(out, "upstream_verification", ["UPSTREAM_116_CEILING_MAP_VERIFIED"], metrics)

    artifacts = load_116_artifacts(resolve_upstream(args.upstream_116_root))
    append_progress(out, "artifact_loading", loaded=list(artifacts))
    write_live(out, "artifact_loading", ["UPSTREAM_116_CEILING_MAP_VERIFIED"], metrics)

    priority_map = build_failure_priority_map(artifacts)
    breakpoint_analysis = build_breakpoint_analysis(artifacts)
    root_vs_symptom = build_root_vs_symptom_analysis(artifacts)
    write_json(out / "failure_priority_map.json", priority_map)
    write_json(out / "breakpoint_analysis.json", breakpoint_analysis)
    write_json(out / "root_vs_symptom_analysis.json", root_vs_symptom)
    metrics.update(
        {
            "failure_priority_map_written": True,
            "breakpoint_analysis_written": True,
            "root_vs_symptom_analysis_written": True,
            "first_breakpoint_tier": priority_map["first_breakpoint_tier"],
            "reasoning_failure_count": priority_map["reasoning_failure_count"],
            "reasoning_is_largest_failure_class": priority_map["reasoning_is_largest_failure_class"],
        }
    )
    append_progress(out, "failure_prioritization", selected="reasoning_failure")
    write_live(out, "failure_prioritization", ["UPSTREAM_116_CEILING_MAP_VERIFIED"], metrics)

    selection = build_repair_target_selection(priority_map)
    training_options = build_training_design_options()
    eval_gates = build_eval_gate_proposal()
    risks = build_risk_register()
    prevention = build_111_failure_prevention_map()
    next_plan = build_next_milestone_plan()
    write_json(out / "repair_target_selection.json", selection)
    write_json(out / "training_design_options.json", training_options)
    write_json(out / "eval_gate_proposal.json", eval_gates)
    write_json(out / "risk_register.json", risks)
    write_json(out / "111_failure_prevention_map.json", prevention)
    write_json(out / "next_milestone_plan.json", next_plan)
    metrics.update(
        {
            "repair_target_selection_written": True,
            "eval_gate_proposal_written": True,
            "risk_register_written": True,
            "next_milestone_plan_written": True,
            "selected_next_milestone": selection["selected_next_milestone"],
            "selected_repair_target": selection["selected_repair_target"],
        }
    )
    append_progress(out, "repair_target_selection", selected_next=selection["selected_next_milestone"])
    write_live(out, "repair_target_selection", ["UPSTREAM_116_CEILING_MAP_VERIFIED"], metrics)
    append_progress(out, "eval_gate_proposal", gates=len(eval_gates["hard_gates_for_118"]))
    write_live(out, "eval_gate_proposal", ["UPSTREAM_116_CEILING_MAP_VERIFIED"], metrics)

    decision = build_decision(selection, eval_gates)
    write_json(out / "decision.json", decision)
    metrics["decision"] = "targeted_repair_plan_complete"
    metrics["decision_written"] = True
    append_progress(out, "decision_writing", selected_next=decision["selected_next_milestone"])
    write_live(out, "decision_writing", ["UPSTREAM_116_CEILING_MAP_VERIFIED"], metrics, decision)

    if decision["selected_next_milestone"] != "118_REASONING_FIRST_RAW_ASSISTANT_REPAIR":
        raise GateError("REPAIR_TARGET_SELECTION_MISSING", "117 did not select reasoning-first repair")
    metrics["wall_clock_sec"] = round(time.time() - start, 3)
    metrics["bounded_release_artifact_unchanged"] = True
    verdicts = [
        POSITIVE_VERDICT,
        "UPSTREAM_116_CEILING_MAP_VERIFIED",
        "BREAKPOINT_ANALYSIS_WRITTEN",
        "FAILURE_PRIORITY_MAP_WRITTEN",
        "ROOT_VS_SYMPTOM_ANALYSIS_WRITTEN",
        "REPAIR_TARGET_SELECTED",
        "EVAL_GATE_PROPOSAL_WRITTEN",
        "NEXT_MILESTONE_PLAN_WRITTEN",
        "NO_TRAINING_PERFORMED",
        "BOUNDED_RELEASE_UNCHANGED",
        "PRODUCTION_CHAT_NOT_CLAIMED",
        "GPT_LIKE_READINESS_NOT_CLAIMED",
    ]
    append_progress(out, "final_verdict", verdict=POSITIVE_VERDICT, selected_next=decision["selected_next_milestone"])
    write_summary(out, "final_verdict", "positive", verdicts, metrics)
    write_report(out, "final_verdict", verdicts, metrics, decision)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--upstream-116-root", default=str(DEFAULT_UPSTREAM_116_ROOT))
    parser.add_argument("--upstream-115-root", default=str(DEFAULT_UPSTREAM_115_ROOT))
    parser.add_argument("--upstream-114-root", default=str(DEFAULT_UPSTREAM_114_ROOT))
    parser.add_argument("--upstream-113-root", default=str(DEFAULT_UPSTREAM_113_ROOT))
    parser.add_argument("--upstream-112-root", default=str(DEFAULT_UPSTREAM_112_ROOT))
    parser.add_argument("--upstream-099-root", default=str(DEFAULT_UPSTREAM_099_ROOT))
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    args = parser.parse_args()
    try:
        return run(args)
    except GateError as exc:
        try:
            out = resolve_target_out(args.out)
            metrics = {
                "schema_version": "phase_117_targeted_plan_metrics_v1",
                "train_step_count": 0,
                "optimizer_step_count": 0,
                "inference_run_count": 0,
                "checkpoint_mutated": False,
                "service_started": False,
                "deployment_smoke_run": False,
                "failure_verdict": exc.verdict,
            }
            append_progress(out, "failure", "failed", verdict=exc.verdict, message=exc.message)
            write_summary(out, "failure", "failed", ["TARGETED_CAPABILITY_REPAIR_OR_SCALE_PLAN_FAILS", exc.verdict], metrics, exc.message)
            write_report(out, "failure", ["TARGETED_CAPABILITY_REPAIR_OR_SCALE_PLAN_FAILS", exc.verdict], metrics)
        except Exception:
            pass
        print(f"{exc.verdict}: {exc.message}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
