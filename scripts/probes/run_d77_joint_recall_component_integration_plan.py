#!/usr/bin/env python3
"""D77 joint-recall component integration plan.

D77 is a planning milestone. It consumes the D76 scale-confirmation artifacts and
selects a concrete integration surface for the scale-confirmed
JOINT_RECALL_COMPONENT_COST_AWARE component in the controlled symbolic ECF/IPF
support-routing stack. It does not claim a new broad architecture mechanism and
it does not invent new empirical metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

TASK = "D77_JOINT_RECALL_COMPONENT_INTEGRATION_PLAN"
D76_COMMIT = "b0a5f3986441ef566ed83e74163b5e34d212bed2"
PILOT_ROOT = Path("target/pilot_wave")
D76_OUT = PILOT_ROOT / "d76_joint_recall_component_scale_confirm"
DEFAULT_OUT = PILOT_ROOT / "d77_joint_recall_component_integration_plan"
D76_RUNNER = Path("scripts/probes/run_d76_joint_recall_component_scale_confirm.py")
D76_CHECKER = Path("scripts/probes/run_d76_joint_recall_component_scale_confirm_check.py")

BOUNDARY = (
    "D77 only plans integration of the scale-confirmed joint-recall component "
    "in controlled symbolic ECF/IPF joint formula discovery. It does not prove "
    "full VRAXION brain, raw visual Raven, Raven solved, AGI, consciousness, "
    "DNA/genome success, architecture superiority, or production readiness."
)

CANDIDATES = [
    "PRE_GATE_JOINT_RECALL_SCORER",
    "POLICY_GATE_JOINT_RECALL_FEATURE",
    "COUNTER_ACTION_ROUTER_JOINT_RECALL_MODULE",
    "POSTCHECK_JOINT_RECALL_ESCALATION",
    "HYBRID_JOINT_RECALL_AND_EXTERNAL_ROUTING",
    "JOINT_RECALL_AS_RUST_SPARSE_DIAGNOSTIC_COMPONENT",
]

REQUIRED_REPORTS = [
    "d76_upstream_manifest.json",
    "integration_target_selection_report.json",
    "component_interface_report.json",
    "required_input_feature_report.json",
    "action_influence_report.json",
    "D68_regression_prevention_report.json",
    "Rust_sparse_integration_surface_report.json",
    "D78_proof_gate_report.json",
    "risk_register.json",
    "truth_leak_audit_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return load_json(path)
    except json.JSONDecodeError:
        return {"decode_error": True, "path": str(path)}


def append_jsonl(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(data, sort_keys=True) + "\n")


def append_progress(out: Path, phase: str, message: str, **extra: Any) -> None:
    append_jsonl(out / "progress.jsonl", {"time": round(time.time(), 3), "task": TASK, "phase": phase, "message": message, **extra})


def run_git(args: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(["git", *args], text=True, capture_output=True, check=False)
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def repo_state() -> dict[str, str]:
    def read(args: list[str]) -> str:
        rc, out, err = run_git(args)
        return out if rc == 0 else err

    return {
        "branch": read(["branch", "--show-current"]),
        "head": read(["rev-parse", "HEAD"]),
        "status_short": read(["status", "--short", "--branch"]),
    }


def git_contains_d76() -> dict[str, Any]:
    rc, _out, err = run_git(["cat-file", "-e", f"{D76_COMMIT}^{{commit}}"])
    arc, _aout, aerr = run_git(["merge-base", "--is-ancestor", D76_COMMIT, "HEAD"])
    return {
        "commit": D76_COMMIT,
        "present": rc == 0,
        "present_returncode": rc,
        "present_stderr": err,
        "ancestor_of_head": arc == 0,
        "ancestor_returncode": arc,
        "ancestor_stderr": aerr,
    }


def ensure_d76_artifacts(args: argparse.Namespace) -> dict[str, Any]:
    required = [
        D76_OUT / "decision.json",
        D76_OUT / "aggregate_metrics.json",
        D76_OUT / "d75_upstream_manifest.json",
        D76_OUT / "truth_leak_audit_report.json",
        D76_OUT / "rust_invocation_report.json",
    ]
    missing_before = [str(path) for path in required if not path.exists()]
    commit_status = git_contains_d76()
    rerun_needed = bool(missing_before) or not commit_status["present"] or not commit_status["ancestor_of_head"]
    report: dict[str, Any] = {
        "rerun_attempted": False,
        "rerun_succeeded": not missing_before,
        "rerun_reason": "not_needed" if not rerun_needed else "missing_artifacts_or_unavailable_requested_D76_commit",
        "missing_before": missing_before,
        "missing_after": [],
        "d76_commit_status": commit_status,
        "runner_present": D76_RUNNER.exists(),
        "checker_present": D76_CHECKER.exists(),
        "command": None,
        "checker_command": None,
        "returncode": None,
        "checker_returncode": None,
        "stdout_tail": "",
        "stderr_tail": "",
        "checker_stdout_tail": "",
        "checker_stderr_tail": "",
        "note": "D76 availability is audited explicitly; D77 does not silently assume D76 was pushed.",
    }
    if not rerun_needed:
        return report
    if not D76_RUNNER.exists():
        report["missing_after"] = [str(path) for path in required if not path.exists()]
        report["rerun_succeeded"] = False
        report["stderr_tail"] = f"missing D76 runner: {D76_RUNNER}"
        return report

    cmd = [
        sys.executable,
        str(D76_RUNNER),
        "--out",
        str(D76_OUT),
        "--workers",
        args.workers,
        "--cpu-target",
        args.cpu_target,
        "--heartbeat-sec",
        str(args.heartbeat_sec),
    ]
    report["rerun_attempted"] = True
    report["command"] = cmd
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    report["returncode"] = proc.returncode
    report["stdout_tail"] = proc.stdout[-4000:]
    report["stderr_tail"] = proc.stderr[-4000:]

    if D76_CHECKER.exists():
        check_cmd = [sys.executable, str(D76_CHECKER), "--out", str(D76_OUT)]
        report["checker_command"] = check_cmd
        check = subprocess.run(check_cmd, text=True, capture_output=True, check=False)
        report["checker_returncode"] = check.returncode
        report["checker_stdout_tail"] = check.stdout[-4000:]
        report["checker_stderr_tail"] = check.stderr[-4000:]

    report["missing_after"] = [str(path) for path in required if not path.exists()]
    report["rerun_succeeded"] = proc.returncode == 0 and not report["missing_after"] and (report["checker_returncode"] in (None, 0))
    return report


def d76_upstream_manifest(rerun_report: dict[str, Any]) -> dict[str, Any]:
    decision = safe_json(D76_OUT / "decision.json") or {}
    aggregate = safe_json(D76_OUT / "aggregate_metrics.json") or {}
    best = aggregate.get("best_fair_arm", {}) if isinstance(aggregate, dict) else {}
    return {
        "task": TASK,
        "repo": repo_state(),
        "d76_commit": D76_COMMIT,
        "d76_commit_present": git_contains_d76(),
        "d76_docs_present": {
            "contract": Path("docs/research/D76_JOINT_RECALL_COMPONENT_SCALE_CONFIRM_CONTRACT.md").exists(),
            "result": Path("docs/research/D76_JOINT_RECALL_COMPONENT_SCALE_CONFIRM_RESULT.md").exists(),
            "runner": D76_RUNNER.exists(),
            "checker": D76_CHECKER.exists(),
        },
        "d76_artifacts": {
            "path": str(D76_OUT),
            "decision_present": (D76_OUT / "decision.json").exists(),
            "aggregate_present": (D76_OUT / "aggregate_metrics.json").exists(),
            "decision": decision.get("decision"),
            "next": decision.get("next"),
            "scaled_arm": decision.get("scaled_arm"),
            "best_fair_arm": decision.get("best_fair_arm"),
            "average_total_support_used": best.get("average_total_support_used"),
            "distance_to_concrete_oracle_support": best.get("distance_to_concrete_oracle_support"),
            "gap_reduction_vs_D73_bound": best.get("gap_reduction_vs_D73_bound"),
            "exact_joint_accuracy": best.get("exact_joint_accuracy"),
            "joint_counter_recall_on_joint_required_rows": best.get("joint_counter_recall_on_joint_required_rows"),
            "external_recall_on_external_required_rows": best.get("external_recall_on_external_required_rows"),
            "wrong_concrete_counter_rate": best.get("wrong_concrete_counter_rate"),
            "weak_top1_top2_path_failure_rate": best.get("weak_top1_top2_path_failure_rate"),
            "D68_loss_repair_preservation_rate": best.get("D68_loss_repair_preservation_rate"),
            "routing_failure_rows": best.get("routing_failure_rows"),
            "rust_path_invoked": best.get("rust_path_invoked") or aggregate.get("rust_path_invoked"),
            "fallback_rows": aggregate.get("fallback_rows"),
            "failed_jobs": aggregate.get("failed_jobs"),
        },
        "expected_upstream": {
            "decision": "joint_recall_component_scale_confirmed",
            "next": TASK,
            "scaled_arm": "D75_JOINT_RECALL_COST_AWARE_REPLAY",
        },
        "rerun": rerun_report,
    }


def candidate_assessments() -> dict[str, dict[str, Any]]:
    return {
        "PRE_GATE_JOINT_RECALL_SCORER": {
            "expected_support_effect": "unclear; risks extra early joint scoring before sufficiency is known",
            "expected_oracle_gap_effect": "low_to_medium",
            "implementation_complexity": "medium",
            "D68_regression_risk": "medium",
            "safety_margin_risk": "medium",
            "dependency_on_symbolic_stack": "high",
            "dependency_on_Rust_sparse_path": "medium",
            "integration_ROI_estimate": "medium",
            "selection_status": "rejected",
            "reason": "too early in the controller; can repeat cheap-top1 mistakes if used before sufficiency guards",
        },
        "POLICY_GATE_JOINT_RECALL_FEATURE": {
            "expected_support_effect": "medium; feature can reproduce D76 behavior if isolated",
            "expected_oracle_gap_effect": "medium",
            "implementation_complexity": "medium",
            "D68_regression_risk": "medium",
            "safety_margin_risk": "medium",
            "dependency_on_symbolic_stack": "high",
            "dependency_on_Rust_sparse_path": "medium",
            "integration_ROI_estimate": "medium",
            "selection_status": "rejected",
            "reason": "feature-only integration is less auditable than a bounded action-router module",
        },
        "COUNTER_ACTION_ROUTER_JOINT_RECALL_MODULE": {
            "expected_support_effect": "preserve D76 support profile around 6.65 while limiting unnecessary joint actions",
            "expected_oracle_gap_effect": "preserve D76 gap reduction above 0.1500 versus D73 bound in integrated prototype",
            "implementation_complexity": "medium",
            "D68_regression_risk": "low_with_explicit_sufficiency_guard",
            "safety_margin_risk": "low_to_medium_with_postcheck",
            "dependency_on_symbolic_stack": "high",
            "dependency_on_Rust_sparse_path": "high",
            "integration_ROI_estimate": "highest_clear_ROI: D76 component effect maps directly to counter-action routing",
            "selection_status": "selected",
            "reason": "places the component exactly where D76 showed value: bounded counter/joint action selection after sufficiency evidence and before external escalation",
        },
        "POSTCHECK_JOINT_RECALL_ESCALATION": {
            "expected_support_effect": "limited; late escalation preserves safety but loses D76 cost-aware routing effect",
            "expected_oracle_gap_effect": "low",
            "implementation_complexity": "low",
            "D68_regression_risk": "low",
            "safety_margin_risk": "low",
            "dependency_on_symbolic_stack": "medium",
            "dependency_on_Rust_sparse_path": "medium",
            "integration_ROI_estimate": "low_to_medium",
            "selection_status": "rejected",
            "reason": "too late to plan support cost effectively; better as a D78 fallback/postcheck subgate",
        },
        "HYBRID_JOINT_RECALL_AND_EXTERNAL_ROUTING": {
            "expected_support_effect": "potentially high but confounds joint-recall and external-recall changes",
            "expected_oracle_gap_effect": "medium_to_high_but_not_isolated",
            "implementation_complexity": "high",
            "D68_regression_risk": "medium",
            "safety_margin_risk": "medium",
            "dependency_on_symbolic_stack": "high",
            "dependency_on_Rust_sparse_path": "high",
            "integration_ROI_estimate": "defer_until_joint_component_integrated",
            "selection_status": "deferred",
            "reason": "D76 asks for joint-recall component integration first; external combo would hide component attribution",
        },
        "JOINT_RECALL_AS_RUST_SPARSE_DIAGNOSTIC_COMPONENT": {
            "expected_support_effect": "diagnostic only; does not route actions by itself",
            "expected_oracle_gap_effect": "low_until_connected_to_action_router",
            "implementation_complexity": "medium",
            "D68_regression_risk": "low",
            "safety_margin_risk": "low",
            "dependency_on_symbolic_stack": "medium",
            "dependency_on_Rust_sparse_path": "high",
            "integration_ROI_estimate": "useful as implementation substrate, not sufficient as next milestone alone",
            "selection_status": "supporting_surface",
            "reason": "should provide sparse scoring/telemetry inside the selected router, not replace integrated controller proof",
        },
    }


def build_plan(out: Path, manifest: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    d76 = manifest.get("d76_artifacts", {})
    failed_jobs: list[str] = []
    if d76.get("decision") != "joint_recall_component_scale_confirmed":
        failed_jobs.append("d76_decision_not_confirmed")
    if d76.get("failed_jobs"):
        failed_jobs.append("d76_failed_jobs_present")

    selected = "COUNTER_ACTION_ROUTER_JOINT_RECALL_MODULE"
    assessments = candidate_assessments()
    interface = {
        "component_name": "JointRecallCostAwareCounterRouter",
        "integration_target": selected,
        "location_in_stack": "Rust sparse ECF/IPF controller counter-action router, after top1/top2 sufficiency evaluation and before external-test escalation/postcheck abstain",
        "input_contract": {
            "top1_top2_margin": "signed margin and confidence summary from sparse support scorer",
            "top1_top2_sufficient_flag": "D68 guard input; true rows cannot be forced into joint action without postcheck evidence",
            "joint_required_signal": "symbolic joint-formula ambiguity/support signal, never the truth label",
            "counter_candidate_supports": "sparse support scores for concrete counter candidates",
            "external_required_signal": "external-test-needed diagnostic used only as a separate escalation guard",
            "indistinguishable_signal": "abstain/safety signal for rows without enough discriminating evidence",
            "rust_sparse_trace": "non-label sparse provenance and component telemetry",
        },
        "output_contract": {
            "joint_action_score": "bounded score used by the counter-action router",
            "counter_action": "one of keep_top1, select_concrete_counter, request_joint_counter, request_external_test, abstain_indistinguishable",
            "reason_codes": ["top1_top2_sufficient", "joint_counter_required", "external_test_required", "indistinguishable_abstain", "safety_postcheck"],
            "audit_fields": ["support_used", "counter_support_used", "joint_score", "pre_guard_passed", "postcheck_passed", "rust_path_invoked"],
        },
        "non_goals": ["no truth-label feature", "no label echo oracle", "no full architecture claim", "no raw visual Raven claim"],
    }

    input_features = [
        {"feature": "top1_top2_margin", "source": "Rust sparse support scorer", "truth_leak_guard": "non-label score only", "required_for_D78": True},
        {"feature": "top1_top2_sufficient_flag", "source": "D68 guard", "truth_leak_guard": "computed from fair support scores", "required_for_D78": True},
        {"feature": "joint_required_signal", "source": "symbolic ECF/IPF ambiguity/support diagnostics", "truth_leak_guard": "no regime label; no row-id lookup", "required_for_D78": True},
        {"feature": "counter_candidate_supports", "source": "Rust sparse counter scorer", "truth_leak_guard": "candidate scores only", "required_for_D78": True},
        {"feature": "external_required_signal", "source": "external routing diagnostic", "truth_leak_guard": "separate gate, not joint truth", "required_for_D78": True},
        {"feature": "indistinguishable_signal", "source": "safety/abstain margin watch", "truth_leak_guard": "margin-based only", "required_for_D78": True},
    ]

    action_influence = {
        "influenced_actions": ["select_concrete_counter", "request_joint_counter", "request_external_test", "abstain_indistinguishable"],
        "not_influenced_actions": ["truth_label_assignment", "oracle_reference_arm", "row_id_lookup"],
        "gates": {
            "pre_guard": "top1/top2 sufficiency guard must run first",
            "joint_gate": "joint score can request joint counter only when sufficiency is false or postcheck finds concrete counter ambiguity",
            "external_gate": "external-required rows remain measurable separately and cannot be hidden inside joint recall",
            "postcheck": "selected concrete counter must pass correctness/safety postcheck before support reduction is counted",
            "abstain_gate": "indistinguishable rows must preserve D76 abstain margin",
        },
    }

    d68_guard = {
        "explicit_prevention_required": True,
        "rules": [
            "Never bypass the D68 top1/top2 sufficiency flag with the joint-recall score alone.",
            "If top1/top2 is sufficient, keep the cheap path unless a non-label postcheck detects concrete counter ambiguity.",
            "If the module selects a concrete counter, require selected-counter postcheck before committing support savings.",
            "Track top1_top2_sufficient_false_joint_rate and weak_top1_top2_path_failure_rate as hard D78 gates.",
            "Preserve D68 loss repair rows exactly; any loss preservation rate below 1.0 fails D78.",
        ],
        "D78_guard_metrics": {
            "weak_top1_top2_path_failure_rate": "<= 0.0005 target, hard cap <= 0.0006",
            "top1_top2_sufficient_false_joint_rate": "<= 0.0011 target, hard cap <= 0.0015",
            "D68_loss_repair_preservation_rate": "= 1.0",
            "routing_failure_rows": "= 0",
        },
        "passed": True,
    }

    rust_surface = {
        "module_surface": "Rust sparse ECF/IPF support-routing stack",
        "proposed_files_or_modules": [
            "instnct-core/src/experimental_route_grammar.rs or adjacent route-controller module",
            "instnct-core/src/network/audit.rs for telemetry/audit fields if needed",
            "instnct-core/examples/phase_lane_* integration prototype harness for D78",
        ],
        "required_trait_shape": {
            "input": "JointRecallRouterInput { sparse_support, top1_top2_margin, joint_signal, counter_candidates, external_signal, abstain_signal }",
            "output": "JointRecallRouterDecision { action, joint_score, support_used, reason_codes, audit }",
        },
        "rust_path_invocation_requirement": "D78 must prove rust_path_invoked=true with fallback_rows=0",
        "diagnostic_component_role": "sparse diagnostic scoring is a supporting subcomponent inside the action router, not a separate prerequisite milestone",
    }

    d78_gates = {
        "next_milestone": "D78_JOINT_RECALL_INTEGRATED_CONTROLLER_PROTOTYPE",
        "must_prove": [
            "Integrated Rust sparse controller invokes JointRecallCostAwareCounterRouter on fair arms.",
            "D76 support and oracle-gap benefits are preserved within integration tolerance.",
            "D68 cheap-top1 regression prevention remains explicit and measured.",
            "External-required rows remain separately measured and above recall gate.",
            "Truth labels, support-regime labels, label echo fair oracles, row-id lookup, and Python hash remain unavailable to fair arms.",
        ],
        "measurable_gates": {
            "average_total_support_used": "<= 6.70",
            "distance_to_concrete_oracle_support": "<= 0.38",
            "gap_reduction_vs_D73_bound": ">= 0.1500",
            "exact_joint_accuracy": ">= 0.9990",
            "correlated_echo_accuracy": ">= 0.995",
            "adversarial_distractor_accuracy": ">= 0.995",
            "external_test_required_accuracy": ">= 0.995",
            "joint_counter_recall_on_joint_required_rows": ">= 0.9940",
            "external_recall_on_external_required_rows": ">= 0.9957",
            "wrong_concrete_counter_rate": "<= 0.0007",
            "weak_top1_top2_path_failure_rate": "<= 0.0006",
            "top1_top2_sufficient_false_joint_rate": "<= 0.0015",
            "false_confidence_rate": "<= 0.0044",
            "indistinguishable_abstain_rate": ">= 0.9948",
            "D68_loss_repair_preservation_rate": "= 1.0",
            "routing_failure_rows": "= 0",
            "rust_path_invoked": "true",
            "fallback_rows": "= 0",
            "failed_jobs": "=[]",
        },
        "passed": True,
    }

    risk_register = {
        "risks": [
            {"risk": "D68 cheap-top1 regression", "level": "high_if_unguarded", "mitigation": "pre-guard and hard D78 metrics"},
            {"risk": "external routing confound", "level": "medium", "mitigation": "separate external gate/report in D78"},
            {"risk": "truth leakage through regime labels", "level": "high", "mitigation": "feature allowlist and truth leak audit"},
            {"risk": "support saving counted before postcheck", "level": "medium", "mitigation": "selected-counter correctness postcheck"},
            {"risk": "Rust sparse telemetry fallback", "level": "medium", "mitigation": "rust_path_invoked=true and fallback_rows=0 hard gate"},
        ],
        "failed_jobs": failed_jobs,
        "passed": not failed_jobs,
    }

    truth_leak = {
        "truth_hidden_from_fair_arms": True,
        "fair_arms_using_truth_label": [],
        "fair_arms_using_support_regime_label": [],
        "label_echo_fair_oracle_used": False,
        "row_id_lookup_used": False,
        "python_hash_used": False,
        "required_truth_leak_guards": ["feature allowlist", "no label/regime fields in router input", "reference-only oracle arms", "audit reason codes only"],
        "passed": True,
    }

    single_best_clear = selected in assessments and assessments[selected]["selection_status"] == "selected" and not failed_jobs
    decision = "joint_recall_integration_plan_selected" if single_best_clear else "joint_recall_integration_plan_not_ready"
    next_step = "D78_JOINT_RECALL_INTEGRATED_CONTROLLER_PROTOTYPE" if single_best_clear else "D77_REPAIR"

    aggregate = {
        "task": TASK,
        "selected_integration_target": selected if single_best_clear else None,
        "candidate_assessments": assessments,
        "planning_metrics_are_estimates_not_empirical_D77_results": True,
        "planning_metrics": {
            "expected_support_effect": assessments[selected]["expected_support_effect"],
            "expected_oracle_gap_effect": assessments[selected]["expected_oracle_gap_effect"],
            "implementation_complexity": assessments[selected]["implementation_complexity"],
            "D68_regression_risk": assessments[selected]["D68_regression_risk"],
            "safety_margin_risk": assessments[selected]["safety_margin_risk"],
            "dependency_on_symbolic_stack": assessments[selected]["dependency_on_symbolic_stack"],
            "dependency_on_Rust_sparse_path": assessments[selected]["dependency_on_Rust_sparse_path"],
            "required_truth_leak_guards": truth_leak["required_truth_leak_guards"],
            "D78_measurable_gates": d78_gates["measurable_gates"],
            "integration_ROI_estimate": assessments[selected]["integration_ROI_estimate"],
        },
        "component_interface": interface,
        "D68_regression_prevention": d68_guard,
        "D78_proof_gates": d78_gates,
        "truth_leak_audit": truth_leak,
        "failed_jobs": failed_jobs,
        "boundary": BOUNDARY,
    }

    decision_json = {
        "task": TASK,
        "decision": decision,
        "next": next_step,
        "selected_integration_target": selected if single_best_clear else None,
        "failed_jobs": failed_jobs,
        "boundary": BOUNDARY,
    }

    reports = {
        "integration_target_selection_report.json": {"selected_integration_target": selected if single_best_clear else None, "candidate_assessments": assessments, "single_best_clear": single_best_clear, "decision_basis": assessments[selected]["reason"], "passed": single_best_clear},
        "component_interface_report.json": {**interface, "passed": True},
        "required_input_feature_report.json": {"features": input_features, "truth_leak_guarded": True, "passed": True},
        "action_influence_report.json": {**action_influence, "passed": True},
        "D68_regression_prevention_report.json": d68_guard,
        "Rust_sparse_integration_surface_report.json": {**rust_surface, "passed": True},
        "D78_proof_gate_report.json": d78_gates,
        "risk_register.json": risk_register,
        "truth_leak_audit_report.json": truth_leak,
    }
    for name, data in reports.items():
        write_json(out / name, data)
    write_json(out / "aggregate_metrics.json", aggregate)
    write_json(out / "decision.json", decision_json)
    write_json(out / "summary.json", {"task": TASK, "decision": decision, "next": next_step, "selected_integration_target": selected if single_best_clear else None, "artifact_path": str(out), "failed_jobs": failed_jobs, "boundary": BOUNDARY})
    write_report(out, decision_json, reports, d76)
    return aggregate, decision_json


def write_report(out: Path, decision: dict[str, Any], reports: dict[str, Any], d76: dict[str, Any]) -> None:
    selected = decision.get("selected_integration_target")
    lines = [
        f"# {TASK}",
        "",
        "D77 is a planning milestone for integrating the D76 scale-confirmed joint-recall component.",
        "",
        "## Decision",
        "",
        f"- decision: `{decision['decision']}`",
        f"- next: `{decision['next']}`",
        f"- selected integration target: `{selected}`",
        f"- failed_jobs: `{decision['failed_jobs']}`",
        "",
        "## D76 upstream snapshot",
        "",
        f"- decision: `{d76.get('decision')}`",
        f"- scaled arm: `{d76.get('scaled_arm')}`",
        f"- support: `{d76.get('average_total_support_used')}`",
        f"- oracle distance: `{d76.get('distance_to_concrete_oracle_support')}`",
        f"- gap reduction vs D73: `{d76.get('gap_reduction_vs_D73_bound')}`",
        "",
        "## Candidate selection",
        "",
        "| candidate | status | complexity | D68 risk | Rust dependency | reason |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for candidate, row in reports["integration_target_selection_report.json"]["candidate_assessments"].items():
        lines.append(f"| {candidate} | {row['selection_status']} | {row['implementation_complexity']} | {row['D68_regression_risk']} | {row['dependency_on_Rust_sparse_path']} | {row['reason']} |")
    lines.extend([
        "",
        "## Component interface",
        "",
        f"- component: `{reports['component_interface_report.json']['component_name']}`",
        f"- location: {reports['component_interface_report.json']['location_in_stack']}",
        "",
        "## D78 proof gates",
        "",
    ])
    for key, value in reports["D78_proof_gate_report.json"]["measurable_gates"].items():
        lines.append(f"- `{key}` {value}")
    lines.extend(["", "## Boundary", "", BOUNDARY, ""])
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--workers", default="auto")
    parser.add_argument("--cpu-target", default="50-75")
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    args = parser.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "queue.json", {"task": TASK, "created_at": round(time.time(), 3), "workers": args.workers, "cpu_target": args.cpu_target, "heartbeat_sec": args.heartbeat_sec, "phases": ["repo_upstream_audit", "d76_restore_or_rerun", "integration_plan", "reporting"]})
    append_progress(out, "phase0", "starting D77 repo/upstream audit")
    rerun_report = ensure_d76_artifacts(args)
    write_json(out / "artifact_restore_report.json", rerun_report)
    manifest = d76_upstream_manifest(rerun_report)
    write_json(out / "d76_upstream_manifest.json", manifest)
    append_progress(out, "phase1", "building D77 integration plan")
    aggregate, decision = build_plan(out, manifest)
    append_progress(out, "complete", "D77 integration plan complete", decision=decision["decision"], selected=decision.get("selected_integration_target"))
    print(json.dumps({"task": TASK, "out": str(out), "decision": decision["decision"], "next": decision["next"], "selected_integration_target": decision.get("selected_integration_target"), "d76_rerun_attempted": rerun_report["rerun_attempted"], "failed_jobs": aggregate["failed_jobs"]}, indent=2))


if __name__ == "__main__":
    main()
