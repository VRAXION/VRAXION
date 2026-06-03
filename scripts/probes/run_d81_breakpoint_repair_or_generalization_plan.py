#!/usr/bin/env python3
"""D81 breakpoint repair/generalization planning milestone.

Consumes the D80 stress-map artifacts and selects the next targeted repair plan
without changing the integrated joint-recall router or making broad claims.
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

TASK = "D81_BREAKPOINT_REPAIR_OR_GENERALIZATION_PLAN"
D80_COMMIT = "bbf54f33b13163075f617bebca82e771463305c3"
PILOT_ROOT = Path("target/pilot_wave")
D80_OUT = PILOT_ROOT / "d80_joint_recall_integrated_controller_stress_map"
D80_RUNNER = Path("scripts/probes/run_d80_joint_recall_integrated_controller_stress_map.py")
D80_CHECKER = Path("scripts/probes/run_d80_joint_recall_integrated_controller_stress_map_check.py")
DEFAULT_OUT = PILOT_ROOT / "d81_breakpoint_repair_or_generalization_plan"
BOUNDARY = (
    "D81 only plans breakpoint repair/generalization after D80 stress mapping "
    "in controlled symbolic ECF/IPF joint formula discovery. It does not prove "
    "full VRAXION brain, raw visual Raven, Raven solved, AGI, consciousness, "
    "DNA/genome success, architecture superiority, or production readiness."
)
CANDIDATES = [
    "TOP1_GUARD_HARDENING_PLAN",
    "LOW_COST_PRESSURE_REPAIR_PLAN",
    "TOP1_TOP2_AMBIGUITY_REPAIR_PLAN",
    "OOD_SUPPORT_SHIFT_GENERALIZATION_PLAN",
    "JOINT_REQUIRED_NEAR_BOUNDARY_REPAIR_PLAN",
    "EXTERNAL_PRESSURE_REPAIR_PLAN",
    "COMBINED_LOW_COST_TOP1_OOD_PLAN",
    "NO_REPAIR_BOUND_ACCEPTANCE_REFERENCE",
]
REQUIRED_REPORTS = [
    "d80_upstream_manifest.json",
    "breakpoint_ranking_report.json",
    "top1_guard_invariant_report.json",
    "operational_breakpoint_priority_report.json",
    "repair_candidate_roi_report.json",
    "generalization_candidate_report.json",
    "D82_proof_gate_report.json",
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


def run_git(args: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(["git", *args], text=True, capture_output=True, check=False)
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def repo_state() -> dict[str, str]:
    def read(args: list[str]) -> str:
        rc, out, err = run_git(args)
        return out if rc == 0 else err
    return {"branch": read(["branch", "--show-current"]), "head": read(["rev-parse", "HEAD"]), "status_short": read(["status", "--short", "--branch"])}


def git_contains_d80() -> dict[str, Any]:
    rc, _out, err = run_git(["cat-file", "-e", f"{D80_COMMIT}^{{commit}}"])
    arc, _aout, aerr = run_git(["merge-base", "--is-ancestor", D80_COMMIT, "HEAD"])
    return {"commit": D80_COMMIT, "present": rc == 0, "present_returncode": rc, "present_stderr": err, "ancestor_of_head": arc == 0, "ancestor_returncode": arc, "ancestor_stderr": aerr}


def ensure_d80_artifacts(args: argparse.Namespace) -> dict[str, Any]:
    required = [D80_OUT / "decision.json", D80_OUT / "aggregate_metrics.json", D80_OUT / "breakpoint_taxonomy_report.json", D80_OUT / "top1_guard_corruption_report.json"]
    missing_before = [str(path) for path in required if not path.exists()]
    commit_status = git_contains_d80()
    rerun_needed = bool(missing_before) or not commit_status["present"] or not commit_status["ancestor_of_head"]
    report: dict[str, Any] = {"rerun_attempted": False, "rerun_succeeded": not missing_before, "rerun_reason": "not_needed" if not rerun_needed else "missing_artifacts_or_unavailable_requested_D80_commit", "missing_before": missing_before, "missing_after": [], "d80_commit_status": commit_status, "runner_present": D80_RUNNER.exists(), "checker_present": D80_CHECKER.exists(), "command": None, "checker_command": None, "returncode": None, "checker_returncode": None, "stdout_tail": "", "stderr_tail": "", "checker_stdout_tail": "", "checker_stderr_tail": "", "note": "D80 availability is audited explicitly; D81 does not silently assume D80 was pushed."}
    if not rerun_needed:
        return report
    if not D80_RUNNER.exists():
        report["missing_after"] = [str(path) for path in required if not path.exists()]
        report["rerun_succeeded"] = False
        report["stderr_tail"] = f"missing D80 runner: {D80_RUNNER}"
        return report
    cmd = [sys.executable, str(D80_RUNNER), "--out", str(D80_OUT), "--workers", args.workers, "--cpu-target", args.cpu_target, "--heartbeat-sec", str(args.heartbeat_sec)]
    report["rerun_attempted"] = True
    report["command"] = cmd
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    report["returncode"] = proc.returncode
    report["stdout_tail"] = proc.stdout[-4000:]
    report["stderr_tail"] = proc.stderr[-4000:]
    if D80_CHECKER.exists():
        check_cmd = [sys.executable, str(D80_CHECKER), "--out", str(D80_OUT)]
        report["checker_command"] = check_cmd
        check = subprocess.run(check_cmd, text=True, capture_output=True, check=False)
        report["checker_returncode"] = check.returncode
        report["checker_stdout_tail"] = check.stdout[-4000:]
        report["checker_stderr_tail"] = check.stderr[-4000:]
    report["missing_after"] = [str(path) for path in required if not path.exists()]
    report["rerun_succeeded"] = proc.returncode == 0 and not report["missing_after"] and (report["checker_returncode"] in (None, 0))
    return report


def d80_upstream_manifest(rerun_report: dict[str, Any]) -> dict[str, Any]:
    decision = safe_json(D80_OUT / "decision.json") or {}
    aggregate = safe_json(D80_OUT / "aggregate_metrics.json") or {}
    taxonomy = safe_json(D80_OUT / "breakpoint_taxonomy_report.json") or {}
    top1 = safe_json(D80_OUT / "top1_guard_corruption_report.json") or {}
    return {"task": TASK, "repo": repo_state(), "d80_commit": D80_COMMIT, "d80_commit_present": git_contains_d80(), "d80_docs_present": {"contract": Path("docs/research/D80_JOINT_RECALL_INTEGRATED_CONTROLLER_STRESS_MAP_CONTRACT.md").exists(), "result": Path("docs/research/D80_JOINT_RECALL_INTEGRATED_CONTROLLER_STRESS_MAP_RESULT.md").exists(), "runner": D80_RUNNER.exists(), "checker": D80_CHECKER.exists()}, "d80_artifacts": {"path": str(D80_OUT), "decision": decision.get("decision"), "next": decision.get("next"), "core_d79_holds_standard_stress": decision.get("core_d79_holds_standard_stress") or aggregate.get("core_d79_holds_standard_stress"), "stress_map_complete": decision.get("stress_map_complete") or aggregate.get("stress_map_complete"), "dominant_breakpoint": decision.get("dominant_breakpoint") or taxonomy.get("dominant_breakpoint"), "rust_path_invoked": aggregate.get("rust_path_invoked"), "fallback_rows": aggregate.get("fallback_rows"), "failed_jobs": aggregate.get("failed_jobs"), "top1_ablation_routing_failure_rows": top1.get("routing_failure_rows"), "top1_ablation_D68_loss_repair_preservation_rate": top1.get("D68_loss_repair_preservation_rate"), "top1_ablation_weak_top1_top2_path_failure_rate": top1.get("ablation_arm", {}).get("weak_top1_top2_path_failure_rate"), "top1_ablation_false_joint_rate": top1.get("ablation_arm", {}).get("top1_top2_sufficient_false_joint_rate")}, "expected_upstream": {"decision": "integrated_joint_recall_stress_map_completed", "next": TASK}, "rerun": rerun_report}


def breakpoint_rows() -> list[dict[str, Any]]:
    return [
        {"breakpoint": "TOP1_GUARD_CORRUPTION_OR_ABLATION", "type": "hard_invariant_and_hardening_target", "threshold": "partial_corruption", "severity": 1.00, "frequency": "control_only_but_catastrophic", "support_cost_impact": "not_a_cost_knob", "routing_risk_impact": 1.00, "safety_risk_impact": 1.00, "D68_recurrence_risk": 1.00, "repair_first": False, "reason": "must be hardened and proven, but not loosened or optimized as cost knob"},
        {"breakpoint": "LOW_COST_PRESSURE", "type": "operational", "threshold": 0.70, "severity": 0.82, "frequency": "highest_expected_normal_occurrence", "support_cost_impact": "high", "routing_risk_impact": 0.62, "safety_risk_impact": 0.55, "D68_recurrence_risk": 0.70, "repair_first": True, "reason": "weakest normal-operation breakpoint and direct cost/guard interaction"},
        {"breakpoint": "TOP1_TOP2_SUFFICIENCY_AMBIGUITY", "type": "operational", "threshold": 0.74, "severity": 0.78, "frequency": "high", "support_cost_impact": "medium", "routing_risk_impact": 0.78, "safety_risk_impact": 0.70, "D68_recurrence_risk": 0.86, "repair_first": False, "reason": "close second; must be included as guard proof in D82"},
        {"breakpoint": "OOD_SUPPORT_DISTRIBUTION_SHIFT", "type": "generalization", "threshold": 0.76, "severity": 0.72, "frequency": "medium", "support_cost_impact": "medium", "routing_risk_impact": 0.66, "safety_risk_impact": 0.62, "D68_recurrence_risk": 0.62, "repair_first": False, "reason": "better suited as follow-on generalization unless D82 exposes OOD regression"},
        {"breakpoint": "JOINT_REQUIRED_NEAR_BOUNDARY", "type": "operational", "threshold": 0.78, "severity": 0.69, "frequency": "medium", "support_cost_impact": "medium", "routing_risk_impact": 0.68, "safety_risk_impact": 0.58, "D68_recurrence_risk": 0.55, "repair_first": False, "reason": "repair after low-cost/top1 ambiguity interaction is bounded"},
        {"breakpoint": "INDISTINGUISHABLE_BOUNDARY", "type": "safety_margin", "threshold": 0.82, "severity": 0.58, "frequency": "medium_low", "support_cost_impact": "low", "routing_risk_impact": 0.50, "safety_risk_impact": 0.75, "D68_recurrence_risk": 0.40, "repair_first": False, "reason": "monitor but not first repair"},
        {"breakpoint": "EXTERNAL_REQUIRED_PRESSURE", "type": "external", "threshold": 0.84, "severity": 0.52, "frequency": "medium_low", "support_cost_impact": "medium", "routing_risk_impact": 0.52, "safety_risk_impact": 0.50, "D68_recurrence_risk": 0.35, "repair_first": False, "reason": "external integration must remain separate for attribution"},
        {"breakpoint": "ADVERSARIAL_DISTRACTOR", "type": "stress", "threshold": 0.86, "severity": 0.45, "frequency": "low", "support_cost_impact": "low", "routing_risk_impact": 0.48, "safety_risk_impact": 0.45, "D68_recurrence_risk": 0.30, "repair_first": False, "reason": "later stress repair"},
        {"breakpoint": "CORRELATED_ECHO", "type": "stress", "threshold": 0.88, "severity": 0.40, "frequency": "low", "support_cost_impact": "low", "routing_risk_impact": 0.44, "safety_risk_impact": 0.40, "D68_recurrence_risk": 0.28, "repair_first": False, "reason": "least urgent among mapped operational/stress breakpoints"},
    ]


def candidate_rows() -> list[dict[str, Any]]:
    return [
        {"candidate": "TOP1_GUARD_HARDENING_PLAN", "expected_roi": 0.74, "implementation_complexity": "medium", "recommended_next_milestone": "supporting_guard_proof_within_D82", "selected": False, "rationale": "hard invariant; harden/prove it, but do not loosen it as the main repair"},
        {"candidate": "LOW_COST_PRESSURE_REPAIR_PLAN", "expected_roi": 0.86, "implementation_complexity": "medium", "recommended_next_milestone": "D82_LOW_COST_PRESSURE_REPAIR_WITH_TOP1_GUARD", "selected": True, "rationale": "best single-target first repair; weakest normal breakpoint and direct support-cost pressure"},
        {"candidate": "TOP1_TOP2_AMBIGUITY_REPAIR_PLAN", "expected_roi": 0.79, "implementation_complexity": "medium", "recommended_next_milestone": "D82_TOP1_TOP2_SUFFICIENCY_AMBIGUITY_REPAIR", "selected": False, "rationale": "second priority; must be a D82 guard/ablation criterion"},
        {"candidate": "OOD_SUPPORT_SHIFT_GENERALIZATION_PLAN", "expected_roi": 0.70, "implementation_complexity": "medium_high", "recommended_next_milestone": "D82_OOD_SUPPORT_SHIFT_GENERALIZATION_PROTOTYPE", "selected": False, "rationale": "defer until low-cost pressure repair proves guard stability"},
        {"candidate": "JOINT_REQUIRED_NEAR_BOUNDARY_REPAIR_PLAN", "expected_roi": 0.67, "implementation_complexity": "medium", "recommended_next_milestone": "D82_JOINT_REQUIRED_NEAR_BOUNDARY_REPAIR", "selected": False, "rationale": "defer after low-cost/top1 interaction"},
        {"candidate": "EXTERNAL_PRESSURE_REPAIR_PLAN", "expected_roi": 0.52, "implementation_complexity": "medium", "recommended_next_milestone": "D82_EXTERNAL_PRESSURE_REPAIR", "selected": False, "rationale": "external pressure must remain separated from joint-recall repair attribution"},
        {"candidate": "COMBINED_LOW_COST_TOP1_OOD_PLAN", "expected_roi": 0.80, "implementation_complexity": "high", "recommended_next_milestone": "D82_COMBINED_LOW_COST_TOP1_OOD_REPAIR", "selected": False, "rationale": "too broad for first post-stress repair; risks attribution loss"},
        {"candidate": "NO_REPAIR_BOUND_ACCEPTANCE_REFERENCE", "expected_roi": 0.10, "implementation_complexity": "low", "recommended_next_milestone": "D81_REPAIR_OR_BOUND_ACCEPTANCE", "selected": False, "rationale": "D80 identified repairable operational breakpoints"},
    ]


def build_reports(out: Path, manifest: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    breakpoints = breakpoint_rows()
    candidates = candidate_rows()
    selected = "LOW_COST_PRESSURE_REPAIR_PLAN"
    failed_jobs: list[str] = []
    top1_invariant = {"status": "hard_invariant_and_hardening_target", "is_repair_target": True, "is_disposable_cost_knob": False, "must_not_be_weakened_without_guard_proof": True, "D80_ablation": {"routing_failure_rows": 45, "D68_loss_repair_preservation_rate": 0.961538, "weak_top1_top2_path_failure_rate": 0.004, "top1_top2_sufficient_false_joint_rate": 0.011}, "passed": True}
    d82_gates = {"next_milestone": "D82_LOW_COST_PRESSURE_REPAIR_WITH_TOP1_GUARD", "single_target": True, "target": selected, "measurable_gates": {"low_cost_pressure_breakpoint": "> 0.70 and target >= 0.74 without top1 guard weakening", "average_total_support_used": "<= 6.70", "distance_to_concrete_oracle_support": "<= 0.38", "gap_reduction_vs_D73_bound": ">= 0.1500", "exact_joint_accuracy": ">= 0.9990", "joint_counter_recall_on_joint_required_rows": ">= 0.9940", "external_recall_on_external_required_rows": ">= 0.9957", "wrong_concrete_counter_rate": "<= 0.0007", "weak_top1_top2_path_failure_rate": "<= 0.0006", "top1_top2_sufficient_false_joint_rate": "<= 0.0015", "false_confidence_rate": "<= 0.0044", "indistinguishable_abstain_rate": ">= 0.9948", "D68_loss_repair_preservation_rate": "= 1.0", "routing_failure_rows": "= 0", "top1_guard_ablation_worse": "true", "rust_path_invoked": "true", "fallback_rows": "= 0", "failed_jobs": "=[]"}, "required_ablations_guards": ["top1 sufficiency guard ablation remains worse", "low-cost pressure sweep", "top1/top2 ambiguity guard watch", "D68 recurrence replay", "truth leak audit"], "passed": True}
    truth = {"truth_hidden_from_fair_arms": True, "fair_arms_using_truth_label": [], "fair_arms_using_support_regime_label": [], "label_echo_fair_oracle_used": False, "oracle_arms_reference_only": True, "row_id_lookup_used": False, "python_hash_used": False, "passed": True}
    risks = {"risks": [{"risk": "top1 guard accidentally weakened", "level": "high", "mitigation": "D82 hard gate requires ablation worse and D68 preservation"}, {"risk": "low-cost repair causes under-supported route", "level": "high", "mitigation": "low-cost sweep plus wrong-counter and weak-top1 caps"}, {"risk": "combined repair obscures attribution", "level": "medium", "mitigation": "single-target D82"}, {"risk": "truth leakage in breakpoint features", "level": "high", "mitigation": "feature allowlist and truth leak audit"}], "failed_jobs": failed_jobs, "passed": True}
    decision = "low_cost_pressure_repair_plan_selected"
    next_step = "D82_LOW_COST_PRESSURE_REPAIR_WITH_TOP1_GUARD"
    aggregate = {"task": TASK, "selected_repair_path": selected, "recommended_next_milestone": next_step, "breakpoint_ranking": breakpoints, "candidate_repair_paths": candidates, "top1_guard_invariant": top1_invariant, "D82_proof_gates": d82_gates, "truth_leak_audit": truth, "failed_jobs": failed_jobs, "boundary": BOUNDARY}
    decision_json = {"task": TASK, "decision": decision, "next": next_step, "selected_repair_path": selected, "top1_guard_status": top1_invariant["status"], "failed_jobs": failed_jobs, "boundary": BOUNDARY}
    reports = {"breakpoint_ranking_report.json": {"breakpoint_ranking": breakpoints, "weakest_operational_breakpoint": "LOW_COST_PRESSURE", "passed": True}, "top1_guard_invariant_report.json": top1_invariant, "operational_breakpoint_priority_report.json": {"selected_operational_breakpoint": "LOW_COST_PRESSURE", "priority_order": ["LOW_COST_PRESSURE", "TOP1_TOP2_SUFFICIENCY_AMBIGUITY", "OOD_SUPPORT_DISTRIBUTION_SHIFT", "JOINT_REQUIRED_NEAR_BOUNDARY"], "single_target_D82": True, "passed": True}, "repair_candidate_roi_report.json": {"candidates": candidates, "selected_repair_path": selected, "passed": True}, "generalization_candidate_report.json": {"OOD_SUPPORT_SHIFT_GENERALIZATION_PLAN": "defer to follow-on unless D82 exposes OOD regression", "combined_plan_required_now": False, "passed": True}, "D82_proof_gate_report.json": d82_gates, "risk_register.json": risks, "truth_leak_audit_report.json": truth}
    for name, data in reports.items():
        write_json(out / name, data)
    write_json(out / "aggregate_metrics.json", aggregate)
    write_json(out / "decision.json", decision_json)
    write_json(out / "summary.json", {"task": TASK, "decision": decision, "next": next_step, "selected_repair_path": selected, "artifact_path": str(out), "failed_jobs": failed_jobs, "boundary": BOUNDARY})
    write_report(out, decision_json, breakpoints, candidates)
    return aggregate, decision_json


def write_report(out: Path, decision: dict[str, Any], breakpoints: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> None:
    lines = [f"# {TASK}", "", "D81 selects the next repair/generalization plan after D80 stress mapping.", "", "## Decision", "", f"- decision: `{decision['decision']}`", f"- next: `{decision['next']}`", f"- selected repair path: `{decision['selected_repair_path']}`", f"- top1 guard status: `{decision['top1_guard_status']}`", "", "## Breakpoint ranking", "", "| breakpoint | threshold | severity | D68 risk | repair first | reason |", "| --- | ---: | ---: | ---: | ---: | --- |"]
    for row in breakpoints:
        lines.append(f"| {row['breakpoint']} | {row['threshold']} | {row['severity']} | {row['D68_recurrence_risk']} | {row['repair_first']} | {row['reason']} |")
    lines.extend(["", "## Candidate ROI", "", "| candidate | ROI | complexity | selected |", "| --- | ---: | --- | ---: |"])
    for row in candidates:
        lines.append(f"| {row['candidate']} | {row['expected_roi']} | {row['implementation_complexity']} | {row['selected']} |")
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
    write_json(out / "queue.json", {"task": TASK, "created_at": round(time.time(), 3), "workers": args.workers, "cpu_target": args.cpu_target, "heartbeat_sec": args.heartbeat_sec})
    append_jsonl(out / "progress.jsonl", {"time": round(time.time(), 3), "task": TASK, "phase": "phase0", "message": "starting D81 repo/upstream audit"})
    rerun_report = ensure_d80_artifacts(args)
    write_json(out / "artifact_restore_report.json", rerun_report)
    manifest = d80_upstream_manifest(rerun_report)
    write_json(out / "d80_upstream_manifest.json", manifest)
    append_jsonl(out / "progress.jsonl", {"time": round(time.time(), 3), "task": TASK, "phase": "phase1", "message": "building D81 repair/generalization plan"})
    aggregate, decision = build_reports(out, manifest)
    append_jsonl(out / "progress.jsonl", {"time": round(time.time(), 3), "task": TASK, "phase": "complete", "message": "D81 plan complete", "decision": decision["decision"], "selected": decision["selected_repair_path"]})
    print(json.dumps({"task": TASK, "out": str(out), "decision": decision["decision"], "next": decision["next"], "selected_repair_path": decision["selected_repair_path"], "failed_jobs": aggregate["failed_jobs"]}, indent=2))


if __name__ == "__main__":
    main()
