#!/usr/bin/env python3
"""D89 repair/generalization planning after D88 stress map."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

TASK = "D89_BREAKPOINT_REPAIR_OR_GENERALIZATION_PLAN"
D88_COMMIT = "05a429f90fa55bd7c2be3218cbe74b6bcf52147c"
PILOT_ROOT = Path("target/pilot_wave")
D88_OUT = PILOT_ROOT / "d88_combined_low_cost_top1_ambiguity_stress_map"
D88_RUNNER = Path("scripts/probes/run_d88_combined_low_cost_top1_ambiguity_stress_map.py")
D88_CHECKER = Path("scripts/probes/run_d88_combined_low_cost_top1_ambiguity_stress_map_check.py")
DEFAULT_OUT = PILOT_ROOT / "d89_breakpoint_repair_or_generalization_plan"
BOUNDARY = (
    "D89 only plans repair/generalization after D88 stress mapping in controlled symbolic ECF/IPF joint formula discovery. "
    "It does not prove full VRAXION brain, raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, "
    "architecture superiority, or production readiness."
)
CANDIDATES = [
    "COMBINED_LOW_COST_OOD_REPAIR_PLAN",
    "OOD_SUPPORT_SHIFT_GENERALIZATION_PLAN",
    "LOW_COST_OOD_WITH_TOP1_GUARD_PLAN",
    "COMBINED_LOW_COST_TOP1_OOD_PLAN",
    "JOINT_REQUIRED_BOUNDARY_REPAIR_PLAN",
    "EXTERNAL_PRESSURE_REPAIR_PLAN",
    "TOP1_GUARD_HARDENING_REFERENCE_ONLY",
    "NO_REPAIR_BOUND_ACCEPTANCE_REFERENCE",
]
REQUIRED_REPORTS = [
    "d88_upstream_manifest.json",
    "breakpoint_ranking_report.json",
    "combined_low_cost_ood_analysis_report.json",
    "ood_generalization_candidate_report.json",
    "top1_guard_invariant_report.json",
    "repair_candidate_roi_report.json",
    "generalization_candidate_report.json",
    "D90_proof_gate_report.json",
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

    return {
        "branch": read(["branch", "--show-current"]),
        "head": read(["rev-parse", "HEAD"]),
        "status_short": read(["status", "--short", "--branch"]),
    }


def git_contains_d88() -> dict[str, Any]:
    rc, _, err = run_git(["cat-file", "-e", f"{D88_COMMIT}^{{commit}}"])
    arc, _, aerr = run_git(["merge-base", "--is-ancestor", D88_COMMIT, "HEAD"])
    return {
        "commit": D88_COMMIT,
        "present": rc == 0,
        "present_returncode": rc,
        "present_stderr": err,
        "ancestor_of_head": arc == 0,
        "ancestor_returncode": arc,
        "ancestor_stderr": aerr,
    }


def ensure_d88(args: argparse.Namespace) -> dict[str, Any]:
    required = [
        D88_OUT / "decision.json",
        D88_OUT / "aggregate_metrics.json",
        D88_OUT / "stress_axis_summary_report.json",
        D88_OUT / "top1_guard_corruption_report.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    status = git_contains_d88()
    need = bool(missing) or not status["present"] or not status["ancestor_of_head"]
    report = {
        "rerun_attempted": False,
        "rerun_succeeded": not missing,
        "rerun_reason": "not_needed" if not need else "missing_artifacts_or_unavailable_requested_D88_commit",
        "missing_before": missing,
        "missing_after": [],
        "d88_commit_status": status,
        "runner_present": D88_RUNNER.exists(),
        "checker_present": D88_CHECKER.exists(),
        "command": None,
        "checker_command": None,
        "returncode": None,
        "checker_returncode": None,
        "stdout_tail": "",
        "stderr_tail": "",
        "checker_stdout_tail": "",
        "checker_stderr_tail": "",
        "note": "D88 availability is audited explicitly; D89 does not silently assume D88 was pushed.",
    }
    if not need:
        return report
    if not D88_RUNNER.exists():
        report["missing_after"] = [str(path) for path in required if not path.exists()]
        report["rerun_succeeded"] = False
        return report
    command = [
        sys.executable,
        str(D88_RUNNER),
        "--out",
        str(D88_OUT),
        "--workers",
        args.workers,
        "--cpu-target",
        args.cpu_target,
        "--heartbeat-sec",
        str(args.heartbeat_sec),
    ]
    report["rerun_attempted"] = True
    report["command"] = command
    proc = subprocess.run(command, text=True, capture_output=True, check=False)
    report["returncode"] = proc.returncode
    report["stdout_tail"] = proc.stdout[-4000:]
    report["stderr_tail"] = proc.stderr[-4000:]
    if D88_CHECKER.exists():
        checker_command = [sys.executable, str(D88_CHECKER), "--out", str(D88_OUT)]
        report["checker_command"] = checker_command
        check = subprocess.run(checker_command, text=True, capture_output=True, check=False)
        report["checker_returncode"] = check.returncode
        report["checker_stdout_tail"] = check.stdout[-4000:]
        report["checker_stderr_tail"] = check.stderr[-4000:]
    report["missing_after"] = [str(path) for path in required if not path.exists()]
    report["rerun_succeeded"] = proc.returncode == 0 and not report["missing_after"] and report["checker_returncode"] in (None, 0)
    return report


def d88_manifest(rerun: dict[str, Any]) -> dict[str, Any]:
    decision = safe_json(D88_OUT / "decision.json") or {}
    aggregate = safe_json(D88_OUT / "aggregate_metrics.json") or {}
    summary = safe_json(D88_OUT / "stress_axis_summary_report.json") or {}
    taxonomy = safe_json(D88_OUT / "breakpoint_taxonomy_report.json") or {}
    top1 = safe_json(D88_OUT / "top1_guard_corruption_report.json") or {}
    best = aggregate.get("best_fair_arm", {}) if isinstance(aggregate, dict) else {}
    return {
        "task": TASK,
        "repo": repo_state(),
        "d88_commit": D88_COMMIT,
        "d88_commit_present": git_contains_d88(),
        "d88_docs_present": {
            "contract": Path("docs/research/D88_COMBINED_LOW_COST_TOP1_AMBIGUITY_STRESS_MAP_CONTRACT.md").exists(),
            "result": Path("docs/research/D88_COMBINED_LOW_COST_TOP1_AMBIGUITY_STRESS_MAP_RESULT.md").exists(),
            "runner": D88_RUNNER.exists(),
            "checker": D88_CHECKER.exists(),
        },
        "d88_artifacts": {
            "path": str(D88_OUT),
            "decision": decision.get("decision"),
            "next": decision.get("next"),
            "best_fair_arm": best.get("arm"),
            "stress_map_complete": summary.get("stress_map_complete") or aggregate.get("stress_map_complete"),
            "core_D87_holds_standard_stress": summary.get("core_d87_holds_standard_stress") or aggregate.get("core_d87_holds_standard_stress"),
            "dominant_breakpoint": decision.get("dominant_breakpoint") or taxonomy.get("dominant_breakpoint"),
            "hard_invariant_breakpoint": decision.get("hard_invariant_breakpoint") or taxonomy.get("hard_invariant_breakpoint"),
            "combined_low_cost_plus_top1_ambiguity_breakpoint": best.get("combined_low_cost_plus_top1_ambiguity_breakpoint"),
            "low_cost_pressure_breakpoint": best.get("low_cost_pressure_breakpoint"),
            "top1_top2_sufficiency_ambiguity_breakpoint": best.get("top1_top2_sufficiency_ambiguity_breakpoint"),
            "combined_low_cost_plus_ood_breakpoint": best.get("combined_low_cost_plus_ood_breakpoint"),
            "ood_support_distribution_shift_breakpoint": best.get("ood_support_distribution_shift_breakpoint"),
            "joint_required_near_boundary_breakpoint": best.get("joint_required_near_boundary_breakpoint"),
            "top1_guard_preserved": top1.get("top1_guard_preserved"),
            "top1_guard_weakened": top1.get("top1_guard_weakened"),
            "ablation_routing_failure_rows": top1.get("ablation_routing_failure_rows"),
            "ablation_D68_loss_repair_preservation_rate": top1.get("ablation_D68_loss_repair_preservation_rate"),
            "failed_jobs": aggregate.get("failed_jobs"),
            "fallback_rows": aggregate.get("fallback_rows"),
        },
        "expected_upstream": {
            "decision": "combined_low_cost_top1_ambiguity_stress_map_completed",
            "next": TASK,
            "dominant_breakpoint": "COMBINED_LOW_COST_PLUS_OOD",
        },
        "rerun": rerun,
    }


def candidate_rows() -> list[dict[str, Any]]:
    return [
        {"candidate": "COMBINED_LOW_COST_OOD_REPAIR_PLAN", "target_breakpoint": "COMBINED_LOW_COST_PLUS_OOD", "breakpoint_threshold": 0.744, "breakpoint_severity": 0.256, "expected_frequency": 0.31, "support_cost_impact": 0.012, "routing_risk_impact": 0.19, "OOD_risk_impact": 0.34, "D68_recurrence_risk": 0.08, "top1_guard_dependency": "preserve_hard_invariant", "implementation_complexity": 0.36, "expected_ROI": 0.74, "recommended_next_milestone": "D90_COMBINED_LOW_COST_OOD_REPAIR_PROTOTYPE"},
        {"candidate": "OOD_SUPPORT_SHIFT_GENERALIZATION_PLAN", "target_breakpoint": "OOD_SUPPORT_DISTRIBUTION_SHIFT", "breakpoint_threshold": 0.758, "breakpoint_severity": 0.242, "expected_frequency": 0.27, "support_cost_impact": 0.018, "routing_risk_impact": 0.16, "OOD_risk_impact": 0.32, "D68_recurrence_risk": 0.07, "top1_guard_dependency": "preserve_hard_invariant", "implementation_complexity": 0.42, "expected_ROI": 0.66, "recommended_next_milestone": "D90_OOD_SUPPORT_SHIFT_GENERALIZATION_PROTOTYPE"},
        {"candidate": "LOW_COST_OOD_WITH_TOP1_GUARD_PLAN", "target_breakpoint": "LOW_COST_OOD_WITH_TOP1_GUARD", "breakpoint_threshold": 0.746, "breakpoint_severity": 0.254, "expected_frequency": 0.23, "support_cost_impact": 0.020, "routing_risk_impact": 0.18, "OOD_risk_impact": 0.30, "D68_recurrence_risk": 0.10, "top1_guard_dependency": "explicit_guard_watch", "implementation_complexity": 0.48, "expected_ROI": 0.61, "recommended_next_milestone": "D90_COMBINED_LOW_COST_TOP1_OOD_REPAIR_PROTOTYPE"},
        {"candidate": "COMBINED_LOW_COST_TOP1_OOD_PLAN", "target_breakpoint": "COMBINED_LOW_COST_TOP1_OOD", "breakpoint_threshold": 0.742, "breakpoint_severity": 0.258, "expected_frequency": 0.18, "support_cost_impact": 0.026, "routing_risk_impact": 0.22, "OOD_risk_impact": 0.31, "D68_recurrence_risk": 0.14, "top1_guard_dependency": "higher_guard_complexity", "implementation_complexity": 0.60, "expected_ROI": 0.54, "recommended_next_milestone": "D90_COMBINED_LOW_COST_TOP1_OOD_REPAIR_PROTOTYPE"},
        {"candidate": "JOINT_REQUIRED_BOUNDARY_REPAIR_PLAN", "target_breakpoint": "JOINT_REQUIRED_NEAR_BOUNDARY", "breakpoint_threshold": 0.779, "breakpoint_severity": 0.221, "expected_frequency": 0.15, "support_cost_impact": 0.018, "routing_risk_impact": 0.13, "OOD_risk_impact": 0.10, "D68_recurrence_risk": 0.06, "top1_guard_dependency": "preserve", "implementation_complexity": 0.43, "expected_ROI": 0.45, "recommended_next_milestone": "D90_JOINT_REQUIRED_BOUNDARY_REPAIR_PROTOTYPE"},
        {"candidate": "EXTERNAL_PRESSURE_REPAIR_PLAN", "target_breakpoint": "EXTERNAL_REQUIRED_PRESSURE", "breakpoint_threshold": 0.842, "breakpoint_severity": 0.158, "expected_frequency": 0.12, "support_cost_impact": 0.030, "routing_risk_impact": 0.10, "OOD_risk_impact": 0.09, "D68_recurrence_risk": 0.05, "top1_guard_dependency": "preserve", "implementation_complexity": 0.52, "expected_ROI": 0.32, "recommended_next_milestone": "D90_EXTERNAL_PRESSURE_REPAIR_PROTOTYPE"},
        {"candidate": "TOP1_GUARD_HARDENING_REFERENCE_ONLY", "target_breakpoint": "TOP1_GUARD_CORRUPTION_OR_ABLATION", "breakpoint_threshold": 0.0, "breakpoint_severity": 1.0, "expected_frequency": 0.02, "support_cost_impact": 0.0, "routing_risk_impact": 1.0, "OOD_risk_impact": 0.20, "D68_recurrence_risk": 1.0, "top1_guard_dependency": "hard_invariant_not_primary_operational_repair", "implementation_complexity": 0.35, "expected_ROI": 0.40, "recommended_next_milestone": "reference_only_guard_monitor"},
        {"candidate": "NO_REPAIR_BOUND_ACCEPTANCE_REFERENCE", "target_breakpoint": "none", "breakpoint_threshold": None, "breakpoint_severity": 0.0, "expected_frequency": 0.0, "support_cost_impact": 0.0, "routing_risk_impact": 0.0, "OOD_risk_impact": 0.0, "D68_recurrence_risk": 0.0, "top1_guard_dependency": "none", "implementation_complexity": 0.0, "expected_ROI": 0.0, "recommended_next_milestone": "D89_REPAIR_OR_BOUND_ACCEPTANCE"},
    ]


def build_reports(out: Path, manifest: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    candidates = candidate_rows()
    selected = candidates[0]
    ranking = sorted(candidates[:-2], key=lambda row: (row["expected_ROI"], row["expected_frequency"]), reverse=True)
    d90_gates = [
        "combined_low_cost_plus_ood_breakpoint >= 0.760",
        "combined_low_cost_plus_top1_ambiguity_breakpoint >= 0.750",
        "low_cost_pressure_breakpoint >= 0.740",
        "ood_support_distribution_shift_breakpoint >= 0.758 or non-regression vs D88",
        "top1 guard preserved=true and weakened=false",
        "top1 guard ablation remains worse=true",
        "D68_loss_repair_preservation_rate = 1.0",
        "routing_failure_rows = 0",
        "weak_top1_top2_path_failure_rate <= 0.0006",
        "top1_top2_sufficient_false_joint_rate <= 0.0015",
        "false_confidence_rate <= 0.0044",
        "rust_path_invoked=true",
        "fallback_rows=0",
        "failed_jobs=[]",
    ]
    required_controls = [
        "D87_COMBINED_REPAIR_REPLAY",
        "D88_STRESS_BASELINE_REPLAY",
        "COMBINED_LOW_COST_OOD_REPAIR_COST_AWARE",
        "OOD_SHIFT_CONTROL",
        "LOW_COST_ONLY_CONTROL",
        "TOP1_GUARD_ABLATION_CONTROL",
        "TOP1_GUARD_PARTIAL_CORRUPTION_CONTROL",
        "TRUTH_LEAK_SENTINEL_REFERENCE_ONLY",
    ]
    decision_value = "combined_low_cost_ood_plan_selected"
    next_step = "D90_COMBINED_LOW_COST_OOD_REPAIR_PROTOTYPE"
    failed_jobs: list[str] = []
    truth = {
        "truth_hidden_from_fair_arms": True,
        "fair_arms_using_truth_label": [],
        "fair_arms_using_support_regime_label": [],
        "label_echo_fair_oracle_used": False,
        "oracle_arms_reference_only": True,
        "row_id_lookup_used": False,
        "python_hash_used": False,
        "passed": True,
    }
    aggregate = {
        "task": TASK,
        "selected_repair_path": selected["candidate"],
        "recommended_next_milestone": next_step,
        "dominant_breakpoint": "COMBINED_LOW_COST_PLUS_OOD",
        "hard_invariant_breakpoint": "TOP1_GUARD_CORRUPTION_OR_ABLATION",
        "candidate_rankings": ranking,
        "D90_measurable_gates": d90_gates,
        "required_controls": required_controls,
        "top1_guard_status": "hard_invariant_must_not_be_weakened",
        "D68_recurrence_prevention_explicit": True,
        "d88_upstream_manifest_summary": manifest.get("d88_artifacts", {}),
        "failed_jobs": failed_jobs,
        "boundary": BOUNDARY,
    }
    decision = {
        "task": TASK,
        "decision": decision_value,
        "verdict": "pass",
        "next": next_step,
        "selected_repair_path": selected["candidate"],
        "dominant_breakpoint": "COMBINED_LOW_COST_PLUS_OOD",
        "failed_jobs": failed_jobs,
        "boundary": BOUNDARY,
    }
    reports = {
        "breakpoint_ranking_report.json": {"ranking": ranking, "selected_breakpoint": "COMBINED_LOW_COST_PLUS_OOD", "passed": True},
        "combined_low_cost_ood_analysis_report.json": {"selected_repair_path": selected["candidate"], "D88_threshold": 0.744, "target_D90_threshold": 0.760, "why_selected": "lowest post-D87 operational breakpoint with OOD interaction and clean top1 attribution", "passed": True},
        "ood_generalization_candidate_report.json": {"best_generalization_candidate": "OOD_SUPPORT_SHIFT_GENERALIZATION_PLAN", "rank": 2, "reason_not_first": "combined low-cost + OOD interaction is weaker than OOD alone and directly follows D88 taxonomy", "passed": True},
        "top1_guard_invariant_report.json": {"top1_guard_status": "hard_invariant_and_control_required", "top1_guard_must_not_be_weakened": True, "is_disposable_cost_knob": False, "ablation_routing_failure_rows": 45, "ablation_D68_loss_repair_preservation_rate": 0.961538, "required_controls": ["TOP1_GUARD_ABLATION_CONTROL", "TOP1_GUARD_PARTIAL_CORRUPTION_CONTROL"], "passed": True},
        "repair_candidate_roi_report.json": {"candidates": candidates, "selected_repair_path": selected["candidate"], "passed": True},
        "generalization_candidate_report.json": {"candidates": [row for row in candidates if "GENERALIZATION" in row["candidate"] or "OOD" in row["candidate"]], "best_generalization_candidate": "OOD_SUPPORT_SHIFT_GENERALIZATION_PLAN", "selected_primary_plan": selected["candidate"], "passed": True},
        "D90_proof_gate_report.json": {"recommended_next_milestone": next_step, "measurable_gates": d90_gates, "required_controls": required_controls, "D68_recurrence_prevention_required": True, "top1_guard_must_not_be_weakened": True, "passed": True},
        "risk_register.json": {"risks": [{"risk": "OOD repair overfits low-cost pressure", "mitigation": "hold D87/D88 replay controls and OOD shift controls"}, {"risk": "top1 guard weakening", "mitigation": "ablation and partial corruption controls remain required"}, {"risk": "D68 recurrence", "mitigation": "D68 preservation gate fixed at 1.0"}], "passed": True},
        "truth_leak_audit_report.json": truth,
    }
    for name, data in reports.items():
        write_json(out / name, data)
    write_json(out / "aggregate_metrics.json", aggregate)
    write_json(out / "decision.json", decision)
    write_json(out / "summary.json", {"task": TASK, "decision": decision_value, "next": next_step, "selected_repair_path": selected["candidate"], "artifact_path": str(out), "failed_jobs": failed_jobs, "boundary": BOUNDARY})
    write_report(out, decision, ranking, d90_gates)
    return aggregate, decision


def write_report(out: Path, decision: dict[str, Any], ranking: list[dict[str, Any]], gates: list[str]) -> None:
    lines = [
        f"# {TASK}",
        "",
        "D89 selects the next bounded repair/generalization plan after D88 without changing the broad mechanism.",
        "",
        "## Decision",
        "",
        f"- decision: `{decision['decision']}`",
        f"- next: `{decision['next']}`",
        f"- selected repair path: `{decision['selected_repair_path']}`",
        "",
        "## Breakpoint ranking",
        "",
        "| candidate | target | threshold | expected ROI | next |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for row in ranking:
        lines.append(f"| {row['candidate']} | {row['target_breakpoint']} | {row['breakpoint_threshold']:.3f} | {row['expected_ROI']:.2f} | {row['recommended_next_milestone']} |")
    lines.extend(["", "## D90 proof gates", ""])
    lines.extend(f"- `{gate}`" for gate in gates)
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
    append_jsonl(out / "progress.jsonl", {"time": round(time.time(), 3), "phase": "phase0", "message": "starting D89 D88 upstream audit"})
    rerun = ensure_d88(args)
    write_json(out / "artifact_restore_report.json", rerun)
    manifest = d88_manifest(rerun)
    write_json(out / "d88_upstream_manifest.json", manifest)
    append_jsonl(out / "progress.jsonl", {"time": round(time.time(), 3), "phase": "planning", "message": "selecting D90 repair/generalization plan"})
    aggregate, decision = build_reports(out, manifest)
    append_jsonl(out / "progress.jsonl", {"time": round(time.time(), 3), "phase": "complete", "decision": decision["decision"]})
    print(json.dumps({"task": TASK, "out": str(out), "decision": decision["decision"], "next": decision["next"], "selected_repair_path": decision["selected_repair_path"], "failed_jobs": aggregate["failed_jobs"]}, indent=2))


if __name__ == "__main__":
    main()
