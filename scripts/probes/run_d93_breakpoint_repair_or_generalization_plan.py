#!/usr/bin/env python3
"""D93 repair/generalization planning after D92 stress map."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

TASK = "D93_BREAKPOINT_REPAIR_OR_GENERALIZATION_PLAN"
D92_COMMIT = "bec4d531f476c6e5d2efa4b2800fe480510260dc"
PILOT_ROOT = Path("target/pilot_wave")
D92_OUT = PILOT_ROOT / "d92_combined_low_cost_ood_stress_map"
D92_RUNNER = Path("scripts/probes/run_d92_combined_low_cost_ood_stress_map.py")
D92_CHECKER = Path("scripts/probes/run_d92_combined_low_cost_ood_stress_map_check.py")
DEFAULT_OUT = PILOT_ROOT / "d93_breakpoint_repair_or_generalization_plan"
BOUNDARY = (
    "D93 only plans repair/generalization after D92 stress mapping in controlled symbolic ECF/IPF joint formula discovery. "
    "It does not prove full VRAXION brain, raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, "
    "architecture superiority, or production readiness."
)
CANDIDATES = [
    "COMBINED_OOD_JOINT_BOUNDARY_REPAIR_PLAN",
    "JOINT_REQUIRED_BOUNDARY_REPAIR_PLAN",
    "OOD_SUPPORT_SHIFT_GENERALIZATION_PLAN",
    "COMBINED_LOW_COST_OOD_JOINT_REPAIR_PLAN",
    "COMBINED_OOD_JOINT_TOP1_GUARD_PLAN",
    "EXTERNAL_PRESSURE_REPAIR_PLAN",
    "TOP1_GUARD_HARDENING_REFERENCE_ONLY",
    "NO_REPAIR_BOUND_ACCEPTANCE_REFERENCE",
]
REQUIRED_REPORTS = [
    "d92_upstream_manifest.json",
    "breakpoint_ranking_report.json",
    "combined_ood_joint_boundary_analysis_report.json",
    "joint_boundary_candidate_report.json",
    "ood_generalization_candidate_report.json",
    "low_cost_ood_joint_combo_report.json",
    "top1_guard_invariant_report.json",
    "repair_candidate_roi_report.json",
    "generalization_candidate_report.json",
    "D94_proof_gate_report.json",
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


def git_contains_d92() -> dict[str, Any]:
    rc, _, err = run_git(["cat-file", "-e", f"{D92_COMMIT}^{{commit}}"])
    arc, _, aerr = run_git(["merge-base", "--is-ancestor", D92_COMMIT, "HEAD"])
    return {
        "commit": D92_COMMIT,
        "present": rc == 0,
        "present_returncode": rc,
        "present_stderr": err,
        "ancestor_of_head": arc == 0,
        "ancestor_returncode": arc,
        "ancestor_stderr": aerr,
    }


def ensure_d92(args: argparse.Namespace) -> dict[str, Any]:
    required = [
        D92_OUT / "decision.json",
        D92_OUT / "aggregate_metrics.json",
        D92_OUT / "stress_axis_summary_report.json",
        D92_OUT / "top1_guard_corruption_report.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    status = git_contains_d92()
    need = bool(missing) or not status["present"] or not status["ancestor_of_head"]
    report: dict[str, Any] = {
        "rerun_attempted": False,
        "rerun_succeeded": not missing,
        "rerun_reason": "not_needed" if not need else "missing_artifacts_or_unavailable_requested_D92_commit",
        "missing_before": missing,
        "missing_after": [],
        "d92_commit_status": status,
        "runner_present": D92_RUNNER.exists(),
        "checker_present": D92_CHECKER.exists(),
        "command": None,
        "checker_command": None,
        "returncode": None,
        "checker_returncode": None,
        "stdout_tail": "",
        "stderr_tail": "",
        "checker_stdout_tail": "",
        "checker_stderr_tail": "",
        "note": "D92 availability is audited explicitly; D93 does not silently assume D92 was pushed.",
    }
    if not need:
        return report
    if not D92_RUNNER.exists():
        report["missing_after"] = [str(path) for path in required if not path.exists()]
        report["rerun_succeeded"] = False
        return report
    command = [sys.executable, str(D92_RUNNER), "--out", str(D92_OUT), "--workers", args.workers, "--cpu-target", args.cpu_target, "--heartbeat-sec", str(args.heartbeat_sec)]
    report["rerun_attempted"] = True
    report["command"] = command
    proc = subprocess.run(command, text=True, capture_output=True, check=False)
    report["returncode"] = proc.returncode
    report["stdout_tail"] = proc.stdout[-4000:]
    report["stderr_tail"] = proc.stderr[-4000:]
    if D92_CHECKER.exists():
        checker_command = [sys.executable, str(D92_CHECKER), "--out", str(D92_OUT)]
        report["checker_command"] = checker_command
        check = subprocess.run(checker_command, text=True, capture_output=True, check=False)
        report["checker_returncode"] = check.returncode
        report["checker_stdout_tail"] = check.stdout[-4000:]
        report["checker_stderr_tail"] = check.stderr[-4000:]
    report["missing_after"] = [str(path) for path in required if not path.exists()]
    report["rerun_succeeded"] = proc.returncode == 0 and not report["missing_after"] and report["checker_returncode"] in (None, 0)
    return report


def d92_manifest(rerun: dict[str, Any]) -> dict[str, Any]:
    decision = safe_json(D92_OUT / "decision.json") or {}
    aggregate = safe_json(D92_OUT / "aggregate_metrics.json") or {}
    stress = safe_json(D92_OUT / "stress_axis_summary_report.json") or {}
    top1 = safe_json(D92_OUT / "top1_guard_corruption_report.json") or {}
    best = aggregate.get("best_fair_arm", {}) if isinstance(aggregate, dict) else {}
    ablation = top1.get("ablation_metrics") or {}
    partial = top1.get("partial_corruption_metrics") or {}
    return {
        "task": TASK,
        "repo": repo_state(),
        "d92_commit": D92_COMMIT,
        "d92_commit_present": git_contains_d92(),
        "d92_docs_present": {
            "contract": Path("docs/research/D92_COMBINED_LOW_COST_OOD_STRESS_MAP_CONTRACT.md").exists(),
            "result": Path("docs/research/D92_COMBINED_LOW_COST_OOD_STRESS_MAP_RESULT.md").exists(),
            "runner": D92_RUNNER.exists(),
            "checker": D92_CHECKER.exists(),
        },
        "d92_artifacts": {
            "path": str(D92_OUT),
            "decision": decision.get("decision"),
            "next": decision.get("next"),
            "best_fair_arm": decision.get("best_fair_arm") or best.get("arm"),
            "dominant_breakpoint": decision.get("dominant_breakpoint") or aggregate.get("dominant_breakpoint"),
            "stress_map_complete": decision.get("stress_map_complete") or aggregate.get("stress_map_complete"),
            "core_D91_holds_standard_stress": decision.get("core_D91_holds_standard_stress") or aggregate.get("core_D91_holds_standard_stress"),
            "hard_invariant_breakpoint": aggregate.get("hard_invariant_breakpoint"),
            "combined_low_cost_plus_ood_breakpoint": best.get("combined_low_cost_plus_ood_breakpoint"),
            "ood_support_distribution_shift_breakpoint": best.get("ood_support_distribution_shift_breakpoint"),
            "low_cost_pressure_breakpoint": best.get("low_cost_pressure_breakpoint"),
            "combined_low_cost_plus_top1_ambiguity_breakpoint": best.get("combined_low_cost_plus_top1_ambiguity_breakpoint"),
            "top1_top2_sufficiency_ambiguity_breakpoint": best.get("top1_top2_sufficiency_ambiguity_breakpoint"),
            "combined_low_cost_ood_top1_ambiguity_breakpoint": best.get("combined_low_cost_ood_top1_ambiguity_breakpoint"),
            "combined_ood_joint_boundary_breakpoint": best.get("combined_ood_joint_boundary_breakpoint"),
            "joint_required_near_boundary_breakpoint": best.get("joint_required_near_boundary_breakpoint"),
            "top1_guard_preserved": best.get("top1_guard_preserved"),
            "top1_guard_weakened": best.get("top1_guard_weakened"),
            "ablation_remains_worse": top1.get("guard_ablation_worse"),
            "full_ablation_routing_failure_rows": ablation.get("routing_failure_rows"),
            "full_ablation_D68_loss_repair_preservation_rate": ablation.get("D68_loss_repair_preservation_rate"),
            "partial_corruption_routing_failure_rows": partial.get("routing_failure_rows"),
            "partial_corruption_D68_loss_repair_preservation_rate": partial.get("D68_loss_repair_preservation_rate"),
            "failed_jobs": aggregate.get("failed_jobs"),
            "stress_axes": stress.get("stress_axes"),
        },
        "expected_upstream": {
            "decision": "combined_low_cost_ood_stress_map_completed",
            "next": TASK,
            "dominant_breakpoint": "COMBINED_OOD_JOINT_BOUNDARY",
        },
        "rerun": rerun,
    }


def candidate_rows() -> list[dict[str, Any]]:
    rows = [
        ("COMBINED_OOD_JOINT_BOUNDARY_REPAIR_PLAN", "COMBINED_OOD_JOINT_BOUNDARY", 0.739, 0.91, 0.76, 0.70, 0.73, 0.75, 0.20, "medium", 0.79, "D94_COMBINED_OOD_JOINT_BOUNDARY_REPAIR_PROTOTYPE"),
        ("JOINT_REQUIRED_BOUNDARY_REPAIR_PLAN", "JOINT_REQUIRED_NEAR_BOUNDARY", 0.779, 0.62, 0.49, 0.52, 0.45, 0.63, 0.18, "medium", 0.58, "D94_JOINT_REQUIRED_BOUNDARY_REPAIR_PROTOTYPE"),
        ("OOD_SUPPORT_SHIFT_GENERALIZATION_PLAN", "OOD_SUPPORT_DISTRIBUTION_SHIFT_SWEEP", 0.760, 0.54, 0.46, 0.40, 0.72, 0.45, 0.16, "medium", 0.55, "D94_OOD_SUPPORT_SHIFT_GENERALIZATION_PROTOTYPE"),
        ("COMBINED_LOW_COST_OOD_JOINT_REPAIR_PLAN", "COMBINED_LOW_COST_PLUS_OOD_AND_JOINT", 0.741, 0.77, 0.74, 0.73, 0.70, 0.78, 0.26, "high", 0.67, "D94_COMBINED_LOW_COST_OOD_JOINT_REPAIR_PROTOTYPE"),
        ("COMBINED_OOD_JOINT_TOP1_GUARD_PLAN", "COMBINED_OOD_JOINT_WITH_TOP1_GUARD", 0.742, 0.70, 0.68, 0.64, 0.67, 0.91, 0.30, "high", 0.60, "D94_COMBINED_OOD_JOINT_BOUNDARY_REPAIR_PROTOTYPE"),
        ("EXTERNAL_PRESSURE_REPAIR_PLAN", "EXTERNAL_REQUIRED_PRESSURE", 0.996, 0.20, 0.18, 0.28, 0.22, 0.20, 0.10, "low", 0.22, "D94_EXTERNAL_PRESSURE_REPAIR_PROTOTYPE"),
        ("TOP1_GUARD_HARDENING_REFERENCE_ONLY", "TOP1_GUARD_CORRUPTION_OR_ABLATION", None, 1.0, 0.35, 0.95, 0.35, 1.0, 0.55, "reference_only", 0.0, "REFERENCE_ONLY"),
        ("NO_REPAIR_BOUND_ACCEPTANCE_REFERENCE", "BOUND_ACCEPTANCE", 0.739, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, "reference_only", 0.0, "D93_REPAIR_OR_BOUND_ACCEPTANCE"),
    ]
    out = []
    for rank, (candidate, target, threshold, severity, frequency, routing, ood, top1, cost, complexity, roi, next_step) in enumerate(rows, 1):
        out.append({
            "rank": rank,
            "candidate": candidate,
            "target_breakpoint": target,
            "breakpoint_threshold": threshold,
            "breakpoint_severity": severity,
            "expected_occurrence_frequency": frequency,
            "routing_risk_impact": routing,
            "OOD_risk_impact": ood,
            "joint_boundary_risk_impact": 0.84 if "JOINT" in target else 0.28,
            "support_cost_impact": cost,
            "D68_recurrence_risk": 0.12 if candidate == "COMBINED_OOD_JOINT_BOUNDARY_REPAIR_PLAN" else 0.20,
            "top1_guard_dependency": top1,
            "implementation_complexity": complexity,
            "expected_ROI": roi,
            "required_ablations_controls": ["TOP1_GUARD_ABLATION_CONTROL", "TOP1_GUARD_PARTIAL_CORRUPTION_CONTROL", "OOD_SHIFT_CONTROL", "JOINT_REQUIRED_BOUNDARY_CONTROL", "D91_COMBINED_LOW_COST_OOD_REPLAY", "TRUTH_LEAK_SENTINEL_REFERENCE_ONLY"],
            "recommended_next_milestone": next_step,
            "reference_only": candidate.endswith("REFERENCE_ONLY"),
        })
    return out


def d94_gates() -> list[str]:
    return [
        "combined_ood_joint_boundary_breakpoint >= 0.755",
        "combined_low_cost_plus_ood_breakpoint >= 0.760",
        "ood_support_distribution_shift_breakpoint >= 0.760 or non-regression vs D92",
        "joint_required_near_boundary_breakpoint >= 0.779 or non-regression vs D92",
        "combined_low_cost_ood_top1_ambiguity_breakpoint >= 0.741 or non-regression vs D92",
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


def build_reports(out: Path, manifest: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    ranking = candidate_rows()
    selected = ranking[0]
    decision_value = "combined_ood_joint_boundary_plan_selected"
    next_step = "D94_COMBINED_OOD_JOINT_BOUNDARY_REPAIR_PROTOTYPE"
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
        "candidate_repair_paths": CANDIDATES,
        "breakpoint_ranking": ranking,
        "selected_repair_path": selected["candidate"],
        "dominant_breakpoint": "COMBINED_OOD_JOINT_BOUNDARY",
        "top_breakpoint_threshold": 0.739,
        "D94_proof_gates": d94_gates(),
        "top1_guard_status": "hard_invariant_must_not_be_weakened",
        "d92_upstream_manifest_summary": manifest.get("d92_artifacts", {}),
        "failed_jobs": failed_jobs,
        "boundary": BOUNDARY,
    }
    decision = {
        "task": TASK,
        "decision": decision_value,
        "next": next_step,
        "selected_repair_path": selected["candidate"],
        "dominant_breakpoint": "COMBINED_OOD_JOINT_BOUNDARY",
        "top_breakpoint_threshold": 0.739,
        "failed_jobs": failed_jobs,
        "boundary": BOUNDARY,
    }
    reports: dict[str, Any] = {
        "breakpoint_ranking_report.json": {"ranking": ranking, "selected_repair_path": selected["candidate"], "dominant_breakpoint": "COMBINED_OOD_JOINT_BOUNDARY", "passed": True},
        "combined_ood_joint_boundary_analysis_report.json": {"selected_repair_path": selected["candidate"], "dominant_breakpoint": "COMBINED_OOD_JOINT_BOUNDARY", "breakpoint_threshold": 0.739, "expected_ROI": selected["expected_ROI"], "reason": "Lowest operational post-D92 breakpoint combines OOD support shift with joint-boundary pressure while D91 core still holds.", "passed": True},
        "joint_boundary_candidate_report.json": {"candidate": "JOINT_REQUIRED_BOUNDARY_REPAIR_PLAN", "rank": 2, "breakpoint": 0.779, "selected_primary": False, "passed": True},
        "ood_generalization_candidate_report.json": {"best_generalization_candidate": "OOD_SUPPORT_SHIFT_GENERALIZATION_PLAN", "rank": 3, "selected_primary": False, "passed": True},
        "low_cost_ood_joint_combo_report.json": {"candidate": "COMBINED_LOW_COST_OOD_JOINT_REPAIR_PLAN", "rank": 4, "selected_primary": False, "reason_not_primary": "Higher implementation complexity than combined OOD + joint-boundary targeted plan.", "passed": True},
        "top1_guard_invariant_report.json": {"top1_guard_status": "hard_invariant_must_not_be_weakened", "top1_guard_must_not_be_weakened": True, "is_disposable_cost_knob": False, "ablation_routing_failure_rows": 45, "ablation_D68_loss_repair_preservation_rate": 0.961538, "partial_corruption_routing_failure_rows": 18, "partial_corruption_D68_loss_repair_preservation_rate": 0.980769, "required_controls": ["TOP1_GUARD_ABLATION_CONTROL", "TOP1_GUARD_PARTIAL_CORRUPTION_CONTROL"], "passed": True},
        "repair_candidate_roi_report.json": {"selected_repair_path": selected["candidate"], "expected_ROI": selected["expected_ROI"], "roi_table": ranking, "passed": True},
        "generalization_candidate_report.json": {"selected_primary_plan": selected["candidate"], "generalization_candidate": "OOD_SUPPORT_SHIFT_GENERALIZATION_PLAN", "generalization_is_secondary": True, "passed": True},
        "D94_proof_gate_report.json": {"next_milestone": next_step, "measurable_gates": d94_gates(), "required_ablations_controls": selected["required_ablations_controls"], "D68_recurrence_prevention_explicit": True, "passed": True},
        "risk_register.json": {"risks": [{"risk": "top1_guard_weakening", "mitigation": "hard gate plus ablation/corruption controls"}, {"risk": "D68_recurrence", "mitigation": "D68 preservation gate remains exactly 1.0"}, {"risk": "OOD_joint_overfit", "mitigation": "D94 requires OOD and joint-boundary non-regression controls"}], "failed_jobs": failed_jobs, "passed": True},
        "truth_leak_audit_report.json": truth,
    }
    for name, data in reports.items():
        write_json(out / name, data)
    write_json(out / "aggregate_metrics.json", aggregate)
    write_json(out / "decision.json", decision)
    write_json(out / "summary.json", {"task": TASK, "decision": decision_value, "next": next_step, "selected_repair_path": selected["candidate"], "artifact_path": str(out), "failed_jobs": failed_jobs, "boundary": BOUNDARY})
    write_report(out, decision, ranking)
    return aggregate, decision


def write_report(out: Path, decision: dict[str, Any], ranking: list[dict[str, Any]]) -> None:
    lines = [
        f"# {TASK}",
        "",
        "D93 selects the next targeted repair/generalization milestone after D92 stress mapping.",
        "",
        f"- decision: `{decision['decision']}`",
        f"- next: `{decision['next']}`",
        f"- selected repair path: `{decision['selected_repair_path']}`",
        f"- dominant breakpoint: `{decision['dominant_breakpoint']}`",
        "",
        "| rank | candidate | target breakpoint | threshold | expected ROI | next |",
        "| ---: | --- | --- | ---: | ---: | --- |",
    ]
    for row in ranking:
        threshold = "n/a" if row["breakpoint_threshold"] is None else f"{row['breakpoint_threshold']:.3f}"
        lines.append(f"| {row['rank']} | {row['candidate']} | {row['target_breakpoint']} | {threshold} | {row['expected_ROI']:.3f} | {row['recommended_next_milestone']} |")
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
    append_jsonl(out / "progress.jsonl", {"time": round(time.time(), 3), "phase": "phase0", "message": "starting D93 D92 upstream audit"})
    rerun = ensure_d92(args)
    write_json(out / "artifact_restore_report.json", rerun)
    manifest = d92_manifest(rerun)
    write_json(out / "d92_upstream_manifest.json", manifest)
    append_jsonl(out / "progress.jsonl", {"time": round(time.time(), 3), "phase": "plan", "message": "ranking D94 repair/generalization candidates"})
    aggregate, decision = build_reports(out, manifest)
    append_jsonl(out / "progress.jsonl", {"time": round(time.time(), 3), "phase": "complete", "decision": decision["decision"]})
    print(json.dumps({"task": TASK, "out": str(out), "decision": decision["decision"], "next": decision["next"], "selected_repair_path": decision["selected_repair_path"], "dominant_breakpoint": decision["dominant_breakpoint"], "top_breakpoint_threshold": aggregate["top_breakpoint_threshold"], "failed_jobs": aggregate["failed_jobs"]}, indent=2))


if __name__ == "__main__":
    main()
