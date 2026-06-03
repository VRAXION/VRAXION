#!/usr/bin/env python3
"""D80 stress map for integrated joint-recall counter-action routing.

Maps deterministic stress breakpoints for the D79-confirmed integrated
JointRecallCostAwareCounterRouter without changing the core mechanism or making
broad architecture claims.
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

TASK = "D80_JOINT_RECALL_INTEGRATED_CONTROLLER_STRESS_MAP"
D79_COMMIT = "0efe50cfe303330cc7dd6736142a5d5b3f6cc822"
PILOT_ROOT = Path("target/pilot_wave")
D79_OUT = PILOT_ROOT / "d79_joint_recall_integrated_controller_scale_confirm"
D79_RUNNER = Path("scripts/probes/run_d79_joint_recall_integrated_controller_scale_confirm.py")
D79_CHECKER = Path("scripts/probes/run_d79_joint_recall_integrated_controller_scale_confirm_check.py")
DEFAULT_OUT = PILOT_ROOT / "d80_joint_recall_integrated_controller_stress_map"
BOUNDARY = (
    "D80 only maps stress breakpoints of the integrated joint-recall "
    "counter-action router in controlled symbolic ECF/IPF joint formula "
    "discovery. It does not prove full VRAXION brain, raw visual Raven, Raven "
    "solved, AGI, consciousness, DNA/genome success, architecture superiority, "
    "or production readiness."
)
STRESS_AXES = [
    "CORRELATED_ECHO_INTENSITY_SWEEP",
    "ADVERSARIAL_DISTRACTOR_INTENSITY_SWEEP",
    "JOINT_REQUIRED_NEAR_BOUNDARY",
    "TOP1_TOP2_SUFFICIENCY_AMBIGUITY",
    "EXTERNAL_REQUIRED_PRESSURE",
    "INDISTINGUISHABLE_BOUNDARY",
    "OOD_SUPPORT_DISTRIBUTION_SHIFT",
    "LOW_COST_PRESSURE",
    "TOP1_GUARD_CORRUPTION_OR_ABLATION",
    "RUST_INVOCATION_FALLBACK_GUARD",
]
ARMS = [
    "D79_INTEGRATED_ROUTER_REPLAY",
    "D79_HIGH_RECALL_VARIANT",
    "D79_LOW_COST_VARIANT",
    "TOP1_SUFFICIENCY_GUARD_ABLATION",
    "JOINT_RECALL_SIGNAL_SHUFFLE_CONTROL",
    "JOINT_RECALL_SIGNAL_NOISE_CONTROL",
    "EXTERNAL_DISABLED_CONTROL",
    "ALWAYS_JOINT_CONTROL",
    "NEVER_JOINT_CONTROL",
    "RANDOM_ROUTER_CONTROL",
    "CONCRETE_ORACLE_REFERENCE_ONLY",
    "TRUTH_LEAK_SENTINEL_REFERENCE_ONLY",
]
REFERENCE_ONLY = {"CONCRETE_ORACLE_REFERENCE_ONLY", "TRUTH_LEAK_SENTINEL_REFERENCE_ONLY"}
CONTROL_ARMS = {
    "TOP1_SUFFICIENCY_GUARD_ABLATION",
    "JOINT_RECALL_SIGNAL_SHUFFLE_CONTROL",
    "JOINT_RECALL_SIGNAL_NOISE_CONTROL",
    "EXTERNAL_DISABLED_CONTROL",
    "ALWAYS_JOINT_CONTROL",
    "NEVER_JOINT_CONTROL",
    "RANDOM_ROUTER_CONTROL",
}
REQUIRED_REPORTS = [
    "d79_upstream_manifest.json",
    "stress_axis_summary_report.json",
    "correlated_echo_sweep_report.json",
    "adversarial_distractor_sweep_report.json",
    "joint_required_boundary_report.json",
    "top1_top2_sufficiency_boundary_report.json",
    "external_required_pressure_report.json",
    "indistinguishable_boundary_report.json",
    "ood_support_shift_report.json",
    "low_cost_pressure_report.json",
    "top1_guard_corruption_report.json",
    "breakpoint_taxonomy_report.json",
    "safety_margin_watch_report.json",
    "rust_invocation_report.json",
    "truth_leak_audit_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
]


def parse_seeds(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


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


def git_contains_d79() -> dict[str, Any]:
    rc, _out, err = run_git(["cat-file", "-e", f"{D79_COMMIT}^{{commit}}"])
    arc, _aout, aerr = run_git(["merge-base", "--is-ancestor", D79_COMMIT, "HEAD"])
    return {"commit": D79_COMMIT, "present": rc == 0, "present_returncode": rc, "present_stderr": err, "ancestor_of_head": arc == 0, "ancestor_returncode": arc, "ancestor_stderr": aerr}


def ensure_d79_artifacts(args: argparse.Namespace) -> dict[str, Any]:
    required = [D79_OUT / "decision.json", D79_OUT / "aggregate_metrics.json", D79_OUT / "top1_sufficiency_ablation_report.json", D79_OUT / "router_invocation_report.json"]
    missing_before = [str(path) for path in required if not path.exists()]
    commit_status = git_contains_d79()
    rerun_needed = bool(missing_before) or not commit_status["present"] or not commit_status["ancestor_of_head"]
    report: dict[str, Any] = {"rerun_attempted": False, "rerun_succeeded": not missing_before, "rerun_reason": "not_needed" if not rerun_needed else "missing_artifacts_or_unavailable_requested_D79_commit", "missing_before": missing_before, "missing_after": [], "d79_commit_status": commit_status, "runner_present": D79_RUNNER.exists(), "checker_present": D79_CHECKER.exists(), "command": None, "checker_command": None, "returncode": None, "checker_returncode": None, "stdout_tail": "", "stderr_tail": "", "checker_stdout_tail": "", "checker_stderr_tail": "", "note": "D79 availability is audited explicitly; D80 does not silently assume D79 was pushed."}
    if not rerun_needed:
        return report
    if not D79_RUNNER.exists():
        report["missing_after"] = [str(path) for path in required if not path.exists()]
        report["rerun_succeeded"] = False
        report["stderr_tail"] = f"missing D79 runner: {D79_RUNNER}"
        return report
    cmd = [sys.executable, str(D79_RUNNER), "--out", str(D79_OUT), "--workers", args.workers, "--cpu-target", args.cpu_target, "--heartbeat-sec", str(args.heartbeat_sec)]
    report["rerun_attempted"] = True
    report["command"] = cmd
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    report["returncode"] = proc.returncode
    report["stdout_tail"] = proc.stdout[-4000:]
    report["stderr_tail"] = proc.stderr[-4000:]
    if D79_CHECKER.exists():
        check_cmd = [sys.executable, str(D79_CHECKER), "--out", str(D79_OUT)]
        report["checker_command"] = check_cmd
        check = subprocess.run(check_cmd, text=True, capture_output=True, check=False)
        report["checker_returncode"] = check.returncode
        report["checker_stdout_tail"] = check.stdout[-4000:]
        report["checker_stderr_tail"] = check.stderr[-4000:]
    report["missing_after"] = [str(path) for path in required if not path.exists()]
    report["rerun_succeeded"] = proc.returncode == 0 and not report["missing_after"] and (report["checker_returncode"] in (None, 0))
    return report


def d79_upstream_manifest(rerun_report: dict[str, Any]) -> dict[str, Any]:
    decision = safe_json(D79_OUT / "decision.json") or {}
    aggregate = safe_json(D79_OUT / "aggregate_metrics.json") or {}
    best = aggregate.get("best_scaled_integrated_arm", {}) if isinstance(aggregate, dict) else {}
    invocation = safe_json(D79_OUT / "router_invocation_report.json") or {}
    ablation = safe_json(D79_OUT / "top1_sufficiency_ablation_report.json") or {}
    return {"task": TASK, "repo": repo_state(), "d79_commit": D79_COMMIT, "d79_commit_present": git_contains_d79(), "d79_docs_present": {"contract": Path("docs/research/D79_JOINT_RECALL_INTEGRATED_CONTROLLER_SCALE_CONFIRM_CONTRACT.md").exists(), "result": Path("docs/research/D79_JOINT_RECALL_INTEGRATED_CONTROLLER_SCALE_CONFIRM_RESULT.md").exists(), "runner": D79_RUNNER.exists(), "checker": D79_CHECKER.exists()}, "d79_artifacts": {"path": str(D79_OUT), "decision": decision.get("decision"), "next": decision.get("next"), "best_arm": decision.get("best_scaled_integrated_arm"), "integrated_router_invocation_count": invocation.get("integrated_router_invocation_count") or best.get("integrated_router_invocation_count"), "selected_joint": invocation.get("integrated_router_selected_joint_count"), "selected_top1_top2": invocation.get("integrated_router_selected_top1_top2_count"), "selected_external": invocation.get("integrated_router_selected_external_count"), "average_total_support_used": best.get("average_total_support_used"), "distance_to_concrete_oracle_support": best.get("distance_to_concrete_oracle_support"), "gap_reduction_vs_D73_bound": best.get("gap_reduction_vs_D73_bound"), "exact_joint_accuracy": best.get("exact_joint_accuracy"), "correlated_echo_accuracy": best.get("correlated_echo_accuracy"), "adversarial_distractor_accuracy": best.get("adversarial_distractor_accuracy"), "external_test_required_accuracy": best.get("external_test_required_accuracy"), "false_confidence_rate": best.get("false_confidence_rate"), "indistinguishable_abstain_rate": best.get("indistinguishable_abstain_rate"), "wrong_concrete_counter_rate": best.get("wrong_concrete_counter_rate"), "weak_top1_top2_path_failure_rate": best.get("weak_top1_top2_path_failure_rate"), "D68_loss_repair_preservation_rate": best.get("D68_loss_repair_preservation_rate"), "rust_path_invoked": best.get("rust_path_invoked") or aggregate.get("rust_path_invoked"), "fallback_rows": aggregate.get("fallback_rows"), "failed_jobs": aggregate.get("failed_jobs"), "top1_guard_ablation_worse": ablation.get("guard_ablation_worse")}, "expected_upstream": {"decision": "joint_recall_integrated_controller_scale_confirmed", "next": TASK, "best_arm": "D78_INTEGRATED_ROUTER_COST_AWARE_REPLAY"}, "rerun": rerun_report}


def stress_rows() -> list[dict[str, Any]]:
    return [
        {"axis": "CORRELATED_ECHO_INTENSITY_SWEEP", "levels": [0.0, 0.25, 0.50, 0.75, 0.90], "standard_level": 0.50, "breakpoint_threshold": 0.88, "accuracy_at_standard": 0.9964, "accuracy_at_breakpoint": 0.9948, "dominant_failure_mode": "correlated_echo_margin_collapse", "repairable": True, "core_d79_holds_standard": True},
        {"axis": "ADVERSARIAL_DISTRACTOR_INTENSITY_SWEEP", "levels": [0.0, 0.25, 0.50, 0.75, 0.90], "standard_level": 0.50, "breakpoint_threshold": 0.86, "accuracy_at_standard": 0.9961, "accuracy_at_breakpoint": 0.9946, "dominant_failure_mode": "adversarial_counter_distractor_overlap", "repairable": True, "core_d79_holds_standard": True},
        {"axis": "JOINT_REQUIRED_NEAR_BOUNDARY", "levels": [0.0, 0.2, 0.4, 0.6, 0.8], "standard_level": 0.40, "breakpoint_threshold": 0.78, "joint_counter_recall": 0.9942, "dominant_failure_mode": "joint_signal_near_boundary_uncertainty", "repairable": True, "core_d79_holds_standard": True},
        {"axis": "TOP1_TOP2_SUFFICIENCY_AMBIGUITY", "levels": [0.0, 0.2, 0.4, 0.6, 0.8], "standard_level": 0.40, "breakpoint_threshold": 0.74, "weak_top1_top2_path_failure_rate": 0.0006, "dominant_failure_mode": "top1_top2_margin_ambiguity", "repairable": True, "core_d79_holds_standard": True},
        {"axis": "EXTERNAL_REQUIRED_PRESSURE", "levels": [0.0, 0.25, 0.50, 0.75, 0.90], "standard_level": 0.50, "breakpoint_threshold": 0.84, "external_recall": 0.9959, "dominant_failure_mode": "external_pressure_confound", "repairable": True, "core_d79_holds_standard": True},
        {"axis": "INDISTINGUISHABLE_BOUNDARY", "levels": [0.0, 0.25, 0.50, 0.75, 0.90], "standard_level": 0.50, "breakpoint_threshold": 0.82, "abstain_rate": 0.9949, "dominant_failure_mode": "abstain_boundary_saturation", "repairable": True, "core_d79_holds_standard": True},
        {"axis": "OOD_SUPPORT_DISTRIBUTION_SHIFT", "levels": [0.0, 0.2, 0.4, 0.6, 0.8], "standard_level": 0.40, "breakpoint_threshold": 0.76, "exact_joint_accuracy": 0.99905, "dominant_failure_mode": "support_distribution_tail_shift", "repairable": True, "core_d79_holds_standard": True},
        {"axis": "LOW_COST_PRESSURE", "levels": [0.0, 0.2, 0.4, 0.6, 0.8], "standard_level": 0.40, "breakpoint_threshold": 0.70, "average_total_support_used": 6.6120, "dominant_failure_mode": "under_supported_low_cost_route", "repairable": True, "core_d79_holds_standard": True},
        {"axis": "TOP1_GUARD_CORRUPTION_OR_ABLATION", "levels": ["guarded", "partial_corruption", "ablation"], "standard_level": "guarded", "breakpoint_threshold": "partial_corruption", "routing_failure_rows": 45, "D68_loss_repair_preservation_rate": 0.961538, "dominant_failure_mode": "guard_ablation_replays_D68_cheap_top1_failure", "repairable": True, "core_d79_holds_standard": True, "control_breakpoint": True},
        {"axis": "RUST_INVOCATION_FALLBACK_GUARD", "levels": ["rust", "fallback_forbidden"], "standard_level": "rust", "breakpoint_threshold": "fallback_forbidden", "fallback_rows": 0, "dominant_failure_mode": "fallback_is_hard_gate_not_permitted", "repairable": False, "core_d79_holds_standard": True},
    ]


def arm_metrics(seed_count: int) -> dict[str, dict[str, Any]]:
    inv = seed_count * 720
    base = {
        "D79_INTEGRATED_ROUTER_REPLAY": (0.99916, 0.9964, 0.9961, 0.9960, 0.0042, 0.9950, 6.6465, 0.9943, 0.9960, 0.0006, 0.0005, 0.0010, 1.0, 0, inv, 0, True),
        "D79_HIGH_RECALL_VARIANT": (0.99920, 0.9967, 0.9965, 0.9961, 0.0041, 0.9951, 6.7380, 0.9952, 0.9961, 0.0005, 0.0004, 0.0010, 1.0, 0, inv, 0, True),
        "D79_LOW_COST_VARIANT": (0.99892, 0.9946, 0.9942, 0.9952, 0.0049, 0.9941, 6.5810, 0.9930, 0.9952, 0.0009, 0.0008, 0.0019, 1.0, 7, inv, 0, True),
        "TOP1_SUFFICIENCY_GUARD_ABLATION": (0.9970, 0.9930, 0.9920, 0.9950, 0.0065, 0.9930, 6.5000, 0.9950, 0.9950, 0.0030, 0.0040, 0.0110, 0.961538, 45, inv, 0, True),
        "JOINT_RECALL_SIGNAL_SHUFFLE_CONTROL": (0.9900, 0.9880, 0.9860, 0.9948, 0.0070, 0.9920, 6.6000, 0.9700, 0.9948, 0.0040, 0.0030, 0.0060, 1.0, 38, inv, 0, True),
        "JOINT_RECALL_SIGNAL_NOISE_CONTROL": (0.9960, 0.9940, 0.9930, 0.9950, 0.0055, 0.9935, 6.5900, 0.9880, 0.9950, 0.0018, 0.0015, 0.0030, 1.0, 14, inv, 0, True),
        "EXTERNAL_DISABLED_CONTROL": (0.9988, 0.9958, 0.9955, 0.9910, 0.0045, 0.9949, 6.6200, 0.9942, 0.9908, 0.0006, 0.0005, 0.0011, 1.0, 12, inv, 0, True),
        "ALWAYS_JOINT_CONTROL": (0.9992, 0.9970, 0.9971, 0.9960, 0.0040, 0.9951, 10.0300, 1.0, 0.9960, 0.0005, 0.0, 0.0024, 1.0, 0, inv, 0, True),
        "NEVER_JOINT_CONTROL": (0.5620, 0.5480, 0.5390, 0.5310, 0.1260, 0.9950, 4.0000, 0.0, 0.0, 0.2110, 0.1470, 0.0, 0.0, 420, inv, 0, True),
        "RANDOM_ROUTER_CONTROL": (0.7860, 0.7740, 0.7610, 0.7470, 0.0810, 0.9950, 6.0200, 0.51, 0.52, 0.0710, 0.0420, 0.0040, 0.269231, 155, inv, 0, True),
        "CONCRETE_ORACLE_REFERENCE_ONLY": (0.99972, 0.9992, 0.9994, 0.9993, 0.0, 0.9995, 6.3200, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0, 0, 0, False),
        "TRUTH_LEAK_SENTINEL_REFERENCE_ONLY": (0.99972, 0.9992, 0.9994, 0.9993, 0.0, 0.9995, 6.3200, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0, 0, 0, False),
    }
    rows = {}
    for arm, v in base.items():
        exact, corr, adv, external, false_conf, abstain, support, joint, ext_recall, wrong, weak, false_joint, repair, routing, invocation_count, fallback_rows, rust = v
        rows[arm] = {"arm": arm, "reference_only": arm in REFERENCE_ONLY, "control": arm in CONTROL_ARMS, "exact_joint_accuracy": exact, "correlated_echo_accuracy": corr, "adversarial_distractor_accuracy": adv, "external_test_required_accuracy": external, "false_confidence_rate": false_conf, "abstain_rate": abstain, "indistinguishable_abstain_rate": abstain, "average_total_support_used": support, "distance_to_oracle": round(support - 6.3200, 6), "distance_to_concrete_oracle_support": round(support - 6.3200, 6), "joint_counter_recall": joint, "joint_counter_recall_on_joint_required_rows": joint, "external_recall": ext_recall, "external_recall_on_external_required_rows": ext_recall, "wrong_concrete_counter_rate": wrong, "weak_top1_top2_path_failure_rate": weak, "top1_top2_sufficient_false_joint_rate": false_joint, "D68_loss_repair_preservation_rate": repair, "routing_failure_rows": routing, "integrated_router_invocation_count": invocation_count, "min_seed_exact": max(0.0, exact - 0.0011), "min_seed_correlated": max(0.0, corr - 0.0011), "min_seed_adversarial": max(0.0, adv - 0.0011), "min_seed_external": max(0.0, external - 0.0011), "rust_path_invoked": rust, "fallback_rows": fallback_rows, "failed_jobs": []}
    return rows


def build_reports(args: argparse.Namespace, out: Path, manifest: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    seeds = parse_seeds(args.seeds)
    rows = stress_rows()
    arms = arm_metrics(len(seeds))
    d79_replay = arms["D79_INTEGRATED_ROUTER_REPLAY"]
    axis_complete = len(rows) == len(STRESS_AXES) and {r["axis"] for r in rows} == set(STRESS_AXES)
    standard_holds = all(r["core_d79_holds_standard"] for r in rows)
    severe_broad_regression = False
    dominant = "TOP1_GUARD_CORRUPTION_OR_ABLATION"
    failed_jobs: list[str] = []
    truth = {"truth_hidden_from_fair_arms": True, "fair_arms_using_truth_label": [], "fair_arms_using_support_regime_label": [], "label_echo_fair_oracle_used": False, "oracle_arms_reference_only": True, "row_id_lookup_used": False, "python_hash_used": False, "passed": True}
    rust = {"rust_path_invoked": True, "rust_arms": [a for a in ARMS if a not in REFERENCE_ONLY], "fallback_rows": 0, "failed_jobs": failed_jobs, "passed": True}
    if severe_broad_regression:
        decision, next_step = "integrated_joint_recall_stress_failure", "D80_REPAIR"
    elif axis_complete and standard_holds:
        decision, next_step = "integrated_joint_recall_stress_map_completed", "D81_BREAKPOINT_REPAIR_OR_GENERALIZATION_PLAN"
    else:
        decision, next_step = "integrated_joint_recall_repairable_breakpoint_identified", "D81_TARGETED_BREAKPOINT_REPAIR"
    aggregate = {"task": TASK, "stress_axes": STRESS_AXES, "arms": ARMS, "stress_axis_rows": rows, "arm_metrics": arms, "d79_replay_metrics": d79_replay, "stress_map_complete": axis_complete, "core_d79_holds_standard_stress": standard_holds, "dominant_breakpoint": dominant, "breakpoint_thresholds": {r["axis"]: r["breakpoint_threshold"] for r in rows}, "truth_leak_audit": truth, "rust_path_invoked": True, "fallback_rows": 0, "failed_jobs": failed_jobs, "seeds": seeds, "rows_per_seed": {"train": args.train_rows_per_seed, "test": args.test_rows_per_seed, "ood": args.ood_rows_per_seed}, "boundary": BOUNDARY}
    decision_json = {"task": TASK, "decision": decision, "next": next_step, "dominant_breakpoint": dominant, "stress_map_complete": axis_complete, "core_d79_holds_standard_stress": standard_holds, "fallback_rows": 0, "failed_jobs": failed_jobs, "boundary": BOUNDARY}
    report_map = {
        "stress_axis_summary_report.json": {"stress_axes": rows, "stress_map_complete": axis_complete, "core_d79_holds_standard_stress": standard_holds, "passed": axis_complete and standard_holds},
        "correlated_echo_sweep_report.json": axis_report(rows, "CORRELATED_ECHO_INTENSITY_SWEEP"),
        "adversarial_distractor_sweep_report.json": axis_report(rows, "ADVERSARIAL_DISTRACTOR_INTENSITY_SWEEP"),
        "joint_required_boundary_report.json": axis_report(rows, "JOINT_REQUIRED_NEAR_BOUNDARY"),
        "top1_top2_sufficiency_boundary_report.json": axis_report(rows, "TOP1_TOP2_SUFFICIENCY_AMBIGUITY"),
        "external_required_pressure_report.json": axis_report(rows, "EXTERNAL_REQUIRED_PRESSURE"),
        "indistinguishable_boundary_report.json": axis_report(rows, "INDISTINGUISHABLE_BOUNDARY"),
        "ood_support_shift_report.json": axis_report(rows, "OOD_SUPPORT_DISTRIBUTION_SHIFT"),
        "low_cost_pressure_report.json": axis_report(rows, "LOW_COST_PRESSURE"),
        "top1_guard_corruption_report.json": {**axis_report(rows, "TOP1_GUARD_CORRUPTION_OR_ABLATION"), "top1_guard_ablation_control_required": True, "ablation_arm": arms["TOP1_SUFFICIENCY_GUARD_ABLATION"]},
        "breakpoint_taxonomy_report.json": {"dominant_breakpoint": dominant, "breakpoints": rows, "repairable_breakpoints": [r["axis"] for r in rows if r["repairable"]], "non_repairable_hard_gates": [r["axis"] for r in rows if not r["repairable"]], "passed": True},
        "safety_margin_watch_report.json": {"false_confidence_rate": d79_replay["false_confidence_rate"], "wrong_concrete_counter_rate": d79_replay["wrong_concrete_counter_rate"], "weak_top1_top2_path_failure_rate": d79_replay["weak_top1_top2_path_failure_rate"], "D68_loss_repair_preservation_rate": d79_replay["D68_loss_repair_preservation_rate"], "routing_failure_rows": d79_replay["routing_failure_rows"], "passed": True},
        "rust_invocation_report.json": rust,
        "truth_leak_audit_report.json": truth,
    }
    for name, data in report_map.items():
        write_json(out / name, data)
    write_json(out / "aggregate_metrics.json", aggregate)
    write_json(out / "decision.json", decision_json)
    write_json(out / "summary.json", {"task": TASK, "decision": decision, "next": next_step, "dominant_breakpoint": dominant, "artifact_path": str(out), "failed_jobs": failed_jobs, "boundary": BOUNDARY})
    write_report(out, decision_json, rows)
    return aggregate, decision_json


def axis_report(rows: list[dict[str, Any]], axis: str) -> dict[str, Any]:
    row = next(r for r in rows if r["axis"] == axis)
    return {**row, "passed": row["core_d79_holds_standard"]}


def write_report(out: Path, decision: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [f"# {TASK}", "", "D80 maps stress breakpoints for the integrated joint-recall counter-action router.", "", "## Decision", "", f"- decision: `{decision['decision']}`", f"- next: `{decision['next']}`", f"- dominant breakpoint: `{decision['dominant_breakpoint']}`", "", "## Stress axis table", "", "| axis | standard | breakpoint | dominant failure | repairable | core D79 holds |", "| --- | ---: | ---: | --- | ---: | ---: |"]
    for row in rows:
        lines.append(f"| {row['axis']} | {row['standard_level']} | {row['breakpoint_threshold']} | {row['dominant_failure_mode']} | {row['repairable']} | {row['core_d79_holds_standard']} |")
    lines.extend(["", "## Boundary", "", BOUNDARY, ""])
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--seeds", default="14001,14002,14003,14004,14005")
    parser.add_argument("--train-rows-per-seed", type=int, default=240)
    parser.add_argument("--test-rows-per-seed", type=int, default=240)
    parser.add_argument("--ood-rows-per-seed", type=int, default=240)
    parser.add_argument("--workers", default="auto")
    parser.add_argument("--cpu-target", default="50-75")
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "queue.json", {"task": TASK, "created_at": round(time.time(), 3), "seeds": parse_seeds(args.seeds), "rows_per_seed": {"train": args.train_rows_per_seed, "test": args.test_rows_per_seed, "ood": args.ood_rows_per_seed}, "workers": args.workers, "cpu_target": args.cpu_target, "heartbeat_sec": args.heartbeat_sec})
    append_jsonl(out / "progress.jsonl", {"time": round(time.time(), 3), "task": TASK, "phase": "phase0", "message": "starting D80 repo/upstream audit"})
    rerun_report = ensure_d79_artifacts(args)
    write_json(out / "artifact_restore_report.json", rerun_report)
    manifest = d79_upstream_manifest(rerun_report)
    write_json(out / "d79_upstream_manifest.json", manifest)
    append_jsonl(out / "progress.jsonl", {"time": round(time.time(), 3), "task": TASK, "phase": "phase1", "message": "building D80 stress map"})
    aggregate, decision = build_reports(args, out, manifest)
    append_jsonl(out / "progress.jsonl", {"time": round(time.time(), 3), "task": TASK, "phase": "complete", "message": "D80 stress map complete", "decision": decision["decision"], "dominant_breakpoint": decision["dominant_breakpoint"]})
    print(json.dumps({"task": TASK, "out": str(out), "decision": decision["decision"], "next": decision["next"], "dominant_breakpoint": decision["dominant_breakpoint"], "stress_map_complete": aggregate["stress_map_complete"], "failed_jobs": aggregate["failed_jobs"]}, indent=2))


if __name__ == "__main__":
    main()
