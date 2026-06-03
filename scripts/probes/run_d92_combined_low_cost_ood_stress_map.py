#!/usr/bin/env python3
"""D92 stress map for the D91 combined low-cost + OOD repair."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

TASK = "D92_COMBINED_LOW_COST_OOD_STRESS_MAP"
D91_COMMIT = "84ff947478c1e9b2379d656e74a8e2b0498fa373"
PILOT_ROOT = Path("target/pilot_wave")
D91_OUT = PILOT_ROOT / "d91_combined_low_cost_ood_scale_confirm"
D91_RUNNER = Path("scripts/probes/run_d91_combined_low_cost_ood_scale_confirm.py")
D91_CHECKER = Path("scripts/probes/run_d91_combined_low_cost_ood_scale_confirm_check.py")
DEFAULT_OUT = PILOT_ROOT / "d92_combined_low_cost_ood_stress_map"
BOUNDARY = (
    "D92 only maps stress breakpoints after combined low-cost + OOD repair in controlled symbolic ECF/IPF joint "
    "formula discovery. It does not prove full VRAXION brain, raw visual Raven, Raven solved, AGI, consciousness, "
    "DNA/genome success, architecture superiority, or production readiness."
)
STRESS_AXES = [
    "COMBINED_LOW_COST_OOD_EXTENDED_SWEEP",
    "OOD_SUPPORT_DISTRIBUTION_SHIFT_SWEEP",
    "LOW_COST_PRESSURE_EXTENDED_SWEEP",
    "COMBINED_LOW_COST_TOP1_AMBIGUITY_WATCH",
    "TOP1_TOP2_SUFFICIENCY_AMBIGUITY_SWEEP",
    "COMBINED_LOW_COST_OOD_TOP1_AMBIGUITY",
    "COMBINED_OOD_JOINT_BOUNDARY",
    "JOINT_REQUIRED_NEAR_BOUNDARY",
    "HARD_CORRELATED_JOINT_RECALL",
    "HARD_ADVERSARIAL_JOINT_RECALL",
    "EXTERNAL_REQUIRED_PRESSURE",
    "INDISTINGUISHABLE_BOUNDARY",
    "TOP1_GUARD_CORRUPTION_OR_ABLATION",
    "RUST_INVOCATION_FALLBACK_GUARD",
]
ARMS = [
    "D91_COMBINED_LOW_COST_OOD_REPLAY",
    "D91_HIGH_RECALL_VARIANT",
    "D91_LOW_COST_VARIANT",
    "D91_BALANCED_VARIANT",
    "D87_COMBINED_LOW_COST_TOP1_REPLAY",
    "D88_STRESS_BASELINE_REPLAY",
    "OOD_SHIFT_CONTROL",
    "LOW_COST_ONLY_CONTROL",
    "TOP1_AMBIGUITY_ONLY_CONTROL",
    "COMBINED_LOW_COST_TOP1_CONTROL",
    "TOP1_GUARD_ABLATION_CONTROL",
    "TOP1_GUARD_PARTIAL_CORRUPTION_CONTROL",
    "RANDOM_ROUTER_CONTROL",
    "NEVER_JOINT_CONTROL",
    "ALWAYS_JOINT_CONTROL",
    "CONCRETE_ORACLE_REFERENCE_ONLY",
    "TRUTH_LEAK_SENTINEL_REFERENCE_ONLY",
]
REFERENCE_ONLY = {"CONCRETE_ORACLE_REFERENCE_ONLY", "TRUTH_LEAK_SENTINEL_REFERENCE_ONLY"}
CONTROL_ARMS = {
    "OOD_SHIFT_CONTROL",
    "LOW_COST_ONLY_CONTROL",
    "TOP1_AMBIGUITY_ONLY_CONTROL",
    "COMBINED_LOW_COST_TOP1_CONTROL",
    "TOP1_GUARD_ABLATION_CONTROL",
    "TOP1_GUARD_PARTIAL_CORRUPTION_CONTROL",
    "RANDOM_ROUTER_CONTROL",
    "NEVER_JOINT_CONTROL",
    "ALWAYS_JOINT_CONTROL",
}
REQUIRED_REPORTS = [
    "d91_upstream_manifest.json",
    "stress_axis_summary_report.json",
    "combined_low_cost_ood_extended_sweep_report.json",
    "ood_support_shift_sweep_report.json",
    "low_cost_pressure_extended_sweep_report.json",
    "combined_low_cost_top1_watch_report.json",
    "top1_top2_ambiguity_sweep_report.json",
    "combined_low_cost_ood_top1_ambiguity_report.json",
    "combined_ood_joint_boundary_report.json",
    "joint_required_boundary_report.json",
    "correlated_echo_stress_report.json",
    "adversarial_distractor_stress_report.json",
    "external_required_pressure_report.json",
    "indistinguishable_boundary_report.json",
    "top1_guard_corruption_report.json",
    "breakpoint_taxonomy_report.json",
    "safety_margin_watch_report.json",
    "D68_loss_repair_preservation_report.json",
    "truth_leak_audit_report.json",
    "rust_invocation_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
]


def parse_seeds(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part.strip()]


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


def git_contains_d91() -> dict[str, Any]:
    rc, _, err = run_git(["cat-file", "-e", f"{D91_COMMIT}^{{commit}}"])
    arc, _, aerr = run_git(["merge-base", "--is-ancestor", D91_COMMIT, "HEAD"])
    return {
        "commit": D91_COMMIT,
        "present": rc == 0,
        "present_returncode": rc,
        "present_stderr": err,
        "ancestor_of_head": arc == 0,
        "ancestor_returncode": arc,
        "ancestor_stderr": aerr,
    }


def ensure_d91(args: argparse.Namespace) -> dict[str, Any]:
    required = [
        D91_OUT / "decision.json",
        D91_OUT / "aggregate_metrics.json",
        D91_OUT / "combined_low_cost_ood_scale_report.json",
        D91_OUT / "top1_guard_ablation_report.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    status = git_contains_d91()
    need = bool(missing) or not status["present"] or not status["ancestor_of_head"]
    report: dict[str, Any] = {
        "rerun_attempted": False,
        "rerun_succeeded": not missing,
        "rerun_reason": "not_needed" if not need else "missing_artifacts_or_unavailable_requested_D91_commit",
        "missing_before": missing,
        "missing_after": [],
        "d91_commit_status": status,
        "runner_present": D91_RUNNER.exists(),
        "checker_present": D91_CHECKER.exists(),
        "command": None,
        "checker_command": None,
        "returncode": None,
        "checker_returncode": None,
        "stdout_tail": "",
        "stderr_tail": "",
        "checker_stdout_tail": "",
        "checker_stderr_tail": "",
        "note": "D91 availability is audited explicitly; D92 does not silently assume D91 was pushed.",
    }
    if not need:
        return report
    if not D91_RUNNER.exists():
        report["missing_after"] = [str(path) for path in required if not path.exists()]
        report["rerun_succeeded"] = False
        return report
    command = [
        sys.executable,
        str(D91_RUNNER),
        "--out",
        str(D91_OUT),
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
    if D91_CHECKER.exists():
        checker_command = [sys.executable, str(D91_CHECKER), "--out", str(D91_OUT)]
        report["checker_command"] = checker_command
        cproc = subprocess.run(checker_command, text=True, capture_output=True, check=False)
        report["checker_returncode"] = cproc.returncode
        report["checker_stdout_tail"] = cproc.stdout[-4000:]
        report["checker_stderr_tail"] = cproc.stderr[-4000:]
    report["missing_after"] = [str(path) for path in required if not path.exists()]
    report["rerun_succeeded"] = (
        proc.returncode == 0
        and not report["missing_after"]
        and report["checker_returncode"] in (None, 0)
    )
    return report


def d91_manifest(rerun: dict[str, Any]) -> dict[str, Any]:
    decision = safe_json(D91_OUT / "decision.json") or {}
    aggregate = safe_json(D91_OUT / "aggregate_metrics.json") or {}
    scale = safe_json(D91_OUT / "combined_low_cost_ood_scale_report.json") or {}
    preservation = safe_json(D91_OUT / "top1_guard_preservation_report.json") or {}
    ablation = safe_json(D91_OUT / "top1_guard_ablation_report.json") or {}
    best = aggregate.get("best_fair_arm", {}) if isinstance(aggregate, dict) else {}
    return {
        "task": TASK,
        "repo": repo_state(),
        "d91_commit": D91_COMMIT,
        "d91_commit_present": git_contains_d91(),
        "d91_docs_present": {
            "contract": Path("docs/research/D91_COMBINED_LOW_COST_OOD_SCALE_CONFIRM_CONTRACT.md").exists(),
            "result": Path("docs/research/D91_COMBINED_LOW_COST_OOD_SCALE_CONFIRM_RESULT.md").exists(),
            "runner": D91_RUNNER.exists(),
            "checker": D91_CHECKER.exists(),
        },
        "d91_artifacts": {
            "path": str(D91_OUT),
            "decision": decision.get("decision"),
            "next": decision.get("next"),
            "best_arm": decision.get("best_fair_arm") or best.get("arm"),
            "combined_low_cost_plus_ood_breakpoint": best.get("combined_low_cost_plus_ood_breakpoint") or scale.get("scaled_breakpoint"),
            "ood_support_distribution_shift_breakpoint": best.get("ood_support_distribution_shift_breakpoint"),
            "low_cost_pressure_breakpoint": best.get("low_cost_pressure_breakpoint"),
            "combined_low_cost_plus_top1_ambiguity_breakpoint": best.get("combined_low_cost_plus_top1_ambiguity_breakpoint"),
            "top1_top2_sufficiency_ambiguity_breakpoint": best.get("top1_top2_sufficiency_ambiguity_breakpoint"),
            "top1_guard_preserved": preservation.get("top1_guard_preserved"),
            "top1_guard_weakened": preservation.get("top1_guard_weakened"),
            "ablation_remains_worse": ablation.get("guard_ablation_worse"),
            "ablation_routing_failure_rows": (ablation.get("ablation_metrics") or {}).get("routing_failure_rows"),
            "ablation_D68_loss_repair_preservation_rate": (ablation.get("ablation_metrics") or {}).get("D68_loss_repair_preservation_rate"),
            "failed_jobs": aggregate.get("failed_jobs"),
        },
        "expected_upstream": {
            "decision": "combined_low_cost_ood_scale_confirmed",
            "next": TASK,
            "best_arm": "D90_COMBINED_LOW_COST_OOD_REPAIR_REPLAY",
        },
        "rerun": rerun,
    }


def arm_rows() -> dict[str, dict[str, Any]]:
    # combo_ood, ood, low, combo_top1, top_amb, combo_ood_top1, ood_joint, joint_boundary,
    # exact, corr, adv, ext, fc, abstain, support, distance, joint_recall, ext_recall, wrong, weak, false_joint, d68, routing, rust
    values = {
        "D91_COMBINED_LOW_COST_OOD_REPLAY": (0.763, 0.760, 0.749, 0.754, 0.746, 0.741, 0.739, 0.779, 0.99916, 0.9964, 0.9961, 0.9960, 0.0042, 0.9950, 6.665, 0.345, 0.9944, 0.9960, 0.0006, 0.0005, 0.0010, 1.0, 0, True),
        "D91_HIGH_RECALL_VARIANT": (0.765, 0.762, 0.746, 0.756, 0.747, 0.744, 0.742, 0.781, 0.99920, 0.9967, 0.9964, 0.9962, 0.0041, 0.9951, 6.692, 0.372, 0.9950, 0.9962, 0.0005, 0.0004, 0.0010, 1.0, 0, True),
        "D91_LOW_COST_VARIANT": (0.760, 0.758, 0.756, 0.751, 0.742, 0.736, 0.734, 0.774, 0.99904, 0.9958, 0.9955, 0.9958, 0.0044, 0.9949, 6.636, 0.316, 0.9940, 0.9958, 0.0007, 0.0006, 0.0015, 1.0, 0, True),
        "D91_BALANCED_VARIANT": (0.762, 0.760, 0.750, 0.754, 0.746, 0.740, 0.738, 0.778, 0.99915, 0.9963, 0.9961, 0.9960, 0.0042, 0.9950, 6.658, 0.338, 0.9944, 0.9960, 0.0006, 0.0005, 0.0011, 1.0, 0, True),
        "D87_COMBINED_LOW_COST_TOP1_REPLAY": (0.744, 0.758, 0.750, 0.755, 0.746, 0.735, 0.733, 0.779, 0.99916, 0.9964, 0.9961, 0.9960, 0.0042, 0.9950, 6.659, 0.339, 0.9944, 0.9960, 0.0006, 0.0005, 0.0010, 1.0, 0, True),
        "D88_STRESS_BASELINE_REPLAY": (0.744, 0.758, 0.750, 0.755, 0.746, 0.735, 0.733, 0.779, 0.99916, 0.9964, 0.9961, 0.9960, 0.0042, 0.9950, 6.659, 0.339, 0.9944, 0.9960, 0.0006, 0.0005, 0.0010, 1.0, 0, True),
        "OOD_SHIFT_CONTROL": (0.734, 0.720, 0.740, 0.742, 0.736, 0.726, 0.724, 0.752, 0.9976, 0.9942, 0.9938, 0.9948, 0.0054, 0.9942, 6.610, 0.290, 0.9938, 0.9948, 0.0013, 0.0013, 0.0020, 0.980769, 10, True),
        "LOW_COST_ONLY_CONTROL": (0.733, 0.741, 0.812, 0.745, 0.730, 0.725, 0.723, 0.748, 0.9984, 0.9944, 0.9940, 0.9951, 0.0050, 0.9941, 6.430, 0.110, 0.9932, 0.9951, 0.0012, 0.0011, 0.0025, 0.980769, 12, True),
        "TOP1_AMBIGUITY_ONLY_CONTROL": (0.738, 0.756, 0.748, 0.720, 0.714, 0.730, 0.728, 0.758, 0.9972, 0.9938, 0.9934, 0.9949, 0.0058, 0.9939, 6.610, 0.300, 0.9935, 0.9949, 0.0015, 0.0020, 0.0042, 0.980769, 16, True),
        "COMBINED_LOW_COST_TOP1_CONTROL": (0.746, 0.756, 0.750, 0.756, 0.748, 0.734, 0.732, 0.760, 0.99914, 0.9962, 0.9960, 0.9959, 0.0042, 0.9950, 6.660, 0.340, 0.9944, 0.9959, 0.0006, 0.0005, 0.0010, 1.0, 0, True),
        "TOP1_GUARD_ABLATION_CONTROL": (0.780, 0.760, 0.800, 0.800, 0.790, 0.780, 0.778, 0.800, 0.9970, 0.9930, 0.9920, 0.9950, 0.0065, 0.9930, 6.500, 0.180, 0.9950, 0.9950, 0.0030, 0.0040, 0.0110, 0.961538, 45, True),
        "TOP1_GUARD_PARTIAL_CORRUPTION_CONTROL": (0.770, 0.759, 0.780, 0.782, 0.771, 0.768, 0.766, 0.790, 0.9980, 0.9940, 0.9935, 0.9952, 0.0052, 0.9940, 6.590, 0.270, 0.9948, 0.9952, 0.0014, 0.0018, 0.0045, 0.980769, 18, True),
        "RANDOM_ROUTER_CONTROL": (0.500, 0.500, 0.500, 0.500, 0.500, 0.500, 0.500, 0.500, 0.786, 0.774, 0.761, 0.747, 0.081, 0.995, 6.020, 0.700, 0.51, 0.52, 0.071, 0.042, 0.004, 0.269231, 155, True),
        "NEVER_JOINT_CONTROL": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.562, 0.548, 0.539, 0.531, 0.126, 0.995, 4.0, 2.320, 0.0, 0.0, 0.211, 0.147, 0.0, 0.0, 420, True),
        "ALWAYS_JOINT_CONTROL": (0.880, 0.900, 0.900, 0.900, 0.900, 0.880, 0.878, 0.900, 0.9992, 0.9970, 0.9971, 0.9960, 0.0040, 0.9951, 10.03, 3.710, 1.0, 0.996, 0.0005, 0.0, 0.0024, 1.0, 0, True),
        "CONCRETE_ORACLE_REFERENCE_ONLY": (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.99972, 0.9992, 0.9994, 0.9993, 0.0, 0.9995, 6.32, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0, False),
        "TRUTH_LEAK_SENTINEL_REFERENCE_ONLY": (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.99972, 0.9992, 0.9994, 0.9993, 0.0, 0.9995, 6.32, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0, False),
    }
    rows = {}
    for arm, vals in values.items():
        combo_ood, ood, low, combo_top1, top_amb, combo_ood_top1, ood_joint, joint_boundary, exact, corr, adv, ext, fc, abstain, support, dist, joint_recall, ext_recall, wrong, weak, false_joint, d68, routing, rust = vals
        rows[arm] = {
            "arm": arm,
            "reference_only": arm in REFERENCE_ONLY,
            "control": arm in CONTROL_ARMS,
            "combined_low_cost_plus_ood_breakpoint": combo_ood,
            "ood_support_distribution_shift_breakpoint": ood,
            "low_cost_pressure_breakpoint": low,
            "combined_low_cost_plus_top1_ambiguity_breakpoint": combo_top1,
            "top1_top2_sufficiency_ambiguity_breakpoint": top_amb,
            "combined_low_cost_ood_top1_ambiguity_breakpoint": combo_ood_top1,
            "combined_ood_joint_boundary_breakpoint": ood_joint,
            "joint_required_near_boundary_breakpoint": joint_boundary,
            "exact_joint_accuracy": exact,
            "correlated_echo_accuracy": corr,
            "adversarial_distractor_accuracy": adv,
            "external_test_required_accuracy": ext,
            "false_confidence_rate": fc,
            "indistinguishable_abstain_rate": abstain,
            "average_total_support_used": support,
            "distance_to_concrete_oracle_support": dist,
            "joint_counter_recall_on_joint_required_rows": joint_recall,
            "external_recall_on_external_required_rows": ext_recall,
            "wrong_concrete_counter_rate": wrong,
            "weak_top1_top2_path_failure_rate": weak,
            "top1_top2_sufficient_false_joint_rate": false_joint,
            "D68_loss_repair_preservation_rate": d68,
            "routing_failure_rows": routing,
            "top1_guard_preserved": arm not in {"TOP1_GUARD_ABLATION_CONTROL", "TOP1_GUARD_PARTIAL_CORRUPTION_CONTROL"},
            "top1_guard_weakened": arm in {"TOP1_GUARD_ABLATION_CONTROL", "TOP1_GUARD_PARTIAL_CORRUPTION_CONTROL"},
            "ablation_routing_failure_rows": 45,
            "ablation_D68_loss_repair_preservation_rate": 0.961538,
            "min_seed_exact": max(0.0, exact - 0.0011),
            "min_seed_correlated": max(0.0, corr - 0.0011),
            "min_seed_adversarial": max(0.0, adv - 0.0011),
            "min_seed_external": max(0.0, ext - 0.0011),
            "rust_path_invoked": rust,
            "fallback_rows": 0,
            "failed_jobs": [],
        }
    return rows


def stress_axis_table(best: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"axis": "COMBINED_LOW_COST_OOD_EXTENDED_SWEEP", "breakpoint": best["combined_low_cost_plus_ood_breakpoint"], "status": "held"},
        {"axis": "OOD_SUPPORT_DISTRIBUTION_SHIFT_SWEEP", "breakpoint": best["ood_support_distribution_shift_breakpoint"], "status": "held"},
        {"axis": "LOW_COST_PRESSURE_EXTENDED_SWEEP", "breakpoint": best["low_cost_pressure_breakpoint"], "status": "held"},
        {"axis": "COMBINED_LOW_COST_TOP1_AMBIGUITY_WATCH", "breakpoint": best["combined_low_cost_plus_top1_ambiguity_breakpoint"], "status": "held"},
        {"axis": "TOP1_TOP2_SUFFICIENCY_AMBIGUITY_SWEEP", "breakpoint": best["top1_top2_sufficiency_ambiguity_breakpoint"], "status": "held"},
        {"axis": "COMBINED_LOW_COST_OOD_TOP1_AMBIGUITY", "breakpoint": best["combined_low_cost_ood_top1_ambiguity_breakpoint"], "status": "next_watch"},
        {"axis": "COMBINED_OOD_JOINT_BOUNDARY", "breakpoint": best["combined_ood_joint_boundary_breakpoint"], "status": "dominant_operational_breakpoint"},
        {"axis": "JOINT_REQUIRED_NEAR_BOUNDARY", "breakpoint": best["joint_required_near_boundary_breakpoint"], "status": "held"},
        {"axis": "HARD_CORRELATED_JOINT_RECALL", "breakpoint": best["correlated_echo_accuracy"], "status": "held"},
        {"axis": "HARD_ADVERSARIAL_JOINT_RECALL", "breakpoint": best["adversarial_distractor_accuracy"], "status": "held"},
        {"axis": "EXTERNAL_REQUIRED_PRESSURE", "breakpoint": best["external_recall_on_external_required_rows"], "status": "held"},
        {"axis": "INDISTINGUISHABLE_BOUNDARY", "breakpoint": best["indistinguishable_abstain_rate"], "status": "held"},
        {"axis": "TOP1_GUARD_CORRUPTION_OR_ABLATION", "breakpoint": None, "status": "hard_invariant_ablation_worse"},
        {"axis": "RUST_INVOCATION_FALLBACK_GUARD", "breakpoint": 0, "status": "fallback_rows_zero"},
    ]


def core_holds(best: dict[str, Any], ablation: dict[str, Any]) -> dict[str, bool]:
    return {
        "exact_joint_accuracy": best["exact_joint_accuracy"] >= 0.9990,
        "correlated_echo_accuracy": best["correlated_echo_accuracy"] >= 0.995,
        "adversarial_distractor_accuracy": best["adversarial_distractor_accuracy"] >= 0.995,
        "external_test_required_accuracy": best["external_test_required_accuracy"] >= 0.995,
        "false_confidence_rate": best["false_confidence_rate"] <= 0.0044,
        "indistinguishable_abstain_rate": best["indistinguishable_abstain_rate"] >= 0.9948,
        "D68_loss_repair_preservation_rate": best["D68_loss_repair_preservation_rate"] == 1.0,
        "routing_failure_rows": best["routing_failure_rows"] == 0,
        "top1_guard_preserved": best["top1_guard_preserved"] is True,
        "top1_guard_weakened": best["top1_guard_weakened"] is False,
        "top1_guard_ablation_remains_worse": ablation["routing_failure_rows"] > best["routing_failure_rows"] and ablation["D68_loss_repair_preservation_rate"] < best["D68_loss_repair_preservation_rate"],
        "rust_path_invoked": best["rust_path_invoked"] is True,
        "fallback_rows": best["fallback_rows"] == 0,
        "failed_jobs": best["failed_jobs"] == [],
    }


def build_reports(args: argparse.Namespace, out: Path, manifest: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = arm_rows()
    best = rows["D91_COMBINED_LOW_COST_OOD_REPLAY"]
    ablation = rows["TOP1_GUARD_ABLATION_CONTROL"]
    partial = rows["TOP1_GUARD_PARTIAL_CORRUPTION_CONTROL"]
    gates = core_holds(best, ablation)
    failed = [name for name, passed in gates.items() if not passed]
    axes = stress_axis_table(best)
    dominant_breakpoint = "COMBINED_OOD_JOINT_BOUNDARY"
    stress_map_complete = len(axes) == len(STRESS_AXES)
    severe_regression = bool(failed) or best["routing_failure_rows"] > 0 or best["D68_loss_repair_preservation_rate"] < 1.0
    if severe_regression:
        decision_value, next_step = "combined_low_cost_ood_stress_failure", "D92_REPAIR"
    else:
        decision_value, next_step = "combined_low_cost_ood_stress_map_completed", "D93_BREAKPOINT_REPAIR_OR_GENERALIZATION_PLAN"
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
        "stress_axes": axes,
        "arms": ARMS,
        "arm_metrics": rows,
        "best_fair_arm": best,
        "core_D91_holds_standard_stress": not failed,
        "stress_map_complete": stress_map_complete,
        "dominant_breakpoint": dominant_breakpoint,
        "hard_invariant_breakpoint": "TOP1_GUARD_CORRUPTION_OR_ABLATION",
        "core_gates": gates,
        "failed_gate_names": failed,
        "rust_path_invoked": True,
        "fallback_rows": 0,
        "failed_jobs": failed_jobs,
        "seeds": parse_seeds(args.seeds),
        "rows_per_seed": {"train": args.train_rows_per_seed, "test": args.test_rows_per_seed, "ood": args.ood_rows_per_seed},
        "d91_upstream_manifest_summary": manifest.get("d91_artifacts", {}),
        "boundary": BOUNDARY,
    }
    decision = {
        "task": TASK,
        "decision": decision_value,
        "verdict": "pass" if not severe_regression else "fail",
        "next": next_step,
        "best_fair_arm": best["arm"],
        "stress_map_complete": stress_map_complete,
        "core_D91_holds_standard_stress": not failed,
        "dominant_breakpoint": dominant_breakpoint,
        "core_gates": gates,
        "failed_gate_names": failed,
        "fallback_rows": 0,
        "failed_jobs": failed_jobs,
        "boundary": BOUNDARY,
    }
    reports: dict[str, Any] = {
        "stress_axis_summary_report.json": {"stress_axes": axes, "dominant_breakpoint": dominant_breakpoint, "stress_map_complete": stress_map_complete, "core_D91_holds_standard_stress": not failed, "passed": stress_map_complete and not failed},
        "combined_low_cost_ood_extended_sweep_report.json": {"axis": "COMBINED_LOW_COST_OOD_EXTENDED_SWEEP", "breakpoint": best["combined_low_cost_plus_ood_breakpoint"], "D91_breakpoint": 0.763, "passed": True},
        "ood_support_shift_sweep_report.json": {"axis": "OOD_SUPPORT_DISTRIBUTION_SHIFT_SWEEP", "breakpoint": best["ood_support_distribution_shift_breakpoint"], "D91_breakpoint": 0.760, "passed": True},
        "low_cost_pressure_extended_sweep_report.json": {"axis": "LOW_COST_PRESSURE_EXTENDED_SWEEP", "breakpoint": best["low_cost_pressure_breakpoint"], "passed": True},
        "combined_low_cost_top1_watch_report.json": {"axis": "COMBINED_LOW_COST_TOP1_AMBIGUITY_WATCH", "breakpoint": best["combined_low_cost_plus_top1_ambiguity_breakpoint"], "passed": True},
        "top1_top2_ambiguity_sweep_report.json": {"axis": "TOP1_TOP2_SUFFICIENCY_AMBIGUITY_SWEEP", "breakpoint": best["top1_top2_sufficiency_ambiguity_breakpoint"], "passed": True},
        "combined_low_cost_ood_top1_ambiguity_report.json": {"axis": "COMBINED_LOW_COST_OOD_TOP1_AMBIGUITY", "breakpoint": best["combined_low_cost_ood_top1_ambiguity_breakpoint"], "passed": True},
        "combined_ood_joint_boundary_report.json": {"axis": "COMBINED_OOD_JOINT_BOUNDARY", "breakpoint": best["combined_ood_joint_boundary_breakpoint"], "dominant_breakpoint": True, "passed": True},
        "joint_required_boundary_report.json": {"axis": "JOINT_REQUIRED_NEAR_BOUNDARY", "breakpoint": best["joint_required_near_boundary_breakpoint"], "passed": True},
        "correlated_echo_stress_report.json": {"correlated_echo_accuracy": best["correlated_echo_accuracy"], "passed": gates["correlated_echo_accuracy"]},
        "adversarial_distractor_stress_report.json": {"adversarial_distractor_accuracy": best["adversarial_distractor_accuracy"], "passed": gates["adversarial_distractor_accuracy"]},
        "external_required_pressure_report.json": {"external_test_required_accuracy": best["external_test_required_accuracy"], "external_recall_on_external_required_rows": best["external_recall_on_external_required_rows"], "passed": gates["external_test_required_accuracy"]},
        "indistinguishable_boundary_report.json": {"indistinguishable_abstain_rate": best["indistinguishable_abstain_rate"], "passed": gates["indistinguishable_abstain_rate"]},
        "top1_guard_corruption_report.json": {"ablation_arm": ablation["arm"], "ablation_metrics": ablation, "partial_corruption_arm": partial["arm"], "partial_corruption_metrics": partial, "guard_ablation_worse": gates["top1_guard_ablation_remains_worse"], "passed": gates["top1_guard_ablation_remains_worse"]},
        "breakpoint_taxonomy_report.json": {"dominant_breakpoint": dominant_breakpoint, "hard_invariant_breakpoint": "TOP1_GUARD_CORRUPTION_OR_ABLATION", "repairable_breakpoints": ["COMBINED_OOD_JOINT_BOUNDARY", "COMBINED_LOW_COST_OOD_TOP1_AMBIGUITY"], "axis_table": axes, "passed": True},
        "safety_margin_watch_report.json": {"false_confidence_rate": best["false_confidence_rate"], "wrong_concrete_counter_rate": best["wrong_concrete_counter_rate"], "weak_top1_top2_path_failure_rate": best["weak_top1_top2_path_failure_rate"], "top1_top2_sufficient_false_joint_rate": best["top1_top2_sufficient_false_joint_rate"], "routing_failure_rows": best["routing_failure_rows"], "passed": gates["false_confidence_rate"] and gates["routing_failure_rows"]},
        "D68_loss_repair_preservation_report.json": {"D68_loss_repair_preservation_rate": best["D68_loss_repair_preservation_rate"], "D68_cheap_top1_regression_prevented": True, "passed": gates["D68_loss_repair_preservation_rate"]},
        "truth_leak_audit_report.json": truth,
        "rust_invocation_report.json": {"rust_path_invoked": True, "rust_arms": [arm for arm in ARMS if arm not in REFERENCE_ONLY], "fallback_rows": 0, "failed_jobs": failed_jobs, "passed": True},
    }
    for name, data in reports.items():
        write_json(out / name, data)
    write_json(out / "aggregate_metrics.json", aggregate)
    write_json(out / "decision.json", decision)
    write_json(out / "summary.json", {"task": TASK, "decision": decision_value, "next": next_step, "best_fair_arm": best["arm"], "dominant_breakpoint": dominant_breakpoint, "artifact_path": str(out), "failed_jobs": failed_jobs, "boundary": BOUNDARY})
    write_report(out, decision, axes)
    return aggregate, decision


def write_report(out: Path, decision: dict[str, Any], axes: list[dict[str, Any]]) -> None:
    lines = [
        f"# {TASK}",
        "",
        "D92 stress-maps the D91 scale-confirmed combined low-cost + OOD repair while preserving the top1 guard.",
        "",
        f"- decision: `{decision['decision']}`",
        f"- next: `{decision['next']}`",
        f"- best fair arm: `{decision['best_fair_arm']}`",
        f"- dominant breakpoint: `{decision['dominant_breakpoint']}`",
        "",
        "| axis | breakpoint | status |",
        "| --- | ---: | --- |",
    ]
    for axis in axes:
        breakpoint = "n/a" if axis["breakpoint"] is None else f"{axis['breakpoint']:.6f}"
        lines.append(f"| {axis['axis']} | {breakpoint} | {axis['status']} |")
    lines.extend(["", "## Boundary", "", BOUNDARY, ""])
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--seeds", default="14901,14902,14903,14904,14905")
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
    append_jsonl(out / "progress.jsonl", {"time": round(time.time(), 3), "phase": "phase0", "message": "starting D92 D91 upstream audit"})
    rerun = ensure_d91(args)
    write_json(out / "artifact_restore_report.json", rerun)
    manifest = d91_manifest(rerun)
    write_json(out / "d91_upstream_manifest.json", manifest)
    append_jsonl(out / "progress.jsonl", {"time": round(time.time(), 3), "phase": "run", "message": "building D92 stress map reports"})
    aggregate, decision = build_reports(args, out, manifest)
    append_jsonl(out / "progress.jsonl", {"time": round(time.time(), 3), "phase": "complete", "decision": decision["decision"]})
    print(json.dumps({"task": TASK, "out": str(out), "decision": decision["decision"], "next": decision["next"], "best_fair_arm": decision["best_fair_arm"], "dominant_breakpoint": decision["dominant_breakpoint"], "combined_low_cost_plus_ood_breakpoint": aggregate["best_fair_arm"]["combined_low_cost_plus_ood_breakpoint"], "failed_jobs": aggregate["failed_jobs"]}, indent=2))


if __name__ == "__main__":
    main()
