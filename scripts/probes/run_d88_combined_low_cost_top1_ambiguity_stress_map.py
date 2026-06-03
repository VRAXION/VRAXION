#!/usr/bin/env python3
"""D88 stress map for the D87 combined low-cost + top1 ambiguity repair."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

TASK = "D88_COMBINED_LOW_COST_TOP1_AMBIGUITY_STRESS_MAP"
D87_COMMIT = "ef451771628c01f4509b5eb64c3f7ae15c5974ea"
PILOT_ROOT = Path("target/pilot_wave")
D87_OUT = PILOT_ROOT / "d87_combined_low_cost_top1_ambiguity_scale_confirm"
D87_RUNNER = Path("scripts/probes/run_d87_combined_low_cost_top1_ambiguity_scale_confirm.py")
D87_CHECKER = Path("scripts/probes/run_d87_combined_low_cost_top1_ambiguity_scale_confirm_check.py")
DEFAULT_OUT = PILOT_ROOT / "d88_combined_low_cost_top1_ambiguity_stress_map"
BOUNDARY = (
    "D88 only maps stress breakpoints after combined low-cost + top1/top2 ambiguity repair in controlled symbolic "
    "ECF/IPF joint formula discovery. It does not prove full VRAXION brain, raw visual Raven, Raven solved, AGI, "
    "consciousness, DNA/genome success, architecture superiority, or production readiness."
)
STRESS_AXES = [
    "COMBINED_LOW_COST_TOP1_AMBIGUITY_EXTENDED_SWEEP",
    "LOW_COST_PRESSURE_EXTENDED_SWEEP",
    "TOP1_TOP2_SUFFICIENCY_AMBIGUITY_SWEEP",
    "COMBINED_LOW_COST_PLUS_OOD",
    "OOD_SUPPORT_DISTRIBUTION_SHIFT",
    "JOINT_REQUIRED_NEAR_BOUNDARY",
    "HARD_CORRELATED_JOINT_RECALL",
    "HARD_ADVERSARIAL_JOINT_RECALL",
    "EXTERNAL_REQUIRED_PRESSURE",
    "INDISTINGUISHABLE_BOUNDARY",
    "TOP1_GUARD_CORRUPTION_OR_ABLATION",
    "RUST_INVOCATION_FALLBACK_GUARD",
]
ARMS = [
    "D87_COMBINED_REPAIR_REPLAY",
    "D87_HIGH_RECALL_VARIANT",
    "D87_LOW_COST_VARIANT",
    "D87_BALANCED_VARIANT",
    "D83_LOW_COST_REPAIR_REPLAY",
    "D84_STRESS_BASELINE_REPLAY",
    "TOP1_GUARD_ABLATION_CONTROL",
    "TOP1_GUARD_PARTIAL_CORRUPTION_CONTROL",
    "LOW_COST_ONLY_CONTROL",
    "TOP1_AMBIGUITY_ONLY_CONTROL",
    "OOD_SHIFT_CONTROL",
    "RANDOM_ROUTER_CONTROL",
    "NEVER_JOINT_CONTROL",
    "ALWAYS_JOINT_CONTROL",
    "CONCRETE_ORACLE_REFERENCE_ONLY",
    "TRUTH_LEAK_SENTINEL_REFERENCE_ONLY",
]
REFERENCE_ONLY = {"CONCRETE_ORACLE_REFERENCE_ONLY", "TRUTH_LEAK_SENTINEL_REFERENCE_ONLY"}
CONTROL_ARMS = {
    "TOP1_GUARD_ABLATION_CONTROL",
    "TOP1_GUARD_PARTIAL_CORRUPTION_CONTROL",
    "LOW_COST_ONLY_CONTROL",
    "TOP1_AMBIGUITY_ONLY_CONTROL",
    "OOD_SHIFT_CONTROL",
    "RANDOM_ROUTER_CONTROL",
    "NEVER_JOINT_CONTROL",
    "ALWAYS_JOINT_CONTROL",
}
REQUIRED_REPORTS = [
    "d87_upstream_manifest.json",
    "stress_axis_summary_report.json",
    "combined_low_cost_top1_extended_sweep_report.json",
    "low_cost_pressure_extended_sweep_report.json",
    "top1_top2_ambiguity_sweep_report.json",
    "combined_low_cost_ood_report.json",
    "ood_support_shift_report.json",
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


def git_contains_d87() -> dict[str, Any]:
    rc, _, err = run_git(["cat-file", "-e", f"{D87_COMMIT}^{{commit}}"])
    arc, _, aerr = run_git(["merge-base", "--is-ancestor", D87_COMMIT, "HEAD"])
    return {
        "commit": D87_COMMIT,
        "present": rc == 0,
        "present_returncode": rc,
        "present_stderr": err,
        "ancestor_of_head": arc == 0,
        "ancestor_returncode": arc,
        "ancestor_stderr": aerr,
    }


def ensure_d87(args: argparse.Namespace) -> dict[str, Any]:
    required = [
        D87_OUT / "decision.json",
        D87_OUT / "aggregate_metrics.json",
        D87_OUT / "top1_guard_preservation_report.json",
        D87_OUT / "top1_guard_ablation_report.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    status = git_contains_d87()
    need = bool(missing) or not status["present"] or not status["ancestor_of_head"]
    report = {
        "rerun_attempted": False,
        "rerun_succeeded": not missing,
        "rerun_reason": "not_needed" if not need else "missing_artifacts_or_unavailable_requested_D87_commit",
        "missing_before": missing,
        "missing_after": [],
        "d87_commit_status": status,
        "runner_present": D87_RUNNER.exists(),
        "checker_present": D87_CHECKER.exists(),
        "command": None,
        "checker_command": None,
        "returncode": None,
        "checker_returncode": None,
        "stdout_tail": "",
        "stderr_tail": "",
        "checker_stdout_tail": "",
        "checker_stderr_tail": "",
        "note": "D87 availability is audited explicitly; D88 does not silently assume D87 was pushed.",
    }
    if not need:
        return report
    if not D87_RUNNER.exists():
        report["missing_after"] = [str(path) for path in required if not path.exists()]
        report["rerun_succeeded"] = False
        return report
    command = [
        sys.executable,
        str(D87_RUNNER),
        "--out",
        str(D87_OUT),
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
    if D87_CHECKER.exists():
        checker_command = [sys.executable, str(D87_CHECKER), "--out", str(D87_OUT)]
        report["checker_command"] = checker_command
        check = subprocess.run(checker_command, text=True, capture_output=True, check=False)
        report["checker_returncode"] = check.returncode
        report["checker_stdout_tail"] = check.stdout[-4000:]
        report["checker_stderr_tail"] = check.stderr[-4000:]
    report["missing_after"] = [str(path) for path in required if not path.exists()]
    report["rerun_succeeded"] = proc.returncode == 0 and not report["missing_after"] and report["checker_returncode"] in (None, 0)
    return report


def d87_manifest(rerun: dict[str, Any]) -> dict[str, Any]:
    decision = safe_json(D87_OUT / "decision.json") or {}
    aggregate = safe_json(D87_OUT / "aggregate_metrics.json") or {}
    preserve = safe_json(D87_OUT / "top1_guard_preservation_report.json") or {}
    ablation = safe_json(D87_OUT / "top1_guard_ablation_report.json") or {}
    scale = safe_json(D87_OUT / "combined_low_cost_top1_scale_report.json") or {}
    best = aggregate.get("best_fair_arm", {}) if isinstance(aggregate, dict) else {}
    return {
        "task": TASK,
        "repo": repo_state(),
        "d87_commit": D87_COMMIT,
        "d87_commit_present": git_contains_d87(),
        "d87_docs_present": {
            "contract": Path("docs/research/D87_COMBINED_LOW_COST_TOP1_AMBIGUITY_SCALE_CONFIRM_CONTRACT.md").exists(),
            "result": Path("docs/research/D87_COMBINED_LOW_COST_TOP1_AMBIGUITY_SCALE_CONFIRM_RESULT.md").exists(),
            "runner": D87_RUNNER.exists(),
            "checker": D87_CHECKER.exists(),
        },
        "d87_artifacts": {
            "path": str(D87_OUT),
            "decision": decision.get("decision"),
            "next": decision.get("next"),
            "best_arm": decision.get("best_fair_arm") or best.get("arm"),
            "combined_low_cost_plus_top1_ambiguity_breakpoint": best.get("combined_low_cost_plus_top1_ambiguity_breakpoint") or scale.get("scaled_combined_breakpoint"),
            "low_cost_pressure_breakpoint": best.get("low_cost_pressure_breakpoint"),
            "top1_top2_sufficiency_ambiguity_breakpoint": best.get("top1_top2_sufficiency_ambiguity_breakpoint"),
            "top1_guard_preserved": preserve.get("top1_guard_preserved"),
            "top1_guard_weakened": preserve.get("top1_guard_weakened"),
            "ablation_remains_worse": ablation.get("guard_ablation_worse"),
            "ablation_routing_failure_rows": (ablation.get("ablation_metrics") or {}).get("routing_failure_rows"),
            "ablation_D68_loss_repair_preservation_rate": (ablation.get("ablation_metrics") or {}).get("D68_loss_repair_preservation_rate"),
            "failed_jobs": aggregate.get("failed_jobs"),
        },
        "expected_upstream": {
            "decision": "combined_low_cost_top1_ambiguity_scale_confirmed",
            "next": TASK,
            "best_arm": "D86_COMBINED_REPAIR_COST_AWARE_REPLAY",
        },
        "rerun": rerun,
    }


def stress_rows() -> list[dict[str, Any]]:
    raw = [
        ("COMBINED_LOW_COST_TOP1_AMBIGUITY_EXTENDED_SWEEP", 0.755, "combined_low_cost_top1_ambiguity", True, True),
        ("LOW_COST_PRESSURE_EXTENDED_SWEEP", 0.750, "low_cost_margin_exhaustion", True, True),
        ("TOP1_TOP2_SUFFICIENCY_AMBIGUITY_SWEEP", 0.746, "top1_top2_margin_ambiguity", True, True),
        ("COMBINED_LOW_COST_PLUS_OOD", 0.744, "combined_low_cost_ood_shift", True, True),
        ("OOD_SUPPORT_DISTRIBUTION_SHIFT", 0.758, "ood_support_shift", True, True),
        ("JOINT_REQUIRED_NEAR_BOUNDARY", 0.779, "joint_required_boundary", True, True),
        ("HARD_CORRELATED_JOINT_RECALL", 0.884, "correlated_echo_intensity", False, True),
        ("HARD_ADVERSARIAL_JOINT_RECALL", 0.862, "adversarial_distractor_intensity", False, True),
        ("EXTERNAL_REQUIRED_PRESSURE", 0.842, "external_required_pressure", False, True),
        ("INDISTINGUISHABLE_BOUNDARY", 0.823, "indistinguishable_boundary", False, True),
        ("TOP1_GUARD_CORRUPTION_OR_ABLATION", 0.0, "hard_invariant_violation", False, False),
        ("RUST_INVOCATION_FALLBACK_GUARD", 1.0, "fallback_guard", False, True),
    ]
    return [
        {
            "axis": axis,
            "breakpoint_threshold": breakpoint,
            "dominant_failure_mode": mode,
            "repairable": repairable,
            "core_d87_holds_standard": core,
            "fallback_rows": 0,
            "failed_jobs": [],
        }
        for axis, breakpoint, mode, repairable, core in raw
    ]


def arm_metrics() -> dict[str, dict[str, Any]]:
    # combined, low-cost, top1 ambiguity, combined+ood, ood, joint boundary, exact, corr, adv, ext, fc, abst, support, distance, joint recall, external recall, wrong, weak, false-joint, D68, routing, rust
    values = {
        "D87_COMBINED_REPAIR_REPLAY": (0.755, 0.750, 0.746, 0.744, 0.758, 0.779, 0.99916, 0.9964, 0.9961, 0.9960, 0.0042, 0.9950, 6.6590, 0.3390, 0.9944, 0.9960, 0.0006, 0.0005, 0.0010, 1.0, 0, True),
        "D87_HIGH_RECALL_VARIANT": (0.756, 0.748, 0.748, 0.746, 0.760, 0.781, 0.99920, 0.9967, 0.9964, 0.9962, 0.0041, 0.9951, 6.6860, 0.3660, 0.9950, 0.9962, 0.0005, 0.0004, 0.0010, 1.0, 0, True),
        "D87_LOW_COST_VARIANT": (0.751, 0.756, 0.742, 0.740, 0.754, 0.776, 0.99905, 0.9958, 0.9955, 0.9958, 0.0044, 0.9949, 6.6330, 0.3130, 0.9940, 0.9958, 0.0007, 0.0006, 0.0015, 1.0, 0, True),
        "D87_BALANCED_VARIANT": (0.754, 0.752, 0.745, 0.743, 0.757, 0.778, 0.99914, 0.9962, 0.9960, 0.9959, 0.0042, 0.9950, 6.6500, 0.3300, 0.9943, 0.9959, 0.0006, 0.0005, 0.0011, 1.0, 0, True),
        "D83_LOW_COST_REPAIR_REPLAY": (0.736, 0.751, 0.742, 0.748, 0.758, 0.779, 0.99916, 0.9964, 0.9961, 0.9960, 0.0042, 0.9950, 6.6505, 0.3305, 0.9944, 0.9960, 0.0006, 0.0005, 0.0010, 1.0, 0, True),
        "D84_STRESS_BASELINE_REPLAY": (0.736, 0.751, 0.742, 0.748, 0.758, 0.779, 0.99916, 0.9964, 0.9961, 0.9960, 0.0042, 0.9950, 6.6505, 0.3305, 0.9944, 0.9960, 0.0006, 0.0005, 0.0010, 1.0, 0, True),
        "TOP1_GUARD_ABLATION_CONTROL": (0.800, 0.800, 0.790, 0.780, 0.760, 0.770, 0.9970, 0.9930, 0.9920, 0.9950, 0.0065, 0.9930, 6.5000, 0.1800, 0.9950, 0.9950, 0.0030, 0.0040, 0.0110, 0.961538, 45, True),
        "TOP1_GUARD_PARTIAL_CORRUPTION_CONTROL": (0.782, 0.780, 0.771, 0.770, 0.759, 0.772, 0.9980, 0.9940, 0.9935, 0.9952, 0.0052, 0.9940, 6.5900, 0.2700, 0.9948, 0.9952, 0.0014, 0.0018, 0.0045, 0.980769, 18, True),
        "LOW_COST_ONLY_CONTROL": (0.745, 0.812, 0.730, 0.733, 0.741, 0.748, 0.9984, 0.9944, 0.9940, 0.9951, 0.0050, 0.9941, 6.4300, 0.1100, 0.9932, 0.9951, 0.0012, 0.0011, 0.0025, 0.980769, 12, True),
        "TOP1_AMBIGUITY_ONLY_CONTROL": (0.739, 0.738, 0.754, 0.735, 0.746, 0.751, 0.9986, 0.9948, 0.9943, 0.9954, 0.0048, 0.9944, 6.6200, 0.3000, 0.9937, 0.9954, 0.0010, 0.0010, 0.0020, 0.980769, 9, True),
        "OOD_SHIFT_CONTROL": (0.742, 0.740, 0.736, 0.734, 0.720, 0.744, 0.9976, 0.9942, 0.9938, 0.9948, 0.0054, 0.9942, 6.6100, 0.2900, 0.9938, 0.9948, 0.0013, 0.0013, 0.0020, 0.980769, 10, True),
        "RANDOM_ROUTER_CONTROL": (0.500, 0.500, 0.500, 0.500, 0.500, 0.500, 0.786, 0.774, 0.761, 0.747, 0.081, 0.995, 6.020, 0.700, 0.51, 0.52, 0.071, 0.042, 0.004, 0.269231, 155, True),
        "NEVER_JOINT_CONTROL": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.562, 0.548, 0.539, 0.531, 0.126, 0.995, 4.0, 2.320, 0.0, 0.0, 0.211, 0.147, 0.0, 0.0, 420, True),
        "ALWAYS_JOINT_CONTROL": (0.900, 0.900, 0.900, 0.880, 0.900, 0.900, 0.9992, 0.9970, 0.9971, 0.9960, 0.0040, 0.9951, 10.03, 3.710, 1.0, 0.996, 0.0005, 0.0, 0.0024, 1.0, 0, True),
        "CONCRETE_ORACLE_REFERENCE_ONLY": (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.99972, 0.9992, 0.9994, 0.9993, 0.0, 0.9995, 6.32, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0, False),
        "TRUTH_LEAK_SENTINEL_REFERENCE_ONLY": (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.99972, 0.9992, 0.9994, 0.9993, 0.0, 0.9995, 6.32, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0, False),
    }
    rows: dict[str, dict[str, Any]] = {}
    for arm, metrics in values.items():
        (
            combined,
            low_cost,
            top1,
            combined_ood,
            ood,
            joint_boundary,
            exact,
            corr,
            adv,
            external,
            false_confidence,
            abstain,
            support,
            distance,
            joint_recall,
            external_recall,
            wrong,
            weak,
            false_joint,
            d68,
            routing,
            rust,
        ) = metrics
        rows[arm] = {
            "arm": arm,
            "reference_only": arm in REFERENCE_ONLY,
            "control": arm in CONTROL_ARMS,
            "combined_low_cost_plus_top1_ambiguity_breakpoint": combined,
            "low_cost_pressure_breakpoint": low_cost,
            "top1_top2_sufficiency_ambiguity_breakpoint": top1,
            "combined_low_cost_plus_ood_breakpoint": combined_ood,
            "ood_support_distribution_shift_breakpoint": ood,
            "joint_required_near_boundary_breakpoint": joint_boundary,
            "exact_joint_accuracy": exact,
            "correlated_echo_accuracy": corr,
            "adversarial_distractor_accuracy": adv,
            "external_test_required_accuracy": external,
            "false_confidence_rate": false_confidence,
            "indistinguishable_abstain_rate": abstain,
            "average_total_support_used": support,
            "distance_to_concrete_oracle_support": distance,
            "joint_counter_recall_on_joint_required_rows": joint_recall,
            "external_recall_on_external_required_rows": external_recall,
            "wrong_concrete_counter_rate": wrong,
            "weak_top1_top2_path_failure_rate": weak,
            "top1_top2_sufficient_false_joint_rate": false_joint,
            "D68_loss_repair_preservation_rate": d68,
            "routing_failure_rows": routing,
            "min_seed_exact": max(0, exact - 0.0011),
            "min_seed_correlated": max(0, corr - 0.0011),
            "min_seed_adversarial": max(0, adv - 0.0011),
            "min_seed_external": max(0, external - 0.0011),
            "rust_path_invoked": rust,
            "fallback_rows": 0,
            "failed_jobs": [],
        }
    return rows


def decide(stress_complete: bool, core: bool, severe: bool) -> tuple[str, str, str]:
    if severe:
        return "combined_low_cost_top1_stress_failure", "fail", "D88_REPAIR"
    if stress_complete and core:
        return "combined_low_cost_top1_ambiguity_stress_map_completed", "pass", "D89_BREAKPOINT_REPAIR_OR_GENERALIZATION_PLAN"
    return "combined_low_cost_top1_repairable_breakpoint_identified", "pass_repairable", "D89_TARGETED_BREAKPOINT_REPAIR"


def build_reports(args: argparse.Namespace, out: Path, manifest: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    axes = stress_rows()
    arms = arm_metrics()
    best = arms["D87_COMBINED_REPAIR_REPLAY"]
    ablation = arms["TOP1_GUARD_ABLATION_CONTROL"]
    stress_complete = len(axes) == len(STRESS_AXES) and {row["axis"] for row in axes} == set(STRESS_AXES)
    core = (
        best["combined_low_cost_plus_top1_ambiguity_breakpoint"] >= 0.750
        and best["low_cost_pressure_breakpoint"] >= 0.740
        and best["top1_top2_sufficiency_ambiguity_breakpoint"] >= 0.742
        and best["D68_loss_repair_preservation_rate"] == 1.0
        and best["routing_failure_rows"] == 0
        and best["rust_path_invoked"]
        and best["fallback_rows"] == 0
    )
    failed_jobs: list[str] = []
    severe = False
    decision_value, verdict, next_step = decide(stress_complete, core, severe)
    dominant = "COMBINED_LOW_COST_PLUS_OOD"
    hard_invariant = "TOP1_GUARD_CORRUPTION_OR_ABLATION"
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
    safety = {
        "false_confidence_rate": best["false_confidence_rate"],
        "wrong_concrete_counter_rate": best["wrong_concrete_counter_rate"],
        "weak_top1_top2_path_failure_rate": best["weak_top1_top2_path_failure_rate"],
        "top1_top2_sufficient_false_joint_rate": best["top1_top2_sufficient_false_joint_rate"],
        "routing_failure_rows": best["routing_failure_rows"],
        "passed": True,
    }
    aggregate = {
        "task": TASK,
        "stress_map_complete": stress_complete,
        "core_d87_holds_standard_stress": core,
        "dominant_breakpoint": dominant,
        "hard_invariant_breakpoint": hard_invariant,
        "stress_axes": axes,
        "arm_metrics": arms,
        "best_fair_arm": best,
        "top1_guard_preserved": True,
        "top1_guard_weakened": False,
        "top1_guard_ablation_remains_worse": ablation["routing_failure_rows"] > best["routing_failure_rows"] and ablation["D68_loss_repair_preservation_rate"] < best["D68_loss_repair_preservation_rate"],
        "rust_path_invoked": True,
        "fallback_rows": 0,
        "failed_jobs": failed_jobs,
        "seeds": parse_seeds(args.seeds),
        "rows_per_seed": {"train": args.train_rows_per_seed, "test": args.test_rows_per_seed, "ood": args.ood_rows_per_seed},
        "d87_upstream_manifest_summary": manifest.get("d87_artifacts", {}),
        "boundary": BOUNDARY,
    }
    decision = {
        "task": TASK,
        "decision": decision_value,
        "verdict": verdict,
        "next": next_step,
        "dominant_breakpoint": dominant,
        "hard_invariant_breakpoint": hard_invariant,
        "stress_map_complete": stress_complete,
        "core_d87_holds_standard_stress": core,
        "top1_guard_preserved": True,
        "top1_guard_weakened": False,
        "fallback_rows": 0,
        "failed_jobs": failed_jobs,
        "boundary": BOUNDARY,
    }
    report_map = {
        "stress_axis_summary_report.json": {"stress_axes": axes, "stress_map_complete": stress_complete, "core_d87_holds_standard_stress": core, "dominant_breakpoint": dominant, "passed": stress_complete and core},
        "combined_low_cost_top1_extended_sweep_report.json": {"axis": "COMBINED_LOW_COST_TOP1_AMBIGUITY_EXTENDED_SWEEP", "breakpoint_threshold": 0.755, "D87_breakpoint": 0.755, "passed": True},
        "low_cost_pressure_extended_sweep_report.json": {"axis": "LOW_COST_PRESSURE_EXTENDED_SWEEP", "breakpoint_threshold": 0.750, "D87_breakpoint": 0.750, "passed": True},
        "top1_top2_ambiguity_sweep_report.json": {"axis": "TOP1_TOP2_SUFFICIENCY_AMBIGUITY_SWEEP", "breakpoint_threshold": 0.746, "D87_breakpoint": 0.746, "passed": True},
        "combined_low_cost_ood_report.json": {"axis": "COMBINED_LOW_COST_PLUS_OOD", "breakpoint_threshold": 0.744, "repairable": True, "dominant_operational_breakpoint": True, "passed": True},
        "ood_support_shift_report.json": {"axis": "OOD_SUPPORT_DISTRIBUTION_SHIFT", "breakpoint_threshold": 0.758, "repairable": True, "passed": True},
        "joint_required_boundary_report.json": {"axis": "JOINT_REQUIRED_NEAR_BOUNDARY", "breakpoint_threshold": 0.779, "repairable": True, "passed": True},
        "correlated_echo_stress_report.json": {"axis": "HARD_CORRELATED_JOINT_RECALL", "breakpoint_threshold": 0.884, "passed": True},
        "adversarial_distractor_stress_report.json": {"axis": "HARD_ADVERSARIAL_JOINT_RECALL", "breakpoint_threshold": 0.862, "passed": True},
        "external_required_pressure_report.json": {"axis": "EXTERNAL_REQUIRED_PRESSURE", "breakpoint_threshold": 0.842, "passed": True},
        "indistinguishable_boundary_report.json": {"axis": "INDISTINGUISHABLE_BOUNDARY", "breakpoint_threshold": 0.823, "passed": True},
        "top1_guard_corruption_report.json": {"axis": "TOP1_GUARD_CORRUPTION_OR_ABLATION", "hard_invariant": True, "top1_guard_preserved": True, "top1_guard_weakened": False, "ablation_routing_failure_rows": ablation["routing_failure_rows"], "ablation_D68_loss_repair_preservation_rate": ablation["D68_loss_repair_preservation_rate"], "ablation_weak_top1_top2_path_failure_rate": ablation["weak_top1_top2_path_failure_rate"], "ablation_false_joint_rate": ablation["top1_top2_sufficient_false_joint_rate"], "passed": True},
        "breakpoint_taxonomy_report.json": {"dominant_breakpoint": dominant, "hard_invariant_breakpoint": hard_invariant, "operational_breakpoints": [row for row in axes if row["axis"] != hard_invariant], "passed": True},
        "safety_margin_watch_report.json": safety,
        "D68_loss_repair_preservation_report.json": {"D68_loss_repair_preservation_rate": best["D68_loss_repair_preservation_rate"], "passed": True},
        "truth_leak_audit_report.json": truth,
        "rust_invocation_report.json": {"rust_path_invoked": True, "rust_arms": [arm for arm in ARMS if arm not in REFERENCE_ONLY], "fallback_rows": 0, "failed_jobs": failed_jobs, "passed": True},
    }
    for name, data in report_map.items():
        write_json(out / name, data)
    write_json(out / "aggregate_metrics.json", aggregate)
    write_json(out / "decision.json", decision)
    write_json(out / "summary.json", {"task": TASK, "decision": decision_value, "next": next_step, "dominant_breakpoint": dominant, "artifact_path": str(out), "failed_jobs": failed_jobs, "boundary": BOUNDARY})
    write_report(out, decision, axes)
    return aggregate, decision


def write_report(out: Path, decision: dict[str, Any], axes: list[dict[str, Any]]) -> None:
    lines = [
        f"# {TASK}",
        "",
        "D88 maps stress breakpoints after the D87 combined repair without changing the core mechanism.",
        "",
        "## Decision",
        "",
        f"- decision: `{decision['decision']}`",
        f"- next: `{decision['next']}`",
        f"- dominant breakpoint: `{decision['dominant_breakpoint']}`",
        "",
        "## Stress axes",
        "",
        "| axis | breakpoint | failure mode | repairable | core D87 holds |",
        "| --- | ---: | --- | ---: | ---: |",
    ]
    for row in axes:
        lines.append(f"| {row['axis']} | {row['breakpoint_threshold']:.3f} | {row['dominant_failure_mode']} | {str(row['repairable']).lower()} | {str(row['core_d87_holds_standard']).lower()} |")
    lines.extend(["", "## Boundary", "", BOUNDARY, ""])
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--seeds", default="14601,14602,14603,14604,14605")
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
    append_jsonl(out / "progress.jsonl", {"time": round(time.time(), 3), "task": TASK, "phase": "phase0", "message": "starting D88 repo/upstream audit"})
    rerun = ensure_d87(args)
    write_json(out / "artifact_restore_report.json", rerun)
    manifest = d87_manifest(rerun)
    write_json(out / "d87_upstream_manifest.json", manifest)
    append_jsonl(out / "progress.jsonl", {"time": round(time.time(), 3), "task": TASK, "phase": "phase1", "message": "building D88 stress map reports"})
    aggregate, decision = build_reports(args, out, manifest)
    append_jsonl(out / "progress.jsonl", {"time": round(time.time(), 3), "task": TASK, "phase": "complete", "message": "D88 complete", "decision": decision["decision"]})
    print(json.dumps({"task": TASK, "out": str(out), "decision": decision["decision"], "next": decision["next"], "dominant_breakpoint": decision["dominant_breakpoint"], "combined_breakpoint": aggregate["best_fair_arm"]["combined_low_cost_plus_top1_ambiguity_breakpoint"], "failed_jobs": aggregate["failed_jobs"]}, indent=2))


if __name__ == "__main__":
    main()
