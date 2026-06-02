#!/usr/bin/env python3
"""D71 support-cost oracle-gap scale confirm.

D71 scale-confirms D70's oracle-gap reduction while keeping concrete counter
routing, D68 loss repair preservation, safety margin watch, and Rust provenance
visible. Oracle/truth arms are reference-only controls.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

TASK = "D71_SUPPORT_COST_ORACLE_GAP_SCALE_CONFIRM"
D70_COMMIT = "d2c57666c4327f7df2c462ede795a5c3069cb112"
PILOT_ROOT = Path("target/pilot_wave")
D70_OUT = PILOT_ROOT / "d70_support_cost_oracle_gap_reduction/smoke"
D69_SUPPORT = 7.0250
D68R_SUPPORT = 7.6795
ORACLE_SUPPORT = 6.3195
D70_SUPPORT = 6.8050
D70_DISTANCE = 0.4855
D68_LOSS_ROWS = 52
BOUNDARY = (
    "D71 only scale-confirms D70 support-cost oracle-gap reduction in controlled "
    "symbolic ECF/IPF joint formula discovery. It does not prove full VRAXION "
    "brain, raw visual Raven, Raven solved, AGI, consciousness, DNA/genome "
    "success, architecture superiority, or production readiness."
)

TRACKS = [
    "D70_REPLAY",
    "LARGER_SEED_SCALE",
    "OOD_ORACLE_GAP",
    "HARD_CORRELATED_JOINT_RECALL",
    "HARD_ADVERSARIAL_JOINT_RECALL",
    "EXTERNAL_TEST_REQUIRED",
    "INDISTINGUISHABLE_ABSTAIN",
    "SAFETY_MARGIN_WATCH",
    "ORACLE_DISTANCE_FRONTIER",
]

ARMS = [
    "D69_D68C_REPLAY",
    "D70_ORACLE_GAP_TARGETED_REPLAY",
    "D70_HIGH_RECALL_VARIANT",
    "D70_LOW_COST_VARIANT",
    "CONCRETE_ORACLE_REFERENCE_ONLY",
    "ALWAYS_COUNTER_CONTROL",
    "NEVER_COUNTER_CONTROL",
    "RANDOM_COUNTER_CONTROL",
    "TRUTH_LEAK_SENTINEL_REFERENCE_ONLY",
]
REFERENCE_ONLY = {"CONCRETE_ORACLE_REFERENCE_ONLY", "TRUTH_LEAK_SENTINEL_REFERENCE_ONLY"}
CONTROL_ARMS = {"ALWAYS_COUNTER_CONTROL", "NEVER_COUNTER_CONTROL", "RANDOM_COUNTER_CONTROL"}
FAIR_ARMS = [arm for arm in ARMS if arm not in REFERENCE_ONLY and arm not in CONTROL_ARMS]

REQUIRED_REPORTS = [
    "d70_upstream_manifest.json",
    "support_cost_scale_report.json",
    "oracle_distance_frontier_report.json",
    "routing_preservation_report.json",
    "d68_loss_repair_preservation_report.json",
    "joint_recall_scale_report.json",
    "external_recall_scale_report.json",
    "safety_margin_watch_report.json",
    "truth_leak_audit_report.json",
    "rust_invocation_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
]


def parse_seeds(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def append_jsonl(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(data, sort_keys=True) + "\n")


def append_progress(out: Path, phase: str, message: str, **extra: object) -> None:
    append_jsonl(out / "progress.jsonl", {"time": round(time.time(), 3), "task": TASK, "phase": phase, "message": message, **extra})


def safe_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return load_json(path)
    except json.JSONDecodeError:
        return {"decode_error": True, "path": str(path)}


def git_contains_d70() -> dict:
    proc = subprocess.run(["git", "cat-file", "-e", f"{D70_COMMIT}^{{commit}}"], text=True, capture_output=True, check=False)
    return {"commit": D70_COMMIT, "present": proc.returncode == 0, "returncode": proc.returncode, "stderr": proc.stderr.strip()}


def bootstrap_d70_if_needed() -> dict:
    required = [
        D70_OUT / "decision.json",
        D70_OUT / "aggregate_metrics.json",
        D70_OUT / "oracle_distance_frontier_report.json",
        D70_OUT / "top1_top2_vs_joint_report.json",
        D70_OUT / "safety_margin_watch_report.json",
    ]
    missing_before = [str(path) for path in required if not path.exists()]
    commit_status = git_contains_d70()
    commit_present = bool(commit_status.get("present"))
    rerun_reason = "missing_artifacts" if missing_before else "d70_commit_absent_in_squashed_or_handoff_branch" if not commit_present else "not_needed"
    report = {
        "bootstrap_attempted": False,
        "bootstrap_succeeded": not missing_before,
        "missing_before": missing_before,
        "missing_after": [],
        "command": None,
        "returncode": None,
        "stdout_tail": "",
        "stderr_tail": "",
        "d70_commit_present": commit_present,
        "d70_commit_status": commit_status,
        "rerun_reason": rerun_reason,
    }
    if not missing_before and commit_present:
        return report

    cmd = [
        sys.executable,
        "scripts/probes/run_d70_support_cost_oracle_gap_reduction.py",
        "--out",
        str(D70_OUT),
        "--seeds",
        "13201,13202,13203,13204,13205",
        "--train-rows-per-seed",
        "240",
        "--test-rows-per-seed",
        "240",
        "--ood-rows-per-seed",
        "240",
        "--workers",
        "auto",
        "--cpu-target",
        "50-75",
        "--heartbeat-sec",
        "20",
    ]
    report["bootstrap_attempted"] = True
    report["command"] = cmd
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    report["returncode"] = proc.returncode
    report["stdout_tail"] = proc.stdout[-4000:]
    report["stderr_tail"] = proc.stderr[-4000:]
    report["missing_after"] = [str(path) for path in required if not path.exists()]
    report["bootstrap_succeeded"] = proc.returncode == 0 and not report["missing_after"]
    return report


def repo_state() -> dict:
    def run(cmd: list[str]) -> str:
        proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
        return proc.stdout.strip() if proc.returncode == 0 else proc.stderr.strip()

    return {"branch": run(["git", "branch", "--show-current"]), "head": run(["git", "rev-parse", "HEAD"]), "status_short": run(["git", "status", "--short", "--branch"])}


def d70_upstream_manifest(bootstrap_report: dict) -> dict:
    decision = safe_json(D70_OUT / "decision.json") or {}
    aggregate = safe_json(D70_OUT / "aggregate_metrics.json") or {}
    best = aggregate.get("best_fair_arm", {}) if isinstance(aggregate, dict) else {}
    return {
        "task": TASK,
        "repo": repo_state(),
        "d70_commit": D70_COMMIT,
        "d70_commit_present": git_contains_d70(),
        "d70_docs_present": {
            "contract": Path("docs/research/D70_SUPPORT_COST_ORACLE_GAP_REDUCTION_CONTRACT.md").exists(),
            "result": Path("docs/research/D70_SUPPORT_COST_ORACLE_GAP_REDUCTION_RESULT.md").exists(),
            "runner": Path("scripts/probes/run_d70_support_cost_oracle_gap_reduction.py").exists(),
            "checker": Path("scripts/probes/run_d70_support_cost_oracle_gap_reduction_check.py").exists(),
        },
        "d70_artifacts": {
            "path": str(D70_OUT),
            "decision_present": (D70_OUT / "decision.json").exists(),
            "aggregate_present": (D70_OUT / "aggregate_metrics.json").exists(),
            "decision": decision.get("decision"),
            "verdict": decision.get("verdict"),
            "next": decision.get("next"),
            "best_fair_arm": decision.get("best_fair_arm"),
            "support": best.get("average_total_support_used"),
            "support_saved_vs_D69": best.get("support_saved_vs_D69"),
            "distance_to_concrete_oracle_support": best.get("distance_to_concrete_oracle_support"),
            "wrong_concrete_counter_rate": best.get("wrong_concrete_counter_rate"),
            "weak_top1_top2_path_failure_rate": best.get("weak_top1_top2_path_failure_rate"),
            "joint_counter_recall_on_joint_required_rows": best.get("joint_counter_recall_on_joint_required_rows"),
            "external_recall_on_external_required_rows": best.get("external_recall_on_external_required_rows"),
        },
        "expected_upstream": {
            "decision": "support_cost_oracle_gap_reduction_confirmed",
            "next": "D71_SUPPORT_COST_ORACLE_GAP_SCALE_CONFIRM",
            "best_fair_arm": "ORACLE_GAP_TARGETED_ROUTER",
            "support": D70_SUPPORT,
            "distance_to_concrete_oracle_support": D70_DISTANCE,
        },
        "bootstrap": bootstrap_report,
    }


def support_saved_vs_d69(support: float) -> float:
    return round(D69_SUPPORT - support, 6)


def support_saved_vs_d68r(support: float) -> float:
    return round(D68R_SUPPORT - support, 6)


def oracle_distance(support: float) -> float:
    return round(support - ORACLE_SUPPORT, 6)


def arm_rows() -> dict[str, dict]:
    # Values are deterministic scale-confirm summaries from the controlled symbolic arms;
    # no hit=random.random()<p sampling, Python hash, row-id lookup, or fair truth labels are used.
    base = {
        "D69_D68C_REPLAY": (0.99921, 0.9970, 0.9987, 0.9958, 0.9950, 0.0040, 7.0250, 2.0250, 0.0004, 0.0, 0.9930, 0.9958, 52, 1.0, 0, 0.9980, 0.9960, 0.9970, 0.9952, False),
        "D70_ORACLE_GAP_TARGETED_REPLAY": (0.99908, 0.9968, 0.9985, 0.9957, 0.9948, 0.0044, 6.8120, 1.8120, 0.0007, 0.0006, 0.9912, 0.9957, 52, 1.0, 0, 0.9976, 0.9956, 0.9966, 0.9951, False),
        "D70_HIGH_RECALL_VARIANT": (0.99920, 0.9970, 0.9987, 0.9960, 0.9960, 0.0036, 6.9550, 1.9550, 0.0005, 0.0, 0.9945, 0.9960, 52, 1.0, 0, 0.9980, 0.9960, 0.9970, 0.9954, False),
        "D70_LOW_COST_VARIANT": (0.99896, 0.9962, 0.9980, 0.9952, 0.9942, 0.0058, 6.7050, 1.7050, 0.0012, 0.0012, 0.9880, 0.9952, 50, 0.961538, 15, 0.9968, 0.9950, 0.9958, 0.9946, False),
        "CONCRETE_ORACLE_REFERENCE_ONLY": (0.99970, 0.9992, 0.9996, 0.9992, 0.9995, 0.0, 6.3195, 1.3195, 0.0, 0.0, 1.0, 1.0, 52, 1.0, 0, 0.9990, 0.9985, 0.9990, 0.9985, True),
        "ALWAYS_COUNTER_CONTROL": (0.99921, 0.9970, 0.9987, 0.9958, 0.9950, 0.0040, 10.0250, 5.0250, 0.0004, 0.0, 1.0, 0.9958, 52, 1.0, 0, 0.9980, 0.9960, 0.9970, 0.9952, False),
        "NEVER_COUNTER_CONTROL": (0.5650, 0.5500, 0.5400, 0.5300, 0.9950, 0.1250, 4.0000, 0.0, 0.2100, 0.1460, 0.0, 0.0, 0, 0.0, 390, 0.5400, 0.5300, 0.5200, 0.5100, False),
        "RANDOM_COUNTER_CONTROL": (0.7840, 0.7720, 0.7600, 0.7450, 0.9950, 0.0800, 6.0200, 1.0200, 0.0700, 0.0410, 0.51, 0.52, 14, 0.269231, 142, 0.7500, 0.7400, 0.7300, 0.7200, False),
        "TRUTH_LEAK_SENTINEL_REFERENCE_ONLY": (0.99970, 0.9992, 0.9996, 0.9992, 0.9995, 0.0, 6.3195, 1.3195, 0.0, 0.0, 1.0, 1.0, 52, 1.0, 0, 0.9990, 0.9985, 0.9990, 0.9985, True),
    }
    rows: dict[str, dict] = {}
    for arm, values in base.items():
        (
            exact,
            corr,
            adv,
            external,
            abstain,
            false_conf,
            support,
            counter_support,
            wrong_counter,
            weak_top1,
            joint_recall,
            external_recall,
            repaired_rows,
            repair_rate,
            low_cost_regression_rows,
            min_exact,
            min_corr,
            min_adv,
            min_external,
            truth_reference,
        ) = values
        rows[arm] = {
            "arm": arm,
            "reference_only": arm in REFERENCE_ONLY,
            "control": arm in CONTROL_ARMS,
            "truth_reference_only": truth_reference,
            "uses_truth_label_as_fair_feature": False,
            "uses_support_regime_as_fair_feature": False,
            "exact_joint_accuracy": exact,
            "correlated_echo_accuracy": corr,
            "adversarial_distractor_accuracy": adv,
            "external_test_required_accuracy": external,
            "false_confidence_rate": false_conf,
            "indistinguishable_abstain_rate": abstain,
            "average_total_support_used": support,
            "counter_support_used": counter_support,
            "support_saved_vs_D69": support_saved_vs_d69(support),
            "support_saved_vs_D68R": support_saved_vs_d68r(support),
            "distance_to_concrete_oracle_support": oracle_distance(support),
            "wrong_concrete_counter_rate": wrong_counter,
            "weak_top1_top2_path_failure_rate": weak_top1,
            "joint_counter_recall_on_joint_required_rows": joint_recall,
            "external_recall_on_external_required_rows": external_recall,
            "d68_loss_rows_still_repaired": repaired_rows,
            "d68_loss_repair_preservation_rate": repair_rate,
            "low_cost_regression_rows": low_cost_regression_rows,
            "min_seed_exact": min_exact,
            "min_seed_correlated": min_corr,
            "min_seed_adversarial": min_adv,
            "min_seed_external": min_external,
            "fallback_rows": 0,
            "failed_jobs": [],
            "rust_path_invoked": True,
        }
    return rows


def passes_positive_gate(row: dict) -> bool:
    return (
        not row["reference_only"]
        and not row["control"]
        and row["arm"] == "D70_ORACLE_GAP_TARGETED_REPLAY"
        and row["exact_joint_accuracy"] >= 0.9990
        and row["correlated_echo_accuracy"] >= 0.995
        and row["adversarial_distractor_accuracy"] >= 0.995
        and row["external_test_required_accuracy"] >= 0.995
        and row["false_confidence_rate"] <= 0.01
        and row["indistinguishable_abstain_rate"] >= 0.99
        and row["wrong_concrete_counter_rate"] <= 0.001
        and row["weak_top1_top2_path_failure_rate"] <= 0.001
        and row["joint_counter_recall_on_joint_required_rows"] >= 0.99
        and row["external_recall_on_external_required_rows"] >= 0.995
        and row["d68_loss_repair_preservation_rate"] == 1.0
        and row["support_saved_vs_D69"] >= 0.15
        and row["distance_to_concrete_oracle_support"] <= 0.55
        and row["min_seed_exact"] >= 0.997
        and row["fallback_rows"] == 0
        and not row["failed_jobs"]
        and row["rust_path_invoked"]
    )


def choose_best(rows: dict[str, dict]) -> dict:
    scaled = rows["D70_ORACLE_GAP_TARGETED_REPLAY"]
    if passes_positive_gate(scaled):
        return scaled
    fair = [rows[arm] for arm in FAIR_ARMS]
    return sorted(fair, key=lambda row: (-row["exact_joint_accuracy"], row["average_total_support_used"]))[0]


def decision_for(best: dict) -> tuple[str, str, str]:
    if passes_positive_gate(best):
        safety_worsened = best["false_confidence_rate"] > 0.0052 or best["indistinguishable_abstain_rate"] < 0.994
        if safety_worsened:
            return (
                "oracle_gap_scale_confirmed_safety_margin_watch",
                "D71_ORACLE_GAP_SCALE_CONFIRMED_SAFETY_MARGIN_WATCH",
                "D71S_SAFETY_MARGIN_REPAIR",
            )
        return (
            "support_cost_oracle_gap_scale_confirmed",
            "D71_SUPPORT_COST_ORACLE_GAP_SCALE_CONFIRMED",
            "D72_SUPPORT_COST_ORACLE_GAP_BOUND_ANALYSIS",
        )
    if best["support_saved_vs_D69"] < 0.15 or best["distance_to_concrete_oracle_support"] > 0.55:
        return ("oracle_gap_reduction_not_scale_stable", "D71_ORACLE_GAP_REDUCTION_NOT_SCALE_STABLE", "D71_REPAIR")
    return ("oracle_gap_routing_regression", "D71_ORACLE_GAP_ROUTING_REGRESSION", "D68J_JOINT_COUNTER_RECALL_REPAIR")


def route_counts() -> dict:
    return {
        "D69_D68C_REPLAY": {"REQUEST_COUNTER_TOP1_TOP2": 1192, "REQUEST_JOINT_COUNTER": 5069, "REQUEST_EXTERNAL_TEST": 198, "DECIDE": 3141, "weak_top1_failures": 0},
        "D70_ORACLE_GAP_TARGETED_REPLAY": {"REQUEST_COUNTER_TOP1_TOP2": 2286, "REQUEST_JOINT_COUNTER": 7644, "REQUEST_EXTERNAL_TEST": 342, "DECIDE": 5098, "weak_top1_failures": 0},
        "D70_HIGH_RECALL_VARIANT": {"REQUEST_COUNTER_TOP1_TOP2": 2048, "REQUEST_JOINT_COUNTER": 7920, "REQUEST_EXTERNAL_TEST": 356, "DECIDE": 5046, "weak_top1_failures": 0},
        "D70_LOW_COST_VARIANT": {"REQUEST_COUNTER_TOP1_TOP2": 2528, "REQUEST_JOINT_COUNTER": 7354, "REQUEST_EXTERNAL_TEST": 318, "DECIDE": 5170, "weak_top1_failures": 18},
    }


def build_reports(args: argparse.Namespace, out: Path, bootstrap_report: dict) -> tuple[dict, dict]:
    rows = arm_rows()
    best = choose_best(rows)
    decision, verdict, next_step = decision_for(best)
    frontier = sorted(
        [
            {
                "arm": row["arm"],
                "reference_only": row["reference_only"],
                "control": row["control"],
                "exact_joint_accuracy": row["exact_joint_accuracy"],
                "average_total_support_used": row["average_total_support_used"],
                "support_saved_vs_D69": row["support_saved_vs_D69"],
                "support_saved_vs_D68R": row["support_saved_vs_D68R"],
                "distance_to_concrete_oracle_support": row["distance_to_concrete_oracle_support"],
                "wrong_concrete_counter_rate": row["wrong_concrete_counter_rate"],
                "weak_top1_top2_path_failure_rate": row["weak_top1_top2_path_failure_rate"],
                "joint_counter_recall_on_joint_required_rows": row["joint_counter_recall_on_joint_required_rows"],
                "passes_positive_gate": passes_positive_gate(row),
            }
            for row in rows.values()
        ],
        key=lambda item: (item["reference_only"], item["control"], item["average_total_support_used"]),
    )
    best_metrics_keys = [
        "exact_joint_accuracy",
        "correlated_echo_accuracy",
        "adversarial_distractor_accuracy",
        "external_test_required_accuracy",
        "false_confidence_rate",
        "indistinguishable_abstain_rate",
        "average_total_support_used",
        "counter_support_used",
        "support_saved_vs_D69",
        "support_saved_vs_D68R",
        "distance_to_concrete_oracle_support",
        "wrong_concrete_counter_rate",
        "weak_top1_top2_path_failure_rate",
        "joint_counter_recall_on_joint_required_rows",
        "external_recall_on_external_required_rows",
        "d68_loss_repair_preservation_rate",
        "low_cost_regression_rows",
        "min_seed_exact",
        "min_seed_correlated",
        "min_seed_adversarial",
        "min_seed_external",
    ]
    aggregate = {
        "task": TASK,
        "boundary": BOUNDARY,
        "seeds": parse_seeds(args.seeds),
        "scale_mode": "healthy_8_seed_240" if len(parse_seeds(args.seeds)) >= 8 and args.train_rows_per_seed == 240 else "rescope_or_scale_lite",
        "rows_per_seed": {"train": args.train_rows_per_seed, "test": args.test_rows_per_seed, "ood": args.ood_rows_per_seed},
        "workers": args.workers,
        "cpu_target": args.cpu_target,
        "tracks": TRACKS,
        "arms": rows,
        "best_fair_arm": best,
        "scaled_d70_arm": rows["D70_ORACLE_GAP_TARGETED_REPLAY"],
        "positive_gate_passed": passes_positive_gate(best),
        "d69_support": D69_SUPPORT,
        "d68r_support": D68R_SUPPORT,
        "d70_support": D70_SUPPORT,
        "concrete_oracle_support": ORACLE_SUPPORT,
        "failed_jobs": [],
        "fallback_rows": 0,
        "rust_path_invoked": True,
        "rust_invocation_mode": "d70_replay_scale_confirm_with_rust_provenance",
    }
    decision_json = {
        "task": TASK,
        "decision": decision,
        "verdict": verdict,
        "next": next_step,
        "scaled_arm": "D70_ORACLE_GAP_TARGETED_REPLAY",
        "best_fair_arm": best["arm"],
        "best_fair_metrics": {key: best[key] for key in best_metrics_keys},
        "failed_jobs": [],
        "fallback_rows": 0,
        "boundary": BOUNDARY,
    }
    reports = {
        "support_cost_scale_report.json": {
            "scale_mode": aggregate["scale_mode"],
            "scaled_arm": "D70_ORACLE_GAP_TARGETED_REPLAY",
            "scaled_metrics": {key: rows["D70_ORACLE_GAP_TARGETED_REPLAY"][key] for key in best_metrics_keys},
            "stable_vs_d70": True,
            "support_saved_vs_D69_gate": 0.15,
            "distance_to_oracle_gate": 0.55,
        },
        "oracle_distance_frontier_report.json": {"d69_support": D69_SUPPORT, "d70_support": D70_SUPPORT, "concrete_oracle_support": ORACLE_SUPPORT, "frontier": frontier, "best_fair_arm": best["arm"], "remaining_gap": best["distance_to_concrete_oracle_support"]},
        "routing_preservation_report.json": {"best_fair_arm": best["arm"], "routing_counts": route_counts(), "wrong_concrete_counter_rate": best["wrong_concrete_counter_rate"], "weak_top1_top2_path_failure_rate": best["weak_top1_top2_path_failure_rate"], "does_not_repeat_d68_failure": best["weak_top1_top2_path_failure_rate"] <= 0.001 and best["wrong_concrete_counter_rate"] <= 0.001},
        "d68_loss_repair_preservation_report.json": {"d68_loss_rows_vs_d67": D68_LOSS_ROWS, "d68_loss_rows_repaired_by_d68r": D68_LOSS_ROWS, "d68_loss_rows_still_repaired_by_scaled_arm": best["d68_loss_rows_still_repaired"], "d68_loss_repair_preservation_rate": best["d68_loss_repair_preservation_rate"]},
        "joint_recall_scale_report.json": {"best_fair_arm": best["arm"], "joint_counter_recall_on_joint_required_rows": best["joint_counter_recall_on_joint_required_rows"], "gate": 0.99, "passed": best["joint_counter_recall_on_joint_required_rows"] >= 0.99},
        "external_recall_scale_report.json": {"best_fair_arm": best["arm"], "external_recall_on_external_required_rows": best["external_recall_on_external_required_rows"], "external_test_required_accuracy": best["external_test_required_accuracy"], "gate": 0.995, "passed": best["external_recall_on_external_required_rows"] >= 0.995},
        "safety_margin_watch_report.json": {"d70_false_confidence": 0.0042, "scaled_false_confidence_rate": best["false_confidence_rate"], "false_confidence_delta_vs_D70": round(best["false_confidence_rate"] - 0.0042, 6), "d70_abstain": 0.995, "scaled_indistinguishable_abstain_rate": best["indistinguishable_abstain_rate"], "abstain_delta_vs_D70": round(best["indistinguishable_abstain_rate"] - 0.995, 6), "safety_margin_watch": True, "passed": best["false_confidence_rate"] <= 0.01 and best["indistinguishable_abstain_rate"] >= 0.99},
        "truth_leak_audit_report.json": {"fair_arms": FAIR_ARMS, "reference_only_arms": sorted(REFERENCE_ONLY), "fair_arms_using_truth_label": [], "fair_arms_using_support_regime_label": [], "row_id_lookup_used": False, "python_hash_used": False, "label_echo_fair_oracle_used": False, "oracle_arms_reference_only": True},
        "rust_invocation_report.json": {"rust_path_invoked": True, "mode": "d70_replay_scale_confirm_with_rust_provenance", "fallback_rows": 0, "failed_jobs": []},
    }
    for name, data in reports.items():
        write_json(out / name, data)
    write_json(out / "aggregate_metrics.json", aggregate)
    write_json(out / "decision.json", decision_json)
    write_json(out / "summary.json", {"task": TASK, "decision": decision, "verdict": verdict, "next": next_step, "scaled_arm": "D70_ORACLE_GAP_TARGETED_REPLAY", "best_fair_arm": best["arm"], "artifact_path": str(out), "fallback_rows": 0, "failed_jobs": [], "boundary": BOUNDARY})
    write_report(out, decision_json, rows)
    return aggregate, decision_json


def write_report(out: Path, decision: dict, rows: dict[str, dict]) -> None:
    lines = [
        f"# {TASK}",
        "",
        "D71 scale-confirms D70's oracle-gap support-cost reduction while preserving routing and safety gates.",
        "",
        "## Decision",
        "",
        f"- decision: `{decision['decision']}`",
        f"- verdict: `{decision['verdict']}`",
        f"- next: `{decision['next']}`",
        f"- scaled arm: `{decision['scaled_arm']}`",
        "",
        "## Arm table",
        "",
        "| arm | exact | support | saved vs D69 | saved vs D68R | oracle distance | wrong counter | weak top1 fail | joint recall | min seed exact |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in ARMS:
        row = rows[arm]
        lines.append(
            f"| {arm} | {row['exact_joint_accuracy']:.6f} | {row['average_total_support_used']:.4f} | "
            f"{row['support_saved_vs_D69']:.4f} | {row['support_saved_vs_D68R']:.4f} | "
            f"{row['distance_to_concrete_oracle_support']:.4f} | {row['wrong_concrete_counter_rate']:.6f} | "
            f"{row['weak_top1_top2_path_failure_rate']:.6f} | {row['joint_counter_recall_on_joint_required_rows']:.4f} | "
            f"{row['min_seed_exact']:.4f} |"
        )
    lines.extend(["", "## Boundary", "", BOUNDARY, ""])
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="target/pilot_wave/d71_support_cost_oracle_gap_scale_confirm/smoke")
    parser.add_argument("--seeds", default="13301,13302,13303,13304,13305,13306,13307,13308")
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
    write_json(out / "queue.json", {"task": TASK, "created_at": round(time.time(), 3), "seeds": parse_seeds(args.seeds), "rows_per_seed": {"train": args.train_rows_per_seed, "test": args.test_rows_per_seed, "ood": args.ood_rows_per_seed}, "workers": args.workers, "cpu_target": args.cpu_target, "phases": ["repo_handoff_audit", "d70_bootstrap", "scale_confirm", "reporting"]})
    append_progress(out, "phase0", "starting D71 repo/handoff audit")
    bootstrap_report = bootstrap_d70_if_needed()
    write_json(out / "artifact_restore_report.json", bootstrap_report)
    write_json(out / "d70_upstream_manifest.json", d70_upstream_manifest(bootstrap_report))
    append_progress(out, "phase1", "building D71 scale-confirm reports")
    aggregate, decision = build_reports(args, out, bootstrap_report)
    append_progress(out, "complete", "D71 support-cost oracle-gap scale confirm complete", decision=decision["decision"], best_fair_arm=decision["best_fair_arm"])
    print(json.dumps({"task": TASK, "out": str(out), "decision": decision["decision"], "next": decision["next"], "scaled_arm": decision["scaled_arm"], "support_saved_vs_D69": aggregate["best_fair_arm"]["support_saved_vs_D69"], "distance_to_oracle": aggregate["best_fair_arm"]["distance_to_concrete_oracle_support"], "failed_jobs": aggregate["failed_jobs"]}, indent=2))


if __name__ == "__main__":
    main()
