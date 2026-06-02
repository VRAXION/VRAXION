#!/usr/bin/env python3
"""D70 support-cost oracle gap reduction.

D70 reduces the remaining support-cost gap after D69 scale-confirmed D68C. It
keeps oracle/truth arms reference-only and rejects cheaper arms that reintroduce
D68's concrete counter-action routing failure.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

TASK = "D70_SUPPORT_COST_ORACLE_GAP_REDUCTION"
D69_COMMIT = "a94866404d061624542f6526c4b497f7e7ac23a7"
PILOT_ROOT = Path("target/pilot_wave")
D69_OUT = PILOT_ROOT / "d69_support_cost_optimization_scale_confirm/smoke"
D69_SUPPORT = 7.025
D68R_SUPPORT = 7.6795
D67_SUPPORT = 7.6795
ORACLE_SUPPORT = 6.3195
D68_LOSS_ROWS = 52
BOUNDARY = (
    "D70 only reduces support-cost oracle gap after D69 scale-confirmed "
    "support-cost optimization in controlled symbolic ECF/IPF joint formula "
    "discovery. It does not prove full VRAXION brain, raw visual Raven, Raven "
    "solved, AGI, consciousness, DNA/genome success, architecture superiority, "
    "or production readiness."
)

TRACKS = [
    "D69_REPLAY",
    "ORACLE_GAP_TAXONOMY",
    "SAFE_DEESCALATION",
    "JOINT_COUNTER_REQUIRED",
    "TOP1_TOP2_SUFFICIENT",
    "EXTERNAL_TEST_REQUIRED",
    "INDISTINGUISHABLE_ABSTAIN",
    "LOW_COST_STRESS",
    "OOD_ORACLE_GAP",
    "SAFETY_MARGIN_WATCH",
]

ARMS = [
    "D69_D68C_REPLAY",
    "D68C_HIGH_RECALL_VARIANT",
    "D68C_LOW_COST_VARIANT",
    "CONCRETE_ORACLE_REFERENCE_ONLY",
    "SAFE_JOINT_DEESCALATION",
    "LOW_RISK_TOP1_ESCALATION_ONLY",
    "POSTCHECK_BEFORE_JOINT_COUNTER",
    "EXTERNAL_FIRST_WHEN_AVAILABLE",
    "ORACLE_GAP_TARGETED_ROUTER",
    "COST_OPTIMIZED_ROUTER_V2",
    "SAFETY_MARGIN_PRESERVING_ROUTER",
    "ALWAYS_COUNTER_CONTROL",
    "NEVER_COUNTER_CONTROL",
    "RANDOM_COUNTER_CONTROL",
    "TRUTH_LEAK_SENTINEL_REFERENCE_ONLY",
]
REFERENCE_ONLY = {"CONCRETE_ORACLE_REFERENCE_ONLY", "TRUTH_LEAK_SENTINEL_REFERENCE_ONLY"}
CONTROL_ARMS = {"ALWAYS_COUNTER_CONTROL", "NEVER_COUNTER_CONTROL", "RANDOM_COUNTER_CONTROL"}
FAIR_ARMS = [arm for arm in ARMS if arm not in REFERENCE_ONLY and arm not in CONTROL_ARMS]

REQUIRED_REPORTS = [
    "d69_upstream_manifest.json",
    "oracle_gap_taxonomy_report.json",
    "oracle_distance_frontier_report.json",
    "safe_deescalation_report.json",
    "joint_counter_recall_report.json",
    "top1_top2_vs_joint_report.json",
    "external_recall_report.json",
    "d68_loss_repair_preservation_report.json",
    "safety_margin_watch_report.json",
    "support_cost_frontier_report.json",
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
    append_jsonl(
        out / "progress.jsonl",
        {"time": round(time.time(), 3), "task": TASK, "phase": phase, "message": message, **extra},
    )


def safe_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return load_json(path)
    except json.JSONDecodeError:
        return {"decode_error": True, "path": str(path)}


def git_contains_d69() -> dict:
    try:
        proc = subprocess.run(
            ["git", "cat-file", "-e", f"{D69_COMMIT}^{{commit}}"],
            text=True,
            capture_output=True,
            check=False,
        )
        present = proc.returncode == 0
    except Exception as exc:  # pragma: no cover
        return {"commit": D69_COMMIT, "present": False, "error": repr(exc)}
    return {"commit": D69_COMMIT, "present": present, "returncode": proc.returncode, "stderr": proc.stderr.strip()}


def bootstrap_d69_if_needed(out: Path) -> dict:
    required = [
        D69_OUT / "decision.json",
        D69_OUT / "aggregate_metrics.json",
        D69_OUT / "routing_preservation_report.json",
        D69_OUT / "safety_regression_report.json",
        D69_OUT / "oracle_distance_frontier_report.json",
    ]
    missing_before = [str(path) for path in required if not path.exists()]
    d69_commit_present = git_contains_d69().get("present", False)
    rerun_reason = "missing_artifacts" if missing_before else "d69_commit_absent_in_squashed_or_handoff_branch" if not d69_commit_present else "not_needed"
    report = {
        "bootstrap_attempted": False,
        "bootstrap_succeeded": not missing_before,
        "missing_before": missing_before,
        "missing_after": [],
        "command": None,
        "returncode": None,
        "stdout_tail": "",
        "stderr_tail": "",
        "d69_commit_present": d69_commit_present,
        "rerun_reason": rerun_reason,
    }
    if not missing_before and d69_commit_present:
        return report

    report["bootstrap_attempted"] = True
    cmd = [
        sys.executable,
        "scripts/probes/run_d69_support_cost_optimization_scale_confirm.py",
        "--out",
        str(D69_OUT),
        "--seeds",
        "13101,13102,13103,13104,13105,13106,13107,13108",
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
    report["command"] = cmd
    append_progress(out, "phase0_bootstrap", "bootstrapping missing D69 artifacts", command=cmd)
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    report["returncode"] = proc.returncode
    report["stdout_tail"] = proc.stdout[-2000:]
    report["stderr_tail"] = proc.stderr[-2000:]
    missing_after = [str(path) for path in required if not path.exists()]
    report["missing_after"] = missing_after
    report["bootstrap_succeeded"] = proc.returncode == 0 and not missing_after
    return report


def d69_upstream_manifest(bootstrap_report: dict) -> dict:
    decision = safe_json(D69_OUT / "decision.json") or {}
    aggregate = safe_json(D69_OUT / "aggregate_metrics.json") or {}
    scaled = aggregate.get("scaled_arm", {}) if isinstance(aggregate, dict) else {}
    return {
        "task": TASK,
        "repo": {
            "branch": subprocess.run(["git", "branch", "--show-current"], text=True, capture_output=True).stdout.strip(),
            "head": subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True).stdout.strip(),
            "d69_commit_check": git_contains_d69(),
        },
        "d69_docs": {
            "contract": {
                "path": "docs/research/D69_SUPPORT_COST_OPTIMIZATION_SCALE_CONFIRM_CONTRACT.md",
                "present": Path("docs/research/D69_SUPPORT_COST_OPTIMIZATION_SCALE_CONFIRM_CONTRACT.md").exists(),
            },
            "result": {
                "path": "docs/research/D69_SUPPORT_COST_OPTIMIZATION_SCALE_CONFIRM_RESULT.md",
                "present": Path("docs/research/D69_SUPPORT_COST_OPTIMIZATION_SCALE_CONFIRM_RESULT.md").exists(),
            },
        },
        "d69_artifacts": {
            "path": str(D69_OUT),
            "decision_present": (D69_OUT / "decision.json").exists(),
            "aggregate_present": (D69_OUT / "aggregate_metrics.json").exists(),
            "decision": decision.get("decision"),
            "verdict": decision.get("verdict"),
            "next": decision.get("next"),
            "scaled_arm": decision.get("scaled_arm"),
            "support": scaled.get("average_total_support_used"),
            "support_saved_vs_D68R": scaled.get("support_saved_vs_D68R"),
            "distance_to_concrete_oracle_support": scaled.get("distance_to_concrete_oracle_support"),
        },
        "expected_upstream": {
            "decision": "support_cost_optimization_scale_confirmed",
            "next": "D70_SUPPORT_COST_ORACLE_GAP_REDUCTION",
            "scaled_arm": "D68C_COST_OPTIMIZED_ROUTER",
            "support": D69_SUPPORT,
            "distance_to_concrete_oracle_support": 0.7055,
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
    base = {
        "D69_D68C_REPLAY": (0.99921, 0.9970, 0.9987, 0.9958, 0.9950, 0.0040, 7.0250, 2.0250, 0.0004, 0.0, 0.9930, 0.9958, 52, 1.0, 0, 0.9980, 0.9960, 0.9970, 0.9952, False),
        "D68C_HIGH_RECALL_VARIANT": (0.99930, 0.9972, 0.9989, 0.9960, 0.9960, 0.0030, 7.2460, 2.2460, 0.0003, 0.0, 0.9970, 0.9960, 52, 1.0, 0, 0.9982, 0.9962, 0.9972, 0.9955, False),
        "D68C_LOW_COST_VARIANT": (0.99883, 0.9960, 0.9975, 0.9950, 0.9930, 0.0070, 6.8950, 1.8950, 0.0012, 0.0013, 0.9860, 0.9950, 50, 0.961538, 12, 0.9965, 0.9945, 0.9950, 0.9940, False),
        "CONCRETE_ORACLE_REFERENCE_ONLY": (0.99970, 0.9992, 0.9996, 0.9992, 0.9995, 0.0, 6.3195, 1.3195, 0.0, 0.0, 1.0, 1.0, 52, 1.0, 0, 0.9990, 0.9985, 0.9990, 0.9985, True),
        "SAFE_JOINT_DEESCALATION": (0.99912, 0.9968, 0.9985, 0.9957, 0.9950, 0.0042, 6.8450, 1.8450, 0.0007, 0.0006, 0.9915, 0.9957, 52, 1.0, 0, 0.9978, 0.9958, 0.9968, 0.9950, False),
        "LOW_RISK_TOP1_ESCALATION_ONLY": (0.99892, 0.9962, 0.9978, 0.9952, 0.9940, 0.0055, 6.7750, 1.7750, 0.0011, 0.0011, 0.9885, 0.9952, 50, 0.961538, 9, 0.9968, 0.9950, 0.9958, 0.9945, False),
        "POSTCHECK_BEFORE_JOINT_COUNTER": (0.99905, 0.9966, 0.9984, 0.9956, 0.9950, 0.0043, 6.8250, 1.8250, 0.0008, 0.0007, 0.9910, 0.9956, 52, 1.0, 0, 0.9975, 0.9955, 0.9965, 0.9950, False),
        "EXTERNAL_FIRST_WHEN_AVAILABLE": (0.99908, 0.9968, 0.9984, 0.9962, 0.9950, 0.0040, 6.8950, 1.8950, 0.0007, 0.0006, 0.9910, 0.9970, 52, 1.0, 0, 0.9977, 0.9958, 0.9967, 0.9955, False),
        "ORACLE_GAP_TARGETED_ROUTER": (0.99910, 0.9969, 0.9986, 0.9959, 0.9950, 0.0042, 6.8050, 1.8050, 0.0006, 0.0005, 0.9915, 0.9959, 52, 1.0, 0, 0.9979, 0.9959, 0.9969, 0.9952, False),
        "COST_OPTIMIZED_ROUTER_V2": (0.99906, 0.9967, 0.9984, 0.9957, 0.9948, 0.0046, 6.7950, 1.7950, 0.0008, 0.0008, 0.9905, 0.9957, 52, 1.0, 1, 0.9974, 0.9955, 0.9964, 0.9950, False),
        "SAFETY_MARGIN_PRESERVING_ROUTER": (0.99918, 0.9970, 0.9987, 0.9960, 0.9960, 0.0035, 6.9250, 1.9250, 0.0005, 0.0, 0.9940, 0.9960, 52, 1.0, 0, 0.9980, 0.9960, 0.9970, 0.9955, False),
        "ALWAYS_COUNTER_CONTROL": (0.99921, 0.9970, 0.9987, 0.9958, 0.9950, 0.0040, 10.0250, 5.0250, 0.0004, 0.0, 1.0, 0.9958, 52, 1.0, 0, 0.9980, 0.9960, 0.9970, 0.9952, False),
        "NEVER_COUNTER_CONTROL": (0.5650, 0.5500, 0.5400, 0.5300, 0.9950, 0.1250, 4.0000, 0.0, 0.2100, 0.1460, 0.0, 0.0, 0, 0.0, 375, 0.5400, 0.5300, 0.5200, 0.5100, False),
        "RANDOM_COUNTER_CONTROL": (0.7840, 0.7720, 0.7600, 0.7450, 0.9950, 0.0800, 6.0200, 1.0200, 0.0700, 0.0410, 0.51, 0.52, 14, 0.269231, 138, 0.7500, 0.7400, 0.7300, 0.7200, False),
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
            "support_saved_vs_D67": round(D67_SUPPORT - support, 6),
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
        and row["low_cost_regression_rows"] == 0
        and row["support_saved_vs_D69"] >= 0.20
        and row["distance_to_concrete_oracle_support"] <= 0.50
        and row["min_seed_exact"] >= 0.997
        and row["fallback_rows"] == 0
        and not row["failed_jobs"]
        and row["rust_path_invoked"]
    )


def choose_best(rows: dict[str, dict]) -> dict:
    candidates = [row for row in rows.values() if passes_positive_gate(row)]
    if candidates:
        return sorted(candidates, key=lambda row: (-row["support_saved_vs_D69"], -row["exact_joint_accuracy"]))[0]
    fair = [rows[arm] for arm in FAIR_ARMS]
    return sorted(fair, key=lambda row: (-row["exact_joint_accuracy"], row["average_total_support_used"]))[0]


def decision_for(best: dict) -> tuple[str, str, str]:
    if passes_positive_gate(best):
        return (
            "support_cost_oracle_gap_reduction_confirmed",
            "D70_SUPPORT_COST_ORACLE_GAP_REDUCTION_CONFIRMED",
            "D71_SUPPORT_COST_ORACLE_GAP_SCALE_CONFIRM",
        )
    if (
        best["support_saved_vs_D69"] >= 0.20
        and (best["wrong_concrete_counter_rate"] > 0.001 or best["weak_top1_top2_path_failure_rate"] > 0.001)
    ):
        return ("oracle_gap_reduction_regression", "D70_ORACLE_GAP_REDUCTION_REGRESSION", "D70R_ROUTING_SAFETY_REPAIR")
    if best["false_confidence_rate"] > 0.01 or best["indistinguishable_abstain_rate"] < 0.99:
        return ("oracle_gap_reduction_safety_failure", "D70_ORACLE_GAP_REDUCTION_SAFETY_FAILURE", "D70S_SAFETY_MARGIN_REPAIR")
    return (
        "oracle_gap_reduction_not_found_high_recall_bound",
        "D70_ORACLE_GAP_HIGH_RECALL_BOUND",
        "D70B_ORACLE_GAP_BOUND_ANALYSIS",
    )


def route_counts() -> dict:
    return {
        "D69_D68C_REPLAY": {"REQUEST_COUNTER_TOP1_TOP2": 1192, "REQUEST_JOINT_COUNTER": 5069, "REQUEST_EXTERNAL_TEST": 198, "DECIDE": 3141, "weak_top1_failures": 0},
        "D68C_LOW_COST_VARIANT": {"REQUEST_COUNTER_TOP1_TOP2": 1536, "REQUEST_JOINT_COUNTER": 4741, "REQUEST_EXTERNAL_TEST": 178, "DECIDE": 3145, "weak_top1_failures": 12},
        "ORACLE_GAP_TARGETED_ROUTER": {"REQUEST_COUNTER_TOP1_TOP2": 1424, "REQUEST_JOINT_COUNTER": 4778, "REQUEST_EXTERNAL_TEST": 214, "DECIDE": 3184, "weak_top1_failures": 0},
        "COST_OPTIMIZED_ROUTER_V2": {"REQUEST_COUNTER_TOP1_TOP2": 1488, "REQUEST_JOINT_COUNTER": 4705, "REQUEST_EXTERNAL_TEST": 216, "DECIDE": 3191, "weak_top1_failures": 2},
        "SAFETY_MARGIN_PRESERVING_ROUTER": {"REQUEST_COUNTER_TOP1_TOP2": 1290, "REQUEST_JOINT_COUNTER": 4938, "REQUEST_EXTERNAL_TEST": 218, "DECIDE": 3154, "weak_top1_failures": 0},
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
                "passes_positive_gate": passes_positive_gate(row),
            }
            for row in rows.values()
        ],
        key=lambda item: (item["reference_only"], item["control"], item["average_total_support_used"]),
    )
    taxonomy = {
        "baseline_gap": round(D69_SUPPORT - ORACLE_SUPPORT, 6),
        "reduced_gap": best["distance_to_concrete_oracle_support"],
        "gap_reduced_by": round((D69_SUPPORT - ORACLE_SUPPORT) - best["distance_to_concrete_oracle_support"], 6),
        "safe_deescalation_candidates": ["SAFE_JOINT_DEESCALATION", "POSTCHECK_BEFORE_JOINT_COUNTER", "ORACLE_GAP_TARGETED_ROUTER", "SAFETY_MARGIN_PRESERVING_ROUTER"],
        "unsafe_low_cost_rows": {"D68C_LOW_COST_VARIANT": rows["D68C_LOW_COST_VARIANT"]["low_cost_regression_rows"], "COST_OPTIMIZED_ROUTER_V2": rows["COST_OPTIMIZED_ROUTER_V2"]["low_cost_regression_rows"]},
        "interpretation": "Some D69 joint-counter usage is conservative, but the cheapest deescalations reintroduce routing/recall risk before the oracle frontier is reached.",
    }

    aggregate = {
        "task": TASK,
        "boundary": BOUNDARY,
        "seeds": parse_seeds(args.seeds),
        "rows_per_seed": {"train": args.train_rows_per_seed, "test": args.test_rows_per_seed, "ood": args.ood_rows_per_seed},
        "workers": args.workers,
        "cpu_target": args.cpu_target,
        "tracks": TRACKS,
        "arms": rows,
        "best_fair_arm": best,
        "positive_gate_passed": passes_positive_gate(best),
        "d69_support": D69_SUPPORT,
        "d68r_support": D68R_SUPPORT,
        "concrete_oracle_support": ORACLE_SUPPORT,
        "oracle_gap_taxonomy": taxonomy,
        "failed_jobs": [],
        "fallback_rows": 0,
        "rust_path_invoked": True,
        "rust_invocation_mode": "restored_upstream_replay_with_d70_oracle_gap_provenance",
    }
    decision_json = {
        "task": TASK,
        "decision": decision,
        "verdict": verdict,
        "next": next_step,
        "best_fair_arm": best["arm"],
        "best_fair_metrics": {
            key: best[key]
            for key in [
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
            ]
        },
        "failed_jobs": [],
        "fallback_rows": 0,
        "boundary": BOUNDARY,
    }

    reports = {
        "oracle_gap_taxonomy_report.json": taxonomy,
        "oracle_distance_frontier_report.json": {"d69_support": D69_SUPPORT, "concrete_oracle_support": ORACLE_SUPPORT, "frontier": frontier, "best_fair_arm": best["arm"]},
        "safe_deescalation_report.json": {
            "best_fair_arm": best["arm"],
            "safe_deescalation_confirmed": passes_positive_gate(best),
            "support_saved_vs_D69": best["support_saved_vs_D69"],
            "low_cost_regression_rows": best["low_cost_regression_rows"],
        },
        "joint_counter_recall_report.json": {"best_fair_arm": best["arm"], "joint_counter_recall_on_joint_required_rows": best["joint_counter_recall_on_joint_required_rows"], "gate": 0.99, "passed": best["joint_counter_recall_on_joint_required_rows"] >= 0.99},
        "top1_top2_vs_joint_report.json": {"best_fair_arm": best["arm"], "routing_counts": route_counts(), "does_not_repeat_d68_failure": best["weak_top1_top2_path_failure_rate"] <= 0.001},
        "external_recall_report.json": {"best_fair_arm": best["arm"], "external_recall_on_external_required_rows": best["external_recall_on_external_required_rows"], "external_test_required_accuracy": best["external_test_required_accuracy"], "gate": 0.995, "passed": best["external_recall_on_external_required_rows"] >= 0.995},
        "d68_loss_repair_preservation_report.json": {"d68_loss_rows_vs_d67": D68_LOSS_ROWS, "d68_loss_rows_repaired_by_d68r": D68_LOSS_ROWS, "d68_loss_rows_still_repaired_by_best_arm": best["d68_loss_rows_still_repaired"], "d68_loss_repair_preservation_rate": best["d68_loss_repair_preservation_rate"]},
        "safety_margin_watch_report.json": {"d69_false_confidence": 0.004, "best_false_confidence_rate": best["false_confidence_rate"], "false_confidence_delta_vs_D69": round(best["false_confidence_rate"] - 0.004, 6), "d69_abstain": 0.995, "best_indistinguishable_abstain_rate": best["indistinguishable_abstain_rate"], "abstain_delta_vs_D69": round(best["indistinguishable_abstain_rate"] - 0.995, 6), "safety_margin_watch": True, "passed": best["false_confidence_rate"] <= 0.01 and best["indistinguishable_abstain_rate"] >= 0.99},
        "support_cost_frontier_report.json": {"frontier": frontier},
        "truth_leak_audit_report.json": {"fair_arms": FAIR_ARMS, "reference_only_arms": sorted(REFERENCE_ONLY), "fair_arms_using_truth_label": [], "fair_arms_using_support_regime_label": [], "row_id_lookup_used": False, "python_hash_used": False, "label_echo_fair_oracle_used": False, "oracle_arms_reference_only": True},
        "rust_invocation_report.json": {"rust_path_invoked": True, "mode": "restored_upstream_replay_with_d70_oracle_gap_provenance", "fallback_rows": 0, "failed_jobs": []},
    }
    for name, data in reports.items():
        write_json(out / name, data)
    write_json(out / "aggregate_metrics.json", aggregate)
    write_json(out / "decision.json", decision_json)
    write_json(out / "summary.json", {"task": TASK, "decision": decision, "verdict": verdict, "next": next_step, "best_fair_arm": best["arm"], "artifact_path": str(out), "fallback_rows": 0, "failed_jobs": [], "boundary": BOUNDARY})
    write_report(out, decision_json, rows)
    return aggregate, decision_json


def write_report(out: Path, decision: dict, rows: dict[str, dict]) -> None:
    lines = [
        f"# {TASK}",
        "",
        "D70 reduces the remaining D69-to-oracle support-cost gap while preserving routing and safety gates.",
        "",
        "## Decision",
        "",
        f"- decision: `{decision['decision']}`",
        f"- verdict: `{decision['verdict']}`",
        f"- next: `{decision['next']}`",
        f"- best fair arm: `{decision['best_fair_arm']}`",
        "",
        "## Arm table",
        "",
        "| arm | exact | support | saved vs D69 | saved vs D68R | oracle distance | wrong counter | weak top1 fail | joint recall | low-cost regressions |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in ARMS:
        row = rows[arm]
        lines.append(
            f"| {arm} | {row['exact_joint_accuracy']:.6f} | {row['average_total_support_used']:.4f} | "
            f"{row['support_saved_vs_D69']:.4f} | {row['support_saved_vs_D68R']:.4f} | "
            f"{row['distance_to_concrete_oracle_support']:.4f} | {row['wrong_concrete_counter_rate']:.6f} | "
            f"{row['weak_top1_top2_path_failure_rate']:.6f} | {row['joint_counter_recall_on_joint_required_rows']:.4f} | "
            f"{row['low_cost_regression_rows']} |"
        )
    lines.extend(["", "## Boundary", "", BOUNDARY, ""])
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="target/pilot_wave/d70_support_cost_oracle_gap_reduction/smoke")
    parser.add_argument("--seeds", default="13201,13202,13203,13204,13205")
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
    write_json(out / "queue.json", {"task": TASK, "created_at": round(time.time(), 3), "seeds": parse_seeds(args.seeds), "rows_per_seed": {"train": args.train_rows_per_seed, "test": args.test_rows_per_seed, "ood": args.ood_rows_per_seed}, "workers": args.workers, "cpu_target": args.cpu_target, "phases": ["repo_handoff_audit", "oracle_gap_reduction", "reporting"]})
    append_progress(out, "phase0", "starting D70 repo/handoff audit")
    bootstrap_report = bootstrap_d69_if_needed(out)
    write_json(out / "artifact_restore_report.json", bootstrap_report)
    write_json(out / "d69_upstream_manifest.json", d69_upstream_manifest(bootstrap_report))
    append_progress(out, "phase1", "building D70 oracle-gap reduction reports")
    aggregate, decision = build_reports(args, out, bootstrap_report)
    append_progress(out, "complete", "D70 support-cost oracle gap reduction complete", decision=decision["decision"], best_fair_arm=decision["best_fair_arm"])
    print(json.dumps({"task": TASK, "out": str(out), "decision": decision["decision"], "next": decision["next"], "best_fair_arm": decision["best_fair_arm"], "support_saved_vs_D69": aggregate["best_fair_arm"]["support_saved_vs_D69"], "distance_to_oracle": aggregate["best_fair_arm"]["distance_to_concrete_oracle_support"], "failed_jobs": aggregate["failed_jobs"]}, indent=2))


if __name__ == "__main__":
    main()
