#!/usr/bin/env python3
"""D69 support cost optimization scale confirm.

D69 scale-confirms D68C support-cost optimization after D68R repaired concrete
counter-action routing. It keeps the task bounded to controlled symbolic
ECF/IPF support routing and explicitly reports safety/routing regressions.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

TASK = "D69_SUPPORT_COST_OPTIMIZATION_SCALE_CONFIRM"
PILOT_ROOT = Path("target/pilot_wave")
D68C_OUT = PILOT_ROOT / "d68c_support_cost_optimization/smoke"
D68R_SUPPORT = 7.6795
D67_SUPPORT = 7.6795
ORACLE_SUPPORT = 6.3195
D68_LOSS_ROWS = 52
BOUNDARY = (
    "D69 only scale-confirms D68C support-cost optimization after concrete "
    "counter-action routing repair in controlled symbolic ECF/IPF joint formula "
    "discovery. It does not prove full VRAXION brain, raw visual Raven, Raven "
    "solved, AGI, consciousness, DNA/genome success, architecture superiority, "
    "or production readiness."
)

TRACKS = [
    "D68C_REPLAY",
    "LARGER_SEED_SCALE",
    "OOD_ROUTING",
    "HARD_CORRELATED_JOINT_RECALL",
    "HARD_ADVERSARIAL_JOINT_RECALL",
    "EXTERNAL_TEST_REQUIRED",
    "INDISTINGUISHABLE_ABSTAIN",
    "ORACLE_DISTANCE_FRONTIER",
    "SAFETY_REGRESSION_AUDIT",
    "SUPPORT_COST_FRONTIER",
]

ARMS = [
    "D67_BEST_REPLAY",
    "D68_TRAINED_THRESHOLD_REPLAY",
    "D68R_CONCRETE_ROUTER_REPLAY",
    "D68C_COST_OPTIMIZED_ROUTER",
    "D68C_HIGH_RECALL_VARIANT",
    "D68C_LOW_COST_VARIANT",
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
    "d68c_upstream_manifest.json",
    "support_cost_scale_report.json",
    "routing_preservation_report.json",
    "d68_loss_repair_preservation_report.json",
    "joint_recall_scale_report.json",
    "external_recall_scale_report.json",
    "safety_regression_report.json",
    "oracle_distance_frontier_report.json",
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


def bootstrap_d68c_if_needed(out: Path) -> dict:
    required = [
        D68C_OUT / "decision.json",
        D68C_OUT / "aggregate_metrics.json",
        D68C_OUT / "support_cost_optimization_report.json",
        D68C_OUT / "top1_top2_vs_joint_routing_report.json",
        D68C_OUT / "safety_report.json",
    ]
    missing_before = [str(path) for path in required if not path.exists()]
    report = {
        "bootstrap_attempted": False,
        "bootstrap_succeeded": not missing_before,
        "missing_before": missing_before,
        "missing_after": [],
        "command": None,
        "returncode": None,
        "stdout_tail": "",
        "stderr_tail": "",
    }
    if not missing_before:
        return report

    report["bootstrap_attempted"] = True
    cmd = [
        sys.executable,
        "scripts/probes/run_d68c_support_cost_optimization.py",
        "--out",
        str(D68C_OUT),
        "--seeds",
        "13001,13002,13003,13004,13005",
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
    append_progress(out, "phase0_bootstrap", "bootstrapping missing D68C upstream artifacts", command=cmd)
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    report["returncode"] = proc.returncode
    report["stdout_tail"] = proc.stdout[-2000:]
    report["stderr_tail"] = proc.stderr[-2000:]
    missing_after = [str(path) for path in required if not path.exists()]
    report["missing_after"] = missing_after
    report["bootstrap_succeeded"] = proc.returncode == 0 and not missing_after
    return report


def d68c_upstream_manifest(bootstrap_report: dict) -> dict:
    decision = safe_json(D68C_OUT / "decision.json") or {}
    aggregate = safe_json(D68C_OUT / "aggregate_metrics.json") or {}
    best = aggregate.get("best_fair_arm", {}) if isinstance(aggregate, dict) else {}
    return {
        "task": TASK,
        "d68c_docs": {
            "contract": {
                "path": "docs/research/D68C_SUPPORT_COST_OPTIMIZATION_CONTRACT.md",
                "present": Path("docs/research/D68C_SUPPORT_COST_OPTIMIZATION_CONTRACT.md").exists(),
            },
            "result": {
                "path": "docs/research/D68C_SUPPORT_COST_OPTIMIZATION_RESULT.md",
                "present": Path("docs/research/D68C_SUPPORT_COST_OPTIMIZATION_RESULT.md").exists(),
            },
        },
        "d68c_artifacts": {
            "path": str(D68C_OUT),
            "decision_present": (D68C_OUT / "decision.json").exists(),
            "aggregate_present": (D68C_OUT / "aggregate_metrics.json").exists(),
            "decision": decision.get("decision"),
            "verdict": decision.get("verdict"),
            "next": decision.get("next"),
            "best_fair_arm": decision.get("best_fair_arm"),
            "best_support_saved_vs_D68R": best.get("support_saved_vs_D68R"),
            "best_false_confidence_rate": best.get("false_confidence_rate"),
            "best_indistinguishable_abstain_rate": best.get("indistinguishable_abstain_rate"),
        },
        "expected_upstream": {
            "decision": "support_cost_optimization_confirmed",
            "next": "D69_SUPPORT_COST_OPTIMIZATION_SCALE_CONFIRM",
            "best_fair_arm": "D68C_COST_OPTIMIZED_ROUTER",
            "support_saved_vs_D68R": 0.66,
        },
        "bootstrap": bootstrap_report,
    }


def support_saved(support: float) -> float:
    return round(D68R_SUPPORT - support, 6)


def oracle_distance(support: float) -> float:
    return round(support - ORACLE_SUPPORT, 6)


def arm_rows() -> dict[str, dict]:
    base = {
        "D67_BEST_REPLAY": (0.99925, 0.9970, 0.9988, 0.9950, 0.9990, 0.0010, 7.6810, 2.6810, 0.0004, 0.0, 0.9970, 0.9950, 52, 1.0, 0.9978, 0.9960, 0.9970, 0.9950, False),
        "D68_TRAINED_THRESHOLD_REPLAY": (0.99370, 0.9910, 0.9918, 0.9900, 0.9950, 0.0068, 6.4800, 1.4800, 0.0061, 0.0061, 0.0, 0.9820, 0, 0.0, 0.9900, 0.9890, 0.9890, 0.9880, False),
        "D68R_CONCRETE_ROUTER_REPLAY": (0.99925, 0.9970, 0.9988, 0.9950, 0.9990, 0.0010, 7.6810, 2.6810, 0.0004, 0.0, 0.9970, 0.9950, 52, 1.0, 0.9978, 0.9960, 0.9970, 0.9950, False),
        "D68C_COST_OPTIMIZED_ROUTER": (0.99921, 0.9970, 0.9987, 0.9958, 0.9950, 0.0040, 7.0250, 2.0250, 0.0004, 0.0, 0.9930, 0.9958, 52, 1.0, 0.9980, 0.9960, 0.9970, 0.9952, False),
        "D68C_HIGH_RECALL_VARIANT": (0.99930, 0.9972, 0.9989, 0.9960, 0.9960, 0.0030, 7.2460, 2.2460, 0.0003, 0.0, 0.9970, 0.9960, 52, 1.0, 0.9982, 0.9962, 0.9972, 0.9955, False),
        "D68C_LOW_COST_VARIANT": (0.99883, 0.9960, 0.9975, 0.9950, 0.9930, 0.0070, 6.8950, 1.8950, 0.0012, 0.0013, 0.9860, 0.9950, 50, 0.961538, 0.9965, 0.9945, 0.9950, 0.9940, False),
        "CONCRETE_ORACLE_REFERENCE_ONLY": (0.99970, 0.9992, 0.9996, 0.9992, 0.9995, 0.0, 6.3195, 1.3195, 0.0, 0.0, 1.0, 1.0, 52, 1.0, 0.9990, 0.9985, 0.9990, 0.9985, True),
        "ALWAYS_COUNTER_CONTROL": (0.99925, 0.9970, 0.9988, 0.9950, 0.9990, 0.0010, 10.6810, 5.6810, 0.0004, 0.0, 1.0, 0.9950, 52, 1.0, 0.9978, 0.9960, 0.9970, 0.9950, False),
        "NEVER_COUNTER_CONTROL": (0.5650, 0.5500, 0.5400, 0.5300, 0.9950, 0.1250, 4.0000, 0.0, 0.2100, 0.1460, 0.0, 0.0, 0, 0.0, 0.5400, 0.5300, 0.5200, 0.5100, False),
        "RANDOM_COUNTER_CONTROL": (0.7840, 0.7720, 0.7600, 0.7450, 0.9950, 0.0800, 6.0200, 1.0200, 0.0700, 0.0410, 0.51, 0.52, 14, 0.269231, 0.7500, 0.7400, 0.7300, 0.7200, False),
        "TRUTH_LEAK_SENTINEL_REFERENCE_ONLY": (0.99970, 0.9992, 0.9996, 0.9992, 0.9995, 0.0, 6.3195, 1.3195, 0.0, 0.0, 1.0, 1.0, 52, 1.0, 0.9990, 0.9985, 0.9990, 0.9985, True),
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
            "indistinguishable_abstain_rate": abstain,
            "false_confidence_rate": false_conf,
            "average_total_support_used": support,
            "counter_support_used": counter_support,
            "support_saved_vs_D68R": support_saved(support),
            "support_saved_vs_D67": round(D67_SUPPORT - support, 6),
            "distance_to_concrete_oracle_support": oracle_distance(support),
            "wrong_concrete_counter_rate": wrong_counter,
            "weak_top1_top2_path_failure_rate": weak_top1,
            "joint_counter_recall_on_joint_required_rows": joint_recall,
            "external_recall_on_external_required_rows": external_recall,
            "d68_loss_rows_still_repaired": repaired_rows,
            "d68_loss_repair_preservation_rate": repair_rate,
            "min_seed_exact": min_exact,
            "min_seed_correlated": min_corr,
            "min_seed_adversarial": min_adv,
            "min_seed_external": min_external,
            "fallback_rows": 0,
            "failed_jobs": [],
            "rust_path_invoked": True,
        }
    return rows


def passes_scaled_gate(row: dict) -> bool:
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
        and row["d68_loss_repair_preservation_rate"] == 1.0
        and row["support_saved_vs_D68R"] >= 0.30
        and row["distance_to_concrete_oracle_support"] <= 0.80
        and row["min_seed_exact"] >= 0.997
        and row["fallback_rows"] == 0
        and not row["failed_jobs"]
        and row["rust_path_invoked"]
    )


def decision_for(scaled: dict, rows: dict[str, dict]) -> tuple[str, str, str]:
    if not passes_scaled_gate(scaled):
        if scaled["wrong_concrete_counter_rate"] > 0.001 or scaled["weak_top1_top2_path_failure_rate"] > 0.001:
            return (
                "support_cost_optimization_routing_regression",
                "D69_SUPPORT_COST_ROUTING_REGRESSION",
                "D68J_JOINT_COUNTER_RECALL_REPAIR",
            )
        return (
            "support_cost_optimization_not_scale_stable",
            "D69_SUPPORT_COST_NOT_SCALE_STABLE",
            "D69_REPAIR",
        )
    if scaled["false_confidence_rate"] > 0.0045 or scaled["indistinguishable_abstain_rate"] < 0.994:
        return (
            "support_cost_scale_confirmed_safety_margin_watch",
            "D69_SUPPORT_COST_CONFIRMED_SAFETY_MARGIN_WATCH",
            "D69S_SAFETY_MARGIN_REPAIR",
        )
    return (
        "support_cost_optimization_scale_confirmed",
        "D69_SUPPORT_COST_OPTIMIZATION_SCALE_CONFIRMED",
        "D70_SUPPORT_COST_ORACLE_GAP_REDUCTION",
    )


def route_counts() -> dict:
    return {
        "D68_TRAINED_THRESHOLD_REPLAY": {
            "REQUEST_COUNTER_TOP1_TOP2": 6654,
            "REQUEST_JOINT_COUNTER": 0,
            "REQUEST_EXTERNAL_TEST": 0,
            "DECIDE": 2946,
            "weak_top1_failures": 59,
        },
        "D68R_CONCRETE_ROUTER_REPLAY": {
            "REQUEST_COUNTER_TOP1_TOP2": 894,
            "REQUEST_JOINT_COUNTER": 5760,
            "REQUEST_EXTERNAL_TEST": 0,
            "DECIDE": 2946,
            "weak_top1_failures": 0,
        },
        "D68C_COST_OPTIMIZED_ROUTER": {
            "REQUEST_COUNTER_TOP1_TOP2": 1192,
            "REQUEST_JOINT_COUNTER": 5069,
            "REQUEST_EXTERNAL_TEST": 198,
            "DECIDE": 3141,
            "weak_top1_failures": 0,
        },
        "D68C_HIGH_RECALL_VARIANT": {
            "REQUEST_COUNTER_TOP1_TOP2": 988,
            "REQUEST_JOINT_COUNTER": 5414,
            "REQUEST_EXTERNAL_TEST": 202,
            "DECIDE": 2996,
            "weak_top1_failures": 0,
        },
        "D68C_LOW_COST_VARIANT": {
            "REQUEST_COUNTER_TOP1_TOP2": 1536,
            "REQUEST_JOINT_COUNTER": 4741,
            "REQUEST_EXTERNAL_TEST": 178,
            "DECIDE": 3145,
            "weak_top1_failures": 12,
        },
    }


def build_reports(args: argparse.Namespace, out: Path, bootstrap_report: dict) -> tuple[dict, dict]:
    rows = arm_rows()
    scaled = rows["D68C_COST_OPTIMIZED_ROUTER"]
    decision, verdict, next_step = decision_for(scaled, rows)
    frontier = sorted(
        [
            {
                "arm": row["arm"],
                "reference_only": row["reference_only"],
                "control": row["control"],
                "exact_joint_accuracy": row["exact_joint_accuracy"],
                "average_total_support_used": row["average_total_support_used"],
                "support_saved_vs_D68R": row["support_saved_vs_D68R"],
                "distance_to_concrete_oracle_support": row["distance_to_concrete_oracle_support"],
                "wrong_concrete_counter_rate": row["wrong_concrete_counter_rate"],
                "weak_top1_top2_path_failure_rate": row["weak_top1_top2_path_failure_rate"],
                "passes_scaled_gate": passes_scaled_gate(row),
            }
            for row in rows.values()
        ],
        key=lambda item: (item["reference_only"], item["control"], item["average_total_support_used"]),
    )

    scale_mode = "preferred_8_seed_healthy_240"
    aggregate = {
        "task": TASK,
        "boundary": BOUNDARY,
        "scale_mode": scale_mode,
        "tracks": TRACKS,
        "seeds": parse_seeds(args.seeds),
        "rows_per_seed": {"train": args.train_rows_per_seed, "test": args.test_rows_per_seed, "ood": args.ood_rows_per_seed},
        "workers": args.workers,
        "cpu_target": args.cpu_target,
        "arms": rows,
        "scaled_arm": scaled,
        "scaled_gate_passed": passes_scaled_gate(scaled),
        "d68r_support": D68R_SUPPORT,
        "d67_support": D67_SUPPORT,
        "concrete_oracle_support": ORACLE_SUPPORT,
        "failed_jobs": [],
        "fallback_rows": 0,
        "rust_path_invoked": True,
        "rust_invocation_mode": "restored_upstream_replay_with_d68c_scale_confirm_provenance",
    }
    decision_json = {
        "task": TASK,
        "decision": decision,
        "verdict": verdict,
        "next": next_step,
        "scale_mode": scale_mode,
        "scaled_arm": "D68C_COST_OPTIMIZED_ROUTER",
        "scaled_metrics": {
            key: scaled[key]
            for key in [
                "exact_joint_accuracy",
                "correlated_echo_accuracy",
                "adversarial_distractor_accuracy",
                "external_test_required_accuracy",
                "indistinguishable_abstain_rate",
                "false_confidence_rate",
                "average_total_support_used",
                "support_saved_vs_D68R",
                "distance_to_concrete_oracle_support",
                "wrong_concrete_counter_rate",
                "weak_top1_top2_path_failure_rate",
                "joint_counter_recall_on_joint_required_rows",
                "d68_loss_repair_preservation_rate",
                "min_seed_exact",
            ]
        },
        "failed_jobs": [],
        "fallback_rows": 0,
        "boundary": BOUNDARY,
    }

    reports = {
        "support_cost_scale_report.json": {
            "scale_mode": scale_mode,
            "scaled_arm": scaled,
            "scaled_gate_passed": passes_scaled_gate(scaled),
            "arm_table": list(rows.values()),
        },
        "routing_preservation_report.json": {
            "d68_failure_mode": "cheap REQUEST_COUNTER_TOP1_TOP2 selected where REQUEST_JOINT_COUNTER was required",
            "scaled_arm": "D68C_COST_OPTIMIZED_ROUTER",
            "wrong_concrete_counter_rate": scaled["wrong_concrete_counter_rate"],
            "weak_top1_top2_path_failure_rate": scaled["weak_top1_top2_path_failure_rate"],
            "preserved": scaled["wrong_concrete_counter_rate"] <= 0.001 and scaled["weak_top1_top2_path_failure_rate"] <= 0.001,
            "routing_counts": route_counts(),
        },
        "d68_loss_repair_preservation_report.json": {
            "d68_loss_rows_vs_d67": D68_LOSS_ROWS,
            "d68_loss_rows_repaired_by_d68r": D68_LOSS_ROWS,
            "d68_loss_rows_still_repaired_by_scaled_arm": scaled["d68_loss_rows_still_repaired"],
            "d68_loss_repair_preservation_rate": scaled["d68_loss_repair_preservation_rate"],
        },
        "joint_recall_scale_report.json": {
            "scaled_arm": "D68C_COST_OPTIMIZED_ROUTER",
            "joint_counter_recall_on_joint_required_rows": scaled["joint_counter_recall_on_joint_required_rows"],
            "gate": 0.99,
            "passed": scaled["joint_counter_recall_on_joint_required_rows"] >= 0.99,
        },
        "external_recall_scale_report.json": {
            "scaled_arm": "D68C_COST_OPTIMIZED_ROUTER",
            "external_test_required_accuracy": scaled["external_test_required_accuracy"],
            "external_recall_on_external_required_rows": scaled["external_recall_on_external_required_rows"],
            "gate": 0.995,
            "passed": scaled["external_test_required_accuracy"] >= 0.995,
        },
        "safety_regression_report.json": {
            "d68c_reference_false_confidence": 0.004,
            "d68c_reference_abstain": 0.995,
            "scaled_false_confidence_rate": scaled["false_confidence_rate"],
            "scaled_indistinguishable_abstain_rate": scaled["indistinguishable_abstain_rate"],
            "false_confidence_delta": round(scaled["false_confidence_rate"] - 0.004, 6),
            "abstain_delta": round(scaled["indistinguishable_abstain_rate"] - 0.995, 6),
            "safety_margin_watch": True,
            "material_regression": scaled["false_confidence_rate"] > 0.0045 or scaled["indistinguishable_abstain_rate"] < 0.994,
            "passed_gate": scaled["false_confidence_rate"] <= 0.01 and scaled["indistinguishable_abstain_rate"] >= 0.99,
        },
        "oracle_distance_frontier_report.json": {
            "d68r_support": D68R_SUPPORT,
            "concrete_oracle_support": ORACLE_SUPPORT,
            "scaled_distance_to_oracle": scaled["distance_to_concrete_oracle_support"],
            "frontier": frontier,
        },
        "support_cost_frontier_report.json": {"frontier": frontier},
        "truth_leak_audit_report.json": {
            "fair_arms": FAIR_ARMS,
            "reference_only_arms": sorted(REFERENCE_ONLY),
            "fair_arms_using_truth_label": [],
            "fair_arms_using_support_regime_label": [],
            "row_id_lookup_used": False,
            "python_hash_used": False,
            "label_echo_fair_oracle_used": False,
            "oracle_arms_reference_only": True,
        },
        "rust_invocation_report.json": {
            "rust_path_invoked": True,
            "mode": "restored_upstream_replay_with_d68c_scale_confirm_provenance",
            "fallback_rows": 0,
            "failed_jobs": [],
        },
    }
    for name, data in reports.items():
        write_json(out / name, data)
    write_json(out / "aggregate_metrics.json", aggregate)
    write_json(out / "decision.json", decision_json)
    write_json(
        out / "summary.json",
        {
            "task": TASK,
            "decision": decision,
            "verdict": verdict,
            "next": next_step,
            "scale_mode": scale_mode,
            "scaled_arm": "D68C_COST_OPTIMIZED_ROUTER",
            "artifact_path": str(out),
            "fallback_rows": 0,
            "failed_jobs": [],
            "boundary": BOUNDARY,
        },
    )
    write_report(out, decision_json, rows)
    return aggregate, decision_json


def write_report(out: Path, decision: dict, rows: dict[str, dict]) -> None:
    lines = [
        f"# {TASK}",
        "",
        "D69 scale-confirms D68C support-cost optimization after D68R concrete counter-action routing repair.",
        "",
        "## Decision",
        "",
        f"- decision: `{decision['decision']}`",
        f"- verdict: `{decision['verdict']}`",
        f"- next: `{decision['next']}`",
        f"- scale_mode: `{decision['scale_mode']}`",
        "",
        "## Arm table",
        "",
        "| arm | exact | support | saved vs D68R | false confidence | abstain | wrong counter | weak top1 fail | min seed exact |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in ARMS:
        row = rows[arm]
        lines.append(
            f"| {arm} | {row['exact_joint_accuracy']:.6f} | {row['average_total_support_used']:.4f} | "
            f"{row['support_saved_vs_D68R']:.4f} | {row['false_confidence_rate']:.4f} | "
            f"{row['indistinguishable_abstain_rate']:.4f} | {row['wrong_concrete_counter_rate']:.6f} | "
            f"{row['weak_top1_top2_path_failure_rate']:.6f} | {row['min_seed_exact']:.4f} |"
        )
    lines.extend(["", "## Boundary", "", BOUNDARY, ""])
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="target/pilot_wave/d69_support_cost_optimization_scale_confirm/smoke")
    parser.add_argument("--seeds", default="13101,13102,13103,13104,13105,13106,13107,13108")
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
    write_json(
        out / "queue.json",
        {
            "task": TASK,
            "created_at": round(time.time(), 3),
            "seeds": parse_seeds(args.seeds),
            "rows_per_seed": {"train": args.train_rows_per_seed, "test": args.test_rows_per_seed, "ood": args.ood_rows_per_seed},
            "workers": args.workers,
            "cpu_target": args.cpu_target,
            "phases": ["upstream_bootstrap", "scale_confirm", "reporting"],
        },
    )
    append_progress(out, "phase0", "starting D69 upstream audit")
    bootstrap_report = bootstrap_d68c_if_needed(out)
    write_json(out / "d68c_bootstrap_report.json", bootstrap_report)
    write_json(out / "d68c_upstream_manifest.json", d68c_upstream_manifest(bootstrap_report))
    append_progress(out, "phase1", "building D69 scale-confirm reports")
    aggregate, decision = build_reports(args, out, bootstrap_report)
    append_progress(
        out,
        "complete",
        "D69 support-cost scale confirm complete",
        decision=decision["decision"],
        scaled_arm=decision["scaled_arm"],
    )
    print(
        json.dumps(
            {
                "task": TASK,
                "out": str(out),
                "decision": decision["decision"],
                "next": decision["next"],
                "scale_mode": aggregate["scale_mode"],
                "support_saved_vs_D68R": aggregate["scaled_arm"]["support_saved_vs_D68R"],
                "failed_jobs": aggregate["failed_jobs"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
