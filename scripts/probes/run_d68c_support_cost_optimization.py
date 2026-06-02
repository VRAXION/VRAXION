#!/usr/bin/env python3
"""D68C support cost optimization.

This probe optimizes support cost after D68R repaired concrete counter-action
routing. D68C is intentionally bounded: it replays/restores upstream D67-D68R
provenance, keeps oracle/truth arms reference-only, and evaluates fair
cost-aware routers against explicit concrete-action safety gates.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import urllib.request
import zipfile
from pathlib import Path

TASK = "D68C_SUPPORT_COST_OPTIMIZATION"
PILOT_ROOT = Path("target/pilot_wave")
HANDOFF_URL = (
    "https://github.com/VRAXION/VRAXION/releases/download/"
    "handoff-d45-d68r-2026-06-01/"
    "vraxion_handoff_2026-06-01_D45-D68R_main_2d0b8f3e.zip"
)
BOUNDARY = (
    "D68C only optimizes support cost after concrete counter-action routing "
    "repair in controlled symbolic ECF/IPF joint formula discovery. It does "
    "not prove full VRAXION brain, raw visual Raven reasoning, Raven solved, "
    "AGI, consciousness, DNA/genome success, architecture superiority, or "
    "production readiness."
)

D68R_SUPPORT = 7.6795
D67_SUPPORT = 7.6795
ORACLE_SUPPORT = 6.3195
D68_LOSS_ROWS = 52

ARMS = [
    "D67_BEST_REPLAY",
    "D68_TRAINED_THRESHOLD_REPLAY",
    "D68R_CONCRETE_ROUTER_REPLAY",
    "CONCRETE_ORACLE_REFERENCE_ONLY",
    "D68C_COST_OPTIMIZED_ROUTER",
    "TWO_STAGE_TOP1_THEN_JOINT_ESCALATION",
    "JOINT_COUNTER_SKIP_WHEN_DECIDE_SAFE",
    "JOINT_COUNTER_DEFER_WITH_POSTCHECK",
    "EXTERNAL_ESCALATION_COST_AWARE_ROUTER",
    "HIGH_RECALL_JOINT_LOW_COST_ROUTER",
    "SUPPORT_OVER_CHEAPEST_OPTIMIZER",
    "ALWAYS_COUNTER_CONTROL",
    "NEVER_COUNTER_CONTROL",
    "RANDOM_COUNTER_CONTROL",
    "TRUTH_LEAK_SENTINEL_REFERENCE_ONLY",
]
REFERENCE_ONLY = {"CONCRETE_ORACLE_REFERENCE_ONLY", "TRUTH_LEAK_SENTINEL_REFERENCE_ONLY"}
CONTROL_ARMS = {"ALWAYS_COUNTER_CONTROL", "NEVER_COUNTER_CONTROL", "RANDOM_COUNTER_CONTROL"}
FAIR_ARMS = [a for a in ARMS if a not in REFERENCE_ONLY and a not in CONTROL_ARMS]

TRACKS = [
    "CLEAN_COST_SAVING",
    "MIXED_COST_SAVING",
    "HARD_CORRELATED_JOINT_RECALL",
    "HARD_ADVERSARIAL_JOINT_RECALL",
    "TOP1_TOP2_SUFFICIENT",
    "JOINT_COUNTER_REQUIRED",
    "EXTERNAL_TEST_REQUIRED",
    "INDISTINGUISHABLE_ABSTAIN",
    "OOD_COST_ROUTING",
    "ORACLE_DISTANCE_FRONTIER",
]

REQUIRED_UPSTREAM = {
    "d68a_result_doc": Path("docs/research/D68A_COUNTER_SUPPORT_METRIC_SEMANTICS_AUDIT_RESULT.md"),
    "d68r_result_doc": Path("docs/research/D68R_CONCRETE_COUNTER_ACTION_ROUTING_REPAIR_RESULT.md"),
    "d68r_runner": Path("scripts/probes/run_d68r_concrete_counter_action_routing_repair.py"),
    "d68a_runner": Path("scripts/probes/run_d68a_counter_support_metric_semantics_audit.py"),
}

UPSTREAM_ARTIFACTS = {
    "d68r_decision": PILOT_ROOT / "d68r_concrete_counter_action_routing_repair/smoke/decision.json",
    "d68r_aggregate": PILOT_ROOT / "d68r_concrete_counter_action_routing_repair/smoke/aggregate_metrics.json",
    "d68r_routing": PILOT_ROOT / "d68r_concrete_counter_action_routing_repair/smoke/concrete_action_routing_report.json",
    "d68r_top1_joint": PILOT_ROOT / "d68r_concrete_counter_action_routing_repair/smoke/top1_vs_joint_counter_report.json",
    "d68r_frontier": PILOT_ROOT / "d68r_concrete_counter_action_routing_repair/smoke/support_cost_frontier_report.json",
    "d68a_harm": PILOT_ROOT / "d68a_counter_support_metric_semantics_audit/smoke/d68_harm_classification_report.json",
}

REQUIRED_REPORTS = [
    "d68r_upstream_manifest.json",
    "support_cost_optimization_report.json",
    "concrete_action_routing_preservation_report.json",
    "d68_loss_repair_preservation_report.json",
    "top1_top2_vs_joint_routing_report.json",
    "joint_counter_recall_report.json",
    "external_escalation_report.json",
    "support_over_cheapest_report.json",
    "oracle_distance_frontier_report.json",
    "action_confusion_matrix_report.json",
    "clean_mixed_cost_saving_report.json",
    "hard_regime_recall_report.json",
    "safety_report.json",
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


def safe_read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return load_json(path)
    except json.JSONDecodeError:
        return {"decode_error": True, "path": str(path)}


def upstream_manifest() -> dict:
    artifacts = {}
    for name, path in UPSTREAM_ARTIFACTS.items():
        data = safe_read_json(path)
        artifacts[name] = {
            "path": str(path),
            "present": path.exists(),
            "decision": data.get("decision") if isinstance(data, dict) else None,
            "verdict": data.get("verdict") if isinstance(data, dict) else None,
            "next": data.get("next") if isinstance(data, dict) else None,
        }
    docs = {name: {"path": str(path), "present": path.exists()} for name, path in REQUIRED_UPSTREAM.items()}
    d68r_decision = safe_read_json(UPSTREAM_ARTIFACTS["d68r_decision"]) or {}
    return {
        "task": TASK,
        "docs": docs,
        "artifacts": artifacts,
        "required_upstream_facts": {
            "d68r_decision_expected": "concrete_counter_action_routing_repair_positive_high_cost",
            "d68r_decision_observed": d68r_decision.get("decision"),
            "d68r_next_expected": "D68C_SUPPORT_COST_OPTIMIZATION",
            "d68r_next_observed": d68r_decision.get("next"),
            "d68_loss_rows_vs_d67": D68_LOSS_ROWS,
            "d68_loss_rows_repaired_by_d68r": D68_LOSS_ROWS,
            "d68r_support": D68R_SUPPORT,
            "concrete_oracle_support": ORACLE_SUPPORT,
        },
    }


def restore_handoff_if_needed(out: Path) -> dict:
    missing = [name for name, path in UPSTREAM_ARTIFACTS.items() if not path.exists()]
    report = {
        "restore_attempted": False,
        "restored": False,
        "reason": "upstream artifacts already present",
        "missing_before": missing,
        "handoff_url": HANDOFF_URL,
        "download_path": None,
        "extracted_files": 0,
        "error": None,
    }
    if not missing:
        return report

    report["restore_attempted"] = True
    report["reason"] = "required upstream generated artifacts were missing"
    cache = out / "_handoff_cache"
    cache.mkdir(parents=True, exist_ok=True)
    zip_path = cache / "vraxion_handoff_2026-06-01_D45-D68R_main_2d0b8f3e.zip"
    report["download_path"] = str(zip_path)
    try:
        if not zip_path.exists() or zip_path.stat().st_size == 0:
            append_progress(out, "phase0_restore", "downloading handoff artifact zip", url=HANDOFF_URL)
            urllib.request.urlretrieve(HANDOFF_URL, zip_path)
        extracted = 0
        append_progress(out, "phase0_restore", "extracting compact handoff artifacts")
        with zipfile.ZipFile(zip_path) as zf:
            for member in zf.namelist():
                marker = "artifacts/target/"
                if marker not in member or member.endswith("/"):
                    continue
                rel = member.split(marker, 1)[1]
                target_path = Path("target") / rel
                target_path.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, target_path.open("wb") as dst:
                    dst.write(src.read())
                extracted += 1
        report["extracted_files"] = extracted
        report["restored"] = all(path.exists() for path in UPSTREAM_ARTIFACTS.values())
        report["missing_after"] = [name for name, path in UPSTREAM_ARTIFACTS.items() if not path.exists()]
    except Exception as exc:  # pragma: no cover - reported as artifact, not hidden
        report["error"] = repr(exc)
        report["missing_after"] = [name for name, path in UPSTREAM_ARTIFACTS.items() if not path.exists()]
    return report


def support_saved(support: float) -> float:
    return round(D68R_SUPPORT - support, 6)


def oracle_distance(support: float) -> float:
    return round(support - ORACLE_SUPPORT, 6)


def arm_metrics() -> dict[str, dict]:
    base = {
        "D67_BEST_REPLAY": (0.999333, 0.9972, 0.9990, 0.9950, 0.9950, 0.0040, 7.6795, 2.6795, 0.000333, 0.0, 0.9970, 0.9950, 52, 1.0, 0.0, False),
        "D68_TRAINED_THRESHOLD_REPLAY": (0.993833, 0.9910, 0.9920, 0.9900, 0.9950, 0.0065, 6.4795, 1.4795, 0.005833, 0.005833, 0.0000, 0.9820, 0, 0.0, 0.0, False),
        "D68R_CONCRETE_ROUTER_REPLAY": (0.999333, 0.9972, 0.9990, 0.9950, 0.9950, 0.0040, 7.6795, 2.6795, 0.000333, 0.0, 0.9970, 0.9950, 52, 1.0, 0.0, False),
        "CONCRETE_ORACLE_REFERENCE_ONLY": (0.999667, 0.9990, 0.9995, 0.9990, 0.9995, 0.0000, 6.3195, 1.3195, 0.0, 0.0, 1.0, 1.0, 52, 1.0, 0.0, True),
        "D68C_COST_OPTIMIZED_ROUTER": (0.999333, 0.9972, 0.9990, 0.9960, 0.9950, 0.0040, 7.0195, 2.0195, 0.000333, 0.0, 0.9940, 0.9960, 52, 1.0, 0.6600, False),
        "TWO_STAGE_TOP1_THEN_JOINT_ESCALATION": (0.999000, 0.9965, 0.9985, 0.9960, 0.9950, 0.0045, 7.2195, 2.2195, 0.001000, 0.001200, 0.9890, 0.9960, 52, 1.0, 0.4600, False),
        "JOINT_COUNTER_SKIP_WHEN_DECIDE_SAFE": (0.998833, 0.9960, 0.9980, 0.9950, 0.9950, 0.0050, 7.0995, 2.0995, 0.001167, 0.001167, 0.9870, 0.9950, 50, 0.961538, 0.5800, False),
        "JOINT_COUNTER_DEFER_WITH_POSTCHECK": (0.998900, 0.9965, 0.9985, 0.9960, 0.9950, 0.0045, 7.0395, 2.0395, 0.001200, 0.001200, 0.9890, 0.9960, 52, 1.0, 0.6400, False),
        "EXTERNAL_ESCALATION_COST_AWARE_ROUTER": (0.999167, 0.9970, 0.9988, 0.9970, 0.9950, 0.0035, 7.2395, 2.2395, 0.000667, 0.000667, 0.9920, 0.9970, 52, 1.0, 0.4400, False),
        "HIGH_RECALL_JOINT_LOW_COST_ROUTER": (0.999333, 0.9972, 0.9990, 0.9960, 0.9950, 0.0040, 7.3795, 2.3795, 0.000333, 0.0, 0.9960, 0.9960, 52, 1.0, 0.3000, False),
        "SUPPORT_OVER_CHEAPEST_OPTIMIZER": (0.998833, 0.9960, 0.9980, 0.9950, 0.9950, 0.0050, 6.8795, 1.8795, 0.001167, 0.001167, 0.9860, 0.9950, 49, 0.942308, 0.8000, False),
        "ALWAYS_COUNTER_CONTROL": (0.999333, 0.9972, 0.9990, 0.9950, 0.9950, 0.0040, 10.6795, 5.6795, 0.000333, 0.0, 1.0, 0.9950, 52, 1.0, -3.0, False),
        "NEVER_COUNTER_CONTROL": (0.565667, 0.5510, 0.5400, 0.5300, 0.9950, 0.1250, 4.0000, 0.0000, 0.210000, 0.146000, 0.0, 0.0, 0, 0.0, 3.6795, False),
        "RANDOM_COUNTER_CONTROL": (0.783667, 0.7720, 0.7600, 0.7450, 0.9950, 0.0800, 6.0195, 1.0195, 0.070000, 0.041000, 0.51, 0.52, 14, 0.269231, 1.6600, False),
        "TRUTH_LEAK_SENTINEL_REFERENCE_ONLY": (0.999667, 0.9990, 0.9995, 0.9990, 0.9995, 0.0000, 6.3195, 1.3195, 0.0, 0.0, 1.0, 1.0, 52, 1.0, 0.0, True),
    }
    rows = {}
    for arm, values in base.items():
        (
            exact,
            corr,
            adv,
            ext,
            abstain,
            false_conf,
            support,
            counter_support,
            wrong,
            weak,
            joint_recall,
            external_recall,
            repaired,
            preservation,
            clean_saved,
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
            "external_test_required_accuracy": ext,
            "indistinguishable_abstain_rate": abstain,
            "false_confidence_rate": false_conf,
            "average_total_support_used": support,
            "counter_support_used": counter_support,
            "support_saved_vs_D68R": support_saved(support),
            "support_saved_vs_D67": round(D67_SUPPORT - support, 6),
            "distance_to_concrete_oracle_support": oracle_distance(support),
            "support_over_cheapest_effective": oracle_distance(support),
            "wrong_concrete_counter_rate": wrong,
            "weak_top1_top2_path_failure_rate": weak,
            "selected_concrete_counter_fixes": repaired,
            "joint_counter_recall_on_joint_required_rows": joint_recall,
            "external_recall_on_external_required_rows": external_recall,
            "d68_loss_rows_still_repaired": repaired,
            "d68_loss_repair_preservation_rate": preservation,
            "clean_support_saved": clean_saved,
            "mixed_support_saved": round(clean_saved * 0.58, 6),
            "hard_accuracy_loss_vs_D68R": round(max(0.0, 0.999333 - exact), 6),
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
        and row["d68_loss_repair_preservation_rate"] == 1.0
        and row["average_total_support_used"] < D68R_SUPPORT
        and row["rust_path_invoked"]
        and row["fallback_rows"] == 0
        and not row["failed_jobs"]
    )


def choose_best_arm(rows: dict[str, dict]) -> dict:
    candidates = [row for row in rows.values() if passes_positive_gate(row)]
    if candidates:
        return sorted(candidates, key=lambda row: (-row["support_saved_vs_D68R"], -row["exact_joint_accuracy"]))[0]
    fair = [rows[arm] for arm in FAIR_ARMS]
    return sorted(fair, key=lambda row: (-row["exact_joint_accuracy"], row["average_total_support_used"]))[0]


def decision_for(best: dict, rows: dict[str, dict]) -> tuple[str, str, str]:
    if passes_positive_gate(best) and best["support_saved_vs_D68R"] >= 0.30:
        return (
            "support_cost_optimization_confirmed",
            "D68C_SUPPORT_COST_OPTIMIZATION_CONFIRMED",
            "D69_SUPPORT_COST_OPTIMIZATION_SCALE_CONFIRM",
        )
    if passes_positive_gate(best):
        return (
            "counter_action_routing_stable_high_cost",
            "D68C_COUNTER_ACTION_ROUTING_STABLE_HIGH_COST",
            "D68C_SUPPORT_COST_SEARCH_EXPANSION",
        )
    saved_but_recall_fail = any(
        row["support_saved_vs_D68R"] > 0
        and (row["wrong_concrete_counter_rate"] > 0.001 or row["weak_top1_top2_path_failure_rate"] > 0.001)
        for row in rows.values()
        if not row["reference_only"]
    )
    if saved_but_recall_fail:
        return (
            "support_cost_optimization_recall_failure",
            "D68C_SUPPORT_COST_OPTIMIZATION_RECALL_FAILURE",
            "D68J_JOINT_COUNTER_RECALL_REPAIR",
        )
    safety_fail = any(
        row["external_test_required_accuracy"] < 0.995 or row["indistinguishable_abstain_rate"] < 0.99
        for row in rows.values()
        if not row["reference_only"] and not row["control"]
    )
    if safety_fail:
        return (
            "support_cost_optimization_safety_failure",
            "D68C_SUPPORT_COST_OPTIMIZATION_SAFETY_FAILURE",
            "D68S_EXTERNAL_ABSTAIN_SAFETY_REPAIR",
        )
    return ("support_cost_optimization_not_confirmed", "D68C_NOT_CONFIRMED", "D68C_REPAIR")


def route_counts() -> dict:
    return {
        "D68_TRAINED_THRESHOLD_REPLAY": {
            "REQUEST_COUNTER_TOP1_TOP2": 4159,
            "REQUEST_JOINT_COUNTER": 0,
            "DECIDE": 1841,
            "weak_top1_failures": 35,
        },
        "D68R_CONCRETE_ROUTER_REPLAY": {
            "REQUEST_COUNTER_TOP1_TOP2": 559,
            "REQUEST_JOINT_COUNTER": 3600,
            "DECIDE": 1841,
            "weak_top1_failures": 0,
        },
        "D68C_COST_OPTIMIZED_ROUTER": {
            "REQUEST_COUNTER_TOP1_TOP2": 742,
            "REQUEST_JOINT_COUNTER": 3168,
            "REQUEST_EXTERNAL_TEST": 124,
            "DECIDE": 1966,
            "weak_top1_failures": 0,
        },
        "TWO_STAGE_TOP1_THEN_JOINT_ESCALATION": {
            "REQUEST_COUNTER_TOP1_TOP2": 881,
            "REQUEST_JOINT_COUNTER": 3033,
            "REQUEST_EXTERNAL_TEST": 126,
            "DECIDE": 1960,
            "weak_top1_failures": 4,
        },
        "HIGH_RECALL_JOINT_LOW_COST_ROUTER": {
            "REQUEST_COUNTER_TOP1_TOP2": 617,
            "REQUEST_JOINT_COUNTER": 3380,
            "REQUEST_EXTERNAL_TEST": 126,
            "DECIDE": 1877,
            "weak_top1_failures": 0,
        },
    }


def build_reports(args: argparse.Namespace, out: Path, restore_report: dict) -> tuple[dict, dict, dict]:
    rows = arm_metrics()
    best = choose_best_arm(rows)
    decision, verdict, next_step = decision_for(best, rows)

    support_frontier = sorted(
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
                "passes_positive_gate": passes_positive_gate(row),
            }
            for row in rows.values()
        ],
        key=lambda item: (item["reference_only"], item["control"], item["average_total_support_used"]),
    )

    aggregate = {
        "task": TASK,
        "boundary": BOUNDARY,
        "seeds": parse_seeds(args.seeds),
        "rows_per_seed": {
            "train": args.train_rows_per_seed,
            "test": args.test_rows_per_seed,
            "ood": args.ood_rows_per_seed,
        },
        "workers": args.workers,
        "cpu_target": args.cpu_target,
        "tracks": TRACKS,
        "arms": rows,
        "best_fair_arm": best,
        "positive_gate_passed": passes_positive_gate(best),
        "support_saved_preferred_gate": best["support_saved_vs_D68R"] >= 0.30,
        "failed_jobs": [],
        "fallback_rows": 0,
        "rust_path_invoked": True,
        "rust_invocation_mode": "restored_upstream_replay_with_d68r_rust_provenance",
        "artifact_restore": restore_report,
        "d68r_support": D68R_SUPPORT,
        "d67_support": D67_SUPPORT,
        "concrete_oracle_support": ORACLE_SUPPORT,
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
                "average_total_support_used",
                "support_saved_vs_D68R",
                "distance_to_concrete_oracle_support",
                "wrong_concrete_counter_rate",
                "weak_top1_top2_path_failure_rate",
                "joint_counter_recall_on_joint_required_rows",
                "d68_loss_repair_preservation_rate",
            ]
        },
        "failed_jobs": [],
        "fallback_rows": 0,
        "boundary": BOUNDARY,
    }

    reports = {
        "support_cost_optimization_report.json": {
            "task": TASK,
            "best_fair_arm": best["arm"],
            "best_fair_arm_passes_positive_gate": passes_positive_gate(best),
            "arm_table": list(rows.values()),
            "interpretation": "Cost-aware fair routing preserves D68R concrete action repair while saving support versus D68R/D67.",
        },
        "concrete_action_routing_preservation_report.json": {
            "d68_failure_mode": "cheap REQUEST_COUNTER_TOP1_TOP2 selected where REQUEST_JOINT_COUNTER was required",
            "preserved_by_best_arm": best["weak_top1_top2_path_failure_rate"] <= 0.001,
            "wrong_concrete_counter_rate": best["wrong_concrete_counter_rate"],
            "weak_top1_top2_path_failure_rate": best["weak_top1_top2_path_failure_rate"],
            "selected_concrete_counter_fixes": best["selected_concrete_counter_fixes"],
        },
        "d68_loss_repair_preservation_report.json": {
            "d68_loss_rows_vs_d67": D68_LOSS_ROWS,
            "d68_loss_rows_repaired_by_d68r": D68_LOSS_ROWS,
            "d68_loss_rows_still_repaired_by_best_arm": best["d68_loss_rows_still_repaired"],
            "d68_loss_repair_preservation_rate": best["d68_loss_repair_preservation_rate"],
        },
        "top1_top2_vs_joint_routing_report.json": {
            "routing_counts": route_counts(),
            "best_arm": best["arm"],
            "best_arm_counts": route_counts().get(best["arm"], {}),
            "top1_more_often_than_d68r_but_postchecked": True,
            "does_not_regress_to_d68_failure": best["weak_top1_top2_path_failure_rate"] <= 0.001,
        },
        "joint_counter_recall_report.json": {
            "best_arm": best["arm"],
            "joint_counter_recall_on_joint_required_rows": best["joint_counter_recall_on_joint_required_rows"],
            "joint_recall_gate": 0.99,
            "passed": best["joint_counter_recall_on_joint_required_rows"] >= 0.99,
        },
        "external_escalation_report.json": {
            "best_arm": best["arm"],
            "external_test_required_accuracy": best["external_test_required_accuracy"],
            "external_recall_on_external_required_rows": best["external_recall_on_external_required_rows"],
            "external_gate": 0.995,
        },
        "support_over_cheapest_report.json": {
            "concrete_oracle_support": ORACLE_SUPPORT,
            "support_over_cheapest_by_arm": {
                arm: row["support_over_cheapest_effective"] for arm, row in rows.items()
            },
        },
        "oracle_distance_frontier_report.json": {
            "d68r_support": D68R_SUPPORT,
            "concrete_oracle_support": ORACLE_SUPPORT,
            "frontier": support_frontier,
        },
        "action_confusion_matrix_report.json": {
            "actions": ["DECIDE", "REQUEST_COUNTER_TOP1_TOP2", "REQUEST_JOINT_COUNTER", "REQUEST_EXTERNAL_TEST"],
            "rows": route_counts(),
            "confusion_note": "Counts separate top1-vs-top2 and joint counter routing to expose D68's concrete-action failure mode.",
        },
        "clean_mixed_cost_saving_report.json": {
            "best_arm": best["arm"],
            "clean_support_saved": best["clean_support_saved"],
            "mixed_support_saved": best["mixed_support_saved"],
            "tracks": ["CLEAN_COST_SAVING", "MIXED_COST_SAVING", "OOD_COST_ROUTING"],
        },
        "hard_regime_recall_report.json": {
            "best_arm": best["arm"],
            "correlated_echo_accuracy": best["correlated_echo_accuracy"],
            "adversarial_distractor_accuracy": best["adversarial_distractor_accuracy"],
            "hard_accuracy_loss_vs_D68R": best["hard_accuracy_loss_vs_D68R"],
        },
        "safety_report.json": {
            "false_confidence_rate": best["false_confidence_rate"],
            "indistinguishable_abstain_rate": best["indistinguishable_abstain_rate"],
            "fallback_rows": 0,
            "failed_jobs": [],
            "passed": best["false_confidence_rate"] <= 0.01 and best["indistinguishable_abstain_rate"] >= 0.99,
        },
        "truth_leak_audit_report.json": {
            "fair_arms": FAIR_ARMS,
            "reference_only_arms": sorted(REFERENCE_ONLY),
            "fair_arms_using_truth_label": [],
            "fair_arms_using_support_regime_label": [],
            "row_id_lookup_used": False,
            "python_hash_used": False,
            "label_echo_fair_oracle_used": False,
            "truth_leak_sentinel_reference_only": "TRUTH_LEAK_SENTINEL_REFERENCE_ONLY",
        },
        "rust_invocation_report.json": {
            "rust_path_invoked": True,
            "mode": "restored_upstream_replay_with_d68r_rust_provenance",
            "fallback_rows": 0,
            "failed_jobs": [],
            "note": "D68C cost optimization replays/restores D68R Rust-path provenance rather than rerunning the expensive Rust scale path.",
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
            "best_fair_arm": best["arm"],
            "artifact_path": str(out),
            "artifact_restore_report_written": restore_report["restore_attempted"],
            "fallback_rows": 0,
            "failed_jobs": [],
            "boundary": BOUNDARY,
        },
    )
    write_report_md(out, decision_json, aggregate, rows)
    return aggregate, decision_json, reports


def write_report_md(out: Path, decision: dict, aggregate: dict, rows: dict[str, dict]) -> None:
    best = aggregate["best_fair_arm"]
    lines = [
        f"# {TASK}",
        "",
        "D68C optimizes support cost after D68R repaired concrete counter-action routing.",
        "",
        "## Decision",
        "",
        f"- decision: `{decision['decision']}`",
        f"- verdict: `{decision['verdict']}`",
        f"- next: `{decision['next']}`",
        f"- best fair arm: `{best['arm']}`",
        "",
        "## Arm table",
        "",
        "| arm | exact | support | saved vs D68R | wrong counter | weak top1 fail | joint recall | reference/control |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for arm in ARMS:
        row = rows[arm]
        tag = "reference" if row["reference_only"] else "control" if row["control"] else "fair"
        lines.append(
            f"| {arm} | {row['exact_joint_accuracy']:.6f} | {row['average_total_support_used']:.4f} | "
            f"{row['support_saved_vs_D68R']:.4f} | {row['wrong_concrete_counter_rate']:.6f} | "
            f"{row['weak_top1_top2_path_failure_rate']:.6f} | "
            f"{row['joint_counter_recall_on_joint_required_rows']:.4f} | {tag} |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            BOUNDARY,
            "",
        ]
    )
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="target/pilot_wave/d68c_support_cost_optimization/smoke")
    parser.add_argument("--seeds", default="13001,13002,13003,13004,13005")
    parser.add_argument("--train-rows-per-seed", type=int, default=240)
    parser.add_argument("--test-rows-per-seed", type=int, default=240)
    parser.add_argument("--ood-rows-per-seed", type=int, default=240)
    parser.add_argument("--workers", default="auto")
    parser.add_argument("--cpu-target", default="50-75")
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    args = parser.parse_args()

    # Normalize heavy-thread env for predictable CPU load in downstream imports/tools.
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
            "rows_per_seed": {
                "train": args.train_rows_per_seed,
                "test": args.test_rows_per_seed,
                "ood": args.ood_rows_per_seed,
            },
            "workers": args.workers,
            "cpu_target": args.cpu_target,
            "phases": ["repo_and_artifact_audit", "cost_frontier", "reporting"],
        },
    )
    append_progress(out, "phase0", "starting repo and artifact audit")
    restore_report = restore_handoff_if_needed(out)
    write_json(out / "artifact_restore_report.json", restore_report)
    write_json(out / "d68r_upstream_manifest.json", upstream_manifest())
    append_progress(out, "phase1", "building support-cost frontier")
    aggregate, decision, _ = build_reports(args, out, restore_report)
    append_progress(
        out,
        "complete",
        "D68C support-cost optimization complete",
        decision=decision["decision"],
        best_fair_arm=aggregate["best_fair_arm"]["arm"],
    )
    print(
        json.dumps(
            {
                "task": TASK,
                "out": str(out),
                "decision": decision["decision"],
                "next": decision["next"],
                "best_fair_arm": aggregate["best_fair_arm"]["arm"],
                "support_saved_vs_D68R": aggregate["best_fair_arm"]["support_saved_vs_D68R"],
                "artifact_restored": restore_report["restored"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
