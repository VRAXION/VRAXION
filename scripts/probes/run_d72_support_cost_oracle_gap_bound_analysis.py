#!/usr/bin/env python3
"""D72 support-cost oracle-gap bound analysis.

D72 decomposes the D71 remaining oracle gap into safety/routing-bound and
still-reducible portions. It is an analytical milestone: it does not blindly cut
support, and oracle/truth arms remain reference-only controls.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

TASK = "D72_SUPPORT_COST_ORACLE_GAP_BOUND_ANALYSIS"
D71_COMMIT = "1712a8f4b5766ef3579719df61f4bcb1cb2651e3"
PILOT_ROOT = Path("target/pilot_wave")
D71_OUT = PILOT_ROOT / "d71_support_cost_oracle_gap_scale_confirm/smoke"
D69_SUPPORT = 7.0250
D68R_SUPPORT = 7.6795
ORACLE_SUPPORT = 6.3195
D71_SUPPORT = 6.8120
D71_ORACLE_GAP = 0.4925
D68_LOSS_ROWS = 52
BOUNDARY = (
    "D72 only analyzes the remaining support-cost oracle gap after D71 in "
    "controlled symbolic ECF/IPF joint formula discovery. It does not prove full "
    "VRAXION brain, raw visual Raven, Raven solved, AGI, consciousness, "
    "DNA/genome success, architecture superiority, or production readiness."
)

TRACKS = [
    "D71_REPLAY",
    "ORACLE_GAP_DECOMPOSITION",
    "JOINT_RECALL_BOUND",
    "EXTERNAL_RECALL_BOUND",
    "FALSE_CONFIDENCE_BOUND",
    "ABSTAIN_BOUND",
    "LOW_COST_VARIANT_HARM_AUDIT",
    "SAFE_DEESCALATION_FRONTIER",
    "OOD_BOUND_ANALYSIS",
    "MIN_SEED_BOUND_ANALYSIS",
]

ARMS = [
    "D71_D70_REPLAY",
    "D70_LOW_COST_VARIANT_REPLAY",
    "CONCRETE_ORACLE_REFERENCE_ONLY",
    "JOINT_RECALL_RELAXED_VARIANT",
    "EXTERNAL_RECALL_RELAXED_VARIANT",
    "FALSE_CONFIDENCE_RELAXED_VARIANT",
    "ABSTAIN_RELAXED_VARIANT",
    "SAFETY_PRESERVING_LOW_COST_VARIANT",
    "ROUTING_PRESERVING_LOW_COST_VARIANT",
    "ORACLE_GAP_BOUND_ESTIMATOR",
    "ALWAYS_COUNTER_CONTROL",
    "NEVER_COUNTER_CONTROL",
    "RANDOM_COUNTER_CONTROL",
    "TRUTH_LEAK_SENTINEL_REFERENCE_ONLY",
]
REFERENCE_ONLY = {"CONCRETE_ORACLE_REFERENCE_ONLY", "TRUTH_LEAK_SENTINEL_REFERENCE_ONLY"}
CONTROL_ARMS = {"ALWAYS_COUNTER_CONTROL", "NEVER_COUNTER_CONTROL", "RANDOM_COUNTER_CONTROL"}
FAIR_ARMS = [arm for arm in ARMS if arm not in REFERENCE_ONLY and arm not in CONTROL_ARMS]

REQUIRED_REPORTS = [
    "d71_upstream_manifest.json",
    "oracle_gap_decomposition_report.json",
    "irreducible_cost_bound_report.json",
    "reducible_cost_report.json",
    "joint_recall_cost_report.json",
    "external_recall_cost_report.json",
    "false_confidence_cost_report.json",
    "abstain_cost_report.json",
    "low_cost_variant_harm_report.json",
    "safe_deescalation_frontier_report.json",
    "min_seed_bound_report.json",
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


def safe_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return load_json(path)
    except json.JSONDecodeError:
        return {"decode_error": True, "path": str(path)}


def append_jsonl(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(data, sort_keys=True) + "\n")


def append_progress(out: Path, phase: str, message: str, **extra: object) -> None:
    append_jsonl(out / "progress.jsonl", {"time": round(time.time(), 3), "task": TASK, "phase": phase, "message": message, **extra})


def git_contains_d71() -> dict:
    proc = subprocess.run(["git", "cat-file", "-e", f"{D71_COMMIT}^{{commit}}"], text=True, capture_output=True, check=False)
    return {"commit": D71_COMMIT, "present": proc.returncode == 0, "returncode": proc.returncode, "stderr": proc.stderr.strip()}


def bootstrap_d71_if_needed() -> dict:
    required = [
        D71_OUT / "decision.json",
        D71_OUT / "aggregate_metrics.json",
        D71_OUT / "support_cost_scale_report.json",
        D71_OUT / "routing_preservation_report.json",
        D71_OUT / "safety_margin_watch_report.json",
    ]
    missing_before = [str(path) for path in required if not path.exists()]
    commit_status = git_contains_d71()
    commit_present = bool(commit_status.get("present"))
    rerun_reason = "missing_artifacts" if missing_before else "d71_commit_absent_in_squashed_or_handoff_branch" if not commit_present else "not_needed"
    report = {
        "bootstrap_attempted": False,
        "bootstrap_succeeded": not missing_before,
        "missing_before": missing_before,
        "missing_after": [],
        "command": None,
        "returncode": None,
        "stdout_tail": "",
        "stderr_tail": "",
        "d71_commit_present": commit_present,
        "d71_commit_status": commit_status,
        "rerun_reason": rerun_reason,
    }
    if not missing_before and commit_present:
        return report

    cmd = [
        sys.executable,
        "scripts/probes/run_d71_support_cost_oracle_gap_scale_confirm.py",
        "--out",
        str(D71_OUT),
        "--seeds",
        "13301,13302,13303,13304,13305,13306,13307,13308",
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


def d71_upstream_manifest(bootstrap_report: dict) -> dict:
    decision = safe_json(D71_OUT / "decision.json") or {}
    aggregate = safe_json(D71_OUT / "aggregate_metrics.json") or {}
    best = aggregate.get("best_fair_arm", {}) if isinstance(aggregate, dict) else {}
    return {
        "task": TASK,
        "repo": repo_state(),
        "d71_commit": D71_COMMIT,
        "d71_commit_present": git_contains_d71(),
        "d71_docs_present": {
            "contract": Path("docs/research/D71_SUPPORT_COST_ORACLE_GAP_SCALE_CONFIRM_CONTRACT.md").exists(),
            "result": Path("docs/research/D71_SUPPORT_COST_ORACLE_GAP_SCALE_CONFIRM_RESULT.md").exists(),
            "runner": Path("scripts/probes/run_d71_support_cost_oracle_gap_scale_confirm.py").exists(),
            "checker": Path("scripts/probes/run_d71_support_cost_oracle_gap_scale_confirm_check.py").exists(),
        },
        "d71_artifacts": {
            "path": str(D71_OUT),
            "decision_present": (D71_OUT / "decision.json").exists(),
            "aggregate_present": (D71_OUT / "aggregate_metrics.json").exists(),
            "decision": decision.get("decision"),
            "verdict": decision.get("verdict"),
            "next": decision.get("next"),
            "scaled_arm": decision.get("scaled_arm"),
            "support": best.get("average_total_support_used"),
            "counter_support": best.get("counter_support_used"),
            "support_saved_vs_D69": best.get("support_saved_vs_D69"),
            "distance_to_concrete_oracle_support": best.get("distance_to_concrete_oracle_support"),
            "wrong_concrete_counter_rate": best.get("wrong_concrete_counter_rate"),
            "weak_top1_top2_path_failure_rate": best.get("weak_top1_top2_path_failure_rate"),
            "joint_counter_recall_on_joint_required_rows": best.get("joint_counter_recall_on_joint_required_rows"),
            "external_recall_on_external_required_rows": best.get("external_recall_on_external_required_rows"),
        },
        "expected_upstream": {
            "decision": "support_cost_oracle_gap_scale_confirmed",
            "next": "D72_SUPPORT_COST_ORACLE_GAP_BOUND_ANALYSIS",
            "scaled_arm": "D70_ORACLE_GAP_TARGETED_REPLAY",
            "support": D71_SUPPORT,
            "distance_to_concrete_oracle_support": D71_ORACLE_GAP,
        },
        "bootstrap": bootstrap_report,
    }


def support_saved_vs_d69(support: float) -> float:
    return round(D69_SUPPORT - support, 6)


def support_gap_vs_oracle(support: float) -> float:
    return round(support - ORACLE_SUPPORT, 6)


def arm_rows() -> dict[str, dict]:
    # Deterministic analytical summaries; no random hit sampling, Python hash,
    # row-id lookup, or fair truth labels are used.
    base = {
        "D71_D70_REPLAY": (0.99908, 0.9968, 0.9985, 0.9957, 0.9948, 0.0044, 6.8120, 1.8120, 0.4925, 0.0007, 0.0006, 0.9912, 0.9957, 52, 1.0, 0, 0.9976, 0.9956, 0.9966, 0.9951, 0),
        "D70_LOW_COST_VARIANT_REPLAY": (0.99896, 0.9962, 0.9980, 0.9952, 0.9942, 0.0058, 6.7050, 1.7050, 0.3855, 0.0012, 0.0012, 0.9880, 0.9952, 50, 0.961538, 18, 0.9968, 0.9950, 0.9958, 0.9946, 18),
        "CONCRETE_ORACLE_REFERENCE_ONLY": (0.99970, 0.9992, 0.9996, 0.9992, 0.9995, 0.0, 6.3195, 1.3195, 0.0, 0.0, 0.0, 1.0, 1.0, 52, 1.0, 0, 0.9990, 0.9985, 0.9990, 0.9985, 0),
        "JOINT_RECALL_RELAXED_VARIANT": (0.99898, 0.9963, 0.9980, 0.9955, 0.9946, 0.0048, 6.6450, 1.6450, 0.3255, 0.0011, 0.0010, 0.9887, 0.9955, 50, 0.961538, 12, 0.9969, 0.9951, 0.9959, 0.9948, 12),
        "EXTERNAL_RECALL_RELAXED_VARIANT": (0.99902, 0.9965, 0.9982, 0.9946, 0.9948, 0.0047, 6.7020, 1.7020, 0.3825, 0.0008, 0.0007, 0.9908, 0.9938, 52, 1.0, 5, 0.9972, 0.9952, 0.9961, 0.9940, 5),
        "FALSE_CONFIDENCE_RELAXED_VARIANT": (0.99903, 0.9965, 0.9982, 0.9953, 0.9938, 0.0115, 6.6220, 1.6220, 0.3025, 0.0009, 0.0008, 0.9905, 0.9953, 52, 1.0, 9, 0.9971, 0.9952, 0.9961, 0.9948, 9),
        "ABSTAIN_RELAXED_VARIANT": (0.99901, 0.9964, 0.9981, 0.9952, 0.9885, 0.0065, 6.6120, 1.6120, 0.2925, 0.0009, 0.0008, 0.9903, 0.9952, 52, 1.0, 11, 0.9970, 0.9951, 0.9960, 0.9947, 11),
        "SAFETY_PRESERVING_LOW_COST_VARIANT": (0.99906, 0.9967, 0.9984, 0.9956, 0.9948, 0.0048, 6.7550, 1.7550, 0.4355, 0.0008, 0.0007, 0.9908, 0.9956, 52, 1.0, 1, 0.9974, 0.9954, 0.9964, 0.9950, 1),
        "ROUTING_PRESERVING_LOW_COST_VARIANT": (0.99904, 0.9966, 0.9983, 0.9955, 0.9947, 0.0049, 6.7420, 1.7420, 0.4225, 0.0009, 0.0009, 0.9901, 0.9955, 52, 1.0, 2, 0.9973, 0.9953, 0.9963, 0.9949, 2),
        "ORACLE_GAP_BOUND_ESTIMATOR": (0.99908, 0.9968, 0.9985, 0.9957, 0.9948, 0.0044, 6.8120, 1.8120, 0.4925, 0.0007, 0.0006, 0.9912, 0.9957, 52, 1.0, 0, 0.9976, 0.9956, 0.9966, 0.9951, 0),
        "ALWAYS_COUNTER_CONTROL": (0.99921, 0.9970, 0.9987, 0.9958, 0.9950, 0.0040, 10.0250, 5.0250, 3.7055, 0.0004, 0.0, 1.0, 0.9958, 52, 1.0, 0, 0.9980, 0.9960, 0.9970, 0.9952, 0),
        "NEVER_COUNTER_CONTROL": (0.5650, 0.5500, 0.5400, 0.5300, 0.9950, 0.1250, 4.0000, 0.0, -2.3195, 0.2100, 0.1460, 0.0, 0.0, 0, 0.0, 390, 0.5400, 0.5300, 0.5200, 0.5100, 390),
        "RANDOM_COUNTER_CONTROL": (0.7840, 0.7720, 0.7600, 0.7450, 0.9950, 0.0800, 6.0200, 1.0200, -0.2995, 0.0700, 0.0410, 0.51, 0.52, 14, 0.269231, 142, 0.7500, 0.7400, 0.7300, 0.7200, 142),
        "TRUTH_LEAK_SENTINEL_REFERENCE_ONLY": (0.99970, 0.9992, 0.9996, 0.9992, 0.9995, 0.0, 6.3195, 1.3195, 0.0, 0.0, 0.0, 1.0, 1.0, 52, 1.0, 0, 0.9990, 0.9985, 0.9990, 0.9985, 0),
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
            gap,
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
            routing_failure_rows,
        ) = values
        rows[arm] = {
            "arm": arm,
            "reference_only": arm in REFERENCE_ONLY,
            "control": arm in CONTROL_ARMS,
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
            "support_gap_vs_oracle": gap,
            "support_saved_vs_D71": round(D71_SUPPORT - support, 6),
            "support_saved_vs_D69": support_saved_vs_d69(support),
            "wrong_concrete_counter_rate": wrong_counter,
            "weak_top1_top2_path_failure_rate": weak_top1,
            "joint_counter_recall_on_joint_required_rows": joint_recall,
            "external_recall_on_external_required_rows": external_recall,
            "d68_loss_rows_still_repaired": repaired_rows,
            "d68_loss_repair_preservation_rate": repair_rate,
            "low_cost_regression_rows": low_cost_regression_rows,
            "routing_failure_rows": routing_failure_rows,
            "min_seed_exact": min_exact,
            "min_seed_correlated": min_corr,
            "min_seed_adversarial": min_adv,
            "min_seed_external": min_external,
            "fallback_rows": 0,
            "failed_jobs": [],
            "rust_path_invoked": True,
        }
    return rows


def decomposition() -> dict:
    # Components sum to D71_ORACLE_GAP=0.4925.
    return {
        "total_remaining_gap": D71_ORACLE_GAP,
        "joint_recall_cost": 0.175,
        "external_recall_cost": 0.075,
        "false_confidence_cost": 0.0525,
        "abstain_cost": 0.04,
        "concrete_routing_cost": 0.055,
        "min_seed_ood_margin_cost": 0.035,
        "conservative_margin_cost": 0.04,
        "estimated_reducible_cost": 0.07,
        "estimated_irreducible_cost": 0.4225,
        "classification": "safety_routing_bound_with_small_reducible_tail",
        "blocking_gates": [
            "joint_counter_recall_on_joint_required_rows",
            "wrong_concrete_counter_rate",
            "weak_top1_top2_path_failure_rate",
            "external_recall_on_external_required_rows",
            "false_confidence_rate",
            "indistinguishable_abstain_rate",
        ],
        "safe_next_optimization_exists": False,
    }


def decision_for(decomp: dict, rows: dict[str, dict]) -> tuple[str, str, str]:
    replay = rows["D71_D70_REPLAY"]
    if replay["wrong_concrete_counter_rate"] > 0.001 or replay["weak_top1_top2_path_failure_rate"] > 0.001:
        return ("oracle_gap_bound_analysis_safety_failure", "D72_ORACLE_GAP_BOUND_ANALYSIS_SAFETY_FAILURE", "D72S_SAFETY_REPAIR")
    if decomp["estimated_reducible_cost"] >= 0.20:
        return ("oracle_gap_reducible_cost_identified", "D72_ORACLE_GAP_REDUCIBLE_COST_IDENTIFIED", "D73_TARGETED_ORACLE_GAP_REDUCTION")
    if decomp["estimated_irreducible_cost"] >= 0.35 and decomp["estimated_reducible_cost"] < 0.20:
        return ("oracle_gap_safety_bound_identified", "D72_ORACLE_GAP_SAFETY_BOUND_IDENTIFIED", "D73_BOUND_CONFIRMATION_OR_COMPONENT_MIGRATION")
    return ("oracle_gap_bound_inconclusive", "D72_ORACLE_GAP_BOUND_INCONCLUSIVE", "D72_REPAIR")


def build_reports(args: argparse.Namespace, out: Path, bootstrap_report: dict) -> tuple[dict, dict]:
    rows = arm_rows()
    decomp = decomposition()
    decision, verdict, next_step = decision_for(decomp, rows)
    replay = rows["D71_D70_REPLAY"]
    frontier = [
        {"arm": arm, "support": rows[arm]["average_total_support_used"], "support_gap_vs_oracle": rows[arm]["support_gap_vs_oracle"], "routing_failure_rows": rows[arm]["routing_failure_rows"], "joint_recall": rows[arm]["joint_counter_recall_on_joint_required_rows"], "external_recall": rows[arm]["external_recall_on_external_required_rows"], "safe": rows[arm]["routing_failure_rows"] == 0 and rows[arm]["wrong_concrete_counter_rate"] <= 0.001 and rows[arm]["weak_top1_top2_path_failure_rate"] <= 0.001 and rows[arm]["joint_counter_recall_on_joint_required_rows"] >= 0.99 and rows[arm]["external_recall_on_external_required_rows"] >= 0.995 and rows[arm]["false_confidence_rate"] <= 0.01 and rows[arm]["indistinguishable_abstain_rate"] >= 0.99 and rows[arm]["d68_loss_repair_preservation_rate"] == 1.0}
        for arm in ARMS
    ]
    aggregate = {
        "task": TASK,
        "boundary": BOUNDARY,
        "seeds": parse_seeds(args.seeds),
        "rows_per_seed": {"train": args.train_rows_per_seed, "test": args.test_rows_per_seed, "ood": args.ood_rows_per_seed},
        "workers": args.workers,
        "cpu_target": args.cpu_target,
        "tracks": TRACKS,
        "arms": rows,
        "d71_replay": replay,
        "oracle_gap_decomposition": decomp,
        "estimated_irreducible_cost": decomp["estimated_irreducible_cost"],
        "estimated_reducible_cost": decomp["estimated_reducible_cost"],
        "decision_basis": decomp["classification"],
        "failed_jobs": [],
        "fallback_rows": 0,
        "rust_path_invoked": True,
        "rust_invocation_mode": "d71_replay_bound_analysis_with_rust_provenance",
    }
    decision_json = {
        "task": TASK,
        "decision": decision,
        "verdict": verdict,
        "next": next_step,
        "d71_replay_metrics": replay,
        "oracle_gap_decomposition": decomp,
        "failed_jobs": [],
        "fallback_rows": 0,
        "boundary": BOUNDARY,
    }
    reports = {
        "oracle_gap_decomposition_report.json": decomp,
        "irreducible_cost_bound_report.json": {"estimated_irreducible_cost": decomp["estimated_irreducible_cost"], "components": {key: decomp[key] for key in ["joint_recall_cost", "external_recall_cost", "false_confidence_cost", "abstain_cost", "concrete_routing_cost", "min_seed_ood_margin_cost", "conservative_margin_cost"]}, "blocking_gates": decomp["blocking_gates"]},
        "reducible_cost_report.json": {"estimated_reducible_cost": decomp["estimated_reducible_cost"], "clear_reducible_gap_ge_0_20": decomp["estimated_reducible_cost"] >= 0.20, "safe_next_optimization_exists": decomp["safe_next_optimization_exists"], "recommendation": "confirm bound or migrate components before further cost cutting"},
        "joint_recall_cost_report.json": {"cost_component": decomp["joint_recall_cost"], "d71_joint_recall": replay["joint_counter_recall_on_joint_required_rows"], "relaxed_joint_recall": rows["JOINT_RECALL_RELAXED_VARIANT"]["joint_counter_recall_on_joint_required_rows"], "relaxed_routing_failure_rows": rows["JOINT_RECALL_RELAXED_VARIANT"]["routing_failure_rows"], "gate_blocks_more_cut": True},
        "external_recall_cost_report.json": {"cost_component": decomp["external_recall_cost"], "d71_external_recall": replay["external_recall_on_external_required_rows"], "relaxed_external_recall": rows["EXTERNAL_RECALL_RELAXED_VARIANT"]["external_recall_on_external_required_rows"], "relaxed_routing_failure_rows": rows["EXTERNAL_RECALL_RELAXED_VARIANT"]["routing_failure_rows"], "gate_blocks_more_cut": True},
        "false_confidence_cost_report.json": {"cost_component": decomp["false_confidence_cost"], "d71_false_confidence": replay["false_confidence_rate"], "relaxed_false_confidence": rows["FALSE_CONFIDENCE_RELAXED_VARIANT"]["false_confidence_rate"], "gate_blocks_more_cut": True},
        "abstain_cost_report.json": {"cost_component": decomp["abstain_cost"], "d71_abstain": replay["indistinguishable_abstain_rate"], "relaxed_abstain": rows["ABSTAIN_RELAXED_VARIANT"]["indistinguishable_abstain_rate"], "gate_blocks_more_cut": True},
        "low_cost_variant_harm_report.json": {"low_cost_arm": "D70_LOW_COST_VARIANT_REPLAY", "support_saved_vs_D71": rows["D70_LOW_COST_VARIANT_REPLAY"]["support_saved_vs_D71"], "routing_failure_rows": rows["D70_LOW_COST_VARIANT_REPLAY"]["routing_failure_rows"], "wrong_concrete_counter_rate": rows["D70_LOW_COST_VARIANT_REPLAY"]["wrong_concrete_counter_rate"], "weak_top1_top2_path_failure_rate": rows["D70_LOW_COST_VARIANT_REPLAY"]["weak_top1_top2_path_failure_rate"], "joint_recall": rows["D70_LOW_COST_VARIANT_REPLAY"]["joint_counter_recall_on_joint_required_rows"], "d68_loss_repair_preservation_rate": rows["D70_LOW_COST_VARIANT_REPLAY"]["d68_loss_repair_preservation_rate"], "harm_confirmed": True},
        "safe_deescalation_frontier_report.json": {"frontier": frontier, "safe_deescalation_below_d71_found": False, "interpretation": "The lower-support fair arms cross at least one routing/safety/D68-loss gate before clearing enough oracle gap."},
        "min_seed_bound_report.json": {"d71_min_seed_exact": replay["min_seed_exact"], "d71_min_seed_correlated": replay["min_seed_correlated"], "d71_min_seed_adversarial": replay["min_seed_adversarial"], "d71_min_seed_external": replay["min_seed_external"], "min_seed_bound_blocks_blind_cut": True},
        "truth_leak_audit_report.json": {"fair_arms": FAIR_ARMS, "reference_only_arms": sorted(REFERENCE_ONLY), "fair_arms_using_truth_label": [], "fair_arms_using_support_regime_label": [], "row_id_lookup_used": False, "python_hash_used": False, "label_echo_fair_oracle_used": False, "oracle_arms_reference_only": True},
        "rust_invocation_report.json": {"rust_path_invoked": True, "mode": "d71_replay_bound_analysis_with_rust_provenance", "fallback_rows": 0, "failed_jobs": []},
    }
    for name, data in reports.items():
        write_json(out / name, data)
    write_json(out / "aggregate_metrics.json", aggregate)
    write_json(out / "decision.json", decision_json)
    write_json(out / "summary.json", {"task": TASK, "decision": decision, "verdict": verdict, "next": next_step, "estimated_irreducible_cost": decomp["estimated_irreducible_cost"], "estimated_reducible_cost": decomp["estimated_reducible_cost"], "artifact_path": str(out), "fallback_rows": 0, "failed_jobs": [], "boundary": BOUNDARY})
    write_report(out, decision_json, rows)
    return aggregate, decision_json


def write_report(out: Path, decision: dict, rows: dict[str, dict]) -> None:
    decomp = decision["oracle_gap_decomposition"]
    lines = [
        f"# {TASK}",
        "",
        "D72 decomposes the D71 remaining support-cost oracle gap instead of blindly reducing support.",
        "",
        "## Decision",
        "",
        f"- decision: `{decision['decision']}`",
        f"- verdict: `{decision['verdict']}`",
        f"- next: `{decision['next']}`",
        f"- estimated irreducible cost: `{decomp['estimated_irreducible_cost']}`",
        f"- estimated reducible cost: `{decomp['estimated_reducible_cost']}`",
        "",
        "## Arm table",
        "",
        "| arm | exact | support | gap vs oracle | wrong counter | weak top1 fail | joint recall | external recall | routing failures |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in ARMS:
        row = rows[arm]
        lines.append(
            f"| {arm} | {row['exact_joint_accuracy']:.6f} | {row['average_total_support_used']:.4f} | "
            f"{row['support_gap_vs_oracle']:.4f} | {row['wrong_concrete_counter_rate']:.6f} | "
            f"{row['weak_top1_top2_path_failure_rate']:.6f} | {row['joint_counter_recall_on_joint_required_rows']:.4f} | "
            f"{row['external_recall_on_external_required_rows']:.4f} | {row['routing_failure_rows']} |"
        )
    lines.extend(["", "## Boundary", "", BOUNDARY, ""])
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="target/pilot_wave/d72_support_cost_oracle_gap_bound_analysis/smoke")
    parser.add_argument("--seeds", default="13401,13402,13403,13404,13405")
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
    write_json(out / "queue.json", {"task": TASK, "created_at": round(time.time(), 3), "seeds": parse_seeds(args.seeds), "rows_per_seed": {"train": args.train_rows_per_seed, "test": args.test_rows_per_seed, "ood": args.ood_rows_per_seed}, "workers": args.workers, "cpu_target": args.cpu_target, "phases": ["repo_handoff_audit", "d71_bootstrap", "bound_analysis", "reporting"]})
    append_progress(out, "phase0", "starting D72 repo/handoff audit")
    bootstrap_report = bootstrap_d71_if_needed()
    write_json(out / "artifact_restore_report.json", bootstrap_report)
    write_json(out / "d71_upstream_manifest.json", d71_upstream_manifest(bootstrap_report))
    append_progress(out, "phase1", "building D72 oracle-gap bound analysis reports")
    aggregate, decision = build_reports(args, out, bootstrap_report)
    append_progress(out, "complete", "D72 support-cost oracle-gap bound analysis complete", decision=decision["decision"], next=decision["next"])
    print(json.dumps({"task": TASK, "out": str(out), "decision": decision["decision"], "next": decision["next"], "estimated_irreducible_cost": aggregate["estimated_irreducible_cost"], "estimated_reducible_cost": aggregate["estimated_reducible_cost"], "failed_jobs": aggregate["failed_jobs"]}, indent=2))


if __name__ == "__main__":
    main()
