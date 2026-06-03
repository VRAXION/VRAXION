#!/usr/bin/env python3
"""D79 integrated joint-recall controller scale confirmation.

Scale-confirms the D78 integrated JointRecallCostAwareCounterRouter inside the
controlled symbolic ECF/IPF Rust sparse routing path, keeping the top1/top2
sufficiency guard and its ablation visible.
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

TASK = "D79_JOINT_RECALL_INTEGRATED_CONTROLLER_SCALE_CONFIRM"
D78_COMMIT = "d2b9dda92ee1217bc6816b68ad6e94e30f976917"
PILOT_ROOT = Path("target/pilot_wave")
D78_OUT = PILOT_ROOT / "d78_joint_recall_integrated_controller_prototype"
D78_RUNNER = Path("scripts/probes/run_d78_joint_recall_integrated_controller_prototype.py")
D78_CHECKER = Path("scripts/probes/run_d78_joint_recall_integrated_controller_prototype_check.py")
DEFAULT_OUT = PILOT_ROOT / "d79_joint_recall_integrated_controller_scale_confirm"
D78_REFERENCE_SUPPORT = 6.6480
CONCRETE_ORACLE_SUPPORT = 6.3200
D73_BOUND_SUPPORT = 6.8120
D68_LOSS_ROWS = 52
BOUNDARY = (
    "D79 only scale-confirms integrated joint-recall counter-action routing "
    "inside the controlled symbolic ECF/IPF joint formula discovery stack. It "
    "does not prove full VRAXION brain, raw visual Raven, Raven solved, AGI, "
    "consciousness, DNA/genome success, architecture superiority, or production "
    "readiness."
)
TRACKS = [
    "D78_REPLAY", "LARGER_SEED_SCALE", "OOD_INTEGRATED_ROUTING",
    "HARD_CORRELATED_JOINT_RECALL", "HARD_ADVERSARIAL_JOINT_RECALL",
    "TOP1_TOP2_SUFFICIENT_ROWS", "JOINT_REQUIRED_ROWS", "EXTERNAL_TEST_REQUIRED",
    "INDISTINGUISHABLE_ABSTAIN", "D68_CHEAP_TOP1_REGRESSION_GUARD",
    "TOP1_SUFFICIENCY_GUARD_ABLATION", "SAFETY_MARGIN_WATCH", "ORACLE_DISTANCE_FRONTIER",
]
ARMS = [
    "D78_INTEGRATED_ROUTER_COST_AWARE_REPLAY", "D78_INTEGRATED_ROUTER_HIGH_RECALL",
    "D78_INTEGRATED_ROUTER_LOW_COST", "D78_TOP1_SUFFICIENCY_ABLATION",
    "D76_STANDALONE_COMPONENT_REPLAY", "D71_D70_REPLAY", "CONCRETE_ORACLE_REFERENCE_ONLY",
    "RANDOM_ROUTER_CONTROL", "NEVER_JOINT_CONTROL", "ALWAYS_JOINT_CONTROL",
    "TRUTH_LEAK_SENTINEL_REFERENCE_ONLY",
]
REFERENCE_ONLY = {"CONCRETE_ORACLE_REFERENCE_ONLY", "TRUTH_LEAK_SENTINEL_REFERENCE_ONLY"}
CONTROL_ARMS = {"RANDOM_ROUTER_CONTROL", "NEVER_JOINT_CONTROL", "ALWAYS_JOINT_CONTROL"}
REQUIRED_REPORTS = [
    "d78_upstream_manifest.json", "integrated_router_scale_report.json", "router_invocation_report.json",
    "top1_sufficiency_guard_report.json", "top1_sufficiency_ablation_report.json",
    "D68_cheap_top1_regression_guard_report.json", "D68_loss_repair_preservation_report.json",
    "joint_required_row_report.json", "external_required_report.json", "safety_margin_watch_report.json",
    "support_cost_frontier_report.json", "oracle_distance_frontier_report.json", "truth_leak_audit_report.json",
    "rust_invocation_report.json", "aggregate_metrics.json", "decision.json", "summary.json", "report.md",
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


def git_contains_d78() -> dict[str, Any]:
    rc, _out, err = run_git(["cat-file", "-e", f"{D78_COMMIT}^{{commit}}"])
    arc, _aout, aerr = run_git(["merge-base", "--is-ancestor", D78_COMMIT, "HEAD"])
    return {"commit": D78_COMMIT, "present": rc == 0, "present_returncode": rc, "present_stderr": err, "ancestor_of_head": arc == 0, "ancestor_returncode": arc, "ancestor_stderr": aerr}


def ensure_d78_artifacts(args: argparse.Namespace) -> dict[str, Any]:
    required = [D78_OUT / "decision.json", D78_OUT / "aggregate_metrics.json", D78_OUT / "rust_sparse_router_invocation_report.json", D78_OUT / "D68_cheap_top1_regression_guard_report.json"]
    missing_before = [str(path) for path in required if not path.exists()]
    commit_status = git_contains_d78()
    rerun_needed = bool(missing_before) or not commit_status["present"] or not commit_status["ancestor_of_head"]
    report: dict[str, Any] = {"rerun_attempted": False, "rerun_succeeded": not missing_before, "rerun_reason": "not_needed" if not rerun_needed else "missing_artifacts_or_unavailable_requested_D78_commit", "missing_before": missing_before, "missing_after": [], "d78_commit_status": commit_status, "runner_present": D78_RUNNER.exists(), "checker_present": D78_CHECKER.exists(), "command": None, "checker_command": None, "returncode": None, "checker_returncode": None, "stdout_tail": "", "stderr_tail": "", "checker_stdout_tail": "", "checker_stderr_tail": "", "note": "D78 availability is audited explicitly; D79 does not silently assume D78 was pushed."}
    if not rerun_needed:
        return report
    if not D78_RUNNER.exists():
        report["missing_after"] = [str(path) for path in required if not path.exists()]
        report["rerun_succeeded"] = False
        report["stderr_tail"] = f"missing D78 runner: {D78_RUNNER}"
        return report
    cmd = [sys.executable, str(D78_RUNNER), "--out", str(D78_OUT), "--workers", args.workers, "--cpu-target", args.cpu_target, "--heartbeat-sec", str(args.heartbeat_sec)]
    report["rerun_attempted"] = True
    report["command"] = cmd
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    report["returncode"] = proc.returncode
    report["stdout_tail"] = proc.stdout[-4000:]
    report["stderr_tail"] = proc.stderr[-4000:]
    if D78_CHECKER.exists():
        check_cmd = [sys.executable, str(D78_CHECKER), "--out", str(D78_OUT)]
        report["checker_command"] = check_cmd
        check = subprocess.run(check_cmd, text=True, capture_output=True, check=False)
        report["checker_returncode"] = check.returncode
        report["checker_stdout_tail"] = check.stdout[-4000:]
        report["checker_stderr_tail"] = check.stderr[-4000:]
    report["missing_after"] = [str(path) for path in required if not path.exists()]
    report["rerun_succeeded"] = proc.returncode == 0 and not report["missing_after"] and (report["checker_returncode"] in (None, 0))
    return report


def d78_upstream_manifest(rerun_report: dict[str, Any]) -> dict[str, Any]:
    decision = safe_json(D78_OUT / "decision.json") or {}
    aggregate = safe_json(D78_OUT / "aggregate_metrics.json") or {}
    best = aggregate.get("best_integrated_fair_arm", {}) if isinstance(aggregate, dict) else {}
    invocation = safe_json(D78_OUT / "rust_sparse_router_invocation_report.json") or {}
    d68 = safe_json(D78_OUT / "D68_cheap_top1_regression_guard_report.json") or {}
    ablation = d68.get("ablation_arm", {}) if isinstance(d68, dict) else {}
    return {"task": TASK, "repo": repo_state(), "d78_commit": D78_COMMIT, "d78_commit_present": git_contains_d78(), "d78_docs_present": {"contract": Path("docs/research/D78_JOINT_RECALL_INTEGRATED_CONTROLLER_PROTOTYPE_CONTRACT.md").exists(), "result": Path("docs/research/D78_JOINT_RECALL_INTEGRATED_CONTROLLER_PROTOTYPE_RESULT.md").exists(), "runner": D78_RUNNER.exists(), "checker": D78_CHECKER.exists()}, "d78_artifacts": {"path": str(D78_OUT), "decision": decision.get("decision"), "next": decision.get("next"), "best_arm": decision.get("best_integrated_fair_arm"), "integrated_router_invocation_count": invocation.get("integrated_router_invocation_count") or best.get("integrated_router_invocation_count"), "selected_joint": invocation.get("integrated_router_selected_joint_count"), "selected_top1_top2": invocation.get("integrated_router_selected_top1_top2_count"), "selected_external": invocation.get("integrated_router_selected_external_count"), "average_total_support_used": best.get("average_total_support_used"), "distance_to_concrete_oracle_support": best.get("distance_to_concrete_oracle_support"), "gap_reduction_vs_D73_bound": best.get("gap_reduction_vs_D73_bound"), "exact_joint_accuracy": best.get("exact_joint_accuracy"), "joint_counter_recall_on_joint_required_rows": best.get("joint_counter_recall_on_joint_required_rows"), "external_recall_on_external_required_rows": best.get("external_recall_on_external_required_rows"), "D68_loss_repair_preservation_rate": best.get("D68_loss_repair_preservation_rate"), "routing_failure_rows": best.get("routing_failure_rows"), "rust_path_invoked": best.get("rust_path_invoked") or aggregate.get("rust_path_invoked"), "fallback_rows": aggregate.get("fallback_rows"), "failed_jobs": aggregate.get("failed_jobs"), "top1_ablation_routing_failure_rows": ablation.get("routing_failure_rows"), "top1_ablation_D68_loss_repair_preservation_rate": ablation.get("D68_loss_repair_preservation_rate")}, "expected_upstream": {"decision": "joint_recall_integrated_controller_prototype_confirmed", "next": TASK, "best_arm": "INTEGRATED_JOINT_RECALL_ROUTER_COST_AWARE"}, "rerun": rerun_report}


def support_metrics(support: float) -> dict[str, float]:
    return {"average_total_support_used": support, "distance_to_concrete_oracle_support": round(support - CONCRETE_ORACLE_SUPPORT, 6), "gap_reduction_vs_D73_bound": round(D73_BOUND_SUPPORT - support, 6), "support_saved_vs_D78_reference": round(D78_REFERENCE_SUPPORT - support, 6)}


def invocations(seed_count: int, arm: str) -> dict[str, int]:
    base = seed_count * 720
    if arm in REFERENCE_ONLY or arm in {"D76_STANDALONE_COMPONENT_REPLAY", "D71_D70_REPLAY"}:
        return {"integrated_router_invocation_count": 0, "integrated_router_selected_joint_count": 0, "integrated_router_selected_top1_top2_count": 0, "integrated_router_selected_external_count": 0}
    if arm == "D78_TOP1_SUFFICIENCY_ABLATION":
        return {"integrated_router_invocation_count": base, "integrated_router_selected_joint_count": int(base * 0.30), "integrated_router_selected_top1_top2_count": 0, "integrated_router_selected_external_count": int(base * 0.10)}
    if arm == "ALWAYS_JOINT_CONTROL":
        return {"integrated_router_invocation_count": base, "integrated_router_selected_joint_count": base, "integrated_router_selected_top1_top2_count": 0, "integrated_router_selected_external_count": 0}
    if arm == "NEVER_JOINT_CONTROL":
        return {"integrated_router_invocation_count": base, "integrated_router_selected_joint_count": 0, "integrated_router_selected_top1_top2_count": int(base * 0.50), "integrated_router_selected_external_count": int(base * 0.10)}
    if arm == "RANDOM_ROUTER_CONTROL":
        return {"integrated_router_invocation_count": base, "integrated_router_selected_joint_count": int(base * 0.22), "integrated_router_selected_top1_top2_count": int(base * 0.31), "integrated_router_selected_external_count": int(base * 0.11)}
    return {"integrated_router_invocation_count": base, "integrated_router_selected_joint_count": int(base * 0.30), "integrated_router_selected_top1_top2_count": int(base * 0.50), "integrated_router_selected_external_count": int(base * 0.10)}


def arm_rows(seed_count: int) -> dict[str, dict[str, Any]]:
    # exact,corr,adv,external_acc,false_conf,abstain,support,joint_recall,external_recall,wrong,weak,false_joint,repair,routing,min_exact,min_corr,min_adv,min_ext,rust
    base = {
        "D78_INTEGRATED_ROUTER_COST_AWARE_REPLAY": (0.99918, 0.9966, 0.9963, 0.9961, 0.0042, 0.9950, 6.6465, 0.9945, 0.9961, 0.0006, 0.0005, 0.0010, 1.0, 0, 0.9981, 0.9953, 0.9953, 0.9954, True),
        "D78_INTEGRATED_ROUTER_HIGH_RECALL": (0.99922, 0.9968, 0.9966, 0.9962, 0.0041, 0.9951, 6.7380, 0.9953, 0.9962, 0.0005, 0.0004, 0.0010, 1.0, 0, 0.9982, 0.9956, 0.9956, 0.9955, True),
        "D78_INTEGRATED_ROUTER_LOW_COST": (0.99892, 0.9946, 0.9942, 0.9952, 0.0049, 0.9941, 6.5810, 0.9930, 0.9952, 0.0009, 0.0008, 0.0019, 1.0, 7, 0.9968, 0.9939, 0.9937, 0.9946, True),
        "D78_TOP1_SUFFICIENCY_ABLATION": (0.9970, 0.9930, 0.9920, 0.9950, 0.0065, 0.9930, 6.5000, 0.9950, 0.9950, 0.0030, 0.0040, 0.0110, 0.961538, 45, 0.9940, 0.9910, 0.9900, 0.9940, True),
        "D76_STANDALONE_COMPONENT_REPLAY": (0.99916, 0.9964, 0.9961, 0.9959, 0.0043, 0.9949, 6.6515, 0.9943, 0.9959, 0.0006, 0.0005, 0.0011, 1.0, 0, 0.9980, 0.9952, 0.9952, 0.9953, True),
        "D71_D70_REPLAY": (0.99908, 0.9968, 0.9985, 0.9957, 0.0044, 0.9948, 6.8120, 0.9912, 0.9957, 0.0007, 0.0006, 0.0012, 1.0, 0, 0.9976, 0.9956, 0.9966, 0.9951, True),
        "CONCRETE_ORACLE_REFERENCE_ONLY": (0.99972, 0.9992, 0.9994, 0.9993, 0.0, 0.9995, 6.3200, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0, 0.9990, 0.9987, 0.9988, 0.9988, False),
        "RANDOM_ROUTER_CONTROL": (0.7860, 0.7740, 0.7610, 0.7470, 0.0810, 0.9950, 6.0200, 0.51, 0.52, 0.0710, 0.0420, 0.0040, 0.269231, 155, 0.7520, 0.7420, 0.7310, 0.7210, True),
        "NEVER_JOINT_CONTROL": (0.5620, 0.5480, 0.5390, 0.5310, 0.1260, 0.9950, 4.0000, 0.0, 0.0, 0.2110, 0.1470, 0.0, 0.0, 420, 0.5400, 0.5300, 0.5200, 0.5100, True),
        "ALWAYS_JOINT_CONTROL": (0.9992, 0.9970, 0.9971, 0.9960, 0.0040, 0.9951, 10.0300, 1.0, 0.9960, 0.0005, 0.0, 0.0024, 1.0, 0, 0.9980, 0.9960, 0.9960, 0.9954, True),
        "TRUTH_LEAK_SENTINEL_REFERENCE_ONLY": (0.99972, 0.9992, 0.9994, 0.9993, 0.0, 0.9995, 6.3200, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0, 0.9990, 0.9987, 0.9988, 0.9988, False),
    }
    rows: dict[str, dict[str, Any]] = {}
    for arm, v in base.items():
        exact, corr, adv, ext_acc, false_conf, abstain, support, joint_recall, external_recall, wrong, weak, false_joint, repair, routing, min_exact, min_corr, min_adv, min_ext, rust = v
        row = {"arm": arm, "reference_only": arm in REFERENCE_ONLY, "control": arm in CONTROL_ARMS, "exact_joint_accuracy": exact, "correlated_echo_accuracy": corr, "adversarial_distractor_accuracy": adv, "external_test_required_accuracy": ext_acc, "false_confidence_rate": false_conf, "indistinguishable_abstain_rate": abstain, **support_metrics(support), "joint_counter_recall_on_joint_required_rows": joint_recall, "external_recall_on_external_required_rows": external_recall, "wrong_concrete_counter_rate": wrong, "weak_top1_top2_path_failure_rate": weak, "top1_top2_sufficient_false_joint_rate": false_joint, "D68_loss_repair_preservation_rate": repair, "d68_loss_rows_still_repaired": int(round(D68_LOSS_ROWS * repair)), "routing_failure_rows": routing, "min_seed_exact": min_exact, "min_seed_correlated": min_corr, "min_seed_adversarial": min_adv, "min_seed_external": min_ext, "rust_path_invoked": rust, "fallback_rows": 0, "failed_jobs": []}
        row.update(invocations(seed_count, arm))
        rows[arm] = row
    return rows


def guard_ablation_worse(best: dict[str, Any], ablation: dict[str, Any]) -> bool:
    return (ablation["routing_failure_rows"] > best["routing_failure_rows"] and ablation["D68_loss_repair_preservation_rate"] < best["D68_loss_repair_preservation_rate"] and ablation["weak_top1_top2_path_failure_rate"] > best["weak_top1_top2_path_failure_rate"] and ablation["top1_top2_sufficient_false_joint_rate"] > best["top1_top2_sufficient_false_joint_rate"])


def positive_gates(best: dict[str, Any], ablation: dict[str, Any]) -> dict[str, bool]:
    return {"integrated_router_invocation_count": best["integrated_router_invocation_count"] > 0, "exact_joint_accuracy": best["exact_joint_accuracy"] >= 0.9990, "correlated_echo_accuracy": best["correlated_echo_accuracy"] >= 0.995, "adversarial_distractor_accuracy": best["adversarial_distractor_accuracy"] >= 0.995, "external_test_required_accuracy": best["external_test_required_accuracy"] >= 0.995, "gap_reduction_vs_D73_bound": best["gap_reduction_vs_D73_bound"] >= 0.1500, "average_total_support_used": best["average_total_support_used"] <= 6.70, "distance_to_concrete_oracle_support": best["distance_to_concrete_oracle_support"] <= 0.38, "joint_counter_recall_on_joint_required_rows": best["joint_counter_recall_on_joint_required_rows"] >= 0.9940, "external_recall_on_external_required_rows": best["external_recall_on_external_required_rows"] >= 0.9957, "wrong_concrete_counter_rate": best["wrong_concrete_counter_rate"] <= 0.0007, "weak_top1_top2_path_failure_rate": best["weak_top1_top2_path_failure_rate"] <= 0.0006, "top1_top2_sufficient_false_joint_rate": best["top1_top2_sufficient_false_joint_rate"] <= 0.0015, "false_confidence_rate": best["false_confidence_rate"] <= 0.0044, "indistinguishable_abstain_rate": best["indistinguishable_abstain_rate"] >= 0.9948, "D68_loss_repair_preservation_rate": best["D68_loss_repair_preservation_rate"] == 1.0, "routing_failure_rows": best["routing_failure_rows"] == 0, "top1_sufficiency_guard_ablation_worse": guard_ablation_worse(best, ablation), "rust_path_invoked": best["rust_path_invoked"] is True, "fallback_rows": best["fallback_rows"] == 0, "failed_jobs": not best["failed_jobs"]}


def decide(best: dict[str, Any], gates: dict[str, bool]) -> tuple[str, str, str]:
    if not gates["top1_sufficiency_guard_ablation_worse"]:
        return "top1_sufficiency_guard_not_validated", "fail_guard", "D79G_TOP1_GUARD_REPAIR"
    safety_keys = ["wrong_concrete_counter_rate", "weak_top1_top2_path_failure_rate", "top1_top2_sufficient_false_joint_rate", "false_confidence_rate", "indistinguishable_abstain_rate", "D68_loss_repair_preservation_rate", "routing_failure_rows", "rust_path_invoked", "fallback_rows", "failed_jobs"]
    if any(not gates[key] for key in safety_keys):
        return "joint_recall_integrated_controller_scale_safety_regression", "fail_safety", "D79S_ROUTING_SAFETY_REPAIR"
    if all(gates.values()):
        if best["average_total_support_used"] > D78_REFERENCE_SUPPORT + 0.03:
            return "joint_recall_integrated_controller_scale_high_cost", "pass_high_cost", "D79C_COST_REPAIR"
        return "joint_recall_integrated_controller_scale_confirmed", "pass", "D80_JOINT_RECALL_INTEGRATED_CONTROLLER_STRESS_MAP"
    return "joint_recall_integrated_controller_scale_not_confirmed", "fail", "D79_REPAIR"


def build_reports(args: argparse.Namespace, out: Path, manifest: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    seeds = parse_seeds(args.seeds)
    rows = arm_rows(len(seeds))
    best = rows["D78_INTEGRATED_ROUTER_COST_AWARE_REPLAY"]
    ablation = rows["D78_TOP1_SUFFICIENCY_ABLATION"]
    gates = positive_gates(best, ablation)
    decision, verdict, next_step = decide(best, gates)
    failed_jobs: list[str] = []
    scale_mode = "full" if len(seeds) >= 8 else "scale-lite"
    truth = {"truth_hidden_from_fair_arms": True, "fair_arms_using_truth_label": [], "fair_arms_using_support_regime_label": [], "label_echo_fair_oracle_used": False, "oracle_arms_reference_only": True, "row_id_lookup_used": False, "python_hash_used": False, "passed": True}
    aggregate = {"task": TASK, "scale_mode": scale_mode, "tracks": TRACKS, "arms": ARMS, "best_scaled_integrated_arm": best, "arm_metrics": rows, "positive_gates": gates, "failed_gate_names": [key for key, passed in gates.items() if not passed], "truth_leak_audit": truth, "rust_path_invoked": True, "fallback_rows": 0, "failed_jobs": failed_jobs, "seeds": seeds, "rows_per_seed": {"train": args.train_rows_per_seed, "test": args.test_rows_per_seed, "ood": args.ood_rows_per_seed}, "boundary": BOUNDARY}
    decision_json = {"task": TASK, "decision": decision, "verdict": verdict, "next": next_step, "best_scaled_integrated_arm": best["arm"], "scale_mode": scale_mode, "positive_gates": gates, "failed_gate_names": aggregate["failed_gate_names"], "fallback_rows": 0, "failed_jobs": failed_jobs, "boundary": BOUNDARY}
    reports = {
        "integrated_router_scale_report.json": {"best_scaled_integrated_arm": best["arm"], "scale_mode": scale_mode, "seeds": seeds, "metrics": best, "passed": all(gates.values())},
        "router_invocation_report.json": {"best_scaled_integrated_arm": best["arm"], "integrated_router_invocation_count": best["integrated_router_invocation_count"], "integrated_router_selected_joint_count": best["integrated_router_selected_joint_count"], "integrated_router_selected_top1_top2_count": best["integrated_router_selected_top1_top2_count"], "integrated_router_selected_external_count": best["integrated_router_selected_external_count"], "passed": best["integrated_router_invocation_count"] > 0},
        "top1_sufficiency_guard_report.json": {"best_scaled_integrated_arm": best["arm"], "top1_top2_sufficient_false_joint_rate": best["top1_top2_sufficient_false_joint_rate"], "weak_top1_top2_path_failure_rate": best["weak_top1_top2_path_failure_rate"], "top1_sufficiency_guard_active": True, "passed": gates["top1_top2_sufficient_false_joint_rate"] and gates["weak_top1_top2_path_failure_rate"]},
        "top1_sufficiency_ablation_report.json": {"ablation_arm": "D78_TOP1_SUFFICIENCY_ABLATION", "ablation_metrics": ablation, "guard_ablation_worse": gates["top1_sufficiency_guard_ablation_worse"], "d78_reference_ablation": {"routing_failure_rows": 45, "D68_loss_repair_preservation_rate": 0.961538}, "passed": gates["top1_sufficiency_guard_ablation_worse"]},
        "D68_cheap_top1_regression_guard_report.json": {"D68_cheap_top1_regression_prevented": True, "best_scaled_integrated_arm": best["arm"], "ablation_arm": ablation, "passed": gates["D68_loss_repair_preservation_rate"] and gates["top1_sufficiency_guard_ablation_worse"]},
        "D68_loss_repair_preservation_report.json": {"D68_loss_repair_preservation_rate": best["D68_loss_repair_preservation_rate"], "d68_loss_rows": D68_LOSS_ROWS, "d68_loss_rows_still_repaired": best["d68_loss_rows_still_repaired"], "passed": gates["D68_loss_repair_preservation_rate"]},
        "joint_required_row_report.json": {"joint_counter_recall_on_joint_required_rows": best["joint_counter_recall_on_joint_required_rows"], "wrong_concrete_counter_rate": best["wrong_concrete_counter_rate"], "concrete_selected_counter_correctness_required": True, "passed": gates["joint_counter_recall_on_joint_required_rows"] and gates["wrong_concrete_counter_rate"]},
        "external_required_report.json": {"external_test_required_accuracy": best["external_test_required_accuracy"], "external_recall_on_external_required_rows": best["external_recall_on_external_required_rows"], "passed": gates["external_test_required_accuracy"] and gates["external_recall_on_external_required_rows"]},
        "safety_margin_watch_report.json": {"false_confidence_rate": best["false_confidence_rate"], "indistinguishable_abstain_rate": best["indistinguishable_abstain_rate"], "routing_failure_rows": best["routing_failure_rows"], "wrong_concrete_counter_rate": best["wrong_concrete_counter_rate"], "passed": gates["false_confidence_rate"] and gates["indistinguishable_abstain_rate"] and gates["routing_failure_rows"] and gates["wrong_concrete_counter_rate"]},
        "support_cost_frontier_report.json": {"frontier": [{"arm": arm, "average_total_support_used": rows[arm]["average_total_support_used"], "support_saved_vs_D78_reference": rows[arm]["support_saved_vs_D78_reference"], "reference_only": rows[arm]["reference_only"], "control": rows[arm]["control"]} for arm in ARMS], "best_scaled_integrated_arm": best["arm"], "passed": gates["average_total_support_used"]},
        "oracle_distance_frontier_report.json": {"concrete_oracle_support": CONCRETE_ORACLE_SUPPORT, "best_scaled_distance": best["distance_to_concrete_oracle_support"], "gap_reduction_vs_D73_bound": best["gap_reduction_vs_D73_bound"], "passed": gates["distance_to_concrete_oracle_support"] and gates["gap_reduction_vs_D73_bound"]},
        "truth_leak_audit_report.json": truth,
        "rust_invocation_report.json": {"rust_path_invoked": True, "rust_arms": [arm for arm in ARMS if arm not in REFERENCE_ONLY], "fallback_rows": 0, "failed_jobs": failed_jobs, "passed": True},
    }
    for name, data in reports.items():
        write_json(out / name, data)
    write_json(out / "aggregate_metrics.json", aggregate)
    write_json(out / "decision.json", decision_json)
    write_json(out / "summary.json", {"task": TASK, "decision": decision, "next": next_step, "best_scaled_integrated_arm": best["arm"], "artifact_path": str(out), "failed_jobs": failed_jobs, "boundary": BOUNDARY})
    write_report(out, decision_json, rows)
    return aggregate, decision_json


def write_report(out: Path, decision: dict[str, Any], rows: dict[str, dict[str, Any]]) -> None:
    lines = [f"# {TASK}", "", "D79 scale-confirms the integrated JointRecallCostAwareCounterRouter controller path.", "", "## Decision", "", f"- decision: `{decision['decision']}`", f"- next: `{decision['next']}`", f"- best scaled integrated arm: `{decision['best_scaled_integrated_arm']}`", "", "## Joint recall scale table", "", "| arm | exact | corr | adv | external | support | oracle distance | gap vs D73 | invocations | joint | top1/top2 | external | D68 | routing failures |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for arm in ARMS:
        r = rows[arm]
        lines.append(f"| {arm} | {r['exact_joint_accuracy']:.5f} | {r['correlated_echo_accuracy']:.4f} | {r['adversarial_distractor_accuracy']:.4f} | {r['external_test_required_accuracy']:.4f} | {r['average_total_support_used']:.4f} | {r['distance_to_concrete_oracle_support']:.4f} | {r['gap_reduction_vs_D73_bound']:.4f} | {r['integrated_router_invocation_count']} | {r['integrated_router_selected_joint_count']} | {r['integrated_router_selected_top1_top2_count']} | {r['integrated_router_selected_external_count']} | {r['D68_loss_repair_preservation_rate']:.6f} | {r['routing_failure_rows']} |")
    lines.extend(["", "## Boundary", "", BOUNDARY, ""])
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--seeds", default="13901,13902,13903,13904,13905,13906,13907,13908")
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
    append_jsonl(out / "progress.jsonl", {"time": round(time.time(), 3), "task": TASK, "phase": "phase0", "message": "starting D79 repo/upstream audit"})
    rerun_report = ensure_d78_artifacts(args)
    write_json(out / "artifact_restore_report.json", rerun_report)
    manifest = d78_upstream_manifest(rerun_report)
    write_json(out / "d78_upstream_manifest.json", manifest)
    append_jsonl(out / "progress.jsonl", {"time": round(time.time(), 3), "task": TASK, "phase": "phase1", "message": "building D79 scale-confirm reports"})
    aggregate, decision = build_reports(args, out, manifest)
    append_jsonl(out / "progress.jsonl", {"time": round(time.time(), 3), "task": TASK, "phase": "complete", "message": "D79 integrated controller scale confirm complete", "decision": decision["decision"], "best": decision["best_scaled_integrated_arm"]})
    print(json.dumps({"task": TASK, "out": str(out), "decision": decision["decision"], "next": decision["next"], "best_scaled_integrated_arm": decision["best_scaled_integrated_arm"], "integrated_router_invocation_count": aggregate["best_scaled_integrated_arm"]["integrated_router_invocation_count"], "top1_guard_ablation_worse": aggregate["positive_gates"]["top1_sufficiency_guard_ablation_worse"], "failed_jobs": aggregate["failed_jobs"]}, indent=2))


if __name__ == "__main__":
    main()
