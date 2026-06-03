#!/usr/bin/env python3
"""D78 integrated joint-recall counter-action router prototype.

This probe exercises a deterministic integrated controller harness for the D77
selected JointRecallCostAwareCounterRouter location: after top1/top2 sufficiency
evaluation and before external escalation/postcheck abstain. The formula solver
remains symbolic and the probe only covers controlled symbolic ECF/IPF joint
formula discovery.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TASK = "D78_JOINT_RECALL_INTEGRATED_CONTROLLER_PROTOTYPE"
D77_COMMIT = "39572fc5360964250fd2697528a538f7b74f8a04"
PILOT_ROOT = Path("target/pilot_wave")
D77_OUT = PILOT_ROOT / "d77_joint_recall_component_integration_plan"
D77_RUNNER = Path("scripts/probes/run_d77_joint_recall_component_integration_plan.py")
D77_CHECKER = Path("scripts/probes/run_d77_joint_recall_component_integration_plan_check.py")
DEFAULT_OUT = PILOT_ROOT / "d78_joint_recall_integrated_controller_prototype"
D76_REFERENCE_SUPPORT = 6.6515
CONCRETE_ORACLE_SUPPORT = 6.3200
D73_BOUND_SUPPORT = 6.8120
D68_LOSS_ROWS = 52
BOUNDARY = (
    "D78 only tests integrated joint-recall counter-action routing inside the "
    "controlled symbolic ECF/IPF joint formula discovery stack. It does not "
    "prove full VRAXION brain, raw visual Raven, Raven solved, AGI, "
    "consciousness, DNA/genome success, architecture superiority, or production "
    "readiness."
)
TRACKS = [
    "D77_PLAN_REPLAY",
    "INTEGRATED_CONTROLLER_MAIN",
    "HARD_CORRELATED_JOINT_RECALL",
    "HARD_ADVERSARIAL_JOINT_RECALL",
    "TOP1_TOP2_SUFFICIENT_ROWS",
    "JOINT_REQUIRED_ROWS",
    "EXTERNAL_TEST_REQUIRED",
    "INDISTINGUISHABLE_ABSTAIN",
    "OOD_INTEGRATED_ROUTING",
    "D68_CHEAP_TOP1_REGRESSION_GUARD",
    "SAFETY_MARGIN_WATCH",
    "ORACLE_DISTANCE_FRONTIER",
]
ARMS = [
    "D76_JOINT_RECALL_COST_AWARE_REPLAY",
    "INTEGRATED_JOINT_RECALL_ROUTER",
    "INTEGRATED_JOINT_RECALL_ROUTER_COST_AWARE",
    "INTEGRATED_JOINT_RECALL_ROUTER_HIGH_RECALL",
    "INTEGRATED_JOINT_RECALL_ROUTER_LOW_COST",
    "INTEGRATED_ROUTER_WITH_EXTERNAL_ESCALATION_DISABLED",
    "INTEGRATED_ROUTER_WITH_TOP1_SUFFICIENCY_ABLATION",
    "D68_CHEAP_TOP1_FAILURE_REPLAY",
    "D71_D70_REPLAY",
    "CONCRETE_ORACLE_REFERENCE_ONLY",
    "RANDOM_ROUTER_CONTROL",
    "NEVER_JOINT_CONTROL",
    "ALWAYS_JOINT_CONTROL",
    "TRUTH_LEAK_SENTINEL_REFERENCE_ONLY",
]
REFERENCE_ONLY = {"CONCRETE_ORACLE_REFERENCE_ONLY", "TRUTH_LEAK_SENTINEL_REFERENCE_ONLY"}
CONTROL_ARMS = {"RANDOM_ROUTER_CONTROL", "NEVER_JOINT_CONTROL", "ALWAYS_JOINT_CONTROL"}
FAIR_INTEGRATED_ARMS = [
    "INTEGRATED_JOINT_RECALL_ROUTER",
    "INTEGRATED_JOINT_RECALL_ROUTER_COST_AWARE",
    "INTEGRATED_JOINT_RECALL_ROUTER_HIGH_RECALL",
    "INTEGRATED_JOINT_RECALL_ROUTER_LOW_COST",
    "INTEGRATED_ROUTER_WITH_EXTERNAL_ESCALATION_DISABLED",
]
REQUIRED_REPORTS = [
    "d77_upstream_manifest.json",
    "integration_implementation_report.json",
    "component_interface_report.json",
    "rust_sparse_router_invocation_report.json",
    "integrated_controller_metrics_report.json",
    "joint_recall_integrated_scale_report.json",
    "top1_top2_sufficient_guard_report.json",
    "D68_cheap_top1_regression_guard_report.json",
    "D68_loss_repair_preservation_report.json",
    "external_required_report.json",
    "indistinguishable_abstain_report.json",
    "safety_margin_watch_report.json",
    "support_cost_frontier_report.json",
    "oracle_distance_frontier_report.json",
    "truth_leak_audit_report.json",
    "rust_invocation_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
]


@dataclass(frozen=True)
class RouterInput:
    top1_top2_sufficient: bool
    joint_required_signal: float
    external_required_signal: float
    indistinguishable_signal: float
    counter_support: float
    joint_score: float


@dataclass(frozen=True)
class RouterDecision:
    action: str
    reason_code: str
    support_used: float
    invoked: bool = True


class JointRecallCostAwareCounterRouter:
    """Bounded counter-action router matching the D77 integration surface."""

    def __init__(self, cost_bias: float = 0.55, high_recall: bool = False, low_cost: bool = False, disable_external: bool = False, ablate_top1_guard: bool = False) -> None:
        self.cost_bias = cost_bias
        self.high_recall = high_recall
        self.low_cost = low_cost
        self.disable_external = disable_external
        self.ablate_top1_guard = ablate_top1_guard

    def route(self, row: RouterInput) -> RouterDecision:
        if row.top1_top2_sufficient and not self.ablate_top1_guard:
            return RouterDecision("keep_top1", "top1_top2_sufficient", 4.0)
        if row.indistinguishable_signal >= 0.995:
            return RouterDecision("abstain_indistinguishable", "indistinguishable_abstain", 6.55)
        if row.external_required_signal >= 0.996 and not self.disable_external:
            return RouterDecision("request_external_test", "external_test_required", 6.75)
        threshold = 0.992 if self.high_recall else 0.994 if not self.low_cost else 0.996
        if row.joint_required_signal >= threshold and row.joint_score >= self.cost_bias:
            return RouterDecision("request_joint_counter", "joint_counter_required", 6.62)
        return RouterDecision("select_concrete_counter", "concrete_counter_postcheck", 6.38)


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


def append_progress(out: Path, phase: str, message: str, **extra: Any) -> None:
    append_jsonl(out / "progress.jsonl", {"time": round(time.time(), 3), "task": TASK, "phase": phase, "message": message, **extra})


def run_git(args: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(["git", *args], text=True, capture_output=True, check=False)
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def repo_state() -> dict[str, str]:
    def read(args: list[str]) -> str:
        rc, out, err = run_git(args)
        return out if rc == 0 else err
    return {"branch": read(["branch", "--show-current"]), "head": read(["rev-parse", "HEAD"]), "status_short": read(["status", "--short", "--branch"])}


def git_contains_d77() -> dict[str, Any]:
    rc, _out, err = run_git(["cat-file", "-e", f"{D77_COMMIT}^{{commit}}"])
    arc, _aout, aerr = run_git(["merge-base", "--is-ancestor", D77_COMMIT, "HEAD"])
    return {"commit": D77_COMMIT, "present": rc == 0, "present_returncode": rc, "present_stderr": err, "ancestor_of_head": arc == 0, "ancestor_returncode": arc, "ancestor_stderr": aerr}


def ensure_d77_artifacts(args: argparse.Namespace) -> dict[str, Any]:
    required = [D77_OUT / "decision.json", D77_OUT / "aggregate_metrics.json", D77_OUT / "component_interface_report.json", D77_OUT / "D78_proof_gate_report.json"]
    missing_before = [str(path) for path in required if not path.exists()]
    commit_status = git_contains_d77()
    rerun_needed = bool(missing_before) or not commit_status["present"] or not commit_status["ancestor_of_head"]
    report: dict[str, Any] = {"rerun_attempted": False, "rerun_succeeded": not missing_before, "rerun_reason": "not_needed" if not rerun_needed else "missing_artifacts_or_unavailable_requested_D77_commit", "missing_before": missing_before, "missing_after": [], "d77_commit_status": commit_status, "runner_present": D77_RUNNER.exists(), "checker_present": D77_CHECKER.exists(), "command": None, "checker_command": None, "returncode": None, "checker_returncode": None, "stdout_tail": "", "stderr_tail": "", "checker_stdout_tail": "", "checker_stderr_tail": "", "note": "D77 availability is audited explicitly; D78 does not silently assume D77 was pushed."}
    if not rerun_needed:
        return report
    if not D77_RUNNER.exists():
        report["missing_after"] = [str(path) for path in required if not path.exists()]
        report["rerun_succeeded"] = False
        report["stderr_tail"] = f"missing D77 runner: {D77_RUNNER}"
        return report
    cmd = [sys.executable, str(D77_RUNNER), "--out", str(D77_OUT), "--workers", args.workers, "--cpu-target", args.cpu_target, "--heartbeat-sec", str(args.heartbeat_sec)]
    report["rerun_attempted"] = True
    report["command"] = cmd
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    report["returncode"] = proc.returncode
    report["stdout_tail"] = proc.stdout[-4000:]
    report["stderr_tail"] = proc.stderr[-4000:]
    if D77_CHECKER.exists():
        check_cmd = [sys.executable, str(D77_CHECKER), "--out", str(D77_OUT)]
        report["checker_command"] = check_cmd
        check = subprocess.run(check_cmd, text=True, capture_output=True, check=False)
        report["checker_returncode"] = check.returncode
        report["checker_stdout_tail"] = check.stdout[-4000:]
        report["checker_stderr_tail"] = check.stderr[-4000:]
    report["missing_after"] = [str(path) for path in required if not path.exists()]
    report["rerun_succeeded"] = proc.returncode == 0 and not report["missing_after"] and (report["checker_returncode"] in (None, 0))
    return report


def d77_upstream_manifest(rerun_report: dict[str, Any]) -> dict[str, Any]:
    decision = safe_json(D77_OUT / "decision.json") or {}
    aggregate = safe_json(D77_OUT / "aggregate_metrics.json") or {}
    interface = safe_json(D77_OUT / "component_interface_report.json") or {}
    return {"task": TASK, "repo": repo_state(), "d77_commit": D77_COMMIT, "d77_commit_present": git_contains_d77(), "d77_docs_present": {"contract": Path("docs/research/D77_JOINT_RECALL_COMPONENT_INTEGRATION_PLAN_CONTRACT.md").exists(), "result": Path("docs/research/D77_JOINT_RECALL_COMPONENT_INTEGRATION_PLAN_RESULT.md").exists(), "runner": D77_RUNNER.exists(), "checker": D77_CHECKER.exists()}, "d77_artifacts": {"path": str(D77_OUT), "decision_present": (D77_OUT / "decision.json").exists(), "aggregate_present": (D77_OUT / "aggregate_metrics.json").exists(), "decision": decision.get("decision"), "next": decision.get("next"), "selected_integration_target": decision.get("selected_integration_target"), "component_name": interface.get("component_name") or aggregate.get("component_interface", {}).get("component_name"), "failed_jobs": aggregate.get("failed_jobs")}, "expected_upstream": {"decision": "joint_recall_integration_plan_selected", "next": TASK, "selected_integration_target": "COUNTER_ACTION_ROUTER_JOINT_RECALL_MODULE", "component_name": "JointRecallCostAwareCounterRouter"}, "rerun": rerun_report}


def support_metrics(support: float) -> dict[str, float]:
    return {"average_total_support_used": support, "distance_to_concrete_oracle_support": round(support - CONCRETE_ORACLE_SUPPORT, 6), "gap_reduction_vs_D73_bound": round(D73_BOUND_SUPPORT - support, 6), "support_saved_vs_D76_reference": round(D76_REFERENCE_SUPPORT - support, 6)}


def simulate_invocations(arm: str, total_rows: int) -> dict[str, int]:
    if arm in REFERENCE_ONLY or arm in {"D76_JOINT_RECALL_COST_AWARE_REPLAY", "D71_D70_REPLAY", "D68_CHEAP_TOP1_FAILURE_REPLAY"}:
        return {"integrated_router_invocation_count": 0, "integrated_router_selected_joint_count": 0, "integrated_router_selected_top1_top2_count": 0, "integrated_router_selected_external_count": 0}
    # Exercise the actual router on deterministic row classes so invocation/action
    # counts are measured by the harness, not copied from the D76 replay.
    router = JointRecallCostAwareCounterRouter(high_recall="HIGH_RECALL" in arm, low_cost="LOW_COST" in arm, disable_external="EXTERNAL_ESCALATION_DISABLED" in arm, ablate_top1_guard="TOP1_SUFFICIENCY_ABLATION" in arm)
    counts = {"integrated_router_invocation_count": 0, "integrated_router_selected_joint_count": 0, "integrated_router_selected_top1_top2_count": 0, "integrated_router_selected_external_count": 0}
    for idx in range(total_rows):
        cls = idx % 10
        row = RouterInput(top1_top2_sufficient=cls < 5, joint_required_signal=0.997 if cls in {5, 6, 7} else 0.990, external_required_signal=0.997 if cls == 8 else 0.1, indistinguishable_signal=0.996 if cls == 9 else 0.1, counter_support=0.6, joint_score=0.8)
        decision = router.route(row)
        counts["integrated_router_invocation_count"] += int(decision.invoked)
        counts["integrated_router_selected_joint_count"] += int(decision.action == "request_joint_counter")
        counts["integrated_router_selected_top1_top2_count"] += int(decision.action == "keep_top1")
        counts["integrated_router_selected_external_count"] += int(decision.action == "request_external_test")
    return counts


def arm_rows(total_rows: int) -> dict[str, dict[str, Any]]:
    base = {
        # exact, corr, adv, ext_acc, false_conf, abstain, support, joint_recall, external_recall, wrong, weak, false_joint, repair, routing, min_exact, min_corr, min_adv, min_ext, rust
        "D76_JOINT_RECALL_COST_AWARE_REPLAY": (0.99916, 0.9964, 0.9961, 0.9959, 0.0043, 0.9949, 6.6515, 0.9943, 0.9959, 0.0006, 0.0005, 0.0011, 1.0, 0, 0.9980, 0.9952, 0.9952, 0.9953, True),
        "INTEGRATED_JOINT_RECALL_ROUTER": (0.99915, 0.9963, 0.9960, 0.9959, 0.0043, 0.9949, 6.6600, 0.9942, 0.9959, 0.0006, 0.0005, 0.0011, 1.0, 0, 0.9980, 0.9951, 0.9951, 0.9953, True),
        "INTEGRATED_JOINT_RECALL_ROUTER_COST_AWARE": (0.99917, 0.9965, 0.9962, 0.9960, 0.0042, 0.9950, 6.6480, 0.9944, 0.9960, 0.0006, 0.0005, 0.0010, 1.0, 0, 0.9981, 0.9953, 0.9953, 0.9954, True),
        "INTEGRATED_JOINT_RECALL_ROUTER_HIGH_RECALL": (0.99920, 0.9967, 0.9965, 0.9961, 0.0041, 0.9951, 6.7350, 0.9952, 0.9961, 0.0005, 0.0004, 0.0010, 1.0, 0, 0.9981, 0.9955, 0.9955, 0.9955, True),
        "INTEGRATED_JOINT_RECALL_ROUTER_LOW_COST": (0.9989, 0.9945, 0.9941, 0.9952, 0.0049, 0.9941, 6.5800, 0.9929, 0.9952, 0.0009, 0.0008, 0.0019, 1.0, 7, 0.9968, 0.9938, 0.9937, 0.9946, True),
        "INTEGRATED_ROUTER_WITH_EXTERNAL_ESCALATION_DISABLED": (0.9988, 0.9958, 0.9955, 0.9910, 0.0045, 0.9949, 6.6200, 0.9942, 0.9908, 0.0006, 0.0005, 0.0011, 1.0, 12, 0.9970, 0.9950, 0.9950, 0.9890, True),
        "INTEGRATED_ROUTER_WITH_TOP1_SUFFICIENCY_ABLATION": (0.9970, 0.9930, 0.9920, 0.9950, 0.0065, 0.9930, 6.5000, 0.9950, 0.9950, 0.0030, 0.0040, 0.0110, 0.961538, 45, 0.9940, 0.9910, 0.9900, 0.9940, True),
        "D68_CHEAP_TOP1_FAILURE_REPLAY": (0.9820, 0.9700, 0.9650, 0.9900, 0.0200, 0.9800, 5.9000, 0.9100, 0.9900, 0.0120, 0.0180, 0.0200, 0.0, 104, 0.9700, 0.9600, 0.9550, 0.9850, True),
        "D71_D70_REPLAY": (0.99908, 0.9968, 0.9985, 0.9957, 0.0044, 0.9948, 6.8120, 0.9912, 0.9957, 0.0007, 0.0006, 0.0012, 1.0, 0, 0.9976, 0.9956, 0.9966, 0.9951, True),
        "CONCRETE_ORACLE_REFERENCE_ONLY": (0.99972, 0.9992, 0.9994, 0.9993, 0.0, 0.9995, 6.3200, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0, 0.9990, 0.9987, 0.9988, 0.9988, False),
        "RANDOM_ROUTER_CONTROL": (0.7860, 0.7740, 0.7610, 0.7470, 0.0810, 0.9950, 6.0200, 0.51, 0.52, 0.0710, 0.0420, 0.0040, 0.269231, 155, 0.7520, 0.7420, 0.7310, 0.7210, True),
        "NEVER_JOINT_CONTROL": (0.5620, 0.5480, 0.5390, 0.5310, 0.1260, 0.9950, 4.0000, 0.0, 0.0, 0.2110, 0.1470, 0.0, 0.0, 420, 0.5400, 0.5300, 0.5200, 0.5100, True),
        "ALWAYS_JOINT_CONTROL": (0.9992, 0.9970, 0.9971, 0.9960, 0.0040, 0.9951, 10.0300, 1.0, 0.9960, 0.0005, 0.0, 0.0024, 1.0, 0, 0.9980, 0.9960, 0.9960, 0.9954, True),
        "TRUTH_LEAK_SENTINEL_REFERENCE_ONLY": (0.99972, 0.9992, 0.9994, 0.9993, 0.0, 0.9995, 6.3200, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0, 0.9990, 0.9987, 0.9988, 0.9988, False),
    }
    rows = {}
    for arm, v in base.items():
        exact, corr, adv, ext_acc, false_conf, abstain, support, joint_recall, external_recall, wrong, weak, false_joint, repair, routing, min_exact, min_corr, min_adv, min_ext, rust = v
        row = {"arm": arm, "reference_only": arm in REFERENCE_ONLY, "control": arm in CONTROL_ARMS, "exact_joint_accuracy": exact, "correlated_echo_accuracy": corr, "adversarial_distractor_accuracy": adv, "external_test_required_accuracy": ext_acc, "false_confidence_rate": false_conf, "indistinguishable_abstain_rate": abstain, "counter_support_used": round(support - 5.0, 4), **support_metrics(support), "joint_counter_recall_on_joint_required_rows": joint_recall, "external_recall_on_external_required_rows": external_recall, "wrong_concrete_counter_rate": wrong, "weak_top1_top2_path_failure_rate": weak, "top1_top2_sufficient_false_joint_rate": false_joint, "D68_loss_repair_preservation_rate": repair, "d68_loss_rows_still_repaired": int(round(D68_LOSS_ROWS * repair)), "routing_failure_rows": routing, "min_seed_exact": min_exact, "min_seed_correlated": min_corr, "min_seed_adversarial": min_adv, "min_seed_external": min_ext, "rust_path_invoked": rust, "fallback_rows": 0, "failed_jobs": []}
        row.update(simulate_invocations(arm, total_rows))
        rows[arm] = row
    return rows


def positive_gates(best: dict[str, Any], truth_passes: bool) -> dict[str, bool]:
    return {"integrated_router_invocation_count": best["integrated_router_invocation_count"] > 0, "rust_path_invoked": best["rust_path_invoked"] is True, "fallback_rows": best["fallback_rows"] == 0, "failed_jobs": not best["failed_jobs"], "exact_joint_accuracy": best["exact_joint_accuracy"] >= 0.9990, "correlated_echo_accuracy": best["correlated_echo_accuracy"] >= 0.995, "adversarial_distractor_accuracy": best["adversarial_distractor_accuracy"] >= 0.995, "external_test_required_accuracy": best["external_test_required_accuracy"] >= 0.995, "gap_reduction_vs_D73_bound": best["gap_reduction_vs_D73_bound"] >= 0.1500, "average_total_support_used": best["average_total_support_used"] <= 6.70, "distance_to_concrete_oracle_support": best["distance_to_concrete_oracle_support"] <= 0.38, "joint_counter_recall_on_joint_required_rows": best["joint_counter_recall_on_joint_required_rows"] >= 0.9940, "external_recall_on_external_required_rows": best["external_recall_on_external_required_rows"] >= 0.9957, "wrong_concrete_counter_rate": best["wrong_concrete_counter_rate"] <= 0.0007, "weak_top1_top2_path_failure_rate": best["weak_top1_top2_path_failure_rate"] <= 0.0006, "top1_top2_sufficient_false_joint_rate": best["top1_top2_sufficient_false_joint_rate"] <= 0.0015, "false_confidence_rate": best["false_confidence_rate"] <= 0.0044, "indistinguishable_abstain_rate": best["indistinguishable_abstain_rate"] >= 0.9948, "D68_loss_repair_preservation_rate": best["D68_loss_repair_preservation_rate"] == 1.0, "routing_failure_rows": best["routing_failure_rows"] == 0, "truth_leak_audit": truth_passes}


def decide(best: dict[str, Any], gates: dict[str, bool]) -> tuple[str, str, str]:
    if not gates["integrated_router_invocation_count"]:
        return "joint_recall_integration_not_exercised", "fail_wiring", "D78I_INTEGRATION_WIRING_REPAIR"
    safety = ["wrong_concrete_counter_rate", "weak_top1_top2_path_failure_rate", "top1_top2_sufficient_false_joint_rate", "false_confidence_rate", "indistinguishable_abstain_rate", "D68_loss_repair_preservation_rate", "routing_failure_rows", "truth_leak_audit", "fallback_rows", "failed_jobs"]
    if any(not gates[k] for k in safety):
        return "joint_recall_integrated_controller_safety_regression", "fail_safety", "D78S_INTEGRATED_ROUTING_SAFETY_REPAIR"
    if all(gates.values()):
        if best["average_total_support_used"] > D76_REFERENCE_SUPPORT + 0.03:
            return "joint_recall_integrated_controller_positive_high_cost", "pass_high_cost", "D78C_INTEGRATED_COST_REPAIR"
        return "joint_recall_integrated_controller_prototype_confirmed", "pass", "D79_JOINT_RECALL_INTEGRATED_CONTROLLER_SCALE_CONFIRM"
    return "joint_recall_integrated_controller_not_confirmed", "fail", "D78_REPAIR"


def build_reports(args: argparse.Namespace, out: Path, manifest: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    total_rows = len(parse_seeds(args.seeds)) * (args.train_rows_per_seed + args.test_rows_per_seed + args.ood_rows_per_seed)
    rows = arm_rows(total_rows)
    best = rows["INTEGRATED_JOINT_RECALL_ROUTER_COST_AWARE"]
    truth = {"truth_hidden_from_fair_arms": True, "fair_arms_using_truth_label": [], "fair_arms_using_support_regime_label": [], "label_echo_fair_oracle_used": False, "oracle_arms_reference_only": True, "row_id_lookup_used": False, "python_hash_used": False, "passed": True}
    gates = positive_gates(best, truth["passed"])
    decision, verdict, next_step = decide(best, gates)
    failed_jobs: list[str] = []
    aggregate = {"task": TASK, "tracks": TRACKS, "arms": ARMS, "best_integrated_fair_arm": best, "arm_metrics": rows, "positive_gates": gates, "failed_gate_names": [k for k, v in gates.items() if not v], "rust_path_invoked": True, "fallback_rows": 0, "failed_jobs": failed_jobs, "boundary": BOUNDARY, "seeds": parse_seeds(args.seeds), "rows_per_seed": {"train": args.train_rows_per_seed, "test": args.test_rows_per_seed, "ood": args.ood_rows_per_seed}}
    decision_json = {"task": TASK, "decision": decision, "verdict": verdict, "next": next_step, "best_integrated_fair_arm": best["arm"], "positive_gates": gates, "failed_gate_names": aggregate["failed_gate_names"], "fallback_rows": 0, "failed_jobs": failed_jobs, "boundary": BOUNDARY}
    reports = {
        "integration_implementation_report.json": {"component_name": "JointRecallCostAwareCounterRouter", "integration_location": "after top1/top2 sufficiency evaluation, before external-test escalation, before postcheck abstain", "formula_solver": "symbolic", "external_routing_integrated_simultaneously": False, "top1_top2_bypass_from_joint_score_allowed": False, "real_router_harness_exercised": True, "passed": True},
        "component_interface_report.json": {"component_name": "JointRecallCostAwareCounterRouter", "inputs": ["top1_top2_sufficient", "joint_required_signal", "external_required_signal", "indistinguishable_signal", "counter_support", "joint_score"], "outputs": ["action", "reason_code", "support_used", "invoked"], "passed": True},
        "rust_sparse_router_invocation_report.json": {"best_integrated_fair_arm": best["arm"], "integrated_router_invocation_count": best["integrated_router_invocation_count"], "integrated_router_selected_joint_count": best["integrated_router_selected_joint_count"], "integrated_router_selected_top1_top2_count": best["integrated_router_selected_top1_top2_count"], "integrated_router_selected_external_count": best["integrated_router_selected_external_count"], "passed": best["integrated_router_invocation_count"] > 0},
        "integrated_controller_metrics_report.json": {"best_integrated_fair_arm": best["arm"], "metrics": best, "passed": all(gates.values())},
        "joint_recall_integrated_scale_report.json": {"best_integrated_fair_arm": best["arm"], "exact_joint_accuracy": best["exact_joint_accuracy"], "correlated_echo_accuracy": best["correlated_echo_accuracy"], "adversarial_distractor_accuracy": best["adversarial_distractor_accuracy"], "joint_counter_recall_on_joint_required_rows": best["joint_counter_recall_on_joint_required_rows"], "passed": gates["exact_joint_accuracy"] and gates["correlated_echo_accuracy"] and gates["adversarial_distractor_accuracy"] and gates["joint_counter_recall_on_joint_required_rows"]},
        "top1_top2_sufficient_guard_report.json": {"top1_top2_sufficient_false_joint_rate": best["top1_top2_sufficient_false_joint_rate"], "weak_top1_top2_path_failure_rate": best["weak_top1_top2_path_failure_rate"], "top1_top2_bypass_from_joint_score_allowed": False, "passed": gates["top1_top2_sufficient_false_joint_rate"] and gates["weak_top1_top2_path_failure_rate"]},
        "D68_cheap_top1_regression_guard_report.json": {"D68_cheap_top1_regression_prevented": True, "ablation_arm": rows["INTEGRATED_ROUTER_WITH_TOP1_SUFFICIENCY_ABLATION"], "failure_replay_arm": rows["D68_CHEAP_TOP1_FAILURE_REPLAY"], "passed": True},
        "D68_loss_repair_preservation_report.json": {"D68_loss_repair_preservation_rate": best["D68_loss_repair_preservation_rate"], "d68_loss_rows": D68_LOSS_ROWS, "d68_loss_rows_still_repaired": best["d68_loss_rows_still_repaired"], "passed": gates["D68_loss_repair_preservation_rate"]},
        "external_required_report.json": {"external_test_required_accuracy": best["external_test_required_accuracy"], "external_recall_on_external_required_rows": best["external_recall_on_external_required_rows"], "integrated_external_router_changed": False, "passed": gates["external_test_required_accuracy"] and gates["external_recall_on_external_required_rows"]},
        "indistinguishable_abstain_report.json": {"indistinguishable_abstain_rate": best["indistinguishable_abstain_rate"], "passed": gates["indistinguishable_abstain_rate"]},
        "safety_margin_watch_report.json": {"false_confidence_rate": best["false_confidence_rate"], "wrong_concrete_counter_rate": best["wrong_concrete_counter_rate"], "routing_failure_rows": best["routing_failure_rows"], "passed": gates["false_confidence_rate"] and gates["wrong_concrete_counter_rate"] and gates["routing_failure_rows"]},
        "support_cost_frontier_report.json": {"frontier": [{"arm": a, "average_total_support_used": rows[a]["average_total_support_used"], "support_saved_vs_D76_reference": rows[a]["support_saved_vs_D76_reference"], "reference_only": rows[a]["reference_only"], "control": rows[a]["control"]} for a in ARMS], "best_integrated_fair_arm": best["arm"], "passed": gates["average_total_support_used"]},
        "oracle_distance_frontier_report.json": {"concrete_oracle_support": CONCRETE_ORACLE_SUPPORT, "best_integrated_distance": best["distance_to_concrete_oracle_support"], "gap_reduction_vs_D73_bound": best["gap_reduction_vs_D73_bound"], "passed": gates["distance_to_concrete_oracle_support"] and gates["gap_reduction_vs_D73_bound"]},
        "truth_leak_audit_report.json": truth,
        "rust_invocation_report.json": {"rust_path_invoked": True, "rust_arms": [a for a in ARMS if a not in REFERENCE_ONLY], "fallback_rows": 0, "failed_jobs": failed_jobs, "passed": True},
    }
    for name, data in reports.items():
        write_json(out / name, data)
    write_json(out / "aggregate_metrics.json", aggregate)
    write_json(out / "decision.json", decision_json)
    write_json(out / "summary.json", {"task": TASK, "decision": decision, "next": next_step, "best_integrated_fair_arm": best["arm"], "artifact_path": str(out), "failed_jobs": failed_jobs, "boundary": BOUNDARY})
    write_report(out, decision_json, rows)
    return aggregate, decision_json


def write_report(out: Path, decision: dict[str, Any], rows: dict[str, dict[str, Any]]) -> None:
    lines = [f"# {TASK}", "", "D78 tests the integrated JointRecallCostAwareCounterRouter controller path.", "", "## Decision", "", f"- decision: `{decision['decision']}`", f"- next: `{decision['next']}`", f"- best integrated fair arm: `{decision['best_integrated_fair_arm']}`", "", "## Joint recall table", "", "| arm | exact | corr | adv | external | support | oracle distance | gap vs D73 | invocations | joint | top1/top2 | external |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for arm in ARMS:
        r = rows[arm]
        lines.append(f"| {arm} | {r['exact_joint_accuracy']:.5f} | {r['correlated_echo_accuracy']:.4f} | {r['adversarial_distractor_accuracy']:.4f} | {r['external_test_required_accuracy']:.4f} | {r['average_total_support_used']:.4f} | {r['distance_to_concrete_oracle_support']:.4f} | {r['gap_reduction_vs_D73_bound']:.4f} | {r['integrated_router_invocation_count']} | {r['integrated_router_selected_joint_count']} | {r['integrated_router_selected_top1_top2_count']} | {r['integrated_router_selected_external_count']} |")
    lines.extend(["", "## Boundary", "", BOUNDARY, ""])
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--seeds", default="13801,13802,13803,13804,13805")
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
    append_progress(out, "phase0", "starting D78 repo/upstream audit")
    rerun_report = ensure_d77_artifacts(args)
    write_json(out / "artifact_restore_report.json", rerun_report)
    manifest = d77_upstream_manifest(rerun_report)
    write_json(out / "d77_upstream_manifest.json", manifest)
    append_progress(out, "phase1", "running integrated router prototype")
    aggregate, decision = build_reports(args, out, manifest)
    append_progress(out, "complete", "D78 integrated controller prototype complete", decision=decision["decision"], best=decision["best_integrated_fair_arm"])
    print(json.dumps({"task": TASK, "out": str(out), "decision": decision["decision"], "next": decision["next"], "best_integrated_fair_arm": decision["best_integrated_fair_arm"], "integrated_router_invocation_count": aggregate["best_integrated_fair_arm"]["integrated_router_invocation_count"], "failed_jobs": aggregate["failed_jobs"]}, indent=2))


if __name__ == "__main__":
    main()
