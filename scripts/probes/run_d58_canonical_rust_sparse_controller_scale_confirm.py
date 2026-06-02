#!/usr/bin/env python3
"""D58 canonical Rust sparse controller scale confirm."""

import argparse
import copy
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import run_d51_mutable_ecf_controller_prototype as d51
import run_d53_mutable_ecf_integration_with_vraxion_mutation_architecture as d53
import run_d55_sparse_firing_ecf_controller_prototype as d55
import run_d57_canonical_rust_sparse_path_bridge as d57

PRIMARY_SPACE = d55.PRIMARY_SPACE
SUPPORT_COUNT = d55.SUPPORT_COUNT
REGIMES = d55.REGIMES
CORE_REGIMES = d55.CORE_REGIMES
ACTIONS = d55.ACTIONS
FEATURE_NAMES = d55.FEATURE_NAMES
ROW_SAMPLE_PER_ARM_REGIME_SPLIT = d55.ROW_SAMPLE_PER_ARM_REGIME_SPLIT
BRIDGE_CALLS = ["Network::new", "Network::graph_mut().add_edge", "Network::propagate_sparse", "Network::spike_data"]

BOUNDARY = (
    "D58 only scale-confirms a canonical Rust sparse ECF action controller on controlled "
    "symbolic joint formula discovery. It does not prove full VRAXION brain, raw visual Raven "
    "reasoning, Raven solved, AGI, consciousness, DNA/genome success, architecture superiority, "
    "or production readiness."
)

ARMS = [
    "D57_CANONICAL_RUST_REPLAY",
    "CANONICAL_RUST_SPARSE_CONTROLLER",
    "PYTHON_SPARSE_REFERENCE",
    "RUST_SPIKE_SHUFFLE_CONTROL",
    "RUST_THRESHOLD_ABLATION",
    "RUST_REWIRE_ABLATION",
    "RUST_PATH_DISABLED_CONTROL",
    "RANDOM_POLICY_CONTROL",
    "GREEDY_DECIDE_CONTROL",
    "ALWAYS_COUNTER_CONTROL",
]

d57.ARMS = ARMS


def write_json(path, value):
    d51.write_json(path, value)


def append_jsonl(path, value):
    d51.append_jsonl(path, value)


def append_progress(out, event, started, data):
    d51.append_progress(out, event, started, data)


def parse_seeds(raw):
    return [int(item.strip()) for item in str(raw).split(",") if item.strip()]


def load_json_if_present(path):
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def make_d57_upstream_manifest(repo_root):
    root = repo_root / "target/pilot_wave/d57_canonical_rust_sparse_path_bridge/smoke"
    manifest = {
        "upstream": "D57_CANONICAL_RUST_SPARSE_PATH_BRIDGE",
        "expected_decision": "canonical_rust_sparse_path_bridge_positive",
        "expected_next": "D58_CANONICAL_RUST_SPARSE_CONTROLLER_SCALE_CONFIRM",
        "artifact_root": str(root),
        "decision_present": (root / "decision.json").exists(),
        "summary_present": (root / "summary.json").exists(),
        "trained_policy_manifest_present": (root / "trained_policy_manifest.json").exists(),
    }
    for name in ["decision.json", "summary.json", "rust_path_usage_report.json", "python_vs_rust_action_parity_report.json"]:
        value = load_json_if_present(root / name)
        if value is not None:
            manifest[name.replace(".json", "")] = value
    trained = load_json_if_present(root / "trained_policy_manifest.json")
    manifest["d57_canonical_controller_loaded"] = bool(
        trained and trained.get("controllers", {}).get("CANONICAL_RUST_SPARSE_NETWORK_BRIDGE")
    )
    return manifest


def load_d57_controller(repo_root):
    trained = load_json_if_present(repo_root / "target/pilot_wave/d57_canonical_rust_sparse_path_bridge/smoke/trained_policy_manifest.json")
    if not trained:
        return None
    return trained.get("controllers", {}).get("CANONICAL_RUST_SPARSE_NETWORK_BRIDGE")


def make_canonical_sparse_path_report(repo_root, invoked):
    audit = d55.canonical_sparse_firing_audit(repo_root)
    audit.update(
        {
            "canonical_rust_network_path_probe_arm": "CANONICAL_RUST_SPARSE_CONTROLLER",
            "canonical_rust_network_path_available": audit["canonical_rust_network_audited"],
            "canonical_rust_network_path_invoked": bool(invoked),
            "canonical_rust_probe_status": "invoked_in_d58" if invoked else "not_invoked",
            "rust_bridge_uses_path_dependency": True,
            "rust_bridge_calls": BRIDGE_CALLS,
        }
    )
    return audit


def output_from_action(pack, arm, action, trace=None):
    row = d55.output_from_action(pack, arm, action, trace)
    row["rust_network_path_invoked"] = bool(trace and trace.get("rust_network_path_invoked"))
    row["rust_propagate_sparse_called"] = bool(trace and trace.get("rust_propagate_sparse_called"))
    row["python_fallback_used"] = bool(trace and trace.get("python_fallback_used"))
    row["rust_marker_charge_sum"] = int(trace.get("marker_charge_sum", 0)) if trace else 0
    return row


def evaluate_pack_all_arms(pack, index, controllers, rust_actions, sparse_stats):
    rows = []
    py_controller = controllers["PYTHON_SPARSE_REFERENCE"]
    py_action, py_sparse_trace = d55.choose_sparse_action(py_controller, pack["features"], sparse_stats["PYTHON_SPARSE_REFERENCE"])
    rows.append(output_from_action(pack, "PYTHON_SPARSE_REFERENCE", py_action, d57.python_trace(py_sparse_trace)))

    for arm in ["D57_CANONICAL_RUST_REPLAY", "CANONICAL_RUST_SPARSE_CONTROLLER"]:
        action_record = rust_actions[arm][index]
        rows.append(output_from_action(pack, arm, action_record["action"], d57.rust_trace(action_record)))

    canonical_action = rust_actions["CANONICAL_RUST_SPARSE_CONTROLLER"][index]
    shuffle = d55.spike_shuffle_mapping()
    rows.append(output_from_action(pack, "RUST_SPIKE_SHUFFLE_CONTROL", shuffle[canonical_action["action"]], d57.rust_trace(canonical_action)))

    for arm in ["RUST_THRESHOLD_ABLATION", "RUST_REWIRE_ABLATION"]:
        action_record = rust_actions[arm][index]
        rows.append(output_from_action(pack, arm, action_record["action"], d57.rust_trace(action_record)))

    rows.append(output_from_action(pack, "RUST_PATH_DISABLED_CONTROL", "DECIDE", d57.disabled_rust_trace()))
    rows.append(output_from_action(pack, "RANDOM_POLICY_CONTROL", d51.random_policy_action(f"D58:{pack['row_id']}")))
    rows.append(output_from_action(pack, "GREEDY_DECIDE_CONTROL", "DECIDE"))
    rows.append(output_from_action(pack, "ALWAYS_COUNTER_CONTROL", "REQUEST_JOINT_COUNTER"))
    return rows


def summarize_outputs(outputs):
    by_arm = defaultdict(list)
    by_arm_core = defaultdict(list)
    by_arm_regime = defaultdict(list)
    by_action = defaultdict(Counter)
    by_error = defaultdict(Counter)
    by_seed_core = defaultdict(list)
    by_seed_regime = defaultdict(list)
    rust_counts = defaultdict(Counter)
    for row in outputs:
        arm = row["arm"]
        by_arm[arm].append(row)
        by_arm_regime[(arm, row["support_regime"])].append(row)
        by_action[arm][row["selected_action"]] += 1
        by_error[(arm, row["support_regime"])][row["error_type"]] += 1
        rust_counts[arm]["rows"] += 1
        if row["rust_network_path_invoked"]:
            rust_counts[arm]["rust_rows"] += 1
        if row["python_fallback_used"]:
            rust_counts[arm]["python_fallback_rows"] += 1
        if row["support_regime"] in CORE_REGIMES:
            by_arm_core[arm].append(row)
            by_seed_core[(arm, row["seed"])].append(row)
        by_seed_regime[(arm, row["seed"], row["support_regime"])].append(row)
    return {
        "by_arm": {arm: d51.summarize(by_arm[arm]) for arm in ARMS if arm in by_arm},
        "by_arm_core": {arm: d51.summarize(by_arm_core[arm]) for arm in ARMS if arm in by_arm_core},
        "by_arm_and_regime": {
            arm: {regime: d51.summarize(by_arm_regime[(arm, regime)]) for regime in REGIMES if (arm, regime) in by_arm_regime}
            for arm in ARMS
        },
        "action_distribution": {arm: {action: by_action[arm][action] for action in ACTIONS} for arm in ARMS},
        "error_taxonomy": {
            arm: {regime: dict(by_error[(arm, regime)]) for regime in REGIMES if (arm, regime) in by_error}
            for arm in ARMS
        },
        "by_seed_core": {
            arm: {str(seed): d51.summarize(rows) for (a, seed), rows in by_seed_core.items() if a == arm}
            for arm in ARMS
        },
        "by_seed_regime": {
            arm: {
                str(seed): {
                    regime: d51.summarize(by_seed_regime[(arm, seed, regime)])
                    for regime in REGIMES
                    if (arm, seed, regime) in by_seed_regime
                }
                for seed in sorted({seed for (a, seed, _regime) in by_seed_regime if a == arm})
            }
            for arm in ARMS
        },
        "rust_usage": {arm: dict(counts) for arm, counts in rust_counts.items()},
    }


def record_batch(batch, outputs, sample_counts, path):
    for row in batch:
        outputs.append(row)
        key = (row["arm"], row["support_regime"])
        if sample_counts[key] < ROW_SAMPLE_PER_ARM_REGIME_SPLIT:
            append_jsonl(path, row)
            sample_counts[key] += 1


def write_partial_eval(out, split, outputs, completed, started):
    # Keep heartbeat writes bounded. Full summarization is O(rows) and becomes
    # expensive at D58 scale; final reports still summarize the complete split.
    recent = outputs[-min(len(outputs), 2000):]
    partial = summarize_outputs(recent)
    write_json(
        out / f"partial_{split}_metrics_snapshot.json",
        {
            "split": split,
            "completed_outputs": completed,
            "elapsed_sec": time.time() - started,
            "canonical_rust_core": partial["by_arm_core"].get("CANONICAL_RUST_SPARSE_CONTROLLER", {}),
        },
    )
    append_progress(out, "eval_progress", started, {"split": split, "completed_outputs": completed})


def evaluate_packs(packs, controllers, out_path, out, split, started, heartbeat_sec, repo_root):
    if out_path.exists():
        out_path.unlink()
    rust_actions = {}
    rust_reports = {}
    for arm in ["D57_CANONICAL_RUST_REPLAY", "CANONICAL_RUST_SPARSE_CONTROLLER", "RUST_THRESHOLD_ABLATION", "RUST_REWIRE_ABLATION"]:
        rust_actions[arm], rust_reports[arm] = d57.run_rust_bridge(out, repo_root, controllers[arm], packs, split, arm, started)
    outputs = []
    sparse_stats = defaultdict(
        lambda: {
            "calls": 0,
            "spike_update_executed_count": 0,
            "fired_gate_count": 0,
            "total_input_charge": 0,
            "output_charge_sum": 0,
            "action_counts": Counter(),
            "fired_gate_by_feature": Counter(),
            "fired_gate_by_action": Counter(),
        }
    )
    sample_counts = Counter()
    completed = 0
    total = len(packs) * len(ARMS)
    last = 0.0
    for idx, pack in enumerate(packs):
        batch = evaluate_pack_all_arms(pack, idx, controllers, rust_actions, sparse_stats)
        record_batch(batch, outputs, sample_counts, out_path)
        completed += len(batch)
        now = time.time()
        if now - last >= heartbeat_sec or completed >= total:
            last = now
            write_partial_eval(out, split, outputs, completed, started)
    return outputs, d55.normalize_sparse_stats(sparse_stats), rust_reports


def regime_accuracy(metrics, arm, regime):
    return metrics["by_arm_and_regime"][arm][regime]["accuracy"]


def min_seed_metric(metrics, arm, metric):
    rows = [seed_row[metric] for seed_row in metrics["by_seed_core"].get(arm, {}).values()]
    return min(rows) if rows else 0.0


def min_seed_regime_accuracy(metrics, arm, regime):
    values = []
    for seed_rows in metrics["by_seed_regime"].get(arm, {}).values():
        if regime in seed_rows:
            values.append(seed_rows[regime]["accuracy"])
    return min(values) if values else 0.0


def action_parity(outputs):
    grouped = defaultdict(dict)
    for row in outputs:
        if row["arm"] in {"PYTHON_SPARSE_REFERENCE", "CANONICAL_RUST_SPARSE_CONTROLLER"}:
            key = (row["split"], row["row_id"], row["support_regime"])
            grouped[key][row["arm"]] = row["selected_action"]
    compared = 0
    matched = 0
    mismatches = []
    for key, value in grouped.items():
        if "PYTHON_SPARSE_REFERENCE" in value and "CANONICAL_RUST_SPARSE_CONTROLLER" in value:
            compared += 1
            if value["PYTHON_SPARSE_REFERENCE"] == value["CANONICAL_RUST_SPARSE_CONTROLLER"]:
                matched += 1
            elif len(mismatches) < 25:
                mismatches.append({"key": key, "python": value["PYTHON_SPARSE_REFERENCE"], "rust": value["CANONICAL_RUST_SPARSE_CONTROLLER"]})
    return {
        "compared": compared,
        "matched": matched,
        "parity_rate": matched / compared if compared else 0.0,
        "sample_mismatches": mismatches,
    }


def gates_pass(metrics, failed_jobs, rust_usage, parity):
    if failed_jobs:
        return False
    arm = "CANONICAL_RUST_SPARSE_CONTROLLER"
    core = metrics["by_arm_core"]
    regimes = metrics["by_arm_and_regime"]
    row = core[arm]
    corr = regime_accuracy(metrics, arm, "CORRELATED_ECHO_SUPPORT")
    adv = regime_accuracy(metrics, arm, "ADVERSARIAL_DISTRACTOR_SUPPORT")
    external = regime_accuracy(metrics, arm, "EXTERNAL_TEST_REQUIRED_SUPPORT")
    indist = regimes[arm]["INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT"]
    rust_rows = rust_usage.get(arm, {}).get("rust_rows", 0)
    fallback_rows = rust_usage.get(arm, {}).get("python_fallback_rows", 0)
    return (
        rust_rows > 0
        and fallback_rows == 0
        and row["exact_joint_accuracy"] >= 0.995
        and corr >= 0.995
        and adv >= 0.995
        and external >= 0.99
        and indist["abstain_rate"] >= 0.99
        and indist["false_confidence_rate"] <= 0.01
        and row["average_total_support_used"] <= core["ALWAYS_COUNTER_CONTROL"]["average_total_support_used"]
        and row["accuracy"] > core["RANDOM_POLICY_CONTROL"]["accuracy"]
        and row["accuracy"] > core["GREEDY_DECIDE_CONTROL"]["accuracy"]
        and row["accuracy"] > core["RUST_SPIKE_SHUFFLE_CONTROL"]["accuracy"]
        and row["accuracy"] > core["RUST_THRESHOLD_ABLATION"]["accuracy"]
        and row["accuracy"] > core["RUST_REWIRE_ABLATION"]["accuracy"]
        and min_seed_metric(metrics, arm, "exact_joint_accuracy") >= 0.99
        and min_seed_regime_accuracy(metrics, arm, "CORRELATED_ECHO_SUPPORT") >= 0.99
        and min_seed_regime_accuracy(metrics, arm, "ADVERSARIAL_DISTRACTOR_SUPPORT") >= 0.99
        and parity["parity_rate"] >= 0.995
    )


def make_decision(metrics, failed_jobs, rust_usage, parity, scale_mode):
    arm = "CANONICAL_RUST_SPARSE_CONTROLLER"
    rust_rows = rust_usage.get(arm, {}).get("rust_rows", 0)
    fallback_rows = rust_usage.get(arm, {}).get("python_fallback_rows", 0)
    if rust_rows == 0 or fallback_rows > 0:
        return {
            "decision": "rust_sparse_path_not_cleanly_exercised",
            "verdict": "D58_RUST_PATH_NOT_CLEANLY_EXERCISED",
            "next": "D58R_RUST_PATH_REPAIR",
            "best_arm": arm,
            "boundary": BOUNDARY,
        }
    if gates_pass(metrics, failed_jobs, rust_usage, parity):
        if scale_mode == "full":
            return {
                "decision": "canonical_rust_sparse_controller_scale_confirmed",
                "verdict": "D58_CANONICAL_RUST_SPARSE_CONTROLLER_SCALE_CONFIRMED",
                "next": "D59_RUST_SPARSE_ECF_CONTROLLER_WITH_MUTATION",
                "best_arm": arm,
                "boundary": BOUNDARY,
            }
        return {
            "decision": "canonical_rust_sparse_controller_scale_lite_confirmed",
            "verdict": "D58_CANONICAL_RUST_SPARSE_CONTROLLER_SCALE_LITE_CONFIRMED",
            "next": "D58F_FULL_SCALE_RERUN_OR_D59",
            "best_arm": arm,
            "boundary": BOUNDARY,
        }
    return {
        "decision": "canonical_rust_sparse_controller_scale_not_confirmed",
        "verdict": "D58_CANONICAL_RUST_SPARSE_CONTROLLER_SCALE_NOT_CONFIRMED",
        "next": "D58_REPAIR",
        "best_arm": arm,
        "boundary": BOUNDARY,
    }


def make_reports(out, aggregate, decision, d57_manifest, canonical_report, controllers):
    test = aggregate["test_metrics"]
    core = test["by_arm_core"]
    regimes = test["by_arm_and_regime"]
    rust_usage = test["rust_usage"]
    parity = aggregate["python_vs_rust_action_parity"]
    reports = {
        "d57_upstream_manifest.json": d57_manifest,
        "rust_bridge_invocation_report.json": aggregate["rust_bridge_invocation_report"],
        "rust_path_usage_report.json": {
            "canonical_rust_network_path_invoked": rust_usage.get("CANONICAL_RUST_SPARSE_CONTROLLER", {}).get("rust_rows", 0) > 0,
            "rust_propagate_sparse_called": rust_usage.get("CANONICAL_RUST_SPARSE_CONTROLLER", {}).get("rust_rows", 0) > 0,
            "canonical_arm_python_fallback_rows": rust_usage.get("CANONICAL_RUST_SPARSE_CONTROLLER", {}).get("python_fallback_rows", 0),
            "by_arm": rust_usage,
        },
        "python_fallback_audit.json": {
            "python_sparse_reference_present": True,
            "canonical_arm_python_fallback_used": rust_usage.get("CANONICAL_RUST_SPARSE_CONTROLLER", {}).get("python_fallback_rows", 0) > 0,
            "python_reference_arm": "PYTHON_SPARSE_REFERENCE",
        },
        "python_vs_rust_action_parity_report.json": parity,
        "canonical_sparse_path_report.json": canonical_report,
        "firing_dynamics_report.json": {
            "python_sparse_stats": aggregate["python_sparse_stats"],
            "rust_rows": {
                arm: rust_usage.get(arm, {}).get("rust_rows", 0)
                for arm in ARMS
            },
        },
        "network_topology_report.json": d55.network_topology_report(controllers),
        "support_cost_frontier_report.json": {
            arm: {
                "exact_joint_accuracy": core[arm]["exact_joint_accuracy"],
                "support": core[arm]["average_total_support_used"],
                "counter_support": core[arm]["average_counter_support_used"],
                "external_test": core[arm]["average_external_test_used"],
                "cost_adjusted": d53.cost_adjusted(core[arm]),
            }
            for arm in sorted(ARMS, key=lambda item: (core[item]["average_total_support_used"], -core[item]["exact_joint_accuracy"]))
        },
        "false_confidence_report.json": {
            arm: {
                "core_false_confidence": core[arm]["false_confidence_rate"],
                "indistinguishable_false_confidence": regimes[arm]["INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT"]["false_confidence_rate"],
                "indistinguishable_abstain": regimes[arm]["INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT"]["abstain_rate"],
            }
            for arm in ARMS
        },
        "regime_breakdown_report.json": regimes,
        "ablation_report.json": {
            arm: {
                "exact_joint_accuracy": core[arm]["exact_joint_accuracy"],
                "correlated_echo": regime_accuracy(test, arm, "CORRELATED_ECHO_SUPPORT"),
                "adversarial_distractor": regime_accuracy(test, arm, "ADVERSARIAL_DISTRACTOR_SUPPORT"),
                "support": core[arm]["average_total_support_used"],
            }
            for arm in ["RUST_SPIKE_SHUFFLE_CONTROL", "RUST_THRESHOLD_ABLATION", "RUST_REWIRE_ABLATION", "RUST_PATH_DISABLED_CONTROL", "RANDOM_POLICY_CONTROL", "GREEDY_DECIDE_CONTROL"]
        },
        "controller_comparison_report.json": {
            arm: {
                "exact_joint_accuracy": core[arm]["exact_joint_accuracy"],
                "correlated_echo": regime_accuracy(test, arm, "CORRELATED_ECHO_SUPPORT"),
                "adversarial_distractor": regime_accuracy(test, arm, "ADVERSARIAL_DISTRACTOR_SUPPORT"),
                "external_test_required": regime_accuracy(test, arm, "EXTERNAL_TEST_REQUIRED_SUPPORT"),
                "indistinguishable_abstain": regimes[arm]["INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT"]["abstain_rate"],
                "false_confidence": regimes[arm]["INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT"]["false_confidence_rate"],
                "support": core[arm]["average_total_support_used"],
                "counter_support": core[arm]["average_counter_support_used"],
                "rust_network_path_invoked": rust_usage.get(arm, {}).get("rust_rows", 0) > 0,
            }
            for arm in ARMS
        },
        "min_seed_gate_report.json": {
            arm: {
                "min_seed_exact_joint": min_seed_metric(test, arm, "exact_joint_accuracy"),
                "min_seed_correlated_echo": min_seed_regime_accuracy(test, arm, "CORRELATED_ECHO_SUPPORT"),
                "min_seed_adversarial_distractor": min_seed_regime_accuracy(test, arm, "ADVERSARIAL_DISTRACTOR_SUPPORT"),
            }
            for arm in ARMS
        },
    }
    for name, value in reports.items():
        write_json(out / name, value)
    return reports


def write_report(out, aggregate, decision):
    test = aggregate["test_metrics"]
    core = test["by_arm_core"]
    regimes = test["by_arm_and_regime"]
    rust_usage = test["rust_usage"]
    lines = [
        "# D58 Canonical Rust Sparse Controller Scale Confirm Result",
        "",
        "Decision:",
        "",
        "```text",
        f"decision = {decision['decision']}",
        f"verdict = {decision.get('verdict')}",
        f"next = {decision.get('next')}",
        "```",
        "",
        "Boundary:",
        "",
        "```text",
        BOUNDARY,
        "```",
        "",
        "Controller comparison:",
        "",
        "| arm | exact | corr | adv | external | support | rust |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        lines.append(
            f"| {arm} | {core[arm]['exact_joint_accuracy']:.5f} | "
            f"{regimes[arm]['CORRELATED_ECHO_SUPPORT']['accuracy']:.5f} | "
            f"{regimes[arm]['ADVERSARIAL_DISTRACTOR_SUPPORT']['accuracy']:.5f} | "
            f"{regimes[arm]['EXTERNAL_TEST_REQUIRED_SUPPORT']['accuracy']:.5f} | "
            f"{core[arm]['average_total_support_used']:.3f} | "
            f"{rust_usage.get(arm, {}).get('rust_rows', 0)} |"
        )
    lines.extend(
        [
            "",
            "Bridge audit:",
            "",
            "```text",
            f"rust_network_path_invoked = {aggregate['canonical_sparse_path_report']['canonical_rust_network_path_invoked']}",
            f"python_vs_rust_action_parity = {aggregate['python_vs_rust_action_parity']['parity_rate']:.5f}",
            f"canonical_arm_python_fallback_rows = {rust_usage.get('CANONICAL_RUST_SPARSE_CONTROLLER', {}).get('python_fallback_rows', 0)}",
            "```",
            "",
        ]
    )
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def make_summary(aggregate, decision):
    test = aggregate["test_metrics"]
    arm = "CANONICAL_RUST_SPARSE_CONTROLLER"
    core = test["by_arm_core"][arm]
    regimes = test["by_arm_and_regime"][arm]
    return {
        "task": "D58_CANONICAL_RUST_SPARSE_CONTROLLER_SCALE_CONFIRM",
        "decision": decision["decision"],
        "verdict": decision.get("verdict"),
        "next": decision.get("next"),
        "scale_mode": aggregate["scale_mode"],
        "best_arm": arm,
        "rust_network_path_invoked": aggregate["canonical_sparse_path_report"]["canonical_rust_network_path_invoked"],
        "python_fallback_rows": test["rust_usage"][arm].get("python_fallback_rows", 0),
        "python_vs_rust_action_parity": aggregate["python_vs_rust_action_parity"]["parity_rate"],
        "key_metrics": {
            "exact_joint_accuracy": core["exact_joint_accuracy"],
            "correlated_echo": regimes["CORRELATED_ECHO_SUPPORT"]["accuracy"],
            "adversarial_distractor": regimes["ADVERSARIAL_DISTRACTOR_SUPPORT"]["accuracy"],
            "external_test_required": regimes["EXTERNAL_TEST_REQUIRED_SUPPORT"]["accuracy"],
            "indistinguishable_abstain": regimes["INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT"]["abstain_rate"],
            "false_confidence": regimes["INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT"]["false_confidence_rate"],
            "support": core["average_total_support_used"],
        },
        "boundary": BOUNDARY,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", default="11201,11202,11203,11204,11205,11206,11207,11208")
    parser.add_argument("--train-rows-per-seed", type=int, default=1200)
    parser.add_argument("--test-rows-per-seed", type=int, default=1200)
    parser.add_argument("--ood-rows-per-seed", type=int, default=1200)
    parser.add_argument("--workers", default="auto")
    parser.add_argument("--cpu-target", default="50-75")
    parser.add_argument("--heartbeat-sec", type=float, default=20.0)
    parser.add_argument("--scale-mode", default="full")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    seeds = parse_seeds(args.seeds)
    repo_root = Path(__file__).resolve().parents[2]
    failed_jobs = []
    write_json(out / "queue.json", {"task": "D58_CANONICAL_RUST_SPARSE_CONTROLLER_SCALE_CONFIRM", "args": vars(args), "seeds": seeds, "boundary": BOUNDARY})
    append_progress(out, "started", started, {"seeds": seeds, "scale_mode": args.scale_mode})
    write_json(out / "compute_probe.json", {"cpu_count": os.cpu_count(), "workers": d51.worker_count_from_arg(args.workers), "cpu_target": args.cpu_target})

    d57_manifest = make_d57_upstream_manifest(repo_root)
    write_json(out / "d57_upstream_manifest.json", d57_manifest)
    base_controller = load_d57_controller(repo_root)
    if base_controller is None:
        failed_jobs.append("missing_d57_canonical_controller")
        base_controller = d57.load_d56_best_controller(repo_root)

    controllers = {
        "D57_CANONICAL_RUST_REPLAY": copy.deepcopy(base_controller),
        "CANONICAL_RUST_SPARSE_CONTROLLER": copy.deepcopy(base_controller),
        "PYTHON_SPARSE_REFERENCE": copy.deepcopy(base_controller),
        "RUST_THRESHOLD_ABLATION": d55.make_threshold_ablation(base_controller),
        "RUST_REWIRE_ABLATION": d55.make_rewire_ablation(base_controller),
    }

    bundle = d55.d49_bundle()
    write_json(
        out / "dataset_manifest.json",
        {
            "task": "controlled_symbolic_joint_formula_discovery",
            "primary_space": PRIMARY_SPACE,
            "support_count": SUPPORT_COUNT,
            "regimes": REGIMES,
            "core_regimes": CORE_REGIMES,
            "feature_names": FEATURE_NAMES,
            "truth_hidden_from_controller_inputs": True,
            "controller_only_not_formula_solver": True,
            "formula_solver_learning_used": False,
        },
    )

    train_rows = d51.make_rows_with_progress(seeds, args.train_rows_per_seed, "train", bundle, out, started, args.heartbeat_sec)
    test_rows = d51.make_rows_with_progress(seeds, args.test_rows_per_seed, "test", bundle, out, started, args.heartbeat_sec)
    ood_rows = d51.make_rows_with_progress(seeds, args.ood_rows_per_seed, "ood", bundle, out, started, args.heartbeat_sec)
    append_progress(out, "rows_generated", started, {"train": len(train_rows), "test": len(test_rows), "ood": len(ood_rows)})
    train_packs = d51.build_packs(train_rows, bundle, out, started, args.heartbeat_sec, args.workers, "train")
    test_packs = d51.build_packs(test_rows, bundle, out, started, args.heartbeat_sec, args.workers, "test")
    ood_packs = d51.build_packs(ood_rows, bundle, out, started, args.heartbeat_sec, args.workers, "ood")
    append_progress(out, "packs_built", started, {"train": len(train_packs), "test": len(test_packs), "ood": len(ood_packs)})

    test_outputs, test_sparse_stats, test_rust_reports = evaluate_packs(test_packs, controllers, out / "row_outputs_test.jsonl", out, "test", started, args.heartbeat_sec, repo_root)
    ood_outputs, ood_sparse_stats, ood_rust_reports = evaluate_packs(ood_packs, controllers, out / "row_outputs_ood.jsonl", out, "ood", started, args.heartbeat_sec, repo_root)
    all_outputs = test_outputs + ood_outputs
    test_metrics = summarize_outputs(test_outputs)
    ood_metrics = summarize_outputs(ood_outputs)
    parity = action_parity(all_outputs)
    rust_invoked = test_metrics["rust_usage"].get("CANONICAL_RUST_SPARSE_CONTROLLER", {}).get("rust_rows", 0) > 0
    canonical_report = make_canonical_sparse_path_report(repo_root, rust_invoked)

    aggregate = {
        "task": "D58_CANONICAL_RUST_SPARSE_CONTROLLER_SCALE_CONFIRM",
        "scale_mode": args.scale_mode,
        "failed_jobs": failed_jobs,
        "test_metrics": test_metrics,
        "ood_metrics": ood_metrics,
        "python_sparse_stats": {"test": test_sparse_stats, "ood": ood_sparse_stats},
        "rust_bridge_invocation_report": {"test": test_rust_reports, "ood": ood_rust_reports},
        "python_vs_rust_action_parity": parity,
        "canonical_sparse_path_report": canonical_report,
        "boundary": BOUNDARY,
    }
    decision = make_decision(test_metrics, failed_jobs, test_metrics["rust_usage"], parity, args.scale_mode)
    aggregate["decision"] = decision
    write_json(out / "aggregate_metrics.json", aggregate)
    write_json(out / "decision.json", decision)
    reports = make_reports(out, aggregate, decision, d57_manifest, canonical_report, controllers)
    write_json(out / "summary.json", make_summary(aggregate, decision))
    write_report(out, aggregate, decision)
    write_json(
        out / "trained_policy_manifest.json",
        {
            "controllers": controllers,
            "rust_bridge_harness": str(out / "rust_bridge_harness"),
            "decision": decision,
        },
    )
    append_progress(out, "completed", started, {"decision": decision["decision"], "reports": sorted(reports.keys())})
    print(json.dumps(make_summary(aggregate, decision), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
