#!/usr/bin/env python3
"""D56 sparse firing ECF controller scale confirm."""

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

PRIMARY_SPACE = d55.PRIMARY_SPACE
SUPPORT_COUNT = d55.SUPPORT_COUNT
REGIMES = d55.REGIMES
CORE_REGIMES = d55.CORE_REGIMES
ACTIONS = d55.ACTIONS
FEATURE_NAMES = d55.FEATURE_NAMES
ROW_SAMPLE_PER_ARM_REGIME_SPLIT = d55.ROW_SAMPLE_PER_ARM_REGIME_SPLIT

BOUNDARY = (
    "D56 only scale-confirms a sparse-firing ECF controller for controlled symbolic joint formula "
    "discovery. It does not prove full VRAXION brain, raw visual Raven reasoning, Raven solved, "
    "AGI, consciousness, DNA/genome success, architecture superiority, or production readiness."
)

REFERENCE_ARMS = [
    "D55_BEST_SPARSE_REPLAY",
    "D54_BEST_HYBRID_REPLAY",
    "D50_HANDCODED_FULL_REFERENCE",
]

SPARSE_ARMS = [
    "REAL_SPARSE_FIRING_CONTROLLER_WITH_MUTATION",
    "REAL_SPARSE_FIRING_CONTROLLER_NO_MUTATION",
    "SMALL_SPARSE_CONTROLLER",
    "MEDIUM_SPARSE_CONTROLLER",
]

CONTROL_ARMS = [
    "SPIKE_SHUFFLE_CONTROL",
    "THRESHOLD_ABLATION",
    "CONNECTION_REWIRE_ABLATION",
    "RANDOM_POLICY_CONTROL",
    "GREEDY_DECIDE_CONTROL",
    "ALWAYS_COUNTER_CONTROL",
]

ARMS = REFERENCE_ARMS + SPARSE_ARMS + CONTROL_ARMS


def write_json(path, value):
    d51.write_json(path, value)


def append_jsonl(path, value):
    d51.append_jsonl(path, value)


def append_progress(out, event, started, data):
    d51.append_progress(out, event, started, data)


def load_json_if_present(path):
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


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


def parse_seeds(raw):
    return [int(item.strip()) for item in str(raw).split(",") if item.strip()]


def make_d55_upstream_manifest(repo_root):
    root = repo_root / "target/pilot_wave/d55_sparse_firing_ecf_controller_prototype/smoke"
    manifest = {
        "upstream": "D55_SPARSE_FIRING_ECF_CONTROLLER_PROTOTYPE",
        "expected_decision": "sparse_firing_ecf_controller_prototype_strong_positive",
        "expected_next": "D56_SPARSE_FIRING_ECF_CONTROLLER_SCALE_CONFIRM",
        "artifact_root": str(root),
        "decision_present": (root / "decision.json").exists(),
        "summary_present": (root / "summary.json").exists(),
        "trained_policy_manifest_present": (root / "trained_policy_manifest.json").exists(),
        "sparse_firing_usage_present": (root / "sparse_firing_usage_report.json").exists(),
    }
    for name in ["decision.json", "summary.json", "sparse_firing_usage_report.json"]:
        value = load_json_if_present(root / name)
        if value is not None:
            manifest[name.replace(".json", "")] = value
    return manifest


def load_d55_best_controller(repo_root, manifest):
    trained = load_json_if_present(repo_root / "target/pilot_wave/d55_sparse_firing_ecf_controller_prototype/smoke/trained_policy_manifest.json")
    if not trained:
        manifest["d55_best_sparse_loaded"] = False
        return None
    controller = trained.get("sparse_controllers", {}).get("REAL_SPARSE_FIRING_CONTROLLER_WITH_MUTATION")
    manifest["d55_best_sparse_loaded"] = controller is not None
    return controller


def make_canonical_sparse_path_report(repo_root):
    audit = d55.canonical_sparse_firing_audit(repo_root)
    audit.update(
        {
            "canonical_rust_network_path_probe_arm": "CANONICAL_RUST_NETWORK_PATH_PROBE",
            "canonical_rust_network_path_available": audit["canonical_rust_network_audited"],
            "canonical_rust_network_path_invoked": False,
            "canonical_rust_probe_status": "not_invoked_in_d56",
            "next_required_bridge": "D57_CANONICAL_RUST_SPARSE_PATH_BRIDGE",
        }
    )
    return audit


def renamed_reference(pack, arm):
    result = d55.renamed_reference(pack, arm)
    result["arm"] = arm
    return result


def output_from_action(pack, arm, action, sparse_trace=None):
    return d55.output_from_action(pack, arm, action, sparse_trace)


def make_sparse_controllers(d55_controller, d54_sparse_gate, train_examples, args, out, started):
    if d55_controller is None:
        d55_controller = d55.make_sparse_controller_from_gate(d54_sparse_gate, "fallback_d54_sparse_gate_seed", hidden_neurons=0)
    replay = copy.deepcopy(d55_controller)
    replay["name"] = "D55_BEST_SPARSE_REPLAY"
    no_mut = copy.deepcopy(d55_controller)
    no_mut["name"] = "REAL_SPARSE_FIRING_CONTROLLER_NO_MUTATION"
    small = d55.make_sparse_controller_from_gate(d54_sparse_gate, "SMALL_SPARSE_CONTROLLER", hidden_neurons=0)
    medium = d55.make_sparse_controller_from_gate(d54_sparse_gate, "MEDIUM_SPARSE_CONTROLLER", hidden_neurons=32)
    trained, report = d55.train_sparse_controller(
        copy.deepcopy(d55_controller),
        train_examples,
        args.generations,
        args.population,
        56_000,
        out,
        started,
        args.heartbeat_sec,
    )
    trained["name"] = "REAL_SPARSE_FIRING_CONTROLLER_WITH_MUTATION"
    controllers = {
        "D55_BEST_SPARSE_REPLAY": replay,
        "REAL_SPARSE_FIRING_CONTROLLER_WITH_MUTATION": trained,
        "REAL_SPARSE_FIRING_CONTROLLER_NO_MUTATION": no_mut,
        "SMALL_SPARSE_CONTROLLER": small,
        "MEDIUM_SPARSE_CONTROLLER": medium,
        "THRESHOLD_ABLATION": d55.make_threshold_ablation(trained),
        "CONNECTION_REWIRE_ABLATION": d55.make_rewire_ablation(trained),
    }
    return controllers, report


def evaluate_pack_all_arms(pack, d54_policies, sparse_controllers, sparse_stats):
    rows = []
    d55_replay = sparse_controllers["D55_BEST_SPARSE_REPLAY"]
    action, trace = d55.choose_sparse_action(d55_replay, pack["features"], sparse_stats["D55_BEST_SPARSE_REPLAY"])
    rows.append(output_from_action(pack, "D55_BEST_SPARSE_REPLAY", action, trace))

    hybrid = d54_policies.get("VRAXION_MUTABLE_HYBRID_CONTROLLER")
    if hybrid is not None:
        rows.append(output_from_action(pack, "D54_BEST_HYBRID_REPLAY", d53.choose_vraxion_action(hybrid, pack["features"])))
    rows.append(renamed_reference(pack, "D50_HANDCODED_FULL_REFERENCE"))

    for arm in SPARSE_ARMS:
        controller = sparse_controllers[arm]
        action, trace = d55.choose_sparse_action(controller, pack["features"], sparse_stats[arm])
        rows.append(output_from_action(pack, arm, action, trace))

    shuffle = d55.spike_shuffle_mapping()
    action, trace = d55.choose_sparse_action(
        sparse_controllers["REAL_SPARSE_FIRING_CONTROLLER_WITH_MUTATION"],
        pack["features"],
        sparse_stats["SPIKE_SHUFFLE_CONTROL"],
        action_shuffle=shuffle,
    )
    rows.append(output_from_action(pack, "SPIKE_SHUFFLE_CONTROL", action, trace))

    for arm in ["THRESHOLD_ABLATION", "CONNECTION_REWIRE_ABLATION"]:
        action, trace = d55.choose_sparse_action(sparse_controllers[arm], pack["features"], sparse_stats[arm])
        rows.append(output_from_action(pack, arm, action, trace))

    rows.append(output_from_action(pack, "RANDOM_POLICY_CONTROL", d51.random_policy_action(f"D56:{pack['row_id']}")))
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
    for row in outputs:
        arm = row["arm"]
        by_arm[arm].append(row)
        by_arm_regime[(arm, row["support_regime"])].append(row)
        by_action[arm][row["selected_action"]] += 1
        by_error[(arm, row["support_regime"])][row["error_type"]] += 1
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
    }


def record_batch(batch, outputs, sample_counts, path):
    for row in batch:
        outputs.append(row)
        key = (row["arm"], row["support_regime"])
        if sample_counts[key] < ROW_SAMPLE_PER_ARM_REGIME_SPLIT:
            append_jsonl(path, row)
            sample_counts[key] += 1


def write_partial_eval(out, split, outputs, completed, started):
    partial = summarize_outputs(outputs)
    write_json(
        out / f"partial_{split}_metrics_snapshot.json",
        {
            "split": split,
            "completed_outputs": completed,
            "elapsed_sec": time.time() - started,
            "with_mutation_core": partial["by_arm_core"].get("REAL_SPARSE_FIRING_CONTROLLER_WITH_MUTATION", {}),
        },
    )
    append_progress(out, "eval_progress", started, {"split": split, "completed_outputs": completed})


def evaluate_packs(packs, d54_policies, sparse_controllers, out_path, out, split, started, heartbeat_sec):
    if out_path.exists():
        out_path.unlink()
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
    for pack in packs:
        batch = evaluate_pack_all_arms(pack, d54_policies, sparse_controllers, sparse_stats)
        record_batch(batch, outputs, sample_counts, out_path)
        completed += len(batch)
        now = time.time()
        if now - last >= heartbeat_sec or completed >= total:
            last = now
            write_partial_eval(out, split, outputs, completed, started)
    return outputs, d55.normalize_sparse_stats(sparse_stats)


def best_sparse_arm(metrics):
    core = metrics["by_arm_core"]
    candidates = ["REAL_SPARSE_FIRING_CONTROLLER_WITH_MUTATION", "D55_BEST_SPARSE_REPLAY"]
    return max(candidates, key=lambda arm: (core[arm]["exact_joint_accuracy"], -core[arm]["average_total_support_used"]))


def make_decision(metrics, failed_jobs, sparse_usage, canonical_report):
    if failed_jobs:
        return {
            "decision": "sparse_firing_ecf_controller_scale_not_confirmed",
            "verdict": "D56_FAILED_JOBS_PRESENT",
            "next": "D56_REPAIR",
            "boundary": BOUNDARY,
        }
    arm = best_sparse_arm(metrics)
    core = metrics["by_arm_core"]
    regimes = metrics["by_arm_and_regime"]
    row = core[arm]
    corr = regime_accuracy(metrics, arm, "CORRELATED_ECHO_SUPPORT")
    adv = regime_accuracy(metrics, arm, "ADVERSARIAL_DISTRACTOR_SUPPORT")
    external = regime_accuracy(metrics, arm, "EXTERNAL_TEST_REQUIRED_SUPPORT")
    indist = regimes[arm]["INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT"]
    passes = (
        sparse_usage["sparse_firing_used"]
        and row["exact_joint_accuracy"] >= 0.995
        and corr >= 0.995
        and adv >= 0.995
        and external >= 0.99
        and indist["abstain_rate"] >= 0.99
        and indist["false_confidence_rate"] <= 0.01
        and row["average_total_support_used"] <= core["ALWAYS_COUNTER_CONTROL"]["average_total_support_used"]
        and row["accuracy"] > core["RANDOM_POLICY_CONTROL"]["accuracy"]
        and row["accuracy"] > core["GREEDY_DECIDE_CONTROL"]["accuracy"]
        and row["accuracy"] > core["SPIKE_SHUFFLE_CONTROL"]["accuracy"]
        and row["accuracy"] > core["THRESHOLD_ABLATION"]["accuracy"]
        and row["accuracy"] > core["CONNECTION_REWIRE_ABLATION"]["accuracy"]
        and min_seed_metric(metrics, arm, "exact_joint_accuracy") >= 0.99
        and min_seed_regime_accuracy(metrics, arm, "CORRELATED_ECHO_SUPPORT") >= 0.99
        and min_seed_regime_accuracy(metrics, arm, "ADVERSARIAL_DISTRACTOR_SUPPORT") >= 0.99
    )
    if passes and canonical_report["canonical_rust_network_path_invoked"]:
        return {
            "decision": "sparse_firing_ecf_controller_scale_confirmed",
            "verdict": "D56_SPARSE_FIRING_ECF_CONTROLLER_SCALE_CONFIRMED",
            "next": "D57_CANONICAL_VRAXION_SPARSE_NETWORK_INTEGRATION",
            "best_sparse_arm": arm,
            "boundary": BOUNDARY,
        }
    if passes:
        return {
            "decision": "sparse_firing_controller_scale_confirmed_python_path_only",
            "verdict": "D56_SPARSE_FIRING_CONTROLLER_SCALE_CONFIRMED_PYTHON_PATH_ONLY",
            "next": "D57_CANONICAL_RUST_SPARSE_PATH_BRIDGE",
            "best_sparse_arm": arm,
            "boundary": BOUNDARY,
        }
    return {
        "decision": "sparse_firing_ecf_controller_scale_not_confirmed",
        "verdict": "D56_SPARSE_FIRING_ECF_CONTROLLER_SCALE_NOT_CONFIRMED",
        "next": "D56_REPAIR",
        "best_sparse_arm": arm,
        "boundary": BOUNDARY,
    }


def make_reports(out, aggregate, decision, d55_manifest, canonical_report, sparse_controllers):
    test = aggregate["test_metrics"]
    core = test["by_arm_core"]
    regimes = test["by_arm_and_regime"]
    best = decision.get("best_sparse_arm", best_sparse_arm(test))
    reports = {
        "d55_upstream_manifest.json": d55_manifest,
        "sparse_firing_usage_report.json": aggregate["sparse_firing_usage"],
        "canonical_sparse_path_report.json": canonical_report,
        "python_local_vs_rust_path_report.json": {
            "python_controller_local_sparse_path_used": True,
            "rust_network_path_invoked": canonical_report["canonical_rust_network_path_invoked"],
            "canonical_rust_network_audited": canonical_report["canonical_rust_network_audited"],
            "decision_route_expected_if_pass": "sparse_firing_controller_scale_confirmed_python_path_only",
        },
        "network_topology_report.json": d55.network_topology_report(sparse_controllers),
        "firing_dynamics_report.json": aggregate["sparse_firing_stats"],
        "mutation_acceptance_report.json": aggregate["policy_reports"],
        "action_readout_report.json": {
            arm: {
                "readout": controller["readout"],
                "default_action": controller["default_action"],
                "action_distribution": test["action_distribution"].get(arm, {}),
            }
            for arm, controller in sparse_controllers.items()
        },
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
            for arm in ["SPIKE_SHUFFLE_CONTROL", "THRESHOLD_ABLATION", "CONNECTION_REWIRE_ABLATION", "RANDOM_POLICY_CONTROL", "GREEDY_DECIDE_CONTROL"]
        },
        "mutation_causality_report.json": {
            "with_mutation": core["REAL_SPARSE_FIRING_CONTROLLER_WITH_MUTATION"],
            "no_mutation": core["REAL_SPARSE_FIRING_CONTROLLER_NO_MUTATION"],
            "d55_replay": core["D55_BEST_SPARSE_REPLAY"],
            "mutation_changed_accuracy": core["REAL_SPARSE_FIRING_CONTROLLER_WITH_MUTATION"]["exact_joint_accuracy"]
            - core["REAL_SPARSE_FIRING_CONTROLLER_NO_MUTATION"]["exact_joint_accuracy"],
            "interpretation": "Reports whether D56 mutation improved the sparse controller or mainly exercised the mutation path.",
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
                "sparse_firing_used": arm in SPARSE_ARMS or arm in {"D55_BEST_SPARSE_REPLAY", "SPIKE_SHUFFLE_CONTROL", "THRESHOLD_ABLATION", "CONNECTION_REWIRE_ABLATION"},
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
    best = decision.get("best_sparse_arm", best_sparse_arm(test))
    lines = [
        "# D56 Sparse Firing ECF Controller Scale Confirm Result",
        "",
        "Status:",
        "",
        "```text",
        "completed",
        f"scale_mode = {aggregate['scale_mode']}",
        "```",
        "",
        "Decision:",
        "",
        "```text",
        f"decision = {decision['decision']}",
        f"verdict = {decision.get('verdict')}",
        f"next = {decision.get('next')}",
        f"best_sparse_arm = {best}",
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
        "| arm | exact | corr | adv | external | support |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        lines.append(
            f"| {arm} | {core[arm]['exact_joint_accuracy']:.5f} | "
            f"{regimes[arm]['CORRELATED_ECHO_SUPPORT']['accuracy']:.5f} | "
            f"{regimes[arm]['ADVERSARIAL_DISTRACTOR_SUPPORT']['accuracy']:.5f} | "
            f"{regimes[arm]['EXTERNAL_TEST_REQUIRED_SUPPORT']['accuracy']:.5f} | "
            f"{core[arm]['average_total_support_used']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Sparse path:",
            "",
            "```text",
            f"sparse_firing_used = {aggregate['sparse_firing_usage']['sparse_firing_used']}",
            f"spike_update_executed_count = {aggregate['sparse_firing_usage']['spike_update_executed_count']}",
            f"rust_network_path_invoked = {aggregate['canonical_sparse_path_report']['canonical_rust_network_path_invoked']}",
            "```",
            "",
        ]
    )
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def make_summary(aggregate, decision):
    test = aggregate["test_metrics"]
    best = decision.get("best_sparse_arm", best_sparse_arm(test))
    core = test["by_arm_core"][best]
    regimes = test["by_arm_and_regime"][best]
    return {
        "task": "D56_SPARSE_FIRING_ECF_CONTROLLER_SCALE_CONFIRM",
        "decision": decision["decision"],
        "verdict": decision.get("verdict"),
        "next": decision.get("next"),
        "scale_mode": aggregate["scale_mode"],
        "best_sparse_arm": best,
        "sparse_firing_used": aggregate["sparse_firing_usage"]["sparse_firing_used"],
        "rust_network_path_invoked": aggregate["canonical_sparse_path_report"]["canonical_rust_network_path_invoked"],
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
    parser.add_argument("--seeds", default="11001,11002,11003,11004,11005,11006,11007,11008")
    parser.add_argument("--train-rows-per-seed", type=int, default=1200)
    parser.add_argument("--test-rows-per-seed", type=int, default=1200)
    parser.add_argument("--ood-rows-per-seed", type=int, default=1200)
    parser.add_argument("--generations", type=int, default=240)
    parser.add_argument("--population", type=int, default=80)
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
    write_json(
        out / "queue.json",
        {
            "task": "D56_SPARSE_FIRING_ECF_CONTROLLER_SCALE_CONFIRM",
            "args": vars(args),
            "seeds": seeds,
            "created_unix_ms": int(started * 1000),
            "boundary": BOUNDARY,
        },
    )
    append_progress(out, "started", started, {"seeds": seeds, "scale_mode": args.scale_mode})
    write_json(
        out / "compute_probe.json",
        {
            "cpu_count": os.cpu_count(),
            "workers": d51.worker_count_from_arg(args.workers),
            "cpu_target": args.cpu_target,
            "cuda_probe": "not_used_controller_local_python",
        },
    )

    d55_manifest = make_d55_upstream_manifest(repo_root)
    d55_controller = load_d55_best_controller(repo_root, d55_manifest)
    write_json(out / "d55_upstream_manifest.json", d55_manifest)
    if d55_controller is None:
        failed_jobs.append("missing_d55_best_sparse_controller")

    d54_manifest = {}
    d54_policies = d55.load_d54_policies(repo_root, d54_manifest)
    if "VRAXION_MUTABLE_SPARSE_GATE_CONTROLLER" not in d54_policies:
        failed_jobs.append("missing_d54_sparse_gate_policy")
    if "VRAXION_MUTABLE_HYBRID_CONTROLLER" not in d54_policies:
        failed_jobs.append("missing_d54_hybrid_policy")

    canonical_report = make_canonical_sparse_path_report(repo_root)
    write_json(out / "canonical_sparse_path_report.json", canonical_report)
    if not canonical_report["canonical_rust_network_audited"]:
        failed_jobs.append("canonical_sparse_surface_missing")

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

    train_examples = [{"features": pack["features"], "action_compact": pack["action_compact"]} for pack in train_packs]
    sparse_seed_policy = d54_policies.get("VRAXION_MUTABLE_SPARSE_GATE_CONTROLLER")
    sparse_controllers, sparse_report = make_sparse_controllers(d55_controller, sparse_seed_policy, train_examples, args, out, started)

    policy_reports = {"REAL_SPARSE_FIRING_CONTROLLER_WITH_MUTATION": sparse_report}
    for arm in ["D55_BEST_SPARSE_REPLAY", "REAL_SPARSE_FIRING_CONTROLLER_NO_MUTATION", "SMALL_SPARSE_CONTROLLER", "MEDIUM_SPARSE_CONTROLLER", "THRESHOLD_ABLATION", "CONNECTION_REWIRE_ABLATION"]:
        score, counts = d55.full_sparse_policy_score(sparse_controllers[arm], train_examples)
        policy_reports[arm] = {
            "kind": "controller_local_sparse_firing",
            "fitness": score,
            "action_counts": counts,
            "mutation_counts": {},
            "accepted_mutation_counts": {},
            "history": [],
        }
    write_json(out / "trained_policy_manifest.json", {"sparse_controllers": sparse_controllers, "policy_reports": policy_reports})
    append_progress(out, "sparse_training_complete", started, {"fitness": sparse_report["fitness"]})

    test_outputs, test_sparse_stats = evaluate_packs(
        test_packs, d54_policies, sparse_controllers, out / "row_outputs_test.jsonl", out, "test", started, args.heartbeat_sec
    )
    ood_outputs, ood_sparse_stats = evaluate_packs(
        ood_packs, d54_policies, sparse_controllers, out / "row_outputs_ood.jsonl", out, "ood", started, args.heartbeat_sec
    )
    test_metrics = summarize_outputs(test_outputs)
    ood_metrics = summarize_outputs(ood_outputs)
    total_spike_updates = sum(stats["spike_update_executed_count"] for stats in test_sparse_stats.values()) + sum(
        stats["spike_update_executed_count"] for stats in ood_sparse_stats.values()
    )
    sparse_usage = {
        "sparse_firing_used": total_spike_updates > 0,
        "actual_spike_update_executed": total_spike_updates > 0,
        "spike_update_executed_count": total_spike_updates,
        "controller_local_sparse_firing_path_used": True,
        "full_sparse_firing_brain_trained": False,
        "controller_only_not_formula_solver": True,
        "rust_network_path_invoked": canonical_report["canonical_rust_network_path_invoked"],
        "canonical_rust_network_audited": canonical_report["canonical_rust_network_audited"],
    }
    aggregate = {
        "task": "D56_SPARSE_FIRING_ECF_CONTROLLER_SCALE_CONFIRM",
        "scale_mode": args.scale_mode,
        "seeds": seeds,
        "train_rows_per_seed": args.train_rows_per_seed,
        "test_rows_per_seed": args.test_rows_per_seed,
        "ood_rows_per_seed": args.ood_rows_per_seed,
        "generations": args.generations,
        "population": args.population,
        "failed_jobs": failed_jobs,
        "test_metrics": test_metrics,
        "ood_metrics": ood_metrics,
        "policy_reports": policy_reports,
        "sparse_firing_stats": {"test": test_sparse_stats, "ood": ood_sparse_stats},
        "sparse_firing_usage": sparse_usage,
        "canonical_sparse_path_report": canonical_report,
        "boundary": BOUNDARY,
    }
    decision = make_decision(test_metrics, failed_jobs, sparse_usage, canonical_report)
    aggregate["decision"] = decision
    reports = make_reports(out, aggregate, decision, d55_manifest, canonical_report, sparse_controllers)
    aggregate["reports_written"] = sorted(reports.keys())
    write_json(out / "aggregate_metrics.json", aggregate)
    write_json(out / "decision.json", decision)
    write_json(out / "summary.json", make_summary(aggregate, decision))
    write_report(out, aggregate, decision)
    append_progress(out, "completed", started, {"decision": decision["decision"], "elapsed_sec": time.time() - started})


if __name__ == "__main__":
    main()
