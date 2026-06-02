#!/usr/bin/env python3
"""D52 mutable ECF controller scale confirm.

This probe keeps the D50/D51 controlled symbolic joint formula solver fixed and
scale-confirms only the mutable controller policy. The controller chooses among
decision/support/counter-support/external-test/abstain actions from diagnostic
features; it does not receive true cells, true operators, or answer labels.
"""

import argparse
import copy
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import run_d51_mutable_ecf_controller_prototype as d51

PRIMARY_SPACE = d51.PRIMARY_SPACE
SUPPORT_COUNT = d51.SUPPORT_COUNT
REGIMES = d51.REGIMES
CORE_REGIMES = d51.CORE_REGIMES
ACTIONS = d51.ACTIONS
FEATURE_NAMES = d51.FEATURE_NAMES

BOUNDARY = (
    "D52 only scale-confirms mutable control policy for controlled symbolic joint formula discovery. "
    "It does not prove raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, "
    "or architecture superiority."
)

REFERENCE_MAP = {
    "D50_FULL_HANDCODED_REFERENCE": "HANDCODED_D50_FULL_REPAIRED_ECF_REFERENCE",
    "CAP_7_REFERENCE": "HANDCODED_CAP_7_REFERENCE",
    "CAP_9_REFERENCE": "HANDCODED_CAP_9_REFERENCE",
}

REFERENCE_ARMS = list(REFERENCE_MAP.keys())
CONTROL_ARMS = [
    "ALWAYS_COUNTER_CONTROL",
    "RANDOM_POLICY_CONTROL",
    "GREEDY_DECIDE_CONTROL",
    "COST_ONLY_MUTABLE_CONTROL",
]
MUTABLE_ARMS = [
    "MUTABLE_LINEAR_CONTROLLER",
    "MUTABLE_RULE_TABLE_CONTROLLER",
    "MUTABLE_SMALL_TREE_CONTROLLER",
    "MUTABLE_HYBRID_CONTROLLER",
]
REPLAY_ARMS = ["BEST_D51_REPLAY"]
ARMS = REFERENCE_ARMS + CONTROL_ARMS + MUTABLE_ARMS + REPLAY_ARMS


def write_json(path, value):
    d51.write_json(path, value)


def append_jsonl(path, value):
    d51.append_jsonl(path, value)


def append_progress(out, event, started, data):
    d51.append_progress(out, event, started, data)


def mean(values):
    return d51.mean(values)


def cost_adjusted(row):
    return row["exact_joint_accuracy"] - 0.0010 * row["average_total_support_used"] - 0.0015 * row["average_counter_support_used"]


def regime_accuracy(metrics, arm, regime):
    return metrics["by_arm_and_regime"][arm][regime]["accuracy"]


def mixed_accuracy(metrics, arm):
    return mean(
        [
            regime_accuracy(metrics, arm, "MIXED_CLEAN_AND_CORRELATED"),
            regime_accuracy(metrics, arm, "MIXED_CLEAN_AND_ADVERSARIAL"),
        ]
    )


def load_json_if_present(path):
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def make_d51_upstream_manifest(repo_root):
    root = repo_root / "target/pilot_wave/d51_mutable_ecf_controller_prototype/smoke"
    manifest = {
        "upstream": "D51_MUTABLE_ECF_CONTROLLER_PROTOTYPE",
        "expected_decision": "mutable_ecf_controller_prototype_positive",
        "expected_next": "D52_MUTABLE_ECF_CONTROLLER_SCALE_CONFIRM",
        "decision_present": (root / "decision.json").exists(),
        "summary_present": (root / "summary.json").exists(),
        "trained_policy_manifest_present": (root / "trained_policy_manifest.json").exists(),
    }
    decision = load_json_if_present(root / "decision.json")
    summary = load_json_if_present(root / "summary.json")
    aggregate = load_json_if_present(root / "aggregate_metrics.json")
    if decision:
        manifest["decision_json"] = decision
    if summary:
        manifest["summary_json"] = {
            "scale_mode": summary.get("scale_mode"),
            "decision": summary.get("decision"),
            "best_mutable_arm": summary.get("best_mutable_arm"),
        }
        key = summary.get("key_metrics", {})
        if "best_mutable" in key:
            best = key["best_mutable"]
            manifest["key_metrics"] = {
                "best_exact_joint": best["exact_joint_accuracy"],
                "best_support": best["average_total_support_used"],
                "best_counter_support": best["average_counter_support_used"],
                "d50_full_support": key.get("d50_full_reference", {}).get("average_total_support_used"),
                "cap9_support": key.get("cap9_reference", {}).get("average_total_support_used"),
            }
    if aggregate:
        manifest["aggregate_best_mutable_arm"] = aggregate.get("best_mutable_arm")
    return manifest


def load_best_d51_policy(repo_root, upstream_manifest):
    root = repo_root / "target/pilot_wave/d51_mutable_ecf_controller_prototype/smoke"
    trained = load_json_if_present(root / "trained_policy_manifest.json")
    aggregate = load_json_if_present(root / "aggregate_metrics.json")
    if not trained or not aggregate:
        upstream_manifest["best_d51_replay_loaded"] = False
        return None
    best = aggregate.get("best_mutable_arm")
    policy = trained.get("policies", {}).get(best)
    upstream_manifest["best_d51_replay_loaded"] = policy is not None
    upstream_manifest["best_d51_replay_arm"] = best
    return policy


def renamed_reference(pack, arm):
    result = copy.deepcopy(pack["references"][REFERENCE_MAP[arm]])
    result["arm"] = arm
    return result


def output_from_action(pack, arm, action):
    result = copy.deepcopy(pack["actions"][action])
    result["arm"] = arm
    result["selected_action"] = action
    return result


def cost_only_action(pack):
    candidates = []
    for action in ACTIONS:
        result = pack["actions"][action]
        candidates.append(
            (
                result["total_support_used"],
                result["counter_support_used"],
                result["external_test_used"],
                ACTIONS.index(action),
                action,
            )
        )
    candidates.sort()
    return candidates[0][-1]


def evaluate_pack_all_arms(pack, policies):
    rows = []
    for arm in REFERENCE_ARMS:
        rows.append(renamed_reference(pack, arm))
    rows.append(output_from_action(pack, "ALWAYS_COUNTER_CONTROL", "REQUEST_JOINT_COUNTER"))
    rows.append(output_from_action(pack, "RANDOM_POLICY_CONTROL", d51.random_policy_action(f"D52:{pack['row_id']}")))
    rows.append(output_from_action(pack, "GREEDY_DECIDE_CONTROL", "DECIDE"))
    rows.append(output_from_action(pack, "COST_ONLY_MUTABLE_CONTROL", cost_only_action(pack)))
    for arm in MUTABLE_ARMS + REPLAY_ARMS:
        policy = policies.get(arm)
        if policy is None:
            continue
        action = d51.choose_action(policy, pack["features"])
        rows.append(output_from_action(pack, arm, action))
    return rows


def summarize(rows):
    return d51.summarize(rows)


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

    seed_core = {
        arm: {str(seed): summarize(rows) for (a, seed), rows in by_seed_core.items() if a == arm}
        for arm in ARMS
    }
    seed_regime = {
        arm: {
            str(seed): {
                regime: summarize(by_seed_regime[(arm, seed, regime)])
                for regime in REGIMES
                if (arm, seed, regime) in by_seed_regime
            }
            for seed in sorted({seed for (a, seed, _regime) in by_seed_regime if a == arm})
        }
        for arm in ARMS
    }
    return {
        "by_arm": {arm: summarize(by_arm[arm]) for arm in ARMS if arm in by_arm},
        "by_arm_core": {arm: summarize(by_arm_core[arm]) for arm in ARMS if arm in by_arm_core},
        "by_arm_and_regime": {
            arm: {regime: summarize(by_arm_regime[(arm, regime)]) for regime in REGIMES if (arm, regime) in by_arm_regime}
            for arm in ARMS
        },
        "action_distribution": {
            arm: {action: by_action[arm][action] for action in ACTIONS}
            for arm in ARMS
        },
        "error_taxonomy": {
            arm: {
                regime: dict(by_error[(arm, regime)])
                for regime in REGIMES
                if (arm, regime) in by_error
            }
            for arm in ARMS
        },
        "by_seed_core": seed_core,
        "by_seed_regime": seed_regime,
    }


def record_batch(batch, outputs, sample_counts, path):
    for row in batch:
        outputs.append(row)
        key = (row["arm"], row["support_regime"])
        if sample_counts[key] < d51.ROW_SAMPLE_PER_ARM_REGIME_SPLIT:
            append_jsonl(path, row)
            sample_counts[key] += 1


def write_partial_eval(out, split, outputs, completed, started):
    partial = summarize_outputs(outputs)
    best = best_mutable_arm(partial) if all(arm in partial["by_arm_core"] for arm in MUTABLE_ARMS) else None
    write_json(
        out / f"partial_{split}_metrics_snapshot.json",
        {
            "split": split,
            "completed_outputs": completed,
            "elapsed_sec": time.time() - started,
            "by_arm_core": partial["by_arm_core"],
            "best_mutable_arm_so_far": best,
            "best_mutable_regime_so_far": partial["by_arm_and_regime"].get(best, {}) if best else {},
        },
    )
    append_progress(out, "eval_progress", started, {"split": split, "completed_outputs": completed, "best_mutable_so_far": best})


def evaluate_packs(packs, policies, out_path, out, split, started, heartbeat_sec):
    if out_path.exists():
        out_path.unlink()
    outputs = []
    sample_counts = Counter()
    completed = 0
    total = len(packs) * len(ARMS)
    last = 0.0
    for pack in packs:
        batch = evaluate_pack_all_arms(pack, policies)
        record_batch(batch, outputs, sample_counts, out_path)
        completed += len(batch)
        now = time.time()
        if now - last >= heartbeat_sec or completed >= total:
            last = now
            write_partial_eval(out, split, outputs, completed, started)
    return outputs


def best_mutable_arm(metrics):
    core = metrics["by_arm_core"]
    return max(MUTABLE_ARMS, key=lambda arm: (cost_adjusted(core[arm]), core[arm]["exact_joint_accuracy"]))


def min_seed_metric(metrics, arm, metric):
    rows = [seed_row[metric] for seed_row in metrics["by_seed_core"].get(arm, {}).values()]
    return min(rows) if rows else 0.0


def min_seed_regime_accuracy(metrics, arm, regime):
    values = []
    for seed_rows in metrics["by_seed_regime"].get(arm, {}).values():
        if regime in seed_rows:
            values.append(seed_rows[regime]["accuracy"])
    return min(values) if values else 0.0


def cost_only_passes_safety(metrics):
    arm = "COST_ONLY_MUTABLE_CONTROL"
    core = metrics["by_arm_core"][arm]
    corr = regime_accuracy(metrics, arm, "CORRELATED_ECHO_SUPPORT")
    adv = regime_accuracy(metrics, arm, "ADVERSARIAL_DISTRACTOR_SUPPORT")
    external = regime_accuracy(metrics, arm, "EXTERNAL_TEST_REQUIRED_SUPPORT")
    indist = metrics["by_arm_and_regime"][arm]["INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT"]
    return (
        core["exact_joint_accuracy"] >= 0.995
        and corr >= 0.995
        and adv >= 0.995
        and external >= 0.99
        and indist["false_confidence_rate"] <= 0.01
        and indist["abstain_rate"] >= 0.99
    )


def make_decision(metrics, failed_jobs):
    if failed_jobs:
        return {
            "decision": "mutable_controller_failed_jobs_present",
            "verdict": "D52_FAILED_JOBS_PRESENT",
            "next": "D52_REPAIR",
            "boundary": BOUNDARY,
        }
    best = best_mutable_arm(metrics)
    core = metrics["by_arm_core"]
    best_row = core[best]
    corr = regime_accuracy(metrics, best, "CORRELATED_ECHO_SUPPORT")
    adv = regime_accuracy(metrics, best, "ADVERSARIAL_DISTRACTOR_SUPPORT")
    external = regime_accuracy(metrics, best, "EXTERNAL_TEST_REQUIRED_SUPPORT")
    indist = metrics["by_arm_and_regime"][best]["INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT"]
    d50_full = core["D50_FULL_HANDCODED_REFERENCE"]
    random_control = core["RANDOM_POLICY_CONTROL"]
    greedy_control = core["GREEDY_DECIDE_CONTROL"]
    always_counter = core["ALWAYS_COUNTER_CONTROL"]
    min_exact = min_seed_metric(metrics, best, "exact_joint_accuracy")
    min_corr = min_seed_regime_accuracy(metrics, best, "CORRELATED_ECHO_SUPPORT")
    min_adv = min_seed_regime_accuracy(metrics, best, "ADVERSARIAL_DISTRACTOR_SUPPORT")
    pass_accuracy = (
        best_row["exact_joint_accuracy"] >= 0.995
        and corr >= 0.995
        and adv >= 0.995
        and external >= 0.99
        and indist["false_confidence_rate"] <= 0.01
        and indist["abstain_rate"] >= 0.99
        and min_exact >= 0.99
        and min_corr >= 0.99
        and min_adv >= 0.99
    )
    pass_cost = (
        best_row["average_total_support_used"] <= d50_full["average_total_support_used"]
        and always_counter["average_total_support_used"] > best_row["average_total_support_used"]
    )
    controls_worse = (
        random_control["exact_joint_accuracy"] < best_row["exact_joint_accuracy"]
        and greedy_control["exact_joint_accuracy"] < best_row["exact_joint_accuracy"]
    )
    if cost_only_passes_safety(metrics):
        return {
            "decision": "mutable_controller_fitness_exploit_detected",
            "verdict": "D52_FITNESS_EXPLOIT_DETECTED",
            "next": "D52F_FITNESS_REPAIR",
            "best_mutable_arm": best,
            "boundary": BOUNDARY,
        }
    if pass_accuracy and pass_cost and controls_worse:
        return {
            "decision": "mutable_ecf_controller_scale_confirmed",
            "verdict": "D52_MUTABLE_ECF_CONTROLLER_SCALE_CONFIRMED",
            "next": "D53_MUTABLE_ECF_INTEGRATION_WITH_VRAXION_MUTATION_ARCHITECTURE",
            "best_mutable_arm": best,
            "boundary": BOUNDARY,
        }
    if pass_accuracy and controls_worse:
        return {
            "decision": "mutable_ecf_controller_scale_confirmed_high_cost",
            "verdict": "D52_MUTABLE_ECF_CONTROLLER_SCALE_CONFIRMED_HIGH_COST",
            "next": "D52C_SUPPORT_COST_OPTIMIZATION",
            "best_mutable_arm": best,
            "boundary": BOUNDARY,
        }
    return {
        "decision": "mutable_ecf_controller_scale_not_confirmed",
        "verdict": "D52_MUTABLE_ECF_CONTROLLER_SCALE_NOT_CONFIRMED",
        "next": "D52_REPAIR",
        "best_mutable_arm": best,
        "boundary": BOUNDARY,
    }


def make_reports(out, aggregate, decision, upstream_manifest):
    test = aggregate["test_metrics"]
    ood = aggregate["ood_metrics"]
    core = test["by_arm_core"]
    best = decision.get("best_mutable_arm", best_mutable_arm(test))
    support_rows = {
        arm: {
            "original_support": SUPPORT_COUNT,
            "average_total_support_used": core[arm]["average_total_support_used"],
            "average_counter_support_used": core[arm]["average_counter_support_used"],
            "average_external_test_used": core[arm]["average_external_test_used"],
            "support_cost_delta_vs_d50_full": core[arm]["average_total_support_used"] - core["D50_FULL_HANDCODED_REFERENCE"]["average_total_support_used"],
        }
        for arm in ARMS
    }
    reports = {
        "fitness_audit_report.json": {
            "feature_names": FEATURE_NAMES,
            "truth_hidden_from_controller_inputs": True,
            "fitness_uses_outcome_labels_only_during_training": True,
            "fitness_penalizes_false_confidence": True,
            "fitness_penalizes_unavailable_external_test": True,
            "fitness_penalizes_support_cost_after_accuracy": True,
            "cost_only_control_passes_safety": cost_only_passes_safety(test),
            "policy_reports": aggregate["policy_reports"],
            "d51_replay_loaded": upstream_manifest.get("best_d51_replay_loaded", False),
        },
        "support_accounting_report.json": {
            "definitions": {
                "original_support_used": "The fixed base support boards supplied before controller action.",
                "counter_support_used": "Cell/operator/joint/random counter-support rows, excluding external tests.",
                "external_test_used": "Marked external/interventional test rows.",
                "total_support_used": "original_support_used + counter_support_used + external_test_used.",
            },
            "audit_rule": "Every row must satisfy total = original + counter + external.",
            "per_arm": support_rows,
        },
        "action_distribution_report.json": test["action_distribution"],
        "mutation_acceptance_report.json": aggregate["policy_reports"],
        "support_cost_frontier_report.json": {
            arm: {
                "exact_joint_accuracy": core[arm]["exact_joint_accuracy"],
                "support": core[arm]["average_total_support_used"],
                "counter_support": core[arm]["average_counter_support_used"],
                "external_test": core[arm]["average_external_test_used"],
                "cost_adjusted": cost_adjusted(core[arm]),
            }
            for arm in sorted(ARMS, key=lambda item: (core[item]["average_total_support_used"], -core[item]["exact_joint_accuracy"]))
        },
        "false_confidence_report.json": {
            arm: {
                "core_false_confidence": core[arm]["false_confidence_rate"],
                "indistinguishable_false_confidence": test["by_arm_and_regime"][arm]["INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT"]["false_confidence_rate"],
                "indistinguishable_abstain": test["by_arm_and_regime"][arm]["INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT"]["abstain_rate"],
            }
            for arm in ARMS
        },
        "regime_breakdown_report.json": test["by_arm_and_regime"],
        "controller_generalization_report.json": {
            "best_mutable_arm": best,
            "test_core": core[best],
            "test_regimes": test["by_arm_and_regime"][best],
            "ood_core": ood["by_arm_core"][best],
            "ood_regimes": ood["by_arm_and_regime"][best],
            "min_seed_exact": min_seed_metric(test, best, "exact_joint_accuracy"),
            "min_seed_correlated": min_seed_regime_accuracy(test, best, "CORRELATED_ECHO_SUPPORT"),
            "min_seed_adversarial": min_seed_regime_accuracy(test, best, "ADVERSARIAL_DISTRACTOR_SUPPORT"),
        },
        "controller_comparison_report.json": {
            arm: {
                "exact_joint_accuracy": core[arm]["exact_joint_accuracy"],
                "correlated_echo": regime_accuracy(test, arm, "CORRELATED_ECHO_SUPPORT"),
                "adversarial_distractor": regime_accuracy(test, arm, "ADVERSARIAL_DISTRACTOR_SUPPORT"),
                "external_test_required": regime_accuracy(test, arm, "EXTERNAL_TEST_REQUIRED_SUPPORT"),
                "indistinguishable_abstain": test["by_arm_and_regime"][arm]["INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT"]["abstain_rate"],
                "false_confidence": test["by_arm_and_regime"][arm]["INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT"]["false_confidence_rate"],
                "support": core[arm]["average_total_support_used"],
                "counter_support": core[arm]["average_counter_support_used"],
            }
            for arm in ARMS
        },
        "best_policy_report.json": {
            "best_mutable_arm": best,
            "policy": aggregate["policies"][best],
            "metrics": core[best],
            "action_distribution": test["action_distribution"][best],
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


def write_report(out, decision, aggregate):
    core = aggregate["test_metrics"]["by_arm_core"]
    regimes = aggregate["test_metrics"]["by_arm_and_regime"]
    best = decision.get("best_mutable_arm", best_mutable_arm(aggregate["test_metrics"]))
    lines = [
        "# D52 Mutable ECF Controller Scale Confirm Result",
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
        f"verdict = {decision['verdict']}",
        f"next = {decision['next']}",
        f"best_mutable_arm = {best}",
        "```",
        "",
        "Controller comparison:",
        "",
        "```text",
    ]
    for arm in ARMS:
        row = core[arm]
        lines.append(
            f"{arm}: exact={row['exact_joint_accuracy']:.4f}, corr={regimes[arm]['CORRELATED_ECHO_SUPPORT']['accuracy']:.4f}, adv={regimes[arm]['ADVERSARIAL_DISTRACTOR_SUPPORT']['accuracy']:.4f}, support={row['average_total_support_used']:.3f}, counter={row['average_counter_support_used']:.3f}"
        )
    lines.extend(["```", "", f"Boundary: {BOUNDARY}", ""])
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", default="10601,10602,10603,10604,10605,10606,10607,10608")
    parser.add_argument("--train-rows-per-seed", type=int, default=1200)
    parser.add_argument("--test-rows-per-seed", type=int, default=1200)
    parser.add_argument("--ood-rows-per-seed", type=int, default=1200)
    parser.add_argument("--generations", type=int, default=300)
    parser.add_argument("--population", type=int, default=96)
    parser.add_argument("--workers", default="auto")
    parser.add_argument("--cpu-target", default="50-75")
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    parser.add_argument("--scale-mode", default="full", choices=["full", "scale_lite"])
    return parser.parse_args()


def main():
    args = parse_args()
    started = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = [int(seed) for seed in args.seeds.split(",") if seed]
    bundle = d51.d49.make_bundle("ALL28_UNORDERED")
    repo_root = Path(__file__).resolve().parents[2]
    write_json(
        out / "queue.json",
        {
            "task": "D52 mutable ECF controller scale confirm",
            "status": "running",
            "scale_mode": args.scale_mode,
            "seeds": seeds,
            "train_rows_per_seed": args.train_rows_per_seed,
            "test_rows_per_seed": args.test_rows_per_seed,
            "ood_rows_per_seed": args.ood_rows_per_seed,
            "generations": args.generations,
            "population": args.population,
            "workers": args.workers,
            "cpu_target": args.cpu_target,
            "heartbeat_sec": args.heartbeat_sec,
            "no_black_box": True,
        },
    )
    append_progress(out, "queue_written", started, {"out": str(out), "scale_mode": args.scale_mode})
    upstream_manifest = make_d51_upstream_manifest(repo_root)
    write_json(out / "d51_upstream_manifest.json", upstream_manifest)
    write_json(
        out / "dataset_manifest.json",
        {
            "task": "D52_MUTABLE_ECF_CONTROLLER_SCALE_CONFIRM",
            "primary_space": PRIMARY_SPACE,
            "support_regimes": REGIMES,
            "actions": ACTIONS,
            "arms": ARMS,
            "feature_names": FEATURE_NAMES,
            "truth_hidden_from_controller_inputs": True,
            "controller_only_not_formula_solver": True,
            "failed_jobs_visible": True,
            "boundary": BOUNDARY,
        },
    )
    write_json(
        out / "compute_probe.json",
        {
            "mode": "local_python_cpu",
            "workers_requested": args.workers,
            "cpu_target": args.cpu_target,
            "note": "D52 uses deterministic symbolic scoring and mutable controller search; no external model/API/download used.",
        },
    )
    d51_replay_policy = load_best_d51_policy(repo_root, upstream_manifest)
    write_json(out / "best_d51_replay_manifest.json", {"loaded": d51_replay_policy is not None, "upstream": upstream_manifest})
    failed_jobs = [] if d51_replay_policy is not None else ["missing_best_d51_replay_policy"]

    train_rows = d51.make_rows_with_progress(seeds, args.train_rows_per_seed, "train", bundle, out, started, args.heartbeat_sec)
    test_rows = d51.make_rows_with_progress(seeds, args.test_rows_per_seed, "test", bundle, out, started, args.heartbeat_sec)
    ood_rows = d51.make_rows_with_progress(seeds, args.ood_rows_per_seed, "ood", bundle, out, started, args.heartbeat_sec)
    append_progress(out, "rows_generated", started, {"train": len(train_rows), "test": len(test_rows), "ood": len(ood_rows)})
    train_packs = d51.build_packs(train_rows, bundle, out, started, args.heartbeat_sec, args.workers, "train")
    train_examples = [
        {"features": pack["features"], "action_compact": pack["action_compact"]}
        for pack in train_packs
    ]
    policies = {"BEST_D51_REPLAY": d51_replay_policy} if d51_replay_policy is not None else {}
    policy_reports = {}
    arm_to_kind = {
        "MUTABLE_LINEAR_CONTROLLER": "linear",
        "MUTABLE_RULE_TABLE_CONTROLLER": "rule_table",
        "MUTABLE_SMALL_TREE_CONTROLLER": "small_tree",
        "MUTABLE_HYBRID_CONTROLLER": "hybrid",
    }
    for arm, kind in arm_to_kind.items():
        policy, report = d51.train_policy(kind, train_examples, args.generations, args.population, 52_000 + d51.stable_seed(arm), out, started, args.heartbeat_sec)
        policies[arm] = policy
        policy_reports[arm] = report
        append_progress(out, "mutation_complete", started, {"arm": arm, "fitness": report["fitness"]})
    if d51_replay_policy is not None:
        score, counts = d51.full_policy_score(d51_replay_policy, train_examples)
        policy_reports["BEST_D51_REPLAY"] = {"kind": "d51_replay", "fitness": score, "action_counts": counts, "history": []}
    write_json(out / "trained_policy_manifest.json", {"policies": policies, "reports": policy_reports})

    test_packs = d51.build_packs(test_rows, bundle, out, started, args.heartbeat_sec, args.workers, "test")
    ood_packs = d51.build_packs(ood_rows, bundle, out, started, args.heartbeat_sec, args.workers, "ood")
    test_outputs = evaluate_packs(test_packs, policies, out / "row_outputs_test.jsonl", out, "test", started, args.heartbeat_sec)
    ood_outputs = evaluate_packs(ood_packs, policies, out / "row_outputs_ood.jsonl", out, "ood", started, args.heartbeat_sec)
    test_metrics = summarize_outputs(test_outputs)
    ood_metrics = summarize_outputs(ood_outputs)
    decision = make_decision(test_metrics, failed_jobs)
    aggregate = {
        "task": "D52_MUTABLE_ECF_CONTROLLER_SCALE_CONFIRM",
        "scale_mode": args.scale_mode,
        "test_metrics": test_metrics,
        "ood_metrics": ood_metrics,
        "policies": policies,
        "policy_reports": policy_reports,
        "best_mutable_arm": decision.get("best_mutable_arm", best_mutable_arm(test_metrics)),
        "failed_jobs": failed_jobs,
        "decision": decision["decision"],
        "verdict": decision["verdict"],
        "next": decision["next"],
        "boundary": BOUNDARY,
    }
    reports = make_reports(out, aggregate, decision, upstream_manifest)
    write_json(out / "aggregate_metrics.json", aggregate)
    write_json(out / "decision.json", decision)
    write_json(
        out / "summary.json",
        {
            "scale_mode": args.scale_mode,
            "decision": decision,
            "best_mutable_arm": aggregate["best_mutable_arm"],
            "key_metrics": {
                "best_mutable": test_metrics["by_arm_core"][aggregate["best_mutable_arm"]],
                "best_by_regime": test_metrics["by_arm_and_regime"][aggregate["best_mutable_arm"]],
                "d50_full_reference": test_metrics["by_arm_core"]["D50_FULL_HANDCODED_REFERENCE"],
                "cap7_reference": test_metrics["by_arm_core"]["CAP_7_REFERENCE"],
                "cap9_reference": test_metrics["by_arm_core"]["CAP_9_REFERENCE"],
                "always_counter": test_metrics["by_arm_core"]["ALWAYS_COUNTER_CONTROL"],
                "best_d51_replay": test_metrics["by_arm_core"].get("BEST_D51_REPLAY"),
                "support_cost_frontier": reports["support_cost_frontier_report.json"],
                "action_distribution": test_metrics["action_distribution"][aggregate["best_mutable_arm"]],
                "min_seed_gates": reports["min_seed_gate_report.json"][aggregate["best_mutable_arm"]],
            },
            "boundary": BOUNDARY,
        },
    )
    write_report(out, decision, aggregate)
    queue = json.loads((out / "queue.json").read_text(encoding="utf-8"))
    write_json(out / "queue.json", {**queue, "status": "complete"})
    append_progress(out, "final_decision", started, decision)
    print(
        json.dumps(
            {
                "decision": decision["decision"],
                "verdict": decision["verdict"],
                "next": decision["next"],
                "best": aggregate["best_mutable_arm"],
                "scale_mode": args.scale_mode,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
