#!/usr/bin/env python3
"""D54 VRAXION mutable ECF controller scale confirm.

This probe scale-confirms the D53 VRAXION-style mutable controller genome
integration. It keeps the D50-D53 controlled symbolic joint formula task fixed.
It does not train or claim a full sparse firing VRAXION brain.
"""

import argparse
import copy
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import run_d51_mutable_ecf_controller_prototype as d51
import run_d53_mutable_ecf_integration_with_vraxion_mutation_architecture as d53

PRIMARY_SPACE = d53.PRIMARY_SPACE
SUPPORT_COUNT = d53.SUPPORT_COUNT
REGIMES = d53.REGIMES
CORE_REGIMES = d53.CORE_REGIMES
ACTIONS = d53.ACTIONS
FEATURE_NAMES = d53.FEATURE_NAMES
ROW_SAMPLE_PER_ARM_REGIME_SPLIT = d53.ROW_SAMPLE_PER_ARM_REGIME_SPLIT

BOUNDARY = (
    "D54 only scale-confirms VRAXION-style mutable ECF controller integration "
    "for controlled symbolic joint formula discovery. It does not prove full "
    "VRAXION sparse firing brain learning, raw visual Raven solving, Raven "
    "solved, AGI, consciousness, DNA/genome success, architecture superiority, "
    "or production readiness."
)

REFERENCE_ARMS = [
    "D53_BEST_HYBRID_REPLAY",
    "D52_RULE_TABLE_REPLAY",
    "D50_HANDCODED_FULL_REFERENCE",
]

CONTROL_ARMS = [
    "RANDOM_POLICY_CONTROL",
    "GREEDY_DECIDE_CONTROL",
    "ALWAYS_COUNTER_CONTROL",
    "COST_ONLY_MUTATION_CONTROL",
]

VRAXION_ARMS = [
    "VRAXION_MUTABLE_RULE_TABLE",
    "VRAXION_MUTABLE_SPARSE_GATE_CONTROLLER",
    "VRAXION_MUTABLE_POCKET_STATE_CONTROLLER",
    "VRAXION_MUTABLE_HYBRID_CONTROLLER",
]

ABLATION_ARMS = [
    "SPARSE_GATE_ABLATION",
    "POCKET_STATE_ABLATION",
    "MUTATION_DISABLED_CONTROL",
]

ARMS = REFERENCE_ARMS + CONTROL_ARMS + VRAXION_ARMS + ABLATION_ARMS


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


def cost_adjusted(row):
    return d53.cost_adjusted(row)


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


def make_d53_upstream_manifest(repo_root):
    root = repo_root / "target/pilot_wave/d53_mutable_ecf_integration_with_vraxion_mutation_architecture/smoke"
    manifest = {
        "upstream": "D53_MUTABLE_ECF_INTEGRATION_WITH_VRAXION_MUTATION_ARCHITECTURE",
        "expected_decision": "vraxion_mutable_ecf_controller_integration_positive",
        "expected_next": "D54_VRAXION_MUTABLE_ECF_CONTROLLER_SCALE_CONFIRM",
        "decision_present": (root / "decision.json").exists(),
        "summary_present": (root / "summary.json").exists(),
        "trained_policy_manifest_present": (root / "trained_policy_manifest.json").exists(),
    }
    decision = load_json_if_present(root / "decision.json")
    summary = load_json_if_present(root / "summary.json")
    controller = load_json_if_present(root / "controller_comparison_report.json")
    if decision:
        manifest["decision_json"] = decision
    if summary:
        manifest["scale_mode"] = summary.get("scale_mode")
        manifest["best_vraxion_arm"] = summary.get("best_vraxion_arm")
    if controller and "VRAXION_MUTABLE_HYBRID_CONTROLLER" in controller:
        manifest["d53_best_hybrid_metrics"] = controller["VRAXION_MUTABLE_HYBRID_CONTROLLER"]
    return manifest


def load_d53_best_hybrid_policy(repo_root, manifest):
    root = repo_root / "target/pilot_wave/d53_mutable_ecf_integration_with_vraxion_mutation_architecture/smoke"
    trained = load_json_if_present(root / "trained_policy_manifest.json")
    if not trained:
        manifest["best_d53_hybrid_loaded"] = False
        return None
    policy = trained.get("policies", {}).get("VRAXION_MUTABLE_HYBRID_CONTROLLER")
    manifest["best_d53_hybrid_loaded"] = policy is not None
    return policy


def load_d52_rule_table_policy(repo_root, manifest):
    d52_manifest = {"expected_decision": "mutable_ecf_controller_scale_confirmed"}
    policy = d53.load_best_d52_policy(repo_root, d52_manifest)
    manifest["d52_rule_table_loaded"] = policy is not None
    manifest["d52_manifest"] = d52_manifest
    return policy


def d54_canonical_audit(repo_root):
    base = d53.canonical_vraxion_audit(repo_root)
    base["sparse_firing_used_in_d54"] = False
    base["d54_scale_confirm"] = True
    base["notes"] = [
        "D54 audits canonical Rust VRAXION surfaces but keeps inference in the fixed D50-D53 symbolic runner.",
        "Sparse-gate means controller feature gates, not full sparse firing brain training.",
    ]
    return base


def make_sparse_gate_ablation(examples):
    rng = random.Random(54_001)
    policy = d53.make_vraxion_policy("sparse_gate", rng, examples)
    for gate in policy["gates"]:
        if gate["action"] in {"REQUEST_COUNTER_TOP1_TOP2", "REQUEST_JOINT_COUNTER", "REQUEST_SUPPORT"}:
            gate["threshold"] = 17
            gate["weight"] = 0
    policy["default_action"] = "DECIDE"
    policy["ablation"] = "counter_support_gates_disabled"
    return policy


def make_pocket_state_ablation(examples):
    rng = random.Random(54_002)
    policy = d53.make_vraxion_policy("pocket_state", rng, examples)
    for pocket in policy["pockets"]:
        if pocket["action"] in {"REQUEST_COUNTER_TOP1_TOP2", "REQUEST_JOINT_COUNTER", "REQUEST_SUPPORT"}:
            pocket["threshold"] = 17
    policy["default_action"] = "DECIDE"
    policy["ablation"] = "counter_support_pockets_disabled"
    return policy


def make_mutation_disabled_control():
    rng = random.Random(54_003)
    policy = d53.make_vraxion_policy("hybrid", rng, examples=None)
    policy["ablation"] = "bootstrap_defaults_no_training_mutation"
    return policy


def cost_only_action(pack):
    return d53.cost_only_action(pack)


def output_from_action(pack, arm, action):
    return d53.output_from_action(pack, arm, action)


def renamed_reference(pack, arm):
    result = copy.deepcopy(pack["references"]["HANDCODED_D50_FULL_REPAIRED_ECF_REFERENCE"])
    result["arm"] = arm
    return result


def evaluate_pack_all_arms(pack, policies):
    rows = []
    d53_policy = policies.get("D53_BEST_HYBRID_REPLAY")
    if d53_policy is not None:
        rows.append(output_from_action(pack, "D53_BEST_HYBRID_REPLAY", d53.choose_vraxion_action(d53_policy, pack["features"])))
    d52_policy = policies.get("D52_RULE_TABLE_REPLAY")
    if d52_policy is not None:
        rows.append(output_from_action(pack, "D52_RULE_TABLE_REPLAY", d51.choose_action(d52_policy, pack["features"])))
    rows.append(renamed_reference(pack, "D50_HANDCODED_FULL_REFERENCE"))
    rows.append(output_from_action(pack, "RANDOM_POLICY_CONTROL", d51.random_policy_action(f"D54:{pack['row_id']}")))
    rows.append(output_from_action(pack, "GREEDY_DECIDE_CONTROL", "DECIDE"))
    rows.append(output_from_action(pack, "ALWAYS_COUNTER_CONTROL", "REQUEST_JOINT_COUNTER"))
    rows.append(output_from_action(pack, "COST_ONLY_MUTATION_CONTROL", cost_only_action(pack)))
    for arm in VRAXION_ARMS + ABLATION_ARMS:
        policy = policies.get(arm)
        if policy is not None:
            rows.append(output_from_action(pack, arm, d53.choose_vraxion_action(policy, pack["features"])))
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
    return {
        "by_arm": {arm: summarize(by_arm[arm]) for arm in ARMS if arm in by_arm},
        "by_arm_core": {arm: summarize(by_arm_core[arm]) for arm in ARMS if arm in by_arm_core},
        "by_arm_and_regime": {
            arm: {regime: summarize(by_arm_regime[(arm, regime)]) for regime in REGIMES if (arm, regime) in by_arm_regime}
            for arm in ARMS
        },
        "action_distribution": {arm: {action: by_action[arm][action] for action in ACTIONS} for arm in ARMS},
        "error_taxonomy": {
            arm: {regime: dict(by_error[(arm, regime)]) for regime in REGIMES if (arm, regime) in by_error}
            for arm in ARMS
        },
        "by_seed_core": {
            arm: {str(seed): summarize(rows) for (a, seed), rows in by_seed_core.items() if a == arm}
            for arm in ARMS
        },
        "by_seed_regime": {
            arm: {
                str(seed): {
                    regime: summarize(by_seed_regime[(arm, seed, regime)])
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


def best_vraxion_arm(metrics):
    core = metrics["by_arm_core"]
    return max(VRAXION_ARMS, key=lambda arm: (cost_adjusted(core[arm]), core[arm]["exact_joint_accuracy"]))


def write_partial_eval(out, split, outputs, completed, started):
    partial = summarize_outputs(outputs)
    best = best_vraxion_arm(partial) if all(arm in partial["by_arm_core"] for arm in VRAXION_ARMS) else None
    write_json(
        out / f"partial_{split}_metrics_snapshot.json",
        {
            "split": split,
            "completed_outputs": completed,
            "elapsed_sec": time.time() - started,
            "by_arm_core": partial["by_arm_core"],
            "best_vraxion_arm_so_far": best,
            "best_vraxion_regime_so_far": partial["by_arm_and_regime"].get(best, {}) if best else {},
        },
    )
    append_progress(out, "eval_progress", started, {"split": split, "completed_outputs": completed, "best_vraxion_so_far": best})


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


def cost_only_passes_safety(metrics):
    arm = "COST_ONLY_MUTATION_CONTROL"
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


def sparse_gate_and_pocket_pass(metrics):
    for arm in ["VRAXION_MUTABLE_SPARSE_GATE_CONTROLLER", "VRAXION_MUTABLE_POCKET_STATE_CONTROLLER"]:
        row = metrics["by_arm_core"][arm]
        if row["exact_joint_accuracy"] < 0.995:
            return False
        if regime_accuracy(metrics, arm, "CORRELATED_ECHO_SUPPORT") < 0.995:
            return False
        if regime_accuracy(metrics, arm, "ADVERSARIAL_DISTRACTOR_SUPPORT") < 0.995:
            return False
        if metrics["by_arm_and_regime"][arm]["INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT"]["abstain_rate"] < 0.99:
            return False
    return True


def make_decision(metrics, failed_jobs):
    if failed_jobs:
        return {
            "decision": "vraxion_mutable_ecf_controller_scale_not_confirmed",
            "verdict": "D54_FAILED_JOBS_PRESENT",
            "next": "D54_REPAIR",
            "boundary": BOUNDARY,
        }
    best = best_vraxion_arm(metrics)
    core = metrics["by_arm_core"]
    best_row = core[best]
    corr = regime_accuracy(metrics, best, "CORRELATED_ECHO_SUPPORT")
    adv = regime_accuracy(metrics, best, "ADVERSARIAL_DISTRACTOR_SUPPORT")
    external = regime_accuracy(metrics, best, "EXTERNAL_TEST_REQUIRED_SUPPORT")
    indist = metrics["by_arm_and_regime"][best]["INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT"]
    full = core["D50_HANDCODED_FULL_REFERENCE"]
    random_control = core["RANDOM_POLICY_CONTROL"]
    greedy_control = core["GREEDY_DECIDE_CONTROL"]
    cost_control = core["COST_ONLY_MUTATION_CONTROL"]
    always = core["ALWAYS_COUNTER_CONTROL"]
    if cost_only_passes_safety(metrics):
        return {
            "decision": "vraxion_mutable_ecf_controller_scale_not_confirmed",
            "verdict": "D54_COST_ONLY_CONTROL_PASSED",
            "next": "D54_REPAIR",
            "best_vraxion_arm": best,
            "boundary": BOUNDARY,
        }
    pass_accuracy = (
        best_row["exact_joint_accuracy"] >= 0.995
        and corr >= 0.995
        and adv >= 0.995
        and external >= 0.99
        and indist["false_confidence_rate"] <= 0.01
        and indist["abstain_rate"] >= 0.99
        and min_seed_metric(metrics, best, "exact_joint_accuracy") >= 0.99
        and min_seed_regime_accuracy(metrics, best, "CORRELATED_ECHO_SUPPORT") >= 0.99
        and min_seed_regime_accuracy(metrics, best, "ADVERSARIAL_DISTRACTOR_SUPPORT") >= 0.99
    )
    controls_worse = (
        random_control["exact_joint_accuracy"] < best_row["exact_joint_accuracy"]
        and greedy_control["exact_joint_accuracy"] < best_row["exact_joint_accuracy"]
        and cost_control["exact_joint_accuracy"] < best_row["exact_joint_accuracy"]
        and always["average_total_support_used"] > best_row["average_total_support_used"]
    )
    pass_cost = best_row["average_total_support_used"] <= full["average_total_support_used"]
    if pass_accuracy and controls_worse and pass_cost:
        if sparse_gate_and_pocket_pass(metrics):
            return {
                "decision": "vraxion_sparse_gate_controller_path_confirmed",
                "verdict": "D54_VRAXION_SPARSE_GATE_CONTROLLER_PATH_CONFIRMED",
                "next": "D55_SPARSE_FIRING_ECF_CONTROLLER_PROTOTYPE",
                "best_vraxion_arm": best,
                "boundary": BOUNDARY,
            }
        if best in {"VRAXION_MUTABLE_RULE_TABLE", "VRAXION_MUTABLE_HYBRID_CONTROLLER"}:
            return {
                "decision": "vraxion_mutable_controller_scale_confirmed_non_sparse",
                "verdict": "D54_VRAXION_MUTABLE_CONTROLLER_SCALE_CONFIRMED_NON_SPARSE",
                "next": "D55_SPARSE_FIRING_ECF_CONTROLLER_PROTOTYPE",
                "best_vraxion_arm": best,
                "boundary": BOUNDARY,
            }
        return {
            "decision": "vraxion_mutable_ecf_controller_scale_confirmed",
            "verdict": "D54_VRAXION_MUTABLE_ECF_CONTROLLER_SCALE_CONFIRMED",
            "next": "D55_SPARSE_FIRING_ECF_CONTROLLER_PROTOTYPE",
            "best_vraxion_arm": best,
            "boundary": BOUNDARY,
        }
    return {
        "decision": "vraxion_mutable_ecf_controller_scale_not_confirmed",
        "verdict": "D54_VRAXION_MUTABLE_ECF_CONTROLLER_SCALE_NOT_CONFIRMED",
        "next": "D54_REPAIR",
        "best_vraxion_arm": best,
        "boundary": BOUNDARY,
    }


def make_reports(out, aggregate, decision, d53_manifest, canonical_report):
    test = aggregate["test_metrics"]
    core = test["by_arm_core"]
    best = decision.get("best_vraxion_arm", best_vraxion_arm(test))
    reports = {
        "representation_report.json": {
            "canonical_vraxion_audit": canonical_report,
            "d54_representation_level": "scale_confirmed_mutable_controller_genome_above_fixed_symbolic_ecf",
            "full_sparse_firing_brain_used": False,
            "formula_solver_learning_used": False,
            "representations": {
                "VRAXION_MUTABLE_RULE_TABLE": "threshold/action-route genome",
                "VRAXION_MUTABLE_SPARSE_GATE_CONTROLLER": "integer sparse feature gates to action scores",
                "VRAXION_MUTABLE_POCKET_STATE_CONTROLLER": "feature pockets with priority and action writeback",
                "VRAXION_MUTABLE_HYBRID_CONTROLLER": "rule-table plus sparse-gate overlay",
            },
        },
        "sparse_firing_usage_report.json": {
            "sparse_firing_used_in_d54": False,
            "full_sparse_firing_brain_trained": False,
            "sparse_gate_controller_used": True,
            "sparse_gate_controller_is_not_sparse_firing_brain": True,
            "next_sparse_firing_step": "D55_SPARSE_FIRING_ECF_CONTROLLER_PROTOTYPE",
        },
        "mutation_acceptance_report.json": aggregate["policy_reports"],
        "fitness_landscape_report.json": {
            arm: {
                "fitness": aggregate["policy_reports"][arm]["fitness"],
                "mutation_counts": aggregate["policy_reports"][arm].get("mutation_counts", {}),
                "accepted_mutation_counts": aggregate["policy_reports"][arm].get("accepted_mutation_counts", {}),
                "history": aggregate["policy_reports"][arm].get("history", []),
            }
            for arm in VRAXION_ARMS
        },
        "action_distribution_report.json": test["action_distribution"],
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
        "component_ablation_report.json": {
            arm: {
                "exact_joint_accuracy": core[arm]["exact_joint_accuracy"],
                "correlated_echo": regime_accuracy(test, arm, "CORRELATED_ECHO_SUPPORT"),
                "adversarial_distractor": regime_accuracy(test, arm, "ADVERSARIAL_DISTRACTOR_SUPPORT"),
                "support": core[arm]["average_total_support_used"],
                "counter_support": core[arm]["average_counter_support_used"],
            }
            for arm in ABLATION_ARMS + VRAXION_ARMS
        },
        "regime_breakdown_report.json": test["by_arm_and_regime"],
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
    best = decision.get("best_vraxion_arm", best_vraxion_arm(aggregate["test_metrics"]))
    lines = [
        "# D54 VRAXION Mutable ECF Controller Scale Confirm Result",
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
        f"best_vraxion_arm = {best}",
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
    parser.add_argument("--seeds", default="10801,10802,10803,10804,10805,10806,10807,10808")
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
            "task": "D54 VRAXION mutable ECF controller scale confirm",
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
    d53_manifest = make_d53_upstream_manifest(repo_root)
    canonical_report = d54_canonical_audit(repo_root)
    write_json(out / "canonical_vraxion_audit_report.json", canonical_report)
    write_json(
        out / "dataset_manifest.json",
        {
            "task": "D54_VRAXION_MUTABLE_ECF_CONTROLLER_SCALE_CONFIRM",
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
            "note": "D54 uses deterministic symbolic scoring and mutable controller genome search; no external model/API/download used.",
        },
    )
    d53_policy = load_d53_best_hybrid_policy(repo_root, d53_manifest)
    d52_policy = load_d52_rule_table_policy(repo_root, d53_manifest)
    write_json(out / "d53_upstream_manifest.json", d53_manifest)
    failed_jobs = []
    if d53_policy is None:
        failed_jobs.append("missing_best_d53_hybrid_policy")
    if d52_policy is None:
        failed_jobs.append("missing_d52_rule_table_policy")
    if not canonical_report["source_smoke_passed"] or not canonical_report["action_output_encoding_smoke_passed"]:
        failed_jobs.append("canonical_vraxion_source_smoke_failed")

    train_rows = d51.make_rows_with_progress(seeds, args.train_rows_per_seed, "train", bundle, out, started, args.heartbeat_sec)
    test_rows = d51.make_rows_with_progress(seeds, args.test_rows_per_seed, "test", bundle, out, started, args.heartbeat_sec)
    ood_rows = d51.make_rows_with_progress(seeds, args.ood_rows_per_seed, "ood", bundle, out, started, args.heartbeat_sec)
    append_progress(out, "rows_generated", started, {"train": len(train_rows), "test": len(test_rows), "ood": len(ood_rows)})
    train_packs = d51.build_packs(train_rows, bundle, out, started, args.heartbeat_sec, args.workers, "train")
    train_examples = [{"features": pack["features"], "action_compact": pack["action_compact"]} for pack in train_packs]

    policies = {}
    if d53_policy is not None:
        policies["D53_BEST_HYBRID_REPLAY"] = d53_policy
    if d52_policy is not None:
        policies["D52_RULE_TABLE_REPLAY"] = d52_policy
    policy_reports = {}
    arm_to_kind = {
        "VRAXION_MUTABLE_RULE_TABLE": "rule_table",
        "VRAXION_MUTABLE_SPARSE_GATE_CONTROLLER": "sparse_gate",
        "VRAXION_MUTABLE_POCKET_STATE_CONTROLLER": "pocket_state",
        "VRAXION_MUTABLE_HYBRID_CONTROLLER": "hybrid",
    }
    for arm, kind in arm_to_kind.items():
        policy, report = d53.train_vraxion_policy(kind, train_examples, args.generations, args.population, 54_000 + d51.stable_seed(arm), out, started, args.heartbeat_sec)
        policies[arm] = policy
        policy_reports[arm] = report
        append_progress(out, "vraxion_mutation_complete", started, {"arm": arm, "fitness": report["fitness"]})
    policies["SPARSE_GATE_ABLATION"] = make_sparse_gate_ablation(train_examples)
    policies["POCKET_STATE_ABLATION"] = make_pocket_state_ablation(train_examples)
    policies["MUTATION_DISABLED_CONTROL"] = make_mutation_disabled_control()
    for arm in ABLATION_ARMS:
        score, counts = d53.full_policy_score(policies[arm], train_examples)
        policy_reports[arm] = {"kind": policies[arm]["kind"], "fitness": score, "action_counts": counts, "mutation_counts": {}, "accepted_mutation_counts": {}, "history": []}
    if d53_policy is not None:
        score, counts = d53.full_policy_score(d53_policy, train_examples)
        policy_reports["D53_BEST_HYBRID_REPLAY"] = {"kind": "d53_best_hybrid_replay", "fitness": score, "action_counts": counts, "history": []}
    if d52_policy is not None:
        score, counts = d51.full_policy_score(d52_policy, train_examples)
        policy_reports["D52_RULE_TABLE_REPLAY"] = {"kind": "d52_rule_table_replay", "fitness": score, "action_counts": counts, "history": []}
    write_json(out / "trained_policy_manifest.json", {"policies": policies, "reports": policy_reports})

    test_packs = d51.build_packs(test_rows, bundle, out, started, args.heartbeat_sec, args.workers, "test")
    ood_packs = d51.build_packs(ood_rows, bundle, out, started, args.heartbeat_sec, args.workers, "ood")
    test_outputs = evaluate_packs(test_packs, policies, out / "row_outputs_test.jsonl", out, "test", started, args.heartbeat_sec)
    ood_outputs = evaluate_packs(ood_packs, policies, out / "row_outputs_ood.jsonl", out, "ood", started, args.heartbeat_sec)
    test_metrics = summarize_outputs(test_outputs)
    ood_metrics = summarize_outputs(ood_outputs)
    decision = make_decision(test_metrics, failed_jobs)
    aggregate = {
        "task": "D54_VRAXION_MUTABLE_ECF_CONTROLLER_SCALE_CONFIRM",
        "scale_mode": args.scale_mode,
        "test_metrics": test_metrics,
        "ood_metrics": ood_metrics,
        "policies": policies,
        "policy_reports": policy_reports,
        "best_vraxion_arm": decision.get("best_vraxion_arm", best_vraxion_arm(test_metrics)),
        "failed_jobs": failed_jobs,
        "decision": decision["decision"],
        "verdict": decision["verdict"],
        "next": decision["next"],
        "boundary": BOUNDARY,
    }
    reports = make_reports(out, aggregate, decision, d53_manifest, canonical_report)
    write_json(out / "aggregate_metrics.json", aggregate)
    write_json(out / "decision.json", decision)
    write_json(
        out / "summary.json",
        {
            "scale_mode": args.scale_mode,
            "decision": decision,
            "best_vraxion_arm": aggregate["best_vraxion_arm"],
            "key_metrics": {
                "best_vraxion": test_metrics["by_arm_core"][aggregate["best_vraxion_arm"]],
                "best_by_regime": test_metrics["by_arm_and_regime"][aggregate["best_vraxion_arm"]],
                "d53_replay": test_metrics["by_arm_core"].get("D53_BEST_HYBRID_REPLAY"),
                "d52_replay": test_metrics["by_arm_core"].get("D52_RULE_TABLE_REPLAY"),
                "d50_full_reference": test_metrics["by_arm_core"]["D50_HANDCODED_FULL_REFERENCE"],
                "support_cost_frontier": reports["support_cost_frontier_report.json"],
                "component_ablation": reports["component_ablation_report.json"],
                "action_distribution": test_metrics["action_distribution"][aggregate["best_vraxion_arm"]],
                "min_seed_gates": reports["min_seed_gate_report.json"][aggregate["best_vraxion_arm"]],
                "sparse_firing_usage": reports["sparse_firing_usage_report.json"],
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
                "best": aggregate["best_vraxion_arm"],
                "scale_mode": args.scale_mode,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
