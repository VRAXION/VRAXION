#!/usr/bin/env python3
"""D55 sparse firing ECF controller prototype.

This probe keeps the D50-D54 controlled symbolic joint formula task fixed and
replaces only the ECF action controller with a small sparse-firing controller.
It does not train or claim a full sparse firing formula-solving brain.
"""

import argparse
import copy
import json
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import run_d51_mutable_ecf_controller_prototype as d51
import run_d53_mutable_ecf_integration_with_vraxion_mutation_architecture as d53
import run_d54_vraxion_mutable_ecf_controller_scale_confirm as d54

PRIMARY_SPACE = d54.PRIMARY_SPACE
SUPPORT_COUNT = d54.SUPPORT_COUNT
REGIMES = d54.REGIMES
CORE_REGIMES = d54.CORE_REGIMES
ACTIONS = d54.ACTIONS
FEATURE_NAMES = d54.FEATURE_NAMES
ROW_SAMPLE_PER_ARM_REGIME_SPLIT = d54.ROW_SAMPLE_PER_ARM_REGIME_SPLIT

BOUNDARY = (
    "D55 only tests a controller-local sparse firing ECF action policy for controlled symbolic joint "
    "formula discovery. It does not prove full VRAXION sparse firing brain learning, raw visual Raven "
    "reasoning, Raven solved, AGI, consciousness, DNA/genome success, architecture superiority, "
    "or production readiness."
)

REFERENCE_ARMS = [
    "D54_BEST_HYBRID_REPLAY",
    "D54_SPARSE_GATE_REPLAY",
    "HANDCODED_D50_FULL_REFERENCE",
]

CONTROL_ARMS = [
    "RANDOM_POLICY_CONTROL",
    "GREEDY_DECIDE_CONTROL",
    "ALWAYS_COUNTER_CONTROL",
    "MUTABLE_RULE_TABLE_REFERENCE",
]

SPARSE_ARMS = [
    "REAL_SPARSE_FIRING_CONTROLLER_SMALL",
    "REAL_SPARSE_FIRING_CONTROLLER_MEDIUM",
    "REAL_SPARSE_FIRING_CONTROLLER_WITH_MUTATION",
    "REAL_SPARSE_FIRING_CONTROLLER_NO_MUTATION",
]

ABLATION_ARMS = [
    "SPIKE_SHUFFLE_CONTROL",
    "FIRING_THRESHOLD_ABLATION",
    "CONNECTION_REWIRE_ABLATION",
]

ARMS = REFERENCE_ARMS + CONTROL_ARMS + SPARSE_ARMS + ABLATION_ARMS
PHASE_BASE = [7, 8, 10, 12, 13, 12, 10, 8]
GATE_CHANNEL = 7


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


def mean(values):
    return d51.mean(values)


def stable_seed(text):
    return d51.stable_seed(text)


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


def make_d54_upstream_manifest(repo_root):
    root = repo_root / "target/pilot_wave/d54_vraxion_mutable_ecf_controller_scale_confirm/smoke"
    manifest = {
        "upstream": "D54_VRAXION_MUTABLE_ECF_CONTROLLER_SCALE_CONFIRM",
        "expected_decision": "vraxion_sparse_gate_controller_path_confirmed",
        "expected_next": "D55_SPARSE_FIRING_ECF_CONTROLLER_PROTOTYPE",
        "artifact_root": str(root),
        "decision_present": (root / "decision.json").exists(),
        "summary_present": (root / "summary.json").exists(),
        "trained_policy_manifest_present": (root / "trained_policy_manifest.json").exists(),
        "aggregate_metrics_present": (root / "aggregate_metrics.json").exists(),
    }
    decision = load_json_if_present(root / "decision.json")
    summary = load_json_if_present(root / "summary.json")
    aggregate = load_json_if_present(root / "aggregate_metrics.json")
    trained = load_json_if_present(root / "trained_policy_manifest.json")
    if decision:
        manifest["decision_json"] = decision
    if summary:
        manifest["summary_json"] = summary
    if aggregate:
        best = aggregate.get("best_vraxion_arm")
        manifest["best_vraxion_arm"] = best
        test = aggregate.get("test_metrics", {})
        by_arm = test.get("by_arm_core", {})
        by_regime = test.get("by_arm_and_regime", {})
        if best in by_arm:
            manifest["best_metrics"] = {
                "exact_joint_accuracy": by_arm[best]["exact_joint_accuracy"],
                "correlated_echo": by_regime[best]["CORRELATED_ECHO_SUPPORT"]["accuracy"],
                "adversarial_distractor": by_regime[best]["ADVERSARIAL_DISTRACTOR_SUPPORT"]["accuracy"],
                "external_test_required": by_regime[best]["EXTERNAL_TEST_REQUIRED_SUPPORT"]["accuracy"],
                "support": by_arm[best]["average_total_support_used"],
            }
    if trained:
        manifest["loaded_policy_names"] = sorted(trained.get("policies", {}).keys())
    return manifest


def load_d54_policies(repo_root, manifest):
    trained = load_json_if_present(
        repo_root / "target/pilot_wave/d54_vraxion_mutable_ecf_controller_scale_confirm/smoke/trained_policy_manifest.json"
    )
    if not trained:
        manifest["d54_policy_load_error"] = "trained_policy_manifest_missing"
        return {}
    policies = trained.get("policies", {})
    required = [
        "VRAXION_MUTABLE_HYBRID_CONTROLLER",
        "VRAXION_MUTABLE_SPARSE_GATE_CONTROLLER",
        "VRAXION_MUTABLE_RULE_TABLE",
    ]
    manifest["required_d54_policies_loaded"] = {name: name in policies for name in required}
    return {name: policies[name] for name in required if name in policies}


def canonical_sparse_firing_audit(repo_root):
    network_path = repo_root / "instnct-core/src/network.rs"
    evolution_path = repo_root / "instnct-core/src/evolution.rs"
    network = network_path.read_text(encoding="utf-8") if network_path.exists() else ""
    evolution = evolution_path.read_text(encoding="utf-8") if evolution_path.exists() else ""
    required_network = [
        "pub struct SpikeData",
        "pub fn propagate_sparse",
        "threshold",
        "charge",
        "channel",
        "phase_base",
        "use_refractory",
    ]
    required_mutations = [
        "add_edge",
        "remove_edge",
        "rewire",
        "reverse",
        "mirror",
        "enhance",
        "theta",
        "channel",
        "loop2",
        "loop3",
        "projection_weight",
    ]
    return {
        "network_path": str(network_path),
        "evolution_path": str(evolution_path),
        "network_present": network_path.exists(),
        "evolution_present": evolution_path.exists(),
        "network_surface_hits": {needle: needle in network for needle in required_network},
        "mutation_surface_hits": {needle: needle in evolution for needle in required_mutations},
        "canonical_rust_network_audited": network_path.exists() and all(needle in network for needle in required_network),
        "canonical_mutation_surface_audited": evolution_path.exists() and all(needle in evolution for needle in required_mutations),
        "rust_network_binary_invoked": False,
        "note": "D55 uses a controller-local sparse firing implementation matched to the audited charge/threshold/channel sparse tick semantics.",
    }


def stored_threshold(gate_threshold):
    return max(0, min(15, int(gate_threshold) - 1))


def make_sparse_controller_from_gate(policy, name, hidden_neurons=0):
    gates = []
    for idx, gate in enumerate(policy["gates"]):
        gates.append(
            {
                "gate_id": idx,
                "feature": gate["feature"],
                "action": gate["action"],
                "threshold": int(gate["threshold"]),
                "stored_threshold": stored_threshold(gate["threshold"]),
                "weight": int(gate["weight"]),
                "priority": int(gate["priority"]),
                "channel": GATE_CHANNEL,
                "polarity": 1 if int(gate["weight"]) >= 0 else -1,
            }
        )
    return {
        "kind": "controller_local_sparse_firing",
        "name": name,
        "default_action": policy.get("default_action", "DECIDE"),
        "feature_names": FEATURE_NAMES,
        "actions": ACTIONS,
        "input_neurons": len(FEATURE_NAMES),
        "gate_neurons": len(gates),
        "hidden_neurons": hidden_neurons,
        "output_neurons": len(ACTIONS),
        "phase_base": PHASE_BASE,
        "gates": gates,
        "readout": "priority_spike_then_output_charge_argmax",
        "controller_only_not_formula_solver": True,
        "mutation_history": [],
    }


def controller_edge_count(controller):
    return len(controller["gates"]) * 2 + controller.get("hidden_neurons", 0)


def phase_multiplier(channel, tick=0):
    if 1 <= channel <= 8:
        return PHASE_BASE[(tick + 9 - channel) & 7]
    return 10


def quant_feature(features, feature_name):
    return d53.quant_feature(features, feature_name)


def sparse_fire_gate(gate, features):
    value = quant_feature(features, gate["feature"])
    if gate["threshold"] > 16:
        return False, value, 0
    threshold_x10 = (gate["stored_threshold"] + 1) * phase_multiplier(gate["channel"])
    charge_x10 = value * 10
    fired = charge_x10 >= threshold_x10
    return fired, value, threshold_x10


def choose_sparse_action(controller, features, stats=None, action_shuffle=None):
    output_charge = {action: 0 for action in ACTIONS}
    fired_gates = []
    total_charge = 0
    spike_updates = 0
    for gate in controller["gates"]:
        spike_updates += 1
        fired, value, threshold_x10 = sparse_fire_gate(gate, features)
        total_charge += value
        if fired:
            signed_weight = gate["weight"] * gate.get("polarity", 1)
            output_charge[gate["action"]] += signed_weight
            fired_gates.append(
                {
                    "gate_id": gate["gate_id"],
                    "feature": gate["feature"],
                    "action": gate["action"],
                    "priority": gate["priority"],
                    "weight": gate["weight"],
                    "charge": value,
                    "threshold_x10": threshold_x10,
                }
            )
    if not fired_gates:
        action = controller["default_action"]
    else:
        priority_gate = max(fired_gates, key=lambda item: (item["priority"], item["weight"], -item["gate_id"]))
        if priority_gate["priority"] >= 80:
            action = priority_gate["action"]
        else:
            action = max(ACTIONS, key=lambda item: (output_charge[item], -ACTIONS.index(item)))
    if action_shuffle:
        action = action_shuffle[action]
    if stats is not None:
        stats["calls"] += 1
        stats["spike_update_executed_count"] += spike_updates
        stats["fired_gate_count"] += len(fired_gates)
        stats["total_input_charge"] += total_charge
        stats["output_charge_sum"] += sum(abs(value) for value in output_charge.values())
        stats["action_counts"][action] += 1
        for gate in fired_gates:
            stats["fired_gate_by_feature"][gate["feature"]] += 1
            stats["fired_gate_by_action"][gate["action"]] += 1
    return action, {
        "fired_gates": fired_gates,
        "output_charge": output_charge,
        "spike_updates": spike_updates,
        "total_input_charge": total_charge,
    }


def make_threshold_ablation(controller):
    out = copy.deepcopy(controller)
    out["name"] = "FIRING_THRESHOLD_ABLATION"
    out["ablation"] = "counter_support_gate_thresholds_disabled"
    for gate in out["gates"]:
        if gate["action"] in {"REQUEST_COUNTER_TOP1_TOP2", "REQUEST_JOINT_COUNTER", "REQUEST_SUPPORT"}:
            gate["threshold"] = 17
            gate["stored_threshold"] = 15
            gate["weight"] = 0
    return out


def make_rewire_ablation(controller):
    out = copy.deepcopy(controller)
    out["name"] = "CONNECTION_REWIRE_ABLATION"
    out["ablation"] = "gate_actions_rotated"
    for gate in out["gates"]:
        idx = (ACTIONS.index(gate["action"]) + 2) % len(ACTIONS)
        gate["action"] = ACTIONS[idx]
    return out


def spike_shuffle_mapping():
    return {action: ACTIONS[(idx + 3) % len(ACTIONS)] for idx, action in enumerate(ACTIONS)}


def mutate_sparse_controller(controller, rng):
    out = copy.deepcopy(controller)
    gate = rng.choice(out["gates"])
    roll = rng.random()
    if roll < 0.30:
        gate["threshold"] = max(0, min(17, gate["threshold"] + rng.choice([-2, -1, 1, 2])))
        gate["stored_threshold"] = stored_threshold(gate["threshold"])
        mutation_type = "gate_threshold"
    elif roll < 0.55:
        gate["weight"] = max(-32, min(48, gate["weight"] + rng.choice([-4, -2, 2, 4])))
        gate["polarity"] = 1 if gate["weight"] >= 0 else -1
        mutation_type = "gate_weight"
    elif roll < 0.72:
        gate["priority"] = max(0, min(120, gate["priority"] + rng.choice([-10, -5, 5, 10])))
        mutation_type = "gate_priority"
    elif roll < 0.86:
        gate["channel"] = rng.randint(1, 8)
        mutation_type = "gate_channel"
    else:
        gate["action"] = rng.choice(ACTIONS)
        mutation_type = "gate_action"
    out["mutation_history"] = out.get("mutation_history", []) + [mutation_type]
    out["_last_mutation"] = mutation_type
    return out, mutation_type


def sparse_policy_fitness(controller, examples, indexes):
    if not indexes:
        indexes = list(range(len(examples)))
    total = 0.0
    counts = Counter()
    for idx in indexes:
        ex = examples[idx]
        action, _trace = choose_sparse_action(controller, ex["features"])
        counts[action] += 1
        total += d51.action_fitness(ex["action_compact"][action])
    return total / len(indexes), counts


def full_sparse_policy_score(controller, examples):
    indexes = list(range(len(examples)))
    score, counts = sparse_policy_fitness(controller, examples, indexes)
    return score, dict(counts)


def train_sparse_controller(seed_controller, examples, generations, population, seed, out, started, heartbeat_sec):
    rng = random.Random(seed + stable_seed("d55_sparse_firing_controller"))
    population_items = [copy.deepcopy(seed_controller)]
    while len(population_items) < population:
        child, _mutation = mutate_sparse_controller(seed_controller, rng)
        population_items.append(child)
    best_controller = copy.deepcopy(seed_controller)
    best_score, _counts = full_sparse_policy_score(best_controller, examples)
    history = []
    mutation_counts = Counter()
    accepted_mutation_counts = Counter()
    last = 0.0
    batch_size = min(2048, len(examples))
    for gen in range(generations):
        batch_rng = random.Random(seed * 10_000 + gen)
        indexes = [batch_rng.randrange(len(examples)) for _ in range(batch_size)]
        scored = []
        for controller in population_items:
            score, _ = sparse_policy_fitness(controller, examples, indexes)
            scored.append((score, controller))
            if score > best_score:
                best_score = score
                best_controller = copy.deepcopy(controller)
        scored.sort(key=lambda item: item[0], reverse=True)
        elites = [copy.deepcopy(controller) for _score, controller in scored[: max(2, population // 5)]]
        for elite in elites:
            mutation_type = elite.get("_last_mutation")
            if mutation_type:
                accepted_mutation_counts[mutation_type] += 1
        next_population = elites[:]
        while len(next_population) < population:
            parent = copy.deepcopy(rng.choice(elites))
            child, mutation_type = mutate_sparse_controller(parent, rng)
            mutation_counts[mutation_type] += 1
            next_population.append(child)
        population_items = next_population
        if gen % max(1, generations // 10) == 0 or gen == generations - 1:
            full_score, counts = full_sparse_policy_score(best_controller, examples)
            history.append(
                {
                    "generation": gen,
                    "batch_best_fitness": scored[0][0],
                    "full_best_fitness": full_score,
                    "action_counts": counts,
                    "mutation_counts": dict(mutation_counts),
                    "accepted_mutation_counts": dict(accepted_mutation_counts),
                }
            )
        now = time.time()
        if now - last >= heartbeat_sec:
            last = now
            append_progress(out, "sparse_mutation_progress", started, {"generation": gen, "best_score": best_score})
            write_json(
                out / "partial_sparse_mutation.json",
                {
                    "generation": gen,
                    "best_score": best_score,
                    "mutation_counts": dict(mutation_counts),
                    "accepted_mutation_counts": dict(accepted_mutation_counts),
                },
            )
    final_score, counts = full_sparse_policy_score(best_controller, examples)
    return best_controller, {
        "kind": "controller_local_sparse_firing",
        "fitness": final_score,
        "action_counts": counts,
        "mutation_counts": dict(mutation_counts),
        "accepted_mutation_counts": dict(accepted_mutation_counts),
        "history": history,
    }


def renamed_reference(pack, arm):
    result = copy.deepcopy(pack["references"]["HANDCODED_D50_FULL_REPAIRED_ECF_REFERENCE"])
    result["arm"] = arm
    result["sparse_firing_used"] = False
    result["spike_update_executed"] = False
    result["spike_update_count"] = 0
    result["fired_gate_count"] = 0
    result["sparse_output_charge"] = {}
    return result


def output_from_action(pack, arm, action, sparse_trace=None):
    result = d54.output_from_action(pack, arm, action)
    if sparse_trace is not None:
        result["sparse_firing_used"] = True
        result["spike_update_executed"] = True
        result["spike_update_count"] = sparse_trace["spike_updates"]
        result["fired_gate_count"] = len(sparse_trace["fired_gates"])
        result["sparse_output_charge"] = sparse_trace["output_charge"]
    else:
        result["sparse_firing_used"] = False
        result["spike_update_executed"] = False
        result["spike_update_count"] = 0
        result["fired_gate_count"] = 0
        result["sparse_output_charge"] = {}
    return result


def evaluate_pack_all_arms(pack, policies, sparse_controllers, sparse_stats):
    rows = []
    hybrid = policies.get("VRAXION_MUTABLE_HYBRID_CONTROLLER")
    sparse_gate = policies.get("VRAXION_MUTABLE_SPARSE_GATE_CONTROLLER")
    rule = policies.get("VRAXION_MUTABLE_RULE_TABLE")
    if hybrid is not None:
        rows.append(output_from_action(pack, "D54_BEST_HYBRID_REPLAY", d53.choose_vraxion_action(hybrid, pack["features"])))
    if sparse_gate is not None:
        rows.append(output_from_action(pack, "D54_SPARSE_GATE_REPLAY", d53.choose_vraxion_action(sparse_gate, pack["features"])))
    rows.append(renamed_reference(pack, "HANDCODED_D50_FULL_REFERENCE"))
    rows.append(output_from_action(pack, "RANDOM_POLICY_CONTROL", d51.random_policy_action(f"D55:{pack['row_id']}")))
    rows.append(output_from_action(pack, "GREEDY_DECIDE_CONTROL", "DECIDE"))
    rows.append(output_from_action(pack, "ALWAYS_COUNTER_CONTROL", "REQUEST_JOINT_COUNTER"))
    if rule is not None:
        rows.append(output_from_action(pack, "MUTABLE_RULE_TABLE_REFERENCE", d53.choose_vraxion_action(rule, pack["features"])))

    for arm in SPARSE_ARMS:
        controller = sparse_controllers.get(arm)
        if controller is None:
            continue
        action, trace = choose_sparse_action(controller, pack["features"], sparse_stats[arm])
        rows.append(output_from_action(pack, arm, action, trace))

    shuffle = spike_shuffle_mapping()
    controller = sparse_controllers.get("REAL_SPARSE_FIRING_CONTROLLER_WITH_MUTATION")
    if controller is not None:
        action, trace = choose_sparse_action(controller, pack["features"], sparse_stats["SPIKE_SHUFFLE_CONTROL"], action_shuffle=shuffle)
        rows.append(output_from_action(pack, "SPIKE_SHUFFLE_CONTROL", action, trace))
    for arm in ["FIRING_THRESHOLD_ABLATION", "CONNECTION_REWIRE_ABLATION"]:
        controller = sparse_controllers.get(arm)
        if controller is not None:
            action, trace = choose_sparse_action(controller, pack["features"], sparse_stats[arm])
            rows.append(output_from_action(pack, arm, action, trace))
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


def evaluate_packs(packs, policies, sparse_controllers, out_path, out, split, started, heartbeat_sec):
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
        batch = evaluate_pack_all_arms(pack, policies, sparse_controllers, sparse_stats)
        record_batch(batch, outputs, sample_counts, out_path)
        completed += len(batch)
        now = time.time()
        if now - last >= heartbeat_sec or completed >= total:
            last = now
            write_partial_eval(out, split, outputs, completed, started)
    return outputs, normalize_sparse_stats(sparse_stats)


def normalize_sparse_stats(sparse_stats):
    out = {}
    for arm, stats in sparse_stats.items():
        calls = stats["calls"] or 1
        out[arm] = {
            "calls": stats["calls"],
            "spike_update_executed_count": stats["spike_update_executed_count"],
            "average_spike_updates_per_call": stats["spike_update_executed_count"] / calls,
            "average_fired_gates_per_call": stats["fired_gate_count"] / calls,
            "average_input_charge_per_call": stats["total_input_charge"] / calls,
            "average_output_charge_abs_per_call": stats["output_charge_sum"] / calls,
            "action_counts": dict(stats["action_counts"]),
            "fired_gate_by_feature": dict(stats["fired_gate_by_feature"]),
            "fired_gate_by_action": dict(stats["fired_gate_by_action"]),
        }
    return out


def make_decision(metrics, failed_jobs, sparse_usage):
    arm = "REAL_SPARSE_FIRING_CONTROLLER_WITH_MUTATION"
    if failed_jobs:
        return {
            "decision": "sparse_firing_ecf_controller_not_confirmed",
            "verdict": "D55_FAILED_JOBS_PRESENT",
            "next": "D55R_SPARSE_CONTROLLER_REPAIR",
            "boundary": BOUNDARY,
        }
    if not sparse_usage.get("sparse_firing_used"):
        return {
            "decision": "sparse_firing_path_not_exercised",
            "verdict": "D55_SPARSE_FIRING_PATH_NOT_EXERCISED",
            "next": "D55R_SPARSE_INTEGRATION_REPAIR",
            "boundary": BOUNDARY,
        }
    core = metrics["by_arm_core"]
    regimes = metrics["by_arm_and_regime"]
    row = core[arm]
    corr = regime_accuracy(metrics, arm, "CORRELATED_ECHO_SUPPORT")
    adv = regime_accuracy(metrics, arm, "ADVERSARIAL_DISTRACTOR_SUPPORT")
    external = regime_accuracy(metrics, arm, "EXTERNAL_TEST_REQUIRED_SUPPORT")
    indist = regimes[arm]["INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT"]
    positive = (
        row["exact_joint_accuracy"] >= 0.98
        and corr >= 0.95
        and adv >= 0.95
        and external >= 0.95
        and indist["abstain_rate"] >= 0.99
        and indist["false_confidence_rate"] <= 0.01
        and row["average_total_support_used"] <= core["ALWAYS_COUNTER_CONTROL"]["average_total_support_used"]
        and row["accuracy"] > core["RANDOM_POLICY_CONTROL"]["accuracy"]
        and row["accuracy"] > core["GREEDY_DECIDE_CONTROL"]["accuracy"]
        and row["accuracy"] > core["SPIKE_SHUFFLE_CONTROL"]["accuracy"]
    )
    strong = (
        positive
        and row["exact_joint_accuracy"] >= 0.995
        and corr >= 0.995
        and adv >= 0.995
        and external >= 0.99
        and min_seed_metric(metrics, arm, "exact_joint_accuracy") >= 0.99
    )
    if strong:
        return {
            "decision": "sparse_firing_ecf_controller_prototype_strong_positive",
            "verdict": "D55_SPARSE_FIRING_ECF_CONTROLLER_PROTOTYPE_STRONG_POSITIVE",
            "next": "D56_SPARSE_FIRING_ECF_CONTROLLER_SCALE_CONFIRM",
            "best_sparse_arm": arm,
            "boundary": BOUNDARY,
        }
    if positive:
        return {
            "decision": "sparse_firing_ecf_controller_prototype_positive",
            "verdict": "D55_SPARSE_FIRING_ECF_CONTROLLER_PROTOTYPE_POSITIVE",
            "next": "D56_SPARSE_FIRING_CONTROLLER_HARDENING",
            "best_sparse_arm": arm,
            "boundary": BOUNDARY,
        }
    return {
        "decision": "sparse_firing_ecf_controller_not_confirmed",
        "verdict": "D55_SPARSE_FIRING_ECF_CONTROLLER_NOT_CONFIRMED",
        "next": "D55R_SPARSE_CONTROLLER_REPAIR",
        "best_sparse_arm": arm,
        "boundary": BOUNDARY,
    }


def network_topology_report(controllers):
    report = {}
    for arm, controller in controllers.items():
        thresholds = [gate["stored_threshold"] for gate in controller["gates"]]
        channels = [gate["channel"] for gate in controller["gates"]]
        report[arm] = {
            "input_neurons": controller["input_neurons"],
            "gate_neurons": controller["gate_neurons"],
            "hidden_neurons": controller["hidden_neurons"],
            "output_neurons": controller["output_neurons"],
            "total_neurons": controller["input_neurons"] + controller["gate_neurons"] + controller["hidden_neurons"] + controller["output_neurons"],
            "edge_count": controller_edge_count(controller),
            "threshold_min": min(thresholds) if thresholds else None,
            "threshold_max": max(thresholds) if thresholds else None,
            "channels": sorted(set(channels)),
            "readout": controller["readout"],
            "controller_only_not_formula_solver": True,
        }
    return report


def make_reports(out, aggregate, decision, d54_manifest, canonical_report, sparse_usage, controllers):
    test = aggregate["test_metrics"]
    core = test["by_arm_core"]
    regimes = test["by_arm_and_regime"]
    with_mut = "REAL_SPARSE_FIRING_CONTROLLER_WITH_MUTATION"
    reports = {
        "d54_upstream_manifest.json": d54_manifest,
        "canonical_sparse_firing_audit_report.json": canonical_report,
        "sparse_firing_usage_report.json": sparse_usage,
        "network_topology_report.json": network_topology_report(controllers),
        "firing_dynamics_report.json": aggregate["sparse_firing_stats"],
        "mutation_acceptance_report.json": aggregate["policy_reports"],
        "action_readout_report.json": {
            arm: {
                "readout": controllers[arm]["readout"],
                "default_action": controllers[arm]["default_action"],
                "gate_to_action": [
                    {
                        "feature": gate["feature"],
                        "action": gate["action"],
                        "threshold": gate["threshold"],
                        "priority": gate["priority"],
                        "weight": gate["weight"],
                    }
                    for gate in controllers[arm]["gates"]
                ],
                "action_distribution": test["action_distribution"].get(arm, {}),
            }
            for arm in controllers
        },
        "threshold_ablation_report.json": {
            "ablation_arm": "FIRING_THRESHOLD_ABLATION",
            "metrics": core.get("FIRING_THRESHOLD_ABLATION", {}),
            "regime_metrics": regimes.get("FIRING_THRESHOLD_ABLATION", {}),
            "ablation": "counter-support gate thresholds disabled",
        },
        "spike_shuffle_control_report.json": {
            "mapping": spike_shuffle_mapping(),
            "metrics": core.get("SPIKE_SHUFFLE_CONTROL", {}),
            "regime_metrics": regimes.get("SPIKE_SHUFFLE_CONTROL", {}),
            "expected_worse_than_real_sparse": core["SPIKE_SHUFFLE_CONTROL"]["accuracy"] < core[with_mut]["accuracy"],
        },
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
                "indistinguishable_false_confidence": regimes[arm]["INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT"]["false_confidence_rate"],
                "indistinguishable_abstain": regimes[arm]["INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT"]["abstain_rate"],
            }
            for arm in ARMS
        },
        "regime_breakdown_report.json": regimes,
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
                "sparse_firing_used": arm in SPARSE_ARMS + ABLATION_ARMS,
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
    arm = "REAL_SPARSE_FIRING_CONTROLLER_WITH_MUTATION"
    lines = [
        "# D55 Sparse Firing ECF Controller Prototype Result",
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
        "```",
        "",
        "Boundary:",
        "",
        "```text",
        BOUNDARY,
        "```",
        "",
        "Main sparse controller:",
        "",
        "```text",
        f"arm = {arm}",
        f"exact_joint_accuracy = {core[arm]['exact_joint_accuracy']:.6f}",
        f"correlated_echo = {regimes[arm]['CORRELATED_ECHO_SUPPORT']['accuracy']:.6f}",
        f"adversarial_distractor = {regimes[arm]['ADVERSARIAL_DISTRACTOR_SUPPORT']['accuracy']:.6f}",
        f"external_test_required = {regimes[arm]['EXTERNAL_TEST_REQUIRED_SUPPORT']['accuracy']:.6f}",
        f"indistinguishable_abstain = {regimes[arm]['INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT']['abstain_rate']:.6f}",
        f"false_confidence = {regimes[arm]['INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT']['false_confidence_rate']:.6f}",
        f"support = {core[arm]['average_total_support_used']:.6f}",
        "```",
        "",
        "Controller comparison:",
        "",
        "| arm | exact | corr | adv | external | support |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in ARMS:
        lines.append(
            f"| {item} | {core[item]['exact_joint_accuracy']:.4f} | "
            f"{regimes[item]['CORRELATED_ECHO_SUPPORT']['accuracy']:.4f} | "
            f"{regimes[item]['ADVERSARIAL_DISTRACTOR_SUPPORT']['accuracy']:.4f} | "
            f"{regimes[item]['EXTERNAL_TEST_REQUIRED_SUPPORT']['accuracy']:.4f} | "
            f"{core[item]['average_total_support_used']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Sparse firing statement:",
            "",
            "```text",
            "The sparse arms select actions from a controller-local spike update using charge, threshold, channel/phase, and sparse gate readout.",
            "The Rust Network source was audited, but the Rust network binary path was not invoked in D55.",
            "This is not a full sparse firing formula-solving brain.",
            "```",
            "",
        ]
    )
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def make_summary(aggregate, decision):
    arm = "REAL_SPARSE_FIRING_CONTROLLER_WITH_MUTATION"
    test = aggregate["test_metrics"]
    core = test["by_arm_core"][arm]
    regimes = test["by_arm_and_regime"][arm]
    return {
        "task": "D55_SPARSE_FIRING_ECF_CONTROLLER_PROTOTYPE",
        "decision": decision["decision"],
        "verdict": decision.get("verdict"),
        "next": decision.get("next"),
        "scale_mode": aggregate["scale_mode"],
        "best_sparse_arm": arm,
        "sparse_firing_used": aggregate["sparse_firing_usage"]["sparse_firing_used"],
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


def parse_seeds(raw):
    return [int(item.strip()) for item in str(raw).split(",") if item.strip()]


def worker_count_from_arg(raw):
    return d51.worker_count_from_arg(raw)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", default="10901,10902,10903,10904,10905")
    parser.add_argument("--train-rows-per-seed", type=int, default=800)
    parser.add_argument("--test-rows-per-seed", type=int, default=800)
    parser.add_argument("--ood-rows-per-seed", type=int, default=800)
    parser.add_argument("--generations", type=int, default=160)
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--workers", default="auto")
    parser.add_argument("--cpu-target", default="50-75")
    parser.add_argument("--heartbeat-sec", type=float, default=20.0)
    parser.add_argument("--scale-mode", default="prototype")
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
            "task": "D55_SPARSE_FIRING_ECF_CONTROLLER_PROTOTYPE",
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
            "workers": worker_count_from_arg(args.workers),
            "cpu_target": args.cpu_target,
            "cuda_probe": "not_used_controller_local_python",
        },
    )

    d54_manifest = make_d54_upstream_manifest(repo_root)
    policies = load_d54_policies(repo_root, d54_manifest)
    write_json(out / "d54_upstream_manifest.json", d54_manifest)
    if not policies:
        failed_jobs.append("missing_d54_policies")

    canonical_report = canonical_sparse_firing_audit(repo_root)
    write_json(out / "canonical_sparse_firing_audit_report.json", canonical_report)
    if not canonical_report["canonical_rust_network_audited"]:
        failed_jobs.append("canonical_sparse_firing_surface_missing")

    bundle = d49_bundle()
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
            "label_echo_reference_only": False,
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
    sparse_seed = make_sparse_controller_from_gate(
        policies.get("VRAXION_MUTABLE_SPARSE_GATE_CONTROLLER", d53.make_vraxion_policy("sparse_gate", random.Random(55_001))),
        "D54_sparse_gate_seed",
        hidden_neurons=0,
    )
    sparse_small = copy.deepcopy(sparse_seed)
    sparse_small["name"] = "REAL_SPARSE_FIRING_CONTROLLER_SMALL"
    sparse_medium = make_sparse_controller_from_gate(
        policies.get("VRAXION_MUTABLE_SPARSE_GATE_CONTROLLER", d53.make_vraxion_policy("sparse_gate", random.Random(55_002))),
        "REAL_SPARSE_FIRING_CONTROLLER_MEDIUM",
        hidden_neurons=24,
    )
    sparse_no_mutation = copy.deepcopy(sparse_seed)
    sparse_no_mutation["name"] = "REAL_SPARSE_FIRING_CONTROLLER_NO_MUTATION"
    sparse_with_mutation, sparse_report = train_sparse_controller(
        sparse_seed,
        train_examples,
        args.generations,
        args.population,
        55_000,
        out,
        started,
        args.heartbeat_sec,
    )
    sparse_with_mutation["name"] = "REAL_SPARSE_FIRING_CONTROLLER_WITH_MUTATION"

    sparse_controllers = {
        "REAL_SPARSE_FIRING_CONTROLLER_SMALL": sparse_small,
        "REAL_SPARSE_FIRING_CONTROLLER_MEDIUM": sparse_medium,
        "REAL_SPARSE_FIRING_CONTROLLER_WITH_MUTATION": sparse_with_mutation,
        "REAL_SPARSE_FIRING_CONTROLLER_NO_MUTATION": sparse_no_mutation,
        "FIRING_THRESHOLD_ABLATION": make_threshold_ablation(sparse_with_mutation),
        "CONNECTION_REWIRE_ABLATION": make_rewire_ablation(sparse_with_mutation),
    }
    policy_reports = {
        "REAL_SPARSE_FIRING_CONTROLLER_WITH_MUTATION": sparse_report,
    }
    for arm in ["REAL_SPARSE_FIRING_CONTROLLER_SMALL", "REAL_SPARSE_FIRING_CONTROLLER_MEDIUM", "REAL_SPARSE_FIRING_CONTROLLER_NO_MUTATION"]:
        score, counts = full_sparse_policy_score(sparse_controllers[arm], train_examples)
        policy_reports[arm] = {
            "kind": "controller_local_sparse_firing",
            "fitness": score,
            "action_counts": counts,
            "mutation_counts": {},
            "accepted_mutation_counts": {},
            "history": [],
        }
    for arm in ["FIRING_THRESHOLD_ABLATION", "CONNECTION_REWIRE_ABLATION"]:
        score, counts = full_sparse_policy_score(sparse_controllers[arm], train_examples)
        policy_reports[arm] = {
            "kind": "controller_local_sparse_firing_ablation",
            "fitness": score,
            "action_counts": counts,
            "mutation_counts": {},
            "accepted_mutation_counts": {},
            "history": [],
        }
    write_json(
        out / "trained_policy_manifest.json",
        {
            "sparse_controllers": sparse_controllers,
            "policy_reports": policy_reports,
            "d54_replay_policies": policies,
        },
    )
    append_progress(out, "sparse_training_complete", started, {"fitness": sparse_report["fitness"]})

    test_outputs, test_sparse_stats = evaluate_packs(
        test_packs, policies, sparse_controllers, out / "row_outputs_test.jsonl", out, "test", started, args.heartbeat_sec
    )
    ood_outputs, ood_sparse_stats = evaluate_packs(
        ood_packs, policies, sparse_controllers, out / "row_outputs_ood.jsonl", out, "ood", started, args.heartbeat_sec
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
        "rust_network_path_invoked": False,
        "canonical_rust_network_audited": canonical_report["canonical_rust_network_audited"],
        "python_sparse_firing_semantics": "controller-local charge/threshold/channel sparse tick with action readout",
    }
    write_json(out / "sparse_firing_usage_report.json", sparse_usage)

    aggregate = {
        "task": "D55_SPARSE_FIRING_ECF_CONTROLLER_PROTOTYPE",
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
        "boundary": BOUNDARY,
    }
    decision = make_decision(test_metrics, failed_jobs, sparse_usage)
    aggregate["decision"] = decision
    reports = make_reports(out, aggregate, decision, d54_manifest, canonical_report, sparse_usage, sparse_controllers)
    aggregate["reports_written"] = sorted(reports.keys())
    write_json(out / "aggregate_metrics.json", aggregate)
    write_json(out / "decision.json", decision)
    write_json(out / "summary.json", make_summary(aggregate, decision))
    write_report(out, decision, aggregate)
    append_progress(out, "completed", started, {"decision": decision["decision"], "elapsed_sec": time.time() - started})


def d49_bundle():
    import run_d49_joint_cell_operator_discovery_with_robust_support as d49

    return d49.make_bundle()


if __name__ == "__main__":
    main()
