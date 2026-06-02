#!/usr/bin/env python3
"""D53 mutable ECF integration with VRAXION mutation architecture.

This probe does not learn the formula solver. It keeps the D50-D52 controlled
symbolic joint formula task fixed and tests whether the D52 controller policy
can be represented and optimized as a VRAXION-style mutable controller genome.
"""

import argparse
import copy
import json
import math
import os
import random
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
ROW_SAMPLE_PER_ARM_REGIME_SPLIT = d51.ROW_SAMPLE_PER_ARM_REGIME_SPLIT

BOUNDARY = (
    "D53 only tests VRAXION-style mutation integration for mutable ECF controller policy "
    "in controlled symbolic joint formula discovery. It does not prove raw visual Raven, "
    "Raven solved, AGI, consciousness, DNA/genome success, full VRAXION brain learning, "
    "or architecture superiority."
)

REFERENCE_ARMS = [
    "D52_BEST_RULE_TABLE_REPLAY",
    "HANDCODED_D50_FULL_REFERENCE",
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

ARMS = REFERENCE_ARMS + CONTROL_ARMS + VRAXION_ARMS

CANONICAL_VRAXION_PATHS = {
    "current_architecture_research_path": "instnct-core/examples/neuron_grower.rs",
    "spiking_network_surface": "instnct-core/src/network.rs",
    "mutation_schedule_surface": "instnct-core/src/evolution.rs",
    "run_contract": "docs/GROWER_RUN_CONTRACT.md",
}


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


def load_json_if_present(path):
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def quant_feature(features, feature_name):
    idx = FEATURE_NAMES.index(feature_name)
    value = max(0.0, min(1.0, float(features[idx])))
    return int(round(value * 16.0))


def best_action_for_examples(examples):
    scores = {}
    for action in ACTIONS:
        scores[action] = mean([d51.action_fitness(ex["action_compact"][action]) for ex in examples])
    return max(scores.items(), key=lambda item: (item[1], -ACTIONS.index(item[0])))[0]


def split_examples_by_controller_context(examples):
    idx = {name: FEATURE_NAMES.index(name) for name in FEATURE_NAMES}
    bins = {
        "external": [],
        "unresolvable": [],
        "dominant": [],
        "uncertain": [],
        "default": [],
    }
    for ex in examples:
        features = ex["features"]
        if features[idx["external_channel_available"]] > 0.5:
            bins["external"].append(ex)
        elif features[idx["internal_unresolvable_indicator"]] > 0.5:
            bins["unresolvable"].append(ex)
        elif features[idx["dominant_cluster_fraction"]] >= 0.55:
            bins["dominant"].append(ex)
        elif features[idx["inverse_margin"]] >= 0.45 or features[idx["entropy_norm"]] >= 0.65:
            bins["uncertain"].append(ex)
        else:
            bins["default"].append(ex)
    return bins


def bootstrap_actions(examples):
    bins = split_examples_by_controller_context(examples)
    return {
        "external": best_action_for_examples(bins["external"]) if bins["external"] else "REQUEST_EXTERNAL_TEST",
        "unresolvable": best_action_for_examples(bins["unresolvable"]) if bins["unresolvable"] else "ABSTAIN",
        "dominant": best_action_for_examples(bins["dominant"]) if bins["dominant"] else "REQUEST_JOINT_COUNTER",
        "uncertain": best_action_for_examples(bins["uncertain"]) if bins["uncertain"] else "REQUEST_COUNTER_TOP1_TOP2",
        "default": best_action_for_examples(bins["default"]) if bins["default"] else "DECIDE",
    }


def make_d52_upstream_manifest(repo_root):
    root = repo_root / "target/pilot_wave/d52_mutable_ecf_controller_scale_confirm/smoke"
    manifest = {
        "upstream": "D52_MUTABLE_ECF_CONTROLLER_SCALE_CONFIRM",
        "expected_decision": "mutable_ecf_controller_scale_confirmed",
        "expected_next": "D53_MUTABLE_ECF_INTEGRATION_WITH_VRAXION_MUTATION_ARCHITECTURE",
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
                "correlated_echo": key.get("best_by_regime", {}).get("CORRELATED_ECHO_SUPPORT", {}).get("accuracy"),
                "adversarial_distractor": key.get("best_by_regime", {}).get("ADVERSARIAL_DISTRACTOR_SUPPORT", {}).get("accuracy"),
                "external_test": key.get("best_by_regime", {}).get("EXTERNAL_TEST_REQUIRED_SUPPORT", {}).get("accuracy"),
                "indist_abstain": key.get("best_by_regime", {}).get("INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT", {}).get("abstain_rate"),
                "false_confidence": key.get("best_by_regime", {}).get("INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT", {}).get("false_confidence_rate"),
            }
    if aggregate:
        manifest["aggregate_best_mutable_arm"] = aggregate.get("best_mutable_arm")
    return manifest


def load_best_d52_policy(repo_root, manifest):
    root = repo_root / "target/pilot_wave/d52_mutable_ecf_controller_scale_confirm/smoke"
    trained = load_json_if_present(root / "trained_policy_manifest.json")
    aggregate = load_json_if_present(root / "aggregate_metrics.json")
    if not trained or not aggregate:
        manifest["best_d52_replay_loaded"] = False
        return None
    best = aggregate.get("best_mutable_arm")
    policy = trained.get("policies", {}).get(best)
    manifest["best_d52_replay_loaded"] = policy is not None
    manifest["best_d52_replay_arm"] = best
    return policy


def canonical_vraxion_audit(repo_root):
    report = {
        "canonical_paths": CANONICAL_VRAXION_PATHS,
        "paths_present": {},
        "canonical_vraxion_module_path": CANONICAL_VRAXION_PATHS["current_architecture_research_path"],
        "spiking_network_module_path": CANONICAL_VRAXION_PATHS["spiking_network_surface"],
        "mutation_schedule_module_path": CANONICAL_VRAXION_PATHS["mutation_schedule_surface"],
        "threshold_state_limits": {"stored_threshold": "[0,15]", "effective_threshold": "[1,16]"},
        "mutation_operator_list": [],
        "sparse_firing_used_in_d53": False,
        "controller_genome_state_used": True,
        "full_formula_solver_learning_used": False,
        "action_output_encoding_smoke_passed": False,
        "source_smoke_passed": False,
        "notes": [
            "D53 audits the canonical Rust VRAXION surfaces but keeps inference in the D50-D52 controlled symbolic Python runner.",
            "The VRAXION-style arms are mutable controller genomes, not full sparse firing brain training.",
        ],
    }
    for label, rel in CANONICAL_VRAXION_PATHS.items():
        report["paths_present"][label] = (repo_root / rel).exists()
    evolution = (repo_root / CANONICAL_VRAXION_PATHS["mutation_schedule_surface"]).read_text(encoding="utf-8")
    network = (repo_root / CANONICAL_VRAXION_PATHS["spiking_network_surface"]).read_text(encoding="utf-8")
    grower = (repo_root / CANONICAL_VRAXION_PATHS["current_architecture_research_path"]).read_text(encoding="utf-8")
    for line in evolution.splitlines():
        line = line.strip()
        if line.startswith('id: "'):
            report["mutation_operator_list"].append(line.split('"')[1])
    report["source_smoke_passed"] = all(
        [
            "pub const MUTATION_OPERATORS" in evolution,
            "pub struct SpikeData" in network,
            "threshold" in network and "[0,15]" in network,
            "dot >= threshold" in grower,
            "state.tsv" in grower,
        ]
    )
    output_actions = set(ACTIONS)
    report["action_output_encoding_smoke_passed"] = output_actions == {
        "DECIDE",
        "REQUEST_SUPPORT",
        "REQUEST_COUNTER_TOP1_TOP2",
        "REQUEST_JOINT_COUNTER",
        "REQUEST_EXTERNAL_TEST",
        "ABSTAIN",
    }
    return report


def make_vraxion_policy(kind, rng, examples=None):
    actions = bootstrap_actions(examples or []) if examples else {
        "external": "REQUEST_EXTERNAL_TEST",
        "unresolvable": "ABSTAIN",
        "dominant": "REQUEST_JOINT_COUNTER",
        "uncertain": "REQUEST_COUNTER_TOP1_TOP2",
        "default": "DECIDE",
    }
    if kind == "rule_table":
        return {
            "kind": "vraxion_rule_table",
            "genome_schema": "threshold_gates_plus_action_routes",
            "margin_threshold": 0.55,
            "dominant_threshold": 0.55,
            "entropy_threshold": 0.65,
            "confidence_threshold": 0.08,
            "action_external": actions["external"],
            "action_unresolvable": actions["unresolvable"],
            "action_dominant": actions["dominant"],
            "action_low_margin": actions["uncertain"],
            "action_uncertain": actions["uncertain"],
            "action_default": actions["default"],
        }
    if kind == "sparse_gate":
        return {
            "kind": "vraxion_sparse_gate",
            "genome_schema": "integer_feature_gates_threshold_0_16_to_action_scores",
            "default_action": actions["default"],
            "gates": [
                {"feature": "external_channel_available", "threshold": 8, "action": actions["external"], "weight": 32, "priority": 100},
                {"feature": "internal_unresolvable_indicator", "threshold": 8, "action": actions["unresolvable"], "weight": 32, "priority": 90},
                {"feature": "dominant_cluster_fraction", "threshold": 9, "action": actions["dominant"], "weight": 24, "priority": 70},
                {"feature": "inverse_margin", "threshold": 8, "action": actions["uncertain"], "weight": 16, "priority": 40},
                {"feature": "entropy_norm", "threshold": 10, "action": actions["uncertain"], "weight": 12, "priority": 30},
                {"feature": "scalar_confidence", "threshold": 2, "action": actions["default"], "weight": 8, "priority": 10},
            ],
        }
    if kind == "pocket_state":
        return {
            "kind": "vraxion_pocket_state",
            "genome_schema": "feature_pockets_with_priority_and_action_writeback",
            "pockets": [
                {"pocket": "external_test", "feature": "external_channel_available", "threshold": 8, "action": actions["external"], "priority": 100},
                {"pocket": "abstain_boundary", "feature": "internal_unresolvable_indicator", "threshold": 8, "action": actions["unresolvable"], "priority": 90},
                {"pocket": "joint_counter", "feature": "dominant_cluster_fraction", "threshold": 9, "action": actions["dominant"], "priority": 70},
                {"pocket": "top1_top2_counter", "feature": "inverse_margin", "threshold": 8, "action": actions["uncertain"], "priority": 40},
            ],
            "default_action": actions["default"],
        }
    if kind == "hybrid":
        return {
            "kind": "vraxion_hybrid",
            "genome_schema": "rule_table_plus_sparse_gate_overlay",
            "rule": make_vraxion_policy("rule_table", rng, examples),
            "gate": make_vraxion_policy("sparse_gate", rng, examples),
            "gate_override_threshold": 28,
        }
    raise ValueError(kind)


def mutate_vraxion_policy(policy, rng, rate=0.14):
    out = copy.deepcopy(policy)
    mutation_type = "noop"
    kind = out["kind"]
    if kind == "vraxion_rule_table":
        numeric = ["margin_threshold", "dominant_threshold", "entropy_threshold", "confidence_threshold"]
        action_keys = ["action_external", "action_unresolvable", "action_dominant", "action_low_margin", "action_uncertain", "action_default"]
        if rng.random() < 0.55:
            key = rng.choice(numeric)
            out[key] = max(0.0, min(1.5, out[key] + rng.gauss(0.0, 0.10)))
            mutation_type = "threshold_gate"
        else:
            key = rng.choice(action_keys)
            out[key] = rng.choice(ACTIONS)
            mutation_type = "action_route"
    elif kind == "vraxion_sparse_gate":
        gate = rng.choice(out["gates"])
        roll = rng.random()
        if roll < 0.35:
            gate["threshold"] = max(0, min(16, gate["threshold"] + rng.choice([-2, -1, 1, 2])))
            mutation_type = "gate_threshold"
        elif roll < 0.60:
            gate["weight"] = max(-32, min(32, gate["weight"] + rng.choice([-4, -2, 2, 4])))
            mutation_type = "gate_weight"
        elif roll < 0.80:
            gate["action"] = rng.choice(ACTIONS)
            mutation_type = "gate_action"
        else:
            gate["priority"] = max(0, min(120, gate["priority"] + rng.choice([-10, -5, 5, 10])))
            mutation_type = "gate_priority"
    elif kind == "vraxion_pocket_state":
        if rng.random() < 0.75:
            pocket = rng.choice(out["pockets"])
            roll = rng.random()
            if roll < 0.40:
                pocket["threshold"] = max(0, min(16, pocket["threshold"] + rng.choice([-2, -1, 1, 2])))
                mutation_type = "pocket_threshold"
            elif roll < 0.70:
                pocket["action"] = rng.choice(ACTIONS)
                mutation_type = "pocket_action"
            else:
                pocket["priority"] = max(0, min(120, pocket["priority"] + rng.choice([-10, -5, 5, 10])))
                mutation_type = "pocket_priority"
        else:
            out["default_action"] = rng.choice(ACTIONS)
            mutation_type = "default_action"
    elif kind == "vraxion_hybrid":
        if rng.random() < 0.50:
            out["rule"] = mutate_vraxion_policy(out["rule"], rng, rate)[0]
            mutation_type = "hybrid_rule"
        elif rng.random() < 0.85:
            out["gate"] = mutate_vraxion_policy(out["gate"], rng, rate)[0]
            mutation_type = "hybrid_gate"
        else:
            out["gate_override_threshold"] = max(0, min(48, out["gate_override_threshold"] + rng.choice([-4, -2, 2, 4])))
            mutation_type = "hybrid_override_threshold"
    return out, mutation_type


def choose_vraxion_action(policy, features):
    feature = {name: features[idx] for idx, name in enumerate(FEATURE_NAMES)}
    kind = policy["kind"]
    if kind == "vraxion_rule_table":
        if feature["external_channel_available"] >= 0.5:
            return policy["action_external"]
        if feature["internal_unresolvable_indicator"] >= 0.5:
            return policy["action_unresolvable"]
        if feature["dominant_cluster_fraction"] >= policy["dominant_threshold"]:
            return policy["action_dominant"]
        if feature["inverse_margin"] >= policy["margin_threshold"]:
            return policy["action_low_margin"]
        if feature["entropy_norm"] >= policy["entropy_threshold"] or feature["scalar_confidence"] <= policy["confidence_threshold"]:
            return policy["action_uncertain"]
        return policy["action_default"]
    if kind == "vraxion_sparse_gate":
        fired = []
        scores = Counter()
        for gate in policy["gates"]:
            value = quant_feature(features, gate["feature"])
            if value >= gate["threshold"]:
                scores[gate["action"]] += gate["weight"]
                fired.append((gate["priority"], gate["action"], gate["weight"]))
        if not fired:
            return policy["default_action"]
        fired.sort(reverse=True)
        if fired[0][0] >= 80:
            return fired[0][1]
        return max(ACTIONS, key=lambda action: (scores[action], -ACTIONS.index(action)))
    if kind == "vraxion_pocket_state":
        fired = []
        for pocket in policy["pockets"]:
            value = quant_feature(features, pocket["feature"])
            if value >= pocket["threshold"]:
                fired.append((pocket["priority"], pocket["action"]))
        if not fired:
            return policy["default_action"]
        fired.sort(reverse=True)
        return fired[0][1]
    if kind == "vraxion_hybrid":
        gate_action = choose_vraxion_action(policy["gate"], features)
        if gate_action in {"REQUEST_EXTERNAL_TEST", "ABSTAIN"}:
            return gate_action
        gate_score = 0
        for gate in policy["gate"]["gates"]:
            if quant_feature(features, gate["feature"]) >= gate["threshold"]:
                gate_score += abs(gate["weight"])
        if gate_score >= policy["gate_override_threshold"]:
            return gate_action
        return choose_vraxion_action(policy["rule"], features)
    raise ValueError(kind)


def policy_fitness(policy, examples, indexes):
    if not indexes:
        indexes = range(len(examples))
    total = 0.0
    counts = Counter()
    for idx in indexes:
        ex = examples[idx]
        action = choose_vraxion_action(policy, ex["features"])
        counts[action] += 1
        total += d51.action_fitness(ex["action_compact"][action])
    indexes_len = len(indexes) if hasattr(indexes, "__len__") else len(list(indexes))
    return total / indexes_len, counts


def full_policy_score(policy, examples):
    indexes = list(range(len(examples)))
    score, counts = policy_fitness(policy, examples, indexes)
    return score, dict(counts)


def train_vraxion_policy(kind, examples, generations, population, seed, out, started, heartbeat_sec):
    rng = random.Random(seed + d51.stable_seed(kind))
    population_items = [make_vraxion_policy(kind, rng, examples) for _ in range(population)]
    best_policy = population_items[0]
    best_score = -1e9
    history = []
    mutation_counter = Counter()
    accepted_counter = Counter()
    last = 0.0
    batch_size = min(2048, len(examples))
    for gen in range(generations):
        batch_rng = random.Random(seed * 10_000 + gen)
        indexes = [batch_rng.randrange(len(examples)) for _ in range(batch_size)]
        scored = []
        for policy in population_items:
            score, _counts = policy_fitness(policy, examples, indexes)
            scored.append((score, policy))
            if score > best_score:
                best_score = score
                best_policy = copy.deepcopy(policy)
        scored.sort(key=lambda item: item[0], reverse=True)
        elites = [copy.deepcopy(policy) for _score, policy in scored[: max(2, population // 5)]]
        next_population = elites[:]
        elite_scores = [score for score, _policy in scored[: max(2, population // 5)]]
        while len(next_population) < population:
            parent_idx = rng.randrange(len(elites))
            parent = copy.deepcopy(elites[parent_idx])
            child, mutation_type = mutate_vraxion_policy(parent, rng, rate=0.14)
            mutation_counter[mutation_type] += 1
            child_score, _ = policy_fitness(child, examples, indexes)
            if child_score >= elite_scores[parent_idx] - 0.02:
                accepted_counter[mutation_type] += 1
            next_population.append(child)
        population_items = next_population
        if gen % max(1, generations // 10) == 0 or gen == generations - 1:
            full_score, counts = full_policy_score(best_policy, examples)
            history.append(
                {
                    "generation": gen,
                    "batch_best_fitness": scored[0][0],
                    "full_best_fitness": full_score,
                    "action_counts": counts,
                    "mutation_counts": dict(mutation_counter),
                    "accepted_mutation_counts": dict(accepted_counter),
                }
            )
        now = time.time()
        if now - last >= heartbeat_sec:
            last = now
            append_progress(out, "vraxion_mutation_progress", started, {"kind": kind, "generation": gen, "best_batch_score": best_score})
            write_json(out / f"partial_vraxion_mutation_{kind}.json", {"kind": kind, "generation": gen, "best_batch_score": best_score, "mutation_counts": dict(mutation_counter)})
    final_score, counts = full_policy_score(best_policy, examples)
    return best_policy, {
        "kind": kind,
        "fitness": final_score,
        "action_counts": counts,
        "mutation_counts": dict(mutation_counter),
        "accepted_mutation_counts": dict(accepted_counter),
        "history": history,
    }


def renamed_reference(pack, arm):
    result = copy.deepcopy(pack["references"]["HANDCODED_D50_FULL_REPAIRED_ECF_REFERENCE"])
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
    d52_policy = policies.get("D52_BEST_RULE_TABLE_REPLAY")
    if d52_policy is not None:
        rows.append(output_from_action(pack, "D52_BEST_RULE_TABLE_REPLAY", d51.choose_action(d52_policy, pack["features"])))
    rows.append(renamed_reference(pack, "HANDCODED_D50_FULL_REFERENCE"))
    rows.append(output_from_action(pack, "RANDOM_POLICY_CONTROL", d51.random_policy_action(f"D53:{pack['row_id']}")))
    rows.append(output_from_action(pack, "GREEDY_DECIDE_CONTROL", "DECIDE"))
    rows.append(output_from_action(pack, "ALWAYS_COUNTER_CONTROL", "REQUEST_JOINT_COUNTER"))
    rows.append(output_from_action(pack, "COST_ONLY_MUTATION_CONTROL", cost_only_action(pack)))
    for arm in VRAXION_ARMS:
        policy = policies.get(arm)
        if policy is not None:
            rows.append(output_from_action(pack, arm, choose_vraxion_action(policy, pack["features"])))
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


def sparse_or_pocket_failed(metrics):
    sparse_ok = metrics["by_arm_core"]["VRAXION_MUTABLE_SPARSE_GATE_CONTROLLER"]["exact_joint_accuracy"] >= 0.995
    pocket_ok = metrics["by_arm_core"]["VRAXION_MUTABLE_POCKET_STATE_CONTROLLER"]["exact_joint_accuracy"] >= 0.995
    return not (sparse_ok and pocket_ok)


def make_decision(metrics, failed_jobs):
    if failed_jobs:
        return {
            "decision": "vraxion_mutation_controller_failed_jobs_present",
            "verdict": "D53_FAILED_JOBS_PRESENT",
            "next": "D53_REPAIR",
            "boundary": BOUNDARY,
        }
    best = best_vraxion_arm(metrics)
    core = metrics["by_arm_core"]
    best_row = core[best]
    corr = regime_accuracy(metrics, best, "CORRELATED_ECHO_SUPPORT")
    adv = regime_accuracy(metrics, best, "ADVERSARIAL_DISTRACTOR_SUPPORT")
    external = regime_accuracy(metrics, best, "EXTERNAL_TEST_REQUIRED_SUPPORT")
    indist = metrics["by_arm_and_regime"][best]["INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT"]
    full = core["HANDCODED_D50_FULL_REFERENCE"]
    random_control = core["RANDOM_POLICY_CONTROL"]
    greedy_control = core["GREEDY_DECIDE_CONTROL"]
    cost_control = core["COST_ONLY_MUTATION_CONTROL"]
    always = core["ALWAYS_COUNTER_CONTROL"]
    if cost_only_passes_safety(metrics):
        return {
            "decision": "vraxion_mutation_controller_not_confirmed",
            "verdict": "D53_COST_ONLY_CONTROL_PASSED",
            "next": "D53R_MUTATION_REPRESENTATION_REPAIR",
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
        if best == "VRAXION_MUTABLE_RULE_TABLE" and sparse_or_pocket_failed(metrics):
            return {
                "decision": "vraxion_integration_partial_rule_table_only",
                "verdict": "D53_PARTIAL_RULE_TABLE_ONLY",
                "next": "D53S_SPARSE_CONTROLLER_REPAIR",
                "best_vraxion_arm": best,
                "boundary": BOUNDARY,
            }
        return {
            "decision": "vraxion_mutable_ecf_controller_integration_positive",
            "verdict": "D53_VRAXION_MUTABLE_ECF_CONTROLLER_INTEGRATION_POSITIVE",
            "next": "D54_VRAXION_MUTABLE_ECF_CONTROLLER_SCALE_CONFIRM",
            "best_vraxion_arm": best,
            "boundary": BOUNDARY,
        }
    if pass_accuracy and controls_worse:
        return {
            "decision": "vraxion_mutable_ecf_controller_positive_high_cost",
            "verdict": "D53_VRAXION_MUTABLE_ECF_CONTROLLER_POSITIVE_HIGH_COST",
            "next": "D53C_SUPPORT_COST_OPTIMIZATION",
            "best_vraxion_arm": best,
            "boundary": BOUNDARY,
        }
    return {
        "decision": "vraxion_mutation_controller_not_confirmed",
        "verdict": "D53_VRAXION_MUTATION_CONTROLLER_NOT_CONFIRMED",
        "next": "D53R_MUTATION_REPRESENTATION_REPAIR",
        "best_vraxion_arm": best,
        "boundary": BOUNDARY,
    }


def make_reports(out, aggregate, decision, d52_manifest, canonical_report):
    test = aggregate["test_metrics"]
    core = test["by_arm_core"]
    best = decision.get("best_vraxion_arm", best_vraxion_arm(test))
    reports = {
        "representation_report.json": {
            "canonical_vraxion_audit": canonical_report,
            "d53_representation_level": "mutable_controller_genome_above_fixed_symbolic_ecf",
            "full_sparse_firing_brain_used": False,
            "formula_solver_learning_used": False,
            "representations": {
                "VRAXION_MUTABLE_RULE_TABLE": "threshold/action-route genome",
                "VRAXION_MUTABLE_SPARSE_GATE_CONTROLLER": "integer sparse feature gates to action scores",
                "VRAXION_MUTABLE_POCKET_STATE_CONTROLLER": "feature pockets with priority and action writeback",
                "VRAXION_MUTABLE_HYBRID_CONTROLLER": "rule-table plus sparse-gate overlay",
            },
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
        "policy_action_distribution_report.json": test["action_distribution"],
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
        "vraxion_integration_boundary_report.json": {
            "boundary": BOUNDARY,
            "d52_caveat_preserved": True,
            "controller_integration_only": True,
            "sparse_firing_used": False,
            "canonical_vraxion_source_audited": canonical_report["source_smoke_passed"],
            "no_formula_solver_learning": True,
            "no_architecture_superiority_claim": True,
        },
        "best_policy_report.json": {
            "best_vraxion_arm": best,
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
    best = decision.get("best_vraxion_arm", best_vraxion_arm(aggregate["test_metrics"]))
    lines = [
        "# D53 Mutable ECF Integration With VRAXION Mutation Architecture Result",
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
    parser.add_argument("--seeds", default="10701,10702,10703,10704,10705")
    parser.add_argument("--train-rows-per-seed", type=int, default=800)
    parser.add_argument("--test-rows-per-seed", type=int, default=800)
    parser.add_argument("--ood-rows-per-seed", type=int, default=800)
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument("--population", type=int, default=80)
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
            "task": "D53 mutable ECF integration with VRAXION mutation architecture",
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
    d52_manifest = make_d52_upstream_manifest(repo_root)
    write_json(out / "d52_upstream_manifest.json", d52_manifest)
    canonical_report = canonical_vraxion_audit(repo_root)
    write_json(out / "canonical_vraxion_smoke_report.json", canonical_report)
    write_json(
        out / "dataset_manifest.json",
        {
            "task": "D53_MUTABLE_ECF_INTEGRATION_WITH_VRAXION_MUTATION_ARCHITECTURE",
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
            "note": "D53 uses deterministic symbolic scoring and mutable controller genome search; no external model/API/download used.",
        },
    )
    d52_replay_policy = load_best_d52_policy(repo_root, d52_manifest)
    failed_jobs = []
    if d52_replay_policy is None:
        failed_jobs.append("missing_best_d52_replay_policy")
    if not canonical_report["source_smoke_passed"] or not canonical_report["action_output_encoding_smoke_passed"]:
        failed_jobs.append("canonical_vraxion_source_smoke_failed")

    train_rows = d51.make_rows_with_progress(seeds, args.train_rows_per_seed, "train", bundle, out, started, args.heartbeat_sec)
    test_rows = d51.make_rows_with_progress(seeds, args.test_rows_per_seed, "test", bundle, out, started, args.heartbeat_sec)
    ood_rows = d51.make_rows_with_progress(seeds, args.ood_rows_per_seed, "ood", bundle, out, started, args.heartbeat_sec)
    append_progress(out, "rows_generated", started, {"train": len(train_rows), "test": len(test_rows), "ood": len(ood_rows)})
    train_packs = d51.build_packs(train_rows, bundle, out, started, args.heartbeat_sec, args.workers, "train")
    train_examples = [
        {"features": pack["features"], "action_compact": pack["action_compact"]}
        for pack in train_packs
    ]
    policies = {"D52_BEST_RULE_TABLE_REPLAY": d52_replay_policy} if d52_replay_policy is not None else {}
    policy_reports = {}
    arm_to_kind = {
        "VRAXION_MUTABLE_RULE_TABLE": "rule_table",
        "VRAXION_MUTABLE_SPARSE_GATE_CONTROLLER": "sparse_gate",
        "VRAXION_MUTABLE_POCKET_STATE_CONTROLLER": "pocket_state",
        "VRAXION_MUTABLE_HYBRID_CONTROLLER": "hybrid",
    }
    for arm, kind in arm_to_kind.items():
        policy, report = train_vraxion_policy(kind, train_examples, args.generations, args.population, 53_000 + d51.stable_seed(arm), out, started, args.heartbeat_sec)
        policies[arm] = policy
        policy_reports[arm] = report
        append_progress(out, "vraxion_mutation_complete", started, {"arm": arm, "fitness": report["fitness"]})
    if d52_replay_policy is not None:
        score, counts = d51.full_policy_score(d52_replay_policy, train_examples)
        policy_reports["D52_BEST_RULE_TABLE_REPLAY"] = {"kind": "d52_replay", "fitness": score, "action_counts": counts, "history": []}
    write_json(out / "trained_policy_manifest.json", {"policies": policies, "reports": policy_reports})

    test_packs = d51.build_packs(test_rows, bundle, out, started, args.heartbeat_sec, args.workers, "test")
    ood_packs = d51.build_packs(ood_rows, bundle, out, started, args.heartbeat_sec, args.workers, "ood")
    test_outputs = evaluate_packs(test_packs, policies, out / "row_outputs_test.jsonl", out, "test", started, args.heartbeat_sec)
    ood_outputs = evaluate_packs(ood_packs, policies, out / "row_outputs_ood.jsonl", out, "ood", started, args.heartbeat_sec)
    test_metrics = summarize_outputs(test_outputs)
    ood_metrics = summarize_outputs(ood_outputs)
    decision = make_decision(test_metrics, failed_jobs)
    aggregate = {
        "task": "D53_MUTABLE_ECF_INTEGRATION_WITH_VRAXION_MUTATION_ARCHITECTURE",
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
    reports = make_reports(out, aggregate, decision, d52_manifest, canonical_report)
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
                "d52_replay": test_metrics["by_arm_core"].get("D52_BEST_RULE_TABLE_REPLAY"),
                "d50_full_reference": test_metrics["by_arm_core"]["HANDCODED_D50_FULL_REFERENCE"],
                "always_counter": test_metrics["by_arm_core"]["ALWAYS_COUNTER_CONTROL"],
                "support_cost_frontier": reports["support_cost_frontier_report.json"],
                "action_distribution": test_metrics["action_distribution"][aggregate["best_vraxion_arm"]],
                "min_seed_gates": reports["min_seed_gate_report.json"][aggregate["best_vraxion_arm"]],
                "canonical_vraxion_smoke": canonical_report,
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
