#!/usr/bin/env python3
"""D68R concrete counter-action routing repair.

D68A showed that D68 did not merely request "too much" support. D68 reduced
support cost but sometimes selected `REQUEST_COUNTER_TOP1_TOP2` where
`REQUEST_JOINT_COUNTER` was the concrete action that fixed the row. D68R trains
a fair feature-based router for that concrete action choice and evaluates it
against D67/D68 replays and controls.
"""

import argparse
import copy
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import run_d51_mutable_ecf_controller_prototype as d51
import run_d55_sparse_firing_ecf_controller_prototype as d55
import run_d59_rust_sparse_ecf_controller_with_mutation as d59
import run_d65_set_invariant_ipf_aggregation_prototype as d65
import run_d68_counter_support_triage_repair as d68
import run_d68a_counter_support_metric_semantics_audit as d68a

TASK = "D68R_CONCRETE_COUNTER_ACTION_ROUTING_REPAIR"
BOUNDARY = (
    "D68R only tests concrete counter-action routing repair for controlled "
    "symbolic joint formula discovery. It does not prove full VRAXION brain, "
    "raw visual Raven reasoning, Raven solved, AGI, consciousness, DNA/genome "
    "success, architecture superiority, or production readiness."
)

SUPPORT_COUNT = d68.SUPPORT_COUNT
REGIMES = d68.REGIMES
CORE_REGIMES = d68.CORE_REGIMES
COUNTER_ACTIONS = d68.COUNTER_ACTIONS
INTERNAL_COUNTER_ACTIONS = d68.INTERNAL_COUNTER_ACTIONS
EXTERNAL_ACTION = d68.EXTERNAL_ACTION

ARMS = [
    "D67_BEST_REPLAY",
    "D68_TRAINED_THRESHOLD_REPLAY",
    "D68R_CONCRETE_ROUTER",
    "D68R_CONCRETE_ROUTER_COST_WEIGHTED",
    "D68R_CONSERVATIVE_JOINT_REPAIR",
    "TOP1_ONLY_CONTROL",
    "JOINT_ONLY_CONTROL",
    "NEVER_COUNTER_CONTROL",
    "RANDOM_COUNTER_CONTROL",
    "SHUFFLED_ROUTER_CONTROL",
    "CONCRETE_COUNTER_ORACLE_REFERENCE_ONLY",
    "CHEAPEST_CORRECT_ORACLE_REFERENCE_ONLY",
]

REFERENCE_ONLY_ARMS = [
    "CONCRETE_COUNTER_ORACLE_REFERENCE_ONLY",
    "CHEAPEST_CORRECT_ORACLE_REFERENCE_ONLY",
]
CONTROL_ARMS = [
    "TOP1_ONLY_CONTROL",
    "JOINT_ONLY_CONTROL",
    "NEVER_COUNTER_CONTROL",
    "RANDOM_COUNTER_CONTROL",
    "SHUFFLED_ROUTER_CONTROL",
]
FAIR_ARMS = [arm for arm in ARMS if arm not in REFERENCE_ONLY_ARMS]

SOURCE_FOR_ARM = {
    "D67_BEST_REPLAY": "RUST_SPARSE_SET_INVARIANT_IPF_AGGREGATION",
    "D68_TRAINED_THRESHOLD_REPLAY": "RUST_SPARSE_COUNTERFACTUAL_DELTA_SET_AGGREGATION",
    "D68R_CONCRETE_ROUTER": "RUST_SPARSE_COUNTERFACTUAL_DELTA_SET_AGGREGATION",
    "D68R_CONCRETE_ROUTER_COST_WEIGHTED": "RUST_SPARSE_COUNTERFACTUAL_DELTA_SET_AGGREGATION",
    "D68R_CONSERVATIVE_JOINT_REPAIR": "RUST_SPARSE_COUNTERFACTUAL_DELTA_SET_AGGREGATION",
    "TOP1_ONLY_CONTROL": "RUST_SPARSE_COUNTERFACTUAL_DELTA_SET_AGGREGATION",
    "JOINT_ONLY_CONTROL": "RUST_SPARSE_COUNTERFACTUAL_DELTA_SET_AGGREGATION",
    "NEVER_COUNTER_CONTROL": "RUST_SPARSE_COUNTERFACTUAL_DELTA_SET_AGGREGATION",
    "RANDOM_COUNTER_CONTROL": "RUST_SPARSE_COUNTERFACTUAL_DELTA_SET_AGGREGATION",
    "SHUFFLED_ROUTER_CONTROL": "RUST_SPARSE_COUNTERFACTUAL_DELTA_SET_AGGREGATION",
    "CONCRETE_COUNTER_ORACLE_REFERENCE_ONLY": "RUST_SPARSE_COUNTERFACTUAL_DELTA_SET_AGGREGATION",
    "CHEAPEST_CORRECT_ORACLE_REFERENCE_ONLY": "RUST_SPARSE_COUNTERFACTUAL_DELTA_SET_AGGREGATION",
}

ROW_SAMPLE_PER_ARM_REGIME = 14


write_json = d68a.write_json
append_jsonl = d68a.append_jsonl
append_progress = d68a.append_progress
load_json = d68a.load_json
parse_seeds = d68a.parse_seeds
effective_correct = d68a.effective_correct
action_support = d68a.action_support
cheapest_effective_action = d68a.cheapest_effective_action


def policy_from_action(action):
    return d68a.policy_from_action(action)


def router_feature_score(features, config):
    values = [float(features.get(name, 0.0)) for name in config.get("joint_features", [])]
    if not values:
        return 0.0
    mode = config.get("score_mode", "max")
    if mode == "mean":
        return sum(values) / len(values)
    if mode == "weighted":
        return (
            0.42 * float(features.get("adversarial_pressure_norm", 0.0))
            + 0.22 * float(features.get("counterfactual_pressure_norm", 0.0))
            + 0.18 * float(features.get("inverse_margin", 0.0))
            + 0.10 * float(features.get("entropy_norm", 0.0))
            + 0.08 * float(features.get("dominant_cluster_fraction", 0.0))
        )
    return max(values)


def concrete_router_policy(pack, config, basis_prefix):
    features = d68.d68_gate_features(pack)
    preflight, reason = d68.triage_preflight(features)
    if preflight:
        return preflight, d68.action_for_policy_name(preflight), features, f"{basis_prefix}_{reason}", False

    risk = d68.triage_risk(features)
    request = risk >= float(config.get("risk_threshold", 0.64))
    request = request or (
        float(features.get("counterfactual_pressure_norm", 0.0)) >= float(config.get("force_request_threshold", 1.1))
    )
    if not request:
        return "DECIDE_POLICY", "DECIDE", features, f"{basis_prefix}_risk_decide", False

    joint_score = router_feature_score(features, config)
    collision_ok = float(features.get("collision_norm", 0.0)) >= float(config.get("collision_min", 0.0))
    inverse_ok = float(features.get("inverse_margin", 0.0)) >= float(config.get("inverse_margin_min", 0.0))
    echo_ok = float(features.get("dominant_cluster_fraction", 0.0)) >= float(config.get("echo_min", 0.0))
    joint = joint_score >= float(config.get("joint_threshold", 0.66)) and collision_ok and inverse_ok and echo_ok
    if config.get("mode") == "always_joint_when_requested":
        joint = True
    if config.get("mode") == "original_d68":
        joint = (
            float(features.get("adversarial_pressure_norm", 0.0)) >= float(config.get("joint_threshold", 0.66))
            and float(features.get("collision_norm", 0.0)) >= float(config.get("collision_min", 0.22))
        )
    if config.get("mode") == "shuffled_inverse":
        joint = joint_score <= float(config.get("joint_threshold", 0.66))

    if joint:
        return "ADVERSARIAL_REPAIR_POLICY", "REQUEST_JOINT_COUNTER", features, f"{basis_prefix}_joint_score_{joint_score:.3f}", False
    return "COUNTERFACTUAL_POLICY", "REQUEST_COUNTER_TOP1_TOP2", features, f"{basis_prefix}_top1_score_{joint_score:.3f}", False


def deterministic_random_policy(pack):
    policy = d68.deterministic_random_counter(pack)
    return policy, d68.action_for_policy_name(policy), d68.d68_gate_features(pack), "deterministic_random_counter_control", False


def select_policy(arm, pack, learned_triage, routers):
    if arm == "D67_BEST_REPLAY":
        policy, features, basis = d68.support_scoring_policy(pack)
        return policy, d68.action_for_policy_name(policy), features, basis, False
    if arm == "D68_TRAINED_THRESHOLD_REPLAY":
        policy, features, basis = d68.counter_support_triage_policy(pack, "TRAINED_THRESHOLD_TRIAGE_GATE", learned_triage)
        return policy, d68.action_for_policy_name(policy), features, basis, False
    if arm == "D68R_CONCRETE_ROUTER":
        return concrete_router_policy(pack, routers["accuracy_first"], "d68r_accuracy_first")
    if arm == "D68R_CONCRETE_ROUTER_COST_WEIGHTED":
        return concrete_router_policy(pack, routers["cost_weighted"], "d68r_cost_weighted")
    if arm == "D68R_CONSERVATIVE_JOINT_REPAIR":
        config = {
            "mode": "conservative_joint",
            "risk_threshold": 0.64,
            "joint_threshold": 0.64,
            "joint_features": ["adversarial_pressure_norm"],
            "score_mode": "max",
            "collision_min": 0.0,
            "inverse_margin_min": 0.0,
            "echo_min": 0.0,
        }
        return concrete_router_policy(pack, config, "d68r_conservative_joint")
    if arm == "TOP1_ONLY_CONTROL":
        features = d68.d68_gate_features(pack)
        preflight, reason = d68.triage_preflight(features)
        if preflight:
            return preflight, d68.action_for_policy_name(preflight), features, f"top1_only_keeps_{reason}", False
        if d68.triage_risk(features) >= float(learned_triage.get("risk_threshold", 0.64)):
            return "COUNTERFACTUAL_POLICY", "REQUEST_COUNTER_TOP1_TOP2", features, "forced_top1_counter", False
        return "DECIDE_POLICY", "DECIDE", features, "top1_only_decide", False
    if arm == "JOINT_ONLY_CONTROL":
        features = d68.d68_gate_features(pack)
        preflight, reason = d68.triage_preflight(features)
        if preflight:
            return preflight, d68.action_for_policy_name(preflight), features, f"joint_only_keeps_{reason}", False
        if d68.triage_risk(features) >= float(learned_triage.get("risk_threshold", 0.64)):
            return "ADVERSARIAL_REPAIR_POLICY", "REQUEST_JOINT_COUNTER", features, "forced_joint_counter", False
        return "DECIDE_POLICY", "DECIDE", features, "joint_only_decide", False
    if arm == "NEVER_COUNTER_CONTROL":
        features = d68.d68_gate_features(pack)
        preflight, reason = d68.triage_preflight(features)
        if preflight:
            return preflight, d68.action_for_policy_name(preflight), features, f"never_counter_keeps_{reason}", False
        return "DECIDE_POLICY", "DECIDE", features, "forced_never_counter", False
    if arm == "RANDOM_COUNTER_CONTROL":
        return deterministic_random_policy(pack)
    if arm == "SHUFFLED_ROUTER_CONTROL":
        config = dict(routers["accuracy_first"])
        config["mode"] = "shuffled_inverse"
        return concrete_router_policy(pack, config, "shuffled_router_control")
    if arm == "CONCRETE_COUNTER_ORACLE_REFERENCE_ONLY":
        action, _, basis, used_truth = d68a.select_oracle_action(pack, "concrete_counter")
        return policy_from_action(action), action, d68.d68_gate_features(pack), basis, used_truth
    if arm == "CHEAPEST_CORRECT_ORACLE_REFERENCE_ONLY":
        action, _, basis, used_truth = d68a.select_oracle_action(pack, "cheapest")
        return policy_from_action(action), action, d68.d68_gate_features(pack), basis, used_truth
    raise KeyError(f"unknown arm {arm}")


def decorate_pack(pack, arm, source):
    out = copy.deepcopy(pack)
    out["d68r_arm"] = arm
    out["d65_source_arm"] = source
    out["support_budget_cap"] = None
    out["support_scoring_used"] = arm in {
        "D67_BEST_REPLAY",
        "D68_TRAINED_THRESHOLD_REPLAY",
        "D68R_CONCRETE_ROUTER",
        "D68R_CONCRETE_ROUTER_COST_WEIGHTED",
        "D68R_CONSERVATIVE_JOINT_REPAIR",
    }
    return out


def build_items(rows, bundle, rust_features, out, started, heartbeat_sec, split):
    items = []
    total = len(rows) * len(ARMS)
    completed = 0
    last = time.time()
    for idx, row in enumerate(rows):
        source_cache = {}
        for arm in ARMS:
            source = SOURCE_FOR_ARM[arm]
            if source not in source_cache:
                source_cache[source] = d65.build_pack(row, bundle, source, idx, rust_features)
            items.append({"arm": arm, "pack": decorate_pack(source_cache[source], arm, source)})
            completed += 1
        now = time.time()
        if now - last >= heartbeat_sec:
            last = now
            write_json(out / f"partial_{split}_d68r_pack_build.json", {"completed": completed, "total": total})
            append_progress(out, "d68r_pack_build_progress", started, {"split": split, "completed": completed, "total": total})
    write_json(out / f"partial_{split}_d68r_pack_build.json", {"completed": total, "total": total})
    append_progress(out, "d68r_pack_build_complete", started, {"split": split, "packs": len(items)})
    return items


def build_training_packs(rows, bundle, out, repo_root, started, heartbeat_sec):
    rust_features, aggregation_report = d68.run_blocking_with_heartbeat(
        out,
        "d68r_train_rust_aggregation_bridge_wait",
        "train",
        started,
        heartbeat_sec,
        d65.run_rust_aggregation_bridge,
        out,
        repo_root,
        rows,
        bundle,
        "train",
        started,
        heartbeat_sec,
    )
    packs = []
    last = time.time()
    for idx, row in enumerate(rows):
        pack = d65.build_pack(row, bundle, "RUST_SPARSE_COUNTERFACTUAL_DELTA_SET_AGGREGATION", idx, rust_features)
        packs.append(pack)
        now = time.time()
        if now - last >= heartbeat_sec:
            last = now
            write_json(out / "partial_train_concrete_router_pack_build.json", {"completed": idx + 1, "total": len(rows)})
            append_progress(out, "train_concrete_router_pack_build", started, {"completed": idx + 1, "total": len(rows)})
    write_json(out / "partial_train_concrete_router_pack_build.json", {"completed": len(rows), "total": len(rows)})
    return packs, aggregation_report


def config_candidates():
    feature_sets = [
        ("adv_only", ["adversarial_pressure_norm"], "max"),
        ("pressure_any", ["adversarial_pressure_norm", "counterfactual_pressure_norm"], "max"),
        ("pressure_or_margin", ["adversarial_pressure_norm", "counterfactual_pressure_norm", "inverse_margin"], "max"),
        ("broad_max", ["adversarial_pressure_norm", "counterfactual_pressure_norm", "inverse_margin", "entropy_norm", "dominant_cluster_fraction"], "max"),
        ("weighted", ["adversarial_pressure_norm", "counterfactual_pressure_norm", "inverse_margin", "entropy_norm", "dominant_cluster_fraction"], "weighted"),
        ("original_d68", ["adversarial_pressure_norm"], "max"),
        ("always_joint_when_requested", ["adversarial_pressure_norm"], "max"),
    ]
    risk_thresholds = [0.56, 0.60, 0.64, 0.68, 0.72]
    joint_thresholds = [0.48, 0.54, 0.60, 0.64, 0.66, 0.70, 0.74, 0.78]
    collision_mins = [0.0, 0.10, 0.18, 0.22]
    inverse_mins = [0.0, 0.25, 0.50]
    echo_mins = [0.0, 0.58, 0.82]
    out = []
    for mode, features, score_mode in feature_sets:
        for risk in risk_thresholds:
            for joint in joint_thresholds:
                for collision in collision_mins:
                    if mode in {"adv_only", "pressure_any", "pressure_or_margin", "broad_max", "weighted", "always_joint_when_requested"} and collision != 0.0:
                        continue
                    for inverse in inverse_mins:
                        if mode in {"adv_only", "pressure_any", "weighted", "always_joint_when_requested"} and inverse != 0.0:
                            continue
                        for echo in echo_mins:
                            if mode not in {"broad_max"} and echo != 0.0:
                                continue
                            out.append(
                                {
                                    "mode": mode,
                                    "risk_threshold": risk,
                                    "joint_threshold": joint,
                                    "joint_features": features,
                                    "score_mode": score_mode,
                                    "collision_min": collision,
                                    "inverse_margin_min": inverse,
                                    "echo_min": echo,
                                }
                            )
    return out


def training_rows_for_config(packs, config, label):
    rows = []
    for pack in packs:
        policy, action, features, basis, _ = concrete_router_policy(pack, config, label)
        audit = d68a.concrete_action_audit(pack, action)
        rows.append(
            {
                "arm": label,
                "seed": pack["seed"],
                "support_regime": pack["support_regime"],
                "selected_action": action,
                "exact_joint_correct": audit["selected_action_effective_correct"],
                "effective_correct": audit["selected_action_effective_correct"],
                "correct": audit["selected_action_effective_correct"],
                "total_support_used": audit["selected_support_used"],
                "selected_support_used": audit["selected_support_used"],
                "selected_counter_support_used": audit["selected_counter_support_used"],
                "selected_external_test_used": audit["selected_external_test_used"],
                "false_confidence": False,
                "abstained": action == "ABSTAIN",
                "concrete_selected_counter_missed": audit["concrete_selected_counter_missed"],
                "wrong_concrete_counter": audit["wrong_concrete_counter"],
                "weak_top1_top2_path_failure": audit["weak_top1_top2_path_failure"],
                "causal_unnecessary_counter_support": audit["causal_unnecessary_counter_support"],
                "selected_concrete_counter_fixes": audit["selected_concrete_counter_fixes"],
                "support_over_cheapest_effective": audit["support_over_cheapest_effective"],
                "gate_features": features,
                "gate_basis": basis,
                "gate_used_truth_label": False,
            }
        )
    return rows


def summarize_training_rows(rows):
    return {
        "rows": len(rows),
        "exact_joint_accuracy": d51.mean([1.0 if row["exact_joint_correct"] else 0.0 for row in rows]),
        "effective_accuracy": d51.mean([1.0 if row["effective_correct"] else 0.0 for row in rows]),
        "average_total_support_used": d51.mean([row["total_support_used"] for row in rows]),
        "selected_counter_support_used_mean": d51.mean([row["selected_counter_support_used"] for row in rows]),
        "selected_external_test_used_mean": d51.mean([row["selected_external_test_used"] for row in rows]),
        "causal_unnecessary_counter_support_rate": d51.mean([1.0 if row["causal_unnecessary_counter_support"] else 0.0 for row in rows]),
        "concrete_selected_counter_missed_rate": d51.mean([1.0 if row["concrete_selected_counter_missed"] else 0.0 for row in rows]),
        "wrong_concrete_counter_rate": d51.mean([1.0 if row["wrong_concrete_counter"] else 0.0 for row in rows]),
        "weak_top1_top2_path_failure_rate": d51.mean([1.0 if row["weak_top1_top2_path_failure"] else 0.0 for row in rows]),
        "selected_concrete_counter_fixes_rate": d51.mean([1.0 if row["selected_concrete_counter_fixes"] else 0.0 for row in rows]),
        "support_over_cheapest_effective_mean": d51.mean([row["support_over_cheapest_effective"] for row in rows]),
        "abstain_rate": d51.mean([1.0 if row["abstained"] else 0.0 for row in rows]),
    }


def score_accuracy_first(summary):
    exact = summary["exact_joint_accuracy"]
    support = summary["average_total_support_used"]
    wrong = summary["wrong_concrete_counter_rate"]
    weak = summary["weak_top1_top2_path_failure_rate"]
    unnecessary = summary["causal_unnecessary_counter_support_rate"]
    return (
        exact >= 0.999,
        exact - 4.0 * wrong - 4.0 * weak - 0.05 * unnecessary,
        -support,
    )


def score_cost_weighted(summary):
    exact = summary["exact_joint_accuracy"]
    support = summary["average_total_support_used"]
    wrong = summary["wrong_concrete_counter_rate"]
    weak = summary["weak_top1_top2_path_failure_rate"]
    unnecessary = summary["causal_unnecessary_counter_support_rate"]
    return exact - 2.5 * wrong - 2.5 * weak - 0.0025 * max(0.0, support - SUPPORT_COUNT) - 0.02 * unnecessary


def train_concrete_router(train_packs, out, started, heartbeat_sec):
    candidates = []
    configs = config_candidates()
    last = time.time()
    for idx, config in enumerate(configs):
        rows = training_rows_for_config(train_packs, config, "TRAIN_CONFIG")
        summary = summarize_training_rows(rows)
        candidates.append(
            {
                "config": config,
                "summary": summary,
                "accuracy_first_score": score_accuracy_first(summary),
                "cost_weighted_score": score_cost_weighted(summary),
            }
        )
        now = time.time()
        if now - last >= heartbeat_sec:
            last = now
            write_json(out / "partial_train_concrete_router_search.json", {"completed": idx + 1, "total": len(configs)})
            append_progress(out, "train_concrete_router_search", started, {"completed": idx + 1, "total": len(configs)})

    accuracy_sorted = sorted(candidates, key=lambda item: item["accuracy_first_score"], reverse=True)
    cost_sorted = sorted(candidates, key=lambda item: item["cost_weighted_score"], reverse=True)
    report = {
        "candidate_count": len(candidates),
        "selected_accuracy_first": accuracy_sorted[0],
        "selected_cost_weighted": cost_sorted[0],
        "top_accuracy_first": accuracy_sorted[:20],
        "top_cost_weighted": cost_sorted[:20],
        "feature_source": "D68 fair gate features only; labels used only to select the offline config",
    }
    write_json(out / "concrete_router_training_report.json", report)
    append_progress(
        out,
        "train_concrete_router_complete",
        started,
        {
            "accuracy_first": report["selected_accuracy_first"]["config"],
            "cost_weighted": report["selected_cost_weighted"]["config"],
        },
    )
    return {
        "accuracy_first": report["selected_accuracy_first"]["config"],
        "cost_weighted": report["selected_cost_weighted"]["config"],
    }, report


def record_row(row, outputs, sample_counts, path):
    outputs.append(row)
    key = (row["arm"], row["support_regime"])
    if sample_counts[key] < ROW_SAMPLE_PER_ARM_REGIME:
        append_jsonl(path, row)
        sample_counts[key] += 1


def summarize_audit_rows(rows):
    base = d68a.summarize_audit_rows(rows)
    base["wrong_concrete_counter_rate"] = d51.mean([1.0 if row.get("wrong_concrete_counter") else 0.0 for row in rows])
    base["weak_top1_top2_path_failure_rate"] = d51.mean([1.0 if row.get("weak_top1_top2_path_failure") else 0.0 for row in rows])
    base["concrete_selected_counter_missed_rate"] = d51.mean([1.0 if row.get("concrete_selected_counter_missed") else 0.0 for row in rows])
    base["selected_concrete_counter_fixes_rate"] = d51.mean([1.0 if row.get("selected_concrete_counter_fixes") else 0.0 for row in rows])
    base["causal_unnecessary_counter_support_rate"] = d51.mean([1.0 if row.get("causal_unnecessary_counter_support") else 0.0 for row in rows])
    base["support_over_cheapest_effective_mean"] = d51.mean([row.get("support_over_cheapest_effective", 0.0) for row in rows])
    return base


def summarize_outputs(outputs):
    by_arm = defaultdict(list)
    by_arm_core = defaultdict(list)
    by_arm_regime = defaultdict(list)
    action_counts = defaultdict(Counter)
    rust_usage = defaultdict(Counter)
    for row in outputs:
        arm = row["arm"]
        by_arm[arm].append(row)
        by_arm_regime[(arm, row["support_regime"])].append(row)
        action_counts[arm][row["selected_action"]] += 1
        rust_usage[arm]["rows"] += 1
        if row.get("rust_network_path_invoked"):
            rust_usage[arm]["controller_rust_rows"] += 1
        if row.get("rust_aggregation_used"):
            rust_usage[arm]["aggregation_rust_rows"] += 1
        if row.get("python_fallback_used"):
            rust_usage[arm]["python_fallback_rows"] += 1
        if row["support_regime"] in CORE_REGIMES:
            by_arm_core[arm].append(row)
    return {
        "by_arm": {arm: summarize_audit_rows(by_arm[arm]) for arm in ARMS if arm in by_arm},
        "by_arm_core": {arm: summarize_audit_rows(by_arm_core[arm]) for arm in ARMS if arm in by_arm_core},
        "by_arm_and_regime": {
            arm: {regime: summarize_audit_rows(by_arm_regime[(arm, regime)]) for regime in REGIMES if (arm, regime) in by_arm_regime}
            for arm in ARMS
        },
        "action_distribution": {arm: dict(action_counts[arm]) for arm in ARMS},
        "rust_usage": {arm: dict(rust_usage[arm]) for arm in ARMS},
    }


def write_partial(out, split, outputs, completed, started):
    metrics = summarize_outputs(outputs)
    write_json(
        out / f"partial_{split}_d68r_metrics_snapshot.json",
        {
            "completed_outputs": completed,
            "elapsed_sec": time.time() - started,
            "by_arm_core": metrics["by_arm_core"],
            "action_distribution": metrics["action_distribution"],
        },
    )


def evaluate_split(rows, bundle, policy_controllers, learned_triage, routers, out, split, started, heartbeat_sec, repo_root, row_output_path):
    rust_features, aggregation_report = d68.run_blocking_with_heartbeat(
        out,
        "d68r_rust_aggregation_bridge_wait",
        split,
        started,
        heartbeat_sec,
        d65.run_rust_aggregation_bridge,
        out,
        repo_root,
        rows,
        bundle,
        split,
        started,
        heartbeat_sec,
    )
    items = build_items(rows, bundle, rust_features, out, started, heartbeat_sec, split)
    packs = [item["pack"] for item in items]
    policy_actions, policy_report = d68.run_blocking_with_heartbeat(
        out,
        "d68r_rust_policy_bridge_wait",
        split,
        started,
        heartbeat_sec,
        d59.run_rust_multi_bridge,
        out,
        repo_root,
        policy_controllers,
        packs,
        split,
        "d68r_policy_eval",
        started,
    )

    outputs = []
    sample_counts = Counter()
    last = 0.0
    for idx, item in enumerate(items):
        arm = item["arm"]
        pack = item["pack"]
        policy, selected_action, gate_features, basis, used_truth = select_policy(arm, pack, learned_triage, routers)
        action_record = policy_actions[policy][idx]
        if action_record["action"] != selected_action:
            action_record = copy.deepcopy(action_record)
            action_record["action"] = selected_action
        row = d68a.make_output_row(pack, arm, policy, selected_action, action_record, gate_features, basis, used_truth)
        row["d68r_arm"] = arm
        row["fair_arm"] = arm in FAIR_ARMS
        row["control_arm"] = arm in CONTROL_ARMS
        row["reference_only_arm"] = arm in REFERENCE_ONLY_ARMS
        record_row(row, outputs, sample_counts, row_output_path)
        now = time.time()
        if now - last >= heartbeat_sec or len(outputs) == len(items):
            last = now
            append_progress(out, "d68r_eval_progress", started, {"split": split, "completed_outputs": len(outputs)})
            write_partial(out, split, outputs, len(outputs), started)
    return outputs, {"aggregation": aggregation_report, "controller": policy_report}


def load_upstream_trained_triage(d68_root):
    return d68a.load_upstream_trained_triage(d68_root)


def d68a_upstream_manifest(d68a_root):
    return {
        "artifact_root": str(d68a_root),
        "decision": load_json(d68a_root / "decision.json"),
        "summary": load_json(d68a_root / "summary.json"),
        "concrete_counter_action_report": load_json(d68a_root / "concrete_counter_action_report.json"),
        "d68_harm_classification_report": load_json(d68a_root / "d68_harm_classification_report.json"),
    }


def harm_repair_report(outputs):
    by_row = defaultdict(dict)
    for row in outputs:
        by_row[row["row_id"]][row["arm"]] = row
    counts = Counter()
    examples = []
    total_d68_losses = 0
    repaired = 0
    for row_id, rows in by_row.items():
        d67 = rows.get("D67_BEST_REPLAY")
        d68r = rows.get("D68_TRAINED_THRESHOLD_REPLAY")
        repair = rows.get("D68R_CONCRETE_ROUTER")
        if not d67 or not d68r or not repair:
            continue
        if d67["effective_correct"] and not d68r["effective_correct"]:
            total_d68_losses += 1
            if repair["effective_correct"]:
                repaired += 1
                key = "d68_loss_repaired"
            else:
                key = "d68_loss_not_repaired"
            counts[key] += 1
            if len(examples) < 40:
                examples.append(
                    {
                        "row_id": row_id,
                        "support_regime": repair["support_regime"],
                        "d67_action": d67["selected_action"],
                        "d68_action": d68r["selected_action"],
                        "d68r_action": repair["selected_action"],
                        "top1_effective": repair["request_top1_top2_effective"],
                        "joint_effective": repair["request_joint_counter_effective"],
                        "repaired": repair["effective_correct"],
                    }
                )
    return {
        "d68_loss_rows_vs_d67": total_d68_losses,
        "d68_loss_rows_repaired_by_d68r": repaired,
        "d68_loss_repair_rate": repaired / max(1, total_d68_losses),
        "classification_counts": dict(counts),
        "examples": examples,
    }


def support_cost_frontier(metrics):
    core = metrics["by_arm_core"]
    return {
        arm: {
            "exact": values.get("exact_joint_accuracy"),
            "support": values.get("average_total_support_used"),
            "wrong_concrete_counter": values.get("wrong_concrete_counter_rate"),
            "weak_top1_top2_failure": values.get("weak_top1_top2_path_failure_rate"),
            "causal_unnecessary": values.get("causal_unnecessary_counter_support_rate"),
            "support_over_cheapest": values.get("support_over_cheapest_effective_mean"),
        }
        for arm, values in core.items()
    }


def control_report():
    return {
        "fair_arms": FAIR_ARMS,
        "reference_only_arms": REFERENCE_ONLY_ARMS,
        "control_arms": CONTROL_ARMS,
        "fair_arms_using_truth_label": [],
        "fair_arms_using_regime_label": [],
        "truth_hidden_from_fair_arms": True,
        "oracle_labels_reference_only": True,
        "forbidden_features": sorted(d68.FORBIDDEN_TRIAGE_FEATURES),
        "allowed_features": d68.D68_ALLOWED_TRIAGE_FEATURES,
    }


def make_decision(aggregate):
    core = aggregate["test_metrics"]["by_arm_core"]
    d67 = core.get("D67_BEST_REPLAY", {})
    d68r = core.get("D68R_CONCRETE_ROUTER", {})
    d68 = core.get("D68_TRAINED_THRESHOLD_REPLAY", {})
    random_control = core.get("RANDOM_COUNTER_CONTROL", {})
    never_control = core.get("NEVER_COUNTER_CONTROL", {})
    shuffled = core.get("SHUFFLED_ROUTER_CONTROL", {})
    failed = aggregate["failed_jobs"]
    if failed:
        decision = "concrete_counter_action_routing_repair_not_confirmed"
        verdict = "D68R_FAILED_JOBS"
        next_step = "D68R_REPAIR"
    else:
        exact = d68r.get("exact_joint_accuracy", 0.0)
        support = d68r.get("average_total_support_used", 99.0)
        d67_support = d67.get("average_total_support_used", 0.0)
        wrong = d68r.get("wrong_concrete_counter_rate", 1.0)
        weak = d68r.get("weak_top1_top2_path_failure_rate", 1.0)
        controls_worse = (
            random_control.get("exact_joint_accuracy", 1.0) < exact
            and never_control.get("exact_joint_accuracy", 1.0) < exact
            and shuffled.get("exact_joint_accuracy", 1.0) < exact
        )
        if exact >= 0.999 and wrong <= 0.001 and weak <= 0.001 and support < d67_support and controls_worse:
            decision = "concrete_counter_action_routing_repair_positive"
            verdict = "D68R_CONCRETE_COUNTER_ACTION_ROUTING_REPAIR_POSITIVE"
            next_step = "D69_CONCRETE_COUNTER_ROUTING_SCALE_CONFIRM"
        elif exact >= 0.999 and wrong <= 0.001 and weak <= 0.001:
            decision = "concrete_counter_action_routing_repair_positive_high_cost"
            verdict = "D68R_CONCRETE_COUNTER_ACTION_ROUTING_REPAIR_HIGH_COST"
            next_step = "D68C_SUPPORT_COST_OPTIMIZATION"
        elif exact > d68.get("exact_joint_accuracy", 0.0) and weak < d68.get("weak_top1_top2_path_failure_rate", 1.0):
            decision = "concrete_counter_action_routing_repair_partial"
            verdict = "D68R_CONCRETE_COUNTER_ACTION_ROUTING_REPAIR_PARTIAL"
            next_step = "D68R2_ROUTER_FEATURE_REPAIR"
        else:
            decision = "concrete_counter_action_routing_repair_not_confirmed"
            verdict = "D68R_CONCRETE_COUNTER_ACTION_ROUTING_REPAIR_NOT_CONFIRMED"
            next_step = "D68R_REPAIR"
    return {
        "decision": decision,
        "verdict": verdict,
        "next": next_step,
        "reason": {
            "d67_exact": d67.get("exact_joint_accuracy"),
            "d67_support": d67.get("average_total_support_used"),
            "d68_exact": d68.get("exact_joint_accuracy"),
            "d68_support": d68.get("average_total_support_used"),
            "d68r_exact": d68r.get("exact_joint_accuracy"),
            "d68r_support": d68r.get("average_total_support_used"),
            "d68r_wrong_concrete_counter": d68r.get("wrong_concrete_counter_rate"),
            "d68r_weak_top1_top2_failure": d68r.get("weak_top1_top2_path_failure_rate"),
            "d68r_causal_unnecessary": d68r.get("causal_unnecessary_counter_support_rate"),
        },
    }


def write_reports(out, aggregate, decision):
    test = aggregate["test_metrics"]
    core = test["by_arm_core"]
    reports = {
        "d68a_upstream_manifest.json": aggregate["d68a_upstream_manifest"],
        "concrete_router_training_report.json": aggregate["concrete_router_training_report"],
        "concrete_action_routing_report.json": {
            arm: {
                key: values.get(key)
                for key in [
                    "exact_joint_accuracy",
                    "effective_accuracy",
                    "average_total_support_used",
                    "selected_counter_support_used_mean",
                    "causal_unnecessary_counter_support_rate",
                    "concrete_selected_counter_missed_rate",
                    "wrong_concrete_counter_rate",
                    "weak_top1_top2_path_failure_rate",
                    "selected_concrete_counter_fixes_rate",
                    "support_over_cheapest_effective_mean",
                    "false_confidence_rate",
                    "abstain_rate",
                ]
            }
            for arm, values in core.items()
        },
        "top1_vs_joint_counter_report.json": {
            "action_distribution": test["action_distribution"],
            "d68r_harm_repair": aggregate["d68r_harm_repair_report"],
            "test_regime_metrics": test["by_arm_and_regime"].get("D68R_CONCRETE_ROUTER", {}),
        },
        "support_cost_frontier_report.json": support_cost_frontier(test),
        "d68r_harm_repair_report.json": aggregate["d68r_harm_repair_report"],
        "control_report.json": control_report(),
        "regime_breakdown_report.json": test["by_arm_and_regime"],
        "aggregate_metrics.json": aggregate,
        "decision.json": decision,
        "summary.json": {
            "task": TASK,
            "decision": decision["decision"],
            "verdict": decision["verdict"],
            "next": decision["next"],
            "artifact_root": str(out),
            "failed_jobs": aggregate["failed_jobs"],
            "fallback_rows": aggregate["fallback_rows"],
        },
    }
    for name, value in reports.items():
        write_json(out / name, value)

    rows = [
        "# D68R Concrete Counter-Action Routing Repair Report",
        "",
        f"Decision: `{decision['decision']}`",
        f"Verdict: `{decision['verdict']}`",
        f"Next: `{decision['next']}`",
        "",
        "## Core Comparison",
        "",
        "| arm | exact | support | wrong concrete | weak top1 | causal unnecessary |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in ["D67_BEST_REPLAY", "D68_TRAINED_THRESHOLD_REPLAY", "D68R_CONCRETE_ROUTER", "D68R_CONCRETE_ROUTER_COST_WEIGHTED", "TOP1_ONLY_CONTROL", "JOINT_ONLY_CONTROL"]:
        values = core.get(arm, {})
        rows.append(
            f"| {arm} | {values.get('exact_joint_accuracy', 0.0):.6f} | "
            f"{values.get('average_total_support_used', 0.0):.4f} | "
            f"{values.get('wrong_concrete_counter_rate', 0.0):.6f} | "
            f"{values.get('weak_top1_top2_path_failure_rate', 0.0):.6f} | "
            f"{values.get('causal_unnecessary_counter_support_rate', 0.0):.6f} |"
        )
    rows.extend(["", "## Boundary", "", BOUNDARY])
    (out / "report.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--d68-root", default="target/pilot_wave/d68_counter_support_triage_repair/smoke")
    parser.add_argument("--d68a-root", default="target/pilot_wave/d68a_counter_support_metric_semantics_audit/smoke")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--train-rows-per-seed", type=int, default=0)
    parser.add_argument("--test-rows-per-seed", type=int, default=0)
    parser.add_argument("--ood-rows-per-seed", type=int, default=0)
    parser.add_argument("--workers", default="auto")
    parser.add_argument("--cpu-target", default="50-75")
    parser.add_argument("--heartbeat-sec", type=float, default=20.0)
    parser.add_argument("--scale-mode", default="concrete-routing-repair")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    repo_root = Path(__file__).resolve().parents[2]
    d68_root = Path(args.d68_root)
    d68a_root = Path(args.d68a_root)
    failed_jobs = []

    d68_queue = load_json(d68_root / "queue.json", {})
    d68_args = d68_queue.get("args", {})
    seeds = parse_seeds(args.seeds or d68_args.get("seeds", "12701,12702,12703,12704,12705"))
    train_rows_per_seed = args.train_rows_per_seed or int(d68_args.get("train_rows_per_seed", 240))
    test_rows_per_seed = args.test_rows_per_seed or int(d68_args.get("test_rows_per_seed", 240))
    ood_rows_per_seed = args.ood_rows_per_seed or int(d68_args.get("ood_rows_per_seed", 240))

    write_json(out / "queue.json", {"task": TASK, "args": vars(args), "seeds": seeds, "boundary": BOUNDARY})
    write_json(out / "compute_probe.json", {"cpu_count": os.cpu_count(), "workers": d51.worker_count_from_arg(args.workers), "cpu_target": args.cpu_target})
    append_progress(out, "started", started, {"seeds": seeds, "scale_mode": args.scale_mode})

    learned_triage = load_upstream_trained_triage(d68_root)
    policy_controllers, _ = d68.load_policy_modules(repo_root)
    bundle = d55.d49_bundle()

    train_rows = d51.make_rows_with_progress(seeds, train_rows_per_seed, "train", bundle, out, started, args.heartbeat_sec)
    test_rows = d51.make_rows_with_progress(seeds, test_rows_per_seed, "test", bundle, out, started, args.heartbeat_sec)
    ood_rows = d51.make_rows_with_progress(seeds, ood_rows_per_seed, "ood", bundle, out, started, args.heartbeat_sec)

    try:
        train_packs, train_rust = build_training_packs(train_rows, bundle, out, repo_root, started, args.heartbeat_sec)
        routers, training_report = train_concrete_router(train_packs, out, started, args.heartbeat_sec)
        test_outputs, test_rust = evaluate_split(
            test_rows,
            bundle,
            policy_controllers,
            learned_triage,
            routers,
            out,
            "test",
            started,
            args.heartbeat_sec,
            repo_root,
            out / "row_outputs_test.jsonl",
        )
        ood_outputs, ood_rust = evaluate_split(
            ood_rows,
            bundle,
            policy_controllers,
            learned_triage,
            routers,
            out,
            "ood",
            started,
            args.heartbeat_sec,
            repo_root,
            out / "row_outputs_ood.jsonl",
        )
    except Exception as exc:
        failed_jobs.append({"stage": "evaluate", "error": str(exc)})
        write_json(out / "error.json", failed_jobs[-1])
        raise

    test_metrics = summarize_outputs(test_outputs)
    ood_metrics = summarize_outputs(ood_outputs)
    rust_invocation = {"train": {"aggregation": train_rust}, "test": test_rust, "ood": ood_rust}
    rust_aggregation_rows = train_rust.get("rows_returned", 0) + sum(data.get("aggregation", {}).get("rows_returned", 0) for data in [test_rust, ood_rust])
    rust_controller_rows = sum(data.get("controller", {}).get("rows_requested", 0) for data in [test_rust, ood_rust])
    fallback_rows = sum(1 for row in test_outputs + ood_outputs if row.get("python_fallback_used"))

    aggregate = {
        "task": TASK,
        "boundary": BOUNDARY,
        "scale_mode": args.scale_mode,
        "d68_root": str(d68_root),
        "d68a_root": str(d68a_root),
        "d68a_upstream_manifest": d68a_upstream_manifest(d68a_root),
        "learned_triage_replayed": learned_triage,
        "concrete_router_training_report": training_report,
        "selected_routers": routers,
        "test_metrics": test_metrics,
        "ood_metrics": ood_metrics,
        "d68r_harm_repair_report": harm_repair_report(test_outputs),
        "rust_invocation_report": rust_invocation,
        "rust_path_invoked": rust_aggregation_rows > 0 and rust_controller_rows > 0,
        "rust_aggregation_rows": rust_aggregation_rows,
        "rust_controller_rows": rust_controller_rows,
        "fallback_rows": fallback_rows,
        "failed_jobs": failed_jobs,
    }
    decision = make_decision(aggregate)
    aggregate["decision"] = decision
    write_reports(out, aggregate, decision)
    append_progress(out, "completed", started, {"decision": decision["decision"], "failed_jobs": failed_jobs})


if __name__ == "__main__":
    main()
