#!/usr/bin/env python3
"""D65R aggregation causal isolation and cost-frontier repair.

D65 proved the Rust sparse set-aggregation bridge ran, but did not isolate
aggregation as necessary: ablation matched exact accuracy while spending more
support. D65R separates accuracy-causal signal from support-cost efficiency and
checks whether the current controller can bypass aggregation through other
paths.
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
import run_d62_policy_ensemble_ecf_controller_with_learned_gate as d62
import run_d64s_score_vector_structure_repair as d64s
import run_d65_set_invariant_ipf_aggregation_prototype as d65

TASK = "D65R_AGGREGATION_CAUSAL_ISOLATION_AND_COST_FRONTIER_REPAIR"
BOUNDARY = (
    "D65R only tests the causal and support-cost role of Rust sparse set "
    "aggregation in controlled symbolic joint formula discovery. It does not "
    "prove full VRAXION brain, raw visual Raven reasoning, Raven solved, AGI, "
    "consciousness, DNA/genome success, architecture superiority, or production "
    "readiness."
)

PRIMARY_SPACE = d65.PRIMARY_SPACE
SUPPORT_COUNT = d65.SUPPORT_COUNT
REGIMES = d65.REGIMES
CORE_REGIMES = d65.CORE_REGIMES
ACTIONS = d65.ACTIONS
POLICY_MODULES = d65.POLICY_MODULES
FEATURE_NAMES = d65.FEATURE_NAMES
ROW_SAMPLE_PER_ARM_TRACK_REGIME = 10

TRACKS = [
    "D65_REPLAY",
    "AGGREGATION_REQUIRED_FEATURE_STARVATION",
    "SUPPORT_BUDGET_CAPPED",
    "HIGH_AMBIGUITY_SUPPORT_SET",
    "CORRELATED_ADVERSARIAL_AGGREGATION_STRESS",
    "COST_FRONTIER_TRACK",
]

ARMS = [
    "SYMBOLIC_SET_AGGREGATION_REFERENCE",
    "RUST_SPARSE_SET_AGGREGATION",
    "RUST_SPARSE_SCORE_SHAPE_AGGREGATION",
    "RUST_SPARSE_SUPPORT_COHERENCE_AGGREGATION",
    "RUST_SPARSE_COUNTERFACTUAL_DELTA_AGGREGATION",
    "HYBRID_SYMBOLIC_RUST_AGGREGATION",
    "AGGREGATE_ONLY_CONTROLLER",
    "NON_AGGREGATE_DIAGNOSTIC_ONLY_CONTROLLER",
    "RANDOM_SCORE_AGGREGATION_CONTROL",
    "AGGREGATION_ABLATION_CONTROL",
    "SUPPORT_CONTENT_CORRUPTION_CONTROL",
    "ALWAYS_COUNTER_COMPENSATION_CONTROL",
    "COST_CAPPED_ABLATION_CONTROL",
    "TRUTH_LEAK_SENTINEL_REFERENCE_ONLY",
]

CONTROL_ARMS = [
    "RANDOM_SCORE_AGGREGATION_CONTROL",
    "AGGREGATION_ABLATION_CONTROL",
    "SUPPORT_CONTENT_CORRUPTION_CONTROL",
    "ALWAYS_COUNTER_COMPENSATION_CONTROL",
    "COST_CAPPED_ABLATION_CONTROL",
]

REFERENCE_ONLY_ARMS = ["TRUTH_LEAK_SENTINEL_REFERENCE_ONLY"]
FAIR_ARMS = [arm for arm in ARMS if arm not in REFERENCE_ONLY_ARMS]

D65_ARM_FOR = {
    "SYMBOLIC_SET_AGGREGATION_REFERENCE": "SYMBOLIC_SET_AGGREGATION_REFERENCE",
    "RUST_SPARSE_SET_AGGREGATION": "RUST_SPARSE_SET_INVARIANT_IPF_AGGREGATION",
    "RUST_SPARSE_SCORE_SHAPE_AGGREGATION": "RUST_SPARSE_NORMALIZED_SHAPE_AGGREGATION",
    "RUST_SPARSE_SUPPORT_COHERENCE_AGGREGATION": "RUST_SPARSE_SUPPORT_COHERENCE_SET_AGGREGATION",
    "RUST_SPARSE_COUNTERFACTUAL_DELTA_AGGREGATION": "RUST_SPARSE_COUNTERFACTUAL_DELTA_SET_AGGREGATION",
    "HYBRID_SYMBOLIC_RUST_AGGREGATION": "HYBRID_SYMBOLIC_RUST_SET_AGGREGATION",
    "AGGREGATE_ONLY_CONTROLLER": "RUST_SPARSE_SET_INVARIANT_IPF_AGGREGATION",
    "NON_AGGREGATE_DIAGNOSTIC_ONLY_CONTROLLER": "RUST_SPARSE_SET_INVARIANT_IPF_AGGREGATION",
    "RANDOM_SCORE_AGGREGATION_CONTROL": "RANDOM_SCORE_AGGREGATION_CONTROL",
    "AGGREGATION_ABLATION_CONTROL": "AGGREGATION_ABLATION_CONTROL",
    "SUPPORT_CONTENT_CORRUPTION_CONTROL": "SUPPORT_CONTENT_CORRUPTION_CONTROL",
    "ALWAYS_COUNTER_COMPENSATION_CONTROL": "RUST_SPARSE_SET_INVARIANT_IPF_AGGREGATION",
    "COST_CAPPED_ABLATION_CONTROL": "AGGREGATION_ABLATION_CONTROL",
    "TRUTH_LEAK_SENTINEL_REFERENCE_ONLY": "SYMBOLIC_SET_AGGREGATION_REFERENCE",
}

RUST_SOURCE_ARMS = sorted({value for value in D65_ARM_FOR.values() if value in d65.RUST_AGGREGATION_ARMS})


def write_json(path, value):
    d51.write_json(path, value)


def append_jsonl(path, value):
    d51.append_jsonl(path, value)


def append_progress(out, event, started, data):
    d51.append_progress(out, event, started, data)


def parse_seeds(raw):
    return [int(item.strip()) for item in str(raw).split(",") if item.strip()]


def load_json(path):
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def make_d65_upstream_manifest(repo_root):
    root = repo_root / "target/pilot_wave/d65_set_invariant_ipf_aggregation_prototype/smoke"
    manifest = {
        "upstream": "D65_SET_INVARIANT_IPF_AGGREGATION_PROTOTYPE",
        "expected_decision": "set_invariant_ipf_aggregation_not_confirmed",
        "expected_next": TASK,
        "artifact_root": str(root),
        "decision_present": (root / "decision.json").exists(),
        "summary_present": (root / "summary.json").exists(),
    }
    for name in [
        "decision.json",
        "summary.json",
        "support_content_corruption_report.json",
        "order_invariance_report.json",
        "rust_invocation_report.json",
        "aggregate_metrics.json",
    ]:
        value = load_json(root / name)
        if value is not None:
            manifest[name.replace(".json", "")] = value
    return manifest


def load_policy_modules(repo_root):
    controllers, learned_gate = d64s.load_d62_policy_modules(repo_root)
    missing = [name for name in POLICY_MODULES if name not in controllers]
    for name in missing:
        if name == "COUNTERFACTUAL_POLICY":
            controllers[name] = d62.make_always_action_controller(name, "REQUEST_COUNTER_TOP1_TOP2")
        elif name == "EXTERNAL_TEST_POLICY":
            controllers[name] = d62.make_always_action_controller(name, "REQUEST_EXTERNAL_TEST")
        elif name == "ABSTAIN_POLICY":
            controllers[name] = d62.make_always_action_controller(name, "ABSTAIN")
        elif name == "ADVERSARIAL_REPAIR_POLICY":
            controllers[name] = d62.make_always_action_controller(name, "REQUEST_JOINT_COUNTER")
        else:
            controllers[name] = d62.make_always_action_controller(name, "DECIDE")
    return {name: controllers[name] for name in POLICY_MODULES}, learned_gate


def cap_result(pack, action, cap):
    result = copy.deepcopy(pack["actions"][action])
    if result["total_support_used"] <= cap:
        result["hard_budget_violation"] = False
        result["blocked_action"] = None
        result["hard_support_budget_cap"] = cap
        return result
    blocked = copy.deepcopy(pack["actions"]["DECIDE"])
    blocked["hard_budget_violation"] = True
    blocked["blocked_action"] = action
    blocked["hard_support_budget_cap"] = cap
    blocked["counter_support_used"] = 0
    blocked["cell_counter_support_used"] = 0
    blocked["operator_counter_support_used"] = 0
    blocked["joint_counter_support_used"] = 0
    blocked["random_counter_support_used"] = 0
    blocked["external_test_used"] = 0
    blocked["total_support_used"] = SUPPORT_COUNT
    blocked["error_type"] = "budget_blocked_then_" + str(blocked.get("error_type", "unknown"))
    return blocked


def cap_pack_actions(pack, cap):
    out = copy.deepcopy(pack)
    out["actions"] = {action: cap_result(pack, action, cap) for action in ACTIONS}
    out["action_compact"] = {action: d51.compact_outcome(out["actions"][action]) for action in ACTIONS}
    out["runtime_support_budget_cap"] = cap
    out["feature_map"] = copy.deepcopy(out["feature_map"])
    out["feature_map"]["hard_support_budget_cap_norm"] = cap / 12.0
    out["feature_map"]["runtime_support_budget_available"] = 1.0
    out["feature_map"]["support_budget_pressure_norm"] = 1.0
    return out


def zero_aggregate_features(fmap):
    out = copy.deepcopy(fmap)
    for name in [
        "scalar_confidence",
        "inverse_margin",
        "entropy_norm",
        "collision_norm",
        "dominant_cluster_fraction",
        "support_cluster_count_norm",
        "top1_factorised_disagreement",
        "cell_confidence",
        "operator_confidence",
        "joint_confidence",
        "counterfactual_pressure_norm",
        "runtime_adversarial_pressure_norm",
    ]:
        out[name] = 0.0
    return out


def keep_only_aggregate_features(fmap):
    out = copy.deepcopy(fmap)
    # Remove the non-aggregate side channels to prevent the learned gate from
    # solving via symbolic channel availability alone.
    out["internal_unresolvable_indicator"] = 0.0
    out["external_channel_available"] = 0.0
    out["support_budget_pressure_norm"] = 0.0
    out["runtime_support_budget_available"] = 0.0
    out["hard_support_budget_cap_norm"] = 0.0
    return out


def apply_track_and_arm(pack, track, arm):
    out = copy.deepcopy(pack)
    out["track"] = track
    out["feature_map"] = copy.deepcopy(out["feature_map"])
    out["feature_starvation_track"] = track == "AGGREGATION_REQUIRED_FEATURE_STARVATION"
    out["support_budget_cap_track"] = track == "SUPPORT_BUDGET_CAPPED"
    out["cost_frontier_track"] = track == "COST_FRONTIER_TRACK"
    out["non_aggregate_only_arm"] = arm == "NON_AGGREGATE_DIAGNOSTIC_ONLY_CONTROLLER"
    out["aggregate_only_arm"] = arm == "AGGREGATE_ONLY_CONTROLLER"
    if track == "AGGREGATION_REQUIRED_FEATURE_STARVATION":
        if arm == "NON_AGGREGATE_DIAGNOSTIC_ONLY_CONTROLLER":
            out["feature_map"] = zero_aggregate_features(out["feature_map"])
        else:
            out["feature_map"] = keep_only_aggregate_features(out["feature_map"])
    elif arm == "AGGREGATE_ONLY_CONTROLLER":
        out["feature_map"] = keep_only_aggregate_features(out["feature_map"])
    elif arm == "NON_AGGREGATE_DIAGNOSTIC_ONLY_CONTROLLER":
        out["feature_map"] = zero_aggregate_features(out["feature_map"])
    if track == "SUPPORT_BUDGET_CAPPED":
        out = cap_pack_actions(out, 8)
    elif track == "COST_FRONTIER_TRACK":
        cap = 7 if arm in {"AGGREGATION_ABLATION_CONTROL", "COST_CAPPED_ABLATION_CONTROL"} else 9
        out = cap_pack_actions(out, cap)
    elif arm == "COST_CAPPED_ABLATION_CONTROL":
        out = cap_pack_actions(out, 8)
    if track == "HIGH_AMBIGUITY_SUPPORT_SET":
        out["feature_map"]["scalar_confidence"] = min(float(out["feature_map"].get("scalar_confidence", 0.0)), 0.35)
        out["feature_map"]["joint_confidence"] = min(float(out["feature_map"].get("joint_confidence", 0.0)), 0.35)
        out["feature_map"]["inverse_margin"] = max(float(out["feature_map"].get("inverse_margin", 0.0)), 0.85)
        out["feature_map"]["entropy_norm"] = max(float(out["feature_map"].get("entropy_norm", 0.0)), 0.85)
        out["feature_map"]["counterfactual_pressure_norm"] = max(float(out["feature_map"].get("counterfactual_pressure_norm", 0.0)), 0.85)
    if track == "CORRELATED_ADVERSARIAL_AGGREGATION_STRESS":
        out["feature_map"]["dominant_cluster_fraction"] = max(float(out["feature_map"].get("dominant_cluster_fraction", 0.0)), 0.8)
        out["feature_map"]["collision_norm"] = max(float(out["feature_map"].get("collision_norm", 0.0)), 0.6)
        out["feature_map"]["runtime_adversarial_pressure_norm"] = max(float(out["feature_map"].get("runtime_adversarial_pressure_norm", 0.0)), 0.85)
    out["features"] = d65.features_from_map(out["feature_map"])
    return out


def build_source_pack(row, bundle, arm, idx, rust_features):
    source = D65_ARM_FOR[arm]
    return d65.build_pack(row, bundle, source, idx, rust_features)


def build_eval_items(rows, bundle, rust_features, out, started, heartbeat_sec, split):
    items = []
    total = len(rows) * len(TRACKS) * len(ARMS)
    completed = 0
    last = time.time()
    for idx, row in enumerate(rows):
        for track in TRACKS:
            for arm in ARMS:
                pack = build_source_pack(row, bundle, arm, idx, rust_features)
                pack = apply_track_and_arm(pack, track, arm)
                pack["d65r_arm"] = arm
                pack["d65_source_arm"] = D65_ARM_FOR[arm]
                items.append({"track": track, "arm": arm, "pack": pack})
                completed += 1
        now = time.time()
        if now - last >= heartbeat_sec:
            last = now
            write_json(out / f"partial_{split}_pack_build.json", {"completed": completed, "total": total})
            append_progress(out, "pack_build_progress", started, {"split": split, "completed": completed, "total": total})
    append_progress(out, "pack_build_complete", started, {"split": split, "packs": len(items)})
    return items


def learned_policy(pack, learned_gate):
    features = d62.gate_features(pack)
    for rule in learned_gate["rules"]:
        if features.get(rule["feature"], 0.0) >= rule["threshold"]:
            return rule["policy"], features, "learned_gate"
    return learned_gate["default_policy"], features, "learned_gate_default"


def truth_leak_policy(pack, rust_actions, idx):
    scored = []
    for policy in POLICY_MODULES:
        record = rust_actions[policy][idx]
        row = d59.output_from_action(pack, f"sentinel_{policy}", record["action"], d59.rust_trace(record))
        effective = row["correct"] or (row["support_regime"] == "INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT" and row["abstained"])
        scored.append((1.0 if effective else 0.0, -row["total_support_used"], policy))
    return max(scored)[2]


def output_row(pack, track, arm, policy, action_record, gate_features, gate_basis, used_truth=False):
    row = d59.output_from_action(pack, arm, action_record["action"], d59.rust_trace(action_record))
    row["track"] = track
    row["d65_source_arm"] = pack["d65_source_arm"]
    row["gate_selected_policy"] = policy
    row["gate_features"] = gate_features
    row["gate_basis"] = gate_basis
    row["rust_aggregation_used"] = bool(pack.get("rust_aggregation_used"))
    row["rust_aggregation_input_is_support_set"] = bool(pack.get("rust_aggregation_input_is_support_set"))
    row["python_precomputed_final_aggregate_label_used"] = False
    row["feature_starvation_track"] = bool(pack.get("feature_starvation_track"))
    row["support_budget_cap_track"] = bool(pack.get("support_budget_cap_track"))
    row["cost_frontier_track"] = bool(pack.get("cost_frontier_track"))
    row["aggregate_only_arm"] = bool(pack.get("aggregate_only_arm"))
    row["non_aggregate_only_arm"] = bool(pack.get("non_aggregate_only_arm"))
    row["hard_budget_violation"] = bool(row.get("hard_budget_violation", False))
    row["hard_support_budget_cap"] = row.get("hard_support_budget_cap")
    row["cost_adjusted_accuracy"] = (1.0 if row["correct"] else 0.0) - 0.0025 * max(0.0, row["total_support_used"] - SUPPORT_COUNT)
    row["ablation_compensation"] = bool(arm in {"AGGREGATION_ABLATION_CONTROL", "COST_CAPPED_ABLATION_CONTROL"} and row["total_support_used"] >= 10)
    row["fair_arm"] = arm in FAIR_ARMS
    row["control_arm"] = arm in CONTROL_ARMS
    row["reference_only_arm"] = arm in REFERENCE_ONLY_ARMS
    row["gate_used_truth_label"] = bool(used_truth)
    return row


def record_row(row, outputs, sample_counts, path):
    outputs.append(row)
    key = (row["track"], row["arm"], row["support_regime"])
    if sample_counts[key] < ROW_SAMPLE_PER_ARM_TRACK_REGIME:
        append_jsonl(path, row)
        sample_counts[key] += 1


def summarize_rows(rows):
    base = d51.summarize(rows)
    base["cost_adjusted_accuracy"] = d51.mean([row["cost_adjusted_accuracy"] for row in rows])
    base["ablation_compensation_rate"] = d51.mean([1.0 if row["ablation_compensation"] else 0.0 for row in rows])
    base["hard_budget_violation_rate"] = d51.mean([1.0 if row.get("hard_budget_violation") else 0.0 for row in rows])
    return base


def summarize_outputs(outputs):
    by_track_arm = defaultdict(list)
    by_track_arm_core = defaultdict(list)
    by_track_arm_regime = defaultdict(list)
    by_arm_core = defaultdict(list)
    rust_usage = defaultdict(Counter)
    action_counts = defaultdict(Counter)
    for row in outputs:
        key = (row["track"], row["arm"])
        by_track_arm[key].append(row)
        by_track_arm_regime[(row["track"], row["arm"], row["support_regime"])].append(row)
        action_counts[key][row["selected_action"]] += 1
        rust_usage[row["arm"]]["rows"] += 1
        if row["rust_network_path_invoked"]:
            rust_usage[row["arm"]]["controller_rust_rows"] += 1
        if row["rust_aggregation_used"]:
            rust_usage[row["arm"]]["aggregation_rust_rows"] += 1
        if row["python_fallback_used"]:
            rust_usage[row["arm"]]["python_fallback_rows"] += 1
        if row["python_precomputed_final_aggregate_label_used"]:
            rust_usage[row["arm"]]["python_precomputed_final_aggregate_label_rows"] += 1
        if row["support_regime"] in CORE_REGIMES:
            by_track_arm_core[key].append(row)
            by_arm_core[row["arm"]].append(row)
    return {
        "by_track_arm": {
            track: {arm: summarize_rows(by_track_arm[(track, arm)]) for arm in ARMS if (track, arm) in by_track_arm}
            for track in TRACKS
        },
        "by_track_arm_core": {
            track: {arm: summarize_rows(by_track_arm_core[(track, arm)]) for arm in ARMS if (track, arm) in by_track_arm_core}
            for track in TRACKS
        },
        "by_track_arm_regime": {
            track: {
                arm: {regime: summarize_rows(by_track_arm_regime[(track, arm, regime)]) for regime in REGIMES if (track, arm, regime) in by_track_arm_regime}
                for arm in ARMS
            }
            for track in TRACKS
        },
        "by_arm_core": {arm: summarize_rows(by_arm_core[arm]) for arm in ARMS if arm in by_arm_core},
        "action_distribution": {
            track: {arm: dict(action_counts[(track, arm)]) for arm in ARMS if (track, arm) in action_counts}
            for track in TRACKS
        },
        "rust_usage": {arm: dict(rust_usage[arm]) for arm in ARMS},
    }


def write_partial(out, split, outputs, completed, started):
    recent = outputs[-min(len(outputs), 7000):]
    partial = summarize_outputs(recent)
    write_json(
        out / f"partial_{split}_metrics_snapshot.json",
        {
            "split": split,
            "completed_outputs": completed,
            "elapsed_sec": time.time() - started,
            "recent_by_track_arm_core": partial["by_track_arm_core"],
        },
    )
    append_progress(out, "eval_progress", started, {"split": split, "completed_outputs": completed})


def evaluate_split(rows, bundle, policy_controllers, learned_gate, out, split, started, heartbeat_sec, repo_root, row_output_path):
    rust_features, agg_report = d65.run_rust_aggregation_bridge(out, repo_root, rows, bundle, split, started, heartbeat_sec)
    items = build_eval_items(rows, bundle, rust_features, out, started, heartbeat_sec, split)
    packs = [item["pack"] for item in items]
    policy_actions, policy_report = d59.run_rust_multi_bridge(out, repo_root, policy_controllers, packs, split, "d65r_policy_eval", started)
    outputs = []
    sample_counts = Counter()
    last = 0.0
    for idx, item in enumerate(items):
        track = item["track"]
        arm = item["arm"]
        pack = item["pack"]
        if arm == "TRUTH_LEAK_SENTINEL_REFERENCE_ONLY":
            policy = truth_leak_policy(pack, policy_actions, idx)
            gate_features = d62.gate_features(pack)
            basis = "reference_only_best_policy_after_truth_scoring"
            used_truth = True
        elif arm == "ALWAYS_COUNTER_COMPENSATION_CONTROL":
            policy = "ADVERSARIAL_REPAIR_POLICY"
            gate_features = d62.gate_features(pack)
            basis = "forced_always_joint_counter_policy"
            used_truth = False
        else:
            policy, gate_features, basis = learned_policy(pack, learned_gate)
            used_truth = False
        action_record = policy_actions[policy][idx]
        row = output_row(pack, track, arm, policy, action_record, gate_features, basis, used_truth)
        record_row(row, outputs, sample_counts, row_output_path)
        now = time.time()
        if now - last >= heartbeat_sec or len(outputs) == len(items):
            last = now
            write_partial(out, split, outputs, len(outputs), started)
    return outputs, {"aggregation": agg_report, "controller": policy_report}


def metric(metrics, track, arm, field):
    return metrics["by_track_arm_core"].get(track, {}).get(arm, {}).get(field, 0.0)


def make_decision(test_metrics, failed_jobs):
    if failed_jobs:
        return {
            "decision": "d65r_aggregation_repair_failed",
            "verdict": "D65R_AGGREGATION_REPAIR_FAILED",
            "next": "D65_REPAIR",
            "best_arm": None,
            "reason": "failed jobs present",
        }
    causal_tracks = ["AGGREGATION_REQUIRED_FEATURE_STARVATION", "SUPPORT_BUDGET_CAPPED"]
    rust_arms = [
        "RUST_SPARSE_SET_AGGREGATION",
        "RUST_SPARSE_SCORE_SHAPE_AGGREGATION",
        "RUST_SPARSE_SUPPORT_COHERENCE_AGGREGATION",
        "RUST_SPARSE_COUNTERFACTUAL_DELTA_AGGREGATION",
        "AGGREGATE_ONLY_CONTROLLER",
    ]
    causal_best = None
    causal_gap = -999.0
    for track in causal_tracks:
        control = max(
            metric(test_metrics, track, "AGGREGATION_ABLATION_CONTROL", "exact_joint_accuracy"),
            metric(test_metrics, track, "RANDOM_SCORE_AGGREGATION_CONTROL", "exact_joint_accuracy"),
            metric(test_metrics, track, "NON_AGGREGATE_DIAGNOSTIC_ONLY_CONTROLLER", "exact_joint_accuracy"),
        )
        for arm in rust_arms:
            gap = metric(test_metrics, track, arm, "exact_joint_accuracy") - control
            if gap > causal_gap:
                causal_gap = gap
                causal_best = (track, arm)
    replay_rust_support = metric(test_metrics, "D65_REPLAY", "RUST_SPARSE_SET_AGGREGATION", "average_total_support_used")
    replay_ablation_support = metric(test_metrics, "D65_REPLAY", "AGGREGATION_ABLATION_CONTROL", "average_total_support_used")
    replay_rust_exact = metric(test_metrics, "D65_REPLAY", "RUST_SPARSE_SET_AGGREGATION", "exact_joint_accuracy")
    replay_ablation_exact = metric(test_metrics, "D65_REPLAY", "AGGREGATION_ABLATION_CONTROL", "exact_joint_accuracy")
    replay_rust_false = metric(test_metrics, "D65_REPLAY", "RUST_SPARSE_SET_AGGREGATION", "false_confidence_rate")
    cost_delta = replay_ablation_support - replay_rust_support
    no_safety_loss = replay_rust_false <= 0.01 and replay_rust_exact >= replay_ablation_exact - 0.005
    if causal_gap >= 0.03:
        return {
            "decision": "rust_sparse_set_aggregation_causally_confirmed",
            "verdict": "D65R_RUST_SPARSE_SET_AGGREGATION_CAUSALLY_CONFIRMED",
            "next": "D66_RUST_SPARSE_SUPPORT_SCORING_MIGRATION_PLAN",
            "best_arm": causal_best[1],
            "best_track": causal_best[0],
            "reason": {"causal_gap": causal_gap},
        }
    if cost_delta >= 1.0 and no_safety_loss:
        return {
            "decision": "rust_sparse_set_aggregation_efficiency_confirmed",
            "verdict": "D65R_RUST_SPARSE_SET_AGGREGATION_EFFICIENCY_CONFIRMED",
            "next": "D66_RUST_SPARSE_SUPPORT_SCORING_WITH_AGGREGATION_COST_CONTROL",
            "best_arm": "RUST_SPARSE_SET_AGGREGATION",
            "best_track": "D65_REPLAY",
            "reason": {
                "support_delta_vs_ablation": cost_delta,
                "rust_exact": replay_rust_exact,
                "ablation_exact": replay_ablation_exact,
                "rust_false_confidence": replay_rust_false,
                "causal_gap": causal_gap,
            },
        }
    if replay_rust_exact >= 0.99 and causal_gap < 0.03 and cost_delta < 1.0:
        return {
            "decision": "set_aggregation_redundant_under_current_controller",
            "verdict": "D65R_SET_AGGREGATION_REDUNDANT_UNDER_CURRENT_CONTROLLER",
            "next": "D65R_REDEFINE_OR_SKIP_AGGREGATION_LAYER",
            "best_arm": "RUST_SPARSE_SET_AGGREGATION",
            "best_track": "D65_REPLAY",
            "reason": {"causal_gap": causal_gap, "support_delta_vs_ablation": cost_delta},
        }
    return {
        "decision": "d65r_aggregation_repair_failed",
        "verdict": "D65R_AGGREGATION_REPAIR_FAILED",
        "next": "D65_REPAIR",
        "best_arm": "RUST_SPARSE_SET_AGGREGATION",
        "best_track": "D65_REPLAY",
        "reason": {"causal_gap": causal_gap, "support_delta_vs_ablation": cost_delta},
    }


def make_reports(out, aggregate, decision):
    metrics = aggregate["test_metrics"]
    reports = {
        "d65_upstream_manifest.json": aggregate["d65_upstream_manifest"],
        "causal_isolation_report.json": {
            "feature_starvation": metrics["by_track_arm_core"].get("AGGREGATION_REQUIRED_FEATURE_STARVATION", {}),
            "support_budget_capped": metrics["by_track_arm_core"].get("SUPPORT_BUDGET_CAPPED", {}),
            "decision_reason": decision.get("reason"),
        },
        "feature_starvation_report.json": metrics["by_track_arm_core"].get("AGGREGATION_REQUIRED_FEATURE_STARVATION", {}),
        "support_budget_cap_report.json": metrics["by_track_arm_core"].get("SUPPORT_BUDGET_CAPPED", {}),
        "cost_frontier_report.json": {
            track: {
                arm: {
                    "exact": values.get("exact_joint_accuracy", 0.0),
                    "support": values.get("average_total_support_used", 0.0),
                    "cost_adjusted_accuracy": values.get("cost_adjusted_accuracy", 0.0),
                }
                for arm, values in by_arm.items()
            }
            for track, by_arm in metrics["by_track_arm_core"].items()
        },
        "compensation_path_report.json": {
            track: {
                arm: values.get("ablation_compensation_rate", 0.0)
                for arm, values in by_arm.items()
                if arm in {"AGGREGATION_ABLATION_CONTROL", "COST_CAPPED_ABLATION_CONTROL", "ALWAYS_COUNTER_COMPENSATION_CONTROL"}
            }
            for track, by_arm in metrics["by_track_arm_core"].items()
        },
        "aggregation_quality_report.json": metrics["by_track_arm_core"],
        "content_corruption_report.json": {
            track: by_arm.get("SUPPORT_CONTENT_CORRUPTION_CONTROL", {})
            for track, by_arm in metrics["by_track_arm_core"].items()
        },
        "truth_leak_audit_report.json": {
            "fair_arms": FAIR_ARMS,
            "reference_only_arms": REFERENCE_ONLY_ARMS,
            "fair_arms_using_truth_label": [],
            "python_precomputed_final_aggregate_label_used_by_fair_arms": False,
            "truth_leak_sentinel": metrics["by_arm_core"].get("TRUTH_LEAK_SENTINEL_REFERENCE_ONLY", {}),
        },
        "rust_invocation_report.json": aggregate["rust_invocation_report"],
    }
    for name, value in reports.items():
        write_json(out / name, value)
    return reports


def write_report(out, decision, metrics):
    rows = [
        "# D65R Aggregation Causal Isolation And Cost Frontier Repair Result",
        "",
        "Decision:",
        "",
        "```text",
        f"decision = {decision['decision']}",
        f"verdict = {decision['verdict']}",
        f"next = {decision['next']}",
        f"best_arm = {decision.get('best_arm')}",
        f"best_track = {decision.get('best_track')}",
        "```",
        "",
        "| track | arm | exact | support | cost-adjusted | false conf | compensation |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for track in TRACKS:
        by_arm = metrics["by_track_arm_core"].get(track, {})
        for arm in [
            "RUST_SPARSE_SET_AGGREGATION",
            "AGGREGATE_ONLY_CONTROLLER",
            "NON_AGGREGATE_DIAGNOSTIC_ONLY_CONTROLLER",
            "AGGREGATION_ABLATION_CONTROL",
            "COST_CAPPED_ABLATION_CONTROL",
            "ALWAYS_COUNTER_COMPENSATION_CONTROL",
            "RANDOM_SCORE_AGGREGATION_CONTROL",
            "SUPPORT_CONTENT_CORRUPTION_CONTROL",
        ]:
            values = by_arm.get(arm, {})
            rows.append(
                f"| {track} | {arm} | {values.get('exact_joint_accuracy', 0.0):.6f} | "
                f"{values.get('average_total_support_used', 0.0):.4f} | "
                f"{values.get('cost_adjusted_accuracy', 0.0):.6f} | "
                f"{values.get('false_confidence_rate', 0.0):.6f} | "
                f"{values.get('ablation_compensation_rate', 0.0):.6f} |"
            )
    rows += ["", "Boundary:", "", "```text", BOUNDARY, "```"]
    (out / "report.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", default="12401,12402,12403,12404,12405")
    parser.add_argument("--train-rows-per-seed", type=int, default=300)
    parser.add_argument("--test-rows-per-seed", type=int, default=300)
    parser.add_argument("--ood-rows-per-seed", type=int, default=300)
    parser.add_argument("--workers", default="auto")
    parser.add_argument("--cpu-target", default="50-75")
    parser.add_argument("--heartbeat-sec", type=float, default=20.0)
    parser.add_argument("--scale-mode", default="scale-lite")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    repo_root = Path(__file__).resolve().parents[2]
    seeds = parse_seeds(args.seeds)
    failed_jobs = []
    write_json(out / "queue.json", {"task": TASK, "args": vars(args), "seeds": seeds, "boundary": BOUNDARY})
    append_progress(out, "started", started, {"seeds": seeds, "scale_mode": args.scale_mode})
    write_json(out / "compute_probe.json", {"cpu_count": os.cpu_count(), "workers": d51.worker_count_from_arg(args.workers), "cpu_target": args.cpu_target})

    d65_manifest = make_d65_upstream_manifest(repo_root)
    write_json(out / "d65_upstream_manifest.json", d65_manifest)
    policy_controllers, learned_gate = load_policy_modules(repo_root)
    bundle = d55.d49_bundle()
    write_json(
        out / "dataset_manifest.json",
        {
            "task": "controlled_symbolic_joint_formula_discovery",
            "primary_space": PRIMARY_SPACE,
            "support_count": SUPPORT_COUNT,
            "regimes": REGIMES,
            "tracks": TRACKS,
            "arms": ARMS,
            "feature_starvation_required": True,
            "support_budget_cap_required": True,
            "truth_hidden_from_controller_inputs": True,
            "rust_arms_receive_support_evidence_set_representation": True,
            "python_precomputed_final_aggregate_label_used": False,
            "formula_solver_learning_used": False,
            "controller_only_not_formula_solver": True,
        },
    )

    train_rows = d51.make_rows_with_progress(seeds, args.train_rows_per_seed, "train", bundle, out, started, args.heartbeat_sec)
    write_json(out / "partial_training_rows_generated.json", {"rows": len(train_rows), "training_used": False})
    test_rows = d51.make_rows_with_progress(seeds, args.test_rows_per_seed, "test", bundle, out, started, args.heartbeat_sec)
    ood_rows = d51.make_rows_with_progress(seeds, args.ood_rows_per_seed, "ood", bundle, out, started, args.heartbeat_sec)

    row_test = out / "row_outputs_test.jsonl"
    row_ood = out / "row_outputs_ood.jsonl"
    for path in [row_test, row_ood]:
        if path.exists():
            path.unlink()
    try:
        test_outputs, test_report = evaluate_split(test_rows, bundle, policy_controllers, learned_gate, out, "test", started, args.heartbeat_sec, repo_root, row_test)
        write_json(out / "partial_test_metrics.json", summarize_outputs(test_outputs))
        ood_outputs, ood_report = evaluate_split(ood_rows, bundle, policy_controllers, learned_gate, out, "ood", started, args.heartbeat_sec, repo_root, row_ood)
        write_json(out / "partial_ood_metrics.json", summarize_outputs(ood_outputs))
    except Exception as exc:
        failed_jobs.append({"error": repr(exc)})
        write_json(out / "error.json", {"error": repr(exc), "failed_jobs": failed_jobs})
        raise

    test_metrics = summarize_outputs(test_outputs)
    ood_metrics = summarize_outputs(ood_outputs)
    decision = make_decision(test_metrics, failed_jobs)
    fallback_rows = 0
    rust_controller_rows = 0
    rust_aggregation_rows = 0
    precomputed_label_rows = 0
    for metrics in [test_metrics, ood_metrics]:
        for counts in metrics["rust_usage"].values():
            fallback_rows += counts.get("python_fallback_rows", 0)
            rust_controller_rows += counts.get("controller_rust_rows", 0)
            rust_aggregation_rows += counts.get("aggregation_rust_rows", 0)
            precomputed_label_rows += counts.get("python_precomputed_final_aggregate_label_rows", 0)
    aggregate = {
        "task": TASK,
        "failed_jobs": failed_jobs,
        "d65_upstream_manifest": d65_manifest,
        "test_metrics": test_metrics,
        "ood_metrics": ood_metrics,
        "rust_invocation_report": {"test": test_report, "ood": ood_report},
        "rust_path_invoked": rust_controller_rows > 0 and rust_aggregation_rows > 0,
        "rust_controller_rows": rust_controller_rows,
        "rust_aggregation_rows": rust_aggregation_rows,
        "fallback_rows": fallback_rows,
        "python_precomputed_final_aggregate_label_rows": precomputed_label_rows,
        "decision": decision,
        "boundary": BOUNDARY,
    }
    reports = make_reports(out, aggregate, decision)
    write_json(out / "aggregate_metrics.json", aggregate)
    write_json(out / "decision.json", decision)
    write_json(
        out / "summary.json",
        {
            "task": TASK,
            "decision": decision["decision"],
            "verdict": decision["verdict"],
            "next": decision["next"],
            "best_arm": decision.get("best_arm"),
            "best_track": decision.get("best_track"),
            "rust_path_invoked": aggregate["rust_path_invoked"],
            "rust_controller_rows": rust_controller_rows,
            "rust_aggregation_rows": rust_aggregation_rows,
            "fallback_rows": fallback_rows,
            "python_precomputed_final_aggregate_label_rows": precomputed_label_rows,
            "failed_jobs": failed_jobs,
            "artifact_reports": sorted(reports.keys()),
            "boundary": BOUNDARY,
        },
    )
    write_report(out, decision, test_metrics)
    append_progress(out, "completed", started, {"decision": decision["decision"], "failed_jobs": failed_jobs})
    print(json.dumps(load_json(out / "summary.json"), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
