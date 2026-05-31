#!/usr/bin/env python3
"""D66 Rust sparse support scoring with aggregation cost control.

D65R found that Rust sparse set aggregation was not accuracy-necessary in the
current task, but it reduced support cost. D66 tests that narrower claim
directly: can aggregation-backed support scoring keep accuracy/safety while
spending less counter-support than ablation or always-counter compensation?
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
import run_d65r_aggregation_causal_isolation_and_cost_frontier_repair as d65r

TASK = "D66_RUST_SPARSE_SUPPORT_SCORING_WITH_AGGREGATION_COST_CONTROL"
BOUNDARY = (
    "D66 only tests Rust sparse aggregation-backed support scoring and "
    "counter-support cost control in controlled symbolic joint formula "
    "discovery. It does not prove full VRAXION brain, raw visual Raven "
    "reasoning, Raven solved, AGI, consciousness, DNA/genome success, "
    "architecture superiority, or production readiness."
)

PRIMARY_SPACE = d65.PRIMARY_SPACE
SUPPORT_COUNT = d65.SUPPORT_COUNT
REGIMES = d65.REGIMES
CORE_REGIMES = d65.CORE_REGIMES
ACTIONS = d65.ACTIONS
POLICY_MODULES = d65.POLICY_MODULES

ARMS = [
    "D65R_RUST_SET_AGG_REFERENCE",
    "SUPPORT_SCORING_WITH_RUST_AGGREGATION",
    "COUNTER_SUPPORT_TRIAGE_WITH_RUST_AGGREGATION",
    "SUPPORT_BUDGET_CAPPED_RUST_AGGREGATION",
    "AGGREGATION_ABLATION_CONTROL",
    "RANDOM_SCORE_CONTROL",
    "COST_MATCHED_RANDOM_SUPPORT_CONTROL",
    "ALWAYS_COUNTER_CONTROL",
    "NON_AGGREGATE_DIAGNOSTIC_ONLY",
    "SUPPORT_CONTENT_CORRUPTION_CONTROL",
    "TRUTH_LEAK_SENTINEL_REFERENCE_ONLY",
]

REFERENCE_ONLY_ARMS = ["TRUTH_LEAK_SENTINEL_REFERENCE_ONLY"]
CONTROL_ARMS = [
    "AGGREGATION_ABLATION_CONTROL",
    "RANDOM_SCORE_CONTROL",
    "COST_MATCHED_RANDOM_SUPPORT_CONTROL",
    "ALWAYS_COUNTER_CONTROL",
    "NON_AGGREGATE_DIAGNOSTIC_ONLY",
    "SUPPORT_CONTENT_CORRUPTION_CONTROL",
]
FAIR_ARMS = [arm for arm in ARMS if arm not in REFERENCE_ONLY_ARMS]

D65_ARM_FOR = {
    "D65R_RUST_SET_AGG_REFERENCE": "RUST_SPARSE_SET_INVARIANT_IPF_AGGREGATION",
    "SUPPORT_SCORING_WITH_RUST_AGGREGATION": "RUST_SPARSE_SET_INVARIANT_IPF_AGGREGATION",
    "COUNTER_SUPPORT_TRIAGE_WITH_RUST_AGGREGATION": "RUST_SPARSE_COUNTERFACTUAL_DELTA_SET_AGGREGATION",
    "SUPPORT_BUDGET_CAPPED_RUST_AGGREGATION": "RUST_SPARSE_SET_INVARIANT_IPF_AGGREGATION",
    "AGGREGATION_ABLATION_CONTROL": "AGGREGATION_ABLATION_CONTROL",
    "RANDOM_SCORE_CONTROL": "RANDOM_SCORE_AGGREGATION_CONTROL",
    "COST_MATCHED_RANDOM_SUPPORT_CONTROL": "RANDOM_SCORE_AGGREGATION_CONTROL",
    "ALWAYS_COUNTER_CONTROL": "RUST_SPARSE_SET_INVARIANT_IPF_AGGREGATION",
    "NON_AGGREGATE_DIAGNOSTIC_ONLY": "RUST_SPARSE_SET_INVARIANT_IPF_AGGREGATION",
    "SUPPORT_CONTENT_CORRUPTION_CONTROL": "SUPPORT_CONTENT_CORRUPTION_CONTROL",
    "TRUTH_LEAK_SENTINEL_REFERENCE_ONLY": "SYMBOLIC_SET_AGGREGATION_REFERENCE",
}

ROW_SAMPLE_PER_ARM_REGIME = 14
SUPPORT_BUDGET_CAP = 9


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


def make_d65r_upstream_manifest(repo_root):
    root = repo_root / "target/pilot_wave/d65r_aggregation_causal_isolation_and_cost_frontier_repair/smoke"
    manifest = {
        "upstream": "D65R_AGGREGATION_CAUSAL_ISOLATION_AND_COST_FRONTIER_REPAIR",
        "expected_decision": "rust_sparse_set_aggregation_efficiency_confirmed",
        "expected_next": TASK,
        "artifact_root": str(root),
        "decision_present": (root / "decision.json").exists(),
        "summary_present": (root / "summary.json").exists(),
    }
    for name in [
        "decision.json",
        "summary.json",
        "causal_isolation_report.json",
        "cost_frontier_report.json",
        "rust_invocation_report.json",
        "truth_leak_audit_report.json",
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


def build_source_pack(row, bundle, arm, idx, rust_features):
    source = D65_ARM_FOR[arm]
    pack = d65.build_pack(row, bundle, source, idx, rust_features)
    out = copy.deepcopy(pack)
    out["d66_arm"] = arm
    out["d65_source_arm"] = source
    out["support_budget_cap"] = None
    out["support_scoring_used"] = arm in {
        "SUPPORT_SCORING_WITH_RUST_AGGREGATION",
        "COUNTER_SUPPORT_TRIAGE_WITH_RUST_AGGREGATION",
        "SUPPORT_BUDGET_CAPPED_RUST_AGGREGATION",
    }
    if arm == "NON_AGGREGATE_DIAGNOSTIC_ONLY":
        out["feature_map"] = d65r.zero_aggregate_features(out["feature_map"])
        out["features"] = d65.features_from_map(out["feature_map"])
    if arm in {"SUPPORT_BUDGET_CAPPED_RUST_AGGREGATION", "COST_MATCHED_RANDOM_SUPPORT_CONTROL"}:
        out = d65r.cap_pack_actions(out, SUPPORT_BUDGET_CAP)
        out["support_budget_cap"] = SUPPORT_BUDGET_CAP
    return out


def build_eval_items(rows, bundle, rust_features, out, started, heartbeat_sec, split):
    items = []
    total = len(rows) * len(ARMS)
    completed = 0
    last = time.time()
    for idx, row in enumerate(rows):
        for arm in ARMS:
            items.append({"arm": arm, "pack": build_source_pack(row, bundle, arm, idx, rust_features)})
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


def support_scoring_policy(pack):
    features = d62.gate_features(pack)
    confidence = min(
        float(pack["feature_map"].get("scalar_confidence", 0.0)),
        float(pack["feature_map"].get("joint_confidence", 0.0)),
    )
    if features["external_channel_available"] >= 0.25:
        return "EXTERNAL_TEST_POLICY", features, "aggregation_support_scoring_external"
    if features["internal_unresolvable_indicator"] >= 0.25:
        return "ABSTAIN_POLICY", features, "aggregation_support_scoring_abstain"
    if confidence >= 0.74 and features["inverse_margin"] <= 0.32 and features["collision_norm"] <= 0.25:
        return "SATURATED_POLICY", features, "aggregation_support_scoring_high_confidence"
    if features["adversarial_pressure_norm"] >= 0.68:
        return "ADVERSARIAL_REPAIR_POLICY", features, "aggregation_support_scoring_adversarial"
    if features["counterfactual_pressure_norm"] >= 0.58:
        return "COUNTERFACTUAL_POLICY", features, "aggregation_support_scoring_counterfactual"
    return "SATURATED_POLICY", features, "aggregation_support_scoring_default"


def counter_support_triage_policy(pack):
    features = d62.gate_features(pack)
    if features["external_channel_available"] >= 0.25:
        return "EXTERNAL_TEST_POLICY", features, "counter_support_triage_external"
    if features["internal_unresolvable_indicator"] >= 0.25:
        return "ABSTAIN_POLICY", features, "counter_support_triage_abstain"
    if features["adversarial_pressure_norm"] >= 0.82 and features["collision_norm"] >= 0.30:
        return "ADVERSARIAL_REPAIR_POLICY", features, "counter_support_triage_joint"
    if features["counterfactual_pressure_norm"] >= 0.45 or features["adversarial_pressure_norm"] >= 0.55:
        return "COUNTERFACTUAL_POLICY", features, "counter_support_triage_top1_top2"
    return "SATURATED_POLICY", features, "counter_support_triage_default"


def select_policy(arm, pack, learned_gate, policy_actions, idx):
    if arm == "TRUTH_LEAK_SENTINEL_REFERENCE_ONLY":
        scored = []
        for policy in POLICY_MODULES:
            record = policy_actions[policy][idx]
            row = d59.output_from_action(pack, f"sentinel_{policy}", record["action"], d59.rust_trace(record))
            effective = row["correct"] or (row["support_regime"] == "INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT" and row["abstained"])
            scored.append((1.0 if effective else 0.0, -row["total_support_used"], policy))
        features = d62.gate_features(pack)
        return max(scored)[2], features, "reference_only_best_policy_after_truth_scoring", True
    if arm == "ALWAYS_COUNTER_CONTROL":
        return "ADVERSARIAL_REPAIR_POLICY", d62.gate_features(pack), "forced_always_joint_counter", False
    if arm == "SUPPORT_SCORING_WITH_RUST_AGGREGATION":
        policy, features, basis = support_scoring_policy(pack)
        return policy, features, basis, False
    if arm == "COUNTER_SUPPORT_TRIAGE_WITH_RUST_AGGREGATION":
        policy, features, basis = counter_support_triage_policy(pack)
        return policy, features, basis, False
    policy, features, basis = learned_policy(pack, learned_gate)
    return policy, features, basis, False


def action_audit(pack, selected_action):
    decide = pack["actions"]["DECIDE"]
    selected = pack["actions"][selected_action]
    alternatives = list(pack["actions"].values())
    any_counter_correct = any(
        item["correct"] for action, item in pack["actions"].items()
        if action in {"REQUEST_COUNTER_TOP1_TOP2", "REQUEST_JOINT_COUNTER", "REQUEST_EXTERNAL_TEST"}
    )
    selected_is_counter = selected_action in {"REQUEST_COUNTER_TOP1_TOP2", "REQUEST_JOINT_COUNTER", "REQUEST_EXTERNAL_TEST"}
    unnecessary = selected_is_counter and bool(decide["correct"])
    missed = selected_action == "DECIDE" and (not selected["correct"]) and any_counter_correct
    cheapest_correct = min([item["total_support_used"] for item in alternatives if item["correct"]] or [selected["total_support_used"]])
    return {
        "unnecessary_counter_support": unnecessary,
        "missed_counter_support": missed,
        "cheapest_correct_support_used": cheapest_correct,
        "support_over_cheapest_correct": max(0.0, selected["total_support_used"] - cheapest_correct),
    }


def output_row(pack, arm, policy, action_record, gate_features, gate_basis, used_truth=False):
    row = d59.output_from_action(pack, arm, action_record["action"], d59.rust_trace(action_record))
    audit = action_audit(pack, action_record["action"])
    row.update(audit)
    row["d65_source_arm"] = pack["d65_source_arm"]
    row["gate_selected_policy"] = policy
    row["gate_features"] = gate_features
    row["gate_basis"] = gate_basis
    row["rust_aggregation_used"] = bool(pack.get("rust_aggregation_used"))
    row["rust_aggregation_input_is_support_set"] = bool(pack.get("rust_aggregation_input_is_support_set"))
    row["python_precomputed_final_aggregate_label_used"] = False
    row["support_scoring_used"] = bool(pack.get("support_scoring_used"))
    row["support_budget_cap"] = pack.get("support_budget_cap")
    row["cost_adjusted_accuracy"] = (1.0 if row["correct"] else 0.0) - 0.0025 * max(0.0, row["total_support_used"] - SUPPORT_COUNT)
    row["fair_arm"] = arm in FAIR_ARMS
    row["control_arm"] = arm in CONTROL_ARMS
    row["reference_only_arm"] = arm in REFERENCE_ONLY_ARMS
    row["gate_used_truth_label"] = bool(used_truth)
    return row


def record_row(row, outputs, sample_counts, path):
    outputs.append(row)
    key = (row["arm"], row["support_regime"])
    if sample_counts[key] < ROW_SAMPLE_PER_ARM_REGIME:
        append_jsonl(path, row)
        sample_counts[key] += 1


def summarize_rows(rows):
    base = d51.summarize(rows)
    base["cost_adjusted_accuracy"] = d51.mean([row["cost_adjusted_accuracy"] for row in rows])
    base["unnecessary_counter_support_rate"] = d51.mean([1.0 if row["unnecessary_counter_support"] else 0.0 for row in rows])
    base["missed_counter_support_rate"] = d51.mean([1.0 if row["missed_counter_support"] else 0.0 for row in rows])
    base["support_over_cheapest_correct_mean"] = d51.mean([row["support_over_cheapest_correct"] for row in rows])
    return base


def summarize_outputs(outputs):
    by_arm = defaultdict(list)
    by_arm_core = defaultdict(list)
    by_arm_regime = defaultdict(list)
    by_seed_core = defaultdict(list)
    rust_usage = defaultdict(Counter)
    action_counts = defaultdict(Counter)
    for row in outputs:
        arm = row["arm"]
        by_arm[arm].append(row)
        by_arm_regime[(arm, row["support_regime"])].append(row)
        action_counts[arm][row["selected_action"]] += 1
        rust_usage[arm]["rows"] += 1
        if row["rust_network_path_invoked"]:
            rust_usage[arm]["controller_rust_rows"] += 1
        if row["rust_aggregation_used"]:
            rust_usage[arm]["aggregation_rust_rows"] += 1
        if row["python_fallback_used"]:
            rust_usage[arm]["python_fallback_rows"] += 1
        if row["python_precomputed_final_aggregate_label_used"]:
            rust_usage[arm]["python_precomputed_final_aggregate_label_rows"] += 1
        if row["support_regime"] in CORE_REGIMES:
            by_arm_core[arm].append(row)
            by_seed_core[(arm, row["seed"])].append(row)
    return {
        "by_arm": {arm: summarize_rows(by_arm[arm]) for arm in ARMS if arm in by_arm},
        "by_arm_core": {arm: summarize_rows(by_arm_core[arm]) for arm in ARMS if arm in by_arm_core},
        "by_arm_and_regime": {
            arm: {regime: summarize_rows(by_arm_regime[(arm, regime)]) for regime in REGIMES if (arm, regime) in by_arm_regime}
            for arm in ARMS
        },
        "by_seed_core": {
            arm: {str(seed): summarize_rows(rows) for (a, seed), rows in by_seed_core.items() if a == arm and rows}
            for arm in ARMS
        },
        "action_distribution": {arm: dict(action_counts[arm]) for arm in ARMS},
        "rust_usage": {arm: dict(rust_usage[arm]) for arm in ARMS},
    }


def write_partial(out, split, outputs, completed, started):
    recent = outputs[-min(len(outputs), 8000):]
    snapshot = summarize_outputs(recent)
    write_json(
        out / f"partial_{split}_metrics_snapshot.json",
        {
            "split": split,
            "completed_outputs": completed,
            "elapsed_sec": time.time() - started,
            "recent_by_arm_core": snapshot["by_arm_core"],
        },
    )
    write_json(out / "partial_support_cost_report.json", make_support_cost_report(snapshot["by_arm_core"]))
    append_progress(out, "eval_progress", started, {"split": split, "completed_outputs": completed})


def evaluate_split(rows, bundle, policy_controllers, learned_gate, out, split, started, heartbeat_sec, repo_root, row_output_path):
    rust_features, aggregation_report = d65.run_rust_aggregation_bridge(out, repo_root, rows, bundle, split, started, heartbeat_sec)
    items = build_eval_items(rows, bundle, rust_features, out, started, heartbeat_sec, split)
    packs = [item["pack"] for item in items]
    policy_actions, policy_report = d59.run_rust_multi_bridge(out, repo_root, policy_controllers, packs, split, "d66_policy_eval", started)
    outputs = []
    sample_counts = Counter()
    last = 0.0
    for idx, item in enumerate(items):
        arm = item["arm"]
        pack = item["pack"]
        policy, gate_features, basis, used_truth = select_policy(arm, pack, learned_gate, policy_actions, idx)
        action_record = policy_actions[policy][idx]
        row = output_row(pack, arm, policy, action_record, gate_features, basis, used_truth)
        record_row(row, outputs, sample_counts, row_output_path)
        now = time.time()
        if now - last >= heartbeat_sec or len(outputs) == len(items):
            last = now
            write_partial(out, split, outputs, len(outputs), started)
    return outputs, {"aggregation": aggregation_report, "controller": policy_report}


def make_support_cost_report(by_arm):
    return {
        arm: {
            "exact": values.get("exact_joint_accuracy", 0.0),
            "support": values.get("average_total_support_used", 0.0),
            "counter_support": values.get("average_counter_support_used", 0.0),
            "cost_adjusted_accuracy": values.get("cost_adjusted_accuracy", 0.0),
            "unnecessary_counter_support_rate": values.get("unnecessary_counter_support_rate", 0.0),
            "missed_counter_support_rate": values.get("missed_counter_support_rate", 0.0),
        }
        for arm, values in by_arm.items()
    }


def metric(metrics, arm, field):
    return metrics["by_arm_core"].get(arm, {}).get(field, 0.0)


def regime_metric(metrics, arm, regime, field):
    return metrics["by_arm_and_regime"].get(arm, {}).get(regime, {}).get(field, 0.0)


def make_decision(test_metrics, failed_jobs):
    if failed_jobs:
        return {
            "decision": "support_scoring_cost_control_not_confirmed",
            "verdict": "D66_FAILED_JOBS",
            "next": "D66_REPAIR",
            "best_arm": None,
            "reason": "failed_jobs not empty",
        }
    candidate_arms = [
        "SUPPORT_SCORING_WITH_RUST_AGGREGATION",
        "COUNTER_SUPPORT_TRIAGE_WITH_RUST_AGGREGATION",
        "SUPPORT_BUDGET_CAPPED_RUST_AGGREGATION",
        "D65R_RUST_SET_AGG_REFERENCE",
    ]
    reference = test_metrics["by_arm_core"].get("D65R_RUST_SET_AGG_REFERENCE", {})
    ablation = test_metrics["by_arm_core"].get("AGGREGATION_ABLATION_CONTROL", {})
    always = test_metrics["by_arm_core"].get("ALWAYS_COUNTER_CONTROL", {})
    random_control = test_metrics["by_arm_core"].get("RANDOM_SCORE_CONTROL", {})
    content = test_metrics["by_arm_core"].get("SUPPORT_CONTENT_CORRUPTION_CONTROL", {})
    best = max(
        candidate_arms,
        key=lambda arm: (
            metric(test_metrics, arm, "cost_adjusted_accuracy"),
            metric(test_metrics, arm, "exact_joint_accuracy"),
            -metric(test_metrics, arm, "average_total_support_used"),
        ),
    )
    best_exact = metric(test_metrics, best, "exact_joint_accuracy")
    ref_exact = reference.get("exact_joint_accuracy", 0.0)
    best_support = metric(test_metrics, best, "average_total_support_used")
    ablation_support = ablation.get("average_total_support_used", 0.0)
    always_support = always.get("average_total_support_used", 0.0)
    support_saved_vs_ablation = ablation_support - best_support
    support_saved_vs_always = always_support - best_support
    false_conf = metric(test_metrics, best, "false_confidence_rate")
    indist_abstain = regime_metric(test_metrics, best, "INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT", "abstain_rate")
    controls_worse = (
        random_control.get("cost_adjusted_accuracy", 0.0) < metric(test_metrics, best, "cost_adjusted_accuracy")
        and content.get("cost_adjusted_accuracy", 0.0) < metric(test_metrics, best, "cost_adjusted_accuracy")
        and ablation.get("average_total_support_used", 0.0) > best_support
    )
    pass_gate = (
        best_exact >= ref_exact - 0.003
        and support_saved_vs_ablation >= 1.0
        and support_saved_vs_always >= 1.0
        and false_conf <= 0.01
        and indist_abstain >= 0.99
        and controls_worse
    )
    reason = {
        "best_exact": best_exact,
        "reference_exact": ref_exact,
        "best_support": best_support,
        "ablation_support": ablation_support,
        "always_counter_support": always_support,
        "support_saved_vs_ablation": support_saved_vs_ablation,
        "support_saved_vs_always": support_saved_vs_always,
        "false_confidence": false_conf,
        "indistinguishable_abstain": indist_abstain,
        "controls_worse_or_more_expensive": controls_worse,
    }
    if pass_gate:
        return {
            "decision": "rust_sparse_support_scoring_cost_control_confirmed",
            "verdict": "D66_RUST_SPARSE_SUPPORT_SCORING_COST_CONTROL_CONFIRMED",
            "next": "D67_RUST_SPARSE_SUPPORT_SCORING_SCALE_CONFIRM",
            "best_arm": best,
            "reason": reason,
        }
    return {
        "decision": "support_scoring_cost_control_not_confirmed",
        "verdict": "D66_SUPPORT_SCORING_COST_CONTROL_NOT_CONFIRMED",
        "next": "D66_REPAIR",
        "best_arm": best,
        "reason": reason,
    }


def write_reports(out, aggregate, decision):
    metrics = aggregate["test_metrics"]
    support_cost = make_support_cost_report(metrics["by_arm_core"])
    reports = {
        "d65r_upstream_manifest.json": aggregate["d65r_upstream_manifest"],
        "support_scoring_report.json": {
            arm: metrics["by_arm_core"].get(arm, {})
            for arm in [
                "D65R_RUST_SET_AGG_REFERENCE",
                "SUPPORT_SCORING_WITH_RUST_AGGREGATION",
                "COUNTER_SUPPORT_TRIAGE_WITH_RUST_AGGREGATION",
                "SUPPORT_BUDGET_CAPPED_RUST_AGGREGATION",
            ]
        },
        "support_triage_report.json": {
            arm: {
                "unnecessary_counter_support_rate": values.get("unnecessary_counter_support_rate", 0.0),
                "missed_counter_support_rate": values.get("missed_counter_support_rate", 0.0),
                "support_over_cheapest_correct_mean": values.get("support_over_cheapest_correct_mean", 0.0),
            }
            for arm, values in metrics["by_arm_core"].items()
        },
        "counter_support_triage_report.json": metrics["action_distribution"],
        "support_cost_frontier_report.json": support_cost,
        "cost_frontier_report.json": support_cost,
        "support_budget_report.json": {
            "budget_cap": SUPPORT_BUDGET_CAP,
            "capped_arm": metrics["by_arm_core"].get("SUPPORT_BUDGET_CAPPED_RUST_AGGREGATION", {}),
            "cost_matched_random": metrics["by_arm_core"].get("COST_MATCHED_RANDOM_SUPPORT_CONTROL", {}),
        },
        "ablation_compensation_report.json": {
            "reference": metrics["by_arm_core"].get("D65R_RUST_SET_AGG_REFERENCE", {}),
            "ablation": metrics["by_arm_core"].get("AGGREGATION_ABLATION_CONTROL", {}),
            "always_counter": metrics["by_arm_core"].get("ALWAYS_COUNTER_CONTROL", {}),
            "decision_reason": decision.get("reason"),
        },
        "content_corruption_report.json": metrics["by_arm_core"].get("SUPPORT_CONTENT_CORRUPTION_CONTROL", {}),
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
        "# D66 Rust Sparse Support Scoring With Aggregation Cost Control Result",
        "",
        "Decision:",
        "",
        "```text",
        f"decision = {decision['decision']}",
        f"verdict = {decision['verdict']}",
        f"next = {decision['next']}",
        f"best_arm = {decision.get('best_arm')}",
        "```",
        "",
        "| arm | exact | support | counter | cost-adjusted | unnecessary counter | missed counter | false conf |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in ARMS:
        values = metrics["by_arm_core"].get(arm, {})
        rows.append(
            f"| {arm} | {values.get('exact_joint_accuracy', 0.0):.6f} | "
            f"{values.get('average_total_support_used', 0.0):.4f} | "
            f"{values.get('average_counter_support_used', 0.0):.4f} | "
            f"{values.get('cost_adjusted_accuracy', 0.0):.6f} | "
            f"{values.get('unnecessary_counter_support_rate', 0.0):.6f} | "
            f"{values.get('missed_counter_support_rate', 0.0):.6f} | "
            f"{values.get('false_confidence_rate', 0.0):.6f} |"
        )
    rows += ["", "Boundary:", "", "```text", BOUNDARY, "```"]
    (out / "report.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", default="12501,12502,12503,12504,12505")
    parser.add_argument("--train-rows-per-seed", type=int, default=800)
    parser.add_argument("--test-rows-per-seed", type=int, default=800)
    parser.add_argument("--ood-rows-per-seed", type=int, default=800)
    parser.add_argument("--workers", default="auto")
    parser.add_argument("--cpu-target", default="50-75")
    parser.add_argument("--heartbeat-sec", type=float, default=20.0)
    parser.add_argument("--scale-mode", default="healthy")
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

    d65r_manifest = make_d65r_upstream_manifest(repo_root)
    write_json(out / "d65r_upstream_manifest.json", d65r_manifest)
    policy_controllers, learned_gate = load_policy_modules(repo_root)
    bundle = d55.d49_bundle()
    write_json(
        out / "dataset_manifest.json",
        {
            "task": "controlled_symbolic_joint_formula_discovery",
            "primary_space": PRIMARY_SPACE,
            "support_count": SUPPORT_COUNT,
            "regimes": REGIMES,
            "arms": ARMS,
            "truth_hidden_from_controller_inputs": True,
            "rust_arms_receive_support_evidence_set_representation": True,
            "python_precomputed_final_aggregate_label_used": False,
            "formula_solver_learning_used": False,
            "controller_only_not_formula_solver": True,
            "support_budget_cap": SUPPORT_BUDGET_CAP,
            "healthy_milestone_not_micro": True,
        },
    )

    train_rows = d51.make_rows_with_progress(seeds, args.train_rows_per_seed, "train", bundle, out, started, args.heartbeat_sec)
    write_json(out / "partial_training_rows_generated.json", {"rows": len(train_rows), "training_used": False})
    test_rows = d51.make_rows_with_progress(seeds, args.test_rows_per_seed, "test", bundle, out, started, args.heartbeat_sec)
    ood_rows = d51.make_rows_with_progress(seeds, args.ood_rows_per_seed, "ood", bundle, out, started, args.heartbeat_sec)

    try:
        test_outputs, test_rust = evaluate_split(
            test_rows,
            bundle,
            policy_controllers,
            learned_gate,
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
            learned_gate,
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
    rust_invocation = {"test": test_rust, "ood": ood_rust}
    rust_aggregation_rows = sum(
        data.get("aggregation", {}).get("rows_returned", 0)
        for data in rust_invocation.values()
    )
    rust_controller_rows = sum(
        data.get("controller", {}).get("rows_requested", 0)
        for data in rust_invocation.values()
    )
    fallback_rows = sum(
        values.get("python_fallback_rows", 0)
        for metrics in [test_metrics, ood_metrics]
        for values in metrics["rust_usage"].values()
    )
    precomputed_rows = sum(
        values.get("python_precomputed_final_aggregate_label_rows", 0)
        for metrics in [test_metrics, ood_metrics]
        for values in metrics["rust_usage"].values()
    )
    aggregate = {
        "task": TASK,
        "boundary": BOUNDARY,
        "scale_mode": args.scale_mode,
        "d65r_upstream_manifest": d65r_manifest,
        "test_metrics": test_metrics,
        "ood_metrics": ood_metrics,
        "rust_invocation_report": rust_invocation,
        "rust_path_invoked": rust_aggregation_rows > 0 and rust_controller_rows > 0,
        "rust_aggregation_rows": rust_aggregation_rows,
        "rust_controller_rows": rust_controller_rows,
        "fallback_rows": fallback_rows,
        "python_precomputed_final_aggregate_label_rows": precomputed_rows,
        "failed_jobs": failed_jobs,
    }
    decision = make_decision(test_metrics, failed_jobs)
    aggregate["decision"] = decision
    write_reports(out, aggregate, decision)
    write_json(out / "aggregate_metrics.json", aggregate)
    write_json(out / "decision.json", decision)
    summary = {
        "task": TASK,
        "decision": decision["decision"],
        "verdict": decision["verdict"],
        "next": decision["next"],
        "best_arm": decision.get("best_arm"),
        "rust_path_invoked": aggregate["rust_path_invoked"],
        "rust_aggregation_rows": rust_aggregation_rows,
        "rust_controller_rows": rust_controller_rows,
        "fallback_rows": fallback_rows,
        "python_precomputed_final_aggregate_label_rows": precomputed_rows,
        "failed_jobs": failed_jobs,
        "artifact_reports": [
            "d65r_upstream_manifest.json",
            "support_scoring_report.json",
            "support_triage_report.json",
            "counter_support_triage_report.json",
            "support_cost_frontier_report.json",
            "support_budget_report.json",
            "ablation_compensation_report.json",
            "content_corruption_report.json",
            "truth_leak_audit_report.json",
            "rust_invocation_report.json",
        ],
        "boundary": BOUNDARY,
    }
    write_json(out / "summary.json", summary)
    write_report(out, decision, test_metrics)
    append_progress(out, "completed", started, {"decision": decision["decision"], "failed_jobs": failed_jobs})
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
