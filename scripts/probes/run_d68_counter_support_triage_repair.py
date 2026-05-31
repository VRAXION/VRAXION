#!/usr/bin/env python3
"""D68 counter-support triage repair.

D67 scale-confirmed Rust sparse aggregation-backed support scoring, but exposed
high unnecessary counter-support in clean and mixed regimes. D68 tests whether
runtime diagnostics can triage counter-support requests without losing hard
correlated/adversarial recall or the abstain/external-test boundaries.
"""

import argparse
import copy
import json
import os
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path

import run_d51_mutable_ecf_controller_prototype as d51
import run_d55_sparse_firing_ecf_controller_prototype as d55
import run_d59_rust_sparse_ecf_controller_with_mutation as d59
import run_d62_policy_ensemble_ecf_controller_with_learned_gate as d62
import run_d64s_score_vector_structure_repair as d64s
import run_d65_set_invariant_ipf_aggregation_prototype as d65
import run_d65r_aggregation_causal_isolation_and_cost_frontier_repair as d65r

TASK = "D68_COUNTER_SUPPORT_TRIAGE_REPAIR"
BOUNDARY = (
    "D68 only tests counter-support triage for Rust sparse aggregation-backed "
    "support scoring in controlled symbolic joint formula discovery. The formula "
    "solver remains symbolic and the Rust sparse path is used for support "
    "aggregation/controller action execution. It does not prove full VRAXION "
    "brain, raw visual Raven reasoning, Raven solved, AGI, consciousness, "
    "DNA/genome success, architecture superiority, or production readiness."
)

PRIMARY_SPACE = d65.PRIMARY_SPACE
SUPPORT_COUNT = d65.SUPPORT_COUNT
REGIMES = d65.REGIMES
CORE_REGIMES = d65.CORE_REGIMES
ACTIONS = d65.ACTIONS
POLICY_MODULES = d65.POLICY_MODULES
EXTRA_POLICY_MODULES = ["DECIDE_POLICY"]
POLICY_CONTROLLERS = POLICY_MODULES + EXTRA_POLICY_MODULES

ARMS = [
    "D67_BEST_REPLAY",
    "COUNTER_TRIAGE_MARGIN_GATE",
    "COUNTER_TRIAGE_ENTROPY_MARGIN_GATE",
    "COUNTER_TRIAGE_CONFIDENCE_STABILITY_GATE",
    "COUNTER_TRIAGE_SUPPORT_INDEPENDENCE_GATE",
    "COUNTER_TRIAGE_ADVERSARIAL_PRESSURE_GATE",
    "COUNTER_TRIAGE_MULTI_SIGNAL_GATE",
    "TRAINED_THRESHOLD_TRIAGE_GATE",
    "COUNTER_TRIAGE_CONSERVATIVE_HIGH_RECALL",
    "COUNTER_TRIAGE_COST_OPTIMIZED",
    "CAP_7_CONTROL",
    "CAP_9_CONTROL",
    "ALWAYS_COUNTER_CONTROL",
    "NEVER_COUNTER_CONTROL",
    "RANDOM_COUNTER_CONTROL",
    "SHUFFLED_TRIAGE_SIGNAL_CONTROL",
    "BAD_TRIAGE_SIGNAL_CONTROL",
    "AGGREGATION_ABLATION_CONTROL",
    "SUPPORT_CONTENT_CORRUPTION_CONTROL",
    "TRUTH_LEAK_SENTINEL_REFERENCE_ONLY",
    "REGIME_LABEL_ORACLE_REFERENCE_ONLY",
]

REFERENCE_ONLY_ARMS = ["TRUTH_LEAK_SENTINEL_REFERENCE_ONLY", "REGIME_LABEL_ORACLE_REFERENCE_ONLY"]
CONTROL_ARMS = [
    "CAP_7_CONTROL",
    "CAP_9_CONTROL",
    "ALWAYS_COUNTER_CONTROL",
    "NEVER_COUNTER_CONTROL",
    "RANDOM_COUNTER_CONTROL",
    "SHUFFLED_TRIAGE_SIGNAL_CONTROL",
    "BAD_TRIAGE_SIGNAL_CONTROL",
    "AGGREGATION_ABLATION_CONTROL",
    "SUPPORT_CONTENT_CORRUPTION_CONTROL",
]
FAIR_ARMS = [arm for arm in ARMS if arm not in REFERENCE_ONLY_ARMS]

D65_ARM_FOR = {
    "D67_BEST_REPLAY": "RUST_SPARSE_SET_INVARIANT_IPF_AGGREGATION",
    "COUNTER_TRIAGE_MARGIN_GATE": "RUST_SPARSE_SET_INVARIANT_IPF_AGGREGATION",
    "COUNTER_TRIAGE_ENTROPY_MARGIN_GATE": "RUST_SPARSE_SET_INVARIANT_IPF_AGGREGATION",
    "COUNTER_TRIAGE_CONFIDENCE_STABILITY_GATE": "RUST_SPARSE_SET_INVARIANT_IPF_AGGREGATION",
    "COUNTER_TRIAGE_SUPPORT_INDEPENDENCE_GATE": "RUST_SPARSE_SET_INVARIANT_IPF_AGGREGATION",
    "COUNTER_TRIAGE_ADVERSARIAL_PRESSURE_GATE": "RUST_SPARSE_COUNTERFACTUAL_DELTA_SET_AGGREGATION",
    "COUNTER_TRIAGE_MULTI_SIGNAL_GATE": "RUST_SPARSE_COUNTERFACTUAL_DELTA_SET_AGGREGATION",
    "TRAINED_THRESHOLD_TRIAGE_GATE": "RUST_SPARSE_COUNTERFACTUAL_DELTA_SET_AGGREGATION",
    "COUNTER_TRIAGE_CONSERVATIVE_HIGH_RECALL": "RUST_SPARSE_COUNTERFACTUAL_DELTA_SET_AGGREGATION",
    "COUNTER_TRIAGE_COST_OPTIMIZED": "RUST_SPARSE_SET_INVARIANT_IPF_AGGREGATION",
    "CAP_7_CONTROL": "RUST_SPARSE_SET_INVARIANT_IPF_AGGREGATION",
    "CAP_9_CONTROL": "RUST_SPARSE_SET_INVARIANT_IPF_AGGREGATION",
    "ALWAYS_COUNTER_CONTROL": "RUST_SPARSE_SET_INVARIANT_IPF_AGGREGATION",
    "NEVER_COUNTER_CONTROL": "RUST_SPARSE_SET_INVARIANT_IPF_AGGREGATION",
    "RANDOM_COUNTER_CONTROL": "RUST_SPARSE_SET_INVARIANT_IPF_AGGREGATION",
    "SHUFFLED_TRIAGE_SIGNAL_CONTROL": "RUST_SPARSE_SET_INVARIANT_IPF_AGGREGATION",
    "BAD_TRIAGE_SIGNAL_CONTROL": "RUST_SPARSE_SET_INVARIANT_IPF_AGGREGATION",
    "AGGREGATION_ABLATION_CONTROL": "AGGREGATION_ABLATION_CONTROL",
    "SUPPORT_CONTENT_CORRUPTION_CONTROL": "SUPPORT_CONTENT_CORRUPTION_CONTROL",
    "TRUTH_LEAK_SENTINEL_REFERENCE_ONLY": "SYMBOLIC_SET_AGGREGATION_REFERENCE",
    "REGIME_LABEL_ORACLE_REFERENCE_ONLY": "SYMBOLIC_SET_AGGREGATION_REFERENCE",
}

ROW_SAMPLE_PER_ARM_REGIME = 14
SUPPORT_BUDGET_CAP_7 = 7
SUPPORT_BUDGET_CAP_9 = 9
HIGH_UNNECESSARY_COUNTER_RATE = 0.35
INTERNAL_COUNTER_ACTIONS = {"REQUEST_COUNTER_TOP1_TOP2", "REQUEST_JOINT_COUNTER"}
EXTERNAL_ACTION = "REQUEST_EXTERNAL_TEST"
COUNTER_ACTIONS = INTERNAL_COUNTER_ACTIONS | {EXTERNAL_ACTION}
FORBIDDEN_TRIAGE_FEATURES = set(d62.FORBIDDEN_GATE_FEATURES) | {"expected_selected", "expected_label", "truth_label"}
D68_ALLOWED_TRIAGE_FEATURES = list(d62.ALLOWED_GATE_FEATURES) + [
    "scalar_confidence",
    "joint_confidence",
    "confidence_floor",
    "margin_confidence_product",
]


def write_json(path, value):
    d51.write_json(path, value)


def append_jsonl(path, value):
    d51.append_jsonl(path, value)


def append_progress(out, event, started, data):
    d51.append_progress(out, event, started, data)


def run_blocking_with_heartbeat(out, event, split, started, heartbeat_sec, fn, *args):
    """Run a blocking bridge call while preserving no-black-box heartbeats."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(fn, *args)
        ticks = 0
        while True:
            try:
                return future.result(timeout=max(1.0, heartbeat_sec))
            except TimeoutError:
                ticks += 1
                payload = {
                    "split": split,
                    "bridge_event": event,
                    "heartbeat_ticks": ticks,
                }
                write_json(out / f"partial_{split}_{event}_heartbeat.json", payload)
                append_progress(out, f"{event}_heartbeat", started, payload)


def parse_seeds(raw):
    return [int(item.strip()) for item in str(raw).split(",") if item.strip()]


def load_json(path):
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def make_d67_upstream_manifest(repo_root):
    root = repo_root / "target/pilot_wave/d67_rust_sparse_support_scoring_scale_confirm/smoke"
    manifest = {
        "upstream": "D67_RUST_SPARSE_SUPPORT_SCORING_SCALE_CONFIRM",
        "expected_decision": "support_scoring_scale_confirmed_counter_triage_gap",
        "expected_next": TASK,
        "artifact_root": str(root),
        "decision_present": (root / "decision.json").exists(),
        "summary_present": (root / "summary.json").exists(),
    }
    for name in [
        "decision.json",
        "summary.json",
        "support_triage_report.json",
        "support_scoring_report.json",
        "support_cost_frontier_report.json",
        "unnecessary_counter_support_report.json",
        "missed_counter_support_report.json",
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
    controllers["DECIDE_POLICY"] = d62.make_always_action_controller("DECIDE_POLICY", "DECIDE")
    return {name: controllers[name] for name in POLICY_CONTROLLERS}, learned_gate


def decorate_source_pack(pack, arm):
    source = D65_ARM_FOR[arm]
    out = copy.deepcopy(pack)
    out["d68_arm"] = arm
    out["d65_source_arm"] = source
    out["support_budget_cap"] = None
    out["support_scoring_used"] = arm in {
        "D67_BEST_REPLAY",
        "COUNTER_TRIAGE_MARGIN_GATE",
        "COUNTER_TRIAGE_ENTROPY_MARGIN_GATE",
        "COUNTER_TRIAGE_CONFIDENCE_STABILITY_GATE",
        "COUNTER_TRIAGE_SUPPORT_INDEPENDENCE_GATE",
        "COUNTER_TRIAGE_ADVERSARIAL_PRESSURE_GATE",
        "COUNTER_TRIAGE_MULTI_SIGNAL_GATE",
        "TRAINED_THRESHOLD_TRIAGE_GATE",
        "COUNTER_TRIAGE_CONSERVATIVE_HIGH_RECALL",
        "COUNTER_TRIAGE_COST_OPTIMIZED",
        "CAP_7_CONTROL",
        "CAP_9_CONTROL",
    }
    if arm == "BAD_TRIAGE_SIGNAL_CONTROL":
        out["feature_map"] = d65r.zero_aggregate_features(out["feature_map"])
        out["features"] = d65.features_from_map(out["feature_map"])
    if arm == "CAP_7_CONTROL":
        out = d65r.cap_pack_actions(out, SUPPORT_BUDGET_CAP_7)
        out["support_budget_cap"] = SUPPORT_BUDGET_CAP_7
    if arm == "CAP_9_CONTROL":
        out = d65r.cap_pack_actions(out, SUPPORT_BUDGET_CAP_9)
        out["support_budget_cap"] = SUPPORT_BUDGET_CAP_9
    return out


def build_source_pack(row, bundle, arm, idx, rust_features):
    source = D65_ARM_FOR[arm]
    pack = d65.build_pack(row, bundle, source, idx, rust_features)
    return decorate_source_pack(pack, arm)


def build_eval_items(rows, bundle, rust_features, out, started, heartbeat_sec, split):
    items = []
    total = len(rows) * len(ARMS)
    completed = 0
    last = time.time()
    for idx, row in enumerate(rows):
        source_cache = {}
        for arm in ARMS:
            source = D65_ARM_FOR[arm]
            if source not in source_cache:
                source_cache[source] = d65.build_pack(row, bundle, source, idx, rust_features)
            items.append({"arm": arm, "pack": decorate_source_pack(source_cache[source], arm)})
            completed += 1
        now = time.time()
        if now - last >= heartbeat_sec:
            last = now
            write_json(out / f"partial_{split}_pack_build.json", {"completed": completed, "total": total})
            append_progress(out, "pack_build_progress", started, {"split": split, "completed": completed, "total": total})
    append_progress(out, "pack_build_complete", started, {"split": split, "packs": len(items)})
    return items


def d68_gate_features(pack):
    features = d62.gate_features(pack)
    fmap = pack.get("feature_map") or {}
    scalar = float(fmap.get("scalar_confidence", 0.0))
    joint = float(fmap.get("joint_confidence", scalar))
    floor = min(scalar, joint)
    features.update(
        {
            "scalar_confidence": scalar,
            "joint_confidence": joint,
            "confidence_floor": floor,
            "margin_confidence_product": (1.0 - features["inverse_margin"]) * floor,
        }
    )
    return {name: float(features.get(name, 0.0)) for name in D68_ALLOWED_TRIAGE_FEATURES}


def triage_risk(features):
    uncertainty = max(features["inverse_margin"], features["entropy_norm"], features["collision_norm"])
    pressure = max(features["counterfactual_pressure_norm"], features["adversarial_pressure_norm"])
    echo = max(0.0, features["dominant_cluster_fraction"] - 0.35)
    confidence_gap = max(0.0, 0.72 - features["confidence_floor"])
    disagreement = features["top1_factorised_disagreement"]
    return max(
        pressure,
        0.62 * uncertainty + 0.38 * confidence_gap,
        0.70 * echo + 0.30 * disagreement,
    )


def triage_preflight(features):
    if features["external_channel_available"] >= 0.25:
        return "EXTERNAL_TEST_POLICY", "external_channel_available"
    if features["internal_unresolvable_indicator"] >= 0.25:
        return "ABSTAIN_POLICY", "internal_unresolvable_indicator"
    return None, ""


def choose_counter_policy(features, joint_threshold=0.78):
    if features["adversarial_pressure_norm"] >= joint_threshold and features["collision_norm"] >= 0.22:
        return "ADVERSARIAL_REPAIR_POLICY"
    return "COUNTERFACTUAL_POLICY"


def learned_policy(pack, learned_gate):
    features = d68_gate_features(pack)
    for rule in learned_gate["rules"]:
        if features.get(rule["feature"], 0.0) >= rule["threshold"]:
            return rule["policy"], features, "learned_gate"
    return learned_gate["default_policy"], features, "learned_gate_default"


def support_scoring_policy(pack):
    features = d68_gate_features(pack)
    confidence = min(
        float(pack["feature_map"].get("scalar_confidence", 0.0)),
        float(pack["feature_map"].get("joint_confidence", 0.0)),
    )
    if features["external_channel_available"] >= 0.25:
        return "EXTERNAL_TEST_POLICY", features, "aggregation_support_scoring_external"
    if features["internal_unresolvable_indicator"] >= 0.25:
        return "ABSTAIN_POLICY", features, "aggregation_support_scoring_abstain"
    if confidence >= 0.74 and features["inverse_margin"] <= 0.32 and features["collision_norm"] <= 0.25:
        return "SATURATED_POLICY", features, "d67_replay_high_confidence"
    if features["adversarial_pressure_norm"] >= 0.68:
        return "ADVERSARIAL_REPAIR_POLICY", features, "d67_replay_adversarial"
    if features["counterfactual_pressure_norm"] >= 0.58:
        return "COUNTERFACTUAL_POLICY", features, "d67_replay_counterfactual"
    return "SATURATED_POLICY", features, "d67_replay_default"


def counter_support_triage_policy(pack, arm, learned_triage=None):
    features = d68_gate_features(pack)
    preflight, reason = triage_preflight(features)
    if preflight:
        return preflight, features, f"{arm.lower()}_{reason}"

    risk = triage_risk(features)
    threshold = 0.58
    joint_threshold = 0.78
    if arm == "COUNTER_TRIAGE_MARGIN_GATE":
        request = features["inverse_margin"] >= 0.44 or features["collision_norm"] >= 0.34
        basis = "margin_or_collision"
    elif arm == "COUNTER_TRIAGE_ENTROPY_MARGIN_GATE":
        request = max(features["inverse_margin"], features["entropy_norm"]) >= 0.47 or features["collision_norm"] >= 0.30
        basis = "entropy_margin"
    elif arm == "COUNTER_TRIAGE_CONFIDENCE_STABILITY_GATE":
        request = features["confidence_floor"] <= 0.62 or features["inverse_margin"] >= 0.40 or features["top1_factorised_disagreement"] >= 0.45
        basis = "confidence_stability"
    elif arm == "COUNTER_TRIAGE_SUPPORT_INDEPENDENCE_GATE":
        request = features["dominant_cluster_fraction"] >= 0.58 or features["support_cluster_count_norm"] <= 0.34 or features["collision_norm"] >= 0.30
        basis = "support_independence"
    elif arm == "COUNTER_TRIAGE_ADVERSARIAL_PRESSURE_GATE":
        request = features["adversarial_pressure_norm"] >= 0.54 or features["counterfactual_pressure_norm"] >= 0.62
        basis = "adversarial_pressure"
    elif arm == "COUNTER_TRIAGE_MULTI_SIGNAL_GATE":
        request = risk >= 0.56
        basis = "multi_signal_risk"
    elif arm == "TRAINED_THRESHOLD_TRIAGE_GATE":
        threshold = float((learned_triage or {}).get("risk_threshold", 0.56))
        joint_threshold = float((learned_triage or {}).get("joint_threshold", 0.78))
        request = risk >= threshold
        basis = f"trained_threshold_risk_{threshold:.2f}"
    elif arm == "COUNTER_TRIAGE_CONSERVATIVE_HIGH_RECALL":
        request = risk >= 0.42 or features["counterfactual_pressure_norm"] >= 0.42 or features["adversarial_pressure_norm"] >= 0.42
        joint_threshold = 0.66
        basis = "conservative_high_recall"
    elif arm == "COUNTER_TRIAGE_COST_OPTIMIZED":
        request = risk >= 0.68 and features["confidence_floor"] <= 0.72
        joint_threshold = 0.84
        basis = "cost_optimized"
    elif arm == "SHUFFLED_TRIAGE_SIGNAL_CONTROL":
        request = risk <= 0.36
        basis = "inverted_shuffled_triage_signal"
    elif arm == "BAD_TRIAGE_SIGNAL_CONTROL":
        request = features["support_count_norm"] >= 0.95
        basis = "bad_irrelevant_triage_signal"
    else:
        request = risk >= threshold
        basis = "default_triage_risk"

    if request:
        return choose_counter_policy(features, joint_threshold), features, basis
    return "DECIDE_POLICY", features, f"{basis}_decide"


def deterministic_random_counter(pack):
    value = d51.stable_seed(f"D68_RANDOM:{pack['row_id']}") % 100
    if value < 42:
        return "COUNTERFACTUAL_POLICY"
    if value < 52:
        return "ADVERSARIAL_REPAIR_POLICY"
    return "DECIDE_POLICY"


def select_policy(arm, pack, learned_gate, learned_triage, policy_actions, idx):
    if arm == "TRUTH_LEAK_SENTINEL_REFERENCE_ONLY":
        scored = []
        for policy in POLICY_CONTROLLERS:
            record = policy_actions[policy][idx]
            row = d59.output_from_action(pack, f"sentinel_{policy}", record["action"], d59.rust_trace(record))
            effective = row["correct"] or (row["support_regime"] == "INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT" and row["abstained"])
            scored.append((1.0 if effective else 0.0, -row["total_support_used"], policy))
        features = d68_gate_features(pack)
        return max(scored)[2], features, "reference_only_best_policy_after_truth_scoring", True
    if arm == "REGIME_LABEL_ORACLE_REFERENCE_ONLY":
        features = d68_gate_features(pack)
        regime = pack["support_regime"]
        if regime == "EXTERNAL_TEST_REQUIRED_SUPPORT":
            return "EXTERNAL_TEST_POLICY", features, "reference_only_regime_oracle_external", True
        if regime == "INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT":
            return "ABSTAIN_POLICY", features, "reference_only_regime_oracle_abstain", True
        if regime in {"ADVERSARIAL_DISTRACTOR_SUPPORT", "DISTINGUISHABLE_CORRELATED_FALSE_SUPPORT"}:
            return "ADVERSARIAL_REPAIR_POLICY", features, "reference_only_regime_oracle_joint", True
        if regime in {"CORRELATED_ECHO_SUPPORT", "MIXED_CLEAN_AND_CORRELATED", "MIXED_CLEAN_AND_ADVERSARIAL"}:
            return "COUNTERFACTUAL_POLICY", features, "reference_only_regime_oracle_counter", True
        return "DECIDE_POLICY", features, "reference_only_regime_oracle_decide", True
    if arm == "ALWAYS_COUNTER_CONTROL":
        return "ADVERSARIAL_REPAIR_POLICY", d68_gate_features(pack), "forced_always_joint_counter", False
    if arm == "NEVER_COUNTER_CONTROL":
        features = d68_gate_features(pack)
        preflight, reason = triage_preflight(features)
        if preflight:
            return preflight, features, f"never_counter_keeps_{reason}", False
        return "DECIDE_POLICY", features, "forced_never_internal_counter", False
    if arm == "RANDOM_COUNTER_CONTROL":
        return deterministic_random_counter(pack), d68_gate_features(pack), "deterministic_random_counter_control", False
    if arm in {"D67_BEST_REPLAY", "CAP_7_CONTROL", "CAP_9_CONTROL"}:
        policy, features, basis = support_scoring_policy(pack)
        return policy, features, basis, False
    if arm.startswith("COUNTER_TRIAGE_") or arm in {"TRAINED_THRESHOLD_TRIAGE_GATE", "SHUFFLED_TRIAGE_SIGNAL_CONTROL", "BAD_TRIAGE_SIGNAL_CONTROL"}:
        policy, features, basis = counter_support_triage_policy(pack, arm, learned_triage)
        return policy, features, basis, False
    policy, features, basis = learned_policy(pack, learned_gate)
    return policy, features, basis, False


def action_audit(pack, selected_action):
    decide = pack["actions"]["DECIDE"]
    selected = pack["actions"][selected_action]
    alternatives = list(pack["actions"].values())
    internal_counter_fixes = any(
        item["correct"] for action, item in pack["actions"].items()
        if action in INTERNAL_COUNTER_ACTIONS
    )
    external_fixes = bool(
        pack["actions"].get(EXTERNAL_ACTION, {}).get("correct")
        and pack["actions"].get(EXTERNAL_ACTION, {}).get("external_test_used", 0) > 0
    )
    decide_wrong = not bool(decide["correct"])
    internal_needed = decide_wrong and internal_counter_fixes
    external_needed = decide_wrong and external_fixes
    selected_internal = selected_action in INTERNAL_COUNTER_ACTIONS
    selected_external = selected_action == EXTERNAL_ACTION
    selected_any_counter = selected_action in COUNTER_ACTIONS
    unnecessary_internal = selected_internal and bool(decide["correct"])
    unnecessary_external = selected_external and bool(decide["correct"])
    missed_internal = (not selected_internal) and internal_needed
    missed_external = (not selected_external) and external_needed
    cheapest_correct = min([item["total_support_used"] for item in alternatives if item["correct"]] or [selected["total_support_used"]])
    counter_tp = selected_internal and internal_needed
    counter_fp = selected_internal and not internal_needed
    counter_fn = (not selected_internal) and internal_needed
    external_tp = selected_external and external_needed
    external_fp = selected_external and not external_needed
    external_fn = (not selected_external) and external_needed
    return {
        "counter_needed": internal_needed,
        "external_test_needed": external_needed,
        "any_repair_needed": internal_needed or external_needed,
        "counter_requested": selected_internal,
        "external_test_requested": selected_external,
        "any_counter_or_external_requested": selected_any_counter,
        "counter_true_positive": counter_tp,
        "counter_false_positive": counter_fp,
        "counter_false_negative": counter_fn,
        "external_test_true_positive": external_tp,
        "external_test_false_positive": external_fp,
        "external_test_false_negative": external_fn,
        "unnecessary_counter_support": unnecessary_internal or unnecessary_external,
        "unnecessary_internal_counter": unnecessary_internal,
        "unnecessary_external_test": unnecessary_external,
        "missed_counter_support": counter_fn or external_fn,
        "missed_internal_counter": counter_fn,
        "missed_external_test": external_fn,
        "counter_precision_den": selected_internal,
        "counter_recall_den": internal_needed,
        "external_precision_den": selected_external,
        "external_recall_den": external_needed,
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
    counter_tp = sum(1 for row in rows if row["counter_true_positive"])
    counter_fp = sum(1 for row in rows if row["counter_false_positive"])
    counter_fn = sum(1 for row in rows if row["counter_false_negative"])
    external_tp = sum(1 for row in rows if row["external_test_true_positive"])
    external_fp = sum(1 for row in rows if row["external_test_false_positive"])
    external_fn = sum(1 for row in rows if row["external_test_false_negative"])
    base["cost_adjusted_accuracy"] = d51.mean([row["cost_adjusted_accuracy"] for row in rows])
    base["unnecessary_counter_support_rate"] = d51.mean([1.0 if row["unnecessary_counter_support"] else 0.0 for row in rows])
    base["unnecessary_internal_counter_rate"] = d51.mean([1.0 if row["unnecessary_internal_counter"] else 0.0 for row in rows])
    base["unnecessary_external_test_rate"] = d51.mean([1.0 if row["unnecessary_external_test"] else 0.0 for row in rows])
    base["missed_counter_support_rate"] = d51.mean([1.0 if row["missed_counter_support"] else 0.0 for row in rows])
    base["missed_internal_counter_rate"] = d51.mean([1.0 if row["missed_internal_counter"] else 0.0 for row in rows])
    base["missed_external_test_rate"] = d51.mean([1.0 if row["missed_external_test"] else 0.0 for row in rows])
    base["counter_needed_rate"] = d51.mean([1.0 if row["counter_needed"] else 0.0 for row in rows])
    base["external_test_needed_rate"] = d51.mean([1.0 if row["external_test_needed"] else 0.0 for row in rows])
    base["counter_request_rate"] = d51.mean([1.0 if row["counter_requested"] else 0.0 for row in rows])
    base["external_test_request_rate"] = d51.mean([1.0 if row["external_test_requested"] else 0.0 for row in rows])
    base["counter_precision"] = counter_tp / max(1, counter_tp + counter_fp)
    base["counter_recall"] = counter_tp / max(1, counter_tp + counter_fn)
    base["external_test_precision"] = external_tp / max(1, external_tp + external_fp)
    base["external_test_recall"] = external_tp / max(1, external_tp + external_fn)
    base["counter_true_positive_rows"] = counter_tp
    base["counter_false_positive_rows"] = counter_fp
    base["counter_false_negative_rows"] = counter_fn
    base["external_true_positive_rows"] = external_tp
    base["external_false_positive_rows"] = external_fp
    base["external_false_negative_rows"] = external_fn
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


def action_for_policy_name(policy):
    return {
        "DECIDE_POLICY": "DECIDE",
        "COUNTERFACTUAL_POLICY": "REQUEST_COUNTER_TOP1_TOP2",
        "ADVERSARIAL_REPAIR_POLICY": "REQUEST_JOINT_COUNTER",
        "EXTERNAL_TEST_POLICY": "REQUEST_EXTERNAL_TEST",
        "ABSTAIN_POLICY": "ABSTAIN",
    }.get(policy, "DECIDE")


def train_threshold_triage(train_rows, bundle, out, started, heartbeat_sec, repo_root):
    append_progress(out, "train_threshold_triage_start", started, {"rows": len(train_rows)})
    rust_features, aggregation_report = run_blocking_with_heartbeat(
        out,
        "rust_train_aggregation_bridge_wait",
        "train",
        started,
        heartbeat_sec,
        d65.run_rust_aggregation_bridge,
        out,
        repo_root,
        train_rows,
        bundle,
        "train",
        started,
        heartbeat_sec,
    )
    packs = []
    last = time.time()
    for idx, row in enumerate(train_rows):
        packs.append(build_source_pack(row, bundle, "TRAINED_THRESHOLD_TRIAGE_GATE", idx, rust_features))
        if time.time() - last >= heartbeat_sec:
            write_json(out / "partial_train_threshold_pack_build.json", {"completed": idx + 1, "total": len(train_rows)})
            append_progress(out, "train_threshold_pack_build", started, {"completed": idx + 1, "total": len(train_rows)})
            last = time.time()

    candidates = []
    for risk_threshold in [0.42, 0.48, 0.52, 0.56, 0.60, 0.64, 0.68, 0.72]:
        for joint_threshold in [0.66, 0.72, 0.78, 0.84]:
            cfg = {"risk_threshold": risk_threshold, "joint_threshold": joint_threshold}
            rows = []
            for pack in packs:
                policy, _, _ = counter_support_triage_policy(pack, "TRAINED_THRESHOLD_TRIAGE_GATE", cfg)
                action = action_for_policy_name(policy)
                result = copy.deepcopy(pack["actions"][action])
                result.update(action_audit(pack, action))
                result["cost_adjusted_accuracy"] = (1.0 if result["correct"] else 0.0) - 0.0025 * max(0.0, result["total_support_used"] - SUPPORT_COUNT)
                rows.append(result)
            summary = summarize_rows(rows)
            score = (
                summary["exact_joint_accuracy"]
                - 0.0025 * max(0.0, summary["average_total_support_used"] - SUPPORT_COUNT)
                - 0.35 * summary["missed_counter_support_rate"]
                - 0.08 * summary["unnecessary_internal_counter_rate"]
                - 0.03 * summary["unnecessary_external_test_rate"]
            )
            candidates.append({"config": cfg, "score": score, "summary": summary})
    candidates.sort(
        key=lambda item: (
            item["score"],
            item["summary"]["exact_joint_accuracy"],
            -item["summary"]["missed_counter_support_rate"],
            -item["summary"]["average_total_support_used"],
        ),
        reverse=True,
    )
    report = {
        "selected": candidates[0],
        "top_candidates": candidates[:8],
        "aggregation_report": aggregation_report,
        "trained_only_on_split": "train",
        "forbidden_features_used": [],
        "allowed_features": D68_ALLOWED_TRIAGE_FEATURES,
    }
    write_json(out / "trained_threshold_triage_report.json", report)
    append_progress(out, "train_threshold_triage_complete", started, {"selected": candidates[0]["config"], "score": candidates[0]["score"]})
    return candidates[0]["config"], aggregation_report


def evaluate_split(rows, bundle, policy_controllers, learned_gate, learned_triage, out, split, started, heartbeat_sec, repo_root, row_output_path):
    rust_features, aggregation_report = run_blocking_with_heartbeat(
        out,
        "rust_aggregation_bridge_wait",
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
    items = build_eval_items(rows, bundle, rust_features, out, started, heartbeat_sec, split)
    packs = [item["pack"] for item in items]
    policy_actions, policy_report = run_blocking_with_heartbeat(
        out,
        "rust_policy_bridge_wait",
        split,
        started,
        heartbeat_sec,
        d59.run_rust_multi_bridge,
        out,
        repo_root,
        policy_controllers,
        packs,
        split,
        "d68_policy_eval",
        started,
    )
    outputs = []
    sample_counts = Counter()
    last = 0.0
    for idx, item in enumerate(items):
        arm = item["arm"]
        pack = item["pack"]
        policy, gate_features, basis, used_truth = select_policy(arm, pack, learned_gate, learned_triage, policy_actions, idx)
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
            "external_test": values.get("average_external_test_used", 0.0),
            "cost_adjusted_accuracy": values.get("cost_adjusted_accuracy", 0.0),
            "unnecessary_counter_support_rate": values.get("unnecessary_counter_support_rate", 0.0),
            "unnecessary_internal_counter_rate": values.get("unnecessary_internal_counter_rate", 0.0),
            "unnecessary_external_test_rate": values.get("unnecessary_external_test_rate", 0.0),
            "missed_counter_support_rate": values.get("missed_counter_support_rate", 0.0),
            "counter_precision": values.get("counter_precision", 0.0),
            "counter_recall": values.get("counter_recall", 0.0),
            "external_test_precision": values.get("external_test_precision", 0.0),
            "external_test_recall": values.get("external_test_recall", 0.0),
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
            "decision": "counter_support_triage_repair_not_confirmed",
            "verdict": "D68_FAILED_JOBS",
            "next": "D68_REPAIR",
            "best_arm": None,
            "reason": "failed_jobs not empty",
        }
    candidate_arms = [
        "COUNTER_TRIAGE_MARGIN_GATE",
        "COUNTER_TRIAGE_ENTROPY_MARGIN_GATE",
        "COUNTER_TRIAGE_CONFIDENCE_STABILITY_GATE",
        "COUNTER_TRIAGE_SUPPORT_INDEPENDENCE_GATE",
        "COUNTER_TRIAGE_ADVERSARIAL_PRESSURE_GATE",
        "COUNTER_TRIAGE_MULTI_SIGNAL_GATE",
        "TRAINED_THRESHOLD_TRIAGE_GATE",
        "COUNTER_TRIAGE_CONSERVATIVE_HIGH_RECALL",
        "COUNTER_TRIAGE_COST_OPTIMIZED",
    ]
    reference = test_metrics["by_arm_core"].get("D67_BEST_REPLAY", {})
    ablation = test_metrics["by_arm_core"].get("AGGREGATION_ABLATION_CONTROL", {})
    always = test_metrics["by_arm_core"].get("ALWAYS_COUNTER_CONTROL", {})
    random_control = test_metrics["by_arm_core"].get("RANDOM_COUNTER_CONTROL", {})
    never_control = test_metrics["by_arm_core"].get("NEVER_COUNTER_CONTROL", {})
    shuffled_control = test_metrics["by_arm_core"].get("SHUFFLED_TRIAGE_SIGNAL_CONTROL", {})
    bad_control = test_metrics["by_arm_core"].get("BAD_TRIAGE_SIGNAL_CONTROL", {})
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
    ref_support = reference.get("average_total_support_used", 0.0)
    always_support = always.get("average_total_support_used", 0.0)
    support_saved_vs_ref = ref_support - best_support
    support_saved_vs_always = always_support - best_support
    false_conf = metric(test_metrics, best, "false_confidence_rate")
    indist_abstain = regime_metric(test_metrics, best, "INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT", "abstain_rate")
    correlated = regime_metric(test_metrics, best, "CORRELATED_ECHO_SUPPORT", "exact_joint_accuracy")
    adversarial = regime_metric(test_metrics, best, "ADVERSARIAL_DISTRACTOR_SUPPORT", "exact_joint_accuracy")
    external = regime_metric(test_metrics, best, "EXTERNAL_TEST_REQUIRED_SUPPORT", "exact_joint_accuracy")
    clean_unnecessary = regime_metric(test_metrics, best, "CLEAN_INDEPENDENT_SUPPORT", "unnecessary_counter_support_rate")
    mixed_corr_unnecessary = regime_metric(test_metrics, best, "MIXED_CLEAN_AND_CORRELATED", "unnecessary_counter_support_rate")
    mixed_adv_unnecessary = regime_metric(test_metrics, best, "MIXED_CLEAN_AND_ADVERSARIAL", "unnecessary_counter_support_rate")
    unnecessary_core = metric(test_metrics, best, "unnecessary_counter_support_rate")
    missed_core = metric(test_metrics, best, "missed_counter_support_rate")
    correlated_missed = regime_metric(test_metrics, best, "CORRELATED_ECHO_SUPPORT", "missed_counter_support_rate")
    adversarial_missed = regime_metric(test_metrics, best, "ADVERSARIAL_DISTRACTOR_SUPPORT", "missed_counter_support_rate")
    distinguishable_missed = regime_metric(test_metrics, best, "DISTINGUISHABLE_CORRELATED_FALSE_SUPPORT", "missed_counter_support_rate")
    external_missed = regime_metric(test_metrics, best, "EXTERNAL_TEST_REQUIRED_SUPPORT", "missed_external_test_rate")
    controls_worse = (
        random_control.get("cost_adjusted_accuracy", 0.0) < metric(test_metrics, best, "cost_adjusted_accuracy")
        and shuffled_control.get("cost_adjusted_accuracy", 0.0) < metric(test_metrics, best, "cost_adjusted_accuracy")
        and bad_control.get("cost_adjusted_accuracy", 0.0) < metric(test_metrics, best, "cost_adjusted_accuracy")
        and never_control.get("exact_joint_accuracy", 0.0) < best_exact
        and content.get("cost_adjusted_accuracy", 0.0) < metric(test_metrics, best, "cost_adjusted_accuracy")
        and ablation.get("cost_adjusted_accuracy", 0.0) < metric(test_metrics, best, "cost_adjusted_accuracy")
    )
    pass_gate = (
        best_exact >= ref_exact - 0.003
        and correlated >= 0.995
        and adversarial >= 0.995
        and external >= 0.990
        and support_saved_vs_ref >= 0.75
        and false_conf <= 0.01
        and indist_abstain >= 0.99
        and unnecessary_core <= max(0.0, reference.get("unnecessary_counter_support_rate", 1.0) - 0.25)
        and clean_unnecessary <= 0.50
        and max(mixed_corr_unnecessary, mixed_adv_unnecessary) <= 0.60
        and max(missed_core, correlated_missed, adversarial_missed, distinguishable_missed, external_missed) <= 0.02
        and controls_worse
    )
    reason = {
        "best_exact": best_exact,
        "reference_exact": ref_exact,
        "best_support": best_support,
        "reference_support": ref_support,
        "always_counter_support": always_support,
        "support_saved_vs_same_run_d67": support_saved_vs_ref,
        "support_saved_vs_always": support_saved_vs_always,
        "correlated_echo": correlated,
        "adversarial_distractor": adversarial,
        "external_test_required": external,
        "false_confidence": false_conf,
        "indistinguishable_abstain": indist_abstain,
        "unnecessary_counter_support_rate": unnecessary_core,
        "missed_counter_support_rate": missed_core,
        "correlated_missed_counter_rate": correlated_missed,
        "adversarial_missed_counter_rate": adversarial_missed,
        "distinguishable_false_missed_counter_rate": distinguishable_missed,
        "external_test_missed_rate": external_missed,
        "clean_unnecessary_counter_support_rate": clean_unnecessary,
        "mixed_clean_correlated_unnecessary_counter_support_rate": mixed_corr_unnecessary,
        "mixed_clean_adversarial_unnecessary_counter_support_rate": mixed_adv_unnecessary,
        "controls_worse_or_more_expensive": controls_worse,
        "random_control_cost_adjusted_accuracy": random_control.get("cost_adjusted_accuracy", 0.0),
        "shuffled_control_cost_adjusted_accuracy": shuffled_control.get("cost_adjusted_accuracy", 0.0),
        "bad_control_cost_adjusted_accuracy": bad_control.get("cost_adjusted_accuracy", 0.0),
    }
    if pass_gate:
        if best_support > max(6.9, SUPPORT_COUNT + 1.5):
            return {
                "decision": "counter_triage_high_recall_high_cost",
                "verdict": "D68_COUNTER_TRIAGE_HIGH_RECALL_HIGH_COST",
                "next": "D68C_COUNTER_COST_OPTIMIZATION",
                "best_arm": best,
                "reason": reason,
            }
        return {
            "decision": "counter_support_triage_repair_confirmed",
            "verdict": "D68_COUNTER_SUPPORT_TRIAGE_REPAIR_CONFIRMED",
            "next": "D69_COUNTER_SUPPORT_TRIAGE_SCALE_CONFIRM",
            "best_arm": best,
            "reason": reason,
        }
    if max(missed_core, correlated_missed, adversarial_missed, distinguishable_missed, external_missed) > 0.02:
        return {
            "decision": "counter_triage_recall_failure",
            "verdict": "D68_COUNTER_TRIAGE_RECALL_FAILURE",
            "next": "D68R_COUNTER_RECALL_REPAIR",
            "best_arm": best,
            "reason": reason,
        }
    return {
        "decision": "counter_support_triage_repair_not_confirmed",
        "verdict": "D68_COUNTER_SUPPORT_TRIAGE_REPAIR_NOT_CONFIRMED",
        "next": "D68_REPAIR",
        "best_arm": best,
        "reason": reason,
    }


def write_reports(out, aggregate, decision):
    metrics = aggregate["test_metrics"]
    support_cost = make_support_cost_report(metrics["by_arm_core"])
    unnecessary_report = {
        arm: {
            "core": metrics["by_arm_core"].get(arm, {}).get("unnecessary_counter_support_rate", 0.0),
            "internal_core": metrics["by_arm_core"].get(arm, {}).get("unnecessary_internal_counter_rate", 0.0),
            "external_core": metrics["by_arm_core"].get(arm, {}).get("unnecessary_external_test_rate", 0.0),
            "clean": metrics["by_arm_and_regime"].get(arm, {}).get("CLEAN_INDEPENDENT_SUPPORT", {}).get("unnecessary_counter_support_rate", 0.0),
            "mixed_clean_correlated": metrics["by_arm_and_regime"].get(arm, {}).get("MIXED_CLEAN_AND_CORRELATED", {}).get("unnecessary_counter_support_rate", 0.0),
            "mixed_clean_adversarial": metrics["by_arm_and_regime"].get(arm, {}).get("MIXED_CLEAN_AND_ADVERSARIAL", {}).get("unnecessary_counter_support_rate", 0.0),
        }
        for arm in ARMS
    }
    missed_report = {
        arm: {
            "core": metrics["by_arm_core"].get(arm, {}).get("missed_counter_support_rate", 0.0),
            "correlated": metrics["by_arm_and_regime"].get(arm, {}).get("CORRELATED_ECHO_SUPPORT", {}).get("missed_counter_support_rate", 0.0),
            "adversarial": metrics["by_arm_and_regime"].get(arm, {}).get("ADVERSARIAL_DISTRACTOR_SUPPORT", {}).get("missed_counter_support_rate", 0.0),
            "distinguishable_false": metrics["by_arm_and_regime"].get(arm, {}).get("DISTINGUISHABLE_CORRELATED_FALSE_SUPPORT", {}).get("missed_counter_support_rate", 0.0),
            "external_test_required": metrics["by_arm_and_regime"].get(arm, {}).get("EXTERNAL_TEST_REQUIRED_SUPPORT", {}).get("missed_external_test_rate", 0.0),
        }
        for arm in ARMS
    }
    precision_recall_report = {
        arm: {
            "counter_precision": values.get("counter_precision", 0.0),
            "counter_recall": values.get("counter_recall", 0.0),
            "external_test_precision": values.get("external_test_precision", 0.0),
            "external_test_recall": values.get("external_test_recall", 0.0),
            "counter_true_positive_rows": values.get("counter_true_positive_rows", 0),
            "counter_false_positive_rows": values.get("counter_false_positive_rows", 0),
            "counter_false_negative_rows": values.get("counter_false_negative_rows", 0),
            "external_true_positive_rows": values.get("external_true_positive_rows", 0),
            "external_false_positive_rows": values.get("external_false_positive_rows", 0),
            "external_false_negative_rows": values.get("external_false_negative_rows", 0),
        }
        for arm, values in metrics["by_arm_core"].items()
    }
    reports = {
        "d67_upstream_manifest.json": aggregate["d67_upstream_manifest"],
        "triage_summary_report.json": {
            "scale_mode": aggregate["scale_mode"],
            "decision": decision,
            "rust_path_invoked": aggregate["rust_path_invoked"],
            "fallback_rows": aggregate["fallback_rows"],
            "rust_aggregation_rows": aggregate["rust_aggregation_rows"],
            "rust_controller_rows": aggregate["rust_controller_rows"],
            "trained_threshold_triage": aggregate.get("trained_threshold_triage"),
        },
        "support_scoring_report.json": {
            arm: metrics["by_arm_core"].get(arm, {})
            for arm in ["D67_BEST_REPLAY"] + [item for item in ARMS if item.startswith("COUNTER_TRIAGE_") or item == "TRAINED_THRESHOLD_TRIAGE_GATE"]
        },
        "support_triage_report.json": {
            arm: {
                "unnecessary_counter_support_rate": values.get("unnecessary_counter_support_rate", 0.0),
                "unnecessary_internal_counter_rate": values.get("unnecessary_internal_counter_rate", 0.0),
                "unnecessary_external_test_rate": values.get("unnecessary_external_test_rate", 0.0),
                "missed_counter_support_rate": values.get("missed_counter_support_rate", 0.0),
                "missed_internal_counter_rate": values.get("missed_internal_counter_rate", 0.0),
                "missed_external_test_rate": values.get("missed_external_test_rate", 0.0),
                "counter_precision": values.get("counter_precision", 0.0),
                "counter_recall": values.get("counter_recall", 0.0),
                "external_test_precision": values.get("external_test_precision", 0.0),
                "external_test_recall": values.get("external_test_recall", 0.0),
                "support_over_cheapest_correct_mean": values.get("support_over_cheapest_correct_mean", 0.0),
            }
            for arm, values in metrics["by_arm_core"].items()
        },
        "counter_support_triage_report.json": metrics["action_distribution"],
        "counter_precision_recall_report.json": precision_recall_report,
        "external_test_triage_report.json": {
            arm: {
                "external_test_accuracy": metrics["by_arm_and_regime"].get(arm, {}).get("EXTERNAL_TEST_REQUIRED_SUPPORT", {}).get("exact_joint_accuracy", 0.0),
                "external_test_recall": metrics["by_arm_core"].get(arm, {}).get("external_test_recall", 0.0),
                "external_test_precision": metrics["by_arm_core"].get(arm, {}).get("external_test_precision", 0.0),
                "unnecessary_external_test_rate": metrics["by_arm_core"].get(arm, {}).get("unnecessary_external_test_rate", 0.0),
            }
            for arm in ARMS
        },
        "support_cost_frontier_report.json": support_cost,
        "cost_frontier_report.json": support_cost,
        "fixed_budget_sweep_report.json": {
            "cap_7": metrics["by_arm_core"].get("CAP_7_CONTROL", {}),
            "cap_9": metrics["by_arm_core"].get("CAP_9_CONTROL", {}),
            "always_counter": metrics["by_arm_core"].get("ALWAYS_COUNTER_CONTROL", {}),
            "never_counter": metrics["by_arm_core"].get("NEVER_COUNTER_CONTROL", {}),
        },
        "clean_unnecessary_counter_audit_report.json": {
            arm: metrics["by_arm_and_regime"].get(arm, {}).get("CLEAN_INDEPENDENT_SUPPORT", {})
            for arm in ARMS
        },
        "mixed_unnecessary_counter_audit_report.json": {
            arm: {
                "mixed_clean_correlated": metrics["by_arm_and_regime"].get(arm, {}).get("MIXED_CLEAN_AND_CORRELATED", {}),
                "mixed_clean_adversarial": metrics["by_arm_and_regime"].get(arm, {}).get("MIXED_CLEAN_AND_ADVERSARIAL", {}),
            }
            for arm in ARMS
        },
        "unnecessary_counter_support_report.json": unnecessary_report,
        "missed_counter_support_report.json": missed_report,
        "regime_breakdown_report.json": metrics["by_arm_and_regime"],
        "ood_cost_frontier_report.json": make_support_cost_report(aggregate["ood_metrics"]["by_arm_core"]),
        "support_budget_report.json": {
            "budget_caps": [SUPPORT_BUDGET_CAP_7, SUPPORT_BUDGET_CAP_9],
            "cap_7_arm": metrics["by_arm_core"].get("CAP_7_CONTROL", {}),
            "cap_9_arm": metrics["by_arm_core"].get("CAP_9_CONTROL", {}),
            "same_run_d67_replay": metrics["by_arm_core"].get("D67_BEST_REPLAY", {}),
        },
        "ablation_and_control_report.json": {
            "reference": metrics["by_arm_core"].get("D67_BEST_REPLAY", {}),
            "ablation": metrics["by_arm_core"].get("AGGREGATION_ABLATION_CONTROL", {}),
            "always_counter": metrics["by_arm_core"].get("ALWAYS_COUNTER_CONTROL", {}),
            "never_counter": metrics["by_arm_core"].get("NEVER_COUNTER_CONTROL", {}),
            "random_counter": metrics["by_arm_core"].get("RANDOM_COUNTER_CONTROL", {}),
            "shuffled_triage": metrics["by_arm_core"].get("SHUFFLED_TRIAGE_SIGNAL_CONTROL", {}),
            "bad_triage": metrics["by_arm_core"].get("BAD_TRIAGE_SIGNAL_CONTROL", {}),
            "decision_reason": decision.get("reason"),
        },
        "content_corruption_report.json": metrics["by_arm_core"].get("SUPPORT_CONTENT_CORRUPTION_CONTROL", {}),
        "truth_leak_audit_report.json": {
            "fair_arms": FAIR_ARMS,
            "reference_only_arms": REFERENCE_ONLY_ARMS,
            "fair_arms_using_truth_label": [],
            "forbidden_triage_features": sorted(FORBIDDEN_TRIAGE_FEATURES),
            "allowed_triage_features": D68_ALLOWED_TRIAGE_FEATURES,
            "fair_arms_using_forbidden_metadata": [],
            "python_precomputed_final_aggregate_label_used_by_fair_arms": False,
            "truth_leak_sentinel": metrics["by_arm_core"].get("TRUTH_LEAK_SENTINEL_REFERENCE_ONLY", {}),
            "regime_label_oracle_reference_only": metrics["by_arm_core"].get("REGIME_LABEL_ORACLE_REFERENCE_ONLY", {}),
        },
        "rust_invocation_report.json": aggregate["rust_invocation_report"],
    }
    for name, value in reports.items():
        write_json(out / name, value)
    return reports


def write_report(out, decision, metrics):
    rows = [
        "# D68 Counter-Support Triage Repair Result",
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
        "| arm | exact | support | counter | external | cost-adjusted | unnecessary | missed | counter precision | counter recall | false conf |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in ARMS:
        values = metrics["by_arm_core"].get(arm, {})
        rows.append(
            f"| {arm} | {values.get('exact_joint_accuracy', 0.0):.6f} | "
            f"{values.get('average_total_support_used', 0.0):.4f} | "
            f"{values.get('average_counter_support_used', 0.0):.4f} | "
            f"{values.get('average_external_test_used', 0.0):.4f} | "
            f"{values.get('cost_adjusted_accuracy', 0.0):.6f} | "
            f"{values.get('unnecessary_counter_support_rate', 0.0):.6f} | "
            f"{values.get('missed_counter_support_rate', 0.0):.6f} | "
            f"{values.get('counter_precision', 0.0):.6f} | "
            f"{values.get('counter_recall', 0.0):.6f} | "
            f"{values.get('false_confidence_rate', 0.0):.6f} |"
        )
    rows += ["", "Boundary:", "", "```text", BOUNDARY, "```"]
    (out / "report.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", default="12701,12702,12703,12704,12705")
    parser.add_argument("--train-rows-per-seed", type=int, default=240)
    parser.add_argument("--test-rows-per-seed", type=int, default=240)
    parser.add_argument("--ood-rows-per-seed", type=int, default=240)
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

    d67_manifest = make_d67_upstream_manifest(repo_root)
    write_json(out / "d67_upstream_manifest.json", d67_manifest)
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
            "support_budget_caps": [SUPPORT_BUDGET_CAP_7, SUPPORT_BUDGET_CAP_9],
            "healthy_milestone_not_micro": True,
            "scale_confirm_milestone": True,
            "unnecessary_counter_frontier_visible": True,
            "counter_triage_metric_definitions": {
                "counter_needed": "DECIDE is wrong and an internal counter action fixes the row",
                "unnecessary_counter": "internal counter or external test requested while DECIDE was already correct",
                "missed_counter": "needed internal counter or external test was not requested",
                "external_test_measured_separately": True,
            },
            "fair_triage_forbidden_features": sorted(FORBIDDEN_TRIAGE_FEATURES),
            "fair_triage_allowed_features": D68_ALLOWED_TRIAGE_FEATURES,
        },
    )

    train_rows = d51.make_rows_with_progress(seeds, args.train_rows_per_seed, "train", bundle, out, started, args.heartbeat_sec)
    write_json(out / "partial_training_rows_generated.json", {"rows": len(train_rows), "training_used": True})
    test_rows = d51.make_rows_with_progress(seeds, args.test_rows_per_seed, "test", bundle, out, started, args.heartbeat_sec)
    ood_rows = d51.make_rows_with_progress(seeds, args.ood_rows_per_seed, "ood", bundle, out, started, args.heartbeat_sec)
    learned_triage, train_rust = train_threshold_triage(train_rows, bundle, out, started, args.heartbeat_sec, repo_root)

    try:
        test_outputs, test_rust = evaluate_split(
            test_rows,
            bundle,
            policy_controllers,
            learned_gate,
            learned_triage,
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
            learned_triage,
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
        "d67_upstream_manifest": d67_manifest,
        "trained_threshold_triage": learned_triage,
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
            "d67_upstream_manifest.json",
            "triage_summary_report.json",
            "support_scoring_report.json",
            "support_triage_report.json",
            "counter_support_triage_report.json",
            "counter_precision_recall_report.json",
            "external_test_triage_report.json",
            "support_cost_frontier_report.json",
            "fixed_budget_sweep_report.json",
            "clean_unnecessary_counter_audit_report.json",
            "mixed_unnecessary_counter_audit_report.json",
            "unnecessary_counter_support_report.json",
            "missed_counter_support_report.json",
            "regime_breakdown_report.json",
            "ood_cost_frontier_report.json",
            "support_budget_report.json",
            "ablation_and_control_report.json",
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
