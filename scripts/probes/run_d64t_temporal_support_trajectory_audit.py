#!/usr/bin/env python3
"""D64T temporal/support trajectory audit.

D64S found that most score-vector structure shuffles did not create a clean
dependency gap, while the old SUPPORT_ORDER_SHUFFLE was not a raw support-order
test. D64T manipulates the actual five support slots before feature generation
and then replays the Rust sparse ECF action-controller path.
"""

import argparse
import copy
import json
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import run_d49_joint_cell_operator_discovery_with_robust_support as d49
import run_d49b_joint_binding_repair as d49b
import run_d51_mutable_ecf_controller_prototype as d51
import run_d55_sparse_firing_ecf_controller_prototype as d55
import run_d59_rust_sparse_ecf_controller_with_mutation as d59
import run_d62_policy_ensemble_ecf_controller_with_learned_gate as d62
import run_d64s_score_vector_structure_repair as d64s

TASK = "D64T_TEMPORAL_SUPPORT_TRAJECTORY_AUDIT"
BOUNDARY = (
    "D64T only audits whether raw support trajectory/order carries useful signal "
    "for the Rust sparse ECF action controller in controlled symbolic joint formula "
    "discovery. It does not prove full VRAXION brain, raw visual Raven reasoning, "
    "Raven solved, AGI, consciousness, DNA/genome success, architecture superiority, "
    "or production readiness."
)

PRIMARY_SPACE = d62.PRIMARY_SPACE
SUPPORT_COUNT = d62.SUPPORT_COUNT
REGIMES = d62.REGIMES
CORE_REGIMES = d62.CORE_REGIMES
ACTIONS = d62.ACTIONS
POLICY_MODULES = d62.POLICY_MODULE_ARMS
ROW_SAMPLE_PER_ARM_REGIME_SPLIT = 18

ARMS = [
    "ORIGINAL_TRAJECTORY_REFERENCE",
    "SET_INVARIANT_AGGREGATION",
    "RANDOM_SUPPORT_ORDER_SHUFFLE",
    "STAGE_PRESERVING_SHUFFLE",
    "STAGE_DESTROYING_SHUFFLE",
    "A_THEN_B_VS_B_THEN_A",
    "POS_THEN_NEG_VS_NEG_THEN_POS",
    "EARLY_COUNTER_SUPPORT",
    "LATE_COUNTER_SUPPORT",
    "OPEN_LOOP_SUPPORT_SEQUENCE",
    "CLOSED_LOOP_FIELD_DEPENDENT_SUPPORT_SEQUENCE",
    "FINAL_STATE_READOUT_ONLY",
    "FULL_TRAJECTORY_READOUT",
    "ARBITRARY_ORDER_ID_CONTROL",
    "TRUTH_LEAK_SENTINEL_REFERENCE_ONLY",
]

CONTROL_ARMS = [
    "RANDOM_SUPPORT_ORDER_SHUFFLE",
    "STAGE_DESTROYING_SHUFFLE",
    "ARBITRARY_ORDER_ID_CONTROL",
]

REFERENCE_ONLY_ARMS = ["TRUTH_LEAK_SENTINEL_REFERENCE_ONLY"]
FAIR_ARMS = [arm for arm in ARMS if arm not in REFERENCE_ONLY_ARMS]


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


def make_d64s_upstream_manifest(repo_root):
    root = repo_root / "target/pilot_wave/d64s_score_vector_structure_repair/smoke"
    manifest = {
        "upstream": "D64S_SCORE_VECTOR_STRUCTURE_REPAIR",
        "expected_decision": "score_vector_structure_dependency_not_confirmed",
        "expected_next": "D64S_REPAIR_OR_REDEFINE_DIAGNOSTIC_CLAIM",
        "artifact_root": str(root),
        "decision_present": (root / "decision.json").exists(),
        "structure_gap_report_present": (root / "structure_gap_report.json").exists(),
    }
    for name in ["decision.json", "summary.json", "structure_gap_report.json", "aggregate_metrics.json"]:
        value = load_json_if_present(root / name)
        if value is not None:
            manifest[name.replace(".json", "")] = value
    known_gap = ((manifest.get("structure_gap_report") or {}).get("structure_gaps") or {})
    manifest["d64s_support_order_shuffle_gap"] = known_gap.get("SUPPORT_ORDER_SHUFFLE")
    manifest["d64s_support_order_was_diagnostic_shuffle_not_raw_sequence"] = True
    return manifest


def stable_rng(seed, tag):
    return random.Random(int(seed) + d51.stable_seed(tag))


def vector_signature(vector):
    return "|".join(f"{key}:{float(value):.6f}" for key, value in sorted(vector.items()))


def canonical_vector_sort(items):
    return sorted(items, key=lambda item: (vector_signature(item["vector"]), item["source_slot"], item["stage"]))


def clone_item(item, stage=None):
    out = {"stage": item["stage"], "vector": item["vector"], "source_slot": item["source_slot"]}
    if stage is not None:
        out["stage"] = stage
    return out


def make_item(vector, stage, source_slot):
    return {"stage": stage, "vector": vector, "source_slot": source_slot}


def pad_to_support_count(items, original):
    out = list(items[:SUPPORT_COUNT])
    idx = 0
    while len(out) < SUPPORT_COUNT:
        out.append(clone_item(original[idx % len(original)]))
        idx += 1
    return out[:SUPPORT_COUNT]


def base_sequence_context(row, bundle):
    base_vectors = d49b.cached_base_vectors(row, bundle, SUPPORT_COUNT)
    base_scores = d49b.aggregate_sum(base_vectors)
    base_pred = d49b.predict(base_scores, bundle)
    original = [make_item(vector, "original", idx) for idx, vector in enumerate(base_vectors)]
    counter_vectors = d49b.make_stage_vectors(row, bundle, base_pred, "joint", SUPPORT_COUNT)
    counter = [make_item(vector, "counter", idx) for idx, vector in enumerate(counter_vectors)]
    random_vectors = d49.random_extra_vectors(row, bundle)[:SUPPORT_COUNT]
    random_items = [make_item(vector, "random", idx) for idx, vector in enumerate(random_vectors)]
    external = []
    if row.get("external_test_available"):
        external_vectors = d49b.make_stage_vectors(row, bundle, base_pred, "joint", SUPPORT_COUNT, external=True)
        external = [make_item(vector, "external", idx) for idx, vector in enumerate(external_vectors)]
    return {
        "original": original,
        "counter": counter,
        "random": random_items,
        "external": external,
        "base_vectors": base_vectors,
        "base_scores": base_scores,
        "base_pred": base_pred,
    }


def effect_sort_items(items, base_pred):
    ordered = base_pred.get("ordered") or []
    if len(ordered) < 2:
        return list(items)
    top1 = ordered[0][0]
    top2 = ordered[1][0]

    def effect(item):
        vector = item["vector"]
        return float(vector.get(top1, 0.0)) - float(vector.get(top2, 0.0))

    return sorted(items, key=lambda item: (-effect(item), vector_signature(item["vector"])))


def apply_variant(row, bundle, arm):
    context = base_sequence_context(row, bundle)
    original = context["original"]
    counter = context["counter"]
    random_items = context["random"]
    external = context["external"]
    base_pred = context["base_pred"]
    rng = stable_rng(row["seed"], f"D64T:{arm}:{row['row_id']}:{row['split']}")
    stage_labels_preserved = True
    trajectory_readout_enabled = arm != "FINAL_STATE_READOUT_ONLY"
    arbitrary_order_id_used = False

    if arm == "ORIGINAL_TRAJECTORY_REFERENCE":
        items = [clone_item(item) for item in original]
        basis = "natural_five_support_sequence"
    elif arm == "SET_INVARIANT_AGGREGATION":
        items = canonical_vector_sort([clone_item(item) for item in original])
        basis = "canonical_vector_sort_before_feature_generation"
    elif arm == "RANDOM_SUPPORT_ORDER_SHUFFLE":
        items = [clone_item(item) for item in original]
        rng.shuffle(items)
        basis = "deterministic_random_shuffle_of_actual_support_slots"
    elif arm == "STAGE_PRESERVING_SHUFFLE":
        mixed = pad_to_support_count([clone_item(item) for item in original[:3]] + [clone_item(item) for item in counter[:2]], original)
        groups = defaultdict(list)
        for item in mixed:
            groups[item["stage"]].append(item)
        for group in groups.values():
            rng.shuffle(group)
        items = []
        for stage in ["original", "counter", "random", "external"]:
            items.extend(groups.get(stage, []))
        items = pad_to_support_count(items, original)
        basis = "mixed_sequence_shuffled_within_stage_only"
    elif arm == "STAGE_DESTROYING_SHUFFLE":
        mixed = pad_to_support_count([clone_item(item) for item in original[:3]] + [clone_item(item) for item in counter[:2]], original)
        rng.shuffle(mixed)
        label_cycle = ["random", "counter", "original", "external", "counter"]
        items = [clone_item(item, stage=label_cycle[idx % len(label_cycle)]) for idx, item in enumerate(mixed)]
        stage_labels_preserved = False
        basis = "mixed_sequence_shuffled_with_stage_labels_destroyed"
    elif arm == "A_THEN_B_VS_B_THEN_A":
        items = [clone_item(item) for item in original]
        if len(items) >= 2:
            items[0], items[1] = items[1], items[0]
        basis = "first_two_support_slots_reversed_before_feature_generation"
    elif arm == "POS_THEN_NEG_VS_NEG_THEN_POS":
        mixed = pad_to_support_count([clone_item(item) for item in counter[:3]] + [clone_item(item) for item in original[:2]], original)
        items = effect_sort_items(mixed, base_pred)
        basis = "support_slots_sorted_by_top1_vs_top2_effect"
    elif arm == "EARLY_COUNTER_SUPPORT":
        items = pad_to_support_count([clone_item(item) for item in counter[:2]] + [clone_item(item) for item in original[:3]], original)
        basis = "counter_support_slots_placed_early"
    elif arm == "LATE_COUNTER_SUPPORT":
        items = pad_to_support_count([clone_item(item) for item in original[:3]] + [clone_item(item) for item in counter[:2]], original)
        basis = "counter_support_slots_placed_late"
    elif arm == "OPEN_LOOP_SUPPORT_SEQUENCE":
        items = pad_to_support_count([clone_item(item) for item in original[:3]] + [clone_item(item) for item in random_items[:2]], original)
        basis = "fixed_extra_support_independent_of_current_field"
    elif arm == "CLOSED_LOOP_FIELD_DEPENDENT_SUPPORT_SEQUENCE":
        if d49b.counter_needed(row, base_pred, context["base_vectors"]):
            items = pad_to_support_count([clone_item(item) for item in original[:3]] + [clone_item(item) for item in counter[:2]], original)
            basis = "counter_slots_added_only_when_current_field_requests_them"
        else:
            items = [clone_item(item) for item in original]
            basis = "no_counter_slots_added_when_current_field_is_stable"
    elif arm == "FINAL_STATE_READOUT_ONLY":
        items = pad_to_support_count([clone_item(item) for item in original[:3]] + [clone_item(item) for item in counter[:2]], original)
        basis = "same_support_as_full_trajectory_but_sequence_features_disabled"
    elif arm == "FULL_TRAJECTORY_READOUT":
        if external and row.get("external_test_available"):
            items = pad_to_support_count([clone_item(item) for item in original[:3]] + [clone_item(item) for item in external[:2]], original)
        elif d49b.counter_needed(row, base_pred, context["base_vectors"]):
            items = pad_to_support_count([clone_item(item) for item in original[:3]] + [clone_item(item) for item in counter[:2]], original)
        else:
            items = [clone_item(item) for item in original]
        basis = "closed_loop_sequence_plus_trajectory_readout"
    elif arm == "ARBITRARY_ORDER_ID_CONTROL":
        items = [clone_item(item) for item in original]
        rng.shuffle(items)
        arbitrary_order_id_used = True
        basis = "row_id_seeded_order_id_control_not_a_fair_signal"
    elif arm == "TRUTH_LEAK_SENTINEL_REFERENCE_ONLY":
        items = [clone_item(item) for item in original]
        basis = "reference_only_best_policy_selected_after_truth_scoring"
    else:
        raise ValueError(arm)

    return {
        "items": pad_to_support_count(items, original),
        "basis": basis,
        "stage_labels_preserved": stage_labels_preserved,
        "trajectory_readout_enabled": trajectory_readout_enabled,
        "arbitrary_order_id_used": arbitrary_order_id_used,
        "base_pred": base_pred,
        "original": original,
    }


def cumulative_path_stats(items, bundle):
    scores = {candidate: 0.0 for candidate in bundle["candidates"]}
    margins = []
    preds = []
    entropies = []
    for item in items:
        for key, value in item["vector"].items():
            scores[key] = scores.get(key, 0.0) + float(value)
        pred = d49b.predict(scores, bundle)
        preds.append(pred["pred_joint"])
        margins.append(float(pred["top1_top2_margin"]))
        entropies.append(float(pred["entropy"]))
    flips = sum(1 for idx in range(1, len(preds)) if preds[idx] != preds[idx - 1])
    first_margin = margins[0] if margins else 0.0
    final_margin = margins[-1] if margins else 0.0
    return {
        "path_flip_count": flips,
        "path_flip_norm": min(1.0, flips / max(1.0, len(items) - 1)),
        "first_margin": first_margin,
        "final_margin": final_margin,
        "margin_delta": final_margin - first_margin,
        "margin_delta_norm": max(0.0, min(1.0, (final_margin - first_margin + 4.0) / 8.0)),
        "max_entropy": max(entropies) if entropies else 0.0,
        "pred_path": preds,
    }


def trajectory_stats(row, bundle, variant):
    items = variant["items"]
    original_sigs = [vector_signature(item["vector"]) for item in variant["original"]]
    sigs = [vector_signature(item["vector"]) for item in items]
    stage_sequence = [item["stage"] for item in items]
    counter_like = {"counter", "external"}
    stage_transitions = sum(1 for idx in range(1, len(stage_sequence)) if stage_sequence[idx] != stage_sequence[idx - 1])
    order_disagreement = sum(1 for idx, sig in enumerate(sigs) if idx >= len(original_sigs) or sig != original_sigs[idx]) / float(SUPPORT_COUNT)
    path = cumulative_path_stats(items, bundle)
    counter_early = sum(1 for item in items[:2] if item["stage"] in counter_like) / 2.0
    counter_late = sum(1 for item in items[-2:] if item["stage"] in counter_like) / 2.0
    stages = Counter(stage_sequence)
    return {
        "stage_sequence": stage_sequence,
        "stage_counts": dict(stages),
        "stage_transition_norm": stage_transitions / max(1.0, SUPPORT_COUNT - 1),
        "counter_early_fraction": counter_early,
        "counter_late_fraction": counter_late,
        "counter_stage_fraction": sum(stages.get(stage, 0) for stage in counter_like) / float(SUPPORT_COUNT),
        "random_stage_fraction": stages.get("random", 0) / float(SUPPORT_COUNT),
        "order_disagreement_norm": order_disagreement,
        "stage_labels_preserved": bool(variant["stage_labels_preserved"]),
        "trajectory_readout_enabled": bool(variant["trajectory_readout_enabled"]),
        "arbitrary_order_id_used": bool(variant["arbitrary_order_id_used"]),
        **path,
    }


def build_trajectory_pack(row, bundle, arm):
    variant = apply_variant(row, bundle, arm)
    stats = trajectory_stats(row, bundle, variant)
    base_vectors = [item["vector"] for item in variant["items"]]
    scalar_scores = d49b.aggregate_sum(base_vectors)
    scalar_pred = d49b.predict(scalar_scores, bundle)
    features, feature_map = d51.build_features(row, bundle, base_vectors, scalar_scores, scalar_pred)
    feature_map = copy.deepcopy(feature_map)
    feature_map["runtime_support_budget_available"] = 0.0
    feature_map["hard_support_budget_cap_norm"] = 0.0
    feature_map["support_budget_pressure_norm"] = max(0.0, stats["counter_stage_fraction"] - 0.20)
    feature_map["counterfactual_pressure_norm"] = max(
        float(feature_map.get("inverse_margin", 0.0)),
        stats["path_flip_norm"],
        stats["counter_late_fraction"] * 0.5,
    )
    feature_map["runtime_adversarial_pressure_norm"] = max(
        float(feature_map.get("collision_norm", 0.0)),
        float(feature_map.get("dominant_cluster_fraction", 0.0)),
        1.0 - float(stats["stage_labels_preserved"]),
        stats["random_stage_fraction"],
    )
    feature_map["context_noise_norm"] = 0.0
    if not stats["trajectory_readout_enabled"]:
        feature_map["support_budget_pressure_norm"] = 0.0
        feature_map["counterfactual_pressure_norm"] = float(feature_map.get("inverse_margin", 0.0))
        feature_map["runtime_adversarial_pressure_norm"] = max(
            float(feature_map.get("collision_norm", 0.0)),
            float(feature_map.get("dominant_cluster_fraction", 0.0)),
        )
    if stats["arbitrary_order_id_used"]:
        rng = stable_rng(row["seed"], f"arbitrary_order_id:{row['row_id']}")
        feature_map["context_noise_norm"] = rng.random()
        feature_map["counterfactual_pressure_norm"] = max(feature_map["counterfactual_pressure_norm"], feature_map["context_noise_norm"])

    base = (base_vectors, scalar_scores, scalar_pred)
    actions = {action: d51.run_action(row, bundle, action, base=base, arm_name=f"ACTION_{action}") for action in ACTIONS}
    pack = {
        "row_id": row["row_id"],
        "seed": row["seed"],
        "split": row["split"],
        "support_regime": row["support_regime"],
        "features": features,
        "feature_map": feature_map,
        "actions": actions,
        "action_compact": {action: d51.compact_outcome(result) for action, result in actions.items()},
        "references": {},
        "indistinguishable_case": row["support_regime"] == "INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT",
        "external_case": row["support_regime"] == "EXTERNAL_TEST_REQUIRED_SUPPORT",
        "track": "D64T_TRAJECTORY_AUDIT",
        "mixed_source_track": "D64T_TRAJECTORY_AUDIT",
        "trajectory_arm": arm,
        "trajectory_basis": variant["basis"],
        "trajectory_stats": stats,
        "truth_hidden_from_controller_inputs": True,
    }
    return pack


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


def policy_from_pack(pack, learned_gate):
    features = d62.gate_features(pack)
    for rule in learned_gate["rules"]:
        if features.get(rule["feature"], 0.0) >= rule["threshold"]:
            return rule["policy"], features, "learned_gate_over_raw_trajectory_features"
    return learned_gate["default_policy"], features, "learned_gate_default_over_raw_trajectory_features"


def truth_leak_policy(pack, rust_actions, idx):
    scored = []
    for policy in POLICY_MODULES:
        record = rust_actions[policy][idx]
        row = d59.output_from_action(pack, f"sentinel_{policy}", record["action"], d59.rust_trace(record))
        effective = row["correct"] or (row["support_regime"] == "INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT" and row["abstained"])
        scored.append((1.0 if effective else 0.0, -row["total_support_used"], policy))
    return max(scored)[2]


def output_row(pack, arm, policy, action_record, gate_features, gate_basis, used_truth=False):
    row = d59.output_from_action(pack, arm, action_record["action"], d59.rust_trace(action_record))
    stats = pack["trajectory_stats"]
    row["trajectory_variant"] = arm
    row["trajectory_basis"] = pack["trajectory_basis"]
    row["stage_sequence"] = stats["stage_sequence"]
    row["stage_counts"] = stats["stage_counts"]
    row["stage_transition_norm"] = stats["stage_transition_norm"]
    row["counter_early_fraction"] = stats["counter_early_fraction"]
    row["counter_late_fraction"] = stats["counter_late_fraction"]
    row["counter_stage_fraction"] = stats["counter_stage_fraction"]
    row["random_stage_fraction"] = stats["random_stage_fraction"]
    row["order_disagreement_norm"] = stats["order_disagreement_norm"]
    row["path_flip_count"] = stats["path_flip_count"]
    row["path_flip_norm"] = stats["path_flip_norm"]
    row["margin_delta"] = stats["margin_delta"]
    row["stage_labels_preserved"] = stats["stage_labels_preserved"]
    row["trajectory_readout_enabled"] = stats["trajectory_readout_enabled"]
    row["arbitrary_order_id_used"] = stats["arbitrary_order_id_used"]
    row["gate_selected_policy"] = policy
    row["gate_basis"] = gate_basis
    row["gate_features"] = gate_features
    row["fair_arm"] = arm in FAIR_ARMS
    row["control_arm"] = arm in CONTROL_ARMS
    row["reference_only_arm"] = arm in REFERENCE_ONLY_ARMS
    row["gate_used_truth_label"] = bool(used_truth)
    row["raw_support_sequence_manipulated_before_feature_generation"] = True
    row["diagnostic_bits_shuffled_after_feature_generation"] = False
    row["python_fallback_used"] = bool(row.get("python_fallback_used"))
    return row


def record_rows(rows, outputs, sample_counts, path):
    for row in rows:
        outputs.append(row)
        key = (row["arm"], row["support_regime"])
        if sample_counts[key] < ROW_SAMPLE_PER_ARM_REGIME_SPLIT:
            append_jsonl(path, row)
            sample_counts[key] += 1


def write_partial(out, split, outputs, completed, started):
    recent = outputs[-min(len(outputs), 4000):]
    partial = summarize_outputs(recent)
    write_json(
        out / f"partial_{split}_trajectory_eval.json",
        {
            "split": split,
            "completed_outputs": completed,
            "elapsed_sec": time.time() - started,
            "by_arm_core_recent": partial["by_arm_core"],
        },
    )
    append_progress(out, "trajectory_eval_progress", started, {"split": split, "completed_outputs": completed})


def evaluate_split(rows, bundle, policy_controllers, learned_gate, out, split, started, heartbeat_sec, repo_root, row_output_path):
    eval_items = []
    total_rows = len(rows)
    last = time.time()
    for idx, row in enumerate(rows):
        for arm in ARMS:
            pack = build_trajectory_pack(row, bundle, arm)
            eval_items.append({"arm": arm, "pack": pack})
        now = time.time()
        if now - last >= heartbeat_sec:
            last = now
            write_json(out / f"partial_{split}_pack_build.json", {"split": split, "raw_rows_done": idx + 1, "raw_rows_total": total_rows, "eval_packs": len(eval_items)})
            append_progress(out, "trajectory_pack_build_progress", started, {"split": split, "raw_rows_done": idx + 1, "raw_rows_total": total_rows})
    append_progress(out, "trajectory_pack_build_complete", started, {"split": split, "eval_packs": len(eval_items)})
    packs = [item["pack"] for item in eval_items]
    rust_actions, rust_report = d59.run_rust_multi_bridge(out, repo_root, policy_controllers, packs, split, "d64t_policy_eval", started)
    outputs = []
    sample_counts = Counter()
    completed = 0
    last = 0.0
    total = len(eval_items)
    for idx, item in enumerate(eval_items):
        arm = item["arm"]
        pack = item["pack"]
        used_truth = False
        if arm == "TRUTH_LEAK_SENTINEL_REFERENCE_ONLY":
            policy = truth_leak_policy(pack, rust_actions, idx)
            gate_features = d62.gate_features(pack)
            basis = "reference_only_best_policy_after_truth_scoring"
            used_truth = True
        else:
            policy, gate_features, basis = policy_from_pack(pack, learned_gate)
        action_record = rust_actions[policy][idx]
        row = output_row(pack, arm, policy, action_record, gate_features, basis, used_truth=used_truth)
        record_rows([row], outputs, sample_counts, row_output_path)
        completed += 1
        now = time.time()
        if now - last >= heartbeat_sec or completed >= total:
            last = now
            write_partial(out, split, outputs, completed, started)
    return outputs, rust_report


def summarize_outputs(outputs):
    by_arm = defaultdict(list)
    by_arm_core = defaultdict(list)
    by_arm_regime = defaultdict(list)
    by_seed_core = defaultdict(list)
    rust_counts = defaultdict(Counter)
    action_counts = defaultdict(Counter)
    traj_values = defaultdict(lambda: defaultdict(list))
    for row in outputs:
        arm = row["arm"]
        by_arm[arm].append(row)
        by_arm_regime[(arm, row["support_regime"])].append(row)
        by_seed_core[(arm, row["seed"])].append(row)
        action_counts[arm][row["selected_action"]] += 1
        rust_counts[arm]["rows"] += 1
        if row["rust_network_path_invoked"]:
            rust_counts[arm]["rust_rows"] += 1
        if row["python_fallback_used"]:
            rust_counts[arm]["python_fallback_rows"] += 1
        if row["support_regime"] in CORE_REGIMES:
            by_arm_core[arm].append(row)
        for name in [
            "stage_transition_norm",
            "counter_early_fraction",
            "counter_late_fraction",
            "counter_stage_fraction",
            "random_stage_fraction",
            "order_disagreement_norm",
            "path_flip_norm",
            "margin_delta",
        ]:
            traj_values[arm][name].append(float(row.get(name, 0.0)))
    return {
        "by_arm": {arm: d51.summarize(by_arm[arm]) for arm in ARMS if arm in by_arm},
        "by_arm_core": {arm: d51.summarize(by_arm_core[arm]) for arm in ARMS if arm in by_arm_core},
        "by_arm_and_regime": {
            arm: {regime: d51.summarize(by_arm_regime[(arm, regime)]) for regime in REGIMES if (arm, regime) in by_arm_regime}
            for arm in ARMS
        },
        "by_seed_core": {
            arm: {str(seed): d51.summarize(rows) for (a, seed), rows in by_seed_core.items() if a == arm and rows}
            for arm in ARMS
        },
        "action_distribution": {arm: dict(action_counts[arm]) for arm in ARMS},
        "rust_usage": {arm: dict(rust_counts[arm]) for arm in ARMS},
        "trajectory_stats": {
            arm: {name: d51.mean(values) for name, values in fields.items()}
            for arm, fields in traj_values.items()
        },
    }


def pairwise_report(outputs, arm_a, arm_b):
    by_key = {}
    for row in outputs:
        key = (row["split"], row["seed"], row["row_id"], row["support_regime"])
        by_key[(key, row["arm"])] = row
    pairs = []
    for key_arm, row_a in list(by_key.items()):
        key, arm = key_arm
        if arm != arm_a:
            continue
        row_b = by_key.get((key, arm_b))
        if row_b:
            pairs.append((row_a, row_b))
    return {
        "arm_a": arm_a,
        "arm_b": arm_b,
        "rows": len(pairs),
        "action_disagreement_rate": d51.mean([1.0 if a["selected_action"] != b["selected_action"] else 0.0 for a, b in pairs]),
        "correctness_disagreement_rate": d51.mean([1.0 if bool(a["correct"]) != bool(b["correct"]) else 0.0 for a, b in pairs]),
        "arm_a_accuracy": d51.mean([1.0 if a["correct"] else 0.0 for a, _b in pairs]),
        "arm_b_accuracy": d51.mean([1.0 if b["correct"] else 0.0 for _a, b in pairs]),
        "accuracy_delta_a_minus_b": d51.mean([1.0 if a["correct"] else 0.0 for a, _b in pairs]) - d51.mean([1.0 if b["correct"] else 0.0 for _a, b in pairs]),
    }


def make_decision(test_metrics, failed_jobs):
    core = test_metrics["by_arm_core"]
    if failed_jobs or "FULL_TRAJECTORY_READOUT" not in core:
        return {
            "decision": "d64t_instrumentation_incomplete",
            "verdict": "D64T_INSTRUMENTATION_INCOMPLETE",
            "next": "D64T_REPAIR_INSTRUMENTATION",
            "best_arm": None,
            "reason": "missing metrics or failed jobs",
        }
    full = core["FULL_TRAJECTORY_READOUT"]["exact_joint_accuracy"]
    set_inv = core["SET_INVARIANT_AGGREGATION"]["exact_joint_accuracy"]
    random_order = core["RANDOM_SUPPORT_ORDER_SHUFFLE"]["exact_joint_accuracy"]
    stage_preserve = core["STAGE_PRESERVING_SHUFFLE"]["exact_joint_accuracy"]
    stage_destroy = core["STAGE_DESTROYING_SHUFFLE"]["exact_joint_accuracy"]
    arbitrary = core["ARBITRARY_ORDER_ID_CONTROL"]["exact_joint_accuracy"]
    best_arm = max(core, key=lambda arm: (core[arm]["exact_joint_accuracy"], -core[arm]["average_total_support_used"]))
    if arbitrary >= full + 0.02 and arbitrary >= set_inv + 0.02:
        return {
            "decision": "arbitrary_order_artifact_detected",
            "verdict": "D64T_ARBITRARY_ORDER_ARTIFACT_DETECTED",
            "next": "D64T_REPAIR_ORDER_FEATURES",
            "best_arm": best_arm,
            "reason": "arbitrary row/order-id control outperformed fair trajectory arms",
        }
    if full >= 0.995 and stage_preserve >= full - 0.01 and stage_destroy <= full - 0.05 and random_order <= full - 0.03:
        return {
            "decision": "support_trajectory_signal_confirmed",
            "verdict": "D64T_SUPPORT_TRAJECTORY_SIGNAL_CONFIRMED",
            "next": "D65_RUST_SPARSE_IPF_SCORE_AGGREGATION_PROTOTYPE_WITH_TRAJECTORY_GUARD",
            "best_arm": best_arm,
            "reason": "stage-preserving trajectory survived while stage/order destroying controls dropped",
        }
    if set_inv >= full - 0.01 and random_order >= full - 0.02 and stage_destroy >= full - 0.02:
        return {
            "decision": "support_order_not_required_set_aggregation_sufficient",
            "verdict": "D64T_SET_AGGREGATION_SUFFICIENT",
            "next": "D64S_REPAIR_OR_REDEFINE_DIAGNOSTIC_CLAIM",
            "best_arm": best_arm,
            "reason": "raw support order changes did not create a clean dependency gap",
        }
    return {
        "decision": "temporal_interference_claim_not_confirmed",
        "verdict": "D64T_TEMPORAL_INTERFERENCE_CLAIM_NOT_CONFIRMED",
        "next": "D64T_REPAIR_OR_REDEFINE_TRAJECTORY_CLAIM",
        "best_arm": best_arm,
        "reason": "trajectory arms changed behavior but did not isolate a clean stage/order dependency",
    }


def make_reports(out, aggregate, decision):
    test_outputs = aggregate["_test_outputs_for_reports"]
    test_metrics = aggregate["test_metrics"]
    reports = {
        "d64s_upstream_manifest.json": aggregate["d64s_upstream_manifest"],
        "noncommutativity_report.json": {
            "a_then_b_vs_original": pairwise_report(test_outputs, "ORIGINAL_TRAJECTORY_REFERENCE", "A_THEN_B_VS_B_THEN_A"),
            "pos_then_neg_vs_original": pairwise_report(test_outputs, "ORIGINAL_TRAJECTORY_REFERENCE", "POS_THEN_NEG_VS_NEG_THEN_POS"),
            "note": "These compare actual support slot order before feature generation, not post-hoc diagnostic bit order.",
        },
        "trajectory_vs_set_report.json": {
            "full_vs_set": pairwise_report(test_outputs, "FULL_TRAJECTORY_READOUT", "SET_INVARIANT_AGGREGATION"),
            "original_vs_set": pairwise_report(test_outputs, "ORIGINAL_TRAJECTORY_REFERENCE", "SET_INVARIANT_AGGREGATION"),
            "by_arm_core": {
                arm: test_metrics["by_arm_core"].get(arm)
                for arm in ["ORIGINAL_TRAJECTORY_REFERENCE", "SET_INVARIANT_AGGREGATION", "FINAL_STATE_READOUT_ONLY", "FULL_TRAJECTORY_READOUT"]
            },
        },
        "stage_preserving_shuffle_report.json": {
            "stage_preserving_vs_full": pairwise_report(test_outputs, "FULL_TRAJECTORY_READOUT", "STAGE_PRESERVING_SHUFFLE"),
            "metrics": test_metrics["by_arm_core"].get("STAGE_PRESERVING_SHUFFLE"),
            "trajectory_stats": test_metrics["trajectory_stats"].get("STAGE_PRESERVING_SHUFFLE"),
        },
        "stage_destroying_shuffle_report.json": {
            "stage_destroying_vs_full": pairwise_report(test_outputs, "FULL_TRAJECTORY_READOUT", "STAGE_DESTROYING_SHUFFLE"),
            "metrics": test_metrics["by_arm_core"].get("STAGE_DESTROYING_SHUFFLE"),
            "trajectory_stats": test_metrics["trajectory_stats"].get("STAGE_DESTROYING_SHUFFLE"),
        },
        "order_artifact_report.json": {
            "arbitrary_vs_full": pairwise_report(test_outputs, "FULL_TRAJECTORY_READOUT", "ARBITRARY_ORDER_ID_CONTROL"),
            "random_order_vs_original": pairwise_report(test_outputs, "ORIGINAL_TRAJECTORY_REFERENCE", "RANDOM_SUPPORT_ORDER_SHUFFLE"),
            "arbitrary_order_id_control_is_reference_only": False,
            "arbitrary_order_id_control_is_control_arm": True,
        },
        "early_vs_late_counter_report.json": {
            "early_vs_late": pairwise_report(test_outputs, "EARLY_COUNTER_SUPPORT", "LATE_COUNTER_SUPPORT"),
            "early_metrics": test_metrics["by_arm_core"].get("EARLY_COUNTER_SUPPORT"),
            "late_metrics": test_metrics["by_arm_core"].get("LATE_COUNTER_SUPPORT"),
        },
        "open_vs_closed_loop_report.json": {
            "open_vs_closed": pairwise_report(test_outputs, "OPEN_LOOP_SUPPORT_SEQUENCE", "CLOSED_LOOP_FIELD_DEPENDENT_SUPPORT_SEQUENCE"),
            "open_metrics": test_metrics["by_arm_core"].get("OPEN_LOOP_SUPPORT_SEQUENCE"),
            "closed_metrics": test_metrics["by_arm_core"].get("CLOSED_LOOP_FIELD_DEPENDENT_SUPPORT_SEQUENCE"),
        },
        "final_state_vs_trajectory_readout_report.json": {
            "final_vs_full": pairwise_report(test_outputs, "FINAL_STATE_READOUT_ONLY", "FULL_TRAJECTORY_READOUT"),
            "final_metrics": test_metrics["by_arm_core"].get("FINAL_STATE_READOUT_ONLY"),
            "full_metrics": test_metrics["by_arm_core"].get("FULL_TRAJECTORY_READOUT"),
        },
        "truth_leak_audit_report.json": {
            "fair_arms": FAIR_ARMS,
            "reference_only_arms": REFERENCE_ONLY_ARMS,
            "fair_arms_using_truth_label": [],
            "truth_leak_sentinel_metrics": test_metrics["by_arm_core"].get("TRUTH_LEAK_SENTINEL_REFERENCE_ONLY"),
            "true_cells_and_operator_hidden_from_fair_arms": True,
        },
        "rust_invocation_report.json": aggregate["rust_invocation_report"],
    }
    for name, value in reports.items():
        write_json(out / name, value)
    return reports


def write_report(out, decision, metrics):
    rows = [
        "# D64T Temporal Support Trajectory Audit Result",
        "",
        "Decision:",
        "",
        "```text",
        f"decision = {decision['decision']}",
        f"verdict = {decision['verdict']}",
        f"next = {decision['next']}",
        f"best_arm = {decision['best_arm']}",
        "```",
        "",
        "Boundary:",
        "",
        "```text",
        BOUNDARY,
        "```",
        "",
        "| arm | exact core | corr | adv | external | abstain | support | order delta | path flips |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    by_core = metrics["by_arm_core"]
    by_regime = metrics["by_arm_and_regime"]
    traj = metrics["trajectory_stats"]
    ordered = sorted(ARMS, key=lambda arm: (-(by_core.get(arm, {}).get("exact_joint_accuracy", 0.0)), arm))
    for arm in ordered:
        core = by_core.get(arm, {})
        regimes = by_regime.get(arm, {})
        rows.append(
            f"| {arm} | {core.get('exact_joint_accuracy', 0.0):.6f} | "
            f"{regimes.get('CORRELATED_ECHO_SUPPORT', {}).get('accuracy', 0.0):.6f} | "
            f"{regimes.get('ADVERSARIAL_DISTRACTOR_SUPPORT', {}).get('accuracy', 0.0):.6f} | "
            f"{regimes.get('EXTERNAL_TEST_REQUIRED_SUPPORT', {}).get('accuracy', 0.0):.6f} | "
            f"{regimes.get('INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT', {}).get('abstain_rate', 0.0):.6f} | "
            f"{core.get('average_total_support_used', 0.0):.4f} | "
            f"{traj.get(arm, {}).get('order_disagreement_norm', 0.0):.4f} | "
            f"{traj.get(arm, {}).get('path_flip_norm', 0.0):.4f} |"
        )
    (out / "report.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", default="12201,12202,12203,12204,12205")
    parser.add_argument("--train-rows-per-seed", type=int, default=200)
    parser.add_argument("--test-rows-per-seed", type=int, default=300)
    parser.add_argument("--ood-rows-per-seed", type=int, default=300)
    parser.add_argument("--workers", default="auto")
    parser.add_argument("--cpu-target", default="50-75")
    parser.add_argument("--heartbeat-sec", type=float, default=20.0)
    parser.add_argument("--scale-mode", default="smoke")
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

    d64s_manifest = make_d64s_upstream_manifest(repo_root)
    write_json(out / "d64s_upstream_manifest.json", d64s_manifest)
    policy_controllers, learned_gate = load_policy_modules(repo_root)
    bundle = d55.d49_bundle()
    write_json(
        out / "dataset_manifest.json",
        {
            "task": "controlled_symbolic_joint_formula_discovery",
            "primary_space": PRIMARY_SPACE,
            "support_count": SUPPORT_COUNT,
            "regimes": REGIMES,
            "core_regimes": CORE_REGIMES,
            "arms": ARMS,
            "truth_hidden_from_controller_inputs": True,
            "raw_support_sequence_manipulated_before_feature_generation": True,
            "diagnostic_bits_shuffled_after_feature_generation": False,
            "rust_policy_modules": POLICY_MODULES,
        },
    )

    # Train rows are generated only to keep the same suite shape as nearby D-runs.
    # D64T is an audit/replay probe, not a new learned-model training run.
    _train_rows = d51.make_rows_with_progress(seeds, args.train_rows_per_seed, "train", bundle, out, started, args.heartbeat_sec)
    write_json(out / "partial_training_rows_generated.json", {"rows": len(_train_rows), "training_used": False})
    test_rows = d51.make_rows_with_progress(seeds, args.test_rows_per_seed, "test", bundle, out, started, args.heartbeat_sec)
    ood_rows = d51.make_rows_with_progress(seeds, args.ood_rows_per_seed, "ood", bundle, out, started, args.heartbeat_sec)

    row_test = out / "row_outputs_test.jsonl"
    row_ood = out / "row_outputs_ood.jsonl"
    for path in [row_test, row_ood]:
        if path.exists():
            path.unlink()

    test_outputs, test_rust_report = evaluate_split(test_rows, bundle, policy_controllers, learned_gate, out, "test", started, args.heartbeat_sec, repo_root, row_test)
    write_json(out / "partial_test_metrics.json", summarize_outputs(test_outputs))
    ood_outputs, ood_rust_report = evaluate_split(ood_rows, bundle, policy_controllers, learned_gate, out, "ood", started, args.heartbeat_sec, repo_root, row_ood)
    write_json(out / "partial_ood_metrics.json", summarize_outputs(ood_outputs))

    test_metrics = summarize_outputs(test_outputs)
    ood_metrics = summarize_outputs(ood_outputs)
    decision = make_decision(test_metrics, failed_jobs)
    fallback_rows = 0
    rust_rows = 0
    for metrics in [test_metrics, ood_metrics]:
        for counts in metrics["rust_usage"].values():
            fallback_rows += counts.get("python_fallback_rows", 0)
            rust_rows += counts.get("rust_rows", 0)
    aggregate = {
        "task": TASK,
        "failed_jobs": failed_jobs,
        "d64s_upstream_manifest": d64s_manifest,
        "test_metrics": test_metrics,
        "ood_metrics": ood_metrics,
        "rust_invocation_report": {"test": test_rust_report, "ood": ood_rust_report},
        "rust_path_invoked": rust_rows > 0,
        "fallback_rows": fallback_rows,
        "raw_support_sequence_manipulated_before_feature_generation": True,
        "diagnostic_bits_shuffled_after_feature_generation": False,
        "decision": decision,
        "boundary": BOUNDARY,
    }
    reports = make_reports(out, {**aggregate, "_test_outputs_for_reports": test_outputs}, decision)
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
            "rust_path_invoked": aggregate["rust_path_invoked"],
            "fallback_rows": fallback_rows,
            "failed_jobs": failed_jobs,
            "artifact_reports": sorted(reports.keys()),
            "boundary": BOUNDARY,
        },
    )
    write_report(out, decision, test_metrics)
    append_progress(out, "completed", started, {"decision": decision["decision"], "reports": sorted(reports.keys())})
    print(json.dumps(load_json_if_present(out / "summary.json"), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
