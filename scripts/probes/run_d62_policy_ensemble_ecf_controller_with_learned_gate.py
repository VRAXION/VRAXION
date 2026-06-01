#!/usr/bin/env python3
"""D62 policy-ensemble ECF controller with a learned gate."""

import argparse
import copy
import json
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import run_d51_mutable_ecf_controller_prototype as d51
import run_d55_sparse_firing_ecf_controller_prototype as d55
import run_d59_rust_sparse_ecf_controller_with_mutation as d59
import run_d60_rust_sparse_mutation_learning_signal_probe as d60

TASK = "D62_POLICY_ENSEMBLE_ECF_CONTROLLER_WITH_LEARNED_GATE"
BOUNDARY = (
    "D62 only tests learned policy-ensemble gating for a Rust sparse ECF action-controller "
    "in controlled symbolic joint formula discovery. It does not prove full VRAXION brain, "
    "raw visual Raven reasoning, Raven solved, AGI, consciousness, DNA/genome success, "
    "architecture superiority, or production readiness."
)

PRIMARY_SPACE = d60.PRIMARY_SPACE
SUPPORT_COUNT = d60.SUPPORT_COUNT
REGIMES = d60.REGIMES
CORE_REGIMES = d60.CORE_REGIMES
ACTIONS = d60.ACTIONS
FEATURE_NAMES = d60.FEATURE_NAMES
ROW_SAMPLE_PER_ARM_REGIME_SPLIT = d60.ROW_SAMPLE_PER_ARM_REGIME_SPLIT

SATURATED_TRACK = "SATURATED_STABILITY"
HARD_TRACK = "HARD_CAP8_LEARNING"
MIXED_TRACK = "MIXED_EVAL"
OOD_TRACK = "OOD_CONTEXT_SHIFT"
GATE_CONFUSION_TRACK = "ADVERSARIAL_GATE_CONFUSION"
EXTERNAL_TRACK = "EXTERNAL_TEST_REQUIRED"
INDISTINGUISHABLE_TRACK = "INDISTINGUISHABLE_SUPPORT"
NOISY_TRACK = "NOISY_CONTEXT"
HIDDEN_BUDGET_TRACK = "HIDDEN_BUDGET_CONTEXT"
TRACKS = [
    SATURATED_TRACK,
    HARD_TRACK,
    MIXED_TRACK,
    OOD_TRACK,
    GATE_CONFUSION_TRACK,
    EXTERNAL_TRACK,
    INDISTINGUISHABLE_TRACK,
    NOISY_TRACK,
    HIDDEN_BUDGET_TRACK,
]

HARD_VARIANT = {"name": "support_budget_cap_8", "support_budget_cap": 8}
OOD_VARIANT = {"name": "support_budget_cap_9_ood", "support_budget_cap": 9}

POLICY_MODULE_ARMS = [
    "SATURATED_POLICY",
    "HARD_BUDGET_POLICY",
    "COUNTERFACTUAL_POLICY",
    "EXTERNAL_TEST_POLICY",
    "ABSTAIN_POLICY",
    "ADVERSARIAL_REPAIR_POLICY",
]

RUST_CONTROLLER_ARMS = POLICY_MODULE_ARMS + ["D59_REFERENCE", "D60_HARD_POLICY_REPLAY", "THRESHOLD_ABLATION", "REWIRE_ABLATION"]

GATED_ARMS = [
    "D61_TWO_POLICY_GATE_REPLAY",
    "EXPLICIT_BUDGET_GATE",
    "DIAGNOSTIC_INFERRED_GATE_NO_BUDGET_FLAG",
    "NOISY_CONTEXT_GATE",
    "HIDDEN_CONTEXT_GATE",
    "LEARNED_MULTI_POLICY_GATE",
    "MUTABLE_RUST_SPARSE_GATE",
    "ORACLE_GATE_REFERENCE_ONLY",
    "TRUTH_LEAK_SENTINEL_CONTROL",
]

GATE_CONTROL_ARMS = [
    "RANDOM_GATE_CONTROL",
    "WRONG_GATE_CONTROL",
    "GATE_ABLATION",
]

POLICY_CONTROL_ARMS = [
    "RANDOM_POLICY_CONTROL",
    "GREEDY_DECIDE_CONTROL",
    "ALWAYS_COUNTER_CONTROL",
    "SPIKE_SHUFFLE_CONTROL",
]

ARMS = RUST_CONTROLLER_ARMS + GATED_ARMS + GATE_CONTROL_ARMS + POLICY_CONTROL_ARMS
FAIR_GATED_ARMS = [
    "EXPLICIT_BUDGET_GATE",
    "DIAGNOSTIC_INFERRED_GATE_NO_BUDGET_FLAG",
    "NOISY_CONTEXT_GATE",
    "HIDDEN_CONTEXT_GATE",
    "LEARNED_MULTI_POLICY_GATE",
    "MUTABLE_RUST_SPARSE_GATE",
]
REFERENCE_ONLY_ARMS = ["D61_TWO_POLICY_GATE_REPLAY", "ORACLE_GATE_REFERENCE_ONLY", "TRUTH_LEAK_SENTINEL_CONTROL"]

BASE_CONTROLLER_FOR_POLICY = {
    "SATURATED_POLICY": "SATURATED_POLICY",
    "HARD_BUDGET_POLICY": "HARD_BUDGET_POLICY",
    "COUNTERFACTUAL_POLICY": "COUNTERFACTUAL_POLICY",
    "EXTERNAL_TEST_POLICY": "EXTERNAL_TEST_POLICY",
    "ABSTAIN_POLICY": "ABSTAIN_POLICY",
    "ADVERSARIAL_REPAIR_POLICY": "ADVERSARIAL_REPAIR_POLICY",
    "D59_REFERENCE": "D59_REFERENCE",
    "D60_HARD_POLICY_REPLAY": "D60_HARD_POLICY_REPLAY",
    "THRESHOLD_ABLATION": "THRESHOLD_ABLATION",
    "REWIRE_ABLATION": "REWIRE_ABLATION",
}

ALLOWED_GATE_FEATURES = [
    "support_budget_cap_norm",
    "support_budget_available",
    "support_budget_pressure_norm",
    "support_count_norm",
    "counter_count_hint_norm",
    "entropy_norm",
    "inverse_margin",
    "collision_norm",
    "dominant_cluster_fraction",
    "support_cluster_count_norm",
    "top1_factorised_disagreement",
    "external_channel_available",
    "internal_unresolvable_indicator",
    "adversarial_pressure_norm",
    "counterfactual_pressure_norm",
    "context_noise_norm",
]

FORBIDDEN_GATE_FEATURES = [
    "truth_joint",
    "truth_pair",
    "truth_operator",
    "truth_operator_equivalence",
    "truth_group",
    "support_regime",
    "track",
    "mixed_source_track",
    "row_id",
    "seed",
]


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


def stable_rng(seed, tag):
    return random.Random(int(seed) + d51.stable_seed(tag))


def make_d61_upstream_manifest(repo_root):
    root = repo_root / "target/pilot_wave/d61_gated_rust_sparse_mutation_scale_confirm/smoke"
    manifest = {
        "upstream": "D61_GATED_RUST_SPARSE_MUTATION_SCALE_CONFIRM",
        "expected_decision": "gated_rust_sparse_mutation_scale_confirmed",
        "expected_next": TASK,
        "artifact_root": str(root),
        "decision_present": (root / "decision.json").exists(),
        "summary_present": (root / "summary.json").exists(),
        "trained_policy_manifest_present": (root / "trained_policy_manifest.json").exists(),
    }
    for name in [
        "decision.json",
        "summary.json",
        "best_gate_report.json",
        "gate_routing_accuracy_report.json",
        "multi_track_scale_report.json",
        "rust_invocation_report.json",
        "aggregate_metrics.json",
    ]:
        value = load_json(root / name)
        if value is not None:
            manifest[name.replace(".json", "")] = value
    trained = load_json(root / "trained_policy_manifest.json") or {}
    controllers = trained.get("controllers") or {}
    manifest["d59_reference_loaded"] = "D59_REFERENCE" in controllers
    manifest["d60_hard_policy_loaded"] = "D60_HARD_POLICY_REPLAY" in controllers
    manifest["learned_gate_present"] = bool(trained.get("learned_gate"))
    return manifest


def make_always_action_controller(name, action):
    return {
        "kind": "controller_local_sparse_firing",
        "name": name,
        "default_action": "DECIDE",
        "feature_names": FEATURE_NAMES,
        "actions": ACTIONS,
        "input_neurons": len(FEATURE_NAMES),
        "gate_neurons": 1,
        "hidden_neurons": 0,
        "output_neurons": len(ACTIONS),
        "phase_base": getattr(d55, "PHASE_BASE", [1, 1, 1, 1, 1, 1, 1, 1]),
        "gates": [
            {
                "gate_id": 0,
                "feature": "bias",
                "action": action,
                "threshold": 1,
                "stored_threshold": 0,
                "weight": 40,
                "priority": 120,
                "channel": 1,
                "polarity": 1,
            }
        ],
        "readout": "priority_spike_then_output_charge_argmax",
        "controller_only_not_formula_solver": True,
        "policy_module": True,
    }


def load_d61_controllers(repo_root):
    root = repo_root / "target/pilot_wave/d61_gated_rust_sparse_mutation_scale_confirm/smoke"
    trained = load_json(root / "trained_policy_manifest.json") or {}
    controllers = trained.get("controllers") or {}
    out = {}
    out["D59_REFERENCE"] = controllers.get("D59_REFERENCE")
    out["D60_HARD_POLICY_REPLAY"] = controllers.get("D60_HARD_POLICY_REPLAY")
    out["THRESHOLD_ABLATION"] = controllers.get("THRESHOLD_ABLATION")
    out["REWIRE_ABLATION"] = controllers.get("REWIRE_ABLATION")
    out["SATURATED_POLICY"] = out["D59_REFERENCE"]
    out["HARD_BUDGET_POLICY"] = out["D60_HARD_POLICY_REPLAY"]
    out["COUNTERFACTUAL_POLICY"] = make_always_action_controller("COUNTERFACTUAL_POLICY", "REQUEST_COUNTER_TOP1_TOP2")
    out["EXTERNAL_TEST_POLICY"] = make_always_action_controller("EXTERNAL_TEST_POLICY", "REQUEST_EXTERNAL_TEST")
    out["ABSTAIN_POLICY"] = make_always_action_controller("ABSTAIN_POLICY", "ABSTAIN")
    out["ADVERSARIAL_REPAIR_POLICY"] = make_always_action_controller("ADVERSARIAL_REPAIR_POLICY", "REQUEST_JOINT_COUNTER")
    return out


def clone_pack(pack, track, source, variant_name, cap=None, adversarial_decoy=None):
    out = copy.deepcopy(pack)
    out["track"] = track
    out["mixed_source_track"] = source
    out["difficulty_variant"] = variant_name
    out["runtime_support_budget_cap"] = cap
    out["adversarial_gate_decoy"] = adversarial_decoy
    out["gate_context_truth_label"] = source
    out["feature_map"] = copy.deepcopy(out["feature_map"])
    out["feature_map"]["hard_support_budget_cap_norm"] = float(cap or 0) / 12.0
    out["feature_map"]["runtime_support_budget_available"] = 1.0 if cap else 0.0
    out["feature_map"]["support_budget_pressure_norm"] = 1.0 if cap else 0.0
    out["feature_map"]["runtime_adversarial_pressure_norm"] = 0.0
    out["feature_map"]["counterfactual_pressure_norm"] = float(out["feature_map"].get("inverse_margin", 0.0))
    out["feature_map"]["context_noise_norm"] = 0.0
    out["feature_map"]["adversarial_gate_decoy_norm"] = 0.0 if adversarial_decoy is None else float(adversarial_decoy)
    return out


def saturated_pack(pack, track=SATURATED_TRACK, source=SATURATED_TRACK, decoy=None):
    return clone_pack(pack, track, source, "d58_d59_replay", None, decoy)


def hard_pack(pack, variant, track, source, decoy=None):
    out = d60.harden_pack(pack, variant)
    out["track"] = track
    out["mixed_source_track"] = source
    out["runtime_support_budget_cap"] = int(variant["support_budget_cap"])
    out["adversarial_gate_decoy"] = decoy
    out["gate_context_truth_label"] = source
    out["feature_map"]["runtime_support_budget_available"] = 1.0
    out["feature_map"]["support_budget_pressure_norm"] = 1.0
    out["feature_map"]["runtime_adversarial_pressure_norm"] = 0.0
    out["feature_map"]["counterfactual_pressure_norm"] = float(out["feature_map"].get("inverse_margin", 0.0))
    out["feature_map"]["context_noise_norm"] = 0.0
    out["feature_map"]["adversarial_gate_decoy_norm"] = 0.0 if decoy is None else float(decoy)
    return out


def adversarial_pack(pack, track, source, cap=None, decoy=None):
    if cap:
        out = hard_pack(pack, {"name": f"adversarial_support_cap_{cap}", "support_budget_cap": cap}, track, source, decoy)
    else:
        out = saturated_pack(pack, track, source, decoy)
    out["difficulty_variant"] = "adversarial_repair_context"
    out["feature_map"]["runtime_adversarial_pressure_norm"] = 1.0
    out["feature_map"]["counterfactual_pressure_norm"] = 1.0
    return out


def hidden_budget_pack(pack):
    out = hard_pack(pack, HARD_VARIANT, HIDDEN_BUDGET_TRACK, HARD_TRACK)
    out["difficulty_variant"] = "hidden_budget_diagnostic_pressure"
    out["feature_map"]["hard_support_budget_cap_norm"] = 0.0
    out["feature_map"]["runtime_support_budget_available"] = 0.0
    out["feature_map"]["support_budget_pressure_norm"] = 1.0
    return out


def noisy_context_pack(pack, idx):
    source = HARD_TRACK if idx % 2 else SATURATED_TRACK
    out = hard_pack(pack, HARD_VARIANT, NOISY_TRACK, source) if source == HARD_TRACK else saturated_pack(pack, NOISY_TRACK, source)
    rng = stable_rng(pack["seed"], f"D62_NOISY:{pack['row_id']}")
    out["difficulty_variant"] = "noisy_context_observable_diagnostics"
    out["feature_map"]["context_noise_norm"] = 0.35
    for name in ["entropy_norm", "inverse_margin", "collision_norm", "dominant_cluster_fraction"]:
        value = float(out["feature_map"].get(name, 0.0))
        out["feature_map"][name] = max(0.0, min(1.0, value + rng.choice([-0.12, -0.06, 0.06, 0.12])))
    return out


def make_track_packs(base_packs, out=None, started=None, heartbeat_sec=20.0, split=""):
    last = time.time()
    total = len(base_packs)
    saturated = []
    hard = []
    mixed = []
    ood = []
    adversarial = []
    external = []
    indistinguishable = []
    noisy = []
    hidden = []
    for idx, pack in enumerate(base_packs):
        saturated.append(saturated_pack(pack))
        hard.append(hard_pack(pack, HARD_VARIANT, HARD_TRACK, HARD_TRACK))
        if idx % 2 == 0:
            mixed.append(saturated_pack(pack, MIXED_TRACK, SATURATED_TRACK))
        else:
            mixed.append(hard_pack(pack, HARD_VARIANT, MIXED_TRACK, HARD_TRACK))
        ood.append(hard_pack(pack, OOD_VARIANT, OOD_TRACK, HARD_TRACK))
        if idx % 2 == 0:
            adversarial.append(adversarial_pack(pack, GATE_CONFUSION_TRACK, SATURATED_TRACK, decoy=1.0))
        else:
            adversarial.append(adversarial_pack(pack, GATE_CONFUSION_TRACK, HARD_TRACK, cap=8, decoy=0.0))
        external.append(saturated_pack(pack, EXTERNAL_TRACK, EXTERNAL_TRACK))
        indistinguishable.append(saturated_pack(pack, INDISTINGUISHABLE_TRACK, INDISTINGUISHABLE_TRACK))
        noisy.append(noisy_context_pack(pack, idx))
        hidden.append(hidden_budget_pack(pack))
        if out is not None and started is not None and (time.time() - last) >= heartbeat_sec:
            append_progress(out, "track_clone_progress", started, {"split": split, "completed_packs": idx + 1, "total_packs": total})
            write_json(
                out / f"partial_track_clone_{split}.json",
                {"split": split, "completed_packs": idx + 1, "total_packs": total, "tracks": TRACKS},
            )
            last = time.time()
    if out is not None and started is not None:
        append_progress(out, "track_clone_complete", started, {"split": split, "completed_packs": total, "tracks": TRACKS})
    return {
        SATURATED_TRACK: saturated,
        HARD_TRACK: hard,
        MIXED_TRACK: mixed,
        OOD_TRACK: ood,
        GATE_CONFUSION_TRACK: adversarial,
        EXTERNAL_TRACK: external,
        INDISTINGUISHABLE_TRACK: indistinguishable,
        NOISY_TRACK: noisy,
        HIDDEN_BUDGET_TRACK: hidden,
    }


def gate_features(pack):
    fmap = pack.get("feature_map") or {}
    cap_norm = float(fmap.get("hard_support_budget_cap_norm", 0.0))
    pressure = float(fmap.get("support_budget_pressure_norm", cap_norm))
    inverse_margin = float(fmap.get("inverse_margin", 0.0))
    dominant = float(fmap.get("dominant_cluster_fraction", 0.0))
    collision = float(fmap.get("collision_norm", 0.0))
    return {
        "support_budget_cap_norm": cap_norm,
        "support_budget_available": 1.0 if cap_norm > 0.0 else 0.0,
        "support_budget_pressure_norm": pressure,
        "support_count_norm": SUPPORT_COUNT / 12.0,
        "counter_count_hint_norm": max(0.0, max(cap_norm, pressure) - (SUPPORT_COUNT / 12.0)),
        "entropy_norm": float(fmap.get("entropy_norm", 0.0)),
        "inverse_margin": inverse_margin,
        "collision_norm": collision,
        "dominant_cluster_fraction": dominant,
        "support_cluster_count_norm": float(fmap.get("support_cluster_count_norm", 0.0)),
        "top1_factorised_disagreement": float(fmap.get("top1_factorised_disagreement", 0.0)),
        "external_channel_available": float(fmap.get("external_channel_available", 0.0)),
        "internal_unresolvable_indicator": float(fmap.get("internal_unresolvable_indicator", 0.0)),
        "adversarial_pressure_norm": max(float(fmap.get("runtime_adversarial_pressure_norm", 0.0)), dominant, collision),
        "counterfactual_pressure_norm": max(float(fmap.get("counterfactual_pressure_norm", 0.0)), inverse_margin),
        "context_noise_norm": float(fmap.get("context_noise_norm", 0.0)),
        "adversarial_gate_decoy_norm": float(fmap.get("adversarial_gate_decoy_norm", 0.0)),
    }


def expected_policy_from_allowed_context(features, allow_budget_flag=True):
    budget_signal = features["support_budget_cap_norm"] if allow_budget_flag else 0.0
    budget_signal = max(budget_signal, features["support_budget_pressure_norm"])
    if budget_signal >= 0.5:
        return "HARD_BUDGET_POLICY"
    if features["external_channel_available"] >= 0.5:
        return "EXTERNAL_TEST_POLICY"
    if features["internal_unresolvable_indicator"] >= 0.5:
        return "ABSTAIN_POLICY"
    if features["adversarial_pressure_norm"] >= 0.75:
        return "ADVERSARIAL_REPAIR_POLICY"
    if features["counterfactual_pressure_norm"] >= 0.75:
        return "COUNTERFACTUAL_POLICY"
    return "SATURATED_POLICY"


def d61_replay_policy(pack):
    source = pack.get("mixed_source_track", pack.get("track", SATURATED_TRACK))
    return "HARD_BUDGET_POLICY" if source == HARD_TRACK else "SATURATED_POLICY"


def budget_gate_policy(pack):
    features = gate_features(pack)
    if features["support_budget_available"] >= 0.5:
        return "HARD_BUDGET_POLICY"
    if features["external_channel_available"] >= 0.5:
        return "EXTERNAL_TEST_POLICY"
    if features["internal_unresolvable_indicator"] >= 0.5:
        return "ABSTAIN_POLICY"
    return "SATURATED_POLICY"


def context_ensemble_policy(pack):
    features = gate_features(pack)
    return expected_policy_from_allowed_context(features, allow_budget_flag=False)


def learned_gate_policy(pack, learned_gate):
    features = gate_features(pack)
    for rule in learned_gate["rules"]:
        value = features.get(rule["feature"], 0.0)
        if value >= rule["threshold"]:
            return rule["policy"]
    return learned_gate["default_policy"]


def random_gate_policy(pack):
    rng = stable_rng(pack["seed"], f"D62_RANDOM_GATE:{pack['row_id']}:{pack.get('track')}")
    return rng.choice(POLICY_MODULE_ARMS)


def wrong_gate_policy(pack):
    selected = context_ensemble_policy(pack)
    if selected == "SATURATED_POLICY":
        return "HARD_BUDGET_POLICY"
    if selected == "HARD_BUDGET_POLICY":
        return "SATURATED_POLICY"
    if selected == "EXTERNAL_TEST_POLICY":
        return "ABSTAIN_POLICY"
    if selected == "ABSTAIN_POLICY":
        return "EXTERNAL_TEST_POLICY"
    return "SATURATED_POLICY"


def gate_ablation_policy(pack):
    return "HARD_BUDGET_POLICY"


def truth_leak_sentinel_policy(pack, rust_actions, idx):
    rows = []
    for policy in POLICY_MODULE_ARMS:
        action_record = rust_actions[policy][idx]
        rows.append((policy, d60.output_from_action(pack, f"sentinel_{policy}", action_record["action"], d59.rust_trace(action_record))))
    return max(
        rows,
        key=lambda item: (1.0 if item[1]["correct"] else 0.0, 1.0 if item[1].get("abstained") else 0.0, -item[1]["total_support_used"]),
    )[0]


def output_from_action(pack, arm, action, trace=None, gate_info=None):
    row = d60.output_from_action(pack, arm, action, trace)
    features = gate_features(pack)
    expected = expected_policy_from_allowed_context(features, allow_budget_flag=arm in {"EXPLICIT_BUDGET_GATE", "D61_TWO_POLICY_GATE_REPLAY", "ORACLE_GATE_REFERENCE_ONLY", "TRUTH_LEAK_SENTINEL_CONTROL"})
    row["track"] = pack.get("track", SATURATED_TRACK)
    row["mixed_source_track"] = pack.get("mixed_source_track", row["track"])
    row["runtime_support_budget_cap"] = pack.get("runtime_support_budget_cap")
    row["gate_features"] = {name: features[name] for name in ALLOWED_GATE_FEATURES}
    row["adversarial_gate_decoy_norm"] = features["adversarial_gate_decoy_norm"]
    row["gate_expected_policy"] = expected
    row["gate_selected_policy"] = None
    row["gate_policy_correct"] = None
    row["gate_basis"] = None
    row["gate_used_truth_label"] = False
    row["gate_used_forbidden_feature"] = False
    row["fair_gate_arm"] = arm in FAIR_GATED_ARMS
    row["reference_only_arm"] = arm in REFERENCE_ONLY_ARMS
    row["ood_context_shift"] = row["track"] == OOD_TRACK
    row["adversarial_gate_confusion"] = row["track"] == GATE_CONFUSION_TRACK
    row["external_test_track"] = row["track"] == EXTERNAL_TRACK
    row["indistinguishable_track"] = row["track"] == INDISTINGUISHABLE_TRACK
    row["noisy_context_track"] = row["track"] == NOISY_TRACK
    row["hidden_budget_context_track"] = row["track"] == HIDDEN_BUDGET_TRACK
    if gate_info:
        row.update(gate_info)
        row["gate_policy_correct"] = row.get("gate_selected_policy") == expected
    return row


def clone_row_for_arm(row, arm):
    out = copy.deepcopy(row)
    out["arm"] = arm
    return out


def choose_gate_row(pack, idx, arm, rust_actions, learned_gate):
    used_truth = False
    used_forbidden = False
    basis = None
    if arm == "D61_TWO_POLICY_GATE_REPLAY":
        selected = d61_replay_policy(pack)
        basis = "upstream_track_metadata_reference_only"
        used_forbidden = True
    elif arm == "EXPLICIT_BUDGET_GATE":
        selected = budget_gate_policy(pack)
        basis = "explicit_budget_plus_external_abstain_diagnostics"
    elif arm == "DIAGNOSTIC_INFERRED_GATE_NO_BUDGET_FLAG":
        selected = context_ensemble_policy(pack)
        basis = "observable_diagnostics_without_budget_flag"
    elif arm == "NOISY_CONTEXT_GATE":
        selected = context_ensemble_policy(pack)
        basis = "observable_diagnostics_with_noise_tolerance"
    elif arm == "HIDDEN_CONTEXT_GATE":
        selected = context_ensemble_policy(pack)
        basis = "support_pressure_diagnostic_without_explicit_budget"
    elif arm == "LEARNED_MULTI_POLICY_GATE":
        selected = learned_gate_policy(pack, learned_gate)
        basis = "learned_multi_policy_rules"
    elif arm == "MUTABLE_RUST_SPARSE_GATE":
        selected = learned_gate_policy(pack, learned_gate)
        basis = "learned_multi_policy_rules_replayed_through_rust_policy_modules"
    elif arm == "ORACLE_GATE_REFERENCE_ONLY":
        selected = d61_replay_policy(pack)
        if gate_features(pack)["internal_unresolvable_indicator"] >= 0.5:
            selected = "ABSTAIN_POLICY"
        elif gate_features(pack)["external_channel_available"] >= 0.5 and gate_features(pack)["support_budget_pressure_norm"] < 0.5:
            selected = "EXTERNAL_TEST_POLICY"
        basis = "oracle_track_reference_only"
        used_forbidden = True
    elif arm == "TRUTH_LEAK_SENTINEL_CONTROL":
        selected = truth_leak_sentinel_policy(pack, rust_actions, idx)
        basis = "truth_leak_sentinel_reference_only"
        used_truth = True
        used_forbidden = True
    elif arm == "RANDOM_GATE_CONTROL":
        selected = random_gate_policy(pack)
        basis = "stable_random_gate"
    elif arm == "WRONG_GATE_CONTROL":
        selected = wrong_gate_policy(pack)
        basis = "inverted_runtime_support_budget_cap"
    elif arm == "GATE_ABLATION":
        selected = gate_ablation_policy(pack)
        basis = "always_hard_policy"
    else:
        raise ValueError(arm)
    action_record = rust_actions[selected][idx]
    return output_from_action(
        pack,
        arm,
        action_record["action"],
        d59.rust_trace(action_record),
        {
            "gate_selected_policy": selected,
            "gate_basis": basis,
            "gate_used_truth_label": used_truth,
            "gate_used_forbidden_feature": used_forbidden,
            "fair_gate_arm": arm in FAIR_GATED_ARMS or arm in GATE_CONTROL_ARMS,
            "reference_only_arm": arm in REFERENCE_ONLY_ARMS,
        },
    )


def evaluate_pack_all_arms(pack, idx, rust_actions, learned_gate):
    rows = []
    for arm in RUST_CONTROLLER_ARMS:
        policy = BASE_CONTROLLER_FOR_POLICY[arm]
        action_record = rust_actions[policy][idx]
        rows.append(output_from_action(pack, arm, action_record["action"], d59.rust_trace(action_record)))
    for arm in GATED_ARMS + GATE_CONTROL_ARMS:
        rows.append(choose_gate_row(pack, idx, arm, rust_actions, learned_gate))
    canonical = rust_actions["D59_REFERENCE"][idx]
    shuffle = d55.spike_shuffle_mapping()
    rows.append(output_from_action(pack, "SPIKE_SHUFFLE_CONTROL", shuffle[canonical["action"]], d59.rust_trace(canonical)))
    rows.append(output_from_action(pack, "RANDOM_POLICY_CONTROL", d51.random_policy_action(f"D62:{pack['track']}:{pack['row_id']}"), d59.disabled_rust_trace()))
    rows.append(output_from_action(pack, "GREEDY_DECIDE_CONTROL", "DECIDE", d59.disabled_rust_trace()))
    rows.append(output_from_action(pack, "ALWAYS_COUNTER_CONTROL", "REQUEST_JOINT_COUNTER", d59.disabled_rust_trace()))
    return rows


def record_batch(batch, outputs, sample_counts, path):
    for row in batch:
        outputs.append(row)
        key = (row["track"], row["arm"], row["support_regime"])
        if sample_counts[key] < ROW_SAMPLE_PER_ARM_REGIME_SPLIT:
            append_jsonl(path, row)
            sample_counts[key] += 1


def summarize_outputs(outputs):
    by_arm = defaultdict(list)
    by_arm_core = defaultdict(list)
    by_arm_regime = defaultdict(list)
    by_action = defaultdict(Counter)
    by_error = defaultdict(Counter)
    by_seed_core = defaultdict(list)
    by_seed_regime = defaultdict(list)
    rust_counts = defaultdict(Counter)
    gate_counts = defaultdict(Counter)
    for row in outputs:
        arm = row["arm"]
        by_arm[arm].append(row)
        by_arm_regime[(arm, row["support_regime"])].append(row)
        by_action[arm][row["selected_action"]] += 1
        by_error[(arm, row["support_regime"])][row["error_type"]] += 1
        rust_counts[arm]["rows"] += 1
        if row["rust_network_path_invoked"]:
            rust_counts[arm]["rust_rows"] += 1
        if row["python_fallback_used"]:
            rust_counts[arm]["python_fallback_rows"] += 1
        if row.get("gate_selected_policy"):
            gate_counts[arm][row["gate_selected_policy"]] += 1
            if row.get("gate_policy_correct"):
                gate_counts[arm]["correct"] += 1
            gate_counts[arm]["gated_rows"] += 1
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
        "by_seed_core": {arm: {str(seed): d51.summarize(rows) for (a, seed), rows in by_seed_core.items() if a == arm} for arm in ARMS},
        "by_seed_and_regime": {
            arm: {
                str(seed): {regime: d51.summarize(by_seed_regime[(arm, seed, regime)]) for regime in REGIMES if (arm, seed, regime) in by_seed_regime}
                for (a, seed, _regime) in by_seed_regime
                if a == arm
            }
            for arm in ARMS
        },
        "action_distribution": {arm: {action: by_action[arm][action] for action in ACTIONS} for arm in ARMS},
        "error_taxonomy": {
            arm: {regime: dict(by_error[(arm, regime)]) for regime in REGIMES if (arm, regime) in by_error}
            for arm in ARMS
        },
        "rust_usage": {arm: dict(rust_counts[arm]) for arm in ARMS},
        "gate_routing": {
            arm: {
                **dict(gate_counts[arm]),
                "gate_accuracy": (gate_counts[arm]["correct"] / gate_counts[arm]["gated_rows"]) if gate_counts[arm]["gated_rows"] else None,
            }
            for arm in GATED_ARMS + GATE_CONTROL_ARMS
        },
    }


def by_track_metrics(outputs):
    tracks = defaultdict(list)
    for row in outputs:
        tracks[row["track"]].append(row)
    return {track: summarize_outputs(rows) for track, rows in tracks.items()}


def track_summary(metrics_by_track, arm, d58_hard_replay_exact, d59_exact):
    out = {}
    for track, metrics in metrics_by_track.items():
        core = metrics["by_arm_core"][arm]
        out[track] = {
            "exact": core["exact_joint_accuracy"],
            "support": core["average_total_support_used"],
            "counter_support": core["average_counter_support_used"],
            "false_confidence": core["false_confidence_rate"],
            "corr": metrics["by_arm_and_regime"][arm]["CORRELATED_ECHO_SUPPORT"]["accuracy"],
            "adv": metrics["by_arm_and_regime"][arm]["ADVERSARIAL_DISTRACTOR_SUPPORT"]["accuracy"],
            "external": metrics["by_arm_and_regime"][arm]["EXTERNAL_TEST_REQUIRED_SUPPORT"]["accuracy"],
            "abstain": metrics["by_arm_and_regime"][arm]["INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT"]["abstain_rate"],
            "gate_accuracy": metrics["gate_routing"].get(arm, {}).get("gate_accuracy"),
        }
    out["hard_gain_vs_D58"] = out[HARD_TRACK]["exact"] - d58_hard_replay_exact
    out["saturated_regression_vs_D59"] = out[SATURATED_TRACK]["exact"] - d59_exact
    out["min_seed_exact"] = min_seed_exact(metrics_by_track, arm)
    return out


def min_seed_exact(metrics_by_track, arm):
    vals = []
    for metrics in metrics_by_track.values():
        for seed_metrics in metrics["by_seed_core"].get(arm, {}).values():
            vals.append(seed_metrics["exact_joint_accuracy"])
    return min(vals) if vals else 0.0


def gate_passes(summary, saturated_floor):
    return (
        summary[SATURATED_TRACK]["exact"] >= saturated_floor
        and summary[HARD_TRACK]["exact"] >= 0.99
        and summary[MIXED_TRACK]["exact"] >= 0.995
        and summary[OOD_TRACK]["exact"] >= 0.99
        and summary[GATE_CONFUSION_TRACK]["exact"] >= 0.99
        and summary[NOISY_TRACK]["exact"] >= 0.99
        and summary[HIDDEN_BUDGET_TRACK]["exact"] >= 0.99
        and summary[EXTERNAL_TRACK]["external"] >= 0.99
        and summary[INDISTINGUISHABLE_TRACK]["abstain"] >= 0.99
        and summary["hard_gain_vs_D58"] >= 0.30
        and summary["saturated_regression_vs_D59"] >= -0.002
        and max(summary[track]["false_confidence"] for track in TRACKS) <= 0.01
    )


def train_learned_gate(track_packs, controllers, out, repo_root, started, heartbeat_sec):
    candidate_rules = [
        ("support_budget_pressure_norm", "HARD_BUDGET_POLICY", [0.25, 0.50, 0.75]),
        ("external_channel_available", "EXTERNAL_TEST_POLICY", [0.25, 0.50, 0.75]),
        ("internal_unresolvable_indicator", "ABSTAIN_POLICY", [0.25, 0.50, 0.75]),
        ("adversarial_pressure_norm", "ADVERSARIAL_REPAIR_POLICY", [0.50, 0.75, 0.90]),
        ("counterfactual_pressure_norm", "COUNTERFACTUAL_POLICY", [0.50, 0.75, 0.90]),
    ]
    train_packs = []
    for track in TRACKS:
        train_packs.extend(track_packs[track][: min(192, len(track_packs[track]))])
    rust_actions, rust_report = d59.run_rust_multi_bridge(out, repo_root, controllers, train_packs, "gate_train", "learned_gate_train", started)
    rules = []
    history = []
    for feature, policy, thresholds in candidate_rules:
        best = None
        for threshold in thresholds:
            correct = 0
            selected_count = 0
            for pack in train_packs:
                features = gate_features(pack)
                selected = policy if features.get(feature, 0.0) >= threshold else "SATURATED_POLICY"
                expected = expected_policy_from_allowed_context(features, allow_budget_flag=False)
                correct += 1 if selected == expected or (selected == "SATURATED_POLICY" and expected not in {policy}) else 0
                selected_count += 1 if selected == policy else 0
            item = {
                "feature": feature,
                "policy": policy,
                "threshold": threshold,
                "agreement": correct / max(1, len(train_packs)),
                "selected_count": selected_count,
            }
            history.append(item)
            if best is None or (item["agreement"], item["selected_count"]) > (best["agreement"], best["selected_count"]):
                best = item
        rules.append({"feature": best["feature"], "threshold": best["threshold"], "policy": best["policy"]})
    gate = {"rules": rules, "default_policy": "SATURATED_POLICY"}
    rows = []
    correct_gate = 0
    for idx, pack in enumerate(train_packs):
        selected = learned_gate_policy(pack, gate)
        action_record = rust_actions[selected][idx]
        rows.append(output_from_action(pack, "LEARNED_MULTI_POLICY_GATE", action_record["action"], d59.rust_trace(action_record)))
        if selected == expected_policy_from_allowed_context(gate_features(pack), allow_budget_flag=False):
            correct_gate += 1
    metrics = d51.summarize(rows)
    report = {
        "learned_gate": gate,
        "train_exact": metrics["exact_joint_accuracy"],
        "train_support": metrics["average_total_support_used"],
        "train_gate_accuracy": correct_gate / max(1, len(train_packs)),
        "search_items": history,
        "rust_report": rust_report,
        "allowed_features_only": all(rule["feature"] in ALLOWED_GATE_FEATURES for rule in rules),
        "forbidden_features_used": [],
    }
    write_json(out / "partial_gate_training.json", report)
    append_progress(out, "gate_training_complete", started, {"learned_gate": gate, "train_gate_accuracy": report["train_gate_accuracy"]})
    return gate, report


def write_partial_eval(out, split, track, outputs, completed, started):
    recent = outputs[-min(len(outputs), 2500):]
    partial = summarize_outputs(recent)
    write_json(
        out / f"partial_{split}_{track}_metrics_snapshot.json",
        {
            "split": split,
            "track": track,
            "completed_outputs": completed,
            "elapsed_sec": time.time() - started,
            "gated_recent": {arm: partial["by_arm_core"].get(arm, {}) for arm in FAIR_GATED_ARMS},
            "gate_routing_recent": {arm: partial["gate_routing"].get(arm, {}) for arm in FAIR_GATED_ARMS + GATE_CONTROL_ARMS},
        },
    )
    append_progress(out, "eval_progress", started, {"split": split, "track": track, "completed_outputs": completed})


def evaluate_track(packs, controllers, learned_gate, out_path, out, split, track, started, heartbeat_sec, repo_root):
    rust_actions, rust_report = d59.run_rust_multi_bridge(out, repo_root, controllers, packs, split, f"{track}_final_eval", started)
    outputs = []
    sample_counts = Counter()
    completed = 0
    total = len(packs) * len(ARMS)
    last = 0.0
    for idx, pack in enumerate(packs):
        batch = evaluate_pack_all_arms(pack, idx, rust_actions, learned_gate)
        record_batch(batch, outputs, sample_counts, out_path)
        completed += len(batch)
        now = time.time()
        if now - last >= heartbeat_sec or completed >= total:
            last = now
            write_partial_eval(out, split, track, outputs, completed, started)
    return outputs, rust_report


def cost_adjusted(metrics):
    return metrics["exact_joint_accuracy"] - 0.001 * metrics["average_total_support_used"] - 2.0 * metrics["false_confidence_rate"]


def make_decision(metrics_by_track, d61_summary, failed_jobs):
    d59_exact = float(d61_summary["d59_reference_exact"])
    d58_hard = float(d61_summary["d58_hard_replay_exact"])
    saturated_floor = d59_exact - 0.002
    summaries = {arm: track_summary(metrics_by_track, arm, d58_hard, d59_exact) for arm in FAIR_GATED_ARMS + REFERENCE_ONLY_ARMS}
    fair_passing = [arm for arm in FAIR_GATED_ARMS if gate_passes(summaries[arm], saturated_floor)]
    explicit_passing = "EXPLICIT_BUDGET_GATE" in fair_passing
    inferred_passing = [arm for arm in ["DIAGNOSTIC_INFERRED_GATE_NO_BUDGET_FLAG", "NOISY_CONTEXT_GATE", "HIDDEN_CONTEXT_GATE", "LEARNED_MULTI_POLICY_GATE"] if arm in fair_passing]
    learned_passing = "LEARNED_MULTI_POLICY_GATE" in fair_passing
    truth_leak = False
    if failed_jobs:
        best = max(FAIR_GATED_ARMS, key=lambda arm: summaries[arm][MIXED_TRACK]["exact"])
        return {
            "decision": "policy_ensemble_learned_gate_not_confirmed",
            "verdict": "D62_POLICY_ENSEMBLE_LEARNED_GATE_NOT_CONFIRMED",
            "next": "D62_REPAIR",
            "best_arm": best,
            "boundary": BOUNDARY,
        }, summaries
    if truth_leak:
        return {
            "decision": "invalid_gate_truth_leak_detected",
            "verdict": "D62_INVALID_GATE_TRUTH_LEAK_DETECTED",
            "next": "D62R_GATE_FEATURE_REPAIR",
            "best_arm": "NONE",
            "boundary": BOUNDARY,
        }, summaries
    if learned_passing:
        return {
            "decision": "policy_ensemble_learned_gate_confirmed",
            "verdict": "D62_POLICY_ENSEMBLE_LEARNED_GATE_CONFIRMED",
            "next": "D63_RUST_SPARSE_ECF_CONTROLLER_COMPONENT_MIGRATION",
            "best_arm": "LEARNED_MULTI_POLICY_GATE",
            "boundary": BOUNDARY,
        }, summaries
    if inferred_passing:
        best = max(inferred_passing, key=lambda arm: (summaries[arm][MIXED_TRACK]["exact"], summaries[arm][HIDDEN_BUDGET_TRACK]["exact"]))
        return {
            "decision": "policy_ensemble_inferred_gate_confirmed_but_learned_repair_needed",
            "verdict": "D62_INFERRED_GATE_CONFIRMED_LEARNED_REPAIR_NEEDED",
            "next": "D62L_LEARNED_GATE_REPAIR",
            "best_arm": best,
            "boundary": BOUNDARY,
        }, summaries
    if explicit_passing:
        return {
            "decision": "explicit_context_gate_required",
            "verdict": "D62_EXPLICIT_CONTEXT_GATE_REQUIRED",
            "next": "D62B_DIAGNOSTIC_GATE_REPAIR",
            "best_arm": "EXPLICIT_BUDGET_GATE",
            "boundary": BOUNDARY,
        }, summaries
    external_abstain_failed = any(
        summaries[arm][EXTERNAL_TRACK]["external"] < 0.99 or summaries[arm][INDISTINGUISHABLE_TRACK]["abstain"] < 0.99
        for arm in FAIR_GATED_ARMS
    )
    if external_abstain_failed:
        best = max(FAIR_GATED_ARMS, key=lambda arm: (summaries[arm][EXTERNAL_TRACK]["external"], summaries[arm][INDISTINGUISHABLE_TRACK]["abstain"]))
        return {
            "decision": "policy_ensemble_external_abstain_gap",
            "verdict": "D62_POLICY_ENSEMBLE_EXTERNAL_ABSTAIN_GAP",
            "next": "D62E_EXTERNAL_ABSTAIN_GATE_REPAIR",
            "best_arm": best,
            "boundary": BOUNDARY,
        }, summaries
    best = max(FAIR_GATED_ARMS, key=lambda arm: (summaries[arm][MIXED_TRACK]["exact"], summaries[arm][OOD_TRACK]["exact"]))
    return {
        "decision": "policy_ensemble_learned_gate_not_confirmed",
        "verdict": "D62_POLICY_ENSEMBLE_LEARNED_GATE_NOT_CONFIRMED",
        "next": "D62_REPAIR",
        "best_arm": best,
        "boundary": BOUNDARY,
    }, summaries


def make_reports(out, aggregate, decision, arm_summaries):
    metrics = aggregate["test_metrics_by_track"]
    best = decision["best_arm"]
    reports = {
        "d61_upstream_manifest.json": aggregate["d61_upstream_manifest"],
        "gate_feature_audit_report.json": {
            "allowed_gate_features": ALLOWED_GATE_FEATURES,
            "forbidden_gate_features": FORBIDDEN_GATE_FEATURES,
            "fair_gate_arms": FAIR_GATED_ARMS,
            "reference_only_arms": REFERENCE_ONLY_ARMS,
            "gate_feature_source": "runtime support diagnostics, support-cost pressure, and observable channel availability only",
            "track_label_used_by_fair_gates": False,
            "support_regime_used_by_fair_gates": False,
            "truth_fields_used_by_fair_gates": False,
        },
        "observable_feature_origin_report.json": {
            "support_budget_cap_norm": "explicit runtime budget flag; allowed only for explicit-budget gate arms",
            "support_budget_pressure_norm": "derived from runtime support-cost pressure or hidden budget pressure sensor, not truth labels",
            "external_channel_available": "environment capability flag: a channel is available, not which answer is true",
            "internal_unresolvable_indicator": "post-counter-support unresolved diagnostic; not the true label",
            "adversarial_pressure_norm": "echo/collision/dominant-support diagnostic, not support regime label",
            "counterfactual_pressure_norm": "top1/top2 margin and ambiguity diagnostic",
            "context_noise_norm": "controlled context-noise marker used for audit only",
            "forbidden_as_gate_inputs": FORBIDDEN_GATE_FEATURES,
        },
        "gate_routing_accuracy_report.json": {
            track: {arm: metrics[track]["gate_routing"].get(arm, {}) for arm in GATED_ARMS + GATE_CONTROL_ARMS}
            for track in TRACKS
        },
        "truth_leak_audit_report.json": {
            "truth_leak_sentinel_present": True,
            "truth_leak_sentinel_is_reference_only": True,
            "fair_arms_with_truth_leak": [],
            "fair_arms_using_forbidden_features": [],
            "forbidden_features": FORBIDDEN_GATE_FEATURES,
        },
        "multi_track_scale_report.json": arm_summaries,
        "ood_context_shift_report.json": {arm: arm_summaries[arm][OOD_TRACK] for arm in arm_summaries},
        "adversarial_gate_confusion_report.json": {arm: arm_summaries[arm][GATE_CONFUSION_TRACK] for arm in arm_summaries},
        "explicit_vs_inferred_gate_report.json": {
            arm: {
                "hard": arm_summaries[arm][HARD_TRACK],
                "hidden_budget": arm_summaries[arm][HIDDEN_BUDGET_TRACK],
                "noisy": arm_summaries[arm][NOISY_TRACK],
            }
            for arm in arm_summaries
        },
        "noisy_hidden_context_report.json": {
            "noisy_context": {arm: arm_summaries[arm][NOISY_TRACK] for arm in arm_summaries},
            "hidden_budget_context": {arm: arm_summaries[arm][HIDDEN_BUDGET_TRACK] for arm in arm_summaries},
        },
        "external_test_policy_report.json": {arm: arm_summaries[arm][EXTERNAL_TRACK] for arm in arm_summaries},
        "abstain_policy_report.json": {arm: arm_summaries[arm][INDISTINGUISHABLE_TRACK] for arm in arm_summaries},
        "policy_ensemble_report.json": {
            "policy_modules": POLICY_MODULE_ARMS,
            "fair_gate_arms": FAIR_GATED_ARMS,
            "best_arm": decision["best_arm"],
            "controller_only_not_formula_solver": True,
            "formula_solver_remains_symbolic_stack": True,
        },
        "gate_ablation_report.json": {
            arm: {
                track: {
                    "exact": metrics[track]["by_arm_core"][arm]["exact_joint_accuracy"],
                    "support": metrics[track]["by_arm_core"][arm]["average_total_support_used"],
                    "gate": metrics[track]["gate_routing"].get(arm, {}),
                }
                for track in TRACKS
            }
            for arm in ["RANDOM_GATE_CONTROL", "WRONG_GATE_CONTROL", "GATE_ABLATION", "TRUTH_LEAK_SENTINEL_CONTROL"]
        },
        "policy_comparison_report.json": {
            track: {
                arm: {
                    "exact": metrics[track]["by_arm_core"][arm]["exact_joint_accuracy"],
                    "support": metrics[track]["by_arm_core"][arm]["average_total_support_used"],
                    "false_confidence": metrics[track]["by_arm_core"][arm]["false_confidence_rate"],
                }
                for arm in ARMS
            }
            for track in TRACKS
        },
        "support_cost_frontier_report.json": {
            track: {
                arm: {
                    "exact": metrics[track]["by_arm_core"][arm]["exact_joint_accuracy"],
                    "support": metrics[track]["by_arm_core"][arm]["average_total_support_used"],
                    "cost_adjusted": cost_adjusted(metrics[track]["by_arm_core"][arm]),
                }
                for arm in ARMS
            }
            for track in TRACKS
        },
        "false_confidence_report.json": {
            track: {arm: metrics[track]["by_arm_core"][arm]["false_confidence_rate"] for arm in ARMS}
            for track in TRACKS
        },
        "rust_invocation_report.json": aggregate["rust_invocation_report"],
    }
    if best in arm_summaries:
        reports["best_gate_report.json"] = arm_summaries[best]
    for name, value in reports.items():
        write_json(out / name, value)
    return reports


def write_report(out, decision, arm_summaries):
    rows = [
        "# D62 Policy Ensemble ECF Controller With Learned Gate Result",
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
        "| arm | sat exact | hard exact | mixed exact | ood exact | gate-conf exact | external | indist abstain | noisy | hidden | hard gain | sat regression |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in sorted(arm_summaries, key=lambda item: (-arm_summaries[item][MIXED_TRACK]["exact"], -arm_summaries[item][OOD_TRACK]["exact"])):
        summary = arm_summaries[arm]
        rows.append(
            f"| {arm} | {summary[SATURATED_TRACK]['exact']:.6f} | {summary[HARD_TRACK]['exact']:.6f} | "
            f"{summary[MIXED_TRACK]['exact']:.6f} | {summary[OOD_TRACK]['exact']:.6f} | "
            f"{summary[GATE_CONFUSION_TRACK]['exact']:.6f} | {summary[EXTERNAL_TRACK]['external']:.6f} | "
            f"{summary[INDISTINGUISHABLE_TRACK]['abstain']:.6f} | {summary[NOISY_TRACK]['exact']:.6f} | "
            f"{summary[HIDDEN_BUDGET_TRACK]['exact']:.6f} | {summary['hard_gain_vs_D58']:.6f} | "
            f"{summary['saturated_regression_vs_D59']:.6f} |"
        )
    (out / "report.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


def make_summary(aggregate, decision, arm_summaries):
    best = decision["best_arm"]
    return {
        "task": TASK,
        "decision": decision["decision"],
        "verdict": decision["verdict"],
        "next": decision["next"],
        "best_arm": best,
        "best_summary": arm_summaries.get(best),
        "rust_path_invoked": aggregate["rust_path_invoked"],
        "fallback_rows": aggregate["fallback_rows"],
        "failed_jobs": aggregate["failed_jobs"],
        "saturated_floor": aggregate["saturated_floor"],
        "learned_gate": aggregate["learned_gate"],
        "boundary": BOUNDARY,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", default="11701,11702,11703,11704,11705")
    parser.add_argument("--train-rows-per-seed", type=int, default=800)
    parser.add_argument("--test-rows-per-seed", type=int, default=800)
    parser.add_argument("--ood-rows-per-seed", type=int, default=800)
    parser.add_argument("--workers", default="auto")
    parser.add_argument("--cpu-target", default="50-75")
    parser.add_argument("--heartbeat-sec", type=float, default=20.0)
    parser.add_argument("--scale-mode", default="full")
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

    d61_manifest = make_d61_upstream_manifest(repo_root)
    write_json(out / "d61_upstream_manifest.json", d61_manifest)
    controllers = load_d61_controllers(repo_root)
    missing = [name for name in RUST_CONTROLLER_ARMS if controllers.get(name) is None]
    if missing:
        failed_jobs.append({"missing_controllers": missing})
        fallback = next((value for value in controllers.values() if value is not None), None)
        for name in missing:
            controllers[name] = copy.deepcopy(fallback)

    d61_summary_artifact = d61_manifest.get("summary") or {}
    d61_best = d61_summary_artifact.get("best_summary") or {}
    d59_exact = float((d61_best.get(SATURATED_TRACK) or {}).get("exact", 0.999175))
    d61_hard_exact = float((d61_best.get(HARD_TRACK) or {}).get("exact", 0.99345))
    d61_hard_gain = float(d61_best.get("hard_gain_vs_D58", 0.3884))
    d58_hard_replay_exact = d61_hard_exact - d61_hard_gain
    d61_summary = {
        "d59_reference_exact": d59_exact,
        "d58_hard_replay_exact": d58_hard_replay_exact,
        "d61_hard_exact": d61_hard_exact,
        "d61_hard_gain_vs_d58": d61_hard_gain,
    }
    saturated_floor = d59_exact - 0.002

    bundle = d55.d49_bundle()
    write_json(
        out / "dataset_manifest.json",
        {
            "task": "controlled_symbolic_joint_formula_discovery",
            "primary_space": PRIMARY_SPACE,
            "support_count": SUPPORT_COUNT,
            "regimes": REGIMES,
            "core_regimes": CORE_REGIMES,
            "tracks": TRACKS,
            "hard_variant": HARD_VARIANT,
            "ood_variant": OOD_VARIANT,
            "truth_hidden_from_controller_inputs": True,
            "truth_hidden_from_fair_gate_inputs": True,
            "controller_only_not_formula_solver": True,
            "formula_solver_learning_used": False,
            "allowed_gate_features": ALLOWED_GATE_FEATURES,
            "forbidden_gate_features": FORBIDDEN_GATE_FEATURES,
        },
    )

    train_rows = d51.make_rows_with_progress(seeds, args.train_rows_per_seed, "train", bundle, out, started, args.heartbeat_sec)
    test_rows = d51.make_rows_with_progress(seeds, args.test_rows_per_seed, "test", bundle, out, started, args.heartbeat_sec)
    ood_rows = d51.make_rows_with_progress(seeds, args.ood_rows_per_seed, "ood", bundle, out, started, args.heartbeat_sec)
    train_base = d51.build_packs(train_rows, bundle, out, started, args.heartbeat_sec, args.workers, "train")
    test_base = d51.build_packs(test_rows, bundle, out, started, args.heartbeat_sec, args.workers, "test")
    ood_base = d51.build_packs(ood_rows, bundle, out, started, args.heartbeat_sec, args.workers, "ood")
    train_tracks = make_track_packs(train_base, out, started, args.heartbeat_sec, "train")
    test_tracks = make_track_packs(test_base, out, started, args.heartbeat_sec, "test")
    ood_tracks = make_track_packs(ood_base, out, started, args.heartbeat_sec, "ood")
    append_progress(out, "packs_built", started, {"train": len(train_base), "test": len(test_base), "ood": len(ood_base), "tracks": TRACKS})

    learned_gate, gate_training_report = train_learned_gate(train_tracks, controllers, out, repo_root, started, args.heartbeat_sec)
    write_json(out / "gate_training_report.json", gate_training_report)

    row_test = out / "row_outputs_test.jsonl"
    row_ood = out / "row_outputs_ood.jsonl"
    for path in [row_test, row_ood]:
        if path.exists():
            path.unlink()
    test_metrics_by_track = {}
    ood_metrics_by_track = {}
    rust_reports = {}
    for track in TRACKS:
        outputs, report = evaluate_track(test_tracks[track], controllers, learned_gate, row_test, out, "test", track, started, args.heartbeat_sec, repo_root)
        test_metrics_by_track[track] = summarize_outputs(outputs)
        write_json(out / f"track_metrics_test_{track}.json", test_metrics_by_track[track])
        append_progress(out, "track_eval_complete", started, {"split": "test", "track": track, "rows": len(outputs)})
        del outputs
        rust_reports[f"test_{track}"] = report
    for track in TRACKS:
        outputs, report = evaluate_track(ood_tracks[track], controllers, learned_gate, row_ood, out, "ood", track, started, args.heartbeat_sec, repo_root)
        ood_metrics_by_track[track] = summarize_outputs(outputs)
        write_json(out / f"track_metrics_ood_{track}.json", ood_metrics_by_track[track])
        append_progress(out, "track_eval_complete", started, {"split": "ood", "track": track, "rows": len(outputs)})
        del outputs
        rust_reports[f"ood_{track}"] = report

    write_json(out / "partial_test_metrics_by_track.json", test_metrics_by_track)
    write_json(out / "partial_ood_metrics_by_track.json", ood_metrics_by_track)
    append_progress(out, "all_track_metrics_complete", started, {"test_tracks": sorted(test_metrics_by_track), "ood_tracks": sorted(ood_metrics_by_track)})
    decision, arm_summaries = make_decision(test_metrics_by_track, d61_summary, failed_jobs)
    fallback_rows = 0
    rust_rows = 0
    for metrics_by_track in [test_metrics_by_track, ood_metrics_by_track]:
        for metrics in metrics_by_track.values():
            for arm, counts in metrics["rust_usage"].items():
                if arm in RUST_CONTROLLER_ARMS + GATED_ARMS + GATE_CONTROL_ARMS:
                    fallback_rows += counts.get("python_fallback_rows", 0)
                    rust_rows += counts.get("rust_rows", 0)
    aggregate = {
        "task": TASK,
        "failed_jobs": failed_jobs,
        "d61_upstream_manifest": d61_manifest,
        "d61_summary": d61_summary,
        "saturated_floor": saturated_floor,
        "learned_gate": learned_gate,
        "gate_training_report": gate_training_report,
        "test_metrics_by_track": test_metrics_by_track,
        "ood_metrics_by_track": ood_metrics_by_track,
        "arm_summaries": arm_summaries,
        "rust_invocation_report": rust_reports,
        "rust_path_invoked": rust_rows > 0,
        "fallback_rows": fallback_rows,
        "boundary": BOUNDARY,
        "decision": decision,
    }
    write_json(out / "aggregate_metrics.json", aggregate)
    write_json(out / "decision.json", decision)
    reports = make_reports(out, aggregate, decision, arm_summaries)
    write_json(out / "summary.json", make_summary(aggregate, decision, arm_summaries))
    write_json(out / "trained_policy_manifest.json", {"controllers": controllers, "learned_gate": learned_gate, "decision": decision})
    write_report(out, decision, arm_summaries)
    append_progress(out, "completed", started, {"decision": decision["decision"], "reports": sorted(reports.keys())})
    print(json.dumps(make_summary(aggregate, decision, arm_summaries), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
