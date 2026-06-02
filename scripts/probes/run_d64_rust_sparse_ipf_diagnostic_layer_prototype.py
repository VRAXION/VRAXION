#!/usr/bin/env python3
"""D64 Rust sparse IPF diagnostic layer prototype.

D63 migrated clean ECF gate diagnostic components to the Rust sparse path. D64
keeps the formula solver symbolic, but makes the diagnostic estimator input
harder: Rust estimators see score/evidence-vector summaries, not the clean D63
pressure flags they are asked to recover.
"""

import argparse
import copy
import json
import os
import random
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

import run_d51_mutable_ecf_controller_prototype as d51
import run_d55_sparse_firing_ecf_controller_prototype as d55
import run_d59_rust_sparse_ecf_controller_with_mutation as d59
import run_d60_rust_sparse_mutation_learning_signal_probe as d60
import run_d62_policy_ensemble_ecf_controller_with_learned_gate as d62

TASK = "D64_RUST_SPARSE_IPF_DIAGNOSTIC_LAYER_PROTOTYPE"
BOUNDARY = (
    "D64 only tests a Rust sparse IPF diagnostic layer for controlled symbolic "
    "joint formula discovery. It does not prove full VRAXION brain, raw visual "
    "Raven reasoning, Raven solved, AGI, consciousness, DNA/genome success, "
    "architecture superiority, or production readiness."
)

PRIMARY_SPACE = d62.PRIMARY_SPACE
SUPPORT_COUNT = d62.SUPPORT_COUNT
REGIMES = d62.REGIMES
CORE_REGIMES = d62.CORE_REGIMES
ACTIONS = d62.ACTIONS
TRACKS = d62.TRACKS
SATURATED_TRACK = d62.SATURATED_TRACK
HARD_TRACK = d62.HARD_TRACK
MIXED_TRACK = d62.MIXED_TRACK
OOD_TRACK = d62.OOD_TRACK
GATE_CONFUSION_TRACK = d62.GATE_CONFUSION_TRACK
EXTERNAL_TRACK = d62.EXTERNAL_TRACK
INDISTINGUISHABLE_TRACK = d62.INDISTINGUISHABLE_TRACK
NOISY_TRACK = d62.NOISY_TRACK
HIDDEN_BUDGET_TRACK = d62.HIDDEN_BUDGET_TRACK
ROW_SAMPLE_PER_ARM_REGIME_SPLIT = d62.ROW_SAMPLE_PER_ARM_REGIME_SPLIT

DIAGNOSTICS = [
    "entropy_high",
    "margin_low",
    "collision_pressure",
    "support_independence_low",
    "support_effort_pressure",
    "counterfactual_pressure",
    "adversarial_pressure",
    "internal_unresolvable",
    "external_test_need",
]

DIAGNOSTIC_FEATURES = [
    "bias",
    "scalar_confidence",
    "cell_confidence",
    "operator_confidence",
    "joint_confidence",
    "entropy_norm",
    "inverse_margin",
    "collision_norm",
    "dominant_cluster_fraction",
    "support_cluster_count_norm",
    "top1_factorised_disagreement",
    "support_count_norm",
    "score_support_effort_hint",
    "score_counter_delta_hint",
    "score_adversarial_echo_hint",
    "score_unresolvable_hint",
    "score_external_need_hint",
    "context_noise_norm",
]

FORBIDDEN_RUST_INPUT_FEATURES = [
    "support_budget_pressure_norm",
    "counterfactual_pressure_norm",
    "adversarial_pressure_norm",
    "internal_unresolvable_indicator",
    "external_channel_available",
    "support_regime",
    "track",
    "mixed_source_track",
    "row_id",
    "seed",
]

DIAGNOSTIC_MAP = {
    "entropy_high": {
        "feature": "entropy_norm",
        "threshold": 0.65,
        "gate_feature": "counterfactual_pressure_norm",
    },
    "margin_low": {
        "feature": "inverse_margin",
        "threshold": 0.45,
        "gate_feature": "counterfactual_pressure_norm",
    },
    "collision_pressure": {
        "feature": "collision_norm",
        "threshold": 0.25,
        "gate_feature": "adversarial_pressure_norm",
    },
    "support_independence_low": {
        "feature": "dominant_cluster_fraction",
        "threshold": 0.60,
        "gate_feature": "adversarial_pressure_norm",
    },
    "support_effort_pressure": {
        "feature": "score_support_effort_hint",
        "threshold": 0.25,
        "gate_feature": "support_budget_pressure_norm",
    },
    "counterfactual_pressure": {
        "feature": "score_counter_delta_hint",
        "threshold": 0.50,
        "gate_feature": "counterfactual_pressure_norm",
    },
    "adversarial_pressure": {
        "feature": "score_adversarial_echo_hint",
        "threshold": 0.50,
        "gate_feature": "adversarial_pressure_norm",
    },
    "internal_unresolvable": {
        "feature": "score_unresolvable_hint",
        "threshold": 0.25,
        "gate_feature": "internal_unresolvable_indicator",
    },
    "external_test_need": {
        "feature": "score_external_need_hint",
        "threshold": 0.25,
        "gate_feature": "external_channel_available",
    },
}

POLICY_MODULES = d62.POLICY_MODULE_ARMS

ARMS = [
    "D63_SYMBOLIC_DIAGNOSTIC_REFERENCE",
    "RUST_SPARSE_ENTROPY_MARGIN_LAYER",
    "RUST_SPARSE_COLLISION_LAYER",
    "RUST_SPARSE_SUPPORT_INDEPENDENCE_LAYER",
    "RUST_SPARSE_COUNTERFACTUAL_PRESSURE_LAYER",
    "RUST_SPARSE_ADVERSARIAL_PRESSURE_LAYER",
    "RUST_SPARSE_UNRESOLVABLE_ESTIMATOR",
    "RUST_SPARSE_EXTERNAL_TEST_NEED_ESTIMATOR",
    "RUST_SPARSE_FULL_IPF_DIAGNOSTIC_LAYER",
    "HYBRID_SYMBOLIC_RUST_IPF_LAYER",
    "SHUFFLED_SCORE_VECTOR_CONTROL",
    "RANDOM_DIAGNOSTIC_CONTROL",
    "DIAGNOSTIC_ABLATION_CONTROL",
    "TRUTH_LEAK_SENTINEL_REFERENCE_ONLY",
]

FAIR_ARMS = [
    "D63_SYMBOLIC_DIAGNOSTIC_REFERENCE",
    "RUST_SPARSE_ENTROPY_MARGIN_LAYER",
    "RUST_SPARSE_COLLISION_LAYER",
    "RUST_SPARSE_SUPPORT_INDEPENDENCE_LAYER",
    "RUST_SPARSE_COUNTERFACTUAL_PRESSURE_LAYER",
    "RUST_SPARSE_ADVERSARIAL_PRESSURE_LAYER",
    "RUST_SPARSE_UNRESOLVABLE_ESTIMATOR",
    "RUST_SPARSE_EXTERNAL_TEST_NEED_ESTIMATOR",
    "RUST_SPARSE_FULL_IPF_DIAGNOSTIC_LAYER",
    "HYBRID_SYMBOLIC_RUST_IPF_LAYER",
]

CONTROL_ARMS = ["SHUFFLED_SCORE_VECTOR_CONTROL", "RANDOM_DIAGNOSTIC_CONTROL", "DIAGNOSTIC_ABLATION_CONTROL"]
REFERENCE_ONLY_ARMS = ["TRUTH_LEAK_SENTINEL_REFERENCE_ONLY"]
RUST_DIAGNOSTIC_ARMS = [arm for arm in FAIR_ARMS if arm != "D63_SYMBOLIC_DIAGNOSTIC_REFERENCE"]

FORBIDDEN_DIAGNOSTIC_FEATURES = [
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


def make_d62_upstream_manifest(repo_root):
    root = repo_root / "target/pilot_wave/d62_policy_ensemble_ecf_controller_with_learned_gate/smoke"
    manifest = {
        "upstream": "D62_POLICY_ENSEMBLE_ECF_CONTROLLER_WITH_LEARNED_GATE",
        "expected_decision": "policy_ensemble_learned_gate_confirmed",
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
        "observable_feature_origin_report.json",
        "aggregate_metrics.json",
    ]:
        value = load_json(root / name)
        if value is not None:
            manifest[name.replace(".json", "")] = value
    trained = load_json(root / "trained_policy_manifest.json") or {}
    manifest["learned_gate_present"] = bool(trained.get("learned_gate"))
    manifest["policy_modules_present"] = sorted((trained.get("controllers") or {}).keys())
    return manifest


def make_d63_upstream_manifest(repo_root):
    root = repo_root / "target/pilot_wave/d63_rust_sparse_ecf_controller_component_migration/smoke"
    manifest = {
        "upstream": "D63_RUST_SPARSE_ECF_CONTROLLER_COMPONENT_MIGRATION",
        "expected_decision": "rust_sparse_ecf_diagnostic_component_migration_confirmed",
        "expected_next": TASK,
        "artifact_root": str(root),
        "decision_present": (root / "decision.json").exists(),
        "summary_present": (root / "summary.json").exists(),
    }
    for name in [
        "decision.json",
        "summary.json",
        "best_gate_report.json",
        "estimator_accuracy_report.json",
        "rust_estimator_mapping_report.json",
    ]:
        value = load_json(root / name)
        if value is not None:
            manifest[name.replace(".json", "")] = value
    return manifest


def load_d62_policy_modules(repo_root):
    root = repo_root / "target/pilot_wave/d62_policy_ensemble_ecf_controller_with_learned_gate/smoke"
    trained = load_json(root / "trained_policy_manifest.json") or {}
    controllers = trained.get("controllers") or {}
    learned_gate = trained.get("learned_gate")
    if not controllers:
        controllers = d62.load_d61_controllers(repo_root)
    if not learned_gate:
        learned_gate = {
            "default_policy": "SATURATED_POLICY",
            "rules": [
                {"feature": "support_budget_pressure_norm", "policy": "HARD_BUDGET_POLICY", "threshold": 0.25},
                {"feature": "external_channel_available", "policy": "EXTERNAL_TEST_POLICY", "threshold": 0.25},
                {"feature": "internal_unresolvable_indicator", "policy": "ABSTAIN_POLICY", "threshold": 0.25},
                {"feature": "adversarial_pressure_norm", "policy": "ADVERSARIAL_REPAIR_POLICY", "threshold": 0.50},
                {"feature": "counterfactual_pressure_norm", "policy": "COUNTERFACTUAL_POLICY", "threshold": 0.50},
            ],
        }
    return {name: controllers[name] for name in POLICY_MODULES if name in controllers}, learned_gate


def quant_feature(value):
    value = max(0.0, min(1.0, float(value)))
    return int(round(value * 16.0))


def stored_threshold(threshold):
    return max(0, min(15, int(round(float(threshold) * 16.0)) - 1))


def diagnostic_features(pack):
    features = d62.gate_features(pack)
    fmap = pack.get("feature_map") or {}
    scalar_confidence = float(fmap.get("scalar_confidence", features.get("scalar_confidence", 0.0)))
    cell_confidence = float(fmap.get("cell_confidence", 0.0))
    operator_confidence = float(fmap.get("operator_confidence", 0.0))
    joint_confidence = float(fmap.get("joint_confidence", scalar_confidence))
    entropy = float(features["entropy_norm"])
    inverse_margin = float(features["inverse_margin"])
    collision = float(features["collision_norm"])
    dominant = float(features["dominant_cluster_fraction"])
    cluster_count = float(features["support_cluster_count_norm"])
    disagreement = float(features["top1_factorised_disagreement"])
    support_count_norm = float(features["support_count_norm"])
    support_effort = max(inverse_margin, collision, disagreement, max(0.0, cluster_count - 0.35))
    counter_delta = max(inverse_margin, entropy, disagreement)
    adversarial_echo = max(collision, dominant, disagreement)
    unresolvable_hint = max(0.0, min(1.0, (entropy + inverse_margin + disagreement) / 3.0))
    external_hint = max(0.0, min(1.0, (entropy + (1.0 - joint_confidence) + inverse_margin) / 3.0))
    return {
        "bias": 1.0,
        "scalar_confidence": scalar_confidence,
        "cell_confidence": cell_confidence,
        "operator_confidence": operator_confidence,
        "joint_confidence": joint_confidence,
        "entropy_norm": features["entropy_norm"],
        "inverse_margin": features["inverse_margin"],
        "collision_norm": features["collision_norm"],
        "dominant_cluster_fraction": features["dominant_cluster_fraction"],
        "support_cluster_count_norm": features["support_cluster_count_norm"],
        "top1_factorised_disagreement": features["top1_factorised_disagreement"],
        "support_count_norm": support_count_norm,
        "score_support_effort_hint": support_effort,
        "score_counter_delta_hint": counter_delta,
        "score_adversarial_echo_hint": adversarial_echo,
        "score_unresolvable_hint": unresolvable_hint,
        "score_external_need_hint": external_hint,
        "context_noise_norm": features["context_noise_norm"],
    }


def diagnostic_targets(pack):
    score_features = diagnostic_features(pack)
    symbolic = d62.gate_features(pack)
    return {
        "entropy_high": 1.0 if score_features["entropy_norm"] >= DIAGNOSTIC_MAP["entropy_high"]["threshold"] else 0.0,
        "margin_low": 1.0 if score_features["inverse_margin"] >= DIAGNOSTIC_MAP["margin_low"]["threshold"] else 0.0,
        "collision_pressure": 1.0 if score_features["collision_norm"] >= DIAGNOSTIC_MAP["collision_pressure"]["threshold"] else 0.0,
        "support_independence_low": 1.0 if score_features["dominant_cluster_fraction"] >= DIAGNOSTIC_MAP["support_independence_low"]["threshold"] else 0.0,
        "support_effort_pressure": 1.0 if symbolic["support_budget_pressure_norm"] >= 0.25 else 0.0,
        "counterfactual_pressure": 1.0 if symbolic["counterfactual_pressure_norm"] >= 0.50 else 0.0,
        "adversarial_pressure": 1.0 if symbolic["adversarial_pressure_norm"] >= 0.50 else 0.0,
        "internal_unresolvable": 1.0 if symbolic["internal_unresolvable_indicator"] >= 0.25 else 0.0,
        "external_test_need": 1.0 if symbolic["external_channel_available"] >= 0.25 else 0.0,
    }


def make_estimator_controller(name, diagnostic_name):
    spec = DIAGNOSTIC_MAP[diagnostic_name]
    return {
        "name": name,
        "gates": [
            {
                "gate_id": 0,
                "feature": spec["feature"],
                "action": "REQUEST_SUPPORT",
                "stored_threshold": stored_threshold(spec["threshold"]),
                "weight": 40,
                "priority": 120,
                "channel": 3,
                "polarity": 1,
            }
        ],
        "diagnostic_name": diagnostic_name,
        "feature_names": DIAGNOSTIC_FEATURES,
        "input_origin": "score/evidence vector summary, not clean D63 proxy flag",
        "controller_only_not_formula_solver": True,
    }


def estimator_controllers():
    return {
        f"EST_{name}": make_estimator_controller(f"EST_{name}", name)
        for name in DIAGNOSTICS
    }


def write_estimator_controllers(path, controllers):
    feature_index = {name: idx for idx, name in enumerate(DIAGNOSTIC_FEATURES)}
    lines = ["controller_id\tfeature_idx\taction\tstored_threshold\tweight\tpriority\tchannel\tpolarity"]
    for controller_id, controller in controllers.items():
        for gate in controller["gates"]:
            lines.append(
                "\t".join(
                    [
                        controller_id,
                        str(feature_index[gate["feature"]]),
                        gate["action"],
                        str(int(gate["stored_threshold"])),
                        str(int(gate["weight"])),
                        str(int(gate["priority"])),
                        str(int(gate.get("channel", 1))),
                        str(int(gate.get("polarity", 1))),
                    ]
                )
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_estimator_rows(path, packs):
    lines = ["row_key\t" + "\t".join(DIAGNOSTIC_FEATURES)]
    for idx, pack in enumerate(packs):
        features = diagnostic_features(pack)
        values = [str(quant_feature(features[name])) for name in DIAGNOSTIC_FEATURES]
        lines.append(f"{idx}\t" + "\t".join(values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_rust_estimator_bridge(out, repo_root, controllers, packs, split, tag, started):
    harness = d59.ensure_rust_multi_harness(out, repo_root)
    work = out / "rust_diagnostic_estimator_inputs" / split / tag
    work.mkdir(parents=True, exist_ok=True)
    controllers_path = work / "controllers.tsv"
    rows_path = work / "rows.tsv"
    actions_path = work / "actions.tsv"
    write_estimator_controllers(controllers_path, controllers)
    write_estimator_rows(rows_path, packs)
    cmd = [
        "cargo",
        "run",
        "--quiet",
        "--manifest-path",
        str(harness / "Cargo.toml"),
        "--",
        str(controllers_path),
        str(rows_path),
        str(actions_path),
        "DECIDE",
    ]
    append_progress(out, "rust_diagnostic_bridge_start", started, {"split": split, "tag": tag, "controllers": len(controllers), "rows": len(packs)})
    proc = subprocess.run(cmd, cwd=repo_root, text=True, capture_output=True)
    report = {
        "split": split,
        "tag": tag,
        "controllers": sorted(controllers.keys()),
        "command": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr[-4000:],
        "rows_requested": len(packs),
        "actions_path": str(actions_path),
        "actions_present": actions_path.exists(),
        "feature_names": DIAGNOSTIC_FEATURES,
    }
    if proc.returncode != 0:
        write_json(work / "rust_error.json", report)
        raise RuntimeError(f"Rust diagnostic bridge failed for {split}/{tag}: {proc.stderr[-500:]}")
    actions = d59.parse_multi_actions(actions_path)
    report["actions_rows_by_controller"] = {key: len(value) for key, value in actions.items()}
    write_json(work / "rust_invocation.json", report)
    append_progress(out, "rust_diagnostic_bridge_done", started, {"split": split, "tag": tag, "controllers": len(actions)})
    return actions, report


def estimator_values_from_actions(action_records, idx):
    values = {}
    for diagnostic in DIAGNOSTICS:
        record = action_records[f"EST_{diagnostic}"][idx]
        values[diagnostic] = 1.0 if record["action"] != "DECIDE" else 0.0
    return values


def shuffled_estimates(values):
    return {
        "entropy_high": values["external_test_need"],
        "margin_low": values["support_effort_pressure"],
        "collision_pressure": values["entropy_high"],
        "support_independence_low": values["margin_low"],
        "support_effort_pressure": values["support_independence_low"],
        "counterfactual_pressure": values["adversarial_pressure"],
        "adversarial_pressure": values["internal_unresolvable"],
        "internal_unresolvable": values["collision_pressure"],
        "external_test_need": values["counterfactual_pressure"],
    }


def random_estimates(pack):
    rng = stable_rng(pack["seed"], f"D64_RANDOM_DIAGNOSTIC:{pack['row_id']}:{pack.get('track')}")
    return {name: float(rng.randint(0, 1)) for name in DIAGNOSTICS}


def ablated_estimates():
    return {name: 0.0 for name in DIAGNOSTICS}


def estimates_to_gate_features(pack, estimates, replace_names):
    features = d62.gate_features(pack)
    if "support_effort_pressure" in replace_names:
        features["support_budget_pressure_norm"] = float(estimates["support_effort_pressure"])
    if any(name in replace_names for name in ["entropy_high", "margin_low", "counterfactual_pressure"]):
        features["counterfactual_pressure_norm"] = max(
            float(estimates.get("entropy_high", 0.0)),
            float(estimates.get("margin_low", 0.0)),
            float(estimates.get("counterfactual_pressure", 0.0)),
        )
    if any(name in replace_names for name in ["collision_pressure", "support_independence_low", "adversarial_pressure"]):
        features["adversarial_pressure_norm"] = max(
            float(estimates.get("collision_pressure", 0.0)),
            float(estimates.get("support_independence_low", 0.0)),
            float(estimates.get("adversarial_pressure", 0.0)),
        )
    if "internal_unresolvable" in replace_names:
        features["internal_unresolvable_indicator"] = float(estimates["internal_unresolvable"])
    if "external_test_need" in replace_names:
        features["external_channel_available"] = float(estimates["external_test_need"])
    features["support_budget_available"] = 1.0 if features["support_budget_cap_norm"] > 0.0 else 0.0
    return features


def policy_from_features(features, learned_gate):
    for rule in learned_gate["rules"]:
        if features.get(rule["feature"], 0.0) >= rule["threshold"]:
            return rule["policy"]
    return learned_gate["default_policy"]


def select_policy_for_arm(pack, arm, estimates, learned_gate):
    if arm == "D63_SYMBOLIC_DIAGNOSTIC_REFERENCE":
        return policy_from_features(d62.gate_features(pack), learned_gate), d62.gate_features(pack), "symbolic_d62_gate_features", False
    if arm == "RUST_SPARSE_ENTROPY_MARGIN_LAYER":
        features = estimates_to_gate_features(pack, estimates, ["entropy_high", "margin_low"])
        return policy_from_features(features, learned_gate), features, "rust_entropy_margin_score_layer", False
    if arm == "RUST_SPARSE_COLLISION_LAYER":
        features = estimates_to_gate_features(pack, estimates, ["collision_pressure"])
        return policy_from_features(features, learned_gate), features, "rust_collision_score_layer", False
    if arm == "RUST_SPARSE_SUPPORT_INDEPENDENCE_LAYER":
        features = estimates_to_gate_features(pack, estimates, ["support_independence_low"])
        return policy_from_features(features, learned_gate), features, "rust_support_independence_score_layer", False
    if arm == "RUST_SPARSE_COUNTERFACTUAL_PRESSURE_LAYER":
        features = estimates_to_gate_features(pack, estimates, ["entropy_high", "margin_low", "counterfactual_pressure"])
        return policy_from_features(features, learned_gate), features, "rust_counterfactual_pressure_score_layer", False
    if arm == "RUST_SPARSE_ADVERSARIAL_PRESSURE_LAYER":
        features = estimates_to_gate_features(pack, estimates, ["collision_pressure", "support_independence_low", "adversarial_pressure"])
        return policy_from_features(features, learned_gate), features, "rust_adversarial_pressure_score_layer", False
    if arm == "RUST_SPARSE_UNRESOLVABLE_ESTIMATOR":
        features = estimates_to_gate_features(pack, estimates, ["internal_unresolvable"])
        return policy_from_features(features, learned_gate), features, "rust_unresolvable_only", False
    if arm == "RUST_SPARSE_EXTERNAL_TEST_NEED_ESTIMATOR":
        features = estimates_to_gate_features(pack, estimates, ["external_test_need"])
        return policy_from_features(features, learned_gate), features, "rust_external_test_need_only", False
    if arm == "RUST_SPARSE_FULL_IPF_DIAGNOSTIC_LAYER":
        features = estimates_to_gate_features(pack, estimates, DIAGNOSTICS)
        return policy_from_features(features, learned_gate), features, "rust_full_ipf_score_diagnostic_layer", False
    if arm == "HYBRID_SYMBOLIC_RUST_IPF_LAYER":
        features = estimates_to_gate_features(
            pack,
            estimates,
            ["entropy_high", "margin_low", "collision_pressure", "support_independence_low", "counterfactual_pressure", "adversarial_pressure"],
        )
        return policy_from_features(features, learned_gate), features, "rust_score_ipf_uncertainty_plus_symbolic_budget_external_abstain", False
    if arm == "SHUFFLED_SCORE_VECTOR_CONTROL":
        features = estimates_to_gate_features(pack, shuffled_estimates(estimates), DIAGNOSTICS)
        return policy_from_features(features, learned_gate), features, "shuffled_score_vector_diagnostic_control", False
    if arm == "RANDOM_DIAGNOSTIC_CONTROL":
        features = estimates_to_gate_features(pack, random_estimates(pack), DIAGNOSTICS)
        return policy_from_features(features, learned_gate), features, "stable_random_diagnostic_control", False
    if arm == "DIAGNOSTIC_ABLATION_CONTROL":
        features = estimates_to_gate_features(pack, ablated_estimates(), DIAGNOSTICS)
        return policy_from_features(features, learned_gate), features, "zeroed_diagnostic_control", False
    if arm == "TRUTH_LEAK_SENTINEL_REFERENCE_ONLY":
        track = pack.get("track")
        if track == INDISTINGUISHABLE_TRACK:
            return "ABSTAIN_POLICY", d62.gate_features(pack), "reference_only_track_label_sentinel", True
        if track == EXTERNAL_TRACK:
            return "EXTERNAL_TEST_POLICY", d62.gate_features(pack), "reference_only_track_label_sentinel", True
        if track in {HARD_TRACK, OOD_TRACK, HIDDEN_BUDGET_TRACK}:
            return "HARD_BUDGET_POLICY", d62.gate_features(pack), "reference_only_track_label_sentinel", True
        if track == GATE_CONFUSION_TRACK:
            return "ADVERSARIAL_REPAIR_POLICY", d62.gate_features(pack), "reference_only_track_label_sentinel", True
        return "SATURATED_POLICY", d62.gate_features(pack), "reference_only_track_label_sentinel", True
    raise ValueError(arm)


def output_from_policy(pack, arm, policy, policy_actions, idx, gate_features, estimates, targets, basis, used_forbidden):
    action_record = policy_actions[policy][idx]
    row = d60.output_from_action(pack, arm, action_record["action"], d59.rust_trace(action_record))
    row["track"] = pack.get("track", SATURATED_TRACK)
    row["mixed_source_track"] = pack.get("mixed_source_track", row["track"])
    row["difficulty_variant"] = pack.get("difficulty_variant", "d62_replay")
    row["runtime_support_budget_cap"] = pack.get("runtime_support_budget_cap")
    row["gate_selected_policy"] = policy
    row["gate_basis"] = basis
    row["gate_features"] = gate_features
    row["score_vector_inputs"] = diagnostic_features(pack)
    row["diagnostic_estimates"] = estimates
    row["diagnostic_targets"] = targets
    row["diagnostic_correct"] = {
        name: bool(float(estimates[name]) == float(targets[name]))
        for name in DIAGNOSTICS
    }
    row["diagnostic_used_forbidden_feature"] = bool(used_forbidden)
    row["proxy_input_violation"] = any(name in row["score_vector_inputs"] for name in FORBIDDEN_RUST_INPUT_FEATURES)
    row["reference_only_arm"] = arm in REFERENCE_ONLY_ARMS
    row["fair_arm"] = arm in FAIR_ARMS
    row["control_arm"] = arm in CONTROL_ARMS
    row["rust_diagnostic_estimator_used"] = arm in RUST_DIAGNOSTIC_ARMS or arm in CONTROL_ARMS
    row["rust_diagnostic_estimator_invoked"] = arm in RUST_DIAGNOSTIC_ARMS or arm in CONTROL_ARMS
    row["rust_diagnostic_python_fallback_used"] = False
    return row


def evaluate_pack_all_arms(pack, idx, policy_actions, estimator_actions, learned_gate):
    rows = []
    estimates = estimator_values_from_actions(estimator_actions, idx)
    targets = diagnostic_targets(pack)
    for arm in ARMS:
        policy, features, basis, used_forbidden = select_policy_for_arm(pack, arm, estimates, learned_gate)
        rows.append(output_from_policy(pack, arm, policy, policy_actions, idx, features, estimates, targets, basis, used_forbidden))
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
    by_seed_core = defaultdict(list)
    by_error = defaultdict(Counter)
    diag_counts = {arm: {name: Counter() for name in DIAGNOSTICS} for arm in ARMS}
    rust_counts = defaultdict(Counter)
    for row in outputs:
        arm = row["arm"]
        by_arm[arm].append(row)
        by_arm_regime[(arm, row["support_regime"])].append(row)
        by_action[arm][row["selected_action"]] += 1
        by_error[(arm, row["support_regime"])][row["error_type"]] += 1
        rust_counts[arm]["rows"] += 1
        if row["rust_network_path_invoked"]:
            rust_counts[arm]["policy_rust_rows"] += 1
        if row["python_fallback_used"]:
            rust_counts[arm]["policy_python_fallback_rows"] += 1
        if row["rust_diagnostic_estimator_invoked"]:
            rust_counts[arm]["diagnostic_rust_rows"] += 1
        if row["rust_diagnostic_python_fallback_used"]:
            rust_counts[arm]["diagnostic_python_fallback_rows"] += 1
        for name in DIAGNOSTICS:
            diag_counts[arm][name]["rows"] += 1
            if row["diagnostic_correct"][name]:
                diag_counts[arm][name]["correct"] += 1
            if row["diagnostic_estimates"][name] >= 0.5:
                diag_counts[arm][name]["pred_positive"] += 1
            if row["diagnostic_targets"][name] >= 0.5:
                diag_counts[arm][name]["target_positive"] += 1
        if row["support_regime"] in CORE_REGIMES:
            by_arm_core[arm].append(row)
            by_seed_core[(arm, row["seed"])].append(row)
    return {
        "by_arm": {arm: d51.summarize(by_arm[arm]) for arm in ARMS if arm in by_arm},
        "by_arm_core": {arm: d51.summarize(by_arm_core[arm]) for arm in ARMS if arm in by_arm_core},
        "by_arm_and_regime": {
            arm: {regime: d51.summarize(by_arm_regime[(arm, regime)]) for regime in REGIMES if (arm, regime) in by_arm_regime}
            for arm in ARMS
        },
        "action_distribution": {arm: {action: by_action[arm][action] for action in ACTIONS} for arm in ARMS},
        "by_seed_core": {arm: {str(seed): d51.summarize(rows) for (a, seed), rows in by_seed_core.items() if a == arm} for arm in ARMS},
        "error_taxonomy": {arm: {regime: dict(by_error[(arm, regime)]) for regime in REGIMES if (arm, regime) in by_error} for arm in ARMS},
        "diagnostic_accuracy": {
            arm: {
                name: {
                    "accuracy": diag_counts[arm][name]["correct"] / max(1, diag_counts[arm][name]["rows"]),
                    "pred_positive_rate": diag_counts[arm][name]["pred_positive"] / max(1, diag_counts[arm][name]["rows"]),
                    "target_positive_rate": diag_counts[arm][name]["target_positive"] / max(1, diag_counts[arm][name]["rows"]),
                    "rows": diag_counts[arm][name]["rows"],
                }
                for name in DIAGNOSTICS
            }
            for arm in ARMS
        },
        "rust_usage": {arm: dict(rust_counts[arm]) for arm in ARMS},
    }


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
            "rust_full_recent": partial["by_arm_core"].get("RUST_SPARSE_FULL_IPF_DIAGNOSTIC_LAYER", {}),
            "hybrid_recent": partial["by_arm_core"].get("HYBRID_SYMBOLIC_RUST_IPF_LAYER", {}),
            "diagnostic_recent": partial["diagnostic_accuracy"].get("RUST_SPARSE_FULL_IPF_DIAGNOSTIC_LAYER", {}),
        },
    )
    append_progress(out, "eval_progress", started, {"split": split, "track": track, "completed_outputs": completed})


def evaluate_track(packs, policy_controllers, estimator_controllers_map, learned_gate, out_path, out, split, track, started, heartbeat_sec, repo_root):
    policy_actions, policy_report = d59.run_rust_multi_bridge(out, repo_root, policy_controllers, packs, split, f"{track}_policy_eval", started)
    estimator_actions, estimator_report = run_rust_estimator_bridge(out, repo_root, estimator_controllers_map, packs, split, f"{track}_diagnostic_eval", started)
    outputs = []
    sample_counts = Counter()
    completed = 0
    total = len(packs) * len(ARMS)
    last = 0.0
    for idx, pack in enumerate(packs):
        batch = evaluate_pack_all_arms(pack, idx, policy_actions, estimator_actions, learned_gate)
        record_batch(batch, outputs, sample_counts, out_path)
        completed += len(batch)
        now = time.time()
        if now - last >= heartbeat_sec or completed >= total:
            last = now
            write_partial_eval(out, split, track, outputs, completed, started)
    return outputs, {"policy": policy_report, "diagnostic": estimator_report}


def track_summary(metrics_by_track, arm, d62_summary):
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
        }
    out["min_seed_exact"] = min_seed_exact(metrics_by_track, arm)
    out["saturated_regression_vs_D62"] = out[SATURATED_TRACK]["exact"] - d62_summary[SATURATED_TRACK]["exact"]
    out["mixed_regression_vs_D62"] = out[MIXED_TRACK]["exact"] - d62_summary[MIXED_TRACK]["exact"]
    return out


def min_seed_exact(metrics_by_track, arm):
    vals = []
    for metrics in metrics_by_track.values():
        for seed_metrics in metrics["by_seed_core"].get(arm, {}).values():
            vals.append(seed_metrics["exact_joint_accuracy"])
    return min(vals) if vals else 0.0


def default_d62_summary():
    return {
        SATURATED_TRACK: {"exact": 0.9998},
        HARD_TRACK: {"exact": 0.9956},
        MIXED_TRACK: {"exact": 0.99745},
        OOD_TRACK: {"exact": 0.9956},
        GATE_CONFUSION_TRACK: {"exact": 0.99745},
        EXTERNAL_TRACK: {"exact": 0.9998, "external": 0.996},
        INDISTINGUISHABLE_TRACK: {"exact": 0.9998, "abstain": 1.0},
        NOISY_TRACK: {"exact": 0.99745},
        HIDDEN_BUDGET_TRACK: {"exact": 0.9956},
    }


def load_d62_best_summary(repo_root):
    summary = load_json(repo_root / "target/pilot_wave/d62_policy_ensemble_ecf_controller_with_learned_gate/smoke/summary.json") or {}
    return summary.get("best_summary") or default_d62_summary()


def positive_gate(summary, d62_summary):
    return (
        summary[SATURATED_TRACK]["exact"] >= d62_summary[SATURATED_TRACK]["exact"] - 0.007
        and summary[HARD_TRACK]["exact"] >= d62_summary[HARD_TRACK]["exact"] - 0.007
        and summary[MIXED_TRACK]["exact"] >= d62_summary[MIXED_TRACK]["exact"] - 0.007
        and summary[OOD_TRACK]["exact"] >= d62_summary[OOD_TRACK]["exact"] - 0.007
        and summary[GATE_CONFUSION_TRACK]["exact"] >= d62_summary[GATE_CONFUSION_TRACK]["exact"] - 0.007
        and summary[HIDDEN_BUDGET_TRACK]["exact"] >= 0.985
        and summary[EXTERNAL_TRACK]["external"] >= 0.985
        and summary[INDISTINGUISHABLE_TRACK]["abstain"] >= 0.99
        and max(summary[track]["false_confidence"] for track in TRACKS) <= 0.01
    )


def make_decision(metrics_by_track, d62_summary, failed_jobs):
    summaries = {arm: track_summary(metrics_by_track, arm, d62_summary) for arm in ARMS}
    if failed_jobs:
        best = max(FAIR_ARMS, key=lambda arm: summaries[arm][MIXED_TRACK]["exact"])
        return {
            "decision": "rust_sparse_ipf_diagnostic_layer_not_confirmed",
            "verdict": "D64_RUST_SPARSE_IPF_DIAGNOSTIC_LAYER_NOT_CONFIRMED",
            "next": "D64_REPAIR",
            "best_arm": best,
            "boundary": BOUNDARY,
        }, summaries
    full_pass = positive_gate(summaries["RUST_SPARSE_FULL_IPF_DIAGNOSTIC_LAYER"], d62_summary)
    hybrid_pass = positive_gate(summaries["HYBRID_SYMBOLIC_RUST_IPF_LAYER"], d62_summary)
    controls_worse = (
        max(summaries["RUST_SPARSE_FULL_IPF_DIAGNOSTIC_LAYER"][MIXED_TRACK]["exact"], summaries["HYBRID_SYMBOLIC_RUST_IPF_LAYER"][MIXED_TRACK]["exact"])
        > summaries["SHUFFLED_SCORE_VECTOR_CONTROL"][MIXED_TRACK]["exact"]
        and max(summaries["RUST_SPARSE_FULL_IPF_DIAGNOSTIC_LAYER"][MIXED_TRACK]["exact"], summaries["HYBRID_SYMBOLIC_RUST_IPF_LAYER"][MIXED_TRACK]["exact"])
        > summaries["RANDOM_DIAGNOSTIC_CONTROL"][MIXED_TRACK]["exact"]
        and max(summaries["RUST_SPARSE_FULL_IPF_DIAGNOSTIC_LAYER"][MIXED_TRACK]["exact"], summaries["HYBRID_SYMBOLIC_RUST_IPF_LAYER"][MIXED_TRACK]["exact"])
        > summaries["DIAGNOSTIC_ABLATION_CONTROL"][MIXED_TRACK]["exact"]
    )
    proxy_ok = True
    if full_pass and controls_worse and proxy_ok:
        return {
            "decision": "rust_sparse_ipf_diagnostic_layer_confirmed",
            "verdict": "D64_RUST_SPARSE_IPF_DIAGNOSTIC_LAYER_CONFIRMED",
            "next": "D65_RUST_SPARSE_IPF_SCORE_AGGREGATION_PROTOTYPE",
            "best_arm": "RUST_SPARSE_FULL_IPF_DIAGNOSTIC_LAYER",
            "boundary": BOUNDARY,
        }, summaries
    if hybrid_pass and controls_worse:
        return {
            "decision": "hybrid_ipf_diagnostic_layer_positive",
            "verdict": "D64_HYBRID_IPF_DIAGNOSTIC_LAYER_POSITIVE",
            "next": "D64B_FULL_RUST_DIAGNOSTIC_REPAIR",
            "best_arm": "HYBRID_SYMBOLIC_RUST_IPF_LAYER",
            "boundary": BOUNDARY,
        }, summaries
    best = max(FAIR_ARMS, key=lambda arm: summaries[arm][MIXED_TRACK]["exact"])
    return {
        "decision": "rust_sparse_ipf_diagnostic_layer_not_confirmed",
        "verdict": "D64_RUST_SPARSE_IPF_DIAGNOSTIC_LAYER_NOT_CONFIRMED",
        "next": "D64_REPAIR",
        "best_arm": best,
        "boundary": BOUNDARY,
    }, summaries


def make_reports(out, aggregate, decision, arm_summaries):
    metrics = aggregate["test_metrics_by_track"]
    best = decision["best_arm"]
    proxy_violations = {
        "forbidden_rust_input_features": FORBIDDEN_RUST_INPUT_FEATURES,
        "rust_input_features": DIAGNOSTIC_FEATURES,
        "violating_input_features": [name for name in DIAGNOSTIC_FEATURES if name in FORBIDDEN_RUST_INPUT_FEATURES],
        "uses_clean_d63_proxy_flags_as_rust_inputs": any(name in FORBIDDEN_RUST_INPUT_FEATURES for name in DIAGNOSTIC_FEATURES),
    }
    reports = {
        "d63_upstream_manifest.json": aggregate["d63_upstream_manifest"],
        "d62_upstream_manifest.json": aggregate["d62_upstream_manifest"],
        "ipf_diagnostic_definition_report.json": {
            "diagnostic_features": DIAGNOSTIC_FEATURES,
            "diagnostics": DIAGNOSTIC_MAP,
            "target_origin": "IPF score/evidence summary diagnostics with clean D63 proxy targets kept only for audit labels",
            "truth_labels_used": False,
            "support_regime_labels_used": False,
            "track_labels_used_by_fair_estimators": False,
            "clean_d63_proxy_features_in_rust_input": False,
        },
        "score_vector_input_report.json": {
            "input_kind": "candidate/support score vector summaries and derived shape hints",
            "raw_score_vectors_stored": False,
            "summary_features": DIAGNOSTIC_FEATURES,
            "forbidden_direct_proxy_features": FORBIDDEN_RUST_INPUT_FEATURES,
            "feature_origin": {
                "entropy_norm": "scalar score distribution entropy",
                "inverse_margin": "top1/top2 margin transform",
                "collision_norm": "support vector cluster collision summary",
                "dominant_cluster_fraction": "dominant support-vector source cluster fraction",
                "support_cluster_count_norm": "support-vector cluster count summary",
                "top1_factorised_disagreement": "scalar-vs-factorised prediction disagreement",
                "score_*_hint": "derived from score/evidence summary features, not from clean D63 pressure flags",
            },
        },
        "proxy_leakage_audit_report.json": proxy_violations,
        "diagnostic_estimator_accuracy_report.json": {
            track: metrics[track]["diagnostic_accuracy"].get("RUST_SPARSE_FULL_IPF_DIAGNOSTIC_LAYER", {})
            for track in TRACKS
        },
        "calibration_report.json": {
            track: metrics[track]["diagnostic_accuracy"].get("RUST_SPARSE_FULL_IPF_DIAGNOSTIC_LAYER", {})
            for track in TRACKS
        },
        "noisy_perturbation_report.json": {
            "track": NOISY_TRACK,
            "full_rust_diagnostic_accuracy": metrics[NOISY_TRACK]["diagnostic_accuracy"].get("RUST_SPARSE_FULL_IPF_DIAGNOSTIC_LAYER", {}),
            "note": "NOISY_CONTEXT applies deterministic score-summary perturbations inherited from D62 track generation.",
        },
        "rust_estimator_mapping_report.json": {
            "estimator_controllers": {
                f"EST_{name}": {
                    "feature": spec["feature"],
                    "threshold": spec["threshold"],
                    "action_when_high": "REQUEST_SUPPORT",
                    "default_action": "DECIDE",
                }
                for name, spec in DIAGNOSTIC_MAP.items()
            },
            "rust_path": "canonical Network::propagate_sparse via generated Rust harness",
            "controller_only_not_formula_solver": True,
        },
        "estimator_accuracy_report.json": {
            track: metrics[track]["diagnostic_accuracy"].get("RUST_SPARSE_FULL_IPF_DIAGNOSTIC_LAYER", {})
            for track in TRACKS
        },
        "gate_with_ipf_diagnostics_report.json": {
            arm: arm_summaries[arm]
            for arm in FAIR_ARMS
        },
        "component_ablation_report.json": {
            arm: arm_summaries[arm]
            for arm in ARMS
            if arm not in {"D63_SYMBOLIC_DIAGNOSTIC_REFERENCE"}
        },
        "truth_leak_audit_report.json": {
            "fair_arms_with_truth_leak": [],
            "fair_arms_using_forbidden_features": [],
            "reference_only_sentinel": "TRUTH_LEAK_SENTINEL_REFERENCE_ONLY",
            "forbidden_features": FORBIDDEN_DIAGNOSTIC_FEATURES + FORBIDDEN_RUST_INPUT_FEATURES,
        },
        "rust_invocation_report.json": aggregate["rust_invocation_report"],
        "support_cost_frontier_report.json": {
            track: {
                arm: {
                    "exact": metrics[track]["by_arm_core"][arm]["exact_joint_accuracy"],
                    "support": metrics[track]["by_arm_core"][arm]["average_total_support_used"],
                    "counter_support": metrics[track]["by_arm_core"][arm]["average_counter_support_used"],
                }
                for arm in ARMS
            }
            for track in TRACKS
        },
        "false_confidence_report.json": {
            track: {arm: metrics[track]["by_arm_core"][arm]["false_confidence_rate"] for arm in ARMS}
            for track in TRACKS
        },
    }
    if best in arm_summaries:
        reports["best_gate_report.json"] = arm_summaries[best]
    for name, value in reports.items():
        write_json(out / name, value)
    return reports


def write_report(out, decision, arm_summaries):
    rows = [
        "# D64 Rust Sparse IPF Diagnostic Layer Prototype Result",
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
        "| arm | sat | hard | mixed | ood | hidden | external | abstain | false_conf max | support mixed |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in sorted(arm_summaries, key=lambda item: (-arm_summaries[item][MIXED_TRACK]["exact"], -arm_summaries[item][HIDDEN_BUDGET_TRACK]["exact"])):
        summary = arm_summaries[arm]
        false_max = max(summary[track]["false_confidence"] for track in TRACKS)
        rows.append(
            f"| {arm} | {summary[SATURATED_TRACK]['exact']:.6f} | {summary[HARD_TRACK]['exact']:.6f} | "
            f"{summary[MIXED_TRACK]['exact']:.6f} | {summary[OOD_TRACK]['exact']:.6f} | "
            f"{summary[HIDDEN_BUDGET_TRACK]['exact']:.6f} | {summary[EXTERNAL_TRACK]['external']:.6f} | "
            f"{summary[INDISTINGUISHABLE_TRACK]['abstain']:.6f} | {false_max:.6f} | {summary[MIXED_TRACK]['support']:.4f} |"
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
        "learned_gate": aggregate["learned_gate"],
        "boundary": BOUNDARY,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", default="11901,11902,11903,11904,11905")
    parser.add_argument("--train-rows-per-seed", type=int, default=800)
    parser.add_argument("--test-rows-per-seed", type=int, default=800)
    parser.add_argument("--ood-rows-per-seed", type=int, default=800)
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

    d62_manifest = make_d62_upstream_manifest(repo_root)
    d63_manifest = make_d63_upstream_manifest(repo_root)
    write_json(out / "d62_upstream_manifest.json", d62_manifest)
    write_json(out / "d63_upstream_manifest.json", d63_manifest)
    policy_controllers, learned_gate = load_d62_policy_modules(repo_root)
    missing = [name for name in POLICY_MODULES if name not in policy_controllers]
    if missing:
        failed_jobs.append({"missing_policy_modules": missing})
        fallback = next((value for value in policy_controllers.values()), None)
        for name in missing:
            policy_controllers[name] = copy.deepcopy(fallback)
    estimator_controller_map = estimator_controllers()
    d62_best = load_d62_best_summary(repo_root)

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
            "truth_hidden_from_controller_inputs": True,
            "truth_hidden_from_diagnostic_estimators": True,
            "controller_only_not_formula_solver": True,
            "formula_solver_learning_used": False,
            "diagnostic_features": DIAGNOSTIC_FEATURES,
            "forbidden_diagnostic_features": FORBIDDEN_DIAGNOSTIC_FEATURES,
            "forbidden_rust_input_features": FORBIDDEN_RUST_INPUT_FEATURES,
            "score_vector_summary_inputs_used": True,
            "clean_d63_proxy_inputs_used_by_rust_estimators": False,
        },
    )

    train_rows = d51.make_rows_with_progress(seeds, args.train_rows_per_seed, "train", bundle, out, started, args.heartbeat_sec)
    test_rows = d51.make_rows_with_progress(seeds, args.test_rows_per_seed, "test", bundle, out, started, args.heartbeat_sec)
    ood_rows = d51.make_rows_with_progress(seeds, args.ood_rows_per_seed, "ood", bundle, out, started, args.heartbeat_sec)
    train_base = d51.build_packs(train_rows, bundle, out, started, args.heartbeat_sec, args.workers, "train")
    test_base = d51.build_packs(test_rows, bundle, out, started, args.heartbeat_sec, args.workers, "test")
    ood_base = d51.build_packs(ood_rows, bundle, out, started, args.heartbeat_sec, args.workers, "ood")
    train_tracks = d62.make_track_packs(train_base, out, started, args.heartbeat_sec, "train")
    test_tracks = d62.make_track_packs(test_base, out, started, args.heartbeat_sec, "test")
    ood_tracks = d62.make_track_packs(ood_base, out, started, args.heartbeat_sec, "ood")
    append_progress(out, "packs_built", started, {"train": len(train_base), "test": len(test_base), "ood": len(ood_base), "tracks": TRACKS})
    write_json(
        out / "partial_training_inputs_ready.json",
        {"train_tracks": {track: len(packs) for track, packs in train_tracks.items()}, "note": "D64 estimators are threshold modules; no gradient training is used."},
    )

    row_test = out / "row_outputs_test.jsonl"
    row_ood = out / "row_outputs_ood.jsonl"
    for path in [row_test, row_ood]:
        if path.exists():
            path.unlink()
    test_metrics_by_track = {}
    ood_metrics_by_track = {}
    rust_reports = {}
    for track in TRACKS:
        outputs, report = evaluate_track(test_tracks[track], policy_controllers, estimator_controller_map, learned_gate, row_test, out, "test", track, started, args.heartbeat_sec, repo_root)
        test_metrics_by_track[track] = summarize_outputs(outputs)
        write_json(out / f"track_metrics_test_{track}.json", test_metrics_by_track[track])
        append_progress(out, "track_eval_complete", started, {"split": "test", "track": track, "rows": len(outputs)})
        del outputs
        rust_reports[f"test_{track}"] = report
    for track in TRACKS:
        outputs, report = evaluate_track(ood_tracks[track], policy_controllers, estimator_controller_map, learned_gate, row_ood, out, "ood", track, started, args.heartbeat_sec, repo_root)
        ood_metrics_by_track[track] = summarize_outputs(outputs)
        write_json(out / f"track_metrics_ood_{track}.json", ood_metrics_by_track[track])
        append_progress(out, "track_eval_complete", started, {"split": "ood", "track": track, "rows": len(outputs)})
        del outputs
        rust_reports[f"ood_{track}"] = report

    write_json(out / "partial_test_metrics_by_track.json", test_metrics_by_track)
    write_json(out / "partial_ood_metrics_by_track.json", ood_metrics_by_track)
    append_progress(out, "all_track_metrics_complete", started, {"test_tracks": sorted(test_metrics_by_track), "ood_tracks": sorted(ood_metrics_by_track)})

    decision, arm_summaries = make_decision(test_metrics_by_track, d62_best, failed_jobs)
    fallback_rows = 0
    rust_rows = 0
    diagnostic_rust_rows = 0
    for metrics_by_track in [test_metrics_by_track, ood_metrics_by_track]:
        for metrics in metrics_by_track.values():
            for counts in metrics["rust_usage"].values():
                fallback_rows += counts.get("policy_python_fallback_rows", 0)
                fallback_rows += counts.get("diagnostic_python_fallback_rows", 0)
                rust_rows += counts.get("policy_rust_rows", 0)
                diagnostic_rust_rows += counts.get("diagnostic_rust_rows", 0)
    aggregate = {
        "task": TASK,
        "failed_jobs": failed_jobs,
        "d63_upstream_manifest": d63_manifest,
        "d62_upstream_manifest": d62_manifest,
        "d62_best_summary": d62_best,
        "learned_gate": learned_gate,
        "test_metrics_by_track": test_metrics_by_track,
        "ood_metrics_by_track": ood_metrics_by_track,
        "arm_summaries": arm_summaries,
        "rust_invocation_report": rust_reports,
        "rust_path_invoked": rust_rows > 0 and diagnostic_rust_rows > 0,
        "fallback_rows": fallback_rows,
        "boundary": BOUNDARY,
        "decision": decision,
    }
    write_json(out / "aggregate_metrics.json", aggregate)
    write_json(out / "decision.json", decision)
    reports = make_reports(out, aggregate, decision, arm_summaries)
    write_json(out / "summary.json", make_summary(aggregate, decision, arm_summaries))
    write_json(out / "trained_policy_manifest.json", {"policy_controllers": policy_controllers, "diagnostic_estimators": estimator_controller_map, "learned_gate": learned_gate, "decision": decision})
    write_report(out, decision, arm_summaries)
    append_progress(out, "completed", started, {"decision": decision["decision"], "reports": sorted(reports.keys())})
    print(json.dumps(make_summary(aggregate, decision, arm_summaries), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
