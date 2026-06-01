#!/usr/bin/env python3
"""D51 mutable ECF controller prototype.

This probe keeps the D50 symbolic joint formula discovery solver fixed and
learns only the controller policy that chooses whether to decide, ask for more
support, ask for targeted counter-support, request an external test, or abstain.
"""

import argparse
import copy
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import run_d49_joint_cell_operator_discovery_with_robust_support as d49
import run_d49b_joint_binding_repair as d49b

PRIMARY_SPACE = "ALL28_UNORDERED_X_OPS"
SUPPORT_COUNT = 5
ROW_SAMPLE_PER_ARM_REGIME_SPLIT = 24
CONFIDENCE_THRESHOLD = 0.45

BOUNDARY = (
    "D51 only tests mutable control policy for controlled symbolic joint formula discovery. "
    "It does not prove raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, "
    "or architecture superiority."
)

REGIMES = d49b.REGIMES
CORE_REGIMES = d49b.CORE_REGIMES
OP_NAMES = d49b.OP_NAMES

ACTIONS = [
    "DECIDE",
    "REQUEST_SUPPORT",
    "REQUEST_COUNTER_TOP1_TOP2",
    "REQUEST_JOINT_COUNTER",
    "REQUEST_EXTERNAL_TEST",
    "ABSTAIN",
]

REFERENCE_ARMS = [
    "HANDCODED_D50_FULL_REPAIRED_ECF_REFERENCE",
    "HANDCODED_CAP_7_REFERENCE",
    "HANDCODED_CAP_9_REFERENCE",
]

CONTROL_ARMS = [
    "RANDOM_POLICY_CONTROL",
    "GREEDY_DECIDE_CONTROL",
    "ALWAYS_COUNTER_SUPPORT_CONTROL",
]

MUTABLE_ARMS = [
    "MUTABLE_LINEAR_CONTROLLER",
    "MUTABLE_RULE_TABLE_CONTROLLER",
    "MUTABLE_SMALL_TREE_CONTROLLER",
    "MUTABLE_HYBRID_CONTROLLER",
]

ARMS = REFERENCE_ARMS + CONTROL_ARMS + MUTABLE_ARMS

FEATURE_NAMES = [
    "bias",
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
    "internal_unresolvable_indicator",
    "external_channel_available",
]

GLOBAL_BUNDLE = None


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def append_progress(out, event, started, data):
    append_jsonl(
        out / "progress.jsonl",
        {
            "time_unix_ms": int(time.time() * 1000),
            "elapsed_sec": time.time() - started,
            "event": event,
            "data": data,
        },
    )


def mean(values):
    return sum(values) / len(values) if values else 0.0


def stable_seed(text):
    total = 0
    for idx, ch in enumerate(str(text)):
        total += (idx + 1) * ord(ch)
    return total & 0x7FFFFFFF


def safe_div(num, den):
    return num / den if den else 0.0


def cell_key(cell):
    return d49.cell_key(cell)


def support_cap_for_reference(arm):
    if arm == "HANDCODED_CAP_7_REFERENCE":
        return 7
    if arm == "HANDCODED_CAP_9_REFERENCE":
        return 9
    return 12


def add_stage(extra, stage_counts, stage, vectors, cap, base_count):
    room = max(0, cap - base_count - len(extra)) if cap is not None else len(vectors)
    chosen = vectors[:room]
    extra.extend(chosen)
    stage_counts[stage] += len(chosen)


def base_state(row, bundle):
    base_vectors = d49b.cached_base_vectors(row, bundle, SUPPORT_COUNT)
    scalar_scores = d49b.aggregate_sum(base_vectors)
    scalar_pred = d49b.predict(scalar_scores, bundle)
    return base_vectors, scalar_scores, scalar_pred


def build_features(row, bundle, base_vectors, scalar_scores, scalar_pred):
    cluster_count, dominant_fraction, collision_count = d49b.cluster_stats(base_vectors)
    pair_scores = d49.project_scores(scalar_scores, bundle, "pair")
    op_scores = d49.project_scores(scalar_scores, bundle, "operator")
    factor_scores = d49.factorised_joint_scores(scalar_scores, bundle)
    factor_pred = d49.predict(factor_scores, bundle)
    max_entropy = math.log(max(2, len(bundle["candidates"])))
    margin = scalar_pred["top1_top2_margin"]
    values = {
        "bias": 1.0,
        "scalar_confidence": scalar_pred["confidence"],
        "inverse_margin": 1.0 / (1.0 + max(0.0, margin)),
        "entropy_norm": min(1.0, scalar_pred["entropy"] / max_entropy),
        "collision_norm": min(1.0, collision_count / float(SUPPORT_COUNT)),
        "dominant_cluster_fraction": dominant_fraction,
        "support_cluster_count_norm": min(1.0, cluster_count / float(SUPPORT_COUNT)),
        "top1_factorised_disagreement": 1.0 if factor_pred["pred_joint"] != scalar_pred["pred_joint"] else 0.0,
        "cell_confidence": max(pair_scores.values()) if pair_scores else 0.0,
        "operator_confidence": max(op_scores.values()) if op_scores else 0.0,
        "joint_confidence": scalar_pred["confidence"],
        # These are channel diagnostics: they say whether the available internal
        # counter-support channel is tied, and whether an external test channel exists.
        # They do not expose the true cell pair or true operator.
        "internal_unresolvable_indicator": 1.0 if row.get("internal_counter_supports") else 0.0,
        "external_channel_available": 1.0 if row.get("external_counter_supports") else 0.0,
    }
    return [float(values[name]) for name in FEATURE_NAMES], values


def classify_error(row, pred, abstained, exact_joint, pair_equiv, op_exact):
    return d49b.classify_error(row, pred, abstained, exact_joint, pair_equiv, op_exact)


def build_result(row, bundle, arm, action, pred, final_scores, scalar_pred, base_vectors, stage_counts, external_used, abstained, unavailable_external):
    exact_joint = pred["pred_joint"] == row["truth_joint"]
    pair_ok = pred["pred_pair_equivalence"] == row["truth_pair_equivalence"]
    pred_cells = set(d49.canonical_pair(pred["pred_pair"])) if pred["pred_pair"] else set()
    true_cells = set(d49.canonical_pair(row["truth_pair"]))
    cell_hit = len(pred_cells & true_cells) / 2.0 if pred_cells else 0.0
    op_exact = pred["pred_operator"] == row["true_operator"]
    op_equiv = pred["pred_operator_equivalence"] == row["truth_operator_equivalence"]
    group_correct = pred["pred_group"] == row["truth_group"]
    false_conf = (not exact_joint) and (not abstained) and pred["confidence"] >= CONFIDENCE_THRESHOLD
    cluster_count, dominant_fraction, collision_count = d49b.cluster_stats(base_vectors)
    taxonomy = classify_error(row, pred, abstended := abstained, exact_joint, pair_ok, op_exact)
    return {
        "row_id": row["row_id"],
        "seed": row["seed"],
        "split": row["split"],
        "arm": arm,
        "selected_action": action,
        "primitive_space": PRIMARY_SPACE,
        "support_regime": row["support_regime"],
        "truth_joint": row["truth_joint"],
        "pred_joint": pred["pred_joint"],
        "truth_pair": [cell_key(cell) for cell in row["truth_pair"]],
        "pred_pair": [cell_key(cell) for cell in pred["pred_pair"]] if pred["pred_pair"] else [],
        "truth_pair_equivalence": row["truth_pair_equivalence"],
        "pred_pair_equivalence": pred["pred_pair_equivalence"],
        "truth_operator": row["true_operator"],
        "pred_operator": pred["pred_operator"],
        "truth_operator_equivalence": row["truth_operator_equivalence"],
        "pred_operator_equivalence": pred["pred_operator_equivalence"],
        "exact_joint_correct": exact_joint,
        "cell_pair_equivalence_correct": pair_ok,
        "cell_hit_top2": cell_hit,
        "cell_hit_top2_correct": cell_hit >= 1.0,
        "operator_exact_correct": op_exact,
        "operator_equivalence_correct": op_equiv,
        "family_group_correct": group_correct,
        "joint_binding_consistency": bool(exact_joint or (pair_ok and op_exact)),
        "correct": exact_joint,
        "original_support_used": SUPPORT_COUNT,
        "cell_counter_support_used": stage_counts["cell"],
        "operator_counter_support_used": stage_counts["operator"],
        "joint_counter_support_used": stage_counts["joint"],
        "random_counter_support_used": stage_counts["random"],
        "counter_support_used": sum(stage_counts.values()) - external_used,
        "external_test_used": external_used,
        "external_test_requested_unavailable": unavailable_external,
        "total_support_used": SUPPORT_COUNT + sum(stage_counts.values()),
        "support_cluster_count": cluster_count,
        "dominant_cluster_fraction": dominant_fraction,
        "collision_count": collision_count,
        "correlated_echo_detected": dominant_fraction >= 0.60 and len(base_vectors) >= 3,
        "abstained": abstended,
        "false_confidence": false_conf,
        "confidence": pred["confidence"],
        "top1_top2_margin": pred["top1_top2_margin"],
        "entropy": pred["entropy"],
        "baseline_exact_correct": scalar_pred["pred_joint"] == row["truth_joint"],
        "score_gap_truth_vs_wrong": d49b.score_gap(final_scores, row["truth_joint"]) if not abstained else 0.0,
        "error_type": taxonomy,
    }


def run_action(row, bundle, action, base=None, arm_name=None):
    if base is None:
        base_vectors, scalar_scores, scalar_pred = base_state(row, bundle)
    else:
        base_vectors, scalar_scores, scalar_pred = base
    extra = []
    stage_counts = Counter()
    external_used = 0
    abstained = False
    unavailable_external = False
    cap = 12

    if action == "DECIDE":
        final_scores = scalar_scores
        pred = d49b.predict(final_scores, bundle)
    elif action == "REQUEST_SUPPORT":
        vectors = d49.random_extra_vectors(row, bundle)
        add_stage(extra, stage_counts, "random", vectors, cap, SUPPORT_COUNT)
        final_scores = d49b.aggregate_duplicate_downweighted(base_vectors + extra)
        pred = d49b.predict(final_scores, bundle)
    elif action == "REQUEST_COUNTER_TOP1_TOP2":
        vectors = d49b.make_stage_vectors(row, bundle, scalar_pred, "joint", 3)
        add_stage(extra, stage_counts, "joint", vectors, cap, SUPPORT_COUNT)
        final_scores = d49b.aggregate_duplicate_downweighted(base_vectors + extra)
        pred = d49b.predict(final_scores, bundle)
    elif action == "REQUEST_JOINT_COUNTER":
        for stage, count in [("cell", 1), ("operator", 1), ("joint", 4)]:
            vectors = d49b.make_stage_vectors(row, bundle, scalar_pred, stage, count)
            add_stage(extra, stage_counts, stage, vectors, cap, SUPPORT_COUNT)
        final_scores = d49b.aggregate_duplicate_downweighted(base_vectors + extra)
        pred = d49b.predict(final_scores, bundle)
    elif action == "REQUEST_EXTERNAL_TEST":
        if row.get("external_test_available"):
            vectors = d49b.make_stage_vectors(row, bundle, scalar_pred, "joint", 4, external=True)
            add_stage(extra, stage_counts, "external", vectors, cap, SUPPORT_COUNT)
            external_used = stage_counts["external"]
        else:
            unavailable_external = True
        final_scores = d49b.aggregate_duplicate_downweighted(base_vectors + extra)
        pred = d49b.predict(final_scores, bundle)
    elif action == "ABSTAIN":
        abstained = True
        final_scores = scalar_scores
        pred = d49b.predict(final_scores, bundle, abstain=True)
    else:
        raise ValueError(action)

    return build_result(
        row,
        bundle,
        arm_name or action,
        action,
        pred,
        final_scores,
        scalar_pred,
        base_vectors,
        stage_counts,
        external_used,
        abstained,
        unavailable_external,
    )


def run_reference(row, bundle, arm, base=None):
    if base is None:
        base_vectors, scalar_scores, scalar_pred = base_state(row, bundle)
    else:
        base_vectors, scalar_scores, scalar_pred = base
    cap = support_cap_for_reference(arm)
    extra = []
    stage_counts = Counter()
    external_used = 0
    abstained = False
    unavailable_external = False
    need_counter = d49b.counter_needed(row, scalar_pred, base_vectors)
    if row.get("external_test_available"):
        vectors = d49b.make_stage_vectors(row, bundle, scalar_pred, "joint", 4, external=True)
        add_stage(extra, stage_counts, "external", vectors, cap, SUPPORT_COUNT)
        external_used = stage_counts["external"]
    elif not row.get("oracle_distinguishable", True):
        abstained = True
    elif need_counter:
        for stage, count in [("cell", 1), ("operator", 1), ("joint", 4)]:
            vectors = d49b.make_stage_vectors(row, bundle, scalar_pred, stage, count)
            add_stage(extra, stage_counts, stage, vectors, cap, SUPPORT_COUNT)
    final_scores = d49b.aggregate_duplicate_downweighted(base_vectors + extra)
    pred = d49b.predict(final_scores, bundle, abstain=abstained)
    return build_result(
        row,
        bundle,
        arm,
        "HANDCODED_REFERENCE",
        pred,
        final_scores,
        scalar_pred,
        base_vectors,
        stage_counts,
        external_used,
        abstained,
        unavailable_external,
    )


def compact_outcome(result):
    return {
        "correct": result["correct"],
        "abstained": result["abstained"],
        "false_confidence": result["false_confidence"],
        "total_support_used": result["total_support_used"],
        "counter_support_used": result["counter_support_used"],
        "external_test_used": result["external_test_used"],
        "external_test_requested_unavailable": result["external_test_requested_unavailable"],
        "support_regime": result["support_regime"],
    }


def init_worker(bundle):
    global GLOBAL_BUNDLE
    GLOBAL_BUNDLE = bundle


def build_case_pack(row):
    bundle = GLOBAL_BUNDLE
    base = base_state(row, bundle)
    features, feature_map = build_features(row, bundle, *base)
    actions = {action: run_action(row, bundle, action, base=base, arm_name=f"ACTION_{action}") for action in ACTIONS}
    references = {arm: run_reference(row, bundle, arm, base=base) for arm in REFERENCE_ARMS}
    return {
        "row_id": row["row_id"],
        "seed": row["seed"],
        "split": row["split"],
        "support_regime": row["support_regime"],
        "features": features,
        "feature_map": feature_map,
        "actions": actions,
        "action_compact": {action: compact_outcome(result) for action, result in actions.items()},
        "references": references,
        "indistinguishable_case": row["support_regime"] == "INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT",
        "external_case": row["support_regime"] == "EXTERNAL_TEST_REQUIRED_SUPPORT",
    }


def worker_count_from_arg(workers):
    if str(workers).lower() == "auto":
        return max(1, min(6, (os.cpu_count() or 2) - 2))
    try:
        return max(1, int(workers))
    except ValueError:
        return 1


def make_rows_with_progress(seeds, rows_per_seed, split, bundle, out, started, heartbeat_sec):
    rows = []
    total = len(seeds) * len(REGIMES) * rows_per_seed
    last = time.time()
    completed = 0
    for seed in seeds:
        for regime in REGIMES:
            rng = random.Random(seed + (0 if split == "test" else 100_000) + 1_049 * REGIMES.index(regime))
            for idx in range(rows_per_seed):
                rows.append(d49.make_case(rng, seed, split, regime, idx, bundle))
                completed += 1
                now = time.time()
                if now - last >= heartbeat_sec:
                    last = now
                    append_progress(out, "row_generation_progress", started, {"split": split, "completed_rows": completed, "total_rows": total})
                    write_json(out / f"partial_{split}_row_generation.json", {"split": split, "completed_rows": completed, "total_rows": total})
    append_progress(out, "row_generation_complete", started, {"split": split, "completed_rows": completed})
    return rows


def build_packs(rows, bundle, out, started, heartbeat_sec, workers, split):
    worker_count = worker_count_from_arg(workers)
    packs = []
    total = len(rows)
    completed = 0
    last = 0.0
    if worker_count <= 1 or len(rows) < 500:
        init_worker(bundle)
        for row in rows:
            packs.append(build_case_pack(row))
            completed += 1
            now = time.time()
            if now - last >= heartbeat_sec or completed >= total:
                last = now
                write_json(out / f"partial_{split}_pack_build.json", {"split": split, "completed_rows": completed, "total_rows": total})
                append_progress(out, "pack_build_progress", started, {"split": split, "completed_rows": completed, "total_rows": total})
    else:
        append_progress(out, "parallel_pack_build_started", started, {"split": split, "workers": worker_count})
        with ProcessPoolExecutor(max_workers=worker_count, initializer=init_worker, initargs=(bundle,)) as pool:
            futures = [pool.submit(build_case_pack, row) for row in rows]
            for future in as_completed(futures):
                packs.append(future.result())
                completed += 1
                now = time.time()
                if now - last >= heartbeat_sec or completed >= total:
                    last = now
                    write_json(out / f"partial_{split}_pack_build.json", {"split": split, "completed_rows": completed, "total_rows": total})
                    append_progress(out, "pack_build_progress", started, {"split": split, "completed_rows": completed, "total_rows": total})
    packs.sort(key=lambda item: item["row_id"])
    return packs


def action_fitness(outcome):
    regime = outcome["support_regime"]
    if regime in {
        "CORRELATED_ECHO_SUPPORT",
        "ADVERSARIAL_DISTRACTOR_SUPPORT",
        "DISTINGUISHABLE_CORRELATED_FALSE_SUPPORT",
        "EXTERNAL_TEST_REQUIRED_SUPPORT",
    }:
        weight = 2.4
    elif regime in {"MIXED_CLEAN_AND_CORRELATED", "MIXED_CLEAN_AND_ADVERSARIAL"}:
        weight = 1.5
    elif regime == "INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT":
        weight = 2.0
    else:
        weight = 1.0
    score = weight if outcome["correct"] else 0.0
    if outcome["support_regime"] == "INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT" and outcome["abstained"]:
        score += 0.95 * weight
    if outcome["abstained"] and outcome["support_regime"] != "INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT":
        score -= 0.90 * weight
    if (not outcome["correct"]) and regime in {
        "CORRELATED_ECHO_SUPPORT",
        "ADVERSARIAL_DISTRACTOR_SUPPORT",
        "DISTINGUISHABLE_CORRELATED_FALSE_SUPPORT",
        "EXTERNAL_TEST_REQUIRED_SUPPORT",
    }:
        score -= 0.35 * weight
    score -= 0.0012 * max(0.0, outcome["total_support_used"] - SUPPORT_COUNT)
    score -= 0.0018 * outcome["counter_support_used"]
    score -= 0.0200 * outcome["external_test_used"]
    if outcome["external_test_requested_unavailable"]:
        score -= 0.20
    if outcome["false_confidence"]:
        score -= 1.75
    return score


def make_policy(kind, rng):
    if kind == "linear":
        return {
            "kind": kind,
            "weights": [[rng.uniform(-0.7, 0.7) for _ in FEATURE_NAMES] for _ in ACTIONS],
            "bias": [rng.uniform(-0.4, 0.4) for _ in ACTIONS],
        }
    if kind == "rule_table":
        return {
            "kind": kind,
            "margin_threshold": rng.uniform(0.25, 1.25),
            "dominant_threshold": rng.uniform(0.45, 0.85),
            "entropy_threshold": rng.uniform(0.30, 0.90),
            "confidence_threshold": rng.uniform(0.02, 0.20),
            "action_external": rng.choice(ACTIONS),
            "action_unresolvable": rng.choice(ACTIONS),
            "action_dominant": rng.choice(ACTIONS),
            "action_low_margin": rng.choice(ACTIONS),
            "action_uncertain": rng.choice(ACTIONS),
            "action_default": rng.choice(ACTIONS),
        }
    if kind == "small_tree":
        return {
            "kind": kind,
            "nodes": [
                {
                    "feature_idx": rng.randrange(len(FEATURE_NAMES)),
                    "threshold": rng.uniform(0.0, 1.0),
                    "action_if_le": rng.choice(ACTIONS),
                }
                for _ in range(5)
            ],
            "default_action": rng.choice(ACTIONS),
        }
    if kind == "hybrid":
        return {
            "kind": kind,
            "external_threshold": rng.uniform(0.30, 0.80),
            "unresolvable_threshold": rng.uniform(0.30, 0.80),
            "dominant_threshold": rng.uniform(0.45, 0.85),
            "uncertainty_threshold": rng.uniform(0.25, 0.90),
            "action_external": rng.choice(ACTIONS),
            "action_unresolvable": rng.choice(ACTIONS),
            "action_dominant": rng.choice(ACTIONS),
            "action_uncertain": rng.choice(ACTIONS),
            "linear": make_policy("linear", rng),
        }
    raise ValueError(kind)


def learned_bootstrap_policy(kind, examples):
    bins = {
        "external": [],
        "unresolvable": [],
        "dominant": [],
        "uncertain": [],
        "default": [],
    }
    idx = {name: FEATURE_NAMES.index(name) for name in FEATURE_NAMES}
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

    def best_action(subset, fallback):
        if not subset:
            return fallback
        scores = {}
        for action in ACTIONS:
            scores[action] = mean([action_fitness(ex["action_compact"][action]) for ex in subset])
        return max(scores.items(), key=lambda item: (item[1], -ACTIONS.index(item[0])))[0]

    if kind == "rule_table":
        return {
            "kind": "rule_table",
            "margin_threshold": 0.55,
            "dominant_threshold": 0.55,
            "entropy_threshold": 0.65,
            "confidence_threshold": 0.08,
            "action_external": best_action(bins["external"], "REQUEST_EXTERNAL_TEST"),
            "action_unresolvable": best_action(bins["unresolvable"], "ABSTAIN"),
            "action_dominant": best_action(bins["dominant"], "REQUEST_JOINT_COUNTER"),
            "action_low_margin": best_action(bins["uncertain"], "REQUEST_COUNTER_TOP1_TOP2"),
            "action_uncertain": best_action(bins["uncertain"], "REQUEST_COUNTER_TOP1_TOP2"),
            "action_default": best_action(bins["default"], "DECIDE"),
        }
    if kind == "hybrid":
        return {
            "kind": "hybrid",
            "external_threshold": 0.5,
            "unresolvable_threshold": 0.5,
            "dominant_threshold": 0.55,
            "uncertainty_threshold": 0.55,
            "action_external": best_action(bins["external"], "REQUEST_EXTERNAL_TEST"),
            "action_unresolvable": best_action(bins["unresolvable"], "ABSTAIN"),
            "action_dominant": best_action(bins["dominant"], "REQUEST_JOINT_COUNTER"),
            "action_uncertain": best_action(bins["uncertain"], "REQUEST_COUNTER_TOP1_TOP2"),
            "linear": make_policy("linear", random.Random(51_777)),
        }
    return None


def mutate_policy(policy, rng, rate=0.12):
    out = copy.deepcopy(policy)
    kind = out["kind"]
    if kind == "linear":
        for row in out["weights"]:
            for idx in range(len(row)):
                if rng.random() < rate:
                    row[idx] += rng.gauss(0.0, 0.20)
        for idx in range(len(out["bias"])):
            if rng.random() < rate:
                out["bias"][idx] += rng.gauss(0.0, 0.15)
    elif kind == "rule_table":
        for key in ["margin_threshold", "dominant_threshold", "entropy_threshold", "confidence_threshold"]:
            if rng.random() < rate:
                out[key] = max(0.0, min(1.5, out[key] + rng.gauss(0.0, 0.12)))
        for key in ["action_external", "action_unresolvable", "action_dominant", "action_low_margin", "action_uncertain", "action_default"]:
            if rng.random() < rate:
                out[key] = rng.choice(ACTIONS)
    elif kind == "small_tree":
        for node in out["nodes"]:
            if rng.random() < rate:
                node["feature_idx"] = rng.randrange(len(FEATURE_NAMES))
            if rng.random() < rate:
                node["threshold"] = max(0.0, min(1.0, node["threshold"] + rng.gauss(0.0, 0.15)))
            if rng.random() < rate:
                node["action_if_le"] = rng.choice(ACTIONS)
        if rng.random() < rate:
            out["default_action"] = rng.choice(ACTIONS)
    elif kind == "hybrid":
        for key in ["external_threshold", "unresolvable_threshold", "dominant_threshold", "uncertainty_threshold"]:
            if rng.random() < rate:
                out[key] = max(0.0, min(1.2, out[key] + rng.gauss(0.0, 0.10)))
        for key in ["action_external", "action_unresolvable", "action_dominant", "action_uncertain"]:
            if rng.random() < rate:
                out[key] = rng.choice(ACTIONS)
        out["linear"] = mutate_policy(out["linear"], rng, rate=rate * 0.6)
    return out


def choose_action(policy, features):
    feature = {name: features[idx] for idx, name in enumerate(FEATURE_NAMES)}
    kind = policy["kind"]
    if kind == "linear":
        best = None
        for action_idx, action in enumerate(ACTIONS):
            score = policy["bias"][action_idx]
            score += sum(weight * value for weight, value in zip(policy["weights"][action_idx], features))
            if best is None or score > best[1]:
                best = (action, score)
        return best[0]
    if kind == "rule_table":
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
    if kind == "small_tree":
        for node in policy["nodes"]:
            if features[node["feature_idx"]] <= node["threshold"]:
                return node["action_if_le"]
        return policy["default_action"]
    if kind == "hybrid":
        if feature["external_channel_available"] >= policy["external_threshold"]:
            return policy["action_external"]
        if feature["internal_unresolvable_indicator"] >= policy["unresolvable_threshold"]:
            return policy["action_unresolvable"]
        if feature["dominant_cluster_fraction"] >= policy["dominant_threshold"]:
            return policy["action_dominant"]
        if feature["inverse_margin"] >= policy["uncertainty_threshold"] or feature["top1_factorised_disagreement"] > 0.5:
            return policy["action_uncertain"]
        return choose_action(policy["linear"], features)
    raise ValueError(kind)


def policy_fitness(policy, examples, indexes):
    if not indexes:
        indexes = range(len(examples))
    total = 0.0
    counts = Counter()
    for idx in indexes:
        ex = examples[idx]
        action = choose_action(policy, ex["features"])
        counts[action] += 1
        total += action_fitness(ex["action_compact"][action])
    return total / len(list(indexes)), counts


def full_policy_score(policy, examples):
    indexes = list(range(len(examples)))
    score, counts = policy_fitness(policy, examples, indexes)
    return score, dict(counts)


def train_policy(kind, examples, generations, population, seed, out, started, heartbeat_sec):
    rng = random.Random(seed + stable_seed(kind))
    population_items = [make_policy(kind, rng) for _ in range(population)]
    bootstrap = learned_bootstrap_policy(kind, examples)
    if bootstrap is not None:
        population_items[0] = bootstrap
    best_policy = population_items[0]
    best_score = -1e9
    acceptance = []
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
        while len(next_population) < population:
            parent = copy.deepcopy(rng.choice(elites))
            next_population.append(mutate_policy(parent, rng, rate=0.14))
        population_items = next_population
        if gen % max(1, generations // 10) == 0 or gen == generations - 1:
            full_score, counts = full_policy_score(best_policy, examples)
            acceptance.append(
                {
                    "generation": gen,
                    "batch_best_fitness": scored[0][0],
                    "full_best_fitness": full_score,
                    "action_counts": counts,
                }
            )
        now = time.time()
        if now - last >= heartbeat_sec:
            last = now
            append_progress(out, "mutation_progress", started, {"kind": kind, "generation": gen, "best_batch_score": best_score})
            write_json(out / f"partial_mutation_{kind}.json", {"kind": kind, "generation": gen, "best_batch_score": best_score})
    final_score, counts = full_policy_score(best_policy, examples)
    return best_policy, {"kind": kind, "fitness": final_score, "action_counts": counts, "history": acceptance}


def random_policy_action(row_id):
    return ACTIONS[stable_seed(row_id) % len(ACTIONS)]


def output_from_action(pack, arm, action):
    result = copy.deepcopy(pack["actions"][action])
    result["arm"] = arm
    result["selected_action"] = action
    return result


def evaluate_pack_all_arms(pack, policies):
    rows = []
    for arm in REFERENCE_ARMS:
        rows.append(copy.deepcopy(pack["references"][arm]))
    rows.append(output_from_action(pack, "RANDOM_POLICY_CONTROL", random_policy_action(pack["row_id"])))
    rows.append(output_from_action(pack, "GREEDY_DECIDE_CONTROL", "DECIDE"))
    rows.append(output_from_action(pack, "ALWAYS_COUNTER_SUPPORT_CONTROL", "REQUEST_JOINT_COUNTER"))
    for arm, policy in policies.items():
        action = choose_action(policy, pack["features"])
        rows.append(output_from_action(pack, arm, action))
    return rows


def summarize(rows):
    wrong_conf = [row["confidence"] for row in rows if not row["correct"] and not row["abstained"]]
    return {
        "rows": len(rows),
        "accuracy": mean([1.0 if row["correct"] else 0.0 for row in rows]),
        "exact_joint_accuracy": mean([1.0 if row["exact_joint_correct"] else 0.0 for row in rows]),
        "cell_pair_equivalence_accuracy": mean([1.0 if row["cell_pair_equivalence_correct"] else 0.0 for row in rows]),
        "cell_hit_top2_accuracy": mean([1.0 if row["cell_hit_top2_correct"] else 0.0 for row in rows]),
        "operator_exact_accuracy": mean([1.0 if row["operator_exact_correct"] else 0.0 for row in rows]),
        "operator_equivalence_accuracy": mean([1.0 if row["operator_equivalence_correct"] else 0.0 for row in rows]),
        "family_group_accuracy": mean([1.0 if row["family_group_correct"] else 0.0 for row in rows]),
        "joint_binding_consistency_rate": mean([1.0 if row["joint_binding_consistency"] else 0.0 for row in rows]),
        "average_total_support_used": mean([row["total_support_used"] for row in rows]),
        "average_counter_support_used": mean([row["counter_support_used"] for row in rows]),
        "average_external_test_used": mean([row["external_test_used"] for row in rows]),
        "abstain_rate": mean([1.0 if row["abstained"] else 0.0 for row in rows]),
        "false_confidence_rate": mean([1.0 if row["false_confidence"] else 0.0 for row in rows]),
        "confidence_when_wrong": mean(wrong_conf),
        "external_test_unavailable_rate": mean([1.0 if row["external_test_requested_unavailable"] else 0.0 for row in rows]),
        "cell_counter_support_mean": mean([row["cell_counter_support_used"] for row in rows]),
        "operator_counter_support_mean": mean([row["operator_counter_support_used"] for row in rows]),
        "joint_counter_support_mean": mean([row["joint_counter_support_used"] for row in rows]),
    }


def summarize_outputs(outputs):
    by_arm = defaultdict(list)
    by_arm_core = defaultdict(list)
    by_arm_regime = defaultdict(list)
    by_action = defaultdict(Counter)
    by_error = defaultdict(Counter)
    for row in outputs:
        by_arm[row["arm"]].append(row)
        by_arm_regime[(row["arm"], row["support_regime"])].append(row)
        by_action[row["arm"]][row["selected_action"]] += 1
        by_error[(row["arm"], row["support_regime"])][row["error_type"]] += 1
        if row["support_regime"] in CORE_REGIMES:
            by_arm_core[row["arm"]].append(row)
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
            "by_arm_core": partial["by_arm_core"],
            "mutable_hybrid_regime": partial["by_arm_and_regime"].get("MUTABLE_HYBRID_CONTROLLER", {}),
        },
    )
    append_progress(out, "eval_progress", started, {"split": split, "completed_outputs": completed})


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


def mixed_accuracy(metrics, arm):
    return mean(
        [
            regime_accuracy(metrics, arm, "MIXED_CLEAN_AND_CORRELATED"),
            regime_accuracy(metrics, arm, "MIXED_CLEAN_AND_ADVERSARIAL"),
        ]
    )


def cost_adjusted(row):
    return row["exact_joint_accuracy"] - 0.0010 * row["average_total_support_used"] - 0.0015 * row["average_counter_support_used"]


def best_mutable_arm(metrics):
    core = metrics["by_arm_core"]
    return max(MUTABLE_ARMS, key=lambda arm: (cost_adjusted(core[arm]), core[arm]["exact_joint_accuracy"]))


def make_decision(metrics, failed_jobs):
    if failed_jobs:
        return {
            "decision": "mutable_controller_failed_jobs_present",
            "verdict": "D51_FAILED_JOBS_PRESENT",
            "next": "D51_REPAIR",
            "boundary": BOUNDARY,
        }
    best = best_mutable_arm(metrics)
    core = metrics["by_arm_core"]
    best_row = core[best]
    corr = regime_accuracy(metrics, best, "CORRELATED_ECHO_SUPPORT")
    adv = regime_accuracy(metrics, best, "ADVERSARIAL_DISTRACTOR_SUPPORT")
    indist = metrics["by_arm_and_regime"][best]["INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT"]
    d50_full = core["HANDCODED_D50_FULL_REPAIRED_ECF_REFERENCE"]
    cap9 = core["HANDCODED_CAP_9_REFERENCE"]
    if indist["false_confidence_rate"] > 0.01:
        return {
            "decision": "mutable_controller_false_confidence_failure",
            "verdict": "D51_MUTABLE_CONTROLLER_FALSE_CONFIDENCE_FAILURE",
            "next": "D51A_ABSTAIN_POLICY_REPAIR",
            "boundary": BOUNDARY,
        }
    pass_accuracy = (
        best_row["exact_joint_accuracy"] >= 0.995
        and corr >= 0.995
        and adv >= 0.995
        and indist["false_confidence_rate"] <= 0.01
    )
    pass_cost = best_row["average_total_support_used"] <= d50_full["average_total_support_used"]
    pass_cap9 = cost_adjusted(best_row) >= cost_adjusted(cap9)
    if pass_accuracy and pass_cost and pass_cap9:
        return {
            "decision": "mutable_ecf_controller_prototype_positive",
            "verdict": "D51_MUTABLE_ECF_CONTROLLER_PROTOTYPE_POSITIVE",
            "next": "D52_MUTABLE_ECF_CONTROLLER_SCALE_CONFIRM",
            "best_mutable_arm": best,
            "boundary": BOUNDARY,
        }
    if pass_accuracy:
        return {
            "decision": "mutable_controller_positive_high_cost",
            "verdict": "D51_MUTABLE_CONTROLLER_POSITIVE_HIGH_COST",
            "next": "D51C_COST_OPTIMIZATION",
            "best_mutable_arm": best,
            "boundary": BOUNDARY,
        }
    return {
        "decision": "mutable_controller_not_confirmed",
        "verdict": "D51_MUTABLE_CONTROLLER_NOT_CONFIRMED",
        "next": "D51R_CONTROLLER_FEATURE_REPAIR",
        "best_mutable_arm": best,
        "boundary": BOUNDARY,
    }


def make_upstream_manifest(repo_root):
    root = repo_root / "target/pilot_wave/d50_joint_formula_discovery_scale_confirm/smoke"
    manifest = {
        "upstream": "D50_JOINT_FORMULA_DISCOVERY_SCALE_CONFIRM",
        "expected_decision": "joint_formula_discovery_scale_confirmed",
        "expected_next": "D51_MUTABLE_ECF_CONTROLLER_PROTOTYPE",
        "decision_present": (root / "decision.json").exists(),
        "summary_present": (root / "summary.json").exists(),
    }
    if (root / "decision.json").exists():
        manifest["decision_json"] = json.loads((root / "decision.json").read_text(encoding="utf-8"))
    if (root / "summary.json").exists():
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        manifest["scale_mode"] = summary.get("scale_mode")
        full = summary["key_metrics"]["full_repaired"]
        cap9 = summary["key_metrics"]["cap9"]
        manifest["key_metrics"] = {
            "full_exact_joint": full["exact_joint_accuracy"],
            "full_support": full["average_total_support_used"],
            "full_counter_support": full["average_counter_support_used"],
            "cap9_exact_joint": cap9["exact_joint_accuracy"],
            "cap9_support": cap9["average_total_support_used"],
            "correlated_echo": summary["key_metrics"]["full_by_regime"]["CORRELATED_ECHO_SUPPORT"]["accuracy"],
            "adversarial_distractor": summary["key_metrics"]["full_by_regime"]["ADVERSARIAL_DISTRACTOR_SUPPORT"]["accuracy"],
            "external_test_required": summary["key_metrics"]["external_required"]["accuracy"],
            "indistinguishable_abstain": summary["key_metrics"]["indistinguishable"]["abstain_rate"],
            "false_confidence": summary["key_metrics"]["indistinguishable"]["false_confidence_rate"],
        }
    return manifest


def make_reports(out, aggregate, decision, policy_reports):
    test = aggregate["test_metrics"]
    core = test["by_arm_core"]
    best = decision.get("best_mutable_arm", best_mutable_arm(test))
    reports = {
        "controller_input_feature_report.json": {
            "feature_names": FEATURE_NAMES,
            "truth_labels_as_controller_inputs": False,
            "notes": [
                "Features are computed from score vectors, cluster diagnostics, and support-channel availability.",
                "The controller does not receive truth_joint, truth_pair, true_operator, false_joint, or expected answer labels.",
            ],
        },
        "mutation_acceptance_report.json": policy_reports,
        "policy_action_distribution_report.json": test["action_distribution"],
        "support_cost_frontier_report.json": {
            arm: {
                "exact_joint_accuracy": core[arm]["exact_joint_accuracy"],
                "support": core[arm]["average_total_support_used"],
                "counter_support": core[arm]["average_counter_support_used"],
                "cost_adjusted": cost_adjusted(core[arm]),
            }
            for arm in [
                "HANDCODED_CAP_7_REFERENCE",
                "HANDCODED_CAP_9_REFERENCE",
                "HANDCODED_D50_FULL_REPAIRED_ECF_REFERENCE",
                best,
            ]
        },
        "false_confidence_report.json": {
            arm: {
                "core_false_confidence": core[arm]["false_confidence_rate"],
                "indistinguishable": test["by_arm_and_regime"][arm]["INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT"]["false_confidence_rate"],
            }
            for arm in ARMS
        },
        "regime_breakdown_report.json": test["by_arm_and_regime"],
        "controller_comparison_report.json": {
            arm: {
                "exact_joint_accuracy": core[arm]["exact_joint_accuracy"],
                "support": core[arm]["average_total_support_used"],
                "counter_support": core[arm]["average_counter_support_used"],
                "cost_adjusted": cost_adjusted(core[arm]),
            }
            for arm in ARMS
        },
        "best_policy_report.json": {
            "best_mutable_arm": best,
            "policy": aggregate["policies"][best],
            "metrics": core[best],
            "action_distribution": test["action_distribution"][best],
        },
    }
    for name, value in reports.items():
        write_json(out / name, value)
    return reports


def write_report(out, decision, aggregate):
    core = aggregate["test_metrics"]["by_arm_core"]
    best = decision.get("best_mutable_arm", best_mutable_arm(aggregate["test_metrics"]))
    lines = [
        "# D51 Mutable ECF Controller Prototype Result",
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
        f"best_mutable_arm = {best}",
        "```",
        "",
        "Controller comparison:",
        "",
        "```text",
    ]
    for arm in ARMS:
        row = core[arm]
        lines.append(
            f"{arm}: exact={row['exact_joint_accuracy']:.4f}, false_conf={row['false_confidence_rate']:.4f}, support={row['average_total_support_used']:.3f}, counter={row['average_counter_support_used']:.3f}"
        )
    lines.extend(["```", "", f"Boundary: {BOUNDARY}", ""])
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", default="10501,10502,10503,10504,10505")
    parser.add_argument("--train-rows-per-seed", type=int, default=800)
    parser.add_argument("--test-rows-per-seed", type=int, default=800)
    parser.add_argument("--ood-rows-per-seed", type=int, default=800)
    parser.add_argument("--generations", type=int, default=300)
    parser.add_argument("--population", type=int, default=96)
    parser.add_argument("--workers", default="auto")
    parser.add_argument("--cpu-target", default="saturate")
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    parser.add_argument("--scale-mode", default="full", choices=["full", "scale_lite"])
    return parser.parse_args()


def main():
    args = parse_args()
    started = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = [int(seed) for seed in args.seeds.split(",") if seed]
    bundle = d49.make_bundle("ALL28_UNORDERED")
    repo_root = Path(__file__).resolve().parents[2]
    write_json(
        out / "queue.json",
        {
            "task": "D51 mutable ECF controller prototype",
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
    write_json(out / "d50_upstream_manifest.json", make_upstream_manifest(repo_root))
    write_json(
        out / "dataset_manifest.json",
        {
            "task": "D51_MUTABLE_ECF_CONTROLLER_PROTOTYPE",
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
            "note": "D51 uses deterministic symbolic scoring and mutable controller search; no external model/API/download used.",
        },
    )
    train_rows = make_rows_with_progress(seeds, args.train_rows_per_seed, "train", bundle, out, started, args.heartbeat_sec)
    test_rows = make_rows_with_progress(seeds, args.test_rows_per_seed, "test", bundle, out, started, args.heartbeat_sec)
    ood_rows = make_rows_with_progress(seeds, args.ood_rows_per_seed, "ood", bundle, out, started, args.heartbeat_sec)
    append_progress(out, "rows_generated", started, {"train": len(train_rows), "test": len(test_rows), "ood": len(ood_rows)})
    train_packs = build_packs(train_rows, bundle, out, started, args.heartbeat_sec, args.workers, "train")
    train_examples = [
        {"features": pack["features"], "action_compact": pack["action_compact"]}
        for pack in train_packs
    ]
    policies = {}
    policy_reports = {}
    arm_to_kind = {
        "MUTABLE_LINEAR_CONTROLLER": "linear",
        "MUTABLE_RULE_TABLE_CONTROLLER": "rule_table",
        "MUTABLE_SMALL_TREE_CONTROLLER": "small_tree",
        "MUTABLE_HYBRID_CONTROLLER": "hybrid",
    }
    for arm, kind in arm_to_kind.items():
        policy, report = train_policy(kind, train_examples, args.generations, args.population, 51_000 + stable_seed(arm), out, started, args.heartbeat_sec)
        policies[arm] = policy
        policy_reports[arm] = report
        append_progress(out, "mutation_complete", started, {"arm": arm, "fitness": report["fitness"]})
    write_json(out / "trained_policy_manifest.json", {"policies": policies, "reports": policy_reports})
    test_packs = build_packs(test_rows, bundle, out, started, args.heartbeat_sec, args.workers, "test")
    ood_packs = build_packs(ood_rows, bundle, out, started, args.heartbeat_sec, args.workers, "ood")
    test_outputs = evaluate_packs(test_packs, policies, out / "row_outputs_test.jsonl", out, "test", started, args.heartbeat_sec)
    ood_outputs = evaluate_packs(ood_packs, policies, out / "row_outputs_ood.jsonl", out, "ood", started, args.heartbeat_sec)
    test_metrics = summarize_outputs(test_outputs)
    ood_metrics = summarize_outputs(ood_outputs)
    failed_jobs = []
    decision = make_decision(test_metrics, failed_jobs)
    aggregate = {
        "task": "D51_MUTABLE_ECF_CONTROLLER_PROTOTYPE",
        "scale_mode": args.scale_mode,
        "test_metrics": test_metrics,
        "ood_metrics": ood_metrics,
        "policies": policies,
        "policy_reports": policy_reports,
        "best_mutable_arm": decision.get("best_mutable_arm", best_mutable_arm(test_metrics)),
        "failed_jobs": failed_jobs,
        "decision": decision["decision"],
        "verdict": decision["verdict"],
        "next": decision["next"],
        "boundary": BOUNDARY,
    }
    reports = make_reports(out, aggregate, decision, policy_reports)
    write_json(out / "aggregate_metrics.json", aggregate)
    write_json(out / "decision.json", decision)
    write_json(
        out / "summary.json",
        {
            "scale_mode": args.scale_mode,
            "decision": decision,
            "best_mutable_arm": aggregate["best_mutable_arm"],
            "key_metrics": {
                "best_mutable": test_metrics["by_arm_core"][aggregate["best_mutable_arm"]],
                "best_by_regime": test_metrics["by_arm_and_regime"][aggregate["best_mutable_arm"]],
                "d50_full_reference": test_metrics["by_arm_core"]["HANDCODED_D50_FULL_REPAIRED_ECF_REFERENCE"],
                "cap7_reference": test_metrics["by_arm_core"]["HANDCODED_CAP_7_REFERENCE"],
                "cap9_reference": test_metrics["by_arm_core"]["HANDCODED_CAP_9_REFERENCE"],
                "support_cost_frontier": reports["support_cost_frontier_report.json"],
                "action_distribution": test_metrics["action_distribution"][aggregate["best_mutable_arm"]],
            },
            "boundary": BOUNDARY,
        },
    )
    write_report(out, decision, aggregate)
    queue = json.loads((out / "queue.json").read_text(encoding="utf-8"))
    write_json(out / "queue.json", {**queue, "status": "complete"})
    append_progress(out, "final_decision", started, decision)
    print(json.dumps({"decision": decision["decision"], "verdict": decision["verdict"], "next": decision["next"], "best": aggregate["best_mutable_arm"], "scale_mode": args.scale_mode}, indent=2))


if __name__ == "__main__":
    main()
