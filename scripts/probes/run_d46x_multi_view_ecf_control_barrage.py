#!/usr/bin/env python3
"""D46X multi-view ECF control barrage for controlled symbolic IPF support."""

import argparse
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import run_d45_robust_support_policy_prototype as d45

PRIMARY_SPACE = "ALL28_UNORDERED"
PRIMARY_BUDGET = 5
D46_ROBUST_COMBINED_SUPPORT = 6.863675

SPACES = [
    "CURRENT5",
    "ALL28_UNORDERED",
    "ORDERED56_CONTROL",
    "CURRENT5_PLUS_DISTRACTORS_20",
    "CURRENT5_PLUS_DISTRACTORS_50",
]
REGIMES = [
    "CLEAN_INDEPENDENT_SUPPORT",
    "CORRELATED_NOISE_SUPPORT",
    "ADVERSARIAL_DISTRACTOR_SUPPORT",
    "MIXED_CLEAN_AND_CORRELATED",
    "MIXED_CLEAN_AND_ADVERSARIAL",
]
SUPPORT_BUDGETS = [1, 2, 3, 4, 5]
ARMS = [
    "SCALAR_ARGMAX_ONLY",
    "VECTOR_FIELD_ONLY",
    "ENTROPY_SUPPORT_POLICY",
    "MARGIN_SUPPORT_POLICY",
    "COLLISION_SUPPORT_POLICY",
    "SUPPORT_INDEPENDENCE_DEDUP_POLICY",
    "COUNTERFACTUAL_TOP1_TOP2_POLICY",
    "SCALAR+VECTOR",
    "VECTOR+ENTROPY+MARGIN",
    "VECTOR+COLLISION+INDEPENDENCE",
    "FULL_MULTI_VIEW_ECF_POLICY",
    "RANDOM_EXTRA_SUPPORT_CONTROL",
    "BAD_VIEW_CONTROL",
    "SHUFFLED_VECTOR_FIELD_CONTROL",
    "NO_COUNTERFACTUAL_CONTROL",
]
SINGLE_VIEW_ARMS = [
    "SCALAR_ARGMAX_ONLY",
    "VECTOR_FIELD_ONLY",
    "ENTROPY_SUPPORT_POLICY",
    "MARGIN_SUPPORT_POLICY",
    "COLLISION_SUPPORT_POLICY",
    "SUPPORT_INDEPENDENCE_DEDUP_POLICY",
    "COUNTERFACTUAL_TOP1_TOP2_POLICY",
]
PAIR_ARMS = ["SCALAR+VECTOR"]
TRIPLE_ARMS = ["VECTOR+ENTROPY+MARGIN", "VECTOR+COLLISION+INDEPENDENCE"]
CONTROL_ARMS = [
    "RANDOM_EXTRA_SUPPORT_CONTROL",
    "BAD_VIEW_CONTROL",
    "SHUFFLED_VECTOR_FIELD_CONTROL",
    "NO_COUNTERFACTUAL_CONTROL",
]
VIEW_SIGNALS = [
    "entropy_high",
    "margin_low",
    "collision_high",
    "support_independence_bad",
    "leave_one_out_unstable",
    "scalar_vector_disagree",
    "counterfactual_needed",
]
ROW_SAMPLE_LIMIT = 2


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True))
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


def safe_div(num, den):
    return num / den if den else 0.0


def truth_equivalence(row):
    return f"{d45.canonical_key(d45.TRUE_PAIRS[row['truth_family']])}::add"


def make_rows(seeds, rows_per_seed, split):
    return [row for row in d45.make_rows(seeds, rows_per_seed, split) if row["support_regime"] in REGIMES]


def cap_rows(rows, space, per_regime_cap):
    if space == PRIMARY_SPACE:
        return rows
    counts = Counter()
    selected = []
    for row in rows:
        key = row["support_regime"]
        if counts[key] < per_regime_cap:
            counts[key] += 1
            selected.append(row)
    return selected


def normalize_vector(vector):
    keys = list(vector)
    avg = mean([vector[key] for key in keys])
    centered = {key: vector[key] - avg for key in keys}
    scale = math.sqrt(sum(value * value for value in centered.values())) or 1.0
    return {key: value / scale for key, value in centered.items()}


def aggregate_vector_field(vectors):
    out = defaultdict(float)
    for vector in vectors:
        for cid, value in normalize_vector(vector).items():
            out[cid] += value
    return dict(out)


def blend_scores(left, right, left_weight=0.75):
    keys = set(left) | set(right)
    return {key: left_weight * left.get(key, 0.0) + (1.0 - left_weight) * right.get(key, 0.0) for key in keys}


def shuffled_scores(scores, rng):
    keys = sorted(scores)
    values = [scores[key] for key in keys]
    rng.shuffle(values)
    return dict(zip(keys, values))


def cluster_details(vectors):
    cluster_count, dominant_fraction, collision_count = d45.cluster_stats(vectors)
    duplicate_score = max(0.0, dominant_fraction - safe_div(1.0, max(1, cluster_count)))
    return cluster_count, dominant_fraction, collision_count, duplicate_score


def leave_one_out_flip_count(vectors, bundle, baseline_family):
    if len(vectors) <= 1:
        return 0
    flips = 0
    for idx in range(len(vectors)):
        sub = [vector for pos, vector in enumerate(vectors) if pos != idx]
        pred = d45.predict_from_scores(d45.aggregate_sum(sub), bundle)["pred_family"]
        if pred != baseline_family:
            flips += 1
    return flips


def wrong_family_counter_support(row, rng, count):
    wrong = [family for family in d45.FAMILIES if family != row["truth_family"]]
    return [d45.make_truth_board(rng, rng.choice(wrong), row["split"]) for _ in range(count)]


def entropy_cutoff(bundle):
    return math.log(max(2, len(bundle["candidates"]))) * 0.55


def build_view_state(row, bundle, support_budget):
    original_used = min(support_budget, len(row["supports"]))
    vectors = d45.support_vectors(row, bundle, original_used)
    scalar_scores = d45.aggregate_sum(vectors)
    scalar_pred = d45.predict_from_scores(scalar_scores, bundle)
    vector_scores = aggregate_vector_field(vectors)
    vector_pred = d45.predict_from_scores(vector_scores, bundle)
    dedup_scores = d45.aggregate_duplicate_downweighted(vectors)
    median_scores = d45.aggregate_median(vectors, trim=True) if vectors else {}
    cluster_count, dominant_fraction, collision_count, duplicate_score = cluster_details(vectors)
    loo_flips = leave_one_out_flip_count(vectors, bundle, scalar_pred["pred_family"])
    signals = {
        "entropy_high": scalar_pred["entropy"] >= entropy_cutoff(bundle),
        "margin_low": scalar_pred["top1_top2_margin"] <= 0.5,
        "collision_high": collision_count > 0 and dominant_fraction >= 0.40,
        "support_independence_bad": dominant_fraction >= 0.60 and original_used >= 3,
        "leave_one_out_unstable": loo_flips > 0,
        "scalar_vector_disagree": scalar_pred["pred_family"] != vector_pred["pred_family"],
    }
    signals["counterfactual_needed"] = (
        scalar_pred["pred_family"] == "distractor"
        or signals["margin_low"]
        or signals["support_independence_bad"]
        or signals["leave_one_out_unstable"]
    )
    return {
        "original_used": original_used,
        "vectors": vectors,
        "scalar_scores": scalar_scores,
        "scalar_pred": scalar_pred,
        "vector_scores": vector_scores,
        "vector_pred": vector_pred,
        "dedup_scores": dedup_scores,
        "median_scores": median_scores,
        "cluster_count": cluster_count,
        "dominant_fraction": dominant_fraction,
        "collision_count": collision_count,
        "duplicate_score": duplicate_score,
        "leave_one_out_flip_count": loo_flips,
        "signals": signals,
    }


def request_counter_support(row, bundle, pred, rng, count):
    boards = d45.generate_counter_support(row, bundle, pred, rng, count=count)
    return [d45.support_score_vector(board, bundle) for board in boards]


def predict_with_scores(scores, bundle, bad_projection=False):
    return d45.predict_from_scores(scores, bundle, bad_projection=bad_projection)


def evaluate_arm(row, bundle, arm, support_budget, rng):
    state = build_view_state(row, bundle, support_budget)
    vectors = list(state["vectors"])
    counter_used = 0
    counter_requested = False
    counter_target = []
    pred = state["scalar_pred"]
    scores_used = state["scalar_scores"]

    if arm == "SCALAR_ARGMAX_ONLY":
        pred = state["scalar_pred"]
        scores_used = state["scalar_scores"]
    elif arm == "VECTOR_FIELD_ONLY":
        pred = state["vector_pred"]
        scores_used = state["vector_scores"]
    elif arm == "ENTROPY_SUPPORT_POLICY":
        scores_used = state["median_scores"] if state["signals"]["entropy_high"] else state["scalar_scores"]
        pred = predict_with_scores(scores_used, bundle)
    elif arm == "MARGIN_SUPPORT_POLICY":
        scores_used = state["dedup_scores"] if state["signals"]["margin_low"] else state["scalar_scores"]
        pred = predict_with_scores(scores_used, bundle)
    elif arm == "COLLISION_SUPPORT_POLICY":
        scores_used = state["dedup_scores"] if state["signals"]["collision_high"] else state["scalar_scores"]
        pred = predict_with_scores(scores_used, bundle)
    elif arm == "SUPPORT_INDEPENDENCE_DEDUP_POLICY":
        scores_used = state["dedup_scores"]
        pred = predict_with_scores(scores_used, bundle)
    elif arm == "COUNTERFACTUAL_TOP1_TOP2_POLICY":
        if state["signals"]["counterfactual_needed"]:
            counter_requested = True
            counter_target = [cid for cid, _score in pred["ordered"][:2]]
            extra = request_counter_support(row, bundle, pred, rng, count=1)
            vectors += extra
            counter_used = len(extra)
            scores_used = d45.aggregate_sum(vectors)
            pred = predict_with_scores(scores_used, bundle)
    elif arm == "SCALAR+VECTOR":
        scores_used = blend_scores(state["scalar_scores"], state["vector_scores"], 0.72)
        pred = predict_with_scores(scores_used, bundle)
    elif arm == "VECTOR+ENTROPY+MARGIN":
        base = blend_scores(state["scalar_scores"], state["vector_scores"], 0.55)
        if state["signals"]["entropy_high"] or state["signals"]["margin_low"]:
            base = blend_scores(base, state["median_scores"], 0.70)
        scores_used = base
        pred = predict_with_scores(scores_used, bundle)
    elif arm == "VECTOR+COLLISION+INDEPENDENCE":
        base = blend_scores(state["scalar_scores"], state["vector_scores"], 0.55)
        if state["signals"]["collision_high"] or state["signals"]["support_independence_bad"]:
            base = blend_scores(state["dedup_scores"], state["vector_scores"], 0.82)
        scores_used = base
        pred = predict_with_scores(scores_used, bundle)
    elif arm == "FULL_MULTI_VIEW_ECF_POLICY":
        base = blend_scores(state["dedup_scores"], state["vector_scores"], 0.84)
        pred = predict_with_scores(base, bundle)
        suspicious = (
            state["signals"]["support_independence_bad"]
            or state["signals"]["leave_one_out_unstable"]
            or state["scalar_pred"]["pred_family"] == "distractor"
            or pred["top1_top2_margin"] <= 0.5
        )
        if suspicious:
            counter_requested = True
            counter_target = [cid for cid, _score in pred["ordered"][:2]]
            request_count = 2 if support_budget >= 3 else 1
            extra = request_counter_support(row, bundle, pred, rng, count=request_count)
            vectors += extra
            counter_used = len(extra)
            scores_used = d45.aggregate_duplicate_downweighted(vectors)
            pred = predict_with_scores(scores_used, bundle)
        else:
            scores_used = base
    elif arm == "RANDOM_EXTRA_SUPPORT_CONTROL":
        pred = state["scalar_pred"]
        if state["signals"]["support_independence_bad"] or state["signals"]["counterfactual_needed"]:
            counter_requested = True
            boards = [d45.make_truth_board(rng, row["truth_family"], row["split"])]
            extra = [d45.support_score_vector(board, bundle) for board in boards]
            vectors += extra
            counter_used = len(extra)
            scores_used = d45.aggregate_sum(vectors)
            pred = predict_with_scores(scores_used, bundle)
    elif arm == "BAD_VIEW_CONTROL":
        pred = predict_with_scores(state["scalar_scores"], bundle, bad_projection=True)
        scores_used = state["scalar_scores"]
    elif arm == "SHUFFLED_VECTOR_FIELD_CONTROL":
        scores_used = shuffled_scores(state["vector_scores"], rng)
        pred = predict_with_scores(scores_used, bundle)
    elif arm == "NO_COUNTERFACTUAL_CONTROL":
        scores_used = blend_scores(state["dedup_scores"], state["vector_scores"], 0.84)
        pred = predict_with_scores(scores_used, bundle)
    else:
        raise ValueError(arm)

    truth_family = row["truth_family"]
    truth_equiv = truth_equivalence(row)
    correct = pred["pred_family"] == truth_family
    scalar_correct = state["scalar_pred"]["pred_family"] == truth_family
    counter_resolved = bool(counter_requested and correct and not scalar_correct)
    total_support_used = state["original_used"] + counter_used
    family_correct = correct
    candidate_correct = pred["pred_candidate"] == bundle["exact_truth"].get(truth_family)
    equivalence_correct = pred["pred_equivalence"] == truth_equiv
    return {
        "truth_family": truth_family,
        "pred_family": pred["pred_family"],
        "truth_equivalence": truth_equiv,
        "pred_equivalence": pred["pred_equivalence"],
        "pred_candidate": pred["pred_candidate"],
        "candidate_correct": candidate_correct,
        "family_correct": family_correct,
        "equivalence_correct": equivalence_correct,
        "correct": family_correct,
        "support_budget_cap": support_budget,
        "original_support_used": state["original_used"],
        "counter_support_used": counter_used,
        "total_support_used": total_support_used,
        "support_cluster_count": state["cluster_count"],
        "dominant_cluster_fraction": state["dominant_fraction"],
        "duplicate_support_score": state["duplicate_score"],
        "collision_count": state["collision_count"],
        "leave_one_out_flip_count": state["leave_one_out_flip_count"],
        "entropy": pred["entropy"],
        "top1_top2_margin": pred["top1_top2_margin"],
        "scalar_pred_family": state["scalar_pred"]["pred_family"],
        "vector_pred_family": state["vector_pred"]["pred_family"],
        "scalar_correct": scalar_correct,
        "vector_correct": state["vector_pred"]["pred_family"] == truth_family,
        "counter_support_requested": counter_requested,
        "counter_support_target": counter_target,
        "counter_support_resolved": counter_resolved,
        "correlated_support_detected": state["signals"]["support_independence_bad"],
        "adversarial_support_detected": state["signals"]["counterfactual_needed"],
        "view_signals": state["signals"],
        "error_type": d45.classify_error(truth_family, pred["pred_family"], truth_equiv, pred["pred_equivalence"]),
    }


def empty_stats():
    return Counter()


def update_stats(bucket, row):
    bucket["n"] += 1
    bucket["correct"] += int(row["correct"])
    bucket["candidate_correct"] += int(row["candidate_correct"])
    bucket["family_correct"] += int(row["family_correct"])
    bucket["equivalence_correct"] += int(row["equivalence_correct"])
    bucket["original_support"] += row["original_support_used"]
    bucket["counter_support"] += row["counter_support_used"]
    bucket["total_support"] += row["total_support_used"]
    bucket["counter_requested"] += int(row["counter_support_requested"])
    bucket["counter_resolved"] += int(row["counter_support_resolved"])
    bucket["scalar_failure"] += int(not row["scalar_correct"])
    bucket["vector_correct"] += int(row["vector_correct"])
    bucket["entropy"] += row["entropy"]
    bucket["margin"] += row["top1_top2_margin"]
    bucket["collision_count"] += row["collision_count"]
    bucket["independence_detected"] += int(row["correlated_support_detected"])
    bucket["adversarial_detected"] += int(row["adversarial_support_detected"])


def finalize_stats(bucket):
    n = bucket["n"]
    return {
        "rows": int(n),
        "accuracy": safe_div(bucket["correct"], n),
        "candidate_accuracy": safe_div(bucket["candidate_correct"], n),
        "family_accuracy": safe_div(bucket["family_correct"], n),
        "equivalence_accuracy": safe_div(bucket["equivalence_correct"], n),
        "average_original_support_used": safe_div(bucket["original_support"], n),
        "average_counter_support_used": safe_div(bucket["counter_support"], n),
        "average_total_support_used": safe_div(bucket["total_support"], n),
        "counter_support_request_rate": safe_div(bucket["counter_requested"], n),
        "counter_support_resolution_rate": safe_div(bucket["counter_resolved"], n),
        "scalar_argmax_failure_rate": safe_div(bucket["scalar_failure"], n),
        "vector_alignment_accuracy": safe_div(bucket["vector_correct"], n),
        "mean_entropy": safe_div(bucket["entropy"], n),
        "mean_top1_top2_margin": safe_div(bucket["margin"], n),
        "mean_collision_count": safe_div(bucket["collision_count"], n),
        "support_independence_detection_rate": safe_div(bucket["independence_detected"], n),
        "adversarial_detection_rate": safe_div(bucket["adversarial_detected"], n),
    }


def update_binary_confusion(bucket, pred, actual):
    if pred and actual:
        bucket["tp"] += 1
    elif pred and not actual:
        bucket["fp"] += 1
    elif not pred and actual:
        bucket["fn"] += 1
    else:
        bucket["tn"] += 1


def finalize_confusion(bucket):
    tp = bucket["tp"]
    fp = bucket["fp"]
    fn = bucket["fn"]
    tn = bucket["tn"]
    return {
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "precision": safe_div(tp, tp + fp),
        "recall": safe_div(tp, tp + fn),
        "accuracy": safe_div(tp + tn, tp + fp + tn + fn),
    }


def signal_correlation(signal_rows):
    matrix = {}
    for left in VIEW_SIGNALS:
        matrix[left] = {}
        for right in VIEW_SIGNALS:
            xs = [1.0 if row[left] else 0.0 for row in signal_rows]
            ys = [1.0 if row[right] else 0.0 for row in signal_rows]
            mx = mean(xs)
            my = mean(ys)
            cov = mean([(x - mx) * (y - my) for x, y in zip(xs, ys)])
            vx = mean([(x - mx) ** 2 for x in xs])
            vy = mean([(y - my) ** 2 for y in ys])
            matrix[left][right] = safe_div(cov, math.sqrt(vx * vy)) if vx and vy else (1.0 if left == right else 0.0)
    return matrix


def metrics_for_policy(policy_table, policy):
    return policy_table[str(PRIMARY_BUDGET)][policy]


def build_tables(stats, regime_stats, split_stats, seed_stats):
    policy_table = defaultdict(dict)
    regime_table = defaultdict(lambda: defaultdict(dict))
    split_table = defaultdict(lambda: defaultdict(dict))
    seed_table = defaultdict(lambda: defaultdict(dict))
    for (space, budget, policy), bucket in stats.items():
        policy_table[space].setdefault(str(budget), {})[policy] = finalize_stats(bucket)
    for (space, budget, policy, regime), bucket in regime_stats.items():
        regime_table[space][str(budget)].setdefault(policy, {})[regime] = finalize_stats(bucket)
    for (space, budget, policy, split), bucket in split_stats.items():
        split_table[space][str(budget)].setdefault(policy, {})[split] = finalize_stats(bucket)
    for (space, budget, policy, seed, regime), bucket in seed_stats.items():
        seed_table[space][str(budget)].setdefault(policy, {}).setdefault(str(seed), {})[regime] = finalize_stats(bucket)
    return policy_table, regime_table, split_table, seed_table


def add_accuracy_aliases(policy_table, regime_table):
    for space, budget_map in policy_table.items():
        for budget, policies in budget_map.items():
            for policy, metric in policies.items():
                regimes = regime_table[space][budget][policy]
                clean = regimes.get("CLEAN_INDEPENDENT_SUPPORT", {})
                corr = regimes.get("CORRELATED_NOISE_SUPPORT", {})
                adv = regimes.get("ADVERSARIAL_DISTRACTOR_SUPPORT", {})
                mixed_rows = Counter()
                for regime in ["MIXED_CLEAN_AND_CORRELATED", "MIXED_CLEAN_AND_ADVERSARIAL"]:
                    raw = regimes.get(regime, {})
                    n = raw.get("rows", 0)
                    mixed_rows["n"] += n
                    mixed_rows["correct"] += raw.get("accuracy", 0.0) * n
                    mixed_rows["candidate_correct"] += raw.get("candidate_accuracy", 0.0) * n
                    mixed_rows["equivalence_correct"] += raw.get("equivalence_accuracy", 0.0) * n
                metric["clean_accuracy"] = clean.get("accuracy", 0.0)
                metric["correlated_accuracy"] = corr.get("accuracy", 0.0)
                metric["adversarial_accuracy"] = adv.get("accuracy", 0.0)
                metric["mixed_accuracy"] = safe_div(mixed_rows["correct"], mixed_rows["n"])
                metric["candidate_accuracy"] = metric.get("candidate_accuracy", 0.0)
                metric["family_accuracy"] = metric.get("family_accuracy", metric.get("accuracy", 0.0))
                metric["equivalence_accuracy"] = metric.get("equivalence_accuracy", 0.0)


def compact_table_for_report(primary):
    return {
        policy: {
            "clean": primary[policy]["clean_accuracy"],
            "correlated": primary[policy]["correlated_accuracy"],
            "adversarial": primary[policy]["adversarial_accuracy"],
            "mixed": primary[policy]["mixed_accuracy"],
            "support": primary[policy]["average_total_support_used"],
            "counter": primary[policy]["average_counter_support_used"],
        }
        for policy in ARMS
    }


def write_report(out, decision, primary, synergy, support_frontier):
    lines = [
        "# D46X Multi-View ECF Control Barrage",
        "",
        f"Decision: `{decision['decision']}`",
        f"Verdict: `{decision['verdict']}`",
        f"Next: `{decision['next']}`",
        "",
        "| arm | clean | correlated | adversarial | mixed | support | counter |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        metric = primary[arm]
        lines.append(
            f"| {arm} | {metric['clean_accuracy']:.4f} | {metric['correlated_accuracy']:.4f} | "
            f"{metric['adversarial_accuracy']:.4f} | {metric['mixed_accuracy']:.4f} | "
            f"{metric['average_total_support_used']:.3f} | {metric['average_counter_support_used']:.3f} |"
        )
    lines += [
        "",
        f"Best single view: `{synergy['best_single_view']['arm']}`",
        f"Best pair/triple: `{synergy['best_pair_or_triple']['arm']}`",
        f"Full multiview support: `{support_frontier['FULL_MULTI_VIEW_ECF_POLICY']['average_total_support_used']:.3f}`",
        "",
        "Boundary: D46X only tests multi-view ECF control in controlled symbolic primitive discovery. It does not prove raw visual Raven reasoning, Raven solved, AGI, consciousness, architecture superiority, or that intelligence is literally a force.",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", default="9901,9902,9903,9904,9905")
    parser.add_argument("--train-rows-per-seed", type=int, default=800)
    parser.add_argument("--test-rows-per-seed", type=int, default=800)
    parser.add_argument("--ood-rows-per-seed", type=int, default=800)
    parser.add_argument("--workers", default="auto")
    parser.add_argument("--cpu-target", default="saturate")
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    parser.add_argument("--non-primary-regime-cap", type=int, default=220)
    return parser.parse_args()


def main():
    args = parse_args()
    started = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = [int(seed) for seed in args.seeds.split(",") if seed]
    write_json(
        out / "queue.json",
        {
            "task": "D46X multi-view ECF control barrage",
            "status": "running",
            "seeds": seeds,
            "rows": {"train": args.train_rows_per_seed, "test": args.test_rows_per_seed, "ood": args.ood_rows_per_seed},
            "workers": args.workers,
            "cpu_target": args.cpu_target,
            "no_black_box": True,
            "time_based_progress_writeouts": True,
        },
    )
    append_progress(out, "queue_written", started, {"out": str(out)})
    write_json(
        out / "compute_probe.json",
        {
            "cpu_count": os.cpu_count(),
            "workers_requested": args.workers,
            "cpu_target": args.cpu_target,
            "implementation": "single-process streaming evaluator with heartbeat snapshots; accepts worker args for experiment contract compatibility",
        },
    )
    test_base = make_rows(seeds, args.test_rows_per_seed, "test")
    ood_base = make_rows(seeds, args.ood_rows_per_seed, "ood")
    bundles = {space: d45.make_candidates(space) for space in SPACES}
    write_json(
        out / "dataset_manifest.json",
        {
            "seeds": seeds,
            "train_rows_per_seed_argument": args.train_rows_per_seed,
            "test_rows": len(test_base),
            "ood_rows": len(ood_base),
            "spaces": SPACES,
            "primary_space": PRIMARY_SPACE,
            "non_primary_regime_cap": args.non_primary_regime_cap,
            "regimes": REGIMES,
            "support_budgets": SUPPORT_BUDGETS,
            "arms": ARMS,
            "candidate_family_equivalence_metrics_separated": True,
            "row_outputs_sampled_for_size": True,
        },
    )
    append_progress(out, "dataset_built", started, {"test_rows": len(test_base), "ood_rows": len(ood_base)})

    stats = defaultdict(empty_stats)
    regime_stats = defaultdict(empty_stats)
    split_stats = defaultdict(empty_stats)
    seed_stats = defaultdict(empty_stats)
    signal_confusions = defaultdict(empty_stats)
    signal_rows = []
    sample_counts = Counter()
    row_count = 0
    last_heartbeat = time.time()

    for space in SPACES:
        bundle = bundles[space]
        split_rows = {
            "test": cap_rows(test_base, space, args.non_primary_regime_cap),
            "ood": cap_rows(ood_base, space, args.non_primary_regime_cap),
        }
        for support_budget in SUPPORT_BUDGETS:
            for arm in ARMS:
                rng = random.Random(460_000 + SPACES.index(space) * 10_000 + support_budget * 100 + ARMS.index(arm))
                for split, rows in split_rows.items():
                    for row in rows:
                        result = evaluate_arm(row, bundle, arm, support_budget, rng)
                        flat = {
                            "row_id": row["row_id"],
                            "seed": row["seed"],
                            "split": split,
                            "policy": arm,
                            "primitive_space": space,
                            "support_regime": row["support_regime"],
                            **result,
                        }
                        key = (space, support_budget, arm)
                        update_stats(stats[key], flat)
                        update_stats(regime_stats[(space, support_budget, arm, row["support_regime"])], flat)
                        update_stats(split_stats[(space, support_budget, arm, split)], flat)
                        update_stats(seed_stats[(space, support_budget, arm, row["seed"], row["support_regime"])], flat)
                        if space == PRIMARY_SPACE and support_budget == PRIMARY_BUDGET and arm == "FULL_MULTI_VIEW_ECF_POLICY" and split == "test":
                            scalar_failed = not flat["scalar_correct"]
                            update_binary_confusion(signal_confusions["entropy_failure_prediction"], flat["view_signals"]["entropy_high"], scalar_failed)
                            update_binary_confusion(signal_confusions["margin_failure_prediction"], flat["view_signals"]["margin_low"], scalar_failed)
                            update_binary_confusion(signal_confusions["collision_ambiguity_prediction"], flat["view_signals"]["collision_high"], scalar_failed)
                            update_binary_confusion(
                                signal_confusions["support_independence_precision_recall"],
                                flat["view_signals"]["support_independence_bad"],
                                flat["support_regime"] in {"CORRELATED_NOISE_SUPPORT", "MIXED_CLEAN_AND_CORRELATED"},
                            )
                            update_binary_confusion(
                                signal_confusions["counterfactual_need_prediction"],
                                flat["view_signals"]["counterfactual_needed"],
                                scalar_failed,
                            )
                            signal_rows.append(dict(flat["view_signals"]))
                        sample_key = (split, space, support_budget, arm, row["support_regime"])
                        if sample_counts[sample_key] < ROW_SAMPLE_LIMIT:
                            append_jsonl(out / f"row_outputs_{split}.jsonl", flat)
                            sample_counts[sample_key] += 1
                        row_count += 1
                        if time.time() - last_heartbeat >= args.heartbeat_sec:
                            append_progress(
                                out,
                                "streaming_eval_progress",
                                started,
                                {"space": space, "support_budget": support_budget, "arm": arm, "rows_evaluated": row_count},
                            )
                            write_json(
                                out / "partial_metrics_snapshot.json",
                                {
                                    "rows_evaluated": row_count,
                                    "last_space": space,
                                    "last_support_budget": support_budget,
                                    "last_arm": arm,
                                    "elapsed_sec": time.time() - started,
                                },
                            )
                            last_heartbeat = time.time()
                append_progress(
                    out,
                    "arm_budget_evaluated",
                    started,
                    {"space": space, "support_budget": support_budget, "arm": arm, "rows_evaluated": row_count},
                )

    policy_table, regime_table, split_table, seed_table = build_tables(stats, regime_stats, split_stats, seed_stats)
    add_accuracy_aliases(policy_table, regime_table)
    primary = policy_table[PRIMARY_SPACE][str(PRIMARY_BUDGET)]
    scalar = primary["SCALAR_ARGMAX_ONLY"]
    full = primary["FULL_MULTI_VIEW_ECF_POLICY"]
    random_control = primary["RANDOM_EXTRA_SUPPORT_CONTROL"]
    bad_control = primary["BAD_VIEW_CONTROL"]
    shuffled_control = primary["SHUFFLED_VECTOR_FIELD_CONTROL"]
    no_counter_control = primary["NO_COUNTERFACTUAL_CONTROL"]
    for arm, metric in primary.items():
        metric["robust_gain_vs_scalar_correlated"] = metric["correlated_accuracy"] - scalar["correlated_accuracy"]
        metric["robust_gain_vs_scalar_adversarial"] = metric["adversarial_accuracy"] - scalar["adversarial_accuracy"]
        metric["robust_gain_vs_random_extra_correlated"] = metric["correlated_accuracy"] - random_control["correlated_accuracy"]
        metric["clean_regression_vs_scalar"] = scalar["clean_accuracy"] - metric["clean_accuracy"]
        metric["failed_seed_count"] = 0

    best_single = max(SINGLE_VIEW_ARMS, key=lambda arm: primary[arm]["correlated_accuracy"] + primary[arm]["adversarial_accuracy"])
    best_pair_or_triple = max(PAIR_ARMS + TRIPLE_ARMS, key=lambda arm: primary[arm]["correlated_accuracy"] + primary[arm]["adversarial_accuracy"])
    full_beats_single_controls = all(
        full["correlated_accuracy"] > primary[arm]["correlated_accuracy"]
        and full["adversarial_accuracy"] >= primary[arm]["adversarial_accuracy"]
        for arm in SINGLE_VIEW_ARMS + CONTROL_ARMS
    )
    controls_worse = (
        full["correlated_accuracy"] > random_control["correlated_accuracy"]
        and full["correlated_accuracy"] > bad_control["correlated_accuracy"]
        and full["correlated_accuracy"] > shuffled_control["correlated_accuracy"]
        and full["correlated_accuracy"] > no_counter_control["correlated_accuracy"]
    )
    clean_regression = scalar["clean_accuracy"] - full["clean_accuracy"]
    support_ok = full["average_total_support_used"] <= D46_ROBUST_COMBINED_SUPPORT
    positive_gate = (
        full_beats_single_controls
        and controls_worse
        and full["clean_accuracy"] >= 0.995
        and full["correlated_accuracy"] >= 0.95
        and full["adversarial_accuracy"] >= 0.95
        and clean_regression <= 0.005
        and support_ok
    )
    if positive_gate:
        decision = {
            "decision": "multi_view_ecf_control_barrage_positive",
            "verdict": "D46X_MULTI_VIEW_ECF_CONTROL_BARRAGE_POSITIVE",
            "next": "D47_CELL_REFERENCE_DISCOVERY_WITH_ROBUST_SUPPORT",
        }
    elif full["clean_accuracy"] >= 0.995 and full["correlated_accuracy"] >= 0.95 and full["adversarial_accuracy"] >= 0.95:
        decision = {
            "decision": "multi_view_ecf_positive_high_cost",
            "verdict": "D46X_MULTI_VIEW_ECF_POSITIVE_HIGH_COST",
            "next": "D46Y_VIEW_COST_OPTIMIZATION",
        }
    elif primary[best_single]["correlated_accuracy"] >= full["correlated_accuracy"] and primary[best_single]["adversarial_accuracy"] >= full["adversarial_accuracy"]:
        decision = {
            "decision": "single_view_dominates_multiview_redundant",
            "verdict": "D46X_SINGLE_VIEW_DOMINATES",
            "next": "D47_WITH_DOMINANT_VIEW",
        }
    else:
        decision = {
            "decision": "multiview_ecf_not_robust",
            "verdict": "D46X_MULTI_VIEW_ECF_NOT_ROBUST",
            "next": "D46_REPAIR_SUPPORT_ROBUSTNESS",
        }
    decision["failed_jobs"] = []
    decision["boundary"] = "D46X only tests multi-view ECF control in controlled symbolic primitive discovery; no raw visual Raven reasoning, Raven solved, AGI, consciousness, architecture superiority, or literal-force claim."

    support_frontier = {arm: {str(budget): policy_table[PRIMARY_SPACE][str(budget)][arm] for budget in SUPPORT_BUDGETS} for arm in ARMS}
    primary_support_frontier = {arm: policy_table[PRIMARY_SPACE][str(PRIMARY_BUDGET)][arm] for arm in ARMS}
    diagnostics = {
        "scalar_argmax_failure_rate": scalar["scalar_argmax_failure_rate"],
        "vector_alignment_accuracy": primary["VECTOR_FIELD_ONLY"]["vector_alignment_accuracy"],
        "entropy_failure_prediction": finalize_confusion(signal_confusions["entropy_failure_prediction"]),
        "margin_stability_prediction": finalize_confusion(signal_confusions["margin_failure_prediction"]),
        "collision_ambiguity_prediction": finalize_confusion(signal_confusions["collision_ambiguity_prediction"]),
        "support_independence_precision_recall": finalize_confusion(signal_confusions["support_independence_precision_recall"]),
        "counterfactual_resolution_rate": full["counter_support_resolution_rate"],
    }
    ablations = {
        arm: {
            "accuracy_delta_full_minus_arm": full["accuracy"] - primary[arm]["accuracy"],
            "correlated_delta_full_minus_arm": full["correlated_accuracy"] - primary[arm]["correlated_accuracy"],
            "adversarial_delta_full_minus_arm": full["adversarial_accuracy"] - primary[arm]["adversarial_accuracy"],
            "support_delta_full_minus_arm": full["average_total_support_used"] - primary[arm]["average_total_support_used"],
        }
        for arm in ARMS
        if arm != "FULL_MULTI_VIEW_ECF_POLICY"
    }
    redundancy = signal_correlation(signal_rows)
    synergy = {
        "best_single_view": {"arm": best_single, **primary[best_single]},
        "best_pair_or_triple": {"arm": best_pair_or_triple, **primary[best_pair_or_triple]},
        "full_multiview": full,
        "full_beats_all_single_views_and_controls": full_beats_single_controls,
        "support_cost_at_or_below_d46_robust_combined": support_ok,
        "view_redundancy_interpretation": "Signals are complementary when off-diagonal correlations remain below 0.95 and ablation deltas differ by stress regime.",
    }
    stress = {
        "high_collision": regime_table[PRIMARY_SPACE][str(PRIMARY_BUDGET)]["FULL_MULTI_VIEW_ECF_POLICY"]["MIXED_CLEAN_AND_ADVERSARIAL"],
        "correlated_echo_support": regime_table[PRIMARY_SPACE][str(PRIMARY_BUDGET)]["FULL_MULTI_VIEW_ECF_POLICY"]["CORRELATED_NOISE_SUPPORT"],
        "adversarial_distractors": regime_table[PRIMARY_SPACE][str(PRIMARY_BUDGET)]["FULL_MULTI_VIEW_ECF_POLICY"]["ADVERSARIAL_DISTRACTOR_SUPPORT"],
        "ordered_alias_space": policy_table["ORDERED56_CONTROL"][str(PRIMARY_BUDGET)]["FULL_MULTI_VIEW_ECF_POLICY"],
        "support_budget_caps": {str(budget): policy_table[PRIMARY_SPACE][str(budget)]["FULL_MULTI_VIEW_ECF_POLICY"] for budget in SUPPORT_BUDGETS},
        "shuffled_field_control": shuffled_control,
        "bad_signal_control": bad_control,
    }
    aggregate = {
        "primary_space": PRIMARY_SPACE,
        "primary_support_budget": PRIMARY_BUDGET,
        "primary_policy_metrics": primary,
        "policy_space_budget_metrics": policy_table,
        "regime_by_policy_report": regime_table,
        "split_by_policy_report": split_table,
        "seed_by_policy_report": seed_table,
        "support_cost_frontier": support_frontier,
        "support_cost_frontier_primary_budget": primary_support_frontier,
        "view_diagnostics": diagnostics,
        "per_view_ablation_delta": ablations,
        "view_redundancy_correlation_matrix": redundancy,
        "view_synergy": synergy,
        "stress_tests": stress,
        "clean_regression_vs_scalar": clean_regression,
        "robust_gain_vs_scalar_correlated": full["correlated_accuracy"] - scalar["correlated_accuracy"],
        "robust_gain_vs_scalar_adversarial": full["adversarial_accuracy"] - scalar["adversarial_accuracy"],
        "robust_gain_vs_random_extra_correlated": full["correlated_accuracy"] - random_control["correlated_accuracy"],
        "controls_worse": controls_worse,
        "failed_jobs": [],
        "decision": decision,
    }
    reports = {
        "policy_comparison_report.json": policy_table,
        "regime_by_policy_report.json": regime_table,
        "primitive_space_by_policy_report.json": {space: policy_table[space] for space in SPACES},
        "support_cost_frontier_report.json": support_frontier,
        "view_diagnostics_report.json": diagnostics,
        "per_view_ablation_delta_report.json": ablations,
        "view_redundancy_correlation_matrix.json": redundancy,
        "view_synergy_report.json": synergy,
        "stress_test_report.json": stress,
        "control_report.json": {
            "random_extra": random_control,
            "bad_view": bad_control,
            "shuffled_vector_field": shuffled_control,
            "no_counterfactual": no_counter_control,
            "controls_worse": controls_worse,
        },
        "aggregate_metrics.json": aggregate,
        "decision.json": decision,
        "summary.json": {"decision": decision, "aggregate_metrics": aggregate},
    }
    for name, payload in reports.items():
        write_json(out / name, payload)
    write_report(out, decision, primary, synergy, primary_support_frontier)
    append_progress(out, "complete", started, {"decision": decision["decision"], "elapsed_sec": time.time() - started})
    write_json(out / "queue.json", {"task": "D46X multi-view ECF control barrage", "status": "complete", "decision": decision, "rows_evaluated": row_count, "elapsed_sec": time.time() - started})
    print(json.dumps({"out": str(out), "decision": decision["decision"], "verdict": decision["verdict"], "next": decision["next"], "rows_evaluated": row_count}, indent=2))


if __name__ == "__main__":
    main()
