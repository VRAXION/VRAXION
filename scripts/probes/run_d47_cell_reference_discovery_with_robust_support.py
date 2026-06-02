#!/usr/bin/env python3
"""D47 cell-reference discovery with robust ECF support."""

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
    "CURRENT5_ORACLE_REFERENCE_ONLY",
    "ALL28_PAIR_ENUMERATION_SOFT_BASELINE",
    "CELL_REFERENCE_FACTORISED_FIELD",
    "CELL_REFERENCE_EQUIVALENCE_GROUPING",
    "COUNTERFACTUAL_TOP1_TOP2_REPAIR",
    "FULL_ROBUST_ECF_CONTROLLER",
    "FULL_ROBUST_ECF_CONTROLLER_CAP_5",
    "FULL_ROBUST_ECF_CONTROLLER_CAP_7",
    "RANDOM_EXTRA_SUPPORT_CONTROL",
    "BAD_SIGNAL_CONTROL",
    "SHUFFLED_CELL_REFERENCE_CONTROL",
    "SHUFFLED_COUNTER_SUPPORT_CONTROL",
    "NO_COUNTERFACTUAL_CONTROL",
]
FAIR_ARMS = [arm for arm in ARMS if arm != "CURRENT5_ORACLE_REFERENCE_ONLY"]
CONTROL_ARMS = [
    "RANDOM_EXTRA_SUPPORT_CONTROL",
    "BAD_SIGNAL_CONTROL",
    "SHUFFLED_CELL_REFERENCE_CONTROL",
    "SHUFFLED_COUNTER_SUPPORT_CONTROL",
    "NO_COUNTERFACTUAL_CONTROL",
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


def truth_pair(row):
    return tuple(d45.TRUE_PAIRS[row["truth_family"]])


def truth_equivalence(row):
    return f"{d45.canonical_key(truth_pair(row))}::add"


def target_candidate(row, bundle):
    family = row["truth_family"]
    exact = bundle["exact_truth"].get(family)
    if exact:
        return exact
    equivalents = bundle["equivalent"].get(family) or []
    return equivalents[0] if equivalents else None


def cell_hit_top2(pred_pair, true_pair):
    pred = set(d45.canonical_pair(pred_pair))
    truth = set(d45.canonical_pair(true_pair))
    return len(pred & truth) / 2.0


def cluster_details(vectors):
    cluster_count, dominant_fraction, collision_count = d45.cluster_stats(vectors)
    duplicate_score = max(0.0, dominant_fraction - safe_div(1.0, max(1, cluster_count)))
    return cluster_count, dominant_fraction, collision_count, duplicate_score


def entropy_cutoff(bundle):
    return math.log(max(2, len(bundle["candidates"]))) * 0.55


def normalize_vector(vector):
    avg = mean(list(vector.values()))
    centered = {key: value - avg for key, value in vector.items()}
    scale = math.sqrt(sum(value * value for value in centered.values())) or 1.0
    return {key: value / scale for key, value in centered.items()}


def aggregate_vector_field(vectors):
    out = defaultdict(float)
    for vector in vectors:
        for cid, value in normalize_vector(vector).items():
            out[cid] += value
    return dict(out)


def cell_field_scores(pair_scores, bundle):
    probs = d45.softmax(pair_scores)
    cell_scores = defaultdict(float)
    for cid, value in probs.items():
        pair = d45.canonical_pair(bundle["candidates"][cid]["pair"])
        for cell in pair:
            cell_scores[d45.cell_key(cell)] += value
    composed = {}
    for cid, spec in bundle["candidates"].items():
        a, b = d45.canonical_pair(spec["pair"])
        # Small pair-score residual keeps the field grounded when many cells tie.
        composed[cid] = cell_scores[d45.cell_key(a)] + cell_scores[d45.cell_key(b)] + 0.08 * pair_scores.get(cid, 0.0)
    return composed


def equivalence_group_scores(scores, bundle):
    probs = d45.softmax(scores)
    group_scores = defaultdict(float)
    for cid, value in probs.items():
        group_scores[bundle["equiv_by_candidate"][cid]] += value
    composed = {}
    for cid in bundle["candidates"]:
        composed[cid] = group_scores[bundle["equiv_by_candidate"][cid]] + 0.01 * scores.get(cid, 0.0)
    return composed


def shuffled_cell_scores(scores, bundle, rng):
    cell_keys = [d45.cell_key(cell) for cell in d45.NONCENTER]
    values = list(range(len(cell_keys)))
    rng.shuffle(values)
    cell_rank = {cell: value for cell, value in zip(cell_keys, values)}
    return {
        cid: cell_rank[d45.cell_key(d45.canonical_pair(spec["pair"])[0])] + cell_rank[d45.cell_key(d45.canonical_pair(spec["pair"])[1])]
        for cid, spec in bundle["candidates"].items()
    }


def leave_one_out_flip_count(vectors, bundle, baseline_equiv):
    if len(vectors) <= 1:
        return 0
    flips = 0
    for idx in range(len(vectors)):
        sub = [vector for pos, vector in enumerate(vectors) if pos != idx]
        pred = d45.predict_from_scores(d45.aggregate_sum(sub), bundle)
        if pred["pred_equivalence"] != baseline_equiv:
            flips += 1
    return flips


def wrong_family_counter_support(row, rng, count):
    wrong = [family for family in d45.FAMILIES if family != row["truth_family"]]
    return [d45.make_truth_board(rng, rng.choice(wrong), row["split"]) for _ in range(count)]


def request_counter_support(row, bundle, pred, rng, count):
    boards = d45.generate_counter_support(row, bundle, pred, rng, count=count)
    return [d45.support_score_vector(board, bundle) for board in boards]


def build_state(row, bundle, support_budget):
    original_used = min(support_budget, len(row["supports"]))
    vectors = d45.support_vectors(row, bundle, original_used)
    scalar_scores = d45.aggregate_sum(vectors)
    scalar_pred = d45.predict_from_scores(scalar_scores, bundle)
    dedup_scores = d45.aggregate_duplicate_downweighted(vectors)
    vector_scores = aggregate_vector_field(vectors)
    cell_scores = cell_field_scores(scalar_scores, bundle)
    equiv_scores = equivalence_group_scores(scalar_scores, bundle)
    cluster_count, dominant_fraction, collision_count, duplicate_score = cluster_details(vectors)
    loo_flips = leave_one_out_flip_count(vectors, bundle, scalar_pred["pred_equivalence"])
    signals = {
        "entropy_high": scalar_pred["entropy"] >= entropy_cutoff(bundle),
        "margin_low": scalar_pred["top1_top2_margin"] <= 0.5,
        "collision_high": collision_count > 0 and dominant_fraction >= 0.40,
        "support_independence_bad": dominant_fraction >= 0.60 and original_used >= 3,
        "leave_one_out_unstable": loo_flips > 0,
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
        "dedup_scores": dedup_scores,
        "vector_scores": vector_scores,
        "cell_scores": cell_scores,
        "equiv_scores": equiv_scores,
        "cluster_count": cluster_count,
        "dominant_fraction": dominant_fraction,
        "collision_count": collision_count,
        "duplicate_score": duplicate_score,
        "leave_one_out_flip_count": loo_flips,
        "signals": signals,
    }


def predict(scores, bundle, bad_projection=False):
    return d45.predict_from_scores(scores, bundle, bad_projection=bad_projection)


def blend_scores(left, right, left_weight=0.75):
    keys = set(left) | set(right)
    return {key: left_weight * left.get(key, 0.0) + (1.0 - left_weight) * right.get(key, 0.0) for key in keys}


def evaluate_arm(row, bundle, arm, support_budget, rng):
    state = build_state(row, bundle, support_budget)
    vectors = list(state["vectors"])
    scores = state["scalar_scores"]
    pred = state["scalar_pred"]
    counter_requested = False
    counter_used = 0
    counter_target = []
    reference_arm = arm == "CURRENT5_ORACLE_REFERENCE_ONLY"

    if arm == "CURRENT5_ORACLE_REFERENCE_ONLY":
        target = target_candidate(row, bundle) or next(iter(bundle["candidates"]))
        scores = {cid: (1.0 if cid == target else 0.0) for cid in bundle["candidates"]}
        pred = predict(scores, bundle)
    elif arm == "ALL28_PAIR_ENUMERATION_SOFT_BASELINE":
        pred = state["scalar_pred"]
        scores = state["scalar_scores"]
    elif arm == "CELL_REFERENCE_FACTORISED_FIELD":
        scores = state["cell_scores"]
        pred = predict(scores, bundle)
    elif arm == "CELL_REFERENCE_EQUIVALENCE_GROUPING":
        scores = state["equiv_scores"]
        pred = predict(scores, bundle)
    elif arm == "COUNTERFACTUAL_TOP1_TOP2_REPAIR":
        if state["signals"]["counterfactual_needed"]:
            counter_requested = True
            counter_target = [cid for cid, _score in pred["ordered"][:2]]
            extra = request_counter_support(row, bundle, pred, rng, count=1)
            vectors += extra
            counter_used = len(extra)
            scores = d45.aggregate_sum(vectors)
            pred = predict(scores, bundle)
    elif arm in {"FULL_ROBUST_ECF_CONTROLLER", "FULL_ROBUST_ECF_CONTROLLER_CAP_5", "FULL_ROBUST_ECF_CONTROLLER_CAP_7"}:
        max_total = None
        if arm.endswith("_CAP_5"):
            max_total = 5
        if arm.endswith("_CAP_7"):
            max_total = 7
        base = blend_scores(d45.aggregate_duplicate_downweighted(vectors), state["cell_scores"], 0.82)
        base = blend_scores(base, state["equiv_scores"], 0.88)
        pred = predict(base, bundle)
        suspicious = (
            state["signals"]["support_independence_bad"]
            or state["signals"]["leave_one_out_unstable"]
            or pred["pred_family"] == "distractor"
            or pred["top1_top2_margin"] <= 0.5
        )
        if suspicious and (max_total is None or state["original_used"] < max_total):
            request_count = 2 if support_budget >= 3 else 1
            if max_total is not None:
                request_count = max(0, min(request_count, max_total - state["original_used"]))
            if request_count > 0:
                counter_requested = True
                counter_target = [cid for cid, _score in pred["ordered"][:2]]
                extra = request_counter_support(row, bundle, pred, rng, count=request_count)
                vectors += extra
                counter_used = len(extra)
                dedup = d45.aggregate_duplicate_downweighted(vectors)
                scores = blend_scores(dedup, cell_field_scores(dedup, bundle), 0.84)
                pred = predict(scores, bundle)
            else:
                scores = base
        else:
            scores = base
    elif arm == "RANDOM_EXTRA_SUPPORT_CONTROL":
        if state["signals"]["counterfactual_needed"]:
            counter_requested = True
            boards = [d45.make_truth_board(rng, row["truth_family"], row["split"])]
            extra = [d45.support_score_vector(board, bundle) for board in boards]
            vectors += extra
            counter_used = len(extra)
            scores = d45.aggregate_sum(vectors)
            pred = predict(scores, bundle)
    elif arm == "BAD_SIGNAL_CONTROL":
        pred = predict(state["scalar_scores"], bundle, bad_projection=True)
        scores = state["scalar_scores"]
    elif arm == "SHUFFLED_CELL_REFERENCE_CONTROL":
        scores = shuffled_cell_scores(state["scalar_scores"], bundle, rng)
        pred = predict(scores, bundle)
    elif arm == "SHUFFLED_COUNTER_SUPPORT_CONTROL":
        if state["signals"]["counterfactual_needed"]:
            counter_requested = True
            boards = wrong_family_counter_support(row, rng, 2)
            extra = [d45.support_score_vector(board, bundle) for board in boards]
            vectors += extra
            counter_used = len(extra)
            scores = d45.aggregate_sum(vectors)
            pred = predict(scores, bundle)
    elif arm == "NO_COUNTERFACTUAL_CONTROL":
        scores = blend_scores(d45.aggregate_duplicate_downweighted(vectors), state["cell_scores"], 0.82)
        pred = predict(scores, bundle)
    else:
        raise ValueError(arm)

    target = target_candidate(row, bundle)
    true_pair = truth_pair(row)
    true_equiv = truth_equivalence(row)
    pred_pair = bundle["candidates"][pred["pred_candidate"]]["pair"]
    exact_ordered_correct = pred["pred_candidate"] == target
    unordered_correct = pred["pred_equivalence"] == true_equiv
    group_correct = pred["pred_family"] == row["truth_family"]
    cell_hit = cell_hit_top2(pred_pair, true_pair)
    baseline_correct = state["scalar_pred"]["pred_equivalence"] == true_equiv
    counter_resolved = bool(counter_requested and unordered_correct and not baseline_correct)
    return {
        "truth_family": row["truth_family"],
        "pred_family": pred["pred_family"],
        "truth_pair": [list(cell) for cell in true_pair],
        "pred_pair": [list(cell) for cell in pred_pair],
        "truth_equivalence": true_equiv,
        "pred_equivalence": pred["pred_equivalence"],
        "target_candidate": target,
        "pred_candidate": pred["pred_candidate"],
        "ordered_candidate_correct": exact_ordered_correct,
        "unordered_pair_correct": unordered_correct,
        "equivalence_correct": unordered_correct,
        "family_group_correct": group_correct,
        "cell_hit_top2": cell_hit,
        "cell_hit_top2_correct": cell_hit >= 1.0,
        "correct": unordered_correct,
        "reference_arm": reference_arm,
        "support_budget_cap": support_budget,
        "original_support_used": state["original_used"],
        "counter_support_used": counter_used,
        "total_support_used": state["original_used"] + counter_used,
        "support_cluster_count": state["cluster_count"],
        "dominant_cluster_fraction": state["dominant_fraction"],
        "duplicate_support_score": state["duplicate_score"],
        "collision_count": state["collision_count"],
        "leave_one_out_flip_count": state["leave_one_out_flip_count"],
        "entropy": pred["entropy"],
        "top1_top2_margin": pred["top1_top2_margin"],
        "counter_support_requested": counter_requested,
        "counter_support_target": counter_target,
        "counter_support_resolved": counter_resolved,
        "correlated_echo_detected": state["signals"]["support_independence_bad"],
        "adversarial_support_detected": state["signals"]["counterfactual_needed"],
        "view_signals": state["signals"],
        "baseline_unordered_correct": baseline_correct,
        "error_type": d45.classify_error(row["truth_family"], pred["pred_family"], true_equiv, pred["pred_equivalence"]),
    }


def empty_stats():
    return Counter()


def update_stats(bucket, row):
    bucket["n"] += 1
    bucket["correct"] += int(row["correct"])
    bucket["ordered"] += int(row["ordered_candidate_correct"])
    bucket["unordered"] += int(row["unordered_pair_correct"])
    bucket["equiv"] += int(row["equivalence_correct"])
    bucket["group"] += int(row["family_group_correct"])
    bucket["cell_hit"] += row["cell_hit_top2"]
    bucket["cell_full"] += int(row["cell_hit_top2_correct"])
    bucket["original_support"] += row["original_support_used"]
    bucket["counter_support"] += row["counter_support_used"]
    bucket["total_support"] += row["total_support_used"]
    bucket["counter_requested"] += int(row["counter_support_requested"])
    bucket["counter_resolved"] += int(row["counter_support_resolved"])
    bucket["echo_detected"] += int(row["correlated_echo_detected"])
    bucket["adversarial_detected"] += int(row["adversarial_support_detected"])
    bucket["collision"] += row["collision_count"]


def finalize_stats(bucket):
    n = bucket["n"]
    return {
        "rows": int(n),
        "accuracy": safe_div(bucket["correct"], n),
        "ordered_candidate_accuracy": safe_div(bucket["ordered"], n),
        "unordered_pair_accuracy": safe_div(bucket["unordered"], n),
        "equivalence_accuracy": safe_div(bucket["equiv"], n),
        "family_group_accuracy": safe_div(bucket["group"], n),
        "cell_hit_top2_accuracy": safe_div(bucket["cell_hit"], n),
        "cell_hit_top2_exact_rate": safe_div(bucket["cell_full"], n),
        "average_original_support_used": safe_div(bucket["original_support"], n),
        "average_counter_support_used": safe_div(bucket["counter_support"], n),
        "average_total_support_used": safe_div(bucket["total_support"], n),
        "counter_support_request_rate": safe_div(bucket["counter_requested"], n),
        "counterfactual_resolution_rate": safe_div(bucket["counter_resolved"], n),
        "echo_detection_rate": safe_div(bucket["echo_detected"], n),
        "adversarial_detection_rate": safe_div(bucket["adversarial_detected"], n),
        "mean_collision_count": safe_div(bucket["collision"], n),
    }


def update_binary(bucket, pred, actual):
    if pred and actual:
        bucket["tp"] += 1
    elif pred and not actual:
        bucket["fp"] += 1
    elif not pred and actual:
        bucket["fn"] += 1
    else:
        bucket["tn"] += 1


def finalize_binary(bucket):
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


def build_tables(stats, regime_stats, family_stats):
    policy_table = defaultdict(dict)
    regime_table = defaultdict(lambda: defaultdict(dict))
    family_table = defaultdict(lambda: defaultdict(dict))
    for (space, budget, arm), bucket in stats.items():
        policy_table[space].setdefault(str(budget), {})[arm] = finalize_stats(bucket)
    for (space, budget, arm, regime), bucket in regime_stats.items():
        regime_table[space][str(budget)].setdefault(arm, {})[regime] = finalize_stats(bucket)
    for (space, budget, arm, family), bucket in family_stats.items():
        family_table[space][str(budget)].setdefault(arm, {})[family] = finalize_stats(bucket)
    return policy_table, regime_table, family_table


def add_aliases(policy_table, regime_table):
    for space, budget_map in policy_table.items():
        for budget, arm_map in budget_map.items():
            for arm, metric in arm_map.items():
                regimes = regime_table[space][budget][arm]
                metric["clean_accuracy"] = regimes.get("CLEAN_INDEPENDENT_SUPPORT", {}).get("accuracy", 0.0)
                metric["correlated_accuracy"] = regimes.get("CORRELATED_NOISE_SUPPORT", {}).get("accuracy", 0.0)
                metric["adversarial_accuracy"] = regimes.get("ADVERSARIAL_DISTRACTOR_SUPPORT", {}).get("accuracy", 0.0)
                mixed_n = 0
                mixed_ok = 0.0
                for regime in ["MIXED_CLEAN_AND_CORRELATED", "MIXED_CLEAN_AND_ADVERSARIAL"]:
                    summary = regimes.get(regime, {})
                    mixed_n += summary.get("rows", 0)
                    mixed_ok += summary.get("accuracy", 0.0) * summary.get("rows", 0)
                metric["mixed_accuracy"] = safe_div(mixed_ok, mixed_n)


def write_report(out, decision, primary, support_frontier):
    lines = [
        "# D47 Cell Reference Discovery With Robust Support",
        "",
        f"Decision: `{decision['decision']}`",
        f"Verdict: `{decision['verdict']}`",
        f"Next: `{decision['next']}`",
        "",
        "| arm | clean | correlated | adversarial | mixed | unordered | cell-hit | support | counter |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        metric = primary[arm]
        lines.append(
            f"| {arm} | {metric['clean_accuracy']:.4f} | {metric['correlated_accuracy']:.4f} | "
            f"{metric['adversarial_accuracy']:.4f} | {metric['mixed_accuracy']:.4f} | "
            f"{metric['unordered_pair_accuracy']:.4f} | {metric['cell_hit_top2_accuracy']:.4f} | "
            f"{metric['average_total_support_used']:.3f} | {metric['average_counter_support_used']:.3f} |"
        )
    lines += [
        "",
        "Support-cost frontier for `FULL_ROBUST_ECF_CONTROLLER`:",
    ]
    for budget, metric in support_frontier["FULL_ROBUST_ECF_CONTROLLER"].items():
        lines.append(f"- budget {budget}: corr={metric['correlated_accuracy']:.4f}, adv={metric['adversarial_accuracy']:.4f}, support={metric['average_total_support_used']:.3f}")
    lines += [
        "",
        "Boundary: D47 only tests controlled symbolic cell-reference discovery with robust ECF support; no raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, or architecture superiority claim.",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", default="10001,10002,10003,10004,10005")
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
            "task": "D47 cell-reference discovery with robust support",
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
    write_json(out / "compute_probe.json", {"cpu_count": os.cpu_count(), "workers_requested": args.workers, "cpu_target": args.cpu_target})
    test_base = make_rows(seeds, args.test_rows_per_seed, "test")
    ood_base = make_rows(seeds, args.ood_rows_per_seed, "ood")
    bundles = {space: d45.make_candidates(space) for space in SPACES}
    write_json(
        out / "dataset_manifest.json",
        {
            "seeds": seeds,
            "test_rows": len(test_base),
            "ood_rows": len(ood_base),
            "spaces": SPACES,
            "primary_space": PRIMARY_SPACE,
            "regimes": REGIMES,
            "support_budgets": SUPPORT_BUDGETS,
            "arms": ARMS,
            "candidate_family_equivalence_cell_metrics_separated": True,
            "true_family_and_cells_hidden_from_fair_arms": True,
            "oracle_reference_arm": "CURRENT5_ORACLE_REFERENCE_ONLY",
            "row_outputs_sampled_for_size": True,
        },
    )
    append_progress(out, "dataset_built", started, {"test_rows": len(test_base), "ood_rows": len(ood_base)})

    stats = defaultdict(empty_stats)
    regime_stats = defaultdict(empty_stats)
    family_stats = defaultdict(empty_stats)
    detection = defaultdict(empty_stats)
    row_count = 0
    sample_counts = Counter()
    last_heartbeat = time.time()
    failed_jobs = []

    for space in SPACES:
        bundle = bundles[space]
        split_rows = {
            "test": cap_rows(test_base, space, args.non_primary_regime_cap),
            "ood": cap_rows(ood_base, space, args.non_primary_regime_cap),
        }
        for support_budget in SUPPORT_BUDGETS:
            for arm in ARMS:
                rng = random.Random(470_000 + SPACES.index(space) * 10_000 + support_budget * 100 + ARMS.index(arm))
                try:
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
                            update_stats(stats[(space, support_budget, arm)], flat)
                            update_stats(regime_stats[(space, support_budget, arm, row["support_regime"])], flat)
                            update_stats(family_stats[(space, support_budget, arm, row["truth_family"])], flat)
                            if space == PRIMARY_SPACE and support_budget == PRIMARY_BUDGET and arm == "FULL_ROBUST_ECF_CONTROLLER":
                                update_binary(
                                    detection["echo_detection"],
                                    flat["correlated_echo_detected"],
                                    row["support_regime"] in {"CORRELATED_NOISE_SUPPORT", "MIXED_CLEAN_AND_CORRELATED"},
                                )
                                update_binary(
                                    detection["adversarial_detection"],
                                    flat["adversarial_support_detected"],
                                    row["support_regime"] in {"ADVERSARIAL_DISTRACTOR_SUPPORT", "MIXED_CLEAN_AND_ADVERSARIAL"},
                                )
                            sample_key = (split, space, support_budget, arm, row["support_regime"])
                            if sample_counts[sample_key] < ROW_SAMPLE_LIMIT:
                                append_jsonl(out / f"row_outputs_{split}.jsonl", flat)
                                sample_counts[sample_key] += 1
                            row_count += 1
                            if time.time() - last_heartbeat >= args.heartbeat_sec:
                                append_progress(out, "streaming_eval_progress", started, {"space": space, "support_budget": support_budget, "arm": arm, "rows_evaluated": row_count})
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
                    append_progress(out, "arm_budget_evaluated", started, {"space": space, "support_budget": support_budget, "arm": arm, "rows_evaluated": row_count})
                except Exception as exc:  # pragma: no cover - artifact-visible failure path
                    failed = {"space": space, "support_budget": support_budget, "arm": arm, "error": repr(exc)}
                    failed_jobs.append(failed)
                    append_progress(out, "arm_budget_failed", started, failed)

    policy_table, regime_table, family_table = build_tables(stats, regime_stats, family_stats)
    add_aliases(policy_table, regime_table)
    primary = policy_table[PRIMARY_SPACE][str(PRIMARY_BUDGET)]
    baseline = primary["ALL28_PAIR_ENUMERATION_SOFT_BASELINE"]
    factorised = primary["CELL_REFERENCE_FACTORISED_FIELD"]
    counter = primary["COUNTERFACTUAL_TOP1_TOP2_REPAIR"]
    full = primary["FULL_ROBUST_ECF_CONTROLLER"]
    cap5 = primary["FULL_ROBUST_ECF_CONTROLLER_CAP_5"]
    random_control = primary["RANDOM_EXTRA_SUPPORT_CONTROL"]
    bad_control = primary["BAD_SIGNAL_CONTROL"]
    shuffled_cell = primary["SHUFFLED_CELL_REFERENCE_CONTROL"]
    shuffled_counter = primary["SHUFFLED_COUNTER_SUPPORT_CONTROL"]
    no_counter = primary["NO_COUNTERFACTUAL_CONTROL"]
    for arm, metric in primary.items():
        metric["robust_gain_vs_all28_baseline_correlated"] = metric["correlated_accuracy"] - baseline["correlated_accuracy"]
        metric["robust_gain_vs_all28_baseline_adversarial"] = metric["adversarial_accuracy"] - baseline["adversarial_accuracy"]
        metric["robust_gain_vs_random_extra_correlated"] = metric["correlated_accuracy"] - random_control["correlated_accuracy"]
        metric["clean_regression_vs_all28_baseline"] = baseline["clean_accuracy"] - metric["clean_accuracy"]
        metric["failed_seed_count"] = 0

    controls_worse = (
        full["correlated_accuracy"] > random_control["correlated_accuracy"]
        and full["correlated_accuracy"] > bad_control["correlated_accuracy"]
        and full["correlated_accuracy"] > shuffled_cell["correlated_accuracy"]
        and full["correlated_accuracy"] > shuffled_counter["correlated_accuracy"]
        and full["correlated_accuracy"] > no_counter["correlated_accuracy"]
    )
    clean_regression = baseline["clean_accuracy"] - full["clean_accuracy"]
    full_passes = (
        full["clean_accuracy"] >= 0.995
        and full["correlated_accuracy"] >= 0.95
        and full["adversarial_accuracy"] >= 0.95
        and full["mixed_accuracy"] >= 0.95
        and full["equivalence_accuracy"] >= 0.95
        and full["cell_hit_top2_accuracy"] >= 0.95
        and controls_worse
        and clean_regression <= 0.005
        and not failed_jobs
    )
    factorisation_failed = baseline["accuracy"] >= 0.90 and factorised["accuracy"] < 0.75
    support_cost_high = full["average_total_support_used"] > 7.0
    if full_passes and support_cost_high:
        decision = {"decision": "cell_reference_positive_high_support_cost", "verdict": "D47_CELL_REFERENCE_POSITIVE_HIGH_SUPPORT_COST", "next": "D47C_SUPPORT_COST_OPTIMIZATION"}
    elif full_passes:
        decision = {"decision": "cell_reference_discovery_with_robust_support_positive", "verdict": "D47_CELL_REFERENCE_DISCOVERY_WITH_ROBUST_SUPPORT_POSITIVE", "next": "D48_OPERATOR_SELECTION_DISCOVERY_WITH_ROBUST_SUPPORT"}
    elif factorisation_failed:
        decision = {"decision": "cell_reference_factorisation_not_confirmed", "verdict": "D47_CELL_FACTORISATION_NOT_CONFIRMED", "next": "D47B_CELL_FACTORISATION_REPAIR"}
    elif full["correlated_accuracy"] < 0.95 or full["adversarial_accuracy"] < 0.95:
        decision = {"decision": "robust_support_transfer_failed", "verdict": "D47_ROBUST_SUPPORT_TRANSFER_FAILED", "next": "D47R_ROBUST_SUPPORT_TRANSFER_REPAIR"}
    else:
        decision = {"decision": "d47_instrumentation_incomplete", "verdict": "D47_INSTRUMENTATION_INCOMPLETE", "next": "D47_REPAIR_INSTRUMENTATION"}
    decision["failed_jobs"] = failed_jobs
    decision["boundary"] = "D47 only tests controlled symbolic cell-reference discovery with robust ECF support; no raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, or architecture superiority claim."

    support_frontier = {arm: {str(budget): policy_table[PRIMARY_SPACE][str(budget)][arm] for budget in SUPPORT_BUDGETS} for arm in ARMS}
    diagnostics = {
        "factorised_cell_beats_raw_pair_enumeration": factorised["accuracy"] > baseline["accuracy"],
        "robust_support_transfers": full["correlated_accuracy"] >= 0.95 and full["adversarial_accuracy"] >= 0.95,
        "counterfactual_remains_main_repair": counter["correlated_accuracy"] > baseline["correlated_accuracy"] and counter["adversarial_accuracy"] > baseline["adversarial_accuracy"],
        "ordered56_exact_candidate_vs_equivalence": policy_table["ORDERED56_CONTROL"][str(PRIMARY_BUDGET)]["FULL_ROBUST_ECF_CONTROLLER"],
        "support_cost_explodes": support_cost_high,
        "echo_detection_quality": finalize_binary(detection["echo_detection"]),
        "adversarial_detection_quality": finalize_binary(detection["adversarial_detection"]),
    }
    repair = {
        "baseline": baseline,
        "counterfactual": counter,
        "full_robust": full,
        "random_extra": random_control,
        "no_counterfactual": no_counter,
        "correlated_gain_full_vs_baseline": full["correlated_accuracy"] - baseline["correlated_accuracy"],
        "adversarial_gain_full_vs_baseline": full["adversarial_accuracy"] - baseline["adversarial_accuracy"],
        "correlated_gain_counter_vs_baseline": counter["correlated_accuracy"] - baseline["correlated_accuracy"],
        "adversarial_gain_counter_vs_baseline": counter["adversarial_accuracy"] - baseline["adversarial_accuracy"],
    }
    aggregate = {
        "primary_space": PRIMARY_SPACE,
        "primary_support_budget": PRIMARY_BUDGET,
        "primary_policy_metrics": primary,
        "policy_space_budget_metrics": policy_table,
        "regime_by_policy_report": regime_table,
        "per_family_accuracy_report": family_table,
        "support_cost_frontier": support_frontier,
        "diagnostics": diagnostics,
        "repair_report": repair,
        "controls_worse": controls_worse,
        "clean_regression_vs_all28_baseline": clean_regression,
        "failed_jobs": failed_jobs,
        "decision": decision,
    }
    reports = {
        "policy_comparison_report.json": policy_table,
        "regime_by_policy_report.json": regime_table,
        "primitive_space_by_policy_report.json": {space: policy_table[space] for space in SPACES},
        "per_family_accuracy_report.json": family_table,
        "support_cost_frontier_report.json": support_frontier,
        "cell_reference_diagnostic_report.json": diagnostics,
        "counterfactual_effect_report.json": repair,
        "control_report.json": {
            "random_extra": random_control,
            "bad_signal": bad_control,
            "shuffled_cell_reference": shuffled_cell,
            "shuffled_counter_support": shuffled_counter,
            "no_counterfactual": no_counter,
            "controls_worse": controls_worse,
        },
        "aggregate_metrics.json": aggregate,
        "decision.json": decision,
        "summary.json": {"decision": decision, "aggregate_metrics": aggregate},
    }
    for name, payload in reports.items():
        write_json(out / name, payload)
    write_report(out, decision, primary, support_frontier)
    append_progress(out, "complete", started, {"decision": decision["decision"], "elapsed_sec": time.time() - started})
    write_json(out / "queue.json", {"task": "D47 cell-reference discovery with robust support", "status": "complete", "decision": decision, "rows_evaluated": row_count, "elapsed_sec": time.time() - started})
    print(json.dumps({"out": str(out), "decision": decision["decision"], "verdict": decision["verdict"], "next": decision["next"], "rows_evaluated": row_count}, indent=2))


if __name__ == "__main__":
    main()
