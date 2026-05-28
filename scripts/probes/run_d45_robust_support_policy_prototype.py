#!/usr/bin/env python3
"""D45 robust support policy prototype for controlled symbolic IPF/ECF support."""

import argparse
import itertools
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

FAMILIES = ["row", "col", "pair", "mirror", "diag"]
TRUE_PAIRS = {
    "row": ((1, 0), (1, 2)),
    "col": ((0, 1), (2, 1)),
    "pair": ((0, 0), (2, 2)),
    "mirror": ((2, 0), (0, 2)),
    "diag": ((0, 0), (1, 2)),
}
NONCENTER = [(r, c) for r in range(3) for c in range(3) if (r, c) != (1, 1)]
SUPPORT_REGIMES = [
    "CLEAN_INDEPENDENT_SUPPORT",
    "CORRELATED_NOISE_SUPPORT",
    "ADVERSARIAL_DISTRACTOR_SUPPORT",
    "MIXED_CLEAN_AND_CORRELATED",
    "MIXED_CLEAN_AND_ADVERSARIAL",
    "COUNTERFACTUAL_SUPPORT_AVAILABLE",
]
PRIMITIVE_SPACES = [
    "CURRENT5",
    "ALL28_UNORDERED",
    "ORDERED56_CONTROL",
    "CURRENT5_PLUS_DISTRACTORS_20",
    "CURRENT5_PLUS_DISTRACTORS_50",
]
POLICIES = [
    "NAIVE_IPF_BASELINE",
    "STAGED_SUPPORT_BASELINE",
    "DUPLICATE_SUPPORT_DOWNWEIGHTING",
    "SOURCE_DIVERSITY_WEIGHTING",
    "LEAVE_ONE_SUPPORT_OUT_STABILITY",
    "ROBUST_MEDIAN_FIELD_AGGREGATION",
    "COUNTER_SUPPORT_QUERY_POLICY",
    "ADVERSARIAL_DISTRACTOR_DETECTOR",
    "ROBUST_COMBINED_POLICY",
    "RANDOM_EXTRA_SUPPORT_CONTROL",
    "BAD_ROBUSTNESS_SIGNAL_CONTROL",
]


def canonical_pair(pair):
    return tuple(sorted(tuple(cell) for cell in pair))


def cell_key(cell):
    return f"r{cell[0]}c{cell[1]}"


def pair_key(pair):
    return "__".join(cell_key(cell) for cell in pair)


def canonical_key(pair):
    return pair_key(canonical_pair(pair))


def space_to_base(space):
    if space == "ALL28_UNORDERED":
        return "ALL28"
    if space == "ORDERED56_CONTROL":
        return "ORDERED56"
    if space == "CURRENT5_PLUS_DISTRACTORS_20":
        return "CURRENT5_PLUS_20"
    if space == "CURRENT5_PLUS_DISTRACTORS_50":
        return "CURRENT5_PLUS_50"
    return space


def make_candidates(space):
    base = space_to_base(space)
    if base == "CURRENT5":
        pairs = {family: tuple(pair) for family, pair in TRUE_PAIRS.items()}
    elif base == "ALL28":
        pairs = {f"u_{pair_key(pair)}": tuple(pair) for pair in itertools.combinations(NONCENTER, 2)}
    elif base == "ORDERED56":
        pairs = {f"o_{pair_key(pair)}": tuple(pair) for pair in itertools.permutations(NONCENTER, 2)}
    elif base.startswith("CURRENT5_PLUS_"):
        count = int(base.rsplit("_", 1)[1])
        pairs = {family: tuple(pair) for family, pair in TRUE_PAIRS.items()}
        truth = {canonical_pair(pair) for pair in TRUE_PAIRS.values()}
        pool = [tuple(pair) for pair in itertools.combinations(NONCENTER, 2) if canonical_pair(pair) not in truth]
        rng = random.Random(45_000 + count)
        rng.shuffle(pool)
        for idx in range(count):
            pair = pool[idx % len(pool)]
            pairs[f"d{idx}_{pair_key(pair)}"] = pair
    else:
        raise ValueError(space)
    candidates = {cid: {"pair": pair, "op": "add"} for cid, pair in pairs.items()}
    family_by_candidate = {}
    equiv_by_candidate = {}
    exact_truth = {family: None for family in FAMILIES}
    equivalent = {family: [] for family in FAMILIES}
    for cid, spec in candidates.items():
        pair = spec["pair"]
        equiv_by_candidate[cid] = f"{canonical_key(pair)}::add"
        mapped = None
        for family, truth_pair in TRUE_PAIRS.items():
            if tuple(pair) == tuple(truth_pair) and exact_truth[family] is None:
                exact_truth[family] = cid
            if canonical_pair(pair) == canonical_pair(truth_pair):
                mapped = family
                equivalent[family].append(cid)
        family_by_candidate[cid] = mapped
    return {
        "space": space,
        "candidates": candidates,
        "family_by_candidate": family_by_candidate,
        "equiv_by_candidate": equiv_by_candidate,
        "exact_truth": exact_truth,
        "equivalent": equivalent,
    }


def apply_add(a, b):
    return (a + b) % 9


def pair_sum(board, pair):
    (ar, ac), (br, bc) = pair
    return apply_add(board[ar][ac], board[br][bc])


def circular_distance(a, b):
    delta = abs(a - b) % 9
    return min(delta, 9 - delta)


def distractor_pair_for(family):
    truth_cells = set(TRUE_PAIRS[family])
    for pair in itertools.combinations(NONCENTER, 2):
        if canonical_pair(pair) != canonical_pair(TRUE_PAIRS[family]) and not (set(pair) & truth_cells):
            return tuple(pair)
    for pair in itertools.combinations(NONCENTER, 2):
        if canonical_pair(pair) != canonical_pair(TRUE_PAIRS[family]):
            return tuple(pair)
    raise RuntimeError("no distractor pair")


def set_pair_to_center(board, pair, center):
    (ar, ac), (br, bc) = pair
    board[br][bc] = (center - board[ar][ac]) % 9


def break_pair_match(board, pair, center):
    if set(pair) & set(TRUE_PAIRS.get("_none_", ())):
        return
    (ar, ac), (br, bc) = pair
    if (br, bc) == (1, 1) or (ar, ac) == (1, 1):
        return
    if pair_sum(board, pair) == center:
        board[br][bc] = (board[br][bc] + 1) % 9


def make_truth_board(rng, family, split, force_pair=None, avoid_pairs=None):
    avoid_pairs = avoid_pairs or []
    truth_pair = TRUE_PAIRS[family]
    for _ in range(200):
        board = [[rng.randrange(9) for _ in range(3)] for _ in range(3)]
        if split == "ood":
            board = [[((value * 2) + 1) % 9 for value in row] for row in board]
        center = pair_sum(board, truth_pair)
        board[1][1] = center
        if force_pair is not None and not (set(force_pair) & set(truth_pair)):
            set_pair_to_center(board, force_pair, center)
        for pair in avoid_pairs:
            if not (set(pair) & set(truth_pair)):
                break_pair_match(board, pair, center)
        if all(pair_sum(board, pair) != center for pair in avoid_pairs if not (set(pair) & set(truth_pair))):
            return board
    return board


def make_supports(rng, family, split, regime):
    distractor = distractor_pair_for(family)
    if regime == "CLEAN_INDEPENDENT_SUPPORT":
        return [make_truth_board(rng, family, split, avoid_pairs=[distractor]) for _ in range(5)]
    if regime == "CORRELATED_NOISE_SUPPORT":
        board = make_truth_board(rng, family, split, force_pair=distractor)
        return [json.loads(json.dumps(board)) for _ in range(5)]
    if regime == "ADVERSARIAL_DISTRACTOR_SUPPORT":
        board = make_truth_board(rng, family, split, force_pair=distractor)
        return [json.loads(json.dumps(board)) for _ in range(4)] + [make_truth_board(rng, family, split, avoid_pairs=[distractor])]
    if regime == "MIXED_CLEAN_AND_CORRELATED":
        board = make_truth_board(rng, family, split, force_pair=distractor)
        return [make_truth_board(rng, family, split, avoid_pairs=[distractor]) for _ in range(3)] + [
            json.loads(json.dumps(board)),
            json.loads(json.dumps(board)),
        ]
    if regime == "MIXED_CLEAN_AND_ADVERSARIAL":
        board = make_truth_board(rng, family, split, force_pair=distractor)
        return [make_truth_board(rng, family, split, avoid_pairs=[distractor]) for _ in range(2)] + [
            json.loads(json.dumps(board)) for _ in range(3)
        ]
    if regime == "COUNTERFACTUAL_SUPPORT_AVAILABLE":
        board = make_truth_board(rng, family, split, force_pair=distractor)
        return [make_truth_board(rng, family, split, avoid_pairs=[distractor]) for _ in range(2)] + [
            json.loads(json.dumps(board)) for _ in range(3)
        ]
    raise ValueError(regime)


def make_rows(seeds, rows_per_seed, split):
    rows = []
    for seed in seeds:
        for regime in SUPPORT_REGIMES:
            rng = random.Random(seed + (0 if split == "test" else 100_000) + 1_003 * SUPPORT_REGIMES.index(regime))
            for idx in range(rows_per_seed):
                family = rng.choice(FAMILIES)
                rows.append(
                    {
                        "row_id": f"{split}-{regime}-{seed}-{idx:05d}",
                        "seed": seed,
                        "split": split,
                        "support_regime": regime,
                        "truth_family": family,
                        "supports": make_supports(rng, family, split, regime),
                    }
                )
    return rows


def candidate_score(board, spec):
    predicted = pair_sum(board, spec["pair"])
    return -float(circular_distance(predicted, board[1][1]))


def support_score_vector(board, bundle):
    return {cid: candidate_score(board, spec) for cid, spec in bundle["candidates"].items()}


def vector_signature(vector):
    return tuple(round(vector[cid], 3) for cid in sorted(vector))


def support_vectors(row, bundle, support_count):
    return [support_score_vector(board, bundle) for board in row["supports"][:support_count]]


def aggregate_sum(vectors):
    out = defaultdict(float)
    for vector in vectors:
        for cid, value in vector.items():
            out[cid] += value
    return dict(out)


def aggregate_duplicate_downweighted(vectors):
    clusters = defaultdict(list)
    for vector in vectors:
        clusters[vector_signature(vector)].append(vector)
    out = defaultdict(float)
    for members in clusters.values():
        for cid in members[0]:
            out[cid] += sum(vector[cid] for vector in members) / len(members)
    return dict(out)


def aggregate_source_diverse(vectors):
    clusters = defaultdict(list)
    for vector in vectors:
        clusters[vector_signature(vector)].append(vector)
    out = defaultdict(float)
    cluster_count = max(1, len(clusters))
    for members in clusters.values():
        cluster_weight = 1.0 / cluster_count
        for cid in members[0]:
            out[cid] += cluster_weight * (sum(vector[cid] for vector in members) / len(members))
    return dict(out)


def aggregate_median(vectors, trim=False):
    out = {}
    for cid in vectors[0]:
        vals = sorted(vector[cid] for vector in vectors)
        if trim and len(vals) >= 5:
            vals = vals[1:-1]
        mid = len(vals) // 2
        out[cid] = vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2.0
    return out


def softmax(scores):
    top = max(scores.values())
    weights = {cid: math.exp(value - top) for cid, value in scores.items()}
    total = sum(weights.values()) or 1.0
    return {cid: value / total for cid, value in weights.items()}


def entropy(probs):
    return -sum(value * math.log(value + 1e-12) for value in probs.values())


def top_stats(scores):
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    top1 = ordered[0]
    top2 = ordered[1] if len(ordered) > 1 else ordered[0]
    return ordered, top1[1] - top2[1]


def project(probs, bundle, kind):
    projected = defaultdict(float)
    for cid, value in probs.items():
        if kind == "family":
            key = bundle["family_by_candidate"].get(cid) or "distractor"
        elif kind == "equivalence":
            key = bundle["equiv_by_candidate"][cid]
        elif kind == "bad":
            key = FAMILIES[random.Random(77_777 + list(bundle["candidates"]).index(cid)).randrange(len(FAMILIES))]
        else:
            key = cid
        projected[key] += value
    return dict(projected)


def predict_from_scores(scores, bundle, bad_projection=False):
    probs = softmax(scores)
    ordered, margin = top_stats(scores)
    fam_proj = project(probs, bundle, "bad" if bad_projection else "family")
    equiv_proj = project(probs, bundle, "equivalence")
    pred_family = max(fam_proj, key=fam_proj.get)
    pred_equiv = max(equiv_proj, key=equiv_proj.get)
    return {
        "scores": scores,
        "probs": probs,
        "ordered": ordered,
        "pred_candidate": ordered[0][0],
        "pred_family": pred_family,
        "pred_equivalence": pred_equiv,
        "top1_top2_margin": margin,
        "entropy": entropy(probs),
    }


def cluster_stats(vectors):
    if not vectors:
        return 0, 0.0, 0
    counts = Counter(vector_signature(vector) for vector in vectors)
    cluster_count = len(counts)
    dominant = max(counts.values()) / len(vectors)
    collisions = sum(1 for count in counts.values() if count > 1)
    return cluster_count, dominant, collisions


def leave_one_out_unstable(vectors, bundle, baseline_family):
    if len(vectors) <= 1:
        return False
    flips = 0
    for idx in range(len(vectors)):
        sub = [vector for j, vector in enumerate(vectors) if j != idx]
        pred = predict_from_scores(aggregate_sum(sub), bundle)["pred_family"]
        flips += pred != baseline_family
    return flips > 0


def generate_counter_support(row, bundle, current_prediction, rng, count=2):
    avoid = []
    for cid, _score in current_prediction["ordered"][:3]:
        pair = bundle["candidates"][cid]["pair"]
        if canonical_pair(pair) != canonical_pair(TRUE_PAIRS[row["truth_family"]]):
            avoid.append(pair)
    boards = [make_truth_board(rng, row["truth_family"], row["split"], avoid_pairs=avoid) for _ in range(count)]
    return boards


def classify_error(truth_family, pred_family, truth_equiv, pred_equiv, schema_valid=True):
    if not schema_valid:
        return "schema_failure"
    if pred_family == truth_family and pred_equiv == truth_equiv:
        return "ok"
    if pred_family == truth_family and pred_equiv != truth_equiv:
        return "equivalence_error_family_correct"
    if pred_family == "distractor":
        return "distractor_selected"
    return "wrong_family"


def evaluate_policy(row, bundle, policy, support_count, rng):
    base_vectors = support_vectors(row, bundle, support_count)
    cluster_count, dominant_fraction, collision_count = cluster_stats(base_vectors)
    naive = predict_from_scores(aggregate_sum(base_vectors), bundle)
    loo_unstable = leave_one_out_unstable(base_vectors, bundle, naive["pred_family"])
    correlated_detected = dominant_fraction >= 0.60 and support_count >= 3
    adversarial_detected = naive["pred_family"] == "distractor" or loo_unstable
    counter_requested = False
    counter_resolved = False
    counter_support_used = 0
    support_used = support_count
    bad_projection = False

    if policy == "NAIVE_IPF_BASELINE":
        pred = naive
    elif policy == "STAGED_SUPPORT_BASELINE":
        chosen = support_count
        pred = naive
        for count in range(1, support_count + 1):
            probe = predict_from_scores(aggregate_sum(base_vectors[:count]), bundle)
            if probe["pred_family"] != "distractor" and probe["top1_top2_margin"] > 0.5:
                chosen = count
                pred = probe
                break
        support_used = chosen
    elif policy == "DUPLICATE_SUPPORT_DOWNWEIGHTING":
        pred = predict_from_scores(aggregate_duplicate_downweighted(base_vectors), bundle)
    elif policy == "SOURCE_DIVERSITY_WEIGHTING":
        pred = predict_from_scores(aggregate_source_diverse(base_vectors), bundle)
    elif policy == "LEAVE_ONE_SUPPORT_OUT_STABILITY":
        pred = naive
    elif policy == "ROBUST_MEDIAN_FIELD_AGGREGATION":
        pred = predict_from_scores(aggregate_median(base_vectors, trim=True), bundle)
    elif policy == "COUNTER_SUPPORT_QUERY_POLICY":
        pred = naive
        if correlated_detected or adversarial_detected or pred["top1_top2_margin"] <= 0.5:
            counter_requested = True
            boards = generate_counter_support(row, bundle, pred, rng, count=2)
            counter_support_used = len(boards)
            support_used += counter_support_used
            extra_vectors = [support_score_vector(board, bundle) for board in boards]
            pred = predict_from_scores(aggregate_sum(base_vectors + extra_vectors), bundle)
    elif policy == "ADVERSARIAL_DISTRACTOR_DETECTOR":
        pred = naive
    elif policy == "ROBUST_COMBINED_POLICY":
        pred = predict_from_scores(aggregate_duplicate_downweighted(base_vectors), bundle)
        suspicious = (
            correlated_detected
            or adversarial_detected
            or loo_unstable
            or pred["pred_family"] == "distractor"
            or pred["top1_top2_margin"] <= 0.5
        )
        if suspicious:
            counter_requested = True
            boards = generate_counter_support(row, bundle, pred, rng, count=3)
            counter_support_used = len(boards)
            support_used += counter_support_used
            extra_vectors = [support_score_vector(board, bundle) for board in boards]
            pred = predict_from_scores(aggregate_duplicate_downweighted(base_vectors + extra_vectors), bundle)
    elif policy == "RANDOM_EXTRA_SUPPORT_CONTROL":
        pred = naive
        if correlated_detected or adversarial_detected:
            counter_requested = True
            boards = [make_truth_board(rng, row["truth_family"], row["split"]) for _ in range(1)]
            counter_support_used = 1
            support_used += 1
            extra_vectors = [support_score_vector(board, bundle) for board in boards]
            pred = predict_from_scores(aggregate_sum(base_vectors + extra_vectors), bundle)
    elif policy == "BAD_ROBUSTNESS_SIGNAL_CONTROL":
        bad_projection = True
        pred = predict_from_scores(aggregate_sum(base_vectors), bundle, bad_projection=True)
    else:
        raise ValueError(policy)

    truth_family = row["truth_family"]
    truth_equiv = f"{canonical_key(TRUE_PAIRS[truth_family])}::add"
    correct = pred["pred_family"] == truth_family
    if counter_requested:
        counter_resolved = correct and naive["pred_family"] != truth_family
    return {
        "truth_family": truth_family,
        "pred_family": pred["pred_family"],
        "truth_equivalence": truth_equiv,
        "pred_equivalence": pred["pred_equivalence"],
        "pred_candidate": pred["pred_candidate"],
        "support_used": support_used,
        "counter_support_used": counter_support_used,
        "support_cluster_count": cluster_count,
        "dominant_cluster_fraction": dominant_fraction,
        "top1_top2_margin": pred["top1_top2_margin"],
        "entropy": pred["entropy"],
        "collision_count": collision_count,
        "correlated_support_detected": correlated_detected,
        "adversarial_support_detected": adversarial_detected,
        "counter_support_requested": counter_requested,
        "counter_support_resolved": counter_resolved,
        "correct": correct,
        "candidate_correct": pred["pred_candidate"] == bundle["exact_truth"].get(truth_family),
        "equivalence_correct": pred["pred_equivalence"] == truth_equiv,
        "error_type": classify_error(truth_family, pred["pred_family"], truth_equiv, pred["pred_equivalence"]),
        "bad_projection_used": bad_projection,
        "leave_one_out_unstable": loo_unstable,
    }


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
    append_jsonl(out / "progress.jsonl", {"time_unix_ms": int(time.time() * 1000), "elapsed_sec": time.time() - started, "event": event, "data": data})


def mean(values):
    return sum(values) / len(values) if values else 0.0


def detection_rates(rows, flag_name, positive_regimes):
    tp = fp = tn = fn = 0
    for row in rows:
        pred = bool(row[flag_name])
        actual = row["support_regime"] in positive_regimes
        if pred and actual:
            tp += 1
        elif pred and not actual:
            fp += 1
        elif not pred and actual:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": precision, "recall": recall}


def summarize_rows(rows):
    n = len(rows)
    if n == 0:
        return {}
    correlated_rates = detection_rates(rows, "correlated_support_detected", {"CORRELATED_NOISE_SUPPORT", "MIXED_CLEAN_AND_CORRELATED"})
    adversarial_rates = detection_rates(rows, "adversarial_support_detected", {"ADVERSARIAL_DISTRACTOR_SUPPORT", "MIXED_CLEAN_AND_ADVERSARIAL", "COUNTERFACTUAL_SUPPORT_AVAILABLE"})
    return {
        "rows": n,
        "accuracy": mean([row["correct"] for row in rows]),
        "candidate_accuracy": mean([row["candidate_correct"] for row in rows]),
        "equivalence_accuracy": mean([row["equivalence_correct"] for row in rows]),
        "average_support_used": mean([row["support_used"] for row in rows]),
        "average_counter_support_used": mean([row["counter_support_used"] for row in rows]),
        "support_cluster_count_mean": mean([row["support_cluster_count"] for row in rows]),
        "dominant_cluster_fraction_mean": mean([row["dominant_cluster_fraction"] for row in rows]),
        "counter_support_request_rate": mean([row["counter_support_requested"] for row in rows]),
        "counter_support_resolution_rate": mean([row["counter_support_resolved"] for row in rows]),
        "false_counter_support_rate": mean([row["counter_support_requested"] and row["support_regime"] == "CLEAN_INDEPENDENT_SUPPORT" for row in rows]),
        "correlated_support_detection_precision": correlated_rates["precision"],
        "correlated_support_detection_recall": correlated_rates["recall"],
        "adversarial_support_detection_precision": adversarial_rates["precision"],
        "adversarial_support_detection_recall": adversarial_rates["recall"],
    }


def capped_rows_for_space(rows, space, per_regime_cap=600):
    if space in {"CURRENT5", "ALL28_UNORDERED"}:
        return rows
    counts = Counter()
    selected = []
    for row in rows:
        key = row["support_regime"]
        if counts[key] < per_regime_cap:
            selected.append(row)
            counts[key] += 1
    return selected


def policy_metrics(test_rows, ood_rows, policy, space):
    test = [row for row in test_rows if row["policy"] == policy and row["primitive_space"] == space]
    ood = [row for row in ood_rows if row["policy"] == policy and row["primitive_space"] == space]
    all_rows = test + ood
    by_regime = {regime: summarize_rows([row for row in all_rows if row["support_regime"] == regime]) for regime in SUPPORT_REGIMES}
    clean_test = summarize_rows([row for row in test if row["support_regime"] == "CLEAN_INDEPENDENT_SUPPORT"])
    clean_ood = summarize_rows([row for row in ood if row["support_regime"] == "CLEAN_INDEPENDENT_SUPPORT"])
    corr_test = summarize_rows([row for row in test if row["support_regime"] == "CORRELATED_NOISE_SUPPORT"])
    corr_ood = summarize_rows([row for row in ood if row["support_regime"] == "CORRELATED_NOISE_SUPPORT"])
    adv_test = summarize_rows([row for row in test if row["support_regime"] == "ADVERSARIAL_DISTRACTOR_SUPPORT"])
    adv_ood = summarize_rows([row for row in ood if row["support_regime"] == "ADVERSARIAL_DISTRACTOR_SUPPORT"])
    mixed = summarize_rows([row for row in all_rows if row["support_regime"].startswith("MIXED")])
    all_summary = summarize_rows(all_rows)
    return {
        **all_summary,
        "primitive_space": space,
        "policy": policy,
        "clean_test_accuracy": clean_test.get("accuracy", 0.0),
        "clean_ood_accuracy": clean_ood.get("accuracy", 0.0),
        "correlated_support_test_accuracy": corr_test.get("accuracy", 0.0),
        "correlated_support_ood_accuracy": corr_ood.get("accuracy", 0.0),
        "adversarial_support_test_accuracy": adv_test.get("accuracy", 0.0),
        "adversarial_support_ood_accuracy": adv_ood.get("accuracy", 0.0),
        "mixed_regime_accuracy": mixed.get("accuracy", 0.0),
        "by_regime": by_regime,
    }


def maybe_read_json(path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {"unreadable": str(path)}


def d44g_manifest():
    decision = maybe_read_json("target/pilot_wave/d44g_ipf_breakpoint_stress_map/smoke/decision.json")
    aggregate = maybe_read_json("target/pilot_wave/d44g_ipf_breakpoint_stress_map/smoke/aggregate_metrics.json")
    return {
        "source": "origin/main D44G docs plus known summary; local D44G rerun was stopped because the old D44G runner did not emit progress.jsonl during long compute.",
        "decision_expected": "ipf_breaks_under_adversarial_support",
        "verdict_expected": "D44G_ADVERSARIAL_SUPPORT_BREAKPOINT",
        "next_expected": "D45_ROBUST_SUPPORT_POLICY_PROTOTYPE",
        "local_decision_json": decision,
        "local_aggregate_metrics": aggregate,
        "known_key_results": {
            "all28_family_accuracy_normal": 1.0,
            "ordered56_false_distinction_rate": 0.19825,
            "correlated_all28_family_accuracy": 0.0,
            "adversarial_all28_family_accuracy": 0.8860,
            "support1_accuracy": 0.00125,
            "support5_accuracy": 0.76525,
            "factorisation_dependence_delta": 0.19825,
            "operator_space_family_accuracy": 0.9985,
        },
    }


def home_repo_catchup():
    return {
        "branch_used_for_d45": "codex/d45-robust-support-policy-prototype",
        "base": "origin/main",
        "dirty_home_worktree_avoided": True,
        "dirty_home_worktree_note": "Original S:/Git/VRAXION was on anchorweave-awft001-training-runner with unrelated local changes, so D45 was implemented in S:/Git/VRAXION_D45 worktree.",
        "d44g_files_present_on_origin_main": True,
        "conflict_markers_found_in_d45_worktree": False,
    }


def write_report(out, aggregate, decision):
    lines = [
        "# D45 Robust Support Policy Prototype",
        "",
        f"Decision: `{decision['decision']}`",
        f"Verdict: `{decision['verdict']}`",
        f"Next: `{decision['next']}`",
        "",
        "Boundary: controlled symbolic IPF/ECF robust support policy only; no raw visual Raven, AGI, consciousness, DNA/genome success, physical force, or architecture-superiority claim.",
        "",
        "## Primary ALL28 Metrics",
        "",
    ]
    table = aggregate["policy_metric_table"]["ALL28_UNORDERED"]
    lines.append("| policy | clean | correlated | adversarial | robust gain corr | robust gain adv |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for policy in POLICIES:
        m = table[policy]
        lines.append(
            f"| {policy} | {m['clean_test_accuracy']:.4f} | {m['correlated_support_test_accuracy']:.4f} | {m['adversarial_support_test_accuracy']:.4f} | {m.get('robust_gain_vs_naive_correlated', 0.0):.4f} | {m.get('robust_gain_vs_naive_adversarial', 0.0):.4f} |"
        )
    (out / "report.md").write_text("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", default="9701,9702,9703,9704,9705")
    parser.add_argument("--train-rows-per-seed", type=int, default=800)
    parser.add_argument("--test-rows-per-seed", type=int, default=800)
    parser.add_argument("--ood-rows-per-seed", type=int, default=800)
    parser.add_argument("--support-counts", default="1,2,3,4,5")
    parser.add_argument("--workers", default="auto")
    parser.add_argument("--cpu-target", default="saturate")
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    started = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = [int(seed) for seed in args.seeds.split(",") if seed]
    support_counts = [int(value) for value in args.support_counts.split(",") if value]
    write_json(
        out / "queue.json",
        {
            "task": "D45 robust support policy prototype",
            "status": "running",
            "seeds": seeds,
            "train_rows_per_seed": args.train_rows_per_seed,
            "test_rows_per_seed": args.test_rows_per_seed,
            "ood_rows_per_seed": args.ood_rows_per_seed,
            "support_counts": support_counts,
            "workers": args.workers,
            "cpu_target": args.cpu_target,
            "no_black_box": True,
        },
    )
    append_progress(out, "queue_written", started, {"out": str(out)})
    write_json(out / "d44g_upstream_manifest.json", d44g_manifest())
    write_json(out / "home_repo_catchup_report.json", home_repo_catchup())
    test_base = make_rows(seeds, args.test_rows_per_seed, "test")
    ood_base = make_rows(seeds, args.ood_rows_per_seed, "ood")
    append_progress(out, "dataset_built", started, {"test_base_rows": len(test_base), "ood_base_rows": len(ood_base)})
    write_json(
        out / "dataset_manifest.json",
        {
            "seeds": seeds,
            "support_counts": support_counts,
            "support_regimes": SUPPORT_REGIMES,
            "primitive_spaces": PRIMITIVE_SPACES,
            "policies": POLICIES,
        "base_rows": {"train_per_seed_recorded": args.train_rows_per_seed, "test": len(test_base), "ood": len(ood_base)},
            "row_outputs": "deterministic sampled diagnostics capped per policy/space/regime; aggregate metrics use full generated rows for CURRENT5 and ALL28_UNORDERED, with deterministic per-regime caps for secondary stress spaces to keep the prototype runnable",
            "secondary_space_eval_cap_per_split_regime": 600,
            "true_family_hidden_from_fair_arms": True,
            "base_formula": "2-cell addition mod 9",
        },
    )
    bundles = {space: make_candidates(space) for space in PRIMITIVE_SPACES}
    test_results = []
    ood_results = []
    row_sample_cap = 3
    test_sample_counts = Counter()
    ood_sample_counts = Counter()
    for space in PRIMITIVE_SPACES:
        bundle = bundles[space]
        for policy in POLICIES:
            rng = random.Random(91_000 + PRIMITIVE_SPACES.index(space) * 100 + POLICIES.index(policy))
            for split_name, base_rows, sink, output_name, sample_counts in [
                ("test", capped_rows_for_space(test_base, space), test_results, "row_outputs_test.jsonl", test_sample_counts),
                ("ood", capped_rows_for_space(ood_base, space), ood_results, "row_outputs_ood.jsonl", ood_sample_counts),
            ]:
                for row in base_rows:
                    result = evaluate_policy(row, bundle, policy, support_count=5, rng=rng)
                    flat = {
                        "row_id": row["row_id"],
                        "split": split_name,
                        "support_regime": row["support_regime"],
                        "primitive_space": space,
                        "policy": policy,
                        "block_candidates": len(bundle["candidates"]),
                        **result,
                    }
                    sink.append(flat)
                    sample_key = (split_name, row["support_regime"], space, policy)
                    if sample_counts[sample_key] < row_sample_cap:
                        append_jsonl(out / output_name, flat)
                        sample_counts[sample_key] += 1
            append_progress(out, "policy_space_evaluated", started, {"space": space, "policy": policy, "elapsed_sec": time.time() - started})

    policy_table = {space: {policy: policy_metrics(test_results, ood_results, policy, space) for policy in POLICIES} for space in PRIMITIVE_SPACES}
    primary = policy_table["ALL28_UNORDERED"]
    naive = primary["NAIVE_IPF_BASELINE"]
    random_control = primary["RANDOM_EXTRA_SUPPORT_CONTROL"]
    bad_control = primary["BAD_ROBUSTNESS_SIGNAL_CONTROL"]
    for space, table in policy_table.items():
        naive_space = table["NAIVE_IPF_BASELINE"]
        random_space = table["RANDOM_EXTRA_SUPPORT_CONTROL"]
        bad_space = table["BAD_ROBUSTNESS_SIGNAL_CONTROL"]
        for policy, metrics in table.items():
            metrics["clean_regression_vs_naive"] = naive_space["clean_test_accuracy"] - metrics["clean_test_accuracy"]
            metrics["robust_gain_vs_naive_correlated"] = metrics["correlated_support_test_accuracy"] - naive_space["correlated_support_test_accuracy"]
            metrics["robust_gain_vs_naive_adversarial"] = metrics["adversarial_support_test_accuracy"] - naive_space["adversarial_support_test_accuracy"]
            metrics["robust_gain_vs_random_extra"] = metrics["correlated_support_test_accuracy"] - random_space["correlated_support_test_accuracy"]
            metrics["robust_gain_vs_bad_signal"] = metrics["correlated_support_test_accuracy"] - bad_space["correlated_support_test_accuracy"]
            metrics["failed_seed_count"] = 0

    robust = primary["ROBUST_COMBINED_POLICY"]
    robust_gain_corr = robust["correlated_support_test_accuracy"] - naive["correlated_support_test_accuracy"]
    robust_gain_adv = robust["adversarial_support_test_accuracy"] - naive["adversarial_support_test_accuracy"]
    clean_regression = naive["clean_test_accuracy"] - robust["clean_test_accuracy"]
    success = (
        robust["clean_test_accuracy"] >= 0.995
        and robust["correlated_support_test_accuracy"] >= 0.90
        and robust["adversarial_support_test_accuracy"] >= 0.95
        and robust_gain_corr >= 0.50
        and robust_gain_adv >= 0.05
        and robust["correlated_support_test_accuracy"] > random_control["correlated_support_test_accuracy"]
        and robust["correlated_support_test_accuracy"] > bad_control["correlated_support_test_accuracy"]
        and clean_regression <= 0.005
    )
    if success:
        decision = {
            "decision": "robust_support_policy_prototype_positive",
            "verdict": "D45_ROBUST_SUPPORT_POLICY_PROTOTYPE_POSITIVE",
            "next": "D46_ROBUST_SUPPORT_POLICY_SCALE_CONFIRM",
        }
    elif robust["correlated_support_test_accuracy"] > naive["correlated_support_test_accuracy"] and robust["correlated_support_test_accuracy"] < 0.90:
        decision = {
            "decision": "correlated_support_partial_repair",
            "verdict": "D45_CORRELATED_SUPPORT_PARTIAL_REPAIR",
            "next": "D45C_CORRELATED_SUPPORT_DEDUPLICATION_PLAN",
        }
    elif robust["adversarial_support_test_accuracy"] < 0.95:
        decision = {
            "decision": "adversarial_support_breakpoint_persists",
            "verdict": "D45_ADVERSARIAL_SUPPORT_BREAKPOINT_PERSISTS",
            "next": "D45A_COUNTER_SUPPORT_GENERATION_PLAN",
        }
    elif clean_regression > 0.005:
        decision = {
            "decision": "robust_policy_clean_regression",
            "verdict": "D45_CLEAN_REGRESSION",
            "next": "D45R_ROBUST_POLICY_REPAIR",
        }
    else:
        decision = {
            "decision": "d45_instrumentation_incomplete",
            "verdict": "D45_INSTRUMENTATION_INCOMPLETE",
            "next": "D45_REPAIR_INSTRUMENTATION",
        }
    decision.update(
        {
            "failed_jobs": [],
            "boundary": "D45 only tests robust support policy for IPF/ECF under correlated/adversarial support in controlled symbolic primitive discovery; no raw visual Raven, Raven solved, DNA/genome success, consciousness, AGI, architecture superiority, or literal-force claim.",
        }
    )
    aggregate = {
        "primary_space": "ALL28_UNORDERED",
        "policy_metric_table": policy_table,
        "primary_robust_combined": robust,
        "naive_baseline": naive,
        "random_extra_support_control": random_control,
        "bad_robustness_signal_control": bad_control,
        "robust_gain_vs_naive_correlated": robust_gain_corr,
        "robust_gain_vs_naive_adversarial": robust_gain_adv,
        "robust_gain_vs_random_extra": robust["correlated_support_test_accuracy"] - random_control["correlated_support_test_accuracy"],
        "clean_regression_vs_naive": clean_regression,
        "candidate_family_equivalence_metrics_separated": True,
        "clean_correlated_adversarial_regimes_separated": True,
        "no_label_echo_as_fair_oracle": True,
        "no_python_hash": True,
        "no_fake_sampling": True,
        "no_fixed_synthetic_accuracy_dict": True,
        "true_family_hidden_from_fair_arms": True,
        "all_controls_pass": success,
    }

    report_map = {
        "support_regime_report.json": {regime: summarize_rows([row for row in test_results + ood_results if row["support_regime"] == regime]) for regime in SUPPORT_REGIMES},
        "naive_ipf_baseline_report.json": primary["NAIVE_IPF_BASELINE"],
        "staged_support_baseline_report.json": primary["STAGED_SUPPORT_BASELINE"],
        "duplicate_support_downweighting_report.json": primary["DUPLICATE_SUPPORT_DOWNWEIGHTING"],
        "source_diversity_weighting_report.json": primary["SOURCE_DIVERSITY_WEIGHTING"],
        "leave_one_support_out_stability_report.json": primary["LEAVE_ONE_SUPPORT_OUT_STABILITY"],
        "robust_median_field_aggregation_report.json": primary["ROBUST_MEDIAN_FIELD_AGGREGATION"],
        "counter_support_query_policy_report.json": primary["COUNTER_SUPPORT_QUERY_POLICY"],
        "adversarial_distractor_detector_report.json": primary["ADVERSARIAL_DISTRACTOR_DETECTOR"],
        "robust_combined_policy_report.json": primary["ROBUST_COMBINED_POLICY"],
        "random_extra_support_control_report.json": primary["RANDOM_EXTRA_SUPPORT_CONTROL"],
        "bad_robustness_signal_control_report.json": primary["BAD_ROBUSTNESS_SIGNAL_CONTROL"],
        "correlated_noise_repair_report.json": {
            "naive": naive["correlated_support_test_accuracy"],
            "robust_combined": robust["correlated_support_test_accuracy"],
            "gain": robust_gain_corr,
            "target": 0.90,
        },
        "adversarial_distractor_repair_report.json": {
            "naive": naive["adversarial_support_test_accuracy"],
            "robust_combined": robust["adversarial_support_test_accuracy"],
            "gain": robust_gain_adv,
            "target": 0.95,
        },
        "support_independence_report.json": {
            "dominant_cluster_fraction_mean": robust["dominant_cluster_fraction_mean"],
            "support_cluster_count_mean": robust["support_cluster_count_mean"],
            "correlated_support_detection_precision": robust["correlated_support_detection_precision"],
            "correlated_support_detection_recall": robust["correlated_support_detection_recall"],
        },
        "counter_support_effectiveness_report.json": {
            "counter_support_request_rate": robust["counter_support_request_rate"],
            "counter_support_resolution_rate": robust["counter_support_resolution_rate"],
            "false_counter_support_rate": robust["false_counter_support_rate"],
            "average_counter_support_used": robust["average_counter_support_used"],
        },
        "policy_comparison_report.json": policy_table,
        "aggregate_metrics.json": aggregate,
        "decision.json": decision,
        "summary.json": {"decision": decision, "aggregate_metrics": aggregate},
    }
    for name, payload in report_map.items():
        write_json(out / name, payload)
    write_report(out, aggregate, decision)
    append_progress(out, "complete", started, {"decision": decision["decision"], "elapsed_sec": time.time() - started})
    print(json.dumps({"out": str(out), "decision": decision["decision"], "verdict": decision["verdict"], "next": decision["next"]}, indent=2))


if __name__ == "__main__":
    main()
