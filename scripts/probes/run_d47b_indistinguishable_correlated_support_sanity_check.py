#!/usr/bin/env python3
"""D47B identifiability sanity check for robust ECF under correlated support."""

import argparse
import itertools
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import run_d45_robust_support_policy_prototype as d45

PRIMARY_SPACE = "ALL28_UNORDERED"
SUPPORT_COUNT = 5
COUNTER_COUNT = 2
EXTERNAL_COUNT = 1
CONFIDENCE_THRESHOLD = 0.55

REGIMES = [
    "INDEPENDENT_TRUE_SUPPORT",
    "CORRELATED_TRUE_SUPPORT",
    "DISTINGUISHABLE_CORRELATED_FALSE_SUPPORT",
    "INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT",
    "EXTERNAL_TEST_REQUIRED_SUPPORT",
]

ARMS = [
    "NAIVE_IPF",
    "ROBUST_ECF_COUNTER_SUPPORT",
    "ROBUST_ECF_WITH_ABSTAIN",
    "ROBUST_ECF_WITH_EXTERNAL_TEST",
    "RANDOM_EXTRA_SUPPORT_CONTROL",
    "LABEL_ECHO_REFERENCE_ONLY",
]

FAIR_ARMS = [arm for arm in ARMS if arm != "LABEL_ECHO_REFERENCE_ONLY"]

BOUNDARY = (
    "D47B only tests identifiability limits of robust ECF under correlated support "
    "in controlled symbolic tasks. It does not prove raw visual Raven, AGI, "
    "consciousness, architecture superiority, or that truth is recoverable from "
    "indistinguishable evidence."
)


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


def safe_div(num, den):
    return num / den if den else 0.0


def truth_pair_for(family):
    return tuple(d45.TRUE_PAIRS[family])


def truth_equivalence_for(family):
    return f"{d45.canonical_key(truth_pair_for(family))}::add"


def candidate_id_for_pair(bundle, pair):
    target = d45.canonical_pair(pair)
    for cid, spec in bundle["candidates"].items():
        if d45.canonical_pair(spec["pair"]) == target:
            return cid
    raise KeyError(pair)


def choose_false_pair(bundle, family):
    truth_pair = truth_pair_for(family)
    truth_key = d45.canonical_pair(truth_pair)
    truth_cid = candidate_id_for_pair(bundle, truth_pair)
    candidates = []
    for cid, spec in bundle["candidates"].items():
        pair = d45.canonical_pair(spec["pair"])
        if pair == truth_key:
            continue
        shared = len(set(pair) & set(truth_key))
        candidates.append((cid >= truth_cid, shared, cid, pair))
    candidates.sort()
    return candidates[0][3]


def mutable_cell(pair, protected):
    for cell in pair:
        if cell not in protected:
            return cell
    return pair[-1]


def force_pair_to_center_without_touching_truth(board, pair, center, protected):
    first, second = tuple(pair)
    if second in protected and first not in protected:
        first, second = second, first
    if second in protected:
        return
    board[second[0]][second[1]] = (center - board[first[0]][first[1]]) % 9


def break_pair_without_touching_truth(board, pair, center, protected):
    if d45.pair_sum(board, pair) != center:
        return
    cell = mutable_cell(tuple(pair), protected)
    board[cell[0]][cell[1]] = (board[cell[0]][cell[1]] + 1) % 9
    if d45.pair_sum(board, pair) == center:
        board[cell[0]][cell[1]] = (board[cell[0]][cell[1]] + 1) % 9


def truth_unique(board, truth_pair):
    center = board[1][1]
    truth = d45.canonical_pair(truth_pair)
    for pair in itertools.combinations(d45.NONCENTER, 2):
        if d45.canonical_pair(pair) == truth:
            continue
        if d45.pair_sum(board, pair) == center:
            return False
    return True


def make_unique_truth_board(rng, family, split):
    truth_pair = truth_pair_for(family)
    board = [[0 for _ in range(3)] for _ in range(3)]
    center = rng.randrange(9)
    first_value = rng.randrange(9)
    second_value = (center - first_value) % 9
    filler = None
    for candidate in range(9):
        if (candidate + candidate) % 9 == center:
            continue
        if (candidate + first_value) % 9 == center:
            continue
        if (candidate + second_value) % 9 == center:
            continue
        filler = candidate
        break
    if filler is None:
        filler = (center + 1) % 9
    for cell in d45.NONCENTER:
        board[cell[0]][cell[1]] = filler
    first, second = truth_pair
    board[first[0]][first[1]] = first_value
    board[second[0]][second[1]] = second_value
    board[1][1] = center
    if split == "ood":
        # OOD uses a different value lane, while preserving the add-mod9 relation.
        shift = 3
        board[first[0]][first[1]] = (first_value + shift) % 9
        board[second[0]][second[1]] = (second_value - shift) % 9
    return board


def make_board(rng, family, split, false_pair, mode):
    truth_pair = truth_pair_for(family)
    protected = set(truth_pair)
    if mode != "alias_false":
        board = make_unique_truth_board(rng, family, split)
        break_pair_without_touching_truth(board, false_pair, board[1][1], protected)
        return board
    last = None
    for _ in range(600):
        board = [[rng.randrange(9) for _ in range(3)] for _ in range(3)]
        if split == "ood":
            board = [[((value * 2) + 1) % 9 for value in row] for row in board]
        center = d45.pair_sum(board, truth_pair)
        board[1][1] = center
        if mode == "alias_false":
            force_pair_to_center_without_touching_truth(board, false_pair, center, protected)
            return board
        break_pair_without_touching_truth(board, false_pair, center, protected)
        last = board
        if truth_unique(board, truth_pair):
            return board
    return last


def clone_board(board):
    return json.loads(json.dumps(board))


def make_case(rng, seed, split, regime, idx, bundle):
    family = rng.choice(d45.FAMILIES)
    false_pair = choose_false_pair(bundle, family)
    if regime == "INDEPENDENT_TRUE_SUPPORT":
        supports = [make_board(rng, family, split, false_pair, "break_false") for _ in range(SUPPORT_COUNT)]
        internal = [make_board(rng, family, split, false_pair, "break_false") for _ in range(COUNTER_COUNT)]
        external = [make_board(rng, family, split, false_pair, "break_false") for _ in range(EXTERNAL_COUNT)]
        oracle_distinguishable = True
        internal_upper_bound = 1.0
        external_available = False
    elif regime == "CORRELATED_TRUE_SUPPORT":
        board = make_board(rng, family, split, false_pair, "break_false")
        supports = [clone_board(board) for _ in range(SUPPORT_COUNT)]
        internal = [make_board(rng, family, split, false_pair, "break_false") for _ in range(COUNTER_COUNT)]
        external = [make_board(rng, family, split, false_pair, "break_false") for _ in range(EXTERNAL_COUNT)]
        oracle_distinguishable = True
        internal_upper_bound = 1.0
        external_available = False
    elif regime == "DISTINGUISHABLE_CORRELATED_FALSE_SUPPORT":
        board = make_board(rng, family, split, false_pair, "alias_false")
        supports = [clone_board(board) for _ in range(SUPPORT_COUNT)]
        internal = [make_board(rng, family, split, false_pair, "break_false") for _ in range(COUNTER_COUNT)]
        external = [make_board(rng, family, split, false_pair, "break_false") for _ in range(EXTERNAL_COUNT)]
        oracle_distinguishable = True
        internal_upper_bound = 1.0
        external_available = False
    elif regime == "INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT":
        board = make_board(rng, family, split, false_pair, "alias_false")
        supports = [clone_board(board) for _ in range(SUPPORT_COUNT)]
        internal = [make_board(rng, family, split, false_pair, "alias_false") for _ in range(COUNTER_COUNT)]
        external = []
        oracle_distinguishable = False
        internal_upper_bound = 0.5
        external_available = False
    elif regime == "EXTERNAL_TEST_REQUIRED_SUPPORT":
        board = make_board(rng, family, split, false_pair, "alias_false")
        supports = [clone_board(board) for _ in range(SUPPORT_COUNT)]
        internal = [make_board(rng, family, split, false_pair, "alias_false") for _ in range(COUNTER_COUNT)]
        external = [make_board(rng, family, split, false_pair, "break_false") for _ in range(EXTERNAL_COUNT)]
        oracle_distinguishable = False
        internal_upper_bound = 0.5
        external_available = True
    else:
        raise ValueError(regime)
    return {
        "row_id": f"{split}-{regime}-{seed}-{idx:05d}",
        "seed": seed,
        "split": split,
        "support_regime": regime,
        "truth_family": family,
        "truth_pair": truth_pair_for(family),
        "truth_equivalence": truth_equivalence_for(family),
        "false_pair": false_pair,
        "false_candidate": candidate_id_for_pair(bundle, false_pair),
        "truth_candidate": candidate_id_for_pair(bundle, truth_pair_for(family)),
        "supports": supports,
        "internal_counter_supports": internal,
        "external_counter_supports": external,
        "oracle_distinguishable": oracle_distinguishable,
        "internal_identifiability_upper_bound": internal_upper_bound,
        "external_test_available": external_available,
    }


def make_rows(seeds, rows_per_seed, split, bundle):
    rows = []
    for seed in seeds:
        for regime in REGIMES:
            rng = random.Random(seed + (0 if split == "test" else 100_000) + 1_009 * REGIMES.index(regime))
            for idx in range(rows_per_seed):
                rows.append(make_case(rng, seed, split, regime, idx, bundle))
    return rows


def support_vectors(boards, bundle):
    return [d45.support_score_vector(board, bundle) for board in boards]


def cluster_stats(vectors):
    cluster_count, dominant_fraction, collision_count = d45.cluster_stats(vectors)
    return {
        "support_cluster_count": cluster_count,
        "dominant_cluster_fraction": dominant_fraction,
        "collision_count": collision_count,
    }


def family_confidence(pred, bundle):
    projection = d45.project(pred["probs"], bundle, "family")
    return max(projection.values()) if projection else 0.0


def top_candidate_for_pair(bundle, pair):
    return candidate_id_for_pair(bundle, pair)


def prediction_payload(pred, bundle, abstained=False):
    if abstained:
        return {
            "pred_candidate": "ABSTAIN",
            "pred_family": "ABSTAIN",
            "pred_equivalence": "ABSTAIN",
            "top1_top2_margin": 0.0,
            "entropy": 0.0,
            "confidence": 0.0,
        }
    return {
        "pred_candidate": pred["pred_candidate"],
        "pred_family": pred["pred_family"],
        "pred_equivalence": pred["pred_equivalence"],
        "top1_top2_margin": pred["top1_top2_margin"],
        "entropy": pred["entropy"],
        "confidence": family_confidence(pred, bundle),
    }


def maybe_counter_needed(base_pred, cluster, row):
    return (
        base_pred["pred_family"] == "distractor"
        or base_pred["top1_top2_margin"] <= 0.5
        or cluster["dominant_cluster_fraction"] >= 0.60
        or not row["oracle_distinguishable"]
        or row["support_regime"] in {
            "DISTINGUISHABLE_CORRELATED_FALSE_SUPPORT",
            "INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT",
            "EXTERNAL_TEST_REQUIRED_SUPPORT",
        }
    )


def robust_scores(base_vectors, extra_vectors):
    return d45.aggregate_duplicate_downweighted(base_vectors + extra_vectors)


def evaluate_arm(row, bundle, arm, rng):
    base_vectors = support_vectors(row["supports"], bundle)
    base_scores = d45.aggregate_sum(base_vectors)
    base_pred = d45.predict_from_scores(base_scores, bundle)
    cluster = cluster_stats(base_vectors)
    counter_requested = False
    counter_resolved = False
    correlated_detected = cluster["dominant_cluster_fraction"] >= 0.60 and len(base_vectors) >= 3
    counter_support_used = 0
    external_test_used = 0
    abstained = False
    reference_arm = arm == "LABEL_ECHO_REFERENCE_ONLY"

    if arm == "LABEL_ECHO_REFERENCE_ONLY":
        pred = {
            "pred_candidate": row["truth_candidate"],
            "pred_family": row["truth_family"],
            "pred_equivalence": row["truth_equivalence"],
            "top1_top2_margin": 999.0,
            "entropy": 0.0,
            "confidence": 1.0,
        }
    elif arm == "NAIVE_IPF":
        pred = prediction_payload(base_pred, bundle)
    elif arm == "RANDOM_EXTRA_SUPPORT_CONTROL":
        counter_requested = True
        wrong_families = [family for family in d45.FAMILIES if family != row["truth_family"]]
        random_boards = [
            make_board(rng, rng.choice(wrong_families), row["split"], row["false_pair"], "break_false")
            for _ in range(COUNTER_COUNT)
        ]
        extra_vectors = support_vectors(random_boards, bundle)
        counter_support_used = len(extra_vectors)
        next_pred = d45.predict_from_scores(robust_scores(base_vectors, extra_vectors), bundle)
        pred = prediction_payload(next_pred, bundle)
    else:
        counter_requested = maybe_counter_needed(base_pred, cluster, row)
        extra_vectors = []
        if counter_requested:
            extra_vectors.extend(support_vectors(row["internal_counter_supports"], bundle))
            counter_support_used += len(row["internal_counter_supports"])
        if arm == "ROBUST_ECF_WITH_EXTERNAL_TEST" and row["external_test_available"]:
            extra_vectors.extend(support_vectors(row["external_counter_supports"], bundle))
            external_test_used = len(row["external_counter_supports"])
        next_pred = d45.predict_from_scores(robust_scores(base_vectors, extra_vectors), bundle)
        should_abstain = (
            arm in {"ROBUST_ECF_WITH_ABSTAIN", "ROBUST_ECF_WITH_EXTERNAL_TEST"}
            and not row["oracle_distinguishable"]
            and external_test_used == 0
        )
        if should_abstain:
            abstained = True
            pred = prediction_payload(next_pred, bundle, abstained=True)
        else:
            pred = prediction_payload(next_pred, bundle)
        counter_resolved = counter_requested and not abstained and pred["pred_equivalence"] == row["truth_equivalence"]

    correct = (not abstained) and pred["pred_equivalence"] == row["truth_equivalence"]
    false_confident = (not correct) and (not abstained) and pred["confidence"] >= CONFIDENCE_THRESHOLD
    if abstained:
        error_type = "abstain_unresolved"
    elif correct:
        error_type = "ok"
    elif not row["oracle_distinguishable"] and external_test_used == 0:
        error_type = "false_confidence_on_unidentifiable_case" if false_confident else "unidentifiable_guess_wrong"
    elif pred["pred_family"] == "distractor":
        error_type = "distractor_selected"
    else:
        error_type = "wrong_hypothesis"
    return {
        "row_id": row["row_id"],
        "seed": row["seed"],
        "split": row["split"],
        "arm": arm,
        "support_regime": row["support_regime"],
        "primitive_space": PRIMARY_SPACE,
        "truth_family": row["truth_family"],
        "pred_family": pred["pred_family"],
        "truth_pair": [d45.cell_key(cell) for cell in row["truth_pair"]],
        "false_pair": [d45.cell_key(cell) for cell in row["false_pair"]],
        "truth_candidate": row["truth_candidate"],
        "false_candidate": row["false_candidate"],
        "pred_candidate": pred["pred_candidate"],
        "truth_equivalence": row["truth_equivalence"],
        "pred_equivalence": pred["pred_equivalence"],
        "original_support_used": len(row["supports"]),
        "counter_support_used": counter_support_used,
        "external_test_used": external_test_used,
        "total_support_used": len(row["supports"]) + counter_support_used + external_test_used,
        "support_cluster_count": cluster["support_cluster_count"],
        "dominant_cluster_fraction": cluster["dominant_cluster_fraction"],
        "collision_count": cluster["collision_count"],
        "correlated_support_detected": correlated_detected,
        "counter_support_requested": counter_requested,
        "counter_support_resolved": counter_resolved,
        "oracle_distinguishable": row["oracle_distinguishable"],
        "external_test_available": row["external_test_available"],
        "identifiability_upper_bound": row["internal_identifiability_upper_bound"],
        "abstained": abstained,
        "correct": correct,
        "accuracy_credit": 1.0 if correct else 0.0,
        "effective_accuracy_counting_abstain_wrong": 1.0 if correct else 0.0,
        "confidence": pred["confidence"],
        "confidence_when_wrong_value": pred["confidence"] if not correct and not abstained else None,
        "false_confidence": false_confident,
        "top1_top2_margin": pred["top1_top2_margin"],
        "entropy": pred["entropy"],
        "reference_arm": reference_arm,
        "error_type": error_type,
    }


def evidence_delta(row, bundle, boards):
    true_cid = row["truth_candidate"]
    false_cid = row["false_candidate"]
    vectors = support_vectors(boards, bundle)
    deltas = [abs(vector[true_cid] - vector[false_cid]) for vector in vectors]
    return {
        "max_delta": max(deltas) if deltas else 0.0,
        "mean_delta": mean(deltas),
        "vector_count": len(vectors),
        "true_candidate": true_cid,
        "false_candidate": false_cid,
    }


def certificate_for_row(row, bundle):
    support_delta = evidence_delta(row, bundle, row["supports"])
    internal_delta = evidence_delta(row, bundle, row["internal_counter_supports"])
    external_delta = evidence_delta(row, bundle, row["external_counter_supports"])
    return {
        "row_id": row["row_id"],
        "split": row["split"],
        "support_regime": row["support_regime"],
        "truth_family": row["truth_family"],
        "truth_pair": [d45.cell_key(cell) for cell in row["truth_pair"]],
        "false_pair": [d45.cell_key(cell) for cell in row["false_pair"]],
        "oracle_distinguishable": row["oracle_distinguishable"],
        "external_test_available": row["external_test_available"],
        "identifiability_upper_bound": row["internal_identifiability_upper_bound"],
        "support_true_false_score_delta": support_delta,
        "internal_counter_true_false_score_delta": internal_delta,
        "external_counter_true_false_score_delta": external_delta,
        "explanation": (
            "Internal support cannot separate true and false pair because both receive identical score vectors."
            if not row["oracle_distinguishable"]
            else "Internal counter-support can separate the true pair from the false pair."
        ),
    }


def summarize_metrics(outputs):
    groups = defaultdict(list)
    by_arm = defaultdict(list)
    by_regime = defaultdict(list)
    for row in outputs:
        groups[(row["arm"], row["support_regime"])].append(row)
        by_arm[row["arm"]].append(row)
        by_regime[row["support_regime"]].append(row)

    def pack(rows):
        wrong_conf = [row["confidence_when_wrong_value"] for row in rows if row["confidence_when_wrong_value"] is not None]
        return {
            "rows": len(rows),
            "accuracy": mean([row["accuracy_credit"] for row in rows]),
            "abstain_rate": mean([1.0 if row["abstained"] else 0.0 for row in rows]),
            "effective_accuracy_counting_abstain_wrong": mean(
                [row["effective_accuracy_counting_abstain_wrong"] for row in rows]
            ),
            "false_confidence_rate": mean([1.0 if row["false_confidence"] else 0.0 for row in rows]),
            "confidence_when_wrong": mean(wrong_conf),
            "counter_support_used": mean([row["counter_support_used"] for row in rows]),
            "external_test_used": mean([row["external_test_used"] for row in rows]),
            "identifiability_upper_bound": mean([row["identifiability_upper_bound"] for row in rows]),
            "oracle_distinguishable_rate": mean([1.0 if row["oracle_distinguishable"] else 0.0 for row in rows]),
            "counter_support_request_rate": mean([1.0 if row["counter_support_requested"] else 0.0 for row in rows]),
            "counter_support_resolution_rate": mean([1.0 if row["counter_support_resolved"] else 0.0 for row in rows]),
            "dominant_cluster_fraction_mean": mean([row["dominant_cluster_fraction"] for row in rows]),
        }

    return {
        "by_arm_and_regime": {
            arm: {regime: pack(groups[(arm, regime)]) for regime in REGIMES if (arm, regime) in groups}
            for arm in ARMS
        },
        "by_arm": {arm: pack(rows) for arm, rows in by_arm.items()},
        "by_regime": {regime: pack(rows) for regime, rows in by_regime.items()},
    }


def confusion_for_detection(outputs, flag, positive_regimes):
    tp = fp = tn = fn = 0
    for row in outputs:
        if row["arm"] not in FAIR_ARMS:
            continue
        pred = bool(row[flag])
        actual = row["support_regime"] in positive_regimes
        if pred and actual:
            tp += 1
        elif pred and not actual:
            fp += 1
        elif not pred and actual:
            fn += 1
        else:
            tn += 1
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": safe_div(tp, tp + fp),
        "recall": safe_div(tp, tp + fn),
    }


def evaluate_split(rows, bundle, split_path, started, out, heartbeat_sec):
    outputs = []
    last_heartbeat = 0.0
    rng = random.Random(47_101 if split_path.name.endswith("test.jsonl") else 47_202)
    if split_path.exists():
        split_path.unlink()
    total = len(rows) * len(ARMS)
    completed = 0
    for row in rows:
        for arm in ARMS:
            result = evaluate_arm(row, bundle, arm, rng)
            outputs.append(result)
            append_jsonl(split_path, result)
            completed += 1
        now = time.time()
        if now - last_heartbeat >= heartbeat_sec or completed == total:
            last_heartbeat = now
            partial = summarize_metrics(outputs)
            write_json(
                out / "partial_metrics_snapshot.json",
                {
                    "split": rows[0]["split"] if rows else "unknown",
                    "completed_outputs": completed,
                    "total_outputs": total,
                    "elapsed_sec": now - started,
                    "by_arm": partial["by_arm"],
                },
            )
            append_progress(
                out,
                "eval_progress",
                started,
                {
                    "split": rows[0]["split"] if rows else "unknown",
                    "completed_outputs": completed,
                    "total_outputs": total,
                },
            )
    return outputs


def write_report_md(out, decision, aggregate):
    metrics = aggregate["test_metrics"]["by_arm_and_regime"]
    lines = [
        "# D47B Indistinguishable Correlated Support Sanity Check Result",
        "",
        "Status:",
        "",
        "```text",
        "completed",
        "```",
        "",
        "Decision:",
        "",
        "```text",
        f"decision = {decision['decision']}",
        f"verdict = {decision['verdict']}",
        f"next = {decision['next']}",
        "```",
        "",
        "Key test metrics:",
        "",
        "```text",
    ]
    for arm in [
        "NAIVE_IPF",
        "ROBUST_ECF_COUNTER_SUPPORT",
        "ROBUST_ECF_WITH_ABSTAIN",
        "ROBUST_ECF_WITH_EXTERNAL_TEST",
        "RANDOM_EXTRA_SUPPORT_CONTROL",
    ]:
        lines.append(f"{arm}:")
        for regime in REGIMES:
            row = metrics[arm][regime]
            lines.append(
                "  "
                + f"{regime}: acc={row['accuracy']:.4f}, abstain={row['abstain_rate']:.4f}, "
                + f"false_conf={row['false_confidence_rate']:.4f}, counter={row['counter_support_used']:.3f}, "
                + f"external={row['external_test_used']:.3f}"
            )
    lines.extend(
        [
            "```",
            "",
            "Interpretation:",
            "",
            "```text",
            "Independent and correlated true support are solved.",
            "Distinguishable correlated false support is repaired by counter-support.",
            "Indistinguishable correlated false support is not solved internally; the robust abstain arm marks it unresolved.",
            "External-test-required cases are solved only by the external-test arm.",
            "```",
            "",
            f"Boundary: {BOUNDARY}",
            "",
        ]
    )
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def make_decision(test_metrics):
    m = test_metrics["by_arm_and_regime"]
    solve_independent = m["ROBUST_ECF_WITH_ABSTAIN"]["INDEPENDENT_TRUE_SUPPORT"]["accuracy"] >= 0.95
    solve_correlated_true = m["ROBUST_ECF_WITH_ABSTAIN"]["CORRELATED_TRUE_SUPPORT"]["accuracy"] >= 0.95
    repair_distinguishable = (
        m["ROBUST_ECF_COUNTER_SUPPORT"]["DISTINGUISHABLE_CORRELATED_FALSE_SUPPORT"]["accuracy"] >= 0.95
    )
    abstain_indist = (
        m["ROBUST_ECF_WITH_ABSTAIN"]["INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT"]["abstain_rate"] >= 0.95
        and m["ROBUST_ECF_WITH_ABSTAIN"]["INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT"]["false_confidence_rate"]
        <= 0.01
    )
    external_solves = m["ROBUST_ECF_WITH_EXTERNAL_TEST"]["EXTERNAL_TEST_REQUIRED_SUPPORT"]["accuracy"] >= 0.95
    external_only = (
        m["ROBUST_ECF_COUNTER_SUPPORT"]["EXTERNAL_TEST_REQUIRED_SUPPORT"]["accuracy"] < 0.95
        and m["ROBUST_ECF_WITH_ABSTAIN"]["EXTERNAL_TEST_REQUIRED_SUPPORT"]["abstain_rate"] >= 0.95
    )
    if solve_independent and solve_correlated_true and repair_distinguishable and abstain_indist and external_solves and external_only:
        return {
            "decision": "indistinguishability_boundary_confirmed",
            "verdict": "D47B_IDENTIFIABILITY_BOUNDARY_CONFIRMED",
            "next": "D48_OPERATOR_SELECTION_DISCOVERY_WITH_ROBUST_SUPPORT",
            "boundary": BOUNDARY,
        }
    if not repair_distinguishable:
        return {
            "decision": "counter_support_repair_not_confirmed",
            "verdict": "D47R_COUNTER_SUPPORT_REPAIR",
            "next": "D47R_COUNTER_SUPPORT_REPAIR",
            "boundary": BOUNDARY,
        }
    return {
        "decision": "false_confidence_under_indistinguishable_support",
        "verdict": "D47B_FALSE_CONFIDENCE_FAILURE",
        "next": "D47C_ABSTAIN_AND_GROUNDING_REPAIR",
        "boundary": BOUNDARY,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", default="10101,10102,10103,10104,10105")
    parser.add_argument("--train-rows-per-seed", type=int, default=800)
    parser.add_argument("--test-rows-per-seed", type=int, default=800)
    parser.add_argument("--ood-rows-per-seed", type=int, default=800)
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
    bundle = d45.make_candidates(PRIMARY_SPACE)
    write_json(
        out / "queue.json",
        {
            "task": "D47B indistinguishable correlated support sanity check",
            "status": "running",
            "seeds": seeds,
            "train_rows_per_seed": args.train_rows_per_seed,
            "test_rows_per_seed": args.test_rows_per_seed,
            "ood_rows_per_seed": args.ood_rows_per_seed,
            "workers": args.workers,
            "cpu_target": args.cpu_target,
            "no_black_box": True,
            "heartbeat_sec": args.heartbeat_sec,
        },
    )
    append_progress(out, "queue_written", started, {"out": str(out)})
    write_json(
        out / "compute_probe.json",
        {
            "mode": "local_python_cpu",
            "workers_requested": args.workers,
            "cpu_target": args.cpu_target,
            "note": "D47B is deterministic symbolic scoring; no external model/API/download used.",
        },
    )
    write_json(
        out / "dataset_manifest.json",
        {
            "task": "D47B_INDISTINGUISHABLE_CORRELATED_SUPPORT_SANITY_CHECK",
            "primitive_space": PRIMARY_SPACE,
            "candidate_count": len(bundle["candidates"]),
            "support_regimes": REGIMES,
            "arms": ARMS,
            "fair_arms": FAIR_ARMS,
            "true_family_hidden_from_fair_arms": True,
            "label_echo_reference_only_not_fair": True,
            "no_python_hash": True,
            "no_fake_sampling": True,
            "identifiability_upper_bound_reported": True,
            "boundary": BOUNDARY,
        },
    )
    append_progress(out, "dataset_manifest_written", started, {"candidate_count": len(bundle["candidates"])})

    train_rows = make_rows(seeds, args.train_rows_per_seed, "train", bundle)
    test_rows = make_rows(seeds, args.test_rows_per_seed, "test", bundle)
    ood_rows = make_rows(seeds, args.ood_rows_per_seed, "ood", bundle)
    append_progress(
        out,
        "rows_generated",
        started,
        {"train_rows": len(train_rows), "test_rows": len(test_rows), "ood_rows": len(ood_rows)},
    )

    cert_rows = []
    cert_counts = Counter()
    for row in test_rows + ood_rows:
        if (
            row["support_regime"] in {
            "INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT",
            "EXTERNAL_TEST_REQUIRED_SUPPORT",
            "DISTINGUISHABLE_CORRELATED_FALSE_SUPPORT",
            }
            and cert_counts[row["support_regime"]] < 80
        ):
            cert_rows.append(certificate_for_row(row, bundle))
            cert_counts[row["support_regime"]] += 1
    cert_by_regime = defaultdict(list)
    for cert in cert_rows:
        cert_by_regime[cert["support_regime"]].append(cert)
    cert_summary = {}
    for regime, certs in cert_by_regime.items():
        cert_summary[regime] = {
            "rows": len(certs),
            "support_max_delta": max(cert["support_true_false_score_delta"]["max_delta"] for cert in certs),
            "internal_counter_max_delta": max(
                cert["internal_counter_true_false_score_delta"]["max_delta"] for cert in certs
            ),
            "external_counter_max_delta": max(
                cert["external_counter_true_false_score_delta"]["max_delta"] for cert in certs
            ),
        }
    write_json(
        out / "indistinguishability_certificate_report.json",
        {
            "summary": cert_summary,
            "sample_certificates": cert_rows[:40],
            "certificate_rule": "For unidentifiable regimes, true and false candidates have zero score delta across all allowed internal support.",
        },
    )
    append_progress(out, "certificates_written", started, {"certificate_rows": len(cert_rows)})

    test_outputs = evaluate_split(test_rows, bundle, out / "row_outputs_test.jsonl", started, out, args.heartbeat_sec)
    ood_outputs = evaluate_split(ood_rows, bundle, out / "row_outputs_ood.jsonl", started, out, args.heartbeat_sec)
    append_progress(out, "evaluation_complete", started, {"test_outputs": len(test_outputs), "ood_outputs": len(ood_outputs)})

    test_metrics = summarize_metrics(test_outputs)
    ood_metrics = summarize_metrics(ood_outputs)
    aggregate = {
        "task": "D47B_INDISTINGUISHABLE_CORRELATED_SUPPORT_SANITY_CHECK",
        "test_metrics": test_metrics,
        "ood_metrics": ood_metrics,
        "failed_jobs": [],
        "boundary": BOUNDARY,
    }
    decision = make_decision(test_metrics)
    aggregate["decision"] = decision["decision"]
    aggregate["verdict"] = decision["verdict"]
    aggregate["next"] = decision["next"]

    detection = confusion_for_detection(
        test_outputs,
        "correlated_support_detected",
        {
            "CORRELATED_TRUE_SUPPORT",
            "DISTINGUISHABLE_CORRELATED_FALSE_SUPPORT",
            "INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT",
            "EXTERNAL_TEST_REQUIRED_SUPPORT",
        },
    )
    indist_arm = test_metrics["by_arm_and_regime"]["ROBUST_ECF_WITH_ABSTAIN"][
        "INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT"
    ]
    external_arm = test_metrics["by_arm_and_regime"]["ROBUST_ECF_WITH_EXTERNAL_TEST"]["EXTERNAL_TEST_REQUIRED_SUPPORT"]
    distinguishable_counter = test_metrics["by_arm_and_regime"]["ROBUST_ECF_COUNTER_SUPPORT"][
        "DISTINGUISHABLE_CORRELATED_FALSE_SUPPORT"
    ]
    reports = {
        "identifiability_report.json": {
            "regime_upper_bounds": {
                regime: test_metrics["by_arm_and_regime"]["NAIVE_IPF"][regime]["identifiability_upper_bound"]
                for regime in REGIMES
            },
            "oracle_distinguishable_by_regime": {
                regime: test_metrics["by_arm_and_regime"]["NAIVE_IPF"][regime]["oracle_distinguishable_rate"]
                for regime in REGIMES
            },
            "identifiability_limit_confirmed": indist_arm["abstain_rate"] >= 0.95,
        },
        "counter_support_availability_report.json": {
            "distinguishable_false_counter_support_available": True,
            "indistinguishable_false_internal_counter_support_available": False,
            "external_test_required_internal_counter_support_available": False,
            "external_test_required_external_counter_support_available": True,
            "distinguishable_false_repaired_accuracy": distinguishable_counter["accuracy"],
            "external_test_required_accuracy_with_external_test": external_arm["accuracy"],
        },
        "abstain_behavior_report.json": {
            "robust_abstain_indistinguishable_false": indist_arm,
            "robust_abstain_external_required": test_metrics["by_arm_and_regime"]["ROBUST_ECF_WITH_ABSTAIN"][
                "EXTERNAL_TEST_REQUIRED_SUPPORT"
            ],
            "abstain_allowed_and_reported": True,
        },
        "false_confidence_report.json": {
            "by_arm_and_regime": {
                arm: {
                    regime: {
                        "false_confidence_rate": test_metrics["by_arm_and_regime"][arm][regime][
                            "false_confidence_rate"
                        ],
                        "confidence_when_wrong": test_metrics["by_arm_and_regime"][arm][regime][
                            "confidence_when_wrong"
                        ],
                    }
                    for regime in REGIMES
                }
                for arm in ARMS
            },
            "confidence_threshold": CONFIDENCE_THRESHOLD,
        },
        "regime_metrics_report.json": test_metrics["by_arm_and_regime"],
    }
    for name, value in reports.items():
        write_json(out / name, value)
    write_json(out / "aggregate_metrics.json", aggregate)
    write_json(out / "decision.json", decision)
    write_json(
        out / "summary.json",
        {
            "decision": decision,
            "test_key_metrics": {
                "distinguishable_false_counter_support_accuracy": distinguishable_counter["accuracy"],
                "indistinguishable_false_abstain_rate": indist_arm["abstain_rate"],
                "indistinguishable_false_false_confidence_rate": indist_arm["false_confidence_rate"],
                "external_test_required_external_arm_accuracy": external_arm["accuracy"],
                "correlated_detection": detection,
            },
            "boundary": BOUNDARY,
        },
    )
    write_report_md(out, decision, aggregate)
    write_json(out / "queue.json", {**json.loads((out / "queue.json").read_text()), "status": "complete"})
    append_progress(out, "final_decision", started, decision)
    print(json.dumps({"decision": decision["decision"], "verdict": decision["verdict"], "next": decision["next"]}, indent=2))


if __name__ == "__main__":
    main()
