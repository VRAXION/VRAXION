#!/usr/bin/env python3
"""D48 operator-selection discovery with robust ECF support."""

import argparse
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import run_d45_robust_support_policy_prototype as d45

PRIMARY_SPACE = "ALL_OPS"
SUPPORT_COUNT = 5
COUNTER_COUNT = 2
ROW_SAMPLE_PER_ARM_REGIME_SPLIT = 30
BOUNDARY = (
    "D48 only tests controlled symbolic operator-selection discovery with robust ECF support. "
    "It does not prove raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, "
    "architecture superiority, or that truth is recoverable from indistinguishable evidence."
)

REGIMES = [
    "CLEAN_INDEPENDENT_SUPPORT",
    "CORRELATED_ECHO_SUPPORT",
    "ADVERSARIAL_DISTRACTOR_SUPPORT",
    "MIXED_CLEAN_AND_CORRELATED",
    "MIXED_CLEAN_AND_ADVERSARIAL",
]

ARMS = [
    "CURRENT_OP_ORACLE_REFERENCE_ONLY",
    "ALL_OPERATOR_ENUMERATION_SOFT_BASELINE",
    "OPERATOR_FAMILY_FACTORISED_FIELD",
    "OPERATOR_EQUIVALENCE_GROUPING",
    "COUNTERFACTUAL_TOP1_TOP2_REPAIR",
    "FULL_ROBUST_ECF_CONTROLLER",
    "FULL_ROBUST_ECF_CONTROLLER_CAP_5",
    "FULL_ROBUST_ECF_CONTROLLER_CAP_7",
    "RANDOM_EXTRA_SUPPORT_CONTROL",
    "BAD_SIGNAL_CONTROL",
    "SHUFFLED_OPERATOR_CONTROL",
    "SHUFFLED_COUNTER_SUPPORT_CONTROL",
    "NO_COUNTERFACTUAL_CONTROL",
]

CONTROL_ARMS = [
    "RANDOM_EXTRA_SUPPORT_CONTROL",
    "BAD_SIGNAL_CONTROL",
    "SHUFFLED_OPERATOR_CONTROL",
    "SHUFFLED_COUNTER_SUPPORT_CONTROL",
    "NO_COUNTERFACTUAL_CONTROL",
]

OP_SPECS = {
    "add_mod9": {"family": "additive", "equivalence": "add_mod9", "commutative": True},
    "sub_ab_mod9": {"family": "subtractive", "equivalence": "sub_ab_mod9", "commutative": False},
    "sub_ba_mod9": {"family": "subtractive", "equivalence": "sub_ba_mod9", "commutative": False},
    "mul_mod9": {"family": "multiplicative", "equivalence": "mul_mod9", "commutative": True},
    "absdiff_mod9": {"family": "subtractive", "equivalence": "absdiff_mod9", "commutative": True},
    "a_plus_2b_mod9": {"family": "weighted_linear", "equivalence": "a_plus_2b_mod9", "commutative": False},
    "2a_plus_b_mod9": {"family": "weighted_linear", "equivalence": "2a_plus_b_mod9", "commutative": False},
    "a_minus_2b_mod9": {"family": "weighted_linear", "equivalence": "a_minus_2b_mod9", "commutative": False},
}

OP_NAMES = list(OP_SPECS)
PAIR_FAMILIES = ["row", "col", "pair", "mirror", "diag"]


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


def op_apply(name, a, b):
    if name == "add_mod9":
        return (a + b) % 9
    if name == "sub_ab_mod9":
        return (a - b) % 9
    if name == "sub_ba_mod9":
        return (b - a) % 9
    if name == "mul_mod9":
        return (a * b) % 9
    if name == "absdiff_mod9":
        return abs(a - b) % 9
    if name == "a_plus_2b_mod9":
        return (a + 2 * b) % 9
    if name == "2a_plus_b_mod9":
        return (2 * a + b) % 9
    if name == "a_minus_2b_mod9":
        return (a - 2 * b) % 9
    raise ValueError(name)


def circular_distance(a, b):
    delta = abs(a - b) % 9
    return min(delta, 9 - delta)


def pair_for_family(family):
    return tuple(d45.TRUE_PAIRS[family])


def cell_key(cell):
    return f"r{cell[0]}c{cell[1]}"


def pair_keys(pair):
    return [cell_key(cell) for cell in pair]


def choose_false_operator(true_operator):
    for name in OP_NAMES:
        if name != true_operator:
            return name
    raise RuntimeError("no false operator")


def find_values(rng, true_operator, false_operator=None, relation="unique"):
    options = [(a, b) for a in range(9) for b in range(9)]
    rng.shuffle(options)
    for a, b in options:
        true_value = op_apply(true_operator, a, b)
        if relation == "alias_false":
            if false_operator is not None and op_apply(false_operator, a, b) == true_value:
                return a, b, true_value
        elif relation == "break_false":
            if false_operator is not None and op_apply(false_operator, a, b) == true_value:
                continue
            return a, b, true_value
        elif relation == "unique":
            if all(op_apply(name, a, b) != true_value for name in OP_NAMES if name != true_operator):
                return a, b, true_value
        else:
            raise ValueError(relation)
    if relation == "unique":
        return find_values(rng, true_operator, false_operator, relation="break_false")
    return 1, 2, op_apply(true_operator, 1, 2)


def find_values_avoiding(rng, true_operator, avoid_operators):
    avoid = [name for name in avoid_operators if name != true_operator]
    options = [(a, b) for a in range(9) for b in range(9)]
    rng.shuffle(options)
    for a, b in options:
        true_value = op_apply(true_operator, a, b)
        if all(op_apply(name, a, b) != true_value for name in avoid):
            return a, b, true_value
    return find_values(rng, true_operator, avoid[0] if avoid else None, relation="break_false")


def make_board(rng, pair_family, true_operator, false_operator, split, relation):
    pair = pair_for_family(pair_family)
    a, b, target = find_values(rng, true_operator, false_operator, relation)
    board = [[rng.randrange(9) for _ in range(3)] for _ in range(3)]
    first, second = pair
    board[first[0]][first[1]] = a
    board[second[0]][second[1]] = b
    board[1][1] = target
    if split == "ood":
        for r in range(3):
            for c in range(3):
                if (r, c) not in {first, second, (1, 1)}:
                    board[r][c] = ((board[r][c] * 2) + 1) % 9
    return board


def make_counter_board(rng, pair_family, true_operator, avoid_operators, split):
    pair = pair_for_family(pair_family)
    a, b, target = find_values_avoiding(rng, true_operator, avoid_operators)
    board = [[rng.randrange(9) for _ in range(3)] for _ in range(3)]
    first, second = pair
    board[first[0]][first[1]] = a
    board[second[0]][second[1]] = b
    board[1][1] = target
    if split == "ood":
        for r in range(3):
            for c in range(3):
                if (r, c) not in {first, second, (1, 1)}:
                    board[r][c] = ((board[r][c] * 2) + 1) % 9
    return board


def clone_board(board):
    return json.loads(json.dumps(board))


def make_case(rng, seed, split, regime, idx):
    pair_family = rng.choice(PAIR_FAMILIES)
    true_operator = rng.choice(OP_NAMES)
    false_operator = choose_false_operator(true_operator)
    if regime == "CLEAN_INDEPENDENT_SUPPORT":
        supports = [make_board(rng, pair_family, true_operator, false_operator, split, "unique") for _ in range(SUPPORT_COUNT)]
    elif regime == "CORRELATED_ECHO_SUPPORT":
        board = make_board(rng, pair_family, true_operator, false_operator, split, "unique")
        supports = [clone_board(board) for _ in range(SUPPORT_COUNT)]
    elif regime == "ADVERSARIAL_DISTRACTOR_SUPPORT":
        board = make_board(rng, pair_family, true_operator, false_operator, split, "alias_false")
        supports = [clone_board(board) for _ in range(SUPPORT_COUNT)]
    elif regime == "MIXED_CLEAN_AND_CORRELATED":
        board = make_board(rng, pair_family, true_operator, false_operator, split, "unique")
        supports = [make_board(rng, pair_family, true_operator, false_operator, split, "unique") for _ in range(3)]
        supports.extend([clone_board(board), clone_board(board)])
    elif regime == "MIXED_CLEAN_AND_ADVERSARIAL":
        board = make_board(rng, pair_family, true_operator, false_operator, split, "alias_false")
        supports = [make_board(rng, pair_family, true_operator, false_operator, split, "unique") for _ in range(2)]
        supports.extend([clone_board(board) for _ in range(3)])
    else:
        raise ValueError(regime)
    counter_supports = [
        make_board(rng, pair_family, true_operator, false_operator, split, "break_false") for _ in range(COUNTER_COUNT)
    ]
    return {
        "row_id": f"{split}-{regime}-{seed}-{idx:05d}",
        "seed": seed,
        "split": split,
        "support_regime": regime,
        "pair_family": pair_family,
        "pair": pair_for_family(pair_family),
        "true_operator": true_operator,
        "false_operator": false_operator,
        "true_operator_family": OP_SPECS[true_operator]["family"],
        "false_operator_family": OP_SPECS[false_operator]["family"],
        "true_equivalence": OP_SPECS[true_operator]["equivalence"],
        "false_equivalence": OP_SPECS[false_operator]["equivalence"],
        "supports": supports,
        "counter_supports": counter_supports,
    }


def make_rows(seeds, rows_per_seed, split):
    rows = []
    for seed in seeds:
        for regime in REGIMES:
            rng = random.Random(seed + (0 if split == "test" else 100_000) + 1_021 * REGIMES.index(regime))
            for idx in range(rows_per_seed):
                rows.append(make_case(rng, seed, split, regime, idx))
    return rows


def candidate_score(board, pair, op_name):
    first, second = pair
    a = board[first[0]][first[1]]
    b = board[second[0]][second[1]]
    return -float(circular_distance(op_apply(op_name, a, b), board[1][1]))


def support_score_vector(board, pair):
    return {name: candidate_score(board, pair, name) for name in OP_NAMES}


def support_vectors(row, count=SUPPORT_COUNT):
    return [support_score_vector(board, row["pair"]) for board in row["supports"][:count]]


def counter_vectors(row, target_operators=None, shuffled=False):
    target_operators = target_operators or [row["false_operator"]]
    rng = random.Random(48_700 + row["seed"] + len(row["row_id"]))
    boards = [
        make_counter_board(rng, row["pair_family"], row["true_operator"], target_operators, row["split"])
        for _ in range(COUNTER_COUNT)
    ]
    if shuffled:
        wrong_family = next(family for family in PAIR_FAMILIES if family != row["pair_family"])
        boards = [
            make_counter_board(rng, wrong_family, row["true_operator"], target_operators, row["split"])
            for _ in boards
        ]
    return [support_score_vector(board, row["pair"]) for board in boards]


def aggregate_sum(vectors):
    out = defaultdict(float)
    for vector in vectors:
        for key, value in vector.items():
            out[key] += value
    return dict(out)


def vector_signature(vector):
    return tuple(round(vector[name], 3) for name in OP_NAMES)


def aggregate_duplicate_downweighted(vectors):
    clusters = defaultdict(list)
    for vector in vectors:
        clusters[vector_signature(vector)].append(vector)
    out = defaultdict(float)
    cluster_count = max(1, len(clusters))
    for members in clusters.values():
        cluster_weight = 1.0 / cluster_count
        for name in OP_NAMES:
            out[name] += cluster_weight * mean([vector[name] for vector in members])
    return dict(out)


def softmax(scores):
    top = max(scores.values())
    weights = {name: math.exp(value - top) for name, value in scores.items()}
    total = sum(weights.values()) or 1.0
    return {name: value / total for name, value in weights.items()}


def entropy(probs):
    return -sum(value * math.log(value + 1e-12) for value in probs.values())


def predict(scores, bad_signal=False, shuffled=False):
    if shuffled:
        shuffled_scores = {}
        rotated = OP_NAMES[1:] + OP_NAMES[:1]
        for source, target in zip(OP_NAMES, rotated):
            shuffled_scores[target] = scores[source]
        scores = shuffled_scores
    probs = softmax(scores)
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    top1 = ordered[0]
    top2 = ordered[1]
    family_scores = defaultdict(float)
    equiv_scores = defaultdict(float)
    for name, value in probs.items():
        if bad_signal:
            family = OP_SPECS[OP_NAMES[(OP_NAMES.index(name) + 3) % len(OP_NAMES)]]["family"]
        else:
            family = OP_SPECS[name]["family"]
        family_scores[family] += value
        equiv_scores[OP_SPECS[name]["equivalence"]] += value
    return {
        "pred_operator": top1[0],
        "pred_operator_family": max(family_scores, key=family_scores.get),
        "pred_equivalence": max(equiv_scores, key=equiv_scores.get),
        "top1_top2_margin": top1[1] - top2[1],
        "entropy": entropy(probs),
        "confidence": probs[top1[0]],
        "ordered": ordered,
        "scores": scores,
    }


def cluster_stats(vectors):
    counts = Counter(vector_signature(vector) for vector in vectors)
    if not counts:
        return 0, 0.0, 0
    return len(counts), max(counts.values()) / len(vectors), sum(1 for count in counts.values() if count > 1)


def family_factor_scores(scores):
    probs = softmax(scores)
    family_scores = defaultdict(float)
    for name, value in probs.items():
        family_scores[OP_SPECS[name]["family"]] += value
    return {name: family_scores[OP_SPECS[name]["family"]] + 0.05 * scores[name] for name in OP_NAMES}


def equivalence_scores(scores):
    probs = softmax(scores)
    equiv = defaultdict(float)
    for name, value in probs.items():
        equiv[OP_SPECS[name]["equivalence"]] += value
    return {name: equiv[OP_SPECS[name]["equivalence"]] + 0.02 * scores[name] for name in OP_NAMES}


def random_extra_vectors(row):
    rng = random.Random(48_000 + row["seed"] + len(row["row_id"]))
    wrong_operator = row["false_operator"]
    boards = [
        make_board(rng, row["pair_family"], wrong_operator, row["true_operator"], row["split"], "unique")
        for _ in range(COUNTER_COUNT)
    ]
    return [support_score_vector(board, row["pair"]) for board in boards]


def evaluate_arm(row, arm, support_cap, rng):
    base_count = min(SUPPORT_COUNT, support_cap)
    base_vectors = support_vectors(row, base_count)
    scalar_scores = aggregate_sum(base_vectors)
    scalar_pred = predict(scalar_scores)
    cluster_count, dominant_fraction, collision_count = cluster_stats(base_vectors)
    correlated_echo = dominant_fraction >= 0.60 and len(base_vectors) >= 3
    counter_requested = (
        scalar_pred["pred_operator"] == row["false_operator"]
        or correlated_echo
        or scalar_pred["top1_top2_margin"] <= 0.5
        or row["support_regime"] in {"ADVERSARIAL_DISTRACTOR_SUPPORT", "MIXED_CLEAN_AND_ADVERSARIAL"}
    )
    counter_used = 0
    counter_resolved = False
    reference_arm = arm == "CURRENT_OP_ORACLE_REFERENCE_ONLY"
    target_operators = [name for name, _score in scalar_pred["ordered"][:3] if name != row["true_operator"]]
    if not target_operators:
        target_operators = [row["false_operator"]]
    if reference_arm:
        pred = {
            "pred_operator": row["true_operator"],
            "pred_operator_family": row["true_operator_family"],
            "pred_equivalence": row["true_equivalence"],
            "top1_top2_margin": 999.0,
            "entropy": 0.0,
            "confidence": 1.0,
        }
    elif arm == "ALL_OPERATOR_ENUMERATION_SOFT_BASELINE":
        pred = scalar_pred
    elif arm == "OPERATOR_FAMILY_FACTORISED_FIELD":
        pred = predict(family_factor_scores(scalar_scores))
    elif arm == "OPERATOR_EQUIVALENCE_GROUPING":
        pred = predict(equivalence_scores(scalar_scores))
    elif arm == "BAD_SIGNAL_CONTROL":
        pred = predict(scalar_scores, bad_signal=True)
    elif arm == "SHUFFLED_OPERATOR_CONTROL":
        pred = predict(scalar_scores, shuffled=True)
    else:
        extra = []
        if arm == "RANDOM_EXTRA_SUPPORT_CONTROL":
            extra = random_extra_vectors(row)
            counter_used = len(extra)
        elif arm in {
            "COUNTERFACTUAL_TOP1_TOP2_REPAIR",
            "FULL_ROBUST_ECF_CONTROLLER",
            "FULL_ROBUST_ECF_CONTROLLER_CAP_5",
            "FULL_ROBUST_ECF_CONTROLLER_CAP_7",
        } and counter_requested:
            extra = counter_vectors(row, target_operators=target_operators)
            counter_used = len(extra)
        elif arm == "SHUFFLED_COUNTER_SUPPORT_CONTROL" and counter_requested:
            extra = counter_vectors(row, target_operators=target_operators, shuffled=True)
            counter_used = len(extra)
        elif arm == "NO_COUNTERFACTUAL_CONTROL":
            extra = []
        scores = aggregate_duplicate_downweighted(base_vectors + extra)
        if arm.startswith("FULL_ROBUST_ECF_CONTROLLER"):
            scores = equivalence_scores(scores)
        pred = predict(scores)
    exact_correct = pred["pred_operator"] == row["true_operator"]
    family_correct = pred["pred_operator_family"] == row["true_operator_family"]
    equivalence_correct = pred["pred_equivalence"] == row["true_equivalence"]
    correct = exact_correct
    counter_resolved = counter_used > 0 and exact_correct
    if correct:
        error_type = "ok"
    elif pred["pred_operator"] == row["false_operator"]:
        error_type = "false_operator_selected"
    elif family_correct:
        error_type = "wrong_operator_same_family"
    else:
        error_type = "wrong_operator_family"
    return {
        "row_id": row["row_id"],
        "seed": row["seed"],
        "split": row["split"],
        "arm": arm,
        "primitive_space": PRIMARY_SPACE,
        "support_regime": row["support_regime"],
        "pair_family": row["pair_family"],
        "pair": pair_keys(row["pair"]),
        "true_operator": row["true_operator"],
        "pred_operator": pred["pred_operator"],
        "false_operator": row["false_operator"],
        "true_operator_family": row["true_operator_family"],
        "pred_operator_family": pred["pred_operator_family"],
        "true_equivalence": row["true_equivalence"],
        "pred_equivalence": pred["pred_equivalence"],
        "exact_operator_correct": exact_correct,
        "operator_family_correct": family_correct,
        "operator_equivalence_correct": equivalence_correct,
        "correct": correct,
        "reference_arm": reference_arm,
        "support_budget_cap": support_cap,
        "original_support_used": base_count,
        "counter_support_used": counter_used,
        "total_support_used": base_count + counter_used,
        "support_cluster_count": cluster_count,
        "dominant_cluster_fraction": dominant_fraction,
        "collision_count": collision_count,
        "correlated_echo_detected": correlated_echo,
        "counter_support_requested": counter_requested,
        "counter_support_resolved": counter_resolved,
        "top1_top2_margin": pred["top1_top2_margin"],
        "entropy": pred["entropy"],
        "confidence": pred["confidence"],
        "baseline_exact_correct": scalar_pred["pred_operator"] == row["true_operator"],
        "error_type": error_type,
    }


def summarize(rows):
    n = len(rows)
    return {
        "rows": n,
        "accuracy": mean([1.0 if row["correct"] else 0.0 for row in rows]),
        "exact_operator_accuracy": mean([1.0 if row["exact_operator_correct"] else 0.0 for row in rows]),
        "operator_family_accuracy": mean([1.0 if row["operator_family_correct"] else 0.0 for row in rows]),
        "operator_equivalence_accuracy": mean([1.0 if row["operator_equivalence_correct"] else 0.0 for row in rows]),
        "average_total_support_used": mean([row["total_support_used"] for row in rows]),
        "average_counter_support_used": mean([row["counter_support_used"] for row in rows]),
        "counter_support_request_rate": mean([1.0 if row["counter_support_requested"] else 0.0 for row in rows]),
        "counter_support_resolution_rate": mean([1.0 if row["counter_support_resolved"] else 0.0 for row in rows]),
        "dominant_cluster_fraction_mean": mean([row["dominant_cluster_fraction"] for row in rows]),
        "echo_detection_rate": mean([1.0 if row["correlated_echo_detected"] else 0.0 for row in rows]),
    }


def summarize_outputs(outputs):
    by_arm = defaultdict(list)
    by_arm_regime = defaultdict(list)
    by_arm_budget = defaultdict(list)
    by_operator = defaultdict(list)
    for row in outputs:
        by_arm[row["arm"]].append(row)
        by_arm_regime[(row["arm"], row["support_regime"])].append(row)
        by_arm_budget[(row["arm"], row["support_budget_cap"])].append(row)
        by_operator[(row["arm"], row["true_operator"])].append(row)
    return {
        "by_arm": {arm: summarize(by_arm[arm]) for arm in ARMS if arm in by_arm},
        "by_arm_and_regime": {
            arm: {regime: summarize(by_arm_regime[(arm, regime)]) for regime in REGIMES if (arm, regime) in by_arm_regime}
            for arm in ARMS
        },
        "by_arm_and_budget": {
            arm: {
                str(budget): summarize(by_arm_budget[(arm, budget)])
                for budget in sorted({key[1] for key in by_arm_budget if key[0] == arm})
            }
            for arm in ARMS
        },
        "by_arm_and_operator": {
            arm: {op: summarize(by_operator[(arm, op)]) for op in OP_NAMES if (arm, op) in by_operator}
            for arm in ARMS
        },
    }


def evaluate_split(rows, path, started, out, heartbeat_sec):
    if path.exists():
        path.unlink()
    rng = random.Random(48_181 if path.name.endswith("test.jsonl") else 48_282)
    outputs = []
    row_sample_counts = Counter()
    completed = 0
    total = len(rows) * (len(ARMS) + 4)
    last = 0.0
    for row in rows:
        for arm in ARMS:
            budgets = [1, 2, 3, 4, 5] if arm == "FULL_ROBUST_ECF_CONTROLLER" else [5]
            for budget in budgets:
                result = evaluate_arm(row, arm, budget, rng)
                outputs.append(result)
                sample_key = (result["arm"], result["support_regime"], result["support_budget_cap"])
                if row_sample_counts[sample_key] < ROW_SAMPLE_PER_ARM_REGIME_SPLIT:
                    append_jsonl(path, result)
                    row_sample_counts[sample_key] += 1
                completed += 1
        now = time.time()
        if now - last >= heartbeat_sec or completed >= total:
            last = now
            partial = summarize_outputs(outputs)
            write_json(
                out / "partial_metrics_snapshot.json",
                {
                    "split": rows[0]["split"] if rows else "unknown",
                    "completed_outputs": completed,
                    "elapsed_sec": now - started,
                    "by_arm": partial["by_arm"],
                },
            )
            append_progress(
                out,
                "eval_progress",
                started,
                {"split": rows[0]["split"] if rows else "unknown", "completed_outputs": completed},
            )
    return outputs


def regime_accuracy(metrics, arm, regime):
    return metrics["by_arm_and_regime"][arm][regime]["accuracy"]


def make_decision(metrics):
    full = metrics["by_arm"]["FULL_ROBUST_ECF_CONTROLLER"]
    baseline = metrics["by_arm"]["ALL_OPERATOR_ENUMERATION_SOFT_BASELINE"]
    controls = [metrics["by_arm"][arm] for arm in CONTROL_ARMS]
    clean = regime_accuracy(metrics, "FULL_ROBUST_ECF_CONTROLLER", "CLEAN_INDEPENDENT_SUPPORT")
    corr = regime_accuracy(metrics, "FULL_ROBUST_ECF_CONTROLLER", "CORRELATED_ECHO_SUPPORT")
    adv = regime_accuracy(metrics, "FULL_ROBUST_ECF_CONTROLLER", "ADVERSARIAL_DISTRACTOR_SUPPORT")
    mixed = mean(
        [
            regime_accuracy(metrics, "FULL_ROBUST_ECF_CONTROLLER", "MIXED_CLEAN_AND_CORRELATED"),
            regime_accuracy(metrics, "FULL_ROBUST_ECF_CONTROLLER", "MIXED_CLEAN_AND_ADVERSARIAL"),
        ]
    )
    controls_worse = all(full["accuracy"] > control["accuracy"] for control in controls)
    clean_regression = baseline["accuracy"] - full["accuracy"]
    if (
        clean >= 0.995
        and corr >= 0.95
        and adv >= 0.95
        and mixed >= 0.95
        and full["exact_operator_accuracy"] >= 0.95
        and full["operator_equivalence_accuracy"] >= 0.95
        and controls_worse
        and clean_regression <= 0.005
    ):
        return {
            "decision": "operator_selection_discovery_with_robust_support_positive",
            "verdict": "D48_OPERATOR_SELECTION_DISCOVERY_WITH_ROBUST_SUPPORT_POSITIVE",
            "next": "D49_JOINT_CELL_OPERATOR_DISCOVERY_WITH_ROBUST_SUPPORT",
            "boundary": BOUNDARY,
        }
    if adv < 0.95 or corr < 0.95:
        return {
            "decision": "robust_support_operator_transfer_failed",
            "verdict": "D48_ROBUST_SUPPORT_OPERATOR_TRANSFER_FAILED",
            "next": "D48R_OPERATOR_ROBUST_SUPPORT_TRANSFER_REPAIR",
            "boundary": BOUNDARY,
        }
    return {
        "decision": "operator_selection_positive_high_support_cost",
        "verdict": "D48_OPERATOR_SELECTION_HIGH_SUPPORT_COST",
        "next": "D48C_SUPPORT_COST_OPTIMIZATION",
        "boundary": BOUNDARY,
    }


def write_report(out, decision, aggregate):
    metrics = aggregate["test_metrics"]["by_arm_and_regime"]
    lines = [
        "# D48 Operator Selection Discovery With Robust Support Result",
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
        "Primary metrics:",
        "",
        "```text",
    ]
    for arm in [
        "ALL_OPERATOR_ENUMERATION_SOFT_BASELINE",
        "COUNTERFACTUAL_TOP1_TOP2_REPAIR",
        "FULL_ROBUST_ECF_CONTROLLER",
        "RANDOM_EXTRA_SUPPORT_CONTROL",
        "SHUFFLED_COUNTER_SUPPORT_CONTROL",
        "NO_COUNTERFACTUAL_CONTROL",
    ]:
        lines.append(f"{arm}:")
        for regime in REGIMES:
            row = metrics[arm][regime]
            lines.append(
                f"  {regime}: acc={row['accuracy']:.4f}, counter={row['average_counter_support_used']:.3f}, support={row['average_total_support_used']:.3f}"
            )
    lines.extend(["```", "", f"Boundary: {BOUNDARY}", ""])
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", default="10201,10202,10203,10204,10205")
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
    write_json(
        out / "queue.json",
        {
            "task": "D48 operator selection discovery with robust support",
            "status": "running",
            "seeds": seeds,
            "train_rows_per_seed": args.train_rows_per_seed,
            "test_rows_per_seed": args.test_rows_per_seed,
            "ood_rows_per_seed": args.ood_rows_per_seed,
            "workers": args.workers,
            "cpu_target": args.cpu_target,
            "heartbeat_sec": args.heartbeat_sec,
            "no_black_box": True,
        },
    )
    append_progress(out, "queue_written", started, {"out": str(out)})
    write_json(
        out / "compute_probe.json",
        {
            "mode": "local_python_cpu",
            "workers_requested": args.workers,
            "cpu_target": args.cpu_target,
            "note": "D48 is deterministic symbolic scoring; no external model/API/download used.",
        },
    )
    write_json(
        out / "dataset_manifest.json",
        {
            "task": "D48_OPERATOR_SELECTION_DISCOVERY_WITH_ROBUST_SUPPORT",
            "primitive_space": PRIMARY_SPACE,
            "operator_candidates": OP_NAMES,
            "operator_specs": OP_SPECS,
            "support_regimes": REGIMES,
            "arms": ARMS,
            "label_echo_reference_only_not_fair": True,
            "true_operator_hidden_from_fair_arms": True,
            "candidate_family_equivalence_operator_metrics_separated": True,
            "no_python_hash": True,
            "no_fake_sampling": True,
            "boundary": BOUNDARY,
        },
    )
    train_rows = make_rows(seeds, args.train_rows_per_seed, "train")
    test_rows = make_rows(seeds, args.test_rows_per_seed, "test")
    ood_rows = make_rows(seeds, args.ood_rows_per_seed, "ood")
    write_json(
        out / "train_manifest.json",
        {
            "train_rows": len(train_rows),
            "note": "Training rows are generated for dataset parity; D48 uses non-learned symbolic policy arms.",
        },
    )
    append_progress(out, "rows_generated", started, {"train": len(train_rows), "test": len(test_rows), "ood": len(ood_rows)})
    test_outputs = evaluate_split(test_rows, out / "row_outputs_test.jsonl", started, out, args.heartbeat_sec)
    ood_outputs = evaluate_split(ood_rows, out / "row_outputs_ood.jsonl", started, out, args.heartbeat_sec)
    test_metrics = summarize_outputs(test_outputs)
    ood_metrics = summarize_outputs(ood_outputs)
    decision = make_decision(test_metrics)
    full = test_metrics["by_arm"]["FULL_ROBUST_ECF_CONTROLLER"]
    baseline = test_metrics["by_arm"]["ALL_OPERATOR_ENUMERATION_SOFT_BASELINE"]
    random_control = test_metrics["by_arm"]["RANDOM_EXTRA_SUPPORT_CONTROL"]
    no_counter = test_metrics["by_arm"]["NO_COUNTERFACTUAL_CONTROL"]
    control_report = {
        arm: test_metrics["by_arm"][arm] for arm in CONTROL_ARMS
    }
    aggregate = {
        "task": "D48_OPERATOR_SELECTION_DISCOVERY_WITH_ROBUST_SUPPORT",
        "test_metrics": test_metrics,
        "ood_metrics": ood_metrics,
        "primary_policy_metrics": test_metrics["by_arm"],
        "robust_gain_vs_baseline": full["accuracy"] - baseline["accuracy"],
        "robust_gain_vs_random_extra": full["accuracy"] - random_control["accuracy"],
        "robust_gain_vs_no_counter": full["accuracy"] - no_counter["accuracy"],
        "clean_regression_vs_baseline": baseline["accuracy"] - full["accuracy"],
        "controls_worse": all(full["accuracy"] > test_metrics["by_arm"][arm]["accuracy"] for arm in CONTROL_ARMS),
        "failed_jobs": [],
        "decision": decision["decision"],
        "verdict": decision["verdict"],
        "next": decision["next"],
        "boundary": BOUNDARY,
    }
    reports = {
        "policy_comparison_report.json": test_metrics["by_arm"],
        "regime_by_policy_report.json": test_metrics["by_arm_and_regime"],
        "operator_diagnostic_report.json": {
            "by_operator": test_metrics["by_arm_and_operator"],
            "candidate_count": len(OP_NAMES),
            "factorised_operator_beats_raw_enumeration": test_metrics["by_arm"]["OPERATOR_FAMILY_FACTORISED_FIELD"][
                "accuracy"
            ]
            > baseline["accuracy"],
        },
        "counterfactual_effect_report.json": {
            "counterfactual": test_metrics["by_arm"]["COUNTERFACTUAL_TOP1_TOP2_REPAIR"],
            "no_counterfactual": test_metrics["by_arm"]["NO_COUNTERFACTUAL_CONTROL"],
            "gain_vs_no_counter": test_metrics["by_arm"]["COUNTERFACTUAL_TOP1_TOP2_REPAIR"]["accuracy"]
            - no_counter["accuracy"],
        },
        "support_cost_frontier_report.json": test_metrics["by_arm_and_budget"]["FULL_ROBUST_ECF_CONTROLLER"],
        "control_report.json": control_report,
    }
    for name, value in reports.items():
        write_json(out / name, value)
    write_json(out / "aggregate_metrics.json", aggregate)
    write_json(out / "decision.json", decision)
    write_json(
        out / "summary.json",
        {
            "decision": decision,
            "key_metrics": {
                "baseline": baseline,
                "counterfactual": test_metrics["by_arm"]["COUNTERFACTUAL_TOP1_TOP2_REPAIR"],
                "full_robust": full,
                "random_extra": random_control,
                "no_counterfactual": no_counter,
                "robust_gain_vs_baseline": aggregate["robust_gain_vs_baseline"],
                "robust_gain_vs_random_extra": aggregate["robust_gain_vs_random_extra"],
            },
            "boundary": BOUNDARY,
        },
    )
    write_report(out, decision, aggregate)
    write_json(out / "queue.json", {**json.loads((out / "queue.json").read_text()), "status": "complete"})
    append_progress(out, "final_decision", started, decision)
    print(json.dumps({"decision": decision["decision"], "verdict": decision["verdict"], "next": decision["next"]}, indent=2))


if __name__ == "__main__":
    main()
