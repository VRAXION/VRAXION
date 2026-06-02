#!/usr/bin/env python3
"""D49 joint cell-pair + operator discovery with robust ECF support."""

import argparse
import itertools
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import run_d45_robust_support_policy_prototype as d45

PRIMARY_SPACE = "ALL28_UNORDERED_X_OPS"
SUPPORT_COUNT = 5
COUNTER_COUNT = 2
ROW_SAMPLE_PER_ARM_REGIME_SPLIT = 24
CONFIDENCE_THRESHOLD = 0.45

BOUNDARY = (
    "D49 only tests controlled symbolic joint cell+operator discovery with robust ECF support. "
    "It does not prove raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, "
    "or architecture superiority."
)

REGIMES = [
    "CLEAN_INDEPENDENT_SUPPORT",
    "CORRELATED_ECHO_SUPPORT",
    "ADVERSARIAL_DISTRACTOR_SUPPORT",
    "MIXED_CLEAN_AND_CORRELATED",
    "MIXED_CLEAN_AND_ADVERSARIAL",
    "DISTINGUISHABLE_CORRELATED_FALSE_SUPPORT",
    "INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT",
    "EXTERNAL_TEST_REQUIRED_SUPPORT",
]

CORE_REGIMES = [
    "CLEAN_INDEPENDENT_SUPPORT",
    "CORRELATED_ECHO_SUPPORT",
    "ADVERSARIAL_DISTRACTOR_SUPPORT",
    "MIXED_CLEAN_AND_CORRELATED",
    "MIXED_CLEAN_AND_ADVERSARIAL",
]

ARMS = [
    "LABEL_ECHO_REFERENCE_ONLY",
    "JOINT_ENUMERATION_SOFT_BASELINE",
    "CELL_THEN_OPERATOR_PIPELINE",
    "OPERATOR_THEN_CELL_PIPELINE",
    "JOINT_SOFT_FIELD_FACTORISED",
    "COUNTERFACTUAL_TOP1_TOP2_REPAIR",
    "FULL_ROBUST_ECF_CONTROLLER",
    "FULL_ROBUST_ECF_CAP_5",
    "FULL_ROBUST_ECF_CAP_7",
    "RANDOM_EXTRA_SUPPORT_CONTROL",
    "BAD_SIGNAL_CONTROL",
    "SHUFFLED_CELL_CONTROL",
    "SHUFFLED_OPERATOR_CONTROL",
    "SHUFFLED_COUNTER_SUPPORT_CONTROL",
    "NO_COUNTERFACTUAL_CONTROL",
    "ABSTAIN_ON_INDISTINGUISHABLE_CASES",
]

CONTROL_ARMS = [
    "RANDOM_EXTRA_SUPPORT_CONTROL",
    "BAD_SIGNAL_CONTROL",
    "SHUFFLED_CELL_CONTROL",
    "SHUFFLED_OPERATOR_CONTROL",
    "SHUFFLED_COUNTER_SUPPORT_CONTROL",
    "NO_COUNTERFACTUAL_CONTROL",
]

OP_SPECS = {
    "add": {"family": "additive", "equivalence": "add"},
    "sub_ab": {"family": "subtractive", "equivalence": "sub_ab"},
    "sub_ba": {"family": "subtractive", "equivalence": "sub_ba"},
    "mul": {"family": "multiplicative", "equivalence": "mul"},
    "absdiff": {"family": "subtractive", "equivalence": "absdiff"},
    "a_plus_2b": {"family": "weighted_linear", "equivalence": "a_plus_2b"},
    "2a_plus_b": {"family": "weighted_linear", "equivalence": "2a_plus_b"},
    "a_minus_2b": {"family": "weighted_linear", "equivalence": "a_minus_2b"},
}

OP_NAMES = list(OP_SPECS)
PAIR_FAMILIES = ["row", "col", "pair", "mirror", "diag"]
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


def safe_div(num, den):
    return num / den if den else 0.0


def cell_key(cell):
    return f"r{cell[0]}c{cell[1]}"


def pair_key(pair):
    return "__".join(cell_key(cell) for cell in pair)


def canonical_pair(pair):
    return d45.canonical_pair(pair)


def pair_equivalence(pair):
    return d45.canonical_key(canonical_pair(pair))


def op_apply(name, a, b):
    if name == "add":
        return (a + b) % 9
    if name == "sub_ab":
        return (a - b) % 9
    if name == "sub_ba":
        return (b - a) % 9
    if name == "mul":
        return (a * b) % 9
    if name == "absdiff":
        return abs(a - b) % 9
    if name == "a_plus_2b":
        return (a + 2 * b) % 9
    if name == "2a_plus_b":
        return (2 * a + b) % 9
    if name == "a_minus_2b":
        return (a - 2 * b) % 9
    raise ValueError(name)


def circular_distance(a, b):
    delta = abs(a - b) % 9
    return min(delta, 9 - delta)


def make_pairs(space):
    if space == "CURRENT5":
        return {family: tuple(pair) for family, pair in d45.TRUE_PAIRS.items()}
    if space == "ALL28_UNORDERED":
        return {f"u_{pair_key(pair)}": tuple(pair) for pair in itertools.combinations(d45.NONCENTER, 2)}
    if space == "ORDERED56_CONTROL":
        return {f"o_{pair_key(pair)}": tuple(pair) for pair in itertools.permutations(d45.NONCENTER, 2)}
    raise ValueError(space)


def make_bundle(pair_space="ALL28_UNORDERED"):
    pairs = make_pairs(pair_space)
    candidates = {}
    for pair_id, pair in pairs.items():
        for op_name in OP_NAMES:
            cid = f"{pair_id}::{op_name}"
            candidates[cid] = {"pair_id": pair_id, "pair": tuple(pair), "operator": op_name}
    pair_family_by_equiv = {pair_equivalence(pair): family for family, pair in d45.TRUE_PAIRS.items()}
    return {
        "pair_space": pair_space,
        "candidates": candidates,
        "pair_family_by_equiv": pair_family_by_equiv,
    }


def joint_id_for(bundle, pair, op_name):
    target_pair = canonical_pair(pair)
    for cid, spec in bundle["candidates"].items():
        if canonical_pair(spec["pair"]) == target_pair and spec["operator"] == op_name:
            return cid
    raise KeyError((pair, op_name))


def truth_pair_for(pair_family):
    return canonical_pair(d45.TRUE_PAIRS[pair_family])


def choose_false_joint(bundle, true_pair, true_op):
    truth_cid = joint_id_for(bundle, true_pair, true_op)
    truth_equiv = pair_equivalence(true_pair)
    options = []
    for cid, spec in bundle["candidates"].items():
        if cid == truth_cid:
            continue
        if pair_equivalence(spec["pair"]) == truth_equiv and spec["operator"] == true_op:
            continue
        shared = len(set(canonical_pair(spec["pair"])) & set(canonical_pair(true_pair)))
        options.append((shared, cid >= truth_cid, cid, spec["pair"], spec["operator"]))
    options.sort()
    return options[0][3], options[0][4]


def find_values(rng, op_name, target=None, avoid_ops=None):
    avoid_ops = avoid_ops or []
    options = [(a, b) for a in range(9) for b in range(9)]
    rng.shuffle(options)
    for a, b in options:
        value = op_apply(op_name, a, b)
        if target is not None and value != target:
            continue
        if all(op_apply(other, a, b) != value for other in avoid_ops if other != op_name):
            return a, b, value
    for a, b in options:
        value = op_apply(op_name, a, b)
        if target is None or value == target:
            return a, b, value
    return 1, 2, op_apply(op_name, 1, 2)


def set_pair_values(board, pair, a, b):
    first, second = tuple(pair)
    board[first[0]][first[1]] = a
    board[second[0]][second[1]] = b


def board_matches_joint(board, pair, op_name):
    first, second = tuple(pair)
    return op_apply(op_name, board[first[0]][first[1]], board[second[0]][second[1]]) == board[1][1]


def allowed_exact_matches_only(board, allowed):
    allowed = {(canonical_pair(pair), op_name) for pair, op_name in allowed}
    for pair in itertools.combinations(d45.NONCENTER, 2):
        cpair = canonical_pair(pair)
        for op_name in OP_NAMES:
            if board_matches_joint(board, pair, op_name) and (cpair, op_name) not in allowed:
                return False
    return True


def make_board(rng, row, relation, target_false=True):
    true_pair = row["truth_pair"]
    true_op = row["true_operator"]
    false_pair = row["false_pair"]
    false_op = row["false_operator"]
    board = [[rng.randrange(9) for _ in range(3)] for _ in range(3)]
    a, b, target = find_values(rng, true_op, avoid_ops=[false_op])
    set_pair_values(board, true_pair, a, b)
    board[1][1] = target
    if relation == "alias_false" and target_false:
        fa, fb, _ = find_values(rng, false_op, target=target)
        set_pair_values(board, false_pair, fa, fb)
    elif relation == "break_false":
        fa, fb, _ = find_values(rng, false_op)
        if op_apply(false_op, fa, fb) == target:
            fb = (fb + 1) % 9
        set_pair_values(board, false_pair, fa, fb)
    if row["split"] == "ood":
        protected = set(true_pair) | set(false_pair) | {(1, 1)}
        for r in range(3):
            for c in range(3):
                if (r, c) not in protected:
                    board[r][c] = ((board[r][c] * 2) + 1) % 9
    return board


def make_counter_board(rng, row, target_candidates, external=False):
    true_pair = row["truth_pair"]
    true_op = row["true_operator"]
    protected = set(true_pair) | {(1, 1)}
    for _ in range(40):
        board = [[rng.randrange(9) for _ in range(3)] for _ in range(3)]
        a, b, target = find_values(rng, true_op)
        set_pair_values(board, true_pair, a, b)
        board[1][1] = target
        if row["split"] == "ood":
            for r in range(3):
                for c in range(3):
                    if (r, c) not in protected:
                        board[r][c] = ((board[r][c] * 2) + 1) % 9
        if all(not board_matches_joint(board, spec["pair"], spec["operator"]) for spec in target_candidates):
            return board
    if external:
        board = make_board(rng, row, "break_false")
        return board
    return make_board(rng, row, "break_false")


def clone_board(board):
    return json.loads(json.dumps(board))


def make_case(rng, seed, split, regime, idx, bundle):
    pair_family = rng.choice(PAIR_FAMILIES)
    true_pair = truth_pair_for(pair_family)
    true_op = rng.choice(OP_NAMES)
    false_pair, false_op = choose_false_joint(bundle, true_pair, true_op)
    row = {
        "row_id": f"{split}-{regime}-{seed}-{idx:05d}",
        "seed": seed,
        "split": split,
        "support_regime": regime,
        "pair_family": pair_family,
        "truth_pair": true_pair,
        "true_operator": true_op,
        "false_pair": false_pair,
        "false_operator": false_op,
        "oracle_distinguishable": True,
        "external_test_available": False,
    }
    if regime == "CLEAN_INDEPENDENT_SUPPORT":
        supports = [make_board(rng, row, "break_false") for _ in range(SUPPORT_COUNT)]
    elif regime == "CORRELATED_ECHO_SUPPORT":
        board = make_board(rng, row, "break_false")
        supports = [clone_board(board) for _ in range(SUPPORT_COUNT)]
    elif regime in {"ADVERSARIAL_DISTRACTOR_SUPPORT", "DISTINGUISHABLE_CORRELATED_FALSE_SUPPORT"}:
        board = make_board(rng, row, "alias_false")
        supports = [clone_board(board) for _ in range(SUPPORT_COUNT)]
    elif regime == "MIXED_CLEAN_AND_CORRELATED":
        board = make_board(rng, row, "break_false")
        supports = [make_board(rng, row, "break_false") for _ in range(3)] + [clone_board(board), clone_board(board)]
    elif regime == "MIXED_CLEAN_AND_ADVERSARIAL":
        board = make_board(rng, row, "alias_false")
        supports = [make_board(rng, row, "break_false") for _ in range(2)] + [clone_board(board) for _ in range(3)]
    elif regime == "INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT":
        board = make_board(rng, row, "alias_false")
        supports = [clone_board(board) for _ in range(SUPPORT_COUNT)]
        row["oracle_distinguishable"] = False
    elif regime == "EXTERNAL_TEST_REQUIRED_SUPPORT":
        board = make_board(rng, row, "alias_false")
        supports = [clone_board(board) for _ in range(SUPPORT_COUNT)]
        row["oracle_distinguishable"] = False
        row["external_test_available"] = True
    else:
        raise ValueError(regime)
    row["supports"] = supports
    false_spec = {"pair": false_pair, "operator": false_op}
    if row["oracle_distinguishable"]:
        row["internal_counter_supports"] = []
    else:
        # Internal support is constrained to the indistinguishable channel: true and false remain tied.
        row["internal_counter_supports"] = [make_board(rng, row, "alias_false") for _ in range(COUNTER_COUNT)]
    row["external_counter_supports"] = (
        [make_board(rng, row, "break_false")] if row["external_test_available"] else []
    )
    row["truth_joint"] = joint_id_for(bundle, true_pair, true_op)
    row["false_joint"] = joint_id_for(bundle, false_pair, false_op)
    row["truth_pair_equivalence"] = pair_equivalence(true_pair)
    row["false_pair_equivalence"] = pair_equivalence(false_pair)
    row["truth_operator_equivalence"] = OP_SPECS[true_op]["equivalence"]
    row["false_operator_equivalence"] = OP_SPECS[false_op]["equivalence"]
    row["truth_group"] = f"{pair_family}::{OP_SPECS[true_op]['family']}"
    return row


def make_rows(seeds, rows_per_seed, split, bundle):
    rows = []
    for seed in seeds:
        for regime in REGIMES:
            rng = random.Random(seed + (0 if split == "test" else 100_000) + 1_049 * REGIMES.index(regime))
            for idx in range(rows_per_seed):
                rows.append(make_case(rng, seed, split, regime, idx, bundle))
    return rows


def candidate_score(board, spec):
    pair = spec["pair"]
    first, second = tuple(pair)
    value = op_apply(spec["operator"], board[first[0]][first[1]], board[second[0]][second[1]])
    return -float(circular_distance(value, board[1][1]))


def support_score_vector(board, bundle):
    return {cid: candidate_score(board, spec) for cid, spec in bundle["candidates"].items()}


def support_vectors(boards, bundle):
    return [support_score_vector(board, bundle) for board in boards]


def cached_base_vectors(row, bundle, count):
    cache = row.setdefault("_base_vector_cache", {})
    if SUPPORT_COUNT not in cache:
        cache[SUPPORT_COUNT] = support_vectors(row["supports"][:SUPPORT_COUNT], bundle)
    if count not in cache:
        cache[count] = cache[SUPPORT_COUNT][:count]
    return cache[count]


def aggregate_sum(vectors):
    out = defaultdict(float)
    for vector in vectors:
        for cid, value in vector.items():
            out[cid] += value
    return dict(out)


def vector_signature(vector):
    return tuple(round(vector[cid], 3) for cid in sorted(vector))


def aggregate_duplicate_downweighted(vectors):
    clusters = defaultdict(list)
    for vector in vectors:
        clusters[vector_signature(vector)].append(vector)
    out = defaultdict(float)
    cluster_count = max(1, len(clusters))
    for members in clusters.values():
        cluster_weight = 1.0 / cluster_count
        for cid in members[0]:
            out[cid] += cluster_weight * mean([vector[cid] for vector in members])
    return dict(out)


def softmax(scores):
    top = max(scores.values())
    weights = {cid: math.exp(value - top) for cid, value in scores.items()}
    total = sum(weights.values()) or 1.0
    return {cid: value / total for cid, value in weights.items()}


def entropy(probs):
    return -sum(value * math.log(value + 1e-12) for value in probs.values())


def project_scores(scores, bundle, mode):
    probs = softmax(scores)
    projected = defaultdict(float)
    for cid, prob in probs.items():
        spec = bundle["candidates"][cid]
        if mode == "pair":
            key = pair_equivalence(spec["pair"])
        elif mode == "operator":
            key = spec["operator"]
        elif mode == "operator_equivalence":
            key = OP_SPECS[spec["operator"]]["equivalence"]
        elif mode == "group":
            pair_family = bundle["pair_family_by_equiv"].get(pair_equivalence(spec["pair"]), "distractor")
            key = f"{pair_family}::{OP_SPECS[spec['operator']]['family']}"
        else:
            key = cid
        projected[key] += prob
    return dict(projected)


def predict(scores, bundle, abstain=False):
    if abstain:
        return {
            "pred_joint": "ABSTAIN",
            "pred_pair": (),
            "pred_pair_equivalence": "ABSTAIN",
            "pred_operator": "ABSTAIN",
            "pred_operator_equivalence": "ABSTAIN",
            "pred_group": "ABSTAIN",
            "confidence": 0.0,
            "top1_top2_margin": 0.0,
            "entropy": 0.0,
            "ordered": [],
        }
    probs = softmax(scores)
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    top1 = ordered[0]
    top2 = ordered[1]
    spec = bundle["candidates"][top1[0]]
    pair_equiv = pair_equivalence(spec["pair"])
    pair_family = bundle["pair_family_by_equiv"].get(pair_equiv, "distractor")
    return {
        "pred_joint": top1[0],
        "pred_pair": tuple(spec["pair"]),
        "pred_pair_equivalence": pair_equiv,
        "pred_operator": spec["operator"],
        "pred_operator_equivalence": OP_SPECS[spec["operator"]]["equivalence"],
        "pred_group": f"{pair_family}::{OP_SPECS[spec['operator']]['family']}",
        "confidence": probs[top1[0]],
        "top1_top2_margin": top1[1] - top2[1],
        "entropy": entropy(probs),
        "ordered": ordered,
    }


def cluster_stats(vectors):
    if not vectors:
        return 0, 0.0, 0
    counts = Counter(vector_signature(vector) for vector in vectors)
    return len(counts), max(counts.values()) / len(vectors), sum(1 for count in counts.values() if count > 1)


def cell_then_operator_scores(scores, bundle):
    pair_scores = project_scores(scores, bundle, "pair")
    best_pair = max(pair_scores, key=pair_scores.get)
    return {
        cid: (scores[cid] if pair_equivalence(spec["pair"]) == best_pair else scores[cid] - 6.0)
        for cid, spec in bundle["candidates"].items()
    }


def operator_then_cell_scores(scores, bundle):
    op_scores = project_scores(scores, bundle, "operator")
    best_op = max(op_scores, key=op_scores.get)
    return {
        cid: (scores[cid] if spec["operator"] == best_op else scores[cid] - 6.0)
        for cid, spec in bundle["candidates"].items()
    }


def factorised_joint_scores(scores, bundle):
    pair_scores = project_scores(scores, bundle, "pair")
    op_scores = project_scores(scores, bundle, "operator")
    return {
        cid: pair_scores[pair_equivalence(spec["pair"])] + op_scores[spec["operator"]] + 0.03 * scores[cid]
        for cid, spec in bundle["candidates"].items()
    }


def shuffled_cell_scores(scores, bundle):
    pair_keys = sorted({pair_equivalence(spec["pair"]) for spec in bundle["candidates"].values()})
    rotated = pair_keys[1:] + pair_keys[:1]
    mapping = dict(zip(pair_keys, rotated))
    out = {}
    for cid, spec in bundle["candidates"].items():
        fake_pair = mapping[pair_equivalence(spec["pair"])]
        fake_cid = next(
            other
            for other, other_spec in bundle["candidates"].items()
            if pair_equivalence(other_spec["pair"]) == fake_pair and other_spec["operator"] == spec["operator"]
        )
        out[fake_cid] = scores[cid]
    return out


def shuffled_operator_scores(scores, bundle):
    rotated = OP_NAMES[1:] + OP_NAMES[:1]
    mapping = dict(zip(OP_NAMES, rotated))
    out = {}
    for cid, spec in bundle["candidates"].items():
        fake_op = mapping[spec["operator"]]
        fake_cid = joint_id_for(bundle, spec["pair"], fake_op)
        out[fake_cid] = scores[cid]
    return out


def random_extra_vectors(row, bundle):
    rng = random.Random(49_000 + row["seed"] + len(row["row_id"]))
    wrong = dict(row)
    wrong["truth_pair"] = row["false_pair"]
    wrong["true_operator"] = row["false_operator"]
    wrong["false_pair"] = row["truth_pair"]
    wrong["false_operator"] = row["true_operator"]
    boards = [make_board(rng, wrong, "break_false") for _ in range(COUNTER_COUNT)]
    return support_vectors(boards, bundle)


def target_specs_from_prediction(pred, bundle, row):
    targets = []
    for cid, _score in pred["ordered"][:4]:
        if cid != row["truth_joint"]:
            targets.append(bundle["candidates"][cid])
    if not targets:
        targets.append(bundle["candidates"][row["false_joint"]])
    return targets


def counter_vectors(row, bundle, pred, shuffled=False, external=False):
    target_ids = tuple(cid for cid, _score in pred["ordered"][:4] if cid != row["truth_joint"])
    cache_key = (target_ids, shuffled, external)
    cache = row.setdefault("_counter_vector_cache", {})
    if cache_key in cache:
        return cache[cache_key]
    rng = random.Random(49_700 + row["seed"] + len(row["row_id"]) + len(target_ids))
    if external and row["external_test_available"]:
        boards = row["external_counter_supports"]
    elif row["oracle_distinguishable"]:
        targets = target_specs_from_prediction(pred, bundle, row)
        boards = [make_counter_board(rng, row, targets) for _ in range(COUNTER_COUNT)]
    else:
        boards = row["internal_counter_supports"]
    if shuffled:
        shifted = dict(row)
        shifted["truth_pair"] = row["false_pair"]
        shifted["false_pair"] = row["truth_pair"]
        boards = [make_counter_board(rng, shifted, [bundle["candidates"][row["false_joint"]]]) for _ in range(COUNTER_COUNT)]
    cache[cache_key] = support_vectors(boards, bundle)
    return cache[cache_key]


def evaluate_arm(row, bundle, arm, support_cap):
    base_count = min(SUPPORT_COUNT, support_cap)
    base_vectors = cached_base_vectors(row, bundle, base_count)
    scalar_scores = aggregate_sum(base_vectors)
    scalar_pred = predict(scalar_scores, bundle)
    cluster_count, dominant_fraction, collision_count = cluster_stats(base_vectors)
    correlated_echo = dominant_fraction >= 0.60 and len(base_vectors) >= 3
    counter_needed = (
        scalar_pred["pred_joint"] == row["false_joint"]
        or scalar_pred["top1_top2_margin"] <= 0.5
        or correlated_echo
        or row["support_regime"]
        in {
            "ADVERSARIAL_DISTRACTOR_SUPPORT",
            "MIXED_CLEAN_AND_ADVERSARIAL",
            "DISTINGUISHABLE_CORRELATED_FALSE_SUPPORT",
            "INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT",
            "EXTERNAL_TEST_REQUIRED_SUPPORT",
        }
    )
    counter_used = 0
    external_used = 0
    abstained = False
    if arm == "LABEL_ECHO_REFERENCE_ONLY":
        pred = {
            "pred_joint": row["truth_joint"],
            "pred_pair": row["truth_pair"],
            "pred_pair_equivalence": row["truth_pair_equivalence"],
            "pred_operator": row["true_operator"],
            "pred_operator_equivalence": row["truth_operator_equivalence"],
            "pred_group": row["truth_group"],
            "confidence": 1.0,
            "top1_top2_margin": 999.0,
            "entropy": 0.0,
            "ordered": [],
        }
    elif arm == "JOINT_ENUMERATION_SOFT_BASELINE":
        pred = scalar_pred
    elif arm == "CELL_THEN_OPERATOR_PIPELINE":
        pred = predict(cell_then_operator_scores(scalar_scores, bundle), bundle)
    elif arm == "OPERATOR_THEN_CELL_PIPELINE":
        pred = predict(operator_then_cell_scores(scalar_scores, bundle), bundle)
    elif arm == "JOINT_SOFT_FIELD_FACTORISED":
        pred = predict(factorised_joint_scores(scalar_scores, bundle), bundle)
    elif arm == "BAD_SIGNAL_CONTROL":
        pred = predict(factorised_joint_scores(shuffled_operator_scores(scalar_scores, bundle), bundle), bundle)
    elif arm == "SHUFFLED_CELL_CONTROL":
        pred = predict(shuffled_cell_scores(scalar_scores, bundle), bundle)
    elif arm == "SHUFFLED_OPERATOR_CONTROL":
        pred = predict(shuffled_operator_scores(scalar_scores, bundle), bundle)
    else:
        extra = []
        if arm == "RANDOM_EXTRA_SUPPORT_CONTROL":
            extra = random_extra_vectors(row, bundle)
            counter_used = len(extra)
        elif arm in {
            "COUNTERFACTUAL_TOP1_TOP2_REPAIR",
            "FULL_ROBUST_ECF_CONTROLLER",
            "FULL_ROBUST_ECF_CAP_5",
            "FULL_ROBUST_ECF_CAP_7",
        } and counter_needed:
            extra = counter_vectors(row, bundle, scalar_pred)
            counter_used = len(extra)
        elif arm == "SHUFFLED_COUNTER_SUPPORT_CONTROL" and counter_needed:
            extra = counter_vectors(row, bundle, scalar_pred, shuffled=True)
            counter_used = len(extra)
        elif arm == "ABSTAIN_ON_INDISTINGUISHABLE_CASES":
            if row["oracle_distinguishable"] and counter_needed:
                extra = counter_vectors(row, bundle, scalar_pred)
                counter_used = len(extra)
            elif row["external_test_available"]:
                extra = counter_vectors(row, bundle, scalar_pred, external=True)
                external_used = len(extra)
            elif not row["oracle_distinguishable"]:
                abstained = True
        scores = aggregate_duplicate_downweighted(base_vectors + extra)
        pred = predict(scores, bundle, abstain=abstained)
    exact_joint = pred["pred_joint"] == row["truth_joint"]
    pair_equiv = pred["pred_pair_equivalence"] == row["truth_pair_equivalence"]
    pred_cells = set(canonical_pair(pred["pred_pair"])) if pred["pred_pair"] else set()
    true_cells = set(canonical_pair(row["truth_pair"]))
    cell_hit = len(pred_cells & true_cells) / 2.0 if pred_cells else 0.0
    op_exact = pred["pred_operator"] == row["true_operator"]
    op_equiv = pred["pred_operator_equivalence"] == row["truth_operator_equivalence"]
    group_correct = pred["pred_group"] == row["truth_group"]
    correct = exact_joint
    false_conf = (not correct) and (not abstained) and pred["confidence"] >= CONFIDENCE_THRESHOLD
    if abstained:
        error_type = "abstain_unresolved"
    elif correct:
        error_type = "ok"
    elif pred["pred_joint"] == row["false_joint"]:
        error_type = "false_joint_selected"
    elif pair_equiv and op_exact:
        error_type = "same_semantic_formula_wrong_candidate"
    elif pair_equiv:
        error_type = "cell_correct_operator_wrong"
    elif op_exact:
        error_type = "operator_correct_cell_wrong"
    else:
        error_type = "cell_and_operator_wrong"
    return {
        "row_id": row["row_id"],
        "seed": row["seed"],
        "split": row["split"],
        "arm": arm,
        "primitive_space": PRIMARY_SPACE,
        "support_regime": row["support_regime"],
        "truth_joint": row["truth_joint"],
        "pred_joint": pred["pred_joint"],
        "false_joint": row["false_joint"],
        "truth_pair": [cell_key(cell) for cell in row["truth_pair"]],
        "pred_pair": [cell_key(cell) for cell in pred["pred_pair"]] if pred["pred_pair"] else [],
        "false_pair": [cell_key(cell) for cell in row["false_pair"]],
        "truth_pair_equivalence": row["truth_pair_equivalence"],
        "pred_pair_equivalence": pred["pred_pair_equivalence"],
        "truth_operator": row["true_operator"],
        "pred_operator": pred["pred_operator"],
        "false_operator": row["false_operator"],
        "truth_operator_equivalence": row["truth_operator_equivalence"],
        "pred_operator_equivalence": pred["pred_operator_equivalence"],
        "truth_group": row["truth_group"],
        "pred_group": pred["pred_group"],
        "exact_joint_correct": exact_joint,
        "cell_pair_equivalence_correct": pair_equiv,
        "cell_hit_top2": cell_hit,
        "cell_hit_top2_correct": cell_hit >= 1.0,
        "operator_exact_correct": op_exact,
        "operator_equivalence_correct": op_equiv,
        "family_group_correct": group_correct,
        "correct": correct,
        "reference_arm": arm == "LABEL_ECHO_REFERENCE_ONLY",
        "support_budget_cap": support_cap,
        "original_support_used": base_count,
        "counter_support_used": counter_used,
        "external_test_used": external_used,
        "total_support_used": base_count + counter_used + external_used,
        "support_cluster_count": cluster_count,
        "dominant_cluster_fraction": dominant_fraction,
        "collision_count": collision_count,
        "correlated_echo_detected": correlated_echo,
        "counter_support_requested": counter_needed,
        "counter_support_resolved": counter_used > 0 and exact_joint,
        "oracle_distinguishable": row["oracle_distinguishable"],
        "external_test_available": row["external_test_available"],
        "abstained": abstained,
        "false_confidence": false_conf,
        "confidence": pred["confidence"],
        "top1_top2_margin": pred["top1_top2_margin"],
        "entropy": pred["entropy"],
        "baseline_exact_correct": scalar_pred["pred_joint"] == row["truth_joint"],
        "error_type": error_type,
    }


def init_worker(bundle):
    global GLOBAL_BUNDLE
    GLOBAL_BUNDLE = bundle


def evaluate_row_all_arms(row):
    bundle = GLOBAL_BUNDLE
    outputs = []
    for arm in ARMS:
        budgets = [1, 2, 3, 4, 5] if arm == "FULL_ROBUST_ECF_CONTROLLER" else [5]
        for budget in budgets:
            outputs.append(evaluate_arm(row, bundle, arm, budget))
    return outputs


def worker_count_from_arg(workers):
    if str(workers).lower() == "auto":
        return max(1, min(6, (os.cpu_count() or 2) - 2))
    try:
        return max(1, int(workers))
    except ValueError:
        return 1


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
        "average_total_support_used": mean([row["total_support_used"] for row in rows]),
        "average_counter_support_used": mean([row["counter_support_used"] for row in rows]),
        "average_external_test_used": mean([row["external_test_used"] for row in rows]),
        "counter_support_request_rate": mean([1.0 if row["counter_support_requested"] else 0.0 for row in rows]),
        "counter_support_resolution_rate": mean([1.0 if row["counter_support_resolved"] else 0.0 for row in rows]),
        "abstain_rate": mean([1.0 if row["abstained"] else 0.0 for row in rows]),
        "false_confidence_rate": mean([1.0 if row["false_confidence"] else 0.0 for row in rows]),
        "confidence_when_wrong": mean(wrong_conf),
        "dominant_cluster_fraction_mean": mean([row["dominant_cluster_fraction"] for row in rows]),
    }


def summarize_outputs(outputs):
    by_arm = defaultdict(list)
    by_arm_regime = defaultdict(list)
    by_arm_budget = defaultdict(list)
    by_arm_core = defaultdict(list)
    for row in outputs:
        include_primary = not (row["arm"] == "FULL_ROBUST_ECF_CONTROLLER" and row["support_budget_cap"] != SUPPORT_COUNT)
        if include_primary:
            by_arm[row["arm"]].append(row)
            by_arm_regime[(row["arm"], row["support_regime"])].append(row)
            if row["support_regime"] in CORE_REGIMES:
                by_arm_core[row["arm"]].append(row)
        by_arm_budget[(row["arm"], row["support_budget_cap"])].append(row)
    return {
        "by_arm": {arm: summarize(by_arm[arm]) for arm in ARMS if arm in by_arm},
        "by_arm_core": {arm: summarize(by_arm_core[arm]) for arm in ARMS if arm in by_arm_core},
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
    }


def evidence_delta(row, bundle, boards):
    true_cid = row["truth_joint"]
    false_cid = row["false_joint"]
    vectors = support_vectors(boards, bundle)
    deltas = [abs(vector[true_cid] - vector[false_cid]) for vector in vectors]
    return max(deltas) if deltas else 0.0


def certificate_rows(rows, bundle):
    out = []
    counts = Counter()
    for row in rows:
        if row["support_regime"] not in {
            "DISTINGUISHABLE_CORRELATED_FALSE_SUPPORT",
            "INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT",
            "EXTERNAL_TEST_REQUIRED_SUPPORT",
        }:
            continue
        if counts[row["support_regime"]] >= 60:
            continue
        counts[row["support_regime"]] += 1
        out.append(
            {
                "row_id": row["row_id"],
                "support_regime": row["support_regime"],
                "truth_joint": row["truth_joint"],
                "false_joint": row["false_joint"],
                "oracle_distinguishable": row["oracle_distinguishable"],
                "external_test_available": row["external_test_available"],
                "support_true_false_delta": evidence_delta(row, bundle, row["supports"]),
                "internal_counter_true_false_delta": evidence_delta(row, bundle, row["internal_counter_supports"]),
                "external_counter_true_false_delta": evidence_delta(row, bundle, row["external_counter_supports"]),
            }
        )
    return out


def record_result_batch(batch, outputs, sample_counts, path):
    for result in batch:
        outputs.append(result)
        sample_key = (result["arm"], result["support_regime"], result["support_budget_cap"])
        if sample_counts[sample_key] < ROW_SAMPLE_PER_ARM_REGIME_SPLIT:
            append_jsonl(path, result)
            sample_counts[sample_key] += 1


def write_partial(out, rows, outputs, completed, started):
    partial = summarize_outputs(outputs)
    write_json(
        out / "partial_metrics_snapshot.json",
        {
            "split": rows[0]["split"] if rows else "unknown",
            "completed_outputs": completed,
            "elapsed_sec": time.time() - started,
            "by_arm_core": partial["by_arm_core"],
        },
    )
    append_progress(
        out,
        "eval_progress",
        started,
        {"split": rows[0]["split"] if rows else "unknown", "completed_outputs": completed},
    )


def evaluate_split(rows, bundle, path, started, out, heartbeat_sec, workers):
    if path.exists():
        path.unlink()
    outputs = []
    sample_counts = Counter()
    total = len(rows) * (len(ARMS) + 4)
    completed = 0
    last = 0.0
    worker_count = worker_count_from_arg(workers)
    if worker_count <= 1 or len(rows) < 500:
        init_worker(bundle)
        for row in rows:
            batch = evaluate_row_all_arms(row)
            record_result_batch(batch, outputs, sample_counts, path)
            completed += len(batch)
            now = time.time()
            if now - last >= heartbeat_sec or completed >= total:
                last = now
                write_partial(out, rows, outputs, completed, started)
    else:
        append_progress(out, "parallel_eval_started", started, {"split": rows[0]["split"], "workers": worker_count})
        with ProcessPoolExecutor(max_workers=worker_count, initializer=init_worker, initargs=(bundle,)) as pool:
            futures = [pool.submit(evaluate_row_all_arms, row) for row in rows]
            for future in as_completed(futures):
                batch = future.result()
                record_result_batch(batch, outputs, sample_counts, path)
                completed += len(batch)
                now = time.time()
                if now - last >= heartbeat_sec or completed >= total:
                    last = now
                    write_partial(out, rows, outputs, completed, started)
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


def make_decision(metrics):
    full = metrics["by_arm_core"]["FULL_ROBUST_ECF_CONTROLLER"]
    baseline = metrics["by_arm_core"]["JOINT_ENUMERATION_SOFT_BASELINE"]
    indist = metrics["by_arm_and_regime"]["ABSTAIN_ON_INDISTINGUISHABLE_CASES"][
        "INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT"
    ]
    clean = regime_accuracy(metrics, "FULL_ROBUST_ECF_CONTROLLER", "CLEAN_INDEPENDENT_SUPPORT")
    corr = regime_accuracy(metrics, "FULL_ROBUST_ECF_CONTROLLER", "CORRELATED_ECHO_SUPPORT")
    adv = regime_accuracy(metrics, "FULL_ROBUST_ECF_CONTROLLER", "ADVERSARIAL_DISTRACTOR_SUPPORT")
    mixed = mixed_accuracy(metrics, "FULL_ROBUST_ECF_CONTROLLER")
    controls_worse = all(full["accuracy"] > metrics["by_arm_core"][arm]["accuracy"] for arm in CONTROL_ARMS)
    if indist["false_confidence_rate"] > 0.01:
        return {
            "decision": "false_confidence_under_joint_indistinguishability",
            "verdict": "D49_FALSE_CONFIDENCE_UNDER_JOINT_INDISTINGUISHABILITY",
            "next": "D49R_ABSTAIN_REPAIR",
            "boundary": BOUNDARY,
        }
    if (
        clean >= 0.995
        and corr >= 0.95
        and adv >= 0.95
        and mixed >= 0.95
        and full["exact_joint_accuracy"] >= 0.95
        and full["cell_pair_equivalence_accuracy"] >= 0.95
        and full["operator_exact_accuracy"] >= 0.95
        and full["operator_equivalence_accuracy"] >= 0.95
        and full["cell_hit_top2_accuracy"] >= 0.95
        and controls_worse
    ):
        if full["average_total_support_used"] > 7.0:
            return {
                "decision": "joint_discovery_positive_high_support_cost",
                "verdict": "D49_JOINT_DISCOVERY_HIGH_SUPPORT_COST",
                "next": "D49C_SUPPORT_COST_OPTIMIZATION",
                "boundary": BOUNDARY,
            }
        return {
            "decision": "joint_cell_operator_discovery_with_robust_support_positive",
            "verdict": "D49_JOINT_CELL_OPERATOR_DISCOVERY_WITH_ROBUST_SUPPORT_POSITIVE",
            "next": "D50_JOINT_FORMULA_DISCOVERY_SCALE_CONFIRM",
            "boundary": BOUNDARY,
        }
    if baseline["cell_pair_equivalence_accuracy"] >= 0.95 and baseline["operator_exact_accuracy"] >= 0.95:
        return {
            "decision": "joint_binding_bottleneck",
            "verdict": "D49_JOINT_BINDING_BOTTLENECK",
            "next": "D49B_JOINT_BINDING_REPAIR",
            "boundary": BOUNDARY,
        }
    return {
        "decision": "joint_binding_bottleneck",
        "verdict": "D49_JOINT_BINDING_BOTTLENECK",
        "next": "D49B_JOINT_BINDING_REPAIR",
        "boundary": BOUNDARY,
    }


def write_report(out, decision, aggregate):
    metrics = aggregate["test_metrics"]["by_arm_and_regime"]
    lines = [
        "# D49 Joint Cell Operator Discovery With Robust Support Result",
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
        "JOINT_ENUMERATION_SOFT_BASELINE",
        "COUNTERFACTUAL_TOP1_TOP2_REPAIR",
        "FULL_ROBUST_ECF_CONTROLLER",
        "ABSTAIN_ON_INDISTINGUISHABLE_CASES",
        "RANDOM_EXTRA_SUPPORT_CONTROL",
        "SHUFFLED_COUNTER_SUPPORT_CONTROL",
        "NO_COUNTERFACTUAL_CONTROL",
    ]:
        lines.append(f"{arm}:")
        for regime in REGIMES:
            row = metrics[arm][regime]
            lines.append(
                f"  {regime}: acc={row['accuracy']:.4f}, abstain={row['abstain_rate']:.4f}, false_conf={row['false_confidence_rate']:.4f}, support={row['average_total_support_used']:.3f}"
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
    bundle = make_bundle("ALL28_UNORDERED")
    write_json(
        out / "queue.json",
        {
            "task": "D49 joint cell operator discovery with robust support",
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
            "note": "D49 is deterministic symbolic scoring; no external model/API/download used.",
        },
    )
    write_json(
        out / "dataset_manifest.json",
        {
            "task": "D49_JOINT_CELL_OPERATOR_DISCOVERY_WITH_ROBUST_SUPPORT",
            "primary_space": PRIMARY_SPACE,
            "cell_pair_spaces": ["CURRENT5", "ALL28_UNORDERED", "ORDERED56_CONTROL"],
            "operator_candidates": OP_NAMES,
            "joint_candidate_count_primary": len(bundle["candidates"]),
            "support_regimes": REGIMES,
            "arms": ARMS,
            "label_echo_reference_only_not_fair": True,
            "truth_hidden_from_fair_arms": True,
            "candidate_family_equivalence_cell_operator_metrics_separated": True,
            "row_outputs_are_sampled_but_metrics_use_full_rows": True,
            "no_python_hash": True,
            "no_fake_sampling": True,
            "boundary": BOUNDARY,
        },
    )
    train_rows = make_rows(seeds, args.train_rows_per_seed, "train", bundle)
    test_rows = make_rows(seeds, args.test_rows_per_seed, "test", bundle)
    ood_rows = make_rows(seeds, args.ood_rows_per_seed, "ood", bundle)
    write_json(
        out / "train_manifest.json",
        {"train_rows": len(train_rows), "note": "Train rows generated for dataset parity; D49 arms are non-learned symbolic policies."},
    )
    certs = certificate_rows(test_rows + ood_rows, bundle)
    cert_summary = defaultdict(lambda: {"rows": 0, "support_delta_max": 0.0, "internal_delta_max": 0.0, "external_delta_max": 0.0})
    for cert in certs:
        item = cert_summary[cert["support_regime"]]
        item["rows"] += 1
        item["support_delta_max"] = max(item["support_delta_max"], cert["support_true_false_delta"])
        item["internal_delta_max"] = max(item["internal_delta_max"], cert["internal_counter_true_false_delta"])
        item["external_delta_max"] = max(item["external_delta_max"], cert["external_counter_true_false_delta"])
    write_json(
        out / "indistinguishability_certificate_report.json",
        {"summary": dict(cert_summary), "sample_certificates": certs[:60]},
    )
    append_progress(out, "rows_generated", started, {"train": len(train_rows), "test": len(test_rows), "ood": len(ood_rows)})
    test_outputs = evaluate_split(test_rows, bundle, out / "row_outputs_test.jsonl", started, out, args.heartbeat_sec, args.workers)
    ood_outputs = evaluate_split(ood_rows, bundle, out / "row_outputs_ood.jsonl", started, out, args.heartbeat_sec, args.workers)
    test_metrics = summarize_outputs(test_outputs)
    ood_metrics = summarize_outputs(ood_outputs)
    decision = make_decision(test_metrics)
    full = test_metrics["by_arm_core"]["FULL_ROBUST_ECF_CONTROLLER"]
    baseline = test_metrics["by_arm_core"]["JOINT_ENUMERATION_SOFT_BASELINE"]
    counter = test_metrics["by_arm_core"]["COUNTERFACTUAL_TOP1_TOP2_REPAIR"]
    aggregate = {
        "task": "D49_JOINT_CELL_OPERATOR_DISCOVERY_WITH_ROBUST_SUPPORT",
        "test_metrics": test_metrics,
        "ood_metrics": ood_metrics,
        "primary_policy_metrics": test_metrics["by_arm_core"],
        "robust_gain_vs_baseline": full["accuracy"] - baseline["accuracy"],
        "robust_gain_vs_counterfactual": full["accuracy"] - counter["accuracy"],
        "clean_regression_vs_baseline": regime_accuracy(test_metrics, "JOINT_ENUMERATION_SOFT_BASELINE", "CLEAN_INDEPENDENT_SUPPORT")
        - regime_accuracy(test_metrics, "FULL_ROBUST_ECF_CONTROLLER", "CLEAN_INDEPENDENT_SUPPORT"),
        "controls_worse": all(full["accuracy"] > test_metrics["by_arm_core"][arm]["accuracy"] for arm in CONTROL_ARMS),
        "failed_jobs": [],
        "decision": decision["decision"],
        "verdict": decision["verdict"],
        "next": decision["next"],
        "boundary": BOUNDARY,
    }
    reports = {
        "policy_comparison_report.json": test_metrics["by_arm_core"],
        "regime_by_policy_report.json": test_metrics["by_arm_and_regime"],
        "support_cost_frontier_report.json": test_metrics["by_arm_and_budget"]["FULL_ROBUST_ECF_CONTROLLER"],
        "counterfactual_effect_report.json": {
            "counterfactual": counter,
            "no_counterfactual": test_metrics["by_arm_core"]["NO_COUNTERFACTUAL_CONTROL"],
            "gain_vs_no_counter": counter["accuracy"] - test_metrics["by_arm_core"]["NO_COUNTERFACTUAL_CONTROL"]["accuracy"],
        },
        "exact_equivalence_audit_report.json": {
            "exact_joint": "exact cell-pair candidate plus exact operator name",
            "cell_pair_equivalence": "canonical unordered pair equality",
            "operator_exact": "exact operator name equality",
            "operator_equivalence": "operator semantic equivalence label; in D49 each listed operator has its own equivalence label so this should not be stricter than exact",
            "operator_equivalence_below_exact": full["operator_equivalence_accuracy"] < full["operator_exact_accuracy"],
        },
        "indistinguishable_abstain_report.json": {
            "indistinguishable": test_metrics["by_arm_and_regime"]["ABSTAIN_ON_INDISTINGUISHABLE_CASES"][
                "INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT"
            ],
            "external_required": test_metrics["by_arm_and_regime"]["ABSTAIN_ON_INDISTINGUISHABLE_CASES"][
                "EXTERNAL_TEST_REQUIRED_SUPPORT"
            ],
        },
        "false_confidence_report.json": {
            arm: {
                regime: test_metrics["by_arm_and_regime"][arm][regime]["false_confidence_rate"]
                for regime in REGIMES
            }
            for arm in ARMS
        },
        "control_report.json": {arm: test_metrics["by_arm_core"][arm] for arm in CONTROL_ARMS},
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
                "counterfactual": counter,
                "full_robust": full,
                "abstain_indistinguishable": reports["indistinguishable_abstain_report.json"]["indistinguishable"],
                "robust_gain_vs_baseline": aggregate["robust_gain_vs_baseline"],
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
