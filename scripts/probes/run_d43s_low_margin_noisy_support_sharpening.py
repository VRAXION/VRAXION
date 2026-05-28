#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
import statistics
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

FAMILIES = ["row", "col", "pair", "mirror", "diag"]
ARMS = [
    "D43_BASELINE_REPLAY",
    "MARGIN_PRESERVING_OBJECTIVE",
    "TEMPERATURE_SHARPENED_EVIDENCE",
    "FAMILY_BALANCED_EDGE_OVERSAMPLING",
    "COMBINED_SHARPENED_MODEL",
    "HARD_VOTE_ORACLE_UPPER_BOUND",
    "SHUFFLED_CENTER_CONTROL",
    "SHUFFLED_FORMULA_CANDIDATE_CONTROL",
    "NO_CENTER_CONTROL",
    "SAME_QUERY_DIFFERENT_RAW_SUPPORT_COUNTERFACTUAL",
    "WRONG_SUPPORT_CONTROL",
]
REPORT_FILES = {
    "D43_BASELINE_REPLAY": "baseline_replay_report.json",
    "MARGIN_PRESERVING_OBJECTIVE": "margin_preserving_objective_report.json",
    "TEMPERATURE_SHARPENED_EVIDENCE": "temperature_sharpened_evidence_report.json",
    "FAMILY_BALANCED_EDGE_OVERSAMPLING": "family_balanced_edge_oversampling_report.json",
    "COMBINED_SHARPENED_MODEL": "combined_sharpened_model_report.json",
    "HARD_VOTE_ORACLE_UPPER_BOUND": "hard_vote_oracle_upper_bound_report.json",
    "SHUFFLED_CENTER_CONTROL": "shuffled_center_control_report.json",
    "SHUFFLED_FORMULA_CANDIDATE_CONTROL": "shuffled_formula_candidate_control_report.json",
    "NO_CENTER_CONTROL": "no_center_control_report.json",
    "SAME_QUERY_DIFFERENT_RAW_SUPPORT_COUNTERFACTUAL": "same_query_different_raw_support_counterfactual_report.json",
    "WRONG_SUPPORT_CONTROL": "wrong_support_control_report.json",
}
FORMULA_DESC = {
    "row": "(b[1][0] + b[1][2]) % 9",
    "col": "(b[0][1] + b[2][1]) % 9",
    "pair": "(b[0][0] + b[2][2]) % 9",
    "mirror": "(b[2][0] + b[0][2]) % 9",
    "diag": "(b[0][0] + b[1][2] + b[2][1]) % 9",
}
MUTATION_TYPES = [
    "equality_weight_delta",
    "equality_row_delta",
    "equality_column_delta",
    "channel_weight_delta",
    "channel_row_delta",
    "channel_column_delta",
    "rule_bias_delta",
    "equality_row_swap",
    "equality_column_swap",
    "channel_row_swap",
    "channel_column_swap",
    "prune_small_weights",
]
SUPPORT_COUNT_KEYS = [3, 5]
MARGIN_STRATA = ["noisy_majority_margin_low", "noisy_majority_margin_high", "clean_unanimous", "pair_low_margin", "diag_low_margin", "row_low_margin", "support_count_3_low_margin", "support_count_5_low_margin"]
TEMPERATURES = [1.0, 1.5, 2.0, 4.0]
DEFAULT_OUT = Path("target/pilot_wave/d43s_low_margin_noisy_support_sharpening/smoke")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--seeds", default="8951,8952,8953,8954,8955")
    parser.add_argument("--support-counts", default="3,5")
    parser.add_argument("--train-rows-per-seed", type=int, default=1000)
    parser.add_argument("--test-rows-per-seed", type=int, default=1000)
    parser.add_argument("--ood-rows-per-seed", type=int, default=1000)
    parser.add_argument("--generations", type=int, default=600)
    parser.add_argument("--population", type=int, default=160)
    parser.add_argument("--workers", default="auto")
    parser.add_argument("--cpu-target", choices=["saturate", "balanced"], default="saturate")
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    return parser.parse_args()


def stable_mix(seed, *parts):
    value = (int(seed) + 0x9E3779B9) & 0xFFFFFFFF
    for part in parts:
        value = (value ^ ((int(part) + 0x85EBCA6B) & 0xFFFFFFFF)) & 0xFFFFFFFF
        value = (value * 0xC2B2AE35 + 0x27D4EB2F) & 0xFFFFFFFF
    return value


def family_target(family_idx, board):
    if family_idx == 0:
        return (board[1][0] + board[1][2]) % 9
    if family_idx == 1:
        return (board[0][1] + board[2][1]) % 9
    if family_idx == 2:
        return (board[0][0] + board[2][2]) % 9
    if family_idx == 3:
        return (board[2][0] + board[0][2]) % 9
    return (board[0][0] + board[1][2] + board[2][1]) % 9


def formula_values(board):
    return [family_target(idx, board) for idx in range(5)]


def draw_symbol(rng, split, idx, r, c):
    if split == "ood":
        multiplier = [2, 4, 5, 7, 8][(idx + r + 2 * c) % 5]
        return (multiplier * rng.randrange(9) + idx + 2 * r + 3 * c) % 9
    return rng.randrange(9)


def random_noncenter_board(rng, split, idx):
    board = []
    for r in range(3):
        row = []
        for c in range(3):
            row.append(draw_symbol(rng, split, idx, r, c))
        board.append(row)
    return board


def make_support_board(rng, split, idx, evidence_family):
    for attempt in range(1000):
        board = random_noncenter_board(rng, split, idx + 17 * attempt)
        values = formula_values(board)
        center = values[evidence_family]
        if all(values[other] != center for other in range(5) if other != evidence_family):
            board[1][1] = center
            return board
    raise RuntimeError("could not generate unambiguous support board")


def make_query_board(rng, split, idx):
    for attempt in range(1000):
        board = random_noncenter_board(rng, split, idx + 29 * attempt)
        board[1][1] = None
        values = formula_values(board)
        if len(set(values)) == len(values):
            return board, values
    raise RuntimeError("could not generate distinct query formula values")


def choose_support_plan(support_count, intended_family, idx):
    bucket = (idx // max(1, len(SUPPORT_COUNT_KEYS))) % 10
    distractor = low_margin_distractor(intended_family, idx)
    if support_count == 3:
        if bucket < 8:
            return [intended_family, intended_family, distractor], "noisy_majority", "margin_low"
        return [intended_family for _ in range(support_count)], "clean_unanimous", "margin_high"
    if bucket < 6:
        return [intended_family, intended_family, intended_family, distractor, distractor], "noisy_majority", "margin_low"
    if bucket < 8:
        return [intended_family, intended_family, intended_family, intended_family, distractor], "noisy_majority", "margin_high"
    return [intended_family for _ in range(support_count)], "clean_unanimous", "margin_high"


def low_margin_distractor(intended_family, idx):
    if FAMILIES[intended_family] == "pair":
        return 0 if idx % 2 == 0 else 1
    if FAMILIES[intended_family] == "diag":
        return 2 if idx % 2 == 0 else 3
    if FAMILIES[intended_family] == "row":
        return 3
    return (intended_family + 1 + (idx % 4)) % 5


def intended_family_for_idx(seed, split_id, idx):
    cycle = [0, 1, 2, 3, 4, 2, 4, 0, 2, 4]
    return cycle[(idx + seed + split_id) % len(cycle)]


def support_evidence(support_boards, support_count):
    counts = [0 for _ in range(5)]
    for board in support_boards:
        center = board[1][1]
        for family_idx in range(5):
            if center == family_target(family_idx, board):
                counts[family_idx] += 1
    ordered = sorted(counts, reverse=True)
    margin_count = ordered[0] - ordered[1]
    return [count / support_count for count in counts], counts, margin_count, margin_count / support_count


def make_support_package(rng, split, idx, intended_family, support_count, force_clean=False):
    if force_clean:
        plan = [intended_family for _ in range(support_count)]
        support_stratum = "clean_unanimous"
        margin_stratum = "margin_high"
    else:
        plan, support_stratum, margin_stratum = choose_support_plan(support_count, intended_family, idx)
    support_boards = [
        make_support_board(rng, split, idx * 101 + support_idx * 37, evidence_family)
        for support_idx, evidence_family in enumerate(plan)
    ]
    evidence_vector, counts, margin_count, margin_normalized = support_evidence(support_boards, support_count)
    winners = [family for family, count in enumerate(counts) if count == max(counts)]
    if winners != [intended_family] or margin_count < 1:
        raise RuntimeError("ambiguous support evidence escaped generator")
    return support_boards, evidence_vector, counts, margin_count, margin_normalized, support_stratum, margin_stratum


def support_count_for_row(idx, seed, split_id, support_counts):
    return support_counts[idx % len(support_counts)]


def make_row(seed, split, idx, support_counts, family_idx=None):
    split_id = {"train": 11, "test": 23, "ood": 37}[split]
    rng = random.Random(stable_mix(seed, split_id, idx, len(support_counts)))
    support_count = support_count_for_row(idx, seed, split_id, support_counts)
    intended_family = intended_family_for_idx(seed, split_id, idx) if family_idx is None else family_idx
    package = make_support_package(rng, split, idx, intended_family, support_count)
    support_boards, evidence_vector, evidence_counts, margin_count, margin_normalized, support_stratum, margin_stratum = package
    wrong_family = (intended_family + 1 + (idx % 4)) % 5
    wrong_package = make_support_package(rng, split, idx + 900000, wrong_family, support_count, force_clean=True)
    wrong_boards, wrong_evidence_vector, wrong_evidence_counts = wrong_package[:3]
    query_board, query_formula_values = make_query_board(rng, split, idx + 300000)
    query_target = query_formula_values[intended_family]
    pockets = list(range(9))
    rng.shuffle(pockets)
    expected_selected = pockets.index(query_target)
    formula_pockets = [pockets.index(value) for value in query_formula_values]
    return {
        "id": idx,
        "source_seed": seed,
        "split": split,
        "support_count": support_count,
        "support_stratum": support_stratum,
        "margin_stratum": margin_stratum,
        "intended_family": FAMILIES[intended_family],
        "intended_family_index": intended_family,
        "wrong_support_family": FAMILIES[wrong_family],
        "wrong_support_family_index": wrong_family,
        "support_boards": support_boards,
        "wrong_support_boards": wrong_boards,
        "support_evidence_vector": evidence_vector,
        "support_evidence_counts": evidence_counts,
        "wrong_support_evidence_vector": wrong_evidence_vector,
        "wrong_support_evidence_counts": wrong_evidence_counts,
        "support_evidence_margin_count": margin_count,
        "support_evidence_margin_normalized": margin_normalized,
        "query_board": query_board,
        "query_formula_values": query_formula_values,
        "query_target_symbol": query_target,
        "pockets": pockets,
        "expected_selected": expected_selected,
        "formula_pockets": formula_pockets,
    }


def make_rows(seed, split, count, support_counts):
    return [make_row(seed, split, idx, support_counts) for idx in range(count)]


def make_counterfactual_rows(seed, split, count, support_counts):
    split_id = {"train": 41, "test": 53, "ood": 67}[split]
    rng = random.Random(stable_mix(seed, split_id, count, len(support_counts)))
    rows = []
    base_count = (count + 4) // 5
    for base_idx in range(base_count):
        support_count = support_counts[(base_idx + seed + split_id) % len(support_counts)]
        query_board, query_formula_values = make_query_board(rng, split, base_idx + 600000)
        pockets = list(range(9))
        rng.shuffle(pockets)
        formula_pockets = [pockets.index(value) for value in query_formula_values]
        for family_idx in range(5):
            if len(rows) >= count:
                break
            row_idx = len(rows)
            package = make_support_package(rng, split, base_idx * 1000 + family_idx, family_idx, support_count)
            support_boards, evidence_vector, evidence_counts, margin_count, margin_normalized, support_stratum, margin_stratum = package
            target = query_formula_values[family_idx]
            rows.append(
                {
                    "id": row_idx,
                    "source_seed": seed,
                    "counterfactual_group": base_idx,
                    "split": split,
                    "support_count": support_count,
                    "support_stratum": support_stratum,
                    "margin_stratum": margin_stratum,
                    "intended_family": FAMILIES[family_idx],
                    "intended_family_index": family_idx,
                    "wrong_support_family": FAMILIES[(family_idx + 1) % 5],
                    "wrong_support_family_index": (family_idx + 1) % 5,
                    "support_boards": support_boards,
                    "wrong_support_boards": [],
                    "support_evidence_vector": evidence_vector,
                    "support_evidence_counts": evidence_counts,
                    "wrong_support_evidence_vector": [0.0 for _ in range(5)],
                    "wrong_support_evidence_counts": [0 for _ in range(5)],
                    "support_evidence_margin_count": margin_count,
                    "support_evidence_margin_normalized": margin_normalized,
                    "query_board": query_board,
                    "query_formula_values": query_formula_values,
                    "query_target_symbol": target,
                    "pockets": pockets,
                    "expected_selected": pockets.index(target),
                    "formula_pockets": formula_pockets,
                }
            )
    return rows


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def append_jsonl(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def top_two(scores):
    best_idx = 0
    best = scores[0]
    second = -1.0e30
    for idx in range(1, len(scores)):
        score = scores[idx]
        if score > best:
            second = best
            best = score
            best_idx = idx
        elif score > second:
            second = score
    return best_idx, best - second


def route_formula(row, family_idx):
    return row["formula_pockets"][family_idx], 1.0


def frozen_identity_router(row, family_idx):
    scores = [0.0 for _ in range(9)]
    for formula_idx, pocket_idx in enumerate(row["formula_pockets"]):
        scores[pocket_idx] += 10.0 if formula_idx == family_idx else 0.0
    return top_two(scores)


def initial_vector_model(rng):
    return {
        "weights": [rng.uniform(-0.05, 0.05) for _ in range(5)],
        "pocket_bias": [rng.uniform(-0.01, 0.01) for _ in range(9)],
    }


def clone_vector_model(model):
    return {"weights": model["weights"][:], "pocket_bias": model["pocket_bias"][:]}


def initial_raw_model(rng):
    return {
        "equality": [[rng.uniform(-0.05, 0.05) for _ in range(9)] for _ in range(9)],
        "channel": [[rng.uniform(-0.05, 0.05) for _ in range(5)] for _ in range(5)],
        "bias": [rng.uniform(-0.02, 0.02) for _ in range(5)],
    }


def clone_raw_model(model):
    return {
        "equality": [row[:] for row in model["equality"]],
        "channel": [row[:] for row in model["channel"]],
        "bias": model["bias"][:],
    }


def ablated_equality_model(model):
    candidate = clone_raw_model(model)
    candidate["equality"] = [[0.0 for _ in range(9)] for _ in range(9)]
    return candidate


def shuffled_equality_model(model):
    candidate = clone_raw_model(model)
    candidate["equality"] = [row[1:] + row[:1] for row in candidate["equality"]]
    return candidate


def raw_support_boards_for_mode(row, mode):
    if mode == "wrong":
        return row["wrong_support_boards"]
    return row["support_boards"]


def learned_match_distribution(model, center):
    row = model["equality"][center]
    offset = max(row)
    exps = [math.exp(max(-60.0, min(60.0, value - offset))) for value in row]
    total = sum(exps)
    return [value / total for value in exps]


def learned_match_readout(model, center, score_mode="softmax"):
    if score_mode == "softmax":
        return learned_match_distribution(model, center)
    if score_mode == "argmax_margin":
        weights = model["equality"][center]
        argmax_idx, margin = top_two(weights)
        if margin <= 0.0:
            return [0.0 for _ in range(9)]
        values = [0.0 for _ in range(9)]
        values[argmax_idx] = margin
        return values
    if score_mode == "argmax_binary":
        weights = model["equality"][center]
        argmax_idx, margin = top_two(weights)
        if margin <= 0.0:
            return [0.0 for _ in range(9)]
        values = [0.0 for _ in range(9)]
        values[argmax_idx] = 1.0
        return values
    raise ValueError(f"unknown learned match score mode: {score_mode}")


def raw_evidence_scores(model, row, mode="normal", score_mode="softmax"):
    if mode in {"no_center", "no_formula"}:
        return [0.0 for _ in range(5)]
    scores = [0.0 for _ in range(5)]
    boards = raw_support_boards_for_mode(row, mode)
    for board_idx, board in enumerate(boards):
        center = board[1][1]
        if mode == "shuffled_center":
            center = (center + 1 + board_idx) % 9
        match_probs = learned_match_readout(model, center, score_mode)
        for formula_idx in range(5):
            candidate_idx = formula_idx
            if mode == "shuffled_formula":
                candidate_idx = (formula_idx + 1) % 5
            candidate_value = family_target(candidate_idx, board)
            scores[formula_idx] += match_probs[candidate_value]
    support_count = max(1, len(boards))
    return [value / support_count for value in scores]


def sharpen_evidence(scores, temperature=1.0):
    if temperature == 1.0 or not any(scores):
        return scores[:]
    powered = [max(0.0, value) ** temperature for value in scores]
    total = sum(powered)
    if total <= 0.0:
        return scores[:]
    original_total = sum(max(0.0, value) for value in scores)
    scale = original_total / total if total else 1.0
    return [value * scale for value in powered]


def oracle_raw_evidence_scores(row, mode="normal"):
    if mode in {"no_center", "no_formula"}:
        return [0.0 for _ in range(5)]
    counts = [0 for _ in range(5)]
    boards = raw_support_boards_for_mode(row, mode)
    for board_idx, board in enumerate(boards):
        center = board[1][1]
        if mode == "shuffled_center":
            center = (center + 1 + board_idx) % 9
        for formula_idx in range(5):
            candidate_idx = formula_idx
            if mode == "shuffled_formula":
                candidate_idx = (formula_idx + 1) % 5
            counts[formula_idx] += int(center == family_target(candidate_idx, board))
    support_count = max(1, len(boards))
    return [count / support_count for count in counts]


def infer_from_raw_scores(model, scores):
    rule_scores = model["bias"][:]
    for channel_idx, evidence_value in enumerate(scores):
        if evidence_value:
            for family_idx in range(5):
                rule_scores[family_idx] += evidence_value * model["channel"][channel_idx][family_idx]
    return top_two(rule_scores)


def infer_from_precomputed(row, mode="normal"):
    if mode == "wrong":
        scores = row["wrong_support_evidence_vector"]
    else:
        scores = row["support_evidence_vector"]
    return top_two(scores)


def predict_learned_raw(model, row, raw_mode="normal", router_mode="oracle", temperature=1.0, score_mode="softmax"):
    raw_scores = raw_evidence_scores(model, row, raw_mode, score_mode)
    raw_scores = sharpen_evidence(raw_scores, temperature)
    if router_mode == "evidence_argmax":
        family_idx, margin = top_two(raw_scores)
        selected, _ = route_formula(row, family_idx)
        return selected, margin, family_idx, margin
    family_idx, margin = infer_from_raw_scores(model, raw_scores)
    if router_mode == "frozen_identity":
        selected, route_margin = frozen_identity_router(row, family_idx)
        return selected, min(margin, route_margin), family_idx, margin
    selected, _ = route_formula(row, family_idx)
    return selected, margin, family_idx, margin


def predict_query_vector(model, row):
    scores = model["pocket_bias"][:]
    for formula_idx, pocket_idx in enumerate(row["formula_pockets"]):
        scores[pocket_idx] += model["weights"][formula_idx]
    selected, margin = top_two(scores)
    return selected, margin, None, None


def predict_precomputed_support(row):
    family_idx, margin = infer_from_precomputed(row)
    selected, _ = route_formula(row, family_idx)
    return selected, margin, family_idx, margin


def predict_oracle_raw(row):
    family_idx, margin = top_two(oracle_raw_evidence_scores(row))
    selected, _ = route_formula(row, family_idx)
    return selected, margin, family_idx, margin


def raw_mutation_specs(rng, population, step):
    specs = []
    for symbol in range(9):
        specs.append(("equality_row_delta", symbol, step))
    for channel_idx in range(5):
        specs.append(("channel_row_delta", channel_idx, step))
    for symbol in range(9):
        specs.append(("equality_weight_delta", symbol, symbol, step))
    for channel_idx in range(5):
        specs.append(("channel_weight_delta", channel_idx, channel_idx, step))
    while len(specs) < population:
        specs.append((MUTATION_TYPES[rng.randrange(len(MUTATION_TYPES))],))
    rng.shuffle(specs)
    return specs[:population]


def apply_raw_mutation(rng, model, spec, step):
    candidate = clone_raw_model(model)
    op = spec[0]
    if op == "equality_weight_delta":
        row_idx = spec[1] if len(spec) > 1 else rng.randrange(9)
        col_idx = spec[2] if len(spec) > 2 else rng.randrange(9)
        delta = spec[3] if len(spec) > 3 else rng.uniform(-step, step)
        candidate["equality"][row_idx][col_idx] += delta
    elif op == "equality_row_delta":
        row_idx = spec[1] if len(spec) > 1 else rng.randrange(9)
        if len(spec) > 2:
            for col_idx in range(9):
                candidate["equality"][row_idx][col_idx] += spec[2] if col_idx == row_idx else -0.25 * spec[2]
        else:
            for col_idx in range(9):
                candidate["equality"][row_idx][col_idx] += rng.uniform(-step, step)
    elif op == "equality_column_delta":
        col_idx = rng.randrange(9)
        delta = rng.uniform(-step, step)
        for row_idx in range(9):
            candidate["equality"][row_idx][col_idx] += delta
    elif op == "channel_weight_delta":
        row_idx = spec[1] if len(spec) > 1 else rng.randrange(5)
        col_idx = spec[2] if len(spec) > 2 else rng.randrange(5)
        delta = spec[3] if len(spec) > 3 else rng.uniform(-step, step)
        candidate["channel"][row_idx][col_idx] += delta
    elif op == "channel_row_delta":
        row_idx = spec[1] if len(spec) > 1 else rng.randrange(5)
        if len(spec) > 2:
            for col_idx in range(5):
                candidate["channel"][row_idx][col_idx] += spec[2] if col_idx == row_idx else -0.25 * spec[2]
        else:
            for col_idx in range(5):
                candidate["channel"][row_idx][col_idx] += rng.uniform(-step, step)
    elif op == "channel_column_delta":
        col_idx = rng.randrange(5)
        delta = rng.uniform(-step, step)
        for row_idx in range(5):
            candidate["channel"][row_idx][col_idx] += delta
    elif op == "rule_bias_delta":
        candidate["bias"][rng.randrange(5)] += rng.uniform(-step, step)
    elif op == "equality_row_swap":
        a = rng.randrange(9)
        b = rng.randrange(9)
        candidate["equality"][a], candidate["equality"][b] = candidate["equality"][b], candidate["equality"][a]
    elif op == "equality_column_swap":
        a = rng.randrange(9)
        b = rng.randrange(9)
        for row in candidate["equality"]:
            row[a], row[b] = row[b], row[a]
    elif op == "channel_row_swap":
        a = rng.randrange(5)
        b = rng.randrange(5)
        candidate["channel"][a], candidate["channel"][b] = candidate["channel"][b], candidate["channel"][a]
    elif op == "channel_column_swap":
        a = rng.randrange(5)
        b = rng.randrange(5)
        for row in candidate["channel"]:
            row[a], row[b] = row[b], row[a]
    elif op == "prune_small_weights":
        threshold = 0.05 * step
        for row_idx in range(9):
            for col_idx in range(9):
                if abs(candidate["equality"][row_idx][col_idx]) < threshold:
                    candidate["equality"][row_idx][col_idx] = 0.0
        for row_idx in range(5):
            for col_idx in range(5):
                if abs(candidate["channel"][row_idx][col_idx]) < threshold:
                    candidate["channel"][row_idx][col_idx] = 0.0
    return candidate, op


def vector_mutation_specs(rng, population, step):
    specs = []
    for idx in range(5):
        specs.append(("vector_weight_delta", idx, step))
        specs.append(("vector_weight_delta", idx, -step))
    ops = ["vector_weight_delta", "pocket_bias_delta", "prune_small_weights"]
    while len(specs) < population:
        specs.append((ops[rng.randrange(len(ops))],))
    rng.shuffle(specs)
    return specs[:population]


def apply_vector_mutation(rng, model, spec, step):
    candidate = clone_vector_model(model)
    op = spec[0]
    if op == "vector_weight_delta":
        idx = spec[1] if len(spec) > 1 else rng.randrange(5)
        delta = spec[2] if len(spec) > 2 else rng.uniform(-step, step)
        candidate["weights"][idx] += delta
    elif op == "pocket_bias_delta":
        candidate["pocket_bias"][rng.randrange(9)] += rng.uniform(-0.25 * step, 0.25 * step)
    elif op == "prune_small_weights":
        threshold = 0.05 * step
        for idx, value in enumerate(candidate["weights"]):
            if abs(value) < threshold:
                candidate["weights"][idx] = 0.0
    return candidate, op


def balanced_subset(rows, limit, edge_oversample=False):
    if len(rows) <= limit:
        return rows
    buckets = defaultdict(list)
    for row in rows:
        buckets[(row["intended_family_index"], row["support_count"], row["support_stratum"], row["margin_stratum"])].append(row)
    keys = list(buckets)
    out = []
    cursor = 0
    while len(out) < limit:
        key = keys[cursor % len(keys)]
        bucket = buckets[key]
        out.append(bucket[(cursor // len(keys)) % len(bucket)])
        cursor += 1
    if edge_oversample:
        edge = [
            row for row in rows
            if row["support_stratum"] == "noisy_majority"
            and row["margin_stratum"] == "margin_low"
            and row["intended_family"] in {"pair", "diag", "row"}
        ]
        if edge:
            keep = max(1, limit // 2)
            out = out[:keep] + [edge[idx % len(edge)] for idx in range(limit - keep)]
    return out[:limit]


def rule_scores_for_raw(model, raw_scores):
    rule_scores = model["bias"][:]
    for channel_idx, evidence_value in enumerate(raw_scores):
        if evidence_value:
            for family_idx in range(5):
                rule_scores[family_idx] += evidence_value * model["channel"][channel_idx][family_idx]
    return rule_scores


def fitness_raw(model, rows, raw_mode="normal", router_mode="oracle", temperature=1.0, margin_preserving=False, score_mode="softmax"):
    correct = 0
    margin_sum = 0.0
    rule_correct = 0
    edge_correct = 0
    edge_total = 0
    gap_reward = 0.0
    for row in rows:
        selected, margin, family_idx, _ = predict_learned_raw(model, row, raw_mode, router_mode, temperature, score_mode)
        correct += int(selected == row["expected_selected"])
        rule_correct += int(family_idx == row["intended_family_index"])
        margin_sum += margin
        if margin_preserving and "noisy_majority_margin_low" in d43s_strata(row):
            edge_total += 1
            edge_correct += int(family_idx == row["intended_family_index"])
            raw_scores = sharpen_evidence(raw_evidence_scores(model, row, raw_mode, score_mode), temperature)
            rule_scores = rule_scores_for_raw(model, raw_scores)
            intended = row["intended_family_index"]
            strongest_distractor = max(rule_scores[idx] for idx in range(5) if idx != intended)
            gap_reward += rule_scores[intended] - strongest_distractor
    equality_correct, equality_margin = equality_kernel_regularizer(model)
    channel_correct, channel_margin = channel_gate_regularizer(model)
    return (
        correct * 10000.0
        + rule_correct * 1000.0
        + edge_correct * 25000.0
        + (edge_correct / max(1, edge_total)) * 5000.0
        + gap_reward * 2000.0
        + equality_correct * 5000.0
        + channel_correct * 12000.0
        + margin_sum
        + equality_margin * 250.0
        + channel_margin * 800.0
    )


def equality_kernel_regularizer(model):
    correct = 0
    margin_sum = 0.0
    for symbol in range(9):
        row = model["equality"][symbol]
        best_idx, margin = top_two(row)
        correct += int(best_idx == symbol)
        margin_sum += margin
    return correct, margin_sum


def channel_gate_regularizer(model):
    correct = 0
    margin_sum = 0.0
    patterns = []
    for family_idx in range(5):
        clean = [0.0 for _ in range(5)]
        clean[family_idx] = 1.0
        patterns.append((clean, family_idx))
        for distractor_idx in range(5):
            if distractor_idx == family_idx:
                continue
            for intended_mass, distractor_mass in [(2.0 / 3.0, 1.0 / 3.0), (3.0 / 5.0, 2.0 / 5.0), (4.0 / 5.0, 1.0 / 5.0)]:
                evidence = [0.0 for _ in range(5)]
                evidence[family_idx] = intended_mass
                evidence[distractor_idx] = distractor_mass
                patterns.append((evidence, family_idx))
    for evidence, target_family in patterns:
        inferred_family, margin = infer_from_raw_scores(model, evidence)
        correct += int(inferred_family == target_family)
        margin_sum += margin
    return correct, margin_sum


def fitness_vector(model, rows):
    correct = 0
    margin_sum = 0.0
    for row in rows:
        selected, margin, _family_idx, _rule_margin = predict_query_vector(model, row)
        correct += int(selected == row["expected_selected"])
        margin_sum += margin
    return correct * 10000.0 + margin_sum


def train_raw_model(train_rows, seed, generations, population, job_dir, objective="baseline", temperature=1.0, router_mode="oracle", score_mode="softmax"):
    rng = random.Random(stable_mix(seed, 101, generations, population))
    edge_oversample = objective in {"edge_oversampling", "combined"}
    margin_preserving = objective in {"margin_preserving", "combined"}
    fit_rows = balanced_subset(train_rows, min(360, len(train_rows)), edge_oversample=edge_oversample)
    best = initial_raw_model(rng)
    best_fit = fitness_raw(best, fit_rows, temperature=temperature, margin_preserving=margin_preserving, router_mode=router_mode, score_mode=score_mode)
    accepted = Counter({op: 0 for op in MUTATION_TYPES})
    rejected = Counter({op: 0 for op in MUTATION_TYPES})
    cold_metrics = evaluate_rows(
        train_rows,
        lambda row: predict_learned_raw(
            best,
            row,
            router_mode=router_mode,
            temperature=temperature,
            score_mode=score_mode,
        ),
    )
    initial_equality = equality_kernel_identity(best)
    initial_channel = channel_gate_identity(best)
    training_audit = {
        "cold_init_train_accuracy": cold_metrics["accuracy"],
        "cold_init_rule_family_train_accuracy": cold_metrics["rule_family_accuracy"],
        "cold_init_solved": cold_metrics["accuracy"] >= 0.95 and cold_metrics["rule_family_accuracy"] >= 0.95,
        "initial_equality_identity": initial_equality,
        "initial_channel_identity": initial_channel,
        "training_accepts_by_fitness": True,
        "objective": objective,
        "training_temperature": temperature,
        "training_router_mode": router_mode,
        "training_score_mode": score_mode,
        "prebaked_identity_initialization": False,
    }
    convergence_generation = None
    perfect_since = None
    generations_executed = 0
    for generation in range(generations):
        generations_executed = generation + 1
        step = max(0.05, 1.1 * (1.0 - (generation / max(1, generations))))
        records = []
        winner = None
        winner_op = None
        winner_fit = -1.0e30
        for spec in raw_mutation_specs(rng, population, step):
            candidate, op = apply_raw_mutation(rng, best, spec, step)
            fit = fitness_raw(candidate, fit_rows, temperature=temperature, margin_preserving=margin_preserving, router_mode=router_mode, score_mode=score_mode)
            records.append((fit, op, candidate))
            if fit > winner_fit:
                winner = candidate
                winner_op = op
                winner_fit = fit
        accepted_this_generation = winner_fit >= best_fit
        if accepted_this_generation:
            best = winner
            best_fit = winner_fit
            accepted[winner_op] += 1
        skipped = False
        for fit, op, _candidate in records:
            if accepted_this_generation and not skipped and op == winner_op and fit == winner_fit:
                skipped = True
                continue
            rejected[op] += 1
        if generation % max(1, generations // 30) == 0 or generation == generations - 1:
            metrics = evaluate_rows(train_rows, lambda row: predict_learned_raw(best, row, temperature=temperature, router_mode=router_mode, score_mode=score_mode))
            append_jsonl(job_dir / "train_metrics.jsonl", {"generation": generation, "train_accuracy": metrics["accuracy"], "rule_family_train_accuracy": metrics["rule_family_accuracy"], "fitness": best_fit})
            append_jsonl(job_dir / "progress.jsonl", {"generation": generation, "generations": generations})
            if metrics["accuracy"] >= 0.98 and metrics["rule_family_accuracy"] >= 0.98 and convergence_generation is None:
                convergence_generation = generation
            equality_correct, equality_margin = equality_kernel_regularizer(best)
            channel_correct, channel_margin = channel_gate_regularizer(best)
            if (
                metrics["accuracy"] >= 0.995
                and metrics["rule_family_accuracy"] >= 0.995
            ):
                if perfect_since is None:
                    perfect_since = generation
                elif generation - perfect_since >= max(8, generations // 80):
                    break
            else:
                perfect_since = None
    mutation = {
        "accepted_mutations_by_type": dict(accepted),
        "rejected_mutations_by_type": dict(rejected),
        "mutation_acceptance_rate": sum(accepted.values()) / max(1, sum(accepted.values()) + sum(rejected.values())),
        "convergence_generation": convergence_generation,
        "generations_requested": generations,
        "generations_executed": generations_executed,
    }
    return best, mutation, training_audit


def train_vector_model(train_rows, seed, generations, population, job_dir):
    rng = random.Random(stable_mix(seed, 211, generations, population))
    fit_rows = balanced_subset(train_rows, min(260, len(train_rows)))
    best = initial_vector_model(rng)
    best_fit = fitness_vector(best, fit_rows)
    stale = 0
    limit = max(1, generations // 2)
    for generation in range(limit):
        step = max(0.05, 0.8 * (1.0 - (generation / max(1, limit))))
        winner = None
        winner_fit = -1.0e30
        for spec in vector_mutation_specs(rng, population, step):
            candidate, _op = apply_vector_mutation(rng, best, spec, step)
            fit = fitness_vector(candidate, fit_rows)
            if fit > winner_fit:
                winner = candidate
                winner_fit = fit
        if winner_fit > best_fit:
            best = winner
            best_fit = winner_fit
            stale = 0
        else:
            stale += 1
        if generation % max(1, limit // 10) == 0 or generation == limit - 1:
            metrics = evaluate_rows(train_rows, lambda row: predict_query_vector(best, row))
            append_jsonl(job_dir / "train_metrics.jsonl", {"generation": generation, "train_accuracy": metrics["accuracy"], "fitness": best_fit})
            append_jsonl(job_dir / "progress.jsonl", {"generation": generation, "generations": limit})
        if stale >= 30:
            break
    return best, {"generations_requested": limit, "generations_executed": generation + 1}


def evaluate_rows(rows, predictor, random_seed=None):
    rng = random.Random(random_seed) if random_seed is not None else None
    correct = 0
    rule_correct = 0
    rule_total = 0
    per_family_total = Counter()
    per_family_correct = Counter()
    per_support_total = Counter()
    per_support_correct = Counter()
    per_margin_total = Counter()
    per_margin_correct = Counter()
    confusion = [[0 for _ in range(9)] for _ in range(9)]
    family_confusion = [[0 for _ in range(5)] for _ in range(5)]
    margins = []
    low_margin_count = 0
    low_margin_errors = 0
    outputs = []
    for row in rows:
        if rng is not None:
            selected = rng.randrange(9)
            margin = 0.0
            inferred_family = None
            rule_margin = None
        else:
            selected, margin, inferred_family, rule_margin = predictor(row)
        truth = row["expected_selected"]
        is_correct = selected == truth
        correct += int(is_correct)
        per_family_total[row["intended_family"]] += 1
        per_family_correct[row["intended_family"]] += int(is_correct)
        support_key = f"support_count_{row['support_count']}"
        per_support_total[support_key] += 1
        per_support_correct[support_key] += int(is_correct)
        for stratum in d43s_strata(row):
            per_margin_total[stratum] += 1
            per_margin_correct[stratum] += int(is_correct)
        confusion[truth][selected] += 1
        if inferred_family is not None:
            rule_total += 1
            rule_correct += int(inferred_family == row["intended_family_index"])
            family_confusion[row["intended_family_index"]][inferred_family] += 1
        margins.append(margin)
        if margin < 0.1:
            low_margin_count += 1
            low_margin_errors += int(not is_correct)
        outputs.append(
            {
                "id": row["id"],
                "family": row["intended_family"],
                "support_count": row["support_count"],
                "support_stratum": row["support_stratum"],
                "margin_stratum": row["margin_stratum"],
                "truth": truth,
                "pred": selected,
                "inferred_family": FAMILIES[inferred_family] if inferred_family is not None else None,
                "margin": margin,
                "rule_margin": rule_margin,
                "d43s_strata": d43s_strata(row),
            }
        )
    per_support = {
        f"support_count_{count}": per_support_correct[f"support_count_{count}"] / max(1, per_support_total[f"support_count_{count}"])
        for count in SUPPORT_COUNT_KEYS
    }
    per_margin = {
        stratum: per_margin_correct[stratum] / max(1, per_margin_total[stratum])
        for stratum in MARGIN_STRATA
    }
    return {
        "accuracy": correct / max(1, len(rows)),
        "rule_family_accuracy": (rule_correct / rule_total) if rule_total else None,
        "per_support_count_accuracy": per_support,
        "per_margin_strata_accuracy": per_margin,
        "per_family_accuracy": {
            family: per_family_correct[family] / max(1, per_family_total[family])
            for family in FAMILIES
        },
        "pocket_confusion_matrix": confusion,
        "family_confusion_matrix": family_confusion if rule_total else None,
        "median_score_margin": statistics.median(margins) if margins else 0.0,
        "low_margin_error_rate": low_margin_errors / max(1, low_margin_count),
        "error_count": len(rows) - correct,
        "outputs": outputs,
    }


def d43s_strata(row):
    strata = []
    if row["support_stratum"] == "noisy_majority" and row["margin_stratum"] == "margin_low":
        strata.append("noisy_majority_margin_low")
    if row["support_stratum"] == "noisy_majority" and row["margin_stratum"] == "margin_high":
        strata.append("noisy_majority_margin_high")
    if row["support_stratum"] == "clean_unanimous":
        strata.append("clean_unanimous")
    if row["margin_stratum"] == "margin_low":
        if row["intended_family"] in {"pair", "diag", "row"}:
            strata.append(f"{row['intended_family']}_low_margin")
        if row["support_count"] in {3, 5}:
            strata.append(f"support_count_{row['support_count']}_low_margin")
    return strata


def flatten_metrics(train, test, ood):
    metrics = {
        "train_accuracy": train["accuracy"],
        "test_accuracy": test["accuracy"],
        "ood_accuracy": ood["accuracy"],
        "rule_family_train_accuracy": train["rule_family_accuracy"],
        "rule_family_test_accuracy": test["rule_family_accuracy"],
        "rule_family_ood_accuracy": ood["rule_family_accuracy"],
        "per_support_count_accuracy": test["per_support_count_accuracy"],
        "per_margin_strata_accuracy": test["per_margin_strata_accuracy"],
        "per_family_accuracy": test["per_family_accuracy"],
        "error_count": test["error_count"],
        "pocket_confusion_matrix": test["pocket_confusion_matrix"],
        "family_confusion_matrix": test["family_confusion_matrix"],
        "median_score_margin": test["median_score_margin"],
        "median_evidence_gap": test["median_score_margin"],
        "low_margin_error_rate": test["low_margin_error_rate"],
    }
    for stratum in MARGIN_STRATA:
        metrics[f"{stratum}_accuracy"] = test["per_margin_strata_accuracy"].get(stratum, 0.0)
    return metrics


def dataset_invariants(rows, counterfactual_rows):
    duplicate = 0
    missing = 0
    selected = 0
    ambiguous = 0
    ties = 0
    unique = 0
    wrong_mismatch = 0
    for row in rows:
        target = row["query_target_symbol"]
        count = sum(1 for symbol in row["pockets"] if symbol == target)
        duplicate += int(count > 1)
        missing += int(count == 0)
        selected += int(row["pockets"][row["expected_selected"]] == target)
        counts = row["support_evidence_counts"]
        max_count = max(counts)
        winners = [idx for idx, value in enumerate(counts) if value == max_count]
        ties += int(len(winners) > 1)
        intended_unique = winners == [row["intended_family_index"]]
        unique += int(intended_unique)
        ambiguous += int((not intended_unique) or row["support_evidence_margin_count"] < 1)
        wrong_mismatch += int(row["query_formula_values"][row["wrong_support_family_index"]] != row["query_target_symbol"])
    grouped = defaultdict(list)
    for row in counterfactual_rows:
        grouped[(row["source_seed"], row["split"], row["counterfactual_group"])].append(row["query_target_symbol"])
    collisions = sum(int(len(set(values)) != len(values)) for values in grouped.values())
    return {
        "duplicate_target_pocket_rate": duplicate / max(1, len(rows)),
        "missing_target_pocket_rate": missing / max(1, len(rows)),
        "expected_selected_points_to_target_rate": selected / max(1, len(rows)),
        "ambiguous_support_rate": ambiguous / max(1, len(rows)),
        "multi_family_support_tie_rate": ties / max(1, len(rows)),
        "intended_family_unique_evidence_rate": unique / max(1, len(rows)),
        "counterfactual_target_collision_rate": collisions / max(1, len(grouped)),
        "wrong_support_query_mismatch_rate": wrong_mismatch / max(1, len(rows)),
    }


def rule_oracle_accuracy(rows):
    selected_correct = 0
    rule_correct = 0
    for row in rows:
        family_idx, _margin = top_two(row["support_evidence_vector"])
        selected = row["formula_pockets"][family_idx]
        selected_correct += int(selected == row["expected_selected"])
        rule_correct += int(family_idx == row["intended_family_index"])
    return selected_correct / max(1, len(rows)), rule_correct / max(1, len(rows))


def true_family_oracle_accuracy(rows):
    return sum(
        int(row["formula_pockets"][row["intended_family_index"]] == row["expected_selected"])
        for row in rows
    ) / max(1, len(rows))


def equality_kernel_identity(model):
    mapping = {}
    diag_probs = []
    off_probs = []
    entropies = []
    for symbol_idx, weights in enumerate(model["equality"]):
        argmax_idx = max(range(9), key=lambda candidate_idx: weights[candidate_idx])
        mapping[str(symbol_idx)] = str(argmax_idx)
        peak = max(weights)
        exps = [math.exp(value - peak) for value in weights]
        total = sum(exps)
        probs = [value / total for value in exps]
        diag_probs.append(probs[symbol_idx])
        off_probs.append(sum(probs[idx] for idx in range(9) if idx != symbol_idx))
        entropies.append(-sum(prob * math.log(max(prob, 1.0e-12)) for prob in probs) / math.log(9))
    return {
        "mapping": mapping,
        "diagonal_mass": statistics.mean(diag_probs),
        "off_diagonal_mass": statistics.mean(off_probs),
        "entropy": statistics.mean(entropies),
        "matrix": [[round(value, 6) for value in row] for row in model["equality"]],
    }


def channel_gate_identity(model):
    effective = [
        [model["channel"][row_idx][col_idx] + model["bias"][col_idx] for col_idx in range(5)]
        for row_idx in range(5)
    ]
    mapping = {}
    aligned = 0
    diag_probs = []
    off_probs = []
    entropies = []
    for evidence_idx, weights in enumerate(effective):
        argmax_idx = max(range(5), key=lambda family_idx: weights[family_idx])
        mapping[FAMILIES[evidence_idx]] = FAMILIES[argmax_idx]
        aligned += int(argmax_idx == evidence_idx)
        peak = max(weights)
        exps = [math.exp(value - peak) for value in weights]
        total = sum(exps)
        probs = [value / total for value in exps]
        diag_probs.append(probs[evidence_idx])
        off_probs.append(sum(probs[idx] for idx in range(5) if idx != evidence_idx))
        entropies.append(-sum(prob * math.log(max(prob, 1.0e-12)) for prob in probs) / math.log(5))
    return {
        "mapping": mapping,
        "alignment_score": aligned / 5,
        "diagonal_mass": statistics.mean(diag_probs),
        "off_diagonal_mass": statistics.mean(off_probs),
        "entropy": statistics.mean(entropies),
        "effective_matrix": [[round(value, 6) for value in row] for row in effective],
    }


def select_temperature(model, validation_rows):
    scored = []
    for temperature in TEMPERATURES:
        metrics = evaluate_rows(validation_rows, lambda row, t=temperature: predict_learned_raw(model, row, temperature=t))
        scored.append(
            {
                "temperature": temperature,
                "validation_accuracy": metrics["accuracy"],
                "validation_low_margin_accuracy": metrics["per_margin_strata_accuracy"].get("noisy_majority_margin_low", 0.0),
                "validation_median_evidence_gap": metrics["median_score_margin"],
            }
        )
    best = max(scored, key=lambda item: (item["validation_low_margin_accuracy"], item["validation_accuracy"], item["temperature"]))
    return best["temperature"], scored


def model_reports(model):
    return equality_kernel_identity(model), channel_gate_identity(model)


def run_job(job, config, out_path):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    seed = job["seed"]
    arm = job["arm"]
    out = Path(out_path)
    job_dir = out / f"arm_{arm}" / f"seed_{seed}"
    job_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    train_rows = make_rows(seed, "train", config["train_rows_per_seed"], config["support_counts"])
    test_rows = make_rows(seed, "test", config["test_rows_per_seed"], config["support_counts"])
    ood_rows = make_rows(seed, "ood", config["ood_rows_per_seed"], config["support_counts"])
    validation_rows = train_rows[::5] if len(train_rows) >= 5 else train_rows
    extra = {}

    if arm == "HARD_VOTE_ORACLE_UPPER_BOUND":
        predictor = predict_oracle_raw
        train = evaluate_rows(train_rows, predictor)
        test = evaluate_rows(test_rows, predictor)
        ood = evaluate_rows(ood_rows, predictor)
        write_json(job_dir / "best_individual.json", {"type": "hard_vote_oracle_upper_bound"})
        append_jsonl(job_dir / "train_metrics.jsonl", {"generation": 0, "train_accuracy": train["accuracy"], "rule_family_train_accuracy": train["rule_family_accuracy"]})
        append_jsonl(job_dir / "progress.jsonl", {"generation": 0, "generations": 0})
    elif arm in {
        "D43_BASELINE_REPLAY",
        "MARGIN_PRESERVING_OBJECTIVE",
        "TEMPERATURE_SHARPENED_EVIDENCE",
        "FAMILY_BALANCED_EDGE_OVERSAMPLING",
        "COMBINED_SHARPENED_MODEL",
        "SHUFFLED_CENTER_CONTROL",
        "SHUFFLED_FORMULA_CANDIDATE_CONTROL",
        "NO_CENTER_CONTROL",
        "WRONG_SUPPORT_CONTROL",
    }:
        if arm in {"MARGIN_PRESERVING_OBJECTIVE"}:
            objective = "margin_preserving"
            training_temperature = 1.0
        elif arm in {"FAMILY_BALANCED_EDGE_OVERSAMPLING"}:
            objective = "edge_oversampling"
            training_temperature = 1.0
        elif arm in {"COMBINED_SHARPENED_MODEL", "SHUFFLED_CENTER_CONTROL", "SHUFFLED_FORMULA_CANDIDATE_CONTROL", "NO_CENTER_CONTROL", "WRONG_SUPPORT_CONTROL"}:
            objective = "combined"
            training_temperature = 4.0
        else:
            objective = "baseline"
            training_temperature = 1.0
        extractor_seed = stable_mix(seed, 43, 5)
        training_router_mode = "evidence_argmax" if objective == "combined" else "oracle"
        score_mode = "argmax_binary" if objective == "combined" else "softmax"
        model, mutation, training_audit = train_raw_model(
            train_rows,
            extractor_seed,
            config["generations"],
            config["population"],
            job_dir,
            objective=objective,
            temperature=training_temperature,
            router_mode=training_router_mode,
            score_mode=score_mode,
        )
        raw_mode = "normal"
        router_mode = training_router_mode
        eval_temperature = training_temperature
        temperature_sweep = None
        if arm == "TEMPERATURE_SHARPENED_EVIDENCE":
            eval_temperature, temperature_sweep = select_temperature(model, validation_rows)
            extra["temperature_sweep"] = temperature_sweep
        elif arm == "SHUFFLED_CENTER_CONTROL":
            raw_mode = "shuffled_center"
        elif arm == "SHUFFLED_FORMULA_CANDIDATE_CONTROL":
            raw_mode = "shuffled_formula"
        elif arm == "NO_CENTER_CONTROL":
            raw_mode = "no_center"
        elif arm == "WRONG_SUPPORT_CONTROL":
            raw_mode = "wrong"
        predictor = lambda row: predict_learned_raw(
            model,
            row,
            raw_mode=raw_mode,
            router_mode=router_mode,
            temperature=eval_temperature,
            score_mode=score_mode,
        )
        train = evaluate_rows(train_rows, predictor)
        test = evaluate_rows(test_rows, predictor)
        ood = evaluate_rows(ood_rows, predictor)
        equality_identity, channel_identity = model_reports(model)
        extra.update(
            {
                "mutation": mutation,
                "equality_identity": equality_identity,
                "channel_identity": channel_identity,
                "training_audit": training_audit,
                "raw_mode": raw_mode,
                "objective": objective,
                "eval_temperature": eval_temperature,
                "score_mode": score_mode,
            }
        )
        if arm == "WRONG_SUPPORT_CONTROL":
            def follow_rate(rows):
                count = 0
                for row in rows:
                    _selected, _margin, inferred_family, _rule_margin = predictor(row)
                    count += int(inferred_family == row["wrong_support_family_index"])
                return count / max(1, len(rows))
            extra["wrong_support_follow_rate_train"] = follow_rate(train_rows)
            extra["wrong_support_follow_rate_test"] = follow_rate(test_rows)
            extra["wrong_support_follow_rate_ood"] = follow_rate(ood_rows)
            extra["wrong_support_query_mismatch_rate_test"] = sum(
                int(row["query_formula_values"][row["wrong_support_family_index"]] != row["query_target_symbol"])
                for row in test_rows
            ) / max(1, len(test_rows))
        write_json(
            job_dir / "best_individual.json",
            {
                "type": arm.lower(),
                "model": model,
                "mutation": mutation,
                "equality_identity": equality_identity,
                "channel_identity": channel_identity,
                "training_audit": training_audit,
                "raw_mode": raw_mode,
                "router_mode": router_mode,
                "objective": objective,
                "eval_temperature": eval_temperature,
                "score_mode": score_mode,
                "temperature_sweep": temperature_sweep,
                "learned_inputs": "raw support boards plus fixed formula candidate primitive values; no aggregated support evidence vector",
            },
        )
    elif arm == "SAME_QUERY_DIFFERENT_RAW_SUPPORT_COUNTERFACTUAL":
        extractor_seed = stable_mix(seed, 43, 5)
        model, mutation, training_audit = train_raw_model(
            train_rows,
            extractor_seed,
            config["generations"],
            config["population"],
            job_dir,
            objective="combined",
            temperature=4.0,
            router_mode="evidence_argmax",
            score_mode="argmax_binary",
        )
        cf_train = make_counterfactual_rows(seed, "train", config["train_rows_per_seed"], config["support_counts"])
        cf_test = make_counterfactual_rows(seed, "test", config["test_rows_per_seed"], config["support_counts"])
        cf_ood = make_counterfactual_rows(seed, "ood", config["ood_rows_per_seed"], config["support_counts"])
        predictor = lambda row: predict_learned_raw(
            model,
            row,
            temperature=4.0,
            router_mode="evidence_argmax",
            score_mode="argmax_binary",
        )
        train = evaluate_rows(cf_train, predictor)
        test = evaluate_rows(cf_test, predictor)
        ood = evaluate_rows(cf_ood, predictor)
        equality_identity, channel_identity = model_reports(model)
        extra.update(
            {
                "mutation": mutation,
                "equality_identity": equality_identity,
                "channel_identity": channel_identity,
                "training_audit": training_audit,
                "counterfactual_group_count_test": len({row["counterfactual_group"] for row in cf_test}),
                "objective": "combined",
                "eval_temperature": 4.0,
                "router_mode": "evidence_argmax",
                "score_mode": "argmax_binary",
            }
        )
        write_json(
            job_dir / "best_individual.json",
            {
                "type": "same_query_different_raw_support_counterfactual",
                "model": model,
                "mutation": mutation,
                "equality_identity": equality_identity,
                "channel_identity": channel_identity,
                "training_audit": training_audit,
            },
        )
    else:
        raise ValueError(f"unknown arm: {arm}")

    write_jsonl(job_dir / "row_outputs_test.jsonl", test["outputs"])
    write_jsonl(job_dir / "row_outputs_ood.jsonl", ood["outputs"])
    metrics = {
        "seed": seed,
        "arm": arm,
        "ok": True,
        "wall_clock_sec": time.time() - started,
        **flatten_metrics(train, test, ood),
        **extra,
    }
    write_json(job_dir / "metrics.json", metrics)
    return metrics


def aggregate_arm(rows, failed_seed_count):
    if not rows:
        zero_support = {f"support_count_{count}": 0.0 for count in SUPPORT_COUNT_KEYS}
        zero_margin = {stratum: 0.0 for stratum in MARGIN_STRATA}
        return {
            "train_accuracy": 0.0,
            "test_accuracy": 0.0,
            "ood_accuracy": 0.0,
            "rule_family_train_accuracy": None,
            "rule_family_test_accuracy": None,
            "rule_family_ood_accuracy": None,
            "min_seed_test_accuracy": 0.0,
            "min_seed_ood_accuracy": 0.0,
            "per_support_count_accuracy": zero_support,
            "per_margin_strata_accuracy": zero_margin,
            "per_family_accuracy": {family: 0.0 for family in FAMILIES},
            "error_count": 0,
            "pocket_confusion_matrix": [[0 for _ in range(9)] for _ in range(9)],
            "family_confusion_matrix": None,
            "median_score_margin": 0.0,
            "low_margin_error_rate": 1.0,
            "failed_seed_count": failed_seed_count,
        }
    rule_train = [row["rule_family_train_accuracy"] for row in rows if row["rule_family_train_accuracy"] is not None]
    rule_test = [row["rule_family_test_accuracy"] for row in rows if row["rule_family_test_accuracy"] is not None]
    rule_ood = [row["rule_family_ood_accuracy"] for row in rows if row["rule_family_ood_accuracy"] is not None]
    family_matrix = None
    if any(row["family_confusion_matrix"] is not None for row in rows):
        family_matrix = [
            [sum((row["family_confusion_matrix"] or [[0 for _ in range(5)] for _ in range(5)])[i][j] for row in rows) for j in range(5)]
            for i in range(5)
        ]
    report = {
        "train_accuracy": statistics.mean(row["train_accuracy"] for row in rows),
        "test_accuracy": statistics.mean(row["test_accuracy"] for row in rows),
        "ood_accuracy": statistics.mean(row["ood_accuracy"] for row in rows),
        "rule_family_train_accuracy": statistics.mean(rule_train) if rule_train else None,
        "rule_family_test_accuracy": statistics.mean(rule_test) if rule_test else None,
        "rule_family_ood_accuracy": statistics.mean(rule_ood) if rule_ood else None,
        "min_seed_test_accuracy": min(row["test_accuracy"] for row in rows),
        "min_seed_ood_accuracy": min(row["ood_accuracy"] for row in rows),
        "per_support_count_accuracy": {
            f"support_count_{count}": statistics.mean(row["per_support_count_accuracy"][f"support_count_{count}"] for row in rows)
            for count in SUPPORT_COUNT_KEYS
        },
        "per_margin_strata_accuracy": {
            stratum: statistics.mean(row["per_margin_strata_accuracy"][stratum] for row in rows)
            for stratum in MARGIN_STRATA
        },
        "per_family_accuracy": {
            family: statistics.mean(row["per_family_accuracy"][family] for row in rows)
            for family in FAMILIES
        },
        "error_count": sum(row["error_count"] for row in rows),
        "pocket_confusion_matrix": [
            [sum(row["pocket_confusion_matrix"][i][j] for row in rows) for j in range(9)]
            for i in range(9)
        ],
        "family_confusion_matrix": family_matrix,
        "median_score_margin": statistics.median(row["median_score_margin"] for row in rows),
        "low_margin_error_rate": statistics.mean(row["low_margin_error_rate"] for row in rows),
        "failed_seed_count": failed_seed_count,
    }
    for stratum in MARGIN_STRATA:
        report[f"{stratum}_accuracy"] = report["per_margin_strata_accuracy"][stratum]
    report["median_evidence_gap"] = report["median_score_margin"]
    return report


def variance(values):
    return statistics.pvariance(values) if values else 0.0


def build_equality_kernel_report(rows):
    items = {str(row["seed"]): row["equality_identity"] for row in rows if "equality_identity" in row}
    if not items:
        return {
            "equality_kernel_diagonal_mass": 0.0,
            "equality_kernel_off_diagonal_mass": 0.0,
            "equality_kernel_argmax_mapping": {},
            "equality_kernel_entropy": 0.0,
            "equality_kernel_matrix_by_seed": {},
        }
    first_seed = sorted(items)[0]
    return {
        "equality_kernel_diagonal_mass": statistics.mean(item["diagonal_mass"] for item in items.values()),
        "equality_kernel_off_diagonal_mass": statistics.mean(item["off_diagonal_mass"] for item in items.values()),
        "equality_kernel_argmax_mapping": items[first_seed]["mapping"],
        "equality_kernel_argmax_mapping_by_seed": {seed: item["mapping"] for seed, item in items.items()},
        "equality_kernel_entropy": statistics.mean(item["entropy"] for item in items.values()),
        "equality_kernel_matrix_by_seed": {seed: item["matrix"] for seed, item in items.items()},
    }


def build_channel_gate_report(rows):
    items = {str(row["seed"]): row["channel_identity"] for row in rows if "channel_identity" in row}
    if not items:
        return {
            "channel_gate_argmax_mapping_by_seed": {},
            "channel_gate_identity_alignment_score": 0.0,
            "channel_gate_diagonal_mass": 0.0,
            "channel_gate_off_diagonal_mass": 0.0,
            "channel_gate_entropy": 0.0,
            "effective_channel_gate_by_seed": {},
        }
    scores = [item["alignment_score"] for item in items.values()]
    return {
        "channel_gate_argmax_mapping_by_seed": {seed: item["mapping"] for seed, item in items.items()},
        "channel_gate_identity_alignment_score": statistics.mean(scores),
        "channel_gate_identity_alignment_score_min": min(scores),
        "channel_gate_diagonal_mass": statistics.mean(item["diagonal_mass"] for item in items.values()),
        "channel_gate_off_diagonal_mass": statistics.mean(item["off_diagonal_mass"] for item in items.values()),
        "channel_gate_entropy": statistics.mean(item["entropy"] for item in items.values()),
        "effective_channel_gate_by_seed": {seed: item["effective_matrix"] for seed, item in items.items()},
    }


def build_cold_init_report(rows):
    audits = {str(row["seed"]): row["training_audit"] for row in rows if "training_audit" in row}
    if not audits:
        return {
            "cold_init_train_accuracy_mean": 0.0,
            "cold_init_rule_family_train_accuracy_mean": 0.0,
            "cold_init_solved_seed_count": 0,
            "cold_init_by_seed": {},
        }
    return {
        "cold_init_train_accuracy_mean": statistics.mean(item["cold_init_train_accuracy"] for item in audits.values()),
        "cold_init_rule_family_train_accuracy_mean": statistics.mean(item["cold_init_rule_family_train_accuracy"] for item in audits.values()),
        "cold_init_solved_seed_count": sum(int(item["cold_init_solved"]) for item in audits.values()),
        "cold_init_by_seed": {
            seed: {
                "cold_init_train_accuracy": item["cold_init_train_accuracy"],
                "cold_init_rule_family_train_accuracy": item["cold_init_rule_family_train_accuracy"],
                "cold_init_solved": item["cold_init_solved"],
            }
            for seed, item in audits.items()
        },
    }


def build_initial_equality_report(rows):
    audits = {str(row["seed"]): row["training_audit"]["initial_equality_identity"] for row in rows if "training_audit" in row}
    if not audits:
        return {
            "initial_equality_kernel_diagonal_mass_mean": 0.0,
            "initial_equality_kernel_off_diagonal_mass_mean": 0.0,
            "initial_equality_kernel_entropy_mean": 0.0,
            "initial_equality_kernel_argmax_mapping_by_seed": {},
        }
    return {
        "initial_equality_kernel_diagonal_mass_mean": statistics.mean(item["diagonal_mass"] for item in audits.values()),
        "initial_equality_kernel_off_diagonal_mass_mean": statistics.mean(item["off_diagonal_mass"] for item in audits.values()),
        "initial_equality_kernel_entropy_mean": statistics.mean(item["entropy"] for item in audits.values()),
        "initial_equality_kernel_argmax_mapping_by_seed": {seed: item["mapping"] for seed, item in audits.items()},
    }


def build_equality_intervention_report(rows, key):
    items = {str(row["seed"]): row[key] for row in rows if key in row}
    if not items:
        return {
            "test_accuracy": 0.0,
            "ood_accuracy": 0.0,
            "rule_family_test_accuracy": 0.0,
            "rule_family_ood_accuracy": 0.0,
            "by_seed": {},
        }
    return {
        "test_accuracy": statistics.mean(item["test_accuracy"] for item in items.values()),
        "ood_accuracy": statistics.mean(item["ood_accuracy"] for item in items.values()),
        "rule_family_test_accuracy": statistics.mean(item["rule_family_test_accuracy"] for item in items.values()),
        "rule_family_ood_accuracy": statistics.mean(item["rule_family_ood_accuracy"] for item in items.values()),
        "by_seed": items,
    }


def build_no_prebaked_audit(rows, cold_report, initial_report):
    audits = [row["training_audit"] for row in rows if "training_audit" in row]
    return {
        "cold_init_solved_seed_count": cold_report["cold_init_solved_seed_count"],
        "cold_init_already_solved": cold_report["cold_init_solved_seed_count"] > 0,
        "initial_equality_kernel_diagonal_mass_mean": initial_report["initial_equality_kernel_diagonal_mass_mean"],
        "initial_equality_kernel_identity_like": initial_report["initial_equality_kernel_diagonal_mass_mean"] >= 0.90,
        "prebaked_identity_initialization": any(item["prebaked_identity_initialization"] for item in audits),
        "training_accepts_by_fitness": all(item["training_accepts_by_fitness"] for item in audits) if audits else False,
        "verdict": "no_prebaked_equality_shortcut_detected"
        if audits
        and cold_report["cold_init_solved_seed_count"] == 0
        and initial_report["initial_equality_kernel_diagonal_mass_mean"] < 0.90
        and not any(item["prebaked_identity_initialization"] for item in audits)
        and all(item["training_accepts_by_fitness"] for item in audits)
        else "prebaked_or_cold_init_shortcut_risk",
    }


def build_mutation_report(rows):
    accepted = Counter({op: 0 for op in MUTATION_TYPES})
    rejected = Counter({op: 0 for op in MUTATION_TYPES})
    convergence = []
    generations = {}
    for row in rows:
        mutation = row.get("mutation")
        if not mutation:
            continue
        accepted.update(mutation["accepted_mutations_by_type"])
        rejected.update(mutation["rejected_mutations_by_type"])
        if mutation["convergence_generation"] is not None:
            convergence.append(mutation["convergence_generation"])
        generations[str(row["seed"])] = mutation["generations_executed"]
    return {
        "accepted_mutations_by_type": dict(accepted),
        "rejected_mutations_by_type": dict(rejected),
        "mutation_acceptance_rate": sum(accepted.values()) / max(1, sum(accepted.values()) + sum(rejected.values())),
        "convergence_generation_median": statistics.median(convergence) if convergence else None,
        "generations_executed_by_seed": generations,
    }


def d42_upstream_manifest():
    decision_path = Path("target/pilot_wave/d42_raw_support_evidence_extraction_prototype/smoke/decision.json")
    aggregate_path = Path("target/pilot_wave/d42_raw_support_evidence_extraction_prototype/smoke/aggregate_metrics.json")
    learned_path = Path("target/pilot_wave/d42_raw_support_evidence_extraction_prototype/smoke/mutable_learned_raw_support_evidence_extractor_report.json")
    composition_path = Path("target/pilot_wave/d42_raw_support_evidence_extraction_prototype/smoke/raw_support_plus_learned_router_composition_report.json")
    equality_path = Path("target/pilot_wave/d42_raw_support_evidence_extraction_prototype/smoke/equality_kernel_report.json")
    mutation_path = Path("target/pilot_wave/d42_raw_support_evidence_extraction_prototype/smoke/mutation_acceptance_report.json")
    contract = Path("docs/research/D42_RAW_SUPPORT_EVIDENCE_EXTRACTION_PROTOTYPE_CONTRACT.md")
    result = Path("docs/research/D42_RAW_SUPPORT_EVIDENCE_EXTRACTION_PROTOTYPE_RESULT.md")
    runner = Path("scripts/probes/run_d42_raw_support_evidence_extraction_prototype.py")
    checker = Path("scripts/probes/run_d42_raw_support_evidence_extraction_prototype_check.py")
    payload = {
        "d42_commit": "e8a95449435e0a4dcb6af87ac1c3a50e63875e74",
        "required_files": {
            contract.as_posix(): contract.exists(),
            result.as_posix(): result.exists(),
            runner.as_posix(): runner.exists(),
            checker.as_posix(): checker.exists(),
        },
        "artifact_paths": {
            "decision": decision_path.as_posix(),
            "aggregate_metrics": aggregate_path.as_posix(),
            "learned_report": learned_path.as_posix(),
            "composition_report": composition_path.as_posix(),
        },
        "read_summary": {
            "d41_result_doc_present": Path("docs/research/D41_RULE_FAMILY_INFERENCE_ROUTER_SCALE_CONFIRM_RESULT.md").exists(),
            "d42_result_doc_present": result.exists(),
            "d42_runner_present": runner.exists(),
            "d42_checker_present": checker.exists(),
        },
        "d42_decision": "raw_support_evidence_extraction_prototype_positive",
        "d42_verdict": "D42_RAW_SUPPORT_EVIDENCE_EXTRACTION_PROTOTYPE_POSITIVE",
        "d42_next": "D43_RAW_SUPPORT_EVIDENCE_EXTRACTION_SCALE_CONFIRM",
        "d42_learned_raw_extractor_selected_pocket_test_accuracy": 1.0,
        "d42_learned_raw_extractor_selected_pocket_ood_accuracy": 1.0,
        "d42_raw_support_plus_learned_router_test_accuracy": 1.0,
        "d42_raw_support_plus_learned_router_ood_accuracy": 1.0,
        "d42_controls_collapsed": True,
        "d42_wrong_support_control_pass": True,
        "d42_same_query_counterfactual_control_pass": True,
        "d42_caveat_mutation_acceptance_rate": 1.0,
        "d42_caveat_convergence_generation_median": 0,
        "d42_caveat_equality_kernel_near_identity": True,
    }
    if decision_path.exists():
        payload["artifact_decision"] = json.loads(decision_path.read_text())
    if learned_path.exists():
        learned = json.loads(learned_path.read_text())
        payload["artifact_learned_selected_pocket_test_accuracy"] = learned.get("learned_selected_pocket_test_accuracy")
        payload["artifact_learned_selected_pocket_ood_accuracy"] = learned.get("learned_selected_pocket_ood_accuracy")
        payload["artifact_rule_family_test_accuracy"] = learned.get("learned_rule_family_test_accuracy")
        payload["artifact_rule_family_ood_accuracy"] = learned.get("learned_rule_family_ood_accuracy")
        payload["artifact_min_support_count_accuracy"] = learned.get("min_support_count_accuracy")
        payload["artifact_min_margin_strata_accuracy"] = learned.get("min_margin_strata_accuracy")
    if composition_path.exists():
        composition = json.loads(composition_path.read_text())
        payload["artifact_raw_support_plus_learned_router_test_accuracy"] = composition.get("test_accuracy")
        payload["artifact_raw_support_plus_learned_router_ood_accuracy"] = composition.get("ood_accuracy")
    if equality_path.exists():
        equality = json.loads(equality_path.read_text())
        payload["artifact_equality_kernel_diagonal_mass"] = equality.get("equality_kernel_diagonal_mass")
        payload["artifact_equality_kernel_entropy"] = equality.get("equality_kernel_entropy")
    if mutation_path.exists():
        mutation = json.loads(mutation_path.read_text())
        payload["artifact_mutation_acceptance_rate"] = mutation.get("mutation_acceptance_rate")
        payload["artifact_convergence_generation_median"] = mutation.get("convergence_generation_median")
    if aggregate_path.exists():
        aggregate = json.loads(aggregate_path.read_text())
        payload["artifact_wrong_support_follow_rate"] = aggregate.get("wrong_support_behavior", {}).get("wrong_support_follow_rate_test")
        payload["artifact_wrong_support_selected_pocket_accuracy"] = aggregate.get("wrong_support_behavior", {}).get("wrong_support_selected_pocket_test_accuracy")
        payload["artifact_same_query_different_raw_support_accuracy"] = aggregate.get("same_query_different_raw_support_accuracy")
    return payload


def compact_arm_summary(report):
    return {
        "train_accuracy": report["train_accuracy"],
        "test_accuracy": report["test_accuracy"],
        "ood_accuracy": report["ood_accuracy"],
        "rule_family_train_accuracy": report["rule_family_train_accuracy"],
        "rule_family_test_accuracy": report["rule_family_test_accuracy"],
        "rule_family_ood_accuracy": report["rule_family_ood_accuracy"],
        "min_seed_test_accuracy": report["min_seed_test_accuracy"],
        "min_seed_ood_accuracy": report["min_seed_ood_accuracy"],
        "per_support_count_accuracy": report["per_support_count_accuracy"],
        "per_margin_strata_accuracy": report["per_margin_strata_accuracy"],
        "per_family_accuracy": report["per_family_accuracy"],
        "failed_seed_count": report["failed_seed_count"],
    }


def make_report(decision_payload, arm_reports, deltas, artifact_path):
    lines = [
        "# D43 Raw Support Evidence Extraction Scale Confirm",
        "",
        f"artifact_path = `{artifact_path}`",
        f"decision = `{decision_payload['decision']}`",
        f"verdict = `{decision_payload['verdict']}`",
        f"next = `{decision_payload['next']}`",
        "",
        "| Arm | train | test | ood |",
        "| --- | ---: | ---: | ---: |",
    ]
    for arm in ARMS:
        report = arm_reports[arm]
        lines.append(f"| {arm} | {report['train_accuracy']:.4f} | {report['test_accuracy']:.4f} | {report['ood_accuracy']:.4f} |")
    lines.extend(
        [
            "",
            "## Deltas",
            "",
            f"- learned_vs_query_only_test_delta = {deltas['learned_vs_query_only_test_delta']:.4f}",
            f"- learned_vs_shuffled_center_test_delta = {deltas['learned_vs_shuffled_center_test_delta']:.4f}",
            f"- learned_vs_no_center_test_delta = {deltas['learned_vs_no_center_test_delta']:.4f}",
            "",
            "## Boundary",
            "",
            "A positive D43 proves only that a learned raw symbolic support-evidence extractor can feed the previously confirmed support-rule-router stack on a controlled symbolic task, with fixed formula primitive candidates available. It does not prove raw visual Raven reasoning, formula primitive discovery, full hidden-rule Raven solving, natural-language reasoning, DNA/genome success, Raven solved, architecture superiority, consciousness, or general intelligence.",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    started = time.time()
    out = Path(args.out)
    if "target/pilot_wave/d43_raw_support_evidence_extraction_scale_confirm/" not in out.as_posix():
        raise SystemExit("--out must stay under target/pilot_wave/d43_raw_support_evidence_extraction_scale_confirm/")
    out.mkdir(parents=True, exist_ok=True)
    progress_path = out / "progress.jsonl"
    if progress_path.exists():
        progress_path.unlink()
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    support_counts = [int(item) for item in args.support_counts.split(",") if item.strip()]
    jobs = [{"seed": seed, "arm": arm} for seed in seeds for arm in ARMS]
    config = {
        "support_counts": support_counts,
        "train_rows_per_seed": args.train_rows_per_seed,
        "test_rows_per_seed": args.test_rows_per_seed,
        "ood_rows_per_seed": args.ood_rows_per_seed,
        "generations": args.generations,
        "population": args.population,
    }
    worker_limit = os.cpu_count() or 1
    if args.workers != "auto":
        worker_limit = int(args.workers)
    elif args.cpu_target == "balanced":
        worker_limit = max(1, worker_limit // 2)
    workers = min(worker_limit, len(jobs))
    write_json(out / "queue.json", {"jobs": jobs, "seeds": seeds, "arms": ARMS, "support_counts": support_counts, "workers": workers, "cpu_target": args.cpu_target, "heartbeat_sec": args.heartbeat_sec})

    all_rows = []
    test_rows = []
    ood_rows = []
    cf_rows = []
    for seed in seeds:
        for split, count in [("train", args.train_rows_per_seed), ("test", args.test_rows_per_seed), ("ood", args.ood_rows_per_seed)]:
            rows = make_rows(seed, split, count, support_counts)
            all_rows.extend(rows)
            if split == "test":
                test_rows.extend(rows)
                cf_rows.extend(make_counterfactual_rows(seed, split, count, support_counts))
            if split == "ood":
                ood_rows.extend(rows)
                cf_rows.extend(make_counterfactual_rows(seed, split, count, support_counts))
    invariants = dataset_invariants(all_rows, cf_rows)
    write_json(out / "dataset_invariant_report.json", invariants)
    support_test_selected, support_test_rule = rule_oracle_accuracy(test_rows)
    support_ood_selected, support_ood_rule = rule_oracle_accuracy(ood_rows)
    support_audit = {
        "ambiguous_support_rate": invariants["ambiguous_support_rate"],
        "multi_family_support_tie_rate": invariants["multi_family_support_tie_rate"],
        "intended_family_unique_evidence_rate": invariants["intended_family_unique_evidence_rate"],
        "support_evidence_oracle_test_accuracy": support_test_rule,
        "support_evidence_oracle_ood_accuracy": support_ood_rule,
        "support_selected_pocket_oracle_test_accuracy": support_test_selected,
        "support_selected_pocket_oracle_ood_accuracy": support_ood_selected,
    }
    write_json(out / "support_evidence_oracle_audit.json", support_audit)
    ood_audit = {
        "support_evidence_oracle_test_accuracy": support_test_rule,
        "support_evidence_oracle_ood_accuracy": support_ood_rule,
        "support_selected_pocket_oracle_test_accuracy": support_test_selected,
        "support_selected_pocket_oracle_ood_accuracy": support_ood_selected,
        "known_rule_oracle_test_accuracy": true_family_oracle_accuracy(test_rows),
        "known_rule_oracle_ood_accuracy": true_family_oracle_accuracy(ood_rows),
        "ood_label_rule_changed": False,
    }
    write_json(out / "ood_rule_invariance_audit.json", ood_audit)
    write_json(
        out / "dataset_manifest.json",
        {
            "families": FAMILIES,
            "formulas": FORMULA_DESC,
            "support_counts": support_counts,
            "support_strata": ["clean_unanimous", "noisy_majority"],
            "margin_strata": ["margin_low", "margin_high"],
            "noisy_majority_unavailable": False,
            "seeds": seeds,
            "train_rows_per_seed": args.train_rows_per_seed,
            "test_rows_per_seed": args.test_rows_per_seed,
            "ood_rows_per_seed": args.ood_rows_per_seed,
            "query_formula_values_required_distinct": True,
            "support_rows_required_unambiguous": True,
            "tier_a_fixed_formula_candidate_primitives": True,
            "formula_primitive_discovery_unavailable": True,
            "ood_semantics": "held-out symbolic distributions with support centers and query targets recomputed by the same formulas",
        },
    )
    write_json(out / "d42_upstream_manifest.json", d42_upstream_manifest())

    append_jsonl(progress_path, {"completed_jobs": 0, "failed_jobs": 0, "total_jobs": len(jobs), "elapsed_sec": 0.0})
    results = []
    failures = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_job, job, config, out.as_posix()): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                failure = {**job, "ok": False, "error": str(exc)}
                failures.append(failure)
                write_json(out / f"arm_{job['arm']}" / f"seed_{job['seed']}" / "error.json", failure)
            append_jsonl(progress_path, {"completed_jobs": len(results), "failed_jobs": len(failures), "total_jobs": len(jobs), "elapsed_sec": time.time() - started})

    arm_reports = {}
    for arm in ARMS:
        arm_rows = [row for row in results if row["arm"] == arm]
        failed_seed_count = sum(1 for failure in failures if failure["arm"] == arm)
        arm_reports[arm] = aggregate_arm(arm_rows, failed_seed_count)
        write_json(out / REPORT_FILES[arm], arm_reports[arm])

    learned_rows = [row for row in results if row["arm"] == "MUTABLE_LEARNED_RAW_SUPPORT_EVIDENCE_EXTRACTOR"]
    equality_report = build_equality_kernel_report(learned_rows)
    channel_report = build_channel_gate_report(learned_rows)
    mutation_report = build_mutation_report(learned_rows)
    cold_init_report = build_cold_init_report(learned_rows)
    initial_equality_report = build_initial_equality_report(learned_rows)
    equality_ablation_report = build_equality_intervention_report(learned_rows, "equality_kernel_ablation")
    equality_shuffle_report = build_equality_intervention_report(learned_rows, "equality_kernel_shuffle")
    no_prebaked_audit = build_no_prebaked_audit(learned_rows, cold_init_report, initial_equality_report)
    learned_input_audit = {
        "learned_extractor_receives_raw_support_boards": True,
        "learned_extractor_receives_formula_candidate_values": True,
        "learned_extractor_receives_precomputed_support_evidence": False,
        "learned_extractor_receives_boolean_equality": False,
        "learned_extractor_receives_expected_selected": False,
        "learned_extractor_receives_true_family": False,
        "learned_extractor_receives_query_target": False,
        "tier_a_fixed_formula_candidate_primitives": True,
        "formula_primitive_discovery_unavailable": True,
    }
    seed_variance = {
        "test_accuracy_variance": variance([row["test_accuracy"] for row in learned_rows]),
        "ood_accuracy_variance": variance([row["ood_accuracy"] for row in learned_rows]),
        "rule_family_test_accuracy_variance": variance([row["rule_family_test_accuracy"] for row in learned_rows if row["rule_family_test_accuracy"] is not None]),
        "rule_family_ood_accuracy_variance": variance([row["rule_family_ood_accuracy"] for row in learned_rows if row["rule_family_ood_accuracy"] is not None]),
    }
    learned_report = dict(arm_reports["MUTABLE_LEARNED_RAW_SUPPORT_EVIDENCE_EXTRACTOR"])
    learned_report.update(
        {
            "learned_raw_extractor_rule_family_train_accuracy": learned_report["rule_family_train_accuracy"],
            "learned_raw_extractor_rule_family_test_accuracy": learned_report["rule_family_test_accuracy"],
            "learned_raw_extractor_rule_family_ood_accuracy": learned_report["rule_family_ood_accuracy"],
            "learned_raw_extractor_selected_pocket_train_accuracy": learned_report["train_accuracy"],
            "learned_raw_extractor_selected_pocket_test_accuracy": learned_report["test_accuracy"],
            "learned_raw_extractor_selected_pocket_ood_accuracy": learned_report["ood_accuracy"],
            "min_seed_learned_raw_extractor_test_accuracy": learned_report["min_seed_test_accuracy"],
            "min_seed_learned_raw_extractor_ood_accuracy": learned_report["min_seed_ood_accuracy"],
            "min_support_count_accuracy": min(learned_report["per_support_count_accuracy"].values()),
            "min_margin_strata_accuracy": min(learned_report["per_margin_strata_accuracy"].values()),
            "equality_kernel_diagonal_mass": equality_report["equality_kernel_diagonal_mass"],
            "equality_kernel_off_diagonal_mass": equality_report["equality_kernel_off_diagonal_mass"],
            "equality_kernel_argmax_mapping": equality_report["equality_kernel_argmax_mapping"],
            "equality_kernel_entropy": equality_report["equality_kernel_entropy"],
            "channel_gate_identity_alignment_score": channel_report["channel_gate_identity_alignment_score"],
            "channel_gate_diagonal_mass": channel_report["channel_gate_diagonal_mass"],
            "channel_gate_off_diagonal_mass": channel_report["channel_gate_off_diagonal_mass"],
            "channel_gate_argmax_mapping_by_seed": channel_report["channel_gate_argmax_mapping_by_seed"],
            "accepted_mutations_by_type": mutation_report["accepted_mutations_by_type"],
            "rejected_mutations_by_type": mutation_report["rejected_mutations_by_type"],
            "mutation_acceptance_rate": mutation_report["mutation_acceptance_rate"],
            "convergence_generation_median": mutation_report["convergence_generation_median"],
            "seed_variance": seed_variance,
            "cold_init_train_accuracy_mean": cold_init_report["cold_init_train_accuracy_mean"],
            "cold_init_rule_family_train_accuracy_mean": cold_init_report["cold_init_rule_family_train_accuracy_mean"],
            "cold_init_solved_seed_count": cold_init_report["cold_init_solved_seed_count"],
            "initial_equality_kernel_diagonal_mass_mean": initial_equality_report["initial_equality_kernel_diagonal_mass_mean"],
            "equality_kernel_ablation_test_accuracy": equality_ablation_report["test_accuracy"],
            "equality_kernel_ablation_ood_accuracy": equality_ablation_report["ood_accuracy"],
            "equality_kernel_shuffle_test_accuracy": equality_shuffle_report["test_accuracy"],
            "equality_kernel_shuffle_ood_accuracy": equality_shuffle_report["ood_accuracy"],
        }
    )
    write_json(out / "mutable_learned_raw_support_evidence_extractor_report.json", learned_report)
    extractor_matrix_report = {
        "equality_kernel": {key: value for key, value in equality_report.items() if key != "equality_kernel_matrix_by_seed"},
        "channel_gate": {key: value for key, value in channel_report.items() if key != "effective_channel_gate_by_seed"},
    }
    write_json(out / "raw_evidence_extractor_matrix_report.json", extractor_matrix_report)
    write_json(out / "equality_kernel_report.json", equality_report)
    write_json(out / "channel_gate_report.json", channel_report)
    write_json(out / "cold_init_accuracy_report.json", cold_init_report)
    write_json(out / "initial_equality_kernel_report.json", initial_equality_report)
    write_json(out / "equality_kernel_ablation_report.json", equality_ablation_report)
    write_json(out / "equality_kernel_shuffle_report.json", equality_shuffle_report)
    write_json(out / "no_prebaked_equality_audit.json", no_prebaked_audit)
    write_json(out / "learned_extractor_input_audit.json", learned_input_audit)
    mappings = list(channel_report["channel_gate_argmax_mapping_by_seed"].values())
    agreements = []
    for left_idx in range(len(mappings)):
        for right_idx in range(left_idx + 1, len(mappings)):
            agreements.append(sum(int(mappings[left_idx][family] == mappings[right_idx][family]) for family in FAMILIES) / 5)
    write_json(out / "raw_evidence_extractor_stability_report.json", {"pairwise_channel_mapping_agreement_mean": statistics.mean(agreements) if agreements else 0.0, "seed_count": len(mappings)})
    write_json(out / "mutation_acceptance_report.json", mutation_report)
    write_json(out / "convergence_report.json", {"convergence_generation_median": mutation_report["convergence_generation_median"], "generations_executed_by_seed": mutation_report["generations_executed_by_seed"]})
    seed_variance_report = {
        arm: {
            "test_accuracy_variance": variance([row["test_accuracy"] for row in results if row["arm"] == arm]),
            "ood_accuracy_variance": variance([row["ood_accuracy"] for row in results if row["arm"] == arm]),
        }
        for arm in ARMS
    }
    write_json(out / "seed_variance_report.json", seed_variance_report)
    write_json(out / "support_count_generalization_report.json", {arm: arm_reports[arm]["per_support_count_accuracy"] for arm in ARMS})
    write_json(out / "support_margin_strata_report.json", {arm: arm_reports[arm]["per_margin_strata_accuracy"] for arm in ARMS})
    write_json(
        out / "noisy_support_report.json",
        {
            "noisy_majority_unavailable": False,
            "learned_noisy_majority_accuracy": learned_report["per_margin_strata_accuracy"]["noisy_majority"],
            "learned_clean_unanimous_accuracy": learned_report["per_margin_strata_accuracy"]["clean_unanimous"],
        },
    )
    write_json(
        out / "per_seed_report.json",
        {
            "seeds": seeds,
            "attempted_jobs": jobs,
            "completed_jobs": [
                {
                    "seed": row["seed"],
                    "arm": row["arm"],
                    "train_accuracy": row["train_accuracy"],
                    "test_accuracy": row["test_accuracy"],
                    "ood_accuracy": row["ood_accuracy"],
                    "rule_family_test_accuracy": row["rule_family_test_accuracy"],
                    "rule_family_ood_accuracy": row["rule_family_ood_accuracy"],
                }
                for row in results
            ],
            "failed_jobs": failures,
        },
    )
    write_json(out / "per_family_report.json", {arm: arm_reports[arm]["per_family_accuracy"] for arm in ARMS})
    write_json(out / "pocket_confusion_matrix.json", {arm: arm_reports[arm]["pocket_confusion_matrix"] for arm in ARMS})
    write_json(out / "family_confusion_matrix.json", {arm: arm_reports[arm]["family_confusion_matrix"] for arm in ARMS})
    write_json(out / "score_margin_report.json", {arm: {"median_score_margin": arm_reports[arm]["median_score_margin"], "low_margin_error_rate": arm_reports[arm]["low_margin_error_rate"]} for arm in ARMS})

    wrong_rows = [row for row in results if row["arm"] == "WRONG_SUPPORT_CONTROL"]
    wrong_behavior = {
        "wrong_support_follow_rate_test": statistics.mean(row.get("wrong_support_follow_rate_test", 0.0) for row in wrong_rows) if wrong_rows else 0.0,
        "wrong_support_follow_rate_ood": statistics.mean(row.get("wrong_support_follow_rate_ood", 0.0) for row in wrong_rows) if wrong_rows else 0.0,
        "wrong_support_selected_pocket_test_accuracy": arm_reports["WRONG_SUPPORT_CONTROL"]["test_accuracy"],
        "wrong_support_selected_pocket_ood_accuracy": arm_reports["WRONG_SUPPORT_CONTROL"]["ood_accuracy"],
        "wrong_support_query_mismatch_rate_test": statistics.mean(row.get("wrong_support_query_mismatch_rate_test", 0.0) for row in wrong_rows) if wrong_rows else 0.0,
    }
    counterfactual_accuracy = arm_reports["SAME_QUERY_DIFFERENT_RAW_SUPPORT_COUNTERFACTUAL"]["test_accuracy"]
    deltas = {
        "learned_vs_query_only_test_delta": learned_report["test_accuracy"] - arm_reports["QUERY_ONLY_BASELINE"]["test_accuracy"],
        "learned_vs_shuffled_center_test_delta": learned_report["test_accuracy"] - arm_reports["SHUFFLED_CENTER_CONTROL"]["test_accuracy"],
        "learned_vs_no_center_test_delta": learned_report["test_accuracy"] - arm_reports["NO_CENTER_CONTROL"]["test_accuracy"],
        "learned_vs_shuffled_formula_candidate_test_delta": learned_report["test_accuracy"] - arm_reports["SHUFFLED_FORMULA_CANDIDATE_CONTROL"]["test_accuracy"],
        "learned_vs_no_formula_candidate_test_delta": learned_report["test_accuracy"] - arm_reports["NO_FORMULA_CANDIDATE_CONTROL"]["test_accuracy"],
    }
    write_json(out / "arm_comparison_report.json", {"deltas": deltas, "wrong_support_behavior": wrong_behavior, "same_query_different_raw_support_accuracy": counterfactual_accuracy})

    precomputed_oracle = arm_reports["PRECOMPUTED_SUPPORT_EVIDENCE_UPPER_BOUND"]
    raw_oracle = arm_reports["ORACLE_RAW_SUPPORT_EVIDENCE_EXTRACTOR"]
    if (
        invariants["duplicate_target_pocket_rate"] != 0.0
        or invariants["missing_target_pocket_rate"] != 0.0
        or invariants["expected_selected_points_to_target_rate"] != 1.0
        or invariants["ambiguous_support_rate"] != 0.0
        or invariants["multi_family_support_tie_rate"] != 0.0
        or invariants["intended_family_unique_evidence_rate"] != 1.0
        or invariants["counterfactual_target_collision_rate"] != 0.0
        or invariants["wrong_support_query_mismatch_rate"] != 1.0
    ):
        decision = "d43_dataset_invariant_failure"
        verdict = "D43_DATASET_INVARIANT_FAILURE"
        next_step = "D43B_DATASET_REPAIR"
    elif precomputed_oracle["test_accuracy"] < 0.99 or precomputed_oracle["ood_accuracy"] < 0.99:
        decision = "d43_support_evidence_oracle_failure"
        verdict = "D43_SUPPORT_EVIDENCE_ORACLE_FAILURE"
        next_step = "D43C_SUPPORT_EVIDENCE_ORACLE_REPAIR"
    elif (
        ood_audit["support_evidence_oracle_test_accuracy"] != 1.0
        or ood_audit["support_evidence_oracle_ood_accuracy"] != 1.0
        or ood_audit["known_rule_oracle_test_accuracy"] != 1.0
        or ood_audit["known_rule_oracle_ood_accuracy"] != 1.0
    ):
        decision = "d43_ood_rule_invariance_failure"
        verdict = "D43_OOD_RULE_INVARIANCE_FAILURE"
        next_step = "D43D_OOD_REPAIR"
    else:
        mean_scale_pass = (
            learned_report["test_accuracy"] >= 0.98
            and learned_report["ood_accuracy"] >= 0.98
            and learned_report["rule_family_test_accuracy"] >= 0.98
            and learned_report["rule_family_ood_accuracy"] >= 0.98
            and learned_report["min_seed_test_accuracy"] >= 0.95
            and learned_report["min_seed_ood_accuracy"] >= 0.95
        )
        strata_pass = learned_report["min_support_count_accuracy"] >= 0.95 and learned_report["min_margin_strata_accuracy"] >= 0.95
        anti_shortcut_pass = (
            no_prebaked_audit["verdict"] == "no_prebaked_equality_shortcut_detected"
            and equality_ablation_report["test_accuracy"] <= 0.35
            and equality_ablation_report["ood_accuracy"] <= 0.35
            and equality_shuffle_report["test_accuracy"] <= 0.35
            and equality_shuffle_report["ood_accuracy"] <= 0.35
        )
        control_pass = (
            precomputed_oracle["test_accuracy"] >= 0.99
            and precomputed_oracle["ood_accuracy"] >= 0.99
            and raw_oracle["test_accuracy"] >= 0.99
            and raw_oracle["ood_accuracy"] >= 0.99
            and deltas["learned_vs_query_only_test_delta"] >= 0.60
            and arm_reports["SHUFFLED_CENTER_CONTROL"]["test_accuracy"] <= 0.25
            and arm_reports["SHUFFLED_FORMULA_CANDIDATE_CONTROL"]["test_accuracy"] <= 0.25
            and arm_reports["NO_CENTER_CONTROL"]["test_accuracy"] <= 0.35
            and arm_reports["NO_FORMULA_CANDIDATE_CONTROL"]["test_accuracy"] <= 0.35
            and wrong_behavior["wrong_support_follow_rate_test"] >= 0.98
            and wrong_behavior["wrong_support_selected_pocket_test_accuracy"] <= 0.05
        )
        counterfactual_pass = counterfactual_accuracy >= 0.98
        if mean_scale_pass and strata_pass and control_pass and counterfactual_pass and anti_shortcut_pass:
            decision = "raw_support_evidence_extraction_scale_confirmed"
            verdict = "D43_RAW_SUPPORT_EVIDENCE_EXTRACTION_SCALE_CONFIRMED"
            next_step = "D44_FORMULA_PRIMITIVE_DISCOVERY_PLAN"
        elif mean_scale_pass and strata_pass and control_pass and counterfactual_pass and not anti_shortcut_pass:
            decision = "raw_support_evidence_scale_positive_antishortcut_audit_failed"
            verdict = "D43_ANTISHORTCUT_AUDIT_GAP"
            next_step = "D43A_ANTISHORTCUT_REPAIR"
        elif raw_oracle["test_accuracy"] >= 0.99 and raw_oracle["ood_accuracy"] >= 0.99 and not mean_scale_pass:
            decision = "oracle_raw_support_evidence_confirmed_learned_extractor_failed"
            verdict = "D43_ORACLE_RAW_SUPPORT_CONFIRMED_LEARNED_EXTRACTOR_FAILED"
            next_step = "D43L_RAW_EXTRACTOR_OPTIMIZATION_PLAN"
        elif mean_scale_pass and not strata_pass:
            decision = "raw_support_extraction_mean_positive_support_strata_gap"
            verdict = "D43_RAW_SUPPORT_STRATA_EDGE_GAP"
            next_step = "D43S_RAW_SUPPORT_STRATA_HARDENING_PLAN"
        elif deltas["learned_vs_query_only_test_delta"] > 0.0:
            decision = "raw_support_evidence_extraction_partial_signal"
            verdict = "D43_RAW_SUPPORT_EVIDENCE_EXTRACTION_PARTIAL_SIGNAL"
            next_step = "D43L_RAW_EXTRACTOR_OPTIMIZATION_PLAN"
        else:
            decision = "raw_support_evidence_extraction_not_confirmed"
            verdict = "D43_RAW_SUPPORT_EVIDENCE_EXTRACTION_NOT_CONFIRMED"
            next_step = "D43_FEATURE_SPACE_DIAGNOSTIC"
    decision_payload = {
        "decision": decision,
        "verdict": verdict,
        "next": next_step,
        "boundary": "controlled symbolic raw support-board evidence extraction with fixed formula primitive candidates plus the previously confirmed support-rule-router stack only",
        "non_claims": {
            "raw_visual_raven_reasoning": False,
            "full_hidden_rule_raven_solving": False,
            "formula_primitive_discovery": False,
            "natural_language_reasoning": False,
            "dna_genome_success": False,
            "raven_solved": False,
            "architecture_superiority": False,
            "consciousness": False,
            "general_intelligence": False,
        },
    }
    write_json(out / "decision.json", decision_payload)
    write_json(out / "summary.json", {"decision": decision, "verdict": verdict, "next": next_step})
    aggregate = {
        "arms": {arm: compact_arm_summary(arm_reports[arm]) for arm in ARMS},
        "deltas": deltas,
        "same_query_different_raw_support_accuracy": counterfactual_accuracy,
        "wrong_support_behavior": wrong_behavior,
        "learned_raw_extractor": {
            "learned_raw_extractor_rule_family_train_accuracy": learned_report["learned_raw_extractor_rule_family_train_accuracy"],
            "learned_raw_extractor_rule_family_test_accuracy": learned_report["learned_raw_extractor_rule_family_test_accuracy"],
            "learned_raw_extractor_rule_family_ood_accuracy": learned_report["learned_raw_extractor_rule_family_ood_accuracy"],
            "learned_raw_extractor_selected_pocket_train_accuracy": learned_report["learned_raw_extractor_selected_pocket_train_accuracy"],
            "learned_raw_extractor_selected_pocket_test_accuracy": learned_report["learned_raw_extractor_selected_pocket_test_accuracy"],
            "learned_raw_extractor_selected_pocket_ood_accuracy": learned_report["learned_raw_extractor_selected_pocket_ood_accuracy"],
            "min_seed_learned_raw_extractor_test_accuracy": learned_report["min_seed_learned_raw_extractor_test_accuracy"],
            "min_seed_learned_raw_extractor_ood_accuracy": learned_report["min_seed_learned_raw_extractor_ood_accuracy"],
            "min_support_count_accuracy": learned_report["min_support_count_accuracy"],
            "min_margin_strata_accuracy": learned_report["min_margin_strata_accuracy"],
            "seed_variance": seed_variance,
        },
        "equality_kernel": {key: value for key, value in equality_report.items() if key != "equality_kernel_matrix_by_seed"},
        "channel_gate": {key: value for key, value in channel_report.items() if key != "effective_channel_gate_by_seed"},
        "cold_init": cold_init_report,
        "initial_equality_kernel": initial_equality_report,
        "equality_kernel_ablation": equality_ablation_report,
        "equality_kernel_shuffle": equality_shuffle_report,
        "no_prebaked_equality_audit": no_prebaked_audit,
        "learned_extractor_input_audit": learned_input_audit,
        "mutation": mutation_report,
        "tier": {
            "tier_a_fixed_formula_candidate_primitives": True,
            "formula_primitive_discovery_unavailable": True,
        },
        "failed_jobs": failures,
        "decision": decision_payload,
    }
    write_json(out / "aggregate_metrics.json", aggregate)
    write_json(
        out / "machine_utilization_report.json",
        {
            "os_cpu_count": os.cpu_count(),
            "worker_count": workers,
            "thread_env": {
                "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
                "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            },
            "wall_clock_sec": time.time() - started,
            "completed_jobs": len(results),
            "failed_jobs": len(failures),
        },
    )
    (out / "report.md").write_text(make_report(decision_payload, arm_reports, deltas, out.as_posix()))
    print(json.dumps({"status": "ok", "decision": decision, "verdict": verdict, "next": next_step, "failed_jobs": len(failures)}, indent=2))


def d43s_upstream_manifest():
    decision_path = Path("target/pilot_wave/d43_raw_support_evidence_extraction_scale_confirm/smoke/decision.json")
    aggregate_path = Path("target/pilot_wave/d43_raw_support_evidence_extraction_scale_confirm/smoke/aggregate_metrics.json")
    result = Path("docs/research/D43_RAW_SUPPORT_EVIDENCE_EXTRACTION_SCALE_CONFIRM_RESULT.md")
    runner = Path("scripts/probes/run_d43_raw_support_evidence_extraction_scale_confirm.py")
    payload = {
        "d43_decision": "raw_support_evidence_extraction_scale_confirmed",
        "d43_verdict": "D43_RAW_SUPPORT_EVIDENCE_EXTRACTION_SCALE_CONFIRMED",
        "d43_next": "D44_FORMULA_PRIMITIVE_DISCOVERY_PLAN",
        "d43_result_doc_present": result.exists(),
        "d43_runner_present": runner.exists(),
        "d43_decision_artifact_present": decision_path.exists(),
        "d43_aggregate_artifact_present": aggregate_path.exists(),
    }
    if decision_path.exists():
        payload["artifact_decision"] = json.loads(decision_path.read_text())
    if aggregate_path.exists():
        aggregate = json.loads(aggregate_path.read_text())
        learned = aggregate.get("learned_raw_extractor", {})
        payload["d43_learned_test_accuracy"] = learned.get("learned_raw_extractor_selected_pocket_test_accuracy")
        payload["d43_learned_ood_accuracy"] = learned.get("learned_raw_extractor_selected_pocket_ood_accuracy")
        payload["d43_cold_init_train_accuracy_mean"] = aggregate.get("cold_init", {}).get("cold_init_train_accuracy_mean")
        payload["d43_equality_ablation_test_accuracy"] = aggregate.get("equality_kernel_ablation", {}).get("test_accuracy")
        payload["d43_equality_shuffle_test_accuracy"] = aggregate.get("equality_kernel_shuffle", {}).get("test_accuracy")
    return payload


def d43_failure_tail_manifest():
    return {
        "source": "D43 row-output failure analysis",
        "tail_pattern": {
            "support_stratum": "noisy_majority",
            "margin_stratum": "margin_low",
            "support_counts": [3, 5],
            "dominant_confusions": ["pair->row", "pair->col", "diag->pair", "diag->mirror", "row->mirror"],
            "diagnosis": "learned soft equality/channel scores sometimes compress small support-evidence gaps",
        },
        "d43_main_test_errors": 4,
        "d43_main_ood_errors": 7,
        "d43_baseline_tail_accuracy_context": {
            "test_accuracy": 0.9995833333333334,
            "ood_accuracy": 0.9992708333333333,
        },
    }


def collect_tail_errors(out, arm):
    by_split = {}
    for split in ["test", "ood"]:
        errors = []
        for path in sorted((out / f"arm_{arm}").glob(f"seed_*/row_outputs_{split}.jsonl")):
            seed = path.parent.name.split("_")[1]
            for line in path.read_text().splitlines():
                row = json.loads(line)
                if row["pred"] == row["truth"]:
                    continue
                if "noisy_majority_margin_low" not in row.get("d43s_strata", []):
                    continue
                row["seed"] = seed
                errors.append(row)
        by_split[split] = {
            "error_count": len(errors),
            "by_seed": dict(Counter(row["seed"] for row in errors)),
            "by_family": dict(Counter(row["family"] for row in errors)),
            "by_support_count": dict(Counter(str(row["support_count"]) for row in errors)),
            "family_to_inferred": dict(Counter(f"{row['family']}->{row['inferred_family']}" for row in errors)),
            "rows": errors[:40],
        }
    return by_split


def d43s_report(decision_payload, arm_reports, comparison, artifact_path):
    lines = [
        "# D43S Low-Margin Noisy Support Sharpening",
        "",
        f"artifact_path = `{artifact_path}`",
        f"decision = `{decision_payload['decision']}`",
        f"verdict = `{decision_payload['verdict']}`",
        f"next = `{decision_payload['next']}`",
        "",
        "| Arm | train | test | OOD | noisy-low | clean |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in ARMS:
        report = arm_reports[arm]
        lines.append(
            f"| {arm} | {report['train_accuracy']:.6f} | {report['test_accuracy']:.6f} | {report['ood_accuracy']:.6f} | "
            f"{report['noisy_majority_margin_low_accuracy']:.6f} | {report['clean_unanimous_accuracy']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Tail Comparison",
            "",
            f"- baseline_noisy_low_accuracy = {comparison['baseline_noisy_majority_margin_low_accuracy']:.6f}",
            f"- combined_noisy_low_accuracy = {comparison['combined_noisy_majority_margin_low_accuracy']:.6f}",
            f"- noisy_low_delta = {comparison['noisy_low_delta']:.6f}",
            f"- easy_case_regression = {comparison['easy_case_regression']:.6f}",
            "",
            "## Boundary",
            "",
            "D43S only tests low-margin noisy-support tail hardening for D43 on a controlled symbolic task with fixed formula primitive candidates. It does not prove formula primitive discovery, raw visual Raven reasoning, Raven solved, DNA/genome success, consciousness, AGI, or architecture superiority.",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    started = time.time()
    out = Path(args.out)
    if "target/pilot_wave/d43s_low_margin_noisy_support_sharpening/" not in out.as_posix():
        raise SystemExit("--out must stay under target/pilot_wave/d43s_low_margin_noisy_support_sharpening/")
    out.mkdir(parents=True, exist_ok=True)
    progress_path = out / "progress.jsonl"
    if progress_path.exists():
        progress_path.unlink()
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    support_counts = [int(item) for item in args.support_counts.split(",") if item.strip()]
    if support_counts != [3, 5]:
        raise SystemExit("D43S requires --support-counts 3,5")
    jobs = [{"seed": seed, "arm": arm} for seed in seeds for arm in ARMS]
    config = {
        "support_counts": support_counts,
        "train_rows_per_seed": args.train_rows_per_seed,
        "test_rows_per_seed": args.test_rows_per_seed,
        "ood_rows_per_seed": args.ood_rows_per_seed,
        "generations": args.generations,
        "population": args.population,
    }
    worker_limit = os.cpu_count() or 1
    if args.workers != "auto":
        worker_limit = int(args.workers)
    elif args.cpu_target == "balanced":
        worker_limit = max(1, worker_limit // 2)
    workers = min(worker_limit, len(jobs))
    write_json(out / "queue.json", {"jobs": jobs, "seeds": seeds, "arms": ARMS, "support_counts": support_counts, "workers": workers, "cpu_target": args.cpu_target, "heartbeat_sec": args.heartbeat_sec})

    all_rows = []
    test_rows = []
    ood_rows = []
    cf_rows = []
    for seed in seeds:
        for split, count in [("train", args.train_rows_per_seed), ("test", args.test_rows_per_seed), ("ood", args.ood_rows_per_seed)]:
            rows = make_rows(seed, split, count, support_counts)
            all_rows.extend(rows)
            if split == "test":
                test_rows.extend(rows)
                cf_rows.extend(make_counterfactual_rows(seed, split, count, support_counts))
            if split == "ood":
                ood_rows.extend(rows)
                cf_rows.extend(make_counterfactual_rows(seed, split, count, support_counts))
    invariants = dataset_invariants(all_rows, cf_rows)
    write_json(out / "dataset_invariant_report.json", invariants)
    support_test_selected, support_test_rule = rule_oracle_accuracy(test_rows)
    support_ood_selected, support_ood_rule = rule_oracle_accuracy(ood_rows)
    ood_audit = {
        "support_evidence_oracle_test_accuracy": support_test_rule,
        "support_evidence_oracle_ood_accuracy": support_ood_rule,
        "support_selected_pocket_oracle_test_accuracy": support_test_selected,
        "support_selected_pocket_oracle_ood_accuracy": support_ood_selected,
        "known_rule_oracle_test_accuracy": true_family_oracle_accuracy(test_rows),
        "known_rule_oracle_ood_accuracy": true_family_oracle_accuracy(ood_rows),
        "ood_label_rule_changed": False,
    }
    write_json(out / "ood_rule_invariance_audit.json", ood_audit)
    strata_counts = Counter(stratum for row in all_rows for stratum in d43s_strata(row))
    noisy_low_rate = strata_counts["noisy_majority_margin_low"] / max(1, len(all_rows))
    write_json(
        out / "dataset_manifest.json",
        {
            "families": FAMILIES,
            "formulas": FORMULA_DESC,
            "support_counts": support_counts,
            "required_strata": MARGIN_STRATA,
            "noisy_majority_margin_low_rate": noisy_low_rate,
            "family_counts": dict(Counter(row["intended_family"] for row in all_rows)),
            "strata_counts": dict(strata_counts),
            "tier_a_fixed_formula_candidate_primitives": True,
            "formula_primitive_discovery_unavailable": True,
            "learned_arms_receive_precomputed_support_evidence": False,
            "learned_arms_receive_boolean_equality": False,
        },
    )
    write_json(out / "d43_upstream_manifest.json", d43s_upstream_manifest())
    write_json(out / "d43_failure_tail_manifest.json", d43_failure_tail_manifest())

    append_jsonl(progress_path, {"completed_jobs": 0, "failed_jobs": 0, "total_jobs": len(jobs), "elapsed_sec": 0.0})
    results = []
    failures = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_job, job, config, out.as_posix()): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                failure = {**job, "ok": False, "error": str(exc)}
                failures.append(failure)
                write_json(out / f"arm_{job['arm']}" / f"seed_{job['seed']}" / "error.json", failure)
            append_jsonl(progress_path, {"completed_jobs": len(results), "failed_jobs": len(failures), "total_jobs": len(jobs), "elapsed_sec": time.time() - started})

    arm_reports = {}
    for arm in ARMS:
        rows = [row for row in results if row["arm"] == arm]
        arm_reports[arm] = aggregate_arm(rows, sum(1 for failure in failures if failure["arm"] == arm))
    temp_rows = [row for row in results if row["arm"] == "TEMPERATURE_SHARPENED_EVIDENCE"]
    arm_reports["TEMPERATURE_SHARPENED_EVIDENCE"]["temperature_sweep_by_seed"] = {
        str(row["seed"]): row.get("temperature_sweep", []) for row in temp_rows
    }
    for arm in ARMS:
        write_json(out / REPORT_FILES[arm], arm_reports[arm])

    combined_rows = [row for row in results if row["arm"] == "COMBINED_SHARPENED_MODEL"]
    equality_report = build_equality_kernel_report(combined_rows)
    channel_report = build_channel_gate_report(combined_rows)
    mutation_report = build_mutation_report(combined_rows)
    seed_variance_report = {
        arm: {
            "test_accuracy_variance": variance([row["test_accuracy"] for row in results if row["arm"] == arm]),
            "ood_accuracy_variance": variance([row["ood_accuracy"] for row in results if row["arm"] == arm]),
        }
        for arm in ARMS
    }
    write_json(out / "equality_kernel_report.json", equality_report)
    write_json(out / "channel_gate_report.json", channel_report)
    write_json(out / "mutation_acceptance_report.json", mutation_report)
    write_json(out / "convergence_report.json", {"convergence_generation_median": mutation_report["convergence_generation_median"], "generations_executed_by_seed": mutation_report["generations_executed_by_seed"]})
    write_json(out / "seed_variance_report.json", seed_variance_report)

    baseline = arm_reports["D43_BASELINE_REPLAY"]
    combined = arm_reports["COMBINED_SHARPENED_MODEL"]
    hard_vote = arm_reports["HARD_VOTE_ORACLE_UPPER_BOUND"]
    wrong_rows = [row for row in results if row["arm"] == "WRONG_SUPPORT_CONTROL"]
    wrong_behavior = {
        "wrong_support_follow_rate_test": statistics.mean(row.get("wrong_support_follow_rate_test", 0.0) for row in wrong_rows) if wrong_rows else 0.0,
        "wrong_support_follow_rate_ood": statistics.mean(row.get("wrong_support_follow_rate_ood", 0.0) for row in wrong_rows) if wrong_rows else 0.0,
        "wrong_support_selected_pocket_test_accuracy": arm_reports["WRONG_SUPPORT_CONTROL"]["test_accuracy"],
        "wrong_support_selected_pocket_ood_accuracy": arm_reports["WRONG_SUPPORT_CONTROL"]["ood_accuracy"],
    }
    comparison = {
        "baseline_noisy_majority_margin_low_accuracy": baseline["noisy_majority_margin_low_accuracy"],
        "combined_noisy_majority_margin_low_accuracy": combined["noisy_majority_margin_low_accuracy"],
        "noisy_low_delta": combined["noisy_majority_margin_low_accuracy"] - baseline["noisy_majority_margin_low_accuracy"],
        "easy_case_regression": max(
            0.0,
            baseline["clean_unanimous_accuracy"] - combined["clean_unanimous_accuracy"],
            baseline["noisy_majority_margin_high_accuracy"] - combined["noisy_majority_margin_high_accuracy"],
        ),
        "controls": {
            "shuffled_center_test_accuracy": arm_reports["SHUFFLED_CENTER_CONTROL"]["test_accuracy"],
            "shuffled_formula_candidate_test_accuracy": arm_reports["SHUFFLED_FORMULA_CANDIDATE_CONTROL"]["test_accuracy"],
            "no_center_test_accuracy": arm_reports["NO_CENTER_CONTROL"]["test_accuracy"],
            "same_query_different_raw_support_accuracy": arm_reports["SAME_QUERY_DIFFERENT_RAW_SUPPORT_COUNTERFACTUAL"]["test_accuracy"],
            **wrong_behavior,
        },
    }
    low_margin_tail_error_report = {
        "baseline_replay": collect_tail_errors(out, "D43_BASELINE_REPLAY"),
        "combined_sharpened": collect_tail_errors(out, "COMBINED_SHARPENED_MODEL"),
        "comparison": comparison,
    }
    write_json(out / "low_margin_tail_error_report.json", low_margin_tail_error_report)
    write_json(out / "family_confusion_tail_report.json", {"baseline": baseline["family_confusion_matrix"], "combined": combined["family_confusion_matrix"]})
    write_json(
        out / "support_count_tail_report.json",
        {
            "baseline": {
                "support_count_3_low_margin_accuracy": baseline["support_count_3_low_margin_accuracy"],
                "support_count_5_low_margin_accuracy": baseline["support_count_5_low_margin_accuracy"],
            },
            "combined": {
                "support_count_3_low_margin_accuracy": combined["support_count_3_low_margin_accuracy"],
                "support_count_5_low_margin_accuracy": combined["support_count_5_low_margin_accuracy"],
            },
        },
    )
    write_json(
        out / "evidence_gap_preservation_report.json",
        {
            "baseline_median_evidence_gap": baseline["median_evidence_gap"],
            "combined_median_evidence_gap": combined["median_evidence_gap"],
            "baseline_low_margin_error_rate": baseline["low_margin_error_rate"],
            "combined_low_margin_error_rate": combined["low_margin_error_rate"],
            "temperature_sweep_by_seed": arm_reports["TEMPERATURE_SHARPENED_EVIDENCE"].get("temperature_sweep_by_seed", {}),
        },
    )

    write_json(out / "per_seed_report.json", {"seeds": seeds, "attempted_jobs": jobs, "completed_jobs": results, "failed_jobs": failures})
    write_json(out / "aggregate_metrics.json", {
        "arms": {
            arm: compact_arm_summary(arm_reports[arm]) | {f"{key}_accuracy": arm_reports[arm][f"{key}_accuracy"] for key in MARGIN_STRATA}
            for arm in ARMS
        },
        "comparison": comparison,
        "low_margin_tail_errors": low_margin_tail_error_report,
        "equality_kernel": {key: value for key, value in equality_report.items() if key != "equality_kernel_matrix_by_seed"},
        "channel_gate": {key: value for key, value in channel_report.items() if key != "effective_channel_gate_by_seed"},
        "mutation": mutation_report,
        "failed_jobs": failures,
    })

    controls_collapse = (
        comparison["controls"]["shuffled_center_test_accuracy"] <= 0.25
        and comparison["controls"]["shuffled_formula_candidate_test_accuracy"] <= 0.25
        and comparison["controls"]["no_center_test_accuracy"] <= 0.35
        and comparison["controls"]["same_query_different_raw_support_accuracy"] >= 0.98
        and comparison["controls"]["wrong_support_follow_rate_test"] >= 0.98
        and comparison["controls"]["wrong_support_selected_pocket_test_accuracy"] <= 0.05
    )
    invariant_fail = any([
        invariants["duplicate_target_pocket_rate"] != 0.0,
        invariants["missing_target_pocket_rate"] != 0.0,
        invariants["expected_selected_points_to_target_rate"] != 1.0,
        invariants["ambiguous_support_rate"] != 0.0,
        invariants["multi_family_support_tie_rate"] != 0.0,
        invariants["intended_family_unique_evidence_rate"] != 1.0,
        invariants["counterfactual_target_collision_rate"] != 0.0,
        invariants["wrong_support_query_mismatch_rate"] != 1.0,
        ood_audit["support_evidence_oracle_test_accuracy"] != 1.0,
        ood_audit["support_evidence_oracle_ood_accuracy"] != 1.0,
        ood_audit["known_rule_oracle_test_accuracy"] != 1.0,
        ood_audit["known_rule_oracle_ood_accuracy"] != 1.0,
        ood_audit["ood_label_rule_changed"] is not False,
    ])
    if invariant_fail:
        decision = "d43s_dataset_or_ood_failure"
        verdict = "D43S_DATASET_OR_OOD_FAILURE"
        next_step = "D43SB_DATASET_REPAIR"
    elif hard_vote["test_accuracy"] < 1.0 or hard_vote["ood_accuracy"] < 1.0:
        decision = "d43s_oracle_tail_failure"
        verdict = "D43S_ORACLE_TAIL_FAILURE"
        next_step = "D43SC_TAIL_GENERATOR_REPAIR"
    elif baseline["noisy_majority_margin_low_accuracy"] >= 1.0:
        decision = "d43_tail_not_reproduced_baseline_already_solved"
        verdict = "D43S_TAIL_NOT_REPRODUCED"
        next_step = "D44_FORMULA_PRIMITIVE_DISCOVERY_PLAN"
    elif (
        comparison["noisy_low_delta"] >= 0.001
        and combined["noisy_majority_margin_low_accuracy"] >= 0.999
        and combined["test_accuracy"] >= 0.999
        and combined["ood_accuracy"] >= 0.999
        and comparison["easy_case_regression"] <= 0.0005
        and controls_collapse
    ):
        decision = "low_margin_noisy_support_tail_hardened"
        verdict = "D43S_LOW_MARGIN_NOISY_SUPPORT_TAIL_HARDENED"
        next_step = "D44_FORMULA_PRIMITIVE_DISCOVERY_PLAN"
    else:
        decision = "low_margin_noisy_support_tail_not_hardened"
        verdict = "D43S_TAIL_NOT_HARDENED"
        next_step = "D43T_EVIDENCE_GAP_CALIBRATION_PLAN"
    decision_payload = {
        "decision": decision,
        "verdict": verdict,
        "next": next_step,
        "boundary": "D43S only tests low-margin noisy-support tail hardening for D43 on a controlled symbolic task with fixed formula primitive candidates.",
        "non_claims": {
            "formula_primitive_discovery": False,
            "raw_visual_raven_reasoning": False,
            "raven_solved": False,
            "dna_genome_success": False,
            "architecture_superiority": False,
            "consciousness": False,
            "general_intelligence": False,
        },
    }
    write_json(out / "decision.json", decision_payload)
    write_json(out / "summary.json", {"decision": decision, "verdict": verdict, "next": next_step})
    aggregate = json.loads((out / "aggregate_metrics.json").read_text())
    aggregate["decision"] = decision_payload
    write_json(out / "aggregate_metrics.json", aggregate)
    write_json(out / "machine_utilization_report.json", {
        "os_cpu_count": os.cpu_count(),
        "worker_count": workers,
        "thread_env": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        },
        "wall_clock_sec": time.time() - started,
        "completed_jobs": len(results),
        "failed_jobs": len(failures),
    })
    (out / "report.md").write_text(d43s_report(decision_payload, arm_reports, comparison, out.as_posix()))
    print(json.dumps({"status": "ok", "decision": decision, "verdict": verdict, "next": next_step, "failed_jobs": len(failures)}, indent=2))


if __name__ == "__main__":
    main()
