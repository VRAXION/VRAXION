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
    "RANDOM_BASELINE",
    "QUERY_ONLY_MONOLITHIC_BASELINE",
    "SUPPORT_EVIDENCE_ORACLE_RULE_SELECTOR",
    "TRUE_FAMILY_ORACLE_UPPER_BOUND",
    "MUTABLE_LEARNED_RULE_FAMILY_INFERENCE",
    "MUTABLE_LEARNED_RULE_INFERENCE_PLUS_LEARNED_ROUTER",
    "SHUFFLED_SUPPORT_EVIDENCE_CONTROL",
    "NO_SUPPORT_EVIDENCE_CONTROL",
    "WRONG_SUPPORT_CONTROL",
    "SAME_QUERY_DIFFERENT_SUPPORT_COUNTERFACTUAL",
]
REPORT_FILES = {
    "RANDOM_BASELINE": "random_baseline_report.json",
    "QUERY_ONLY_MONOLITHIC_BASELINE": "query_only_monolithic_baseline_report.json",
    "SUPPORT_EVIDENCE_ORACLE_RULE_SELECTOR": "support_evidence_oracle_rule_selector_report.json",
    "TRUE_FAMILY_ORACLE_UPPER_BOUND": "true_family_oracle_upper_bound_report.json",
    "MUTABLE_LEARNED_RULE_FAMILY_INFERENCE": "mutable_learned_rule_family_inference_report.json",
    "MUTABLE_LEARNED_RULE_INFERENCE_PLUS_LEARNED_ROUTER": "mutable_learned_rule_inference_plus_router_report.json",
    "SHUFFLED_SUPPORT_EVIDENCE_CONTROL": "shuffled_support_evidence_control_report.json",
    "NO_SUPPORT_EVIDENCE_CONTROL": "no_support_evidence_control_report.json",
    "WRONG_SUPPORT_CONTROL": "wrong_support_control_report.json",
    "SAME_QUERY_DIFFERENT_SUPPORT_COUNTERFACTUAL": "same_query_different_support_counterfactual_report.json",
}
FORMULA_DESC = {
    "row": "(b[1][0] + b[1][2]) % 9",
    "col": "(b[0][1] + b[2][1]) % 9",
    "pair": "(b[0][0] + b[2][2]) % 9",
    "mirror": "(b[2][0] + b[0][2]) % 9",
    "diag": "(b[0][0] + b[1][2] + b[2][1]) % 9",
}
MUTATION_TYPES = [
    "rule_weight_delta",
    "rule_row_delta",
    "rule_column_delta",
    "rule_bias_delta",
    "rule_row_swap",
    "rule_column_swap",
    "prune_small_weights",
]
DEFAULT_OUT = Path("target/pilot_wave/d40_rule_family_inference_router_prototype/smoke")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--seeds", default="8601,8602,8603,8604,8605")
    parser.add_argument("--support-count", type=int, default=3)
    parser.add_argument("--train-rows-per-seed", type=int, default=700)
    parser.add_argument("--test-rows-per-seed", type=int, default=700)
    parser.add_argument("--ood-rows-per-seed", type=int, default=700)
    parser.add_argument("--generations", type=int, default=500)
    parser.add_argument("--population", type=int, default=128)
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
        multiplier = [2, 4, 5, 7, 8][(idx + r + c) % 5]
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


def make_support_board(rng, split, idx, family_idx):
    for attempt in range(1000):
        board = random_noncenter_board(rng, split, idx + attempt)
        values = formula_values(board)
        center = values[family_idx]
        if all(values[other] != center for other in range(5) if other != family_idx):
            board[1][1] = center
            return board
    raise RuntimeError("could not generate unambiguous support board")


def make_query_board(rng, split, idx):
    for attempt in range(1000):
        board = random_noncenter_board(rng, split, idx + 17 * attempt)
        board[1][1] = None
        values = formula_values(board)
        if len(set(values)) == len(values):
            return board, values
    raise RuntimeError("could not generate distinct query formula values")


def support_evidence(support_boards, support_count):
    counts = [0 for _ in range(5)]
    for board in support_boards:
        center = board[1][1]
        for family_idx in range(5):
            if center == family_target(family_idx, board):
                counts[family_idx] += 1
    vector = [count / support_count for count in counts]
    ordered = sorted(counts, reverse=True)
    margin_count = ordered[0] - ordered[1]
    margin_normalized = margin_count / support_count
    return vector, counts, margin_count, margin_normalized


def make_support_package(rng, split, idx, family_idx, support_count):
    support_boards = [
        make_support_board(rng, split, idx * 101 + support_idx * 17, family_idx)
        for support_idx in range(support_count)
    ]
    evidence_vector, counts, margin_count, margin_normalized = support_evidence(support_boards, support_count)
    winners = [family for family, count in enumerate(counts) if count == max(counts)]
    if winners != [family_idx] or not (margin_count >= 1 or margin_normalized >= 0.34):
        raise RuntimeError("ambiguous support evidence escaped generator")
    return support_boards, evidence_vector, counts, margin_count, margin_normalized


def make_row(seed, split, idx, support_count, family_idx=None):
    split_id = {"train": 11, "test": 23, "ood": 37}[split]
    rng = random.Random(stable_mix(seed, split_id, idx, support_count))
    intended_family = (idx + seed + split_id) % 5 if family_idx is None else family_idx
    support_boards, evidence_vector, evidence_counts, margin_count, margin_normalized = make_support_package(
        rng, split, idx, intended_family, support_count
    )
    wrong_family = (intended_family + 1 + (idx % 4)) % 5
    wrong_support_boards, wrong_evidence_vector, wrong_evidence_counts, _, _ = make_support_package(
        rng, split, idx + 900000, wrong_family, support_count
    )
    query_board, query_formula_values = make_query_board(rng, split, idx + 300000)
    query_target = query_formula_values[intended_family]
    pockets = list(range(9))
    rng.shuffle(pockets)
    expected_selected = pockets.index(query_target)
    formula_pockets = [pockets.index(value) for value in query_formula_values]
    return {
        "id": idx,
        "split": split,
        "support_count": support_count,
        "intended_family": FAMILIES[intended_family],
        "intended_family_index": intended_family,
        "wrong_support_family": FAMILIES[wrong_family],
        "wrong_support_family_index": wrong_family,
        "support_boards": support_boards,
        "wrong_support_boards": wrong_support_boards,
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


def make_rows(seed, split, count, support_count):
    return [make_row(seed, split, idx, support_count) for idx in range(count)]


def make_counterfactual_rows(seed, split, count, support_count):
    split_id = {"train": 41, "test": 53, "ood": 67}[split]
    rng = random.Random(stable_mix(seed, split_id, count, support_count))
    rows = []
    base_count = (count + 4) // 5
    for base_idx in range(base_count):
        query_board, query_formula_values = make_query_board(rng, split, base_idx + 600000)
        pockets = list(range(9))
        rng.shuffle(pockets)
        formula_pockets = [pockets.index(value) for value in query_formula_values]
        for family_idx in range(5):
            if len(rows) >= count:
                break
            row_idx = len(rows)
            support_boards, evidence_vector, evidence_counts, margin_count, margin_normalized = make_support_package(
                rng, split, base_idx * 1000 + family_idx, family_idx, support_count
            )
            target = query_formula_values[family_idx]
            rows.append(
                {
                    "id": row_idx,
                    "counterfactual_group": base_idx,
                    "split": split,
                    "support_count": support_count,
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


def initial_rule_model(rng):
    return {
        "matrix": [[rng.uniform(-0.05, 0.05) for _ in range(5)] for _ in range(5)],
        "bias": [rng.uniform(-0.02, 0.02) for _ in range(5)],
    }


def clone_rule_model(model):
    return {"matrix": [row[:] for row in model["matrix"]], "bias": model["bias"][:]}


def initial_vector_model(rng):
    return {
        "weights": [rng.uniform(-0.05, 0.05) for _ in range(5)],
        "pocket_bias": [rng.uniform(-0.01, 0.01) for _ in range(9)],
    }


def clone_vector_model(model):
    return {"weights": model["weights"][:], "pocket_bias": model["pocket_bias"][:]}


def evidence_for_mode(row, mode):
    if mode == "normal":
        return row["support_evidence_vector"]
    if mode == "shuffled":
        vector = row["support_evidence_vector"]
        return vector[1:] + vector[:1]
    if mode == "none":
        return [0.0 for _ in range(5)]
    if mode == "wrong":
        return row["wrong_support_evidence_vector"]
    raise ValueError(f"unknown evidence mode: {mode}")


def infer_family(model, evidence_vector):
    scores = model["bias"][:]
    for evidence_idx, evidence_value in enumerate(evidence_vector):
        if evidence_value:
            for family_idx in range(5):
                scores[family_idx] += evidence_value * model["matrix"][evidence_idx][family_idx]
    return top_two(scores)


def predict_learned_rule(model, row, evidence_mode="normal", router_mode="oracle"):
    family_idx, margin = infer_family(model, evidence_for_mode(row, evidence_mode))
    if router_mode == "frozen_identity":
        selected, route_margin = frozen_identity_router(row, family_idx)
        return selected, min(margin, route_margin), family_idx, margin
    selected, _route_margin = route_formula(row, family_idx)
    return selected, margin, family_idx, margin


def predict_query_vector(model, row):
    scores = model["pocket_bias"][:]
    for formula_idx, pocket_idx in enumerate(row["formula_pockets"]):
        scores[pocket_idx] += model["weights"][formula_idx]
    selected, margin = top_two(scores)
    return selected, margin, None, None


def predict_support_oracle(row):
    family_idx, margin = top_two(row["support_evidence_vector"])
    selected, _ = route_formula(row, family_idx)
    return selected, margin, family_idx, margin


def predict_true_family(row):
    selected, _ = route_formula(row, row["intended_family_index"])
    return selected, 1.0, row["intended_family_index"], 1.0


def predict_wrong_support(row):
    family_idx, margin = top_two(row["wrong_support_evidence_vector"])
    selected, _ = route_formula(row, family_idx)
    return selected, margin, family_idx, margin


def rule_mutation_specs(rng, population, step):
    specs = []
    for row_idx in range(5):
        for col_idx in range(5):
            specs.append(("rule_weight_delta", row_idx, col_idx, step))
            specs.append(("rule_weight_delta", row_idx, col_idx, -step))
    while len(specs) < population:
        specs.append((MUTATION_TYPES[rng.randrange(len(MUTATION_TYPES))],))
    rng.shuffle(specs)
    return specs[:population]


def apply_rule_mutation(rng, model, spec, step):
    candidate = clone_rule_model(model)
    op = spec[0]
    if op == "rule_weight_delta":
        row_idx = spec[1] if len(spec) > 1 else rng.randrange(5)
        col_idx = spec[2] if len(spec) > 2 else rng.randrange(5)
        delta = spec[3] if len(spec) > 3 else rng.uniform(-step, step)
        candidate["matrix"][row_idx][col_idx] += delta
    elif op == "rule_row_delta":
        row_idx = rng.randrange(5)
        for col_idx in range(5):
            candidate["matrix"][row_idx][col_idx] += rng.uniform(-step, step)
    elif op == "rule_column_delta":
        col_idx = rng.randrange(5)
        delta = rng.uniform(-step, step)
        for row_idx in range(5):
            candidate["matrix"][row_idx][col_idx] += delta
    elif op == "rule_bias_delta":
        candidate["bias"][rng.randrange(5)] += rng.uniform(-step, step)
    elif op == "rule_row_swap":
        a = rng.randrange(5)
        b = rng.randrange(5)
        candidate["matrix"][a], candidate["matrix"][b] = candidate["matrix"][b], candidate["matrix"][a]
    elif op == "rule_column_swap":
        a = rng.randrange(5)
        b = rng.randrange(5)
        for row in candidate["matrix"]:
            row[a], row[b] = row[b], row[a]
    elif op == "prune_small_weights":
        threshold = 0.05 * step
        for row_idx in range(5):
            for col_idx in range(5):
                if abs(candidate["matrix"][row_idx][col_idx]) < threshold:
                    candidate["matrix"][row_idx][col_idx] = 0.0
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


def balanced_subset(rows, limit):
    if len(rows) <= limit:
        return rows
    buckets = defaultdict(list)
    for row in rows:
        buckets[row["intended_family_index"]].append(row)
    out = []
    cursor = 0
    while len(out) < limit:
        bucket = buckets[cursor % 5]
        if bucket:
            out.append(bucket[(cursor // 5) % len(bucket)])
        cursor += 1
    return out


def fitness_rule(model, rows, evidence_mode="normal", router_mode="oracle"):
    correct = 0
    margin_sum = 0.0
    for row in rows:
        selected, margin, _family_idx, _rule_margin = predict_learned_rule(model, row, evidence_mode, router_mode)
        correct += int(selected == row["expected_selected"])
        margin_sum += margin
    return correct * 10000.0 + margin_sum


def fitness_vector(model, rows):
    correct = 0
    margin_sum = 0.0
    for row in rows:
        selected, margin, _family_idx, _rule_margin = predict_query_vector(model, row)
        correct += int(selected == row["expected_selected"])
        margin_sum += margin
    return correct * 10000.0 + margin_sum


def train_rule_model(train_rows, seed, generations, population, job_dir):
    rng = random.Random(stable_mix(seed, 101, generations, population))
    fit_rows = balanced_subset(train_rows, min(180, len(train_rows)))
    best = initial_rule_model(rng)
    best_fit = fitness_rule(best, fit_rows)
    accepted = Counter({op: 0 for op in MUTATION_TYPES})
    rejected = Counter({op: 0 for op in MUTATION_TYPES})
    convergence_generation = None
    perfect_since = None
    generations_executed = 0
    for generation in range(generations):
        generations_executed = generation + 1
        step = max(0.05, 1.0 * (1.0 - (generation / max(1, generations))))
        records = []
        winner = None
        winner_op = None
        winner_fit = -1.0e30
        for spec in rule_mutation_specs(rng, population, step):
            candidate, op = apply_rule_mutation(rng, best, spec, step)
            fit = fitness_rule(candidate, fit_rows)
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
        winner_rejected_skipped = False
        for fit, op, _candidate in records:
            if accepted_this_generation and not winner_rejected_skipped and op == winner_op and fit == winner_fit:
                winner_rejected_skipped = True
                continue
            rejected[op] += 1
        if generation % max(1, generations // 25) == 0 or generation == generations - 1:
            metrics = evaluate_rows(train_rows, lambda row: predict_learned_rule(best, row))
            append_jsonl(job_dir / "train_metrics.jsonl", {"generation": generation, "train_accuracy": metrics["accuracy"], "rule_family_train_accuracy": metrics["rule_family_accuracy"], "fitness": best_fit})
            append_jsonl(job_dir / "progress.jsonl", {"generation": generation, "generations": generations})
            if metrics["accuracy"] >= 0.95 and metrics["rule_family_accuracy"] >= 0.95 and convergence_generation is None:
                convergence_generation = generation
            if metrics["accuracy"] == 1.0 and metrics["rule_family_accuracy"] == 1.0:
                if perfect_since is None:
                    perfect_since = generation
                elif generation - perfect_since >= max(20, generations // 20):
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
    return best, mutation


def train_vector_model(train_rows, seed, generations, population, job_dir):
    rng = random.Random(stable_mix(seed, 211, generations, population))
    fit_rows = balanced_subset(train_rows, min(180, len(train_rows)))
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
        family_name = row["intended_family"]
        per_family_total[family_name] += 1
        per_family_correct[family_name] += int(is_correct)
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
                "truth": truth,
                "pred": selected,
                "inferred_family": FAMILIES[inferred_family] if inferred_family is not None else None,
                "margin": margin,
                "rule_margin": rule_margin,
            }
        )
    return {
        "accuracy": correct / max(1, len(rows)),
        "rule_family_accuracy": (rule_correct / rule_total) if rule_total else None,
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


def flatten_metrics(train, test, ood):
    payload = {
        "train_accuracy": train["accuracy"],
        "test_accuracy": test["accuracy"],
        "ood_accuracy": ood["accuracy"],
        "rule_family_train_accuracy": train["rule_family_accuracy"],
        "rule_family_test_accuracy": test["rule_family_accuracy"],
        "rule_family_ood_accuracy": ood["rule_family_accuracy"],
        "per_family_accuracy": test["per_family_accuracy"],
        "error_count": test["error_count"],
        "pocket_confusion_matrix": test["pocket_confusion_matrix"],
        "family_confusion_matrix": test["family_confusion_matrix"],
        "median_score_margin": test["median_score_margin"],
        "low_margin_error_rate": test["low_margin_error_rate"],
    }
    return payload


def dataset_invariants(rows):
    duplicate = 0
    missing = 0
    selected = 0
    ambiguous = 0
    ties = 0
    unique = 0
    margins_count = []
    margins_normalized = []
    for row in rows:
        target = row["query_target_symbol"]
        count = sum(1 for symbol in row["pockets"] if symbol == target)
        duplicate += int(count > 1)
        missing += int(count == 0)
        selected += int(row["pockets"][row["expected_selected"]] == target)
        counts = row["support_evidence_counts"]
        max_count = max(counts)
        winners = [idx for idx, value in enumerate(counts) if value == max_count]
        tied = len(winners) > 1
        ties += int(tied)
        intended_unique = winners == [row["intended_family_index"]]
        unique += int(intended_unique)
        ambiguous += int((not intended_unique) or row["support_evidence_margin_count"] < 1 or row["support_evidence_margin_normalized"] < 0.34)
        margins_count.append(row["support_evidence_margin_count"])
        margins_normalized.append(row["support_evidence_margin_normalized"])
    return {
        "duplicate_target_pocket_rate": duplicate / max(1, len(rows)),
        "missing_target_pocket_rate": missing / max(1, len(rows)),
        "expected_selected_points_to_target_rate": selected / max(1, len(rows)),
        "ambiguous_support_rate": ambiguous / max(1, len(rows)),
        "multi_family_support_tie_rate": ties / max(1, len(rows)),
        "intended_family_unique_evidence_rate": unique / max(1, len(rows)),
        "support_evidence_margin_count_min": min(margins_count) if margins_count else 0,
        "support_evidence_margin_normalized_min": min(margins_normalized) if margins_normalized else 0.0,
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


def rule_identity(model):
    effective = [
        [model["matrix"][row_idx][col_idx] + model["bias"][col_idx] for col_idx in range(5)]
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
        "rule_entropy": statistics.mean(entropies),
        "effective_matrix": [[round(value, 6) for value in row] for row in effective],
    }


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
    train_rows = make_rows(seed, "train", config["train_rows_per_seed"], config["support_count"])
    test_rows = make_rows(seed, "test", config["test_rows_per_seed"], config["support_count"])
    ood_rows = make_rows(seed, "ood", config["ood_rows_per_seed"], config["support_count"])
    extra = {}

    if arm == "RANDOM_BASELINE":
        predictor = None
        train = evaluate_rows(train_rows, lambda row: None, random_seed=stable_mix(seed, 1))
        test = evaluate_rows(test_rows, lambda row: None, random_seed=stable_mix(seed, 2))
        ood = evaluate_rows(ood_rows, lambda row: None, random_seed=stable_mix(seed, 3))
        write_json(job_dir / "best_individual.json", {"type": "deterministic_random_baseline", "seed": seed})
        append_jsonl(job_dir / "train_metrics.jsonl", {"generation": 0, "train_accuracy": train["accuracy"]})
        append_jsonl(job_dir / "progress.jsonl", {"generation": 0, "generations": 0})
    elif arm == "QUERY_ONLY_MONOLITHIC_BASELINE":
        model, training = train_vector_model(train_rows, stable_mix(seed, 4), config["generations"], config["population"], job_dir)
        predictor = lambda row: predict_query_vector(model, row)
        train = evaluate_rows(train_rows, predictor)
        test = evaluate_rows(test_rows, predictor)
        ood = evaluate_rows(ood_rows, predictor)
        write_json(job_dir / "best_individual.json", {"type": "query_only_vector", "model": model, "training": training})
    elif arm == "SUPPORT_EVIDENCE_ORACLE_RULE_SELECTOR":
        predictor = predict_support_oracle
        train = evaluate_rows(train_rows, predictor)
        test = evaluate_rows(test_rows, predictor)
        ood = evaluate_rows(ood_rows, predictor)
        write_json(job_dir / "best_individual.json", {"type": "support_evidence_argmax_oracle"})
        append_jsonl(job_dir / "train_metrics.jsonl", {"generation": 0, "train_accuracy": train["accuracy"], "rule_family_train_accuracy": train["rule_family_accuracy"]})
        append_jsonl(job_dir / "progress.jsonl", {"generation": 0, "generations": 0})
    elif arm == "TRUE_FAMILY_ORACLE_UPPER_BOUND":
        predictor = predict_true_family
        train = evaluate_rows(train_rows, predictor)
        test = evaluate_rows(test_rows, predictor)
        ood = evaluate_rows(ood_rows, predictor)
        write_json(job_dir / "best_individual.json", {"type": "true_family_oracle_upper_bound"})
        append_jsonl(job_dir / "train_metrics.jsonl", {"generation": 0, "train_accuracy": train["accuracy"], "rule_family_train_accuracy": train["rule_family_accuracy"]})
        append_jsonl(job_dir / "progress.jsonl", {"generation": 0, "generations": 0})
    elif arm in {
        "MUTABLE_LEARNED_RULE_FAMILY_INFERENCE",
        "MUTABLE_LEARNED_RULE_INFERENCE_PLUS_LEARNED_ROUTER",
        "SHUFFLED_SUPPORT_EVIDENCE_CONTROL",
        "NO_SUPPORT_EVIDENCE_CONTROL",
    }:
        model, mutation = train_rule_model(train_rows, stable_mix(seed, ARMS.index(arm), 5), config["generations"], config["population"], job_dir)
        evidence_mode = "normal"
        router_mode = "oracle"
        if arm == "MUTABLE_LEARNED_RULE_INFERENCE_PLUS_LEARNED_ROUTER":
            router_mode = "frozen_identity"
            extra["router_mode"] = "frozen_identity_confirmed_router"
        elif arm == "SHUFFLED_SUPPORT_EVIDENCE_CONTROL":
            evidence_mode = "shuffled"
        elif arm == "NO_SUPPORT_EVIDENCE_CONTROL":
            evidence_mode = "none"
        predictor = lambda row: predict_learned_rule(model, row, evidence_mode=evidence_mode, router_mode=router_mode)
        train = evaluate_rows(train_rows, predictor)
        test = evaluate_rows(test_rows, predictor)
        ood = evaluate_rows(ood_rows, predictor)
        identity = rule_identity(model)
        extra.update({"mutation": mutation, "rule_identity": identity, "evidence_mode": evidence_mode})
        write_json(job_dir / "best_individual.json", {"type": "mutable_rule_inference", "model": model, "mutation": mutation, "rule_identity": identity, "evidence_mode": evidence_mode, "router_mode": router_mode})
    elif arm == "WRONG_SUPPORT_CONTROL":
        predictor = predict_wrong_support
        train = evaluate_rows(train_rows, predictor)
        test = evaluate_rows(test_rows, predictor)
        ood = evaluate_rows(ood_rows, predictor)
        follow_rates = []
        for rows in [train_rows, test_rows, ood_rows]:
            follow_rates.append(sum(int(top_two(row["wrong_support_evidence_vector"])[0] == row["wrong_support_family_index"]) for row in rows) / max(1, len(rows)))
        extra["wrong_support_follow_rate_train"] = follow_rates[0]
        extra["wrong_support_follow_rate_test"] = follow_rates[1]
        extra["wrong_support_follow_rate_ood"] = follow_rates[2]
        extra["wrong_support_query_mismatch_rate_test"] = 1.0 - test["accuracy"]
        write_json(job_dir / "best_individual.json", {"type": "wrong_support_argmax_control"})
        append_jsonl(job_dir / "train_metrics.jsonl", {"generation": 0, "train_accuracy": train["accuracy"], "rule_family_train_accuracy": train["rule_family_accuracy"]})
        append_jsonl(job_dir / "progress.jsonl", {"generation": 0, "generations": 0})
    elif arm == "SAME_QUERY_DIFFERENT_SUPPORT_COUNTERFACTUAL":
        model, mutation = train_rule_model(train_rows, stable_mix(seed, 91), config["generations"], config["population"], job_dir)
        cf_train = make_counterfactual_rows(seed, "train", config["train_rows_per_seed"], config["support_count"])
        cf_test = make_counterfactual_rows(seed, "test", config["test_rows_per_seed"], config["support_count"])
        cf_ood = make_counterfactual_rows(seed, "ood", config["ood_rows_per_seed"], config["support_count"])
        predictor = lambda row: predict_learned_rule(model, row)
        train = evaluate_rows(cf_train, predictor)
        test = evaluate_rows(cf_test, predictor)
        ood = evaluate_rows(cf_ood, predictor)
        identity = rule_identity(model)
        extra.update({"mutation": mutation, "rule_identity": identity, "counterfactual_group_count_test": len({row["counterfactual_group"] for row in cf_test})})
        write_json(job_dir / "best_individual.json", {"type": "same_query_different_support_counterfactual", "model": model, "mutation": mutation, "rule_identity": identity})
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
        return {
            "train_accuracy": 0.0,
            "test_accuracy": 0.0,
            "ood_accuracy": 0.0,
            "rule_family_train_accuracy": None,
            "rule_family_test_accuracy": None,
            "rule_family_ood_accuracy": None,
            "min_seed_test_accuracy": 0.0,
            "min_seed_ood_accuracy": 0.0,
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
    return {
        "train_accuracy": statistics.mean(row["train_accuracy"] for row in rows),
        "test_accuracy": statistics.mean(row["test_accuracy"] for row in rows),
        "ood_accuracy": statistics.mean(row["ood_accuracy"] for row in rows),
        "rule_family_train_accuracy": statistics.mean(rule_train) if rule_train else None,
        "rule_family_test_accuracy": statistics.mean(rule_test) if rule_test else None,
        "rule_family_ood_accuracy": statistics.mean(rule_ood) if rule_ood else None,
        "min_seed_test_accuracy": min(row["test_accuracy"] for row in rows),
        "min_seed_ood_accuracy": min(row["ood_accuracy"] for row in rows),
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


def variance(values):
    return statistics.pvariance(values) if values else 0.0


def build_rule_matrix_report(rows):
    items = {str(row["seed"]): row["rule_identity"] for row in rows if "rule_identity" in row}
    if not items:
        return {
            "rule_argmax_mapping_by_seed": {},
            "rule_identity_alignment_score_mean": 0.0,
            "rule_identity_alignment_score_min": 0.0,
            "rule_diagonal_mass_mean": 0.0,
            "rule_off_diagonal_mass_mean": 0.0,
            "rule_entropy_mean": 0.0,
            "effective_rule_matrix_by_seed": {},
        }
    scores = [item["alignment_score"] for item in items.values()]
    return {
        "rule_argmax_mapping_by_seed": {seed: item["mapping"] for seed, item in items.items()},
        "rule_identity_alignment_score_mean": statistics.mean(scores),
        "rule_identity_alignment_score_min": min(scores),
        "rule_diagonal_mass_mean": statistics.mean(item["diagonal_mass"] for item in items.values()),
        "rule_off_diagonal_mass_mean": statistics.mean(item["off_diagonal_mass"] for item in items.values()),
        "rule_entropy_mean": statistics.mean(item["rule_entropy"] for item in items.values()),
        "effective_rule_matrix_by_seed": {seed: item["effective_matrix"] for seed, item in items.items()},
    }


def build_mutation_report(rows):
    accepted = Counter({op: 0 for op in MUTATION_TYPES})
    rejected = Counter({op: 0 for op in MUTATION_TYPES})
    convergence = []
    for row in rows:
        mutation = row.get("mutation")
        if not mutation:
            continue
        accepted.update(mutation["accepted_mutations_by_type"])
        rejected.update(mutation["rejected_mutations_by_type"])
        if mutation["convergence_generation"] is not None:
            convergence.append(mutation["convergence_generation"])
    return {
        "accepted_mutations_by_type": dict(accepted),
        "rejected_mutations_by_type": dict(rejected),
        "mutation_acceptance_rate": sum(accepted.values()) / max(1, sum(accepted.values()) + sum(rejected.values())),
        "convergence_generation_median": statistics.median(convergence) if convergence else None,
    }


def d39_upstream_manifest():
    paths = [
        "docs/research/D39_LEARNED_ROUTER_LAYER_SCALE_CONFIRM_CONTRACT.md",
        "docs/research/D39_LEARNED_ROUTER_LAYER_SCALE_CONFIRM_RESULT.md",
        "scripts/probes/run_d39_learned_router_layer_scale_confirm.py",
        "scripts/probes/run_d39_learned_router_layer_scale_confirm_check.py",
    ]
    decision = Path("target/pilot_wave/d39_learned_router_layer_scale_confirm/smoke/decision.json")
    payload = {
        "required_files": {path: Path(path).exists() for path in paths},
        "artifact_decision_path": decision.as_posix(),
        "artifact_decision_present": decision.exists(),
    }
    if decision.exists():
        payload["artifact_decision"] = json.loads(decision.read_text())
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
        "per_family_accuracy": report["per_family_accuracy"],
        "failed_seed_count": report["failed_seed_count"],
    }


def make_report(decision_payload, arm_reports, deltas, artifact_path):
    lines = [
        "# D40 Rule Family Inference Router Prototype",
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
            f"- learned_vs_shuffled_support_test_delta = {deltas['learned_vs_shuffled_support_test_delta']:.4f}",
            f"- learned_vs_no_support_test_delta = {deltas['learned_vs_no_support_test_delta']:.4f}",
            "",
            "## Boundary",
            "",
            "A positive D40 proves only that support-evidence-based rule-family inference can feed the known-rule router path on a controlled symbolic pocket task. It does not prove raw visual Raven reasoning, full hidden-rule Raven solving, natural-language reasoning, DNA/genome success, Raven solved, architecture superiority, consciousness, or general intelligence.",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    started = time.time()
    out = Path(args.out)
    if "target/pilot_wave/d40_rule_family_inference_router_prototype/" not in out.as_posix():
        raise SystemExit("--out must stay under target/pilot_wave/d40_rule_family_inference_router_prototype/")
    out.mkdir(parents=True, exist_ok=True)
    progress_path = out / "progress.jsonl"
    if progress_path.exists():
        progress_path.unlink()
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    jobs = [{"seed": seed, "arm": arm} for seed in seeds for arm in ARMS]
    config = {
        "support_count": args.support_count,
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
    write_json(out / "queue.json", {"jobs": jobs, "seeds": seeds, "arms": ARMS, "workers": workers, "cpu_target": args.cpu_target, "heartbeat_sec": args.heartbeat_sec})

    all_rows = []
    test_rows = []
    ood_rows = []
    for seed in seeds:
        for split, count in [("train", args.train_rows_per_seed), ("test", args.test_rows_per_seed), ("ood", args.ood_rows_per_seed)]:
            rows = make_rows(seed, split, count, args.support_count)
            all_rows.extend(rows)
            if split == "test":
                test_rows.extend(rows)
            if split == "ood":
                ood_rows.extend(rows)
    invariants = dataset_invariants(all_rows)
    write_json(out / "dataset_invariant_report.json", invariants)
    support_test_selected, support_test_rule = rule_oracle_accuracy(test_rows)
    support_ood_selected, support_ood_rule = rule_oracle_accuracy(ood_rows)
    support_audit = {
        "ambiguous_support_rate": invariants["ambiguous_support_rate"],
        "multi_family_support_tie_rate": invariants["multi_family_support_tie_rate"],
        "intended_family_unique_evidence_rate": invariants["intended_family_unique_evidence_rate"],
        "support_evidence_margin_count_min": invariants["support_evidence_margin_count_min"],
        "support_evidence_margin_normalized_min": invariants["support_evidence_margin_normalized_min"],
        "support_rule_oracle_test_accuracy": support_test_rule,
        "support_rule_oracle_ood_accuracy": support_ood_rule,
        "support_selected_pocket_oracle_test_accuracy": support_test_selected,
        "support_selected_pocket_oracle_ood_accuracy": support_ood_selected,
    }
    write_json(out / "support_evidence_audit.json", support_audit)
    ood_audit = {
        "support_rule_oracle_test_accuracy": support_test_rule,
        "support_rule_oracle_ood_accuracy": support_ood_rule,
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
            "support_count": args.support_count,
            "seeds": seeds,
            "train_rows_per_seed": args.train_rows_per_seed,
            "test_rows_per_seed": args.test_rows_per_seed,
            "ood_rows_per_seed": args.ood_rows_per_seed,
            "query_formula_values_required_distinct": True,
            "support_rows_required_unambiguous": True,
            "ood_semantics": "held-out symbol distributions with support centers and query targets recomputed by the same formulas",
        },
    )
    write_json(out / "d39_upstream_manifest.json", d39_upstream_manifest())

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

    learned_rows = [row for row in results if row["arm"] == "MUTABLE_LEARNED_RULE_FAMILY_INFERENCE"]
    rule_matrix_report = build_rule_matrix_report(learned_rows)
    mutation_report = build_mutation_report(learned_rows)
    seed_variance = {
        "test_accuracy_variance": variance([row["test_accuracy"] for row in learned_rows]),
        "ood_accuracy_variance": variance([row["ood_accuracy"] for row in learned_rows]),
        "rule_family_test_accuracy_variance": variance([row["rule_family_test_accuracy"] for row in learned_rows if row["rule_family_test_accuracy"] is not None]),
        "rule_family_ood_accuracy_variance": variance([row["rule_family_ood_accuracy"] for row in learned_rows if row["rule_family_ood_accuracy"] is not None]),
    }
    learned_report = dict(arm_reports["MUTABLE_LEARNED_RULE_FAMILY_INFERENCE"])
    learned_report.update(
        {
            "learned_rule_family_train_accuracy": learned_report["rule_family_train_accuracy"],
            "learned_rule_family_test_accuracy": learned_report["rule_family_test_accuracy"],
            "learned_rule_family_ood_accuracy": learned_report["rule_family_ood_accuracy"],
            "learned_selected_pocket_train_accuracy": learned_report["train_accuracy"],
            "learned_selected_pocket_test_accuracy": learned_report["test_accuracy"],
            "learned_selected_pocket_ood_accuracy": learned_report["ood_accuracy"],
            "min_seed_learned_test_accuracy": learned_report["min_seed_test_accuracy"],
            "min_seed_learned_ood_accuracy": learned_report["min_seed_ood_accuracy"],
            "rule_identity_alignment_score_mean": rule_matrix_report["rule_identity_alignment_score_mean"],
            "rule_identity_alignment_score_min": rule_matrix_report["rule_identity_alignment_score_min"],
            "rule_diagonal_mass_mean": rule_matrix_report["rule_diagonal_mass_mean"],
            "rule_off_diagonal_mass_mean": rule_matrix_report["rule_off_diagonal_mass_mean"],
            "rule_argmax_mapping_by_seed": rule_matrix_report["rule_argmax_mapping_by_seed"],
            "rule_entropy_mean": rule_matrix_report["rule_entropy_mean"],
            "accepted_mutations_by_type": mutation_report["accepted_mutations_by_type"],
            "rejected_mutations_by_type": mutation_report["rejected_mutations_by_type"],
            "mutation_acceptance_rate": mutation_report["mutation_acceptance_rate"],
            "convergence_generation_median": mutation_report["convergence_generation_median"],
            "seed_variance": seed_variance,
        }
    )
    write_json(out / "mutable_learned_rule_family_inference_report.json", learned_report)
    write_json(out / "rule_inference_matrix_report.json", rule_matrix_report)
    write_json(
        out / "rule_identity_alignment_report.json",
        {
            "rule_identity_alignment_score_mean": rule_matrix_report["rule_identity_alignment_score_mean"],
            "rule_identity_alignment_score_min": rule_matrix_report["rule_identity_alignment_score_min"],
            "rule_argmax_mapping_by_seed": rule_matrix_report["rule_argmax_mapping_by_seed"],
        },
    )
    write_json(out / "mutation_acceptance_report.json", mutation_report)
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
    write_json(out / "score_margin_report.json", {arm: {"median_score_margin": arm_reports[arm]["median_score_margin"], "low_margin_error_rate": arm_reports[arm]["low_margin_error_rate"]} for arm in ARMS})

    wrong_rows = [row for row in results if row["arm"] == "WRONG_SUPPORT_CONTROL"]
    wrong_behavior = {
        "wrong_support_selected_pocket_test_accuracy": arm_reports["WRONG_SUPPORT_CONTROL"]["test_accuracy"],
        "wrong_support_selected_pocket_ood_accuracy": arm_reports["WRONG_SUPPORT_CONTROL"]["ood_accuracy"],
        "wrong_support_follow_rate_test": statistics.mean(row.get("wrong_support_follow_rate_test", 0.0) for row in wrong_rows) if wrong_rows else 0.0,
        "wrong_support_follow_rate_ood": statistics.mean(row.get("wrong_support_follow_rate_ood", 0.0) for row in wrong_rows) if wrong_rows else 0.0,
        "wrong_support_query_mismatch_rate_test": statistics.mean(row.get("wrong_support_query_mismatch_rate_test", 0.0) for row in wrong_rows) if wrong_rows else 0.0,
    }
    counterfactual_accuracy = arm_reports["SAME_QUERY_DIFFERENT_SUPPORT_COUNTERFACTUAL"]["test_accuracy"]
    deltas = {
        "learned_vs_query_only_test_delta": arm_reports["MUTABLE_LEARNED_RULE_FAMILY_INFERENCE"]["test_accuracy"] - arm_reports["QUERY_ONLY_MONOLITHIC_BASELINE"]["test_accuracy"],
        "learned_vs_shuffled_support_test_delta": arm_reports["MUTABLE_LEARNED_RULE_FAMILY_INFERENCE"]["test_accuracy"] - arm_reports["SHUFFLED_SUPPORT_EVIDENCE_CONTROL"]["test_accuracy"],
        "learned_vs_no_support_test_delta": arm_reports["MUTABLE_LEARNED_RULE_FAMILY_INFERENCE"]["test_accuracy"] - arm_reports["NO_SUPPORT_EVIDENCE_CONTROL"]["test_accuracy"],
    }
    write_json(out / "arm_comparison_report.json", {"deltas": deltas, "wrong_support_behavior": wrong_behavior, "same_query_different_support_accuracy": counterfactual_accuracy})

    learned = arm_reports["MUTABLE_LEARNED_RULE_FAMILY_INFERENCE"]
    support_oracle = arm_reports["SUPPORT_EVIDENCE_ORACLE_RULE_SELECTOR"]
    true_oracle = arm_reports["TRUE_FAMILY_ORACLE_UPPER_BOUND"]
    plus_router = arm_reports["MUTABLE_LEARNED_RULE_INFERENCE_PLUS_LEARNED_ROUTER"]
    if (
        invariants["duplicate_target_pocket_rate"] != 0.0
        or invariants["missing_target_pocket_rate"] != 0.0
        or invariants["expected_selected_points_to_target_rate"] != 1.0
        or invariants["ambiguous_support_rate"] != 0.0
        or invariants["multi_family_support_tie_rate"] != 0.0
        or invariants["intended_family_unique_evidence_rate"] != 1.0
    ):
        decision = "d40_dataset_invariant_failure"
        verdict = "D40_DATASET_INVARIANT_FAILURE"
        next_step = "D40B_DATASET_REPAIR"
    elif support_oracle["test_accuracy"] < 0.99 or support_oracle["ood_accuracy"] < 0.99:
        decision = "d40_support_evidence_not_inferable"
        verdict = "D40_SUPPORT_EVIDENCE_NOT_INFERABLE"
        next_step = "D40C_SUPPORT_GENERATOR_REPAIR"
    elif (
        ood_audit["support_rule_oracle_test_accuracy"] != 1.0
        or ood_audit["support_rule_oracle_ood_accuracy"] != 1.0
        or ood_audit["known_rule_oracle_test_accuracy"] != 1.0
        or ood_audit["known_rule_oracle_ood_accuracy"] != 1.0
    ):
        decision = "d40_ood_rule_invariance_failure"
        verdict = "D40_OOD_RULE_INVARIANCE_FAILURE"
        next_step = "D40D_OOD_REPAIR"
    elif (
        learned["test_accuracy"] >= 0.95
        and learned["ood_accuracy"] >= 0.95
        and learned["rule_family_test_accuracy"] >= 0.95
        and learned["rule_family_ood_accuracy"] >= 0.95
        and learned["min_seed_test_accuracy"] >= 0.90
        and learned["min_seed_ood_accuracy"] >= 0.90
        and support_oracle["test_accuracy"] >= 0.99
        and support_oracle["ood_accuracy"] >= 0.99
        and true_oracle["test_accuracy"] >= 0.99
        and true_oracle["ood_accuracy"] >= 0.99
        and plus_router["test_accuracy"] >= 0.95
        and deltas["learned_vs_query_only_test_delta"] >= 0.40
        and deltas["learned_vs_shuffled_support_test_delta"] >= 0.70
        and deltas["learned_vs_no_support_test_delta"] >= 0.40
        and counterfactual_accuracy >= 0.95
    ):
        decision = "rule_family_inference_router_prototype_positive"
        verdict = "D40_RULE_FAMILY_INFERENCE_ROUTER_PROTOTYPE_POSITIVE"
        next_step = "D41_RULE_FAMILY_INFERENCE_ROUTER_SCALE_CONFIRM"
    elif learned["test_accuracy"] >= 0.95 and learned["ood_accuracy"] >= 0.95:
        decision = "rule_inference_positive_but_counterfactual_control_failed"
        verdict = "D40_COUNTERFACTUAL_CONTROL_GAP"
        next_step = "D40K_COUNTERFACTUAL_SUPPORT_BINDING_ANALYSIS"
    elif deltas["learned_vs_query_only_test_delta"] > 0.0:
        decision = "rule_family_inference_partial_signal"
        verdict = "D40_RULE_FAMILY_INFERENCE_PARTIAL_SIGNAL"
        next_step = "D40L_RULE_INFERENCE_OPTIMIZATION_PLAN"
    else:
        decision = "rule_family_inference_not_confirmed"
        verdict = "D40_RULE_FAMILY_INFERENCE_NOT_CONFIRMED"
        next_step = "D41_FEATURE_SPACE_DIAGNOSTIC"

    decision_payload = {
        "decision": decision,
        "verdict": verdict,
        "next": next_step,
        "boundary": "controlled symbolic support-context rule-family inference plus known-rule router path only",
        "non_claims": {
            "raw_visual_raven_reasoning": False,
            "full_hidden_rule_raven_solving": False,
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
        "wrong_support_behavior": wrong_behavior,
        "same_query_different_support_accuracy": counterfactual_accuracy,
        "learned_rule_inference": {
            "learned_rule_family_train_accuracy": learned_report["learned_rule_family_train_accuracy"],
            "learned_rule_family_test_accuracy": learned_report["learned_rule_family_test_accuracy"],
            "learned_rule_family_ood_accuracy": learned_report["learned_rule_family_ood_accuracy"],
            "learned_selected_pocket_train_accuracy": learned_report["learned_selected_pocket_train_accuracy"],
            "learned_selected_pocket_test_accuracy": learned_report["learned_selected_pocket_test_accuracy"],
            "learned_selected_pocket_ood_accuracy": learned_report["learned_selected_pocket_ood_accuracy"],
            "min_seed_learned_test_accuracy": learned_report["min_seed_learned_test_accuracy"],
            "min_seed_learned_ood_accuracy": learned_report["min_seed_learned_ood_accuracy"],
            "seed_variance": seed_variance,
        },
        "rule_identity": {key: value for key, value in rule_matrix_report.items() if key != "effective_rule_matrix_by_seed"},
        "mutation": mutation_report,
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


if __name__ == "__main__":
    main()
