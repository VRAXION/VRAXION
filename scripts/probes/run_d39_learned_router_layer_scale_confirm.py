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
    "MONOLITHIC_FORMULA_BASELINE",
    "ORACLE_GATED_RULE_FORMULA_UPPER_BOUND",
    "MUTABLE_LEARNED_ROUTER_GATE",
    "SHUFFLED_GATE_CONTROL",
    "NO_FAMILY_INPUT_CONTROL",
    "EXPLICIT_TARGET_STATE_UPPER_BOUND",
]
FORMULA_DESC = {
    "row": "(b[1][0] + b[1][2]) % 9",
    "col": "(b[0][1] + b[2][1]) % 9",
    "pair": "(b[0][0] + b[2][2]) % 9",
    "mirror": "(b[2][0] + b[0][2]) % 9",
    "diag": "(b[0][0] + b[1][2] + b[2][1]) % 9",
}
MUTATION_TYPES = [
    "gate_weight_delta",
    "gate_row_delta",
    "gate_column_delta",
    "gate_bias_delta",
    "pocket_bias_delta",
    "gate_row_swap",
    "gate_column_swap",
    "prune_small_weights",
]
DEFAULT_OUT = Path("target/pilot_wave/d39_learned_router_layer_scale_confirm/smoke")
REPORT_FILES = {
    "MONOLITHIC_FORMULA_BASELINE": "monolithic_formula_baseline_report.json",
    "ORACLE_GATED_RULE_FORMULA_UPPER_BOUND": "oracle_gated_rule_formula_report.json",
    "MUTABLE_LEARNED_ROUTER_GATE": "mutable_learned_router_gate_report.json",
    "SHUFFLED_GATE_CONTROL": "shuffled_gate_control_report.json",
    "NO_FAMILY_INPUT_CONTROL": "no_family_input_control_report.json",
    "EXPLICIT_TARGET_STATE_UPPER_BOUND": "explicit_target_state_upper_bound_report.json",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--seeds", default="8501,8502,8503,8504,8505,8506,8507,8508")
    parser.add_argument("--train-rows-per-seed", type=int, default=800)
    parser.add_argument("--test-rows-per-seed", type=int, default=800)
    parser.add_argument("--ood-rows-per-seed", type=int, default=800)
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


def make_rows(seed, split, count):
    split_id = {"train": 11, "test": 23, "ood": 37}[split]
    rng = random.Random(stable_mix(seed, split_id, count))
    rows = []
    for idx in range(count):
        family_idx = (idx + seed + split_id) % len(FAMILIES)
        board = [[rng.randrange(9) for _ in range(3)] for _ in range(3)]
        if split == "ood":
            multiplier = [2, 4, 5, 7, 8][idx % 5]
            shift = rng.randrange(9)
            board = [
                [
                    (multiplier * board[r][c] + shift + ((r + 2 * c + idx) % 3)) % 9
                    for c in range(3)
                ]
                for r in range(3)
            ]
        target_symbol = family_target(family_idx, board)
        wrong_symbols = [symbol for symbol in range(9) if symbol != target_symbol]
        rng.shuffle(wrong_symbols)
        expected_selected = rng.randrange(9)
        pockets = []
        wrong_at = 0
        for pocket_idx in range(9):
            if pocket_idx == expected_selected:
                pockets.append(target_symbol)
            else:
                pockets.append(wrong_symbols[wrong_at])
                wrong_at += 1
        formula_pockets = []
        for formula_idx in range(len(FAMILIES)):
            formula_symbol = family_target(formula_idx, board)
            formula_pockets.append(pockets.index(formula_symbol))
        rows.append(
            {
                "id": idx,
                "split": split,
                "family": FAMILIES[family_idx],
                "family_index": family_idx,
                "board": board,
                "target_symbol": target_symbol,
                "pockets": pockets,
                "expected_selected": expected_selected,
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


def initial_gate_model(rng):
    return {
        "matrix": [[rng.uniform(-0.05, 0.05) for _ in range(5)] for _ in range(5)],
        "gate_bias": [rng.uniform(-0.02, 0.02) for _ in range(5)],
        "pocket_bias": [rng.uniform(-0.01, 0.01) for _ in range(9)],
    }


def clone_gate(model):
    return {
        "matrix": [row[:] for row in model["matrix"]],
        "gate_bias": model["gate_bias"][:],
        "pocket_bias": model["pocket_bias"][:],
    }


def initial_vector_model(rng):
    return {
        "weights": [rng.uniform(-0.05, 0.05) for _ in range(5)],
        "pocket_bias": [rng.uniform(-0.01, 0.01) for _ in range(9)],
    }


def clone_vector(model):
    return {"weights": model["weights"][:], "pocket_bias": model["pocket_bias"][:]}


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


def predict_gate(model, row, family_override=None):
    family_idx = row["family_index"] if family_override is None else family_override
    weights = model["matrix"][family_idx]
    scores = model["pocket_bias"][:]
    for formula_idx, pocket_idx in enumerate(row["formula_pockets"]):
        scores[pocket_idx] += weights[formula_idx] + model["gate_bias"][formula_idx]
    return top_two(scores)


def predict_vector(model, row):
    scores = model["pocket_bias"][:]
    for formula_idx, pocket_idx in enumerate(row["formula_pockets"]):
        scores[pocket_idx] += model["weights"][formula_idx]
    return top_two(scores)


def predict_oracle(row):
    return row["formula_pockets"][row["family_index"]], 1.0


def predict_explicit(row):
    return row["pockets"].index(row["target_symbol"]), 1.0


def fitness_gate(model, rows, shuffled=False):
    correct = 0
    margin_sum = 0.0
    for row in rows:
        family_override = ((row["family_index"] + 1) % 5) if shuffled else None
        pred, margin = predict_gate(model, row, family_override=family_override)
        if pred == row["expected_selected"]:
            correct += 1
        margin_sum += margin
    return correct * 10000.0 + margin_sum


def fitness_vector(model, rows):
    correct = 0
    margin_sum = 0.0
    for row in rows:
        pred, margin = predict_vector(model, row)
        if pred == row["expected_selected"]:
            correct += 1
        margin_sum += margin
    return correct * 10000.0 + margin_sum


def gate_mutation_specs(rng, population, step):
    specs = []
    for row_idx in range(5):
        for col_idx in range(5):
            specs.append(("gate_weight_delta", row_idx, col_idx, step))
            specs.append(("gate_weight_delta", row_idx, col_idx, -step))
    while len(specs) < population:
        op = MUTATION_TYPES[rng.randrange(len(MUTATION_TYPES))]
        specs.append((op,))
    rng.shuffle(specs)
    return specs[:population]


def apply_gate_mutation(rng, model, spec, step):
    candidate = clone_gate(model)
    op = spec[0]
    if op == "gate_weight_delta":
        row_idx = spec[1] if len(spec) > 1 else rng.randrange(5)
        col_idx = spec[2] if len(spec) > 2 else rng.randrange(5)
        delta = spec[3] if len(spec) > 3 else rng.uniform(-step, step)
        candidate["matrix"][row_idx][col_idx] += delta
    elif op == "gate_row_delta":
        row_idx = rng.randrange(5)
        for col_idx in range(5):
            candidate["matrix"][row_idx][col_idx] += rng.uniform(-step, step)
    elif op == "gate_column_delta":
        col_idx = rng.randrange(5)
        delta = rng.uniform(-step, step)
        for row_idx in range(5):
            candidate["matrix"][row_idx][col_idx] += delta
    elif op == "gate_bias_delta":
        candidate["gate_bias"][rng.randrange(5)] += rng.uniform(-step, step)
    elif op == "pocket_bias_delta":
        candidate["pocket_bias"][rng.randrange(9)] += rng.uniform(-0.25 * step, 0.25 * step)
    elif op == "gate_row_swap":
        a = rng.randrange(5)
        b = rng.randrange(5)
        candidate["matrix"][a], candidate["matrix"][b] = candidate["matrix"][b], candidate["matrix"][a]
    elif op == "gate_column_swap":
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
    for col_idx in range(5):
        specs.append(("vector_weight_delta", col_idx, step))
        specs.append(("vector_weight_delta", col_idx, -step))
    vector_ops = ["vector_weight_delta", "pocket_bias_delta", "prune_small_weights"]
    while len(specs) < population:
        specs.append((vector_ops[rng.randrange(len(vector_ops))],))
    rng.shuffle(specs)
    return specs[:population]


def apply_vector_mutation(rng, model, spec, step):
    candidate = clone_vector(model)
    op = spec[0]
    if op == "vector_weight_delta":
        col_idx = spec[1] if len(spec) > 1 else rng.randrange(5)
        delta = spec[2] if len(spec) > 2 else rng.uniform(-step, step)
        candidate["weights"][col_idx] += delta
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
        buckets[row["family_index"]].append(row)
    out = []
    cursor = 0
    while len(out) < limit:
        bucket = buckets[cursor % 5]
        if bucket:
            out.append(bucket[(cursor // 5) % len(bucket)])
        cursor += 1
    return out


def accuracy_only_gate(model, rows):
    return sum(int(predict_gate(model, row)[0] == row["expected_selected"]) for row in rows) / max(1, len(rows))


def train_gate(train_rows, seed, generations, population):
    rng = random.Random(stable_mix(seed, 101, generations, population))
    fit_rows = balanced_subset(train_rows, min(180, len(train_rows)))
    best = initial_gate_model(rng)
    best_fit = fitness_gate(best, fit_rows)
    accepted = Counter({op: 0 for op in MUTATION_TYPES})
    rejected = Counter({op: 0 for op in MUTATION_TYPES})
    trace = []
    convergence_generation = None
    perfect_since = None
    generations_executed = 0
    for generation in range(generations):
        generations_executed = generation + 1
        step = max(0.05, 1.0 * (1.0 - (generation / max(1, generations))))
        winner = None
        winner_op = None
        winner_fit = -1.0e30
        records = []
        for spec in gate_mutation_specs(rng, population, step):
            candidate, op = apply_gate_mutation(rng, best, spec, step)
            fit = fitness_gate(candidate, fit_rows)
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
        for fit, op, _candidate in records:
            if not (accepted_this_generation and op == winner_op and fit == winner_fit):
                rejected[op] += 1
        if generation % max(1, generations // 25) == 0 or generation == generations - 1:
            train_metrics = evaluate_rows(train_rows, lambda row: predict_gate(best, row))
            trace.append(
                {
                    "generation": generation,
                    "train_accuracy": train_metrics["accuracy"],
                    "fitness": best_fit,
                }
            )
            if train_metrics["accuracy"] >= 0.95 and convergence_generation is None:
                convergence_generation = generation
            if train_metrics["accuracy"] == 1.0:
                if perfect_since is None:
                    perfect_since = generation
                elif generation - perfect_since >= max(20, generations // 20):
                    break
            else:
                perfect_since = None
    return best, {
        "accepted_mutations_by_type": dict(accepted),
        "rejected_mutations_by_type": dict(rejected),
        "mutation_acceptance_rate": sum(accepted.values()) / max(1, sum(accepted.values()) + sum(rejected.values())),
        "convergence_generation": convergence_generation,
        "generations_requested": generations,
        "generations_executed": generations_executed,
        "trace": trace,
    }


def train_vector(train_rows, seed, generations, population):
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
        if stale >= 30:
            break
    return best, {"generations_requested": limit, "generations_executed": generation + 1}


def evaluate_rows(rows, predictor):
    correct = 0
    per_family_total = Counter()
    per_family_correct = Counter()
    confusion = [[0 for _ in range(9)] for _ in range(9)]
    margins = []
    low_margin_count = 0
    low_margin_errors = 0
    for row in rows:
        pred, margin = predictor(row)
        truth = row["expected_selected"]
        is_correct = pred == truth
        correct += int(is_correct)
        per_family_total[row["family"]] += 1
        per_family_correct[row["family"]] += int(is_correct)
        confusion[truth][pred] += 1
        margins.append(margin)
        if margin < 0.1:
            low_margin_count += 1
            low_margin_errors += int(not is_correct)
    return {
        "accuracy": correct / max(1, len(rows)),
        "per_family_accuracy": {
            family: per_family_correct[family] / max(1, per_family_total[family])
            for family in FAMILIES
        },
        "pocket_confusion_matrix": confusion,
        "median_score_margin": statistics.median(margins) if margins else 0.0,
        "low_margin_error_rate": low_margin_errors / max(1, low_margin_count),
    }


def flatten_metrics(train, test, ood):
    return {
        "train_accuracy": train["accuracy"],
        "test_accuracy": test["accuracy"],
        "ood_accuracy": ood["accuracy"],
        "per_family_accuracy": test["per_family_accuracy"],
        "pocket_confusion_matrix": test["pocket_confusion_matrix"],
        "median_score_margin": test["median_score_margin"],
        "low_margin_error_rate": test["low_margin_error_rate"],
    }


def dataset_invariants(rows):
    duplicate = 0
    missing = 0
    selected = 0
    for row in rows:
        count = sum(1 for symbol in row["pockets"] if symbol == row["target_symbol"])
        duplicate += int(count > 1)
        missing += int(count == 0)
        selected += int(row["pockets"][row["expected_selected"]] == row["target_symbol"])
    return {
        "duplicate_target_pocket_rate": duplicate / max(1, len(rows)),
        "missing_target_pocket_rate": missing / max(1, len(rows)),
        "expected_selected_points_to_target_rate": selected / max(1, len(rows)),
    }


def oracle_accuracy(rows):
    return sum(
        int(row["formula_pockets"][row["family_index"]] == row["expected_selected"])
        for row in rows
    ) / max(1, len(rows))


def gate_identity(model):
    effective = [
        [model["matrix"][row_idx][col_idx] + model["gate_bias"][col_idx] for col_idx in range(5)]
        for row_idx in range(5)
    ]
    mapping = {}
    aligned = 0
    diag_probs = []
    off_probs = []
    entropies = []
    for row_idx, weights in enumerate(effective):
        argmax_idx = max(range(5), key=lambda col_idx: weights[col_idx])
        mapping[FAMILIES[row_idx]] = FAMILIES[argmax_idx]
        aligned += int(argmax_idx == row_idx)
        peak = max(weights)
        exps = [math.exp(value - peak) for value in weights]
        total = sum(exps)
        probs = [value / total for value in exps]
        diag_probs.append(probs[row_idx])
        off_probs.append(sum(probs[col_idx] for col_idx in range(5) if col_idx != row_idx))
        entropies.append(-sum(prob * math.log(max(prob, 1.0e-12)) for prob in probs) / math.log(5))
    rounded_effective = [[round(value, 6) for value in row] for row in effective]
    return {
        "mapping": mapping,
        "alignment_score": aligned / 5,
        "diagonal_gate_mass": statistics.mean(diag_probs),
        "off_diagonal_gate_mass": statistics.mean(off_probs),
        "gate_entropy": statistics.mean(entropies),
        "effective_matrix": rounded_effective,
    }


def run_job(job, config):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    seed = job["seed"]
    arm = job["arm"]
    train_rows = make_rows(seed, "train", config["train_rows_per_seed"])
    test_rows = make_rows(seed, "test", config["test_rows_per_seed"])
    ood_rows = make_rows(seed, "ood", config["ood_rows_per_seed"])
    started = time.time()

    extra = {}
    if arm in {"MONOLITHIC_FORMULA_BASELINE", "NO_FAMILY_INPUT_CONTROL"}:
        model, training = train_vector(
            train_rows,
            stable_mix(seed, ARMS.index(arm), 1),
            config["generations"],
            config["population"],
        )
        predictor = lambda row: predict_vector(model, row)
        extra["vector_model"] = model
        extra["training"] = training
    elif arm == "ORACLE_GATED_RULE_FORMULA_UPPER_BOUND":
        predictor = predict_oracle
    elif arm == "EXPLICIT_TARGET_STATE_UPPER_BOUND":
        predictor = predict_explicit
    elif arm in {"MUTABLE_LEARNED_ROUTER_GATE", "SHUFFLED_GATE_CONTROL"}:
        model, mutation = train_gate(
            train_rows,
            stable_mix(seed, ARMS.index(arm), 2),
            config["generations"],
            config["population"],
        )
        if arm == "SHUFFLED_GATE_CONTROL":
            predictor = lambda row: predict_gate(model, row, family_override=(row["family_index"] + 1) % 5)
            extra["shuffle_mapping"] = {FAMILIES[idx]: FAMILIES[(idx + 1) % 5] for idx in range(5)}
        else:
            predictor = lambda row: predict_gate(model, row)
            extra["gate_model"] = model
            extra["gate_identity"] = gate_identity(model)
        extra["mutation"] = mutation
    else:
        raise ValueError(f"unknown arm: {arm}")

    train = evaluate_rows(train_rows, predictor)
    test = evaluate_rows(test_rows, predictor)
    ood = evaluate_rows(ood_rows, predictor)
    result = {
        "seed": seed,
        "arm": arm,
        "ok": True,
        "wall_clock_sec": time.time() - started,
        **flatten_metrics(train, test, ood),
        **extra,
    }
    return result


def aggregate_arm(rows, failed_seed_count):
    if not rows:
        return {
            "train_accuracy": 0.0,
            "test_accuracy": 0.0,
            "ood_accuracy": 0.0,
            "min_seed_test_accuracy": 0.0,
            "min_seed_ood_accuracy": 0.0,
            "per_family_accuracy": {family: 0.0 for family in FAMILIES},
            "pocket_confusion_matrix": [[0 for _ in range(9)] for _ in range(9)],
            "median_score_margin": 0.0,
            "low_margin_error_rate": 1.0,
            "failed_seed_count": failed_seed_count,
        }
    return {
        "train_accuracy": statistics.mean(row["train_accuracy"] for row in rows),
        "test_accuracy": statistics.mean(row["test_accuracy"] for row in rows),
        "ood_accuracy": statistics.mean(row["ood_accuracy"] for row in rows),
        "min_seed_test_accuracy": min(row["test_accuracy"] for row in rows),
        "min_seed_ood_accuracy": min(row["ood_accuracy"] for row in rows),
        "per_family_accuracy": {
            family: statistics.mean(row["per_family_accuracy"][family] for row in rows)
            for family in FAMILIES
        },
        "pocket_confusion_matrix": [
            [sum(row["pocket_confusion_matrix"][i][j] for row in rows) for j in range(9)]
            for i in range(9)
        ],
        "median_score_margin": statistics.median(row["median_score_margin"] for row in rows),
        "low_margin_error_rate": statistics.mean(row["low_margin_error_rate"] for row in rows),
        "failed_seed_count": failed_seed_count,
    }


def variance(values):
    return statistics.pvariance(values) if values else 0.0


def build_gate_reports(learned_rows):
    gate_items = {str(row["seed"]): row["gate_identity"] for row in learned_rows}
    alignment_scores = [item["alignment_score"] for item in gate_items.values()]
    if not gate_items:
        return {
            "gate_argmax_mapping_by_seed": {},
            "gate_identity_alignment_score_mean": 0.0,
            "gate_identity_alignment_score_min": 0.0,
            "diagonal_gate_mass_mean": 0.0,
            "off_diagonal_gate_mass_mean": 0.0,
            "gate_entropy_mean": 0.0,
            "effective_gate_matrix_by_seed": {},
        }
    return {
        "gate_argmax_mapping_by_seed": {seed: item["mapping"] for seed, item in gate_items.items()},
        "gate_identity_alignment_score_mean": statistics.mean(alignment_scores),
        "gate_identity_alignment_score_min": min(alignment_scores),
        "diagonal_gate_mass_mean": statistics.mean(item["diagonal_gate_mass"] for item in gate_items.values()),
        "off_diagonal_gate_mass_mean": statistics.mean(item["off_diagonal_gate_mass"] for item in gate_items.values()),
        "gate_entropy_mean": statistics.mean(item["gate_entropy"] for item in gate_items.values()),
        "effective_gate_matrix_by_seed": {seed: item["effective_matrix"] for seed, item in gate_items.items()},
    }


def build_mutation_report(learned_rows):
    accepted = Counter({op: 0 for op in MUTATION_TYPES})
    rejected = Counter({op: 0 for op in MUTATION_TYPES})
    convergence = []
    generations_executed = {}
    for row in learned_rows:
        mutation = row["mutation"]
        accepted.update(mutation["accepted_mutations_by_type"])
        rejected.update(mutation["rejected_mutations_by_type"])
        if mutation["convergence_generation"] is not None:
            convergence.append(mutation["convergence_generation"])
        generations_executed[str(row["seed"])] = mutation["generations_executed"]
    return {
        "accepted_mutations_by_type": dict(accepted),
        "rejected_mutations_by_type": dict(rejected),
        "mutation_acceptance_rate": sum(accepted.values()) / max(1, sum(accepted.values()) + sum(rejected.values())),
        "convergence_generation_median": statistics.median(convergence) if convergence else None,
        "convergence_generation_by_seed": {
            str(row["seed"]): row["mutation"]["convergence_generation"] for row in learned_rows
        },
        "generations_executed_by_seed": generations_executed,
    }


def strip_large_job_fields(row):
    out = dict(row)
    out.pop("gate_model", None)
    out.pop("vector_model", None)
    return out


def d38_upstream_manifest():
    paths = [
        "docs/research/D38_LEARNED_CONDITIONING_ROUTER_FIELD_PROOF_CONTRACT.md",
        "docs/research/D38_LEARNED_CONDITIONING_ROUTER_FIELD_PROOF_RESULT.md",
        "scripts/probes/run_d38_learned_conditioning_router_field_proof.py",
        "scripts/probes/run_d38_learned_conditioning_router_field_proof_check.py",
    ]
    artifact_decision = Path("target/pilot_wave/d38_learned_conditioning_router_field_proof/smoke/decision.json")
    payload = {
        "required_files": {path: Path(path).exists() for path in paths},
        "artifact_decision_path": artifact_decision.as_posix(),
        "artifact_decision_present": artifact_decision.exists(),
        "known_prior_commit": "7462e5c57bd970bbc7c611794882391f3cec6a91",
        "known_prior_branch": "work",
        "known_prior_pushed_to_origin": False,
    }
    if artifact_decision.exists():
        payload["artifact_decision"] = json.loads(artifact_decision.read_text())
    return payload


def make_report_markdown(decision_payload, arm_reports, deltas, artifact_path):
    lines = [
        "# D39 Learned Router Layer Scale Confirm",
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
        lines.append(
            f"| {arm} | {report['train_accuracy']:.4f} | {report['test_accuracy']:.4f} | {report['ood_accuracy']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Deltas",
            "",
            f"- monolithic_vs_learned_test_delta = {deltas['monolithic_vs_learned_test_delta']:.4f}",
            f"- learned_vs_shuffled_test_delta = {deltas['learned_vs_shuffled_test_delta']:.4f}",
            f"- learned_vs_no_family_test_delta = {deltas['learned_vs_no_family_test_delta']:.4f}",
            "",
            "## Boundary",
            "",
            "A positive D39 proves only that the learned conditioning/router field scale-confirms on a controlled known-rule symbolic pocket task. It does not prove hidden-rule Raven reasoning, natural-language reasoning, DNA/genome success, Raven solved, architecture superiority, consciousness, or general intelligence.",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    started = time.time()
    out = Path(args.out)
    if "target/pilot_wave/d39_learned_router_layer_scale_confirm/" not in out.as_posix():
        raise SystemExit("--out must stay under target/pilot_wave/d39_learned_router_layer_scale_confirm/")
    out.mkdir(parents=True, exist_ok=True)
    progress_path = out / "progress.jsonl"
    if progress_path.exists():
        progress_path.unlink()
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    jobs = [{"seed": seed, "arm": arm} for seed in seeds for arm in ARMS]
    config = {
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

    write_json(
        out / "queue.json",
        {
            "jobs": jobs,
            "seeds": seeds,
            "arms": ARMS,
            "workers": workers,
            "cpu_target": args.cpu_target,
            "heartbeat_sec": args.heartbeat_sec,
        },
    )

    all_rows = []
    test_rows = []
    ood_rows = []
    for seed in seeds:
        for split, count in [
            ("train", args.train_rows_per_seed),
            ("test", args.test_rows_per_seed),
            ("ood", args.ood_rows_per_seed),
        ]:
            rows = make_rows(seed, split, count)
            all_rows.extend(rows)
            if split == "test":
                test_rows.extend(rows)
            if split == "ood":
                ood_rows.extend(rows)

    invariants = dataset_invariants(all_rows)
    write_json(out / "dataset_invariant_report.json", invariants)
    ood_audit = {
        "known_rule_oracle_test_accuracy": oracle_accuracy(test_rows),
        "known_rule_oracle_ood_accuracy": oracle_accuracy(ood_rows),
        "ood_label_rule_changed": False,
    }
    write_json(out / "ood_rule_invariance_audit.json", ood_audit)
    write_json(
        out / "dataset_manifest.json",
        {
            "families": FAMILIES,
            "formulas": FORMULA_DESC,
            "seeds": seeds,
            "train_rows_per_seed": args.train_rows_per_seed,
            "test_rows_per_seed": args.test_rows_per_seed,
            "ood_rows_per_seed": args.ood_rows_per_seed,
            "pocket_count": 9,
            "target_pockets_per_row": 1,
            "ood_transform": "affine_mod9_plus_position_offset_then_recompute_known_formula_target",
        },
    )
    write_json(out / "d38_upstream_manifest.json", d38_upstream_manifest())

    results = []
    failures = []
    append_jsonl(progress_path, {"completed_jobs": 0, "failed_jobs": 0, "total_jobs": len(jobs), "elapsed_sec": 0.0})
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_job, job, config): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                result = future.result()
                results.append(result)
                job_dir = out / "job_results" / job["arm"] / f"seed_{job['seed']}"
                write_json(job_dir / "result.json", strip_large_job_fields(result))
            except Exception as exc:
                failure = {**job, "ok": False, "error": str(exc)}
                failures.append(failure)
                write_json(out / "job_results" / job["arm"] / f"seed_{job['seed']}" / "error.json", failure)
            append_jsonl(
                progress_path,
                {
                    "completed_jobs": len(results),
                    "failed_jobs": len(failures),
                    "total_jobs": len(jobs),
                    "elapsed_sec": time.time() - started,
                },
            )

    arm_reports = {}
    for arm in ARMS:
        arm_rows = [row for row in results if row["arm"] == arm]
        failed_seed_count = sum(1 for failure in failures if failure["arm"] == arm)
        arm_reports[arm] = aggregate_arm(arm_rows, failed_seed_count)
        write_json(out / REPORT_FILES[arm], arm_reports[arm])

    learned_rows = [row for row in results if row["arm"] == "MUTABLE_LEARNED_ROUTER_GATE"]
    gate_report = build_gate_reports(learned_rows)
    mutation_report = build_mutation_report(learned_rows)
    learned_report = dict(arm_reports["MUTABLE_LEARNED_ROUTER_GATE"])
    learned_report.update(
        {
            "learned_gate_train_accuracy": learned_report["train_accuracy"],
            "learned_gate_test_accuracy": learned_report["test_accuracy"],
            "learned_gate_ood_accuracy": learned_report["ood_accuracy"],
            "min_seed_learned_gate_test_accuracy": learned_report["min_seed_test_accuracy"],
            "min_seed_learned_gate_ood_accuracy": learned_report["min_seed_ood_accuracy"],
            "gate_identity_alignment_score_mean": gate_report["gate_identity_alignment_score_mean"],
            "gate_identity_alignment_score_min": gate_report["gate_identity_alignment_score_min"],
            "diagonal_gate_mass_mean": gate_report["diagonal_gate_mass_mean"],
            "off_diagonal_gate_mass_mean": gate_report["off_diagonal_gate_mass_mean"],
            "gate_argmax_mapping_by_seed": gate_report["gate_argmax_mapping_by_seed"],
            "gate_entropy_mean": gate_report["gate_entropy_mean"],
            "accepted_mutations_by_type": mutation_report["accepted_mutations_by_type"],
            "rejected_mutations_by_type": mutation_report["rejected_mutations_by_type"],
            "mutation_acceptance_rate": mutation_report["mutation_acceptance_rate"],
            "convergence_generation_median": mutation_report["convergence_generation_median"],
            "seed_variance": {
                "test_accuracy_variance": variance([row["test_accuracy"] for row in learned_rows]),
                "ood_accuracy_variance": variance([row["ood_accuracy"] for row in learned_rows]),
            },
        }
    )
    write_json(out / "mutable_learned_router_gate_report.json", learned_report)
    write_json(out / "gate_matrix_report.json", gate_report)
    write_json(
        out / "gate_identity_alignment_report.json",
        {
            "gate_identity_alignment_score_mean": gate_report["gate_identity_alignment_score_mean"],
            "gate_identity_alignment_score_min": gate_report["gate_identity_alignment_score_min"],
            "gate_argmax_mapping_by_seed": gate_report["gate_argmax_mapping_by_seed"],
        },
    )
    mappings = list(gate_report["gate_argmax_mapping_by_seed"].values())
    stability_pairs = []
    for left_idx in range(len(mappings)):
        for right_idx in range(left_idx + 1, len(mappings)):
            left = mappings[left_idx]
            right = mappings[right_idx]
            stability_pairs.append(sum(int(left[family] == right[family]) for family in FAMILIES) / 5)
    write_json(
        out / "gate_stability_report.json",
        {
            "pairwise_mapping_agreement_mean": statistics.mean(stability_pairs) if stability_pairs else 0.0,
            "identity_mapping_seed_count": sum(
                int(all(mapping[family] == family for family in FAMILIES))
                for mapping in gate_report["gate_argmax_mapping_by_seed"].values()
            ),
            "seed_count": len(learned_rows),
        },
    )
    seed_variance_report = {
        arm: {
            "test_accuracy_variance": variance([row["test_accuracy"] for row in results if row["arm"] == arm]),
            "ood_accuracy_variance": variance([row["ood_accuracy"] for row in results if row["arm"] == arm]),
        }
        for arm in ARMS
    }
    write_json(out / "seed_variance_report.json", seed_variance_report)
    write_json(
        out / "convergence_report.json",
        {
            "convergence_generation_median": mutation_report["convergence_generation_median"],
            "convergence_generation_by_seed": mutation_report["convergence_generation_by_seed"],
            "generations_executed_by_seed": mutation_report["generations_executed_by_seed"],
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
                }
                for row in results
            ],
            "failed_jobs": failures,
        },
    )
    write_json(
        out / "per_family_report.json",
        {arm: arm_reports[arm]["per_family_accuracy"] for arm in ARMS},
    )
    write_json(
        out / "pocket_confusion_matrix.json",
        {arm: arm_reports[arm]["pocket_confusion_matrix"] for arm in ARMS},
    )
    write_json(
        out / "score_margin_report.json",
        {
            arm: {
                "median_score_margin": arm_reports[arm]["median_score_margin"],
                "low_margin_error_rate": arm_reports[arm]["low_margin_error_rate"],
            }
            for arm in ARMS
        },
    )

    deltas = {
        "monolithic_vs_learned_test_delta": arm_reports["MUTABLE_LEARNED_ROUTER_GATE"]["test_accuracy"] - arm_reports["MONOLITHIC_FORMULA_BASELINE"]["test_accuracy"],
        "learned_vs_shuffled_test_delta": arm_reports["MUTABLE_LEARNED_ROUTER_GATE"]["test_accuracy"] - arm_reports["SHUFFLED_GATE_CONTROL"]["test_accuracy"],
        "learned_vs_no_family_test_delta": arm_reports["MUTABLE_LEARNED_ROUTER_GATE"]["test_accuracy"] - arm_reports["NO_FAMILY_INPUT_CONTROL"]["test_accuracy"],
    }
    arm_comparison_report = {"deltas": deltas, "arms": arm_reports}
    write_json(out / "arm_comparison_report.json", arm_comparison_report)

    learned = arm_reports["MUTABLE_LEARNED_ROUTER_GATE"]
    oracle = arm_reports["ORACLE_GATED_RULE_FORMULA_UPPER_BOUND"]
    explicit = arm_reports["EXPLICIT_TARGET_STATE_UPPER_BOUND"]
    shuffled = arm_reports["SHUFFLED_GATE_CONTROL"]
    no_family = arm_reports["NO_FAMILY_INPUT_CONTROL"]
    if (
        invariants["duplicate_target_pocket_rate"] != 0.0
        or invariants["missing_target_pocket_rate"] != 0.0
        or invariants["expected_selected_points_to_target_rate"] != 1.0
    ):
        decision = "d39_dataset_invariant_failure"
        verdict = "D39_DATASET_INVARIANT_FAILURE"
        next_step = "D39B_DATASET_REPAIR"
    elif ood_audit["known_rule_oracle_test_accuracy"] != 1.0 or ood_audit["known_rule_oracle_ood_accuracy"] != 1.0:
        decision = "d39_ood_rule_invariance_failure"
        verdict = "D39_OOD_RULE_INVARIANCE_FAILURE"
        next_step = "D39C_OOD_REPAIR"
    elif (
        learned["test_accuracy"] >= 0.95
        and learned["ood_accuracy"] >= 0.95
        and learned["min_seed_test_accuracy"] >= 0.90
        and learned["min_seed_ood_accuracy"] >= 0.90
        and oracle["test_accuracy"] >= 0.99
        and oracle["ood_accuracy"] >= 0.99
        and explicit["test_accuracy"] >= 0.99
        and explicit["ood_accuracy"] >= 0.99
        and deltas["monolithic_vs_learned_test_delta"] >= 0.45
        and deltas["learned_vs_shuffled_test_delta"] >= 0.70
        and deltas["learned_vs_no_family_test_delta"] >= 0.45
        and shuffled["test_accuracy"] <= 0.25
        and no_family["test_accuracy"] <= 0.50
    ):
        decision = "learned_conditioning_router_field_scale_confirmed"
        verdict = "D39_LEARNED_CONDITIONING_ROUTER_FIELD_SCALE_CONFIRMED"
        next_step = "D40_RULE_FAMILY_INFERENCE_ROUTER_PROTOTYPE"
    elif learned["test_accuracy"] >= 0.95 and learned["ood_accuracy"] >= 0.95:
        decision = "learned_router_mean_positive_seed_variance_edge"
        verdict = "D39_LEARNED_ROUTER_SEED_VARIANCE_EDGE"
        next_step = "D39S_SEED_VARIANCE_HARDENING_PLAN"
    elif deltas["monolithic_vs_learned_test_delta"] > 0.0:
        decision = "learned_router_scale_partial_signal"
        verdict = "D39_LEARNED_ROUTER_SCALE_PARTIAL_SIGNAL"
        next_step = "D39L_LEARNED_ROUTER_OPTIMIZATION_PLAN"
    else:
        decision = "learned_router_scale_not_confirmed"
        verdict = "D39_LEARNED_ROUTER_SCALE_NOT_CONFIRMED"
        next_step = "D40_FEATURE_SPACE_DIAGNOSTIC"

    decision_payload = {
        "decision": decision,
        "verdict": verdict,
        "next": next_step,
        "boundary": "controlled known-rule symbolic pocket task only",
        "non_claims": {
            "hidden_rule_raven_reasoning": False,
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
        "arms": arm_reports,
        "deltas": deltas,
        "learned_router_gate": learned_report,
        "gate_identity": gate_report,
        "mutation": mutation_report,
        "seed_variance": seed_variance_report,
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
    (out / "report.md").write_text(make_report_markdown(decision_payload, arm_reports, deltas, out.as_posix()))
    print(json.dumps({"status": "ok", "decision": decision, "verdict": verdict, "next": next_step, "failed_jobs": len(failures)}, indent=2))


if __name__ == "__main__":
    main()
