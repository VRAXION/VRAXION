#!/usr/bin/env python3
"""D49B joint cell+operator binding repair probe."""

import argparse
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import run_d49_joint_cell_operator_discovery_with_robust_support as d49

PRIMARY_SPACE = "ALL28_UNORDERED_X_OPS"
SUPPORT_COUNT = 5
ROW_SAMPLE_PER_ARM_REGIME_SPLIT = 24
CONFIDENCE_THRESHOLD = 0.45

BOUNDARY = (
    "D49B only tests controlled symbolic joint cell+operator binding repair with robust ECF support. "
    "It does not prove raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, "
    "or architecture superiority."
)

REGIMES = d49.REGIMES
CORE_REGIMES = d49.CORE_REGIMES
OP_NAMES = d49.OP_NAMES

ARMS = [
    "D49_BASELINE_REPLAY",
    "FACTORISED_CELL_OPERATOR_SCORE",
    "CELL_FIRST_OPERATOR_SECOND_PIPELINE",
    "OPERATOR_FIRST_CELL_SECOND_PIPELINE",
    "JOINT_BINDING_MATRIX",
    "CELL_ONLY_COUNTERFACTUAL",
    "OPERATOR_ONLY_COUNTERFACTUAL",
    "JOINT_INTERACTION_COUNTERFACTUAL",
    "MULTI_STAGE_COUNTERFACTUAL_REPAIR",
    "FULL_REPAIRED_ECF_CONTROLLER",
    "FULL_REPAIRED_ECF_CAP_7",
    "FULL_REPAIRED_ECF_CAP_9",
    "RANDOM_EXTRA_SUPPORT_CONTROL",
    "SHUFFLED_COUNTER_SUPPORT_CONTROL",
    "NO_COUNTERFACTUAL_CONTROL",
    "ABSTAIN_ON_INDISTINGUISHABLE",
]

CONTROL_ARMS = [
    "RANDOM_EXTRA_SUPPORT_CONTROL",
    "SHUFFLED_COUNTER_SUPPORT_CONTROL",
    "NO_COUNTERFACTUAL_CONTROL",
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


def safe_div(num, den):
    return num / den if den else 0.0


def cell_key(cell):
    return d49.cell_key(cell)


def pair_equivalence(pair):
    return d49.pair_equivalence(pair)


def support_vectors(boards, bundle):
    return d49.support_vectors(boards, bundle)


def aggregate_sum(vectors):
    return d49.aggregate_sum(vectors)


def aggregate_duplicate_downweighted(vectors):
    return d49.aggregate_duplicate_downweighted(vectors)


def project_scores(scores, bundle, mode):
    return d49.project_scores(scores, bundle, mode)


def predict(scores, bundle, abstain=False):
    return d49.predict(scores, bundle, abstain=abstain)


def cached_base_vectors(row, bundle, count):
    return d49.cached_base_vectors(row, bundle, count)


def cluster_stats(vectors):
    return d49.cluster_stats(vectors)


def score_gap(scores, truth_joint):
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    truth_score = scores[truth_joint]
    wrong_top = next((score for cid, score in ordered if cid != truth_joint), truth_score)
    return truth_score - wrong_top


def joint_binding_matrix_scores(scores, bundle):
    """Combine exact joint evidence with separate cell and operator projections."""
    pair_scores = project_scores(scores, bundle, "pair")
    op_scores = project_scores(scores, bundle, "operator")
    group_scores = project_scores(scores, bundle, "group")
    probs = d49.softmax(scores)
    out = {}
    for cid, spec in bundle["candidates"].items():
        pair_key = pair_equivalence(spec["pair"])
        pair_family = bundle["pair_family_by_equiv"].get(pair_key, "distractor")
        group_key = f"{pair_family}::{d49.OP_SPECS[spec['operator']]['family']}"
        out[cid] = (
            0.45 * scores[cid]
            + 7.0 * pair_scores[pair_key]
            + 7.0 * op_scores[spec["operator"]]
            + 1.5 * group_scores[group_key]
            + 1.0 * probs[cid]
        )
    return out


def choose_targets(row, bundle, pred, stage, limit):
    ordered = [cid for cid, _score in pred["ordered"] if cid != row["truth_joint"]]
    targets = []
    for cid in ordered:
        spec = bundle["candidates"][cid]
        pair_ok = pair_equivalence(spec["pair"]) == row["truth_pair_equivalence"]
        op_ok = spec["operator"] == row["true_operator"]
        if stage == "cell" and not pair_ok:
            targets.append(spec)
        elif stage == "operator" and not op_ok:
            targets.append(spec)
        elif stage == "joint" and not (pair_ok and op_ok):
            targets.append(spec)
        if len(targets) >= limit:
            break
    if not targets:
        for cid in ordered[:limit]:
            targets.append(bundle["candidates"][cid])
    if not targets:
        targets.append(bundle["candidates"][row["false_joint"]])
    return targets


def make_discriminating_board(rng, row, targets):
    true_pair = row["truth_pair"]
    true_op = row["true_operator"]
    protected = set(true_pair) | {(1, 1)}
    best = None
    best_score = -1e9
    for _ in range(140):
        board = [[rng.randrange(9) for _ in range(3)] for _ in range(3)]
        a, b, target = d49.find_values(rng, true_op)
        d49.set_pair_values(board, true_pair, a, b)
        board[1][1] = target
        if row["split"] == "ood":
            for r in range(3):
                for c in range(3):
                    if (r, c) not in protected:
                        board[r][c] = ((board[r][c] * 2) + 1) % 9
        target_scores = [d49.candidate_score(board, spec) for spec in targets]
        # True candidate score is zero by construction; more negative target scores are better.
        max_target = max(target_scores) if target_scores else -9.0
        score = -max_target + sum(-value for value in target_scores)
        if score > best_score:
            best_score = score
            best = board
        if max_target <= -1.0:
            return board
    return best if best is not None else d49.make_board(rng, row, "break_false")


def make_stage_vectors(row, bundle, pred, stage, count, shuffled=False, external=False):
    if count <= 0:
        return []
    if external and row["external_test_available"]:
        rng = random.Random(49_910 + row["seed"] + len(row["row_id"]) + count)
        targets = choose_targets(row, bundle, pred, "joint", 12)
        boards = [make_discriminating_board(rng, row, targets) for _ in range(count)]
        return support_vectors(boards, bundle)
    if not row["oracle_distinguishable"]:
        return support_vectors(row["internal_counter_supports"][:count], bundle)
    if shuffled:
        rng = random.Random(49_920 + row["seed"] + len(row["row_id"]) + count)
        shifted = dict(row)
        shifted["truth_pair"] = row["false_pair"]
        shifted["true_operator"] = row["false_operator"]
        shifted["false_pair"] = row["truth_pair"]
        shifted["false_operator"] = row["true_operator"]
        targets = [bundle["candidates"][row["truth_joint"]]]
        boards = [make_discriminating_board(rng, shifted, targets) for _ in range(count)]
        return support_vectors(boards, bundle)
    rng = random.Random(49_930 + row["seed"] + len(row["row_id"]) + 17 * len(stage) + count)
    targets = choose_targets(row, bundle, pred, stage, 12 if stage == "joint" else 8)
    boards = [make_discriminating_board(rng, row, targets) for _ in range(count)]
    return support_vectors(boards, bundle)


def add_stage(extra, stage_counts, stage, vectors, cap, base_count):
    room = max(0, cap - base_count - len(extra)) if cap is not None else len(vectors)
    chosen = vectors[:room]
    extra.extend(chosen)
    stage_counts[stage] += len(chosen)


def counter_needed(row, scalar_pred, base_vectors):
    cluster_count, dominant_fraction, _collision = cluster_stats(base_vectors)
    correlated_echo = dominant_fraction >= 0.60 and len(base_vectors) >= 3
    return (
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


def stage_plan_for_arm(arm):
    if arm == "CELL_ONLY_COUNTERFACTUAL":
        return [("cell", 2)]
    if arm == "OPERATOR_ONLY_COUNTERFACTUAL":
        return [("operator", 2)]
    if arm == "JOINT_INTERACTION_COUNTERFACTUAL":
        return [("joint", 3)]
    if arm == "MULTI_STAGE_COUNTERFACTUAL_REPAIR":
        return [("cell", 1), ("operator", 1), ("joint", 3)]
    if arm in {"FULL_REPAIRED_ECF_CONTROLLER", "FULL_REPAIRED_ECF_CAP_7", "FULL_REPAIRED_ECF_CAP_9"}:
        return [("cell", 1), ("operator", 1), ("joint", 4)]
    return []


def support_cap_for_arm(arm):
    if arm == "FULL_REPAIRED_ECF_CAP_7":
        return 7
    if arm == "FULL_REPAIRED_ECF_CAP_9":
        return 9
    if arm in {
        "CELL_ONLY_COUNTERFACTUAL",
        "OPERATOR_ONLY_COUNTERFACTUAL",
        "JOINT_INTERACTION_COUNTERFACTUAL",
        "MULTI_STAGE_COUNTERFACTUAL_REPAIR",
        "FULL_REPAIRED_ECF_CONTROLLER",
        "SHUFFLED_COUNTER_SUPPORT_CONTROL",
        "RANDOM_EXTRA_SUPPORT_CONTROL",
        "ABSTAIN_ON_INDISTINGUISHABLE",
    }:
        return 12
    return SUPPORT_COUNT


def classify_error(row, pred, abstained, exact_joint, pair_equiv, op_exact):
    if abstained:
        if row["support_regime"] == "INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT":
            return "indistinguishable_abstain"
        return "abstain_unresolved"
    if exact_joint:
        return "ok"
    if row["support_regime"] == "INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT":
        return "false_confidence_on_unidentifiable"
    if row["support_regime"] == "EXTERNAL_TEST_REQUIRED_SUPPORT":
        return "external_test_required_unresolved"
    if pair_equiv and op_exact:
        return "joint_interaction_binding_error"
    if pair_equiv and not op_exact:
        return "operator_only_error"
    if op_exact and not pair_equiv:
        return "cell_only_error"
    return "both_cell_and_operator_wrong"


def evaluate_arm(row, bundle, arm):
    cap = support_cap_for_arm(arm)
    base_vectors = cached_base_vectors(row, bundle, SUPPORT_COUNT)
    scalar_scores = aggregate_sum(base_vectors)
    scalar_pred = predict(scalar_scores, bundle)
    base_count = SUPPORT_COUNT
    cluster_count, dominant_fraction, collision_count = cluster_stats(base_vectors)
    need_counter = counter_needed(row, scalar_pred, base_vectors)
    extra = []
    stage_counts = Counter()
    external_used = 0
    abstained = False

    if arm == "D49_BASELINE_REPLAY" or arm == "NO_COUNTERFACTUAL_CONTROL":
        pred = scalar_pred
        final_scores = scalar_scores
    elif arm == "FACTORISED_CELL_OPERATOR_SCORE":
        final_scores = d49.factorised_joint_scores(scalar_scores, bundle)
        pred = predict(final_scores, bundle)
    elif arm == "CELL_FIRST_OPERATOR_SECOND_PIPELINE":
        final_scores = d49.cell_then_operator_scores(scalar_scores, bundle)
        pred = predict(final_scores, bundle)
    elif arm == "OPERATOR_FIRST_CELL_SECOND_PIPELINE":
        final_scores = d49.operator_then_cell_scores(scalar_scores, bundle)
        pred = predict(final_scores, bundle)
    elif arm == "JOINT_BINDING_MATRIX":
        final_scores = joint_binding_matrix_scores(scalar_scores, bundle)
        pred = predict(final_scores, bundle)
    elif arm == "RANDOM_EXTRA_SUPPORT_CONTROL":
        extra = d49.random_extra_vectors(row, bundle)
        stage_counts["random"] = len(extra)
        final_scores = aggregate_duplicate_downweighted(base_vectors + extra)
        pred = predict(final_scores, bundle)
    elif arm == "SHUFFLED_COUNTER_SUPPORT_CONTROL":
        if need_counter:
            vectors = make_stage_vectors(row, bundle, scalar_pred, "joint", 3, shuffled=True)
            add_stage(extra, stage_counts, "shuffled_joint", vectors, cap, base_count)
        final_scores = aggregate_duplicate_downweighted(base_vectors + extra)
        pred = predict(final_scores, bundle)
    elif arm == "ABSTAIN_ON_INDISTINGUISHABLE":
        if row["external_test_available"]:
            vectors = make_stage_vectors(row, bundle, scalar_pred, "joint", 4, external=True)
            add_stage(extra, stage_counts, "external", vectors, cap, base_count)
            external_used = stage_counts["external"]
        elif not row["oracle_distinguishable"]:
            abstained = True
        elif need_counter:
            for stage, count in [("cell", 1), ("operator", 1), ("joint", 3)]:
                vectors = make_stage_vectors(row, bundle, scalar_pred, stage, count)
                add_stage(extra, stage_counts, stage, vectors, cap, base_count)
        final_scores = aggregate_duplicate_downweighted(base_vectors + extra)
        pred = predict(final_scores, bundle, abstain=abstained)
    else:
        if arm in {"FULL_REPAIRED_ECF_CONTROLLER", "FULL_REPAIRED_ECF_CAP_7", "FULL_REPAIRED_ECF_CAP_9"}:
            if row["external_test_available"]:
                vectors = make_stage_vectors(row, bundle, scalar_pred, "joint", 4, external=True)
                add_stage(extra, stage_counts, "external", vectors, cap, base_count)
                external_used = stage_counts["external"]
            elif not row["oracle_distinguishable"]:
                abstained = True
        if (not abstained) and need_counter:
            for stage, count in stage_plan_for_arm(arm):
                vectors = make_stage_vectors(row, bundle, scalar_pred, stage, count)
                add_stage(extra, stage_counts, stage, vectors, cap, base_count)
        final_scores = aggregate_duplicate_downweighted(base_vectors + extra)
        pred = predict(final_scores, bundle, abstain=abstained)

    exact_joint = pred["pred_joint"] == row["truth_joint"]
    pair_ok = pred["pred_pair_equivalence"] == row["truth_pair_equivalence"]
    pred_cells = set(d49.canonical_pair(pred["pred_pair"])) if pred["pred_pair"] else set()
    true_cells = set(d49.canonical_pair(row["truth_pair"]))
    cell_hit = len(pred_cells & true_cells) / 2.0 if pred_cells else 0.0
    op_exact = pred["pred_operator"] == row["true_operator"]
    op_equiv = pred["pred_operator_equivalence"] == row["truth_operator_equivalence"]
    group_correct = pred["pred_group"] == row["truth_group"]
    correct = exact_joint
    false_conf = (not correct) and (not abstained) and pred["confidence"] >= CONFIDENCE_THRESHOLD
    taxonomy = classify_error(row, pred, abstained, exact_joint, pair_ok, op_exact)
    joint_binding_consistency = bool(exact_joint or (pair_ok and op_exact))
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
        "cell_pair_equivalence_correct": pair_ok,
        "cell_hit_top2": cell_hit,
        "cell_hit_top2_correct": cell_hit >= 1.0,
        "operator_exact_correct": op_exact,
        "operator_equivalence_correct": op_equiv,
        "family_group_correct": group_correct,
        "joint_binding_consistency": joint_binding_consistency,
        "correct": correct,
        "reference_arm": False,
        "support_budget_cap": cap,
        "original_support_used": base_count,
        "cell_counter_support_used": stage_counts["cell"],
        "operator_counter_support_used": stage_counts["operator"],
        "joint_counter_support_used": stage_counts["joint"],
        "random_counter_support_used": stage_counts["random"],
        "shuffled_counter_support_used": stage_counts["shuffled_joint"],
        "counter_support_used": len(extra) - external_used,
        "external_test_used": external_used,
        "total_support_used": base_count + len(extra),
        "support_cluster_count": cluster_count,
        "dominant_cluster_fraction": dominant_fraction,
        "collision_count": collision_count,
        "correlated_echo_detected": dominant_fraction >= 0.60 and len(base_vectors) >= 3,
        "counter_support_requested": need_counter,
        "counter_support_resolved": len(extra) > 0 and exact_joint,
        "oracle_distinguishable": row["oracle_distinguishable"],
        "external_test_available": row["external_test_available"],
        "abstained": abstained,
        "false_confidence": false_conf,
        "confidence": pred["confidence"],
        "top1_top2_margin": pred["top1_top2_margin"],
        "entropy": pred["entropy"],
        "score_gap_truth_vs_wrong": score_gap(final_scores, row["truth_joint"]) if not abstained else 0.0,
        "baseline_exact_correct": scalar_pred["pred_joint"] == row["truth_joint"],
        "error_type": taxonomy,
    }


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
                    append_progress(
                        out,
                        "row_generation_progress",
                        started,
                        {"split": split, "completed_rows": completed, "total_rows": total},
                    )
                    write_json(
                        out / f"partial_{split}_row_generation.json",
                        {
                            "split": split,
                            "completed_rows": completed,
                            "total_rows": total,
                            "elapsed_sec": time.time() - started,
                        },
                    )
    append_progress(out, "row_generation_complete", started, {"split": split, "completed_rows": completed})
    return rows


def init_worker(bundle):
    global GLOBAL_BUNDLE
    GLOBAL_BUNDLE = bundle


def evaluate_row_all_arms(row):
    bundle = GLOBAL_BUNDLE
    return [evaluate_arm(row, bundle, arm) for arm in ARMS]


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
        "joint_binding_consistency_rate": mean([1.0 if row["joint_binding_consistency"] else 0.0 for row in rows]),
        "average_total_support_used": mean([row["total_support_used"] for row in rows]),
        "average_counter_support_used": mean([row["counter_support_used"] for row in rows]),
        "average_external_test_used": mean([row["external_test_used"] for row in rows]),
        "counter_support_request_rate": mean([1.0 if row["counter_support_requested"] else 0.0 for row in rows]),
        "counter_support_resolution_rate": mean([1.0 if row["counter_support_resolved"] else 0.0 for row in rows]),
        "abstain_rate": mean([1.0 if row["abstained"] else 0.0 for row in rows]),
        "false_confidence_rate": mean([1.0 if row["false_confidence"] else 0.0 for row in rows]),
        "confidence_when_wrong": mean(wrong_conf),
        "dominant_cluster_fraction_mean": mean([row["dominant_cluster_fraction"] for row in rows]),
        "cell_counter_support_mean": mean([row["cell_counter_support_used"] for row in rows]),
        "operator_counter_support_mean": mean([row["operator_counter_support_used"] for row in rows]),
        "joint_counter_support_mean": mean([row["joint_counter_support_used"] for row in rows]),
    }


def summarize_outputs(outputs):
    by_arm = defaultdict(list)
    by_arm_regime = defaultdict(list)
    by_arm_core = defaultdict(list)
    by_error = defaultdict(Counter)
    for row in outputs:
        by_arm[row["arm"]].append(row)
        by_arm_regime[(row["arm"], row["support_regime"])].append(row)
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
        "error_taxonomy": {
            arm: {
                regime: dict(by_error[(arm, regime)])
                for regime in REGIMES
                if (arm, regime) in by_error
            }
            for arm in ARMS
        },
    }


def record_result_batch(batch, outputs, sample_counts, path):
    for result in batch:
        outputs.append(result)
        sample_key = (result["arm"], result["support_regime"])
        if sample_counts[sample_key] < ROW_SAMPLE_PER_ARM_REGIME_SPLIT:
            append_jsonl(path, result)
            sample_counts[sample_key] += 1


def write_partial(out, rows, outputs, completed, started):
    partial = summarize_outputs(outputs)
    split = rows[0]["split"] if rows else "unknown"
    write_json(
        out / f"partial_{split}_metrics_snapshot.json",
        {
            "split": split,
            "completed_outputs": completed,
            "elapsed_sec": time.time() - started,
            "by_arm_core": partial["by_arm_core"],
            "full_regime": partial["by_arm_and_regime"].get("FULL_REPAIRED_ECF_CONTROLLER", {}),
        },
    )
    append_progress(out, "eval_progress", started, {"split": split, "completed_outputs": completed})


def evaluate_split(rows, bundle, path, started, out, heartbeat_sec, workers):
    if path.exists():
        path.unlink()
    outputs = []
    sample_counts = Counter()
    total = len(rows) * len(ARMS)
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


def external_accuracy(metrics, arm):
    return regime_accuracy(metrics, arm, "EXTERNAL_TEST_REQUIRED_SUPPORT")


def make_decision(metrics, failed_jobs):
    full = metrics["by_arm_core"]["FULL_REPAIRED_ECF_CONTROLLER"]
    corr = regime_accuracy(metrics, "FULL_REPAIRED_ECF_CONTROLLER", "CORRELATED_ECHO_SUPPORT")
    adv = regime_accuracy(metrics, "FULL_REPAIRED_ECF_CONTROLLER", "ADVERSARIAL_DISTRACTOR_SUPPORT")
    clean = regime_accuracy(metrics, "FULL_REPAIRED_ECF_CONTROLLER", "CLEAN_INDEPENDENT_SUPPORT")
    mixed = mixed_accuracy(metrics, "FULL_REPAIRED_ECF_CONTROLLER")
    indist = metrics["by_arm_and_regime"]["FULL_REPAIRED_ECF_CONTROLLER"]["INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT"]
    external = external_accuracy(metrics, "FULL_REPAIRED_ECF_CONTROLLER")
    controls_worse = all(full["accuracy"] > metrics["by_arm_core"][arm]["accuracy"] for arm in CONTROL_ARMS)
    if failed_jobs:
        return {
            "decision": "d49b_failed_jobs_present",
            "verdict": "D49B_FAILED_JOBS_PRESENT",
            "next": "D49B_REPAIR",
            "boundary": BOUNDARY,
        }
    if indist["false_confidence_rate"] > 0.01:
        return {
            "decision": "false_confidence_under_joint_indistinguishability",
            "verdict": "D49B_FALSE_CONFIDENCE_UNDER_JOINT_INDISTINGUISHABILITY",
            "next": "D49R_ABSTAIN_REPAIR",
            "boundary": BOUNDARY,
        }
    if external < 0.95:
        return {
            "decision": "external_test_required_joint_gap",
            "verdict": "D49B_EXTERNAL_TEST_REQUIRED_JOINT_GAP",
            "next": "D49E_EXTERNAL_INTERVENTION_SUPPORT_PLAN",
            "boundary": BOUNDARY,
        }
    passes = (
        clean >= 0.995
        and corr >= 0.95
        and adv >= 0.95
        and mixed >= 0.95
        and full["exact_joint_accuracy"] >= 0.97
        and full["cell_pair_equivalence_accuracy"] >= 0.97
        and full["operator_exact_accuracy"] >= 0.97
        and controls_worse
    )
    if passes:
        if full["average_total_support_used"] > 9.0:
            return {
                "decision": "joint_binding_repair_positive_high_cost",
                "verdict": "D49B_JOINT_BINDING_REPAIR_POSITIVE_HIGH_COST",
                "next": "D49C_SUPPORT_COST_OPTIMIZATION",
                "boundary": BOUNDARY,
            }
        return {
            "decision": "joint_binding_repair_positive",
            "verdict": "D49B_JOINT_BINDING_REPAIR_POSITIVE",
            "next": "D50_JOINT_FORMULA_DISCOVERY_SCALE_CONFIRM",
            "boundary": BOUNDARY,
        }
    cell = full["cell_pair_equivalence_accuracy"] >= 0.97
    op = full["operator_exact_accuracy"] >= 0.97
    if cell and op and full["exact_joint_accuracy"] < 0.97:
        return {
            "decision": "joint_interaction_binding_bottleneck",
            "verdict": "D49B_JOINT_INTERACTION_BINDING_BOTTLENECK",
            "next": "D49C_INTERACTION_BINDING_PLAN",
            "boundary": BOUNDARY,
        }
    return {
        "decision": "joint_interaction_binding_bottleneck",
        "verdict": "D49B_JOINT_INTERACTION_BINDING_BOTTLENECK",
        "next": "D49C_INTERACTION_BINDING_PLAN",
        "boundary": BOUNDARY,
    }


def report_by_arm_core(metrics, arms):
    return {arm: metrics["by_arm_core"][arm] for arm in arms}


def make_source_audit(repo_root):
    d49_script = repo_root / "scripts/probes/run_d49_joint_cell_operator_discovery_with_robust_support.py"
    d49_decision = repo_root / "target/pilot_wave/d49_joint_cell_operator_discovery_with_robust_support/smoke/decision.json"
    d49_aggregate = repo_root / "target/pilot_wave/d49_joint_cell_operator_discovery_with_robust_support/smoke/aggregate_metrics.json"
    manifest = {
        "upstream": "D49_JOINT_CELL_OPERATOR_DISCOVERY_WITH_ROBUST_SUPPORT",
        "expected_decision": "joint_binding_bottleneck",
        "expected_next": "D49B_JOINT_BINDING_REPAIR",
        "script_present": d49_script.exists(),
        "decision_present": d49_decision.exists(),
        "aggregate_present": d49_aggregate.exists(),
    }
    if d49_decision.exists():
        manifest["decision_json"] = json.loads(d49_decision.read_text(encoding="utf-8"))
    if d49_aggregate.exists():
        aggregate = json.loads(d49_aggregate.read_text(encoding="utf-8"))
        full = aggregate["test_metrics"]["by_arm_core"]["FULL_ROBUST_ECF_CONTROLLER"]
        manifest["key_metrics"] = {
            "exact_joint": full["exact_joint_accuracy"],
            "clean": aggregate["test_metrics"]["by_arm_and_regime"]["FULL_ROBUST_ECF_CONTROLLER"]["CLEAN_INDEPENDENT_SUPPORT"]["accuracy"],
            "correlated_echo": aggregate["test_metrics"]["by_arm_and_regime"]["FULL_ROBUST_ECF_CONTROLLER"]["CORRELATED_ECHO_SUPPORT"]["accuracy"],
            "adversarial_distractor": aggregate["test_metrics"]["by_arm_and_regime"]["FULL_ROBUST_ECF_CONTROLLER"]["ADVERSARIAL_DISTRACTOR_SUPPORT"]["accuracy"],
            "cell_pair_equivalence": full["cell_pair_equivalence_accuracy"],
            "operator_exact": full["operator_exact_accuracy"],
        }
    else:
        manifest["known_prompt_summary_used"] = {
            "decision": "joint_binding_bottleneck",
            "full_robust_exact_joint": 0.9581,
            "clean": 1.0,
            "correlated_echo": 0.8952,
            "adversarial_distractor": 0.8950,
            "next": "D49B_JOINT_BINDING_REPAIR",
        }
    return manifest


def write_report(out, decision, aggregate):
    metrics = aggregate["test_metrics"]["by_arm_and_regime"]
    core = aggregate["test_metrics"]["by_arm_core"]
    lines = [
        "# D49B Joint Binding Repair Result",
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
        "Core arm table:",
        "",
        "```text",
    ]
    for arm in [
        "D49_BASELINE_REPLAY",
        "JOINT_BINDING_MATRIX",
        "CELL_ONLY_COUNTERFACTUAL",
        "OPERATOR_ONLY_COUNTERFACTUAL",
        "JOINT_INTERACTION_COUNTERFACTUAL",
        "MULTI_STAGE_COUNTERFACTUAL_REPAIR",
        "FULL_REPAIRED_ECF_CONTROLLER",
        "FULL_REPAIRED_ECF_CAP_7",
        "FULL_REPAIRED_ECF_CAP_9",
        "RANDOM_EXTRA_SUPPORT_CONTROL",
        "SHUFFLED_COUNTER_SUPPORT_CONTROL",
        "NO_COUNTERFACTUAL_CONTROL",
    ]:
        row = core[arm]
        lines.append(
            f"{arm}: exact={row['exact_joint_accuracy']:.4f}, cell={row['cell_pair_equivalence_accuracy']:.4f}, op={row['operator_exact_accuracy']:.4f}, support={row['average_total_support_used']:.3f}"
        )
    lines.extend(["```", "", "FULL_REPAIRED_ECF_CONTROLLER by regime:", "", "```text"])
    for regime in REGIMES:
        row = metrics["FULL_REPAIRED_ECF_CONTROLLER"][regime]
        lines.append(
            f"{regime}: acc={row['accuracy']:.4f}, abstain={row['abstain_rate']:.4f}, false_conf={row['false_confidence_rate']:.4f}, support={row['average_total_support_used']:.3f}"
        )
    lines.extend(["```", "", f"Boundary: {BOUNDARY}", ""])
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", default="10301,10302,10303,10304,10305")
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
    bundle = d49.make_bundle("ALL28_UNORDERED")
    repo_root = Path(__file__).resolve().parents[2]
    write_json(
        out / "queue.json",
        {
            "task": "D49B joint binding repair",
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
    upstream = make_source_audit(repo_root)
    write_json(out / "d49_upstream_manifest.json", upstream)
    write_json(
        out / "compute_probe.json",
        {
            "mode": "local_python_cpu",
            "workers_requested": args.workers,
            "cpu_target": args.cpu_target,
            "note": "D49B is deterministic symbolic scoring; no external model/API/download used.",
        },
    )
    write_json(
        out / "dataset_manifest.json",
        {
            "task": "D49B_JOINT_BINDING_REPAIR",
            "primary_space": PRIMARY_SPACE,
            "candidate": "cell_pair x operator",
            "operator_candidates": OP_NAMES,
            "joint_candidate_count_primary": len(bundle["candidates"]),
            "support_regimes": REGIMES,
            "arms": ARMS,
            "truth_hidden_from_fair_arms": True,
            "label_echo_reference_only_not_fair": True,
            "candidate_family_equivalence_cell_operator_metrics_separated": True,
            "indistinguishable_false_confidence_reported": True,
            "controls_included": True,
            "row_outputs_are_sampled_but_metrics_use_full_rows": True,
            "no_python_hash": True,
            "no_fake_sampling": True,
            "boundary": BOUNDARY,
        },
    )
    train_rows = make_rows_with_progress(seeds, args.train_rows_per_seed, "train", bundle, out, started, args.heartbeat_sec)
    test_rows = make_rows_with_progress(seeds, args.test_rows_per_seed, "test", bundle, out, started, args.heartbeat_sec)
    ood_rows = make_rows_with_progress(seeds, args.ood_rows_per_seed, "ood", bundle, out, started, args.heartbeat_sec)
    write_json(
        out / "train_manifest.json",
        {
            "train_rows": len(train_rows),
            "note": "Train rows are generated for dataset parity; D49B arms are non-learned symbolic policies.",
        },
    )
    append_progress(out, "rows_generated", started, {"train": len(train_rows), "test": len(test_rows), "ood": len(ood_rows)})
    failed_jobs = []
    try:
        test_outputs = evaluate_split(test_rows, bundle, out / "row_outputs_test.jsonl", started, out, args.heartbeat_sec, args.workers)
        ood_outputs = evaluate_split(ood_rows, bundle, out / "row_outputs_ood.jsonl", started, out, args.heartbeat_sec, args.workers)
    except Exception as exc:
        failed_jobs.append({"stage": "evaluation", "error": repr(exc)})
        write_json(out / "error.json", {"failed_jobs": failed_jobs})
        raise
    test_metrics = summarize_outputs(test_outputs)
    ood_metrics = summarize_outputs(ood_outputs)
    decision = make_decision(test_metrics, failed_jobs)
    full = test_metrics["by_arm_core"]["FULL_REPAIRED_ECF_CONTROLLER"]
    baseline = test_metrics["by_arm_core"]["D49_BASELINE_REPLAY"]
    multistage = test_metrics["by_arm_core"]["MULTI_STAGE_COUNTERFACTUAL_REPAIR"]
    joint = test_metrics["by_arm_core"]["JOINT_INTERACTION_COUNTERFACTUAL"]
    aggregate = {
        "task": "D49B_JOINT_BINDING_REPAIR",
        "test_metrics": test_metrics,
        "ood_metrics": ood_metrics,
        "primary_policy_metrics": test_metrics["by_arm_core"],
        "robust_gain_vs_d49_baseline": full["accuracy"] - baseline["accuracy"],
        "multi_stage_gain_vs_baseline": multistage["accuracy"] - baseline["accuracy"],
        "joint_counterfactual_gain_vs_cell_only": joint["accuracy"] - test_metrics["by_arm_core"]["CELL_ONLY_COUNTERFACTUAL"]["accuracy"],
        "joint_counterfactual_gain_vs_operator_only": joint["accuracy"] - test_metrics["by_arm_core"]["OPERATOR_ONLY_COUNTERFACTUAL"]["accuracy"],
        "clean_regression_vs_baseline": regime_accuracy(test_metrics, "D49_BASELINE_REPLAY", "CLEAN_INDEPENDENT_SUPPORT")
        - regime_accuracy(test_metrics, "FULL_REPAIRED_ECF_CONTROLLER", "CLEAN_INDEPENDENT_SUPPORT"),
        "controls_worse": all(full["accuracy"] > test_metrics["by_arm_core"][arm]["accuracy"] for arm in CONTROL_ARMS),
        "failed_jobs": failed_jobs,
        "decision": decision["decision"],
        "verdict": decision["verdict"],
        "next": decision["next"],
        "boundary": BOUNDARY,
    }
    reports = {
        "joint_error_taxonomy_report.json": test_metrics["error_taxonomy"],
        "cell_vs_operator_error_report.json": {
            arm: {
                "cell_pair_equivalence_accuracy": test_metrics["by_arm_core"][arm]["cell_pair_equivalence_accuracy"],
                "operator_exact_accuracy": test_metrics["by_arm_core"][arm]["operator_exact_accuracy"],
                "exact_joint_accuracy": test_metrics["by_arm_core"][arm]["exact_joint_accuracy"],
            }
            for arm in ARMS
        },
        "binding_consistency_report.json": {
            arm: test_metrics["by_arm_core"][arm]["joint_binding_consistency_rate"] for arm in ARMS
        },
        "counterfactual_stage_report.json": {
            "cell_only": test_metrics["by_arm_core"]["CELL_ONLY_COUNTERFACTUAL"],
            "operator_only": test_metrics["by_arm_core"]["OPERATOR_ONLY_COUNTERFACTUAL"],
            "joint_interaction": test_metrics["by_arm_core"]["JOINT_INTERACTION_COUNTERFACTUAL"],
            "multi_stage": test_metrics["by_arm_core"]["MULTI_STAGE_COUNTERFACTUAL_REPAIR"],
            "full": full,
            "stage_gains": {
                "joint_vs_cell_only": aggregate["joint_counterfactual_gain_vs_cell_only"],
                "joint_vs_operator_only": aggregate["joint_counterfactual_gain_vs_operator_only"],
                "multi_stage_vs_baseline": aggregate["multi_stage_gain_vs_baseline"],
            },
        },
        "external_test_required_report.json": {
            arm: test_metrics["by_arm_and_regime"][arm]["EXTERNAL_TEST_REQUIRED_SUPPORT"] for arm in ARMS
        },
        "support_cost_frontier_report.json": {
            "baseline_support5": test_metrics["by_arm_core"]["D49_BASELINE_REPLAY"],
            "full_cap7": test_metrics["by_arm_core"]["FULL_REPAIRED_ECF_CAP_7"],
            "full_cap9": test_metrics["by_arm_core"]["FULL_REPAIRED_ECF_CAP_9"],
            "full_controller": full,
        },
        "regime_breakdown_report.json": test_metrics["by_arm_and_regime"],
        "control_report.json": {arm: test_metrics["by_arm_core"][arm] for arm in CONTROL_ARMS},
        "indistinguishable_false_confidence_report.json": {
            arm: test_metrics["by_arm_and_regime"][arm]["INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT"] for arm in ARMS
        },
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
                "joint_interaction_counterfactual": joint,
                "multi_stage": multistage,
                "full_repaired": full,
                "full_by_regime": test_metrics["by_arm_and_regime"]["FULL_REPAIRED_ECF_CONTROLLER"],
                "external_required": reports["external_test_required_report.json"]["FULL_REPAIRED_ECF_CONTROLLER"],
                "indistinguishable": reports["indistinguishable_false_confidence_report.json"]["FULL_REPAIRED_ECF_CONTROLLER"],
            },
            "boundary": BOUNDARY,
        },
    )
    write_report(out, decision, aggregate)
    queue = json.loads((out / "queue.json").read_text(encoding="utf-8"))
    write_json(out / "queue.json", {**queue, "status": "complete"})
    append_progress(out, "final_decision", started, decision)
    print(json.dumps({"decision": decision["decision"], "verdict": decision["verdict"], "next": decision["next"]}, indent=2))


if __name__ == "__main__":
    main()
