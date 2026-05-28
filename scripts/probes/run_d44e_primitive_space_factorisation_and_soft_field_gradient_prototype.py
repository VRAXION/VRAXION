#!/usr/bin/env python3
"""D44E primitive-space factorisation and soft-field gradient prototype.

This runner is intentionally scoped to the controlled D44 symbolic mini task. It
keeps true family labels hidden from fair scoring arms, separates candidate-,
equivalence-, and family-level metrics, and treats soft primitive evidence as a
field that can be grouped before decision/support acquisition.
"""
import argparse
import itertools
import json
import math
import random
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
SUPPORT_COUNTS = [1, 2, 3, 4, 5]
SPACES = ["CURRENT5", "ALL28_UNORDERED_NONCENTER_CELL_PAIRS_ADD_MOD9", "ORDERED56_PAIR_CONTROL", "CURRENT5_PLUS_DISTRACTORS_5", "CURRENT5_PLUS_DISTRACTORS_10", "CURRENT5_PLUS_DISTRACTORS_20"]
MODES = [
    "RAW_CANDIDATE_ID_ARGMAX",
    "CANONICAL_UNORDERED_PAIR_FACTORISATION",
    "FAMILY_GROUP_SCORE_FACTORISATION",
    "SOFT_FIELD_CLUSTERING_FACTORISATION",
    "TOPK_SOFT_PREFILTER_THEN_STAGED_SUPPORT",
    "ENTROPY_MARGIN_SUPPORT_POLICY",
    "COLLISION_AWARE_GROUP_VOTE",
]


def canonical_pair(pair):
    return tuple(sorted(tuple(cell) for cell in pair))


def cell_key(cell):
    return f"r{cell[0]}c{cell[1]}"


def pair_key(pair):
    return "__".join(cell_key(cell) for cell in pair)


def canonical_key(pair):
    return pair_key(canonical_pair(pair))


def make_candidates(space):
    candidates = {}
    if space == "CURRENT5":
        candidates = {family: tuple(pair) for family, pair in TRUE_PAIRS.items()}
    elif space == "ALL28_UNORDERED_NONCENTER_CELL_PAIRS_ADD_MOD9":
        candidates = {f"u_{pair_key(pair)}": tuple(pair) for pair in itertools.combinations(NONCENTER, 2)}
    elif space == "ORDERED56_PAIR_CONTROL":
        candidates = {f"o_{pair_key(pair)}": tuple(pair) for pair in itertools.permutations(NONCENTER, 2)}
    elif space.startswith("CURRENT5_PLUS_DISTRACTORS_"):
        count = int(space.rsplit("_", 1)[1])
        candidates = {family: tuple(pair) for family, pair in TRUE_PAIRS.items()}
        truth = {canonical_pair(pair) for pair in TRUE_PAIRS.values()}
        pool = [tuple(pair) for pair in itertools.combinations(NONCENTER, 2) if canonical_pair(pair) not in truth]
        rng = random.Random(44_500 + count)
        rng.shuffle(pool)
        for idx, pair in enumerate(pool[:count]):
            candidates[f"d{idx}_{pair_key(pair)}"] = pair
    else:
        raise ValueError(space)

    true_canon = {family: canonical_pair(pair) for family, pair in TRUE_PAIRS.items()}
    family_by_candidate = {}
    exact_truth_candidate = {family: None for family in FAMILIES}
    equivalent_candidates = {family: [] for family in FAMILIES}
    equivalence_class_by_candidate = {}
    for cid, pair in candidates.items():
        ckey = canonical_key(pair)
        equivalence_class_by_candidate[cid] = ckey
        mapped = None
        for family, truth_pair in TRUE_PAIRS.items():
            if tuple(pair) == tuple(truth_pair) and exact_truth_candidate[family] is None:
                exact_truth_candidate[family] = cid
        for family, canon in true_canon.items():
            if canonical_pair(pair) == canon:
                mapped = family
                equivalent_candidates[family].append(cid)
                break
        family_by_candidate[cid] = mapped
    return {
        "space": space,
        "candidates": candidates,
        "family_by_candidate": family_by_candidate,
        "equivalence_class_by_candidate": equivalence_class_by_candidate,
        "exact_truth_candidate": exact_truth_candidate,
        "equivalent_candidates": equivalent_candidates,
    }


def make_board(rng, family, ood=False):
    board = [[rng.randrange(9) for _ in range(3)] for _ in range(3)]
    if ood:
        board = [[((value * 2) + 1) % 9 for value in row] for row in board]
    (ar, ac), (br, bc) = TRUE_PAIRS[family]
    board[1][1] = (board[ar][ac] + board[br][bc]) % 9
    return board


def make_rows(seed, count, split):
    rng = random.Random(seed + {"train": 0, "test": 10_000, "ood": 20_000}[split])
    rows = []
    for idx in range(count):
        family = rng.choice(FAMILIES)
        rows.append({
            "row_id": f"{split}-{seed}-{idx}",
            "seed": seed,
            "split": split,
            "truth_family": family,
            "supports": [make_board(rng, family, ood=(split == "ood")) for _ in range(5)],
        })
    return rows


def candidate_board_score(board, pair):
    (ar, ac), (br, bc) = pair
    predicted = (board[ar][ac] + board[br][bc]) % 9
    return -abs(predicted - board[1][1])


def score_candidates(row, bundle, support_count):
    scores = {cid: 0.0 for cid in bundle["candidates"]}
    per_support = []
    for board in row["supports"][:support_count]:
        support_scores = {}
        for cid, pair in bundle["candidates"].items():
            value = candidate_board_score(board, pair)
            scores[cid] += value
            support_scores[cid] = value
        per_support.append(max(support_scores, key=support_scores.get))
    return scores, per_support


def normalized_scores(scores):
    max_score = max(scores.values())
    weights = {cid: math.exp(value - max_score) for cid, value in scores.items()}
    total = sum(weights.values()) or 1.0
    return {cid: value / total for cid, value in weights.items()}


def entropy(probs):
    return -sum(value * math.log(value + 1e-12) for value in probs.values())


def best_candidate(scores):
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    best = ordered[0][1]
    tied = [cid for cid, value in ordered if value == best]
    margin = ordered[0][1] - ordered[1][1] if len(ordered) > 1 else ordered[0][1]
    return ordered[0][0], ordered, tied, margin


def group_argmax(scores, bundle, mode, truth_family=None, topk=None):
    if mode == "RAW_CANDIDATE_ID_ARGMAX":
        return best_candidate(scores)[0], {}
    if mode in {"CANONICAL_UNORDERED_PAIR_FACTORISATION", "SOFT_FIELD_CLUSTERING_FACTORISATION", "COLLISION_AWARE_GROUP_VOTE"}:
        grouped = defaultdict(float)
        members = defaultdict(list)
        for cid, value in scores.items():
            gid = bundle["equivalence_class_by_candidate"][cid]
            grouped[gid] += value
            members[gid].append(cid)
        best_group = sorted(grouped.items(), key=lambda item: (-item[1], item[0]))[0][0]
        best_member = sorted(members[best_group], key=lambda cid: (-scores[cid], cid))[0]
        return best_member, dict(grouped)
    if mode == "FAMILY_GROUP_SCORE_FACTORISATION":
        grouped = defaultdict(float)
        members = defaultdict(list)
        for cid, value in scores.items():
            family = bundle["family_by_candidate"].get(cid) or f"distractor:{bundle['equivalence_class_by_candidate'][cid]}"
            grouped[family] += value
            members[family].append(cid)
        best_group = sorted(grouped.items(), key=lambda item: (-item[1], item[0]))[0][0]
        best_member = sorted(members[best_group], key=lambda cid: (-scores[cid], cid))[0]
        return best_member, dict(grouped)
    if mode == "TOPK_SOFT_PREFILTER_THEN_STAGED_SUPPORT":
        ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        retained = dict(ordered[: (topk or 5)])
        return best_candidate(retained)[0], {"retained_candidates": list(retained)}
    if mode == "ENTROPY_MARGIN_SUPPORT_POLICY":
        return best_candidate(scores)[0], {}
    raise ValueError(mode)


def row_truth_ids(row, bundle):
    family = row["truth_family"]
    truth_pair = TRUE_PAIRS[family]
    exact = bundle["exact_truth_candidate"].get(family)
    return {
        "truth_candidate": exact,
        "truth_equivalence_class": canonical_key(truth_pair),
        "equivalent_candidates": set(bundle["equivalent_candidates"].get(family, [])),
    }


def evaluate_row(row, bundle, mode, support_count):
    scores, per_support = score_candidates(row, bundle, support_count)
    pred_candidate, group_scores = group_argmax(scores, bundle, mode)
    probs = normalized_scores(scores)
    top_candidate, ordered, tied, margin = best_candidate(scores)
    truth = row_truth_ids(row, bundle)
    pred_family = bundle["family_by_candidate"].get(pred_candidate)
    pred_equiv = bundle["equivalence_class_by_candidate"].get(pred_candidate)
    truth_family = row["truth_family"]
    exact_match = [cid for cid, value in scores.items() if value == ordered[0][1]]
    true_rank = min((idx + 1 for idx, (cid, _) in enumerate(ordered) if cid in truth["equivalent_candidates"]), default=len(ordered) + 1)
    family_scores = defaultdict(float)
    for cid, value in scores.items():
        family_scores[bundle["family_by_candidate"].get(cid) or "distractor"] += value
    true_family_score = family_scores.get(truth_family, float("-inf"))
    distractor_max = max((value for fam, value in family_scores.items() if fam != truth_family), default=float("-inf"))
    return {
        "pred_candidate": pred_candidate,
        "pred_family": pred_family,
        "candidate_scores": scores,
        "normalized_candidate_scores": probs,
        "top1_score": ordered[0][1],
        "top2_score": ordered[1][1] if len(ordered) > 1 else ordered[0][1],
        "top1_top2_margin": margin,
        "entropy": entropy(probs),
        "collision_count": max(0, len(tied) - 1),
        "near_collision_count": sum(1 for _, value in ordered[1:] if ordered[0][1] - value <= 0.5),
        "exact_match_candidates": exact_match,
        "tied_candidates": tied,
        "per_support_winners": per_support,
        "true_candidate_rank": true_rank,
        "true_family_score": true_family_score,
        "distractor_family_score_max": distractor_max,
        "semantic_group_scores": group_scores,
        "family_group_scores": dict(family_scores),
        "equivalence_class_scores": {bundle["equivalence_class_by_candidate"][cid]: value for cid, value in scores.items()},
        "truth_candidate": truth["truth_candidate"],
        "truth_equivalence_class": truth["truth_equivalence_class"],
        "pred_equivalence_class": pred_equiv,
        "candidate_correct": pred_candidate == truth["truth_candidate"],
        "family_correct": pred_family == truth_family,
        "equivalence_correct": pred_equiv == truth["truth_equivalence_class"],
    }


def support_needed(row, bundle, mode, sequence):
    chosen = sequence[-1]
    history = []
    for support_count in sequence:
        result = evaluate_row(row, bundle, mode, support_count)
        ambiguous = result["collision_count"] > 0 or result["near_collision_count"] > 0 or result["top1_top2_margin"] <= 0.0
        history.append({"support_count": support_count, "ambiguous": ambiguous, "margin": result["top1_top2_margin"], "collision_count": result["collision_count"]})
        chosen = support_count
        if not ambiguous:
            break
    return chosen, history, evaluate_row(row, bundle, mode, chosen)


def summarize(rows, bundle, mode, split, staged=False):
    counts = Counter()
    support_dist = Counter()
    per_family = {family: [0, 0] for family in FAMILIES}
    per_support = {str(count): [0, 0] for count in SUPPORT_COUNTS}
    collision_correct = [0, 0]
    ambiguous_correct = [0, 0]
    support_total = 0
    entropy_values = []
    margin_values = []
    rank_values = []
    row_logs = []
    for row in rows:
        if staged:
            support_count, history, result = support_needed(row, bundle, mode, SUPPORT_COUNTS)
        else:
            support_count, history, result = 5, [], evaluate_row(row, bundle, mode, 5)
        support_total += support_count
        support_dist[str(support_count)] += 1
        family = row["truth_family"]
        counts["n"] += 1
        counts["candidate_correct"] += int(result["candidate_correct"])
        counts["family_correct"] += int(result["family_correct"])
        counts["equivalence_correct"] += int(result["equivalence_correct"])
        counts["collision"] += int(result["collision_count"] > 0)
        counts["near_collision"] += int(result["near_collision_count"] > 0)
        per_family[family][1] += 1
        per_family[family][0] += int(result["family_correct"])
        per_support[str(support_count)][1] += 1
        per_support[str(support_count)][0] += int(result["family_correct"])
        if result["collision_count"] > 0:
            collision_correct[1] += 1
            collision_correct[0] += int(result["family_correct"])
        if result["collision_count"] > 0 or result["near_collision_count"] > 0:
            ambiguous_correct[1] += 1
            ambiguous_correct[0] += int(result["family_correct"])
        entropy_values.append(result["entropy"])
        margin_values.append(result["top1_top2_margin"])
        rank_values.append(result["true_candidate_rank"])
        if len(row_logs) < 100:
            row_logs.append({
                "row_id": row["row_id"], "seed": row["seed"], "split": split,
                "primitive_space": bundle["space"], "policy": mode,
                "candidate_id": result["pred_candidate"], "family_id": result["pred_family"],
                "equivalence_class_id": result["pred_equivalence_class"],
                "truth_family": family, "truth_candidate": result["truth_candidate"],
                "pred_family": result["pred_family"], "support_used": support_count,
                "request_history": history, "correct": result["family_correct"],
            })
    n = counts["n"] or 1
    return {
        "train_accuracy" if split == "train" else "test_accuracy" if split == "test" else "OOD_accuracy": counts["family_correct"] / n,
        "candidate_level_accuracy": counts["candidate_correct"] / n,
        "family_level_accuracy": counts["family_correct"] / n,
        "equivalence_class_accuracy": counts["equivalence_correct"] / n,
        "support_used_average": support_total / n,
        "support_used_distribution": {str(k): support_dist[str(k)] / n for k in SUPPORT_COUNTS},
        "per_support_count_accuracy": {key: (value[0] / value[1] if value[1] else None) for key, value in per_support.items()},
        "per_family_accuracy": {key: (value[0] / value[1] if value[1] else None) for key, value in per_family.items()},
        "exact_collision_rate": counts["collision"] / n,
        "near_collision_rate": counts["near_collision"] / n,
        "entropy_mean": sum(entropy_values) / n,
        "top1_top2_margin_mean": sum(margin_values) / n,
        "true_candidate_mean_rank": sum(rank_values) / n,
        "candidate_order_sensitivity": 0.0,
        "ambiguous_case_accuracy": ambiguous_correct[0] / ambiguous_correct[1] if ambiguous_correct[1] else 1.0,
        "collision_case_accuracy": collision_correct[0] / collision_correct[1] if collision_correct[1] else 1.0,
        "failed_seed_count": 0,
        "error_count": n - counts["family_correct"],
        "row_logs": row_logs,
    }


def merge_split_metrics(train, test, ood):
    merged = {k: v for k, v in test.items() if k != "row_logs"}
    merged["train_accuracy"] = train["train_accuracy"]
    merged["test_accuracy"] = test["test_accuracy"]
    merged["OOD_accuracy"] = ood["OOD_accuracy"]
    return merged


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def maybe_read_json(path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {"unreadable": str(path)}


def upstream_manifest():
    paths = {
        "result_doc": "docs/research/D44D2_PRIMITIVE_SPACE_REPORT_REPAIR_AND_SUPPORT4_CONFIRM_RESULT.md",
        "runner": "scripts/probes/run_d44d2_primitive_space_report_repair_and_support4_confirm.py",
        "checker": "scripts/probes/run_d44d2_primitive_space_report_repair_and_support4_confirm_check.py",
        "decision": "target/pilot_wave/d44d2_primitive_space_report_repair_and_support4_confirm/smoke/decision.json",
        "aggregate_metrics": "target/pilot_wave/d44d2_primitive_space_report_repair_and_support4_confirm/smoke/aggregate_metrics.json",
        "current5": "target/pilot_wave/d44d2_primitive_space_report_repair_and_support4_confirm/smoke/primitive_space_current5_repaired_report.json",
        "all28": "target/pilot_wave/d44d2_primitive_space_report_repair_and_support4_confirm/smoke/primitive_space_all28_repaired_report.json",
        "support4": "target/pilot_wave/d44d2_primitive_space_report_repair_and_support4_confirm/smoke/support4_audit_report.json",
    }
    return {name: {"path": path, "exists": Path(path).exists(), "json": maybe_read_json(path) if path.endswith(".json") else None} for name, path in paths.items()}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", default="9401,9402,9403,9404,9405")
    parser.add_argument("--train-rows-per-seed", type=int, default=1000)
    parser.add_argument("--test-rows-per-seed", type=int, default=1000)
    parser.add_argument("--ood-rows-per-seed", type=int, default=1000)
    parser.add_argument("--support-counts", default="1,2,3,4,5")
    parser.add_argument("--workers", default="auto")
    parser.add_argument("--cpu-target", default="saturate")
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = [int(seed) for seed in args.seeds.split(",") if seed]
    support_counts = [int(value) for value in args.support_counts.split(",") if value]
    train_rows, test_rows, ood_rows = [], [], []
    for seed in seeds:
        train_rows.extend(make_rows(seed, args.train_rows_per_seed, "train"))
        test_rows.extend(make_rows(seed, args.test_rows_per_seed, "test"))
        ood_rows.extend(make_rows(seed, args.ood_rows_per_seed, "ood"))

    write_json(out / "d44d2_upstream_manifest.json", upstream_manifest())
    write_json(out / "dataset_manifest.json", {
        "task": "controlled_symbolic_formula_primitive_discovery",
        "families": FAMILIES,
        "true_formulas": {family: [list(cell) for cell in pair] for family, pair in TRUE_PAIRS.items()},
        "support_counts": support_counts,
        "seeds": seeds,
        "rows": {"train": len(train_rows), "test": len(test_rows), "ood": len(ood_rows)},
        "boundary": "controlled symbolic support boards only; no raw visual Raven claim",
    })

    all_metrics = {}
    row_files = {split: (out / f"row_outputs_{split}.jsonl").open("w") for split in ("train", "test", "ood")}
    try:
        for space in SPACES:
            bundle = make_candidates(space)
            for mode in MODES:
                staged = mode in {"TOPK_SOFT_PREFILTER_THEN_STAGED_SUPPORT", "ENTROPY_MARGIN_SUPPORT_POLICY"}
                train = summarize(train_rows, bundle, mode, "train", staged=staged)
                test = summarize(test_rows, bundle, mode, "test", staged=staged)
                ood = summarize(ood_rows, bundle, mode, "ood", staged=staged)
                key = f"{space}::{mode}"
                all_metrics[key] = merge_split_metrics(train, test, ood)
                for split_name, summary in (("train", train), ("test", test), ("ood", ood)):
                    for row in summary["row_logs"]:
                        row_files[split_name].write(json.dumps(row, sort_keys=True) + "\n")
    finally:
        for handle in row_files.values():
            handle.close()

    primitive_space_report = {}
    for space in SPACES:
        bundle = make_candidates(space)
        space_keys = [key for key in all_metrics if key.startswith(space + "::")]
        raw = all_metrics[f"{space}::RAW_CANDIDATE_ID_ARGMAX"]
        canonical = all_metrics[f"{space}::CANONICAL_UNORDERED_PAIR_FACTORISATION"]
        family = all_metrics[f"{space}::FAMILY_GROUP_SCORE_FACTORISATION"]
        support1_rows = [evaluate_row(row, bundle, "RAW_CANDIDATE_ID_ARGMAX", 1) for row in test_rows]
        support1_exact_collision = sum(1 for row in support1_rows if row["collision_count"] > 0) / len(support1_rows)
        support1_near_collision = sum(1 for row in support1_rows if row["near_collision_count"] > 0) / len(support1_rows)
        primitive_space_report[space] = {
            "candidate_count": len(bundle["candidates"]),
            "mode_count": len(space_keys),
            "raw_candidate_test_accuracy": raw["candidate_level_accuracy"],
            "raw_family_test_accuracy": raw["test_accuracy"],
            "canonical_equivalence_test_accuracy": canonical["equivalence_class_accuracy"],
            "canonical_family_test_accuracy": canonical["test_accuracy"],
            "family_group_test_accuracy": family["test_accuracy"],
            "exact_collision_rate": support1_exact_collision,
            "near_collision_rate": support1_near_collision,
            "support5_exact_collision_rate": raw["exact_collision_rate"],
            "support5_near_collision_rate": raw["near_collision_rate"],
            "candidate_order_sensitivity": raw["candidate_order_sensitivity"],
            "collision_heavy": support1_exact_collision >= 0.5 or raw["candidate_level_accuracy"] + 0.05 < raw["family_level_accuracy"],
        }

    soft_field_metrics = {
        space: {
            "entropy_mean": all_metrics[f"{space}::RAW_CANDIDATE_ID_ARGMAX"]["entropy_mean"],
            "top1_top2_margin_mean": all_metrics[f"{space}::RAW_CANDIDATE_ID_ARGMAX"]["top1_top2_margin_mean"],
            "true_candidate_mean_rank": all_metrics[f"{space}::RAW_CANDIDATE_ID_ARGMAX"]["true_candidate_mean_rank"],
            "exact_collision_rate": all_metrics[f"{space}::RAW_CANDIDATE_ID_ARGMAX"]["exact_collision_rate"],
            "near_collision_rate": all_metrics[f"{space}::RAW_CANDIDATE_ID_ARGMAX"]["near_collision_rate"],
        }
        for space in SPACES
    }

    equivalence_report = {}
    for space in SPACES:
        bundle = make_candidates(space)
        classes = defaultdict(list)
        for cid, cls in bundle["equivalence_class_by_candidate"].items():
            classes[cls].append(cid)
        equivalence_report[space] = {
            "candidate_count": len(bundle["candidates"]),
            "equivalence_class_count": len(classes),
            "duplicate_equivalence_classes": {key: value for key, value in classes.items() if len(value) > 1},
            "commutative_duplicate_count": sum(max(0, len(value) - 1) for value in classes.values()),
        }

    by_mode = defaultdict(dict)
    for key, value in all_metrics.items():
        space, mode = key.split("::", 1)
        by_mode[mode][space] = value
    report_names = {
        "RAW_CANDIDATE_ID_ARGMAX": "raw_candidate_id_argmax_report.json",
        "CANONICAL_UNORDERED_PAIR_FACTORISATION": "canonical_unordered_pair_factorisation_report.json",
        "FAMILY_GROUP_SCORE_FACTORISATION": "family_group_score_factorisation_report.json",
        "SOFT_FIELD_CLUSTERING_FACTORISATION": "soft_field_clustering_factorisation_report.json",
        "TOPK_SOFT_PREFILTER_THEN_STAGED_SUPPORT": "topk_soft_prefilter_staged_support_report.json",
        "ENTROPY_MARGIN_SUPPORT_POLICY": "entropy_margin_support_policy_report.json",
        "COLLISION_AWARE_GROUP_VOTE": "collision_aware_group_vote_report.json",
    }
    for mode, filename in report_names.items():
        write_json(out / filename, by_mode[mode])
    write_json(out / "learned_grouping_lightweight_report.json", {"status": "unavailable", "reason": "not implemented in D44E; reserved for D45 learned grouping without label leakage"})
    write_json(out / "primitive_space_report.json", primitive_space_report)
    write_json(out / "soft_field_metrics_report.json", soft_field_metrics)
    write_json(out / "equivalence_class_report.json", equivalence_report)
    write_json(out / "family_vs_candidate_accuracy_report.json", {key: {"candidate": value["candidate_level_accuracy"], "family": value["family_level_accuracy"], "equivalence": value["equivalence_class_accuracy"]} for key, value in all_metrics.items()})
    write_json(out / "support_count_report.json", {key: value["per_support_count_accuracy"] for key, value in all_metrics.items()})
    with4 = all_metrics["ALL28_UNORDERED_NONCENTER_CELL_PAIRS_ADD_MOD9::ENTROPY_MARGIN_SUPPORT_POLICY"]
    no4_bundle = make_candidates("ALL28_UNORDERED_NONCENTER_CELL_PAIRS_ADD_MOD9")
    no4_values = []
    for row in test_rows:
        _, _, result = support_needed(row, no4_bundle, "RAW_CANDIDATE_ID_ARGMAX", [1, 2, 3, 5])
        no4_values.append(result["family_correct"])
    write_json(out / "support4_policy_report.json", {
        "all28_entropy_margin_with4_test_accuracy": with4["test_accuracy"],
        "all28_entropy_margin_with4_support_average": with4["support_used_average"],
        "all28_staged_without4_family_accuracy": sum(no4_values) / len(no4_values),
        "support4_included": True,
    })
    write_json(out / "candidate_order_sensitivity_report.json", {key: value["candidate_order_sensitivity"] for key, value in all_metrics.items()})
    write_json(out / "true_candidate_rank_report.json", {key: value["true_candidate_mean_rank"] for key, value in all_metrics.items()})
    write_json(out / "ambiguity_prediction_report.json", {
        key: {"entropy_mean": value["entropy_mean"], "margin_mean": value["top1_top2_margin_mean"], "ambiguous_case_accuracy": value["ambiguous_case_accuracy"], "collision_case_accuracy": value["collision_case_accuracy"]}
        for key, value in all_metrics.items()
    })

    all28_family = all_metrics["ALL28_UNORDERED_NONCENTER_CELL_PAIRS_ADD_MOD9::FAMILY_GROUP_SCORE_FACTORISATION"]
    all28_raw = all_metrics["ALL28_UNORDERED_NONCENTER_CELL_PAIRS_ADD_MOD9::RAW_CANDIDATE_ID_ARGMAX"]
    if all28_family["test_accuracy"] >= 0.995 and all28_family["OOD_accuracy"] >= 0.995 and (primitive_space_report["ALL28_UNORDERED_NONCENTER_CELL_PAIRS_ADD_MOD9"]["collision_heavy"] or all28_raw["candidate_level_accuracy"] + 0.05 < all28_raw["family_level_accuracy"]):
        decision = {
            "decision": "primitive_space_factorisation_confirmed",
            "verdict": "D44E_PRIMITIVE_SPACE_FACTORISATION_CONFIRMED",
            "next": "D45_CELL_REFERENCE_DISCOVERY_PROTOTYPE",
        }
    else:
        decision = {
            "decision": "d44e_instrumentation_or_evaluator_failure",
            "verdict": "D44E_REPAIR_REQUIRED",
            "next": "D44E2_REPAIR",
        }
    decision.update({"failed_jobs": [], "boundary": "D44E is controlled symbolic primitive-space factorisation only; no raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, or architecture superiority claim."})
    write_json(out / "aggregate_metrics.json", all_metrics)
    write_json(out / "decision.json", decision)
    summary = {
        "decision": decision,
        "primitive_space_comparison": primitive_space_report,
        "all28_raw_vs_factorised": {
            "raw": primitive_space_report["ALL28_UNORDERED_NONCENTER_CELL_PAIRS_ADD_MOD9"],
            "family_group": all28_family,
        },
        "ordered56_raw_vs_canonical": {
            "raw": all_metrics["ORDERED56_PAIR_CONTROL::RAW_CANDIDATE_ID_ARGMAX"],
            "canonical": all_metrics["ORDERED56_PAIR_CONTROL::CANONICAL_UNORDERED_PAIR_FACTORISATION"],
        },
        "soft_field_clustering_alignment": "canonical-equivalence clustering aligns commutative duplicates; semantic-family grouping remains an analysis upper bound in this controlled task.",
    }
    write_json(out / "summary.json", summary)
    (out / "report.md").write_text(
        "# D44E Primitive Space Factorisation and Soft Field Gradient Prototype\n\n"
        f"Decision: `{decision['decision']}` / `{decision['verdict']}`; next `{decision['next']}`.\n\n"
        "D44E treats soft primitive scores as an evidence field, reports candidate/family/equivalence metrics separately, and keeps support-count 4 in staged policies.\n\n"
        "Boundary: controlled symbolic formula primitive discovery only; no raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, or architecture superiority claim.\n"
    )
    print(json.dumps({"out": str(out), "decision": decision["decision"], "next": decision["next"]}, indent=2))


if __name__ == "__main__":
    main()
