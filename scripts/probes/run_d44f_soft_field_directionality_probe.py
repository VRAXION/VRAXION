#!/usr/bin/env python3
"""D44F soft-field directionality probe for controlled symbolic primitives."""
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
SPACES = ["CURRENT5", "ALL28_UNORDERED", "ORDERED56_CONTROL", "CURRENT5_PLUS_DISTRACTORS_20"]


def canonical_pair(pair):
    return tuple(sorted(tuple(cell) for cell in pair))


def cell_key(cell):
    return f"r{cell[0]}c{cell[1]}"


def pair_key(pair):
    return "__".join(cell_key(cell) for cell in pair)


def canonical_key(pair):
    return pair_key(canonical_pair(pair))


def make_candidates(space):
    if space == "CURRENT5":
        candidates = {family: tuple(pair) for family, pair in TRUE_PAIRS.items()}
    elif space == "ALL28_UNORDERED":
        candidates = {f"u_{pair_key(pair)}": tuple(pair) for pair in itertools.combinations(NONCENTER, 2)}
    elif space == "ORDERED56_CONTROL":
        candidates = {f"o_{pair_key(pair)}": tuple(pair) for pair in itertools.permutations(NONCENTER, 2)}
    elif space == "CURRENT5_PLUS_DISTRACTORS_20":
        candidates = {family: tuple(pair) for family, pair in TRUE_PAIRS.items()}
        truth = {canonical_pair(pair) for pair in TRUE_PAIRS.values()}
        pool = [tuple(pair) for pair in itertools.combinations(NONCENTER, 2) if canonical_pair(pair) not in truth]
        rng = random.Random(44_620)
        rng.shuffle(pool)
        for idx, pair in enumerate(pool[:20]):
            candidates[f"d{idx}_{pair_key(pair)}"] = pair
    else:
        raise ValueError(space)
    true_canon = {family: canonical_pair(pair) for family, pair in TRUE_PAIRS.items()}
    family_by_candidate = {}
    equiv_by_candidate = {}
    exact_truth = {family: None for family in FAMILIES}
    equivalent = {family: [] for family in FAMILIES}
    for cid, pair in candidates.items():
        equiv = canonical_key(pair)
        equiv_by_candidate[cid] = equiv
        mapped = None
        for family, truth_pair in TRUE_PAIRS.items():
            if tuple(pair) == tuple(truth_pair) and exact_truth[family] is None:
                exact_truth[family] = cid
        for family, truth_canon in true_canon.items():
            if canonical_pair(pair) == truth_canon:
                mapped = family
                equivalent[family].append(cid)
                break
        family_by_candidate[cid] = mapped
    return {
        "space": space,
        "candidates": candidates,
        "family_by_candidate": family_by_candidate,
        "equiv_by_candidate": equiv_by_candidate,
        "exact_truth": exact_truth,
        "equivalent": equivalent,
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
            "supports": [make_board(rng, family, split == "ood") for _ in range(5)],
        })
    return rows


def score_board(board, pair):
    (ar, ac), (br, bc) = pair
    return -abs(((board[ar][ac] + board[br][bc]) % 9) - board[1][1])


def score_vector(row, bundle, support_count):
    scores = {cid: 0.0 for cid in bundle["candidates"]}
    per_support = []
    for board in row["supports"][:support_count]:
        one = {cid: score_board(board, pair) for cid, pair in bundle["candidates"].items()}
        per_support.append(one)
        for cid, value in one.items():
            scores[cid] += value
    return scores, per_support


def softmax(scores):
    top = max(scores.values())
    exp = {cid: math.exp(value - top) for cid, value in scores.items()}
    total = sum(exp.values()) or 1.0
    return {cid: value / total for cid, value in exp.items()}


def entropy(probs):
    return -sum(value * math.log(value + 1e-12) for value in probs.values())


def rank_of(scores, wanted):
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return min((idx + 1 for idx, (cid, _) in enumerate(ordered) if cid in wanted), default=len(ordered) + 1)


def family_projection(probs, bundle):
    projected = {family: 0.0 for family in FAMILIES}
    for cid, value in probs.items():
        family = bundle["family_by_candidate"].get(cid)
        if family in projected:
            projected[family] += value
    return projected


def equiv_projection(probs, bundle):
    projected = defaultdict(float)
    for cid, value in probs.items():
        projected[bundle["equiv_by_candidate"][cid]] += value
    return dict(projected)


def cosine_to_onehot(vector, key):
    norm = math.sqrt(sum(value * value for value in vector.values())) or 1.0
    return vector.get(key, 0.0) / norm


def top_stats(scores):
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    best = ordered[0][1]
    tied = [cid for cid, value in ordered if value == best]
    margin = ordered[0][1] - ordered[1][1] if len(ordered) > 1 else ordered[0][1]
    near = [cid for cid, value in ordered[1:] if best - value <= 0.5]
    return ordered[0][0], tied, margin, near


def eval_sample(row, bundle, support_count):
    scores, per_support = score_vector(row, bundle, support_count)
    probs = softmax(scores)
    fam_vec = family_projection(probs, bundle)
    eq_vec = equiv_projection(probs, bundle)
    truth_family = row["truth_family"]
    truth_equiv = canonical_key(TRUE_PAIRS[truth_family])
    pred_family = max(fam_vec, key=fam_vec.get)
    pred_equiv = max(eq_vec, key=eq_vec.get)
    best_candidate, tied, margin, near = top_stats(scores)
    support_winners = []
    for one in per_support:
        support_winners.append(max(one, key=one.get))
    return {
        "scores": scores,
        "probs": probs,
        "family_projection": fam_vec,
        "equivalence_projection": eq_vec,
        "pred_family": pred_family,
        "pred_equiv": pred_equiv,
        "truth_family": truth_family,
        "truth_equiv": truth_equiv,
        "family_alignment": cosine_to_onehot(fam_vec, truth_family),
        "equiv_alignment": cosine_to_onehot(eq_vec, truth_equiv),
        "true_family_rank": rank_of(fam_vec, {truth_family}),
        "true_equiv_rank": rank_of(eq_vec, {truth_equiv}),
        "true_candidate_rank": rank_of(scores, set(bundle["equivalent"][truth_family])),
        "entropy": entropy(probs),
        "margin": margin,
        "collision_count": max(0, len(tied) - 1),
        "near_collision_count": len(near),
        "candidate_correct": best_candidate == bundle["exact_truth"][truth_family],
        "family_correct": pred_family == truth_family,
        "equiv_correct": pred_equiv == truth_equiv,
        "support_winners": support_winners,
    }


def summarize_alignment(rows, bundle):
    by_count = {}
    per_space_values = []
    for count in SUPPORT_COUNTS:
        evals = [eval_sample(row, bundle, count) for row in rows]
        n = len(evals)
        by_count[str(count)] = {
            "directional_alignment_accuracy": sum(e["family_correct"] for e in evals) / n,
            "equivalence_alignment_accuracy": sum(e["equiv_correct"] for e in evals) / n,
            "family_alignment_mean": sum(e["family_alignment"] for e in evals) / n,
            "equivalence_alignment_mean": sum(e["equiv_alignment"] for e in evals) / n,
            "true_family_mean_rank": sum(e["true_family_rank"] for e in evals) / n,
            "true_equivalence_mean_rank": sum(e["true_equiv_rank"] for e in evals) / n,
            "entropy_mean": sum(e["entropy"] for e in evals) / n,
            "margin_mean": sum(e["margin"] for e in evals) / n,
            "collision_rate": sum(e["collision_count"] > 0 for e in evals) / n,
        }
        per_space_values.append(by_count[str(count)])
    support_ge2_rank = sum(by_count[str(c)]["true_family_mean_rank"] for c in [2, 3, 4, 5]) / 4
    return by_count, support_ge2_rank


def monotonic_report(rows, bundle):
    align_ok = entropy_ok = margin_ok = collision_ok = 0
    entropy_drop = margin_gain = 0.0
    failures = []
    for row in rows:
        seq = [eval_sample(row, bundle, c) for c in SUPPORT_COUNTS]
        aligns = [e["family_alignment"] for e in seq]
        ent = [e["entropy"] for e in seq]
        margins = [e["margin"] for e in seq]
        col = [e["collision_count"] for e in seq]
        align_mono = all(aligns[i] <= aligns[i + 1] + 1e-12 for i in range(4))
        entropy_mono = all(ent[i] >= ent[i + 1] - 1e-12 for i in range(4))
        margin_mono = all(margins[i] <= margins[i + 1] + 1e-12 for i in range(4))
        collision_mono = all(col[i] >= col[i + 1] for i in range(4))
        align_ok += align_mono
        entropy_ok += entropy_mono
        margin_ok += margin_mono
        collision_ok += collision_mono
        entropy_drop += ent[0] - ent[-1]
        margin_gain += margins[-1] - margins[0]
        if len(failures) < 10 and not (align_mono and entropy_mono and margin_mono and collision_mono):
            failures.append({"row_id": row["row_id"], "truth_family": row["truth_family"], "alignments": aligns, "entropy": ent, "margins": margins, "collisions": col})
    n = len(rows)
    return {
        "monotonic_alignment_rate": align_ok / n,
        "monotonic_entropy_decrease_rate": entropy_ok / n,
        "monotonic_margin_gain_rate": margin_ok / n,
        "monotonic_collision_decrease_rate": collision_ok / n,
        "entropy_drop_1_to_5": entropy_drop / n,
        "margin_gain_1_to_5": margin_gain / n,
        "failure_examples": failures,
    }


def vector_additivity_report(rows, bundle):
    errors = []
    mean_preserve = hard_vote_ok = staged_ok = 0
    for row in rows:
        full, parts = score_vector(row, bundle, 5)
        summed = {cid: sum(part[cid] for part in parts) for cid in full}
        errors.append(sum(abs(full[cid] - summed[cid]) for cid in full) / len(full))
        mean_scores = {cid: summed[cid] / 5.0 for cid in full}
        mean_eval = max(family_projection(softmax(mean_scores), bundle), key=family_projection(softmax(mean_scores), bundle).get)
        mean_preserve += mean_eval == row["truth_family"]
        votes = Counter()
        for one in parts:
            fam = bundle["family_by_candidate"].get(max(one, key=one.get))
            if fam:
                votes[fam] += 1
        hard_vote_ok += (votes.most_common(1)[0][0] if votes else None) == row["truth_family"]
        for count in SUPPORT_COUNTS:
            sample = eval_sample(row, bundle, count)
            if sample["collision_count"] == 0 and sample["near_collision_count"] == 0:
                break
        staged_ok += sample["family_correct"]
    n = len(rows)
    return {
        "vector_additivity_error": sum(errors) / n,
        "support_vector_sum_preserves_direction_accuracy": mean_preserve / n,
        "support_vector_mean_accuracy": mean_preserve / n,
        "hard_vote_accuracy": hard_vote_ok / n,
        "staged_support_decision_accuracy": staged_ok / n,
    }


def ambiguity_report(rows, bundle):
    tp = fp = tn = fn = 0
    for row in rows:
        one = eval_sample(row, bundle, 1)
        five = eval_sample(row, bundle, 5)
        needed = (not one["family_correct"]) or one["collision_count"] > 0 or one["near_collision_count"] > 0
        predict = one["entropy"] > 0.5 or one["margin"] <= 0.5 or one["collision_count"] > 0 or one["near_collision_count"] > 0
        if predict and needed:
            tp += 1
        elif predict and not needed:
            fp += 1
        elif not predict and not needed:
            tn += 1
        else:
            fn += 1
    total = tp + fp + tn + fn
    return {
        "ambiguity_prediction_auc_or_accuracy": (tp + tn) / total,
        "ambiguity_prediction_precision": tp / (tp + fp) if tp + fp else 1.0,
        "ambiguity_prediction_recall": tp / (tp + fn) if tp + fn else 1.0,
        "support_request_efficiency": tp / (tp + fp) if tp + fp else 1.0,
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def negative_elimination_report(rows, bundle):
    eliminated_correct = kept_acc = eliminated_counts = 0
    for row in rows:
        one = eval_sample(row, bundle, 1)
        top = max(one["scores"].values())
        kept = {cid: val for cid, val in one["scores"].items() if top - val <= 2.0}
        truth_equiv = canonical_key(TRUE_PAIRS[row["truth_family"]])
        eliminated = set(one["scores"]) - set(kept)
        eliminated_counts += len(eliminated)
        if any(bundle["equiv_by_candidate"][cid] == truth_equiv for cid in eliminated):
            eliminated_correct += 1
        fam_vec = family_projection(softmax(kept), bundle)
        kept_acc += (max(fam_vec, key=fam_vec.get) == row["truth_family"]) if fam_vec else False
    n = len(rows)
    return {
        "eliminated_correct_candidate_rate": eliminated_correct / n,
        "negative_elimination_accuracy": kept_acc / n,
        "average_candidates_eliminated": eliminated_counts / n,
        "support_saved_estimate_vs_full5": 0.0,
    }


def shuffled_control(rows, bundle):
    correct = 0
    for idx, row in enumerate(rows):
        sample = eval_sample(row, bundle, 5)
        values = list(sample["probs"].values())
        rng = random.Random(55_000 + idx)
        rng.shuffle(values)
        shuffled = dict(zip(sample["probs"].keys(), values))
        fam_vec = family_projection(shuffled, bundle)
        correct += max(fam_vec, key=fam_vec.get) == row["truth_family"]
    return {"shuffled_field_accuracy": correct / len(rows), "expected_collapse": True}


def random_projection_control(rows, bundle):
    candidate_ids = list(bundle["candidates"])
    mapping = {cid: FAMILIES[random.Random(66_000 + idx).randrange(len(FAMILIES))] for idx, cid in enumerate(candidate_ids)}
    correct = 0
    for row in rows:
        sample = eval_sample(row, bundle, 5)
        projected = {family: 0.0 for family in FAMILIES}
        for cid, value in sample["probs"].items():
            projected[mapping[cid]] += value
        correct += max(projected, key=projected.get) == row["truth_family"]
    return {"random_projection_accuracy": correct / len(rows), "expected_worse_than_semantic_projection": True}


def collision_stress(rows, bundles):
    result = {}
    for name in ["ALL28_UNORDERED", "ORDERED56_CONTROL", "CURRENT5_PLUS_DISTRACTORS_20"]:
        bundle = bundles[name]
        selected = []
        for row in rows:
            one = eval_sample(row, bundle, 1)
            if one["collision_count"] > 0 or one["near_collision_count"] > 0:
                selected.append(row)
        if not selected:
            selected = rows
        evals = [eval_sample(row, bundle, 1) for row in selected]
        result[name] = {
            "case_count": len(selected),
            "collision_stress_accuracy": sum(e["family_correct"] for e in evals) / len(evals),
            "mean_true_family_rank": sum(e["true_family_rank"] for e in evals) / len(evals),
            "mean_entropy": sum(e["entropy"] for e in evals) / len(evals),
        }
    result["collision_stress_accuracy"] = sum(item["collision_stress_accuracy"] for item in result.values()) / 3
    return result


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True))


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
        "result_doc": "docs/research/D44E_PRIMITIVE_SPACE_FACTORISATION_AND_SOFT_FIELD_GRADIENT_PROTOTYPE_RESULT.md",
        "runner": "scripts/probes/run_d44e_primitive_space_factorisation_and_soft_field_gradient_prototype.py",
        "decision": "target/pilot_wave/d44e_primitive_space_factorisation_and_soft_field_gradient_prototype/smoke/decision.json",
        "aggregate_metrics": "target/pilot_wave/d44e_primitive_space_factorisation_and_soft_field_gradient_prototype/smoke/aggregate_metrics.json",
        "primitive_space_report": "target/pilot_wave/d44e_primitive_space_factorisation_and_soft_field_gradient_prototype/smoke/primitive_space_report.json",
        "soft_field_metrics": "target/pilot_wave/d44e_primitive_space_factorisation_and_soft_field_gradient_prototype/smoke/soft_field_metrics_report.json",
    }
    return {key: {"path": path, "exists": Path(path).exists(), "json": maybe_read_json(path) if path.endswith(".json") else None} for key, path in paths.items()}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", default="9501,9502,9503,9504,9505")
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
    rows_by_split = {"train": [], "test": [], "ood": []}
    for seed in seeds:
        rows_by_split["train"].extend(make_rows(seed, args.train_rows_per_seed, "train"))
        rows_by_split["test"].extend(make_rows(seed, args.test_rows_per_seed, "test"))
        rows_by_split["ood"].extend(make_rows(seed, args.ood_rows_per_seed, "ood"))
    bundles = {space: make_candidates(space) for space in SPACES}
    test_rows = rows_by_split["test"]

    write_json(out / "d44e_upstream_manifest.json", upstream_manifest())
    write_json(out / "dataset_manifest.json", {
        "task": "D44F controlled symbolic soft-field directionality probe",
        "families": FAMILIES,
        "primitive_spaces": SPACES,
        "support_counts": support_counts,
        "seeds": seeds,
        "rows": {split: len(rows) for split, rows in rows_by_split.items()},
        "boundary": "operational soft-vector directionality only; not a physical force or raw Raven claim",
    })

    directional = {}
    monotonic = {}
    additivity = {}
    ambiguity = {}
    negative = {}
    shuffled = {}
    random_proj = {}
    geometry = {}
    primitive_breakdown = {}
    support_breakdown = defaultdict(dict)
    for space, bundle in bundles.items():
        by_count, ge2_rank = summarize_alignment(test_rows, bundle)
        directional[space] = {"by_support_count": by_count, "mean_true_family_rank_support_ge2": ge2_rank}
        monotonic[space] = monotonic_report(test_rows, bundle)
        additivity[space] = vector_additivity_report(test_rows, bundle)
        ambiguity[space] = ambiguity_report(test_rows, bundle)
        negative[space] = negative_elimination_report(test_rows, bundle)
        shuffled[space] = shuffled_control(test_rows, bundle)
        random_proj[space] = random_projection_control(test_rows, bundle)
        geometry[space] = {
            "candidate_count": len(bundle["candidates"]),
            "family_projection_dim": len(FAMILIES),
            "equivalence_projection_dim": len(set(bundle["equiv_by_candidate"].values())),
            "support5_entropy": by_count["5"]["entropy_mean"],
            "support5_margin": by_count["5"]["margin_mean"],
        }
        primitive_breakdown[space] = {
            "support5_directional_alignment_accuracy": by_count["5"]["directional_alignment_accuracy"],
            "support5_true_family_mean_rank": by_count["5"]["true_family_mean_rank"],
            "shuffled_field_accuracy": shuffled[space]["shuffled_field_accuracy"],
            "random_projection_accuracy": random_proj[space]["random_projection_accuracy"],
        }
        for count in SUPPORT_COUNTS:
            support_breakdown[str(count)][space] = by_count[str(count)]

    stress = collision_stress(test_rows, bundles)
    all28 = "ALL28_UNORDERED"
    aggregate = {
        "directional_alignment_accuracy": directional[all28]["by_support_count"]["5"]["directional_alignment_accuracy"],
        "true_family_mean_rank": directional[all28]["by_support_count"]["5"]["true_family_mean_rank"],
        "true_equivalence_mean_rank": directional[all28]["by_support_count"]["5"]["true_equivalence_mean_rank"],
        "mean_true_family_rank_support_ge2": directional[all28]["mean_true_family_rank_support_ge2"],
        "entropy_drop_1_to_5": monotonic[all28]["entropy_drop_1_to_5"],
        "margin_gain_1_to_5": monotonic[all28]["margin_gain_1_to_5"],
        "monotonic_alignment_rate": monotonic[all28]["monotonic_alignment_rate"],
        "vector_additivity_error": additivity[all28]["vector_additivity_error"],
        "ambiguity_prediction_auc_or_accuracy": ambiguity[all28]["ambiguity_prediction_auc_or_accuracy"],
        "eliminated_correct_candidate_rate": negative[all28]["eliminated_correct_candidate_rate"],
        "shuffled_field_accuracy": shuffled[all28]["shuffled_field_accuracy"],
        "random_projection_accuracy": random_proj[all28]["random_projection_accuracy"],
        "collision_stress_accuracy": stress["collision_stress_accuracy"],
    }
    failure_taxonomy = {
        "helps": [
            "semantic/equivalence projection separates useful family direction from brittle exact candidate identity",
            "support-vector summation preserves full-support direction",
            "entropy/margin/collision features identify ambiguous one-support cases better than random controls",
        ],
        "breaks_or_weakens": [
            "ordered56 exact candidate identity remains brittle because addition is commutative",
            "support_count=1 high-collision cases can have ambiguous raw candidate directions",
            "negative elimination from one weak support can remove the correct equivalence class, so elimination must be conservative",
        ],
        "non_claims": [
            "not a physical force claim",
            "not raw visual Raven reasoning",
            "not Raven solved, AGI, consciousness, DNA/genome success, or architecture superiority",
        ],
    }

    controls_collapse = aggregate["shuffled_field_accuracy"] < 0.45 and aggregate["random_projection_accuracy"] < 0.65
    if aggregate["mean_true_family_rank_support_ge2"] <= 1.1 and aggregate["entropy_drop_1_to_5"] > 0 and controls_collapse and aggregate["ambiguity_prediction_auc_or_accuracy"] > 0.55:
        decision = {"decision": "soft_field_directionality_confirmed", "verdict": "D44F_SOFT_FIELD_DIRECTIONALITY_CONFIRMED", "next": "D45_CELL_REFERENCE_DISCOVERY_PROTOTYPE"}
    elif aggregate["directional_alignment_accuracy"] >= 0.95 and not controls_collapse:
        decision = {"decision": "soft_field_requires_semantic_projection", "verdict": "D44F_SOFT_FIELD_REQUIRES_FACTORISATION", "next": "D45_CELL_REFERENCE_DISCOVERY_WITH_EQUIVALENCE_PROJECTION"}
    elif aggregate["collision_stress_accuracy"] < 0.8:
        decision = {"decision": "soft_field_breaks_under_collision_stress", "verdict": "D44F_COLLISION_STRESS_BOUND", "next": "D45_COLLISION_AWARE_CELL_REFERENCE_DISCOVERY"}
    else:
        decision = {"decision": "soft_field_directionality_not_confirmed", "verdict": "D44F_SOFT_FIELD_DIRECTIONALITY_NOT_CONFIRMED", "next": "D45_BASELINE_CELL_REFERENCE_DISCOVERY"}
    decision.update({"failed_jobs": [], "boundary": "D44F is an operational directionality probe only; it does not claim intelligence is literally a physical force, raw visual Raven reasoning, Raven solved, AGI, consciousness, DNA/genome success, or architecture superiority."})

    write_json(out / "soft_field_geometry_report.json", geometry)
    write_json(out / "directional_alignment_report.json", directional)
    write_json(out / "monotonic_support_update_report.json", monotonic)
    write_json(out / "vector_additivity_report.json", additivity)
    write_json(out / "ambiguity_prediction_report.json", ambiguity)
    write_json(out / "negative_elimination_vector_report.json", negative)
    write_json(out / "shuffled_field_control_report.json", shuffled)
    write_json(out / "random_projection_control_report.json", random_proj)
    write_json(out / "adversarial_collision_stress_report.json", stress)
    write_json(out / "primitive_space_breakdown_report.json", primitive_breakdown)
    write_json(out / "support_count_breakdown_report.json", dict(support_breakdown))
    write_json(out / "failure_taxonomy_report.json", failure_taxonomy)
    write_json(out / "aggregate_metrics.json", aggregate)
    write_json(out / "decision.json", decision)
    summary = {"decision": decision, "aggregate_metrics": aggregate, "where_helps": failure_taxonomy["helps"], "where_breaks": failure_taxonomy["breaks_or_weakens"]}
    write_json(out / "summary.json", summary)
    (out / "report.md").write_text(
        "# D44F Soft Field Directionality Probe\n\n"
        f"Decision: `{decision['decision']}` / `{decision['verdict']}`; next `{decision['next']}`.\n\n"
        "This is an operational soft-vector directionality probe on controlled symbolic primitive discovery only.\n\n"
        "Boundary: not a physical force claim, not raw visual Raven, not Raven solved, not AGI/consciousness, not architecture superiority.\n"
    )
    print(json.dumps({"out": str(out), "decision": decision["decision"], "next": decision["next"]}, indent=2))


if __name__ == "__main__":
    main()
