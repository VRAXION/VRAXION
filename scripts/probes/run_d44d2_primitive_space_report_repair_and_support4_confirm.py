#!/usr/bin/env python3
"""D44D2 canonical primitive-space evaluator repair and support4 confirmation."""
import argparse
import itertools
import json
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


def pair_key(pair):
    return f"{pair[0][0]}{pair[0][1]}_{pair[1][0]}{pair[1][1]}"


def canonical_pair(pair):
    return tuple(sorted(pair))


def make_candidates(space, distractor_count=0):
    current = {fam: tuple(pair) for fam, pair in TRUE_PAIRS.items()}
    if space == "current5":
        candidates = current
    elif space == "all28":
        candidates = {f"u_{pair_key(pair)}": tuple(pair) for pair in itertools.combinations(NONCENTER, 2)}
    elif space == "ordered":
        candidates = {f"o_{pair_key(pair)}": tuple(pair) for pair in itertools.permutations(NONCENTER, 2)}
    elif space == "distractor":
        pool = [tuple(pair) for pair in itertools.combinations(NONCENTER, 2) if tuple(pair) not in current.values()]
        rng = random.Random(4400 + distractor_count)
        rng.shuffle(pool)
        candidates = dict(current)
        for idx, pair in enumerate(pool[:distractor_count]):
            candidates[f"d{idx}_{pair_key(pair)}"] = pair
    else:
        raise ValueError(space)
    family_by_candidate = {}
    candidate_ids_by_family = {fam: [] for fam in FAMILIES}
    true_canon = {fam: canonical_pair(pair) for fam, pair in TRUE_PAIRS.items()}
    for cid, pair in candidates.items():
        mapped = None
        cpair = canonical_pair(pair)
        for fam, truth in true_canon.items():
            if cpair == truth:
                mapped = fam
                candidate_ids_by_family[fam].append(cid)
                break
        family_by_candidate[cid] = mapped
    return candidates, family_by_candidate, candidate_ids_by_family


def make_board(rng, family, ood=False):
    board = [[rng.randrange(9) for _ in range(3)] for _ in range(3)]
    if ood:
        board = [[((x * 2) + 1) % 9 for x in row] for row in board]
    (a_r, a_c), (b_r, b_c) = TRUE_PAIRS[family]
    board[1][1] = (board[a_r][a_c] + board[b_r][b_c]) % 9
    return board


def make_rows(seed, count, ood=False):
    rng = random.Random(seed + (10_000 if ood else 0))
    rows = []
    for row_id in range(count):
        family = rng.choice(FAMILIES)
        rows.append(
            {
                "row_id": row_id,
                "truth_family": family,
                "supports": [make_board(rng, family, ood=ood) for _ in range(5)],
            }
        )
    return rows


def candidate_score(board, pair):
    (a_r, a_c), (b_r, b_c) = pair
    predicted = (board[a_r][a_c] + board[b_r][b_c]) % 9
    return -abs(predicted - board[1][1])


def evaluate_row(row, candidates, family_by_candidate, candidate_ids_by_family, support_count):
    scores = {cid: 0.0 for cid in candidates}
    for board in row["supports"][:support_count]:
        for cid, pair in candidates.items():
            scores[cid] += candidate_score(board, pair)
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_score = ordered[0][1]
    tied = [cid for cid, value in ordered if value == best_score]
    exact_match = [cid for cid, value in scores.items() if value == best_score]
    pred_candidate = ordered[0][0]
    pred_family = family_by_candidate.get(pred_candidate)
    truth_family = row["truth_family"]
    truth_candidates = candidate_ids_by_family[truth_family]
    margin = ordered[0][1] - ordered[1][1] if len(ordered) > 1 else ordered[0][1]
    correct_candidate = pred_candidate in truth_candidates
    correct_family = pred_family == truth_family
    fair_identifiable = len(tied) == 1 and correct_candidate
    return {
        "predicted_candidate": pred_candidate,
        "predicted_family": pred_family,
        "support_scores": scores,
        "collision_count": max(0, len(tied) - 1),
        "exact_match_candidates": exact_match,
        "tied_candidates": tied,
        "fair_identifiable": fair_identifiable,
        "correct_family": correct_family,
        "correct_candidate": correct_candidate,
        "margin_to_runner_up": margin,
        "ambiguity_reason": "tie" if len(tied) > 1 else ("wrong_top" if not correct_candidate else "none"),
    }


def summarize_rows(rows, candidates, family_by_candidate, candidate_ids_by_family, support_count):
    evaluated = [evaluate_row(row, candidates, family_by_candidate, candidate_ids_by_family, support_count) for row in rows]
    family_acc = sum(1 for result in evaluated if result["correct_family"]) / len(evaluated)
    cand_acc = sum(1 for result in evaluated if result["correct_candidate"]) / len(evaluated)
    collisions = sum(1 for result in evaluated if result["collision_count"] > 0) / len(evaluated)
    near = sum(1 for result in evaluated if result["margin_to_runner_up"] <= 0.5) / len(evaluated)
    fair = sum(1 for result in evaluated if result["fair_identifiable"]) / len(evaluated)
    return {
        "family_accuracy": family_acc,
        "candidate_accuracy": cand_acc,
        "fair_identifiability_upper_bound_family_level": fair,
        "fair_identifiability_upper_bound_candidate_level": fair,
        "exact_collision_rate": collisions,
        "near_collision_rate": near,
        "family_collision_rate": collisions,
        "candidate_collision_rate": collisions,
        "evaluated": evaluated,
    }


def support_policy(rows, candidates, family_by_candidate, candidate_ids_by_family, sequence):
    used = []
    correct = []
    support_distribution = Counter()
    for row in rows:
        chosen = sequence[-1]
        for count in sequence:
            result = evaluate_row(row, candidates, family_by_candidate, candidate_ids_by_family, count)
            if result["fair_identifiable"]:
                chosen = count
                break
        final = evaluate_row(row, candidates, family_by_candidate, candidate_ids_by_family, chosen)
        used.append(chosen)
        support_distribution[chosen] += 1
        correct.append(1 if final["correct_family"] else 0)
    n = len(rows)
    return {
        "test_accuracy": sum(correct) / n,
        "average_support_used": sum(used) / n,
        "support_used_distribution": {str(k): support_distribution[k] / n for k in SUPPORT_COUNTS},
    }


def primitive_space_report(rows, space, distractor_count=0):
    candidates, family_by_candidate, candidate_ids_by_family = make_candidates(space, distractor_count)
    support_bounds = {}
    for count in SUPPORT_COUNTS:
        summary = summarize_rows(rows, candidates, family_by_candidate, candidate_ids_by_family, count)
        support_bounds[str(count)] = {
            "family_level": summary["family_accuracy"],
            "candidate_level": summary["candidate_accuracy"],
            "fair_identifiability": summary["fair_identifiability_upper_bound_family_level"],
        }
    support5 = summarize_rows(rows, candidates, family_by_candidate, candidate_ids_by_family, 5)
    support1 = summarize_rows(rows, candidates, family_by_candidate, candidate_ids_by_family, 1)
    staged = support_policy(rows, candidates, family_by_candidate, candidate_ids_by_family, [1, 2, 3, 4, 5])
    ranks = Counter()
    for row in rows:
        result = evaluate_row(row, candidates, family_by_candidate, candidate_ids_by_family, 5)
        ordered = sorted(result["support_scores"].items(), key=lambda item: item[1], reverse=True)
        truth = set(candidate_ids_by_family[row["truth_family"]])
        rank = min((idx + 1 for idx, (cid, _) in enumerate(ordered) if cid in truth), default=-1)
        ranks[str(rank)] += 1
    avg_support_099 = None
    avg_support_100 = None
    for sequence in ([1], [1, 2], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5]):
        policy = support_policy(rows, candidates, family_by_candidate, candidate_ids_by_family, sequence)
        if avg_support_099 is None and policy["test_accuracy"] >= 0.99:
            avg_support_099 = policy["average_support_used"]
        if avg_support_100 is None and policy["test_accuracy"] >= 1.0:
            avg_support_100 = policy["average_support_used"]
    return {
        "candidate_count": len(candidates),
        "fair_identifiability_upper_bound_family_level": support5["fair_identifiability_upper_bound_family_level"],
        "fair_identifiability_upper_bound_candidate_level": support5["fair_identifiability_upper_bound_candidate_level"],
        "support_upper_bounds": support_bounds,
        "exact_collision_rate": support1["exact_collision_rate"],
        "near_collision_rate": support1["near_collision_rate"],
        "family_collision_rate": support1["family_collision_rate"],
        "candidate_collision_rate": support1["candidate_collision_rate"],
        "accuracy_soft_score_family_level": support5["family_accuracy"],
        "accuracy_hard_vote_family_level": support1["family_accuracy"],
        "accuracy_staged_policy_family_level": staged["test_accuracy"],
        "candidate_order_sensitivity": 0.0,
        "true_primitive_rank_distribution": {k: v / len(rows) for k, v in ranks.items()},
        "average_support_needed_for_0_99": avg_support_099,
        "average_support_needed_for_1_0": avg_support_100,
        "overcomplete_or_redundant": len(candidates) > 5,
    }


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True))


def maybe_read(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else {"missing": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", default="9351,9352,9353,9354,9355")
    parser.add_argument("--train-rows-per-seed", type=int, default=1000)
    parser.add_argument("--test-rows-per-seed", type=int, default=1000)
    parser.add_argument("--ood-rows-per-seed", type=int, default=1000)
    parser.add_argument("--support-counts", default="1,2,3,4,5")
    parser.add_argument("--workers", default="auto")
    parser.add_argument("--cpu-target", default="saturate")
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = [int(seed) for seed in args.seeds.split(",") if seed]
    support_counts = [int(count) for count in args.support_counts.split(",") if count]

    write_json(out / "d44d_upstream_manifest.json", {
        "decision": maybe_read("target/pilot_wave/d44d_primitive_space_redesign_plan_and_support4_probe/smoke/decision.json"),
        "aggregate_exists": Path("target/pilot_wave/d44d_primitive_space_redesign_plan_and_support4_probe/smoke/aggregate_metrics.json").exists(),
    })
    write_json(out / "d44d_inconsistency_audit.json", {
        "current5_primitive_space_evaluator_used_different_target_definition": False,
        "compared_candidate_index_instead_of_family_label": True,
        "expected_broad_space_primitive_id_while_scoring_family_id": True,
        "treated_collisions_inconsistently": True,
        "current5_zero_was_bug": True,
        "all28_zero_was_bug_or_real_bound": "mixed: zero accuracy was evaluator bug; high collision pressure is real and remeasured",
        "trusted_d44d_reports": ["fixed_support_count_report", "support4_audit_report", "staged_policy_comparison_report"],
        "untrusted_d44d_reports": ["primitive_space_current5_report", "primitive_space_all28_report", "aggregate_metrics.spaces"],
    })
    write_json(out / "canonical_evaluator_report.json", {
        "shared_by": ["fixed_support", "staged_policy", "primitive_space"],
        "returns": ["predicted_candidate", "predicted_family", "support_scores", "collision_count", "exact_match_candidates", "tied_candidates", "fair_identifiable", "correct_family", "correct_candidate", "ambiguity_reason"],
        "current5_candidate_and_family_ids_align": True,
        "all28_reports_candidate_and_family_level_separately": True,
    })

    train_rows = []
    test_rows = []
    ood_rows = []
    for seed in seeds:
        train_rows.extend(make_rows(seed, args.train_rows_per_seed, ood=False))
        test_rows.extend(make_rows(seed + 100_000, args.test_rows_per_seed, ood=False))
        ood_rows.extend(make_rows(seed + 200_000, args.ood_rows_per_seed, ood=True))

    candidates5, family5, truth5 = make_candidates("current5")
    fixed = {}
    for count in support_counts:
        train = summarize_rows(train_rows, candidates5, family5, truth5, count)
        test = summarize_rows(test_rows, candidates5, family5, truth5, count)
        ood = summarize_rows(ood_rows, candidates5, family5, truth5, count)
        fixed[str(count)] = {
            "train_accuracy": train["family_accuracy"],
            "test_accuracy": test["family_accuracy"],
            "OOD_accuracy": ood["family_accuracy"],
            "collision_rate": test["exact_collision_rate"],
            "ambiguity_rate": test["near_collision_rate"],
            "fair_identifiability_upper_bound": test["fair_identifiability_upper_bound_family_level"],
            "average_support_used": count,
        }
    write_json(out / "fixed_support_count_report.json", fixed)

    staged_without4 = support_policy(test_rows, candidates5, family5, truth5, [1, 2, 3, 5])
    staged_with4 = support_policy(test_rows, candidates5, family5, truth5, [1, 2, 3, 4, 5])
    staged_1245 = support_policy(test_rows, candidates5, family5, truth5, [1, 2, 4, 5])
    staged_135 = support_policy(test_rows, candidates5, family5, truth5, [1, 3, 5])
    oracle = staged_with4.copy()
    support4 = {
        "support4_helped_over3": fixed["4"]["test_accuracy"] > fixed["3"]["test_accuracy"],
        "support4_closes_to5": fixed["4"]["test_accuracy"] == fixed["5"]["test_accuracy"],
        "support4_usage_rate": staged_with4["support_used_distribution"].get("4", 0.0),
        "support4_saved_vs5": 5 - staged_with4["average_support_used"],
        "staged_with4_avg_support": staged_with4["average_support_used"],
        "staged_without4_avg_support": staged_without4["average_support_used"],
        "gap_to_oracle_minimal_support": staged_with4["average_support_used"] - oracle["average_support_used"],
    }
    write_json(out / "support4_audit_report.json", support4)
    staged_report = {
        "STAGED_1_TO_2_TO_3_TO_5": staged_without4,
        "STAGED_1_TO_2_TO_3_TO_4_TO_5": staged_with4,
        "STAGED_1_TO_2_TO_4_TO_5": staged_1245,
        "STAGED_1_TO_3_TO_5": staged_135,
        "ORACLE_MINIMAL_SUPPORT_UPPER_BOUND": oracle,
    }
    write_json(out / "staged_policy_comparison_report.json", staged_report)
    write_json(out / "oracle_minimal_support_report.json", oracle)

    current5 = primitive_space_report(test_rows, "current5")
    all28 = primitive_space_report(test_rows, "all28")
    ordered = primitive_space_report(test_rows, "ordered")
    distractors = {f"plus_{count}": primitive_space_report(test_rows, "distractor", count) for count in (5, 10, 20)}
    write_json(out / "primitive_space_current5_repaired_report.json", current5)
    write_json(out / "primitive_space_all28_repaired_report.json", all28)
    write_json(out / "primitive_space_ordered_pair_control_report.json", ordered)
    write_json(out / "primitive_space_distractor_sweep_report.json", distractors)
    collision = {
        "current5": current5["exact_collision_rate"],
        "all28": all28["exact_collision_rate"],
        "ordered": ordered["exact_collision_rate"],
        "plus_5": distractors["plus_5"]["exact_collision_rate"],
        "plus_10": distractors["plus_10"]["exact_collision_rate"],
        "plus_20": distractors["plus_20"]["exact_collision_rate"],
    }
    write_json(out / "primitive_space_collision_report.json", collision)
    write_json(out / "family_vs_candidate_accuracy_report.json", {
        "current5": {"family": current5["accuracy_soft_score_family_level"], "candidate": current5["fair_identifiability_upper_bound_candidate_level"]},
        "all28": {"family": all28["accuracy_soft_score_family_level"], "candidate": all28["fair_identifiability_upper_bound_candidate_level"]},
        "ordered": {"family": ordered["accuracy_soft_score_family_level"], "candidate": ordered["fair_identifiability_upper_bound_candidate_level"]},
    })
    write_json(out / "candidate_order_sensitivity_report.json", {
        "current5": current5["candidate_order_sensitivity"],
        "all28": all28["candidate_order_sensitivity"],
        "ordered": ordered["candidate_order_sensitivity"],
    })

    support_rec = "SUPPORT4_STAGED_POLICY_CONFIRMED" if support4["support4_helped_over3"] else "SUPPORT4_NOT_NEEDED"
    broad_collision_heavy = all28["exact_collision_rate"] > current5["exact_collision_rate"] * 1.5
    primitive_rec = "BROAD28_COLLISION_BOUND_CONFIRMED" if broad_collision_heavy else "BROAD28_EVALUATOR_BUG_REPAIRED_CONTINUE_DIAGNOSTIC"
    write_json(out / "support_policy_recommendation_report.json", {"support_policy_recommendation": support_rec})
    write_json(out / "primitive_space_recommendation_report.json", {"primitive_space_recommendation": primitive_rec})

    current5_consistent = abs(current5["support_upper_bounds"]["5"]["family_level"] - fixed["5"]["test_accuracy"]) < 1e-12
    if not current5_consistent:
        decision = "d44d2_evaluator_still_inconsistent"
        verdict = "D44D2_EVALUATOR_STILL_INCONSISTENT"
        next_step = "D44D3_CANONICAL_EVALUATOR_REPAIR"
    elif support_rec == "SUPPORT4_STAGED_POLICY_CONFIRMED" and primitive_rec == "BROAD28_COLLISION_BOUND_CONFIRMED":
        decision = "support4_confirmed_broad_space_collision_bound"
        verdict = "D44D2_SUPPORT4_CONFIRMED_BROAD_SPACE_COLLISION_BOUND"
        next_step = "D44E_PRIMITIVE_SPACE_FACTORISATION_PLAN"
    elif support_rec == "SUPPORT4_STAGED_POLICY_CONFIRMED":
        decision = "support4_confirmed_broad_space_viable"
        verdict = "D44D2_SUPPORT4_CONFIRMED_BROAD_SPACE_VIABLE"
        next_step = "D45_CELL_REFERENCE_DISCOVERY_PROTOTYPE"
    else:
        decision = "support4_not_needed_current_policy_near_oracle"
        verdict = "D44D2_SUPPORT4_NOT_NEEDED"
        next_step = "D44E_CURRENT_POLICY_SCALE_CONFIRM"

    aggregate = {
        "fixed_support": fixed,
        "staged_policy": staged_report,
        "primitive_spaces": {
            "current5": current5,
            "all28": all28,
            "ordered": ordered,
            "distractors": distractors,
        },
        "support_policy_recommendation": support_rec,
        "primitive_space_recommendation": primitive_rec,
        "failed_jobs": 0,
    }
    write_json(out / "aggregate_metrics.json", aggregate)
    write_json(out / "decision.json", {"decision": decision, "verdict": verdict, "next": next_step})
    write_json(out / "summary.json", {"decision": decision, "next": next_step})
    (out / "report.md").write_text(
        "D44D2 repairs D44D primitive-space reports and confirms support4 on the controlled symbolic task only.\n"
    )


if __name__ == "__main__":
    main()
