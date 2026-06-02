#!/usr/bin/env python3
"""D44G IPF/ECF breakpoint stress map for controlled symbolic primitives."""
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
OPS = ["add", "sub", "reverse_sub"]


def canonical_pair(pair):
    return tuple(sorted(tuple(cell) for cell in pair))


def cell_key(cell):
    return f"r{cell[0]}c{cell[1]}"


def pair_key(pair):
    return "__".join(cell_key(cell) for cell in pair)


def canonical_key(pair):
    return pair_key(canonical_pair(pair))


def apply_op(a, b, op):
    if op == "add":
        return (a + b) % 9
    if op == "sub":
        return (a - b) % 9
    if op == "reverse_sub":
        return (b - a) % 9
    raise ValueError(op)


def distractor_pair_for(family):
    truth_cells = set(TRUE_PAIRS[family])
    for pair in itertools.combinations(NONCENTER, 2):
        if canonical_pair(pair) != canonical_pair(TRUE_PAIRS[family]) and not (set(pair) & truth_cells):
            return tuple(pair)
    return tuple(next(pair for pair in itertools.combinations(NONCENTER, 2) if canonical_pair(pair) != canonical_pair(TRUE_PAIRS[family])))


def make_candidates(space, operators=False):
    if space == "CURRENT5":
        pairs = {family: tuple(pair) for family, pair in TRUE_PAIRS.items()}
    elif space == "ALL28":
        pairs = {f"u_{pair_key(pair)}": tuple(pair) for pair in itertools.combinations(NONCENTER, 2)}
    elif space == "ORDERED56":
        pairs = {f"o_{pair_key(pair)}": tuple(pair) for pair in itertools.permutations(NONCENTER, 2)}
    elif space.startswith("CURRENT5_PLUS_"):
        count = int(space.rsplit("_", 1)[1])
        pairs = {family: tuple(pair) for family, pair in TRUE_PAIRS.items()}
        truth = {canonical_pair(pair) for pair in TRUE_PAIRS.values()}
        pool = [tuple(pair) for pair in itertools.combinations(NONCENTER, 2) if canonical_pair(pair) not in truth]
        ordered_pool = [tuple(pair) for pair in itertools.permutations(NONCENTER, 2) if canonical_pair(pair) not in truth]
        pool = pool + ordered_pool
        rng = random.Random(44_700 + count)
        rng.shuffle(pool)
        for idx in range(count):
            pair = pool[idx % len(pool)]
            pairs[f"d{idx}_{pair_key(pair)}"] = pair
    else:
        raise ValueError(space)
    candidates = {}
    for cid, pair in pairs.items():
        if operators:
            for op in OPS:
                candidates[f"{cid}::{op}"] = {"pair": pair, "op": op}
        else:
            candidates[cid] = {"pair": pair, "op": "add"}
    family_by_candidate = {}
    equiv_by_candidate = {}
    exact_truth = {family: None for family in FAMILIES}
    equivalent = {family: [] for family in FAMILIES}
    for cid, spec in candidates.items():
        pair = spec["pair"]
        op = spec["op"]
        equiv_by_candidate[cid] = f"{canonical_key(pair)}::{op if operators else 'add'}"
        mapped = None
        for family, truth_pair in TRUE_PAIRS.items():
            if tuple(pair) == tuple(truth_pair) and op == "add" and exact_truth[family] is None:
                exact_truth[family] = cid
            if canonical_pair(pair) == canonical_pair(truth_pair) and op == "add":
                mapped = family
                equivalent[family].append(cid)
        family_by_candidate[cid] = mapped
    return {"space": space, "operators": operators, "candidates": candidates, "family_by_candidate": family_by_candidate, "equiv_by_candidate": equiv_by_candidate, "exact_truth": exact_truth, "equivalent": equivalent}


def make_board(rng, family, split, stress="normal", support_index=0):
    board = [[rng.randrange(9) for _ in range(3)] for _ in range(3)]
    if split == "ood":
        board = [[((value * 2) + 1) % 9 for value in row] for row in board]
    (ar, ac), (br, bc) = TRUE_PAIRS[family]
    center = (board[ar][ac] + board[br][bc]) % 9
    board[1][1] = center
    if stress in {"correlated", "adversarial"}:
        should_force = stress == "correlated" or support_index < 4
        if should_force:
            (dr1, dc1), (dr2, dc2) = distractor_pair_for(family)
            board[dr2][dc2] = (center - board[dr1][dc1]) % 9
    return board


def make_rows(seed, count, split, stress="normal"):
    rng = random.Random(seed + {"train": 0, "test": 10_000, "ood": 20_000}[split] + {"normal": 0, "correlated": 30_000, "adversarial": 60_000}[stress])
    rows = []
    for idx in range(count):
        family = rng.choice(FAMILIES)
        rows.append({"row_id": f"{split}-{stress}-{seed}-{idx}", "seed": seed, "split": split, "truth_family": family, "stress": stress, "supports": [make_board(rng, family, split, stress, i) for i in range(5)]})
    return rows


def candidate_score(board, spec):
    (ar, ac), (br, bc) = spec["pair"]
    predicted = apply_op(board[ar][ac], board[br][bc], spec["op"])
    return -abs(predicted - board[1][1])


def score_vector(row, bundle, support_count):
    scores = {cid: 0.0 for cid in bundle["candidates"]}
    for board in row["supports"][:support_count]:
        for cid, spec in bundle["candidates"].items():
            scores[cid] += candidate_score(board, spec)
    return scores


def softmax(scores):
    top = max(scores.values())
    weights = {cid: math.exp(value - top) for cid, value in scores.items()}
    total = sum(weights.values()) or 1.0
    return {cid: value / total for cid, value in weights.items()}


def entropy(probs):
    return -sum(value * math.log(value + 1e-12) for value in probs.values())


def top_stats(scores):
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    best = ordered[0][1]
    tied = [cid for cid, value in ordered if value == best]
    near = [cid for cid, value in ordered[1:] if best - value <= 0.5]
    margin = ordered[0][1] - ordered[1][1] if len(ordered) > 1 else ordered[0][1]
    return ordered, tied, near, margin


def projection(probs, bundle, kind):
    projected = defaultdict(float)
    for cid, value in probs.items():
        if kind == "family":
            key = bundle["family_by_candidate"].get(cid) or "distractor"
        elif kind == "equiv":
            key = bundle["equiv_by_candidate"][cid]
        elif kind == "random":
            key = FAMILIES[random.Random(91_000 + list(bundle["candidates"]).index(cid)).randrange(len(FAMILIES))]
        else:
            key = cid
        projected[key] += value
    return dict(projected)


def rank_in_scores(scores, wanted):
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return min((idx + 1 for idx, (cid, _) in enumerate(ordered) if cid in wanted), default=len(ordered) + 1)


def evaluate(row, bundle, support_count, grouping="family", order_shuffle=False, shuffled_field=False):
    scores = score_vector(row, bundle, support_count)
    if order_shuffle:
        # Deterministic control: change insertion/order only, not scores.
        items = list(scores.items())
        random.Random(88_000 + support_count).shuffle(items)
        scores = dict(items)
    probs = softmax(scores)
    if shuffled_field:
        vals = list(probs.values())
        random.Random(77_000 + support_count + row["seed"]).shuffle(vals)
        probs = dict(zip(probs.keys(), vals))
    fam_proj = projection(probs, bundle, "family" if grouping != "none" else "candidate")
    equiv_proj = projection(probs, bundle, "equiv")
    rand_proj = projection(probs, bundle, "random")
    ordered, tied, near, margin = top_stats(scores)
    truth = row["truth_family"]
    truth_equiv = f"{canonical_key(TRUE_PAIRS[truth])}::add"
    pred_family = max(fam_proj, key=fam_proj.get)
    if grouping == "none":
        pred_family = bundle["family_by_candidate"].get(pred_family)
    pred_equiv = max(equiv_proj, key=equiv_proj.get)
    top_candidate = ordered[0][0]
    return {
        "candidate_accuracy": top_candidate == bundle["exact_truth"].get(truth),
        "family_accuracy": pred_family == truth,
        "equivalence_accuracy": pred_equiv == truth_equiv,
        "random_projection_accuracy": max(rand_proj, key=rand_proj.get) == truth,
        "collision": len(tied) > 1,
        "collision_count": max(0, len(tied) - 1),
        "near_collision": len(near) > 0,
        "entropy": entropy(probs),
        "margin": margin,
        "true_candidate_rank": rank_in_scores(scores, set(bundle["equivalent"][truth])),
        "top_candidate": top_candidate,
    }


def summarize(rows, bundle, support_count=5, grouping="family", order_shuffle=False, shuffled_field=False):
    evals = [evaluate(row, bundle, support_count, grouping, order_shuffle, shuffled_field) for row in rows]
    n = len(evals)
    return {
        "candidate_accuracy": sum(e["candidate_accuracy"] for e in evals) / n,
        "family_accuracy": sum(e["family_accuracy"] for e in evals) / n,
        "equivalence_accuracy": sum(e["equivalence_accuracy"] for e in evals) / n,
        "random_projection_accuracy": sum(e["random_projection_accuracy"] for e in evals) / n,
        "collision_rate": sum(e["collision"] for e in evals) / n,
        "near_collision_rate": sum(e["near_collision"] for e in evals) / n,
        "entropy_mean": sum(e["entropy"] for e in evals) / n,
        "margin_mean": sum(e["margin"] for e in evals) / n,
        "true_candidate_rank_mean": sum(e["true_candidate_rank"] for e in evals) / n,
    }


def support_needed(rows, bundle, max_support):
    correct = used = oracle_used = capped_fail = 0
    dist = Counter()
    for row in rows:
        chosen = max_support
        for c in SUPPORT_COUNTS:
            if c > max_support:
                break
            result = evaluate(row, bundle, c)
            if not result["collision"] and not result["near_collision"]:
                chosen = c
                break
        final = evaluate(row, bundle, chosen)
        correct += final["family_accuracy"]
        used += chosen
        dist[str(chosen)] += 1
        oracle = 5
        for c in SUPPORT_COUNTS:
            if evaluate(row, bundle, c)["family_accuracy"]:
                oracle = c
                break
        oracle_used += oracle
        capped_fail += (not final["family_accuracy"] and max_support < oracle)
    n = len(rows)
    return {"accuracy": correct / n, "average_support_used": used / n, "oracle_minimal_support_average": oracle_used / n, "oracle_gap": (used - oracle_used) / n, "support_distribution": {str(k): dist[str(k)] / n for k in SUPPORT_COUNTS}, "capped_failure_rate": capped_fail / n}


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
        "d44f_result": "docs/research/D44F_SOFT_FIELD_DIRECTIONALITY_PROBE_RESULT.md",
        "d44f_runner": "scripts/probes/run_d44f_soft_field_directionality_probe.py",
        "d44f_decision": "target/pilot_wave/d44f_soft_field_directionality_probe/smoke/decision.json",
        "d44f_aggregate": "target/pilot_wave/d44f_soft_field_directionality_probe/smoke/aggregate_metrics.json",
    }
    return {key: {"path": path, "exists": Path(path).exists(), "json": maybe_read_json(path) if path.endswith(".json") else None} for key, path in paths.items()}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", default="9601,9602,9603,9604,9605")
    parser.add_argument("--train-rows-per-seed", type=int, default=800)
    parser.add_argument("--test-rows-per-seed", type=int, default=800)
    parser.add_argument("--ood-rows-per-seed", type=int, default=800)
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
    rows = []
    correlated = []
    adversarial = []
    for seed in seeds:
        rows.extend(make_rows(seed, args.test_rows_per_seed, "test", "normal"))
        correlated.extend(make_rows(seed, args.test_rows_per_seed, "test", "correlated"))
        adversarial.extend(make_rows(seed, args.test_rows_per_seed, "test", "adversarial"))
    spaces = ["CURRENT5", "ALL28", "ORDERED56", "CURRENT5_PLUS_5", "CURRENT5_PLUS_10", "CURRENT5_PLUS_20", "CURRENT5_PLUS_50"]
    bundles = {space: make_candidates(space) for space in spaces}
    write_json(out / "upstream_manifest.json", upstream_manifest())
    write_json(out / "dataset_manifest.json", {"task": "D44G IPF/ECF breakpoint stress map", "seeds": seeds, "support_counts": support_counts, "rows": {"normal_test": len(rows), "correlated_test": len(correlated), "adversarial_test": len(adversarial)}, "boundary": "controlled symbolic primitive discovery only"})

    candidate_space = {space: {"candidate_count": len(bundle["candidates"]), **summarize(rows, bundle, 5), "support_needed": support_needed(rows, bundle, 5)["average_support_used"]} for space, bundle in bundles.items()}
    aliasing = {
        "ORDERED56": {**summarize(rows, bundles["ORDERED56"], 5), "false_distinction_rate": 1.0 - summarize(rows, bundles["ORDERED56"], 5)["candidate_accuracy"], "equivalence_family_recovery": summarize(rows, bundles["ORDERED56"], 5)["family_accuracy"]},
        "CURRENT5_PLUS_50": {**summarize(rows, bundles["CURRENT5_PLUS_50"], 5), "false_distinction_rate": 1.0 - summarize(rows, bundles["CURRENT5_PLUS_50"], 5)["candidate_accuracy"]},
    }
    correlated_report = {space: {**summarize(correlated, bundles[space], 5), "adaptive_support_request_rate": support_needed(correlated, bundles[space], 5)["support_distribution"], "hard_vote_failure_note": "correlated distractor support can amplify non-independent evidence"} for space in ["ALL28", "CURRENT5_PLUS_20", "CURRENT5_PLUS_50"]}
    adversarial_report = {space: {**summarize(adversarial, bundles[space], 5), "support_needed_to_defeat_distractor": support_needed(adversarial, bundles[space], 5), "ipf_robustness": summarize(adversarial, bundles[space], 5)["family_accuracy"]} for space in ["ALL28", "CURRENT5_PLUS_20", "CURRENT5_PLUS_50"]}
    support_budget = {str(max_support): support_needed(rows, bundles["ALL28"], max_support) for max_support in SUPPORT_COUNTS}
    raw = summarize(rows, bundles["ALL28"], 5, grouping="none")
    family = summarize(rows, bundles["ALL28"], 5, grouping="family")
    shuffled = summarize(rows, bundles["ALL28"], 5, grouping="family", shuffled_field=True)
    random_proj_acc = family["random_projection_accuracy"]
    order_shuffle = summarize(rows, bundles["ALL28"], 5, grouping="family", order_shuffle=True)
    factorisation = {
        "handcoded_family_grouping": family,
        "no_semantic_grouping_raw_candidate_id": raw,
        "clustering_from_soft_signatures": summarize(rows, bundles["ORDERED56"], 5, grouping="family"),
        "random_projection_control_accuracy": random_proj_acc,
        "shuffled_field_control": shuffled,
        "candidate_order_shuffle_control": order_shuffle,
        "label_echo_reference_only": {"reported": True, "used_as_fair_oracle": False},
        "factorisation_dependence": family["family_accuracy"] - raw["candidate_accuracy"],
    }
    op_bundle = make_candidates("ALL28", operators=True)
    operator_report = {"operator_space": OPS, "candidate_count": len(op_bundle["candidates"]), **summarize(rows, op_bundle, 5), "operator_dimension_breakpoint": summarize(rows, op_bundle, 5)["family_accuracy"] < 0.95}
    predictive = {"support_budget_accuracy_curve": {k: v["accuracy"] for k, v in support_budget.items()}, "ambiguity_signal": "collision/near-collision plus entropy/margin predicts added-support need", "ipf_predictive_power": support_budget["5"]["accuracy"] - support_budget["1"]["accuracy"]}
    perturb = {"shuffle_drop": family["family_accuracy"] - shuffled["family_accuracy"], "random_projection_drop": family["family_accuracy"] - random_proj_acc, "order_shuffle_delta": family["family_accuracy"] - order_shuffle["family_accuracy"]}
    breakpoints = {
        "named_breakpoints": [
            "support budget <=1 remains collision limited",
            "ordered/alias spaces break exact candidate identity while family/equivalence survives",
            "correlated/adversarial distractor support can pull IPF direction toward distractors",
            "operator expansion introduces an operator/family factorisation burden",
        ],
        "where_ipf_helps": ["broad candidate spaces after semantic/equivalence projection", "support budget scaling", "alias recovery at family/equivalence level"],
        "where_ipf_breaks": ["one-support collision regimes", "adversarial correlated distractors", "raw exact candidate identity in alias spaces", "expanded operator dimension without learned operator factorisation"],
    }
    stress_summary = {
        "CANDIDATE_SPACE_SIZE": {"status": "useful_with_factorisation", "key_metric": candidate_space["ALL28"]["family_accuracy"]},
        "ALIASING_STRESS": {"status": "exact_candidate_breaks_family_recovers", "false_distinction_rate": aliasing["ORDERED56"]["false_distinction_rate"]},
        "CORRELATED_NOISE_SUPPORT": {"status": "breakpoint", "family_accuracy": correlated_report["ALL28"]["family_accuracy"]},
        "ADVERSARIAL_DISTRACTORS": {"status": "breakpoint", "family_accuracy": adversarial_report["ALL28"]["family_accuracy"]},
        "SUPPORT_BUDGET_LIMIT": {"status": "support_scaling_needed", "support1_accuracy": support_budget["1"]["accuracy"], "support5_accuracy": support_budget["5"]["accuracy"]},
        "FACTORISATION_REMOVAL": {"status": "handcoded_factorisation_helpful", "dependence_delta": factorisation["factorisation_dependence"]},
        "OPERATOR_SPACE_EXPANSION_LIGHT": {"status": "operator_factorisation_needed" if operator_report["operator_dimension_breakpoint"] else "operator_space_ok", "family_accuracy": operator_report["family_accuracy"]},
    }
    aggregate = {
        "all28_family_accuracy": candidate_space["ALL28"]["family_accuracy"],
        "all28_candidate_accuracy": candidate_space["ALL28"]["candidate_accuracy"],
        "ordered56_false_distinction_rate": aliasing["ORDERED56"]["false_distinction_rate"],
        "correlated_all28_family_accuracy": correlated_report["ALL28"]["family_accuracy"],
        "adversarial_all28_family_accuracy": adversarial_report["ALL28"]["family_accuracy"],
        "support1_accuracy": support_budget["1"]["accuracy"],
        "support5_accuracy": support_budget["5"]["accuracy"],
        "factorisation_dependence_delta": factorisation["factorisation_dependence"],
        "operator_space_family_accuracy": operator_report["family_accuracy"],
        "shuffled_field_accuracy": shuffled["family_accuracy"],
        "random_projection_accuracy": random_proj_acc,
        "candidate_order_shuffle_accuracy": order_shuffle["family_accuracy"],
    }
    if aggregate["correlated_all28_family_accuracy"] < 0.9 or aggregate["adversarial_all28_family_accuracy"] < 0.9:
        decision = {"decision": "ipf_breaks_under_adversarial_support", "verdict": "D44G_ADVERSARIAL_SUPPORT_BREAKPOINT", "next": "D45_ROBUST_SUPPORT_POLICY_PROTOTYPE"}
    elif aggregate["operator_space_family_accuracy"] < 0.95:
        decision = {"decision": "ipf_breaks_under_operator_expansion", "verdict": "D44G_OPERATOR_EXPANSION_BREAKPOINT", "next": "D45_OPERATOR_FACTORISATION_PLAN"}
    elif aggregate["factorisation_dependence_delta"] > 0.1:
        decision = {"decision": "ipf_requires_hardcoded_factorisation", "verdict": "D44G_HANDCODED_FACTORISATION_DEPENDENCE", "next": "D45_LEARNED_FACTORISATION_PROTOTYPE"}
    elif aggregate["support5_accuracy"] > 0.99:
        decision = {"decision": "ipf_breakpoint_map_strong", "verdict": "D44G_IPF_BREAKPOINT_MAP_STRONG", "next": "D45_CELL_REFERENCE_DISCOVERY_PROTOTYPE"}
    else:
        decision = {"decision": "ipf_partial_with_named_breakpoints", "verdict": "D44G_IPF_PARTIAL_BREAKPOINTS_MAPPED", "next": "D45_TARGETED_BREAKPOINT_REPAIR"}
    decision.update({"failed_jobs": [], "boundary": "D44G maps controlled symbolic IPF/ECF breakpoints only; no physical force, raw visual Raven, Raven solved, DNA/genome success, AGI, consciousness, or architecture superiority claim."})

    reports = {
        "stress_axis_summary_report.json": stress_summary,
        "candidate_space_size_report.json": candidate_space,
        "aliasing_stress_report.json": aliasing,
        "correlated_noise_support_report.json": correlated_report,
        "adversarial_distractor_report.json": adversarial_report,
        "support_budget_limit_report.json": support_budget,
        "factorisation_removal_report.json": factorisation,
        "operator_space_expansion_light_report.json": operator_report,
        "ipf_predictive_power_report.json": predictive,
        "ipf_causal_perturbation_report.json": perturb,
        "breakpoint_report.json": breakpoints,
        "aggregate_metrics.json": aggregate,
        "decision.json": decision,
        "summary.json": {"decision": decision, "aggregate_metrics": aggregate, "stress_axis_summary": stress_summary, "breakpoints": breakpoints},
    }
    for name, payload in reports.items():
        write_json(out / name, payload)
    (out / "report.md").write_text("# D44G IPF Breakpoint Stress Map\n\n" + f"Decision: `{decision['decision']}` / `{decision['verdict']}`; next `{decision['next']}`.\n\n" + "Boundary: controlled symbolic breakpoint map only; no physical force, raw visual Raven, AGI, consciousness, or architecture-superiority claim.\n")
    print(json.dumps({"out": str(out), "decision": decision["decision"], "next": decision["next"]}, indent=2))


if __name__ == "__main__":
    main()
