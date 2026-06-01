#!/usr/bin/env python3
"""D50 joint formula discovery scale confirm."""

import argparse
import json
import os
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import run_d49_joint_cell_operator_discovery_with_robust_support as d49
import run_d49b_joint_binding_repair as d49b

PRIMARY_SPACE = "ALL28_UNORDERED_X_OPS"
SUPPORT_COUNT = 5
ROW_SAMPLE_PER_ARM_REGIME_SPLIT = 24
CONFIDENCE_THRESHOLD = 0.45

BOUNDARY = (
    "D50 only scale-confirms controlled symbolic joint formula discovery with robust ECF support. "
    "It does not prove raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, "
    "or architecture superiority."
)

REGIMES = d49b.REGIMES
CORE_REGIMES = d49b.CORE_REGIMES
OP_NAMES = d49b.OP_NAMES

ARMS = [
    "D49B_BASELINE_REPLAY",
    "JOINT_INTERACTION_COUNTERFACTUAL",
    "MULTI_STAGE_COUNTERFACTUAL_REPAIR",
    "FULL_REPAIRED_ECF_CONTROLLER",
    "FULL_REPAIRED_ECF_CAP_7",
    "FULL_REPAIRED_ECF_CAP_9",
    "NO_CELL_COUNTERFACTUAL",
    "NO_OPERATOR_COUNTERFACTUAL",
    "NO_JOINT_INTERACTION_COUNTERFACTUAL",
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

MAPPED_ARMS = {
    "D49B_BASELINE_REPLAY": "D49_BASELINE_REPLAY",
    "JOINT_INTERACTION_COUNTERFACTUAL": "JOINT_INTERACTION_COUNTERFACTUAL",
    "MULTI_STAGE_COUNTERFACTUAL_REPAIR": "MULTI_STAGE_COUNTERFACTUAL_REPAIR",
    "FULL_REPAIRED_ECF_CONTROLLER": "FULL_REPAIRED_ECF_CONTROLLER",
    "FULL_REPAIRED_ECF_CAP_7": "FULL_REPAIRED_ECF_CAP_7",
    "FULL_REPAIRED_ECF_CAP_9": "FULL_REPAIRED_ECF_CAP_9",
    "RANDOM_EXTRA_SUPPORT_CONTROL": "RANDOM_EXTRA_SUPPORT_CONTROL",
    "SHUFFLED_COUNTER_SUPPORT_CONTROL": "SHUFFLED_COUNTER_SUPPORT_CONTROL",
    "NO_COUNTERFACTUAL_CONTROL": "NO_COUNTERFACTUAL_CONTROL",
    "ABSTAIN_ON_INDISTINGUISHABLE": "ABSTAIN_ON_INDISTINGUISHABLE",
}

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


def cell_key(cell):
    return d49.cell_key(cell)


def support_cap_for_arm(arm):
    if arm == "FULL_REPAIRED_ECF_CAP_7":
        return 7
    if arm == "FULL_REPAIRED_ECF_CAP_9":
        return 9
    if arm in {
        "JOINT_INTERACTION_COUNTERFACTUAL",
        "MULTI_STAGE_COUNTERFACTUAL_REPAIR",
        "FULL_REPAIRED_ECF_CONTROLLER",
        "NO_CELL_COUNTERFACTUAL",
        "NO_OPERATOR_COUNTERFACTUAL",
        "NO_JOINT_INTERACTION_COUNTERFACTUAL",
        "SHUFFLED_COUNTER_SUPPORT_CONTROL",
        "RANDOM_EXTRA_SUPPORT_CONTROL",
        "ABSTAIN_ON_INDISTINGUISHABLE",
    }:
        return 12
    return SUPPORT_COUNT


def stage_plan_for_ablation(arm):
    if arm == "NO_CELL_COUNTERFACTUAL":
        return [("operator", 1), ("joint", 4)]
    if arm == "NO_OPERATOR_COUNTERFACTUAL":
        return [("cell", 1), ("joint", 4)]
    if arm == "NO_JOINT_INTERACTION_COUNTERFACTUAL":
        return [("cell", 2), ("operator", 2)]
    raise ValueError(arm)


def classify_error(row, pred, abstained, exact_joint, pair_equiv, op_exact):
    return d49b.classify_error(row, pred, abstained, exact_joint, pair_equiv, op_exact)


def build_result(row, bundle, arm, pred, final_scores, scalar_pred, base_vectors, stage_counts, external_used, abstained):
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
    cluster_count, dominant_fraction, collision_count = d49b.cluster_stats(base_vectors)
    taxonomy = classify_error(row, pred, abstained, exact_joint, pair_ok, op_exact)
    return {
        "row_id": row["row_id"],
        "seed": row["seed"],
        "split": row["split"],
        "arm": arm,
        "primitive_space": PRIMARY_SPACE,
        "support_regime": row["support_regime"],
        "truth_joint": row["truth_joint"],
        "pred_joint": pred["pred_joint"],
        "truth_pair": [cell_key(cell) for cell in row["truth_pair"]],
        "pred_pair": [cell_key(cell) for cell in pred["pred_pair"]] if pred["pred_pair"] else [],
        "truth_pair_equivalence": row["truth_pair_equivalence"],
        "pred_pair_equivalence": pred["pred_pair_equivalence"],
        "truth_operator": row["true_operator"],
        "pred_operator": pred["pred_operator"],
        "truth_operator_equivalence": row["truth_operator_equivalence"],
        "pred_operator_equivalence": pred["pred_operator_equivalence"],
        "exact_joint_correct": exact_joint,
        "cell_pair_equivalence_correct": pair_ok,
        "cell_hit_top2": cell_hit,
        "cell_hit_top2_correct": cell_hit >= 1.0,
        "operator_exact_correct": op_exact,
        "operator_equivalence_correct": op_equiv,
        "family_group_correct": group_correct,
        "joint_binding_consistency": bool(exact_joint or (pair_ok and op_exact)),
        "correct": correct,
        "reference_arm": False,
        "support_budget_cap": support_cap_for_arm(arm),
        "original_support_used": SUPPORT_COUNT,
        "cell_counter_support_used": stage_counts["cell"],
        "operator_counter_support_used": stage_counts["operator"],
        "joint_counter_support_used": stage_counts["joint"],
        "random_counter_support_used": stage_counts["random"],
        "shuffled_counter_support_used": stage_counts["shuffled_joint"],
        "counter_support_used": sum(stage_counts.values()) - external_used,
        "external_test_used": external_used,
        "total_support_used": SUPPORT_COUNT + sum(stage_counts.values()),
        "support_cluster_count": cluster_count,
        "dominant_cluster_fraction": dominant_fraction,
        "collision_count": collision_count,
        "correlated_echo_detected": dominant_fraction >= 0.60 and len(base_vectors) >= 3,
        "counter_support_requested": d49b.counter_needed(row, scalar_pred, base_vectors),
        "counter_support_resolved": sum(stage_counts.values()) > 0 and exact_joint,
        "oracle_distinguishable": row["oracle_distinguishable"],
        "external_test_available": row["external_test_available"],
        "abstained": abstained,
        "false_confidence": false_conf,
        "confidence": pred["confidence"],
        "top1_top2_margin": pred["top1_top2_margin"],
        "entropy": pred["entropy"],
        "score_gap_truth_vs_wrong": d49b.score_gap(final_scores, row["truth_joint"]) if not abstained else 0.0,
        "baseline_exact_correct": scalar_pred["pred_joint"] == row["truth_joint"],
        "error_type": taxonomy,
    }


def evaluate_ablation_arm(row, bundle, arm):
    cap = support_cap_for_arm(arm)
    base_vectors = d49b.cached_base_vectors(row, bundle, SUPPORT_COUNT)
    scalar_scores = d49b.aggregate_sum(base_vectors)
    scalar_pred = d49b.predict(scalar_scores, bundle)
    need_counter = d49b.counter_needed(row, scalar_pred, base_vectors)
    extra = []
    stage_counts = Counter()
    external_used = 0
    abstained = False
    if row["support_regime"] == "INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT":
        abstained = True
    elif row["external_test_available"] and arm != "NO_JOINT_INTERACTION_COUNTERFACTUAL":
        vectors = d49b.make_stage_vectors(row, bundle, scalar_pred, "joint", 4, external=True)
        d49b.add_stage(extra, stage_counts, "external", vectors, cap, SUPPORT_COUNT)
        external_used = stage_counts["external"]
    elif need_counter:
        for stage, count in stage_plan_for_ablation(arm):
            vectors = d49b.make_stage_vectors(row, bundle, scalar_pred, stage, count)
            d49b.add_stage(extra, stage_counts, stage, vectors, cap, SUPPORT_COUNT)
    final_scores = d49b.aggregate_duplicate_downweighted(base_vectors + extra)
    pred = d49b.predict(final_scores, bundle, abstain=abstained)
    return build_result(row, bundle, arm, pred, final_scores, scalar_pred, base_vectors, stage_counts, external_used, abstained)


def evaluate_arm(row, bundle, arm):
    if arm in MAPPED_ARMS:
        result = d49b.evaluate_arm(row, bundle, MAPPED_ARMS[arm])
        result["arm"] = arm
        return result
    return evaluate_ablation_arm(row, bundle, arm)


def init_worker(bundle):
    global GLOBAL_BUNDLE
    GLOBAL_BUNDLE = bundle


def evaluate_row_all_arms(row):
    return [evaluate_arm(row, GLOBAL_BUNDLE, arm) for arm in ARMS]


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
    }


def summarize_outputs(outputs):
    by_arm = defaultdict(list)
    by_arm_regime = defaultdict(list)
    by_arm_core = defaultdict(list)
    by_seed_arm_core = defaultdict(list)
    by_seed_arm_regime = defaultdict(list)
    by_error = defaultdict(Counter)
    for row in outputs:
        by_arm[row["arm"]].append(row)
        by_arm_regime[(row["arm"], row["support_regime"])].append(row)
        by_error[(row["arm"], row["support_regime"])][row["error_type"]] += 1
        if row["support_regime"] in CORE_REGIMES:
            by_arm_core[row["arm"]].append(row)
            by_seed_arm_core[(row["seed"], row["arm"])].append(row)
        by_seed_arm_regime[(row["seed"], row["arm"], row["support_regime"])].append(row)
    seeds = sorted({row["seed"] for row in outputs})
    return {
        "by_arm": {arm: summarize(by_arm[arm]) for arm in ARMS if arm in by_arm},
        "by_arm_core": {arm: summarize(by_arm_core[arm]) for arm in ARMS if arm in by_arm_core},
        "by_arm_and_regime": {
            arm: {regime: summarize(by_arm_regime[(arm, regime)]) for regime in REGIMES if (arm, regime) in by_arm_regime}
            for arm in ARMS
        },
        "by_seed": {
            str(seed): {
                "FULL_REPAIRED_ECF_CONTROLLER": {
                    "core": summarize(by_seed_arm_core[(seed, "FULL_REPAIRED_ECF_CONTROLLER")]),
                    "correlated_echo": summarize(by_seed_arm_regime[(seed, "FULL_REPAIRED_ECF_CONTROLLER", "CORRELATED_ECHO_SUPPORT")]),
                    "adversarial_distractor": summarize(by_seed_arm_regime[(seed, "FULL_REPAIRED_ECF_CONTROLLER", "ADVERSARIAL_DISTRACTOR_SUPPORT")]),
                }
            }
            for seed in seeds
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


def min_seed_metrics(metrics):
    seeds = metrics["by_seed"]
    exact = [item["FULL_REPAIRED_ECF_CONTROLLER"]["core"]["exact_joint_accuracy"] for item in seeds.values()]
    corr = [item["FULL_REPAIRED_ECF_CONTROLLER"]["correlated_echo"]["accuracy"] for item in seeds.values()]
    adv = [item["FULL_REPAIRED_ECF_CONTROLLER"]["adversarial_distractor"]["accuracy"] for item in seeds.values()]
    return {
        "min_seed_exact_joint": min(exact) if exact else 0.0,
        "min_seed_correlated_echo": min(corr) if corr else 0.0,
        "min_seed_adversarial_distractor": min(adv) if adv else 0.0,
    }


def make_decision(metrics, failed_jobs):
    full = metrics["by_arm_core"]["FULL_REPAIRED_ECF_CONTROLLER"]
    mins = min_seed_metrics(metrics)
    corr = regime_accuracy(metrics, "FULL_REPAIRED_ECF_CONTROLLER", "CORRELATED_ECHO_SUPPORT")
    adv = regime_accuracy(metrics, "FULL_REPAIRED_ECF_CONTROLLER", "ADVERSARIAL_DISTRACTOR_SUPPORT")
    external = regime_accuracy(metrics, "FULL_REPAIRED_ECF_CONTROLLER", "EXTERNAL_TEST_REQUIRED_SUPPORT")
    indist = metrics["by_arm_and_regime"]["FULL_REPAIRED_ECF_CONTROLLER"]["INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT"]
    controls_worse = all(full["accuracy"] > metrics["by_arm_core"][arm]["accuracy"] for arm in CONTROL_ARMS)
    no_joint = metrics["by_arm_core"]["NO_JOINT_INTERACTION_COUNTERFACTUAL"]
    joint = metrics["by_arm_core"]["JOINT_INTERACTION_COUNTERFACTUAL"]
    if failed_jobs:
        return {
            "decision": "joint_formula_discovery_scale_not_confirmed",
            "verdict": "D50_FAILED_JOBS_PRESENT",
            "next": "D50_REPAIR",
            "boundary": BOUNDARY,
        }
    pass_gate = (
        full["exact_joint_accuracy"] >= 0.995
        and corr >= 0.995
        and adv >= 0.995
        and external >= 0.99
        and indist["false_confidence_rate"] <= 0.01
        and mins["min_seed_exact_joint"] >= 0.99
        and controls_worse
    )
    if pass_gate:
        if full["average_total_support_used"] > 9.0:
            return {
                "decision": "joint_formula_discovery_scale_confirmed_high_cost",
                "verdict": "D50_JOINT_FORMULA_DISCOVERY_SCALE_CONFIRMED_HIGH_COST",
                "next": "D50C_SUPPORT_COST_OPTIMIZATION",
                "boundary": BOUNDARY,
            }
        if no_joint["exact_joint_accuracy"] < 0.95 and joint["exact_joint_accuracy"] >= 0.995:
            return {
                "decision": "joint_interaction_counterfactual_required_confirmed",
                "verdict": "D50_JOINT_INTERACTION_COUNTERFACTUAL_REQUIRED_CONFIRMED",
                "next": "D51_MUTABLE_JOINT_COUNTERFACTUAL_POLICY",
                "boundary": BOUNDARY,
            }
        return {
            "decision": "joint_formula_discovery_scale_confirmed",
            "verdict": "D50_JOINT_FORMULA_DISCOVERY_SCALE_CONFIRMED",
            "next": "D51_MUTABLE_ECF_CONTROLLER_PROTOTYPE",
            "boundary": BOUNDARY,
        }
    return {
        "decision": "joint_formula_discovery_scale_not_confirmed",
        "verdict": "D50_JOINT_FORMULA_DISCOVERY_SCALE_NOT_CONFIRMED",
        "next": "D50_REPAIR",
        "boundary": BOUNDARY,
    }


def make_upstream_manifest(repo_root):
    root = repo_root / "target/pilot_wave/d49b_joint_binding_repair/smoke"
    manifest = {
        "upstream": "D49B_JOINT_BINDING_REPAIR",
        "expected_decision": "joint_binding_repair_positive",
        "expected_next": "D50_JOINT_FORMULA_DISCOVERY_SCALE_CONFIRM",
        "decision_present": (root / "decision.json").exists(),
        "summary_present": (root / "summary.json").exists(),
    }
    if (root / "decision.json").exists():
        manifest["decision_json"] = json.loads((root / "decision.json").read_text(encoding="utf-8"))
    if (root / "summary.json").exists():
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        full = summary["key_metrics"]["full_repaired"]
        manifest["key_metrics"] = {
            "exact_joint": full["exact_joint_accuracy"],
            "cell": full["cell_pair_equivalence_accuracy"],
            "operator": full["operator_exact_accuracy"],
            "support": full["average_total_support_used"],
            "correlated_echo": summary["key_metrics"]["full_by_regime"]["CORRELATED_ECHO_SUPPORT"]["accuracy"],
            "adversarial_distractor": summary["key_metrics"]["full_by_regime"]["ADVERSARIAL_DISTRACTOR_SUPPORT"]["accuracy"],
            "external_test_required": summary["key_metrics"]["external_required"]["accuracy"],
            "indistinguishable_abstain": summary["key_metrics"]["indistinguishable"]["abstain_rate"],
            "false_confidence": summary["key_metrics"]["indistinguishable"]["false_confidence_rate"],
        }
    return manifest


def write_report(out, decision, aggregate):
    core = aggregate["test_metrics"]["by_arm_core"]
    regimes = aggregate["test_metrics"]["by_arm_and_regime"]["FULL_REPAIRED_ECF_CONTROLLER"]
    lines = [
        "# D50 Joint Formula Discovery Scale Confirm Result",
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
        f"Scale mode: `{aggregate['scale_mode']}`",
        "",
        "Core arm table:",
        "",
        "```text",
    ]
    for arm in ARMS:
        row = core[arm]
        lines.append(
            f"{arm}: exact={row['exact_joint_accuracy']:.4f}, cell={row['cell_pair_equivalence_accuracy']:.4f}, op={row['operator_exact_accuracy']:.4f}, support={row['average_total_support_used']:.3f}"
        )
    lines.extend(["```", "", "FULL_REPAIRED_ECF_CONTROLLER by regime:", "", "```text"])
    for regime in REGIMES:
        row = regimes[regime]
        lines.append(
            f"{regime}: acc={row['accuracy']:.4f}, abstain={row['abstain_rate']:.4f}, false_conf={row['false_confidence_rate']:.4f}, support={row['average_total_support_used']:.3f}"
        )
    lines.extend(["```", "", f"Boundary: {BOUNDARY}", ""])
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", default="10401,10402,10403,10404,10405,10406,10407,10408")
    parser.add_argument("--train-rows-per-seed", type=int, default=1200)
    parser.add_argument("--test-rows-per-seed", type=int, default=1200)
    parser.add_argument("--ood-rows-per-seed", type=int, default=1200)
    parser.add_argument("--workers", default="auto")
    parser.add_argument("--cpu-target", default="saturate")
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    parser.add_argument("--scale-mode", default="full", choices=["full", "scale_lite"])
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
            "task": "D50 joint formula discovery scale confirm",
            "status": "running",
            "scale_mode": args.scale_mode,
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
    append_progress(out, "queue_written", started, {"out": str(out), "scale_mode": args.scale_mode})
    write_json(out / "d49b_upstream_manifest.json", make_upstream_manifest(repo_root))
    write_json(
        out / "dataset_manifest.json",
        {
            "task": "D50_JOINT_FORMULA_DISCOVERY_SCALE_CONFIRM",
            "scale_mode": args.scale_mode,
            "primary_space": PRIMARY_SPACE,
            "candidate": "ALL28 cell pairs x D48 operator set",
            "operator_candidates": OP_NAMES,
            "joint_candidate_count_primary": len(bundle["candidates"]),
            "support_regimes": REGIMES,
            "arms": ARMS,
            "truth_hidden_from_fair_arms": True,
            "candidate_cell_operator_equivalence_metrics_separated": True,
            "indistinguishable_abstain_false_confidence_included": True,
            "controls_required": True,
            "failed_jobs_visible": True,
            "row_outputs_are_sampled_but_metrics_use_full_rows": True,
            "no_python_hash": True,
            "no_fake_sampling": True,
            "boundary": BOUNDARY,
        },
    )
    write_json(
        out / "compute_probe.json",
        {
            "mode": "local_python_cpu",
            "workers_requested": args.workers,
            "cpu_target": args.cpu_target,
            "note": "D50 is deterministic symbolic scoring; no external model/API/download used.",
        },
    )
    train_rows = d49b.make_rows_with_progress(seeds, args.train_rows_per_seed, "train", bundle, out, started, args.heartbeat_sec)
    test_rows = d49b.make_rows_with_progress(seeds, args.test_rows_per_seed, "test", bundle, out, started, args.heartbeat_sec)
    ood_rows = d49b.make_rows_with_progress(seeds, args.ood_rows_per_seed, "ood", bundle, out, started, args.heartbeat_sec)
    write_json(
        out / "train_manifest.json",
        {"train_rows": len(train_rows), "note": "Train rows generated for dataset parity; D50 arms are non-learned symbolic policies."},
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
    controls_worse = all(full["accuracy"] > test_metrics["by_arm_core"][arm]["accuracy"] for arm in CONTROL_ARMS)
    mins = min_seed_metrics(test_metrics)
    aggregate = {
        "task": "D50_JOINT_FORMULA_DISCOVERY_SCALE_CONFIRM",
        "scale_mode": args.scale_mode,
        "test_metrics": test_metrics,
        "ood_metrics": ood_metrics,
        "primary_policy_metrics": test_metrics["by_arm_core"],
        "min_seed_metrics": mins,
        "controls_worse": controls_worse,
        "failed_jobs": failed_jobs,
        "decision": decision["decision"],
        "verdict": decision["verdict"],
        "next": decision["next"],
        "boundary": BOUNDARY,
    }
    reports = {
        "scale_summary.json": {
            "scale_mode": args.scale_mode,
            "seeds": seeds,
            "train_rows": len(train_rows),
            "test_rows": len(test_rows),
            "ood_rows": len(ood_rows),
            "min_seed_metrics": mins,
            "full_repaired": full,
        },
        "component_ablation_report.json": {
            arm: test_metrics["by_arm_core"][arm]
            for arm in [
                "JOINT_INTERACTION_COUNTERFACTUAL",
                "MULTI_STAGE_COUNTERFACTUAL_REPAIR",
                "FULL_REPAIRED_ECF_CONTROLLER",
                "NO_CELL_COUNTERFACTUAL",
                "NO_OPERATOR_COUNTERFACTUAL",
                "NO_JOINT_INTERACTION_COUNTERFACTUAL",
            ]
        },
        "support_cost_frontier_report.json": {
            "cap7": test_metrics["by_arm_core"]["FULL_REPAIRED_ECF_CAP_7"],
            "cap9": test_metrics["by_arm_core"]["FULL_REPAIRED_ECF_CAP_9"],
            "full": full,
        },
        "regime_breakdown_report.json": test_metrics["by_arm_and_regime"],
        "indistinguishability_report.json": {
            arm: test_metrics["by_arm_and_regime"][arm]["INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT"] for arm in ARMS
        },
        "external_test_required_report.json": {
            arm: test_metrics["by_arm_and_regime"][arm]["EXTERNAL_TEST_REQUIRED_SUPPORT"] for arm in ARMS
        },
        "error_taxonomy_report.json": test_metrics["error_taxonomy"],
        "control_report.json": {arm: test_metrics["by_arm_core"][arm] for arm in CONTROL_ARMS},
    }
    for name, value in reports.items():
        write_json(out / name, value)
    write_json(out / "aggregate_metrics.json", aggregate)
    write_json(out / "decision.json", decision)
    write_json(
        out / "summary.json",
        {
            "scale_mode": args.scale_mode,
            "decision": decision,
            "key_metrics": {
                "full_repaired": full,
                "full_by_regime": test_metrics["by_arm_and_regime"]["FULL_REPAIRED_ECF_CONTROLLER"],
                "cap7": test_metrics["by_arm_core"]["FULL_REPAIRED_ECF_CAP_7"],
                "cap9": test_metrics["by_arm_core"]["FULL_REPAIRED_ECF_CAP_9"],
                "component_ablation": reports["component_ablation_report.json"],
                "indistinguishable": reports["indistinguishability_report.json"]["FULL_REPAIRED_ECF_CONTROLLER"],
                "external_required": reports["external_test_required_report.json"]["FULL_REPAIRED_ECF_CONTROLLER"],
                "controls": reports["control_report.json"],
                "min_seed_metrics": mins,
            },
            "boundary": BOUNDARY,
        },
    )
    write_report(out, decision, aggregate)
    queue = json.loads((out / "queue.json").read_text(encoding="utf-8"))
    write_json(out / "queue.json", {**queue, "status": "complete"})
    append_progress(out, "final_decision", started, decision)
    print(json.dumps({"decision": decision["decision"], "verdict": decision["verdict"], "next": decision["next"], "scale_mode": args.scale_mode}, indent=2))


if __name__ == "__main__":
    main()
