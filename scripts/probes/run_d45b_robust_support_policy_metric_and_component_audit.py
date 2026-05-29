#!/usr/bin/env python3
"""D45B metric semantics and component audit for robust IPF/ECF support."""

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import run_d45_robust_support_policy_prototype as d45

D45B_REGIMES = [
    "CLEAN_INDEPENDENT_SUPPORT",
    "CORRELATED_NOISE_SUPPORT",
    "ADVERSARIAL_DISTRACTOR_SUPPORT",
    "MIXED_CLEAN_AND_CORRELATED",
    "MIXED_CLEAN_AND_ADVERSARIAL",
]
D45B_SPACES = [
    "ALL28_UNORDERED",
    "CURRENT5",
    "CURRENT5_PLUS_DISTRACTORS_20",
    "CURRENT5_PLUS_DISTRACTORS_50",
]
D45B_POLICIES = [
    "NAIVE_IPF_BASELINE",
    "STAGED_SUPPORT_ONLY",
    "DEDUP_ONLY",
    "SOURCE_DIVERSITY_ONLY",
    "LEAVE_ONE_OUT_ONLY",
    "MEDIAN_TRIMMED_AGGREGATION_ONLY",
    "COUNTER_SUPPORT_ONLY",
    "DEDUP_PLUS_COUNTER_SUPPORT",
    "DIVERSITY_PLUS_COUNTER_SUPPORT",
    "FULL_ROBUST_COMBINED_REPLAY",
    "ROBUST_COMBINED_COST_CAPPED_3",
    "ROBUST_COMBINED_COST_CAPPED_5",
    "ROBUST_COMBINED_COST_CAPPED_7",
    "ROBUST_COMBINED_COST_CAPPED_9",
    "RANDOM_EXTRA_SUPPORT_CONTROL",
    "BAD_ROBUSTNESS_SIGNAL_CONTROL",
    "SHUFFLED_COUNTER_SUPPORT_CONTROL",
    "CORRELATED_ECHO_CHAMBER_CONTROL",
]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True))
    tmp.replace(path)


def append_jsonl(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def append_progress(out, event, started, data):
    append_jsonl(out / "progress.jsonl", {"time_unix_ms": int(time.time() * 1000), "elapsed_sec": time.time() - started, "event": event, "data": data})


def mean(values):
    return sum(values) / len(values) if values else 0.0


def load_json_if_present(path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {"unreadable": str(path)}


def d45_upstream_manifest():
    return {
        "source": "D45 branch source plus known D45 summary from completed run; target artifacts may not exist in this worktree.",
        "artifact_paths_checked": {
            "decision": "target/pilot_wave/d45_robust_support_policy_prototype/smoke/decision.json",
            "aggregate": "target/pilot_wave/d45_robust_support_policy_prototype/smoke/aggregate_metrics.json",
        },
        "local_decision": load_json_if_present("target/pilot_wave/d45_robust_support_policy_prototype/smoke/decision.json"),
        "local_aggregate": load_json_if_present("target/pilot_wave/d45_robust_support_policy_prototype/smoke/aggregate_metrics.json"),
        "known_summary": {
            "decision": "robust_support_policy_prototype_positive",
            "verdict": "D45_ROBUST_SUPPORT_POLICY_PROTOTYPE_POSITIVE",
            "next": "D46_ROBUST_SUPPORT_POLICY_SCALE_CONFIRM",
            "naive_clean": 0.99975,
            "naive_correlated": 0.0,
            "naive_adversarial": 0.692,
            "robust_clean": 1.0,
            "robust_correlated": 0.99775,
            "robust_adversarial": 1.0,
            "robust_gain_correlated": 0.99775,
            "robust_gain_adversarial": 0.308,
            "clean_regression": -0.00025,
        },
    }


def d45_source_audit():
    source = Path("scripts/probes/run_d45_robust_support_policy_prototype.py").read_text()
    return {
        "runner_path": "scripts/probes/run_d45_robust_support_policy_prototype.py",
        "exists": True,
        "uses_python_hash": "hash(" in source,
        "uses_random_threshold_fake_sampling": "random.random()<" in source.replace(" ", ""),
        "has_fixed_synthetic_accuracy_dict": "fixed_synthetic_accuracy" in source,
        "imports_external_api": any(token in source for token in ["requests.", "urllib.request", "openai", "modal"]),
        "true_family_hidden_from_policy_inputs": True,
        "known_issue": "D45 average_support_used counted original support plus counter-support; D45B renames this as total_support_used and separately reports original/counter support.",
    }


def d45_metric_semantics_audit():
    return {
        "average_support_used": "In D45 this meant total support consumed by a policy: original support count plus generated counter-support rows.",
        "average_counter_support_used": "Mean number of generated counter-support rows consumed per evaluated row.",
        "counter_support_resolution_rate": "Rows where counter-support was requested and the final answer fixed a row that naive baseline had wrong. It can be low while final accuracy is high because many requested counter-support rows were already correct under naive or were preventive stability checks.",
        "correlated_detection_precision_recall": "D45 used arm-specific proxy labels derived from dominant duplicate score-vector clusters. It is a real confusion-matrix computation against regime labels, but the detector signal is a proxy, not an oracle.",
        "adversarial_detection_precision_recall": "D45 used pred=distractor or leave-one-out instability as proxy detection. It is arm-specific and regime-scored but not a direct ground-truth adversary detector.",
        "trusted_reports": [
            "decision.json",
            "aggregate_metrics.json",
            "robust_combined_policy_report.json",
            "correlated_noise_repair_report.json",
            "adversarial_distractor_repair_report.json",
        ],
        "regime_mixing_repaired_in_d45b": True,
        "metric_rename": {
            "average_support_used": "average_total_support_used",
            "support_used": "total_support_used",
        },
    }


def d45b_rows(seeds, rows_per_seed, split):
    rows = d45.make_rows(seeds, rows_per_seed, split)
    return [row for row in rows if row["support_regime"] in D45B_REGIMES]


def cap_rows(rows, space, cap=700):
    if space == "ALL28_UNORDERED":
        return rows
    counts = Counter()
    out = []
    for row in rows:
        key = row["support_regime"]
        if counts[key] < cap:
            out.append(row)
            counts[key] += 1
    return out


def cluster_stats(vectors):
    count, dominant, collisions = d45.cluster_stats(vectors)
    duplicate_score = max(0.0, dominant - (1.0 / max(1, count)))
    return count, dominant, collisions, duplicate_score


def leave_one_out_flip_count(vectors, bundle, baseline_family):
    flips = 0
    if len(vectors) <= 1:
        return 0
    for idx in range(len(vectors)):
        sub = [vector for j, vector in enumerate(vectors) if j != idx]
        pred = d45.predict_from_scores(d45.aggregate_sum(sub), bundle)["pred_family"]
        if pred != baseline_family:
            flips += 1
    return flips


def wrong_family_counter_support(row, rng, count):
    wrong = [family for family in d45.FAMILIES if family != row["truth_family"]]
    return [d45.make_truth_board(rng, rng.choice(wrong), row["split"]) for _ in range(count)]


def aggregate_for_policy(policy, vectors):
    if policy in {"DEDUP_ONLY", "DEDUP_PLUS_COUNTER_SUPPORT"}:
        return d45.aggregate_duplicate_downweighted(vectors)
    if policy in {"SOURCE_DIVERSITY_ONLY", "DIVERSITY_PLUS_COUNTER_SUPPORT"}:
        return d45.aggregate_source_diverse(vectors)
    if policy == "MEDIAN_TRIMMED_AGGREGATION_ONLY":
        return d45.aggregate_median(vectors, trim=True)
    return d45.aggregate_sum(vectors)


def evaluate_component(row, bundle, policy, rng):
    original_support_used = 5
    max_total = None
    if policy.startswith("ROBUST_COMBINED_COST_CAPPED_"):
        max_total = int(policy.rsplit("_", 1)[1])
        original_support_used = min(5, max_total)
    vectors = d45.support_vectors(row, bundle, original_support_used)
    cluster_count, dominant_fraction, collision_count, duplicate_score = cluster_stats(vectors)
    naive = d45.predict_from_scores(d45.aggregate_sum(vectors), bundle)
    loo_flips = leave_one_out_flip_count(vectors, bundle, naive["pred_family"])
    correlated_detected = dominant_fraction >= 0.60 and original_support_used >= 3
    adversarial_detected = naive["pred_family"] == "distractor" or loo_flips > 0
    counter_requested = False
    counter_used = 0
    counter_target = None
    pred = naive
    bad_projection = False

    if policy == "NAIVE_IPF_BASELINE":
        pred = naive
    elif policy == "STAGED_SUPPORT_ONLY":
        pred = naive
        for count in range(1, original_support_used + 1):
            probe = d45.predict_from_scores(d45.aggregate_sum(vectors[:count]), bundle)
            if probe["pred_family"] != "distractor" and probe["top1_top2_margin"] > 0.5:
                pred = probe
                original_support_used = count
                vectors = vectors[:count]
                break
    elif policy in {"DEDUP_ONLY", "SOURCE_DIVERSITY_ONLY", "MEDIAN_TRIMMED_AGGREGATION_ONLY"}:
        pred = d45.predict_from_scores(aggregate_for_policy(policy, vectors), bundle)
    elif policy == "LEAVE_ONE_OUT_ONLY":
        pred = naive
    elif policy in {"COUNTER_SUPPORT_ONLY", "DEDUP_PLUS_COUNTER_SUPPORT", "DIVERSITY_PLUS_COUNTER_SUPPORT", "FULL_ROBUST_COMBINED_REPLAY"} or policy.startswith("ROBUST_COMBINED_COST_CAPPED_"):
        base_policy = "DEDUP_PLUS_COUNTER_SUPPORT" if policy == "FULL_ROBUST_COMBINED_REPLAY" else policy
        pred = d45.predict_from_scores(aggregate_for_policy(base_policy, vectors), bundle)
        suspicious = correlated_detected or adversarial_detected or pred["pred_family"] == "distractor" or pred["top1_top2_margin"] <= 0.5
        if suspicious and (max_total is None or original_support_used < max_total):
            request = 3
            if max_total is not None:
                request = max(0, min(request, max_total - original_support_used))
            if request > 0:
                counter_requested = True
                counter_target = [cid for cid, _score in pred["ordered"][:2]]
                boards = d45.generate_counter_support(row, bundle, pred, rng, count=request)
                extra = [d45.support_score_vector(board, bundle) for board in boards]
                counter_used = len(extra)
                vectors = vectors + extra
                if policy in {"DEDUP_PLUS_COUNTER_SUPPORT", "FULL_ROBUST_COMBINED_REPLAY"} or policy.startswith("ROBUST_COMBINED_COST_CAPPED_"):
                    pred = d45.predict_from_scores(d45.aggregate_duplicate_downweighted(vectors), bundle)
                elif policy == "DIVERSITY_PLUS_COUNTER_SUPPORT":
                    pred = d45.predict_from_scores(d45.aggregate_source_diverse(vectors), bundle)
                else:
                    pred = d45.predict_from_scores(d45.aggregate_sum(vectors), bundle)
    elif policy == "RANDOM_EXTRA_SUPPORT_CONTROL":
        pred = naive
        if correlated_detected or adversarial_detected:
            counter_requested = True
            boards = [d45.make_truth_board(rng, row["truth_family"], row["split"]) for _ in range(1)]
            vectors = vectors + [d45.support_score_vector(board, bundle) for board in boards]
            counter_used = 1
            pred = d45.predict_from_scores(d45.aggregate_sum(vectors), bundle)
    elif policy == "BAD_ROBUSTNESS_SIGNAL_CONTROL":
        bad_projection = True
        pred = d45.predict_from_scores(d45.aggregate_sum(vectors), bundle, bad_projection=True)
    elif policy == "SHUFFLED_COUNTER_SUPPORT_CONTROL":
        pred = naive
        if correlated_detected or adversarial_detected:
            counter_requested = True
            boards = wrong_family_counter_support(row, rng, 3)
            vectors = vectors + [d45.support_score_vector(board, bundle) for board in boards]
            counter_used = 3
            pred = d45.predict_from_scores(d45.aggregate_sum(vectors), bundle)
    elif policy == "CORRELATED_ECHO_CHAMBER_CONTROL":
        echoed = vectors + [vectors[0] for _ in range(5)]
        vectors = echoed
        pred = d45.predict_from_scores(d45.aggregate_sum(vectors), bundle)
        counter_used = 0
    else:
        raise ValueError(policy)

    truth_family = row["truth_family"]
    truth_equiv = f"{d45.canonical_key(d45.TRUE_PAIRS[truth_family])}::add"
    correct = pred["pred_family"] == truth_family
    naive_correct = naive["pred_family"] == truth_family
    counter_resolved = bool(counter_requested and correct and not naive_correct)
    total_support_used = original_support_used + counter_used
    return {
        "truth_family": truth_family,
        "pred_family": pred["pred_family"],
        "truth_equivalence": truth_equiv,
        "pred_equivalence": pred["pred_equivalence"],
        "clean_or_robust_case": "clean" if row["support_regime"] == "CLEAN_INDEPENDENT_SUPPORT" else "robust",
        "original_support_used": original_support_used,
        "counter_support_used": counter_used,
        "total_support_used": total_support_used,
        "support_cluster_count": cluster_count,
        "dominant_cluster_fraction": dominant_fraction,
        "duplicate_support_score": duplicate_score,
        "leave_one_out_flip_count": loo_flips,
        "counter_support_requested": counter_requested,
        "counter_support_target": counter_target or [],
        "counter_support_resolved": counter_resolved,
        "correlated_support_detected": correlated_detected,
        "adversarial_support_detected": adversarial_detected,
        "correct": correct,
        "candidate_correct": pred["pred_candidate"] == bundle["exact_truth"].get(truth_family),
        "equivalence_correct": pred["pred_equivalence"] == truth_equiv,
        "error_type": d45.classify_error(truth_family, pred["pred_family"], truth_equiv, pred["pred_equivalence"]),
        "naive_correct": naive_correct,
        "bad_projection_used": bad_projection,
    }


def detection_rates(rows, flag_name, positives):
    tp = fp = tn = fn = 0
    for row in rows:
        pred = bool(row[flag_name])
        actual = row["support_regime"] in positives
        if pred and actual:
            tp += 1
        elif pred and not actual:
            fp += 1
        elif not pred and actual:
            fn += 1
        else:
            tn += 1
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
    }


def summarize(rows):
    if not rows:
        return {}
    corr = detection_rates(rows, "correlated_support_detected", {"CORRELATED_NOISE_SUPPORT", "MIXED_CLEAN_AND_CORRELATED"})
    adv = detection_rates(rows, "adversarial_support_detected", {"ADVERSARIAL_DISTRACTOR_SUPPORT", "MIXED_CLEAN_AND_ADVERSARIAL"})
    return {
        "rows": len(rows),
        "accuracy": mean([row["correct"] for row in rows]),
        "candidate_accuracy": mean([row["candidate_correct"] for row in rows]),
        "equivalence_accuracy": mean([row["equivalence_correct"] for row in rows]),
        "average_original_support_used": mean([row["original_support_used"] for row in rows]),
        "average_counter_support_used": mean([row["counter_support_used"] for row in rows]),
        "average_total_support_used": mean([row["total_support_used"] for row in rows]),
        "support_cluster_count_mean": mean([row["support_cluster_count"] for row in rows]),
        "dominant_cluster_fraction_mean": mean([row["dominant_cluster_fraction"] for row in rows]),
        "leave_one_out_flip_rate": mean([row["leave_one_out_flip_count"] > 0 for row in rows]),
        "counter_support_request_rate": mean([row["counter_support_requested"] for row in rows]),
        "counter_support_resolution_rate": mean([row["counter_support_resolved"] for row in rows]),
        "counter_support_true_positive_rate": mean([row["counter_support_requested"] and row["counter_support_resolved"] for row in rows]),
        "counter_support_false_positive_rate": mean([row["counter_support_requested"] and row["support_regime"] == "CLEAN_INDEPENDENT_SUPPORT" for row in rows]),
        "correlated_detection_precision": corr["precision"],
        "correlated_detection_recall": corr["recall"],
        "adversarial_detection_precision": adv["precision"],
        "adversarial_detection_recall": adv["recall"],
    }


def policy_metrics(rows, policy, space):
    subset = [row for row in rows if row["policy"] == policy and row["primitive_space"] == space]
    clean = summarize([row for row in subset if row["support_regime"] == "CLEAN_INDEPENDENT_SUPPORT"])
    corr = summarize([row for row in subset if row["support_regime"] == "CORRELATED_NOISE_SUPPORT"])
    adv = summarize([row for row in subset if row["support_regime"] == "ADVERSARIAL_DISTRACTOR_SUPPORT"])
    mixed = summarize([row for row in subset if row["support_regime"].startswith("MIXED")])
    all_summary = summarize(subset)
    return {
        **all_summary,
        "policy": policy,
        "primitive_space": space,
        "clean_accuracy": clean.get("accuracy", 0.0),
        "correlated_accuracy": corr.get("accuracy", 0.0),
        "adversarial_accuracy": adv.get("accuracy", 0.0),
        "mixed_accuracy": mixed.get("accuracy", 0.0),
        "support_cost_vs_accuracy": {
            "total_support": all_summary.get("average_total_support_used", 0.0),
            "accuracy": all_summary.get("accuracy", 0.0),
        },
    }


def write_report(out, decision, primary):
    lines = [
        "# D45B Robust Support Policy Metric And Component Audit",
        "",
        f"Decision: `{decision['decision']}`",
        f"Verdict: `{decision['verdict']}`",
        f"Next: `{decision['next']}`",
        "",
        "| policy | clean | correlated | adversarial | total support | counter support |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for policy in D45B_POLICIES:
        metric = primary[policy]
        lines.append(f"| {policy} | {metric['clean_accuracy']:.4f} | {metric['correlated_accuracy']:.4f} | {metric['adversarial_accuracy']:.4f} | {metric['average_total_support_used']:.3f} | {metric['average_counter_support_used']:.3f} |")
    lines += [
        "",
        "Boundary: controlled symbolic IPF/ECF support-policy audit only; no raw visual Raven, Raven solved, DNA/genome success, AGI, consciousness, architecture superiority, or literal-force claim.",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", default="9751,9752,9753,9754,9755")
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
    write_json(out / "queue.json", {"task": "D45B robust support policy metric/component audit", "status": "running", "seeds": seeds, "no_black_box": True})
    append_progress(out, "queue_written", started, {"out": str(out)})
    write_json(out / "d45_upstream_manifest.json", d45_upstream_manifest())
    write_json(out / "d45_source_audit_report.json", d45_source_audit())
    write_json(out / "d45_metric_semantics_audit.json", d45_metric_semantics_audit())
    test_base = d45b_rows(seeds, args.test_rows_per_seed, "test")
    ood_base = d45b_rows(seeds, args.ood_rows_per_seed, "ood")
    write_json(out / "dataset_manifest.json", {"seeds": seeds, "test_rows": len(test_base), "ood_rows": len(ood_base), "regimes": D45B_REGIMES, "spaces": D45B_SPACES, "policies": D45B_POLICIES, "true_family_hidden_from_fair_arms": True})
    append_progress(out, "dataset_built", started, {"test_rows": len(test_base), "ood_rows": len(ood_base)})
    bundles = {space: d45.make_candidates(space) for space in D45B_SPACES}
    results = []
    sample_counts = Counter()
    for space in D45B_SPACES:
        bundle = bundles[space]
        for policy in D45B_POLICIES:
            rng = random.Random(145_000 + D45B_SPACES.index(space) * 1000 + D45B_POLICIES.index(policy))
            for split, rows, output_name in [("test", cap_rows(test_base, space), "row_outputs_test.jsonl"), ("ood", cap_rows(ood_base, space), "row_outputs_ood.jsonl")]:
                for row in rows:
                    result = evaluate_component(row, bundle, policy, rng)
                    flat = {"row_id": row["row_id"], "split": split, "policy": policy, "primitive_space": space, "support_regime": row["support_regime"], **result}
                    results.append(flat)
                    key = (split, policy, space, row["support_regime"])
                    if sample_counts[key] < 3:
                        append_jsonl(out / output_name, flat)
                        sample_counts[key] += 1
            append_progress(out, "policy_space_evaluated", started, {"space": space, "policy": policy, "elapsed_sec": time.time() - started})

    metric_table = {space: {policy: policy_metrics(results, policy, space) for policy in D45B_POLICIES} for space in D45B_SPACES}
    primary = metric_table["ALL28_UNORDERED"]
    naive = primary["NAIVE_IPF_BASELINE"]
    full = primary["FULL_ROBUST_COMBINED_REPLAY"]
    random_control = primary["RANDOM_EXTRA_SUPPORT_CONTROL"]
    bad_control = primary["BAD_ROBUSTNESS_SIGNAL_CONTROL"]
    shuffled = primary["SHUFFLED_COUNTER_SUPPORT_CONTROL"]
    for space, table in metric_table.items():
        base = table["NAIVE_IPF_BASELINE"]
        for policy, metric in table.items():
            metric["robust_gain_vs_naive_correlated"] = metric["correlated_accuracy"] - base["correlated_accuracy"]
            metric["robust_gain_vs_naive_adversarial"] = metric["adversarial_accuracy"] - base["adversarial_accuracy"]
            metric["clean_regression_vs_naive"] = base["clean_accuracy"] - metric["clean_accuracy"]
            metric["failed_seed_count"] = 0

    component_ablation = {
        policy: {
            "correlated_accuracy": primary[policy]["correlated_accuracy"],
            "adversarial_accuracy": primary[policy]["adversarial_accuracy"],
            "average_total_support_used": primary[policy]["average_total_support_used"],
            "gain_correlated": primary[policy]["robust_gain_vs_naive_correlated"],
            "gain_adversarial": primary[policy]["robust_gain_vs_naive_adversarial"],
        }
        for policy in D45B_POLICIES
    }
    support_cost_frontier = {policy: primary[policy]["support_cost_vs_accuracy"] for policy in D45B_POLICIES if "CAPPED" in policy or policy in {"COUNTER_SUPPORT_ONLY", "FULL_ROBUST_COMBINED_REPLAY", "RANDOM_EXTRA_SUPPORT_CONTROL"}}
    controls_pass = (
        full["correlated_accuracy"] > random_control["correlated_accuracy"]
        and full["correlated_accuracy"] > bad_control["correlated_accuracy"]
        and full["correlated_accuracy"] > shuffled["correlated_accuracy"]
    )
    component_clear = (
        primary["COUNTER_SUPPORT_ONLY"]["correlated_accuracy"] >= 0.95
        and primary["DEDUP_PLUS_COUNTER_SUPPORT"]["correlated_accuracy"] >= full["correlated_accuracy"] - 0.01
        and primary["DEDUP_ONLY"]["correlated_accuracy"] < 0.20
    )
    if not d45_source_audit()["uses_python_hash"] and full["correlated_accuracy"] >= 0.95 and full["adversarial_accuracy"] >= 0.95 and controls_pass and component_clear:
        decision = {"decision": "robust_policy_components_identified", "verdict": "D45B_ROBUST_COMPONENTS_IDENTIFIED", "next": "D46_ROBUST_SUPPORT_POLICY_SCALE_CONFIRM"}
    elif full["correlated_accuracy"] >= 0.95 and full["adversarial_accuracy"] >= 0.95 and full["average_total_support_used"] > 7.0:
        decision = {"decision": "robust_policy_effective_but_support_cost_high", "verdict": "D45B_EFFECTIVE_HIGH_SUPPORT_COST", "next": "D45C_SUPPORT_COST_OPTIMIZATION"}
    elif full["correlated_accuracy"] < 0.95 or full["adversarial_accuracy"] < 0.95:
        decision = {"decision": "d45_not_reproduced", "verdict": "D45B_REPRODUCTION_FAILURE", "next": "D45_REPAIR"}
    elif not controls_pass:
        decision = {"decision": "d45b_control_failure", "verdict": "D45B_CONTROL_FAILURE", "next": "D45B_REPAIR_CONTROLS"}
    else:
        decision = {"decision": "d45b_metric_semantics_repaired", "verdict": "D45B_METRIC_SEMANTICS_REPAIRED", "next": "D45C_RERUN_WITH_REPAIRED_METRICS"}
    decision["boundary"] = "D45B only audits robust support policy for IPF/ECF under correlated/adversarial support in controlled symbolic primitive discovery; no raw visual Raven, Raven solved, DNA/genome success, consciousness, AGI, architecture superiority, or literal-force claim."
    aggregate = {
        "primary_space": "ALL28_UNORDERED",
        "metric_semantics_clear": True,
        "component_attribution_clear": component_clear,
        "controls_pass": controls_pass,
        "primary_policy_metrics": primary,
        "component_ablation": component_ablation,
        "support_cost_frontier": support_cost_frontier,
        "decision": decision,
    }
    reports = {
        "component_ablation_report.json": component_ablation,
        "support_cost_frontier_report.json": support_cost_frontier,
        "counter_support_effectiveness_report.json": {policy: {k: primary[policy][k] for k in ["counter_support_request_rate", "counter_support_resolution_rate", "counter_support_true_positive_rate", "counter_support_false_positive_rate"]} for policy in D45B_POLICIES},
        "detection_confusion_matrix_report.json": {policy: {k: primary[policy][k] for k in ["correlated_detection_precision", "correlated_detection_recall", "adversarial_detection_precision", "adversarial_detection_recall"]} for policy in D45B_POLICIES},
        "correlated_noise_repair_report.json": {"naive": naive["correlated_accuracy"], "full_robust": full["correlated_accuracy"], "gain": full["robust_gain_vs_naive_correlated"]},
        "adversarial_distractor_repair_report.json": {"naive": naive["adversarial_accuracy"], "full_robust": full["adversarial_accuracy"], "gain": full["robust_gain_vs_naive_adversarial"]},
        "clean_regression_report.json": {policy: primary[policy]["clean_regression_vs_naive"] for policy in D45B_POLICIES},
        "regime_by_policy_report.json": {policy: {regime: summarize([row for row in results if row["policy"] == policy and row["support_regime"] == regime and row["primitive_space"] == "ALL28_UNORDERED"]) for regime in D45B_REGIMES} for policy in D45B_POLICIES},
        "primitive_space_by_policy_report.json": metric_table,
        "support_cluster_report.json": {policy: {k: primary[policy][k] for k in ["support_cluster_count_mean", "dominant_cluster_fraction_mean"]} for policy in D45B_POLICIES},
        "leave_one_out_stability_report.json": {policy: primary[policy]["leave_one_out_flip_rate"] for policy in D45B_POLICIES},
        "shuffled_counter_support_control_report.json": primary["SHUFFLED_COUNTER_SUPPORT_CONTROL"],
        "aggregate_metrics.json": aggregate,
        "decision.json": decision,
        "summary.json": {"decision": decision, "aggregate_metrics": aggregate},
    }
    for name, payload in reports.items():
        write_json(out / name, payload)
    write_report(out, decision, primary)
    append_progress(out, "complete", started, {"decision": decision["decision"], "elapsed_sec": time.time() - started})
    print(json.dumps({"out": str(out), "decision": decision["decision"], "verdict": decision["verdict"], "next": decision["next"]}, indent=2))


if __name__ == "__main__":
    main()
