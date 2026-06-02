#!/usr/bin/env python3
"""D46 scale confirm for robust IPF/ECF support policy."""

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import run_d45b_robust_support_policy_metric_and_component_audit as d45b
import run_d45_robust_support_policy_prototype as d45

D46_SPACES = ["ALL28_UNORDERED", "CURRENT5_PLUS_DISTRACTORS_20", "CURRENT5_PLUS_DISTRACTORS_50"]
D46_REGIMES = [
    "CLEAN_INDEPENDENT_SUPPORT",
    "CORRELATED_NOISE_SUPPORT",
    "ADVERSARIAL_DISTRACTOR_SUPPORT",
    "MIXED_CLEAN_AND_CORRELATED",
    "MIXED_CLEAN_AND_ADVERSARIAL",
]
D46_POLICIES = [
    "NAIVE_IPF_BASELINE",
    "COUNTER_SUPPORT_ONLY",
    "FULL_ROBUST_COMBINED_REPLAY",
    "ROBUST_COMBINED_COST_CAPPED_5",
    "ROBUST_COMBINED_COST_CAPPED_7",
    "ROBUST_COMBINED_COST_CAPPED_9",
    "RANDOM_EXTRA_SUPPORT_CONTROL",
    "BAD_ROBUSTNESS_SIGNAL_CONTROL",
    "SHUFFLED_COUNTER_SUPPORT_CONTROL",
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


def make_rows(seeds, rows_per_seed, split):
    return [row for row in d45.make_rows(seeds, rows_per_seed, split) if row["support_regime"] in D46_REGIMES]


def cap_rows(rows, space, cap=900):
    if space == "ALL28_UNORDERED":
        return rows
    counts = Counter()
    out = []
    for row in rows:
        key = row["support_regime"]
        if counts[key] < cap:
            counts[key] += 1
            out.append(row)
    return out


def summarize(rows):
    if not rows:
        return {}
    return {
        "rows": len(rows),
        "accuracy": mean([row["correct"] for row in rows]),
        "average_total_support_used": mean([row["total_support_used"] for row in rows]),
        "average_counter_support_used": mean([row["counter_support_used"] for row in rows]),
        "counter_support_request_rate": mean([row["counter_support_requested"] for row in rows]),
        "counter_support_resolution_rate": mean([row["counter_support_resolved"] for row in rows]),
    }


def policy_space_metrics(rows, policy, space):
    subset = [row for row in rows if row["policy"] == policy and row["primitive_space"] == space]
    clean = summarize([row for row in subset if row["support_regime"] == "CLEAN_INDEPENDENT_SUPPORT"])
    corr = summarize([row for row in subset if row["support_regime"] == "CORRELATED_NOISE_SUPPORT"])
    adv = summarize([row for row in subset if row["support_regime"] == "ADVERSARIAL_DISTRACTOR_SUPPORT"])
    mixed = summarize([row for row in subset if row["support_regime"].startswith("MIXED")])
    by_seed = {}
    for seed in sorted({row["seed"] for row in subset}):
        seed_rows = [row for row in subset if row["seed"] == seed]
        by_seed[str(seed)] = {
            "clean": summarize([row for row in seed_rows if row["support_regime"] == "CLEAN_INDEPENDENT_SUPPORT"]).get("accuracy", 0.0),
            "correlated": summarize([row for row in seed_rows if row["support_regime"] == "CORRELATED_NOISE_SUPPORT"]).get("accuracy", 0.0),
            "adversarial": summarize([row for row in seed_rows if row["support_regime"] == "ADVERSARIAL_DISTRACTOR_SUPPORT"]).get("accuracy", 0.0),
            "mixed": summarize([row for row in seed_rows if row["support_regime"].startswith("MIXED")]).get("accuracy", 0.0),
        }
    return {
        "policy": policy,
        "primitive_space": space,
        "clean_accuracy": clean.get("accuracy", 0.0),
        "correlated_accuracy": corr.get("accuracy", 0.0),
        "adversarial_accuracy": adv.get("accuracy", 0.0),
        "mixed_accuracy": mixed.get("accuracy", 0.0),
        "average_total_support_used": summarize(subset).get("average_total_support_used", 0.0),
        "average_counter_support_used": summarize(subset).get("average_counter_support_used", 0.0),
        "counter_support_request_rate": summarize(subset).get("counter_support_request_rate", 0.0),
        "counter_support_resolution_rate": summarize(subset).get("counter_support_resolution_rate", 0.0),
        "by_seed": by_seed,
        "min_seed_correlated": min((m["correlated"] for m in by_seed.values()), default=0.0),
        "min_seed_adversarial": min((m["adversarial"] for m in by_seed.values()), default=0.0),
    }


def write_report(out, decision, primary):
    lines = [
        "# D46 Robust Support Policy Scale Confirm",
        "",
        f"Decision: `{decision['decision']}`",
        f"Verdict: `{decision['verdict']}`",
        f"Next: `{decision['next']}`",
        "",
        "| policy | clean | correlated | adversarial | mixed | min seed corr | min seed adv | total support |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for policy in D46_POLICIES:
        m = primary[policy]
        lines.append(f"| {policy} | {m['clean_accuracy']:.4f} | {m['correlated_accuracy']:.4f} | {m['adversarial_accuracy']:.4f} | {m['mixed_accuracy']:.4f} | {m['min_seed_correlated']:.4f} | {m['min_seed_adversarial']:.4f} | {m['average_total_support_used']:.3f} |")
    lines.append("")
    lines.append("Boundary: controlled symbolic IPF/ECF robust support scale confirm only; no raw visual Raven, Raven solved, DNA/genome success, AGI, consciousness, architecture superiority, or literal-force claim.")
    (out / "report.md").write_text("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", default="9801,9802,9803,9804,9805,9806,9807,9808")
    parser.add_argument("--train-rows-per-seed", type=int, default=1200)
    parser.add_argument("--test-rows-per-seed", type=int, default=1200)
    parser.add_argument("--ood-rows-per-seed", type=int, default=1200)
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
    write_json(out / "queue.json", {"task": "D46 robust support scale confirm", "status": "running", "seeds": seeds, "rows": {"train": args.train_rows_per_seed, "test": args.test_rows_per_seed, "ood": args.ood_rows_per_seed}, "no_black_box": True})
    append_progress(out, "queue_written", started, {"out": str(out)})
    write_json(out / "d45b_upstream_manifest.json", {"expected_decision": "robust_policy_components_identified", "expected_next": "D46_ROBUST_SUPPORT_POLICY_SCALE_CONFIRM", "source": "D45B branch/artifacts", "local_decision": None})
    test_base = make_rows(seeds, args.test_rows_per_seed, "test")
    ood_base = make_rows(seeds, args.ood_rows_per_seed, "ood")
    write_json(out / "dataset_manifest.json", {"seeds": seeds, "test_rows": len(test_base), "ood_rows": len(ood_base), "spaces": D46_SPACES, "policies": D46_POLICIES, "regimes": D46_REGIMES, "scale_lite": len(seeds) <= 5 and args.test_rows_per_seed <= 800})
    append_progress(out, "dataset_built", started, {"test_rows": len(test_base), "ood_rows": len(ood_base)})
    bundles = {space: d45.make_candidates(space) for space in D46_SPACES}
    rows_out = []
    sample_counts = Counter()
    for space in D46_SPACES:
        bundle = bundles[space]
        for policy in D46_POLICIES:
            rng = random.Random(246_000 + D46_SPACES.index(space) * 100 + D46_POLICIES.index(policy))
            for split, base_rows, output_name in [("test", cap_rows(test_base, space), "row_outputs_test.jsonl"), ("ood", cap_rows(ood_base, space), "row_outputs_ood.jsonl")]:
                for row in base_rows:
                    result = d45b.evaluate_component(row, bundle, policy, rng)
                    flat = {"row_id": row["row_id"], "seed": row["seed"], "split": split, "policy": policy, "primitive_space": space, "support_regime": row["support_regime"], **result}
                    rows_out.append(flat)
                    key = (split, policy, space, row["support_regime"])
                    if sample_counts[key] < 3:
                        append_jsonl(out / output_name, flat)
                        sample_counts[key] += 1
            append_progress(out, "policy_space_evaluated", started, {"space": space, "policy": policy, "elapsed_sec": time.time() - started})
    table = {space: {policy: policy_space_metrics(rows_out, policy, space) for policy in D46_POLICIES} for space in D46_SPACES}
    primary = table["ALL28_UNORDERED"]
    naive = primary["NAIVE_IPF_BASELINE"]
    robust = primary["FULL_ROBUST_COMBINED_REPLAY"]
    random_control = primary["RANDOM_EXTRA_SUPPORT_CONTROL"]
    bad_control = primary["BAD_ROBUSTNESS_SIGNAL_CONTROL"]
    shuffled = primary["SHUFFLED_COUNTER_SUPPORT_CONTROL"]
    clean_regression = naive["clean_accuracy"] - robust["clean_accuracy"]
    controls_worse = robust["correlated_accuracy"] > random_control["correlated_accuracy"] and robust["correlated_accuracy"] > bad_control["correlated_accuracy"] and robust["correlated_accuracy"] > shuffled["correlated_accuracy"]
    pass_gate = (
        robust["clean_accuracy"] >= 0.995
        and robust["correlated_accuracy"] >= 0.95
        and robust["adversarial_accuracy"] >= 0.95
        and robust["mixed_accuracy"] >= 0.95
        and robust["min_seed_correlated"] >= 0.90
        and robust["min_seed_adversarial"] >= 0.90
        and clean_regression <= 0.005
        and controls_worse
    )
    high_cost = robust["average_total_support_used"] > 7.0
    if pass_gate and high_cost:
        decision = {"decision": "robust_support_scale_confirmed_high_cost", "verdict": "D46_ROBUST_SUPPORT_HIGH_COST", "next": "D46C_SUPPORT_COST_OPTIMIZATION"}
    elif pass_gate:
        decision = {"decision": "robust_support_policy_scale_confirmed", "verdict": "D46_ROBUST_SUPPORT_POLICY_SCALE_CONFIRMED", "next": "D47_CELL_REFERENCE_DISCOVERY_WITH_ROBUST_SUPPORT"}
    else:
        decision = {"decision": "robust_support_policy_scale_not_confirmed", "verdict": "D46_SCALE_NOT_CONFIRMED", "next": "D46_REPAIR"}
    decision["failed_jobs"] = []
    decision["boundary"] = "D46 only scale-confirms robust support policy for IPF/ECF under correlated/adversarial support in controlled symbolic primitive discovery; no raw visual Raven, Raven solved, DNA/genome success, consciousness, AGI, architecture superiority, or literal-force claim."
    aggregate = {
        "primary_space": "ALL28_UNORDERED",
        "primary_policy_metrics": primary,
        "policy_space_metrics": table,
        "clean_regression_vs_naive": clean_regression,
        "controls_worse": controls_worse,
        "failed_jobs": [],
        "support_cost_reported": True,
        "decision": decision,
    }
    reports = {
        "policy_comparison_report.json": table,
        "support_cost_report.json": {policy: {"average_total_support_used": primary[policy]["average_total_support_used"], "accuracy": primary[policy]["mixed_accuracy"]} for policy in D46_POLICIES},
        "control_report.json": {"random": random_control, "bad": bad_control, "shuffled": shuffled, "controls_worse": controls_worse},
        "seed_variance_report.json": robust["by_seed"],
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
