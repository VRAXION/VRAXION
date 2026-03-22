"""Canonical convergence/budget/patience runner for permutation learning."""

from __future__ import annotations

import argparse
import statistics
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tests.harness import (
    PermutationHarnessConfig,
    build_net_and_targets,
    build_policy,
    run_budgeted_search,
)


DEFAULT_SEEDS = [42, 77, 123, 0, 1]


def parse_int_list(value: str):
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", choices=["budget", "patience", "convergence"], default="budget")
    ap.add_argument("--seeds", default=",".join(str(x) for x in DEFAULT_SEEDS))
    ap.add_argument("--vocab", type=int, default=64)
    ap.add_argument("--neurons", type=int, default=192)
    ap.add_argument("--density", type=float, default=0.06)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--ticks", type=int, default=8)
    ap.add_argument("--budgets", default="16000,32000,64000")
    ap.add_argument("--windows", default="10,25,50,100,200,500")
    ap.add_argument("--policy", default="drive")
    ap.add_argument("--policies", default="flip_on_reject,darwinian")
    ap.add_argument("--vocab-values", default="16,32,64")
    return ap.parse_args()


def run_single(policy_name, seed, config, checkpoints=None):
    net, targets = build_net_and_targets(config, seed)
    outcome = run_budgeted_search(net, targets, config, build_policy(policy_name), checkpoints=checkpoints)
    return outcome


def run_budget_study(args, seeds):
    budgets = parse_int_list(args.budgets)
    policies = [x.strip() for x in args.policies.split(",") if x.strip()]
    print("BUDGET STUDY", flush=True)
    print(f"policies={policies} budgets={budgets} seeds={seeds}", flush=True)
    print("=" * 90, flush=True)
    results = []
    for budget in budgets:
        config = PermutationHarnessConfig(
            vocab=args.vocab,
            neurons=args.neurons,
            density=args.density,
            threshold=args.threshold,
            ticks=args.ticks,
            budget=budget,
            stale_limit=0,
        )
        for policy_name in policies:
            for seed in seeds:
                outcome = run_single(policy_name, seed, config)
                results.append((budget, policy_name, seed, outcome.best_score))
                print(
                    f"  {policy_name:16s} budget={budget:6d} seed={seed:3d} "
                    f"score={outcome.best_score*100:5.1f}% acc={outcome.best_acc*100:5.1f}%",
                    flush=True,
                )

    print(f"\n{'='*90}", flush=True)
    print("BUDGET SUMMARY", flush=True)
    for budget in budgets:
        for policy_name in policies:
            scores = [score for b, p, _, score in results if b == budget and p == policy_name]
            print(
                f"  {policy_name:16s} budget={budget:6d} "
                f"score={statistics.mean(scores)*100:5.1f}% +/-{statistics.pstdev(scores)*100:4.1f}pp",
                flush=True,
            )
    print(f"{'='*90}", flush=True)


def run_patience_study(args, seeds):
    windows = parse_int_list(args.windows)
    print("PATIENCE / WINDOW STUDY", flush=True)
    print(f"windows={windows} seeds={seeds}", flush=True)
    print("=" * 90, flush=True)
    baseline_cfg = PermutationHarnessConfig(
        vocab=args.vocab,
        neurons=args.neurons,
        density=args.density,
        threshold=args.threshold,
        ticks=args.ticks,
        budget=max(parse_int_list(args.budgets)),
        stale_limit=0,
    )
    results = []

    for seed in seeds:
        outcome = run_single("flip_on_reject", seed, baseline_cfg)
        results.append(("flip_on_reject", seed, outcome.best_score))
        print(f"  flip_on_reject seed={seed:3d} score={outcome.best_score*100:5.1f}%", flush=True)

    for window in windows:
        policy_name = f"window_{window}"
        for seed in seeds:
            outcome = run_single(policy_name, seed, baseline_cfg)
            results.append((policy_name, seed, outcome.best_score))
            print(f"  {policy_name:16s} seed={seed:3d} score={outcome.best_score*100:5.1f}%", flush=True)

    print(f"\n{'='*90}", flush=True)
    baseline_scores = [score for policy, _, score in results if policy == "flip_on_reject"]
    baseline_mean = statistics.mean(baseline_scores) * 100
    print(f"  {'policy':16s} {'mean':>7s} {'std':>7s} {'vs_flip':>8s}", flush=True)
    print(
        f"  {'flip_on_reject':16s} {baseline_mean:6.1f}% "
        f"{statistics.pstdev(baseline_scores)*100:6.1f}pp     ---",
        flush=True,
    )
    for window in windows:
        policy_name = f"window_{window}"
        scores = [score for policy, _, score in results if policy == policy_name]
        mean = statistics.mean(scores) * 100
        std = statistics.pstdev(scores) * 100
        print(
            f"  {policy_name:16s} {mean:6.1f}% {std:6.1f}pp {mean - baseline_mean:+7.1f}pp",
            flush=True,
        )
    print(f"{'='*90}", flush=True)


def run_convergence_study(args, seeds):
    vocab_values = parse_int_list(args.vocab_values)
    print("CONVERGENCE STUDY", flush=True)
    print(f"policy={args.policy} vocab_values={vocab_values} seeds={seeds}", flush=True)
    print("=" * 90, flush=True)

    schedule = {
        16: (50000, [500, 1000, 2000, 4000, 8000, 16000, 30000, 50000]),
        32: (100000, [1000, 2000, 4000, 8000, 16000, 32000, 64000, 100000]),
        64: (200000, [2000, 4000, 8000, 16000, 32000, 64000, 128000, 200000]),
    }

    for vocab in vocab_values:
        budget, checkpoints = schedule.get(vocab, (50000, [1000, 2000, 4000, 8000, 16000, 32000, 50000]))
        config = PermutationHarnessConfig(
            vocab=vocab,
            neurons=vocab * 3,
            density=args.density,
            threshold=args.threshold,
            ticks=args.ticks,
            budget=budget,
            stale_limit=0,
        )
        curves = []
        final_scores = []
        print(f"\nV={vocab} budget={budget}", flush=True)
        for seed in seeds:
            outcome = run_single(args.policy, seed, config, checkpoints=checkpoints)
            curves.append(outcome.curve)
            final_scores.append(outcome.best_acc)
            print(
                f"  seed={seed:3d} final_acc={outcome.best_acc*100:5.1f}% "
                f"score={outcome.best_score*100:5.1f}% steps={outcome.steps:6d}",
                flush=True,
            )

        print("  Average checkpoints:", flush=True)
        for idx, checkpoint in enumerate(checkpoints):
            accs = []
            for curve in curves:
                if idx < len(curve):
                    accs.append(curve[idx]["best_acc"] * 100)
            if accs:
                print(
                    f"    @{checkpoint:7d}: {statistics.mean(accs):5.1f}% +/-{statistics.pstdev(accs):4.1f}pp",
                    flush=True,
                )
        converged = sum(1 for score in final_scores if score >= 1.0)
        print(f"  Converged to 100%: {converged}/{len(final_scores)}", flush=True)

    print(f"\n{'='*90}", flush=True)


def main():
    args = parse_args()
    seeds = parse_int_list(args.seeds)
    if args.study == "budget":
        run_budget_study(args, seeds)
    elif args.study == "patience":
        run_patience_study(args, seeds)
    else:
        run_convergence_study(args, seeds)


if __name__ == "__main__":
    main()

