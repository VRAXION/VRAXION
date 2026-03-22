"""Canonical mutation-policy comparison runner for permutation learning.

Supersedes the former one-off policy scripts in archive/legacy_harness/.
"""

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


DEFAULT_POLICIES = [
    "drive",
    "mode",
    "bool_mood",
    "add_remove",
    "flip_on_reject",
    "darwinian",
    "window_25",
    "window_50",
    "window_100",
]
DEFAULT_SEEDS = [0, 1, 2, 10, 42]


def parse_int_list(value: str):
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_str_list(value: str):
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", default=",".join(DEFAULT_POLICIES))
    ap.add_argument("--seeds", default=",".join(str(x) for x in DEFAULT_SEEDS))
    ap.add_argument("--vocab", type=int, default=64)
    ap.add_argument("--neurons", type=int, default=192)
    ap.add_argument("--density", type=float, default=0.06)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--ticks", type=int, default=8)
    ap.add_argument("--budget", type=int, default=16000)
    ap.add_argument("--stale-limit", type=int, default=6000)
    return ap.parse_args()


def run_policy(policy_name, seed, config):
    net, targets = build_net_and_targets(config, seed)
    policy = build_policy(policy_name)
    outcome = run_budgeted_search(net, targets, config, policy)
    return {
        "policy": policy_name,
        "seed": seed,
        "best_score": outcome.best_score,
        "best_acc": outcome.best_acc,
        "steps": outcome.steps,
        "connections": outcome.connections,
        "state": outcome.policy_state,
    }


def main():
    args = parse_args()
    config = PermutationHarnessConfig(
        vocab=args.vocab,
        neurons=args.neurons,
        density=args.density,
        threshold=args.threshold,
        ticks=args.ticks,
        budget=args.budget,
        stale_limit=args.stale_limit,
    )
    policies = parse_str_list(args.policies)
    seeds = parse_int_list(args.seeds)

    print("CANONICAL MUTATION POLICY A/B", flush=True)
    print(
        f"V={config.vocab} N={config.neurons} density={config.density} "
        f"threshold={config.threshold} ticks={config.ticks} budget={config.budget}",
        flush=True,
    )
    print(f"Policies={policies} seeds={seeds}", flush=True)
    print("=" * 90, flush=True)

    results = []
    for policy_name in policies:
        print(f"\n[{policy_name}]", flush=True)
        for seed in seeds:
            result = run_policy(policy_name, seed, config)
            results.append(result)
            print(
                f"  seed={seed:3d} score={result['best_score']*100:5.1f}% "
                f"acc={result['best_acc']*100:5.1f}% "
                f"steps={result['steps']:5d} conns={result['connections']:4d} "
                f"state={result['state']}",
                flush=True,
            )

    print(f"\n{'='*90}", flush=True)
    print("SUMMARY", flush=True)
    for policy_name in policies:
        rows = [row for row in results if row["policy"] == policy_name]
        score_mean = statistics.mean(row["best_score"] for row in rows) * 100
        score_std = statistics.pstdev(row["best_score"] for row in rows) * 100
        acc_mean = statistics.mean(row["best_acc"] for row in rows) * 100
        print(
            f"  {policy_name:16s} score={score_mean:5.1f}% +/-{score_std:4.1f}pp "
            f"acc={acc_mean:5.1f}%",
            flush=True,
        )
    print(f"{'='*90}", flush=True)


if __name__ == "__main__":
    main()

