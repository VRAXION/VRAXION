"""Which discrete leak values actually get used and which are dead zones?

Run training, track every accepted leak value. Build histogram of
what the network actually visits vs what it avoids.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import numpy as np
import random
import time
from model.graph import SelfWiringGraph
from collections import Counter

SEEDS = [42, 77, 123]
BUDGET = 32000
CONFIGS = {
    "V64_N192": (64, 192, 0.06),
    "V128_N384": (128, 384, 0.06),
    "V128_dense": (128, 384, 0.15),
    "V256_N768": (256, 768, 0.06),
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="V64_N192", help="Comma-separated config names")
    ap.add_argument("--budget", type=int, default=BUDGET)
    ap.add_argument("--seeds", default="42,77,123")
    return ap.parse_args()


def parse_int_csv(s):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def run_one(vocab, neurons, density, budget, seed):
    np.random.seed(seed); random.seed(seed)
    net = SelfWiringGraph(neurons, vocab, density=density)
    perm = np.random.permutation(vocab)

    # Track every accepted leak value (quantized to 2 decimals)
    leak_accepts = Counter()
    leak_trajectory = []

    def eval_b():
        raw_out = net.forward_batch(ticks=8)
        e = np.exp(raw_out - raw_out.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
        acc = (np.argmax(probs, axis=1) == perm[:vocab]).mean()
        tp = probs[np.arange(vocab), perm[:vocab]].mean()
        return acc, 0.5 * acc + 0.5 * tp

    _, score = eval_b()
    best_acc = 0.0

    for att in range(budget):
        sm = net.mask.copy()
        mood_s = net.mood; int_s = net.intensity; lk_s = net.leak

        net.mutate_with_mood()

        a, s = eval_b()
        if s > score:
            score = s; best_acc = max(best_acc, a)
            # Record accepted leak
            lk_int = int(round(net.leak * 100))
            leak_accepts[lk_int] += 1
        else:
            net.mask = sm; net.mood = mood_s; net.intensity = int_s; net.leak = lk_s

        if att % 4000 == 0:
            leak_trajectory.append((att, round(net.leak, 3)))

    leak_trajectory.append((budget, round(net.leak, 3)))
    return best_acc, net.leak, leak_accepts, leak_trajectory


def print_histogram(all_accepts):
    print("GLOBAL LEAK HISTOGRAM (all seeds combined):", flush=True)
    print(f"  {'leak':>5s} {'accepts':>8s} {'bar'}", flush=True)
    max_count = max(all_accepts.values()) if all_accepts else 1
    for lk in range(50, 100):
        cnt = all_accepts.get(lk, 0)
        bar = '#' * int(cnt / max_count * 40) if cnt > 0 else ''
        if cnt > 0:
            print(f"  {lk/100:.2f} {cnt:8d} {bar}", flush=True)

    visited = set(all_accepts.keys())
    dead = [v for v in range(50, 100) if v not in visited]
    active = [v for v in range(50, 100) if v in visited]
    print(f"\nActive values: {len(active)}/50", flush=True)
    print(f"Dead values: {len(dead)}/50", flush=True)
    if dead:
        print(f"Dead range: {min(dead)/100:.2f} - {max(dead)/100:.2f}", flush=True)
        dead_ranges = []
        start = dead[0]
        for i in range(1, len(dead)):
            if dead[i] != dead[i-1] + 1:
                dead_ranges.append((start, dead[i-1]))
                start = dead[i]
        dead_ranges.append((start, dead[-1]))
        print(f"Dead zones: {[(f'{a/100:.2f}-{b/100:.2f}') for a,b in dead_ranges]}", flush=True)


def main():
    args = parse_args()
    config_names = [x.strip() for x in args.configs.split(",") if x.strip()]
    seeds = parse_int_csv(args.seeds)

    print("LEAK DISCRETE VALUE SWEEP", flush=True)
    print(f"configs={config_names} budget={args.budget} seeds={seeds}", flush=True)
    print("=" * 80, flush=True)

    for config_name in config_names:
        if config_name not in CONFIGS:
            raise SystemExit(f"Unknown config: {config_name}")
        vocab, neurons, density = CONFIGS[config_name]

        print(f"\n--- {config_name} (V={vocab} N={neurons} density={density}) ---", flush=True)
        all_accepts = Counter()

        for seed in seeds:
            t0 = time.perf_counter()
            acc, final_leak, accepts, trajectory = run_one(vocab, neurons, density, args.budget, seed)
            elapsed = time.perf_counter() - t0
            all_accepts += accepts

            print(f"\nseed={seed}: acc={acc*100:.1f}% final_leak={final_leak:.3f} ({elapsed:.0f}s)", flush=True)
            print(f"  trajectory: {trajectory}", flush=True)
            print(f"  top accepted leak values:", flush=True)
            for lk, cnt in accepts.most_common(10):
                print(f"    {lk/100:.2f}: {cnt} accepts", flush=True)

        print(f"\n{'-'*80}", flush=True)
        print_histogram(all_accepts)
        print(f"\n{'='*80}", flush=True)


if __name__ == '__main__':
    main()
