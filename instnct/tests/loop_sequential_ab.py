"""A/B test: loop mutations on a SEQUENTIAL task.

Previous test showed loops hurt on permutation (stateless) tasks.
This test checks whether loops help on a sequential task where
the network must remember previous tokens — the temporal memory
that reverberating loops could provide.

Toy task: predict next token in repeating sequences.
  - Pattern A: 0 1 2 3 0 1 2 3 ...  (cycle-4)
  - Pattern B: 0 1 0 2 0 1 0 2 ...  (cycle-4 with interleaving)
  - Pattern C: 0 0 1 1 2 2 0 0 1 1 2 2 ...  (doubled cycle)

The network processes tokens one at a time, keeping hidden state
between tokens. Loops should help maintain memory of recent tokens.
"""

from __future__ import annotations

import statistics
import sys
import os
import random

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.graph import SelfWiringGraph


# --- Toy sequential patterns ---

PATTERNS = {
    "cycle4":    [0, 1, 2, 3],
    "interleave": [0, 1, 0, 2],
    "doubled":   [0, 0, 1, 1, 2, 2],
}


def make_sequence(pattern, length=60):
    """Tile pattern to length."""
    reps = length // len(pattern) + 1
    return (pattern * reps)[:length]


def eval_sequential(net, sequence, ticks=8):
    """Feed sequence token-by-token, measure next-token prediction accuracy."""
    net.reset()
    V = net.V
    correct = 0
    total = 0
    for i in range(len(sequence) - 1):
        token = sequence[i]
        target = sequence[i + 1]
        one_hot = np.zeros(V, dtype=np.float32)
        one_hot[token] = 1.0
        logits = net.forward(one_hot, ticks=ticks)
        pred = int(np.argmax(logits))
        if pred == target:
            correct += 1
        total += 1
    return correct / total if total else 0.0


def run_search(net, sequence, budget, policy, ticks=8):
    """Hill-climbing search with given mutation policy."""
    best_acc = eval_sequential(net, sequence, ticks=ticks)
    stale = 0
    accepts = {"add": 0, "add_loop": 0, "other": 0}

    for step in range(1, budget + 1):
        state = net.save_state()

        # Mutation
        if policy == "drive":
            net.mutate()
        elif policy == "loop_only":
            net.mutate(forced_op="add_loop")
        elif policy.startswith("loop_mix_"):
            prob = float(policy.split("_")[-1])
            if random.random() < prob:
                net.mutate(forced_op="add_loop")
                accepts["add_loop"] += 1
            else:
                net.mutate()
                accepts["other"] += 1
        else:
            net.mutate()

        acc = eval_sequential(net, sequence, ticks=ticks)
        if acc > best_acc:
            best_acc = acc
            stale = 0
        else:
            net.restore_state(state)
            stale += 1

        if best_acc >= 1.0:
            break
        if stale >= 2000:
            break

    return {
        "best_acc": best_acc,
        "steps": step,
        "connections": net.count_connections(),
        "accepts": accepts,
    }


def main():
    V = 8
    H = 64
    TICKS = 8
    BUDGET = 4000
    DENSITY = 0.20
    THETA = 1.0
    SEEDS = [0, 1, 2, 10, 42]
    POLICIES = ["drive", "loop_mix_0.3", "loop_mix_0.5", "loop_mix_0.7", "loop_only"]

    print("LOOP MUTATION — SEQUENTIAL TASK A/B", flush=True)
    print(f"V={V} H={H} ticks={TICKS} budget={BUDGET} density={DENSITY} theta={THETA}", flush=True)
    print(f"seeds={SEEDS}", flush=True)
    print(f"Policies={POLICIES}", flush=True)
    print("=" * 90, flush=True)

    all_results = []

    for pat_name, pattern in PATTERNS.items():
        seq = make_sequence(pattern, length=60)
        print(f"\n--- Pattern: {pat_name} = {pattern} ---", flush=True)

        for policy in POLICIES:
            accs = []
            for seed in SEEDS:
                np.random.seed(seed)
                random.seed(seed)
                net = SelfWiringGraph(V, hidden=H, density=DENSITY, theta_init=THETA,
                                      decay_init=1.0, seed=seed)
                result = run_search(net, seq, BUDGET, policy, ticks=TICKS)
                accs.append(result["best_acc"])
                all_results.append({
                    "pattern": pat_name,
                    "policy": policy,
                    "seed": seed,
                    **result,
                })

            mean_acc = statistics.mean(accs) * 100
            std_acc = statistics.pstdev(accs) * 100
            print(
                f"  {policy:16s}  acc={mean_acc:5.1f}% ±{std_acc:4.1f}pp  "
                f"[{', '.join(f'{a*100:.0f}' for a in accs)}]",
                flush=True,
            )

    # Grand summary
    print(f"\n{'='*90}", flush=True)
    print("GRAND SUMMARY (averaged across all patterns)", flush=True)
    for policy in POLICIES:
        rows = [r for r in all_results if r["policy"] == policy]
        mean_acc = statistics.mean(r["best_acc"] for r in rows) * 100
        std_acc = statistics.pstdev(r["best_acc"] for r in rows) * 100
        mean_conns = statistics.mean(r["connections"] for r in rows)
        print(
            f"  {policy:16s}  acc={mean_acc:5.1f}% ±{std_acc:4.1f}pp  conns={mean_conns:.0f}",
            flush=True,
        )
    print(f"{'='*90}", flush=True)


if __name__ == "__main__":
    main()
