"""
v22 Logic Test: Composition (A + B > C)
=======================================
Two-step logic: first compute A+B, then compare to C.
This is the hardest test — requires chained computation.

Input: one-hot(A) || one-hot(B) || one-hot(C)
Output: 2 classes (0 = A+B <= C, 1 = A+B > C)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
import time
from v22_best_config import SelfWiringGraph, softmax


def make_composition_dataset(N):
    """Generate all (A, B, C) triples with A+B > C label."""
    inputs_a, inputs_b, inputs_c, targets = [], [], [], []
    max_sum = 2 * (N - 1)  # max possible A+B
    for a in range(N):
        for b in range(N):
            for c in range(max_sum + 1):
                if c >= N:
                    continue  # keep C in same range for encoding
                inputs_a.append(a)
                inputs_b.append(b)
                inputs_c.append(c)
                targets.append(1 if (a + b) > c else 0)
    return inputs_a, inputs_b, inputs_c, targets


def train_composition(net, inputs_a, inputs_b, inputs_c, targets, range_n,
                      n_neurons, max_attempts=8000, ticks=6):
    input_dim = range_n * 3
    V = 2

    def evaluate():
        net.reset()
        correct = 0
        n = len(inputs_a)
        for p in range(2):
            for i in range(n):
                world = np.zeros(input_dim, dtype=np.float32)
                world[inputs_a[i]] = 1.0
                world[range_n + inputs_b[i]] = 1.0
                world[range_n * 2 + inputs_c[i]] = 1.0
                logits = net.forward(world, ticks)
                probs = softmax(logits[:V])
                if p == 1 and np.argmax(probs) == targets[i]:
                    correct += 1
        acc = correct / n
        net.last_acc = acc
        return acc

    score = evaluate()
    best = score
    phase = "STRUCTURE"
    kept = 0
    stale = 0
    switched = False

    for att in range(max_attempts):
        state = net.save_state()

        if phase == "STRUCTURE":
            net.mutate_structure(0.05)
        else:
            if random.random() < 0.3:
                net.mutate_structure(0.02)
            else:
                net.mutate_weights()

        net.self_wire()
        new_score = evaluate()

        if new_score > score:
            score = new_score
            kept += 1
            stale = 0
            best = max(best, score)
        else:
            net.restore_state(state)
            stale += 1

        if phase == "STRUCTURE" and stale > 2500 and not switched:
            phase = "BOTH"
            switched = True
            stale = 0

        if (att + 1) % 1000 == 0:
            print(f"  [{att+1:5d}] Acc: {best*100:5.1f}% | "
                  f"Conns: {net.count_connections():4d} | "
                  f"Kept: {kept:3d} | Phase: {phase}")

        if best >= 0.99 or stale >= 6000:
            break

    return best, kept


if __name__ == "__main__":
    SEED = 42
    N_NEURONS = 256

    configs = [
        (3, "A+B > C (range 0-2, 27 triples)"),
        (4, "A+B > C (range 0-3, 64 triples)"),
        (5, "A+B > C (range 0-4, 125 triples)"),
    ]

    print("=" * 60)
    print("v22 Logic Test: Composition (A + B > C)")
    print("=" * 60)

    results = []
    for range_n, label in configs:
        np.random.seed(SEED)
        random.seed(SEED)

        inputs_a, inputs_b, inputs_c, targets = make_composition_dataset(range_n)
        input_dim = range_n * 3
        net = SelfWiringGraph(N_NEURONS, input_dim)

        n1 = sum(targets)
        majority = max(n1, len(targets)-n1) / len(targets)

        print(f"\n--- {label} ---")
        print(f"  Neurons: {N_NEURONS}, Input dim: {input_dim}, Output: 2 classes")
        print(f"  Dataset: {len(targets)} triples")
        print(f"  Balance: {n1} true / {len(targets)-n1} false (majority: {majority*100:.1f}%)")

        t0 = time.time()
        best, kept = train_composition(net, inputs_a, inputs_b, inputs_c,
                                        targets, range_n, N_NEURONS)
        elapsed = time.time() - t0

        print(f"\n  Result: {best*100:.1f}% (majority: {majority*100:.1f}%) "
              f"| {elapsed:.1f}s | kept: {kept}")
        results.append((label, best, majority, elapsed))

    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    fmt = "  {:<35s} {:>8s} {:>10s} {:>8s}"
    print(fmt.format("Task", "Acc", "Majority", "Time"))
    print(fmt.format("-"*35, "-"*8, "-"*10, "-"*8))
    for label, best, maj, t in results:
        marker = " ***" if best > maj + 0.05 else ""
        print(fmt.format(label, f"{best*100:.1f}%", f"{maj*100:.1f}%", f"{t:.1f}s") + marker)
