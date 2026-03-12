"""
v22 Logic Test: Modular Arithmetic (A + B mod N = C)
====================================================
Tests whether the network can COMPUTE, not just memorize lookup tables.

Encoding: one-hot(A) concatenated with one-hot(B) as input.
Output: one-hot(C) where C = (A + B) mod N.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
import time
from v22_best_config import SelfWiringGraph, softmax


def make_dataset(N):
    """Generate all (A+B) mod N pairs."""
    inputs_a, inputs_b, targets = [], [], []
    for a in range(N):
        for b in range(N):
            inputs_a.append(a)
            inputs_b.append(b)
            targets.append((a + b) % N)
    return inputs_a, inputs_b, targets


def train_arithmetic(net, inputs_a, inputs_b, targets, mod_n, n_neurons,
                     max_attempts=8000, ticks=6):
    """Train on A+B mod N. Input = one-hot(A)||one-hot(B), output = first mod_n neurons."""
    V = mod_n  # output classes
    input_dim = mod_n * 2  # concatenated one-hot

    def evaluate():
        net.reset()
        correct = 0
        n = len(inputs_a)
        for p in range(2):
            for i in range(n):
                world = np.zeros(input_dim, dtype=np.float32)
                world[inputs_a[i]] = 1.0
                world[mod_n + inputs_b[i]] = 1.0
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
            pos, neg = net.pos_neg_ratio()
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
        (4, "4+4 mod 4 (16 pairs)"),
        (6, "6+6 mod 6 (36 pairs)"),
        (8, "8+8 mod 8 (64 pairs)"),
    ]

    print("=" * 60)
    print("v22 Logic Test: Modular Arithmetic (A + B mod N)")
    print("=" * 60)

    results = []
    for mod_n, label in configs:
        np.random.seed(SEED)
        random.seed(SEED)

        inputs_a, inputs_b, targets = make_dataset(mod_n)
        input_dim = mod_n * 2
        net = SelfWiringGraph(N_NEURONS, input_dim)

        print(f"\n--- {label} ---")
        print(f"  Neurons: {N_NEURONS}, Input dim: {input_dim}, Output: {mod_n} classes")
        print(f"  Dataset: {len(inputs_a)} pairs")

        t0 = time.time()
        best, kept = train_arithmetic(net, inputs_a, inputs_b, targets,
                                       mod_n, N_NEURONS)
        elapsed = time.time() - t0

        random_baseline = 1.0 / mod_n
        print(f"\n  Result: {best*100:.1f}% (random: {random_baseline*100:.1f}%) "
              f"| {elapsed:.1f}s | kept: {kept}")
        results.append((label, best, random_baseline, elapsed))

    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    fmt = "  {:<25s} {:>8s} {:>8s} {:>8s}"
    print(fmt.format("Task", "Acc", "Random", "Time"))
    print(fmt.format("-"*25, "-"*8, "-"*8, "-"*8))
    for label, best, rnd, t in results:
        print(fmt.format(label, f"{best*100:.1f}%", f"{rnd*100:.1f}%", f"{t:.1f}s"))
