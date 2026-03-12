"""
A/B Test: Random Mutation vs Diff-Guided Mutation
=================================================
Tests whether targeting mutations at the worst-performing output neuron
breaks through the multi-class wall.

Runs both strategies on:
  1. 16-class lookup (the baseline)
  2. 8+8 mod 8 arithmetic (the hard logic task)
  3. 64-class lookup (the wall)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
import time
from v22_best_config import SelfWiringGraph, softmax


def train_ab(net, inputs, targets, vocab, mode='random', max_attempts=8000,
             ticks=6, encode_fn=None):
    """Train with either random or diff-guided mutation.
    encode_fn: if None, uses simple one-hot. Otherwise callable(input_idx) -> world vector."""

    def make_world(i):
        if encode_fn is not None:
            return encode_fn(i)
        world = np.zeros(vocab, dtype=np.float32)
        world[inputs[i]] = 1.0
        return world

    def evaluate(return_diff=False):
        net.reset()
        correct = 0
        # Accumulate per-output error for diff-guided mode
        diff_acc = np.zeros(vocab, dtype=np.float32) if return_diff else None
        for p in range(2):
            for i in range(len(inputs)):
                world = make_world(i)
                logits = net.forward(world, ticks)
                probs = softmax(logits[:vocab])
                if p == 1:
                    pred = np.argmax(probs)
                    if pred == targets[i]:
                        correct += 1
                    if return_diff:
                        # Target one-hot
                        target_vec = np.zeros(vocab, dtype=np.float32)
                        target_vec[targets[i]] = 1.0
                        diff_acc += (probs - target_vec)
        acc = correct / len(inputs)
        net.last_acc = acc
        if return_diff:
            diff_acc /= len(inputs)  # average diff per output
        return acc, diff_acc

    need_diff = (mode == 'diff_guided')
    score, diff = evaluate(return_diff=need_diff)
    best = score
    phase = "STRUCTURE"
    kept = 0
    stale = 0
    switched = False

    for att in range(max_attempts):
        state = net.save_state()

        # Mutate based on mode
        if mode == 'diff_guided' and diff is not None:
            # 70% diff-guided, 30% random (exploration)
            if random.random() < 0.7:
                net.mutate_diff_guided(diff, 0.05 if phase == "STRUCTURE" else 0.02)
            else:
                if phase == "STRUCTURE":
                    net.mutate_structure(0.05)
                else:
                    if random.random() < 0.3:
                        net.mutate_structure(0.02)
                    else:
                        net.mutate_weights()
        else:
            if phase == "STRUCTURE":
                net.mutate_structure(0.05)
            else:
                if random.random() < 0.3:
                    net.mutate_structure(0.02)
                else:
                    net.mutate_weights()

        net.self_wire()
        new_score, new_diff = evaluate(return_diff=need_diff)

        if new_score > score:
            score = new_score
            if need_diff:
                diff = new_diff
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
                  f"Kept: {kept:3d} | Phase: {phase} | Mode: {mode}")

        if best >= 0.99 or stale >= 6000:
            break

    return best, kept


def run_ab_test(task_name, setup_fn, vocab, seeds=[42, 7, 99]):
    """Run A/B comparison across multiple seeds."""
    print(f"\n{'='*60}")
    print(f"  A/B TEST: {task_name}")
    print(f"{'='*60}")

    results = {'random': [], 'diff_guided': []}

    for seed in seeds:
        for mode in ['random', 'diff_guided']:
            np.random.seed(seed)
            random.seed(seed)

            net, inputs, targets, encode_fn = setup_fn()

            print(f"\n  --- seed={seed}, mode={mode} ---")
            t0 = time.time()
            best, kept = train_ab(net, inputs, targets, vocab,
                                   mode=mode, encode_fn=encode_fn)
            elapsed = time.time() - t0
            print(f"  Result: {best*100:.1f}% | {elapsed:.1f}s | kept: {kept}")
            results[mode].append((seed, best, kept, elapsed))

    # Summary
    print(f"\n  {'='*50}")
    print(f"  SUMMARY: {task_name}")
    print(f"  {'='*50}")
    fmt = "  {:<15s} " + " ".join([f"seed={s:>3d}" for s in seeds]) + "   AVG"
    print(fmt.format("Mode"))
    print("  " + "-"*50)
    for mode in ['random', 'diff_guided']:
        accs = [r[1] for r in results[mode]]
        avg = np.mean(accs)
        vals = " ".join([f"{a*100:6.1f}%" for a in accs])
        marker = " ***" if mode == 'diff_guided' and avg > np.mean([r[1] for r in results['random']]) + 0.02 else ""
        print(f"  {mode:<15s} {vals}  {avg*100:5.1f}%{marker}")

    return results


if __name__ == "__main__":
    N_NEURONS = 256
    SEEDS = [42, 7, 99]

    # Task 1: 16-class lookup
    def setup_16class():
        V = 16
        net = SelfWiringGraph(N_NEURONS, V)
        perm = np.random.permutation(V)
        return net, list(range(V)), perm.tolist(), None

    r1 = run_ab_test("16-class lookup", setup_16class, 16, SEEDS)

    # Task 2: 8+8 mod 8 arithmetic
    def setup_mod8():
        mod_n = 8
        input_dim = mod_n * 2
        net = SelfWiringGraph(N_NEURONS, input_dim)
        inputs_a, inputs_b, targets = [], [], []
        for a in range(mod_n):
            for b in range(mod_n):
                inputs_a.append(a)
                inputs_b.append(b)
                targets.append((a + b) % mod_n)

        def encode_fn(i):
            world = np.zeros(input_dim, dtype=np.float32)
            world[inputs_a[i]] = 1.0
            world[mod_n + inputs_b[i]] = 1.0
            return world

        return net, list(range(len(inputs_a))), targets, encode_fn

    r2 = run_ab_test("mod 8 arithmetic", setup_mod8, 16, SEEDS)  # vocab=16 (input_dim)

    # Task 3: 64-class lookup
    def setup_64class():
        V = 64
        net = SelfWiringGraph(N_NEURONS, V)
        perm = np.random.permutation(V)
        return net, list(range(V)), perm.tolist(), None

    r3 = run_ab_test("64-class lookup", setup_64class, 64, SEEDS)

    # Final comparison
    print(f"\n{'='*60}")
    print(f"  FINAL COMPARISON")
    print(f"{'='*60}")
    for name, res in [("16-class lookup", r1), ("mod 8 arithmetic", r2), ("64-class lookup", r3)]:
        avg_r = np.mean([r[1] for r in res['random']])
        avg_d = np.mean([r[1] for r in res['diff_guided']])
        delta = (avg_d - avg_r) * 100
        marker = "+++" if delta > 5 else "++" if delta > 2 else "+" if delta > 0 else "---"
        print(f"  {name:<25s}  Random: {avg_r*100:5.1f}%  Diff-guided: {avg_d*100:5.1f}%  "
              f"Delta: {delta:+.1f}% {marker}")
