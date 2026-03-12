"""
A/B Test: Capacitor (Integrate-and-Fire) vs Leaky ReLU
======================================================
Tests whether biological capacitor neuron model creates natural sparsity
and breaks the multi-class interference wall.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
import time
from v22_best_config import SelfWiringGraph, softmax


def train_mode(net, inputs, targets, vocab, fwd_mode='capacitor',
               max_attempts=4000, ticks=6):
    """Train with specified forward mode."""

    def evaluate():
        net.reset()
        correct = 0
        for p in range(2):
            for i in range(len(inputs)):
                world = np.zeros(vocab, dtype=np.float32)
                world[inputs[i]] = 1.0
                logits = net.forward(world, ticks, mode=fwd_mode)
                probs = softmax(logits[:vocab])
                if p == 1 and np.argmax(probs) == targets[i]:
                    correct += 1
        acc = correct / len(inputs)
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
            # Measure sparsity: how many neurons have charge > 0.1
            net.reset()
            active_counts = []
            for i in range(len(inputs)):
                world = np.zeros(vocab, dtype=np.float32)
                world[inputs[i]] = 1.0
                net.forward(world, ticks, mode=fwd_mode)
                active = (np.abs(net.state) > 0.1).sum()
                active_counts.append(active)
            avg_active = np.mean(active_counts)
            sparsity = 1.0 - avg_active / net.N

            print(f"  [{att+1:5d}] Acc: {best*100:5.1f}% | "
                  f"Conns: {net.count_connections():4d} | "
                  f"Kept: {kept:3d} | "
                  f"Active: {avg_active:.0f}/{net.N} ({sparsity*100:.0f}% sparse) | "
                  f"Mode: {fwd_mode}")

        if best >= 0.99 or stale >= 3500:
            break

    return best, kept


def run_test(task_name, n_classes, n_neurons, seeds=[42, 7, 99]):
    print(f"\n{'='*65}")
    print(f"  A/B TEST: {task_name}")
    print(f"{'='*65}")

    results = {}
    for mode in ['leaky_relu', 'capacitor']:
        results[mode] = []
        for seed in seeds:
            np.random.seed(seed)
            random.seed(seed)

            V = n_classes
            net = SelfWiringGraph(n_neurons, V)
            perm = np.random.permutation(V)
            inputs = list(range(V))

            print(f"\n  --- seed={seed}, mode={mode} ---")
            t0 = time.time()
            best, kept = train_mode(net, inputs, perm.tolist(), V,
                                     fwd_mode=mode)
            elapsed = time.time() - t0
            print(f"  Result: {best*100:.1f}% | {elapsed:.1f}s | kept: {kept}")
            results[mode].append((seed, best, kept, elapsed))

    # Summary
    print(f"\n  {'='*55}")
    print(f"  SUMMARY: {task_name}")
    print(f"  {'='*55}")
    fmt = "  {:<15s} " + " ".join([f"seed={s:>3d}" for s in seeds]) + "   AVG"
    print(fmt.format("Mode"))
    print("  " + "-"*55)
    for mode in ['leaky_relu', 'capacitor']:
        accs = [r[1] for r in results[mode]]
        avg = np.mean(accs)
        vals = " ".join([f"{a*100:6.1f}%" for a in accs])
        print(f"  {mode:<15s} {vals}  {avg*100:5.1f}%")

    return results


if __name__ == "__main__":
    N = 256
    SEEDS = [42]  # 1 seed for speed — 10 min A/B limit

    r1 = run_test("16-class lookup", 16, N, SEEDS)
    r2 = run_test("64-class lookup", 64, N, SEEDS)

    # Final
    print(f"\n{'='*65}")
    print(f"  FINAL COMPARISON")
    print(f"{'='*65}")
    for name, res in [("16-class", r1), ("64-class", r2)]:
        avg_lr = np.mean([r[1] for r in res['leaky_relu']])
        avg_cap = np.mean([r[1] for r in res['capacitor']])
        delta = (avg_cap - avg_lr) * 100
        marker = "+++" if delta > 5 else "++" if delta > 2 else "+" if delta > 0 else "---"
        print(f"  {name:<15s}  LeakyReLU: {avg_lr*100:5.1f}%  "
              f"Capacitor: {avg_cap*100:5.1f}%  Delta: {delta:+.1f}% {marker}")
