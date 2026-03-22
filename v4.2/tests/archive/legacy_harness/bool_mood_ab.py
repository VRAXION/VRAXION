"""A/B test: int mood vs bool mood.

Int mood: mood=0-3, ±1 step
Bool mood: grow=T/F, refine=T/F, independent flips
Intensity: int 1-15 vs bool aggressive T/F (3 vs 10)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
import time
from model.graph import SelfWiringGraph

SEEDS = [42, 77, 123]
BUDGET = 16000
V, N, DENSITY = 64, 192, 0.06


def run_int_mood(seed):
    """Current: int mood 0-3, int intensity 1-15."""
    np.random.seed(seed); random.seed(seed)
    net = SelfWiringGraph(N, V, density=DENSITY)
    perm = np.random.permutation(V)

    def eval_b():
        logits = net.forward_batch(ticks=8)
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
        acc = (np.argmax(probs, axis=1) == perm[:V]).mean()
        tp = probs[np.arange(V), perm[:V]].mean()
        return 0.5 * acc + 0.5 * tp

    score = eval_b()
    best = 0.0
    for att in range(BUDGET):
        state = net.save_state()
        net.mutate_with_mood()
        s = eval_b()
        if s > score:
            score = s; best = max(best, score)
        else:
            net.restore_state(state)
    return best, int(net.mood), int(net.intensity), int(net.loss_pct)


def run_bool_mood(seed):
    """Bool mood: grow(T/F) + refine(T/F), aggressive(T/F)."""
    np.random.seed(seed); random.seed(seed)
    net = SelfWiringGraph(N, V, density=DENSITY)
    perm = np.random.permutation(V)

    # Override mood with bools
    grow = True       # True=add, False=remove
    refine = False    # True=flip signs, False=change topology
    aggressive = True # True=10 changes, False=3

    def eval_b():
        logits = net.forward_batch(ticks=8)
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
        acc = (np.argmax(probs, axis=1) == perm[:V]).mean()
        tp = probs[np.arange(V), perm[:V]].mean()
        return 0.5 * acc + 0.5 * tp

    score = eval_b()
    best = 0.0

    for att in range(BUDGET):
        sm = net.mask.copy()
        lk_s = int(net.loss_pct)
        grow_s = grow; refine_s = refine; agg_s = aggressive

        # Bool mutation: independent coin flips
        if random.random() < 0.35:
            grow = not grow
        if random.random() < 0.35:
            refine = not refine
        if random.random() < 0.35:
            aggressive = not aggressive

        # Loss step
        if random.random() < 0.2:
            net.loss_pct = np.int8(max(1, min(50, int(net.loss_pct) + random.randint(-3, 3))))

        # Mask mutations based on bools
        n_changes = 10 if aggressive else 3
        for _ in range(n_changes):
            if grow and not refine:      # scout: add
                if random.random() < 0.7:
                    net._add()
                else:
                    net._flip()
            elif grow and refine:        # rewirer: reroute + add
                r = random.random()
                if r < 0.6:
                    net._rewire()
                elif r < 0.8:
                    net._flip()
                else:
                    net._add()
            elif not grow and refine:    # refiner: flip signs
                if random.random() < 0.8:
                    net._flip()
                else:
                    net._rewire()
            else:                         # pruner: remove
                r = random.random()
                if r < 0.7:
                    net._remove()
                elif r < 0.9:
                    net._flip()
                else:
                    net._rewire()

        s = eval_b()
        if s > score:
            score = s; best = max(best, score)
        else:
            net.mask = sm; net.resync_alive()
            net.loss_pct = np.int8(lk_s)
            grow = grow_s; refine = refine_s; aggressive = agg_s

    return best, grow, refine, aggressive, int(net.loss_pct)


def main():
    print("BOOL vs INT MOOD A/B TEST", flush=True)
    print(f"V={V} N={N} budget={BUDGET}", flush=True)
    print("=" * 70, flush=True)

    print("\nINT MOOD (current):", flush=True)
    int_scores = []
    for seed in SEEDS:
        t0 = time.perf_counter()
        score, mood, inten, loss = run_int_mood(seed)
        elapsed = time.perf_counter() - t0
        int_scores.append(score)
        print(f"  seed={seed}: {score*100:.1f}% mood={mood} int={inten} loss={loss}% ({elapsed:.0f}s)", flush=True)
    print(f"  MEAN: {np.mean(int_scores)*100:.1f}%", flush=True)

    print("\nBOOL MOOD:", flush=True)
    bool_scores = []
    for seed in SEEDS:
        t0 = time.perf_counter()
        score, grow, refine, agg, loss = run_bool_mood(seed)
        elapsed = time.perf_counter() - t0
        bool_scores.append(score)
        print(f"  seed={seed}: {score*100:.1f}% grow={grow} refine={refine} agg={agg} loss={loss}% ({elapsed:.0f}s)", flush=True)
    print(f"  MEAN: {np.mean(bool_scores)*100:.1f}%", flush=True)

    diff = np.mean(bool_scores) - np.mean(int_scores)
    print(f"\nDIFF: {diff*100:+.1f}pp", flush=True)
    print("=" * 70, flush=True)


if __name__ == '__main__':
    main()
