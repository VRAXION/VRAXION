"""
VRAXION v22 -- Split I/O vs Shared I/O
========================================
A: Shared I/O (current) -- neurons 0..V-1 = input AND output
B: Split I/O            -- neurons 0..V-1 = input, neurons N-V..N-1 = output

Same total neuron count, same training loop, 3 seeds.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
import time
import copy
from v22_best_config import SelfWiringGraph, softmax


class SplitIOGraph(SelfWiringGraph):
    """Modified: first V = input only, last V = output only."""

    def __init__(self, n_neurons, vocab, **kw):
        assert n_neurons >= 2 * vocab, f"Need N >= 2*V for split I/O (N={n_neurons}, V={vocab})"
        super().__init__(n_neurons, vocab, **kw)
        # output zone = last V neurons
        self.out_start = n_neurons - vocab

    def forward(self, world, ticks=8):
        act = self.state.copy()
        Weff = self.W * self.mask

        for t in range(ticks):
            if t == 0:
                # Inject into FIRST V neurons (input zone)
                act[:self.V] = world

            raw = act @ Weff + act * 0.1
            np.nan_to_num(raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

            self.charge += raw * 0.3
            self.charge *= self.leak
            act = np.maximum(self.charge - self.threshold, 0.0)
            self.charge = np.clip(self.charge,
                                  -self.threshold * 2,
                                  self.threshold * 2)

        self.state = act.copy()
        # Read from LAST V neurons (output zone)
        return self.charge[self.out_start:]

    def forward_batch(self, ticks=8):
        Weff = self.W * self.mask
        V, N = self.V, self.N
        # Input: one-hot into first V neurons
        worlds = np.eye(V, dtype=np.float32)
        charges = np.zeros((V, N), dtype=np.float32)
        acts = np.zeros((V, N), dtype=np.float32)

        for t in range(ticks):
            if t == 0:
                acts[:, :V] = worlds
            raw = acts @ Weff + acts * 0.1
            np.nan_to_num(raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            charges += raw * 0.3
            charges *= self.leak
            acts = np.maximum(charges - self.threshold, 0.0)
            charges = np.clip(charges, -self.threshold * 2,
                              self.threshold * 2)

        # Read from LAST V neurons (output zone)
        return charges[:, self.out_start:]


def score_batch(net, targets, V, ticks=8):
    logits = net.forward_batch(ticks)
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    preds = np.argmax(probs, axis=1)
    acc = (preds == targets).mean()
    tp = probs[np.arange(V), targets].mean()
    return 0.5 * acc + 0.5 * tp, acc


def train_ab(net, targets, V, max_att=4000, ticks=8):
    score_best, acc_best = score_batch(net, targets, V, ticks)
    stale = 0
    kept = 0

    for att in range(max_att):
        saved_m = net.mask.copy()
        saved_w = net.W.copy()

        net.mutate_structure(0.05)

        sc, ac = score_batch(net, targets, V, ticks)
        if sc > score_best:
            score_best = sc
            acc_best = ac
            kept += 1
            stale = 0
        else:
            net.mask = saved_m
            net.W = saved_w
            stale += 1

        if stale >= 3000:
            break
        if acc_best >= 0.99:
            break

    return acc_best, kept


# ============================================================
#  RUN
# ============================================================
print("=" * 60)
print("  SPLIT I/O vs SHARED I/O -- A/B Test")
print("=" * 60)

CONFIGS = [
    (16, 80),    # small
    (32, 128),   # medium
    (64, 256),   # large
]

SEEDS = [42, 77, 123]
MAX_ATT = 4000

for V, N in CONFIGS:
    print(f"\n  --- V={V}, N={N} (internals: shared={N-V}, split={N-2*V}) ---")

    shared_accs = []
    split_accs = []

    for seed in SEEDS:
        # A: Shared I/O (original)
        np.random.seed(seed); random.seed(seed)
        perm = np.random.permutation(V)

        np.random.seed(seed + 1000); random.seed(seed + 1000)
        net_shared = SelfWiringGraph(N, V)
        t0 = time.time()
        acc_s, kept_s = train_ab(net_shared, perm, V, MAX_ATT)
        t_s = time.time() - t0

        # B: Split I/O
        np.random.seed(seed + 1000); random.seed(seed + 1000)
        net_split = SplitIOGraph(N, V)
        t0 = time.time()
        acc_p, kept_p = train_ab(net_split, perm, V, MAX_ATT)
        t_p = time.time() - t0

        shared_accs.append(acc_s)
        split_accs.append(acc_p)

        print(f"    seed={seed:3d}  shared={acc_s*100:5.1f}% ({t_s:.1f}s)  "
              f"split={acc_p*100:5.1f}% ({t_p:.1f}s)  "
              f"delta={((acc_p - acc_s)*100):+.1f}%")

    avg_s = np.mean(shared_accs) * 100
    avg_p = np.mean(split_accs) * 100
    delta = avg_p - avg_s
    print(f"    AVG:     shared={avg_s:.1f}%  split={avg_p:.1f}%  delta={delta:+.1f}%")


# ============================================================
#  VERDICT
# ============================================================
print(f"\n{'='*60}")
print(f"  VERDICT")
print(f"{'='*60}")
print(f"  If split > shared: signal needs DISTANCE to develop")
print(f"  If shared > split: shortcut (same neuron) helps learning")
print(f"  If equal: I/O topology doesn't matter at this scale")
print(f"\n{'='*60}", flush=True)
