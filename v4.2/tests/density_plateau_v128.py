"""V128 density plateau test — runs until convergence.

Shows acc, target_prob, score, density, connections at every checkpoint.
V128 never reaches 100%, so we can see where density actually stabilizes.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
import time
from model.graph import SelfWiringGraph
from scipy import sparse

SEED = 42
MAX_ATTEMPTS = 200000
STALE_WINDOW = 15000   # plateau: no improvement in 15K attempts
CHECK_INTERVAL = 2000

# V128 — the hard config where acc plateaus around 80%
V, N, DENSITY, THRESHOLD = 128, 384, 0.06, 0.5


def main():
    np.random.seed(SEED); random.seed(SEED)
    net = SelfWiringGraph(N, V, density=DENSITY, threshold=THRESHOLD)
    perm = np.random.permutation(V)

    init_conns = int((net.mask != 0).sum())

    def eval_b():
        Weff_csr = sparse.csr_matrix(net.mask.astype(np.float32) * net.gain)
        clip_bound = net.threshold * net.clip_factor
        charges = np.zeros((V, N), dtype=np.float32)
        acts = np.zeros((V, N), dtype=np.float32)
        for t in range(8):
            if t == 0: acts[:, :V] = np.eye(V, dtype=np.float32)
            raw = np.asarray(acts @ Weff_csr) + acts * net.self_conn
            np.nan_to_num(raw, copy=False)
            charges += raw * net.charge_rate
            charges *= np.float32(net.leak)
            acts = np.maximum(charges - net.threshold, 0)
            charges = np.clip(charges, -clip_bound, clip_bound)
        out = charges[:, net.out_start:net.out_start + V]
        e = np.exp(out - out.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
        preds = np.argmax(probs, axis=1)
        acc = (preds == perm[:V]).mean()
        tp = probs[np.arange(V), perm[:V]].mean()
        return acc, tp, 0.5 * acc + 0.5 * tp

    print(f"V128_N384 DENSITY PLATEAU TEST", flush=True)
    print(f"Seed={SEED}, stale_window={STALE_WINDOW}, max={MAX_ATTEMPTS}", flush=True)
    print(f"Init: {init_conns} connections, density={DENSITY:.3f}", flush=True)
    print(f"{'='*95}", flush=True)
    print(f"  {'att':>7s} {'acc':>6s} {'tgt_prob':>8s} {'score':>7s} "
          f"{'conns':>6s} {'density':>8s} {'leak':>6s} {'stale':>6s} {'elapsed':>8s}", flush=True)
    print(f"  {'-'*85}", flush=True)

    acc0, tp0, score = eval_b()
    best_acc = 0.0
    best_tp = 0.0
    stale = 0
    t0 = time.perf_counter()

    for att in range(MAX_ATTEMPTS):
        sm = net.mask.copy()
        mx_s = net.mood_x; mz_s = net.mood_z; lk_s = net.leak

        net.mutate_with_mood()

        a, tp, s = eval_b()
        if s > score:
            score = s
            best_acc = max(best_acc, a)
            best_tp = tp
            stale = 0
        else:
            net.mask = sm; net.mood_x = mx_s; net.mood_z = mz_s; net.leak = lk_s
            stale += 1

        if (att + 1) % CHECK_INTERVAL == 0:
            conns = int((net.mask != 0).sum())
            dens = conns / (N * N - N)
            elapsed = time.perf_counter() - t0
            print(f"  {att+1:7d} {best_acc*100:5.1f}% {best_tp:8.4f} {score:7.4f} "
                  f"{conns:6d} {dens:8.4f} {net.leak:6.3f} {stale:6d} {elapsed:7.0f}s",
                  flush=True)

        if stale >= STALE_WINDOW:
            conns = int((net.mask != 0).sum())
            dens = conns / (N * N - N)
            elapsed = time.perf_counter() - t0
            print(f"\n  PLATEAU at attempt {att+1}", flush=True)
            print(f"  Final: acc={best_acc*100:.1f}% tgt_prob={best_tp:.4f} "
                  f"score={score:.4f}", flush=True)
            print(f"  Density: {DENSITY:.3f} -> {dens:.4f} "
                  f"({init_conns} -> {conns} connections)", flush=True)
            print(f"  Leak: {net.leak:.4f}", flush=True)
            print(f"  Time: {elapsed:.0f}s", flush=True)
            break
    else:
        conns = int((net.mask != 0).sum())
        dens = conns / (N * N - N)
        elapsed = time.perf_counter() - t0
        print(f"\n  HIT MAX ({MAX_ATTEMPTS}) — did NOT plateau!", flush=True)
        print(f"  Final: acc={best_acc*100:.1f}% density={dens:.4f} "
              f"conns={conns}", flush=True)

    print(f"{'='*95}", flush=True)


if __name__ == '__main__':
    main()
