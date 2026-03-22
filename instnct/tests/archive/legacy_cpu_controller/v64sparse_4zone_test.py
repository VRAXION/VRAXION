"""Quick V64_sparse test: does 4-zone mood work with different init densities?
The earlier failure was at init=0.02. Try higher starts too."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
import time
from model.graph import SelfWiringGraph
from scipy import sparse

SEEDS = [42, 77, 123]
BUDGET = 32000
V, N, THRESHOLD = 64, 192, 0.5

INIT_DENSITIES = [0.02, 0.06, 0.10, 0.15]


def run_one(density, seed):
    np.random.seed(seed); random.seed(seed)
    net = SelfWiringGraph(N, V, density=density, threshold=THRESHOLD)
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
        return acc, 0.5 * acc + 0.5 * tp

    _, score = eval_b()
    best_acc = 0.0

    for att in range(BUDGET):
        sm = net.mask.copy()
        mx_s = net.mood_x; mz_s = net.mood_z; lk_s = net.leak

        net.mutate_with_mood()

        a, s = eval_b()
        if s > score:
            score = s; best_acc = max(best_acc, a)
        else:
            net.mask = sm; net.resync_alive(); net.mood_x = mx_s; net.mood_z = mz_s; net.leak = lk_s

    final_conns = int((net.mask != 0).sum())
    final_density = final_conns / (N * N - N)
    return best_acc, net.leak, net.mood_x, init_conns, final_conns, final_density


def main():
    print(f"V64_SPARSE 4-ZONE MOOD TEST", flush=True)
    print(f"V={V}, N={N}, budget={BUDGET}, seeds={SEEDS}", flush=True)
    print(f"{'='*85}", flush=True)
    print(f"  {'init_d':>7s} {'seed':>5s} {'acc':>6s} {'leak':>6s} {'mood_x':>7s} "
          f"{'conns':>12s} {'final_d':>8s}", flush=True)
    print(f"  {'-'*75}", flush=True)

    for d in INIT_DENSITIES:
        accs = []
        for seed in SEEDS:
            t0 = time.perf_counter()
            acc, leak, mood_x, ic, fc, fd = run_one(d, seed)
            elapsed = time.perf_counter() - t0
            accs.append(acc)
            print(f"  {d:7.3f} {seed:5d} {acc*100:5.1f}% {leak:6.3f} {mood_x:7.2f} "
                  f"{ic:5d}->{fc:5d} {fd:8.4f}  ({elapsed:.0f}s)", flush=True)
        print(f"  {d:7.3f}  MEAN {np.mean(accs)*100:5.1f}%", flush=True)
        print(flush=True)

    print(f"{'='*85}", flush=True)


if __name__ == '__main__':
    main()
