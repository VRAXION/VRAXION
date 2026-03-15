"""Test: can we skip float CSR and use int8 mask directly?

Current: acts @ sparse_csr_float32(mask * gain)
Option A: acts @ mask.astype(float32) * gain  (dense, no CSR)
Option B: acts @ sparse_csr_int8(mask) * gain  (sparse int8, gain after)
Option C: separate pos/neg masks, pure int8 logic

Check: same accuracy? Speed?
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
import time
from scipy import sparse
from model.graph import SelfWiringGraph

SEEDS = [42, 77, 123]
BUDGET = 16000


def run_with_method(method, V, N, density, seed):
    """Run training with different matmul methods."""
    np.random.seed(seed); random.seed(seed)
    net = SelfWiringGraph(N, V, density=density, threshold=0.5)
    perm = np.random.permutation(V)

    def eval_b():
        mask = net.mask
        clip_bound = net.threshold * net.clip_factor
        charges = np.zeros((V, N), dtype=np.float32)
        acts = np.zeros((V, N), dtype=np.float32)

        for t in range(8):
            if t == 0: acts[:, :V] = np.eye(V, dtype=np.float32)

            if method == 'float32_csr':
                # Current: float32 CSR
                Weff = sparse.csr_matrix(mask.astype(np.float32) * net.gain)
                raw = np.asarray(acts @ Weff) + acts * net.self_conn
            elif method == 'int8_dense':
                # Dense: cast int8 to float32 on the fly, multiply gain after
                raw = acts @ mask.astype(np.float32) * np.float32(net.gain) + acts * net.self_conn
            elif method == 'int8_csr':
                # Sparse CSR but store int8, multiply gain after
                Weff = sparse.csr_matrix(mask)
                raw = np.asarray(acts @ Weff).astype(np.float32) * np.float32(net.gain) + acts * net.self_conn
            elif method == 'int8_csr_cached':
                # Same as int8_csr but cache the CSR (rebuild only when dirty)
                if not hasattr(net, '_mask_csr') or net._weff_dirty:
                    net._mask_csr = sparse.csr_matrix(mask)
                    net._weff_dirty = False
                raw = np.asarray(acts @ net._mask_csr).astype(np.float32) * np.float32(net.gain) + acts * net.self_conn
            elif method == 'posneg':
                # Separate pos/neg: no float weights at all
                pos = (mask == 1)
                neg = (mask == -1)
                raw_pos = acts @ pos.astype(np.float32)
                raw_neg = acts @ neg.astype(np.float32)
                raw = (raw_pos - raw_neg) * np.float32(net.gain) + acts * net.self_conn

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
    return best_acc


def benchmark_forward(method, V, N, density, repeats=50):
    """Benchmark just the forward pass speed."""
    np.random.seed(42)
    net = SelfWiringGraph(N, V, density=density, threshold=0.5)
    mask = net.mask
    clip_bound = net.threshold * net.clip_factor

    # Warmup
    for _ in range(3):
        charges = np.zeros((V, N), dtype=np.float32)
        acts = np.zeros((V, N), dtype=np.float32)
        acts[:, :V] = np.eye(V, dtype=np.float32)
        _ = acts @ mask.astype(np.float32)

    times = []
    for _ in range(repeats):
        charges = np.zeros((V, N), dtype=np.float32)
        acts = np.zeros((V, N), dtype=np.float32)
        t0 = time.perf_counter()
        for t in range(8):
            if t == 0: acts[:, :V] = np.eye(V, dtype=np.float32)
            if method == 'float32_csr':
                Weff = sparse.csr_matrix(mask.astype(np.float32) * 2.0)
                raw = np.asarray(acts @ Weff) + acts * 0.05
            elif method == 'int8_dense':
                raw = acts @ mask.astype(np.float32) * np.float32(2.0) + acts * 0.05
            elif method == 'int8_csr_cached':
                if not hasattr(net, '_test_csr'):
                    net._test_csr = sparse.csr_matrix(mask)
                raw = np.asarray(acts @ net._test_csr).astype(np.float32) * np.float32(2.0) + acts * 0.05
            elif method == 'posneg':
                pos = (mask == 1).astype(np.float32)
                neg = (mask == -1).astype(np.float32)
                raw = (acts @ pos - acts @ neg) * np.float32(2.0) + acts * 0.05
            np.nan_to_num(raw, copy=False)
            charges += raw * 0.3
            charges *= 0.85
            acts = np.maximum(charges - 0.5, 0)
            charges = np.clip(charges, -1.0, 1.0)
        times.append(time.perf_counter() - t0)
    return np.median(times) * 1000


def main():
    methods = ['float32_csr', 'int8_dense', 'int8_csr_cached', 'posneg']

    print("INT8 MATMUL TEST", flush=True)
    print("=" * 80, flush=True)

    # Speed benchmark
    print("\nSPEED (forward_batch, median ms):", flush=True)
    for label, V, N, d in [("V64_N192", 64, 192, 0.06), ("V128_N384", 128, 384, 0.06)]:
        print(f"\n  {label} (density={d}):", flush=True)
        for m in methods:
            ms = benchmark_forward(m, V, N, d)
            print(f"    {m:20s}: {ms:6.2f} ms", flush=True)

    # Accuracy: V64 only (fast)
    print(f"\nACCURACY (V64_N192, 16K budget):", flush=True)
    print(f"  {'method':20s} {'s42':>6s} {'s77':>6s} {'s123':>6s} {'mean':>6s}", flush=True)
    for m in methods:
        accs = []
        for seed in SEEDS:
            t0 = time.perf_counter()
            acc = run_with_method(m, 64, 192, 0.06, seed)
            accs.append(acc)
            elapsed = time.perf_counter() - t0
        mean = np.mean(accs) * 100
        print(f"  {m:20s} {accs[0]*100:5.1f}% {accs[1]*100:5.1f}% {accs[2]*100:5.1f}% {mean:5.1f}%",
              flush=True)

    print(f"\n{'='*80}", flush=True)


if __name__ == '__main__':
    main()
