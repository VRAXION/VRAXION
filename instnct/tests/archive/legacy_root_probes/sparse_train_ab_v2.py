"""
Sparse Forward in Training — Fair A/B (v2)
============================================
Uses net.forward_batch() for dense baseline (not a reimplementation)
to ensure we're comparing against the actual NumPy BLAS path.

Compares:
  A) Dense: net.forward_batch (original graph.py method)
  B) C CSR: csr_forward with C CSR builder (rebuild every attempt)
  C) C CSR lazy: only rebuild on mutation, reuse on reject
"""

import sys, os, time
import numpy as np
import random as pyrandom

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model.graph import SelfWiringGraph

# Clear stale compiled lib
import glob as globmod
for f in globmod.glob('/tmp/edge_scatter*'):
    os.remove(f)

sys.path.insert(0, os.path.dirname(__file__))
from sparse_forward_ab import (
    forward_batch_c_csr, build_csr_arrays, _get_c_lib, check_correctness,
)


def score(logits, targets, V):
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    acc = (np.argmax(probs, axis=1)[:V] == targets[:V]).mean()
    tp = probs[np.arange(V), targets[:V]].mean()
    return 0.5 * acc + 0.5 * tp


def train_dense(net, targets, V, max_att=4000, ticks=8, stale_limit=3000):
    """Dense baseline using net.forward_batch (actual BLAS path)."""
    logits = net.forward_batch(ticks)
    sc = score(logits, targets, V)
    best = sc
    stale = 0

    for att in range(max_att):
        old_loss = int(net.loss_pct)
        old_drive = int(net.drive)
        undo = net.mutate()

        logits = net.forward_batch(ticks)
        new_sc = score(logits, targets, V)

        if new_sc > sc:
            sc = new_sc
            best = max(best, sc)
            stale = 0
        else:
            net.replay(undo)
            net.loss_pct = np.int8(old_loss)
            net.drive = np.int8(old_drive)
            stale += 1

        if best >= 0.99 or stale >= stale_limit:
            break

    return best, att + 1


def train_csr_lazy(net, targets, V, max_att=4000, ticks=8, stale_limit=3000):
    """C CSR training — rebuild on mutation, reuse old CSR on reject."""
    csr = build_csr_arrays(net)
    logits = forward_batch_c_csr(net, ticks, csr)
    sc = score(logits, targets, V)
    best = sc
    stale = 0

    for att in range(max_att):
        old_loss = int(net.loss_pct)
        old_drive = int(net.drive)
        old_csr = csr

        undo = net.mutate()
        csr = build_csr_arrays(net)

        logits = forward_batch_c_csr(net, ticks, csr)
        new_sc = score(logits, targets, V)

        if new_sc > sc:
            sc = new_sc
            best = max(best, sc)
            stale = 0
        else:
            net.replay(undo)
            net.loss_pct = np.int8(old_loss)
            net.drive = np.int8(old_drive)
            csr = old_csr
            stale += 1

        if best >= 0.99 or stale >= stale_limit:
            break

    return best, att + 1


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab", type=str, default="32,64,128,256")
    parser.add_argument("--attempts", type=int, default=4000)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--ticks", type=int, default=8)
    args = parser.parse_args()

    vocab_list = [int(v) for v in args.vocab.split(",")]

    print("Compiling C sparse library...")
    lib = _get_c_lib()
    if not lib:
        print("FAILED")
        sys.exit(1)
    print("OK\n")

    # Correctness
    print("Correctness check (V=32):")
    np.random.seed(42)
    net32 = SelfWiringGraph(32)
    check_correctness(net32)
    print()

    print("=" * 80)
    print("SPARSE CSR TRAINING — FAIR A/B (dense=net.forward_batch)")
    print("=" * 80)
    print(f"Attempts={args.attempts}, Seeds={args.seeds}, Ticks={args.ticks}\n")

    header = (f"{'V':>5} {'seed':>4} | {'dense_sc':>8} {'dense_ms':>9} | "
              f"{'csr_sc':>8} {'csr_ms':>9} {'speedup':>7}")
    print(header)
    print("-" * len(header))

    for V in vocab_list:
        dense_ms_all = []
        csr_ms_all = []
        for s in range(args.seeds):
            seed = 42 + s

            # Dense
            np.random.seed(seed)
            pyrandom.seed(seed)
            net = SelfWiringGraph(V)
            targets = np.random.permutation(V)
            t0 = time.perf_counter()
            sc_d, steps_d = train_dense(net, targets, V, args.attempts, args.ticks)
            ms_d = (time.perf_counter() - t0) / steps_d * 1000

            # CSR lazy
            np.random.seed(seed)
            pyrandom.seed(seed)
            net2 = SelfWiringGraph(V)
            targets2 = np.random.permutation(V)
            t0 = time.perf_counter()
            sc_c, steps_c = train_csr_lazy(net2, targets2, V, args.attempts, args.ticks)
            ms_c = (time.perf_counter() - t0) / steps_c * 1000

            speedup = ms_d / ms_c if ms_c > 0 else 0
            dense_ms_all.append(ms_d)
            csr_ms_all.append(ms_c)

            print(f"{V:5d} {s:4d} | {sc_d*100:7.1f}% {ms_d:8.2f}ms | "
                  f"{sc_c*100:7.1f}% {ms_c:8.2f}ms {speedup:6.2f}x")
            sys.stdout.flush()

        avg_dense = np.mean(dense_ms_all)
        avg_csr = np.mean(csr_ms_all)
        avg_speedup = avg_dense / avg_csr if avg_csr > 0 else 0
        print(f"  AVG        | {'':>8s} {avg_dense:8.2f}ms | "
              f"{'':>8s} {avg_csr:8.2f}ms {avg_speedup:6.2f}x")
        print()

    print("Notes:")
    print("  dense uses net.forward_batch() — actual NumPy BLAS matmul")
    print("  csr uses C csr_forward + C build_csr_from_mask (rebuild each mutation)")
    print("  lazy: on reject, reuses pre-mutation CSR cache (no rebuild needed)")
    print("  Scores should match (same seed → same mutation sequence)")
