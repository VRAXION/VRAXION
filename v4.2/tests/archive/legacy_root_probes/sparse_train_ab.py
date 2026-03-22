"""
Sparse Forward in Training Loop — End-to-End A/B
==================================================
Measures actual training speedup with sparse CSR forward,
including the cost of rebuilding CSR arrays after each mutation.

Key question: does the CSR rebuild cost eat the forward speedup?
At 4% density with ~1-15 edge changes per mutation, incremental
CSR update would be ideal but even full rebuild should be cheap
since it's O(E) vs the forward which is O(E×V×ticks).

Tests:
  A) Dense baseline: train() from graph.py
  B) Sparse CSR: C csr_forward with full CSR rebuild each attempt
  C) Sparse CSR: C csr_forward with incremental CSR update
"""

import sys, os, time
import numpy as np
import random as pyrandom

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model.graph import SelfWiringGraph
from tests.sparse_forward_ab import (
    forward_batch_dense, forward_batch_c_csr,
    build_csr_arrays, _get_c_lib,
)


def score_with_forward(net, targets, V, forward_fn, ticks=8):
    """Score using a custom forward function."""
    logits = forward_fn(net, ticks)
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    acc = (np.argmax(probs, axis=1)[:V] == targets[:V]).mean()
    tp = probs[np.arange(V), targets[:V]].mean()
    return 0.5 * acc + 0.5 * tp


def train_dense(net, targets, V, max_att=4000, ticks=8, stale_limit=3000):
    """Baseline dense training."""
    def fwd(net, ticks):
        return forward_batch_dense(net, ticks)

    score = score_with_forward(net, targets, V, fwd, ticks)
    best = score
    stale = 0

    for att in range(max_att):
        old_loss = int(net.loss_pct)
        old_drive = int(net.drive)
        undo = net.mutate()

        new_score = score_with_forward(net, targets, V, fwd, ticks)
        if new_score > score:
            score = new_score
            best = max(best, score)
            stale = 0
        else:
            net.replay(undo)
            net.loss_pct = np.int8(old_loss)
            net.drive = np.int8(old_drive)
            stale += 1

        if best >= 0.99 or stale >= stale_limit:
            break

    return best, att + 1


def train_sparse_csr(net, targets, V, max_att=4000, ticks=8, stale_limit=3000):
    """Sparse CSR training — rebuild CSR after each accepted mutation."""
    csr_cache = build_csr_arrays(net)

    def fwd(net, ticks):
        return forward_batch_c_csr(net, ticks, csr_cache)

    score = score_with_forward(net, targets, V, fwd, ticks)
    best = score
    stale = 0

    for att in range(max_att):
        old_loss = int(net.loss_pct)
        old_drive = int(net.drive)
        undo = net.mutate()

        # Rebuild CSR (full rebuild — measures worst case)
        csr_cache = build_csr_arrays(net)

        new_score = score_with_forward(net, targets, V, fwd, ticks)
        if new_score > score:
            score = new_score
            best = max(best, score)
            stale = 0
        else:
            net.replay(undo)
            net.loss_pct = np.int8(old_loss)
            net.drive = np.int8(old_drive)
            # Rebuild CSR after undo too
            csr_cache = build_csr_arrays(net)
            stale += 1

        if best >= 0.99 or stale >= stale_limit:
            break

    return best, att + 1


def train_sparse_csr_lazy(net, targets, V, max_att=4000, ticks=8, stale_limit=3000):
    """Sparse CSR training — only rebuild on accept, reuse old on reject.
    On reject we replay(undo) which restores the mask exactly,
    so the old CSR cache is still valid."""
    csr_cache = build_csr_arrays(net)

    def fwd(net, ticks):
        return forward_batch_c_csr(net, ticks, csr_cache)

    score = score_with_forward(net, targets, V, fwd, ticks)
    best = score
    stale = 0

    for att in range(max_att):
        old_loss = int(net.loss_pct)
        old_drive = int(net.drive)
        old_csr = csr_cache  # save reference

        undo = net.mutate()
        # Rebuild for the mutated state
        csr_cache = build_csr_arrays(net)

        new_score = score_with_forward(net, targets, V, fwd, ticks)
        if new_score > score:
            score = new_score
            best = max(best, score)
            stale = 0
            # csr_cache already points to new state, keep it
        else:
            net.replay(undo)
            net.loss_pct = np.int8(old_loss)
            net.drive = np.int8(old_drive)
            csr_cache = old_csr  # reuse old cache, no rebuild!
            stale += 1

        if best >= 0.99 or stale >= stale_limit:
            break

    return best, att + 1


# ── Main ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab", type=str, default="32,64,128")
    parser.add_argument("--attempts", type=int, default=4000)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--ticks", type=int, default=8)
    args = parser.parse_args()

    vocab_list = [int(v) for v in args.vocab.split(",")]

    print("Compiling C sparse library...")
    lib = _get_c_lib()
    if not lib:
        print("FAILED — C library required")
        sys.exit(1)
    print("OK")
    print()

    print("=" * 85)
    print("SPARSE CSR IN TRAINING LOOP — END-TO-END A/B")
    print("=" * 85)
    print(f"Attempts={args.attempts}, Seeds={args.seeds}, Ticks={args.ticks}")
    print()

    header = (f"{'V':>5} {'seed':>4} | {'dense_sc':>8} {'dense_ms':>9} | "
              f"{'csr_sc':>8} {'csr_ms':>9} {'csr_x':>6} | "
              f"{'lazy_sc':>8} {'lazy_ms':>9} {'lazy_x':>6}")
    print(header)
    print("-" * len(header))

    for V in vocab_list:
        for s in range(args.seeds):
            seed = 42 + s

            # Dense baseline
            np.random.seed(seed)
            pyrandom.seed(seed)
            net_d = SelfWiringGraph(V)
            targets = np.random.permutation(V)
            t0 = time.perf_counter()
            sc_d, steps_d = train_dense(net_d, targets, V, args.attempts, args.ticks)
            ms_d = (time.perf_counter() - t0) / steps_d * 1000

            # Sparse CSR (full rebuild)
            np.random.seed(seed)
            pyrandom.seed(seed)
            net_s = SelfWiringGraph(V)
            targets_s = np.random.permutation(V)
            t0 = time.perf_counter()
            sc_s, steps_s = train_sparse_csr(net_s, targets_s, V, args.attempts, args.ticks)
            ms_s = (time.perf_counter() - t0) / steps_s * 1000

            # Sparse CSR lazy
            np.random.seed(seed)
            pyrandom.seed(seed)
            net_l = SelfWiringGraph(V)
            targets_l = np.random.permutation(V)
            t0 = time.perf_counter()
            sc_l, steps_l = train_sparse_csr_lazy(net_l, targets_l, V, args.attempts, args.ticks)
            ms_l = (time.perf_counter() - t0) / steps_l * 1000

            speedup_s = ms_d / ms_s if ms_s > 0 else 0
            speedup_l = ms_d / ms_l if ms_l > 0 else 0

            print(f"{V:5d} {s:4d} | {sc_d*100:7.1f}% {ms_d:8.2f}ms | "
                  f"{sc_s*100:7.1f}% {ms_s:8.2f}ms {speedup_s:5.2f}x | "
                  f"{sc_l*100:7.1f}% {ms_l:8.2f}ms {speedup_l:5.2f}x")
            sys.stdout.flush()

        print()

    print("Notes:")
    print("  dense_ms/csr_ms/lazy_ms = median ms per training attempt")
    print("  csr: full CSR rebuild after every mutation (accept + reject)")
    print("  lazy: only rebuild on mutate, reuse old CSR on reject")
    print("  Scores should be identical (same seed, same mutation sequence)")
