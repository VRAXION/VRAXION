"""
Batch Forward Test -- v22 SelfWiringGraph
==========================================
The profiling showed forward() is 15-49x slower than raw matmul because
of Python loop overhead (8 ticks x 5 numpy calls per tick).

The scoring loop runs 2 passes x V inputs = 2V sequential forward() calls.
For V=64: 128 forward calls, each with 8 ticks = 1024 Python loop iterations.

IDEA: Process all V inputs simultaneously as a (V, N) batch.
One big matmul (V,N)@(N,N) instead of V small matmuls (N,)@(N,N).

COMPLICATION: Currently state persists across inputs (sequential).
Test whether independent (batch) processing gives similar accuracy.

Phase 1: Implement batch forward, verify accuracy vs sequential
Phase 2: Measure speedup
Phase 3: Full training comparison (does batch eval hurt learning?)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
import time
from v22_best_config import SelfWiringGraph, softmax


# ============================================================
#  Batch forward: all V inputs at once
# ============================================================

def forward_batch(net, worlds, ticks=8):
    """
    Batch forward: worlds is (B, V) where B = number of inputs.
    Returns (B, V) logits (charge of first V neurons per input).
    Each input is INDEPENDENT (no cross-input state).
    """
    B = worlds.shape[0]
    N = net.N
    V = net.V

    Weff = net.W * net.mask  # (N, N)

    # Batch state: (B, N)
    charges = np.zeros((B, N), dtype=np.float32)
    acts = np.zeros((B, N), dtype=np.float32)

    for t in range(ticks):
        if t == 0:
            acts[:, :V] = worlds  # inject input (B, V)

        # Batch matmul: (B, N) @ (N, N) = (B, N)
        raw = acts @ Weff + acts * 0.1

        # Capacitor dynamics (element-wise on batch)
        charges += raw * 0.3
        charges *= net.leak
        acts = np.maximum(charges - net.threshold, 0.0)
        charges = np.clip(charges, -net.threshold * 2, net.threshold * 2)

    return charges[:, :V]  # (B, V)


def forward_batch_2pass(net, V, ticks=8):
    """
    2-pass batch evaluation:
    Pass 1: warmup -- all V inputs, average final state as 'priming'
    Pass 2: score -- all V inputs from primed state
    Returns (V, V) logits for pass 2.
    """
    N = net.N
    Weff = net.W * net.mask

    # Build identity-like input matrix
    worlds = np.eye(V, dtype=np.float32)  # (V, V)

    # === PASS 1: warmup (independent per input) ===
    charges_1 = np.zeros((V, N), dtype=np.float32)
    acts_1 = np.zeros((V, N), dtype=np.float32)

    for t in range(ticks):
        if t == 0:
            acts_1[:, :V] = worlds
        raw = acts_1 @ Weff + acts_1 * 0.1
        charges_1 += raw * 0.3
        charges_1 *= net.leak
        acts_1 = np.maximum(charges_1 - net.threshold, 0.0)
        charges_1 = np.clip(charges_1, -net.threshold * 2, net.threshold * 2)

    # === PASS 2: score (start from mean state of pass 1) ===
    mean_charge = charges_1.mean(axis=0)  # (N,) -- averaged priming
    mean_act = np.maximum(mean_charge - net.threshold, 0.0)

    charges_2 = np.tile(mean_charge, (V, 1))  # (V, N)
    acts_2 = np.tile(mean_act, (V, 1))

    for t in range(ticks):
        if t == 0:
            acts_2[:, :V] = worlds
        raw = acts_2 @ Weff + acts_2 * 0.1
        charges_2 += raw * 0.3
        charges_2 *= net.leak
        acts_2 = np.maximum(charges_2 - net.threshold, 0.0)
        charges_2 = np.clip(charges_2, -net.threshold * 2, net.threshold * 2)

    return charges_2[:, :V]  # (V, V)


def forward_batch_nopriming(net, V, ticks=8):
    """
    Single-pass batch: no warmup, just process all V inputs from zero state.
    Simplest possible batch forward.
    """
    Weff = net.W * net.mask
    N = net.N
    worlds = np.eye(V, dtype=np.float32)

    charges = np.zeros((V, N), dtype=np.float32)
    acts = np.zeros((V, N), dtype=np.float32)

    for t in range(ticks):
        if t == 0:
            acts[:, :V] = worlds
        raw = acts @ Weff + acts * 0.1
        charges += raw * 0.3
        charges *= net.leak
        acts = np.maximum(charges - net.threshold, 0.0)
        charges = np.clip(charges, -net.threshold * 2, net.threshold * 2)

    return charges[:, :V]


# ============================================================
#  Sequential scoring (original)
# ============================================================

def score_sequential(net, targets, V, ticks=8):
    """Original sequential scoring: 2 pass x V inputs."""
    net.reset()
    correct = 0
    total_score = 0.0
    for p in range(2):
        for inp in range(V):
            world = np.zeros(V, dtype=np.float32)
            world[inp] = 1.0
            logits = net.forward(world, ticks)
            probs = softmax(logits[:V])
            if p == 1:
                tgt = targets[inp]
                acc_i = 1.0 if np.argmax(probs) == tgt else 0.0
                tp = float(probs[tgt])
                total_score += 0.5 * acc_i + 0.5 * tp
                if acc_i > 0:
                    correct += 1
    return total_score / V, correct / V


# ============================================================
#  Batch scoring variants
# ============================================================

def score_batch_2pass(net, targets, V, ticks=8):
    """Batch scoring with 2-pass (warmup + score)."""
    logits_all = forward_batch_2pass(net, V, ticks)  # (V, V)
    correct = 0
    total_score = 0.0
    for inp in range(V):
        probs = softmax(logits_all[inp])
        tgt = targets[inp]
        acc_i = 1.0 if np.argmax(probs) == tgt else 0.0
        tp = float(probs[tgt])
        total_score += 0.5 * acc_i + 0.5 * tp
        if acc_i > 0:
            correct += 1
    return total_score / V, correct / V


def score_batch_nopriming(net, targets, V, ticks=8):
    """Batch scoring, single pass, no warmup."""
    logits_all = forward_batch_nopriming(net, V, ticks)  # (V, V)
    correct = 0
    total_score = 0.0
    for inp in range(V):
        probs = softmax(logits_all[inp])
        tgt = targets[inp]
        acc_i = 1.0 if np.argmax(probs) == tgt else 0.0
        tp = float(probs[tgt])
        total_score += 0.5 * acc_i + 0.5 * tp
        if acc_i > 0:
            correct += 1
    return total_score / V, correct / V


def score_batch_softmax_vectorized(net, targets, V, ticks=8):
    """Fully vectorized: batch forward + batch softmax + batch scoring."""
    logits_all = forward_batch_nopriming(net, V, ticks)  # (V, V)
    # Vectorized softmax
    e = np.exp(logits_all - logits_all.max(axis=1, keepdims=True))
    probs_all = e / e.sum(axis=1, keepdims=True)  # (V, V)
    # Vectorized scoring
    preds = np.argmax(probs_all, axis=1)  # (V,)
    acc = (preds == targets).mean()
    target_probs = probs_all[np.arange(V), targets]  # (V,)
    score = 0.5 * (preds == targets).astype(float).mean() + 0.5 * target_probs.mean()
    return score, acc


# ============================================================
#  Training with batch scoring
# ============================================================

def train_sequential(V, internal, seed, max_attempts=8000, ticks=8):
    np.random.seed(seed)
    random.seed(seed)
    N = V + internal
    net = SelfWiringGraph(N, V)
    perm = np.random.permutation(V)

    score, acc = score_sequential(net, perm, V, ticks)
    best_acc = acc
    kept = 0
    stale = 0
    phase = "STRUCTURE"
    t0 = time.perf_counter()

    for att in range(max_attempts):
        state = net.save_state()
        if phase == "STRUCTURE":
            net.mutate_structure(0.05)
        else:
            if random.random() < 0.3:
                net.mutate_structure(0.02)
            else:
                net.mutate_weights()

        new_score, new_acc = score_sequential(net, perm, V, ticks)
        if new_score > score:
            score = new_score
            kept += 1
            stale = 0
            best_acc = max(best_acc, new_acc)
        else:
            net.restore_state(state)
            stale += 1

        if phase == "STRUCTURE" and stale > 3000:
            phase = "BOTH"
            stale = 0
        if best_acc >= 0.99:
            break
        if stale >= 6000:
            break

    elapsed = time.perf_counter() - t0
    return best_acc, kept, att + 1, elapsed


def train_batch(V, internal, seed, max_attempts=8000, ticks=8):
    np.random.seed(seed)
    random.seed(seed)
    N = V + internal
    net = SelfWiringGraph(N, V)
    perm = np.random.permutation(V)

    score, acc = score_batch_softmax_vectorized(net, perm, V, ticks)
    best_acc = acc
    kept = 0
    stale = 0
    phase = "STRUCTURE"
    t0 = time.perf_counter()

    for att in range(max_attempts):
        state = net.save_state()
        if phase == "STRUCTURE":
            net.mutate_structure(0.05)
        else:
            if random.random() < 0.3:
                net.mutate_structure(0.02)
            else:
                net.mutate_weights()

        new_score, new_acc = score_batch_softmax_vectorized(net, perm, V, ticks)
        if new_score > score:
            score = new_score
            kept += 1
            stale = 0
            best_acc = max(best_acc, new_acc)
        else:
            net.restore_state(state)
            stale += 1

        if phase == "STRUCTURE" and stale > 3000:
            phase = "BOTH"
            stale = 0
        if best_acc >= 0.99:
            break
        if stale >= 6000:
            break

    elapsed = time.perf_counter() - t0
    return best_acc, kept, att + 1, elapsed


# ============================================================
#  Main
# ============================================================

if __name__ == "__main__":

    # =========================================================
    # PHASE 1: Accuracy comparison (same network, different eval)
    # =========================================================
    print(f"  Batch Forward Test -- v22 SelfWiringGraph")
    print(f"  {'='*55}")

    print(f"\n  PHASE 1: Accuracy comparison (same trained network)")
    print(f"  {'-'*55}")

    for V, internal in [(16, 64), (64, 64)]:
        N = V + internal
        np.random.seed(42)
        random.seed(42)
        net = SelfWiringGraph(N, V)
        perm = np.random.permutation(V)

        # Train a bit with sequential (500 steps)
        score, acc = score_sequential(net, perm, V)
        for att in range(500):
            state = net.save_state()
            net.mutate_structure(0.05)
            new_score, new_acc = score_sequential(net, perm, V)
            if new_score > score:
                score = new_score
            else:
                net.restore_state(state)

        # Now compare all scoring methods on the SAME network
        s_seq, a_seq = score_sequential(net, perm, V)
        s_b2p, a_b2p = score_batch_2pass(net, perm, V)
        s_bnp, a_bnp = score_batch_nopriming(net, perm, V)
        s_bvec, a_bvec = score_batch_softmax_vectorized(net, perm, V)

        print(f"\n  V={V} (N={N}), trained 2K steps:")
        print(f"    sequential:     score={s_seq:.4f}  acc={a_seq*100:5.1f}%")
        print(f"    batch_2pass:    score={s_b2p:.4f}  acc={a_b2p*100:5.1f}%")
        print(f"    batch_nopriming:score={s_bnp:.4f}  acc={a_bnp*100:5.1f}%")
        print(f"    batch_vectorized:score={s_bvec:.4f} acc={a_bvec*100:5.1f}%")

    sys.stdout.flush()

    # =========================================================
    # PHASE 2: Speed comparison
    # =========================================================
    print(f"\n  PHASE 2: Speed comparison")
    print(f"  {'-'*55}")

    for V, internal in [(16, 64), (64, 64), (128, 64)]:
        N = V + internal
        np.random.seed(42)
        net = SelfWiringGraph(N, V)
        perm = np.random.permutation(V)

        # Fill the network (make it ~90% dense for realistic test)
        net.mask = np.where(
            np.random.rand(N, N) > 0.05,
            np.random.choice([-1.0, 1.0], size=(N, N)).astype(np.float32),
            np.float32(0.0))
        np.fill_diagonal(net.mask, 0)

        REPS = max(5, min(50, int(0.5 / max(0.001, V * 0.0001))))

        # Sequential
        t0 = time.perf_counter()
        for _ in range(REPS):
            score_sequential(net, perm, V)
        t_seq = (time.perf_counter() - t0) / REPS

        # Batch 2pass
        t0 = time.perf_counter()
        for _ in range(REPS):
            score_batch_2pass(net, perm, V)
        t_b2p = (time.perf_counter() - t0) / REPS

        # Batch nopriming
        t0 = time.perf_counter()
        for _ in range(REPS):
            score_batch_nopriming(net, perm, V)
        t_bnp = (time.perf_counter() - t0) / REPS

        # Batch vectorized
        t0 = time.perf_counter()
        for _ in range(REPS):
            score_batch_softmax_vectorized(net, perm, V)
        t_bvec = (time.perf_counter() - t0) / REPS

        speedup_2p = t_seq / t_b2p
        speedup_np = t_seq / t_bnp
        speedup_vec = t_seq / t_bvec

        print(f"\n  V={V}, N={N}:")
        print(f"    sequential:      {t_seq*1000:8.2f} ms")
        print(f"    batch_2pass:     {t_b2p*1000:8.2f} ms  ({speedup_2p:.1f}x)")
        print(f"    batch_nopriming: {t_bnp*1000:8.2f} ms  ({speedup_np:.1f}x)")
        print(f"    batch_vectorized:{t_bvec*1000:8.2f} ms  ({speedup_vec:.1f}x)")

    sys.stdout.flush()

    # =========================================================
    # PHASE 3: Training comparison (does batch eval affect learning?)
    # =========================================================
    print(f"\n  PHASE 3: FAIR comparison (same wall-clock time)")
    print(f"  Sequential 2K att vs Batch 18K att (both ~35s on 64-class)")
    print(f"  {'-'*55}")

    V, internal = 64, 64
    N = V + internal
    seed = 42

    # Sequential: 2K attempts
    a_seq, k_seq, s_seq, t_seq = train_sequential(V, internal, seed, 2000)
    print(f"  sequential  2K att: acc={a_seq*100:5.1f}% kept={k_seq} time={t_seq:.1f}s")

    # Batch: use the speedup to run MORE attempts in same time
    # ~9x speedup -> 18K attempts should take ~same time as 2K sequential
    a_bat, k_bat, s_bat, t_bat = train_batch(V, internal, seed, 18000)
    print(f"  batch      18K att: acc={a_bat*100:5.1f}% kept={k_bat} time={t_bat:.1f}s")

    # Also batch 2K for direct comparison
    a_bat2, k_bat2, s_bat2, t_bat2 = train_batch(V, internal, seed, 2000)
    print(f"  batch       2K att: acc={a_bat2*100:5.1f}% kept={k_bat2} time={t_bat2:.1f}s")

    print(f"\n  VERDICT:")
    print(f"    Same time budget (~{t_seq:.0f}s):")
    print(f"      sequential 2K:  {a_seq*100:.1f}%")
    print(f"      batch 18K:      {a_bat*100:.1f}%")
    if a_bat > a_seq:
        print(f"    -> BATCH WINS by {(a_bat-a_seq)*100:.1f}% with {k_bat}x more search")
    elif a_seq > a_bat:
        print(f"    -> SEQUENTIAL WINS by {(a_seq-a_bat)*100:.1f}% (cross-input state matters)")
    else:
        print(f"    -> TIE")

    sys.stdout.flush()

    print(f"\n  {'='*55}")
    print(f"  DONE")
    print(f"  {'='*55}", flush=True)
