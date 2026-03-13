"""
Code Optimization Test — v22 SelfWiringGraph
==============================================
Pure code-level speedups. NO logic changes.
Same mutation, same forward, same scoring — just faster.

Optimizations tested:
  1. Minimal save/restore (only mask+W, not all 6 arrays)
  2. Cached Weff (W*mask computed once after mutation, not per forward)
  3. Pre-allocated buffers (reuse charge/acts arrays)
  4. Vectorized mutation (numpy batch ops instead of Python loops)
  5. ALL COMBINED

Measured on V=64 (N=128) and V=128 (N=256) with 2K attempts.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
import time
from v22_best_config import SelfWiringGraph, softmax


# ============================================================
#  BASELINE: current code (from batch_forward_test.py)
# ============================================================

def forward_batch_baseline(net, V, ticks=8):
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


def score_baseline(net, targets, V, ticks=8):
    logits_all = forward_batch_baseline(net, V, ticks)
    e = np.exp(logits_all - logits_all.max(axis=1, keepdims=True))
    probs_all = e / e.sum(axis=1, keepdims=True)
    preds = np.argmax(probs_all, axis=1)
    acc = (preds == targets).mean()
    target_probs = probs_all[np.arange(V), targets]
    score = 0.5 * (preds == targets).astype(float).mean() + 0.5 * target_probs.mean()
    return score, acc


def train_baseline(V, internal, seed, max_att=2000, ticks=8):
    np.random.seed(seed)
    random.seed(seed)
    N = V + internal
    net = SelfWiringGraph(N, V)
    perm = np.random.permutation(V)

    score, acc = score_baseline(net, perm, V, ticks)
    best_acc = acc
    kept = 0
    stale = 0
    phase = "STRUCTURE"
    t0 = time.perf_counter()

    for att in range(max_att):
        state = net.save_state()  # 6 array copies
        if phase == "STRUCTURE":
            net.mutate_structure(0.05)
        else:
            if random.random() < 0.3:
                net.mutate_structure(0.02)
            else:
                net.mutate_weights()

        new_score, new_acc = score_baseline(net, perm, V, ticks)
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
    return best_acc, kept, elapsed


# ============================================================
#  OPT 1: Minimal save/restore (only mask + W)
# ============================================================

def train_opt1_minimal_save(V, internal, seed, max_att=2000, ticks=8):
    np.random.seed(seed)
    random.seed(seed)
    N = V + internal
    net = SelfWiringGraph(N, V)
    perm = np.random.permutation(V)

    score, acc = score_baseline(net, perm, V, ticks)
    best_acc = acc
    kept = 0
    stale = 0
    phase = "STRUCTURE"
    t0 = time.perf_counter()

    for att in range(max_att):
        # Only save what mutation touches
        saved_mask = net.mask.copy()
        saved_W = net.W.copy()

        if phase == "STRUCTURE":
            net.mutate_structure(0.05)
        else:
            if random.random() < 0.3:
                net.mutate_structure(0.02)
            else:
                net.mutate_weights()

        new_score, new_acc = score_baseline(net, perm, V, ticks)
        if new_score > score:
            score = new_score
            kept += 1
            stale = 0
            best_acc = max(best_acc, new_acc)
        else:
            # Only restore what changed
            net.mask = saved_mask
            net.W = saved_W

            stale += 1

        if phase == "STRUCTURE" and stale > 3000:
            phase = "BOTH"
            stale = 0
        if best_acc >= 0.99:
            break
        if stale >= 6000:
            break

    elapsed = time.perf_counter() - t0
    return best_acc, kept, elapsed


# ============================================================
#  OPT 2: Cached Weff
# ============================================================

def forward_cached_weff(Weff, V, N, leak, threshold, ticks=8):
    """Forward with pre-computed Weff."""
    worlds = np.eye(V, dtype=np.float32)
    charges = np.zeros((V, N), dtype=np.float32)
    acts = np.zeros((V, N), dtype=np.float32)
    for t in range(ticks):
        if t == 0:
            acts[:, :V] = worlds
        raw = acts @ Weff + acts * 0.1
        charges += raw * 0.3
        charges *= leak
        acts = np.maximum(charges - threshold, 0.0)
        charges = np.clip(charges, -threshold * 2, threshold * 2)
    return charges[:, :V]


def score_cached(Weff, targets, V, N, leak, threshold, ticks=8):
    logits_all = forward_cached_weff(Weff, V, N, leak, threshold, ticks)
    e = np.exp(logits_all - logits_all.max(axis=1, keepdims=True))
    probs_all = e / e.sum(axis=1, keepdims=True)
    preds = np.argmax(probs_all, axis=1)
    acc = (preds == targets).mean()
    target_probs = probs_all[np.arange(V), targets]
    score = 0.5 * (preds == targets).astype(float).mean() + 0.5 * target_probs.mean()
    return score, acc


def train_opt2_cached_weff(V, internal, seed, max_att=2000, ticks=8):
    np.random.seed(seed)
    random.seed(seed)
    N = V + internal
    net = SelfWiringGraph(N, V)
    perm = np.random.permutation(V)

    Weff = net.W * net.mask
    score, acc = score_cached(Weff, perm, V, N, net.leak, net.threshold, ticks)
    best_acc = acc
    kept = 0
    stale = 0
    phase = "STRUCTURE"
    t0 = time.perf_counter()

    for att in range(max_att):
        saved_mask = net.mask.copy()
        saved_W = net.W.copy()

        if phase == "STRUCTURE":
            net.mutate_structure(0.05)
        else:
            if random.random() < 0.3:
                net.mutate_structure(0.02)
            else:
                net.mutate_weights()

        # Compute Weff ONCE after mutation
        Weff = net.W * net.mask

        new_score, new_acc = score_cached(Weff, perm, V, N, net.leak,
                                           net.threshold, ticks)
        if new_score > score:
            score = new_score
            kept += 1
            stale = 0
            best_acc = max(best_acc, new_acc)
        else:
            net.mask = saved_mask
            net.W = saved_W
            Weff = saved_W * saved_mask  # restore cached Weff
            stale += 1

        if phase == "STRUCTURE" and stale > 3000:
            phase = "BOTH"
            stale = 0
        if best_acc >= 0.99:
            break
        if stale >= 6000:
            break

    elapsed = time.perf_counter() - t0
    return best_acc, kept, elapsed


# ============================================================
#  OPT 3: Pre-allocated buffers
# ============================================================

def forward_prealloc(Weff, V, N, leak, threshold, charges_buf, acts_buf,
                     worlds, raw_buf, ticks=8):
    """Forward reusing pre-allocated buffers."""
    charges_buf[:] = 0
    acts_buf[:] = 0
    for t in range(ticks):
        if t == 0:
            acts_buf[:, :V] = worlds
        np.dot(acts_buf, Weff, out=raw_buf)
        raw_buf += acts_buf * 0.1
        charges_buf += raw_buf * 0.3
        charges_buf *= leak
        np.maximum(charges_buf - threshold, 0.0, out=acts_buf)
        np.clip(charges_buf, -threshold * 2, threshold * 2, out=charges_buf)
    return charges_buf[:, :V].copy()


def train_opt3_prealloc(V, internal, seed, max_att=2000, ticks=8):
    np.random.seed(seed)
    random.seed(seed)
    N = V + internal
    net = SelfWiringGraph(N, V)
    perm = np.random.permutation(V)

    # Pre-allocate
    charges_buf = np.zeros((V, N), dtype=np.float32)
    acts_buf = np.zeros((V, N), dtype=np.float32)
    raw_buf = np.zeros((V, N), dtype=np.float32)
    worlds = np.eye(V, dtype=np.float32)

    Weff = net.W * net.mask
    logits = forward_prealloc(Weff, V, N, net.leak, net.threshold,
                               charges_buf, acts_buf, worlds, raw_buf, ticks)
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs_all = e / e.sum(axis=1, keepdims=True)
    preds = np.argmax(probs_all, axis=1)
    acc = (preds == perm).mean()
    target_probs = probs_all[np.arange(V), perm]
    score = 0.5 * (preds == perm).astype(float).mean() + 0.5 * target_probs.mean()

    best_acc = acc
    kept = 0
    stale = 0
    phase = "STRUCTURE"
    t0 = time.perf_counter()

    for att in range(max_att):
        saved_mask = net.mask.copy()
        saved_W = net.W.copy()

        if phase == "STRUCTURE":
            net.mutate_structure(0.05)
        else:
            if random.random() < 0.3:
                net.mutate_structure(0.02)
            else:
                net.mutate_weights()

        Weff = net.W * net.mask
        logits = forward_prealloc(Weff, V, N, net.leak, net.threshold,
                                   charges_buf, acts_buf, worlds, raw_buf, ticks)
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs_all = e / e.sum(axis=1, keepdims=True)
        preds = np.argmax(probs_all, axis=1)
        new_acc = (preds == perm).mean()
        target_probs = probs_all[np.arange(V), perm]
        new_score = 0.5 * (preds == perm).astype(float).mean() + 0.5 * target_probs.mean()

        if new_score > score:
            score = new_score
            kept += 1
            stale = 0
            best_acc = max(best_acc, new_acc)
        else:
            net.mask = saved_mask
            net.W = saved_W
            Weff = saved_W * saved_mask
            stale += 1

        if phase == "STRUCTURE" and stale > 3000:
            phase = "BOTH"
            stale = 0
        if best_acc >= 0.99:
            break
        if stale >= 6000:
            break

    elapsed = time.perf_counter() - t0
    return best_acc, kept, elapsed


# ============================================================
#  OPT 4: Vectorized mutation
# ============================================================

def mutate_structure_vectorized(net, rate=0.05):
    """Same logic as net.mutate_structure but with numpy batch ops."""
    r = random.random()
    N = net.N

    if r < net.flip_rate:
        # FLIP: toggle sign of existing connections
        alive = np.argwhere(net.mask != 0)
        if len(alive) > 0:
            n = max(1, int(len(alive) * rate * 0.5))
            idx = alive[np.random.choice(len(alive), min(n, len(alive)),
                                          replace=False)]
            # Vectorized flip (no Python loop)
            rows, cols = idx[:, 0], idx[:, 1]
            net.mask[rows, cols] *= -1
    else:
        action = random.choice(['add_pos', 'add_neg', 'remove', 'rewire'])

        if action in ('add_pos', 'add_neg'):
            dead = np.argwhere(net.mask == 0)
            dead = dead[dead[:, 0] != dead[:, 1]]
            if len(dead) > 0:
                n = max(1, int(len(dead) * rate))
                idx = dead[np.random.choice(len(dead), min(n, len(dead)),
                                             replace=False)]
                rows, cols = idx[:, 0], idx[:, 1]
                sign = 1.0 if action == 'add_pos' else -1.0
                net.mask[rows, cols] = sign
                # Vectorized weight assignment
                net.W[rows, cols] = np.where(
                    np.random.rand(len(rows)) > 0.5,
                    np.float32(0.5), np.float32(1.5))

        elif action == 'remove':
            alive = np.argwhere(net.mask != 0)
            if len(alive) > 3:
                n = max(1, int(len(alive) * rate))
                idx = alive[np.random.choice(len(alive), min(n, len(alive)),
                                              replace=False)]
                rows, cols = idx[:, 0], idx[:, 1]
                net.mask[rows, cols] = 0

        else:  # rewire
            alive = np.argwhere(net.mask != 0)
            if len(alive) > 0:
                n = max(1, int(len(alive) * rate))
                idx = alive[np.random.choice(len(alive), min(n, len(alive)),
                                              replace=False)]
                for j in range(len(idx)):
                    r2, c = int(idx[j][0]), int(idx[j][1])
                    old_sign = net.mask[r2, c]
                    old_w = net.W[r2, c]
                    net.mask[r2, c] = 0
                    nc = random.randint(0, N - 1)
                    while nc == r2:
                        nc = random.randint(0, N - 1)
                    net.mask[r2, nc] = old_sign
                    net.W[r2, nc] = old_w


def mutate_weights_vectorized(net):
    """Same logic as net.mutate_weights but vectorized."""
    alive = np.argwhere(net.mask != 0)
    if len(alive) > 0:
        n = max(1, int(len(alive) * 0.05))
        idx = alive[np.random.choice(len(alive), min(n, len(alive)),
                                      replace=False)]
        rows, cols = idx[:, 0], idx[:, 1]
        # Vectorized toggle
        net.W[rows, cols] = np.where(
            net.W[rows, cols] < 1.0,
            np.float32(1.5), np.float32(0.5))


def train_opt4_vec_mutation(V, internal, seed, max_att=2000, ticks=8):
    np.random.seed(seed)
    random.seed(seed)
    N = V + internal
    net = SelfWiringGraph(N, V)
    perm = np.random.permutation(V)

    score, acc = score_baseline(net, perm, V, ticks)
    best_acc = acc
    kept = 0
    stale = 0
    phase = "STRUCTURE"
    t0 = time.perf_counter()

    for att in range(max_att):
        saved_mask = net.mask.copy()
        saved_W = net.W.copy()

        if phase == "STRUCTURE":
            mutate_structure_vectorized(net, 0.05)
        else:
            if random.random() < 0.3:
                mutate_structure_vectorized(net, 0.02)
            else:
                mutate_weights_vectorized(net)

        new_score, new_acc = score_baseline(net, perm, V, ticks)
        if new_score > score:
            score = new_score
            kept += 1
            stale = 0
            best_acc = max(best_acc, new_acc)
        else:
            net.mask = saved_mask
            net.W = saved_W
            stale += 1

        if phase == "STRUCTURE" and stale > 3000:
            phase = "BOTH"
            stale = 0
        if best_acc >= 0.99:
            break
        if stale >= 6000:
            break

    elapsed = time.perf_counter() - t0
    return best_acc, kept, elapsed


# ============================================================
#  OPT 5: ALL COMBINED
# ============================================================

def train_opt5_all(V, internal, seed, max_att=2000, ticks=8):
    np.random.seed(seed)
    random.seed(seed)
    N = V + internal
    net = SelfWiringGraph(N, V)
    perm = np.random.permutation(V)

    # Pre-allocate
    charges_buf = np.zeros((V, N), dtype=np.float32)
    acts_buf = np.zeros((V, N), dtype=np.float32)
    raw_buf = np.zeros((V, N), dtype=np.float32)
    worlds = np.eye(V, dtype=np.float32)
    leak = net.leak
    threshold = net.threshold

    Weff = net.W * net.mask
    logits = forward_prealloc(Weff, V, N, leak, threshold,
                               charges_buf, acts_buf, worlds, raw_buf, ticks)
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs_all = e / e.sum(axis=1, keepdims=True)
    preds = np.argmax(probs_all, axis=1)
    acc = (preds == perm).mean()
    target_probs = probs_all[np.arange(V), perm]
    score = 0.5 * (preds == perm).astype(float).mean() + 0.5 * target_probs.mean()

    best_acc = acc
    kept = 0
    stale = 0
    phase = "STRUCTURE"
    t0 = time.perf_counter()

    for att in range(max_att):
        # OPT 1: minimal save (only mask + W)
        saved_mask = net.mask.copy()
        saved_W = net.W.copy()

        # OPT 4: vectorized mutation
        if phase == "STRUCTURE":
            mutate_structure_vectorized(net, 0.05)
        else:
            if random.random() < 0.3:
                mutate_structure_vectorized(net, 0.02)
            else:
                mutate_weights_vectorized(net)

        # OPT 2: cached Weff
        Weff = net.W * net.mask

        # OPT 3: pre-allocated buffers
        logits = forward_prealloc(Weff, V, N, leak, threshold,
                                   charges_buf, acts_buf, worlds, raw_buf, ticks)
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs_all = e / e.sum(axis=1, keepdims=True)
        preds = np.argmax(probs_all, axis=1)
        new_acc = (preds == perm).mean()
        target_probs = probs_all[np.arange(V), perm]
        new_score = 0.5 * (preds == perm).astype(float).mean() + 0.5 * target_probs.mean()

        if new_score > score:
            score = new_score
            kept += 1
            stale = 0
            best_acc = max(best_acc, new_acc)
        else:
            net.mask = saved_mask
            net.W = saved_W
            Weff = saved_W * saved_mask
            stale += 1

        if phase == "STRUCTURE" and stale > 3000:
            phase = "BOTH"
            stale = 0
        if best_acc >= 0.99:
            break
        if stale >= 6000:
            break

    elapsed = time.perf_counter() - t0
    return best_acc, kept, elapsed


# ============================================================
#  Main
# ============================================================

if __name__ == "__main__":
    print(f"  Code Optimization Test — v22 SelfWiringGraph")
    print(f"  {'='*60}")
    print(f"  Same logic, faster code. Measuring wall-clock speedup.")
    print()

    for V, internal in [(64, 64), (128, 128)]:
        N = V + internal
        seed = 42
        max_att = 2000

        print(f"\n  --- V={V}, N={N}, {max_att} attempts ---")
        print(f"  {'-'*50}")

        # Baseline
        a0, k0, t0 = train_baseline(V, internal, seed, max_att)
        print(f"  baseline:           acc={a0*100:5.1f}% kept={k0:3d} time={t0:.2f}s")
        sys.stdout.flush()

        # Opt 1: minimal save
        a1, k1, t1 = train_opt1_minimal_save(V, internal, seed, max_att)
        print(f"  opt1_minimal_save:  acc={a1*100:5.1f}% kept={k1:3d} "
              f"time={t1:.2f}s ({t0/t1:.2f}x)")
        sys.stdout.flush()

        # Opt 2: cached Weff
        a2, k2, t2 = train_opt2_cached_weff(V, internal, seed, max_att)
        print(f"  opt2_cached_weff:   acc={a2*100:5.1f}% kept={k2:3d} "
              f"time={t2:.2f}s ({t0/t2:.2f}x)")
        sys.stdout.flush()

        # Opt 3: pre-alloc buffers
        a3, k3, t3 = train_opt3_prealloc(V, internal, seed, max_att)
        print(f"  opt3_prealloc:      acc={a3*100:5.1f}% kept={k3:3d} "
              f"time={t3:.2f}s ({t0/t3:.2f}x)")
        sys.stdout.flush()

        # Opt 4: vectorized mutation
        a4, k4, t4 = train_opt4_vec_mutation(V, internal, seed, max_att)
        print(f"  opt4_vec_mutation:  acc={a4*100:5.1f}% kept={k4:3d} "
              f"time={t4:.2f}s ({t0/t4:.2f}x)")
        sys.stdout.flush()

        # Opt 5: ALL combined
        a5, k5, t5 = train_opt5_all(V, internal, seed, max_att)
        print(f"  opt5_ALL_COMBINED:  acc={a5*100:5.1f}% kept={k5:3d} "
              f"time={t5:.2f}s ({t0/t5:.2f}x)")
        sys.stdout.flush()

        # Summary
        print(f"\n  SPEEDUP SUMMARY (V={V}):")
        opts = [
            ('minimal_save', t1),
            ('cached_weff', t2),
            ('prealloc', t3),
            ('vec_mutation', t4),
            ('ALL_COMBINED', t5),
        ]
        for name, t in sorted(opts, key=lambda x: x[1]):
            bar = '#' * int(t0 / t * 20)
            print(f"    {name:<18s} {t0/t:.2f}x  {bar}")

    print(f"\n  {'='*60}")
    print(f"  DONE")
    print(f"  {'='*60}", flush=True)
