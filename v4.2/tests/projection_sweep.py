"""
Projection Method Sweep for PassiveIO
=======================================
Sweeps W_in / W_out projection strategies to find the best one.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random as pyrandom
from model.passive_io import PassiveIOGraph, train_passive

V = 27
H = V * 3  # 81
BUDGET = 10_000
TICKS = 8
STALE = 8_000
SEEDS = [42, 123, 777, 999, 314]


def make_net_with_proj(seed, w_in, w_out):
    """Create PassiveIOGraph but override W_in/W_out."""
    np.random.seed(seed)
    pyrandom.seed(seed)
    net = PassiveIOGraph(V, h_ratio=3, proj='random')
    # Override projections
    net.W_in = w_in.astype(np.float32).copy()
    net.W_out = w_out.astype(np.float32).copy()
    return net


def proj_random(seed):
    """Two independent random unit-norm projections."""
    rng = np.random.RandomState(seed + 10000)
    W_in = rng.randn(V, H)
    W_in /= np.linalg.norm(W_in, axis=1, keepdims=True)
    W_out = rng.randn(H, V)
    W_out /= np.linalg.norm(W_out, axis=0, keepdims=True)
    return W_in, W_out


def proj_ortho(seed):
    """QR-orthogonalized projections."""
    rng = np.random.RandomState(seed + 10000)
    A = rng.randn(H, V)
    Q, _ = np.linalg.qr(A)
    W_in = Q[:, :V].T  # V × H
    B = rng.randn(H, V)
    Q2, _ = np.linalg.qr(B)
    W_out = Q2[:, :V]  # H × V
    return W_in, W_out


def proj_phi_shift(seed):
    """Phi-based (golden ratio) phase-shifted sinusoidal projections."""
    phi = (1 + np.sqrt(5)) / 2
    phases_in = np.arange(V).reshape(-1, 1) * phi  # V × 1
    freqs = np.arange(H).reshape(1, -1) / H        # 1 × H
    W_in = np.sin(2 * np.pi * (phases_in + freqs))
    W_in /= np.linalg.norm(W_in, axis=1, keepdims=True)

    phases_out = np.arange(V).reshape(1, -1) * phi * 1.3  # offset
    freqs_out = np.arange(H).reshape(-1, 1) / H
    W_out = np.cos(2 * np.pi * (phases_out + freqs_out))
    W_out /= np.linalg.norm(W_out, axis=0, keepdims=True)
    return W_in, W_out


def proj_sparse(seed, density=0.3):
    """Sparse random: only 30% of entries nonzero."""
    rng = np.random.RandomState(seed + 10000)
    W_in = rng.randn(V, H)
    mask_in = rng.rand(V, H) < density
    W_in *= mask_in
    norms = np.linalg.norm(W_in, axis=1, keepdims=True)
    norms[norms == 0] = 1
    W_in /= norms

    W_out = rng.randn(H, V)
    mask_out = rng.rand(H, V) < density
    W_out *= mask_out
    norms2 = np.linalg.norm(W_out, axis=0, keepdims=True)
    norms2[norms2 == 0] = 1
    W_out /= norms2
    return W_in, W_out


def proj_scaled_random(seed, scale=2.0):
    """Random but with higher amplitude — stronger injection."""
    W_in, W_out = proj_random(seed)
    return W_in * scale, W_out * scale


def proj_hadamard_asym(seed):
    """Hadamard-like W_in, random W_out — asymmetric."""
    rng = np.random.RandomState(seed + 10000)
    A = rng.randn(H, V)
    Q, _ = np.linalg.qr(A)
    W_in = Q[:, :V].T  # V × H, orthogonal

    W_out = rng.randn(H, V)
    W_out /= np.linalg.norm(W_out, axis=0, keepdims=True)
    return W_in, W_out


def proj_binary(seed):
    """Binary ±1 random projection (like LSH)."""
    rng = np.random.RandomState(seed + 10000)
    W_in = rng.choice([-1.0, 1.0], size=(V, H)) / np.sqrt(H)
    W_out = rng.choice([-1.0, 1.0], size=(H, V)) / np.sqrt(V)
    return W_in, W_out


def proj_identity_padded(seed):
    """Identity for first V neurons, zero rest. Asymmetric output."""
    rng = np.random.RandomState(seed + 10000)
    W_in = np.zeros((V, H), dtype=np.float32)
    W_in[np.arange(V), np.arange(V)] = 1.0

    W_out = rng.randn(H, V)
    W_out /= np.linalg.norm(W_out, axis=0, keepdims=True)
    return W_in, W_out


def proj_phi_lattice(seed):
    """Golden ratio lattice — each input maps to phi-spaced hidden neurons."""
    phi = (1 + np.sqrt(5)) / 2
    W_in = np.zeros((V, H), dtype=np.float32)
    for v in range(V):
        for h in range(H):
            angle = 2 * np.pi * ((v * phi + h * phi**2) % 1.0)
            W_in[v, h] = np.cos(angle)
    norms = np.linalg.norm(W_in, axis=1, keepdims=True)
    norms[norms == 0] = 1
    W_in /= norms

    W_out = np.zeros((H, V), dtype=np.float32)
    for h in range(H):
        for v in range(V):
            angle = 2 * np.pi * ((h * phi * 1.5 + v * phi**2 * 0.7) % 1.0)
            W_out[h, v] = np.sin(angle)
    norms2 = np.linalg.norm(W_out, axis=0, keepdims=True)
    norms2[norms2 == 0] = 1
    W_out /= norms2
    return W_in, W_out


def proj_sparse10(seed):
    return proj_sparse(seed, density=0.1)


def proj_scaled3(seed):
    return proj_scaled_random(seed, scale=3.0)


def proj_scaled05(seed):
    return proj_scaled_random(seed, scale=0.5)


# ── All projection configs ──
PROJECTIONS = {
    'random':          proj_random,
    'ortho':           proj_ortho,
    'phi-shift':       proj_phi_shift,
    'phi-lattice':     proj_phi_lattice,
    'sparse-30%':      proj_sparse,
    'sparse-10%':      proj_sparse10,
    'scaled-2x':       proj_scaled_random,
    'scaled-3x':       proj_scaled3,
    'scaled-0.5x':     proj_scaled05,
    'hadamard-asym':   proj_hadamard_asym,
    'binary-LSH':      proj_binary,
    'identity+rndOut': proj_identity_padded,
}


# ══════════════════════════════════════════════════════
print(f"{'='*80}")
print(f"  PROJECTION SWEEP — {len(PROJECTIONS)} methods × {len(SEEDS)} seeds")
print(f"  V={V}, H={H}, Budget={BUDGET}, Ticks={TICKS}")
print(f"{'='*80}\n")

results = {}
for name, proj_fn in PROJECTIONS.items():
    scores = []
    conns_list = []
    t0 = time.time()
    for seed in SEEDS:
        W_in, W_out = proj_fn(seed)
        net = make_net_with_proj(seed, W_in, W_out)
        targets = np.random.permutation(V)
        score = train_passive(net, targets, V, max_attempts=BUDGET,
                              ticks=TICKS, stale_limit=STALE, verbose=False)
        scores.append(score)
        conns_list.append(net.count_connections())
    dt = time.time() - t0
    avg = np.mean(scores)
    std = np.std(scores)
    results[name] = (avg, std, scores, np.mean(conns_list), dt)
    print(f"  {name:<18s}  avg={avg*100:5.1f}% ±{std*100:4.1f}  "
          f"conns={np.mean(conns_list):5.0f}  [{', '.join(f'{s*100:.0f}' for s in scores)}]  "
          f"({dt:.1f}s)")

# ── Sorted ranking ──
print(f"\n{'='*80}")
print(f"  RANKING (by avg score)")
print(f"{'='*80}")
print(f"  {'#':>2s}  {'Method':<18s}  {'Avg':>6s}  {'±':>5s}  {'Conns':>6s}  {'Best':>5s}  {'Worst':>5s}")
print(f"  {'─'*65}")
ranked = sorted(results.items(), key=lambda x: -x[1][0])
for rank, (name, (avg, std, scores, conns, dt)) in enumerate(ranked):
    print(f"  {rank+1:2d}  {name:<18s}  {avg*100:5.1f}%  ±{std*100:4.1f}  "
          f"{conns:6.0f}  {max(scores)*100:4.0f}%  {min(scores)*100:4.0f}%")

print(f"\n{'='*80}")
print("DONE")
