"""
Input Geometry Sweep
=====================
Does the hidden layer adapt to ANY input geometry?
Tests: random, concentric rings, 2D grid walls, hypercube, simplex,
spiral, clustered, orthogonal basis, etc.

All scaled to INJ_SCALE=3.0. Same mask init, same training.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random as pyrandom
from model.graph import SelfWiringGraph, train

V = 27
H = V * 3  # 81
BUDGET = 10_000
TICKS = 8
STALE = 8_000
SEEDS = [42, 123, 777, 999, 314]
SCALE = 3.0


def make_net_with_geometry(seed, w_in, w_out):
    np.random.seed(seed)
    pyrandom.seed(seed)
    net = SelfWiringGraph(V)
    net.W_in = w_in.astype(np.float32).copy()
    net.W_out = w_out.astype(np.float32).copy()
    return net


def norm_rows(W):
    n = np.linalg.norm(W, axis=1, keepdims=True)
    n[n == 0] = 1
    return W / n * SCALE


def norm_cols(W):
    n = np.linalg.norm(W, axis=0, keepdims=True)
    n[n == 0] = 1
    return W / n * SCALE


# ══════════════════════════════════════════════════════
#  GEOMETRY GENERATORS
# ══════════════════════════════════════════════════════

def geom_random(seed):
    """Baseline: random unit-norm projections."""
    rng = np.random.RandomState(seed + 10000)
    W_in = rng.randn(V, H)
    W_out = rng.randn(H, V)
    return norm_rows(W_in), norm_cols(W_out)


def geom_concentric_rings(seed):
    """Each input maps to a ring of hidden neurons at different radii."""
    W_in = np.zeros((V, H))
    for v in range(V):
        radius = (v + 1) / V
        for h in range(H):
            angle = 2 * np.pi * h / H
            # Gaussian bump centered on ring
            dist = abs(np.sqrt((np.cos(angle) * radius)**2 +
                              (np.sin(angle) * radius)**2) - radius)
            W_in[v, h] = np.exp(-dist * 5) * np.sin(angle * (v + 1))
    rng = np.random.RandomState(seed + 10000)
    W_out = rng.randn(H, V)
    return norm_rows(W_in), norm_cols(W_out)


def geom_2d_grid_walls(seed):
    """Hidden neurons on a 9×9 grid. Each input activates a vertical 'wall'."""
    side = 9  # 9×9 = 81
    W_in = np.zeros((V, H))
    for v in range(V):
        # Input v activates column (v % side) with some spread
        col = v % side
        for row in range(side):
            h = row * side + col
            W_in[v, h] = 1.0
            # Add neighbors with decay
            for dc in [-1, 1]:
                nc = col + dc
                if 0 <= nc < side:
                    nh = row * side + nc
                    W_in[v, nh] = 0.5
    rng = np.random.RandomState(seed + 10000)
    W_out = rng.randn(H, V)
    return norm_rows(W_in), norm_cols(W_out)


def geom_2d_grid_horizontal(seed):
    """Like walls but horizontal bands instead of vertical."""
    side = 9
    W_in = np.zeros((V, H))
    for v in range(V):
        row = v % side
        for col in range(side):
            h = row * side + col
            W_in[v, h] = 1.0
            for dr in [-1, 1]:
                nr = row + dr
                if 0 <= nr < side:
                    nh = nr * side + col
                    W_in[v, nh] = 0.5
    rng = np.random.RandomState(seed + 10000)
    W_out = rng.randn(H, V)
    return norm_rows(W_in), norm_cols(W_out)


def geom_hypercube(seed):
    """Project inputs onto vertices of a high-dimensional hypercube."""
    rng = np.random.RandomState(seed + 10000)
    # Random binary vectors as hypercube vertices
    bits = rng.randint(0, 2, size=(V, H)).astype(np.float32)
    bits = bits * 2 - 1  # {-1, +1}
    W_out = rng.randn(H, V)
    return norm_rows(bits), norm_cols(W_out)


def geom_simplex(seed):
    """Each input is a vertex of a V-simplex embedded in H dimensions."""
    rng = np.random.RandomState(seed + 10000)
    # Random orthogonal frame, then place simplex vertices
    A = rng.randn(H, V)
    Q, _ = np.linalg.qr(A)
    # Simplex vertices: centroid at origin
    verts = Q[:, :V].T  # V × H
    centroid = verts.mean(axis=0)
    verts -= centroid
    W_out = rng.randn(H, V)
    return norm_rows(verts), norm_cols(W_out)


def geom_spiral(seed):
    """Inputs placed along a spiral in 2D, projected to H dims."""
    rng = np.random.RandomState(seed + 10000)
    # 2D spiral
    angles = np.linspace(0, 4 * np.pi, V)
    radii = np.linspace(0.5, 2.0, V)
    coords_2d = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)  # V×2
    # Random projection 2D → H
    proj = rng.randn(2, H)
    W_in = coords_2d @ proj  # V × H
    W_out = rng.randn(H, V)
    return norm_rows(W_in), norm_cols(W_out)


def geom_clustered(seed):
    """3 clusters of ~9 inputs each, tight within cluster, far between."""
    rng = np.random.RandomState(seed + 10000)
    n_clusters = 3
    W_in = np.zeros((V, H))
    for v in range(V):
        cluster = v % n_clusters
        center = np.zeros(H)
        # Each cluster has a different center region
        start = cluster * (H // n_clusters)
        end = start + H // n_clusters
        center[start:end] = 1.0
        noise = rng.randn(H) * 0.3
        W_in[v] = center + noise
    W_out = rng.randn(H, V)
    return norm_rows(W_in), norm_cols(W_out)


def geom_sparse_binary(seed):
    """Each input activates exactly 8 random hidden neurons (sparse code)."""
    rng = np.random.RandomState(seed + 10000)
    k = 8
    W_in = np.zeros((V, H))
    for v in range(V):
        idx = rng.choice(H, size=k, replace=False)
        signs = rng.choice([-1.0, 1.0], size=k)
        W_in[v, idx] = signs
    W_out = rng.randn(H, V)
    return norm_rows(W_in), norm_cols(W_out)


def geom_fourier(seed):
    """Each input is a different frequency in a Fourier basis."""
    W_in = np.zeros((V, H))
    for v in range(V):
        freq = v + 1
        for h in range(H):
            if v % 2 == 0:
                W_in[v, h] = np.sin(2 * np.pi * freq * h / H)
            else:
                W_in[v, h] = np.cos(2 * np.pi * freq * h / H)
    rng = np.random.RandomState(seed + 10000)
    W_out = rng.randn(H, V)
    return norm_rows(W_in), norm_cols(W_out)


def geom_identity_block(seed):
    """First V neurons = dedicated 1:1, rest = zero. Minimal."""
    W_in = np.zeros((V, H))
    W_in[np.arange(V), np.arange(V)] = SCALE
    rng = np.random.RandomState(seed + 10000)
    W_out = rng.randn(H, V)
    return W_in.astype(np.float32), norm_cols(W_out)


def geom_3d_helix(seed):
    """3D helix projected to H dimensions."""
    rng = np.random.RandomState(seed + 10000)
    t = np.linspace(0, 3 * np.pi, V)
    coords_3d = np.stack([np.cos(t), np.sin(t), t / (3 * np.pi)], axis=1)  # V×3
    proj = rng.randn(3, H)
    W_in = coords_3d @ proj
    W_out = rng.randn(H, V)
    return norm_rows(W_in), norm_cols(W_out)


# ══════════════════════════════════════════════════════
GEOMETRIES = {
    'random':           geom_random,
    'concentric':       geom_concentric_rings,
    'grid-vertical':    geom_2d_grid_walls,
    'grid-horizontal':  geom_2d_grid_horizontal,
    'hypercube':        geom_hypercube,
    'simplex':          geom_simplex,
    'spiral':           geom_spiral,
    'clustered':        geom_clustered,
    'sparse-8':         geom_sparse_binary,
    'fourier':          geom_fourier,
    'identity-block':   geom_identity_block,
    '3d-helix':         geom_3d_helix,
}

print(f"{'='*80}")
print(f"  GEOMETRY SWEEP — {len(GEOMETRIES)} geometries × {len(SEEDS)} seeds")
print(f"  V={V}, H={H}, Budget={BUDGET}, Scale={SCALE}")
print(f"{'='*80}\n")

results = {}
for name, geom_fn in GEOMETRIES.items():
    scores = []
    conns_list = []
    t0 = time.time()
    for seed in SEEDS:
        W_in, W_out = geom_fn(seed)
        net = make_net_with_geometry(seed, W_in, W_out)
        targets = np.random.permutation(V)
        score = train(net, targets, V, max_attempts=BUDGET,
                      ticks=TICKS, stale_limit=STALE, verbose=False)
        scores.append(score)
        conns_list.append(net.count_connections())
    dt = time.time() - t0
    avg = np.mean(scores)
    std = np.std(scores)
    results[name] = (avg, std, scores, np.mean(conns_list), dt)
    print(f"  {name:<18s}  avg={avg*100:5.1f}% ±{std*100:4.1f}  "
          f"conns={np.mean(conns_list):5.0f}  "
          f"[{', '.join(f'{s*100:.0f}' for s in scores)}]  ({dt:.1f}s)")

# ── Ranking ──
print(f"\n{'='*80}")
print(f"  RANKING")
print(f"{'='*80}")
print(f"  {'#':>2s}  {'Geometry':<18s}  {'Avg':>6s}  {'±':>5s}  {'Conns':>6s}  {'Best':>5s}  {'Worst':>5s}")
print(f"  {'─'*60}")
ranked = sorted(results.items(), key=lambda x: -x[1][0])
for rank, (name, (avg, std, scores, conns, dt)) in enumerate(ranked):
    print(f"  {rank+1:2d}  {name:<18s}  {avg*100:5.1f}%  ±{std*100:4.1f}  "
          f"{conns:6.0f}  {max(scores)*100:4.0f}%  {min(scores)*100:4.0f}%")

# ── How spread are the results? ──
avgs = [v[0] for v in results.values()]
print(f"\n  Overall range: {min(avgs)*100:.1f}% — {max(avgs)*100:.1f}%")
print(f"  Spread: {(max(avgs)-min(avgs))*100:.1f}pp")
if max(avgs) - min(avgs) < 0.15:
    print(f"  → A hidden layer VALÓBAN ADAPTÁLÓDIK a geometriához!")
else:
    print(f"  → Geometria-függő — a hidden layer NEM teljesen geometria-agnosztikus")

print(f"\n{'='*80}")
print("DONE")
