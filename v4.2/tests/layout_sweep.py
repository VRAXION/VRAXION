"""Spatial layout sweep: which neuron arrangement + highway works best?
Tests: grid, ring, linear, phi-spiral, hub-spoke, hexagonal.
Each layout defines neuron positions + natural highway + distance-biased mutation.
Tested on V=32 permutation task with divnorm for speed."""
import sys, os, time, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
from model.graph import SelfWiringGraph

ALPHA = 2.0
V = 32
BUDGET = 8000
TICKS = 8
SEED = 42


def forward_divnorm_batch(net, ticks=TICKS, alpha=ALPHA):
    V, N = net.V, net.N
    charges = np.zeros((V, N), dtype=np.float32)
    acts = np.zeros((V, N), dtype=np.float32)
    for t in range(ticks):
        if t == 0:
            acts[:, :V] = np.eye(V, dtype=np.float32)
        raw = acts @ net.mask
        np.nan_to_num(raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        charges += raw
        total = np.abs(acts).sum(axis=1, keepdims=True) + 1e-6
        acts = np.maximum(charges - net.THRESHOLD, 0.0)
        acts /= (1.0 + alpha * total)
        charges = np.clip(charges, -1.0, 1.0)
    return charges[:, net.out_start:net.out_start + V]


def evaluate(net, targets):
    logits = forward_divnorm_batch(net)
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    acc = (np.argmax(probs, axis=1)[:V] == targets[:V]).mean()
    tp = probs[np.arange(V), targets[:V]].mean()
    return 0.5 * acc + 0.5 * tp, float(acc)


# ============================================================
# Spatial layouts — each returns (positions, highway_edges)
# positions: (N, 2) array of (x, y)
# highway_edges: list of (src, dst) tuples — pre-wired backbone
# ============================================================

def layout_grid(N, V):
    """Rectangular grid. Input left, output right, compute middle.
    Highway: horizontal connections left→right."""
    cols = int(math.ceil(math.sqrt(N * 1.5)))
    rows = int(math.ceil(N / cols))
    positions = np.zeros((N, 2), dtype=np.float32)
    for i in range(N):
        r, c = divmod(i, cols)
        positions[i] = [c / max(cols-1, 1), r / max(rows-1, 1)]
    # Highway: connect each neuron to its right neighbor in the grid
    highway = []
    for i in range(N):
        r, c = divmod(i, cols)
        right = r * cols + c + 1
        if c + 1 < cols and right < N:
            highway.append((i, right))
    return positions, highway


def layout_ring(N, V):
    """Neurons on a circle. Input/compute/output in sectors.
    Highway: circular chain."""
    positions = np.zeros((N, 2), dtype=np.float32)
    for i in range(N):
        angle = 2 * math.pi * i / N
        positions[i] = [math.cos(angle), math.sin(angle)]
    # Highway: ring i→i+1
    highway = [(i, (i + 1) % N) for i in range(N)]
    return positions, highway


def layout_linear(N, V):
    """Neurons in a line left→right. Input at start, output at end.
    Highway: chain connection."""
    positions = np.zeros((N, 2), dtype=np.float32)
    for i in range(N):
        positions[i] = [i / (N - 1), 0.5 + 0.3 * math.sin(i * 0.5)]  # slight wave
    # Highway: chain
    highway = [(i, i + 1) for i in range(N - 1)]
    return positions, highway


def layout_phi_spiral(N, V):
    """Golden angle spiral — neurons placed by phi rotation.
    Highway: spiral connection following placement order."""
    phi = (1 + math.sqrt(5)) / 2  # golden ratio
    golden_angle = 2 * math.pi / (phi * phi)
    positions = np.zeros((N, 2), dtype=np.float32)
    for i in range(N):
        r = math.sqrt(i / N)  # radius grows with sqrt for uniform density
        theta = i * golden_angle
        positions[i] = [r * math.cos(theta), r * math.sin(theta)]
    # Highway: spiral chain following placement order
    highway = [(i, i + 1) for i in range(N - 1)]
    # Plus: close the spiral (last to first)
    highway.append((N - 1, 0))
    return positions, highway


def layout_hub_spoke(N, V):
    """Central hub neurons connected to all. Like thalamus.
    Highway: hub↔all connections."""
    n_hubs = max(2, N // 20)  # ~5% are hub neurons
    positions = np.zeros((N, 2), dtype=np.float32)
    # Hubs at center
    for i in range(n_hubs):
        angle = 2 * math.pi * i / n_hubs
        positions[i] = [0.1 * math.cos(angle), 0.1 * math.sin(angle)]
    # Others on outer ring
    for i in range(n_hubs, N):
        angle = 2 * math.pi * (i - n_hubs) / (N - n_hubs)
        positions[i] = [math.cos(angle), math.sin(angle)]
    # Highway: each hub connects to every non-hub
    highway = []
    for h in range(n_hubs):
        for i in range(n_hubs, N):
            highway.append((h, i))
    return positions, highway


def layout_hexagonal(N, V):
    """Hexagonal lattice. Input left, output right.
    Highway: hex neighbor connections."""
    cols = int(math.ceil(math.sqrt(N * 1.15)))
    rows = int(math.ceil(N / cols))
    positions = np.zeros((N, 2), dtype=np.float32)
    for i in range(N):
        r, c = divmod(i, cols)
        x = c + (0.5 if r % 2 else 0)  # hex offset
        positions[i] = [x / max(cols, 1), r / max(rows-1, 1)]
    # Highway: hex neighbors (6 directions)
    highway = []
    for i in range(N):
        r, c = divmod(i, cols)
        # Right neighbor
        if c + 1 < cols:
            highway.append((i, i + 1))
        # Bottom neighbors
        if r + 1 < rows:
            below = (r + 1) * cols + c
            if below < N:
                highway.append((i, below))
            # Hex offset neighbor
            offset = c + (1 if r % 2 else -1)
            if 0 <= offset < cols:
                below_off = (r + 1) * cols + offset
                if below_off < N:
                    highway.append((i, below_off))
    return positions, highway


LAYOUTS = {
    'grid':       layout_grid,
    'ring':       layout_ring,
    'linear':     layout_linear,
    'phi_spiral': layout_phi_spiral,
    'hub_spoke':  layout_hub_spoke,
    'hexagonal':  layout_hexagonal,
}


# ============================================================
# Distance-biased mutation
# ============================================================

def make_spatial_net(V, seed, layout_fn):
    """Create network with spatial layout and distance-biased initial wiring."""
    np.random.seed(seed)
    random.seed(seed)
    net = SelfWiringGraph(V)
    N = net.N
    positions, highway = layout_fn(N, V)

    # Pre-compute distance matrix for biased mutation
    dists = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        dists[i] = np.sqrt(((positions - positions[i]) ** 2).sum(axis=1))

    # Connection probability: inversely proportional to distance
    conn_prob = 1.0 / (1.0 + 5.0 * dists)  # 5.0 = distance sensitivity
    np.fill_diagonal(conn_prob, 0)
    # Normalize per row
    row_sums = conn_prob.sum(axis=1, keepdims=True)
    conn_prob /= np.where(row_sums > 0, row_sums, 1)

    # Add highway connections (protected)
    protected = set()
    for src, dst in highway:
        if src < N and dst < N and src != dst:
            if net.mask[src, dst] == 0:
                net.mask[src, dst] = net.DRIVE
                net.alive.append((src, dst))
                net.alive_set.add((src, dst))
            protected.add((src, dst))

    return net, positions, conn_prob, protected


def train_spatial(net, targets, budget, conn_prob, protected,
                  stale_limit=6000, log_every=2000):
    """Train with distance-biased add mutation."""
    N = net.N

    def spatial_add(undo):
        """Add connection biased by distance."""
        r = random.randint(0, N - 1)
        c = np.random.choice(N, p=conn_prob[r])
        if r != c and net.mask[r, c] == 0:
            net.mask[r, c] = net.DRIVE if random.randint(0, 1) else -net.DRIVE
            net.alive.append((r, c))
            net.alive_set.add((r, c))
            undo.append(('A', r, c))

    score, acc = evaluate(net, targets)
    best_score = score
    best_acc = acc
    stale = 0
    trajectory = [(0, float(best_score), float(best_acc))]

    for att in range(budget):
        old_loss = int(net.loss_pct)
        old_drive = int(net.drive)

        # Use spatial add instead of net.mutate()
        # Still drift loss_pct and drive
        if random.randint(1, 5) == 1:
            net.loss_pct = np.int8(max(1, min(50, int(net.loss_pct) + random.randint(-3, 3))))
        if random.randint(1, 20) <= 7:
            net.drive = np.int8(max(-15, min(15, int(net.drive) + random.choice([-1, 1]))))

        undo = []
        d = int(net.drive)
        if d > 0:
            for _ in range(d):
                spatial_add(undo)
        elif d < 0:
            for _ in range(-d):
                net._remove(undo)

        new_score, new_acc = evaluate(net, targets)

        if new_score > score:
            score = new_score
            best_score = max(best_score, score)
            best_acc = max(best_acc, new_acc)
            stale = 0
        else:
            net.replay(undo)
            net.loss_pct = np.int8(old_loss)
            net.drive = np.int8(old_drive)
            # Re-protect highway
            for r, c in protected:
                if net.mask[r, c] == 0:
                    net.mask[r, c] = net.DRIVE
                    if (r, c) not in net.alive_set:
                        net.alive.append((r, c))
                        net.alive_set.add((r, c))
            stale += 1

        if (att + 1) % log_every == 0:
            trajectory.append((att + 1, float(best_score), float(best_acc)))
        if best_score >= 0.99 or stale >= stale_limit:
            break

    if (att + 1) % log_every != 0:
        trajectory.append((att + 1, float(best_score), float(best_acc)))

    return float(best_score), float(best_acc), att + 1, trajectory


# ============================================================
# Also test baseline (no spatial, no highway) for comparison
# ============================================================

def train_baseline(net, targets, budget, stale_limit=6000, log_every=2000):
    score, acc = evaluate(net, targets)
    best_score = score
    best_acc = acc
    stale = 0
    trajectory = [(0, float(best_score), float(best_acc))]

    for att in range(budget):
        old_loss = int(net.loss_pct)
        old_drive = int(net.drive)
        undo = net.mutate()
        new_score, new_acc = evaluate(net, targets)
        if new_score > score:
            score = new_score
            best_score = max(best_score, score)
            best_acc = max(best_acc, new_acc)
            stale = 0
        else:
            net.replay(undo)
            net.loss_pct = np.int8(old_loss)
            net.drive = np.int8(old_drive)
            stale += 1
        if (att + 1) % log_every == 0:
            trajectory.append((att + 1, float(best_score), float(best_acc)))
        if best_score >= 0.99 or stale >= stale_limit:
            break
    if (att + 1) % log_every != 0:
        trajectory.append((att + 1, float(best_score), float(best_acc)))
    return float(best_score), float(best_acc), att + 1, trajectory


# ============================================================
# Main sweep
# ============================================================

print(f"LAYOUT SWEEP | V={V} N={V*3} budget={BUDGET} ticks={TICKS} seed={SEED}")
print(f"{'='*75}")

# Baseline first
print(f"\n--- baseline (no spatial, no highway) ---")
np.random.seed(SEED); random.seed(SEED)
net = SelfWiringGraph(V)
targets = np.arange(V); np.random.shuffle(targets)
random.seed(SEED * 1000 + 1)
t0 = time.time()
bs, ba, bsteps, btraj = train_baseline(net, targets, BUDGET)
be = time.time() - t0
traj_str = " -> ".join(f"{s*100:.1f}" for _, s, _ in btraj)
print(f"Score: {bs*100:.1f}% Acc: {ba*100:.0f}% Steps: {bsteps} Time: {be:.0f}s Conns: {net.count_connections()}")
print(f"Trajectory: {traj_str}")

results = [('baseline', bs, ba, bsteps, be, 0, net.count_connections())]

# Each layout
for name, layout_fn in LAYOUTS.items():
    print(f"\n--- {name} ---")
    np.random.seed(SEED); random.seed(SEED)
    # Need same targets
    _tmpnet = SelfWiringGraph(V)
    targets = np.arange(V); np.random.shuffle(targets)

    net, positions, conn_prob, protected = make_spatial_net(V, SEED, layout_fn)

    random.seed(SEED * 1000 + 1)
    t0 = time.time()
    sc, ac, steps, traj = train_spatial(net, targets, BUDGET, conn_prob, protected)
    elapsed = time.time() - t0
    traj_str = " -> ".join(f"{s*100:.1f}" for _, s, _ in traj)
    print(f"Score: {sc*100:.1f}% Acc: {ac*100:.0f}% Steps: {steps} Time: {elapsed:.0f}s "
          f"Conns: {net.count_connections()} Highway: {len(protected)}")
    print(f"Trajectory: {traj_str}")
    results.append((name, sc, ac, steps, elapsed, len(protected), net.count_connections()))

# Summary
print(f"\n{'='*75}")
print(f"LAYOUT SWEEP SUMMARY | V={V} budget={BUDGET}")
print(f"{'name':<14s} {'score':>7s} {'acc':>5s} {'steps':>6s} {'time':>5s} {'hwy':>5s} {'conns':>6s}")
print("-" * 55)
for name, sc, ac, steps, elapsed, hwy, conns in sorted(results, key=lambda x: -x[1]):
    print(f"{name:<14s} {sc*100:6.1f}% {ac*100:4.0f}% {steps:6d} {elapsed:4.0f}s {hwy:5d} {conns:6d}")
