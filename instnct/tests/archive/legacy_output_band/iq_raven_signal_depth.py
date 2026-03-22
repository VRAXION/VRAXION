"""Signal propagation depth test for Raven matrices.

Hypothesis: 8 ticks is not enough for information from cell[0][0]
to reach the output neurons through a sparse 4% graph.

Tests:
1. Ticks sweep: 8, 16, 32, 64, 128 — does more propagation help?
2. Density sweep: 4%, 8%, 16%, 32% — does denser graph help?
3. Path analysis: what's the actual shortest path from input to output?
4. NV_RATIO sweep: 3, 5, 8 — more neurons = more routing options?
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
from collections import deque
from model.graph import SelfWiringGraph


# ============================================================
# Minimal Raven setup (from v2)
# ============================================================

N_S, N_C = 3, 3
N_ANSWERS = N_S * N_C


def cell_id(s, c):
    return s * N_C + c


def puzzle_2x2_diagonal():
    vals = random.sample(range(N_ANSWERS), 2)
    return [[vals[0], vals[1]], [vals[1], vals[0]]]


def puzzle_2x2_row_shape():
    s = random.sample(range(N_S), 2)
    c = random.sample(range(N_C), 2)
    return [[cell_id(s[0], c[0]), cell_id(s[0], c[1])],
            [cell_id(s[1], c[0]), cell_id(s[1], c[1])]]


def build_dataset(puzzle_fn, n, seed=42):
    old = random.getstate()
    random.seed(seed)
    V_in = 3 * N_ANSWERS
    inputs = np.zeros((n, V_in), dtype=np.float32)
    targets = np.zeros(n, dtype=np.int32)
    for i in range(n):
        g = puzzle_fn()
        ctx = [g[0][0], g[0][1], g[1][0]]
        for j, cid in enumerate(ctx):
            inputs[i, j * N_ANSWERS + cid] = 1.0
        targets[i] = g[1][1]
    random.setstate(old)
    return inputs, targets


# ============================================================
# Path analysis
# ============================================================

def analyze_paths(net):
    """BFS shortest paths from each input neuron to each output neuron."""
    N = net.N
    V = net.V
    adj = [[] for _ in range(N)]
    for r, c in net.alive:
        adj[r].append(c)

    input_neurons = list(range(V))
    output_neurons = list(range(net.out_start, net.out_start + V))

    # BFS from each input
    min_paths = []
    reachable_count = 0
    unreachable_count = 0

    for src in input_neurons[:9]:  # first 9 = first cell's one-hot slots
        dist = [-1] * N
        dist[src] = 0
        q = deque([src])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)

        for dst in output_neurons[:9]:  # first 9 output slots
            if dist[dst] >= 0:
                min_paths.append(dist[dst])
                reachable_count += 1
            else:
                unreachable_count += 1

    return {
        'min': min(min_paths) if min_paths else -1,
        'max': max(min_paths) if min_paths else -1,
        'mean': np.mean(min_paths) if min_paths else -1,
        'median': np.median(min_paths) if min_paths else -1,
        'reachable': reachable_count,
        'unreachable': unreachable_count,
        'total_pairs': reachable_count + unreachable_count,
    }


def analyze_full_reachability(net, ticks):
    """What fraction of output neurons are reachable from ANY input within `ticks` hops?"""
    N = net.N
    V = net.V
    adj = [[] for _ in range(N)]
    for r, c in net.alive:
        adj[r].append(c)

    # BFS from ALL input neurons simultaneously
    reached = set()
    frontier = set(range(V))  # all input neurons
    reached.update(frontier)

    for t in range(ticks):
        next_frontier = set()
        for u in frontier:
            for v in adj[u]:
                if v not in reached:
                    reached.add(v)
                    next_frontier.add(v)
        frontier = next_frontier
        if not frontier:
            break

    output_neurons = set(range(net.out_start, net.out_start + V))
    output_reached = output_neurons & reached
    return len(output_reached) / len(output_neurons) if output_neurons else 0


# ============================================================
# Forward + eval + train (parametric ticks)
# ============================================================

def forward_batch(net, patterns, ticks):
    K = patterns.shape[0]
    N = net.N
    V = net.V
    charges = np.zeros((K, N), dtype=np.float32)
    acts = np.zeros((K, N), dtype=np.float32)
    retain = float(net.retention)
    for t in range(ticks):
        if t == 0:
            acts[:, :V] = patterns
        raw = acts @ net.mask
        np.nan_to_num(raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        charges += raw
        charges *= retain
        acts = np.maximum(charges - net.THRESHOLD, 0.0)
        charges = np.clip(charges, -1.0, 1.0)
    return charges[:, net.out_start:net.out_start + V]


def evaluate(net, inputs, targets, ticks, sample=128):
    idx = np.random.choice(len(inputs), min(sample, len(inputs)), replace=False)
    logits = forward_batch(net, inputs[idx], ticks)[:, :N_ANSWERS]
    return float((np.argmax(logits, axis=1) == targets[idx]).mean())


def accuracy_full(net, inputs, targets, ticks):
    correct = 0
    for s in range(0, len(inputs), 256):
        e = min(s + 256, len(inputs))
        logits = forward_batch(net, inputs[s:e], ticks)[:, :N_ANSWERS]
        correct += (np.argmax(logits, axis=1) == targets[s:e]).sum()
    return correct / len(inputs)


def train_puzzle(net, inputs, targets, budget, ticks):
    stale_limit = budget // 2
    rewire_threshold = stale_limit // 3
    score = evaluate(net, inputs, targets, ticks)
    best = score
    stale = 0
    for att in range(budget):
        old_l = int(net.loss_pct)
        old_d = int(net.drive)
        undo = net.mutate()
        new = evaluate(net, inputs, targets, ticks)
        if new > score:
            score = new
            best = max(best, score)
            stale = 0
        else:
            net.replay(undo)
            net.loss_pct = np.int8(old_l)
            net.drive = np.int8(old_d)
            stale += 1
            if stale > rewire_threshold:
                rw = []
                net._rewire(rw)
                rw_s = evaluate(net, inputs, targets, ticks)
                if rw_s > score:
                    score = rw_s
                    best = max(best, score)
                    stale = 0
                else:
                    net.replay(rw)
        if best >= 0.99 or stale >= stale_limit:
            break
    return best, att + 1


# ============================================================
# Test 1: PATH ANALYSIS — how far is input from output?
# ============================================================

SEED = 42
print("=" * 72)
print("TEST 1: PATH ANALYSIS — INPUT→OUTPUT DISTANCE")
print("=" * 72)

for nv in [3, 5, 8]:
    for density_pct in [4, 8, 16, 32]:
        np.random.seed(SEED)
        random.seed(SEED)
        V = 27  # 3 cells × 9 answers
        net = SelfWiringGraph(V)
        # Override density
        net.NV_RATIO = nv
        N = V * nv
        net.N = N
        net.out_start = N - V if N >= 2 * V else 0
        d = density_pct / 100
        r = np.random.rand(N, N)
        net.mask = np.zeros((N, N), dtype=np.float32)
        net.mask[r < d / 2] = -net.DRIVE
        net.mask[r > 1 - d / 2] = net.DRIVE
        np.fill_diagonal(net.mask, 0)
        net.resync_alive()
        net.state = np.zeros(N, dtype=np.float32)
        net.charge = np.zeros(N, dtype=np.float32)

        paths = analyze_paths(net)
        reach_8 = analyze_full_reachability(net, 8)
        reach_16 = analyze_full_reachability(net, 16)
        reach_32 = analyze_full_reachability(net, 32)

        print(f"  NV={nv} density={density_pct:2d}% N={N:4d} conns={net.count_connections():5d} | "
              f"path min={paths['min']:2d} mean={paths['mean']:.1f} max={paths['max']:2d} | "
              f"reach@8={reach_8*100:.0f}% @16={reach_16*100:.0f}% @32={reach_32*100:.0f}% | "
              f"unreachable={paths['unreachable']}/{paths['total_pairs']}",
              flush=True)


# ============================================================
# Test 2: TICKS SWEEP — does more propagation help?
# ============================================================

print()
print("=" * 72)
print("TEST 2: TICKS SWEEP — 2x2 diagonal puzzle")
print("=" * 72)

N_TRAIN, N_TEST = 300, 100
BUDGET = 20000
train_in, train_tgt = build_dataset(puzzle_2x2_diagonal, N_TRAIN, SEED)
test_in, test_tgt = build_dataset(puzzle_2x2_diagonal, N_TEST, SEED + 999)

print(f"{'Ticks':>6s} {'Train':>7s} {'Test':>7s} {'Steps':>6s} {'Time':>5s}")
print("-" * 40)

for ticks in [4, 8, 16, 32, 64]:
    np.random.seed(SEED)
    random.seed(SEED)
    V = train_in.shape[1]
    net = SelfWiringGraph(V)

    t0 = time.time()
    best, steps = train_puzzle(net, train_in, train_tgt, BUDGET, ticks)
    elapsed = time.time() - t0
    tr = accuracy_full(net, train_in, train_tgt, ticks)
    te = accuracy_full(net, test_in, test_tgt, ticks)
    print(f"{ticks:6d} {tr*100:6.1f}% {te*100:6.1f}% {steps:6d} {elapsed:4.0f}s", flush=True)


# ============================================================
# Test 3: TICKS SWEEP — 2x2 row=shape puzzle
# ============================================================

print()
print("=" * 72)
print("TEST 3: TICKS SWEEP — 2x2 row=shape puzzle")
print("=" * 72)

train_in2, train_tgt2 = build_dataset(puzzle_2x2_row_shape, N_TRAIN, SEED)
test_in2, test_tgt2 = build_dataset(puzzle_2x2_row_shape, N_TEST, SEED + 999)

print(f"{'Ticks':>6s} {'Train':>7s} {'Test':>7s} {'Steps':>6s} {'Time':>5s}")
print("-" * 40)

for ticks in [4, 8, 16, 32, 64]:
    np.random.seed(SEED)
    random.seed(SEED)
    V = train_in2.shape[1]
    net = SelfWiringGraph(V)

    t0 = time.time()
    best, steps = train_puzzle(net, train_in2, train_tgt2, BUDGET, ticks)
    elapsed = time.time() - t0
    tr = accuracy_full(net, train_in2, train_tgt2, ticks)
    te = accuracy_full(net, test_in2, test_tgt2, ticks)
    print(f"{ticks:6d} {tr*100:6.1f}% {te*100:6.1f}% {steps:6d} {elapsed:4.0f}s", flush=True)


# ============================================================
# Test 4: DENSITY SWEEP — denser graph = better signal propagation?
# ============================================================

print()
print("=" * 72)
print("TEST 4: DENSITY SWEEP — 2x2 diagonal, ticks=32")
print("=" * 72)

TICKS_FIXED = 32

print(f"{'Density':>8s} {'Conns':>6s} {'Train':>7s} {'Test':>7s} {'Steps':>6s} {'Time':>5s}")
print("-" * 48)

for density_pct in [4, 8, 16, 32]:
    np.random.seed(SEED)
    random.seed(SEED)
    V = train_in.shape[1]

    # Build custom graph with different density
    net = SelfWiringGraph(V)
    N = net.N
    d = density_pct / 100
    r = np.random.rand(N, N)
    net.mask = np.zeros((N, N), dtype=np.float32)
    net.mask[r < d / 2] = -net.DRIVE
    net.mask[r > 1 - d / 2] = net.DRIVE
    np.fill_diagonal(net.mask, 0)
    net.resync_alive()

    t0 = time.time()
    best, steps = train_puzzle(net, train_in, train_tgt, BUDGET, TICKS_FIXED)
    elapsed = time.time() - t0
    tr = accuracy_full(net, train_in, train_tgt, TICKS_FIXED)
    te = accuracy_full(net, test_in, test_tgt, TICKS_FIXED)
    print(f"{density_pct:7d}% {net.count_connections():6d} {tr*100:6.1f}% {te*100:6.1f}% "
          f"{steps:6d} {elapsed:4.0f}s", flush=True)


# ============================================================
# Test 5: SIGNAL TRACE — watch activation spread tick by tick
# ============================================================

print()
print("=" * 72)
print("TEST 5: SIGNAL TRACE — activation spread per tick")
print("=" * 72)

np.random.seed(SEED)
random.seed(SEED)
V = train_in.shape[1]
net = SelfWiringGraph(V)

# Single input
x = train_in[0:1]
N = net.N
charges = np.zeros((1, N), dtype=np.float32)
acts = np.zeros((1, N), dtype=np.float32)
retain = float(net.retention)

print(f"  N={N} | V={V} | out_start={net.out_start} | conns={net.count_connections()}")
print(f"  {'Tick':>4s} {'Active':>7s} {'Mean|act|':>10s} {'Max|act|':>10s} "
      f"{'OutActive':>10s} {'OutMean':>10s} {'OutMax':>10s}")
print("  " + "-" * 70)

for t in range(64):
    if t == 0:
        acts[:, :V] = x
    raw = acts @ net.mask
    np.nan_to_num(raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    charges += raw
    charges *= retain
    acts = np.maximum(charges - net.THRESHOLD, 0.0)
    charges_clipped = np.clip(charges, -1.0, 1.0)

    n_active = int((acts[0] > 0).sum())
    mean_act = float(acts[0][acts[0] > 0].mean()) if n_active > 0 else 0
    max_act = float(acts[0].max())

    out_slice = acts[0, net.out_start:net.out_start + V]
    out_active = int((out_slice > 0).sum())
    out_mean = float(out_slice[out_slice > 0].mean()) if out_active > 0 else 0
    out_max = float(out_slice.max())

    charges = charges_clipped

    if t < 16 or t % 8 == 0:
        print(f"  {t:4d} {n_active:7d} {mean_act:10.6f} {max_act:10.6f} "
              f"{out_active:10d} {out_mean:10.6f} {out_max:10.6f}")

print()
print("=" * 72)
print("DONE — check if ticks/density/paths explain the Raven failure")
print("=" * 72)
