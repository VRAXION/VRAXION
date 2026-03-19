"""Layout sweep on byte computation tasks.
Tests which spatial layout helps the network build logic circuits.
6 layouts × 3 tasks (XOR, NOT, parity) = 18 experiments + 3 baselines."""
import sys, os, time, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
from model.graph import SelfWiringGraph

ALPHA = 2.0
V = 64  # 192 neurons total (64 input + 64 compute + 64 output)
BUDGET = 16000
TICKS = 8
SEED = 42


def forward_patterns(net, patterns, ticks=TICKS, alpha=ALPHA):
    K = patterns.shape[0]
    N = net.N
    charges = np.zeros((K, N), dtype=np.float32)
    acts = np.zeros((K, N), dtype=np.float32)
    for t in range(ticks):
        if t == 0:
            acts[:, :V] = patterns
        raw = acts @ net.mask
        np.nan_to_num(raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        charges += raw
        total = np.abs(acts).sum(axis=1, keepdims=True) + 1e-6
        acts = np.maximum(charges - net.THRESHOLD, 0.0)
        acts /= (1.0 + alpha * total)
        charges = np.clip(charges, -1.0, 1.0)
    return charges[:, net.out_start:net.out_start + V]


def evaluate_patterns(net, inputs, targets):
    outputs = forward_patterns(net, inputs)
    predicted = (outputs > 0).astype(np.int8)
    bit_acc = (predicted == targets).mean()
    pat_acc = (predicted == targets).all(axis=1).mean()
    return float(bit_acc), float(pat_acc)


# ============================================================
# Tasks
# ============================================================

def make_random_patterns(n_patterns, n_bits):
    """Generate random binary patterns."""
    return np.random.randint(0, 2, size=(n_patterns, n_bits)).astype(np.float32)

def task_xor(inputs):
    V = inputs.shape[1]
    mask = np.array([1, 0] * (V // 2), dtype=np.int8)[:V]
    return (inputs.astype(np.int8) ^ mask)

def task_not(inputs):
    return (1 - inputs).astype(np.int8)

def task_parity(inputs):
    parity = inputs.astype(np.int8).sum(axis=1) % 2
    out = np.zeros_like(inputs, dtype=np.int8)
    out[:, 0] = parity
    return out

TASKS = {'XOR': task_xor, 'NOT': task_not, 'parity': task_parity}


# ============================================================
# Layouts
# ============================================================

def layout_none(N, V):
    return None, []

def layout_grid(N, V):
    cols = int(math.ceil(math.sqrt(N * 1.5)))
    rows = int(math.ceil(N / cols))
    positions = np.zeros((N, 2), dtype=np.float32)
    for i in range(N):
        r, c = divmod(i, cols)
        positions[i] = [c / max(cols-1, 1), r / max(rows-1, 1)]
    highway = []
    for i in range(N):
        r, c = divmod(i, cols)
        right = r * cols + c + 1
        if c + 1 < cols and right < N:
            highway.append((i, right))
    return positions, highway

def layout_ring(N, V):
    positions = np.zeros((N, 2), dtype=np.float32)
    for i in range(N):
        angle = 2 * math.pi * i / N
        positions[i] = [math.cos(angle), math.sin(angle)]
    highway = [(i, (i + 1) % N) for i in range(N)]
    return positions, highway

def layout_phi_spiral(N, V):
    phi = (1 + math.sqrt(5)) / 2
    golden_angle = 2 * math.pi / (phi * phi)
    positions = np.zeros((N, 2), dtype=np.float32)
    for i in range(N):
        r = math.sqrt(i / N)
        theta = i * golden_angle
        positions[i] = [r * math.cos(theta), r * math.sin(theta)]
    highway = [(i, (i + 1) % N) for i in range(N)]
    return positions, highway

def layout_hexagonal(N, V):
    cols = int(math.ceil(math.sqrt(N * 1.15)))
    rows = int(math.ceil(N / cols))
    positions = np.zeros((N, 2), dtype=np.float32)
    for i in range(N):
        r, c = divmod(i, cols)
        x = c + (0.5 if r % 2 else 0)
        positions[i] = [x / max(cols, 1), r / max(rows-1, 1)]
    highway = []
    for i in range(N):
        r, c = divmod(i, cols)
        if c + 1 < cols: highway.append((i, i + 1))
        if r + 1 < rows:
            below = (r + 1) * cols + c
            if below < N: highway.append((i, below))
            offset = c + (1 if r % 2 else -1)
            if 0 <= offset < cols:
                below_off = (r + 1) * cols + offset
                if below_off < N: highway.append((i, below_off))
    return positions, highway

def layout_linear(N, V):
    positions = np.zeros((N, 2), dtype=np.float32)
    for i in range(N):
        positions[i] = [i / (N - 1), 0.5]
    highway = [(i, i + 1) for i in range(N - 1)]
    return positions, highway

LAYOUTS = {
    'baseline': layout_none,
    'grid': layout_grid,
    'ring': layout_ring,
    'phi_spiral': layout_phi_spiral,
    'hexagonal': layout_hexagonal,
    'linear': layout_linear,
}


# ============================================================
# Training with spatial bias
# ============================================================

def setup_net(V, seed, layout_fn):
    np.random.seed(seed); random.seed(seed)
    net = SelfWiringGraph(V)
    N = net.N
    result = layout_fn(N, V)
    positions, highway = result if result[0] is not None else (None, [])

    protected = set()
    for src, dst in highway:
        if src < N and dst < N and src != dst:
            if net.mask[src, dst] == 0:
                net.mask[src, dst] = net.DRIVE
                net.alive.append((src, dst))
                net.alive_set.add((src, dst))
            protected.add((src, dst))

    conn_prob = None
    if positions is not None:
        dists = np.zeros((N, N), dtype=np.float32)
        for i in range(N):
            dists[i] = np.sqrt(((positions - positions[i]) ** 2).sum(axis=1))
        conn_prob = 1.0 / (1.0 + 5.0 * dists)
        np.fill_diagonal(conn_prob, 0)
        row_sums = conn_prob.sum(axis=1, keepdims=True)
        conn_prob /= np.where(row_sums > 0, row_sums, 1)

    return net, conn_prob, protected


def train_task(net, inputs, targets, budget, conn_prob, protected, stale_limit=4000):
    N = net.N

    def spatial_add(undo):
        r = random.randint(0, N - 1)
        if conn_prob is not None:
            c = np.random.choice(N, p=conn_prob[r])
        else:
            c = random.randint(0, N - 1)
        if r != c and net.mask[r, c] == 0:
            net.mask[r, c] = net.DRIVE if random.randint(0, 1) else -net.DRIVE
            net.alive.append((r, c))
            net.alive_set.add((r, c))
            undo.append(('A', r, c))

    bit_acc, pat_acc = evaluate_patterns(net, inputs, targets)
    best_bit = bit_acc
    best_pat = pat_acc
    stale = 0

    for att in range(budget):
        old_loss = int(net.loss_pct)
        old_drive = int(net.drive)

        if random.randint(1, 5) == 1:
            net.loss_pct = np.int8(max(1, min(50, int(net.loss_pct) + random.randint(-3, 3))))
        if random.randint(1, 20) <= 7:
            net.drive = np.int8(max(-15, min(15, int(net.drive) + random.choice([-1, 1]))))

        undo = []
        d = int(net.drive)
        if d > 0:
            for _ in range(d): spatial_add(undo)
        elif d < 0:
            for _ in range(-d): net._remove(undo)

        new_bit, new_pat = evaluate_patterns(net, inputs, targets)
        if new_bit > bit_acc:
            bit_acc = new_bit
            best_bit = max(best_bit, bit_acc)
            best_pat = max(best_pat, new_pat)
            stale = 0
        else:
            net.replay(undo)
            net.loss_pct = np.int8(old_loss)
            net.drive = np.int8(old_drive)
            for r, c in protected:
                if net.mask[r, c] == 0:
                    net.mask[r, c] = net.DRIVE
                    if (r, c) not in net.alive_set:
                        net.alive.append((r, c))
                        net.alive_set.add((r, c))
            stale += 1

        if best_bit >= 0.999 or stale >= stale_limit:
            break

    return best_bit, best_pat, att + 1


# ============================================================
# Main
# ============================================================

np.random.seed(0)
all_inputs = make_random_patterns(512, V)
train_inputs = all_inputs[:128]
test_inputs = all_inputs[128:]

print(f"LAYOUT × BYTE TASK SWEEP | V={V} N={V*3} budget={BUDGET}")
print(f"Train: {len(train_inputs)} patterns, Test: {len(test_inputs)} patterns")
print(f"{'='*80}")
print(f"{'layout':<12s} {'task':<8s} {'tr_bit':>7s} {'tr_pat':>7s} "
      f"{'full_bit':>8s} {'full_pat':>8s} {'steps':>6s} {'time':>5s} {'hwy':>4s}")
print("-" * 75)

results = []

for layout_name, layout_fn in LAYOUTS.items():
    for task_name, task_fn in TASKS.items():
        train_targets = task_fn(train_inputs)
        test_targets = task_fn(test_inputs)

        net, conn_prob, protected = setup_net(V, SEED, layout_fn)
        random.seed(SEED * 1000 + 1)

        t0 = time.time()
        tb, tp, steps = train_task(net, train_inputs, train_targets, BUDGET,
                                    conn_prob, protected)
        elapsed = time.time() - t0

        fb, fp = evaluate_patterns(net, test_inputs, test_targets)

        print(f"{layout_name:<12s} {task_name:<8s} {tb*100:6.1f}% {tp*100:6.1f}% "
              f"{fb*100:7.1f}% {fp*100:7.1f}% {steps:6d} {elapsed:4.1f}s {len(protected):4d}",
              flush=True)
        results.append((layout_name, task_name, tb, tp, fb, fp, steps, len(protected)))

# Summary per layout (averaged across tasks)
print(f"\n{'='*80}")
print(f"AVERAGE ACROSS TASKS (full 256 pattern bit accuracy)")
print(f"{'layout':<14s} {'avg_bit':>8s} {'avg_pat':>8s} {'best_task':>10s}")
print("-" * 45)
for layout_name in LAYOUTS:
    layout_results = [r for r in results if r[0] == layout_name]
    avg_bit = np.mean([r[4] for r in layout_results])
    avg_pat = np.mean([r[5] for r in layout_results])
    best = max(layout_results, key=lambda x: x[4])
    print(f"{layout_name:<14s} {avg_bit*100:7.1f}% {avg_pat*100:7.1f}% {best[1]:<10s}")
