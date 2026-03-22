"""Raven's Progressive Matrices v2 — compact encoding.

v1 failed because V=64 (8 cells × 8-dim one-hot) was too large.
v2 fixes: encode each cell as a SINGLE integer index, feed the
8-cell context as a small one-hot vector.

Encoding: cell = shape * N_COUNTS + count (0..15)
Grid input = 8 integers → one-hot of 16^8? No, too large.

Better: POSITIONAL encoding. Each of the 8 grid positions gets
its own vocab slot, and we encode "position P has value V" as
a single active neuron at offset P*16 + V.

So input = 8 × 16 = 128? Still big.

SMALLEST: 2 shapes × 3 counts = 6 answers. 8 cells × 6 = 48.
Or even simpler: just 2 attributes × 3 values each = 9.

Let's try the MINIMAL version: 3 shapes × 3 counts = 9 answers.
Positional: 8 × 9 = 72 input. Still too much.

NEW IDEA: Don't give all 8 cells. Give the ROW pattern.
Just the last row: cell[2][0], cell[2][1], ?
Plus a "rule hint" from row 1: cell[0][0], cell[0][1], cell[0][2]
= 5 cells × small vocab.

Actually the SIMPLEST Raven: just 2×2 grid with 1 missing.
A  B
C  ?
Rule: rows share attribute, columns share attribute.
4 possible answers. 3 cells input.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
from model.graph import SelfWiringGraph


# ============================================================
# Minimal Raven: 2×2 grid, 3 cells → predict 4th
# ============================================================

SHAPES = ['△', '□', '○']  # 3
COUNTS = ['1', '2', '3']   # 3
N_S = 3
N_C = 3
N_ANSWERS = N_S * N_C  # 9


def cell_id(shape, count):
    return shape * N_C + count


def cell_str(cid):
    s, c = divmod(cid, N_C)
    return f"{SHAPES[s]}{COUNTS[c]}"


def encode_cell_onehot(cid, n=N_ANSWERS):
    v = np.zeros(n, dtype=np.float32)
    v[cid] = 1.0
    return v


# ---- 2×2 Puzzles ----

def puzzle_2x2_row_shape_col_count():
    """Row = same shape, Col = same count.
    △1  △2
    □1  □2  → answer: □2
    """
    s = random.sample(range(N_S), 2)
    c = random.sample(range(N_C), 2)
    grid = [[cell_id(s[0], c[0]), cell_id(s[0], c[1])],
            [cell_id(s[1], c[0]), cell_id(s[1], c[1])]]
    return grid


def puzzle_2x2_row_count_col_shape():
    """Row = same count, Col = same shape.
    △1  □1
    △2  □2  → answer: □2
    """
    s = random.sample(range(N_S), 2)
    c = random.sample(range(N_C), 2)
    grid = [[cell_id(s[0], c[0]), cell_id(s[1], c[0])],
            [cell_id(s[0], c[1]), cell_id(s[1], c[1])]]
    return grid


def puzzle_2x2_diagonal():
    """Diagonal = same, anti-diagonal = same.
    △1  □2
    □2  △1  → answer: △1
    """
    vals = random.sample(range(N_ANSWERS), 2)
    grid = [[vals[0], vals[1]],
            [vals[1], vals[0]]]
    return grid


PUZZLES_2X2 = [
    ("2x2 row=shape", puzzle_2x2_row_shape_col_count),
    ("2x2 row=count", puzzle_2x2_row_count_col_shape),
    ("2x2 diagonal",  puzzle_2x2_diagonal),
]


# ---- 3×3 Puzzles (compact) ----

def puzzle_3x3_simple():
    """Row = same shape, Col = same count. 3×3."""
    s = random.sample(range(N_S), 3)
    c = random.sample(range(N_C), 3)
    grid = []
    for r in range(3):
        row = [cell_id(s[r], c[col]) for col in range(3)]
        grid.append(row)
    return grid


def puzzle_3x3_transpose():
    """Row = same count, Col = same shape. 3×3."""
    s = random.sample(range(N_S), 3)
    c = random.sample(range(N_C), 3)
    grid = []
    for r in range(3):
        row = [cell_id(s[col], c[r]) for col in range(3)]
        grid.append(row)
    return grid


def puzzle_3x3_latin():
    """Latin square: each row/col has all 3 shapes and all 3 counts."""
    s = random.sample(range(N_S), 3)
    c = random.sample(range(N_C), 3)
    perms = [[0,1,2], [1,2,0], [2,0,1]]
    grid = []
    for r in range(3):
        row = [cell_id(s[perms[r][col]], c[perms[(r+1)%3][col]]) for col in range(3)]
        grid.append(row)
    return grid


PUZZLES_3X3 = [
    ("3x3 row=shape",   puzzle_3x3_simple),
    ("3x3 row=count",   puzzle_3x3_transpose),
    ("3x3 latin",       puzzle_3x3_latin),
]


# ============================================================
# Dataset building
# ============================================================

def build_dataset(puzzle_fn, n_samples, grid_size, seed=42):
    """Build input/target arrays from puzzle instances."""
    rng_state = random.getstate()
    random.seed(seed)

    if grid_size == 2:
        n_context = 3  # cells [0][0], [0][1], [1][0]
    else:
        n_context = 8  # all but [2][2]

    input_dim = n_context * N_ANSWERS
    inputs = np.zeros((n_samples, input_dim), dtype=np.float32)
    targets = np.zeros(n_samples, dtype=np.int32)

    for i in range(n_samples):
        grid = puzzle_fn()
        if grid_size == 2:
            context = [grid[0][0], grid[0][1], grid[1][0]]
            answer = grid[1][1]
        else:
            context = []
            for r in range(3):
                for c in range(3):
                    if r == 2 and c == 2:
                        continue
                    context.append(grid[r][c])
            answer = grid[2][2]

        for j, cid in enumerate(context):
            inputs[i, j * N_ANSWERS + cid] = 1.0
        targets[i] = answer

    random.setstate(rng_state)
    return inputs, targets


# ============================================================
# Forward + train
# ============================================================

def forward_batch(net, patterns, ticks=8):
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


def evaluate(net, inputs, targets, sample=128, ticks=8):
    idx = np.random.choice(len(inputs), min(sample, len(inputs)), replace=False)
    logits = forward_batch(net, inputs[idx], ticks)[:, :N_ANSWERS]
    preds = np.argmax(logits, axis=1)
    return float((preds == targets[idx]).mean())


def accuracy_full(net, inputs, targets, ticks=8):
    correct = 0
    for s in range(0, len(inputs), 256):
        e = min(s + 256, len(inputs))
        logits = forward_batch(net, inputs[s:e], ticks)[:, :N_ANSWERS]
        preds = np.argmax(logits, axis=1)
        correct += (preds == targets[s:e]).sum()
    return correct / len(inputs)


def train_puzzle(net, inputs, targets, budget=20000, ticks=8):
    stale_limit = budget // 2
    rewire_threshold = stale_limit // 3
    score = evaluate(net, inputs, targets, ticks=ticks)
    best = score
    stale = 0
    for att in range(budget):
        old_loss = int(net.loss_pct)
        old_drive = int(net.drive)
        undo = net.mutate()
        new = evaluate(net, inputs, targets, ticks=ticks)
        if new > score:
            score = new
            best = max(best, score)
            stale = 0
        else:
            net.replay(undo)
            net.loss_pct = np.int8(old_loss)
            net.drive = np.int8(old_drive)
            stale += 1
            if stale > rewire_threshold:
                rw = []
                net._rewire(rw)
                rw_s = evaluate(net, inputs, targets, ticks=ticks)
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
# Main
# ============================================================

SEED = 42
N_TRAIN = 300
N_TEST = 100
BUDGET = 25000
TICKS = 8

print("=" * 72)
print("RAVEN MATRICES v2 — COMPACT ENCODING")
print(f"Shapes={N_S} × Counts={N_C} = {N_ANSWERS} answers | Random={100/N_ANSWERS:.1f}%")
print("=" * 72)

# Show examples
print("\nExample 2×2:")
random.seed(0)
g = puzzle_2x2_row_shape_col_count()
print(f"  {cell_str(g[0][0])} {cell_str(g[0][1])}")
print(f"  {cell_str(g[1][0])}  ?   → {cell_str(g[1][1])}")
print()

all_puzzles = [(name, fn, 2) for name, fn in PUZZLES_2X2] + \
              [(name, fn, 3) for name, fn in PUZZLES_3X3]

print(f"{'Puzzle':<22s} {'Grid':>4s} {'V':>4s} {'N':>5s} "
      f"{'Train':>7s} {'Test':>7s} {'Rand':>5s} {'Steps':>6s} {'Time':>5s}")
print("-" * 78)

results = []

for name, gen_fn, grid_size in all_puzzles:
    np.random.seed(SEED)
    random.seed(SEED)

    train_in, train_tgt = build_dataset(gen_fn, N_TRAIN, grid_size, seed=SEED)
    test_in, test_tgt = build_dataset(gen_fn, N_TEST, grid_size, seed=SEED + 999)

    V = train_in.shape[1]
    net = SelfWiringGraph(V)

    random_base = 1.0 / N_ANSWERS

    t0 = time.time()
    best, steps = train_puzzle(net, train_in, train_tgt, BUDGET, TICKS)
    elapsed = time.time() - t0

    train_acc = accuracy_full(net, train_in, train_tgt, TICKS)
    test_acc = accuracy_full(net, test_in, test_tgt, TICKS)

    marker = ""
    if test_acc > random_base * 2: marker = " *"
    if test_acc > 0.5: marker = " **"
    if test_acc > 0.8: marker = " ***"

    gs = f"{grid_size}×{grid_size}"
    print(f"{name:<22s} {gs:>4s} {V:4d} {net.N:5d} "
          f"{train_acc*100:6.1f}% {test_acc*100:6.1f}% "
          f"{random_base*100:4.1f}% {steps:6d} {elapsed:4.0f}s{marker}", flush=True)

    results.append({'name': name, 'grid': grid_size, 'train': train_acc,
                    'test': test_acc, 'V': V})

# --- GENERALIZATION TEST ---
# Train on 2x2 row=shape, test on UNSEEN shape/count combos
print()
print("=" * 72)
print("GENERALIZATION: train on some shape combos, test on held-out combos")
print("=" * 72)

np.random.seed(SEED)
random.seed(SEED)

# Generate all possible 2x2 row=shape puzzles (3C2 shapes × 3C2 counts = 3×3 = 9 combos)
all_grids = []
for s0 in range(N_S):
    for s1 in range(N_S):
        if s0 == s1: continue
        for c0 in range(N_C):
            for c1 in range(N_C):
                if c0 == c1: continue
                g = [[cell_id(s0, c0), cell_id(s0, c1)],
                     [cell_id(s1, c0), cell_id(s1, c1)]]
                all_grids.append(g)

random.shuffle(all_grids)
split = int(len(all_grids) * 0.6)
train_grids = all_grids[:split]
test_grids = all_grids[split:]

def grids_to_arrays(grids):
    n = len(grids)
    V = 3 * N_ANSWERS  # 3 context cells
    inputs = np.zeros((n, V), dtype=np.float32)
    targets = np.zeros(n, dtype=np.int32)
    for i, g in enumerate(grids):
        context = [g[0][0], g[0][1], g[1][0]]
        for j, cid in enumerate(context):
            inputs[i, j * N_ANSWERS + cid] = 1.0
        targets[i] = g[1][1]
    return inputs, targets

gen_train_in, gen_train_tgt = grids_to_arrays(train_grids)
gen_test_in, gen_test_tgt = grids_to_arrays(test_grids)

V = gen_train_in.shape[1]
net = SelfWiringGraph(V)

t0 = time.time()
best, steps = train_puzzle(net, gen_train_in, gen_train_tgt, BUDGET, TICKS)
elapsed = time.time() - t0

tr_acc = accuracy_full(net, gen_train_in, gen_train_tgt, TICKS)
te_acc = accuracy_full(net, gen_test_in, gen_test_tgt, TICKS)

print(f"  All combos: {len(all_grids)} | Train: {len(train_grids)} | Test: {len(test_grids)}")
print(f"  Train acc: {tr_acc*100:.1f}% | Test acc: {te_acc*100:.1f}% | Random: {100/N_ANSWERS:.1f}%")
print(f"  Steps: {steps} | Time: {elapsed:.0f}s")

if te_acc > 0.8:
    print("  → GENERALIZES to unseen combos!")
elif te_acc > 0.3:
    print("  → Partial generalization — learned some structure.")
else:
    print("  → Memorized, doesn't generalize to new combos.")

print()
print("=" * 72)
print("SUMMARY")
print("=" * 72)
solved_2x2 = sum(1 for r in results if r['grid'] == 2 and r['test'] > 0.5)
solved_3x3 = sum(1 for r in results if r['grid'] == 3 and r['test'] > 0.5)
print(f"2×2 solved: {solved_2x2}/{sum(1 for r in results if r['grid']==2)}")
print(f"3×3 solved: {solved_3x3}/{sum(1 for r in results if r['grid']==3)}")
print(f"Generalization: {'YES' if te_acc > 0.5 else 'NO'} ({te_acc*100:.1f}%)")
