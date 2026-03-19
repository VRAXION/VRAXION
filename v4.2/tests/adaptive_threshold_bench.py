"""Adaptive Threshold Benchmark
=================================
Tests 4 adaptive threshold strategies against the fixed-threshold baseline.

Strategies:
  1. RAMP     — start low, linearly ramp to final threshold over ticks
  2. ACTIVITY — threshold = base + alpha * mean_activity (auto-regulation)
  3. PERCENTILE — threshold = P-th percentile of positive charges
  4. HOMEOSTATIC — target K active neurons, adjust threshold each tick

Each is tested on DIAG and ROW puzzles with 3 seeds, 20k budget, ticks=8 and 32.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random as pyrandom
from model.graph import SelfWiringGraph

# ── Puzzle definitions ──
N_S, N_C = 3, 3
N_ANSWERS = N_S * N_C
V_IN = 3 * N_ANSWERS  # 27

def cell_id(s, c):
    return s * N_C + c

def puzzle_2x2_diagonal():
    vals = pyrandom.sample(range(N_ANSWERS), 2)
    return [[vals[0], vals[1]], [vals[1], vals[0]]]

def puzzle_2x2_row_shape():
    s = pyrandom.sample(range(N_S), 2)
    c = pyrandom.sample(range(N_C), 2)
    return [[cell_id(s[0], c[0]), cell_id(s[0], c[1])],
            [cell_id(s[1], c[0]), cell_id(s[1], c[1])]]

def build_dataset(puzzle_fn, n, seed=42):
    old = pyrandom.getstate()
    pyrandom.seed(seed)
    inputs = np.zeros((n, V_IN), dtype=np.float32)
    targets = np.zeros(n, dtype=np.int32)
    for i in range(n):
        g = puzzle_fn()
        ctx = [g[0][0], g[0][1], g[1][0]]
        for j, cid in enumerate(ctx):
            inputs[i, j * N_ANSWERS + cid] = 1.0
        targets[i] = g[1][1]
    pyrandom.setstate(old)
    return inputs, targets


# ══════════════════════════════════════════════════════════════
#  ADAPTIVE FORWARD PASSES
# ══════════════════════════════════════════════════════════════

def forward_fixed(net, patterns, ticks, thresh):
    """Baseline: fixed threshold."""
    K, N, V = patterns.shape[0], net.N, net.V
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
        acts = np.maximum(charges - thresh, 0.0)
        charges = np.clip(charges, -1.0, 1.0)
    return charges[:, net.out_start:net.out_start + V]


def forward_ramp(net, patterns, ticks, thresh_lo, thresh_hi):
    """RAMP: linearly ramp threshold from thresh_lo to thresh_hi over ticks."""
    K, N, V = patterns.shape[0], net.N, net.V
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
        # Linear ramp
        frac = t / max(ticks - 1, 1)
        thresh_t = thresh_lo + (thresh_hi - thresh_lo) * frac
        acts = np.maximum(charges - thresh_t, 0.0)
        charges = np.clip(charges, -1.0, 1.0)
    return charges[:, net.out_start:net.out_start + V]


def forward_activity(net, patterns, ticks, base_thresh, alpha):
    """ACTIVITY: threshold = base + alpha * mean_activity across batch.
    When network is very active, threshold rises → suppresses noise.
    When quiet, threshold drops → lets signal through."""
    K, N, V = patterns.shape[0], net.N, net.V
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
        # Adaptive threshold based on mean activity
        mean_act = float(acts.mean()) if t > 0 else 0.0
        thresh_t = base_thresh + alpha * mean_act
        acts = np.maximum(charges - thresh_t, 0.0)
        charges = np.clip(charges, -1.0, 1.0)
    return charges[:, net.out_start:net.out_start + V]


def forward_percentile(net, patterns, ticks, pct):
    """PERCENTILE: threshold = pct-th percentile of positive charges.
    Relative threshold — always lets top (100-pct)% of neurons fire."""
    K, N, V = patterns.shape[0], net.N, net.V
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
        # Percentile-based threshold per sample
        pos = charges.copy()
        pos[pos <= 0] = 0
        # Per-sample percentile
        thresh_arr = np.percentile(pos, pct, axis=1, keepdims=True)
        acts = np.maximum(charges - thresh_arr, 0.0)
        charges = np.clip(charges, -1.0, 1.0)
    return charges[:, net.out_start:net.out_start + V]


def forward_homeostatic(net, patterns, ticks, target_frac):
    """HOMEOSTATIC: target a fraction of neurons active.
    Adjusts threshold each tick to hit target_frac active neurons."""
    K, N, V = patterns.shape[0], net.N, net.V
    charges = np.zeros((K, N), dtype=np.float32)
    acts = np.zeros((K, N), dtype=np.float32)
    retain = float(net.retention)
    target_k = max(1, int(N * target_frac))
    for t in range(ticks):
        if t == 0:
            acts[:, :V] = patterns
        raw = acts @ net.mask
        np.nan_to_num(raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        charges += raw
        charges *= retain
        # Per-sample: find threshold so that ~target_k neurons are active
        # Use kth-largest charge as threshold
        # np.partition is O(N) — fast
        sorted_charges = np.sort(charges, axis=1)[:, ::-1]  # descending
        k_idx = min(target_k, N - 1)
        thresh_arr = sorted_charges[:, k_idx:k_idx+1]
        thresh_arr = np.maximum(thresh_arr, 0.0)  # never negative threshold
        acts = np.maximum(charges - thresh_arr, 0.0)
        charges = np.clip(charges, -1.0, 1.0)
    return charges[:, net.out_start:net.out_start + V]


# ══════════════════════════════════════════════════════════════
#  TRAINING HARNESS
# ══════════════════════════════════════════════════════════════

def evaluate(net, inputs, targets, ticks, fwd_fn, fwd_kwargs, sample=128):
    idx = np.random.choice(len(inputs), min(sample, len(inputs)), replace=False)
    logits = fwd_fn(net, inputs[idx], ticks, **fwd_kwargs)[:, :N_ANSWERS]
    return float((np.argmax(logits, axis=1) == targets[idx]).mean())


def accuracy_full(net, inputs, targets, ticks, fwd_fn, fwd_kwargs):
    correct = 0
    for s in range(0, len(inputs), 256):
        e = min(s + 256, len(inputs))
        logits = fwd_fn(net, inputs[s:e], ticks, **fwd_kwargs)[:, :N_ANSWERS]
        correct += (np.argmax(logits, axis=1) == targets[s:e]).sum()
    return correct / len(inputs)


def train_puzzle(net, inputs, targets, budget, ticks, fwd_fn, fwd_kwargs):
    stale_limit = budget // 2
    rewire_threshold = stale_limit // 3
    score = evaluate(net, inputs, targets, ticks, fwd_fn, fwd_kwargs)
    best = score
    stale = 0
    for att in range(budget):
        old_l = int(net.loss_pct)
        old_d = int(net.drive)
        undo = net.mutate()
        new = evaluate(net, inputs, targets, ticks, fwd_fn, fwd_kwargs)
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
                rw_s = evaluate(net, inputs, targets, ticks, fwd_fn, fwd_kwargs)
                if rw_s > score:
                    score = rw_s
                    best = max(best, score)
                    stale = 0
                else:
                    net.replay(rw)
        if best >= 0.99 or stale >= stale_limit:
            break
    return best, att + 1


def make_net(seed, drive=0.6):
    """Create network with given seed and drive."""
    np.random.seed(seed)
    pyrandom.seed(seed)
    net = SelfWiringGraph(V_IN)
    d_pct = net.DENSITY / 100
    r_mat = np.random.rand(net.N, net.N)
    net.mask = np.zeros((net.N, net.N), dtype=np.float32)
    net.mask[r_mat < d_pct / 2] = -drive
    net.mask[r_mat > 1 - d_pct / 2] = drive
    np.fill_diagonal(net.mask, 0)
    net.resync_alive()
    return net


# ══════════════════════════════════════════════════════════════
#  CONFIG DEFINITIONS
# ══════════════════════════════════════════════════════════════

CONFIGS = [
    # (label, drive, loss_pct, fwd_fn, fwd_kwargs)
    ("BASELINE fixed=0.5",     0.6, 15, forward_fixed,       {"thresh": 0.5}),
    ("low-thresh fixed=0.1",   0.6, 15, forward_fixed,       {"thresh": 0.1}),
    ("RAMP 0.05→0.5",         0.6, 15, forward_ramp,         {"thresh_lo": 0.05, "thresh_hi": 0.5}),
    ("RAMP 0.0→0.3",          0.6, 15, forward_ramp,         {"thresh_lo": 0.0,  "thresh_hi": 0.3}),
    ("RAMP 0.1→0.4",          0.6, 15, forward_ramp,         {"thresh_lo": 0.1,  "thresh_hi": 0.4}),
    ("ACTIVITY b=0.1 a=2.0",   0.6, 15, forward_activity,    {"base_thresh": 0.1, "alpha": 2.0}),
    ("ACTIVITY b=0.05 a=3.0",  0.6, 15, forward_activity,    {"base_thresh": 0.05, "alpha": 3.0}),
    ("ACTIVITY b=0.1 a=5.0",   0.6, 15, forward_activity,    {"base_thresh": 0.1, "alpha": 5.0}),
    ("PCTILE 80",              0.6, 15, forward_percentile,   {"pct": 80}),
    ("PCTILE 90",              0.6, 15, forward_percentile,   {"pct": 90}),
    ("PCTILE 95",              0.6, 15, forward_percentile,   {"pct": 95}),
    ("HOMEO 10%",              0.6, 15, forward_homeostatic,  {"target_frac": 0.10}),
    ("HOMEO 5%",               0.6, 15, forward_homeostatic,  {"target_frac": 0.05}),
    ("HOMEO 15%",              0.6, 15, forward_homeostatic,  {"target_frac": 0.15}),
    # Also test adaptive + higher drive combos
    ("RAMP 0.05→0.3 D=1.0",  1.0, 10, forward_ramp,         {"thresh_lo": 0.05, "thresh_hi": 0.3}),
    ("ACTIVITY b=0.1 a=3 D=1.0", 1.0, 10, forward_activity,  {"base_thresh": 0.1, "alpha": 3.0}),
    ("PCTILE 90 D=1.0",       1.0, 10, forward_percentile,   {"pct": 90}),
    ("HOMEO 10% D=1.0",       1.0, 10, forward_homeostatic,  {"target_frac": 0.10}),
]


# ══════════════════════════════════════════════════════════════
#  MAIN BENCHMARK
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    N_TRAIN, N_TEST = 300, 100
    BUDGET = 20000
    SEEDS = [42, 123, 777]

    # Build datasets
    train_diag, tgt_diag = build_dataset(puzzle_2x2_diagonal, N_TRAIN, 42)
    test_diag, tgt_diag_te = build_dataset(puzzle_2x2_diagonal, N_TEST, 999)
    train_row, tgt_row = build_dataset(puzzle_2x2_row_shape, N_TRAIN, 42)
    test_row, tgt_row_te = build_dataset(puzzle_2x2_row_shape, N_TEST, 999)

    puzzles = [
        ("DIAGONAL", train_diag, tgt_diag, test_diag, tgt_diag_te),
        ("ROW=SHAPE", train_row, tgt_row, test_row, tgt_row_te),
    ]

    for ticks in [8, 32]:
        print(f"\n{'='*95}")
        print(f"  ticks={ticks} | budget={BUDGET} | {len(SEEDS)} seeds")
        print(f"{'='*95}")
        print(f"  {'Label':<28s} | {'DiagTr':>7s} {'DiagTe':>7s} | "
              f"{'RowTr':>7s} {'RowTe':>7s} | {'Time':>5s}")
        print(f"  {'-'*80}")

        for label, drive, loss_pct, fwd_fn, fwd_kwargs in CONFIGS:
            diag_tr_all, diag_te_all = [], []
            row_tr_all, row_te_all = [], []
            t0 = time.time()

            for seed in SEEDS:
                for pname, tr_in, tr_tgt, te_in, te_tgt in puzzles:
                    net = make_net(seed, drive=drive)
                    net.loss_pct = np.int8(loss_pct)

                    train_puzzle(net, tr_in, tr_tgt, BUDGET, ticks, fwd_fn, fwd_kwargs)
                    tr_acc = accuracy_full(net, tr_in, tr_tgt, ticks, fwd_fn, fwd_kwargs)
                    te_acc = accuracy_full(net, te_in, te_tgt, ticks, fwd_fn, fwd_kwargs)

                    if pname == "DIAGONAL":
                        diag_tr_all.append(tr_acc)
                        diag_te_all.append(te_acc)
                    else:
                        row_tr_all.append(tr_acc)
                        row_te_all.append(te_acc)

            elapsed = time.time() - t0
            d_tr = np.mean(diag_tr_all) * 100
            d_te = np.mean(diag_te_all) * 100
            r_tr = np.mean(row_tr_all) * 100
            r_te = np.mean(row_te_all) * 100
            marker = " <<<" if "BASELINE" in label else ""
            print(f"  {label:<28s} | {d_tr:6.1f}% {d_te:6.1f}% | "
                  f"{r_tr:6.1f}% {r_te:6.1f}% | {elapsed:4.0f}s{marker}",
                  flush=True)

    print(f"\n{'='*95}")
    print("DONE")
    print(f"{'='*95}")
