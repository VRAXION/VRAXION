"""Signal Gain Deep Test — focused on promising combos with bigger budget.

Key finding from sweep: lower threshold helps row=shape puzzle significantly
when combined with more ticks. Now test with 20k budget and multiple seeds.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
from model.graph import SelfWiringGraph


N_S, N_C = 3, 3
N_ANSWERS = N_S * N_C
V_IN = 3 * N_ANSWERS  # 27


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
    inputs = np.zeros((n, V_IN), dtype=np.float32)
    targets = np.zeros(n, dtype=np.int32)
    for i in range(n):
        g = puzzle_fn()
        ctx = [g[0][0], g[0][1], g[1][0]]
        for j, cid in enumerate(ctx):
            inputs[i, j * N_ANSWERS + cid] = 1.0
        targets[i] = g[1][1]
    random.setstate(old)
    return inputs, targets


def forward_batch(net, patterns, ticks, thresh):
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
        acts = np.maximum(charges - thresh, 0.0)
        charges = np.clip(charges, -1.0, 1.0)
    return charges[:, net.out_start:net.out_start + V]


def evaluate(net, inputs, targets, ticks, thresh, sample=128):
    idx = np.random.choice(len(inputs), min(sample, len(inputs)), replace=False)
    logits = forward_batch(net, inputs[idx], ticks, thresh)[:, :N_ANSWERS]
    return float((np.argmax(logits, axis=1) == targets[idx]).mean())


def accuracy_full(net, inputs, targets, ticks, thresh):
    correct = 0
    for s in range(0, len(inputs), 256):
        e = min(s + 256, len(inputs))
        logits = forward_batch(net, inputs[s:e], ticks, thresh)[:, :N_ANSWERS]
        correct += (np.argmax(logits, axis=1) == targets[s:e]).sum()
    return correct / len(inputs)


def make_net(drive, loss_pct, seed):
    """Build a SelfWiringGraph with custom DRIVE in mask."""
    np.random.seed(seed)
    random.seed(seed)
    net = SelfWiringGraph(V_IN)
    # Rebuild mask with custom drive
    N = net.N
    d_pct = net.DENSITY / 100
    r_mat = np.random.rand(N, N)
    net.mask = np.zeros((N, N), dtype=np.float32)
    net.mask[r_mat < d_pct / 2] = -drive
    net.mask[r_mat > 1 - d_pct / 2] = drive
    np.fill_diagonal(net.mask, 0)
    net.resync_alive()
    net.loss_pct = np.int8(loss_pct)
    return net


def train_puzzle(net, inputs, targets, budget, ticks, thresh):
    stale_limit = budget // 2
    rewire_threshold = stale_limit // 3
    score = evaluate(net, inputs, targets, ticks, thresh)
    best = score
    stale = 0
    for att in range(budget):
        old_l = int(net.loss_pct)
        old_d = int(net.drive)
        undo = net.mutate()
        new = evaluate(net, inputs, targets, ticks, thresh)
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
                rw_s = evaluate(net, inputs, targets, ticks, thresh)
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
# EXPERIMENT: Focused combos, 3 seeds, 20k budget
# ============================================================

N_TRAIN, N_TEST = 300, 100
BUDGET = 20000
SEEDS = [42, 123, 777]

# Combos: (DRIVE, THRESHOLD, loss_pct, label)
# loss_pct: 15 → ret=0.85, 10 → ret=0.90, 5 → ret=0.95
COMBOS = [
    (0.6, 0.50, 15, "BASELINE"),
    (0.6, 0.10, 15, "low-thresh"),
    (0.6, 0.00, 15, "zero-thresh"),
    (1.0, 0.10, 10, "D1.0/T.1/R.9"),
    (1.0, 0.05, 15, "D1.0/T.05/R.85"),
    (1.0, 0.05, 10, "D1.0/T.05/R.9"),
    (1.2, 0.10, 10, "D1.2/T.1/R.9"),
    (1.2, 0.20, 10, "D1.2/T.2/R.9"),
    (1.5, 0.30, 10, "D1.5/T.3/R.9"),
]

for puzzle_name, puzzle_fn in [("DIAGONAL", puzzle_2x2_diagonal),
                                ("ROW=SHAPE", puzzle_2x2_row_shape)]:
    for ticks in [8, 32]:
        print()
        print("=" * 85)
        print(f"  {puzzle_name} puzzle | ticks={ticks} | budget={BUDGET} | {len(SEEDS)} seeds")
        print("=" * 85)
        print(f"  {'Label':>18s} | {'Seed':>4s} {'Train':>7s} {'Test':>7s} {'Steps':>6s} {'Time':>5s} | "
              f"{'AvgTrain':>8s} {'AvgTest':>8s}")
        print("  " + "-" * 78)

        for drive, thresh, lpct, label in COMBOS:
            train_accs = []
            test_accs = []
            for seed in SEEDS:
                train_in, train_tgt = build_dataset(puzzle_fn, N_TRAIN, seed)
                test_in, test_tgt = build_dataset(puzzle_fn, N_TEST, seed + 10000)

                net = make_net(drive, lpct, seed)

                t0 = time.time()
                best, steps = train_puzzle(net, train_in, train_tgt, BUDGET, ticks, thresh)
                elapsed = time.time() - t0
                tr = accuracy_full(net, train_in, train_tgt, ticks, thresh)
                te = accuracy_full(net, test_in, test_tgt, ticks, thresh)
                train_accs.append(tr)
                test_accs.append(te)

                is_last = (seed == SEEDS[-1])
                avg_tr = np.mean(train_accs)
                avg_te = np.mean(test_accs)
                suffix = f"  {avg_tr*100:7.1f}% {avg_te*100:7.1f}%" if is_last else ""
                marker = " <<<" if is_last and label == "BASELINE" else ""
                print(f"  {label:>18s} | {seed:4d} {tr*100:6.1f}% {te*100:6.1f}% "
                      f"{steps:6d} {elapsed:4.0f}s |{suffix}{marker}", flush=True)


# ============================================================
# SIGNAL TRACE on a TRAINED network
# ============================================================

print()
print("=" * 85)
print("BONUS: Signal trace on a TRAINED network (best combo vs baseline)")
print("=" * 85)

for drive, thresh, lpct, label in [(0.6, 0.5, 15, "BASELINE"),
                                     (1.0, 0.05, 10, "BEST?")]:
    train_in, train_tgt = build_dataset(puzzle_2x2_row_shape, N_TRAIN, 42)
    net = make_net(drive, lpct, 42)
    train_puzzle(net, train_in, train_tgt, BUDGET, 32, thresh)

    # Now trace with actual Raven input
    x = train_in[0:1]
    N = net.N
    V = net.V
    charges = np.zeros((1, N), dtype=np.float32)
    acts = np.zeros((1, N), dtype=np.float32)
    retain = float(net.retention)

    print(f"\n  {label}: DRIVE={drive} THRESH={thresh} RET={retain:.2f} conns={net.count_connections()}")
    print(f"  {'Tick':>6s} {'Active':>7s} {'MeanAct':>8s} {'MaxAct':>8s} {'OutAct':>7s} {'OutMax':>8s}")

    for t in range(48):
        if t == 0:
            acts[:, :V] = x
        raw = acts @ net.mask
        np.nan_to_num(raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        charges += raw
        charges *= retain
        acts = np.maximum(charges - thresh, 0.0)
        charges = np.clip(charges, -1.0, 1.0)

        n_active = int((acts[0] > 0).sum())
        mean_act = float(acts[0][acts[0] > 0].mean()) if n_active > 0 else 0
        max_act = float(acts[0].max())
        out_s = acts[0, net.out_start:net.out_start + V]
        out_a = int((out_s > 0).sum())
        out_m = float(out_s.max())

        if t < 12 or t % 8 == 0:
            print(f"  {t:6d} {n_active:7d} {mean_act:8.4f} {max_act:8.4f} {out_a:7d} {out_m:8.4f}")

print()
print("=" * 85)
print("DONE")
print("=" * 85)
