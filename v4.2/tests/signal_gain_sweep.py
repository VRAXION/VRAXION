"""Signal Gain Sweep — Step 1+2 of the propagation fix plan.

1) Analytical gain calculation per hop for every DRIVE×THRESHOLD×retention combo
2) Simulated signal trace: inject input, watch activation spread over 64 ticks
3) Raven puzzle accuracy for the most promising combos (quick 5k budget)

Goal: find combos where gain ≈ 1.0 (stable propagation, no explosion, no death)
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
from model.graph import SelfWiringGraph


# ============================================================
# STEP 1: ANALYTICAL GAIN PER HOP
# ============================================================
# Single-neuron model:
#   charge = signal_in * DRIVE * retention
#   act = max(charge - THRESHOLD, 0)
#   gain = act / signal_in
#
# For signal_in = 1.0 (first hop from input):
#   gain = max(DRIVE * retention - THRESHOLD, 0)
#
# For subsequent hops, signal_in = previous gain:
#   gain_n = max(gain_{n-1} * DRIVE * retention - THRESHOLD, 0)

print("=" * 80)
print("STEP 1: ANALYTICAL GAIN PER HOP")
print("=" * 80)

DRIVES = [0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0]
THRESHOLDS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
RETENTIONS = [0.85, 0.90, 0.95]

print(f"\n{'DRIVE':>5s} {'THRESH':>6s} {'RET':>5s} | {'Hop1':>6s} {'Hop2':>6s} "
      f"{'Hop3':>6s} {'Hop4':>6s} {'Hop5':>6s} {'Hop8':>6s} | {'Verdict':>12s}")
print("-" * 85)

good_combos = []

for drive in DRIVES:
    for thresh in THRESHOLDS:
        for ret in RETENTIONS:
            gains = []
            sig = 1.0
            for hop in range(8):
                charge = sig * drive * ret
                act = max(charge - thresh, 0.0)
                gains.append(act)
                sig = act
                if sig < 1e-6:
                    gains.extend([0.0] * (7 - hop))
                    break

            # Verdict
            if gains[3] > 0.1 and gains[7] < 100:
                verdict = "GOOD"
                good_combos.append((drive, thresh, ret, gains))
            elif gains[7] > 100:
                verdict = "EXPLODES"
            elif gains[1] < 0.01:
                verdict = "DIES@2"
            elif gains[3] < 0.01:
                verdict = "DIES@4"
            else:
                verdict = "weak"

            # Only print interesting ones
            if verdict in ("GOOD", "EXPLODES") or (drive == 0.6 and thresh == 0.5):
                print(f"{drive:5.1f} {thresh:6.2f} {ret:5.2f} | "
                      f"{gains[0]:6.3f} {gains[1]:6.3f} {gains[2]:6.3f} "
                      f"{gains[3]:6.3f} {gains[4]:6.3f} {gains[7]:6.3f} | "
                      f"{verdict:>12s}")

# Show current baseline
print(f"\n  >>> CURRENT: DRIVE=0.6 THRESH=0.5 RET=0.85 → signal dies at hop 2")
print(f"  >>> Found {len(good_combos)} GOOD combos (signal survives 4+ hops, doesn't explode)")

# Rank good combos by hop8 stability (closest to reasonable signal level)
good_combos.sort(key=lambda x: abs(x[3][4] - 0.3))  # target ~0.3 at hop5
print(f"\n  TOP 10 most stable combos (hop5 signal ≈ 0.3):")
for i, (d, t, r, g) in enumerate(good_combos[:10]):
    print(f"    {i+1}. DRIVE={d:.1f} THRESH={t:.2f} RET={r:.2f} | "
          f"hops: {g[0]:.3f} → {g[1]:.3f} → {g[2]:.3f} → {g[3]:.3f} → {g[4]:.3f} → ... → {g[7]:.3f}")


# ============================================================
# STEP 2: SIMULATED SIGNAL TRACE (top combos)
# ============================================================

print()
print("=" * 80)
print("STEP 2: SIMULATED SIGNAL TRACE — real network, 64 ticks")
print("=" * 80)

# Pick the top 5 combos + baseline
test_combos = []
# Always include baseline
test_combos.append((0.6, 0.5, 0.85, "BASELINE"))
# Add top good combos (deduplicate)
seen = set()
for d, t, r, g in good_combos[:8]:
    key = (d, t, r)
    if key not in seen:
        seen.add(key)
        test_combos.append((d, t, r, "CANDIDATE"))
    if len(test_combos) >= 8:
        break

# Build Raven-like input
N_S, N_C = 3, 3
N_ANSWERS = N_S * N_C
V_in = 3 * N_ANSWERS  # 27

for drive, thresh, ret, label in test_combos:
    np.random.seed(42)
    random.seed(42)

    # Build network with custom params
    net = SelfWiringGraph(V_in)
    N = net.N
    V = net.V

    # Override mask with custom DRIVE
    d_pct = net.DENSITY / 100
    r_mat = np.random.rand(N, N)
    net.mask = np.zeros((N, N), dtype=np.float32)
    net.mask[r_mat < d_pct / 2] = -drive
    net.mask[r_mat > 1 - d_pct / 2] = drive
    np.fill_diagonal(net.mask, 0)
    net.resync_alive()

    # Compute loss_pct from retention
    loss_pct = int(round((1 - ret) * 100))
    loss_pct = max(1, min(50, loss_pct))

    # Single input: one-hot cell 0
    x = np.zeros((1, V), dtype=np.float32)
    x[0, 0] = 1.0

    charges = np.zeros((1, N), dtype=np.float32)
    acts = np.zeros((1, N), dtype=np.float32)
    retain = (100 - loss_pct) / 100.0

    tick_data = []
    for t in range(64):
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
        out_slice = acts[0, net.out_start:net.out_start + V]
        out_active = int((out_slice > 0).sum())

        tick_data.append((t, n_active, mean_act, max_act, out_active))

    # Summary
    peak_active = max(d[1] for d in tick_data)
    first_output = -1
    for td in tick_data:
        if td[4] > 0:
            first_output = td[0]
            break
    last_active = -1
    for td in reversed(tick_data):
        if td[1] > 0:
            last_active = td[0]
            break
    stable_ticks = sum(1 for d in tick_data if d[1] > 0)

    print(f"\n  DRIVE={drive:.1f} THRESH={thresh:.2f} RET={retain:.2f} [{label}]")
    print(f"    Peak active: {peak_active:3d} neurons | "
          f"Output reached: tick {first_output} | "
          f"Active ticks: {stable_ticks}/64 | "
          f"Last active: tick {last_active}")

    # Print selected ticks
    print(f"    {'Tick':>6s} {'Active':>7s} {'MeanAct':>8s} {'MaxAct':>8s} {'OutAct':>7s}")
    for t, na, ma, mx, oa in tick_data:
        if t < 8 or t % 8 == 0 or (na > 0 and tick_data[min(t+1, 63)][1] == 0):
            print(f"    {t:6d} {na:7d} {ma:8.4f} {mx:8.4f} {oa:7d}")


# ============================================================
# STEP 3: QUICK RAVEN TEST — top combos vs baseline
# ============================================================

print()
print("=" * 80)
print("STEP 3: QUICK RAVEN TEST — 2x2 diagonal puzzle, 10k budget")
print("=" * 80)


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


N_TRAIN, N_TEST = 300, 100
BUDGET = 10000
train_in, train_tgt = build_dataset(puzzle_2x2_diagonal, N_TRAIN, 42)
test_in, test_tgt = build_dataset(puzzle_2x2_diagonal, N_TEST, 999)
train_in2, train_tgt2 = build_dataset(puzzle_2x2_row_shape, N_TRAIN, 42)
test_in2, test_tgt2 = build_dataset(puzzle_2x2_row_shape, N_TEST, 999)

# Filter test combos to promising ones only
raven_combos = [(0.6, 0.5, 0.85, "BASELINE")]
seen_r = {(0.6, 0.5, 0.85)}
for d, t, r, g in good_combos[:6]:
    key = (d, t, r)
    if key not in seen_r:
        seen_r.add(key)
        raven_combos.append((d, t, r, "CANDIDATE"))

print(f"\n{'DRIVE':>5s} {'THRESH':>6s} {'RET':>5s} | {'DiagTr':>7s} {'DiagTe':>7s} "
      f"{'RowTr':>7s} {'RowTe':>7s} | {'Ticks':>5s} {'Time':>5s}")
print("-" * 72)

for ticks in [8, 32]:
    for drive, thresh, ret, label in raven_combos:
        np.random.seed(42)
        random.seed(42)

        # Build network with custom params
        net = SelfWiringGraph(V_in)
        N = net.N

        # Override mask with custom DRIVE
        d_pct = net.DENSITY / 100
        r_mat = np.random.rand(N, N)
        net.mask = np.zeros((N, N), dtype=np.float32)
        net.mask[r_mat < d_pct / 2] = -drive
        net.mask[r_mat > 1 - d_pct / 2] = drive
        np.fill_diagonal(net.mask, 0)
        net.resync_alive()

        loss_pct = int(round((1 - ret) * 100))
        net.loss_pct = np.int8(max(1, min(50, loss_pct)))

        t0 = time.time()
        # Diagonal puzzle
        net_d = SelfWiringGraph(V_in)
        np.random.seed(42)
        random.seed(42)
        r_mat = np.random.rand(net_d.N, net_d.N)
        net_d.mask = np.zeros((net_d.N, net_d.N), dtype=np.float32)
        net_d.mask[r_mat < d_pct / 2] = -drive
        net_d.mask[r_mat > 1 - d_pct / 2] = drive
        np.fill_diagonal(net_d.mask, 0)
        net_d.resync_alive()
        net_d.loss_pct = np.int8(max(1, min(50, loss_pct)))

        best_d, _ = train_puzzle(net_d, train_in, train_tgt, BUDGET, ticks, thresh)
        tr_d = accuracy_full(net_d, train_in, train_tgt, ticks, thresh)
        te_d = accuracy_full(net_d, test_in, test_tgt, ticks, thresh)

        # Row=shape puzzle
        np.random.seed(42)
        random.seed(42)
        net_r = SelfWiringGraph(V_in)
        r_mat = np.random.rand(net_r.N, net_r.N)
        net_r.mask = np.zeros((net_r.N, net_r.N), dtype=np.float32)
        net_r.mask[r_mat < d_pct / 2] = -drive
        net_r.mask[r_mat > 1 - d_pct / 2] = drive
        np.fill_diagonal(net_r.mask, 0)
        net_r.resync_alive()
        net_r.loss_pct = np.int8(max(1, min(50, loss_pct)))

        best_r, _ = train_puzzle(net_r, train_in2, train_tgt2, BUDGET, ticks, thresh)
        tr_r = accuracy_full(net_r, train_in2, train_tgt2, ticks, thresh)
        te_r = accuracy_full(net_r, test_in2, test_tgt2, ticks, thresh)

        elapsed = time.time() - t0
        marker = " <<<" if label == "BASELINE" else ""
        print(f"{drive:5.1f} {thresh:6.2f} {ret:5.2f} | {tr_d*100:6.1f}% {te_d*100:6.1f}% "
              f"{tr_r*100:6.1f}% {te_r*100:6.1f}% | {ticks:5d} {elapsed:4.0f}s{marker}",
              flush=True)
    print()

print("=" * 80)
print("DONE — compare candidates vs baseline")
print("=" * 80)
