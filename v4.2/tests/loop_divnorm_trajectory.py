"""Learning curves: retain vs divnorm vs both at different tick counts.
Longer budget (24k) with trajectory to see who plateaus when."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
from model.graph import SelfWiringGraph

V = 32
BUDGET = 24000
SEED = 42
ALPHA = 2.0
LOG_EVERY = 2000


def forward_divnorm(net, ticks, alpha=ALPHA):
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


def forward_both(net, ticks, alpha=ALPHA):
    V, N = net.V, net.N
    charges = np.zeros((V, N), dtype=np.float32)
    acts = np.zeros((V, N), dtype=np.float32)
    retain = float(net.retention)
    for t in range(ticks):
        if t == 0:
            acts[:, :V] = np.eye(V, dtype=np.float32)
        raw = acts @ net.mask
        np.nan_to_num(raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        charges += raw
        charges *= retain
        total = np.abs(acts).sum(axis=1, keepdims=True) + 1e-6
        acts = np.maximum(charges - net.THRESHOLD, 0.0)
        acts /= (1.0 + alpha * total)
        charges = np.clip(charges, -1.0, 1.0)
    return charges[:, net.out_start:net.out_start + V]


def train_traj(net, targets, budget, forward_fn, ticks):
    def evaluate():
        logits = forward_fn(net, ticks)
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
        V = net.V
        acc = (np.argmax(probs, axis=1)[:V] == targets[:V]).mean()
        tp = probs[np.arange(V), targets[:V]].mean()
        return 0.5 * acc + 0.5 * tp

    score = evaluate()
    best = score
    stale = 0
    trajectory = [(0, best)]

    for att in range(budget):
        old_loss = int(net.loss_pct)
        old_drive = int(net.drive)
        undo = net.mutate()
        new_score = evaluate()
        if new_score > score:
            score = new_score
            best = max(best, score)
            stale = 0
        else:
            net.replay(undo)
            net.loss_pct = np.int8(old_loss)
            net.drive = np.int8(old_drive)
            stale += 1
        if (att + 1) % LOG_EVERY == 0:
            trajectory.append((att + 1, best))
        if best >= 0.99 or stale >= 6000:
            break

    if (att + 1) % LOG_EVERY != 0:
        trajectory.append((att + 1, best))
    return best, att + 1, trajectory


CONFIGS = [
    ('retain_4t',  lambda n, t: n.forward_batch(t), 4),
    ('retain_8t',  lambda n, t: n.forward_batch(t), 8),
    ('divnorm_8t', forward_divnorm, 8),
    ('divnorm_16t', forward_divnorm, 16),
    ('both_8t',    forward_both, 8),
    ('both_16t',   forward_both, 16),
]

print(f"LEARNING CURVES | V={V} budget={BUDGET} seed={SEED}")
print(f"{'config':<14s} {'score':>7s} {'steps':>6s} {'time':>5s}  trajectory (every {LOG_EVERY})")
print("-" * 100)

all_trajs = {}

for name, forward_fn, ticks in CONFIGS:
    np.random.seed(SEED)
    random.seed(SEED)
    net = SelfWiringGraph(V)
    targets = np.arange(V)
    np.random.shuffle(targets)

    random.seed(SEED * 1000 + 1)
    t0 = time.time()
    best, steps, traj = train_traj(net, targets, BUDGET, forward_fn, ticks)
    elapsed = time.time() - t0
    all_trajs[name] = traj

    traj_str = " → ".join(f"{b*100:.1f}" for _, b in traj)
    print(f"{name:<14s} {best*100:6.1f}% {steps:6d} {t0:4.0f}s  {traj_str}", flush=True)

# Comparison table
print(f"\n{'='*90}")
print(f"TRAJECTORY TABLE (best % at each checkpoint)")
print(f"{'step':>6s}", end="")
for name, _, _ in CONFIGS:
    print(f" {name:>14s}", end="")
print()
print("-" * (6 + 15 * len(CONFIGS)))

checkpoints = [i * LOG_EVERY for i in range(BUDGET // LOG_EVERY + 1)]
for cp in checkpoints:
    print(f"{cp:6d}", end="")
    for name, _, _ in CONFIGS:
        traj = all_trajs[name]
        val = traj[0][1]
        for step, b in traj:
            if step <= cp:
                val = b
        print(f" {val*100:13.1f}%", end="")
    print()

# Show delta from previous checkpoint (learning speed)
print(f"\n{'='*90}")
print("LEARNING SPEED (pp gained per 2k steps)")
print(f"{'step':>6s}", end="")
for name, _, _ in CONFIGS:
    print(f" {name:>14s}", end="")
print()
print("-" * (6 + 15 * len(CONFIGS)))

for i, cp in enumerate(checkpoints):
    if i == 0:
        continue
    print(f"{cp:6d}", end="")
    for name, _, _ in CONFIGS:
        traj = all_trajs[name]
        prev_val = traj[0][1]
        curr_val = traj[0][1]
        for step, b in traj:
            if step <= checkpoints[i - 1]:
                prev_val = b
            if step <= cp:
                curr_val = b
        delta = (curr_val - prev_val) * 100
        marker = " ***" if delta > 1.0 else ""
        print(f" {delta:+12.1f}pp", end="")
    print()
