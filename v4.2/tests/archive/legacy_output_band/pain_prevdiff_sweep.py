"""Prev-diff pain sweep: inject PREVIOUS eval's diff at tick 0 with normal input.
Like brain pain memory — you remember the error, not compute it live.
Longer runs (100k budget) with trajectory tracking to see learning curves."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
from model.graph import SelfWiringGraph

V = 64
BUDGET = 100000
STALE = 15000
SEEDS = [0, 42, 123]
LOG_EVERY = 10000

CONDITIONS = {
    'baseline':        dict(pain_intensity=0.0, neuro_strength=0.0, inject='compute'),
    'prevdiff':        dict(pain_intensity=0.5, neuro_strength=0.0, inject='compute'),
    'prevdiff_lo':     dict(pain_intensity=0.2, neuro_strength=0.0, inject='compute'),
    'prevdiff_input':  dict(pain_intensity=0.5, neuro_strength=0.0, inject='input'),
    'prevdiff_neuro':  dict(pain_intensity=0.5, neuro_strength=1.0, inject='compute'),
}


def forward_batch_prevdiff(net, prev_diff=None, ticks=8, pain_intensity=0.0,
                           neuro_strength=0.0, inject='compute'):
    """Forward pass with previous eval's diff injected at tick 0."""
    V, N = net.V, net.N
    out_start = net.out_start
    charges = np.zeros((V, N), dtype=np.float32)
    acts = np.zeros((V, N), dtype=np.float32)
    retain = float(net.retention)
    threshold = net.THRESHOLD

    for t in range(ticks):
        if t == 0:
            acts[:, :V] = np.eye(V, dtype=np.float32)

            # Inject prev_diff at tick 0 — pain memory from last eval
            if pain_intensity > 0 and prev_diff is not None:
                if inject == 'compute':
                    charges[:, V:2*V] += prev_diff * pain_intensity
                elif inject == 'input':
                    charges[:, :V] += prev_diff * pain_intensity

            # Neuromod based on prev_diff magnitude
            if neuro_strength > 0 and prev_diff is not None:
                error_mag = np.abs(prev_diff).mean()
                threshold = net.THRESHOLD * (1.0 - neuro_strength * error_mag)
                threshold = max(0.05, threshold)
                retain = float(net.retention) * (1.0 + neuro_strength * error_mag * 0.5)
                retain = min(0.99, retain)

        raw = acts @ net.mask
        np.nan_to_num(raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        charges += raw
        charges *= retain
        acts = np.maximum(charges - threshold, 0.0)
        charges = np.clip(charges, -1.0, 1.0)

    # Compute current diff for next step's prev_diff
    out = charges[:, out_start:out_start + V]
    e = np.exp(out - out.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    current_diff = np.eye(V, dtype=np.float32) - probs

    return out, current_diff


def score_logits(logits, targets, V):
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    acc = (np.argmax(probs, axis=1)[:V] == targets[:V]).mean()
    tp = probs[np.arange(V), targets[:V]].mean()
    return 0.5 * acc + 0.5 * tp


def train_prevdiff(net, targets, budget, pain_intensity=0.0,
                   neuro_strength=0.0, inject='compute',
                   stale_limit=STALE, log_every=LOG_EVERY):
    # Initial eval — no pain memory yet
    logits, prev_diff = forward_batch_prevdiff(net, None, pain_intensity=0.0)
    score = score_logits(logits, targets, net.V)
    best = score
    stale = 0
    trajectory = [(0, best)]

    for att in range(budget):
        old_loss = int(net.loss_pct)
        old_drive = int(net.drive)
        old_diff = prev_diff.copy()

        undo = net.mutate()
        logits, new_diff = forward_batch_prevdiff(
            net, prev_diff, pain_intensity=pain_intensity,
            neuro_strength=neuro_strength, inject=inject)
        new_score = score_logits(logits, targets, net.V)

        if new_score > score:
            score = new_score
            best = max(best, score)
            prev_diff = new_diff
            stale = 0
        else:
            net.replay(undo)
            net.loss_pct = np.int8(old_loss)
            net.drive = np.int8(old_drive)
            prev_diff = old_diff  # revert diff too
            stale += 1

        if (att + 1) % log_every == 0:
            trajectory.append((att + 1, best))

        if best >= 0.99 or stale >= stale_limit:
            break

    # Final point
    if (att + 1) % log_every != 0:
        trajectory.append((att + 1, best))

    return best, att + 1, trajectory


# --- Main ---
print(f"PREV-DIFF PAIN SWEEP | V={V} budget={BUDGET} stale={STALE}")
print(f"{'condition':<16s} {'seed':>4s}  {'score':>7s} {'steps':>7s} {'time':>6s}  trajectory")
print("-" * 90)

results = {}
trajectories = {}

for name, kwargs in CONDITIONS.items():
    results[name] = []
    trajectories[name] = []
    for seed in SEEDS:
        np.random.seed(seed)
        random.seed(seed)
        net = SelfWiringGraph(V)
        targets = np.arange(V)
        np.random.shuffle(targets)

        random.seed(seed * 1000 + 1)
        t0 = time.time()
        best, steps, traj = train_prevdiff(net, targets, BUDGET, **kwargs)
        elapsed = time.time() - t0
        results[name].append(best)
        trajectories[name].append(traj)

        traj_str = " → ".join(f"{b*100:.1f}" for _, b in traj)
        print(f"{name:<16s} {seed:4d}  {best*100:6.1f}% {steps:7d} {elapsed:5.0f}s  {traj_str}",
              flush=True)

# Summary
print(f"\n{'='*70}")
print(f"PREV-DIFF SWEEP SUMMARY | V={V} budget={BUDGET}")
print(f"{'condition':<16s} {'mean':>7s} {'std':>6s} {'vs_base':>8s}  per-seed")
print("-" * 70)
base_mean = np.mean(results['baseline']) * 100
best_mean = -1
best_name = ''
for name in CONDITIONS:
    scores = results[name]
    m = np.mean(scores) * 100
    s = np.std(scores) * 100
    diff = m - base_mean
    per = " ".join(f"{sc*100:.1f}" for sc in scores)
    if m > best_mean:
        best_mean = m
        best_name = name
    print(f"{name:<16s} {m:6.1f}% {s:5.1f}pp {diff:+7.1f}pp  [{per}]")

# Trajectory comparison at key checkpoints
print(f"\n{'='*70}")
print("TRAJECTORY (best % at each checkpoint, averaged across seeds)")
print(f"{'step':>7s}", end="")
for name in CONDITIONS:
    print(f" {name:>16s}", end="")
print()
print("-" * (7 + 17 * len(CONDITIONS)))

# Find common checkpoint steps
checkpoints = [i * LOG_EVERY for i in range(1, BUDGET // LOG_EVERY + 1)]
for cp in checkpoints:
    print(f"{cp:7d}", end="")
    for name in CONDITIONS:
        vals = []
        for traj in trajectories[name]:
            # Find closest checkpoint <= cp
            best_at_cp = traj[0][1]  # initial
            for step, b in traj:
                if step <= cp:
                    best_at_cp = b
            vals.append(best_at_cp)
        avg = np.mean(vals) * 100
        print(f" {avg:15.1f}%", end="")
    print()

print(f"\nBest: {best_name} ({best_mean:.1f}%)")
print(f"{'='*70}")
