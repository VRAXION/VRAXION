"""Test: drive with rewire at 0.
drive > 0 → add N connections
drive < 0 → remove N connections
drive = 0 → rewire (move connections, same count)
Single param, three behaviors."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random

from model.graph import SelfWiringGraph

SEEDS = [0, 1, 2, 10, 42]
VOCAB = 64
BUDGET = 16000


def evaluate(net, targets):
    logits = net.forward_batch(8)
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    V = net.V
    acc = (np.argmax(probs, axis=1)[:V] == targets[:V]).mean()
    tp = probs[np.arange(V), targets[:V]].mean()
    return 0.5 * acc + 0.5 * tp


def train_drive_only(net, targets, budget):
    """Drive without rewire (previous best)."""
    score = evaluate(net, targets)
    best = score
    stale = 0
    drive = 1

    for att in range(budget):
        old_loss = int(net.loss_pct)
        old_drive = drive

        if random.randint(1, 5) == 1:
            net.loss_pct = np.int8(max(1, min(50, int(net.loss_pct) + random.randint(-3, 3))))
        if random.randint(1, 20) <= 7:
            drive = max(-15, min(15, drive + random.choice([-1, 1])))

        undo = []
        if drive > 0:
            for _ in range(drive):
                net._add(undo)
        elif drive < 0:
            for _ in range(-drive):
                net._remove(undo)
        # drive == 0: nothing

        new_score = evaluate(net, targets)
        if new_score > score:
            score = new_score
            best = max(best, score)
            stale = 0
        else:
            net.replay(undo)
            net.loss_pct = np.int8(old_loss)
            drive = old_drive
            stale += 1

        if best >= 0.99 or stale >= 6000:
            break

    return best, net.count_connections()


def train_drive_rewire(net, targets, budget):
    """Drive with rewire at 0."""
    score = evaluate(net, targets)
    best = score
    stale = 0
    drive = 1
    # Track how much time in each mode
    add_steps = 0
    rem_steps = 0
    rew_steps = 0
    drive_trace = []  # every 1000 steps

    for att in range(budget):
        old_loss = int(net.loss_pct)
        old_drive = drive

        if random.randint(1, 5) == 1:
            net.loss_pct = np.int8(max(1, min(50, int(net.loss_pct) + random.randint(-3, 3))))
        if random.randint(1, 20) <= 7:
            drive = max(-15, min(15, drive + random.choice([-1, 1])))

        undo = []
        if drive > 0:
            for _ in range(drive):
                net._add(undo)
            add_steps += 1
        elif drive < 0:
            for _ in range(-drive):
                net._remove(undo)
            rem_steps += 1
        else:
            # drive == 0: rewire (intensity times for fair comparison)
            for _ in range(max(3, abs(old_drive) if old_drive != 0 else 3)):
                net._rewire(undo)
            rew_steps += 1

        new_score = evaluate(net, targets)
        if new_score > score:
            score = new_score
            best = max(best, score)
            stale = 0
        else:
            net.replay(undo)
            net.loss_pct = np.int8(old_loss)
            drive = old_drive
            stale += 1

        if (att + 1) % 1000 == 0:
            drive_trace.append(drive)

        if best >= 0.99 or stale >= 6000:
            break

    total = add_steps + rem_steps + rew_steps
    return best, net.count_connections(), drive, add_steps, rem_steps, rew_steps, total, drive_trace


print(f"Drive+rewire@0 test | V={VOCAB} budget={BUDGET}")
print(f"{'seed':>6s}  {'drv_only':>9s}  {'drv+rew':>9s}  {'diff':>7s}  {'d':>3s}  {'add%':>5s} {'rem%':>5s} {'rew%':>5s}  {'conns':>6s}  trajectory")
print("-" * 105)

d_total = 0
dr_total = 0

for seed in SEEDS:
    # Drive only
    np.random.seed(seed); random.seed(seed)
    net1 = SelfWiringGraph(VOCAB)
    targets = np.arange(VOCAB)
    np.random.shuffle(targets)
    random.seed(seed * 1000 + 1)
    d_score, d_conns = train_drive_only(net1, targets, BUDGET)

    # Drive + rewire — same init
    np.random.seed(seed); random.seed(seed)
    net2 = SelfWiringGraph(VOCAB)
    targets2 = np.arange(VOCAB)
    np.random.shuffle(targets2)
    random.seed(seed * 1000 + 1)
    dr_score, dr_conns, dr_d, adds, rems, rews, total, trace = train_drive_rewire(net2, targets2, BUDGET)

    diff = (dr_score - d_score) * 100
    d_total += d_score
    dr_total += dr_score

    ap = adds/total*100 if total else 0
    rp = rems/total*100 if total else 0
    wp = rews/total*100 if total else 0
    traj = " ".join(f"{d:+d}" for d in trace)

    print(f"{seed:6d}  {d_score*100:8.1f}%  {dr_score*100:8.1f}%  {diff:+6.1f}%  {dr_d:+3d}  {ap:4.0f}% {rp:4.0f}% {wp:4.0f}%  {dr_conns:6d}  [{traj}]")
    sys.stdout.flush()

print("-" * 105)
d_avg = d_total / len(SEEDS) * 100
dr_avg = dr_total / len(SEEDS) * 100
print(f"{'avg':>6s}  {d_avg:8.1f}%  {dr_avg:8.1f}%  {dr_avg-d_avg:+6.1f}%")
