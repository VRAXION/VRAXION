"""A/B test: baseline vs single signed 'drive' param.
drive > 0 → add N connections
drive < 0 → remove N connections
drive = 0 → no structural change
Drifts ±1, reverts on reject."""
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


def train_baseline(net, targets, budget):
    """Original signal/grow random bits."""
    score = evaluate(net, targets)
    best = score
    stale = 0

    for att in range(budget):
        old_loss = int(net.loss_pct)
        undo = net.mutate()
        new_score = evaluate(net, targets)

        if new_score > score:
            score = new_score
            best = max(best, score)
            stale = 0
        else:
            net.replay(undo)
            net.loss_pct = np.int8(old_loss)
            stale += 1
            if random.randint(1, 20) <= 7:
                net.signal = np.int8(1 - int(net.signal))
            if random.randint(1, 20) <= 7:
                net.grow = np.int8(1 - int(net.grow))

        if best >= 0.99 or stale >= 6000:
            break

    return best, att + 1


def train_drive(net, targets, budget):
    """Single signed drive param: +N=add N, -N=remove N, 0=nothing."""
    score = evaluate(net, targets)
    best = score
    stale = 0

    drive = 1  # start slightly positive (want to grow initially)
    drive_history = []

    for att in range(budget):
        old_loss = int(net.loss_pct)
        old_drive = drive

        # Loss drift — same as original
        if random.randint(1, 5) == 1:
            net.loss_pct = np.int8(max(1, min(50, int(net.loss_pct) + random.randint(-3, 3))))

        # Drive drift: ±1, clamp to [-15, +15]
        if random.randint(1, 20) <= 7:
            drive = max(-15, min(15, drive + random.choice([-1, 1])))

        # Execute drive
        undo = []
        if drive > 0:
            for _ in range(drive):
                net._add(undo)
        elif drive < 0:
            for _ in range(-drive):
                net._remove(undo)
        # drive == 0: no structural change

        new_score = evaluate(net, targets)

        if new_score > score:
            score = new_score
            best = max(best, score)
            stale = 0
        else:
            net.replay(undo)
            net.loss_pct = np.int8(old_loss)
            drive = old_drive  # REVERT on reject
            stale += 1

        if att % 1000 == 0:
            drive_history.append(drive)

        if best >= 0.99 or stale >= 6000:
            break

    return best, att + 1, drive, drive_history, net.count_connections()


print(f"A/B: baseline vs signed drive | V={VOCAB} budget={BUDGET}")
print(f"{'seed':>6s}  {'base':>8s}  {'drive':>8s}  {'diff':>7s}  {'final_d':>7s}  {'conns':>6s}  drive trajectory")
print("-" * 85)

b_total = 0
d_total = 0

for seed in SEEDS:
    # Baseline
    np.random.seed(seed); random.seed(seed)
    net1 = SelfWiringGraph(VOCAB)
    targets = np.arange(VOCAB)
    np.random.shuffle(targets)
    random.seed(seed * 1000 + 1)
    b_score, b_steps = train_baseline(net1, targets, BUDGET)

    # Drive version — same init
    np.random.seed(seed); random.seed(seed)
    net2 = SelfWiringGraph(VOCAB)
    targets2 = np.arange(VOCAB)
    np.random.shuffle(targets2)
    random.seed(seed * 1000 + 1)
    d_score, d_steps, final_d, d_hist, conns = train_drive(net2, targets2, BUDGET)

    diff = (d_score - b_score) * 100
    b_total += b_score
    d_total += d_score

    traj = " → ".join(f"{d:+d}" for d in d_hist)
    print(f"{seed:6d}  {b_score*100:7.1f}%  {d_score*100:7.1f}%  {diff:+6.1f}%  {final_d:+7d}  {conns:6d}  [{traj}]")
    sys.stdout.flush()

print("-" * 85)
b_avg = b_total / len(SEEDS) * 100
d_avg = d_total / len(SEEDS) * 100
print(f"{'avg':>6s}  {b_avg:7.1f}%  {d_avg:7.1f}%  {d_avg-b_avg:+6.1f}%")
