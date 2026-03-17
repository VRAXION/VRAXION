"""A/B test: drive-only vs drive+tune (parallel channels).
drive: signed int, +N=add, -N=remove, 0=nothing
tune:  unsigned int, N=flip N signs, 0=nothing
Both learned, both revert on reject."""
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
    """Drive only (previous best)."""
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

    return best, drive, net.count_connections()


def train_drive_tune(net, targets, budget):
    """Drive + tune parallel channels."""
    score = evaluate(net, targets)
    best = score
    stale = 0
    drive = 1   # structural: +add, -remove
    tune = 3    # signal: flip N signs

    for att in range(budget):
        old_loss = int(net.loss_pct)
        old_drive = drive
        old_tune = tune

        if random.randint(1, 5) == 1:
            net.loss_pct = np.int8(max(1, min(50, int(net.loss_pct) + random.randint(-3, 3))))
        if random.randint(1, 20) <= 7:
            drive = max(-15, min(15, drive + random.choice([-1, 1])))
        if random.randint(1, 20) <= 7:
            tune = max(0, min(15, tune + random.choice([-1, 1])))

        # Parallel: both channels execute
        undo = []
        # Channel 1: structure
        if drive > 0:
            for _ in range(drive):
                net._add(undo)
        elif drive < 0:
            for _ in range(-drive):
                net._remove(undo)
        # Channel 2: signal tuning
        for _ in range(tune):
            net._flip(undo)

        new_score = evaluate(net, targets)
        if new_score > score:
            score = new_score
            best = max(best, score)
            stale = 0
        else:
            net.replay(undo)
            net.loss_pct = np.int8(old_loss)
            drive = old_drive
            tune = old_tune
            stale += 1

        if best >= 0.99 or stale >= 6000:
            break

    return best, drive, tune, net.count_connections()


print(f"A/B: drive-only vs drive+tune | V={VOCAB} budget={BUDGET}")
print(f"{'seed':>6s}  {'drv_only':>9s}  {'drv+tune':>9s}  {'diff':>7s}  {'d':>3s} {'t':>3s}  {'conns':>6s}")
print("-" * 60)

d_total = 0
dt_total = 0

for seed in SEEDS:
    # Drive only
    np.random.seed(seed); random.seed(seed)
    net1 = SelfWiringGraph(VOCAB)
    targets = np.arange(VOCAB)
    np.random.shuffle(targets)
    random.seed(seed * 1000 + 1)
    d_score, d_drv, d_conns = train_drive_only(net1, targets, BUDGET)

    # Drive + tune — same init
    np.random.seed(seed); random.seed(seed)
    net2 = SelfWiringGraph(VOCAB)
    targets2 = np.arange(VOCAB)
    np.random.shuffle(targets2)
    random.seed(seed * 1000 + 1)
    dt_score, dt_drv, dt_tune, dt_conns = train_drive_tune(net2, targets2, BUDGET)

    diff = (dt_score - d_score) * 100
    d_total += d_score
    dt_total += dt_score

    print(f"{seed:6d}  {d_score*100:8.1f}%  {dt_score*100:8.1f}%  {diff:+6.1f}%  {dt_drv:+3d} {dt_tune:3d}  {dt_conns:6d}")
    sys.stdout.flush()

print("-" * 60)
d_avg = d_total / len(SEEDS) * 100
dt_avg = dt_total / len(SEEDS) * 100
print(f"{'avg':>6s}  {d_avg:8.1f}%  {dt_avg:8.1f}%  {dt_avg-d_avg:+6.1f}%")
