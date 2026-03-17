"""A/B test: baseline vs learned add/remove weights.
Minimal: only add + remove, two learned weight params."""
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


def train_addrem(net, targets, budget):
    """Learned add/remove weights. Two params, revert on reject."""
    score = evaluate(net, targets)
    best = score
    stale = 0

    add_w = 50   # start equal: 50/50
    rem_w = 50
    add_count = 0
    rem_count = 0

    for att in range(budget):
        old_loss = int(net.loss_pct)
        old_add_w = add_w
        old_rem_w = rem_w

        # Intensity drift — same as original
        if random.randint(1, 20) <= 7:
            net.intensity = np.int8(max(1, min(15, int(net.intensity) + random.choice([-1, 1]))))

        # Loss drift — same as original
        if random.randint(1, 5) == 1:
            net.loss_pct = np.int8(max(1, min(50, int(net.loss_pct) + random.randint(-3, 3))))

        # Weight drift (7/20 chance each)
        if random.randint(1, 20) <= 7:
            add_w = max(1, min(99, add_w + random.choice([-3, -1, 1, 3])))
        if random.randint(1, 20) <= 7:
            rem_w = max(1, min(99, rem_w + random.choice([-3, -1, 1, 3])))

        # Mutate based on learned weights
        undo = []
        total = add_w + rem_w
        for _ in range(int(net.intensity)):
            roll = random.randint(1, total)
            if roll <= add_w:
                net._add(undo)
                add_count += 1
            else:
                net._remove(undo)
                rem_count += 1

        new_score = evaluate(net, targets)

        if new_score > score:
            score = new_score
            best = max(best, score)
            stale = 0
        else:
            net.replay(undo)
            net.loss_pct = np.int8(old_loss)
            add_w = old_add_w   # REVERT weights on reject
            rem_w = old_rem_w
            stale += 1

        if best >= 0.99 or stale >= 6000:
            break

    return best, att + 1, add_w, rem_w, add_count, rem_count


print(f"A/B: baseline vs learned add/remove weights | V={VOCAB} budget={BUDGET}")
print(f"{'seed':>6s}  {'base':>8s}  {'addrem':>8s}  {'diff':>7s}  {'add_w':>5s} {'rem_w':>5s}  {'add%':>5s}")
print("-" * 65)

b_total = 0
a_total = 0

for seed in SEEDS:
    # Baseline
    np.random.seed(seed); random.seed(seed)
    net1 = SelfWiringGraph(VOCAB)
    targets = np.arange(VOCAB)
    np.random.shuffle(targets)
    random.seed(seed * 1000 + 1)
    b_score, b_steps = train_baseline(net1, targets, BUDGET)

    # Add/remove version — same init
    np.random.seed(seed); random.seed(seed)
    net2 = SelfWiringGraph(VOCAB)
    targets2 = np.arange(VOCAB)
    np.random.shuffle(targets2)
    random.seed(seed * 1000 + 1)
    a_score, a_steps, aw, rw, ac, rc = train_addrem(net2, targets2, BUDGET)

    diff = (a_score - b_score) * 100
    b_total += b_score
    a_total += a_score
    add_pct = ac / (ac + rc) * 100 if (ac + rc) > 0 else 50

    print(f"{seed:6d}  {b_score*100:7.1f}%  {a_score*100:7.1f}%  {diff:+6.1f}%  {aw:5d} {rw:5d}  {add_pct:4.0f}%")
    sys.stdout.flush()

print("-" * 65)
b_avg = b_total / len(SEEDS) * 100
a_avg = a_total / len(SEEDS) * 100
print(f"{'avg':>6s}  {b_avg:7.1f}%  {a_avg:7.1f}%  {a_avg-b_avg:+6.1f}%")
