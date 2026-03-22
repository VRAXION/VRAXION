"""A/B test: current signal/grow random bits vs learned mode param.
Deterministic: same seeds, same masks, same targets."""
import sys, os, time, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random

from model.graph import SelfWiringGraph

SEEDS = [0, 1, 2, 10, 42]
VOCAB = 64
BUDGET = 16000
OPS = ['flip', 'add', 'remove', 'rewire']


def evaluate(net, targets):
    logits = net.forward_batch(8)
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    V = net.V
    acc = (np.argmax(probs, axis=1)[:V] == targets[:V]).mean()
    tp = probs[np.arange(V), targets[:V]].mean()
    return 0.5 * acc + 0.5 * tp


def train_baseline(net, targets, budget):
    """Original: signal/grow random bits."""
    score = evaluate(net, targets)
    best = score
    stale = 0
    mode_hist = {'SIGNAL': 0, 'GROW': 0, 'SHRINK': 0}

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

        mode = "SIGNAL" if net.signal else ("GROW" if net.grow else "SHRINK")
        mode_hist[mode] = mode_hist.get(mode, 0) + 1

        if best >= 0.99 or stale >= 6000:
            break

    return best, att + 1, mode_hist


def mutate_mode(net, mode_param):
    """Mode-based mutation: mode is a learned int 0-3."""
    # Intensity drift — same as original
    if random.randint(1, 20) <= 7:
        net.intensity = np.int8(max(1, min(15, int(net.intensity) + random.choice([-1, 1]))))

    # Loss drift — same as original
    if random.randint(1, 5) == 1:
        net.loss_pct = np.int8(max(1, min(50, int(net.loss_pct) + random.randint(-3, 3))))

    # Mode drift (±1 with wrap, 7/20 chance) — REVERTS on reject
    if random.randint(1, 20) <= 7:
        mode_param = (mode_param + random.choice([-1, 1])) % 4

    undo = []
    for _ in range(int(net.intensity)):
        if mode_param == 0:
            net._flip(undo)
        elif mode_param == 1:
            net._add(undo)
        elif mode_param == 2:
            net._remove(undo)
        else:
            net._rewire(undo)

    return undo, mode_param


def train_mode(net, targets, budget):
    """New: learned mode param that reverts on reject."""
    score = evaluate(net, targets)
    best = score
    stale = 0
    mode_param = 1  # start with add
    mode_hist = {op: 0 for op in OPS}

    for att in range(budget):
        old_loss = int(net.loss_pct)
        old_mode = mode_param

        undo, mode_param = mutate_mode(net, mode_param)
        new_score = evaluate(net, targets)

        mode_hist[OPS[mode_param]] += 1

        if new_score > score:
            score = new_score
            best = max(best, score)
            stale = 0
        else:
            net.replay(undo)
            net.loss_pct = np.int8(old_loss)
            mode_param = old_mode  # REVERT mode on reject!
            stale += 1

        if best >= 0.99 or stale >= 6000:
            break

    return best, att + 1, mode_hist


print(f"A/B: signal/grow random bits vs learned mode | V={VOCAB} budget={BUDGET}")
print(f"{'seed':>6s}  {'baseline':>10s}  {'mode':>10s}  {'diff':>8s}  base_modes              mode_converged")
print("-" * 95)

base_total = 0
mode_total = 0

for seed in SEEDS:
    # Baseline
    np.random.seed(seed); random.seed(seed)
    net1 = SelfWiringGraph(VOCAB)
    targets = np.arange(VOCAB)
    np.random.shuffle(targets)

    t0 = time.time()
    random.seed(seed * 1000 + 1)  # separate mutation RNG seed
    b_score, b_steps, b_hist = train_baseline(net1, targets, BUDGET)
    t_base = time.time() - t0

    # Mode version — same init
    np.random.seed(seed); random.seed(seed)
    net2 = SelfWiringGraph(VOCAB)
    targets2 = np.arange(VOCAB)
    np.random.shuffle(targets2)

    t0 = time.time()
    random.seed(seed * 1000 + 1)  # same mutation RNG seed
    m_score, m_steps, m_hist = train_mode(net2, targets2, BUDGET)
    t_mode = time.time() - t0

    diff = (m_score - b_score) * 100
    base_total += b_score
    mode_total += m_score

    # Mode distribution summary
    b_str = " ".join(f"{k}:{v}" for k, v in sorted(b_hist.items()) if v > 0)
    m_str = " ".join(f"{k}:{v}" for k, v in sorted(m_hist.items()) if v > 0)

    print(f"{seed:6d}  {b_score*100:9.1f}%  {m_score*100:9.1f}%  {diff:+7.1f}%  {b_str}")
    print(f"{'':6s}  {'':10s}  {'':10s}  {'':8s}  {m_str}")
    sys.stdout.flush()

print("-" * 95)
b_avg = base_total / len(SEEDS) * 100
m_avg = mode_total / len(SEEDS) * 100
print(f"{'avg':>6s}  {b_avg:9.1f}%  {m_avg:9.1f}%  {m_avg-b_avg:+7.1f}%")
