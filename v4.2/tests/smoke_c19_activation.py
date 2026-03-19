"""Smoke test: C19 dual-phi activation in SelfWiringGraph.

Replaces the ReLU activation (max(charge - threshold, 0)) with
C19 periodic parabolic wave from the archive. Tests on the
bigram task + small sliding window to see if it helps.
"""
import sys, os, time, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
from collections import Counter
from model.graph import SelfWiringGraph

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

PHI = (1 + math.sqrt(5)) / 2      # 1.618...
PHI_INV = (math.sqrt(5) - 1) / 2  # 0.618...

CHARS = ' abcdefghijklmnopqrstuvwxyz'
CHAR2IDX = {c: i for i, c in enumerate(CHARS)}
N_CHARS = len(CHARS)


# ============================================================
# C19 activation (numpy, dual-phi variant)
# ============================================================

def c19_activation(x, rho=4.0, C=math.pi):
    """C19 periodic parabolic wave with dual-phi gain.
    Positive arches scaled by 1/phi, negative arches by phi.
    Creates irrational magnitude ratio that breaks resonance."""
    inv_c = 1.0 / C
    l = 6.0 * C
    scaled = x * inv_c
    n = np.floor(scaled)
    t = scaled - n
    h = t * (1.0 - t)
    is_odd = (np.fmod(n, 2.0) >= 1.0).astype(np.float32)
    sgn = 1.0 - 2.0 * is_odd
    # Dual-phi gain: even arches → 1/phi, odd arches → phi
    gain = is_odd * (PHI - PHI_INV) + PHI_INV
    core = C * h * (sgn + rho * h) * gain
    # Linear tails outside periodic region
    result = np.where(x <= -l, x + l, core)
    result = np.where(x >= l, x - l, result)
    return result


def c19_simple(x, rho=4.0, C=math.pi):
    """C19 without dual-phi (baseline symmetric version)."""
    inv_c = 1.0 / C
    l = 6.0 * C
    scaled = x * inv_c
    n = np.floor(scaled)
    t = scaled - n
    h = t * (1.0 - t)
    is_even = (np.fmod(n, 2.0) < 1.0).astype(np.float32)
    sgn = 2.0 * is_even - 1.0
    core = C * (sgn * h + rho * h * h)
    result = np.where(x <= -l, x + l, core)
    result = np.where(x >= l, x - l, result)
    return result


# ============================================================
# Forward pass variants
# ============================================================

def forward_relu(net, patterns, ticks=8):
    """Original ReLU forward (batch)."""
    K = patterns.shape[0]
    N = net.N
    charges = np.zeros((K, N), dtype=np.float32)
    acts = np.zeros((K, N), dtype=np.float32)
    retain = float(net.retention)
    for t in range(ticks):
        if t == 0:
            acts[:, :net.V] = patterns
        raw = acts @ net.mask
        np.nan_to_num(raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        charges += raw
        charges *= retain
        acts = np.maximum(charges - net.THRESHOLD, 0.0)
        charges = np.clip(charges, -1.0, 1.0)
    return charges[:, net.out_start:net.out_start + N_CHARS]


def forward_c19(net, patterns, ticks=8, activation=None):
    """C19-based forward (batch). Replaces ReLU with C19."""
    if activation is None:
        activation = c19_activation
    K = patterns.shape[0]
    N = net.N
    charges = np.zeros((K, N), dtype=np.float32)
    acts = np.zeros((K, N), dtype=np.float32)
    retain = float(net.retention)
    for t in range(ticks):
        if t == 0:
            acts[:, :net.V] = patterns
        raw = acts @ net.mask
        np.nan_to_num(raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        charges += raw
        charges *= retain
        # C19 instead of ReLU
        acts = activation(charges)
        charges = np.clip(charges, -1.0, 1.0)
    return charges[:, net.out_start:net.out_start + N_CHARS]


# ============================================================
# Eval + Train helpers
# ============================================================

def load_and_clean(fn):
    with open(os.path.join(DATA_DIR, fn), 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read().lower()
    return ' '.join(''.join(c if 'a' <= c <= 'z' or c == ' ' else ' ' for c in text).split())


def make_windows(text, window, max_samples=15000):
    V = window * N_CHARS
    n = min(len(text) - window, max_samples)
    inputs = np.zeros((n, V), dtype=np.float32)
    targets = np.zeros(n, dtype=np.int32)
    for i in range(n):
        for j in range(window):
            idx = CHAR2IDX.get(text[i + j], 0)
            inputs[i, j * N_CHARS + idx] = 1.0
        targets[i] = CHAR2IDX.get(text[i + window], 0)
    return inputs, targets


def evaluate(net, inputs, targets, forward_fn, sample_size=512, ticks=8):
    idx = np.random.choice(len(inputs), min(sample_size, len(inputs)), replace=False)
    logits = forward_fn(net, inputs[idx], ticks)
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    preds = np.argmax(probs, axis=1)
    t = targets[idx]
    acc = (preds == t).mean()
    tp = probs[np.arange(len(t)), t].mean()
    return float(0.5 * acc + 0.5 * tp)


def accuracy_full(net, inputs, targets, forward_fn, ticks=8):
    batch = 1024
    correct = 0
    for s in range(0, len(inputs), batch):
        e = min(s + batch, len(inputs))
        logits = forward_fn(net, inputs[s:e], ticks)
        preds = np.argmax(logits, axis=1)
        correct += (preds == targets[s:e]).sum()
    return correct / len(inputs)


def train_ab(net, inputs, targets, forward_fn, budget, stale_limit=None):
    if stale_limit is None:
        stale_limit = budget // 2
    score = evaluate(net, inputs, targets, forward_fn)
    best = score
    stale = 0
    rewire_threshold = stale_limit // 3

    for att in range(budget):
        old_loss = int(net.loss_pct)
        old_drive = int(net.drive)
        undo = net.mutate()
        new_score = evaluate(net, inputs, targets, forward_fn)

        if new_score > score:
            score = new_score
            best = max(best, score)
            stale = 0
        else:
            net.replay(undo)
            net.loss_pct = np.int8(old_loss)
            net.drive = np.int8(old_drive)
            stale += 1

            if stale > rewire_threshold:
                rw_undo = []
                net._rewire(rw_undo)
                rw_score = evaluate(net, inputs, targets, forward_fn)
                if rw_score > score:
                    score = rw_score
                    best = max(best, score)
                    stale = 0
                else:
                    net.replay(rw_undo)

        if best >= 0.99 or stale >= stale_limit:
            break

    return best, att + 1


# ============================================================
# Main: A/B test ReLU vs C19 vs C19-dualphi
# ============================================================

print("=" * 72)
print("SMOKE TEST: C19 ACTIVATION IN SELFWIRINGGRAPH")
print("=" * 72)

train_text = load_and_clean('pride_prejudice.txt')[:30000]
test_text = load_and_clean('alice.txt')[:5000]

# Bigram baseline
transitions = Counter()
for i in range(len(train_text) - 1):
    transitions[(train_text[i], train_text[i+1])] += 1
best_next = {}
for c in set(train_text):
    nexts = {c2: cnt for (c1, c2), cnt in transitions.items() if c1 == c}
    if nexts:
        best_next[c] = max(nexts, key=nexts.get)
bi_test = sum(1 for i in range(len(test_text)-1)
              if best_next.get(test_text[i]) == test_text[i+1]) / (len(test_text)-1)

print(f"Baselines:  random={100/N_CHARS:.1f}%  bigram={bi_test*100:.1f}%")

VARIANTS = [
    ("ReLU",         forward_relu, None),
    ("C19-simple",   lambda net, p, t: forward_c19(net, p, t, c19_simple), None),
    ("C19-dualphi",  lambda net, p, t: forward_c19(net, p, t, c19_activation), None),
]

WINDOWS = [1, 2, 3]
BUDGET = 12000
SEED = 42

print(f"\n{'variant':>14s} {'win':>4s} {'V':>5s} {'train%':>7s} {'test%':>7s} "
      f"{'vs_bi':>7s} {'conns':>6s} {'time':>5s}")
print("-" * 65)

for window in WINDOWS:
    train_in, train_tgt = make_windows(train_text, window)
    test_in, test_tgt = make_windows(test_text, window)
    V = window * N_CHARS

    for name, fwd_fn, _ in VARIANTS:
        np.random.seed(SEED)
        random.seed(SEED)
        net = SelfWiringGraph(V)

        t0 = time.time()
        best_score, steps = train_ab(net, train_in, train_tgt, fwd_fn, BUDGET)
        elapsed = time.time() - t0

        train_acc = accuracy_full(net, train_in, train_tgt, fwd_fn)
        test_acc = accuracy_full(net, test_in, test_tgt, fwd_fn)
        diff = test_acc - bi_test
        marker = " <<<" if test_acc > bi_test else ""

        print(f"{name:>14s} {window:4d} {V:5d} {train_acc*100:6.1f}% {test_acc*100:6.1f}% "
              f"{diff*100:+6.1f}pp {net.count_connections():6d} {elapsed:4.0f}s{marker}",
              flush=True)

print(f"\n{'='*72}")
print("DONE — compare ReLU vs C19 variants across window sizes")
print(f"{'='*72}")
