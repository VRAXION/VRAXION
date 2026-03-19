"""Binary window sweep: 5-bit encoding per char, wake/sleep training.
Test different window sizes to see what works."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
from collections import Counter
from model.graph import SelfWiringGraph

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
ALPHA = 2.0
BITS_PER_CHAR = 5

# Custom 5-bit encoding: space=0, a=1, ..., z=26
CHARS = ' abcdefghijklmnopqrstuvwxyz'
CHAR2CODE = {c: i for i, c in enumerate(CHARS)}
N_CHARS = len(CHARS)  # 27


def load_and_clean(fn):
    with open(os.path.join(DATA_DIR, fn), 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read().lower()
    return ' '.join(''.join(c if 'a' <= c <= 'z' or c == ' ' else ' ' for c in text).split())


def char_to_5bit(c):
    code = CHAR2CODE.get(c, 0)
    return [(code >> i) & 1 for i in range(BITS_PER_CHAR)]


def bits_to_char(bits):
    code = sum(int(b > 0) << i for i in range(BITS_PER_CHAR))
    return CHARS[code] if code < N_CHARS else '?'


def make_windows(text, window_size, max_samples=30000):
    n_input = window_size * BITS_PER_CHAR
    n_samples = min(len(text) - window_size, max_samples)
    inputs = np.zeros((n_samples, n_input), dtype=np.float32)
    targets = np.zeros((n_samples, BITS_PER_CHAR), dtype=np.int8)
    target_chars = np.zeros(n_samples, dtype=np.int32)

    for i in range(n_samples):
        for j in range(window_size):
            bits = char_to_5bit(text[i + j])
            inputs[i, j * BITS_PER_CHAR:(j+1) * BITS_PER_CHAR] = bits
        target_bits = char_to_5bit(text[i + window_size])
        targets[i] = target_bits
        target_chars[i] = CHAR2CODE.get(text[i + window_size], 0)
    return inputs, targets, target_chars


def forward_divnorm(net, patterns, ticks=8, alpha=ALPHA):
    K = patterns.shape[0]
    V = net.V; N = net.N
    charges = np.zeros((K, N), dtype=np.float32)
    acts = np.zeros((K, N), dtype=np.float32)
    for t in range(ticks):
        if t == 0:
            acts[:, :V] = patterns
        raw = acts @ net.mask
        charges += raw
        total = np.abs(acts).sum(axis=1, keepdims=True) + 1e-6
        acts = np.maximum(charges - net.THRESHOLD, 0.0)
        acts /= (1.0 + alpha * total)
        charges = np.clip(charges, -1.0, 1.0)
    return charges[:, net.out_start:net.out_start + BITS_PER_CHAR]


def evaluate(net, inputs, targets, target_chars, sample_size=512, ticks=8):
    idx = np.random.choice(len(inputs), min(sample_size, len(inputs)), replace=False)
    outputs = forward_divnorm(net, inputs[idx], ticks)
    pred_bits = (outputs > 0).astype(np.int8)
    bit_acc = (pred_bits == targets[idx]).mean()
    # Decode to chars and check
    pred_codes = np.array([sum(int(pred_bits[i, b] > 0) << b
                               for b in range(BITS_PER_CHAR))
                           for i in range(len(idx))])
    char_acc = (pred_codes == target_chars[idx]).mean()
    return float(bit_acc), float(char_acc)


def wake_phase(net, inputs, targets, target_chars, budget, ticks=8, add_prob=0.15):
    bit_acc, char_acc = evaluate(net, inputs, targets, target_chars, ticks=ticks)
    best_bit = bit_acc
    best_char = char_acc
    stale = 0
    improved = 0

    for step in range(budget):
        undo = []
        if random.random() < add_prob:
            net._add(undo)
        else:
            net._rewire(undo)

        new_bit, new_char = evaluate(net, inputs, targets, target_chars, ticks=ticks)
        if new_bit > bit_acc:
            bit_acc = new_bit
            char_acc = new_char
            best_bit = max(best_bit, bit_acc)
            best_char = max(best_char, char_acc)
            stale = 0
            improved += 1
        else:
            net.replay(undo)
            stale += 1

        if stale >= budget // 3:
            break

    return best_bit, best_char, step + 1, improved


def sleep_phase(net, inputs, targets, target_chars, ticks=8, max_attempts=3000):
    bit_acc, char_acc = evaluate(net, inputs, targets, target_chars, ticks=ticks)
    removed = 0
    stale = 0

    edges = sorted(net.alive, key=lambda rc: abs(net.mask[rc[0], rc[1]]))
    for r, c in edges:
        if stale >= max_attempts // 3:
            break
        if net.mask[r, c] == 0:
            continue
        old_val = net.mask[r, c]
        net.mask[r, c] = 0
        net.alive_set.discard((r, c))

        new_bit, new_char = evaluate(net, inputs, targets, target_chars, ticks=ticks)
        if new_bit >= bit_acc:
            bit_acc = new_bit
            char_acc = new_char
            removed += 1
            stale = 0
        else:
            net.mask[r, c] = old_val
            net.alive_set.add((r, c))
            stale += 1

    net.resync_alive()
    return removed, bit_acc, char_acc


# ============================================================
# Main sweep
# ============================================================

train_text = load_and_clean('pride_prejudice.txt')[:30000]
test_text = load_and_clean('alice.txt')[:5000]

# Bigram baseline
transitions = Counter()
for i in range(len(train_text) - 1):
    transitions[(train_text[i], train_text[i+1])] += 1
best_next = {}
for c in set(train_text):
    nexts = {c2: cnt for (c1, c2), cnt in transitions.items() if c1 == c}
    if nexts: best_next[c] = max(nexts, key=nexts.get)
bigram_acc = sum(1 for i in range(len(test_text)-1)
                 if best_next.get(test_text[i]) == test_text[i+1]) / (len(test_text)-1)

print(f"BINARY WINDOW SWEEP | {BITS_PER_CHAR} bits/char")
print(f"Bigram baseline: {bigram_acc*100:.1f}% | Random: {100/N_CHARS:.1f}%")
print(f"{'='*70}")

CONFIGS = [
    # window, V (=win*5), wake_budget, n_cycles
    (2,   10,  8000, 5),
    (3,   15,  8000, 5),
    (4,   20,  8000, 5),
    (6,   30,  8000, 5),
    (8,   40,  8000, 5),
    (10,  50,  8000, 4),
    (16,  80,  6000, 3),
    (20, 100,  4000, 3),
]

print(f"\n{'win':>4s} {'V':>4s} {'N':>5s} {'bit%':>6s} {'char%':>6s} "
      f"{'test_b':>6s} {'test_c':>6s} {'vs_bi':>6s} {'conns':>6s} {'time':>5s}")
print("-" * 65)

for window, V_size, wake_budget, n_cycles in CONFIGS:
    train_inputs, train_targets, train_tchars = make_windows(train_text, window)
    test_inputs, test_targets, test_tchars = make_windows(test_text, window)

    np.random.seed(42); random.seed(42)
    net = SelfWiringGraph(V_size)

    t0 = time.time()

    # Wake/sleep cycles
    for cycle in range(n_cycles):
        w_bit, w_char, w_steps, w_imp = wake_phase(
            net, train_inputs, train_targets, train_tchars, wake_budget)
        s_rem, s_bit, s_char = sleep_phase(
            net, train_inputs, train_targets, train_tchars)

    elapsed = time.time() - t0

    # Test
    test_bit, test_char = evaluate(net, test_inputs, test_targets, test_tchars)
    diff_bi = test_char - bigram_acc
    marker = " <<<" if test_char > bigram_acc else ""

    print(f"{window:4d} {V_size:4d} {V_size*3:5d} {w_bit*100:5.1f}% {w_char*100:5.1f}% "
          f"{test_bit*100:5.1f}% {test_char*100:5.1f}% {diff_bi*100:+5.1f}pp "
          f"{net.count_connections():6d} {elapsed:4.0f}s{marker}", flush=True)
