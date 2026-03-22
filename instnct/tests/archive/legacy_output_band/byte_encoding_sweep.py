"""Byte encoding test: 8 input neurons = 8 bits, multi-hot patterns.
Can the network do actual computation, not just lookup?
Tasks: copy, NOT, XOR, shift, parity, add1."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
from model.graph import SelfWiringGraph

ALPHA = 2.0
V = 8  # 8 bits = 1 byte
BUDGET = 8000
TICKS = 8
SEED = 42


# ============================================================
# Multi-hot forward pass
# ============================================================

def forward_patterns(net, patterns, ticks=TICKS, alpha=ALPHA):
    """Forward K arbitrary multi-hot input patterns through the network."""
    K = patterns.shape[0]
    N = net.N
    charges = np.zeros((K, N), dtype=np.float32)
    acts = np.zeros((K, N), dtype=np.float32)
    for t in range(ticks):
        if t == 0:
            acts[:, :V] = patterns
        raw = acts @ net.mask
        np.nan_to_num(raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        charges += raw
        total = np.abs(acts).sum(axis=1, keepdims=True) + 1e-6
        acts = np.maximum(charges - net.THRESHOLD, 0.0)
        acts /= (1.0 + alpha * total)
        charges = np.clip(charges, -1.0, 1.0)
    return charges[:, net.out_start:net.out_start + V]


def evaluate_patterns(net, inputs, targets):
    """Evaluate multi-hot input→output. Returns (bit_acc, pattern_acc)."""
    outputs = forward_patterns(net, inputs)
    predicted = (outputs > 0).astype(np.int8)
    bit_acc = (predicted == targets).mean()
    pattern_acc = (predicted == targets).all(axis=1).mean()
    return float(bit_acc), float(pattern_acc)


# ============================================================
# Task definitions: input patterns → expected output patterns
# ============================================================

def make_all_bytes():
    """All 256 possible 8-bit patterns."""
    return np.array([[int(b) for b in format(i, '08b')] for i in range(256)], dtype=np.float32)


def task_copy(inputs):
    """Output = input (identity copy)."""
    return inputs.astype(np.int8)


def task_not(inputs):
    """Output = bitwise NOT."""
    return (1 - inputs).astype(np.int8)


def task_xor_const(inputs):
    """Output = input XOR 10101010."""
    mask = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.int8)
    return (inputs.astype(np.int8) ^ mask)


def task_shift_left(inputs):
    """Output = circular shift left by 1."""
    return np.roll(inputs, -1, axis=1).astype(np.int8)


def task_reverse(inputs):
    """Output = bit reverse."""
    return inputs[:, ::-1].astype(np.int8)


def task_parity(inputs):
    """Output bit 0 = XOR of all inputs, rest = 0."""
    parity = inputs.astype(np.int8).sum(axis=1) % 2
    out = np.zeros_like(inputs, dtype=np.int8)
    out[:, 0] = parity
    return out


def task_add1(inputs):
    """Treat as unsigned 8-bit number, add 1, output binary."""
    nums = inputs.astype(np.int8).dot(1 << np.arange(7, -1, -1))
    nums = (nums + 1) % 256
    return np.array([[int(b) for b in format(n, '08b')] for n in nums], dtype=np.int8)


TASKS = {
    'copy':      task_copy,
    'NOT':       task_not,
    'XOR_0xAA':  task_xor_const,
    'shift_L':   task_shift_left,
    'reverse':   task_reverse,
    'parity':    task_parity,
    'add1':      task_add1,
}


# ============================================================
# Training
# ============================================================

def train_byte(net, inputs, targets, budget=BUDGET, stale_limit=4000):
    bit_acc, pat_acc = evaluate_patterns(net, inputs, targets)
    best_bit = bit_acc
    best_pat = pat_acc
    stale = 0

    for att in range(budget):
        old_loss = int(net.loss_pct)
        old_drive = int(net.drive)
        undo = net.mutate()
        new_bit, new_pat = evaluate_patterns(net, inputs, targets)

        if new_bit > bit_acc:
            bit_acc = new_bit
            best_bit = max(best_bit, bit_acc)
            best_pat = max(best_pat, new_pat)
            stale = 0
        else:
            net.replay(undo)
            net.loss_pct = np.int8(old_loss)
            net.drive = np.int8(old_drive)
            stale += 1

        if best_bit >= 0.999 or stale >= stale_limit:
            break

    return best_bit, best_pat, att + 1


# ============================================================
# Sweep
# ============================================================

all_inputs = make_all_bytes()  # 256 × 8

# Use subset for speed (64 patterns)
np.random.seed(0)
subset_idx = np.random.choice(256, 64, replace=False)
subset_inputs = all_inputs[subset_idx]

print(f"BYTE ENCODING SWEEP | V={V} N={V*3} budget={BUDGET} ticks={TICKS}")
print(f"Input: 8 bits multi-hot, 64 test patterns (of 256)")
print(f"{'='*65}")

# Baseline: one-hot (current system)
print(f"\n--- BASELINE: one-hot (8 patterns, lookup table) ---")
np.random.seed(SEED); random.seed(SEED)
net = SelfWiringGraph(V)
targets_onehot = np.arange(V); np.random.shuffle(targets_onehot)
from tests.overnight_text_sweep import forward_divnorm as _fd
# Quick baseline eval
def eval_onehot(net):
    logits = forward_patterns(net, np.eye(V, dtype=np.float32))
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    acc = (np.argmax(probs, axis=1) == targets_onehot).mean()
    return float(acc)

random.seed(SEED * 1000 + 1)
score = eval_onehot(net)
best = score; stale = 0
for att in range(BUDGET):
    ol, od = int(net.loss_pct), int(net.drive)
    undo = net.mutate()
    ns = eval_onehot(net)
    if ns > score:
        score = ns; best = max(best, score); stale = 0
    else:
        net.replay(undo); net.loss_pct = np.int8(ol); net.drive = np.int8(od); stale += 1
    if best >= 0.99 or stale >= 4000: break
print(f"One-hot accuracy: {best*100:.0f}% in {att+1} steps")

# Multi-hot tasks
print(f"\n--- MULTI-HOT BYTE TASKS ---")
print(f"{'task':<12s} {'bit_acc':>8s} {'pat_acc':>8s} {'steps':>6s} {'time':>5s}")
print("-" * 45)

results = []
for task_name, task_fn in TASKS.items():
    targets = task_fn(subset_inputs)

    np.random.seed(SEED); random.seed(SEED)
    net = SelfWiringGraph(V)
    random.seed(SEED * 1000 + 1)

    t0 = time.time()
    bit_acc, pat_acc, steps = train_byte(net, subset_inputs, targets)
    elapsed = time.time() - t0

    # Also test on FULL 256 patterns (generalization)
    full_targets = task_fn(all_inputs)
    full_bit, full_pat = evaluate_patterns(net, all_inputs, full_targets)

    print(f"{task_name:<12s} {bit_acc*100:7.1f}% {pat_acc*100:7.1f}% {steps:6d} {elapsed:4.1f}s  "
          f"(full: bit={full_bit*100:.1f}% pat={full_pat*100:.1f}%)")
    results.append((task_name, bit_acc, pat_acc, full_bit, full_pat, steps))

print(f"\n{'='*65}")
print(f"SUMMARY (sorted by full pattern accuracy)")
print(f"{'task':<12s} {'train_bit':>10s} {'train_pat':>10s} {'full_bit':>10s} {'full_pat':>10s}")
print("-" * 55)
for name, tb, tp, fb, fp, _ in sorted(results, key=lambda x: -x[4]):
    print(f"{name:<12s} {tb*100:9.1f}% {tp*100:9.1f}% {fb*100:9.1f}% {fp*100:9.1f}%")
