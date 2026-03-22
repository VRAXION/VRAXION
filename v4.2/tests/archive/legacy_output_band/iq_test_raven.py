"""IQ test for SelfWiringGraph — Raven-style pattern completion.

Instead of language modeling (huge search space), test abstract
pattern recognition with small vocabularies:

Level 1: Counting       — 1,2,3,? → 4
Level 2: Arithmetic     — 2,4,6,? → 8
Level 3: Repeat         — A,B,A,B,? → A
Level 4: Mirror         — 1,2,3,2,? → 1
Level 5: Modular        — 0,1,2,0,1,? → 2
Level 6: Fibonacci-ish  — 1,1,2,3,? → 5 (mod vocab)
Level 7: Nested         — 1,1,2,2,3,3,? → 4 (each repeated)

Each level uses the minimum vocab needed. We train on examples
of the pattern and test on unseen continuations.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
from model.graph import SelfWiringGraph


# ============================================================
# Pattern generators — each returns (sequence, vocab_size)
# ============================================================

def gen_counting(length, start=0, vocab=8):
    """1,2,3,4,5,... mod vocab"""
    return [(start + i) % vocab for i in range(length)], vocab

def gen_arithmetic(length, step=2, start=0, vocab=10):
    """0,2,4,6,8,... mod vocab"""
    return [(start + i * step) % vocab for i in range(length)], vocab

def gen_repeat(length, pattern=None, vocab=4):
    """A,B,C,A,B,C,... repeating motif"""
    if pattern is None:
        pattern = [0, 1, 2]
    seq = [pattern[i % len(pattern)] for i in range(length)]
    return seq, vocab

def gen_mirror(length, base=None, vocab=6):
    """1,2,3,2,1,2,3,2,1,... palindrome wave"""
    if base is None:
        base = [0, 1, 2, 3]
    wave = base + base[-2:0:-1]  # e.g. [0,1,2,3,2,1]
    seq = [wave[i % len(wave)] for i in range(length)]
    return seq, vocab

def gen_modular(length, mod=3, vocab=None):
    """0,1,2,0,1,2,... simple mod counter"""
    if vocab is None:
        vocab = mod
    return [i % mod for i in range(length)], vocab

def gen_fibonacci_mod(length, vocab=8):
    """Fibonacci mod vocab: 1,1,2,3,5,0,5,5,2,7,..."""
    seq = [1, 1]
    for i in range(2, length):
        seq.append((seq[-1] + seq[-2]) % vocab)
    return seq[:length], vocab

def gen_double(length, vocab=6):
    """Each value repeated: 0,0,1,1,2,2,3,3,..."""
    seq = [i // 2 % vocab for i in range(length)]
    return seq, vocab

def gen_xor_pair(length, vocab=4):
    """A,B,A^B, A,B,A^B, ... — XOR triplets"""
    seq = []
    rng = random.Random(42)
    while len(seq) < length:
        a = rng.randint(0, vocab - 1)
        b = rng.randint(0, vocab - 1)
        seq.extend([a, b, a ^ b])
    return seq[:length], vocab

def gen_alternating_step(length, vocab=6):
    """Step +1, +2, +1, +2: 0,1,3,4,6,7,..."""
    seq = [0]
    steps = [1, 2]
    for i in range(1, length):
        seq.append((seq[-1] + steps[(i - 1) % 2]) % vocab)
    return seq[:length], vocab


# ============================================================
# IQ test levels
# ============================================================

IQ_LEVELS = [
    # (name, generator_fn, difficulty_description)
    ("L1 count mod3",       lambda n: gen_modular(n, 3),            "0,1,2,0,1,2,..."),
    ("L2 count mod5",       lambda n: gen_modular(n, 5),            "0,1,2,3,4,0,1,..."),
    ("L3 repeat ABC",       lambda n: gen_repeat(n, [0,1,2], 3),    "A,B,C,A,B,C,..."),
    ("L4 repeat ABBA",      lambda n: gen_repeat(n, [0,1,1,0], 2),  "A,B,B,A,A,B,B,A,..."),
    ("L5 double",           lambda n: gen_double(n, 4),              "0,0,1,1,2,2,3,3,..."),
    ("L6 mirror 0123",      lambda n: gen_mirror(n, [0,1,2,3], 4),  "0,1,2,3,2,1,0,1,..."),
    ("L7 step +1+2",        lambda n: gen_alternating_step(n, 6),    "0,1,3,4,6,7,..."),
    ("L8 arithmetic +2",    lambda n: gen_arithmetic(n, 2, 0, 8),    "0,2,4,6,0,2,..."),
    ("L9 fib mod5",         lambda n: gen_fibonacci_mod(n, 5),       "1,1,2,3,0,3,..."),
    ("L10 xor triplets",    lambda n: gen_xor_pair(n, 4),            "a,b,a^b,..."),
]


# ============================================================
# Training: feed sequence pairs (context → next token)
# ============================================================

def make_seq_pairs(sequence, vocab, context_len=3):
    """Turn sequence into (one-hot context, target) pairs.
    Context = last context_len tokens flattened as one-hot."""
    V = context_len * vocab
    n = len(sequence) - context_len
    inputs = np.zeros((n, V), dtype=np.float32)
    targets = np.zeros(n, dtype=np.int32)
    for i in range(n):
        for j in range(context_len):
            inputs[i, j * vocab + sequence[i + j]] = 1.0
        targets[i] = sequence[i + context_len]
    return inputs, targets


def forward_batch(net, patterns, ticks=8):
    """Standard ReLU forward (batch)."""
    K = patterns.shape[0]
    N = net.N
    V = net.V
    charges = np.zeros((K, N), dtype=np.float32)
    acts = np.zeros((K, N), dtype=np.float32)
    retain = float(net.retention)
    for t in range(ticks):
        if t == 0:
            acts[:, :V] = patterns
        raw = acts @ net.mask
        np.nan_to_num(raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        charges += raw
        charges *= retain
        acts = np.maximum(charges - net.THRESHOLD, 0.0)
        charges = np.clip(charges, -1.0, 1.0)
    return charges[:, net.out_start:net.out_start + V]


def evaluate(net, inputs, targets, vocab, sample_size=256, ticks=8):
    idx = np.random.choice(len(inputs), min(sample_size, len(inputs)), replace=False)
    # Output is V-wide but targets are in [0, vocab), need to slice
    logits = forward_batch(net, inputs[idx], ticks)
    # Only look at first vocab columns for prediction
    logits_v = logits[:, :vocab]
    preds = np.argmax(logits_v, axis=1)
    return float((preds == targets[idx]).mean())


def train_pattern(net, inputs, targets, vocab, budget=15000, ticks=8):
    """Train with mutation+selection, return best accuracy."""
    stale_limit = budget // 2
    rewire_threshold = stale_limit // 3

    score = evaluate(net, inputs, targets, vocab, ticks=ticks)
    best = score
    stale = 0

    for att in range(budget):
        old_loss = int(net.loss_pct)
        old_drive = int(net.drive)
        undo = net.mutate()
        new_score = evaluate(net, inputs, targets, vocab, ticks=ticks)

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
                rw_score = evaluate(net, inputs, targets, vocab, ticks=ticks)
                if rw_score > score:
                    score = rw_score
                    best = max(best, score)
                    stale = 0
                else:
                    net.replay(rw_undo)

        if best >= 0.99 or stale >= stale_limit:
            break

    return best, att + 1


def accuracy_full(net, inputs, targets, vocab, ticks=8):
    """Full eval (no sampling)."""
    batch = 512
    correct = 0
    for s in range(0, len(inputs), batch):
        e = min(s + batch, len(inputs))
        logits = forward_batch(net, inputs[s:e], ticks)[:, :vocab]
        preds = np.argmax(logits, axis=1)
        correct += (preds == targets[s:e]).sum()
    return correct / len(inputs)


# ============================================================
# Main — run IQ test battery
# ============================================================

SEED = 42
SEQ_LEN = 120        # total sequence length to generate
CONTEXT = 3          # context window for prediction
BUDGET = 15000       # mutation budget per level
TICKS = 8

print("=" * 72)
print("IQ TEST FOR SELFWIRINGGRAPH — RAVEN-STYLE PATTERN COMPLETION")
print(f"Context={CONTEXT} tokens | Budget={BUDGET} mutations | Ticks={TICKS}")
print("=" * 72)
print()

# Split: first 80 for training, rest for test
TRAIN_LEN = 80
TEST_LEN = SEQ_LEN - TRAIN_LEN - CONTEXT

results = []

print(f"{'Level':<22s} {'Pattern':<22s} {'V':>3s} {'Input':>5s} "
      f"{'Train':>6s} {'Test':>6s} {'Rand':>5s} {'Steps':>6s} {'Time':>5s}")
print("-" * 90)

for name, gen_fn, desc in IQ_LEVELS:
    np.random.seed(SEED)
    random.seed(SEED)

    # Generate sequence
    seq, vocab = gen_fn(SEQ_LEN)

    # Split into train/test pairs
    train_seq = seq[:TRAIN_LEN]
    test_seq = seq  # full sequence, test pairs come from latter portion

    train_in, train_tgt = make_seq_pairs(train_seq, vocab, CONTEXT)
    test_in, test_tgt = make_seq_pairs(test_seq, vocab, CONTEXT)
    # Use only latter portion for test
    test_start = len(train_in)
    test_in = test_in[test_start:]
    test_tgt = test_tgt[test_start:]

    if len(test_in) == 0:
        print(f"{name:<22s} {'SKIP - no test data'}")
        continue

    V_input = CONTEXT * vocab
    net = SelfWiringGraph(V_input)

    random_baseline = 1.0 / vocab

    t0 = time.time()
    best_score, steps = train_pattern(net, train_in, train_tgt, vocab, BUDGET, TICKS)
    elapsed = time.time() - t0

    train_acc = accuracy_full(net, train_in, train_tgt, vocab, TICKS)
    test_acc = accuracy_full(net, test_in, test_tgt, vocab, TICKS)

    marker = ""
    if test_acc > random_baseline * 1.5:
        marker = " *"
    if test_acc > 0.8:
        marker = " **"
    if test_acc > 0.95:
        marker = " ***"

    print(f"{name:<22s} {desc:<22s} {vocab:3d} {V_input:5d} "
          f"{train_acc*100:5.1f}% {test_acc*100:5.1f}% "
          f"{random_baseline*100:4.1f}% {steps:6d} {elapsed:4.0f}s{marker}",
          flush=True)

    results.append({
        'name': name,
        'vocab': vocab,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'random': random_baseline,
        'steps': steps,
    })

# Summary
print()
print("=" * 72)
print("IQ SCORE SUMMARY")
print("=" * 72)

solved = sum(1 for r in results if r['test_acc'] > 0.8)
partial = sum(1 for r in results if r['test_acc'] > r['random'] * 1.5 and r['test_acc'] <= 0.8)
failed = len(results) - solved - partial

print(f"Solved (>80%):  {solved}/{len(results)}")
print(f"Partial:        {partial}/{len(results)}")
print(f"Failed:         {failed}/{len(results)}")
print()

# IQ score analogy
iq = 70 + solved * 10 + partial * 3
print(f"Graph IQ estimate: ~{iq}")
print(f"  70 = baseline (exists)")
print(f"  +10 per solved level")
print(f"  +3 per partial level")
print()

if solved >= 5:
    print("VERDICT: The graph can reason about simple abstract patterns!")
elif solved >= 2:
    print("VERDICT: Basic pattern recognition works, complex reasoning fails.")
elif solved >= 1:
    print("VERDICT: Minimal pattern detection — needs architectural help.")
else:
    print("VERDICT: Cannot solve even simple patterns — mutation search too weak?")
