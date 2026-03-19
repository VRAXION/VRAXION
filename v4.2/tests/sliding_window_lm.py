"""Sliding window character-level LM.

One-hot encode the last K characters as input, predict next char.
Input = K × 27 one-hot vectors concatenated → V = K * 27.
Output = first 27 neurons of the output block → softmax over charset.

Tests whether giving the network explicit multi-char context
breaks through the bigram ceiling.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
from collections import Counter
from model.graph import SelfWiringGraph

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

CHARS = ' abcdefghijklmnopqrstuvwxyz'
CHAR2IDX = {c: i for i, c in enumerate(CHARS)}
N_CHARS = len(CHARS)  # 27


def load_and_clean(fn):
    with open(os.path.join(DATA_DIR, fn), 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read().lower()
    return ' '.join(''.join(c if 'a' <= c <= 'z' or c == ' ' else ' ' for c in text).split())


def make_windows(text, window, max_samples=20000):
    """Build (samples, V) input matrix and (samples,) target index array."""
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


def forward_batch_window(net, patterns, ticks=8):
    """Batch forward for window inputs. Returns (K, N_CHARS) logits."""
    K = patterns.shape[0]
    V, N = net.V, net.N
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
    # Read only the first N_CHARS neurons from the output block
    return charges[:, net.out_start:net.out_start + N_CHARS]


def evaluate(net, inputs, targets, sample_size=512, ticks=8):
    """Score: 0.5 * accuracy + 0.5 * mean_target_prob."""
    idx = np.random.choice(len(inputs), min(sample_size, len(inputs)), replace=False)
    logits = forward_batch_window(net, inputs[idx], ticks)
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    preds = np.argmax(probs, axis=1)
    t = targets[idx]
    acc = (preds == t).mean()
    tp = probs[np.arange(len(t)), t].mean()
    return float(0.5 * acc + 0.5 * tp)


def accuracy(net, inputs, targets, ticks=8):
    """Pure accuracy over all samples (batched)."""
    batch = 1024
    correct = 0
    for start in range(0, len(inputs), batch):
        end = min(start + batch, len(inputs))
        logits = forward_batch_window(net, inputs[start:end], ticks)
        preds = np.argmax(logits, axis=1)
        correct += (preds == targets[start:end]).sum()
    return correct / len(inputs)


def train_window(net, inputs, targets, budget, ticks=8, stale_limit=None):
    """Mutation + selection training with stale-triggered rewire."""
    if stale_limit is None:
        stale_limit = budget // 2

    score = evaluate(net, inputs, targets, ticks=ticks)
    best = score
    stale = 0
    rewire_threshold = stale_limit // 3

    for att in range(budget):
        old_loss = int(net.loss_pct)
        old_drive = int(net.drive)
        undo = net.mutate()
        new_score = evaluate(net, inputs, targets, ticks=ticks)

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
                rw_score = evaluate(net, inputs, targets, ticks=ticks)
                if rw_score > score:
                    score = rw_score
                    best = max(best, score)
                    stale = 0
                else:
                    net.replay(rw_undo)

        if (att + 1) % 2000 == 0:
            acc = accuracy(net, inputs, targets, ticks)
            print(f"    [{att+1:6d}] score={best*100:.1f}%  acc={acc*100:.1f}%  "
                  f"conns={net.count_connections()}  drive={int(net.drive):+d}  "
                  f"stale={stale}", flush=True)

        if best >= 0.99 or stale >= stale_limit:
            break

    return best, att + 1


# ============================================================
# Baselines
# ============================================================

def bigram_accuracy(train_text, test_text):
    transitions = Counter()
    for i in range(len(train_text) - 1):
        transitions[(train_text[i], train_text[i+1])] += 1
    best_next = {}
    for c in set(train_text):
        nexts = {c2: cnt for (c1, c2), cnt in transitions.items() if c1 == c}
        if nexts:
            best_next[c] = max(nexts, key=nexts.get)
    return sum(1 for i in range(len(test_text)-1)
               if best_next.get(test_text[i]) == test_text[i+1]) / (len(test_text)-1)


def trigram_accuracy(train_text, test_text):
    transitions = Counter()
    for i in range(len(train_text) - 2):
        transitions[(train_text[i], train_text[i+1], train_text[i+2])] += 1
    best_next = {}
    for (c1, c2), _ in Counter((train_text[i], train_text[i+1])
                                 for i in range(len(train_text)-1)).items():
        nexts = {c3: cnt for (a, b, c3), cnt in transitions.items() if a == c1 and b == c2}
        if nexts:
            best_next[(c1, c2)] = max(nexts, key=nexts.get)
    return sum(1 for i in range(len(test_text)-2)
               if best_next.get((test_text[i], test_text[i+1])) == test_text[i+2]) / (len(test_text)-2)


# ============================================================
# Main sweep
# ============================================================

print("=" * 72)
print("SLIDING WINDOW CHARACTER LM")
print("=" * 72)

train_text = load_and_clean('pride_prejudice.txt')[:50000]
test_text = load_and_clean('alice.txt')[:10000]

bi_train = bigram_accuracy(train_text, train_text)
bi_test = bigram_accuracy(train_text, test_text)
tri_test = trigram_accuracy(train_text, test_text)

print(f"Train: {len(train_text)} chars | Test: {len(test_text)} chars")
print(f"Baselines:  random={100/N_CHARS:.1f}%  bigram={bi_test*100:.1f}%  trigram={tri_test*100:.1f}%")
print()

CONFIGS = [
    # (window, budget, seed)
    (1,  16000, 42),   # baseline: single char (= bigram learner)
    (2,  16000, 42),   # bigram context
    (3,  24000, 42),   # trigram context
    (4,  24000, 42),   # 4-gram context
    (6,  32000, 42),   # 6-gram context
]

print(f"{'win':>4s} {'V':>5s} {'N':>5s} {'train%':>7s} {'test%':>7s} "
      f"{'vs_bi':>7s} {'vs_tri':>7s} {'conns':>6s} {'time':>6s}")
print("-" * 60)

for window, budget, seed in CONFIGS:
    V = window * N_CHARS
    N = V * 3

    np.random.seed(seed)
    random.seed(seed)

    train_in, train_tgt = make_windows(train_text, window)
    test_in, test_tgt = make_windows(test_text, window)

    net = SelfWiringGraph(V)

    print(f"\n  Training window={window}  V={V}  N={N}  budget={budget}  "
          f"samples={len(train_in)}")

    t0 = time.time()
    best_score, steps = train_window(net, train_in, train_tgt, budget)
    elapsed = time.time() - t0

    train_acc = accuracy(net, train_in, train_tgt)
    test_acc = accuracy(net, test_in, test_tgt)

    diff_bi = test_acc - bi_test
    diff_tri = test_acc - tri_test
    marker = ""
    if test_acc > bi_test:
        marker += " >bigram"
    if test_acc > tri_test:
        marker += " >trigram"

    print(f"\n{window:4d} {V:5d} {N:5d} {train_acc*100:6.1f}% {test_acc*100:6.1f}% "
          f"{diff_bi*100:+6.1f}pp {diff_tri*100:+6.1f}pp "
          f"{net.count_connections():6d} {elapsed:5.0f}s{marker}")

print(f"\n{'='*72}")
print("DONE")
print(f"{'='*72}")
