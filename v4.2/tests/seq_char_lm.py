"""Sequential character-level language model.
Feeds chars one-by-one using stateful forward() with divnorm.
The feedback loops should carry context from previous characters.
Train on Pride&Prejudice + Frankenstein, test on Alice (never seen)."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
from collections import Counter
from model.graph import SelfWiringGraph

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
ALPHA = 2.0


# ============================================================
# Data
# ============================================================

def load_and_clean(filename):
    with open(os.path.join(DATA_DIR, filename), 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read().lower()
    # Only keep a-z and space (drop accented chars)
    clean = ''.join(c if (c >= 'a' and c <= 'z') or c == ' ' else ' ' for c in text)
    return ' '.join(clean.split())


def build_bigram_baseline(text, chars, char2idx):
    """Compute bigram next-char accuracy on text."""
    transitions = Counter()
    for i in range(len(text) - 1):
        transitions[(text[i], text[i+1])] += 1
    # Most likely next char for each char
    best_next = {}
    for c in chars:
        nexts = {c2: cnt for (c1, c2), cnt in transitions.items() if c1 == c}
        if nexts:
            best_next[c] = max(nexts, key=nexts.get)
        else:
            best_next[c] = ' '
    # Accuracy on text
    correct = sum(1 for i in range(len(text)-1) if best_next.get(text[i]) == text[i+1])
    return correct / (len(text) - 1), best_next


# ============================================================
# Stateful forward with divnorm
# ============================================================

def forward_divnorm_seq(net, world, ticks=8, alpha=ALPHA):
    """Single-input stateful forward with divisive normalization."""
    act = net.state.copy()
    for t in range(ticks):
        if t == 0:
            act[:net.V] = world
        raw = act @ net.mask
        np.nan_to_num(raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        net.charge += raw
        total = np.abs(act).sum() + 1e-6
        act = np.maximum(net.charge - net.THRESHOLD, 0.0)
        act /= (1.0 + alpha * total)
        net.charge = np.clip(net.charge, -1.0, 1.0)
    net.state = act.copy()
    return net.charge[net.out_start:net.out_start + net.V]


# ============================================================
# Sequential evaluation
# ============================================================

def evaluate_seq(net, text, char2idx, ticks=8, max_chars=200):
    """Feed chars one-by-one, predict next, measure accuracy."""
    net.reset()
    correct = 0
    V = net.V
    length = min(len(text) - 1, max_chars)
    for i in range(length):
        world = np.zeros(V, dtype=np.float32)
        world[char2idx[text[i]]] = 1.0
        output = forward_divnorm_seq(net, world, ticks)
        pred = np.argmax(output)
        if pred == char2idx[text[i + 1]]:
            correct += 1
    return correct / length


def evaluate_seq_fixed(net, text, char2idx, ticks=8, start=0, length=200):
    """Evaluate on a fixed text segment (for consistent measurement)."""
    net.reset()
    correct = 0
    end = min(start + length, len(text) - 1)
    actual_len = end - start
    for i in range(start, end):
        world = np.zeros(net.V, dtype=np.float32)
        world[char2idx[text[i]]] = 1.0
        output = forward_divnorm_seq(net, world, ticks)
        pred = np.argmax(output)
        if pred == char2idx[text[i + 1]]:
            correct += 1
    return correct / actual_len


# ============================================================
# Training
# ============================================================

def train_seq(net, train_text, char2idx, budget, ticks=8,
              seq_len=1000, stale_limit=6000, log_every=2000):
    """Train on fixed sequential text segments. Read in order, not random."""

    # Fixed eval segment — always the same text for clean hill-climbing signal
    eval_text = train_text[:seq_len + 1]

    def evaluate():
        return evaluate_seq(net, eval_text, char2idx, ticks, seq_len)

    score = evaluate()
    best = score
    stale = 0
    trajectory = [(0, float(best))]

    for att in range(budget):
        old_loss = int(net.loss_pct)
        old_drive = int(net.drive)

        undo = net.mutate()
        new_score = evaluate()

        if new_score > score:
            score = new_score
            best = max(best, score)
            stale = 0
        else:
            net.replay(undo)
            net.loss_pct = np.int8(old_loss)
            net.drive = np.int8(old_drive)
            stale += 1

        if (att + 1) % log_every == 0:
            trajectory.append((att + 1, float(best)))
            sys.stdout.flush()

        if best >= 0.99 or stale >= stale_limit:
            break

    if (att + 1) % log_every != 0:
        trajectory.append((att + 1, float(best)))

    return float(best), att + 1, trajectory


# ============================================================
# Text generation
# ============================================================

def generate_text(net, chars, char2idx, seed_text="the ", length=200, ticks=8):
    """Generate text by feeding chars sequentially."""
    net.reset()
    result = list(seed_text)
    # Feed seed
    for c in seed_text:
        world = np.zeros(net.V, dtype=np.float32)
        world[char2idx.get(c, 0)] = 1.0
        output = forward_divnorm_seq(net, world, ticks)
    # Generate
    seen_bigrams = set()
    for _ in range(length):
        pred_idx = np.argmax(output)
        pred_char = chars[pred_idx]
        result.append(pred_char)
        # Check for loop
        if len(result) >= 2:
            bigram = (result[-2], result[-1])
            if bigram in seen_bigrams and len(result) > len(seed_text) + 5:
                result.append('[LOOP]')
                break
            seen_bigrams.add(bigram)
        # Feed predicted char
        world = np.zeros(net.V, dtype=np.float32)
        world[pred_idx] = 1.0
        output = forward_divnorm_seq(net, world, ticks)
    return ''.join(result)


# ============================================================
# Main
# ============================================================

print("=" * 70)
print("SEQUENTIAL CHARACTER-LEVEL LANGUAGE MODEL")
print("=" * 70)

# Load data — train/test split by book
train_text = load_and_clean('pride_prejudice.txt') + ' ' + load_and_clean('frankenstein.txt')
test_text = load_and_clean('alice.txt')

chars = sorted(set(train_text + test_text))
char2idx = {c: i for i, c in enumerate(chars)}
V = len(chars)

print(f"Vocab: {V} chars: {''.join(chars)}")
print(f"Train: {len(train_text)} chars (Pride&Prejudice + Frankenstein)")
print(f"Test:  {len(test_text)} chars (Alice in Wonderland)")

# Baselines
freq_baseline_char = max(Counter(train_text).values()) / len(train_text)
bigram_train_acc, bigram_best_next = build_bigram_baseline(train_text, chars, char2idx)
bigram_test_acc = sum(1 for i in range(len(test_text)-1)
                      if bigram_best_next.get(test_text[i]) == test_text[i+1]) / (len(test_text)-1)

print(f"\nBaselines:")
print(f"  Random:      {100/V:.1f}%")
print(f"  Frequency:   {freq_baseline_char*100:.1f}% (always predict most common char)")
print(f"  Bigram train: {bigram_train_acc*100:.1f}%")
print(f"  Bigram test:  {bigram_test_acc*100:.1f}%")

# Train
BUDGET = 48000
SEQ_LEN = 1000
TICKS = 8
SEEDS = [0, 42]

for seed in SEEDS:
    print(f"\n{'='*70}")
    print(f"SEED {seed} | V={V} N={V*3} budget={BUDGET} seq_len={SEQ_LEN} ticks={TICKS}")
    print(f"{'='*70}")

    np.random.seed(seed)
    random.seed(seed)
    net = SelfWiringGraph(V)

    random.seed(seed * 1000 + 1)
    t0 = time.time()
    best, steps, trajectory = train_seq(net, train_text, char2idx, BUDGET,
                                         ticks=TICKS, seq_len=SEQ_LEN)
    elapsed = time.time() - t0

    traj_str = " → ".join(f"{b*100:.1f}" for _, b in trajectory)
    print(f"Train best: {best*100:.1f}% in {steps} steps ({elapsed:.0f}s)")
    print(f"Trajectory: {traj_str}")
    print(f"Conns: {net.count_connections()}")

    # Test on held-out Alice text (multiple segments, averaged)
    test_accs = []
    for start in range(0, min(len(test_text)-SEQ_LEN-1, 5000), SEQ_LEN):
        acc = evaluate_seq_fixed(net, test_text, char2idx, TICKS, start, SEQ_LEN)
        test_accs.append(acc)
    test_avg = np.mean(test_accs)
    test_std = np.std(test_accs)

    # Also measure train accuracy on fixed segments
    train_accs = []
    for start in range(0, min(len(train_text)-SEQ_LEN-1, 5000), SEQ_LEN):
        acc = evaluate_seq_fixed(net, train_text, char2idx, TICKS, start, SEQ_LEN)
        train_accs.append(acc)
    train_avg = np.mean(train_accs)

    print(f"\nFinal evaluation:")
    print(f"  Train accuracy: {train_avg*100:.1f}% (avg over {len(train_accs)} segments)")
    print(f"  Test accuracy:  {test_avg*100:.1f}% ± {test_std*100:.1f}pp (avg over {len(test_accs)} segments)")
    print(f"  vs bigram test: {bigram_test_acc*100:.1f}%")
    print(f"  vs random:      {100/V:.1f}%")

    # Generate text
    gen = generate_text(net, chars, char2idx, "the ", 200, TICKS)
    print(f"\n  Generated: {gen[:150]}")

print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"Baselines: random={100/V:.1f}%, frequency={freq_baseline_char*100:.1f}%, "
      f"bigram_train={bigram_train_acc*100:.1f}%, bigram_test={bigram_test_acc*100:.1f}%")
