"""Window-based byte language model.
Input: N chars × 8 bits = N*8 input neurons (sliding window)
Output: 8 bits = next char prediction
Uses batch forward (mutation+selection) — the approach that WORKS."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
from model.graph import SelfWiringGraph

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
ALPHA = 2.0


def load_and_clean(fn):
    with open(os.path.join(DATA_DIR, fn), 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read().lower()
    return ' '.join(''.join(c if 'a' <= c <= 'z' or c == ' ' else ' ' for c in text).split())


def char_to_bits(c):
    """Encode char as 8-bit vector."""
    b = ord(c) & 0xFF
    return np.array([(b >> i) & 1 for i in range(8)], dtype=np.float32)


def bits_to_char(bits):
    """Decode 8-bit vector to char."""
    b = sum(int(bits[i] > 0) << i for i in range(8))
    return chr(b) if 32 <= b < 127 else '?'


def make_windows(text, window_size):
    """Create input/output pairs: window of chars → next char, all as bits."""
    n_bits = window_size * 8
    n_samples = len(text) - window_size
    inputs = np.zeros((n_samples, n_bits), dtype=np.float32)
    targets = np.zeros((n_samples, 8), dtype=np.int8)

    for i in range(n_samples):
        # Input: window_size chars encoded as bits
        for j in range(window_size):
            inputs[i, j*8:(j+1)*8] = char_to_bits(text[i + j])
        # Target: next char as bits
        targets[i] = char_to_bits(text[i + window_size]).astype(np.int8)

    return inputs, targets


def forward_patterns(net, patterns, ticks=8, alpha=ALPHA):
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
    # Output: first 8 neurons of output zone
    return charges[:, net.out_start:net.out_start + 8]


def evaluate(net, inputs, targets, sample_size=256):
    """Evaluate on random sample of windows."""
    idx = np.random.choice(len(inputs), min(sample_size, len(inputs)), replace=False)
    outputs = forward_patterns(net, inputs[idx])
    predicted = (outputs > 0).astype(np.int8)
    bit_acc = (predicted == targets[idx]).mean()
    # Char accuracy: all 8 bits correct
    char_acc = (predicted == targets[idx]).all(axis=1).mean()
    return float(bit_acc), float(char_acc)


def train_window(net, train_inputs, train_targets, budget, sample_size=256,
                 stale_limit=6000, log_every=2000):
    bit_acc, char_acc = evaluate(net, train_inputs, train_targets, sample_size)
    best_bit = bit_acc
    best_char = char_acc
    stale = 0
    trajectory = [(0, float(best_bit), float(best_char))]

    for att in range(budget):
        old_loss = int(net.loss_pct)
        old_drive = int(net.drive)
        undo = net.mutate()

        new_bit, new_char = evaluate(net, train_inputs, train_targets, sample_size)
        if new_bit > bit_acc:
            bit_acc = new_bit
            best_bit = max(best_bit, bit_acc)
            best_char = max(best_char, new_char)
            stale = 0
        else:
            net.replay(undo)
            net.loss_pct = np.int8(old_loss)
            net.drive = np.int8(old_drive)
            stale += 1

        if (att + 1) % log_every == 0:
            trajectory.append((att + 1, float(best_bit), float(best_char)))
            print(f"    [{att+1:6d}] bit={best_bit*100:.1f}% char={best_char*100:.1f}% "
                  f"conns={net.count_connections()}", flush=True)

        if best_bit >= 0.999 or stale >= stale_limit:
            break

    return best_bit, best_char, att + 1, trajectory


def generate_text(net, seed_text, chars_to_gen, window_size):
    """Generate text by sliding the window."""
    text = list(seed_text)
    for _ in range(chars_to_gen):
        window = ''.join(text[-window_size:])
        inp = np.zeros((1, window_size * 8), dtype=np.float32)
        for j, c in enumerate(window):
            inp[0, j*8:(j+1)*8] = char_to_bits(c)
        out = forward_patterns(net, inp)
        pred_char = bits_to_char(out[0])
        text.append(pred_char)
    return ''.join(text)


# ============================================================
# Main
# ============================================================

train_text = load_and_clean('pride_prejudice.txt')[:50000]  # first 50k chars
test_text = load_and_clean('alice.txt')[:10000]

# Baselines
from collections import Counter
# Bigram
transitions = Counter()
for i in range(len(train_text) - 1):
    transitions[(train_text[i], train_text[i+1])] += 1
best_next = {}
for c in set(train_text):
    nexts = {c2: cnt for (c1, c2), cnt in transitions.items() if c1 == c}
    if nexts: best_next[c] = max(nexts, key=nexts.get)
bigram_test = sum(1 for i in range(len(test_text)-1)
                  if best_next.get(test_text[i]) == test_text[i+1]) / (len(test_text)-1)

print(f"WINDOW-BASED BYTE LANGUAGE MODEL")
print(f"Bigram baseline: {bigram_test*100:.1f}% | Random char: {100/128:.1f}%")
print(f"{'='*70}")

# Sweep window sizes
CONFIGS = [
    # window, V (=window*8), budget, sample_size
    (2,  16, 16000, 256),  # 2 chars = 16 bits
    (3,  24, 16000, 256),  # 3 chars = 24 bits
    (4,  32, 16000, 256),  # 4 chars = 32 bits
    (6,  48, 24000, 256),  # 6 chars = 48 bits
    (8,  64, 24000, 256),  # 8 chars = 64 bits
]

print(f"\n{'window':>6s} {'V':>4s} {'N':>5s} {'train_bit':>10s} {'train_char':>11s} "
      f"{'test_bit':>9s} {'test_char':>10s} {'steps':>6s} {'time':>5s}")
print("-" * 75)

for window_size, V_size, budget, sample_size in CONFIGS:
    print(f"\n--- Window={window_size} chars ({V_size} input neurons, N={V_size*3}) ---")

    # Create training data
    train_inputs, train_targets = make_windows(train_text, window_size)
    test_inputs, test_targets = make_windows(test_text, window_size)
    print(f"  Train samples: {len(train_inputs)} | Test samples: {len(test_inputs)}")

    np.random.seed(42); random.seed(42)
    net = SelfWiringGraph(V_size)

    random.seed(42 * 1000 + 1)
    t0 = time.time()
    best_bit, best_char, steps, traj = train_window(
        net, train_inputs, train_targets, budget, sample_size)
    elapsed = time.time() - t0

    # Test
    test_bit, test_char = evaluate(net, test_inputs, test_targets, min(1000, len(test_inputs)))

    # Generate sample
    seed = train_text[:window_size]
    gen = generate_text(net, seed, 50, window_size)

    print(f"  Generated: {gen[:60]}")
    print(f"{window_size:6d} {V_size:4d} {V_size*3:5d} {best_bit*100:9.1f}% {best_char*100:10.1f}% "
          f"{test_bit*100:8.1f}% {test_char*100:9.1f}% {steps:6d} {elapsed:4.0f}s")
