"""
VRAXION v22 -- Binary Byte Encoding Test
==========================================
Instead of one-hot (1 neuron = 1 char), use 8-bit binary patterns.

  one-hot "t":  [0,0,0,...,1,...,0]   29 neurons, 1 active
  binary "t":   [0,1,1,1,0,1,0,0]    8 neurons, pattern

The network MUST learn pattern-to-pattern transformation.
Can't get lucky with argmax on uniform output -- ALL 8 bits must be right.

V=8 (8 bit I/O), split I/O: first 8 = input bits, last 8 = output bits.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
import time
from v22_best_config import SelfWiringGraph

# =============================================================
# Corpus + binary encoding
# =============================================================
TEXT = """the quick brown fox jumps over the lazy dog.
the cat sat on the mat. the dog chased the cat around the garden.
she sells sea shells by the sea shore. peter piper picked a peck of pickled peppers.
to be or not to be that is the question. all that glitters is not gold.
a stitch in time saves nine. the early bird catches the worm.
actions speak louder than words. practice makes perfect.
knowledge is power. time is money. better late than never.
the pen is mightier than the sword. where there is a will there is a way.
an apple a day keeps the doctor away. birds of a feather flock together.
every cloud has a silver lining. fortune favors the bold.
the best things in life are free. honesty is the best policy.""".lower()

chars = sorted(set(TEXT))
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for c, i in char_to_idx.items()}
VOCAB = len(chars)
BITS = 8  # 8-bit encoding

print(f"  Vocab: {VOCAB} chars, encoded in {BITS} bits")


def char_to_bits(ch):
    """Char -> 8-bit binary pattern as float array."""
    idx = char_to_idx[ch]
    bits = np.zeros(BITS, dtype=np.float32)
    for b in range(BITS):
        bits[b] = float((idx >> b) & 1)
    return bits


def bits_to_idx(bits):
    """8-bit float array -> char index (threshold at 0.5)."""
    idx = 0
    for b in range(BITS):
        if bits[b] > 0.0:  # threshold: positive charge = 1
            idx |= (1 << b)
    return idx


def bits_to_char(bits):
    """8-bit float array -> char (or '?' if invalid)."""
    idx = bits_to_idx(bits)
    return idx_to_char.get(idx, '?')


# Build bigram targets (most common next char per char)
bigram_counts = np.zeros((VOCAB, VOCAB), dtype=np.int32)
for i in range(len(TEXT) - 1):
    a, b = char_to_idx[TEXT[i]], char_to_idx[TEXT[i+1]]
    bigram_counts[a, b] += 1

TARGETS = np.argmax(bigram_counts, axis=1)  # most common next char per input char

# Active inputs (chars that actually appear)
ACTIVE = [i for i in range(VOCAB) if bigram_counts[i].sum() > 0]

# Pre-encode all targets as bit patterns
TARGET_BITS = np.zeros((VOCAB, BITS), dtype=np.float32)
for i in range(VOCAB):
    for b in range(BITS):
        TARGET_BITS[i, b] = float((TARGETS[i] >> b) & 1)

# Pre-encode all inputs as bit patterns
INPUT_BITS = np.zeros((VOCAB, BITS), dtype=np.float32)
for i in range(VOCAB):
    for b in range(BITS):
        INPUT_BITS[i, b] = float((i >> b) & 1)

print(f"  Active inputs: {len(ACTIVE)}/{VOCAB}")
print(f"  Example: '{idx_to_char[0]}' = bits {INPUT_BITS[0].astype(int).tolist()} "
      f"-> '{idx_to_char[TARGETS[0]]}' = bits {TARGET_BITS[0].astype(int).tolist()}")
print(f"  Example: '{idx_to_char[20]}' = bits {INPUT_BITS[20].astype(int).tolist()} "
      f"-> '{idx_to_char[TARGETS[20]]}' = bits {TARGET_BITS[20].astype(int).tolist()}")


# =============================================================
# Scoring
# =============================================================
def score_binary(net, ticks=8):
    """Score: how many bits are correct in the output pattern?
    Returns (bit_accuracy, exact_match_rate, per_char_details)."""
    V = BITS  # V=8

    total_bits_correct = 0
    total_bits = 0
    exact_matches = 0

    for i in ACTIVE:
        # Forward single input
        net.reset()
        world = INPUT_BITS[i].copy()
        logits = net.forward(world, ticks)  # returns charge[out_start:out_start+V]

        # Compare output bits to target bits
        pred_bits = (logits > 0.0).astype(float)
        target = TARGET_BITS[i]

        bits_match = (pred_bits == target).sum()
        total_bits_correct += bits_match
        total_bits += BITS

        if bits_match == BITS:
            exact_matches += 1

    bit_acc = total_bits_correct / max(total_bits, 1)
    exact_rate = exact_matches / max(len(ACTIVE), 1)

    return bit_acc, exact_rate


def score_binary_batch(net, ticks=8):
    """Batch version -- construct custom input matrix."""
    V = BITS
    N = net.N
    Weff = net.W * net.mask

    # We have VOCAB inputs but V=8, so we run VOCAB forward passes batched
    n_inputs = len(ACTIVE)
    charges = np.zeros((n_inputs, N), dtype=np.float32)
    acts = np.zeros((n_inputs, N), dtype=np.float32)

    for t in range(ticks):
        if t == 0:
            # Inject input bit patterns into first V=8 neurons
            for row, i in enumerate(ACTIVE):
                acts[row, :V] = INPUT_BITS[i]
        raw = acts @ Weff + acts * 0.1
        np.nan_to_num(raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        charges += raw * 0.3
        charges *= net.leak
        acts = np.maximum(charges - net.threshold, 0.0)
        charges = np.clip(charges, -net.threshold * 2, net.threshold * 2)

    # Read output zone
    out_start = net.out_start
    outputs = charges[:, out_start:out_start + V]

    # Score
    total_bits_correct = 0
    total_bits = 0
    exact_matches = 0

    for row, i in enumerate(ACTIVE):
        pred_bits = (outputs[row] > 0.0).astype(float)
        target = TARGET_BITS[i]
        bits_match = (pred_bits == target).sum()
        total_bits_correct += bits_match
        total_bits += BITS
        if bits_match == BITS:
            exact_matches += 1

    bit_acc = total_bits_correct / max(total_bits, 1)
    exact_rate = exact_matches / max(len(ACTIVE), 1)

    # Combined score for mutation selection
    combined = 0.5 * bit_acc + 0.5 * exact_rate
    return combined, bit_acc, exact_rate


# =============================================================
# Training
# =============================================================
def train(net, budget, ticks=8, label=''):
    best_sc, best_bit, best_exact = score_binary_batch(net, ticks)
    best_mask = net.mask.copy()
    best_W = net.W.copy()
    kept = 0
    stale = 0
    phase = 'STRUCTURE'
    switched = False

    t0 = time.time()
    for att in range(budget):
        sm = net.mask.copy()
        sw = net.W.copy()

        if phase == 'STRUCTURE':
            net.mutate_structure(0.05)
        else:
            if random.random() < 0.4:
                net.mutate_weights()
            else:
                net.mutate_structure(0.07)

        sc, bit_acc, exact = score_binary_batch(net, ticks)

        if sc > best_sc:
            best_sc = sc
            best_bit = bit_acc
            best_exact = exact
            best_mask = net.mask.copy()
            best_W = net.W.copy()
            kept += 1
            stale = 0
        else:
            net.mask = sm
            net.W = sw
            stale += 1

        if phase == 'STRUCTURE' and stale > 2500 and not switched:
            phase = 'BOTH'
            switched = True
            stale = 0

        if (att + 1) % 2000 == 0:
            elapsed = time.time() - t0
            print(f"    [{label:10s}] att={att+1:5d} bit_acc={best_bit*100:5.1f}% "
                  f"exact={best_exact*100:5.1f}% kept={kept:3d} "
                  f"phase={phase} time={elapsed:.0f}s")
            sys.stdout.flush()

        if stale >= 6000:
            break

    net.mask = best_mask.copy()
    net.W = best_W.copy()
    elapsed = time.time() - t0
    return best_sc, best_bit, best_exact, kept, elapsed


def show_predictions(net, ticks=8):
    """Show what the network predicts for each input char."""
    V = BITS
    lines = []
    for i in ACTIVE:
        net.reset()
        logits = net.forward(INPUT_BITS[i].copy(), ticks)
        pred_bits = (logits > 0.0).astype(int)
        target_bits = TARGET_BITS[i].astype(int)
        pred_idx = bits_to_idx(logits)
        pred_ch = idx_to_char.get(pred_idx, '?')
        target_ch = idx_to_char.get(TARGETS[i], '?')
        match = 'OK' if pred_idx == TARGETS[i] else '  '
        bit_match = (pred_bits == target_bits).sum()
        lines.append(f"    '{idx_to_char[i]}' -> pred='{pred_ch}' target='{target_ch}' "
                     f"bits={bit_match}/{BITS} {match}")
    return lines


# =============================================================
# Main
# =============================================================
def main():
    print(f"\n{'=' * 70}")
    print(f"  BINARY BYTE ENCODING TEST")
    print(f"{'=' * 70}")
    print(f"  V=8 (bits), split I/O, N=128")
    print(f"  Task: char bigram as bit-pattern transformation")
    print(f"  Random bit accuracy: 50% (each bit is coin flip)")
    print(f"  Random exact match: {(0.5**8)*100:.2f}% (all 8 bits correct)")
    print()
    sys.stdout.flush()

    SEEDS = [42, 77, 123]
    N_NEURONS = 128  # V=8, so 8 input + 112 internal + 8 output
    BUDGET = 16000

    all_results = []

    for seed in SEEDS:
        print(f"\n  --- SEED {seed} ---")
        sys.stdout.flush()

        np.random.seed(seed)
        random.seed(seed)

        net = SelfWiringGraph(N_NEURONS, BITS)
        print(f"    io_mode={net.io_mode}, out_start={net.out_start}")
        print(f"    Initial connections: {net.count_connections()}")

        # Initial score
        sc0, bit0, exact0 = score_binary_batch(net, ticks=8)
        print(f"    Initial: bit_acc={bit0*100:.1f}% exact={exact0*100:.1f}%")
        sys.stdout.flush()

        # Train
        sc, bit_acc, exact, kept, elapsed = train(
            net, BUDGET, ticks=8, label=f'seed{seed}')

        print(f"\n    Final: bit_acc={bit_acc*100:.1f}% exact={exact*100:.1f}% "
              f"kept={kept} time={elapsed:.0f}s")

        # Show predictions
        preds = show_predictions(net, ticks=8)
        for line in preds[:10]:  # first 10
            print(line)
        if len(preds) > 10:
            print(f"    ... ({len(preds)} total)")
        sys.stdout.flush()

        all_results.append({
            'seed': seed,
            'bit_acc': bit_acc,
            'exact': exact,
            'kept': kept,
            'time': elapsed,
        })

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY")
    print(f"{'=' * 70}\n")

    print(f"  {'Seed':>6s}  {'Bit Acc':>8s}  {'Exact':>8s}  {'Kept':>5s}  {'Time':>6s}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*5}  {'-'*6}")
    for r in all_results:
        print(f"  {r['seed']:6d}  {r['bit_acc']*100:7.1f}%  {r['exact']*100:7.1f}%  "
              f"{r['kept']:5d}  {r['time']:5.0f}s")

    mean_bit = np.mean([r['bit_acc'] for r in all_results])
    mean_exact = np.mean([r['exact'] for r in all_results])
    print(f"\n  {'Mean':>6s}  {mean_bit*100:7.1f}%  {mean_exact*100:7.1f}%")
    print(f"\n  Random: bit=50.0%, exact=0.39%")

    if mean_exact > 0.1:
        print(f"\n  VERDICT: Network LEARNS bit-pattern transformations!")
    elif mean_bit > 0.6:
        print(f"\n  VERDICT: Partial learning (bits improve, exact match weak)")
    else:
        print(f"\n  VERDICT: Cannot learn bit-pattern mapping at this scale")

    print(f"\n{'=' * 70}", flush=True)


if __name__ == '__main__':
    main()
