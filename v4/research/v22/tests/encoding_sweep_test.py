"""
VRAXION v22 -- Encoding Sweep Test
====================================
What's the best pattern encoding for the Self-Wiring Graph Network?

Phase A: Encoding type at 8 bits (byte, random, spread, balanced)
         + Hadamard at 32 bits (native size for 29 chars)
Phase B: Best encoding at different bit widths (4, 6, 8, 12, 16)
Phase C: Density control (natural, uniform 50%, uniform 25%)

Fair comparison: internal neurons fixed at 112. N = 112 + 2*bits.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
import time
import csv
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.linalg import hadamard as scipy_hadamard
from v22_best_config import SelfWiringGraph
from v22_log import live_log, log_msg

# =============================================================
# Corpus
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

# Bigram targets
bigram_counts = np.zeros((VOCAB, VOCAB), dtype=np.int32)
for i in range(len(TEXT) - 1):
    a, b = char_to_idx[TEXT[i]], char_to_idx[TEXT[i+1]]
    bigram_counts[a, b] += 1
TARGETS = np.argmax(bigram_counts, axis=1)
ACTIVE = [i for i in range(VOCAB) if bigram_counts[i].sum() > 0]

# =============================================================
# Encoding generators
# =============================================================
def make_byte_encoding(vocab, bits):
    """Natural binary: idx -> binary representation."""
    codebook = np.zeros((vocab, bits), dtype=np.float32)
    for i in range(vocab):
        for b in range(bits):
            codebook[i, b] = float((i >> b) & 1)
    return codebook


def make_random_encoding(vocab, bits, seed=42):
    """Random binary pattern per char."""
    rng = np.random.RandomState(seed + 7777)
    codebook = rng.randint(0, 2, size=(vocab, bits)).astype(np.float32)
    # Ensure no duplicates
    seen = set()
    for i in range(vocab):
        key = tuple(codebook[i].astype(int))
        while key in seen:
            codebook[i] = rng.randint(0, 2, size=bits).astype(np.float32)
            key = tuple(codebook[i].astype(int))
        seen.add(key)
    return codebook


def make_hadamard_encoding(vocab):
    """Hadamard matrix rows. bits = next power of 2 >= vocab."""
    # Find smallest power of 2 >= vocab
    bits = 1
    while bits < vocab:
        bits *= 2
    H = scipy_hadamard(bits)  # {-1, +1} matrix
    codebook = ((H + 1) / 2).astype(np.float32)  # map to {0, 1}
    return codebook[:vocab]  # first vocab rows


def make_spread_encoding(vocab, bits, seed=42):
    """Greedy spread: each new pattern maximizes min Hamming distance."""
    rng = np.random.RandomState(seed + 8888)
    codebook = np.zeros((vocab, bits), dtype=np.float32)

    # First pattern: random
    codebook[0] = rng.randint(0, 2, size=bits).astype(np.float32)

    for i in range(1, vocab):
        best_pattern = None
        best_min_dist = -1

        # Try 200 random candidates, keep the one with max min-distance
        for _ in range(200):
            candidate = rng.randint(0, 2, size=bits).astype(np.float32)
            # Min Hamming distance to all existing patterns
            dists = np.sum(np.abs(codebook[:i] - candidate), axis=1)
            min_dist = dists.min()
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_pattern = candidate.copy()

        codebook[i] = best_pattern

    return codebook


def make_balanced_encoding(vocab, bits, density=0.5, seed=42):
    """Random but each pattern has exactly density*bits ones."""
    rng = np.random.RandomState(seed + 9999)
    n_ones = int(bits * density)
    codebook = np.zeros((vocab, bits), dtype=np.float32)
    seen = set()

    for i in range(vocab):
        while True:
            pattern = np.zeros(bits, dtype=np.float32)
            idx = rng.choice(bits, n_ones, replace=False)
            pattern[idx] = 1.0
            key = tuple(pattern.astype(int))
            if key not in seen:
                seen.add(key)
                codebook[i] = pattern
                break

    return codebook


# =============================================================
# Hamming distance stats
# =============================================================
def hamming_stats(codebook):
    """Return min, avg, std of pairwise Hamming distances."""
    n = len(codebook)
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sum(np.abs(codebook[i] - codebook[j]))
            dists.append(d)
    dists = np.array(dists)
    return float(dists.min()), float(dists.mean()), float(dists.std())


# =============================================================
# Training + scoring
# =============================================================
def train_and_eval(config_name, input_codes, target_codes, bits, N,
                   budget, train_seed, ticks=8, log_q=None):
    """Train network on pattern-to-pattern bigram mapping."""
    np.random.seed(train_seed)
    random.seed(train_seed)

    net = SelfWiringGraph(N, bits)
    out_start = net.out_start

    # Pre-compute scoring arrays
    n_active = len(ACTIVE)
    input_matrix = np.zeros((n_active, bits), dtype=np.float32)
    target_matrix = np.zeros((n_active, bits), dtype=np.float32)
    for row, i in enumerate(ACTIVE):
        input_matrix[row] = input_codes[i]
        target_matrix[row] = target_codes[TARGETS[i]]

    def score():
        Weff = net.W * net.mask
        charges = np.zeros((n_active, N), dtype=np.float32)
        acts = np.zeros((n_active, N), dtype=np.float32)
        for t in range(ticks):
            if t == 0:
                acts[:, :bits] = input_matrix
            raw = acts @ Weff + acts * 0.1
            np.nan_to_num(raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            charges += raw * 0.3
            charges *= net.leak
            acts = np.maximum(charges - net.threshold, 0.0)
            charges = np.clip(charges, -net.threshold * 2, net.threshold * 2)

        outputs = charges[:, out_start:out_start + bits]
        pred_bits = (outputs > 0.0).astype(float)

        bits_correct = (pred_bits == target_matrix).sum()
        total_bits = n_active * bits
        bit_acc = bits_correct / total_bits

        exact = sum(1 for row in range(n_active)
                    if np.array_equal(pred_bits[row], target_matrix[row]))
        exact_rate = exact / n_active

        return 0.5 * bit_acc + 0.5 * exact_rate, bit_acc, exact_rate

    t0 = time.time()
    best_sc, best_bit, best_exact = score()
    best_mask = net.mask.copy()
    best_W = net.W.copy()
    kept = 0
    stale = 0
    phase = 'STRUCTURE'
    switched = False

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

        sc, bit_acc, exact = score()

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

        if stale >= 6000:
            break

    elapsed = time.time() - t0
    result = {
        'config': config_name,
        'seed': train_seed,
        'bits': bits,
        'N': N,
        'bit_acc': best_bit,
        'exact': best_exact,
        'kept': kept,
        'time': elapsed,
    }
    # Live log via queue (cross-process safe)
    log_msg(log_q, f"{config_name:25s} seed={train_seed} bits={bits:2d} "
            f"bit_acc={best_bit*100:5.1f}% exact={best_exact*100:5.1f}% "
            f"time={elapsed:.0f}s")
    return result


# =============================================================
# Config builder
# =============================================================
INTERNAL_NEURONS = 112
SEEDS = [42, 77, 123]
BUDGET = 16000


def build_configs():
    """Build all test configurations."""
    configs = []

    # Phase A: Encoding type (8-bit for most, 32-bit for Hadamard)
    phase_a = {
        'byte_8bit':      (make_byte_encoding(VOCAB, 8), 8),
        'random_8bit':    (make_random_encoding(VOCAB, 8), 8),
        'spread_8bit':    (make_spread_encoding(VOCAB, 8), 8),
        'balanced50_8bit': (make_balanced_encoding(VOCAB, 8, 0.5), 8),
        'balanced25_8bit': (make_balanced_encoding(VOCAB, 8, 0.25), 8),
        'hadamard_32bit': (make_hadamard_encoding(VOCAB), 32),
    }

    for name, (codebook, bits) in phase_a.items():
        N = INTERNAL_NEURONS + 2 * bits
        # Use same codebook for both input and target encoding
        target_codes = np.zeros((VOCAB, bits), dtype=np.float32)
        for i in range(VOCAB):
            target_codes[i] = codebook[TARGETS[i]] if TARGETS[i] < VOCAB else codebook[0]
        configs.append({
            'name': name,
            'input_codes': codebook,
            'target_codes': codebook,
            'bits': bits,
            'N': N,
            'phase': 'A',
        })

    # Phase B: Bit width sweep (using spread encoding)
    for bits in [4, 6, 8, 12, 16]:
        codebook = make_spread_encoding(VOCAB, bits)
        N = INTERNAL_NEURONS + 2 * bits
        configs.append({
            'name': f'spread_{bits}bit',
            'input_codes': codebook,
            'target_codes': codebook,
            'bits': bits,
            'N': N,
            'phase': 'B',
        })

    # Phase C: Density control at 8 bits
    for density, label in [(0.125, '12pct'), (0.25, '25pct'), (0.375, '37pct'),
                            (0.5, '50pct'), (0.625, '62pct'), (0.75, '75pct')]:
        codebook = make_balanced_encoding(VOCAB, 8, density)
        N = INTERNAL_NEURONS + 2 * 8
        configs.append({
            'name': f'density_{label}_8bit',
            'input_codes': codebook,
            'target_codes': codebook,
            'bits': 8,
            'N': N,
            'phase': 'C',
        })

    return configs


# =============================================================
# Main
# =============================================================
def main():
    configs = build_configs()

    print("=" * 70)
    print("  ENCODING SWEEP TEST")
    print("=" * 70)
    print(f"  Vocab: {VOCAB} chars")
    print(f"  Internal neurons: {INTERNAL_NEURONS} (fixed)")
    print(f"  Budget: {BUDGET} attempts per config")
    print(f"  Seeds: {SEEDS}")
    print(f"  Configs: {len(configs)} (Phase A={sum(1 for c in configs if c['phase']=='A')}, "
          f"B={sum(1 for c in configs if c['phase']=='B')}, "
          f"C={sum(1 for c in configs if c['phase']=='C')})")
    print()

    # Print Hamming stats
    print("  HAMMING DISTANCE STATS:")
    print(f"  {'Config':25s} {'Bits':>5s} {'Min':>5s} {'Avg':>6s} {'Std':>5s}")
    print(f"  {'-'*25} {'-'*5} {'-'*5} {'-'*6} {'-'*5}")
    for c in configs:
        mn, avg, std = hamming_stats(c['input_codes'])
        print(f"  {c['name']:25s} {c['bits']:5d} {mn:5.1f} {avg:5.1f} {std:5.2f}")
    print()
    sys.stdout.flush()

    # Run all configs with LIVE LOGGING
    total_jobs = len(configs) * len(SEEDS)
    print(f"  Running {total_jobs} jobs (max 20 workers)...")
    print(f"  {'-' * 50}")
    sys.stdout.flush()

    all_results = []
    completed = 0

    with live_log('encoding_sweep') as (log_q, log_path):
        log_msg(log_q, f"Starting {total_jobs} jobs")
        with ProcessPoolExecutor(max_workers=20) as pool:
            futures = []
            for c in configs:
                for seed in SEEDS:
                    fut = pool.submit(
                        train_and_eval,
                        c['name'], c['input_codes'], c['target_codes'],
                        c['bits'], c['N'], BUDGET, seed, log_q=log_q)
                    futures.append((c['name'], c['phase'], seed, fut))

            for fut_tuple in futures:
                name, phase, seed, fut = fut_tuple
                try:
                    res = fut.result(timeout=600)
                    res['phase'] = phase
                    all_results.append(res)
                    completed += 1
                except Exception as ex:
                    log_msg(log_q, f"[ERR] {name} seed={seed}: {ex}")
        log_msg(log_q, f"All {completed}/{total_jobs} jobs complete")

    # Save CSV
    csv_path = os.path.join(os.path.dirname(__file__), 'encoding_sweep_results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'config', 'phase', 'seed', 'bits', 'N',
            'bit_acc', 'exact', 'kept', 'time'])
        writer.writeheader()
        for r in all_results:
            writer.writerow({k: r.get(k, '') for k in writer.fieldnames})
    print(f"\n  CSV: {csv_path}")

    # Summary by config
    from collections import defaultdict
    groups = defaultdict(list)
    for r in all_results:
        groups[r['config']].append(r)

    for phase_label, phase_key in [('PHASE A: Encoding Type', 'A'),
                                     ('PHASE B: Bit Width', 'B'),
                                     ('PHASE C: Density', 'C')]:
        print(f"\n{'=' * 70}")
        print(f"  {phase_label}")
        print(f"{'=' * 70}\n")

        ranked = []
        for cname, runs in groups.items():
            if runs[0].get('phase') != phase_key:
                continue
            bits_all = [r['bit_acc'] for r in runs]
            exacts = [r['exact'] for r in runs]
            times = [r['time'] for r in runs]
            ranked.append({
                'config': cname,
                'bits': runs[0]['bits'],
                'N': runs[0]['N'],
                'mean_bit': np.mean(bits_all),
                'mean_exact': np.mean(exacts),
                'std_exact': np.std(exacts),
                'mean_time': np.mean(times),
            })

        ranked.sort(key=lambda x: -x['mean_exact'])

        print(f"  {'Config':25s} {'Bits':>5s} {'N':>4s} "
              f"{'BitAcc':>7s} {'Exact':>7s} {'Std':>5s} {'Time':>6s}")
        print(f"  {'-'*25} {'-'*5} {'-'*4} {'-'*7} {'-'*7} {'-'*5} {'-'*6}")
        for r in ranked:
            print(f"  {r['config']:25s} {r['bits']:5d} {r['N']:4d} "
                  f"{r['mean_bit']*100:6.1f}% {r['mean_exact']*100:6.1f}% "
                  f"{r['std_exact']*100:4.1f}% {r['mean_time']:5.0f}s")

    # Overall winner
    all_ranked = []
    for cname, runs in groups.items():
        exacts = [r['exact'] for r in runs]
        all_ranked.append({'config': cname, 'mean_exact': np.mean(exacts),
                           'bits': runs[0]['bits']})
    all_ranked.sort(key=lambda x: -x['mean_exact'])

    print(f"\n{'=' * 70}")
    print(f"  OVERALL WINNER: {all_ranked[0]['config']} "
          f"({all_ranked[0]['mean_exact']*100:.1f}% exact, "
          f"{all_ranked[0]['bits']} bits)")
    print(f"{'=' * 70}", flush=True)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
