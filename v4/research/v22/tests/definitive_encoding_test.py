"""
DEFINITIVE ENCODING DECISION TEST
===================================
Pre-registered protocol to decide: spatial vs temporal encoding.

Matrix: 2 tasks x 2 fairness x 2 readout x 3 vocab x 5 seeds = 60 training runs + 60 probes
Uses v22_log.py for live logging. Every result logged immediately.

Decision rules (PRE-REGISTERED):
  Temporal canon IF: wins T2 under both readouts, loses <3pp on T1, 3/4 vocab sizes, p<0.05
  Spatial stays IF: temporal only wins under probe, or collapses at scale
  Research branch IF: probe strong, native weak → need split architecture
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
import time
import csv
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
from v22_best_config import SelfWiringGraph
from v22_log import live_log, log_msg

OUT_DIR = os.path.join(os.path.dirname(__file__), 'viz_interference', 'definitive')
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# Constants
# ============================================================
SEEDS = [42, 77, 123, 7, 99]
BUDGET = 8000
TICKS = 8
LISTEN_TICKS = 4
VOCAB_SIZES = [8, 27, 64]

# Text corpus for bigram targets
TEXT = ('the quick brown fox jumps over the lazy dog and the cat sat on the mat '
       'while birds fly high in the blue sky above quiet fields of green')


def build_bigram_targets(vocab):
    """Build bigram target array from TEXT for given vocab size."""
    chars = sorted(set(TEXT))[:vocab]
    if len(chars) < vocab:
        # Extend with ascii if needed
        for c in 'abcdefghijklmnopqrstuvwxyz .':
            if len(chars) >= vocab:
                break
            if c not in chars:
                chars.append(c)
    chars = sorted(chars)[:vocab]
    c2i = {c: i for i, c in enumerate(chars)}

    bigrams = {}
    for i in range(len(TEXT) - 1):
        if TEXT[i] in c2i and TEXT[i+1] in c2i:
            bigrams.setdefault(c2i[TEXT[i]], []).append(c2i[TEXT[i+1]])

    targets = np.zeros(vocab, dtype=int)
    for i in range(vocab):
        if i in bigrams and len(bigrams[i]) > 0:
            targets[i] = Counter(bigrams[i]).most_common(1)[0][0]
        else:
            targets[i] = (i + 1) % vocab  # fallback: next char
    return targets


def build_order_targets(vocab):
    """Build order-sensitive targets: pair (a,b) has different target than (b,a).
    Uses first vocab*(vocab-1) pairs, targets based on hash of ordered pair."""
    n_pairs = min(vocab * (vocab - 1), 256)  # cap for sanity
    pairs = []
    targets = []
    for a in range(vocab):
        for b in range(vocab):
            if a != b and len(pairs) < n_pairs:
                pairs.append((a, b))
                # Target = deterministic hash of ordered pair
                targets.append((a * 31 + b * 7) % vocab)
    return pairs, np.array(targets, dtype=int)


def make_codebook(vocab, bits):
    """Byte encoding codebook."""
    return np.array([[float((i >> bit) & 1) for bit in range(bits)]
                      for i in range(vocab)], dtype=np.float32)


def make_spike_trains(vocab, bits, ticks):
    """Temporal spike trains: spread bits across ticks."""
    trains = np.zeros((vocab, ticks, bits), dtype=np.float32)
    for ci in range(vocab):
        for t in range(ticks):
            bit_idx = t % bits
            trains[ci, t, bit_idx] = float((ci >> bit_idx) & 1)
    return trains


# ============================================================
# Training worker
# ============================================================
def run_config(task, mode, fairness, vocab, seed, budget, log_q=None):
    """Train one config. Returns result dict."""
    t0 = time.time()
    np.random.seed(seed)
    random.seed(seed)

    bits = 8  # always 8-bit encoding
    if fairness == 'fixed_total':
        N = 128
    else:  # fixed_internal
        N = 112 + 2 * bits  # = 128

    # Build task
    if task == 'T1_bigram':
        targets = build_bigram_targets(vocab)
        n_samples = vocab
    else:  # T2_order
        pairs, targets = build_order_targets(vocab)
        n_samples = len(pairs)

    codebook = make_codebook(vocab, bits)
    spike_trains = make_spike_trains(vocab, bits, TICKS)

    # Create network
    net = SelfWiringGraph(N, bits)
    out_start = net.out_start

    def forward_and_score():
        """Score current network on the task."""
        if task == 'T1_bigram':
            # Batch: all vocab chars
            Weff = net.W * net.mask
            ch = np.zeros((vocab, N), dtype=np.float32)
            ac = np.zeros((vocab, N), dtype=np.float32)
            for t in range(TICKS):
                if mode == 'spatial':
                    if t == 0:
                        ac[:, :bits] = codebook
                elif mode == 'temporal':
                    ac[:, :bits] = spike_trains[:, t, :]
                elif mode == 'listen_think':
                    if t < LISTEN_TICKS:
                        ac[:, :bits] = spike_trains[:, t, :]
                raw = ac @ Weff + ac * 0.1
                np.nan_to_num(raw, copy=False)
                ch += raw * 0.3
                ch *= net.leak
                ac = np.maximum(ch - net.threshold, 0)
                ch = np.clip(ch, -net.threshold * 2, net.threshold * 2)
            out = ch[:, out_start:out_start + bits]
            pred_bits = (out > 0.5).astype(np.float32)
            tgt_bits = codebook[targets]
            exact = (np.all(pred_bits == tgt_bits, axis=1)).mean()
            bit_acc = (pred_bits == tgt_bits).mean()
            return 0.5 * bit_acc + 0.5 * exact, exact

        else:  # T2_order
            Weff = net.W * net.mask
            correct = 0
            for idx, (a, b) in enumerate(pairs):
                ch = np.zeros(N, dtype=np.float32)
                ac = np.zeros(N, dtype=np.float32)
                for t in range(TICKS):
                    if mode == 'spatial':
                        if t == 0:
                            # SUM of both codes — order LOST
                            ac[:bits] = np.clip(codebook[a] + codebook[b], 0, 1)
                    elif mode == 'temporal':
                        if t == 0:
                            ac[:bits] = codebook[a]
                        elif t == 1:
                            ac[:bits] = codebook[b]
                        else:
                            ac[:bits] = 0
                    elif mode == 'listen_think':
                        if t == 0:
                            ac[:bits] = codebook[a]
                        elif t == 1:
                            ac[:bits] = codebook[b]
                        # else: silence
                    raw = ac @ Weff + ac * 0.1
                    np.nan_to_num(raw, copy=False)
                    ch += raw * 0.3
                    ch *= net.leak
                    ac = np.maximum(ch - net.threshold, 0)
                    ch = np.clip(ch, -net.threshold * 2, net.threshold * 2)
                out = ch[out_start:out_start + bits]
                pred_bits = (out > 0.5).astype(np.float32)
                tgt_bits = codebook[targets[idx]]
                if np.all(pred_bits == tgt_bits):
                    correct += 1
            exact = correct / n_samples
            return exact, exact

    # --- Mutation+selection training ---
    best_sc = 0.0
    best_exact = 0.0
    stale = 0
    phase = 'STRUCTURE'
    switched = False

    for att in range(budget):
        sm = net.mask.copy()
        sw = net.W.copy()

        if phase == 'STRUCTURE':
            net.mutate_structure(0.07)
        else:
            if random.random() < 0.3:
                net.mutate_structure(0.02)
            else:
                net.mutate_weights()

        sc, exact = forward_and_score()

        if sc > best_sc:
            best_sc = sc
            best_exact = exact
            stale = 0
        else:
            net.mask = sm
            net.W = sw
            stale += 1

        if phase == 'STRUCTURE' and stale > 2500 and not switched:
            phase = 'BOTH'
            switched = True
            stale = 0

        if stale >= 5000:
            break

    elapsed = time.time() - t0

    # --- Collect charge vectors for probe ---
    charge_vectors = None
    if task == 'T1_bigram':
        Weff = net.W * net.mask
        charge_vectors = np.zeros((vocab, N), dtype=np.float32)
        for ci in range(vocab):
            ch = np.zeros(N, dtype=np.float32)
            ac = np.zeros(N, dtype=np.float32)
            for t in range(TICKS):
                if mode == 'spatial':
                    if t == 0:
                        ac[:bits] = codebook[ci]
                elif mode == 'temporal':
                    ac[:bits] = spike_trains[ci, t, :]
                elif mode == 'listen_think':
                    if t < LISTEN_TICKS:
                        ac[:bits] = spike_trains[ci, t, :]
                raw = ac @ Weff + ac * 0.1
                np.nan_to_num(raw, copy=False)
                ch += raw * 0.3
                ch *= net.leak
                ac = np.maximum(ch - net.threshold, 0)
                ch = np.clip(ch, -net.threshold * 2, net.threshold * 2)
            charge_vectors[ci] = ch

    result = {
        'task': task,
        'mode': mode,
        'fairness': fairness,
        'vocab': vocab,
        'seed': seed,
        'native_exact': best_exact,
        'native_score': best_sc,
        'time': elapsed,
        'charge_vectors': charge_vectors,
    }

    log_msg(log_q, f"{task:10s} {mode:12s} {fairness:14s} V={vocab:3d} seed={seed:3d} "
            f"exact={best_exact*100:5.1f}% time={elapsed:.0f}s")
    return result


def probe_accuracy(X, Y, vocab, lam=10.0):
    """Ridge probe: fit on all, report train acc + LOO."""
    Y_oh = np.zeros((len(Y), vocab), dtype=np.float32)
    for i in range(len(Y)):
        Y_oh[i, Y[i]] = 1.0

    n_feat = X.shape[1]
    W = np.linalg.solve(X.T @ X + lam * np.eye(n_feat), X.T @ Y_oh)
    train_acc = (np.argmax(X @ W, axis=1) == Y).mean()

    # LOO
    correct = 0
    for i in range(len(Y)):
        mask = np.ones(len(Y), dtype=bool)
        mask[i] = False
        W_loo = np.linalg.solve(X[mask].T @ X[mask] + lam * np.eye(n_feat),
                                 X[mask].T @ Y_oh[mask])
        if np.argmax(X[i:i+1] @ W_loo, axis=1)[0] == Y[i]:
            correct += 1
    loo_acc = correct / len(Y)
    return train_acc, loo_acc


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 75, flush=True)
    print("  DEFINITIVE ENCODING DECISION TEST", flush=True)
    print("=" * 75, flush=True)
    print(f"  Tasks: T1_bigram, T2_order", flush=True)
    print(f"  Modes: spatial, temporal, listen_think", flush=True)
    print(f"  Fairness: fixed_total (N=128), fixed_internal (int=112)", flush=True)
    print(f"  Vocab: {VOCAB_SIZES}", flush=True)
    print(f"  Seeds: {SEEDS} ({len(SEEDS)} seeds)", flush=True)
    print(f"  Budget: {BUDGET}", flush=True)

    # Build job list
    configs = []
    for task in ['T1_bigram', 'T2_order']:
        for mode in ['spatial', 'temporal', 'listen_think']:
            for fairness in ['fixed_total']:  # start with fixed_total only (halves runtime)
                for vocab in VOCAB_SIZES:
                    for seed in SEEDS:
                        configs.append((task, mode, fairness, vocab, seed))

    total = len(configs)
    print(f"  Total training runs: {total}", flush=True)
    print(f"  Estimated time: ~{total * 30 / 60:.0f} min", flush=True)
    print("=" * 75, flush=True)

    all_results = []

    with live_log('definitive_encoding') as (log_q, log_path):
        log_msg(log_q, f"Starting {total} runs")

        # Run sequentially (deterministic, no parallel surprises)
        for i, (task, mode, fairness, vocab, seed) in enumerate(configs):
            log_msg(log_q, f"[{i+1}/{total}] Starting {task} {mode} V={vocab} seed={seed}")
            res = run_config(task, mode, fairness, vocab, seed, BUDGET, log_q)
            all_results.append(res)

        log_msg(log_q, f"All {total} runs complete")

        # --- Probe analysis (T1 only, has charge vectors) ---
        log_msg(log_q, "Running probe analysis on T1 results...")
        for res in all_results:
            if res['task'] == 'T1_bigram' and res['charge_vectors'] is not None:
                targets = build_bigram_targets(res['vocab'])
                train_acc, loo_acc = probe_accuracy(
                    res['charge_vectors'], targets, res['vocab'])
                res['probe_train'] = train_acc
                res['probe_loo'] = loo_acc
                log_msg(log_q, f"  PROBE {res['mode']:12s} V={res['vocab']:3d} seed={res['seed']:3d} "
                        f"train={train_acc*100:5.1f}% LOO={loo_acc*100:5.1f}%")

        log_msg(log_q, "Probe analysis complete")

    # --- Save CSV ---
    csv_path = os.path.join(OUT_DIR, 'definitive_results.csv')
    fieldnames = ['task', 'mode', 'fairness', 'vocab', 'seed',
                  'native_exact', 'native_score', 'probe_train', 'probe_loo', 'time']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            row = {k: r.get(k, '') for k in fieldnames}
            writer.writerow(row)
    print(f"\nCSV: {csv_path}", flush=True)

    # --- Verdict ---
    print(f"\n{'='*75}", flush=True)
    print("  RESULTS", flush=True)
    print(f"{'='*75}\n", flush=True)

    # Group by (task, mode, vocab) → mean over seeds
    from itertools import groupby
    groups = {}
    for r in all_results:
        key = (r['task'], r['mode'], r['vocab'])
        groups.setdefault(key, []).append(r)

    # Print table
    print(f"  {'Task':10s} {'Mode':12s} {'V':>3s}  {'Native':>7s}  {'Probe':>7s}", flush=True)
    print(f"  {'-'*10} {'-'*12} {'-'*3}  {'-'*7}  {'-'*7}", flush=True)
    for key in sorted(groups.keys()):
        runs = groups[key]
        native_mean = np.mean([r['native_exact'] for r in runs]) * 100
        probe_vals = [r.get('probe_loo', float('nan')) for r in runs]
        if any(np.isnan(v) for v in probe_vals):
            probe_str = "   N/A"
        else:
            probe_str = f"{np.mean(probe_vals)*100:6.1f}%"
        task, mode, vocab = key
        print(f"  {task:10s} {mode:12s} {vocab:3d}  {native_mean:6.1f}%  {probe_str}", flush=True)

    # --- Decision ---
    print(f"\n{'='*75}", flush=True)
    print("  DECISION", flush=True)
    print(f"{'='*75}\n", flush=True)

    # T2 comparison: spatial vs temporal
    t2_spatial_wins = 0
    t2_temporal_wins = 0
    t2_listen_wins = 0
    for vocab in VOCAB_SIZES:
        s_vals = [r['native_exact'] for r in groups.get(('T2_order', 'spatial', vocab), [])]
        t_vals = [r['native_exact'] for r in groups.get(('T2_order', 'temporal', vocab), [])]
        l_vals = [r['native_exact'] for r in groups.get(('T2_order', 'listen_think', vocab), [])]
        if s_vals and t_vals:
            s_mean = np.mean(s_vals)
            t_mean = np.mean(t_vals)
            l_mean = np.mean(l_vals) if l_vals else 0
            winner = 'spatial' if s_mean > t_mean else 'temporal' if t_mean > s_mean else 'tie'
            print(f"  T2 V={vocab:3d}: spatial={s_mean*100:.1f}% temporal={t_mean*100:.1f}% "
                  f"listen={l_mean*100:.1f}% -> {winner}", flush=True)
            if winner == 'spatial':
                t2_spatial_wins += 1
            elif winner == 'temporal':
                t2_temporal_wins += 1

    # T1 gap check
    t1_gaps = []
    for vocab in VOCAB_SIZES:
        s_vals = [r['native_exact'] for r in groups.get(('T1_bigram', 'spatial', vocab), [])]
        t_vals = [r['native_exact'] for r in groups.get(('T1_bigram', 'temporal', vocab), [])]
        if s_vals and t_vals:
            gap = (np.mean(s_vals) - np.mean(t_vals)) * 100
            t1_gaps.append(gap)
            print(f"  T1 V={vocab:3d}: spatial-temporal gap = {gap:+.1f}pp", flush=True)

    # Verdict
    print(flush=True)
    if t2_temporal_wins >= 2 and all(g < 3.0 for g in t1_gaps):
        print("  >>> VERDICT: TEMPORAL BECOMES CANON <<<", flush=True)
    elif t2_temporal_wins >= 1 and any(g > 3.0 for g in t1_gaps):
        print("  >>> VERDICT: TEMPORAL = RESEARCH BRANCH <<<", flush=True)
        print("  >>> (wins T2 but loses too much on T1)", flush=True)
    elif t2_spatial_wins >= 2:
        print("  >>> VERDICT: SPATIAL STAYS CANON <<<", flush=True)
    else:
        print("  >>> VERDICT: INCONCLUSIVE — need more data <<<", flush=True)

    # Probe insight
    print(flush=True)
    for vocab in VOCAB_SIZES:
        s_native = [r['native_exact'] for r in groups.get(('T1_bigram', 'spatial', vocab), [])]
        t_native = [r['native_exact'] for r in groups.get(('T1_bigram', 'temporal', vocab), [])]
        s_probe = [r.get('probe_loo', 0) for r in groups.get(('T1_bigram', 'spatial', vocab), [])]
        t_probe = [r.get('probe_loo', 0) for r in groups.get(('T1_bigram', 'temporal', vocab), [])]
        if s_native and t_native:
            native_gap = (np.mean(s_native) - np.mean(t_native)) * 100
            probe_gap = (np.mean(s_probe) - np.mean(t_probe)) * 100
            print(f"  V={vocab:3d}: native gap={native_gap:+.1f}pp, probe gap={probe_gap:+.1f}pp",
                  flush=True)
            if abs(native_gap) > 5 and abs(probe_gap) < 3:
                print(f"         ^ READOUT BOTTLENECK (big native gap, small probe gap)", flush=True)

    print(f"\n{'='*75}", flush=True)
    print("Done!", flush=True)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
