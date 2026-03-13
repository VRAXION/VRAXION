"""
Scale Test — v22 SelfWiringGraph at 1K-10K neurons
====================================================
Batch sparse forward for speed. MSE distribution scoring.
Tests whether more neurons help learn English bigram distributions.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
import time
from scipy import sparse
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
from v22_best_config import SelfWiringGraph, softmax

# ============================================================
#  English text corpus
# ============================================================

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
the best things in life are free. honesty is the best policy.
if at first you do not succeed try try again. rome was not built in a day.
the grass is always greener on the other side. curiosity killed the cat.
do not count your chickens before they hatch. a penny saved is a penny earned.
two wrongs do not make a right. when in rome do as the romans do.
the squeaky wheel gets the grease. you can not judge a book by its cover.
beauty is in the eye of the beholder. absence makes the heart grow fonder.
the journey of a thousand miles begins with a single step.
where there is smoke there is fire. still waters run deep.
a rolling stone gathers no moss. look before you leap.
necessity is the mother of invention. blood is thicker than water.
the apple does not fall far from the tree. there is no place like home.
you can lead a horse to water but you can not make it drink.
every dog has its day. do not put all your eggs in one basket.""".lower()

chars = sorted(set(TEXT))
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for c, i in char_to_idx.items()}
VOCAB = len(chars)

def compute_bigram_dist():
    counts = np.zeros((VOCAB, VOCAB), dtype=np.float32)
    for i in range(len(TEXT) - 1):
        a, b = TEXT[i], TEXT[i+1]
        if a in char_to_idx and b in char_to_idx:
            counts[char_to_idx[a], char_to_idx[b]] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return counts / row_sums

BIGRAM_DIST = compute_bigram_dist()
ACTIVE_INPUTS = [i for i in range(VOCAB) if BIGRAM_DIST[i].sum() > 0.01]


# ============================================================
#  Batch sparse evaluation
# ============================================================

def batch_forward_mse(net, true_dist, active_inputs, ticks=8):
    """Batch sparse forward: all inputs at once, MSE scoring."""
    V = net.V
    N = net.N
    n_inputs = len(active_inputs)

    Weff = net.W * net.mask
    Weff_sp = sparse.csr_matrix(Weff)

    # Build batch: n_inputs × N
    # Each row is one input's activation state
    charge_batch = np.zeros((n_inputs, N), dtype=np.float32)
    act_batch = np.zeros((n_inputs, N), dtype=np.float32)

    for t in range(ticks):
        if t == 0:
            # Inject inputs
            for i, inp in enumerate(active_inputs):
                act_batch[i, inp] = 1.0

        # Batch matmul: (n_inputs × N) @ (N × N).T = (n_inputs × N)
        raw_batch = act_batch @ Weff_sp.T + act_batch * 0.1

        # Capacitor dynamics
        charge_batch += raw_batch * 0.3
        charge_batch *= net.leak

        # Threshold
        act_batch = np.maximum(charge_batch - net.threshold, 0.0)

        # Clamp
        clamp = net.threshold * 2
        np.clip(charge_batch, -clamp, clamp, out=charge_batch)

    # Compute MSE for each input
    total_mse = 0.0
    for i, inp in enumerate(active_inputs):
        logits = charge_batch[i, :V]
        e = np.exp(logits - logits.max())
        pred = e / e.sum()
        total_mse += np.mean((pred - true_dist[inp]) ** 2)

    return -total_mse / n_inputs


def batch_eval_metrics(net, true_dist, active_inputs, ticks=8):
    """Full metrics using batch forward."""
    V = net.V
    N = net.N
    n_inputs = len(active_inputs)

    Weff = net.W * net.mask
    Weff_sp = sparse.csr_matrix(Weff)

    charge_batch = np.zeros((n_inputs, N), dtype=np.float32)
    act_batch = np.zeros((n_inputs, N), dtype=np.float32)

    for t in range(ticks):
        if t == 0:
            for i, inp in enumerate(active_inputs):
                act_batch[i, inp] = 1.0
        raw_batch = act_batch @ Weff_sp.T + act_batch * 0.1
        charge_batch += raw_batch * 0.3
        charge_batch *= net.leak
        act_batch = np.maximum(charge_batch - net.threshold, 0.0)
        clamp = net.threshold * 2
        np.clip(charge_batch, -clamp, clamp, out=charge_batch)

    total_mse = 0.0
    top1 = 0
    top3 = 0
    per_char = {}

    for i, inp in enumerate(active_inputs):
        logits = charge_batch[i, :V]
        e = np.exp(logits - logits.max())
        pred = e / e.sum()
        target = true_dist[inp]

        total_mse += np.mean((pred - target) ** 2)
        if np.argmax(pred) == np.argmax(target):
            top1 += 1
        if np.argmax(target) in set(np.argsort(-pred)[:3]):
            top3 += 1
        per_char[inp] = pred.copy()

    n = len(active_inputs)
    return {
        'mse': total_mse / n,
        'top1': top1 / n,
        'top3': top3 / n,
        'per_char': per_char,
    }


# ============================================================
#  Training
# ============================================================

def train_config(label, n_neurons, seed, max_attempts=8000, ticks=8):
    """Train with batch sparse MSE scoring."""
    np.random.seed(seed)
    random.seed(seed)

    V = VOCAB
    net = SelfWiringGraph(n_neurons, V)
    td = BIGRAM_DIST
    ai = ACTIVE_INPUTS

    score = batch_forward_mse(net, td, ai, ticks)
    best_score = score
    kept = 0
    stale = 0
    phase = "STRUCTURE"
    switched = False
    trajectory = []

    t0 = time.time()

    for att in range(max_attempts):
        # Log every 2000
        if att % 2000 == 0:
            m = batch_eval_metrics(net, td, ai, ticks)
            trajectory.append({
                'step': att, 'mse': m['mse'], 'top1': m['top1'],
                'top3': m['top3'],
                'accept_rate': (kept / max(att, 1)) * 100,
            })
            elapsed_so_far = time.time() - t0
            print(f"    [{label}] step={att} mse={m['mse']:.6f} "
                  f"top1={m['top1']*100:.1f}% accept={trajectory[-1]['accept_rate']:.1f}% "
                  f"t={elapsed_so_far:.0f}s", flush=True)

        state = net.save_state()

        if phase == "STRUCTURE":
            net.mutate_structure(0.05)
        else:
            if random.random() < 0.3:
                net.mutate_structure(0.02)
            else:
                net.mutate_weights()

        net.self_wire()
        new_score = batch_forward_mse(net, td, ai, ticks)

        if new_score > score:
            score = new_score
            kept += 1
            stale = 0
            best_score = max(best_score, score)
        else:
            net.restore_state(state)
            stale += 1

        if phase == "STRUCTURE" and stale > 2500 and not switched:
            phase = "BOTH"
            switched = True
            stale = 0

        if stale >= 3500:
            break

    elapsed = time.time() - t0
    accept_rate = (kept / max(att + 1, 1)) * 100

    final = batch_eval_metrics(net, td, ai, ticks)

    return {
        'label': label,
        'seed': seed,
        'n_neurons': n_neurons,
        'final_mse': final['mse'],
        'final_top1': final['top1'],
        'final_top3': final['top3'],
        'accept_rate': accept_rate,
        'kept': kept,
        'attempts': att + 1,
        'time': elapsed,
        'trajectory': trajectory,
        'per_char': final['per_char'],
        'conns': net.count_connections(),
    }


def worker(args):
    return train_config(**args)


# ============================================================
#  Display
# ============================================================

def print_char_dist(ch_idx, pred, true_dist, top_n=6):
    ch = idx_to_char.get(ch_idx, '?')
    target = true_dist[ch_idx]
    top_true = np.argsort(-target)[:top_n]
    top_pred = np.argsort(-pred)[:top_n]
    true_str = " ".join(f"{idx_to_char.get(i,'?')}={target[i]*100:.0f}%"
                        for i in top_true if target[i] > 0.01)
    pred_str = " ".join(f"{idx_to_char.get(i,'?')}={pred[i]*100:.0f}%"
                        for i in top_pred)
    print(f"    '{ch}': TRUE=[{true_str}]")
    print(f"    {' '*len(ch)+' '}: PRED=[{pred_str}]")


# ============================================================
#  Main
# ============================================================

SEEDS = [42, 123, 777]  # 3 seeds for speed

if __name__ == "__main__":
    n_workers = os.cpu_count() or 24

    print(f"  Scale Test — v22 SelfWiringGraph (batch sparse)")
    print(f"  {'='*60}")
    print(f"  Vocab: {VOCAB}, Active: {len(ACTIVE_INPUTS)}")
    print(f"  CPU cores: {n_workers}")

    # Memory check
    for N in [160, 1000, 5000, 10000]:
        mem_mb = N * N * 4 * 2 / 1e6  # mask + W dense
        nnz = int(N * N * 0.06)
        sparse_mb = nnz * 12 / 1e6  # value + row + col
        print(f"    {N:6d}n: dense={mem_mb:.0f}MB sparse={sparse_mb:.0f}MB "
              f"conns~{nnz:,}")

    # =========================================================
    # Phase 1: Scale comparison
    # =========================================================
    # Start small, go bigger. Limit attempts to keep runtime sane.
    configs = []
    for n_neurons, max_att in [(160, 8000), (1000, 8000), (5000, 4000)]:
        for seed in SEEDS:
            configs.append(dict(
                label=f'{n_neurons}n',
                n_neurons=n_neurons, seed=seed,
                max_attempts=max_att, ticks=8))

    print(f"\n{'#'*65}")
    print(f"  Phase 1: Scale comparison (160n/1Kn/5Kn, MSE scoring)")
    print(f"  {len(configs)} jobs, parallel")
    print(f"{'#'*65}", flush=True)

    # Run smaller ones first, then bigger
    # Group by size to manage memory
    for n_target in [160, 1000, 5000]:
        batch = [c for c in configs if c['n_neurons'] == n_target]
        # Limit workers for large models (memory)
        if n_target >= 5000:
            workers = min(4, n_workers)
        elif n_target >= 1000:
            workers = min(8, n_workers)
        else:
            workers = n_workers

        print(f"\n  Running {len(batch)} jobs @ {n_target}n "
              f"({workers} workers)...", flush=True)
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=workers) as pool:
            batch_results = list(pool.map(worker, batch))
        print(f"  {n_target}n done in {time.time()-t0:.0f}s", flush=True)

        # Print results immediately
        groups = defaultdict(list)
        for r in batch_results:
            groups[r['label']].append(r)
        for label in groups:
            runs = groups[label]
            mses = [r['final_mse'] for r in runs]
            t1s = [r['final_top1'] for r in runs]
            t3s = [r['final_top3'] for r in runs]
            ars = [r['accept_rate'] for r in runs]
            tms = [r['time'] for r in runs]
            conns = [r['conns'] for r in runs]
            print(f"  {label:<8s} MSE={np.mean(mses):.6f} "
                  f"Top1={np.mean(t1s)*100:.1f}% "
                  f"Top3={np.mean(t3s)*100:.1f}% "
                  f"Accept={np.mean(ars):.2f}% "
                  f"Conns={np.mean(conns):.0f} "
                  f"Time={np.mean(tms):.0f}s")

    print(f"\n  {'='*65}")
    print(f"  DONE")
    print(f"  {'='*65}", flush=True)
