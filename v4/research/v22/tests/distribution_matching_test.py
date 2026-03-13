"""
Distribution Matching Scoring — v22 SelfWiringGraph
=====================================================
Instead of scoring "did you get THE right answer?",
score "how close is your output DISTRIBUTION to the true distribution?"

The key insight: English bigrams are NOT deterministic.
After 't', the true distribution is h=45%, space=13%, i=13%, ...
The scoring should reward matching this CURVE, not guessing one letter.

Scoring: negative MSE between true_dist and softmax output.
MSE is smooth, hyperparameter-free, and every bin improvement counts —
ideal for evolutionary (mutation+selection) search.

Phases:
  A) MSE vs combined vs hybrid scoring on English bigrams
  B) Synthetic ambiguity sweep (0% to 100%)
  C) Per-character distribution visualization
  D) Final validation with winner
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
import time
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
from v22_best_config import SelfWiringGraph, softmax

# ============================================================
#  English text corpus (reused from english_text_test.py)
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

def build_bigrams(text):
    pairs = []
    for i in range(len(text) - 1):
        if text[i] in char_to_idx and text[i+1] in char_to_idx:
            pairs.append((char_to_idx[text[i]], char_to_idx[text[i+1]]))
    return pairs

ALL_PAIRS = build_bigrams(TEXT)

def compute_bigram_dist():
    counts = np.zeros((VOCAB, VOCAB), dtype=np.float32)
    for a, b in ALL_PAIRS:
        counts[a, b] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return counts / row_sums

BIGRAM_DIST = compute_bigram_dist()

# Which input chars have observed bigrams?
ACTIVE_INPUTS = [i for i in range(VOCAB) if BIGRAM_DIST[i].sum() > 0.01]

# ============================================================
#  Scoring functions
# ============================================================

def score_mse(net, true_dist, vocab, active_inputs, ticks=8):
    """Negative MSE between true and predicted distributions.
    Lower MSE = better, so we negate for maximization."""
    total_mse = 0.0
    for inp in active_inputs:
        net.reset()
        world = np.zeros(vocab, dtype=np.float32)
        world[inp] = 1.0
        logits = net.forward(world, ticks)
        pred = softmax(logits[:vocab])
        total_mse += np.mean((pred - true_dist[inp]) ** 2)
    return -total_mse / len(active_inputs)


def score_combined(net, pairs, vocab, ticks=8, sample_n=120):
    """Old combined scoring: 0.5*accuracy + 0.5*target_prob on sampled pairs."""
    net.reset()
    total_score = 0.0
    sample = random.sample(pairs, min(sample_n, len(pairs)))
    for p in range(2):
        for inp_idx, tgt_idx in sample:
            world = np.zeros(vocab, dtype=np.float32)
            world[inp_idx] = 1.0
            logits = net.forward(world, ticks)
            probs = softmax(logits[:vocab])
            if p == 1:
                acc_i = 1.0 if np.argmax(probs) == tgt_idx else 0.0
                tp = float(probs[tgt_idx])
                total_score += 0.5 * acc_i + 0.5 * tp
    return total_score / len(sample)


def score_hybrid(net, true_dist, vocab, active_inputs, ticks=8):
    """Hybrid: 0.5*top1_match + 0.5*(-MSE_normalized)."""
    total_mse = 0.0
    top1_match = 0
    for inp in active_inputs:
        net.reset()
        world = np.zeros(vocab, dtype=np.float32)
        world[inp] = 1.0
        logits = net.forward(world, ticks)
        pred = softmax(logits[:vocab])
        mse = np.mean((pred - true_dist[inp]) ** 2)
        total_mse += mse
        if np.argmax(pred) == np.argmax(true_dist[inp]):
            top1_match += 1
    mean_mse = total_mse / len(active_inputs)
    top1_acc = top1_match / len(active_inputs)
    # Normalize MSE to [0,1] range (random MSE ~ 0.0083)
    mse_score = 1.0 - mean_mse / 0.01
    return 0.5 * top1_acc + 0.5 * mse_score


# ============================================================
#  Evaluation metrics (for reporting, not for scoring)
# ============================================================

def eval_metrics(net, true_dist, vocab, active_inputs, ticks=8):
    """Compute full metrics for reporting."""
    total_mse = 0.0
    total_kl = 0.0
    top1 = 0
    top3 = 0
    per_char = {}

    for inp in active_inputs:
        net.reset()
        world = np.zeros(vocab, dtype=np.float32)
        world[inp] = 1.0
        logits = net.forward(world, ticks)
        pred = softmax(logits[:vocab])
        target = true_dist[inp]

        mse = np.mean((pred - target) ** 2)
        total_mse += mse

        # KL(true || pred) with clipping
        mask = target > 0
        kl = np.sum(target[mask] * np.log(target[mask] / np.clip(pred[mask], 1e-10, None)))
        total_kl += kl

        if np.argmax(pred) == np.argmax(target):
            top1 += 1
        top3_pred = set(np.argsort(-pred)[:3])
        if np.argmax(target) in top3_pred:
            top3 += 1

        per_char[inp] = pred.copy()

    n = len(active_inputs)
    return {
        'mse': total_mse / n,
        'kl': total_kl / n,
        'top1': top1 / n,
        'top3': top3 / n,
        'per_char': per_char,
    }


# ============================================================
#  Synthetic ambiguity task
# ============================================================

def make_ambiguous_task(n_classes, ambiguity, seed):
    """Create a synthetic task with controlled ambiguity.
    ambiguity=0: deterministic (A->B always)
    ambiguity=0.5: A->B 50%, A->C 50%
    """
    rng = np.random.RandomState(seed + 999)
    perm = rng.permutation(n_classes)
    true_dist = np.zeros((n_classes, n_classes), dtype=np.float32)
    for i in range(n_classes):
        primary = perm[i]
        secondary = perm[(i + 1) % n_classes]
        true_dist[i, primary] = 1.0 - ambiguity
        if ambiguity > 0:
            true_dist[i, secondary] = ambiguity
    return true_dist, perm


# ============================================================
#  Training
# ============================================================

def train_config(label, n_neurons, seed, max_attempts=8000, ticks=8,
                 score_mode='mse', true_dist=None, pairs=None,
                 active_inputs=None, vocab=None, log_interval=2000,
                 viz_chars=None):
    """Train with configurable scoring."""
    np.random.seed(seed)
    random.seed(seed)

    V = vocab or VOCAB
    net = SelfWiringGraph(n_neurons, V)
    td = true_dist if true_dist is not None else BIGRAM_DIST
    pr = pairs if pairs is not None else ALL_PAIRS
    ai = active_inputs if active_inputs is not None else ACTIVE_INPUTS

    def evaluate():
        if score_mode == 'mse':
            return score_mse(net, td, V, ai, ticks)
        elif score_mode == 'combined':
            return score_combined(net, pr, V, ticks)
        elif score_mode == 'hybrid':
            return score_hybrid(net, td, V, ai, ticks)
        else:
            raise ValueError(f"Unknown score_mode: {score_mode}")

    score = evaluate()
    best_score = score
    kept = 0
    stale = 0
    phase = "STRUCTURE"
    switched = False
    trajectory = []

    t0 = time.time()

    for att in range(max_attempts):
        # Log trajectory
        if log_interval and (att % log_interval == 0):
            m = eval_metrics(net, td, V, ai, ticks)
            entry = {'step': att, 'mse': m['mse'], 'top1': m['top1'],
                     'top3': m['top3'], 'kl': m['kl'],
                     'accept_rate': (kept / max(att, 1)) * 100}
            # Per-char viz
            if viz_chars:
                for ch_idx in viz_chars:
                    if ch_idx in m['per_char']:
                        entry[f'pred_{ch_idx}'] = m['per_char'][ch_idx].copy()
            trajectory.append(entry)

        state = net.save_state()

        if phase == "STRUCTURE":
            net.mutate_structure(0.05)
        else:
            if random.random() < 0.3:
                net.mutate_structure(0.02)
            else:
                net.mutate_weights()

        net.self_wire()
        new_score = evaluate()

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

    # Final metrics
    final = eval_metrics(net, td, V, ai, ticks)

    return {
        'label': label,
        'seed': seed,
        'score_mode': score_mode,
        'final_mse': final['mse'],
        'final_kl': final['kl'],
        'final_top1': final['top1'],
        'final_top3': final['top3'],
        'accept_rate': accept_rate,
        'kept': kept,
        'attempts': att + 1,
        'time': elapsed,
        'trajectory': trajectory,
        'per_char': final['per_char'],
    }


def worker(args):
    return train_config(**args)


# ============================================================
#  Display helpers
# ============================================================

def print_char_dist(ch_idx, pred, true_dist, vocab, top_n=6):
    """Print predicted vs true distribution for a character."""
    ch = idx_to_char.get(ch_idx, '?')
    target = true_dist[ch_idx]
    top_true = np.argsort(-target)[:top_n]
    top_pred = np.argsort(-pred)[:top_n]

    true_str = " ".join(f"{idx_to_char.get(i,'?')}={target[i]*100:.0f}%" for i in top_true if target[i] > 0.01)
    pred_str = " ".join(f"{idx_to_char.get(i,'?')}={pred[i]*100:.0f}%" for i in top_pred)
    print(f"    '{ch}': TRUE=[{true_str}]")
    print(f"    {' '*len(ch)+' '}: PRED=[{pred_str}]")


# ============================================================
#  Main
# ============================================================

SEEDS = [42, 123, 777, 1337, 2024]

if __name__ == "__main__":
    n_workers = os.cpu_count() or 24

    # Compute reference values
    random_mse = 0.0
    for inp in ACTIVE_INPUTS:
        uniform = np.ones(VOCAB, dtype=np.float32) / VOCAB
        random_mse += np.mean((uniform - BIGRAM_DIST[inp]) ** 2)
    random_mse /= len(ACTIVE_INPUTS)

    oracle_correct = sum(1 for a, b in ALL_PAIRS if np.argmax(BIGRAM_DIST[a]) == b)
    oracle_acc = oracle_correct / len(ALL_PAIRS)

    print(f"  Distribution Matching Scoring -- v22 SelfWiringGraph")
    print(f"  {'='*60}")
    print(f"  Vocab: {VOCAB} chars, Active inputs: {len(ACTIVE_INPUTS)}")
    print(f"  Bigram pairs: {len(ALL_PAIRS)}")
    print(f"  Random MSE: {random_mse:.6f} | Perfect MSE: 0.000000")
    print(f"  Oracle top-1: {oracle_acc*100:.1f}%")
    print(f"  Random top-1: {100/VOCAB:.1f}%")
    print(f"  CPU cores: {n_workers}")

    # Viz chars: 't', ' ', 'e'
    viz_chars = [char_to_idx.get('t'), char_to_idx.get(' '), char_to_idx.get('e')]
    viz_chars = [c for c in viz_chars if c is not None]

    # =========================================================
    # PHASE A: Scoring comparison on English bigrams
    # =========================================================
    print(f"\n{'#'*65}")
    print(f"  PHASE A: Scoring comparison (160n, 8K att, 5 seeds)")
    print(f"{'#'*65}", flush=True)

    configs_a = []
    for mode in ['mse', 'combined', 'hybrid']:
        for seed in SEEDS:
            configs_a.append(dict(
                label=f'english_{mode}',
                n_neurons=160, seed=seed, max_attempts=8000, ticks=8,
                score_mode=mode, log_interval=2000, viz_chars=viz_chars))

    print(f"  Running {len(configs_a)} jobs...", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        results_a = list(pool.map(worker, configs_a))
    print(f"  Completed in {time.time()-t0:.0f}s", flush=True)

    # Aggregate Phase A
    groups_a = defaultdict(list)
    for r in results_a:
        groups_a[r['label']].append(r)

    print(f"\n  {'Config':<22s} {'MSE':>10s} {'Top-1':>7s} {'Top-3':>7s} {'KL':>7s} {'Accept%':>9s} {'Time':>6s}")
    print(f"  {'-'*70}")
    for label in ['english_mse', 'english_combined', 'english_hybrid']:
        runs = groups_a[label]
        mses = [r['final_mse'] for r in runs]
        t1s = [r['final_top1'] for r in runs]
        t3s = [r['final_top3'] for r in runs]
        kls = [r['final_kl'] for r in runs]
        ars = [r['accept_rate'] for r in runs]
        tms = [r['time'] for r in runs]
        print(f"  {label:<22s} {np.mean(mses):.6f} {np.mean(t1s)*100:5.1f}% "
              f"{np.mean(t3s)*100:5.1f}% {np.mean(kls):5.2f} {np.mean(ars):7.3f}% {np.mean(tms):5.0f}s")

    # Show per-char distributions from best MSE run
    mse_runs = groups_a.get('english_mse', [])
    if mse_runs:
        best_mse = min(mse_runs, key=lambda r: r['final_mse'])
        print(f"\n  Per-char distributions (best MSE run, seed={best_mse['seed']}):")
        for ch_idx in viz_chars:
            if ch_idx in best_mse['per_char']:
                print_char_dist(ch_idx, best_mse['per_char'][ch_idx], BIGRAM_DIST, VOCAB)

    # Show trajectory for best MSE run
    if mse_runs and best_mse['trajectory']:
        print(f"\n  MSE trajectory (seed={best_mse['seed']}):")
        print(f"  {'Step':>6s} {'MSE':>10s} {'Top-1':>7s} {'Accept%':>9s}")
        for t in best_mse['trajectory']:
            print(f"  {t['step']:6d} {t['mse']:.6f} {t['top1']*100:5.1f}% {t['accept_rate']:7.3f}%")

    # =========================================================
    # PHASE B: Synthetic ambiguity sweep
    # =========================================================
    print(f"\n{'#'*65}")
    print(f"  PHASE B: Synthetic ambiguity sweep (16c, 80n, 4K att)")
    print(f"{'#'*65}", flush=True)

    ambiguities = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
    configs_b = []

    for amb in ambiguities:
        td_synth, perm = make_ambiguous_task(16, amb, seed=42)
        active_synth = list(range(16))
        # Generate pairs matching the distribution (for combined scoring)
        pairs_synth = []
        for i in range(16):
            for j in range(16):
                count = int(td_synth[i, j] * 100)
                pairs_synth.extend([(i, j)] * count)
        if not pairs_synth:
            pairs_synth = [(i, perm[i]) for i in range(16)]

        for mode in ['mse', 'combined']:
            for seed in SEEDS:
                configs_b.append(dict(
                    label=f'amb{amb:.1f}_{mode}',
                    n_neurons=80, seed=seed, max_attempts=4000, ticks=8,
                    score_mode=mode, true_dist=td_synth, pairs=pairs_synth,
                    active_inputs=active_synth, vocab=16, log_interval=0))

    print(f"  Running {len(configs_b)} jobs...", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        results_b = list(pool.map(worker, configs_b))
    print(f"  Completed in {time.time()-t0:.0f}s", flush=True)

    # Aggregate Phase B
    groups_b = defaultdict(list)
    for r in results_b:
        groups_b[r['label']].append(r)

    print(f"\n  {'Ambiguity':>9s} {'MSE-scorer':>12s} {'Combined':>12s} {'MSE-wins?':>10s}")
    print(f"  {'-'*50}")
    for amb in ambiguities:
        mse_label = f'amb{amb:.1f}_mse'
        comb_label = f'amb{amb:.1f}_combined'
        mse_acc = np.mean([r['final_top1'] for r in groups_b.get(mse_label, [])])
        comb_acc = np.mean([r['final_top1'] for r in groups_b.get(comb_label, [])])
        mse_mse = np.mean([r['final_mse'] for r in groups_b.get(mse_label, [])])
        comb_mse = np.mean([r['final_mse'] for r in groups_b.get(comb_label, [])])
        wins = "YES" if mse_mse < comb_mse else "no"
        print(f"  {amb:9.1f} {mse_acc*100:10.1f}% {comb_acc*100:10.1f}% {wins:>10s}")

    # MSE comparison
    print(f"\n  MSE values (lower = better distribution match):")
    print(f"  {'Ambiguity':>9s} {'MSE-scorer':>12s} {'Combined':>12s}")
    print(f"  {'-'*40}")
    for amb in ambiguities:
        mse_label = f'amb{amb:.1f}_mse'
        comb_label = f'amb{amb:.1f}_combined'
        mse_v = np.mean([r['final_mse'] for r in groups_b.get(mse_label, [])])
        comb_v = np.mean([r['final_mse'] for r in groups_b.get(comb_label, [])])
        print(f"  {amb:9.1f} {mse_v:12.6f} {comb_v:12.6f}")

    # =========================================================
    # PHASE D: Final validation with winner
    # =========================================================
    # Determine winner from Phase A
    best_mode = min(groups_a, key=lambda k: np.mean([r['final_mse'] for r in groups_a[k]]))
    best_mode_name = best_mode.split('_')[1]

    print(f"\n{'#'*65}")
    print(f"  PHASE D: Final validation with winner ({best_mode_name})")
    print(f"  160n + 256n, 16K att, 5 seeds")
    print(f"{'#'*65}", flush=True)

    configs_d = []
    for nn in [160, 256]:
        for seed in SEEDS:
            configs_d.append(dict(
                label=f'{nn}n_{best_mode_name}_16K',
                n_neurons=nn, seed=seed, max_attempts=16000, ticks=8,
                score_mode=best_mode_name, log_interval=4000,
                viz_chars=viz_chars))

    print(f"  Running {len(configs_d)} jobs...", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        results_d = list(pool.map(worker, configs_d))
    print(f"  Completed in {time.time()-t0:.0f}s", flush=True)

    groups_d = defaultdict(list)
    for r in results_d:
        groups_d[r['label']].append(r)

    print(f"\n  {'Config':<25s} {'MSE':>10s} {'Top-1':>7s} {'Top-3':>7s} {'KL':>7s} {'Accept%':>9s}")
    print(f"  {'-'*70}")
    for label in sorted(groups_d):
        runs = groups_d[label]
        print(f"  {label:<25s} {np.mean([r['final_mse'] for r in runs]):.6f} "
              f"{np.mean([r['final_top1'] for r in runs])*100:5.1f}% "
              f"{np.mean([r['final_top3'] for r in runs])*100:5.1f}% "
              f"{np.mean([r['final_kl'] for r in runs]):5.2f} "
              f"{np.mean([r['accept_rate'] for r in runs]):7.3f}%")

    # Best run per-char viz
    all_d = [r for r in results_d if best_mode_name in r['label']]
    if all_d:
        best_d = min(all_d, key=lambda r: r['final_mse'])
        print(f"\n  Per-char distributions (best run, {best_d['label']}, seed={best_d['seed']}):")
        for ch_idx in viz_chars:
            if ch_idx in best_d['per_char']:
                print_char_dist(ch_idx, best_d['per_char'][ch_idx], BIGRAM_DIST, VOCAB)

        if best_d['trajectory']:
            print(f"\n  Trajectory:")
            print(f"  {'Step':>6s} {'MSE':>10s} {'Top-1':>7s} {'Accept%':>9s}")
            for t in best_d['trajectory']:
                print(f"  {t['step']:6d} {t['mse']:.6f} {t['top1']*100:5.1f}% {t['accept_rate']:7.3f}%")

    print(f"\n  {'='*65}")
    print(f"  DONE")
    print(f"  {'='*65}", flush=True)
