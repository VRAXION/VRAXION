"""
English Text Prediction — v22 SelfWiringGraph on real language
===============================================================
Can the capacitor network learn English bigram statistics?

Task: given current character, predict next character.
This tests whether the architecture handles REAL statistical
patterns, not just synthetic permutation lookups.

Uses the canonical v22_best_config SelfWiringGraph with combined scoring.
Parallel execution, 5 seeds.
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

# Build vocabulary
chars = sorted(set(TEXT))
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for c, i in char_to_idx.items()}
VOCAB = len(chars)

# Build bigram pairs
def build_bigrams(text):
    pairs = []
    for i in range(len(text) - 1):
        if text[i] in char_to_idx and text[i+1] in char_to_idx:
            pairs.append((char_to_idx[text[i]], char_to_idx[text[i+1]]))
    return pairs

ALL_PAIRS = build_bigrams(TEXT)

# Compute true bigram distribution for reference
def compute_bigram_dist():
    counts = np.zeros((VOCAB, VOCAB), dtype=np.float32)
    for a, b in ALL_PAIRS:
        counts[a, b] += 1
    # Normalize per row
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return counts / row_sums

BIGRAM_DIST = compute_bigram_dist()


# ============================================================
#  Training
# ============================================================

def train_config(label, n_neurons, seed, max_attempts=8000, ticks=8,
                 sample_n=120):
    """Train on English bigram prediction."""
    np.random.seed(seed)
    random.seed(seed)

    V = VOCAB
    net = SelfWiringGraph(n_neurons, V)

    pairs = ALL_PAIRS.copy()

    def evaluate():
        net.reset()
        correct = 0
        total_score = 0.0
        sample = random.sample(pairs, min(sample_n, len(pairs)))

        # 2 passes for state buildup
        for p in range(2):
            for inp_idx, tgt_idx in sample:
                world = np.zeros(V, dtype=np.float32)
                world[inp_idx] = 1.0
                logits = net.forward(world, ticks)
                probs = softmax(logits[:V])

                if p == 1:
                    acc_i = 1.0 if np.argmax(probs) == tgt_idx else 0.0
                    tp = float(probs[tgt_idx])
                    total_score += 0.5 * acc_i + 0.5 * tp
                    if acc_i > 0:
                        correct += 1

        n = len(sample)
        acc = correct / n
        score = total_score / n
        net.last_acc = acc
        return score, acc

    score, acc = evaluate()
    best_score = score
    best_acc = acc
    kept = 0
    stale = 0
    phase = "STRUCTURE"
    switched = False

    t0 = time.time()

    for att in range(max_attempts):
        state = net.save_state()

        if phase == "STRUCTURE":
            net.mutate_structure(0.05)
        else:
            if random.random() < 0.3:
                net.mutate_structure(0.02)
            else:
                net.mutate_weights()

        net.self_wire()
        new_score, new_acc = evaluate()

        if new_score > score:
            score = new_score
            kept += 1
            stale = 0
            best_score = max(best_score, score)
            best_acc = max(best_acc, new_acc)
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

    # Final evaluation on ALL pairs (not sampled)
    net.reset()
    final_correct = 0
    final_prob = 0.0
    for p in range(2):
        for inp_idx, tgt_idx in pairs:
            world = np.zeros(V, dtype=np.float32)
            world[inp_idx] = 1.0
            logits = net.forward(world, ticks)
            probs = softmax(logits[:V])
            if p == 1:
                if np.argmax(probs) == tgt_idx:
                    final_correct += 1
                final_prob += probs[tgt_idx]

    final_acc = final_correct / len(pairs)
    final_avg_prob = final_prob / len(pairs)

    # Sample predictions for display
    predictions = []
    net.reset()
    test_inputs = "the qu"
    for ch in test_inputs:
        if ch in char_to_idx:
            world = np.zeros(V, dtype=np.float32)
            world[char_to_idx[ch]] = 1.0
            logits = net.forward(world, ticks)
            probs = softmax(logits[:V])
            top3_idx = np.argsort(probs)[-3:][::-1]
            top3 = [(idx_to_char[i], float(probs[i])) for i in top3_idx]
            predictions.append((ch, top3))

    return {
        'label': label,
        'seed': seed,
        'train_acc': best_acc,
        'train_score': best_score,
        'final_acc': final_acc,
        'final_prob': final_avg_prob,
        'kept': kept,
        'accept_rate': accept_rate,
        'time': elapsed,
        'conns': net.count_connections(),
        'predictions': predictions,
    }


def worker(args):
    return train_config(**args)


# ============================================================
#  Main
# ============================================================

SEEDS = [42, 123, 777, 1337, 2024]

if __name__ == "__main__":
    n_workers = os.cpu_count() or 24

    print(f"  English Text Prediction — v22 SelfWiringGraph")
    print(f"  {'='*60}")
    print(f"  Vocabulary: {VOCAB} chars: {''.join(chars)}")
    print(f"  Text length: {len(TEXT)} chars")
    print(f"  Bigram pairs: {len(ALL_PAIRS)}")
    print(f"  Random baseline: {100/VOCAB:.1f}%")
    print(f"  CPU cores: {n_workers}")
    print(f"  Seeds: {SEEDS}")

    # Compute oracle (perfect bigram model)
    oracle_correct = 0
    for inp_idx, tgt_idx in ALL_PAIRS:
        if np.argmax(BIGRAM_DIST[inp_idx]) == tgt_idx:
            oracle_correct += 1
    oracle_acc = oracle_correct / len(ALL_PAIRS)
    print(f"  Oracle (perfect bigram): {oracle_acc*100:.1f}%")

    # =========================================================
    # PHASE 1: Different network sizes
    # =========================================================
    print(f"\n{'#'*65}")
    print(f"  PHASE 1: Network size comparison (8K attempts)")
    print(f"{'#'*65}", flush=True)

    configs = []
    for nn in [80, 160, 256]:
        label = f'{nn}n bigram'
        for seed in SEEDS:
            configs.append(dict(
                label=label, n_neurons=nn, seed=seed,
                max_attempts=8000, ticks=8, sample_n=120))

    print(f"\n  Running {len(configs)} jobs...", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        results = list(pool.map(worker, configs))
    print(f"  Completed in {time.time()-t0:.0f}s", flush=True)

    # Aggregate
    groups = defaultdict(list)
    for r in results:
        groups[r['label']].append(r)

    print(f"\n  {'='*65}")
    print(f"  RESULTS")
    print(f"  {'='*65}")
    print(f"  {'Config':<20s} {'Train':>7s} {'Final':>7s} {'Prob':>6s} {'Accept%':>8s} {'Time':>6s}")
    print(f"  {'-'*60}")

    for label in sorted(groups, key=lambda k: int(k.split('n')[0])):
        runs = groups[label]
        train_accs = [r['train_acc'] for r in runs]
        final_accs = [r['final_acc'] for r in runs]
        probs = [r['final_prob'] for r in runs]
        rates = [r['accept_rate'] for r in runs]
        times = [r['time'] for r in runs]
        print(f"  {label:<20s} {np.mean(train_accs)*100:5.1f}% {np.mean(final_accs)*100:5.1f}% "
              f"{np.mean(probs):5.3f} {np.mean(rates):7.3f}% {np.mean(times):5.0f}s")
        ratio = np.mean(final_accs) / (1/VOCAB)
        print(f"  {'':20s} vs random: {ratio:.1f}x")

    # Show predictions from best run
    best_run = max(results, key=lambda r: r['final_acc'])
    print(f"\n  Best run predictions ({best_run['label']}, seed={best_run['seed']}, "
          f"acc={best_run['final_acc']*100:.1f}%):")
    for ch, top3 in best_run['predictions']:
        print(f"    '{ch}' -> {top3[0][0]}({top3[0][1]:.2f}) "
              f"{top3[1][0]}({top3[1][1]:.2f}) "
              f"{top3[2][0]}({top3[2][1]:.2f})")

    # =========================================================
    # PHASE 2: Extended run with best size (16K)
    # =========================================================
    best_size = max(groups, key=lambda k: np.mean([r['final_acc'] for r in groups[k]]))
    nn = int(best_size.split('n')[0])

    print(f"\n{'#'*65}")
    print(f"  PHASE 2: Extended 16K with best size ({nn} neurons)")
    print(f"{'#'*65}", flush=True)

    ext_configs = []
    for seed in SEEDS:
        ext_configs.append(dict(
            label=f'{nn}n 16K', n_neurons=nn, seed=seed,
            max_attempts=16000, ticks=8, sample_n=120))

    print(f"\n  Running {len(ext_configs)} jobs...", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        ext_results = list(pool.map(worker, ext_configs))
    print(f"  Completed in {time.time()-t0:.0f}s", flush=True)

    ext_accs = [r['final_acc'] for r in ext_results]
    ext_probs = [r['final_prob'] for r in ext_results]
    ext_rates = [r['accept_rate'] for r in ext_results]

    print(f"\n  {'='*65}")
    print(f"  EXTENDED RESULTS ({nn}n, 16K)")
    print(f"  {'='*65}")
    print(f"  Final acc:  {np.mean(ext_accs)*100:.1f}% (std {np.std(ext_accs)*100:.1f}%)")
    print(f"  Avg prob:   {np.mean(ext_probs):.3f}")
    print(f"  Accept:     {np.mean(ext_rates):.3f}%")
    print(f"  vs random:  {np.mean(ext_accs)/(1/VOCAB):.1f}x")
    print(f"  vs oracle:  {np.mean(ext_accs)/oracle_acc*100:.1f}% of perfect bigram")

    best_ext = max(ext_results, key=lambda r: r['final_acc'])
    print(f"\n  Best run predictions (seed={best_ext['seed']}, "
          f"acc={best_ext['final_acc']*100:.1f}%):")
    for ch, top3 in best_ext['predictions']:
        print(f"    '{ch}' -> {top3[0][0]}({top3[0][1]:.2f}) "
              f"{top3[1][0]}({top3[1][1]:.2f}) "
              f"{top3[2][0]}({top3[2][1]:.2f})")

    print(f"\n  {'='*65}")
    print(f"  DONE")
    print(f"  {'='*65}", flush=True)
