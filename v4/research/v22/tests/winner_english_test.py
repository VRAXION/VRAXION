"""
VRAXION v22 -- Winner Config on English Text (Overnight Run)
=============================================================
Best ensemble from mega-test (ens_tick_mixed, 56.2% on synthetic)
applied to real English bigram prediction.

Winner config:
  - 4 slaves: 4-tick, 8-tick, 16-tick, 8-tick-aggr
  - Full budget per slave (no splitting)
  - 4 rounds with jackpot broadcast
  - Global mutations (no region restriction)
  - Full eval (no shortcuts)
  - Split I/O

Baselines for comparison:
  - Single searcher (8-tick, refiner_aggr, same total budget)
  - Single searcher (8-tick, default train loop)

Corpus: English proverbs/phrases (~1800 chars, ~30 unique chars)
Task: bigram prediction (given char, predict next char)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
import time
from v22_best_config import SelfWiringGraph, softmax

# =============================================================
# English corpus
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

# Build bigram distribution
def compute_bigram_dist():
    counts = np.zeros((VOCAB, VOCAB), dtype=np.float32)
    for i in range(len(TEXT) - 1):
        a, b = char_to_idx[TEXT[i]], char_to_idx[TEXT[i+1]]
        counts[a, b] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return counts / row_sums

BIGRAM_DIST = compute_bigram_dist()
ACTIVE_INPUTS = [i for i in range(VOCAB) if BIGRAM_DIST[i].sum() > 0.01]

# Most common bigram target per input (ground truth for top-1)
TARGETS = np.argmax(BIGRAM_DIST, axis=1)

# =============================================================
# Scoring -- bigram distribution matching
# =============================================================
def score_bigram(net, ticks=8):
    """Score network on bigram prediction.
    Uses forward_batch for speed.
    Returns (combined_score, top1_acc, avg_target_prob)."""
    logits = net.forward_batch(ticks=ticks)  # (V, V)
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)

    # Top-1 accuracy: does argmax match most common next char?
    preds = np.argmax(probs, axis=1)
    correct = 0
    total = 0
    for i in ACTIVE_INPUTS:
        if preds[i] == TARGETS[i]:
            correct += 1
        total += 1
    top1 = correct / max(total, 1)

    # Distribution matching: MSE between predicted and true bigram dist
    mse = 0.0
    for i in ACTIVE_INPUTS:
        diff = probs[i] - BIGRAM_DIST[i]
        mse += (diff * diff).sum()
    mse /= max(len(ACTIVE_INPUTS), 1)

    # Combined: accuracy + distribution quality
    dist_score = max(0, 1.0 - mse / 0.01)
    combined = 0.5 * top1 + 0.5 * dist_score

    return combined, top1, dist_score

def eval_predictions(net, ticks=8):
    """Show sample predictions for display."""
    logits = net.forward_batch(ticks=ticks)
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)

    test_chars = "the qu.asinof"
    lines = []
    for ch in test_chars:
        if ch in char_to_idx:
            i = char_to_idx[ch]
            top5_idx = np.argsort(probs[i])[-5:][::-1]
            top5 = [(idx_to_char[j], f"{probs[i,j]*100:.0f}%") for j in top5_idx]
            true_next = idx_to_char[TARGETS[i]] if TARGETS[i] < VOCAB else '?'
            lines.append(f"    '{ch}' -> {top5}  (true: '{true_next}')")
    return lines

# =============================================================
# Training strategies
# =============================================================
def train_single(net, budget, ticks=8, mut_style='refiner_aggr', label='single'):
    """Single searcher baseline."""
    best_sc, best_top1, _ = score_bigram(net, ticks)
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
        elif mut_style == 'refiner_aggr':
            if random.random() < 0.4:
                net.mutate_weights()
            else:
                net.mutate_structure(0.07)
        elif mut_style == 'refiner':
            if random.random() < 0.7:
                net.mutate_weights()
            else:
                net.mutate_structure(0.02)
        else:
            net.mutate_structure(0.05)

        sc, top1, _ = score_bigram(net, ticks)

        if sc > best_sc:
            best_sc = sc
            best_top1 = top1
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
            _, cur_top1, cur_dist = score_bigram(net, ticks)
            print(f"      [{label}] att={att+1:5d} top1={cur_top1*100:5.1f}% "
                  f"dist={cur_dist*100:5.1f}% kept={kept}")
            sys.stdout.flush()

    net.mask = best_mask.copy()
    net.W = best_W.copy()
    return best_sc, best_top1, kept

def train_ensemble_tick_mixed(net, budget_per_slave, rounds=4, label='ensemble'):
    """Winner ensemble: 4 slaves with different tick counts."""
    slave_configs = {
        'slave_4tick':      {'ticks': 4, 'mut_style': 'refiner_aggr'},
        'slave_8tick':      {'ticks': 8, 'mut_style': 'refiner_aggr'},
        'slave_16tick':     {'ticks': 16, 'mut_style': 'refiner_aggr'},
        'slave_8tick_aggr': {'ticks': 8, 'mut_style': 'refiner'},
    }

    champion_mask = net.mask.copy()
    champion_W = net.W.copy()
    champion_sc, champion_top1, _ = score_bigram(net, ticks=8)
    wins = {}

    for rnd in range(rounds):
        round_best_sc = champion_sc
        round_best_mask = champion_mask.copy()
        round_best_W = champion_W.copy()
        round_winner = None

        for sname, scfg in slave_configs.items():
            # Each slave starts from champion
            net.mask = champion_mask.copy()
            net.W = champion_W.copy()

            ticks = scfg['ticks']
            mut_style = scfg['mut_style']
            best_sc = champion_sc
            best_mask = net.mask.copy()
            best_W = net.W.copy()

            for att in range(budget_per_slave):
                sm = net.mask.copy()
                sw = net.W.copy()

                if mut_style == 'refiner_aggr':
                    if random.random() < 0.4:
                        net.mutate_weights()
                    else:
                        net.mutate_structure(0.07)
                else:
                    if random.random() < 0.7:
                        net.mutate_weights()
                    else:
                        net.mutate_structure(0.02)

                sc, top1, _ = score_bigram(net, ticks=ticks)
                if sc > best_sc:
                    best_sc = sc
                    best_mask = net.mask.copy()
                    best_W = net.W.copy()
                else:
                    net.mask = sm
                    net.W = sw

            if best_sc > round_best_sc:
                round_best_sc = best_sc
                round_best_mask = best_mask.copy()
                round_best_W = best_W.copy()
                round_winner = sname

        if round_winner:
            champion_sc = round_best_sc
            champion_mask = round_best_mask.copy()
            champion_W = round_best_W.copy()
            wins[round_winner] = wins.get(round_winner, 0) + 1

        # Eval at standard 8 ticks
        net.mask = champion_mask.copy()
        net.W = champion_W.copy()
        _, cur_top1, cur_dist = score_bigram(net, ticks=8)
        winner_str = round_winner or 'no_improvement'
        print(f"      [{label}] round {rnd+1}/{rounds}: top1={cur_top1*100:5.1f}% "
              f"dist={cur_dist*100:5.1f}% winner={winner_str}")
        sys.stdout.flush()

    net.mask = champion_mask.copy()
    net.W = champion_W.copy()
    _, final_top1, _ = score_bigram(net, ticks=8)
    return champion_sc, final_top1, wins

# =============================================================
# Main
# =============================================================
def main():
    print("=" * 70)
    print("  WINNER CONFIG ON ENGLISH TEXT")
    print("=" * 70)
    print(f"  Corpus: {len(TEXT)} chars, {VOCAB} unique chars")
    print(f"  Active inputs: {len(ACTIVE_INPUTS)}/{VOCAB}")
    print(f"  Task: bigram prediction (char -> next char)")
    print(f"  Random baseline: {1.0/VOCAB*100:.1f}%")
    print()

    # Show bigram distribution stats
    entropies = []
    for i in ACTIVE_INPUTS:
        p = BIGRAM_DIST[i]
        p = p[p > 0]
        entropies.append(-np.sum(p * np.log2(p)))
    print(f"  Avg bigram entropy: {np.mean(entropies):.2f} bits")
    print(f"  (lower = more predictable, higher = harder)")
    print(f"  {'=' * 70}")
    sys.stdout.flush()

    SEEDS = [42, 77, 123]
    N_NEURONS = 256
    BUDGET_SINGLE = 32000      # single searcher: 32K attempts
    BUDGET_ENSEMBLE = 2000     # per slave per round: 2K * 4 slaves * 4 rounds = 32K total
    ROUNDS = 4

    all_results = []

    for seed in SEEDS:
        print(f"\n  --- SEED {seed} ---")
        sys.stdout.flush()

        # A) Single searcher baseline (refiner_aggr, 32K attempts)
        print(f"\n  [A] Single searcher (refiner_aggr, {BUDGET_SINGLE} att):")
        np.random.seed(seed); random.seed(seed)
        net_a = SelfWiringGraph(N_NEURONS, VOCAB)
        t0 = time.time()
        sc_a, top1_a, kept_a = train_single(net_a, BUDGET_SINGLE, ticks=8,
                                             mut_style='refiner_aggr', label='A')
        time_a = time.time() - t0
        _, _, dist_a = score_bigram(net_a, ticks=8)
        preds_a = eval_predictions(net_a, ticks=8)
        print(f"    => top1={top1_a*100:.1f}% dist={dist_a*100:.1f}% "
              f"kept={kept_a} time={time_a:.0f}s")
        for line in preds_a:
            print(line)
        sys.stdout.flush()

        # B) Ensemble tick-mixed (4 slaves, 4 rounds, 2K/slave = 32K total)
        print(f"\n  [B] Ensemble tick-mixed (4 slaves, {ROUNDS} rounds, {BUDGET_ENSEMBLE}/slave):")
        np.random.seed(seed); random.seed(seed)
        net_b = SelfWiringGraph(N_NEURONS, VOCAB)
        t0 = time.time()
        sc_b, top1_b, wins_b = train_ensemble_tick_mixed(
            net_b, BUDGET_ENSEMBLE, rounds=ROUNDS, label='B')
        time_b = time.time() - t0
        _, _, dist_b = score_bigram(net_b, ticks=8)
        preds_b = eval_predictions(net_b, ticks=8)
        print(f"    => top1={top1_b*100:.1f}% dist={dist_b*100:.1f}% "
              f"wins={wins_b} time={time_b:.0f}s")
        for line in preds_b:
            print(line)
        sys.stdout.flush()

        # C) Single searcher with MORE budget (64K -- 2x fair comparison)
        print(f"\n  [C] Single searcher extended ({BUDGET_SINGLE*2} att):")
        np.random.seed(seed); random.seed(seed)
        net_c = SelfWiringGraph(N_NEURONS, VOCAB)
        t0 = time.time()
        sc_c, top1_c, kept_c = train_single(net_c, BUDGET_SINGLE * 2, ticks=8,
                                             mut_style='refiner_aggr', label='C')
        time_c = time.time() - t0
        _, _, dist_c = score_bigram(net_c, ticks=8)
        print(f"    => top1={top1_c*100:.1f}% dist={dist_c*100:.1f}% "
              f"kept={kept_c} time={time_c:.0f}s")
        sys.stdout.flush()

        all_results.append({
            'seed': seed,
            'A_single_top1': top1_a, 'A_time': time_a,
            'B_ensemble_top1': top1_b, 'B_time': time_b, 'B_wins': wins_b,
            'C_extended_top1': top1_c, 'C_time': time_c,
        })

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY")
    print(f"{'=' * 70}\n")

    print(f"  {'Seed':>6s}  {'A single':>10s}  {'B ensemble':>10s}  {'C extended':>10s}  {'B vs A':>8s}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}")
    for r in all_results:
        diff = (r['B_ensemble_top1'] - r['A_single_top1']) * 100
        sign = '+' if diff >= 0 else ''
        print(f"  {r['seed']:6d}  {r['A_single_top1']*100:9.1f}%  "
              f"{r['B_ensemble_top1']*100:9.1f}%  {r['C_extended_top1']*100:9.1f}%  "
              f"{sign}{diff:6.1f}%")

    a_mean = np.mean([r['A_single_top1'] for r in all_results])
    b_mean = np.mean([r['B_ensemble_top1'] for r in all_results])
    c_mean = np.mean([r['C_extended_top1'] for r in all_results])
    diff_mean = (b_mean - a_mean) * 100

    print(f"\n  {'Mean':>6s}  {a_mean*100:9.1f}%  {b_mean*100:9.1f}%  {c_mean*100:9.1f}%  "
          f"{'+' if diff_mean >= 0 else ''}{diff_mean:6.1f}%")

    print(f"\n  Random baseline: {1.0/VOCAB*100:.1f}%")

    if diff_mean > 0:
        print(f"\n  VERDICT: Ensemble WINS by {diff_mean:.1f}% on English text!")
    elif diff_mean > -1:
        print(f"\n  VERDICT: Roughly equal on English text (diff={diff_mean:.1f}%)")
    else:
        print(f"\n  VERDICT: Single searcher wins on English text (diff={diff_mean:.1f}%)")

    print(f"\n{'=' * 70}", flush=True)


if __name__ == '__main__':
    main()
