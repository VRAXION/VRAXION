"""
VRAXION v22 — Shared Utilities
===============================
Consolidated from 45 test files. Import this instead of copy-pasting.

Usage from tests/:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from v22_utils import score_combined, score_batch, flip_single, VOCAB
"""

import numpy as np
import random
from v22_best_config import SelfWiringGraph, softmax


# ============================================================
#  Scoring functions
# ============================================================

def score_combined(net, targets, vocab, ticks=8):
    """Combined scoring: 0.5*accuracy + 0.5*target_prob.
    Sequential 2-pass evaluation (warmup + scoring).
    Used by 17+ test files."""
    net.reset()
    correct = 0
    total_score = 0.0
    n = vocab
    for p in range(2):
        for inp in range(n):
            world = np.zeros(vocab, dtype=np.float32)
            world[inp] = 1.0
            logits = net.forward(world, ticks)
            probs = softmax(logits[:vocab])
            if p == 1:
                tgt = targets[inp]
                acc_i = 1.0 if np.argmax(probs) == tgt else 0.0
                tp = float(probs[tgt])
                total_score += 0.5 * acc_i + 0.5 * tp
                if acc_i > 0:
                    correct += 1
    acc = correct / n
    net.last_acc = acc
    return total_score / n, acc


def score_combined_perclass(net, targets, vocab, ticks=8):
    """Combined scoring + per-class accuracy array."""
    net.reset()
    correct = 0
    total_score = 0.0
    per_class = np.zeros(vocab, dtype=np.float32)
    n = vocab
    for p in range(2):
        for inp in range(n):
            world = np.zeros(vocab, dtype=np.float32)
            world[inp] = 1.0
            logits = net.forward(world, ticks)
            probs = softmax(logits[:vocab])
            if p == 1:
                tgt = targets[inp]
                acc_i = 1.0 if np.argmax(probs) == tgt else 0.0
                tp = float(probs[tgt])
                total_score += 0.5 * acc_i + 0.5 * tp
                if acc_i > 0:
                    correct += 1
                per_class[inp] = acc_i
    acc = correct / n
    net.last_acc = acc
    return total_score / n, acc, per_class


def score_batch(net, targets, V, ticks=8):
    """Batch scoring: combined 0.5*acc + 0.5*target_prob.
    Uses forward_batch() — 17x faster than sequential at V=64."""
    logits_all = net.forward_batch(ticks)  # (V, V)
    e = np.exp(logits_all - logits_all.max(axis=1, keepdims=True))
    probs_all = e / e.sum(axis=1, keepdims=True)
    preds = np.argmax(probs_all, axis=1)
    acc = (preds == targets).mean()
    target_probs = probs_all[np.arange(V), targets]
    score = 0.5 * (preds == targets).astype(float).mean() + 0.5 * target_probs.mean()
    return score, acc


# ============================================================
#  Mutation helpers
# ============================================================

def flip_single(net):
    """Flip sign of a single random existing connection."""
    alive = np.argwhere(net.mask != 0)
    if len(alive) == 0:
        return
    idx = alive[random.randint(0, len(alive) - 1)]
    net.mask[int(idx[0]), int(idx[1])] *= -1


def capped_mutate(net, cap=0.15, rate=0.03):
    """Single mutation op, density-aware (no add if over cap)."""
    N = net.N
    density = (net.mask != 0).sum() / (N * (N - 1))

    if density >= cap:
        r = random.random()
        if r < net.flip_rate:
            flip_single(net)
        else:
            action = random.choice(['remove', 'rewire'])
            alive = np.argwhere(net.mask != 0)
            if action == 'remove' and len(alive) > 3:
                idx = alive[random.randint(0, len(alive) - 1)]
                net.mask[int(idx[0]), int(idx[1])] = 0
            elif action == 'rewire' and len(alive) > 0:
                idx = alive[random.randint(0, len(alive) - 1)]
                r2, c = int(idx[0]), int(idx[1])
                old_sign = net.mask[r2, c]
                old_w = net.W[r2, c]
                net.mask[r2, c] = 0
                nc = random.randint(0, N - 1)
                while nc == r2:
                    nc = random.randint(0, N - 1)
                net.mask[r2, nc] = old_sign
                net.W[r2, nc] = old_w
    else:
        net.mutate_structure(rate)


def temp_mutate(net, temperature, cap=0.20):
    """Temperature-modulated mutation with density cap.
    Zones: <0.5 FOCUSED, 0.5-1.5 NORMAL, 1.5-3.0 WIDE, >3.0 EARTHQUAKE."""
    if temperature < 0.5:
        flip_single(net)
    elif temperature < 1.5:
        if random.random() < 0.3:
            flip_single(net)
        else:
            capped_mutate(net, cap, 0.05)
    elif temperature < 3.0:
        n_changes = int(2 + temperature)
        for _ in range(n_changes):
            if random.random() < 0.5:
                flip_single(net)
            else:
                capped_mutate(net, cap, 0.03)
    else:
        n_changes = int(temperature * 2)
        for _ in range(n_changes):
            capped_mutate(net, cap, 0.03)
        N = net.N
        region = random.sample(range(N), min(10, N))
        for n in region:
            alive_row = np.argwhere(net.mask[n] != 0).flatten()
            if len(alive_row) > 0:
                idx = alive_row[random.randint(0, len(alive_row) - 1)]
                net.mask[n, idx] *= -1


# ============================================================
#  English text corpus & bigram data
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
    """Compute bigram probability distribution from TEXT corpus."""
    counts = np.zeros((VOCAB, VOCAB), dtype=np.float32)
    for i in range(len(TEXT) - 1):
        a, b = TEXT[i], TEXT[i + 1]
        if a in char_to_idx and b in char_to_idx:
            counts[char_to_idx[a], char_to_idx[b]] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return counts / row_sums


BIGRAM_DIST = compute_bigram_dist()
ACTIVE_INPUTS = [i for i in range(VOCAB) if BIGRAM_DIST[i].sum() > 0.01]


# ============================================================
#  Bigram scoring
# ============================================================

def score_hybrid_bigram(net, true_dist, vocab, active_inputs, ticks=8):
    """Hybrid scoring for bigram: 0.5*top1 + 0.5*(1 - MSE/0.01)."""
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
    mse_score = 1.0 - mean_mse / 0.01
    return 0.5 * top1_acc + 0.5 * mse_score


def eval_bigram_metrics(net, true_dist, vocab, active_inputs, ticks=8):
    """Full bigram metrics: MSE, top-1, top-3."""
    total_mse = 0.0
    top1 = 0
    top3 = 0
    for inp in active_inputs:
        net.reset()
        world = np.zeros(vocab, dtype=np.float32)
        world[inp] = 1.0
        logits = net.forward(world, ticks)
        pred = softmax(logits[:vocab])
        target = true_dist[inp]
        total_mse += np.mean((pred - target) ** 2)
        if np.argmax(pred) == np.argmax(target):
            top1 += 1
        if np.argmax(target) in set(np.argsort(-pred)[:3]):
            top3 += 1
    n = len(active_inputs)
    return {'mse': total_mse / n, 'top1': top1 / n, 'top3': top3 / n}


# ============================================================
#  Generic training loop
# ============================================================

def train_loop(net, targets, V, score_fn, mutate_fn=None,
               max_att=8000, ticks=8, stale_limit=6000, phase_switch=2500):
    """Generic mutation + selection training loop.

    Args:
        net: SelfWiringGraph instance
        targets: target permutation array
        V: vocabulary size
        score_fn: callable(net, targets, V, ticks) -> (score, acc)
        mutate_fn: callable(net, phase, rate) or None for default
        max_att: max attempts
        ticks: forward pass ticks
        stale_limit: stop after this many stale attempts
        phase_switch: switch to BOTH phase after this many stale

    Returns:
        dict with acc, kept, attempts, phase
    """
    import time

    def default_mutate(net, phase, _rate):
        if phase == "STRUCTURE":
            net.mutate_structure(0.05)
        else:
            if random.random() < 0.3:
                net.mutate_structure(0.02)
            else:
                net.mutate_weights()

    if mutate_fn is None:
        mutate_fn = default_mutate

    score, acc = score_fn(net, targets, V, ticks)
    best_acc = acc
    kept = 0
    stale = 0
    phase = "STRUCTURE"
    t0 = time.time()

    for att in range(max_att):
        saved_mask = net.mask.copy()
        saved_W = net.W.copy()

        mutate_fn(net, phase, 0.05 if phase == "STRUCTURE" else 0.02)

        new_score, new_acc = score_fn(net, targets, V, ticks)
        if new_score > score:
            score = new_score
            kept += 1
            stale = 0
            best_acc = max(best_acc, new_acc)
        else:
            net.mask = saved_mask
            net.W = saved_W
            stale += 1

        if phase == "STRUCTURE" and stale > phase_switch:
            phase = "BOTH"
            stale = 0
        if best_acc >= 0.99:
            break
        if stale >= stale_limit:
            break

    elapsed = time.time() - t0
    return {
        'acc': best_acc, 'kept': kept, 'attempts': att + 1,
        'phase': phase, 'time': elapsed,
        'conns': net.count_connections(),
        'density': (net.mask != 0).sum() / (net.N * (net.N - 1)),
    }
