"""
Scoring, mutation helpers, and generic training loop for v22 experiments.
"""

import numpy as np
import random


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def score_combined(net, targets, vocab, ticks=8):
    """0.5*accuracy + 0.5*mean_target_prob. Uses 2-pass sequential eval."""
    net.reset()
    correct = 0
    total_tp = 0.0
    for p in range(2):
        for i in range(vocab):
            world = np.zeros(vocab, dtype=np.float32)
            world[i] = 1.0
            logits = net.forward(world, ticks)
            probs = softmax(logits)
            if p == 1:
                if np.argmax(probs) == targets[i]:
                    correct += 1
                total_tp += probs[targets[i]]
    acc = correct / vocab
    tp = total_tp / vocab
    return 0.5 * acc + 0.5 * tp, acc


def score_batch(net, targets, V, ticks=8):
    """Batch scoring using forward_batch. Returns (combined_score, accuracy)."""
    logits_all = net.forward_batch(ticks)
    e = np.exp(logits_all - logits_all.max(axis=1, keepdims=True))
    probs_all = e / e.sum(axis=1, keepdims=True)
    preds = np.argmax(probs_all, axis=1)
    acc = (preds == targets[:V]).mean()
    target_probs = probs_all[np.arange(V), targets[:V]]
    score = 0.5 * acc + 0.5 * target_probs.mean()
    return score, acc


def flip_single(net):
    """Flip exactly one random existing connection."""
    alive = np.argwhere(net.mask != 0)
    if len(alive) > 0:
        idx = alive[np.random.randint(len(alive))]
        net.mask[idx[0], idx[1]] *= -1


def capped_mutate(net, cap=0.15, rate=0.03):
    """Density-aware mutation: only add if below cap, always allow remove/flip."""
    density = net.count_connections() / (net.N * (net.N - 1))
    if density < cap:
        net.mutate_structure(rate)
    else:
        r = random.random()
        if r < 0.5:
            flip_single(net)
        elif r < 0.8:
            alive = np.argwhere(net.mask != 0)
            if len(alive) > 3:
                idx = alive[np.random.randint(len(alive))]
                net.mask[idx[0], idx[1]] = 0
        else:
            net.mutate_weights()


def temp_mutate(net, temperature, cap=0.20):
    """Temperature-modulated mutation (4 zones based on temperature)."""
    density = net.count_connections() / (net.N * (net.N - 1))
    if temperature < 0.5:
        if density < cap:
            net.mutate_structure(0.02)
        else:
            flip_single(net)
    elif temperature < 1.5:
        net.mutate_structure(0.05)
    elif temperature < 3.0:
        net.mutate_structure(0.10)
        if random.random() < 0.3:
            net.mutate_weights()
    else:
        net.mutate_structure(0.15)
        net.mutate_weights()


def train_loop(net, targets, V, score_fn, mutate_fn=None,
               max_att=8000, ticks=8, stale_limit=6000, phase_switch=2500):
    """Generic training loop with mutation + selection.

    Args:
        net: SelfWiringGraph instance
        targets: target array
        V: vocab size
        score_fn: callable(net, targets, V, ticks) -> (score, acc)
        mutate_fn: optional custom mutation. If None, uses default structure/weight.
        max_att: max attempts
        ticks: forward pass ticks
        stale_limit: stop if no improvement for this many attempts
        phase_switch: switch from STRUCTURE to BOTH after this many stale attempts

    Returns:
        (best_score, best_acc, kept_count)
    """
    sc, acc = score_fn(net, targets, V, ticks)
    best_sc = sc
    best_acc = acc
    stale = 0
    kept = 0
    phase = 'STRUCTURE'
    switched = False

    for att in range(max_att):
        saved_mask = net.mask.copy()
        saved_W = net.W.copy()

        if mutate_fn:
            mutate_fn(net)
        else:
            if phase == 'STRUCTURE':
                net.mutate_structure(0.05)
            else:
                if random.random() < 0.3:
                    net.mutate_structure(0.02)
                else:
                    net.mutate_weights()

        sc, acc = score_fn(net, targets, V, ticks)

        if sc > best_sc:
            best_sc = sc
            best_acc = acc
            kept += 1
            stale = 0
        else:
            net.mask = saved_mask
            net.W = saved_W
            stale += 1

        if phase == 'STRUCTURE' and stale > phase_switch and not switched:
            phase = 'BOTH'
            switched = True
            stale = 0

        if stale >= stale_limit:
            break

    return best_sc, best_acc, kept
