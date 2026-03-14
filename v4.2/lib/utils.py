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
        state = net.save_state()

        if mutate_fn:
            mutate_fn(net)
        else:
            net.mutate_with_mood()

        sc, acc = score_fn(net, targets, V, ticks)

        if sc > best_sc:
            best_sc = sc
            best_acc = acc
            kept += 1
            stale = 0
        else:
            net.restore_state(state)
            stale += 1

        if phase == 'STRUCTURE' and stale > phase_switch and not switched:
            phase = 'BOTH'
            switched = True
            stale = 0

        if stale >= stale_limit:
            break

    return best_sc, best_acc, kept
