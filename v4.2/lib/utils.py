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


def train_cyclic(net, targets, V, score_fn, ticks=8,
                 max_att=20000, stale_limit=6000,
                 add_every=50, crystal_plateau=500,
                 verbose=True):
    """Cyclic phase training: REWIRE → CRYSTALLIZE → repeat.

    Phase 1 - REWIRE: mostly rewire ops, with one add every `add_every` attempts.
              Runs until stale_limit is hit.
    Phase 2 - CRYSTALLIZE: remove-only. Prunes weak edges.
              Plateau detection: if no edge removed (accepted) in `crystal_plateau`
              consecutive attempts → phase ends, back to REWIRE.

    Plateau detection in CRYSTALLIZE is edge-count based, NOT accuracy based,
    because pruning hurts accuracy short-term but enables better rewiring next cycle.

    Args:
        net: SelfWiringGraph instance
        targets: target array
        V: vocab size
        score_fn: callable(net, targets, V, ticks) -> (score, acc)
        ticks: forward pass ticks
        max_att: total max attempts across all cycles
        stale_limit: stale limit for REWIRE phase
        add_every: inject one add every N attempts during REWIRE
        crystal_plateau: attempts without accepted remove → end CRYSTALLIZE
        verbose: print phase transitions and progress

    Returns:
        (best_score, best_acc, kept_count, cycle_count)
    """
    sc, acc = score_fn(net, targets, V, ticks)
    best_sc = sc
    best_acc = acc
    kept = 0
    total_att = 0
    cycle = 0

    while total_att < max_att:
        cycle += 1

        # ── Phase 1: REWIRE (+ sparse add) ──
        phase_att = 0
        stale = 0
        rewire_kept = 0
        if verbose:
            conns = net.count_connections()
            print(f"\n{'='*50}")
            print(f"  Cycle {cycle} | REWIRE phase | Conns: {conns} | "
                  f"Score: {best_sc*100:.1f}%")
            print(f"{'='*50}")

        while stale < stale_limit and total_att < max_att:
            state = net.save_state()
            old_loss = int(net.loss_pct)

            # Every add_every attempts, force one add; otherwise rewire
            if phase_att > 0 and phase_att % add_every == 0:
                undo = net.mutate(forced_op='add')
            else:
                undo = net.mutate(forced_op='rewire')

            sc, acc = score_fn(net, targets, V, ticks)

            if sc > best_sc:
                best_sc = sc
                best_acc = acc
                kept += 1
                rewire_kept += 1
                stale = 0
            else:
                net.replay(undo)
                net.loss_pct = np.int8(old_loss)
                stale += 1

            phase_att += 1
            total_att += 1

            if verbose and phase_att % 1000 == 0:
                print(f"  [{total_att:6d}] REWIRE  Score: {best_sc*100:5.1f}% | "
                      f"Conns: {net.count_connections():4d} | "
                      f"kept={rewire_kept} stale={stale}")

            if best_sc >= 0.99:
                break

        if best_sc >= 0.99:
            break

        # ── Phase 2: CRYSTALLIZE (remove-only) ──
        phase_att = 0
        no_shrink = 0  # attempts since last accepted remove
        edges_before = net.count_connections()
        crystal_kept = 0
        if verbose:
            print(f"\n  Cycle {cycle} | CRYSTALLIZE phase | "
                  f"Conns: {edges_before} | Score: {best_sc*100:.1f}%")

        while no_shrink < crystal_plateau and total_att < max_att:
            state = net.save_state()
            old_loss = int(net.loss_pct)
            edges_pre = net.count_connections()

            undo = net.mutate(forced_op='remove')

            sc, acc = score_fn(net, targets, V, ticks)

            if sc > best_sc:
                best_sc = sc
                best_acc = acc
                kept += 1
                crystal_kept += 1
                # Check if edges actually decreased (remove accepted)
                if net.count_connections() < edges_pre:
                    no_shrink = 0
                else:
                    no_shrink += 1
            else:
                net.replay(undo)
                net.loss_pct = np.int8(old_loss)
                no_shrink += 1

            phase_att += 1
            total_att += 1

            if verbose and phase_att % 500 == 0:
                edges_now = net.count_connections()
                pruned = edges_before - edges_now
                print(f"  [{total_att:6d}] CRYSTAL Score: {best_sc*100:5.1f}% | "
                      f"Conns: {edges_now:4d} (pruned {pruned}) | "
                      f"no_shrink={no_shrink}")

        if verbose:
            edges_after = net.count_connections()
            print(f"  CRYSTALLIZE done: {edges_before} → {edges_after} edges "
                  f"(pruned {edges_before - edges_after})")

        if best_sc >= 0.99:
            break

    if verbose:
        print(f"\n  FINAL: {cycle} cycles | Score: {best_sc*100:.1f}% | "
              f"Acc: {best_acc*100:.1f}% | Conns: {net.count_connections()} | "
              f"Kept: {kept} / {total_att}")

    return best_sc, best_acc, kept, cycle
