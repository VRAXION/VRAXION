"""
Convergence Test: Does SWG plateau or just learn slower?
=========================================================
Run SWG with increasing budgets and log the learning curve.
If score keeps climbing → just slow. If it flattens → architectural plateau.
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random as pyrandom
from model.graph import SelfWiringGraph

def set_seeds(seed):
    np.random.seed(seed)
    pyrandom.seed(seed)


def train_with_curve(V, targets, seed, max_attempts, checkpoints):
    """Train SWG and record score at each checkpoint."""
    set_seeds(seed)
    N = V * 3
    net = SelfWiringGraph(N, V)

    def evaluate():
        logits = net.forward_batch(ticks=8)
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
        acc = (np.argmax(probs, axis=1)[:V] == targets[:V]).mean()
        tp = probs[np.arange(V), targets[:V]].mean()
        return acc, 0.5 * acc + 0.5 * tp

    acc, score = evaluate()
    best_score = score
    best_acc = acc
    stale = 0
    curve = []
    next_cp = 0
    t0 = time.time()

    for att in range(max_attempts):
        old_loss = int(net.loss_pct)
        undo = net.mutate()
        new_acc, new_score = evaluate()

        if new_score > score:
            score = new_score
            best_score = max(best_score, score)
            best_acc = max(best_acc, new_acc)
            stale = 0
        else:
            net.replay(undo)
            net.loss_pct = np.int8(old_loss)
            stale += 1
            if pyrandom.randint(1, 20) <= 7:
                net.signal = np.int8(1 - int(net.signal))
            if pyrandom.randint(1, 20) <= 7:
                net.grow = np.int8(1 - int(net.grow))

        # Record at checkpoints
        if next_cp < len(checkpoints) and (att + 1) >= checkpoints[next_cp]:
            elapsed = time.time() - t0
            curve.append({
                'attempt': att + 1,
                'best_acc': best_acc,
                'best_combined': best_score,
                'stale': stale,
                'conns': net.count_connections(),
                'time': elapsed,
                'mode': 'SIGNAL' if net.signal else ('GROW' if net.grow else 'SHRINK'),
                'intensity': int(net.intensity),
                'loss_pct': int(net.loss_pct),
            })
            next_cp += 1

        if best_acc >= 1.0:
            # Record final point
            elapsed = time.time() - t0
            curve.append({
                'attempt': att + 1,
                'best_acc': best_acc,
                'best_combined': best_score,
                'stale': 0,
                'conns': net.count_connections(),
                'time': elapsed,
                'mode': 'CONVERGED',
                'intensity': int(net.intensity),
                'loss_pct': int(net.loss_pct),
            })
            break

    return curve, best_acc, best_score


def print_curve(V, curve):
    print(f"\n  {'Attempt':>8}  {'Acc':>7}  {'Combined':>9}  {'Conns':>6}  "
          f"{'Stale':>6}  {'Mode':>8}  {'Int':>3}  {'Loss%':>5}  {'Time':>6}")
    print(f"  {'─'*72}")
    for p in curve:
        print(f"  {p['attempt']:>8}  {p['best_acc']*100:>6.1f}%  "
              f"{p['best_combined']*100:>8.1f}%  {p['conns']:>6}  "
              f"{p['stale']:>6}  {p['mode']:>8}  {p['intensity']:>3}  "
              f"{p['loss_pct']:>5}  {p['time']:>5.1f}s")


if __name__ == '__main__':
    print("=" * 75)
    print("  CONVERGENCE TEST: Does SWG plateau or just learn slower?")
    print("=" * 75)

    # ── Test 1: V=16, high budget — can it reach 100% reliably? ──
    for V in [16, 32, 64]:
        if V == 16:
            max_att = 50000
            checkpoints = [500, 1000, 2000, 4000, 8000, 16000, 30000, 50000]
        elif V == 32:
            max_att = 100000
            checkpoints = [1000, 2000, 4000, 8000, 16000, 32000, 64000, 100000]
        else:
            max_att = 200000
            checkpoints = [2000, 4000, 8000, 16000, 32000, 64000, 128000, 200000]

        n_seeds = 10
        print(f"\n{'='*75}")
        print(f"  V={V}  |  budget={max_att}  |  {n_seeds} seeds  |  NO stale limit")
        print(f"{'='*75}")

        all_curves = []
        converged_count = 0

        for si, seed in enumerate(range(200, 200 + n_seeds)):
            set_seeds(seed)
            targets = np.random.permutation(V)

            curve, best_acc, best_comb = train_with_curve(
                V, targets, seed, max_att, checkpoints)
            all_curves.append(curve)

            status = "CONVERGED" if best_acc >= 1.0 else f"PLATEAU {best_acc*100:.1f}%"
            if best_acc >= 1.0:
                converged_count += 1
                final_att = curve[-1]['attempt']
                print(f"  Seed {seed}: {status} @ attempt {final_att}")
            else:
                print(f"  Seed {seed}: {status} (stale={curve[-1]['stale']})")

        # ── Average learning curve ──
        print(f"\n  AVERAGE LEARNING CURVE (V={V}):")
        max_len = max(len(c) for c in all_curves)
        for ci in range(len(checkpoints)):
            accs = []
            for curve in all_curves:
                # Find the checkpoint entry (or last entry if converged early)
                if ci < len(curve):
                    accs.append(curve[ci]['best_acc'])
                elif curve:
                    accs.append(curve[-1]['best_acc'])
            if accs:
                accs = np.array(accs)
                budget_at = checkpoints[ci] if ci < len(checkpoints) else max_att
                print(f"    @ {budget_at:>7} attempts:  "
                      f"{accs.mean()*100:5.1f}% ± {accs.std()*100:4.1f}%  "
                      f"(min={accs.min()*100:.1f}%, max={accs.max()*100:.1f}%)")

        print(f"\n  Converged to 100%: {converged_count}/{n_seeds} "
              f"({converged_count/n_seeds*100:.0f}%)")

        # Show one detailed curve
        print(f"\n  DETAILED CURVE (seed 200):")
        if all_curves:
            print_curve(V, all_curves[0])

    print(f"\n{'='*75}")
    print("  VERDICT")
    print(f"{'='*75}")
    print("  If accuracy keeps climbing with more budget → SLOW LEARNER (good!)")
    print("  If accuracy flattens despite more budget  → ARCHITECTURAL PLATEAU")
    print(f"{'='*75}")
