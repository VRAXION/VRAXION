"""
Adaptive Mutation Step Size Probe
==================================
Tests: does escalating mutation count on plateau break through?

Strategy:
  stale < 1000  → 1 change
  stale < 3000  → 2-3 changes
  stale < 5000  → 5-8 changes
  stale < 8000  → 10-15 changes
  stale > 8000  → reset to best, restart

When ANY improvement found → back to 1 change.

Compares: fixed (current) vs adaptive, same budget.
Logs to JSONL for live monitoring.
"""

import sys, os, time, json, random
import numpy as np

from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))
from graph import SelfWiringGraph


def evaluate(net, targets, ticks=6):
    logits = net.forward_batch(ticks)
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    V = min(net.V, len(targets))
    acc = (np.argmax(probs, axis=1)[:V] == targets[:V]).mean()
    tp = probs[np.arange(V), targets[:V]].mean()
    return float(0.5 * acc + 0.5 * tp)


def adaptive_n_changes(stale):
    """Escalating mutation count based on staleness."""
    if stale < 1000:
        return 1
    elif stale < 3000:
        return random.choice([2, 3])
    elif stale < 5000:
        return random.choice([5, 6, 7, 8])
    elif stale < 8000:
        return random.choice([10, 12, 15])
    else:
        return -1  # signal: reset


def multi_mutate(net, n_changes):
    """Apply n_changes mutations, return combined undo log."""
    undo_all = []
    old_loss = int(net.loss_pct)
    old_mutation_drive = int(net.mutation_drive)

    for i in range(n_changes):
        op = random.choice(['add', 'remove', 'rewire', 'flip'])
        undo = net.mutate(forced_op=op)
        undo_all.extend(undo)

    return undo_all, old_loss, old_mutation_drive


def run_trial(V, seed, budget, mode, ticks=6):
    """Run one trial. mode='fixed' or 'adaptive'."""
    random.seed(seed)
    np.random.seed(seed)

    net = SelfWiringGraph(V)
    targets = np.arange(V)
    np.random.shuffle(targets)

    score = evaluate(net, targets, ticks)
    best = score
    best_state = net.save_state()
    stale = 0
    accepts = 0
    resets = 0
    t0 = time.time()

    history = []

    for att in range(1, budget + 1):
        if mode == 'adaptive':
            n_ch = adaptive_n_changes(stale)
            if n_ch == -1:
                # Reset to best, restart
                net.restore_state(best_state)
                score = best
                stale = 0
                resets += 1
                continue
            undo, old_loss, old_mutation_drive = multi_mutate(net, n_ch)
        else:
            # Fixed: use default mutate (drive-based)
            old_loss = int(net.loss_pct)
            old_mutation_drive = int(net.mutation_drive)
            undo = net.mutate()
            n_ch = max(1, abs(int(net.mutation_drive)))

        new_score = evaluate(net, targets, ticks)

        if new_score > score:
            score = new_score
            if score > best:
                best = score
                best_state = net.save_state()
            stale = 0
            accepts += 1
        else:
            net.replay(undo)
            net.loss_pct = np.int8(old_loss)
            net.mutation_drive = np.int8(old_mutation_drive)
            stale += 1

            # Fixed mode: also do stale rewire (matching current train())
            if mode == 'fixed' and stale > 2000:
                rw_undo = []
                net._rewire(rw_undo)
                rw_score = evaluate(net, targets, ticks)
                if rw_score > score:
                    score = rw_score
                    best = max(best, score)
                    best_state = net.save_state()
                    stale = 0
                    accepts += 1
                else:
                    net.replay(rw_undo)

        if att % 500 == 0:
            elapsed = time.time() - t0
            entry = {
                "att": att, "score": round(score, 5), "best": round(best, 5),
                "stale": stale, "accepts": accepts, "resets": resets,
                "edges": net.count_connections(),
                "n_ch": n_ch, "mode": mode,
                "rate": round(att / elapsed, 1),
                "elapsed": round(elapsed, 1)
            }
            history.append(entry)

    return {
        "V": V, "seed": seed, "budget": budget, "mode": mode,
        "final_best": round(best, 5),
        "accepts": accepts, "resets": resets,
        "elapsed": round(time.time() - t0, 1),
        "history": history
    }


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    LOG = os.path.join(base, "adaptive_mutation_log.jsonl")
    LIVE = os.path.join(base, "adaptive_mutation_live.txt")

    with open(LIVE, "w") as f:
        f.write(f"Adaptive Mutation Probe | {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    with open(LOG, "w") as f:
        pass

    configs = [
        (16, 42, 8000),
        (16, 77, 8000),
        (32, 42, 8000),
        (32, 77, 8000),
    ]

    all_results = []

    for V, seed, budget in configs:
        for mode in ['fixed', 'adaptive']:
            label = f"V={V} seed={seed} {mode}"
            print(f"\n{'='*50}")
            print(f"  {label} | budget={budget}")
            print(f"{'='*50}")

            result = run_trial(V, seed, budget, mode)
            all_results.append(result)

            with open(LOG, "a") as f:
                f.write(json.dumps(result) + "\n")

            # Live summary
            line = (f"  {label:30s} | best={result['final_best']:.4f} "
                    f"| accepts={result['accepts']} | resets={result['resets']} "
                    f"| {result['elapsed']:.0f}s")
            print(line)
            with open(LIVE, "a") as f:
                f.write(line + "\n")

            # Print trajectory
            for h in result['history']:
                tline = (f"    [{h['att']:5d}] score={h['best']:.4f} "
                         f"stale={h['stale']} edges={h['edges']} n_ch={h['n_ch']}")
                print(tline)

    # Final comparison
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")

    with open(LIVE, "a") as f:
        f.write(f"\n{'='*60}\n  SUMMARY\n{'='*60}\n")

    for V in [16, 32]:
        fixed = [r for r in all_results if r['V'] == V and r['mode'] == 'fixed']
        adaptive = [r for r in all_results if r['V'] == V and r['mode'] == 'adaptive']
        f_avg = np.mean([r['final_best'] for r in fixed])
        a_avg = np.mean([r['final_best'] for r in adaptive])
        diff = a_avg - f_avg
        winner = "ADAPTIVE" if diff > 0.001 else "FIXED" if diff < -0.001 else "TIE"
        line = f"  V={V}: fixed={f_avg:.4f} adaptive={a_avg:.4f} diff={diff:+.4f} → {winner}"
        print(line)
        with open(LIVE, "a") as f:
            f.write(line + "\n")


if __name__ == "__main__":
    main()
