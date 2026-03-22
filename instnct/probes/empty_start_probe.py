"""
Empty Start Probe
==================
Start from ZERO edges. Grow sparse. No drive, no burst.

Tests:
  A) add-only
  B) add + rewire (50/50)
  C) add + rewire + flip (33/33/33)
  D) add + rewire + remove (40/40/20)

Also: deterministic op effectiveness at checkpoints.
  - Try ALL possible flips, count wins
  - Try ALL possible rewires (sample), count wins
  - Try ALL possible adds (sample), count wins
  - Try ALL possible removes, count wins

V=16, V=32, seed=42
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


def make_empty_net(V, seed):
    """Create SWG with ZERO edges."""
    random.seed(seed)
    np.random.seed(seed)
    net = SelfWiringGraph(V)
    # Wipe mask completely
    net.mask[:] = 0.0
    net.alive = []
    net.alive_set = set()
    net.state *= 0
    net.charge *= 0
    net.loss_pct = np.int8(15)
    net.mutation_drive = np.int8(0)
    return net


def op_effectiveness(net, targets, ticks=6):
    """Deterministic: try EVERY possible single-op move, count wins."""
    base = evaluate(net, targets, ticks)
    mask0 = net.mask.copy()
    H = net.H
    drive = float(net.mutation_drive)
    alive = list(net.alive)

    results = {}

    # FLIP: try every alive edge
    wins, total = 0, 0
    for r, c in alive:
        net.mask[r, c] = -mask0[r, c]
        if evaluate(net, targets, ticks) > base + 1e-12:
            wins += 1
        total += 1
        net.mask[:] = mask0
    results['flip'] = {'wins': wins, 'total': total,
                       'rate': round(wins/max(total,1), 4)}

    # REMOVE: try every alive edge
    wins, total = 0, 0
    for r, c in alive:
        net.mask[r, c] = 0.0
        if evaluate(net, targets, ticks) > base + 1e-12:
            wins += 1
        total += 1
        net.mask[:] = mask0
    results['remove'] = {'wins': wins, 'total': total,
                         'rate': round(wins/max(total,1), 4)}

    # ADD: sample up to 2000 empty cells
    empty_cells = []
    for r in range(H):
        for c in range(H):
            if r != c and mask0[r, c] == 0:
                empty_cells.append((r, c))
    sample = random.sample(empty_cells, min(2000, len(empty_cells)))
    wins, total = 0, 0
    for r, c in sample:
        sign = drive if random.random() < 0.5 else -drive
        net.mask[r, c] = sign
        if evaluate(net, targets, ticks) > base + 1e-12:
            wins += 1
        total += 1
        net.mask[:] = mask0
    results['add'] = {'wins': wins, 'total': total,
                      'rate': round(wins/max(total,1), 4),
                      'sampled': len(sample) < len(empty_cells)}

    # REWIRE: sample up to 2000
    rewire_moves = []
    for r, c in alive:
        for nc in range(H):
            if nc != r and nc != c and mask0[r, nc] == 0:
                rewire_moves.append((r, c, nc))
    sample_rw = random.sample(rewire_moves, min(2000, len(rewire_moves)))
    wins, total = 0, 0
    for r, c, nc in sample_rw:
        val = mask0[r, c]
        net.mask[r, c] = 0.0
        net.mask[r, nc] = val
        if evaluate(net, targets, ticks) > base + 1e-12:
            wins += 1
        total += 1
        net.mask[:] = mask0
    results['rewire'] = {'wins': wins, 'total': total,
                         'rate': round(wins/max(total,1), 4),
                         'sampled': len(sample_rw) < len(rewire_moves)}

    return base, results


POLICIES = {
    'add_only':       ['add'],
    'add_rewire':     ['add', 'rewire'],
    'add_rewire_flip': ['add', 'rewire', 'flip'],
    'add_rewire_remove': ['add', 'add', 'rewire', 'rewire', 'remove'],
}


def run_trial(V, seed, budget, policy_name, ticks=6, checkpoints=None):
    if checkpoints is None:
        checkpoints = set()

    net = make_empty_net(V, seed)
    # Need deterministic targets too
    rng_t = np.random.RandomState(seed + 1000)
    targets = np.arange(V)
    rng_t.shuffle(targets)

    ops = POLICIES[policy_name]
    score = evaluate(net, targets, ticks)
    best = score
    stale = 0
    accepts = 0
    per_op_accepts = {op: 0 for op in set(ops)}
    per_op_attempts = {op: 0 for op in set(ops)}
    t0 = time.time()

    history = []
    checkpoint_data = []

    for att in range(1, budget + 1):
        op = random.choice(ops)

        # For add_only: if no edges yet, can only add
        if op in ('rewire', 'flip', 'remove') and not net.alive:
            op = 'add'

        undo = net.mutate(forced_op=op)
        per_op_attempts[op] = per_op_attempts.get(op, 0) + 1
        new_score = evaluate(net, targets, ticks)

        if new_score > score:
            score = new_score
            best = max(best, score)
            stale = 0
            accepts += 1
            per_op_accepts[op] = per_op_accepts.get(op, 0) + 1
        else:
            net.replay(undo)
            stale += 1

        if att % 1000 == 0:
            elapsed = time.time() - t0
            h = {
                "att": att, "score": round(score, 5), "best": round(best, 5),
                "stale": stale, "accepts": accepts,
                "edges": net.count_connections(),
                "rate": round(att/elapsed, 0),
                "elapsed": round(elapsed, 1),
            }
            history.append(h)
            # Op rates
            for o in set(ops):
                a = per_op_attempts.get(o, 0)
                w = per_op_accepts.get(o, 0)
                h[f'{o}_rate'] = round(w/max(a,1), 4)

        if att in checkpoints:
            base, eff = op_effectiveness(net, targets, ticks)
            checkpoint_data.append({
                "att": att, "score": round(base, 5),
                "edges": net.count_connections(),
                "effectiveness": eff
            })

    return {
        "V": V, "seed": seed, "budget": budget,
        "policy": policy_name,
        "final_best": round(best, 5),
        "final_edges": net.count_connections(),
        "accepts": accepts,
        "per_op_accepts": {k: v for k, v in per_op_accepts.items()},
        "per_op_attempts": {k: v for k, v in per_op_attempts.items()},
        "elapsed": round(time.time() - t0, 1),
        "history": history,
        "checkpoints": checkpoint_data,
    }


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    LOG = os.path.join(base_dir, "empty_start_log.jsonl")
    LIVE = os.path.join(base_dir, "empty_start_live.txt")

    with open(LIVE, "w") as f:
        f.write(f"Empty Start Probe | {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    with open(LOG, "w") as f:
        pass

    configs = [
        (16, 42, 8000),
        (16, 77, 8000),
        (32, 42, 12000),
        (32, 77, 12000),
    ]

    checkpoints = {2000, 4000, 6000, 8000}

    all_results = []

    for V, seed, budget in configs:
        for policy in POLICIES:
            label = f"V={V} s={seed} {policy}"
            print(f"  {label}...", end=" ", flush=True)

            result = run_trial(V, seed, budget, policy, checkpoints=checkpoints)
            all_results.append(result)

            with open(LOG, "a") as f:
                f.write(json.dumps(result) + "\n")

            # Per-op accept rates
            op_info = ""
            for op in sorted(set(POLICIES[policy])):
                a = result['per_op_attempts'].get(op, 0)
                w = result['per_op_accepts'].get(op, 0)
                if a > 0:
                    op_info += f" {op}={w}/{a}({w/a*100:.1f}%)"

            line = (f"  {label:35s} | best={result['final_best']:.4f} "
                    f"| edges={result['final_edges']:4d} "
                    f"| accepts={result['accepts']:4d} "
                    f"|{op_info}")
            print(f"best={result['final_best']:.4f} edges={result['final_edges']} "
                  f"{result['elapsed']:.0f}s")

            with open(LIVE, "a") as f:
                f.write(line + "\n")

            # Print checkpoint effectiveness if any
            for cp in result.get('checkpoints', []):
                eff = cp['effectiveness']
                eff_line = (f"    @{cp['att']}: score={cp['score']:.4f} "
                           f"edges={cp['edges']} | "
                           f"flip={eff['flip']['rate']:.1%} "
                           f"remove={eff['remove']['rate']:.1%} "
                           f"add={eff['add']['rate']:.1%} "
                           f"rewire={eff['rewire']['rate']:.1%}")
                print(eff_line)
                with open(LIVE, "a") as f:
                    f.write(eff_line + "\n")

    # Summary table
    print(f"\n{'='*70}")
    print(f"  FINAL SCORES")
    print(f"{'='*70}")
    summary_lines = []
    for V in [16, 32]:
        for policy in POLICIES:
            scores = [r['final_best'] for r in all_results
                      if r['V'] == V and r['policy'] == policy]
            avg = np.mean(scores)
            line = f"  V={V:2d} {policy:20s} avg={avg:.4f}  scores={[round(s,4) for s in scores]}"
            print(line)
            summary_lines.append(line)

    with open(LIVE, "a") as f:
        f.write(f"\n{'='*70}\n  FINAL SCORES\n{'='*70}\n")
        for line in summary_lines:
            f.write(line + "\n")


if __name__ == "__main__":
    main()

