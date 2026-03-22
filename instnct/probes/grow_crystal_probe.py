"""
Grow-Crystal Probe
===================
Alternating phases:
  GROWTH:  add + rewire (70/30) until stale > growth_patience
  CRYSTAL: remove-only until remove_stale > crystal_patience (no remove accepted)
  repeat

No flip. Empty start. No drive.
Logs every phase transition + checkpoints.
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
    random.seed(seed)
    np.random.seed(seed)
    net = SelfWiringGraph(V)
    net.mask[:] = 0.0
    net.alive = []
    net.alive_set = set()
    net.state *= 0
    net.charge *= 0
    net.loss_pct = np.int8(15)
    net.mutation_drive = np.int8(0)
    return net


def run_trial(V, seed, budget, growth_cycle=1000, crystal_patience=500,
              ticks=6, log_f=None, live_f=None):
    net = make_empty_net(V, seed)
    rng_t = np.random.RandomState(seed + 1000)
    targets = np.arange(V)
    rng_t.shuffle(targets)

    score = evaluate(net, targets, ticks)
    best = score
    stale = 0
    phase = 'GROWTH'
    phase_num = 0
    crystal_stale = 0
    growth_step = 0
    total_accepts = 0
    growth_accepts = 0
    crystal_accepts = 0
    crystal_removed_total = 0
    t0 = time.time()

    history = []
    phase_log = []

    def log_phase_change(att, new_phase, extra=""):
        nonlocal phase_num
        entry = {
            "att": att, "phase": new_phase, "phase_num": phase_num,
            "score": round(score, 5), "best": round(best, 5),
            "edges": net.count_connections(),
            "growth_accepts": growth_accepts,
            "crystal_removed_total": crystal_removed_total,
        }
        phase_log.append(entry)
        label = f"  [{att:6d}] -> {new_phase:8s} | score={score:.4f} best={best:.4f} edges={net.count_connections()} {extra}"
        print(label)
        if live_f:
            live_f.write(label + "\n")
            live_f.flush()
        phase_num += 1

    log_phase_change(0, 'GROWTH')

    for att in range(1, budget + 1):
        if phase == 'GROWTH':
            op = 'add' if random.random() < 0.7 else 'rewire'
            if not net.alive and op == 'rewire':
                op = 'add'

            undo = net.mutate(forced_op=op)
            new_score = evaluate(net, targets, ticks)

            if new_score > score:
                score = new_score
                best = max(best, score)
                stale = 0
                total_accepts += 1
                growth_accepts += 1
            else:
                net.replay(undo)
                stale += 1

            growth_step += 1
            if growth_step >= growth_cycle:
                # Every growth_cycle steps -> crystallize
                phase = 'CRYSTAL'
                crystal_stale = 0
                crystal_removed_this = 0
                growth_step = 0
                log_phase_change(att, 'CRYSTAL')

        elif phase == 'CRYSTAL':
            if not net.alive:
                phase = 'GROWTH'
                growth_step = 0
                growth_accepts = 0
                log_phase_change(att, 'GROWTH', "empty-net")
                continue

            undo = net.mutate(forced_op='remove')
            new_score = evaluate(net, targets, ticks)

            if new_score >= score:  # >= : equal is fine for cleanup
                if new_score > score:
                    score = new_score
                    best = max(best, score)
                score = new_score
                crystal_stale = 0
                total_accepts += 1
                crystal_accepts += 1
                crystal_removed_total += 1
                crystal_removed_this = crystal_removed_this + 1 if 'crystal_removed_this' in dir() else 1
            else:
                net.replay(undo)
                crystal_stale += 1

            if crystal_stale >= crystal_patience:
                phase = 'GROWTH'
                growth_step = 0
                growth_accepts = 0
                log_phase_change(att, 'GROWTH', f"pruned={crystal_removed_total}")

        # Checkpoint logging
        if att % 2000 == 0:
            elapsed = time.time() - t0
            h = {
                "att": att, "score": round(score, 5), "best": round(best, 5),
                "edges": net.count_connections(), "phase": phase,
                "total_accepts": total_accepts,
                "crystal_total_removed": sum(p.get('crystal_removed', 0) for p in phase_log),
                "rate": round(att / elapsed, 0),
                "elapsed": round(elapsed, 1),
            }
            history.append(h)
            label = (f"  [{att:6d}] score={score:.4f} best={best:.4f} "
                     f"edges={net.count_connections()} phase={phase} "
                     f"accepts={total_accepts} {elapsed:.0f}s")
            print(label)
            if live_f:
                live_f.write(label + "\n")
                live_f.flush()

        if best >= 0.999:
            break

    return {
        "V": V, "seed": seed, "budget": budget,
        "growth_cycle": growth_cycle,
        "crystal_patience": crystal_patience,
        "final_best": round(best, 5),
        "final_edges": net.count_connections(),
        "total_accepts": total_accepts,
        "crystal_accepts": crystal_accepts,
        "elapsed": round(time.time() - t0, 1),
        "phases": phase_log,
        "history": history,
    }


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    LOG = os.path.join(base_dir, "grow_crystal_log.jsonl")
    LIVE = os.path.join(base_dir, "grow_crystal_live.txt")

    live_f = open(LIVE, "w", encoding="utf-8")
    live_f.write(f"Grow-Crystal Probe | {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    with open(LOG, "w") as f:
        pass

    configs = [
        # V, seed, budget, growth_cycle, crystal_patience
        (16, 42, 10000, 1000, 300),
        (16, 77, 10000, 1000, 300),
        (32, 42, 20000, 1000, 500),
        (32, 77, 20000, 1000, 500),
        (64, 42, 40000, 1000, 800),
        (64, 77, 40000, 1000, 800),
    ]

    all_results = []

    for V, seed, budget, gc, cp in configs:
        label = f"V={V} seed={seed}"
        live_f.write(f"\n{'='*55}\n  {label} | budget={budget} gc={gc} cp={cp}\n{'='*55}\n")
        live_f.flush()
        print(f"\n{'='*55}")
        print(f"  {label} | budget={budget} gc={gc} cp={cp}")
        print(f"{'='*55}")

        result = run_trial(V, seed, budget, gc, cp, log_f=None, live_f=live_f)
        all_results.append(result)

        with open(LOG, "a") as f:
            f.write(json.dumps(result) + "\n")

        summary = (f"  DONE: best={result['final_best']:.4f} "
                   f"edges={result['final_edges']} "
                   f"accepts={result['total_accepts']} "
                   f"crystal_removed={result['crystal_accepts']} "
                   f"phases={len(result['phases'])} "
                   f"{result['elapsed']:.0f}s")
        print(summary)
        live_f.write(summary + "\n")
        live_f.flush()

    # Summary
    print(f"\n{'='*60}")
    print(f"  FINAL COMPARISON")
    print(f"{'='*60}")
    live_f.write(f"\n{'='*60}\n  FINAL COMPARISON\n{'='*60}\n")

    for V in [16, 32, 64]:
        results_v = [r for r in all_results if r['V'] == V]
        if not results_v:
            continue
        avg = np.mean([r['final_best'] for r in results_v])
        edges = [r['final_edges'] for r in results_v]
        phases = [len(r['phases']) for r in results_v]
        crystal = [r['crystal_accepts'] for r in results_v]
        line = (f"  V={V:3d}: avg={avg:.4f} "
                f"edges={edges} phases={phases} crystal_removed={crystal}")
        print(line)
        live_f.write(line + "\n")

    live_f.close()


if __name__ == "__main__":
    main()

