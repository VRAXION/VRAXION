"""
v22 Diagnostic Deep Test — 16-class vs 64-class with full diagnostics
=====================================================================
Runs the SAME architecture on both easy (16) and hard (64) tasks,
then compares internal metrics to see WHERE the 64-class wall comes from.

Metrics:
  - Per-class accuracy breakdown
  - Activation overlap (connection sharing between classes)
  - Interference matrix (mutation helps class A, hurts class B?)
  - Mutation acceptance rate over time
  - Self-wiring contribution
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
import time
from v22_best_config import SelfWiringGraph, softmax


def train_with_diagnostics(net, inputs, targets, vocab, max_attempts=8000,
                           ticks=6, diag_every=1000):
    """Train with periodic diagnostic snapshots."""

    def evaluate():
        net.reset()
        correct = 0
        for p in range(2):
            for i in range(len(inputs)):
                world = np.zeros(vocab, dtype=np.float32)
                world[inputs[i]] = 1.0
                logits = net.forward(world, ticks)
                probs = softmax(logits)
                if p == 1 and np.argmax(probs) == targets[i]:
                    correct += 1
        acc = correct / len(inputs)
        net.last_acc = acc
        return acc

    snapshots = []
    score = evaluate()
    best = score
    phase = "STRUCTURE"
    kept = 0
    stale = 0
    switched = False
    kept_window = 0  # kept in current window
    sw_count = 0     # self-wiring connections added

    for att in range(max_attempts):
        state = net.save_state()
        conns_before = net.count_connections()

        # Mutate
        if phase == "STRUCTURE":
            net.mutate_structure(0.05)
        else:
            if random.random() < 0.3:
                net.mutate_structure(0.02)
            else:
                net.mutate_weights()

        # Self-wiring
        net.self_wire()
        conns_after_sw = net.count_connections()
        sw_added = max(0, conns_after_sw - conns_before)

        # Evaluate
        new_score = evaluate()

        if new_score > score:
            score = new_score
            kept += 1
            kept_window += 1
            stale = 0
            best = max(best, score)
            sw_count += sw_added
        else:
            net.restore_state(state)
            stale += 1

        # Phase transition
        if phase == "STRUCTURE" and stale > 2500 and not switched:
            phase = "BOTH"
            switched = True
            stale = 0

        # Diagnostic snapshot
        if (att + 1) % diag_every == 0:
            diag = net.diagnose(inputs, targets, vocab, ticks)
            diag['step'] = att + 1
            diag['best_acc'] = best
            diag['current_acc'] = score
            diag['phase'] = phase
            diag['kept_total'] = kept
            diag['kept_window'] = kept_window
            diag['sw_connections'] = sw_count
            diag['stale'] = stale
            snapshots.append(diag)

            # Print progress
            pos, neg = net.pos_neg_ratio()
            print(f"  [{att+1:5d}] Acc: {best*100:5.1f}% | "
                  f"Conns: {net.count_connections():4d} (+:{pos} -:{neg}) | "
                  f"Kept: {kept_window:3d}/{diag_every} | "
                  f"SW: {sw_count:3d} | Phase: {phase}")
            kept_window = 0

        if best >= 0.99:
            break
        if stale >= 6000:
            break

    return best, snapshots


def print_per_class(diag, label=""):
    """Print per-class accuracy as a compact heatmap."""
    pc = diag['per_class']
    n = diag['n_classes']
    print(f"\n  Per-class accuracy {label}:")
    line = "  "
    for c in range(n):
        if c in pc and pc[c][1] > 0:
            acc = pc[c][0] / pc[c][1]
            if acc >= 0.99:
                line += "# "  # perfect
            elif acc > 0:
                line += ". "  # partial
            else:
                line += "_ "  # zero
        else:
            line += "? "  # no data
        if (c + 1) % 32 == 0:
            print(line)
            line = "  "
    if line.strip():
        print(line)

    # Count
    perfect = sum(1 for c in range(n) if c in pc and pc[c][1] > 0 and pc[c][0] == pc[c][1])
    partial = sum(1 for c in range(n) if c in pc and pc[c][1] > 0 and 0 < pc[c][0] < pc[c][1])
    zero = sum(1 for c in range(n) if c in pc and pc[c][1] > 0 and pc[c][0] == 0)
    print(f"  Legend: #=100% .=partial _=0%  |  Perfect:{perfect} Partial:{partial} Zero:{zero}")


def print_interference_summary(deltas):
    """Summarize interference matrix."""
    n_samples, n_classes = deltas.shape
    # For each class pair, how often does improving one hurt the other?
    print(f"\n  Interference analysis ({n_samples} random mutations):")

    # How many mutations are pure-positive (improve at least 1, hurt 0)?
    improved = (deltas > 0).any(axis=1)
    hurt = (deltas < 0).any(axis=1)
    pure_pos = (improved & ~hurt).sum()
    pure_neg = (~improved & hurt).sum()
    mixed = (improved & hurt).sum()
    neutral = (~improved & ~hurt).sum()

    print(f"  Pure positive (help only):  {pure_pos:3d} ({100*pure_pos/n_samples:.1f}%)")
    print(f"  Pure negative (hurt only):  {pure_neg:3d} ({100*pure_neg/n_samples:.1f}%)")
    print(f"  Mixed (help some, hurt others): {mixed:3d} ({100*mixed/n_samples:.1f}%)")
    print(f"  Neutral (no change):        {neutral:3d} ({100*neutral/n_samples:.1f}%)")

    # Average interference: when a mutation improves class A, how much does it hurt class B?
    if mixed > 0:
        mixed_rows = deltas[improved & hurt]
        avg_gain = mixed_rows[mixed_rows > 0].mean()
        avg_loss = mixed_rows[mixed_rows < 0].mean()
        print(f"  Avg gain when mixed: +{avg_gain*100:.1f}%  |  Avg loss: {avg_loss*100:.1f}%")


def run_task(n_classes, n_neurons, seed=42):
    """Run one full diagnostic experiment."""
    np.random.seed(seed)
    random.seed(seed)

    V = n_classes
    N = n_neurons
    perm = np.random.permutation(V)
    inputs = list(range(V))

    print(f"\n{'='*60}")
    print(f"  {V}-CLASS LOOKUP | {N} neurons | seed={seed}")
    print(f"{'='*60}")

    net = SelfWiringGraph(N, V)
    t0 = time.time()
    best, snapshots = train_with_diagnostics(net, inputs, perm, V)
    elapsed = time.time() - t0

    print(f"\n  Result: {best*100:.1f}% in {elapsed:.1f}s")

    # Per-class breakdown from final snapshot
    if snapshots:
        print_per_class(snapshots[-1], f"(step {snapshots[-1]['step']})")

        # Internal metrics
        s = snapshots[-1]
        print(f"\n  Internal metrics:")
        print(f"    Active neurons: {s['active_neurons_total']}/{N} "
              f"({100*s['active_neurons_total']/N:.1f}%)")
        print(f"    Active per input: {s['active_per_input_mean']:.1f} "
              f"(+/- {s['active_per_input_std']:.1f})")
        print(f"    Connection overlap mean: {s['conn_overlap_mean']:.2f} "
              f"(max: {s['conn_overlap_max']:.0f})")
        print(f"    Mutation acceptance: {s['kept_total']}/{s['step']} "
              f"({100*s['kept_total']/s['step']:.1f}%)")
        print(f"    Self-wiring conns added: {s['sw_connections']}")

    # Interference test
    print(f"\n  Running interference test...")
    t1 = time.time()
    deltas = net.interference_test(inputs, perm, V, n_samples=200)
    print(f"  ({time.time()-t1:.1f}s)")
    print_interference_summary(deltas)

    # Acceptance rate over time
    if len(snapshots) > 1:
        print(f"\n  Acceptance rate over time:")
        for s in snapshots:
            bar = "#" * s['kept_window']
            print(f"    step {s['step']:5d}: {s['kept_window']:3d} kept  {bar}")

    return best, snapshots, net


if __name__ == "__main__":
    N = 256  # neurons for both tasks

    # Task 1: 16-class (should work well)
    best16, snap16, net16 = run_task(16, N, seed=42)

    # Task 2: 64-class (the wall)
    best64, snap64, net64 = run_task(64, N, seed=42)

    # Summary comparison
    print(f"\n{'='*60}")
    print(f"  COMPARISON: 16-class vs 64-class")
    print(f"{'='*60}")
    s16 = snap16[-1] if snap16 else {}
    s64 = snap64[-1] if snap64 else {}
    fmt = "  {:<30s} {:>12s} {:>12s}"
    print(fmt.format("Metric", "16-class", "64-class"))
    print(fmt.format("-"*30, "-"*12, "-"*12))
    print(fmt.format("Best accuracy",
                      f"{best16*100:.1f}%", f"{best64*100:.1f}%"))
    if s16 and s64:
        print(fmt.format("Active neurons",
                          f"{s16.get('active_neurons_total','?')}/{N}",
                          f"{s64.get('active_neurons_total','?')}/{N}"))
        print(fmt.format("Active per input",
                          f"{s16.get('active_per_input_mean',0):.1f}",
                          f"{s64.get('active_per_input_mean',0):.1f}"))
        print(fmt.format("Conn overlap mean",
                          f"{s16.get('conn_overlap_mean',0):.2f}",
                          f"{s64.get('conn_overlap_mean',0):.2f}"))
        print(fmt.format("Mutation acceptance",
                          f"{s16.get('kept_total',0)}/{s16.get('step',1)}",
                          f"{s64.get('kept_total',0)}/{s64.get('step',1)}"))
        print(fmt.format("Self-wiring conns",
                          str(s16.get('sw_connections', '?')),
                          str(s64.get('sw_connections', '?'))))
