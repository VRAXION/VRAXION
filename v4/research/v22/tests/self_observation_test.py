"""
Self-Observation / Dream Phase Benchmark
=========================================
Tests whether the network benefits from "seeing" its own output.

7 variants, all using capacitor neuron dynamics:
  a) Baseline        — standard 6-tick capacitor
  b) Self-Observe 4+2 — 4 process + 2 reflect ticks
  c) Self-Observe 3+3 — 3 process + 3 reflect
  d) Self-Observe 5+1 — 5 process + 1 reflect
  e) Reentrant 70/30  — every tick: 70% world + 30% own output
  f) Reentrant 50/50  — every tick: 50/50 mix
  g) Dream            — normal forward + dream replay every 100 evals
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
import time
from v22_best_config import SelfWiringGraph, softmax


# ============================================================
#  Forward pass variants (all capacitor-based)
# ============================================================

def entropy(probs):
    """Shannon entropy of a probability distribution."""
    p = probs[probs > 1e-10]
    return -np.sum(p * np.log2(p))


def forward_baseline(net, world):
    """Standard 6-tick capacitor forward. Returns (output_acc, None)."""
    return net._forward_capacitor(world, ticks=6), None


def forward_self_observe(net, world, ticks_process, ticks_reflect):
    """Capacitor forward with self-observation phase.
    Phase 1: process input normally.
    Phase 2: inject softmax(intermediate output) as new input.
    Returns (final_output_acc, first_guess_probs).
    """
    charge = net.state.copy()
    Weff = net.W * net.mask
    threshold, leak = 0.5, 0.85
    output_acc = np.zeros(net.V, dtype=np.float32)

    # PHASE 1: Process
    for t in range(ticks_process):
        charge *= leak
        if t == 0:
            charge[:net.V] += world * 2.0
        spikes = (charge > threshold).astype(np.float32)
        charge += (spikes @ Weff) * 0.3
        fired = charge > threshold
        charge[fired] = 0.0
        output_acc += charge[:net.V]

    # Intermediate output — what the network "thinks"
    first_guess = softmax(output_acc.copy())

    # PHASE 2: Reflect — network sees its own output
    for t in range(ticks_reflect):
        charge *= leak
        charge[:net.V] += first_guess * 2.0  # own output as new input
        spikes = (charge > threshold).astype(np.float32)
        charge += (spikes @ Weff) * 0.3
        fired = charge > threshold
        charge[fired] = 0.0
        output_acc += charge[:net.V]

    net.state = charge.copy()
    return output_acc, first_guess


def forward_reentrant(net, world, ticks, world_ratio):
    """Every tick after t=0: mix world + own accumulated output.
    Returns (output_acc, None).
    """
    charge = net.state.copy()
    Weff = net.W * net.mask
    threshold, leak = 0.5, 0.85
    output_acc = np.zeros(net.V, dtype=np.float32)

    for t in range(ticks):
        charge *= leak
        if t == 0:
            charge[:net.V] += world * 2.0
        else:
            own = softmax(output_acc.copy())
            mix = world * world_ratio + own * (1.0 - world_ratio)
            charge[:net.V] += mix * 2.0
        spikes = (charge > threshold).astype(np.float32)
        charge += (spikes @ Weff) * 0.3
        fired = charge > threshold
        charge[fired] = 0.0
        output_acc += charge[:net.V]

    net.state = charge.copy()
    return output_acc, None


def dream_replay(net, recent_outputs, ticks=4):
    """Replay own recent outputs through the network.
    Only modifies state (implicit memory), NOT topology.
    """
    Weff = net.W * net.mask
    threshold, leak = 0.5, 0.85
    for prev_out in recent_outputs:
        charge = net.state.copy()
        for t in range(ticks):
            charge *= leak
            if t == 0:
                charge[:net.V] += prev_out * 2.0
            spikes = (charge > threshold).astype(np.float32)
            charge += (spikes @ Weff) * 0.3
            fired = charge > threshold
            charge[fired] = 0.0
        net.state = charge.copy()


# ============================================================
#  Variant definitions
# ============================================================

VARIANTS = {
    'baseline':       lambda net, w: forward_baseline(net, w),
    'observe_4+2':    lambda net, w: forward_self_observe(net, w, 4, 2),
    'observe_3+3':    lambda net, w: forward_self_observe(net, w, 3, 3),
    'observe_5+1':    lambda net, w: forward_self_observe(net, w, 5, 1),
    'reentrant_70/30': lambda net, w: forward_reentrant(net, w, 6, 0.7),
    'reentrant_50/50': lambda net, w: forward_reentrant(net, w, 6, 0.5),
    'dream':          lambda net, w: forward_baseline(net, w),  # same fwd, dream happens in train loop
}


# ============================================================
#  Training loop
# ============================================================

def train_variant(variant_name, n_classes, n_neurons, seed,
                  max_attempts=4000, verbose=True):
    """Train with a specific forward variant."""
    np.random.seed(seed)
    random.seed(seed)

    V = n_classes
    net = SelfWiringGraph(n_neurons, V)
    perm = np.random.permutation(V)
    inputs = list(range(V))
    targets = perm.tolist()

    fwd_fn = VARIANTS[variant_name]
    is_dream = (variant_name == 'dream')

    # Dream state: store recent outputs for replay
    recent_outputs = []
    dream_interval = 100  # dream every N evals

    eval_count = [0]

    def evaluate():
        net.reset()
        correct = 0
        entropy_sum = 0.0
        reflect_delta_sum = 0.0
        n_samples = 0

        for p in range(2):
            for i in range(len(inputs)):
                world = np.zeros(V, dtype=np.float32)
                world[inputs[i]] = 1.0
                output, first_guess = fwd_fn(net, world)
                probs = softmax(output[:V])

                if p == 1:
                    if np.argmax(probs) == targets[i]:
                        correct += 1
                    entropy_sum += entropy(probs)
                    n_samples += 1

                    # Measure reflect delta (how much did self-observation change?)
                    if first_guess is not None:
                        reflect_delta_sum += np.abs(probs - first_guess).sum()

                    # Store for dream replay
                    if is_dream:
                        recent_outputs.append(probs.copy())
                        if len(recent_outputs) > V * 2:
                            recent_outputs.pop(0)

        acc = correct / len(inputs)
        avg_entropy = entropy_sum / max(n_samples, 1)
        avg_reflect_delta = reflect_delta_sum / max(n_samples, 1)
        net.last_acc = acc
        eval_count[0] += 1

        # Dream replay
        if is_dream and eval_count[0] % dream_interval == 0 and len(recent_outputs) > 0:
            state_norm_before = np.linalg.norm(net.state)
            dream_replay(net, recent_outputs[-V:])
            state_norm_after = np.linalg.norm(net.state)
            if verbose and eval_count[0] % 500 == 0:
                print(f"    [dream] state norm: {state_norm_before:.2f} -> {state_norm_after:.2f}")

        return acc, avg_entropy, avg_reflect_delta

    score, _, _ = evaluate()
    best = score
    kept = 0
    stale = 0
    phase = "STRUCTURE"
    switched = False
    last_entropy = 0.0
    last_reflect_delta = 0.0

    t0 = time.time()

    for att in range(max_attempts):
        state = net.save_state()

        if phase == "STRUCTURE":
            net.mutate_structure(0.05)
        else:
            if random.random() < 0.3:
                net.mutate_structure(0.02)
            else:
                net.mutate_weights()

        net.self_wire()
        new_score, avg_ent, avg_rd = evaluate()

        if new_score > score:
            score = new_score
            kept += 1
            stale = 0
            best = max(best, score)
            last_entropy = avg_ent
            last_reflect_delta = avg_rd
        else:
            net.restore_state(state)
            stale += 1

        if phase == "STRUCTURE" and stale > 2500 and not switched:
            phase = "BOTH"
            switched = True
            stale = 0

        if verbose and (att + 1) % 1000 == 0:
            elapsed = time.time() - t0
            extras = f"Entropy: {last_entropy:.2f}"
            if last_reflect_delta > 0:
                extras += f"  ReflectD: {last_reflect_delta:.3f}"
            print(f"    [{att+1:5d}] Acc: {best*100:5.1f}%  "
                  f"Kept: {kept:3d}  Conns: {net.count_connections():4d}  "
                  f"{extras}  ({elapsed:.0f}s)", flush=True)

        if best >= 0.99 or stale >= 3500:
            break

    elapsed = time.time() - t0
    return best, kept, elapsed, last_entropy, last_reflect_delta


# ============================================================
#  Main benchmark
# ============================================================

def run_benchmark(n_classes, n_neurons, seed, max_attempts=4000):
    """Run all 7 variants on a given task."""
    print(f"\n{'='*70}")
    print(f"  SELF-OBSERVATION BENCHMARK — {n_classes}-class, {n_neurons} neurons, seed={seed}")
    print(f"{'='*70}", flush=True)

    results = {}
    for name in VARIANTS:
        print(f"\n  --- {name} ---", flush=True)
        acc, kept, elapsed, ent, rd = train_variant(
            name, n_classes, n_neurons, seed, max_attempts)
        results[name] = (acc, kept, elapsed, ent, rd)
        print(f"  RESULT: {name:<18s} Acc={acc*100:5.1f}%  "
              f"Kept={kept:3d}  Time={elapsed:.0f}s  "
              f"Entropy={ent:.2f}  ReflectD={rd:.3f}", flush=True)

    # Summary table
    print(f"\n  {'='*70}")
    print(f"  SUMMARY — {n_classes}-class")
    print(f"  {'='*70}")
    print(f"  {'Variant':<18s} {'Acc':>7s} {'Kept':>5s} {'Time':>6s} "
          f"{'Entropy':>8s} {'ReflD':>7s}", flush=True)
    print(f"  {'-'*55}")

    baseline_acc = results['baseline'][0]
    for name, (acc, kept, elapsed, ent, rd) in results.items():
        delta = (acc - baseline_acc) * 100
        marker = f" ({delta:+.1f}%)" if name != 'baseline' else " (base)"
        print(f"  {name:<18s} {acc*100:6.1f}% {kept:5d} {elapsed:5.0f}s "
              f"{ent:8.2f} {rd:7.3f}{marker}", flush=True)

    return results


if __name__ == "__main__":
    N = 256
    SEED = 42

    # Phase 1: 7-way on 16-class
    r16 = run_benchmark(16, N, SEED, max_attempts=4000)

    # Find winner
    winner_name = max(r16, key=lambda k: r16[k][0])
    baseline_acc = r16['baseline'][0]
    winner_acc = r16[winner_name][0]

    print(f"\n  WINNER: {winner_name} ({winner_acc*100:.1f}%) "
          f"vs baseline ({baseline_acc*100:.1f}%)", flush=True)

    # Phase 2: winner + baseline on 32-class and 64-class
    if winner_name != 'baseline':
        for nc in [32, 64]:
            print(f"\n{'='*70}")
            print(f"  PHASE 2 — {nc}-class: {winner_name} vs baseline")
            print(f"{'='*70}", flush=True)

            for vname in ['baseline', winner_name]:
                print(f"\n  --- {vname} ---", flush=True)
                acc, kept, elapsed, ent, rd = train_variant(
                    vname, nc, N, SEED, max_attempts=4000)
                print(f"  RESULT: {vname:<18s} Acc={acc*100:5.1f}%  "
                      f"Kept={kept:3d}  Time={elapsed:.0f}s", flush=True)
    else:
        print("\n  Baseline won — no self-observation variant improved accuracy.")
        print("  Running baseline on 32 and 64-class for reference...", flush=True)
        for nc in [32, 64]:
            print(f"\n  --- baseline {nc}-class ---", flush=True)
            acc, kept, elapsed, ent, rd = train_variant(
                'baseline', nc, N, SEED, max_attempts=4000)
            print(f"  RESULT: baseline Acc={acc*100:5.1f}%  "
                  f"Kept={kept:3d}  Time={elapsed:.0f}s", flush=True)

    print(f"\n{'='*70}")
    print(f"  DONE")
    print(f"{'='*70}", flush=True)
