"""
Neuron Capacity Sweep — N-centric scaling analysis
====================================================
Primary question: Given N neurons and T ticks, what is the MAXIMUM
vocab size V the network can learn?

We fix N and increase V until accuracy collapses.
This gives us the fundamental capacity metric: bits-per-neuron.

Metrics:
  - max_V(N, T):  largest V that reaches ≥90% accuracy
  - knowledge_density = max_V / N  (vocab units per neuron)
  - bits_per_neuron = log2(max_V!) / N  (information bits stored per neuron)
  - edges_per_neuron = final_edges / N
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
import math
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from model.graph import SelfWiringGraph


# ── Config ──────────────────────────────────────────────────────────
# N values to test (the physical substrate sizes)
N_VALUES = [48, 96, 192, 384, 768]

# For each N, we try V values from small to large.
# V must be ≤ N//2 (need room for split I/O: input V + output V ≤ N)
# We'll generate V candidates dynamically per N.

# Ticks to test
TICK_VALUES = [4, 8, 12, 16]

# Training budget — scales with task difficulty
def budget_for(N, V):
    """More budget for harder configs (bigger V relative to N)."""
    base = max(8000, V * 100)
    return min(base, 64000)

# Accuracy threshold to consider "learned"
ACC_THRESHOLD = 0.90

# Seeds
SEEDS = [42, 77, 123]

# ── Helpers ─────────────────────────────────────────────────────────

def v_candidates(N):
    """Generate V values to try for a given N.
    V must satisfy: 2*V ≤ N (split I/O needs V input + V output neurons).
    We use roughly logarithmic spacing."""
    max_v = N // 2  # hard upper bound from split I/O
    # But also NV_RATIO=3 means normal V = N/3, so we go beyond that
    candidates = []
    v = 4
    while v <= max_v:
        candidates.append(v)
        if v < 16:
            v += 4
        elif v < 64:
            v += 8
        elif v < 256:
            v += 16
        else:
            v += 32
    if candidates and candidates[-1] != max_v:
        candidates.append(max_v)
    return candidates


def evaluate(net, targets, V, ticks):
    """Score a network. Returns (accuracy, composite_score)."""
    logits = net.forward_batch(ticks)
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    acc = float((np.argmax(probs, axis=1)[:V] == targets[:V]).mean())
    tp = float(probs[np.arange(V), targets[:V]].mean())
    return acc, 0.5 * acc + 0.5 * tp


def permutation_bits(V):
    """Information content of a random permutation: log2(V!)
    This is how many bits a perfect learner needs to memorize the mapping."""
    return sum(math.log2(i) for i in range(2, V + 1))


# ── Single run ──────────────────────────────────────────────────────

def run_one(N, V, ticks, seed):
    """Train one (N, V, T) config. Returns stats dict."""
    t0 = time.time()
    np.random.seed(seed)
    random.seed(seed)

    # Create network with explicit N, V
    net = SelfWiringGraph(N, V)
    assert net.N == N, f"Expected N={N}, got {net.N}"
    targets = np.random.permutation(V)

    budget = budget_for(N, V)
    _, score = evaluate(net, targets, V, ticks)
    best_acc = 0.0
    best_score = score
    stale = 0

    for att in range(budget):
        old_loss = int(net.loss_pct)
        old_drive = int(net.drive)
        undo = net.mutate()
        acc, s = evaluate(net, targets, V, ticks)

        if s > score:
            score = s
            best_acc = max(best_acc, acc)
            best_score = max(best_score, s)
            stale = 0
        else:
            net.replay(undo)
            net.loss_pct = np.int8(old_loss)
            net.drive = np.int8(old_drive)
            stale += 1

        if best_acc >= 1.0 or stale >= 6000:
            break

    edges = net.count_connections()
    elapsed = time.time() - t0

    info_bits = permutation_bits(V)

    return {
        'N': N,
        'V': V,
        'ticks': ticks,
        'seed': seed,
        'acc': best_acc,
        'score': best_score,
        'edges': edges,
        'budget': budget,
        'attempts': att + 1,
        'elapsed': elapsed,
        # Derived metrics
        'nv_ratio_actual': N / V,
        'knowledge_density': V / N,           # vocab per neuron
        'edges_per_neuron': edges / N,
        'bits_total': info_bits,               # log2(V!) bits in the task
        'bits_per_neuron': info_bits / N,      # bits stored per neuron
        'bits_per_edge': info_bits / edges if edges > 0 else 0,
    }


# ── Main ────────────────────────────────────────────────────────────

def main():
    ncpu = multiprocessing.cpu_count()

    # Build job list
    jobs = []
    for N in N_VALUES:
        for V in v_candidates(N):
            for T in TICK_VALUES:
                for seed in SEEDS:
                    jobs.append((N, V, T, seed))

    total = len(jobs)
    print(f"NEURON CAPACITY SWEEP — N-centric analysis")
    print(f"  N values:    {N_VALUES}")
    print(f"  Tick values: {TICK_VALUES}")
    print(f"  Seeds:       {SEEDS}")
    print(f"  Total jobs:  {total}")
    print(f"  CPU cores:   {ncpu}")
    print(f"  Workers:     {min(ncpu, total)}")
    print(f"  ACC threshold: {ACC_THRESHOLD*100:.0f}%")
    print("=" * 120)

    results = []
    done = 0

    with ProcessPoolExecutor(max_workers=min(ncpu, total)) as pool:
        futures = {pool.submit(run_one, *j): j for j in jobs}
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            done += 1
            if done % 10 == 0 or done == total:
                print(f"  [{done:4d}/{total}] N={r['N']:4d} V={r['V']:3d} T={r['ticks']:2d} "
                      f"seed={r['seed']:3d}  acc={r['acc']*100:5.1f}%  "
                      f"edges={r['edges']:5d}  bits/neuron={r['bits_per_neuron']:.2f}  "
                      f"({r['elapsed']:.1f}s)", flush=True)

    # ── Aggregate: for each (N, T), find max V that reaches threshold ──
    print(f"\n{'='*120}")
    print("MAX LEARNABLE VOCAB PER NEURON COUNT")
    print(f"(accuracy ≥ {ACC_THRESHOLD*100:.0f}% on ≥2 of {len(SEEDS)} seeds)")
    print(f"\n{'N':>5s}", end="")
    for T in TICK_VALUES:
        print(f"  {'T='+str(T)+' max_V':>12s} {'V/N':>6s} {'bits/N':>7s} {'e/N':>6s}", end="")
    print()
    print("-" * 120)

    capacity_data = {}  # (N, T) -> {max_V, bits_per_neuron, ...}

    for N in N_VALUES:
        print(f"{N:5d}", end="")
        for T in TICK_VALUES:
            # For each V, check if majority of seeds passed threshold
            max_v = 0
            max_v_bits = 0
            max_v_edges = 0
            for V in v_candidates(N):
                runs = [r for r in results
                        if r['N'] == N and r['V'] == V and r['ticks'] == T]
                passing = sum(1 for r in runs if r['acc'] >= ACC_THRESHOLD)
                if passing >= 2:  # majority of seeds
                    max_v = V
                    max_v_bits = np.mean([r['bits_per_neuron'] for r in runs])
                    max_v_edges = np.mean([r['edges_per_neuron'] for r in runs])

            if max_v > 0:
                print(f"  {max_v:12d} {max_v/N:6.3f} {max_v_bits:7.2f} {max_v_edges:6.1f}", end="")
                capacity_data[(N, T)] = {
                    'max_V': max_v, 'V_per_N': max_v / N,
                    'bits_per_N': max_v_bits, 'edges_per_N': max_v_edges,
                }
            else:
                print(f"  {'---':>12s} {'---':>6s} {'---':>7s} {'---':>6s}", end="")
        print()

    # ── Detailed breakdown per N ──
    print(f"\n{'='*120}")
    print("DETAILED ACCURACY BY (N, V, T) — averaged over seeds")
    print(f"\n{'N':>5s} {'V':>5s} {'N/V':>5s}", end="")
    for T in TICK_VALUES:
        print(f"  {'T='+str(T):>8s}", end="")
    print(f"  {'bits(V!)':>8s}")
    print("-" * 100)

    for N in N_VALUES:
        for V in v_candidates(N):
            print(f"{N:5d} {V:5d} {N/V:5.1f}", end="")
            for T in TICK_VALUES:
                runs = [r for r in results
                        if r['N'] == N and r['V'] == V and r['ticks'] == T]
                if runs:
                    mean_acc = np.mean([r['acc'] for r in runs])
                    marker = " *" if mean_acc >= ACC_THRESHOLD else "  "
                    print(f"  {mean_acc*100:5.1f}%{marker}", end="")
                else:
                    print(f"  {'---':>8s}", end="")
            print(f"  {permutation_bits(V):8.1f}")
        print()

    # ── Scaling law fits ──
    print(f"\n{'='*120}")
    print("SCALING LAW: max_V = a × N^b  (per tick count)")
    for T in TICK_VALUES:
        ns = []
        vs = []
        for N in N_VALUES:
            key = (N, T)
            if key in capacity_data and capacity_data[key]['max_V'] > 0:
                ns.append(N)
                vs.append(capacity_data[key]['max_V'])
        if len(ns) >= 3:
            log_n = np.log(np.array(ns, dtype=float))
            log_v = np.log(np.array(vs, dtype=float))
            coeffs = np.polyfit(log_n, log_v, 1)
            b = coeffs[0]
            a = np.exp(coeffs[1])
            predicted = a * np.array(ns) ** b
            ss_res = np.sum((np.array(vs) - predicted) ** 2)
            ss_tot = np.sum((np.array(vs) - np.mean(vs)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            print(f"  T={T:2d}:  max_V = {a:.4f} × N^{b:.4f}  (R²={r2:.4f})")

            # Extrapolation
            for N_ext in [1536, 3072, 6144]:
                pred_V = a * N_ext ** b
                print(f"         N={N_ext:5d} → max_V ≈ {pred_V:.0f}  "
                      f"(V/N={pred_V/N_ext:.3f})")
        else:
            print(f"  T={T:2d}:  insufficient data points ({len(ns)})")

    # ── Tick effect analysis ──
    print(f"\n{'='*120}")
    print("TICK EFFECT: does more T increase capacity?")
    for N in N_VALUES:
        t_vs = []
        for T in TICK_VALUES:
            key = (N, T)
            if key in capacity_data:
                t_vs.append((T, capacity_data[key]['max_V']))
        if len(t_vs) >= 2:
            ts = [x[0] for x in t_vs]
            vs = [x[1] for x in t_vs]
            improvement = vs[-1] / vs[0] if vs[0] > 0 else float('inf')
            print(f"  N={N:4d}: T={ts[0]}→max_V={vs[0]:3d}  "
                  f"T={ts[-1]}→max_V={vs[-1]:3d}  "
                  f"({improvement:.2f}× from {ts[0]}→{ts[-1]} ticks)")

    # ── Key findings summary ──
    print(f"\n{'='*120}")
    print("KEY FINDINGS")
    if capacity_data:
        # Best bits/neuron across all configs
        best_key = max(capacity_data.keys(), key=lambda k: capacity_data[k].get('bits_per_N', 0))
        best = capacity_data[best_key]
        print(f"  Best bits/neuron: {best['bits_per_N']:.2f} at N={best_key[0]}, T={best_key[1]}")
        print(f"  Best V/N ratio:   {best['V_per_N']:.3f} (max_V={best['max_V']})")

        # Average V/N across T=8
        t8_data = [(k, v) for k, v in capacity_data.items() if k[1] == 8]
        if t8_data:
            avg_vn = np.mean([v['V_per_N'] for _, v in t8_data])
            print(f"  Average V/N at T=8: {avg_vn:.3f}")
            print(f"  → 1 neuron ≈ {avg_vn:.3f} vocab unit capacity")
            print(f"  → For V=32K, need N ≈ {32000/avg_vn:,.0f} neurons")

    print("=" * 120)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
