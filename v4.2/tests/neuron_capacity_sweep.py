"""
Neuron Capacity Sweep — N-centric, plateau-based
==================================================
Q: Given N neurons and T ticks, how much knowledge (V) fits?

Design:
  - N is the primary axis (physical substrate)
  - For each N, try increasing V until the network can't learn it
  - Train until plateau (stale_limit), not fixed budget
  - Use sparse forward for N≥256 (much faster)
  - Track learning curves: accuracy over attempts

Output:
  - max_V(N):       largest V reaching ≥90% accuracy
  - V/N:            knowledge density (vocab per neuron)
  - bits/neuron:    log2(V!) / N  (information per neuron)
  - plateau_at:     attempt number where learning stopped
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
import math
import time
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy import sparse as sp
from model.graph import SelfWiringGraph


# ── Config ──────────────────────────────────────────────────────────

N_VALUES = [48, 96, 192]

TICK_PHASE1 = [8]
TICK_PHASE2 = [4, 8, 12, 16]

STALE_LIMIT = 4000
MAX_ATTEMPTS = 32000

ACC_THRESHOLD = 0.90
SNAPSHOT_EVERY = 500
SEEDS = [42, 77, 123]

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')

# Use sparse forward for N >= this threshold
SPARSE_THRESHOLD = 256


# ── Helpers ─────────────────────────────────────────────────────────

def v_candidates(N):
    """V values to try for a given N. Max V = N//2 (split I/O).
    Coarse spacing — we want ~8 points per N, not 20."""
    max_v = N // 2
    candidates = set()
    v = 4
    while v <= max_v:
        candidates.add(v)
        # ~doublings: 4, 8, 16, 24, 32, 48, 64, 96, ...
        if v < 8:
            v = 8
        elif v < 16:
            v = 16
        elif v < 32:
            v += 8
        else:
            v += max(16, v // 2)
    candidates.add(max_v)
    return sorted(candidates)


def forward_batch_sparse(net, ticks=8):
    """Sparse forward using scipy CSR — much faster for large N."""
    V, N = net.V, net.N
    mask_csr = sp.csr_matrix(net.mask)
    charges = np.zeros((V, N), dtype=np.float32)
    acts = np.zeros((V, N), dtype=np.float32)
    retain = float(net.retention)
    for t in range(ticks):
        if t == 0:
            acts[:, :V] = np.eye(V, dtype=np.float32)
        raw = acts @ mask_csr
        if sp.issparse(raw):
            raw = raw.toarray()
        else:
            raw = np.asarray(raw)
        charges += raw
        charges *= retain
        acts = np.maximum(charges - net.THRESHOLD, 0.0)
        charges = np.clip(charges, -1.0, 1.0)
    return charges[:, net.out_start:net.out_start + V]


def evaluate(net, targets, V, ticks, use_sparse=False):
    """Returns (accuracy, composite_score)."""
    if use_sparse:
        logits = forward_batch_sparse(net, ticks)
    else:
        logits = net.forward_batch(ticks)
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    acc = float((np.argmax(probs, axis=1)[:V] == targets[:V]).mean())
    tp = float(probs[np.arange(V), targets[:V]].mean())
    return acc, 0.5 * acc + 0.5 * tp


def permutation_bits(V):
    """log2(V!) — bits needed to encode a random permutation of V."""
    return sum(math.log2(i) for i in range(2, V + 1))


# ── Single run with plateau detection ──────────────────────────────

def run_one(N, V, ticks, seed):
    """Train until plateau. Track learning curve."""
    t0 = time.time()
    np.random.seed(seed)
    random.seed(seed)

    net = SelfWiringGraph(N, V)
    targets = np.random.permutation(V)
    use_sparse = N >= SPARSE_THRESHOLD

    _, score = evaluate(net, targets, V, ticks, use_sparse)
    best_acc = 0.0
    best_score = score
    stale = 0
    plateau_at = 0
    curve = []

    for att in range(MAX_ATTEMPTS):
        old_loss = int(net.loss_pct)
        old_drive = int(net.drive)
        undo = net.mutate()
        acc, s = evaluate(net, targets, V, ticks, use_sparse)

        if s > score:
            score = s
            if acc > best_acc:
                best_acc = acc
            best_score = max(best_score, s)
            stale = 0
        else:
            net.replay(undo)
            net.loss_pct = np.int8(old_loss)
            net.drive = np.int8(old_drive)
            stale += 1

        if (att + 1) % SNAPSHOT_EVERY == 0:
            curve.append({
                'att': att + 1,
                'acc': round(best_acc, 4),
                'score': round(best_score, 4),
                'edges': net.count_connections(),
            })

        if best_acc >= 1.0:
            plateau_at = att + 1
            break
        if stale >= STALE_LIMIT:
            plateau_at = att + 1 - STALE_LIMIT
            break
    else:
        plateau_at = MAX_ATTEMPTS

    edges = net.count_connections()
    elapsed = time.time() - t0
    info_bits = permutation_bits(V)

    return {
        'N': N, 'V': V, 'ticks': ticks, 'seed': seed,
        'acc': best_acc, 'score': best_score,
        'edges': edges, 'attempts': att + 1,
        'plateau_at': plateau_at, 'elapsed': elapsed,
        'V_over_N': V / N,
        'bits_total': info_bits,
        'bits_per_neuron': info_bits / N,
        'bits_per_edge': info_bits / edges if edges > 0 else 0,
        'edges_per_neuron': edges / N,
        'density_pct': edges / (N * (N - 1)) * 100,
        'curve': curve,
    }


# ── Main ────────────────────────────────────────────────────────────

def main():
    ncpu = multiprocessing.cpu_count()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ═══════════════════════════════════════════════════════════════
    # PHASE 1: T=8 fixed, sweep N × V
    # ═══════════════════════════════════════════════════════════════
    print("=" * 120)
    print("PHASE 1: CAPACITY LIMITS  (T=8, plateau-based stopping)")
    print("=" * 120)

    jobs_p1 = []
    for N in N_VALUES:
        vs = v_candidates(N)
        print(f"  N={N:4d}: V candidates = {vs}")
        for V in vs:
            for seed in SEEDS:
                jobs_p1.append((N, V, 8, seed))

    total = len(jobs_p1)
    workers = min(ncpu, total)
    print(f"\n  Total Phase 1 jobs: {total}, Workers: {workers}")
    print(f"  Stale limit: {STALE_LIMIT}, Max attempts: {MAX_ATTEMPTS}")
    print(f"  Sparse forward for N≥{SPARSE_THRESHOLD}")
    print("-" * 120)

    results_p1 = []
    done = 0

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_one, *j): j for j in jobs_p1}
        for fut in as_completed(futures):
            r = fut.result()
            results_p1.append(r)
            done += 1
            tag = "OK" if r['acc'] >= ACC_THRESHOLD else "  "
            print(f"  [{done:3d}/{total}] N={r['N']:4d} V={r['V']:3d} T=8 "
                  f"seed={r['seed']:3d}  acc={r['acc']*100:5.1f}% "
                  f"plateau@{r['plateau_at']:5d} "
                  f"edges={r['edges']:5d} "
                  f"bits/N={r['bits_per_neuron']:.2f} "
                  f"[{tag}] ({r['elapsed']:.1f}s)", flush=True)

    # ── Phase 1 analysis ──
    print(f"\n{'='*120}")
    print("PHASE 1 RESULTS: CAPACITY TABLE (T=8)")
    print(f"(✓ = ≥{ACC_THRESHOLD*100:.0f}% on ≥2/{len(SEEDS)} seeds)\n")

    header = f"{'N':>5s} {'V':>5s} {'N/V':>5s} {'V/N':>5s} {'mean_acc':>9s} {'pass':>5s} {'plateau':>8s} {'edges':>6s} {'bits/N':>7s} {'e/N':>6s} {'dens%':>6s}"
    print(header)
    print("-" * len(header))

    capacity = {}

    for N in N_VALUES:
        max_v_passing = 0
        max_v_data = None
        for V in v_candidates(N):
            runs = [r for r in results_p1 if r['N'] == N and r['V'] == V]
            if not runs:
                continue
            accs = [r['acc'] for r in runs]
            mean_acc = np.mean(accs)
            passing = sum(1 for a in accs if a >= ACC_THRESHOLD)
            mean_plateau = np.mean([r['plateau_at'] for r in runs])
            mean_edges = np.mean([r['edges'] for r in runs])
            mean_bpn = np.mean([r['bits_per_neuron'] for r in runs])
            mean_epn = np.mean([r['edges_per_neuron'] for r in runs])
            mean_dens = np.mean([r['density_pct'] for r in runs])

            mark = "✓" if passing >= 2 else " "
            print(f"{N:5d} {V:5d} {N/V:5.1f} {V/N:5.3f} "
                  f"{mean_acc*100:7.1f}%  {passing}/{len(runs)}  "
                  f"{mean_plateau:7.0f} {mean_edges:6.0f} "
                  f"{mean_bpn:7.2f} {mean_epn:6.1f} {mean_dens:5.2f}% {mark}")

            if passing >= 2:
                max_v_passing = V
                max_v_data = {
                    'max_V': V, 'V_per_N': V / N,
                    'bits_per_N': mean_bpn, 'edges_per_N': mean_epn,
                    'plateau': mean_plateau, 'density': mean_dens,
                }

        if max_v_data:
            capacity[N] = max_v_data
        print()

    # ── Capacity summary ──
    print(f"{'='*120}")
    print("CAPACITY SUMMARY (T=8)\n")
    print(f"{'N':>5s} {'max_V':>6s} {'V/N':>6s} {'bits/N':>8s} {'plateau':>8s} {'edges/N':>8s}")
    print("-" * 50)
    for N in N_VALUES:
        if N in capacity:
            c = capacity[N]
            print(f"{N:5d} {c['max_V']:6d} {c['V_per_N']:6.3f} "
                  f"{c['bits_per_N']:8.2f} {c['plateau']:8.0f} {c['edges_per_N']:8.1f}")
        else:
            print(f"{N:5d}    ---")

    # ── Scaling law ──
    ns_fit = [N for N in N_VALUES if N in capacity]
    vs_fit = [capacity[N]['max_V'] for N in ns_fit]

    print(f"\n{'='*120}")
    print("SCALING LAW: max_V = a × N^b")
    if len(ns_fit) >= 3:
        log_n = np.log(np.array(ns_fit, dtype=float))
        log_v = np.log(np.array(vs_fit, dtype=float))
        coeffs = np.polyfit(log_n, log_v, 1)
        b = coeffs[0]
        a = np.exp(coeffs[1])
        predicted = a * np.array(ns_fit) ** b
        ss_res = np.sum((np.array(vs_fit) - predicted) ** 2)
        ss_tot = np.sum((np.array(vs_fit) - np.mean(vs_fit)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        print(f"  max_V = {a:.4f} × N^{b:.4f}  (R²={r2:.4f})")

        if b > 0.9:
            print(f"  → Near-linear (b≈{b:.2f}): doubling N ≈ doubles capacity")
        elif b > 0.5:
            print(f"  → Sub-linear (b≈{b:.2f}): diminishing returns per neuron")
        else:
            print(f"  → Strongly sub-linear (b≈{b:.2f})")

        print(f"\n  Extrapolation:")
        for N_ext in [768, 1536, 3072, 6144, 96000]:
            pred_V = a * N_ext ** b
            print(f"    N={N_ext:6d} → max_V ≈ {pred_V:8.0f}  (V/N={pred_V/N_ext:.3f})")
    else:
        print(f"  Insufficient data ({len(ns_fit)} points, need ≥3)")

    # ═══════════════════════════════════════════════════════════════
    # PHASE 2: T sweep near breakpoints
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*120}")
    print("PHASE 2: TICK EFFECT ON CAPACITY")
    print("=" * 120)

    jobs_p2 = []
    for N in N_VALUES:
        if N not in capacity:
            continue
        max_v = capacity[N]['max_V']
        test_vs = sorted(set([v for v in v_candidates(N)
                              if abs(v - max_v) <= max(8, max_v // 2)]))
        if not test_vs:
            test_vs = [max_v]
        for V in test_vs:
            for T in [4, 12, 16]:
                for seed in SEEDS:
                    jobs_p2.append((N, V, T, seed))

    if jobs_p2:
        total_p2 = len(jobs_p2)
        print(f"  Phase 2 jobs: {total_p2}")
        print("-" * 120)

        results_p2 = []
        done = 0
        with ProcessPoolExecutor(max_workers=min(ncpu, total_p2)) as pool:
            futures = {pool.submit(run_one, *j): j for j in jobs_p2}
            for fut in as_completed(futures):
                r = fut.result()
                results_p2.append(r)
                done += 1
                tag = "OK" if r['acc'] >= ACC_THRESHOLD else "  "
                print(f"  [{done:3d}/{total_p2}] N={r['N']:4d} V={r['V']:3d} T={r['ticks']:2d} "
                      f"seed={r['seed']:3d}  acc={r['acc']*100:5.1f}% "
                      f"plateau@{r['plateau_at']:5d} [{tag}] ({r['elapsed']:.1f}s)",
                      flush=True)

        all_results = results_p1 + results_p2

        # ── Tick effect table ──
        print(f"\n{'='*120}")
        print("TICK EFFECT: max_V by (N, T)")
        print(f"\n{'N':>5s}", end="")
        for T in TICK_PHASE2:
            print(f"  {'T='+str(T):>10s}", end="")
        print(f"  {'T4→T16':>8s}")
        print("-" * 70)

        for N in N_VALUES:
            if N not in capacity:
                continue
            print(f"{N:5d}", end="")
            t_maxv = {}
            for T in TICK_PHASE2:
                max_v = 0
                for V in v_candidates(N):
                    runs = [r for r in all_results
                            if r['N'] == N and r['V'] == V and r['ticks'] == T]
                    passing = sum(1 for r in runs if r['acc'] >= ACC_THRESHOLD)
                    if passing >= 2:
                        max_v = V
                if max_v > 0:
                    print(f"  V={max_v:4d}", end="")
                    t_maxv[T] = max_v
                else:
                    print(f"  {'---':>10s}", end="")

            if 4 in t_maxv and 16 in t_maxv and t_maxv[4] > 0:
                ratio = t_maxv[16] / t_maxv[4]
                print(f"  {ratio:6.2f}×", end="")
            print()
    else:
        all_results = results_p1
        print("  No Phase 2 jobs")

    # ═══════════════════════════════════════════════════════════════
    # LEARNING CURVES
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*120}")
    print("LEARNING CURVES (T=8, seed=42)")
    print("attempts → accuracy\n")

    for N in N_VALUES:
        print(f"N={N}:")
        for V in v_candidates(N):
            runs = [r for r in results_p1
                    if r['N'] == N and r['V'] == V and r['seed'] == 42]
            if not runs:
                continue
            r = runs[0]
            curve = r['curve']
            if not curve:
                continue
            pts = curve[::max(1, len(curve) // 6)]
            line = " ".join([f"{p['att']:5d}:{p['acc']*100:4.0f}%" for p in pts])
            final = f"→ {r['acc']*100:.1f}%"
            print(f"  V={V:3d} (V/N={V/N:.3f}): {line} {final}")
        print()

    # ═══════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ═══════════════════════════════════════════════════════════════
    print(f"{'='*120}")
    print("FINAL SUMMARY\n")

    if capacity:
        v_per_n_values = [capacity[N]['V_per_N'] for N in capacity]
        bits_per_n_values = [capacity[N]['bits_per_N'] for N in capacity]
        print(f"  V/N (knowledge per neuron):  {min(v_per_n_values):.3f} – {max(v_per_n_values):.3f}  (mean {np.mean(v_per_n_values):.3f})")
        print(f"  bits/neuron:                 {min(bits_per_n_values):.2f} – {max(bits_per_n_values):.2f}  (mean {np.mean(bits_per_n_values):.2f})")

        avg_vn = np.mean(v_per_n_values)
        print(f"\n  Projections at V/N = {avg_vn:.3f}:")
        for target_V in [256, 1024, 8192, 32000]:
            needed_N = target_V / avg_vn
            print(f"    V={target_V:6d} → N ≈ {needed_N:8.0f}")

    # Save JSON
    save_path = os.path.join(RESULTS_DIR, 'neuron_capacity_sweep.json')
    save_data = []
    for r in all_results:
        d = {k: v for k, v in r.items() if k != 'curve'}
        d['curve_final'] = r['curve'][-1] if r['curve'] else None
        save_data.append(d)
    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Results saved: {save_path}")
    print("=" * 120)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
