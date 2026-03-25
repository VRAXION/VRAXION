"""Multi-scale resonator sweep: is the optimal inhib/reciprocal ratio universal?

Tests the same inhibitory_fraction × reciprocal_fraction grid at multiple
network sizes (32, 64, 128, 256 neurons) to check if:
  1. The optimal inhibitory fraction shifts with scale
  2. Reciprocal tolerance changes with network size
  3. The fly brain's 40% inhib + 50% reciprocal becomes viable at larger scales

Deterministic — no randomness beyond topology construction (fixed seeds).
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def run_dynamics(adj_matrix, polarity, input_vec, ticks=8, decay=0.10, theta=1.5):
    """Minimal spike dynamics."""
    H = adj_matrix.shape[0]
    charge = np.zeros(H, dtype=np.float64)
    act = np.zeros(H, dtype=np.float64)
    history = []

    for tick in range(ticks):
        charge = np.maximum(charge - decay, 0.0)
        if tick < 2:
            act = act + input_vec
        signal = (act * polarity) @ adj_matrix.astype(np.float64)
        charge += signal
        charge = np.clip(charge, 0.0, 15.0)
        fired = charge >= theta
        act = fired.astype(np.float64)
        charge[fired] = 0.0
        history.append({
            'spikes': fired.copy(),
            'n_active': int(np.sum(fired)),
            'energy': float(np.sum(charge)),
        })
    return history


def make_topology(H, n_edges, inhib_frac, recip_frac, seed=42):
    """Build topology with specified inhibitory and reciprocal fractions."""
    rng = np.random.RandomState(seed)
    adj = np.zeros((H, H), dtype=np.float64)
    polarity = np.ones(H, dtype=np.float64)

    # Set inhibitory neurons
    n_inhib = max(1, int(H * inhib_frac))
    inhib_neurons = rng.choice(H, n_inhib, replace=False)
    polarity[inhib_neurons] = -1.0

    excit = [i for i in range(H) if polarity[i] > 0]
    inhib = [i for i in range(H) if polarity[i] < 0]

    # Phase 1: Reciprocal E↔I pairs
    n_recip_edges = int(n_edges * recip_frac)
    n_recip_pairs = n_recip_edges // 2
    placed = 0
    pairs_made = 0
    attempts = 0
    while pairs_made < n_recip_pairs and excit and inhib and attempts < n_recip_pairs * 20:
        attempts += 1
        e = excit[rng.randint(0, len(excit))]
        i = inhib[rng.randint(0, len(inhib))]
        if adj[e, i] == 0 and adj[i, e] == 0:
            adj[e, i] = 1.0
            adj[i, e] = 1.0
            pairs_made += 1
            placed += 2

    # Phase 2: Fill remaining edges randomly
    while placed < n_edges:
        r, c = rng.randint(0, H), rng.randint(0, H)
        if r != c and adj[r, c] == 0:
            adj[r, c] = 1.0
            placed += 1

    return adj, polarity


def run_sweep_at_scale(H, edges_per_neuron=4, n_inputs=8, ticks=8):
    """Run inhib × reciprocal sweep for a given network size."""
    n_edges = H * edges_per_neuron

    # Build diverse inputs
    inputs = []
    for i in range(n_inputs):
        v = np.zeros(H, dtype=np.float64)
        n_active = max(4, H // 8)  # Scale input neurons with network size
        for j in range(n_active):
            v[(i * n_active + j) % H] = 5.0
        inputs.append(v)

    inhib_fracs = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]
    recip_fracs = [0.0, 0.15, 0.30, 0.50]

    results = {}
    for inhib_f in inhib_fracs:
        for recip_f in recip_fracs:
            adj, pol = make_topology(H, n_edges, inhib_f, recip_f, seed=42)

            histories = [run_dynamics(adj, pol, inp, ticks=ticks) for inp in inputs]

            # Separation: distinct output patterns
            final_patterns = [tuple(h[-1]['spikes']) for h in histories]
            n_distinct = len(set(final_patterns))

            # Dead outputs
            n_dead = sum(1 for h in histories if h[-1]['n_active'] == 0)

            # Mean active neurons at final tick
            mean_active = np.mean([h[-1]['n_active'] for h in histories])

            # Pairwise hamming distance
            dists = []
            for i in range(len(histories)):
                for j in range(i + 1, len(histories)):
                    s1 = histories[i][-1]['spikes']
                    s2 = histories[j][-1]['spikes']
                    dists.append(float(np.sum(s1 != s2)))
            mean_dist = np.mean(dists) if dists else 0.0

            results[(inhib_f, recip_f)] = {
                'distinct': n_distinct,
                'dead': n_dead,
                'mean_active': mean_active,
                'mean_dist': mean_dist,
            }

    return results, inhib_fracs, recip_fracs


if __name__ == "__main__":
    SCALES = [32, 64, 128, 256, 512]
    N_INPUTS = 8

    print("=" * 70)
    print("  MULTI-SCALE RESONATOR SWEEP")
    print(f"  Scales: {SCALES}, edges_per_neuron=4, inputs={N_INPUTS}")
    print("=" * 70)

    all_scale_results = {}

    for H in SCALES:
        print(f"\n{'━' * 70}")
        print(f"  SCALE: H={H} neurons, {H*4} edges")
        print(f"{'━' * 70}")

        results, inhib_fracs, recip_fracs = run_sweep_at_scale(H, n_inputs=N_INPUTS)
        all_scale_results[H] = results

        # Print grid
        header = f"{'inhib↓ / recip→':>18s}"
        for rf in recip_fracs:
            header += f"  {rf*100:4.0f}%"
        print(header)
        print("-" * (18 + 7 * len(recip_fracs)))

        for inf in inhib_fracs:
            row = f"  {inf*100:4.0f}% inhib    "
            for rf in recip_fracs:
                r = results[(inf, rf)]
                if r['dead'] == N_INPUTS:
                    row += f"  DEAD"
                elif r['distinct'] == N_INPUTS:
                    row += f"  {r['distinct']}/{N_INPUTS}✓"
                else:
                    row += f"  {r['distinct']}/{N_INPUTS} "

            print(row)

        # Best configs for this scale
        best = max(results.items(), key=lambda x: (x[1]['distinct'], -x[1]['dead'], x[1]['mean_dist']))
        print(f"\n  Best: inhib={best[0][0]*100:.0f}%, recip={best[0][1]*100:.0f}% "
              f"→ distinct={best[1]['distinct']}/{N_INPUTS}, "
              f"dead={best[1]['dead']}, active={best[1]['mean_active']:.1f}")

    # Cross-scale comparison
    print(f"\n{'=' * 70}")
    print("  CROSS-SCALE COMPARISON: Best configs per scale")
    print(f"{'=' * 70}")
    print(f"{'Scale':>8s}  {'Best inhib':>10s}  {'Best recip':>10s}  {'Distinct':>8s}  {'Dead':>6s}  {'Active':>8s}")
    print("-" * 60)

    for H in SCALES:
        results = all_scale_results[H]
        # Find configs with max distinct and min dead
        best_configs = []
        for (inf, rf), r in results.items():
            best_configs.append((inf, rf, r))
        best_configs.sort(key=lambda x: (-x[2]['distinct'], x[2]['dead'], -x[2]['mean_dist']))
        b = best_configs[0]
        print(f"  H={H:>4d}  {b[0]*100:>9.0f}%  {b[1]*100:>9.0f}%  "
              f"{b[2]['distinct']:>4d}/{N_INPUTS}  {b[2]['dead']:>5d}  {b[2]['mean_active']:>7.1f}")

    # Check fly-brain config (40% inhib, 50% recip) across scales
    print(f"\n  Fly-brain config (40% inhib, 50% recip) across scales:")
    for H in SCALES:
        r = all_scale_results[H].get((0.40, 0.50), None)
        if r:
            status = "DEAD" if r['dead'] == N_INPUTS else f"{r['distinct']}/{N_INPUTS}"
            print(f"    H={H:>4d}: {status}, active={r['mean_active']:.1f}, dead={r['dead']}")

    # Check our INSTNCT config (20% inhib, 0% recip) across scales
    print(f"\n  INSTNCT config (20% inhib, 0% recip) across scales:")
    for H in SCALES:
        r = all_scale_results[H].get((0.20, 0.0), None)
        if r:
            status = "DEAD" if r['dead'] == N_INPUTS else f"{r['distinct']}/{N_INPUTS}"
            print(f"    H={H:>4d}: {status}, active={r['mean_active']:.1f}, dead={r['dead']}")

    print(f"\n{'=' * 70}")
