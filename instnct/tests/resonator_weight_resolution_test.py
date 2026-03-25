"""Weight resolution sweep: how many bits do edge weights need?

Tests whether graded weights rescue the fly-brain config (40% inhib + 50% recip)
that dies with binary (0/1) edges.

Weight schemes (from coarsest to finest):
  1. binary:     0 or 1                    (1 bit)
  2. 2-level:    0, 0.5, 1.0               (1.5 bits — "strong yes/no")
  3. 3-level:    0, 0.3, 0.7, 1.0          (2 bits — "low/med/high")
  4. int3:       0, 1/7, 2/7, ..., 7/7     (3 bits)
  5. int4:       0, 1/15, 2/15, ..., 15/15 (4 bits)
  6. float:      continuous [0, 1]          (32 bits)

For each scheme, we test: does the fly-brain config (40% inhib, 50% recip)
produce input separation?

Key insight: inhibitory edges might need LOWER weight than excitatory
to avoid killing the network. In the real brain, E→I is often stronger
than I→E (the inhibition is calibrated).
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def run_dynamics(adj_matrix, polarity, input_vec, ticks=8, decay=0.10, theta=1.5):
    """Minimal spike dynamics."""
    H = adj_matrix.shape[0]
    charge = np.zeros(H, dtype=np.float64)
    state = np.zeros(H, dtype=np.float64)  # binary firing state
    history = []

    for tick in range(ticks):
        charge = np.maximum(charge - decay, 0.0)
        if tick < 2:
            state = state + input_vec
        signal = (state * polarity) @ adj_matrix
        charge += signal
        charge = np.clip(charge, 0.0, 15.0)
        fired = charge >= theta
        state = fired.astype(np.float64)
        charge[fired] = 0.0
        history.append({
            'spikes': fired.copy(),
            'n_active': int(np.sum(fired)),
            'energy': float(np.sum(charge)),
        })
    return history


def quantize_weights(raw_weights, scheme):
    """Quantize continuous [0,1] weights to a specific resolution."""
    if scheme == 'binary':
        return (raw_weights > 0.5).astype(np.float64)
    elif scheme == '2-level':
        # 0, 0.5, 1.0
        q = np.round(raw_weights * 2) / 2
        return q
    elif scheme == '3-level':
        # 0, 0.33, 0.67, 1.0
        q = np.round(raw_weights * 3) / 3
        return q
    elif scheme == 'int3':
        q = np.round(raw_weights * 7) / 7
        return q
    elif scheme == 'int4':
        q = np.round(raw_weights * 15) / 15
        return q
    elif scheme == 'float':
        return raw_weights
    else:
        raise ValueError(f"Unknown scheme: {scheme}")


def make_topology_graded(H, n_edges, inhib_frac, recip_frac, scheme,
                         ei_weight_ratio=1.0, seed=42):
    """Build topology with graded weights.

    ei_weight_ratio: how strong I→E edges are relative to E→I.
      1.0 = equal strength
      0.5 = inhibitory feedback is half strength
      0.3 = inhibitory feedback is weak
    """
    rng = np.random.RandomState(seed)
    adj = np.zeros((H, H), dtype=np.float64)
    polarity = np.ones(H, dtype=np.float64)

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
            # E→I: draw from [0.5, 1.0] — excludes very weak connections
            # (matching FlyWire: median connection ~3 synapses out of max ~2400)
            w_ei = rng.uniform(0.5, 1.0)
            # I→E: same range, scaled by ei_weight_ratio
            w_ie = rng.uniform(0.5, 1.0) * ei_weight_ratio
            adj[e, i] = w_ei
            adj[i, e] = w_ie
            pairs_made += 1
            placed += 2

    # Phase 2: Fill remaining edges randomly
    while placed < n_edges:
        r, c = rng.randint(0, H), rng.randint(0, H)
        if r != c and adj[r, c] == 0:
            w = rng.uniform(0.3, 1.0)
            adj[r, c] = w
            placed += 1

    # Quantize
    nonzero = adj > 0
    adj[nonzero] = quantize_weights(adj[nonzero], scheme)
    # Make sure quantization didn't zero out edges (except binary)
    if scheme != 'binary':
        adj[nonzero & (adj == 0)] = 1.0 / 15  # minimum weight

    return adj, polarity


def measure_separation(adj, polarity, H, n_inputs=8, ticks=8):
    """Measure input separation quality."""
    inputs = []
    n_active = max(4, H // 8)
    for i in range(n_inputs):
        v = np.zeros(H, dtype=np.float64)
        for j in range(n_active):
            v[(i * n_active + j) % H] = 5.0
        inputs.append(v)

    histories = [run_dynamics(adj, polarity, inp, ticks=ticks) for inp in inputs]

    final_patterns = [tuple(h[-1]['spikes']) for h in histories]
    n_distinct = len(set(final_patterns))
    n_dead = sum(1 for h in histories if h[-1]['n_active'] == 0)
    mean_active = np.mean([h[-1]['n_active'] for h in histories])

    dists = []
    for i in range(len(histories)):
        for j in range(i + 1, len(histories)):
            s1 = histories[i][-1]['spikes']
            s2 = histories[j][-1]['spikes']
            dists.append(float(np.sum(s1 != s2)))
    mean_dist = np.mean(dists) if dists else 0.0

    return {
        'distinct': n_distinct,
        'dead': n_dead,
        'mean_active': mean_active,
        'mean_dist': mean_dist,
    }


if __name__ == "__main__":
    N_INPUTS = 8
    SCHEMES = ['binary', '2-level', '3-level', 'int3', 'int4', 'float']
    SCALES = [32, 64, 128, 256]

    print("=" * 75)
    print("  WEIGHT RESOLUTION SWEEP")
    print(f"  How many bits do edge weights need?")
    print(f"  Scales: {SCALES}, schemes: {SCHEMES}")
    print("=" * 75)

    # ── Part 1: Fix fly-brain config, vary weight resolution ──
    print(f"\n{'━' * 75}")
    print("  PART 1: Fly-brain config (40% inhib, 50% recip)")
    print("  Equal E→I and I→E strength (ratio=1.0)")
    print(f"{'━' * 75}")

    for H in SCALES:
        n_edges = H * 4
        print(f"\n  H={H}:")
        row = f"    {'scheme':<12s}"
        for s in SCHEMES:
            row += f" {s:>8s}"
        print(row)
        print(f"    {'':<12s}" + "-" * (9 * len(SCHEMES)))

        row_dist = f"    {'distinct':<12s}"
        row_dead = f"    {'dead':<12s}"
        row_act = f"    {'active':<12s}"

        for scheme in SCHEMES:
            adj, pol = make_topology_graded(H, n_edges, 0.40, 0.50, scheme,
                                            ei_weight_ratio=1.0)
            r = measure_separation(adj, pol, H, n_inputs=N_INPUTS)
            row_dist += f" {r['distinct']:>4d}/{N_INPUTS} "
            row_dead += f" {r['dead']:>6d}  "
            row_act += f" {r['mean_active']:>7.1f} "

        print(row_dist)
        print(row_dead)
        print(row_act)

    # ── Part 2: Vary inhibitory feedback strength ──
    print(f"\n{'━' * 75}")
    print("  PART 2: What if inhibitory feedback is WEAKER?")
    print("  Fly config (40% inhib, 50% recip) with different I→E ratios")
    print("  Using float weights, testing I→E strength = 0.1 to 1.0")
    print(f"{'━' * 75}")

    EI_RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]

    for H in SCALES:
        n_edges = H * 4
        print(f"\n  H={H}:")
        header = f"    {'I→E ratio':<12s}"
        for ratio in EI_RATIOS:
            header += f" {ratio:>6.1f}"
        print(header)
        print(f"    {'':<12s}" + "-" * (7 * len(EI_RATIOS)))

        row_dist = f"    {'distinct':<12s}"
        row_dead = f"    {'dead':<12s}"
        row_act = f"    {'active':<12s}"

        for ratio in EI_RATIOS:
            adj, pol = make_topology_graded(H, n_edges, 0.40, 0.50, 'float',
                                            ei_weight_ratio=ratio)
            r = measure_separation(adj, pol, H, n_inputs=N_INPUTS)
            row_dist += f"  {r['distinct']:>2d}/{N_INPUTS}"
            row_dead += f"  {r['dead']:>4d} "
            row_act += f"  {r['mean_active']:>4.0f} "

        print(row_dist)
        print(row_dead)
        print(row_act)

    # ── Part 3: Best resolution per config ──
    print(f"\n{'━' * 75}")
    print("  PART 3: Full grid — scheme × inhib × recip at H=128")
    print(f"{'━' * 75}")

    H = 128
    n_edges = H * 4
    INHIB_FRACS = [0.20, 0.30, 0.40]
    RECIP_FRACS = [0.0, 0.15, 0.30, 0.50]

    for scheme in SCHEMES:
        print(f"\n  Scheme: {scheme}")
        header = f"    {'inhib↓/recip→':>16s}"
        for rf in RECIP_FRACS:
            header += f"  {rf*100:4.0f}%"
        print(header)
        print(f"    {'':<16s}" + "-" * (7 * len(RECIP_FRACS)))

        for inf in INHIB_FRACS:
            row = f"    {inf*100:5.0f}% inhib   "
            for rf in RECIP_FRACS:
                adj, pol = make_topology_graded(H, n_edges, inf, rf, scheme)
                r = measure_separation(adj, pol, H, n_inputs=N_INPUTS)
                if r['dead'] == N_INPUTS:
                    row += f"  DEAD"
                elif r['distinct'] == N_INPUTS:
                    row += f"  {r['distinct']}/{N_INPUTS}✓"
                else:
                    row += f"  {r['distinct']}/{N_INPUTS} "
            print(row)

    print(f"\n{'=' * 75}")
