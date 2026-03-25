"""Deterministic toy test: Resonator Chamber hypothesis.

Tests whether fly-brain-inspired topology (reciprocal E↔I pairs,
clustered structure) produces better signal filtering than
random or feedforward topologies.

Predictions to test:
  P1: Reciprocal E↔I topology separates different inputs better
  P2: Clustered topology is more selective (fewer neurons active)
  P3: Signal energy decays over ticks (interference kills most paths)
  P4: Surviving pattern is INPUT-DEPENDENT (not random noise)

All tests are DETERMINISTIC — no randomness, no training.
We hand-craft topologies and measure signal properties directly.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Minimal spike dynamics (no graph.py dependency) ──────────────────

def run_dynamics(adj_matrix, polarity, input_vec, ticks=8, decay=0.10, theta=1.5):
    """Run spike dynamics on a small network. Returns per-tick state.

    Args:
        adj_matrix: (H, H) binary connectivity matrix
        polarity: (H,) per-neuron sign (+1 excitatory, -1 inhibitory)
        input_vec: (H,) input current injection
        ticks: number of simulation steps (default 8 ≈ 1× diameter for H=32)
        decay: charge leak per tick (subtractive)
        theta: firing threshold

    Returns:
        list of dicts per tick with keys: charge, spikes, spike_pattern,
        energy, n_active
    """
    H = adj_matrix.shape[0]
    charge = np.zeros(H, dtype=np.float64)
    state = np.zeros(H, dtype=np.float64)  # binary firing state per neuron
    history = []

    for tick in range(ticks):
        # Decay
        charge = np.maximum(charge - decay, 0.0)

        # Inject input first 2 ticks
        if tick < 2:
            state = state + input_vec

        # Propagate: state * polarity through adjacency
        signal = (state * polarity) @ adj_matrix.astype(np.float64)
        charge += signal
        charge = np.clip(charge, 0.0, 15.0)

        # Spike decision
        fired = charge >= theta
        state = fired.astype(np.float64)
        charge[fired] = 0.0  # hard reset

        spike_pattern = state * polarity
        history.append({
            'charge': charge.copy(),
            'spikes': fired.copy(),
            'spike_pattern': spike_pattern.copy(),
            'energy': float(np.sum(charge)),
            'n_active': int(np.sum(fired)),
        })

    return history


# ── Topology builders ────────────────────────────────────────────────

def make_feedforward(H, layers=4):
    """Pure feedforward: layer 0 → layer 1 → ... → layer N."""
    adj = np.zeros((H, H), dtype=np.float64)
    layer_size = H // layers
    for l in range(layers - 1):
        for i in range(layer_size):
            src = l * layer_size + i
            for j in range(layer_size):
                dst = (l + 1) * layer_size + j
                if src < H and dst < H:
                    adj[src, dst] = 1.0
    polarity = np.ones(H, dtype=np.float64)
    return adj, polarity, "feedforward"


def make_random_recurrent(H, n_edges, seed=42):
    """Random graph with n_edges, 20% inhibitory."""
    rng = np.random.RandomState(seed)
    adj = np.zeros((H, H), dtype=np.float64)
    placed = 0
    while placed < n_edges:
        r, c = rng.randint(0, H), rng.randint(0, H)
        if r != c and adj[r, c] == 0:
            adj[r, c] = 1.0
            placed += 1
    polarity = np.ones(H, dtype=np.float64)
    inhib = rng.choice(H, H // 5, replace=False)
    polarity[inhib] = -1.0
    return adj, polarity, "random_recurrent"


def make_fly_inspired(H, n_edges, seed=42):
    """Fly-brain-inspired: reciprocal E↔I pairs, clustered.

    - 40% inhibitory (matching fly brain)
    - Reciprocal E↔I pairs as core structure
    - Local clustering (neighbors connect to neighbors)
    """
    rng = np.random.RandomState(seed)
    adj = np.zeros((H, H), dtype=np.float64)

    # 40% inhibitory
    polarity = np.ones(H, dtype=np.float64)
    inhib_neurons = rng.choice(H, int(H * 0.4), replace=False)
    polarity[inhib_neurons] = -1.0

    excit = [i for i in range(H) if polarity[i] > 0]
    inhib = [i for i in range(H) if polarity[i] < 0]
    placed = 0

    # Phase 1: Reciprocal E↔I pairs (50% of edges, matching fly brain)
    target_recip = n_edges // 2
    pairs_made = 0
    while pairs_made < target_recip // 2 and excit and inhib:
        e = excit[rng.randint(0, len(excit))]
        i = inhib[rng.randint(0, len(inhib))]
        if adj[e, i] == 0 and adj[i, e] == 0:
            adj[e, i] = 1.0  # E → I
            adj[i, e] = 1.0  # I → E (reciprocal)
            pairs_made += 1
            placed += 2

    # Phase 2: Local clustering (neighbors of neighbors connect)
    attempts = 0
    while placed < n_edges and attempts < n_edges * 10:
        attempts += 1
        # Pick a random edge, connect to its neighbor's neighbor
        src = rng.randint(0, H)
        targets = np.where(adj[src] > 0)[0]
        if len(targets) == 0:
            continue
        mid = targets[rng.randint(0, len(targets))]
        mid_targets = np.where(adj[mid] > 0)[0]
        if len(mid_targets) == 0:
            continue
        dst = mid_targets[rng.randint(0, len(mid_targets))]
        if dst != src and adj[src, dst] == 0:
            adj[src, dst] = 1.0
            placed += 1

    # Phase 3: Fill remaining randomly
    while placed < n_edges:
        r, c = rng.randint(0, H), rng.randint(0, H)
        if r != c and adj[r, c] == 0:
            adj[r, c] = 1.0
            placed += 1

    return adj, polarity, "fly_inspired"


def make_random_EI_matched(H, n_edges, seed=42):
    """Control: same 40% inhibitory, same edge count, but NO reciprocal bias."""
    rng = np.random.RandomState(seed)
    adj = np.zeros((H, H), dtype=np.float64)

    polarity = np.ones(H, dtype=np.float64)
    inhib_neurons = rng.choice(H, int(H * 0.4), replace=False)
    polarity[inhib_neurons] = -1.0

    placed = 0
    while placed < n_edges:
        r, c = rng.randint(0, H), rng.randint(0, H)
        if r != c and adj[r, c] == 0:
            adj[r, c] = 1.0
            placed += 1

    return adj, polarity, "random_40pct_inhib"


# ── Metrics ──────────────────────────────────────────────────────────

def pattern_distance(h1, h2):
    """Hamming distance between two spike patterns."""
    s1 = h1[-1]['spikes']
    s2 = h2[-1]['spikes']
    return float(np.sum(s1 != s2))


def energy_curve(history):
    """Energy (total charge) over time."""
    return [h['energy'] for h in history]


def active_curve(history):
    """Active neuron count over time."""
    return [h['n_active'] for h in history]


def output_entropy(history):
    """How spread out is the final spike pattern? Lower = more selective."""
    spikes = history[-1]['spikes'].astype(float)
    if spikes.sum() == 0:
        return 0.0
    p = spikes / spikes.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def pattern_consistency(adj, polarity, input_vec, n_runs=5, ticks=8):
    """Run same input multiple times (from reset), check if same output.
    Should be 100% for deterministic system."""
    patterns = []
    for _ in range(n_runs):
        h = run_dynamics(adj, polarity, input_vec, ticks=ticks)
        patterns.append(h[-1]['spikes'].copy())
    # All should be identical
    same = all(np.array_equal(patterns[0], p) for p in patterns[1:])
    return same


# ── Main test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    H = 32
    N_EDGES = 120  # ~3.75 edges/neuron
    TICKS = 8      # ≈ 1× diameter for H=32 (see RESONATOR_THEORY.md Finding 3)
    N_INPUTS = 8   # different input patterns to test

    print("=" * 70)
    print("  RESONATOR CHAMBER — DETERMINISTIC TOY TEST")
    print(f"  H={H}, edges={N_EDGES}, ticks={TICKS}, inputs={N_INPUTS}")
    print("=" * 70)

    # Build topologies
    topologies = [
        make_feedforward(H),
        make_random_recurrent(H, N_EDGES),
        make_random_EI_matched(H, N_EDGES),
        make_fly_inspired(H, N_EDGES),
    ]

    # Build diverse input patterns
    inputs = []
    for i in range(N_INPUTS):
        v = np.zeros(H, dtype=np.float64)
        # Each input activates different neurons
        for j in range(4):
            v[(i * 4 + j) % H] = 5.0
        inputs.append(v)

    # ── TEST P1: Input separation ────────────────────────────────
    print(f"\n{'─'*60}")
    print("  P1: INPUT SEPARATION — do different inputs produce")
    print("      different outputs? (higher = better separation)")
    print(f"{'─'*60}")

    for adj, pol, name in topologies:
        histories = [run_dynamics(adj, pol, inp, ticks=TICKS) for inp in inputs]

        # Pairwise pattern distance
        dists = []
        for i in range(len(histories)):
            for j in range(i + 1, len(histories)):
                dists.append(pattern_distance(histories[i], histories[j]))

        # Count distinct output patterns
        final_patterns = [tuple(h[-1]['spikes']) for h in histories]
        n_distinct = len(set(final_patterns))

        # Count dead outputs (all zeros)
        n_dead = sum(1 for h in histories if h[-1]['n_active'] == 0)

        print(f"  {name:<22s}: mean_dist={np.mean(dists):5.1f}, "
              f"distinct={n_distinct}/{N_INPUTS}, "
              f"dead={n_dead}/{N_INPUTS}")

    # ── TEST P2: Selectivity ─────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  P2: SELECTIVITY — how many neurons survive the filtering?")
    print("      (lower active count = more selective = better filter)")
    print(f"{'─'*60}")

    for adj, pol, name in topologies:
        active_counts = []
        entropies = []
        for inp in inputs:
            h = run_dynamics(adj, pol, inp, ticks=TICKS)
            active_counts.append(h[-1]['n_active'])
            entropies.append(output_entropy(h))

        print(f"  {name:<22s}: mean_active={np.mean(active_counts):5.1f}/{H}, "
              f"entropy={np.mean(entropies):.2f}")

    # ── TEST P3: Energy decay ────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  P3: ENERGY DECAY — does total charge decrease over ticks?")
    print("      (prediction: most energy should die, not amplify)")
    print(f"{'─'*60}")

    for adj, pol, name in topologies:
        all_curves = []
        for inp in inputs:
            h = run_dynamics(adj, pol, inp, ticks=TICKS)
            all_curves.append(energy_curve(h))

        mean_curve = np.mean(all_curves, axis=0)
        peak = np.argmax(mean_curve)
        decay_ratio = mean_curve[-1] / (mean_curve[peak] + 1e-10)

        curve_str = " → ".join(f"{e:.1f}" for e in mean_curve)
        print(f"  {name:<22s}: {curve_str}")
        print(f"  {'':22s}  peak@tick{peak}, final/peak={decay_ratio:.2f}")

    # ── TEST P4: Determinism ─────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  P4: DETERMINISM — same input always produces same output?")
    print("      (must be 100% for resonator to be predictive)")
    print(f"{'─'*60}")

    for adj, pol, name in topologies:
        all_consistent = True
        for inp in inputs:
            if not pattern_consistency(adj, pol, inp):
                all_consistent = False
                break
        print(f"  {name:<22s}: {'✓ 100% deterministic' if all_consistent else '✗ NOT deterministic'}")

    # ── TEST P5: Reciprocal contribution ─────────────────────────
    print(f"\n{'─'*60}")
    print("  P5: RECIPROCAL EDGE ABLATION — what happens if we")
    print("      remove reciprocal edges from fly-inspired topology?")
    print(f"{'─'*60}")

    fly_adj, fly_pol, _ = make_fly_inspired(H, N_EDGES)

    # Find reciprocal edges
    reciprocal_mask = (fly_adj > 0) & (fly_adj.T > 0)
    n_recip_edges = int(reciprocal_mask.sum())

    # Ablate: remove one direction of each reciprocal pair
    ablated_adj = fly_adj.copy()
    for i in range(H):
        for j in range(i + 1, H):
            if reciprocal_mask[i, j]:
                ablated_adj[j, i] = 0.0  # remove backward direction

    n_ablated = int(np.sum(fly_adj) - np.sum(ablated_adj))

    configs = [
        (fly_adj, fly_pol, "fly_with_reciprocal"),
        (ablated_adj, fly_pol, "fly_NO_reciprocal"),
    ]

    for adj, pol, name in configs:
        histories = [run_dynamics(adj, pol, inp, ticks=TICKS) for inp in inputs]
        dists = []
        for i in range(len(histories)):
            for j in range(i + 1, len(histories)):
                dists.append(pattern_distance(histories[i], histories[j]))
        final_patterns = [tuple(h[-1]['spikes']) for h in histories]
        n_distinct = len(set(final_patterns))
        n_dead = sum(1 for h in histories if h[-1]['n_active'] == 0)
        active = [h[-1]['n_active'] for h in histories]

        print(f"  {name:<25s}: edges={int(np.sum(adj))}, "
              f"separation={np.mean(dists):.1f}, "
              f"distinct={n_distinct}/{N_INPUTS}, "
              f"dead={n_dead}, "
              f"active={np.mean(active):.1f}")

    print(f"  (Ablated {n_ablated} reciprocal edges out of {int(np.sum(fly_adj))})")

    # ── TOPOLOGY STATS ───────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  TOPOLOGY STATS")
    print(f"{'─'*60}")

    for adj, pol, name in topologies:
        n_edges = int(np.sum(adj))
        n_recip = int(np.sum((adj > 0) & (adj.T > 0))) // 2
        n_inhib = int(np.sum(pol < 0))
        # Clustering
        ccs = []
        for n in range(H):
            nbs = np.where(adj[n] > 0)[0]
            if len(nbs) < 2:
                continue
            tri = sum(1 for i, a in enumerate(nbs) for b in nbs[i+1:]
                      if adj[a, b] > 0 or adj[b, a] > 0)
            ccs.append(tri / (len(nbs) * (len(nbs) - 1) / 2))
        cc = np.mean(ccs) if ccs else 0

        print(f"  {name:<22s}: edges={n_edges}, reciprocal={n_recip}, "
              f"inhib={n_inhib}/{H}, clustering={cc:.3f}")

    # ── FINAL VERDICT ────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  VERDICT")
    print(f"{'='*70}")
    print("""
  The test checks whether fly-brain-inspired topology produces
  qualitatively different signal processing than random or
  feedforward alternatives. Key predictions:

  P1 (Separation): Does fly-inspired separate inputs better?
  P2 (Selectivity): Does it activate fewer, more specific neurons?
  P3 (Energy decay): Does energy dissipate (filtering)?
  P4 (Determinism): Is the output predictable from input+topology?
  P5 (Reciprocal): Does removing reciprocal edges degrade performance?
""")
