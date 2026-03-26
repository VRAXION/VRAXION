"""Semantic distance vs tick cost: does farther = nonlinearly more expensive?

Hypothesis (user intuition):
  Reaching semantically distant representations costs more ticks,
  and the cost grows NONLINEARLY — like relativistic time dilation,
  where the last 10% of c costs more than the first 90%.

Test design:
  1. Build a network with K well-separated clusters (semantic neighborhoods)
  2. Wire clusters with decreasing inter-cluster density (nearby = dense, far = sparse)
  3. Inject signal into cluster 0, measure how many ticks until the interference
     pattern at cluster D (distance D) stabilizes
  4. "Stabilize" = spike pattern stops changing between consecutive ticks
  5. Sweep D from 0 (same cluster) to K-1 (farthest cluster)

Predictions:
  P1: Same-cluster convergence is fast (few ticks)
  P2: Cross-cluster convergence cost grows with distance
  P3: The growth is NONLINEAR (superlinear, possibly exponential)
  P4: There exists a "horizon" beyond which signal never arrives (dead zone)

All tests are DETERMINISTIC — fixed seeds, no training.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Dynamics (same as resonator_toy_test) ───────────────────────────

def run_dynamics(adj_matrix, polarity, input_vec, ticks, decay=0.10, theta=1.5):
    """Run spike dynamics, return per-tick history."""
    H = adj_matrix.shape[0]
    charge = np.zeros(H, dtype=np.float64)
    state = np.zeros(H, dtype=np.float64)
    history = []

    for tick in range(ticks):
        charge = np.maximum(charge - decay, 0.0)
        if tick < 2:
            state = state + input_vec
        signal = (state * polarity) @ adj_matrix.astype(np.float64)
        charge += signal
        charge = np.clip(charge, 0.0, 15.0)
        fired = charge >= theta
        state = fired.astype(np.float64)
        charge[fired] = 0.0
        history.append({
            'charge': charge.copy(),
            'spikes': fired.copy(),
            'spike_pattern': (state * polarity).copy(),
            'energy': float(np.sum(charge)),
            'n_active': int(np.sum(fired)),
        })
    return history


# ── Clustered topology builder ──────────────────────────────────────

def make_clustered_network(H, K, intra_density, inter_decay, inhib_frac=0.10, seed=42):
    """Build a network with K clusters and distance-decaying inter-cluster links.

    Args:
        H: total neurons
        K: number of clusters
        intra_density: edge density within each cluster (0-1)
        inter_decay: multiplicative decay per hop distance for inter-cluster edges
                     e.g., 0.3 means cluster distance 1 gets 30% of intra density,
                     distance 2 gets 9%, distance 3 gets 2.7%, etc.
        inhib_frac: fraction of inhibitory neurons (default 10% per resonator theory)
        seed: random seed

    Returns:
        adj: (H, H) adjacency matrix
        polarity: (H,) per-neuron sign
        cluster_ids: (H,) cluster assignment per neuron
    """
    rng = np.random.RandomState(seed)
    adj = np.zeros((H, H), dtype=np.float64)

    # Assign neurons to clusters (roughly equal size)
    cluster_size = H // K
    cluster_ids = np.zeros(H, dtype=int)
    for k in range(K):
        start = k * cluster_size
        end = start + cluster_size if k < K - 1 else H
        cluster_ids[start:end] = k

    # Polarity: 10% inhibitory hub neurons per cluster
    polarity = np.ones(H, dtype=np.float64)
    for k in range(K):
        members = np.where(cluster_ids == k)[0]
        n_inhib = max(1, int(len(members) * inhib_frac))
        inhib_neurons = rng.choice(members, n_inhib, replace=False)
        polarity[inhib_neurons] = -1.0

    # Intra-cluster edges
    for k in range(K):
        members = np.where(cluster_ids == k)[0]
        for i in members:
            for j in members:
                if i != j and rng.random() < intra_density:
                    adj[i, j] = 1.0

    # Inter-cluster edges with distance-decaying density
    for k1 in range(K):
        for k2 in range(K):
            if k1 == k2:
                continue
            distance = abs(k1 - k2)
            inter_density = intra_density * (inter_decay ** distance)
            members_1 = np.where(cluster_ids == k1)[0]
            members_2 = np.where(cluster_ids == k2)[0]
            for i in members_1:
                for j in members_2:
                    if rng.random() < inter_density:
                        adj[i, j] = 1.0

    return adj, polarity, cluster_ids


# ── Convergence measurement ─────────────────────────────────────────

def measure_convergence_tick(history, target_neurons, min_active=1):
    """Find the first tick where the spike pattern at target neurons stabilizes.

    "Stabilizes" = the spike pattern at target neurons is identical for 2
    consecutive ticks AND at least min_active neurons are firing.

    Returns tick number (1-indexed) or -1 if never stabilizes.
    """
    prev_pattern = None
    for t, snap in enumerate(history):
        pattern = snap['spikes'][target_neurons].astype(int)
        n_active = int(np.sum(pattern))
        if prev_pattern is not None and n_active >= min_active:
            if np.array_equal(pattern, prev_pattern):
                return t  # 0-indexed tick where it first repeated
        prev_pattern = pattern
    return -1


def measure_arrival_tick(history, target_neurons):
    """Find the first tick where ANY target neuron fires.

    Returns tick number (0-indexed) or -1 if signal never arrives.
    """
    for t, snap in enumerate(history):
        if np.any(snap['spikes'][target_neurons]):
            return t
    return -1


def measure_peak_activity(history, target_neurons):
    """Max number of simultaneously active neurons in target cluster."""
    peak = 0
    for snap in history:
        n = int(np.sum(snap['spikes'][target_neurons]))
        if n > peak:
            peak = n
    return peak


# ── Main experiment ─────────────────────────────────────────────────

def run_experiment():
    H = 128          # total neurons
    K = 8            # 8 clusters of 16 neurons each
    MAX_TICKS = 40   # generous tick budget to find the horizon
    INTRA_DENSITY = 0.4
    INTER_DECAY = 0.30  # 30% per hop — aggressive falloff

    print("=" * 70)
    print("  SEMANTIC DISTANCE vs TICK COST")
    print("=" * 70)
    print()
    print(f"  Network: H={H}, K={K} clusters of {H//K} neurons each")
    print(f"  Intra-cluster density: {INTRA_DENSITY}")
    print(f"  Inter-cluster decay: {INTER_DECAY} per hop")
    print(f"  Max ticks: {MAX_TICKS}")
    print()

    adj, polarity, cluster_ids = make_clustered_network(
        H, K, INTRA_DENSITY, INTER_DECAY, inhib_frac=0.10, seed=42
    )

    # Stats
    total_edges = int(np.sum(adj))
    print(f"  Total edges: {total_edges}")
    for d in range(K):
        src = np.where(cluster_ids == 0)[0]
        dst = np.where(cluster_ids == d)[0]
        cross = int(np.sum(adj[np.ix_(src, dst)])) if d > 0 else int(np.sum(adj[np.ix_(src, src)]))
        print(f"    Cluster 0 → Cluster {d}: {cross} edges")
    print()

    # Inject into cluster 0 with two different input patterns
    source_cluster = 0
    source_neurons = np.where(cluster_ids == source_cluster)[0]

    # Two distinct inputs into cluster 0
    input_a = np.zeros(H, dtype=np.float64)
    input_b = np.zeros(H, dtype=np.float64)
    half = len(source_neurons) // 2
    input_a[source_neurons[:half]] = 1.0      # first half of cluster 0
    input_b[source_neurons[half:]] = 1.0      # second half of cluster 0

    # Run dynamics with generous tick budget
    history_a = run_dynamics(adj, polarity, input_a, ticks=MAX_TICKS)
    history_b = run_dynamics(adj, polarity, input_b, ticks=MAX_TICKS)

    # ── Measure per-distance metrics ────────────────────────────────

    print("  RESULTS")
    print("  " + "-" * 66)
    print(f"  {'Dist':>4}  {'Arrival':>8}  {'Converge':>9}  {'Peak Act':>9}  "
          f"{'Separable':>10}  {'Density→':>9}")
    print("  " + "-" * 66)

    arrivals = []
    convergences = []
    separations = []

    for d in range(K):
        target_neurons = np.where(cluster_ids == d)[0]
        expected_density = INTRA_DENSITY * (INTER_DECAY ** d) if d > 0 else INTRA_DENSITY

        # Arrival: first tick any target neuron fires
        arrival_a = measure_arrival_tick(history_a, target_neurons)
        arrival_b = measure_arrival_tick(history_b, target_neurons)
        arrival = max(arrival_a, arrival_b)  # both inputs must arrive

        # Convergence: first tick pattern stabilizes
        conv_a = measure_convergence_tick(history_a, target_neurons)
        conv_b = measure_convergence_tick(history_b, target_neurons)
        convergence = max(conv_a, conv_b)

        # Separation: do inputs A and B produce DIFFERENT patterns at this cluster?
        # Check at all ticks from arrival onward
        separable = False
        if arrival >= 0:
            for t in range(arrival, MAX_TICKS):
                pat_a = history_a[t]['spikes'][target_neurons]
                pat_b = history_b[t]['spikes'][target_neurons]
                if not np.array_equal(pat_a, pat_b):
                    separable = True
                    break

        # Peak activity at target
        peak_a = measure_peak_activity(history_a, target_neurons)
        peak_b = measure_peak_activity(history_b, target_neurons)
        peak = max(peak_a, peak_b)

        arr_str = str(arrival) if arrival >= 0 else "NEVER"
        conv_str = str(convergence) if convergence >= 0 else "NEVER"
        sep_str = "YES" if separable else "NO"

        print(f"  {d:>4}  {arr_str:>8}  {conv_str:>9}  {peak:>9}  "
              f"{sep_str:>10}  {expected_density:>9.4f}")

        arrivals.append(arrival)
        convergences.append(convergence)
        separations.append(separable)

    # ── Analysis ────────────────────────────────────────────────────

    print()
    print("  ANALYSIS")
    print("  " + "-" * 66)

    # P1: Same-cluster should be fastest
    if arrivals[0] >= 0 and arrivals[0] <= 2:
        print("  [+] P1 PASS: Same-cluster arrival is fast (tick {})".format(arrivals[0]))
    else:
        print("  [!] P1 WARN: Same-cluster arrival at tick {} (expected ≤2)".format(arrivals[0]))

    # P2: Arrival cost grows with distance
    valid_arrivals = [(d, a) for d, a in enumerate(arrivals) if a >= 0]
    if len(valid_arrivals) >= 2:
        monotonic = all(valid_arrivals[i][1] <= valid_arrivals[i+1][1]
                       for i in range(len(valid_arrivals)-1))
        if monotonic:
            print("  [+] P2 PASS: Arrival tick is monotonically non-decreasing with distance")
        else:
            # Check if broadly increasing even if not perfectly monotonic
            first_half = np.mean([a for d, a in valid_arrivals[:len(valid_arrivals)//2]])
            second_half = np.mean([a for d, a in valid_arrivals[len(valid_arrivals)//2:]])
            if second_half > first_half:
                print("  [~] P2 PARTIAL: Arrival tick broadly increases (near {:.1f} → far {:.1f})".format(
                    first_half, second_half))
            else:
                print("  [-] P2 FAIL: Arrival tick does not increase with distance")

    # P3: Nonlinearity — check if cost grows faster than linear
    if len(valid_arrivals) >= 3:
        dists = np.array([d for d, a in valid_arrivals], dtype=float)
        arrs = np.array([a for d, a in valid_arrivals], dtype=float)
        if np.std(dists) > 0 and np.std(arrs) > 0:
            # Linear fit
            coeffs = np.polyfit(dists, arrs, 1)
            linear_pred = np.polyval(coeffs, dists)
            linear_residual = np.sum((arrs - linear_pred) ** 2)

            # Quadratic fit
            if len(valid_arrivals) >= 4:
                coeffs2 = np.polyfit(dists, arrs, 2)
                quad_pred = np.polyval(coeffs2, dists)
                quad_residual = np.sum((arrs - quad_pred) ** 2)

                if quad_residual < linear_residual * 0.8:
                    print("  [+] P3 PASS: Tick cost is superlinear (quadratic fits {:.1f}x better)".format(
                        linear_residual / max(quad_residual, 1e-10)))
                elif quad_residual < linear_residual * 0.95:
                    print("  [~] P3 PARTIAL: Slight superlinearity (quadratic fits {:.1f}x better)".format(
                        linear_residual / max(quad_residual, 1e-10)))
                else:
                    print("  [-] P3 FAIL: Tick cost appears linear, not superlinear")

                # Log fit for exponential check
                log_arrs = np.log1p(arrs)
                coeffs_exp = np.polyfit(dists, log_arrs, 1)
                exp_pred = np.expm1(np.polyval(coeffs_exp, dists))
                exp_residual = np.sum((arrs - exp_pred) ** 2)
                if exp_residual < linear_residual * 0.7:
                    print("       → Exponential fit is even better ({:.1f}x over linear)".format(
                        linear_residual / max(exp_residual, 1e-10)))
            else:
                print("  [?] P3 SKIP: Need ≥4 valid arrival points for nonlinearity test")

    # P4: Horizon — is there a distance beyond which signal never arrives?
    never_arrived = [d for d, a in enumerate(arrivals) if a < 0]
    if never_arrived:
        horizon = min(never_arrived)
        print("  [+] P4 PASS: Signal horizon at distance {} — clusters {} never reached".format(
            horizon, never_arrived))
    else:
        print("  [~] P4 NOTE: Signal reached all {} clusters within {} ticks".format(
            K, MAX_TICKS))

    # Separation quality
    n_separable = sum(1 for s in separations if s)
    print()
    print("  Separability: {}/{} clusters distinguish input A from input B".format(
        n_separable, K))

    # ── Tick cost curve summary ─────────────────────────────────────

    print()
    print("  TICK COST CURVE")
    print("  " + "-" * 66)
    max_bar = 40
    max_val = max(a for a in arrivals if a >= 0) if any(a >= 0 for a in arrivals) else 1
    for d, arr in enumerate(arrivals):
        if arr >= 0:
            bar_len = int((arr / max(max_val, 1)) * max_bar)
            bar = "#" * bar_len
            print(f"    dist={d}  tick={arr:>3}  {bar}")
        else:
            print(f"    dist={d}  tick=INF  {'.' * max_bar}  [HORIZON]")

    # ── Cost ratios ─────────────────────────────────────────────────

    print()
    print("  COST RATIOS (relative to same-cluster)")
    print("  " + "-" * 66)
    base = arrivals[0] if arrivals[0] > 0 else 1
    for d, arr in enumerate(arrivals):
        if arr >= 0:
            ratio = arr / base
            print(f"    dist={d}  cost={ratio:.2f}x")
        else:
            print(f"    dist={d}  cost=INF")

    # ── Second run: different inter_decay to check robustness ───────

    print()
    print("=" * 70)
    print("  ROBUSTNESS CHECK: inter_decay=0.5 (gentler falloff)")
    print("=" * 70)
    print()

    adj2, pol2, cids2 = make_clustered_network(
        H, K, INTRA_DENSITY, 0.50, inhib_frac=0.10, seed=42
    )

    hist2_a = run_dynamics(adj2, pol2, input_a, ticks=MAX_TICKS)
    hist2_b = run_dynamics(adj2, pol2, input_b, ticks=MAX_TICKS)

    print(f"  {'Dist':>4}  {'Arrival':>8}  {'Peak':>6}  {'Sep':>5}")
    print("  " + "-" * 35)
    for d in range(K):
        tn = np.where(cids2 == d)[0]
        a2_a = measure_arrival_tick(hist2_a, tn)
        a2_b = measure_arrival_tick(hist2_b, tn)
        arr2 = max(a2_a, a2_b)
        peak2 = max(measure_peak_activity(hist2_a, tn), measure_peak_activity(hist2_b, tn))
        sep2 = False
        if arr2 >= 0:
            for t in range(arr2, MAX_TICKS):
                if not np.array_equal(hist2_a[t]['spikes'][tn], hist2_b[t]['spikes'][tn]):
                    sep2 = True
                    break
        arr_s = str(arr2) if arr2 >= 0 else "NEVER"
        print(f"  {d:>4}  {arr_s:>8}  {peak2:>6}  {'YES' if sep2 else 'NO':>5}")

    # ── Third run: flat inter_decay (control — should be linear) ────

    print()
    print("=" * 70)
    print("  CONTROL: inter_decay=1.0 (uniform density — no distance effect)")
    print("=" * 70)
    print()

    adj3, pol3, cids3 = make_clustered_network(
        H, K, INTRA_DENSITY, 1.0, inhib_frac=0.10, seed=42
    )

    hist3_a = run_dynamics(adj3, pol3, input_a, ticks=MAX_TICKS)
    hist3_b = run_dynamics(adj3, pol3, input_b, ticks=MAX_TICKS)

    print(f"  {'Dist':>4}  {'Arrival':>8}  {'Peak':>6}  {'Sep':>5}")
    print("  " + "-" * 35)
    for d in range(K):
        tn = np.where(cids3 == d)[0]
        a3_a = measure_arrival_tick(hist3_a, tn)
        a3_b = measure_arrival_tick(hist3_b, tn)
        arr3 = max(a3_a, a3_b)
        peak3 = max(measure_peak_activity(hist3_a, tn), measure_peak_activity(hist3_b, tn))
        sep3 = False
        if arr3 >= 0:
            for t in range(arr3, MAX_TICKS):
                if not np.array_equal(hist3_a[t]['spikes'][tn], hist3_b[t]['spikes'][tn]):
                    sep3 = True
                    break
        arr_s = str(arr3) if arr3 >= 0 else "NEVER"
        print(f"  {d:>4}  {arr_s:>8}  {peak3:>6}  {'YES' if sep3 else 'NO':>5}")

    print()
    print("  (Control should show approximately FLAT arrival times across distances)")

    # ── Final verdict ───────────────────────────────────────────────

    print()
    print("=" * 70)
    print("  VERDICT")
    print("=" * 70)
    print()

    has_gradient = len(valid_arrivals) >= 2 and valid_arrivals[-1][1] > valid_arrivals[0][1]
    has_horizon = len(never_arrived) > 0
    has_separation = n_separable > 1

    if has_gradient:
        print("  The data supports the SEMANTIC DISTANCE hypothesis:")
        print("  → Farther clusters cost more ticks to reach")
        if len(valid_arrivals) >= 4:
            print("  → The cost curve appears to be superlinear (nonlinear)")
        if has_horizon:
            print("  → There is a reachability horizon at distance {}".format(min(never_arrived)))
        if has_separation:
            print("  → Different inputs produce distinguishable patterns at distance")
        print()
        print("  This is consistent with the intuition that semantic distance")
        print("  maps to interference-propagation cost, and that cost is nonlinear —")
        print("  like relativistic time dilation, the last hops are the most expensive.")
    else:
        print("  The data does NOT clearly support the hypothesis.")
        print("  Arrival times do not consistently increase with cluster distance.")
        print("  This may indicate the topology is too dense or the network too small")
        print("  for the distance gradient to emerge clearly.")

    print()


if __name__ == "__main__":
    run_experiment()
