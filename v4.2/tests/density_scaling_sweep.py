"""
Density Scaling Law Sweep
=========================
Measure converged density across vocab sizes V=8..1024+
to find the power law and extrapolate to V=32K/50K.

For each V: train to convergence, prune to minimum, measure final density.
All configs run in parallel across all available cores.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from model.graph import SelfWiringGraph, train

# Vocab sizes to test — logarithmic spacing
VOCAB_SIZES = [8, 16, 32, 64, 96, 128, 192, 256, 384, 512, 640, 800, 1024]

# Seeds per config
SEEDS = [42, 77, 123, 256, 999]

# Budget scales with V (bigger networks need more attempts)
def budget_for(V):
    if V <= 32:
        return 6000
    elif V <= 128:
        return 8000
    elif V <= 256:
        return 12000
    elif V <= 512:
        return 16000
    elif V <= 800:
        return 20000
    else:
        return 25000


def prune_minimal(net, targets, V, ticks=8, tolerance=0.005):
    """Prune edges one-by-one until accuracy drops. Returns final edge count."""
    def evaluate():
        logits = net.forward_batch(ticks)
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
        acc = (np.argmax(probs, axis=1)[:V] == targets[:V]).mean()
        tp = probs[np.arange(V), targets[:V]].mean()
        return 0.5 * acc + 0.5 * tp

    baseline = evaluate()
    threshold = baseline - tolerance
    removed = 0

    # Sort edges by absolute mask value (weakest first = smallest magnitude)
    edges_scored = []
    for r, c in net.alive:
        # Use absolute charge flow as importance proxy
        edges_scored.append((abs(float(net.mask[r, c])), r, c))

    # For randomized importance: try removing in random order but batch
    # For speed: remove in batches of increasing size
    batch = max(1, len(net.alive) // 100)  # 1% at a time

    while net.alive:
        # Pick weakest batch
        candidates = []
        indices_to_remove = random.sample(range(len(net.alive)),
                                          min(batch, len(net.alive)))
        saved = []
        for idx in sorted(indices_to_remove, reverse=True):
            r, c = net.alive[idx]
            saved.append((r, c, net.mask[r, c]))
            net.mask[r, c] = 0
            net.alive_set.discard((r, c))

        net.alive = list(net.alive_set)
        new_score = evaluate()

        if new_score >= threshold:
            removed += len(saved)
        else:
            # Restore
            for r, c, val in saved:
                net.mask[r, c] = val
                net.alive_set.add((r, c))
            net.alive = list(net.alive_set)

            if batch == 1:
                break
            # Reduce batch size and retry
            batch = max(1, batch // 2)

    return len(net.alive), removed


def run_one(V, seed):
    """Train + prune one config. Returns stats dict."""
    t0 = time.time()
    np.random.seed(seed)
    random.seed(seed)

    net = SelfWiringGraph(V)
    N = net.N
    targets = np.random.permutation(V)
    max_N2 = N * (N - 1)  # max possible edges

    # Train
    budget = budget_for(V)
    score = train(net, targets, V, max_attempts=budget, ticks=8, verbose=False)

    # Stats before prune
    edges_trained = net.count_connections()
    density_trained = edges_trained / max_N2

    # Prune
    final_edges, pruned = prune_minimal(net, targets, V)
    density_pruned = final_edges / max_N2

    elapsed = time.time() - t0

    result = {
        'V': V,
        'N': N,
        'seed': seed,
        'score': score,
        'budget': budget,
        'edges_trained': edges_trained,
        'density_trained': density_trained,
        'edges_pruned': final_edges,
        'density_pruned': density_pruned,
        'pruned_count': pruned,
        'compression': pruned / edges_trained if edges_trained > 0 else 0,
        'max_edges': max_N2,
        'elapsed': elapsed,
    }
    return result


def fit_power_law(Vs, densities):
    """Fit density = a * V^b using log-linear regression."""
    valid = [(v, d) for v, d in zip(Vs, densities) if d > 0]
    if len(valid) < 2:
        return None, None, None
    log_v = np.log(np.array([v for v, _ in valid]))
    log_d = np.log(np.array([d for _, d in valid]))
    # Linear fit: log(d) = log(a) + b * log(V)
    coeffs = np.polyfit(log_v, log_d, 1)
    b = coeffs[0]
    a = np.exp(coeffs[1])
    # R²
    predicted = a * np.array([v for v, _ in valid]) ** b
    actual = np.array([d for _, d in valid])
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return a, b, r2


def main():
    ncpu = multiprocessing.cpu_count()
    total_jobs = len(VOCAB_SIZES) * len(SEEDS)

    print(f"DENSITY SCALING LAW SWEEP")
    print(f"  Vocab sizes: {VOCAB_SIZES}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Total jobs: {total_jobs}")
    print(f"  CPU cores: {ncpu}")
    print(f"  Workers: {min(ncpu, total_jobs)}")
    print("=" * 100)

    jobs = [(V, seed) for V in VOCAB_SIZES for seed in SEEDS]
    results = []
    completed = 0

    with ProcessPoolExecutor(max_workers=min(ncpu, total_jobs)) as pool:
        futures = {pool.submit(run_one, V, seed): (V, seed) for V, seed in jobs}
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            completed += 1
            print(f"  [{completed:3d}/{total_jobs}] V={r['V']:4d} seed={r['seed']:3d} "
                  f"score={r['score']*100:5.1f}% "
                  f"edges: {r['edges_trained']:5d}→{r['edges_pruned']:5d} "
                  f"density: {r['density_trained']*100:5.2f}%→{r['density_pruned']*100:5.2f}% "
                  f"compression={r['compression']*100:4.1f}% "
                  f"({r['elapsed']:.1f}s)", flush=True)

    # Aggregate by V
    print("\n" + "=" * 100)
    print("AGGREGATED RESULTS (mean ± std across seeds)")
    print(f"{'V':>5s} {'N':>5s} {'Score':>8s} {'Edges_T':>8s} {'Edges_P':>8s} "
          f"{'Dens_T%':>8s} {'Dens_P%':>8s} {'Compress%':>10s} {'Max_E':>8s} {'Time':>6s}")
    print("-" * 100)

    agg_V = []
    agg_density_trained = []
    agg_density_pruned = []
    agg_edges_pruned = []

    for V in VOCAB_SIZES:
        runs = [r for r in results if r['V'] == V]
        if not runs:
            continue

        scores = [r['score'] for r in runs]
        et = [r['edges_trained'] for r in runs]
        ep = [r['edges_pruned'] for r in runs]
        dt = [r['density_trained'] for r in runs]
        dp = [r['density_pruned'] for r in runs]
        comp = [r['compression'] for r in runs]
        times = [r['elapsed'] for r in runs]

        print(f"{V:5d} {runs[0]['N']:5d} "
              f"{np.mean(scores)*100:6.1f}±{np.std(scores)*100:3.1f} "
              f"{np.mean(et):7.0f}±{np.std(et):4.0f} "
              f"{np.mean(ep):7.0f}±{np.std(ep):4.0f} "
              f"{np.mean(dt)*100:7.3f}% "
              f"{np.mean(dp)*100:7.3f}% "
              f"{np.mean(comp)*100:8.1f}% "
              f"{runs[0]['max_edges']:8d} "
              f"{np.mean(times):5.1f}s")

        agg_V.append(V)
        agg_density_trained.append(np.mean(dt))
        agg_density_pruned.append(np.mean(dp))
        agg_edges_pruned.append(np.mean(ep))

    # Power law fit
    print("\n" + "=" * 100)
    print("POWER LAW FIT:  density = a × V^b")

    a_t, b_t, r2_t = fit_power_law(agg_V, agg_density_trained)
    a_p, b_p, r2_p = fit_power_law(agg_V, agg_density_pruned)

    if a_t is not None:
        print(f"  Trained:  density = {a_t:.6f} × V^{b_t:.4f}  (R²={r2_t:.4f})")
    if a_p is not None:
        print(f"  Pruned:   density = {a_p:.6f} × V^{b_p:.4f}  (R²={r2_p:.4f})")

    # Also fit edges = a * V^b
    a_e, b_e, r2_e = fit_power_law(agg_V, agg_edges_pruned)
    if a_e is not None:
        print(f"  Edges(P): edges   = {a_e:.4f} × V^{b_e:.4f}  (R²={r2_e:.4f})")

    # Extrapolate
    print("\n" + "=" * 100)
    print("EXTRAPOLATION TO LARGER VOCAB SIZES")
    print(f"{'V':>8s} {'N':>8s} {'Density_P%':>12s} {'Edges_P':>12s} "
          f"{'Bits (1.58b)':>14s} {'MB':>8s} {'State_MB':>10s} {'Total_MB':>10s}")
    print("-" * 100)

    extrap_Vs = [1024, 2048, 4096, 8192, 16384, 32000, 50000]
    for V in extrap_Vs:
        N = V * 3  # NV_RATIO = 3
        max_e = N * (N - 1)
        if a_p is not None and a_e is not None:
            pred_density = a_p * V ** b_p
            pred_edges = a_e * V ** b_e
            bits = pred_edges * 1.58
            mb_weights = bits / 8 / 1024 / 1024
            mb_state = N * 2 * 4 / 1024 / 1024  # state + charge, float32
            total_mb = mb_weights + mb_state
            print(f"{V:8d} {N:8d} {pred_density*100:11.5f}% {pred_edges:11.0f} "
                  f"{bits:13.0f} {mb_weights:7.2f} {mb_state:9.2f} {total_mb:9.2f}")

    # Edge scaling law summary
    if a_e is not None:
        print(f"\n  KEY FINDING: edges ∝ V^{b_e:.2f}")
        if b_e < 1.5:
            print(f"  → SUB-QUADRATIC scaling! Network is sparse-efficient.")
        elif b_e < 2.0:
            print(f"  → Between linear and quadratic. Moderate scaling.")
        else:
            print(f"  → Quadratic or worse. Density doesn't help at scale.")

        print(f"\n  For V=32K English model:")
        pred_e = a_e * 32000 ** b_e
        pred_mb = pred_e * 1.58 / 8 / 1024 / 1024
        state_mb = 32000 * 3 * 2 * 4 / 1024 / 1024
        print(f"    Edges:      {pred_e:,.0f}")
        print(f"    Weight mem: {pred_mb:,.1f} MB")
        print(f"    State mem:  {state_mb:,.1f} MB")
        print(f"    TOTAL:      {pred_mb + state_mb:,.1f} MB")
        print(f"\n  vs GPT-2 Small: ~248 MB weights + ~75 MB KV-cache = ~323 MB")
        print(f"  → SWG/GPT-2 ratio: {(pred_mb + state_mb) / 323:.1f}×")

    print("=" * 100)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
