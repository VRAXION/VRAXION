"""Find the density knee: where does sparse matmul spill out of cache?

Measures forward_batch speed at different densities.
Maps CSR working set size to L1/L2/L3 cache boundaries.

Ryzen 9 3900X: L1=32KB/core, L2=512KB/core, L3=64MB shared
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import time
from scipy import sparse

# Cache sizes (Ryzen 9 3900X)
L1_KB = 32
L2_KB = 512
L3_KB = 64 * 1024  # 64MB shared

CONFIGS = [
    ("V64_N192",  64, 192),
    ("V128_N384", 128, 384),
    ("V256_N768", 256, 768),  # future scaling
]

DENSITIES = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

TICKS = 8
REPEATS = 20
GAIN = 2.0
THRESHOLD = 0.5
SELF_CONN = 0.1
CLIP_FACTOR = 2.0


def measure_forward(V, N, density, ticks=TICKS, repeats=REPEATS):
    """Measure forward_batch time at given density."""
    np.random.seed(42)
    r = np.random.rand(N, N)
    mask = np.zeros((N, N), dtype=np.int8)
    mask[r < density / 2] = -1
    mask[r > 1 - density / 2] = 1
    np.fill_diagonal(mask, 0)

    Weff_csr = sparse.csr_matrix(mask.astype(np.float32) * GAIN)
    clip_bound = THRESHOLD * CLIP_FACTOR
    out_start = N - V if N >= 2 * V else 0

    # Warmup
    for _ in range(3):
        charges = np.zeros((V, N), dtype=np.float32)
        acts = np.zeros((V, N), dtype=np.float32)
        for t in range(ticks):
            if t == 0: acts[:, :V] = np.eye(V, dtype=np.float32)
            raw = np.asarray(acts @ Weff_csr) + acts * SELF_CONN
            charges += raw * 0.3
            charges *= 0.85
            acts = np.maximum(charges - THRESHOLD, 0)
            charges = np.clip(charges, -clip_bound, clip_bound)

    # Timed runs
    times = []
    for _ in range(repeats):
        charges = np.zeros((V, N), dtype=np.float32)
        acts = np.zeros((V, N), dtype=np.float32)
        t0 = time.perf_counter()
        for t in range(ticks):
            if t == 0: acts[:, :V] = np.eye(V, dtype=np.float32)
            raw = np.asarray(acts @ Weff_csr) + acts * SELF_CONN
            charges += raw * 0.3
            charges *= 0.85
            acts = np.maximum(charges - THRESHOLD, 0)
            charges = np.clip(charges, -clip_bound, clip_bound)
        times.append(time.perf_counter() - t0)

    return np.median(times) * 1000  # ms


def calc_working_set(V, N, density):
    """Calculate working set size in KB for sparse matmul."""
    nnz = int(density * N * N)

    # CSR storage: values(float32) + col_indices(int32) + row_ptr(int32)
    csr_kb = (nnz * 4 + nnz * 4 + (N + 1) * 4) / 1024

    # Activations: acts(V,N) float32
    acts_kb = V * N * 4 / 1024

    # Result: raw(V,N) float32
    result_kb = V * N * 4 / 1024

    # Charges: (V,N) float32
    charges_kb = V * N * 4 / 1024

    # Total working set for one matmul tick
    total_kb = csr_kb + acts_kb + result_kb + charges_kb

    return nnz, csr_kb, acts_kb, total_kb


def cache_level(kb):
    if kb <= L1_KB: return "L1"
    elif kb <= L2_KB: return "L2"
    elif kb <= L3_KB: return "L3"
    else: return "RAM"


def main():
    print(f"DENSITY vs CACHE KNEE BENCHMARK", flush=True)
    print(f"CPU: Ryzen 9 3900X | L1=32KB | L2=512KB | L3=64MB", flush=True)
    print(f"Ticks={TICKS}, Repeats={REPEATS}", flush=True)
    print("=" * 110, flush=True)

    for config_name, V, N in CONFIGS:
        print(f"\n--- {config_name} (V={V}, N={N}) ---", flush=True)
        print(f"  {'density':>7s} {'nnz':>7s} {'CSR_KB':>8s} {'acts_KB':>8s} {'total_KB':>9s} "
              f"{'cache':>5s} {'time_ms':>8s} {'rel':>6s}", flush=True)

        base_time = None
        results = []

        for d in DENSITIES:
            nnz, csr_kb, acts_kb, total_kb = calc_working_set(V, N, d)

            # Skip if matrix too large for reasonable test
            if total_kb > L3_KB * 2:
                print(f"  {d:7.2f} {nnz:7d} {csr_kb:8.1f} {acts_kb:8.1f} {total_kb:9.1f} "
                      f"{'RAM':>5s} {'skip':>8s}", flush=True)
                continue

            ms = measure_forward(V, N, d)
            cl = cache_level(total_kb)

            if base_time is None:
                base_time = ms

            rel = ms / base_time
            results.append((d, nnz, total_kb, cl, ms, rel))

            print(f"  {d:7.2f} {nnz:7d} {csr_kb:8.1f} {acts_kb:8.1f} {total_kb:9.1f} "
                  f"{cl:>5s} {ms:8.2f} {rel:5.1f}x", flush=True)

        # Find knee: biggest speed jump between consecutive densities
        if len(results) >= 3:
            max_jump = 0
            knee_idx = 0
            for i in range(1, len(results)):
                jump = results[i][5] / results[i-1][5]
                if jump > max_jump:
                    max_jump = jump
                    knee_idx = i

            knee = results[knee_idx]
            prev = results[knee_idx - 1] if knee_idx > 0 else results[0]
            print(f"\n  KNEE: density {prev[0]:.2f} -> {knee[0]:.2f} "
                  f"({prev[4]:.1f}ms -> {knee[4]:.1f}ms, {max_jump:.1f}x jump) "
                  f"cache: {prev[3]} -> {knee[3]}", flush=True)
            print(f"  RECOMMENDED MAX DENSITY: {prev[0]:.2f} "
                  f"({prev[1]} connections, {prev[2]:.0f}KB working set, fits {prev[3]})", flush=True)

    # Also measure DENSE matmul for comparison
    print(f"\n--- DENSE MATMUL COMPARISON ---", flush=True)
    for config_name, V, N in CONFIGS:
        np.random.seed(42)
        mask = np.zeros((N, N), dtype=np.int8)
        r = np.random.rand(N, N)
        mask[r < 0.06 / 2] = -1
        mask[r > 1 - 0.06 / 2] = 1
        Weff_dense = mask.astype(np.float32) * GAIN
        clip_bound = THRESHOLD * CLIP_FACTOR

        times = []
        for _ in range(REPEATS):
            charges = np.zeros((V, N), dtype=np.float32)
            acts = np.zeros((V, N), dtype=np.float32)
            t0 = time.perf_counter()
            for t in range(TICKS):
                if t == 0: acts[:, :V] = np.eye(V, dtype=np.float32)
                raw = acts @ Weff_dense + acts * SELF_CONN
                charges += raw * 0.3
                charges *= 0.85
                acts = np.maximum(charges - THRESHOLD, 0)
                charges = np.clip(charges, -clip_bound, clip_bound)
            times.append(time.perf_counter() - t0)

        ms = np.median(times) * 1000
        dense_kb = N * N * 4 / 1024
        print(f"  {config_name:12s} dense={ms:.2f}ms  matrix={dense_kb:.0f}KB", flush=True)

    print(f"\n{'='*110}", flush=True)


if __name__ == '__main__':
    main()
