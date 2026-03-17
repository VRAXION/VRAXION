"""RNG quality comparison: MT19937 vs Phi-Pi RNG.
Tests: uniformity, serial correlation, chi-square, runs, bit entropy."""
import sys, os, time, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import random as stdlib_random
from model.phipi_rng import PhiPiRNG

N = 100_000  # samples per test


def collect_samples(rng_func, n):
    return [rng_func() for _ in range(n)]


def chi_square_uniformity(samples, bins=100):
    """Chi-square test for uniform distribution in [0,1)."""
    counts = [0] * bins
    for s in samples:
        b = min(int(s * bins), bins - 1)
        counts[b] += 1
    expected = len(samples) / bins
    chi2 = sum((c - expected) ** 2 / expected for c in counts)
    # For bins-1 degrees of freedom, critical value at p=0.05 is ~123.2 (99 df)
    return chi2, bins - 1


def serial_correlation(samples):
    """Lag-1 autocorrelation."""
    n = len(samples)
    mean = sum(samples) / n
    var = sum((s - mean) ** 2 for s in samples) / n
    if var == 0:
        return 1.0
    cov = sum((samples[i] - mean) * (samples[i + 1] - mean) for i in range(n - 1)) / (n - 1)
    return cov / var


def runs_test(samples):
    """Runs test: count runs above/below median."""
    median = sorted(samples)[len(samples) // 2]
    bits = [1 if s >= median else 0 for s in samples]
    runs = 1
    for i in range(1, len(bits)):
        if bits[i] != bits[i - 1]:
            runs += 1
    n1 = sum(bits)
    n0 = len(bits) - n1
    # Expected runs and std
    exp_runs = 1 + 2 * n0 * n1 / (n0 + n1)
    std_runs = math.sqrt(2 * n0 * n1 * (2 * n0 * n1 - n0 - n1) /
                         ((n0 + n1) ** 2 * (n0 + n1 - 1)))
    z = (runs - exp_runs) / std_runs if std_runs > 0 else 0
    return runs, exp_runs, z


def bit_entropy(samples, bits=8):
    """Estimate entropy of discretized samples."""
    buckets = [0] * (2 ** bits)
    for s in samples:
        b = min(int(s * (2 ** bits)), (2 ** bits) - 1)
        buckets[b] += 1
    n = len(samples)
    ent = 0
    for c in buckets:
        if c > 0:
            p = c / n
            ent -= p * math.log2(p)
    return ent, bits  # max entropy = bits


def gap_test(samples, alpha=0.0, beta=0.5, max_gap=50):
    """Gap test: distribution of gaps between samples in [alpha, beta)."""
    gaps = []
    current_gap = 0
    for s in samples:
        if alpha <= s < beta:
            gaps.append(min(current_gap, max_gap))
            current_gap = 0
        else:
            current_gap += 1
    # Expected: geometric distribution with p = beta - alpha
    p = beta - alpha
    if not gaps:
        return float('inf')
    observed = [0] * (max_gap + 1)
    for g in gaps:
        observed[g] += 1
    n = len(gaps)
    chi2 = 0
    for k in range(max_gap):
        expected = n * p * (1 - p) ** k
        if expected > 0:
            chi2 += (observed[k] - expected) ** 2 / expected
    # Last bucket: cumulative
    expected_last = n * (1 - p) ** max_gap
    if expected_last > 0:
        chi2 += (observed[max_gap] - expected_last) ** 2 / expected_last
    return chi2


def int_uniformity(rng_int_func, lo, hi, n):
    """Chi-square for integer uniformity."""
    r = hi - lo + 1
    counts = [0] * r
    for _ in range(n):
        v = rng_int_func(lo, hi)
        counts[v - lo] += 1
    expected = n / r
    chi2 = sum((c - expected) ** 2 / expected for c in counts)
    return chi2, r - 1


def test_rng(name, float_func, int_func, seed_func):
    print(f"\n{'=' * 50}")
    print(f"  {name}")
    print(f"{'=' * 50}")

    seed_func(42)
    t0 = time.time()
    samples = collect_samples(float_func, N)
    gen_time = time.time() - t0

    # Basic stats
    mean = sum(samples) / len(samples)
    var = sum((s - mean) ** 2 for s in samples) / len(samples)
    mn, mx = min(samples), max(samples)
    print(f"  Samples: {N:,} in {gen_time:.2f}s ({N/gen_time:,.0f}/s)")
    print(f"  Mean:    {mean:.6f}  (ideal: 0.500000)")
    print(f"  Var:     {var:.6f}  (ideal: 0.083333)")
    print(f"  Range:   [{mn:.6f}, {mx:.6f}]")

    # Chi-square uniformity
    chi2, df = chi_square_uniformity(samples)
    status = "PASS" if chi2 < df * 1.5 else ("WARN" if chi2 < df * 2 else "FAIL")
    print(f"  Chi2:    {chi2:.1f} / {df} df  [{status}]")

    # Serial correlation
    corr = serial_correlation(samples)
    status = "PASS" if abs(corr) < 0.01 else ("WARN" if abs(corr) < 0.03 else "FAIL")
    print(f"  Corr:    {corr:+.6f}  (ideal: 0.000) [{status}]")

    # Runs test
    runs, exp, z = runs_test(samples)
    status = "PASS" if abs(z) < 1.96 else ("WARN" if abs(z) < 2.58 else "FAIL")
    print(f"  Runs:    {runs} / {exp:.0f} expected  z={z:+.2f} [{status}]")

    # Entropy
    ent, max_ent = bit_entropy(samples)
    status = "PASS" if ent > max_ent * 0.99 else ("WARN" if ent > max_ent * 0.95 else "FAIL")
    print(f"  Entropy: {ent:.3f} / {max_ent:.3f} bits [{status}]")

    # Gap test
    gap_chi2 = gap_test(samples)
    status = "PASS" if gap_chi2 < 80 else ("WARN" if gap_chi2 < 120 else "FAIL")
    print(f"  Gap:     chi2={gap_chi2:.1f} [{status}]")

    # Integer uniformity [1,20] (like mutation code uses)
    seed_func(42)
    int_chi2, int_df = int_uniformity(int_func, 1, 20, N)
    status = "PASS" if int_chi2 < int_df * 1.5 else ("WARN" if int_chi2 < int_df * 2 else "FAIL")
    print(f"  Int[1,20]: chi2={int_chi2:.1f} / {int_df} df [{status}]")

    return gen_time


# --- MT19937 ---
test_rng(
    "MT19937 (stdlib random)",
    stdlib_random.random,
    stdlib_random.randint,
    stdlib_random.seed,
)

# --- Phi-Pi RNG ---
pp = PhiPiRNG(42)
test_rng(
    "Phi-Pi RNG",
    pp.random,
    pp.randint,
    pp.seed,
)

print()
