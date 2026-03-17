"""
RNG Battle: standardized quality comparison.
Tests modeled after TestU01/PractRand with scoring.
Compares: xorshift32, MT19937, PCG32, Phi-v2d.
"""
import sys, os, struct, math, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

N = 1_000_000  # 1M samples — serious test

# ══════════════════════════════════════════════
#  RNG implementations
# ══════════════════════════════════════════════

class Xorshift32:
    def __init__(self, seed=42):
        self.state = seed if seed else 1
    def seed(self, s): self.state = s if s else 1
    def _next(self):
        x = self.state & 0xFFFFFFFF
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= x >> 17
        x ^= (x << 5) & 0xFFFFFFFF
        self.state = x
        return x
    def random(self):
        return self._next() / 4294967296.0
    def randint(self, a, b):
        return a + int(self.random() * (b - a + 1))
    def choice(self, seq):
        return seq[self.randint(0, len(seq)-1)]


class PCG32:
    """PCG-XSH-RR-64/32"""
    def __init__(self, seed=42):
        self.seed(seed)
    def seed(self, s):
        self.state = 0
        self.inc = (s << 1) | 1
        self._next()
        self.state = (self.state + s) & 0xFFFFFFFFFFFFFFFF
        self._next()
    def _next(self):
        old = self.state
        self.state = ((old * 6364136223846793005) + self.inc) & 0xFFFFFFFFFFFFFFFF
        xorshifted = (((old >> 18) ^ old) >> 27) & 0xFFFFFFFF
        rot = (old >> 59) & 0x1F
        return ((xorshifted >> rot) | (xorshifted << (32 - rot))) & 0xFFFFFFFF
    def random(self):
        return self._next() / 4294967296.0
    def randint(self, a, b):
        return a + int(self.random() * (b - a + 1))
    def choice(self, seq):
        return seq[self.randint(0, len(seq)-1)]


import random as _random
class MT19937Wrapper:
    def __init__(self, seed=42):
        self._r = _random.Random(seed)
    def seed(self, s): self._r.seed(s)
    def random(self): return self._r.random()
    def randint(self, a, b): return self._r.randint(a, b)
    def choice(self, seq): return self._r.choice(seq)

from model.phipi_rng import PhiPiRNG

# ══════════════════════════════════════════════
#  Test Suite
# ══════════════════════════════════════════════

def test_chi2_uniformity(samples, bins=1000):
    """Chi-square with 1000 bins (much harder than 100)."""
    counts = [0] * bins
    for s in samples:
        counts[min(int(s * bins), bins - 1)] += 1
    exp = len(samples) / bins
    chi2 = sum((c - exp)**2 / exp for c in counts)
    df = bins - 1
    # Normalized: |chi2 - df| / sqrt(2*df). Good RNG → ~0-2, bad → >>3
    z = abs(chi2 - df) / math.sqrt(2 * df)
    return z, f"chi2={chi2:.0f} df={df} z={z:.2f}"


def test_serial_pairs(samples, bins=100):
    """2D chi-square: consecutive pairs should be uniform in [0,1)^2."""
    grid = bins * bins
    counts = [0] * grid
    for i in range(0, len(samples) - 1, 2):
        bx = min(int(samples[i] * bins), bins - 1)
        by = min(int(samples[i+1] * bins), bins - 1)
        counts[bx * bins + by] += 1
    n = len(samples) // 2
    exp = n / grid
    chi2 = sum((c - exp)**2 / exp for c in counts)
    df = grid - 1
    z = abs(chi2 - df) / math.sqrt(2 * df)
    return z, f"chi2={chi2:.0f} df={df} z={z:.2f}"


def test_birthday_spacing(samples, n=4096, m=2**24):
    """Birthday spacings test: map n samples to m bins, count collisions.
    Too many or too few collisions = bad RNG."""
    # Take first n samples, map to [0, m)
    points = sorted(int(s * m) % m for s in samples[:n])
    spacings = sorted((points[i+1] - points[i]) % m for i in range(n - 1))
    # Count repeated spacings
    repeats = sum(1 for i in range(len(spacings)-1) if spacings[i] == spacings[i+1])
    # Expected repeats ≈ n^3 / (4*m) for n << sqrt(m)
    expected = n**3 / (4 * m)
    z = abs(repeats - expected) / max(math.sqrt(expected), 1)
    return z, f"repeats={repeats} expected={expected:.1f} z={z:.2f}"


def test_runs_up_down(samples):
    """Runs up/down test: count ascending/descending runs, check distribution."""
    n = len(samples)
    runs = 1
    for i in range(1, n):
        if (samples[i] > samples[i-1]) != (samples[i-1] > samples[i-2] if i > 1 else True):
            runs += 1
    expected = (2 * n - 1) / 3
    var = (16 * n - 29) / 90
    z = abs(runs - expected) / math.sqrt(var)
    return z, f"runs={runs} expected={expected:.0f} z={z:.2f}"


def test_gap(samples, alpha=0.0, beta=0.25, max_gap=80):
    """Gap test: gaps between hits in [alpha, beta)."""
    gaps = []
    current = 0
    for s in samples:
        if alpha <= s < beta:
            gaps.append(min(current, max_gap))
            current = 0
        else:
            current += 1
    if len(gaps) < 100:
        return 0, "too few gaps"
    p = beta - alpha
    observed = [0] * (max_gap + 1)
    for g in gaps: observed[g] += 1
    n = len(gaps)
    chi2 = 0
    for k in range(max_gap):
        exp = n * p * (1 - p)**k
        if exp > 1: chi2 += (observed[k] - exp)**2 / exp
    exp_last = n * (1 - p)**max_gap
    if exp_last > 1: chi2 += (observed[max_gap] - exp_last)**2 / exp_last
    df = min(max_gap, 60)
    z = abs(chi2 - df) / math.sqrt(2 * df)
    return z, f"chi2={chi2:.1f} df~{df} z={z:.2f}"


def test_permutation(samples, t=5):
    """Permutation test: consecutive t-tuples should appear in all t! orderings equally."""
    from itertools import permutations
    n_perms = math.factorial(t)
    counts = [0] * n_perms

    # Map each t-tuple to its permutation index
    all_perms = list(permutations(range(t)))
    perm_to_idx = {p: i for i, p in enumerate(all_perms)}

    n_tuples = len(samples) // t
    for i in range(n_tuples):
        chunk = samples[i*t:(i+1)*t]
        # Rank the elements
        ranked = sorted(range(t), key=lambda k: chunk[k])
        inv = [0] * t
        for rank, orig in enumerate(ranked):
            inv[orig] = rank
        counts[perm_to_idx[tuple(inv)]] += 1

    exp = n_tuples / n_perms
    chi2 = sum((c - exp)**2 / exp for c in counts)
    df = n_perms - 1
    z = abs(chi2 - df) / math.sqrt(2 * df)
    return z, f"chi2={chi2:.1f} df={df} z={z:.2f}"


def test_coupon_collector(samples, d=8):
    """Coupon collector: how many samples to see all d values?"""
    lengths = []
    seen = set()
    count = 0
    for s in samples:
        v = min(int(s * d), d - 1)
        seen.add(v)
        count += 1
        if len(seen) == d:
            lengths.append(count)
            seen = set()
            count = 0
    if len(lengths) < 50:
        return 0, "too few completions"
    # Expected length = d * H_d where H_d = harmonic number
    H_d = sum(1/k for k in range(1, d+1))
    expected_mean = d * H_d
    actual_mean = sum(lengths) / len(lengths)
    # Variance of coupon collector
    expected_var = d * d * sum(1/k/k for k in range(1, d+1))
    z = abs(actual_mean - expected_mean) / math.sqrt(expected_var / len(lengths))
    return z, f"mean_len={actual_mean:.1f} expected={expected_mean:.1f} z={z:.2f}"


def test_max_of_t(samples, t=8):
    """Maximum-of-t test: max of t consecutive values should follow Beta(t,1) = x^t CDF."""
    maxes = []
    for i in range(0, len(samples) - t + 1, t):
        maxes.append(max(samples[i:i+t]))
    # KS test against Beta(t,1) CDF: F(x) = x^t
    maxes.sort()
    n = len(maxes)
    ks = 0
    for i, x in enumerate(maxes):
        expected = (i + 1) / n
        actual = x ** t
        ks = max(ks, abs(actual - expected))
    z = ks * math.sqrt(n)
    return z, f"KS={ks:.6f} z={z:.2f}"


TESTS = [
    ("Chi2-1000bins", test_chi2_uniformity),
    ("Serial-pairs", test_serial_pairs),
    ("Birthday-spacing", test_birthday_spacing),
    ("Runs-up/down", test_runs_up_down),
    ("Gap-test", test_gap),
    ("Permutation-5", test_permutation),
    ("Coupon-collect", test_coupon_collector),
    ("Max-of-8", test_max_of_t),
]

# ══════════════════════════════════════════════
#  Scoring
# ══════════════════════════════════════════════

def score_z(z):
    """Convert z-score to 0-100 score. z<1 = perfect, z>5 = zero."""
    if z < 1.0: return 100
    if z < 1.5: return 90
    if z < 2.0: return 75
    if z < 2.5: return 60
    if z < 3.0: return 40
    if z < 4.0: return 20
    if z < 5.0: return 5
    return 0


def battle(name, rng, seed=42):
    rng.seed(seed)
    t0 = time.time()
    samples = [rng.random() for _ in range(N)]
    gen_time = time.time() - t0

    total_score = 0
    details = []
    for test_name, test_func in TESTS:
        z, info = test_func(samples)
        s = score_z(z)
        total_score += s
        details.append((test_name, s, z, info))

    max_score = len(TESTS) * 100
    pct = total_score / max_score * 100
    return pct, gen_time, details


# ══════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════

rngs = [
    ("xorshift32", Xorshift32(42)),
    ("MT19937", MT19937Wrapper(42)),
    ("PCG32", PCG32(42)),
    ("Phi-v2d", PhiPiRNG(42)),
]

print(f"RNG Battle — {N:,} samples, 8 tests, seed=42")
print(f"Scoring: z<1→100, z<1.5→90, z<2→75, z<2.5→60, z<3→40, z<4→20, z<5→5, z>5→0")
print("=" * 80)

results = {}
for name, rng in rngs:
    pct, gen_time, details = battle(name, rng)
    results[name] = pct
    print(f"\n  {name:12s}  Score: {pct:.0f}/100  ({gen_time:.2f}s)")
    for test_name, s, z, info in details:
        marker = "██" if s == 100 else ("▓▓" if s >= 75 else ("░░" if s >= 40 else "  "))
        print(f"    {marker} {test_name:18s} {s:3d}/100  z={z:.2f}  ({info})")

# Multi-seed average
print("\n" + "=" * 80)
print("Multi-seed average (seeds 0-9):")
print(f"{'RNG':>12s}", end="")
for s in range(10):
    print(f"  s={s:d}", end="")
print("   AVG")
print("-" * 90)

for name, rng in rngs:
    scores = []
    for seed in range(10):
        pct, _, _ = battle(name, rng, seed=seed)
        scores.append(pct)
    avg = sum(scores) / len(scores)
    print(f"{name:>12s}", end="")
    for s in scores:
        print(f"  {s:3.0f}", end="")
    print(f"  {avg:5.1f}")
