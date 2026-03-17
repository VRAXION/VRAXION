# RNG Performance Analysis Results

## 1. RNG Tier Benchmark (V=64, 5 seeds)

**Finding: Sharp "knee" between Tier 3 (xorshift32) and Tier 4 (LCG)**

| Tier | RNG | Score | Status |
|------|-----|-------|--------|
| 1 | PCG64 | 57.8% | ← gold standard |
| 2 | MT19937 | 57.3% | ← statistically same |
| 3 | xorshift32 | 57.3% | ← cheapest that works |
| **KNEE** | | **-29% DROP** | |
| 4 | LCG-32 | 28.2% | ← BROKEN |
| 5 | LCG-16 | 28.8% | ← BROKEN |
| 6 | Counter | 12.2% | ← trash |
| 7 | Alternating | 6.2% | ← dead |

**Why LCG breaks**: Sequential correlation. Consecutive LCG outputs are linearly
dependent (`x_{n+1} = a*x_n + c mod m`), so when you pick edge A then edge B,
they're correlated. Mutations can't explore the search space properly.

**Recommendation**: xorshift32 — 4 lines of code, 32-bit state, and performs
identically to PCG64/MT19937 for this workload.

## 2. Vocabulary Scaling (Dense Forward)

| V | Score | ms/att | Density | Cache pressure |
|---|-------|--------|---------|----------------|
| 32 | 59.1% | 0.83 | 21.6% | L2 (144KB) |
| 64 | 48.6% | 2.60 | 12.4% | L2 (576KB) |
| 96 | 43.6% | 5.43 | 9.2% | L2/L3 boundary |
| 128 | 32.3% | 8.87 | 6.9% | L3 (2.3MB) |

**Bottleneck**: Not the edges — the activation matrices (acts, charges, raw).
These grow as V² × NV_RATIO² and evict edges from cache.

Per-edge cost = V multiplies (full row), so `total cost = conns × V × ticks`.

## 3. Sparse Forward + Connection Cap (cap=5000)

| V | N | Conns | Density | Score | ms/att | Mode |
|---|---|-------|---------|-------|--------|------|
| 32 | 96 | 1993 | 21.6% | 59.1% | 0.91 | dense |
| 64 | 192 | 4569 | 12.4% | 48.6% | 4.54 | dense |
| 96 | 288 | 5000 | 6.0% | 30.5% | 4.43 | sparse |
| 128 | 384 | 5000 | 3.4% | 14.7% | 7.55 | sparse |
| 192 | 576 | 5000* | 1.6% | 8.7% | 14.21 | sparse |

*\* Cap was NOT enforced at init — V=192 had 5405 conns (init 4% = ~13K > cap).*

**Scaling**: ms/att grows ~V² with sparse (vs ~V³ dense). The sparse CSR matmul
only touches nonzero edges, so cost = `conns × V` which is linear in V when
conns are capped.

**Bug found**: Init density (4%) exceeds cap for V≥128:
- V=128: N=384, init conns ≈ 5900 > 5000 cap
- V=192: N=576, init conns ≈ 13000 > 5000 cap

**Fix**: `enforce_conn_cap()` prunes random edges at init to respect the cap.

## 4. Key Conclusions

1. **RNG quality doesn't matter above xorshift32** — the mutation loop only needs
   uncorrelated successive samples, not cryptographic randomness.

2. **Dense forward is the scaling bottleneck** — V² memory growth evicts working
   set from L2 cache. Sparse forward with CSR keeps cost linear in edges.

3. **Fixed connection cap doesn't scale** — 5000 conns at V=192 gives 1.6% density,
   which is too sparse to learn anything useful. The cap should scale with V.

4. **Next step**: Adaptive cap (`cap = k × V`) so density stays roughly constant
   as V grows, while still benefiting from sparse forward.

## Reproducing

```bash
# RNG tier benchmark
python v4.2/tests/rng_tier_benchmark.py --seeds 5 --vocab 64

# Sparse scaling benchmark
python v4.2/tests/sparse_scaling_benchmark.py --vocab-list 32,64,96,128,192 --cap 5000
```
