"""Quick microbenchmark: binary sparse cache (add only) vs legacy (multiply)."""
import sys, os, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model.graph import SelfWiringGraph

H = 1024
DENSITY = 0.04
ITERS = 500

np.random.seed(42)
mask = np.zeros((H, H), dtype=np.int8)
r = np.random.rand(H, H)
mask[r < DENSITY] = 1  # binary: only {0, 1}
np.fill_diagonal(mask, 0)

edges = int(np.count_nonzero(mask))
print(f"H={H}, edges={edges}, density={edges/(H*H)*100:.1f}%\n")

# Build both cache formats
# Binary (2-tuple: rows, cols)
bin_cache = SelfWiringGraph.build_sparse_cache(mask, edge_magnitude=1.0)
assert len(bin_cache) == 2, f"Expected 2-tuple, got {len(bin_cache)}"

# Legacy (3-tuple: rows, cols, vals)
rows, cols = np.where(mask != 0)
vals = mask[rows, cols].astype(np.float32)
legacy_cache = (rows.astype(np.intp), cols.astype(np.intp), vals)

act = np.random.randn(H).astype(np.float32)

# Warmup
for _ in range(50):
    SelfWiringGraph._sparse_mul_1d_from_cache(H, act, bin_cache)
    SelfWiringGraph._sparse_mul_1d_from_cache(H, act, legacy_cache)

# Verify identical results
out_bin = SelfWiringGraph._sparse_mul_1d_from_cache(H, act, bin_cache)
out_legacy = SelfWiringGraph._sparse_mul_1d_from_cache(H, act, legacy_cache)
max_diff = np.max(np.abs(out_bin - out_legacy))
print(f"Max diff binary vs legacy: {max_diff:.2e}")
assert max_diff < 1e-5, f"Results diverge! max_diff={max_diff}"

# Benchmark 1D
t0 = time.perf_counter()
for _ in range(ITERS):
    SelfWiringGraph._sparse_mul_1d_from_cache(H, act, bin_cache)
t_bin_1d = time.perf_counter() - t0

t0 = time.perf_counter()
for _ in range(ITERS):
    SelfWiringGraph._sparse_mul_1d_from_cache(H, act, legacy_cache)
t_legacy_1d = time.perf_counter() - t0

print(f"\n1D sparse mul ({ITERS} iters):")
print(f"  binary:   {t_bin_1d*1000:.1f} ms")
print(f"  legacy:   {t_legacy_1d*1000:.1f} ms")
print(f"  speedup:  {t_legacy_1d/t_bin_1d:.2f}x")

# Benchmark 2D (batch)
BATCH = 32
acts = np.random.randn(BATCH, H).astype(np.float32)

# Verify 2D
out_bin_2d = SelfWiringGraph._sparse_mul_2d_from_cache(H, acts, bin_cache)
out_legacy_2d = SelfWiringGraph._sparse_mul_2d_from_cache(H, acts, legacy_cache)
max_diff_2d = np.max(np.abs(out_bin_2d - out_legacy_2d))
print(f"\nMax diff 2D binary vs legacy: {max_diff_2d:.2e}")

t0 = time.perf_counter()
for _ in range(ITERS):
    SelfWiringGraph._sparse_mul_2d_from_cache(H, acts, bin_cache)
t_bin_2d = time.perf_counter() - t0

t0 = time.perf_counter()
for _ in range(ITERS):
    SelfWiringGraph._sparse_mul_2d_from_cache(H, acts, legacy_cache)
t_legacy_2d = time.perf_counter() - t0

print(f"\n2D sparse mul batch={BATCH} ({ITERS} iters):")
print(f"  binary:   {t_bin_2d*1000:.1f} ms")
print(f"  legacy:   {t_legacy_2d*1000:.1f} ms")
print(f"  speedup:  {t_legacy_2d/t_bin_2d:.2f}x")
