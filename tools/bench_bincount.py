import numpy as np
import time

H = 1024
DENSITY = 0.05
ITERS = 500

rng = np.random.default_rng(42)
mask = (rng.random((H, H)) < DENSITY).astype(np.int8)
rows, cols = np.where(mask != 0)
act = rng.random(H).astype(np.float32)

def run_coo_add_at():
    raw = np.zeros(H, dtype=np.float32)
    np.add.at(raw, cols, act[rows])
    return raw

def run_coo_bincount():
    # This is equivalent to act @ mask in COO format
    return np.bincount(cols, weights=act[rows], minlength=H)

# Warmup
run_coo_add_at()
run_coo_bincount()

t0 = time.perf_counter()
for _ in range(ITERS): run_coo_add_at()
t1 = time.perf_counter()
print(f"NumPy add.at: { (t1-t0)*1000/ITERS :.4f} ms")

t0 = time.perf_counter()
for _ in range(ITERS): run_coo_bincount()
t1 = time.perf_counter()
print(f"NumPy bincount: { (t1-t0)*1000/ITERS :.4f} ms")
print(f"Speedup: { ( (t1-t0)*1000/ITERS ) / ( (t1-t0)*1000/ITERS ) if False else 0 }x")
