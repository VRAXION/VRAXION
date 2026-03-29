"""
Benchmark: C edge vs Python rollout_token
==========================================
Process N tokens, measure time for both.
"""
import sys, os, time, subprocess
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'model'))
from graph import SelfWiringGraph

H = 256
PHI = (1 + 5**0.5) / 2
IN_DIM = int(round(H / PHI))
OUT_DIM = int(round(H / PHI))
SDR_K = int(round(IN_DIM * 0.20))
TICKS = 8

# Build same network as C uses
np.random.seed(42)
mask = (np.random.rand(H,H) < 0.05).astype(bool); np.fill_diagonal(mask, False)
theta = np.full(H, 6.0, np.float32)
channel = np.random.randint(1, 9, size=H).astype(np.uint8)
polarity = np.where(np.random.rand(H) > 0.1, 1.0, -1.0).astype(np.float32)

# SDR table
rng = np.random.RandomState(42)
BP_IN = np.zeros((256, IN_DIM), np.float32)
for v in range(256): BP_IN[v, rng.choice(IN_DIM, size=SDR_K, replace=False)] = 1.0

# Test data
test_bytes = bytes(range(256)) * 4  # 1024 tokens
N = len(test_bytes)

print(f"Benchmark: {N} tokens, H={H}")
print()

# === PYTHON BENCHMARK ===
sc = SelfWiringGraph.build_sparse_cache(mask)
state = np.zeros(H, np.float32)
charge = np.zeros(H, np.float32)

t0 = time.perf_counter()
for b in test_bytes:
    inj = np.zeros(H, np.float32)
    inj[:IN_DIM] = BP_IN[b]
    state, charge = SelfWiringGraph.rollout_token(
        inj, mask=mask, theta=theta, decay=np.float32(0.16),
        ticks=TICKS, input_duration=2, state=state, charge=charge,
        sparse_cache=sc, polarity=polarity, channel=channel)
t1 = time.perf_counter()
py_time = t1 - t0
py_tps = N / py_time
print(f"Python:  {py_time:.3f}s  = {py_tps:.0f} tokens/sec")

# === C BENCHMARK ===
c_exe = os.path.join(os.path.dirname(__file__), 'instnct_test')
if sys.platform == 'win32':
    c_exe += '.exe'

if not os.path.exists(c_exe):
    print(f"C binary not found: {c_exe}")
else:
    t0 = time.perf_counter()
    proc = subprocess.run(
        [c_exe],
        input=test_bytes,
        capture_output=True,
        timeout=30
    )
    t1 = time.perf_counter()
    c_time = t1 - t0
    # Subtract overhead: C prints header line first
    c_tps = N / c_time
    print(f"C:       {c_time:.3f}s  = {c_tps:.0f} tokens/sec")
    print()
    print(f"Speedup: {c_tps/py_tps:.1f}x")
    print(f"C output length: {len(proc.stdout)} bytes")
