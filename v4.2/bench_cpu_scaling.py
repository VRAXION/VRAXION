"""CPU scaling: single-worker forward pass time at various sizes.
Multiply by ~1.1 for wall time (18 workers parallel on 24 cores, slight contention).
"""
import numpy as np
import time

def make_bp(io_dim, seed=12345):
    rng = np.random.RandomState(seed)
    p = rng.randn(256, io_dim).astype(np.float32)
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    return p

IO = 256
bp = make_bp(IO)
SEQ_LEN = 200
N_SEQS = 5
TICKS = 6

print("=" * 70)
print("CPU Single-Worker Benchmark (2 evals x 5 seqs x 200b x 6 ticks)")
print("18 workers parallel -> step time ~ worker time (24 cores)")
print("=" * 70)
print(f"{'Neurons':>8} {'H':>6} {'Edges':>8} {'1 worker':>10} {'step/s':>8} {'verdict':>10}")

configs = [
    (256,  [500, 2000, 5000]),
    (512,  [1000, 5000, 10000]),
    (768,  [2000, 5000, 10000, 20000]),
    (1024, [2000, 5000, 10000, 20000]),
    (1536, [5000, 10000, 20000]),
    (2048, [5000, 10000, 20000, 50000]),
    (3072, [10000, 20000, 50000]),
    (4096, [10000, 50000]),
]

for neurons, edge_list in configs:
    H = neurons * 3
    for n_edges in edge_list:
        rng = np.random.RandomState(42)
        rows = rng.randint(0, H, n_edges).astype(np.intp)
        cols = rng.randint(0, H, n_edges).astype(np.intp)
        vals = rng.choice([-0.6, 0.6], n_edges).astype(np.float32)
        theta = rng.uniform(0, 0.3, H).astype(np.float32)
        decay = rng.uniform(0.01, 0.3, H).astype(np.float32)
        ret = (1.0 - decay).astype(np.float32)
        input_projection = rng.randn(256, H).astype(np.float32) * 0.1
        output_projection = rng.randn(H, 256).astype(np.float32) * 0.1

        t0 = time.perf_counter()
        for _ in range(2):  # old + new
            for _ in range(N_SEQS):
                text = rng.randint(0, 256, SEQ_LEN, dtype=np.uint8)
                state = np.zeros(H, dtype=np.float32)
                charge = np.zeros(H, dtype=np.float32)
                for i in range(SEQ_LEN - 1):
                    act = state.copy()
                    for t in range(TICKS):
                        if t == 0:
                            act = act + bp[text[i]] @ input_projection
                        raw = np.zeros(H, dtype=np.float32)
                        np.add.at(raw, cols, act[rows] * vals)
                        charge += raw
                        charge *= ret
                        act = np.maximum(charge - theta, 0.0)
                        charge = np.clip(charge, -1.0, 1.0)
                    state = act.copy()
        worker_time = time.perf_counter() - t0
        sps = 1.0 / (worker_time * 1.1)  # ~10% overhead for parallel contention

        if sps > 1.0:
            verdict = "FAST"
        elif sps > 0.5:
            verdict = "OK"
        elif sps > 0.2:
            verdict = "SLOW"
        else:
            verdict = "TOO SLOW"

        print(f"{neurons:>8} {H:>6} {n_edges:>8} {worker_time:>9.2f}s {sps:>7.2f} {verdict:>10}")

print("\n" + "=" * 70)
print("FAST = >1 step/s | OK = 0.5-1 | SLOW = 0.2-0.5 | TOO SLOW = <0.2")
print("Network starts at ~0 edges and grows. Plan for 10-20K edges long-term.")
print("=" * 70)
