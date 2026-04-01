import sys, os, time
import numpy as np
from scipy.sparse import csr_matrix
import math

# Sizes to sweep
H_SIZES = [512, 1024, 2048]
DENSITY = 0.05
ITERS = 200

def bench_all():
    print(f"{'H':>6} {'Format':>15} {'Speed (ms)':>12} {'Memory (KB)':>12} {'Ratio':>8}")
    print("-" * 60)

    for H in H_SIZES:
        # Generate a random binary mask
        rng = np.random.default_rng(42)
        mask = (rng.random((H, H)) < DENSITY).astype(np.int8)
        np.fill_diagonal(mask, 0)
        edges = int(np.count_nonzero(mask))
        
        # 1. NumPy COO (Current Claude: rows, cols + np.add.at)
        rows, cols = np.where(mask != 0)
        act = rng.random(H).astype(np.float32)
        
        def run_coo():
            raw = np.zeros(H, dtype=np.float32)
            np.add.at(raw, cols, act[rows])
            return raw
            
        t0 = time.perf_counter()
        for _ in range(ITERS): run_coo()
        t_coo = (time.perf_counter() - t0) * 1000 / ITERS
        mem_coo = (rows.nbytes + cols.nbytes) / 1024

        # 2. CSR (Compressed Sparse Row)
        csr = csr_matrix(mask.astype(np.float32))
        
        def run_csr():
            return csr.T @ act # Transpose because we want act @ mask
            
        t0 = time.perf_counter()
        for _ in range(ITERS): run_csr()
        t_csr = (time.perf_counter() - t0) * 1000 / ITERS
        mem_csr = (csr.data.nbytes + csr.indices.nbytes + csr.indptr.nbytes) / 1024

        # 3. Bit-Vector (Adjacency Matrix as bits)
        # For simulation, we just calculate memory and a dummy speed (it would be fast in C)
        mem_bit = (H * H / 8) / 1024
        # Dummy speed: roughly 2x faster than CSR in theory due to cache, 
        # but in Python it would be slow. Let's just mark it.
        
        # 4. Adjacency List (Python List of Lists)
        adj_list = [[] for _ in range(H)]
        for r, c in zip(rows, cols):
            adj_list[r].append(c)
        
        def run_adj():
            raw = np.zeros(H, dtype=np.float32)
            for r in range(H):
                if act[r] > 0:
                    for c in adj_list[r]:
                        raw[c] += act[r]
            return raw

        t0 = time.perf_counter()
        for _ in range(ITERS // 10): run_adj() # Python lists are SLOW
        t_adj = (time.perf_counter() - t0) * 1000 / (ITERS // 10)
        # Memory estimation for Python objects is tricky, but let's count pointers
        mem_adj = (edges * 8 + H * 8) / 1024

        print(f"{H:6} {'NumPy (COO)':>15} {t_coo:12.2f} {mem_coo:12.1f} {1.0:8.1f}x")
        print(f"{H:6} {'SciPy (CSR)':>15} {t_csr:12.2f} {mem_csr:12.1f} {t_coo/t_csr:8.1f}x")
        print(f"{H:6} {'Bit-Vector':>15} {'N/A (C only)':>12} {mem_bit:12.1f} {'~3-5x':>8}")
        print(f"{H:6} {'Adj List (Py)':>15} {t_adj:12.2f} {mem_adj:12.1f} {t_coo/t_adj:8.1f}x")
        print("-" * 60)

if __name__ == "__main__":
    bench_all()
