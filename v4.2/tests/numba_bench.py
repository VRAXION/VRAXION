"""Numba JIT vs numpy CPU forward_batch benchmark."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import numba
from numba import njit
import time
from model.graph import SelfWiringGraph

N, V = 192, 64
GAIN = 2
CR = np.float32(0.3)
SC = np.float32(0.05)
TH = np.float32(0.5)
CB = np.float32(1.0)


@njit(cache=True)
def forward_numba(mask, retain, V, N):
    charges = np.zeros((V, N), dtype=np.float32)
    acts = np.zeros((V, N), dtype=np.float32)
    for t in range(8):
        if t == 0:
            for i in range(V):
                acts[i, i] = np.float32(1.0)
        # matmul + self_conn
        raw = np.zeros((V, N), dtype=np.float32)
        for i in range(V):
            for j in range(N):
                s = np.float32(0.0)
                for k in range(N):
                    s += acts[i, k] * mask[k, j]
                raw[i, j] = s * GAIN + acts[i, j] * SC
        # capacitor dynamics
        for i in range(V):
            for j in range(N):
                charges[i, j] += raw[i, j] * CR
                charges[i, j] *= retain
                a = charges[i, j] - TH
                acts[i, j] = a if a > 0 else np.float32(0.0)
                if charges[i, j] > CB:
                    charges[i, j] = CB
                elif charges[i, j] < -CB:
                    charges[i, j] = -CB
    return charges[:, N - V:]


def main():
    print(f"Numba {numba.__version__}", flush=True)

    mask = np.random.choice([-1, 0, 1], size=(N, N), p=[0.03, 0.94, 0.03]).astype(np.int8)
    np.fill_diagonal(mask, 0)
    retain = np.float32(0.85)

    # JIT compile
    print("Compiling...", flush=True)
    t0 = time.perf_counter()
    forward_numba(mask, retain, V, N)
    print(f"JIT compile: {time.perf_counter() - t0:.1f}s", flush=True)

    # Numba benchmark
    times = []
    for _ in range(50):
        t0 = time.perf_counter()
        forward_numba(mask, retain, V, N)
        times.append(time.perf_counter() - t0)
    numba_ms = np.median(times) * 1000
    print(f"Numba JIT CPU:     {numba_ms:.2f} ms", flush=True)

    # Numpy baseline
    net = SelfWiringGraph(N, V)
    for _ in range(5):
        net.forward_batch()
    times2 = []
    for _ in range(50):
        t0 = time.perf_counter()
        net.forward_batch()
        times2.append(time.perf_counter() - t0)
    numpy_ms = np.median(times2) * 1000
    print(f"Numpy CPU:         {numpy_ms:.2f} ms", flush=True)
    print(f"Speedup:           {numpy_ms / numba_ms:.1f}x", flush=True)

    # Also V128
    N2, V2 = 384, 128
    mask2 = np.random.choice([-1, 0, 1], size=(N2, N2), p=[0.03, 0.94, 0.03]).astype(np.int8)
    np.fill_diagonal(mask2, 0)

    forward_numba(mask2, retain, V2, N2)  # warmup
    times3 = []
    for _ in range(20):
        t0 = time.perf_counter()
        forward_numba(mask2, retain, V2, N2)
        times3.append(time.perf_counter() - t0)
    numba128 = np.median(times3) * 1000

    net2 = SelfWiringGraph(N2, V2)
    for _ in range(5): net2.forward_batch()
    times4 = []
    for _ in range(20):
        t0 = time.perf_counter()
        net2.forward_batch()
        times4.append(time.perf_counter() - t0)
    numpy128 = np.median(times4) * 1000

    print(f"\nV128_N384:", flush=True)
    print(f"Numba JIT CPU:     {numba128:.2f} ms", flush=True)
    print(f"Numpy CPU:         {numpy128:.2f} ms", flush=True)
    print(f"Speedup:           {numpy128 / numba128:.1f}x", flush=True)


if __name__ == '__main__':
    main()
