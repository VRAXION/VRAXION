"""
Profiling Test -- v22 SelfWiringGraph
======================================
Deterministic measurement of every training loop component.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import time
from v22_best_config import SelfWiringGraph, softmax

configs = [
    {'V': 16, 'internal': 32, 'label': '16-class (N=64)'},
    {'V': 32, 'internal': 32, 'label': '32-class (N=96)'},
    {'V': 64, 'internal': 64, 'label': '64-class (N=192)'},
    {'V': 96, 'internal': 64, 'label': '96-class (N=256)'},
    {'V': 128, 'internal': 64, 'label': '128-class (N=320)'},
    {'V': 64, 'internal': 128, 'label': '64-cl 128int (N=256)'},
    {'V': 64, 'internal': 256, 'label': '64-cl 256int (N=384)'},
]

for cfg in configs:
    V = cfg['V']
    internal = cfg['internal']
    N = V + internal + V

    print(f"\n{'='*60}")
    print(f"  {cfg['label']}  (N={N}, connections max={N*(N-1)})")
    print(f"{'='*60}")

    np.random.seed(42)
    net = SelfWiringGraph(N, V)
    perm = np.random.permutation(V)

    # ===== 1. save_state() =====
    REPS = 1000
    t0 = time.perf_counter()
    for _ in range(REPS):
        state = net.save_state()
    t_save = (time.perf_counter() - t0) / REPS

    # ===== 2. mutate() =====
    t0 = time.perf_counter()
    for _ in range(REPS):
        net.restore_state(state)
        net.mutate_structure(0.05)
    t_mutate = (time.perf_counter() - t0) / REPS

    # ===== 3. reset() =====
    t0 = time.perf_counter()
    for _ in range(REPS):
        net.reset()
    t_reset = (time.perf_counter() - t0) / REPS

    # ===== 4. forward() -- SINGLE input, TICKS tick =====
    w = np.zeros(V, dtype=np.float32)
    w[0] = 1.0
    t0 = time.perf_counter()
    for _ in range(REPS):
        net.reset()
        logits = net.forward(w, 6)
    t_forward_single = (time.perf_counter() - t0) / REPS

    # ===== 5. softmax() =====
    logits = np.random.randn(V).astype(np.float32)
    t0 = time.perf_counter()
    for _ in range(REPS * 10):
        softmax(logits)
    t_softmax = (time.perf_counter() - t0) / (REPS * 10)

    # ===== 6. FULL EVAL (2 pass x V inputs) =====
    EVAL_REPS = max(1, min(50, int(1.0 / max(0.001, t_forward_single * V * 2))))
    t0 = time.perf_counter()
    for _ in range(EVAL_REPS):
        net.reset()
        for p in range(2):
            for i in range(V):
                w = np.zeros(V, dtype=np.float32)
                w[i] = 1.0
                logits = net.forward(w, 6)
                pr = softmax(logits[:V])
    t_full_eval = (time.perf_counter() - t0) / EVAL_REPS

    # ===== 7. restore_state() =====
    t0 = time.perf_counter()
    for _ in range(REPS):
        net.restore_state(state)
    t_restore = (time.perf_counter() - t0) / REPS

    # ===== MATRIX MULTIPLY BENCHMARK (the forward core) =====
    A = np.random.randn(N, N).astype(np.float32)
    v_vec = np.random.randn(N).astype(np.float32)
    t0 = time.perf_counter()
    for _ in range(REPS):
        _ = v_vec @ A
    t_matmul = (time.perf_counter() - t0) / REPS

    # ===== SUMMARY =====
    t_one_attempt = t_save + t_mutate + t_full_eval + t_restore

    print(f"\n  COMPONENT TIMES (microsec):")
    print(f"    save_state:     {t_save*1e6:>10.1f} us")
    print(f"    mutate:         {t_mutate*1e6:>10.1f} us")
    print(f"    reset:          {t_reset*1e6:>10.1f} us")
    print(f"    forward (1x):   {t_forward_single*1e6:>10.1f} us")
    print(f"    softmax:        {t_softmax*1e6:>10.1f} us")
    print(f"    matmul (NxN):   {t_matmul*1e6:>10.1f} us")
    print(f"    full_eval:      {t_full_eval*1e6:>10.1f} us")
    print(f"    restore_state:  {t_restore*1e6:>10.1f} us")

    print(f"\n  FULL EVAL BREAKDOWN:")
    fwd_in_eval = t_forward_single * V * 2
    softmax_in_eval = t_softmax * V
    other_in_eval = t_full_eval - fwd_in_eval - softmax_in_eval
    print(f"    forward calls:  {V*2:>5} x {t_forward_single*1e6:.1f}us = {fwd_in_eval*1e6:>10.1f} us ({fwd_in_eval/t_full_eval*100:.0f}%)")
    print(f"    softmax calls:  {V:>5} x {t_softmax*1e6:.1f}us = {softmax_in_eval*1e6:>10.1f} us ({softmax_in_eval/t_full_eval*100:.0f}%)")
    print(f"    overhead:       {other_in_eval*1e6:>10.1f} us ({other_in_eval/t_full_eval*100:.0f}%)")

    print(f"\n  FULL ATTEMPT:")
    print(f"    1 attempt:      {t_one_attempt*1e6:>10.1f} us ({t_one_attempt*1000:.2f} ms)")
    print(f"    attempts/sec:   {1/t_one_attempt:>10.0f}")
    print(f"    8K att time:    {t_one_attempt*8000:.1f}s")
    print(f"    16K att time:   {t_one_attempt*16000:.1f}s")

    print(f"\n  WHERE IS THE TIME?")
    total = t_save + t_mutate + t_full_eval + t_restore
    print(f"    save_state:     {t_save/total*100:>5.1f}%")
    print(f"    mutate:         {t_mutate/total*100:>5.1f}%")
    print(f"    full_eval:      {t_full_eval/total*100:>5.1f}%  <- THIS IS THE QUESTION")
    print(f"    restore_state:  {t_restore/total*100:>5.1f}%")

    print(f"\n  MATMUL vs FORWARD:")
    print(f"    1 matmul (v@A): {t_matmul*1e6:.1f} us")
    print(f"    1 forward:      {t_forward_single*1e6:.1f} us")
    print(f"    forward/matmul: {t_forward_single/max(t_matmul, 1e-9):.1f}x (TICKS=6 + overhead)")

    print(f"\n  'WHAT IF' SCENARIOS:")
    # Half eval (V/2 samples)
    t_half_eval = t_forward_single * V + softmax_in_eval/2 + other_in_eval/2
    t_half_att = t_save + t_mutate + t_half_eval + t_restore
    print(f"    Half eval ({V//2} samples):  {1/t_half_att:.0f} att/s ({1/t_half_att/(1/t_one_attempt):.1f}x faster)")

    # GPU matmul
    t_gpu_matmul = t_matmul * 0.01
    t_gpu_fwd = t_gpu_matmul * 6 + (t_forward_single - t_matmul * 6) * 0.5
    t_gpu_eval = t_gpu_fwd * V * 2 + softmax_in_eval + other_in_eval
    t_gpu_att = t_save + t_mutate + t_gpu_eval + t_restore
    print(f"    GPU matmul:     {1/t_gpu_att:.0f} att/s ({1/t_gpu_att/(1/t_one_attempt):.1f}x faster)")

    # Online (1 sample)
    t_online_att = t_save + t_mutate + t_forward_single*2 + t_softmax*2 + t_restore
    print(f"    Online (1 sample): {1/t_online_att:.0f} att/s ({1/t_online_att/(1/t_one_attempt):.1f}x faster)")

    sys.stdout.flush()

print(f"\n{'='*60}")
print(f"  DONE")
print(f"{'='*60}", flush=True)
