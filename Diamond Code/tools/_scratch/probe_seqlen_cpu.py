"""Probe: seq_len scaling on CPU with depth=2, various D values.
Measures forward pass time only (no backward, no LCX write).
Safe to run while GPU training is active.
"""
import sys, os, time, signal
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
from swarm_model import SwarmByteRingModel

STEP_TIMEOUT = 30

configs = [
    # (D, depth, seq_len, label)
    (618,  2,  128,  "D618  seq128"),
    (618,  2,  256,  "D618  seq256"),
    (618,  2,  512,  "D618  seq512"),
    (1000, 2,  128,  "D1000 seq128"),
    (1000, 2,  256,  "D1000 seq256"),
    (1000, 2,  512,  "D1000 seq512"),
    (2000, 2,  128,  "D2000 seq128"),
    (2000, 2,  256,  "D2000 seq256"),
    (2000, 2,  512,  "D2000 seq512"),
    (4000, 2,  128,  "D4000 seq128"),
    (4000, 2,  256,  "D4000 seq256"),
    (6180, 2,  128,  "D6180 seq128"),
    (6180, 2,  192,  "D6180 seq192"),
]

print(f"{'Label':<22} {'D':>5} {'depth':>5} {'seq':>5} {'params':>10} {'fwd_ms':>8} {'MB_RAM':>8}")
print("-" * 75)

for D, depth, seq_len, label in configs:
    try:
        model = SwarmByteRingModel(
            num_bits=8,
            embedding_dim=D,
            depth=depth,
            num_beings=1,
            num_memory_positions=seq_len,
            lcx_mode='hash',
            lcx_num_levels=1,
            lcx_level_slots=2000,
            lcx_key_dim=max(D // 10, 32),
            lcx_top_k=2,
            num_pointers=1,
            attention_radius=8,
        )
        model.eval()
        model.to('cpu')

        n_params = sum(p.numel() for p in model.parameters())

        # Warmup
        x = torch.randint(0, 2, (1, seq_len, 8), dtype=torch.float32)
        with torch.no_grad():
            model(x)

        # Timed runs
        times = []
        for _ in range(5):
            x = torch.randint(0, 2, (1, seq_len, 8), dtype=torch.float32)
            t0 = time.time()
            with torch.no_grad():
                model(x)
            dt = time.time() - t0
            if dt > STEP_TIMEOUT:
                print(f"TIMEOUT: {label} took {dt:.0f}s, skipping")
                break
            times.append(dt * 1000)

        import os as _os
        try:
            import psutil
            mem_mb = psutil.Process().memory_info().rss / 1024 / 1024
        except ImportError:
            mem_mb = 0

        avg_ms = sum(times) / len(times) if times else -1
        print(f"{label:<22} {D:>5} {depth:>5} {seq_len:>5} {n_params:>10,} {avg_ms:>8.1f} {mem_mb:>8.0f}")

        del model, x

    except Exception as e:
        print(f"{label:<22} ERROR: {e}")

print("\nDone.")
