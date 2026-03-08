#!/usr/bin/env python3
"""Micro-benchmark: batched vs sequential expert loop speedup.

Measures wall-clock per-step time for N=1,2,4,6 with both paths.
"""

import sys, os, time, gc
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for subdir in ("model", "training"):
    p = str(ROOT / subdir)
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import torch.nn.functional as F
from instnct import INSTNCT

SEED = 42
WARMUP = 3
MEASURE = 10


def bench(N, B, T, M, hidden_dim, slot_dim, force_sequential=False):
    torch.manual_seed(SEED)
    model = INSTNCT(
        M=M, hidden_dim=hidden_dim, slot_dim=slot_dim, N=N, R=2,
        embed_mode=True, kernel_mode='vshape',
        pointer_mode='sequential', write_mode='replace',
        embed_encoding='learned', output_encoding='lowrank_c19',
        c19_mode='dualphi',
        pointer_interp_mode='linear',
        pointer_seam_mode='shortest_arc',
    )
    model._diag_enabled = False
    model.train()

    if force_sequential:
        model._force_sequential = True
        orig = type(model)._can_batch_experts
        type(model)._can_batch_experts = property(
            lambda self: not getattr(self, '_force_sequential', False) and orig.fget(self)
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    state = None

    # Warmup
    for _ in range(WARMUP):
        x = torch.randint(0, 256, (B, T))
        tgt = torch.randint(0, 256, (B, T))
        out, state = model(x, state=state)
        loss = F.cross_entropy(out.reshape(-1, 256), tgt.reshape(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        state = {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in state.items()}

    # Measure
    times = []
    for _ in range(MEASURE):
        x = torch.randint(0, 256, (B, T))
        tgt = torch.randint(0, 256, (B, T))
        t0 = time.perf_counter()
        out, state = model(x, state=state)
        loss = F.cross_entropy(out.reshape(-1, 256), tgt.reshape(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        state = {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in state.items()}
        times.append((time.perf_counter() - t0) * 1000)

    if force_sequential:
        type(model)._can_batch_experts = orig

    del model, optimizer
    gc.collect()

    import statistics
    return statistics.mean(times), statistics.stdev(times) if len(times) > 1 else 0.0


def main():
    configs = [
        # (N, B, T, M, hidden_dim, slot_dim, label)
        (1, 16, 48, 128, 256, 64, "N=1 small"),
        (2, 16, 48, 128, 256, 64, "N=2 small"),
        (4, 16, 48, 128, 256, 64, "N=4 small"),
        (6, 16, 48, 128, 256, 64, "N=6 small"),
        (1, 16, 48, 128, 2048, 128, "N=1 large"),
        (2, 16, 48, 128, 2048, 128, "N=2 large"),
        (2, 16, 48, 128, 4096, 128, "N=2 prod"),
    ]

    print("=" * 80)
    print("Expert Loop Batching — Speedup Benchmark")
    print("=" * 80)
    print(f"  Warmup: {WARMUP}, Measure: {MEASURE} steps")
    print(f"  {'Config':<20} {'Sequential':>12} {'Batched':>12} {'Speedup':>10}")
    print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*10}")

    for N, B, T, M, hd, sd, label in configs:
        seq_ms, seq_std = bench(N, B, T, M, hd, sd, force_sequential=True)
        bat_ms, bat_std = bench(N, B, T, M, hd, sd, force_sequential=False)
        speedup = (seq_ms - bat_ms) / seq_ms * 100
        print(f"  {label:<20} {seq_ms:>8.1f}±{seq_std:>3.0f}ms {bat_ms:>8.1f}±{bat_std:>3.0f}ms {speedup:>+8.1f}%")

    print("=" * 80)


if __name__ == '__main__':
    main()
