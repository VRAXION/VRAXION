"""Benchmark: precompute vs old forward — speed + memory comparison."""
import time
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'model'))
from instnct import INSTNCT
from test_precompute import _old_forward

# ── Config (matches real training: M=256, D=256, N=6, batch=24, seq=128) ──
M, D, N = 256, 256, 6
BATCH, SEQ = 24, 128
WARMUP = 3
RUNS = 10
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f'Device: {DEVICE}')
print(f'Config: M={M} D={D} N={N} batch={BATCH} seq={SEQ}')
print(f'Warmup: {WARMUP}  Runs: {RUNS}')
print()

torch.manual_seed(42)
model = INSTNCT(M=M, embed_dim=D, N=N, embed_mode=True).to(DEVICE)
x = torch.randint(0, 256, (BATCH, SEQ), device=DEVICE)


def bench_new():
    """Current (precomputed) forward."""
    out, _ = model(x)
    loss = out.sum()
    loss.backward()
    model.zero_grad()
    return out


def bench_old():
    """Old (per-iteration) forward."""
    out = _old_forward(model, x)
    loss = out.sum()
    loss.backward()
    model.zero_grad()
    return out


def run_bench(name, fn):
    # warmup
    for _ in range(WARMUP):
        fn()
        if DEVICE == 'cuda':
            torch.cuda.synchronize()

    # measure
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    times = []
    for _ in range(RUNS):
        if DEVICE == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if DEVICE == 'cuda':
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    avg = sum(times) / len(times)
    std = (sum((t - avg) ** 2 for t in times) / len(times)) ** 0.5

    if DEVICE == 'cuda':
        peak_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
    else:
        peak_mb = 0

    print(f'{name:12s}  avg={avg*1000:.1f}ms  std={std*1000:.1f}ms  peak_vram={peak_mb:.0f}MB')
    return avg, peak_mb


print('Running OLD (per-iteration weights)...')
old_time, old_mem = run_bench('OLD', bench_old)

print('Running NEW (precomputed weights)...')
new_time, new_mem = run_bench('NEW', bench_new)

print()
speedup = (old_time - new_time) / old_time * 100
mem_saved = (old_mem - new_mem) / old_mem * 100 if old_mem > 0 else 0
print(f'Speed:  {speedup:+.1f}% {"faster" if speedup > 0 else "slower"}')
if DEVICE == 'cuda':
    print(f'VRAM:   {mem_saved:+.1f}% ({old_mem:.0f}MB -> {new_mem:.0f}MB)')
