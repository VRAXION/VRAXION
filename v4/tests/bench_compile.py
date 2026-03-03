"""Benchmark: torch.compile() speedup on INSTNCT forward pass.

Compares eager (default Python) vs compiled (torch.compile) execution.
The main bottleneck is the Python for-loop in _process_chunk — torch.compile
can fuse the small tensor ops inside the loop into optimized kernels.

Tests three strategies:
  1. Eager          — baseline (current behavior)
  2. compile(model) — compile the whole model
  3. compile(_process_chunk) — compile just the inner loop

Usage:
    python v4/tests/bench_compile.py
    python v4/tests/bench_compile.py --tokens 128 --runs 10
    python v4/tests/bench_compile.py --device cuda  # if GPU available
"""

import sys
import time
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for subdir in ('model', 'training', 'datagen'):
    p = str(ROOT / subdir)
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import torch.nn.functional as F
from instnct import INSTNCT


# ── Model configs ──
CONFIGS = {
    'small_N1': dict(
        M=64, hidden_dim=128, slot_dim=32, N=1, R=1,
        embed_mode=True, kernel_mode='vshape', pointer_mode='pilot',
        write_mode='replace', embed_encoding='bitlift',
        output_encoding='lowrank_c19', checkpoint_chunks=0,
    ),
    'prod_N6': dict(
        M=64, hidden_dim=128, slot_dim=32, N=6, R=1,
        embed_mode=True, kernel_mode='vshape', pointer_mode='pilot',
        write_mode='replace', embed_encoding='bitlift',
        output_encoding='lowrank_c19', checkpoint_chunks=0,
    ),
}


def make_model(cfg, device):
    torch.manual_seed(42)
    model = INSTNCT(**cfg).to(device)
    model.eval()
    return model


def bench_eager(model, x, warmup, n_runs):
    """Baseline: eager forward pass."""
    for _ in range(warmup):
        with torch.no_grad():
            model(x)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            out, _ = model(x)
        times.append(time.perf_counter() - t0)
    return times


def bench_compile_full(model, x, warmup, n_runs, backend='inductor'):
    """torch.compile() on the entire model."""
    try:
        compiled = torch.compile(model, backend=backend, fullgraph=False)
    except Exception as e:
        return None, str(e)

    # Extra warmup — first call triggers compilation
    for _ in range(warmup + 2):
        with torch.no_grad():
            compiled(x)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            out, _ = compiled(x)
        times.append(time.perf_counter() - t0)
    return times, None


def bench_compile_chunk(model, x, warmup, n_runs, backend='inductor'):
    """torch.compile() on just _process_chunk (the inner loop)."""
    try:
        model._process_chunk = torch.compile(
            model._process_chunk, backend=backend, fullgraph=False
        )
    except Exception as e:
        return None, str(e)

    for _ in range(warmup + 2):
        with torch.no_grad():
            model(x)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            out, _ = model(x)
        times.append(time.perf_counter() - t0)
    return times, None


def stats(times):
    avg = sum(times) / len(times)
    std = (sum((t - avg)**2 for t in times) / len(times)) ** 0.5
    mn = min(times)
    return avg, std, mn


def run():
    import argparse
    parser = argparse.ArgumentParser(description='torch.compile benchmark for INSTNCT')
    parser.add_argument('--tokens', type=int, default=64, help='sequence length')
    parser.add_argument('--warmup', type=int, default=3, help='warmup iterations')
    parser.add_argument('--runs', type=int, default=10, help='timed runs')
    parser.add_argument('--device', default='cpu', help='cpu or cuda')
    parser.add_argument('--config', default='prod_N6', choices=CONFIGS.keys())
    parser.add_argument('--backend', default='inductor', help='compile backend')
    args = parser.parse_args()

    device = args.device
    cfg = CONFIGS[args.config]
    N = cfg['N']
    tokens = args.tokens

    print(f"{'=' * 70}")
    print(f"  torch.compile() Benchmark — INSTNCT")
    print(f"  Config: {args.config} (N={N}, M={cfg['M']}, hidden={cfg['hidden_dim']})")
    print(f"  Tokens: {tokens}, Device: {device}, Backend: {args.backend}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"{'=' * 70}")

    x = torch.randint(0, 256, (1, tokens), dtype=torch.long, device=device)

    # ── 1. Eager baseline ──
    print(f"\n  [1/3] Eager (baseline)...")
    model_eager = make_model(cfg, device)
    times_eager = bench_eager(model_eager, x, args.warmup, args.runs)
    avg_e, std_e, min_e = stats(times_eager)
    per_tok_e = avg_e / tokens * 1000  # ms/token
    print(f"        avg={avg_e*1000:.1f}ms  std={std_e*1000:.1f}ms  "
          f"min={min_e*1000:.1f}ms  ({per_tok_e:.2f} ms/tok)")

    # ── 2. compile(model) ──
    print(f"\n  [2/3] torch.compile(model, backend='{args.backend}')...")
    print(f"        Compiling (first call triggers JIT, may take a while)...")
    model_full = make_model(cfg, device)
    times_full, err_full = bench_compile_full(
        model_full, x, args.warmup, args.runs, backend=args.backend)

    if err_full:
        print(f"        FAILED: {err_full}")
        avg_f, per_tok_f = None, None
    else:
        avg_f, std_f, min_f = stats(times_full)
        per_tok_f = avg_f / tokens * 1000
        speedup_f = avg_e / avg_f
        print(f"        avg={avg_f*1000:.1f}ms  std={std_f*1000:.1f}ms  "
              f"min={min_f*1000:.1f}ms  ({per_tok_f:.2f} ms/tok)")
        print(f"        Speedup: {speedup_f:.2f}x vs eager")

    # ── 3. compile(_process_chunk) ──
    print(f"\n  [3/3] torch.compile(_process_chunk, backend='{args.backend}')...")
    print(f"        Compiling (first call triggers JIT, may take a while)...")
    model_chunk = make_model(cfg, device)
    times_chunk, err_chunk = bench_compile_chunk(
        model_chunk, x, args.warmup, args.runs, backend=args.backend)

    if err_chunk:
        print(f"        FAILED: {err_chunk}")
        avg_c, per_tok_c = None, None
    else:
        avg_c, std_c, min_c = stats(times_chunk)
        per_tok_c = avg_c / tokens * 1000
        speedup_c = avg_e / avg_c
        print(f"        avg={avg_c*1000:.1f}ms  std={std_c*1000:.1f}ms  "
              f"min={min_c*1000:.1f}ms  ({per_tok_c:.2f} ms/tok)")
        print(f"        Speedup: {speedup_c:.2f}x vs eager")

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY")
    print(f"{'=' * 70}")
    print(f"  {'Method':<30} {'ms/tok':>10} {'Speedup':>10} {'Status':>10}")
    print(f"  {'-'*60}")
    print(f"  {'Eager (baseline)':<30} {per_tok_e:>9.2f}ms {'1.00x':>10} {'OK':>10}")

    if avg_f is not None:
        sp = avg_e / avg_f
        print(f"  {'compile(model)':<30} {per_tok_f:>9.2f}ms {f'{sp:.2f}x':>10} {'OK':>10}")
    else:
        print(f"  {'compile(model)':<30} {'—':>10} {'—':>10} {'FAIL':>10}")

    if avg_c is not None:
        sp = avg_e / avg_c
        print(f"  {'compile(_process_chunk)':<30} {per_tok_c:>9.2f}ms {f'{sp:.2f}x':>10} {'OK':>10}")
    else:
        print(f"  {'compile(_process_chunk)':<30} {'—':>10} {'—':>10} {'FAIL':>10}")

    print(f"{'=' * 70}")

    # ── What's next ──
    print(f"\n  Next steps if compile helps:")
    print(f"    - Try on CUDA: --device cuda")
    print(f"    - Try different backend: --backend eager / aot_eager / inductor")
    print(f"    - Profile: torch.profiler to see fused vs unfused ops")
    print(f"  Next steps if compile doesn't help:")
    print(f"    - Vectorize expert loop (N iterations → batched tensor ops)")
    print(f"    - Custom Triton kernel for the inner loop")
    print()


if __name__ == '__main__':
    run()
