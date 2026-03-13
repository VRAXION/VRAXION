"""Inference benchmark: C19 vs alternatives — forward-only latency.

Measures ACTUAL inference cost per token by running the full model
forward pass with torch.no_grad(). This is what matters in production.

C19 is called 7x per token (1 input + 6 expert hidden):
- Each call does: floor, remainder, 2x where, multiply, sigmoid(rho/C)
- Total: ~98 ops per token just for activation

Alternative activations are much cheaper per call, but the question is:
does the per-call speedup translate to meaningful per-token speedup?

Usage: python v4/tests/bench_inference.py [--tokens 100] [--warmup 5]
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

import numpy as np
import torch
import torch.nn.functional as F

import instnct as instnct_module
from instnct import INSTNCT

_original_c19 = instnct_module._c19_activation

# ── Activation alternatives ──
def _tanh_activation(x, rho=4.0, C=None):
    return torch.tanh(x)

def _silu_activation(x, rho=4.0, C=None):
    return F.silu(x)

def _gelu_activation(x, rho=4.0, C=None):
    return F.gelu(x)

def _learnable_tanh(x, rho=4.0, C=None):
    if C is None: C = math.pi
    return rho * torch.tanh(x / C)

def _hardtanh_activation(x, rho=4.0, C=None):
    return torch.clamp(x, -rho, rho)

def _softsign_activation(x, rho=4.0, C=None):
    return x / (1 + x.abs())


# ── Production-size model config ──
PROD_CFG = dict(
    M=64, hidden_dim=128, slot_dim=32, N=6, R=1,
    embed_mode=True,
    kernel_mode='vshape',
    pointer_mode='pilot',
    write_mode='replace',
    embed_encoding='bitlift',
    output_encoding='lowrank_c19',
    checkpoint_chunks=0,
)

# Small model for quick comparison
SMALL_CFG = dict(
    M=64, hidden_dim=128, slot_dim=32, N=1, R=1,
    embed_mode=True,
    kernel_mode='vshape',
    pointer_mode='pilot',
    write_mode='replace',
    embed_encoding='bitlift',
    output_encoding='lowrank_c19',
    checkpoint_chunks=0,
)


def bench_forward(name, activation_fn, cfg, tokens, warmup, device='cpu'):
    """Benchmark full forward pass latency."""
    instnct_module._c19_activation = activation_fn
    torch.manual_seed(42)

    model = INSTNCT(**cfg).to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())

    # Input: random byte tokens
    x = torch.randint(0, 256, (1, tokens), dtype=torch.long, device=device)

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            model(x)

    # Timed runs
    times = []
    n_runs = max(3, 20 // max(1, tokens // 50))  # more runs for short seqs
    for _ in range(n_runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            out, _ = model(x)
        times.append(time.perf_counter() - t0)

    instnct_module._c19_activation = _original_c19

    avg = sum(times) / len(times)
    per_token = avg / tokens * 1000  # ms per token
    return {
        'name': name,
        'params': n_params,
        'total_ms': avg * 1000,
        'per_token_ms': per_token,
        'tokens_per_sec': tokens / avg,
        'n_runs': n_runs,
        'std_ms': (sum((t - avg)**2 for t in times) / len(times))**0.5 * 1000,
    }


def bench_activation_only():
    """Micro-benchmark: isolated activation calls (no model overhead)."""
    from instnct import _rho_from_raw, _C_from_raw, _rho_init_raw, _C_init_raw

    # Realistic tensor sizes
    sizes = [
        ('hidden (1,128)', (1, 128)),
        ('hidden (8,128)', (8, 128)),
        ('hidden (1,512)', (1, 512)),
    ]

    rho_raw = torch.full((128,), _rho_init_raw(4.0))
    C_raw = torch.full((128,), _C_init_raw())
    rho = _rho_from_raw(rho_raw)
    C_val = _C_from_raw(C_raw)

    activations = [
        ('C19', _original_c19),
        ('tanh', _tanh_activation),
        ('SiLU', _silu_activation),
        ('GELU', _gelu_activation),
        ('Learn-tanh', _learnable_tanh),
        ('Hardtanh', _hardtanh_activation),
        ('Softsign', _softsign_activation),
    ]

    print("=" * 70)
    print("  MICRO-BENCHMARK: Isolated activation call latency")
    print("=" * 70)

    for size_name, shape in sizes:
        x = torch.randn(*shape)
        rho_b = rho[:shape[-1]] if shape[-1] <= 128 else rho.repeat(shape[-1] // 128 + 1)[:shape[-1]]
        C_b = C_val[:shape[-1]] if shape[-1] <= 128 else C_val.repeat(shape[-1] // 128 + 1)[:shape[-1]]

        print(f"\n  Shape: {size_name}")
        print(f"  {'Activation':<14} {'Time (μs)':>10} {'vs C19':>8}  "
              f"{'×7 (per token)':>15} {'vs C19 ×7':>12}")
        print(f"  {'-'*60}")

        c19_time = None
        for act_name, fn in activations:
            # Warmup
            for _ in range(50):
                fn(x, rho=rho_b, C=C_b)
            # Measure
            t_list = []
            for _ in range(200):
                t0 = time.perf_counter()
                fn(x, rho=rho_b, C=C_b)
                t_list.append((time.perf_counter() - t0) * 1e6)
            avg = sum(t_list) / len(t_list)
            if c19_time is None:
                c19_time = avg
            speedup = c19_time / avg
            # Per-token cost = 7 calls (1 input + 6 hidden)
            per_token = avg * 7
            per_token_c19 = c19_time * 7
            token_speedup = per_token_c19 / per_token
            print(f"  {act_name:<14} {avg:>9.1f}μs {speedup:>7.1f}x  "
                  f"{per_token:>13.0f}μs {token_speedup:>11.1f}x")


def run():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokens', type=int, default=64)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--full', action='store_true', help='Run full model benchmark too')
    args = parser.parse_args()

    # Part 1: Micro-benchmark
    bench_activation_only()

    # Part 2: Full model forward pass (N=1 for speed, then N=6 production)
    variants = [
        ('C19', _original_c19),
        ('tanh', _tanh_activation),
        ('SiLU', _silu_activation),
        ('GELU', _gelu_activation),
        ('Learn-tanh', _learnable_tanh),
        ('Hardtanh', _hardtanh_activation),
        ('Softsign', _softsign_activation),
    ]

    for cfg_name, cfg in [('N=1 (small)', SMALL_CFG), ('N=6 (production)', PROD_CFG)]:
        N = cfg['N']
        calls_per_token = 1 + N  # 1 input + N hidden

        print(f"\n{'=' * 70}")
        print(f"  FULL MODEL INFERENCE: {cfg_name}")
        print(f"  Tokens: {args.tokens}, C19 calls/token: {calls_per_token}")
        print(f"{'=' * 70}")
        print(f"  {'Activation':<14} {'Total (ms)':>10} {'Per token':>12} "
              f"{'Tok/sec':>10} {'vs C19':>8}")
        print(f"  {'-'*60}")

        c19_result = None
        results = []
        for name, fn in variants:
            r = bench_forward(name, fn, cfg, args.tokens, args.warmup)
            results.append(r)
            if c19_result is None:
                c19_result = r
            speedup = c19_result['per_token_ms'] / r['per_token_ms']
            print(f"  {r['name']:<14} {r['total_ms']:>9.1f}ms "
                  f"{r['per_token_ms']:>10.2f}ms/tok "
                  f"{r['tokens_per_sec']:>9.1f} {speedup:>7.1f}x")

        # Summary
        print()
        print(f"  Activation cost as % of total forward pass:")
        c19_ms = c19_result['per_token_ms']
        for r in results[1:]:
            saved_ms = c19_ms - r['per_token_ms']
            saved_pct = saved_ms / c19_ms * 100
            print(f"    {r['name']:<14}: saves {saved_ms:.2f}ms/tok "
                  f"({saved_pct:.1f}% of total inference time)")

    # Production cost estimate
    print(f"\n{'=' * 70}")
    print(f"  PRODUCTION COST ESTIMATE")
    print(f"{'=' * 70}")
    c19_prod = None
    best_alt = None
    for r in results:
        if r['name'] == 'C19':
            c19_prod = r
        elif best_alt is None or r['tokens_per_sec'] > best_alt['tokens_per_sec']:
            best_alt = r

    if c19_prod and best_alt:
        c19_tps = c19_prod['tokens_per_sec']
        alt_tps = best_alt['tokens_per_sec']
        speedup = alt_tps / c19_tps
        print(f"  C19:                 {c19_tps:.1f} tok/s")
        print(f"  Best alternative:    {best_alt['name']}: {alt_tps:.1f} tok/s ({speedup:.1f}x)")
        print()
        # 1M token inference cost
        c19_sec = 1_000_000 / c19_tps
        alt_sec = 1_000_000 / alt_tps
        print(f"  1M token inference:  C19 = {c19_sec:.0f}s, "
              f"{best_alt['name']} = {alt_sec:.0f}s "
              f"(saves {c19_sec - alt_sec:.0f}s)")

    print("=" * 70)


if __name__ == '__main__':
    run()
