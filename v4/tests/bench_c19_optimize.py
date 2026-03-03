"""Benchmark: C19 current vs optimized — keeping learnable rho + C.

Goal: reduce the 8+ elementwise ops in _c19_activation without removing
any learnable parameter. The key insight: most ops can be fused by
rewriting the math to minimize temporaries.

Current C19 ops (per call):
  1. x * inv_c           (mul)
  2. torch.floor          (floor)
  3. scaled - n           (sub)
  4. t * (1.0 - t)        (sub + mul)
  5. torch.remainder      (remainder — EXPENSIVE, ~40us)
  6. torch.where (is_even)(where + comparison)
  7. sgn * h + rho * h*h  (mul + mul + mul + add)
  8. C * (...)            (mul)
  9. torch.where (>=l)    (where + comparison)
  10. torch.where (<=-l)  (where + comparison)

Total: ~13 tensor ops + 3 where branches

Strategy for v2:
  - Replace remainder(n,2) < 1.0 with cheaper bit trick: (floor & 1) == 0
  - Merge the two outer where() into a single clamp-based approach
  - Pre-fuse h and h*h into one expression
  - Use torch.frac() instead of floor+subtract
"""

import time
import torch
import torch.nn as nn
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'model'))
from instnct import _c19_activation, _rho_from_raw, _C_from_raw, _rho_init_raw, _C_init_raw

DEVICE = 'cpu'
B = 16
HIDDEN = 2048
WARMUP = 10
REPEATS = 50


def _c19_v2(x, rho=4.0, C=None):
    """Optimized C19: fewer tensor ops, same math, same learnable params."""
    if C is None:
        C = math.pi
    l = 6.0 * C
    inv_c = 1.0 / C

    scaled = x * inv_c
    n = torch.floor(scaled)
    t = scaled - n               # fractional part in [0, 1)
    h = t - t * t                # t*(1-t) rewritten as t - t^2 (same, 1 fewer op)

    # Cheaper even/odd: cast floor to int, check bit 0
    # n.to(torch.int32) & 1 == 0 → even → positive sign
    sgn = 1.0 - 2.0 * (n.to(torch.int32) & 1).float()  # +1 if even, -1 if odd

    core = C * (sgn * h + rho * (h * h))

    # Fuse the two outer where() into one: clamp to [-l, l] range first
    return torch.where(
        (x >= -l) & (x <= l),
        core,
        x - l * torch.sign(x)   # linear pass-through with offset
    )


def _c19_v3(x, rho=4.0, C=None):
    """V3: int bit trick + fused where, floor-based (correct for negatives)."""
    if C is None:
        C = math.pi
    l = 6.0 * C
    inv_c = 1.0 / C

    scaled = x * inv_c
    n = torch.floor(scaled)
    t = scaled - n                # always in [0, 1)
    h = t - t * t                 # t*(1-t)

    sgn = 1.0 - 2.0 * (n.to(torch.int32) & 1).float()

    core = C * (sgn * h + rho * (h * h))

    mask = (x >= -l) & (x <= l)
    return torch.where(mask, core, x - l * torch.sign(x))


def _c19_v4_snake(x, rho=4.0, C=None):
    """V4 'Snake-style': x + (1/C) * sin^2(C*x) — periodic, learnable, 3 ops.
    Preserves learnable C (frequency) and rho (amplitude scale).
    Mathematically different but same parameter roles."""
    if C is None:
        C = math.pi
    s = torch.sin(C * x)
    return x + (rho * 0.25) * (s * s)  # rho controls amplitude of periodic component


def time_fn(fn, warmup=WARMUP, repeats=REPEATS):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return sum(times) / len(times)


def check_correctness():
    """Verify v2 and v3 match original C19 exactly."""
    torch.manual_seed(42)
    x = torch.randn(100, 2048) * 10  # large range to hit clamp branches

    # Test with scalar rho/C
    out_orig = _c19_activation(x, rho=4.0, C=math.pi)
    out_v2 = _c19_v2(x, rho=4.0, C=math.pi)
    out_v3 = _c19_v3(x, rho=4.0, C=math.pi)

    err_v2 = (out_orig - out_v2).abs().max().item()
    err_v3 = (out_orig - out_v3).abs().max().item()
    print(f"Max error v2 vs original: {err_v2:.2e}")
    print(f"Max error v3 vs original: {err_v3:.2e}")

    # Test with per-neuron learnable rho and C (tensor)
    rho_raw = torch.full((2048,), _rho_init_raw(4.0))
    C_raw = torch.full((2048,), _C_init_raw())
    rho = _rho_from_raw(rho_raw)
    C_val = _C_from_raw(C_raw)

    out_orig_t = _c19_activation(x, rho=rho, C=C_val)
    out_v2_t = _c19_v2(x, rho=rho, C=C_val)
    out_v3_t = _c19_v3(x, rho=rho, C=C_val)

    err_v2_t = (out_orig_t - out_v2_t).abs().max().item()
    err_v3_t = (out_orig_t - out_v3_t).abs().max().item()
    print(f"Max error v2 (tensor rho+C): {err_v2_t:.2e}")
    print(f"Max error v3 (tensor rho+C): {err_v3_t:.2e}")

    ok = all(e < 1e-5 for e in [err_v2, err_v3, err_v2_t, err_v3_t])
    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    return ok


def bench_all():
    print("=" * 65)
    print("C19 Activation — Optimization Benchmark")
    print(f"Shape: ({B}, {HIDDEN}), device={DEVICE}")
    print("=" * 65)

    check_correctness()
    print()

    torch.manual_seed(42)
    x = torch.randn(B, HIDDEN)

    # Learnable params (per-neuron tensor)
    rho_raw = torch.full((HIDDEN,), _rho_init_raw(4.0))
    C_raw = torch.full((HIDDEN,), _C_init_raw())
    rho = _rho_from_raw(rho_raw)
    C_val = _C_from_raw(C_raw)

    # ── Scalar versions (fixed rho/C) ──
    ms_orig_scalar = time_fn(lambda: _c19_activation(x, rho=4.0, C=math.pi))
    ms_v2_scalar = time_fn(lambda: _c19_v2(x, rho=4.0, C=math.pi))
    ms_v3_scalar = time_fn(lambda: _c19_v3(x, rho=4.0, C=math.pi))
    ms_v4_scalar = time_fn(lambda: _c19_v4_snake(x, rho=4.0, C=math.pi))

    # ── Tensor versions (learnable per-neuron rho + C) ──
    ms_orig_tensor = time_fn(lambda: _c19_activation(x, rho=rho, C=C_val))
    ms_v2_tensor = time_fn(lambda: _c19_v2(x, rho=rho, C=C_val))
    ms_v3_tensor = time_fn(lambda: _c19_v3(x, rho=rho, C=C_val))
    ms_v4_tensor = time_fn(lambda: _c19_v4_snake(x, rho=rho, C=C_val))

    # ── Include sigmoid bounding cost (full pipeline) ──
    def full_orig():
        r = _rho_from_raw(rho_raw)
        c = _C_from_raw(C_raw)
        return _c19_activation(x, rho=r, C=c)

    def full_v3():
        r = _rho_from_raw(rho_raw)
        c = _C_from_raw(C_raw)
        return _c19_v3(x, rho=r, C=c)

    def full_v4():
        r = _rho_from_raw(rho_raw)
        c = _C_from_raw(C_raw)
        return _c19_v4_snake(x, rho=r, C=c)

    ms_full_orig = time_fn(full_orig)
    ms_full_v3 = time_fn(full_v3)
    ms_full_v4 = time_fn(full_v4)

    # ── Reference: standard activations ──
    ms_relu = time_fn(lambda: torch.relu(x))
    ms_gelu = time_fn(lambda: torch.nn.functional.gelu(x))
    ms_silu = time_fn(lambda: torch.nn.functional.silu(x))
    ms_sin = time_fn(lambda: torch.sin(x))

    print(f"{'Version':<35} {'Scalar':>9} {'Tensor':>9} {'Full pipe':>10} {'Speedup':>8}")
    print("-" * 75)
    print(f"{'Original C19':<35} {ms_orig_scalar:>8.3f}ms {ms_orig_tensor:>8.3f}ms {ms_full_orig:>9.3f}ms {'1.0x':>8}")
    print(f"{'V2 (int bit trick + fused where)':<35} {ms_v2_scalar:>8.3f}ms {ms_v2_tensor:>8.3f}ms {'—':>10} {ms_orig_tensor/ms_v2_tensor:>7.1f}x")
    print(f"{'V3 (frac + int bit + fused where)':<35} {ms_v3_scalar:>8.3f}ms {ms_v3_tensor:>8.3f}ms {ms_full_v3:>9.3f}ms {ms_full_orig/ms_full_v3:>7.1f}x")
    print(f"{'V4 Snake (sin²-based, ~3 ops)':<35} {ms_v4_scalar:>8.3f}ms {ms_v4_tensor:>8.3f}ms {ms_full_v4:>9.3f}ms {ms_full_orig/ms_full_v4:>7.1f}x")
    print("-" * 75)
    print(f"{'ReLU (reference)':<35} {ms_relu:>8.4f}ms")
    print(f"{'GELU (reference)':<35} {ms_gelu:>8.4f}ms")
    print(f"{'SiLU (reference)':<35} {ms_silu:>8.4f}ms")
    print(f"{'torch.sin (reference)':<35} {ms_sin:>8.4f}ms")

    # ── Backward pass comparison ──
    print(f"\n{'─' * 65}")
    print("Backward pass (gradient computation):")
    print(f"{'─' * 65}")

    x_grad = x.clone().requires_grad_(True)
    rho_raw_g = rho_raw.clone().requires_grad_(True)
    C_raw_g = C_raw.clone().requires_grad_(True)

    def bwd_orig():
        x_g = x.clone().requires_grad_(True)
        rr = rho_raw.clone().requires_grad_(True)
        cc = C_raw.clone().requires_grad_(True)
        r = _rho_from_raw(rr)
        c = _C_from_raw(cc)
        out = _c19_activation(x_g, rho=r, C=c)
        out.sum().backward()

    def bwd_v3():
        x_g = x.clone().requires_grad_(True)
        rr = rho_raw.clone().requires_grad_(True)
        cc = C_raw.clone().requires_grad_(True)
        r = _rho_from_raw(rr)
        c = _C_from_raw(cc)
        out = _c19_v3(x_g, rho=r, C=c)
        out.sum().backward()

    def bwd_v4():
        x_g = x.clone().requires_grad_(True)
        rr = rho_raw.clone().requires_grad_(True)
        cc = C_raw.clone().requires_grad_(True)
        r = _rho_from_raw(rr)
        c = _C_from_raw(cc)
        out = _c19_v4_snake(x_g, rho=r, C=c)
        out.sum().backward()

    def bwd_gelu():
        x_g = x.clone().requires_grad_(True)
        out = torch.nn.functional.gelu(x_g)
        out.sum().backward()

    ms_bwd_orig = time_fn(bwd_orig, warmup=5, repeats=20)
    ms_bwd_v3 = time_fn(bwd_v3, warmup=5, repeats=20)
    ms_bwd_v4 = time_fn(bwd_v4, warmup=5, repeats=20)
    ms_bwd_gelu = time_fn(bwd_gelu, warmup=5, repeats=20)

    print(f"{'Original C19 backward':<35} {ms_bwd_orig:>8.3f}ms  {'1.0x':>8}")
    print(f"{'V3 backward':<35} {ms_bwd_v3:>8.3f}ms  {ms_bwd_orig/ms_bwd_v3:>7.1f}x")
    print(f"{'V4 Snake backward':<35} {ms_bwd_v4:>8.3f}ms  {ms_bwd_orig/ms_bwd_v4:>7.1f}x")
    print(f"{'GELU backward':<35} {ms_bwd_gelu:>8.3f}ms  {ms_bwd_orig/ms_bwd_gelu:>7.1f}x")

    # ── Impact estimate ──
    print(f"\n{'─' * 65}")
    print("Estimated full-model impact (C19 = 89% of compute):")
    print(f"{'─' * 65}")
    current_step = 1144  # ms from profiler
    c19_share = 0.89
    for name, speedup_fwd, speedup_bwd in [
        ("V3 (exact same math)", ms_full_orig / ms_full_v3, ms_bwd_orig / ms_bwd_v3),
        ("V4 Snake (sin²)", ms_full_orig / ms_full_v4, ms_bwd_orig / ms_bwd_v4),
    ]:
        avg_speedup = (speedup_fwd + speedup_bwd) / 2
        new_c19_time = c19_share / avg_speedup
        new_total = (1 - c19_share) + new_c19_time
        est_step = current_step * new_total
        print(f"  {name}: ~{avg_speedup:.1f}x C19 speedup → step {est_step:.0f}ms ({1000/est_step:.2f} step/s)")


if __name__ == '__main__':
    bench_all()
