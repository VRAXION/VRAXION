"""Microbench dual-phi fixed-C activation implementations on GPU.

Tests whether the current fixed-C dual-phi used by the WikiText sweep still
has measurable implementation headroom before touching the full training loop.

Variants:
  - current: current sweep implementation
  - gain_v2: same math, but exploit phi - 1/phi == 1
  - bitwise_v3: use integer parity instead of torch.remainder
"""

from __future__ import annotations

import math
import time

import torch

PHI = (1 + math.sqrt(5)) / 2
PHI_INV = 1 / PHI
C_VALUE = math.pi

B = 32
H = 2048
WARMUP = 20
REPEATS = 100


def dualphi_current(x):
    c = C_VALUE
    limit = 6.0 * c
    scaled = x / c
    n = torch.floor(scaled)
    t = scaled - n
    h = t - t * t
    odd = torch.remainder(n, 2.0)
    sgn = 1.0 - 2.0 * odd
    gain = odd * (PHI - PHI_INV) + PHI_INV
    core = c * h * (sgn + 4.0 * h) * gain
    return torch.where(x.abs() > limit, x - x.sign() * limit, core)


def dualphi_gain_v2(x):
    c = C_VALUE
    limit = 6.0 * c
    scaled = x / c
    n = torch.floor(scaled)
    t = scaled - n
    h = t - t * t
    odd = torch.remainder(n, 2.0)
    sgn = 1.0 - 2.0 * odd
    gain = PHI_INV + odd  # because phi - 1/phi == 1
    core = c * h * (sgn + 4.0 * h) * gain
    return torch.where(x.abs() > limit, x - x.sign() * limit, core)


def dualphi_bitwise_v3(x):
    c = C_VALUE
    limit = 6.0 * c
    scaled = x / c
    n = torch.floor(scaled)
    t = scaled - n
    h = t - t * t
    odd = (n.to(torch.int64) & 1).to(x.dtype)
    sgn = 1.0 - 2.0 * odd
    gain = PHI_INV + odd
    core = c * h * (sgn + 4.0 * h) * gain
    return torch.where(x.abs() > limit, x - x.sign() * limit, core)


def time_cuda(fn, x, backward=False):
    for _ in range(WARMUP):
        inp = x.clone().requires_grad_(backward)
        out = fn(inp)
        if backward:
            out.sum().backward()
    torch.cuda.synchronize()

    times = []
    for _ in range(REPEATS):
        inp = x.clone().requires_grad_(backward)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn(inp)
        if backward:
            out.sum().backward()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return sum(times) / len(times)


def main():
    if not torch.cuda.is_available():
        raise SystemExit('CUDA required')

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    x = torch.randn(B, H, device='cuda') * 4.0

    ref = dualphi_current(x)
    for name, fn in (
        ('gain_v2', dualphi_gain_v2),
        ('bitwise_v3', dualphi_bitwise_v3),
    ):
        out = fn(x)
        err = (ref - out).abs().max().item()
        print(f'max diff current vs {name}: {err:.3e}')

    rows = []
    for name, fn in (
        ('current', dualphi_current),
        ('gain_v2', dualphi_gain_v2),
        ('bitwise_v3', dualphi_bitwise_v3),
    ):
        fwd = time_cuda(fn, x, backward=False)
        bwd = time_cuda(fn, x, backward=True)
        rows.append((name, fwd, bwd))

    base_fwd = rows[0][1]
    base_bwd = rows[0][2]
    print()
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Shape: ({B}, {H})  repeats={REPEATS}')
    print(f'{"Variant":12s} {"Forward":>10s} {"Backward":>10s} {"Fwd speedup":>12s} {"Bwd speedup":>12s}')
    print('-' * 62)
    for name, fwd, bwd in rows:
        print(
            f'{name:12s} {fwd:9.3f}ms {bwd:9.3f}ms '
            f'{base_fwd / fwd:11.3f}x {base_bwd / bwd:11.3f}x'
        )


if __name__ == '__main__':
    main()
