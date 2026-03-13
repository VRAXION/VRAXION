"""Benchmark C19 variants: original vs optimized vs cos-based alternatives.

Tests:
1. Speed comparison (ops/sec)
2. Shape fidelity (how close to original)
3. Three key properties: bounded core, linear tails, periodic
4. A/B training comparison on echo task
"""

import sys, time, math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for subdir in ('model', 'training', 'datagen'):
    p = str(ROOT / subdir)
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import torch.nn.functional as F
import numpy as np

# Import original
from instnct import _c19_activation, _C19_C

# ═══════════════════════════════════════════════════════════════
#  VARIANT 1: Original C19 (baseline)
# ═══════════════════════════════════════════════════════════════
def c19_original(x, rho=4.0, C=None):
    """Original C19 — 8 element-wise ops."""
    if C is None:
        C = _C19_C
    l = 6.0 * C
    inv_c = 1.0 / C
    scaled = x * inv_c
    n = torch.floor(scaled)
    t = scaled - n
    h = t * (1.0 - t)
    is_even = torch.remainder(n, 2.0) < 1.0
    sgn = torch.where(is_even, torch.ones_like(x), -torch.ones_like(x))
    core = C * (sgn * h + (rho * h * h))
    return torch.where(x >= l, x - l, torch.where(x <= -l, x + l, core))


# ═══════════════════════════════════════════════════════════════
#  VARIANT 2: Optimized C19 (same math, fewer ops)
# ═══════════════════════════════════════════════════════════════
def c19_opt(x, rho=4.0, C=None):
    """Optimized C19 — frac, fused sign, fewer intermediates."""
    if C is None:
        C = _C19_C
    l = 6.0 * C
    inv_c = 1.0 / C
    scaled = x * inv_c
    t = torch.frac(scaled)                            # replaces floor + sub
    h = t * (1.0 - t)                                 # parabola [0, 0.25]
    # Sign: 1 - 2*(floor(scaled) % 2) — fused into single op
    sgn = 1.0 - 2.0 * (torch.floor(scaled) % 2)      # ±1
    core = C * (sgn * h + rho * (h * h))
    return torch.where(x.abs() > l, x - x.sign() * l, core)


# ═══════════════════════════════════════════════════════════════
#  VARIANT 3: Cos-based C19 (replaces parabolic arches with cos)
# ═══════════════════════════════════════════════════════════════
def c19_cos(x, rho=4.0, C=None):
    """Cos-based: same 3 properties, uses CUDA-optimized cos.

    cos(π·x/C) gives bounded periodic oscillation.
    0.5*(1 - cos(2π·x/C)) gives a bump like h = t(1-t).
    """
    if C is None:
        C = _C19_C
    l = 6.0 * C
    s = math.pi * x / C                             # phase
    # Alternating component (like sgn * h):
    wave = 0.25 * torch.sin(s)                       # bounded ±0.25, period 2C
    # Rectified component (like rho * h²):
    bump = 0.125 * rho * (1.0 - torch.cos(2.0 * s))  # always ≥0, period C
    core = C * (wave + bump)
    return torch.where(x.abs() > l, x - x.sign() * l, core)


# ═══════════════════════════════════════════════════════════════
#  VARIANT 4: Simplified C19 (drop rho, single learnable C)
# ═══════════════════════════════════════════════════════════════
def c19_simple(x, rho=4.0, C=None):
    """Minimal: just the periodic bounded core + linear tails.
    Uses cos for the wave, single amplitude param (rho).
    2 trig ops instead of floor+remainder+where chain.
    """
    if C is None:
        C = _C19_C
    l = 6.0 * C
    s = math.pi * x / C
    # Single smooth periodic bounded function with DC offset from rho
    core = C * 0.25 * (torch.sin(s) + rho * torch.sin(s).square())
    return torch.where(x.abs() > l, x - x.sign() * l, core)


# ═══════════════════════════════════════════════════════════════
#  VARIANT 5: Triangle wave (cheapest periodic, non-smooth)
# ═══════════════════════════════════════════════════════════════
def c19_triangle(x, rho=4.0, C=None):
    """Triangle wave: periodic, bounded, cheapest possible.
    Uses only frac + abs. Not smooth (kinks at peaks).
    """
    if C is None:
        C = _C19_C
    l = 6.0 * C
    inv_2c = 0.5 / C
    # Triangle wave with period 2C, amplitude C/2
    t = torch.frac(x * inv_2c + 0.25)                # shift so peak at x=0
    tri = C * (2.0 * (2.0 * t - 1.0).abs() - 1.0)    # ±C triangle
    # Add rho bias (positive shift like in original)
    h_approx = 0.25 * (1.0 - (2.0 * t - 1.0).abs())  # tent [0, 0.25]
    core = tri * 0.25 + rho * C * h_approx * h_approx
    return torch.where(x.abs() > l, x - x.sign() * l, core)


# ═══════════════════════════════════════════════════════════════
#  BENCHMARKING
# ═══════════════════════════════════════════════════════════════

def benchmark_fn(fn, x, rho, C, n_iter=1000, warmup=100):
    """Benchmark a single activation function."""
    # Warmup
    for _ in range(warmup):
        _ = fn(x, rho=rho, C=C)

    torch.cuda.synchronize() if x.is_cuda else None
    t0 = time.perf_counter()
    for _ in range(n_iter):
        _ = fn(x, rho=rho, C=C)
    torch.cuda.synchronize() if x.is_cuda else None
    elapsed = time.perf_counter() - t0
    return elapsed / n_iter


def check_properties(fn, name, rho=4.0, C=math.pi):
    """Verify the three key properties."""
    x = torch.linspace(-25, 25, 10000)
    y = fn(x, rho=rho, C=C)

    # 1. Bounded core: check max |y| for |x| < 6C
    mask_core = x.abs() < 6 * C
    core_max = y[mask_core].abs().max().item()
    bounded = core_max < 6 * C  # should be bounded

    # 2. Linear tails: check slope ≈ 1 for |x| > 6C
    mask_tail = x > 6 * C + 1
    if mask_tail.sum() > 2:
        x_tail = x[mask_tail]
        y_tail = y[mask_tail]
        slope = (y_tail[-1] - y_tail[0]) / (x_tail[-1] - x_tail[0])
        linear = abs(slope.item() - 1.0) < 0.01
    else:
        linear = False

    # 3. Periodic: check that core has at least 2 zero crossings
    core_y = y[mask_core]
    signs = (core_y[1:] * core_y[:-1]) < 0
    n_crossings = signs.sum().item()
    periodic = n_crossings >= 2

    return bounded, linear, periodic, core_max, n_crossings


def shape_fidelity(fn, rho=4.0, C=math.pi):
    """MSE vs original C19 in the core region."""
    x = torch.linspace(-6 * C, 6 * C, 10000)
    y_orig = c19_original(x, rho=rho, C=C)
    y_new = fn(x, rho=rho, C=C)
    mse = ((y_orig - y_new) ** 2).mean().item()
    max_diff = (y_orig - y_new).abs().max().item()
    return mse, max_diff


def gradient_check(fn, rho=4.0, C=math.pi):
    """Check gradient magnitude (proxy for trainability)."""
    x = torch.linspace(-10, 10, 1000, requires_grad=True)
    rho_t = torch.tensor(rho)
    C_t = torch.tensor(C)
    y = fn(x, rho=rho_t, C=C_t)
    y.sum().backward()
    grad_mean = x.grad.abs().mean().item()
    grad_max = x.grad.abs().max().item()
    zero_grad_frac = (x.grad.abs() < 1e-6).float().mean().item()
    return grad_mean, grad_max, zero_grad_frac


def main():
    variants = [
        ("C19 original", c19_original),
        ("C19 optimized", c19_opt),
        ("C19 cos-based", c19_cos),
        ("C19 simplified", c19_simple),
        ("C19 triangle", c19_triangle),
    ]

    # Also benchmark raw ops for reference
    print("=" * 78)
    print("  C19 ACTIVATION BENCHMARK & ANALYSIS")
    print("=" * 78)

    # ── Speed benchmark ──
    sizes = [(128, 128), (128, 2048), (64, 128)]  # (B*T, hidden_dim) combos
    rho = 4.0
    C = math.pi

    print(f"\n  SPEED (μs per call, averaged over 1000 iterations)")
    print(f"  {'Variant':<20}", end="")
    for b, h in sizes:
        print(f" {f'{b}×{h}':>12}", end="")
    print(f" {'Speedup':>10}")
    print(f"  {'-'*20}", end="")
    for _ in sizes:
        print(f" {'-'*12}", end="")
    print(f" {'-'*10}")

    baseline_times = {}
    for name, fn in variants:
        print(f"  {name:<20}", end="")
        times = []
        for b, h in sizes:
            x = torch.randn(b, h)
            t = benchmark_fn(fn, x, rho, C, n_iter=1000)
            times.append(t)
            print(f" {t*1e6:>10.1f}μs", end="")
        avg_time = np.mean(times)
        if name == "C19 original":
            baseline_times = {s: t for s, t in zip(sizes, times)}
            print(f" {'1.00x':>10}")
        else:
            avg_speedup = np.mean([baseline_times[s] / t for s, t in zip(sizes, times)])
            print(f" {avg_speedup:>9.2f}x")

    # ── Properties check ──
    print(f"\n  THREE KEY PROPERTIES")
    print(f"  {'Variant':<20} {'Bounded':>8} {'Linear':>8} {'Periodic':>9} "
          f"{'CoreMax':>8} {'Crossings':>10}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*9} {'-'*8} {'-'*10}")

    for name, fn in variants:
        bounded, linear, periodic, core_max, crossings = check_properties(fn, name)
        b_str = "✓" if bounded else "✗"
        l_str = "✓" if linear else "✗"
        p_str = "✓" if periodic else "✗"
        print(f"  {name:<20} {b_str:>8} {l_str:>8} {p_str:>9} {core_max:>8.3f} {crossings:>10}")

    # ── Shape fidelity ──
    print(f"\n  SHAPE FIDELITY vs ORIGINAL")
    print(f"  {'Variant':<20} {'MSE':>12} {'Max Diff':>12}")
    print(f"  {'-'*20} {'-'*12} {'-'*12}")
    for name, fn in variants:
        if fn is c19_original:
            print(f"  {name:<20} {'—':>12} {'—':>12}")
            continue
        mse, max_diff = shape_fidelity(fn)
        print(f"  {name:<20} {mse:>12.6f} {max_diff:>12.6f}")

    # ── Gradient properties ──
    print(f"\n  GRADIENT PROPERTIES (x ∈ [-10, 10])")
    print(f"  {'Variant':<20} {'Mean|∇|':>10} {'Max|∇|':>10} {'%ZeroGrad':>10}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10}")
    for name, fn in variants:
        g_mean, g_max, g_zero = gradient_check(fn)
        print(f"  {name:<20} {g_mean:>10.4f} {g_max:>10.4f} {g_zero*100:>9.1f}%")

    print()
    print("=" * 78)

    # ── Quick A/B training test ──
    print("\n  QUICK A/B TRAINING (200 steps, echo task)")
    print("=" * 78)

    import instnct as instnct_module
    from instnct import INSTNCT

    _original_c19 = instnct_module._c19_activation

    MODEL_CFG = dict(
        M=64, hidden_dim=128, slot_dim=32, N=1, R=1,
        embed_mode=True, kernel_mode='vshape', pointer_mode='pilot',
        write_mode='replace', embed_encoding='bitlift',
        output_encoding='lowrank_c19', checkpoint_chunks=0,
    )
    BATCH, SEQ_LEN, SEED, LR = 8, 64, 42, 1e-3
    STEPS = 200

    # Make echo batch
    rng_np = np.random.RandomState(SEED)
    BLOCK, REPEAT = 16, 4
    n_bytes = BATCH * (SEQ_LEN + 1) + BLOCK * REPEAT * 4
    raw_data, raw_mask = [], []
    while len(raw_data) < n_bytes:
        sb = rng_np.randint(0, 256, size=BLOCK, dtype=np.uint8)
        for r in range(REPEAT):
            raw_data.extend(sb)
            raw_mask.extend([0] * BLOCK if r == 0 else [1] * BLOCK)
    raw_data = np.array(raw_data[:n_bytes], dtype=np.uint8)
    raw_mask = np.array(raw_mask[:n_bytes], dtype=np.uint8)
    x_np = np.zeros((BATCH, SEQ_LEN), dtype=np.int64)
    y_np = np.zeros((BATCH, SEQ_LEN), dtype=np.int64)
    mask_np = np.zeros((BATCH, SEQ_LEN), dtype=np.float32)
    for i in range(BATCH):
        off = i * SEQ_LEN
        x_np[i] = raw_data[off:off + SEQ_LEN]
        y_np[i] = raw_data[off + 1:off + SEQ_LEN + 1]
        mask_np[i] = raw_mask[off + 1:off + SEQ_LEN + 1]
    x = torch.from_numpy(x_np)
    y = torch.from_numpy(y_np)
    mask = torch.from_numpy(mask_np)

    # Test only the variants that pass all 3 properties
    train_variants = [
        ("C19 original", c19_original),
        ("C19 optimized", c19_opt),
        ("C19 cos-based", c19_cos),
        ("C19 simplified", c19_simple),
    ]

    results = []
    for vname, vfn in train_variants:
        instnct_module._c19_activation = vfn
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        model = INSTNCT(**MODEL_CFG)
        opt = torch.optim.Adam(model.parameters(), lr=LR)

        t0 = time.perf_counter()
        best_acc = 0.0
        step_90 = None
        for step in range(1, STEPS + 1):
            model.train()
            out, _ = model(x)
            logits = out.view(-1, 256)
            targets = y.view(-1)
            m_flat = mask.view(-1)
            ce = F.cross_entropy(logits, targets, reduction='none')
            loss = (ce * m_flat).sum() / m_flat.sum() if m_flat.sum() > 0 else ce.mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                acc = ((preds == targets).float() * m_flat).sum() / m_flat.sum()
                acc_val = acc.item()
                best_acc = max(best_acc, acc_val)
                if acc_val >= 0.9 and step_90 is None:
                    step_90 = step
            if step in (1, 50, 100, 150, 200):
                elapsed = time.perf_counter() - t0
                print(f"  [{vname:>16s}] step {step:4d}  loss={loss.item():.4f}  "
                      f"acc={acc_val*100:.1f}%  [{elapsed:.1f}s]")

        elapsed = time.perf_counter() - t0
        results.append({
            'name': vname,
            'final_acc': acc_val,
            'best_acc': best_acc,
            'step_90': step_90,
            'elapsed': elapsed,
            'final_loss': loss.item(),
        })

    instnct_module._c19_activation = _original_c19

    print()
    print(f"  {'Variant':<20} {'Final Acc':>10} {'Best Acc':>10} {'→90%':>6} {'Time':>8} {'Speedup':>8}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*6} {'-'*8} {'-'*8}")
    base_time = results[0]['elapsed']
    for r in results:
        s90 = str(r['step_90']) if r['step_90'] else '—'
        speedup = base_time / r['elapsed']
        print(f"  {r['name']:<20} {r['final_acc']*100:>9.1f}% {r['best_acc']*100:>9.1f}% "
              f"{s90:>6} {r['elapsed']:>7.1f}s {speedup:>7.2f}x")

    print("=" * 78)


if __name__ == '__main__':
    main()
