"""C19 variant benchmark v2 — fixed bugs from v1.

Key fixes:
- c19_opt: use (scaled - floor) instead of frac (negative number fix)
- c19_cos: match original's amplitude/DC-offset exactly
- Added c19_cos_matched: properly scaled cos approximation
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

from instnct import _c19_activation, _C19_C
import instnct as instnct_module

SEED = 42


# ═══════════════════════════════════════════════════════════
#  VARIANT 0: Original (reference)
# ═══════════════════════════════════════════════════════════
def c19_original(x, rho=4.0, C=None):
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


# ═══════════════════════════════════════════════════════════
#  VARIANT 1: Optimized (same math, fewer ops, frac bug fixed)
# ═══════════════════════════════════════════════════════════
def c19_opt(x, rho=4.0, C=None):
    """Optimized: merged where, proper floor-based t."""
    if C is None:
        C = _C19_C
    l = 6.0 * C
    inv_c = 1.0 / C
    scaled = x * inv_c
    n = torch.floor(scaled)
    t = scaled - n                                      # [0, 1) always correct
    h = t - t * t                                       # t*(1-t) = t - t²
    sgn = 1.0 - 2.0 * torch.remainder(n, 2.0)          # ±1 fused
    core = C * h * (sgn + rho * h)                      # factor out h: C*h*(sgn + rho*h)
    return torch.where(x.abs() > l, x - x.sign() * l, core)


# ═══════════════════════════════════════════════════════════
#  VARIANT 2: Cos-matched (cos approximation with correct scaling)
# ═══════════════════════════════════════════════════════════
def c19_cos_matched(x, rho=4.0, C=None):
    """Cos-based with amplitude matched to original C19.

    Original peak analysis:
    - Even half: C * (h + rho*h²), peak at t=0.5: C*(0.25 + rho*0.0625)
    - Odd half:  C * (-h + rho*h²), peak at t=0.5: C*(-0.25 + rho*0.0625)

    We use cos(πx/C) for the alternating part and (1-cos(2πx/C))/2 for the
    always-positive bump, but scaled to match the original's amplitudes.
    """
    if C is None:
        C = _C19_C
    l = 6.0 * C
    phase = math.pi * x / C
    # Match original amplitudes:
    # Original h peaks at 0.25, h² peaks at 0.0625
    # cos: -cos(phase) oscillates [-1,1] with period 2C (like sgn)
    #       (1-cos(2*phase))/2 oscillates [0,1] with period C (like h²)
    # Scale: h peak = 0.25, so multiply wave by 0.25
    #        h² peak = 0.0625, so multiply bump by 0.0625
    wave = -0.25 * torch.cos(phase)                    # [-0.25, +0.25]
    bump = 0.0625 * rho * (1.0 - torch.cos(2.0 * phase))  # [0, 0.125*rho]
    core = C * (wave + bump)
    return torch.where(x.abs() > l, x - x.sign() * l, core)


# ═══════════════════════════════════════════════════════════
#  VARIANT 3: Cos-lite (only cos, no bump — minimal)
# ═══════════════════════════════════════════════════════════
def c19_cos_lite(x, rho=4.0, C=None):
    """Minimal cos: single trig op + positive bias from rho.

    Uses cos(πx/C) for periodicity, adds rho-scaled DC shift.
    Preserves: bounded core, linear tails, periodic.
    """
    if C is None:
        C = _C19_C
    l = 6.0 * C
    phase = math.pi * x / C
    # Shifted cosine: rho controls how much positive bias
    # At rho=4: center ≈ 0.5*C, amplitude ≈ 0.25*C
    center = C * rho * 0.0625                           # DC offset (= original's bump at peak)
    amp = C * 0.25                                      # wave amplitude (= original's h peak)
    core = center - amp * torch.cos(phase)              # bounded [center-amp, center+amp]
    return torch.where(x.abs() > l, x - x.sign() * l, core)


# ═══════════════════════════════════════════════════════════
#  VARIANT 4: Compile-wrapped original
# ═══════════════════════════════════════════════════════════
try:
    c19_compiled = torch.compile(c19_original, mode='reduce-overhead')
    HAS_COMPILE = True
except Exception:
    c19_compiled = c19_original
    HAS_COMPILE = False


# ═══════════════════════════════════════════════════════════
#  BENCHMARKING
# ═══════════════════════════════════════════════════════════

def benchmark_fn(fn, x, rho, C, n_iter=2000, warmup=200):
    for _ in range(warmup):
        _ = fn(x, rho=rho, C=C)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        _ = fn(x, rho=rho, C=C)
    return (time.perf_counter() - t0) / n_iter


def check_properties(fn, rho=4.0, C=math.pi):
    x = torch.linspace(-25, 25, 10000)
    y = fn(x, rho=rho, C=C)
    mask_core = x.abs() < 6 * C
    core_max = y[mask_core].abs().max().item()
    bounded = core_max < 6 * C
    mask_tail = x > 6 * C + 1
    if mask_tail.sum() > 2:
        x_t, y_t = x[mask_tail], y[mask_tail]
        slope = (y_t[-1] - y_t[0]) / (x_t[-1] - x_t[0])
        linear = abs(slope.item() - 1.0) < 0.01
    else:
        linear = False
    core_y = y[mask_core]
    crossings = ((core_y[1:] * core_y[:-1]) < 0).sum().item()
    periodic = crossings >= 2
    return bounded, linear, periodic, core_max


def shape_mse(fn, rho=4.0, C=math.pi):
    x = torch.linspace(-6 * C, 6 * C, 10000)
    y_orig = c19_original(x, rho=rho, C=C)
    y_new = fn(x, rho=rho, C=C)
    return ((y_orig - y_new) ** 2).mean().item(), (y_orig - y_new).abs().max().item()


def make_echo_batch(batch=8, seq_len=64, seed=42):
    BLOCK, REPEAT = 16, 4
    rng = np.random.RandomState(seed)
    n_bytes = batch * (seq_len + 1) + BLOCK * REPEAT * 4
    raw_data, raw_mask = [], []
    while len(raw_data) < n_bytes:
        sb = rng.randint(0, 256, size=BLOCK, dtype=np.uint8)
        for r in range(REPEAT):
            raw_data.extend(sb)
            raw_mask.extend([0] * BLOCK if r == 0 else [1] * BLOCK)
    raw_data = np.array(raw_data[:n_bytes], dtype=np.uint8)
    raw_mask = np.array(raw_mask[:n_bytes], dtype=np.uint8)
    x_np = np.zeros((batch, seq_len), dtype=np.int64)
    y_np = np.zeros((batch, seq_len), dtype=np.int64)
    mask_np = np.zeros((batch, seq_len), dtype=np.float32)
    for i in range(batch):
        off = i * seq_len
        x_np[i] = raw_data[off:off + seq_len]
        y_np[i] = raw_data[off + 1:off + seq_len + 1]
        mask_np[i] = raw_mask[off + 1:off + seq_len + 1]
    return (torch.from_numpy(x_np), torch.from_numpy(y_np),
            torch.from_numpy(mask_np))


def train_variant(name, fn, x, y, mask, steps=200):
    from instnct import INSTNCT
    instnct_module._c19_activation = fn

    MODEL_CFG = dict(
        M=64, hidden_dim=128, slot_dim=32, N=1, R=1,
        embed_mode=True, kernel_mode='vshape', pointer_mode='pilot',
        write_mode='replace', embed_encoding='bitlift',
        output_encoding='lowrank_c19', checkpoint_chunks=0,
    )

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = INSTNCT(**MODEL_CFG)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    t0 = time.perf_counter()
    best_acc = 0.0
    step_90 = None

    for step in range(1, steps + 1):
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
        if step in (1, 25, 50, 100, 150, 200):
            elapsed = time.perf_counter() - t0
            print(f"  [{name:>18s}] step {step:4d}  loss={loss.item():.4f}  "
                  f"acc={acc_val*100:.1f}%  [{elapsed:.1f}s]")

    elapsed = time.perf_counter() - t0
    instnct_module._c19_activation = _c19_activation
    return {
        'name': name, 'final_acc': acc_val, 'best_acc': best_acc,
        'step_90': step_90, 'elapsed': elapsed, 'final_loss': loss.item(),
    }


def main():
    rho, C = 4.0, math.pi

    variants = [
        ("C19 original",    c19_original),
        ("C19 optimized",   c19_opt),
        ("C19 cos-matched", c19_cos_matched),
        ("C19 cos-lite",    c19_cos_lite),
    ]
    if HAS_COMPILE:
        variants.append(("C19 compiled", c19_compiled))

    print("=" * 78)
    print("  C19 VARIANT BENCHMARK v2 (bugs fixed)")
    print("=" * 78)

    # ── Properties ──
    print(f"\n  PROPERTIES CHECK")
    print(f"  {'Variant':<20} {'Bounded':>8} {'Linear':>8} {'Periodic':>9} {'CoreMax':>8}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*9} {'-'*8}")
    for name, fn in variants:
        b, l, p, cm = check_properties(fn)
        print(f"  {name:<20} {'✓' if b else '✗':>8} {'✓' if l else '✗':>8} "
              f"{'✓' if p else '✗':>9} {cm:>8.3f}")

    # ── Shape fidelity ──
    print(f"\n  SHAPE FIDELITY vs ORIGINAL")
    print(f"  {'Variant':<20} {'MSE':>12} {'MaxDiff':>12}")
    print(f"  {'-'*20} {'-'*12} {'-'*12}")
    for name, fn in variants:
        if fn is c19_original:
            print(f"  {name:<20} {'—':>12} {'—':>12}")
            continue
        mse, md = shape_mse(fn)
        print(f"  {name:<20} {mse:>12.6f} {md:>12.6f}")

    # ── Speed ──
    print(f"\n  SPEED BENCHMARK (μs per call)")
    x_test = torch.randn(128, 128)
    print(f"  {'Variant':<20} {'Time':>10} {'Speedup':>10}")
    print(f"  {'-'*20} {'-'*10} {'-'*10}")
    base_t = None
    for name, fn in variants:
        t = benchmark_fn(fn, x_test, rho, C)
        if base_t is None:
            base_t = t
        print(f"  {name:<20} {t*1e6:>9.1f}μs {base_t/t:>9.2f}x")

    # ── Training A/B ──
    print(f"\n  TRAINING A/B (200 steps, echo task)")
    print("=" * 78)
    x, y, mask = make_echo_batch()

    results = []
    for name, fn in variants:
        if fn is c19_compiled:
            continue  # torch.compile + monkey-patch is tricky
        print()
        r = train_variant(name, fn, x, y, mask, steps=200)
        results.append(r)

    print()
    print(f"  {'Variant':<20} {'FinalAcc':>10} {'BestAcc':>10} {'→90%':>6} {'Time':>8} {'Speedup':>8}")
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
