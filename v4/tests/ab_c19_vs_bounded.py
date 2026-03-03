"""A/B test v2: C19 vs BOUNDED alternatives.

Round 1 showed unbounded activations (Snake, GELU, SiLU) diverge in the
recurrent hidden state loop. C19 works because it's BOUNDED — output
stays in a finite range regardless of input magnitude.

This test compares C19 against bounded alternatives that preserve the
key property (amplitude limiting) while being computationally cheaper:

  1. C19 (original)        — periodic parabolic, bounded, learnable rho+C
  2. tanh                  — classic bounded [-1,1], no learnable params
  3. Learnable tanh        — C * tanh(x/C), learnable C (period/scale)
  4. Bounded Snake         — tanh(x + rho/4 * sin²(C*x)), learnable rho+C
  5. Periodic tanh         — tanh(sin(C*x) * rho), learnable rho+C
  6. Softsign              — x / (1 + |x|), bounded [-1,1], no params
  7. C19-lite              — same math, but fixed rho+C (no learnable)

Usage: python v4/tests/ab_c19_vs_bounded.py [--steps 2000]
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
import torch.nn as nn
import torch.nn.functional as F

import instnct as instnct_module
from instnct import INSTNCT

_original_c19 = instnct_module._c19_activation

# ═══════════════════════════════════════════════════════════════
#  Bounded activation alternatives
# ═══════════════════════════════════════════════════════════════

def _tanh_activation(x, rho=4.0, C=None):
    """Classic tanh — bounded [-1, 1]. Ignores rho/C."""
    return torch.tanh(x)

def _learnable_tanh(x, rho=4.0, C=None):
    """Learnable tanh: C * tanh(x / C).
    C controls the linear region width. Large C → more linear.
    Uses rho as amplitude: rho * tanh(x / C).
    Both rho and C are learnable (same param count as C19)."""
    if C is None:
        C = math.pi
    return rho * torch.tanh(x / C)

def _bounded_snake(x, rho=4.0, C=None):
    """Bounded Snake: tanh(x + rho/4 * sin²(C*x)).
    Snake periodic component + tanh bounding. Learnable rho + C."""
    if C is None:
        C = math.pi
    s = torch.sin(C * x)
    return torch.tanh(x + (rho * 0.25) * (s * s))

def _periodic_tanh(x, rho=4.0, C=None):
    """Periodic tanh: rho * tanh(sin(C*x)).
    Periodic like C19, bounded by tanh. Learnable rho + C."""
    if C is None:
        C = math.pi
    return rho * torch.tanh(torch.sin(C * x))

def _softsign_activation(x, rho=4.0, C=None):
    """Softsign: x / (1 + |x|). Bounded [-1,1], lighter tail than tanh."""
    return x / (1 + x.abs())

def _c19_fixed(x, rho=4.0, C=None):
    """C19 with FIXED rho=4.0, C=π — no learnable activation params."""
    return instnct_module._c19_activation(x, rho=4.0, C=math.pi)

def _hardtanh_activation(x, rho=4.0, C=None):
    """Hardtanh: clamp to [-rho, rho]. Learnable rho (amplitude bound)."""
    return torch.clamp(x, -rho, rho)

# ═══════════════════════════════════════════════════════════════
#  Config — matches round 1
# ═══════════════════════════════════════════════════════════════

BATCH = 8
SEQ_LEN = 64
LR = 1e-3
MAX_STEPS = 2000
LOG_EVERY = 200
SEED = 42

MODEL_CFG = dict(
    M=64, hidden_dim=128, slot_dim=32, N=1, R=1,
    embed_mode=True,
    kernel_mode='vshape',
    pointer_mode='pilot',
    write_mode='replace',
    embed_encoding='bitlift',
    output_encoding='lowrank_c19',
    checkpoint_chunks=0,
)


def make_echo_batch(batch, seq_len, device, seed=42):
    BLOCK = 16
    REPEAT = 4
    rng = np.random.RandomState(seed)
    n_bytes = batch * (seq_len + 1) + BLOCK * REPEAT * 4
    raw_data, raw_mask = [], []
    while len(raw_data) < n_bytes:
        seed_block = rng.randint(0, 256, size=BLOCK, dtype=np.uint8)
        for r in range(REPEAT):
            raw_data.extend(seed_block)
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
    return (torch.from_numpy(x_np).to(device),
            torch.from_numpy(y_np).to(device),
            torch.from_numpy(mask_np).to(device))


def train_one(name, activation_fn, x, y, mask, max_steps, device):
    instnct_module._c19_activation = activation_fn
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = INSTNCT(**MODEL_CFG).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    history = []
    t0 = time.perf_counter()
    diverged = False

    for step in range(1, max_steps + 1):
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

        lv = loss.item()
        history.append({'step': step, 'loss': lv, 'acc': acc.item()})

        if step <= 5 or step % LOG_EVERY == 0 or step == max_steps:
            elapsed = time.perf_counter() - t0
            print(f"  [{name:>14s}] step {step:5d}  loss={lv:.4f}  "
                  f"acc={acc.item()*100:5.1f}%  [{elapsed:.1f}s]")

        # Early divergence detection
        if math.isnan(lv) or lv > 50:
            if step > 100:
                diverged = True
                print(f"  [{name:>14s}] DIVERGED at step {step} (loss={lv:.2f})")
                break

    elapsed = time.perf_counter() - t0
    instnct_module._c19_activation = _original_c19

    return {
        'name': name,
        'params': n_params,
        'history': history,
        'final_loss': history[-1]['loss'],
        'final_acc': history[-1]['acc'],
        'best_acc': max(h['acc'] for h in history),
        'elapsed': elapsed,
        'diverged': diverged,
        'steps_to_90': next((h['step'] for h in history if h['acc'] >= 0.90), None),
        'steps_to_95': next((h['step'] for h in history if h['acc'] >= 0.95), None),
        'steps_to_99': next((h['step'] for h in history if h['acc'] >= 0.99), None),
    }


def bench_activations():
    """Quick speed benchmark of each activation."""
    x = torch.randn(8, 128)
    from instnct import _rho_from_raw, _C_from_raw, _rho_init_raw, _C_init_raw
    rho_raw = torch.full((128,), _rho_init_raw(4.0))
    C_raw = torch.full((128,), _C_init_raw())
    rho = _rho_from_raw(rho_raw)
    C_val = _C_from_raw(C_raw)

    activations = [
        ('C19', _original_c19),
        ('tanh', _tanh_activation),
        ('Learnable tanh', _learnable_tanh),
        ('Bounded Snake', _bounded_snake),
        ('Periodic tanh', _periodic_tanh),
        ('Softsign', _softsign_activation),
        ('C19-fixed', _c19_fixed),
        ('Hardtanh', _hardtanh_activation),
    ]

    print(f"{'Activation':<20} {'Time (μs)':>10} {'vs C19':>8}")
    print("-" * 40)
    times = {}
    for name, fn in activations:
        # Warmup
        for _ in range(20):
            fn(x, rho=rho, C=C_val)
        t_list = []
        for _ in range(100):
            t0 = time.perf_counter()
            fn(x, rho=rho, C=C_val)
            t_list.append((time.perf_counter() - t0) * 1e6)
        avg = sum(t_list) / len(t_list)
        times[name] = avg

    c19_time = times['C19']
    for name, _ in activations:
        speedup = c19_time / times[name]
        print(f"{name:<20} {times[name]:>9.1f}μs {speedup:>7.1f}x")


def run():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=MAX_STEPS)
    args = parser.parse_args()

    device = 'cpu'
    print("=" * 70)
    print("A/B Test v2: C19 vs BOUNDED Alternatives")
    print(f"Config: B={BATCH}, T={SEQ_LEN}, M={MODEL_CFG['M']}, "
          f"H={MODEL_CFG['hidden_dim']}, slot={MODEL_CFG['slot_dim']}")
    print(f"Task: Echo (byte-level, embed_mode, CrossEntropy)")
    print(f"Steps: {args.steps}, LR: {LR}, Seed: {SEED}")
    print("=" * 70)

    # Speed benchmark first
    print("\nActivation speed benchmark:")
    bench_activations()
    print()

    x, y, mask = make_echo_batch(BATCH, SEQ_LEN, device)
    sup_pct = mask.mean().item() * 100
    print(f"Data: {BATCH}x{SEQ_LEN} bytes, supervised={sup_pct:.1f}%\n")

    variants = [
        ('C19', _original_c19),
        ('tanh', _tanh_activation),
        ('Learn-tanh', _learnable_tanh),
        ('Bound-Snake', _bounded_snake),
        ('Period-tanh', _periodic_tanh),
        ('Softsign', _softsign_activation),
        ('C19-fixed', _c19_fixed),
        ('Hardtanh', _hardtanh_activation),
    ]

    results = []
    for name, fn in variants:
        print(f"{'─' * 70}")
        print(f"Training: {name}")
        print(f"{'─' * 70}")
        r = train_one(name, fn, x, y, mask, args.steps, device)
        results.append(r)
        print()

    # Summary
    print("=" * 70)
    print("  A/B v2 RESULTS — BOUNDED ACTIVATIONS")
    print("=" * 70)
    c19_time = results[0]['elapsed']
    print(f"{'Activation':<16} {'Final Acc':>9} {'Best Acc':>9} {'→90%':>6} {'→99%':>6} "
          f"{'Time':>7} {'Speed':>7} {'Status':>10}")
    print("-" * 70)
    for r in results:
        s90 = str(r['steps_to_90']) if r['steps_to_90'] else '—'
        s99 = str(r['steps_to_99']) if r['steps_to_99'] else '—'
        speedup = f"{c19_time / r['elapsed']:.1f}x"
        status = 'DIVERGED' if r['diverged'] else 'OK'
        print(f"{r['name']:<16} {r['final_acc']*100:>8.1f}% {r['best_acc']*100:>8.1f}% "
              f"{s90:>6} {s99:>6} {r['elapsed']:>6.1f}s {speedup:>7} {status:>10}")
    print("=" * 70)

    # Learning curve comparison
    print("\nLearning curves (accuracy %):")
    checkpoints = [50, 100, 200, 500, 1000, args.steps]
    checkpoints = [c for c in checkpoints if c <= args.steps]
    header = f"{'Step':>6}" + "".join(f"  {r['name']:>14}" for r in results)
    print(header)
    for cp in checkpoints:
        vals = []
        for r in results:
            if cp <= len(r['history']):
                h = r['history'][cp - 1]
                vals.append(f"  {h['acc']*100:>13.1f}%")
            else:
                vals.append(f"  {'(stopped)':>14}")
        print(f"{cp:>6}" + "".join(vals))


if __name__ == '__main__':
    run()
