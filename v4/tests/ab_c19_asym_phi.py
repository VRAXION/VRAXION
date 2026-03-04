"""A/B test: C19 asymmetric negative gain — phi vs 2.0 vs original.

Theory being tested:
- asym neg=2.0 already proven faster+stabler than original (step 121 vs 128 →90%)
- phi ≈ 1.618 might be the "ultimate" asymmetry ratio because:
  1. phi is the eigenvalue of Fibonacci recurrence — natural growth constant
  2. phi is the "most irrational" number — maximally async with itself
  3. In recurrent loops, phi-ratio asymmetry = minimal resonance/interference
  4. The VRAXION architecture already uses phi (0.618) everywhere

Variants tested:
  1. C19 original       — symmetric, baseline
  2. C19 asym neg=1.5   — mild asymmetry
  3. C19 asym neg=phi    — golden ratio asymmetry (1.618...)
  4. C19 asym neg=2.0   — proven winner from previous test
  5. C19 asym neg=phi²  — 2.618... (beyond 2.0, phi-scaled)
  6. C19 asym neg=3.0   — aggressive (had spikes before)

How asymmetry works:
  The negative half of each parabolic arch is multiplied by neg_gain.
  Positive arches stay at gain=1.0 ("anchor").
  This boosts error gradients (negative signal) without destabilizing
  the recurrent loop (positive signal stays bounded).

Usage: python v4/tests/ab_c19_asym_phi.py [--steps 500]
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
from instnct import INSTNCT, _C19_C

_original_c19 = instnct_module._c19_activation

PHI = (1.0 + math.sqrt(5)) / 2.0  # 1.6180339887...

# ═══════════════════════════════════════════════════════════════
#  C19 with asymmetric negative gain
# ═══════════════════════════════════════════════════════════════

def _make_c19_asym(neg_gain):
    """Factory: create C19 variant where negative arches are scaled by neg_gain."""
    def _c19_asym(x, rho=4.0, C=None):
        if C is None:
            C = _C19_C
        l = 6.0 * C
        inv_c = 1.0 / C
        scaled = x * inv_c
        n = torch.floor(scaled)
        t = scaled - n
        h = t - t * t
        sgn = 1.0 - 2.0 * torch.remainder(n, 2.0)
        core = C * h * (sgn + rho * h)
        # Asymmetric boost: scale negative values by neg_gain
        core = torch.where(core < 0, core * neg_gain, core)
        return torch.where(x.abs() > l, x - x.sign() * l, core)
    _c19_asym.__name__ = f'c19_asym_{neg_gain:.4f}'
    return _c19_asym


# ═══════════════════════════════════════════════════════════════
#  Config
# ═══════════════════════════════════════════════════════════════

BATCH = 8
SEQ_LEN = 64
LR = 1e-3
MAX_STEPS = 500
LOG_EVERY = 50
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
    gnorm_history = []
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

        # Track gradient norm BEFORE clipping
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        gnorm_history.append(total_norm)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()

        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            acc = ((preds == targets).float() * m_flat).sum() / m_flat.sum()

        lv = loss.item()
        history.append({'step': step, 'loss': lv, 'acc': acc.item(), 'gnorm': total_norm})

        if step <= 5 or step % LOG_EVERY == 0 or step == max_steps:
            elapsed = time.perf_counter() - t0
            print(f"  [{name:>18s}] step {step:5d}  loss={lv:.4f}  "
                  f"acc={acc.item()*100:5.1f}%  gnorm={total_norm:.1f}  [{elapsed:.1f}s]")

        if math.isnan(lv) or lv > 50:
            if step > 50:
                diverged = True
                print(f"  [{name:>18s}] DIVERGED at step {step} (loss={lv:.2f})")
                break

    elapsed = time.perf_counter() - t0
    instnct_module._c19_activation = _original_c19

    # Gradient norm stats
    gnorms = np.array(gnorm_history)
    gnorm_mean = gnorms.mean()
    gnorm_max = gnorms.max()
    gnorm_std = gnorms.std()
    spike_count = (gnorms > 100).sum()

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
        'steps_to_100': next((h['step'] for h in history if h['acc'] >= 1.00), None),
        'gnorm_mean': gnorm_mean,
        'gnorm_max': gnorm_max,
        'gnorm_std': gnorm_std,
        'gnorm_spikes': int(spike_count),
    }


def run():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=MAX_STEPS)
    args = parser.parse_args()

    device = 'cpu'
    print("=" * 78)
    print("  A/B Test: C19 ASYMMETRIC NEGATIVE GAIN — phi vs 2.0")
    print(f"  phi = {PHI:.10f}")
    print(f"  phi² = {PHI**2:.10f}")
    print(f"  Config: B={BATCH}, T={SEQ_LEN}, M={MODEL_CFG['M']}, "
          f"H={MODEL_CFG['hidden_dim']}, slot={MODEL_CFG['slot_dim']}")
    print(f"  Task: Echo (byte-level, embed_mode, CrossEntropy)")
    print(f"  Steps: {args.steps}, LR: {LR}, Seed: {SEED}")
    print("=" * 78)

    x, y, mask = make_echo_batch(BATCH, SEQ_LEN, device)
    sup_pct = mask.mean().item() * 100
    print(f"  Data: {BATCH}x{SEQ_LEN} bytes, supervised={sup_pct:.1f}%\n")

    variants = [
        ('original (1.0)',  _original_c19),
        ('asym neg=1.5',    _make_c19_asym(1.5)),
        (f'asym neg=phi',   _make_c19_asym(PHI)),
        ('asym neg=2.0',    _make_c19_asym(2.0)),
        (f'asym neg=phi²',  _make_c19_asym(PHI ** 2)),
        ('asym neg=3.0',    _make_c19_asym(3.0)),
    ]

    results = []
    for name, fn in variants:
        print(f"{'─' * 78}")
        print(f"  Training: {name}")
        print(f"{'─' * 78}")
        r = train_one(name, fn, x, y, mask, args.steps, device)
        results.append(r)
        print()

    # ═══════════════════════════════════════════════════════════════
    #  Summary table
    # ═══════════════════════════════════════════════════════════════
    print("=" * 78)
    print("  RESULTS — ASYMMETRIC NEGATIVE GAIN: phi vs 2.0")
    print("=" * 78)
    print(f"  {'Variant':<20} {'→90%':>6} {'→95%':>6} {'→100%':>7} {'Best':>8} "
          f"{'gnorm':>10} {'gnorm_max':>10} {'spikes':>7} {'Status':>8}")
    print(f"  {'─'*20} {'─'*6} {'─'*6} {'─'*7} {'─'*8} "
          f"{'─'*10} {'─'*10} {'─'*7} {'─'*8}")

    for r in results:
        s90 = str(r['steps_to_90']) if r['steps_to_90'] else '—'
        s95 = str(r['steps_to_95']) if r['steps_to_95'] else '—'
        s100 = str(r['steps_to_100']) if r['steps_to_100'] else '—'
        status = 'DIVERGED' if r['diverged'] else 'OK'
        print(f"  {r['name']:<20} {s90:>6} {s95:>6} {s100:>7} {r['best_acc']*100:>7.1f}% "
              f"{r['gnorm_mean']:>10.1f} {r['gnorm_max']:>10.1f} {r['gnorm_spikes']:>7} {status:>8}")

    print()
    print("=" * 78)
    print("  LEARNING CURVES (accuracy %)")
    print("=" * 78)
    checkpoints = [25, 50, 75, 100, 125, 150, 200, 300, 400, args.steps]
    checkpoints = [c for c in checkpoints if c <= args.steps]
    header = f"  {'Step':>6}" + "".join(f" {r['name']:>18}" for r in results)
    print(header)
    for cp in checkpoints:
        vals = []
        for r in results:
            if cp <= len(r['history']):
                h = r['history'][cp - 1]
                vals.append(f" {h['acc']*100:>17.1f}%")
            else:
                vals.append(f" {'(stopped)':>18}")
        print(f"  {cp:>6}" + "".join(vals))

    print()
    print("=" * 78)
    print("  GRADIENT NORM STABILITY")
    print("=" * 78)
    print(f"  {'Variant':<20} {'mean':>8} {'std':>8} {'max':>10} {'spikes>100':>12}")
    print(f"  {'─'*20} {'─'*8} {'─'*8} {'─'*10} {'─'*12}")
    for r in results:
        print(f"  {r['name']:<20} {r['gnorm_mean']:>8.2f} {r['gnorm_std']:>8.2f} "
              f"{r['gnorm_max']:>10.1f} {r['gnorm_spikes']:>12}")

    # Verdict
    print()
    print("=" * 78)
    print("  VERDICT")
    print("=" * 78)
    stable = [r for r in results if not r['diverged'] and r['gnorm_spikes'] == 0]
    if stable:
        fastest = min(stable, key=lambda r: r['steps_to_90'] or 9999)
        print(f"  Fastest to 90%: {fastest['name']} (step {fastest['steps_to_90']})")
        fastest100 = min(stable, key=lambda r: r['steps_to_100'] or 9999)
        if fastest100['steps_to_100']:
            print(f"  Fastest to 100%: {fastest100['name']} (step {fastest100['steps_to_100']})")
        most_stable = min(stable, key=lambda r: r['gnorm_max'])
        print(f"  Most stable: {most_stable['name']} (gnorm_max={most_stable['gnorm_max']:.1f})")
    print("=" * 78)


if __name__ == '__main__':
    run()
