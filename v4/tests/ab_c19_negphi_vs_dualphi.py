"""A/B Test: neg*phi only vs dual-phi on WikiText-103 (GPU, real data).

Variant A (neg-phi): only negative arches scaled by phi, positive arches unscaled
  gain = odd * phi + (1-odd) * 1.0 = odd*(phi-1) + 1

Variant B (dual-phi): both sides scaled — neg*phi, pos*1/phi
  gain = odd * (phi - 1/phi) + 1/phi

Same setup as previous A/B: 500 steps, seed 42, batch 32x256, from scratch.

Usage:
    python tests/ab_c19_negphi_vs_dualphi.py
"""

import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

V4_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V4_ROOT / 'model'))
sys.path.insert(0, str(V4_ROOT / 'training'))

from train import ByteDataset, func_discover_dat, func_maskloss_ce
from model_factory import build_model_from_spec

PHI = (1 + math.sqrt(5)) / 2
PHI_INV = (math.sqrt(5) - 1) / 2
C19_C = math.pi


def c19_negphi(x, rho=4.0, C=None):
    """Neg-phi only: negative arches scaled by phi, positive arches unscaled (gain=1)."""
    if C is None:
        C = C19_C
    l = 6.0 * C
    inv_c = 1.0 / C
    scaled = x * inv_c
    n = torch.floor(scaled)
    t = scaled - n
    h = t - t * t
    odd = torch.remainder(n, 2.0)
    sgn = 1.0 - 2.0 * odd
    gain = odd * (PHI - 1.0) + 1.0  # odd -> phi, even -> 1.0
    core = C * h * (sgn + rho * h) * gain
    return torch.where(x.abs() > l, x - x.sign() * l, core)


def c19_dualphi(x, rho=4.0, C=None):
    """Dual-phi: neg*phi, pos*1/phi."""
    if C is None:
        C = C19_C
    l = 6.0 * C
    inv_c = 1.0 / C
    scaled = x * inv_c
    n = torch.floor(scaled)
    t = scaled - n
    h = t - t * t
    odd = torch.remainder(n, 2.0)
    sgn = 1.0 - 2.0 * odd
    gain = odd * (PHI - PHI_INV) + PHI_INV  # odd -> phi, even -> 1/phi
    core = C * h * (sgn + rho * h) * gain
    return torch.where(x.abs() > l, x - x.sign() * l, core)


def build_model(seed):
    torch.manual_seed(seed)
    spec = {
        'M': 1024, 'embed_dim': None, 'hidden_dim': 2048, 'slot_dim': 128,
        'N': 1, 'R': 1, 'B': 8, 'embed_mode': True,
        'kernel_mode': 'vshape', 'checkpoint_chunks': 0,
        'expert_weighting': False, 'embed_encoding': 'learned',
        'output_encoding': 'learned', 'pointer_mode': 'sequential',
        'write_mode': 'replace', 'bb_enabled': False,
        'bb_gate_bias': 0.0, 'bb_scale': 0.1, 'bb_tau': 4.0,
        'bb_gate_mode': 'learned', 'topk_K': 8,
    }
    record = {'type': 'instnct', 'build_spec': spec}
    return build_model_from_spec(record, 'cuda')


def run_one(variant_name, act_fn, dataset, steps, batch_size, seed, use_amp=True):
    import instnct
    orig_fn = instnct._c19_activation
    instnct._c19_activation = act_fn

    model = build_model(seed)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    losses = []
    accs = []
    grad_norms = []
    max_grad = 0.0
    t0 = time.time()

    for step in range(1, steps + 1):
        xb, yb, mask = dataset.sample_batch(batch_size, 'cuda')

        with torch.amp.autocast('cuda', enabled=use_amp):
            pred, _state = model(xb, state=None)
            _, masked_loss = func_maskloss_ce(pred, yb, mask)

        opt.zero_grad()
        scaler.scale(masked_loss).backward()
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0).item()
        scaler.step(opt)
        scaler.update()

        lv = masked_loss.item()
        losses.append(lv)
        grad_norms.append(gn)
        if gn > max_grad:
            max_grad = gn

        with torch.no_grad():
            preds = pred.argmax(dim=-1)
            correct = (preds == yb).float() * mask
            acc = correct.sum() / mask.sum().clamp(min=1)
            accs.append(acc.item())

        if step % 100 == 0 or step == 1:
            avg_loss = sum(losses[-100:]) / len(losses[-100:])
            avg_acc = sum(accs[-100:]) / len(accs[-100:])
            avg_gn = sum(grad_norms[-100:]) / len(grad_norms[-100:])
            elapsed = time.time() - t0
            spike = '*SPIKE*' if max(grad_norms[-100:]) > 50 else ''
            print(f'  [{variant_name}] step {step:4d}/{steps}  '
                  f'loss={avg_loss:.4f}  bpc={avg_loss*1.4427:.3f}  '
                  f'acc={avg_acc:.3f}  gnorm={avg_gn:.1f}  '
                  f'{elapsed:.0f}s {spike}')

    elapsed = time.time() - t0
    instnct._c19_activation = orig_fn

    tail = min(100, len(losses))
    spikes = sum(1 for g in grad_norms if g > 50)
    return {
        'variant': variant_name,
        'seed': seed,
        'steps': steps,
        'params': n_params,
        'final_loss': sum(losses[-tail:]) / tail,
        'final_bpc': sum(losses[-tail:]) / tail * 1.4427,
        'final_acc': sum(accs[-tail:]) / tail,
        'best_loss': min(losses),
        'best_acc': max(accs),
        'time_s': elapsed,
        's_per_step': elapsed / steps,
        'max_grad': max_grad,
        'grad_spikes': spikes,
        'loss_curve': losses,
        'acc_curve': accs,
    }


def main():
    steps = 500
    batch = 32
    seq = 256
    seed = 42

    print(f'=== Neg-Phi vs Dual-Phi A/B Test ===')
    print(f'Steps: {steps}  Seed: {seed}  Batch: {batch}x{seq}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print()

    data_dir = V4_ROOT / 'training_data'
    files = func_discover_dat(str(data_dir))
    dataset = ByteDataset(files, seq, embed_mode=True, seed=42)
    print(f'Data: {len(files)} shards, {dataset.total_bytes / 1e6:.0f} MB')
    print()

    variants = [
        ('neg-phi', c19_negphi),
        ('dual-phi', c19_dualphi),
    ]

    results = []
    for name, fn in variants:
        dataset.rng = np.random.default_rng(seed)
        r = run_one(name, fn, dataset, steps, batch, seed)
        results.append(r)
        print(f'  -> {name}: loss={r["final_loss"]:.4f} '
              f'bpc={r["final_bpc"]:.3f} '
              f'acc={r["final_acc"]:.3f} '
              f'best_acc={r["best_acc"]:.3f} '
              f'max_gnorm={r["max_grad"]:.1f} '
              f'spikes={r["grad_spikes"]} '
              f'({r["time_s"]:.0f}s)')
        print()

    # Summary
    print('=' * 75)
    print(f'{"Variant":12s} {"Final Acc":>10s} {"Best Acc":>10s} {"Final Loss":>11s} '
          f'{"BPC":>8s} {"Time":>8s} {"MaxGrad":>8s} {"Spikes":>7s}')
    print('-' * 75)
    for r in results:
        print(f'{r["variant"]:12s} {r["final_acc"]:10.3f} {r["best_acc"]:10.3f} '
              f'{r["final_loss"]:11.4f} {r["final_bpc"]:8.3f} '
              f'{r["time_s"]:7.0f}s {r["max_grad"]:8.1f} {r["grad_spikes"]:7d}')

    # Crossover analysis
    r_a, r_b = results
    la, lb = r_a['loss_curve'], r_b['loss_curve']
    # Find where B first leads consistently (10-step window)
    lead_step = None
    for i in range(10, len(la)):
        wa = sum(la[i-10:i]) / 10
        wb = sum(lb[i-10:i]) / 10
        if wb < wa:
            lead_step = i
            break

    delta = (r_b['final_acc'] - r_a['final_acc']) * 100
    winner = r_b['variant'] if delta > 0 else r_a['variant']
    print(f'\nDelta: {delta:+.2f}% accuracy -> {winner} wins')
    if lead_step:
        print(f'Crossover: {winner} took the lead around step {lead_step}')
    else:
        print(f'No clear crossover detected')

    # Stability
    for r in results:
        if r['grad_spikes'] > 0:
            print(f'  {r["variant"]}: {r["grad_spikes"]} gradient spikes (gnorm > 50)')
        else:
            print(f'  {r["variant"]}: no gradient spikes (stable)')
    print('=' * 75)


if __name__ == '__main__':
    main()
