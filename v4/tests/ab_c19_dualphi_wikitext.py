"""A/B Test: C19 baseline vs dual-phi on WikiText-103 (GPU, real data).

Tests whether the dual-phi interference filter improves byte-level
language modeling accuracy on real text data.

Variant A (baseline): original C19 activation
  core = C * (sgn * h + rho * h^2)

Variant B (dual-phi): asymmetric phi scaling on arches
  gain = odd * (phi - 1/phi) + 1/phi   (even->1/phi, odd->phi)
  core = C * h * (sgn + rho * h) * gain

Both variants use rho=4.0, C=pi.
Everything else identical: model arch, data, optimizer, seed.

Usage:
    python tests/ab_c19_dualphi_wikitext.py [--steps 1000] [--seeds 3]
"""

import argparse
import copy
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

# ── Path setup ──
V4_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V4_ROOT / 'model'))
sys.path.insert(0, str(V4_ROOT / 'training'))

from train import ByteDataset, func_discover_dat, func_maskloss_ce
from model_factory import build_model_from_spec

# ── C19 variants ──
PHI = (1 + math.sqrt(5)) / 2
PHI_INV = (math.sqrt(5) - 1) / 2
C19_C = math.pi


def c19_baseline(x, rho=4.0, C=None):
    """Original C19: symmetric arches."""
    if C is None:
        C = C19_C
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


def c19_dualphi(x, rho=4.0, C=None):
    """Dual-phi C19: asymmetric phi scaling on arches."""
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
    gain = odd * (PHI - PHI_INV) + PHI_INV
    core = C * h * (sgn + rho * h) * gain
    return torch.where(x.abs() > l, x - x.sign() * l, core)


def patch_activation(model, act_fn):
    """Replace the C19 activation function inside the model's expert layers."""
    # The model uses _c19_activation via inp layer (c19 module) and output layers
    # We monkey-patch the module's forward to use our variant
    import instnct
    instnct._c19_activation = act_fn


def build_model(seed):
    """Build a fresh INSTNCT model with current config."""
    torch.manual_seed(seed)
    spec = {
        'M': 1024,
        'embed_dim': None,
        'hidden_dim': 2048,
        'slot_dim': 128,
        'N': 1,
        'R': 1,
        'B': 8,
        'embed_mode': True,
        'kernel_mode': 'vshape',
        'checkpoint_chunks': 0,
        'expert_weighting': False,
        'embed_encoding': 'learned',
        'output_encoding': 'learned',
        'pointer_mode': 'sequential',
        'write_mode': 'replace',
        'bb_enabled': False,
        'bb_gate_bias': 0.0,
        'bb_scale': 0.1,
        'bb_tau': 4.0,
        'bb_gate_mode': 'learned',
        'topk_K': 8,
        's_constraint': 'softplus',
    }
    record = {'type': 'instnct', 'build_spec': spec}
    return build_model_from_spec(record, 'cuda')


def run_one(variant_name, act_fn, dataset, steps, batch_size, seq_len, seed, use_amp=True):
    """Train one variant, return metrics dict."""
    import instnct
    # Save original and patch
    orig_fn = instnct._c19_activation
    instnct._c19_activation = act_fn

    model = build_model(seed)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    losses = []
    accs = []
    t0 = time.time()

    for step in range(1, steps + 1):
        xb, yb, mask = dataset.sample_batch(batch_size, 'cuda')

        with torch.amp.autocast('cuda', enabled=use_amp):
            pred, _state = model(xb, state=None)
            _, masked_loss = func_maskloss_ce(pred, yb, mask)

        opt.zero_grad()
        scaler.scale(masked_loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        scaler.step(opt)
        scaler.update()

        lv = masked_loss.item()
        losses.append(lv)

        # Accuracy: argmax match on supervised positions
        with torch.no_grad():
            preds = pred.argmax(dim=-1)  # (B, T)
            correct = (preds == yb).float() * mask
            acc = correct.sum() / mask.sum().clamp(min=1)
            accs.append(acc.item())

        if step % 100 == 0 or step == 1:
            avg_loss = sum(losses[-100:]) / len(losses[-100:])
            avg_acc = sum(accs[-100:]) / len(accs[-100:])
            elapsed = time.time() - t0
            print(f'  [{variant_name}] step {step:4d}/{steps}  '
                  f'loss={avg_loss:.4f}  bpc={avg_loss*1.4427:.3f}  '
                  f'acc={avg_acc:.3f}  '
                  f'{elapsed:.0f}s')

    elapsed = time.time() - t0

    # Restore original
    instnct._c19_activation = orig_fn

    # Final metrics: average of last 100 steps
    tail = min(100, len(losses))
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
        'loss_curve': losses,
        'acc_curve': accs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--seeds', type=int, default=2)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--seq', type=int, default=256)
    args = parser.parse_args()

    print(f'=== C19 Baseline vs Dual-Phi A/B Test ===')
    print(f'Steps: {args.steps}  Seeds: {args.seeds}  Batch: {args.batch}x{args.seq}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print()

    # Load data
    data_dir = V4_ROOT / 'training_data'
    files = func_discover_dat(str(data_dir))
    dataset = ByteDataset(files, args.seq, embed_mode=True, seed=42)
    print(f'Data: {len(files)} shards, {dataset.total_bytes / 1e6:.0f} MB')
    print()

    variants = [
        ('baseline', c19_baseline),
        ('dual-phi', c19_dualphi),
    ]

    all_results = []

    for seed_idx in range(args.seeds):
        seed = 42 + seed_idx * 1000
        print(f'--- Seed {seed} ---')
        for name, fn in variants:
            # Reset dataset RNG for fair comparison
            dataset.rng = np.random.default_rng(seed)
            result = run_one(name, fn, dataset, args.steps, args.batch, args.seq, seed)
            all_results.append(result)
            print(f'  -> {name}: loss={result["final_loss"]:.4f} '
                  f'bpc={result["final_bpc"]:.3f} '
                  f'acc={result["final_acc"]:.3f} '
                  f'best_acc={result["best_acc"]:.3f} '
                  f'({result["time_s"]:.0f}s)')
        print()

    # Summary
    print('=' * 70)
    print(f'{"Variant":12s} {"Avg Loss":>10s} {"Avg BPC":>10s} {"Avg Acc":>10s} {"Best Acc":>10s} {"Time":>8s}')
    print('-' * 70)
    for name, _ in variants:
        runs = [r for r in all_results if r['variant'] == name]
        avg_loss = sum(r['final_loss'] for r in runs) / len(runs)
        avg_bpc = sum(r['final_bpc'] for r in runs) / len(runs)
        avg_acc = sum(r['final_acc'] for r in runs) / len(runs)
        best_acc = max(r['best_acc'] for r in runs)
        avg_time = sum(r['time_s'] for r in runs) / len(runs)
        print(f'{name:12s} {avg_loss:10.4f} {avg_bpc:10.3f} {avg_acc:10.3f} {best_acc:10.3f} {avg_time:7.0f}s')

    # Delta
    base_runs = [r for r in all_results if r['variant'] == 'baseline']
    phi_runs = [r for r in all_results if r['variant'] == 'dual-phi']
    base_acc = sum(r['final_acc'] for r in base_runs) / len(base_runs)
    phi_acc = sum(r['final_acc'] for r in phi_runs) / len(phi_runs)
    delta = (phi_acc - base_acc) * 100
    winner = 'dual-phi' if delta > 0 else 'baseline'
    print(f'\nDelta: {delta:+.2f}% accuracy  -> {winner} wins')
    print('=' * 70)


if __name__ == '__main__':
    main()
