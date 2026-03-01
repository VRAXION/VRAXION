"""Output Activation Sweep — tests activation functions in the low-rank output head.

Fixed: r=64 low-rank factored output (Linear(H→64) → activation → Linear(64→256))
Sweep: activation function between the two layers (none, relu, tanh, c19, silu)

Baseline comparison: r=64 without activation = 43.52% (3-seed mean)

Usage:
    python sweep_output_activation.py
    python sweep_output_activation.py --activations relu tanh
    python sweep_output_activation.py --rank 128 --seeds 42 137 2024
"""

import argparse
import csv
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Path setup ────────────────────────────────────────────────
_TRAINING_DIR = Path(__file__).resolve().parent          # v4/training/
_MODEL_DIR = Path(__file__).resolve().parent.parent / 'model'  # v4/model/
_V4_ROOT = Path(__file__).resolve().parent.parent        # v4/

for d in (_TRAINING_DIR, _MODEL_DIR):
    ds = str(d)
    if ds not in sys.path:
        sys.path.insert(0, ds)

from instnct import INSTNCT  # type: ignore[import-not-found]  # noqa: E402
from train import (  # type: ignore[import-not-found]  # noqa: E402
    ByteDataset,
    func_discover_dat,
    func_maskloss_ce,
    func_accuracy_emb,
    _compute_lr,
)


# ── Activation functions ──────────────────────────────────────

def _c19(x, rho=4.0):
    l = 6.0 * math.pi
    inv_pi = 1.0 / math.pi
    scaled = x * inv_pi
    n = torch.floor(scaled)
    t = scaled - n
    h = t * (1.0 - t)
    is_even = torch.remainder(n, 2.0) < 1.0
    sgn = torch.where(is_even, torch.ones_like(x), -torch.ones_like(x))
    core = math.pi * (sgn * h + (rho * h * h))
    return torch.where(x >= l, x - l, torch.where(x <= -l, x + l, core))


class C19Module(nn.Module):
    """Fixed c19 activation (rho=4.0)."""
    def forward(self, x):
        return _c19(x)


class C19Learnable(nn.Module):
    """C19 with learnable rho, bounded via sigmoid to [rho_min, rho_max]."""
    def __init__(self, width=64, rho_init=4.0, rho_min=0.5, rho_max=8.0, per_neuron=True):
        super().__init__()
        self.rho_min = rho_min
        self.rho_max = rho_max
        shape = (width,) if per_neuron else (1,)
        # Inverse sigmoid to initialize at rho_init
        p = (rho_init - rho_min) / (rho_max - rho_min)
        p = min(max(p, 1e-4), 1 - 1e-4)
        raw0 = torch.log(torch.tensor(p / (1 - p), dtype=torch.float32))
        self.raw_rho = nn.Parameter(raw0.expand(shape).clone())

    def _rho(self):
        return self.rho_min + (self.rho_max - self.rho_min) * torch.sigmoid(self.raw_rho)

    def forward(self, x):
        rho = self._rho()
        l = 6.0 * math.pi
        inv_pi = 1.0 / math.pi
        scaled = x * inv_pi
        n = torch.floor(scaled)
        t = scaled - n
        h = t * (1.0 - t)
        is_even = torch.remainder(n, 2.0) < 1.0
        sgn = torch.where(is_even, torch.ones_like(x), -torch.ones_like(x))
        core = math.pi * (sgn * h + (rho * h * h))
        return torch.where(x >= l, x - l, torch.where(x <= -l, x + l, core))


ACTIVATION_FNS = {
    'none':  None,
    'relu':  nn.ReLU(),
    'tanh':  nn.Tanh(),
    'c19':   None,   # handled below
    'c19_learn': None,  # handled below
    'silu':  nn.SiLU(),
}


def _build_output_head(hidden_dim, rank, activation_name, device):
    """Build low-rank output head with optional activation."""
    layers = [nn.Linear(hidden_dim, rank)]

    if activation_name == 'c19':
        layers.append(C19Module())
    elif activation_name == 'c19_learn':
        layers.append(C19Learnable(width=rank, per_neuron=True))
    elif activation_name != 'none':
        layers.append(ACTIVATION_FNS[activation_name])

    layers.append(nn.Linear(rank, 256))
    return nn.Sequential(*layers).to(device)


# ── Single run ────────────────────────────────────────────────

def run_one(activation_name, rank, steps, batch_size, seq_len, lr_base, device, dataset, seed=42):
    """Train one model with output activation for `steps` steps."""

    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed(seed)

    model = INSTNCT(
        embed_mode=True,
        embed_encoding='learned',
        output_encoding='learned',
    ).to(device)

    hidden_dim = model.hidden_dim
    model.out = _build_output_head(hidden_dim, rank, activation_name, device)

    n_params = sum(p.numel() for p in model.parameters())
    out_params = sum(p.numel() for p in model.out.parameters())

    print(f'  params: {n_params:,}  (output: {out_params:,})')

    optimizer = torch.optim.Adam(model.parameters(), lr=lr_base)
    model.train()

    warmup = 100
    best_loss = float('inf')
    last_loss = 0.0
    last_acc = 0.0
    t0 = time.perf_counter()

    for step in range(1, steps + 1):
        lr = _compute_lr(lr_base, step, warmup, steps)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        xb, yb, mask = dataset.sample_batch(batch_size, device)
        pred, _ = model(xb)
        raw_loss, masked_loss = func_maskloss_ce(pred, yb, mask)

        optimizer.zero_grad()
        masked_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()

        if masked_loss.item() < best_loss:
            best_loss = masked_loss.item()

        if step % 100 == 0:
            raw_acc, masked_acc = func_accuracy_emb(pred, yb, mask)
            elapsed = time.perf_counter() - t0
            print(f'    step {step:>5} | loss {masked_loss.item():.4f} | '
                  f'acc {masked_acc*100:.1f}% | {elapsed:.0f}s')
            last_loss = masked_loss.item()
            last_acc = masked_acc

    if steps % 100 != 0:
        raw_acc, masked_acc = func_accuracy_emb(pred, yb, mask)
        last_loss = masked_loss.item()
        last_acc = masked_acc

    elapsed = time.perf_counter() - t0

    return {
        'activation': activation_name,
        'rank': rank,
        'seed': seed,
        'total_params': n_params,
        'out_params': out_params,
        'best_loss': best_loss,
        'final_loss': last_loss,
        'final_acc': last_acc,
        'elapsed': elapsed,
    }


# ── CLI ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Output activation sweep — test activations in the low-rank output head.',
    )
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--seq', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--rank', type=int, default=64,
                        help='output rank (default: 64)')
    parser.add_argument('--activations', nargs='+',
                        default=['none', 'relu', 'tanh', 'c19'],
                        help='activations to test (default: none relu tanh c19)')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42])
    parser.add_argument('--data', default=None)
    args = parser.parse_args()

    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_path = Path(args.data) if args.data else _V4_ROOT / 'training_data'
    if not data_path.is_absolute():
        data_path = _V4_ROOT / data_path

    activations = args.activations
    seeds = args.seeds
    n_seeds = len(seeds)

    print(f'VRAXION v4 — Output Activation Sweep')
    print(f'{"=" * 60}')
    print(f'Rank:        {args.rank}')
    print(f'Activations: {activations}')
    print(f'Seeds:       {seeds} ({n_seeds} per activation)')
    print(f'Steps:       {args.steps}')
    print(f'Batch:       {args.batch}')
    print(f'Seq:         {args.seq}')
    print(f'LR:          {args.lr}')
    print(f'Device:      {args.device}')
    print(f'Total runs:  {len(activations) * n_seeds}')
    print()

    file_pairs = func_discover_dat(str(data_path))
    dataset = ByteDataset(file_pairs, args.seq, embed_mode=True)
    print(f'Data: {len(file_pairs)} file(s), {dataset.total_bytes / 1024**2:.1f} MB')
    print()

    all_results = []
    for act in activations:
        for s in seeds:
            print(f'=== {act}, Seed {s} {"=" * 40}')
            metrics = run_one(act, args.rank, args.steps, args.batch, args.seq,
                              args.lr, args.device, dataset, seed=s)
            all_results.append(metrics)
            print(f'  DONE — loss: {metrics["final_loss"]:.4f}, '
                  f'acc: {metrics["final_acc"]*100:.1f}%, '
                  f'time: {metrics["elapsed"]:.0f}s')
            print()

    # ── Summary ──
    print()
    print(f'{"=" * 72}')
    print(f'  OUTPUT ACTIVATION SWEEP  (r={args.rank}, {args.steps} steps, {n_seeds} seed(s))')
    print(f'{"=" * 72}')
    print(f'{"Activation":>12} {"Out Params":>11} {"Loss":>8} {"Acc%":>8} {"Time":>6}')
    print(f'{"-" * 72}')
    for m in all_results:
        print(f'{m["activation"]:>12} {m["out_params"]:>11,} '
              f'{m["final_loss"]:>8.4f} {m["final_acc"]*100:>7.1f}% '
              f'{m["elapsed"]:>5.0f}s')
    print(f'{"=" * 72}')

    # ── Save CSV ──
    csv_path = _V4_ROOT / 'training_output' / 'sweep_output_activation.csv'
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=[
            'activation', 'rank', 'seed', 'out_params', 'total_params',
            'best_loss', 'final_loss', 'final_acc', 'elapsed', 'timestamp',
        ])
        w.writeheader()
        ts = datetime.now(timezone.utc).isoformat()
        for m in all_results:
            w.writerow({**m, 'timestamp': ts})
    print(f'\nCSV saved: {csv_path}')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n[INTERRUPTED] Sweep stopped.')
        sys.exit(0)
    except Exception as e:
        print(f'\n[FATAL] {type(e).__name__}: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)
