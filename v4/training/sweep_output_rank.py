"""Output Rank Sweep — tests low-rank factored output head.

For each rank r in [8, 16, 32, 64, 128, 256]:
  - Creates INSTNCT with learned input + factored output
  - Trains for 1000 steps on wikitext
  - Records loss & accuracy

Output layer:
  r < 256: nn.Sequential(Linear(H→r), Linear(r→256))  — low-rank
  r = 256: nn.Linear(H→256)                           — full rank (baseline)

Usage:
    python sweep_output_rank.py
    python sweep_output_rank.py --steps 500 --device cpu
    python sweep_output_rank.py --ranks 32 64 128
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


# ── Single run ────────────────────────────────────────────────

def run_one(rank, steps, batch_size, seq_len, lr_base, device, dataset, seed=42):
    """Train one model with output rank=r for `steps` steps."""

    # Seed everything for reproducibility across seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed(seed)

    # Create model: learned input + learned output (override YAML defaults)
    model = INSTNCT(
        embed_mode=True,
        embed_encoding='learned',
        output_encoding='learned',
    ).to(device)

    # Replace output layer with low-rank factored version
    hidden_dim = model.hidden_dim
    if rank < 256:
        model.out = nn.Sequential(
            nn.Linear(hidden_dim, rank),
            nn.Linear(rank, 256),
        ).to(device)
    # else: keep the default nn.Linear(H, 256)

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
        # LR schedule: warmup + cosine decay
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

        # Progress heartbeat
        if step % 100 == 0:
            raw_acc, masked_acc = func_accuracy_emb(pred, yb, mask)
            elapsed = time.perf_counter() - t0
            print(f'    step {step:>5} | loss {masked_loss.item():.4f} | '
                  f'acc {masked_acc*100:.1f}% | {elapsed:.0f}s')
            last_loss = masked_loss.item()
            last_acc = masked_acc

    # Final metrics (in case steps isn't a multiple of 100)
    if steps % 100 != 0:
        raw_acc, masked_acc = func_accuracy_emb(pred, yb, mask)
        last_loss = masked_loss.item()
        last_acc = masked_acc

    elapsed = time.perf_counter() - t0

    return {
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
        description='Output rank sweep — find the sweet spot for low-rank output head.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--steps', type=int, default=1000,
                        help='training steps per rank (default: 1000)')
    parser.add_argument('--device', default='cuda',
                        help='device (default: cuda)')
    parser.add_argument('--batch', type=int, default=256,
                        help='batch size (default: 256)')
    parser.add_argument('--seq', type=int, default=64,
                        help='sequence length (default: 64)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--ranks', type=int, nargs='+', default=[8, 16, 32, 64, 128, 256],
                        help='output ranks to sweep (default: 8 16 32 64 128 256)')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42],
                        help='random seeds to run per rank (default: 42)')
    parser.add_argument('--data', default=None,
                        help='data path (default: training_data/)')
    args = parser.parse_args()

    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_path = Path(args.data) if args.data else _V4_ROOT / 'training_data'
    if not data_path.is_absolute():
        data_path = _V4_ROOT / data_path

    ranks = sorted(args.ranks)

    seeds = args.seeds
    n_seeds = len(seeds)

    print(f'VRAXION v4 — Output Rank Sweep')
    print(f'{"=" * 60}')
    print(f'Ranks:  {ranks}')
    print(f'Seeds:  {seeds} ({n_seeds} per rank)')
    print(f'Steps:  {args.steps}')
    print(f'Batch:  {args.batch}')
    print(f'Seq:    {args.seq}')
    print(f'LR:     {args.lr}')
    print(f'Device: {args.device}')
    print(f'Input:  learned (baseline)')
    print(f'Output: low-rank factored (sweep variable)')
    print(f'Total runs: {len(ranks) * n_seeds}')
    print()

    # Load data once — shared across all runs
    file_pairs = func_discover_dat(str(data_path))
    dataset = ByteDataset(file_pairs, args.seq, embed_mode=True)
    print(f'Data: {len(file_pairs)} file(s), {dataset.total_bytes / 1024**2:.1f} MB')
    print()

    all_results = []
    for r in ranks:
        for s in seeds:
            print(f'=== Rank {r}, Seed {s} {"=" * 35}')
            metrics = run_one(r, args.steps, args.batch, args.seq, args.lr,
                              args.device, dataset, seed=s)
            all_results.append(metrics)
            print(f'  DONE — loss: {metrics["final_loss"]:.4f}, '
                  f'acc: {metrics["final_acc"]*100:.1f}%, '
                  f'time: {metrics["elapsed"]:.0f}s')
            print()

    # ── Per-rank aggregation ──
    # t-values for 95% CI: df=1→12.706, df=2→4.303, df=3→3.182, df=4→2.776
    _t95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
            6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}

    agg = {}
    for r in ranks:
        accs = [m['final_acc'] for m in all_results if m['rank'] == r]
        losses = [m['final_loss'] for m in all_results if m['rank'] == r]
        out_p = [m['out_params'] for m in all_results if m['rank'] == r][0]
        tot_p = [m['total_params'] for m in all_results if m['rank'] == r][0]

        mean_acc = sum(accs) / len(accs)
        mean_loss = sum(losses) / len(losses)

        if len(accs) >= 2:
            std_acc = (sum((a - mean_acc)**2 for a in accs) / (len(accs) - 1)) ** 0.5
            df = len(accs) - 1
            t_val = _t95.get(df, 2.0)
            ci_acc = t_val * std_acc / math.sqrt(len(accs))
        else:
            std_acc = 0.0
            ci_acc = 0.0

        agg[r] = {
            'rank': r, 'out_params': out_p, 'total_params': tot_p,
            'mean_acc': mean_acc, 'std_acc': std_acc, 'ci95_acc': ci_acc,
            'mean_loss': mean_loss, 'n_seeds': len(accs),
            'accs': accs,
        }

    # ── Summary table ──
    print()
    print(f'{"=" * 80}')
    print(f'  OUTPUT RANK SWEEP RESULTS  ({args.steps} steps, {n_seeds} seed(s) each)')
    print(f'{"=" * 80}')

    if n_seeds >= 2:
        print(f'{"Rank":>6} {"Out Params":>11} {"Mean Acc%":>10} {"Std":>7} '
              f'{"95% CI":>10} {"Mean Loss":>10} {"Seeds":>6}')
        print(f'{"-" * 80}')
        for r in ranks:
            a = agg[r]
            ci_str = f'±{a["ci95_acc"]*100:.2f}%'
            accs_str = ', '.join(f'{x*100:.1f}' for x in a['accs'])
            print(f'{a["rank"]:>6} {a["out_params"]:>11,} '
                  f'{a["mean_acc"]*100:>9.2f}% {a["std_acc"]*100:>6.2f}% '
                  f'{ci_str:>10} {a["mean_loss"]:>10.4f} {a["n_seeds"]:>6}')
            print(f'       runs: [{accs_str}]')
    else:
        print(f'{"Rank":>6} {"Out Params":>11} {"Total Params":>13} '
              f'{"Loss":>8} {"Acc%":>8} {"Time":>6}')
        print(f'{"-" * 80}')
        for m in all_results:
            print(f'{m["rank"]:>6} {m["out_params"]:>11,} {m["total_params"]:>13,} '
                  f'{m["final_loss"]:>8.4f} {m["final_acc"]*100:>7.1f}% '
                  f'{m["elapsed"]:>5.0f}s')
    print(f'{"=" * 80}')

    # ── Overlap check (if 2+ ranks with CI) ──
    if n_seeds >= 2 and len(ranks) >= 2:
        print()
        for i in range(len(ranks)):
            for j in range(i+1, len(ranks)):
                a1, a2 = agg[ranks[i]], agg[ranks[j]]
                lo1 = a1['mean_acc'] - a1['ci95_acc']
                hi1 = a1['mean_acc'] + a1['ci95_acc']
                lo2 = a2['mean_acc'] - a2['ci95_acc']
                hi2 = a2['mean_acc'] + a2['ci95_acc']
                overlap = lo1 <= hi2 and lo2 <= hi1
                diff = abs(a1['mean_acc'] - a2['mean_acc']) * 100
                status = 'OVERLAP (not significant)' if overlap else 'NO OVERLAP (significant)'
                print(f'  r={ranks[i]} vs r={ranks[j]}: '
                      f'diff={diff:.2f}%, {status}')

    # ── Save CSV ──
    csv_path = _V4_ROOT / 'training_output' / 'sweep_output_rank.csv'
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=[
            'rank', 'seed', 'out_params', 'total_params', 'best_loss',
            'final_loss', 'final_acc', 'elapsed', 'timestamp',
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
