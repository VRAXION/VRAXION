"""A/B/C Architecture Sweep on WikiText-103.

Validates whether synthetic sweep winners generalize to real language data.

Config A (baseline):  H=256,  SD=64,  M=256,  R=1  — current default small
Config B (winner):    H=512,  SD=32,  M=128,  R=2  — synthetic sweep winner
Config C (runner-up): H=512,  SD=16,  M=256,  R=1  — isolates H=512 vs R=2

If B ≈ C → H=512 is the key factor.
If B >> C → R=2 (wider attention window) matters more.
If A wins → synthetic sweep conclusions don't transfer.

Usage:
    python sweeps/ab_wikitext_arch_sweep.py
    python sweeps/ab_wikitext_arch_sweep.py --steps 20000 --seeds 3
    python sweeps/ab_wikitext_arch_sweep.py --steps 5000 --configs A,B
"""

import argparse
import csv
import json
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ── Path setup ──
V4_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V4_ROOT / 'model'))
sys.path.insert(0, str(V4_ROOT / 'training'))

from train import ByteDataset, func_discover_dat, func_maskloss_ce
from model_factory import build_model_from_spec, load_model_config

# ── Architecture configs ──
# Each config overrides only H, SD, M, R — everything else from YAML defaults.

ARCH_CONFIGS = {
    'A': {
        'label': 'baseline',
        'desc': 'H=256, SD=64, M=256, R=1 — current default',
        'overrides': {
            'hidden_dim': 256,
            'slot_dim': 64,
            'M': 256,
            'R': 1,
        },
    },
    'B': {
        'label': 'wide+attn',
        'desc': 'H=512, SD=32, M=128, R=2 — synthetic winner',
        'overrides': {
            'hidden_dim': 512,
            'slot_dim': 32,
            'M': 128,
            'R': 2,
        },
    },
    'C': {
        'label': 'wide_only',
        'desc': 'H=512, SD=16, M=256, R=1 — isolates H vs R',
        'overrides': {
            'hidden_dim': 512,
            'slot_dim': 16,
            'M': 256,
            'R': 1,
        },
    },
}


def build_model(base_model_cfg, overrides, embed_mode, seed, device):
    """Build a fresh INSTNCT model with specific arch overrides."""
    torch.manual_seed(seed)
    cfg = dict(base_model_cfg)
    cfg.update(overrides)
    record = build_model_from_spec.__module__  # just to confirm import
    from model_factory import _build_instnct_spec
    spec = _build_instnct_spec(embed_mode, cfg)
    record = {'type': 'instnct', 'build_spec': spec}
    model = build_model_from_spec(record, device)
    return model


def run_one(config_name, config, base_model_cfg, dataset, args, seed, out_dir):
    """Train one config variant, return metrics dict."""
    device = args.device
    use_amp = args.amp
    label = config['label']
    tag = f'{config_name}_{label}'

    model = build_model(base_model_cfg, config['overrides'],
                        embed_mode=True, seed=seed, device=device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and device.startswith('cuda')))

    # LR warmup
    warmup = args.warmup

    # Per-step CSV log
    csv_path = out_dir / f'{tag}_seed{seed}.csv'
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['step', 'loss', 'acc', 'lr', 'elapsed_s'])

    losses = []
    accs = []
    t0 = time.time()

    for step in range(1, args.steps + 1):
        # LR warmup
        if warmup > 0 and step <= warmup:
            lr_now = args.lr * step / warmup
            for pg in opt.param_groups:
                pg['lr'] = lr_now
        else:
            lr_now = args.lr

        xb, yb, mask = dataset.sample_batch(args.batch, device)

        with torch.amp.autocast('cuda', enabled=(use_amp and device.startswith('cuda'))):
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

        # Accuracy
        with torch.no_grad():
            preds = pred.argmax(dim=-1)
            correct = (preds == yb).float() * mask
            acc = correct.sum() / mask.sum().clamp(min=1)
            accs.append(acc.item())

        # CSV log every 10 steps
        if step % 10 == 0 or step == 1:
            elapsed = time.time() - t0
            csv_writer.writerow([step, f'{lv:.6f}', f'{acc.item():.4f}',
                                 f'{lr_now:.6f}', f'{elapsed:.1f}'])

        # Console log
        if step % args.log_every == 0 or step == 1:
            avg_loss = sum(losses[-100:]) / len(losses[-100:])
            avg_acc = sum(accs[-100:]) / len(accs[-100:])
            elapsed = time.time() - t0
            bpc = avg_loss / np.log(2)
            print(f'  [{tag}] step {step:5d}/{args.steps}  '
                  f'loss={avg_loss:.4f}  bpc={bpc:.3f}  '
                  f'acc={avg_acc:.3f}  '
                  f'{elapsed:.0f}s')

    csv_file.close()
    elapsed = time.time() - t0

    # Final metrics: average of last 200 steps (or all if fewer)
    tail = min(200, len(losses))
    final_loss = sum(losses[-tail:]) / tail
    final_acc = sum(accs[-tail:]) / tail

    # Convergence speed: step where rolling avg first reaches 90% of final acc
    conv_step = args.steps  # default if never reached
    if final_acc > 0:
        target = 0.9 * final_acc
        window = min(50, len(accs))
        for i in range(window, len(accs)):
            rolling = sum(accs[i - window:i]) / window
            if rolling >= target:
                conv_step = i
                break

    return {
        'config': config_name,
        'label': label,
        'seed': seed,
        'steps': args.steps,
        'params': n_params,
        'final_loss': final_loss,
        'final_bpc': final_loss / np.log(2),
        'final_acc': final_acc,
        'best_loss': min(losses),
        'best_acc': max(accs),
        'conv_step_90pct': conv_step,
        'time_s': elapsed,
        's_per_step': elapsed / args.steps,
        'csv_log': str(csv_path),
    }


def main():
    parser = argparse.ArgumentParser(
        description='A/B/C Architecture sweep on WikiText-103')
    parser.add_argument('--steps', type=int, default=10000,
                        help='Training steps per variant (default: 10000)')
    parser.add_argument('--seeds', type=int, default=2,
                        help='Number of random seeds (default: 2)')
    parser.add_argument('--batch', type=int, default=64,
                        help='Batch size (default: 64)')
    parser.add_argument('--seq', type=int, default=256,
                        help='Sequence length (default: 256)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--warmup', type=int, default=100,
                        help='LR warmup steps (default: 100)')
    parser.add_argument('--device', default=None,
                        help='Device: auto, cpu, cuda (default: auto)')
    parser.add_argument('--amp', action='store_true', default=True,
                        help='Use mixed precision (default: True)')
    parser.add_argument('--no-amp', dest='amp', action='store_false')
    parser.add_argument('--configs', type=str, default='A,B,C',
                        help='Comma-separated config names (default: A,B,C)')
    parser.add_argument('--log-every', type=int, default=100,
                        help='Console log interval (default: 100)')
    parser.add_argument('--data', type=str, default=None,
                        help='Data dir (default: v4/training_data)')
    args = parser.parse_args()

    # Device
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Parse config selection
    config_names = [c.strip().upper() for c in args.configs.split(',')]
    for c in config_names:
        if c not in ARCH_CONFIGS:
            print(f'[ERROR] Unknown config: {c}. Available: {list(ARCH_CONFIGS.keys())}')
            sys.exit(1)

    # Output directory
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = V4_ROOT / 'sweep_results' / f'arch_sweep_{run_id}'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Header
    print('=' * 72)
    print('  VRAXION v4 — Architecture A/B/C Sweep on WikiText-103')
    print('=' * 72)
    print(f'Run ID:   {run_id}')
    print(f'Steps:    {args.steps}')
    print(f'Seeds:    {args.seeds}')
    print(f'Batch:    {args.batch} x {args.seq}')
    print(f'LR:       {args.lr} (warmup {args.warmup})')
    print(f'AMP:      {args.amp}')
    print(f'Device:   {args.device}')
    if args.device.startswith('cuda'):
        print(f'GPU:      {torch.cuda.get_device_name(0)}')
    print(f'Configs:  {config_names}')
    print(f'Output:   {out_dir}')
    print()

    # Print config table
    print(f'{"Config":8s} {"Label":12s} {"H":>5s} {"SD":>5s} {"M":>5s} {"R":>3s}  Description')
    print('-' * 72)
    for c in config_names:
        cfg = ARCH_CONFIGS[c]
        ov = cfg['overrides']
        print(f'{c:8s} {cfg["label"]:12s} {ov["hidden_dim"]:5d} '
              f'{ov["slot_dim"]:5d} {ov["M"]:5d} {ov["R"]:3d}  {cfg["desc"]}')
    print()

    # Load base model config from YAML (for non-overridden params)
    base_model_cfg = load_model_config(V4_ROOT)

    # Load data
    data_dir = Path(args.data) if args.data else V4_ROOT / 'training_data'
    files = func_discover_dat(str(data_dir))
    if not files:
        print(f'[ERROR] No .traindat files found in {data_dir}')
        print('Run: python datagen/download_wikitext.py first')
        sys.exit(1)
    dataset = ByteDataset(files, args.seq, embed_mode=True, seed=42)
    total_mb = dataset.total_bytes / 1e6
    print(f'Data: {len(files)} shard(s), {total_mb:.0f} MB')
    print()

    # ── Run sweep ──
    all_results = []

    for seed_idx in range(args.seeds):
        seed = 42 + seed_idx * 1000
        print(f'{"=" * 72}')
        print(f'  Seed {seed} ({seed_idx + 1}/{args.seeds})')
        print(f'{"=" * 72}')

        for config_name in config_names:
            config = ARCH_CONFIGS[config_name]
            # Reset dataset RNG for fair comparison
            dataset.rng = np.random.default_rng(seed)

            if args.device.startswith('cuda'):
                torch.cuda.empty_cache()

            result = run_one(config_name, config, base_model_cfg,
                             dataset, args, seed, out_dir)
            all_results.append(result)

            print(f'  -> {config_name} ({config["label"]}): '
                  f'loss={result["final_loss"]:.4f}  '
                  f'bpc={result["final_bpc"]:.3f}  '
                  f'acc={result["final_acc"]:.3f}  '
                  f'best_acc={result["best_acc"]:.3f}  '
                  f'params={result["params"]:,}  '
                  f'conv@90%={result["conv_step_90pct"]}  '
                  f'({result["time_s"]:.0f}s)')
            print()

    # ── Summary table ──
    print()
    print('=' * 90)
    print('  SUMMARY — Architecture A/B/C Sweep')
    print('=' * 90)
    print(f'{"Config":8s} {"Label":12s} {"Params":>8s} {"Avg Loss":>10s} '
          f'{"Avg BPC":>9s} {"Avg Acc":>9s} {"Best Acc":>9s} '
          f'{"Conv@90%":>9s} {"Avg Time":>9s}')
    print('-' * 90)

    summary_rows = []
    for config_name in config_names:
        cfg = ARCH_CONFIGS[config_name]
        runs = [r for r in all_results if r['config'] == config_name]
        n = len(runs)
        avg_loss = sum(r['final_loss'] for r in runs) / n
        avg_bpc = sum(r['final_bpc'] for r in runs) / n
        avg_acc = sum(r['final_acc'] for r in runs) / n
        best_acc = max(r['best_acc'] for r in runs)
        avg_conv = sum(r['conv_step_90pct'] for r in runs) / n
        avg_time = sum(r['time_s'] for r in runs) / n
        params = runs[0]['params']

        row = {
            'config': config_name,
            'label': cfg['label'],
            'params': params,
            'avg_loss': avg_loss,
            'avg_bpc': avg_bpc,
            'avg_acc': avg_acc,
            'best_acc': best_acc,
            'avg_conv_step': avg_conv,
            'avg_time_s': avg_time,
        }
        summary_rows.append(row)

        print(f'{config_name:8s} {cfg["label"]:12s} {params:8,d} '
              f'{avg_loss:10.4f} {avg_bpc:9.3f} {avg_acc:9.3f} '
              f'{best_acc:9.3f} {avg_conv:9.0f} {avg_time:8.0f}s')

    # ── Pairwise deltas ──
    print()
    print('Pairwise Deltas (accuracy):')
    for i, r1 in enumerate(summary_rows):
        for r2 in summary_rows[i + 1:]:
            delta = (r1['avg_acc'] - r2['avg_acc']) * 100
            winner = r1['config'] if delta > 0 else r2['config']
            print(f'  {r1["config"]} vs {r2["config"]}: '
                  f'{delta:+.2f}%  -> {winner} wins')

    # ── Interpretation ──
    if len(config_names) >= 3 and all(c in [r['config'] for r in summary_rows] for c in ['A', 'B', 'C']):
        a = next(r for r in summary_rows if r['config'] == 'A')
        b = next(r for r in summary_rows if r['config'] == 'B')
        c = next(r for r in summary_rows if r['config'] == 'C')

        print()
        print('Interpretation:')
        b_vs_a = (b['avg_acc'] - a['avg_acc']) * 100
        c_vs_a = (c['avg_acc'] - a['avg_acc']) * 100
        b_vs_c = (b['avg_acc'] - c['avg_acc']) * 100

        if b_vs_a > 1.0 and c_vs_a > 1.0:
            if abs(b_vs_c) < 0.5:
                print(f'  B and C both beat A by ~{b_vs_a:.1f}% / ~{c_vs_a:.1f}%.')
                print(f'  B vs C delta is only {b_vs_c:+.2f}% -> H=512 is the key factor, R=2 adds little.')
            elif b_vs_c > 0.5:
                print(f'  B >> C by {b_vs_c:+.2f}% -> R=2 (wider attention) matters on top of H=512.')
            else:
                print(f'  C >> B by {-b_vs_c:+.2f}% -> R=1 with smaller SD is better than R=2.')
        elif b_vs_a > 1.0:
            print(f'  B beats A by {b_vs_a:+.2f}% but C does not ({c_vs_a:+.2f}%).')
            print(f'  -> R=2 is essential, H=512 alone is not enough.')
        elif b_vs_a < -1.0:
            print(f'  A beats B by {-b_vs_a:.2f}% -> synthetic sweep results do NOT transfer!')
            print(f'  -> Need to re-evaluate on real data from scratch.')
        else:
            print(f'  All configs within ~1% -> architecture matters less than training dynamics.')
            print(f'  -> Focus on learning rate, data quality, or longer training instead.')

    # ── Save results ──
    results_json = out_dir / 'results.json'
    with open(results_json, 'w') as f:
        json.dump({
            'run_id': run_id,
            'args': vars(args),
            'configs': {k: ARCH_CONFIGS[k] for k in config_names},
            'results': all_results,
            'summary': summary_rows,
        }, f, indent=2, default=str)

    summary_md = out_dir / 'SUMMARY.md'
    with open(summary_md, 'w') as f:
        f.write(f'# Architecture Sweep — {run_id}\n\n')
        f.write(f'Steps: {args.steps} | Seeds: {args.seeds} | '
                f'Batch: {args.batch}x{args.seq} | LR: {args.lr}\n\n')
        f.write('| Config | Label | Params | Avg Loss | BPC | Acc | Best Acc | Conv@90% | Time |\n')
        f.write('|--------|-------|--------|----------|-----|-----|----------|----------|------|\n')
        for row in summary_rows:
            f.write(f'| {row["config"]} | {row["label"]} | {row["params"]:,} | '
                    f'{row["avg_loss"]:.4f} | {row["avg_bpc"]:.3f} | '
                    f'{row["avg_acc"]:.3f} | {row["best_acc"]:.3f} | '
                    f'{row["avg_conv_step"]:.0f} | {row["avg_time_s"]:.0f}s |\n')

    print()
    print(f'Results saved to: {out_dir}')
    print(f'  - results.json  (full data)')
    print(f'  - SUMMARY.md    (table)')
    print(f'  - *_seed*.csv   (per-step logs)')
    print('=' * 90)


if __name__ == '__main__':
    main()
