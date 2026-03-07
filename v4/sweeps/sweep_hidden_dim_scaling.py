"""Hidden-Dim Scaling Sweep on WikiText-2.

Based on the 8-config hyperparameter sweep findings:
  - hidden_dim is the dominant factor for BPC improvement
  - R (ring rounds) has negligible effect → fixed at R=1
  - learned embedding >> bitlift → fixed at learned
  - lr=1e-3 >> lr=5e-4 at 500 steps → fixed at 1e-3

Configs:
  A: H=384  — interpolate between 256 and 512
  B: H=512  — previous best (anchor)
  C: H=768  — 1.5x
  D: H=1024 — 2x

Usage:
    python sweeps/sweep_hidden_dim_scaling.py
    python sweeps/sweep_hidden_dim_scaling.py --steps 1000 --configs A,B,C,D
"""

import argparse
import csv
import json
import sys
import time
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
from model_factory import build_model_from_spec, load_model_config, _build_instnct_spec

# ── Sweep configs ──
# All use R=1, learned embedding, lr=1e-3. Only hidden_dim varies.
SWEEP_CONFIGS = {
    'A': {
        'label': 'H384',
        'desc': 'H=384, R=1, learned, lr=1e-3',
        'model': {'hidden_dim': 384, 'slot_dim': 64, 'M': 512, 'R': 1,
                  'embed_encoding': 'learned'},
        'lr': 1e-3,
    },
    'B': {
        'label': 'H512',
        'desc': 'H=512, R=1, learned, lr=1e-3 (anchor)',
        'model': {'hidden_dim': 512, 'slot_dim': 64, 'M': 512, 'R': 1,
                  'embed_encoding': 'learned'},
        'lr': 1e-3,
    },
    'C': {
        'label': 'H768',
        'desc': 'H=768, R=1, learned, lr=1e-3',
        'model': {'hidden_dim': 768, 'slot_dim': 64, 'M': 512, 'R': 1,
                  'embed_encoding': 'learned'},
        'lr': 1e-3,
    },
    'D': {
        'label': 'H1024',
        'desc': 'H=1024, R=1, learned, lr=1e-3',
        'model': {'hidden_dim': 1024, 'slot_dim': 64, 'M': 512, 'R': 1,
                  'embed_encoding': 'learned'},
        'lr': 1e-3,
    },
}


def build_model(base_cfg, overrides, seed, device):
    """Build a fresh INSTNCT model with specific overrides."""
    torch.manual_seed(seed)
    cfg = dict(base_cfg)
    cfg.update(overrides)
    spec = _build_instnct_spec(True, cfg)  # embed_mode=True
    record = {'type': 'instnct', 'build_spec': spec}
    model = build_model_from_spec(record, device)
    return model


def run_one(name, config, base_cfg, dataset, args, seed, out_dir):
    """Train one config, return metrics dict."""
    device = args.device
    label = config['label']
    tag = f'{name}_{label}'
    lr = config['lr']

    model = build_model(base_cfg, config['model'], seed, device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # LR warmup
    warmup = args.warmup

    # Per-step CSV
    csv_path = out_dir / f'{tag}_seed{seed}.csv'
    csv_file = open(csv_path, 'w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(['step', 'loss', 'acc', 'bpc', 'lr', 'gnorm', 'elapsed_s'])

    losses = []
    accs = []
    gnorms = []
    t0 = time.time()

    for step in range(1, args.steps + 1):
        # LR warmup
        if warmup > 0 and step <= warmup:
            lr_now = lr * step / warmup
            for pg in opt.param_groups:
                pg['lr'] = lr_now
        else:
            lr_now = lr

        xb, yb, mask = dataset.sample_batch(args.batch, device)

        pred, state = model(xb, state=None)
        _, masked_loss = func_maskloss_ce(pred, yb, mask)

        opt.zero_grad()
        masked_loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).item()
        opt.step()

        lv = masked_loss.item()
        losses.append(lv)
        gnorms.append(gnorm)

        with torch.no_grad():
            preds = pred.argmax(dim=-1)
            correct = (preds == yb).float() * mask
            acc = correct.sum() / mask.sum().clamp(min=1)
            accs.append(acc.item())

        if step % 10 == 0 or step == 1:
            elapsed = time.time() - t0
            bpc = lv / np.log(2)
            writer.writerow([step, f'{lv:.6f}', f'{acc.item():.4f}',
                             f'{bpc:.4f}', f'{lr_now:.6f}',
                             f'{gnorm:.4f}', f'{elapsed:.1f}'])

        if step % args.log_every == 0 or step == 1:
            avg_loss = sum(losses[-100:]) / len(losses[-100:])
            avg_acc = sum(accs[-100:]) / len(accs[-100:])
            avg_gnorm = sum(gnorms[-100:]) / len(gnorms[-100:])
            bpc = avg_loss / np.log(2)
            elapsed = time.time() - t0
            speed = step / elapsed if elapsed > 0 else 0
            print(f'  [{tag}] {step:5d}/{args.steps}  '
                  f'loss={avg_loss:.4f}  bpc={bpc:.3f}  '
                  f'acc={avg_acc:.3f}  gnorm={avg_gnorm:.1f}  '
                  f'{speed:.1f} step/s  ({elapsed:.0f}s)')

    csv_file.close()
    elapsed = time.time() - t0

    tail = min(200, len(losses))
    final_loss = sum(losses[-tail:]) / tail
    final_acc = sum(accs[-tail:]) / tail
    final_gnorm = sum(gnorms[-tail:]) / tail
    best_loss = min(losses)
    best_acc = max(accs)

    return {
        'config': name,
        'label': label,
        'desc': config['desc'],
        'seed': seed,
        'steps': args.steps,
        'params': n_params,
        'lr': lr,
        'final_loss': final_loss,
        'final_bpc': final_loss / np.log(2),
        'best_loss': best_loss,
        'best_bpc': best_loss / np.log(2),
        'final_acc': final_acc,
        'best_acc': best_acc,
        'final_gnorm': final_gnorm,
        'time_s': elapsed,
        's_per_step': elapsed / args.steps,
        'csv_log': str(csv_path),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Hidden-dim scaling sweep on WikiText-2')
    parser.add_argument('--steps', type=int, default=500,
                        help='Training steps per config (default: 500)')
    parser.add_argument('--batch', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--seq', type=int, default=256,
                        help='Sequence length (default: 256)')
    parser.add_argument('--warmup', type=int, default=50,
                        help='LR warmup steps (default: 50)')
    parser.add_argument('--grad-clip', type=float, default=10.0,
                        help='Gradient clipping (default: 10.0)')
    parser.add_argument('--device', default=None)
    parser.add_argument('--configs', type=str, default='A,B,C,D',
                        help='Comma-separated config names')
    parser.add_argument('--log-every', type=int, default=100,
                        help='Console log interval')
    parser.add_argument('--data', type=str, default=None,
                        help='Data dir with WikiText .traindat files')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    args = parser.parse_args()

    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config_names = [c.strip().upper() for c in args.configs.split(',')]
    for c in config_names:
        if c not in SWEEP_CONFIGS:
            print(f'[ERROR] Unknown config: {c}. Available: {list(SWEEP_CONFIGS.keys())}')
            sys.exit(1)

    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = V4_ROOT / 'sweep_results' / f'hidden_dim_sweep_{run_id}'
    out_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 80)
    print('  VRAXION v4 — Hidden-Dim Scaling Sweep on WikiText-2')
    print('=' * 80)
    print(f'Run ID:     {run_id}')
    print(f'Steps:      {args.steps}')
    print(f'Batch:      {args.batch} x {args.seq}')
    print(f'Grad clip:  {args.grad_clip}')
    print(f'Device:     {args.device}')
    print(f'Configs:    {config_names}')
    print(f'Output:     {out_dir}')
    print()

    print(f'{"Cfg":4s} {"Label":8s} {"H":>5s} {"Params":>8s}  Description')
    print('-' * 60)
    for c in config_names:
        cfg = SWEEP_CONFIGS[c]
        m = cfg['model']
        print(f'{c:4s} {cfg["label"]:8s} {m["hidden_dim"]:5d} {"?":>8s}  {cfg["desc"]}')
    print()

    # Load base model config
    base_cfg = load_model_config(V4_ROOT)

    # Load WikiText data
    data_dir = Path(args.data) if args.data else V4_ROOT / 'training_data' / 'real_wikitext'
    if not data_dir.exists():
        data_dir = V4_ROOT / 'training_data'
    files = func_discover_dat(str(data_dir))
    wiki_files = [f for f in files if 'wikitext' in str(f[0]).lower() or 'shard' in str(f[0]).lower()]
    if wiki_files:
        files = wiki_files
    dataset = ByteDataset(files, args.seq, embed_mode=True, seed=args.seed)
    total_mb = dataset.total_bytes / 1e6
    print(f'Data: {len(files)} shard(s), {total_mb:.1f} MB')
    print()

    # ── Run sweep ──
    all_results = []
    seed = args.seed

    for config_name in config_names:
        config = SWEEP_CONFIGS[config_name]
        dataset.rng = np.random.default_rng(seed)

        print(f'{"=" * 80}')
        print(f'  Config {config_name}: {config["desc"]}')
        print(f'{"=" * 80}')

        result = run_one(config_name, config, base_cfg, dataset, args,
                         seed, out_dir)
        all_results.append(result)

        print(f'  => {config_name} ({config["label"]}): '
              f'loss={result["final_loss"]:.4f}  '
              f'bpc={result["final_bpc"]:.3f}  '
              f'acc={result["final_acc"]:.3f}  '
              f'best_bpc={result["best_bpc"]:.3f}  '
              f'params={result["params"]:,}  '
              f'({result["time_s"]:.0f}s)')
        print()

    # ── Summary ──
    print()
    print('=' * 100)
    print('  SUMMARY — Hidden-Dim Scaling Sweep')
    print('=' * 100)
    print(f'{"Cfg":4s} {"Label":8s} {"H":>5s} {"Params":>8s} {"Final Loss":>11s} '
          f'{"Final BPC":>10s} {"Best BPC":>9s} {"Final Acc":>10s} '
          f'{"Best Acc":>9s} {"GNorm":>7s} {"Time":>7s}')
    print('-' * 100)

    for r in all_results:
        cfg = SWEEP_CONFIGS[r['config']]
        h = cfg['model']['hidden_dim']
        print(f'{r["config"]:4s} {r["label"]:8s} {h:5d} {r["params"]:8,d} '
              f'{r["final_loss"]:11.4f} {r["final_bpc"]:10.3f} '
              f'{r["best_bpc"]:9.3f} {r["final_acc"]:10.3f} '
              f'{r["best_acc"]:9.3f} {r["final_gnorm"]:7.1f} '
              f'{r["time_s"]:6.0f}s')

    # ── Scaling analysis ──
    print()
    print('Scaling Analysis:')
    if len(all_results) >= 2:
        sorted_by_h = sorted(all_results,
                              key=lambda r: SWEEP_CONFIGS[r['config']]['model']['hidden_dim'])
        base = sorted_by_h[0]
        for r in sorted_by_h[1:]:
            h_base = SWEEP_CONFIGS[base['config']]['model']['hidden_dim']
            h_cur = SWEEP_CONFIGS[r['config']]['model']['hidden_dim']
            bpc_delta = base['best_bpc'] - r['best_bpc']
            param_ratio = r['params'] / base['params']
            print(f'  H={h_base}→{h_cur}: '
                  f'BPC improvement = {bpc_delta:+.3f}  '
                  f'param ratio = {param_ratio:.2f}x  '
                  f'BPC/param_ratio = {bpc_delta/param_ratio:.4f}')

    # Rankings
    print()
    print('Rankings (by best BPC, lower is better):')
    ranked = sorted(all_results, key=lambda r: r['best_bpc'])
    for i, r in enumerate(ranked, 1):
        delta = r['best_bpc'] - ranked[0]['best_bpc']
        cfg = SWEEP_CONFIGS[r['config']]
        h = cfg['model']['hidden_dim']
        print(f'  #{i}  {r["config"]} (H={h:4d})  '
              f'BPC={r["best_bpc"]:.3f}  '
              f'(+{delta:.3f})  '
              f'acc={r["best_acc"]:.3f}  '
              f'params={r["params"]:,}')

    # ── Save results ──
    results_json = out_dir / 'results.json'
    with open(results_json, 'w') as f:
        json.dump({
            'run_id': run_id,
            'args': vars(args),
            'configs': {k: SWEEP_CONFIGS[k] for k in config_names},
            'results': all_results,
        }, f, indent=2, default=str)

    print()
    print(f'Results saved to: {out_dir}')
    print(f'  - results.json  (full data)')
    print(f'  - *_seed*.csv   (per-step logs)')
    print('=' * 100)


if __name__ == '__main__':
    main()
