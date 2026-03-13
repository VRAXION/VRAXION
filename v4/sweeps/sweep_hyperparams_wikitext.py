"""Hyperparameter Permutation Sweep on WikiText-2.

Tests 8 configurations varying R, hidden_dim, embed_encoding, and LR
to find the best combination for pushing BPC toward lower values.

Configs:
  A: H=256, R=1, embed=learned, lr=1e-3  (baseline small)
  B: H=256, R=2, embed=learned, lr=1e-3  (R effect)
  C: H=512, R=1, embed=learned, lr=1e-3  (H effect)
  D: H=512, R=2, embed=learned, lr=1e-3  (H+R combo)
  E: H=512, R=2, embed=bitlift, lr=1e-3  (embed comparison)
  F: H=512, R=2, embed=learned, lr=5e-4  (lower LR)
  G: H=512, R=3, embed=learned, lr=1e-3  (bigger R)
  H: H=256, R=2, embed=learned, lr=5e-4  (small+slow)

Usage:
    python sweeps/sweep_hyperparams_wikitext.py
    python sweeps/sweep_hyperparams_wikitext.py --steps 2000 --configs A,B,C,D
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
# Each config specifies overrides + training params.
SWEEP_CONFIGS = {
    'A': {
        'label': 'base_small',
        'desc': 'H=256, R=1, learned, lr=1e-3',
        'model': {'hidden_dim': 256, 'slot_dim': 64, 'M': 512, 'R': 1,
                  'embed_encoding': 'learned'},
        'lr': 1e-3,
    },
    'B': {
        'label': 'small_R2',
        'desc': 'H=256, R=2, learned, lr=1e-3',
        'model': {'hidden_dim': 256, 'slot_dim': 64, 'M': 512, 'R': 2,
                  'embed_encoding': 'learned'},
        'lr': 1e-3,
    },
    'C': {
        'label': 'wide',
        'desc': 'H=512, R=1, learned, lr=1e-3',
        'model': {'hidden_dim': 512, 'slot_dim': 64, 'M': 512, 'R': 1,
                  'embed_encoding': 'learned'},
        'lr': 1e-3,
    },
    'D': {
        'label': 'wide_R2',
        'desc': 'H=512, R=2, learned, lr=1e-3',
        'model': {'hidden_dim': 512, 'slot_dim': 64, 'M': 512, 'R': 2,
                  'embed_encoding': 'learned'},
        'lr': 1e-3,
    },
    'E': {
        'label': 'wide_R2_bit',
        'desc': 'H=512, R=2, bitlift, lr=1e-3',
        'model': {'hidden_dim': 512, 'slot_dim': 64, 'M': 512, 'R': 2,
                  'embed_encoding': 'bitlift'},
        'lr': 1e-3,
    },
    'F': {
        'label': 'wide_R2_slow',
        'desc': 'H=512, R=2, learned, lr=5e-4',
        'model': {'hidden_dim': 512, 'slot_dim': 64, 'M': 512, 'R': 2,
                  'embed_encoding': 'learned'},
        'lr': 5e-4,
    },
    'G': {
        'label': 'wide_R3',
        'desc': 'H=512, R=3, learned, lr=1e-3',
        'model': {'hidden_dim': 512, 'slot_dim': 64, 'M': 512, 'R': 3,
                  'embed_encoding': 'learned'},
        'lr': 1e-3,
    },
    'H': {
        'label': 'small_R2_slow',
        'desc': 'H=256, R=2, learned, lr=5e-4',
        'model': {'hidden_dim': 256, 'slot_dim': 64, 'M': 512, 'R': 2,
                  'embed_encoding': 'learned'},
        'lr': 5e-4,
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

    # State for sequential/TBPTT training
    state = None

    for step in range(1, args.steps + 1):
        # LR warmup
        if warmup > 0 and step <= warmup:
            lr_now = lr * step / warmup
            for pg in opt.param_groups:
                pg['lr'] = lr_now
        else:
            lr_now = lr

        xb, yb, mask = dataset.sample_batch(args.batch, device)

        pred, state = model(xb, state=None)  # fresh state each batch for sweep fairness
        _, masked_loss = func_maskloss_ce(pred, yb, mask)

        opt.zero_grad()
        masked_loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).item()
        opt.step()

        lv = masked_loss.item()
        losses.append(lv)
        gnorms.append(gnorm)

        # Accuracy
        with torch.no_grad():
            preds = pred.argmax(dim=-1)
            correct = (preds == yb).float() * mask
            acc = correct.sum() / mask.sum().clamp(min=1)
            accs.append(acc.item())

        # CSV log every 10 steps
        if step % 10 == 0 or step == 1:
            elapsed = time.time() - t0
            bpc = lv / np.log(2)
            writer.writerow([step, f'{lv:.6f}', f'{acc.item():.4f}',
                             f'{bpc:.4f}', f'{lr_now:.6f}',
                             f'{gnorm:.4f}', f'{elapsed:.1f}'])

        # Console log
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

    # Final metrics: last 200 steps
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
        description='Hyperparameter sweep on WikiText-2')
    parser.add_argument('--steps', type=int, default=1000,
                        help='Training steps per config (default: 1000)')
    parser.add_argument('--batch', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--seq', type=int, default=256,
                        help='Sequence length (default: 256)')
    parser.add_argument('--warmup', type=int, default=50,
                        help='LR warmup steps (default: 50)')
    parser.add_argument('--grad-clip', type=float, default=10.0,
                        help='Gradient clipping (default: 10.0)')
    parser.add_argument('--device', default=None)
    parser.add_argument('--configs', type=str, default='A,B,C,D,E,F,G,H',
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
    out_dir = V4_ROOT / 'sweep_results' / f'hyperparam_sweep_{run_id}'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Header
    print('=' * 80)
    print('  VRAXION v4 — Hyperparameter Permutation Sweep on WikiText-2')
    print('=' * 80)
    print(f'Run ID:     {run_id}')
    print(f'Steps:      {args.steps}')
    print(f'Batch:      {args.batch} x {args.seq}')
    print(f'Grad clip:  {args.grad_clip}')
    print(f'Device:     {args.device}')
    print(f'Configs:    {config_names}')
    print(f'Output:     {out_dir}')
    print()

    # Config table
    print(f'{"Cfg":4s} {"Label":16s} {"H":>5s} {"R":>3s} {"SD":>4s} {"M":>5s} '
          f'{"Embed":>8s} {"LR":>8s}  Description')
    print('-' * 80)
    for c in config_names:
        cfg = SWEEP_CONFIGS[c]
        m = cfg['model']
        print(f'{c:4s} {cfg["label"]:16s} {m["hidden_dim"]:5d} {m["R"]:3d} '
              f'{m["slot_dim"]:4d} {m["M"]:5d} '
              f'{m["embed_encoding"]:>8s} {cfg["lr"]:8.1e}  {cfg["desc"]}')
    print()

    # Load base model config
    base_cfg = load_model_config(V4_ROOT)

    # Load WikiText data
    data_dir = Path(args.data) if args.data else V4_ROOT / 'training_data' / 'real_wikitext'
    if not data_dir.exists():
        # Try parent
        data_dir = V4_ROOT / 'training_data'
    files = func_discover_dat(str(data_dir))
    # Filter to only WikiText shards if mixed
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
    print('  SUMMARY — Hyperparameter Permutation Sweep')
    print('=' * 100)
    print(f'{"Cfg":4s} {"Label":16s} {"Params":>8s} {"Final Loss":>11s} '
          f'{"Final BPC":>10s} {"Best BPC":>9s} {"Final Acc":>10s} '
          f'{"Best Acc":>9s} {"GNorm":>7s} {"Time":>7s}')
    print('-' * 100)

    for r in all_results:
        print(f'{r["config"]:4s} {r["label"]:16s} {r["params"]:8,d} '
              f'{r["final_loss"]:11.4f} {r["final_bpc"]:10.3f} '
              f'{r["best_bpc"]:9.3f} {r["final_acc"]:10.3f} '
              f'{r["best_acc"]:9.3f} {r["final_gnorm"]:7.1f} '
              f'{r["time_s"]:6.0f}s')

    # ── Rankings ──
    print()
    print('Rankings (by final BPC, lower is better):')
    ranked = sorted(all_results, key=lambda r: r['final_bpc'])
    for i, r in enumerate(ranked, 1):
        delta = r['final_bpc'] - ranked[0]['final_bpc']
        print(f'  #{i}  {r["config"]} ({r["label"]:16s})  '
              f'BPC={r["final_bpc"]:.3f}  '
              f'(+{delta:.3f})  '
              f'acc={r["final_acc"]:.3f}  '
              f'params={r["params"]:,}')

    # ── Factor analysis ──
    print()
    print('Factor Analysis:')

    # R effect: compare A vs B (R=1 vs R=2 at H=256)
    def get(name):
        return next((r for r in all_results if r['config'] == name), None)

    a, b = get('A'), get('B')
    if a and b:
        delta = (a['final_bpc'] - b['final_bpc'])
        print(f'  R effect (H=256): R=1→R=2  BPC delta = {delta:+.3f}  '
              f'({"R=2 better" if delta > 0 else "R=1 better"})')

    c, d = get('C'), get('D')
    if c and d:
        delta = (c['final_bpc'] - d['final_bpc'])
        print(f'  R effect (H=512): R=1→R=2  BPC delta = {delta:+.3f}  '
              f'({"R=2 better" if delta > 0 else "R=1 better"})')

    d2, g = get('D'), get('G')
    if d2 and g:
        delta = (d2['final_bpc'] - g['final_bpc'])
        print(f'  R effect (H=512): R=2→R=3  BPC delta = {delta:+.3f}  '
              f'({"R=3 better" if delta > 0 else "R=2 better"})')

    # H effect: compare A vs C, B vs D
    if a and c:
        delta = (a['final_bpc'] - c['final_bpc'])
        print(f'  H effect (R=1):   H=256→512  BPC delta = {delta:+.3f}  '
              f'({"H=512 better" if delta > 0 else "H=256 better"})')
    if b and d:
        delta = (b['final_bpc'] - d['final_bpc'])
        print(f'  H effect (R=2):   H=256→512  BPC delta = {delta:+.3f}  '
              f'({"H=512 better" if delta > 0 else "H=256 better"})')

    # Embed effect: D vs E
    d3, e = get('D'), get('E')
    if d3 and e:
        delta = (e['final_bpc'] - d3['final_bpc'])
        print(f'  Embed effect:     learned vs bitlift  BPC delta = {delta:+.3f}  '
              f'({"learned better" if delta > 0 else "bitlift better"})')

    # LR effect: D vs F
    d4, f_ = get('D'), get('F')
    if d4 and f_:
        delta = (d4['final_bpc'] - f_['final_bpc'])
        print(f'  LR effect (H=512,R=2): 1e-3 vs 5e-4  BPC delta = {delta:+.3f}  '
              f'({"5e-4 better" if delta > 0 else "1e-3 better"})')

    b2, h = get('B'), get('H')
    if b2 and h:
        delta = (b2['final_bpc'] - h['final_bpc'])
        print(f'  LR effect (H=256,R=2): 1e-3 vs 5e-4  BPC delta = {delta:+.3f}  '
              f'({"5e-4 better" if delta > 0 else "1e-3 better"})')

    # ── Save results ──
    results_json = out_dir / 'results.json'
    with open(results_json, 'w') as f:
        json.dump({
            'run_id': run_id,
            'args': vars(args),
            'configs': {k: SWEEP_CONFIGS[k] for k in config_names},
            'results': all_results,
        }, f, indent=2, default=str)

    # Master info table (markdown)
    master_md = out_dir / 'MASTER_INFO.md'
    with open(master_md, 'w') as f:
        f.write(f'# Hyperparameter Sweep — {run_id}\n\n')
        f.write(f'**Data**: WikiText-2 ({total_mb:.1f} MB) | '
                f'**Steps**: {args.steps} | **Batch**: {args.batch}x{args.seq} | '
                f'**Device**: {args.device}\n\n')
        f.write('## Results\n\n')
        f.write('| # | Cfg | Label | H | R | Embed | LR | Params | '
                'Final BPC | Best BPC | Final Acc | Best Acc | GNorm | Time |\n')
        f.write('|---|-----|-------|---|---|-------|----|--------|'
                '----------|----------|-----------|----------|-------|------|\n')
        for i, r in enumerate(ranked, 1):
            cfg = SWEEP_CONFIGS[r['config']]
            m = cfg['model']
            f.write(f'| {i} | {r["config"]} | {r["label"]} | '
                    f'{m["hidden_dim"]} | {m["R"]} | {m["embed_encoding"]} | '
                    f'{cfg["lr"]:.0e} | {r["params"]:,} | '
                    f'{r["final_bpc"]:.3f} | {r["best_bpc"]:.3f} | '
                    f'{r["final_acc"]:.3f} | {r["best_acc"]:.3f} | '
                    f'{r["final_gnorm"]:.1f} | {r["time_s"]:.0f}s |\n')

        f.write('\n## Factor Analysis\n\n')
        f.write('| Factor | Comparison | BPC Delta | Winner |\n')
        f.write('|--------|------------|-----------|--------|\n')

        if a and b:
            d_ = a['final_bpc'] - b['final_bpc']
            f.write(f'| R (H=256) | R=1→R=2 | {d_:+.3f} | '
                    f'{"R=2" if d_ > 0 else "R=1"} |\n')
        if c and d:
            d_ = c['final_bpc'] - d['final_bpc']
            f.write(f'| R (H=512) | R=1→R=2 | {d_:+.3f} | '
                    f'{"R=2" if d_ > 0 else "R=1"} |\n')
        if d2 and g:
            d_ = d2['final_bpc'] - g['final_bpc']
            f.write(f'| R (H=512) | R=2→R=3 | {d_:+.3f} | '
                    f'{"R=3" if d_ > 0 else "R=2"} |\n')
        if a and c:
            d_ = a['final_bpc'] - c['final_bpc']
            f.write(f'| H (R=1) | 256→512 | {d_:+.3f} | '
                    f'{"512" if d_ > 0 else "256"} |\n')
        if b and d:
            d_ = b['final_bpc'] - d['final_bpc']
            f.write(f'| H (R=2) | 256→512 | {d_:+.3f} | '
                    f'{"512" if d_ > 0 else "256"} |\n')
        if d3 and e:
            d_ = e['final_bpc'] - d3['final_bpc']
            f.write(f'| Embed | learned vs bitlift | {d_:+.3f} | '
                    f'{"learned" if d_ > 0 else "bitlift"} |\n')
        if d4 and f_:
            d_ = d4['final_bpc'] - f_['final_bpc']
            f.write(f'| LR (H=512,R=2) | 1e-3 vs 5e-4 | {d_:+.3f} | '
                    f'{"5e-4" if d_ > 0 else "1e-3"} |\n')
        if b2 and h:
            d_ = b2['final_bpc'] - h['final_bpc']
            f.write(f'| LR (H=256,R=2) | 1e-3 vs 5e-4 | {d_:+.3f} | '
                    f'{"5e-4" if d_ > 0 else "1e-3"} |\n')

    print()
    print(f'Results saved to: {out_dir}')
    print(f'  - results.json    (full data)')
    print(f'  - MASTER_INFO.md  (ranked table + factor analysis)')
    print(f'  - *_seed*.csv     (per-step logs)')
    print('=' * 100)


if __name__ == '__main__':
    main()
