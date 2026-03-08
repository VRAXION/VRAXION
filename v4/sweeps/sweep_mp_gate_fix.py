"""Multi-Pointer Gate Fix — A/B/C Sweep.

Tests whether softmax gating fixes ring_norm explosion in multi-pointer mode.

Config A (baseline):   mp_enabled=false — single pointer read (control)
Config B (softmax):    mp_enabled=true, mp_gate_mode=softmax — bounded gate sum=1
Config C (sigmoid):    mp_enabled=true, mp_gate_mode=sigmoid — original (unbounded)

Key metrics to watch:
  - ring_norm: should stay ~700 (baseline level) for B, may explode for C
  - ring_signal_norm: should stay <100 for B
  - hidden_norm: should stay <50, no spikes
  - alpha: should rise (model trusts ring), not drop
  - masked_loss: B should be <= A

Usage:
    cd v4
    python -u sweeps/sweep_mp_gate_fix.py
    python -u sweeps/sweep_mp_gate_fix.py --steps 1000 --configs A,B
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
from model_factory import _build_instnct_spec, build_model_from_spec, load_model_config

# ── Configs ──
SWEEP_CONFIGS = {
    'A': {
        'label': 'baseline',
        'desc': 'mp_enabled=false — single pointer (control)',
        'overrides': {
            'mp_enabled': False,
        },
    },
    'B': {
        'label': 'mp_softmax',
        'desc': 'mp_enabled=true, gate_mode=softmax — bounded (fix)',
        'overrides': {
            'mp_enabled': True,
            'mp_heads': 4,
            'mp_gate_mode': 'softmax',
        },
    },
    'C': {
        'label': 'mp_sigmoid',
        'desc': 'mp_enabled=true, gate_mode=sigmoid — unbounded (original)',
        'overrides': {
            'mp_enabled': True,
            'mp_heads': 4,
            'mp_gate_mode': 'sigmoid',
        },
    },
}


def build_model(base_model_cfg, overrides, seed, device):
    torch.manual_seed(seed)
    cfg = dict(base_model_cfg)
    cfg.update(overrides)
    spec = _build_instnct_spec(embed_mode=True, model_config=cfg)
    record = {'type': 'instnct', 'build_spec': spec}
    model = build_model_from_spec(record, device)
    return model


def run_one(config_name, config, base_model_cfg, dataset, args, seed, out_dir):
    device = args.device
    label = config['label']
    tag = f'{config_name}_{label}'

    model = build_model(base_model_cfg, config['overrides'], seed, device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    warmup = args.warmup
    csv_path = out_dir / f'{tag}_seed{seed}.csv'
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    # Extended header for diagnostics
    csv_writer.writerow([
        'step', 'loss', 'acc', 'lr', 'elapsed_s',
        'ring_norm', 'ring_signal_norm_0', 'hidden_norm_0',
        'alpha_0_mean', 'ptr_pos_0',
        'mp_gate_0_head0', 'mp_gate_0_head1', 'mp_gate_0_head2', 'mp_gate_0_head3',
        'mp_gate_entropy_0',
    ])

    losses = []
    accs = []
    t0 = time.time()

    # State for sequential training
    state = None

    for step in range(1, args.steps + 1):
        if warmup > 0 and step <= warmup:
            lr_now = args.lr * step / warmup
            for pg in opt.param_groups:
                pg['lr'] = lr_now
        else:
            lr_now = args.lr

        xb, yb, mask = dataset.sample_batch(args.batch, device)

        pred, state = model(xb, state=state)
        # Detach state to prevent BPTT across batches
        if state is not None:
            state = {k: v.detach() if isinstance(v, torch.Tensor) else v
                     for k, v in state.items()}
        _, masked_loss = func_maskloss_ce(pred, yb, mask)

        opt.zero_grad()
        masked_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()

        lv = masked_loss.item()
        losses.append(lv)

        with torch.no_grad():
            preds = pred.argmax(dim=-1)
            correct = (preds == yb).float() * mask
            acc = correct.sum() / mask.sum().clamp(min=1)
            accs.append(acc.item())

        # Diagnostics from model
        diag = getattr(model, '_diag', {})

        if step % 10 == 0 or step == 1:
            elapsed = time.time() - t0
            csv_writer.writerow([
                step, f'{lv:.6f}', f'{acc.item():.4f}', f'{lr_now:.6f}',
                f'{elapsed:.1f}',
                f'{diag.get("ring_norm", 0):.2f}',
                f'{diag.get("ring_signal_norm_0", 0):.2f}',
                f'{diag.get("hidden_norm_0", 0):.2f}',
                f'{diag.get("alpha_0_mean", 0):.4f}',
                f'{diag.get("ptr_pos_0", 0):.2f}',
                f'{diag.get("mp_gate_0_head0", 0):.4f}',
                f'{diag.get("mp_gate_0_head1", 0):.4f}',
                f'{diag.get("mp_gate_0_head2", 0):.4f}',
                f'{diag.get("mp_gate_0_head3", 0):.4f}',
                f'{diag.get("mp_gate_entropy_0", 0):.4f}',
            ])

        if step % args.log_every == 0 or step == 1:
            avg_loss = sum(losses[-100:]) / len(losses[-100:])
            avg_acc = sum(accs[-100:]) / len(accs[-100:])
            elapsed = time.time() - t0
            bpc = avg_loss / np.log(2)
            rn = diag.get('ring_norm', 0)
            rsn = diag.get('ring_signal_norm_0', 0)
            hn = diag.get('hidden_norm_0', 0)
            al = diag.get('alpha_0_mean', 0)
            print(f'  [{tag}] step {step:5d}/{args.steps}  '
                  f'loss={avg_loss:.4f}  bpc={bpc:.3f}  acc={avg_acc:.3f}  '
                  f'ring={rn:.0f}  sig={rsn:.0f}  hid={hn:.0f}  α={al:.3f}  '
                  f'{elapsed:.0f}s')

    csv_file.close()
    elapsed = time.time() - t0

    tail = min(200, len(losses))
    final_loss = sum(losses[-tail:]) / tail
    final_acc = sum(accs[-tail:]) / tail

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
        'time_s': elapsed,
        'csv_log': str(csv_path),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Multi-Pointer Gate Fix — A/B/C Sweep')
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--seq', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--device', default=None)
    parser.add_argument('--configs', type=str, default='A,B,C')
    parser.add_argument('--log-every', type=int, default=50)
    parser.add_argument('--data', type=str, default=None)
    args = parser.parse_args()

    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config_names = [c.strip().upper() for c in args.configs.split(',')]
    for c in config_names:
        if c not in SWEEP_CONFIGS:
            print(f'[ERROR] Unknown config: {c}. Available: {list(SWEEP_CONFIGS.keys())}')
            sys.exit(1)

    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = V4_ROOT / 'sweep_results' / f'mp_gate_fix_{run_id}'
    out_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 72)
    print('  VRAXION v4 — Multi-Pointer Gate Fix Sweep')
    print('=' * 72)
    print(f'Steps:    {args.steps}')
    print(f'Batch:    {args.batch} x {args.seq}')
    print(f'LR:       {args.lr} (warmup {args.warmup})')
    print(f'Device:   {args.device}')
    print(f'Configs:  {config_names}')
    print(f'Output:   {out_dir}')
    print()

    for c in config_names:
        cfg = SWEEP_CONFIGS[c]
        print(f'  {c}: {cfg["label"]:15s} — {cfg["desc"]}')
    print()

    base_model_cfg = load_model_config(V4_ROOT)
    data_dir = Path(args.data) if args.data else V4_ROOT / 'training_data'
    files = func_discover_dat(str(data_dir))
    if not files:
        print(f'[ERROR] No .traindat files found in {data_dir}')
        sys.exit(1)
    dataset = ByteDataset(files, args.seq, embed_mode=True, seed=42)
    print(f'Data: {len(files)} shard(s), {dataset.total_bytes / 1e6:.0f} MB')
    print()

    all_results = []
    seed = 42

    for config_name in config_names:
        config = SWEEP_CONFIGS[config_name]
        dataset.rng = np.random.default_rng(seed)

        if args.device.startswith('cuda'):
            torch.cuda.empty_cache()

        result = run_one(config_name, config, base_model_cfg,
                         dataset, args, seed, out_dir)
        all_results.append(result)

        print(f'\n  -> {config_name} ({config["label"]}): '
              f'loss={result["final_loss"]:.4f}  '
              f'bpc={result["final_bpc"]:.3f}  '
              f'acc={result["final_acc"]:.3f}  '
              f'params={result["params"]:,}  '
              f'({result["time_s"]:.0f}s)\n')

    # Save results
    results_path = out_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Summary
    print('=' * 80)
    print('  SUMMARY — Multi-Pointer Gate Fix Sweep')
    print('=' * 80)
    print(f'{"Config":8s} {"Label":15s} {"Params":>8s} {"Final Loss":>10s} '
          f'{"BPC":>8s} {"Acc":>8s} {"Time":>8s}')
    print('-' * 80)
    for r in all_results:
        print(f'{r["config"]:8s} {r["label"]:15s} {r["params"]:8,d} '
              f'{r["final_loss"]:10.4f} {r["final_bpc"]:8.3f} '
              f'{r["final_acc"]:8.3f} {r["time_s"]:7.0f}s')
    print()
    print(f'Results saved to: {results_path}')


if __name__ == '__main__':
    main()
