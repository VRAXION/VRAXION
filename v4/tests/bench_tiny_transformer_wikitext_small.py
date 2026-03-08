"""Small, param-matched TinyTransformer baseline on WikiText-103.

This script mirrors the small-model CPU WikiText LL probe:
  - byte-level WikiText shards
  - CPU
  - 10k steps
  - batch=8, seq=8

Default transformer config is chosen to match the small INSTNCT probe size
as closely as possible:
  d_model=32, n_layers=1, n_heads=2, d_ff=32, max_seq=16
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

V4_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V4_ROOT / 'model'))
sys.path.insert(0, str(V4_ROOT / 'training'))

from tiny_transformer import TinyTransformer
from train import ByteDataset, func_discover_dat, func_maskloss_ce


def _default_json_path():
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = V4_ROOT / 'dev_notes' / 'telemetry'
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f'{Path(__file__).stem}_{stamp}.json'


def _set_determinism(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def build_model(device, d_model, n_layers, n_heads, d_ff, max_seq, dropout):
    model = TinyTransformer(
        embed_mode=True,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        max_seq=max_seq,
        dropout=dropout,
    )
    return model.to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--seq', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--d-model', type=int, default=32)
    parser.add_argument('--n-layers', type=int, default=1)
    parser.add_argument('--n-heads', type=int, default=2)
    parser.add_argument('--d-ff', type=int, default=32)
    parser.add_argument('--max-seq', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--json-out', type=str, default='')
    args = parser.parse_args()

    _set_determinism(args.seed)

    print('=== Small TinyTransformer WikiText Baseline ===')
    print(
        f'Steps: {args.steps}  Seed: {args.seed}  Device: {args.device}  '
        f'Batch: {args.batch}x{args.seq}'
    )
    print(
        f'Transformer: d_model={args.d_model} layers={args.n_layers} '
        f'heads={args.n_heads} d_ff={args.d_ff} max_seq={args.max_seq}'
    )
    print()

    data_dir = V4_ROOT / 'training_data'
    if not data_dir.exists():
        fallback_dir = Path(r'S:\AI\work\VRAXION_DEV\v4\training_data')
        if fallback_dir.exists():
            data_dir = fallback_dir
    files = func_discover_dat(str(data_dir))
    dataset = ByteDataset(files, args.seq, embed_mode=True, seed=args.seed)
    print(f'Data: {len(files)} shards, {dataset.total_bytes / 1e6:.0f} MB')

    model = build_model(
        device=args.device,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        max_seq=args.max_seq,
        dropout=args.dropout,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Params: {n_params:,}')
    print()

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    amp_enabled = args.device == 'cuda'
    scaler = torch.amp.GradScaler(args.device, enabled=amp_enabled)

    losses = []
    accs = []
    grad_norms = []
    max_grad = 0.0
    t0 = time.time()

    for step in range(1, args.steps + 1):
        xb, yb, mask = dataset.sample_batch(args.batch, args.device)

        with torch.amp.autocast(args.device, enabled=amp_enabled):
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
        max_grad = max(max_grad, gn)

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
            print(
                f'  [transformer] step {step:5d}/{args.steps}  '
                f'loss={avg_loss:.4f}  bpc={avg_loss * 1.4427:.3f}  '
                f'acc={avg_acc:.3f}  gnorm={avg_gn:.1f}  '
                f'{elapsed:.0f}s {spike}'
            )

    elapsed = time.time() - t0
    tail = min(100, len(losses))
    result = {
        'variant': 'transformer-small',
        'seed': args.seed,
        'steps': args.steps,
        'params': n_params,
        'final_loss': sum(losses[-tail:]) / tail,
        'final_bpc': sum(losses[-tail:]) / tail * 1.4427,
        'final_acc': sum(accs[-tail:]) / tail,
        'best_loss': min(losses),
        'best_acc': max(accs),
        'best_step': int(np.argmax(np.array(accs)) + 1),
        'time_s': elapsed,
        's_per_step': elapsed / args.steps,
        'max_grad': max_grad,
        'grad_spikes': sum(1 for g in grad_norms if g > 50),
        'loss_curve': losses,
        'acc_curve': accs,
        'config': {
            'device': args.device,
            'batch': args.batch,
            'seq': args.seq,
            'd_model': args.d_model,
            'n_layers': args.n_layers,
            'n_heads': args.n_heads,
            'd_ff': args.d_ff,
            'max_seq': args.max_seq,
            'dropout': args.dropout,
            'lr': args.lr,
        },
    }

    json_out = Path(args.json_out) if args.json_out else _default_json_path()
    json_out.parent.mkdir(parents=True, exist_ok=True)
    with open(json_out, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)

    print()
    print(
        f'Final: loss={result["final_loss"]:.4f}  '
        f'bpc={result["final_bpc"]:.3f}  '
        f'acc={result["final_acc"]:.3f}  '
        f'best_acc={result["best_acc"]:.3f}  '
        f'best_step={result["best_step"]}  '
        f'time={result["time_s"]:.0f}s'
    )
    print(f'Saved telemetry JSON: {json_out}')


if __name__ == '__main__':
    main()
