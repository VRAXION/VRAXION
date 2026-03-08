"""Top-5 feature candidate sweep on WikiText-103.

Tests 5 categories of untested features against baseline:
  1. Pointer mode: sequential vs pilot vs jump_gate
  2. Ring size M: 128 vs 256 vs 512
  3. MTAPS: off vs current vs tap_scalar_gate
  4. Slot dim: 64 vs 128 vs 256
  5. Bulletin Board: off vs fixed vs learned

Base config: H=512, slot_dim=64, M=128, N=1, R=2, vshape, replace write,
dualphi c19, learned embed, lowrank_c19 output.

Usage:
  python tests/sweep_top5_candidates.py
  python tests/sweep_top5_candidates.py --steps 500 --batch 32 --seq 256 --device cpu
"""

import argparse
import csv
import json
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

V4_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V4_ROOT / 'model'))
sys.path.insert(0, str(V4_ROOT / 'training'))

from train import ByteDataset, func_discover_dat, func_maskloss_ce
from model_factory import build_model_from_spec


def _set_determinism(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


BASE_SPEC = dict(
    M=128,
    hidden_dim=512,
    slot_dim=64,
    N=1,
    R=2,
    B=8,
    embed_mode=True,
    kernel_mode='vshape',
    read_kernel_mode='vshape',
    embed_encoding='learned',
    output_encoding='lowrank_c19',
    write_mode='replace',
    replace_impl='dense',
    c19_mode='dualphi',
    pointer_mode='sequential',
    pointer_interp_mode='linear',
    pointer_seam_mode='shortest_arc',
    checkpoint_chunks=0,
    expert_weighting=False,
    s_constraint='softplus',
    bb_enabled=False,
    bb_gate_bias=0.0,
    bb_scale=0.05,
    bb_tau=4.0,
    bb_gate_mode='fixed',
    mtaps_enabled=False,
    mtaps_lags=[1, 2, 4, 8, 16, 32, 64],
    mtaps_mixer_mode='current',
    mtaps_aux_fixed_offsets=[],
    topk_K=8,
    read_topk_K=8,
    write_address_mode='pointer',
    write_topk_K=2,
    io_split_mode='off',
    io_writer_count=1,
    io_output_from_readers_only=False,
    gated_write=False,
    jump_gate=False,
)


def make_configs():
    """Define all sweep configs as (name, label, overrides_dict)."""
    configs = []

    # Baseline (shared across categories)
    configs.append(('0_baseline', 'baseline', {}))

    # Cat 1: Pointer mode
    configs.append(('1B_ptr_pilot', 'pilot pointer', {'pointer_mode': 'pilot'}))
    configs.append(('1C_ptr_jumpgate', 'seq + jump_gate', {'jump_gate': True}))

    # Cat 2: Ring size M
    configs.append(('2B_M256', 'M=256', {'M': 256}))
    configs.append(('2C_M512', 'M=512', {'M': 512}))

    # Cat 3: MTAPS
    configs.append(('3B_mtaps_current', 'mtaps current', {
        'mtaps_enabled': True, 'mtaps_mixer_mode': 'current',
    }))
    configs.append(('3C_mtaps_gate', 'mtaps gate', {
        'mtaps_enabled': True, 'mtaps_mixer_mode': 'tap_scalar_gate',
    }))

    # Cat 4: Slot dim
    configs.append(('4B_slot128', 'slot_dim=128', {'slot_dim': 128}))
    configs.append(('4C_slot256', 'slot_dim=256', {'slot_dim': 256}))

    # Cat 5: Bulletin Board
    configs.append(('5B_bb_fixed', 'BB fixed', {
        'bb_enabled': True, 'bb_gate_mode': 'fixed',
    }))
    configs.append(('5C_bb_learned', 'BB learned', {
        'bb_enabled': True, 'bb_gate_mode': 'learned',
    }))

    return configs


def run_one(config_name, spec, dataset, steps, batch_size, seq_len,
            seed, lr, grad_clip, device, out_dir, sequential_train=True,
            log_every=50, warmup_steps=50):
    """Train one config variant and return results dict."""
    _set_determinism(seed)

    record = {'type': 'instnct', 'build_spec': spec}
    model = build_model_from_spec(record, device)
    n_params = sum(p.numel() for p in model.parameters())

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    amp_enabled = (device == 'cuda')
    scaler = torch.amp.GradScaler(device, enabled=amp_enabled)

    # Reset dataset for determinism
    dataset.rng = np.random.default_rng(seed)
    if sequential_train:
        dataset.init_sequential(batch_size)

    csv_path = out_dir / f'{config_name}_seed{seed}.csv'
    csv_file = open(csv_path, 'w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(['step', 'loss', 'bpc', 'acc', 'gnorm', 'lr', 'elapsed_s'])

    losses = []
    accs = []
    gnorms = []
    state = None
    t0 = time.time()

    for step in range(1, steps + 1):
        # LR warmup
        cur_lr = lr * min(1.0, step / max(warmup_steps, 1))
        for pg in opt.param_groups:
            pg['lr'] = cur_lr

        if sequential_train:
            xb, yb, mask = dataset.sample_batch_sequential(batch_size, device)
        else:
            xb, yb, mask = dataset.sample_batch(batch_size, device)

        with torch.amp.autocast(device, enabled=amp_enabled):
            pred, state = model(xb, state=state)
            _, masked_loss = func_maskloss_ce(pred, yb, mask)

        # Detach state for TBPTT
        if state is not None:
            if isinstance(state, dict):
                state = {k: v.detach() if isinstance(v, torch.Tensor) else v
                         for k, v in state.items()}
            else:
                state = tuple(s.detach() if isinstance(s, torch.Tensor) else s for s in state)

        opt.zero_grad()
        scaler.scale(masked_loss).backward()
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip).item()
        scaler.step(opt)
        scaler.update()

        lv = masked_loss.item()
        losses.append(lv)
        gnorms.append(gn)

        with torch.no_grad():
            preds = pred.argmax(dim=-1)
            acc = ((preds == yb).float() * mask).sum() / mask.sum().clamp(min=1)
            accs.append(acc.item())

        elapsed = time.time() - t0
        writer.writerow([step, f'{lv:.6f}', f'{lv * 1.4427:.4f}',
                         f'{acc.item():.6f}', f'{gn:.4f}', f'{cur_lr:.6f}',
                         f'{elapsed:.1f}'])

        if step % log_every == 0 or step == 1:
            tail = min(log_every, len(losses))
            avg_loss = sum(losses[-tail:]) / tail
            avg_acc = sum(accs[-tail:]) / tail
            avg_gn = sum(gnorms[-tail:]) / tail
            print(f'  [{config_name}] step {step:4d}/{steps}  '
                  f'loss={avg_loss:.4f}  bpc={avg_loss*1.4427:.3f}  '
                  f'acc={avg_acc:.3f}  gnorm={avg_gn:.2f}  '
                  f'{elapsed:.0f}s')

    csv_file.close()
    elapsed = time.time() - t0
    tail = min(100, len(losses))

    return {
        'config': config_name,
        'seed': seed,
        'steps': steps,
        'params': n_params,
        'lr': lr,
        'final_loss': sum(losses[-tail:]) / tail,
        'final_bpc': sum(losses[-tail:]) / tail * 1.4427,
        'final_acc': sum(accs[-tail:]) / tail,
        'best_loss': min(losses),
        'best_bpc': min(losses) * 1.4427,
        'best_acc': max(accs),
        'final_gnorm': sum(gnorms[-tail:]) / tail,
        'time_s': elapsed,
        's_per_step': elapsed / steps,
        'csv_log': str(csv_path),
    }


def main():
    parser = argparse.ArgumentParser(description='Top-5 feature candidate sweep')
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--seq', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--grad-clip', type=float, default=10.0)
    parser.add_argument('--warmup', type=int, default=50)
    parser.add_argument('--log-every', type=int, default=50)
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--data', type=str, default='')
    parser.add_argument('--configs', type=str, default='',
                        help='Comma-separated config names to run (empty = all)')
    args = parser.parse_args()

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = V4_ROOT / 'sweep_results' / f'top5_sweep_{stamp}'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Data
    data_dir = Path(args.data) if args.data else V4_ROOT / 'training_data'
    files = func_discover_dat(str(data_dir))
    dataset = ByteDataset(files, args.seq, embed_mode=True, seed=args.seed)

    print('=' * 90)
    print('TOP-5 FEATURE CANDIDATE SWEEP')
    print('=' * 90)
    print(f'Steps: {args.steps}  Batch: {args.batch}  Seq: {args.seq}  '
          f'LR: {args.lr}  Seed: {args.seed}')
    print(f'Device: {args.device}')
    print(f'Data: {len(files)} shards, {dataset.total_bytes / 1e6:.0f} MB')
    print(f'Output: {out_dir}')
    print()

    all_configs = make_configs()

    # Filter configs if specified
    if args.configs:
        selected = set(args.configs.split(','))
        all_configs = [(n, l, o) for n, l, o in all_configs if n in selected]

    print(f'Configs to run: {len(all_configs)}')
    for name, label, overrides in all_configs:
        if overrides:
            print(f'  {name:25s} [{label}]  overrides: {overrides}')
        else:
            print(f'  {name:25s} [{label}]  (baseline)')
    print()

    results = []
    for idx, (name, label, overrides) in enumerate(all_configs):
        spec = deepcopy(BASE_SPEC)
        spec.update(overrides)

        print(f'\n{"="*90}')
        print(f'[{idx+1}/{len(all_configs)}] {name} — {label}')
        print(f'{"="*90}')

        result = run_one(
            config_name=name,
            spec=spec,
            dataset=dataset,
            steps=args.steps,
            batch_size=args.batch,
            seq_len=args.seq,
            seed=args.seed,
            lr=args.lr,
            grad_clip=args.grad_clip,
            device=args.device,
            out_dir=out_dir,
            sequential_train=True,
            log_every=args.log_every,
            warmup_steps=args.warmup,
        )
        result['label'] = label
        result['overrides'] = overrides
        results.append(result)

        print(f'  -> {name}: loss={result["final_loss"]:.4f}  '
              f'bpc={result["final_bpc"]:.3f}  acc={result["final_acc"]:.3f}  '
              f'best_acc={result["best_acc"]:.3f}  params={result["params"]}  '
              f'({result["time_s"]:.0f}s)')

    # Save results
    payload = {
        'run_id': stamp,
        'args': vars(args),
        'base_spec': BASE_SPEC,
        'configs': {name: {'label': label, 'overrides': overrides}
                    for name, label, overrides in all_configs},
        'results': results,
    }
    results_path = out_dir / 'results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, default=str)
    print(f'\nSaved results: {results_path}')

    # Summary table
    print('\n' + '=' * 110)
    print(f'{"Config":25s} {"Label":20s} {"Params":>8s} {"Final Loss":>11s} '
          f'{"BPC":>8s} {"Final Acc":>10s} {"Best Acc":>10s} {"Time":>8s}')
    print('-' * 110)
    for r in results:
        print(f'{r["config"]:25s} {r["label"]:20s} {r["params"]:8d} '
              f'{r["final_loss"]:11.4f} {r["final_bpc"]:8.3f} '
              f'{r["final_acc"]:10.4f} {r["best_acc"]:10.4f} '
              f'{r["time_s"]:7.0f}s')

    # Per-category winners
    categories = {
        'Pointer': ['0_baseline', '1B_ptr_pilot', '1C_ptr_jumpgate'],
        'Ring M': ['0_baseline', '2B_M256', '2C_M512'],
        'MTAPS': ['0_baseline', '3B_mtaps_current', '3C_mtaps_gate'],
        'Slot Dim': ['0_baseline', '4B_slot128', '4C_slot256'],
        'Bulletin Board': ['0_baseline', '5B_bb_fixed', '5C_bb_learned'],
    }
    results_by_name = {r['config']: r for r in results}

    print('\n' + '=' * 70)
    print('CATEGORY WINNERS')
    print('-' * 70)
    for cat_name, members in categories.items():
        cat_results = [results_by_name[m] for m in members if m in results_by_name]
        if cat_results:
            winner = min(cat_results, key=lambda r: r['best_loss'])
            baseline = results_by_name.get('0_baseline')
            delta = ''
            if baseline and winner['config'] != '0_baseline':
                d = (winner['best_loss'] - baseline['best_loss']) / baseline['best_loss'] * 100
                delta = f'  ({d:+.1f}% vs baseline)'
            print(f'  {cat_name:20s} -> {winner["config"]:25s} '
                  f'best_loss={winner["best_loss"]:.4f}{delta}')
    print('=' * 70)


if __name__ == '__main__':
    main()
