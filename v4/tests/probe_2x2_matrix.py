"""2×2 Diagnostic Matrix: {sequential, pilot} × {accumulate, replace}

Feeds identical data through all 4 configs, measures ring health + training speed.
Runs on CPU, ~2 min total. Deterministic (fixed seed).

Usage:
  python tests/probe_2x2_matrix.py
  python tests/probe_2x2_matrix.py --train-steps 400 --stream-steps 200
"""
import sys, os, time, argparse
import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'model'))
sys.path.insert(0, os.path.dirname(__file__))

from instnct import INSTNCT
from bench_fast_memory import generate_repeating_pattern


def create_model(write_mode, pointer_mode, seed=42):
    torch.manual_seed(seed)
    return INSTNCT(
        M=256, hidden_dim=512, slot_dim=64, N=1, R=2,
        embed_mode=True, kernel_mode='vshape',
        embed_encoding='learned', output_encoding='lowrank_c19',
        expert_weighting=False, checkpoint_chunks=0,
        bb_enabled=False, io_split_mode='off',
        pointer_mode=pointer_mode, write_mode=write_mode,
    )


def measure_ring(state):
    """Extract ring health metrics from model state."""
    ring = state['ring']  # (B, M, D)
    B, M, D = ring.shape

    # Ring norm (mean across batch)
    ring_norm = ring.norm(dim=-1).mean().item()

    # Adjacent cosine similarity
    ring_flat = ring[0]  # first batch element
    norms = ring_flat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    ring_normed = ring_flat / norms
    cos_sims = (ring_normed[:-1] * ring_normed[1:]).sum(-1)
    adj_cos = cos_sims.mean().item()

    # Rank (90% variance explained)
    centered = ring_flat - ring_flat.mean(0, keepdim=True)
    try:
        S = torch.linalg.svdvals(centered)
        cum_var = (S ** 2).cumsum(0) / (S ** 2).sum().clamp(min=1e-8)
        rank90 = (cum_var < 0.90).sum().item() + 1
    except Exception:
        rank90 = -1

    # Pointer coverage
    ptr = state.get('ptr')
    if ptr is not None:
        positions = ptr[0].long().clamp(0, M - 1)  # expert 0
        unique = positions.unique().numel()
        ptr_cov = unique / M * 100
    else:
        ptr_cov = 0.0

    return {
        'ring_norm': ring_norm,
        'adj_cos': adj_cos,
        'rank90': rank90,
        'ptr_cov': ptr_cov,
    }


def stream_steps(model, steps, seed=42):
    """Forward-only streaming (no training). Returns final state."""
    torch.manual_seed(seed)
    data, _ = generate_repeating_pattern(B=8, length=steps + 1, period=8, seed=seed)
    state = None
    chunk = 32
    model.eval()
    with torch.no_grad():
        for t in range(0, steps, chunk):
            end = min(t + chunk, steps)
            x = data[:, t:end]
            _, state = model(x, state=state)
    return state


def train_and_measure(model, steps, seed=42):
    """Train on synthetic data, return final loss/acc and state."""
    torch.manual_seed(seed)
    data, mask = generate_repeating_pattern(B=32, length=32, period=8, seed=seed)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    state = None
    final_loss = 0
    final_acc = 0

    for step in range(steps):
        x = data[:, :-1]
        y = data[:, 1:]
        m = mask[:, 1:]

        logits, state = model(x, state=state)
        state = {k: v.detach() for k, v in state.items()}

        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1), reduction='none')
        loss = (loss.view(y.shape) * m).sum() / m.sum().clamp(min=1)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        final_loss = loss.item()
        acc = ((logits.argmax(-1) == y).float() * m).sum() / m.sum()
        final_acc = acc.item()

    return final_loss, final_acc, state


def main():
    parser = argparse.ArgumentParser(description='2x2 diagnostic matrix')
    parser.add_argument('--stream-steps', type=int, default=200)
    parser.add_argument('--train-steps', type=int, default=400)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    configs = [
        ('seq+accum',    'accumulate', 'sequential'),
        ('seq+replace',  'replace',    'sequential'),
        ('pilot+accum',  'accumulate', 'pilot'),
        ('pilot+replace','replace',    'pilot'),
    ]

    print('=' * 80)
    print('  2×2 DIAGNOSTIC MATRIX: {sequential, pilot} × {accumulate, replace}')
    print(f'  Stream: {args.stream_steps} steps | Train: {args.train_steps} steps | Seed: {args.seed}')
    print('=' * 80)

    results = []

    for name, wm, pm in configs:
        print(f'\n--- {name} (write={wm}, pointer={pm}) ---')
        t0 = time.perf_counter()

        # 1. Stream (forward only)
        model = create_model(wm, pm, seed=args.seed)
        n_params = sum(p.numel() for p in model.parameters())
        stream_state = stream_steps(model, args.stream_steps, seed=args.seed)
        stream_metrics = measure_ring(stream_state)
        print(f'  STREAM: adj_cos={stream_metrics["adj_cos"]:.4f}  '
              f'rank90={stream_metrics["rank90"]}  '
              f'ring_norm={stream_metrics["ring_norm"]:.1f}  '
              f'ptr_cov={stream_metrics["ptr_cov"]:.1f}%')

        # 2. Train (fresh model, same seed)
        model2 = create_model(wm, pm, seed=args.seed)
        loss, acc, train_state = train_and_measure(model2, args.train_steps, seed=args.seed)
        train_metrics = measure_ring(train_state)
        elapsed = time.perf_counter() - t0
        print(f'  TRAIN:  loss={loss:.4f}  acc={acc*100:.1f}%  '
              f'adj_cos={train_metrics["adj_cos"]:.4f}  '
              f'ring_norm={train_metrics["ring_norm"]:.1f}  '
              f'({elapsed:.1f}s)')

        results.append({
            'name': name, 'params': n_params,
            'wm': wm, 'pm': pm,
            'stream_adj_cos': stream_metrics['adj_cos'],
            'stream_rank90': stream_metrics['rank90'],
            'stream_ring_norm': stream_metrics['ring_norm'],
            'stream_ptr_cov': stream_metrics['ptr_cov'],
            'train_loss': loss, 'train_acc': acc,
            'train_adj_cos': train_metrics['adj_cos'],
            'train_ring_norm': train_metrics['ring_norm'],
        })

    # Summary table
    print('\n' + '=' * 80)
    print('  SUMMARY')
    print('=' * 80)
    print(f'  {"Config":<16} {"Params":>7} | {"Stream adj_cos":>14} {"rank90":>6} '
          f'| {"Train loss":>10} {"acc":>6} {"adj_cos":>8} {"ring_norm":>10}')
    print(f'  {"-"*16} {"-"*7} | {"-"*14} {"-"*6} | {"-"*10} {"-"*6} {"-"*8} {"-"*10}')

    for r in results:
        blob = 'BLOB!' if r['stream_adj_cos'] > 0.95 else 'ok'
        print(f'  {r["name"]:<16} {r["params"]:>7,} | '
              f'{r["stream_adj_cos"]:>10.4f} [{blob:>4}] {r["stream_rank90"]:>5} | '
              f'{r["train_loss"]:>10.4f} {r["train_acc"]*100:>5.1f}% '
              f'{r["train_adj_cos"]:>8.4f} {r["train_ring_norm"]:>10.1f}')

    # Verdicts
    print('\n  VERDICTS:')
    sr = {r['name']: r for r in results}

    # Replace vs accumulate (sequential)
    if 'seq+accum' in sr and 'seq+replace' in sr:
        a, b = sr['seq+accum'], sr['seq+replace']
        if b['stream_adj_cos'] < a['stream_adj_cos'] - 0.05:
            print(f'    [REPLACE WRITE] Fixes blob: adj_cos {a["stream_adj_cos"]:.3f} -> {b["stream_adj_cos"]:.3f}')
        else:
            print(f'    [REPLACE WRITE] No clear blob fix: {a["stream_adj_cos"]:.3f} vs {b["stream_adj_cos"]:.3f}')

    # Pilot vs sequential (replace)
    if 'seq+replace' in sr and 'pilot+replace' in sr:
        a, b = sr['seq+replace'], sr['pilot+replace']
        if b['train_loss'] < a['train_loss'] * 0.8:
            print(f'    [PILOT POINTER] Faster learning: loss {a["train_loss"]:.4f} -> {b["train_loss"]:.4f} ({b["train_loss"]/a["train_loss"]:.1%})')
        else:
            print(f'    [PILOT POINTER] No clear speed gain: {a["train_loss"]:.4f} vs {b["train_loss"]:.4f}')

    print()


if __name__ == '__main__':
    main()
