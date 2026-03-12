"""Deterministic CUDA microbench for HDD/replace write implementations.

Compares the current write path against mathematically equivalent rewrites
using the proxy-relevant shapes from the C19 WikiText sweep.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import torch

B = 32
M = 1024
W = 3
D = 128
WARMUP = 20
REPEATS = 100


def _default_paths():
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = Path(__file__).resolve().parent.parent / 'dev_notes' / 'telemetry'
    out_dir.mkdir(parents=True, exist_ok=True)
    return (
        out_dir / f'{Path(__file__).stem}_{stamp}.json',
        out_dir / f'{Path(__file__).stem}_{stamp}.txt',
    )


def current_write(ring_tns, write_vec_tns, expanded_idx_tns, weights_tns, write_strength_tns):
    w = weights_tns.unsqueeze(-1) * write_strength_tns.unsqueeze(1)
    current = ring_tns.gather(1, expanded_idx_tns)
    write_val = write_vec_tns.unsqueeze(1).expand(-1, weights_tns.size(1), -1)
    updated = w * write_val + (1.0 - w) * current
    ring_new = ring_tns.clone()
    ring_new.scatter_(1, expanded_idx_tns, updated)
    return ring_new


def lerp_v2_write(ring_tns, write_vec_tns, expanded_idx_tns, weights_tns, write_strength_tns):
    w = weights_tns.unsqueeze(-1) * write_strength_tns.unsqueeze(1)
    current = ring_tns.gather(1, expanded_idx_tns)
    write_val = write_vec_tns.unsqueeze(1).expand(-1, weights_tns.size(1), -1)
    updated = torch.lerp(current, write_val, w)
    ring_new = ring_tns.clone()
    ring_new.scatter_(1, expanded_idx_tns, updated)
    return ring_new


def delta_v3_write(ring_tns, write_vec_tns, expanded_idx_tns, weights_tns, write_strength_tns):
    w = weights_tns.unsqueeze(-1) * write_strength_tns.unsqueeze(1)
    current = ring_tns.gather(1, expanded_idx_tns)
    write_val = write_vec_tns.unsqueeze(1).expand(-1, weights_tns.size(1), -1)
    updated = current + w * (write_val - current)
    ring_new = ring_tns.clone()
    ring_new.scatter_(1, expanded_idx_tns, updated)
    return ring_new


def time_cuda(fn, ring_tns, write_vec_tns, expanded_idx_tns, weights_tns, write_strength_tns, backward=False):
    for _ in range(WARMUP):
        ring = ring_tns.clone().requires_grad_(backward)
        write_vec = write_vec_tns.clone().requires_grad_(backward)
        write_strength = write_strength_tns.clone().requires_grad_(backward)
        out = fn(ring, write_vec, expanded_idx_tns, weights_tns, write_strength)
        if backward:
            out.sum().backward()
    torch.cuda.synchronize()

    times = []
    for _ in range(REPEATS):
        ring = ring_tns.clone().requires_grad_(backward)
        write_vec = write_vec_tns.clone().requires_grad_(backward)
        write_strength = write_strength_tns.clone().requires_grad_(backward)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn(ring, write_vec, expanded_idx_tns, weights_tns, write_strength)
        if backward:
            out.sum().backward()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return sum(times) / len(times)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json-out', type=str, default='')
    parser.add_argument('--txt-out', type=str, default='')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit('CUDA required')

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    ring_tns = torch.randn(B, M, D, device='cuda')
    write_vec_tns = torch.randn(B, D, device='cuda')
    weights_tns = torch.softmax(torch.randn(B, W, device='cuda'), dim=-1)
    write_strength_tns = torch.sigmoid(torch.randn(B, 1, device='cuda'))
    center = torch.randint(0, M, (B,), device='cuda')
    offsets = torch.arange(-(W // 2), W // 2 + 1, device='cuda')
    indices_tns = (center.unsqueeze(1) + offsets) % M
    expanded_idx_tns = indices_tns.unsqueeze(-1).expand(-1, -1, D)

    current_ref = current_write(ring_tns, write_vec_tns, expanded_idx_tns, weights_tns, write_strength_tns)
    rows = []
    winners = []
    for name, fn in (
        ('current', current_write),
        ('lerp_v2', lerp_v2_write),
        ('delta_v3', delta_v3_write),
    ):
        out = fn(ring_tns, write_vec_tns, expanded_idx_tns, weights_tns, write_strength_tns)
        max_diff = float((current_ref - out).abs().max().item())
        fwd_ms = time_cuda(fn, ring_tns, write_vec_tns, expanded_idx_tns, weights_tns, write_strength_tns, backward=False)
        bwd_ms = time_cuda(fn, ring_tns, write_vec_tns, expanded_idx_tns, weights_tns, write_strength_tns, backward=True)
        rows.append(
            {
                'variant': name,
                'max_diff': max_diff,
                'forward_ms': fwd_ms,
                'backward_ms': bwd_ms,
            }
        )

    base = rows[0]
    for row in rows[1:]:
        row['forward_speedup'] = base['forward_ms'] / row['forward_ms']
        row['backward_speedup'] = base['backward_ms'] / row['backward_ms']
        row['promote'] = (
            row['max_diff'] <= 1e-6
            and (row['forward_speedup'] >= 1.10 or row['backward_speedup'] >= 1.10)
        )
        if row['promote']:
            winners.append(row['variant'])
    base['forward_speedup'] = 1.0
    base['backward_speedup'] = 1.0
    base['promote'] = False

    payload = {
        'script': Path(__file__).name,
        'timestamp': datetime.now().isoformat(),
        'config': {
            'seed': 42,
            'batch': B,
            'ring_slots': M,
            'window': W,
            'slot_dim': D,
            'warmup': WARMUP,
            'repeats': REPEATS,
            'gpu': torch.cuda.get_device_name(0),
        },
        'results': rows,
        'promotion_threshold': {
            'max_diff': 1e-6,
            'min_speedup': 1.10,
        },
        'promoted_variants': winners,
    }

    json_path, txt_path = _default_paths()
    if args.json_out:
        json_path = Path(args.json_out)
    if args.txt_out:
        txt_path = Path(args.txt_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.parent.mkdir(parents=True, exist_ok=True)

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)

    lines = []
    lines.append('=== HDD Write Optimization Benchmark ===')
    lines.append(f'GPU: {payload["config"]["gpu"]}')
    lines.append(f'Shape: B={B}, M={M}, W={W}, D={D}  repeats={REPEATS}')
    lines.append('')
    lines.append(f'{"Variant":10s} {"Max diff":>10s} {"Forward":>10s} {"Backward":>10s} {"Fwd speed":>11s} {"Bwd speed":>11s} {"Promote":>9s}')
    lines.append('-' * 79)
    for row in rows:
        lines.append(
            f'{row["variant"]:10s} {row["max_diff"]:10.3e} {row["forward_ms"]:9.3f}ms '
            f'{row["backward_ms"]:9.3f}ms {row["forward_speedup"]:10.3f}x '
            f'{row["backward_speedup"]:10.3f}x {str(row["promote"]):>9s}'
        )
    lines.append('')
    lines.append(f'Promoted variants: {winners if winners else "none"}')
    text = '\n'.join(lines)
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(text)
        f.write('\n')
    print(text)
    print(f'Saved JSON: {json_path}')
    print(f'Saved TXT: {txt_path}')


if __name__ == '__main__':
    main()
