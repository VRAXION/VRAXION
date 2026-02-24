"""B/D ratio sweep — does the expansion ratio matter more than absolute B?

Tests B and D together to separate two effects:
  1. Does B/D ratio determine performance? (A vs B: same ratio, different scale)
  2. Does absolute B matter at fixed ratio? (A vs D: same D, different B)

Usage:
    python sweep_bd_ratio.py
    python sweep_bd_ratio.py --steps 200
"""

import argparse
import random
import time

import numpy as np
import torch
import torch.nn as nn

from instnct import INSTNCT

# ── Echo data generator ──────────────────────────────────────────

BLOCK = 16
ECHO_REPEAT = 8

def gen_echo(size, seed=42):
    random.seed(seed)
    data = bytearray()
    while len(data) < size:
        block = bytes(random.randint(0, 255) for _ in range(BLOCK))
        for _ in range(ECHO_REPEAT):
            data.extend(block)
    return bytes(data[:size])


def load_binary_data(raw_bytes, B, seq_len):
    bits = np.unpackbits(np.frombuffer(raw_bytes, dtype=np.uint8))
    total_pos = len(bits) // B
    bits = bits[:total_pos * B].reshape(total_pos, B).astype(np.float32)
    chunk = seq_len + 1
    n_samples = total_pos // chunk
    bits = bits[:n_samples * chunk].reshape(n_samples, chunk, B)
    x = torch.from_numpy(bits[:, :seq_len].copy())
    y = torch.from_numpy(bits[:, 1:seq_len + 1].copy())
    return x, y


def train_one_config(B, D, raw_bytes, steps, batch_size, seq_len, device):
    x_all, y_all = load_binary_data(raw_bytes, B, seq_len)
    n_samples = x_all.shape[0]
    if n_samples < batch_size:
        batch_size = max(1, n_samples)

    model = INSTNCT(D=D, B=B).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    params = sum(p.numel() for p in model.parameters())
    losses = []
    best_loss = float('inf')
    best_step = 0
    t0 = time.time()

    for step in range(steps):
        idx = torch.randint(0, n_samples, (batch_size,))
        xb = x_all[idx].to(device)
        yb = y_all[idx].to(device)

        pred = model(xb)
        loss = loss_fn(pred, yb)

        opt.zero_grad()
        loss.backward()
        opt.step()

        lv = loss.item()
        losses.append(lv)
        if lv < best_loss:
            best_loss = lv
            best_step = step

        if (step + 1) % 50 == 0:
            print(f"    step {step+1:4d}  loss={lv:.6f}")

    elapsed = time.time() - t0
    return {
        'B': B, 'D': D, 'ratio': D / B,
        'seq_len': seq_len, 'params': params, 'samples': n_samples,
        'final_loss': losses[-1], 'best_loss': best_loss,
        'best_step': best_step, 'elapsed': elapsed,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='B/D ratio sweep for INSTNCT v4')
    parser.add_argument('--steps', type=int, default=200, help='training steps per config')
    parser.add_argument('--batch', type=int, default=32, help='batch size')
    parser.add_argument('--seq-len', type=int, default=32, help='sequence length (short for speed)')
    parser.add_argument('--data-size', type=int, default=32768, help='raw echo data bytes')
    parser.add_argument('--device', default='cpu', help='device (cpu/cuda)')
    args = parser.parse_args()

    # (B, D, label)
    CONFIGS = [
        (8,  64,  '8x  baseline'),
        (16, 128, '8x  scaled up'),
        (8,  32,  '4x  small'),
        (16, 64,  '4x  more bits'),
    ]

    print('B/D Ratio Sweep -- INSTNCT v4')
    print('=' * 65)
    print(f'Steps: {args.steps}  Batch: {args.batch}  Seq: {args.seq_len}  Data: {args.data_size:,}B')
    print(f'Device: {args.device}')
    print()

    raw = gen_echo(args.data_size)
    print(f'Echo data: {len(raw):,} bytes')
    print()

    results = []
    for B, D, label in CONFIGS:
        print(f'--- B={B} D={D} ({D//B}x) {label} ---')
        r = train_one_config(B, D, raw, args.steps, args.batch, args.seq_len, args.device)
        r['label'] = label
        results.append(r)
        print(f'    done in {r["elapsed"]:.1f}s  best={r["best_loss"]:.6f} @ step {r["best_step"]}')
        print()

    print('=' * 65)
    print('RESULTS')
    print('=' * 65)
    print(f'{"B":>3} {"D":>4} {"ratio":>5} {"params":>7} {"samples":>7} {"best":>10} {"final":>10}  {"label"}')
    print('-' * 65)
    best_overall = min(r['best_loss'] for r in results)
    for r in results:
        tag = ' *' if r['best_loss'] == best_overall else ''
        print(f'{r["B"]:>3} {r["D"]:>4} {r["ratio"]:>5.0f}x {r["params"]:>7} {r["samples"]:>7} {r["best_loss"]:>10.4f} {r["final_loss"]:>10.4f}  {r["label"]}{tag}')
    print()
    print('* = lowest best_loss')
    print()
    print('If A ~ B (same ratio, different scale) -> ratio matters, not absolute B')
    print('If A ~ D (same D, different B)         -> D matters, not B')
    print('If A beats all                         -> B=8 D=64 is the sweet spot')
