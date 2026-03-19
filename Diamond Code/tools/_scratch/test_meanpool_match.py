#!/usr/bin/env python3
"""
Quick matching test: verify the mean-pool code change in swarm_model.py
produces the same ~90% accuracy as the monkey-patched probe.

Runs ONE config (r=8, actual model code) with seed=42.
Expected: ~89.4-90.0% tail_median (matching probe_attention_v3 mean_pool result).
"""

import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F

D             = 128
DEPTH         = 4
SEQ_LEN       = 32
BLOCK_SIZE    = 4
BATCH         = 16
LR            = 1e-3
STEPS         = 500
SEED          = 42
NUM_BITS      = 8
STEP_TIMEOUT  = 60
DEVICE        = torch.device('cpu')
RADIUS        = 8

DIAMOND_ROOT = r'S:\AI\work\VRAXION_DEV\Diamond Code'
sys.path.insert(0, DIAMOND_ROOT)

from swarm_model import SwarmByteRingModel


def byte_to_bits(byte_seq, num_bits=8):
    t = torch.tensor(byte_seq, dtype=torch.uint8)
    return ((t.unsqueeze(-1) >> torch.arange(num_bits)) & 1).float()


def make_echo_batch(batch_size, seq_len, block_size=4, num_bits=8):
    xs, ys = [], []
    for _ in range(batch_size):
        block = [random.randint(0, 255) for _ in range(block_size)]
        repeats = (seq_len + 2) // block_size + 1
        data = (block * repeats)[:seq_len + 1]
        xs.append(byte_to_bits(data[:seq_len], num_bits))
        ys.append(byte_to_bits(data[1:seq_len + 1], num_bits))
    return torch.stack(xs).to(DEVICE), torch.stack(ys).to(DEVICE)


if __name__ == '__main__':
    torch.manual_seed(SEED)
    random.seed(SEED)

    model = SwarmByteRingModel(
        embedding_dim=D,
        num_memory_positions=SEQ_LEN,
        num_beings=1,
        depth=DEPTH,
        num_bits=NUM_BITS,
        attention_radius=RADIUS,
        attention_temperature=8.0,  # ignored now (uniform weights)
        think_ticks=0,
        use_lcx=False,
        num_pointers=1,
    ).to(DEVICE)
    model.train()

    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    print(f'Matching test: mean-pool in swarm_model.py (r={RADIUS}, seed={SEED})')
    print(f'  params={n_params:,}')
    print(f'  Expected tail_median: ~0.894 (from probe v3 mean_pool seed=42)')
    print()

    tail_accs = []
    for step in range(STEPS):
        t0 = time.time()
        x, y = make_echo_batch(BATCH, SEQ_LEN, BLOCK_SIZE, NUM_BITS)

        opt.zero_grad()
        out = model(x)
        if isinstance(out, tuple):
            out = out[0]

        loss = F.binary_cross_entropy_with_logits(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        elapsed = time.time() - t0
        if elapsed > STEP_TIMEOUT:
            print(f'TIMEOUT: step {step} took {elapsed:.0f}s')
            sys.exit(1)

        with torch.no_grad():
            pred = (out > 0).float()
            acc = (pred == y).float().mean().item()

        if math.isnan(loss.item()):
            print(f'NaN at step {step}')
            sys.exit(1)

        if step >= STEPS - 100:
            tail_accs.append(acc)

        if step % 100 == 0 or step == STEPS - 1:
            print(f'  step {step:4d} | loss {loss.item():.6f} | acc {acc:.4f} | {elapsed:.2f}s')

    tail_median = sorted(tail_accs)[len(tail_accs) // 2]
    expected = 0.8943  # from probe v3 mean_pool seed=42

    print(f'\n  tail_median = {tail_median:.4f}')
    print(f'  expected    = {expected:.4f}')
    print(f'  delta       = {tail_median - expected:+.4f}')

    if abs(tail_median - expected) < 0.02:
        print(f'\n  MATCH: code change produces same result as monkey-patch')
    else:
        print(f'\n  MISMATCH: investigate! Delta > 2%')
