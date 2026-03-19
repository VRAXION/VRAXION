#!/usr/bin/env python3
"""
Probe: Whiteboard A/B — Brain Only vs Brain + LCX (with squeeze)
================================================================
D=256 for speed (~10x faster than D=2048).
Config A: use_lcx=False (brain only, no whiteboard)
Config B: use_lcx=True  (brain + whiteboard + squeeze glasses)
Same task, same seed, direct comparison.
"""

import gc
import math
import os
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

D              = 256
DEPTH          = 2
SEQ_LEN        = 32
BLOCK_SIZE     = 16
BATCH          = 10
LR             = 1e-4
WARMUP         = 50
NUM_BITS       = 25
STEPS          = 800
STEP_TIMEOUT   = 30
RADIUS         = 8
THINK_TICKS    = 1

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DIAMOND_ROOT = r'S:\AI\work\VRAXION_DEV\Diamond Code'
sys.path.insert(0, DIAMOND_ROOT)
LOG_DIR      = os.path.join(DIAMOND_ROOT, 'logs', 'probe')
LIVE_LOG     = os.path.join(LOG_DIR, 'probe_whiteboard_ab_live.log')
os.makedirs(LOG_DIR, exist_ok=True)

from swarm_model import SwarmByteRingModel


def make_echo_batch_bits(batch_size, seq_len, block_size, num_bits):
    xs, ys = [], []
    for _ in range(batch_size):
        block = torch.randint(0, 2, (block_size, num_bits)).float()
        repeats = (seq_len + 2) // block_size + 1
        data = block.repeat(repeats, 1)[:seq_len + 1]
        xs.append(data[:seq_len])
        ys.append(data[1:seq_len + 1])
    return torch.stack(xs).to(DEVICE), torch.stack(ys).to(DEVICE)


CONFIGS = [
    {'name': 'brain_only',       'use_lcx': False},
    {'name': 'brain+whiteboard', 'use_lcx': True},
]


def run_config(cfg):
    name = cfg['name']
    use_lcx = cfg['use_lcx']

    print(f'\n{"="*60}')
    print(f'  CONFIG: {name}')
    print(f'  D={D} depth={DEPTH} seq={SEQ_LEN} block={BLOCK_SIZE}')
    print(f'  LCX={use_lcx} num_bits={NUM_BITS} steps={STEPS}')
    print(f'{"="*60}')

    gc.collect()
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    torch.manual_seed(42)
    random.seed(42)

    model = SwarmByteRingModel(
        embedding_dim=D,
        num_memory_positions=SEQ_LEN,
        num_beings=1,
        depth=DEPTH,
        num_bits=NUM_BITS,
        attention_radius=RADIUS,
        attention_temperature=8.0,
        think_ticks=THINK_TICKS,
        use_lcx=use_lcx,
        lcx_mode='hash',
        lcx_num_levels=1,
        lcx_level_slots=[2000],
        lcx_key_dim=D // 10,
        lcx_top_k=2,
        num_pointers=1,
    ).to(DEVICE)

    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  params={n_params:,}')

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler('cuda') if DEVICE.type == 'cuda' else None

    def lr_lambda(step):
        if step < WARMUP:
            return step / WARMUP
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    checkpoints = {}
    window_accs = []
    WINDOW = 50
    t_start = time.time()

    for step in range(STEPS):
        t0 = time.time()
        if step < 3:
            print(f'  starting step {step}...', end='', flush=True)

        x, y = make_echo_batch_bits(BATCH, SEQ_LEN, BLOCK_SIZE, NUM_BITS)

        try:
            opt.zero_grad()
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    out = model(x)
                    if isinstance(out, tuple):
                        out = out[0]
                    loss = F.binary_cross_entropy_with_logits(out, y)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                out = model(x)
                if isinstance(out, tuple):
                    out = out[0]
                loss = F.binary_cross_entropy_with_logits(out, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            scheduler.step()
        except torch.cuda.OutOfMemoryError:
            print(f'\n  OOM at step {step}!')
            return {'name': name, 'tail': 0.0, 'error': 'OOM'}

        elapsed = time.time() - t0
        if step < 3:
            print(f' {elapsed:.2f}s', flush=True)

        if elapsed > STEP_TIMEOUT:
            print(f'  TIMEOUT at step {step} ({elapsed:.0f}s)')
            return {'name': name, 'tail': 0.0, 'error': 'TIMEOUT'}

        with torch.no_grad():
            pred = (out > 0).float()
            acc = (pred == y).float().mean().item()

        if math.isnan(loss.item()):
            print(f'  NaN at step {step}')
            return {'name': name, 'tail': 0.0, 'error': 'NaN'}

        window_accs.append(acc)
        if len(window_accs) > WINDOW:
            window_accs.pop(0)
        smooth_acc = sum(window_accs) / len(window_accs)

        if step % 50 == 0 or step == STEPS - 1:
            checkpoints[step] = smooth_acc
            print(f'  step {step:4d} | loss {loss.item():.6f} | acc {acc:.4f} | '
                  f'smooth={smooth_acc:.4f} | {elapsed:.2f}s', flush=True)

        with open(LIVE_LOG, 'a') as lf:
            lf.write(f'step {step} | loss {loss.item():.6f} | '
                     f'acc={acc:.4f} smooth={smooth_acc:.4f} '
                     f'RD:{elapsed:.4f} traction={smooth_acc:.4f} '
                     f'shard=0/0 {name}\n')

    total_time = time.time() - t_start
    tail_accs = [a for s, a in checkpoints.items() if s >= 600]
    tail_avg = sum(tail_accs) / len(tail_accs) if tail_accs else 0.5

    print(f'\n  {name}: tail={tail_avg*100:.2f}%  time={total_time:.0f}s')
    return {'name': name, 'tail': tail_avg, 'time': total_time}


if __name__ == '__main__':
    print('='*60)
    print('PROBE: WHITEBOARD A/B — BRAIN vs BRAIN+LCX')
    print('='*60)
    print(f'  D={D}  depth={DEPTH}  num_bits={NUM_BITS}  {STEPS} steps')
    print(f'  batch={BATCH}  LR={LR}  seed=42  device={DEVICE}')
    if DEVICE.type == 'cuda':
        print(f'  GPU: {torch.cuda.get_device_name(0)}')

    with open(LIVE_LOG, 'w') as f:
        f.write(f'# probe_whiteboard_ab -- {time.strftime("%Y-%m-%d %H:%M:%S")}\n')

    results = []
    for cfg in CONFIGS:
        r = run_config(cfg)
        results.append(r)

    print(f'\n{"="*60}')
    print(f'SUMMARY: WHITEBOARD A/B')
    print(f'{"="*60}')
    for r in results:
        err = r.get('error', '')
        print(f'  {r["name"]:25s}  tail={r["tail"]*100:.2f}%  {err}')

    if len(results) == 2 and all('error' not in r for r in results):
        diff = results[1]['tail'] - results[0]['tail']
        print(f'\n  Whiteboard advantage: {diff*100:+.2f}%')
        if diff > 0.02:
            print(f'  VERDICT: WHITEBOARD_HELPS')
        elif diff > 0.005:
            print(f'  VERDICT: WHITEBOARD_WEAK')
        else:
            print(f'  VERDICT: WHITEBOARD_NO_EFFECT')

    print(f'{"="*60}')
    print(f'Done. Log: {LIVE_LOG}')
