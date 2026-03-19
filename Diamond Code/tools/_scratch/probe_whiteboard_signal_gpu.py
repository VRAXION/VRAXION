#!/usr/bin/env python3
"""
Probe: Whiteboard Signal — Find clear LCX gradient
====================================================
Approach A: D=128, replicate mini-model that showed +13% LCX advantage
Approach B: D=256, longer distance (block=32, seq=64) to force whiteboard need

Each approach: brain-only vs brain+whiteboard, same seed.
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

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DIAMOND_ROOT = r'S:\AI\work\VRAXION_DEV\Diamond Code'
sys.path.insert(0, DIAMOND_ROOT)
LOG_DIR  = os.path.join(DIAMOND_ROOT, 'logs', 'probe')
LIVE_LOG = os.path.join(LOG_DIR, 'probe_whiteboard_signal_live.log')
os.makedirs(LOG_DIR, exist_ok=True)

from swarm_model import SwarmByteRingModel


def make_echo_batch_bits(batch_size, seq_len, block_size, num_bits, device):
    xs, ys = [], []
    for _ in range(batch_size):
        block = torch.randint(0, 2, (block_size, num_bits)).float()
        repeats = (seq_len + 2) // block_size + 1
        data = block.repeat(repeats, 1)[:seq_len + 1]
        xs.append(data[:seq_len])
        ys.append(data[1:seq_len + 1])
    return torch.stack(xs).to(device), torch.stack(ys).to(device)


CONFIGS = [
    # Approach A: replicate mini-model winner
    {
        'name': 'A1_brain_only',
        'D': 128, 'depth': 4, 'batch': 32, 'num_bits': 8,
        'seq_len': 32, 'block_size': 16, 'radius': 8,
        'use_lcx': False, 'lcx_slots': 500, 'top_k': 2,
        'steps': 400, 'lr': 1e-4, 'warmup': 30,
    },
    {
        'name': 'A2_brain+whiteboard',
        'D': 128, 'depth': 4, 'batch': 32, 'num_bits': 8,
        'seq_len': 32, 'block_size': 16, 'radius': 8,
        'use_lcx': True, 'lcx_slots': 500, 'top_k': 2,
        'steps': 400, 'lr': 1e-4, 'warmup': 30,
    },
    # Approach B: longer distance forces whiteboard
    {
        'name': 'B1_brain_only_long',
        'D': 256, 'depth': 2, 'batch': 10, 'num_bits': 25,
        'seq_len': 64, 'block_size': 32, 'radius': 8,
        'use_lcx': False, 'lcx_slots': 2000, 'top_k': 2,
        'steps': 800, 'lr': 1e-4, 'warmup': 50,
    },
    {
        'name': 'B2_brain+whiteboard_long',
        'D': 256, 'depth': 2, 'batch': 10, 'num_bits': 25,
        'seq_len': 64, 'block_size': 32, 'radius': 8,
        'use_lcx': True, 'lcx_slots': 2000, 'top_k': 2,
        'steps': 800, 'lr': 1e-4, 'warmup': 50,
    },
]


def run_config(cfg):
    name = cfg['name']
    D = cfg['D']
    steps = cfg['steps']

    print(f'\n{"="*60}')
    print(f'  CONFIG: {name}')
    print(f'  D={D} depth={cfg["depth"]} seq={cfg["seq_len"]} block={cfg["block_size"]}')
    print(f'  LCX={cfg["use_lcx"]} num_bits={cfg["num_bits"]} steps={steps}')
    print(f'  batch={cfg["batch"]} lr={cfg["lr"]} radius={cfg["radius"]}')
    print(f'{"="*60}')

    gc.collect()
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    torch.manual_seed(42)
    random.seed(42)

    key_dim = max(D // 10, 8)

    model = SwarmByteRingModel(
        embedding_dim=D,
        num_memory_positions=cfg['seq_len'],
        num_beings=1,
        depth=cfg['depth'],
        num_bits=cfg['num_bits'],
        attention_radius=cfg['radius'],
        attention_temperature=8.0,
        think_ticks=1,
        use_lcx=cfg['use_lcx'],
        lcx_mode='hash',
        lcx_num_levels=1,
        lcx_level_slots=[cfg['lcx_slots']],
        lcx_key_dim=key_dim,
        lcx_top_k=cfg['top_k'],
        num_pointers=1,
    ).to(DEVICE)

    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  params={n_params:,}')

    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    scaler = torch.amp.GradScaler('cuda') if DEVICE.type == 'cuda' else None

    warmup = cfg['warmup']
    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    checkpoints = {}
    window_accs = []
    WINDOW = 50
    STEP_TIMEOUT = 60
    t_start = time.time()

    for step in range(steps):
        t0 = time.time()
        if step < 3:
            print(f'  starting step {step}...', end='', flush=True)

        x, y = make_echo_batch_bits(
            cfg['batch'], cfg['seq_len'], cfg['block_size'],
            cfg['num_bits'], DEVICE)

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

        if step % 50 == 0 or step == steps - 1:
            checkpoints[step] = smooth_acc
            print(f'  step {step:4d} | loss {loss.item():.6f} | acc {acc:.4f} | '
                  f'smooth={smooth_acc:.4f} | {elapsed:.2f}s', flush=True)

        with open(LIVE_LOG, 'a') as lf:
            lf.write(f'step {step} | loss {loss.item():.6f} | '
                     f'acc={acc:.4f} smooth={smooth_acc:.4f} '
                     f'RD:{elapsed:.4f} traction={smooth_acc:.4f} '
                     f'shard=0/0 {name}\n')

    total_time = time.time() - t_start
    tail_start = int(steps * 0.75)
    tail_accs = [a for s, a in checkpoints.items() if s >= tail_start]
    tail_avg = sum(tail_accs) / len(tail_accs) if tail_accs else 0.5

    print(f'\n  {name}: tail={tail_avg*100:.2f}%  time={total_time:.0f}s')
    return {'name': name, 'tail': tail_avg, 'time': total_time}


if __name__ == '__main__':
    print('=' * 60)
    print('PROBE: WHITEBOARD SIGNAL — FIND CLEAR LCX GRADIENT')
    print('=' * 60)
    print(f'  device={DEVICE}')
    if DEVICE.type == 'cuda':
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  Approach A: D=128 replicate mini-model (+13% expected)')
    print(f'  Approach B: D=256 long distance (force whiteboard)')

    with open(LIVE_LOG, 'w') as f:
        f.write(f'# probe_whiteboard_signal -- {time.strftime("%Y-%m-%d %H:%M:%S")}\n')

    results = []
    for cfg in CONFIGS:
        r = run_config(cfg)
        results.append(r)

    print(f'\n{"="*60}')
    print(f'SUMMARY: WHITEBOARD SIGNAL')
    print(f'{"="*60}')
    for r in results:
        err = r.get('error', '')
        print(f'  {r["name"]:30s}  tail={r["tail"]*100:.2f}%  {err}')

    # Approach A verdict
    a1 = next((r for r in results if r['name'] == 'A1_brain_only'), None)
    a2 = next((r for r in results if r['name'] == 'A2_brain+whiteboard'), None)
    if a1 and a2 and 'error' not in a1 and 'error' not in a2:
        diff_a = a2['tail'] - a1['tail']
        print(f'\n  Approach A (D=128): whiteboard advantage = {diff_a*100:+.2f}%')
        if diff_a > 0.10:
            print(f'  VERDICT A: LCX_CONFIRMED (+{diff_a*100:.0f}%)')
        elif diff_a > 0.05:
            print(f'  VERDICT A: LCX_MODERATE (+{diff_a*100:.0f}%)')
        elif diff_a > 0.02:
            print(f'  VERDICT A: LCX_WEAK (+{diff_a*100:.1f}%)')
        else:
            print(f'  VERDICT A: LCX_NO_EFFECT')

    # Approach B verdict
    b1 = next((r for r in results if r['name'] == 'B1_brain_only_long'), None)
    b2 = next((r for r in results if r['name'] == 'B2_brain+whiteboard_long'), None)
    if b1 and b2 and 'error' not in b1 and 'error' not in b2:
        diff_b = b2['tail'] - b1['tail']
        print(f'\n  Approach B (D=256 long): whiteboard advantage = {diff_b*100:+.2f}%')
        if diff_b > 0.05:
            print(f'  VERDICT B: DISTANCE_MATTERS (+{diff_b*100:.0f}%)')
        elif diff_b > 0.02:
            print(f'  VERDICT B: DISTANCE_WEAK (+{diff_b*100:.1f}%)')
        else:
            print(f'  VERDICT B: DISTANCE_NO_EFFECT')

    print(f'\n{"="*60}')
    print(f'Done. Log: {LIVE_LOG}')
