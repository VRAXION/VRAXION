#!/usr/bin/env python3
"""
Probe: Per-Neuron Leaky Integrator Memory (EMA, replaces LCX writes)
====================================================================
Each neuron gets one pixel. EMA (alpha=0.05) tracks recent baseline.
All LCX slots filled with the same EMA vector.
Read path uses normal LCX routing. No bottleneck.

v1 (cumulative mean) = DEAD at 50.0% — vector froze after ~100 steps.
v2 (EMA alpha=0.05) = 20-step sliding window, tracks recent context.
"""

import gc
import math
import os
import random
import sys
import time
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

D              = 2048
DEPTH          = 2
SEQ_LEN        = 32
BLOCK_SIZE     = 16
BATCH          = 10
LR             = 1e-4
WARMUP         = 50
NUM_BITS       = 200
STEPS          = 800
STEP_TIMEOUT   = 120
RADIUS         = 8
THINK_TICKS    = 1

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DIAMOND_ROOT = r'S:\AI\work\VRAXION_DEV\Diamond Code'
sys.path.insert(0, DIAMOND_ROOT)
LOG_DIR      = os.path.join(DIAMOND_ROOT, 'logs', 'probe')
LIVE_LOG     = os.path.join(LOG_DIR, 'probe_neuron_memory_gpu_live.log')
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


if __name__ == '__main__':
    print('=' * 70)
    print('PROBE: PER-NEURON LEAKY INTEGRATOR (EMA) MEMORY')
    print('=' * 70)
    print(f'  D={D}  depth={DEPTH}  num_bits={NUM_BITS}  {STEPS} steps')
    print(f'  batch={BATCH}  LR={LR}  seed=42  device={DEVICE}')
    if DEVICE.type == 'cuda':
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print()
    print(f'  Design: LCX routing active, ALL slot values = EMA vector')
    print(f'  Write: EMA update (alpha=0.05, ~20-step window)')
    print(f'  Read: normal routing (returns same vector regardless)')
    print(f'  Bottleneck: OFF')
    print(f'  Memory cost: {D} floats = {D * 4 / 1024:.1f} KB')
    print('=' * 70)

    with open(LIVE_LOG, 'w') as f:
        f.write(f'# probe_neuron_memory_gpu -- {time.strftime("%Y-%m-%d %H:%M:%S")}\n')

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
        use_lcx=True,
        lcx_mode='hash',
        lcx_num_levels=1,
        lcx_level_slots=[2000],
        lcx_key_dim=D // 10,
        lcx_top_k=2,
        num_pointers=1,
    ).to(DEVICE)

    model.lcx_bn_layers = None  # No bottleneck

    # Leaky integrator (EMA) storage — 20-step sliding window
    EMA_ALPHA = 0.05  # new input weight; (1-alpha)=0.95 = decay
    _mem = {
        'ema': torch.zeros(D, device=DEVICE),
        'initialized': False,
    }

    def _patched_write(self, state, write_content, level=0):
        with torch.no_grad():
            h = write_content.detach().mean(dim=0)  # batch mean
            if not _mem['initialized']:
                _mem['ema'] = h.clone()
                _mem['initialized'] = True
            else:
                _mem['ema'] = (1.0 - EMA_ALPHA) * _mem['ema'] + EMA_ALPHA * h
            values = getattr(self, f'lcx_values_{level}')
            values[:] = _mem['ema'].unsqueeze(0)

    model._lcx_flat_write = types.MethodType(_patched_write, model)

    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler('cuda') if DEVICE.type == 'cuda' else None

    def lr_lambda(step):
        if step < WARMUP:
            return step / WARMUP
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    print(f'\n  params={n_params:,}', flush=True)

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
            sys.exit(1)

        elapsed = time.time() - t0
        if step < 3:
            print(f' {elapsed:.2f}s', flush=True)

        if elapsed > STEP_TIMEOUT:
            print(f'  TIMEOUT at step {step} ({elapsed:.0f}s)')
            sys.exit(1)

        with torch.no_grad():
            pred = (out > 0).float()
            acc = (pred == y).float().mean().item()

        if math.isnan(loss.item()):
            print(f'  NaN at step {step}')
            sys.exit(1)

        window_accs.append(acc)
        if len(window_accs) > WINDOW:
            window_accs.pop(0)
        smooth_acc = sum(window_accs) / len(window_accs)

        if step % 50 == 0 or step == STEPS - 1:
            checkpoints[step] = smooth_acc
            mn = _mem['ema'].norm().item()
            print(f'  step {step:4d} | loss {loss.item():.6f} | acc {acc:.4f} | '
                  f'smooth={smooth_acc:.4f} | mem_norm={mn:.1f} | '
                  f'{elapsed:.2f}s', flush=True)

        with open(LIVE_LOG, 'a') as lf:
            lf.write(f'step {step} | loss {loss.item():.6f} | '
                     f'acc={acc:.4f} smooth={smooth_acc:.4f} '
                     f'RD:{elapsed:.4f} traction={smooth_acc:.4f} '
                     f'shard=0/0 neuron_ema\n')

    total_time = time.time() - t_start
    vram_peak = 0.0
    if DEVICE.type == 'cuda':
        vram_peak = torch.cuda.max_memory_allocated() / 1024**3

    tail_accs = [a for s, a in checkpoints.items() if s >= 600]
    tail_avg = sum(tail_accs) / len(tail_accs) if tail_accs else 0.5

    print(f'\n{"="*70}')
    print(f'SUMMARY: PER-NEURON LEAKY INTEGRATOR (EMA a=0.05)')
    print(f'{"="*70}')
    print(f'  Tail (600-799): {tail_avg:.4f}')
    print(f'  VRAM peak: {vram_peak:.2f} GB')
    print(f'  Time: {total_time:.0f}s ({total_time/60:.1f} min)')
    print()
    print(f'  References:')
    print(f'    noBN         : 50.13%  (no transform)')
    print(f'    per_dim_scale: 50.04%  (volume knobs)')
    print(f'    BN=409 2L    : 50.64%  (best squeeze)')
    print(f'    neuron_ema   : {tail_avg*100:.2f}%  (THIS)')
    print(f'\n{"="*70}')
    if tail_avg > 0.5050:
        print(f'  VERDICT: NEURON_MEMORY_WORKS')
    elif tail_avg > 0.5020:
        print(f'  VERDICT: NEURON_MEMORY_WEAK')
    else:
        print(f'  VERDICT: NEURON_MEMORY_DEAD')
    print(f'{"="*70}')
    print(f'\nDone. Log: {LIVE_LOG}')
