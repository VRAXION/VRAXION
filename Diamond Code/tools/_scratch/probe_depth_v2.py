#!/usr/bin/env python3
"""
Probe: DEPTH capacity stress test (full-scale GPU)
===================================================
Single config: depth=2 on a HARDER task for 1000 steps.
block_size=16 (was 4) — 16-byte repeating pattern, requires real memory.

Question: Does depth=2 plateau early, or keep climbing?
- If still climbing at step 1000 -> depth=2 has capacity, lock it.
- If flatlines by step 400-500 -> capacity ceiling, need to test deeper.

Logs accuracy at every step for full learning curve analysis.
"""

import gc
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F

D             = 6180
SEQ_LEN       = 32
BLOCK_SIZE    = 16     # ← 4x harder than v1 (was 4)
BATCH         = 10
LR            = 1e-4
STEPS         = 1000
WARMUP        = 50
SEEDS         = [42]
NUM_BITS      = 8
STEP_TIMEOUT  = 120
RADIUS        = 8
THINK_TICKS   = 1
DEPTH         = 1      # ← testing floor (was 6 in v2b, 2 in v2a)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DIAMOND_ROOT = r'S:\AI\work\VRAXION_DEV\Diamond Code'
sys.path.insert(0, DIAMOND_ROOT)
LOG_DIR      = os.path.join(DIAMOND_ROOT, 'logs', 'probe')
LIVE_LOG     = os.path.join(LOG_DIR, 'probe_depth_v2c_live.log')
os.makedirs(LOG_DIR, exist_ok=True)

with open(LIVE_LOG, 'w') as f:
    f.write(f'# probe_depth_v2 — {time.strftime("%Y-%m-%d %H:%M:%S")}\n')

from swarm_model import SwarmByteRingModel


def byte_to_bits(byte_seq, num_bits=8):
    t = torch.tensor(byte_seq, dtype=torch.uint8)
    return ((t.unsqueeze(-1) >> torch.arange(num_bits)) & 1).float()


def make_echo_batch(batch_size, seq_len, block_size, num_bits=8):
    xs, ys = [], []
    for _ in range(batch_size):
        block = [random.randint(0, 255) for _ in range(block_size)]
        repeats = (seq_len + 2) // block_size + 1
        data = (block * repeats)[:seq_len + 1]
        xs.append(byte_to_bits(data[:seq_len], num_bits))
        ys.append(byte_to_bits(data[1:seq_len + 1], num_bits))
    return torch.stack(xs).to(DEVICE), torch.stack(ys).to(DEVICE)


def make_model(depth):
    return SwarmByteRingModel(
        embedding_dim=D,
        num_memory_positions=SEQ_LEN,
        num_beings=1,
        depth=depth,
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
    )


if __name__ == '__main__':
    print('=' * 70)
    print('PROBE: DEPTH=2 Capacity Stress Test (block_size=16)')
    print('=' * 70)
    print(f'  D={D}  seq_len={SEQ_LEN}  block={BLOCK_SIZE}  tt={THINK_TICKS}')
    print(f'  batch={BATCH}  lr={LR}  steps={STEPS}  seeds={SEEDS}  radius={RADIUS}')
    print(f'  depth={DEPTH}  (testing if shallow model plateaus on harder task)')
    print(f'  LCX: hash, 2000 slots, key_dim={D//10}, top_k=2')
    print(f'  device: {DEVICE}')
    if DEVICE.type == 'cuda':
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print('=' * 70)

    gc.collect()
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    torch.manual_seed(42)
    random.seed(42)

    model = make_model(DEPTH).to(DEVICE)
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler('cuda') if DEVICE.type == 'cuda' else None

    def lr_lambda(step):
        if step < WARMUP:
            return step / WARMUP
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    print(f'\n  params={n_params:,}  depth={DEPTH}', flush=True)
    if DEVICE.type == 'cuda':
        vram_model = torch.cuda.max_memory_allocated() / 1024**3
        print(f'  VRAM after model load: {vram_model:.2f} GB', flush=True)

    # Track accuracy at checkpoints for learning curve analysis
    checkpoints = {}   # step -> acc
    window_accs = []   # rolling window for smoothed curve
    WINDOW = 50

    t_start = time.time()
    for step in range(STEPS):
        t0 = time.time()
        if step < 3:
            print(f'  starting step {step}...', end='', flush=True)

        x, y = make_echo_batch(BATCH, SEQ_LEN, BLOCK_SIZE, NUM_BITS)

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
            break

        elapsed = time.time() - t0
        if step < 3:
            print(f' {elapsed:.2f}s', flush=True)

        if elapsed > STEP_TIMEOUT:
            print(f'  TIMEOUT: step {step} took {elapsed:.0f}s, aborting')
            break

        with torch.no_grad():
            pred = (out > 0).float()
            acc = (pred == y).float().mean().item()

        if math.isnan(loss.item()):
            print(f'  NaN at step {step}, aborting')
            break

        window_accs.append(acc)
        if len(window_accs) > WINDOW:
            window_accs.pop(0)
        smooth_acc = sum(window_accs) / len(window_accs)

        # Log checkpoints at 100-step intervals + key early points
        if step in (25, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 999) or step % 100 == 0:
            checkpoints[step] = smooth_acc

        cur_lr = opt.param_groups[0]['lr']
        if step % 50 == 0 or step == STEPS - 1:
            print(f'  step {step:4d} | loss {loss.item():.6f} | acc {acc:.4f} | '
                  f'smooth={smooth_acc:.4f} | lr {cur_lr:.1e} | {elapsed:.2f}s',
                  flush=True)

        with open(LIVE_LOG, 'a') as lf:
            lf.write(f'step {step} | loss {loss.item():.6f} | '
                     f'acc={acc:.4f} smooth={smooth_acc:.4f} RD:{elapsed:.4f}\n')

    total_time = time.time() - t_start
    vram_peak = 0.0
    if DEVICE.type == 'cuda':
        vram_peak = torch.cuda.max_memory_allocated() / 1024**3

    # Learning curve analysis
    print('\n' + '=' * 70)
    print('LEARNING CURVE: depth=2, block_size=16')
    print('=' * 70)
    print(f'  {"Step":>6} {"Smooth Acc":>10} {"Delta/100":>10}')
    print(f'  {"-"*6} {"-"*10} {"-"*10}')

    sorted_ckpts = sorted(checkpoints.items())
    prev_acc = 0.5  # baseline
    for step, sacc in sorted_ckpts:
        delta = sacc - prev_acc
        bar = '>' * int(max(0, (sacc - 0.5)) * 40)
        print(f'  {step:>6} {sacc:>10.4f} {delta:>+10.4f}  {bar}')
        prev_acc = sacc

    # Plateau detection: compare last 200 steps vs steps 300-500
    tail_200 = [a for s, a in sorted_ckpts if s >= 800]
    mid_200  = [a for s, a in sorted_ckpts if 300 <= s <= 500]

    if tail_200 and mid_200:
        tail_avg = sum(tail_200) / len(tail_200)
        mid_avg  = sum(mid_200) / len(mid_200)
        improvement = tail_avg - mid_avg

        print(f'\n  Mid (300-500) avg: {mid_avg:.4f}')
        print(f'  Tail (800-999) avg: {tail_avg:.4f}')
        print(f'  Late improvement: {improvement:+.4f}')

        print(f'\n  VRAM peak: {vram_peak:.2f} GB')
        print(f'  Total time: {total_time:.0f}s ({total_time/60:.1f} min)')
        print(f'  Avg s/step: {total_time/STEPS:.2f}s')

        print('\n' + '=' * 70)
        if improvement > 0.01:
            print(f'  VERDICT: STILL_CLIMBING (+{improvement:.4f} in late phase)')
            print(f'  depth=2 has NOT plateaued. Capacity is sufficient.')
            print(f'  -> Lock depth=2, proceed to D ablation.')
        elif improvement > -0.005:
            print(f'  VERDICT: PLATEAU (improvement {improvement:+.4f} ~ flat)')
            print(f'  depth=2 leveled off but may still be optimal.')
            print(f'  -> Run depth=6 comparison on same task to confirm.')
        else:
            print(f'  VERDICT: DECLINING ({improvement:+.4f} in late phase)')
            print(f'  depth=2 may be overfitting or capacity-limited.')
            print(f'  -> Run depth=4 and depth=6 on same task.')
    else:
        print(f'\n  Insufficient checkpoints for plateau analysis.')

    print('=' * 70)

    with open(LIVE_LOG, 'a') as lf:
        lf.write(f'\n# RESULTS: depth={DEPTH} block={BLOCK_SIZE}\n')
        for step, sacc in sorted_ckpts:
            lf.write(f'# checkpoint step={step} smooth_acc={sacc:.4f}\n')
