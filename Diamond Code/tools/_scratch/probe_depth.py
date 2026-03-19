#!/usr/bin/env python3
"""
Probe: DEPTH Ablation (full-scale GPU)
=======================================
Test processing layer depth at production D=6180 on GPU with AMP.
depth={2, 4, 6, 8, 12}, all at D=6180, batch=10, tt=1, LCX hash.
200 steps, 1 seed (GPU runs are expensive — add seed 137 later if noisy).
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
BLOCK_SIZE    = 4
BATCH         = 10
LR            = 1e-4    # 1e-3 causes NaN at D=6180 with AMP — fp16 overflow
STEPS         = 300
WARMUP        = 50
SEEDS         = [42]
NUM_BITS      = 8
STEP_TIMEOUT  = 120
RADIUS        = 8
THINK_TICKS   = 1

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CONFIGS = [
    # (label, depth)
    ('depth_2',   2),
    ('depth_4',   4),
    ('depth_6',   6),    # current production (control)
    ('depth_8',   8),
    ('depth_12', 12),
]

DIAMOND_ROOT = r'S:\AI\work\VRAXION_DEV\Diamond Code'
sys.path.insert(0, DIAMOND_ROOT)
LOG_DIR      = os.path.join(DIAMOND_ROOT, 'logs', 'probe')
LIVE_LOG     = os.path.join(LOG_DIR, 'probe_depth_live.log')
os.makedirs(LOG_DIR, exist_ok=True)

with open(LIVE_LOG, 'w') as f:
    f.write(f'# probe_depth — {time.strftime("%Y-%m-%d %H:%M:%S")}\n')

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


def run_config(label, depth, seed):
    gc.collect()
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    torch.manual_seed(seed)
    random.seed(seed)

    try:
        model = make_model(depth).to(DEVICE)
    except torch.cuda.OutOfMemoryError:
        print(f'  OOM constructing {label}!')
        return 0.0, 'OOM', 0.0, 0, 0.0
    except Exception as e:
        print(f'  ERROR constructing {label}: {e}')
        return 0.0, 'ERROR', 0.0, 0, 0.0

    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler('cuda') if DEVICE.type == 'cuda' else None

    # Linear warmup scheduler (matches production --warmup_steps)
    def lr_lambda(step):
        if step < WARMUP:
            return step / WARMUP
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    tail_accs = []
    had_nan = False
    had_div = False
    t_start = time.time()

    print(f'\n  [{label} seed={seed}] params={n_params:,}  depth={depth}  tt={THINK_TICKS}',
          flush=True)
    if DEVICE.type == 'cuda':
        vram_model = torch.cuda.max_memory_allocated() / 1024**3
        print(f'    VRAM after model load: {vram_model:.2f} GB', flush=True)

    for step in range(STEPS):
        t0 = time.time()
        print(f'    starting step {step}...', end='', flush=True) if step < 3 else None
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
            print(f'\n    OOM at step {step}!')
            del model, opt
            gc.collect()
            torch.cuda.empty_cache()
            return 0.0, 'OOM', time.time() - t_start, n_params, 0.0

        elapsed = time.time() - t0
        if step < 3:
            print(f' {elapsed:.2f}s', flush=True)

        if elapsed > STEP_TIMEOUT:
            print(f'    TIMEOUT: step {step} took {elapsed:.0f}s, aborting')
            break

        with torch.no_grad():
            pred = (out > 0).float()
            acc = (pred == y).float().mean().item()

        if math.isnan(loss.item()):
            had_nan = True
            print(f'    NaN at step {step}, aborting')
            break
        if loss.item() > 10.0 and step > 150:
            had_div = True
            print(f'    Divergence at step {step} (loss={loss.item():.4f})')
            break

        if step >= STEPS - 75:
            tail_accs.append(acc)

        cur_lr = opt.param_groups[0]['lr']
        if step % 25 == 0 or step == STEPS - 1:
            print(f'    step {step:4d} | loss {loss.item():.6f} | acc {acc:.4f} | lr {cur_lr:.1e} | {elapsed:.2f}s',
                  flush=True)

        with open(LIVE_LOG, 'a') as lf:
            lf.write(f'[{label} s={seed}] step {step} | loss {loss.item():.6f} | '
                     f'acc={acc:.4f} RD:{elapsed:.4f}\n')

    total_time = time.time() - t_start
    vram_peak = 0.0
    if DEVICE.type == 'cuda':
        vram_peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f'    VRAM peak: {vram_peak:.2f} GB')

    # Cleanup
    del model, opt
    if scaler is not None:
        del scaler
    gc.collect()
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()

    if had_nan:
        return 0.0, 'NAN', total_time, n_params, vram_peak
    if had_div:
        return 0.0, 'DIVERGED', total_time, n_params, vram_peak
    if not tail_accs:
        return 0.0, 'NO_TAIL', total_time, n_params, vram_peak

    tail_median = sorted(tail_accs)[len(tail_accs) // 2]
    return tail_median, 'OK', total_time, n_params, vram_peak


if __name__ == '__main__':
    print('=' * 70)
    print('PROBE: DEPTH Ablation (full-scale GPU)')
    print('=' * 70)
    print(f'  D={D}  seq_len={SEQ_LEN}  block={BLOCK_SIZE}  tt={THINK_TICKS}')
    print(f'  batch={BATCH}  lr={LR}  steps={STEPS}  seeds={SEEDS}  radius={RADIUS}')
    print(f'  LCX: hash, 2000 slots, key_dim={D//10}, top_k=2')
    print(f'  configs: {[(l, d) for l, d in CONFIGS]}')
    print(f'  device: {DEVICE}')
    if DEVICE.type == 'cuda':
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print('=' * 70)

    results = {}
    for label, depth in CONFIGS:
        seed_tails = []
        seed_times = []
        params = 0
        vram = 0.0
        for seed in SEEDS:
            tail, status, elapsed, params, vram = run_config(label, depth, seed)
            seed_tails.append(tail)
            seed_times.append(elapsed)
            print(f'    -> tail_median={tail:.4f}  status={status}  time={elapsed:.1f}s  vram={vram:.2f}G')

        mean_tail = sum(seed_tails) / len(seed_tails)
        seed_gap = max(seed_tails) - min(seed_tails) if len(seed_tails) > 1 else 0.0
        avg_time = sum(seed_times) / len(seed_times)
        s_per_step = avg_time / STEPS if avg_time > 0 else 0
        results[label] = {
            'mean_tail': mean_tail, 'seed_gap': seed_gap,
            'tails': seed_tails, 'times': seed_times,
            'params': params, 'depth': depth,
            'vram': vram, 's_per_step': s_per_step,
        }

    # Results
    print('\n' + '=' * 70)
    print('RESULTS: DEPTH Ablation')
    print('=' * 70)
    print(f'  {"Config":<10} {"Depth":>5} {"Params":>12} {"VRAM":>7} '
          f'{"s/step":>7} {"Mean Tail":>10} {"Seeds":>10}')
    print(f'  {"-"*10} {"-"*5} {"-"*12} {"-"*7} {"-"*7} {"-"*10} {"-"*10}')

    for label, r in results.items():
        seeds_str = ', '.join(f'{t:.4f}' for t in r['tails'])
        print(f'  {label:<10} {r["depth"]:>5} {r["params"]:>12,} {r["vram"]:>6.1f}G '
              f'{r["s_per_step"]:>6.2f}s {r["mean_tail"]:>10.4f} {seeds_str:>10}')

    ctrl = results.get('depth_6', {}).get('mean_tail', 0)
    print(f'\n  Deltas vs depth_6 ({ctrl:.4f}):')
    for label, r in results.items():
        if label != 'depth_6':
            delta = r['mean_tail'] - ctrl
            print(f'    {label:<10}: {delta:+.4f}  ({r["s_per_step"]:.2f}s/step, {r["vram"]:.1f}G)')

    # Efficiency metric: accuracy per second
    print(f'\n  Efficiency (acc per GPU-second):')
    for label, r in results.items():
        if r['s_per_step'] > 0 and r['mean_tail'] > 0:
            eff = r['mean_tail'] / r['s_per_step']
            print(f'    {label:<10}: {eff:.4f} acc/s  (tail={r["mean_tail"]:.4f}, {r["s_per_step"]:.2f}s/step)')

    # Verdict
    print('\n' + '=' * 70)
    valid = {k: v for k, v in results.items() if v['mean_tail'] > 0}
    if not valid:
        print('  VERDICT: ALL FAILED')
    else:
        best_label = max(valid, key=lambda k: valid[k]['mean_tail'])
        worst_label = min(valid, key=lambda k: valid[k]['mean_tail'])
        best = valid[best_label]
        worst = valid[worst_label]
        spread = best['mean_tail'] - worst['mean_tail']

        if spread < 0.015:
            print(f'  VERDICT: DEPTH_CLEARED')
            print(f'  Spread: {spread:.4f} (<1.5%). Depth doesn\'t matter much.')
            print(f'  Keep depth=6 (balanced).')
        elif best['mean_tail'] - ctrl > 0.02:
            print(f'  VERDICT: DEPTH_OPTIMIZED')
            print(f'  Best: {best_label} ({best["mean_tail"]:.4f}) vs control ({ctrl:.4f})')
            print(f'  Delta: {best["mean_tail"] - ctrl:+.4f}')
            print(f'  Cost: {best["vram"]:.1f}G VRAM, {best["s_per_step"]:.2f}s/step')
        else:
            print(f'  VERDICT: DEPTH_MIXED')
            print(f'  Best: {best_label} ({best["mean_tail"]:.4f})')
            print(f'  Worst: {worst_label} ({worst["mean_tail"]:.4f})')
            print(f'  Spread: {spread:.4f}')

    print('=' * 70)

    with open(LIVE_LOG, 'a') as lf:
        lf.write(f'\n# RESULTS:\n')
        for label, r in results.items():
            lf.write(f'# {label}: depth={r["depth"]} mean_tail={r["mean_tail"]:.4f} '
                     f'vram={r["vram"]:.1f}G s/step={r["s_per_step"]:.2f}\n')
