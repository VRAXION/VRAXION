#!/usr/bin/env python3
"""
Probe: Attention Radius + Temperature Ablation
===============================================
Tests Gaussian ring attention settings on echo256 BLOCK=4.

The ring attention formula: logits = -(delta^2) / temperature
Radius hard-clips the window: only offsets [-R, +R] are considered.

CRITICAL: radius < 4 makes offset -4 invisible (echo256 impossible).

Configs test the radius × temperature space:
  r6_t8  — production (control), 2.7% weight at offset -4
  r6_t4  — sharper, same window
  r6_t2  — very sharp, center-focused
  r4_t4  — smaller window, sharp
  r4_t8  — smaller window, broad
  r8_t8  — wider window, same sharpness

All use SwarmByteRingModel D=128, LCX OFF, LR=1e-3.
"""

import math
import os
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── CONFIG ──────────────────────────────────────────────────
D             = 128
DEPTH         = 4
SEQ_LEN       = 32
BLOCK_SIZE    = 4
BATCH         = 16
LR            = 1e-3
STEPS         = 500
SEEDS         = [42, 137]
NUM_BITS      = 8
STEP_TIMEOUT  = 60
DEVICE        = torch.device('cpu')

CONFIGS = [
    # (label, radius, temperature)
    ('r6_t8', 6, 8.0),   # production (control)
    ('r6_t4', 6, 4.0),   # sharper, same window
    ('r6_t2', 6, 2.0),   # very sharp
    ('r4_t4', 4, 4.0),   # smaller window, sharp
    ('r4_t8', 4, 8.0),   # smaller window, broad
    ('r8_t8', 8, 8.0),   # wider window
]

# ─── PATHS ───────────────────────────────────────────────────
DIAMOND_ROOT = r'S:\AI\work\VRAXION_DEV\Diamond Code'
sys.path.insert(0, DIAMOND_ROOT)
LOG_DIR      = os.path.join(DIAMOND_ROOT, 'logs', 'probe')
LIVE_LOG     = os.path.join(LOG_DIR, 'probe_attention_live.log')
os.makedirs(LOG_DIR, exist_ok=True)

with open(LIVE_LOG, 'w') as f:
    f.write(f'# probe_attention — {time.strftime("%Y-%m-%d %H:%M:%S")}\n')

from swarm_model import SwarmByteRingModel


# ─── DATA GENERATION ────────────────────────────────────────
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


# ─── MODEL FACTORY ──────────────────────────────────────────
def make_model(radius, temperature):
    return SwarmByteRingModel(
        embedding_dim=D,
        num_memory_positions=SEQ_LEN,
        num_beings=1,
        depth=DEPTH,
        num_bits=NUM_BITS,
        attention_radius=radius,
        attention_temperature=temperature,
        think_ticks=0,
        use_lcx=False,
        num_pointers=1,
    )


# ─── TRAINING LOOP ──────────────────────────────────────────
def run_config(label, radius, temperature, seed):
    torch.manual_seed(seed)
    random.seed(seed)

    model = make_model(radius, temperature).to(DEVICE)
    model.train()

    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    tail_accs = []
    had_nan = False
    had_div = False
    t_start = time.time()

    print(f'\n  [{label} seed={seed}] params={n_params:,}  r={radius} t={temperature}',
          flush=True)

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
            print(f'  TIMEOUT: step {step} took {elapsed:.0f}s, aborting')
            sys.exit(1)

        with torch.no_grad():
            pred = (out > 0).float()
            acc = (pred == y).float().mean().item()

        if math.isnan(loss.item()):
            had_nan = True
            print(f'  NaN at step {step}, aborting config')
            break
        if loss.item() > 3.0 and step > 200:
            had_div = True
            print(f'  Divergence at step {step} (loss={loss.item():.4f}), aborting config')
            break

        if step >= STEPS - 100:
            tail_accs.append(acc)

        if step % 50 == 0 or step == STEPS - 1:
            print(f'    step {step:4d} | loss {loss.item():.6f} | acc {acc:.4f} | {elapsed:.2f}s',
                  flush=True)

        with open(LIVE_LOG, 'a') as lf:
            lf.write(f'[{label} s={seed}] step {step} | loss {loss.item():.6f} | '
                     f'acc={acc:.4f} RD:{elapsed:.4f}\n')

    total_time = time.time() - t_start

    if had_nan:
        return 0.0, 'NAN', total_time, n_params
    if had_div:
        return 0.0, 'DIVERGED', total_time, n_params
    if not tail_accs:
        return 0.0, 'NO_TAIL', total_time, n_params

    tail_median = sorted(tail_accs)[len(tail_accs) // 2]
    return tail_median, 'OK', total_time, n_params


# ─── MAIN ────────────────────────────────────────────────────
if __name__ == '__main__':
    print('=' * 70)
    print('PROBE: Attention Radius + Temperature Ablation (echo256 BLOCK=4)')
    print('=' * 70)
    print(f'  D={D}  depth={DEPTH}  seq_len={SEQ_LEN}  block={BLOCK_SIZE}')
    print(f'  batch={BATCH}  lr={LR}  steps={STEPS}  seeds={SEEDS}')
    print(f'  configs: {[(l, r, t) for l, r, t in CONFIGS]}')
    print('=' * 70)

    results = {}

    for label, radius, temperature in CONFIGS:
        seed_tails = []
        seed_times = []
        params = 0

        for seed in SEEDS:
            tail, status, elapsed, params = run_config(label, radius, temperature, seed)
            seed_tails.append(tail)
            seed_times.append(elapsed)
            print(f'    -> tail_median={tail:.4f}  status={status}  time={elapsed:.1f}s')

        mean_tail = sum(seed_tails) / len(seed_tails)
        seed_gap = max(seed_tails) - min(seed_tails)
        results[label] = {
            'mean_tail': mean_tail,
            'seed_gap': seed_gap,
            'tails': seed_tails,
            'times': seed_times,
            'params': params,
            'radius': radius,
            'temperature': temperature,
        }

    # ─── RESULTS TABLE ───────────────────────────────────────
    print('\n' + '=' * 70)
    print('RESULTS')
    print('=' * 70)
    print(f'  {"Config":<10} {"R":>3} {"Temp":>6} {"Params":>8} {"Mean Tail":>10} '
          f'{"Gap":>8} {"Seeds":>20}')
    print(f'  {"-"*10} {"-"*3} {"-"*6} {"-"*8} {"-"*10} {"-"*8} {"-"*20}')

    for label, r in results.items():
        seeds_str = ', '.join(f'{t:.4f}' for t in r['tails'])
        print(f'  {label:<10} {r["radius"]:>3} {r["temperature"]:>6.1f} '
              f'{r["params"]:>8,} {r["mean_tail"]:>10.4f} '
              f'{r["seed_gap"]:>8.4f} {seeds_str:>20}')

    prod = results.get('r6_t8', {}).get('mean_tail', 0)

    print(f'\n  Deltas vs production (r6_t8 = {prod:.4f}):')
    for label, r in results.items():
        if label != 'r6_t8':
            delta = r['mean_tail'] - prod
            print(f'    {label:<10}: {delta:+.4f}')

    # ─── VERDICT ─────────────────────────────────────────────
    print('\n' + '=' * 70)

    best_label = max(results, key=lambda k: results[k]['mean_tail'])
    best = results[best_label]
    worst_label = min(results, key=lambda k: results[k]['mean_tail'])
    worst = results[worst_label]
    spread = best['mean_tail'] - worst['mean_tail']

    if spread < 0.03:
        verdict = 'ATTN_CLEARED'
        print(f'  VERDICT: Attention settings CLEARED')
        print(f'  Spread: {spread:.4f} (<3%). All configs perform similarly.')
        print(f'  Keep production r=6/t=8.0.')
    elif best['mean_tail'] - prod > 0.05:
        verdict = 'ATTN_OPTIMIZED'
        print(f'  VERDICT: Better setting FOUND')
        print(f'  Best: {best_label} ({best["mean_tail"]:.4f}) vs production ({prod:.4f})')
        print(f'  Delta: {best["mean_tail"] - prod:+.4f}. Apply to run_goldilocks.bat.')
    elif best['mean_tail'] - prod > 0.03:
        verdict = 'ATTN_MARGINAL'
        print(f'  VERDICT: MARGINAL improvement found')
        print(f'  Best: {best_label} ({best["mean_tail"]:.4f}) vs production ({prod:.4f})')
        print(f'  Delta: {best["mean_tail"] - prod:+.4f}. Consider applying.')
    else:
        verdict = 'ATTN_MIXED'
        print(f'  VERDICT: MIXED results')
        print(f'  Best: {best_label} ({best["mean_tail"]:.4f})')
        print(f'  Worst: {worst_label} ({worst["mean_tail"]:.4f})')
        print(f'  Spread: {spread:.4f}')

    # Temperature trend
    t8_vals = [r['mean_tail'] for l, r in results.items() if r['temperature'] == 8.0]
    t4_vals = [r['mean_tail'] for l, r in results.items() if r['temperature'] == 4.0]
    t2_vals = [r['mean_tail'] for l, r in results.items() if r['temperature'] == 2.0]

    if t8_vals and t4_vals:
        t8_avg = sum(t8_vals) / len(t8_vals)
        t4_avg = sum(t4_vals) / len(t4_vals)
        t2_avg = sum(t2_vals) / len(t2_vals) if t2_vals else 0
        print(f'\n  Temperature trend: t=8.0 avg={t8_avg:.4f}  t=4.0 avg={t4_avg:.4f}  '
              f't=2.0 avg={t2_avg:.4f}')
        if t4_avg > t8_avg + 0.02:
            print('  -> SHARPER IS BETTER (lower temp wins)')
        elif t8_avg > t4_avg + 0.02:
            print('  -> BROADER IS BETTER (higher temp wins)')
        else:
            print('  -> Temperature has minimal effect')

    print('=' * 70)

    with open(LIVE_LOG, 'a') as lf:
        lf.write(f'\n# VERDICT: {verdict}\n')
        for label, r in results.items():
            lf.write(f'# {label}: r={r["radius"]} t={r["temperature"]} '
                     f'mean_tail={r["mean_tail"]:.4f}\n')
