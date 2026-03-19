#!/usr/bin/env python3
"""
Probe: Mean-Pool vs Gaussian Attention (v3)
============================================
v1+v2 showed: higher temp = better (monotonic up to t=64 → 90.3%).
As temp→∞, Gaussian softmax → uniform weights → mean-pooling.

This probe tests: does true mean-pool match or beat t=64?
If yes → delete the temperature hyperparameter entirely.

Configs (all r=8):
  gauss_t64   — Gaussian t=64 (best from v2, control)
  gauss_t256  — Gaussian t=256 (4x beyond best)
  gauss_t10k  — Gaussian t=10000 (mathematically ~uniform)
  mean_pool   — True uniform 1/(2R+1), no softmax at all
"""

import math
import os
import random
import sys
import time
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

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
RADIUS        = 8

CONFIGS = [
    # (label, temperature, use_mean_pool)
    ('gauss_t64',   64.0,   False),   # best from v2 (control)
    ('gauss_t256',  256.0,  False),   # 4x beyond best
    ('gauss_t10k',  10000.0, False),  # mathematically ~uniform
    ('mean_pool',   None,   True),    # true uniform, no softmax
]

DIAMOND_ROOT = r'S:\AI\work\VRAXION_DEV\Diamond Code'
sys.path.insert(0, DIAMOND_ROOT)
LOG_DIR      = os.path.join(DIAMOND_ROOT, 'logs', 'probe')
LIVE_LOG     = os.path.join(LOG_DIR, 'probe_attention_v3_live.log')
os.makedirs(LOG_DIR, exist_ok=True)

with open(LIVE_LOG, 'w') as f:
    f.write(f'# probe_attention_v3 — {time.strftime("%Y-%m-%d %H:%M:%S")}\n')

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


def _uniform_attention_weights(self, pointer_position, num_positions):
    """Replace Gaussian with uniform weights over the same window."""
    B = pointer_position.size(0)
    K = self.attention_radius
    window_size = 2 * K + 1

    offsets = torch.arange(-K, K + 1, device=pointer_position.device)
    base = torch.floor(pointer_position).long().clamp(0, num_positions - 1)
    indices = (base.unsqueeze(1) + offsets.unsqueeze(0)) % num_positions

    # Uniform weights: 1/(2R+1) for every position in the window
    weights = torch.ones(B, window_size, device=pointer_position.device) / window_size

    return indices, weights


def make_model(temperature, use_mean_pool):
    temp = temperature if temperature is not None else 8.0  # placeholder for constructor
    model = SwarmByteRingModel(
        embedding_dim=D,
        num_memory_positions=SEQ_LEN,
        num_beings=1,
        depth=DEPTH,
        num_bits=NUM_BITS,
        attention_radius=RADIUS,
        attention_temperature=temp,
        think_ticks=0,
        use_lcx=False,
        num_pointers=1,
    )

    if use_mean_pool:
        # Monkey-patch: replace Gaussian attention with uniform
        model._gaussian_attention_weights = types.MethodType(
            _uniform_attention_weights, model
        )

    return model


def run_config(label, temperature, use_mean_pool, seed):
    torch.manual_seed(seed)
    random.seed(seed)

    model = make_model(temperature, use_mean_pool).to(DEVICE)
    model.train()

    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    tail_accs = []
    had_nan = False
    had_div = False
    t_start = time.time()

    method = 'mean_pool' if use_mean_pool else f'gauss_t={temperature}'
    print(f'\n  [{label} seed={seed}] params={n_params:,}  r={RADIUS} {method}',
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
            print(f'  NaN at step {step}, aborting')
            break
        if loss.item() > 3.0 and step > 200:
            had_div = True
            print(f'  Divergence at step {step} (loss={loss.item():.4f})')
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


if __name__ == '__main__':
    print('=' * 70)
    print('PROBE: Mean-Pool vs Gaussian Attention (v3)')
    print('=' * 70)
    print(f'  D={D}  depth={DEPTH}  seq_len={SEQ_LEN}  block={BLOCK_SIZE}')
    print(f'  batch={BATCH}  lr={LR}  steps={STEPS}  seeds={SEEDS}  radius={RADIUS}')
    print(f'  configs: {[(l, t, m) for l, t, m in CONFIGS]}')
    print('=' * 70)

    results = {}
    for label, temperature, use_mean_pool in CONFIGS:
        seed_tails = []
        seed_times = []
        params = 0
        for seed in SEEDS:
            tail, status, elapsed, params = run_config(label, temperature, use_mean_pool, seed)
            seed_tails.append(tail)
            seed_times.append(elapsed)
            print(f'    -> tail_median={tail:.4f}  status={status}  time={elapsed:.1f}s')

        mean_tail = sum(seed_tails) / len(seed_tails)
        seed_gap = max(seed_tails) - min(seed_tails)
        results[label] = {
            'mean_tail': mean_tail, 'seed_gap': seed_gap,
            'tails': seed_tails, 'times': seed_times,
            'params': params, 'temperature': temperature,
            'use_mean_pool': use_mean_pool,
        }

    # Results
    print('\n' + '=' * 70)
    print('RESULTS (v3: mean-pool vs Gaussian)')
    print('=' * 70)

    # Reference curve from v1+v2
    print('  Full temperature curve (v1+v2 reference):')
    print('    t=2:  0.8409  |  t=4:  0.8590  |  t=8:  0.8777')
    print('    t=16: 0.8813  |  t=32: 0.8945  |  t=64: 0.9033')
    print()

    print(f'  {"Config":<12} {"Method":<16} {"Mean Tail":>10} {"Gap":>8} {"Seeds":>20}')
    print(f'  {"-"*12} {"-"*16} {"-"*10} {"-"*8} {"-"*20}')

    for label, r in results.items():
        method = 'mean_pool' if r['use_mean_pool'] else f'gauss t={r["temperature"]}'
        seeds_str = ', '.join(f'{t:.4f}' for t in r['tails'])
        print(f'  {label:<12} {method:<16} {r["mean_tail"]:>10.4f} '
              f'{r["seed_gap"]:>8.4f} {seeds_str:>20}')

    ctrl = results.get('gauss_t64', {}).get('mean_tail', 0)
    mp = results.get('mean_pool', {}).get('mean_tail', 0)

    print(f'\n  Deltas vs gauss_t64 ({ctrl:.4f}):')
    for label, r in results.items():
        if label != 'gauss_t64':
            print(f'    {label:<12}: {r["mean_tail"] - ctrl:+.4f}')

    # Verdict
    print('\n' + '=' * 70)
    gap = abs(mp - ctrl)
    if gap < 0.01:
        print('  VERDICT: SIMPLIFY')
        print(f'  Mean-pool ({mp:.4f}) matches Gaussian t=64 ({ctrl:.4f})')
        print(f'  Gap: {gap:.4f} (<1%). Replace Gaussian with mean-pool.')
        print(f'  Delete attention_temperature parameter from model.')
    elif mp > ctrl + 0.02:
        print('  VERDICT: MEAN_POOL WINS')
        print(f'  Mean-pool ({mp:.4f}) > Gaussian t=64 ({ctrl:.4f}) by {mp - ctrl:+.4f}')
        print(f'  Definitely replace Gaussian with mean-pool.')
    elif ctrl > mp + 0.02:
        print('  VERDICT: KEEP GAUSSIAN')
        print(f'  Gaussian t=64 ({ctrl:.4f}) > mean-pool ({mp:.4f}) by {ctrl - mp:+.4f}')
        print(f'  The Gaussian shape still matters. Lock t=64.')
    else:
        print(f'  VERDICT: MARGINAL (gap={gap:.4f})')
        print(f'  Mean-pool ({mp:.4f}) vs Gaussian t=64 ({ctrl:.4f})')
        print(f'  Difference is noise-level. Mean-pool is simpler, recommend switching.')

    # Full combined curve
    print(f'\n  COMPLETE TEMPERATURE CURVE (r=8, all probes):')
    ref = [(2, 0.8409), (4, 0.8590), (8, 0.8777), (16, 0.8813), (32, 0.8945)]
    for t, v in ref:
        print(f'    t={t:<6}  {v:.4f}  (v1/v2)')
    for label, r in results.items():
        if not r['use_mean_pool']:
            t = r['temperature']
            print(f'    t={t:<6.0f}  {r["mean_tail"]:.4f}  (v3)')
    mp_r = results.get('mean_pool', {})
    if mp_r:
        print(f'    uniform   {mp_r["mean_tail"]:.4f}  (v3 mean-pool)')

    print('=' * 70)

    with open(LIVE_LOG, 'a') as lf:
        lf.write(f'\n# RESULTS:\n')
        for label, r in results.items():
            lf.write(f'# {label}: mean_tail={r["mean_tail"]:.4f}\n')
