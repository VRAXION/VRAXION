#!/usr/bin/env python3
"""
Probe: Attention Temperature Extended — Does higher temp keep winning?
======================================================================
v1 showed: t=8 > t=4 > t=2 (broader is better). And r=8 best radius.
Now push temperature higher to find the peak: t=8, t=16, t=32, t=64.
All at r=8 (best radius from v1).

Also test r=8 with extreme temp to see if there's a ceiling.
"""

import math
import os
import random
import sys
import time

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

CONFIGS = [
    # (label, radius, temperature)
    ('r8_t8',   8, 8.0),    # best from v1 (control)
    ('r8_t16',  8, 16.0),   # 2x broader
    ('r8_t32',  8, 32.0),   # 4x broader
    ('r8_t64',  8, 64.0),   # 8x broader (near uniform)
    ('r6_t16',  6, 16.0),   # production radius + higher temp
    ('r6_t32',  6, 32.0),   # production radius + much higher temp
]

DIAMOND_ROOT = r'S:\AI\work\VRAXION_DEV\Diamond Code'
sys.path.insert(0, DIAMOND_ROOT)
LOG_DIR      = os.path.join(DIAMOND_ROOT, 'logs', 'probe')
LIVE_LOG     = os.path.join(LOG_DIR, 'probe_attention_v2_live.log')
os.makedirs(LOG_DIR, exist_ok=True)

with open(LIVE_LOG, 'w') as f:
    f.write(f'# probe_attention_v2 — {time.strftime("%Y-%m-%d %H:%M:%S")}\n')

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
    print('PROBE: Attention Temperature Extended (echo256 BLOCK=4)')
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
            'mean_tail': mean_tail, 'seed_gap': seed_gap,
            'tails': seed_tails, 'times': seed_times,
            'params': params, 'radius': radius, 'temperature': temperature,
        }

    # Combined results (v1 reference + v2 new)
    print('\n' + '=' * 70)
    print('RESULTS (v2 extended)')
    print('=' * 70)
    # Reference from v1
    print('  v1 reference: r6_t8=0.8635  r6_t4=0.8590  r6_t2=0.8409  r8_t8=0.8777')
    print()
    print(f'  {"Config":<10} {"R":>3} {"Temp":>6} {"Mean Tail":>10} {"Gap":>8} {"Seeds":>20}')
    print(f'  {"-"*10} {"-"*3} {"-"*6} {"-"*10} {"-"*8} {"-"*20}')

    for label, r in results.items():
        seeds_str = ', '.join(f'{t:.4f}' for t in r['tails'])
        print(f'  {label:<10} {r["radius"]:>3} {r["temperature"]:>6.1f} '
              f'{r["mean_tail"]:>10.4f} {r["seed_gap"]:>8.4f} {seeds_str:>20}')

    ctrl = results.get('r8_t8', {}).get('mean_tail', 0)
    print(f'\n  Deltas vs r8_t8 ({ctrl:.4f}):')
    for label, r in results.items():
        if label != 'r8_t8':
            print(f'    {label:<10}: {r["mean_tail"] - ctrl:+.4f}')

    # Full temperature curve (including v1 data)
    print('\n  FULL TEMPERATURE CURVE (r=8, all probes combined):')
    print(f'    t=2.0:  ~0.841 (v1 r6_t2)')
    print(f'    t=4.0:  ~0.859 (v1 r6_t4)')
    print(f'    t=8.0:  {results["r8_t8"]["mean_tail"]:.4f} (v1+v2)')
    for t_val in [16, 32, 64]:
        key = f'r8_t{t_val}'
        if key in results:
            print(f'    t={t_val:.1f}: {results[key]["mean_tail"]:.4f}')

    best_label = max(results, key=lambda k: results[k]['mean_tail'])
    best = results[best_label]
    print(f'\n  BEST: {best_label} (r={best["radius"]}, t={best["temperature"]}) = {best["mean_tail"]:.4f}')
    print('=' * 70)

    with open(LIVE_LOG, 'a') as lf:
        lf.write(f'\n# BEST: {best_label} = {best["mean_tail"]:.4f}\n')
        for label, r in results.items():
            lf.write(f'# {label}: mean_tail={r["mean_tail"]:.4f}\n')
