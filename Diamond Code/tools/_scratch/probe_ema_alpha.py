#!/usr/bin/env python3
"""
Probe: STATE_EMA_ALPHA — Golden Ratio Blend Ablation
=====================================================
The state update formula:
  state_update = ALPHA * norm(hidden) + (1-ALPHA) * combined_input

Current: ALPHA=0.618 (golden ratio, never tested).
High ALPHA = strong memory (old state dominant).
Low  ALPHA = fast adaptation (new input dominant).

Test: 0.3, 0.5, 0.618 (control), 0.7, 0.9
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
SEEDS         = [42, 137]
NUM_BITS      = 8
STEP_TIMEOUT  = 60
DEVICE        = torch.device('cpu')
RADIUS        = 8

CONFIGS = [
    # (label, alpha_value)
    ('alpha_0.3',   0.3),    # new input dominant — fast, reactive
    ('alpha_0.5',   0.5),    # standard balanced EMA
    ('alpha_0.618', 0.618),  # golden ratio (control)
    ('alpha_0.7',   0.7),    # strong memory momentum
    ('alpha_0.9',   0.9),    # near-frozen state, very slow update
]

DIAMOND_ROOT = r'S:\AI\work\VRAXION_DEV\Diamond Code'
sys.path.insert(0, DIAMOND_ROOT)
LOG_DIR      = os.path.join(DIAMOND_ROOT, 'logs', 'probe')
LIVE_LOG     = os.path.join(LOG_DIR, 'probe_ema_alpha_live.log')
os.makedirs(LOG_DIR, exist_ok=True)

with open(LIVE_LOG, 'w') as f:
    f.write(f'# probe_ema_alpha — {time.strftime("%Y-%m-%d %H:%M:%S")}\n')

import swarm_model
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


def make_model():
    return SwarmByteRingModel(
        embedding_dim=D,
        num_memory_positions=SEQ_LEN,
        num_beings=1,
        depth=DEPTH,
        num_bits=NUM_BITS,
        attention_radius=RADIUS,
        attention_temperature=8.0,
        think_ticks=0,
        use_lcx=False,
        num_pointers=1,
    )


def run_config(label, alpha_value, seed):
    swarm_model.STATE_EMA_ALPHA = alpha_value

    torch.manual_seed(seed)
    random.seed(seed)

    model = make_model().to(DEVICE)
    model.train()

    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    tail_accs = []
    had_nan = False
    had_div = False
    t_start = time.time()

    print(f'\n  [{label} seed={seed}] params={n_params:,}  ALPHA={alpha_value}',
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
    print('PROBE: STATE_EMA_ALPHA — Golden Ratio Blend Ablation')
    print('=' * 70)
    print(f'  D={D}  depth={DEPTH}  seq_len={SEQ_LEN}  block={BLOCK_SIZE}')
    print(f'  batch={BATCH}  lr={LR}  steps={STEPS}  seeds={SEEDS}  radius={RADIUS}')
    print(f'  configs: {[(l, a) for l, a in CONFIGS]}')
    print('=' * 70)

    results = {}
    for label, alpha_value in CONFIGS:
        seed_tails = []
        seed_times = []
        params = 0
        for seed in SEEDS:
            tail, status, elapsed, params = run_config(label, alpha_value, seed)
            seed_tails.append(tail)
            seed_times.append(elapsed)
            print(f'    -> tail_median={tail:.4f}  status={status}  time={elapsed:.1f}s')

        mean_tail = sum(seed_tails) / len(seed_tails)
        seed_gap = max(seed_tails) - min(seed_tails)
        results[label] = {
            'mean_tail': mean_tail, 'seed_gap': seed_gap,
            'tails': seed_tails, 'times': seed_times,
            'params': params, 'alpha': alpha_value,
        }

    # Restore default
    swarm_model.STATE_EMA_ALPHA = 0.618

    # Results
    print('\n' + '=' * 70)
    print('RESULTS: STATE_EMA_ALPHA Ablation')
    print('=' * 70)
    print(f'  {"Config":<12} {"ALPHA":>6} {"Old:New":>8} {"Mean Tail":>10} '
          f'{"Gap":>8} {"Seeds":>20}')
    print(f'  {"-"*12} {"-"*6} {"-"*8} {"-"*10} {"-"*8} {"-"*20}')

    for label, r in results.items():
        old_pct = int(r['alpha'] * 100)
        new_pct = 100 - old_pct
        ratio_str = f'{old_pct}:{new_pct}'
        seeds_str = ', '.join(f'{t:.4f}' for t in r['tails'])
        print(f'  {label:<12} {r["alpha"]:>6.3f} {ratio_str:>8} '
              f'{r["mean_tail"]:>10.4f} {r["seed_gap"]:>8.4f} {seeds_str:>20}')

    ctrl = results.get('alpha_0.618', {}).get('mean_tail', 0)
    print(f'\n  Deltas vs alpha_0.618 ({ctrl:.4f}):')
    for label, r in results.items():
        if label != 'alpha_0.618':
            print(f'    {label:<12}: {r["mean_tail"] - ctrl:+.4f}')

    # Check for monotonic trend
    alphas = [r['alpha'] for r in results.values()]
    accs = [r['mean_tail'] for r in results.values()]
    increasing = all(accs[i] <= accs[i+1] + 0.005 for i in range(len(accs)-1))
    decreasing = all(accs[i] >= accs[i+1] - 0.005 for i in range(len(accs)-1))

    # Verdict
    print('\n' + '=' * 70)
    best_label = max(results, key=lambda k: results[k]['mean_tail'])
    worst_label = min(results, key=lambda k: results[k]['mean_tail'])
    best = results[best_label]
    worst = results[worst_label]
    spread = best['mean_tail'] - worst['mean_tail']

    if spread < 0.015:
        print(f'  VERDICT: EMA_CLEARED')
        print(f'  Spread: {spread:.4f} (<1.5%). STATE_EMA_ALPHA doesn\'t matter.')
        print(f'  Keep alpha=0.618 (golden ratio).')
    elif best['mean_tail'] - ctrl > 0.02:
        print(f'  VERDICT: EMA_OPTIMIZED')
        print(f'  Best: {best_label} ({best["mean_tail"]:.4f}) vs control ({ctrl:.4f})')
        print(f'  Delta: {best["mean_tail"] - ctrl:+.4f}. Apply alpha={best["alpha"]}.')
        if increasing:
            print(f'  TREND: Higher alpha = better (strong memory momentum wins)')
        elif decreasing:
            print(f'  TREND: Lower alpha = better (fast input integration wins)')
    else:
        print(f'  VERDICT: EMA_MIXED')
        print(f'  Best: {best_label} ({best["mean_tail"]:.4f})')
        print(f'  Worst: {worst_label} ({worst["mean_tail"]:.4f})')
        print(f'  Spread: {spread:.4f}')
        if increasing:
            print(f'  TREND: Higher alpha = better (momentum)')
        elif decreasing:
            print(f'  TREND: Lower alpha = better (reactivity)')

    print('=' * 70)

    with open(LIVE_LOG, 'a') as lf:
        lf.write(f'\n# RESULTS:\n')
        for label, r in results.items():
            lf.write(f'# {label}: ALPHA={r["alpha"]} mean_tail={r["mean_tail"]:.4f}\n')
