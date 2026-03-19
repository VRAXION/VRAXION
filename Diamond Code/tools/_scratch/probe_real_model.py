#!/usr/bin/env python3
"""
Probe: Real SwarmByteRingModel vs MiniPhiRNN — Architecture Ablation
=====================================================================
Tests whether the FULL SwarmByteRingModel architecture (ring memory,
beings, pointers, Gaussian attention) kills echo256 learning.

Config A (mini):   MiniPhiRNN D=128, depth=4          → expect ~92.7% (known)
Config B (real):   SwarmByteRingModel D=128, LCX OFF   → ??? (the test)
Config C (real_lcx): SwarmByteRingModel D=128, LCX ON  → ??? (bonus)

All at LR=1e-3 (known good), echo256 BLOCK=4, seq_len=32.

Verdict:
  A > 80% AND B > 80%  → Architecture CLEARED. Problem is elsewhere.
  A > 80% AND B < 55%  → Ring/architecture CONFIRMED as root cause.
  A > 80% AND B 55-80% → Architecture adds interference but isn't fatal.
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
STEP_TIMEOUT  = 60   # real model is slower, give more time
DEVICE        = torch.device('cpu')

# ─── PATHS ───────────────────────────────────────────────────
DIAMOND_ROOT = r'S:\AI\work\VRAXION_DEV\Diamond Code'
sys.path.insert(0, DIAMOND_ROOT)
LOG_DIR      = os.path.join(DIAMOND_ROOT, 'logs', 'probe')
LIVE_LOG     = os.path.join(LOG_DIR, 'probe_real_model_live.log')
os.makedirs(LOG_DIR, exist_ok=True)

with open(LIVE_LOG, 'w') as f:
    f.write(f'# probe_real_model — {time.strftime("%Y-%m-%d %H:%M:%S")}\n')

# Import real model
from swarm_model import SwarmByteRingModel


# ─── C19 ACTIVATION (from swarm_model.py:29) ────────────────
def c19_activation(x, rho=4.0):
    l = 6.0 * math.pi
    inv_pi = 1.0 / math.pi
    scaled = x * inv_pi
    n = torch.floor(scaled)
    t = scaled - n
    h = t * (1.0 - t)
    is_even = torch.remainder(n, 2.0) < 1.0
    sgn = torch.where(is_even, torch.ones_like(x), -torch.ones_like(x))
    core = math.pi * (sgn * h + (rho * h * h))
    return torch.where(x >= l, x - l, torch.where(x <= -l, x + l, core))


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


# ─── MINI PHI-EMA RNN (control — known good) ────────────────
class MiniPhiRNN(nn.Module):
    def __init__(self, num_bits=8, d=128, depth=4):
        super().__init__()
        self.d = d
        self.input_proj = nn.Linear(num_bits, d)
        self.output_proj = nn.Linear(d, num_bits)
        self.state_norm = nn.LayerNorm(d)
        self.layers = nn.ModuleList([nn.Linear(d, d) for _ in range(depth - 1)])
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(depth - 1)])

    def forward(self, x):
        B, T, _ = x.shape
        h = torch.zeros(B, self.d, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(T):
            inp = self.input_proj(x[:, t, :])
            state = 0.618 * self.state_norm(h) + 0.382 * inp
            for norm, layer in zip(self.norms, self.layers):
                state = state + c19_activation(layer(norm(state)))
            h = state
            outputs.append(self.output_proj(h))
        return torch.stack(outputs, dim=1)


# ─── MODEL FACTORIES ────────────────────────────────────────
def make_mini():
    return MiniPhiRNN(num_bits=NUM_BITS, d=D, depth=DEPTH)


def make_real_no_lcx():
    return SwarmByteRingModel(
        embedding_dim=D,
        num_memory_positions=SEQ_LEN,
        num_beings=1,
        depth=DEPTH,
        num_bits=NUM_BITS,
        attention_radius=6,
        attention_temperature=8.0,
        think_ticks=0,
        use_lcx=False,
        num_pointers=1,
    )


def make_real_lcx():
    return SwarmByteRingModel(
        embedding_dim=D,
        num_memory_positions=SEQ_LEN,
        num_beings=1,
        depth=DEPTH,
        num_bits=NUM_BITS,
        attention_radius=6,
        attention_temperature=8.0,
        think_ticks=1,
        use_lcx=True,
        lcx_mode="dense",
        lcx_num_slots=256,
        lcx_key_dim=12,
        lcx_top_k=2,
        num_pointers=1,
    )


CONFIGS = [
    ('mini',     make_mini),       # A: control (known ~92.7%)
    ('real',     make_real_no_lcx), # B: full arch, no memory
    ('real_lcx', make_real_lcx),    # C: full arch + LCX
]


# ─── TRAINING LOOP ──────────────────────────────────────────
def run_config(label, model_factory, seed):
    torch.manual_seed(seed)
    random.seed(seed)

    model = model_factory().to(DEVICE)
    model.train()

    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    tail_accs = []
    had_nan = False
    had_div = False
    t_start = time.time()

    print(f'\n  [{label} seed={seed}] params={n_params:,}  lr={LR}', flush=True)

    for step in range(STEPS):
        t0 = time.time()

        x, y = make_echo_batch(BATCH, SEQ_LEN, BLOCK_SIZE, NUM_BITS)

        opt.zero_grad()

        # Real model may return stats tuple
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
    print('=' * 60)
    print('PROBE: Real SwarmByteRingModel vs MiniPhiRNN')
    print('=' * 60)
    print(f'  D={D}  depth={DEPTH}  seq_len={SEQ_LEN}  block={BLOCK_SIZE}')
    print(f'  batch={BATCH}  lr={LR}  steps={STEPS}  seeds={SEEDS}')
    print(f'  configs: {[c[0] for c in CONFIGS]}')
    print('=' * 60)

    results = {}

    for label, factory in CONFIGS:
        seed_tails = []
        seed_times = []
        params = 0

        for seed in SEEDS:
            tail, status, elapsed, params = run_config(label, factory, seed)
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
        }

    # ─── RESULTS TABLE ───────────────────────────────────────
    print('\n' + '=' * 60)
    print('RESULTS')
    print('=' * 60)
    print(f'  {"Config":<12} {"Params":>10} {"Mean Tail":>10} {"Gap":>8} {"Seeds":>20}')
    print(f'  {"-"*12} {"-"*10} {"-"*10} {"-"*8} {"-"*20}')
    for label, r in results.items():
        seeds_str = ', '.join(f'{t:.4f}' for t in r['tails'])
        print(f'  {label:<12} {r["params"]:>10,} {r["mean_tail"]:>10.4f} {r["seed_gap"]:>8.4f} {seeds_str:>20}')

    a = results.get('mini', {}).get('mean_tail', 0)
    b = results.get('real', {}).get('mean_tail', 0)
    c = results.get('real_lcx', {}).get('mean_tail', 0)
    delta_ab = b - a
    delta_ac = c - a

    print(f'\n  Delta (real - mini):     {delta_ab:+.4f}')
    print(f'  Delta (real_lcx - mini): {delta_ac:+.4f}')

    # ─── VERDICT ─────────────────────────────────────────────
    print('\n' + '=' * 60)
    if a > 0.80 and b > 0.80:
        verdict = 'ARCH_CLEARED'
        print('  VERDICT: Architecture CLEARED')
        print(f'  Ring + beings + pointers do NOT kill learning (B={b:.4f})')
        if c > 0.80:
            print(f'  LCX also cleared (C={c:.4f})')
        elif c < 0.55:
            print(f'  LCX kills learning (C={c:.4f}) — LCX is the root cause')
        else:
            print(f'  LCX adds interference (C={c:.4f}) — contributing factor')
    elif a > 0.80 and b < 0.55:
        verdict = 'ARCH_CONFIRMED'
        print('  VERDICT: Architecture CONFIRMED as root cause')
        print(f'  Ring/beings/pointers kill echo256 learning (B={b:.4f})')
    elif a > 0.80:
        verdict = 'ARCH_PARTIAL'
        print(f'  VERDICT: Architecture adds interference (B={b:.4f})')
        print('  Not fatal but contributing factor.')
    else:
        verdict = 'UNEXPECTED'
        print(f'  VERDICT: UNEXPECTED (A={a:.4f}, B={b:.4f}, C={c:.4f})')
    print('=' * 60)

    with open(LIVE_LOG, 'a') as lf:
        lf.write(f'\n# VERDICT: {verdict}\n')
        lf.write(f'# mini mean_tail={a:.4f}\n')
        lf.write(f'# real mean_tail={b:.4f}\n')
        lf.write(f'# real_lcx mean_tail={c:.4f}\n')
        lf.write(f'# delta_ab={delta_ab:+.4f}\n')
        lf.write(f'# delta_ac={delta_ac:+.4f}\n')
