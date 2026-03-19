#!/usr/bin/env python3
"""
Probe: Weight Decay Ablation — Is WD=0.01 hurting echo256?
===========================================================
Production uses AdamW with weight_decay=0.01 (hardcoded).
All previous probes used Adam (no WD). Test whether WD matters.

Config A (adam_nowd):   Adam, WD=0     → expect ~92.7% (known baseline)
Config B (adamw_0.01):  AdamW, WD=0.01 → ??? (production setting)
Config C (adamw_0.001): AdamW, WD=0.001→ ??? (lighter decay)
Config D (adamw_0.1):   AdamW, WD=0.1  → ??? (heavy decay)

Verdict:
  All > 85%              → WD CLEARED, keep 0.01
  A > 85% AND B < 70%    → WD CONFIRMED as root cause
  B is 5-15% below A     → WD PARTIAL, contributing factor
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
STEP_TIMEOUT  = 30
DEVICE        = torch.device('cpu')

CONFIGS = [
    ('adam_nowd',   'adam',  0.0),     # A: known baseline (no WD)
    ('adamw_0.01',  'adamw', 0.01),   # B: production setting
    ('adamw_0.001', 'adamw', 0.001),  # C: lighter decay
    ('adamw_0.1',   'adamw', 0.1),    # D: heavy decay
]

# ─── PATHS ───────────────────────────────────────────────────
DIAMOND_ROOT = r'S:\AI\work\VRAXION_DEV\Diamond Code'
LOG_DIR      = os.path.join(DIAMOND_ROOT, 'logs', 'probe')
LIVE_LOG     = os.path.join(LOG_DIR, 'probe_weight_decay_live.log')
os.makedirs(LOG_DIR, exist_ok=True)

with open(LIVE_LOG, 'w') as f:
    f.write(f'# probe_weight_decay — {time.strftime("%Y-%m-%d %H:%M:%S")}\n')


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


# ─── MINI PHI-EMA RNN ───────────────────────────────────────
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


# ─── TRAINING LOOP ──────────────────────────────────────────
def run_config(label, opt_type, wd, seed):
    torch.manual_seed(seed)
    random.seed(seed)

    model = MiniPhiRNN(num_bits=NUM_BITS, d=D, depth=DEPTH).to(DEVICE)
    model.train()

    n_params = sum(p.numel() for p in model.parameters())

    if opt_type == 'adam':
        opt = torch.optim.Adam(model.parameters(), lr=LR)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=wd)

    tail_accs = []
    had_nan = False
    had_div = False
    t_start = time.time()

    print(f'\n  [{label} seed={seed}] params={n_params:,}  opt={opt_type}  wd={wd}', flush=True)

    for step in range(STEPS):
        t0 = time.time()

        x, y = make_echo_batch(BATCH, SEQ_LEN, BLOCK_SIZE, NUM_BITS)

        opt.zero_grad()
        out = model(x)
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
        return 0.0, 'NAN', total_time
    if had_div:
        return 0.0, 'DIVERGED', total_time
    if not tail_accs:
        return 0.0, 'NO_TAIL', total_time

    tail_median = sorted(tail_accs)[len(tail_accs) // 2]
    return tail_median, 'OK', total_time


# ─── MAIN ────────────────────────────────────────────────────
if __name__ == '__main__':
    print('=' * 60)
    print('PROBE: Weight Decay Ablation (echo256 BLOCK=4)')
    print('=' * 60)
    print(f'  D={D}  depth={DEPTH}  seq_len={SEQ_LEN}  block={BLOCK_SIZE}')
    print(f'  batch={BATCH}  lr={LR}  steps={STEPS}  seeds={SEEDS}')
    print(f'  configs: {[(l, o, w) for l, o, w in CONFIGS]}')
    print('=' * 60)

    results = {}

    for label, opt_type, wd in CONFIGS:
        seed_tails = []
        seed_times = []

        for seed in SEEDS:
            tail, status, elapsed = run_config(label, opt_type, wd, seed)
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
            'opt_type': opt_type,
            'wd': wd,
        }

    # ─── RESULTS TABLE ───────────────────────────────────────
    print('\n' + '=' * 60)
    print('RESULTS')
    print('=' * 60)
    print(f'  {"Config":<16} {"Opt":>6} {"WD":>8} {"Mean Tail":>10} {"Gap":>8} {"Seeds":>20}')
    print(f'  {"-"*16} {"-"*6} {"-"*8} {"-"*10} {"-"*8} {"-"*20}')
    for label, r in results.items():
        seeds_str = ', '.join(f'{t:.4f}' for t in r['tails'])
        print(f'  {label:<16} {r["opt_type"]:>6} {r["wd"]:>8.4f} {r["mean_tail"]:>10.4f} '
              f'{r["seed_gap"]:>8.4f} {seeds_str:>20}')

    a = results.get('adam_nowd', {}).get('mean_tail', 0)
    b = results.get('adamw_0.01', {}).get('mean_tail', 0)
    c = results.get('adamw_0.001', {}).get('mean_tail', 0)
    d = results.get('adamw_0.1', {}).get('mean_tail', 0)

    print(f'\n  Delta (adamw_0.01 - adam):   {b - a:+.4f}')
    print(f'  Delta (adamw_0.001 - adam):  {c - a:+.4f}')
    print(f'  Delta (adamw_0.1 - adam):    {d - a:+.4f}')

    # ─── VERDICT ─────────────────────────────────────────────
    print('\n' + '=' * 60)
    if a > 0.85 and b > 0.85 and c > 0.85:
        verdict = 'WD_CLEARED'
        print('  VERDICT: Weight decay CLEARED')
        print(f'  All configs > 85%. WD=0.01 is fine. Keep production setting.')
        if d < 0.70:
            print(f'  (WD=0.1 is too heavy at {d:.4f} — avoid extreme values)')
    elif a > 0.85 and b < 0.70:
        verdict = 'WD_CONFIRMED'
        print('  VERDICT: Weight decay CONFIRMED as root cause')
        print(f'  WD=0.01 kills learning ({b:.4f}). Remove or reduce.')
        if c > 0.85:
            print(f'  WD=0.001 is safe ({c:.4f}). Use that instead.')
    elif a > 0.85 and (a - b) > 0.05:
        verdict = 'WD_PARTIAL'
        print(f'  VERDICT: Weight decay is a PARTIAL factor ({b:.4f} vs {a:.4f})')
        best_wd_label = min(['adamw_0.01', 'adamw_0.001', 'adamw_0.1'],
                            key=lambda k: -results.get(k, {}).get('mean_tail', 0))
        best_wd = results[best_wd_label]
        print(f'  Best WD config: {best_wd_label} ({best_wd["mean_tail"]:.4f})')
    else:
        verdict = 'UNEXPECTED'
        print(f'  VERDICT: UNEXPECTED (A={a:.4f}, B={b:.4f}, C={c:.4f}, D={d:.4f})')
    print('=' * 60)

    with open(LIVE_LOG, 'a') as lf:
        lf.write(f'\n# VERDICT: {verdict}\n')
        lf.write(f'# adam_nowd={a:.4f} adamw_0.01={b:.4f} adamw_0.001={c:.4f} adamw_0.1={d:.4f}\n')
