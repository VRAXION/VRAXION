#!/usr/bin/env python3
"""
Probe: Cheapest Possible Bottleneck Alternatives at D=2048
==========================================================
Single-phase, no warm-up. Direct num_bits=200 for 800 steps.

Configs (ordered by param count):
  1. per_dim_scale:  D params    — learnable volume knob per dimension
  2. per_dim_gate:   D params    — replace scalar zoom_gate with vector gate
  3. affine:         2D params   — scale + shift per dimension
  4. layernorm:      2D params   — normalize memory to brain statistics
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
LIVE_LOG     = os.path.join(LOG_DIR, 'probe_bn_cheapest_gpu_live.log')
os.makedirs(LOG_DIR, exist_ok=True)

from swarm_model import SwarmByteRingModel, c19_activation


# ==================== CHEAP BOTTLENECK INSTALLS ====================

def install_per_dim_scale(model, embed_dim):
    """D learnable scale params. x = scale * lcx_read."""
    scale = nn.Parameter(torch.ones(embed_dim))
    model.register_parameter('lcx_cheap_scale', scale)
    model.lcx_bn_layers = None

    def _bn(self, x):
        return self.lcx_cheap_scale * x
    model._lcx_bottleneck = types.MethodType(_bn, model)


def install_per_dim_gate(model, embed_dim):
    """D learnable gate params. Per-dim sigmoid gate replacing scalar zoom_gate."""
    gate = nn.Parameter(torch.zeros(embed_dim))
    model.register_parameter('lcx_cheap_gate', gate)
    model.lcx_bn_layers = None

    def _bn(self, x):
        return torch.sigmoid(self.lcx_cheap_gate) * x
    model._lcx_bottleneck = types.MethodType(_bn, model)


def install_affine(model, embed_dim):
    """2D learnable params. x = scale * lcx_read + bias."""
    scale = nn.Parameter(torch.ones(embed_dim))
    bias = nn.Parameter(torch.zeros(embed_dim))
    model.register_parameter('lcx_cheap_scale', scale)
    model.register_parameter('lcx_cheap_bias', bias)
    model.lcx_bn_layers = None

    def _bn(self, x):
        return self.lcx_cheap_scale * x + self.lcx_cheap_bias
    model._lcx_bottleneck = types.MethodType(_bn, model)


def install_layernorm(model, embed_dim):
    """2D learnable params (LN affine). Normalize memory to brain stats."""
    ln = nn.LayerNorm(embed_dim)
    model.lcx_cheap_ln = ln
    model.lcx_bn_layers = None

    def _bn(self, x):
        return self.lcx_cheap_ln(x)
    model._lcx_bottleneck = types.MethodType(_bn, model)


CONFIGS = [
    ('per_dim_scale', install_per_dim_scale, D,   'D volume knobs'),
    ('per_dim_gate',  install_per_dim_gate,  D,   'D sigmoid gates'),
    ('affine',        install_affine,        2*D, 'scale + shift'),
    ('layernorm',     install_layernorm,     2*D, 'normalize to brain stats'),
]


def make_model(num_bits):
    return SwarmByteRingModel(
        embedding_dim=D,
        num_memory_positions=SEQ_LEN,
        num_beings=1,
        depth=DEPTH,
        num_bits=num_bits,
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


def make_echo_batch_bits(batch_size, seq_len, block_size, num_bits):
    xs, ys = [], []
    for _ in range(batch_size):
        block = torch.randint(0, 2, (block_size, num_bits)).float()
        repeats = (seq_len + 2) // block_size + 1
        data = block.repeat(repeats, 1)[:seq_len + 1]
        xs.append(data[:seq_len])
        ys.append(data[1:seq_len + 1])
    return torch.stack(xs).to(DEVICE), torch.stack(ys).to(DEVICE)


def run_one(config_name, install_fn, bn_params_count, label, log_file):
    print(f'\n{"="*70}')
    print(f'CONFIG: {config_name} ({label}) — {bn_params_count:,} params')
    print(f'{"="*70}')

    gc.collect()
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    torch.manual_seed(42)
    random.seed(42)

    model = make_model(NUM_BITS)
    install_fn(model, D)
    model = model.to(DEVICE)
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler('cuda') if DEVICE.type == 'cuda' else None

    def lr_lambda(step):
        if step < WARMUP:
            return step / WARMUP
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    print(f'  params={n_params:,}  num_bits={NUM_BITS}  {STEPS} steps', flush=True)

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
            del model, opt, scaler
            gc.collect()
            torch.cuda.empty_cache()
            return {'name': config_name, 'label': label, 'status': 'OOM'}

        elapsed = time.time() - t0
        if step < 3:
            print(f' {elapsed:.2f}s', flush=True)

        if elapsed > STEP_TIMEOUT:
            print(f'  TIMEOUT at step {step} ({elapsed:.0f}s)')
            break

        with torch.no_grad():
            pred = (out > 0).float()
            acc = (pred == y).float().mean().item()

        if math.isnan(loss.item()):
            print(f'  NaN at step {step}')
            break

        window_accs.append(acc)
        if len(window_accs) > WINDOW:
            window_accs.pop(0)
        smooth_acc = sum(window_accs) / len(window_accs)

        if step % 50 == 0 or step == STEPS - 1:
            checkpoints[step] = smooth_acc
            print(f'  step {step:4d} | loss {loss.item():.6f} | acc {acc:.4f} | '
                  f'smooth={smooth_acc:.4f} | {elapsed:.2f}s', flush=True)

        with open(log_file, 'a') as lf:
            lf.write(f'step {step} | loss {loss.item():.6f} | '
                     f'acc={acc:.4f} smooth={smooth_acc:.4f} '
                     f'RD:{elapsed:.4f} traction={smooth_acc:.4f} '
                     f'shard=0/0 {config_name}\n')

    total_time = time.time() - t_start
    vram_peak = 0.0
    if DEVICE.type == 'cuda':
        vram_peak = torch.cuda.max_memory_allocated() / 1024**3

    # Tail: last 200 steps
    tail_accs = [a for s, a in checkpoints.items() if s >= 600]
    tail_avg = sum(tail_accs) / len(tail_accs) if tail_accs else 0.5

    print(f'\n  Tail (600-799): {tail_avg:.4f}  VRAM: {vram_peak:.2f}G  Time: {total_time:.0f}s')

    del model, opt, scaler
    gc.collect()
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()

    return {
        'name': config_name,
        'label': label,
        'status': 'OK',
        'params': n_params,
        'bn_params': bn_params_count,
        'vram_peak': vram_peak,
        'tail': tail_avg,
        'time': total_time,
    }


if __name__ == '__main__':
    print('=' * 70)
    print('PROBE: CHEAPEST BOTTLENECK ALTERNATIVES (no warm-up)')
    print('=' * 70)
    print(f'  D={D}  depth={DEPTH}  num_bits={NUM_BITS}  {STEPS} steps')
    print(f'  batch={BATCH}  LR={LR}  seed=42  device={DEVICE}')
    if DEVICE.type == 'cuda':
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print()
    for name, _, params, label in CONFIGS:
        print(f'    {name:<16} {params:>6,} params  ({label})')
    print('=' * 70)

    with open(LIVE_LOG, 'w') as f:
        f.write(f'# probe_bn_cheapest_gpu -- {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'# D={D} num_bits={NUM_BITS} {STEPS} steps, no warm-up\n')

    results = []
    for name, install_fn, params, label in CONFIGS:
        with open(LIVE_LOG, 'a') as f:
            f.write(f'\n# === {name} ({label}) ===\n')
        r = run_one(name, install_fn, params, label, LIVE_LOG)
        results.append(r)
        if r['status'] == 'OK':
            print(f'  -> {name}: tail={r["tail"]:.4f} ({r["time"]:.0f}s)')

    # ==================== SUMMARY ====================
    print(f'\n{"="*70}')
    print(f'SUMMARY: CHEAPEST ALTERNATIVES (no P1 warm-up)')
    print(f'  D={D}  num_bits={NUM_BITS}  {STEPS} steps direct')
    print(f'{"="*70}')

    print(f'\n  {"Config":<16} {"BN Params":>10} {"Tail":>8} {"VRAM":>6}  {"Label"}')
    print(f'  {"-"*16} {"-"*10} {"-"*8} {"-"*6}  {"-"*25}')

    for r in results:
        if r['status'] == 'OOM':
            print(f'  {r["name"]:<16} {r["bn_params"]:>10,} {"OOM":>8} {"--":>6}  {r["label"]}')
        else:
            print(f'  {r["name"]:<16} {r["bn_params"]:>10,} {r["tail"]:>7.4f} '
                  f'{r["vram_peak"]:>5.2f}G  {r["label"]}')

    ok = [r for r in results if r['status'] == 'OK']
    if ok:
        best = max(ok, key=lambda r: r['tail'])
        worst = min(ok, key=lambda r: r['tail'])
        print(f'\n  Best:  {best["name"]} tail={best["tail"]:.4f} ({best["bn_params"]:,} params)')
        print(f'  Worst: {worst["name"]} tail={worst["tail"]:.4f}')
        print(f'  Spread: {(best["tail"]-worst["tail"])*100:.2f}%')

        print(f'\n{"="*70}')
        if best['tail'] > 0.5020:
            print(f'  VERDICT: CHEAP_WORKS — {best["name"]} shows signal with just '
                  f'{best["bn_params"]:,} params')
        else:
            print(f'  VERDICT: CHEAP_DEAD — none of the cheap transforms show signal')
            print(f'  The squeeze compression is genuinely needed, not just scaling.')
        print(f'{"="*70}')

    total_wall = sum(r['time'] for r in ok)
    print(f'\nDone. Log: {LIVE_LOG}')
    print(f'Total wall time: {total_wall:.0f}s ({total_wall/60:.1f} min)')
