#!/usr/bin/env python3
"""
Probe: Single-Layer D->D C19 Bottleneck at D=2048
==================================================
Simplest possible learned transform: one layer of D C19 neurons.

    hidden = hidden + zoom_gate * C19(W @ lcx_read + b)

No squeeze, no multi-layer, no ratios. One matrix multiply, one activation.

Comparison targets (all D=2048, 800 P2 steps):
  noBN       : 50.13%  (no transform)
  BN=204 2L  : 50.39%  (current 3-Linear squeeze, ratio 1.0:1)
  BN=409 2L  : 50.64%  (wider squeeze, ratio 2.0:1)
  1-layer D  : ???     (THIS PROBE)
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

# --- Configs (match GPU matchup exactly) ---
D              = 2048
DEPTH          = 2
SEQ_LEN        = 32
BLOCK_SIZE     = 16
BATCH          = 10
LR             = 1e-4
WARMUP         = 50
NUM_BITS_P1    = 8
NUM_BITS_P2    = 200
STEPS_P1       = 200
STEPS_P2       = 800       # same as matchup for direct comparison
STEP_TIMEOUT   = 120
RADIUS         = 8
THINK_TICKS    = 1

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DIAMOND_ROOT = r'S:\AI\work\VRAXION_DEV\Diamond Code'
sys.path.insert(0, DIAMOND_ROOT)
LOG_DIR      = os.path.join(DIAMOND_ROOT, 'logs', 'probe')
LIVE_LOG     = os.path.join(LOG_DIR, 'probe_bn_single_layer_gpu_live.log')
os.makedirs(LOG_DIR, exist_ok=True)

from swarm_model import SwarmByteRingModel, c19_activation


def install_single_layer_bn(model, embed_dim):
    """Replace lcx_bn_layers with a single Linear(D,D) and patch the method."""
    model.lcx_bn_layers = nn.ModuleList([
        nn.Linear(embed_dim, embed_dim),
    ])
    nn.init.orthogonal_(model.lcx_bn_layers[0].weight)
    nn.init.zeros_(model.lcx_bn_layers[0].bias)
    model.lcx_bn_layers.to(next(model.parameters()).device)

    # Monkey-patch: apply C19 after the single layer (default code skips activation on last layer)
    def _custom_lcx_bottleneck(self, x):
        x = self.lcx_bn_layers[0](x)
        x = c19_activation(x)
        return x

    model._lcx_bottleneck = types.MethodType(_custom_lcx_bottleneck, model)


def make_model(num_bits):
    """Create D=2048 model, then install single-layer BN."""
    model = SwarmByteRingModel(
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
        lcx_key_dim=D // 10,    # 204 -- FIXED (same routing for all)
        lcx_top_k=2,
        num_pointers=1,
    )
    install_single_layer_bn(model, D)
    return model


def make_echo_batch_bits(batch_size, seq_len, block_size, num_bits):
    """Generate repeating block patterns with arbitrary num_bits width."""
    xs, ys = [], []
    for _ in range(batch_size):
        block = torch.randint(0, 2, (block_size, num_bits)).float()
        repeats = (seq_len + 2) // block_size + 1
        data = block.repeat(repeats, 1)[:seq_len + 1]
        xs.append(data[:seq_len])
        ys.append(data[1:seq_len + 1])
    return torch.stack(xs).to(DEVICE), torch.stack(ys).to(DEVICE)


def expand_model_bits(model, old_bits, new_bits):
    """Expand num_bits: create new model, copy matching weights."""
    new_model = make_model(new_bits).to(DEVICE)
    new_model.train()

    old_state = model.state_dict()
    new_state = new_model.state_dict()

    copied, expanded, skipped = 0, 0, 0
    for key in new_state:
        if key in old_state:
            old_shape = old_state[key].shape
            new_shape = new_state[key].shape
            if old_shape == new_shape:
                new_state[key] = old_state[key]
                copied += 1
            elif len(old_shape) == 2 and len(new_shape) == 2:
                min_r = min(old_shape[0], new_shape[0])
                min_c = min(old_shape[1], new_shape[1])
                new_state[key][:min_r, :min_c] = old_state[key][:min_r, :min_c]
                expanded += 1
            elif len(old_shape) == 1 and len(new_shape) == 1:
                min_len = min(old_shape[0], new_shape[0])
                new_state[key][:min_len] = old_state[key][:min_len]
                expanded += 1
            else:
                skipped += 1
        else:
            skipped += 1

    new_model.load_state_dict(new_state)
    print(f'  Expansion: {copied} copied, {expanded} expanded, {skipped} new')
    return new_model


def count_bn_params(embed_dim):
    """Params in a single D->D layer."""
    return embed_dim * embed_dim + embed_dim  # W + b


def count_old_bn_params(embed_dim):
    """Params in the old 3-layer squeeze (D->D/10->D/10->D)."""
    bn = embed_dim // 10
    return embed_dim * bn + bn + bn * bn + bn + bn * embed_dim + embed_dim


if __name__ == '__main__':
    new_bn_params = count_bn_params(D)
    old_bn_params = count_old_bn_params(D)

    print('=' * 70)
    print('PROBE: SINGLE-LAYER D->D C19 BOTTLENECK')
    print('=' * 70)
    print(f'  D={D}  depth={DEPTH}  key_dim={D//10} (routing UNCHANGED)')
    print(f'  P1: {STEPS_P1} steps at num_bits={NUM_BITS_P1}')
    print(f'  P2: {STEPS_P2} steps at num_bits={NUM_BITS_P2}')
    print(f'  batch={BATCH}  LR={LR}  seed=42')
    print(f'  device: {DEVICE}')
    if DEVICE.type == 'cuda':
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print()
    print(f'  Architecture: lcx_read [D] -> Linear(D,D) -> C19 -> x zoom_gate -> + hidden')
    print(f'  BN params: {new_bn_params:,}  (old 3-layer: {old_bn_params:,}, '
          f'{new_bn_params/old_bn_params:.1f}x)')
    print()
    print(f'  GPU matchup references:')
    print(f'    noBN       : tail=50.13%  (800 P2 steps)')
    print(f'    BN=204 2L  : tail=50.39%  (800 P2 steps)')
    print(f'    BN=409 2L  : tail=50.64%  (800 P2 steps)')
    print(f'    BN=618 2L  : tail=50.63%  (800 P2 steps)')
    print('=' * 70)

    with open(LIVE_LOG, 'w') as f:
        f.write(f'# probe_bn_single_layer_gpu -- {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'# D={D} single Linear(D,D) + C19 bottleneck\n')

    gc.collect()
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    torch.manual_seed(42)
    random.seed(42)

    # ======================== PHASE 1 ========================
    model = make_model(NUM_BITS_P1).to(DEVICE)
    model.train()
    n_params_p1 = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler('cuda') if DEVICE.type == 'cuda' else None

    def lr_lambda(step):
        if step < WARMUP:
            return step / WARMUP
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    print(f'\n  === PHASE 1: num_bits={NUM_BITS_P1}, {STEPS_P1} steps ===')
    print(f'  params={n_params_p1:,}', flush=True)

    checkpoints = {}
    window_accs = []
    WINDOW = 50
    t_start = time.time()
    global_step = 0

    for step in range(STEPS_P1):
        t0 = time.time()
        if step < 3:
            print(f'  starting step {step}...', end='', flush=True)

        x, y = make_echo_batch_bits(BATCH, SEQ_LEN, BLOCK_SIZE, NUM_BITS_P1)

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
            print(f'\n  OOM at step {step} (Phase 1)!')
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

        if step % 50 == 0 or step == STEPS_P1 - 1:
            checkpoints[global_step] = smooth_acc
            print(f'  step {global_step:4d} | loss {loss.item():.6f} | acc {acc:.4f} | '
                  f'smooth={smooth_acc:.4f} | {elapsed:.2f}s [P1]', flush=True)

        with open(LIVE_LOG, 'a') as lf:
            lf.write(f'step {global_step} | loss {loss.item():.6f} | '
                     f'acc={acc:.4f} smooth={smooth_acc:.4f} '
                     f'RD:{elapsed:.4f} traction={smooth_acc:.4f} '
                     f'shard=0/0 1L-DxD P1\n')

        global_step += 1

    p1_final_acc = smooth_acc if window_accs else 0.5
    print(f'\n  Phase 1 done: smooth_acc={p1_final_acc:.4f}')

    # ======================== PHASE 2 ========================
    print(f'\n  === PHASE 2: expanding num_bits {NUM_BITS_P1} -> {NUM_BITS_P2} ===')
    new_model = expand_model_bits(model, NUM_BITS_P1, NUM_BITS_P2)

    del model, opt, scaler
    gc.collect()
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()

    model = new_model
    model.train()
    n_params_p2 = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler('cuda') if DEVICE.type == 'cuda' else None
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)

    print(f'  params={n_params_p2:,} (was {n_params_p1:,})')
    if DEVICE.type == 'cuda':
        vram_after = torch.cuda.max_memory_allocated() / 1024**3
        print(f'  VRAM after expansion: {vram_after:.2f} GB')

    window_accs = []

    for step in range(STEPS_P2):
        t0 = time.time()
        if step < 3:
            print(f'  starting step {global_step}...', end='', flush=True)

        x, y = make_echo_batch_bits(BATCH, SEQ_LEN, BLOCK_SIZE, NUM_BITS_P2)

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
            print(f'\n  OOM at step {global_step} (Phase 2)!')
            sys.exit(1)

        elapsed = time.time() - t0
        if step < 3:
            print(f' {elapsed:.2f}s', flush=True)

        if elapsed > STEP_TIMEOUT:
            print(f'  TIMEOUT at step {global_step} ({elapsed:.0f}s)')
            sys.exit(1)

        with torch.no_grad():
            pred = (out > 0).float()
            acc = (pred == y).float().mean().item()

        if math.isnan(loss.item()):
            print(f'  NaN at step {global_step}')
            sys.exit(1)

        window_accs.append(acc)
        if len(window_accs) > WINDOW:
            window_accs.pop(0)
        smooth_acc = sum(window_accs) / len(window_accs)

        if step % 50 == 0 or step == STEPS_P2 - 1:
            checkpoints[global_step] = smooth_acc
            print(f'  step {global_step:4d} | loss {loss.item():.6f} | acc {acc:.4f} | '
                  f'smooth={smooth_acc:.4f} | {elapsed:.2f}s [P2]', flush=True)

        with open(LIVE_LOG, 'a') as lf:
            lf.write(f'step {global_step} | loss {loss.item():.6f} | '
                     f'acc={acc:.4f} smooth={smooth_acc:.4f} '
                     f'RD:{elapsed:.4f} traction={smooth_acc:.4f} '
                     f'shard=0/0 1L-DxD P2\n')

        global_step += 1

    total_time = time.time() - t_start
    vram_peak = 0.0
    if DEVICE.type == 'cuda':
        vram_peak = torch.cuda.max_memory_allocated() / 1024**3

    # Tail accuracy (step 800-999 = same window as matchup)
    p2_tail_accs = [a for s, a in checkpoints.items() if s >= STEPS_P1 + 600]
    tail_avg = sum(p2_tail_accs) / len(p2_tail_accs) if p2_tail_accs else 0.5

    # ======================== SUMMARY ========================
    print(f'\n{"="*70}')
    print(f'SUMMARY: SINGLE-LAYER D->D C19 BOTTLENECK')
    print(f'  D={D}  key_dim={D//10}  num_bits_P2={NUM_BITS_P2}')
    print(f'  Architecture: Linear({D},{D}) + C19')
    print(f'  BN params: {new_bn_params:,}  (old 3-layer: {old_bn_params:,})')
    print(f'{"="*70}')
    print(f'\n  Phase 1 final acc: {p1_final_acc:.4f}')
    print(f'  Phase 2 tail (step 800-999): {tail_avg:.4f}')
    print(f'  VRAM peak: {vram_peak:.2f} GB')
    print(f'  Total time: {total_time:.0f}s ({total_time/60:.1f} min)')
    print()
    print(f'  Comparison:')
    print(f'    noBN       tail=50.13%  (no transform)')
    print(f'    BN=204 2L  tail=50.39%  (3-layer squeeze, ratio 1.0:1)')
    print(f'    BN=409 2L  tail=50.64%  (3-layer squeeze, ratio 2.0:1)')
    print(f'    BN=618 2L  tail=50.63%  (3-layer squeeze, ratio 3.1:1)')
    print(f'    1L D->D    tail={tail_avg*100:.2f}%  (THIS PROBE)')
    print()

    delta_vs_none = (tail_avg - 0.5013) * 100
    delta_vs_best = (tail_avg - 0.5064) * 100

    print(f'  vs noBN:          {delta_vs_none:+.2f}%')
    print(f'  vs BN=409 (best): {delta_vs_best:+.2f}%')

    print(f'\n{"="*70}')
    if tail_avg >= 0.5060:
        print(f'  VERDICT: SINGLE_LAYER_WINS -- simplicity matches or beats multi-layer')
        print(f'  The squeeze was unnecessary. D->D + C19 is the production design.')
    elif tail_avg >= 0.5030:
        print(f'  VERDICT: SINGLE_LAYER_OK -- transform works but multi-layer helps')
        print(f'  Single layer proves nonlinearity matters. Multi-layer adds refinement.')
    elif tail_avg > 0.5015:
        print(f'  VERDICT: SINGLE_LAYER_WEAK -- barely above noBN')
        print(f'  D->D linear + C19 is insufficient. Multi-layer structure matters.')
    else:
        print(f'  VERDICT: SINGLE_LAYER_DEAD -- no better than noBN')
        print(f'  The learned transform needs depth, not just width.')
    print(f'{"="*70}')

    print(f'\nDone. Log: {LIVE_LOG}')
    print(f'Total wall time: {total_time:.0f}s ({total_time/60:.1f} min)')
