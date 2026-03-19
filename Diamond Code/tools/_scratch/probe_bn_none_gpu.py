#!/usr/bin/env python3
"""
Probe: No-Bottleneck GPU Test — What happens without the keyhole?
=================================================================
The GPU matchup showed pipe width explains D=2048's choke:
  BN=204 (1.0:1): 50.39%  |  BN=409 (2.0:1): 50.64%  |  BN=618 (3.1:1): 50.63%

This test: remove the bottleneck entirely. Raw LCX read → zoom_gate → add to hidden.
Mini-model (D=128) showed noBN = 85.2% vs BN = 95.5%. Does this hold at D=2048?

+15% more P2 steps (800 → 920) to compensate for ~14% compute savings.
"""

import gc
import math
import os
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Configs (match GPU matchup exactly, except BN=None and +15% P2 steps) ---
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
STEPS_P2       = 920       # +15% over matchup's 800
STEP_TIMEOUT   = 120
RADIUS         = 8
THINK_TICKS    = 1

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DIAMOND_ROOT = r'S:\AI\work\VRAXION_DEV\Diamond Code'
sys.path.insert(0, DIAMOND_ROOT)
LOG_DIR      = os.path.join(DIAMOND_ROOT, 'logs', 'probe')
LIVE_LOG     = os.path.join(LOG_DIR, 'probe_bn_none_gpu_live.log')
os.makedirs(LOG_DIR, exist_ok=True)

from swarm_model import SwarmByteRingModel


def make_model(num_bits):
    """Create D=2048 model with NO bottleneck."""
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
        lcx_key_dim=D // 10,    # 204 — FIXED routing
        lcx_top_k=2,
        num_pointers=1,
    )
    # Kill the bottleneck — _lcx_bottleneck() returns x unchanged when None
    model.lcx_bn_layers = None
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
    """Expand num_bits: create new model (no BN), copy matching weights."""
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


if __name__ == '__main__':
    # BN params that would exist at D/10 (what we're saving)
    bn_dim_default = D // 10  # 204
    bn_params_saved = D * bn_dim_default + bn_dim_default + \
                      bn_dim_default * bn_dim_default + bn_dim_default + \
                      bn_dim_default * D + D
    bn_pct = bn_params_saved / 5_545_248 * 100  # vs P1 total params

    print('=' * 70)
    print('PROBE: NO-BOTTLENECK GPU TEST — Raw LCX add, no keyhole')
    print('=' * 70)
    print(f'  D={D}  depth={DEPTH}  key_dim={D//10} (routing UNCHANGED)')
    print(f'  P1: {STEPS_P1} steps at num_bits={NUM_BITS_P1}')
    print(f'  P2: {STEPS_P2} steps at num_bits={NUM_BITS_P2} (+15% over matchup)')
    print(f'  batch={BATCH}  LR={LR}  seed=42')
    print(f'  device: {DEVICE}')
    if DEVICE.type == 'cuda':
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'\n  Bottleneck: REMOVED (lcx_bn_layers = None)')
    print(f'  BN params saved: {bn_params_saved:,} ({bn_pct:.1f}% of model)')
    print(f'\n  GPU matchup references:')
    print(f'    BN=204 (1.0:1): tail=50.39%  (800 P2 steps)')
    print(f'    BN=409 (2.0:1): tail=50.64%  (800 P2 steps)')
    print(f'    BN=618 (3.1:1): tail=50.63%  (800 P2 steps)')
    print('=' * 70)

    with open(LIVE_LOG, 'w') as f:
        f.write(f'# probe_bn_none_gpu -- {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'# D={D} NO BOTTLENECK -- raw LCX add\n')

    gc.collect()
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    torch.manual_seed(42)
    random.seed(42)

    # === PHASE 1 ===
    model = make_model(NUM_BITS_P1).to(DEVICE)
    model.train()
    assert model.lcx_bn_layers is None, "BN should be None!"
    n_params_p1 = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler('cuda') if DEVICE.type == 'cuda' else None

    def lr_lambda(step):
        if step < WARMUP:
            return step / WARMUP
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    print(f'\n  === PHASE 1: num_bits={NUM_BITS_P1}, {STEPS_P1} steps ===')
    print(f'  params={n_params_p1:,} (saved {bn_params_saved:,} by removing BN)', flush=True)

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
                     f'shard=0/0 BN=NONE P1\n')

        global_step += 1

    p1_final_acc = smooth_acc if window_accs else 0.5
    print(f'\n  Phase 1 done: smooth_acc={p1_final_acc:.4f}')

    # === PHASE 2 ===
    print(f'\n  === PHASE 2: expanding num_bits {NUM_BITS_P1} -> {NUM_BITS_P2} ===')
    new_model = expand_model_bits(model, NUM_BITS_P1, NUM_BITS_P2)
    assert new_model.lcx_bn_layers is None, "BN should be None after expansion!"

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

    print(f'  params={n_params_p2:,}')
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

        if step % 50 == 0 or step == STEPS_P2 - 1:
            print(f'  step {global_step:4d} | loss {loss.item():.6f} | acc {acc:.4f} | '
                  f'smooth={smooth_acc:.4f} | {elapsed:.2f}s [P2]', flush=True)

        with open(LIVE_LOG, 'a') as lf:
            lf.write(f'step {global_step} | loss {loss.item():.6f} | '
                     f'acc={acc:.4f} smooth={smooth_acc:.4f} '
                     f'RD:{elapsed:.4f} traction={smooth_acc:.4f} '
                     f'shard=0/0 BN=NONE P2\n')

        global_step += 1

    total_time = time.time() - t_start
    vram_peak = 0.0
    if DEVICE.type == 'cuda':
        vram_peak = torch.cuda.max_memory_allocated() / 1024**3

    # Tail accuracy — two windows for fair comparison
    # "Fair tail": steps 800-999 (same window as matchup)
    fair_tail_accs = [a for s, a in checkpoints.items()
                      if 800 <= s < 1000]
    fair_tail = sum(fair_tail_accs) / len(fair_tail_accs) if fair_tail_accs else 0.5

    # "Extended tail": steps 800-1119 (using all extra steps)
    ext_tail_accs = [a for s, a in checkpoints.items() if s >= 800]
    ext_tail = sum(ext_tail_accs) / len(ext_tail_accs) if ext_tail_accs else 0.5

    # ==================== SUMMARY ====================
    print(f'\n{"="*70}')
    print(f'SUMMARY: NO-BOTTLENECK GPU TEST')
    print(f'  D={D}  key_dim={D//10}  num_bits_P2={NUM_BITS_P2}')
    print(f'  Bottleneck: REMOVED  BN params saved: {bn_params_saved:,}')
    print(f'{"="*70}')
    print(f'\n  Phase 1 final acc: {p1_final_acc:.4f}')
    print(f'  Phase 2 fair tail (step 800-999):  {fair_tail:.4f}')
    print(f'  Phase 2 ext tail  (step 800-1119): {ext_tail:.4f}')
    print(f'  VRAM peak: {vram_peak:.2f} GB')
    print(f'  Total time: {total_time:.0f}s ({total_time/60:.1f} min)')

    print(f'\n  Comparison:')
    print(f'    BN=204  tail=50.39%  (800 P2 steps, matchup control)')
    print(f'    BN=409  tail=50.64%  (800 P2 steps)')
    print(f'    BN=618  tail=50.63%  (800 P2 steps)')
    print(f'    NO BN   tail={fair_tail*100:.2f}%  (800 P2 steps, fair window)')
    print(f'    NO BN   tail={ext_tail*100:.2f}%  (920 P2 steps, extended)')

    gap_vs_control = (fair_tail - 0.5039) * 100
    gap_vs_best = (fair_tail - 0.5064) * 100

    print(f'\n  vs BN=204 control: {gap_vs_control:+.2f}%')
    print(f'  vs BN=409 best:    {gap_vs_best:+.2f}%')

    print(f'\n{"="*70}')
    if fair_tail >= 0.506:
        print('  VERDICT: BN_USELESS — no bottleneck matches BN performance at this scale')
    elif fair_tail >= 0.503:
        print('  VERDICT: BN_HELPS_SLIGHTLY — bottleneck adds ~0.1-0.3%, not critical')
    elif fair_tail >= 0.500:
        print('  VERDICT: BN_MATTERS — learned transform helps integration at scale')
    else:
        print('  VERDICT: BN_ESSENTIAL — model cannot integrate LCX without bottleneck')
    print('=' * 70)

    print(f'\nDone. Log: {LIVE_LOG}')
    print(f'Total wall time: {total_time:.0f}s ({total_time/60:.1f} min)')
