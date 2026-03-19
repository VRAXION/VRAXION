#!/usr/bin/env python3
"""
Probe: BN Width GPU Matchup — Does a wider pipe fix D=2048?
============================================================
The D-ablation showed D=2048 choked at 50.39% with BN=204 (ratio 1.0:1).
D=4096 (BN=409) and D=6180 (BN=618) reached ~50.6%.

Hypothesis: "D=2048 failed because its pipe was too narrow."
Test: Give D=2048 a wider pipe (409, 618) via monkey-patch.

If D=2048 + BN=618 ≈ 50.4%  → BRAIN is the limiter (pipe doesn't matter)
If D=2048 + BN=618 ≈ 50.6%  → PIPE was the limiter (Gemini is right)

Same two-phase design as D-ablation for direct comparison.
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

# --- Configs (match D-ablation exactly) ---
D              = 2048
DEPTH          = 2
SEQ_LEN        = 32
BLOCK_SIZE     = 16
BATCH          = 10        # same as D-ablation
LR             = 1e-4      # same as D-ablation
WARMUP         = 50
NUM_BITS_P1    = 8
NUM_BITS_P2    = 200       # same as D-ablation
STEPS_P1       = 200
STEPS_P2       = 800       # same as D-ablation
STEP_TIMEOUT   = 120
RADIUS         = 8
THINK_TICKS    = 1

# BN configs to test: default, D=4096's pipe, D=6180's pipe
BN_CONFIGS = [
    (204, 'default (D/10)'),
    (409, 'match D=4096 pipe'),
    (618, 'match D=6180 pipe'),
]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DIAMOND_ROOT = r'S:\AI\work\VRAXION_DEV\Diamond Code'
sys.path.insert(0, DIAMOND_ROOT)
LOG_DIR      = os.path.join(DIAMOND_ROOT, 'logs', 'probe')
LIVE_LOG     = os.path.join(LOG_DIR, 'probe_bn_gpu_matchup_live.log')
os.makedirs(LOG_DIR, exist_ok=True)

from swarm_model import SwarmByteRingModel


def override_bottleneck(model, embed_dim, new_bn_dim):
    """Monkey-patch lcx_bn_layers to a new bottleneck width."""
    model.lcx_bn_layers = nn.ModuleList([
        nn.Linear(embed_dim, new_bn_dim),
        nn.Linear(new_bn_dim, new_bn_dim),
        nn.Linear(new_bn_dim, embed_dim),
    ])
    for layer in model.lcx_bn_layers:
        nn.init.orthogonal_(layer.weight)
        nn.init.zeros_(layer.bias)
    model.lcx_bn_layers.to(next(model.parameters()).device)


def make_model(num_bits, bn_dim):
    """Create D=2048 model, then monkey-patch to bn_dim."""
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
        lcx_key_dim=D // 10,    # 204 — FIXED (same routing for all)
        lcx_top_k=2,
        num_pointers=1,
    )
    if bn_dim != D // 10:
        override_bottleneck(model, D, bn_dim)
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


def expand_model_bits(model, old_bits, new_bits, bn_dim):
    """Expand num_bits: create new model, copy matching weights."""
    new_model = make_model(new_bits, bn_dim).to(DEVICE)
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


def run_one(bn_dim, bn_label, log_file):
    """Two-phase probe for a single BN config."""
    bn_ratio = bn_dim / NUM_BITS_P2
    bn_params = D * bn_dim + bn_dim + bn_dim * bn_dim + bn_dim + bn_dim * D + D

    print(f'\n{"="*70}')
    print(f'PROBE: D={D}  BN={bn_dim} ({bn_label})')
    print(f'  BN ratio: {bn_dim} dims carrying {NUM_BITS_P2} bits '
          f'(ratio {bn_ratio:.1f}:1)')
    print(f'  BN params: {bn_params:,}')
    print(f'  Phase 1: {STEPS_P1} steps at num_bits={NUM_BITS_P1}')
    print(f'  Phase 2: {STEPS_P2} steps at num_bits={NUM_BITS_P2}')
    print(f'  key_dim={D//10} (FIXED — routing unchanged)')
    print(f'  device: {DEVICE}')
    if DEVICE.type == 'cuda':
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'{"="*70}')

    gc.collect()
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Same seed as D-ablation for direct comparison
    torch.manual_seed(42)
    random.seed(42)

    # Phase 1
    model = make_model(NUM_BITS_P1, bn_dim).to(DEVICE)
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
            del model, opt, scaler
            gc.collect()
            torch.cuda.empty_cache()
            return {'bn_dim': bn_dim, 'label': bn_label, 'status': 'OOM',
                    'phase': 1, 'step': step}

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

        if step % 50 == 0 or step == STEPS_P1 - 1:
            checkpoints[global_step] = smooth_acc
            print(f'  step {global_step:4d} | loss {loss.item():.6f} | acc {acc:.4f} | '
                  f'smooth={smooth_acc:.4f} | {elapsed:.2f}s [P1]', flush=True)

        with open(log_file, 'a') as lf:
            lf.write(f'step {global_step} | loss {loss.item():.6f} | '
                     f'acc={acc:.4f} smooth={smooth_acc:.4f} '
                     f'RD:{elapsed:.4f} traction={smooth_acc:.4f} '
                     f'shard=0/0 BN={bn_dim} P1\n')

        global_step += 1

    p1_final_acc = smooth_acc if window_accs else 0.5
    print(f'\n  Phase 1 done: smooth_acc={p1_final_acc:.4f}')

    # --- Expand to Phase 2 ---
    print(f'\n  === PHASE 2: expanding num_bits {NUM_BITS_P1} -> {NUM_BITS_P2} ===')
    new_model = expand_model_bits(model, NUM_BITS_P1, NUM_BITS_P2, bn_dim)

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
            del model, opt, scaler
            gc.collect()
            torch.cuda.empty_cache()
            return {'bn_dim': bn_dim, 'label': bn_label, 'status': 'OOM',
                    'phase': 2, 'step': global_step}

        elapsed = time.time() - t0
        if step < 3:
            print(f' {elapsed:.2f}s', flush=True)

        if elapsed > STEP_TIMEOUT:
            print(f'  TIMEOUT at step {global_step} ({elapsed:.0f}s)')
            break

        with torch.no_grad():
            pred = (out > 0).float()
            acc = (pred == y).float().mean().item()

        if math.isnan(loss.item()):
            print(f'  NaN at step {global_step}')
            break

        window_accs.append(acc)
        if len(window_accs) > WINDOW:
            window_accs.pop(0)
        smooth_acc = sum(window_accs) / len(window_accs)

        if step % 50 == 0 or step == STEPS_P2 - 1:
            checkpoints[global_step] = smooth_acc

        if step % 50 == 0 or step == STEPS_P2 - 1:
            print(f'  step {global_step:4d} | loss {loss.item():.6f} | acc {acc:.4f} | '
                  f'smooth={smooth_acc:.4f} | {elapsed:.2f}s [P2]', flush=True)

        with open(log_file, 'a') as lf:
            lf.write(f'step {global_step} | loss {loss.item():.6f} | '
                     f'acc={acc:.4f} smooth={smooth_acc:.4f} '
                     f'RD:{elapsed:.4f} traction={smooth_acc:.4f} '
                     f'shard=0/0 BN={bn_dim} P2\n')

        global_step += 1

    total_time = time.time() - t_start
    vram_peak = 0.0
    if DEVICE.type == 'cuda':
        vram_peak = torch.cuda.max_memory_allocated() / 1024**3

    # Tail accuracy (step 800-999)
    p2_tail_accs = [a for s, a in checkpoints.items() if s >= STEPS_P1 + 600]
    tail_avg = sum(p2_tail_accs) / len(p2_tail_accs) if p2_tail_accs else 0.5

    print(f'\n  Phase 2 tail (step 800-999): {tail_avg:.4f}')
    print(f'  VRAM peak: {vram_peak:.2f} GB')
    print(f'  Total time: {total_time:.0f}s ({total_time/60:.1f} min)')

    del model, opt, scaler
    gc.collect()
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()

    return {
        'bn_dim': bn_dim,
        'label': bn_label,
        'status': 'OK',
        'params_p2': n_params_p2,
        'bn_params': bn_params,
        'bn_ratio': bn_ratio,
        'vram_peak': vram_peak,
        'p1_acc': p1_final_acc,
        'p2_tail': tail_avg,
        'time': total_time,
        'checkpoints': dict(sorted(checkpoints.items())),
    }


if __name__ == '__main__':
    print('=' * 70)
    print('PROBE: BN WIDTH GPU MATCHUP — Does a wider pipe fix D=2048?')
    print('=' * 70)
    print(f'  D={D}  depth={DEPTH}  key_dim={D//10} (routing UNCHANGED)')
    print(f'  P1: {STEPS_P1} steps at num_bits={NUM_BITS_P1}')
    print(f'  P2: {STEPS_P2} steps at num_bits={NUM_BITS_P2}')
    print(f'  batch={BATCH}  LR={LR}  seed=42')
    print(f'  device: {DEVICE}')
    if DEVICE.type == 'cuda':
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print()
    print(f'  D-ablation reference: D=2048 BN=204 tail=50.39%')
    print(f'  D-ablation reference: D=4096 BN=409 tail=50.63%')
    print(f'  D-ablation reference: D=6180 BN=618 tail=50.60%')
    print()
    print(f'  Configs:')
    for bn_dim, label in BN_CONFIGS:
        ratio = bn_dim / NUM_BITS_P2
        print(f'    BN={bn_dim:>4}  ratio={ratio:.1f}:1  ({label})')
    print('=' * 70)

    with open(LIVE_LOG, 'w') as f:
        f.write(f'# probe_bn_gpu_matchup -- {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'# D={D} key_dim={D//10} sweep BN pipe width\n')
        f.write(f'# Hypothesis: wider pipe fixes D=2048 choke?\n')

    results = []
    for bn_dim, label in BN_CONFIGS:
        with open(LIVE_LOG, 'a') as f:
            f.write(f'\n# === BN={bn_dim} ({label}) ===\n')

        r = run_one(bn_dim, label, LIVE_LOG)
        results.append(r)
        if r['status'] == 'OK':
            print(f'\n  -> BN={bn_dim}: tail={r["p2_tail"]:.4f} '
                  f'VRAM={r["vram_peak"]:.2f}G ({r["time"]:.0f}s)')

    # ==================== SUMMARY ====================
    print('\n' + '=' * 70)
    print('SUMMARY: BN WIDTH GPU MATCHUP')
    print(f'  D={D}  key_dim={D//10}  num_bits_P2={NUM_BITS_P2}')
    print('=' * 70)

    print(f'\n  {"BN":>5} {"Ratio":>7} {"BN Params":>10} {"P1 Acc":>8} '
          f'{"P2 Tail":>8} {"VRAM":>6} {"Label"}')
    print(f'  {"-"*5} {"-"*7} {"-"*10} {"-"*8} {"-"*8} {"-"*6} {"-"*20}')

    for r in results:
        if r['status'] == 'OOM':
            print(f'  {r["bn_dim"]:>5} {"--":>7} {"--":>10} {"--":>8} '
                  f'{"OOM":>8} {"--":>6} {r["label"]}')
        else:
            print(f'  {r["bn_dim"]:>5} {r["bn_ratio"]:>6.1f}:1 '
                  f'{r["bn_params"]:>10,} {r["p1_acc"]:>7.4f} '
                  f'{r["p2_tail"]:>7.4f} {r["vram_peak"]:>5.2f}G '
                  f'{r["label"]}')

    # Compare against D-ablation references
    ok_results = [r for r in results if r['status'] == 'OK']
    if ok_results:
        default_r = next((r for r in ok_results if r['bn_dim'] == 204), None)
        d_ablation_ref = 0.5039  # D=2048 from D-ablation

        print(f'\n  D-ablation D=2048 reference: {d_ablation_ref:.4f}')
        if default_r:
            print(f'  This probe  BN=204 control: {default_r["p2_tail"]:.4f} '
                  f'(delta: {(default_r["p2_tail"] - d_ablation_ref)*100:+.2f}%)')

        best = max(ok_results, key=lambda r: r['p2_tail'])
        worst = min(ok_results, key=lambda r: r['p2_tail'])
        spread = best['p2_tail'] - worst['p2_tail']

        print(f'\n  Best:   BN={best["bn_dim"]} tail={best["p2_tail"]:.4f}')
        print(f'  Worst:  BN={worst["bn_dim"]} tail={worst["p2_tail"]:.4f}')
        print(f'  Spread: {spread*100:.2f}%')

        print('\n' + '=' * 70)
        if spread < 0.005:
            print('  VERDICT: PIPE_IRRELEVANT — wider pipe does NOT fix D=2048.')
            print('  D=2048 choked from SMALL BRAIN (6.3M params), not narrow pipe.')
            print('  The bottleneck integration width is not the limiter.')
        elif best['bn_dim'] > 204 and best['p2_tail'] > d_ablation_ref + 0.003:
            print(f'  VERDICT: PIPE_MATTERS — wider pipe helps D=2048!')
            print(f'  BN={best["bn_dim"]} gained +{(best["p2_tail"]-worst["p2_tail"])*100:.2f}%.')
            print('  Gemini was right: pipe width IS causal at production scale.')
        else:
            print('  VERDICT: INCONCLUSIVE — small or mixed signal.')
        print('=' * 70)

    with open(LIVE_LOG, 'a') as f:
        f.write(f'\n# === SUMMARY ===\n')
        for r in results:
            if r['status'] == 'OK':
                f.write(f'# BN={r["bn_dim"]:>4} ratio={r["bn_ratio"]:.1f}:1 '
                        f'p2_tail={r["p2_tail"]:.4f} ({r["label"]})\n')

    print(f'\nDone. Log: {LIVE_LOG}')
    print(f'Total wall time: {sum(r["time"] for r in ok_results):.0f}s '
          f'({sum(r["time"] for r in ok_results)/60:.1f} min)')
