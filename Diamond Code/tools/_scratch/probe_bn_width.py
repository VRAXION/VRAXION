#!/usr/bin/env python3
"""
Probe: Bottleneck Width Isolation (CPU)
=======================================
Isolates the LCX read bottleneck width as a SINGLE variable.

The D ablation showed D doesn't matter once BN ratio > 2:1. But D moves
5 things at once. This probe fixes D=128 and ONLY varies the bottleneck
integration pipe width via monkey-patching model.lcx_bn_layers.

Key_dim stays at D//10=12 for ALL configs (address quality constant).

Sweep BN = {2, 4, 8, 12, 16, 25, 32, 50, 64, 96, 128}
Two-phase: 200 steps P1 (num_bits=8) + 300 steps P2 (num_bits=25)
Two seeds: [42, 137]
CPU only.
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

# --- Configs ---
D              = 128
DEPTH          = 4
SEQ_LEN        = 32
BLOCK_SIZE     = 4
BATCH          = 16
LR             = 1e-3
WARMUP         = 30
NUM_BITS_P1    = 8
NUM_BITS_P2    = 25
STEPS_P1       = 200
STEPS_P2       = 300
STEP_TIMEOUT   = 30
RADIUS         = 8
THINK_TICKS    = 1
LCX_SLOTS      = 100
KEY_DIM        = D // 10   # 12 — fixed for ALL configs
TOP_K          = 2
WINDOW         = 50

BN_VALUES      = [2, 4, 8, 12, 16, 25, 32, 50, 64, 96, 128]
SEEDS          = [42, 137]

DEVICE = torch.device('cpu')  # CPU only — GPU runs production stress test

DIAMOND_ROOT = r'S:\AI\work\VRAXION_DEV\Diamond Code'
sys.path.insert(0, DIAMOND_ROOT)
LOG_DIR      = os.path.join(DIAMOND_ROOT, 'logs', 'probe')
LIVE_LOG     = os.path.join(LOG_DIR, 'probe_bn_width_live.log')
os.makedirs(LOG_DIR, exist_ok=True)

from swarm_model import SwarmByteRingModel


def override_bottleneck(model, embed_dim, new_bn_dim):
    """Monkey-patch lcx_bn_layers to a new bottleneck width.
    Zero production code changes — just replaces the ModuleList."""
    model.lcx_bn_layers = nn.ModuleList([
        nn.Linear(embed_dim, new_bn_dim),        # down: D → BN
        nn.Linear(new_bn_dim, new_bn_dim),        # mid:  BN → BN
        nn.Linear(new_bn_dim, embed_dim),          # up:   BN → D
    ])
    # Lever 7: orthogonal init (locked)
    for layer in model.lcx_bn_layers:
        nn.init.orthogonal_(layer.weight)
        nn.init.zeros_(layer.bias)
    # Move to same device as model
    model.lcx_bn_layers.to(next(model.parameters()).device)


def make_model(num_bits, bn_dim):
    """Create model with standard D//10 bottleneck, then monkey-patch to bn_dim."""
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
        lcx_level_slots=[LCX_SLOTS],
        lcx_key_dim=KEY_DIM,
        lcx_top_k=TOP_K,
        num_pointers=1,
    )
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
    """Expand num_bits: create new model, copy matching weights, re-patch BN."""
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
    print(f'    Expansion: {copied} copied, {expanded} expanded, {skipped} new')
    return new_model


def run_one(bn_dim, seed, log_file):
    """Two-phase probe for a single BN width + seed combo."""
    total_steps = STEPS_P1 + STEPS_P2
    bn_ratio = bn_dim / NUM_BITS_P2
    bn_params = D * bn_dim + bn_dim + bn_dim * bn_dim + bn_dim + bn_dim * D + D  # 3 linear layers

    print(f'\n  --- BN={bn_dim:>3d}  seed={seed}  ratio={bn_ratio:.2f}:1  '
          f'BN_params={bn_params:,} ---')

    torch.manual_seed(seed)
    random.seed(seed)
    gc.collect()

    # Phase 1: narrow bits
    model = make_model(NUM_BITS_P1, bn_dim).to(DEVICE)
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    def lr_lambda(step):
        if step < WARMUP:
            return step / WARMUP
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    checkpoints = {}
    window_accs = []
    t_start = time.time()
    global_step = 0

    # --- Phase 1 ---
    for step in range(STEPS_P1):
        t0 = time.time()
        if step == 0:
            print(f'    starting step 0...', end='', flush=True)

        x, y = make_echo_batch_bits(BATCH, SEQ_LEN, BLOCK_SIZE, NUM_BITS_P1)

        opt.zero_grad()
        out = model(x)
        if isinstance(out, tuple):
            out = out[0]
        loss = F.binary_cross_entropy_with_logits(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()

        elapsed = time.time() - t0
        if step == 0:
            print(f' {elapsed:.2f}s', flush=True)

        if elapsed > STEP_TIMEOUT:
            print(f'    TIMEOUT at step {step} ({elapsed:.0f}s)')
            break

        with torch.no_grad():
            pred = (out > 0).float()
            acc = (pred == y).float().mean().item()

        if math.isnan(loss.item()):
            print(f'    NaN at step {step}')
            break

        window_accs.append(acc)
        if len(window_accs) > WINDOW:
            window_accs.pop(0)
        smooth_acc = sum(window_accs) / len(window_accs)

        if step % 50 == 0 or step == STEPS_P1 - 1:
            checkpoints[global_step] = smooth_acc

        if step == STEPS_P1 - 1:
            print(f'    P1 done: step {global_step} smooth={smooth_acc:.4f} '
                  f'({elapsed:.2f}s/step)', flush=True)

        with open(log_file, 'a') as lf:
            lf.write(f'step {global_step} | loss {loss.item():.6f} | '
                     f'acc={acc:.4f} smooth={smooth_acc:.4f} '
                     f'RD:{elapsed:.4f} traction={smooth_acc:.4f} '
                     f'shard=0/0 BN={bn_dim} seed={seed} P1\n')

        global_step += 1

    p1_final_acc = smooth_acc if window_accs else 0.5

    # --- Expand to Phase 2 ---
    new_model = expand_model_bits(model, NUM_BITS_P1, NUM_BITS_P2, bn_dim)
    del model, opt
    gc.collect()

    model = new_model
    model.train()
    n_params_p2 = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)

    window_accs = []

    # --- Phase 2 ---
    for step in range(STEPS_P2):
        t0 = time.time()
        if step == 0:
            print(f'    P2 starting step {global_step}...', end='', flush=True)

        x, y = make_echo_batch_bits(BATCH, SEQ_LEN, BLOCK_SIZE, NUM_BITS_P2)

        opt.zero_grad()
        out = model(x)
        if isinstance(out, tuple):
            out = out[0]
        loss = F.binary_cross_entropy_with_logits(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()

        elapsed = time.time() - t0
        if step == 0:
            print(f' {elapsed:.2f}s', flush=True)

        if elapsed > STEP_TIMEOUT:
            print(f'    TIMEOUT at step {global_step} ({elapsed:.0f}s)')
            break

        with torch.no_grad():
            pred = (out > 0).float()
            acc = (pred == y).float().mean().item()

        if math.isnan(loss.item()):
            print(f'    NaN at step {global_step}')
            break

        window_accs.append(acc)
        if len(window_accs) > WINDOW:
            window_accs.pop(0)
        smooth_acc = sum(window_accs) / len(window_accs)

        if step % 50 == 0 or step == STEPS_P2 - 1:
            checkpoints[global_step] = smooth_acc

        if step % 100 == 0 or step == STEPS_P2 - 1:
            print(f'    step {global_step:4d} | loss {loss.item():.6f} | '
                  f'smooth={smooth_acc:.4f} | {elapsed:.2f}s [P2]', flush=True)

        with open(log_file, 'a') as lf:
            lf.write(f'step {global_step} | loss {loss.item():.6f} | '
                     f'acc={acc:.4f} smooth={smooth_acc:.4f} '
                     f'RD:{elapsed:.4f} traction={smooth_acc:.4f} '
                     f'shard=0/0 BN={bn_dim} seed={seed} P2\n')

        global_step += 1

    total_time = time.time() - t_start

    # Extract Phase 2 tail accuracy (last 100 steps)
    p2_accs = [a for s, a in checkpoints.items() if s >= STEPS_P1 + 200]
    tail_avg = sum(p2_accs) / len(p2_accs) if p2_accs else 0.5

    # Cleanup
    del model, opt
    gc.collect()

    return {
        'bn_dim': bn_dim,
        'seed': seed,
        'params': n_params_p2,
        'bn_params': bn_params,
        'bn_ratio': bn_ratio,
        'p1_acc': p1_final_acc,
        'p2_tail': tail_avg,
        'time': total_time,
        'checkpoints': dict(sorted(checkpoints.items())),
    }


if __name__ == '__main__':
    print('=' * 70)
    print('PROBE: BOTTLENECK WIDTH ISOLATION (CPU)')
    print('=' * 70)
    print(f'  D={D}  depth={DEPTH}  seq_len={SEQ_LEN}  block={BLOCK_SIZE}')
    print(f'  key_dim={KEY_DIM} (FIXED — address quality constant)')
    print(f'  LCX: {LCX_SLOTS} slots, hash, top_k={TOP_K}, tt={THINK_TICKS}')
    print(f'  P1: {STEPS_P1} steps at num_bits={NUM_BITS_P1}')
    print(f'  P2: {STEPS_P2} steps at num_bits={NUM_BITS_P2}')
    print(f'  LR={LR}  batch={BATCH}  seeds={SEEDS}')
    print(f'  device: {DEVICE}')
    print()
    print(f'  BN sweep at num_bits={NUM_BITS_P2}:')
    print(f'  {"BN":>5} {"Ratio":>7} {"BN params":>10} {"Stress":>12}')
    print(f'  {"-"*5} {"-"*7} {"-"*10} {"-"*12}')
    for bn in BN_VALUES:
        ratio = bn / NUM_BITS_P2
        bn_p = D * bn + bn + bn * bn + bn + bn * D + D
        if ratio < 0.2:
            stress = 'DEAD?'
        elif ratio < 0.5:
            stress = 'CHOKED?'
        elif ratio < 1.0:
            stress = 'STRESSED'
        elif ratio < 2.0:
            stress = 'OK'
        elif ratio < 3.0:
            stress = 'WIDE'
        else:
            stress = 'VERY WIDE'
        print(f'  {bn:>5} {ratio:>6.2f}:1 {bn_p:>10,} {stress:>12}')
    print()
    print(f'  Total runs: {len(BN_VALUES)} × {len(SEEDS)} seeds = '
          f'{len(BN_VALUES) * len(SEEDS)}')
    print('=' * 70)

    with open(LIVE_LOG, 'w') as f:
        f.write(f'# probe_bn_width -- {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'# D={D} key_dim={KEY_DIM} sweep BN={BN_VALUES}\n')

    all_results = []

    for bn_dim in BN_VALUES:
        with open(LIVE_LOG, 'a') as f:
            f.write(f'\n# === BN={bn_dim} ===\n')

        seed_results = []
        for seed in SEEDS:
            r = run_one(bn_dim, seed, LIVE_LOG)
            seed_results.append(r)
            all_results.append(r)
            print(f'    -> BN={bn_dim} seed={seed}: '
                  f'p1={r["p1_acc"]:.4f} p2_tail={r["p2_tail"]:.4f} '
                  f'({r["time"]:.0f}s)')

        # Seed average
        avg_tail = sum(r['p2_tail'] for r in seed_results) / len(seed_results)
        avg_p1 = sum(r['p1_acc'] for r in seed_results) / len(seed_results)
        print(f'  BN={bn_dim:>3d} MEAN: p1={avg_p1:.4f} p2_tail={avg_tail:.4f}')

    # ==================== SUMMARY ====================
    print('\n' + '=' * 70)
    print(f'SUMMARY: BOTTLENECK WIDTH ISOLATION')
    print(f'  D={D}  key_dim={KEY_DIM}  num_bits_P2={NUM_BITS_P2}')
    print('=' * 70)

    # Compute per-BN averages across seeds
    bn_avgs = {}
    for bn in BN_VALUES:
        bn_runs = [r for r in all_results if r['bn_dim'] == bn]
        avg_tail = sum(r['p2_tail'] for r in bn_runs) / len(bn_runs)
        avg_p1 = sum(r['p1_acc'] for r in bn_runs) / len(bn_runs)
        avg_time = sum(r['time'] for r in bn_runs) / len(bn_runs)
        bn_avgs[bn] = {
            'tail': avg_tail, 'p1': avg_p1, 'time': avg_time,
            'ratio': bn / NUM_BITS_P2,
            'params': bn_runs[0]['params'],
            'bn_params': bn_runs[0]['bn_params'],
            'runs': bn_runs,
        }

    print(f'\n  {"BN":>5} {"Ratio":>7} {"BN Params":>10} {"P1 Acc":>8} '
          f'{"P2 Tail":>8} {"Spread":>8} {"Time":>6}')
    print(f'  {"-"*5} {"-"*7} {"-"*10} {"-"*8} {"-"*8} {"-"*8} {"-"*6}')

    for bn in BN_VALUES:
        a = bn_avgs[bn]
        # Spread across seeds
        tails = [r['p2_tail'] for r in a['runs']]
        spread = max(tails) - min(tails) if len(tails) > 1 else 0
        print(f'  {bn:>5} {a["ratio"]:>6.2f}:1 {a["bn_params"]:>10,} '
              f'{a["p1"]:>7.4f} {a["tail"]:>7.4f} '
              f'±{spread*100:.2f}% {a["time"]:>5.0f}s')

    # Find best/worst across BN averages
    best_bn = max(bn_avgs, key=lambda b: bn_avgs[b]['tail'])
    worst_bn = min(bn_avgs, key=lambda b: bn_avgs[b]['tail'])
    spread = bn_avgs[best_bn]['tail'] - bn_avgs[worst_bn]['tail']

    print(f'\n  Best:   BN={best_bn} (ratio {bn_avgs[best_bn]["ratio"]:.2f}:1) '
          f'tail={bn_avgs[best_bn]["tail"]:.4f}')
    print(f'  Worst:  BN={worst_bn} (ratio {bn_avgs[worst_bn]["ratio"]:.2f}:1) '
          f'tail={bn_avgs[worst_bn]["tail"]:.4f}')
    print(f'  Spread: {spread*100:.2f}%')

    # Detect cliff / gradient / flat
    # Check if there's a sharp drop below some threshold
    sorted_bns = sorted(BN_VALUES)
    tails = [bn_avgs[bn]['tail'] for bn in sorted_bns]

    # Find biggest drop between consecutive BN values
    max_drop = 0
    drop_at = None
    for i in range(len(sorted_bns) - 1):
        drop = tails[i + 1] - tails[i]  # positive = improvement going up
        neg_drop = tails[i] - tails[i + 1]  # positive = degradation going down
        if i > 0:  # compare going from larger to smaller
            actual_drop = bn_avgs[sorted_bns[i]]['tail'] - bn_avgs[sorted_bns[i-1]]['tail']
            if abs(actual_drop) > abs(max_drop):
                max_drop = actual_drop
                drop_at = (sorted_bns[i-1], sorted_bns[i])

    # Verdict
    print('\n' + '=' * 70)
    standard_bn = D // 10  # 12
    standard_tail = bn_avgs.get(standard_bn, {}).get('tail', 0)

    if spread < 0.015:
        verdict = 'BN_CLEARED'
        print(f'  VERDICT: {verdict} — pipe width does NOT matter at fixed D={D}.')
        print(f'  Spread {spread*100:.2f}% across BN={BN_VALUES[0]}..{BN_VALUES[-1]}')
        print(f'  D=2048 in D-ablation failed from SMALL BRAIN, not narrow pipe.')
    elif bn_avgs[sorted_bns[0]]['tail'] < 0.49 and bn_avgs[sorted_bns[-1]]['tail'] > 0.51:
        # Check if there's a sharp cliff
        cliff_found = False
        for i in range(len(sorted_bns) - 1):
            gap = bn_avgs[sorted_bns[i+1]]['tail'] - bn_avgs[sorted_bns[i]]['tail']
            if gap > spread * 0.5:  # single step accounts for >50% of total spread
                verdict = 'BN_CLIFF'
                cliff_at = sorted_bns[i]
                cliff_found = True
                print(f'  VERDICT: {verdict} — sharp degradation below BN={cliff_at}')
                print(f'  Ratio {bn_avgs[cliff_at]["ratio"]:.2f}:1 is the choke point.')
                print(f'  Production risk at num_bits=618: needs BN>{cliff_at} '
                      f'(currently 618, safe).')
                break
        if not cliff_found:
            verdict = 'BN_GRADIENT'
            print(f'  VERDICT: {verdict} — gradual improvement with wider pipe.')
            print(f'  No sharp cliff, but wider is measurably better.')
    elif bn_avgs[best_bn]['tail'] > bn_avgs[sorted_bns[-1]]['tail'] + 0.005:
        # Best is NOT the widest — compression helps
        verdict = 'BN_SWEET_SPOT'
        print(f'  VERDICT: {verdict} — optimal BN={best_bn} '
              f'(ratio {bn_avgs[best_bn]["ratio"]:.2f}:1)')
        print(f'  Compression helps! Like depth=2 beating depth=12.')
    else:
        verdict = 'BN_GRADIENT'
        print(f'  VERDICT: {verdict} — wider pipe tends to help ({spread*100:.2f}% spread).')

    print('=' * 70)

    # Write summary to log
    with open(LIVE_LOG, 'a') as f:
        f.write(f'\n# === SUMMARY ===\n')
        f.write(f'# VERDICT: {verdict} (spread {spread*100:.2f}%)\n')
        for bn in BN_VALUES:
            a = bn_avgs[bn]
            f.write(f'# BN={bn:>3d} ratio={a["ratio"]:.2f}:1 '
                    f'p1={a["p1"]:.4f} p2_tail={a["tail"]:.4f}\n')

    print(f'\nDone. Log: {LIVE_LOG}')
    print(f'Total wall time: {sum(r["time"] for r in all_results):.0f}s '
          f'({sum(r["time"] for r in all_results)/60:.1f} min)')
