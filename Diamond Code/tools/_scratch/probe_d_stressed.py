#!/usr/bin/env python3
"""
Probe: D ablation with STRESSED bottleneck (wide num_bits)
==========================================================
Two-phase probe to test if D matters when the pipe is full.

Phase 1 (200 steps): num_bits=8, learn basic routing (plateaus by step 200)
Phase 2 (800 steps): num_bits=200, stress the D/10 bottleneck

At num_bits=200:
  D=2048  bottleneck=204  ratio=1.02:1  PIPE FULL
  D=4096  bottleneck=409  ratio=2.0:1   comfortable
  D=6180  bottleneck=618  ratio=3.1:1   roomy
  D=8192  bottleneck=819  ratio=4.1:1   spacious

If D matters: D=2048 chokes, D=6180+ cruises.
If D doesn't matter: all plateau the same (task-bound, not pipe-bound).
"""

import gc
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F

# --- Configs ---
D_VALUES       = [2048, 4096, 6180, 8192]
DEPTH          = 2
SEQ_LEN        = 32
BLOCK_SIZE     = 16
BATCH          = 10
LR             = 1e-4
WARMUP         = 50
NUM_BITS_P1    = 8       # Phase 1: narrow (warm up routing)
NUM_BITS_P2    = 200     # Phase 2: wide (stress bottleneck)
STEPS_P1       = 200     # Phase 1 steps (plateau confirmed at ~200)
STEPS_P2       = 800     # Phase 2 steps (stressed training)
STEP_TIMEOUT   = 120
RADIUS         = 8
THINK_TICKS    = 1

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DIAMOND_ROOT = r'S:\AI\work\VRAXION_DEV\Diamond Code'
sys.path.insert(0, DIAMOND_ROOT)
LOG_DIR      = os.path.join(DIAMOND_ROOT, 'logs', 'probe')
LIVE_LOG     = os.path.join(LOG_DIR, 'probe_d_stressed_live.log')
os.makedirs(LOG_DIR, exist_ok=True)

from swarm_model import SwarmByteRingModel


def make_echo_batch_bits(batch_size, seq_len, block_size, num_bits):
    """Generate repeating block patterns with arbitrary num_bits width."""
    xs, ys = [], []
    for _ in range(batch_size):
        # Random block: block_size positions, each with num_bits binary features
        block = torch.randint(0, 2, (block_size, num_bits)).float()
        repeats = (seq_len + 2) // block_size + 1
        data = block.repeat(repeats, 1)[:seq_len + 1]
        xs.append(data[:seq_len])
        ys.append(data[1:seq_len + 1])
    return torch.stack(xs).to(DEVICE), torch.stack(ys).to(DEVICE)


def make_model(D, num_bits):
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


def expand_model_bits(model, old_bits, new_bits, D):
    """Expand num_bits: copy old projection weights, randomly init new rows."""
    with torch.no_grad():
        # Find and expand input projection (num_bits -> D)
        for name, param in model.named_parameters():
            if 'input' in name.lower() or 'embed' in name.lower():
                pass  # Will handle via module replacement below

    # The model stores num_bits and uses it in projections.
    # Easiest: create a new model with new_bits and copy matching weights.
    new_model = make_model(D, new_bits).to(DEVICE)
    new_model.train()

    old_state = model.state_dict()
    new_state = new_model.state_dict()

    copied, expanded, skipped = 0, 0, 0
    for key in new_state:
        if key in old_state:
            old_shape = old_state[key].shape
            new_shape = new_state[key].shape
            if old_shape == new_shape:
                # Exact match: copy directly
                new_state[key] = old_state[key]
                copied += 1
            elif len(old_shape) == 2 and len(new_shape) == 2:
                # Matrix: copy overlapping region, keep random init for new rows/cols
                min_r = min(old_shape[0], new_shape[0])
                min_c = min(old_shape[1], new_shape[1])
                new_state[key][:min_r, :min_c] = old_state[key][:min_r, :min_c]
                expanded += 1
            elif len(old_shape) == 1 and len(new_shape) == 1:
                # Bias: copy overlapping region
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


def run_one(D, log_file):
    """Two-phase probe: warm up at num_bits=8, stress at num_bits=200."""
    total_steps = STEPS_P1 + STEPS_P2

    print(f'\n{"="*70}')
    print(f'PROBE: D={D}  depth={DEPTH}  block={BLOCK_SIZE}')
    print(f'  Phase 1: {STEPS_P1} steps at num_bits={NUM_BITS_P1} (warm up)')
    print(f'  Phase 2: {STEPS_P2} steps at num_bits={NUM_BITS_P2} (stress)')
    print(f'  bottleneck: {D}->{D//10}->C19->{D//10}->C19->{D}')
    print(f'  bottleneck vs bits: {D//10} dims carrying {NUM_BITS_P2} bits '
          f'(ratio {D//10/NUM_BITS_P2:.1f}:1)')
    print(f'  LCX: hash, 2000 slots, key_dim={D//10}, top_k=2')
    print(f'  device: {DEVICE}')
    if DEVICE.type == 'cuda':
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'{"="*70}')

    gc.collect()
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    torch.manual_seed(42)
    random.seed(42)

    # Phase 1: narrow bits
    model = make_model(D, NUM_BITS_P1).to(DEVICE)
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
    print(f'  params={n_params_p1:,}  D={D}', flush=True)

    checkpoints = {}
    window_accs = []
    WINDOW = 50

    t_start = time.time()
    global_step = 0

    # --- Phase 1 loop ---
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
            return {'D': D, 'status': 'OOM', 'phase': 1, 'step': step,
                    'params_p1': n_params_p1}

        elapsed = time.time() - t0
        if step < 3:
            print(f' {elapsed:.2f}s', flush=True)

        if elapsed > STEP_TIMEOUT:
            print(f'  TIMEOUT at step {step}')
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
                     f'acc={acc:.4f} smooth={smooth_acc:.4f} RD:{elapsed:.4f} P1\n')

        global_step += 1

    p1_final_acc = smooth_acc if window_accs else 0.5
    print(f'\n  Phase 1 done: smooth_acc={p1_final_acc:.4f}')

    # --- Expand model to Phase 2 ---
    print(f'\n  === PHASE 2: expanding num_bits {NUM_BITS_P1} -> {NUM_BITS_P2} ===')
    new_model = expand_model_bits(model, NUM_BITS_P1, NUM_BITS_P2, D)

    # Clean up old model
    del model, opt, scaler
    gc.collect()
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()

    model = new_model
    model.train()
    n_params_p2 = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler('cuda') if DEVICE.type == 'cuda' else None

    # Fresh scheduler for Phase 2 (no warmup needed, weights are warm)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)

    print(f'  params={n_params_p2:,} (was {n_params_p1:,})')
    if DEVICE.type == 'cuda':
        vram_after = torch.cuda.max_memory_allocated() / 1024**3
        print(f'  VRAM after expansion: {vram_after:.2f} GB')

    # Reset accuracy tracking for Phase 2
    window_accs = []

    # --- Phase 2 loop ---
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
            return {'D': D, 'status': 'OOM', 'phase': 2, 'step': global_step,
                    'params_p1': n_params_p1, 'params_p2': n_params_p2,
                    'p1_acc': p1_final_acc}

        elapsed = time.time() - t0
        if step < 3:
            print(f' {elapsed:.2f}s', flush=True)

        if elapsed > STEP_TIMEOUT:
            print(f'  TIMEOUT at step {global_step}')
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

        if step % 50 == 0 or step == STEPS_P2 - 1 or step in (25,):
            checkpoints[global_step] = smooth_acc

        if step % 50 == 0 or step == STEPS_P2 - 1:
            print(f'  step {global_step:4d} | loss {loss.item():.6f} | acc {acc:.4f} | '
                  f'smooth={smooth_acc:.4f} | {elapsed:.2f}s [P2]', flush=True)

        with open(log_file, 'a') as lf:
            lf.write(f'step {global_step} | loss {loss.item():.6f} | '
                     f'acc={acc:.4f} smooth={smooth_acc:.4f} RD:{elapsed:.4f} P2\n')

        global_step += 1

    total_time = time.time() - t_start
    vram_peak = 0.0
    if DEVICE.type == 'cuda':
        vram_peak = torch.cuda.max_memory_allocated() / 1024**3

    # Extract Phase 2 tail accuracy
    p2_checkpoints = {s: a for s, a in checkpoints.items() if s >= STEPS_P1}
    sorted_p2 = sorted(p2_checkpoints.items())
    tail_accs = [a for s, a in sorted_p2 if s >= STEPS_P1 + 600]
    mid_accs = [a for s, a in sorted_p2 if STEPS_P1 + 200 <= s <= STEPS_P1 + 400]
    tail_avg = sum(tail_accs) / len(tail_accs) if tail_accs else 0
    mid_avg = sum(mid_accs) / len(mid_accs) if mid_accs else 0

    # Print learning curve
    print(f'\n  LEARNING CURVE: D={D}')
    print(f'  {"Step":>6} {"Smooth Acc":>10} {"Phase":>6}')
    for s, a in sorted(checkpoints.items()):
        phase = 'P1' if s < STEPS_P1 else 'P2'
        print(f'  {s:>6} {a:>10.4f}   {phase}')

    print(f'\n  Phase 2 tail (step 800-999): {tail_avg:.4f}')
    print(f'  VRAM peak: {vram_peak:.2f} GB')
    print(f'  Total time: {total_time:.0f}s ({total_time/60:.1f} min)')

    # Cleanup
    del model, opt, scaler
    gc.collect()
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()

    return {
        'D': D,
        'status': 'OK',
        'params_p1': n_params_p1,
        'params_p2': n_params_p2,
        'vram_peak': vram_peak,
        'p1_acc': p1_final_acc,
        'p2_tail_acc': tail_avg,
        'p2_mid_acc': mid_avg,
        'total_time': total_time,
        'checkpoints': dict(sorted(checkpoints.items())),
    }


if __name__ == '__main__':
    print('=' * 70)
    print('PROBE: D ABLATION — STRESSED BOTTLENECK')
    print('=' * 70)
    print(f'  D values: {D_VALUES}')
    print(f'  Phase 1: {STEPS_P1} steps at num_bits={NUM_BITS_P1}')
    print(f'  Phase 2: {STEPS_P2} steps at num_bits={NUM_BITS_P2}')
    print(f'  depth={DEPTH}  block={BLOCK_SIZE}  lr={LR}')
    print(f'  device: {DEVICE}')
    if DEVICE.type == 'cuda':
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print()
    print(f'  Bottleneck stress ratios at num_bits={NUM_BITS_P2}:')
    for D in D_VALUES:
        bn = D // 10
        ratio = bn / NUM_BITS_P2
        stress = 'FULL' if ratio < 1.5 else 'OK' if ratio < 3 else 'ROOMY'
        print(f'    D={D:>5}  BN={bn:>4}  ratio={ratio:.1f}:1  [{stress}]')
    print('=' * 70)

    with open(LIVE_LOG, 'w') as f:
        f.write(f'# probe_d_stressed -- {time.strftime("%Y-%m-%d %H:%M:%S")}\n')

    results = []
    for D in D_VALUES:
        with open(LIVE_LOG, 'a') as f:
            f.write(f'\n# === D={D} ===\n')

        r = run_one(D, LIVE_LOG)
        results.append(r)
        print(f'\n  D={D} done: {r["status"]}')

    # Summary table
    print('\n' + '=' * 70)
    print(f'SUMMARY: D ABLATION (stressed, num_bits={NUM_BITS_P2})')
    print('=' * 70)
    print(f'  {"D":>6} {"BN":>5} {"Ratio":>6} {"Params":>10} {"VRAM":>6} '
          f'{"P1 Acc":>7} {"P2 Tail":>8} {"Status":>6}')
    print(f'  {"-"*6} {"-"*5} {"-"*6} {"-"*10} {"-"*6} '
          f'{"-"*7} {"-"*8} {"-"*6}')

    for r in results:
        if r['status'] == 'OOM':
            print(f'  {r["D"]:>6} {r["D"]//10:>5} '
                  f'{r["D"]//10/NUM_BITS_P2:>5.1f}x '
                  f'{r.get("params_p2", r["params_p1"]):>10,} '
                  f'{"--":>6} {"--":>7} {"--":>8} {"OOM":>6}')
        else:
            print(f'  {r["D"]:>6} {r["D"]//10:>5} '
                  f'{r["D"]//10/NUM_BITS_P2:>5.1f}x '
                  f'{r["params_p2"]:>10,} '
                  f'{r["vram_peak"]:>5.1f}G '
                  f'{r["p1_acc"]:>6.4f} '
                  f'{r["p2_tail_acc"]:>7.4f} '
                  f'{"OK":>6}')

    # Verdict
    ok_results = [r for r in results if r['status'] == 'OK']
    if ok_results:
        best = max(ok_results, key=lambda r: r['p2_tail_acc'])
        worst = min(ok_results, key=lambda r: r['p2_tail_acc'])
        spread = best['p2_tail_acc'] - worst['p2_tail_acc']

        print(f'\n  Best:   D={best["D"]} ({best["p2_tail_acc"]:.4f})')
        print(f'  Worst:  D={worst["D"]} ({worst["p2_tail_acc"]:.4f})')
        print(f'  Spread: {spread:.4f} ({spread*100:.1f}%)')

        print('\n' + '=' * 70)
        if spread < 0.015:
            print(f'  VERDICT: D_FLAT even under stress (spread {spread*100:.1f}%)')
            print(f'  -> Lock D=6180 (golden ratio). Bottleneck width is not the limiter.')
        elif worst['D'] == D_VALUES[0] and best['D'] != D_VALUES[0]:
            print(f'  VERDICT: SMALL_D_CHOKES — D={worst["D"]} bottleneck too narrow')
            print(f'  -> D={best["D"]} wins. Lock it.')
        elif best['D'] == D_VALUES[-1]:
            print(f'  VERDICT: BIGGER_IS_BETTER — wider pipe = better throughput')
            print(f'  -> Consider D={best["D"]} or larger.')
        else:
            print(f'  VERDICT: SWEET_SPOT at D={best["D"]}')
            print(f'  -> Diminishing returns above D={best["D"]}.')
        print('=' * 70)

    with open(LIVE_LOG, 'a') as f:
        f.write(f'\n# === SUMMARY ===\n')
        for r in results:
            if r['status'] == 'OK':
                f.write(f'# D={r["D"]} p1_acc={r["p1_acc"]:.4f} '
                        f'p2_tail={r["p2_tail_acc"]:.4f} '
                        f'vram={r["vram_peak"]:.1f}G\n')
