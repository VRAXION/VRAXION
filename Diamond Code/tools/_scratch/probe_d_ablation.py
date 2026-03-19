#!/usr/bin/env python3
"""
Probe: D (embedding dimension) ablation at depth=2
===================================================
Tests D={2048, 4096, 6180, 8192} on hard task (block_size=16, 1000 steps).
Depth locked at 2. All other params from Goldilocks config.

Question: Does wider D = better content retrieval through the LCX bottleneck?
- Bottleneck width scales as D/10: {204, 409, 618, 819}
- 16-byte pattern = 128 bits flowing through that channel
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
D_VALUES      = [2048, 4096, 6180, 8192]
DEPTH         = 2
SEQ_LEN       = 32
BLOCK_SIZE    = 16
BATCH         = 10
LR            = 1e-4
STEPS         = 1000
WARMUP        = 50
NUM_BITS      = 8
STEP_TIMEOUT  = 120
RADIUS        = 8
THINK_TICKS   = 1

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DIAMOND_ROOT = r'S:\AI\work\VRAXION_DEV\Diamond Code'
sys.path.insert(0, DIAMOND_ROOT)
LOG_DIR      = os.path.join(DIAMOND_ROOT, 'logs', 'probe')
LIVE_LOG     = os.path.join(LOG_DIR, 'probe_d_ablation_live.log')
os.makedirs(LOG_DIR, exist_ok=True)

from swarm_model import SwarmByteRingModel


def byte_to_bits(byte_seq, num_bits=8):
    t = torch.tensor(byte_seq, dtype=torch.uint8)
    return ((t.unsqueeze(-1) >> torch.arange(num_bits)) & 1).float()


def make_echo_batch(batch_size, seq_len, block_size, num_bits=8):
    xs, ys = [], []
    for _ in range(batch_size):
        block = [random.randint(0, 255) for _ in range(block_size)]
        repeats = (seq_len + 2) // block_size + 1
        data = (block * repeats)[:seq_len + 1]
        xs.append(byte_to_bits(data[:seq_len], num_bits))
        ys.append(byte_to_bits(data[1:seq_len + 1], num_bits))
    return torch.stack(xs).to(DEVICE), torch.stack(ys).to(DEVICE)


def make_model(D):
    return SwarmByteRingModel(
        embedding_dim=D,
        num_memory_positions=SEQ_LEN,
        num_beings=1,
        depth=DEPTH,
        num_bits=NUM_BITS,
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


def run_one(D, log_file):
    """Run 1000 steps at a given D value, return results dict."""
    print(f'\n{"="*70}')
    print(f'PROBE: D={D}  depth={DEPTH}  block={BLOCK_SIZE}  steps={STEPS}')
    print(f'  bottleneck: {D}->{D//10}->C19->{D//10}->C19->{D}')
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

    model = make_model(D).to(DEVICE)
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler('cuda') if DEVICE.type == 'cuda' else None

    def lr_lambda(step):
        if step < WARMUP:
            return step / WARMUP
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    print(f'\n  params={n_params:,}  D={D}  depth={DEPTH}', flush=True)
    if DEVICE.type == 'cuda':
        vram_model = torch.cuda.max_memory_allocated() / 1024**3
        print(f'  VRAM after model load: {vram_model:.2f} GB', flush=True)

    checkpoints = {}
    window_accs = []
    WINDOW = 50

    t_start = time.time()
    for step in range(STEPS):
        t0 = time.time()
        if step < 3:
            print(f'  starting step {step}...', end='', flush=True)

        x, y = make_echo_batch(BATCH, SEQ_LEN, BLOCK_SIZE, NUM_BITS)

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
            return {'D': D, 'status': 'OOM', 'step': step, 'params': n_params}

        elapsed = time.time() - t0
        if step < 3:
            print(f' {elapsed:.2f}s', flush=True)

        if elapsed > STEP_TIMEOUT:
            print(f'  TIMEOUT: step {step} took {elapsed:.0f}s, aborting')
            break

        with torch.no_grad():
            pred = (out > 0).float()
            acc = (pred == y).float().mean().item()

        if math.isnan(loss.item()):
            print(f'  NaN at step {step}, aborting')
            break

        window_accs.append(acc)
        if len(window_accs) > WINDOW:
            window_accs.pop(0)
        smooth_acc = sum(window_accs) / len(window_accs)

        if step in (0, 25, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 999) or step % 100 == 0:
            checkpoints[step] = smooth_acc

        cur_lr = opt.param_groups[0]['lr']
        if step % 50 == 0 or step == STEPS - 1:
            print(f'  step {step:4d} | loss {loss.item():.6f} | acc {acc:.4f} | '
                  f'smooth={smooth_acc:.4f} | lr {cur_lr:.1e} | {elapsed:.2f}s',
                  flush=True)

        with open(log_file, 'a') as lf:
            lf.write(f'step {step} | loss {loss.item():.6f} | '
                     f'acc={acc:.4f} smooth={smooth_acc:.4f} RD:{elapsed:.4f}\n')

    total_time = time.time() - t_start
    vram_peak = 0.0
    if DEVICE.type == 'cuda':
        vram_peak = torch.cuda.max_memory_allocated() / 1024**3

    # Extract tail accuracy
    sorted_ckpts = sorted(checkpoints.items())
    tail_accs = [a for s, a in sorted_ckpts if s >= 800]
    mid_accs = [a for s, a in sorted_ckpts if 300 <= s <= 500]
    tail_avg = sum(tail_accs) / len(tail_accs) if tail_accs else 0
    mid_avg = sum(mid_accs) / len(mid_accs) if mid_accs else 0

    # Print learning curve
    print(f'\n  LEARNING CURVE: D={D}')
    print(f'  {"Step":>6} {"Smooth Acc":>10} {"Delta":>10}')
    prev = 0.5
    for s, a in sorted_ckpts:
        delta = a - prev
        print(f'  {s:>6} {a:>10.4f} {delta:>+10.4f}')
        prev = a

    print(f'\n  Tail (800-999): {tail_avg:.4f}')
    print(f'  VRAM peak: {vram_peak:.2f} GB')
    print(f'  Total time: {total_time:.0f}s ({total_time/60:.1f} min)')
    print(f'  Avg s/step: {total_time/STEPS:.2f}s')

    # Cleanup
    del model, opt, scaler
    gc.collect()
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()

    return {
        'D': D,
        'status': 'OK',
        'params': n_params,
        'vram_peak': vram_peak,
        'tail_acc': tail_avg,
        'mid_acc': mid_avg,
        'avg_step_time': total_time / STEPS,
        'total_time': total_time,
        'checkpoints': dict(sorted_ckpts),
    }


if __name__ == '__main__':
    print('=' * 70)
    print('PROBE: D ABLATION (embedding dimension)')
    print('=' * 70)
    print(f'  D values: {D_VALUES}')
    print(f'  depth={DEPTH}  block={BLOCK_SIZE}  steps={STEPS}  lr={LR}')
    print(f'  device: {DEVICE}')
    if DEVICE.type == 'cuda':
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print('=' * 70)

    with open(LIVE_LOG, 'w') as f:
        f.write(f'# probe_d_ablation -- {time.strftime("%Y-%m-%d %H:%M:%S")}\n')

    results = []
    for D in D_VALUES:
        with open(LIVE_LOG, 'a') as f:
            f.write(f'\n# === D={D} ===\n')

        r = run_one(D, LIVE_LOG)
        results.append(r)
        print(f'\n  D={D} done: {r["status"]}')

    # Summary table
    print('\n' + '=' * 70)
    print('SUMMARY: D ABLATION')
    print('=' * 70)
    print(f'  {"D":>6} {"BN dim":>7} {"Params":>10} {"VRAM":>6} {"s/step":>7} '
          f'{"Tail Acc":>9} {"Status":>6}')
    print(f'  {"-"*6} {"-"*7} {"-"*10} {"-"*6} {"-"*7} {"-"*9} {"-"*6}')

    for r in results:
        if r['status'] == 'OOM':
            print(f'  {r["D"]:>6} {r["D"]//10:>7} {r["params"]:>10,} '
                  f'{"--":>6} {"--":>7} {"--":>9} {"OOM":>6}')
        else:
            print(f'  {r["D"]:>6} {r["D"]//10:>7} {r["params"]:>10,} '
                  f'{r["vram_peak"]:>5.1f}G {r["avg_step_time"]:>6.2f}s '
                  f'{r["tail_acc"]:>8.4f} {"OK":>6}')

    # Find best
    ok_results = [r for r in results if r['status'] == 'OK']
    if ok_results:
        best = max(ok_results, key=lambda r: r['tail_acc'])
        worst = min(ok_results, key=lambda r: r['tail_acc'])
        spread = best['tail_acc'] - worst['tail_acc']

        print(f'\n  Best:   D={best["D"]} ({best["tail_acc"]:.4f})')
        print(f'  Worst:  D={worst["D"]} ({worst["tail_acc"]:.4f})')
        print(f'  Spread: {spread:.4f} ({spread*100:.1f}%)')

        print('\n' + '=' * 70)
        if spread < 0.015:
            print(f'  VERDICT: D_FLAT -- spread {spread:.4f} < 1.5%')
            print(f'  Embedding width does not matter for this task.')
            print(f'  -> Keep D=6180 (golden ratio), move to seq_len/ring.')
        elif best['D'] == D_VALUES[-1]:
            print(f'  VERDICT: BIGGER_IS_BETTER -- D={best["D"]} wins')
            print(f'  -> Consider testing even larger D if VRAM allows.')
        elif best['D'] == D_VALUES[0]:
            print(f'  VERDICT: SMALLER_WINS -- D={best["D"]} wins')
            print(f'  -> Smaller D may generalize better. Save VRAM for other axes.')
        else:
            print(f'  VERDICT: SWEET_SPOT at D={best["D"]}')
            print(f'  -> Lock D={best["D"]}, proceed to seq_len/ring.')
        print('=' * 70)

    with open(LIVE_LOG, 'a') as f:
        f.write(f'\n# === SUMMARY ===\n')
        for r in results:
            if r['status'] == 'OK':
                f.write(f'# D={r["D"]} tail_acc={r["tail_acc"]:.4f} '
                        f'vram={r["vram_peak"]:.1f}G s/step={r["avg_step_time"]:.2f}\n')
