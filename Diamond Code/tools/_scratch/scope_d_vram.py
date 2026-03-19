#!/usr/bin/env python3
"""
Scope: D dimension VRAM & throughput estimation
================================================
For each candidate D, construct the model, count params,
do a forward+backward pass, measure VRAM and step time.
This tells us which D values fit on RTX 4070 Ti SUPER (16 GB).
"""

import gc
import os
import sys
import time

import torch

DIAMOND_ROOT = r'S:\AI\work\VRAXION_DEV\Diamond Code'
sys.path.insert(0, DIAMOND_ROOT)

from swarm_model import SwarmByteRingModel

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')
if DEVICE.type == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')

# Fixed architecture params (same as production except D varies)
DEPTH       = 6
SEQ_LEN     = 62
MEMORY_SIZE = 62
NUM_BITS    = 8
NUM_BEINGS  = 1
RADIUS      = 8
THINK_TICKS = 1
NUM_POINTERS = 1

# Candidate D values
CANDIDATES = [
    # (label, D, key_dim, batch_size)
    ('D=2048',  2048,  205,  10),
    ('D=3090',  3090,  309,  10),   # φ × 5000
    ('D=4096',  4096,  410,  10),
    ('D=6180',  6180,  618,  10),   # current production
    ('D=8192',  8192,  819,  5),    # might be tight, try smaller batch
    ('D=10000', 10000, 1000, 2),    # likely OOM at batch=10
]

STEP_TIMEOUT = 120


def test_config(label, d, key_dim, batch_size):
    """Construct model, do 3 forward+backward, measure VRAM and speed."""
    gc.collect()
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    print(f'\n  [{label}] D={d}, key_dim={key_dim}, batch={batch_size}')

    try:
        model = SwarmByteRingModel(
            embedding_dim=d,
            num_memory_positions=MEMORY_SIZE,
            num_beings=NUM_BEINGS,
            depth=DEPTH,
            num_bits=NUM_BITS,
            attention_radius=RADIUS,
            attention_temperature=8.0,
            think_ticks=THINK_TICKS,
            use_lcx=True,
            lcx_mode='hash',
            lcx_num_levels=1,
            lcx_level_slots=[2000],
            lcx_key_dim=key_dim,
            lcx_top_k=2,
            num_pointers=NUM_POINTERS,
        ).to(DEVICE)
    except Exception as e:
        print(f'    FAILED to construct: {e}')
        return None

    n_params = sum(p.numel() for p in model.parameters())
    param_mb = n_params * 4 / 1024**2  # fp32
    print(f'    params: {n_params:,} ({param_mb:.0f} MB fp32)')

    if DEVICE.type == 'cuda':
        vram_model = torch.cuda.max_memory_allocated() / 1024**3
        print(f'    VRAM after model load: {vram_model:.2f} GB')

    # Create dummy input
    x = torch.randn(batch_size, SEQ_LEN, NUM_BITS, device=DEVICE)
    y = torch.randn(batch_size, SEQ_LEN, NUM_BITS, device=DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()

    # Use AMP like production
    scaler = torch.amp.GradScaler('cuda') if DEVICE.type == 'cuda' else None

    step_times = []
    for step in range(3):
        t0 = time.time()
        print(f'    starting step {step}...', flush=True)

        opt.zero_grad()

        if scaler is not None:
            with torch.amp.autocast('cuda'):
                out = model(x)
                if isinstance(out, tuple):
                    out = out[0]
                loss = torch.nn.functional.binary_cross_entropy_with_logits(out, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        else:
            out = model(x)
            if isinstance(out, tuple):
                out = out[0]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        elapsed = time.time() - t0
        step_times.append(elapsed)
        print(f'    step {step}: loss={loss.item():.4f}, time={elapsed:.2f}s')

        if elapsed > STEP_TIMEOUT:
            print(f'    TIMEOUT at step {step}')
            break

    if DEVICE.type == 'cuda':
        vram_peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f'    VRAM peak: {vram_peak:.2f} GB')
    else:
        vram_peak = 0.0

    avg_time = sum(step_times[1:]) / max(len(step_times) - 1, 1)  # skip warmup

    # Cleanup
    del model, opt, x, y, out, loss
    if scaler is not None:
        del scaler
    gc.collect()
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()

    return {
        'label': label, 'd': d, 'key_dim': key_dim, 'batch': batch_size,
        'params': n_params, 'vram_peak_gb': vram_peak,
        'avg_step_s': avg_time, 'param_mb': param_mb,
    }


if __name__ == '__main__':
    print('=' * 70)
    print('SCOPE: D dimension VRAM & throughput')
    print('=' * 70)
    print(f'  depth={DEPTH}  seq={SEQ_LEN}  mem={MEMORY_SIZE}  bits={NUM_BITS}')
    print(f'  beings={NUM_BEINGS}  radius={RADIUS}  tt={THINK_TICKS}  ptrs={NUM_POINTERS}')
    print(f'  LCX: hash, 1 level, 2000 slots, top_k=2')
    print('=' * 70)

    results = []
    for label, d, key_dim, batch in CANDIDATES:
        try:
            r = test_config(label, d, key_dim, batch)
            if r is not None:
                results.append(r)
        except torch.cuda.OutOfMemoryError:
            print(f'    OOM! D={d} does not fit at batch={batch}')
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            print(f'    ERROR: {e}')

    # Summary
    print('\n' + '=' * 70)
    print('SUMMARY: D dimension feasibility')
    print('=' * 70)
    print(f'  {"Label":<12} {"D":>6} {"Params":>12} {"VRAM Peak":>10} '
          f'{"Step Time":>10} {"Batch":>6} {"key_dim":>8}')
    print(f'  {"-"*12} {"-"*6} {"-"*12} {"-"*10} {"-"*10} {"-"*6} {"-"*8}')

    for r in results:
        print(f'  {r["label"]:<12} {r["d"]:>6} {r["params"]:>12,} '
              f'{r["vram_peak_gb"]:>9.2f}G {r["avg_step_s"]:>9.2f}s '
              f'{r["batch"]:>6} {r["key_dim"]:>8}')

    print('\n  16 GB VRAM budget. Configs above ~14 GB are risky (fragmentation).')
    print('=' * 70)
