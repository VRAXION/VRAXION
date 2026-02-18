#!/usr/bin/env python3
"""
Golden Ratio Fractal CPU Probe: Push ALL dimensions to phi scales.
Tests D={618,6180} × depth={6,12,32,62} × num_bits={64,128,624,6184} at seq_len=6, tt=1.
Target: 1-5s/step on CPU.
"""

import sys
import time
import json
import gc
from pathlib import Path
from collections import OrderedDict

DIAMOND_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(DIAMOND_ROOT))

import torch
import torch.nn as nn

from swarm_model import SwarmByteRingModel

SEED = 42
LR = 3e-4
STEPS = 5  # timing only
STEP_TIMEOUT = 120
RAM_LIMIT_GB = 28  # conservative for 32 GB system

CONFIGS = OrderedDict()

# ── D=618 depth scaling (batch=10) ─────────────────────────────
CONFIGS["d618_dep6_b64"] = dict(
    D=618, depth=6, seq_len=6, num_bits=64, batch=10,
    label="D=618  d=6  b=64")

CONFIGS["d618_dep12_b64"] = dict(
    D=618, depth=12, seq_len=6, num_bits=64, batch=10,
    label="D=618  d=12 b=64")

CONFIGS["d618_dep32_b64"] = dict(
    D=618, depth=32, seq_len=6, num_bits=64, batch=10,
    label="D=618  d=32 b=64")

CONFIGS["d618_dep62_b64"] = dict(
    D=618, depth=62, seq_len=6, num_bits=64, batch=10,
    label="D=618  d=62 b=64")

# ── D=618 bits scaling at depth=62 ─────────────────────────────
CONFIGS["d618_dep62_b128"] = dict(
    D=618, depth=62, seq_len=6, num_bits=128, batch=10,
    label="D=618  d=62 b=128")

CONFIGS["d618_dep62_b624"] = dict(
    D=618, depth=62, seq_len=6, num_bits=624, batch=10,
    label="D=618  d=62 b=624")

CONFIGS["d618_dep62_b6184"] = dict(
    D=618, depth=62, seq_len=6, num_bits=6184, batch=1,
    label="D=618  d=62 b=6184")

# ── D=6180 depth scaling (batch=1) ─────────────────────────────
CONFIGS["d6180_dep6_b128"] = dict(
    D=6180, depth=6, seq_len=6, num_bits=128, batch=1,
    label="D=6180 d=6  b=128")

CONFIGS["d6180_dep12_b128"] = dict(
    D=6180, depth=12, seq_len=6, num_bits=128, batch=1,
    label="D=6180 d=12 b=128")

CONFIGS["d6180_dep12_b624"] = dict(
    D=6180, depth=12, seq_len=6, num_bits=624, batch=1,
    label="D=6180 d=12 b=624")

# ── THE DREAM: full phi at all scales ──────────────────────────
CONFIGS["dream_d6180_dep62_b624"] = dict(
    D=6180, depth=62, seq_len=6, num_bits=624, batch=1,
    label="DREAM  D=6180 d=62 b=624")


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def estimate_ram_gb(n_params):
    """fp64 weights + Adam moment + Adam variance = 24 bytes/param."""
    return n_params * 24 / (1024**3)


def rough_param_estimate(D, depth):
    """Quick estimate before model build (processing layers dominate)."""
    proc = max(0, depth - 1) * (D * D + D)
    return int(proc * 1.4)  # ~40% overhead for LCX, attention, embeddings, IO


def build_model(cfg, device='cpu'):
    _bn_dim = max(1, cfg['D'] // 10)
    model = SwarmByteRingModel(
        num_memory_positions=cfg['seq_len'],
        embedding_dim=cfg['D'],
        num_beings=1,
        depth=cfg['depth'],
        num_bits=cfg['num_bits'],
        attention_radius=3,
        think_ticks=1,
        use_lcx=True,
        lcx_mode='hash',
        lcx_num_levels=1,
        lcx_level_slots=[2000],
        lcx_key_dim=_bn_dim,
        lcx_top_k=6,
        num_pointers=1,
        full_view=False,
    )
    return model.double().to(device)


def run_config(name, cfg, device='cpu'):
    D, depth, bits = cfg['D'], cfg['depth'], cfg['num_bits']
    batch = cfg['batch']
    seq = cfg['seq_len']
    ctx_bytes = (bits // 8) * seq

    print(f"\n{'='*70}")
    print(f"  {cfg['label']}")
    print(f"  D={D} depth={depth} seq={seq} bits={bits} batch={batch} ctx={ctx_bytes}B")
    print(f"{'='*70}")

    # Pre-check: estimate RAM
    est_params = rough_param_estimate(D, depth)
    est_ram = estimate_ram_gb(est_params)
    print(f"  Estimate: ~{est_params/1e6:.1f}M params, ~{est_ram:.1f} GB RAM (fp64+Adam)")

    if est_ram > RAM_LIMIT_GB:
        print(f"  SKIP: {est_ram:.1f} GB > {RAM_LIMIT_GB} GB limit")
        return {
            'name': name, 'label': cfg['label'],
            'D': D, 'depth': depth, 'seq_len': seq,
            'num_bits': bits, 'batch': batch,
            'context_bytes': ctx_bytes,
            'est_params_M': est_params / 1e6,
            'est_memory_GB': est_ram,
            'error': f'OOM: est {est_ram:.1f} GB > {RAM_LIMIT_GB} GB',
        }

    # Build model
    try:
        t_build = time.time()
        model = build_model(cfg, device)
        build_s = time.time() - t_build
    except (RuntimeError, MemoryError, Exception) as e:
        print(f"  OOM at build: {e}")
        gc.collect()
        return {'name': name, 'label': cfg['label'], 'D': D, 'depth': depth,
                'seq_len': seq, 'num_bits': bits, 'batch': batch,
                'context_bytes': ctx_bytes,
                'error': f'OOM build: {str(e)[:80]}'}

    n_params = count_params(model)
    ram_gb = estimate_ram_gb(n_params)
    size_fp16_mb = n_params * 2 / (1024 * 1024)
    print(f"  Built in {build_s:.1f}s | {n_params:,} params ({size_fp16_mb:.1f}MB fp16) | ~{ram_gb:.1f} GB RAM")

    # Create optimizer
    try:
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    except (RuntimeError, MemoryError) as e:
        print(f"  OOM at optimizer: {e}")
        del model; gc.collect()
        return {'name': name, 'label': cfg['label'], 'D': D, 'depth': depth,
                'seq_len': seq, 'num_bits': bits, 'batch': batch,
                'params': n_params, 'context_bytes': ctx_bytes,
                'error': f'OOM optimizer: {str(e)[:80]}'}

    # Run timing steps (random data — we only care about speed)
    step_times = []
    losses = []

    for step in range(STEPS):
        print(f"  starting step {step}...", end='', flush=True)
        t0 = time.time()

        try:
            torch.manual_seed(SEED + step)
            x = torch.rand(batch, seq, bits, dtype=torch.float64)
            y = torch.rand(batch, seq, bits).round().to(torch.float64)

            output, stats = model(x, return_stats=True, return_being_outputs=True)
            loss = nn.functional.binary_cross_entropy_with_logits(output, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            dt = time.time() - t0
            loss_val = loss.item()
            step_times.append(dt)
            losses.append(loss_val)
            print(f" loss={loss_val:.4f} ({dt:.2f}s)")

            if dt > STEP_TIMEOUT:
                print(f"  TIMEOUT at step {step}")
                break

        except (RuntimeError, MemoryError) as e:
            dt = time.time() - t0
            print(f" OOM at step {step} ({dt:.1f}s): {e}")
            break

    # Cleanup
    del model, optimizer
    gc.collect()

    if not step_times:
        return {'name': name, 'label': cfg['label'], 'D': D, 'depth': depth,
                'seq_len': seq, 'num_bits': bits, 'batch': batch,
                'params': n_params, 'context_bytes': ctx_bytes,
                'error': 'No steps completed'}

    # Average (skip step 0 warmup)
    warmup = min(1, len(step_times) - 1)
    avg_time = sum(step_times[warmup:]) / len(step_times[warmup:])
    in_budget = "YES" if 1.0 <= avg_time <= 5.0 else ("FAST" if avg_time < 1.0 else "SLOW")

    result = {
        'name': name, 'label': cfg['label'],
        'D': D, 'depth': depth, 'seq_len': seq,
        'num_bits': bits, 'batch': batch,
        'params': n_params, 'params_M': round(n_params / 1e6, 2),
        'size_fp16_MB': round(size_fp16_mb, 1),
        'memory_GB': round(ram_gb, 1),
        'context_bytes': ctx_bytes,
        'avg_step_time': round(avg_time, 3),
        'step_times': [round(t, 3) for t in step_times],
        'final_loss': losses[-1],
        'steps_completed': len(step_times),
        'in_budget': in_budget,
    }

    print(f"\n  RESULT: {avg_time:.2f}s/step [{in_budget}] | "
          f"{n_params:,} params ({size_fp16_mb:.1f}MB fp16) | {ctx_bytes}B context")

    return result


def main():
    print("=" * 70)
    print("GOLDEN RATIO FRACTAL CPU PROBE")
    print(f"phi scales: D={{618,6180}} depth={{6..62}} bits={{64..6184}}")
    print(f"Fixed: seq_len=6, tt=1 | {STEPS} steps for timing")
    print(f"RAM limit: {RAM_LIMIT_GB} GB | Step timeout: {STEP_TIMEOUT}s")
    print("=" * 70)

    all_results = {}
    for name, cfg in CONFIGS.items():
        try:
            result = run_config(name, cfg)
            all_results[name] = result
        except Exception as e:
            print(f"\n  CRASHED: {e}")
            import traceback
            traceback.print_exc()
            all_results[name] = {'name': name, 'error': str(e),
                                 'label': cfg.get('label', name)}

    # ── Summary Table ──────────────────────────────────────────
    print("\n\n" + "=" * 130)
    print("GOLDEN RATIO FRACTAL PROBE — SUMMARY")
    print("=" * 130)
    hdr = (f"{'Config':<32} {'D':>5} {'dep':>3} {'bits':>5} {'bat':>3} "
           f"{'Params':>10} {'fp16':>7} {'RAM':>6} {'Ctx':>5} {'s/step':>7} {'Budget':>6}")
    print(hdr)
    print("-" * 130)

    for name, r in all_results.items():
        if 'error' in r:
            est = ""
            if 'est_params_M' in r:
                est = f" (est ~{r['est_params_M']:.0f}M, ~{r['est_memory_GB']:.0f}GB)"
            print(f"{r.get('label', name):<32} {r.get('D', '?'):>5} {r.get('depth', '?'):>3} "
                  f"{r.get('num_bits', '?'):>5} {r.get('batch', '?'):>3}  "
                  f"{'':>10} {'':>7} {'':>6} {r.get('context_bytes', '?'):>5} "
                  f"{'':>7} FAIL: {r['error'][:40]}")
            continue
        t = r['avg_step_time']
        ib = r['in_budget']
        print(f"{r['label']:<32} {r['D']:>5} {r['depth']:>3} {r['num_bits']:>5} "
              f"{r['batch']:>3} {r['params']:>10,} {r['size_fp16_MB']:>6.1f}M "
              f"{r['memory_GB']:>5.1f}G {r['context_bytes']:>4}B "
              f"{t:>6.2f}s {ib:>6}")

    # ── Winner ─────────────────────────────────────────────────
    in_budget = {k: v for k, v in all_results.items()
                 if 'error' not in v and 1.0 <= v['avg_step_time'] <= 5.0}
    if in_budget:
        best = max(in_budget.values(), key=lambda x: x['params'])
        print(f"\n  WINNER (most params in budget): {best['label']}")
        print(f"    {best['avg_step_time']:.2f}s/step | {best['params']:,} params "
              f"({best['size_fp16_MB']:.1f}MB fp16) | {best['context_bytes']}B context")

    fast = {k: v for k, v in all_results.items()
            if 'error' not in v and v['avg_step_time'] < 1.0}
    if fast:
        biggest_fast = max(fast.values(), key=lambda x: x['params'])
        print(f"\n  BIGGEST FAST (<1s): {biggest_fast['label']}")
        print(f"    {biggest_fast['avg_step_time']:.2f}s/step | {biggest_fast['params']:,} params")

    # ── Save ───────────────────────────────────────────────────
    out_path = DIAMOND_ROOT / "tools" / "_scratch" / "bench_golden_ratio_results.json"
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to: {out_path}")


if __name__ == '__main__':
    main()
