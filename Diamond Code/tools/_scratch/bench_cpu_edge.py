#!/usr/bin/env python3
"""
CPU Edge Benchmark: Find the sweet spot for CPU training + edge deployment.
Tests D × depth × seq_len × num_bits combinations at batch=10, tt=1.
Target: 1-5s/step on CPU.
"""

import sys
import time
import json
from pathlib import Path
from collections import OrderedDict

DIAMOND_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(DIAMOND_ROOT))

import torch
import torch.nn as nn

from swarm_model import SwarmByteRingModel
from traindat_loader import TraindatLoader

SEED = 42
LR = 3e-4
STEPS = 30  # enough for timing + early learning signal
BATCH = 10
STEP_TIMEOUT = 60
DATA_DIR = DIAMOND_ROOT / "data" / "traindat"
WEIGHTS = {
    "copy_echo256.traindat": 1,
    "constant256.traindat": 0,
    "add256.traindat": 0,
    "count256.traindat": 0,
    "delay_echo256.traindat": 0,
    "denoise256.traindat": 0,
    "echo256.traindat": 0,
    "fib256.traindat": 0,
    "gold_origin_echo.traindat": 0,
    "not256.traindat": 0,
    "shift256.traindat": 0,
}

CONFIGS = OrderedDict()

# Tier 1: Tiny (baseline reference)
CONFIGS["tiny_D128_d4_s16_b64"] = dict(
    D=128, depth=4, seq_len=16, num_bits=64,
    label="D=128 depth=4 seq=16 bits=64 (128B ctx)")

# Tier 2: Small
CONFIGS["small_D256_d4_s16_b64"] = dict(
    D=256, depth=4, seq_len=16, num_bits=64,
    label="D=256 depth=4 seq=16 bits=64 (128B ctx)")

# Tier 3: Sweet spot candidates (128B context, short seq)
CONFIGS["sweet_D256_d4_s8_b128"] = dict(
    D=256, depth=4, seq_len=8, num_bits=128,
    label="D=256 depth=4 seq=8 bits=128 (128B ctx)")

CONFIGS["sweet_D256_d6_s8_b128"] = dict(
    D=256, depth=6, seq_len=8, num_bits=128,
    label="D=256 depth=6 seq=8 bits=128 (128B ctx)")

CONFIGS["sweet_D256_d8_s8_b128"] = dict(
    D=256, depth=8, seq_len=8, num_bits=128,
    label="D=256 depth=8 seq=8 bits=128 (128B ctx)")

# Tier 4: Medium (more D, same context)
CONFIGS["med_D384_d4_s8_b128"] = dict(
    D=384, depth=4, seq_len=8, num_bits=128,
    label="D=384 depth=4 seq=8 bits=128 (128B ctx)")

CONFIGS["med_D384_d6_s8_b128"] = dict(
    D=384, depth=6, seq_len=8, num_bits=128,
    label="D=384 depth=6 seq=8 bits=128 (128B ctx)")

# Tier 5: Large context (256B)
CONFIGS["wide_D256_d4_s16_b128"] = dict(
    D=256, depth=4, seq_len=16, num_bits=128,
    label="D=256 depth=4 seq=16 bits=128 (256B ctx)")

CONFIGS["wide_D256_d6_s16_b128"] = dict(
    D=256, depth=6, seq_len=16, num_bits=128,
    label="D=256 depth=6 seq=16 bits=128 (256B ctx)")

# Tier 6: Push it
CONFIGS["push_D384_d8_s8_b128"] = dict(
    D=384, depth=8, seq_len=8, num_bits=128,
    label="D=384 depth=8 seq=8 bits=128 (128B ctx)")

CONFIGS["push_D512_d4_s8_b128"] = dict(
    D=512, depth=4, seq_len=8, num_bits=128,
    label="D=512 depth=4 seq=8 bits=128 (128B ctx)")

# Tier 7: The golden ratio edge candidate
CONFIGS["phi_D618_d4_s8_b64"] = dict(
    D=618, depth=4, seq_len=8, num_bits=64,
    label="D=618 depth=4 seq=8 bits=64 (64B ctx)")

CONFIGS["phi_D618_d4_s8_b128"] = dict(
    D=618, depth=4, seq_len=8, num_bits=128,
    label="D=618 depth=4 seq=8 bits=128 (128B ctx)")


def count_params(model):
    return sum(p.numel() for p in model.parameters())


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


def run_config(name, cfg, loader, device='cpu'):
    print(f"\n{'='*70}")
    print(f"  {cfg['label']}")
    print(f"  D={cfg['D']} depth={cfg['depth']} seq={cfg['seq_len']} bits={cfg['num_bits']}")
    print(f"{'='*70}")

    torch.manual_seed(SEED)
    model = build_model(cfg, device)
    n_params = count_params(model)
    bpp = cfg['num_bits'] // 8
    ctx_bytes = bpp * cfg['seq_len']
    loop_iters = cfg['seq_len'] * 2  # tt=1 means 2x seq_len
    print(f"  Params: {n_params:,} | Context: {ctx_bytes}B | Loop iters: {loop_iters}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    step_times = []
    losses = []
    bit_accs = []
    byte_accs = []

    for step in range(STEPS):
        print(f"  starting step {step}...", end='', flush=True)
        t0 = time.time()

        torch.manual_seed(SEED + step + 1000000)
        x, y = loader.sample_batch(
            n_samples=BATCH, seq_len=cfg['seq_len'],
            num_bits=cfg['num_bits'], seed=SEED + step + 1000000,
            binary_bits_mode=True,
        )
        x = x.to(device).to(torch.float64)
        y = y.to(device).to(torch.float64)

        output, stats = model(x, return_stats=True, return_being_outputs=True)
        loss = nn.functional.binary_cross_entropy_with_logits(output, y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        dt = time.time() - t0
        loss_val = loss.item()

        pred_bits = (torch.sigmoid(output) > 0.5).float()
        bit_acc = (pred_bits == y).float().mean().item()
        nb = cfg['num_bits']
        bc = (pred_bits == y).float().reshape(-1, nb // 8, 8)
        byte_acc = bc.all(dim=-1).float().mean().item()
        throughput = bit_acc * nb

        step_times.append(dt)
        losses.append(loss_val)
        bit_accs.append(bit_acc)
        byte_accs.append(byte_acc)

        if step % 5 == 0 or step == STEPS - 1:
            print(f" loss={loss_val:.4f} bit={bit_acc:.4f} byte={byte_acc:.4f} "
                  f"tput={throughput:.1f}b/pos ({dt:.2f}s)")
        else:
            print(f" {dt:.1f}s", flush=True)

        if dt > STEP_TIMEOUT:
            print(f"  TIMEOUT at step {step}")
            break

    # Skip first 3 steps for warmup
    warmup = min(3, len(step_times) - 1)
    avg_time = sum(step_times[warmup:]) / len(step_times[warmup:])

    result = {
        'name': name,
        'label': cfg['label'],
        'D': cfg['D'],
        'depth': cfg['depth'],
        'seq_len': cfg['seq_len'],
        'num_bits': cfg['num_bits'],
        'params': n_params,
        'context_bytes': ctx_bytes,
        'loop_iters': loop_iters,
        'avg_step_time': avg_time,
        'final_loss': losses[-1],
        'final_bit_acc': bit_accs[-1],
        'final_byte_acc': byte_accs[-1],
        'final_throughput': bit_accs[-1] * cfg['num_bits'],
        'best_bit_acc': max(bit_accs),
        'steps_completed': len(step_times),
    }

    in_budget = "YES" if 1.0 <= avg_time <= 5.0 else ("FAST" if avg_time < 1.0 else "SLOW")
    model_size_kb = n_params * 2 / 1024  # fp16

    print(f"\n  RESULT: {avg_time:.2f}s/step [{in_budget}] | "
          f"{n_params:,} params ({model_size_kb:.0f}KB fp16) | "
          f"bit={bit_accs[-1]:.4f} byte={byte_accs[-1]:.4f} tput={bit_accs[-1]*cfg['num_bits']:.1f}")

    return result


def main():
    print("=" * 70)
    print("CPU EDGE BENCHMARK: Find the sweet spot")
    print(f"Target: 1-5s/step | batch={BATCH} | tt=1 | {STEPS} steps each")
    print("=" * 70)

    loader = TraindatLoader(str(DATA_DIR), weights=WEIGHTS)
    all_results = {}

    for name, cfg in CONFIGS.items():
        try:
            result = run_config(name, cfg, loader)
            all_results[name] = result
        except Exception as e:
            print(f"\n  FAILED: {e}")
            import traceback
            traceback.print_exc()
            all_results[name] = {'name': name, 'error': str(e)}

    # Summary table
    print("\n\n" + "=" * 130)
    print("SUMMARY: CPU EDGE SWEET SPOT")
    print("=" * 130)
    hdr = (f"{'Config':<45} {'D':>4} {'dep':>3} {'seq':>3} {'bits':>4} "
           f"{'Ctx':>4} {'Params':>9} {'Size':>6} {'s/step':>7} {'Budget':>6} "
           f"{'BitAcc':>7} {'Tput':>6}")
    print(hdr)
    print("-" * 130)

    for name, r in all_results.items():
        if 'error' in r:
            print(f"{r.get('label', name):<45} FAILED: {r['error'][:60]}")
            continue
        t = r['avg_step_time']
        in_budget = "YES" if 1.0 <= t <= 5.0 else ("FAST" if t < 1.0 else "SLOW")
        size_kb = r['params'] * 2 / 1024
        print(f"{r['label']:<45} {r['D']:>4} {r['depth']:>3} {r['seq_len']:>3} "
              f"{r['num_bits']:>4} {r['context_bytes']:>4}B {r['params']:>9,} "
              f"{size_kb:>5.0f}KB {t:>6.2f}s {in_budget:>6} "
              f"{r['final_bit_acc']:>7.4f} {r['final_throughput']:>6.1f}")

    # Winner
    in_budget = {k: v for k, v in all_results.items()
                 if 'error' not in v and 1.0 <= v['avg_step_time'] <= 5.0}
    if in_budget:
        best = max(in_budget.values(), key=lambda x: x['params'])
        print(f"\n  WINNER (most params in budget): {best['label']}")
        print(f"    {best['avg_step_time']:.2f}s/step | {best['params']:,} params | "
              f"bit={best['final_bit_acc']:.4f} | {best['context_bytes']}B context")

        smartest = max(in_budget.values(), key=lambda x: x['final_throughput'])
        print(f"\n  SMARTEST (best throughput in budget): {smartest['label']}")
        print(f"    {smartest['avg_step_time']:.2f}s/step | throughput={smartest['final_throughput']:.1f} bits/pos")

    # Save
    out_path = DIAMOND_ROOT / "tools" / "_scratch" / "bench_cpu_edge_results.json"
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to: {out_path}")


if __name__ == '__main__':
    main()
