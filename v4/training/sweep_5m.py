#!/usr/bin/env python3
"""Sweep: 5M-param INSTNCT configurations on real code data.

Measures two things:
  1. Intelligence — masked CE loss + masked accuracy on Python code
  2. Cost — steps/sec (wall-clock throughput)

Runs 5 configs sequentially, 500 steps each, same data/seed/hyperparams.
Produces a summary CSV + comparison table at the end.
"""

import copy
import csv
import os
import sys
import time
import yaml
from pathlib import Path

# ── Setup paths ──────────────────────────────────────────────
V4_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = V4_ROOT / 'model'
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

# Import training components
sys.path.insert(0, str(V4_ROOT / 'training'))
from model_factory import build_model_spec, build_model_from_spec

import numpy as np
import torch
import torch.nn.functional as F

# Reuse train.py's dataset and loss
from train import ByteDataset, func_discover_dat, func_maskloss_ce, func_accuracy_emb


# ── Sweep Configurations ─────────────────────────────────────
# Each overrides the base YAML model config for that run.
# All target ~5M params with embed_mode=True, output_encoding=lowrank_c19.

SWEEP_CONFIGS = {
    'A_wide_H8192': {
        'label': 'A: Wide (H=8192, S=128, N=1)',
        'hidden_dim': 8192,
        'slot_dim': 128,
        'N': 1,
        'M': 128,
    },
    'B_2expert_H5888': {
        'label': 'B: 2-Expert (H=5888, S=128, N=2)',
        'hidden_dim': 5888,
        'slot_dim': 128,
        'N': 2,
        'M': 128,
    },
    'C_wideslot_H6144': {
        'label': 'C: Wide Slots (H=6144, S=256, N=1)',
        'hidden_dim': 6144,
        'slot_dim': 256,
        'N': 1,
        'M': 128,
    },
    'D_4expert_H3072': {
        'label': 'D: 4-Expert (H=3072, S=128, N=4)',
        'hidden_dim': 3072,
        'slot_dim': 128,
        'N': 4,
        'M': 256,
    },
    'E_baseline_H4096': {
        'label': 'E: Baseline (H=4096, S=128, N=1) [current]',
        'hidden_dim': 4096,
        'slot_dim': 128,
        'N': 1,
        'M': 128,
    },
}

# ── Training params (same for all) ──────────────────────────
STEPS = 500
BATCH_SIZE = 8
SEQ_LEN = 256
LR = 3e-4
WARMUP = 50
SEED = 1337
DEVICE = 'cpu'
LOG_EVERY = 50
DATA_DIR = '/tmp/sweep_data'


def load_base_model_config():
    """Load base model config from YAML."""
    cfg_path = V4_ROOT / 'config' / 'vraxion_config.yaml'
    with open(cfg_path) as f:
        root = yaml.safe_load(f)
    return root['model']


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_single_config(name, overrides, base_cfg, dataset):
    """Train one config and return metrics."""
    print(f"\n{'='*70}")
    print(f"  {overrides['label']}")
    print(f"{'='*70}")

    # Build model config with overrides
    model_cfg = copy.deepcopy(base_cfg)
    for k, v in overrides.items():
        if k != 'label':
            model_cfg[k] = v

    # Seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Build model
    model_record = build_model_spec(
        model_type='instnct',
        embed_mode=True,
        model_config=model_cfg,
        training_config={},
    )
    model = build_model_from_spec(model_record, device=DEVICE)
    n_params = count_params(model)
    print(f"  Params: {n_params:,} ({n_params/1e6:.2f}M)")

    # Optimizer
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    # Reset dataset RNG for fairness
    dataset.rng = np.random.default_rng(SEED)

    # Training loop
    model.train()
    losses = []
    accs = []
    step_times = []

    # Init sequence state
    ring = None
    hidden = None
    pointer = None

    for step in range(1, STEPS + 1):
        t0 = time.perf_counter()

        x, y, mask = dataset.sample_batch(BATCH_SIZE, DEVICE)

        # Forward — pass and receive sequence state
        if ring is not None:
            out = model(x, ring=ring.detach(), hidden=hidden.detach(),
                       pointer=pointer.detach())
        else:
            out = model(x)

        # Unpack — model returns (pred, ring, hidden, pointer) or just pred
        if isinstance(out, tuple):
            pred = out[0]
            if len(out) >= 4:
                ring, hidden, pointer = out[1], out[2], out[3]
        else:
            pred = out

        # Loss
        raw_loss, masked_loss = func_maskloss_ce(pred, y, mask)

        # Backward
        opt.zero_grad()

        # Warmup LR
        if step <= WARMUP:
            lr_now = LR * step / WARMUP
            for pg in opt.param_groups:
                pg['lr'] = lr_now

        masked_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()

        dt = time.perf_counter() - t0
        step_times.append(dt)

        # Accuracy
        with torch.no_grad():
            raw_acc, masked_acc = func_accuracy_emb(pred, y, mask)

        losses.append(masked_loss.item())
        accs.append(masked_acc)

        if step % LOG_EVERY == 0 or step == 1:
            avg_loss = np.mean(losses[-LOG_EVERY:])
            avg_acc = np.mean(accs[-LOG_EVERY:])
            avg_dt = np.mean(step_times[-LOG_EVERY:])
            print(f"  step {step:4d} | loss {avg_loss:.4f} | acc {avg_acc:.3f} | "
                  f"{1/avg_dt:.1f} step/s | {avg_dt:.3f}s/step")

    # Final metrics (last 100 steps average)
    final_loss = np.mean(losses[-100:])
    final_acc = np.mean(accs[-100:])
    avg_step_time = np.mean(step_times[WARMUP:])  # exclude warmup
    steps_per_sec = 1.0 / avg_step_time

    # "Intelligence per cost" = accuracy / seconds_per_step
    # Higher = better (more learning per unit of compute)
    efficiency = final_acc / avg_step_time

    result = {
        'name': name,
        'label': overrides['label'],
        'params': n_params,
        'final_loss': final_loss,
        'final_acc': final_acc,
        'step_time': avg_step_time,
        'steps_per_sec': steps_per_sec,
        'efficiency': efficiency,
        'loss_at_100': np.mean(losses[80:100]) if len(losses) >= 100 else losses[-1],
        'loss_at_250': np.mean(losses[230:250]) if len(losses) >= 250 else losses[-1],
        'loss_at_500': np.mean(losses[480:500]) if len(losses) >= 500 else losses[-1],
    }

    print(f"\n  RESULT: loss={final_loss:.4f} acc={final_acc:.3f} "
          f"speed={steps_per_sec:.2f} step/s  efficiency={efficiency:.3f}")

    # Clean up
    del model, opt, ring, hidden, pointer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return result


def main():
    print("INSTNCT 5M Parameter Sweep")
    print(f"Data: {DATA_DIR} | Steps: {STEPS} | Batch: {BATCH_SIZE} | Seq: {SEQ_LEN}")
    print(f"LR: {LR} | Warmup: {WARMUP} | Seed: {SEED} | Device: {DEVICE}")

    # Load data
    files = func_discover_dat(DATA_DIR)
    dataset = ByteDataset(files, SEQ_LEN, embed_mode=True, seed=SEED)
    print(f"Dataset: {dataset.total_bytes:,} bytes, {dataset.n_samples:,} samples")

    # Load base config
    base_cfg = load_base_model_config()

    # Run sweep
    results = []
    for name, overrides in SWEEP_CONFIGS.items():
        result = run_single_config(name, overrides, base_cfg, dataset)
        results.append(result)

    # ── Summary ──────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("  SWEEP RESULTS SUMMARY")
    print("=" * 90)

    # Sort by final loss (lower = more intelligent)
    results.sort(key=lambda r: r['final_loss'])

    print(f"\n{'Config':<45} {'Params':>8} {'Loss':>7} {'Acc':>6} {'Step/s':>7} {'Eff':>7}")
    print("-" * 90)
    for r in results:
        print(f"{r['label']:<45} {r['params']/1e6:>7.2f}M {r['final_loss']:>7.4f} "
              f"{r['final_acc']:>5.1%} {r['steps_per_sec']:>7.2f} {r['efficiency']:>7.3f}")

    print(f"\n  Eff = acc / step_time  (higher = more intelligence per compute)")

    # Loss progression table
    print(f"\n{'Config':<45} {'@100':>8} {'@250':>8} {'@500':>8} {'Delta':>8}")
    print("-" * 90)
    for r in results:
        delta = r['loss_at_100'] - r['loss_at_500']
        print(f"{r['label']:<45} {r['loss_at_100']:>8.4f} {r['loss_at_250']:>8.4f} "
              f"{r['loss_at_500']:>8.4f} {delta:>+8.4f}")

    # Best picks
    best_intel = min(results, key=lambda r: r['final_loss'])
    best_speed = max(results, key=lambda r: r['steps_per_sec'])
    best_eff = max(results, key=lambda r: r['efficiency'])

    print(f"\n  Best intelligence: {best_intel['label']} (loss={best_intel['final_loss']:.4f})")
    print(f"  Best speed:        {best_speed['label']} ({best_speed['steps_per_sec']:.2f} step/s)")
    print(f"  Best efficiency:   {best_eff['label']} (eff={best_eff['efficiency']:.3f})")

    # Save CSV
    out_csv = V4_ROOT / 'training' / 'sweep_5m_results.csv'
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    print(f"\n  Results saved: {out_csv}")


if __name__ == '__main__':
    main()
