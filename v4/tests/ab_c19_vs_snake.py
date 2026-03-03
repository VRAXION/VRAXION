"""A/B test: C19 vs Snake vs GELU activation — learning quality comparison.

Tests whether the cheaper Snake/GELU activations can match C19's learning
ability on the same echo task. Uses embed_mode (production config) with
byte-level CrossEntropy loss.

Three models trained on the SAME fixed batch with SAME seed:
  1. C19 (original) — learnable per-neuron rho + C
  2. Snake (sin²)   — learnable per-neuron rho + C, 5.7x faster
  3. GELU (baseline) — no learnable params, 12x faster

Metrics: loss curve, accuracy curve, convergence speed, final accuracy.

Usage: python v4/tests/ab_c19_vs_snake.py [--steps 2000]
"""

import sys
import time
import math
import copy
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for subdir in ('model', 'training', 'datagen'):
    p = str(ROOT / subdir)
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Monkey-patchable activation ──────────────────────────────
# We'll swap out the activation function in instnct.py at import time

import instnct as instnct_module

# Save original
_original_c19 = instnct_module._c19_activation

def _snake_activation(x, rho=4.0, C=None):
    """Snake: x + (rho/4) * sin²(C*x). Same learnable params as C19."""
    if C is None:
        C = math.pi
    s = torch.sin(C * x)
    return x + (rho * 0.25) * (s * s)

def _gelu_activation(x, rho=4.0, C=None):
    """GELU wrapper — ignores rho/C (no learnable activation params)."""
    return F.gelu(x)

def _silu_activation(x, rho=4.0, C=None):
    """SiLU/Swish wrapper — ignores rho/C."""
    return F.silu(x)

from instnct import INSTNCT

# ═══════════════════════════════════════════════════════════════
#  Config
# ═══════════════════════════════════════════════════════════════

BATCH = 8
SEQ_LEN = 64
LR = 1e-3
MAX_STEPS = 2000
LOG_EVERY = 100
SEED = 42

# Small but realistic embed-mode model
MODEL_CFG = dict(
    M=64, hidden_dim=128, slot_dim=32, N=1, R=1,
    embed_mode=True,
    kernel_mode='vshape',
    pointer_mode='pilot',
    write_mode='replace',
    embed_encoding='bitlift',
    output_encoding='lowrank_c19',
    checkpoint_chunks=0,
)


def make_echo_batch(batch, seq_len, device, seed=42):
    """Fixed echo batch: 16-byte blocks repeated 4x. embed_mode (byte tokens)."""
    BLOCK = 16
    REPEAT = 4
    rng = np.random.RandomState(seed)

    n_bytes = batch * (seq_len + 1) + BLOCK * REPEAT * 4
    raw_data = []
    raw_mask = []
    while len(raw_data) < n_bytes:
        seed_block = rng.randint(0, 256, size=BLOCK, dtype=np.uint8)
        for r in range(REPEAT):
            raw_data.extend(seed_block)
            raw_mask.extend([0] * BLOCK if r == 0 else [1] * BLOCK)

    raw_data = np.array(raw_data[:n_bytes], dtype=np.uint8)
    raw_mask = np.array(raw_mask[:n_bytes], dtype=np.uint8)

    # Slice into batch
    x_np = np.zeros((batch, seq_len), dtype=np.int64)
    y_np = np.zeros((batch, seq_len), dtype=np.int64)
    mask_np = np.zeros((batch, seq_len), dtype=np.float32)
    for i in range(batch):
        off = i * seq_len
        x_np[i] = raw_data[off:off + seq_len]
        y_np[i] = raw_data[off + 1:off + seq_len + 1]
        mask_np[i] = raw_mask[off + 1:off + seq_len + 1]

    x = torch.from_numpy(x_np).to(device)
    y = torch.from_numpy(y_np).to(device)
    mask = torch.from_numpy(mask_np).to(device)
    return x, y, mask


def train_one(name, activation_fn, x, y, mask, max_steps, device):
    """Train one model variant, return history."""
    # Patch the activation
    instnct_module._c19_activation = activation_fn

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = INSTNCT(**MODEL_CFG).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    opt = torch.optim.Adam(model.parameters(), lr=LR)

    history = []
    t0 = time.perf_counter()

    for step in range(1, max_steps + 1):
        model.train()
        out, _ = model(x)
        # CrossEntropy loss on masked positions only
        logits = out.view(-1, 256)      # (B*T, 256)
        targets = y.view(-1)            # (B*T,)
        m_flat = mask.view(-1)          # (B*T,)

        # Full CE loss
        ce = F.cross_entropy(logits, targets, reduction='none')  # (B*T,)
        # Masked: only supervised positions
        if m_flat.sum() > 0:
            loss = (ce * m_flat).sum() / m_flat.sum()
        else:
            loss = ce.mean()

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()

        # Accuracy
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            correct = (preds == targets).float()
            if m_flat.sum() > 0:
                acc = (correct * m_flat).sum() / m_flat.sum()
            else:
                acc = correct.mean()

        history.append({
            'step': step,
            'loss': loss.item(),
            'acc': acc.item(),
        })

        if step <= 5 or step % LOG_EVERY == 0 or step == max_steps:
            elapsed = time.perf_counter() - t0
            print(f"  [{name:>8s}] step {step:5d}  loss={loss.item():.4f}  "
                  f"acc={acc.item():.4f} ({acc.item()*100:.1f}%)  [{elapsed:.1f}s]")

    elapsed = time.perf_counter() - t0

    # Restore
    instnct_module._c19_activation = _original_c19

    return {
        'name': name,
        'params': n_params,
        'history': history,
        'final_loss': history[-1]['loss'],
        'final_acc': history[-1]['acc'],
        'best_acc': max(h['acc'] for h in history),
        'elapsed': elapsed,
        'steps_to_90': next((h['step'] for h in history if h['acc'] >= 0.90), None),
        'steps_to_95': next((h['step'] for h in history if h['acc'] >= 0.95), None),
    }


def run():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=MAX_STEPS)
    args = parser.parse_args()

    device = 'cpu'
    print("=" * 70)
    print("A/B Test: C19 vs Snake vs GELU — Learning Quality")
    print(f"Config: B={BATCH}, T={SEQ_LEN}, M={MODEL_CFG['M']}, "
          f"H={MODEL_CFG['hidden_dim']}, slot={MODEL_CFG['slot_dim']}, "
          f"N={MODEL_CFG['N']}")
    print(f"Task: Echo pattern (byte-level, embed_mode, CrossEntropy)")
    print(f"Steps: {args.steps}, LR: {LR}, Seed: {SEED}")
    print("=" * 70)

    # Fixed batch
    x, y, mask = make_echo_batch(BATCH, SEQ_LEN, device)
    sup_pct = mask.mean().item() * 100
    print(f"Data: {BATCH}x{SEQ_LEN} bytes, supervised={sup_pct:.1f}%")
    print()

    variants = [
        ('C19', _original_c19),
        ('Snake', _snake_activation),
        ('GELU', _gelu_activation),
        ('SiLU', _silu_activation),
    ]

    results = []
    for name, fn in variants:
        print(f"\n{'─' * 70}")
        print(f"Training: {name}")
        print(f"{'─' * 70}")
        r = train_one(name, fn, x, y, mask, args.steps, device)
        results.append(r)

    # ═══════════════════════════════════════════════════════════
    #  Summary table
    # ═══════════════════════════════════════════════════════════
    print()
    print("=" * 70)
    print("  A/B RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Activation':<12} {'Params':>8} {'Final Loss':>11} {'Final Acc':>10} "
          f"{'Best Acc':>9} {'→90%':>6} {'→95%':>6} {'Time':>7}")
    print("-" * 70)
    for r in results:
        s90 = str(r['steps_to_90']) if r['steps_to_90'] else '—'
        s95 = str(r['steps_to_95']) if r['steps_to_95'] else '—'
        print(f"{r['name']:<12} {r['params']:>8,} {r['final_loss']:>10.4f} "
              f"{r['final_acc']*100:>9.1f}% {r['best_acc']*100:>8.1f}% "
              f"{s90:>6} {s95:>6} {r['elapsed']:>6.1f}s")
    print("-" * 70)

    # Speed comparison
    c19_time = results[0]['elapsed']
    for r in results[1:]:
        speedup = c19_time / r['elapsed']
        acc_diff = (r['final_acc'] - results[0]['final_acc']) * 100
        print(f"  {r['name']} vs C19: {speedup:.1f}x faster, {acc_diff:+.1f}% acc difference")

    # Learning curve at key checkpoints
    print()
    print("Learning curves (accuracy at step N):")
    checkpoints = [50, 100, 200, 500, 1000, args.steps]
    checkpoints = [c for c in checkpoints if c <= args.steps]
    header = f"{'Step':>6}" + "".join(f"  {r['name']:>8}" for r in results)
    print(header)
    for cp in checkpoints:
        vals = []
        for r in results:
            h = r['history'][cp - 1] if cp <= len(r['history']) else r['history'][-1]
            vals.append(f"  {h['acc']*100:>7.1f}%")
        print(f"{cp:>6}" + "".join(vals))

    print("=" * 70)


if __name__ == '__main__':
    run()
