"""Overfit diagnostic — can the model memorize ONE fixed batch?

This is a mathematical convergence test, not a unit test.
Linear model + MSE loss = convex optimization → guaranteed convergence.
When loss stops dropping, the model has reached its THEORETICAL CEILING.

If ceiling accuracy < 100%: the architecture cannot represent this mapping.
If ceiling accuracy = 100%: architecture works, generalization is the issue.

Usage:
    python tests/test_overfit.py              # run from v4/
    python tests/test_overfit.py --steps 2000 # override max steps
"""

import sys
import time
import math
from pathlib import Path

# ── path setup (same as conftest.py) ─────────────────────────
ROOT = Path(__file__).resolve().parent.parent
for subdir in ('model', 'training', 'datagen'):
    p = str(ROOT / subdir)
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch
from instnct import INSTNCT
from train import func_maskloss_mse, func_accuracy_bin

# ═══════════════════════════════════════════════════════════════
#  Config
# ═══════════════════════════════════════════════════════════════

BATCH     = 2
SEQ_LEN   = 32
LR        = 1e-3       # constant — no warmup, no decay (overfit ASAP)
MAX_STEPS = 3000
CONV_EPS  = 1e-7       # loss change threshold for convergence
CONV_WAIT = 100        # consecutive steps below eps = converged
LOG_EVERY = 100

# Tiny model — same architecture, far fewer params → fast CPU iteration.
# The question "can a linear ring memorize?" doesn't depend on size.
MODEL_CFG = dict(M=32, embed_dim=16, N=2, R=1, embed_mode=False)


def make_fixed_batch(device: str):
    """Create one fixed echo-pattern batch — same data every step.

    Pattern: 16-byte random block repeated 8 times.
    Mask: first block = 0 (unsupervised), repeats = 1 (supervised).
    This is exactly what the echo training data contains."""

    BLOCK = 16
    REPEAT = 8
    cycle = BLOCK * REPEAT  # 128 bytes per cycle

    rng = np.random.RandomState(42)  # deterministic

    # generate enough raw echo data
    n_bytes = BATCH * (SEQ_LEN + 1) + cycle  # extra for safety
    raw_data = []
    raw_mask = []
    while len(raw_data) < n_bytes:
        seed_block = rng.randint(0, 256, size=BLOCK, dtype=np.uint8)
        for r in range(REPEAT):
            raw_data.extend(seed_block)
            raw_mask.extend([0] * BLOCK if r == 0 else [1] * BLOCK)

    raw_data = np.array(raw_data[:n_bytes], dtype=np.uint8)
    raw_mask = np.array(raw_mask[:n_bytes], dtype=np.uint8)

    # slice into batch
    data_all = np.zeros((BATCH, SEQ_LEN + 1), dtype=np.uint8)
    mask_all = np.zeros((BATCH, SEQ_LEN + 1), dtype=np.uint8)
    for i in range(BATCH):
        off = i * SEQ_LEN  # deterministic offsets (not random)
        data_all[i] = raw_data[off:off + SEQ_LEN + 1]
        mask_all[i] = raw_mask[off:off + SEQ_LEN + 1]

    # unpack bytes → bits (MSB first, matches train.py)
    flat = np.unpackbits(data_all.reshape(-1))
    bits = flat.reshape(BATCH, SEQ_LEN + 1, 8).astype(np.float32)

    x    = torch.from_numpy(bits[:, :SEQ_LEN].copy()).to(device)
    y    = torch.from_numpy(bits[:, 1:SEQ_LEN + 1].copy()).to(device)
    sup  = mask_all[:, 1:].astype(np.float32)
    mask = torch.from_numpy(sup).unsqueeze(-1).to(device)  # (B, T, 1)

    sup_pct = mask.mean().item() * 100
    return x, y, mask, sup_pct


def run():
    import argparse
    parser = argparse.ArgumentParser(description="Overfit diagnostic")
    parser.add_argument('--steps', type=int, default=MAX_STEPS)
    args = parser.parse_args()

    device = 'cpu'  # diagnostic test — small model, CPU is faster (no kernel launch overhead)
    print(f"Device: {device}")
    print(f"Model:  M={MODEL_CFG['M']}, embed_dim={MODEL_CFG.get('embed_dim', MODEL_CFG.get('D', '?'))}, "
          f"N={MODEL_CFG['N']}, R={MODEL_CFG['R']}")
    print(f"Batch:  {BATCH} × {SEQ_LEN} (binary mode)")
    print(f"LR:     {LR} (constant, no schedule)")
    print(f"Convergence: loss_delta < {CONV_EPS} for {CONV_WAIT} steps")
    print("=" * 65)

    # ── Fixed batch ──
    x, y, mask, sup_pct = make_fixed_batch(device)
    print(f"Supervised positions: {sup_pct:.1f}%")
    print(f"x shape: {tuple(x.shape)}, y shape: {tuple(y.shape)}, "
          f"mask shape: {tuple(mask.shape)}")
    print("=" * 65)

    # ── Model + optimizer ──
    model = INSTNCT(**MODEL_CFG).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    opt = torch.optim.Adam(model.parameters(), lr=LR)

    # ── Training loop — SAME batch every step ──
    prev_loss = float('inf')
    flat_count = 0        # consecutive steps with < eps change
    converged_at = None
    t0 = time.perf_counter()

    history = []  # (step, masked_loss, masked_acc)

    for step in range(1, args.steps + 1):
        pred, _ = model(x)
        raw_loss, masked_loss = func_maskloss_mse(pred, y, mask)

        opt.zero_grad()
        masked_loss.backward()
        opt.step()

        lv = masked_loss.item()
        delta = abs(prev_loss - lv)

        # convergence detection
        if delta < CONV_EPS:
            flat_count += 1
        else:
            flat_count = 0

        if flat_count >= CONV_WAIT and converged_at is None:
            converged_at = step - CONV_WAIT  # first flat step

        # accuracy (no grad)
        with torch.no_grad():
            _, masked_acc = func_accuracy_bin(pred, y, mask)

        history.append((step, lv, masked_acc))
        prev_loss = lv

        # logging
        if step <= 5 or step % LOG_EVERY == 0 or step == args.steps:
            elapsed = time.perf_counter() - t0
            print(f"  step {step:5d}  loss={lv:.8f}  acc={masked_acc:.4f}  "
                  f"delta={delta:.2e}  flat={flat_count}/{CONV_WAIT}  "
                  f"[{elapsed:.1f}s]")

        # early exit if converged
        if converged_at is not None and step >= converged_at + CONV_WAIT + 50:
            print(f"\n  >>> CONVERGED at step {converged_at} — stopping early.")
            break

    # ═══════════════════════════════════════════════════════════
    #  Summary
    # ═══════════════════════════════════════════════════════════
    elapsed = time.perf_counter() - t0
    final_loss = history[-1][1]
    final_acc  = history[-1][2]

    print()
    print("=" * 65)
    print("  OVERFIT DIAGNOSTIC RESULT")
    print("=" * 65)
    print(f"  Steps run:      {len(history)}")
    print(f"  Final loss:     {final_loss:.8f}")
    print(f"  Final accuracy: {final_acc:.4f}  ({final_acc*100:.2f}%)")
    print(f"  Elapsed:        {elapsed:.1f}s")
    print()

    if converged_at is not None:
        print(f"  Converged at step {converged_at}")
    else:
        print(f"  DID NOT CONVERGE in {args.steps} steps")
        print(f"  (loss was still changing — try --steps {args.steps * 3})")

    print()
    if final_acc > 0.99:
        print("  VERDICT: Architecture CAN memorize this batch.")
        print("           Bottleneck is generalization, not capacity.")
    elif final_acc > 0.90:
        print("  VERDICT: Nearly there — architecture is close but not perfect.")
        print("           May need more steps or tuning.")
    elif final_acc > 0.70:
        print("  VERDICT: Partial memorization only.")
        print("           Architecture has LIMITED capacity for this mapping.")
    else:
        print("  VERDICT: Architecture CANNOT memorize even 1 batch.")
        print("           The linear model is fundamentally limited.")
        print("           Nonlinearity (C19 / activation) is REQUIRED.")
    print("=" * 65)


if __name__ == '__main__':
    run()
