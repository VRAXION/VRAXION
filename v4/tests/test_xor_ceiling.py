"""XOR ceiling test — proves whether nonlinearity is needed.

XOR is the simplest nonlinear function: output = input_A XOR input_B.
No linear model can GENERALIZE XOR — it can memorize fixed pairs but
cannot predict A^B for unseen A,B combinations.

Test design: RANDOM A,B pairs every step (no memorization possible).
Eval on a separate held-out set every N steps.

Usage:
    python tests/test_xor_ceiling.py
"""

import sys
import time
from pathlib import Path

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

BATCH     = 4
SEQ_LEN   = 36           # divisible by 3 (A, B, A^B triplets)
LR        = 1e-3
MAX_STEPS = 3000
LOG_EVERY = 100
EVAL_EVERY = 200

MODEL_CFG = dict(M=32, embed_dim=16, N=2, R=1, embed_mode=False)


def make_random_xor_batch(batch_size, seq_len, device, rng=None):
    """RANDOM XOR batch — new A,B pairs every call. No memorization possible."""
    if rng is None:
        rng = np.random

    n_triplets = (seq_len + 1) // 3 + 2
    total_bytes = n_triplets * 3

    raw_data = np.empty(total_bytes * batch_size, dtype=np.uint8)
    raw_mask = np.empty(total_bytes * batch_size, dtype=np.uint8)

    for s in range(batch_size):
        off = s * total_bytes
        for t in range(n_triplets):
            a = rng.randint(0, 256)
            b = rng.randint(0, 256)
            base = off + t * 3
            raw_data[base]     = a
            raw_data[base + 1] = b
            raw_data[base + 2] = a ^ b
            raw_mask[base]     = 0
            raw_mask[base + 1] = 0
            raw_mask[base + 2] = 1

    data_all = np.zeros((batch_size, seq_len + 1), dtype=np.uint8)
    mask_all = np.zeros((batch_size, seq_len + 1), dtype=np.uint8)
    for i in range(batch_size):
        off = i * total_bytes
        data_all[i] = raw_data[off:off + seq_len + 1]
        mask_all[i] = raw_mask[off:off + seq_len + 1]

    flat = np.unpackbits(data_all.reshape(-1))
    bits = flat.reshape(batch_size, seq_len + 1, 8).astype(np.float32)

    x    = torch.from_numpy(bits[:, :seq_len].copy()).to(device)
    y    = torch.from_numpy(bits[:, 1:seq_len + 1].copy()).to(device)
    sup  = mask_all[:, 1:].astype(np.float32)
    mask = torch.from_numpy(sup).unsqueeze(-1).to(device)

    return x, y, mask


def run():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=MAX_STEPS)
    args = parser.parse_args()

    device = 'cpu'

    print(f"Device: {device}")
    print(f"Model:  M={MODEL_CFG['M']}, D={MODEL_CFG['D']}, "
          f"N={MODEL_CFG['N']}, R={MODEL_CFG['R']}")
    print(f"Task:   XOR generalization — RANDOM A,B every step")
    print(f"Batch:  {BATCH} x {SEQ_LEN} (binary mode)")
    print(f"Theory: linear ceiling = 50% (random guess)")
    print("=" * 65)

    model = INSTNCT(**MODEL_CFG).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # fixed eval set — same held-out data every eval
    eval_rng = np.random.RandomState(9999)
    ex, ey, emask = make_random_xor_batch(8, SEQ_LEN, device, eval_rng)
    print(f"Eval set: {8} x {SEQ_LEN}, supervised={emask.mean().item()*100:.1f}%")
    print("=" * 65)

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    train_rng = np.random.RandomState(42)
    t0 = time.perf_counter()
    best_eval_acc = 0.0

    for step in range(1, args.steps + 1):
        # ── RANDOM batch every step ──
        x, y, mask = make_random_xor_batch(BATCH, SEQ_LEN, device, train_rng)

        pred, _ = model(x)
        _, masked_loss = func_maskloss_mse(pred, y, mask)

        opt.zero_grad()
        masked_loss.backward()
        opt.step()

        lv = masked_loss.item()

        # ── Eval on held-out set ──
        if step <= 5 or step % EVAL_EVERY == 0 or step == args.steps:
            with torch.no_grad():
                epred, _ = model(ex)
                _, eval_acc = func_accuracy_bin(epred, ey, emask)
            if eval_acc > best_eval_acc:
                best_eval_acc = eval_acc

        if step <= 5 or step % LOG_EVERY == 0 or step == args.steps:
            elapsed = time.perf_counter() - t0
            with torch.no_grad():
                _, train_acc = func_accuracy_bin(pred, y, mask)
            print(f"  step {step:5d}  train_loss={lv:.6f}  "
                  f"train_acc={train_acc:.4f}  eval_acc={eval_acc:.4f}  "
                  f"[{elapsed:.1f}s]")

    elapsed = time.perf_counter() - t0

    print()
    print("=" * 65)
    print("  XOR GENERALIZATION TEST RESULT")
    print("=" * 65)
    print(f"  Steps run:       {args.steps}")
    print(f"  Final train acc: {train_acc:.4f}  ({train_acc*100:.2f}%)")
    print(f"  Final eval acc:  {eval_acc:.4f}  ({eval_acc*100:.2f}%)")
    print(f"  Best eval acc:   {best_eval_acc:.4f}  ({best_eval_acc*100:.2f}%)")
    print(f"  Elapsed:         {elapsed:.1f}s")
    print()

    if best_eval_acc > 0.65:
        print("  VERDICT: Model GENERALIZES XOR beyond linear ceiling.")
        print("           (unexpected for a linear model — investigate)")
    elif best_eval_acc > 0.55:
        print("  VERDICT: Slightly above 50% — likely noise, not learning.")
    else:
        print("  VERDICT: Model STUCK at ~50% (random guess) on held-out data.")
        print("           Linear model CANNOT generalize XOR.")
        print("           Nonlinearity (C19) is REQUIRED for nonlinear tasks.")
    print("=" * 65)


if __name__ == '__main__':
    run()
