"""Isolate which circuit fix causes gradient explosion.

Tests each fix INDIVIDUALLY in replace mode (not additive),
measuring gradient norms. This identifies the culprit.
"""

import sys, time
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
for subdir in ('model', 'training', 'datagen'):
    p = str(ROOT / subdir)
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch
import torch.nn.functional as F
from instnct import INSTNCT

BATCH, SEQ_LEN = 4, 64
SEED = 42
N_PASSES = 5

BASE = dict(
    M=64, hidden_dim=128, slot_dim=32, N=2, R=1,
    embed_mode=True, kernel_mode='vshape', pointer_mode='pilot',
    write_mode='replace',
    embed_encoding='bitlift', output_encoding='lowrank_c19',
    checkpoint_chunks=0,
    ring_decay=False, gate_bias=False,
    expert_output_weights=False, ptr_gradient=False,
)

CONFIGS = [
    ("baseline (no fixes)",           {}),
    ("+ gate_bias only",              {"gate_bias": True}),
    ("+ expert_output_weights only",  {"expert_output_weights": True}),
    ("+ ptr_gradient only",           {"ptr_gradient": True}),
    ("+ ring_decay only (replace)",   {"ring_decay": True}),
    ("+ additive only (no decay)",    {"write_mode": "additive"}),
    ("+ additive + decay",            {"write_mode": "additive", "ring_decay": True}),
    ("+ ALL fixes (replace)",         {"gate_bias": True, "expert_output_weights": True,
                                       "ptr_gradient": True, "ring_decay": True}),
    ("+ ALL fixes (additive+decay)",  {"gate_bias": True, "expert_output_weights": True,
                                       "ptr_gradient": True, "ring_decay": True,
                                       "write_mode": "additive"}),
]


def make_batch(seed, device='cpu'):
    rng = np.random.RandomState(seed)
    BLOCK, REPEAT = 16, 4
    n_bytes = BATCH * (SEQ_LEN + 1) + BLOCK * REPEAT * 4
    raw_data, raw_mask = [], []
    while len(raw_data) < n_bytes:
        sb = rng.randint(0, 256, size=BLOCK, dtype=np.uint8)
        for r in range(REPEAT):
            raw_data.extend(sb)
            raw_mask.extend([0] * BLOCK if r == 0 else [1] * BLOCK)
    raw_data = np.array(raw_data[:n_bytes], dtype=np.uint8)
    raw_mask = np.array(raw_mask[:n_bytes], dtype=np.uint8)
    x_np = np.zeros((BATCH, SEQ_LEN), dtype=np.int64)
    y_np = np.zeros((BATCH, SEQ_LEN), dtype=np.int64)
    mask_np = np.zeros((BATCH, SEQ_LEN), dtype=np.float32)
    for i in range(BATCH):
        off = i * SEQ_LEN
        x_np[i] = raw_data[off:off + SEQ_LEN]
        y_np[i] = raw_data[off + 1:off + SEQ_LEN + 1]
        mask_np[i] = raw_mask[off + 1:off + SEQ_LEN + 1]
    return (torch.from_numpy(x_np).to(device),
            torch.from_numpy(y_np).to(device),
            torch.from_numpy(mask_np).to(device))


def measure(cfg_dict):
    """Average gradient total norm over N_PASSES."""
    norms = []
    for i in range(N_PASSES):
        torch.manual_seed(SEED)
        model = INSTNCT(**cfg_dict)
        x, y, mask = make_batch(seed=SEED + i)

        model.zero_grad()
        out, _ = model(x)
        logits = out.view(-1, 256)
        targets = y.view(-1)
        m_flat = mask.view(-1)
        ce = F.cross_entropy(logits, targets, reduction='none')
        loss = (ce * m_flat).sum() / m_flat.sum() if m_flat.sum() > 0 else ce.mean()
        loss.backward()

        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.norm().item() ** 2
        total_norm = total_norm ** 0.5
        norms.append(total_norm)

    return np.mean(norms), np.std(norms)


def main():
    print("=" * 72)
    print("  GRADIENT EXPLOSION ISOLATION TEST")
    print("  Each fix tested individually in replace mode")
    print(f"  Averaged over {N_PASSES} passes")
    print("=" * 72)
    print()
    print(f"  {'Config':<40} {'Mean‖g‖':>14} {'Std':>12} {'Ratio':>10}")
    print(f"  {'-'*40} {'-'*14} {'-'*12} {'-'*10}")

    baseline_norm = None
    for label, overrides in CONFIGS:
        cfg = {**BASE, **overrides}
        t0 = time.perf_counter()
        mean_n, std_n = measure(cfg)
        elapsed = time.perf_counter() - t0

        if baseline_norm is None:
            baseline_norm = mean_n
            ratio_str = "1.00x"
        else:
            ratio = mean_n / baseline_norm if baseline_norm > 1e-12 else float('inf')
            if ratio > 1000:
                ratio_str = f"{ratio:.0e}x"
            else:
                ratio_str = f"{ratio:.2f}x"

        if mean_n > 1e6:
            norm_str = f"{mean_n:.2e}"
            std_str = f"{std_n:.2e}"
        else:
            norm_str = f"{mean_n:.4f}"
            std_str = f"{std_n:.4f}"

        flag = ""
        if mean_n > baseline_norm * 100 and baseline_norm is not None:
            flag = " ← EXPLODES!"
        elif mean_n > baseline_norm * 10 and baseline_norm is not None:
            flag = " ← elevated"

        print(f"  {label:<40} {norm_str:>14} {std_str:>12} {ratio_str:>10}{flag}")

    print()
    print("=" * 72)


if __name__ == '__main__':
    main()
