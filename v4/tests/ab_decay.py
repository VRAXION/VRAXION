"""A/B test: decay=0.999 vs decay=1.0 (no decay, pure additive)
Runs 800 steps each (~90s per arm), prints side-by-side comparison."""

import sys, time, functools
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for subdir in ('model', 'training', 'datagen'):
    p = str(ROOT / subdir)
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch
import instnct as instnct_mod
from instnct import INSTNCT
from train import func_maskloss_mse, func_accuracy_bin

# ── config ──
BATCH, SEQ_LEN, STEPS, LR = 2, 32, 800, 1e-3
MODEL_CFG = dict(M=32, embed_dim=16, N=2, R=1, embed_mode=False)
SEED = 42


def make_fixed_batch(device='cpu'):
    BLOCK, REPEAT = 16, 8
    rng = np.random.RandomState(SEED)
    n_bytes = BATCH * (SEQ_LEN + 1) + BLOCK * REPEAT
    raw_data, raw_mask = [], []
    while len(raw_data) < n_bytes:
        seed_block = rng.randint(0, 256, size=BLOCK, dtype=np.uint8)
        for r in range(REPEAT):
            raw_data.extend(seed_block)
            raw_mask.extend([0] * BLOCK if r == 0 else [1] * BLOCK)
    raw_data = np.array(raw_data[:n_bytes], dtype=np.uint8)
    raw_mask = np.array(raw_mask[:n_bytes], dtype=np.uint8)
    data_all = np.zeros((BATCH, SEQ_LEN + 1), dtype=np.uint8)
    mask_all = np.zeros((BATCH, SEQ_LEN + 1), dtype=np.uint8)
    for i in range(BATCH):
        off = i * SEQ_LEN
        data_all[i] = raw_data[off:off + SEQ_LEN + 1]
        mask_all[i] = raw_mask[off:off + SEQ_LEN + 1]
    flat = np.unpackbits(data_all.reshape(-1))
    bits = flat.reshape(BATCH, SEQ_LEN + 1, 8).astype(np.float32)
    x = torch.from_numpy(bits[:, :SEQ_LEN].copy()).to(device)
    y = torch.from_numpy(bits[:, 1:SEQ_LEN + 1].copy()).to(device)
    sup = mask_all[:, 1:].astype(np.float32)
    mask = torch.from_numpy(sup).unsqueeze(-1).to(device)
    return x, y, mask


def run_arm(decay_val, x, y, mask):
    """Train with a specific decay value, return history [(step, loss, acc)]."""
    # Monkey-patch the decay default
    original_fn = instnct_mod.func_additive_write_tns

    @functools.wraps(original_fn)
    def patched_fn(ring_tns, write_vec_tns, expanded_idx_tns, weights_tns,
                   write_strength=None, decay=decay_val):
        return original_fn(ring_tns, write_vec_tns, expanded_idx_tns,
                           weights_tns, write_strength=write_strength,
                           decay=decay)

    instnct_mod.func_additive_write_tns = patched_fn

    torch.manual_seed(SEED)
    model = INSTNCT(**MODEL_CFG)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    history = []

    t0 = time.perf_counter()
    for step in range(1, STEPS + 1):
        pred, _ = model(x)
        _, masked_loss = func_maskloss_mse(pred, y, mask)
        opt.zero_grad()
        masked_loss.backward()
        opt.step()
        with torch.no_grad():
            _, masked_acc = func_accuracy_bin(pred, y, mask)
        history.append((step, masked_loss.item(), masked_acc))
    elapsed = time.perf_counter() - t0

    # Restore original
    instnct_mod.func_additive_write_tns = original_fn
    return history, elapsed


def main():
    x, y, mask = make_fixed_batch()
    configs = [
        ("decay=0.999", 0.999),
        ("decay=1.0 (no decay)", 1.0),
    ]

    results = {}
    for label, decay_val in configs:
        print(f"Running: {label} ...")
        hist, elapsed = run_arm(decay_val, x, y, mask)
        results[label] = (hist, elapsed)
        print(f"  done in {elapsed:.1f}s  |  final loss={hist[-1][1]:.6f}  acc={hist[-1][2]:.4f}")
        print()

    # ── Side-by-side table ──
    print("=" * 72)
    print(f"  {'Step':>5}  |  {'decay=0.999':^22}  |  {'decay=1.0':^22}")
    print(f"  {'':>5}  |  {'loss':>10}  {'acc':>8}   |  {'loss':>10}  {'acc':>8}")
    print("-" * 72)

    h1 = results[configs[0][0]][0]
    h2 = results[configs[1][0]][0]

    checkpoints = [1, 5, 50, 100, 200, 300, 400, 500, 600, 700, 800]
    for s in checkpoints:
        if s <= STEPS:
            _, l1, a1 = h1[s - 1]
            _, l2, a2 = h2[s - 1]
            print(f"  {s:5d}  |  {l1:10.6f}  {a1:8.4f}   |  {l2:10.6f}  {a2:8.4f}")

    print("=" * 72)

    # ── Winner ──
    final_acc_1 = h1[-1][2]
    final_acc_2 = h2[-1][2]
    diff = final_acc_1 - final_acc_2
    if abs(diff) < 0.01:
        print(f"\n  RESULT: Effectively tied (diff = {diff:+.4f})")
    elif diff > 0:
        print(f"\n  RESULT: decay=0.999 WINS by {diff:+.4f} acc")
    else:
        print(f"\n  RESULT: no decay WINS by {-diff:+.4f} acc")

    # ── Ring norm check ──
    print("\n  Ring norm at final step:")
    for label, decay_val in configs:
        torch.manual_seed(SEED)
        instnct_mod.func_additive_write_tns.__defaults__ = (None, decay_val)
        model = INSTNCT(**MODEL_CFG)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        for _ in range(STEPS):
            pred, _ = model(x)
            _, ml = func_maskloss_mse(pred, y, mask)
            opt.zero_grad(); ml.backward(); opt.step()
        with torch.no_grad():
            pred, aux = model(x)
            if hasattr(aux, '__len__') and len(aux) > 0:
                print(f"    {label}: aux available")
            # Just check model ring state indirectly via output magnitude
            print(f"    {label}: output norm = {pred.norm().item():.4f}")
    instnct_mod.func_additive_write_tns.__defaults__ = (None, 0.999)


if __name__ == '__main__':
    main()
