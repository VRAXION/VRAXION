"""Ablation: R=1 (current) vs R=32 (half-ring) with V-shaped weights.
Phase ON, tanh, depth removed. Tests whether wider view helps.

Usage: python tests/bench_R_ablation.py
"""

import sys, time, math, numpy as np, torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for subdir in ('model', 'training', 'datagen'):
    p = str(ROOT / subdir)
    if p not in sys.path:
        sys.path.insert(0, p)

from instnct import INSTNCT
from train import func_maskloss_mse, func_accuracy_bin

BATCH     = 4
LR        = 1e-3
STEPS     = 3000
EVAL_EVERY = 200
SEED      = 42
EVAL_SEED = 9999

# ── Data generators ──

def _bytes_to_bits(data_np, mask_np, batch_size, seq_len, device):
    data_all = np.zeros((batch_size, seq_len + 1), dtype=np.uint8)
    mask_all = np.zeros((batch_size, seq_len + 1), dtype=np.uint8)
    for i in range(batch_size):
        data_all[i, :len(data_np[i])] = data_np[i][:seq_len + 1]
        mask_all[i, :len(mask_np[i])] = mask_np[i][:seq_len + 1]
    flat = np.unpackbits(data_all.reshape(-1))
    bits = flat.reshape(batch_size, seq_len + 1, 8).astype(np.float32)
    x    = torch.from_numpy(bits[:, :seq_len].copy()).to(device)
    y    = torch.from_numpy(bits[:, 1:seq_len + 1].copy()).to(device)
    sup  = mask_all[:, 1:seq_len + 1].astype(np.float32)
    mask = torch.from_numpy(sup).unsqueeze(-1).to(device)
    return x, y, mask

def make_xor_batch(batch_size, seq_len, device, rng):
    n_triplets = (seq_len + 1) // 3 + 2
    data, masks = [], []
    for _ in range(batch_size):
        d, m = [], []
        for _ in range(n_triplets):
            a, b = rng.randint(0, 256), rng.randint(0, 256)
            d.extend([a, b, a ^ b]); m.extend([0, 0, 1])
        data.append(np.array(d[:seq_len+1], dtype=np.uint8))
        masks.append(np.array(m[:seq_len+1], dtype=np.uint8))
    return _bytes_to_bits(data, masks, batch_size, seq_len, device)

def make_delayed_xor_batch(batch_size, seq_len, device, rng, delay=8):
    unit = delay + 3
    n_units = (seq_len + 1) // unit + 2
    data, masks = [], []
    for _ in range(batch_size):
        d, m = [], []
        for _ in range(n_units):
            a, b = rng.randint(0, 256), rng.randint(0, 256)
            d.append(a); m.append(0)
            d.append(b); m.append(0)
            for _ in range(delay):
                d.append(rng.randint(0, 256)); m.append(0)
            d.append(a ^ b); m.append(1)
        data.append(np.array(d[:seq_len+1], dtype=np.uint8))
        masks.append(np.array(m[:seq_len+1], dtype=np.uint8))
    return _bytes_to_bits(data, masks, batch_size, seq_len, device)

def make_chained_xor_batch(batch_size, seq_len, device, rng):
    data, masks = [], []
    for _ in range(batch_size):
        d, m = [], []
        while len(d) < seq_len + 1:
            a, b = rng.randint(0, 256), rng.randint(0, 256)
            acc = a ^ b
            d.extend([a, b, acc]); m.extend([0, 0, 1])
            for _ in range(2):
                c = rng.randint(0, 256); acc ^= c
                d.extend([c, acc]); m.extend([0, 1])
            d.append(rng.randint(0, 256)); m.append(0)
        data.append(np.array(d[:seq_len+1], dtype=np.uint8))
        masks.append(np.array(m[:seq_len+1], dtype=np.uint8))
    return _bytes_to_bits(data, masks, batch_size, seq_len, device)

TASKS = {
    'xor':     {'fn': make_xor_batch,         'seq': 36, 'label': 'XOR'},
    'delayed': {'fn': make_delayed_xor_batch,  'seq': 44, 'label': 'Delayed XOR'},
    'chained': {'fn': make_chained_xor_batch,  'seq': 48, 'label': 'Chained XOR'},
}

def run_one(task_info, R_value, device='cpu'):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    cfg = dict(M=64, embed_dim=64, N=2, R=R_value, embed_mode=False)
    model = INSTNCT(**cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    eval_rng = np.random.RandomState(EVAL_SEED)
    ex, ey, emask = task_info['fn'](8, task_info['seq'], device, eval_rng)
    train_rng = np.random.RandomState(SEED)

    best_eval = 0.0
    t0 = time.perf_counter()

    for step in range(1, STEPS + 1):
        x, y, mask = task_info['fn'](BATCH, task_info['seq'], device, train_rng)
        pred, _ = model(x)
        _, loss = func_maskloss_mse(pred, y, mask)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % EVAL_EVERY == 0 or step == STEPS:
            with torch.no_grad():
                epred, _ = model(ex)
                _, acc = func_accuracy_bin(epred, ey, emask)
            if acc > best_eval:
                best_eval = acc

    wall = time.perf_counter() - t0
    return best_eval, wall, n_params

def main():
    print("=" * 70)
    print("  ABLATION: R=1 vs R=32 (V-shaped weights, phase ON)")
    print("=" * 70)
    print(f"  Steps: {STEPS} | Batch: {BATCH} | Seed: {SEED}")
    print()

    for task_name, task_info in TASKS.items():
        print(f"  {task_info['label']}:")
        for R_val in [1, 32]:
            print(f"    R={R_val:>2}...", end='', flush=True)
            acc, wall, n_params = run_one(task_info, R_val)
            print(f"  {acc*100:.2f}%  ({wall:.0f}s, {n_params} params)")
        print()

    print("=" * 70)

if __name__ == '__main__':
    main()
