"""LEGACY BENCH (archived 2026-02-28)
Reason: depends on removed learnable-R API (R_param).
Status: not part of active CI/runtime validation.

Original: Quick R sweep: R=0 (needle), R=1, R=2, R=3.

Forces R_eff by overriding learnable R_param with frozen values.
V-shape kernel: R_eff = R + 0.5 gives window of exactly 2R+1 slots.

Usage: python tests/bench_R_sweep_0123.py
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

BATCH      = 4
LR         = 1e-3
STEPS      = 3000
EVAL_EVERY = 200
SEED       = 42
EVAL_SEED  = 9999
M          = 64

# ── Data generators (same as bench_R_ablation) ──

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


def force_R_eff(model, R_desired):
    """Override learnable R_param to force exact R_eff = R_desired + 0.5.

    V-shape kernel: weight = (1 - |offset| / R_eff).clamp(min=0)
    R_eff = R + 0.5 gives window of exactly 2R+1 slots:
      R=0 → R_eff=0.5 → 1 slot (needle)
      R=1 → R_eff=1.5 → 3 slots
      R=2 → R_eff=2.5 → 5 slots
      R=3 → R_eff=3.5 → 7 slots
    """
    R_eff_target = R_desired + 0.5
    R_scale = M // 2  # sigmoid(R_param) * R_scale = R_eff
    ratio = R_eff_target / R_scale
    # inverse sigmoid: logit(p) = log(p / (1-p))
    R_param_val = math.log(ratio / (1.0 - ratio))
    with torch.no_grad():
        model.R_param.fill_(R_param_val)
    model.R_param.requires_grad_(False)


def run_one(task_info, R_value, device='cpu'):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    cfg = dict(M=M, embed_dim=64, N=2, R=1, embed_mode=False)
    model = INSTNCT(**cfg).to(device)
    force_R_eff(model, R_value)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
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
    R_values = [0, 1, 2, 3]
    window_sizes = {0: 1, 1: 3, 2: 5, 3: 7}

    print("=" * 70)
    print("  R SWEEP: R=0 (needle) | R=1 (3 slots) | R=2 (5 slots) | R=3 (7 slots)")
    print("=" * 70)
    print(f"  Steps: {STEPS} | Batch: {BATCH} | Seed: {SEED} | M: {M}")
    print(f"  Kernel: vshape | R_param: frozen")
    print()

    results = {}
    for task_name, task_info in TASKS.items():
        print(f"  {task_info['label']}:")
        results[task_name] = {}
        for R_val in R_values:
            print(f"    R={R_val} ({window_sizes[R_val]} slots)...", end='', flush=True)
            acc, wall, n_params = run_one(task_info, R_val)
            results[task_name][R_val] = acc
            print(f"  {acc*100:.2f}%  ({wall:.0f}s)")
        print()

    # ── Summary table ──
    print("=" * 70)
    print("  SUMMARY")
    print("-" * 70)
    print(f"  {'R':>3}  {'Window':>6}  {'XOR':>8}  {'Delayed':>8}  {'Chained':>8}  {'GeoMean':>8}")
    print("-" * 70)
    for R_val in R_values:
        accs = [results[t][R_val] for t in TASKS]
        gmean = np.prod(accs) ** (1.0 / len(accs))
        print(f"  {R_val:>3}  {window_sizes[R_val]:>6}  "
              f"{accs[0]*100:>7.2f}%  {accs[1]*100:>7.2f}%  {accs[2]*100:>7.2f}%  "
              f"{gmean*100:>7.2f}%")
    print("=" * 70)


if __name__ == '__main__':
    main()
