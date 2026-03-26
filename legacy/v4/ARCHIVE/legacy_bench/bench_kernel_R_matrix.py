"""LEGACY BENCH (archived 2026-02-28)
Reason: depends on removed learnable-R API (R_param).
Status: not part of active CI/runtime validation.

Original: 3x3 Kernel Shape x R Strategy -- exhaustive benchmark.

Axis 1 -- Kernel: uniform, vshape, gaussian
Axis 2 -- R strategy: fixed R~1, fixed R~M/2, learnable R

9 conditions x 3 tasks x 3000 steps = 27 training runs.
Phase ON, tanh activation, no depth. Deterministic seed.

Usage: python tests/bench_kernel_R_matrix.py
"""

import sys, time, math, numpy as np, torch
from pathlib import Path
from collections import OrderedDict

ROOT = Path(__file__).resolve().parent.parent
for subdir in ('model', 'training', 'datagen'):
    p = str(ROOT / subdir)
    if p not in sys.path:
        sys.path.insert(0, p)

from instnct import INSTNCT
from train import func_maskloss_mse, func_accuracy_bin

# ── Constants ──

BATCH      = 4
LR         = 1e-3
STEPS      = 3000
EVAL_EVERY = 200
SEED       = 42
EVAL_SEED  = 9999
M          = 64
R_MAX      = M // 2  # 32

# ── Kernel and R strategy configs ──

KERNELS = ['uniform', 'vshape', 'gaussian']

R_STRATEGIES = OrderedDict([
    ('fixed_R1', {
        'label': 'Fixed R~1',
        'R_param_init': -3.43,   # sigmoid(-3.43) * 32 ~ 1.0
        'freeze': True,
    }),
    ('fixed_Rhalf', {
        'label': 'Fixed R~M/2',
        'R_param_init': 7.0,     # sigmoid(7.0) * 32 ~ 31.97
        'freeze': True,
    }),
    ('learnable', {
        'label': 'Learnable R',
        'R_param_init': -2.0,    # sigmoid(-2) * 32 ~ 3.8 (default init)
        'freeze': False,
    }),
])

# ── Data generators (identical to bench_learnable_R.py) ──

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

TASKS = OrderedDict([
    ('xor',     {'fn': make_xor_batch,         'seq': 36, 'label': 'XOR'}),
    ('delayed', {'fn': make_delayed_xor_batch,  'seq': 44, 'label': 'Delayed XOR'}),
    ('chained', {'fn': make_chained_xor_batch,  'seq': 48, 'label': 'Chained XOR'}),
])

# ── Training ──

def run_one(task_info, kernel_mode, r_strategy, device='cpu'):
    """Train one model for one task with given kernel and R strategy."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    cfg = dict(M=M, embed_dim=64, N=2, R=1, embed_mode=False, kernel_mode=kernel_mode)
    model = INSTNCT(**cfg).to(device)

    # Apply R strategy
    r_cfg = R_STRATEGIES[r_strategy]
    with torch.no_grad():
        model.R_param.fill_(r_cfg['R_param_init'])
    if r_cfg['freeze']:
        model.R_param.requires_grad_(False)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR
    )

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

    # Final R_eff values
    with torch.no_grad():
        R_effs = (torch.sigmoid(model.R_param) * R_MAX).tolist()

    return best_eval, wall, n_params, R_effs

# ── Main ──

def main():
    total_conditions = len(KERNELS) * len(R_STRATEGIES) * len(TASKS)
    print("=" * 78)
    print("  3x3 KERNEL x R STRATEGY -- EXHAUSTIVE BENCHMARK")
    print("=" * 78)
    print(f"  Config: M={M}, D=64, N=2 | Steps: {STEPS} | Batch: {BATCH} | Seed: {SEED}")
    print(f"  Conditions: {total_conditions} ({len(KERNELS)} kernels x {len(R_STRATEGIES)} R strategies x {len(TASKS)} tasks)")
    print()

    # Store all results: results[task][kernel][r_strategy] = (acc, wall, params, R_effs)
    results = {}
    condition_num = 0

    for task_name, task_info in TASKS.items():
        results[task_name] = {}
        for kernel in KERNELS:
            results[task_name][kernel] = {}
            for r_name, r_cfg in R_STRATEGIES.items():
                condition_num += 1
                label = f"{kernel:>8s} + {r_cfg['label']:<14s}"
                print(f"  [{condition_num:2d}/{total_conditions}] {task_info['label']:>12s} | {label}...",
                      end='', flush=True)

                try:
                    acc, wall, n_params, R_effs = run_one(
                        task_info, kernel, r_name
                    )
                    results[task_name][kernel][r_name] = {
                        'acc': acc, 'wall': wall, 'params': n_params, 'R_effs': R_effs
                    }
                    R_str = ', '.join(f'{r:.1f}' for r in R_effs)
                    print(f"  {acc*100:6.2f}%  R=[{R_str}]  ({wall:.0f}s)")
                except Exception as e:
                    results[task_name][kernel][r_name] = {
                        'acc': 0.0, 'wall': 0.0, 'params': 0, 'R_effs': [], 'error': str(e)
                    }
                    print(f"  ERROR: {e}")

    # ── Summary tables ──
    print()
    print("=" * 78)
    print("  RESULTS MATRIX")
    print("=" * 78)

    r_labels = [R_STRATEGIES[r]['label'] for r in R_STRATEGIES]
    header = f"  {'':>10s}  " + "  ".join(f"{l:>14s}" for l in r_labels)

    for task_name, task_info in TASKS.items():
        print(f"\n  {task_info['label']}:")
        print(header)
        print(f"  {'':>10s}  " + "  ".join(["-" * 14] * len(R_STRATEGIES)))

        for kernel in KERNELS:
            row = f"  {kernel:>10s}  "
            for r_name in R_STRATEGIES:
                res = results[task_name][kernel][r_name]
                if 'error' in res:
                    row += f"  {'ERROR':>14s}"
                else:
                    acc_str = f"{res['acc']*100:.2f}%"
                    if not res['R_effs']:
                        row += f"  {acc_str:>14s}"
                    elif R_STRATEGIES[r_name]['freeze']:
                        row += f"  {acc_str:>14s}"
                    else:
                        R_str = ','.join(f'{r:.0f}' for r in res['R_effs'])
                        row += f"  {acc_str:>7s} R=[{R_str}]"
            print(row)

    # ── Geometric mean across tasks ──
    print(f"\n  Combined (geometric mean across 3 tasks):")
    print(header)
    print(f"  {'':>10s}  " + "  ".join(["-" * 14] * len(R_STRATEGIES)))

    best_gmean = 0.0
    best_combo = ""

    for kernel in KERNELS:
        row = f"  {kernel:>10s}  "
        for r_name in R_STRATEGIES:
            accs = []
            for task_name in TASKS:
                res = results[task_name][kernel][r_name]
                if 'error' not in res:
                    accs.append(res['acc'])
            if len(accs) == len(TASKS):
                gmean = np.exp(np.mean(np.log(np.array(accs) + 1e-10)))
                row += f"  {gmean*100:>12.2f}%"
                if gmean > best_gmean:
                    best_gmean = gmean
                    best_combo = f"{kernel} + {R_STRATEGIES[r_name]['label']}"
            else:
                row += f"  {'N/A':>14s}"
        print(row)

    print()
    print(f"  >>> BEST: {best_combo} = {best_gmean*100:.2f}% geometric mean <<<")

    # ── Learnable R final values ──
    print(f"\n  Learnable R final R_eff values (what each expert learned):")
    for kernel in KERNELS:
        parts = []
        for task_name in TASKS:
            res = results[task_name][kernel].get('learnable', {})
            if 'error' not in res and res.get('R_effs'):
                R_str = ', '.join(f'{r:.1f}' for r in res['R_effs'])
                parts.append(f"{TASKS[task_name]['label']}: [{R_str}]")
        if parts:
            print(f"    {kernel:>10s}:  {' | '.join(parts)}")

    # ── Wall times ──
    total_wall = sum(
        results[t][k][r].get('wall', 0)
        for t in TASKS for k in KERNELS for r in R_STRATEGIES
    )
    print(f"\n  Total wall time: {total_wall/60:.1f} min")
    print("=" * 78)


if __name__ == '__main__':
    main()
