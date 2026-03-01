"""Activation stress test — top 3 activations on 3 difficulty levels.

Task 1: XOR (baseline)         — A, B, A^B   (delay=0)
Task 2: Delayed XOR (memory)   — A, B, [8 noise], A^B   (delay=8)
Task 3: Chained XOR (compose)  — A, B, A^B, C, (A^B)^C, D, ((A^B)^C)^D  (3 hops)

Top 3 from v1 bench: tanh, c19, silu

Usage:
    python tests/bench_activations_v2.py
    python tests/bench_activations_v2.py --steps 5000
"""

import sys
import time
import math
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for subdir in ('model', 'training', 'datagen'):
    p = str(ROOT / subdir)
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch
import torch.nn.functional as F

import instnct as instnct_module
from instnct import INSTNCT
from train import func_maskloss_mse, func_accuracy_bin

# ═══════════════════════════════════════════════════════════════
#  Config
# ═══════════════════════════════════════════════════════════════

MODEL_CFG = dict(M=64, embed_dim=64, N=2, R=1, embed_mode=False)
BATCH     = 4
LR        = 1e-3
MAX_STEPS = 3000
EVAL_EVERY = 200
SEED      = 42
EVAL_SEED = 9999

# ═══════════════════════════════════════════════════════════════
#  Activations (top 3)
# ═══════════════════════════════════════════════════════════════

def _c19(x, rho=4.0):
    l = 6.0 * math.pi
    inv_pi = 1.0 / math.pi
    scaled = x * inv_pi
    n = torch.floor(scaled)
    t = scaled - n
    h = t * (1.0 - t)
    is_even = torch.remainder(n, 2.0) < 1.0
    sgn = torch.where(is_even, torch.ones_like(x), -torch.ones_like(x))
    core = math.pi * (sgn * h + (rho * h * h))
    return torch.where(x >= l, x - l, torch.where(x <= -l, x + l, core))

ACTIVATIONS = {
    'tanh': torch.tanh,
    'c19':  _c19,
    'silu': lambda x: F.silu(x),
}

# ═══════════════════════════════════════════════════════════════
#  Data generators — 3 tasks
# ═══════════════════════════════════════════════════════════════

def _bytes_to_bits(data_np, mask_np, batch_size, seq_len, device):
    """Convert byte arrays + mask to bit tensors for the model."""
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
    """Task 1: Standard XOR — A, B, A^B triplets, no delay."""
    n_triplets = (seq_len + 1) // 3 + 2
    data = []
    masks = []
    for s in range(batch_size):
        d = []
        m = []
        for _ in range(n_triplets):
            a = rng.randint(0, 256)
            b = rng.randint(0, 256)
            d.extend([a, b, a ^ b])
            m.extend([0, 0, 1])
        data.append(np.array(d[:seq_len + 1], dtype=np.uint8))
        masks.append(np.array(m[:seq_len + 1], dtype=np.uint8))
    return _bytes_to_bits(data, masks, batch_size, seq_len, device)


def make_delayed_xor_batch(batch_size, seq_len, device, rng, delay=8):
    """Task 2: Delayed XOR — A, B, [delay noise bytes], A^B.
    Tests whether the ring buffer retains A,B through noise interference."""
    # each "unit" = 2 + delay + 1 = delay+3 bytes
    unit = delay + 3
    n_units = (seq_len + 1) // unit + 2
    data = []
    masks = []
    for s in range(batch_size):
        d = []
        m = []
        for _ in range(n_units):
            a = rng.randint(0, 256)
            b = rng.randint(0, 256)
            # A, B
            d.append(a)
            m.append(0)
            d.append(b)
            m.append(0)
            # noise filler — random bytes the model must ignore
            for _ in range(delay):
                d.append(rng.randint(0, 256))
                m.append(0)
            # supervised: A^B
            d.append(a ^ b)
            m.append(1)
        data.append(np.array(d[:seq_len + 1], dtype=np.uint8))
        masks.append(np.array(m[:seq_len + 1], dtype=np.uint8))
    return _bytes_to_bits(data, masks, batch_size, seq_len, device)


def make_chained_xor_batch(batch_size, seq_len, device, rng):
    """Task 3: Chained XOR — A, B, A^B, C, (A^B)^C, D, ((A^B)^C)^D.
    Tests compositional reasoning — each output depends on ALL prior values.
    Pattern: input, input, supervised, input, supervised, input, supervised, ..."""
    data = []
    masks = []
    for s in range(batch_size):
        d = []
        m = []
        while len(d) < seq_len + 1:
            # start a chain
            a = rng.randint(0, 256)
            b = rng.randint(0, 256)
            acc = a ^ b
            d.extend([a, b, acc])
            m.extend([0, 0, 1])
            # extend chain 2 more hops (total 3 XOR operations)
            for _ in range(2):
                c = rng.randint(0, 256)
                acc = acc ^ c
                d.extend([c, acc])
                m.extend([0, 1])
            # small gap between chains
            gap = rng.randint(0, 256)
            d.append(gap)
            m.append(0)
        data.append(np.array(d[:seq_len + 1], dtype=np.uint8))
        masks.append(np.array(m[:seq_len + 1], dtype=np.uint8))
    return _bytes_to_bits(data, masks, batch_size, seq_len, device)


TASKS = {
    'xor':     {'fn': make_xor_batch,         'seq': 36, 'label': 'XOR (baseline)'},
    'delayed': {'fn': make_delayed_xor_batch,  'seq': 44, 'label': 'Delayed XOR (8-step noise)'},
    'chained': {'fn': make_chained_xor_batch,  'seq': 48, 'label': 'Chained XOR (3-hop compose)'},
}

# ═══════════════════════════════════════════════════════════════
#  Run one (activation, task) pair
# ═══════════════════════════════════════════════════════════════

def run_one(act_name, act_fn, task_name, task_info, steps, device='cpu'):
    # NOTE: monkey-patching removed — model uses torch.tanh() directly

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    seq_len = task_info['seq']
    make_batch = task_info['fn']

    model = INSTNCT(**MODEL_CFG).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    eval_rng = np.random.RandomState(EVAL_SEED)
    ex, ey, emask = make_batch(8, seq_len, device, eval_rng)

    train_rng = np.random.RandomState(SEED)
    best_eval = 0.0
    final_loss = 0.0
    eval_history = []

    # Track gradient norms for analysis
    grad_norms = []

    t0 = time.perf_counter()

    for step in range(1, steps + 1):
        x, y, mask = make_batch(BATCH, seq_len, device, train_rng)

        pred, _ = model(x)
        _, masked_loss = func_maskloss_mse(pred, y, mask)

        opt.zero_grad()
        masked_loss.backward()

        # Record gradient norm every 100 steps
        if step % 100 == 0:
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            grad_norms.append((step, total_norm ** 0.5))

        opt.step()
        final_loss = masked_loss.item()

        if step % EVAL_EVERY == 0 or step == steps:
            with torch.no_grad():
                epred, _ = model(ex)
                _, eval_acc = func_accuracy_bin(epred, ey, emask)
            if eval_acc > best_eval:
                best_eval = eval_acc
            eval_history.append((step, eval_acc))

    wall_time = time.perf_counter() - t0
    # Gradient stability: std of grad norms (lower = more stable)
    gn_values = [g[1] for g in grad_norms]
    grad_mean = np.mean(gn_values) if gn_values else 0
    grad_std  = np.std(gn_values) if gn_values else 0

    return {
        'act': act_name,
        'task': task_name,
        'best_eval': best_eval,
        'final_loss': final_loss,
        'wall_time': wall_time,
        'params': n_params,
        'grad_mean': grad_mean,
        'grad_std': grad_std,
        'eval_history': eval_history,
    }

# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=MAX_STEPS)
    args = parser.parse_args()

    print("=" * 78)
    print("  ACTIVATION STRESS TEST v2 — Top 3 x 3 Tasks")
    print("=" * 78)
    print(f"  Config: M={MODEL_CFG['M']}, embed_dim={MODEL_CFG['embed_dim']}, "
          f"N={MODEL_CFG['N']}, R={MODEL_CFG['R']}")
    print(f"  Steps:  {args.steps}  |  Batch: {BATCH}  |  LR: {LR}")
    print(f"  Seed:   {SEED} (identical init + data)")
    print()
    print("  Tasks:")
    for k, v in TASKS.items():
        print(f"    {k:<10} — {v['label']} (seq={v['seq']})")
    print()
    print("  Activations: tanh, c19, silu")
    print("=" * 78)
    print()

    all_results = []

    for task_name, task_info in TASKS.items():
        print(f"{'='*78}")
        print(f"  TASK: {task_info['label']}")
        print(f"{'='*78}")

        for act_name, act_fn in ACTIVATIONS.items():
            print(f"  Running {act_name}...", end='', flush=True)
            r = run_one(act_name, act_fn, task_name, task_info, args.steps)
            all_results.append(r)
            print(f"  eval={r['best_eval']*100:.2f}%  "
                  f"loss={r['final_loss']:.4f}  "
                  f"grad={r['grad_mean']:.3f}+/-{r['grad_std']:.3f}  "
                  f"({r['wall_time']:.0f}s)")

        print()

    # ── Summary matrix ──
    print()
    print("=" * 78)
    print("  RESULTS MATRIX — Best Eval Accuracy (%)")
    print("=" * 78)
    print()

    # Build matrix
    acts = list(ACTIVATIONS.keys())
    tasks = list(TASKS.keys())

    # Header
    print(f"  {'':12}", end='')
    for t in tasks:
        print(f"  {TASKS[t]['label']:<30}", end='')
    print()
    print(f"  {'':12}", end='')
    for _ in tasks:
        print(f"  {'-'*28}  ", end='')
    print()

    for a in acts:
        print(f"  {a:<12}", end='')
        for t in tasks:
            r = next(x for x in all_results if x['act'] == a and x['task'] == t)
            print(f"  {r['best_eval']*100:>6.2f}%                       ", end='')
        print()

    # -- Gradient analysis --
    print()
    print("=" * 78)
    print("  GRADIENT ANALYSIS -- Mean +/- StdDev of L2 norm")
    print("=" * 78)
    print()
    print(f"  {'':12}", end='')
    for t in tasks:
        print(f"  {TASKS[t]['label']:<30}", end='')
    print()
    print(f"  {'':12}", end='')
    for _ in tasks:
        print(f"  {'-'*28}  ", end='')
    print()

    for a in acts:
        print(f"  {a:<12}", end='')
        for t in tasks:
            r = next(x for x in all_results if x['act'] == a and x['task'] == t)
            print(f"  {r['grad_mean']:>6.3f} +/- {r['grad_std']:<6.3f}            ", end='')
        print()

    # -- Speed comparison --
    print()
    print("=" * 78)
    print("  SPEED -- Wall time (seconds)")
    print("=" * 78)
    print()
    print(f"  {'':12}", end='')
    for t in tasks:
        print(f"  {TASKS[t]['label']:<30}", end='')
    print()
    print(f"  {'':12}", end='')
    for _ in tasks:
        print(f"  {'-'*28}  ", end='')
    print()

    for a in acts:
        print(f"  {a:<12}", end='')
        for t in tasks:
            r = next(x for x in all_results if x['act'] == a and x['task'] == t)
            print(f"  {r['wall_time']:>6.1f}s                       ", end='')
        print()

    # ── Per-task winners ──
    print()
    print("=" * 78)
    print("  WINNERS PER TASK")
    print("=" * 78)
    for t in tasks:
        task_results = [x for x in all_results if x['task'] == t]
        task_results.sort(key=lambda x: x['best_eval'], reverse=True)
        w = task_results[0]
        r = task_results[1]
        gap = (w['best_eval'] - r['best_eval']) * 100
        print(f"  {TASKS[t]['label']:<35} "
              f"WINNER: {w['act'].upper():<6} {w['best_eval']*100:.2f}%  "
              f"(+{gap:.2f}% vs {r['act']})")

    # ── Overall verdict ──
    print()
    wins = {a: 0 for a in acts}
    for t in tasks:
        task_results = [x for x in all_results if x['task'] == t]
        winner = max(task_results, key=lambda x: x['best_eval'])
        wins[winner['act']] += 1

    overall = max(wins, key=wins.get)
    print(f"  OVERALL: {overall.upper()} wins {wins[overall]}/{len(tasks)} tasks")
    print()
    print("=" * 78)


if __name__ == '__main__':
    main()
