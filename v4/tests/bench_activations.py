"""Activation function benchmark — controlled comparison on XOR task.

Tests C19 against standard activations (ReLU, GELU, SiLU/Swish, Tanh, None)
on the same XOR generalization task with identical hyperparameters and seeds.

Measures: eval accuracy, convergence speed, wall time, final loss.

Usage:
    python tests/bench_activations.py
    python tests/bench_activations.py --steps 5000
    python tests/bench_activations.py --only c19,relu,tanh
"""

import sys
import time
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
#  Config — same for ALL activations (controlled experiment)
# ═══════════════════════════════════════════════════════════════

MODEL_CFG = dict(M=64, embed_dim=64, N=2, R=1, embed_mode=False)
BATCH     = 4
SEQ_LEN   = 36
LR        = 1e-3
MAX_STEPS = 3000
EVAL_EVERY = 200
SEED      = 42
EVAL_SEED = 9999

# ═══════════════════════════════════════════════════════════════
#  Activation functions to test
# ═══════════════════════════════════════════════════════════════

def _identity(x):
    """No activation — pure linear baseline."""
    return x

def _c19(x, rho=4.0):
    """Our C19 periodic parabolic wave — copied from instnct.py."""
    import math
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
    'c19':      _c19,
    'relu':     torch.relu,
    'gelu':     lambda x: F.gelu(x),
    'silu':     lambda x: F.silu(x),       # Swish = x * sigmoid(x)
    'tanh':     torch.tanh,
    'none':     _identity,
}

# ═══════════════════════════════════════════════════════════════
#  XOR data generator (from test_xor_ceiling.py)
# ═══════════════════════════════════════════════════════════════

def make_random_xor_batch(batch_size, seq_len, device, rng=None):
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

# ═══════════════════════════════════════════════════════════════
#  Single activation run
# ═══════════════════════════════════════════════════════════════

def run_one(name, activation_fn, steps, device='cpu'):
    """Train one model with the given activation, return results dict."""

    # NOTE: monkey-patching was removed — model uses torch.tanh() directly,
    # so patching c19_activation had no effect. These benchmarks now only
    # measure training dynamics with the model's built-in tanh activation.

    # Deterministic init — same weights every run
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = INSTNCT(**MODEL_CFG).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    # Fixed eval set — same for every activation
    eval_rng = np.random.RandomState(EVAL_SEED)
    ex, ey, emask = make_random_xor_batch(8, SEQ_LEN, device, eval_rng)

    train_rng = np.random.RandomState(SEED)
    best_eval = 0.0
    final_loss = 0.0
    final_eval = 0.0
    step_to_60 = None
    step_to_70 = None
    step_to_80 = None

    # Timing: measure forward+backward per step
    fwd_times = []
    bwd_times = []

    t0 = time.perf_counter()

    for step in range(1, steps + 1):
        x, y, mask = make_random_xor_batch(BATCH, SEQ_LEN, device, train_rng)

        # Forward timing
        t_fwd = time.perf_counter()
        pred, _ = model(x)
        _, masked_loss = func_maskloss_mse(pred, y, mask)
        fwd_times.append(time.perf_counter() - t_fwd)

        # Backward timing
        t_bwd = time.perf_counter()
        opt.zero_grad()
        masked_loss.backward()
        opt.step()
        bwd_times.append(time.perf_counter() - t_bwd)

        final_loss = masked_loss.item()

        # Eval
        if step % EVAL_EVERY == 0 or step == steps:
            with torch.no_grad():
                epred, _ = model(ex)
                _, eval_acc = func_accuracy_bin(epred, ey, emask)
            final_eval = eval_acc
            if eval_acc > best_eval:
                best_eval = eval_acc
            if step_to_60 is None and eval_acc >= 0.60:
                step_to_60 = step
            if step_to_70 is None and eval_acc >= 0.70:
                step_to_70 = step
            if step_to_80 is None and eval_acc >= 0.80:
                step_to_80 = step

    wall_time = time.perf_counter() - t0

    return {
        'name': name,
        'params': n_params,
        'best_eval': best_eval,
        'final_eval': final_eval,
        'final_loss': final_loss,
        'wall_time': wall_time,
        'avg_fwd_ms': np.mean(fwd_times) * 1000,
        'avg_bwd_ms': np.mean(bwd_times) * 1000,
        'step_to_60': step_to_60,
        'step_to_70': step_to_70,
        'step_to_80': step_to_80,
    }

# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Activation function benchmark')
    parser.add_argument('--steps', type=int, default=MAX_STEPS)
    parser.add_argument('--only', type=str, default=None,
                        help='Comma-separated list of activations to test (e.g. c19,relu,tanh)')
    args = parser.parse_args()

    if args.only:
        names = [n.strip().lower() for n in args.only.split(',')]
        to_test = {n: ACTIVATIONS[n] for n in names if n in ACTIVATIONS}
    else:
        to_test = ACTIVATIONS

    print("=" * 78)
    print("  ACTIVATION FUNCTION BENCHMARK — XOR Generalization Task")
    print("=" * 78)
    print(f"  Config: M={MODEL_CFG['M']}, D={MODEL_CFG['D']}, "
          f"N={MODEL_CFG['N']}, R={MODEL_CFG['R']}")
    print(f"  Steps:  {args.steps}  |  Batch: {BATCH}  |  LR: {LR}")
    print(f"  Seed:   {SEED} (same init weights + data for every activation)")
    print(f"  Testing: {', '.join(to_test.keys())}")
    print("=" * 78)
    print()

    results = []

    for name, fn in to_test.items():
        print(f"--- Running: {name.upper()} ---")
        r = run_one(name, fn, args.steps)
        results.append(r)
        print(f"    best_eval={r['best_eval']*100:.2f}%  "
              f"wall={r['wall_time']:.1f}s  "
              f"fwd={r['avg_fwd_ms']:.2f}ms  "
              f"bwd={r['avg_bwd_ms']:.2f}ms")
        print()

    # ── Results table ──
    print()
    print("=" * 78)
    print("  RESULTS — XOR Generalization Comparison")
    print("=" * 78)
    print()

    # Sort by best eval accuracy (descending)
    results.sort(key=lambda r: r['best_eval'], reverse=True)

    # Header
    print(f"  {'Activation':<10} {'Best Eval':>10} {'Final Eval':>11} "
          f"{'Final Loss':>11} {'Wall Time':>10} {'Fwd (ms)':>9} {'Bwd (ms)':>9} "
          f"{'@60%':>6} {'@70%':>6} {'@80%':>6}")
    print("  " + "-" * 105)

    for r in results:
        s60 = f"{r['step_to_60']}" if r['step_to_60'] else "  --"
        s70 = f"{r['step_to_70']}" if r['step_to_70'] else "  --"
        s80 = f"{r['step_to_80']}" if r['step_to_80'] else "  --"
        print(f"  {r['name']:<10} {r['best_eval']*100:>9.2f}% {r['final_eval']*100:>10.2f}% "
              f"{r['final_loss']:>11.6f} {r['wall_time']:>9.1f}s "
              f"{r['avg_fwd_ms']:>9.2f} {r['avg_bwd_ms']:>9.2f} "
              f"{s60:>6} {s70:>6} {s80:>6}")

    print()

    # Winner summary
    winner = results[0]
    runner = results[1] if len(results) > 1 else None
    linear = next((r for r in results if r['name'] == 'none'), None)

    print(f"  WINNER: {winner['name'].upper()} at {winner['best_eval']*100:.2f}% eval accuracy")
    if runner:
        gap = (winner['best_eval'] - runner['best_eval']) * 100
        print(f"  Runner-up: {runner['name'].upper()} at {runner['best_eval']*100:.2f}% (gap: {gap:+.2f}%)")
    if linear:
        print(f"  Linear baseline: {linear['best_eval']*100:.2f}% (random guess = ~50%)")

    # Speed comparison
    print()
    fastest = min(results, key=lambda r: r['avg_fwd_ms'] + r['avg_bwd_ms'])
    c19_r = next((r for r in results if r['name'] == 'c19'), None)
    if c19_r and fastest['name'] != 'c19':
        c19_total = c19_r['avg_fwd_ms'] + c19_r['avg_bwd_ms']
        fast_total = fastest['avg_fwd_ms'] + fastest['avg_bwd_ms']
        overhead = ((c19_total / fast_total) - 1) * 100
        print(f"  Fastest: {fastest['name'].upper()} ({fast_total:.2f}ms/step)")
        print(f"  C19 overhead vs fastest: +{overhead:.1f}%")
    elif c19_r:
        print(f"  C19 is the fastest at {c19_r['avg_fwd_ms'] + c19_r['avg_bwd_ms']:.2f}ms/step")

    print()
    print("=" * 78)


if __name__ == '__main__':
    main()
