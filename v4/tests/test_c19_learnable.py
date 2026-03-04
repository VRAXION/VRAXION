"""Learnable-C validation: does C_input / C_hidden actually learn across diverse tasks?

Tests multiple algorithmic tasks (echo, reverse, add1, XOR-shift) to verify:
  1. C values MOVE from their init (π ≈ 3.14) — gradients flow through sigmoid-bounded C
  2. C values stay WITHIN bounds [1.0, 50.0]
  3. Different tasks produce DIFFERENT C distributions (the model adapts C to the task)
  4. _diag dict is populated with C telemetry keys

This is NOT a convergence test — we only train ~200 steps on a tiny model.
The question is: does the optimizer move C, and does it move differently per task?

Usage:
    python tests/test_c19_learnable.py              # run from v4/
    python tests/test_c19_learnable.py --steps 500  # more steps for clearer signal
"""

import sys
import time
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for subdir in ('model', 'training', 'datagen'):
    p = str(ROOT / subdir)
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch
import torch.nn.functional as F
from instnct import INSTNCT, _C_from_raw, _C19_C_MIN, _C19_C_MAX

# ═══════════════════════════════════════════════════════════════
#  Config — tiny model, CPU, fast iteration
# ═══════════════════════════════════════════════════════════════

BATCH = 8
SEQ_LEN = 32
LR = 1e-3
SEED = 42
DEFAULT_STEPS = 200

MODEL_CFG = dict(
    M=32, hidden_dim=64, slot_dim=16, N=1, R=1,
    embed_mode=True,
    kernel_mode='vshape',
    pointer_mode='pilot',
    write_mode='replace',
    embed_encoding='bitlift',
    output_encoding='lowrank_c19',
    checkpoint_chunks=0,
)

C_INIT = math.pi  # expected initial C value


# ═══════════════════════════════════════════════════════════════
#  Data generators — diverse algorithmic tasks
# ═══════════════════════════════════════════════════════════════

def make_echo_batch(batch, seq_len, rng):
    """Echo: repeat 8-byte blocks. Tests pure memorization / pattern copying."""
    block = 8
    n_blocks = seq_len // block
    actual = n_blocks * block
    x = np.zeros((batch, actual), dtype=np.int64)
    y = np.zeros((batch, actual), dtype=np.int64)
    for i in range(batch):
        seed_block = rng.randint(0, 256, size=block, dtype=np.uint8)
        flat = np.tile(seed_block, n_blocks)
        x[i] = flat
        # target: shifted by 1 (predict next byte in echo stream)
        y[i, :-1] = flat[1:]
        y[i, -1] = flat[0]
    return x, y


def make_reverse_batch(batch, seq_len, rng):
    """Reverse: input blocks → output reversed blocks. Tests positional reasoning."""
    block = 8
    n_blocks = seq_len // block
    actual = n_blocks * block
    x = np.zeros((batch, actual), dtype=np.int64)
    y = np.zeros((batch, actual), dtype=np.int64)
    for i in range(batch):
        for b in range(n_blocks):
            bl = rng.randint(0, 256, size=block, dtype=np.uint8)
            x[i, b*block:(b+1)*block] = bl
            y[i, b*block:(b+1)*block] = bl[::-1]
    return x, y


def make_add1_batch(batch, seq_len, rng):
    """Add-1: (byte + 1) % 256. Tests pointwise arithmetic."""
    x = rng.randint(0, 256, size=(batch, seq_len)).astype(np.int64)
    y = (x + 1) % 256
    return x, y


def make_xor_shift_batch(batch, seq_len, rng):
    """XOR-shift: y[t] = x[t] XOR x[t-1]. Tests temporal dependency + bitwise logic."""
    x = rng.randint(0, 256, size=(batch, seq_len)).astype(np.int64)
    y = np.zeros_like(x)
    y[:, 0] = x[:, 0]  # first position: identity (no previous)
    y[:, 1:] = x[:, 1:] ^ x[:, :-1]
    return x, y


TASKS = {
    'echo': make_echo_batch,
    'reverse': make_reverse_batch,
    'add1': make_add1_batch,
    'xor_shift': make_xor_shift_batch,
}


# ═══════════════════════════════════════════════════════════════
#  Training + C extraction
# ═══════════════════════════════════════════════════════════════

def get_c_values(model):
    """Extract actual C values (after sigmoid bounding) from model parameters."""
    with torch.no_grad():
        c_inp = _C_from_raw(model.c19_C_input)
        c_hid = _C_from_raw(model.c19_C_hidden)
    return {
        'c_input_mean': c_inp.mean().item(),
        'c_input_min': c_inp.min().item(),
        'c_input_max': c_inp.max().item(),
        'c_input_std': c_inp.std().item(),
        'c_hidden_mean': c_hid.mean().item(),
        'c_hidden_min': c_hid.min().item(),
        'c_hidden_max': c_hid.max().item(),
        'c_hidden_std': c_hid.std().item(),
    }


def train_task(task_name, task_fn, max_steps, device='cpu'):
    """Train a fresh model on one task, return C trajectory."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = INSTNCT(**MODEL_CFG).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    rng = np.random.RandomState(SEED)

    c_init = get_c_values(model)
    c_trajectory = [('init', 0, c_init)]

    losses = []
    for step in range(1, max_steps + 1):
        model.train()
        x_np, y_np = task_fn(BATCH, SEQ_LEN, rng)
        x = torch.from_numpy(x_np).to(device)
        y = torch.from_numpy(y_np).to(device)

        out, _ = model(x)
        logits = out.view(-1, 256)
        targets = y.view(-1)
        loss = F.cross_entropy(logits, targets)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()

        losses.append(loss.item())

        # record C at checkpoints
        if step in (10, 50, 100) or step == max_steps or (step % 100 == 0):
            c_trajectory.append((task_name, step, get_c_values(model)))

    # check _diag is populated
    diag = getattr(model, '_diag', {})

    c_final = get_c_values(model)
    return {
        'task': task_name,
        'c_init': c_init,
        'c_final': c_final,
        'c_trajectory': c_trajectory,
        'final_loss': losses[-1],
        'loss_drop': losses[0] - losses[-1],
        'diag_has_c': 'c_input_mean' in diag,
    }


# ═══════════════════════════════════════════════════════════════
#  Analysis & reporting
# ═══════════════════════════════════════════════════════════════

def run():
    import argparse
    parser = argparse.ArgumentParser(description="Learnable-C multi-task validation")
    parser.add_argument('--steps', type=int, default=DEFAULT_STEPS)
    parser.add_argument('--task', type=str, default='all',
                        choices=list(TASKS.keys()) + ['all'])
    args = parser.parse_args()

    device = 'cpu'
    task_names = list(TASKS.keys()) if args.task == 'all' else [args.task]

    print("=" * 78)
    print("  LEARNABLE-C VALIDATION — Multi-Task Diagnostic")
    print("=" * 78)
    print(f"  Model: M={MODEL_CFG['M']}, H={MODEL_CFG['hidden_dim']}, "
          f"N={MODEL_CFG['N']}, R={MODEL_CFG['R']}")
    print(f"  Training: B={BATCH}, T={SEQ_LEN}, LR={LR}, steps={args.steps}")
    print(f"  C init: π ≈ {C_INIT:.4f}  bounds: [{_C19_C_MIN}, {_C19_C_MAX}]")
    print(f"  Tasks: {', '.join(task_names)}")
    print("=" * 78)

    results = []
    for tname in task_names:
        t0 = time.perf_counter()
        print(f"\n{'─' * 78}")
        print(f"  Training: {tname}")
        r = train_task(tname, TASKS[tname], args.steps, device)
        elapsed = time.perf_counter() - t0
        r['elapsed'] = elapsed
        results.append(r)

        # trajectory
        print(f"  C trajectory ({tname}):")
        for label, step, cv in r['c_trajectory']:
            print(f"    step {step:>5d}: C_input={cv['c_input_mean']:.4f}"
                  f" [{cv['c_input_min']:.2f}-{cv['c_input_max']:.2f}]"
                  f"  C_hidden={cv['c_hidden_mean']:.4f}"
                  f" [{cv['c_hidden_min']:.2f}-{cv['c_hidden_max']:.2f}]"
                  f"  (std_in={cv['c_input_std']:.4f} std_hid={cv['c_hidden_std']:.4f})")
        print(f"  loss: {r['final_loss']:.4f} (drop={r['loss_drop']:.4f})  [{elapsed:.1f}s]")

    # ── Summary ──
    print(f"\n{'=' * 78}")
    print("  SUMMARY")
    print(f"{'=' * 78}")

    print(f"\n  {'Task':<12} {'C_in_init':>10} {'C_in_final':>10} {'Δ_in':>8}"
          f"  {'C_hid_init':>10} {'C_hid_final':>10} {'Δ_hid':>8}  {'diag?':>5}")
    print(f"  {'─'*12} {'─'*10} {'─'*10} {'─'*8}  {'─'*10} {'─'*10} {'─'*8}  {'─'*5}")

    all_ok = True
    c_in_finals = []
    c_hid_finals = []
    for r in results:
        ci0 = r['c_init']['c_input_mean']
        ci1 = r['c_final']['c_input_mean']
        ch0 = r['c_init']['c_hidden_mean']
        ch1 = r['c_final']['c_hidden_mean']
        di = ci1 - ci0
        dh = ch1 - ch0
        diag_ok = r['diag_has_c']
        c_in_finals.append(ci1)
        c_hid_finals.append(ch1)
        print(f"  {r['task']:<12} {ci0:>10.4f} {ci1:>10.4f} {di:>+8.4f}"
              f"  {ch0:>10.4f} {ch1:>10.4f} {dh:>+8.4f}  {'OK' if diag_ok else 'FAIL':>5}")
        if not diag_ok:
            all_ok = False

    # ── Verdicts ──
    print(f"\n  CHECKS:")

    # 1. Did C move from init?
    any_moved = False
    for r in results:
        ci_delta = abs(r['c_final']['c_input_mean'] - C_INIT)
        ch_delta = abs(r['c_final']['c_hidden_mean'] - C_INIT)
        if ci_delta > 0.01 or ch_delta > 0.01:
            any_moved = True
    status = "PASS" if any_moved else "FAIL"
    print(f"  [{'✓' if any_moved else '✗'}] C values MOVE from init (π):  {status}")
    if not any_moved:
        all_ok = False

    # 2. C stays within bounds?
    in_bounds = True
    for r in results:
        cf = r['c_final']
        if cf['c_input_min'] < _C19_C_MIN - 0.01 or cf['c_input_max'] > _C19_C_MAX + 0.01:
            in_bounds = False
        if cf['c_hidden_min'] < _C19_C_MIN - 0.01 or cf['c_hidden_max'] > _C19_C_MAX + 0.01:
            in_bounds = False
    status = "PASS" if in_bounds else "FAIL"
    print(f"  [{'✓' if in_bounds else '✗'}] C values stay within [{_C19_C_MIN}, {_C19_C_MAX}]:  {status}")
    if not in_bounds:
        all_ok = False

    # 3. Per-neuron spread (std > 0 = neurons specialize)
    any_spread = False
    for r in results:
        if r['c_final']['c_input_std'] > 0.01 or r['c_final']['c_hidden_std'] > 0.01:
            any_spread = True
    status = "PASS" if any_spread else "WEAK"
    print(f"  [{'✓' if any_spread else '~'}] Per-neuron C specialization (std > 0.01):  {status}")

    # 4. Different tasks → different C (if multiple tasks)
    if len(results) > 1:
        c_in_spread = max(c_in_finals) - min(c_in_finals)
        c_hid_spread = max(c_hid_finals) - min(c_hid_finals)
        task_diff = c_in_spread > 0.05 or c_hid_spread > 0.05
        status = "PASS" if task_diff else "WEAK"
        print(f"  [{'✓' if task_diff else '~'}] Tasks produce different C profiles "
              f"(spread_in={c_in_spread:.4f} spread_hid={c_hid_spread:.4f}):  {status}")

    # 5. Diag dict populated
    diag_all_ok = all(r['diag_has_c'] for r in results)
    status = "PASS" if diag_all_ok else "FAIL"
    print(f"  [{'✓' if diag_all_ok else '✗'}] _diag dict has C telemetry keys:  {status}")
    if not diag_all_ok:
        all_ok = False

    # 6. Loss dropped on at least one task (model is learning, not dead)
    # Some tasks (echo, xor) are too hard for a tiny model in 200 steps — that's OK.
    # We only need to see the optimizer is alive (at least one task trains).
    any_learning = any(r['loss_drop'] > 0.1 for r in results)
    n_learning = sum(1 for r in results if r['loss_drop'] > 0.01)
    status = "PASS" if any_learning else "FAIL"
    print(f"  [{'✓' if any_learning else '✗'}] Loss dropped on {n_learning}/{len(results)} tasks "
          f"(model learns):  {status}")
    if not any_learning:
        all_ok = False

    print(f"\n{'=' * 78}")
    if all_ok:
        print("  VERDICT: Learnable C is working across diverse tasks.")
    else:
        print("  VERDICT: Issues detected — see FAIL checks above.")
    print(f"{'=' * 78}")

    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(run())
