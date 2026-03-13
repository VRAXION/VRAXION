"""Realistic A/B test: C19 phi-asymmetry on algorithmic generalization tasks.

Unlike the echo test (pure memorization), this tests:
  1. GENERALIZATION — train/val split, model must learn the pattern, not memorize
  2. LONGER TRAINING — 3K steps, enough to see late-stage differences
  3. MULTIPLE TASKS — sorting, reversal, copy-with-offset (different difficulty)

Tasks:
  - sort:   input = random 8 bytes → output = sorted bytes
  - reverse: input = random 8 bytes → output = reversed
  - add1:   input = random bytes → output = (byte + 1) % 256

All use the real INSTNCT model, just with small config for speed.

Usage: python v4/tests/ab_c19_realistic.py [--steps 3000] [--task sort]
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

import instnct as instnct_module
from instnct import INSTNCT, _C19_C

_original_c19 = instnct_module._c19_activation
PHI = (1.0 + math.sqrt(5)) / 2.0
PHI_INV = 1.0 / PHI


# ═══════════════════════════════════════════════════════════════
#  C19 variants (same as before)
# ═══════════════════════════════════════════════════════════════

def _make_c19_variant(neg_gain, pos_gain, name=''):
    def _c19_fn(x, rho=4.0, C=None):
        if C is None:
            C = _C19_C
        l = 6.0 * C
        inv_c = 1.0 / C
        scaled = x * inv_c
        n = torch.floor(scaled)
        t = scaled - n
        h = t - t * t
        sgn = 1.0 - 2.0 * torch.remainder(n, 2.0)
        core = C * h * (sgn + rho * h)
        if neg_gain != 1.0 or pos_gain != 1.0:
            gain = torch.where(core < 0,
                               torch.tensor(neg_gain, dtype=core.dtype),
                               torch.tensor(pos_gain, dtype=core.dtype))
            core = core * gain
        return torch.where(x.abs() > l, x - x.sign() * l, core)
    _c19_fn.__name__ = name
    return _c19_fn


# ═══════════════════════════════════════════════════════════════
#  Data generators — algorithmic tasks with train/val split
# ═══════════════════════════════════════════════════════════════

BLOCK = 8  # sequence block size

def make_sort_batch(batch, seq_len, rng):
    """Input: random bytes in blocks of BLOCK. Target: each block sorted."""
    n_blocks = seq_len // BLOCK
    actual_len = n_blocks * BLOCK
    x = np.zeros((batch, actual_len), dtype=np.int64)
    y = np.zeros((batch, actual_len), dtype=np.int64)
    for i in range(batch):
        for b in range(n_blocks):
            block = rng.randint(0, 256, size=BLOCK, dtype=np.uint8)
            x[i, b*BLOCK:(b+1)*BLOCK] = block
            y[i, b*BLOCK:(b+1)*BLOCK] = np.sort(block)
    return x, y


def make_reverse_batch(batch, seq_len, rng):
    """Input: random bytes in blocks. Target: each block reversed."""
    n_blocks = seq_len // BLOCK
    actual_len = n_blocks * BLOCK
    x = np.zeros((batch, actual_len), dtype=np.int64)
    y = np.zeros((batch, actual_len), dtype=np.int64)
    for i in range(batch):
        for b in range(n_blocks):
            block = rng.randint(0, 256, size=BLOCK, dtype=np.uint8)
            x[i, b*BLOCK:(b+1)*BLOCK] = block
            y[i, b*BLOCK:(b+1)*BLOCK] = block[::-1]
    return x, y


def make_add1_batch(batch, seq_len, rng):
    """Input: random bytes. Target: (byte + 1) % 256."""
    x = rng.randint(0, 256, size=(batch, seq_len)).astype(np.int64)
    y = (x + 1) % 256
    return x, y


TASK_FNS = {
    'sort': make_sort_batch,
    'reverse': make_reverse_batch,
    'add1': make_add1_batch,
}


# ═══════════════════════════════════════════════════════════════
#  Training loop
# ═══════════════════════════════════════════════════════════════

BATCH = 16
SEQ_LEN = 64  # 8 blocks of 8
LR = 5e-4
SEED = 42
LOG_EVERY = 100
VAL_EVERY = 200

MODEL_CFG = dict(
    M=64, hidden_dim=128, slot_dim=32, N=1, R=1,
    embed_mode=True,
    kernel_mode='vshape',
    pointer_mode='pilot',
    write_mode='replace',
    embed_encoding='bitlift',
    output_encoding='lowrank_c19',
    checkpoint_chunks=0,
)


def evaluate(model, task_fn, n_batches=4, device='cpu', seed=99999):
    """Evaluate on unseen data (different seed = different data)."""
    model.eval()
    rng = np.random.RandomState(seed)
    total_correct = 0
    total_count = 0
    total_loss = 0.0

    with torch.no_grad():
        for _ in range(n_batches):
            x_np, y_np = task_fn(BATCH, SEQ_LEN, rng)
            x = torch.from_numpy(x_np).to(device)
            y = torch.from_numpy(y_np).to(device)

            out, _ = model(x)
            logits = out.view(-1, 256)
            targets = y.view(-1)
            loss = F.cross_entropy(logits, targets)
            total_loss += loss.item()

            preds = logits.argmax(dim=-1)
            total_correct += (preds == targets).sum().item()
            total_count += targets.numel()

    return {
        'val_loss': total_loss / n_batches,
        'val_acc': total_correct / total_count,
    }


def train_variant(name, activation_fn, task_fn, max_steps, device):
    instnct_module._c19_activation = activation_fn
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = INSTNCT(**MODEL_CFG).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    train_rng = np.random.RandomState(SEED)

    history = []
    gnorms = []
    val_history = []
    t0 = time.perf_counter()

    for step in range(1, max_steps + 1):
        model.train()
        x_np, y_np = task_fn(BATCH, SEQ_LEN, train_rng)
        x = torch.from_numpy(x_np).to(device)
        y = torch.from_numpy(y_np).to(device)

        out, _ = model(x)
        logits = out.view(-1, 256)
        targets = y.view(-1)
        loss = F.cross_entropy(logits, targets)

        opt.zero_grad()
        loss.backward()

        gnorm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                gnorm += p.grad.data.norm(2).item() ** 2
        gnorm = gnorm ** 0.5
        gnorms.append(gnorm)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()

        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            acc = (preds == targets).float().mean().item()

        lv = loss.item()
        history.append({'step': step, 'loss': lv, 'acc': acc, 'gnorm': gnorm})

        # Validation
        do_val = (step % VAL_EVERY == 0) or (step == max_steps) or (step <= 5 and step == 1)
        if do_val:
            vr = evaluate(model, task_fn, n_batches=4, device=device)
            val_history.append({'step': step, **vr})

        if step <= 3 or step % LOG_EVERY == 0 or step == max_steps:
            elapsed = time.perf_counter() - t0
            val_str = ""
            if val_history and val_history[-1]['step'] == step:
                val_str = f"  val_acc={val_history[-1]['val_acc']*100:.1f}%"
            print(f"  [{name:>15s}] step {step:5d}  loss={lv:.4f}  "
                  f"train_acc={acc*100:.1f}%  gnorm={gnorm:.1f}{val_str}  [{elapsed:.1f}s]")

        if math.isnan(lv) or lv > 50:
            if step > 50:
                print(f"  [{name:>15s}] DIVERGED at step {step}")
                break

    elapsed = time.perf_counter() - t0
    instnct_module._c19_activation = _original_c19

    gnorms_arr = np.array(gnorms)

    best_val = max(val_history, key=lambda v: v['val_acc']) if val_history else None

    return {
        'name': name,
        'params': n_params,
        'history': history,
        'val_history': val_history,
        'final_train_loss': history[-1]['loss'],
        'final_train_acc': history[-1]['acc'],
        'best_train_acc': max(h['acc'] for h in history),
        'final_val_acc': val_history[-1]['val_acc'] if val_history else 0,
        'best_val_acc': best_val['val_acc'] if best_val else 0,
        'best_val_step': best_val['step'] if best_val else 0,
        'elapsed': elapsed,
        'gnorm_mean': gnorms_arr.mean(),
        'gnorm_max': gnorms_arr.max(),
        'gnorm_std': gnorms_arr.std(),
        'gnorm_spikes': int((gnorms_arr > 100).sum()),
    }


def run():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--task', type=str, default='all',
                        choices=['sort', 'reverse', 'add1', 'all'])
    args = parser.parse_args()

    device = 'cpu'
    tasks = list(TASK_FNS.keys()) if args.task == 'all' else [args.task]

    variants = [
        ('original',     _original_c19),
        ('neg*phi',      _make_c19_variant(PHI, 1.0, 'neg*phi')),
        ('dual-phi',     _make_c19_variant(PHI, PHI_INV, 'dual-phi')),
    ]

    for task_name in tasks:
        task_fn = TASK_FNS[task_name]
        print("\n" + "=" * 78)
        print(f"  TASK: {task_name.upper()}")
        print(f"  Config: B={BATCH}, T={SEQ_LEN}, M={MODEL_CFG['M']}, "
              f"H={MODEL_CFG['hidden_dim']}, Steps={args.steps}")
        print(f"  Block size: {BLOCK}, LR: {LR}, Seed: {SEED}")
        print(f"  TRAIN data: fresh batch each step (RandomState seed={SEED})")
        print(f"  VAL data: different seed (99999), 4 batches, every {VAL_EVERY} steps")
        print("=" * 78)

        results = []
        for vname, vfn in variants:
            print(f"\n{'─' * 78}")
            print(f"  Training: {vname} on {task_name}")
            print(f"{'─' * 78}")
            r = train_variant(vname, vfn, task_fn, args.steps, device)
            results.append(r)

        # Summary for this task
        print(f"\n{'=' * 78}")
        print(f"  RESULTS — {task_name.upper()}")
        print(f"{'=' * 78}")
        print(f"  {'Variant':<15} {'Train_Acc':>10} {'Val_Acc':>10} {'BestVal':>10} "
              f"{'@Step':>7} {'gnorm_max':>10} {'gnorm_mean':>11}")
        print(f"  {'─'*15} {'─'*10} {'─'*10} {'─'*10} {'─'*7} {'─'*10} {'─'*11}")
        for r in results:
            print(f"  {r['name']:<15} {r['final_train_acc']*100:>9.1f}% "
                  f"{r['final_val_acc']*100:>9.1f}% {r['best_val_acc']*100:>9.1f}% "
                  f"{r['best_val_step']:>7} {r['gnorm_max']:>10.1f} {r['gnorm_mean']:>11.2f}")

        # Validation learning curves
        print(f"\n  Validation Accuracy Curves:")
        checkpoints = sorted(set(
            s for r in results for v in r['val_history'] for s in [v['step']]
        ))
        header = f"  {'Step':>6}" + "".join(f" {r['name']:>14}" for r in results)
        print(header)
        for cp in checkpoints:
            vals = []
            for r in results:
                match = [v for v in r['val_history'] if v['step'] == cp]
                if match:
                    vals.append(f" {match[0]['val_acc']*100:>13.1f}%")
                else:
                    vals.append(f" {'—':>14}")
            print(f"  {cp:>6}" + "".join(vals))

        # Generalization gap
        print(f"\n  Generalization Gap (train_acc - val_acc):")
        for r in results:
            gap = r['final_train_acc'] - r['final_val_acc']
            print(f"    {r['name']:<15}  gap = {gap*100:+.1f}%  "
                  f"(train={r['final_train_acc']*100:.1f}%, val={r['final_val_acc']*100:.1f}%)")

    print(f"\n{'=' * 78}")
    print(f"  OVERALL VERDICT")
    print(f"{'=' * 78}")
    print(f"  If phi-asymmetry helps on generalization tasks (not just memorization),")
    print(f"  it's likely to scale to larger models and real scenarios.")
    print(f"{'=' * 78}")


if __name__ == '__main__':
    run()
