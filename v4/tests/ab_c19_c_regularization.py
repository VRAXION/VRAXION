"""A/B test: C regularization toward π — does it help or hurt?

Compares λ=0 (no reg) vs λ=1e-4 vs λ=1e-3 vs λ=1e-2 on add1 task.
All configs share identical seed, data order, model init.
Measures: final acc/loss, C drift, C spread, training stability.

Usage:
    python tests/ab_c19_c_regularization.py
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
#  Config — deterministic
# ═══════════════════════════════════════════════════════════════

BATCH = 8
SEQ_LEN = 32
LR = 1e-3
STEPS = 1000
SEED = 42
EVAL_BATCHES = 16          # more batches for stable eval
C_INIT = math.pi

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

LAMBDAS = [0.0, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]

# ═══════════════════════════════════════════════════════════════
#  Task
# ═══════════════════════════════════════════════════════════════

def make_add1_batch(batch, seq_len, rng):
    x = rng.randint(0, 256, size=(batch, seq_len)).astype(np.int64)
    y = (x + 1) % 256
    return x, y

# ═══════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════

def get_c_stats(model):
    with torch.no_grad():
        c_inp = _C_from_raw(model.c19_C_input)
        c_hid = _C_from_raw(model.c19_C_hidden)
    return {
        'c_in_mean': c_inp.mean().item(),
        'c_in_std': c_inp.std().item(),
        'c_in_min': c_inp.min().item(),
        'c_in_max': c_inp.max().item(),
        'c_hid_mean': c_hid.mean().item(),
        'c_hid_std': c_hid.std().item(),
        'c_hid_min': c_hid.min().item(),
        'c_hid_max': c_hid.max().item(),
    }

def c_reg_loss(model, lam):
    """L2 regularization pulling C toward π."""
    if lam == 0.0:
        return 0.0
    c_inp = _C_from_raw(model.c19_C_input)
    c_hid = _C_from_raw(model.c19_C_hidden)
    return lam * ((c_inp - C_INIT).pow(2).mean() + (c_hid - C_INIT).pow(2).mean())

def evaluate(model, n_batches, device):
    model.eval()
    rng = np.random.RandomState(9999)  # fixed eval seed
    total_correct = 0
    total_count = 0
    total_loss = 0.0
    with torch.no_grad():
        for _ in range(n_batches):
            x_np, y_np = make_add1_batch(BATCH, SEQ_LEN, rng)
            x = torch.from_numpy(x_np).to(device)
            y = torch.from_numpy(y_np).to(device)
            out, _ = model(x)
            logits = out.view(-1, 256)
            targets = y.view(-1)
            loss = F.cross_entropy(logits, targets)
            preds = logits.argmax(dim=-1)
            total_correct += (preds == targets).sum().item()
            total_count += targets.numel()
            total_loss += loss.item()
    return {
        'acc': total_correct / total_count,
        'loss': total_loss / n_batches,
    }

# ═══════════════════════════════════════════════════════════════
#  Training
# ═══════════════════════════════════════════════════════════════

def train_with_reg(lam, device='cpu'):
    # Fully deterministic init
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.use_deterministic_algorithms(False)  # avoid errors on CPU

    model = INSTNCT(**MODEL_CFG).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    rng = np.random.RandomState(SEED)

    trajectory = []
    losses_task = []
    losses_reg = []

    for step in range(1, STEPS + 1):
        model.train()
        x_np, y_np = make_add1_batch(BATCH, SEQ_LEN, rng)
        x = torch.from_numpy(x_np).to(device)
        y = torch.from_numpy(y_np).to(device)

        out, _ = model(x)
        logits = out.view(-1, 256)
        targets = y.view(-1)
        task_loss = F.cross_entropy(logits, targets)
        reg = c_reg_loss(model, lam)
        loss = task_loss + reg

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()

        losses_task.append(task_loss.item())
        if isinstance(reg, float):
            losses_reg.append(reg)
        else:
            losses_reg.append(reg.item())

        if step % 100 == 0 or step == STEPS:
            cs = get_c_stats(model)
            ev = evaluate(model, EVAL_BATCHES, device)
            trajectory.append({
                'step': step,
                'task_loss': task_loss.item(),
                'reg_loss': losses_reg[-1],
                'total_loss': loss.item() if not isinstance(loss, float) else loss,
                **cs,
                'val_acc': ev['acc'],
                'val_loss': ev['loss'],
            })

    final_eval = evaluate(model, EVAL_BATCHES, device)
    final_cs = get_c_stats(model)

    return {
        'lam': lam,
        'trajectory': trajectory,
        'final_val_acc': final_eval['acc'],
        'final_val_loss': final_eval['loss'],
        'final_task_loss': losses_task[-1],
        'c_final': final_cs,
        'c_drift_in': abs(final_cs['c_in_mean'] - C_INIT),
        'c_drift_hid': abs(final_cs['c_hid_mean'] - C_INIT),
        'model': model,
    }


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main():
    device = 'cpu'

    print("=" * 80)
    print("  A/B TEST: C REGULARIZATION (λ × mean((C - π)²))")
    print("=" * 80)
    print(f"  Task: add1, Steps: {STEPS}, Seed: {SEED}, Eval: {EVAL_BATCHES} batches")
    print(f"  λ values: {LAMBDAS}")
    print(f"  C init: π ≈ {C_INIT:.4f}, bounds: [{_C19_C_MIN}, {_C19_C_MAX}]")
    print("=" * 80)

    results = []
    for lam in LAMBDAS:
        t0 = time.perf_counter()
        print(f"\n{'─' * 80}")
        print(f"  λ = {lam}")
        r = train_with_reg(lam, device)
        elapsed = time.perf_counter() - t0
        r['elapsed'] = elapsed
        results.append(r)

        # Print trajectory
        print(f"  {'step':>6} {'task_loss':>10} {'reg_loss':>10} {'val_acc':>8}"
              f" {'C_in':>7} {'C_hid':>7} {'C_in_std':>8} {'C_hid_std':>9}")
        for t in r['trajectory']:
            print(f"  {t['step']:>6d} {t['task_loss']:>10.4f} {t['reg_loss']:>10.6f}"
                  f" {t['val_acc']*100:>7.1f}%"
                  f" {t['c_in_mean']:>7.4f} {t['c_hid_mean']:>7.4f}"
                  f" {t['c_in_std']:>8.4f} {t['c_hid_std']:>9.4f}")
        print(f"  [{elapsed:.1f}s]")

    # ── Comparison table ──
    print(f"\n{'=' * 80}")
    print("  COMPARISON")
    print(f"{'=' * 80}")
    print(f"\n  {'λ':>8} {'Val Acc':>8} {'Val Loss':>9} {'Task Loss':>10}"
          f" {'C_in':>7} {'C_hid':>7} {'Drift_in':>9} {'Drift_hid':>10}"
          f" {'C_in_std':>9} {'C_hid_std':>10}")
    print(f"  {'─'*8} {'─'*8} {'─'*9} {'─'*10}"
          f" {'─'*7} {'─'*7} {'─'*9} {'─'*10}"
          f" {'─'*9} {'─'*10}")

    best_acc = max(r['final_val_acc'] for r in results)
    for r in results:
        marker = " ←" if r['final_val_acc'] == best_acc else ""
        cs = r['c_final']
        print(f"  {r['lam']:>8.0e} {r['final_val_acc']*100:>7.1f}% {r['final_val_loss']:>9.4f}"
              f" {r['final_task_loss']:>10.4f}"
              f" {cs['c_in_mean']:>7.4f} {cs['c_hid_mean']:>7.4f}"
              f" {r['c_drift_in']:>9.4f} {r['c_drift_hid']:>10.4f}"
              f" {cs['c_in_std']:>9.4f} {cs['c_hid_std']:>10.4f}{marker}")

    # ── Analysis ──
    baseline = results[0]  # λ=0
    print(f"\n  ANALYSIS vs baseline (λ=0):")
    for r in results[1:]:
        acc_delta = r['final_val_acc'] - baseline['final_val_acc']
        loss_delta = r['final_val_loss'] - baseline['final_val_loss']
        drift_in_ratio = r['c_drift_in'] / max(baseline['c_drift_in'], 1e-6)
        drift_hid_ratio = r['c_drift_hid'] / max(baseline['c_drift_hid'], 1e-6)
        print(f"    λ={r['lam']:.0e}: Δacc={acc_delta*100:+.1f}%  Δloss={loss_delta:+.4f}"
              f"  drift_in={drift_in_ratio:.0%} of baseline"
              f"  drift_hid={drift_hid_ratio:.0%} of baseline")

    # ── Verdict ──
    print(f"\n{'=' * 80}")
    reg_helps = any(r['final_val_acc'] > baseline['final_val_acc'] + 0.005 for r in results[1:])
    reg_hurts = any(r['final_val_acc'] < baseline['final_val_acc'] - 0.02 for r in results[1:])

    if reg_helps:
        best_reg = max(results[1:], key=lambda r: r['final_val_acc'])
        print(f"  VERDICT: Regularization HELPS — best λ={best_reg['lam']:.0e}"
              f" ({best_reg['final_val_acc']*100:.1f}% vs {baseline['final_val_acc']*100:.1f}%)")
    elif reg_hurts:
        worst_reg = min(results[1:], key=lambda r: r['final_val_acc'])
        print(f"  VERDICT: Regularization HURTS — λ={worst_reg['lam']:.0e}"
              f" drops to {worst_reg['final_val_acc']*100:.1f}%"
              f" vs baseline {baseline['final_val_acc']*100:.1f}%")
    else:
        print(f"  VERDICT: Regularization is NEUTRAL — no significant effect on accuracy")
    print(f"{'=' * 80}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
