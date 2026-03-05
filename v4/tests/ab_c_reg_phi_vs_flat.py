"""A/B test: flat λ=1e-4 L2 reg vs φ-structured regularization.

Arm A (flat):  λ=1e-4, symmetric L2 toward π:  λ × mean((C - π)²)
Arm B (φ):     λ=φ⁻¹⁹, dual-φ asymmetric reg: λ × (φ⁻¹·mean(|C_in - π|^φ) + φ·mean(|C_hid - π|^φ))

The φ arm uses φ three ways (not just as a number):
  1. λ = φ⁻¹⁹ ≈ 1.07e-4 (φ-derived strength, similar magnitude)
  2. Asymmetric weighting: C_input × φ⁻¹, C_hidden × φ — mirrors the
     dual-phi gain structure in _c19_activation (even→φ⁻¹, odd→φ)
  3. Penalty exponent p=φ instead of p=2 — Lφ norm instead of L2,
     softer near target, sharper far from target

Usage:
    python tests/ab_c_reg_phi_vs_flat.py
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
#  Constants
# ═══════════════════════════════════════════════════════════════

PHI = (1 + math.sqrt(5)) / 2       # φ ≈ 1.6180339887
PHI_INV = (math.sqrt(5) - 1) / 2   # 1/φ ≈ 0.6180339887
C_INIT = math.pi

# ── Config ──
BATCH = 8
SEQ_LEN = 32
LR = 1e-3
STEPS = 1000
SEED = 42
EVAL_BATCHES = 16

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

# ── Arm A: flat L2 reg (baseline winner) ──

def c_reg_flat(model, lam=1e-4):
    """Symmetric L2 toward π: λ × mean((C - π)²)."""
    if lam == 0.0:
        return 0.0
    c_inp = _C_from_raw(model.c19_C_input)
    c_hid = _C_from_raw(model.c19_C_hidden)
    return lam * ((c_inp - C_INIT).pow(2).mean() + (c_hid - C_INIT).pow(2).mean())

# ── Arm B: φ-structured reg ──

def c_reg_phi(model):
    """Dual-φ asymmetric Lφ regularization toward π.

    Three φ-structural elements:
      1. λ = φ⁻¹⁹ ≈ 1.07e-4  (comparable to 1e-4 but φ-derived)
      2. Asymmetric weights: C_input × φ⁻¹, C_hidden × φ
         — mirrors activation's dual-phi gain (even→φ⁻¹, odd→φ)
         — lighter penalty on input (exploratory), heavier on hidden (stability)
      3. Exponent p = φ instead of 2 (Lφ norm)
         — softer gradient near π (allows micro-drift)
         — sharper penalty far from π (prevents blowup)
    """
    lam = PHI ** (-19)  # ≈ 1.0696e-4

    c_inp = _C_from_raw(model.c19_C_input)
    c_hid = _C_from_raw(model.c19_C_hidden)

    # |C - π|^φ — Lφ penalty (φ ≈ 1.618, between L1 and L2)
    pen_inp = (c_inp - C_INIT).abs().pow(PHI).mean()
    pen_hid = (c_hid - C_INIT).abs().pow(PHI).mean()

    # Dual-φ asymmetric weighting
    return lam * (PHI_INV * pen_inp + PHI * pen_hid)

# ═══════════════════════════════════════════════════════════════
#  Training
# ═══════════════════════════════════════════════════════════════

def train_arm(name, reg_fn, device='cpu'):
    """Train one arm with the given regularization function."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.use_deterministic_algorithms(False)

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
        reg = reg_fn(model)
        loss = task_loss + reg

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()

        losses_task.append(task_loss.item())
        losses_reg.append(reg if isinstance(reg, float) else reg.item())

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
        'name': name,
        'trajectory': trajectory,
        'final_val_acc': final_eval['acc'],
        'final_val_loss': final_eval['loss'],
        'final_task_loss': losses_task[-1],
        'c_final': final_cs,
        'c_drift_in': abs(final_cs['c_in_mean'] - C_INIT),
        'c_drift_hid': abs(final_cs['c_hid_mean'] - C_INIT),
        'model': model,
    }


def evaluate(model, n_batches, device):
    model.eval()
    rng = np.random.RandomState(9999)
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
#  Main
# ═══════════════════════════════════════════════════════════════

def main():
    device = 'cpu'

    print("=" * 80)
    print("  A/B TEST: FLAT L2 REG vs φ-STRUCTURED REG")
    print("=" * 80)
    print(f"  Task: add1, Steps: {STEPS}, Seed: {SEED}, Eval: {EVAL_BATCHES} batches")
    print(f"  Arm A (flat):  λ=1e-4, symmetric L2 → π")
    print(f"  Arm B (φ):     λ=φ⁻¹⁹≈{PHI**(-19):.6f}, dual-φ asymmetric Lφ → π")
    print(f"    ├─ λ = φ⁻¹⁹ ≈ {PHI**(-19):.2e}")
    print(f"    ├─ weights: C_input × φ⁻¹={PHI_INV:.4f}, C_hidden × φ={PHI:.4f}")
    print(f"    └─ exponent: p=φ={PHI:.4f} (Lφ norm, between L1 and L2)")
    print(f"  C init: π ≈ {C_INIT:.4f}, bounds: [{_C19_C_MIN}, {_C19_C_MAX}]")
    print("=" * 80)

    # ── Arm A: flat ──
    print(f"\n{'─' * 80}")
    print("  ARM A: flat L2 (λ=1e-4)")
    t0 = time.perf_counter()
    result_a = train_arm("flat_L2", lambda m: c_reg_flat(m, lam=1e-4), device)
    elapsed_a = time.perf_counter() - t0
    _print_trajectory(result_a, elapsed_a)

    # ── Arm B: φ-structured ──
    print(f"\n{'─' * 80}")
    print("  ARM B: φ-structured (λ=φ⁻¹⁹, dual-φ weights, Lφ norm)")
    t0 = time.perf_counter()
    result_b = train_arm("phi_struct", c_reg_phi, device)
    elapsed_b = time.perf_counter() - t0
    _print_trajectory(result_b, elapsed_b)

    # ── Head-to-head ──
    print(f"\n{'=' * 80}")
    print("  HEAD-TO-HEAD COMPARISON")
    print(f"{'=' * 80}")

    hdr = f"  {'Arm':>16} {'Val Acc':>8} {'Val Loss':>9} {'Task Loss':>10}"
    hdr += f" {'C_in':>7} {'C_hid':>7} {'Drift_in':>9} {'Drift_hid':>10}"
    hdr += f" {'C_in_std':>9} {'C_hid_std':>10}"
    print(hdr)
    print(f"  {'─' * 16} {'─' * 8} {'─' * 9} {'─' * 10}"
          f" {'─' * 7} {'─' * 7} {'─' * 9} {'─' * 10}"
          f" {'─' * 9} {'─' * 10}")

    for r in [result_a, result_b]:
        cs = r['c_final']
        marker = " ←" if r['final_val_acc'] >= max(result_a['final_val_acc'],
                                                      result_b['final_val_acc']) else ""
        print(f"  {r['name']:>16} {r['final_val_acc']*100:>7.1f}% {r['final_val_loss']:>9.4f}"
              f" {r['final_task_loss']:>10.4f}"
              f" {cs['c_in_mean']:>7.4f} {cs['c_hid_mean']:>7.4f}"
              f" {r['c_drift_in']:>9.4f} {r['c_drift_hid']:>10.4f}"
              f" {cs['c_in_std']:>9.4f} {cs['c_hid_std']:>10.4f}{marker}")

    # ── Δ analysis ──
    acc_delta = result_b['final_val_acc'] - result_a['final_val_acc']
    loss_delta = result_b['final_val_loss'] - result_a['final_val_loss']
    drift_in_ratio = result_b['c_drift_in'] / max(result_a['c_drift_in'], 1e-6)
    drift_hid_ratio = result_b['c_drift_hid'] / max(result_a['c_drift_hid'], 1e-6)

    print(f"\n  Δ (φ vs flat):")
    print(f"    Accuracy:  {acc_delta*100:+.2f}%")
    print(f"    Val Loss:  {loss_delta:+.4f}")
    print(f"    Drift_in:  {drift_in_ratio:.0%} of flat")
    print(f"    Drift_hid: {drift_hid_ratio:.0%} of flat")

    # ── Per-step convergence comparison ──
    print(f"\n  CONVERGENCE (val_acc % at each checkpoint):")
    print(f"  {'step':>6}  {'flat':>8}  {'φ-struct':>8}  {'Δ':>8}")
    for ta, tb in zip(result_a['trajectory'], result_b['trajectory']):
        d = (tb['val_acc'] - ta['val_acc']) * 100
        print(f"  {ta['step']:>6d}  {ta['val_acc']*100:>7.1f}%  {tb['val_acc']*100:>7.1f}%  {d:>+7.1f}%")

    # ── Verdict ──
    print(f"\n{'=' * 80}")
    THRESHOLD = 0.005  # 0.5% significance threshold
    if acc_delta > THRESHOLD:
        print(f"  VERDICT: φ-structured reg WINS (+{acc_delta*100:.1f}% accuracy)")
        print(f"           φ⁻¹⁹ λ + dual-φ asymmetry + Lφ norm > flat L2")
    elif acc_delta < -THRESHOLD:
        print(f"  VERDICT: flat L2 WINS ({-acc_delta*100:.1f}% better)")
        print(f"           simple 1e-4 symmetric L2 > φ-structured")
    else:
        print(f"  VERDICT: DRAW — both within {THRESHOLD*100:.1f}% (Δ={acc_delta*100:+.2f}%)")
        if abs(loss_delta) > 0.01:
            better = "φ-struct" if loss_delta < 0 else "flat"
            print(f"           (but {better} has slightly better val loss)")
    print(f"{'=' * 80}")

    return 0


def _print_trajectory(result, elapsed):
    print(f"  {'step':>6} {'task_loss':>10} {'reg_loss':>10} {'val_acc':>8}"
          f" {'C_in':>7} {'C_hid':>7} {'C_in_std':>8} {'C_hid_std':>9}")
    for t in result['trajectory']:
        print(f"  {t['step']:>6d} {t['task_loss']:>10.4f} {t['reg_loss']:>10.6f}"
              f" {t['val_acc']*100:>7.1f}%"
              f" {t['c_in_mean']:>7.4f} {t['c_hid_mean']:>7.4f}"
              f" {t['c_in_std']:>8.4f} {t['c_hid_std']:>9.4f}")
    print(f"  [{elapsed:.1f}s]")


if __name__ == '__main__':
    sys.exit(main())
