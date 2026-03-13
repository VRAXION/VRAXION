"""Sweep: C19 tail limit — phi-power K values vs integer K=6.

Tests whether the optimal tail boundary aligns with golden-ratio
powers, given the system is already phi-native (dual-phi gain,
φ⁻¹⁷ regularization).

Grid:
  K ∈ {φ, φ², φ³, φ⁴, 6, φ⁵, 2π, 6φ⁻¹, 6φ}
  (6 = current default for comparison)

With the winning C regularization: λ=φ⁻¹⁷, inv-φ, L2.

Usage:
    python tests/sweep_c19_tail_phi_powers.py
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
from instnct import INSTNCT, _C_from_raw, _c19_activation

# ═══════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════

PHI = (1 + math.sqrt(5)) / 2
PHI_INV = (math.sqrt(5) - 1) / 2
C_INIT = math.pi

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

# Winning C reg from sweep: λ=φ⁻¹⁷, inv-φ asymmetry, L2
REG_LAM = PHI ** (-17)
REG_W_INP = PHI       # input heavy
REG_W_HID = PHI_INV   # hidden light

# Phi-power tail limits to sweep
TAIL_LIMITS = [
    ('φ¹',     PHI ** 1),        # 1.618
    ('φ²',     PHI ** 2),        # 2.618
    ('φ³',     PHI ** 3),        # 4.236
    ('6φ⁻¹',   6 * PHI_INV),    # 3.708 — six scaled by inverse phi
    ('φ⁴',     PHI ** 4),        # 6.854
    ('6',      6.0),             # current default (integer baseline)
    ('2π',     2 * math.pi),     # 6.283 — two pi (natural period)
    ('φ⁵',     PHI ** 5),        # 11.09
    ('6φ',     6 * PHI),         # 9.708 — six scaled by phi
]

# Sort by value for clean reporting
TAIL_LIMITS.sort(key=lambda t: t[1])

# ═══════════════════════════════════════════════════════════════
#  Monkeypatch: replace _c19_activation with configurable limit
# ═══════════════════════════════════════════════════════════════

_CURRENT_TAIL_MULT = 6.0

_original_c19 = _c19_activation

def _c19_with_limit(x, rho=4.0, C=None):
    """C19 with configurable tail limit multiplier."""
    if C is None:
        C = instnct_module._C19_C
    l = _CURRENT_TAIL_MULT * C
    inv_c = 1.0 / C
    scaled = x * inv_c
    n = torch.floor(scaled)
    t = scaled - n
    h = t - t * t
    odd = torch.remainder(n, 2.0)
    sgn = 1.0 - 2.0 * odd
    gain = odd * (PHI - PHI_INV) + PHI_INV
    core = C * h * (sgn + rho * h) * gain
    return torch.where(x.abs() > l, x - x.sign() * l, core)


def set_tail_limit(k):
    global _CURRENT_TAIL_MULT
    _CURRENT_TAIL_MULT = float(k)
    instnct_module._c19_activation = _c19_with_limit


def restore_c19():
    instnct_module._c19_activation = _original_c19


# ═══════════════════════════════════════════════════════════════
#  Task & eval
# ═══════════════════════════════════════════════════════════════

def make_add1_batch(batch, seq_len, rng):
    x = rng.randint(0, 256, size=(batch, seq_len)).astype(np.int64)
    y = (x + 1) % 256
    return x, y


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


def get_c_stats(model):
    with torch.no_grad():
        c_inp = _C_from_raw(model.c19_C_input)
        c_hid = _C_from_raw(model.c19_C_hidden)
    return {
        'c_in_mean': c_inp.mean().item(),
        'c_in_std': c_inp.std().item(),
        'c_hid_mean': c_hid.mean().item(),
        'c_hid_std': c_hid.std().item(),
    }


def c_reg(model):
    c_inp = _C_from_raw(model.c19_C_input)
    c_hid = _C_from_raw(model.c19_C_hidden)
    pen_inp = ((c_inp - C_INIT) ** 2).mean()
    pen_hid = ((c_hid - C_INIT) ** 2).mean()
    return REG_LAM * (REG_W_INP * pen_inp + REG_W_HID * pen_hid)


# ═══════════════════════════════════════════════════════════════
#  Tail usage probe
# ═══════════════════════════════════════════════════════════════

def probe_tail_usage(model, device, rng):
    model.eval()
    with torch.no_grad():
        x_np, _ = make_add1_batch(BATCH, SEQ_LEN, rng)
        x = torch.from_numpy(x_np).to(device)
        bits = ((x.unsqueeze(-1) >> torch.arange(8, device=device)) & 1).float()
        pre_input = model.inp(bits)
        c_inp = _C_from_raw(model.c19_C_input)
        limit_inp = _CURRENT_TAIL_MULT * c_inp
        in_tail_inp = (pre_input.abs() > limit_inp).float().mean().item()
    return {'input_tail_frac': in_tail_inp}


# ═══════════════════════════════════════════════════════════════
#  Training
# ═══════════════════════════════════════════════════════════════

def train_one(tail_k, device='cpu'):
    set_tail_limit(tail_k)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.use_deterministic_algorithms(False)

    model = INSTNCT(**MODEL_CFG).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    rng = np.random.RandomState(SEED)

    checkpoints = []

    for step in range(1, STEPS + 1):
        model.train()
        x_np, y_np = make_add1_batch(BATCH, SEQ_LEN, rng)
        x = torch.from_numpy(x_np).to(device)
        y = torch.from_numpy(y_np).to(device)

        out, _ = model(x)
        logits = out.view(-1, 256)
        targets = y.view(-1)
        task_loss = F.cross_entropy(logits, targets)
        reg = c_reg(model)
        loss = task_loss + reg

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()

        if step in (200, 500, 1000):
            ev = evaluate(model, EVAL_BATCHES, device)
            cs = get_c_stats(model)
            checkpoints.append({
                'step': step,
                'val_acc': ev['acc'],
                'val_loss': ev['loss'],
                'task_loss': task_loss.item(),
                **cs,
            })

    final_eval = evaluate(model, EVAL_BATCHES, device)
    final_cs = get_c_stats(model)

    probe_rng = np.random.RandomState(7777)
    tail_probe = probe_tail_usage(model, device, probe_rng)

    restore_c19()

    return {
        'tail_k': tail_k,
        'checkpoints': checkpoints,
        'final_val_acc': final_eval['acc'],
        'final_val_loss': final_eval['loss'],
        'c_drift_in': abs(final_cs['c_in_mean'] - C_INIT),
        'c_drift_hid': abs(final_cs['c_hid_mean'] - C_INIT),
        'c_in_mean': final_cs['c_in_mean'],
        'c_hid_mean': final_cs['c_hid_mean'],
        'input_tail_frac': tail_probe['input_tail_frac'],
    }


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main():
    device = 'cpu'

    print("=" * 110)
    print("  SWEEP: C19 TAIL LIMIT — PHI POWERS vs INTEGER K=6")
    print("=" * 110)
    print(f"  Task: add1, Steps: {STEPS}, Seed: {SEED}")
    print(f"  C reg: λ=φ⁻¹⁷={REG_LAM:.2e}, inv-φ (w_inp=φ, w_hid=φ⁻¹), L2")
    print(f"  C_init = π ≈ {C_INIT:.4f}")
    print()
    print(f"  Tail limits ({len(TAIL_LIMITS)} configs):")
    for name, val in TAIL_LIMITS:
        boundary = val * C_INIT
        arches = val / 2
        print(f"    {name:>6} = {val:>7.3f}  →  boundary = {boundary:>7.3f}  "
              f"(~{arches:.1f} arch pairs)")
    print("=" * 110)

    results = []
    t_total = time.perf_counter()

    for i, (name, k) in enumerate(TAIL_LIMITS):
        t0 = time.perf_counter()
        r = train_one(k, device)
        elapsed = time.perf_counter() - t0
        r['elapsed'] = elapsed
        r['name'] = name
        results.append(r)

        tail_pct = r['input_tail_frac'] * 100
        print(f"  [{i+1:>2}/{len(TAIL_LIMITS)}] {name:>6} (K={k:<7.3f})  "
              f"acc={r['final_val_acc']*100:5.1f}%  "
              f"loss={r['final_val_loss']:.4f}  "
              f"tail_hit={tail_pct:5.1f}%  "
              f"C_in={r['c_in_mean']:.3f}  "
              f"C_hid={r['c_hid_mean']:.3f}  "
              f"[{elapsed:.0f}s]")

    total_time = time.perf_counter() - t_total

    # ── Sort by accuracy ──
    ranked = sorted(results, key=lambda r: r['final_val_acc'], reverse=True)

    print(f"\n{'=' * 110}")
    print("  RANKING BY ACCURACY")
    print(f"{'=' * 110}")
    print(f"  {'#':>3} {'Name':>6} {'K':>7} {'Acc':>7} {'Loss':>8} {'Tail%':>7} "
          f"{'C_in':>7} {'C_hid':>7} {'Drift_in':>9} {'Drift_hid':>10}")
    print(f"  {'─'*3} {'─'*6} {'─'*7} {'─'*7} {'─'*8} {'─'*7} "
          f"{'─'*7} {'─'*7} {'─'*9} {'─'*10}")

    for i, r in enumerate(ranked):
        marker = " ◀ CURRENT" if r['name'] == '6' else ""
        print(f"  {i+1:>3} {r['name']:>6} {r['tail_k']:>7.3f} "
              f"{r['final_val_acc']*100:>6.1f}% {r['final_val_loss']:>8.4f} "
              f"{r['input_tail_frac']*100:>6.1f}% "
              f"{r['c_in_mean']:>7.3f} {r['c_hid_mean']:>7.3f} "
              f"{r['c_drift_in']:>9.4f} {r['c_drift_hid']:>10.4f}{marker}")

    # ── Convergence ──
    print(f"\n{'=' * 110}")
    print("  CONVERGENCE (val_acc % at step 200 / 500 / 1000)")
    print(f"{'=' * 110}")
    hdr = f"  {'Name':>6} {'K':>7}"
    for step in [200, 500, 1000]:
        hdr += f"  {'@'+str(step):>7}"
    print(hdr)
    for r in results:
        line = f"  {r['name']:>6} {r['tail_k']:>7.3f}"
        for cp in r['checkpoints']:
            line += f"  {cp['val_acc']*100:>6.1f}%"
        print(line)

    # ── Winner analysis ──
    winner = ranked[0]
    baseline_6 = next(r for r in results if r['name'] == '6')

    print(f"\n{'=' * 110}")
    print(f"  WINNER: {winner['name']} (K={winner['tail_k']:.3f})")
    print(f"    acc={winner['final_val_acc']*100:.1f}%  "
          f"loss={winner['final_val_loss']:.4f}  "
          f"tail_hit={winner['input_tail_frac']*100:.1f}%")

    if winner['name'] != '6':
        delta = (winner['final_val_acc'] - baseline_6['final_val_acc']) * 100
        print(f"    vs K=6 (current default): {delta:+.1f}%")
    else:
        print(f"    → K=6 still the best! No phi-power beats integer 6.")

    print(f"\n  K=6 baseline: acc={baseline_6['final_val_acc']*100:.1f}%  "
          f"tail_hit={baseline_6['input_tail_frac']*100:.1f}%")

    # ── Phi alignment analysis ──
    print(f"\n{'=' * 110}")
    print("  PHI ALIGNMENT ANALYSIS")
    print(f"{'=' * 110}")
    print(f"  6 sits between φ³={PHI**3:.3f} and φ⁴={PHI**4:.3f}")
    print(f"  Ratio 6/φ³ = {6/PHI**3:.3f}   6/φ⁴ = {6/PHI**4:.3f}")
    print(f"  10×φ⁻¹ = {10*PHI_INV:.3f}  (close to 6)")
    print(f"  2π = {2*math.pi:.3f}  (also close to 6)")

    # Check if a phi power beats 6
    phi_powers = [r for r in ranked if r['name'].startswith('φ')]
    if phi_powers:
        best_phi = phi_powers[0]
        phi_vs_6 = (best_phi['final_val_acc'] - baseline_6['final_val_acc']) * 100
        print(f"\n  Best phi-power: {best_phi['name']} "
              f"(K={best_phi['tail_k']:.3f}, acc={best_phi['final_val_acc']*100:.1f}%)")
        print(f"  vs K=6: {phi_vs_6:+.2f}%")
        if phi_vs_6 > 0.5:
            print(f"  → φ-aligned K is BETTER! Consider switching default to "
                  f"{best_phi['name']}={best_phi['tail_k']:.3f}")
        elif phi_vs_6 > -0.5:
            print(f"  → Comparable to K=6. The system is phi-compatible but 6 is fine.")
        else:
            print(f"  → K=6 remains superior. Integer boundary wins here.")

    print(f"\n  Total time: {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"{'=' * 110}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
