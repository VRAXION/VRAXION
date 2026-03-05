"""Sweep: C19 tail limit — how many periodic arches matter?

Tests the `l = K * C` boundary that separates the periodic core
from the linear tail in _c19_activation.

Grid:
  K ∈ {1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 1000}
  (1000 ≈ effectively infinite — pure periodic, no tail)

With the winning C regularization: λ=φ⁻¹⁷, inv-φ, L2.

Usage:
    python tests/sweep_c19_tail_limit.py
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
REG_W_INP = PHI       # input heavy (inv-φ)
REG_W_HID = PHI_INV   # hidden light

# Tail limit values to sweep
TAIL_LIMITS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 1000]

# ═══════════════════════════════════════════════════════════════
#  Monkeypatch: replace _c19_activation with configurable limit
# ═══════════════════════════════════════════════════════════════

_CURRENT_TAIL_MULT = 6.0  # global state for current sweep value

_original_c19 = _c19_activation  # save reference

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
    """Set the tail limit multiplier and monkeypatch."""
    global _CURRENT_TAIL_MULT
    _CURRENT_TAIL_MULT = float(k)
    instnct_module._c19_activation = _c19_with_limit


def restore_c19():
    """Restore original activation."""
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
    """Winning C regularization: λ=φ⁻¹⁷, inv-φ, L2."""
    c_inp = _C_from_raw(model.c19_C_input)
    c_hid = _C_from_raw(model.c19_C_hidden)
    pen_inp = ((c_inp - C_INIT) ** 2).mean()
    pen_hid = ((c_hid - C_INIT) ** 2).mean()
    return REG_LAM * (REG_W_INP * pen_inp + REG_W_HID * pen_hid)


# ═══════════════════════════════════════════════════════════════
#  Tail usage probe — measure what fraction of activations hit tail
# ═══════════════════════════════════════════════════════════════

_tail_stats = {'input_total': 0, 'input_tail': 0,
               'hidden_total': 0, 'hidden_tail': 0}

def reset_tail_stats():
    for k in _tail_stats:
        _tail_stats[k] = 0

def probe_tail_usage(model, device, rng):
    """Run one batch and measure what % of activations land in the tail."""
    model.eval()
    with torch.no_grad():
        x_np, _ = make_add1_batch(BATCH, SEQ_LEN, rng)
        x = torch.from_numpy(x_np).to(device)
        # Get input activations before c19
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

    # Probe tail usage at end of training
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

    print("=" * 100)
    print("  SWEEP: C19 TAIL LIMIT (K × C boundary)")
    print("=" * 100)
    print(f"  Task: add1, Steps: {STEPS}, Seed: {SEED}")
    print(f"  C reg: λ=φ⁻¹⁷={REG_LAM:.2e}, inv-φ (w_inp=φ, w_hid=φ⁻¹), L2")
    print(f"  Tail limits: {TAIL_LIMITS}")
    print(f"  Total configs: {len(TAIL_LIMITS)}")
    print("=" * 100)

    results = []
    t_total = time.perf_counter()

    for i, k in enumerate(TAIL_LIMITS):
        t0 = time.perf_counter()
        r = train_one(k, device)
        elapsed = time.perf_counter() - t0
        r['elapsed'] = elapsed
        results.append(r)

        tail_pct = r['input_tail_frac'] * 100
        print(f"  [{i+1:>2}/{len(TAIL_LIMITS)}] K={k:<5}  "
              f"acc={r['final_val_acc']*100:5.1f}%  "
              f"loss={r['final_val_loss']:.4f}  "
              f"tail_hit={tail_pct:5.1f}%  "
              f"C_in={r['c_in_mean']:.3f}  "
              f"C_hid={r['c_hid_mean']:.3f}  "
              f"[{elapsed:.0f}s]")

    total_time = time.perf_counter() - t_total

    # ── Sort by accuracy ──
    ranked = sorted(results, key=lambda r: r['final_val_acc'], reverse=True)

    print(f"\n{'=' * 100}")
    print("  RANKING BY ACCURACY")
    print(f"{'=' * 100}")
    print(f"  {'#':>3} {'K':>5} {'Acc':>7} {'Loss':>8} {'Tail%':>7} "
          f"{'C_in':>7} {'C_hid':>7} {'Drift_in':>9} {'Drift_hid':>10}")
    print(f"  {'─'*3} {'─'*5} {'─'*7} {'─'*8} {'─'*7} "
          f"{'─'*7} {'─'*7} {'─'*9} {'─'*10}")

    for i, r in enumerate(ranked):
        print(f"  {i+1:>3} {r['tail_k']:>5} "
              f"{r['final_val_acc']*100:>6.1f}% {r['final_val_loss']:>8.4f} "
              f"{r['input_tail_frac']*100:>6.1f}% "
              f"{r['c_in_mean']:>7.3f} {r['c_hid_mean']:>7.3f} "
              f"{r['c_drift_in']:>9.4f} {r['c_drift_hid']:>10.4f}")

    # ── Convergence ──
    print(f"\n{'=' * 100}")
    print("  CONVERGENCE (val_acc % at step 200 / 500 / 1000)")
    print(f"{'=' * 100}")
    hdr = f"  {'K':>5}"
    for step in [200, 500, 1000]:
        hdr += f"  {'@'+str(step):>7}"
    print(hdr)
    for r in results:
        line = f"  {r['tail_k']:>5}"
        for cp in r['checkpoints']:
            line += f"  {cp['val_acc']*100:>6.1f}%"
        print(line)

    # ── Winner ──
    winner = ranked[0]
    baseline_6 = next(r for r in results if r['tail_k'] == 6)

    print(f"\n{'=' * 100}")
    print(f"  WINNER: K={winner['tail_k']}")
    print(f"    acc={winner['final_val_acc']*100:.1f}%  loss={winner['final_val_loss']:.4f}")
    print(f"    tail_hit={winner['input_tail_frac']*100:.1f}%")
    if winner['tail_k'] != 6:
        print(f"    vs K=6 (current default): "
              f"{(winner['final_val_acc'] - baseline_6['final_val_acc'])*100:+.1f}%")
    print(f"\n  K=6 (current): acc={baseline_6['final_val_acc']*100:.1f}%  "
          f"tail_hit={baseline_6['input_tail_frac']*100:.1f}%")

    # ── Key insight ──
    print(f"\n  KEY INSIGHT:")
    k1 = next(r for r in results if r['tail_k'] == 1)
    k1000 = next(r for r in results if r['tail_k'] == 1000)
    spread = abs(ranked[0]['final_val_acc'] - ranked[-1]['final_val_acc']) * 100
    print(f"    K=1 (1 arch each side):  acc={k1['final_val_acc']*100:.1f}%  "
          f"tail={k1['input_tail_frac']*100:.1f}%")
    print(f"    K=1000 (no tail):        acc={k1000['final_val_acc']*100:.1f}%  "
          f"tail={k1000['input_tail_frac']*100:.1f}%")
    print(f"    Spread (best-worst):     {spread:.1f}%")
    if spread < 2.0:
        print(f"    → Tail limit has MINIMAL impact — K=6 is fine as-is.")
    else:
        print(f"    → Tail limit MATTERS — worth investigating further!")

    print(f"\n  Total time: {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"{'=' * 100}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
