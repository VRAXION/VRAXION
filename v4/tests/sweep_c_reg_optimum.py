"""Detailed sweep: find optimal C regularization around the λ≈1e-4 region.

Sweeps both flat L2 and φ-structured variants with fine granularity.

Grid:
  Flat L2:       λ ∈ {0, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3}
  φ-struct:      λ=φ⁻ⁿ for n ∈ {15,17,19,21,23} × asymmetry ∈ {none, dual-φ, inv-dual-φ}
                 × exponent ∈ {φ, 2.0}

Usage:
    python tests/sweep_c_reg_optimum.py
"""

import sys
import time
import math
from pathlib import Path
from itertools import product

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

# ═══════════════════════════════════════════════════════════════
#  Regularization configs
# ═══════════════════════════════════════════════════════════════

def make_reg_fn(lam, w_inp=1.0, w_hid=1.0, exp=2.0):
    """Create a regularization function with given params.

    Args:
        lam:   regularization strength
        w_inp: weight for C_input penalty
        w_hid: weight for C_hidden penalty
        exp:   penalty exponent (2.0 = L2, φ ≈ 1.618 = Lφ)
    """
    def reg_fn(model):
        if lam == 0.0:
            return 0.0
        c_inp = _C_from_raw(model.c19_C_input)
        c_hid = _C_from_raw(model.c19_C_hidden)
        pen_inp = (c_inp - C_INIT).abs().pow(exp).mean()
        pen_hid = (c_hid - C_INIT).abs().pow(exp).mean()
        return lam * (w_inp * pen_inp + w_hid * pen_hid)
    return reg_fn


def build_configs():
    """Build all sweep configurations."""
    configs = []

    # ── Baseline: no reg ──
    configs.append({
        'name': 'no_reg',
        'group': 'baseline',
        'reg_fn': make_reg_fn(0.0),
        'desc': 'λ=0',
    })

    # ── Flat L2 grid (fine) ──
    for lam in [2e-5, 5e-5, 7.5e-5, 1e-4, 1.5e-4, 2e-4, 3e-4, 5e-4, 1e-3]:
        configs.append({
            'name': f'flat_{lam:.0e}',
            'group': 'flat_L2',
            'reg_fn': make_reg_fn(lam, w_inp=1.0, w_hid=1.0, exp=2.0),
            'desc': f'λ={lam:.0e} sym L2',
        })

    # ── φ-structured: sweep φ⁻ⁿ × asymmetry × exponent ──
    phi_n_values = [15, 17, 19, 21, 23]
    asymmetries = [
        ('sym',      1.0,     1.0),      # symmetric
        ('dual-φ',   PHI_INV, PHI),      # input light, hidden heavy (original)
        ('inv-φ',    PHI,     PHI_INV),  # input heavy, hidden light (flipped)
    ]
    exponents = [
        ('Lφ', PHI),    # φ ≈ 1.618
        ('L2', 2.0),    # standard L2
    ]

    for n, (asym_name, w_i, w_h), (exp_name, exp_val) in product(
        phi_n_values, asymmetries, exponents
    ):
        lam = PHI ** (-n)
        name = f'φ⁻{n}_{asym_name}_{exp_name}'
        configs.append({
            'name': name,
            'group': 'phi_struct',
            'reg_fn': make_reg_fn(lam, w_inp=w_i, w_hid=w_h, exp=exp_val),
            'desc': f'λ=φ⁻{n}={lam:.2e} {asym_name} {exp_name}',
        })

    return configs

# ═══════════════════════════════════════════════════════════════
#  Training
# ═══════════════════════════════════════════════════════════════

def train_config(cfg, device='cpu'):
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
        reg = cfg['reg_fn'](model)
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

    return {
        **cfg,
        'checkpoints': checkpoints,
        'final_val_acc': final_eval['acc'],
        'final_val_loss': final_eval['loss'],
        'c_drift_in': abs(final_cs['c_in_mean'] - C_INIT),
        'c_drift_hid': abs(final_cs['c_hid_mean'] - C_INIT),
        'c_in_std': final_cs['c_in_std'],
        'c_hid_std': final_cs['c_hid_std'],
    }

# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main():
    device = 'cpu'
    configs = build_configs()

    print("=" * 100)
    print("  DETAILED SWEEP: C REGULARIZATION OPTIMUM")
    print("=" * 100)
    print(f"  Task: add1, Steps: {STEPS}, Seed: {SEED}")
    print(f"  Total configs: {len(configs)}")
    print(f"    Baseline: 1")
    print(f"    Flat L2:  {sum(1 for c in configs if c['group'] == 'flat_L2')}")
    print(f"    φ-struct: {sum(1 for c in configs if c['group'] == 'phi_struct')}")
    print("=" * 100)

    results = []
    t_total = time.perf_counter()

    for i, cfg in enumerate(configs):
        t0 = time.perf_counter()
        r = train_config(cfg, device)
        elapsed = time.perf_counter() - t0
        r['elapsed'] = elapsed
        results.append(r)
        print(f"  [{i+1:>2}/{len(configs)}] {cfg['name']:<28s}  "
              f"acc={r['final_val_acc']*100:5.1f}%  "
              f"loss={r['final_val_loss']:.4f}  "
              f"drift_in={r['c_drift_in']:.3f}  "
              f"drift_hid={r['c_drift_hid']:.3f}  "
              f"[{elapsed:.0f}s]")

    total_time = time.perf_counter() - t_total

    # ── Sort by accuracy ──
    ranked = sorted(results, key=lambda r: r['final_val_acc'], reverse=True)

    print(f"\n{'=' * 100}")
    print("  RANKING BY ACCURACY (top 15)")
    print(f"{'=' * 100}")
    print(f"  {'#':>3} {'Name':>28} {'Group':>12} {'Acc':>7} {'Loss':>8} "
          f"{'Drift_in':>9} {'Drift_hid':>10} {'C_in_std':>9} {'C_hid_std':>10}")
    print(f"  {'─'*3} {'─'*28} {'─'*12} {'─'*7} {'─'*8} "
          f"{'─'*9} {'─'*10} {'─'*9} {'─'*10}")

    for i, r in enumerate(ranked[:15]):
        print(f"  {i+1:>3} {r['name']:>28} {r['group']:>12} "
              f"{r['final_val_acc']*100:>6.1f}% {r['final_val_loss']:>8.4f} "
              f"{r['c_drift_in']:>9.4f} {r['c_drift_hid']:>10.4f} "
              f"{r['c_in_std']:>9.4f} {r['c_hid_std']:>10.4f}")

    # ── Best per group ──
    print(f"\n{'=' * 100}")
    print("  BEST PER GROUP")
    print(f"{'=' * 100}")
    for group in ['baseline', 'flat_L2', 'phi_struct']:
        group_results = [r for r in results if r['group'] == group]
        best = max(group_results, key=lambda r: r['final_val_acc'])
        print(f"  {group:>12}: {best['name']:<28s}  "
              f"acc={best['final_val_acc']*100:.1f}%  "
              f"loss={best['final_val_loss']:.4f}  "
              f"desc: {best['desc']}")

    # ── Convergence of top 5 ──
    print(f"\n{'=' * 100}")
    print("  CONVERGENCE: TOP 5 (val_acc % at step 200/500/1000)")
    print(f"{'=' * 100}")
    top5 = ranked[:5]
    hdr = f"  {'Name':>28}"
    for step in [200, 500, 1000]:
        hdr += f"  {'@'+str(step):>7}"
    print(hdr)
    for r in top5:
        line = f"  {r['name']:>28}"
        for cp in r['checkpoints']:
            line += f"  {cp['val_acc']*100:>6.1f}%"
        print(line)

    # ── Overall winner ──
    winner = ranked[0]
    runner = ranked[1]
    baseline = next(r for r in results if r['group'] == 'baseline')
    best_phi = max((r for r in results if r['group'] == 'phi_struct'),
                   key=lambda r: r['final_val_acc'])

    print(f"\n{'=' * 100}")
    print(f"  WINNER: {winner['name']}")
    print(f"    acc={winner['final_val_acc']*100:.1f}%  loss={winner['final_val_loss']:.4f}")
    print(f"    desc: {winner['desc']}")
    print(f"    vs baseline: {(winner['final_val_acc'] - baseline['final_val_acc'])*100:+.1f}%")
    print(f"    vs runner-up ({runner['name']}): "
          f"{(winner['final_val_acc'] - runner['final_val_acc'])*100:+.1f}%")
    print(f"\n  Best φ-structured: {best_phi['name']}")
    print(f"    acc={best_phi['final_val_acc']*100:.1f}%  desc: {best_phi['desc']}")
    print(f"    vs winner: {(best_phi['final_val_acc'] - winner['final_val_acc'])*100:+.1f}%")
    print(f"\n  Total time: {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"{'=' * 100}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
