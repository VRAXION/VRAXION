"""Gradient flow diagnostic: measures gradient magnitudes at key NTM components.

This doesn't measure task accuracy — it directly shows whether circuit fixes
improve gradient flow through ptr, gate, and memory paths.

Single forward+backward pass = instant results. No training needed.
"""

import sys, time
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
for subdir in ('model', 'training', 'datagen'):
    p = str(ROOT / subdir)
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch
import torch.nn.functional as F
from instnct import INSTNCT

BATCH, SEQ_LEN = 4, 64
SEED = 42
N_PASSES = 10  # average over multiple random batches for stability

MODEL_CFG_BASE = dict(
    M=64, hidden_dim=128, slot_dim=32, N=2, R=1,
    embed_mode=True,
    kernel_mode='vshape',
    pointer_mode='pilot',
    write_mode='replace',
    embed_encoding='bitlift',
    output_encoding='lowrank_c19',
    checkpoint_chunks=0,
    # circuit fixes OFF
    ring_decay=False,
    gate_bias=False,
    expert_output_weights=False,
    ptr_gradient=False,
)

MODEL_CFG_FIXED = dict(
    **{k: v for k, v in MODEL_CFG_BASE.items()
       if k not in ('ring_decay', 'gate_bias', 'expert_output_weights',
                     'ptr_gradient', 'write_mode')},
    write_mode='additive',
    ring_decay=True,
    gate_bias=True,
    expert_output_weights=True,
    ptr_gradient=True,
)


def make_batch(seed, device='cpu'):
    """Random echo batch."""
    rng = np.random.RandomState(seed)
    BLOCK, REPEAT = 16, 4
    n_bytes = BATCH * (SEQ_LEN + 1) + BLOCK * REPEAT * 4
    raw_data, raw_mask = [], []
    while len(raw_data) < n_bytes:
        sb = rng.randint(0, 256, size=BLOCK, dtype=np.uint8)
        for r in range(REPEAT):
            raw_data.extend(sb)
            raw_mask.extend([0] * BLOCK if r == 0 else [1] * BLOCK)
    raw_data = np.array(raw_data[:n_bytes], dtype=np.uint8)
    raw_mask = np.array(raw_mask[:n_bytes], dtype=np.uint8)

    x_np = np.zeros((BATCH, SEQ_LEN), dtype=np.int64)
    y_np = np.zeros((BATCH, SEQ_LEN), dtype=np.int64)
    mask_np = np.zeros((BATCH, SEQ_LEN), dtype=np.float32)
    for i in range(BATCH):
        off = i * SEQ_LEN
        x_np[i] = raw_data[off:off + SEQ_LEN]
        y_np[i] = raw_data[off + 1:off + SEQ_LEN + 1]
        mask_np[i] = raw_mask[off + 1:off + SEQ_LEN + 1]
    return (torch.from_numpy(x_np).to(device),
            torch.from_numpy(y_np).to(device),
            torch.from_numpy(mask_np).to(device))


def classify_param(name):
    """Classify parameter into functional group."""
    name_l = name.lower()
    if 'ptr' in name_l or 'pilot' in name_l or 'pointer' in name_l or 'jump' in name_l:
        return 'ptr/pilot'
    if 'gate' in name_l or 'alpha' in name_l:
        return 'gate'
    if 'ring' in name_l or 'slot' in name_l or 'write' in name_l or 'read' in name_l:
        return 'memory'
    if 'expert' in name_l or 'brain' in name_l or 'fc' in name_l or 'hidden' in name_l:
        return 'expert'
    if 'embed' in name_l or 'enc' in name_l or 'dec' in name_l or 'output' in name_l:
        return 'embed/output'
    if 'rho' in name_l or 'c19' in name_l or 'activ' in name_l:
        return 'activation'
    return 'other'


def measure_gradients(model, x, y, mask):
    """Single forward+backward, return per-group gradient stats."""
    model.zero_grad()
    out, _ = model(x)
    logits = out.view(-1, 256)
    targets = y.view(-1)
    m_flat = mask.view(-1)

    ce = F.cross_entropy(logits, targets, reduction='none')
    if m_flat.sum() > 0:
        loss = (ce * m_flat).sum() / m_flat.sum()
    else:
        loss = ce.mean()

    loss.backward()

    groups = defaultdict(lambda: {'norms': [], 'maxes': [], 'params': []})
    for name, p in model.named_parameters():
        if p.grad is not None:
            g = p.grad
            group = classify_param(name)
            gnorm = g.norm().item()
            gmax = g.abs().max().item()
            groups[group]['norms'].append(gnorm)
            groups[group]['maxes'].append(gmax)
            groups[group]['params'].append(name)

    return {g: {
        'mean_norm': np.mean(d['norms']),
        'max_norm': max(d['norms']),
        'mean_max': np.mean(d['maxes']),
        'n_params': len(d['params']),
        'zero_grad_count': sum(1 for n in d['norms'] if n < 1e-10),
        'params': d['params'],
    } for g, d in groups.items()}, loss.item()


def run_config(label, cfg, n_passes=N_PASSES):
    """Run n_passes forward+backward, average gradient stats."""
    all_stats = defaultdict(lambda: defaultdict(list))
    losses = []

    for i in range(n_passes):
        torch.manual_seed(SEED)  # same model init each time
        model = INSTNCT(**cfg)
        x, y, mask = make_batch(seed=SEED + i)
        stats, loss_val = measure_gradients(model, x, y, mask)
        losses.append(loss_val)
        for group, s in stats.items():
            for k, v in s.items():
                if k != 'params':
                    all_stats[group][k].append(v)
            all_stats[group]['params'] = s['params']  # keep last

    # Average
    avg = {}
    for group, d in all_stats.items():
        avg[group] = {
            'mean_norm': np.mean(d['mean_norm']),
            'max_norm': np.mean(d['max_norm']),
            'mean_max': np.mean(d['mean_max']),
            'n_params': int(np.mean(d['n_params'])),
            'zero_grad_count': int(np.mean(d['zero_grad_count'])),
            'params': d['params'],
        }
    return avg, np.mean(losses)


def main():
    print("=" * 78)
    print("  GRADIENT FLOW DIAGNOSTIC")
    print("  Measures gradient magnitudes at key NTM components")
    print(f"  Averaged over {N_PASSES} random batches, same model init")
    print("=" * 78)

    configs = [
        ("BASELINE (no fixes, replace)", MODEL_CFG_BASE),
        ("ALL FIXES (additive+decay)", MODEL_CFG_FIXED),
    ]

    all_results = {}
    for label, cfg in configs:
        print(f"\n  Measuring: {label} ...")
        t0 = time.perf_counter()
        stats, loss = run_config(label, cfg)
        elapsed = time.perf_counter() - t0
        all_results[label] = (stats, loss)
        print(f"  Done in {elapsed:.1f}s  (loss={loss:.4f})")

    # ── Comparison table ──
    groups_all = sorted(set().union(*(s.keys() for s, _ in all_results.values())))

    print()
    print("=" * 78)
    print("  GRADIENT FLOW COMPARISON")
    print("=" * 78)

    for group in groups_all:
        print(f"\n  --- {group.upper()} ---")
        print(f"  {'Config':<38} {'Mean‖g‖':>10} {'Max‖g‖':>10} "
              f"{'Mean|g|max':>12} {'#params':>8} {'#zero':>6}")
        print(f"  {'-'*38} {'-'*10} {'-'*10} {'-'*12} {'-'*8} {'-'*6}")

        vals = []
        for label, _ in configs:
            stats, _ = all_results[label]
            if group in stats:
                s = stats[group]
                vals.append(s['mean_norm'])
                print(f"  {label:<38} {s['mean_norm']:>10.6f} {s['max_norm']:>10.6f} "
                      f"{s['mean_max']:>12.6f} {s['n_params']:>8} {s['zero_grad_count']:>6}")
            else:
                vals.append(0.0)
                print(f"  {label:<38} {'N/A':>10} {'N/A':>10} {'N/A':>12} {'N/A':>8} {'N/A':>6}")

        # Ratio
        if len(vals) == 2 and vals[0] > 1e-12:
            ratio = vals[1] / vals[0]
            change = "IMPROVED" if ratio > 1.1 else ("WORSE" if ratio < 0.9 else "SIMILAR")
            print(f"  → Ratio (fixed/baseline): {ratio:.2f}x  [{change}]")
        elif len(vals) == 2 and vals[1] > 1e-12:
            print(f"  → Baseline had ZERO gradient, fixes ENABLED it!")

    # ── Per-parameter detail for key groups ──
    print()
    print("=" * 78)
    print("  DETAILED: Parameters with biggest gradient changes")
    print("=" * 78)

    baseline_stats, _ = all_results[configs[0][0]]
    fixed_stats, _ = all_results[configs[1][0]]

    # Show unique params in fixed that don't exist in baseline
    base_params = set()
    for g, s in baseline_stats.items():
        base_params.update(s['params'])
    fix_params = set()
    for g, s in fixed_stats.items():
        fix_params.update(s['params'])

    new_params = fix_params - base_params
    if new_params:
        print(f"\n  NEW trainable parameters (only in FIXED config):")
        for p in sorted(new_params):
            print(f"    + {p}")

    removed_params = base_params - fix_params
    if removed_params:
        print(f"\n  REMOVED parameters (only in BASELINE):")
        for p in sorted(removed_params):
            print(f"    - {p}")

    # Summary verdict
    print()
    print("=" * 78)
    print("  VERDICT")
    print("=" * 78)
    for group in ['ptr/pilot', 'gate', 'memory']:
        if group in baseline_stats and group in fixed_stats:
            b = baseline_stats[group]['mean_norm']
            f = fixed_stats[group]['mean_norm']
            if b > 1e-12:
                ratio = f / b
                if ratio > 1.1:
                    print(f"  {group:>12}: gradient {ratio:.1f}x STRONGER with fixes")
                elif ratio < 0.9:
                    print(f"  {group:>12}: gradient {ratio:.1f}x WEAKER with fixes")
                else:
                    print(f"  {group:>12}: gradient SIMILAR ({ratio:.2f}x)")
            elif f > 1e-12:
                print(f"  {group:>12}: gradient ENABLED by fixes (was zero!)")
            else:
                print(f"  {group:>12}: gradient zero in both configs")
        elif group in fixed_stats:
            print(f"  {group:>12}: NEW group enabled by fixes")

    print("=" * 78)


if __name__ == '__main__':
    main()
