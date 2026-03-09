"""A/B test: does gated_write (erase+add) prevent dotprod gate alpha collapse?

Problem: when using dotprod gate (alpha = sigmoid(tau * cos_sim(input, ring_signal))),
alpha crashes / saturates because ring_norm grows unbounded via scatter_add random walk.
The ring signal dominates → cosine similarity becomes degenerate → gate collapses.

Hypothesis: gated_write (NTM-style erase+add) controls ring norm growth by decaying
old slot content before writing. This should keep ring_signal at healthy magnitudes
→ cosine gate stays centered → alpha doesn't collapse.

Conditions:
  A) dotprod gate + scatter_add (legacy write)   — expected: alpha drifts/collapses
  B) dotprod gate + gated_write (erase+add)      — expected: alpha stays healthy
  C) fixed S=0.3 + scatter_add (champion baseline) — reference

Metrics tracked every N steps:
  - alpha mean/min/max per expert
  - ring_norm, ring SVD rank, adjacent cos_sim
  - masked_acc, loss
  - ring_signal_norm vs input_norm ratio

Usage:
    python tests/bench_gated_write_alpha.py
    python tests/bench_gated_write_alpha.py --steps 1000 --device cuda
"""

import argparse
import math
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# ── Path setup ──
ROOT = Path(__file__).resolve().parent.parent
for subdir in ('model', 'training', 'datagen'):
    p = str(ROOT / subdir)
    if p not in sys.path:
        sys.path.insert(0, p)

from instnct import INSTNCT


# ═══════════════════════════════════════════════════════════
#  DATA: repeating pattern (same as bench_fast_memory.py)
# ═══════════════════════════════════════════════════════════

def generate_repeating_pattern(B, length, period, seed=42):
    data = np.zeros((B, length), dtype=np.int64)
    mask = np.zeros((B, length), dtype=np.float32)
    for b in range(B):
        rng = np.random.RandomState(seed + b)
        pattern = rng.randint(0, 256, size=period)
        for pos in range(length):
            data[b, pos] = pattern[pos % period]
        mask[b, period:] = 1.0
    return torch.from_numpy(data), torch.from_numpy(mask)


def masked_ce_loss(logits, targets, mask):
    B, T, V = logits.shape
    flat_logits = logits.reshape(-1, V)
    flat_targets = targets.reshape(-1)
    flat_mask = mask.reshape(-1)
    per_token = F.cross_entropy(flat_logits, flat_targets, reduction='none')
    n_sup = flat_mask.sum().clamp(min=1)
    return (per_token * flat_mask).sum() / n_sup, int(n_sup.item())


def masked_accuracy(logits, targets, mask):
    preds = logits.argmax(dim=-1)
    correct = (preds == targets).float()
    flat_mask = mask.reshape(-1)
    n_sup = flat_mask.sum().clamp(min=1)
    return (correct.reshape(-1) * flat_mask).sum() / n_sup


# ═══════════════════════════════════════════════════════════
#  RING DIAGNOSTICS
# ═══════════════════════════════════════════════════════════

def ring_diagnostics(state):
    """Ring health: norm, SVD rank, adjacent cosine similarity."""
    if state is None or 'ring' not in state:
        return {}
    ring = state['ring']  # (B, M, slot_dim)
    slot_norms = ring.norm(dim=-1)

    # Adjacent cosine similarity
    cos_adj = F.cosine_similarity(ring[:, :-1], ring[:, 1:], dim=-1)

    # SVD rank on first batch element
    ring_0 = ring[0]
    try:
        _, S_vals, _ = torch.svd(ring_0)
        total_var = (S_vals ** 2).sum()
        cumvar = (S_vals ** 2).cumsum(0) / total_var
        rank_90 = (cumvar < 0.90).sum().item() + 1
    except Exception:
        rank_90 = -1

    return {
        'ring_norm': ring.norm().item(),
        'ring_slot_norm_mean': slot_norms.mean().item(),
        'ring_adj_cos': cos_adj.mean().item(),
        'ring_svd_rank90': rank_90,
    }


# ═══════════════════════════════════════════════════════════
#  SINGLE CONDITION RUN
# ═══════════════════════════════════════════════════════════

def run_condition(label, S_mode, gated_write, steps, batch, seq, hidden_dim,
                  M, slot_dim, N, period, device, lr, log_every, seed):
    """Run one A/B condition with alpha tracking.

    S_mode: 'dotprod' or float (e.g. 0.3)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    total_len = (steps + 20) * seq
    data, mask = generate_repeating_pattern(B=batch, length=total_len, period=period, seed=seed)
    data, mask = data.to(device), mask.to(device)

    # Build model — dotprod mode needs S='dotprod' in config
    use_dotprod = (S_mode == 'dotprod')
    model = INSTNCT(
        M=M, hidden_dim=hidden_dim, slot_dim=slot_dim,
        N=N, R=1, embed_mode=True,
        kernel_mode='vshape',
        embed_encoding='bitlift',
        output_encoding='lowrank_c19',
        expert_weighting=False,
        checkpoint_chunks=0,
        bb_enabled=False,
        gated_write=gated_write,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    warmup = min(50, steps // 10)

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  S_mode={'dotprod' if use_dotprod else S_mode} | gated_write={gated_write}")
    print(f"  hidden={hidden_dim} M={M} N={N} | {n_params:,} params | {device}")
    print(f"{'='*60}")

    # Determine S value for forward pass
    S_val = 'dotprod' if use_dotprod else S_mode

    state = None
    pos = 0
    history = []
    t0 = time.perf_counter()

    for step in range(1, steps + 1):
        # LR warmup
        cur_lr = lr * min(1.0, step / max(warmup, 1))
        for pg in opt.param_groups:
            pg['lr'] = cur_lr

        x = data[:, pos:pos + seq]
        y = data[:, pos + 1:pos + seq + 1]
        m = mask[:, pos + 1:pos + seq + 1]

        logits, new_state = model(x, S=S_val, state=state)

        if new_state is not None:
            state = {k: v.detach() for k, v in new_state.items()}

        loss, n_sup = masked_ce_loss(logits, y, m)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()

        pos += seq
        if pos + seq + 1 >= total_len:
            pos = 0
            state = None

        # ── Log ──
        if step % log_every == 0 or step == steps:
            with torch.no_grad():
                acc = masked_accuracy(logits, y, m).item()
                diag = model._diag.copy()
                rd = ring_diagnostics(state)

            row = {
                'step': step,
                'acc': acc,
                'loss': loss.item(),
            }

            # Alpha stats (only for dotprod mode)
            if use_dotprod:
                for i in range(N):
                    row[f'alpha_{i}_mean'] = diag.get(f'alpha_{i}_mean', -1)
                    row[f'alpha_{i}_min'] = diag.get(f'alpha_{i}_min', -1)
                    row[f'alpha_{i}_max'] = diag.get(f'alpha_{i}_max', -1)
                    row[f'ring_signal_norm_{i}'] = diag.get(f'ring_signal_norm_{i}', -1)
                    row[f'input_norm_{i}'] = diag.get(f'input_norm_{i}', -1)

            row.update(rd)
            history.append(row)

            # Print
            alpha_str = ''
            if use_dotprod:
                alphas = [f"α{i}={diag.get(f'alpha_{i}_mean', -1):.3f}" for i in range(N)]
                rs_norms = [f"rs{i}={diag.get(f'ring_signal_norm_{i}', -1):.0f}" for i in range(N)]
                alpha_str = f" | {' '.join(alphas)} | {' '.join(rs_norms)}"

            ring_str = ''
            if rd:
                ring_str = f" | rn={rd['ring_norm']:.0f} svd90={rd['ring_svd_rank90']}"

            print(f"  step {step:5d} | acc={acc*100:5.1f}% | loss={loss.item():.3f}"
                  f"{alpha_str}{ring_str}")

    wall = time.perf_counter() - t0
    print(f"  Done in {wall:.1f}s ({wall/steps:.3f} s/step)")

    return {
        'label': label,
        'history': history,
        'n_params': n_params,
        'wall_time': wall,
    }


# ═══════════════════════════════════════════════════════════
#  COMPARISON TABLE
# ═══════════════════════════════════════════════════════════

def print_comparison(results, N):
    """Print side-by-side comparison of all conditions."""
    print(f"\n{'='*80}")
    print("  A/B COMPARISON: gated_write vs scatter_add — alpha collapse test")
    print(f"{'='*80}\n")

    headers = ['Condition', 'Peak Acc', 'Final Acc', 'Final Loss']
    has_alpha = any(f'alpha_0_mean' in r['history'][-1] for r in results if r['history'])
    if has_alpha:
        for i in range(N):
            headers += [f'α{i} final', f'α{i} range']
    headers += ['Ring Norm', 'SVD r90', 'Adj Cos', 'Params']

    # Build rows
    rows = []
    for res in results:
        h = res['history']
        if not h:
            continue
        last = h[-1]
        peak_acc = max(r['acc'] for r in h) * 100
        final_acc = last['acc'] * 100
        final_loss = last['loss']

        row = [res['label'], f'{peak_acc:.1f}%', f'{final_acc:.1f}%', f'{final_loss:.3f}']

        if has_alpha:
            for i in range(N):
                a_mean = last.get(f'alpha_{i}_mean', -1)
                a_min_first = h[0].get(f'alpha_{i}_min', -1)
                a_max_first = h[0].get(f'alpha_{i}_max', -1)
                a_min_last = last.get(f'alpha_{i}_min', -1)
                a_max_last = last.get(f'alpha_{i}_max', -1)
                if a_mean >= 0:
                    row.append(f'{a_mean:.3f}')
                    row.append(f'[{a_min_last:.2f}-{a_max_last:.2f}]')
                else:
                    row.append('n/a')
                    row.append('n/a')

        rn = last.get('ring_norm', -1)
        svd = last.get('ring_svd_rank90', -1)
        adj = last.get('ring_adj_cos', -1)
        row.append(f'{rn:.0f}' if rn >= 0 else 'n/a')
        row.append(str(svd) if svd >= 0 else 'n/a')
        row.append(f'{adj:.4f}' if adj >= 0 else 'n/a')
        row.append(f'{res["n_params"]:,}')

        rows.append(row)

    # Print table
    col_widths = [max(len(headers[j]), max(len(r[j]) for r in rows)) for j in range(len(headers))]
    header_line = ' | '.join(h.ljust(w) for h, w in zip(headers, col_widths))
    sep_line = '-+-'.join('-' * w for w in col_widths)
    print(f"  {header_line}")
    print(f"  {sep_line}")
    for row in rows:
        print(f"  {' | '.join(c.ljust(w) for c, w in zip(row, col_widths))}")

    # ── Alpha trajectory comparison ──
    print(f"\n{'─'*60}")
    print("  ALPHA TRAJECTORY (mean per expert over training)")
    print(f"{'─'*60}")
    for res in results:
        h = res['history']
        if not h or f'alpha_0_mean' not in h[0]:
            continue
        print(f"\n  {res['label']}:")
        for i in range(N):
            vals = [f"{r.get(f'alpha_{i}_mean', -1):.3f}" for r in h]
            steps_s = [str(r['step']) for r in h]
            print(f"    Expert {i}: {' → '.join(vals)}")
            print(f"    Steps:    {' → '.join(steps_s)}")

    # ── Ring norm trajectory ──
    print(f"\n{'─'*60}")
    print("  RING NORM TRAJECTORY")
    print(f"{'─'*60}")
    for res in results:
        h = res['history']
        if not h:
            continue
        norms = [f"{r.get('ring_norm', -1):.0f}" for r in h]
        print(f"  {res['label']}: {' → '.join(norms)}")

    # ── Verdict ──
    print(f"\n{'='*80}")
    print("  VERDICT")
    print(f"{'='*80}")

    # Check if gated_write condition has healthier alpha than scatter_add
    dotprod_results = [r for r in results if 'dotprod' in r['label'].lower()]
    if len(dotprod_results) >= 2:
        scatter_res = [r for r in dotprod_results if 'scatter' in r['label'].lower()]
        gated_res = [r for r in dotprod_results if 'gated' in r['label'].lower()]

        if scatter_res and gated_res:
            s_last = scatter_res[0]['history'][-1]
            g_last = gated_res[0]['history'][-1]

            s_alphas = [s_last.get(f'alpha_{i}_mean', 0.5) for i in range(N)]
            g_alphas = [g_last.get(f'alpha_{i}_mean', 0.5) for i in range(N)]

            # Alpha health: how close to 0.5 (centered)?
            s_deviation = sum(abs(a - 0.5) for a in s_alphas) / N
            g_deviation = sum(abs(a - 0.5) for a in g_alphas) / N

            s_acc = s_last['acc'] * 100
            g_acc = g_last['acc'] * 100

            print(f"  Alpha centering (lower = healthier):")
            print(f"    scatter_add: avg |α-0.5| = {s_deviation:.3f}")
            print(f"    gated_write: avg |α-0.5| = {g_deviation:.3f}")
            print()
            print(f"  Accuracy:")
            print(f"    scatter_add: {s_acc:.1f}%")
            print(f"    gated_write: {g_acc:.1f}%")
            print()

            if g_deviation < s_deviation and g_acc >= s_acc - 1.0:
                print("  → gated_write HELPS: healthier alpha without accuracy loss")
            elif g_deviation < s_deviation and g_acc < s_acc - 1.0:
                print("  → gated_write stabilizes alpha but COSTS accuracy")
            elif g_deviation >= s_deviation:
                print("  → gated_write does NOT help alpha stability")

    print(f"{'='*80}\n")


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='A/B test: gated_write vs scatter_add — alpha collapse prevention')
    parser.add_argument('--steps', type=int, default=500, help='Training steps per condition')
    parser.add_argument('--batch', type=int, default=64, help='Batch size')
    parser.add_argument('--seq', type=int, default=64, help='Sequence length')
    parser.add_argument('--hidden-dim', type=int, default=512, help='Hidden state width')
    parser.add_argument('--M', type=int, default=256, help='Ring slots')
    parser.add_argument('--slot-dim', type=int, default=64, help='Ring slot width')
    parser.add_argument('--N', type=int, default=2, help='Expert count')
    parser.add_argument('--period', type=int, default=128, help='Pattern repeat period')
    parser.add_argument('--device', default='auto', help='cuda, cpu, or auto')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--log-every', type=int, default=50, help='Log interval')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    common = dict(
        steps=args.steps, batch=args.batch, seq=args.seq,
        hidden_dim=args.hidden_dim, M=args.M, slot_dim=args.slot_dim,
        N=args.N, period=args.period, device=device, lr=args.lr,
        log_every=args.log_every, seed=args.seed,
    )

    print("╔" + "═"*62 + "╗")
    print("║  A/B TEST: gated_write vs scatter_add — alpha collapse test  ║")
    print("╚" + "═"*62 + "╝")

    results = []

    # ── Condition A: dotprod gate + scatter_add (expected: alpha drifts) ──
    results.append(run_condition(
        label='A) dotprod + scatter_add (legacy)',
        S_mode='dotprod', gated_write=False, **common,
    ))

    # ── Condition B: dotprod gate + gated_write (expected: alpha stable) ──
    results.append(run_condition(
        label='B) dotprod + gated_write (erase+add)',
        S_mode='dotprod', gated_write=True, **common,
    ))

    # ── Condition C: fixed S=0.3 + scatter_add (champion baseline) ──
    results.append(run_condition(
        label='C) S=0.3 fixed + scatter_add (champion)',
        S_mode=0.3, gated_write=False, **common,
    ))

    # ── Comparison ──
    print_comparison(results, args.N)


if __name__ == '__main__':
    main()
