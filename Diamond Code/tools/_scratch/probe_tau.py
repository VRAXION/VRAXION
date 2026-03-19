#!/usr/bin/env python3
"""
probe_tau.py -- LCX softmax temperature (lcx_tau) golden-section sweep
-----------------------------------------------------------------------
QUARANTINED: CPU only. GPU training runs concurrently.
No torch.cuda calls. No GPU memory. Fully isolated.

Setup (2026-02-19):
  - Sweep: lcx_tau in [0.1, 2.0] via golden-section search (~8 evals)
  - Fixed: top_k=2 (locked by #91), 1000 slots, delay_echo (gap=8)
  - Seeds: 42, 137 (winner must hold across both)
  - Steps: 1000 (same as top_k probes v3-v6)
  - Device: CPU fp32

Metrics per eval:
  - tail_median bit_acc (last 200 steps rolling median) — PRIMARY rank
  - score_margin (routing decisiveness)
  - lcx_clarity_index = (max_w - median_w) / max_w  (winner dominance)
  - s/step

Decision threshold:
  - best_tau improves >0.005 over tau=1.0 baseline → lock it, update config
  - flat (<=0.005) → leave tau=1.0, close #93

Issue: github.com/VRAXION/VRAXION/issues/93
"""

import sys
import time
import math
from pathlib import Path

DIAMOND_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(DIAMOND_ROOT))

import torch
import torch.nn.functional as F
from swarm_model import SwarmByteRingModel

# ---- QUARANTINE: CPU only ------------------------------------------------
DEVICE = torch.device('cpu')
# -------------------------------------------------------------------------

TOP_K        = 2          # locked by #91
SEEDS        = [42, 137]
STEPS        = 1000
BATCH_SIZE   = 32
SEQ_LEN      = 16
NUM_BITS     = 8
STEP_TIMEOUT = 30
LR           = 3e-4
DELAY_GAP    = 8
LCX_SLOTS    = 1000

TAIL_WINDOW       = 200
RUNG_PRIMARY      = 700
DIVERGENCE_FACTOR = 3.0

# Golden-section search bracket
TAU_LO = 0.1
TAU_HI = 2.0
GSS_TOL     = 0.05   # stop when bracket < 0.05
GSS_MAX_ITER = 12    # max evals (each = 2 seeds × 1000 steps)

# Baseline from top_k probe (k=2, 1000 slots, tau=1.0 implied)
BASELINE_TAU1 = 0.816

LOG_DIR      = DIAMOND_ROOT / 'logs' / 'probe'
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH     = LOG_DIR / 'probe_live.log'
RESULTS_PATH = LOG_DIR / 'probe_tau_results.txt'


def mini_cfg(tau):
    return dict(
        num_memory_positions=SEQ_LEN,
        embedding_dim=128,
        num_beings=1,
        depth=4,
        num_bits=NUM_BITS,
        attention_radius=3,
        think_ticks=1,
        use_lcx=True,
        lcx_mode='hash',
        lcx_num_levels=1,
        lcx_level_slots=[LCX_SLOTS],
        lcx_key_dim=32,
        lcx_top_k=TOP_K,
        lcx_tau=tau,
        num_pointers=1,
        full_view=True,
        max_hidden=128,
        min_hidden=32,
    )


def write_log(step, loss, bit_acc, step_time):
    line = (f'step {step} | loss {loss:.6f} | '
            f'acc={bit_acc:.4f} RD:{step_time:.4f} traction={bit_acc:.4f} shard=0/0\n')
    with open(LOG_PATH, 'a') as f:
        f.write(line)


def generate_delay_echo(batch_size, seq_len, num_bits, gap=DELAY_GAP):
    x = torch.randint(0, 2, (batch_size, seq_len, num_bits), dtype=torch.float32)
    y = torch.zeros_like(x)
    if gap < seq_len:
        y[:, gap:, :] = x[:, :seq_len - gap, :]
    return x, y


def rolling_median(values, window):
    if len(values) < window:
        window = len(values)
    tail = sorted(values[-window:])
    mid = len(tail) // 2
    if len(tail) % 2 == 0:
        return (tail[mid - 1] + tail[mid]) / 2.0
    return tail[mid]


def lcx_clarity(weights_list):
    """Mean (max_w - median_w) / max_w over all steps. Measures winner dominance."""
    if not weights_list:
        return None
    clarities = []
    for w in weights_list:
        # w: [B, K] tensor
        w_np = w.detach().cpu()
        for b in range(w_np.shape[0]):
            row = sorted(w_np[b].tolist(), reverse=True)
            if len(row) >= 2:
                clarities.append((row[0] - row[len(row)//2]) / (row[0] + 1e-8))
    return sum(clarities) / len(clarities) if clarities else None


def run_one(tau, seed, global_step_offset=0):
    torch.manual_seed(seed)
    model = SwarmByteRingModel(**mini_cfg(tau)).to(DEVICE)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    losses, bit_accs, step_times, score_margins, weight_snapshots = [], [], [], [], []
    had_nan = had_inf = diverged = False
    loss_at_step50 = None

    for step in range(STEPS):
        if step % 100 == 0:
            print(f'    step {step}...', flush=True)

        t0 = time.time()
        x, y = generate_delay_echo(BATCH_SIZE, SEQ_LEN, NUM_BITS)
        optimizer.zero_grad()
        output, stats = model(x, return_stats=True)
        loss = F.binary_cross_entropy_with_logits(output, y)

        for attr in ('_lcx_write_gate_aux_loss', '_lcx_read_attn_aux_loss', '_lcx_zoom_gate_aux_loss'):
            aux = getattr(model, attr, None)
            if aux is not None and hasattr(aux, 'requires_grad') and aux.requires_grad:
                loss = loss + 0.1 * aux

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        elapsed = time.time() - t0

        if elapsed > STEP_TIMEOUT:
            print(f'TIMEOUT: step {step} took {elapsed:.0f}s -- aborting')
            sys.exit(1)

        loss_val = loss.item()
        with torch.no_grad():
            bit_acc = ((output > 0).float() == y).float().mean().item()

        # Grab score margin
        sm = None
        if hasattr(model, '_lcx_score_margin_accum') and model._lcx_score_margin_accum:
            sm = model._lcx_score_margin_accum[-1][0]

        # Grab weights for clarity
        if hasattr(model, '_lcx_read_weights_accum') and model._lcx_read_weights_accum:
            weight_snapshots.append(model._lcx_read_weights_accum[-1].detach().cpu())

        if math.isnan(loss_val):
            had_nan = True; loss_val = 999.0
        if math.isinf(loss_val):
            had_inf = True; loss_val = 999.0

        losses.append(loss_val)
        bit_accs.append(bit_acc)
        step_times.append(elapsed)
        if sm is not None:
            score_margins.append(sm)

        write_log(global_step_offset + step, loss_val, bit_acc, elapsed)

        if step == 50:
            loss_at_step50 = loss_val

        if step == RUNG_PRIMARY and loss_at_step50 is not None:
            if loss_val > DIVERGENCE_FACTOR * loss_at_step50:
                diverged = True
                print(f'    DIVERGED at step {step}')
                break

    tail_median     = rolling_median(bit_accs, TAIL_WINDOW)
    avg_step_time   = sum(step_times) / len(step_times)
    avg_margin      = sum(score_margins) / len(score_margins) if score_margins else None
    clarity         = lcx_clarity(weight_snapshots)

    sm_str  = f'{avg_margin:.4f}' if avg_margin is not None else 'N/A'
    cl_str  = f'{clarity:.4f}'   if clarity    is not None else 'N/A'
    print(f'    DONE | tail={tail_median:.3f} | margin={sm_str} | '
          f'clarity={cl_str} | {avg_step_time:.2f}s/step | '
          f'nan={had_nan} div={diverged}')

    return {
        'tau': tau, 'seed': seed,
        'tail_median': tail_median,
        'avg_step_time': avg_step_time,
        'avg_margin': avg_margin,
        'clarity': clarity,
        'had_nan': had_nan, 'had_inf': had_inf, 'diverged': diverged,
    }


# ---- eval function for golden-section search ---------------------------

eval_cache = {}   # tau → mean_tail (avoid re-running same tau twice)
eval_log   = []   # ordered list of (tau, mean_tail) for final table

def evaluate(tau):
    tau_r = round(tau, 4)
    if tau_r in eval_cache:
        print(f'  [cache] tau={tau_r:.4f} → {eval_cache[tau_r]:.4f}')
        return eval_cache[tau_r]

    print(f'\n{"="*60}')
    print(f'  EVAL tau={tau_r:.4f}  (top_k={TOP_K}, slots={LCX_SLOTS})')
    print(f'{"="*60}')

    results = []
    for seed in SEEDS:
        print(f'  seed={seed}:')
        r = run_one(tau_r, seed, global_step_offset=len(eval_log) * STEPS * len(SEEDS))
        results.append(r)

    valid = [r for r in results if not r['had_nan'] and not r['diverged']]
    if len(valid) == 0:
        mean_tail = 0.0
    else:
        mean_tail = sum(r['tail_median'] for r in valid) / len(valid)

    seed_gap = abs(results[0]['tail_median'] - results[1]['tail_median']) if len(results) == 2 else 0
    avg_margin  = sum(r['avg_margin'] for r in valid if r['avg_margin'] is not None) / max(1, sum(1 for r in valid if r['avg_margin'] is not None))
    avg_clarity = sum(r['clarity']    for r in valid if r['clarity']    is not None) / max(1, sum(1 for r in valid if r['clarity']    is not None))

    print(f'  tau={tau_r:.4f} | mean_tail={mean_tail:.4f} | seed_gap={seed_gap:.4f} | '
          f'margin={avg_margin:.4f} | clarity={avg_clarity:.4f} | '
          f'vs_baseline={mean_tail - BASELINE_TAU1:+.4f}')

    eval_cache[tau_r] = mean_tail
    eval_log.append({
        'tau': tau_r, 'mean_tail': mean_tail, 'seed_gap': seed_gap,
        'avg_margin': avg_margin, 'avg_clarity': avg_clarity,
        'results': results,
    })

    # Flush results so far to disk
    with open(RESULTS_PATH, 'w') as f:
        f.write(f'probe_tau | {time.strftime("%Y-%m-%d %H:%M:%S")} | top_k={TOP_K} slots={LCX_SLOTS}\n')
        f.write(f'baseline (tau=1.0): {BASELINE_TAU1}\n\n')
        f.write(f'{"tau":>8}  {"mean_tail":>10}  {"seed_gap":>9}  {"margin":>8}  {"clarity":>9}  {"vs_base":>8}\n')
        f.write('-' * 70 + '\n')
        for e in sorted(eval_log, key=lambda x: x['tau']):
            delta = e['mean_tail'] - BASELINE_TAU1
            f.write(f'{e["tau"]:8.4f}  {e["mean_tail"]:10.4f}  {e["seed_gap"]:9.4f}  '
                    f'{e["avg_margin"]:8.4f}  {e["avg_clarity"]:9.4f}  {delta:+8.4f}\n')

    return mean_tail


# ---- golden-section search (maximize) ----------------------------------

def golden_maximize(f, a, b, tol=GSS_TOL, max_iter=GSS_MAX_ITER):
    phi = (1 + math.sqrt(5)) / 2
    r   = 2 - phi   # = 1/phi^2 ≈ 0.382
    c = a + r * (b - a)
    d = b - r * (b - a)
    fc = f(c)
    fd = f(d)
    history = [(c, fc), (d, fd)]

    for it in range(max_iter):
        print(f'\n  [GSS iter {it+1}]  bracket=[{a:.3f}, {b:.3f}]  width={b-a:.3f}')
        if (b - a) <= tol:
            break
        if fc > fd:
            b, d, fd = d, c, fc
            c = a + r * (b - a)
            fc = f(c)
            history.append((c, fc))
        else:
            a, c, fc = c, d, fd
            d = b - r * (b - a)
            fd = f(d)
            history.append((d, fd))

    best_tau = (a + b) / 2.0
    return best_tau, history


# ---- main ---------------------------------------------------------------

def main():
    print('=' * 60)
    print(f'probe_tau -- lcx_tau golden-section sweep')
    print(f'CPU ONLY -- quarantined from GPU training')
    print('=' * 60)
    print(f'  bracket:  [{TAU_LO}, {TAU_HI}]')
    print(f'  tol:      {GSS_TOL}  max_iter: {GSS_MAX_ITER}')
    print(f'  top_k:    {TOP_K}  slots: {LCX_SLOTS}  seeds: {SEEDS}')
    print(f'  steps:    {STEPS}  task: delay_echo (gap={DELAY_GAP})')
    print(f'  baseline: tau=1.0 -> {BASELINE_TAU1}')
    print(f'  results:  {RESULTS_PATH}')
    print()

    with open(LOG_PATH, 'w') as f:
        f.write(f'# probe_tau | {time.strftime("%Y-%m-%d %H:%M:%S")} | top_k={TOP_K} tau=sweep\n')

    # Run tau=1.0 first as explicit baseline point
    print('\n--- Baseline: tau=1.0 ---')
    evaluate(1.0)

    # Golden-section search
    print(f'\n--- Golden-section search: [{TAU_LO}, {TAU_HI}] ---')
    best_tau, history = golden_maximize(evaluate, TAU_LO, TAU_HI)

    # ---- Final verdict ----
    best_entry = max(eval_log, key=lambda e: e['mean_tail'])
    baseline_entry = next((e for e in eval_log if abs(e['tau'] - 1.0) < 0.001), None)
    baseline_val = baseline_entry['mean_tail'] if baseline_entry else BASELINE_TAU1
    delta = best_entry['mean_tail'] - baseline_val

    print('\n' + '=' * 60)
    print(f'FINAL RESULTS (tau sweep)')
    print('=' * 60)
    print(f'\n  {"tau":>8}  {"mean_tail":>10}  {"seed_gap":>9}  {"margin":>8}  {"clarity":>9}  {"vs_base":>8}')
    print('  ' + '-' * 66)
    for e in sorted(eval_log, key=lambda x: x['tau']):
        marker = ' ← BEST' if e['tau'] == best_entry['tau'] else (
                 ' ← baseline' if abs(e['tau'] - 1.0) < 0.001 else '')
        delta_e = e['mean_tail'] - baseline_val
        print(f'  {e["tau"]:8.4f}  {e["mean_tail"]:10.4f}  {e["seed_gap"]:9.4f}  '
              f'{e["avg_margin"]:8.4f}  {e["avg_clarity"]:9.4f}  {delta_e:+8.4f}{marker}')

    print(f'\n  Best tau: {best_entry["tau"]:.4f}  (mean_tail={best_entry["mean_tail"]:.4f})')
    print(f'  vs tau=1.0: {delta:+.4f}')

    if delta > 0.005 and best_entry['seed_gap'] < 0.010:
        verdict = f'SIGNAL: tau={best_entry["tau"]:.2f} wins by +{delta:.4f} — lock it, update config'
    elif delta > 0.005 and best_entry['seed_gap'] >= 0.010:
        verdict = f'NOISY: tau={best_entry["tau"]:.2f} wins but seed_gap={best_entry["seed_gap"]:.4f} too high — need 3rd seed'
    elif delta <= 0.005:
        verdict = f'FLAT: tau has no effect (delta={delta:+.4f}) — leave tau=1.0, close #93'
    else:
        verdict = 'BORDERLINE: weak signal — consider 3rd seed'

    print(f'\n  VERDICT: {verdict}')

    # Final write
    with open(RESULTS_PATH, 'a') as f:
        f.write(f'\nBest tau: {best_entry["tau"]:.4f}  mean_tail={best_entry["mean_tail"]:.4f}  '
                f'delta={delta:+.4f}\nVERDICT: {verdict}\n')

    print(f'\nResults: {RESULTS_PATH}')
    print('Done.')


if __name__ == '__main__':
    main()
