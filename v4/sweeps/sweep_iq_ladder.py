"""IQ Ladder Sweep — measure what this model can actually learn.

Uses the 5-tier difficulty ladder from datagen/generate.py to measure
the model's "IQ" as a function of parameter allocation.

Three phases:
  Phase 1: IQ Ceiling — run baseline on all tiers, find where it fails
  Phase 2: Param Allocation — fix ~200K param budget, vary H/SD/M/R
  Phase 3: Scaling — how does IQ grow with param count?

Output: IQ score per config = highest passing tier.

Usage:
    python sweep_iq_ladder.py                          # full run
    python sweep_iq_ladder.py --phase 1                # ceiling only
    python sweep_iq_ladder.py --phase 2                # allocation only
    python sweep_iq_ladder.py --steps 200 --device cpu # override
"""

import argparse
import sys
import os
import random
import time
import traceback
import gc

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'model'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'datagen'))

print("[BOOT] Python %s" % sys.version)
print("[BOOT] torch %s  CUDA available: %s" % (torch.__version__, torch.cuda.is_available()))
if torch.cuda.is_available():
    print("[BOOT] GPU: %s  VRAM: %.1f GB" % (
        torch.cuda.get_device_name(0),
        torch.cuda.get_device_properties(0).total_memory / 1e9))
print()

try:
    from instnct import INSTNCT
    print("[BOOT] instnct.py imported OK")
except Exception as e:
    print("[FATAL] instnct import failed: %s" % e)
    traceback.print_exc()
    sys.exit(1)

try:
    from generate import (
        func_echorepat_byt,
        func_invrtbits_byt,
        func_byterotat_byt,
        func_countbyte_byt,
        func_addandsub_byt,
        func_fiboseqnc_byt,
        func_delayecho_byt,
        func_denoisbyt_byt,
    )
    print("[BOOT] generate.py imported OK")
except Exception as e:
    print("[FATAL] generate.py import failed: %s" % e)
    traceback.print_exc()
    sys.exit(1)

# ── Task Ladder ──────────────────────────────────────────────────
# Ordered by tier. Each task: (name, tier, generator_fn, pass_threshold)
# Pass threshold = minimum best_acc to count as "learning" (random = 0.39%)

TASK_LADDER = [
    ('echo256',       1, func_echorepat_byt, 0.02),   # 2% to pass (random=0.39%)
    ('not256',        1, func_invrtbits_byt, 0.02),
    ('shift256',      2, func_byterotat_byt, 0.015),  # 1.5% to pass
    ('count256',      2, func_countbyte_byt, 0.015),
    ('add256',        3, func_addandsub_byt, 0.01),   # 1% to pass
    ('fib256',        4, func_fiboseqnc_byt, 0.01),   # 1% to pass
    ('delay_echo256', 5, func_delayecho_byt, 0.01),   # 1% to pass
    ('denoise256',    5, func_denoisbyt_byt, 0.01),
]

# ── Data Loading ─────────────────────────────────────────────────

def load_embed(raw_data, raw_mask, seq_len):
    """Load byte data for embed_mode=True. Returns (x, y, mask) tensors.

    raw_data: bytes from generator
    raw_mask: supervision mask bytes (0=unsupervised, 1=supervised)
    seq_len: sequence length

    x: (N, seq_len) input tokens
    y: (N, seq_len) target tokens (shifted by 1)
    mask: (N, seq_len) supervision mask for targets
    """
    arr = np.frombuffer(raw_data, dtype=np.uint8).astype(np.int64)
    marr = np.frombuffer(raw_mask, dtype=np.uint8).astype(np.float32)
    ch = seq_len + 1
    ns = len(arr) // ch
    arr = arr[:ns * ch].reshape(ns, ch)
    marr = marr[:ns * ch].reshape(ns, ch)
    x = torch.from_numpy(arr[:, :seq_len].copy())
    y = torch.from_numpy(arr[:, 1:seq_len + 1].copy())
    m = torch.from_numpy(marr[:, 1:seq_len + 1].copy())
    return x, y, m


# ── Config ───────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="IQ Ladder Sweep — measure model learning capability")
parser.add_argument("--device", default=None)
parser.add_argument("--steps", type=int, default=500, help="Training steps per config per task")
parser.add_argument("--batch", type=int, default=8, help="Batch size")
parser.add_argument("--seq", type=int, default=64, help="Sequence length")
parser.add_argument("--data-size", type=int, default=65536, help="Raw data bytes per task")
parser.add_argument("--log-every", type=int, default=100, help="Step log interval")
parser.add_argument("--phase", type=int, default=0, help="Run only phase N (0=all, 1/2/3)")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
args = parser.parse_args()

DEV = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
USE_CUDA = DEV.startswith("cuda")

# ── Baseline: corrected config (learned embed, lowrank output) ───

BASELINE = dict(
    M=256,
    hidden_dim=256,
    slot_dim=64,
    N=1,
    R=1,
    B=8,
    embed_mode=True,
    kernel_mode='vshape',
    pointer_mode='sequential',
    embed_encoding='learned',
    output_encoding='lowrank_c19',
    write_mode='replace',
    expert_weighting=False,
    bb_enabled=False,
)

print("[CONFIG] steps=%d  batch=%d  seq=%d  data=%d  device=%s  lr=%.0e" % (
    args.steps, args.batch, args.seq, args.data_size, DEV, args.lr))
print("[BASELINE] H=256 SD=64 M=256 N=1 R=1 learned+lowrank vshape sequential")
print()

# ── Sweep Engine ─────────────────────────────────────────────────

def run_config(model_kwargs, task_data, label, S_val='dotprod'):
    """Train one config on one task, return metrics dict."""
    x_all, y_all, m_all = task_data
    n_samples = x_all.shape[0]

    try:
        if USE_CUDA:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        model = INSTNCT(**model_kwargs).to(DEV)
        params = sum(p.numel() for p in model.parameters())
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        loss_fn = nn.CrossEntropyLoss(reduction='none')

        best_loss = float("inf")
        best_acc = 0.0
        best_step = 0
        t0 = time.time()

        for step in range(args.steps):
            idx = torch.randint(0, n_samples, (args.batch,))
            xb = x_all[idx].to(DEV)
            yb = y_all[idx].to(DEV)
            mb = m_all[idx].to(DEV)

            logits, _ = model(xb, S=S_val, state=None)
            per_token_loss = loss_fn(logits.reshape(-1, 256), yb.reshape(-1))
            per_token_loss = per_token_loss.reshape(args.batch, -1)

            # Masked loss: only supervised tokens contribute
            mask_sum = mb.sum()
            if mask_sum > 0:
                loss = (per_token_loss * mb).sum() / mask_sum
            else:
                # Fallback: if no supervised tokens in this batch, use all
                loss = per_token_loss.mean()

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()

            lv = loss.item()
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                if mask_sum > 0:
                    correct = ((preds == yb).float() * mb).sum()
                    acc = (correct / mask_sum).item()
                else:
                    acc = (preds == yb).float().mean().item()

            if lv < best_loss:
                best_loss = lv
                best_step = step
            if acc > best_acc:
                best_acc = acc

            if args.log_every > 0 and ((step + 1) % args.log_every == 0 or step == 0):
                print("  [%3d/%d] loss=%.4f acc=%.1f%%  best_loss=%.4f best_acc=%.1f%%" % (
                    step + 1, args.steps, lv, acc * 100, best_loss, best_acc * 100))

        elapsed = time.time() - t0
        sps = args.steps / elapsed
        peak_mb = (torch.cuda.max_memory_allocated() / 1e6) if USE_CUDA else 0

        del model, opt
        gc.collect()
        if USE_CUDA:
            torch.cuda.empty_cache()

        return dict(label=label, params=params, best_loss=best_loss,
                    best_acc=best_acc, best_step=best_step, sps=sps,
                    peak_mb=peak_mb, elapsed=elapsed, ok=True)

    except Exception as e:
        gc.collect()
        if USE_CUDA:
            torch.cuda.empty_cache()
        print("  [ERROR] %s: %s" % (label, e))
        traceback.print_exc()
        return dict(label=label, params=0, best_loss=0, best_acc=0,
                    best_step=0, sps=0, peak_mb=0, elapsed=0, ok=False, err=str(e))


def print_results(title, rows):
    print()
    print("=" * 95)
    print("  %s" % title)
    print("=" * 95)
    print("  %-35s %8s %10s %8s %8s %6s" % (
        "Config", "Params", "Best Loss", "Best Acc", "steps/s", "MB"))
    print("  %-35s %8s %10s %8s %8s %6s" % (
        "-" * 35, "-" * 8, "-" * 10, "-" * 8, "-" * 8, "-" * 6))
    for r in rows:
        if r['ok']:
            print("  %-35s %8s %10.4f %7.1f%% %8.1f %6.0f" % (
                r['label'], "{:,}".format(r['params']), r['best_loss'],
                r['best_acc'] * 100, r['sps'], r['peak_mb']))
        else:
            print("  %-35s %8s %10s %8s %8s %6s" % (
                r['label'], "ERROR", "-", "-", "-", "-"))

    ok_rows = [r for r in rows if r['ok']]
    if ok_rows:
        best_loss_r = min(ok_rows, key=lambda r: r['best_loss'])
        best_acc_r = max(ok_rows, key=lambda r: r['best_acc'])
        print()
        print("  >> Best loss: %s (%.4f)" % (best_loss_r['label'], best_loss_r['best_loss']))
        print("  >> Best acc:  %s (%.1f%%)" % (best_acc_r['label'], best_acc_r['best_acc'] * 100))


def compute_iq_score(task_results):
    """Compute IQ score = highest tier where all tasks in that tier pass.

    Returns (strict_iq, tasks_passed, total_tasks).
    strict_iq: highest tier where ALL tasks pass (ladder, no skipping)
    tasks_passed: total number of individual tasks that pass (regardless of tier order)
    """
    tier_pass = {}  # tier -> list of bools
    total_passed = 0
    total_tasks = 0
    for task_name, tier, _, threshold, result in task_results:
        if tier not in tier_pass:
            tier_pass[tier] = []
        passed = result['ok'] and result['best_acc'] >= threshold
        tier_pass[tier].append(passed)
        total_tasks += 1
        if passed:
            total_passed += 1

    strict_iq = 0
    for tier in sorted(tier_pass.keys()):
        if all(tier_pass[tier]):
            strict_iq = tier
        else:
            break
    return strict_iq, total_passed, total_tasks


def load_task_data(gen_fn, seed=42):
    """Generate and load one task's data."""
    random.seed(seed)
    raw_data, raw_mask = gen_fn(args.data_size)
    return load_embed(raw_data, raw_mask, args.seq)


# ── Phase 1: IQ Ceiling ─────────────────────────────────────────

def run_phase1():
    """Run baseline on all tasks, determine IQ ceiling."""
    print()
    print("#" * 95)
    print("#  PHASE 1: IQ CEILING — Baseline on all tiers")
    print("#" * 95)

    task_results = []

    for task_name, tier, gen_fn, threshold in TASK_LADDER:
        print()
        print("-" * 60)
        print("  Tier %d: %s  (pass threshold: %.1f%%)" % (tier, task_name, threshold * 100))
        print("-" * 60)

        task_data = load_task_data(gen_fn)
        print("  [DATA] %d samples, seq=%d" % (task_data[0].shape[0], args.seq))

        result = run_config(BASELINE, task_data, "baseline", S_val='dotprod')

        passed = result['ok'] and result['best_acc'] >= threshold
        status = "PASS" if passed else "FAIL"
        marker = "+" if passed else "x"
        if result['ok']:
            print("  >> [%s] %s  acc=%.1f%%  threshold=%.1f%%  [%s]" % (
                marker, task_name, result['best_acc'] * 100, threshold * 100, status))
        else:
            print("  >> [x] %s  ERROR  [FAIL]" % task_name)

        task_results.append((task_name, tier, gen_fn, threshold, result))

    # Compute IQ
    iq, passed, total = compute_iq_score(task_results)

    print()
    print("=" * 95)
    print("  PHASE 1 SUMMARY — IQ Ceiling")
    print("=" * 95)
    for task_name, tier, _, threshold, result in task_results:
        if result['ok']:
            p = result['best_acc'] >= threshold
            marker = "PASS" if p else "FAIL"
            print("  Tier %d  %-20s  acc=%6.1f%%  threshold=%5.1f%%  [%s]" % (
                tier, task_name, result['best_acc'] * 100, threshold * 100, marker))
        else:
            print("  Tier %d  %-20s  ERROR" % (tier, task_name))

    print()
    print("  *** IQ CEILING: Tier %d  (tasks passed: %d/%d) ***" % (iq, passed, total))
    print("  (Baseline: H=%d SD=%d M=%d N=%d R=%d, %d params)" % (
        BASELINE['hidden_dim'], BASELINE['slot_dim'], BASELINE['M'],
        BASELINE['N'], BASELINE['R'],
        task_results[0][4]['params'] if task_results[0][4]['ok'] else 0))

    return task_results, iq


# ── Phase 2: Param Allocation Sweep ──────────────────────────────

# Configs with ~similar param budgets but different allocation strategies.
# M doesn't affect param count (ring is state), so we vary it for "free".
ALLOC_CONFIGS = [
    # (name,           H,    SD,   M,    R)
    ("wide_brain",     512,  32,   128,  1),   # big brain, thin slots
    ("deep_memory",    256,  64,   512,  1),   # medium brain, lots of slots
    ("rich_slots",     256,  128,  128,  1),   # medium brain, thick slots
    ("balanced",       256,  64,   256,  1),   # baseline-like
    ("wide+attn",      512,  32,   128,  2),   # big brain + wider window
    ("memory_monster",  128,  64,  1024,  1),   # small brain, huge memory
    ("wide+memory",    512,  32,   512,  1),   # big brain + lots of slots
    ("tiny_wide",      512,  16,   256,  1),   # big brain, minimal slot width
]


def run_phase2(phase1_results, iq_ceiling):
    """Sweep param allocation on the ceiling tier tasks."""
    print()
    print("#" * 95)
    print("#  PHASE 2: PARAM ALLOCATION — Fixed budget, different allocation")
    print("#" * 95)

    # Find the frontier tasks: tier = ceiling + 1 (where baseline fails)
    # Also include ceiling tier (where baseline passes, to confirm)
    frontier_tier = iq_ceiling + 1
    test_tasks = [(n, t, g, th) for n, t, g, th in TASK_LADDER
                  if t == frontier_tier or t == iq_ceiling]

    if not test_tasks:
        # If baseline passed all tiers, test on the hardest available
        test_tasks = [(n, t, g, th) for n, t, g, th in TASK_LADDER if t == 5]

    print("  Testing on frontier tier %d (+ ceiling tier %d)" % (frontier_tier, iq_ceiling))
    print("  Tasks: %s" % ", ".join(n for n, _, _, _ in test_tasks))
    print()

    all_alloc_results = {}  # config_name -> [(task_name, tier, threshold, result)]

    for task_name, tier, gen_fn, threshold in test_tasks:
        print()
        print("-" * 60)
        print("  Tier %d: %s" % (tier, task_name))
        print("-" * 60)

        task_data = load_task_data(gen_fn)
        rows = []

        for cfg_name, h, sd, m, r in ALLOC_CONFIGS:
            kw = {**BASELINE, 'hidden_dim': h, 'slot_dim': sd, 'M': m, 'R': r}
            label = "%s (H%d SD%d M%d R%d)" % (cfg_name, h, sd, m, r)
            result = run_config(kw, task_data, label, S_val='dotprod')
            rows.append(result)

            if cfg_name not in all_alloc_results:
                all_alloc_results[cfg_name] = []
            all_alloc_results[cfg_name].append((task_name, tier, threshold, result))

        print_results("Tier %d: %s — Param Allocation" % (tier, task_name), rows)

    # Compute IQ per allocation config
    print()
    print("=" * 95)
    print("  PHASE 2 SUMMARY — Param Allocation IQ Scores")
    print("=" * 95)
    print("  %-25s %8s  %s" % ("Config", "Params", "  ".join(
        "%-12s" % n for n, _, _, _ in test_tasks)))
    print("  %-25s %8s  %s" % ("-" * 25, "-" * 8, "  ".join("-" * 12 for _ in test_tasks)))

    for cfg_name, h, sd, m, r in ALLOC_CONFIGS:
        if cfg_name in all_alloc_results:
            results = all_alloc_results[cfg_name]
            params_str = "{:,}".format(results[0][3]['params']) if results[0][3]['ok'] else "?"
            accs = []
            for task_name, tier, threshold, result in results:
                if result['ok']:
                    passed = result['best_acc'] >= threshold
                    accs.append("%5.1f%% %s" % (result['best_acc'] * 100,
                                                  "PASS" if passed else "FAIL"))
                else:
                    accs.append("ERROR       ")
            print("  %-25s %8s  %s" % (
                "%s H%d SD%d M%d" % (cfg_name, h, sd, m),
                params_str,
                "  ".join(accs)))

    return all_alloc_results


# ── Phase 3: Scaling Test ────────────────────────────────────────

def run_phase3(phase1_results, iq_ceiling, alloc_results):
    """Scale the best allocation up/down, measure IQ response."""
    print()
    print("#" * 95)
    print("#  PHASE 3: SCALING — Does more params = more IQ?")
    print("#" * 95)

    # Find best config from Phase 2 by average accuracy
    if not alloc_results:
        print("  [SKIP] No Phase 2 results to scale from.")
        return

    best_cfg = None
    best_avg_acc = -1
    for cfg_name, h, sd, m, r in ALLOC_CONFIGS:
        if cfg_name in alloc_results:
            results = alloc_results[cfg_name]
            avg_acc = np.mean([res[3]['best_acc'] for _, _, _, res in results if res['ok']])
            if avg_acc > best_avg_acc:
                best_avg_acc = avg_acc
                best_cfg = (cfg_name, h, sd, m, r)

    if best_cfg is None:
        print("  [SKIP] No valid configs to scale.")
        return

    cfg_name, base_h, base_sd, base_m, base_r = best_cfg
    print("  Best allocation from Phase 2: %s (H=%d SD=%d M=%d R=%d)" % (
        cfg_name, base_h, base_sd, base_m, base_r))
    print("  Scaling H and SD proportionally, M stays.\n")

    # Scale configs: multiply H and SD by factor
    SCALE_FACTORS = [
        ("0.5x", 0.5),
        ("1x (base)", 1.0),
        ("2x", 2.0),
        ("4x", 4.0),
    ]

    # Test on frontier tier
    frontier_tier = iq_ceiling + 1
    test_tasks = [(n, t, g, th) for n, t, g, th in TASK_LADDER
                  if t == frontier_tier]
    if not test_tasks:
        test_tasks = [(n, t, g, th) for n, t, g, th in TASK_LADDER if t == 5]

    for task_name, tier, gen_fn, threshold in test_tasks:
        print()
        print("-" * 60)
        print("  Tier %d: %s — Scaling test" % (tier, task_name))
        print("-" * 60)

        task_data = load_task_data(gen_fn)
        rows = []

        for scale_name, factor in SCALE_FACTORS:
            h = max(32, int(base_h * factor))
            sd = max(16, int(base_sd * factor))
            # Round to nearest power of 2 for cleanliness
            h = 2 ** round(np.log2(h))
            sd = 2 ** round(np.log2(sd))

            kw = {**BASELINE, 'hidden_dim': h, 'slot_dim': sd, 'M': base_m, 'R': base_r}
            label = "%s: H=%d SD=%d" % (scale_name, h, sd)
            result = run_config(kw, task_data, label, S_val='dotprod')
            rows.append(result)

        print_results("Scaling %s on %s" % (cfg_name, task_name), rows)


# ── Main ─────────────────────────────────────────────────────────

def main():
    t_start = time.time()

    phase1_results = None
    iq_ceiling = 0
    alloc_results = {}

    if args.phase == 0 or args.phase == 1:
        phase1_results, iq_ceiling = run_phase1()

    if args.phase == 0 or args.phase == 2:
        if phase1_results is None:
            print("  [INFO] Running quick Phase 1 to determine ceiling...")
            phase1_results, iq_ceiling = run_phase1()
        alloc_results = run_phase2(phase1_results, iq_ceiling)

    if args.phase == 0 or args.phase == 3:
        if phase1_results is None:
            phase1_results, iq_ceiling = run_phase1()
        if not alloc_results:
            alloc_results = run_phase2(phase1_results, iq_ceiling)
        run_phase3(phase1_results, iq_ceiling, alloc_results)

    total = time.time() - t_start
    print()
    print("=" * 95)
    print("  IQ LADDER SWEEP COMPLETE — total time: %.0fs (%.1f min)" % (total, total / 60))
    print("=" * 95)


if __name__ == '__main__':
    main()
