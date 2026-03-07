"""Minmaxed IQ Sweep — maximize intelligence, minimize compute.

Strategy: N=1 (cheap), sweep only the "high IQ / low cost" parameters
from the parameter importance matrix. Two phases:
  1. Single-axis sweeps: vary one param at a time from a strong baseline
  2. Best-combo candidates: combine the winners

Measures: loss, accuracy (byte-level), steps/sec, param count.
Tasks: echo (easy) + delay_echo (needs memory).

Usage:
    python sweep_minmax_iq.py
    python sweep_minmax_iq.py --device cuda --steps 400
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

# ── Data generators ─────────────────────────────────────────────

BLOCK = 16
DELAY_GAP = 4

def gen_echo(size, seed=42):
    random.seed(seed)
    data = bytearray()
    while len(data) < size:
        block = bytes(random.randint(0, 255) for _ in range(BLOCK))
        for _ in range(8):
            data.extend(block)
    return bytes(data[:size])

def gen_delay_echo(size, seed=42):
    random.seed(seed)
    data = bytearray()
    while len(data) < size:
        original = bytes(random.randint(0, 255) for _ in range(BLOCK))
        data.extend(original)
        for _ in range(DELAY_GAP):
            filler = bytes(random.randint(0, 255) for _ in range(BLOCK))
            data.extend(filler)
        data.extend(original)
    return bytes(data[:size])

TASKS = {'echo': gen_echo, 'delay_echo': gen_delay_echo}

def load_embed(raw, seq_len):
    """Load data for embed_mode=True (byte tokens, CrossEntropy)."""
    arr = np.frombuffer(raw, dtype=np.uint8).astype(np.int64)
    ch = seq_len + 1
    ns = len(arr) // ch
    arr = arr[:ns * ch].reshape(ns, ch)
    x = torch.from_numpy(arr[:, :seq_len].copy())
    y = torch.from_numpy(arr[:, 1:seq_len + 1].copy())
    return x, y

# ── Config ───────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Minmaxed IQ parameter sweep")
parser.add_argument("--device", default=None)
parser.add_argument("--steps", type=int, default=150, help="Training steps per config")
parser.add_argument("--batch", type=int, default=8, help="Batch size")
parser.add_argument("--seq", type=int, default=32, help="Sequence length")
parser.add_argument("--data-size", type=int, default=32768, help="Raw data bytes")
parser.add_argument("--log-every", type=int, default=100, help="Step log interval")
args = parser.parse_args()

DEV = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
USE_CUDA = DEV.startswith("cuda")

# ── Strong baseline: N=1, known-good "cheap" choices ─────────────

BASELINE = dict(
    M=128,
    hidden_dim=256,
    slot_dim=32,
    N=1,
    R=1,
    B=args.batch,
    embed_mode=True,
    kernel_mode='vshape',
    pointer_mode='sequential',
    embed_encoding='bitlift',
    output_encoding='learned',
    write_mode='replace',
    expert_weighting=False,
    bb_enabled=False,
)

print("[CONFIG] steps=%d  batch=%d  seq=%d  data=%d  device=%s" % (
    args.steps, args.batch, args.seq, args.data_size, DEV))
print("[BASELINE] N=1, hidden=256, slot=32, M=128, R=1")
print("[BASELINE] vshape, sequential, bitlift, learned")
print()

# ── Sweep engine ─────────────────────────────────────────────────

def run_config(model_kwargs, task_data, label, S_val=None):
    """Train one config, return metrics dict."""
    x_all, y_all = task_data
    n_samples = x_all.shape[0]

    try:
        if USE_CUDA:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        model = INSTNCT(**model_kwargs).to(DEV)
        params = sum(p.numel() for p in model.parameters())
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()

        best_loss = float("inf")
        best_acc = 0.0
        best_step = 0
        t0 = time.time()

        for step in range(args.steps):
            idx = torch.randint(0, n_samples, (args.batch,))
            xb = x_all[idx].to(DEV)
            yb = y_all[idx].to(DEV)

            fwd_kwargs = dict(state=None)
            if S_val is not None:
                fwd_kwargs['S'] = S_val

            logits, _ = model(xb, **fwd_kwargs)
            loss = loss_fn(logits.reshape(-1, 256), yb.reshape(-1))

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()

            lv = loss.item()
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                acc = (preds == yb.to(DEV)).float().mean().item()

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
    print("=" * 90)
    print("  %s" % title)
    print("=" * 90)
    print("  %-38s %8s %10s %8s %8s %6s" % (
        "Config", "Params", "Best Loss", "Best Acc", "steps/s", "MB"))
    print("  %-38s %8s %10s %8s %8s %6s" % (
        "-" * 38, "-" * 8, "-" * 10, "-" * 8, "-" * 8, "-" * 6))
    for r in rows:
        if r['ok']:
            print("  %-38s %8s %10.4f %7.1f%% %8.1f %6.0f" % (
                r['label'], "{:,}".format(r['params']), r['best_loss'],
                r['best_acc'] * 100, r['sps'], r['peak_mb']))
        else:
            print("  %-38s %8s %10s %8s %8s %6s" % (
                r['label'], "ERROR", "-", "-", "-", "-"))

    # Mark best
    ok_rows = [r for r in rows if r['ok']]
    if ok_rows:
        best_loss_r = min(ok_rows, key=lambda r: r['best_loss'])
        best_acc_r = max(ok_rows, key=lambda r: r['best_acc'])
        print()
        print("  >> Best loss: %s (%.4f)" % (best_loss_r['label'], best_loss_r['best_loss']))
        print("  >> Best acc:  %s (%.1f%%)" % (best_acc_r['label'], best_acc_r['best_acc'] * 100))


# ── Main sweep ───────────────────────────────────────────────────

def main():
    t_start = time.time()
    all_winners = {}  # param_name -> best_value

    for task_name in ['echo', 'delay_echo']:
        print()
        print("#" * 90)
        print("# TASK: %s" % task_name)
        print("#" * 90)

        raw = TASKS[task_name](args.data_size)
        task_data = load_embed(raw, args.seq)
        print("[DATA] %s: %d bytes -> %d samples" % (
            task_name, len(raw), task_data[0].shape[0]))

        # ── Phase 1: Single-axis sweeps ──────────────────────────

        # 1. output_encoding (★★★★☆ IQ, ★☆ cost)
        rows = []
        for enc in ['learned', 'lowrank_c19']:
            kw = {**BASELINE, 'output_encoding': enc}
            rows.append(run_config(kw, task_data, "output_enc=%s" % enc))
        print_results("%s — output_encoding" % task_name, rows)

        # 2. kernel_mode (★★★☆ IQ, ★☆ cost)
        rows = []
        for km in ['vshape', 'gaussian', 'uniform']:
            kw = {**BASELINE, 'kernel_mode': km}
            rows.append(run_config(kw, task_data, "kernel=%s" % km))
        print_results("%s — kernel_mode" % task_name, rows)

        # 3. embed_encoding (★★☆ IQ, ★☆ cost)
        rows = []
        for enc in ['learned', 'bitlift']:
            kw = {**BASELINE, 'embed_encoding': enc}
            rows.append(run_config(kw, task_data, "embed_enc=%s" % enc))
        print_results("%s — embed_encoding" % task_name, rows)

        # 4. pointer_mode (★★☆ IQ, ★★☆ cost)
        rows = []
        for pm in ['sequential', 'pilot']:
            kw = {**BASELINE, 'pointer_mode': pm}
            rows.append(run_config(kw, task_data, "pointer=%s" % pm))
        print_results("%s — pointer_mode" % task_name, rows)

        # 5. S / context_scale (★★☆ IQ, ☆ cost)
        rows = []
        for s_val in [0.0, 0.05, 0.1, 0.3]:
            kw = {**BASELINE}
            rows.append(run_config(kw, task_data, "S=%.2f" % s_val, S_val=s_val))
        # Also test dotprod (learned gate)
        kw = {**BASELINE}
        rows.append(run_config(kw, task_data, "S=dotprod", S_val='dotprod'))
        print_results("%s — context_scale (S)" % task_name, rows)

        # 6. hidden_dim (★★★★★ IQ, ★★★★☆ cost) — the big IQ lever
        rows = []
        for hd in [128, 256, 512]:
            kw = {**BASELINE, 'hidden_dim': hd}
            rows.append(run_config(kw, task_data, "hidden_dim=%d" % hd))
        print_results("%s — hidden_dim (N=1 keeps it affordable)" % task_name, rows)

        # 7. R / attention radius (★★★☆ IQ, ★★☆ cost)
        rows = []
        for r_val in [0, 1, 2]:
            kw = {**BASELINE, 'R': r_val}
            rows.append(run_config(kw, task_data, "R=%d (window=%d)" % (r_val, 2 * r_val + 1)))
        print_results("%s — attention radius R" % task_name, rows)

        # ── Phase 2: Best-combo candidates ───────────────────────

        print()
        print("=" * 90)
        print("  PHASE 2: Combined candidates — %s" % task_name)
        print("=" * 90)

        candidates = [
            ("Baseline (ref)",
             {**BASELINE}, None),

            ("MinCost+MaxIQ: lowrank vshape bitlift",
             {**BASELINE, 'output_encoding': 'lowrank_c19',
              'kernel_mode': 'vshape', 'embed_encoding': 'bitlift',
              'pointer_mode': 'sequential'}, 'dotprod'),

            ("MinCost+MaxIQ+Pilot",
             {**BASELINE, 'output_encoding': 'lowrank_c19',
              'kernel_mode': 'vshape', 'embed_encoding': 'bitlift',
              'pointer_mode': 'pilot'}, 'dotprod'),

            ("BrainBoost: H512 lowrank vshape bitlift",
             {**BASELINE, 'hidden_dim': 512, 'output_encoding': 'lowrank_c19',
              'kernel_mode': 'vshape', 'embed_encoding': 'bitlift',
              'pointer_mode': 'sequential'}, 'dotprod'),

            ("BrainBoost+Pilot: H512 pilot",
             {**BASELINE, 'hidden_dim': 512, 'output_encoding': 'lowrank_c19',
              'kernel_mode': 'vshape', 'embed_encoding': 'bitlift',
              'pointer_mode': 'pilot'}, 'dotprod'),

            ("Cheapest: H128 bitlift seq S=0.05",
             {**BASELINE, 'hidden_dim': 128, 'output_encoding': 'lowrank_c19',
              'kernel_mode': 'vshape', 'embed_encoding': 'bitlift',
              'pointer_mode': 'sequential'}, 0.05),
        ]

        rows = []
        for label, kw, s_val in candidates:
            rows.append(run_config(kw, task_data, label, S_val=s_val))
        print_results("%s — COMBINED CANDIDATES (N=1)" % task_name, rows)

    total = time.time() - t_start
    print()
    print("=" * 90)
    print("  SWEEP COMPLETE — total time: %.0fs (%.1f min)" % (total, total / 60))
    print("=" * 90)


if __name__ == '__main__':
    main()
