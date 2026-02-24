"""M sweep on delay_echo task — does ring size matter when memory is needed?

The echo task showed M is irrelevant (0.263-0.270 for M=32..1024).
delay_echo has a 64-byte gap between original and echo — the model MUST
use ring memory to bridge this gap. If M matters anywhere, it's here.

Usage:
    python sweep_m_delay_echo.py
    python sweep_m_delay_echo.py --device cuda
"""

import argparse
import sys
import random
import time
import traceback

import numpy as np
import torch
import torch.nn as nn

print("[BOOT] Python %s" % sys.version)
print("[BOOT] torch %s  CUDA available: %s" % (torch.__version__, torch.cuda.is_available()))
if torch.cuda.is_available():
    print("[BOOT] GPU: %s  VRAM: %.1f GB" % (
        torch.cuda.get_device_name(0),
        torch.cuda.get_device_properties(0).total_memory / 1e9
    ))
print()

try:
    from instnct import INSTNCT
    print("[BOOT] instnct.py imported OK")
except Exception as e:
    print("[FATAL] instnct import failed: %s" % e)
    traceback.print_exc()
    sys.exit(1)

# -- Data generators -------------------------------------------------------

BLOCK = 16
DELAY_GAP = 4   # 4 filler blocks = 64 byte gap
FLIP_PROB = 0.1

def gen_echo(size, seed=42):
    """Simple echo: block repeated 8x."""
    random.seed(seed)
    data = bytearray()
    while len(data) < size:
        block = bytes(random.randint(0, 255) for _ in range(BLOCK))
        for _ in range(8):
            data.extend(block)
    return bytes(data[:size])

def gen_delay_echo(size, seed=42):
    """[A][random x4][A] — 64-byte gap between original and echo."""
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

def gen_denoise(size, seed=42):
    """[noisy_A][clean_A] — denoise corrupted input."""
    random.seed(seed)
    data = bytearray()
    while len(data) < size:
        clean = bytes(random.randint(0, 255) for _ in range(BLOCK))
        noisy = bytearray(clean)
        for i in range(len(noisy)):
            for bit in range(8):
                if random.random() < FLIP_PROB:
                    noisy[i] ^= (1 << bit)
        data.extend(noisy)
        data.extend(clean)
    return bytes(data[:size])

TASKS = {
    'echo': gen_echo,
    'delay_echo': gen_delay_echo,
    'denoise': gen_denoise,
}

# -- Data loading -----------------------------------------------------------

def load(raw, B, seq_len):
    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))
    tp = len(bits) // B
    bits = bits[:tp * B].reshape(tp, B).astype(np.float32)
    ch = seq_len + 1
    ns = tp // ch
    bits = bits[:ns * ch].reshape(ns, ch, B)
    x = torch.from_numpy(bits[:, :seq_len].copy())
    y = torch.from_numpy(bits[:, 1:seq_len + 1].copy())
    return x, y

# -- Args -------------------------------------------------------------------

def _parse_int_list(csv):
    return [int(x.strip()) for x in csv.split(",") if x.strip()]

parser = argparse.ArgumentParser(description="M sweep across multiple tasks")
parser.add_argument("--device", default=None)
parser.add_argument("--d", type=int, default=256)
parser.add_argument("--seq", type=int, default=64, help="Longer seq to stress memory (default 64)")
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--batch", type=int, default=32)
parser.add_argument("--data-size", type=int, default=32768)
parser.add_argument("--m-values", type=_parse_int_list, default=[32, 64, 128, 256, 512])
parser.add_argument("--tasks", default="echo,delay_echo,denoise", help="Comma-sep task names")
parser.add_argument("--log-every", type=int, default=50)
args = parser.parse_args()

DEV = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
task_names = [t.strip() for t in args.tasks.split(",")]

print("[CONFIG] D=%d  seq=%d  steps=%d  batch=%d  data=%d  device=%s" % (
    args.d, args.seq, args.steps, args.batch, args.data_size, DEV))
print("[CONFIG] M values: %s" % args.m_values)
print("[CONFIG] Tasks: %s" % task_names)
print()

# -- Sweep ------------------------------------------------------------------

B = 8
all_results = {}  # task -> [(M, best_loss, sps, status)]

for task_name in task_names:
    gen_fn = TASKS[task_name]
    print("#" * 60)
    print("# TASK: %s" % task_name)
    print("#" * 60)

    raw = gen_fn(args.data_size)
    x_all, y_all = load(raw, B, args.seq)
    n_samples = x_all.shape[0]
    print("[DATA] %s: %d bytes -> %d samples (seq=%d)" % (task_name, len(raw), n_samples, args.seq))
    print()

    task_results = []

    for M in args.m_values:
        print("=" * 60)
        print("[SWEEP] task=%s  M=%d  D=%d" % (task_name, M, args.d))
        print("=" * 60)

        try:
            if DEV.startswith("cuda"):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            model = INSTNCT(M=M, D=args.d, N=6, B=B).to(DEV)
            params = sum(p.numel() for p in model.parameters())
            print("[MODEL] params: %s" % "{:,}".format(params))

            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            loss_fn = nn.MSELoss()

            best = float("inf")
            best_step = 0
            t0 = time.time()

            for step in range(args.steps):
                idx = torch.randint(0, n_samples, (args.batch,))
                xb = x_all[idx].to(DEV)
                yb = y_all[idx].to(DEV)

                pred = model(xb)
                loss = loss_fn(pred, yb)

                opt.zero_grad()
                loss.backward()
                opt.step()

                lv = loss.item()
                if lv < best:
                    best = lv
                    best_step = step

                should_log = args.log_every > 0 and (
                    (step + 1) % args.log_every == 0 or step == 0 or (step + 1) == args.steps
                )
                if should_log:
                    print("[STEP %3d/%d] loss=%.6f  best=%.6f @%d  elapsed=%.1fs" % (
                        step + 1, args.steps, lv, best, best_step, time.time() - t0))

            elapsed = time.time() - t0
            sps = elapsed / args.steps
            print("[DONE] task=%s M=%d  best=%.6f @%d  %.2fs/step  total=%.1fs" % (
                task_name, M, best, best_step, sps, elapsed))
            print()
            task_results.append((M, best, best_step, sps, "OK"))

            del model, opt
            if DEV.startswith("cuda"):
                torch.cuda.empty_cache()

        except Exception as e:
            print("[ERROR] M=%d failed: %s" % (M, e))
            traceback.print_exc()
            task_results.append((M, 0, 0, 0, str(e)[:60]))

    all_results[task_name] = task_results

# -- Summary ----------------------------------------------------------------

print()
print("=" * 70)
print("FINAL COMPARISON: M sweep across tasks (D=%d, N=6, B=8, seq=%d)" % (args.d, args.seq))
print("=" * 70)

for task_name in task_names:
    results = all_results[task_name]
    ok = [r for r in results if r[4] == "OK"]
    bmin = min(r[1] for r in ok) if ok else -1

    print()
    print("--- %s ---" % task_name)
    print("%6s  %10s  %6s  %8s  %s" % ("M", "best_loss", "best@", "s/step", ""))
    for M, best, bs, sps, status in results:
        if status == "OK":
            tag = " <-- BEST" if best == bmin else ""
            pct = ((best / bmin) - 1) * 100 if bmin > 0 else 0
            print("%6d  %10.6f  %6d  %8.2f  %+.1f%%%s" % (M, best, bs, sps, pct, tag))
        else:
            print("M=%d  FAILED: %s" % (M, status))

print()
print("[EXIT] multi-task M sweep complete")
