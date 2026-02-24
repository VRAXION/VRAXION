"""M (ring buffer size) sweep for INSTNCT v4.

Tests how ring capacity affects learning. Larger M = more memory slots
but same compute per step (attention window is R, not M).

Usage:
    python sweep_ring_size.py
    python sweep_ring_size.py --device cuda --steps 500
    python sweep_ring_size.py --m-values 32,64,128,256,512,1024
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

# -- Data ------------------------------------------------------------------

BLOCK = 16
ECHO_REPEAT = 8

def gen_echo(size, seed=42):
    random.seed(seed)
    data = bytearray()
    while len(data) < size:
        block = bytes(random.randint(0, 255) for _ in range(BLOCK))
        for _ in range(ECHO_REPEAT):
            data.extend(block)
    return bytes(data[:size])

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

# -- Parse -----------------------------------------------------------------

def _parse_int_list(csv: str) -> list[int]:
    return [int(x.strip()) for x in csv.split(",") if x.strip()]

parser = argparse.ArgumentParser(description="M (ring size) sweep for INSTNCT v4")
parser.add_argument("--device", default=None, help="cpu | cuda (default: auto)")
parser.add_argument("--d", type=int, default=256, help="Slot dimension D")
parser.add_argument("--seq", type=int, default=32, help="Sequence length")
parser.add_argument("--steps", type=int, default=300, help="Training steps per config")
parser.add_argument("--batch", type=int, default=32, help="Batch size")
parser.add_argument("--data-size", type=int, default=32768, help="Raw echo bytes")
parser.add_argument("--m-values", type=_parse_int_list, default=[32, 64, 128, 256, 512, 1024, 2048],
                    help="Comma-separated M values to test")
parser.add_argument("--log-every", type=int, default=10, help="Print every N steps")
args = parser.parse_args()

DEV = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
if DEV.startswith("cuda") and not torch.cuda.is_available():
    print("[FATAL] CUDA requested but not available")
    sys.exit(2)

print("[CONFIG] D=%d  seq=%d  steps=%d  batch=%d  data=%d  device=%s" % (
    args.d, args.seq, args.steps, args.batch, args.data_size, DEV))
print("[CONFIG] M values: %s" % args.m_values)
print()

# -- Data ------------------------------------------------------------------

print("[DATA] generating echo data...")
raw = gen_echo(args.data_size)
print("[DATA] raw bytes: %d" % len(raw))

B = 8
x_all, y_all = load(raw, B, args.seq)
n_samples = x_all.shape[0]
print("[DATA] x shape: %s  y shape: %s  samples: %d" % (x_all.shape, y_all.shape, n_samples))
print()

# -- Sweep -----------------------------------------------------------------

results = []

for M in args.m_values:
    print("=" * 60)
    print("[SWEEP] M=%d ring slots, D=%d  (ring buffer = %.1f KB)" % (
        M, args.d, M * args.d * 4 / 1024))
    print("=" * 60)

    try:
        if DEV.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        print("[MODEL] creating INSTNCT(M=%d, D=%d, N=6, B=8)..." % (M, args.d))
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
                elapsed_so_far = time.time() - t0
                print("[STEP %3d/%d] loss=%.6f  best=%.6f @%d  elapsed=%.1fs" % (
                    step + 1, args.steps, lv, best, best_step, elapsed_so_far))

        elapsed = time.time() - t0
        peak_mb = (torch.cuda.max_memory_allocated() / 1e6) if DEV.startswith("cuda") else 0
        sps = elapsed / args.steps

        print()
        print("[DONE] M=%d  best=%.6f @step %d  %.2fs/step  total=%.1fs  VRAM=%dMB" % (
            M, best, best_step, sps, elapsed, peak_mb))
        print()

        results.append((M, params, best, best_step, sps, elapsed, peak_mb, "OK"))

        del model, opt
        if DEV.startswith("cuda"):
            torch.cuda.empty_cache()

    except Exception as e:
        print("[ERROR] M=%d failed: %s" % (M, e))
        traceback.print_exc()
        results.append((M, 0, 0, 0, 0, 0, 0, str(e)[:60]))
        if DEV.startswith("cuda"):
            torch.cuda.empty_cache()

# -- Summary ---------------------------------------------------------------

print()
print("=" * 60)
print("FINAL RESULTS (D=%d, N=6, B=8)" % args.d)
print("=" * 60)

ok = [r for r in results if r[7] == "OK"]
bmin = min(r[2] for r in ok) if ok else -1

print("%6s  %10s  %10s  %10s  %8s  %8s" % (
    "M", "params", "best_loss", "best@", "s/step", "VRAM_MB"))
print("-" * 60)
for M, params, best, bs, sps, elapsed, peak, status in results:
    if status == "OK":
        tag = " <-- BEST" if best == bmin else ""
        print("%6d  %10s  %10.6f  %10d  %8.2f  %8d%s" % (
            M, "{:,}".format(params), best, bs, sps, peak, tag))
    else:
        print("M=%d  FAILED: %s" % (M, status))

print()
print("[EXIT] sweep complete")
