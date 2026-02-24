"""N (expert count) sweep at D=256 on CUDA.

Full verbose logging — every step, every event, no silent failures.

Usage:
    python sweep_experts.py
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

# ── Data ─────────────────────────────────────────────────────────

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

# ── Config ───────────────────────────────────────────────────────

def _parse_int_list(csv: str) -> list[int]:
    raw_items = [item.strip() for item in csv.split(",")]
    items = [item for item in raw_items if item]
    try:
        return [int(item) for item in items]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected comma-separated ints, got: {csv!r}") from exc


parser = argparse.ArgumentParser(description="N (expert count) sweep for INSTNCT v4")
parser.add_argument("--device", default=None, help="cpu | cuda | cuda:0 (default: cuda if available else cpu)")
parser.add_argument("--d", type=int, default=256, help="Slot/hidden dimension (D)")
parser.add_argument("--seq", type=int, default=32, help="Sequence length")
parser.add_argument("--steps", type=int, default=200, help="Training steps per config")
parser.add_argument("--batch", type=int, default=32, help="Batch size")
parser.add_argument("--data-size", type=int, default=32768, help="Raw echo bytes")
parser.add_argument("--n-values", type=_parse_int_list, default=[2, 4, 6, 8, 12, 16, 24], help="Comma-separated N values")
parser.add_argument("--log-every", type=int, default=1, help="Print every N steps (0 disables step logging)")
args = parser.parse_args()

DEV = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
USE_CUDA = DEV.startswith("cuda")
if USE_CUDA and not torch.cuda.is_available():
    print("[FATAL] --device=%s requested but CUDA is not available" % DEV)
    sys.exit(2)

D = args.d
SEQ = args.seq
STEPS = args.steps
BATCH = args.batch
DATA_SIZE = args.data_size
N_VALUES = args.n_values
LOG_EVERY = args.log_every

print("[CONFIG] D=%d  seq=%d  steps=%d  batch=%d  data=%d bytes  device=%s" % (D, SEQ, STEPS, BATCH, DATA_SIZE, DEV))
print()

# ── Generate data ────────────────────────────────────────────────

print("[DATA] generating echo data...")
raw = gen_echo(DATA_SIZE)
print("[DATA] raw bytes: %d" % len(raw))

print("[DATA] loading as B=8 binary vectors...")
x_all, y_all = load(raw, 8, SEQ)
n_samples = x_all.shape[0]
print("[DATA] x shape: %s  y shape: %s  samples: %d" % (x_all.shape, y_all.shape, n_samples))
print()

# ── Sweep ────────────────────────────────────────────────────────

results = []

for N in N_VALUES:
    print("=" * 60)
    print("[SWEEP] N=%d experts, D=%d" % (N, D))
    print("=" * 60)

    try:
        if USE_CUDA:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        print("[MODEL] creating INSTNCT(D=%d, N=%d, B=8)..." % (D, N))
        model = INSTNCT(D=D, N=N, B=8).to(DEV)
        params = sum(p.numel() for p in model.parameters())
        print("[MODEL] params: %s  device: %s" % ("{:,}".format(params), DEV))

        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()

        best = float("inf")
        best_step = 0
        t0 = time.time()

        for step in range(STEPS):
            idx = torch.randint(0, n_samples, (BATCH,))
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

            elapsed_so_far = time.time() - t0
            should_log = LOG_EVERY > 0 and (
                (step + 1) % LOG_EVERY == 0 or step == 0 or (step + 1) == STEPS
            )
            if should_log:
                print("[STEP %3d/%d] loss=%.6f  best=%.6f @%d  elapsed=%.1fs" % (
                    step + 1, STEPS, lv, best, best_step, elapsed_so_far))

        elapsed = time.time() - t0
        peak_mb = (torch.cuda.max_memory_allocated() / 1e6) if USE_CUDA else 0.0
        sps = elapsed / STEPS

        print()
        if USE_CUDA:
            print("[DONE] N=%d  best=%.6f @step %d  %.2fs/step  total=%.1fs  VRAM=%dMB" % (
                N, best, best_step, sps, elapsed, peak_mb))
        else:
            print("[DONE] N=%d  best=%.6f @step %d  %.2fs/step  total=%.1fs" % (
                N, best, best_step, sps, elapsed))
        print()

        results.append((N, params, best, best_step, sps, peak_mb, "OK"))

        del model, opt
        if USE_CUDA:
            torch.cuda.empty_cache()

    except Exception as e:
        print("[ERROR] N=%d failed: %s" % (N, e))
        traceback.print_exc()
        results.append((N, 0, 0, 0, 0, 0, str(e)[:40]))
        if USE_CUDA:
            torch.cuda.empty_cache()

# ── Summary ──────────────────────────────────────────────────────

print()
print("=" * 60)
print("FINAL RESULTS (D=%d)" % D)
print("=" * 60)

if results:
    ok_results = [r for r in results if r[6] == "OK"]
    if ok_results:
        bmin = min(r[2] for r in ok_results)
    else:
        bmin = -1

    for N, params, best, bs, sps, peak, status in results:
        if status == "OK":
            tag = " <-- BEST" if best == bmin else ""
            print("N=%2d  %10s params  best=%.6f @%3d  %.2fs/step  %dMB%s" % (
                N, "{:,}".format(params), best, bs, sps, peak, tag))
        else:
            print("N=%2d  FAILED: %s" % (N, status))
else:
    print("No results!")

print()
print("[EXIT] sweep complete")
