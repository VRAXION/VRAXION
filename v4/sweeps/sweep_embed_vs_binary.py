"""Binary bits vs Byte embedding — head-to-head comparison.

Tests whether nn.Embedding(256, D) + CrossEntropy beats
nn.Linear(8, D) + MSE on the echo task.

Same raw data, same model (INSTNCT v4), same training loop.
Only the input/output layer and loss function differ.

Usage:
    python sweep_embed_vs_binary.py
    python sweep_embed_vs_binary.py --device cuda --steps 500
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

# -- Echo data generator ---------------------------------------------------

BLOCK = 16
ECHO_REPEAT = 8

def gen_echo(size, seed=42):
    """Generate echo pattern: each 16-byte block repeated 8 times."""
    random.seed(seed)
    data = bytearray()
    while len(data) < size:
        block = bytes(random.randint(0, 255) for _ in range(BLOCK))
        for _ in range(ECHO_REPEAT):
            data.extend(block)
    return bytes(data[:size])

# -- Data loading: two modes -----------------------------------------------

def load_binary(raw, B, seq_len):
    """Raw bytes → binary bit vectors. Shape: [n, seq_len, B] float."""
    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))
    total_pos = len(bits) // B
    bits = bits[:total_pos * B].reshape(total_pos, B).astype(np.float32)
    chunk = seq_len + 1
    n = total_pos // chunk
    bits = bits[:n * chunk].reshape(n, chunk, B)
    x = torch.from_numpy(bits[:, :seq_len].copy())
    y = torch.from_numpy(bits[:, 1:seq_len + 1].copy())
    return x, y

def load_embed(raw, seq_len):
    """Raw bytes → byte token sequences. x: [n, seq_len] long, y: [n, seq_len] long."""
    arr = np.frombuffer(raw, dtype=np.uint8)
    chunk = seq_len + 1
    n = len(arr) // chunk
    arr = arr[:n * chunk].reshape(n, chunk)
    x = torch.from_numpy(arr[:, :seq_len].astype(np.int64).copy())
    y = torch.from_numpy(arr[:, 1:seq_len + 1].astype(np.int64).copy())
    return x, y

# -- Training ---------------------------------------------------------------

def train_config(mode, D, raw, steps, batch_size, seq_len, device, log_every):
    """Train one configuration. Returns metrics dict."""

    if mode == "binary":
        B = 8
        x_all, y_all = load_binary(raw, B, seq_len)
        model = INSTNCT(embed_dim=D, B=B, embed_mode=False).to(device)
        loss_fn = nn.MSELoss()
    else:
        x_all, y_all = load_embed(raw, seq_len)
        model = INSTNCT(embed_dim=D, embed_mode=True).to(device)
        loss_fn = nn.CrossEntropyLoss()

    n_samples = x_all.shape[0]
    params = sum(p.numel() for p in model.parameters())

    print("[DATA] %s: %d samples, seq=%d" % (mode, n_samples, seq_len))
    print("[MODEL] %s: %s params" % (mode, "{:,}".format(params)))

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    best = float("inf")
    best_step = 0
    t0 = time.time()

    for step in range(steps):
        idx = torch.randint(0, n_samples, (batch_size,))
        xb = x_all[idx].to(device)
        yb = y_all[idx].to(device)

        pred = model(xb)

        if mode == "binary":
            loss = loss_fn(pred, yb)
        else:
            # CrossEntropy needs (B*T, 256) vs (B*T,)
            loss = loss_fn(pred.reshape(-1, 256), yb.reshape(-1))

        opt.zero_grad()
        loss.backward()
        opt.step()

        lv = loss.item()
        if lv < best:
            best = lv
            best_step = step

        should_log = log_every > 0 and (
            (step + 1) % log_every == 0 or step == 0 or (step + 1) == steps
        )
        if should_log:
            elapsed_so_far = time.time() - t0
            print("[STEP %3d/%d] %s  loss=%.6f  best=%.6f @%d  elapsed=%.1fs" % (
                step + 1, steps, mode, lv, best, best_step, elapsed_so_far))

    elapsed = time.time() - t0

    # -- Accuracy probe (embedding mode: top-1 byte accuracy) --
    acc = 0.0
    if mode == "embed":
        model.eval()
        with torch.no_grad():
            probe_idx = torch.arange(min(256, n_samples))
            xp = x_all[probe_idx].to(device)
            yp = y_all[probe_idx].to(device)
            pred_p = model(xp)
            predicted_bytes = pred_p.argmax(dim=-1)  # [B, T]
            acc = (predicted_bytes == yp).float().mean().item()
        model.train()
        print("[ACC] %s: byte accuracy = %.4f (%.1f%%)" % (mode, acc, acc * 100))

    # -- Bit accuracy (binary mode) --
    if mode == "binary":
        model.eval()
        with torch.no_grad():
            probe_idx = torch.arange(min(256, n_samples))
            xp = x_all[probe_idx].to(device)
            yp = y_all[probe_idx].to(device)
            pred_p = model(xp)
            predicted_bits = (pred_p > 0.5).float()
            acc = (predicted_bits == yp).float().mean().item()
        model.train()
        print("[ACC] %s: bit accuracy = %.4f (%.1f%%)" % (mode, acc, acc * 100))

    del model, opt
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    return {
        'mode': mode,
        'params': params,
        'samples': n_samples,
        'best_loss': best,
        'best_step': best_step,
        'accuracy': acc,
        'elapsed': elapsed,
        's_per_step': elapsed / steps,
    }

# -- Main -------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Binary vs Embedding comparison")
parser.add_argument("--device", default=None)
parser.add_argument("--d", type=int, default=256, help="Slot dimension D")
parser.add_argument("--seq", type=int, default=128, help="Sequence length")
parser.add_argument("--steps", type=int, default=500, help="Training steps per config")
parser.add_argument("--batch", type=int, default=32, help="Batch size")
parser.add_argument("--data-size", type=int, default=65536, help="Raw echo data bytes")
parser.add_argument("--log-every", type=int, default=50, help="Print every N steps")
args = parser.parse_args()

DEV = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("BINARY BITS vs BYTE EMBEDDING — head to head")
print("=" * 60)
print("[CONFIG] D=%d  seq=%d  steps=%d  batch=%d  data=%d  device=%s" % (
    args.d, args.seq, args.steps, args.batch, args.data_size, DEV))
print()

raw = gen_echo(args.data_size)
print("[DATA] echo data generated: %d bytes" % len(raw))
print()

results = []

for mode in ["binary", "embed"]:
    print("=" * 60)
    print("[RUN] mode=%s" % mode)
    print("=" * 60)
    try:
        if DEV.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        r = train_config(mode, args.d, raw, args.steps, args.batch, args.seq, DEV, args.log_every)
        results.append(r)
        print()
        print("[DONE] %s  best=%.6f @%d  acc=%.4f  %.2fs/step  total=%.1fs" % (
            mode, r['best_loss'], r['best_step'], r['accuracy'], r['s_per_step'], r['elapsed']))
        print()
    except Exception as e:
        print("[ERROR] %s failed: %s" % (mode, e))
        traceback.print_exc()
        results.append({'mode': mode, 'error': str(e)[:80]})

# -- Summary ---------------------------------------------------------------

print()
print("=" * 60)
print("FINAL COMPARISON (D=%d, seq=%d, steps=%d)" % (args.d, args.seq, args.steps))
print("=" * 60)
print()
print("%-10s  %10s  %10s  %6s  %10s  %8s" % (
    "Mode", "params", "best_loss", "best@", "accuracy", "s/step"))
print("-" * 60)

for r in results:
    if 'error' in r:
        print("%-10s  FAILED: %s" % (r['mode'], r['error']))
    else:
        print("%-10s  %10s  %10.6f  %6d  %10.4f  %8.2f" % (
            r['mode'],
            "{:,}".format(r['params']),
            r['best_loss'],
            r['best_step'],
            r['accuracy'],
            r['s_per_step']))

print()

# -- Interpretation --
ok = [r for r in results if 'error' not in r]
if len(ok) == 2:
    bin_r = [r for r in ok if r['mode'] == 'binary'][0]
    emb_r = [r for r in ok if r['mode'] == 'embed'][0]
    print("NOTE: Loss scales are NOT directly comparable (MSE vs CrossEntropy).")
    print("      Compare ACCURACY instead:")
    print("        Binary bit acc:  %.1f%%" % (bin_r['accuracy'] * 100))
    print("        Embed byte acc:  %.1f%%" % (emb_r['accuracy'] * 100))
    if emb_r['accuracy'] > bin_r['accuracy']:
        pct = ((emb_r['accuracy'] / max(bin_r['accuracy'], 1e-9)) - 1) * 100
        print("      -> Embedding wins by +%.1f%% accuracy" % pct)
    elif bin_r['accuracy'] > emb_r['accuracy']:
        pct = ((bin_r['accuracy'] / max(emb_r['accuracy'], 1e-9)) - 1) * 100
        print("      -> Binary wins by +%.1f%% accuracy" % pct)
    else:
        print("      -> Tie")
    print()
    print("      Speed: binary=%.2fs/step  embed=%.2fs/step" % (
        bin_r['s_per_step'], emb_r['s_per_step']))

print()
print("[EXIT] comparison complete")
