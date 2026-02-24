"""Input width sweep — how many bits per position can INSTNCT handle?

Tests B = [8, 16, 32, 64] bits per input position on the echo task.
Same raw bytes for all configs → fair comparison. Prints a table of
final loss, best loss, and convergence speed for each B value.

Usage:
    python sweep_input_width.py
    python sweep_input_width.py --steps 500 --device cuda
    python sweep_input_width.py --steps 500 --device cuda --d 256
    python sweep_input_width.py --b-values 8,16,32,64,128
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

# ── Echo data generator (inline, no file dependency) ──────────────

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


# ── Data loading: raw bytes → B-bit binary vectors ───────────────

def load_binary_data(raw_bytes, B, seq_len):
    """Convert raw bytes to [n_samples, seq_len, B] binary tensor.

    Each byte is unpacked to 8 bits, then consecutive bits are grouped
    into B-wide vectors. seq_len positions per sample, next-token target.
    """
    bits = np.unpackbits(np.frombuffer(raw_bytes, dtype=np.uint8))
    total_bits = len(bits)

    # trim to multiple of B
    total_pos = total_bits // B
    bits = bits[:total_pos * B].reshape(total_pos, B).astype(np.float32)

    # chunk into sequences of (seq_len + 1) for x/y split
    chunk = seq_len + 1
    n_samples = total_pos // chunk
    bits = bits[:n_samples * chunk].reshape(n_samples, chunk, B)

    x = torch.from_numpy(bits[:, :seq_len].copy())
    y = torch.from_numpy(bits[:, 1:seq_len + 1].copy())
    return x, y


# ── Training loop ────────────────────────────────────────────────

def train_one_config(B, D, raw_bytes, steps, batch_size, device, log_every=10):
    """Train INSTNCT with given B and D, return metrics dict."""

    seq_len = 128

    x_all, y_all = load_binary_data(raw_bytes, B, seq_len)
    n_samples = x_all.shape[0]

    print("[MODEL] creating INSTNCT(B=%d, D=%d)..." % (B, D))
    model = INSTNCT(B=B, D=D).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    params = sum(p.numel() for p in model.parameters())
    losses = []
    best_loss = float('inf')
    best_step = 0
    t0 = time.time()

    for step in range(steps):
        idx = torch.randint(0, n_samples, (batch_size,))
        xb = x_all[idx].to(device)
        yb = y_all[idx].to(device)

        pred = model(xb)
        loss = loss_fn(pred, yb)

        opt.zero_grad()
        loss.backward()
        opt.step()

        lv = loss.item()
        losses.append(lv)
        if lv < best_loss:
            best_loss = lv
            best_step = step

        should_log = log_every > 0 and (
            (step + 1) % log_every == 0 or step == 0 or (step + 1) == steps
        )
        if should_log:
            elapsed_so_far = time.time() - t0
            print("[STEP %3d/%d] loss=%.6f  best=%.6f @%d  elapsed=%.1fs" % (
                step + 1, steps, lv, best_loss, best_step, elapsed_so_far))

    elapsed = time.time() - t0

    return {
        'B': B,
        'seq_len': seq_len,
        'params': params,
        'samples': n_samples,
        'final_loss': losses[-1],
        'best_loss': best_loss,
        'best_step': best_step,
        'elapsed': elapsed,
    }


# ── Main ─────────────────────────────────────────────────────────

def _parse_int_list(csv: str) -> list[int]:
    return [int(x.strip()) for x in csv.split(",") if x.strip()]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Input width sweep for INSTNCT v4')
    parser.add_argument('--steps', type=int, default=500, help='training steps per config')
    parser.add_argument('--batch', type=int, default=32, help='batch size')
    parser.add_argument('--data-size', type=int, default=65536, help='raw echo data bytes (default 64KB)')
    parser.add_argument('--device', default=None, help='cpu | cuda (default: cuda if available)')
    parser.add_argument('--d', type=int, default=256, help='slot dimension D (default: 256)')
    parser.add_argument('--b-values', type=_parse_int_list, default=[8, 16, 32, 64, 128],
                        help='comma-separated B values to test (default: 8,16,32,64,128)')
    parser.add_argument('--log-every', type=int, default=10, help='print every N steps (default: 10)')
    args = parser.parse_args()

    DEV = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if DEV.startswith("cuda") and not torch.cuda.is_available():
        print("[FATAL] CUDA requested but not available")
        sys.exit(2)

    print('Input Width Sweep --- INSTNCT v4')
    print('=' * 60)
    print('[CONFIG] D=%d  steps=%d  batch=%d  data=%d bytes  device=%s' % (
        args.d, args.steps, args.batch, args.data_size, DEV))
    print('[CONFIG] B values: %s' % args.b_values)
    print()

    raw = gen_echo(args.data_size)
    print('[DATA] echo data generated: %d bytes' % len(raw))
    print()

    results = []
    for B in args.b_values:
        print('=' * 60)
        print('[SWEEP] B=%d (%d byte/pos)  D=%d  ratio=D/B=%.1f' % (B, max(1, B // 8), args.d, args.d / B))
        print('=' * 60)

        try:
            if DEV.startswith("cuda"):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            r = train_one_config(B, args.d, raw, args.steps, args.batch, DEV, args.log_every)
            results.append(r)

            peak_mb = (torch.cuda.max_memory_allocated() / 1e6) if DEV.startswith("cuda") else 0
            print('[DONE] B=%d  best=%.6f @step %d  %.1fs  VRAM=%dMB' % (
                B, r['best_loss'], r['best_step'], r['elapsed'], peak_mb))
            print()

        except Exception as e:
            print('[ERROR] B=%d failed: %s' % (B, e))
            traceback.print_exc()
            results.append({
                'B': B, 'seq_len': 0, 'params': 0, 'samples': 0,
                'final_loss': float('inf'), 'best_loss': float('inf'),
                'best_step': 0, 'elapsed': 0, 'error': str(e)[:60]
            })
            if DEV.startswith("cuda"):
                torch.cuda.empty_cache()

    # summary table
    print()
    print('=' * 60)
    print('FINAL RESULTS (D=%d)' % args.d)
    print('=' * 60)

    ok = [r for r in results if 'error' not in r]
    bmin = min(r['best_loss'] for r in ok) if ok else -1

    print('%4s  %8s  %7s  %7s  %7s  %10s  %10s  %6s' % (
        'B', 'byte/pos', 'ratio', 'samples', 'params', 'best_loss', 'final_loss', 'best@'))
    print('-' * 70)
    for r in results:
        if 'error' in r:
            print('B=%d  FAILED: %s' % (r['B'], r['error']))
        else:
            tag = ' <-- BEST' if r['best_loss'] == bmin else ''
            print('%4d  %8d  %7.1f  %7d  %7d  %10.6f  %10.6f  %5d%s' % (
                r['B'], max(1, r['B'] // 8), args.d / r['B'],
                r['samples'], r['params'],
                r['best_loss'], r['final_loss'], r['best_step'], tag))
    print()
    print('[EXIT] sweep complete')
