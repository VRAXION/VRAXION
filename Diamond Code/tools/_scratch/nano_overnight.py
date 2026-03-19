"""
Nano Overnight Lab — CPU training experiments for finding useful Nano applications.

Runs multiple small SwarmByteRingModel configs on CPU (no GPU needed).
Trains on fineweb_edu.traindat (English text, binary bits encoding).
Measures learning curves + compression quality (bits/byte vs gzip baseline).

Safe to run while GPU Ant training is active (CPU only, no CUDA).
"""

import sys, os, time, math, gzip, zlib, random, json
import signal

# CRITICAL: Limit CPU threads so we don't starve the GPU Ant training
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
torch.set_num_threads(2)  # limit PyTorch CPU parallelism
import torch.nn as nn
import numpy as np
from swarm_model import SwarmByteRingModel
from traindat_loader import TraindatLoader

# === CONFIGURATION ===

STEP_TIMEOUT = 60  # seconds per step max
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'traindat')
LOG_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'logs', 'nano')
os.makedirs(LOG_DIR, exist_ok=True)

# Configs to test: (name, D, depth, seq_len, batch_size, num_steps, lr)
CONFIGS = [
    # --- Round 1: Find the sweet spot ---
    # Small & fast — can it learn anything at all?
    ("nano_D618_s128",   618,  2, 128, 16, 300, 3e-4),
    # Medium — more capacity, still fast
    ("nano_D1000_s128", 1000,  2, 128, 12, 300, 3e-4),
    # Bigger context window
    ("nano_D1000_s256", 1000,  2, 256,  8, 300, 3e-4),
    # Full Nano v2 — best quality, slower
    ("nano_D2000_s256", 2000,  2, 256,  4, 300, 3e-4),
]

# --- Round 2 configs (run if Round 1 finishes fast) ---
ROUND2_CONFIGS = [
    # LCX OFF comparison — does memory help at small scale?
    ("nano_D1000_s128_noLCX", 1000, 2, 128, 12, 300, 3e-4),
    # Higher LR
    ("nano_D1000_s128_hiLR",  1000, 2, 128, 12, 300, 1e-3),
    # Longer seq — how far can context push accuracy?
    ("nano_D1000_s512", 1000, 2, 512, 4, 200, 3e-4),
]


def measure_gzip_bpb(corpus_bytes, sample_size=10000):
    """Measure gzip compression ratio in bits/byte on a sample."""
    sample = corpus_bytes[:sample_size]
    compressed = gzip.compress(sample, compresslevel=9)
    return len(compressed) * 8.0 / len(sample)


def measure_zstd_bpb(corpus_bytes, sample_size=10000):
    """Measure zlib (deflate) compression ratio in bits/byte on a sample."""
    sample = corpus_bytes[:sample_size]
    compressed = zlib.compress(sample, level=9)
    return len(compressed) * 8.0 / len(sample)


def bce_to_bpb(bce_loss):
    """Convert BCE loss (per bit, nats) to bits per byte."""
    # BCE loss is in nats per bit. Convert to bits per bit, then multiply by 8.
    return bce_loss / math.log(2) * 8


def train_config(name, D, depth, seq_len, batch_size, num_steps, lr,
                 use_lcx=True, corpus=None):
    """Train one Nano config and return results dict."""
    print(f"\n{'='*60}")
    print(f"  CONFIG: {name}")
    print(f"  D={D}, depth={depth}, seq_len={seq_len}, batch={batch_size}")
    print(f"  lr={lr}, steps={num_steps}, LCX={'ON' if use_lcx else 'OFF'}")
    print(f"{'='*60}\n")

    log_path = os.path.join(LOG_DIR, f"{name}.log")
    log_file = open(log_path, 'w')

    try:
        # Build model
        model = SwarmByteRingModel(
            num_bits=8,
            embedding_dim=D,
            depth=depth,
            num_beings=1,
            num_memory_positions=seq_len,
            lcx_mode='hash' if use_lcx else 'none',
            lcx_num_levels=1 if use_lcx else 0,
            lcx_level_slots=2000 if use_lcx else 0,
            lcx_key_dim=max(D // 10, 32),
            lcx_top_k=2,
            num_pointers=1,
            attention_radius=8,
        )
        model.train()
        model.to('cpu')

        n_params = sum(p.numel() for p in model.parameters())
        print(f"  params: {n_params:,}")
        print(f"  model size: {n_params * 4 / 1024 / 1024:.1f} MB (fp32)")

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

        # Training loop
        results = {
            'name': name, 'D': D, 'depth': depth, 'seq_len': seq_len,
            'batch': batch_size, 'lr': lr, 'n_params': n_params,
            'use_lcx': use_lcx, 'losses': [], 'accuracies': [],
            'times': [], 'eval_bpb': [],
        }

        best_loss = float('inf')
        loss_window = []

        for step in range(1, num_steps + 1):
            print(f"  starting step {step}...", end='', flush=True)
            t0 = time.time()

            # Generate batch from corpus
            bytes_per_pos = 1  # num_bits=8 -> 1 byte per position
            chunk_bytes = (seq_len + 1) * bytes_per_pos
            max_start = len(corpus) - chunk_bytes

            # Manual batch generation (inline, no loader overhead)
            x_list, y_list = [], []
            for _ in range(batch_size):
                start = random.randint(0, max_start)
                chunk = corpus[start:start + chunk_bytes]
                arr = np.frombuffer(chunk, dtype=np.uint8)
                bits = np.unpackbits(arr.reshape(-1, 1), axis=1).astype(np.float32)
                # bits shape: [chunk_bytes, 8]
                x_list.append(bits[:seq_len])
                y_list.append(bits[1:seq_len + 1])

            x = torch.from_numpy(np.stack(x_list))  # [B, T, 8]
            y = torch.from_numpy(np.stack(y_list))  # [B, T, 8]

            # Forward
            output = model(x)  # [B, T, 8] logits

            # Loss (BCE with logits)
            loss = nn.functional.binary_cross_entropy_with_logits(output, y)

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # Grad clip
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            dt = time.time() - t0
            loss_val = loss.item()

            # Accuracy (per-bit)
            with torch.no_grad():
                preds = (torch.sigmoid(output) > 0.5).float()
                bit_acc = (preds == y).float().mean().item()

            loss_window.append(loss_val)
            if len(loss_window) > 50:
                loss_window.pop(0)
            avg_loss = sum(loss_window) / len(loss_window)

            if loss_val < best_loss:
                best_loss = loss_val

            bpb = bce_to_bpb(loss_val)
            results['losses'].append(loss_val)
            results['accuracies'].append(bit_acc)
            results['times'].append(dt)
            results['eval_bpb'].append(bpb)

            # Log
            line = (f"step {step} | loss {loss_val:.6f} | "
                    f"acc={bit_acc:.4f} RD:{dt:.3f} traction={bit_acc:.4f} shard=0/0")
            log_file.write(line + '\n')
            log_file.flush()

            # Console (every 10 steps or first 5)
            if step <= 5 or step % 10 == 0:
                print(f" loss={loss_val:.4f} acc={bit_acc:.3f} "
                      f"bpb={bpb:.2f} avg50={avg_loss:.4f} {dt:.2f}s")
            else:
                print(f" {dt:.2f}s", flush=True)

            # Timeout guard
            if dt > STEP_TIMEOUT:
                print(f"  TIMEOUT: step took {dt:.0f}s, aborting config")
                results['aborted'] = True
                break

            # Divergence guard
            if loss_val > 10.0 or math.isnan(loss_val):
                print(f"  DIVERGED: loss={loss_val}, aborting config")
                results['aborted'] = True
                break

        # Final eval: measure on held-out data
        model.eval()
        eval_losses = []
        eval_accs = []
        with torch.no_grad():
            for _ in range(20):
                start = random.randint(0, max_start)
                chunk = corpus[start:start + chunk_bytes]
                arr = np.frombuffer(chunk, dtype=np.uint8)
                bits = np.unpackbits(arr.reshape(-1, 1), axis=1).astype(np.float32)
                ex = torch.from_numpy(bits[:seq_len]).unsqueeze(0)
                ey = torch.from_numpy(bits[1:seq_len + 1]).unsqueeze(0)
                eout = model(ex)
                eloss = nn.functional.binary_cross_entropy_with_logits(eout, ey)
                eval_losses.append(eloss.item())
                epreds = (torch.sigmoid(eout) > 0.5).float()
                eval_accs.append((epreds == ey).float().mean().item())

        results['eval_loss'] = sum(eval_losses) / len(eval_losses)
        results['eval_acc'] = sum(eval_accs) / len(eval_accs)
        results['eval_bpb_final'] = bce_to_bpb(results['eval_loss'])
        results['best_loss'] = best_loss
        results['avg_step_time'] = sum(results['times']) / len(results['times'])
        results['total_time'] = sum(results['times'])

        print(f"\n  FINAL EVAL: loss={results['eval_loss']:.4f} "
              f"acc={results['eval_acc']:.3f} "
              f"bpb={results['eval_bpb_final']:.2f}")
        print(f"  best_loss={best_loss:.4f} "
              f"avg_time={results['avg_step_time']:.3f}s "
              f"total={results['total_time']:.0f}s")

        del model, optimizer
        log_file.close()
        return results

    except Exception as e:
        print(f"  ERROR: {e}")
        log_file.close()
        return {'name': name, 'error': str(e)}


def main():
    print("=" * 60)
    print("  NANO OVERNIGHT LAB")
    print(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Load corpus once
    corpus_path = os.path.join(DATA_DIR, 'fineweb_edu.traindat')
    print(f"\nLoading corpus: {corpus_path}")
    with open(corpus_path, 'rb') as f:
        corpus = f.read()
    print(f"  Corpus size: {len(corpus):,} bytes ({len(corpus)/1024/1024:.1f} MB)")

    # Gzip baseline
    gzip_bpb = measure_gzip_bpb(corpus, sample_size=50000)
    zlib_bpb = measure_zstd_bpb(corpus, sample_size=50000)
    print(f"\n  BASELINES:")
    print(f"    Raw:   8.00 bits/byte")
    print(f"    Gzip:  {gzip_bpb:.2f} bits/byte")
    print(f"    Zlib:  {zlib_bpb:.2f} bits/byte")
    print(f"    Random model: ~8.00 bits/byte (BCE ~0.693/bit)")

    # Run Round 1
    all_results = []
    print(f"\n{'='*60}")
    print(f"  ROUND 1: {len(CONFIGS)} configs")
    print(f"{'='*60}")

    for name, D, depth, seq_len, batch, steps, lr_val in CONFIGS:
        result = train_config(name, D, depth, seq_len, batch, steps, lr_val,
                             use_lcx=True, corpus=corpus)
        all_results.append(result)

    # Run Round 2 if time permits (check if total < 30 min)
    total_time_r1 = sum(r.get('total_time', 0) for r in all_results)
    print(f"\n  Round 1 total time: {total_time_r1:.0f}s ({total_time_r1/60:.1f} min)")

    if total_time_r1 < 1800:  # < 30 min
        print(f"\n{'='*60}")
        print(f"  ROUND 2: {len(ROUND2_CONFIGS)} additional configs")
        print(f"{'='*60}")

        for name, D, depth, seq_len, batch, steps, lr_val in ROUND2_CONFIGS:
            use_lcx = 'noLCX' not in name
            result = train_config(name, D, depth, seq_len, batch, steps, lr_val,
                                 use_lcx=use_lcx, corpus=corpus)
            all_results.append(result)

    # === EXTENDED TRAINING: Best config gets 2000 more steps ===
    valid_results = [r for r in all_results if 'error' not in r and not r.get('aborted')]
    if valid_results:
        best = min(valid_results, key=lambda r: r.get('eval_loss', 999))
        print(f"\n{'='*60}")
        print(f"  EXTENDED RUN: {best['name']} (best eval_loss={best['eval_loss']:.4f})")
        print(f"  Running 2000 additional steps...")
        print(f"{'='*60}")

        ext_result = train_config(
            f"{best['name']}_extended",
            best['D'], best['depth'], best['seq_len'],
            best['batch'], 2000, best['lr'],
            use_lcx=best['use_lcx'], corpus=corpus
        )
        all_results.append(ext_result)

    # === SUMMARY ===
    print(f"\n\n{'='*60}")
    print(f"  OVERNIGHT LAB SUMMARY")
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    print(f"\n  Baselines: gzip={gzip_bpb:.2f} zlib={zlib_bpb:.2f} raw=8.00 bits/byte\n")

    header = (f"{'Name':<30} {'Params':>8} {'EvalLoss':>9} {'EvalAcc':>8} "
              f"{'BPB':>6} {'vs_gzip':>8} {'s/step':>7} {'Total':>6}")
    print(header)
    print("-" * len(header))

    for r in all_results:
        if 'error' in r:
            print(f"{r['name']:<30} ERROR: {r['error']}")
            continue
        if r.get('aborted'):
            print(f"{r['name']:<30} ABORTED")
            continue

        vs_gzip = r['eval_bpb_final'] / gzip_bpb
        print(f"{r['name']:<30} {r['n_params']:>8,} {r['eval_loss']:>9.4f} "
              f"{r['eval_acc']:>8.3f} {r['eval_bpb_final']:>6.2f} "
              f"{vs_gzip:>7.2f}x {r['avg_step_time']:>7.3f} "
              f"{r['total_time']:>5.0f}s")

    # Save results JSON
    results_path = os.path.join(LOG_DIR, 'overnight_results.json')
    # Clean for JSON serialization
    save_results = []
    for r in all_results:
        sr = {k: v for k, v in r.items() if k not in ('losses', 'accuracies', 'times', 'eval_bpb')}
        # Save last 50 losses/accs for curve analysis
        if 'losses' in r:
            sr['loss_first10'] = r['losses'][:10]
            sr['loss_last10'] = r['losses'][-10:]
            sr['acc_first10'] = r['accuracies'][:10]
            sr['acc_last10'] = r['accuracies'][-10:]
        save_results.append(sr)

    with open(results_path, 'w') as f:
        json.dump({
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'baselines': {'gzip_bpb': gzip_bpb, 'zlib_bpb': zlib_bpb, 'raw_bpb': 8.0},
            'results': save_results,
        }, f, indent=2)
    print(f"\n  Results saved to: {results_path}")

    print(f"\n  Total lab time: {sum(r.get('total_time', 0) for r in all_results):.0f}s "
          f"({sum(r.get('total_time', 0) for r in all_results)/60:.1f} min)")
    print("\nDone.")


if __name__ == '__main__':
    main()
