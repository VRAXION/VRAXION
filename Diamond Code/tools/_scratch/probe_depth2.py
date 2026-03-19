"""
Probe: Depth=1 vs Depth=2 — D=2000, seq=256, batch=8, r=16.

Tests whether processing capacity (depth) is the Nano bottleneck.
Radius sweep (PF-018) proved attention_radius irrelevant at depth=1.
If depth=2 shows significant improvement, confirms depth is the lever.

CPU only, 8 threads, safe alongside GPU training.
"""

import sys, os, time, math, random, json

os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
torch.set_num_threads(8)
import torch.nn as nn
import numpy as np
from swarm_model import SwarmByteRingModel

STEP_TIMEOUT = 300  # depth=2 may be slower
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'traindat')
LOG_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'logs', 'nano')
os.makedirs(LOG_DIR, exist_ok=True)

# Fixed config (radius=16 chosen: cheapest with same accuracy as all others)
D = 2000
SEQ_LEN = 256
BATCH = 8
LR = 3e-4
NUM_STEPS = 500
EVAL_EVERY = 10
EVAL_SAMPLES = 50
RADIUS = 16

# Sweep
DEPTHS = [1, 2]


def train_one(depth, corpus):
    """Train one config, return results dict."""
    print(f"\n{'='*60}")
    print(f"  DEPTH = {depth}")
    print(f"  D={D}, seq={SEQ_LEN}, batch={BATCH}, radius={RADIUS}")
    print(f"{'='*60}")

    model = SwarmByteRingModel(
        num_bits=8,
        embedding_dim=D,
        depth=depth,
        num_beings=1,
        num_memory_positions=SEQ_LEN,
        lcx_mode='hash',
        lcx_num_levels=1,
        lcx_level_slots=2000,
        lcx_key_dim=max(D // 10, 32),
        lcx_top_k=2,
        num_pointers=1,
        attention_radius=RADIUS,
    )
    model.train()
    model.to('cpu')

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params:,}  ({n_params * 2 / 1024 / 1024:.1f} MB fp16)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)

    log_path = os.path.join(LOG_DIR, f"depth_{depth}.log")
    log_file = open(log_path, 'w')

    chunk_bytes = SEQ_LEN + 1
    max_start = len(corpus) - chunk_bytes

    best_loss = float('inf')
    loss_window = []
    all_losses = []
    all_accs = []
    all_times = []

    for step in range(1, NUM_STEPS + 1):
        t0 = time.time()
        print(f"  starting step {step}...", end='', flush=True)

        # Generate batch
        x_list, y_list = [], []
        for _ in range(BATCH):
            start = random.randint(0, max_start)
            chunk = corpus[start:start + chunk_bytes]
            arr = np.frombuffer(chunk, dtype=np.uint8)
            bits = np.unpackbits(arr.reshape(-1, 1), axis=1).astype(np.float32)
            x_list.append(bits[:SEQ_LEN])
            y_list.append(bits[1:SEQ_LEN + 1])

        x = torch.from_numpy(np.stack(x_list))
        y = torch.from_numpy(np.stack(y_list))

        output = model(x)
        loss = nn.functional.binary_cross_entropy_with_logits(output, y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        dt = time.time() - t0
        loss_val = loss.item()

        with torch.no_grad():
            preds = (torch.sigmoid(output) > 0.5).float()
            bit_acc = (preds == y).float().mean().item()

        loss_window.append(loss_val)
        if len(loss_window) > 50:
            loss_window.pop(0)
        avg_loss = sum(loss_window) / len(loss_window)

        if loss_val < best_loss:
            best_loss = loss_val

        all_losses.append(loss_val)
        all_accs.append(bit_acc)
        all_times.append(dt)

        bpb = loss_val / math.log(2) * 8

        line = (f"step {step} | loss {loss_val:.6f} | "
                f"acc={bit_acc:.4f} RD:{dt:.3f} traction={bit_acc:.4f} shard=0/0")
        log_file.write(line + '\n')
        log_file.flush()

        if step <= 5 or step % EVAL_EVERY == 0:
            print(f" loss={loss_val:.4f} acc={bit_acc:.3f} bpb={bpb:.2f} "
                  f"avg50={avg_loss:.4f} {dt:.2f}s")
        else:
            print(f" {dt:.2f}s", flush=True)

        if dt > STEP_TIMEOUT:
            print(f"  TIMEOUT at step {step}")
            break
        if loss_val > 10.0 or math.isnan(loss_val):
            print(f"  DIVERGED at step {step}")
            break

    # Final eval
    print(f"\n  Running eval ({EVAL_SAMPLES} samples)...")
    model.eval()
    eval_losses, eval_accs = [], []
    eval_per_bit = [[] for _ in range(8)]
    eval_byte_matches = []

    with torch.no_grad():
        for _ in range(EVAL_SAMPLES):
            start = random.randint(0, max_start)
            chunk = corpus[start:start + chunk_bytes]
            arr = np.frombuffer(chunk, dtype=np.uint8)
            bits = np.unpackbits(arr.reshape(-1, 1), axis=1).astype(np.float32)
            ex = torch.from_numpy(bits[:SEQ_LEN]).unsqueeze(0)
            ey = torch.from_numpy(bits[1:SEQ_LEN + 1]).unsqueeze(0)
            eout = model(ex)
            eloss = nn.functional.binary_cross_entropy_with_logits(eout, ey)
            eval_losses.append(eloss.item())
            epreds = (torch.sigmoid(eout) > 0.5).float()
            eval_accs.append((epreds == ey).float().mean().item())
            # Per-bit accuracy
            for b in range(8):
                bacc = (epreds[0, :, b] == ey[0, :, b]).float().mean().item()
                eval_per_bit[b].append(bacc)
            # Byte accuracy
            byte_match = 0
            for t in range(SEQ_LEN):
                pred_byte = 0
                actual_byte = 0
                for b in range(8):
                    pred_byte |= int(epreds[0, t, b].item()) << (7 - b)
                    actual_byte |= int(ey[0, t, b].item()) << (7 - b)
                if pred_byte == actual_byte:
                    byte_match += 1
            eval_byte_matches.append(byte_match / SEQ_LEN)

    eval_loss = sum(eval_losses) / len(eval_losses)
    eval_acc = sum(eval_accs) / len(eval_accs)
    eval_bpb = eval_loss / math.log(2) * 8
    per_bit_avg = [sum(b) / len(b) for b in eval_per_bit]
    eval_byte_acc = sum(eval_byte_matches) / len(eval_byte_matches)

    avg_time = sum(all_times) / len(all_times)
    total_time = sum(all_times)

    print(f"  eval_loss: {eval_loss:.4f}")
    print(f"  eval_acc:  {eval_acc:.3f}")
    print(f"  eval_byte: {eval_byte_acc:.3f}")
    print(f"  eval_bpb:  {eval_bpb:.2f}")
    print(f"  best_loss: {best_loss:.4f}")
    print(f"  per_bit:   {' '.join(f'{a:.0%}' for a in per_bit_avg)}")
    print(f"  speed:     {avg_time:.3f} s/step, total {total_time:.0f}s")

    log_file.close()

    return {
        'depth': depth,
        'n_params': n_params,
        'eval_loss': eval_loss,
        'eval_acc': eval_acc,
        'eval_byte_acc': eval_byte_acc,
        'eval_bpb': eval_bpb,
        'best_train_loss': best_loss,
        'per_bit_acc': per_bit_avg,
        'avg_step_time': avg_time,
        'total_time': total_time,
        'num_steps': len(all_losses),
    }


def main():
    print("=" * 60)
    print("  DEPTH PROBE: depth=1 vs depth=2")
    print(f"  D={D}, seq={SEQ_LEN}, batch={BATCH}, radius={RADIUS}")
    print(f"  CPU threads: {torch.get_num_threads()}")
    print(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Load corpus
    corpus_path = os.path.join(DATA_DIR, 'fineweb_edu.traindat')
    with open(corpus_path, 'rb') as f:
        corpus = f.read()
    print(f"  Corpus: {len(corpus)/1024/1024:.0f} MB")

    # Gzip baseline
    import gzip
    sample = corpus[:50000]
    gzip_bpb = len(gzip.compress(sample, 9)) * 8.0 / len(sample)
    print(f"  Gzip baseline: {gzip_bpb:.2f} bpb")

    # Run sweep
    results = []
    for depth in DEPTHS:
        r = train_one(depth, corpus)
        results.append(r)

    # Summary table
    print(f"\n\n{'='*80}")
    print(f"  DEPTH PROBE — RESULTS")
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Config: D={D}, seq={SEQ_LEN}, batch={BATCH}, radius={RADIUS}, lr={LR}")
    print(f"  Baseline: gzip={gzip_bpb:.2f} bpb, random=8.00 bpb")
    print(f"{'='*80}\n")

    header = (f"{'depth':>5} {'params':>10} {'eval_loss':>9} {'eval_acc':>8} "
              f"{'byte_acc':>8} {'bpb':>6} {'vs_gzip':>7} {'s/step':>7} "
              f"{'per-bit accuracy':>40}")
    print(header)
    print("-" * len(header))

    for r in results:
        vs_gzip = r['eval_bpb'] / gzip_bpb
        pb = ' '.join(f"{a:.0%}" for a in r['per_bit_acc'])
        print(f"{r['depth']:>5} {r['n_params']:>10,} {r['eval_loss']:>9.4f} "
              f"{r['eval_acc']:>8.3f} {r['eval_byte_acc']:>8.3f} "
              f"{r['eval_bpb']:>6.2f} {vs_gzip:>6.2f}x {r['avg_step_time']:>7.3f} "
              f"{pb}")

    # Delta analysis
    if len(results) == 2:
        d1, d2 = results[0], results[1]
        print(f"\n  DEPTH IMPACT:")
        print(f"    depth=1 → depth=2:")
        print(f"      params:   {d1['n_params']:,} → {d2['n_params']:,} ({d2['n_params']/d1['n_params']:.1f}x)")
        print(f"      eval_acc: {d1['eval_acc']:.1%} → {d2['eval_acc']:.1%} ({d2['eval_acc'] - d1['eval_acc']:+.1%})")
        print(f"      byte_acc: {d1['eval_byte_acc']:.1%} → {d2['eval_byte_acc']:.1%} ({d2['eval_byte_acc'] - d1['eval_byte_acc']:+.1%})")
        print(f"      bpb:      {d1['eval_bpb']:.2f} → {d2['eval_bpb']:.2f} ({d2['eval_bpb'] - d1['eval_bpb']:+.2f})")
        print(f"      speed:    {d1['avg_step_time']:.2f} → {d2['avg_step_time']:.2f}s ({d2['avg_step_time']/d1['avg_step_time']:.1f}x)")
        print(f"\n    Per-bit delta (depth=2 - depth=1):")
        for b in range(8):
            delta = d2['per_bit_acc'][b] - d1['per_bit_acc'][b]
            bar = '#' * max(0, int(delta * 100)) if delta > 0 else '-' * max(0, int(-delta * 100))
            print(f"      bit{b}: {d1['per_bit_acc'][b]:.0%} → {d2['per_bit_acc'][b]:.0%} ({delta:+.1%}) {bar}")

        if d2['eval_acc'] - d1['eval_acc'] > 0.02:
            print(f"\n    VERDICT: Depth IS the bottleneck (+{d2['eval_acc'] - d1['eval_acc']:.1%})")
        elif d2['eval_acc'] - d1['eval_acc'] > 0.005:
            print(f"\n    VERDICT: Depth helps marginally (+{d2['eval_acc'] - d1['eval_acc']:.1%})")
        else:
            print(f"\n    VERDICT: Depth NOT the bottleneck ({d2['eval_acc'] - d1['eval_acc']:+.1%})")

    # Save
    results_path = os.path.join(LOG_DIR, 'depth_probe.json')
    with open(results_path, 'w') as f:
        json.dump({
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'config': {'D': D, 'seq_len': SEQ_LEN, 'batch': BATCH,
                       'radius': RADIUS, 'lr': LR, 'steps': NUM_STEPS},
            'baselines': {'gzip_bpb': gzip_bpb, 'raw_bpb': 8.0},
            'results': results,
        }, f, indent=2)
    print(f"\n  Results saved: {results_path}")
    print(f"  Total time: {sum(r['total_time'] for r in results):.0f}s "
          f"({sum(r['total_time'] for r in results)/60:.1f} min)")
    print("\nDone.")


if __name__ == '__main__':
    main()
