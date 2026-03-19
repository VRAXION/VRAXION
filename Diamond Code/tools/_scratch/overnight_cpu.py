"""
Overnight CPU experiment chain.
Runs sequential experiments, logs everything, saves results.
Safe to run alongside GPU training (8 CPU threads only).

Chain:
  1. Wait for depth=2 probe to finish (if still running)
  2. Run depth=4 probe (if depth=2 showed improvement)
     OR run D=4000 probe (if depth=2 didn't help)
  3. Save all results to JSON

Started unattended — all results go to logs/nano/overnight_results.json
"""

import sys, os, time, math, random, json, subprocess

os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
torch.set_num_threads(8)
import torch.nn as nn
import numpy as np
from swarm_model import SwarmByteRingModel

STEP_TIMEOUT = 300
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'traindat')
LOG_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'logs', 'nano')
os.makedirs(LOG_DIR, exist_ok=True)

# Fixed config
SEQ_LEN = 256
BATCH = 8
LR = 3e-4
NUM_STEPS = 500
EVAL_EVERY = 10
EVAL_SAMPLES = 50
RADIUS = 16


def train_config(depth, D, corpus, label=""):
    """Train one config, return results dict."""
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  D={D}, depth={depth}, seq={SEQ_LEN}, batch={BATCH}, radius={RADIUS}")
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")

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
            for b in range(8):
                bacc = (epreds[0, :, b] == ey[0, :, b]).float().mean().item()
                eval_per_bit[b].append(bacc)
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

    return {
        'label': label,
        'depth': depth,
        'D': D,
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


def print_comparison(results):
    """Print comparison table."""
    print(f"\n\n{'='*90}")
    print(f"  OVERNIGHT CPU EXPERIMENTS — RESULTS")
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*90}\n")

    header = (f"{'label':<25} {'depth':>5} {'D':>6} {'params':>10} {'eval_acc':>8} "
              f"{'byte_acc':>8} {'loss':>7} {'bpb':>6} {'s/step':>7} "
              f"{'per-bit (3-7)':>30}")
    print(header)
    print("-" * len(header))

    for r in results:
        pb = ' '.join(f"{r['per_bit_acc'][b]:.0%}" for b in range(3, 8))
        print(f"{r['label']:<25} {r['depth']:>5} {r['D']:>6} {r['n_params']:>10,} "
              f"{r['eval_acc']:>8.3f} {r['eval_byte_acc']:>8.3f} "
              f"{r['eval_loss']:>7.4f} {r['eval_bpb']:>6.2f} {r['avg_step_time']:>7.3f} "
              f"{pb}")

    # Delta analysis
    if len(results) >= 2:
        base = results[0]
        print(f"\n  DELTAS vs {base['label']}:")
        for r in results[1:]:
            delta_acc = r['eval_acc'] - base['eval_acc']
            delta_byte = r['eval_byte_acc'] - base['eval_byte_acc']
            delta_speed = r['avg_step_time'] / base['avg_step_time']
            verdict = "BETTER" if delta_acc > 0.02 else "MARGINAL" if delta_acc > 0.005 else "NO GAIN"
            print(f"    {r['label']}: acc {delta_acc:+.1%}, byte {delta_byte:+.1%}, "
                  f"speed {delta_speed:.1f}x, {verdict}")
            # Per content bit deltas
            for b in range(3, 8):
                d = r['per_bit_acc'][b] - base['per_bit_acc'][b]
                print(f"      bit{b}: {base['per_bit_acc'][b]:.0%} -> {r['per_bit_acc'][b]:.0%} ({d:+.1%})")


def main():
    print("=" * 70)
    print("  OVERNIGHT CPU EXPERIMENT CHAIN")
    print(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  CPU threads: {torch.get_num_threads()}")
    print("=" * 70)

    # Load corpus
    corpus_path = os.path.join(DATA_DIR, 'fineweb_edu.traindat')
    with open(corpus_path, 'rb') as f:
        corpus = f.read()
    print(f"  Corpus: {len(corpus)/1024/1024:.0f} MB")

    # Check if depth probe already has results
    depth_results_path = os.path.join(LOG_DIR, 'depth_probe.json')
    depth2_acc = None
    depth1_acc = None
    if os.path.exists(depth_results_path):
        with open(depth_results_path) as f:
            existing = json.load(f)
        for r in existing.get('results', []):
            if r['depth'] == 1:
                depth1_acc = r['eval_acc']
            if r['depth'] == 2:
                depth2_acc = r['eval_acc']
        print(f"  Found existing depth probe: d1={depth1_acc}, d2={depth2_acc}")

    all_results = []

    # --- Experiment 1: Baseline depth=1 D=2000 (quick, for comparison) ---
    random.seed(42)
    r1 = train_config(1, 2000, corpus, label="baseline d=1 D=2000")
    all_results.append(r1)

    # --- Experiment 2: depth=2 D=2000 ---
    random.seed(42)
    r2 = train_config(2, 2000, corpus, label="depth=2 D=2000")
    all_results.append(r2)

    depth_helps = r2['eval_acc'] - r1['eval_acc'] > 0.01

    if depth_helps:
        print(f"\n  >> DEPTH HELPS (+{r2['eval_acc'] - r1['eval_acc']:.1%})")
        print(f"  >> Running depth=4 to see if more depth = more gain")

        # --- Experiment 3a: depth=4 D=2000 ---
        random.seed(42)
        r3 = train_config(4, 2000, corpus, label="depth=4 D=2000")
        all_results.append(r3)

        # --- Experiment 4a: depth=2 D=4000 (depth + width) ---
        random.seed(42)
        r4 = train_config(2, 4000, corpus, label="depth=2 D=4000")
        all_results.append(r4)

    else:
        print(f"\n  >> DEPTH DOESN'T HELP ({r2['eval_acc'] - r1['eval_acc']:+.1%})")
        print(f"  >> Trying D scaling instead")

        # --- Experiment 3b: D=4000 depth=1 ---
        random.seed(42)
        r3 = train_config(1, 4000, corpus, label="D=4000 d=1")
        all_results.append(r3)

        # --- Experiment 4b: D=6000 depth=1 ---
        random.seed(42)
        r4 = train_config(1, 6000, corpus, label="D=6000 d=1")
        all_results.append(r4)

    # Print final comparison
    print_comparison(all_results)

    # Save results
    results_path = os.path.join(LOG_DIR, 'overnight_results.json')
    with open(results_path, 'w') as f:
        json.dump({
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'config': {'seq_len': SEQ_LEN, 'batch': BATCH, 'radius': RADIUS,
                       'lr': LR, 'steps': NUM_STEPS},
            'depth_helps': depth_helps,
            'results': all_results,
        }, f, indent=2)
    print(f"\n  Results saved: {results_path}")
    total = sum(r['total_time'] for r in all_results)
    print(f"  Total time: {total:.0f}s ({total/60:.0f} min, {total/3600:.1f} hr)")
    print(f"\n  Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


if __name__ == '__main__':
    main()
