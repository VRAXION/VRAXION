"""
Nano Depth Probe — depth=4 vs depth=2 (we already have depth=2 baseline).

D=1000, seq_len=128, batch=12, LCX=ON, fineweb_edu.
Runs on CPU with limited threads (won't starve GPU Ant).
"""

import sys, os, time, math, random, json

# CRITICAL: Limit CPU threads so we don't starve the GPU Ant training
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
torch.set_num_threads(8)
import torch.nn as nn
import numpy as np
from swarm_model import SwarmByteRingModel

STEP_TIMEOUT = 120  # generous for deeper model
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'traindat')
LOG_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'logs', 'nano')
os.makedirs(LOG_DIR, exist_ok=True)

# Config: same as depth=2 baseline but depth=1
D = 1000
DEPTH = 1
SEQ_LEN = 128
BATCH = 12
LR = 3e-4
NUM_STEPS = 500  # longer run to see if depth helps past the plateau
EVAL_EVERY = 10

def main():
    print("=" * 60)
    print(f"  NANO DEPTH PROBE: depth={DEPTH}")
    print(f"  D={D}, seq_len={SEQ_LEN}, batch={BATCH}, lr={LR}")
    print(f"  CPU threads: {torch.get_num_threads()}")
    print(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Load corpus
    corpus_path = os.path.join(DATA_DIR, 'fineweb_edu.traindat')
    print(f"\nLoading corpus: {corpus_path}")
    with open(corpus_path, 'rb') as f:
        corpus = f.read()
    print(f"  Corpus: {len(corpus):,} bytes ({len(corpus)/1024/1024:.1f} MB)")

    # Build model
    model = SwarmByteRingModel(
        num_bits=8,
        embedding_dim=D,
        depth=DEPTH,
        num_beings=1,
        num_memory_positions=SEQ_LEN,
        lcx_mode='hash',
        lcx_num_levels=1,
        lcx_level_slots=2000,
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)

    # Log file
    log_path = os.path.join(LOG_DIR, f"depth_probe_d{DEPTH}.log")
    log_file = open(log_path, 'w')

    chunk_bytes = (SEQ_LEN + 1)
    max_start = len(corpus) - chunk_bytes

    best_loss = float('inf')
    loss_window = []
    all_losses = []
    all_accs = []

    print(f"\n{'Step':>5} {'Loss':>8} {'Acc':>7} {'BPB':>6} {'Avg50':>8} {'s/step':>7}")
    print("-" * 50)

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

        # Forward + loss
        output = model(x)
        loss = nn.functional.binary_cross_entropy_with_logits(output, y)

        # Backward
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

        bpb = loss_val / math.log(2) * 8

        # Log
        line = (f"step {step} | loss {loss_val:.6f} | "
                f"acc={bit_acc:.4f} RD:{dt:.3f} traction={bit_acc:.4f} shard=0/0")
        log_file.write(line + '\n')
        log_file.flush()

        if step <= 5 or step % EVAL_EVERY == 0:
            print(f" {step:>5} {loss_val:>8.4f} {bit_acc:>7.3f} {bpb:>6.2f} {avg_loss:>8.4f} {dt:>7.2f}s")
        else:
            print(f" {dt:.2f}s", flush=True)

        # Timeout guard
        if dt > STEP_TIMEOUT:
            print(f"  TIMEOUT: step took {dt:.0f}s, aborting")
            break

        # Divergence guard
        if loss_val > 10.0 or math.isnan(loss_val):
            print(f"  DIVERGED at step {step}, aborting")
            break

    # Final eval
    print(f"\n{'='*50}")
    print(f"  FINAL EVAL (20 samples)")
    model.eval()
    eval_losses = []
    eval_accs = []
    with torch.no_grad():
        for _ in range(20):
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

    eval_loss = sum(eval_losses) / len(eval_losses)
    eval_acc = sum(eval_accs) / len(eval_accs)
    eval_bpb = eval_loss / math.log(2) * 8

    print(f"  eval_loss: {eval_loss:.4f}")
    print(f"  eval_acc:  {eval_acc:.3f}")
    print(f"  eval_bpb:  {eval_bpb:.2f}")
    print(f"  best_loss: {best_loss:.4f}")

    # Compare with depth=2 baseline
    print(f"\n{'='*50}")
    print(f"  COMPARISON (depth=2 baseline from earlier run):")
    print(f"  depth=2: loss~0.49, acc~74.5%, bpb~5.6 (at step 140)")
    print(f"  depth={DEPTH}: loss={eval_loss:.4f}, acc={eval_acc:.1%}, bpb={eval_bpb:.2f} (at step {len(all_losses)})")
    if eval_loss < 0.48:
        print(f"  >>> DEPTH={DEPTH} WINS! Lower loss than depth=2 baseline")
    elif eval_loss > 0.50:
        print(f"  >>> DEPTH=2 WINS. Deeper model didn't help.")
    else:
        print(f"  >>> INCONCLUSIVE — within noise range")

    # Save results
    results = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'config': {'D': D, 'depth': DEPTH, 'seq_len': SEQ_LEN, 'batch': BATCH,
                   'lr': LR, 'n_params': n_params},
        'eval_loss': eval_loss, 'eval_acc': eval_acc, 'eval_bpb': eval_bpb,
        'best_loss': best_loss,
        'loss_first20': all_losses[:20],
        'loss_last20': all_losses[-20:],
        'acc_first20': all_accs[:20],
        'acc_last20': all_accs[-20:],
    }
    results_path = os.path.join(LOG_DIR, f"depth_probe_d{DEPTH}_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {results_path}")

    log_file.close()
    print(f"\n  Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("Done.")


if __name__ == '__main__':
    main()
