"""
CPU Alpha Training Probe: D=618, depth=2, no LCX, fineweb_edu

Fair comparison to GPU Alpha (D=6180, depth=2, no LCX):
  - Same seq_len=192, same data (fineweb_edu.traindat, 100MB raw text)
  - Same binary bits mode: each byte -> 8 bits, MSB first
  - Same loss: BCE with logits on each bit independently
  - D=618 instead of 6180 (1/10 scale, golden ratio CPU tier)

Purpose: Establish CPU Alpha baseline.
  - How fast does a phi-scaled brain learn English on CPU?
  - What bit_acc plateau does depth=2 reach without LCX?
  - Direct comparison to GPU Alpha (D=6180) learning curve

5000 steps, overnight capable. Eval every 25 steps.
Checkpoints every 100 steps -> checkpoints/cpu_alpha/
Results JSON -> logs/nano/cpu_alpha_results.json
Dashboard log -> logs/probe/probe_live.log
"""

import sys
import os
import time
import math
import random
import json
import signal

# Safe ASCII on Windows terminals
os.environ['PYTHONIOENCODING'] = 'utf-8'

# CPU thread budget: 8 threads, safe alongside GPU training
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
torch.set_num_threads(8)
import torch.nn as nn
import numpy as np
from swarm_model import SwarmByteRingModel
from traindat_loader import generate_batch_binary_bits

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
D = 618               # embedding_dim (phi * 1000, CPU golden tier)
DEPTH = 2             # same as GPU
SEQ_LEN = 192         # SAME as GPU -- fair comparison
BATCH_SIZE = 10       # same as GPU Alpha
NUM_BITS = 8          # byte-level, binary bits
NUM_BEINGS = 1
ATTENTION_RADIUS = 8
NUM_POINTERS = 1
USE_LCX = False       # Alpha mode: no LCX
THINK_TICKS = 0       # Alpha mode: no think ticks

LR = 0.001            # initial learning rate
LR_MIN = 1e-5         # cosine floor
WARMUP_STEPS = 50     # linear warmup
GRAD_CLIP = 1.0       # simple grad clip (no AGC on CPU)
WEIGHT_DECAY = 0      # off

TOTAL_STEPS = 5000    # overnight run
EVAL_EVERY = 25       # eval frequency
CHECKPOINT_EVERY = 100
EVAL_SAMPLES = 20     # eval batch count

STEP_TIMEOUT = 30     # seconds -- abort if one step exceeds this

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(SCRIPT_DIR, '..', '..')

CORPUS_PATH = os.path.join(ROOT_DIR, 'data', 'traindat', 'fineweb_edu.traindat')
CHECKPOINT_DIR = os.path.join(ROOT_DIR, 'checkpoints', 'cpu_alpha')
LOG_DIR = os.path.join(ROOT_DIR, 'logs', 'nano')
PROBE_LOG_DIR = os.path.join(ROOT_DIR, 'logs', 'probe')
RESULTS_PATH = os.path.join(LOG_DIR, 'cpu_alpha_results.json')
DASHBOARD_LOG = os.path.join(PROBE_LOG_DIR, 'probe_live.log')

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(PROBE_LOG_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Eval function
# ---------------------------------------------------------------------------
def evaluate(model, corpus, criterion):
    """Run eval on fresh samples. Returns metrics dict."""
    model.eval()
    eval_losses = []
    eval_bit_accs = []
    eval_byte_accs = []
    eval_per_bit = [[] for _ in range(NUM_BITS)]

    with torch.no_grad():
        for i in range(EVAL_SAMPLES):
            x, y, mask = generate_batch_binary_bits(
                corpus, n_samples=1, seq_len=SEQ_LEN, num_bits=NUM_BITS,
                seed=9000 + i,  # deterministic eval seeds
            )
            out = model(x)  # [1, T, 8]
            loss = criterion(out, y)
            eval_losses.append(loss.item())

            preds = (torch.sigmoid(out) > 0.5).float()
            bit_acc = (preds == y).float().mean().item()
            eval_bit_accs.append(bit_acc)

            # Per-bit accuracy
            for b in range(NUM_BITS):
                bacc = (preds[0, :, b] == y[0, :, b]).float().mean().item()
                eval_per_bit[b].append(bacc)

            # Byte accuracy: all 8 bits must match
            all_match = (preds == y).float().prod(dim=-1)  # [1, T]
            byte_acc = all_match.mean().item()
            eval_byte_accs.append(byte_acc)

    model.train()

    avg_loss = sum(eval_losses) / len(eval_losses)
    avg_bit_acc = sum(eval_bit_accs) / len(eval_bit_accs)
    avg_byte_acc = sum(eval_byte_accs) / len(eval_byte_accs)
    per_bit = [sum(b) / len(b) for b in eval_per_bit]
    bpb = avg_loss / math.log(2) * 8  # bits per bit -> bits per byte

    return {
        'eval_loss': round(avg_loss, 6),
        'eval_bit_acc': round(avg_bit_acc, 4),
        'eval_byte_acc': round(avg_byte_acc, 4),
        'eval_bpb': round(bpb, 2),
        'per_bit': [round(p, 3) for p in per_bit],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("  CPU ALPHA TRAINING: D=618, depth=2, no LCX, fineweb_edu")
    print("=" * 70)
    print(f"  Config: D={D}, depth={DEPTH}, seq={SEQ_LEN}, batch={BATCH_SIZE}")
    print(f"          bits={NUM_BITS}, beings={NUM_BEINGS}, radius={ATTENTION_RADIUS}")
    print(f"          lr={LR}, warmup={WARMUP_STEPS}, lr_min={LR_MIN}")
    print(f"          grad_clip={GRAD_CLIP}, steps={TOTAL_STEPS}")
    print(f"          use_lcx={USE_LCX}, think_ticks={THINK_TICKS}")
    print(f"  CPU threads: {torch.get_num_threads()}")
    print(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 70)

    # Load corpus
    if not os.path.exists(CORPUS_PATH):
        print(f"  ERROR: corpus not found at {CORPUS_PATH}")
        sys.exit(1)
    with open(CORPUS_PATH, 'rb') as f:
        corpus = f.read()
    corpus_mb = len(corpus) / 1024 / 1024
    print(f"  Corpus: {corpus_mb:.1f} MB ({len(corpus):,} bytes)")

    # Build model
    model = SwarmByteRingModel(
        num_bits=NUM_BITS,
        embedding_dim=D,
        depth=DEPTH,
        num_beings=NUM_BEINGS,
        num_memory_positions=SEQ_LEN,
        use_lcx=USE_LCX,
        num_pointers=NUM_POINTERS,
        attention_radius=ATTENTION_RADIUS,
        think_ticks=THINK_TICKS,
    )
    model.train()
    model.to('cpu')

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: {n_params:,} total, {n_trainable:,} trainable")
    print(f"          ({n_params * 4 / 1024 / 1024:.1f} MB fp32, "
          f"{n_params * 2 / 1024 / 1024:.1f} MB fp16)")

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY,
    )
    criterion = nn.BCEWithLogitsLoss()

    # Gzip baseline for reference
    import gzip
    sample = corpus[:50000]
    gzip_bpb = len(gzip.compress(sample, 9)) * 8.0 / len(sample)
    print(f"  Gzip baseline: {gzip_bpb:.2f} bpb (on first 50KB)")

    # Initial eval
    print("-" * 70)
    init_eval = evaluate(model, corpus, criterion)
    pb_str = ' '.join(f"b{i}={p:.0%}" for i, p in enumerate(init_eval['per_bit']))
    print(f"  Init eval: loss={init_eval['eval_loss']:.4f} "
          f"bit_acc={init_eval['eval_bit_acc']:.3f} "
          f"byte_acc={init_eval['eval_byte_acc']:.3f}")
    print(f"  Per-bit:   {pb_str}")
    print("-" * 70)

    # Training state
    best_eval_loss = init_eval['eval_loss']
    best_eval_acc = init_eval['eval_bit_acc']
    loss_window = []
    acc_window = []
    eval_history = [{'step': 0, **init_eval}]
    train_start = time.time()
    total_train_time = 0.0  # time in forward/backward only

    # Open dashboard log
    dash_log = open(DASHBOARD_LOG, 'w')

    try:
        for step in range(1, TOTAL_STEPS + 1):
            print(f'starting step {step}...', flush=True)
            t0 = time.time()

            # --- LR schedule: cosine with linear warmup ---
            if step <= WARMUP_STEPS:
                cur_lr = LR * step / WARMUP_STEPS
            else:
                progress = (step - WARMUP_STEPS) / max(1, TOTAL_STEPS - WARMUP_STEPS)
                cur_lr = LR_MIN + 0.5 * (LR - LR_MIN) * (1.0 + math.cos(math.pi * progress))
            for pg in optimizer.param_groups:
                pg['lr'] = cur_lr

            # --- Generate batch ---
            x, y, mask = generate_batch_binary_bits(
                corpus, n_samples=BATCH_SIZE, seq_len=SEQ_LEN, num_bits=NUM_BITS,
            )

            # --- Forward ---
            out = model(x)  # [B, T, 8]
            loss = criterion(out, y)

            # --- Backward ---
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            step_time = time.time() - t0
            total_train_time += step_time

            # --- Timeout guard ---
            if step_time > STEP_TIMEOUT:
                print(f'TIMEOUT: step took {step_time:.0f}s, aborting')
                sys.exit(1)

            # --- Compute train accuracy ---
            loss_val = loss.item()
            with torch.no_grad():
                preds = (torch.sigmoid(out) > 0.5).float()
                bit_acc = (preds == y).float().mean().item()
                byte_acc = (preds == y).float().prod(dim=-1).mean().item()

            # --- Rolling windows ---
            loss_window.append(loss_val)
            acc_window.append(bit_acc)
            if len(loss_window) > 50:
                loss_window.pop(0)
                acc_window.pop(0)
            avg_loss = sum(loss_window) / len(loss_window)
            avg_acc = sum(acc_window) / len(acc_window)

            # --- Dashboard log ---
            dash_line = (f"step {step} | loss {loss_val:.6f} | "
                         f"acc={bit_acc:.4f} RD:{step_time:.4f} "
                         f"traction={bit_acc:.4f} shard=0/0")
            dash_log.write(dash_line + '\n')
            dash_log.flush()

            # --- Divergence check ---
            if math.isnan(loss_val) or math.isinf(loss_val):
                print(f'  DIVERGED: loss={loss_val} at step {step}')
                break
            if loss_val > 10.0:
                print(f'  DIVERGED: loss={loss_val:.4f} > 10.0 at step {step}')
                break

            # --- Print training progress ---
            if step <= 5 or step % EVAL_EVERY == 0 or step % 50 == 0:
                bpb = loss_val / math.log(2) * 8
                print(f'  step {step:>5} | loss={loss_val:.4f} avg50={avg_loss:.4f} '
                      f'bit_acc={bit_acc:.3f} byte={byte_acc:.3f} '
                      f'bpb={bpb:.2f} lr={cur_lr:.6f} gnorm={grad_norm:.2f} '
                      f'{step_time:.2f}s')

            # --- Eval ---
            if step % EVAL_EVERY == 0:
                em = evaluate(model, corpus, criterion)
                em['step'] = step
                em['train_loss_avg50'] = round(avg_loss, 4)
                em['train_acc_avg50'] = round(avg_acc, 4)
                em['lr'] = round(cur_lr, 7)
                em['grad_norm'] = round(float(grad_norm), 4)
                em['elapsed_s'] = round(time.time() - train_start, 1)
                eval_history.append(em)

                pb_str = ' '.join(f"b{i}={p:.0%}" for i, p in enumerate(em['per_bit']))
                content_avg = sum(em['per_bit'][3:]) / 5 if len(em['per_bit']) > 3 else 0
                print(f'    EVAL:  loss={em["eval_loss"]:.4f} '
                      f'bit_acc={em["eval_bit_acc"]:.3f} '
                      f'byte_acc={em["eval_byte_acc"]:.3f} '
                      f'bpb={em["eval_bpb"]:.2f} '
                      f'content_avg={content_avg:.1%}')
                print(f'    BITS:  {pb_str}')

                # Track best
                if em['eval_loss'] < best_eval_loss:
                    best_eval_loss = em['eval_loss']
                    print(f'    ** NEW BEST eval loss: {best_eval_loss:.4f}')
                if em['eval_bit_acc'] > best_eval_acc:
                    best_eval_acc = em['eval_bit_acc']
                    print(f'    ** NEW BEST eval bit_acc: {best_eval_acc:.3f}')

                # Plateau detection: last 5 evals
                if len(eval_history) >= 6:  # 6 because index 0 is init
                    last5 = [e['eval_bit_acc'] for e in eval_history[-5:]]
                    spread = max(last5) - min(last5)
                    trend = last5[-1] - last5[0]
                    if spread < 0.01:
                        print(f'    ** PLATEAU: last 5 evals spread={spread:.3f} '
                              f'trend={trend:+.3f}')
                    elif spread < 0.02:
                        print(f'    ** NEAR PLATEAU: last 5 evals spread={spread:.3f} '
                              f'trend={trend:+.3f}')

            # --- Checkpoint ---
            if step % CHECKPOINT_EVERY == 0:
                ckpt_path = os.path.join(CHECKPOINT_DIR, f'cpu_alpha_step{step}.pt')
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss_val,
                    'bit_acc': bit_acc,
                    'lr': cur_lr,
                    'config': {
                        'D': D, 'depth': DEPTH, 'seq_len': SEQ_LEN,
                        'batch_size': BATCH_SIZE, 'num_bits': NUM_BITS,
                        'num_beings': NUM_BEINGS, 'attention_radius': ATTENTION_RADIUS,
                        'use_lcx': USE_LCX, 'think_ticks': THINK_TICKS,
                    },
                }, ckpt_path)
                print(f'    Checkpoint saved: {ckpt_path}')

    except KeyboardInterrupt:
        print(f'\n  Interrupted by user at step {step}')

    finally:
        dash_log.close()

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------
    wall_time = time.time() - train_start
    last_step = eval_history[-1]['step'] if eval_history else 0

    print(f"\n\n{'='*70}")
    print(f"  CPU ALPHA TRAINING -- FINAL RESULTS")
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")
    print(f"\n  Config:")
    print(f"    D={D}, depth={DEPTH}, seq_len={SEQ_LEN}, batch={BATCH_SIZE}")
    print(f"    bits={NUM_BITS}, beings={NUM_BEINGS}, radius={ATTENTION_RADIUS}")
    print(f"    lr={LR}, warmup={WARMUP_STEPS}, lr_min={LR_MIN}, grad_clip={GRAD_CLIP}")
    print(f"    use_lcx={USE_LCX}, think_ticks={THINK_TICKS}")
    print(f"\n  Model:")
    print(f"    Params: {n_params:,}")
    print(f"    Size: {n_params * 4 / 1024 / 1024:.1f} MB fp32")
    print(f"\n  Training:")
    print(f"    Steps completed: {last_step}")
    print(f"    Wall time: {wall_time:.0f}s ({wall_time/60:.1f} min, {wall_time/3600:.1f} hr)")
    if last_step > 0:
        print(f"    Avg step time: {total_train_time / last_step:.2f}s")
    print(f"    Gzip baseline: {gzip_bpb:.2f} bpb")

    # Eval history table
    if eval_history:
        print(f"\n  Eval History:")
        print(f"  {'step':>6} {'loss':>8} {'bit_acc':>8} {'byte':>8} "
              f"{'bpb':>6} | {'b0':>4} {'b1':>4} {'b2':>4} {'b3':>4} "
              f"{'b4':>4} {'b5':>4} {'b6':>4} {'b7':>4} | {'content':>7}")
        print(f"  {'-'*90}")
        for e in eval_history:
            pb = e['per_bit']
            content = sum(pb[3:]) / 5 if len(pb) > 3 else 0
            print(f"  {e['step']:>6} {e['eval_loss']:>8.4f} {e['eval_bit_acc']:>8.3f} "
                  f"{e['eval_byte_acc']:>8.3f} {e['eval_bpb']:>6.2f} | "
                  f"{pb[0]:>4.0%} {pb[1]:>4.0%} {pb[2]:>4.0%} {pb[3]:>4.0%} "
                  f"{pb[4]:>4.0%} {pb[5]:>4.0%} {pb[6]:>4.0%} {pb[7]:>4.0%} | "
                  f"{content:>7.1%}")

    # Phase analysis
    if len(eval_history) >= 8:
        q = len(eval_history) // 4
        first_q = eval_history[1:1+q]  # skip init
        last_q = eval_history[-q:]
        if first_q and last_q:
            first_acc = sum(e['eval_bit_acc'] for e in first_q) / len(first_q)
            last_acc = sum(e['eval_bit_acc'] for e in last_q) / len(last_q)
            first_content = sum(sum(e['per_bit'][3:])/5 for e in first_q) / len(first_q)
            last_content = sum(sum(e['per_bit'][3:])/5 for e in last_q) / len(last_q)
            print(f"\n  Phase Analysis:")
            print(f"    First quarter:  bit_acc={first_acc:.3f}  content={first_content:.1%}")
            print(f"    Last quarter:   bit_acc={last_acc:.3f}  content={last_content:.1%}")
            print(f"    Delta:          bit_acc={last_acc-first_acc:+.3f}  "
                  f"content={last_content-first_content:+.1%}")
            if last_acc - first_acc > 0.02:
                print(f"    -> STILL LEARNING")
            elif last_acc - first_acc > 0.005:
                print(f"    -> SLOWING DOWN")
            else:
                print(f"    -> PLATEAUED")

    # Best metrics
    if eval_history:
        best_e = max(eval_history, key=lambda e: e['eval_bit_acc'])
        print(f"\n  Best eval:")
        print(f"    step={best_e['step']} loss={best_e['eval_loss']:.4f} "
              f"bit_acc={best_e['eval_bit_acc']:.3f} byte_acc={best_e['eval_byte_acc']:.3f}")

    # GPU comparison reference
    print(f"\n  GPU Comparison Reference:")
    print(f"    GPU Alpha (D=6180, depth=2, step 305): loss=0.456 bit_acc=77.8%")
    print(f"    CPU Alpha (D=618,  depth=2, step {last_step}): "
          f"loss={eval_history[-1]['eval_loss']:.3f} "
          f"bit_acc={eval_history[-1]['eval_bit_acc']:.1%}"
          if eval_history else "    CPU Alpha: no eval data")

    # Save results JSON
    results = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'config': {
            'D': D,
            'depth': DEPTH,
            'seq_len': SEQ_LEN,
            'batch_size': BATCH_SIZE,
            'num_bits': NUM_BITS,
            'num_beings': NUM_BEINGS,
            'attention_radius': ATTENTION_RADIUS,
            'num_pointers': NUM_POINTERS,
            'use_lcx': USE_LCX,
            'think_ticks': THINK_TICKS,
            'lr': LR,
            'lr_min': LR_MIN,
            'warmup_steps': WARMUP_STEPS,
            'grad_clip': GRAD_CLIP,
            'total_steps': TOTAL_STEPS,
            'eval_every': EVAL_EVERY,
            'eval_samples': EVAL_SAMPLES,
        },
        'n_params': n_params,
        'gzip_bpb': round(gzip_bpb, 2),
        'best_eval_loss': round(best_eval_loss, 6),
        'best_eval_acc': round(best_eval_acc, 4),
        'wall_time_s': round(wall_time, 1),
        'avg_step_time': round(total_train_time / max(1, last_step), 3),
        'eval_history': eval_history,
    }
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {RESULTS_PATH}")
    print(f"  Dashboard log: {DASHBOARD_LOG}")
    print(f"  Checkpoints:   {CHECKPOINT_DIR}")
    print(f"\n  Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


if __name__ == '__main__':
    main()
