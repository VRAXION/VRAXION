"""
CPU Beta Training: Alpha brain + LCX enabled (gate-free, zero-init BN).

Loads a plateaued CPU Alpha checkpoint (D=618, no LCX, ~79% eval bit_acc)
and continues training with LCX ON + think_ticks=1.

This is the Alpha->Beta transition: brain is mature, LCX starts from zero.
Uses the new gate-free architecture (v3.5): no zoom_gate, zero-init last BN layer.

Key question: does LCX break past the 79% CPU Alpha ceiling?
"""

import sys, os, time, math, random, json, signal

os.environ['PYTHONIOENCODING'] = 'utf-8'
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
D = 618
DEPTH = 2
SEQ_LEN = 192
BATCH_SIZE = 10
NUM_BITS = 8
NUM_BEINGS = 1
ATTENTION_RADIUS = 8
NUM_POINTERS = 1

# Beta mode: LCX ON
USE_LCX = True
THINK_TICKS = 1
LCX_MODE = 'hash'
LCX_NUM_LEVELS = 1
LCX_LEVEL_SLOTS = [200]  # Scaled down from GPU's 2000 (1/10 like D)
LCX_KEY_DIM = 61          # D/10, same ratio as GPU (618/10)
LCX_TOP_K = 2

LR = 0.0004              # Lower LR for Beta (brain already trained)
LR_MIN = 1e-5
WARMUP_STEPS = 25        # Short warmup since brain is warm
GRAD_CLIP = 1.0
WEIGHT_DECAY = 0

TOTAL_STEPS = 3000
EVAL_EVERY = 25
CHECKPOINT_EVERY = 100
EVAL_SAMPLES = 20

STEP_TIMEOUT = 60  # Higher timeout for LCX overhead

# Alpha checkpoint to load brain from
ALPHA_CHECKPOINT = None  # Will find latest automatically

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(SCRIPT_DIR, '..', '..')

CORPUS_PATH = os.path.join(ROOT_DIR, 'data', 'traindat', 'fineweb_edu.traindat')
ALPHA_CKPT_DIR = os.path.join(ROOT_DIR, 'checkpoints', 'cpu_alpha')
CHECKPOINT_DIR = os.path.join(ROOT_DIR, 'checkpoints', 'cpu_beta')
LOG_DIR = os.path.join(ROOT_DIR, 'logs', 'nano')
PROBE_LOG_DIR = os.path.join(ROOT_DIR, 'logs', 'probe')
RESULTS_PATH = os.path.join(LOG_DIR, 'cpu_beta_results.json')
DASHBOARD_LOG = os.path.join(PROBE_LOG_DIR, 'probe_live.log')

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(PROBE_LOG_DIR, exist_ok=True)


def find_latest_alpha_checkpoint():
    """Find the highest-step Alpha checkpoint."""
    if not os.path.isdir(ALPHA_CKPT_DIR):
        return None
    files = [f for f in os.listdir(ALPHA_CKPT_DIR) if f.endswith('.pt')]
    if not files:
        return None
    files.sort(key=lambda f: int(f.split('step')[1].split('.')[0]))
    return os.path.join(ALPHA_CKPT_DIR, files[-1])


def evaluate(model, corpus, criterion):
    """Run eval on fresh samples."""
    model.eval()
    eval_losses = []
    eval_bit_accs = []
    eval_byte_accs = []
    eval_per_bit = [[] for _ in range(NUM_BITS)]

    with torch.no_grad():
        for i in range(EVAL_SAMPLES):
            x, y, mask = generate_batch_binary_bits(
                corpus, n_samples=1, seq_len=SEQ_LEN, num_bits=NUM_BITS,
                seed=9000 + i,
            )
            out = model(x)
            loss = criterion(out, y)
            eval_losses.append(loss.item())

            preds = (torch.sigmoid(out) > 0.5).float()
            bit_acc = (preds == y).float().mean().item()
            eval_bit_accs.append(bit_acc)

            for b in range(NUM_BITS):
                bacc = (preds[0, :, b] == y[0, :, b]).float().mean().item()
                eval_per_bit[b].append(bacc)

            all_match = (preds == y).float().prod(dim=-1)
            byte_acc = all_match.mean().item()
            eval_byte_accs.append(byte_acc)

    model.train()

    avg_loss = sum(eval_losses) / len(eval_losses)
    avg_bit_acc = sum(eval_bit_accs) / len(eval_bit_accs)
    avg_byte_acc = sum(eval_byte_accs) / len(eval_byte_accs)
    per_bit = [sum(b) / len(b) for b in eval_per_bit]
    bpb = avg_loss / math.log(2) * 8

    return {
        'eval_loss': round(avg_loss, 6),
        'eval_bit_acc': round(avg_bit_acc, 4),
        'eval_byte_acc': round(avg_byte_acc, 4),
        'eval_bpb': round(bpb, 2),
        'per_bit': [round(p, 3) for p in per_bit],
    }


def main():
    print("=" * 70)
    print("  CPU BETA TRAINING: Alpha brain + LCX (gate-free, zero-init)")
    print("=" * 70)
    print(f"  Config: D={D}, depth={DEPTH}, seq={SEQ_LEN}, batch={BATCH_SIZE}")
    print(f"          bits={NUM_BITS}, beings={NUM_BEINGS}, radius={ATTENTION_RADIUS}")
    print(f"          use_lcx={USE_LCX}, think_ticks={THINK_TICKS}")
    print(f"          lcx_slots={LCX_LEVEL_SLOTS}, key_dim={LCX_KEY_DIM}, top_k={LCX_TOP_K}")
    print(f"          lr={LR}, warmup={WARMUP_STEPS}, lr_min={LR_MIN}")
    print(f"  CPU threads: {torch.get_num_threads()}")
    print(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 70)

    # Load corpus
    with open(CORPUS_PATH, 'rb') as f:
        corpus = f.read()
    print(f"  Corpus: {len(corpus) / 1024 / 1024:.1f} MB")

    # Find Alpha checkpoint
    alpha_ckpt_path = ALPHA_CHECKPOINT or find_latest_alpha_checkpoint()
    if not alpha_ckpt_path or not os.path.exists(alpha_ckpt_path):
        print(f"  ERROR: No Alpha checkpoint found in {ALPHA_CKPT_DIR}")
        sys.exit(1)
    print(f"  Alpha checkpoint: {alpha_ckpt_path}")

    # Load Alpha checkpoint
    alpha_ckpt = torch.load(alpha_ckpt_path, map_location='cpu', weights_only=False)
    alpha_step = alpha_ckpt['step']
    alpha_acc = alpha_ckpt.get('bit_acc', 0)
    alpha_loss = alpha_ckpt.get('loss', 0)
    print(f"  Alpha step: {alpha_step}, acc: {alpha_acc:.3f}, loss: {alpha_loss:.4f}")

    # Build Beta model (WITH LCX)
    model = SwarmByteRingModel(
        num_bits=NUM_BITS,
        embedding_dim=D,
        depth=DEPTH,
        num_beings=NUM_BEINGS,
        num_memory_positions=SEQ_LEN,
        use_lcx=USE_LCX,
        lcx_mode=LCX_MODE,
        lcx_num_levels=LCX_NUM_LEVELS,
        lcx_level_slots=LCX_LEVEL_SLOTS,
        lcx_key_dim=LCX_KEY_DIM,
        lcx_top_k=LCX_TOP_K,
        num_pointers=NUM_POINTERS,
        attention_radius=ATTENTION_RADIUS,
        think_ticks=THINK_TICKS,
    )

    # Load Alpha brain weights, handling size mismatches
    # input_proj changes shape: use_lcx=False -> [D, 16], use_lcx=True hash -> [D, 8]
    alpha_sd = alpha_ckpt['model_state_dict']
    beta_sd = model.state_dict()

    # Filter out keys with shape mismatch
    compatible = {}
    skipped = []
    for k, v in alpha_sd.items():
        if k in beta_sd:
            if v.shape == beta_sd[k].shape:
                compatible[k] = v
            else:
                skipped.append(f"{k}: {list(v.shape)} -> {list(beta_sd[k].shape)}")
        else:
            skipped.append(f"{k}: not in Beta model")

    missing_keys = [k for k in beta_sd if k not in compatible]
    model.load_state_dict(compatible, strict=False)

    print(f"  Loaded brain weights: {len(compatible)} matched, {len(missing_keys)} new LCX params")
    if skipped:
        print(f"  Skipped (shape mismatch):")
        for s in skipped:
            print(f"    {s}")
    if missing_keys:
        for m in missing_keys[:5]:
            print(f"    NEW: {m}")
        if len(missing_keys) > 5:
            print(f"    ... and {len(missing_keys) - 5} more")

    # Verify last BN layer is zero-init (from our swarm_model.py change)
    if hasattr(model, 'lcx_bn_layers') and model.lcx_bn_layers is not None:
        last_bn = model.lcx_bn_layers[-1]
        bn_norm = last_bn.weight.data.norm().item()
        print(f"  BN last layer norm: {bn_norm:.6f} {'(ZERO - correct!)' if bn_norm < 0.001 else '(NOT ZERO - check init!)'}")

    model.train()
    model.to('cpu')

    n_params = sum(p.numel() for p in model.parameters())
    n_alpha = sum(p.numel() for p in alpha_ckpt['model_state_dict'].values())
    n_lcx_new = n_params - n_alpha
    print(f"  Total params: {n_params:,} (Alpha: {n_alpha:,} + LCX: {n_lcx_new:,})")

    # Optimizer (fresh - don't load Alpha optimizer, LR schedule is different)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()

    # Alpha baseline eval (before any Beta training)
    print("-" * 70)
    print("  Evaluating Alpha brain baseline (before Beta training)...")
    baseline = evaluate(model, corpus, criterion)
    pb_str = ' '.join(f"b{i}={p:.0%}" for i, p in enumerate(baseline['per_bit']))
    print(f"  BASELINE: loss={baseline['eval_loss']:.4f} "
          f"bit_acc={baseline['eval_bit_acc']:.3f} "
          f"byte_acc={baseline['eval_byte_acc']:.3f}")
    print(f"  Per-bit:  {pb_str}")
    print(f"  This is the Alpha ceiling. Beta must beat this to prove LCX helps.")
    print("-" * 70)

    alpha_ceiling = baseline['eval_bit_acc']

    # Training state
    best_eval_loss = baseline['eval_loss']
    best_eval_acc = baseline['eval_bit_acc']
    loss_window = []
    acc_window = []
    eval_history = [{'step': 0, 'label': 'alpha_baseline', **baseline}]
    train_start = time.time()
    total_train_time = 0.0
    beat_alpha = False

    dash_log = open(DASHBOARD_LOG, 'w')

    try:
        for step in range(1, TOTAL_STEPS + 1):
            print(f'starting step {step}...', flush=True)
            t0 = time.time()

            # LR schedule
            if step <= WARMUP_STEPS:
                cur_lr = LR * step / WARMUP_STEPS
            else:
                progress = (step - WARMUP_STEPS) / max(1, TOTAL_STEPS - WARMUP_STEPS)
                cur_lr = LR_MIN + 0.5 * (LR - LR_MIN) * (1.0 + math.cos(math.pi * progress))
            for pg in optimizer.param_groups:
                pg['lr'] = cur_lr

            # Generate batch
            x, y, mask = generate_batch_binary_bits(
                corpus, n_samples=BATCH_SIZE, seq_len=SEQ_LEN, num_bits=NUM_BITS,
            )

            # Forward
            out = model(x)
            loss = criterion(out, y)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            step_time = time.time() - t0
            total_train_time += step_time

            if step_time > STEP_TIMEOUT:
                print(f'TIMEOUT: step took {step_time:.0f}s, aborting')
                sys.exit(1)

            loss_val = loss.item()
            with torch.no_grad():
                preds = (torch.sigmoid(out) > 0.5).float()
                bit_acc = (preds == y).float().mean().item()
                byte_acc = (preds == y).float().prod(dim=-1).mean().item()

            loss_window.append(loss_val)
            acc_window.append(bit_acc)
            if len(loss_window) > 50:
                loss_window.pop(0)
                acc_window.pop(0)
            avg_loss = sum(loss_window) / len(loss_window)
            avg_acc = sum(acc_window) / len(acc_window)

            # Dashboard log
            dash_line = (f"step {step} | loss {loss_val:.6f} | "
                         f"acc={bit_acc:.4f} RD:{step_time:.4f} "
                         f"traction={bit_acc:.4f} shard=0/0")
            dash_log.write(dash_line + '\n')
            dash_log.flush()

            # Divergence check
            if math.isnan(loss_val) or math.isinf(loss_val):
                print(f'  DIVERGED: loss={loss_val} at step {step}')
                break
            if loss_val > 10.0:
                print(f'  DIVERGED: loss={loss_val:.4f} > 10.0 at step {step}')
                break

            # Print
            if step <= 10 or step % EVAL_EVERY == 0 or step % 50 == 0:
                # Check BN layer 2 norm (is LCX opening up?)
                bn2_norm = 0.0
                if hasattr(model, 'lcx_bn_layers') and model.lcx_bn_layers is not None:
                    bn2_norm = model.lcx_bn_layers[-1].weight.data.norm().item()

                print(f'  step {step:>5} | loss={loss_val:.4f} avg50={avg_loss:.4f} '
                      f'bit_acc={bit_acc:.3f} byte={byte_acc:.3f} '
                      f'bn2={bn2_norm:.4f} lr={cur_lr:.6f} gnorm={grad_norm:.2f} '
                      f'{step_time:.2f}s')

            # Eval
            if step % EVAL_EVERY == 0:
                em = evaluate(model, corpus, criterion)
                em['step'] = step
                em['train_loss_avg50'] = round(avg_loss, 4)
                em['train_acc_avg50'] = round(avg_acc, 4)
                em['lr'] = round(cur_lr, 7)
                em['grad_norm'] = round(float(grad_norm), 4)
                em['elapsed_s'] = round(time.time() - train_start, 1)

                # BN2 norm tracking
                if hasattr(model, 'lcx_bn_layers') and model.lcx_bn_layers is not None:
                    em['bn2_norm'] = round(model.lcx_bn_layers[-1].weight.data.norm().item(), 4)

                eval_history.append(em)

                pb_str = ' '.join(f"b{i}={p:.0%}" for i, p in enumerate(em['per_bit']))
                content_avg = sum(em['per_bit'][3:]) / 5
                delta_vs_alpha = em['eval_bit_acc'] - alpha_ceiling
                marker = " *** BEAT ALPHA!" if delta_vs_alpha > 0.005 else ""

                print(f'    EVAL:  loss={em["eval_loss"]:.4f} '
                      f'bit_acc={em["eval_bit_acc"]:.3f} '
                      f'byte_acc={em["eval_byte_acc"]:.3f} '
                      f'content={content_avg:.1%} '
                      f'vs_alpha={delta_vs_alpha:+.3f}'
                      f'{marker}')
                print(f'    BITS:  {pb_str}')
                print(f'    BN2:   norm={em.get("bn2_norm", 0):.4f}')

                if not beat_alpha and delta_vs_alpha > 0.005:
                    beat_alpha = True
                    print(f'    *** FIRST TIME BEATING ALPHA CEILING ({alpha_ceiling:.3f}) ***')
                    print(f'    *** LCX IS HELPING! ***')

                if em['eval_loss'] < best_eval_loss:
                    best_eval_loss = em['eval_loss']
                if em['eval_bit_acc'] > best_eval_acc:
                    best_eval_acc = em['eval_bit_acc']

                # Plateau check
                if len(eval_history) >= 6:
                    last5 = [e['eval_bit_acc'] for e in eval_history[-5:]]
                    spread = max(last5) - min(last5)
                    if spread < 0.01:
                        print(f'    ** PLATEAU: spread={spread:.3f}')

            # Checkpoint
            if step % CHECKPOINT_EVERY == 0:
                ckpt_path = os.path.join(CHECKPOINT_DIR, f'cpu_beta_step{step}.pt')
                torch.save({
                    'step': step,
                    'alpha_step': alpha_step,
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
                        'lcx_mode': LCX_MODE, 'lcx_num_levels': LCX_NUM_LEVELS,
                        'lcx_level_slots': LCX_LEVEL_SLOTS,
                        'lcx_key_dim': LCX_KEY_DIM, 'lcx_top_k': LCX_TOP_K,
                    },
                    'alpha_ceiling': alpha_ceiling,
                    'beat_alpha': beat_alpha,
                }, ckpt_path)
                print(f'    Checkpoint saved: {ckpt_path}')

    except KeyboardInterrupt:
        print(f'\n  Interrupted at step {step}')

    finally:
        dash_log.close()

    # Final summary
    wall_time = time.time() - train_start
    last_step = eval_history[-1]['step'] if len(eval_history) > 1 else 0

    print(f"\n\n{'='*70}")
    print(f"  CPU BETA -- FINAL RESULTS")
    print(f"{'='*70}")
    print(f"  Alpha checkpoint: step {alpha_step}, ceiling: {alpha_ceiling:.3f}")
    print(f"  Beta steps: {last_step}")
    print(f"  Beat Alpha: {'YES!' if beat_alpha else 'No'}")
    print(f"  Best eval: loss={best_eval_loss:.4f} bit_acc={best_eval_acc:.3f}")
    print(f"  Wall time: {wall_time:.0f}s ({wall_time/60:.1f} min)")
    if last_step > 0:
        print(f"  Avg step: {total_train_time / last_step:.2f}s")

    # Eval table
    if len(eval_history) > 1:
        print(f"\n  Eval History:")
        print(f"  {'step':>6} {'loss':>8} {'bit':>6} {'byte':>6} | "
              f"{'b0':>4} {'b1':>4} {'b2':>4} {'b3':>4} {'b4':>4} "
              f"{'b5':>4} {'b6':>4} {'b7':>4} | {'bn2':>6} {'vs_a':>6}")
        print(f"  {'-'*85}")
        for e in eval_history:
            pb = e['per_bit']
            delta = e['eval_bit_acc'] - alpha_ceiling
            bn2 = e.get('bn2_norm', 0)
            print(f"  {e['step']:>6} {e['eval_loss']:>8.4f} {e['eval_bit_acc']:>6.3f} "
                  f"{e['eval_byte_acc']:>6.3f} | "
                  f"{pb[0]:>4.0%} {pb[1]:>4.0%} {pb[2]:>4.0%} {pb[3]:>4.0%} "
                  f"{pb[4]:>4.0%} {pb[5]:>4.0%} {pb[6]:>4.0%} {pb[7]:>4.0%} | "
                  f"{bn2:>6.3f} {delta:>+6.3f}")

    # Save results
    results = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'alpha_step': alpha_step,
        'alpha_ceiling': alpha_ceiling,
        'beat_alpha': beat_alpha,
        'best_eval_loss': round(best_eval_loss, 6),
        'best_eval_acc': round(best_eval_acc, 4),
        'wall_time_s': round(wall_time, 1),
        'eval_history': eval_history,
    }
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results: {RESULTS_PATH}")
    print("=" * 70)


if __name__ == '__main__':
    main()
