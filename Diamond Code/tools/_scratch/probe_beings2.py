"""
Beings=2 probe: does a second being break the 79% plateau?

Loads CPU Beta checkpoint (beings=1), builds model with beings=2.
Being 0 = trained weights, Being 1 = random init.
Train 500 steps on fineweb_edu, see if accuracy improves.

Theory: two independent reading heads on the ring might
specialize — one for structure (bit0-2), one for content (bit3-7).
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
# Config — same as CPU Beta EXCEPT beings=2
# ---------------------------------------------------------------------------
D = 618
DEPTH = 2
SEQ_LEN = 192
BATCH_SIZE = 10
NUM_BITS = 8
NUM_BEINGS = 2           # <--- THE CHANGE: was 1
ATTENTION_RADIUS = 8
NUM_POINTERS = 1

USE_LCX = True
THINK_TICKS = 1
LCX_MODE = 'hash'
LCX_NUM_LEVELS = 1
LCX_LEVEL_SLOTS = [200]
LCX_KEY_DIM = 61
LCX_TOP_K = 2

LR = 0.0003
LR_MIN = 1e-5
WARMUP_STEPS = 10
GRAD_CLIP = 1.0
WEIGHT_DECAY = 0

TOTAL_STEPS = 500
EVAL_EVERY = 25
CHECKPOINT_EVERY = 100
EVAL_SAMPLES = 20

STEP_TIMEOUT = 120

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(SCRIPT_DIR, '..', '..')

CORPUS_PATH = os.path.join(ROOT_DIR, 'data', 'traindat', 'fineweb_edu.traindat')
BETA_CKPT_DIR = os.path.join(ROOT_DIR, 'checkpoints', 'cpu_beta')
CHECKPOINT_DIR = os.path.join(ROOT_DIR, 'checkpoints', 'cpu_beings2')
LOG_DIR = os.path.join(ROOT_DIR, 'logs', 'nano')
RESULTS_PATH = os.path.join(LOG_DIR, 'beings2_results.json')

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def find_latest_beta_checkpoint():
    if not os.path.isdir(BETA_CKPT_DIR):
        return None
    files = [f for f in os.listdir(BETA_CKPT_DIR) if f.endswith('.pt')]
    if not files:
        return None
    files.sort(key=lambda f: int(f.split('step')[1].split('.')[0]))
    return os.path.join(BETA_CKPT_DIR, files[-1])


def evaluate(model, corpus, criterion):
    model.eval()
    eval_losses, eval_bit_accs, eval_byte_accs = [], [], []
    eval_per_bit = [[] for _ in range(NUM_BITS)]

    with torch.no_grad():
        for i in range(EVAL_SAMPLES):
            x, y, mask = generate_batch_binary_bits(
                corpus, n_samples=1, seq_len=SEQ_LEN, num_bits=NUM_BITS, seed=9000 + i)
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
            eval_byte_accs.append(all_match.mean().item())

    model.train()
    return {
        'eval_loss': round(sum(eval_losses) / len(eval_losses), 6),
        'eval_bit_acc': round(sum(eval_bit_accs) / len(eval_bit_accs), 4),
        'eval_byte_acc': round(sum(eval_byte_accs) / len(eval_byte_accs), 4),
        'per_bit': [round(sum(b) / len(b), 3) for b in eval_per_bit],
    }


def main():
    print("=" * 70)
    print("  CPU BEINGS=2 PROBE")
    print("=" * 70)
    print(f"  Config: D={D}, depth={DEPTH}, seq={SEQ_LEN}, batch={BATCH_SIZE}")
    print(f"          bits={NUM_BITS}, beings={NUM_BEINGS} *** 2 BEINGS ***")
    print(f"          radius={ATTENTION_RADIUS}, use_lcx={USE_LCX}, tt={THINK_TICKS}")
    print(f"          lcx_slots={LCX_LEVEL_SLOTS}, key_dim={LCX_KEY_DIM}, top_k={LCX_TOP_K}")
    print(f"          lr={LR}, warmup={WARMUP_STEPS}, lr_min={LR_MIN}")
    print(f"  CPU threads: {torch.get_num_threads()}")
    print(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 70)

    # Load corpus
    with open(CORPUS_PATH, 'rb') as f:
        corpus = f.read()
    print(f"  Corpus: {len(corpus) / 1024 / 1024:.1f} MB")

    # Load Beta checkpoint
    beta_ckpt_path = find_latest_beta_checkpoint()
    if not beta_ckpt_path:
        print("  ERROR: No Beta checkpoint found")
        sys.exit(1)
    print(f"  Beta checkpoint: {beta_ckpt_path}")
    beta_ckpt = torch.load(beta_ckpt_path, map_location='cpu', weights_only=False)
    beta_step = beta_ckpt['step']
    print(f"  Beta step: {beta_step}")

    # Build model with beings=2
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

    # Load Beta weights — being 0 matches, being 1 stays random
    missing, unexpected = model.load_state_dict(beta_ckpt['model_state_dict'], strict=False)
    if missing:
        print(f"  Missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
    if not missing and not unexpected:
        print(f"  Loaded all weights cleanly")

    model.train()
    model.to('cpu')

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {n_params:,}")

    # Check being states
    for i in range(NUM_BEINGS):
        being = model.beings[i]
        print(f"  Being {i}: ptr_dest norm={being.pointer_destinations.data.norm().item():.4f}, "
              f"ctx_str={being.context_strength.data.item():.4f}, "
              f"jump_gate bias={being.jump_gate.bias.data.item():.4f}")

    print("-" * 70)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()

    # Baseline eval
    baseline = evaluate(model, corpus, criterion)
    pb = baseline['per_bit']
    print(f"  BASELINE: loss={baseline['eval_loss']:.4f} bit_acc={baseline['eval_bit_acc']:.3f} "
          f"byte={baseline['eval_byte_acc']:.3f}")
    print(f"  Per-bit:  " + "  ".join(f"b{i}={pb[i]*100:.0f}%" for i in range(8)))

    beta_baseline_acc = baseline['eval_bit_acc']
    beta_tt1_baseline = 0.791  # CPU Beta beings=1 result

    eval_history = [{
        'step': 0,
        'loss': baseline['eval_loss'],
        'bit_acc': baseline['eval_bit_acc'],
        'byte_acc': baseline['eval_byte_acc'],
        'per_bit': baseline['per_bit'],
    }]

    print("-" * 70)
    print(f"  Reference: CPU Beta (beings=1) = {beta_tt1_baseline*100:.1f}%")
    print("-" * 70)

    # Training loop
    avg_loss = 0.0
    t_start = time.time()

    for step in range(1, TOTAL_STEPS + 1):
        # LR schedule: cosine with warmup
        if step <= WARMUP_STEPS:
            lr = LR * step / WARMUP_STEPS
        else:
            progress = (step - WARMUP_STEPS) / (TOTAL_STEPS - WARMUP_STEPS)
            lr = LR_MIN + 0.5 * (LR - LR_MIN) * (1 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        t0 = time.time()
        print(f"starting step {step}...", flush=True)

        # Generate batch
        x, y, mask = generate_batch_binary_bits(
            corpus, n_samples=BATCH_SIZE, seq_len=SEQ_LEN, num_bits=NUM_BITS)

        # Forward
        out = model(x)
        loss = criterion(out, y)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        if GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        dt = time.time() - t0
        if dt > STEP_TIMEOUT:
            print(f"  TIMEOUT: step took {dt:.0f}s, aborting")
            sys.exit(1)

        preds = (torch.sigmoid(out) > 0.5).float()
        bit_acc = (preds == y).float().mean().item()
        gnorm = sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5

        if step <= 50:
            avg_loss = avg_loss * (step - 1) / step + loss.item() / step
        else:
            avg_loss = avg_loss * 0.98 + loss.item() * 0.02

        if step <= 10 or step % 25 == 0:
            print(f"  step {step:>5} | loss={loss.item():.4f} avg50={avg_loss:.4f} "
                  f"bit_acc={bit_acc:.3f} byte={((preds == y).float().prod(dim=-1).mean().item()):.3f} "
                  f"lr={lr:.6f} gnorm={gnorm:.2f} {dt:.2f}s")

        # Eval
        if step % EVAL_EVERY == 0:
            res = evaluate(model, corpus, criterion)
            pb = res['per_bit']
            vs_b1 = res['eval_bit_acc'] - beta_tt1_baseline
            print(f"    EVAL:  loss={res['eval_loss']:.4f} bit_acc={res['eval_bit_acc']:.3f} "
                  f"byte_acc={res['eval_byte_acc']:.3f} "
                  f"vs_beings1={vs_b1:+.3f}")
            print(f"    BITS:  " + "  ".join(f"b{i}={pb[i]*100:.0f}%" for i in range(8)))

            if vs_b1 > 0:
                print(f"    >>> BEINGS=2 BEATING BEINGS=1 by {vs_b1*100:+.1f}%!")
            elif vs_b1 > -0.02:
                print(f"    >>> BEINGS=2 matching beings=1")
            else:
                print(f"    >>> BEINGS=2 behind beings=1 by {vs_b1*100:.1f}%")

            eval_history.append({
                'step': step,
                'loss': res['eval_loss'],
                'bit_acc': res['eval_bit_acc'],
                'byte_acc': res['eval_byte_acc'],
                'per_bit': res['per_bit'],
                'vs_beings1': round(vs_b1, 4),
            })

            # Plateau detection
            if len(eval_history) >= 6:
                recent = [e['bit_acc'] for e in eval_history[-6:]]
                spread = max(recent) - min(recent)
                if spread < 0.01:
                    print(f"    ** PLATEAU: spread={spread:.3f}")

        # Checkpoint
        if step % CHECKPOINT_EVERY == 0:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f'cpu_beings2_step{step}.pt')
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'bit_acc': eval_history[-1]['bit_acc'] if eval_history else 0,
                'loss': avg_loss,
            }, ckpt_path)
            print(f"    Checkpoint saved: {ckpt_path}")

        sys.stdout.flush()

    # Final summary
    total_time = time.time() - t_start
    print("\n" + "=" * 70)
    print(f"  Wall time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"  Avg step: {total_time/TOTAL_STEPS:.2f}s")

    print(f"\n  Eval History:")
    print(f"  {'step':>5}  {'loss':>8}  {'bit':>5}  {'byte':>5} | "
          f"{'b0':>4} {'b1':>4} {'b2':>4} {'b3':>4} {'b4':>4} {'b5':>4} {'b6':>4} {'b7':>4} | "
          f"{'vs_b1':>6}")
    print("  " + "-" * 85)
    for e in eval_history:
        pb = e['per_bit']
        vs = e.get('vs_beings1', 0)
        print(f"  {e['step']:>5}  {e['loss']:>8.4f}  {e['bit_acc']*100:>4.1f}%  {e['byte_acc']*100:>4.1f}% | "
              f"{pb[0]*100:>3.0f}% {pb[1]*100:>3.0f}% {pb[2]*100:>3.0f}% {pb[3]*100:>3.0f}% "
              f"{pb[4]*100:>3.0f}% {pb[5]*100:>3.0f}% {pb[6]*100:>3.0f}% {pb[7]*100:>3.0f}% | "
              f"{vs:>+.3f}")

    # Save results
    with open(RESULTS_PATH, 'w') as f:
        json.dump({
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'config': {
                'D': D, 'depth': DEPTH, 'beings': NUM_BEINGS,
                'seq_len': SEQ_LEN, 'batch': BATCH_SIZE,
                'radius': ATTENTION_RADIUS, 'tt': THINK_TICKS,
                'lcx_slots': LCX_LEVEL_SLOTS, 'lr': LR,
            },
            'n_params': n_params,
            'beta_checkpoint_step': beta_step,
            'beta_beings1_baseline': beta_tt1_baseline,
            'eval_history': eval_history,
            'total_time': round(total_time, 1),
        }, f, indent=2)
    print(f"\n  Results: {RESULTS_PATH}")
    print("=" * 70)


if __name__ == '__main__':
    main()
