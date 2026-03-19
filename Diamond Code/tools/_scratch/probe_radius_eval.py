"""
Radius & seq_len eval probe: test if attention window is the bottleneck.

Loads CPU Beta checkpoint (trained with radius=8, seq=192) and evaluates
with different radius and seq_len values. NO training — pure eval.

Key question: does wider radius or longer context break through 79%?
"""

import sys, os, time, json

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
# Fixed config (from CPU Beta)
# ---------------------------------------------------------------------------
D = 618
DEPTH = 2
NUM_BITS = 8
NUM_BEINGS = 1
NUM_POINTERS = 1
BATCH_SIZE = 1

USE_LCX = True
THINK_TICKS = 1
LCX_MODE = 'hash'
LCX_NUM_LEVELS = 1
LCX_LEVEL_SLOTS = [200]
LCX_KEY_DIM = 61
LCX_TOP_K = 2

EVAL_SAMPLES = 50       # More samples for stable signal
STEP_TIMEOUT = 60

# ---------------------------------------------------------------------------
# Test grid
# ---------------------------------------------------------------------------
RADIUS_VALUES = [4, 6, 8, 12, 16, 24, 32]
SEQ_LEN_VALUES = [128, 192, 256, 384]

# Also test LCX OFF for comparison
LCX_TESTS = [True, False]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(SCRIPT_DIR, '..', '..')
CORPUS_PATH = os.path.join(ROOT_DIR, 'data', 'traindat', 'fineweb_edu.traindat')
BETA_CKPT_DIR = os.path.join(ROOT_DIR, 'checkpoints', 'cpu_beta')
LOG_DIR = os.path.join(ROOT_DIR, 'logs', 'nano')
RESULTS_PATH = os.path.join(LOG_DIR, 'radius_eval_results.json')

os.makedirs(LOG_DIR, exist_ok=True)


def find_latest_beta_checkpoint():
    if not os.path.isdir(BETA_CKPT_DIR):
        return None
    files = [f for f in os.listdir(BETA_CKPT_DIR) if f.endswith('.pt')]
    if not files:
        return None
    files.sort(key=lambda f: int(f.split('step')[1].split('.')[0]))
    return os.path.join(BETA_CKPT_DIR, files[-1])


def evaluate(model, corpus, criterion, seq_len):
    """Run eval on fresh samples."""
    model.eval()
    eval_losses = []
    eval_bit_accs = []
    eval_byte_accs = []
    eval_per_bit = [[] for _ in range(NUM_BITS)]

    with torch.no_grad():
        for i in range(EVAL_SAMPLES):
            x, y, mask = generate_batch_binary_bits(
                corpus, n_samples=1, seq_len=seq_len, num_bits=NUM_BITS,
                seed=9000 + i,
            )
            out = model(x)
            # out might have different seq_len dim, match y's shape
            min_len = min(out.shape[1], y.shape[1])
            out = out[:, :min_len, :]
            y = y[:, :min_len, :]

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

    avg_loss = sum(eval_losses) / len(eval_losses)
    avg_bit_acc = sum(eval_bit_accs) / len(eval_bit_accs)
    avg_byte_acc = sum(eval_byte_accs) / len(eval_byte_accs)
    per_bit = [sum(b) / len(b) for b in eval_per_bit]

    return {
        'eval_loss': round(avg_loss, 6),
        'eval_bit_acc': round(avg_bit_acc, 4),
        'eval_byte_acc': round(avg_byte_acc, 4),
        'per_bit': [round(p, 3) for p in per_bit],
    }


def build_model(seq_len, radius, use_lcx):
    """Build a fresh model with given config."""
    model = SwarmByteRingModel(
        num_bits=NUM_BITS,
        embedding_dim=D,
        depth=DEPTH,
        num_beings=NUM_BEINGS,
        num_memory_positions=seq_len,
        use_lcx=use_lcx,
        lcx_mode=LCX_MODE,
        lcx_num_levels=LCX_NUM_LEVELS,
        lcx_level_slots=LCX_LEVEL_SLOTS,
        lcx_key_dim=LCX_KEY_DIM,
        lcx_top_k=LCX_TOP_K,
        num_pointers=NUM_POINTERS,
        attention_radius=radius,
        think_ticks=THINK_TICKS if use_lcx else 0,
    )
    model.to('cpu')
    model.eval()
    return model


def load_checkpoint_flexible(model, state_dict):
    """Load checkpoint, handling seq_len size mismatches by padding/truncating."""
    model_sd = model.state_dict()
    filtered_sd = {}
    skipped = []

    for key, value in state_dict.items():
        if key in model_sd:
            if model_sd[key].shape == value.shape:
                filtered_sd[key] = value
            else:
                # Size mismatch — try to adapt
                target_shape = model_sd[key].shape
                source_shape = value.shape

                if len(target_shape) == 1 and len(source_shape) == 1:
                    # 1D tensor — pad or truncate
                    if target_shape[0] > source_shape[0]:
                        # Pad with zeros
                        padded = torch.zeros(target_shape)
                        padded[:source_shape[0]] = value
                        filtered_sd[key] = padded
                    else:
                        # Truncate
                        filtered_sd[key] = value[:target_shape[0]]
                else:
                    skipped.append(f"{key}: {source_shape} -> {target_shape}")
        else:
            skipped.append(f"{key}: not in model")

    missing, unexpected = model.load_state_dict(filtered_sd, strict=False)
    return missing, skipped


def main():
    print("=" * 70)
    print("  RADIUS & SEQ_LEN EVAL PROBE")
    print("  Checkpoint: CPU Beta step 1100 (trained r=8, seq=192, LCX ON)")
    print("  Question: which parameter is the bottleneck?")
    print("=" * 70)

    # Load corpus
    with open(CORPUS_PATH, 'rb') as f:
        corpus = f.read()
    print(f"  Corpus: {len(corpus) / 1024 / 1024:.1f} MB")

    # Load checkpoint
    beta_ckpt_path = find_latest_beta_checkpoint()
    if not beta_ckpt_path:
        print("  ERROR: No Beta checkpoint found")
        sys.exit(1)
    print(f"  Checkpoint: {beta_ckpt_path}")

    beta_ckpt = torch.load(beta_ckpt_path, map_location='cpu', weights_only=False)
    beta_step = beta_ckpt['step']
    print(f"  Step: {beta_step}")
    print("-" * 70)

    criterion = nn.BCEWithLogitsLoss()
    results = []

    # -----------------------------------------------------------------------
    # Part 1: Radius sweep (fixed seq_len=192, LCX ON)
    # -----------------------------------------------------------------------
    print("\n  PART 1: RADIUS SWEEP (seq=192, LCX ON)")
    print("  " + "-" * 66)
    print(f"  {'radius':>6}  {'bit_acc':>8}  {'byte':>6}  {'loss':>8}  {'b0':>4} {'b1':>4} {'b2':>4} {'b3':>4} {'b4':>4} {'b5':>4} {'b6':>4} {'b7':>4}  {'time':>5}")
    print("  " + "-" * 66)

    for radius in RADIUS_VALUES:
        t0 = time.time()
        model = build_model(seq_len=192, radius=radius, use_lcx=True)
        missing, unexpected = model.load_state_dict(beta_ckpt['model_state_dict'], strict=False)
        if missing:
            print(f"  WARNING r={radius}: missing {len(missing)} keys")
            # Skip if too many missing keys — architecture mismatch
            if len(missing) > 5:
                print(f"  SKIP r={radius}: too many missing keys")
                continue

        res = evaluate(model, corpus, criterion, seq_len=192)
        dt = time.time() - t0

        pb = res['per_bit']
        marker = " <<<" if radius == 8 else ""
        print(f"  {radius:>6}  {res['eval_bit_acc']*100:>7.1f}%  {res['eval_byte_acc']*100:>5.1f}%  {res['eval_loss']:>8.4f}  "
              f"{pb[0]*100:>3.0f}% {pb[1]*100:>3.0f}% {pb[2]*100:>3.0f}% {pb[3]*100:>3.0f}% "
              f"{pb[4]*100:>3.0f}% {pb[5]*100:>3.0f}% {pb[6]*100:>3.0f}% {pb[7]*100:>3.0f}%  "
              f"{dt:>4.0f}s{marker}")

        results.append({
            'test': 'radius_sweep',
            'radius': radius, 'seq_len': 192, 'use_lcx': True,
            **res, 'time_s': round(dt, 1)
        })

        del model
        sys.stdout.flush()

    # -----------------------------------------------------------------------
    # Part 2: Seq_len sweep (fixed radius=8, LCX ON)
    # -----------------------------------------------------------------------
    print(f"\n  PART 2: SEQ_LEN SWEEP (radius=8, LCX ON)")
    print("  " + "-" * 66)
    print(f"  {'seq_len':>7}  {'bit_acc':>8}  {'byte':>6}  {'loss':>8}  {'b0':>4} {'b1':>4} {'b2':>4} {'b3':>4} {'b4':>4} {'b5':>4} {'b6':>4} {'b7':>4}  {'time':>5}")
    print("  " + "-" * 66)

    for seq_len in SEQ_LEN_VALUES:
        t0 = time.time()
        model = build_model(seq_len=seq_len, radius=8, use_lcx=True)
        missing, skipped = load_checkpoint_flexible(model, beta_ckpt['model_state_dict'])
        if skipped:
            print(f"  NOTE seq={seq_len}: adapted {len(skipped)} keys")

        res = evaluate(model, corpus, criterion, seq_len=seq_len)
        dt = time.time() - t0

        pb = res['per_bit']
        marker = " <<<" if seq_len == 192 else ""
        print(f"  {seq_len:>7}  {res['eval_bit_acc']*100:>7.1f}%  {res['eval_byte_acc']*100:>5.1f}%  {res['eval_loss']:>8.4f}  "
              f"{pb[0]*100:>3.0f}% {pb[1]*100:>3.0f}% {pb[2]*100:>3.0f}% {pb[3]*100:>3.0f}% "
              f"{pb[4]*100:>3.0f}% {pb[5]*100:>3.0f}% {pb[6]*100:>3.0f}% {pb[7]*100:>3.0f}%  "
              f"{dt:>4.0f}s{marker}")

        results.append({
            'test': 'seq_len_sweep',
            'radius': 8, 'seq_len': seq_len, 'use_lcx': True,
            **res, 'time_s': round(dt, 1)
        })

        del model
        sys.stdout.flush()

    # -----------------------------------------------------------------------
    # Part 3: LCX ON vs OFF (fixed radius=8, seq=192)
    # -----------------------------------------------------------------------
    print(f"\n  PART 3: LCX ON vs OFF (radius=8, seq=192)")
    print("  " + "-" * 66)
    print(f"  {'lcx':>5}  {'bit_acc':>8}  {'byte':>6}  {'loss':>8}  {'b0':>4} {'b1':>4} {'b2':>4} {'b3':>4} {'b4':>4} {'b5':>4} {'b6':>4} {'b7':>4}  {'time':>5}")
    print("  " + "-" * 66)

    for use_lcx in LCX_TESTS:
        t0 = time.time()
        tt = 1 if use_lcx else 0
        model = build_model(seq_len=192, radius=8, use_lcx=use_lcx)
        missing, unexpected = model.load_state_dict(beta_ckpt['model_state_dict'], strict=False)

        res = evaluate(model, corpus, criterion, seq_len=192)
        dt = time.time() - t0

        pb = res['per_bit']
        label = "ON" if use_lcx else "OFF"
        print(f"  {label:>5}  {res['eval_bit_acc']*100:>7.1f}%  {res['eval_byte_acc']*100:>5.1f}%  {res['eval_loss']:>8.4f}  "
              f"{pb[0]*100:>3.0f}% {pb[1]*100:>3.0f}% {pb[2]*100:>3.0f}% {pb[3]*100:>3.0f}% "
              f"{pb[4]*100:>3.0f}% {pb[5]*100:>3.0f}% {pb[6]*100:>3.0f}% {pb[7]*100:>3.0f}%  "
              f"{dt:>4.0f}s")

        results.append({
            'test': 'lcx_toggle',
            'radius': 8, 'seq_len': 192, 'use_lcx': use_lcx,
            **res, 'time_s': round(dt, 1)
        })

        del model
        sys.stdout.flush()

    # -----------------------------------------------------------------------
    # Part 4: Best combo — radius=16, seq=256, LCX ON
    # -----------------------------------------------------------------------
    print(f"\n  PART 4: D=2000 LONGRUN CONFIG (radius=16, seq=256, LCX ON)")
    print("  " + "-" * 66)

    t0 = time.time()
    model = build_model(seq_len=256, radius=16, use_lcx=True)
    missing, skipped = load_checkpoint_flexible(model, beta_ckpt['model_state_dict'])
    if skipped:
        print(f"  NOTE: adapted {len(skipped)} keys for seq_len change")

    res = evaluate(model, corpus, criterion, seq_len=256)
    dt = time.time() - t0

    pb = res['per_bit']
    print(f"  r=16 s=256  {res['eval_bit_acc']*100:>7.1f}%  {res['eval_byte_acc']*100:>5.1f}%  {res['eval_loss']:>8.4f}  "
          f"{pb[0]*100:>3.0f}% {pb[1]*100:>3.0f}% {pb[2]*100:>3.0f}% {pb[3]*100:>3.0f}% "
          f"{pb[4]*100:>3.0f}% {pb[5]*100:>3.0f}% {pb[6]*100:>3.0f}% {pb[7]*100:>3.0f}%  "
          f"{dt:>4.0f}s")

    results.append({
        'test': 'longrun_combo',
        'radius': 16, 'seq_len': 256, 'use_lcx': True,
        **res, 'time_s': round(dt, 1)
    })

    del model

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    # Find best from each sweep
    radius_results = [r for r in results if r['test'] == 'radius_sweep']
    seq_results = [r for r in results if r['test'] == 'seq_len_sweep']

    if radius_results:
        baseline_r = next((r for r in radius_results if r['radius'] == 8), None)
        best_r = max(radius_results, key=lambda r: r['eval_bit_acc'])
        if baseline_r:
            print(f"  Radius baseline (r=8):  {baseline_r['eval_bit_acc']*100:.1f}%")
            print(f"  Radius best (r={best_r['radius']}):     {best_r['eval_bit_acc']*100:.1f}%  "
                  f"delta={((best_r['eval_bit_acc'] - baseline_r['eval_bit_acc'])*100):+.1f}%")

    if seq_results:
        baseline_s = next((r for r in seq_results if r['seq_len'] == 192), None)
        best_s = max(seq_results, key=lambda r: r['eval_bit_acc'])
        if baseline_s:
            print(f"  Seq baseline (s=192):   {baseline_s['eval_bit_acc']*100:.1f}%")
            print(f"  Seq best (s={best_s['seq_len']}):       {best_s['eval_bit_acc']*100:.1f}%  "
                  f"delta={((best_s['eval_bit_acc'] - baseline_s['eval_bit_acc'])*100):+.1f}%")

    lcx_results = [r for r in results if r['test'] == 'lcx_toggle']
    if len(lcx_results) == 2:
        lcx_on = next(r for r in lcx_results if r['use_lcx'])
        lcx_off = next(r for r in lcx_results if not r['use_lcx'])
        print(f"  LCX ON:   {lcx_on['eval_bit_acc']*100:.1f}%")
        print(f"  LCX OFF:  {lcx_off['eval_bit_acc']*100:.1f}%  "
              f"delta={((lcx_off['eval_bit_acc'] - lcx_on['eval_bit_acc'])*100):+.1f}%")

    combo = next((r for r in results if r['test'] == 'longrun_combo'), None)
    if combo:
        print(f"  Combo (r=16,s=256):     {combo['eval_bit_acc']*100:.1f}%")

    # Save results
    with open(RESULTS_PATH, 'w') as f:
        json.dump({
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'checkpoint': beta_ckpt_path,
            'checkpoint_step': beta_step,
            'trained_config': {'radius': 8, 'seq_len': 192, 'use_lcx': True, 'D': D},
            'eval_samples': EVAL_SAMPLES,
            'results': results,
        }, f, indent=2)
    print(f"\n  Results: {RESULTS_PATH}")
    print("=" * 70)


if __name__ == '__main__':
    main()
