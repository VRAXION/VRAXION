"""
GPU Ant inference comparison: golden vs scratch LCX, and think_ticks sweep.

- draft_step_0000800.pt = golden-equivalent LCX (loaded at restart, became golden)
- checkpoint_latest.pt  = scratch LCX (25+ steps of writes since restart)
- think_ticks sweep: 1, 2, 3, 4 at inference time

Runs on CPU, does not touch GPU.
"""

import sys, os, random, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn as nn
from swarm_model import SwarmByteRingModel

# Must match run_goldilocks.bat
MODEL_CONFIG = dict(
    num_bits=8,
    embedding_dim=6180,
    depth=2,
    num_beings=1,
    num_memory_positions=192,
    use_lcx=True,
    lcx_mode='hash',
    lcx_num_levels=1,
    lcx_level_slots=[2000],
    lcx_key_dim=618,
    lcx_top_k=2,
    num_pointers=1,
    attention_radius=16,
    think_ticks=1,  # will be changed at runtime
)

SEQ_LEN = 192
BASE_DIR = os.path.join(os.path.dirname(__file__), '..', '..')
DATA_PATH = os.path.join(BASE_DIR, 'data', 'traindat', 'fineweb_edu.traindat')
CKPT_DIR = os.path.join(BASE_DIR, 'checkpoints', 'curriculum_v2')


def run_eval(model, samples, label=""):
    """Run inference on pre-encoded samples, return summary stats."""
    model.eval()
    all_bit_acc = []
    all_byte_acc = []
    per_bit_acc = [[] for _ in range(8)]

    with torch.no_grad():
        for x, y in samples:
            output = model(x)
            probs = torch.sigmoid(output)
            preds = (probs > 0.5).float()

            T = x.shape[1]
            bit_correct = (preds[0] == y[0]).float()
            bit_acc = bit_correct.mean().item()
            all_bit_acc.append(bit_acc)

            # Per-bit
            for b in range(8):
                bacc = bit_correct[:, b].mean().item()
                per_bit_acc[b].append(bacc)

            # Byte accuracy
            byte_matches = 0
            for t in range(T):
                pred_byte = 0
                actual_byte = 0
                for b in range(8):
                    pred_byte |= int(preds[0, t, b].item()) << (7 - b)
                    actual_byte |= int(y[0, t, b].item()) << (7 - b)
                if pred_byte == actual_byte:
                    byte_matches += 1
            all_byte_acc.append(byte_matches / T)

    avg_bit = sum(all_bit_acc) / len(all_bit_acc)
    avg_byte = sum(all_byte_acc) / len(all_byte_acc)
    avg_per_bit = [sum(b) / len(b) for b in per_bit_acc]

    return {
        'label': label,
        'bit_acc': avg_bit,
        'byte_acc': avg_byte,
        'per_bit': avg_per_bit,
        'n_samples': len(samples),
    }


def main():
    print("=" * 70)
    print("  GPU ANT INFERENCE COMPARISON")
    print("  Golden vs Scratch LCX + Think Ticks Sweep")
    print("  Running on CPU")
    print("=" * 70)

    # Prepare samples from corpus
    with open(DATA_PATH, 'rb') as f:
        corpus = f.read()

    N_SAMPLES = 10
    random.seed(42)
    samples = []
    for _ in range(N_SAMPLES):
        start = random.randint(0, len(corpus) - SEQ_LEN - 10)
        chunk = corpus[start:start + SEQ_LEN + 1]
        arr = np.frombuffer(chunk, dtype=np.uint8)
        bits = np.unpackbits(arr.reshape(-1, 1), axis=1).astype(np.float32)
        x = torch.from_numpy(bits[:SEQ_LEN]).unsqueeze(0)
        y = torch.from_numpy(bits[1:SEQ_LEN + 1]).unsqueeze(0)
        samples.append((x, y))

    print(f"  Prepared {N_SAMPLES} samples from fineweb_edu ({SEQ_LEN} bytes each)")

    # Create model once
    print(f"  Creating model (D=6180, this takes a moment)...")
    model = SwarmByteRingModel(**MODEL_CONFIG)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Checkpoints to compare
    golden_path = os.path.join(CKPT_DIR, 'drafts', 'draft_step_0000800.pt')
    scratch_path = os.path.join(CKPT_DIR, 'checkpoint_latest.pt')

    checkpoints = []
    if os.path.exists(golden_path):
        checkpoints.append(('GOLDEN (step 800)', golden_path))
    if os.path.exists(scratch_path):
        checkpoints.append(('SCRATCH (latest)', scratch_path))

    # Also try earlier drafts if available
    for step_n in [775, 750, 700]:
        p = os.path.join(CKPT_DIR, 'drafts', f'draft_step_0000{step_n}.pt')
        if os.path.exists(p):
            checkpoints.append((f'DRAFT (step {step_n})', p))
            break  # just one earlier for reference

    # Think ticks to test
    TT_VALUES = [1, 2, 3, 4]

    # Results table
    results = []

    for ckpt_name, ckpt_path in checkpoints:
        print(f"\n  Loading: {ckpt_name} ({os.path.basename(ckpt_path)})")
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        step = ckpt.get('step', '?')

        sd = ckpt.get('model_state_dict', ckpt.get('weights', {}))
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print(f"    New params: {len(missing)}")
        if unexpected:
            print(f"    Unexpected: {len(unexpected)}")
        print(f"    Step: {step}")

        for tt in TT_VALUES:
            # Change think_ticks at runtime
            model.think_ticks = tt
            model.think_ticks_per_being = [tt]

            label = f"{ckpt_name} tt={tt}"
            print(f"    Testing tt={tt}...", end='', flush=True)
            t0 = time.time()
            r = run_eval(model, samples, label)
            dt = time.time() - t0
            r['time'] = dt
            r['step'] = step
            results.append(r)
            print(f" bit={r['bit_acc']:.1%} byte={r['byte_acc']:.1%} ({dt:.1f}s)")

    # Summary table
    print(f"\n\n{'='*90}")
    print(f"  RESULTS COMPARISON ({N_SAMPLES} samples, {SEQ_LEN} bytes each)")
    print(f"{'='*90}\n")

    header = (f"{'Config':<30} {'bit_acc':>7} {'byte_acc':>8} "
              f"{'b0':>4} {'b1':>4} {'b2':>4} {'b3':>4} {'b4':>4} "
              f"{'b5':>4} {'b6':>4} {'b7':>4} {'time':>6}")
    print(header)
    print("-" * len(header))

    for r in results:
        pb = r['per_bit']
        print(f"{r['label']:<30} {r['bit_acc']:>6.1%} {r['byte_acc']:>7.1%} "
              f"{pb[0]:>4.0%} {pb[1]:>4.0%} {pb[2]:>4.0%} {pb[3]:>4.0%} "
              f"{pb[4]:>4.0%} {pb[5]:>4.0%} {pb[6]:>4.0%} {pb[7]:>4.0%} "
              f"{r['time']:>5.1f}s")

    # Analysis
    print(f"\n  --- ANALYSIS ---")

    # Golden vs Scratch at tt=1
    golden_tt1 = [r for r in results if 'GOLDEN' in r['label'] and 'tt=1' in r['label']]
    scratch_tt1 = [r for r in results if 'SCRATCH' in r['label'] and 'tt=1' in r['label']]
    if golden_tt1 and scratch_tt1:
        g, s = golden_tt1[0], scratch_tt1[0]
        delta = g['bit_acc'] - s['bit_acc']
        print(f"  Golden vs Scratch (tt=1): {delta:+.1%} bit_acc")
        if delta > 0.01:
            print(f"    -> Golden IS better (as expected)")
        elif delta < -0.01:
            print(f"    -> Scratch is better?! (unexpected)")
        else:
            print(f"    -> Negligible difference")

    # Think ticks effect
    for ckpt_name, _ in checkpoints:
        tt_results = [r for r in results if ckpt_name in r['label']]
        if len(tt_results) >= 2:
            tt1 = tt_results[0]
            best_tt = max(tt_results, key=lambda r: r['bit_acc'])
            if best_tt['label'] != tt1['label']:
                delta = best_tt['bit_acc'] - tt1['bit_acc']
                print(f"  {ckpt_name}: best is {best_tt['label']} ({delta:+.1%} vs tt=1)")
            else:
                print(f"  {ckpt_name}: tt=1 is already the best")

    print(f"\n{'='*90}")


if __name__ == '__main__':
    main()
