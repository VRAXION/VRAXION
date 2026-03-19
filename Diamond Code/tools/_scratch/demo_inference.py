"""
GPU Ant inference demo — load checkpoint, show character-by-character predictions.

Runs on CPU, does not touch the GPU. Loads the latest checkpoint (weights + LCX)
and feeds in real fineweb_edu text to show what the model can predict.

Usage:
    python tools/_scratch/demo_inference.py
    python tools/_scratch/demo_inference.py --text "The quick brown fox"
    python tools/_scratch/demo_inference.py --checkpoint checkpoints/curriculum_v2/drafts/draft_step_0000800.pt
    python tools/_scratch/demo_inference.py --samples 5
"""

import sys, os, argparse, random
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn as nn
from swarm_model import SwarmByteRingModel

# --- Model config (must match run_goldilocks.bat exactly) ---
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
    attention_radius=16,  # current live value
    think_ticks=1,        # Beta mode: 1 LCX read/write cycle
)
# zoom_gate_init=-4.0 is set post-construction in training; checkpoint already has it

SEQ_LEN = 192
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'traindat', 'fineweb_edu.traindat')
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'checkpoints', 'curriculum_v2')


def bits_to_byte(bits):
    """Convert 8 binary bits (MSB first) to a byte value."""
    val = 0
    for i, b in enumerate(bits):
        val |= (int(b) << (7 - i))
    return val


def byte_to_bits(b):
    """Convert a byte value to 8 binary bits (MSB first)."""
    return [(b >> (7 - i)) & 1 for i in range(8)]


def bit_symbol(acc):
    """Visual symbol for bit accuracy."""
    if acc >= 0.95: return '@'  # perfect
    if acc >= 0.80: return '#'  # strong
    if acc >= 0.65: return '+'  # above chance
    if acc >= 0.55: return '='  # weak
    if acc >= 0.45: return '-'  # chance
    return '.'                  # below chance


def char_display(b):
    """Safe character display."""
    if 32 <= b <= 126:
        return chr(b)
    elif b == 10:
        return '\\n'
    elif b == 13:
        return '\\r'
    elif b == 9:
        return '\\t'
    else:
        return f'x{b:02x}'


def run_inference(model, text_bytes, seq_len=192):
    """Run inference on raw bytes, return predictions and stats."""
    # Ensure we have enough bytes
    n = min(len(text_bytes), seq_len + 1)
    if n < 3:
        print("  ERROR: need at least 3 bytes of input")
        return None

    chunk = text_bytes[:n]
    arr = np.frombuffer(chunk, dtype=np.uint8)

    # Unpack to binary bits: [n, 8]
    bits = np.unpackbits(arr.reshape(-1, 1), axis=1).astype(np.float32)

    # x = bits[0:n-1], y = bits[1:n]
    x = torch.from_numpy(bits[:n-1]).unsqueeze(0)  # [1, T, 8]
    y = torch.from_numpy(bits[1:n]).unsqueeze(0)    # [1, T, 8]

    model.eval()
    with torch.no_grad():
        output = model(x)  # [1, T, 8] logits

    probs = torch.sigmoid(output)
    preds = (probs > 0.5).float()

    # Per-position results
    T = x.shape[1]
    results = []
    for t in range(T):
        actual_bits = y[0, t].numpy()
        pred_bits = preds[0, t].numpy()
        prob_bits = probs[0, t].numpy()

        actual_byte = bits_to_byte(actual_bits)
        pred_byte = bits_to_byte(pred_bits)

        bit_correct = (actual_bits == pred_bits).astype(int)
        bit_acc = bit_correct.mean()

        results.append({
            'pos': t,
            'actual_byte': actual_byte,
            'pred_byte': pred_byte,
            'actual_bits': actual_bits.tolist(),
            'pred_bits': pred_bits.tolist(),
            'prob_bits': prob_bits.tolist(),
            'bit_correct': bit_correct.tolist(),
            'bit_acc': bit_acc,
            'byte_match': actual_byte == pred_byte,
        })

    return results


def print_demo(results, title=""):
    """Print a colorful character-by-character comparison."""
    if not results:
        return

    T = len(results)

    # Per-bit accuracy across all positions
    per_bit = [0.0] * 8
    for r in results:
        for b in range(8):
            per_bit[b] += r['bit_correct'][b]
    per_bit = [v / T for v in per_bit]

    total_bit_acc = sum(r['bit_acc'] for r in results) / T
    byte_matches = sum(1 for r in results if r['byte_match'])
    byte_acc = byte_matches / T

    print(f"\n{'='*80}")
    if title:
        print(f"  {title}")
    print(f"  {T} positions | bit_acc={total_bit_acc:.1%} | byte_acc={byte_acc:.1%} ({byte_matches}/{T})")
    print(f"  per-bit: {' '.join(f'b{i}={per_bit[i]:.0%}' for i in range(8))}")
    print(f"           {' '.join(f' {bit_symbol(per_bit[i])} ' for i in range(8))}")
    print(f"{'='*80}")

    # Actual text
    actual_chars = []
    pred_chars = []
    match_markers = []
    for r in results:
        ac = char_display(r['actual_byte'])
        pc = char_display(r['pred_byte'])
        actual_chars.append(ac)
        pred_chars.append(pc)
        match_markers.append('^' if r['byte_match'] else ' ')

    # Print in chunks of 60 characters
    CHUNK = 60
    for start in range(0, T, CHUNK):
        end = min(start + CHUNK, T)
        chunk_results = results[start:end]

        # Line 1: actual text
        line_actual = ''
        line_pred = ''
        line_match = ''
        line_bits = ''

        for r in chunk_results:
            ac = char_display(r['actual_byte'])
            pc = char_display(r['pred_byte'])

            # Pad to same width
            w = max(len(ac), len(pc))
            ac = ac.ljust(w)
            pc = pc.ljust(w)

            line_actual += ac
            line_pred += pc
            m = '^' if r['byte_match'] else ' '
            line_match += m.ljust(w)
            # Bit accuracy indicator
            ba = r['bit_acc']
            if ba >= 0.875:
                sym = '@'
            elif ba >= 0.75:
                sym = '#'
            elif ba >= 0.625:
                sym = '+'
            else:
                sym = '.'
            line_bits += sym.ljust(w)

        print(f"\n  pos {start:3d}-{end-1:3d}:")
        print(f"  actual: |{line_actual}|")
        print(f"  predict:|{line_pred}|")
        print(f"  match:   {line_match}")
        print(f"  quality: {line_bits}")

    # Show some interesting hits
    print(f"\n  --- BYTE MATCHES ({byte_matches} total) ---")
    hits = [r for r in results if r['byte_match']]
    if hits:
        for r in hits[:20]:
            c = char_display(r['actual_byte'])
            ctx_start = max(0, r['pos'] - 3)
            ctx_end = min(T, r['pos'] + 4)
            context = ''.join(char_display(results[i]['actual_byte'])
                             for i in range(ctx_start, ctx_end))
            pos_in_ctx = r['pos'] - ctx_start
            # Mark the hit position
            print(f"    pos {r['pos']:3d}: '{c}' in ...{context}...")
        if len(hits) > 20:
            print(f"    ... and {len(hits) - 20} more")
    else:
        print("    (none)")

    # Show confidence distribution
    print(f"\n  --- CONFIDENCE DISTRIBUTION ---")
    for b in range(8):
        confs = [r['prob_bits'][b] for r in results]
        avg_conf = sum(confs) / len(confs)
        # For bits that should be 1, confidence should be high
        # For bits that should be 0, confidence should be low
        correct_conf = []
        for r in results:
            if r['actual_bits'][b] > 0.5:
                correct_conf.append(r['prob_bits'][b])
            else:
                correct_conf.append(1.0 - r['prob_bits'][b])
        avg_correct_conf = sum(correct_conf) / len(correct_conf) if correct_conf else 0.5
        print(f"    bit{b}: acc={per_bit[b]:.0%} {bit_symbol(per_bit[b])}  "
              f"avg_prob={avg_conf:.3f}  correct_conf={avg_correct_conf:.3f}")


def main():
    parser = argparse.ArgumentParser(description='GPU Ant inference demo')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint path (default: auto-find latest)')
    parser.add_argument('--text', type=str, default=None,
                        help='Custom text to test (default: sample from fineweb_edu)')
    parser.add_argument('--samples', type=int, default=3,
                        help='Number of random samples to show (default: 3)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    args = parser.parse_args()

    print("=" * 60)
    print("  GPU ANT INFERENCE DEMO")
    print(f"  Model: D={MODEL_CONFIG['embedding_dim']}, depth={MODEL_CONFIG['depth']}")
    print(f"  Running on CPU (does not touch GPU)")
    print("=" * 60)

    # Find checkpoint
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        # Try latest
        latest = os.path.join(CHECKPOINT_DIR, 'checkpoint_latest.pt')
        if os.path.exists(latest):
            ckpt_path = latest
        else:
            # Find latest draft
            draft_dir = os.path.join(CHECKPOINT_DIR, 'drafts')
            if os.path.isdir(draft_dir):
                drafts = sorted([f for f in os.listdir(draft_dir) if f.endswith('.pt')])
                if drafts:
                    ckpt_path = os.path.join(draft_dir, drafts[-1])

    if ckpt_path is None or not os.path.exists(ckpt_path):
        print(f"  ERROR: No checkpoint found at {ckpt_path}")
        sys.exit(1)

    print(f"  Checkpoint: {ckpt_path}")
    ckpt_size = os.path.getsize(ckpt_path) / 1024 / 1024
    print(f"  Size: {ckpt_size:.0f} MB")

    # Create model on CPU
    print(f"\n  Creating model...")
    model = SwarmByteRingModel(**MODEL_CONFIG)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Load checkpoint
    print(f"  Loading checkpoint...")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    step = ckpt.get('step', '?')
    best_acc = ckpt.get('best_accuracy', 0)
    print(f"  Step: {step}, Best accuracy: {best_acc:.4f}")

    # Load state dict (with tolerance for missing/extra keys)
    sd = ckpt.get('model_state_dict', ckpt.get('weights', {}))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  New params (fresh init): {len(missing)}")
    if unexpected:
        print(f"  Unexpected (ignored): {len(unexpected)}")

    model.eval()
    print(f"  Model loaded and ready.\n")

    # Get text
    if args.text:
        texts = [args.text.encode('utf-8')]
    else:
        # Load corpus
        if not os.path.exists(DATA_PATH):
            print(f"  ERROR: Corpus not found at {DATA_PATH}")
            sys.exit(1)
        with open(DATA_PATH, 'rb') as f:
            corpus = f.read()
        print(f"  Corpus: {len(corpus)/1024/1024:.0f} MB")

        random.seed(args.seed)
        texts = []
        for i in range(args.samples):
            start = random.randint(0, len(corpus) - SEQ_LEN - 10)
            chunk = corpus[start:start + SEQ_LEN + 1]
            texts.append(chunk)

    # Run inference on each sample
    for i, text_bytes in enumerate(texts):
        # Show the actual text being tested
        try:
            preview = text_bytes[:80].decode('utf-8', errors='replace')
        except:
            preview = str(text_bytes[:80])

        title = f"Sample {i+1}/{len(texts)} — \"{preview}...\""
        results = run_inference(model, text_bytes, SEQ_LEN)
        if results:
            print_demo(results, title)

    # Overall summary
    print(f"\n{'='*80}")
    print(f"  SUMMARY")
    print(f"  Model: step {step}, D={MODEL_CONFIG['embedding_dim']}, {n_params:,} params")
    print(f"  Checkpoint: {os.path.basename(ckpt_path)}")
    print(f"  Note: This uses the training-state LCX (scratch buffer).")
    print(f"         Golden LCX lives only in GPU memory during training.")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
