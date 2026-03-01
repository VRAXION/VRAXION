"""Ring Activity Diagnostic — loads a checkpoint and tests if the ring is working.

Tests:
1. Ring norm: does the ring accumulate values during forward pass?
2. Pointer movement: do the pointers actually move?
3. S=0 ablation: accuracy WITH ring vs WITHOUT ring (S=0)
4. Ring slot usage: which slots are active?

Usage:
    python tests/ring_diagnostic.py
    python tests/ring_diagnostic.py --ckpt training_output/ckpt_step_001000.pt
"""

import sys
from pathlib import Path

# Add model + training dirs to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'model'))
sys.path.insert(0, str(ROOT / 'training'))
sys.path.insert(0, str(ROOT / 'datagen'))

import argparse
import numpy as np
import torch
from instnct import INSTNCT


def load_checkpoint(ckpt_path):
    """Load model from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    config = ckpt.get('config', {})
    model_cfg = config.get('model', {})

    # S is a forward() parameter, not constructor
    S = model_cfg.get('S', 0.3)

    model = INSTNCT(
        M=model_cfg.get('M', 512),
        hidden_dim=model_cfg.get('hidden_dim', 2048),
        N=model_cfg.get('N', 2),
        R=model_cfg.get('R', 2),
        slot_dim=model_cfg.get('slot_dim', 64),
        embed_mode=config.get('training', {}).get('embed_mode', True),
        embed_encoding=model_cfg.get('embed_encoding', 'learned'),
        output_encoding=model_cfg.get('output_encoding', 'learned'),
        expert_weighting=model_cfg.get('expert_weighting', True),
        checkpoint_chunks=0,
    )

    model.load_state_dict(ckpt['model_state'])
    model.eval()

    step = ckpt.get('step', '?')
    best_loss = ckpt.get('best_loss', '?')
    print(f'Loaded checkpoint: step={step}, best_loss={best_loss}')
    print(f'Model params: {sum(p.numel() for p in model.parameters()):,}')
    print(f'Config: M={model.M}, N={model.N}, S={S}, embed_encoding={model_cfg.get("embed_encoding", "?")}')

    return model, config, S


def make_echo_batch(batch_size=32, seq_len=64, block=16, repeats=8, aligned=True):
    """Generate a batch of echo data (same format as generate.py).

    aligned=True:  each sample starts at a block boundary (easier, clean measurement)
    aligned=False: random offset into a longer stream (matches training distribution)
    """
    if aligned:
        # Original behavior: always starts at block boundary
        data = np.empty((batch_size, seq_len), dtype=np.uint8)
        mask = np.empty((batch_size, seq_len), dtype=np.uint8)

        for b in range(batch_size):
            pos = 0
            while pos < seq_len:
                block_data = np.random.randint(0, 256, size=block, dtype=np.uint8)
                for r in range(repeats):
                    end = min(pos + block, seq_len)
                    data[b, pos:end] = block_data[:end - pos]
                    mask[b, pos:end] = 0 if r == 0 else 1
                    pos = end
                    if pos >= seq_len:
                        break
    else:
        # Generate a longer stream, then cut a random window (like training)
        stream_len = seq_len * 4  # enough for random offsets
        data = np.empty((batch_size, seq_len), dtype=np.uint8)
        mask = np.empty((batch_size, seq_len), dtype=np.uint8)

        for b in range(batch_size):
            # Build full echo stream
            stream_data = bytearray()
            stream_mask = bytearray()
            while len(stream_data) < stream_len:
                block_data = np.random.randint(0, 256, size=block, dtype=np.uint8).tobytes()
                for r in range(repeats):
                    stream_data.extend(block_data)
                    stream_mask.extend(b'\x00' * block if r == 0 else b'\x01' * block)

            # Random offset (like ByteDataset.sample_batch)
            offset = np.random.randint(0, len(stream_data) - seq_len)
            data[b] = np.frombuffer(stream_data[offset:offset + seq_len], dtype=np.uint8)
            mask[b] = np.frombuffer(stream_mask[offset:offset + seq_len], dtype=np.uint8)

    return data, mask


def compute_accuracy(model, x, y, mask, S, embed_mode=True):
    """Compute masked accuracy on a batch."""
    with torch.no_grad():
        pred, state = model(x, S=S)

    if embed_mode:
        pred_tokens = pred.argmax(dim=-1)
        correct = (pred_tokens == y) & (mask == 1)
        total_masked = mask.sum().item()
        if total_masked == 0:
            return 0.0, state
        acc = correct.sum().item() / total_masked
    else:
        acc = 0.0

    return acc, state


def test_ring_activity(model, x, S):
    """Check ring norm before and after forward pass."""
    B = x.shape[0]
    M, N = model.M, model.N

    with torch.no_grad():
        _, state_after = model(x, S=S)

    ring_norm_after = state_after['ring'].norm().item()
    ptr_after = state_after['ptr']

    print(f'\n=== RING ACTIVITY ===')
    print(f'  Ring norm AFTER forward:   {ring_norm_after:.4f}')
    print(f'  Ring is active:            {"YES" if ring_norm_after > 0.01 else "NO"}')

    print(f'\n=== POINTER MOVEMENT ===')
    for n in range(N):
        ptr_vals = ptr_after[n].numpy()
        print(f'  Expert {n} ptr: min={ptr_vals.min():.2f} max={ptr_vals.max():.2f} mean={ptr_vals.mean():.2f} std={ptr_vals.std():.2f}')
        unique = len(np.unique(np.round(ptr_vals, 1)))
        print(f'  Expert {n} unique positions: {unique}/{B}')

    return state_after


def test_s_ablation(model, S, embed_mode=True, n_batches=10):
    """Compare accuracy WITH ring (normal S) vs WITHOUT ring (S=0)."""
    # --- Normal S ---
    accs_normal = []
    for _ in range(n_batches):
        batch_data, batch_mask = make_echo_batch(batch_size=64, seq_len=64)
        x = torch.from_numpy(batch_data[:, :-1]).long()
        y = torch.from_numpy(batch_data[:, 1:]).long()
        m = torch.from_numpy(batch_mask[:, 1:])
        acc, _ = compute_accuracy(model, x, y, m, S, embed_mode)
        accs_normal.append(acc)

    avg_normal = sum(accs_normal) / len(accs_normal)

    # --- S=0 (ring read disabled) ---
    accs_ablated = []
    for _ in range(n_batches):
        batch_data, batch_mask = make_echo_batch(batch_size=64, seq_len=64)
        x = torch.from_numpy(batch_data[:, :-1]).long()
        y = torch.from_numpy(batch_data[:, 1:]).long()
        m = torch.from_numpy(batch_mask[:, 1:])
        acc, _ = compute_accuracy(model, x, y, m, 0.0, embed_mode)
        accs_ablated.append(acc)

    avg_ablated = sum(accs_ablated) / len(accs_ablated)

    print(f'\n=== S ABLATION (ring contribution) ===')
    print(f'  Accuracy WITH ring (S={S}):  {avg_normal*100:.2f}%')
    print(f'  Accuracy WITHOUT ring (S=0):    {avg_ablated*100:.2f}%')
    print(f'  Ring contribution:              {(avg_normal - avg_ablated)*100:+.2f}%')

    if avg_normal > avg_ablated + 0.01:
        print(f'  VERDICT: RING IS CONTRIBUTING (adds {(avg_normal - avg_ablated)*100:.1f}% accuracy)')
    elif avg_ablated > avg_normal + 0.01:
        print(f'  VERDICT: RING IS HURTING (removes {(avg_ablated - avg_normal)*100:.1f}% accuracy)')
    else:
        print(f'  VERDICT: RING HAS NO EFFECT (difference < 1%)')

    return avg_normal, avg_ablated


def test_ring_per_slot(model, x, S):
    """Check which ring slots are most active."""
    with torch.no_grad():
        _, state = model(x, S=S)

    ring = state['ring']  # (B, M, slot_dim)
    slot_norms = ring.norm(dim=-1).mean(dim=0)  # (M,)

    print(f'\n=== RING SLOT USAGE ===')
    print(f'  Total slots: {model.M}')
    active = (slot_norms > 0.01).sum().item()
    print(f'  Active slots (norm > 0.01): {active}/{model.M} ({100*active/model.M:.0f}%)')
    print(f'  Mean slot norm: {slot_norms.mean():.4f}')
    print(f'  Max slot norm:  {slot_norms.max():.4f} (slot {slot_norms.argmax().item()})')
    print(f'  Min slot norm:  {slot_norms.min():.4f}')

    top_vals, top_idx = slot_norms.topk(min(10, model.M))
    print(f'  Top 10 slots: {[f"#{i.item()}={v:.3f}" for i, v in zip(top_idx, top_vals)]}')


def main():
    parser = argparse.ArgumentParser(description='Ring Activity Diagnostic')
    parser.add_argument('--ckpt', default=str(ROOT / 'training_output' / 'ckpt_latest.pt'),
                        help='Checkpoint path')
    args = parser.parse_args()

    print('=' * 60)
    print('VRAXION v4 — Ring Activity Diagnostic')
    print('=' * 60)

    model, config, S = load_checkpoint(args.ckpt)
    embed_mode = config.get('training', {}).get('embed_mode', True)

    # Test 1: Baseline accuracy — ALIGNED (block-boundary start)
    print(f'\nGenerating echo test data (aligned) ...')
    data, mask = make_echo_batch(batch_size=64, seq_len=64, aligned=True)
    x = torch.from_numpy(data[:, :-1]).long()
    y = torch.from_numpy(data[:, 1:]).long()
    m = torch.from_numpy(mask[:, 1:])

    acc_aligned, _ = compute_accuracy(model, x, y, m, S, embed_mode)
    print(f'\n=== BASELINE ACCURACY (aligned) ===')
    print(f'  Echo accuracy (1 batch, 64 samples): {acc_aligned*100:.2f}%')

    # Test 1b: Baseline accuracy — RANDOM OFFSET (matches training distribution)
    print(f'\nGenerating echo test data (random offset) ...')
    data_r, mask_r = make_echo_batch(batch_size=64, seq_len=64, aligned=False)
    xr = torch.from_numpy(data_r[:, :-1]).long()
    yr = torch.from_numpy(data_r[:, 1:]).long()
    mr = torch.from_numpy(mask_r[:, 1:])

    acc_random, _ = compute_accuracy(model, xr, yr, mr, S, embed_mode)
    print(f'\n=== BASELINE ACCURACY (random offset) ===')
    print(f'  Echo accuracy (1 batch, 64 samples): {acc_random*100:.2f}%')
    print(f'  Alignment gap: {(acc_aligned - acc_random)*100:+.2f}%')

    # Test 2: Ring activity
    test_ring_activity(model, x, S)

    # Test 3: Ring slot usage
    test_ring_per_slot(model, x, S)

    # Test 4: S ablation — THE KEY TEST
    test_s_ablation(model, S, embed_mode)

    print(f'\n{"=" * 60}')
    print('Diagnostic complete.')


if __name__ == '__main__':
    main()
