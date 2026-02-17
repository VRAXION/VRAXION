"""
Test a swarm model configuration on multi-task benchmark.

Usage:
    python test_swarm_config.py --embedding 64 --num_beings 2 --depth 2

Logs everything (loss, accuracy, jump gates, swarm metrics) for dashboard viewing.
"""

import math
import os
import torch
# Use 90% of CPU cores for training
torch.set_num_threads(max(1, int(os.cpu_count() * 0.9)))
import torch.nn as nn
import sys
import time
import json
import argparse
from pathlib import Path
import random
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))

from swarm_model import SwarmByteRingModel, fibonacci_split
from byte_data import byte_accuracy, bit_accuracy
from traindat_loader import TraindatLoader, GRAY_ENCODE, gray_scalar_to_byte
from live_controls import read_controls, apply_controls, write_default_controls, _set_being_grad
import influx_writer


def int_to_bits(x, num_bits=8):
    """Convert integer to num_bits-wide bit tensor."""
    bits = []
    for i in range(num_bits):
        bits.append((x >> i) & 1)
    return torch.tensor(bits, dtype=torch.float32)


def load_text_corpus(path: str) -> bytes:
    """Load a text file as raw bytes."""
    with open(path, 'rb') as f:
        return f.read()


def generate_text_batch(corpus: bytes, n_samples: int, seq_len: int = 16, num_bits: int = 64, seed=None):
    """
    Generate next-byte prediction batch from text corpus.

    Each sample is a random chunk of (seq_len+1) bytes from the corpus.
    Input = bytes[0:seq_len], Target = bytes[1:seq_len+1] (shifted by 1 byte).

    Each byte is expanded to 8 bits. If num_bits=64, each "position" is 8 bytes = 64 bits.

    Args:
        corpus: Raw bytes of text.
        n_samples: Batch size.
        seq_len: Number of positions per sample.
        num_bits: Bits per position (must be multiple of 8).
        seed: Random seed.

    Returns:
        x: [n_samples, seq_len, num_bits] input bits
        y: [n_samples, seq_len, num_bits] target bits (next chunk)
    """
    if seed is not None:
        random.seed(seed)

    bytes_per_pos = num_bits // 8  # 64 bits = 8 bytes per position
    chunk_len = (seq_len + 1) * bytes_per_pos  # +1 for the shifted target
    max_start = len(corpus) - chunk_len - bytes_per_pos

    x = torch.zeros(n_samples, seq_len, num_bits)
    y = torch.zeros(n_samples, seq_len, num_bits)

    for i in range(n_samples):
        start = random.randint(0, max_start)
        chunk = corpus[start:start + chunk_len + bytes_per_pos]

        for t in range(seq_len):
            # Input: bytes_per_pos bytes starting at offset t*bytes_per_pos
            offset = t * bytes_per_pos
            for b in range(bytes_per_pos):
                byte_val = chunk[offset + b]
                for bit in range(8):
                    x[i, t, b * 8 + bit] = float((byte_val >> bit) & 1)

            # Target: shifted by 1 byte (next-byte prediction)
            target_offset = offset + bytes_per_pos
            for b in range(bytes_per_pos):
                byte_val = chunk[target_offset + b]
                for bit in range(8):
                    y[i, t, b * 8 + bit] = float((byte_val >> bit) & 1)

    return x, y


# Effort-level ranges: maps effort code -> (min_tt, max_tt)
# Gaps at 3 and 7 create clean separation between effort bands.
EFFORT_RANGES = {
    0: (0, 2),    # fast: 0-2 think ticks
    1: (4, 6),    # medium: 4-6 think ticks
    2: (8, 10),   # slow/deliberate: 8-10 think ticks
}


def sample_effort(seed_val=None):
    """Sample effort level and think_ticks from effort ranges.
    Returns (effort_level, think_ticks)."""
    if seed_val is not None:
        random.seed(seed_val)
    effort = random.choice([0, 1, 2])
    lo, hi = EFFORT_RANGES[effort]
    tt = random.randint(lo, hi)
    return effort, tt


def generate_multitask_batch(n_samples, seq_len=16, max_value=100, seed=None, num_bits=8, task='mixed',
                             effort_level=None):
    """Generate batch with mixed operations (or single op if task != 'mixed').

    Args:
        effort_level: If not None, encode effort (0=fast, 1=medium, 2=slow) in bits 4-5
                      of position 2 (op_code position). Bits 0-3 = operation, bits 4-5 = effort.
    """
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    max_val = min(max_value, 2 ** num_bits - 1)

    op_codes = {
        'add': int_to_bits(1 << 0, num_bits),
        'and': int_to_bits(1 << 1, num_bits),
        'or':  int_to_bits(1 << 2, num_bits),
        'xor': int_to_bits(1 << 3, num_bits),
    }

    x_batch = []
    y_batch = []
    op_indices = []

    for _ in range(n_samples):
        op_name = random.choice(['add', 'and', 'or', 'xor']) if task == 'mixed' else task
        op_idx = {'add': 0, 'and': 1, 'or': 2, 'xor': 3}[op_name]
        op_indices.append(op_idx)

        a = random.randint(0, max_val)
        b = random.randint(0, max_val)

        if op_name == 'add':
            result = (a + b) % (2 ** num_bits)
        elif op_name == 'and':
            result = a & b
        elif op_name == 'or':
            result = a | b
        else:
            result = a ^ b

        x_seq = torch.zeros(seq_len, num_bits)
        x_seq[0, :] = int_to_bits(a, num_bits)
        x_seq[1, :] = int_to_bits(b, num_bits)
        x_seq[2, :] = op_codes[op_name]

        # Encode effort level in spare bits 4-5 of position 2
        if effort_level is not None and num_bits >= 6:
            x_seq[2, 4] = float(effort_level & 1)       # bit 4
            x_seq[2, 5] = float((effort_level >> 1) & 1) # bit 5

        y_seq = torch.zeros(seq_len, num_bits)
        y_seq[0, :] = int_to_bits(a, num_bits)
        y_seq[1, :] = int_to_bits(b, num_bits)
        y_seq[2, :] = int_to_bits(result, num_bits)

        x_batch.append(x_seq)
        y_batch.append(y_seq)

    return torch.stack(x_batch), torch.stack(y_batch), torch.tensor(op_indices)


# --- Gray code decode helpers ---
_GRAY_ENCODE_TENSOR = torch.tensor(GRAY_ENCODE, dtype=torch.long)


def _output_to_bytes(output_slice):
    """Convert model logits [..., num_bits] to predicted byte values [..., num_bits]."""
    probs = torch.sigmoid(output_slice)
    positions = torch.round(probs * 256.0 - 0.5).clamp(0, 255).long()
    return _GRAY_ENCODE_TENSOR[positions]


def _target_to_bytes(target_slice):
    """Convert Gray scalar targets [..., num_bits] to byte values [..., num_bits]."""
    positions = torch.round(target_slice * 256.0 - 0.5).clamp(0, 255).long()
    return _GRAY_ENCODE_TENSOR[positions]


def position_accuracy(output, target, position):
    """All channels decoded to correct byte value."""
    pred_bytes = _output_to_bytes(output[:, position, :])
    tgt_bytes = _target_to_bytes(target[:, position, :])
    return (pred_bytes == tgt_bytes).all(dim=-1).float().mean().item()


def per_operation_accuracy(output, target, op_indices):
    """Calculate accuracy for each operation separately."""
    accuracies = {}
    for op_idx, op_name in enumerate(['add', 'and', 'or', 'xor']):
        mask = (op_indices == op_idx)
        if mask.sum() > 0:
            op_output = output[mask]
            op_target = target[mask]
            accuracies[op_name] = position_accuracy(op_output, op_target, min(2, op_output.size(1) - 1))
        else:
            accuracies[op_name] = 0.0
    return accuracies


def byte_match_accuracy(output, target, position):
    """All channels decoded to correct byte value (same as position_accuracy)."""
    return position_accuracy(output, target, position)


def bit_accuracy_at_position(output, target, position):
    """Per-channel byte accuracy (fraction of channels with correct byte)."""
    pred_bytes = _output_to_bytes(output[:, position, :])
    tgt_bytes = _target_to_bytes(target[:, position, :])
    return (pred_bytes == tgt_bytes).float().mean().item()


def per_bit_accuracy_at_position(output, target, position):
    """Per-channel byte accuracy (list of num_bits values)."""
    pred_bytes = _output_to_bytes(output[:, position, :])
    tgt_bytes = _target_to_bytes(target[:, position, :])
    per_ch = (pred_bytes == tgt_bytes).float().mean(dim=0)
    return per_ch.tolist()


def hamming_distance_at_position(output, target, position):
    """Mean Gray distance (0-255 scale)."""
    pred_bytes = _output_to_bytes(output[:, position, :])
    tgt_bytes = _target_to_bytes(target[:, position, :])
    return (pred_bytes - tgt_bytes).abs().float().mean().item()


def oracle_best_of_n_accuracy(being_outputs, target, position):
    """
    Oracle accuracy: best-of-N selection per sample.
    Upper bound on ensemble potential. Uses Gray code byte decoding.

    Args:
        being_outputs: [num_beings, T, B, 8] ALL timestep outputs from each being
        target: [B, T, 8] ground truth (Gray scalars)
        position: timestep position to evaluate

    Returns:
        Oracle accuracy (best being per sample)
    """
    being_outputs_at_pos = being_outputs[:, position, :, :]  # [num_beings, B, 8]
    tgt_bytes = _target_to_bytes(target[:, position, :])  # [B, 8]

    num_beings = being_outputs_at_pos.size(0)
    B = being_outputs_at_pos.size(1)

    oracle_correct = 0
    for b in range(B):
        for being_idx in range(num_beings):
            pred_bytes = _output_to_bytes(being_outputs_at_pos[being_idx, b:b+1, :])
            if (pred_bytes[0] == tgt_bytes[b]).all().item():
                oracle_correct += 1
                break
    return oracle_correct / B


def bit_oracle_accuracy(being_outputs, target, position):
    """
    Channel-oracle: for each channel, did ANY being decode the correct byte?
    Upper bound for what a perfect channel-level selector could achieve.
    """
    being_at_pos = being_outputs[:, position, :, :]  # [num_beings, B, 8]
    tgt_bytes = _target_to_bytes(target[:, position, :])  # [B, 8]

    any_correct = torch.zeros(tgt_bytes.shape, dtype=torch.bool)
    for i in range(being_at_pos.size(0)):
        pred_bytes = _output_to_bytes(being_at_pos[i])
        any_correct |= (pred_bytes == tgt_bytes)
    return any_correct.float().mean().item()


def compute_specialization(being_outputs, target, op_indices, position):
    """
    Compute specialization score: std of per-being per-operation accuracies.

    High score = beings specialize (being₀ good at OR, being₁ good at AND)
    Low score = beings redundant (all similar performance)

    Args:
        being_outputs: [num_beings, T, B, 8] ALL timestep outputs from each being
        target: [B, T, 8] ground truth
        op_indices: [B] operation indices
        position: timestep position to evaluate

    Returns:
        Specialization score (average std across operations)
    """
    # Extract position slice: [num_beings, B, 8]
    being_outputs_at_pos = being_outputs[:, position, :, :]

    num_beings = being_outputs_at_pos.size(0)

    # being_accs_matrix[op_idx][being_idx] = accuracy
    being_accs_matrix = [[0.0] * num_beings for _ in range(4)]

    for op_idx, op_name in enumerate(['add', 'and', 'or', 'xor']):
        mask = (op_indices == op_idx)
        if mask.sum() > 0:
            for being_idx in range(num_beings):
                being_output = being_outputs_at_pos[being_idx][mask]  # [masked_B, 8]
                op_target = target[mask]  # [masked_B, T, 8]
                pred_bytes = _output_to_bytes(being_output)
                tgt_bytes = _target_to_bytes(op_target[:, position, :])
                matches = (pred_bytes == tgt_bytes).all(dim=-1).float()
                being_accs_matrix[op_idx][being_idx] = matches.mean().item()

    # Compute std across beings for each operation
    specialization_scores = []
    for op_accs in being_accs_matrix:
        if any(acc > 0 for acc in op_accs):  # Only if we have data for this op
            std = torch.tensor(op_accs).std().item()
            specialization_scores.append(std)

    return sum(specialization_scores) / len(specialization_scores) if specialization_scores else 0.0


def evaluate_metrics(output, stats, y, train_loss, num_beings, num_bits, n_contributing=None, receptive_masks=None):
    """Compute all metrics from pre-computed output+stats on CPU. Called every step."""
    eval_pos = min(2, output.size(1) - 1)

    overall_acc = position_accuracy(output, y, eval_pos)
    bit_acc = bit_accuracy_at_position(output, y, eval_pos)
    byte_match = byte_match_accuracy(output, y, eval_pos)
    hamming = hamming_distance_at_position(output, y, eval_pos)
    per_bit_accs = per_bit_accuracy_at_position(output, y, eval_pos)

    # All-position average bit accuracy (reveals learning across ALL positions,
    # not just position 2 which may be random for some tasks like echo)
    T = output.size(1)
    avg_bit_acc = sum(bit_accuracy_at_position(output, y, p) for p in range(T)) / T

    # Per-being metrics
    being_outputs = stats['being_outputs']
    being_accs = []
    being_masked_accs = []
    for i in range(num_beings):
        being_output_transposed = being_outputs[i].transpose(0, 1)
        being_acc = position_accuracy(being_output_transposed, y, eval_pos)
        being_accs.append(being_acc)

        # Mask-aware accuracy: only evaluate covered channels
        if receptive_masks is not None and i < len(receptive_masks):
            mask = receptive_masks[i]  # [num_bits]
            covered = mask.bool()
            if covered.any():
                pred_bytes = _output_to_bytes(being_outputs[i][eval_pos])  # [B, num_bits]
                tgt_bytes = _target_to_bytes(y[:, eval_pos, :])  # [B, num_bits]
                masked_acc = (pred_bytes[:, covered] == tgt_bytes[:, covered]).float().mean().item()
            else:
                masked_acc = 0.0
            being_masked_accs.append(masked_acc)
        else:
            being_masked_accs.append(being_acc)

    # Ensemble diagnostics
    best_individual = max(being_accs) if being_accs else 0.0
    ensemble_benefit = overall_acc - best_individual
    being_outputs_stacked = torch.stack(being_outputs)  # [num_beings, T, B, bits]
    oracle_acc = oracle_best_of_n_accuracy(being_outputs_stacked, y, eval_pos)
    bit_oracle_acc = bit_oracle_accuracy(being_outputs_stacked, y, eval_pos)

    # Spatial metrics
    pointer_positions = stats['pointer_positions_all']
    coverage = len(set(pointer_positions)) / 64.0
    position_counts = Counter(pointer_positions)
    denom = n_contributing if n_contributing else num_beings
    clustering = max(position_counts.values()) / max(denom, 1) if position_counts else 0.0
    circular_spread = stats['circular_spread']

    jump_rates = stats['jump_rates_per_being']

    mask_stats = {}
    if 'min_bit_coverage' in stats:
        mask_stats['min_cov'] = stats['min_bit_coverage']
        mask_stats['max_cov'] = stats['max_bit_coverage']
        mask_stats['mask_div'] = stats['mask_diversity']

    activation_ratio = stats.get('activation_ratio', None)

    return {
        'eval_loss': train_loss, 'overall_acc': overall_acc, 'bit_acc': bit_acc,
        'avg_bit_acc': avg_bit_acc,
        'byte_match': byte_match, 'hamming': hamming, 'per_bit_accs': per_bit_accs,
        'being_accs': being_accs, 'being_masked_accs': being_masked_accs,
        'oracle_acc': oracle_acc, 'bit_oracle_acc': bit_oracle_acc,
        'ensemble_benefit': ensemble_benefit, 'circular_spread': circular_spread,
        'coverage': coverage, 'clustering': clustering, 'jump_rates': jump_rates,
        'mask_stats': mask_stats, 'activation_ratio': activation_ratio,
    }


def format_metrics_line(step, train_loss, step_time, metrics):
    """Format step + metrics into dashboard-compatible log line."""
    m = metrics
    line = (
        f"step {step} | loss {train_loss:.6f} | "
        f"overall={m['overall_acc']:.4f} bit_acc={m['bit_acc']:.4f} "
        f"avg_bit_acc={m.get('avg_bit_acc', 0):.4f} "
        f"byte_match={m['byte_match']:.4f} hamming={m['hamming']:.4f} | "
    )
    for i, ba in enumerate(m['being_accs']):
        line += f"being_{i}={ba:.4f} "
    if 'being_masked_accs' in m:
        for i, bma in enumerate(m['being_masked_accs']):
            line += f"masked_{i}={bma:.4f} "
    line += (
        f"oracle={m['oracle_acc']:.4f} bit_oracle={m['bit_oracle_acc']:.4f} "
        f"ensemble_benefit={m['ensemble_benefit']:+.4f} | "
        f"circular_spread={m['circular_spread']:.4f} "
        f"coverage={m['coverage']:.4f} clustering={m['clustering']:.4f} | "
    )
    for i, jr in enumerate(m['jump_rates']):
        line += f"jump_{i}={jr:.4f} "
    line += f"specialization=0.0000"
    per_bit_str = " ".join([f"bit{i}={a:.4f}" for i, a in enumerate(m['per_bit_accs'])])
    # Visual bit bar: average brightness per bit, dark=wrong light=correct
    n_bits = len(m['per_bit_accs'])
    avg_acc = sum(m['per_bit_accs']) / max(n_bits, 1)
    shades = " .:-=+*#%@"  # 10 levels from dark to bright
    bit_bar = ""
    for a in m['per_bit_accs']:
        idx = min(int(a * (len(shades) - 1)), len(shades) - 1)
        bit_bar += shades[idx]
    avg_idx = min(int(avg_acc * (len(shades) - 1)), len(shades) - 1)
    line += f" | bits[{bit_bar}]{shades[avg_idx]} avg={avg_acc:.3f}"
    line += f" | {per_bit_str}"
    if m['mask_stats']:
        ms = m['mask_stats']
        line += f" | min_cov={ms['min_cov']} max_cov={ms['max_cov']} mask_div={ms['mask_div']:.4f}"
    if m['activation_ratio'] is not None:
        line += f" | active_ratio={m['activation_ratio']:.3f}"
    line += f" | s_per_step={step_time:.3f}"
    return line


def save_checkpoint(model, optimizer, step, best_acc, checkpoint_dir, is_best=False, config=None,
                    milestone_every=5000):
    """Save training checkpoint with architecture config for CPU eval worker.

    Always overwrites checkpoint_latest.pt (rolling save).
    Every milestone_every steps, also saves a permanent checkpoint_step_N.pt.
    """
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_accuracy': best_acc,
        'num_bits': getattr(model, 'num_bits', 8),
        'bits_per_being': getattr(model, 'bits_per_being', 0),
        'min_coverage': getattr(model, 'min_coverage', 0),
    }
    if config is not None:
        checkpoint['config'] = config
    if hasattr(model, 'being_states'):
        checkpoint['being_states'] = model.being_states.copy()

    # Always overwrite latest (rolling save)
    latest_path = Path(checkpoint_dir) / 'checkpoint_latest.pt'
    torch.save(checkpoint, latest_path)
    print(f"  [SAVE] Checkpoint saved: {latest_path} (step {step})")

    # Permanent milestone at every milestone_every steps
    if step > 0 and step % milestone_every == 0:
        milestone_path = Path(checkpoint_dir) / f'checkpoint_step_{step}.pt'
        torch.save(checkpoint, milestone_path)
        print(f"  [MILESTONE] Permanent checkpoint: {milestone_path}")

    # Save best model separately
    if is_best:
        best_path = Path(checkpoint_dir) / 'best_model.pt'
        torch.save(checkpoint, best_path)
        print(f"  [BEST] Best model saved: {best_path}")


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Load training checkpoint."""
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    # RGB LCX migration: old [lcx_size] -> new [3, lcx_size]
    if 'lcx' in checkpoint['model_state_dict']:
        old_lcx = checkpoint['model_state_dict']['lcx']
        if old_lcx is not None and old_lcx.dim() == 1 and model.lcx is not None and model.lcx.dim() == 2:
            print(f"  [LOAD] Migrating LCX: [{old_lcx.shape[0]}] -> [3, {old_lcx.shape[0]}] (old -> R, G/B = 0)")
            new_lcx = torch.zeros_like(model.lcx)
            new_lcx[0] = old_lcx
            checkpoint['model_state_dict']['lcx'] = new_lcx

    # Hash LCX migration: drop dense LCX params when switching to hash mode (and vice versa)
    sd = checkpoint['model_state_dict']
    if model._lcx_hash_mode:
        # Model is hash mode — remove dense LCX keys that would cause shape mismatch
        for key in ['lcx', 'lcx_propose.weight', 'lcx_propose.bias', 'lcx_gate.weight', 'lcx_gate.bias']:
            if key in sd:
                print(f"  [LOAD] Dropping dense LCX param '{key}' (switched to hash mode)")
                del sd[key]
        # Remove old input_proj if shape doesn't match (dense used num_bits², hash uses num_bits)
        if 'input_proj.weight' in sd and model.input_proj is not None:
            if sd['input_proj.weight'].shape != model.input_proj.weight.shape:
                print(f"  [LOAD] Dropping input_proj (shape mismatch: {sd['input_proj.weight'].shape} vs {model.input_proj.weight.shape})")
                del sd['input_proj.weight']
                if 'input_proj.bias' in sd:
                    del sd['input_proj.bias']
        # Zoom LCX migration: old 3-channel [3, S, D] -> per-level buffers (reinit)
        if 'lcx_keys' in sd and sd['lcx_keys'].dim() == 3:
            print(f"  [LOAD] Migrating 3-channel LCX -> zoom levels (reinit)")
            for key in list(sd.keys()):
                if key.startswith('lcx_keys') or key.startswith('lcx_values'):
                    del sd[key]
        # Zoom LCX migration: old single-buffer lcx_keys [S, D] -> per-level (reinit)
        elif 'lcx_keys' in sd and sd['lcx_keys'].dim() == 2:
            print(f"  [LOAD] Migrating single-buffer LCX -> zoom levels (reinit)")
            for key in list(sd.keys()):
                if key == 'lcx_keys' or key == 'lcx_values':
                    del sd[key]
        # Flat LCX migration: drop per-level buffers with mismatched shapes (slot count or key_dim changed)
        for key in list(sd.keys()):
            if key.startswith(('lcx_keys_', 'lcx_values_')):
                model_buf = getattr(model, key, None)
                if model_buf is not None and sd[key].shape != model_buf.shape:
                    print(f"  [LOAD] Dropping LCX buffer '{key}' (shape: {sd[key].shape} vs {model_buf.shape})")
                    del sd[key]
        # Drop lcx_route_query if key_dim changed
        if 'lcx_route_query.weight' in sd and model.lcx_route_query is not None:
            if sd['lcx_route_query.weight'].shape != model.lcx_route_query.weight.shape:
                print(f"  [LOAD] Dropping lcx_route_query (key_dim changed)")
                del sd['lcx_route_query.weight']
                if 'lcx_route_query.bias' in sd:
                    del sd['lcx_route_query.bias']
    elif not model._lcx_hash_mode and model.lcx is not None:
        # Model is dense mode — remove hash LCX keys if present
        for key in list(sd.keys()):
            if key.startswith('lcx_keys') or key.startswith('lcx_values') or \
               key in ['lcx_route_query.weight', 'lcx_route_query.bias',
                        'lcx_write_gate.weight', 'lcx_write_gate.bias']:
                print(f"  [LOAD] Dropping hash LCX param '{key}' (switched to dense mode)")
                del sd[key]

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  [LOAD] New params (initialized fresh): {missing}")
    if unexpected:
        print(f"  [LOAD] Unexpected params (ignored): {unexpected}")

    if optimizer is not None and not missing and not unexpected:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    elif optimizer is not None:
        print(f"  [LOAD] Skipping optimizer state (model architecture changed)")

    if 'being_states' in checkpoint:
        model.being_states = checkpoint['being_states']

    step = checkpoint['step']
    best_acc = checkpoint.get('best_accuracy', 0.0)

    print(f"  [LOAD] Resumed from checkpoint: {checkpoint_path}")
    print(f"        Step: {step}, Best accuracy: {best_acc:.4f}")

    return step, best_acc


def main():
    parser = argparse.ArgumentParser(description='Test swarm model config')
    parser.add_argument('--embedding', type=int, default=64, help='Embedding dimension (default: 64)')
    parser.add_argument('--depth', type=int, default=2, help='Number of layers (default: 2)')
    parser.add_argument('--num_beings', type=int, default=3, help='Number of beings in swarm (default: 3, voting ensemble)')
    parser.add_argument('--steps', type=int, default=100000, help='Training steps (default: 100000, effectively unlimited - stop manually)')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/swarm', help='Directory to save checkpoints')
    parser.add_argument('--checkpoint_every', type=int, default=10, help='Save checkpoint every N steps (default: 10)')
    parser.add_argument('--eval_every', type=int, default=10, help='Eval interval in steps (default: 10)')
    parser.add_argument('--eval_samples', type=int, default=10, help='Number of eval samples (default: 10)')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint file')
    parser.add_argument('--combiner', type=str, default='mean', choices=['mean', 'ring_attention', 'masked'],
                        help='Combiner mode: mean | ring_attention | masked (with receptive fields)')
    parser.add_argument('--entropy_weight', type=float, default=0.01,
                        help='Weight for gate entropy regularizer (ring_attention only, default: 0.01)')
    parser.add_argument('--freeze_gate_steps', type=int, default=0,
                        help='Freeze being params for N steps, train only gate (ring_attention warm start)')
    parser.add_argument('--num_bits', type=int, default=8,
                        help='Input/output bit width (default: 8, use 64 for 8-byte mode)')
    parser.add_argument('--bits_per_being', type=int, default=0,
                        help='Bits per being receptive field (0=all bits). Auto-enables masked combiner.')
    parser.add_argument('--min_coverage', type=int, default=2,
                        help='Minimum number of beings covering each bit (default: 2)')
    parser.add_argument('--mask_seed', type=int, default=42,
                        help='Random seed for receptive field mask generation')
    parser.add_argument('--fibonacci', action='store_true',
                        help='Use Fibonacci-halving K schedule (heterogeneous receptive fields)')
    parser.add_argument('--combinatorial', action='store_true',
                        help='Combinatorial mask placement (greedy pair-diversity, uniform K)')
    parser.add_argument('--use_lcx', action='store_true', default=True,
                        help='Use LCX (per-slot scratchpad) instead of legacy GEM (default: True)')
    parser.add_argument('--no_lcx', action='store_false', dest='use_lcx',
                        help='Disable LCX, fall back to legacy GEM')
    parser.add_argument('--slots_per_being', type=int, default=1,
                        help='LCX slots each being can write (-1 = all slots / giant ant, 1 = flowchart spec, K = K phi-stride slots)')
    parser.add_argument('--lcx_mode', type=str, default='dense', choices=['dense', 'hash'],
                        help='LCX memory mode: dense (legacy pixel grid) or hash (sparse content-addressed slots)')
    parser.add_argument('--lcx_num_slots', type=int, default=618,
                        help='Base memory slots for hash LCX (golden ratio: 618, scales 10x per level)')
    parser.add_argument('--lcx_key_dim', type=int, default=618,
                        help='Routing key dimension for hash LCX (default: 618, matches L0 slot count)')
    parser.add_argument('--lcx_top_k', type=int, default=4,
                        help='Number of slots to read/write per step in hash LCX (default: 4)')
    parser.add_argument('--lcx_num_levels', type=int, default=3,
                        help='Number of zoom levels for hash LCX (default: 3)')
    parser.add_argument('--lcx_level_slots', type=str, default=None,
                        help='Comma-separated slot counts per level, e.g. "256,1024,4096"')
    parser.add_argument('--text', type=str, default=None,
                        help='Path to text file for byte-level language modeling (overrides math tasks)')
    parser.add_argument('--think_ticks', type=int, default=0,
                        help='Extra ring ticks without input (beings read each other, then output). 0=reflex mode.')
    parser.add_argument('--effort_mode', action='store_true',
                        help='Enable effort-level training: randomly sample think_ticks from 3 ranges '
                             '(fast=0-1, medium=2-4, slow=6-12) and encode effort in input bits 4-5')
    parser.add_argument('--temporal_fibonacci', action='store_true',
                        help='Enable temporal Fibonacci tick scheduling (beings fire at Fibonacci-spaced intervals)')
    parser.add_argument('--capacity_fibonacci', action='store_true',
                        help='Per-being hidden dims (whale=max_hidden, ant=min_hidden)')
    parser.add_argument('--full_view', action='store_true',
                        help='Each being sees all bits at different resolutions (no masks)')
    parser.add_argument('--max_hidden', type=int, default=4096,
                        help='Maximum hidden dim for largest being (default: 4096)')
    parser.add_argument('--min_hidden', type=int, default=128,
                        help='Minimum hidden dim floor (default: 128)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: auto | cpu | cuda (default: auto-detect GPU)')
    parser.add_argument('--compile', action='store_true',
                        help='torch.compile() the model for fused GPU kernels')
    parser.add_argument('--amp', action='store_true',
                        help='BF16 mixed precision (halves activation VRAM, uses tensor cores)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Training batch size (default: 32)')
    parser.add_argument('--seq_len', type=int, default=16,
                        help='Sequence length / positions per sample (default: 16)')
    parser.add_argument('--lr', type=float, default=0.0003,
                        help='Learning rate (default: 0.0003)')
    parser.add_argument('--warmup_steps', type=int, default=100,
                        help='LR warmup steps (linear ramp from 0 to lr, default: 100)')
    parser.add_argument('--lr_min', type=float, default=1e-5,
                        help='Minimum LR for cosine decay (default: 1e-5)')
    parser.add_argument('--memory_size', type=int, default=0,
                        help='Ring memory positions (default: 0 = match embedding dim)')
    parser.add_argument('--jump_bias', type=float, default=0.5,
                        help='Initial jump gate bias (default: 0.5 -> 62%% jump). Use -2.0 for ~12%% jump.')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Directory of .traindat files (overrides --text and math modes)')
    parser.add_argument('--data_weights', type=str, default=None,
                        help='JSON string of file weights, e.g. {"shakespeare.traindat": 3.0, "xor.traindat": 1.0}')
    parser.add_argument('--controls_every', type=int, default=1,
                        help='Poll controls.json every N steps (default: 1)')
    parser.add_argument('--controls_path', type=str, default='logs/swarm/controls.json',
                        help='Path to live controls JSON file')
    parser.add_argument('--active_beings', type=str, default='all',
                        help='Comma-separated being indices to train (e.g. "6" or "5,6"). "all" trains everything.')
    parser.add_argument('--grad_accum', type=int, default=1,
                        help='Gradient accumulation steps (default: 1, no accumulation)')
    parser.add_argument('--attention_radius', type=int, default=2,
                        help='Gaussian attention radius for ring memory (default: 2)')
    parser.add_argument('--num_pointers', type=int, default=1,
                        help='Number of ring pointers per being (default: 1, use 2 for dual coverage)')
    parser.add_argument('--start_lcx_off', action='store_true',
                        help='Start with LCX disabled in controls (INFANT stage). '
                             'Model is still built with LCX buffers for later activation.')
    parser.add_argument('--effort', type=str, default='Beta',
                        help='Starting effort tier: Alpha/Beta/Gamma/Delta/Epsilon/Zeta (default: Beta). '
                             'Auto-sets think_ticks, use_lcx, batch_size from tier definition.')
    parser.add_argument('--task', type=str, default='mixed',
                        choices=['mixed', 'xor', 'add', 'and', 'or'],
                        help='Which operation to train on (default: mixed = all ops)')
    parser.add_argument('--diagnostic', type=str, default='none',
                        choices=['none', 'gradient', 'saturation', 'ring', 'all'],
                        help='Run architecture diagnostics (default: none)')
    args = parser.parse_args()

    # Auto-switch combiner to 'masked' if bits_per_being specified
    if args.bits_per_being > 0 and args.combiner == 'mean':
        args.combiner = 'masked'

    embedding_dim = args.embedding
    depth = args.depth
    num_beings = args.num_beings
    num_steps = args.steps
    num_bits = args.num_bits

    # Full view implies capacity fibonacci
    if args.full_view:
        args.capacity_fibonacci = True

    # Capacity fibonacci implies spatial fibonacci
    if args.capacity_fibonacci:
        args.fibonacci = True

    # Fibonacci auto-mode: determine beings from input size
    if args.fibonacci:
        fib_k = fibonacci_split(num_bits, min_k=2)
        octave_size = len(fib_k)
        if args.num_beings > octave_size:
            # User wants more beings: repeat octaves to fill
            num_beings = args.num_beings
            print(f"Fibonacci: {octave_size}-being octave repeated {num_beings // octave_size}x (+{num_beings % octave_size} extra) = {num_beings} beings")
        else:
            num_beings = octave_size
        if args.bits_per_being == 0:
            args.bits_per_being = 8  # trigger mask generation
        args.combiner = 'masked'  # fibonacci needs masked combiner
        args.min_coverage = 1  # no overlap -- each ant owns its bits
        print(f"Fibonacci auto: {num_beings} beings, K octave={fib_k}")

    # Combinatorial mode auto-config
    if args.combinatorial:
        if args.fibonacci or args.capacity_fibonacci:
            parser.error("--combinatorial is incompatible with --fibonacci / --capacity_fibonacci")
        PHI = (1 + 5**0.5) / 2
        if args.bits_per_being == 0:
            args.bits_per_being = round(num_bits / PHI)
            print(f"Combinatorial auto: K = round({num_bits}/phi) = {args.bits_per_being}")
        args.combiner = 'masked'
        print(f"Combinatorial: {num_beings} beings, K={args.bits_per_being}, {num_bits} bits")

    print("="*70)
    print("SWARM CONFIG TEST: Multi-Task Benchmark")
    print("="*70)
    print()
    # Data source init: 3-tier priority (data_dir > text > math)
    text_corpus = None
    traindat_loader = None
    if args.data_dir:
        # Parse weights from JSON string
        data_weights = None
        if args.data_weights:
            data_weights = json.loads(args.data_weights)
        traindat_loader = TraindatLoader(args.data_dir, weights=data_weights)
        print(f"TRAINDAT MODE: {args.data_dir} ({len(traindat_loader.files)} files)")
        for fname in traindat_loader.files:
            w = traindat_loader.weights.get(fname, 1.0)
            print(f"  {fname} (weight={w:.1f})")
    elif args.text:
        text_corpus = load_text_corpus(args.text)
        print(f"TEXT MODE: {args.text} ({len(text_corpus):,} bytes)")

    print(f"Configuration:")
    print(f"  Num bits: {num_bits}")
    print(f"  Embedding: {embedding_dim}D")
    print(f"  Depth: {depth} layers")
    print(f"  Num beings: {num_beings}")
    print(f"  Combiner: {args.combiner}")
    print(f"  Fibonacci K: {args.fibonacci}")
    print(f"  Combinatorial: {args.combinatorial}")
    print(f"  Think ticks: {args.think_ticks}")
    print(f"  Temporal fibonacci: {args.temporal_fibonacci}")
    print(f"  Capacity fibonacci: {args.capacity_fibonacci}" +
          (f" (max_H={args.max_hidden}, min_H={args.min_hidden})" if args.capacity_fibonacci else ""))
    print(f"  Full view: {args.full_view}")
    if args.use_lcx and args.lcx_mode == 'hash':
        print(f"  LCX: HASH mode ({args.lcx_num_slots} slots × {embedding_dim}d, "
              f"key_dim={args.lcx_key_dim}, top_k={args.lcx_top_k})")
    elif args.use_lcx:
        spb = args.slots_per_being
        spb_desc = "ALL (global write)" if spb == -1 else f"{spb} per being"
        print(f"  LCX: DENSE mode ({num_bits}×{num_bits} = {num_bits**2} cells, slots_per_being={spb} ({spb_desc}))")
    else:
        print(f"  GEM: {num_bits} cells (golden ratio EMA, phi^-1=0.618)")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Grad accum: {args.grad_accum} (effective batch={args.batch_size * args.grad_accum})")
    print(f"  Active beings: {args.active_beings}")
    print(f"  Seq len: {args.seq_len}")
    print(f"  LR: {args.lr}")
    print(f"  Jump bias: {args.jump_bias} (sigmoid={1/(1+2.718**(-args.jump_bias)):.0%})")
    if args.combiner == 'ring_attention':
        print(f"  Entropy weight: {args.entropy_weight}")
        print(f"  Freeze gate steps: {args.freeze_gate_steps}")
    if args.bits_per_being > 0:
        print(f"  Bits per being: {args.bits_per_being}")
        print(f"  Min coverage: {args.min_coverage}")
        print(f"  Mask seed: {args.mask_seed}")
    if traindat_loader:
        print(f"  Task: traindat (mixed byte prediction)")
    elif args.text:
        print(f"  Task: text (next-byte prediction)")
    else:
        print(f"  Task: math (ECHO, OR, AND, XOR, ADD)")
    # Device setup
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"  Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"  TF32: enabled")
    print(f"  Steps: {num_steps:,}")
    print("="*70)
    print()

    # Create swarm model
    torch.manual_seed(42)
    memory_size = args.memory_size if args.memory_size > 0 else embedding_dim
    model = SwarmByteRingModel(
        num_memory_positions=memory_size,
        embedding_dim=embedding_dim,
        num_beings=num_beings,
        depth=depth,
        num_bits=num_bits,
        combiner_mode=args.combiner,
        bits_per_being=args.bits_per_being,
        min_coverage=args.min_coverage,
        mask_seed=args.mask_seed,
        fibonacci=args.fibonacci,
        combinatorial=args.combinatorial,
        think_ticks=args.think_ticks,
        temporal_fibonacci=args.temporal_fibonacci,
        capacity_fibonacci=args.capacity_fibonacci,
        max_hidden=args.max_hidden,
        min_hidden=args.min_hidden,
        full_view=args.full_view,
        use_lcx=args.use_lcx,
        slots_per_being=args.slots_per_being,
        lcx_mode=args.lcx_mode,
        lcx_num_slots=args.lcx_num_slots,
        lcx_key_dim=args.lcx_key_dim,
        lcx_top_k=args.lcx_top_k,
        lcx_num_levels=args.lcx_num_levels,
        lcx_level_slots=[int(x) for x in args.lcx_level_slots.split(',')] if args.lcx_level_slots else None,
        attention_radius=args.attention_radius,
        num_pointers=args.num_pointers,
    )

    # Print temporal fibonacci tick schedule
    if model.tick_periods is not None:
        periods = model.tick_periods.tolist()
        unique_periods = sorted(set(periods), reverse=True)
        print(f"\nTemporal Fibonacci Tick Schedule:")
        for p in unique_periods:
            count = periods.count(p)
            print(f"  Period {p:2d}: {count:3d} beings (fire every {p} ticks)")
        avg_active = sum(1.0 / p for p in periods)
        print(f"  Avg active per tick: {avg_active:.1f} / {len(periods)} ({avg_active/len(periods):.0%})")

    # Per-being capacity report
    if model.capacity_fibonacci:
        print(f"\nCapacity Fibonacci (per-being hidden dims):")
        total_being_params = 0
        for i in range(num_beings):
            k_i = int(model.receptive_masks[i].sum().item())
            h_i = model.hidden_dims[i]
            bp = sum(p.numel() for p in model.beings[i].parameters())
            bp += model.being_input_projs[i].weight.numel() + model.being_input_projs[i].bias.numel()
            bp += model.being_output_projs[i].weight.numel() + model.being_output_projs[i].bias.numel()
            bp += model.ring_read_projs[i].weight.numel() + model.ring_read_projs[i].bias.numel()
            bp += model.ring_write_projs[i].weight.numel() + model.ring_write_projs[i].bias.numel()
            if model.being_processing_layers is not None:
                for layer in model.being_processing_layers[i]:
                    bp += layer.weight.numel() + layer.bias.numel()
            total_being_params += bp
            print(f"  Being {i}: K={k_i:3d}  H={h_i:4d}  params={bp:>10,}")
        print(f"  Total being-specific: {total_being_params:,}")

    model = model.to(device)

    if args.compile:
        print("Compiling model with torch.compile()...")
        model = torch.compile(model)
        print("  Compiled. First step will be slow (tracing), then fast.")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    if args.num_pointers > 1:
        mode = "competitive attention" if args.num_pointers >= 3 else "averaging"
        print(f"  Ring pointers: {args.num_pointers} per being ({mode})")

    # Print receptive field masks
    if args.bits_per_being > 0 and model.receptive_masks is not None:
        masks = model.receptive_masks
        fib_tag = " (Fibonacci)" if args.fibonacci else ""
        print(f"\nReceptive Field Masks{fib_tag}:")
        if num_beings <= 20:
            for i in range(num_beings):
                bits = [j for j in range(num_bits) if masks[i, j] > 0.5]
                print(f"  Being {i}: K={len(bits):2d}  bits {bits}")
        else:
            per_being_k = [int(masks[i].sum().item()) for i in range(num_beings)]
            print(f"  K values: min={min(per_being_k)}, max={max(per_being_k)}, "
                  f"unique={len(set(per_being_k))}")
        cov = masks.sum(dim=0)
        print(f"  Coverage: min={int(cov.min().item())}, max={int(cov.max().item())}, "
              f"mean={cov.mean():.1f}")

    print()

    # Architecture config for CPU eval worker (saved in checkpoints)
    arch_config = {
        'num_bits': num_bits,
        'embedding_dim': embedding_dim,
        'depth': depth,
        'num_beings': num_beings,
        'combiner': args.combiner,
        'bits_per_being': args.bits_per_being,
        'min_coverage': args.min_coverage,
        'mask_seed': args.mask_seed,
        'fibonacci': args.fibonacci,
        'combinatorial': args.combinatorial,
        'temporal_fibonacci': args.temporal_fibonacci,
        'capacity_fibonacci': args.capacity_fibonacci,
        'full_view': args.full_view,
        'use_lcx': args.use_lcx,
        'lcx_mode': args.lcx_mode,
        'lcx_num_slots': args.lcx_num_slots,
        'lcx_key_dim': args.lcx_key_dim,
        'lcx_top_k': args.lcx_top_k,
        'max_hidden': args.max_hidden,
        'min_hidden': args.min_hidden,
        'memory_size': memory_size,
        'seq_len': args.seq_len,
        'data_dir': args.data_dir,
        'effort_mode': args.effort_mode,
        'attention_radius': args.attention_radius,
        'lcx_num_levels': args.lcx_num_levels,
        'lcx_level_slots': args.lcx_level_slots,
    }

    # Set initial being states from CLI
    if args.active_beings != 'all':
        active_set = set(int(x.strip()) for x in args.active_beings.split(','))
        for i in range(num_beings):
            if i in active_set:
                model.being_states[i] = 'active'
            else:
                model.being_states[i] = 'null'
                _set_being_grad(model, i, False)
        active_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        print(f"\nBeing states: { {i: model.being_states[i] for i in range(num_beings)} }")
        print(f"  Trainable params: {active_params:,}")
        print(f"  Frozen params: {frozen_params:,}")

    # Training setup
    seq_len = args.seq_len
    batch_size = args.batch_size
    max_value = 2 ** num_bits - 1  # Full bit range (all bits exercised)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=0.01
    )

    # LR schedule: linear warmup then cosine decay
    _base_lr = args.lr
    _warmup_steps = args.warmup_steps
    _lr_min = args.lr_min
    _lr_schedule_active = True  # disabled if user overrides LR via live controls

    def _get_scheduled_lr(step):
        """Warmup + cosine decay. Returns LR for this step."""
        if step < _warmup_steps:
            return _base_lr * (step + 1) / _warmup_steps
        progress = (step - _warmup_steps) / max(num_steps - _warmup_steps, 1)
        return _lr_min + 0.5 * (_base_lr - _lr_min) * (1.0 + math.cos(math.pi * progress))

    print(f"LR schedule: warmup {_warmup_steps} steps -> cosine decay to {_lr_min:.1e}")

    # AMP mixed precision — GradScaler only needed for FP16 (BF16 has FP32 exponent range)
    _use_scaler = args.amp and device.type == 'cuda' and getattr(args, 'amp_dtype', 'bfloat16') == 'float16'
    amp_scaler = torch.amp.GradScaler('cuda') if _use_scaler else None
    if args.amp:
        _dtype_str = getattr(args, 'amp_dtype', 'bfloat16')
        print(f"AMP: {_dtype_str} mixed precision enabled" + (" (GradScaler active)" if amp_scaler else ""))

    # Load checkpoint if resuming
    start_step = 0
    if args.resume:
        start_step, _ = load_checkpoint(args.resume, model, optimizer)
        print()

    # Override jump gate bias AFTER checkpoint load (so it actually takes effect on resume)
    if args.jump_bias != 0.5:
        for being in model.beings:
            nn.init.constant_(being.jump_gate.bias, args.jump_bias)
        print(f"Jump gate bias set to {args.jump_bias} (sigmoid={1/(1+2.718**(-args.jump_bias)):.0%})")

    print(f"Eval: CPU metrics every step (from training output)")
    print()

    # Log setup
    log_dir = Path(__file__).parent / "logs" / "swarm"
    log_dir.mkdir(parents=True, exist_ok=True)

    combiner_tag = f"_{args.combiner}" if args.combiner not in ('mean', 'masked') else ""
    rf_tag = f"_rf{args.bits_per_being}" if args.bits_per_being > 0 else ""
    bits_tag = f"_{num_bits}bit" if num_bits != 8 else ""
    fib_tag = "_fib" if args.fibonacci else ""
    comb_tag = "_comb" if args.combinatorial else ""
    text_tag = "_traindat" if traindat_loader else ("_text" if args.text else "")
    think_tag = f"_think{args.think_ticks}" if args.think_ticks > 0 else ""
    tempo_tag = "_tempo" if args.temporal_fibonacci else ""
    cap_tag = "_cap" if args.capacity_fibonacci else ""
    fv_tag = "_fv" if args.full_view else ""
    gpu_tag = "_gpu" if device.type == 'cuda' else ""
    config_name = f"{num_beings}beings_{embedding_dim}d_{depth}layers{bits_tag}{combiner_tag}{rf_tag}{fib_tag}{comb_tag}{tempo_tag}{cap_tag}{fv_tag}{text_tag}{think_tag}{gpu_tag}"
    log_path = log_dir / f"{config_name}.log"
    if log_path.exists():
        log_path.unlink()

    # Always write to current.log for dashboard auto-discovery
    current_log_path = log_dir / "current.log"
    if current_log_path.exists():
        # Archive previous current.log with timestamp
        import datetime
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_path = log_dir / f"archive_{ts}.log"
        try:
            current_log_path.rename(archive_path)
            print(f"Archived previous log -> {archive_path.name}")
        except PermissionError:
            print(f"Warning: current.log locked by another process, overwriting")

    # InfluxDB metrics writer (no-op if INFLUX_TOKEN not set)
    # Always write as "current" — Grafana is hardcoded to read this.
    # Old data is flushed so only the current run is visible.
    influx_writer.init()
    influx_writer.flush_bucket()
    influx_run_id = "current"

    print(f"Log: {log_path}")
    print(f"Live: {current_log_path} (dashboard reads this)")
    print()
    print("Dashboard: http://localhost:8501 (Streamlit) | http://localhost:3000 (Grafana)")
    print(f"  Launch: python -m streamlit run diamond_dashboard.py -- --log logs/swarm/current.log")
    print("="*70)
    print()

    start_time = time.time()

    with open(log_path, "w") as log_file, open(current_log_path, "w") as current_file:
        # Write masks header line for dashboard bit ownership map
        if hasattr(model, 'receptive_masks') and model.receptive_masks is not None:
            masks_parts = [f"num_bits={num_bits}"]
            if args.full_view:
                masks_parts.append("full_view=1")
            masks_json = {'num_bits': num_bits, 'full_view': args.full_view}
            if model.capacity_fibonacci:
                masks_json['hidden_dims'] = model.hidden_dims
            for i in range(model.num_beings):
                bits = model.receptive_masks[i].nonzero(as_tuple=True)[0].tolist()
                masks_parts.append(f"being_{i}={','.join(str(b) for b in bits)}")
                masks_json[f'being_{i}'] = bits
            masks_line = "masks | " + " | ".join(masks_parts) + "\n"
            log_file.write(masks_line)
            log_file.flush()
            current_file.write(masks_line)
            current_file.flush()
            # Also write companion JSON for dashboard
            masks_json_path = log_dir / "current_masks.json"
            with open(masks_json_path, 'w') as mf:
                json.dump(masks_json, mf)
            # Log mask assignments to InfluxDB for correlation panels
            _mask_dict = {}
            for _mi in range(model.num_beings):
                _mask_dict[_mi] = model.receptive_masks[_mi].nonzero(as_tuple=True)[0].tolist()
            influx_writer.log_masks(influx_run_id, _mask_dict, num_bits)
            # Render mask matrix PNG for Grafana
            try:
                import frame_renderer
                frame_renderer.render_mask_matrix(_mask_dict, num_bits)
            except Exception:
                pass

        # Write default controls.json for live tuning
        controls_path = str(Path(args.controls_path))
        initial_weights = traindat_loader.weights if traindat_loader else {}
        # Effort tier determines initial tt/lcx/batch — --start_lcx_off overrides to Alpha
        _effort = "Alpha" if args.start_lcx_off else args.effort
        write_default_controls(controls_path, args.lr, initial_weights,
                               think_ticks=args.think_ticks, checkpoint_every=args.checkpoint_every,
                               eval_every=args.eval_every, batch_size=batch_size,
                               use_lcx=args.use_lcx, stage="INFANT",
                               effort=_effort)
        print(f"Controls: {controls_path}")

        controls = read_controls(controls_path)
        # Apply controls BEFORE first forward pass (fixes race: model constructed
        # with _lcx_hash_mode=True, but --start_lcx_off needs it False from step 0)
        optimizer, _init_changes = apply_controls(controls, optimizer, traindat_loader, model)
        if _init_changes:
            print(f"  [CTRL init] {_init_changes}")

        # Capture initial LCX for drift tracking
        _lcx_initial = model.lcx.detach().cpu().tolist() if model.lcx is not None else None
        _gem_initial = model.gem.detach().cpu().tolist() if model.gem is not None else None

        for step in range(start_step, num_steps):
            step_start = time.time()

            # Effort-level: read effort_lock from controls (backward compat with effort_mode)
            _effort_lock = controls.get('effort_lock', None)
            if _effort_lock is None:
                _effort_mode = controls.get('effort_mode', args.effort_mode)
                _effort_lock = 'random' if _effort_mode else 'fast'
            _effort_level = None
            _lock_map = {'fast': 0, 'medium': 1, 'slow': 2}
            if _effort_lock == 'random':
                _effort_level, _effort_tt = sample_effort(seed_val=42 + step + 2000000)
                model.think_ticks = _effort_tt
                model.effort_level = _effort_level
            elif _effort_lock in _lock_map:
                model.effort_level = _lock_map[_effort_lock]
                _effort_level = model.effort_level
                lo, hi = EFFORT_RANGES[model.effort_level]
                model.think_ticks = random.randint(lo, hi)
            else:
                model.effort_level = 0

            # Generate training batch
            if traindat_loader:
                x_train, y_train = traindat_loader.sample_batch(
                    n_samples=batch_size, seq_len=seq_len,
                    num_bits=num_bits, seed=42 + step + 1000000,
                )
            elif text_corpus:
                x_train, y_train = generate_text_batch(
                    text_corpus, n_samples=batch_size, seq_len=seq_len,
                    num_bits=num_bits, seed=42 + step + 1000000,
                )
            else:
                raise NotImplementedError(
                    "Multitask binary encoding incompatible with Gray code scalar metrics. "
                    "Use --data_dir for traindat mode."
                )

            # Move training data to device
            x_train = x_train.to(device, non_blocking=True)
            y_train = y_train.to(device, non_blocking=True)

            # Freeze-gate warmup: freeze being params, train only gate
            if args.combiner == 'ring_attention' and args.freeze_gate_steps > 0:
                in_freeze = (step - start_step) < args.freeze_gate_steps
                for name, param in model.named_parameters():
                    if name == 'gate_temperature':
                        param.requires_grad = True  # Always train gate temp
                    else:
                        param.requires_grad = not in_freeze

            # Capture LCX before forward pass (for before/after visualization)
            _lcx_before = model.lcx.detach().cpu().tolist() if model.lcx is not None else None

            # Train step — autocast for AMP regardless of GradScaler
            _amp_ctx_used = args.amp and device.type == 'cuda'
            _amp_ctx = torch.amp.autocast('cuda', dtype=torch.bfloat16) if _amp_ctx_used else None

            if _amp_ctx_used:
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    output, train_stats = model(x_train, return_stats=True, return_being_outputs=True)
                    # Uniform position weighting: all positions contribute equally.
                    # For traindat (echo/not/shift), predictable positions provide
                    # gradient signal while random positions average to zero gradient.
                    loss = nn.functional.binary_cross_entropy_with_logits(output, y_train)
                    if args.combiner == 'ring_attention' and args.entropy_weight > 0:
                        entropy_loss = model._last_entropy_loss
                        loss = loss + args.entropy_weight * entropy_loss
            else:
                output, train_stats = model(x_train, return_stats=True, return_being_outputs=True)
                # Uniform position weighting: all positions contribute equally.
                # For traindat (echo/not/shift), predictable positions provide
                # gradient signal while random positions average to zero gradient.
                loss = nn.functional.binary_cross_entropy_with_logits(output, y_train)
                # Gate entropy regularizer (ring_attention only)
                if args.combiner == 'ring_attention' and args.entropy_weight > 0:
                    entropy_loss = model._last_entropy_loss
                    loss = loss + args.entropy_weight * entropy_loss

            # LCX auxiliary losses for gradient flow (write gate + read attention + zoom gate)
            # fp32 cast prevents bf16 precision loss when adding small aux to larger main loss
            if getattr(model, '_lcx_hash_mode', False):
                _lcx_aux_coeff = 0.1
                _wg_aux = getattr(model, '_lcx_write_gate_aux_loss', None)
                if _wg_aux is not None and _wg_aux.requires_grad:
                    loss = loss.float() + _lcx_aux_coeff * _wg_aux.float()
                _ra_aux = getattr(model, '_lcx_read_attn_aux_loss', None)
                if _ra_aux is not None and _ra_aux.requires_grad:
                    loss = loss.float() + _lcx_aux_coeff * _ra_aux.float()
                _zg_aux = getattr(model, '_lcx_zoom_gate_aux_loss', None)
                if _zg_aux is not None and _zg_aux.requires_grad:
                    loss = loss.float() + _lcx_aux_coeff * _zg_aux.float()

            # Gradient accumulation
            scaled_loss = loss / args.grad_accum
            if amp_scaler:
                amp_scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            # ---- GRADIENT FLOW DIAGNOSTICS (every 10 steps, before zero_grad) ----
            _lcx_grad_cache = None  # populated inside GRAD block, read by LCX logging
            if step % 10 == 0:
                gd_parts = []
                # input_proj gradient
                _ip_g = model.input_proj.weight.grad if model.input_proj is not None else None
                gd_parts.append(f"inp={_ip_g.norm().item():.4e}" if _ip_g is not None else "inp=None")
                # output_proj gradient
                _op_g = model.output_proj.weight.grad if model.output_proj is not None else None
                gd_parts.append(f"out={_op_g.norm().item():.4e}" if _op_g is not None else "out=None")
                # processing layers (show first, middle, last)
                if model.processing_layers is not None and len(model.processing_layers) > 0:
                    _pl = model.processing_layers
                    _pl_norms = []
                    for _li, _layer in enumerate(_pl):
                        _lg = _layer.weight.grad
                        _pl_norms.append(_lg.norm().item() if _lg is not None else 0.0)
                    # Show first, mid, last layer grad norms + ratio first/last
                    _first, _mid, _last = _pl_norms[0], _pl_norms[len(_pl_norms)//2], _pl_norms[-1]
                    _ratio = _first / _last if _last > 1e-12 else float('inf')
                    gd_parts.append(f"depth[0]={_first:.4e} [{len(_pl_norms)//2}]={_mid:.4e} [{len(_pl_norms)-1}]={_last:.4e} ratio={_ratio:.1f}")
                # being-specific (jump_gate, context_strength)
                if len(model.beings) > 0:
                    _bp = model.beings[0]
                    _jg = _bp.jump_gate.weight.grad if hasattr(_bp, 'jump_gate') and _bp.jump_gate.weight.grad is not None else None
                    gd_parts.append(f"jump={_jg.norm().item():.4e}" if _jg is not None else "jump=None")
                    if hasattr(_bp, 'jump_gate_b'):
                        _jg_b = _bp.jump_gate_b.weight.grad if _bp.jump_gate_b.weight.grad is not None else None
                        gd_parts.append(f"jump_b={_jg_b.norm().item():.4e}" if _jg_b is not None else "jump_b=None")
                    if hasattr(_bp, 'jump_gate_c'):
                        _jg_c = _bp.jump_gate_c.weight.grad if _bp.jump_gate_c.weight.grad is not None else None
                        gd_parts.append(f"jump_c={_jg_c.norm().item():.4e}" if _jg_c is not None else "jump_c=None")
                    _cs = _bp.context_strength.grad if hasattr(_bp, 'context_strength') and _bp.context_strength.grad is not None else None
                    gd_parts.append(f"ctx_str={_cs.norm().item():.4e}" if _cs is not None else "ctx_str=None")
                # LCX gradient norms
                _lcx_rq = getattr(model, 'lcx_route_query', None)
                _lcx_rq_g = _lcx_rq.weight.grad if _lcx_rq is not None and _lcx_rq.weight.grad is not None else None
                _lcx_rq_val = _lcx_rq_g.norm().item() if _lcx_rq_g is not None else 0.0
                gd_parts.append(f"lcx_rq={_lcx_rq_val:.4e}")
                _lcx_wg = getattr(model, 'lcx_write_gate', None)
                _lcx_wg_g = _lcx_wg.weight.grad if _lcx_wg is not None and _lcx_wg.weight.grad is not None else None
                _lcx_wg_val = _lcx_wg_g.norm().item() if _lcx_wg_g is not None else 0.0
                gd_parts.append(f"lcx_wg={_lcx_wg_val:.4e}")
                _zg = getattr(model, 'zoom_gate', None)
                _zg_g = _zg.weight.grad if _zg is not None and hasattr(_zg, 'weight') and _zg.weight.grad is not None else None
                _zg_val = _zg_g.norm().item() if _zg_g is not None else 0.0
                gd_parts.append(f"zg={_zg_val:.4e}")
                # Stash LCX grads for InfluxDB (used in LCX logging section below)
                _lcx_grad_cache = {'lcx_rq': _lcx_rq_val, 'lcx_wg': _lcx_wg_val, 'zg': _zg_val}
                # Total grad norm (all params)
                _total_gn = 0.0
                _n_params = 0
                for _p in model.parameters():
                    if _p.grad is not None:
                        _total_gn += _p.grad.norm().item() ** 2
                        _n_params += 1
                _total_gn = _total_gn ** 0.5
                gd_parts.append(f"total={_total_gn:.4e}({_n_params}p)")
                diag_line = f"  [GRAD step {step}] {' | '.join(gd_parts)}"
                print(diag_line)
                log_file.write(diag_line + "\n")
                current_file.write(diag_line + "\n")

            # Optimizer step + zero_grad (AFTER diagnostics so .grad is readable)
            if (step + 1) % args.grad_accum == 0 or step == num_steps - 1:
                if amp_scaler:
                    amp_scaler.unscale_(optimizer)

                # AGC: Adaptive Gradient Control (two-sided band normalizer)
                # Ported from VRAXION agc.py — normalizes gradients to [agc_low, agc_high]
                # Scales DOWN spikes, scales UP weak gradients (fights collapse)
                _agc_on = controls.get('agc_enabled', True)
                _agc_lo = float(controls.get('agc_low', 1.0))
                _agc_hi = float(controls.get('agc_high', 5.0))
                _agc_scale = 1.0
                _agc_norm = 0.0
                if _agc_on:
                    for _p in model.parameters():
                        if _p.grad is not None:
                            _agc_norm += _p.grad.data.norm(2).item() ** 2
                    _agc_norm = _agc_norm ** 0.5
                    if _agc_norm > _agc_hi:
                        _agc_scale = _agc_hi / _agc_norm
                    elif _agc_norm > 0 and _agc_norm < _agc_lo:
                        _agc_scale = _agc_lo / _agc_norm
                    if _agc_scale != 1.0:
                        with torch.no_grad():
                            for _p in model.parameters():
                                if _p.grad is not None:
                                    _p.grad.data.mul_(_agc_scale)
                        if step % 10 == 0:
                            _dir = "DOWN" if _agc_scale < 1.0 else "UP"
                            _agc_msg = f"  [AGC] norm={_agc_norm:.2f} -> scale {_dir} {_agc_scale:.4f} (band [{_agc_lo}, {_agc_hi}])"
                            print(_agc_msg)
                            log_file.write(_agc_msg + "\n")
                            current_file.write(_agc_msg + "\n")

                # Hard clip as emergency backstop (at AGC high band)
                _clip_max = _agc_hi if _agc_on else 1.0
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=_clip_max)
                if amp_scaler:
                    amp_scaler.step(optimizer)
                    amp_scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            # Live controls polling
            if step % args.controls_every == 0:
                controls = read_controls(controls_path)
                # Snapshot controls LR to detect user clicks (not scheduler drift)
                _ctrl_lr_now = controls.get('lr')
                optimizer, changes = apply_controls(controls, optimizer, traindat_loader, model)
                # Detect deliberate user LR change via control panel
                if _ctrl_lr_now is not None and abs(_ctrl_lr_now - _base_lr) > 1e-10:
                    if _lr_schedule_active:
                        _lr_schedule_active = False
                        _base_lr = _ctrl_lr_now
                        print(f"  [LR] Manual override to {_ctrl_lr_now:.6f}, auto schedule disabled")
                # Live batch_size override
                if controls.get('batch_size') is not None:
                    new_bs = int(controls['batch_size'])
                    if new_bs != batch_size:
                        print(f"  [CTRL] batch_size: {batch_size} -> {new_bs}")
                        batch_size = new_bs
                # Handle resize_lcx command from control panel
                _resize_target = controls.pop('_resize_lcx', None)
                if _resize_target and hasattr(model, 'resize_lcx'):
                    print(f"  [CTRL] resize_lcx -> {int(_resize_target):,} slots")
                    model.resize_lcx(int(_resize_target))
                    # Clear command from controls.json so it doesn't fire again
                    with open(controls_path, 'w') as _cf:
                        json.dump(controls, _cf, indent=2)
                if changes:
                    print(f"  [CTRL] {changes}")

            # Apply LR schedule (warmup + cosine decay) — overrides controls LR when active
            if _lr_schedule_active:
                _scheduled_lr = _get_scheduled_lr(step)
                for _pg in optimizer.param_groups:
                    _pg['lr'] = _scheduled_lr

            step_time = time.time() - step_start
            train_loss = loss.item()

            # CPU eval every step: detach outputs, compute metrics on CPU
            with torch.no_grad():
                output_cpu = output.detach().cpu()
                y_cpu = y_train.detach().cpu()
                being_outputs_cpu = [b.detach().cpu() for b in train_stats['being_outputs']]
                stats_cpu = {k: v for k, v in train_stats.items()}
                stats_cpu['being_outputs'] = being_outputs_cpu

            # Count contributing beings for spatial metric denominators
            _n_cont = sum(1 for s in model.being_states.values() if s != 'null')
            _masks = model.receptive_masks.detach().cpu() if model.receptive_masks is not None else None
            metrics = evaluate_metrics(output_cpu, stats_cpu, y_cpu, train_loss, num_beings, num_bits, _n_cont, receptive_masks=_masks)
            _current_lr = optimizer.param_groups[0]['lr']
            log_line = format_metrics_line(step, train_loss, step_time, metrics)
            log_line += f" | lr={_current_lr:.2e}"
            log_file.write(log_line + "\n")
            current_file.write(log_line + "\n")

            # InfluxDB metrics (non-blocking, no-op if not configured)
            influx_writer.log_step(
                influx_run_id, step, train_loss,
                bit_acc=metrics['bit_acc'], byte_match=metrics['byte_match'],
                oracle=metrics['oracle_acc'], bit_oracle=metrics['bit_oracle_acc'],
                ensemble_benefit=metrics['ensemble_benefit'],
                coverage=metrics['coverage'], clustering=metrics['clustering'],
                circular_spread=metrics['circular_spread'], s_per_step=step_time,
                n_bits=num_bits,
                gate_entropy=stats_cpu.get('gate_entropy', 0),
                effort_level=_effort_level if _effort_level is not None else -1,
                think_ticks=model.think_ticks,
                batch_size=batch_size,
                use_lcx=1 if getattr(model, '_lcx_hash_mode', False) else 0,
                effort_name=getattr(model, '_current_effort_name', ''),
                current_stage=getattr(model, '_current_stage', 'UNKNOWN'),
                agc_norm=_agc_norm, agc_scale=_agc_scale)
            coverage_per_bit = model.receptive_masks.sum(dim=0) if model.receptive_masks is not None else None
            for bi in range(num_beings):
                bits_i = model.receptive_masks[bi].nonzero(as_tuple=True)[0] if model.receptive_masks is not None else []
                k = len(bits_i)
                uniq = int((coverage_per_bit[bits_i] == 1).sum().item()) if coverage_per_bit is not None and k > 0 else 0
                influx_writer.log_being(
                    influx_run_id, step, bi,
                    accuracy=metrics['being_accs'][bi],
                    masked_acc=metrics['being_masked_accs'][bi],
                    jump_rate=metrics['jump_rates'][bi] if bi < len(metrics['jump_rates']) else 0,
                    k_bits=k if model.receptive_masks is not None else num_bits,
                    unique_bits=uniq, redundant_bits=k - uniq,
                    ctx_scale=stats_cpu.get('diag_ctx_scales', [0]*num_beings)[bi])
            # Heavy per-bit/per-frame InfluxDB writes disabled — were causing ~82K writes/step
            # log_bits, log_ant_bit_acc, log_frame_snapshot all disabled for disk I/O
            # Per-bit accuracy still logged to the text log file above

            # Write frame snapshot sidecar (tiny JSON, dashboard reads it for live images)
            import json as _json_frame
            _frame_sidecar = {
                'step': step,
                'num_bits': num_bits,
                'seq_len': output_cpu.shape[1] if output_cpu.dim() > 1 else 1,
                'per_bit_accs': metrics['per_bit_accs'],
                'bit_acc': metrics['bit_acc'],
                'byte_match': metrics['byte_match'],
                'bit_bar': metrics.get('bit_bar', ''),
            }
            # Add first sample's input/output/prediction (1 × seq × channels)
            try:
                _x_sample = x_train[0].detach().cpu().tolist()  # [seq, num_bits] Gray scalars
                _y_sample = y_cpu[0].tolist()  # [seq, num_bits] Gray scalars
                _pred_prob = torch.sigmoid(output_cpu[0])  # logits → (0,1) scalars
                _pred_raw = _pred_prob.tolist()  # continuous predictions for dashboard
                _pred_bytes = _output_to_bytes(output_cpu[0:1]).squeeze(0)  # [seq, 8] decoded bytes
                _pred_sample = _pred_bytes.tolist()  # decoded byte values
                _frame_sidecar['input_sample'] = _x_sample
                _frame_sidecar['target_sample'] = _y_sample
                _frame_sidecar['pred_sample'] = _pred_sample
                _frame_sidecar['pred_raw'] = _pred_raw
            except Exception:
                pass
            _frame_path = os.path.join(os.path.dirname(log_path), 'frame_latest.json')
            with open(_frame_path, 'w') as _ff:
                _json_frame.dump(_frame_sidecar, _ff)

            # Memory logging (LCX or GEM)
            if getattr(model, '_lcx_num_levels', 0) > 0 and not model._lcx_hash_mode:
                # LCX OFF (INFANT stage) — compact one-liner
                _stage = getattr(model, '_current_stage', 'INFANT')
                lcx_line = f"  LCX: OFF ({_stage})"
                log_file.write(lcx_line + "\n")
                current_file.write(lcx_line + "\n")
            elif getattr(model, '_lcx_num_levels', 0) > 0 and model._lcx_hash_mode and model.think_ticks < 1:
                # LCX ON but tt=0 — writes only, no readback
                lcx_line = "  LCX: standby (tt=0, writes only)"
                log_file.write(lcx_line + "\n")
                current_file.write(lcx_line + "\n")
            elif model._lcx_hash_mode:
                # Zoom LCX active: full per-level display
                import json as _json
                _level_norms_str_parts = []
                _level_data = {}
                _influx_level_norms = {}
                _influx_level_used = {}
                _influx_level_total = {}
                _influx_slot_norms = {}
                # Level cap: tick N unlocks level N+1. Max active = think_ticks.
                _max_active = min(model.think_ticks, model._lcx_num_levels - 1)
                import math as _math
                for _lvl in range(_max_active + 1):
                    _keys, _vals = model._lcx_level_bufs(_lvl)
                    _norms = _vals.detach().cpu().norm(dim=-1)  # [S_lvl]
                    _used = int((_norms > 0).sum().item())
                    _total = int(_norms.shape[0])
                    _cols = _math.ceil(_total ** 0.5)
                    _rows = _math.ceil(_total / _cols) if _cols > 0 else 1
                    _side = _cols  # legacy compat
                    _norm_sum = _norms.sum().item()
                    _level_norms_str_parts.append(f"L{_lvl}={_norm_sum:.3f}({_used})")
                    _level_data[f'L{_lvl}'] = _norms.tolist()
                    _key_norms = _keys.detach().cpu().norm(dim=-1)  # [S_lvl]
                    _level_data[f'L{_lvl}_keys'] = _key_norms.tolist()
                    _level_data[f'L{_lvl}_used'] = _used
                    _level_data[f'L{_lvl}_total'] = _total
                    _level_data[f'L{_lvl}_side'] = _side
                    _level_data[f'L{_lvl}_cols'] = _cols
                    _level_data[f'L{_lvl}_rows'] = _rows
                    _influx_level_norms[_lvl] = _norm_sum
                    _influx_level_used[_lvl] = _used
                    _influx_level_total[_lvl] = _total
                    if _total <= 64:  # skip large levels — heatmap reads from sidecar
                        _influx_slot_norms[_lvl] = _level_data[f'L{_lvl}']
                _norms_str = ' '.join(_level_norms_str_parts)
                # Heat stats (hot-bin tracking for sparse subdivision)
                _heat_stats = model.lcx_heat_stats() if hasattr(model, 'lcx_heat_stats') else {}
                _alloc_lvls = _heat_stats.get('allocated_levels', '?')
                _heat_parts = []
                for _hlvl in range(_max_active + 1):
                    _hk = f'L{_hlvl}_hot_slots'
                    _tk = f'L{_hlvl}_total_slots'
                    if _hk in _heat_stats:
                        _vk = f'L{_hlvl}_valid_slots'
                        _valid = _heat_stats.get(_vk, '?')
                        _heat_parts.append(f"L{_hlvl}:{_heat_stats[_hk]}/{_heat_stats[_tk]}(v={_valid})")
                _heat_str = ' '.join(_heat_parts) if _heat_parts else ''
                # Bin heat + valid for strip visualization (128 bins per level)
                _STRIP_BINS = 128
                for _hvlvl in range(_max_active + 1):
                    _h_raw = getattr(model, f'lcx_heat_{_hvlvl}', None)
                    _v_raw = getattr(model, f'lcx_valid_{_hvlvl}', None)
                    if _h_raw is not None and _hvlvl in model._lcx_allocated_levels:
                        _h_cpu = _h_raw.cpu().float()
                        _level_data[f'L{_hvlvl}_heat_raw'] = _h_cpu.int().tolist()
                        _n_slots = _h_cpu.shape[0]
                        if _n_slots <= _STRIP_BINS:
                            _level_data[f'L{_hvlvl}_heat_bins'] = _h_cpu.int().tolist()
                        else:
                            _hbins = []
                            for _bi in range(_STRIP_BINS):
                                _s = (_bi * _n_slots) // _STRIP_BINS
                                _e = ((_bi + 1) * _n_slots) // _STRIP_BINS
                                _hbins.append(int(_h_cpu[_s:_e].max().item()))
                            _level_data[f'L{_hvlvl}_heat_bins'] = _hbins
                    if _v_raw is not None and _hvlvl in model._lcx_allocated_levels:
                        _v_cpu = _v_raw.cpu()
                        _n_slots = _v_cpu.shape[0]
                        if _n_slots <= _STRIP_BINS:
                            _level_data[f'L{_hvlvl}_valid_bins'] = _v_cpu.int().tolist()
                        else:
                            _vbins = []
                            for _bi in range(_STRIP_BINS):
                                _s = (_bi * _n_slots) // _STRIP_BINS
                                _e = ((_bi + 1) * _n_slots) // _STRIP_BINS
                                _vbins.append(int(_v_cpu[_s:_e].sum().item()))
                            _level_data[f'L{_hvlvl}_valid_bins'] = _vbins
                    # Spatial rank bins: max-pool downsample for native Grafana table
                    if _h_raw is not None and _hvlvl in model._lcx_allocated_levels:
                        _raw_h = _level_data.get(f'L{_hvlvl}_heat_raw', [])
                        if _raw_h:
                            _N_RANK = 100
                            import math as _mlog
                            for _ri in range(_N_RANK):
                                _s = (_ri * len(_raw_h)) // _N_RANK
                                _e = ((_ri + 1) * len(_raw_h)) // _N_RANK
                                _bin_max = max(_raw_h[_s:_e]) if _e > _s else 0
                                _heat_stats[f'L{_hvlvl}_rk{_ri:02d}'] = round(_mlog.log2(max(_bin_max, 0) + 1), 2)
                        # Entropy metrics for memory utilization
                        _raw_list = _level_data.get(f'L{_hvlvl}_heat_raw', [])
                        _total_h = sum(_raw_list)
                        if _total_h > 0:
                            import math as _m
                            _n_total = len(_raw_list)
                            _n_active = sum(1 for _hv in _raw_list if _hv > 0)
                            _probs = [_hv / _total_h for _hv in _raw_list if _hv > 0]
                            _entropy = -sum(_p * _m.log2(_p) for _p in _probs)
                            _max_ent = _m.log2(_n_total) if _n_total > 1 else 1.0
                            _heat_stats[f'L{_hvlvl}_entropy_pct'] = round(_entropy / _max_ent * 100, 1)
                            _heat_stats[f'L{_hvlvl}_eff_slots'] = round(2 ** _entropy, 1)
                            _heat_stats[f'L{_hvlvl}_active_slots'] = _n_active
                            _heat_stats[f'L{_hvlvl}_total_slots'] = _n_total
                            # GPT Pro metrics: Top-1 mass, Participation Ratio, Top-6 mass
                            _heat_stats[f'L{_hvlvl}_top1_mass'] = round(max(_probs) * 100, 1)
                            _heat_stats[f'L{_hvlvl}_participation_ratio'] = round(1.0 / sum(_p * _p for _p in _probs), 1)
                            _top6 = sorted(_probs, reverse=True)[:6]
                            _heat_stats[f'L{_hvlvl}_top6_mass'] = round(sum(_top6) * 100, 1)
                            # Percentage metrics for level-agnostic dashboard template
                            _heat_stats[f'L{_hvlvl}_active_pct'] = round(_n_active / _n_total * 100, 1)
                            _heat_stats[f'L{_hvlvl}_part_ratio_pct'] = round((1.0 / sum(_p * _p for _p in _probs)) / _n_total * 100, 1)
                            # Value diversity: cosine dissimilarity of stored vectors
                            _vals = getattr(model, f'lcx_values_{_hvlvl}', None)
                            if _vals is not None and _n_active > 1:
                                _active_mask = _h_raw > 0
                                _active_vals = _vals[_active_mask]
                                _vnorms = _active_vals.norm(dim=1, keepdim=True).clamp(min=1e-8)
                                _normed = _active_vals / _vnorms
                                _sim = _normed @ _normed.T
                                _ns = _sim.size(0)
                                _diag_mask = 1.0 - torch.eye(_ns, device=_sim.device)
                                _avg_sim = (_sim * _diag_mask).sum() / (_ns * (_ns - 1))
                                _heat_stats[f'L{_hvlvl}_val_diversity'] = round((1.0 - _avg_sim.item()) * 100, 1)
                            # Score margin: routing quality diagnostic
                            _sm = getattr(model, '_last_score_margin', None)
                            if _sm is not None:
                                _heat_stats[f'L{_hvlvl}_score_margin'] = round(_sm, 4)
                            _st1 = getattr(model, '_last_score_top1', None)
                            if _st1 is not None:
                                _heat_stats[f'L{_hvlvl}_score_top1'] = round(_st1, 4)
                lcx_line = f"  LCX-ZOOM {_max_active + 1}/{model._lcx_num_levels}lvl(alloc={_alloc_lvls}) [{_norms_str}] heat[{_heat_str}]"
                log_file.write(lcx_line + "\n")
                current_file.write(lcx_line + "\n")
                _grad_kw = {}
                if _lcx_grad_cache is not None:
                    _grad_kw['lcx_route_grad'] = _lcx_grad_cache['lcx_rq']
                    _grad_kw['lcx_write_grad'] = _lcx_grad_cache['lcx_wg']
                    _grad_kw['zoom_gate_grad'] = _lcx_grad_cache['zg']
                _grad_kw['write_differentiable'] = True  # write gate aux loss provides gradient
                _grad_kw['current_stage'] = getattr(model, '_current_stage', 'UNKNOWN')
                _wg_aux = getattr(model, '_lcx_write_gate_aux_loss', None)
                _grad_kw['lcx_write_aux_loss'] = _wg_aux.item() if _wg_aux is not None else 0.0
                _ra_aux = getattr(model, '_lcx_read_attn_aux_loss', None)
                _grad_kw['lcx_read_aux_loss'] = _ra_aux.item() if _ra_aux is not None else 0.0
                influx_writer.log_lcx_level_norms(
                    influx_run_id, step,
                    _influx_level_norms,
                    model._lcx_num_levels,
                    getattr(model, '_last_zoom_gate', None),
                    level_used=_influx_level_used,
                    level_total=_influx_level_total,
                    max_active_level=_max_active,
                    slot_norms=_influx_slot_norms,
                    heat_stats=_heat_stats,
                    **_grad_kw,
                )
                # Write JSON sidecar every step (control panel + dashboard read this)
                _zg_val = getattr(model, '_last_zoom_gate', None)
                _lcx_sidecar = {
                    'step': step,
                    'lcx_mode': 'hash',
                    'num_levels': model._lcx_num_levels,
                    'max_active_level': _max_active,
                    'level_slots': model._lcx_level_slots,
                    'total_slots': sum(model._lcx_level_slots),
                    'zoom_gate': _zg_val,
                    **_level_data,
                }
                _sidecar_path = os.path.join(os.path.dirname(log_path), 'lcx_latest.json')
                with open(_sidecar_path, 'w') as _sf:
                    _json.dump(_lcx_sidecar, _sf)
            elif model.lcx is not None:
                _lcx_vals = model.lcx.detach().cpu()  # [3, lcx_size]
                _nb = model.num_bits
                _lcx_norm = _lcx_vals.norm().item()
                _ch_names = ['R', 'G', 'B']
                _ch_norms = ' '.join(f"{_ch_names[i]}={_lcx_vals[i].norm().item():.3f}" for i in range(3))
                lcx_line = f"  LCX norm={_lcx_norm:.4f} ({_nb}x{_nb}x3 RGB) [{_ch_norms}] effort={model.effort_level}"
                log_file.write(lcx_line + "\n")
                current_file.write(lcx_line + "\n")
                # Only send scalar channel norms to InfluxDB (not the 16K-point grids)
                influx_writer.log_lcx_channel_norms(
                    influx_run_id, step,
                    _lcx_vals[0].norm().item(),
                    _lcx_vals[1].norm().item(),
                    _lcx_vals[2].norm().item(),
                    model.effort_level,
                )
                # Drift frame snapshots disabled — 16K InfluxDB writes per step
                # Matrix history disabled — was writing ~10MB/step, hammering disk I/O
                # Use lcx_filmstrip.py post-hoc if needed
                import json as _json
                # Write JSON sidecar every step (control panel + dashboard read this)
                if True:
                    _lcx_sidecar = {
                        'step': step, 'num_bits': _nb, 'side': _nb,
                        'R': _lcx_vals[0].tolist(),
                        'G': _lcx_vals[1].tolist(),
                        'B': _lcx_vals[2].tolist(),
                        'effort_level': model.effort_level,
                    }
                    _sidecar_path = os.path.join(os.path.dirname(log_path), 'lcx_latest.json')
                    with open(_sidecar_path, 'w') as _sf:
                        _json.dump(_lcx_sidecar, _sf)
            elif model.gem is not None:
                _gem_vals = model.gem.detach().cpu().tolist()
                _gem_norm = sum(v**2 for v in _gem_vals) ** 0.5
                _gem_str = " ".join(f"{v:+.3f}" for v in _gem_vals)
                gem_line = f"  GEM[{_gem_str}] norm={_gem_norm:.4f}"
                log_file.write(gem_line + "\n")
                current_file.write(gem_line + "\n")

            # Console output periodically
            if step % 500 == 0 or step == start_step:
                elapsed = time.time() - start_time
                _mem_summary = ""
                if model.lcx is not None:
                    _mem_summary = f" | LCX norm={_lcx_norm:.4f}"
                elif model.gem is not None:
                    _gn = model.gem.detach().norm().item()
                    _mem_summary = f" | GEM_norm={_gn:.4f}"
                print(
                    f"step {step:5d} | loss {train_loss:.6f} | "
                    f"bit_acc={metrics['bit_acc']:.4f} oracle={metrics['oracle_acc']:.4f} | "
                    f"time={elapsed:.1f}s{_mem_summary}"
                )

            log_file.flush()
            current_file.flush()

            # Multi-sample eval pass (stronger signal than single training sample)
            eval_every = controls.get('eval_every') or args.eval_every
            if step > 0 and step % eval_every == 0:
                model.eval()
                model._eval_skip_think = True
                eval_metrics_accum = None
                n_eval = controls.get('eval_samples') or args.eval_samples
                with torch.no_grad():
                    for ei in range(n_eval):
                        if traindat_loader:
                            x_ev, y_ev = traindat_loader.sample_batch(
                                n_samples=batch_size, seq_len=seq_len,
                                num_bits=num_bits, seed=7777 + step * 100 + ei,
                            )
                        elif text_corpus:
                            x_ev, y_ev = generate_text_batch(
                                text_corpus, n_samples=batch_size, seq_len=seq_len,
                                num_bits=num_bits, seed=7777 + step * 100 + ei,
                            )
                        else:
                            raise NotImplementedError(
                                "Multitask binary encoding incompatible with Gray code scalar metrics. "
                                "Use --data_dir for traindat mode."
                            )
                        x_ev = x_ev.to(device, non_blocking=True)
                        y_ev = y_ev.to(device, non_blocking=True)
                        ev_out, ev_stats = model(x_ev, return_stats=True, return_being_outputs=True)
                        ev_loss = nn.functional.binary_cross_entropy_with_logits(ev_out, y_ev).item()
                        ev_out_cpu = ev_out.detach().cpu()
                        y_ev_cpu = y_ev.detach().cpu()
                        ev_being_cpu = [b.detach().cpu() for b in ev_stats['being_outputs']]
                        ev_stats_cpu = {k: v for k, v in ev_stats.items()}
                        ev_stats_cpu['being_outputs'] = ev_being_cpu
                        em = evaluate_metrics(ev_out_cpu, ev_stats_cpu, y_ev_cpu, ev_loss, num_beings, num_bits, _n_cont, receptive_masks=_masks)
                        if eval_metrics_accum is None:
                            eval_metrics_accum = {k: v for k, v in em.items()}
                        else:
                            for k in ['eval_loss', 'overall_acc', 'bit_acc', 'avg_bit_acc', 'byte_match', 'hamming',
                                       'oracle_acc', 'bit_oracle_acc', 'ensemble_benefit',
                                       'circular_spread', 'coverage', 'clustering']:
                                eval_metrics_accum[k] += em[k]
                            for i in range(len(eval_metrics_accum['per_bit_accs'])):
                                eval_metrics_accum['per_bit_accs'][i] += em['per_bit_accs'][i]
                            for i in range(len(eval_metrics_accum['being_accs'])):
                                eval_metrics_accum['being_accs'][i] += em['being_accs'][i]
                            for i in range(len(eval_metrics_accum.get('being_masked_accs', []))):
                                eval_metrics_accum['being_masked_accs'][i] += em['being_masked_accs'][i]
                            for i in range(len(eval_metrics_accum['jump_rates'])):
                                eval_metrics_accum['jump_rates'][i] += em['jump_rates'][i]
                # Average
                for k in ['eval_loss', 'overall_acc', 'bit_acc', 'avg_bit_acc', 'byte_match', 'hamming',
                           'oracle_acc', 'bit_oracle_acc', 'ensemble_benefit',
                           'circular_spread', 'coverage', 'clustering']:
                    eval_metrics_accum[k] /= n_eval
                eval_metrics_accum['per_bit_accs'] = [v / n_eval for v in eval_metrics_accum['per_bit_accs']]
                eval_metrics_accum['being_accs'] = [v / n_eval for v in eval_metrics_accum['being_accs']]
                if 'being_masked_accs' in eval_metrics_accum:
                    eval_metrics_accum['being_masked_accs'] = [v / n_eval for v in eval_metrics_accum['being_masked_accs']]
                eval_metrics_accum['jump_rates'] = [v / n_eval for v in eval_metrics_accum['jump_rates']]
                eval_line = "EVAL | " + format_metrics_line(step, eval_metrics_accum['eval_loss'], 0, eval_metrics_accum)
                log_file.write(eval_line + "\n")
                current_file.write(eval_line + "\n")
                log_file.flush()
                current_file.flush()

                # InfluxDB eval metrics
                influx_writer.log_step(
                    influx_run_id, step, eval_metrics_accum['eval_loss'],
                    bit_acc=eval_metrics_accum['bit_acc'], byte_match=eval_metrics_accum['byte_match'],
                    oracle=eval_metrics_accum['oracle_acc'], bit_oracle=eval_metrics_accum['bit_oracle_acc'],
                    ensemble_benefit=eval_metrics_accum['ensemble_benefit'],
                    coverage=eval_metrics_accum['coverage'], clustering=eval_metrics_accum['clustering'],
                    circular_spread=eval_metrics_accum['circular_spread'], is_eval=True,
                    n_bits=num_bits,
                    batch_size=batch_size,
                    use_lcx=1 if getattr(model, '_lcx_hash_mode', False) else 0,
                    effort_name=getattr(model, '_current_effort_name', ''),
                    current_stage=getattr(model, '_current_stage', 'UNKNOWN'))
                influx_writer.log_bits(influx_run_id, step, eval_metrics_accum['per_bit_accs'])

                model._eval_skip_think = False
                model.train()

            # Save checkpoint periodically (frequency is live-controllable)
            ckpt_every = controls.get('checkpoint_every') or args.checkpoint_every
            if step > 0 and step % ckpt_every == 0:
                save_checkpoint(model, optimizer, step, 0.0, args.checkpoint_dir, config=arch_config)

    total_time = time.time() - start_time

    print()
    print("="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Config: {num_beings}x {embedding_dim}D beings, {depth} layers ({total_params:,} params)")
    print(f"Training time: {total_time:.1f}s")
    print(f"Eval: CPU metrics every step (from training output)")
    print("="*70)

    # Save final checkpoint
    final_step = num_steps - 1 if not args.resume else start_step + (num_steps - start_step)
    save_checkpoint(model, optimizer, final_step, 0.0, args.checkpoint_dir, config=arch_config)

    # Archive current run to git-tracked directory, then close InfluxDB
    import datetime
    _archive_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    _archive_dir = Path(__file__).parent / "archive" / f"{_archive_ts}_{config_name}"
    _harvest = influx_writer.harvest_run(str(_archive_dir), config_name=config_name)
    if _harvest:
        # Copy the log file into the archive too
        import shutil
        try:
            shutil.copy2(str(log_path), str(_archive_dir / "training.log"))
        except Exception:
            pass
        print(f"Archived run -> {_archive_dir.name}/")
        print(f"  {_harvest['total_steps']} steps, loss={_harvest['final_loss']:.6f}, bit_acc={_harvest['final_bit_acc']:.4f}")

    influx_writer.close()


if __name__ == "__main__":
    main()
