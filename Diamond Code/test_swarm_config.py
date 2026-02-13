"""
Test a swarm model configuration on multi-task benchmark.

Usage:
    python test_swarm_config.py --embedding 64 --num_beings 2 --depth 2

Logs everything (loss, accuracy, jump gates, swarm metrics) for dashboard viewing.
"""

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
from traindat_loader import TraindatLoader
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


def generate_multitask_batch(n_samples, seq_len=16, max_value=100, seed=None, num_bits=8, task='mixed'):
    """Generate batch with mixed operations (or single op if task != 'mixed')."""
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

        y_seq = torch.zeros(seq_len, num_bits)
        y_seq[0, :] = int_to_bits(a, num_bits)
        y_seq[1, :] = int_to_bits(b, num_bits)
        y_seq[2, :] = int_to_bits(result, num_bits)

        x_batch.append(x_seq)
        y_batch.append(y_seq)

    return torch.stack(x_batch), torch.stack(y_batch), torch.tensor(op_indices)


def position_accuracy(output, target, position):
    """Calculate byte accuracy at specific position."""
    pred_bits = (output[:, position, :] > 0.0).float()
    matches = (pred_bits == target[:, position, :]).all(dim=-1).float()
    return matches.mean().item()


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
    """Calculate exact byte match accuracy (all 8 bits correct)."""
    pred_bits = (output[:, position, :] > 0.0).float()
    matches = (pred_bits == target[:, position, :]).all(dim=-1).float()
    return matches.mean().item()


def bit_accuracy_at_position(output, target, position):
    """Calculate per-bit accuracy (may be misleading for imbalanced ops)."""
    pred_bits = (output[:, position, :] > 0.0).float()
    bit_matches = (pred_bits == target[:, position, :]).float()
    return bit_matches.mean().item()


def per_bit_accuracy_at_position(output, target, position):
    """Calculate accuracy for each bit position independently."""
    pred_bits = (output[:, position, :] > 0.0).float()
    target_bits = target[:, position, :]
    per_bit = (pred_bits == target_bits).float().mean(dim=0)  # [8]
    return per_bit.tolist()


def hamming_distance_at_position(output, target, position):
    """Calculate mean Hamming distance (# bit errors per byte)."""
    pred_bits = (output[:, position, :] > 0.0).float()
    bit_errors = (pred_bits != target[:, position, :]).float()
    mean_errors = bit_errors.sum(dim=-1).mean().item()
    return mean_errors


def oracle_best_of_n_accuracy(being_outputs, target, position):
    """
    Oracle accuracy: best-of-N selection per sample.
    Upper bound on ensemble potential.

    Args:
        being_outputs: [num_beings, T, B, 8] ALL timestep outputs from each being
        target: [B, T, 8] ground truth
        position: timestep position to evaluate

    Returns:
        Oracle accuracy (best being per sample)
    """
    # Extract position slice: [num_beings, B, 8]
    being_outputs_at_pos = being_outputs[:, position, :, :]

    num_beings = being_outputs_at_pos.size(0)
    B = being_outputs_at_pos.size(1)

    # For each sample, find which being got it right
    oracle_correct = 0
    for b in range(B):
        # Check each being's prediction for this sample
        any_correct = False
        for being_idx in range(num_beings):
            pred_bits = (being_outputs_at_pos[being_idx, b, :] > 0.0).float()  # [8]
            is_correct = (pred_bits == target[b, position, :]).all().item()
            if is_correct:
                any_correct = True
                break
        if any_correct:
            oracle_correct += 1

    return oracle_correct / B


def bit_oracle_accuracy(being_outputs, target, position):
    """
    Bit-oracle: for each bit, did ANY being get it right?
    Upper bound for what a perfect bit-level selector could achieve.
    If bit_oracle >> ensemble, combiner is the bottleneck.
    """
    being_outputs_at_pos = being_outputs[:, position, :, :]  # [num_beings, B, 8]
    target_at_pos = target[:, position, :]  # [B, 8]

    being_binary = (being_outputs_at_pos > 0.0).float()  # [num_beings, B, 8]
    target_binary = (target_at_pos > 0.0).float()  # [B, 8]

    correct_per_being = (being_binary == target_binary.unsqueeze(0))  # [num_beings, B, 8]
    any_correct = correct_per_being.any(dim=0)  # [B, 8]

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
                # Compute accuracy for this being on this operation
                being_output = being_outputs_at_pos[being_idx][mask]  # [masked_B, 8]
                op_target = target[mask]  # [masked_B, T, 8]
                pred_bits = (being_output > 0.0).float()  # [masked_B, 8]
                matches = (pred_bits == op_target[:, position, :]).all(dim=-1).float()
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

    # Per-being metrics
    being_outputs = stats['being_outputs']
    being_accs = []
    being_masked_accs = []
    for i in range(num_beings):
        being_output_transposed = being_outputs[i].transpose(0, 1)
        being_acc = position_accuracy(being_output_transposed, y, eval_pos)
        being_accs.append(being_acc)

        # Mask-aware accuracy: only evaluate covered bits
        if receptive_masks is not None and i < len(receptive_masks):
            mask = receptive_masks[i]  # [num_bits]
            covered = mask.bool()
            if covered.any():
                pred = (being_outputs[i][eval_pos] > 0.0).float()  # [B, num_bits]
                tgt = y[:, eval_pos, :]  # [B, num_bits]
                masked_acc = (pred[:, covered] == tgt[:, covered]).float().mean().item()
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


def save_checkpoint(model, optimizer, step, best_acc, checkpoint_dir, is_best=False, config=None):
    """Save training checkpoint with architecture config for CPU eval worker."""
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

    # Save regular checkpoint
    checkpoint_path = Path(checkpoint_dir) / f'checkpoint_step_{step}.pt'
    torch.save(checkpoint, checkpoint_path)
    print(f"  [SAVE] Checkpoint saved: {checkpoint_path}")

    # Save best model separately
    if is_best:
        best_path = Path(checkpoint_dir) / 'best_model.pt'
        torch.save(checkpoint, best_path)
        print(f"  [BEST] Best model saved: {best_path}")


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Load training checkpoint."""
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    missing, unexpected = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
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
    parser.add_argument('--text', type=str, default=None,
                        help='Path to text file for byte-level language modeling (overrides math tasks)')
    parser.add_argument('--think_ticks', type=int, default=0,
                        help='Extra ring ticks without input (beings read each other, then output). 0=reflex mode.')
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
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Training batch size (default: 32)')
    parser.add_argument('--seq_len', type=int, default=16,
                        help='Sequence length / positions per sample (default: 16)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
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
    if args.use_lcx:
        spb = args.slots_per_being
        spb_desc = "ALL (global write)" if spb == -1 else f"{spb} per being"
        print(f"  LCX: {num_bits}×{num_bits} = {num_bits**2} cells, slots_per_being={spb} ({spb_desc})")
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
        'max_hidden': args.max_hidden,
        'min_hidden': args.min_hidden,
        'memory_size': memory_size,
        'seq_len': args.seq_len,
        'data_dir': args.data_dir,
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
        write_default_controls(controls_path, args.lr, initial_weights,
                               think_ticks=args.think_ticks, checkpoint_every=args.checkpoint_every,
                               eval_every=args.eval_every)
        print(f"Controls: {controls_path}")

        controls = read_controls(controls_path)

        # Capture initial LCX for drift tracking
        _lcx_initial = model.lcx.detach().cpu().tolist() if model.lcx is not None else None
        _gem_initial = model.gem.detach().cpu().tolist() if model.gem is not None else None

        for step in range(start_step, num_steps):
            step_start = time.time()

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
                x_train, y_train, _ = generate_multitask_batch(
                    n_samples=batch_size, seq_len=seq_len, max_value=max_value,
                    seed=42 + step + 1000000, num_bits=num_bits, task=args.task,
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

            # Train step
            output, train_stats = model(x_train, return_stats=True, return_being_outputs=True)
            # Position-weighted loss: upweight the actual task (position 2)
            # Without this, position 2 is only 6.25% of total gradient (1/16)
            # while trivial copy/zero positions consume 93.75%.
            per_pos_loss = nn.functional.binary_cross_entropy_with_logits(
                output, y_train, reduction='none'
            ).mean(dim=(0, 2))  # [T] — mean over batch and bits, per position
            pos_weights = torch.ones(output.size(1), device=output.device)
            task_pos = min(2, output.size(1) - 1)
            pos_weights[task_pos] = 10.0
            loss = (per_pos_loss * pos_weights).sum() / pos_weights.sum()

            # Gate entropy regularizer (ring_attention only)
            if args.combiner == 'ring_attention' and args.entropy_weight > 0:
                entropy_loss = model._last_entropy_loss
                loss = loss + args.entropy_weight * entropy_loss

            # Gradient accumulation
            scaled_loss = loss / args.grad_accum
            scaled_loss.backward()

            # ---- DIAGNOSTICS (must run BEFORE zero_grad!) ----
            diag = args.diagnostic
            if diag != 'none' and step % 100 == 0:
                diag_parts = []
                # Test 2: Gradient flow
                if diag in ('gradient', 'all'):
                    bp = model.beings[0]  # first being's params
                    inp_g = model.being_input_projs[0].weight.grad
                    out_g = model.being_output_projs[0].weight.grad
                    inp_gn = inp_g.norm().item() if inp_g is not None else 0.0
                    out_gn = out_g.norm().item() if out_g is not None else 0.0
                    diag_parts.append(f"grad_inp={inp_gn:.6f} grad_out={out_gn:.6f}")
                    if model.processing_layers is not None:
                        for li, layer in enumerate(model.processing_layers):
                            lg = layer.weight.grad
                            lgn = lg.norm().item() if lg is not None else 0.0
                            diag_parts.append(f"proc{li}={lgn:.6f}")
                    jg = bp.jump_gate.weight.grad
                    diag_parts.append(f"jump_gate={jg.norm().item():.6f}" if jg is not None else "jump_gate=0")
                # Test 3: Hidden state saturation
                if diag in ('saturation', 'all'):
                    h_abs = train_stats.get('diag_hidden_abs', [])
                    h_max = train_stats.get('diag_hidden_max', [])
                    for ti, (ha, hm) in enumerate(zip(h_abs, h_max)):
                        diag_parts.append(f"t{ti}_h_abs={ha:.4f}_max={hm:.4f}")
                # Test 4: ctx_scale and ring reads
                if diag in ('ring', 'all'):
                    ctx_s = train_stats.get('diag_ctx_scales', [])
                    rr = train_stats.get('diag_ring_read_norm', [])
                    diag_parts.append(f"ctx_scales=[{','.join(f'{c:.4f}' for c in ctx_s)}]")
                    for ti, rn in enumerate(rr):
                        diag_parts.append(f"t{ti}_ring_norm={rn:.4f}")
                if diag_parts:
                    diag_line = f"  [DIAG step {step}] " + " | ".join(diag_parts)
                    print(diag_line)
                    log_file.write(diag_line + "\n")
                    current_file.write(diag_line + "\n")

            # Optimizer step + zero_grad (AFTER diagnostics so .grad is readable)
            if (step + 1) % args.grad_accum == 0 or step == num_steps - 1:
                optimizer.step()
                optimizer.zero_grad()

            # Live controls polling
            if step % args.controls_every == 0:
                controls = read_controls(controls_path)
                optimizer, changes = apply_controls(controls, optimizer, traindat_loader, model)
                if changes:
                    print(f"  [CTRL] {changes}")

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
            log_line = format_metrics_line(step, train_loss, step_time, metrics)
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
                gate_entropy=stats_cpu.get('gate_entropy', 0))
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
            influx_writer.log_bits(influx_run_id, step, metrics['per_bit_accs'])
            # Per-ant-per-bit accuracy for correlation panels
            if _masks is not None:
                _ab_pos = min(2, output_cpu.size(1) - 1)
                for _abi in range(num_beings):
                    _ab_bits = _masks[_abi].nonzero(as_tuple=True)[0].tolist()
                    if _ab_bits:
                        _ab_pred = (stats_cpu['being_outputs'][_abi][_ab_pos] > 0.0).float()
                        _ab_tgt = y_cpu[:, _ab_pos, :]
                        _ab_accs = {b: (_ab_pred[:, b] == _ab_tgt[:, b]).float().mean().item() for b in _ab_bits}
                        influx_writer.log_ant_bit_acc(influx_run_id, step, _abi, _ab_accs)

            # Frame snapshots for Grafana (input/output — always, independent of memory type)
            _input_frame = x_train[0, :num_bits, :].detach().cpu().tolist()    # [num_bits, num_bits] = 8x8 square
            _output_frame = output[0, :num_bits, :].detach().cpu().tolist()    # [num_bits, num_bits] = 8x8 square
            influx_writer.log_frame_snapshot(influx_run_id, step, 'input', [v for row in _input_frame for v in row])
            influx_writer.log_frame_snapshot(influx_run_id, step, 'output', [v for row in _output_frame for v in row])

            # Memory logging (LCX or GEM)
            if model.lcx is not None:
                _lcx_vals = model.lcx.detach().cpu()
                _nb = model.num_bits
                _lcx_norm = _lcx_vals.norm().item()
                lcx_line = f"  LCX norm={_lcx_norm:.4f} ({_nb}x{_nb}={_nb**2} cells)"
                log_file.write(lcx_line + "\n")
                current_file.write(lcx_line + "\n")
                influx_writer.log_lcx(influx_run_id, step, _lcx_vals.tolist(), _nb)
                influx_writer.log_frame_snapshot(influx_run_id, step, 'lcx_state', _lcx_vals.tolist())
                # Drift = current - initial (how far each cell moved from start)
                if _lcx_initial is not None:
                    import torch as _t
                    _drift = (_lcx_vals - _t.tensor(_lcx_initial)).tolist()
                    influx_writer.log_frame_snapshot(influx_run_id, step, 'lcx_drift', _drift)
                # Log full matrix history to disk (every step, append JSONL)
                import json as _json
                _matrix_log_path = os.path.join(os.path.dirname(log_path), 'matrix_history.jsonl')
                _matrix_entry = {
                    'step': step, 'num_bits': _nb,
                    'input': _input_frame,
                    'output': _output_frame,
                    'lcx_before': _lcx_before if _lcx_before is not None else _lcx_vals.tolist(),
                    'lcx_after': _lcx_vals.tolist(),
                    'lcx_norm': _lcx_norm,
                }
                with open(_matrix_log_path, 'a') as _mf:
                    _mf.write(_json.dumps(_matrix_entry) + '\n')
                # Write JSON sidecar (latest state for dashboard)
                _lcx_sidecar = {
                    'step': step, 'num_bits': _nb, 'side': _nb,
                    'values': _lcx_vals.tolist(),
                }
                _sidecar_path = os.path.join(os.path.dirname(log_path), 'lcx_latest.json')
                with open(_sidecar_path, 'w') as _sf:
                    _json.dump(_lcx_sidecar, _sf)
            elif model.gem is not None:
                _gem_vals = model.gem.detach().cpu().tolist()
                influx_writer.log_gem(influx_run_id, step, _gem_vals)
                _gem_norm = sum(v**2 for v in _gem_vals) ** 0.5
                _gem_str = " ".join(f"{v:+.3f}" for v in _gem_vals)
                gem_line = f"  GEM[{_gem_str}] norm={_gem_norm:.4f}"
                log_file.write(gem_line + "\n")
                current_file.write(gem_line + "\n")
                # Expand GEM 1D → 8×8 grid (each row = one GEM cell, all cols same value)
                _gem_grid = []
                for _gv in _gem_vals:
                    _gem_grid.extend([_gv] * num_bits)
                influx_writer.log_frame_snapshot(influx_run_id, step, 'lcx_state', _gem_grid, side=num_bits)
                influx_writer.log_lcx(influx_run_id, step, _gem_grid, num_bits)
                # GEM drift (current - initial)
                if _gem_initial is not None:
                    _gem_drift_grid = []
                    for _gc, _gi in zip(_gem_vals, _gem_initial):
                        _gem_drift_grid.extend([_gc - _gi] * num_bits)
                    influx_writer.log_frame_snapshot(influx_run_id, step, 'lcx_drift', _gem_drift_grid, side=num_bits)

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
                            x_ev, y_ev, _ = generate_multitask_batch(
                                n_samples=batch_size, seq_len=seq_len, max_value=max_value,
                                seed=7777 + step * 100 + ei, num_bits=num_bits, task=args.task,
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
                            for k in ['eval_loss', 'overall_acc', 'bit_acc', 'byte_match', 'hamming',
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
                for k in ['eval_loss', 'overall_acc', 'bit_acc', 'byte_match', 'hamming',
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
                    n_bits=num_bits)
                influx_writer.log_bits(influx_run_id, step, eval_metrics_accum['per_bit_accs'])

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
