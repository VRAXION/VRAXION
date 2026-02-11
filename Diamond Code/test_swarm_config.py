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

from swarm_model import SwarmByteRingModel
from byte_data import byte_accuracy, bit_accuracy


def int_to_bits(x, num_bits=8):
    """Convert integer to num_bits-wide bit tensor."""
    bits = []
    for i in range(num_bits):
        bits.append((x >> i) & 1)
    return torch.tensor(bits, dtype=torch.float32)


def generate_multitask_batch(n_samples, seq_len=16, max_value=100, seed=None, num_bits=8):
    """Generate batch with mixed operations."""
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
        op_name = random.choice(['add', 'and', 'or', 'xor'])
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
    pred_bits = (output[:, position, :] > 0.5).float()
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
            accuracies[op_name] = position_accuracy(op_output, op_target, 2)
        else:
            accuracies[op_name] = 0.0
    return accuracies


def byte_match_accuracy(output, target, position):
    """Calculate exact byte match accuracy (all 8 bits correct)."""
    pred_bits = (output[:, position, :] > 0.5).float()
    matches = (pred_bits == target[:, position, :]).all(dim=-1).float()
    return matches.mean().item()


def bit_accuracy_at_position(output, target, position):
    """Calculate per-bit accuracy (may be misleading for imbalanced ops)."""
    pred_bits = (output[:, position, :] > 0.5).float()
    bit_matches = (pred_bits == target[:, position, :]).float()
    return bit_matches.mean().item()


def per_bit_accuracy_at_position(output, target, position):
    """Calculate accuracy for each bit position independently."""
    pred_bits = (output[:, position, :] > 0.5).float()
    target_bits = target[:, position, :]
    per_bit = (pred_bits == target_bits).float().mean(dim=0)  # [8]
    return per_bit.tolist()


def hamming_distance_at_position(output, target, position):
    """Calculate mean Hamming distance (# bit errors per byte)."""
    pred_bits = (output[:, position, :] > 0.5).float()
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
            pred_bits = (being_outputs_at_pos[being_idx, b, :] > 0.5).float()  # [8]
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
                pred_bits = (being_output > 0.5).float()  # [masked_B, 8]
                matches = (pred_bits == op_target[:, position, :]).all(dim=-1).float()
                being_accs_matrix[op_idx][being_idx] = matches.mean().item()

    # Compute std across beings for each operation
    specialization_scores = []
    for op_accs in being_accs_matrix:
        if any(acc > 0 for acc in op_accs):  # Only if we have data for this op
            std = torch.tensor(op_accs).std().item()
            specialization_scores.append(std)

    return sum(specialization_scores) / len(specialization_scores) if specialization_scores else 0.0


def save_checkpoint(model, optimizer, step, best_acc, checkpoint_dir, is_best=False):
    """Save training checkpoint."""
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
    parser.add_argument('--checkpoint_every', type=int, default=1000, help='Save checkpoint every N steps (default: 1000)')
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
    args = parser.parse_args()

    # Auto-switch combiner to 'masked' if bits_per_being specified
    if args.bits_per_being > 0 and args.combiner == 'mean':
        args.combiner = 'masked'

    embedding_dim = args.embedding
    depth = args.depth
    num_beings = args.num_beings
    num_steps = args.steps
    num_bits = args.num_bits

    print("="*70)
    print("SWARM CONFIG TEST: Multi-Task Benchmark")
    print("="*70)
    print()
    print(f"Configuration:")
    print(f"  Num bits: {num_bits}")
    print(f"  Embedding: {embedding_dim}D")
    print(f"  Depth: {depth} layers")
    print(f"  Num beings: {num_beings}")
    print(f"  Combiner: {args.combiner}")
    if args.combiner == 'ring_attention':
        print(f"  Entropy weight: {args.entropy_weight}")
        print(f"  Freeze gate steps: {args.freeze_gate_steps}")
    if args.bits_per_being > 0:
        print(f"  Bits per being: {args.bits_per_being}")
        print(f"  Min coverage: {args.min_coverage}")
        print(f"  Mask seed: {args.mask_seed}")
    print(f"  Steps: {num_steps:,}")
    print("="*70)
    print()

    # Create swarm model
    torch.manual_seed(42)
    model = SwarmByteRingModel(
        num_memory_positions=64,
        embedding_dim=embedding_dim,
        num_beings=num_beings,
        depth=depth,
        num_bits=num_bits,
        combiner_mode=args.combiner,
        bits_per_being=args.bits_per_being,
        min_coverage=args.min_coverage,
        mask_seed=args.mask_seed,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Print receptive field masks
    if args.bits_per_being > 0 and model.receptive_masks is not None:
        masks = model.receptive_masks
        print(f"\nReceptive Field Masks ({args.bits_per_being} bits per being):")
        if num_beings <= 20:
            for i in range(num_beings):
                bits = [j for j in range(num_bits) if masks[i, j] > 0.5]
                print(f"  Being {i}: bits {bits}")
        else:
            print(f"  ({num_beings} beings, too many to list)")
        cov = masks.sum(dim=0)
        print(f"  Coverage: min={int(cov.min().item())}, max={int(cov.max().item())}, "
              f"mean={cov.mean():.1f}")

    print()

    # Training setup
    seq_len = 16
    batch_size = 32
    max_value = 2 ** num_bits - 1  # Full bit range (all bits exercised)
    eval_interval = 50

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    # Load checkpoint if resuming
    start_step = 0
    best_overall = 0.0
    if args.resume:
        start_step, best_overall = load_checkpoint(args.resume, model, optimizer)
        print()

    # Fixed eval set (25 samples per operation = 100 total)
    eval_batches = []
    for op_seed in range(25):
        for op_idx in range(4):
            x, y, ops = generate_multitask_batch(
                n_samples=1, seq_len=seq_len, max_value=max_value,
                seed=9999 + op_seed * 4 + op_idx, num_bits=num_bits,
            )
            eval_batches.append((x, y, ops))

    x_eval = torch.cat([x for x, _, _ in eval_batches])
    y_eval = torch.cat([y for _, y, _ in eval_batches])
    ops_eval = torch.cat([ops for _, _, ops in eval_batches])

    print(f"Eval set: {len(x_eval)} samples (25 per operation)")
    print()

    # Log setup
    log_dir = Path(__file__).parent / "logs" / "swarm"
    log_dir.mkdir(parents=True, exist_ok=True)

    combiner_tag = f"_{args.combiner}" if args.combiner not in ('mean', 'masked') else ""
    rf_tag = f"_rf{args.bits_per_being}" if args.bits_per_being > 0 else ""
    bits_tag = f"_{num_bits}bit" if num_bits != 8 else ""
    config_name = f"{num_beings}beings_{embedding_dim}d_{depth}layers{bits_tag}{combiner_tag}{rf_tag}"
    log_path = log_dir / f"{config_name}.log"
    if log_path.exists():
        log_path.unlink()

    print(f"Log: {log_path}")
    print()
    print("Dashboard: http://localhost:8501")
    print(f"  Launch: python -m streamlit run diamond_dashboard.py -- --log logs/swarm/{config_name}.log")
    print("="*70)
    print()

    # Track best and convergence
    best_step = start_step if args.resume else 0
    converged = {op: -1 for op in ['add', 'and', 'or', 'xor']}

    start_time = time.time()

    with open(log_path, "w") as log_file:
        for step in range(start_step, num_steps):
            step_start = time.time()

            # Generate training batch
            x_train, y_train, _ = generate_multitask_batch(
                n_samples=batch_size, seq_len=seq_len, max_value=max_value,
                seed=42 + step + 1000000, num_bits=num_bits,
            )

            # Freeze-gate warmup: freeze being params, train only gate
            if args.combiner == 'ring_attention' and args.freeze_gate_steps > 0:
                in_freeze = (step - start_step) < args.freeze_gate_steps
                for name, param in model.named_parameters():
                    if name == 'gate_temperature':
                        param.requires_grad = True  # Always train gate temp
                    else:
                        param.requires_grad = not in_freeze

            # Train step
            optimizer.zero_grad()
            output = model(x_train)
            loss = nn.functional.binary_cross_entropy_with_logits(output, y_train)

            # Gate entropy regularizer (ring_attention only)
            if args.combiner == 'ring_attention' and args.entropy_weight > 0:
                entropy_loss = model._last_entropy_loss
                loss = loss + args.entropy_weight * entropy_loss

            loss.backward()
            optimizer.step()

            step_time = time.time() - step_start

            # Eval (ENHANCED with per-being diagnostics)
            if step % eval_interval == 0:
                model.eval()
                with torch.no_grad():
                    eval_output, eval_stats = model(x_eval, return_stats=True, return_being_outputs=True)

                    # Standard metrics
                    overall_acc = position_accuracy(eval_output, y_eval, 2)
                    op_accs = per_operation_accuracy(eval_output, y_eval, ops_eval)
                    jump_rate = eval_stats['jump_gate']
                    circular_spread = eval_stats['circular_spread']
                    output_disagreement = eval_stats['output_disagreement']

                    # Per-being metrics
                    being_outputs = eval_stats['being_outputs']  # [num_beings, B, 8] - final timestep only
                    being_accs = []
                    for i in range(num_beings):
                        # being_outputs[i] is now [T, B, 8]
                        # Transpose to [B, T, 8] to match position_accuracy signature
                        being_output_transposed = being_outputs[i].transpose(0, 1)
                        being_acc = position_accuracy(being_output_transposed, y_eval, 2)
                        being_accs.append(being_acc)

                    # Ensemble diagnostics
                    best_individual = max(being_accs) if being_accs else 0.0
                    ensemble_benefit = overall_acc - best_individual
                    oracle_acc = oracle_best_of_n_accuracy(being_outputs, y_eval, 2)
                    bit_oracle_acc = bit_oracle_accuracy(being_outputs, y_eval, 2)
                    specialization = compute_specialization(being_outputs, y_eval, ops_eval, 2)

                    # Gate stats (ring_attention only)
                    gate_weights_str = ""
                    gate_entropy_val = 0.0
                    gate_temp_val = 1.0
                    if 'gate_weights' in eval_stats:
                        gate_weights_str = " ".join([f"w{i}={w:.3f}" for i, w in enumerate(eval_stats['gate_weights'])])
                        gate_entropy_val = eval_stats.get('gate_entropy', 0.0)
                        gate_temp_val = eval_stats.get('gate_temperature', 1.0)

                    # Accuracy breakdown
                    bit_acc = bit_accuracy_at_position(eval_output, y_eval, 2)
                    byte_match = byte_match_accuracy(eval_output, y_eval, 2)
                    hamming = hamming_distance_at_position(eval_output, y_eval, 2)

                    # Per-bit accuracy (for receptive field analysis)
                    per_bit_accs = per_bit_accuracy_at_position(eval_output, y_eval, 2)

                    # Mask stats
                    mask_stats = {}
                    if 'min_bit_coverage' in eval_stats:
                        mask_stats['min_cov'] = eval_stats['min_bit_coverage']
                        mask_stats['max_cov'] = eval_stats['max_bit_coverage']
                        mask_stats['mask_div'] = eval_stats['mask_diversity']

                    # Spatial metrics
                    pointer_positions = eval_stats['pointer_positions_all']
                    coverage = len(set(pointer_positions)) / 64.0
                    position_counts = Counter(pointer_positions)
                    clustering = max(position_counts.values()) / num_beings if position_counts else 0.0

                    # Per-being jump rates
                    jump_rates = eval_stats['jump_rates_per_being']

                model.train()

                # Track best
                if overall_acc > best_overall:
                    best_overall = overall_acc
                    best_step = step

                # Track convergence
                for op_name, acc in op_accs.items():
                    if converged[op_name] == -1 and acc >= 0.95:
                        converged[op_name] = step
                        print(f"  >> {op_name.upper()} reached 95% at step {step}")

                # ENHANCED log line with all new metrics
                log_line = (
                    f"step {step} | loss {loss.item():.6f} | "
                    f"overall={overall_acc:.4f} bit_acc={bit_acc:.4f} byte_match={byte_match:.4f} hamming={hamming:.4f} | "
                    f"add={op_accs['add']:.4f} and={op_accs['and']:.4f} or={op_accs['or']:.4f} xor={op_accs['xor']:.4f} | "
                )

                # Per-being accuracies
                for i, being_acc in enumerate(being_accs):
                    log_line += f"being_{i}={being_acc:.4f} "

                log_line += (
                    f"oracle={oracle_acc:.4f} bit_oracle={bit_oracle_acc:.4f} ensemble_benefit={ensemble_benefit:+.4f} | "
                    f"circular_spread={circular_spread:.4f} coverage={coverage:.4f} clustering={clustering:.4f} | "
                )

                # Per-being jump rates
                for i, jump_rate_i in enumerate(jump_rates):
                    log_line += f"jump_{i}={jump_rate_i:.4f} "

                log_line += f"specialization={specialization:.4f}"

                # Per-bit accuracy
                per_bit_str = " ".join([f"bit{i}={a:.4f}" for i, a in enumerate(per_bit_accs)])
                log_line += f" | {per_bit_str}"

                # Mask stats (masked combiner only)
                if mask_stats:
                    log_line += f" | min_cov={mask_stats['min_cov']} max_cov={mask_stats['max_cov']} mask_div={mask_stats['mask_div']:.4f}"

                # Gate stats (ring_attention only)
                if gate_weights_str:
                    log_line += f" | {gate_weights_str} gate_entropy={gate_entropy_val:.3f} gate_temp={gate_temp_val:.3f}"

                log_line += f" | s_per_step={step_time:.3f}\n"

                log_file.write(log_line)
                log_file.flush()

                # Console every 500 steps (ENHANCED)
                if step % 500 == 0:
                    elapsed = time.time() - start_time
                    # Compact being display (max 5 shown, else summary)
                    if num_beings <= 5:
                        being_str = " ".join([f"b{i}={ba*100:4.0f}%" for i, ba in enumerate(being_accs)])
                    else:
                        best_b = max(being_accs)
                        worst_b = min(being_accs)
                        avg_b = sum(being_accs) / len(being_accs)
                        being_str = f"best={best_b*100:4.0f}% avg={avg_b*100:4.0f}% worst={worst_b*100:4.0f}%"
                    gate_str = f" | gate=[{gate_weights_str}] ent={gate_entropy_val:.2f}" if gate_weights_str else ""
                    mask_str = f" | rf={args.bits_per_being}b" if args.bits_per_being > 0 else ""
                    print(
                        f"step {step:5d} | loss {loss.item():.6f} | "
                        f"overall={overall_acc*100:5.1f}% | "
                        f"add={op_accs['add']*100:4.0f}% and={op_accs['and']*100:4.0f}% "
                        f"or={op_accs['or']*100:4.0f}% xor={op_accs['xor']*100:4.0f}% | "
                        f"{being_str} oracle={oracle_acc*100:4.0f}% bit_orc={bit_oracle_acc*100:4.0f}% | "
                        f"spec={specialization:.3f} ens_ben={ensemble_benefit:+.2f}{gate_str}{mask_str} | "
                        f"time={elapsed:.1f}s"
                    )

                # Save checkpoint periodically
                if step > 0 and step % args.checkpoint_every == 0:
                    is_best = (overall_acc >= best_overall)
                    save_checkpoint(model, optimizer, step, best_overall, args.checkpoint_dir, is_best=is_best)

    total_time = time.time() - start_time

    # Final eval
    model.eval()
    with torch.no_grad():
        eval_output, eval_stats = model(x_eval, return_stats=True)
        final_overall = position_accuracy(eval_output, y_eval, 2)
        final_op_accs = per_operation_accuracy(eval_output, y_eval, ops_eval)
        final_jump = eval_stats['jump_gate']
        final_pointer_spread = eval_stats['pointer_spread']
        final_output_disagreement = eval_stats['output_disagreement']

    print()
    print("="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"Config: {num_beings}× {embedding_dim}D beings, {depth} layers ({total_params:,} params)")
    print(f"Overall accuracy: {final_overall*100:.1f}% (best: {best_overall*100:.1f}% at step {best_step})")
    print()
    print("Per-operation:")
    for op_name in ['add', 'and', 'or', 'xor']:
        acc = final_op_accs[op_name]
        conv = converged[op_name]
        status = "PASS" if acc >= 0.95 else "FAIL"
        conv_str = f"step {conv}" if conv >= 0 else "Never"
        print(f"  {op_name.upper():4s}: {acc*100:5.1f}% | converged: {conv_str:>10s} | {status}")
    print()
    print(f"Final jump gate: {final_jump*100:.1f}%")
    print(f"Pointer spread: {final_pointer_spread:.2f} ({'diverse' if final_pointer_spread > 5 else 'synchronized'})")
    print(f"Output disagreement: {final_output_disagreement:.4f}")
    print(f"Training time: {total_time:.1f}s")
    print()

    # Pass/fail
    passed = final_overall >= 0.95
    if passed:
        print("STATUS: PASS - Swarm achieved multi-task competence")
    else:
        print(f"STATUS: FAIL - Only {final_overall*100:.1f}% accuracy")

    print("="*70)

    # Save results
    result = {
        'config': {
            'embedding_dim': embedding_dim,
            'depth': depth,
            'num_beings': num_beings,
            'parameters': total_params,
        },
        'final': {
            'overall_acc': final_overall,
            'add_acc': final_op_accs['add'],
            'and_acc': final_op_accs['and'],
            'or_acc': final_op_accs['or'],
            'xor_acc': final_op_accs['xor'],
            'jump_gate': final_jump,
            'pointer_spread': final_pointer_spread,
            'output_disagreement': final_output_disagreement,
            'best_overall': best_overall,
        },
        'convergence': converged,
        'training_time': total_time,
        'passed': passed,
    }

    result_path = log_dir / f"{config_name}_result.json"
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)

    # Save final checkpoint
    final_step = num_steps - 1 if not args.resume else start_step + (num_steps - start_step)
    save_checkpoint(model, optimizer, final_step, best_overall, args.checkpoint_dir, is_best=False)

    print(f"\nResults saved: {result_path}")


if __name__ == "__main__":
    main()
