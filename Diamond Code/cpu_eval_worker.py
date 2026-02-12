"""
CPU Eval Worker — Court-grade async evaluation for Diamond Code swarm training.

Watches checkpoint directory for new checkpoints, loads each on CPU,
runs heavyweight eval (500 samples, all metrics), writes to eval_court.log.

Architecture:
    GPU (training loop) saves checkpoints every N steps with train_bacc pulse eval.
    This worker catches up asynchronously, producing precise "court-grade" metrics.

Launch alongside training:
    python cpu_eval_worker.py --checkpoint_dir checkpoints/swarm --data_dir data/traindat/

Dashboard reads both:
    - logs/swarm/current.log (GPU pulse: loss + train_bacc every step)
    - logs/swarm/eval_court.log (CPU court: all metrics every checkpoint)
"""

import os
import sys
import time
import glob
import shutil
import argparse
import re
import torch
import torch.nn as nn
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))

from swarm_model import SwarmByteRingModel, fibonacci_split
from traindat_loader import TraindatLoader
from test_swarm_config import (
    position_accuracy,
    bit_accuracy_at_position,
    per_bit_accuracy_at_position,
    byte_match_accuracy,
    hamming_distance_at_position,
    oracle_best_of_n_accuracy,
    bit_oracle_accuracy,
    compute_specialization,
)


def build_model_from_config(config):
    """Recreate a SwarmByteRingModel from checkpoint config dict."""
    memory_size = config.get('memory_size', config['embedding_dim'])

    model = SwarmByteRingModel(
        num_memory_positions=memory_size,
        embedding_dim=config['embedding_dim'],
        num_beings=config['num_beings'],
        depth=config['depth'],
        num_bits=config['num_bits'],
        combiner_mode=config.get('combiner', 'masked'),
        bits_per_being=config.get('bits_per_being', 8),
        min_coverage=config.get('min_coverage', 1),
        mask_seed=config.get('mask_seed', 42),
        fibonacci=config.get('fibonacci', True),
        think_ticks=0,  # Eval doesn't need think_ticks from training
        temporal_fibonacci=config.get('temporal_fibonacci', False),
        capacity_fibonacci=config.get('capacity_fibonacci', False),
        max_hidden=config.get('max_hidden', 4096),
        min_hidden=config.get('min_hidden', 128),
    )
    return model


def generate_eval_data(config, n_samples, seed=9999):
    """Generate eval data matching training format."""
    data_dir = config.get('data_dir')
    num_bits = config['num_bits']
    seq_len = config.get('seq_len', 128)

    if data_dir:
        loader = TraindatLoader(data_dir)
        x_eval, y_eval = loader.sample_batch(
            n_samples=n_samples, seq_len=seq_len, num_bits=num_bits, seed=seed
        )
        return x_eval, y_eval, None
    else:
        # Fallback: math mode (unlikely for current usage)
        from test_swarm_config import generate_multitask_batch
        max_value = 2 ** num_bits - 1
        samples_per_op = max(1, n_samples // 4)
        eval_batches = []
        for op_seed in range(samples_per_op):
            for op_idx in range(4):
                x, y, ops = generate_multitask_batch(
                    n_samples=1, seq_len=seq_len, max_value=max_value,
                    seed=seed + op_seed * 4 + op_idx, num_bits=num_bits,
                )
                eval_batches.append((x, y, ops))
        x_eval = torch.cat([x for x, _, _ in eval_batches])
        y_eval = torch.cat([y for _, y, _ in eval_batches])
        ops_eval = torch.cat([ops for _, _, ops in eval_batches])
        return x_eval, y_eval, ops_eval


def evaluate_checkpoint(model, x_eval, y_eval, ops_eval, num_beings, num_bits):
    """Run full court-grade evaluation. Returns dict of all metrics."""
    model.eval()
    eval_pos = min(2, x_eval.size(1) - 1)

    with torch.no_grad():
        eval_output, eval_stats = model(x_eval, return_stats=True, return_being_outputs=True)

        # Core metrics
        loss = nn.functional.binary_cross_entropy_with_logits(eval_output, y_eval).item()
        overall_acc = position_accuracy(eval_output, y_eval, eval_pos)
        bit_acc = bit_accuracy_at_position(eval_output, y_eval, eval_pos)
        byte_match = byte_match_accuracy(eval_output, y_eval, eval_pos)
        hamming = hamming_distance_at_position(eval_output, y_eval, eval_pos)

        # Per-bit accuracy (256 bits)
        per_bit_accs = per_bit_accuracy_at_position(eval_output, y_eval, eval_pos)

        # Per-being metrics
        being_outputs = eval_stats['being_outputs']
        being_accs = []
        for i in range(num_beings):
            being_output_transposed = being_outputs[i].transpose(0, 1)
            being_acc = position_accuracy(being_output_transposed, y_eval, eval_pos)
            being_accs.append(being_acc)

        # Ensemble diagnostics
        best_individual = max(being_accs) if being_accs else 0.0
        ensemble_benefit = overall_acc - best_individual
        oracle_acc = oracle_best_of_n_accuracy(being_outputs, y_eval, eval_pos)
        bit_oracle_acc = bit_oracle_accuracy(being_outputs, y_eval, eval_pos)
        specialization = compute_specialization(being_outputs, y_eval, ops_eval, eval_pos) if ops_eval is not None else 0.0

        # Operation accuracies (traindat mode: ops_eval is None)
        op_accs = {'add': 0, 'and': 0, 'or': 0, 'xor': 0}

        # Spatial metrics
        pointer_positions = eval_stats['pointer_positions_all']
        coverage = len(set(pointer_positions)) / 64.0
        position_counts = Counter(pointer_positions)
        clustering = max(position_counts.values()) / num_beings if position_counts else 0.0
        circular_spread = eval_stats['circular_spread']

        # Per-being jump rates
        jump_rates = eval_stats['jump_rates_per_being']

        # Mask stats
        mask_stats = {}
        if 'min_bit_coverage' in eval_stats:
            mask_stats['min_cov'] = eval_stats['min_bit_coverage']
            mask_stats['max_cov'] = eval_stats['max_bit_coverage']
            mask_stats['mask_div'] = eval_stats['mask_diversity']

        # Temporal fibonacci
        activation_ratio = eval_stats.get('activation_ratio', None)

    return {
        'loss': loss,
        'overall_acc': overall_acc,
        'bit_acc': bit_acc,
        'byte_match': byte_match,
        'hamming': hamming,
        'per_bit_accs': per_bit_accs,
        'being_accs': being_accs,
        'oracle_acc': oracle_acc,
        'bit_oracle_acc': bit_oracle_acc,
        'ensemble_benefit': ensemble_benefit,
        'specialization': specialization,
        'op_accs': op_accs,
        'circular_spread': circular_spread,
        'coverage': coverage,
        'clustering': clustering,
        'jump_rates': jump_rates,
        'mask_stats': mask_stats,
        'activation_ratio': activation_ratio,
    }


def format_log_line(step, metrics):
    """Format metrics into the same log line format as the training loop."""
    m = metrics
    log_line = (
        f"step {step} | loss {m['loss']:.6f} | "
        f"overall={m['overall_acc']:.4f} bit_acc={m['bit_acc']:.4f} "
        f"byte_match={m['byte_match']:.4f} hamming={m['hamming']:.4f} | "
        f"add={m['op_accs']['add']:.4f} and={m['op_accs']['and']:.4f} "
        f"or={m['op_accs']['or']:.4f} xor={m['op_accs']['xor']:.4f} | "
    )

    # Per-being accuracies
    for i, ba in enumerate(m['being_accs']):
        log_line += f"being_{i}={ba:.4f} "

    log_line += (
        f"oracle={m['oracle_acc']:.4f} bit_oracle={m['bit_oracle_acc']:.4f} "
        f"ensemble_benefit={m['ensemble_benefit']:+.4f} | "
        f"circular_spread={m['circular_spread']:.4f} "
        f"coverage={m['coverage']:.4f} clustering={m['clustering']:.4f} | "
    )

    # Per-being jump rates
    for i, jr in enumerate(m['jump_rates']):
        log_line += f"jump_{i}={jr:.4f} "

    log_line += f"specialization={m['specialization']:.4f}"

    # Per-bit accuracy
    per_bit_str = " ".join([f"bit{i}={a:.4f}" for i, a in enumerate(m['per_bit_accs'])])
    log_line += f" | {per_bit_str}"

    # Mask stats
    if m['mask_stats']:
        ms = m['mask_stats']
        log_line += f" | min_cov={ms['min_cov']} max_cov={ms['max_cov']} mask_div={ms['mask_div']:.4f}"

    # Temporal fibonacci
    if m['activation_ratio'] is not None:
        log_line += f" | active_ratio={m['activation_ratio']:.3f}"

    return log_line


def find_checkpoints(checkpoint_dir):
    """Find all checkpoint_step_N.pt files, return sorted by step number."""
    pattern = os.path.join(checkpoint_dir, "checkpoint_step_*.pt")
    files = glob.glob(pattern)
    step_files = []
    for f in files:
        m = re.search(r"checkpoint_step_(\d+)\.pt$", f)
        if m:
            step_files.append((int(m.group(1)), f))
    step_files.sort(key=lambda x: x[0])
    return step_files


def main():
    parser = argparse.ArgumentParser(description='CPU Eval Worker — court-grade async evaluation')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/swarm',
                        help='Checkpoint directory to watch')
    parser.add_argument('--data_dir', type=str, default='data/traindat/',
                        help='Data directory (override checkpoint config)')
    parser.add_argument('--eval_samples', type=int, default=500,
                        help='Number of eval samples (default: 500 for court-grade)')
    parser.add_argument('--log_dir', type=str, default='logs/swarm',
                        help='Directory for eval_court.log')
    parser.add_argument('--poll_interval', type=int, default=5,
                        help='Seconds between checkpoint scans (default: 5)')
    parser.add_argument('--one_shot', action='store_true',
                        help='Evaluate all existing checkpoints and exit (no watching)')
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    court_log_path = log_dir / "eval_court.log"

    print("=" * 70)
    print("CPU EVAL WORKER — Court-Grade Async Evaluation")
    print("=" * 70)
    print(f"Watching: {args.checkpoint_dir}")
    print(f"Eval samples: {args.eval_samples}")
    print(f"Court log: {court_log_path}")
    print(f"Poll interval: {args.poll_interval}s")
    print("=" * 70)
    print()

    evaluated_steps = set()
    best_bit_acc = 0.0
    model = None
    config = None
    x_eval = y_eval = ops_eval = None

    # Load already-evaluated steps from existing log
    if court_log_path.exists():
        with open(court_log_path, 'r') as f:
            for line in f:
                m = re.match(r"step\s+(\d+)\s+\|", line)
                if m:
                    evaluated_steps.add(int(m.group(1)))
        if evaluated_steps:
            print(f"Resuming: {len(evaluated_steps)} steps already evaluated (up to step {max(evaluated_steps)})")

    while True:
        checkpoints = find_checkpoints(args.checkpoint_dir)
        new_checkpoints = [(step, path) for step, path in checkpoints if step not in evaluated_steps]

        if not new_checkpoints:
            if args.one_shot:
                print("One-shot mode: all checkpoints evaluated. Exiting.")
                break
            time.sleep(args.poll_interval)
            continue

        for step, ckpt_path in new_checkpoints:
            eval_start = time.time()

            try:
                # Load checkpoint
                checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
                ckpt_config = checkpoint.get('config')

                if ckpt_config is None:
                    print(f"  [SKIP] step {step}: no config in checkpoint (old format)")
                    evaluated_steps.add(step)
                    continue

                # Override data_dir if specified
                if args.data_dir:
                    ckpt_config['data_dir'] = args.data_dir

                # Rebuild model if config changed
                if config is None or ckpt_config != config:
                    config = ckpt_config
                    model = build_model_from_config(config)
                    print(f"  [MODEL] Built {sum(p.numel() for p in model.parameters()):,} param model on CPU")

                    # Generate eval data (once per config)
                    x_eval, y_eval, ops_eval = generate_eval_data(config, args.eval_samples)
                    print(f"  [DATA] {args.eval_samples} eval samples, seq_len={config.get('seq_len', 128)}")

                # Load weights
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)

                # Run full eval
                metrics = evaluate_checkpoint(
                    model, x_eval, y_eval, ops_eval,
                    config['num_beings'], config['num_bits']
                )

                # Write to court log
                log_line = format_log_line(step, metrics)
                with open(court_log_path, 'a') as f:
                    f.write(log_line + "\n")

                eval_time = time.time() - eval_start

                # Track best
                if metrics['bit_acc'] > best_bit_acc:
                    best_bit_acc = metrics['bit_acc']
                    # Copy to best_model.pt
                    best_path = Path(args.checkpoint_dir) / 'best_model.pt'
                    shutil.copy2(ckpt_path, best_path)
                    print(f"  [COURT] step {step:5d} | bit_acc={metrics['bit_acc']:.4f} "
                          f"oracle={metrics['oracle_acc']:.4f} | {eval_time:.1f}s | NEW BEST")
                else:
                    print(f"  [COURT] step {step:5d} | bit_acc={metrics['bit_acc']:.4f} "
                          f"oracle={metrics['oracle_acc']:.4f} | {eval_time:.1f}s")

                evaluated_steps.add(step)

            except Exception as e:
                print(f"  [ERROR] step {step}: {e}")
                evaluated_steps.add(step)

        if args.one_shot:
            print(f"\nOne-shot complete: {len(evaluated_steps)} checkpoints evaluated.")
            break

        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
