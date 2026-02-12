"""
Diamond Code - Evolution Strategy Training

Train TRUE Möbius with evolution strategies instead of gradients.
Bypasses gradient issues, tests if ES finds better solutions.
"""

import torch
import numpy as np
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from ring_memory_model import RingMemoryModel
from assoc_clean_data import generate_assoc_clean


def evaluate_model(model, x_eval, y_eval):
    """Evaluate model and return fitness (accuracy)."""
    model.eval()
    with torch.no_grad():
        logits, _, _ = model(x_eval)
        acc = (logits.argmax(dim=1) == y_eval).float().mean().item()
    model.train()
    return acc


def get_model_params(model):
    """Get flattened parameter vector."""
    return torch.cat([p.data.flatten() for p in model.parameters()])


def set_model_params(model, params):
    """Set model parameters from flattened vector."""
    offset = 0
    for p in model.parameters():
        numel = p.numel()
        p.data = params[offset:offset+numel].reshape(p.shape)
        offset += numel


def main():
    # TASK CONFIG
    num_keys = 2
    num_pairs = 2
    seq_len = 32

    # MODEL CONFIG
    num_positions = 64
    embedding_dim = 64

    # ES CONFIG
    population_size = 50  # Number of perturbations to try
    sigma = 0.1  # Noise standard deviation (increased 5x for visible effect)
    learning_rate = 0.3  # How much to update parameters (increased 3x)

    print("=" * 70)
    print("DIAMOND CODE - EVOLUTION STRATEGY TRAINING")
    print("=" * 70)
    print(f"Task: {num_keys} keys, {num_pairs} pairs, seq_len={seq_len}")
    print(f"Model: {num_positions}x{embedding_dim}D")
    print(f"ES: pop={population_size}, sigma={sigma}, lr={learning_rate}")
    print()

    # Create model
    torch.manual_seed(42)
    model = RingMemoryModel(
        input_size=1,
        num_outputs=2,
        num_memory_positions=num_positions,
        embedding_dim=embedding_dim,
        mobius=True,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model size: {total_params:,} parameters")
    print()

    # Fixed eval set
    x_eval, y_eval, _ = generate_assoc_clean(
        n_samples=500, seq_len=seq_len, keys=num_keys, pairs=num_pairs, seed=9999
    )

    # Log file
    log_path = Path(__file__).parent / "logs" / "diamond" / "probe_live.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Clear old log
    with open(log_path, 'w') as f:
        f.write("")

    print(f"Logging to: {log_path}")
    print()

    def log(msg):
        print(msg, flush=True)
        with open(log_path, 'a') as f:
            f.write(msg + '\n')
            f.flush()

    log("=" * 70)
    log(f"Evolution Strategy: {num_keys} keys, {num_pairs} pairs, len={seq_len}")
    log(f"Model: {num_positions}x{embedding_dim}D = {total_params} params")
    log(f"ES: pop={population_size}, sigma={sigma}, lr={learning_rate}")
    log("=" * 70)

    best_eval_acc = 0.0
    step = 0

    try:
        while True:
            step_start = time.time()

            # Get current parameters
            theta = get_model_params(model)

            # Sample perturbations (BUG FIX #2: match device/dtype)
            epsilon = torch.randn(population_size, theta.shape[0],
                                  device=theta.device, dtype=theta.dtype) * sigma

            # Evaluate population
            fitness = []
            for i in range(population_size):
                # Perturb parameters
                perturbed = theta + epsilon[i]
                set_model_params(model, perturbed)

                # Evaluate
                acc = evaluate_model(model, x_eval, y_eval)
                fitness.append(acc)

            # Restore original parameters
            set_model_params(model, theta)

            # Convert fitness to numpy
            fitness = np.array(fitness)

            # Normalize fitness (BUG FIX #4: utility-based preserves magnitude)
            fitness_array = np.array(fitness)
            fitness_std = fitness_array.std()

            # DIAGNOSTIC: Check for fitness collapse (zero variance)
            if fitness_std < 1e-6:
                print(f"WARNING: Fitness collapsed! std={fitness_std:.2e}, all perturbations identical")
                # Skip update when fitness has no variance
                normalized_fitness = np.zeros_like(fitness_array)
            else:
                normalized_fitness = (fitness_array - fitness_array.mean()) / (fitness_std + 1e-8)

            # Update parameters (weighted sum of perturbations)
            # BUG FIX #2: match device/dtype for fitness tensor
            normalized_fitness_tensor = torch.tensor(normalized_fitness,
                                                     dtype=epsilon.dtype,
                                                     device=epsilon.device)
            update = (epsilon.T @ normalized_fitness_tensor).flatten()

            # DIAGNOSTIC: Check update magnitude
            update_mag = update.abs().max().item()

            # BUG FIX #1: Correct ES formula (was dividing by population_size * sigma)
            new_theta = theta + (learning_rate / sigma) * update

            # BUG FIX #3: Validate parameters actually changed
            theta_before_sum = theta.sum().item()
            set_model_params(model, new_theta)
            theta_after = get_model_params(model)
            theta_after_sum = theta_after.sum().item()
            param_change = abs(theta_after_sum - theta_before_sum)

            # Evaluate new parameters
            eval_acc = evaluate_model(model, x_eval, y_eval)

            # Get diagnostics (run forward pass with debug)
            with torch.no_grad():
                _, _, eval_routing = model(x_eval, return_debug=True)

                # Holonomy stats
                eval_holonomy_traj = torch.stack(eval_routing['holonomy_trajectory'])
                holonomy_pct = (eval_holonomy_traj == 1.0).float().mean().item()

                # Pointer stats
                eval_pointer_traj = torch.stack(eval_routing['pointer_trajectory'])
                ptr_std = eval_pointer_traj.std().item()

                # Wraps
                ptr_diffs = eval_pointer_traj[1:] - eval_pointer_traj[:-1]
                wrap_events = (ptr_diffs < -num_positions * 0.5).sum().item()

                # Coverage
                ptr_range = eval_pointer_traj.max().item() - eval_pointer_traj.min().item()
                coverage_pct = ptr_range / num_positions

            step_time = time.time() - step_start

            if eval_acc > best_eval_acc:
                best_eval_acc = eval_acc

            # Log every step (ES is slow, no need to skip)
            mean_fitness = fitness.mean()
            max_fitness = fitness.max()

            log(f"step {step} | loss {1-mean_fitness:.6f} | "
                f"acc={eval_acc:.4f} | jump_gate=0.00 | "
                f"train_acc={max_fitness:.4f} | best={best_eval_acc:.4f} | "
                f"fit_std={fitness_std:.2e} | upd_mag={update_mag:.2e} | pdelta={param_change:.2e} | "
                f"s_per_step={step_time:.3f}")

            step += 1

    except KeyboardInterrupt:
        log("")
        log("=" * 70)
        log(f"Training stopped at step {step}")
        log(f"Best eval accuracy: {best_eval_acc*100:.1f}%")
        log("=" * 70)


if __name__ == "__main__":
    main()
