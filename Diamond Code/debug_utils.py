"""Debugging utilities for RingMemoryModel."""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple


def visualize_pointer_trajectory(debug_info: dict, save_path: Optional[str] = None) -> None:
    """
    Plot pointer position over time.

    Args:
        debug_info: Dict from model(x, return_debug=True)
        save_path: Optional path to save figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available - skipping visualization")
        return

    trajectory = [p.numpy() for p in debug_info["pointer_trajectory"]]
    trajectory = np.array(trajectory)  # [T, B]

    plt.figure(figsize=(12, 4))
    for batch_idx in range(min(5, trajectory.shape[1])):
        plt.plot(trajectory[:, batch_idx], alpha=0.7, label=f"Sample {batch_idx}")

    plt.xlabel("Timestep")
    plt.ylabel("Pointer Position")
    plt.title("Pointer Trajectory")
    plt.legend()
    plt.grid(alpha=0.3)

    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def check_gradient_magnitudes(model) -> None:
    """
    Print gradient norms for each parameter.

    Args:
        model: RingMemoryModel instance
    """
    print("\nGradient Magnitudes:")
    print("-" * 60)
    for name, param in model.named_parameters():
        if param.grad is not None:
            norm = param.grad.norm().item()
            print(f"{name:40s} {norm:12.6f}")
        else:
            print(f"{name:40s} {'NO GRAD':>12s}")


def inspect_memory_ring(memory_ring: torch.Tensor) -> None:
    """
    Analyze memory ring statistics.

    Args:
        memory_ring: [B, num_positions, embed_dim] tensor
    """
    print("\nMemory Ring Statistics:")
    print("-" * 60)
    print(f"Shape: {memory_ring.shape}")
    print(f"Mean: {memory_ring.mean().item():.6f}")
    print(f"Std: {memory_ring.std().item():.6f}")
    print(f"Min: {memory_ring.min().item():.6f}")
    print(f"Max: {memory_ring.max().item():.6f}")
    print(f"Sparsity (|x| < 0.01): {(memory_ring.abs() < 0.01).float().mean().item()*100:.1f}%")


def print_model_summary(model):
    """
    Print model architecture summary.

    Args:
        model: RingMemoryModel instance
    """
    print("\nModel Summary:")
    print("=" * 70)
    print(f"Input size: {model.input_size}")
    print(f"Output size: {model.num_outputs}")
    print(f"Memory positions: {model.num_memory_positions}")
    print(f"Embedding dim: {model.embedding_dim}")
    print(f"Attention radius: {model.attention_radius}")
    print(f"Attention temperature: {model.attention_temperature}")
    print(f"Activation: {model.activation_name}")
    print()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("=" * 70)


def trace_forward_pass(model, x: torch.Tensor, verbose: bool = True) -> Tuple[torch.Tensor, float, Dict[str, Any]]:
    """
    Trace a single forward pass with detailed logging.

    Args:
        model: RingMemoryModel instance
        x: Input tensor [B, T, input_size]
        verbose: If True, print detailed info

    Returns:
        logits: Output tensor [B, num_outputs]
        aux_loss: Auxiliary loss value
        debug: Debug information dict
    """
    if verbose:
        print("\n" + "="*70)
        print("Forward Pass Trace")
        print("="*70)
        print(f"Input shape: {x.shape}")
        print()

    logits, aux_loss, debug = model(x, return_debug=True)

    if verbose:
        print(f"Output shape: {logits.shape}")
        print(f"Aux loss: {aux_loss:.6f}")
        print()
        print(f"Pointer trajectory: {len(debug['pointer_trajectory'])} steps")
        print(f"Attention entropy: min={min(debug['attention_entropy']):.4f}, max={max(debug['attention_entropy']):.4f}")
        print("="*70)

    return logits, aux_loss, debug
