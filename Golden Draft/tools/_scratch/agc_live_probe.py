"""AGC Live Test Probe - Dashboard-compatible training run

Runs a simple training loop on assoc_clean with AGC enabled,
writing dashboard-compatible logs to visualize gradient normalization.
"""

import sys
import os
import time
from pathlib import Path

# Add Golden Code to path
sys.path.insert(0, "S:/AI/Golden Code")

import torch
import torch.nn as nn
import torch.optim as optim

from vraxion.instnct.agc import AGCParams, apply_update_agc
from vraxion.instnct.absolute_hallway import AbsoluteHallway

# Simple synthetic task: learn to echo input
def generate_batch(batch_size=16, seq_len=10):
    """Generate simple associative memory task"""
    # Random sequences
    keys = torch.randint(0, 10, (batch_size, seq_len))
    values = torch.randint(0, 10, (batch_size, seq_len))

    # Targets: echo the value at a random key position
    targets = torch.zeros(batch_size, dtype=torch.long)
    for i in range(batch_size):
        key_pos = torch.randint(0, seq_len, (1,)).item()
        targets[i] = values[i, key_pos]

    # One-hot encode
    inputs = torch.cat([
        torch.nn.functional.one_hot(keys, 10).float(),
        torch.nn.functional.one_hot(values, 10).float()
    ], dim=-1)  # [batch, seq, 20]

    return inputs, targets


def main():
    print("=" * 60)
    print("AGC Live Probe - Training with Dashboard Feed")
    print("=" * 60)
    print()
    print("Dashboard: http://localhost:8501")
    print("Log file:  logs/probe/probe_live.log")
    print()

    # Setup
    device = "cpu"
    log_path = Path("S:/AI/work/VRAXION_DEV/Golden Draft/logs/probe/probe_live.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Clear log file
    log_path.write_text("")

    # Model
    model = AbsoluteHallway(
        input_dim=20,
        num_classes=10,
        ring_len=64,
        slot_dim=64,
    ).to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # AGC
    agc_params = AGCParams(enabled=True, grad_low=1.0, grad_high=5.0)

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Training loop
    num_steps = 100
    start_time = time.time()

    print(f"Training for {num_steps} steps with AGC enabled...")
    print(f"AGC range: [{agc_params.grad_low}, {agc_params.grad_high}]")
    print()

    for step in range(1, num_steps + 1):
        # Generate batch
        inputs, targets = generate_batch(batch_size=16, seq_len=10)
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Forward
        optimizer.zero_grad()
        logits, move_penalty = model(inputs)

        # logits is [B, num_classes]
        loss = criterion(logits, targets)

        # Backward
        loss.backward()

        # Compute full model gradient norm BEFORE AGC
        total_grad_norm = 0.0
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2).item()
                    total_grad_norm += param_norm ** 2
            total_grad_norm = total_grad_norm ** 0.5

        # AGC: Normalize gradients
        agc_scale = apply_update_agc(
            model, total_grad_norm, agc_params,
            step=step, log_fn=lambda msg: None  # Suppress AGC logs for now
        )

        # Apply normalization to gradients
        if agc_scale != 1.0:
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.data.mul_(agc_scale)

        # Compute gradient norm AFTER AGC (for logging)
        normalized_grad_norm = total_grad_norm * agc_scale

        # Optimizer step
        optimizer.step()

        # Compute accuracy
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            acc = (preds == targets).float().mean().item()

        # Timing
        elapsed = time.time() - start_time
        s_per_step = elapsed / step

        # Dashboard-compatible log line
        # Format: step N | loss X.XXXXXX | acc=X.XXXX RD:X.XXXX traction=X.XXXX shard=0/0
        log_line = f"step {step} | loss {loss.item():.6f} | acc={acc:.4f} RD:{s_per_step:.4f} traction={acc:.4f} shard=0/0 | grad_raw={total_grad_norm:.2f} grad_norm={normalized_grad_norm:.2f} agc_scale={agc_scale:.4f}\n"

        # Write to log file (append)
        with log_path.open("a") as f:
            f.write(log_line)

        # Console output every 10 steps
        if step % 10 == 0 or step == 1:
            agc_status = "SCALED" if agc_scale != 1.0 else "OK"
            print(f"step {step:3d} | loss {loss.item():.4f} | acc {acc:.4f} | "
                  f"grad_raw {total_grad_norm:6.2f} -> {normalized_grad_norm:5.2f} [{agc_status}] | "
                  f"{s_per_step:.3f}s/step")

            # Log AGC action if it triggered
            if agc_scale < 1.0:
                print(f"         | AGC scaled DOWN by {agc_scale:.4f} (grad too high)")
            elif agc_scale > 1.0:
                print(f"         | AGC scaled UP by {agc_scale:.4f} (grad too low)")

    print()
    print("=" * 60)
    print(f"Training complete: {num_steps} steps in {elapsed:.1f}s ({s_per_step:.3f}s/step)")
    print(f"Log written to: {log_path}")
    print(f"Dashboard: http://localhost:8501")
    print("=" * 60)


if __name__ == "__main__":
    main()
