"""
No-Backprop at Scale — v3 Perturbation vs Forward Gradient
============================================================
Tests both approaches on the real INSTNCT model (5.19M params).

1. Weight Perturbation (v3 approach): random nudge, keep if better
2. Forward Gradient: forward-mode directional derivative, no backprop
3. Backprop baseline: standard Adam

All on the same model, same data, same steps.
"""

import modal
import time

app = modal.App("vraxion-no-backprop-scale")

vraxion_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "pyyaml")
    .add_local_dir("/home/deck/work/vraxion/v4/model", remote_path="/root/vraxion/model")
    .add_local_dir("/home/deck/work/vraxion/v4/training", remote_path="/root/vraxion/training")
    .add_local_dir("/home/deck/work/vraxion/v4/config", remote_path="/root/vraxion/config")
    .add_local_dir("/home/deck/work/vraxion/v4/training_data", remote_path="/root/vraxion/training_data")
)


@app.function(gpu="T4", timeout=900, image=vraxion_image)
def run_experiment():
    import sys
    sys.path.insert(0, "/root/vraxion/training")
    sys.path.insert(0, "/root/vraxion/model")

    import torch
    import torch.nn.functional as F
    import numpy as np
    import random
    from pathlib import Path
    from model_factory import load_model_config, build_model_spec, build_model_from_spec
    from train import ByteDataset, func_discover_dat

    device = "cuda"
    v4_root = Path("/root/vraxion")
    model_config = load_model_config(v4_root)
    training_config = {"embed_mode": True}
    model_record = build_model_spec("instnct", True, model_config, training_config)

    files = func_discover_dat(str(v4_root / "training_data"))

    BATCH = 8
    SEQ = 128
    NUM_STEPS = 100
    LR = 1e-3

    def compute_loss(model, dataset):
        """Forward pass, return loss value (detached). Used as global signal."""
        x, y, mask = dataset.sample_batch(BATCH, device)
        with torch.no_grad():
            pred, _ = model(x, state=None)
        per_pos = F.cross_entropy(pred.transpose(1, 2), y, reduction="none")
        return (per_pos * mask).sum() / mask.sum().clamp(min=1)

    def compute_accuracy(model, dataset):
        x, y, mask = dataset.sample_batch(BATCH, device)
        with torch.no_grad():
            pred, _ = model(x, state=None)
        correct = (pred.argmax(-1) == y).float() * mask
        return correct.sum() / mask.sum().clamp(min=1)

    results = {}

    # ═══════════════════════════════════════════
    #  1. WEIGHT PERTURBATION (v3 approach)
    # ═══════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  1. Weight Perturbation (v3 — random nudge)")
    print(f"{'='*60}")

    torch.manual_seed(1337)
    model = build_model_from_spec(model_record, device=device).eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {total_params:,}")

    dataset = ByteDataset(files, seq_len=SEQ, embed_mode=True, seed=42)
    losses_wp = []
    kept = 0
    t0 = time.time()

    for step in range(NUM_STEPS):
        # Current global signal
        current_loss = compute_loss(model, dataset).item()

        # Random perturbation
        noise = {}
        for name, p in model.named_parameters():
            noise[name] = torch.randn_like(p) * 0.001

        # Apply
        for name, p in model.named_parameters():
            p.data.add_(noise[name])

        # New global signal
        new_loss = compute_loss(model, dataset).item()

        if new_loss < current_loss:
            kept += 1  # keep
        else:
            # Revert
            for name, p in model.named_parameters():
                p.data.sub_(noise[name])
            new_loss = current_loss

        losses_wp.append(new_loss)

        if (step + 1) % 20 == 0:
            acc = compute_accuracy(model, dataset).item() * 100
            elapsed = time.time() - t0
            print(f"  Step {step+1:3d} | Loss: {new_loss:.4f} | "
                  f"Acc: {acc:.2f}% | Kept: {kept}/{step+1} | {elapsed:.1f}s")

    wp_time = time.time() - t0
    results["weight_perturbation"] = {
        "final_loss": losses_wp[-1],
        "kept": kept,
        "time": wp_time,
    }

    # ═══════════════════════════════════════════
    #  2. FORWARD GRADIENT (no backprop)
    # ═══════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  2. Forward Gradient (directional derivative)")
    print(f"{'='*60}")

    torch.manual_seed(1337)
    model = build_model_from_spec(model_record, device=device).train()
    dataset = ByteDataset(files, seq_len=SEQ, embed_mode=True, seed=42)
    losses_fg = []
    t0 = time.time()

    for step in range(NUM_STEPS):
        x, y, mask = dataset.sample_batch(BATCH, device)

        # Random direction vector (same shape as all params)
        v = {name: torch.randn_like(p) for name, p in model.named_parameters()}

        # Forward pass with dual numbers (forward-mode AD via finite diff)
        # f(w + εv) ≈ f(w) + ε * <∇f, v>
        # directional_derivative ≈ (f(w + εv) - f(w)) / ε
        eps = 1e-4

        # f(w)
        with torch.no_grad():
            pred0, _ = model(x, state=None)
            loss0 = F.cross_entropy(pred0.transpose(1, 2), y, reduction="none")
            loss0 = (loss0 * mask).sum() / mask.sum().clamp(min=1)

        # f(w + εv)
        for name, p in model.named_parameters():
            p.data.add_(eps * v[name])
        with torch.no_grad():
            pred1, _ = model(x, state=None)
            loss1 = F.cross_entropy(pred1.transpose(1, 2), y, reduction="none")
            loss1 = (loss1 * mask).sum() / mask.sum().clamp(min=1)

        # Revert perturbation
        for name, p in model.named_parameters():
            p.data.sub_(eps * v[name])

        # Directional derivative
        dir_deriv = (loss1.item() - loss0.item()) / eps

        # Update: w -= lr * dir_deriv * v
        # (This is an unbiased gradient estimator)
        with torch.no_grad():
            for name, p in model.named_parameters():
                p.data.sub_(LR * dir_deriv * v[name])

        losses_fg.append(loss0.item())

        if (step + 1) % 20 == 0:
            acc = compute_accuracy(model, dataset).item() * 100
            elapsed = time.time() - t0
            print(f"  Step {step+1:3d} | Loss: {loss0.item():.4f} | "
                  f"Acc: {acc:.2f}% | dir_deriv: {dir_deriv:.4f} | {elapsed:.1f}s")

    fg_time = time.time() - t0
    results["forward_gradient"] = {
        "final_loss": losses_fg[-1],
        "time": fg_time,
    }

    # ═══════════════════════════════════════════
    #  3. BACKPROP BASELINE (Adam)
    # ═══════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  3. Backprop Baseline (Adam)")
    print(f"{'='*60}")

    torch.manual_seed(1337)
    model = build_model_from_spec(model_record, device=device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    dataset = ByteDataset(files, seq_len=SEQ, embed_mode=True, seed=42)
    losses_bp = []
    t0 = time.time()

    for step in range(NUM_STEPS):
        x, y, mask = dataset.sample_batch(BATCH, device)

        model.zero_grad()
        pred, _ = model(x, state=None)
        per_pos = F.cross_entropy(pred.transpose(1, 2), y, reduction="none")
        loss = (per_pos * mask).sum() / mask.sum().clamp(min=1)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()

        losses_bp.append(loss.item())

        if (step + 1) % 20 == 0:
            acc = compute_accuracy(model, dataset).item() * 100
            elapsed = time.time() - t0
            print(f"  Step {step+1:3d} | Loss: {loss.item():.4f} | "
                  f"Acc: {acc:.2f}% | {elapsed:.1f}s")

    bp_time = time.time() - t0
    results["backprop"] = {
        "final_loss": losses_bp[-1],
        "time": bp_time,
    }

    # ═══════════════════════════════════════════
    #  SUMMARY
    # ═══════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  RESULTS — 5.19M param INSTNCT model, {NUM_STEPS} steps")
    print(f"{'='*60}")
    print(f"")
    print(f"  Weight Perturbation (v3):")
    print(f"    Final loss: {results['weight_perturbation']['final_loss']:.4f}")
    print(f"    Kept: {results['weight_perturbation']['kept']}/{NUM_STEPS}")
    print(f"    Time: {results['weight_perturbation']['time']:.1f}s")
    print(f"")
    print(f"  Forward Gradient:")
    print(f"    Final loss: {results['forward_gradient']['final_loss']:.4f}")
    print(f"    Time: {results['forward_gradient']['time']:.1f}s")
    print(f"")
    print(f"  Backprop (Adam):")
    print(f"    Final loss: {results['backprop']['final_loss']:.4f}")
    print(f"    Time: {results['backprop']['time']:.1f}s")

    return results


@app.local_entrypoint()
def main():
    print("Running no-backprop scaling test on Modal GPU...")
    results = run_experiment.remote()
    print("\nDone!")
