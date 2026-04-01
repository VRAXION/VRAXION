"""
Random Hessian Search (RHS) — Proof of Concept
================================================
Compares standard Adam vs Adam + periodic random Hessian escape.

Runs on Modal with GPU.
"""

import modal
import time

app = modal.App("vraxion-rhs-experiment")

# Build image with vraxion code baked in
vraxion_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "pyyaml")
    .add_local_dir(
        "/home/deck/work/vraxion/v4/model",
        remote_path="/root/vraxion/model",
    )
    .add_local_dir(
        "/home/deck/work/vraxion/v4/training",
        remote_path="/root/vraxion/training",
    )
    .add_local_dir(
        "/home/deck/work/vraxion/v4/config",
        remote_path="/root/vraxion/config",
    )
    .add_local_dir(
        "/home/deck/work/vraxion/v4/training_data",
        remote_path="/root/vraxion/training_data",
    )
)


@app.function(
    gpu="T4",
    timeout=900,
    image=vraxion_image,
)
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
    torch.manual_seed(1337)
    random.seed(1337)
    np.random.seed(1337)

    # ── Load config & build model ──
    v4_root = Path("/root/vraxion")
    model_config = load_model_config(v4_root)
    training_config = {"embed_mode": True}

    model_record = build_model_spec("instnct", True, model_config, training_config)

    # ── Load data ──
    files = func_discover_dat(str(v4_root / "training_data"))
    dataset = ByteDataset(files, seq_len=256, embed_mode=True, seed=42)

    # ── Params ──
    NUM_STEPS = 200
    BATCH_SIZE = 16
    LR = 1e-3
    RHS_EVERY = 100
    RHS_BUDGET = 30      # 30 params = 30 forward+backward passes, ~45s
    RHS_LR = 0.01
    GRAD_CLIP = 10.0

    # ═══════════════════════════════════════════════
    #  Random Hessian Search functions
    # ═══════════════════════════════════════════════

    def get_trainable_params(model):
        """Get only parameters that require grad."""
        return [p for p in model.parameters() if p.requires_grad]

    def flatten_params(model):
        return torch.cat([p.data.view(-1) for p in get_trainable_params(model)])

    def unflatten_params(model, flat):
        offset = 0
        for p in get_trainable_params(model):
            numel = p.numel()
            p.data.copy_(flat[offset:offset + numel].view(p.shape))
            offset += numel

    def compute_loss(model, x, y, mask):
        pred, _ = model(x, state=None)
        per_pos = F.cross_entropy(pred.transpose(1, 2), y, reduction="none")
        masked = per_pos * mask
        return masked.sum() / mask.sum().clamp(min=1)

    def get_flat_grad(model):
        grads = []
        for p in get_trainable_params(model):
            if p.grad is not None:
                grads.append(p.grad.view(-1))
            else:
                grads.append(torch.zeros(p.numel(), device=p.device))
        return torch.cat(grads)

    def random_hessian_escape(model, x, y, mask, budget=200, eps=1e-4):
        """
        Sample `budget` random parameters, compute sub-Hessian via
        finite-difference on gradients, find negative eigenvalue
        directions, return escape direction (or None).
        """
        flat = flatten_params(model)
        total_params = flat.numel()

        indices = sorted(random.sample(range(total_params), min(budget, total_params)))

        # Base gradient
        model.zero_grad()
        loss_base = compute_loss(model, x, y, mask)
        loss_base.backward()
        base_grad = get_flat_grad(model)
        base_grad_sub = base_grad[indices].clone()

        # Sub-Hessian via finite differences
        sub_hessian = torch.zeros(budget, budget, device=device)

        for col_idx, param_idx in enumerate(indices):
            flat_perturbed = flat.clone()
            flat_perturbed[param_idx] += eps
            unflatten_params(model, flat_perturbed)

            model.zero_grad()
            loss_p = compute_loss(model, x, y, mask)
            loss_p.backward()
            perturbed_grad = get_flat_grad(model)
            perturbed_grad_sub = perturbed_grad[indices]

            sub_hessian[:, col_idx] = (perturbed_grad_sub - base_grad_sub) / eps

        # Restore original params
        unflatten_params(model, flat)

        # Symmetrize
        sub_hessian = 0.5 * (sub_hessian + sub_hessian.T)

        # Eigendecomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(sub_hessian)

        min_eval = eigenvalues[0].item()
        if min_eval >= 0:
            return None, 0, eigenvalues

        # Escape direction from most negative eigenvalue
        escape_sub = eigenvectors[:, 0]

        escape_full = torch.zeros(total_params, device=device)
        for i, param_idx in enumerate(indices):
            escape_full[param_idx] = escape_sub[i]

        escape_full = escape_full / (escape_full.norm() + 1e-8)

        return escape_full, min_eval, eigenvalues

    # ═══════════════════════════════════════════════
    #  Run both experiments
    # ═══════════════════════════════════════════════

    results = {}

    for experiment_name in ["adam_baseline", "adam_plus_rhs"]:
        print(f"\n{'='*60}")
        print(f"  Experiment: {experiment_name}")
        print(f"{'='*60}")

        torch.manual_seed(1337)
        model = build_model_from_spec(model_record, device=device).train()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model params: {total_params:,}")

        dataset_run = ByteDataset(files, seq_len=256, embed_mode=True, seed=42)

        losses = []
        escape_info = []
        t_start = time.time()

        for step in range(NUM_STEPS):
            x, y, mask = dataset_run.sample_batch(BATCH_SIZE, device)

            model.zero_grad()
            pred, _ = model(x, state=None)
            per_pos = F.cross_entropy(pred.transpose(1, 2), y, reduction="none")
            masked_loss = (per_pos * mask).sum() / mask.sum().clamp(min=1)

            masked_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            loss_val = masked_loss.item()
            losses.append(loss_val)

            # Random Hessian Search step
            if experiment_name == "adam_plus_rhs" and (step + 1) % RHS_EVERY == 0:
                print(f"  [RHS] Step {step+1}: computing sub-Hessian (budget={RHS_BUDGET})...")
                t_rhs = time.time()

                x_h, y_h, mask_h = dataset_run.sample_batch(BATCH_SIZE, device)
                escape_dir, min_eval, evals = random_hessian_escape(
                    model, x_h, y_h, mask_h,
                    budget=RHS_BUDGET,
                )

                rhs_time = time.time() - t_rhs
                neg_count = (evals < 0).sum().item()

                if escape_dir is not None:
                    flat = flatten_params(model)
                    flat += RHS_LR * escape_dir
                    unflatten_params(model, flat)

                    print(f"  [RHS] ESCAPE! min_eigenvalue={min_eval:.4f}, "
                          f"negative_dirs={neg_count}/{RHS_BUDGET}, "
                          f"time={rhs_time:.1f}s")
                else:
                    print(f"  [RHS] No negative curvature found. "
                          f"min_eigenvalue={min_eval:.4f}, time={rhs_time:.1f}s")

                escape_info.append({
                    "step": step + 1,
                    "min_eigenvalue": min_eval,
                    "negative_count": neg_count,
                    "escaped": escape_dir is not None,
                    "time_sec": rhs_time,
                })

            if (step + 1) % 50 == 0:
                avg = np.mean(losses[-50:])
                elapsed = time.time() - t_start
                print(f"  Step {step+1:4d} | loss={loss_val:.4f} | avg50={avg:.4f} | "
                      f"elapsed={elapsed:.1f}s")

        total_time = time.time() - t_start
        results[experiment_name] = {
            "losses": losses,
            "final_loss": losses[-1],
            "avg_last50": float(np.mean(losses[-50:])),
            "total_time": total_time,
            "escape_info": escape_info,
        }

    # ═══════════════════════════════════════════════
    #  Summary
    # ═══════════════════════════════════════════════

    print(f"\n{'='*60}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*60}")

    for name, r in results.items():
        print(f"\n  {name}:")
        print(f"    Final loss:       {r['final_loss']:.4f}")
        print(f"    Avg last 50:      {r['avg_last50']:.4f}")
        print(f"    Total time:       {r['total_time']:.1f}s")
        if r["escape_info"]:
            escapes = sum(1 for e in r["escape_info"] if e["escaped"])
            avg_neg = np.mean([e["negative_count"] for e in r["escape_info"]])
            avg_min = np.mean([e["min_eigenvalue"] for e in r["escape_info"]])
            print(f"    RHS attempts:     {len(r['escape_info'])}")
            print(f"    Successful escapes: {escapes}")
            print(f"    Avg negative dirs: {avg_neg:.1f}/{RHS_BUDGET}")
            print(f"    Avg min eigenvalue: {avg_min:.4f}")

    baseline = results["adam_baseline"]["avg_last50"]
    rhs = results["adam_plus_rhs"]["avg_last50"]
    diff = baseline - rhs
    pct = (diff / baseline) * 100 if baseline > 0 else 0

    print(f"\n  Delta (baseline - RHS): {diff:+.4f} ({pct:+.1f}%)")
    if diff > 0:
        print(f"  >> RHS is BETTER by {pct:.1f}%")
    elif diff < 0:
        print(f"  >> Baseline is better by {-pct:.1f}%")
    else:
        print(f"  >> No difference")

    return results


@app.local_entrypoint()
def main():
    print("Starting Random Hessian Search experiment on Modal GPU...")
    results = run_experiment.remote()
    print("\nDone! Results returned.")
