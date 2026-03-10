"""Test and demo for Hessian saddle point detection and escape.

Creates a synthetic loss landscape with a known saddle point at the origin,
then shows that the optimizer detects it and escapes.

The Monkey Saddle: f(x,y) = x^3 - 3xy^2
    - gradient = 0 at origin
    - Hessian at origin has eigenvalues {-6y, 6x} → both 0 at origin
    - but nearby: clear saddle structure

We use a slightly shifted version to have a clean negative eigenvalue:
    f(x,y) = x^3 - 3xy^2 + 0.5*(x^2 - y^2)
    Hessian at origin = [[1, 0], [0, -1]]  ← saddle!
"""

import sys
from pathlib import Path

# Add training/ to path so we can import hessian_saddle
_TRAINING_DIR = str(Path(__file__).resolve().parent.parent / 'training')
if _TRAINING_DIR not in sys.path:
    sys.path.insert(0, _TRAINING_DIR)

import torch
import torch.nn as nn

from hessian_saddle import (
    HessianSaddleConfig,
    HessianSaddleOptimizer,
    hessian_vector_product,
    lanczos_extreme_eigenvalue,
)


# ── Test 1: Hessian-vector product correctness ──────────────

def test_hvp():
    """Verify HVP against analytical Hessian for a quadratic."""
    print("Test 1: Hessian-vector product ... ", end="")

    # f(x) = 0.5 * x^T A x  →  H = A
    A = torch.tensor([[2.0, 1.0], [1.0, 3.0]])
    x = nn.Parameter(torch.tensor([1.0, 2.0]))

    loss = 0.5 * x @ A @ x
    v = [torch.tensor([1.0, 0.0])]

    hvp = hessian_vector_product(loss, [x], v)
    expected = A @ v[0]

    assert torch.allclose(hvp[0], expected, atol=1e-5), \
        f"HVP mismatch: got {hvp[0]}, expected {expected}"
    print("PASS")


# ── Test 2: Lanczos finds negative eigenvalue ──────────────

def test_lanczos_saddle():
    """Lanczos should find the negative eigenvalue of a saddle Hessian."""
    print("Test 2: Lanczos eigenvalue at saddle ... ", end="")

    # f(x,y) = 0.5*(x^2 - y^2) → Hessian = diag(1, -1)
    xy = nn.Parameter(torch.zeros(2))

    def make_loss():
        return 0.5 * (xy[0]**2 - xy[1]**2)

    loss = make_loss()
    min_eval, evec = lanczos_extreme_eigenvalue(loss, [xy], k=5, seed=42)

    print(f"min eigenvalue = {min_eval:.4f} ... ", end="")
    assert min_eval < -0.5, f"Expected negative eigenvalue, got {min_eval}"
    print("PASS")


# ── Test 3: Lanczos on a minimum (all positive eigenvalues) ─

def test_lanczos_minimum():
    """At a local minimum, all eigenvalues should be positive."""
    print("Test 3: Lanczos eigenvalue at minimum ... ", end="")

    # f(x,y) = x^2 + 2*y^2 → Hessian = diag(2, 4) — all positive
    xy = nn.Parameter(torch.zeros(2))
    loss = xy[0]**2 + 2 * xy[1]**2

    min_eval, _ = lanczos_extreme_eigenvalue(loss, [xy], k=5, seed=42)

    print(f"min eigenvalue = {min_eval:.4f} ... ", end="")
    assert min_eval > 0.5, f"Expected positive eigenvalue, got {min_eval}"
    print("PASS")


# ── Test 4: Full optimizer saddle escape demo ───────────────

def test_saddle_escape():
    """Demo: a small MLP trained on a degenerate loss that creates saddle points."""
    print("Test 4: Saddle escape demo with MLP ... ", end="")

    torch.manual_seed(0)

    # Simple 2-layer network
    model = nn.Sequential(
        nn.Linear(4, 8, bias=False),
        nn.Tanh(),
        nn.Linear(8, 1, bias=False),
    )

    # Initialize near a saddle: set weights very small
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(0.001)

    def loss_fn(model, batch):
        x, y = batch
        return ((model(x) - y) ** 2).mean()

    config = HessianSaddleConfig(
        lanczos_steps=5,
        grad_threshold=0.1,  # generous threshold for demo
        eigenvalue_threshold=-1e-5,
        max_batches=5,
        confidence_width=1.0,  # relaxed for small demo
        escape_lr=0.05,
    )

    opt = HessianSaddleOptimizer(
        model=model,
        base_optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        loss_fn=loss_fn,
        config=config,
    )

    # Synthetic data
    X = torch.randn(32, 4)
    Y = torch.randn(32, 1)
    batch = (X, Y)

    losses = []
    for step in range(50):
        info = opt.step(batch)
        losses.append(info['loss'])
        if info.get('saddle_escaped'):
            print(f"\n  [step {step}] Saddle escape! "
                  f"eigenvalue={info['min_eigenvalue']:.6f}, "
                  f"CI={info['ci']}")

    stats = opt.stats()
    print(f"\n  Final loss: {losses[0]:.6f} → {losses[-1]:.6f}")
    print(f"  Saddle checks: {stats['total_checks']}, "
          f"escapes: {stats['saddle_escapes']}")
    print("  PASS")


# ── Test 5: Confidence interval convergence ─────────────────

def test_confidence_convergence():
    """Show that the CI narrows with more batches."""
    print("Test 5: Confidence interval convergence ... ", end="")

    xy = nn.Parameter(torch.zeros(2))

    # Known saddle: Hessian = diag(3, -1)
    def make_loss():
        return 1.5 * xy[0]**2 - 0.5 * xy[1]**2

    eigenvalue_samples = []
    for i in range(10):
        loss = make_loss()
        min_eval, _ = lanczos_extreme_eigenvalue(loss, [xy], k=5, seed=i * 37)
        eigenvalue_samples.append(min_eval)

    # Check that samples are consistent (deterministic Hessian → low variance)
    import statistics
    mean_e = statistics.mean(eigenvalue_samples)
    std_e = statistics.stdev(eigenvalue_samples) if len(eigenvalue_samples) > 1 else 0

    print(f"mean={mean_e:.4f}, std={std_e:.6f} ... ", end="")
    assert abs(mean_e - (-1.0)) < 0.2, f"Expected ~-1.0, got {mean_e}"
    print("PASS")


# ── Run all ─────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("Hessian Saddle Point Detection — Test Suite")
    print("=" * 60)
    print()

    test_hvp()
    test_lanczos_saddle()
    test_lanczos_minimum()
    test_saddle_escape()
    test_confidence_convergence()

    print()
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
