"""Learning validation tests for INSTNCT training loop.

These tests assert that train_steps() actually drives learning — not just
that it returns correct keys.  They run on CPU in <60s total and are
CI-gatable.

Tests:
1. test_loss_decreases_on_learnable_task       – loss_slope < 0
2. test_model_beats_random_baseline            – held-out accuracy > 0.7
3. test_loss_converges_on_memorizable_dataset   – final loss < 0.1
4. test_random_labels_do_not_converge           – loss stays > 0.5
"""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import conftest  # noqa: F401  (import side-effect: sys.path bootstrap)
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from tools import instnct_train_steps


# ---------------------------------------------------------------------------
# Shared tiny model – mirrors TinyTP6Model from test_instnct_train_steps.py
# ---------------------------------------------------------------------------


class _TinyLearnModel(nn.Module):
    """Minimal model with enough telemetry attributes for train_steps()."""

    def __init__(self, in_dim: int, num_classes: int, hidden: int = 0) -> None:
        super().__init__()
        if hidden > 0:
            self.fc = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, num_classes),
            )
        else:
            self.fc = nn.Linear(in_dim, num_classes)

        # Telemetry attributes expected by the training loop.
        self.register_buffer("pointer_hist", torch.zeros(num_classes, dtype=torch.float32))
        self.satiety_exits = 0

        self.ptr_flip_rate = 0.0
        self.ptr_mean_dwell = 2.0
        self.ptr_delta_abs_mean = 0.1
        self.ptr_max_dwell = 3
        self.ptr_delta_raw_mean = 0.05

        self.ptr_inertia = 0.3
        self.ptr_deadzone = 0.1
        self.ptr_walk_prob = 0.2
        self.ptr_update_every = 1

    def forward(self, x: torch.Tensor):
        logits = self.fc(x)

        with torch.no_grad():
            preds = logits.argmax(dim=1)
            hist = torch.bincount(preds, minlength=self.pointer_hist.numel()).to(self.pointer_hist.dtype)
            self.pointer_hist.add_(hist)
            if hist.numel() > 0:
                self.ptr_flip_rate = float((hist > 0).float().mean().item())
            else:
                self.ptr_flip_rate = 0.0

        move_pen = logits.new_tensor(0.0)
        return logits, move_pen


# ---------------------------------------------------------------------------
# Shared env patch — disable heavy subsystems, force CPU.
# ---------------------------------------------------------------------------

_CPU_ENV = {
    "VAR_COMPUTE_DEVICE": "cpu",
    "VAR_TRAINING_TRACE_ENABLED": "0",
}


def _make_learnable_data(n: int = 128, in_dim: int = 8, seed: int = 42):
    """y = (x[:, 0] > 0).long()  — trivially learnable binary classification."""
    torch.manual_seed(seed)
    x = torch.randn(n, in_dim)
    y = (x[:, 0] > 0).long()
    return x, y


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLearningValidation(unittest.TestCase):
    """Assert that train_steps() actually learns — not just runs."""

    def setUp(self) -> None:
        torch.manual_seed(0)
        # Suppress log noise during tests.
        instnct_train_steps.log = lambda _msg: None

    # ------------------------------------------------------------------
    # Test 1: loss decreases on a learnable task
    # ------------------------------------------------------------------
    def test_loss_decreases_on_learnable_task(self) -> None:
        """50 training steps on y=(x[:,0]>0) must produce negative loss_slope."""
        x, y = _make_learnable_data(n=128, in_dim=8)
        loader = DataLoader(TensorDataset(x, y), batch_size=16, shuffle=True)
        model = _TinyLearnModel(in_dim=8, num_classes=2)

        with patch.dict(os.environ, _CPU_ENV, clear=False):
            out = instnct_train_steps.train_steps(model, loader, 50, "learn_test", "tiny")

        self.assertIn("loss_slope", out)
        self.assertIsNotNone(out["loss_slope"], "loss_slope should not be None after 50 steps")
        self.assertLess(
            out["loss_slope"],
            0.0,
            f"Loss should decrease on a learnable task, got slope={out['loss_slope']:.6f}",
        )

    # ------------------------------------------------------------------
    # Test 2: model beats random baseline on held-out data
    # ------------------------------------------------------------------
    def test_model_beats_random_baseline(self) -> None:
        """After 200 steps, held-out accuracy must exceed 0.7 (random = 0.5)."""
        x_train, y_train = _make_learnable_data(n=128, in_dim=8, seed=42)
        x_test, y_test = _make_learnable_data(n=64, in_dim=8, seed=99)

        loader = DataLoader(TensorDataset(x_train, y_train), batch_size=16, shuffle=True)
        model = _TinyLearnModel(in_dim=8, num_classes=2, hidden=32)

        with patch.dict(os.environ, _CPU_ENV, clear=False):
            instnct_train_steps.train_steps(model, loader, 200, "baseline_test", "tiny")

        # Evaluate on held-out data.
        model.eval()
        with torch.no_grad():
            logits, _ = model(x_test)
            preds = logits.argmax(dim=1)
            accuracy = float((preds == y_test).float().mean().item())

        self.assertGreater(
            accuracy,
            0.7,
            f"Model should beat random baseline (0.5), got accuracy={accuracy:.4f}",
        )

    # ------------------------------------------------------------------
    # Test 3: loss converges on a memorizable dataset
    # ------------------------------------------------------------------
    def test_loss_converges_on_memorizable_dataset(self) -> None:
        """16 samples, 300 steps — final loss must drop below 0.1."""
        x, y = _make_learnable_data(n=16, in_dim=8, seed=7)
        loader = DataLoader(TensorDataset(x, y), batch_size=16, shuffle=False)
        model = _TinyLearnModel(in_dim=8, num_classes=2, hidden=32)

        with patch.dict(os.environ, _CPU_ENV, clear=False):
            out = instnct_train_steps.train_steps(model, loader, 300, "memorize_test", "tiny")

        # Evaluate final loss on the same data (memorization check).
        model.eval()
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            logits, _ = model(x)
            final_loss = float(criterion(logits, y).item())

        self.assertLess(
            final_loss,
            0.1,
            f"Model should memorize 16 samples in 100 steps, got loss={final_loss:.6f}",
        )

    # ------------------------------------------------------------------
    # Test 4: random labels do NOT converge (negative control)
    # ------------------------------------------------------------------
    def test_random_labels_do_not_converge(self) -> None:
        """Shuffled labels should keep loss high (> 0.5) — negative control."""
        torch.manual_seed(123)
        x = torch.randn(128, 8)
        y = torch.randint(0, 2, (128,))  # random labels, no learnable pattern

        loader = DataLoader(TensorDataset(x, y), batch_size=16, shuffle=True)
        model = _TinyLearnModel(in_dim=8, num_classes=2)

        with patch.dict(os.environ, _CPU_ENV, clear=False):
            out = instnct_train_steps.train_steps(model, loader, 50, "random_test", "tiny")

        # With random labels the loss should stay near -ln(0.5) ≈ 0.693.
        # Evaluate final loss.
        model.eval()
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            logits, _ = model(x)
            final_loss = float(criterion(logits, y).item())

        self.assertGreater(
            final_loss,
            0.5,
            f"Random labels should not converge, got loss={final_loss:.4f}",
        )


if __name__ == "__main__":
    unittest.main()
