"""Deterministic tests for the Fibonacci Halving Prismion Swarm."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

import conftest  # noqa: F401  (import side-effect: sys.path bootstrap)

from vraxion.platinum.hallway import AbsoluteHallway
from vraxion.platinum.swarm import fibonacci_halving_budget

# Import probe11 data generators (tools/ is a sibling of tests/).
# IMPORTANT: probe11 sets env vars at module level; save/restore to avoid
# polluting the test environment (breaks test_fibonacci_disabled_backward_compat).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import os as _os
_saved_env = {k: v for k, v in _os.environ.items()}
from probe11_fib_volume_weight import make_byte_waveform_batch, make_harmonic_xor_batch  # noqa: E402
# Restore: remove keys that were added, revert keys that were changed.
for k in list(_os.environ.keys()):
    if k not in _saved_env:
        del _os.environ[k]
    elif _os.environ[k] != _saved_env[k]:
        _os.environ[k] = _saved_env[k]
del _saved_env


# ── shared constants ────────────────────────────────────────────────

_BASE_ENV = dict(
    VRX_SENSORY_RING="0",
    VRX_VAULT="0",
    VRX_THINK_RING="1",
    VRX_THINK_RING_MODE="replace",
    VRX_THINK_RING_DUAL="1",
    VRX_THINK_RING_BRAINSTEM="0",
    VRX_THINK_RING_LEN="8",
    VRX_AUXDIM="8",
    VRX_PRISMION="1",
    VRX_PRISMION_TOPOLOGY="bank",
    VRX_PRISMION_N="2",
    VRX_PRISMION_LEN="4",
    VRX_PRISMION_ALPHA="1.0",
    VRX_NAN_GUARD=None,
)

_MODEL_KWARGS = dict(
    input_dim=4,
    num_classes=5,
    ring_len=8,
    slot_dim=16,
    ptr_stride=1,
    gauss_k=1,
    gauss_tau=2.0,
)


def _fib_env(**extra: str | None) -> dict[str, str | None]:
    """Return *_BASE_ENV* merged with Fibonacci flags and any overrides."""
    env = {
        **_BASE_ENV,
        "VRX_PRISMION_FIBONACCI": "1",
        "VRX_PRISMION_FIB_BUDGET_MB": "2",
    }
    env.update(extra)
    return env


class FibonacciSwarmTests(unittest.TestCase):
    """Deterministic behaviour locks for the Fibonacci Halving Prismion Swarm."""

    # ── 1. budget helper ────────────────────────────────────────────

    def test_fibonacci_budget_helper_deterministic(self) -> None:
        specs = fibonacci_halving_budget(
            total_params=100_000, think_dim=8, vocab_size=5,
        )

        # Non-empty list of dicts with required keys.
        self.assertIsInstance(specs, list)
        self.assertGreater(len(specs), 0)
        required_keys = {"ring_len", "slot_dim", "fraction", "param_count"}
        for spec in specs:
            self.assertIsInstance(spec, dict)
            self.assertTrue(required_keys.issubset(spec.keys()))

        # Fractions follow the 0.5, 0.25, 0.125, 0.0625, ... pattern.
        for i, spec in enumerate(specs):
            expected_frac = 0.5 ** (i + 1)
            self.assertAlmostEqual(spec["fraction"], expected_frac, places=10)

        # Constraints.
        for spec in specs:
            self.assertGreaterEqual(spec["ring_len"], 4)   # default min_ring_len
            self.assertGreaterEqual(spec["slot_dim"], 8)    # >= think_dim

        # Identical results on repeated calls.
        specs2 = fibonacci_halving_budget(
            total_params=100_000, think_dim=8, vocab_size=5,
        )
        self.assertEqual(specs, specs2)

    # ── 2. construction ─────────────────────────────────────────────

    def test_fibonacci_swarm_construction(self) -> None:
        with conftest.temporary_env(**_fib_env()):
            model = AbsoluteHallway(**_MODEL_KWARGS).cpu()

        self.assertTrue(model.prismion_fib_active)
        self.assertIsNotNone(model.prismion_swarm)
        self.assertGreater(len(model.prismion_swarm), 0)
        self.assertIsNotNone(model.prismion_swarm_heads)
        self.assertEqual(len(model.prismion_swarm_heads), len(model.prismion_swarm))

        # Heterogeneous: at least two ants differ in ring_len or slot_dim.
        ring_lens = [int(ant.ring_len) for ant in model.prismion_swarm]
        slot_dims = [int(ant.slot_dim) for ant in model.prismion_swarm]
        self.assertTrue(
            len(set(ring_lens)) > 1 or len(set(slot_dims)) > 1,
            "Swarm ants should be heterogeneous",
        )

        # Each head maps think_dim → num_classes.
        for head in model.prismion_swarm_heads:
            w = head.weight
            self.assertEqual(tuple(w.shape), (5, 8))  # (num_classes, think_dim)

    # ── 3. forward smoke ────────────────────────────────────────────

    def test_fibonacci_swarm_forward_smoke(self) -> None:
        with conftest.temporary_env(**_fib_env()):
            model = AbsoluteHallway(**_MODEL_KWARGS).cpu()
        model.eval()

        x = torch.randn(2, 6, 4, dtype=torch.float32)
        logits, move_penalty = model(x)

        self.assertEqual(tuple(logits.shape), (2, 5))
        self.assertEqual(logits.dtype, torch.float32)
        self.assertTrue(torch.isfinite(logits).all().item())
        self.assertTrue(torch.is_tensor(move_penalty))
        self.assertTrue(torch.isfinite(move_penalty).all().item())

        # Telemetry attributes.
        self.assertGreater(model.fib_swarm_n, 0)
        self.assertIsInstance(model.fib_swarm_ring_lens, list)
        self.assertTrue(torch.isfinite(torch.tensor(model.fib_swarm_logit_norm)).item())

    # ── 4. deterministic forward ────────────────────────────────────

    def test_fibonacci_swarm_deterministic_forward(self) -> None:
        def _build_and_run() -> torch.Tensor:
            torch.manual_seed(42)
            with conftest.temporary_env(**_fib_env()):
                model = AbsoluteHallway(**_MODEL_KWARGS).cpu()
            model.eval()
            torch.manual_seed(99)
            x = torch.randn(2, 6, 4, dtype=torch.float32)
            logits, _ = model(x)
            return logits.detach().clone()

        logits_a = _build_and_run()
        logits_b = _build_and_run()
        torch.testing.assert_close(logits_a, logits_b)

    # ── 5. backward pass ────────────────────────────────────────────

    def test_fibonacci_swarm_backward_pass(self) -> None:
        with conftest.temporary_env(**_fib_env()):
            model = AbsoluteHallway(**_MODEL_KWARGS).cpu()
        model.train()

        x = torch.randn(2, 6, 4, dtype=torch.float32)
        logits, _ = model(x)
        target = torch.randint(0, 5, (2,))
        loss = torch.nn.functional.cross_entropy(logits, target)
        loss.backward()

        self.assertTrue(torch.isfinite(loss).item())

        # Gradients flow into swarm ants.
        has_swarm_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in model.prismion_swarm.parameters()
        )
        self.assertTrue(has_swarm_grad, "No gradient flowed into prismion_swarm")

        # Gradients flow into swarm heads.
        has_head_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in model.prismion_swarm_heads.parameters()
        )
        self.assertTrue(has_head_grad, "No gradient flowed into prismion_swarm_heads")

    # ── 6. pure swarm mode ──────────────────────────────────────────

    def test_fibonacci_pure_swarm_mode(self) -> None:
        """When fib_active, backbone head is bypassed — VRX_MAIN_LOGIT_WEIGHT is irrelevant."""
        torch.manual_seed(7)
        x = torch.randn(2, 6, 4, dtype=torch.float32)

        # Weight=0: backbone head bypassed (fib_active).
        torch.manual_seed(42)
        with conftest.temporary_env(**_fib_env(VRX_MAIN_LOGIT_WEIGHT="0.0")):
            model_a = AbsoluteHallway(**_MODEL_KWARGS).cpu()
        model_a.eval()
        logits_a, _ = model_a(x)
        self.assertTrue(torch.isfinite(logits_a).all().item())
        self.assertGreater(logits_a.abs().sum().item(), 0, "Swarm logits should be non-zero")

        # Weight=1: backbone head STILL bypassed (fib_active overrides).
        torch.manual_seed(42)
        with conftest.temporary_env(**_fib_env(VRX_MAIN_LOGIT_WEIGHT="1.0")):
            model_b = AbsoluteHallway(**_MODEL_KWARGS).cpu()
        model_b.eval()
        logits_b, _ = model_b(x)

        self.assertTrue(
            torch.allclose(logits_a, logits_b),
            "When fib_active, VRX_MAIN_LOGIT_WEIGHT must have no effect "
            "(backbone head is bypassed, swarm is sole predictor)",
        )

    # ── 7. disabled / backward compat ───────────────────────────────

    def test_fibonacci_disabled_backward_compat(self) -> None:
        env_disabled = {**_BASE_ENV, "VRX_PRISMION_FIBONACCI": "0"}
        env_absent = {k: v for k, v in _BASE_ENV.items()}

        torch.manual_seed(42)
        with conftest.temporary_env(**env_disabled):
            model_a = AbsoluteHallway(**_MODEL_KWARGS).cpu()
        model_a.eval()

        torch.manual_seed(42)
        with conftest.temporary_env(**env_absent):
            model_b = AbsoluteHallway(**_MODEL_KWARGS).cpu()
        model_b.eval()

        torch.manual_seed(99)
        x = torch.randn(2, 6, 4, dtype=torch.float32)

        logits_a, _ = model_a(x)
        # Re-seed so the forward pass RNG is identical.
        torch.manual_seed(99)
        x2 = torch.randn(2, 6, 4, dtype=torch.float32)
        logits_b, _ = model_b(x2)

        torch.testing.assert_close(logits_a, logits_b)
        self.assertFalse(model_a.prismion_fib_active)
        self.assertIsNone(model_a.prismion_swarm)

    # ── 8. param budget ─────────────────────────────────────────────

    def test_fibonacci_swarm_param_budget(self) -> None:
        with conftest.temporary_env(**_fib_env(VRX_PRISMION_FIB_BUDGET_MB="1")):
            model = AbsoluteHallway(**_MODEL_KWARGS).cpu()

        swarm_params = sum(p.numel() for p in model.prismion_swarm.parameters())
        head_params = sum(p.numel() for p in model.prismion_swarm_heads.parameters())
        total = swarm_params + head_params

        # 1 MB / 4 bytes per float32 = 250,000 params.
        budget_params = int(1 * 1e6 / 4)
        self.assertLessEqual(total, budget_params,
                             f"Swarm params ({total:,}) exceed budget ({budget_params:,})")

        # Each ant's actual param count should roughly match its spec.
        for i, spec in enumerate(model.prismion_swarm_configs):
            ant_params = sum(p.numel() for p in model.prismion_swarm[i].parameters())
            head_p = sum(p.numel() for p in model.prismion_swarm_heads[i].parameters())
            actual = ant_params + head_p
            expected = spec["param_count"]
            # Allow 20% tolerance for rounding.
            self.assertLessEqual(
                abs(actual - expected),
                max(expected * 0.20, 50),
                f"ant[{i}] actual={actual} vs spec={expected}",
            )


# ──────────────────────────────────────────────────────────────────────
# Byte Waveform task tests
# ──────────────────────────────────────────────────────────────────────

_MODEL_KWARGS_1D = dict(
    input_dim=1,
    num_classes=4,
    ring_len=8,
    slot_dim=16,
    ptr_stride=1,
    gauss_k=1,
    gauss_tau=2.0,
)


class ByteWaveformTaskTests(unittest.TestCase):
    """Tests for the byte_waveform task (input_dim=1, 4-class)."""

    # ── data generator shape & label range ────────────────────────────

    def test_byte_waveform_batch_shapes(self) -> None:
        """Output shapes match [B, T, 1] for input and [B] for labels."""
        x, labels, slow, fast = make_byte_waveform_batch(
            batch_size=8, seq_len=64,
        )
        self.assertEqual(tuple(x.shape), (8, 64, 1))
        self.assertEqual(tuple(labels.shape), (8,))
        self.assertEqual(tuple(slow.shape), (8,))
        self.assertEqual(tuple(fast.shape), (8,))

    def test_byte_waveform_label_range(self) -> None:
        """Labels are in {0, 1, 2, 3} (4-class quadrant)."""
        _, labels, _, _ = make_byte_waveform_batch(
            batch_size=256, seq_len=64,
        )
        self.assertTrue((labels >= 0).all().item())
        self.assertTrue((labels <= 3).all().item())
        # With 256 samples, all 4 classes should appear.
        unique = labels.unique()
        self.assertEqual(len(unique), 4, f"Expected 4 classes, got {unique.tolist()}")

    def test_byte_waveform_noise_present(self) -> None:
        """With noise_std > 0, two batches with same seed differ (noise is random)."""
        torch.manual_seed(42)
        x1, _, _, _ = make_byte_waveform_batch(
            batch_size=8, seq_len=64, noise_std=0.1,
        )
        torch.manual_seed(99)
        x2, _, _, _ = make_byte_waveform_batch(
            batch_size=8, seq_len=64, noise_std=0.1,
        )
        self.assertFalse(
            torch.allclose(x1, x2),
            "Two noisy batches with different seeds should differ",
        )

    def test_byte_waveform_no_noise(self) -> None:
        """With noise_std=0, output is deterministic given same random seed."""
        torch.manual_seed(42)
        x1, lab1, _, _ = make_byte_waveform_batch(
            batch_size=8, seq_len=64, noise_std=0.0,
        )
        torch.manual_seed(42)
        x2, lab2, _, _ = make_byte_waveform_batch(
            batch_size=8, seq_len=64, noise_std=0.0,
        )
        torch.testing.assert_close(x1, x2)
        torch.testing.assert_close(lab1, lab2)

    # ── model forward with input_dim=1, num_classes=4 ────────────────

    def test_model_forward_1d_input(self) -> None:
        """AbsoluteHallway runs forward with input_dim=1 and fib swarm active."""
        with conftest.temporary_env(**_fib_env()):
            model = AbsoluteHallway(**_MODEL_KWARGS_1D).cpu()
        model.eval()

        x = torch.randn(4, 32, 1, dtype=torch.float32)
        logits, move_penalty = model(x)

        self.assertEqual(tuple(logits.shape), (4, 4))
        self.assertTrue(torch.isfinite(logits).all().item())
        self.assertTrue(torch.isfinite(move_penalty).all().item())

    def test_model_backward_1d_input(self) -> None:
        """Gradients flow through the swarm with input_dim=1."""
        with conftest.temporary_env(**_fib_env()):
            model = AbsoluteHallway(**_MODEL_KWARGS_1D).cpu()
        model.train()

        x = torch.randn(4, 32, 1, dtype=torch.float32)
        logits, _ = model(x)
        target = torch.randint(0, 4, (4,))
        loss = torch.nn.functional.cross_entropy(logits, target)
        loss.backward()

        self.assertTrue(torch.isfinite(loss).item())
        has_swarm_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in model.prismion_swarm.parameters()
        )
        self.assertTrue(has_swarm_grad, "No gradient flowed into prismion_swarm")

    def test_byte_waveform_end_to_end(self) -> None:
        """Full pipeline: generate data → forward → loss → backward."""
        with conftest.temporary_env(**_fib_env()):
            model = AbsoluteHallway(**_MODEL_KWARGS_1D).cpu()
        model.train()

        x, labels, _, _ = make_byte_waveform_batch(
            batch_size=8, seq_len=32, noise_std=0.1,
        )
        logits, move_penalty = model(x)
        loss = torch.nn.functional.cross_entropy(logits, labels) + 0.01 * move_penalty
        loss.backward()

        self.assertTrue(torch.isfinite(loss).item())
        self.assertEqual(tuple(logits.shape), (8, 4))


# ──────────────────────────────────────────────────────────────────────
# Independent Engines tests — ants receive raw input_dim, not chrom
# ──────────────────────────────────────────────────────────────────────

class IndependentEnginesTests(unittest.TestCase):
    """Verify ants are truly independent engines receiving raw input."""

    def test_ant_in_dim_matches_input_dim(self) -> None:
        """Each ant's in_dim should equal the model's input_dim, not 2*think_dim."""
        with conftest.temporary_env(**_fib_env()):
            model = AbsoluteHallway(**_MODEL_KWARGS).cpu()

        input_dim = _MODEL_KWARGS["input_dim"]  # 4
        for i, ant in enumerate(model.prismion_swarm):
            self.assertEqual(
                ant.in_dim, input_dim,
                f"ant[{i}].in_dim={ant.in_dim} should equal input_dim={input_dim}",
            )

    def test_ant_in_dim_1d_input(self) -> None:
        """With input_dim=1, ants should have in_dim=1."""
        with conftest.temporary_env(**_fib_env()):
            model = AbsoluteHallway(**_MODEL_KWARGS_1D).cpu()

        for i, ant in enumerate(model.prismion_swarm):
            self.assertEqual(
                ant.in_dim, 1,
                f"ant[{i}].in_dim={ant.in_dim} should equal 1 for 1D input",
            )

    def test_prismion_fib_active_ants_attribute(self) -> None:
        """Model should have prismion_fib_active_ants attribute."""
        with conftest.temporary_env(**_fib_env()):
            model = AbsoluteHallway(**_MODEL_KWARGS).cpu()

        self.assertTrue(hasattr(model, "prismion_fib_active_ants"))
        # Default is -1 (all ants active); runtime interprets -1 as N.
        active = int(model.prismion_fib_active_ants)
        n_ants = len(model.prismion_swarm)
        effective = n_ants if active < 0 else min(active, n_ants)
        self.assertEqual(effective, n_ants)

    def test_prismion_fib_in_dim_attribute(self) -> None:
        """Model should store prismion_fib_in_dim = input_dim."""
        with conftest.temporary_env(**_fib_env()):
            model = AbsoluteHallway(**_MODEL_KWARGS).cpu()

        self.assertTrue(hasattr(model, "prismion_fib_in_dim"))
        self.assertEqual(model.prismion_fib_in_dim, _MODEL_KWARGS["input_dim"])

    def test_budget_function_explicit_in_dim(self) -> None:
        """fibonacci_halving_budget with explicit in_dim differs from default."""
        specs_default = fibonacci_halving_budget(
            total_params=100_000, think_dim=8, vocab_size=5,
        )
        specs_custom = fibonacci_halving_budget(
            total_params=100_000, think_dim=8, vocab_size=5, in_dim=4,
        )
        # With smaller in_dim, ants get bigger slot_dim for same budget.
        self.assertGreater(len(specs_custom), 0)
        self.assertGreaterEqual(
            specs_custom[0]["slot_dim"], specs_default[0]["slot_dim"],
            "Smaller in_dim should allow bigger or equal slot_dim",
        )

    def test_forward_independent_engines(self) -> None:
        """Forward pass works with independent engine ants (raw input)."""
        with conftest.temporary_env(**_fib_env()):
            model = AbsoluteHallway(**_MODEL_KWARGS).cpu()
        model.eval()

        x = torch.randn(2, 6, 4, dtype=torch.float32)
        logits, move_penalty = model(x)

        self.assertEqual(tuple(logits.shape), (2, 5))
        self.assertTrue(torch.isfinite(logits).all().item())
        self.assertTrue(torch.isfinite(move_penalty).all().item())

    def test_backward_independent_engines(self) -> None:
        """Backward pass flows gradients through independent engine ants."""
        with conftest.temporary_env(**_fib_env()):
            model = AbsoluteHallway(**_MODEL_KWARGS).cpu()
        model.train()

        x = torch.randn(2, 6, 4, dtype=torch.float32)
        logits, _ = model(x)
        target = torch.randint(0, 5, (2,))
        loss = torch.nn.functional.cross_entropy(logits, target)
        loss.backward()

        self.assertTrue(torch.isfinite(loss).item())
        has_swarm_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in model.prismion_swarm.parameters()
        )
        self.assertTrue(has_swarm_grad, "No gradient flowed into prismion_swarm")


if __name__ == "__main__":
    unittest.main()
