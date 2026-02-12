"""Behavior locks for :class:`vraxion.platinum.hallway.AbsoluteHallway`."""

from __future__ import annotations

import unittest

import torch

import conftest  # noqa: F401  (import side-effect: sys.path bootstrap)

from vraxion.platinum.hallway import AbsoluteHallway


class AbsoluteHallwayTests(unittest.TestCase):
    def test_cpu_import_and_tiny_forward(self) -> None:
        with conftest.temporary_env(
            VRX_SENSORY_RING="0",
            VRX_VAULT="0",
            VRX_THINK_RING="0",
            VRX_NAN_GUARD=None,
        ):
            model = AbsoluteHallway(
                input_dim=4,
                num_classes=5,
                ring_len=8,
                slot_dim=16,
                ptr_stride=1,
                gauss_k=1,
                gauss_tau=2.0,
            ).cpu()
            model.eval()

            # No implicit device moves.
            for par in model.parameters():
                self.assertFalse(par.is_cuda)

            x = torch.randn(2, 3, 4, dtype=torch.float32)
            logits, move_penalty = model(x)

            self.assertEqual(tuple(logits.shape), (2, 5))
            self.assertEqual(logits.dtype, torch.float32)
            self.assertTrue(torch.isfinite(logits).all().item())

            self.assertTrue(torch.is_tensor(move_penalty))
            self.assertEqual(tuple(move_penalty.shape), ())
            self.assertTrue(torch.isfinite(move_penalty).all().item())

    def test_forward_with_sensory_ring_enabled(self) -> None:
        with conftest.temporary_env(
            VRX_SENSORY_RING="1",
            VRX_VAULT="0",
            VRX_THINK_RING="0",
            VRX_NAN_GUARD=None,
        ):
            model = AbsoluteHallway(
                input_dim=3,
                num_classes=4,
                ring_len=12,
                slot_dim=24,
                ptr_stride=1,
                gauss_k=1,
                gauss_tau=1.0,
            ).cpu()
            model.eval()

            x = torch.randn(2, 4, 3, dtype=torch.float32)
            logits, move_penalty = model(x)

            self.assertEqual(tuple(logits.shape), (2, 4))
            self.assertTrue(torch.isfinite(logits).all().item())
            self.assertTrue(torch.isfinite(move_penalty).all().item())

    def test_forward_with_xray(self) -> None:
        with conftest.temporary_env(
            VRX_SENSORY_RING="0",
            VRX_VAULT="0",
            VRX_THINK_RING="0",
            VRX_NAN_GUARD=None,
        ):
            model = AbsoluteHallway(
                input_dim=4,
                num_classes=3,
                ring_len=8,
                slot_dim=16,
                ptr_stride=1,
                gauss_k=1,
                gauss_tau=2.0,
            ).cpu()
            model.eval()

            x = torch.randn(2, 5, 4, dtype=torch.float32)
            logits, move_penalty, xray = model(x, return_xray=True)

            self.assertEqual(tuple(logits.shape), (2, 3))
            self.assertTrue(torch.isfinite(logits).all().item())
            self.assertTrue(torch.isfinite(move_penalty).all().item())
            self.assertIsInstance(xray, dict)

            # If telemetry keys exist, values must be finite floats.
            for key, val in xray.items():
                self.assertIsInstance(key, str)
                self.assertTrue(isinstance(val, (float, int)))
                self.assertTrue(torch.isfinite(torch.tensor(float(val))).item())

            # Behavior locks: these keys are expected to be present when xray is enabled.
            self.assertIn("ptr_delta_abs_mean", xray)
            self.assertIn("ptr_delta_raw_mean", xray)

    def test_fib_swarm_only_logits(self) -> None:
        """When fib_active, final logits come ONLY from the swarm, not backbone head."""
        with conftest.temporary_env(
            VRX_SENSORY_RING="0",
            VRX_VAULT="0",
            VRX_THINK_RING="1",
            VRX_AUXDIM="16",
            VRX_PRISMION="1",
            VRX_PRISMION_FIBONACCI="1",
            VRX_PRISMION_FIB_BUDGET_MB="0.3",
            VRX_PRISMION_FIB_MIN_RING="4",
            VRX_PRISMION_FIB_MAX_ANTS="4",
            VRX_PRISMION_FIB_ACTIVE_ANTS="-1",
            VRX_NAN_GUARD=None,
        ):
            model = AbsoluteHallway(
                input_dim=4,
                num_classes=2,
                ring_len=16,
                slot_dim=32,
                ptr_stride=1,
                gauss_k=1,
                gauss_tau=2.0,
            ).cpu()
            model.eval()

            self.assertTrue(model.prismion_fib_active, "Fibonacci swarm must be active")
            self.assertIsNotNone(model.prismion_swarm)
            self.assertGreater(len(model.prismion_swarm), 0)

            x = torch.randn(2, 8, 4, dtype=torch.float32)

            # Forward should work and produce valid logits.
            torch.manual_seed(42)
            logits, mp = model(x)
            self.assertEqual(tuple(logits.shape), (2, 2))
            self.assertTrue(torch.isfinite(logits).all().item())

            # Zero all swarm params → logits must be near-zero.
            # This proves the backbone head is NOT contributing.
            with torch.no_grad():
                for ant in model.prismion_swarm:
                    for p in ant.parameters():
                        p.zero_()
                for head in model.prismion_swarm_heads:
                    for p in head.parameters():
                        p.zero_()

            torch.manual_seed(42)
            logits_zero, _ = model(x)
            max_logit = logits_zero.abs().max().item()
            self.assertLess(max_logit, 1e-4,
                            f"Zeroed swarm should produce near-zero logits "
                            f"(got {max_logit:.6f}). Backbone head is leaking!")

    def test_bypass_ring_path(self) -> None:
        with conftest.temporary_env(
            VRX_SENSORY_RING="0",
            VRX_VAULT="0",
            VRX_THINK_RING="0",
            VRX_NAN_GUARD=None,
        ):
            model = AbsoluteHallway(
                input_dim=4,
                num_classes=3,
                ring_len=8,
                slot_dim=16,
                ptr_stride=1,
                gauss_k=1,
                gauss_tau=2.0,
                bypass_ring=True,
            ).cpu()
            model.eval()

            x = torch.randn(2, 3, 4, dtype=torch.float32)
            logits, move_penalty = model(x)

            self.assertEqual(tuple(logits.shape), (2, 3))
            self.assertTrue(torch.isfinite(logits).all().item())
            self.assertTrue(torch.isfinite(move_penalty).all().item())



