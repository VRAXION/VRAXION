"""Deterministic tests for the Fibonacci Halving Prismion Swarm."""

from __future__ import annotations

import unittest

import torch

import conftest  # noqa: F401  (import side-effect: sys.path bootstrap)

from vraxion.platinum.hallway import AbsoluteHallway
from vraxion.platinum.swarm import fibonacci_halving_budget


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
        torch.manual_seed(7)
        x = torch.randn(2, 6, 4, dtype=torch.float32)

        # Pure swarm: main logits zeroed out.
        torch.manual_seed(42)
        with conftest.temporary_env(**_fib_env(VRX_MAIN_LOGIT_WEIGHT="0.0")):
            model_pure = AbsoluteHallway(**_MODEL_KWARGS).cpu()
        model_pure.eval()
        logits_pure, _ = model_pure(x)
        self.assertTrue(torch.isfinite(logits_pure).all().item())
        self.assertGreater(logits_pure.abs().sum().item(), 0, "Pure swarm logits should be non-zero")

        # Hybrid: main logits at full weight.
        torch.manual_seed(42)
        with conftest.temporary_env(**_fib_env(VRX_MAIN_LOGIT_WEIGHT="1.0")):
            model_hybrid = AbsoluteHallway(**_MODEL_KWARGS).cpu()
        model_hybrid.eval()
        logits_hybrid, _ = model_hybrid(x)

        self.assertFalse(
            torch.allclose(logits_pure, logits_hybrid),
            "Pure-swarm and hybrid logits must differ",
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


if __name__ == "__main__":
    unittest.main()
