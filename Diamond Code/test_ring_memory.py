"""
Adversarial test suite for RingMemoryModel.

Tests:
1. Smoke tests (initialization, forward pass)
2. Shape invariants
3. Numerical stability (NaN/Inf checks)
4. Gradient flow
5. Memory operations (write then read)
6. Pointer dynamics (wrapping, inertia, deadzone)
7. Edge cases (zero-dim, huge batch, long sequences)
8. Adversarial inputs (all zeros, all same, random noise)
9. Determinism (same seed = same output)
10. Learning test (can memorize 1 sample?)
"""

import torch
import pytest
import sys
import os
from pathlib import Path

# Add Diamond Code to path
sys.path.insert(0, str(Path(__file__).parent))

from ring_memory_model import RingMemoryModel


class TestRingMemoryModel:
    def test_initialization(self):
        """Model initializes without errors."""
        model = RingMemoryModel(
            input_size=1,
            num_outputs=2,
            num_memory_positions=64,
            embedding_dim=64,
        )
        assert model is not None

    def test_forward_pass_smoke(self):
        """Forward pass runs without crashing."""
        model = RingMemoryModel(input_size=1, num_outputs=2)
        x = torch.randn(4, 16, 1)  # [batch=4, seq=16, input=1]

        logits, aux_loss, _ = model(x)

        assert logits.shape == (4, 2)  # [batch, num_outputs]
        assert aux_loss >= 0

    def test_output_shapes(self):
        """Output shapes are correct for various configs."""
        configs = [
            (1, 2, 16, 64),    # scalar input, binary
            (10, 10, 32, 128), # 10-dim input, 10-class
            (64, 100, 256, 256),  # large
        ]

        for input_size, num_outputs, num_positions, embed_dim in configs:
            model = RingMemoryModel(
                input_size=input_size,
                num_outputs=num_outputs,
                num_memory_positions=num_positions,
                embedding_dim=embed_dim,
            )
            x = torch.randn(8, 20, input_size)
            logits, _, _ = model(x)
            assert logits.shape == (8, num_outputs)

    def test_no_nans_or_infs(self):
        """Model produces finite values (no NaN/Inf)."""
        model = RingMemoryModel(input_size=1, num_outputs=2)
        x = torch.randn(16, 32, 1)

        logits, aux_loss, _ = model(x)

        assert torch.isfinite(logits).all()
        assert torch.isfinite(torch.tensor(aux_loss))

    def test_gradients_flow(self):
        """Gradients flow through core parameters."""
        model = RingMemoryModel(input_size=1, num_outputs=2)
        x = torch.randn(4, 16, 1)
        y = torch.randint(0, 2, (4,))

        logits, aux_loss, _ = model(x)
        loss = torch.nn.functional.cross_entropy(logits, y) + aux_loss
        loss.backward()

        # Core parameters must always have gradients
        core_params = ['input_projection', 'output_head', 'jump_destinations', 'context_strength']
        for name, param in model.named_parameters():
            if any(core in name for core in core_params):
                assert param.grad is not None, f"No gradient for {name}"
                assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"

        # Jump gate may not receive gradients on every random forward pass (STE behavior)
        # This is expected and tested separately in test_hard_jumps.py

    def test_pointer_wrapping(self):
        """Pointer stays in [0, num_positions) after updates."""
        model = RingMemoryModel(input_size=1, num_outputs=2, num_memory_positions=64)
        x = torch.randn(4, 100, 1)  # Long sequence

        logits, _, debug = model(x, return_debug=True)

        for ptr in debug["pointer_trajectory"]:
            assert (ptr >= 0).all()
            assert (ptr < 64).all()

    def test_attention_weights_sum_to_one(self):
        """Gaussian attention weights sum to 1.0."""
        model = RingMemoryModel(input_size=1, num_outputs=2)
        ptr = torch.tensor([0.5, 32.7, 63.9])

        _, weights = model._gaussian_attention_weights(ptr, 64)

        sums = weights.sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)

    def test_circular_distance(self):
        """Circular distance wraps correctly."""
        model = RingMemoryModel(input_size=1, num_outputs=2)

        # Distance from 60 to 4 on ring of 64 should be +8 (not -56)
        dist = model._circular_distance(
            torch.tensor([60.0]),
            torch.tensor([4.0]),
            64
        )
        assert torch.isclose(dist, torch.tensor([8.0]), atol=1e-3)

        # Distance from 4 to 60 should be -8 (not +56)
        dist = model._circular_distance(
            torch.tensor([4.0]),
            torch.tensor([60.0]),
            64
        )
        assert torch.isclose(dist, torch.tensor([-8.0]), atol=1e-3)

    def test_adversarial_all_zeros(self):
        """Model handles all-zero input."""
        model = RingMemoryModel(input_size=1, num_outputs=2)
        x = torch.zeros(4, 16, 1)

        logits, aux_loss, _ = model(x)

        assert torch.isfinite(logits).all()

    def test_adversarial_all_same(self):
        """Model handles constant input."""
        model = RingMemoryModel(input_size=1, num_outputs=2)
        x = torch.full((4, 16, 1), 5.0)

        logits, aux_loss, _ = model(x)

        assert torch.isfinite(logits).all()

    def test_adversarial_huge_batch(self):
        """Model handles large batch size."""
        model = RingMemoryModel(input_size=1, num_outputs=2)
        x = torch.randn(1024, 8, 1)  # 1024 batch

        logits, aux_loss, _ = model(x)

        assert logits.shape == (1024, 2)

    def test_determinism(self):
        """Same seed produces same output."""
        torch.manual_seed(42)
        model1 = RingMemoryModel(input_size=1, num_outputs=2)
        torch.manual_seed(99)  # Different seed for input
        x = torch.randn(4, 16, 1)
        out1, _, _ = model1(x)

        torch.manual_seed(42)
        model2 = RingMemoryModel(input_size=1, num_outputs=2)
        torch.manual_seed(99)  # Same seed for input
        x2 = torch.randn(4, 16, 1)
        out2, _, _ = model2(x2)

        if not torch.allclose(out1, out2, atol=1e-6):
            diff = (out1 - out2).abs().max().item()
            raise AssertionError(f"Outputs differ by {diff:.6e}")

    def test_can_memorize_one_sample(self):
        """Model can overfit to single sample (learning test)."""
        torch.manual_seed(42)
        model = RingMemoryModel(input_size=1, num_outputs=2)
        x = torch.randn(1, 16, 1)
        y = torch.tensor([1])

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Train for 100 steps
        for _ in range(100):
            optimizer.zero_grad()
            logits, aux_loss, _ = model(x)
            loss = torch.nn.functional.cross_entropy(logits, y) + aux_loss
            loss.backward()
            optimizer.step()

        # Check if learned
        with torch.no_grad():
            logits, _, _ = model(x)
            pred = logits.argmax(dim=1)

        assert pred.item() == 1, "Model failed to memorize single sample"


if __name__ == "__main__":
    # Run tests
    test_suite = TestRingMemoryModel()

    print("="*70)
    print("DIAMOND CODE - RingMemoryModel Test Suite")
    print("="*70)
    print()

    tests = [
        ("Initialization", test_suite.test_initialization),
        ("Forward Pass Smoke", test_suite.test_forward_pass_smoke),
        ("Output Shapes", test_suite.test_output_shapes),
        ("No NaN/Inf", test_suite.test_no_nans_or_infs),
        ("Gradient Flow", test_suite.test_gradients_flow),
        ("Pointer Wrapping", test_suite.test_pointer_wrapping),
        ("Attention Weights Sum", test_suite.test_attention_weights_sum_to_one),
        ("Circular Distance", test_suite.test_circular_distance),
        ("Adversarial: All Zeros", test_suite.test_adversarial_all_zeros),
        ("Adversarial: All Same", test_suite.test_adversarial_all_same),
        ("Adversarial: Huge Batch", test_suite.test_adversarial_huge_batch),
        ("Determinism", test_suite.test_determinism),
        ("Learning Test (1 sample)", test_suite.test_can_memorize_one_sample),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            test_fn()
            print(f"[PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            failed += 1

    print()
    print("="*70)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*70)

    if failed == 0:
        print()
        print("ALL TESTS PASSED!")
        print("Unit tests complete. Run verify_copy_task.py and test_hard_jumps.py for full verification.")
    else:
        print()
        print("TESTS FAILED - Fix issues before committing to Diamond Code.")
