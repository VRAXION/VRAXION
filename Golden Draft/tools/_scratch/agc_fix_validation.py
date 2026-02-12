"""AGC Fix Validation Probe

Tests the stateless gradient normalization implementation.

Expected outcomes:
1. AGC ON: Gradients normalized to [1.0, 5.0] range
2. AGC OFF: No gradient scaling (returns 1.0)
3. Full model gradient norm logged (not just theta_ptr)
4. No lagging behavior (immediate normalization)
"""

import sys
import os
import json
from pathlib import Path

# Add Golden Code to path (where vraxion package lives)
GOLDEN_CODE = Path("S:/AI/Golden Code")
sys.path.insert(0, str(GOLDEN_CODE))

import torch
from vraxion.instnct.agc import AGCParams, apply_update_agc

def test_agc_disabled():
    """Test 1: AGC OFF returns 1.0 (no scaling)"""
    print("\n=== Test 1: AGC OFF ===")

    params = AGCParams(enabled=False, grad_low=1.0, grad_high=5.0)

    # Test various gradient norms
    test_norms = [0.1, 1.0, 5.0, 50.0, 400.0]

    for grad_norm in test_norms:
        scale = apply_update_agc(None, grad_norm, params)
        assert scale == 1.0, f"AGC OFF should return 1.0, got {scale} for grad_norm={grad_norm}"
        print(f"  grad_norm={grad_norm:6.1f} -> scale={scale:.4f} OK")

    print("  OK: AGC OFF test passed")


def test_agc_normalization():
    """Test 2: AGC ON normalizes to target range"""
    print("\n=== Test 2: AGC Normalization ===")

    params = AGCParams(enabled=True, grad_low=1.0, grad_high=5.0)

    # Test cases: (grad_norm, expected_behavior)
    test_cases = [
        (0.5, "scale UP"),     # Below grad_low -> amplify
        (1.0, "no change"),    # At grad_low -> OK
        (3.0, "no change"),    # In range -> OK
        (5.0, "no change"),    # At grad_high -> OK
        (10.0, "scale DOWN"),  # Above grad_high -> shrink
        (50.0, "scale DOWN"),  # Large spike -> shrink
        (400.0, "scale DOWN"), # Huge spike -> shrink
    ]

    for grad_norm, expected in test_cases:
        scale = apply_update_agc(None, grad_norm, params)

        # Verify scale behavior
        if expected == "scale UP":
            assert scale > 1.0, f"Expected scale > 1.0 for grad_norm={grad_norm}, got {scale}"
            # Check that normalized gradient is near grad_low
            normalized = grad_norm * scale
            assert 0.9 <= normalized <= 1.1, f"Expected normalized ~1.0, got {normalized}"
        elif expected == "scale DOWN":
            assert scale < 1.0, f"Expected scale < 1.0 for grad_norm={grad_norm}, got {scale}"
            # Check that normalized gradient is near grad_high
            normalized = grad_norm * scale
            assert 4.5 <= normalized <= 5.5, f"Expected normalized ~5.0, got {normalized}"
        else:  # no change
            assert scale == 1.0, f"Expected scale = 1.0 for grad_norm={grad_norm}, got {scale}"

        normalized = grad_norm * scale
        print(f"  grad_norm={grad_norm:6.1f} -> scale={scale:.4f} -> normalized={normalized:.2f} [{expected}] OK")

    print("  OK: Normalization test passed")


def test_agc_stateless():
    """Test 3: AGC is stateless (no memory between calls)"""
    print("\n=== Test 3: AGC Stateless Behavior ===")

    params = AGCParams(enabled=True, grad_low=1.0, grad_high=5.0)

    # Simulate gradient spikes: high -> low -> high
    # Old (buggy) AGC would accumulate scale changes
    # New (correct) AGC treats each call independently

    grad_sequence = [400.0, 0.5, 400.0, 0.5]

    for i, grad_norm in enumerate(grad_sequence):
        scale = apply_update_agc(None, grad_norm, params)
        normalized = grad_norm * scale

        # Each call should give the same result for the same input
        if grad_norm == 400.0:
            assert abs(scale - 5.0/400.0) < 1e-6, f"Scale should be consistent for grad_norm=400"
            assert 4.5 <= normalized <= 5.5, f"Normalized should be ~5.0"
        elif grad_norm == 0.5:
            assert abs(scale - 1.0/0.5) < 1e-6, f"Scale should be consistent for grad_norm=0.5"
            assert 0.9 <= normalized <= 1.1, f"Normalized should be ~1.0"

        print(f"  Step {i}: grad_norm={grad_norm:6.1f} -> scale={scale:.4f} -> normalized={normalized:.2f} OK")

    print("  OK: Stateless test passed (no memory between calls)")


def test_agc_invalid_inputs():
    """Test 4: AGC handles invalid inputs gracefully"""
    print("\n=== Test 4: Invalid Input Handling ===")

    params = AGCParams(enabled=True, grad_low=1.0, grad_high=5.0)

    # Test invalid gradient norms
    invalid_cases = [
        (None, "None"),
        (float('nan'), "NaN"),
        (float('inf'), "inf"),
    ]

    for grad_norm, name in invalid_cases:
        scale = apply_update_agc(None, grad_norm, params)
        assert scale == 1.0, f"Invalid input ({name}) should return 1.0, got {scale}"
        print(f"  grad_norm={name:6s} -> scale={scale:.4f} OK")

    print("  OK: Invalid input handling passed")


def main():
    print("=" * 60)
    print("AGC Fix Validation Suite")
    print("=" * 60)

    try:
        test_agc_disabled()
        test_agc_normalization()
        test_agc_stateless()
        test_agc_invalid_inputs()

        print("\n" + "=" * 60)
        print("OK: ALL TESTS PASSED")
        print("=" * 60)
        print("\nAGC fix validated successfully:")
        print("  - Stateless gradient normalization (no memory)")
        print("  - Target range [1.0, 5.0] enforced correctly")
        print("  - AGC OFF cleanly disables all scaling")
        print("  - Invalid inputs handled gracefully")
        print("\nNext steps:")
        print("  1. Run training probe: python tools/instnct_train_steps.py --task assoc_clean --steps 100")
        print("  2. Check logs for 'AGC: grad_norm X.XX -> scaled by Y.YYYY' messages")
        print("  3. Verify gradients never exceed 5.0 during training")

        return 0

    except AssertionError as e:
        print(f"\nFAIL: TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\nFAIL: UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
