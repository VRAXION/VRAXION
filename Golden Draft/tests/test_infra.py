"""Behavior locks for :mod:`vraxion.instnct.infra`.

Pins:
- _parse_csv_ints / _parse_csv_floats correctness
- _normalize_weights sum-to-1 invariant
- compute_slope least-squares result
- nan_guard behavior (active only when DEBUG_NAN=True)
- _checkpoint_is_finite NaN/Inf detection
- StaircaseBatcher weighted iteration
"""

from __future__ import annotations

import math
import unittest

import conftest  # noqa: F401

import torch

from vraxion.instnct.infra import (
    StaircaseBatcher,
    StaircaseController,
    _checkpoint_is_finite,
    _normalize_weights,
    _parse_csv_floats,
    _parse_csv_ints,
    compute_slope,
    nan_guard,
)


class TestParseCsvInts(unittest.TestCase):

    def test_valid(self) -> None:
        self.assertEqual(_parse_csv_ints("1,2,3"), [1, 2, 3])

    def test_spaces(self) -> None:
        self.assertEqual(_parse_csv_ints(" 4 , 5 , 6 "), [4, 5, 6])

    def test_empty_returns_none(self) -> None:
        self.assertIsNone(_parse_csv_ints(""))

    def test_negative_returns_none(self) -> None:
        self.assertIsNone(_parse_csv_ints("1,-2,3"))

    def test_zero_returns_none(self) -> None:
        self.assertIsNone(_parse_csv_ints("0,1"))

    def test_non_int_returns_none(self) -> None:
        self.assertIsNone(_parse_csv_ints("a,b"))


class TestParseCsvFloats(unittest.TestCase):

    def test_valid(self) -> None:
        self.assertEqual(_parse_csv_floats("1.5,2.5"), [1.5, 2.5])

    def test_empty_returns_none(self) -> None:
        self.assertIsNone(_parse_csv_floats(""))

    def test_non_float_returns_none(self) -> None:
        self.assertIsNone(_parse_csv_floats("abc"))


class TestNormalizeWeights(unittest.TestCase):

    def test_sums_to_one(self) -> None:
        result = _normalize_weights([3.0, 1.0, 1.0])
        self.assertAlmostEqual(sum(result), 1.0, places=9)

    def test_proportions(self) -> None:
        result = _normalize_weights([2.0, 2.0])
        self.assertAlmostEqual(result[0], 0.5)
        self.assertAlmostEqual(result[1], 0.5)

    def test_negative_treated_as_zero(self) -> None:
        result = _normalize_weights([-1.0, 2.0])
        self.assertAlmostEqual(result[0], 0.0)
        self.assertAlmostEqual(result[1], 1.0)

    def test_all_zero_uniform(self) -> None:
        result = _normalize_weights([0.0, 0.0, 0.0])
        for w in result:
            self.assertAlmostEqual(w, 1.0 / 3.0)


class TestComputeSlope(unittest.TestCase):

    def test_increasing(self) -> None:
        slope = compute_slope([1.0, 2.0, 3.0, 4.0])
        self.assertGreater(slope, 0.0)
        self.assertAlmostEqual(slope, 1.0, places=6)

    def test_decreasing(self) -> None:
        slope = compute_slope([4.0, 3.0, 2.0, 1.0])
        self.assertLess(slope, 0.0)
        self.assertAlmostEqual(slope, -1.0, places=6)

    def test_flat(self) -> None:
        slope = compute_slope([5.0, 5.0, 5.0])
        self.assertAlmostEqual(slope, 0.0, places=6)

    def test_single_returns_nan(self) -> None:
        self.assertTrue(math.isnan(compute_slope([1.0])))

    def test_empty_returns_nan(self) -> None:
        self.assertTrue(math.isnan(compute_slope([])))


class TestNanGuard(unittest.TestCase):

    def test_no_raise_when_disabled(self) -> None:
        import vraxion.instnct.infra as infra

        old = infra.DEBUG_NAN
        try:
            infra.DEBUG_NAN = False
            nan_guard("test", torch.tensor([float("nan")]), step=0)
        finally:
            infra.DEBUG_NAN = old

    def test_raises_when_enabled_and_nan(self) -> None:
        import vraxion.instnct.infra as infra

        old = infra.DEBUG_NAN
        try:
            infra.DEBUG_NAN = True
            with self.assertRaises(RuntimeError):
                nan_guard("test", torch.tensor([float("nan")]), step=0)
        finally:
            infra.DEBUG_NAN = old

    def test_no_raise_when_enabled_and_clean(self) -> None:
        import vraxion.instnct.infra as infra

        old = infra.DEBUG_NAN
        try:
            infra.DEBUG_NAN = True
            nan_guard("test", torch.tensor([1.0, 2.0]), step=0)
        finally:
            infra.DEBUG_NAN = old


class TestCheckpointIsFinite(unittest.TestCase):

    def test_all_finite(self) -> None:
        self.assertTrue(_checkpoint_is_finite(0.5, 1.0, 0.01))

    def test_nan_loss(self) -> None:
        self.assertFalse(_checkpoint_is_finite(float("nan"), 1.0, 0.01))

    def test_inf_grad(self) -> None:
        self.assertFalse(_checkpoint_is_finite(0.5, float("inf"), 0.01))

    def test_none_values_ok(self) -> None:
        self.assertTrue(_checkpoint_is_finite(None, None, None))


class _FakeLoader:
    """Minimal loader stub with .dataset attribute."""

    def __init__(self, items):
        self._items = items
        self.dataset = items

    def __iter__(self):
        return iter(self._items)


class TestStaircaseBatcher(unittest.TestCase):

    def test_iterates_and_resets(self) -> None:
        loader_a = _FakeLoader([1, 2])
        loader_b = _FakeLoader([10, 20])
        batcher = StaircaseBatcher([loader_a, loader_b], [1.0, 0.0], rng_seed=42)
        results = [next(batcher) for _ in range(4)]
        self.assertEqual(results, [1, 2, 1, 2])

    def test_weighted_distribution(self) -> None:
        loader_a = _FakeLoader(list(range(100)))
        loader_b = _FakeLoader(list(range(100, 200)))
        batcher = StaircaseBatcher([loader_a, loader_b], [0.8, 0.2], rng_seed=0)
        results = [next(batcher) for _ in range(200)]
        from_b = sum(1 for r in results if r >= 100)
        self.assertGreater(from_b, 10)
        self.assertLess(from_b, 100)


if __name__ == "__main__":
    unittest.main()
