"""Behavior locks for :mod:`vraxion.instnct.agc`.

Pins:
- scale-up when grad_norm < grad_low
- scale-down when grad_norm > grad_high
- clamp within [floor, cap]
- warmup floor ramp
- step-0 initialization
- disabled AGC passthrough
- NaN/Inf grad_norm safety
"""

from __future__ import annotations

import math
import unittest

import conftest  # noqa: F401

from vraxion.instnct.agc import AGCParams, apply_update_agc


class DummyModel:
    def __init__(self, update_scale: float = 0.5):
        self.update_scale = update_scale


def _default_params(**overrides) -> AGCParams:
    defaults = dict(
        enabled=True,
        grad_low=0.1,
        grad_high=1.0,
        scale_up=1.1,
        scale_down=0.9,
        scale_min=0.01,
        scale_max_default=2.0,
        warmup_steps=10,
        warmup_init=0.001,
    )
    defaults.update(overrides)
    return AGCParams(**defaults)


class TestAGC(unittest.TestCase):

    def test_scale_up_on_low_grad(self) -> None:
        model = DummyModel(update_scale=0.5)
        params = _default_params()
        result = apply_update_agc(model, grad_norm=0.05, params=params)
        self.assertAlmostEqual(result, 0.5 * 1.1, places=6)
        self.assertAlmostEqual(model.update_scale, 0.5 * 1.1, places=6)

    def test_scale_down_on_high_grad(self) -> None:
        model = DummyModel(update_scale=0.5)
        params = _default_params()
        result = apply_update_agc(model, grad_norm=1.5, params=params)
        self.assertAlmostEqual(result, 0.5 * 0.9, places=6)

    def test_no_change_in_band(self) -> None:
        model = DummyModel(update_scale=0.5)
        params = _default_params()
        result = apply_update_agc(model, grad_norm=0.5, params=params)
        self.assertAlmostEqual(result, 0.5, places=6)

    def test_clamp_to_cap(self) -> None:
        model = DummyModel(update_scale=1.95)
        params = _default_params(scale_max_default=2.0)
        result = apply_update_agc(model, grad_norm=0.01, params=params)
        self.assertLessEqual(result, 2.0)

    def test_clamp_to_floor(self) -> None:
        model = DummyModel(update_scale=0.015)
        params = _default_params(scale_min=0.01)
        result = apply_update_agc(model, grad_norm=5.0, params=params)
        self.assertGreaterEqual(result, 0.01)

    def test_warmup_floor_at_step_zero(self) -> None:
        model = DummyModel(update_scale=0.5)
        params = _default_params(warmup_init=0.001, warmup_steps=10, scale_min=0.01)
        result = apply_update_agc(model, grad_norm=0.5, params=params, step=0)
        self.assertAlmostEqual(result, 0.001, places=6)

    def test_warmup_floor_at_full(self) -> None:
        model = DummyModel(update_scale=0.5)
        params = _default_params(warmup_init=0.001, warmup_steps=10, scale_min=0.01)
        result = apply_update_agc(model, grad_norm=0.5, params=params, step=10)
        self.assertAlmostEqual(result, 0.5, places=6)

    def test_disabled_agc_no_scaling(self) -> None:
        model = DummyModel(update_scale=0.5)
        params = _default_params(enabled=False)
        result = apply_update_agc(model, grad_norm=0.01, params=params)
        self.assertAlmostEqual(result, 0.5, places=6)

    def test_nan_grad_norm_no_scaling(self) -> None:
        model = DummyModel(update_scale=0.5)
        params = _default_params()
        result = apply_update_agc(model, grad_norm=float("nan"), params=params)
        self.assertAlmostEqual(result, 0.5, places=6)

    def test_none_grad_norm_no_scaling(self) -> None:
        model = DummyModel(update_scale=0.5)
        params = _default_params()
        result = apply_update_agc(model, grad_norm=None, params=params)
        self.assertAlmostEqual(result, 0.5, places=6)

    def test_model_attributes_set(self) -> None:
        model = DummyModel(update_scale=0.5)
        params = _default_params()
        apply_update_agc(model, grad_norm=0.5, params=params)
        self.assertTrue(hasattr(model, "agc_scale_cap"))
        self.assertTrue(hasattr(model, "debug_scale_out"))
        self.assertEqual(model.update_scale, model.debug_scale_out)


    def test_disabled_agc_preserves_scale_during_warmup(self) -> None:
        """AGC OFF must not override update_scale during warmup steps."""
        model = DummyModel(update_scale=0.5)
        params = _default_params(enabled=False, warmup_init=0.001, warmup_steps=10, scale_min=0.01)
        # Step 0 — previously would force scale to warmup_init (0.001)
        result = apply_update_agc(model, grad_norm=0.05, params=params, step=0)
        self.assertAlmostEqual(result, 0.5, places=6)
        # Mid-warmup — previously would clamp to warmup floor
        model2 = DummyModel(update_scale=0.5)
        result2 = apply_update_agc(model2, grad_norm=0.05, params=params, step=5)
        self.assertAlmostEqual(result2, 0.5, places=6)
        # Post-warmup — scale must still be preserved
        model3 = DummyModel(update_scale=0.5)
        result3 = apply_update_agc(model3, grad_norm=0.05, params=params, step=10)
        self.assertAlmostEqual(result3, 0.5, places=6)


if __name__ == "__main__":
    unittest.main()
