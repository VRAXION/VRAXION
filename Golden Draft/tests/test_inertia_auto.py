"""Behavior locks for :mod:`vraxion.instnct.inertia_auto`.

Pins:
- disabled mode: no mutation
- panic_active: no mutation
- velocity-driven: inertia scales with velocity
- dwell-driven: inertia scales with dwell metric
- EMA smoothing of inertia
- clamp within [inertia_min, inertia_max]
"""

from __future__ import annotations

import unittest

import conftest  # noqa: F401

from vraxion.instnct.inertia_auto import InertiaAutoParams, apply_inertia_auto


class DummyModel:
    def __init__(self, ptr_inertia: float = 0.5):
        self.ptr_inertia = ptr_inertia


def _velocity_params(**overrides) -> InertiaAutoParams:
    defaults = dict(
        enabled=True,
        inertia_min=0.1,
        inertia_max=0.9,
        vel_full=10.0,
        ema_beta=0.0,
        dwell_enabled=False,
        dwell_thresh=0.0,
    )
    defaults.update(overrides)
    return InertiaAutoParams(**defaults)


def _dwell_params(**overrides) -> InertiaAutoParams:
    defaults = dict(
        enabled=True,
        inertia_min=0.1,
        inertia_max=0.9,
        vel_full=10.0,
        ema_beta=0.0,
        dwell_enabled=True,
        dwell_thresh=5.0,
    )
    defaults.update(overrides)
    return InertiaAutoParams(**defaults)


class TestInertiaAutoDisabled(unittest.TestCase):

    def test_disabled_no_mutation(self) -> None:
        model = DummyModel(ptr_inertia=0.5)
        params = _velocity_params(enabled=False)
        apply_inertia_auto(model, ptr_velocity=5.0, params=params)
        self.assertEqual(model.ptr_inertia, 0.5)

    def test_panic_active_no_mutation(self) -> None:
        model = DummyModel(ptr_inertia=0.5)
        params = _velocity_params()
        apply_inertia_auto(model, ptr_velocity=5.0, params=params, panic_active=True)
        self.assertEqual(model.ptr_inertia, 0.5)


class TestInertiaAutoVelocity(unittest.TestCase):

    def test_zero_velocity_gives_min_inertia(self) -> None:
        model = DummyModel(ptr_inertia=0.5)
        params = _velocity_params(ema_beta=0.0)
        apply_inertia_auto(model, ptr_velocity=0.0, params=params)
        self.assertAlmostEqual(model.ptr_inertia, 0.1, places=6)

    def test_full_velocity_gives_max_inertia(self) -> None:
        model = DummyModel(ptr_inertia=0.5)
        params = _velocity_params(ema_beta=0.0, vel_full=10.0)
        apply_inertia_auto(model, ptr_velocity=10.0, params=params)
        self.assertAlmostEqual(model.ptr_inertia, 0.9, places=6)

    def test_half_velocity_gives_midpoint(self) -> None:
        model = DummyModel(ptr_inertia=0.5)
        params = _velocity_params(ema_beta=0.0, vel_full=10.0)
        apply_inertia_auto(model, ptr_velocity=5.0, params=params)
        self.assertAlmostEqual(model.ptr_inertia, 0.5, places=6)

    def test_none_velocity_no_mutation(self) -> None:
        model = DummyModel(ptr_inertia=0.5)
        params = _velocity_params()
        apply_inertia_auto(model, ptr_velocity=None, params=params)
        self.assertEqual(model.ptr_inertia, 0.5)


class TestInertiaAutoDwell(unittest.TestCase):

    def test_zero_dwell_gives_min_inertia(self) -> None:
        model = DummyModel(ptr_inertia=0.5)
        model.ptr_mean_dwell = 0.0
        params = _dwell_params(ema_beta=0.0, dwell_thresh=5.0)
        apply_inertia_auto(model, ptr_velocity=None, params=params)
        self.assertAlmostEqual(model.ptr_inertia, 0.1, places=6)

    def test_full_dwell_gives_max_inertia(self) -> None:
        model = DummyModel(ptr_inertia=0.5)
        model.ptr_mean_dwell = 5.0
        params = _dwell_params(ema_beta=0.0, dwell_thresh=5.0)
        apply_inertia_auto(model, ptr_velocity=None, params=params)
        self.assertAlmostEqual(model.ptr_inertia, 0.9, places=6)

    def test_zero_thresh_gives_max(self) -> None:
        model = DummyModel(ptr_inertia=0.5)
        model.ptr_mean_dwell = 1.0
        params = _dwell_params(ema_beta=0.0, dwell_thresh=0.0)
        apply_inertia_auto(model, ptr_velocity=None, params=params)
        self.assertAlmostEqual(model.ptr_inertia, 0.9, places=6)


class TestInertiaAutoEMA(unittest.TestCase):

    def test_ema_smoothing(self) -> None:
        model = DummyModel(ptr_inertia=0.5)
        params = _velocity_params(ema_beta=0.9, vel_full=10.0)
        apply_inertia_auto(model, ptr_velocity=10.0, params=params)
        # With beta=0.9: ema = 0.9*0.5 + 0.1*0.9 = 0.54
        self.assertAlmostEqual(model.ptr_inertia, 0.54, places=6)

    def test_ema_attribute_set(self) -> None:
        model = DummyModel(ptr_inertia=0.5)
        params = _velocity_params(ema_beta=0.5)
        apply_inertia_auto(model, ptr_velocity=5.0, params=params)
        self.assertTrue(hasattr(model, "ptr_inertia_ema"))
        self.assertEqual(model.ptr_inertia, model.ptr_inertia_ema)


if __name__ == "__main__":
    unittest.main()
