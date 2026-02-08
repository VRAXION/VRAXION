"""Behavior locks for :mod:`vraxion.instnct.panic`.

Pins:
- first call initializes EMA and returns INIT status
- stable loss returns LOCKED status with high inertia
- loss spike triggers PANIC with reduced inertia and walk_prob
- panic recovers gradually
- recovery rate controls decay speed
"""

from __future__ import annotations

import unittest

import conftest  # noqa: F401

from vraxion.instnct.panic import PanicReflex


class TestPanicReflex(unittest.TestCase):

    def test_first_call_returns_init(self) -> None:
        reflex = PanicReflex()
        result = reflex.update(1.0)
        self.assertEqual(result["status"], "INIT")
        self.assertAlmostEqual(result["inertia"], reflex.inertia_high)
        self.assertAlmostEqual(result["walk_prob"], 0.0)

    def test_stable_loss_returns_locked(self) -> None:
        reflex = PanicReflex()
        reflex.update(1.0)  # Init.
        result = reflex.update(1.0)
        self.assertEqual(result["status"], "LOCKED")
        self.assertAlmostEqual(result["inertia"], reflex.inertia_high)
        self.assertAlmostEqual(result["walk_prob"], 0.0)

    def test_loss_spike_triggers_panic(self) -> None:
        reflex = PanicReflex(panic_threshold=1.5)
        reflex.update(1.0)  # Init EMA to 1.0.
        result = reflex.update(2.0)  # ratio = 2.0 > 1.5 threshold.
        self.assertEqual(result["status"], "PANIC")
        self.assertLess(result["inertia"], reflex.inertia_high)
        self.assertGreater(result["walk_prob"], 0.0)

    def test_panic_inertia_is_low_at_full_panic(self) -> None:
        reflex = PanicReflex(panic_threshold=1.5)
        reflex.update(1.0)
        reflex.update(100.0)  # Massive spike.
        self.assertAlmostEqual(reflex.panic_state, 1.0)

    def test_panic_recovers_gradually(self) -> None:
        reflex = PanicReflex(panic_threshold=1.5, recovery_rate=0.1)
        reflex.update(1.0)
        reflex.update(100.0)  # Spike.
        self.assertAlmostEqual(reflex.panic_state, 1.0)

        # Feed stable losses — panic should decay.
        for _ in range(5):
            reflex.update(reflex.loss_ema * 0.9)

        self.assertLess(reflex.panic_state, 1.0)

    def test_full_recovery_to_locked(self) -> None:
        reflex = PanicReflex(panic_threshold=1.5, recovery_rate=0.5)
        reflex.update(1.0)
        reflex.update(100.0)  # Spike.

        # Feed many stable losses.
        for _ in range(50):
            result = reflex.update(reflex.loss_ema * 0.9)

        self.assertEqual(result["status"], "LOCKED")
        self.assertAlmostEqual(reflex.panic_state, 0.0)

    def test_walk_prob_bounded(self) -> None:
        reflex = PanicReflex(walk_prob_max=0.2)
        reflex.update(1.0)
        result = reflex.update(100.0)
        self.assertLessEqual(result["walk_prob"], 0.2)

    def test_ema_tracks_loss(self) -> None:
        reflex = PanicReflex(ema_beta=0.5)
        reflex.update(1.0)  # EMA = 1.0.
        reflex.update(3.0)  # EMA = 0.5*1.0 + 0.5*3.0 = 2.0.
        self.assertAlmostEqual(reflex.loss_ema, 2.0, places=6)


if __name__ == "__main__":
    unittest.main()
