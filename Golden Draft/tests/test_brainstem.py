"""Behavior locks for :mod:`vraxion.instnct.brainstem`.

Pins:
- DEEP mode default, w starts at 0
- high entropy triggers SHIELD (Schmitt engage)
- SHIELD holds for hold_steps minimum
- release requires safe_steps consecutive low-danger readings
- hysteresis: engage > release threshold
- continuous damping in DEEP mode
"""

from __future__ import annotations

import unittest

import conftest  # noqa: F401

from vraxion.instnct.brainstem import BrainstemMixer, BrainstemMixerConfig


class TestBrainstemMixer(unittest.TestCase):

    def test_initial_state_is_deep(self) -> None:
        mixer = BrainstemMixer()
        self.assertEqual(mixer.mode, "DEEP")
        self.assertEqual(mixer.w, 0.0)

    def test_low_entropy_stays_deep(self) -> None:
        mixer = BrainstemMixer()
        w, meta = mixer.update(current_entropy=0.1)
        self.assertEqual(meta["mode"], "DEEP")
        self.assertLess(w, 1.0)

    def test_high_entropy_triggers_shield(self) -> None:
        mixer = BrainstemMixer()
        w, meta = mixer.update(current_entropy=5.0)
        self.assertEqual(meta["mode"], "SHIELD_TRIP")
        self.assertEqual(w, 1.0)
        self.assertEqual(mixer.mode, "SHIELD")

    def test_shield_holds_minimum_steps(self) -> None:
        cfg = BrainstemMixerConfig(hold_steps=5, safe_steps=1)
        mixer = BrainstemMixer(cfg)

        # Trip into shield.
        mixer.update(current_entropy=5.0)
        self.assertEqual(mixer.mode, "SHIELD")

        # Feed low entropy — should stay in SHIELD during hold.
        for _ in range(4):
            w, meta = mixer.update(current_entropy=0.01)
            self.assertEqual(w, 1.0)

    def test_shield_releases_after_safe_steps(self) -> None:
        cfg = BrainstemMixerConfig(hold_steps=1, safe_steps=2)
        mixer = BrainstemMixer(cfg)

        mixer.update(current_entropy=5.0)
        self.assertEqual(mixer.mode, "SHIELD")

        # Burn hold.
        mixer.update(current_entropy=0.01)

        # First safe step.
        mixer.update(current_entropy=0.01)
        # Second safe step — should release.
        w, meta = mixer.update(current_entropy=0.01)
        self.assertEqual(mixer.mode, "DEEP")

    def test_hysteresis_thresholds(self) -> None:
        cfg = BrainstemMixerConfig()
        # engage_threshold (0.618) > release_threshold (0.382) by design.
        self.assertGreater(cfg.engage_threshold, cfg.release_threshold)

    def test_custom_config_applied(self) -> None:
        cfg = BrainstemMixerConfig(hold_steps=99, safe_steps=42)
        mixer = BrainstemMixer(cfg)
        self.assertEqual(mixer.cfg.hold_steps, 99)
        self.assertEqual(mixer.cfg.safe_steps, 42)

    def test_sigmo_bounds(self) -> None:
        self.assertAlmostEqual(BrainstemMixer._sigmo(0.0), 0.5, places=6)
        self.assertGreater(BrainstemMixer._sigmo(100.0), 0.99)
        self.assertLess(BrainstemMixer._sigmo(-100.0), 0.01)

    def test_clp01_clamps(self) -> None:
        self.assertEqual(BrainstemMixer._clp01(-0.5), 0.0)
        self.assertEqual(BrainstemMixer._clp01(1.5), 1.0)
        self.assertAlmostEqual(BrainstemMixer._clp01(0.7), 0.7)

    def test_w_monotonic_increase_under_rising_entropy(self) -> None:
        mixer = BrainstemMixer()
        prev_w = 0.0
        for ent in [0.3, 0.4, 0.5, 0.55]:
            w, _ = mixer.update(current_entropy=ent)
            self.assertGreaterEqual(w, prev_w - 1e-9)
            prev_w = w


if __name__ == "__main__":
    unittest.main()
