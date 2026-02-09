"""Brainstem mixer — shield/deep Schmitt trigger.

Contract (do not break):
  - Preserve the Schmitt trigger behavior + smoothing.
  - Keep env override names stable (VRX_BRAINSTEM_*).

The mixer outputs a *shield weight* ``w``:
  - w == 1.0 => Shield / FAST
  - w == 0.0 => Deep  / SLOW

Callers that mix as ``mix*slow + (1-mix)*fast`` must use: mix = 1 - w.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class BrainstemMixerConfig:
    """Hyperparameters for :class:`BrainstemMixer`."""

    phi: float = (1.0 + 5.0**0.5) / 2.0
    engage_threshold: float = 1.0 / ((1.0 + 5.0**0.5) / 2.0)
    release_threshold: float = 1.0 / (((1.0 + 5.0**0.5) / 2.0) ** 2)

    hold_steps: int = 13
    safe_steps: int = 5
    tau_release: float = 21.0
    tau_attack: float = 2.0

    w_entropy: float = 0.85
    entropy_mid: float = 0.60
    entropy_scale: float = 0.08


class BrainstemMixer:
    """Schmitt-trigger mixer producing a shield weight ``w`` in ``[0, 1]``."""

    def __init__(self, cfg: Optional[BrainstemMixerConfig] = None):
        self.cfg = cfg or BrainstemMixerConfig()
        self._apply_env()

        self.mode = "DEEP"
        self.w = 0.0
        self._hold = 0
        self._safe = 0

    def _apply_env(self) -> None:
        """Apply VRX_BRAINSTEM_* overrides."""
        try:
            self.cfg.engage_threshold = float(
                os.environ.get("VRX_BRAINSTEM_ENGAGE", self.cfg.engage_threshold)
            )
            self.cfg.release_threshold = float(
                os.environ.get("VRX_BRAINSTEM_RELEASE", self.cfg.release_threshold)
            )
            self.cfg.hold_steps = int(
                os.environ.get("VRX_BRAINSTEM_HOLD_STEPS", self.cfg.hold_steps)
            )
            self.cfg.safe_steps = int(
                os.environ.get("VRX_BRAINSTEM_SAFE_STEPS", self.cfg.safe_steps)
            )
            self.cfg.tau_attack = float(
                os.environ.get("VRX_BRAINSTEM_TAU_ATTACK", self.cfg.tau_attack)
            )
            self.cfg.tau_release = float(
                os.environ.get("VRX_BRAINSTEM_TAU_RELEASE", self.cfg.tau_release)
            )
            self.cfg.w_entropy = float(
                os.environ.get("VRX_BRAINSTEM_W_ENTROPY", self.cfg.w_entropy)
            )
            self.cfg.entropy_mid = float(
                os.environ.get("VRX_BRAINSTEM_ENTROPY_MID", self.cfg.entropy_mid)
            )
            self.cfg.entropy_scale = float(
                os.environ.get("VRX_BRAINSTEM_ENTROPY_SCALE", self.cfg.entropy_scale)
            )
        except Exception:
            pass

    @staticmethod
    def _sigmo(x: float) -> float:
        if x >= 0:
            expval = math.exp(-x)
            return 1.0 / (1.0 + expval)
        expval = math.exp(x)
        return expval / (1.0 + expval)

    @staticmethod
    def _clp01(x: float) -> float:
        return max(0.0, min(1.0, float(x)))

    def update(
        self,
        current_entropy: float,
        repetition_count: int = 0,
        dt: float = 1.0,
    ) -> Tuple[float, Dict]:
        """Advance internal state and return ``(w, meta)``."""

        entsco = self._clp01(
            self._sigmo((current_entropy - self.cfg.entropy_mid) / self.cfg.entropy_scale)
        )

        dangvl = self._clp01(self.cfg.w_entropy * entsco)

        if self.mode == "SHIELD":
            if self._hold > 0:
                self._hold -= 1

            if dangvl <= self.cfg.release_threshold:
                self._safe += 1
            else:
                self._safe = 0

            if (self._hold == 0) and (self._safe >= self.cfg.safe_steps):
                self.mode = "DEEP"
            else:
                self.w = 1.0
                return self.w, {"mode": "SHIELD", "danger": dangvl}

        if dangvl >= self.cfg.engage_threshold:
            self.mode = "SHIELD"
            self.w = 1.0
            self._hold = int(self.cfg.hold_steps)
            return self.w, {"mode": "SHIELD_TRIP", "danger": dangvl}

        trgval = self._clp01(dangvl**2.0)

        tauval = self.cfg.tau_attack if trgval > self.w else self.cfg.tau_release
        alphav = 1.0 - math.exp(-dt / max(float(tauval), 1e-6))
        self.w = self._clp01(self.w + alphav * (trgval - self.w))

        return self.w, {"mode": "DEEP", "danger": dangvl, "w": self.w}
