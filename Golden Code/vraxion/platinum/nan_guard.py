"""NaN/Inf detection guard."""

from __future__ import annotations

import torch

from .log import log


# Callers may override at runtime.
DEBUG_NAN = False


def nan_guard(name: str, tensor: torch.Tensor, step: int) -> None:
    """Raise if tensor contains NaN/Inf (only when ``DEBUG_NAN`` is True)."""
    if not DEBUG_NAN:
        return
    if not tensor.is_floating_point():
        return
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        log(f"[nan_guard] step={step:04d} tensor={name} has NaN/Inf")
        raise RuntimeError(f"NaN/Inf in {name} at step {step}")
