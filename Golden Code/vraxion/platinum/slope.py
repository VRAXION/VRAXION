"""Loss slope estimation via least-squares regression."""

from __future__ import annotations

from typing import Sequence


def compute_slope(losses: Sequence[float]) -> float:
    """Compute a least-squares slope over the loss history."""
    if len(losses) < 2:
        return float("nan")

    num = float(len(losses))
    sum_x = float(len(losses) - 1) * num / 2.0
    sum_x2 = float(len(losses) - 1) * num * float(2 * len(losses) - 1) / 6.0

    sum_y = 0.0
    sum_xy = 0.0
    for idx, val in enumerate(losses):
        y = float(val)
        sum_y += y
        sum_xy += float(idx) * y

    denom = num * sum_x2 - sum_x * sum_x
    if denom == 0.0:
        return float("nan")
    return (num * sum_xy - sum_x * sum_y) / denom
