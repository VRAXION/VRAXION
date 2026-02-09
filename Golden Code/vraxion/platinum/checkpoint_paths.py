"""Checkpoint path resolution and artifact rotation."""

from __future__ import annotations

import math
import os
import shutil
import time
from typing import Any, List, Tuple

import torch

from .log import log


# Callers may override at runtime.
ROOT = os.getcwd()

# Checkpoint payload defaults.
USE_AMP = False
UPDATE_SCALE = 1.0
PTR_INERTIA = 0.0
EXPERT_HEADS = 1


def rotate_artifacts() -> None:
    """Move current logs/traces/summaries -> last, and last -> archive/<timestamp>."""
    ts = time.strftime("%Y%m%d_%H%M%S")

    def _rotate_dir(base_dir: str) -> None:
        curdir = os.path.join(base_dir, "current")
        lasdir = os.path.join(base_dir, "last")
        arcdir = os.path.join(base_dir, "archive")

        os.makedirs(curdir, exist_ok=True)
        os.makedirs(lasdir, exist_ok=True)
        os.makedirs(arcdir, exist_ok=True)

        if os.path.isdir(lasdir) and os.listdir(lasdir):
            run_dir = os.path.join(arcdir, ts)
            os.makedirs(run_dir, exist_ok=True)
            for name in os.listdir(lasdir):
                shutil.move(os.path.join(lasdir, name), os.path.join(run_dir, name))

        if os.path.isdir(curdir) and os.listdir(curdir):
            for name in os.listdir(curdir):
                shutil.move(os.path.join(curdir, name), os.path.join(lasdir, name))

    _rotate_dir(os.path.join(ROOT, "logs"))
    _rotate_dir(os.path.join(ROOT, "traces"))
    _rotate_dir(os.path.join(ROOT, "summaries"))


def sync_current_to_last() -> None:
    """Copy current logs/traces/summaries into ``logs/last``."""
    dstdir = os.path.join(ROOT, "logs", "last")
    os.makedirs(dstdir, exist_ok=True)

    for rel in ("logs/current", "traces/current", "summaries/current"):
        srcdir = os.path.join(ROOT, rel)
        if not os.path.isdir(srcdir):
            continue
        for name in os.listdir(srcdir):
            srcpth = os.path.join(srcdir, name)
            if os.path.isfile(srcpth):
                shutil.copy2(srcpth, os.path.join(dstdir, name))


def checkpoint_paths(base_path: str, step: int) -> Tuple[str, str, str]:
    """Return (step_path, bad_path, last_good_path) for a checkpoint base path."""
    base_dir = os.path.dirname(base_path) or ROOT
    base_name = os.path.splitext(os.path.basename(base_path))[0]

    hisdir = os.path.join(base_dir, "checkpoints")
    os.makedirs(hisdir, exist_ok=True)

    step_path = os.path.join(hisdir, f"{base_name}_step_{step:07d}.pt")
    bad_path = os.path.join(hisdir, f"{base_name}_bad_step_{step:07d}.pt")
    last_good_path = os.path.join(base_dir, f"{base_name}_last_good.pt")
    return step_path, bad_path, last_good_path


def checkpoint_is_finite(loss_value: Any, grad_norm_value: Any, raw_delta_value: Any) -> bool:
    """Return False if any provided scalar is non-finite."""
    for value in (loss_value, grad_norm_value, raw_delta_value):
        if value is None:
            continue
        if not math.isfinite(float(value)):
            return False
    return True


def checkpoint_payload(model: Any, optimizer: Any, scaler: Any, step: int, losses: List[float]) -> dict:
    """Build a checkpoint payload dict compatible with the Platinum runtime."""
    return {
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
        "scaler": scaler.state_dict() if USE_AMP else None,
        "step": step,
        "losses": losses,
        "update_scale": 1.0,
        "ptr_inertia": getattr(model, "ptr_inertia", PTR_INERTIA),
        "ptr_inertia_ema": getattr(model, "ptr_inertia_ema", getattr(model, "ptr_inertia", PTR_INERTIA)),
        "ptr_inertia_floor": getattr(model, "ptr_inertia_floor", 0.0),
        "agc_scale_max": 1.0,
        "ground_speed_ema": getattr(model, "ground_speed_ema", None),
        "ground_speed_limit": getattr(model, "ground_speed_limit", None),
        "num_experts": getattr(getattr(model, "head", None), "num_experts", EXPERT_HEADS),
        "param_names": [name for name, _ in model.named_parameters()],
    }
