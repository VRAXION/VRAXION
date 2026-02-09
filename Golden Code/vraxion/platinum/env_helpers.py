"""Centralized environment variable helpers.

Every VRX_* env parse goes through these three functions.
No duplication across modules.
"""

from __future__ import annotations

import os


def env_is_one(name: str, default: bool = False) -> bool:
    """Legacy boolean semantics: only the literal string '1' enables."""
    val = os.environ.get(name)
    if val is None:
        return default
    return str(val).strip() == "1"


def env_float(name: str, default: float) -> float:
    val = os.environ.get(name)
    if val is None or str(val).strip() == "":
        return float(default)
    try:
        return float(val)
    except Exception:
        return float(default)


def env_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    if val is None or str(val).strip() == "":
        return int(default)
    try:
        return int(float(val))
    except Exception:
        return int(default)
