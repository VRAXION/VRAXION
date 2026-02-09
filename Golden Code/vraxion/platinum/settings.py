"""Platinum settings.

Thin compatibility wrapper — canonical implementation lives in :mod:`vraxion.settings`.
"""

from __future__ import annotations

from vraxion.settings import Settings, load_settings

__all__ = [
    "Settings",
    "load_settings",
]
