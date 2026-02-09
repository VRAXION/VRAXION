"""Timestamped logging.

Single concern: print + append to LOG_PATH.
"""

from __future__ import annotations

import os
import time


# Callers may override at runtime.
ROOT = os.getcwd()
LOG_PATH = os.path.join(ROOT, "logs", "current", "vraxion.log")


def log(msg: str) -> None:
    """Print a timestamped message and append it to ``LOG_PATH``."""
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)

    os.makedirs(os.path.dirname(LOG_PATH) or ".", exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as filobj:
        filobj.write(line + "\n")
