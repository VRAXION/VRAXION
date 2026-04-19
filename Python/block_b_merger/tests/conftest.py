"""Pytest configuration for Block B merger tests.

Inserts the repo root into sys.path so that imports resolve without going
through the Python package __init__ chain (which has a stale reference to
Python.byte_encoder that lives in Python/block_a_byte_unit/ now).
"""
import sys
from pathlib import Path

# Repo root = 4 levels up from this file
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Also ensure block_b_merger itself is importable without the broken parent __init__
import importlib.util as _ilu
import types as _types

# If 'Python' is already in sys.modules and broken, patch it with a stub
# that won't raise on import. We only do this if the real __init__ fails.
if "Python" not in sys.modules:
    _pkg = _types.ModuleType("Python")
    _pkg.__path__ = [str(_REPO_ROOT / "Python")]
    _pkg.__package__ = "Python"
    sys.modules["Python"] = _pkg
