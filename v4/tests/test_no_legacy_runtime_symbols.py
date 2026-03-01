"""Anti-legacy drift guard — fail if forbidden symbols appear in runtime files.

Prevents regression back to removed APIs (learnable R_param, etc.).
Part of the consolidation pass (2026-02-28).
"""
from pathlib import Path

# runtime files that must never contain legacy symbols
RUNTIME_FILES = [
    Path("model/instnct.py"),
    Path("training/train.py"),
    Path("training/eval.py"),
]

# symbols that were removed from the codebase — their presence means regression
FORBIDDEN = ["R_param"]


def test_no_legacy_symbols_in_runtime():
    """Scan runtime files for forbidden legacy symbols."""
    root = Path(__file__).resolve().parents[1]
    for rel in RUNTIME_FILES:
        fpath = root / rel
        if not fpath.exists():
            continue
        text = fpath.read_text(encoding="utf-8")
        for sym in FORBIDDEN:
            assert sym not in text, f"{sym} found in runtime file: {rel}"
