"""pytest configuration — adds v4 source dirs to sys.path so tests can import
model, training, and datagen modules without installing the package."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # v4/

for subdir in ('model', 'training', 'datagen'):
    p = str(ROOT / subdir)
    if p not in sys.path:
        sys.path.insert(0, p)
