"""VRAXION Python deploy SDK.

Minimal, deploy-ready Python API over the frozen byte-level pipeline.

Currently exposes:
  - ByteEncoder  (Block A — byte unit, L0, 8 → 16 → 8 tied-mirror autoencoder)
  - L1Merger     (Block B — byte-pair merger, L1, 32 → 81 → 32 single-W mirror)

See also the parallel Rust/ deploy SDK; both read the same champion
artifacts from the repo's output/ directory.
"""
from .block_a_byte_unit import ByteEncoder
from .block_b_merger import L1Merger

__all__ = ["ByteEncoder", "L1Merger"]
__version__ = "0.2.0"
