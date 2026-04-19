"""VRAXION Python deploy SDK.

Minimal, deploy-ready Python API over the frozen byte-level pipeline.

Currently exposes:
  - ByteEncoder  (Block A — byte unit, L0)

See also the parallel Rust/ deploy SDK; both read the same champion
artifacts from the repo's output/ directory.
"""
from .byte_encoder import ByteEncoder

__all__ = ["ByteEncoder"]
__version__ = "0.1.0"
