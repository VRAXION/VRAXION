"""VRAXION Block C byte-pair embedder, Python deploy SDK.

Consumes the packed champion at output/block_c_bytepair_champion/packed.bin
(bake script archived to tag archives/tools-cleanup-20260425).
Layout (little-endian, v1):

  Header (32 bytes):
    magic       b"VCBP"          (4B)
    version     uint8 = 1        (1B)
    scheme      uint8            (0 = cold_shared, 1 = cold_drop)   (1B)
    reserved    (2B)
    vocab_size  uint32           (4B)
    E           uint32           (4B)
    n_hot       uint32           (4B)
    reserved    (12B)

  Scales:       E x fp16         (2E bytes, per-channel int4 scale)
  Shared OOV:   E x fp16         (2E bytes, mean of cold rows for scheme=0)
  Hot bitmap:   (V+7)//8 bytes   (LSB-first, bit i <=> row i is in the hot set)
  Hot rows:     n_hot x (4E//8)  (int4 packed, 2's-complement nibbles)

Total for V=65536, E=32, n_hot=3386: 62,528 B (~61 KB).

Zero ML framework dependency: pure numpy + stdlib.

Usage:
    from block_c_embedder import L2Embedder
    emb = L2Embedder.load_default()
    # Embed the byte-pair 't' followed by 'h' (i.e. 'th')
    v = emb.embed_pair(b'th')                   # (E,) fp32
    # Embed a whole byte stream as a sequence of embeddings
    seq = emb.encode_bytes(b"Hello world!")     # (N, E) fp32, N = len//2
"""
from __future__ import annotations

import hashlib
import struct
from pathlib import Path
from typing import Union

import numpy as np


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_PACKED_PATH = (
    _REPO_ROOT / "output" / "block_c_bytepair_champion" / "packed.bin"
)

_MAGIC = b"VCBP"
_HEADER_SIZE = 32


def _unpack_int4_rows(packed: np.ndarray, E: int) -> np.ndarray:
    """Decode (N, E/2) uint8 -> (N, E) int8 in [-7, 7]."""
    N = packed.shape[0]
    low  = packed & 0x0F
    high = (packed >> 4) & 0x0F
    # sign-extend 4-bit 2's-complement to int8
    low_se  = np.where(low  >= 8, low  - 16, low).astype(np.int8)
    high_se = np.where(high >= 8, high - 16, high).astype(np.int8)
    out = np.empty((N, E), dtype=np.int8)
    out[:, 0::2] = low_se
    out[:, 1::2] = high_se
    return out


def _unpack_bitmap(bm: bytes, V: int) -> np.ndarray:
    """LSB-first bitmap -> bool array of length V."""
    arr = np.frombuffer(bm, dtype=np.uint8)
    bits = np.unpackbits(arr, bitorder="little")
    return bits[:V].astype(bool)


class L2Embedder:
    """Deploy-ready Block C byte-pair embedder.

    Ingest a packed champion binary and serve embeddings per byte-pair ID.
    Hot rows (rows seen frequently in training corpus) are reconstructed
    from int4 + per-channel scales. Cold rows return the shared OOV vector
    (scheme=0) or a zero vector (scheme=1).
    """

    VOCAB = 65536
    _SCHEMES = {0: "cold_shared", 1: "cold_drop"}

    def __init__(self, hot_float: np.ndarray, hot_mask: np.ndarray,
                 oov: np.ndarray, scheme: int, meta: dict):
        self._hot_float = hot_float          # (n_hot, E) fp32 (dequantized)
        self._hot_mask  = hot_mask           # (V,) bool
        self._oov       = oov.astype(np.float32)
        self._scheme    = scheme
        self._meta      = meta
        # Build id -> row_in_hot_float for O(1) lookup
        hot_ids = np.where(hot_mask)[0]
        row_map = np.full(len(hot_mask), -1, dtype=np.int32)
        row_map[hot_ids] = np.arange(len(hot_ids), dtype=np.int32)
        self._row_map = row_map

    # ------------------------------------------------------------------
    # factories

    @classmethod
    def load_default(cls) -> "L2Embedder":
        return cls.from_packed(DEFAULT_PACKED_PATH)

    @classmethod
    def from_packed(cls, path: Union[str, Path]) -> "L2Embedder":
        payload = Path(path).read_bytes()
        if payload[:4] != _MAGIC:
            raise ValueError(f"bad magic {payload[:4]!r}, expected {_MAGIC!r}")
        version = payload[4]
        if version != 1:
            raise ValueError(f"unsupported version {version}")
        scheme = payload[5]
        V = struct.unpack_from("<I", payload, 8)[0]
        E = struct.unpack_from("<I", payload, 12)[0]
        n_hot = struct.unpack_from("<I", payload, 16)[0]
        if V != cls.VOCAB:
            raise ValueError(f"unexpected vocab {V} (expected {cls.VOCAB})")
        if E % 2 != 0:
            raise ValueError(f"E={E} must be even")

        off = _HEADER_SIZE
        scale_bytes = 2 * E
        scales_fp16 = np.frombuffer(payload, dtype=np.float16,
                                    count=E, offset=off).astype(np.float32)
        off += scale_bytes

        oov = np.frombuffer(payload, dtype=np.float16,
                            count=E, offset=off).astype(np.float32)
        off += scale_bytes

        bitmap_bytes = (V + 7) // 8
        bitmap = _unpack_bitmap(payload[off:off + bitmap_bytes], V)
        off += bitmap_bytes

        if int(bitmap.sum()) != n_hot:
            raise ValueError(
                f"bitmap popcount ({int(bitmap.sum())}) != n_hot ({n_hot})"
            )

        hot_packed_bytes = n_hot * (E // 2)
        hot_packed = np.frombuffer(payload, dtype=np.uint8,
                                   count=hot_packed_bytes,
                                   offset=off).reshape(n_hot, E // 2)
        off += hot_packed_bytes

        if off != len(payload):
            raise ValueError(
                f"{len(payload) - off} trailing bytes after decode"
            )

        hot_q = _unpack_int4_rows(hot_packed, E)    # (n_hot, E) int8 [-7, 7]
        hot_float = hot_q.astype(np.float32) * scales_fp16[None, :]

        meta = {
            "version": version, "scheme": cls._SCHEMES.get(scheme, "unknown"),
            "vocab_size": V, "E": E, "n_hot": n_hot,
            "sha256_prefix": hashlib.sha256(payload).hexdigest()[:16],
            "packed_bytes": len(payload),
        }
        return cls(hot_float, bitmap, oov, scheme, meta)

    # ------------------------------------------------------------------
    # core API

    @property
    def E(self) -> int:
        return self._hot_float.shape[1]

    @property
    def meta(self) -> dict:
        return dict(self._meta)

    def embed_id(self, pair_id: int) -> np.ndarray:
        """Return (E,) fp32 embedding for a byte-pair id in [0, 65535]."""
        if pair_id < 0 or pair_id >= self.VOCAB:
            raise IndexError(f"pair_id {pair_id} out of range [0, {self.VOCAB})")
        row = self._row_map[pair_id]
        if row >= 0:
            return self._hot_float[row].copy()
        return self._oov.copy()

    def embed_ids(self, pair_ids: np.ndarray) -> np.ndarray:
        """Vectorised lookup. pair_ids: (N,) int array. Returns (N, E) fp32."""
        ids = np.asarray(pair_ids, dtype=np.int64)
        if np.any((ids < 0) | (ids >= self.VOCAB)):
            raise IndexError("pair_ids out of range")
        rows = self._row_map[ids]
        out = np.empty((len(ids), self.E), dtype=np.float32)
        hot_i = rows >= 0
        out[hot_i] = self._hot_float[rows[hot_i]]
        out[~hot_i] = self._oov
        return out

    def embed_pair(self, pair_bytes: bytes) -> np.ndarray:
        """pair_bytes: 2-byte sequence."""
        if len(pair_bytes) != 2:
            raise ValueError(f"expected 2 bytes, got {len(pair_bytes)}")
        pid = (pair_bytes[0] << 8) | pair_bytes[1]
        return self.embed_id(pid)

    def encode_bytes(self, data: bytes) -> np.ndarray:
        """Encode a byte stream as a sequence of pair embeddings.

        Groups bytes two-at-a-time (drops a trailing odd byte if any).
        Returns (N, E) fp32 where N = len(data) // 2.
        """
        n = len(data) // 2
        if n == 0:
            return np.empty((0, self.E), dtype=np.float32)
        arr = np.frombuffer(data[:n * 2], dtype=np.uint8).reshape(n, 2)
        ids = (arr[:, 0].astype(np.int64) << 8) | arr[:, 1].astype(np.int64)
        return self.embed_ids(ids)

    # ------------------------------------------------------------------
    # diagnostics

    def is_hot(self, pair_id: int) -> bool:
        return bool(self._hot_mask[pair_id])

    def hot_ids(self) -> np.ndarray:
        """All byte-pair IDs that are in the hot set."""
        return np.where(self._hot_mask)[0]

    def __repr__(self) -> str:
        m = self._meta
        return (f"L2Embedder(E={m['E']}, n_hot={m['n_hot']:,}, "
                f"vocab={m['vocab_size']:,}, scheme={m['scheme']!r}, "
                f"packed={m['packed_bytes']:,}B)")
