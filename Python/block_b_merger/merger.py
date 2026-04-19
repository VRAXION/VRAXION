"""VRAXION Block B — L1 Byte-Pair Merger, Python deploy SDK.

Architecture:
  forward(x) = C19(x @ W + b1) @ W.T + b2

  W     : (32, 81)   single mirror-tied weight matrix
  b1    : (81,)      hidden bias
  b2    : (32,)      output bias
  C19   : per-channel piecewise-polynomial non-linearity with per-channel
          c (clamped >= 0.1) and rho (clamped >= 0.0) parameters.

The champion is 100% lossless on all 65,536 byte pairs (sign-match criterion).

Loading: reads the 3,440-byte Huffman-packed binary at
  output/merger_single_w_huffman_pack/packed_model.bin

Zero ML framework dependency — pure numpy + stdlib.
"""
from __future__ import annotations

import math
import struct
from pathlib import Path
from typing import Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Paths (3 levels up from this file: merger.py -> block_b_merger -> Python -> repo root)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_PACKED_PATH = (
    _REPO_ROOT / "output" / "merger_single_w_huffman_pack" / "packed_model.bin"
)
# The merger champion was trained on the "nozero" variant of the byte-unit LUT.
# This LUT lives in tools/ (it is the training-time data source) and is used by
# verify_lossless() to reproduce the 65536/65536 sign-match result.
DEFAULT_LUT_PATH = _REPO_ROOT / "tools" / "byte_embedder_lut_int8_nozero.json"
# Block A champion LUT (int8, different scale) — exposed for callers who need it.
BLOCK_A_LUT_PATH = (
    _REPO_ROOT / "output" / "byte_unit_champion_binary_c19_h16" / "byte_embedder_lut_int8.json"
)

# ---------------------------------------------------------------------------
# Component manifest — matches the packer's COMPONENTS list exactly
# (name, n_elements, raw_fp16_only)

_COMPONENTS: list[tuple[str, int, bool]] = [
    ("W",       2592, False),
    ("b1",        81, False),
    ("b2",        32, True),
    ("c19_c",     81, False),
    ("c19_rho",   81, False),
]

_MAGIC = b"VGH1"

# ---------------------------------------------------------------------------
# Bitstream helpers (ported from diag_byte_single_w_huffman_pack.py)


class _BitReader:
    __slots__ = ("_data", "_pos", "_acc", "_nbits")

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._pos = 0
        self._acc = 0
        self._nbits = 0

    def read(self, length: int) -> int:
        out = 0
        while length > 0:
            if self._nbits == 0:
                if self._pos >= len(self._data):
                    raise EOFError("bitstream exhausted")
                self._acc = self._data[self._pos]
                self._pos += 1
                self._nbits = 8
            take = min(self._nbits, length)
            shift = self._nbits - take
            chunk = (self._acc >> shift) & ((1 << take) - 1)
            out = (out << take) | chunk
            self._nbits -= take
            self._acc &= (1 << self._nbits) - 1 if self._nbits else 0
            length -= take
        return out


def _unpack_bits(data: bytes, n: int) -> list[int]:
    r = _BitReader(data)
    return [r.read(1) for _ in range(n)]


def _unpack_nibbles(data: bytes, n: int) -> list[int]:
    out: list[int] = []
    for b in data:
        out.append(b & 0xF)
        if len(out) == n:
            break
        out.append((b >> 4) & 0xF)
        if len(out) == n:
            break
    return out


def _canonical_decode_table(lengths: dict[int, int]) -> dict[tuple[int, int], int]:
    """Build (length, code) -> symbol table for canonical Huffman decoding."""
    items = sorted(((ln, sym) for sym, ln in lengths.items()), key=lambda x: (x[0], x[1]))
    code = 0
    prev_len = items[0][0]
    table: dict[tuple[int, int], int] = {}
    for ln, sym in items:
        code <<= ln - prev_len
        table[(ln, code)] = sym
        code += 1
        prev_len = ln
    return table


def _decode_symbols(data: bytes, n: int, lengths: dict[int, int]) -> list[int]:
    if n == 0:
        return []
    table = _canonical_decode_table(lengths)
    max_len = max(lengths.values())
    r = _BitReader(data)
    out: list[int] = []
    for _ in range(n):
        code = 0
        for ln in range(1, max_len + 1):
            code = (code << 1) | r.read(1)
            key = (ln, code)
            if key in table:
                out.append(table[key])
                break
        else:
            raise ValueError("invalid Huffman stream")
    return out


# ---------------------------------------------------------------------------
# Component unpacker


def _unpack_component(name: str, n: int, raw_only: bool,
                      payload: bytes, offset: int) -> tuple[np.ndarray, int]:
    """Decode one model component from the packed binary starting at offset.

    Returns (float32 array, new_offset).
    """
    if raw_only:
        end = offset + n * 2
        arr = (
            np.frombuffer(payload[offset:end], dtype=np.uint16)
            .view(np.float16)
            .astype(np.float32)
        )
        return arr, end

    # --- read generator table ---
    g = payload[offset]
    offset += 1
    gens_end = offset + g * 2
    gens = (
        np.frombuffer(payload[offset:gens_end], dtype=np.uint16)
        .view(np.float16)
        .astype(np.float32)
    )
    offset = gens_end

    # --- mode bitmap: 1 = encoded cell, 0 = fallback fp16 ---
    mode_nbytes = math.ceil(n / 8)
    mode_bits = _unpack_bits(payload[offset:offset + mode_nbytes], n)
    offset += mode_nbytes
    n_enc = sum(mode_bits)
    n_fb = n - n_enc

    # --- sign bitmap for encoded cells ---
    sign_nbytes = math.ceil(n_enc / 8)
    sign_bits = _unpack_bits(payload[offset:offset + sign_nbytes], n_enc)
    offset += sign_nbytes

    # --- coef Huffman lengths (7 nibbles, symbols 1..7) ---
    coef_len_nbytes = math.ceil(7 / 2)
    coef_lens_raw = _unpack_nibbles(payload[offset:offset + coef_len_nbytes], 7)
    offset += coef_len_nbytes
    coef_lengths = {sym: ln for sym, ln in zip(range(1, 8), coef_lens_raw) if ln > 0}

    # --- generator-index Huffman lengths (g nibbles) ---
    idx_len_nbytes = math.ceil(g / 2)
    idx_lens_raw = _unpack_nibbles(payload[offset:offset + idx_len_nbytes], g)
    offset += idx_len_nbytes
    idx_lengths = {sym: ln for sym, ln in zip(range(g), idx_lens_raw) if ln > 0}

    # --- blob sizes + blobs ---
    coef_nbytes, idx_nbytes = struct.unpack("<HH", payload[offset:offset + 4])
    offset += 4
    coef_blob = payload[offset:offset + coef_nbytes]
    offset += coef_nbytes
    idx_blob = payload[offset:offset + idx_nbytes]
    offset += idx_nbytes

    coef_syms = _decode_symbols(coef_blob, n_enc, coef_lengths)
    idx_syms  = _decode_symbols(idx_blob,  n_enc, idx_lengths)

    # --- fallback fp16 stream ---
    fb_end = offset + n_fb * 2
    fallback = (
        np.frombuffer(payload[offset:fb_end], dtype=np.uint16)
        .view(np.float16)
        .astype(np.float32)
    )
    offset = fb_end

    # --- reconstruct array ---
    arr = np.empty(n, dtype=np.float32)
    ie = ifb = 0
    for i, m in enumerate(mode_bits):
        if m:
            s  = 1 if sign_bits[ie] else -1
            c  = coef_syms[ie]
            gi = idx_syms[ie]
            arr[i] = np.float32(s * c * gens[gi])
            ie += 1
        else:
            arr[i] = fallback[ifb]
            ifb += 1

    return arr, offset


# ---------------------------------------------------------------------------
# C19 activation (pure numpy, mirrors the torch reference exactly)


def _c19(x: np.ndarray, c: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Piecewise-polynomial parametric non-linearity.

    Per-channel parameters:
      c   : shape (H,), clamped >= 0.1
      rho : shape (H,), clamped >= 0.0

    Formula (inner region, per element):
      scaled = x / c
      n = floor(scaled)
      t = scaled - n
      h = t * (1 - t)
      sgn = +1 if n is even, -1 if n is odd
      interior = c * (sgn * h + rho * h * h)

    Saturation tails at |x| >= 6c: y = x ± 6c (linear extension).

    All operations are kept in float32 to match the torch reference.
    """
    f32 = np.float32
    c_   = np.maximum(c,   f32(0.1)).astype(np.float32)
    rho_ = np.maximum(rho, f32(0.0)).astype(np.float32)
    L      = (f32(6.0) * c_).astype(np.float32)
    scaled = (x / c_).astype(np.float32)
    n      = np.floor(scaled).astype(np.float32)
    t      = (scaled - n).astype(np.float32)
    h      = (t * (f32(1.0) - t)).astype(np.float32)
    sgn    = np.where(
        (n.astype(np.int64) % 2) == 0,
        np.float32(1.0),
        np.float32(-1.0),
    ).astype(np.float32)
    interior = (c_ * (sgn * h + rho_ * h * h)).astype(np.float32)
    return np.where(
        x >= L,
        (x - L).astype(np.float32),
        np.where(x <= -L, (x + L).astype(np.float32), interior),
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# Main class


class L1Merger:
    """Deploy-ready L1 byte-pair merger for Block B.

    Architecture: y = C19(x @ W + b1) @ W.T + b2
      - W    : (32, 81) float32 — single mirror-tied weight matrix
      - b1   : (81,)   float32
      - b2   : (32,)   float32
      - C19  : per-channel parametric non-linearity (81 c + 81 rho params)

    Weights are loaded from the 3,440-byte Huffman-packed binary artifact.
    100% lossless on all 65,536 byte pairs (sign-match criterion).
    """

    IN_DIM = 32
    H      = 81

    def __init__(
        self,
        W:       np.ndarray,
        b1:      np.ndarray,
        b2:      np.ndarray,
        c19_c:   np.ndarray,
        c19_rho: np.ndarray,
    ) -> None:
        assert W.shape       == (self.IN_DIM, self.H), f"W shape: {W.shape}"
        assert b1.shape      == (self.H,),             f"b1 shape: {b1.shape}"
        assert b2.shape      == (self.IN_DIM,),        f"b2 shape: {b2.shape}"
        assert c19_c.shape   == (self.H,),             f"c19_c shape: {c19_c.shape}"
        assert c19_rho.shape == (self.H,),             f"c19_rho shape: {c19_rho.shape}"
        self._W       = W.astype(np.float32)
        self._b1      = b1.astype(np.float32)
        self._b2      = b2.astype(np.float32)
        self._c19_c   = c19_c.astype(np.float32)
        self._c19_rho = c19_rho.astype(np.float32)

    # ------------------------------------------------------------------
    # factories

    @classmethod
    def load_default(cls) -> "L1Merger":
        """Load the champion from output/merger_single_w_huffman_pack/packed_model.bin."""
        return cls.from_packed(DEFAULT_PACKED_PATH)

    @classmethod
    def from_packed(cls, path: Path) -> "L1Merger":
        """Load from a VGH1-format Huffman-packed binary."""
        payload = Path(path).read_bytes()
        if payload[:4] != _MAGIC:
            raise ValueError(f"Bad magic bytes: {payload[:4]!r} (expected {_MAGIC!r})")
        if len(payload) != 3440:
            raise ValueError(f"Unexpected packed size: {len(payload)} (expected 3440)")

        offset = 4
        state: dict[str, np.ndarray] = {}
        for name, n, raw_only in _COMPONENTS:
            arr, offset = _unpack_component(name, n, raw_only, payload, offset)
            state[name] = arr

        if offset != len(payload):
            raise ValueError(
                f"{len(payload) - offset} trailing bytes after decode"
            )

        return cls(
            W       = state["W"].reshape(cls.IN_DIM, cls.H),
            b1      = state["b1"],
            b2      = state["b2"],
            c19_c   = state["c19_c"],
            c19_rho = state["c19_rho"],
        )

    # ------------------------------------------------------------------
    # forward pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Merge a 32-dim byte-pair latent.

        Parameters
        ----------
        x : (32,) float array — two concatenated 16-dim byte-unit latents.

        Returns
        -------
        y : (32,) float32 — reconstructed (lossless in sign-match sense).
        """
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if x.shape != (self.IN_DIM,):
            raise ValueError(f"Expected (32,) input, got {x.shape}")
        h = _c19(x @ self._W + self._b1, self._c19_c, self._c19_rho)
        y = h @ self._W.T + self._b2
        return y

    def forward_batch(self, X: np.ndarray) -> np.ndarray:
        """Vectorized forward pass.

        Parameters
        ----------
        X : (N, 32) float array.

        Returns
        -------
        Y : (N, 32) float32.
        """
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X[np.newaxis, :]
        if X.shape[1] != self.IN_DIM:
            raise ValueError(f"Expected N×32 input, got {X.shape}")
        H = _c19(X @ self._W + self._b1, self._c19_c, self._c19_rho)  # (N, 81)
        Y = H @ self._W.T + self._b2                                   # (N, 32)
        return Y

    # ------------------------------------------------------------------
    # verification

    def verify_lossless(
        self, lut_path: Path = DEFAULT_LUT_PATH
    ) -> Tuple[int, int]:
        """Sign-match test over all 65,536 byte pairs.

        Loads the training-time byte-pair LUT (byte_embedder_lut_int8_nozero.json
        from tools/), builds all 256×256 pair inputs, forwards each through the
        merger, and checks sign(y) == sign(x) on all 32 dims.

        This is the exact data source the champion was trained on; using it
        reproduces the 65536/65536 sign-match result from training.

        Parameters
        ----------
        lut_path : path to a byte LUT JSON.  Default is the nozero training LUT.

        Returns
        -------
        (matches, total)  — expected (65536, 65536) for the champion.
        """
        import json as _json

        blob = _json.loads(Path(lut_path).read_text(encoding="utf-8"))
        scale = float(blob["scale"])
        # The nozero LUT stores float32 values directly; the champion LUT stores int8.
        raw_lut = blob["lut"]
        first_row = raw_lut[0]
        if isinstance(first_row[0], int):
            # int8 format (champion LUT)
            lut_i8 = np.array(raw_lut, dtype=np.int8)
            lut    = lut_i8.astype(np.float32) * scale
        else:
            # float32 format (nozero training LUT)
            lut = np.array(raw_lut, dtype=np.float32) * scale  # (256, 16)

        # Build all 65536 pairs: [lut[a] || lut[b]] for a,b in 0..255
        idx_a = np.repeat(np.arange(256), 256)   # 0,0,...,0, 1,1,...,1, ...
        idx_b = np.tile(np.arange(256), 256)     # 0,1,...,255, 0,1,...,255, ...
        pairs = np.concatenate([lut[idx_a], lut[idx_b]], axis=1)  # (65536, 32)

        Y = self.forward_batch(pairs)   # (65536, 32)

        sign_match = np.sign(Y) == np.sign(pairs)   # (65536, 32)
        per_pair   = sign_match.all(axis=1)          # (65536,)
        matches    = int(per_pair.sum())
        total      = 65536
        return matches, total

    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"L1Merger(champion=single-W mirror, "
            f"in={self.IN_DIM}, H={self.H}, "
            f"c19_c_mean={self._c19_c.mean():.4f}, "
            f"c19_rho_mean={self._c19_rho.mean():.4f})"
        )


# ---------------------------------------------------------------------------
# CLI smoke-test

if __name__ == "__main__":
    m = L1Merger.load_default()
    print(m)
    matches, total = m.verify_lossless()
    print(f"Lossless verify: {matches}/{total} pairs")
