"""VRAXION byte encoder (Block A) — Python deploy SDK.

Reads the frozen binary+C19+H=16 champion LUT and provides a minimal API:

  from Python.byte_encoder import ByteEncoder
  enc = ByteEncoder.load_default()
  vec = enc.encode(0x41)          # 16-dim float vector
  bytes_back = enc.decode(vec)    # recover byte

The encoder uses the baked 256-entry int8 LUT. The decoder uses the saved
binary weight matrices (W1, W2) from the winner JSON.

100% lossless on all 256 bytes (verified at champion-freeze time and
reproduced here in tests/test_byte_encoder.py).

Zero ML framework dependency — pure numpy.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Union

import numpy as np

DEFAULT_LUT_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "output" / "byte_unit_champion_binary_c19_h16" / "byte_embedder_lut_int8.json"
)
DEFAULT_WEIGHTS_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "output" / "byte_unit_champion_binary_c19_h16" / "byte_unit_winner_binary.json"
)


class ByteEncoder:
    """Deploy-ready byte encoder for Block A (binary + C19 + H=16, 100% lossless).

    Two code paths:
      - encode(byte) -> 16-dim vector : O(1) LUT lookup
      - decode(vec)  -> byte          : 1 matmul through tied-mirror weights

    Both are pure numpy. No framework, no autograd, no training here.
    """

    LUT_DIM = 16
    INPUT_BITS = 8
    HIDDEN = 16

    def __init__(self, lut_int8: np.ndarray, lut_scale: float,
                 W1: np.ndarray, W2: np.ndarray):
        assert lut_int8.shape == (256, self.LUT_DIM), f"LUT shape mismatch: {lut_int8.shape}"
        assert lut_int8.dtype == np.int8
        assert W1.shape == (self.INPUT_BITS, self.HIDDEN)
        assert W2.shape == (self.HIDDEN, self.LUT_DIM)
        self._lut_int8 = lut_int8
        self._lut_scale = float(lut_scale)
        self._lut_f32 = lut_int8.astype(np.float32) * self._lut_scale
        self._W1 = W1.astype(np.float32)
        self._W2 = W2.astype(np.float32)

    # ------------------------------------------------------------------
    # factories

    @classmethod
    def load_default(cls) -> "ByteEncoder":
        return cls.from_paths(DEFAULT_LUT_PATH, DEFAULT_WEIGHTS_PATH)

    @classmethod
    def from_paths(cls, lut_path: Path, weights_path: Path) -> "ByteEncoder":
        lut_blob = json.loads(Path(lut_path).read_text(encoding="utf-8"))
        assert lut_blob.get("format") == "int8_lut", f"unexpected LUT format: {lut_blob.get('format')}"
        lut_int8 = np.array(lut_blob["lut"], dtype=np.int8)
        lut_scale = float(lut_blob["scale"])

        w_blob = json.loads(Path(weights_path).read_text(encoding="utf-8"))
        assert w_blob.get("precision") == "binary_scaled", f"unexpected weights precision: {w_blob.get('precision')}"

        # Reconstruct float weights from binary indices + levels (levels already contain alpha scaling).
        W1_idx = np.array(w_blob["W1_binary_idx"], dtype=np.int64)
        W1_levels = np.array(w_blob["W1_levels"], dtype=np.float32)
        W1 = W1_levels[W1_idx]

        W2_idx = np.array(w_blob["W2_binary_idx"], dtype=np.int64)
        W2_levels = np.array(w_blob["W2_levels"], dtype=np.float32)
        W2 = W2_levels[W2_idx]

        return cls(lut_int8=lut_int8, lut_scale=lut_scale, W1=W1, W2=W2)

    # ------------------------------------------------------------------
    # encoding

    def encode(self, byte: Union[int, bytes]) -> np.ndarray:
        """Byte (int 0..255 or 1-byte bytes) -> (16,) float32 latent vector.

        O(1) LUT lookup.
        """
        if isinstance(byte, (bytes, bytearray)):
            assert len(byte) == 1, f"expected 1-byte input, got {len(byte)}"
            b = byte[0]
        else:
            b = int(byte)
        assert 0 <= b < 256, f"byte out of range: {b}"
        return self._lut_f32[b].copy()

    def encode_bytes(self, data: bytes) -> np.ndarray:
        """Vectorized: bytes -> (N, 16) float32 latent matrix."""
        arr = np.frombuffer(data, dtype=np.uint8)
        return self._lut_f32[arr].copy()

    # ------------------------------------------------------------------
    # decoding (tied-mirror)

    def decode(self, latent: np.ndarray) -> int:
        """(16,) latent vector -> byte (int 0..255).

        Runs the tied-mirror decoder: latent @ W2.T @ W1.T -> 8-dim, sign -> bits.
        """
        latent = np.asarray(latent, dtype=np.float32).reshape(-1)
        assert latent.shape == (self.LUT_DIM,), f"latent shape mismatch: {latent.shape}"
        # Decoder: project 16 -> 8 via tied mirror (W2.T then W1.T)
        h = latent @ self._W2.T       # (16,) -> (16,)
        x_hat = h @ self._W1.T        # (16,) -> (8,)
        # Signs of the 8 outputs are the byte bits (bipolar representation)
        bits = (x_hat > 0).astype(np.uint8)
        # Pack bits lowest-first (bit 0 is LSB)
        byte = 0
        for i in range(8):
            byte |= int(bits[i]) << i
        return byte

    def decode_bytes(self, latents: np.ndarray) -> bytes:
        """Vectorized: (N, 16) -> N-byte sequence."""
        latents = np.asarray(latents, dtype=np.float32)
        assert latents.ndim == 2 and latents.shape[1] == self.LUT_DIM
        h = latents @ self._W2.T      # (N, 16)
        x_hat = h @ self._W1.T        # (N, 8)
        bits = (x_hat > 0).astype(np.uint8)
        bytes_out = np.zeros(latents.shape[0], dtype=np.uint8)
        for i in range(8):
            bytes_out |= bits[:, i] << i
        return bytes_out.tobytes()

    # ------------------------------------------------------------------
    # diagnostics

    def verify_lossless(self) -> tuple[int, int]:
        """Encode all 256 bytes, decode them back, return (matches, total)."""
        latents = self._lut_f32  # (256, 16)
        out = self.decode_bytes(latents)
        orig = bytes(range(256))
        matches = sum(1 for a, b in zip(orig, out) if a == b)
        return matches, 256

    def __repr__(self) -> str:
        return (f"ByteEncoder(champion=binary+C19+H=16, "
                f"input=8bits, latent={self.LUT_DIM}dim, "
                f"lut_scale={self._lut_scale:.6e})")


if __name__ == "__main__":
    enc = ByteEncoder.load_default()
    print(enc)
    matches, total = enc.verify_lossless()
    print(f"Self-verify round-trip: {matches}/{total} bytes lossless")

    demo_byte = ord("A")  # 65
    vec = enc.encode(demo_byte)
    print(f"\nencode({demo_byte}, 'A') -> shape={vec.shape}, first 4 = {vec[:4]}")
    back = enc.decode(vec)
    print(f"decode(vec) -> {back} ('{chr(back)}')")
