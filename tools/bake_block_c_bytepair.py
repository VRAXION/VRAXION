"""Bake the Block C byte-pair champion into a packed deploy binary.

Format (little-endian throughout, VCBP1 = VRAXION block C byte-Pair v1):

  Header (32 bytes):
    magic            4B      b"VCBP"
    version          1B      1
    scheme           1B      0 = hot_int4 + cold_shared, 1 = hot_int4 + cold_drop
    reserved         2B
    vocab_size       4B uint32
    E                4B uint32
    n_hot            4B uint32
    reserved         12B

  Scales (hot channel):    E × fp16 = 2E bytes
  Shared OOV vector:       E × fp16 = 2E bytes   (scheme=0 only; scheme=1 zeros)
  Hot bitmap:              (vocab + 7) // 8 bytes
  Hot rows (int4 packed):  n_hot × (4·E // 8) bytes

Layout total: 32 + 2E + 2E + (V+7)//8 + n_hot × (4E//8)
For V=65536, E=32, n_hot=3386:
  = 32 + 64 + 64 + 8192 + 54176 = 62,528 bytes (~61 KB)

Run:
    python3 tools/bake_block_c_bytepair.py \\
        --emb output/bytepair_ft_pull/seed1/seed_1/emb_E32_epoch10.npy \\
        --corpus output/data/fineweb_edu_1gb.txt \\
        --max-bytes 10000000 \\
        --freq-min 5 \\
        --scheme shared \\
        --out output/block_c_bytepair_champion/packed.bin
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np


MAGIC = b"VCBP"
VERSION = 1


def byte_pair_freq(corpus_path: Path, max_bytes: int) -> np.ndarray:
    raw = corpus_path.read_bytes()[:max_bytes]
    n = len(raw) // 2
    arr = np.frombuffer(raw[: n * 2], dtype=np.uint8).reshape(n, 2)
    ids = (arr[:, 0].astype(np.int64) << 8) | arr[:, 1].astype(np.int64)
    return np.bincount(ids, minlength=65536).astype(np.int64)


def quantize_hot(hot_rows: np.ndarray, alpha: float = 0.5
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel symmetric int4. Returns (packed_int4_uint8, scale_fp16)."""
    qmax = 7.0
    amax = alpha * np.max(np.abs(hot_rows), axis=0)
    safe = np.where(amax == 0, np.float32(1.0), amax).astype(np.float32)
    scale = (safe / qmax).astype(np.float16)
    # Quantize
    q = np.round(hot_rows / safe * qmax).clip(-qmax, qmax).astype(np.int8)
    # Pack 2 nibbles per byte: (a[0],a[1]) -> (a[0] & 0x0F) | ((a[1] & 0x0F) << 4)
    # q is (N, E) in [-7, +7]. Store 2's-complement 4-bit:
    # map [-7, 7] -> [-7, 7] then mask with 0x0F (which keeps sign as MSB of nibble)
    # Actually easier: shift [-7,7] -> [0,15] by +7 offset ... but that wastes 1 level.
    # Use 2's-complement 4-bit: values [-8, 7]. Our range is [-7, 7], fits.
    q4 = (q & 0x0F).astype(np.uint8)  # keeps the 4-bit 2's complement correctly
    N, E = q4.shape
    if E % 2 != 0:
        raise ValueError(f"E must be even for int4 packing, got {E}")
    packed = (q4[:, 0::2] | (q4[:, 1::2] << 4)).astype(np.uint8)  # (N, E/2)
    return packed, scale


def build_bitmap(hot_mask: np.ndarray) -> bytes:
    """Pack bool mask into LSB-first bitmap bytes."""
    n = len(hot_mask)
    out = bytearray((n + 7) // 8)
    for i in np.where(hot_mask)[0]:
        out[i // 8] |= 1 << (i % 8)
    return bytes(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--max-bytes", type=int, default=10_000_000)
    ap.add_argument("--freq-min", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--scheme", choices=["shared", "drop"], default="shared")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    emb = np.load(args.emb).astype(np.float32)
    V, E = emb.shape
    print(f"Emb: {emb.shape}  float size = {V*E*4:,} B")

    freq = byte_pair_freq(args.corpus, args.max_bytes)
    hot_mask = freq >= args.freq_min
    n_hot = int(hot_mask.sum())
    print(f"Hot pairs (freq >= {args.freq_min}): {n_hot:,}  cold: {V - n_hot:,}")

    hot_rows = emb[hot_mask]
    packed, scale = quantize_hot(hot_rows, alpha=args.alpha)
    print(f"Hot packed: {packed.shape} (int4 packed to {packed.nbytes:,} B)")
    print(f"Per-channel scale shape: {scale.shape}  ({scale.nbytes:,} B fp16)")

    # OOV vector: mean of cold rows (scheme=shared) or zeros (scheme=drop)
    if args.scheme == "shared":
        oov = emb[~hot_mask].mean(axis=0).astype(np.float16)
        scheme_id = 0
    else:
        oov = np.zeros(E, dtype=np.float16)
        scheme_id = 1

    bitmap = build_bitmap(hot_mask)
    print(f"Bitmap: {len(bitmap):,} B")

    # Build packed file
    header = bytearray(32)
    header[0:4] = MAGIC
    header[4] = VERSION
    header[5] = scheme_id
    # 6-7 reserved (0)
    struct.pack_into("<I", header, 8, V)
    struct.pack_into("<I", header, 12, E)
    struct.pack_into("<I", header, 16, n_hot)
    # 20-31 reserved

    body = bytearray()
    body += scale.tobytes()
    body += oov.tobytes()
    body += bitmap
    body += packed.tobytes()
    total = header + body

    sha = hashlib.sha256(total).hexdigest()[:16]

    args.out.write_bytes(total)
    (args.out.with_suffix(".meta.json")).write_text(json.dumps({
        "magic": "VCBP",
        "version": VERSION,
        "scheme": args.scheme,
        "vocab_size": V,
        "E": E,
        "n_hot": n_hot,
        "alpha": args.alpha,
        "freq_min": args.freq_min,
        "corpus": str(args.corpus),
        "corpus_bytes": args.max_bytes,
        "source_emb": str(args.emb),
        "packed_bytes": len(total),
        "compression_vs_float32": round(V * E * 4 / len(total), 2),
        "sha256_prefix": sha,
    }, indent=2))

    print(f"\nWrote: {args.out}")
    print(f"Total packed: {len(total):,} B  (compression {V*E*4/len(total):.1f}x)")
    print(f"sha256 prefix: {sha}")


if __name__ == "__main__":
    main()
