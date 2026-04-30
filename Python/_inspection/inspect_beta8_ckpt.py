"""
inspect_beta8_ckpt.py
=====================
Standalone inspection script for the VRAXION beta.8 generalist checkpoint.

Format: bincode 1.3 (little-endian, u64 length prefixes, no magic number).
See instnct-core/src/checkpoint.rs + network/disk.rs for the canonical layout.

Usage:
    python Python/_inspection/inspect_beta8_ckpt.py

Pure stdlib only: struct, hashlib, pathlib — no numpy, torch, or Rust bindings.
"""

import hashlib
import struct
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
CKPT_PATH = REPO_ROOT / "output/releases/v5.0.0-beta.8/seed2042_improved_generalist_v1.ckpt"
SHA256_PATH = CKPT_PATH.with_suffix(".sha256")


# ---------------------------------------------------------------------------
# Bincode 1.x helpers
# ---------------------------------------------------------------------------
# Bincode 1.3 default config:
#   - integers: fixed-size, little-endian
#   - usize / isize: encoded as u64 / i64
#   - Vec<T>: u64 LE element-count, then elements
#   - String: u64 LE byte-count, then UTF-8 bytes
#   - f64: IEEE 754 little-endian (same as native on x86)

class _Reader:
    """Cursor over a bytes view with checked reads."""

    def __init__(self, data: bytes):
        self._d = data
        self._pos = 0

    def pos(self) -> int:
        return self._pos

    def remaining(self) -> int:
        return len(self._d) - self._pos

    def read_raw(self, n: int) -> bytes:
        chunk = self._d[self._pos:self._pos + n]
        if len(chunk) != n:
            raise EOFError(f"wanted {n} bytes at offset {self._pos}, only {len(chunk)} remain")
        self._pos += n
        return chunk

    def read_u8(self) -> int:
        return struct.unpack_from("<B", self.read_raw(1))[0]

    def read_u32(self) -> int:
        return struct.unpack_from("<I", self.read_raw(4))[0]

    def read_i32(self) -> int:
        return struct.unpack_from("<i", self.read_raw(4))[0]

    def read_u64(self) -> int:
        return struct.unpack_from("<Q", self.read_raw(8))[0]

    def read_f64(self) -> float:
        return struct.unpack_from("<d", self.read_raw(8))[0]

    def read_vec_u8(self) -> bytes:
        """u64 count, then count bytes."""
        n = self.read_u64()
        return self.read_raw(n)

    def read_vec_u32(self) -> list:
        n = self.read_u64()
        raw = self.read_raw(n * 4)
        return list(struct.unpack_from(f"<{n}I", raw))

    def read_vec_i32(self) -> list:
        n = self.read_u64()
        raw = self.read_raw(n * 4)
        return list(struct.unpack_from(f"<{n}i", raw))

    def read_vec_u64(self) -> list:
        """usize[] stored as u64[]."""
        n = self.read_u64()
        raw = self.read_raw(n * 8)
        return list(struct.unpack_from(f"<{n}Q", raw))

    def read_vec_i8(self) -> list:
        n = self.read_u64()
        raw = self.read_raw(n)
        return list(struct.unpack_from(f"<{n}b", raw))

    def read_string(self) -> str:
        """u64 byte-count, then UTF-8."""
        raw = self.read_vec_u8()
        return raw.decode("utf-8")


# ---------------------------------------------------------------------------
# Checkpoint layout parsers
# ---------------------------------------------------------------------------

def parse_network_disk_v1(data: bytes) -> dict:
    """
    NetworkDiskV1 (instnct-core/src/network/disk.rs):
        u8   version
        u64  neuron_count          (ConnectionGraphDiskV1)
        u64  src_len + src:[u64]
        u64  tgt_len + tgt:[u64]
        u64  thr_len + thr:[u32]
        u64  ch_len  + ch:[u8]
        u64  pol_len + pol:[i32]
    """
    r = _Reader(data)
    version = r.read_u8()
    # ConnectionGraphDiskV1
    neuron_count = r.read_u64()
    sources = r.read_vec_u64()
    targets = r.read_vec_u64()
    # per-neuron params
    thresholds = r.read_vec_u32()
    channels = r.read_vec_u8()          # Vec<u8> stored as u64-prefixed blob
    polarities = r.read_vec_i32()

    unconsumed = r.remaining()
    return {
        "version": version,
        "neuron_count": neuron_count,
        "edge_count": len(sources),
        "sources_sample": sources[:5],
        "targets_sample": targets[:5],
        "threshold_len": len(thresholds),
        "threshold_range": (min(thresholds), max(thresholds)) if thresholds else (None, None),
        "channel_len": len(channels),
        "channel_set": sorted(set(channels)) if channels else [],
        "polarity_len": len(polarities),
        "polarity_set": sorted(set(polarities)) if polarities else [],
        "unconsumed_bytes": unconsumed,
        # Validation flags (matching disk.rs validate())
        "edges_equal_len": len(sources) == len(targets),
        "threshold_count_ok": len(thresholds) == neuron_count,
        "channel_count_ok": len(channels) == neuron_count,
        "polarity_count_ok": len(polarities) == neuron_count,
    }


def parse_projection_disk(data: bytes) -> dict:
    """
    ProjectionDisk (instnct-core/src/projection.rs):
        u64  w_len + w:[i8]
        u64  input_dim
        u64  output_classes
    """
    r = _Reader(data)
    weights = r.read_vec_i8()
    input_dim = r.read_u64()
    output_classes = r.read_u64()

    expected_w = input_dim * output_classes
    unconsumed = r.remaining()
    return {
        "weight_count": len(weights),
        "input_dim": input_dim,
        "output_classes": output_classes,
        "expected_weight_count": expected_w,
        "weight_count_ok": len(weights) == expected_w,
        "weight_min": min(weights) if weights else None,
        "weight_max": max(weights) if weights else None,
        "nonzero_weights": sum(1 for w in weights if w != 0),
        "unconsumed_bytes": unconsumed,
    }


def parse_checkpoint_meta(r: _Reader) -> dict:
    """
    CheckpointMeta (instnct-core/src/checkpoint.rs):
        u64  step
        f64  accuracy
        String  label  (u64 len + UTF-8)
    """
    step = r.read_u64()
    accuracy = r.read_f64()
    label = r.read_string()
    return {"step": step, "accuracy": accuracy, "label": label}


def parse_checkpoint_disk(data: bytes) -> dict:
    """
    CheckpointDisk (instnct-core/src/checkpoint.rs):
        u8         version
        Vec<u8>    network_bytes   (u64 len + raw bincode of NetworkDiskV1)
        Vec<u8>    projection_bytes (u64 len + raw bincode of ProjectionDisk)
        CheckpointMeta: u64 step + f64 accuracy + String label
    """
    r = _Reader(data)
    version = r.read_u8()

    network_start = r.pos()
    network_bytes = r.read_vec_u8()
    network_end = r.pos()

    projection_start = r.pos()
    projection_bytes = r.read_vec_u8()
    projection_end = r.pos()

    meta = parse_checkpoint_meta(r)
    unconsumed = r.remaining()

    return {
        "outer_version": version,
        "network_blob_offset": network_start,
        "network_blob_size": len(network_bytes),
        "network_blob_end": network_end,
        "projection_blob_offset": projection_start,
        "projection_blob_size": len(projection_bytes),
        "projection_blob_end": projection_end,
        "meta": meta,
        "unconsumed_outer_bytes": unconsumed,
        "_network_bytes": network_bytes,
        "_projection_bytes": projection_bytes,
    }


# ---------------------------------------------------------------------------
# SHA256 verification
# ---------------------------------------------------------------------------

def verify_sha256(ckpt_path: Path, sha256_path: Path) -> tuple[bool, str, str]:
    """
    Returns (match: bool, computed_hex: str, expected_hex: str).
    The sidecar stores uppercase hex; we compare case-insensitively.
    """
    raw = sha256_path.read_text(encoding="utf-8").strip()
    # Accept both "HASH" and "HASH  filename" (shasum/certutil formats)
    expected = raw.split()[0].lower()

    digest = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
    return (digest == expected, digest, expected)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("VRAXION beta.8 checkpoint inspection")
    print("=" * 70)

    # ---- File basics ----
    if not CKPT_PATH.exists():
        print(f"ERROR: checkpoint not found: {CKPT_PATH}")
        sys.exit(1)

    raw = CKPT_PATH.read_bytes()
    file_size = len(raw)
    print(f"\nFile      : {CKPT_PATH}")
    print(f"File size : {file_size:,} bytes ({file_size / 1024:.1f} KB)")

    # ---- Hex preview ----
    preview_len = min(256, file_size)
    print(f"\nFirst {preview_len} bytes (hex):")
    for i in range(0, preview_len, 16):
        chunk = raw[i:i + 16]
        hex_part = " ".join(f"{b:02x}" for b in chunk)
        asc_part = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
        print(f"  {i:04x}: {hex_part:<47}  {asc_part}")

    # ---- SHA256 ----
    print(f"\nSHA256 sidecar: {SHA256_PATH}")
    if SHA256_PATH.exists():
        ok, computed, expected = verify_sha256(CKPT_PATH, SHA256_PATH)
        status = "MATCH" if ok else "MISMATCH"
        print(f"  Expected  : {expected}")
        print(f"  Computed  : {computed}")
        print(f"  Result    : {status}")
    else:
        print("  WARNING: sidecar file not found, SHA256 not verified")

    # ---- Format identification ----
    print("\nFormat identification:")
    print("  No magic number -- format starts directly with bincode payload")
    first_byte = raw[0] if raw else None
    print(f"  First byte (outer CheckpointDisk.version) : {first_byte}")
    print("  Serializer : bincode 1.3 (little-endian, u64 length prefixes)")

    # ---- Parse outer checkpoint ----
    print("\nParsing outer CheckpointDisk ...")
    try:
        ckpt = parse_checkpoint_disk(raw)
    except Exception as exc:
        print(f"  ERROR during outer parse: {exc}")
        sys.exit(1)

    print(f"  outer version              : {ckpt['outer_version']}")
    print(f"  network_bytes blob offset  : {ckpt['network_blob_offset']} (size: {ckpt['network_blob_size']:,} B)")
    print(f"  projection_bytes blob off  : {ckpt['projection_blob_offset']} (size: {ckpt['projection_blob_size']:,} B)")
    print(f"  unconsumed outer bytes     : {ckpt['unconsumed_outer_bytes']}")

    m = ckpt["meta"]
    print(f"\nCheckpointMeta:")
    print(f"  step     : {m['step']:,}")
    print(f"  accuracy : {m['accuracy']:.6f}  ({m['accuracy']*100:.3f}%)")
    print(f"  label    : {m['label']!r}")

    # ---- Parse inner NetworkDiskV1 ----
    print("\nParsing inner NetworkDiskV1 (network_bytes) ...")
    try:
        net = parse_network_disk_v1(ckpt["_network_bytes"])
    except Exception as exc:
        print(f"  ERROR during network parse: {exc}")
        sys.exit(1)

    print(f"  inner version    : {net['version']}")
    print(f"  neuron_count     : {net['neuron_count']:,}")
    print(f"  edge_count       : {net['edge_count']:,}")
    print(f"  sources (first 5): {net['sources_sample']}")
    print(f"  targets (first 5): {net['targets_sample']}")
    print(f"  threshold range  : {net['threshold_range']}  (valid 0..=15)")
    print(f"  channel set      : {net['channel_set']}  (valid 1..=8)")
    print(f"  polarity set     : {net['polarity_set']}  (valid +/-1)")
    print(f"  threshold_len ok : {net['threshold_count_ok']}")
    print(f"  channel_len ok   : {net['channel_count_ok']}")
    print(f"  polarity_len ok  : {net['polarity_count_ok']}")
    print(f"  edges equal len  : {net['edges_equal_len']}")
    print(f"  unconsumed bytes : {net['unconsumed_bytes']}")

    # ---- Parse inner ProjectionDisk ----
    print("\nParsing inner ProjectionDisk (projection_bytes) ...")
    try:
        proj = parse_projection_disk(ckpt["_projection_bytes"])
    except Exception as exc:
        print(f"  ERROR during projection parse: {exc}")
        sys.exit(1)

    print(f"  input_dim         : {proj['input_dim']:,}")
    print(f"  output_classes    : {proj['output_classes']:,}")
    print(f"  weight_count      : {proj['weight_count']:,}")
    print(f"  expected_weights  : {proj['expected_weight_count']:,}")
    print(f"  weight_count_ok   : {proj['weight_count_ok']}")
    print(f"  weight range      : [{proj['weight_min']}, {proj['weight_max']}]  (valid -127..=127)")
    print(f"  nonzero weights   : {proj['nonzero_weights']:,} / {proj['weight_count']:,}"
          f"  ({100*proj['nonzero_weights']/max(proj['weight_count'],1):.1f}%)")
    print(f"  unconsumed bytes  : {proj['unconsumed_bytes']}")

    # ---- Section map ----
    print("\nSection map (byte offsets in .ckpt):")
    print(f"  [0x{0:06x}]         outer.version (1 byte)")
    off = 1
    print(f"  [0x{off:06x} - 0x{ckpt['network_blob_end']-1:06x}]  network_bytes blob"
          f" (8-byte len prefix + {ckpt['network_blob_size']:,} B payload)")
    off = ckpt['projection_blob_offset']
    print(f"  [0x{off:06x} - 0x{ckpt['projection_blob_end']-1:06x}]  projection_bytes blob"
          f" (8-byte len prefix + {ckpt['projection_blob_size']:,} B payload)")
    off = ckpt['projection_blob_end']
    print(f"  [0x{off:06x} - EOF]   CheckpointMeta (step + accuracy + label)")

    # ---- Overall validity ----
    all_ok = all([
        ckpt['outer_version'] == 1,
        net['version'] == 1,
        net['edges_equal_len'],
        net['threshold_count_ok'],
        net['channel_count_ok'],
        net['polarity_count_ok'],
        proj['weight_count_ok'],
        ckpt['unconsumed_outer_bytes'] == 0,
        net['unconsumed_bytes'] == 0,
        proj['unconsumed_bytes'] == 0,
    ])
    sha_ok = SHA256_PATH.exists() and verify_sha256(CKPT_PATH, SHA256_PATH)[0]
    print(f"\nParse integrity : {'PASS' if all_ok else 'FAIL'}")
    print(f"SHA256 integrity: {'PASS' if sha_ok else 'FAIL'}")
    print("\nDone.")


if __name__ == "__main__":
    main()
