#!/usr/bin/env python3
"""
AB window codec v1.

Stable, dependency-free deployment surface for the D21-D23 A+B result:

    8 bytes -> A-window 128D -> B-window 64D -> A-window 128D -> 8 bytes

The B decoder is the transpose of the B encoder. It reconstructs an
A-decodable 128D surface, not the full redundant D22 128D reference code:
the visible 8 bit lanes per byte are restored and the redundant lanes are zero.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


VERSION = "ab_window_codec_v1"
WINDOW_BYTES = 8
BYTE_BITS = 8
A_DIMS_PER_BYTE = 16
B_DIMS_PER_BYTE = 8
A_WINDOW_DIMS = WINDOW_BYTES * A_DIMS_PER_BYTE
B_WINDOW_DIMS = WINDOW_BYTES * B_DIMS_PER_BYTE
DEFAULT_SEED = 20260502


@dataclass(frozen=True)
class ABWindowCodec:
    """Reciprocal A+B codec for fixed 8-byte windows."""

    window_bytes: int = WINDOW_BYTES

    @property
    def a_dims(self) -> int:
        return self.window_bytes * A_DIMS_PER_BYTE

    @property
    def b_dims(self) -> int:
        return self.window_bytes * B_DIMS_PER_BYTE

    def encode_byte_a16(self, byte_value: int) -> list[int]:
        byte = _checked_byte(byte_value)
        bits = [1 if ((byte >> bit) & 1) else -1 for bit in range(BYTE_BITS)]
        return bits + bits

    def decode_a16_to_byte(self, code16: Sequence[float | int]) -> int:
        if len(code16) != A_DIMS_PER_BYTE:
            raise ValueError(f"A byte code must have {A_DIMS_PER_BYTE} dims, got {len(code16)}")
        return _bits_to_byte(1 if float(value) >= 0.0 else -1 for value in code16[:BYTE_BITS])

    def encode_window_a128(self, window: bytes | Sequence[int]) -> list[int]:
        values = _checked_window(window, self.window_bytes)
        code: list[int] = []
        for byte in values:
            code.extend(self.encode_byte_a16(byte))
        return code

    def decode_a128_to_window(self, code128: Sequence[float | int]) -> bytes:
        if len(code128) != self.a_dims:
            raise ValueError(f"A window code must have {self.a_dims} dims, got {len(code128)}")
        values = []
        for offset in range(0, self.a_dims, A_DIMS_PER_BYTE):
            values.append(self.decode_a16_to_byte(code128[offset : offset + A_DIMS_PER_BYTE]))
        return bytes(values)

    def encode_a128_to_b64(self, code128: Sequence[float | int]) -> list[int]:
        if len(code128) != self.a_dims:
            raise ValueError(f"A window code must have {self.a_dims} dims, got {len(code128)}")
        latent: list[int] = []
        for offset in range(0, self.a_dims, A_DIMS_PER_BYTE):
            latent.extend(1 if float(value) >= 0.0 else -1 for value in code128[offset : offset + BYTE_BITS])
        return latent

    def decode_b64_to_a128(self, latent64: Sequence[float | int]) -> list[int]:
        if len(latent64) != self.b_dims:
            raise ValueError(f"B window latent must have {self.b_dims} dims, got {len(latent64)}")
        code: list[int] = []
        for offset in range(0, self.b_dims, B_DIMS_PER_BYTE):
            visible = [1 if float(value) >= 0.0 else -1 for value in latent64[offset : offset + B_DIMS_PER_BYTE]]
            code.extend(visible)
            code.extend([0] * (A_DIMS_PER_BYTE - BYTE_BITS))
        return code

    def encode_window_b64(self, window: bytes | Sequence[int]) -> list[int]:
        return self.encode_a128_to_b64(self.encode_window_a128(window))

    def decode_b64_to_window(self, latent64: Sequence[float | int]) -> bytes:
        return self.decode_a128_to_window(self.decode_b64_to_a128(latent64))

    def byte_logits_from_b64(self, latent64: Sequence[float | int]) -> list[list[float]]:
        if len(latent64) != self.b_dims:
            raise ValueError(f"B window latent must have {self.b_dims} dims, got {len(latent64)}")
        logits: list[list[float]] = []
        for offset in range(0, self.b_dims, B_DIMS_PER_BYTE):
            visible = [1 if float(value) >= 0.0 else -1 for value in latent64[offset : offset + B_DIMS_PER_BYTE]]
            byte_logits = []
            for candidate in range(256):
                pattern = [1 if ((candidate >> bit) & 1) else -1 for bit in range(BYTE_BITS)]
                byte_logits.append(float(sum(a * b for a, b in zip(visible, pattern))))
            logits.append(byte_logits)
        return logits

    def b_encoder_entries(self) -> list[dict[str, int | float]]:
        """Sparse B encoder entries. Decoder is the exact transpose."""
        entries: list[dict[str, int | float]] = []
        for byte_idx in range(self.window_bytes):
            a_base = byte_idx * A_DIMS_PER_BYTE
            b_base = byte_idx * B_DIMS_PER_BYTE
            for bit_idx in range(BYTE_BITS):
                entries.append({"b_idx": b_base + bit_idx, "a_idx": a_base + bit_idx, "value": 1.0})
        return entries


def _checked_byte(byte_value: int) -> int:
    byte = int(byte_value)
    if not 0 <= byte <= 255:
        raise ValueError(f"byte value must be in [0,255], got {byte_value!r}")
    return byte


def _checked_window(window: bytes | Sequence[int], expected: int) -> list[int]:
    values = list(window)
    if len(values) != expected:
        raise ValueError(f"window must contain {expected} bytes, got {len(values)}")
    return [_checked_byte(value) for value in values]


def _bits_to_byte(bits: Iterable[int]) -> int:
    value = 0
    for bit_idx, bit in enumerate(bits):
        if int(bit) >= 0:
            value |= 1 << bit_idx
    return value


def byte_margin(logits: Sequence[float], target: int) -> float:
    target = _checked_byte(target)
    target_logit = float(logits[target])
    return target_logit - max(float(value) for idx, value in enumerate(logits) if idx != target)


def byte_margin_from_visible(visible: Sequence[float | int], target: int) -> float:
    """Compute byte margin without enumerating all 256 byte logits."""
    target = _checked_byte(target)
    signed = [1.0 if float(value) >= 0.0 else -1.0 for value in visible]
    target_pattern = [1.0 if ((target >> bit) & 1) else -1.0 for bit in range(BYTE_BITS)]
    target_logit = sum(a * b for a, b in zip(signed, target_pattern))
    abs_sum = float(len(signed))
    if all(a == b for a, b in zip(signed, target_pattern)):
        best_other = abs_sum - 2.0
    else:
        best_other = abs_sum
    return float(target_logit - best_other)


def canonical_artifact(codec: ABWindowCodec) -> dict[str, object]:
    payload: dict[str, object] = {
        "version": VERSION,
        "window_bytes": codec.window_bytes,
        "byte_bits": BYTE_BITS,
        "a_dims_per_byte": A_DIMS_PER_BYTE,
        "b_dims_per_byte": B_DIMS_PER_BYTE,
        "a_window_dims": codec.a_dims,
        "b_window_dims": codec.b_dims,
        "a_byte_encoder": "bits_little_endian_duplicated_2x",
        "b_encoder": "select_first_8_visible_lanes_per_byte",
        "b_decoder": "transpose_of_b_encoder",
        "latent_values": [-1, 1],
        "b_encoder_entries": codec.b_encoder_entries(),
        "notes": [
            "B inverse restores an A-decodable 128D surface.",
            "Redundant D22 lanes are intentionally zero after B inverse.",
            "All byte bit order is little-endian: bit0 first.",
        ],
    }
    payload["sha256"] = artifact_sha256(payload)
    return payload


def artifact_sha256(payload: dict[str, object]) -> str:
    stripped = {key: value for key, value in payload.items() if key != "sha256"}
    blob = json.dumps(stripped, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def verify_artifact(payload: dict[str, object]) -> None:
    expected = payload.get("sha256")
    actual = artifact_sha256(payload)
    if expected != actual:
        raise ValueError(f"artifact sha256 mismatch: expected {expected}, actual {actual}")
    if payload.get("version") != VERSION:
        raise ValueError(f"unsupported artifact version: {payload.get('version')!r}")
    if int(payload.get("window_bytes", -1)) != WINDOW_BYTES:
        raise ValueError("only the fixed 8-byte v1 artifact is supported")


def iter_acceptance_windows(eval_windows: int, seed: int) -> list[bytes]:
    rng = random.Random(seed)
    windows: list[bytes] = []
    zero = [0] * WINDOW_BYTES
    for pos in range(WINDOW_BYTES):
        for value in range(256):
            row = zero.copy()
            row[pos] = value
            windows.append(bytes(row))
    while len(windows) < eval_windows:
        windows.append(bytes(rng.randrange(256) for _ in range(WINDOW_BYTES)))
    return windows[:eval_windows]


def acceptance(eval_windows: int, seed: int) -> dict[str, object]:
    codec = ABWindowCodec()
    windows = iter_acceptance_windows(eval_windows, seed)
    exact = 0
    byte_correct = 0
    bit_correct = 0
    margin_min = float("inf")
    b_codes: dict[tuple[int, ...], bytes] = {}
    collisions = 0
    for window in windows:
        a128 = codec.encode_window_a128(window)
        b64 = codec.encode_a128_to_b64(a128)
        decoded_a128 = codec.decode_b64_to_a128(b64)
        decoded = codec.decode_a128_to_window(decoded_a128)
        if decoded == window:
            exact += 1
        for byte_idx, (pred, target) in enumerate(zip(decoded, window)):
            if pred == target:
                byte_correct += 1
            pred_bits = [(pred >> bit) & 1 for bit in range(BYTE_BITS)]
            target_bits = [(target >> bit) & 1 for bit in range(BYTE_BITS)]
            bit_correct += sum(int(a == b) for a, b in zip(pred_bits, target_bits))
            offset = byte_idx * B_DIMS_PER_BYTE
            margin_min = min(margin_min, byte_margin_from_visible(b64[offset : offset + B_DIMS_PER_BYTE], target))
        key = tuple(b64)
        previous = b_codes.get(key)
        if previous is not None and previous != window:
            collisions += 1
        b_codes[key] = window

    total_windows = len(windows)
    total_bytes = total_windows * WINDOW_BYTES
    total_bits = total_bytes * BYTE_BITS
    artifact = canonical_artifact(codec)
    verify_artifact(artifact)
    return {
        "verdict": "AB_WINDOW_CODEC_V1_PASS"
        if exact == total_windows and byte_correct == total_bytes and bit_correct == total_bits and collisions == 0 and margin_min > 0
        else "AB_WINDOW_CODEC_V1_FAIL",
        "version": VERSION,
        "eval_windows": total_windows,
        "window_exact_acc": exact / total_windows,
        "byte_exact_acc": byte_correct / total_bytes,
        "bit_acc": bit_correct / total_bits,
        "byte_margin_min": margin_min,
        "b_collision_count": collisions,
        "a_window_dims": codec.a_dims,
        "b_window_dims": codec.b_dims,
        "b_encoder_weight_count": len(codec.b_encoder_entries()),
        "artifact_sha256": artifact["sha256"],
    }


def write_report(out_dir: Path, metrics: dict[str, object], artifact_path: Path | None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ab_window_codec_acceptance.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    report = [
        "# AB Window Codec V1 Acceptance",
        "",
        f"Verdict: `{metrics['verdict']}`",
        "",
        "```text",
        "8 bytes -> A128 -> B64 -> A128 -> 8 bytes",
        "```",
        "",
        "## Metrics",
        "",
        f"- eval_windows: `{metrics['eval_windows']}`",
        f"- window_exact_acc: `{float(metrics['window_exact_acc']):.6f}`",
        f"- byte_exact_acc: `{float(metrics['byte_exact_acc']):.6f}`",
        f"- bit_acc: `{float(metrics['bit_acc']):.6f}`",
        f"- byte_margin_min: `{float(metrics['byte_margin_min']):.6f}`",
        f"- b_collision_count: `{metrics['b_collision_count']}`",
        f"- artifact_sha256: `{metrics['artifact_sha256']}`",
        "",
    ]
    if artifact_path is not None:
        report.append(f"Artifact: `{artifact_path.as_posix()}`")
        report.append("")
    (out_dir / "AB_WINDOW_CODEC_V1_ACCEPTANCE.md").write_text("\n".join(report), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true", help="Run AB codec acceptance.")
    parser.add_argument("--eval-windows", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--export-artifact", type=Path)
    parser.add_argument("--verify-artifact", type=Path)
    parser.add_argument("--out", type=Path)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    codec = ABWindowCodec()
    artifact_path: Path | None = None
    if args.export_artifact is not None:
        artifact = canonical_artifact(codec)
        args.export_artifact.parent.mkdir(parents=True, exist_ok=True)
        args.export_artifact.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
        artifact_path = args.export_artifact
    if args.verify_artifact is not None:
        verify_artifact(json.loads(args.verify_artifact.read_text(encoding="utf-8")))
    if args.self_test:
        metrics = acceptance(int(args.eval_windows), int(args.seed))
        if args.out is not None:
            write_report(args.out, metrics, artifact_path)
        print(json.dumps(metrics, indent=2))
        return 0 if metrics["verdict"] == "AB_WINDOW_CODEC_V1_PASS" else 1
    if args.export_artifact is None and args.verify_artifact is None:
        raise SystemExit("No action requested. Use --self-test, --export-artifact, or --verify-artifact.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
