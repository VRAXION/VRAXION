#!/usr/bin/env python3
"""A-v2 AB compatibility gate.

Tests whether the current A-v2 hidden-natural int8 artifact can replace the
A-StableCopy16 byte encoder under an AB-style reciprocal B64 bridge.

The primary compatibility target is not only roundtrip exactness. The useful
pass requires B64 to remain byte-bit semantic, so existing B64 worker surfaces
can plausibly remain compatible.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.ab_window_codec import (  # noqa: E402
    BYTE_BITS,
    WINDOW_BYTES,
    byte_margin,
    iter_acceptance_windows,
)


VERSION = "ab_window_codec_a_v2_candidate_v1"
DEFAULT_A = Path("tools/a_v2_hidden_natural_int8_candidate.json")
DEFAULT_OUT = Path("output/phase_a_v2_ab_compatibility_gate_20260503")
EPS = 1e-9


@dataclass(frozen=True)
class AArtifact:
    path: Path
    payload: dict[str, object]
    encoder: np.ndarray  # code_dim x visible_dim


@dataclass(frozen=True)
class Bridge:
    family: str
    encoder: np.ndarray  # B8 x A16
    is_control: bool = False


def bits_for_byte(byte: int) -> np.ndarray:
    return np.array([1.0 if ((int(byte) >> bit) & 1) else -1.0 for bit in range(BYTE_BITS)], dtype=np.float64)


def bits_to_byte(bits: Iterable[float]) -> int:
    value = 0
    for bit_idx, value_bit in enumerate(bits):
        if float(value_bit) >= 0.0:
            value |= 1 << bit_idx
    return value


def load_a_artifact(path: Path) -> AArtifact:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("storage") != "int8_q6":
        raise ValueError(f"expected int8_q6 artifact, got {payload.get('storage')!r}")
    visible_dim = int(payload["visible_dim"])
    hidden_dim = int(payload["hidden_dim"])
    code_dim = int(payload["code_dim"])
    scale = float(payload["scale"])
    if visible_dim != BYTE_BITS:
        raise ValueError(f"expected visible_dim={BYTE_BITS}, got {visible_dim}")
    hidden_in = np.zeros((hidden_dim, visible_dim), dtype=np.float64)
    hidden_out = np.zeros((code_dim, hidden_dim), dtype=np.float64)
    for row, col, q in payload["hidden_in_q"]:
        hidden_in[int(row), int(col)] = float(q) / scale
    for row, col, q in payload["hidden_out_q"]:
        hidden_out[int(row), int(col)] = float(q) / scale
    return AArtifact(path=path, payload=payload, encoder=hidden_out @ hidden_in)


def stable_copy_encoder() -> np.ndarray:
    encoder = np.zeros((16, BYTE_BITS), dtype=np.float64)
    for bit in range(BYTE_BITS):
        encoder[bit, bit] = 1.0
        encoder[BYTE_BITS + bit, bit] = 1.0
    return encoder


def encode_a16(encoder: np.ndarray, byte: int) -> np.ndarray:
    return encoder @ bits_for_byte(byte)


def decode_a16(encoder: np.ndarray, code16: Sequence[float]) -> tuple[int, np.ndarray]:
    bit_logits = encoder.T @ np.asarray(code16, dtype=np.float64)
    return bits_to_byte(bit_logits), bit_logits


def bridge_families(a_encoder: np.ndarray, *, seed: int) -> list[Bridge]:
    select_first = np.zeros((BYTE_BITS, a_encoder.shape[0]), dtype=np.float64)
    for bit in range(BYTE_BITS):
        select_first[bit, bit] = 1.0

    pinv = np.linalg.pinv(a_encoder)
    gram = a_encoder.T

    rng = np.random.default_rng(seed)
    random_projection = rng.choice([-1.0, 1.0], size=(BYTE_BITS, a_encoder.shape[0])) / 4.0
    # This is not an adversarial null: a B-basis permutation is still a valid
    # reciprocal codec, but it breaks the canonical byte-bit lane semantics.
    shuffled = pinv[[1, 0, 3, 2, 5, 4, 7, 6], :].copy()

    return [
        Bridge("select_first8_current_b", select_first),
        Bridge("pinv_bit_bridge", pinv),
        Bridge("a_transpose_gram_bridge", gram),
        Bridge("b_basis_permutation_gauge", shuffled),
        Bridge("random_projection_control", random_projection, is_control=True),
    ]


def quantized_entries(matrix: np.ndarray, *, scale: int = 4096, eps: float = 1e-12) -> list[dict[str, int | float]]:
    entries: list[dict[str, int | float]] = []
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = float(matrix[row, col])
            if abs(value) <= eps:
                continue
            entries.append({"row": row, "col": col, "value": value, "q": int(round(value * scale))})
    return entries


def byte_metrics(a_encoder: np.ndarray, bridge: Bridge, byte: int) -> tuple[int, int, int, float, tuple[float, ...]]:
    target_bits = bits_for_byte(byte)
    a16 = encode_a16(a_encoder, byte)
    b8 = bridge.encoder @ a16
    decoded_a16 = bridge.encoder.T @ b8
    decoded_byte, bit_logits = decode_a16(a_encoder, decoded_a16)
    bit_pred = np.array([1.0 if value >= 0.0 else -1.0 for value in bit_logits], dtype=np.float64)
    bit_correct = int(np.sum(bit_pred == target_bits))
    semantic_bits = np.array([1.0 if value >= 0.0 else -1.0 for value in b8], dtype=np.float64)
    semantic_correct = int(np.sum(semantic_bits == target_bits))
    logits = []
    for candidate in range(256):
        logits.append(float(np.dot(bits_for_byte(candidate), bit_logits)))
    margin = byte_margin(logits, byte)
    return decoded_byte, bit_correct, semantic_correct, margin, tuple(float(round(v, 12)) for v in b8)


def evaluate_bridge(
    a_name: str,
    a_encoder: np.ndarray,
    bridge: Bridge,
    *,
    eval_windows: int,
    seed: int,
) -> dict[str, object]:
    windows = iter_acceptance_windows(eval_windows, seed)
    exact_windows = 0
    byte_correct = 0
    bit_correct = 0
    semantic_correct = 0
    margin_min = float("inf")
    b_codes: dict[tuple[float, ...], bytes] = {}
    collisions = 0

    byte_cache = {byte: byte_metrics(a_encoder, bridge, byte) for byte in range(256)}
    for window in windows:
        decoded_values: list[int] = []
        b_key_parts: list[float] = []
        for byte in window:
            decoded, bit_ok, semantic_ok, margin, b_key = byte_cache[int(byte)]
            decoded_values.append(decoded)
            if decoded == int(byte):
                byte_correct += 1
            bit_correct += bit_ok
            semantic_correct += semantic_ok
            margin_min = min(margin_min, margin)
            b_key_parts.extend(b_key)
        decoded_window = bytes(decoded_values)
        if decoded_window == window:
            exact_windows += 1
        b_key_tuple = tuple(b_key_parts)
        previous = b_codes.get(b_key_tuple)
        if previous is not None and previous != window:
            collisions += 1
        b_codes[b_key_tuple] = window

    total_windows = len(windows)
    total_bytes = total_windows * WINDOW_BYTES
    total_bits = total_bytes * BYTE_BITS
    p_e = bridge.encoder @ a_encoder
    identity_error = float(np.max(np.abs(p_e - np.eye(BYTE_BITS))))
    return {
        "a_candidate": a_name,
        "bridge_family": bridge.family,
        "is_control": bridge.is_control,
        "eval_windows": total_windows,
        "window_exact_acc": exact_windows / total_windows,
        "byte_exact_acc": byte_correct / total_bytes,
        "bit_acc": bit_correct / total_bits,
        "b_bit_semantic_acc": semantic_correct / total_bits,
        "byte_margin_min": margin_min,
        "b_collision_count": collisions,
        "b_encoder_weight_count_per_byte": int(np.count_nonzero(np.abs(bridge.encoder) > 1e-12)),
        "b_encoder_weight_count_window": int(np.count_nonzero(np.abs(bridge.encoder) > 1e-12)) * WINDOW_BYTES,
        "b_decoder_is_transpose": True,
        "reciprocity_error": 0.0,
        "bit_recovery_identity_max_error": identity_error,
        "verdict": bridge_verdict(bridge, exact_windows, total_windows, byte_correct, total_bytes, semantic_correct, total_bits, margin_min, collisions),
    }


def bridge_verdict(
    bridge: Bridge,
    exact_windows: int,
    total_windows: int,
    byte_correct: int,
    total_bytes: int,
    semantic_correct: int,
    total_bits: int,
    margin_min: float,
    collisions: int,
) -> str:
    exact = exact_windows == total_windows and byte_correct == total_bytes and margin_min > 0.0 and collisions == 0
    semantic = semantic_correct == total_bits
    if bridge.is_control:
        return "CONTROL_LEAK" if exact else "CONTROL_FAILS"
    if exact and semantic:
        return "BRIDGE_EXACT_B64_SEMANTIC_PASS"
    if exact:
        return "BRIDGE_CODEC_ONLY_PASS"
    return "BRIDGE_FAIL"


def final_verdict(rows: Sequence[dict[str, object]]) -> tuple[str, str]:
    primary = [row for row in rows if row["bridge_family"] == "pinv_bit_bridge"]
    controls = [row for row in rows if bool(row["is_control"])]
    control_clean = all(str(row["verdict"]) != "CONTROL_LEAK" and float(row["window_exact_acc"]) <= 0.01 for row in controls)
    if primary:
        row = primary[0]
        if str(row["verdict"]) == "BRIDGE_EXACT_B64_SEMANTIC_PASS" and control_clean:
            return (
                "A_V2_AB_COMPATIBILITY_PASS",
                "A-v2 H12 can sit under an AB-style reciprocal B64 bridge while preserving byte-bit B64 semantics.",
            )
        if str(row["verdict"]) == "BRIDGE_EXACT_B64_SEMANTIC_PASS":
            return "A_V2_AB_CONTROL_LEAK", "primary bridge passes, but at least one control also solves too often"
        if str(row["verdict"]) == "BRIDGE_CODEC_ONLY_PASS":
            return "A_V2_AB_CODEC_ONLY_PASS", "primary bridge reconstructs, but B64 no longer remains byte-bit semantic"
    exact_other = [row for row in rows if not bool(row["is_control"]) and str(row["verdict"]).endswith("_PASS")]
    if exact_other:
        best = max(exact_other, key=lambda row: float(row["window_exact_acc"]))
        return "A_V2_AB_ALT_BRIDGE_PASS", f"non-primary bridge {best['bridge_family']} passed, but pinv bridge did not"
    return "A_V2_AB_INCOMPATIBLE", "no non-control bridge preserved exact AB window reconstruction"


def artifact_sha256(payload: dict[str, object]) -> str:
    stripped = {key: value for key, value in payload.items() if key != "sha256"}
    blob = json.dumps(stripped, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def build_bridge_artifact(a_artifact: AArtifact, bridge: Bridge, verdict_name: str, rows: Sequence[dict[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {
        "version": VERSION,
        "name": "AB-v2-H12-bit-bridge-candidate",
        "source_verdict": verdict_name,
        "a_artifact": str(a_artifact.path),
        "a_name": a_artifact.payload.get("name", ""),
        "a_source_metrics": a_artifact.payload.get("source_metrics", {}),
        "window_bytes": WINDOW_BYTES,
        "byte_bits": BYTE_BITS,
        "a_dims_per_byte": int(a_artifact.encoder.shape[0]),
        "b_dims_per_byte": BYTE_BITS,
        "a_window_dims": WINDOW_BYTES * int(a_artifact.encoder.shape[0]),
        "b_window_dims": WINDOW_BYTES * BYTE_BITS,
        "a_byte_encoder": "tools/a_v2_hidden_natural_int8_candidate.json",
        "b_bridge_family": bridge.family,
        "b_encoder": "pseudoinverse(A_encoder) recovers byte-bit B8 from A16",
        "b_decoder": "transpose_of_b_encoder",
        "b64_semantics": "little_endian_signed_byte_bits",
        "b_encoder_scale_for_q": 4096,
        "b_encoder_entries_per_byte": quantized_entries(bridge.encoder, scale=4096),
        "acceptance_rows": list(rows),
        "notes": [
            "This is an AB-v2 compatibility artifact, not a replacement for ab_window_codec_v1.",
            "B64 remains byte-bit semantic under the pinv bridge.",
            "Generated output data remains untracked; this compact artifact is the tracked candidate.",
        ],
    }
    payload["sha256"] = artifact_sha256(payload)
    return payload


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_report(out: Path, rows: Sequence[dict[str, object]], verdict_name: str, reason: str, artifact_path: Path | None) -> None:
    lines = [
        "# A-v2 AB Compatibility Gate",
        "",
        "Date: 2026-05-03",
        "",
        "## Verdict",
        "",
        "```text",
        verdict_name,
        reason,
        "```",
        "",
        "## Visual",
        "",
        "```text",
        "byte bits",
        "   |",
        "   v",
        "A-v2-H12 A16",
        "   |  B encoder = pseudoinverse(A)",
        "   v",
        "B64 signed byte-bit bus",
        "   |  B decoder = B encoder.T",
        "   v",
        "A-decodable surface",
        "   |  A-v2 mirror decoder",
        "   v",
        "byte bits",
        "```",
        "",
        "## Bridge Results",
        "",
        "```text",
        "bridge                         window byte  bits  b_sem margin coll weights verdict",
        "------------------------------ ------ ----- ----- ----- ------ ---- ------- -------------------------------",
    ]
    for row in rows:
        lines.append(
            f"{str(row['bridge_family'])[:30]:30} "
            f"{float(row['window_exact_acc']):6.3f} "
            f"{float(row['byte_exact_acc']):5.3f} "
            f"{float(row['bit_acc']):5.3f} "
            f"{float(row['b_bit_semantic_acc']):5.3f} "
            f"{float(row['byte_margin_min']):6.3f} "
            f"{int(row['b_collision_count']):4d} "
            f"{int(row['b_encoder_weight_count_window']):7d} "
            f"{row['verdict']}"
        )
    lines.extend(["```", ""])
    if artifact_path is not None:
        lines.extend(["## Artifact", "", f"`{artifact_path.as_posix()}`", ""])
    (out / "A_V2_AB_COMPATIBILITY_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    a_artifact = load_a_artifact(Path(args.a_artifact))
    rows = [
        evaluate_bridge(
            str(a_artifact.payload.get("name", "A-v2")),
            a_artifact.encoder,
            bridge,
            eval_windows=int(args.eval_windows),
            seed=int(args.seed),
        )
        for bridge in bridge_families(a_artifact.encoder, seed=int(args.seed))
    ]

    verdict_name, reason = final_verdict(rows)
    write_csv(out / "compatibility_results.csv", [row for row in rows if not bool(row["is_control"])])
    write_csv(out / "compatibility_controls.csv", [row for row in rows if bool(row["is_control"])])
    artifact_path: Path | None = None
    if verdict_name == "A_V2_AB_COMPATIBILITY_PASS":
        bridge = next(bridge for bridge in bridge_families(a_artifact.encoder, seed=int(args.seed)) if bridge.family == "pinv_bit_bridge")
        payload = build_bridge_artifact(a_artifact, bridge, verdict_name, rows)
        artifact_path = Path(args.export_artifact) if args.export_artifact else out / "ab_v2_compatibility_artifact.json"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    top = {
        "verdict": verdict_name,
        "reason": reason,
        "a_artifact": str(a_artifact.path),
        "eval_windows": int(args.eval_windows),
        "seed": int(args.seed),
        "rows": rows,
        "artifact": str(artifact_path) if artifact_path is not None else "",
    }
    (out / "ab_v2_compatibility_top.json").write_text(json.dumps(top, indent=2), encoding="utf-8")
    write_report(out, rows, verdict_name, reason, artifact_path)
    print(json.dumps({"verdict": verdict_name, "reason": reason, "artifact": str(artifact_path or "")}, indent=2))
    return 0 if verdict_name == "A_V2_AB_COMPATIBILITY_PASS" else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a-artifact", default=str(DEFAULT_A))
    parser.add_argument("--eval-windows", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=20260503)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--export-artifact", default="")
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
