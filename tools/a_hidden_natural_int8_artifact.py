#!/usr/bin/env python3
"""Export and verify native int8 A-HiddenNatural artifacts.

The hidden-natural A candidate is searched with discrete values, but the search
artifacts are JSON floats. This utility converts the layer weights to an
explicit int8 fixed-point format and verifies the full reciprocal roundtrip with
integer accumulation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence


VISIBLE_DIM = 8
HIDDEN_DIM = 8
CODE_DIM = 16
SCALE = 64
VERSION = "a_hidden_natural_margin_int8_v1"


def parse_entries(raw: str) -> list[tuple[int, int, float]]:
    entries: list[tuple[int, int, float]] = []
    for part in raw.split():
        left, mid, value = part.split(":")
        entries.append((int(left), int(mid), float(value)))
    return entries


def quantize_weight(value: float, *, scale: int = SCALE) -> int:
    q = int(round(float(value) * scale))
    if not -128 <= q <= 127:
        raise ValueError(f"weight {value} quantizes to {q}, outside int8 range")
    if abs((q / scale) - float(value)) > 1e-12:
        raise ValueError(f"weight {value} is not exactly representable with scale={scale}")
    return q


def quantize_entries(entries: Iterable[tuple[int, int, float]]) -> list[list[int]]:
    return [[int(row), int(col), quantize_weight(value)] for row, col, value in entries]


def signed_bits(byte: int) -> list[int]:
    return [1 if ((byte >> bit) & 1) else -1 for bit in range(VISIBLE_DIM)]


def bits_to_byte(bits: Sequence[int]) -> int:
    out = 0
    for bit_idx, value in enumerate(bits):
        if value >= 0:
            out |= 1 << bit_idx
    return out


def dense_q(entries: Sequence[Sequence[int]], rows: int, cols: int) -> list[list[int]]:
    matrix = [[0 for _ in range(cols)] for _ in range(rows)]
    for row, col, q in entries:
        matrix[int(row)][int(col)] = int(q)
    return matrix


def mat_vec(matrix: Sequence[Sequence[int]], vector: Sequence[int]) -> list[int]:
    return [sum(int(w) * int(vector[col]) for col, w in enumerate(row)) for row in matrix]


def transpose_mat_vec(matrix: Sequence[Sequence[int]], vector: Sequence[int]) -> list[int]:
    if not matrix:
        return []
    cols = len(matrix[0])
    out = [0 for _ in range(cols)]
    for row_idx, row in enumerate(matrix):
        for col_idx, weight in enumerate(row):
            out[col_idx] += int(weight) * int(vector[row_idx])
    return out


def forward_int(bits: Sequence[int], hidden_in: Sequence[Sequence[int]], hidden_out: Sequence[Sequence[int]]) -> dict[str, list[int]]:
    hidden = mat_vec(hidden_in, bits)
    code = mat_vec(hidden_out, hidden)
    mirror_hidden = transpose_mat_vec(hidden_out, code)
    logits = transpose_mat_vec(hidden_in, mirror_hidden)
    decoded_bits = [1 if value >= 0 else -1 for value in logits]
    return {
        "hidden_acc": hidden,
        "code_acc": code,
        "mirror_hidden_acc": mirror_hidden,
        "logits_acc": logits,
        "decoded_bits": decoded_bits,
    }


def byte_margin_from_logits(logits: Sequence[int], target: int) -> float:
    target_bits = signed_bits(target)
    target_score = sum(logit * bit for logit, bit in zip(logits, target_bits))
    best_other = None
    for candidate in range(256):
        if candidate == target:
            continue
        score = sum(logit * bit for logit, bit in zip(logits, signed_bits(candidate)))
        if best_other is None or score > best_other:
            best_other = score
    assert best_other is not None
    return float(target_score - best_other) / float(SCALE**4)


def verify_payload(payload: dict[str, object]) -> dict[str, object]:
    if payload.get("version") != VERSION:
        raise ValueError(f"unexpected artifact version: {payload.get('version')!r}")
    if int(payload.get("scale", 0)) != SCALE:
        raise ValueError(f"unexpected scale: {payload.get('scale')!r}")
    if payload.get("storage") != "int8_q6":
        raise ValueError(f"unexpected storage: {payload.get('storage')!r}")

    hidden_in = dense_q(payload["hidden_in_q"], HIDDEN_DIM, VISIBLE_DIM)  # type: ignore[index]
    hidden_out = dense_q(payload["hidden_out_q"], CODE_DIM, HIDDEN_DIM)  # type: ignore[index]
    exact = 0
    bit_correct = 0
    margin_min = float("inf")
    code_rows: set[tuple[int, ...]] = set()
    for byte in range(256):
        bits = signed_bits(byte)
        result = forward_int(bits, hidden_in, hidden_out)
        decoded_bits = result["decoded_bits"]
        decoded = bits_to_byte(decoded_bits)
        if decoded == byte:
            exact += 1
        bit_correct += sum(1 for expected, got in zip(bits, decoded_bits) if expected == got)
        margin_min = min(margin_min, byte_margin_from_logits(result["logits_acc"], byte))
        code_rows.add(tuple(result["code_acc"]))

    return {
        "verdict": "A_HIDDEN_NATURAL_INT8_ARTIFACT_PASS" if exact == 256 and bit_correct == 2048 and margin_min > 0 else "A_HIDDEN_NATURAL_INT8_ARTIFACT_FAIL",
        "exact_byte_acc": exact / 256.0,
        "bit_acc": bit_correct / 2048.0,
        "byte_margin_min": margin_min,
        "hidden_collisions": 256 - len(code_rows),
        "hidden_in_edge_count": len(payload["hidden_in_q"]),  # type: ignore[arg-type]
        "hidden_out_edge_count": len(payload["hidden_out_q"]),  # type: ignore[arg-type]
        "scale_power_for_logits": 4,
    }


def export_payload(source: Path) -> dict[str, object]:
    blob = json.loads(source.read_text(encoding="utf-8"))
    candidate = blob.get("best_candidate")
    if not isinstance(candidate, dict):
        raise ValueError(f"missing best_candidate in {source}")
    hidden_in_q = quantize_entries(parse_entries(str(candidate["hidden_in_entries"])))
    hidden_out_q = quantize_entries(parse_entries(str(candidate["hidden_out_entries"])))
    payload: dict[str, object] = {
        "version": VERSION,
        "name": "A-HiddenNaturalMarginPolish-int8",
        "storage": "int8_q6",
        "scale": SCALE,
        "value_formula": "weight = q / 64",
        "visible_dim": VISIBLE_DIM,
        "hidden_dim": HIDDEN_DIM,
        "code_dim": CODE_DIM,
        "decoder": "transpose_chain",
        "source": str(source),
        "source_verdict": blob.get("verdict"),
        "source_metrics": {
            "exact_byte_acc": candidate.get("exact_byte_acc"),
            "bit_acc": candidate.get("bit_acc"),
            "byte_margin_min": candidate.get("byte_margin_min"),
            "ascii_class_geometry": candidate.get("ascii_class_geometry"),
            "effective_copy_penalty": candidate.get("effective_copy_penalty"),
        },
        "hidden_in_q": hidden_in_q,
        "hidden_out_q": hidden_out_q,
    }
    payload["verification"] = verify_payload(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-source", type=Path)
    parser.add_argument("--verify-artifact", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    if args.export_source is None and args.verify_artifact is None:
        parser.error("provide --export-source or --verify-artifact")

    if args.export_source is not None:
        payload = export_payload(args.export_source)
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload["verification"], indent=2))
        return 0 if payload["verification"]["verdict"] == "A_HIDDEN_NATURAL_INT8_ARTIFACT_PASS" else 1  # type: ignore[index]

    assert args.verify_artifact is not None
    payload = json.loads(args.verify_artifact.read_text(encoding="utf-8"))
    metrics = verify_payload(payload)
    print(json.dumps(metrics, indent=2))
    return 0 if metrics["verdict"] == "A_HIDDEN_NATURAL_INT8_ARTIFACT_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
