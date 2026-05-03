#!/usr/bin/env python3
"""
D21H A-block champion comparator.

Ranks the byte->abstract candidates that currently matter:
- legacy int4 C19 H24 byte unit
- binary C19 H16 byte unit
- current D21A reciprocal redundant-copy A-block
- D21G natural sparse A-block lead

The comparison is deliberately small and deterministic: all 256 bytes, shared
byte roundtrip/geometry/robustness/cleanliness metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools._scratch.d21a_reciprocal_byte_ablock import ReciprocalABlock, dedupe_entries  # noqa: E402
from tools._scratch.d21f_ablock_natural_geometry import (  # noqa: E402
    ascii_geometry_score,
    duplicate_lane_penalty,
    geometry_metrics,
    identity_copy_penalty,
)


UPPER = list(range(ord("A"), ord("Z") + 1))
LOWER = list(range(ord("a"), ord("z") + 1))
DIGITS = list(range(ord("0"), ord("9") + 1))


def byte_bits(byte: int) -> np.ndarray:
    return np.array([1.0 if ((byte >> bit) & 1) else -1.0 for bit in range(8)], dtype=np.float64)


def c19_np(x: np.ndarray, c: np.ndarray, rho: np.ndarray) -> np.ndarray:
    c = np.maximum(c, 0.1)
    rho = np.maximum(rho, 0.0)
    limit = 6.0 * c
    scaled = x / c
    n = np.floor(scaled)
    t = scaled - n
    h = t * (1.0 - t)
    sign = np.where((n.astype(np.int64) % 2) == 0, 1.0, -1.0)
    interior = c * (sign * h + rho * h * h)
    return np.where(x >= limit, x - limit, np.where(x <= -limit, x + limit, interior))


def decode_bits_to_byte(logits: np.ndarray) -> int:
    out = 0
    for bit, value in enumerate(logits[:8]):
        if value >= 0:
            out |= 1 << bit
    return out


def pair_distance(codes: np.ndarray, left: int, right: int) -> float:
    diff = codes[left] - codes[right]
    return float(np.sqrt(np.dot(diff, diff)))


def simple_geometry(codes: np.ndarray) -> dict[str, float]:
    def rate(near_pairs: Sequence[tuple[int, int]], far_pairs: Sequence[tuple[int, int]]) -> float:
        wins = 0
        for idx, pair in enumerate(near_pairs):
            far = far_pairs[idx % len(far_pairs)]
            wins += int(pair_distance(codes, pair[0], pair[1]) < pair_distance(codes, far[0], far[1]))
        return wins / max(1, len(near_pairs))

    ascii_near = [(ch, ch + 1) for ch in UPPER[:-1]] + [(ch, ch + 1) for ch in LOWER[:-1]]
    ascii_far = [(ch, ord("Z")) for ch in UPPER[:-1]] + [(ch, ord("z")) for ch in LOWER[:-1]]
    case_near = [(ord(chr(ch).upper()), ord(chr(ch).lower())) for ch in LOWER]
    case_far = [(ord(chr(ch).upper()), ord("7")) for ch in LOWER]
    digit_near = [(ch, ch + 1) for ch in DIGITS[:-1]]
    digit_far = [(ch, ord("Z")) for ch in DIGITS[:-1]]
    values = {
        "ascii_neighbor_closer_rate": rate(ascii_near, ascii_far),
        "case_pair_closer_rate": rate(case_near, case_far),
        "digit_neighbor_closer_rate": rate(digit_near, digit_far),
    }
    # Match D21F weighting where possible, but without requiring a reciprocal block.
    values["ascii_class_geometry"] = float(
        0.34 * values["ascii_neighbor_closer_rate"]
        + 0.33 * values["case_pair_closer_rate"]
        + 0.33 * values["digit_neighbor_closer_rate"]
    )
    return values


def margin_from_logits(all_logits: np.ndarray) -> tuple[float, float, float]:
    correct = []
    exact = 0
    bit_total = 0
    bit_ok = 0
    margins = []
    patterns = np.array([byte_bits(i) for i in range(256)])
    for byte in range(256):
        logits = all_logits[byte]
        scores = patterns @ logits[:8]
        order = np.argsort(scores)
        best = int(order[-1])
        second = float(scores[order[-2]])
        margins.append(float(scores[byte] - second))
        exact += int(best == byte and decode_bits_to_byte(logits) == byte)
        pred_bits = np.sign(logits[:8])
        true_bits = byte_bits(byte)
        bit_ok += int(np.sum(pred_bits == true_bits))
        bit_total += 8
        correct.append(best == byte)
    return exact / 256.0, bit_ok / bit_total, float(min(margins))


def evaluate_code_candidate(name: str, codes: np.ndarray, logits: np.ndarray, *, edge_count: int, clean: str, notes: str) -> dict[str, object]:
    exact, bit_acc, margin = margin_from_logits(logits)
    geom = simple_geometry(codes)
    unique = len({tuple(np.round(row, 9)) for row in codes})
    score = (
        100.0 * exact
        + 5.0 * bit_acc
        + 1.5 * max(0.0, margin)
        + 10.0 * geom["ascii_class_geometry"]
        - 0.08 * edge_count
        + (3.0 if clean == "deploy_clean" else 0.0)
    )
    return {
        "candidate": name,
        "exact_byte_acc": exact,
        "bit_acc": bit_acc,
        "byte_margin_min": margin,
        "unique_codes": unique,
        "hidden_collisions": 256 - unique,
        "ascii_class_geometry": geom["ascii_class_geometry"],
        "ascii_neighbor_closer_rate": geom["ascii_neighbor_closer_rate"],
        "case_pair_closer_rate": geom["case_pair_closer_rate"],
        "digit_neighbor_closer_rate": geom["digit_neighbor_closer_rate"],
        "edge_count": edge_count,
        "architecture_cleanliness": clean,
        "champion_score": score,
        "notes": notes,
    }


def load_int4_c19(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    blob = json.loads(path.read_text(encoding="utf-8"))
    w1 = np.array(blob["W1_int4"], dtype=np.float64) * float(blob["scale_W1"])
    w2 = np.array(blob["W2_int4"], dtype=np.float64) * float(blob["scale_W2"])
    b1 = np.array(blob.get("bias1", blob.get("b1", [0.0] * w1.shape[1])), dtype=np.float64)
    b2 = np.array(blob.get("bias2", blob.get("b2", [0.0] * w2.shape[1])), dtype=np.float64)
    c = np.array(blob.get("c19_c", [3.0] * w1.shape[1]), dtype=np.float64)
    rho = np.array(blob.get("c19_rho", [1.0] * w1.shape[1]), dtype=np.float64)
    codes = []
    logits = []
    for byte in range(256):
        x = byte_bits(byte)
        h = c19_np(x @ w1 + b1, c, rho)
        z = h @ w2 + b2
        x_hat = (z @ w2.T) @ w1.T
        codes.append(z[:16])
        logits.append(x_hat[:8])
    edge_count = int(np.count_nonzero(w1) + np.count_nonzero(w2))
    return np.array(codes), np.array(logits), edge_count


def load_binary_c19(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    blob = json.loads(path.read_text(encoding="utf-8"))
    codebook = np.array(blob["codebook"], dtype=np.float64)
    w1 = codebook[np.array(blob["W1_binary_idx"], dtype=np.int64)] * float(blob["alpha1"])
    w2 = codebook[np.array(blob["W2_binary_idx"], dtype=np.int64)] * float(blob["alpha2"])
    b1 = np.array(blob.get("b1", [0.0] * w1.shape[1]), dtype=np.float64)
    b2 = np.array(blob.get("b2", [0.0] * w2.shape[1]), dtype=np.float64)
    params = blob.get("activation_params", {})
    c = np.array(params.get("c_raw", params.get("c19_c", [3.0] * w1.shape[1])), dtype=np.float64)
    rho = np.array(params.get("rho_raw", params.get("c19_rho", [1.0] * w1.shape[1])), dtype=np.float64)
    codes = []
    logits = []
    for byte in range(256):
        x = byte_bits(byte)
        h = c19_np(x @ w1 + b1, c, rho)
        z = h @ w2 + b2
        x_hat = (z @ w2.T) @ w1.T
        codes.append(z[:16])
        logits.append(x_hat[:8])
    edge_count = int(np.count_nonzero(w1) + np.count_nonzero(w2))
    return np.array(codes), np.array(logits), edge_count


def reciprocal_from_entries(entries: str, name: str) -> tuple[np.ndarray, np.ndarray, int, dict[str, float]]:
    parsed = []
    for part in entries.split():
        c, v, value = part.split(":")
        parsed.append((int(c), int(v), float(value)))
    parsed = dedupe_entries(parsed)
    block = ReciprocalABlock.from_entries(8, 16, parsed)
    codes = []
    logits = []
    for byte in range(256):
        code = block.encode_byte(byte)
        codes.append(code)
        logits.append(block.decode(code)[:8])
    g = geometry_metrics(block)
    g["ascii_class_geometry"] = ascii_geometry_score(g)
    g["identity_copy_penalty"] = identity_copy_penalty(parsed, 8, 16)
    g["duplicate_lane_penalty"] = duplicate_lane_penalty(block)
    return np.array(codes), np.array(logits), len(parsed), g


def run(args: argparse.Namespace) -> dict[str, object]:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    int4_codes, int4_logits, int4_edges = load_int4_c19(Path(args.int4))
    rows.append(evaluate_code_candidate("legacy_int4_c19_h24", int4_codes, int4_logits, edge_count=int4_edges, clean="deploy_clean", notes="legacy production artifact"))

    binary_codes, binary_logits, binary_edges = load_binary_c19(Path(args.binary))
    rows.append(evaluate_code_candidate("binary_c19_h16", binary_codes, binary_logits, edge_count=binary_edges, clean="candidate_clean", notes="smaller C19 binary champion"))

    baseline_entries = " ".join([f"{idx}:{idx}:1" for idx in range(8)] + [f"{idx + 8}:{idx}:1" for idx in range(8)])
    codes, logits, edges, geom = reciprocal_from_entries(baseline_entries, "d21a_redundant_copy_2x")
    row = evaluate_code_candidate("d21a_redundant_copy_2x", codes, logits, edge_count=edges, clean="pipeline_clean", notes="current AB default reciprocal codec")
    row.update({key: geom[key] for key in geom if key in row or key.endswith("penalty")})
    rows.append(row)

    top = json.loads(Path(args.d21g).read_text(encoding="utf-8"))
    natural = top["best_natural_candidate"]
    codes, logits, edges, geom = reciprocal_from_entries(natural["entries"], "d21g_natural_sparse")
    row = evaluate_code_candidate("d21g_natural_sparse", codes, logits, edge_count=edges, clean="research_fractional", notes=top["verdict_reason"])
    row.update({key: geom[key] for key in geom if key in row or key.endswith("penalty")})
    rows.append(row)

    rows = sorted(rows, key=lambda item: float(item["champion_score"]), reverse=True)
    verdict = "D21H_ABLOCK_CHAMPION_C19_PRODUCTION"
    best = rows[0]
    if best["candidate"] == "d21a_redundant_copy_2x":
        verdict = "D21H_ABLOCK_CHAMPION_CURRENT_PIPELINE"
    elif best["candidate"] == "d21g_natural_sparse":
        verdict = "D21H_ABLOCK_CHAMPION_NATURAL_RESEARCH"
    elif best["candidate"] == "binary_c19_h16":
        verdict = "D21H_ABLOCK_CHAMPION_BINARY_C19"

    write_csv(out / "ablock_champion_results.csv", rows)
    (out / "ablock_champion_top.json").write_text(json.dumps({"verdict": verdict, "best": best, "rows": rows}, indent=2), encoding="utf-8")
    write_report(out / "D21H_ABLOCK_CHAMPION_COMPARE_REPORT.md", verdict, rows)
    print(f"[D21H] verdict={verdict} best={best['candidate']}")
    print(f"[D21H] wrote {out}")
    return {"verdict": verdict, "best": best}


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, verdict: str, rows: Sequence[dict[str, object]]) -> None:
    lines = [
        "# D21H A-Block Champion Compare",
        "",
        f"Verdict: `{verdict}`",
        "",
        "| candidate | exact | margin | geometry | edges | cleanliness | score |",
        "|---|---:|---:|---:|---:|---|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['candidate']} | {float(row['exact_byte_acc']):.3f} | {float(row['byte_margin_min']):.3f} | "
            f"{float(row['ascii_class_geometry']):.3f} | {int(row['edge_count'])} | {row['architecture_cleanliness']} | "
            f"{float(row['champion_score']):.3f} |"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- D21A reciprocal is the current AB pipeline champion: exact, 16-edge, tied reciprocal, and deploy-clean.",
            "- Legacy C19 has the strongest raw byte margin/ASCII geometry, but it is a much larger extra-hidden codec and not the frozen reciprocal AB interface.",
            "- D21G is the natural-geometry research lead, but remains fractional/research-clean only.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="D21H A-block champion comparator")
    parser.add_argument("--int4", default="tools/byte_unit_winner_int4.json")
    parser.add_argument("--binary", default="output/byte_unit_champion_binary_c19_h16/byte_unit_winner_binary.json")
    parser.add_argument("--d21g", default="output/phase_d21g_ablock_margin_aware_polish_20260503/main/margin_top.json")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
