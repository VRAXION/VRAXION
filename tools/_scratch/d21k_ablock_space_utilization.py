#!/usr/bin/env python3
"""
D21K A-space utilization diagnostics.

This is a diagnostic phase, not a search. It asks how the current A-block
candidates use their 16D code space:

- lane energy balance
- PCA/eigen spectrum
- pairwise distance spread
- ASCII class separation
- exact byte reconstruction/margin
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools._scratch.d21a_reciprocal_byte_ablock import (  # noqa: E402
    ReciprocalABlock,
    all_visible_patterns,
    evaluate_ablock,
    redundant_copy_entries,
)
from tools._scratch.d21f_ablock_natural_geometry import (  # noqa: E402
    ascii_geometry_score,
    geometry_metrics,
    parse_entries,
)
from tools._scratch.d21h_ablock_champion_compare import (  # noqa: E402
    c19_np,
    load_binary_c19,
    load_int4_c19,
    simple_geometry,
)


VISIBLE_DIM = 8
CODE_DIM = 16
UPPER = list(range(ord("A"), ord("Z") + 1))
LOWER = list(range(ord("a"), ord("z") + 1))
DIGITS = list(range(ord("0"), ord("9") + 1))
PUNCT = [ord(ch) for ch in ",.;:!?()[]{}'\"+-*/=_"]
SPACE = [ord(" "), ord("\n"), ord("\t")]
CLASS_GROUPS = {
    "upper": UPPER,
    "lower": LOWER,
    "digit": DIGITS,
    "punct": PUNCT,
    "space": SPACE,
}


def matrix_from_entries(raw: str) -> np.ndarray:
    matrix = np.zeros((CODE_DIM, VISIBLE_DIM), dtype=np.float64)
    for code_idx, visible_idx, value in parse_entries(raw):
        matrix[code_idx, visible_idx] = float(value)
    return matrix


def codes_from_matrix(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    block = ReciprocalABlock(VISIBLE_DIM, CODE_DIM, matrix)
    patterns = all_visible_patterns(VISIBLE_DIM)
    codes = block.encode_patterns(patterns)
    logits = codes @ matrix
    metrics = evaluate_ablock(block, patterns)
    geom = geometry_metrics(block)
    metrics.update(geom)
    metrics["ascii_class_geometry"] = ascii_geometry_score(metrics)
    return codes.astype(np.float64), logits.astype(np.float64), metrics


def load_d21a() -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    matrix = matrix_from_entries(" ".join(f"{c}:{v}:{w}" for c, v, w in redundant_copy_entries(VISIBLE_DIM, CODE_DIM)))
    return codes_from_matrix(matrix)


def load_d21g(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    blob = json.loads(path.read_text(encoding="utf-8"))
    entries = str(blob["best_natural_candidate"]["entries"])
    return codes_from_matrix(matrix_from_entries(entries))


def load_d21i(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    blob = json.loads(path.read_text(encoding="utf-8"))
    entries = str(blob["best_candidate"]["effective_entries"])
    return codes_from_matrix(matrix_from_entries(entries))


def load_d21j(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    blob = json.loads(path.read_text(encoding="utf-8"))
    entries = str(blob["best_candidate"]["effective_entries"])
    return codes_from_matrix(matrix_from_entries(entries))


def load_c19_int4(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    codes, logits, edges = load_int4_c19(path)
    exact, bit_acc, margin = margin_from_logits(logits)
    geom = simple_geometry(codes)
    return codes, logits, {
        "exact_byte_acc": exact,
        "bit_acc": bit_acc,
        "byte_margin_min": margin,
        "byte_argmax_acc": exact,
        "edge_count": edges,
        "ascii_class_geometry": geom["ascii_class_geometry"],
    }


def load_c19_binary(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    codes, logits, edges = load_binary_c19(path)
    exact, bit_acc, margin = margin_from_logits(logits)
    geom = simple_geometry(codes)
    return codes, logits, {
        "exact_byte_acc": exact,
        "bit_acc": bit_acc,
        "byte_margin_min": margin,
        "byte_argmax_acc": exact,
        "edge_count": edges,
        "ascii_class_geometry": geom["ascii_class_geometry"],
    }


def margin_from_logits(logits: np.ndarray) -> tuple[float, float, float]:
    patterns = all_visible_patterns(VISIBLE_DIM).astype(np.float64)
    exact = 0
    bit_ok = 0
    margins = []
    for byte in range(256):
        scores = patterns @ logits[byte, :VISIBLE_DIM]
        order = np.argsort(scores)
        best = int(order[-1])
        second = float(scores[order[-2]])
        target = byte
        target_bits = patterns[byte]
        pred_bits = np.where(logits[byte, :VISIBLE_DIM] >= 0.0, 1.0, -1.0)
        bit_ok += int(np.sum(pred_bits == target_bits))
        exact += int(best == target and np.all(pred_bits == target_bits))
        margins.append(float(scores[target] - second))
    return exact / 256.0, bit_ok / (256.0 * VISIBLE_DIM), float(min(margins))


def pairwise_distances(codes: np.ndarray) -> np.ndarray:
    diff = codes[:, None, :] - codes[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def eigen_spectrum(codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = codes - np.mean(codes, axis=0, keepdims=True)
    cov = centered.T @ centered / max(1, centered.shape[0] - 1)
    vals = np.linalg.eigvalsh(cov)[::-1]
    vals = np.maximum(vals, 0.0)
    ratios = vals / np.sum(vals) if np.sum(vals) > 1e-12 else vals
    return vals, ratios


def participation_ratio(vals: np.ndarray) -> float:
    total = float(np.sum(vals))
    denom = float(np.sum(vals * vals))
    if denom <= 1e-12:
        return 0.0
    return float((total * total) / denom)


def lane_energy(codes: np.ndarray) -> np.ndarray:
    return np.mean(codes * codes, axis=0)


def active_lane_count(energy: np.ndarray) -> int:
    if len(energy) == 0:
        return 0
    threshold = max(1e-9, float(np.max(energy)) * 0.01)
    return int(np.count_nonzero(energy > threshold))


def gini(values: np.ndarray) -> float:
    x = np.sort(np.asarray(values, dtype=np.float64))
    if len(x) == 0 or np.sum(x) <= 1e-12:
        return 0.0
    n = len(x)
    return float((2.0 * np.sum((np.arange(1, n + 1) * x)) / (n * np.sum(x))) - (n + 1.0) / n)


def class_centroid_stats(codes: np.ndarray) -> dict[str, float]:
    centroids: dict[str, np.ndarray] = {}
    intra: dict[str, float] = {}
    for name, members in CLASS_GROUPS.items():
        arr = codes[members]
        center = np.mean(arr, axis=0)
        centroids[name] = center
        intra[name] = float(np.mean(np.sqrt(np.sum((arr - center) ** 2, axis=1))))
    centroid_distances = []
    names = list(centroids)
    for idx, left in enumerate(names):
        for right in names[idx + 1 :]:
            centroid_distances.append(float(np.linalg.norm(centroids[left] - centroids[right])))
    return {
        "class_intra_mean": float(np.mean(list(intra.values()))),
        "class_centroid_distance_mean": float(np.mean(centroid_distances)) if centroid_distances else 0.0,
        "class_separation_ratio": float(np.mean(centroid_distances) / max(1e-12, np.mean(list(intra.values())))) if centroid_distances else 0.0,
    }


def summarize_candidate(name: str, codes: np.ndarray, logits: np.ndarray, metrics: dict[str, object]) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    energy = lane_energy(codes)
    vals, ratios = eigen_spectrum(codes)
    dist = pairwise_distances(codes)
    upper = dist[np.triu_indices_from(dist, k=1)]
    class_stats = class_centroid_stats(codes)
    row: dict[str, object] = {
        "candidate": name,
        "exact_byte_acc": float(metrics.get("exact_byte_acc", 0.0)),
        "bit_acc": float(metrics.get("bit_acc", 0.0)),
        "byte_margin_min": float(metrics.get("byte_margin_min", 0.0)),
        "ascii_class_geometry": float(metrics.get("ascii_class_geometry", 0.0)),
        "rank": int(np.linalg.matrix_rank(codes - np.mean(codes, axis=0, keepdims=True))),
        "active_lane_count": active_lane_count(energy),
        "lane_energy_mean": float(np.mean(energy)),
        "lane_energy_min": float(np.min(energy)),
        "lane_energy_max": float(np.max(energy)),
        "lane_energy_gini": gini(energy),
        "pca_dim_95": int(np.searchsorted(np.cumsum(ratios), 0.95) + 1) if np.sum(ratios) > 0 else 0,
        "participation_ratio": participation_ratio(vals),
        "top1_var_ratio": float(ratios[0]) if len(ratios) else 0.0,
        "top4_var_ratio": float(np.sum(ratios[:4])),
        "top8_var_ratio": float(np.sum(ratios[:8])),
        "pairwise_min": float(np.min(upper)),
        "pairwise_mean": float(np.mean(upper)),
        "pairwise_max": float(np.max(upper)),
    }
    row.update(class_stats)
    lane_rows = [
        {
            "candidate": name,
            "lane": idx,
            "energy": float(value),
            "energy_share": float(value / max(1e-12, np.sum(energy))),
        }
        for idx, value in enumerate(energy)
    ]
    spectrum_rows = [
        {
            "candidate": name,
            "component": idx + 1,
            "eigenvalue": float(vals[idx]),
            "variance_ratio": float(ratios[idx]),
            "cumulative_ratio": float(np.sum(ratios[: idx + 1])),
        }
        for idx in range(len(vals))
    ]
    return row, lane_rows, spectrum_rows


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def bar(value: float, max_value: float, width: int = 18) -> str:
    if max_value <= 1e-12:
        return "-" * width
    filled = int(round(width * value / max_value))
    return "#" * filled + "-" * (width - filled)


def write_report(out: Path, summary_rows: Sequence[dict[str, object]], lane_rows: Sequence[dict[str, object]], spectrum_rows: Sequence[dict[str, object]]) -> None:
    lines = [
        "# D21K A-Block Space Utilization Report",
        "",
        "Date: 2026-05-03",
        "",
        "## Summary",
        "",
        "```text",
        "candidate              exact margin geom  active rank pca95 PR    lane_gini pair_mean class_sep",
        "---------------------- ----- ------ ----- ------ ---- ----- ----- --------- --------- ---------",
    ]
    for row in summary_rows:
        lines.append(
            f"{str(row['candidate'])[:22]:22} "
            f"{float(row['exact_byte_acc']):5.3f} "
            f"{float(row['byte_margin_min']):6.3f} "
            f"{float(row['ascii_class_geometry']):5.3f} "
            f"{int(row['active_lane_count']):6d} "
            f"{int(row['rank']):4d} "
            f"{int(row['pca_dim_95']):5d} "
            f"{float(row['participation_ratio']):5.2f} "
            f"{float(row['lane_energy_gini']):9.3f} "
            f"{float(row['pairwise_mean']):9.3f} "
            f"{float(row['class_separation_ratio']):9.3f}"
        )
    lines.extend(["```", "", "## Lane Energy", ""])
    for candidate in [str(row["candidate"]) for row in summary_rows]:
        rows = [row for row in lane_rows if row["candidate"] == candidate]
        max_energy = max(float(row["energy"]) for row in rows) if rows else 0.0
        lines.append(f"### {candidate}")
        lines.append("```text")
        for row in rows:
            lines.append(f"A{int(row['lane']):02d} {float(row['energy']):8.3f} {bar(float(row['energy']), max_energy)}")
        lines.append("```")
        lines.append("")
    lines.extend(["## PCA Spectrum", ""])
    for candidate in [str(row["candidate"]) for row in summary_rows]:
        rows = [row for row in spectrum_rows if row["candidate"] == candidate][:8]
        max_ratio = max(float(row["variance_ratio"]) for row in rows) if rows else 0.0
        lines.append(f"### {candidate}")
        lines.append("```text")
        for row in rows:
            lines.append(
                f"PC{int(row['component']):02d} {float(row['variance_ratio']):7.4f} "
                f"cum={float(row['cumulative_ratio']):7.4f} {bar(float(row['variance_ratio']), max_ratio)}"
            )
        lines.append("```")
        lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- `active_lane_count` tells how many of the 16 A lanes carry non-trivial energy.")
    lines.append("- `rank` and PCA show intrinsic dimension; with 8 input bits, rank above 8 is not expected.")
    lines.append("- `lane_energy_gini` near 0 means balanced lane use; higher means concentrated lanes.")
    lines.append("- `class_separation_ratio` compares ASCII class centroid distance against within-class spread.")
    (out / "D21K_ABLOCK_SPACE_UTILIZATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    candidates = {
        "d21a_current": load_d21a,
        "d21g_natural": lambda: load_d21g(Path(args.d21g)),
        "d21i_hidden_gain": lambda: load_d21i(Path(args.d21i)),
        "d21j_hidden_natural": lambda: load_d21j(Path(args.d21j)),
        "legacy_c19_h24": lambda: load_c19_int4(Path(args.int4)),
        "binary_c19_h16": lambda: load_c19_binary(Path(args.binary)),
    }
    selected = [part.strip() for part in str(args.candidates).split(",") if part.strip()]
    summary_rows: list[dict[str, object]] = []
    lane_rows: list[dict[str, object]] = []
    spectrum_rows: list[dict[str, object]] = []
    for name in selected:
        if name not in candidates:
            raise SystemExit(f"unknown candidate: {name}")
        codes, logits, metrics = candidates[name]()
        summary, lanes, spectrum = summarize_candidate(name, codes, logits, metrics)
        summary_rows.append(summary)
        lane_rows.extend(lanes)
        spectrum_rows.extend(spectrum)
    write_csv(out / "a_space_summary.csv", summary_rows)
    write_csv(out / "a_lane_energy.csv", lane_rows)
    write_csv(out / "a_pca_spectrum.csv", spectrum_rows)
    payload = {"candidates": selected, "summary": summary_rows}
    (out / "a_space_top.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(out, summary_rows, lane_rows, spectrum_rows)
    print((out / "D21K_ABLOCK_SPACE_UTILIZATION_REPORT.md").read_text(encoding="utf-8"))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="D21K A-block space utilization diagnostics")
    parser.add_argument("--candidates", default="d21a_current,d21g_natural,d21i_hidden_gain,d21j_hidden_natural,legacy_c19_h24,binary_c19_h16")
    parser.add_argument("--d21g", default="output/phase_d21g_ablock_margin_aware_polish_20260503/main/margin_top.json")
    parser.add_argument("--d21i", default="output/phase_d21i_ablock_hidden_sparse_sweep_20260503/main/hidden_top.json")
    parser.add_argument("--d21j", default="output/phase_d21j_ablock_hidden_natural_search_20260503/main/hidden_natural_top.json")
    parser.add_argument("--int4", default="tools/byte_unit_winner_int4.json")
    parser.add_argument("--binary", default="output/byte_unit_champion_binary_c19_h16/byte_unit_winner_binary.json")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
