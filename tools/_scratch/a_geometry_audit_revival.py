#!/usr/bin/env python3
"""A-GeometryAuditRevival.

Revives the old byte-embedding "similarity structure" audit against the current
A-block candidates. This is not a search. It only measures whether exact byte
codes also form a useful geometry:

- ASCII neighbors closer than far ASCII pairs.
- case pairs closer than unrelated bytes.
- digit / letter / punctuation groups cluster.
- nearest neighbors are human-readable rather than arbitrary.
- effective rank is high enough and cosine overlap is not too high.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

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
    ASCII_FAR_PAIRS,
    ASCII_NEAR_PAIRS,
    CASE_FAR_PAIRS,
    CASE_NEAR_PAIRS,
    DIGIT_FAR_PAIRS,
    DIGIT_NEAR_PAIRS,
    PUNCT,
    SPACE,
    UPPER,
    LOWER,
    DIGITS,
    ascii_geometry_score,
    class_cluster_score,
    closer_rate,
    code_distance,
    duplicate_lane_penalty,
    identity_copy_penalty,
    parse_entries,
    punct_separation_score,
    random_far_margin,
)

VISIBLE_DIM = 8
CODE_DIM = 16

DEFAULT_D21G = Path("output/phase_d21g_ablock_margin_aware_polish_20260503/main/margin_top.json")
DEFAULT_D21I = Path("output/phase_d21i_ablock_hidden_sparse_sweep_20260503/main/hidden_top.json")
DEFAULT_D21J = Path("output/phase_d21j_ablock_hidden_natural_search_20260503/main/hidden_natural_top.json")

GROUPS = {
    "upper": UPPER,
    "lower": LOWER,
    "digits": DIGITS,
    "punct": PUNCT,
    "space": SPACE,
}
PROBE_BYTES = [
    ord("A"),
    ord("B"),
    ord("Z"),
    ord("a"),
    ord("z"),
    ord("0"),
    ord("7"),
    ord("9"),
    ord(","),
    ord("."),
    ord(" "),
]
ASCII_GRID = [
    ("digits", list(range(ord("0"), ord("9") + 1))),
    ("upper_AZ", list(range(ord("A"), ord("Z") + 1))),
    ("lower_az", list(range(ord("a"), ord("z") + 1))),
    ("punct", [ord(ch) for ch in ".,:;!?+-*/_=()[]{}"]),
]


@dataclass(frozen=True)
class Candidate:
    name: str
    source: str
    entries: tuple[tuple[int, int, float], ...]


def entries_to_matrix(entries: Iterable[tuple[int, int, float]]) -> np.ndarray:
    matrix = np.zeros((CODE_DIM, VISIBLE_DIM), dtype=np.float64)
    for code_idx, visible_idx, value in entries:
        if 0 <= code_idx < CODE_DIM and 0 <= visible_idx < VISIBLE_DIM:
            matrix[code_idx, visible_idx] = float(value)
    return matrix


def entries_to_string(entries: Sequence[tuple[int, int, float]]) -> str:
    return " ".join(f"{code}:{visible}:{value:g}" for code, visible, value in entries)


def matrix_to_entries(matrix: np.ndarray) -> tuple[tuple[int, int, float], ...]:
    rows: list[tuple[int, int, float]] = []
    for code_idx in range(matrix.shape[0]):
        for visible_idx in range(matrix.shape[1]):
            value = float(matrix[code_idx, visible_idx])
            if abs(value) > 1e-12:
                rows.append((code_idx, visible_idx, value))
    return tuple(rows)


def byte_label(byte: int) -> str:
    if 32 <= byte <= 126:
        return chr(byte)
    return f"0x{byte:02X}"


def csv_text(value: str) -> str:
    return value.replace("\n", "\\n").replace("\t", "\\t")


def load_json(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def candidate_from_json(path: Path, name: str, keys: Sequence[str], field: str) -> Candidate | None:
    blob = load_json(path)
    if blob is None:
        return None
    obj: object = blob
    for key in keys:
        if not isinstance(obj, dict) or key not in obj:
            return None
        obj = obj[key]
    if not isinstance(obj, dict) or field not in obj:
        return None
    return Candidate(name=name, source=str(path), entries=parse_entries(str(obj[field])))


def built_in_candidates(args: argparse.Namespace) -> list[Candidate]:
    candidates = [
        Candidate(
            name="A-StableCopy16",
            source="built-in redundant_copy_entries",
            entries=tuple(redundant_copy_entries(VISIBLE_DIM, CODE_DIM)),
        )
    ]
    d21g = candidate_from_json(Path(args.natural_sparse), "A-NaturalSparse16", ["best_natural_candidate"], "entries")
    d21i = candidate_from_json(Path(args.hidden_bit_gain), "A-HiddenBitGain16", ["best_candidate"], "effective_entries")
    d21j = candidate_from_json(Path(args.hidden_natural), "A-HiddenNatural16", ["best_candidate"], "effective_entries")
    candidates.extend([candidate for candidate in (d21g, d21i, d21j) if candidate is not None])
    return candidates


def byte_codes(block: ReciprocalABlock) -> np.ndarray:
    return np.stack([block.encode_byte(byte) for byte in range(256)], axis=0)


def euclidean_matrix(codes: np.ndarray) -> np.ndarray:
    diff = codes[:, None, :] - codes[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def cosine_matrix(codes: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(codes, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    normed = codes / norms
    return normed @ normed.T


def effective_rank(codes: np.ndarray) -> tuple[float, int, float]:
    centered = codes - codes.mean(axis=0, keepdims=True)
    _u, singular, _vt = np.linalg.svd(centered, full_matrices=False)
    if float(singular.sum()) <= 1e-12:
        return 0.0, 0, 0.0
    probs = singular / singular.sum()
    eff_rank = float(math.exp(-float(np.sum(probs * np.log(probs + 1e-12)))))
    energy = singular * singular
    cumulative = np.cumsum(energy) / max(float(energy.sum()), 1e-12)
    pca95 = int(np.searchsorted(cumulative, 0.95) + 1)
    top_energy = float(cumulative[0]) if cumulative.size else 0.0
    return eff_rank, pca95, top_energy


def intra_inter_group_metrics(codes: np.ndarray, group: Sequence[int], rest: Sequence[int]) -> tuple[float, float, float]:
    intra: list[float] = []
    for i, left in enumerate(group):
        for right in group[i + 1 :]:
            intra.append(code_distance(codes, left, right))
    inter = [code_distance(codes, left, right) for left in group for right in rest]
    intra_mean = float(np.mean(intra)) if intra else 0.0
    inter_mean = float(np.mean(inter)) if inter else 0.0
    separation = (inter_mean - intra_mean) / inter_mean if inter_mean > 1e-12 else 0.0
    return intra_mean, inter_mean, float(separation)


def summary_metrics(candidate: Candidate) -> tuple[dict[str, object], np.ndarray, np.ndarray, np.ndarray]:
    matrix = entries_to_matrix(candidate.entries)
    block = ReciprocalABlock(VISIBLE_DIM, CODE_DIM, matrix)
    codes = byte_codes(block)
    euclid = euclidean_matrix(codes)
    cosim = cosine_matrix(codes)
    base = evaluate_ablock(block, all_visible_patterns(VISIBLE_DIM))
    eff_rank, pca95, top_energy = effective_rank(codes)
    cos_no_diag = np.abs(cosim.copy())
    np.fill_diagonal(cos_no_diag, 0.0)
    row: dict[str, object] = {
        "name": candidate.name,
        "source": candidate.source,
        "entry_count": len(candidate.entries),
        "entries": entries_to_string(candidate.entries),
        "effective_rank": eff_rank,
        "pca95_dims": pca95,
        "top_pc_energy": top_energy,
        "avg_abs_cosine_overlap": float(cos_no_diag.mean()),
        "max_abs_cosine_overlap": float(cos_no_diag.max()),
        "ascii_neighbor_closer_rate": closer_rate(codes, ASCII_NEAR_PAIRS, ASCII_FAR_PAIRS),
        "case_pair_closer_rate": closer_rate(codes, CASE_NEAR_PAIRS, CASE_FAR_PAIRS),
        "digit_neighbor_closer_rate": closer_rate(codes, DIGIT_NEAR_PAIRS, DIGIT_FAR_PAIRS),
        "class_cluster_score": class_cluster_score(codes),
        "punct_separation_score": punct_separation_score(codes),
        "random_far_margin": random_far_margin(codes),
        "identity_copy_penalty": identity_copy_penalty(candidate.entries, VISIBLE_DIM, CODE_DIM),
        "duplicate_lane_penalty": duplicate_lane_penalty(block),
    }
    row.update(base)
    row["ascii_class_geometry"] = ascii_geometry_score(row)
    row["audit_score"] = audit_score(row)
    return row, codes, euclid, cosim


def audit_score(row: dict[str, object]) -> float:
    if (
        float(row["exact_byte_acc"]) < 1.0
        or float(row["bit_acc"]) < 1.0
        or int(row["hidden_collisions"]) != 0
        or float(row["byte_margin_min"]) <= 0.0
    ):
        return -100.0
    return float(
        20.0
        + 7.5 * float(row["ascii_class_geometry"])
        + 2.0 * max(0.0, float(row["class_cluster_score"]))
        + 1.0 * max(0.0, float(row["punct_separation_score"]))
        + 0.40 * min(4.0, float(row["byte_margin_min"]))
        + 0.20 * float(row["effective_rank"])
        - 1.0 * float(row["avg_abs_cosine_overlap"])
        - 2.0 * float(row["identity_copy_penalty"])
    )


def nearest_rows(name: str, euclid: np.ndarray, k: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for byte in range(256):
        order = np.argsort(euclid[byte])
        neighbors = [other for other in order if int(other) != byte][:k]
        rows.append(
            {
                "candidate": name,
                "byte": byte,
                "label": csv_text(byte_label(byte)),
                "nearest": " ".join(f"{byte_label(int(other))}:{euclid[byte, int(other)]:.3f}" for other in neighbors),
            }
        )
    return rows


def probe_rows(name: str, euclid: np.ndarray) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for byte in PROBE_BYTES:
        order = np.argsort(euclid[byte])
        neighbors = [other for other in order if int(other) != byte][:8]
        rows.append(
            {
                "candidate": name,
                "byte": byte,
                "label": csv_text(byte_label(byte)),
                "nearest": " ".join(f"{byte_label(int(other))}:{euclid[byte, int(other)]:.3f}" for other in neighbors),
                "dist_to_upper_A": f"{euclid[byte, ord('A')]:.6f}",
                "dist_to_lower_a": f"{euclid[byte, ord('a')]:.6f}",
                "dist_to_0": f"{euclid[byte, ord('0')]:.6f}",
                "dist_to_7": f"{euclid[byte, ord('7')]:.6f}",
            }
        )
    return rows


def group_rows(name: str, codes: np.ndarray) -> list[dict[str, object]]:
    all_bytes = list(range(256))
    rows: list[dict[str, object]] = []
    for group_name, group in GROUPS.items():
        rest = [byte for byte in all_bytes if byte not in set(group)]
        intra, inter, sep = intra_inter_group_metrics(codes, group, rest)
        rows.append(
            {
                "candidate": name,
                "group": group_name,
                "intra_mean_distance": f"{intra:.6f}",
                "inter_mean_distance": f"{inter:.6f}",
                "separation": f"{sep:.6f}",
            }
        )
    return rows


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_matrix_csv(path: Path, name: str, matrix: np.ndarray) -> None:
    rows: list[dict[str, object]] = []
    for byte in range(256):
        row: dict[str, object] = {
            "candidate": name,
            "byte": byte,
            "label": csv_text(byte_label(byte)),
        }
        for other in range(256):
            row[f"b{other:03d}"] = f"{matrix[byte, other]:.6f}"
        rows.append(row)
    write_csv(path, rows)


def heat_char(value: float, min_value: float, max_value: float) -> str:
    palette = " .:-=+*#%@"
    if max_value <= min_value + 1e-12:
        return palette[0]
    idx = int(round((value - min_value) / (max_value - min_value) * (len(palette) - 1)))
    return palette[max(0, min(len(palette) - 1, idx))]


def write_ascii_heatmap(path: Path, matrices: dict[str, np.ndarray]) -> None:
    lines = [
        "# A-GeometryAuditRevival ASCII Distance Heatmap",
        "",
        "Darker/brighter means farther apart inside that candidate's own scale.",
        "Rows/columns use selected ASCII groups, not all 256 bytes.",
        "",
    ]
    selected: list[int] = []
    for _label, group in ASCII_GRID:
        selected.extend(group)
    labels = [byte_label(byte) for byte in selected]
    for name, matrix in matrices.items():
        sub = matrix[np.ix_(selected, selected)]
        min_value = float(np.min(sub))
        max_value = float(np.max(sub))
        lines.extend([f"## {name}", "", "```text"])
        lines.append("    " + "".join(label[0] if label != " " else "_" for label in labels))
        for idx, byte in enumerate(selected):
            line = f"{byte_label(byte):>3} "
            line += "".join(heat_char(float(sub[idx, j]), min_value, max_value) for j in range(sub.shape[1]))
            lines.append(line)
        lines.extend(["```", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_report(
    path: Path,
    summary: Sequence[dict[str, object]],
    probe: Sequence[dict[str, object]],
    config: dict[str, object],
) -> None:
    ranked = sorted(summary, key=lambda row: float(row["audit_score"]), reverse=True)
    winner = ranked[0] if ranked else None
    lines = [
        "# A-GeometryAuditRevival",
        "",
        "Date: 2026-05-03",
        "",
        "This revives the old byte-embedding similarity-structure audit:",
        "roundtrip alone is not enough; the A16 space should also put related bytes closer together.",
        "",
        "## Verdict",
        "",
    ]
    if winner is None:
        lines.append("No candidates were evaluated.")
    else:
        lines.extend(
            [
                "```text",
                f"Winner by audit score: {winner['name']}",
                f"score={float(winner['audit_score']):.3f}",
                f"geometry={float(winner['ascii_class_geometry']):.3f}",
                f"margin={float(winner['byte_margin_min']):.3f}",
                "```",
            ]
        )
    lines.extend(
        [
            "",
            "## Candidate Summary",
            "",
            "```text",
            "name                exact margin geom  rank pca95 avgCos copy audit",
            "------------------- ----- ------ ----- ---- ----- ------ ---- ------",
        ]
    )
    for row in ranked:
        lines.append(
            f"{str(row['name'])[:19]:19} "
            f"{float(row['exact_byte_acc']):5.3f} "
            f"{float(row['byte_margin_min']):6.3f} "
            f"{float(row['ascii_class_geometry']):5.3f} "
            f"{float(row['effective_rank']):4.1f} "
            f"{int(row['pca95_dims']):5d} "
            f"{float(row['avg_abs_cosine_overlap']):6.3f} "
            f"{float(row['identity_copy_penalty']):4.2f} "
            f"{float(row['audit_score']):6.2f}"
        )
    lines.extend(["```", "", "## Probe Nearest Neighbors", "", "```text"])
    for row in probe:
        if str(row["label"]) in {"A", "a", "0", "7", ",", " "}:
            lines.append(f"{row['candidate']:19} {row['label']!r:>4} -> {row['nearest']}")
    lines.extend(
        [
            "```",
            "",
            "## Interpretation",
            "",
            "- Higher geometry means ASCII classes and near/far relations are better preserved.",
            "- Higher margin means safer exact byte decoding.",
            "- A good shipped candidate needs both: geometry and margin.",
            "- If a candidate wins geometry but loses margin, it is a research lead, not a default replacement.",
            "",
            "## Config",
            "",
            "```json",
            json.dumps(config, indent=2),
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    candidates = built_in_candidates(args)
    if not candidates:
        raise SystemExit("no candidates available")

    summary: list[dict[str, object]] = []
    nearest: list[dict[str, object]] = []
    probe: list[dict[str, object]] = []
    groups: list[dict[str, object]] = []
    matrices: dict[str, np.ndarray] = {}

    for candidate in candidates:
        row, codes, euclid, cosim = summary_metrics(candidate)
        summary.append(row)
        nearest.extend(nearest_rows(candidate.name, euclid, int(args.nearest_k)))
        probe.extend(probe_rows(candidate.name, euclid))
        groups.extend(group_rows(candidate.name, codes))
        matrices[candidate.name] = euclid
        write_matrix_csv(out / f"{candidate.name}_euclidean_distance_matrix.csv", candidate.name, euclid)
        write_matrix_csv(out / f"{candidate.name}_cosine_similarity_matrix.csv", candidate.name, cosim)

    summary = sorted(summary, key=lambda row: float(row["audit_score"]), reverse=True)
    write_csv(out / "a_geometry_audit_summary.csv", summary)
    write_csv(out / "a_nearest_neighbors.csv", nearest)
    write_csv(out / "a_probe_neighbors.csv", probe)
    write_csv(out / "a_group_cluster_scores.csv", groups)
    write_ascii_heatmap(out / "a_ascii_distance_heatmap.txt", matrices)

    config = {
        "mode": args.mode,
        "natural_sparse": str(args.natural_sparse),
        "hidden_bit_gain": str(args.hidden_bit_gain),
        "hidden_natural": str(args.hidden_natural),
        "nearest_k": int(args.nearest_k),
        "candidate_count": len(candidates),
    }
    write_report(out / "A_GEOMETRY_AUDIT_REVIVAL_REPORT.md", summary, probe, config)
    print(json.dumps({"out": str(out), "winner": summary[0]["name"], "summary": summary}, indent=2))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["smoke", "main"], default="main")
    parser.add_argument("--natural-sparse", default=str(DEFAULT_D21G))
    parser.add_argument("--hidden-bit-gain", default=str(DEFAULT_D21I))
    parser.add_argument("--hidden-natural", default=str(DEFAULT_D21J))
    parser.add_argument("--nearest-k", type=int, default=8)
    parser.add_argument("--out", default="output/phase_a_geometry_audit_revival_20260503/main")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
