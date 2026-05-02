#!/usr/bin/env python3
"""
D23 reciprocal B-block word/window abstraction probe.

Prototype goal:
    8 bytes -> D22 128D A-window code -> B latent -> reconstruct 8 bytes

D23 is not a language model. It tests whether the next reciprocal abstraction
layer can compress D22's robust byte-window surface while preserving exact
reconstruction and useful local geometry.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from d21a_reciprocal_byte_ablock import ReciprocalABlock, all_visible_patterns, redundant_copy_entries
from d22_byte_word_embedder import DEFAULT_WINDOW_BYTES, VISIBLE_DIM, BYTE_CODE_DIM, byte_margin, make_window_batch


DEFAULT_SEED = 20260502
ASCII_SHADE = " .:-=+*#%@"


@dataclass(frozen=True)
class BSpec:
    candidate_id: int
    family: str
    input_width: int
    latent_width: int
    mid_width: int
    encoder: np.ndarray
    encoder2: np.ndarray | None = None


def parse_int_list(raw: str) -> list[int]:
    return [int(part.strip()) for part in str(raw).split(",") if part.strip()]


def byte_block() -> ReciprocalABlock:
    return ReciprocalABlock.from_entries(VISIBLE_DIM, BYTE_CODE_DIM, redundant_copy_entries(VISIBLE_DIM, BYTE_CODE_DIM))


def encode_windows_128(windows: np.ndarray, patterns: np.ndarray) -> np.ndarray:
    block = byte_block()
    flat = patterns[windows.reshape(-1)]
    codes = block.encode_patterns(flat).reshape(windows.shape[0], windows.shape[1], BYTE_CODE_DIM)
    return codes.reshape(windows.shape[0], windows.shape[1] * BYTE_CODE_DIM)


def decode_a_window(code_128: np.ndarray, *, window_bytes: int, patterns: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    decoded_bits = code_128.reshape(code_128.shape[0], window_bytes, BYTE_CODE_DIM)[:, :, :VISIBLE_DIM]
    logits = decoded_bits @ patterns.T
    pred = np.argmax(logits, axis=2)
    pred_bits = np.where(decoded_bits >= 0.0, 1.0, -1.0)
    return pred, pred_bits, logits


def projection_encoder(input_width: int, latent_width: int) -> np.ndarray:
    encoder = np.zeros((latent_width, input_width), dtype=np.float32)
    for idx in range(min(input_width, latent_width)):
        encoder[idx, idx] = 1.0
    return encoder


def grouped_encoder(input_width: int, latent_width: int) -> np.ndarray:
    encoder = np.zeros((latent_width, input_width), dtype=np.float32)
    group = max(1, math.ceil(input_width / latent_width))
    for input_idx in range(input_width):
        latent_idx = min(latent_width - 1, input_idx // group)
        encoder[latent_idx, input_idx] = 1.0 / math.sqrt(group)
    return encoder


def random_orthogonal_encoder(input_width: int, latent_width: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(input_width, latent_width)).astype(np.float32)
    q, _ = np.linalg.qr(matrix)
    return q[:, :latent_width].T.astype(np.float32)


def block_average_encoder(input_width: int, latent_width: int, window_bytes: int) -> np.ndarray:
    encoder = np.zeros((latent_width, input_width), dtype=np.float32)
    per_byte = input_width // window_bytes
    keep_per_byte = max(1, latent_width // window_bytes)
    for byte_idx in range(window_bytes):
        in_start = byte_idx * per_byte
        out_start = byte_idx * keep_per_byte
        for out_local in range(min(keep_per_byte, per_byte)):
            if out_start + out_local < latent_width:
                encoder[out_start + out_local, in_start + out_local] = 1.0
    return encoder


def build_specs(input_width: int, latent_widths: Sequence[int], window_bytes: int, seed: int) -> list[BSpec]:
    specs: list[BSpec] = []
    for latent_width in latent_widths:
        if latent_width == input_width:
            specs.append(BSpec(len(specs), "B0_identity_128", input_width, latent_width, 0, np.eye(input_width, dtype=np.float32)))
        specs.append(BSpec(len(specs), "B0_projection", input_width, latent_width, 0, projection_encoder(input_width, latent_width)))
        specs.append(BSpec(len(specs), "B0_grouped", input_width, latent_width, 0, grouped_encoder(input_width, latent_width)))
        specs.append(BSpec(len(specs), "B0_block_average", input_width, latent_width, 0, block_average_encoder(input_width, latent_width, window_bytes)))
        specs.append(BSpec(len(specs), "B0_random_orthogonal", input_width, latent_width, 0, random_orthogonal_encoder(input_width, latent_width, seed + latent_width)))
    if 64 in latent_widths:
        first = projection_encoder(input_width, 96)
        second = projection_encoder(96, 64)
        specs.append(BSpec(len(specs), "B1_two_layer_mirror", input_width, 64, 96, first, second))
    return specs


def encode_b(spec: BSpec, code_128: np.ndarray) -> np.ndarray:
    first = code_128 @ spec.encoder.T
    if spec.encoder2 is None:
        return first
    return first @ spec.encoder2.T


def decode_b(spec: BSpec, latent: np.ndarray) -> np.ndarray:
    if spec.encoder2 is None:
        return latent @ spec.encoder
    mid = latent @ spec.encoder2
    return mid @ spec.encoder


def hamming_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(a != b, axis=1).astype(np.float32)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    denom = np.maximum(denom, 1e-8)
    return 1.0 - np.sum(a * b, axis=1) / denom


def geometry_metrics(
    spec: BSpec,
    latent: np.ndarray,
    windows: np.ndarray,
    patterns: np.ndarray,
    *,
    pairs: int,
    seed: int,
) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    n = windows.shape[0]
    if n < 4:
        return {
            "hamming_distance_correlation": 0.0,
            "one_byte_neighbor_closer_rate": 0.0,
            "prefix_neighbor_closer_rate": 0.0,
            "random_far_margin": 0.0,
        }
    idx_a = rng.integers(0, n, size=pairs)
    idx_b = rng.integers(0, n, size=pairs)
    ham = hamming_dist(windows[idx_a], windows[idx_b])
    dist = cosine_distance(latent[idx_a], latent[idx_b])
    corr = float(np.corrcoef(ham, dist)[0, 1]) if np.std(ham) > 0 and np.std(dist) > 0 else 0.0

    anchors = rng.integers(0, n, size=pairs)
    near = windows[anchors].copy()
    pos = rng.integers(0, windows.shape[1], size=pairs)
    delta = rng.integers(1, 256, size=pairs)
    near[np.arange(pairs), pos] = (near[np.arange(pairs), pos] + delta) % 256
    far_idx = rng.integers(0, n, size=pairs)
    near_latent = encode_b(spec, encode_windows_128(near, patterns))
    near_dist = cosine_distance(latent[anchors], near_latent)
    far_dist = cosine_distance(latent[anchors], latent[far_idx])
    one_byte_rate = float(np.mean(near_dist < far_dist))

    prefix = windows[anchors].copy()
    prefix[:, -1] = (prefix[:, -1] + delta) % 256
    prefix_latent = encode_b(spec, encode_windows_128(prefix, patterns))
    prefix_dist = cosine_distance(latent[anchors], prefix_latent)
    prefix_rate = float(np.mean(prefix_dist < far_dist))
    return {
        "hamming_distance_correlation": corr,
        "one_byte_neighbor_closer_rate": one_byte_rate,
        "prefix_neighbor_closer_rate": prefix_rate,
        "random_far_margin": float(np.mean(far_dist - near_dist)),
    }

def collision_count(latent: np.ndarray, windows: np.ndarray) -> int:
    used: dict[tuple[float, ...], tuple[int, ...]] = {}
    collisions: set[tuple[float, ...]] = set()
    for row, window in zip(latent, windows):
        key = tuple(np.round(row, 6).tolist())
        wkey = tuple(int(x) for x in window)
        prev = used.get(key)
        if prev is not None and prev != wkey:
            collisions.add(key)
        used[key] = wkey
    return len(collisions)


def control_metrics(spec: BSpec, code_128: np.ndarray, windows: np.ndarray, patterns: np.ndarray, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    shuffled = code_128.reshape(code_128.shape[0], windows.shape[1], BYTE_CODE_DIM).copy()
    shuffled = np.roll(shuffled, 1, axis=1).reshape(code_128.shape)
    sh_recon = decode_b(spec, encode_b(spec, shuffled))
    sh_pred, _bits, _logits = decode_a_window(sh_recon, window_bytes=windows.shape[1], patterns=patterns)

    random_code = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=code_128.shape)
    random_recon = decode_b(spec, encode_b(spec, random_code))
    random_pred, _rbits, _rlogits = decode_a_window(random_recon, window_bytes=windows.shape[1], patterns=patterns)

    random_encoder = random_orthogonal_encoder(code_128.shape[1], spec.latent_width, seed + 991)
    random_latent = code_128 @ random_encoder.T
    random_proj_recon = random_latent @ random_encoder
    random_proj_pred, _pbits, _plogits = decode_a_window(random_proj_recon, window_bytes=windows.shape[1], patterns=patterns)
    return {
        "position_shuffle_control": float(np.mean(np.all(sh_pred == windows, axis=1))),
        "random_code_control": float(np.mean(np.all(random_pred == windows, axis=1))),
        "random_projection_control": float(np.mean(np.all(random_proj_pred == windows, axis=1))),
    }


def evaluate_spec(spec: BSpec, windows: np.ndarray, code_128: np.ndarray, patterns: np.ndarray, geometry_pairs: int, seed: int) -> dict[str, object]:
    latent = encode_b(spec, code_128)
    recon = decode_b(spec, latent)
    pred, pred_bits, logits = decode_a_window(recon, window_bytes=windows.shape[1], patterns=patterns)
    target_bits = patterns[windows]
    margins = byte_margin(logits, windows)
    geom = geometry_metrics(spec, latent, windows, patterns, pairs=geometry_pairs, seed=seed + spec.candidate_id)
    controls = control_metrics(spec, code_128, windows, patterns, seed + spec.candidate_id)
    row: dict[str, object] = {
        "candidate_id": spec.candidate_id,
        "family": spec.family,
        "input_width": spec.input_width,
        "latent_width": spec.latent_width,
        "mid_width": spec.mid_width,
        "reciprocity_error": 0.0,
        "window_exact_acc": float(np.mean(np.all(pred == windows, axis=1))),
        "byte_exact_acc": float(np.mean(pred == windows)),
        "bit_acc": float(np.mean(pred_bits == target_bits)),
        "byte_margin_min": float(np.min(margins)),
        "byte_margin_mean": float(np.mean(margins)),
        "collision_count": collision_count(latent, windows),
        "latent_int8_bytes": spec.latent_width,
        "encoder_weight_count": int(np.count_nonzero(spec.encoder) + (0 if spec.encoder2 is None else np.count_nonzero(spec.encoder2))),
    }
    row.update(geom)
    row.update(controls)
    row["B_score"] = b_score(row)
    row["verdict"] = verdict(row)
    return row


def b_score(row: dict[str, object]) -> float:
    return (
        3.0 * float(row["window_exact_acc"])
        + 1.0 * float(row["byte_exact_acc"])
        + 0.5 * float(row["bit_acc"])
        + 0.5 * float(row["one_byte_neighbor_closer_rate"])
        + 0.25 * float(row["hamming_distance_correlation"])
        - 0.002 * float(row["collision_count"])
        - 0.0001 * float(row["latent_width"])
    )


def verdict(row: dict[str, object]) -> str:
    exact = float(row["window_exact_acc"]) == 1.0 and float(row["byte_exact_acc"]) == 1.0 and float(row["byte_margin_min"]) > 0.0
    geometry_ok = (
        int(row["collision_count"]) == 0
        and float(row["one_byte_neighbor_closer_rate"]) > 0.80
        and float(row["prefix_neighbor_closer_rate"]) > 0.80
        and float(row["random_far_margin"]) > 0.0
    )
    # A full-width random orthogonal projection is invertible, so it is not a
    # meaningful null for the 128D reference arm. Below 128D it must fail.
    random_projection_clean = (
        int(row["latent_width"]) >= int(row["input_width"])
        or float(row["random_projection_control"]) < 0.01
    )
    controls_clean = (
        float(row["position_shuffle_control"]) < 0.01
        and float(row["random_code_control"]) < 0.01
        and random_projection_clean
    )
    if exact and geometry_ok and controls_clean:
        if int(row["latent_width"]) == 64:
            return "D23_BBLOCK_64D_PASS"
        if int(row["latent_width"]) == 96:
            return "D23_BBLOCK_96D_PASS"
        if int(row["latent_width"]) == 128:
            return "D23_BBLOCK_128D_REFERENCE_PASS"
        return "D23_BBLOCK_COMPACT_PASS"
    if exact and not geometry_ok:
        return "D23_BBLOCK_RECON_ONLY"
    if int(row["latent_width"]) in (32, 48):
        return "D23_BBLOCK_TOO_COMPRESSED"
    return "D23_BBLOCK_FAIL"


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def make_heatmap(rows: Sequence[dict[str, object]]) -> str:
    values = [float(row["B_score"]) for row in rows] or [0.0]
    lo = min(values)
    hi = max(values)
    lines = ["D23 B-block heatmap: brighter = B_score, P=pass R=recon-only F=fail"]
    lines.append("latent family                  cell exact geom1 geomP corr verdict")
    for row in sorted(rows, key=lambda item: (int(item["latent_width"]), str(item["family"]))):
        scaled = 0 if hi <= lo else int(round((float(row["B_score"]) - lo) / (hi - lo) * (len(ASCII_SHADE) - 1)))
        scaled = max(0, min(len(ASCII_SHADE) - 1, scaled))
        marker = "P" if str(row["verdict"]).endswith("_PASS") else "R" if str(row["verdict"]) == "D23_BBLOCK_RECON_ONLY" else "F"
        lines.append(
            f"{int(row['latent_width']):>6} {str(row['family'])[:23]:<23} {ASCII_SHADE[scaled]}{marker} "
            f"{float(row['window_exact_acc']):.3f} {float(row['one_byte_neighbor_closer_rate']):.3f} "
            f"{float(row['prefix_neighbor_closer_rate']):.3f} {float(row['hamming_distance_correlation']):.3f} {row['verdict']}"
        )
    return "\n".join(lines)


def write_outputs(out_dir: Path, rows: Sequence[dict[str, object]], mode: str, config: dict[str, object]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sorted_rows = sorted(rows, key=lambda row: float(row["B_score"]), reverse=True)
    write_csv(out_dir / "bblock_candidates.csv", sorted_rows)
    geometry_rows = [
        {
            "candidate_id": row["candidate_id"],
            "family": row["family"],
            "latent_width": row["latent_width"],
            "hamming_distance_correlation": row["hamming_distance_correlation"],
            "one_byte_neighbor_closer_rate": row["one_byte_neighbor_closer_rate"],
            "prefix_neighbor_closer_rate": row["prefix_neighbor_closer_rate"],
            "random_far_margin": row["random_far_margin"],
            "collision_count": row["collision_count"],
            "verdict": row["verdict"],
        }
        for row in sorted_rows
    ]
    write_csv(out_dir / "bblock_geometry.csv", geometry_rows)
    pack_rows = [
        {
            "candidate_id": row["candidate_id"],
            "family": row["family"],
            "latent_width": row["latent_width"],
            "latent_int8_bytes": row["latent_int8_bytes"],
            "encoder_weight_count": row["encoder_weight_count"],
            "verdict": row["verdict"],
        }
        for row in sorted_rows
    ]
    write_csv(out_dir / "bblock_pack_summary.csv", pack_rows)
    heatmap = make_heatmap(sorted_rows)
    (out_dir / "bblock_heatmap.txt").write_text(heatmap + "\n", encoding="utf-8")
    pass_rows = [row for row in sorted_rows if str(row["verdict"]).endswith("_PASS")]
    recon_rows = [row for row in sorted_rows if str(row["verdict"]) == "D23_BBLOCK_RECON_ONLY"]
    if any(str(row["verdict"]) == "D23_BBLOCK_64D_PASS" for row in pass_rows):
        top_verdict = "D23_BBLOCK_64D_PASS"
    elif any(str(row["verdict"]) == "D23_BBLOCK_96D_PASS" for row in pass_rows):
        top_verdict = "D23_BBLOCK_96D_PASS"
    elif recon_rows:
        top_verdict = "D23_BBLOCK_RECON_ONLY"
    elif pass_rows:
        top_verdict = str(pass_rows[0]["verdict"])
    else:
        top_verdict = "D23_BBLOCK_FAIL"
    payload = {
        "verdict": top_verdict,
        "mode": mode,
        "config": config,
        "candidate_count": len(rows),
        "best_candidate": sorted_rows[0] if sorted_rows else None,
        "best_pass_candidate": pass_rows[0] if pass_rows else None,
        "best_recon_only_candidate": recon_rows[0] if recon_rows else None,
    }
    (out_dir / "bblock_top.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report = [
        "# D23 B-Block Word Abstraction Report",
        "",
        f"Mode: `{mode}`",
        f"Verdict: `{top_verdict}`",
        "",
        "## Heatmap",
        "",
        "```text",
        heatmap,
        "```",
        "",
    ]
    if sorted_rows:
        best = sorted_rows[0]
        report.extend(
            [
                "## Best Candidate",
                "",
                f"- family: `{best['family']}`",
                f"- latent_width: `{best['latent_width']}`",
                f"- window_exact_acc: `{float(best['window_exact_acc']):.6f}`",
                f"- byte_margin_min: `{float(best['byte_margin_min']):.6f}`",
                f"- hamming_distance_correlation: `{float(best['hamming_distance_correlation']):.6f}`",
                f"- one_byte_neighbor_closer_rate: `{float(best['one_byte_neighbor_closer_rate']):.6f}`",
                f"- prefix_neighbor_closer_rate: `{float(best['prefix_neighbor_closer_rate']):.6f}`",
                f"- collision_count: `{best['collision_count']}`",
                f"- verdict: `{best['verdict']}`",
                "",
            ]
        )
    (out_dir / "D23_BBLOCK_WORD_ABSTRACTION_REPORT.md").write_text("\n".join(report), encoding="utf-8")


def run_eval(args: argparse.Namespace, mode: str) -> int:
    patterns = all_visible_patterns(VISIBLE_DIM)
    windows = make_window_batch(window_bytes=int(args.window_bytes), eval_windows=int(args.eval_windows), seed=int(args.seed))
    code_128 = encode_windows_128(windows, patterns)
    specs = build_specs(code_128.shape[1], parse_int_list(args.latent_widths), int(args.window_bytes), int(args.seed))
    rows = [evaluate_spec(spec, windows, code_128, patterns, int(args.geometry_pairs), int(args.seed)) for spec in specs]
    write_outputs(
        Path(args.out),
        rows,
        mode,
        {
            "window_bytes": int(args.window_bytes),
            "latent_widths": parse_int_list(args.latent_widths),
            "eval_windows": int(args.eval_windows),
            "geometry_pairs": int(args.geometry_pairs),
            "seed": int(args.seed),
        },
    )
    heatmap = make_heatmap(rows)
    print(heatmap)
    top = json.loads((Path(args.out) / "bblock_top.json").read_text(encoding="utf-8"))
    print(json.dumps({"verdict": top["verdict"], "best": top["best_candidate"]}, indent=2))
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "width-sweep", "crystallize-pack"], required=True)
    parser.add_argument("--window-bytes", type=int, default=DEFAULT_WINDOW_BYTES)
    parser.add_argument("--latent-widths", default="32,48,64,96,128")
    parser.add_argument("--eval-windows", type=int, default=65536)
    parser.add_argument("--geometry-pairs", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", required=True)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    return run_eval(args, args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
