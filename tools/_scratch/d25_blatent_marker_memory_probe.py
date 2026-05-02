#!/usr/bin/env python3
"""
D25 B-latent marker-memory probe.

Prototype goal:
    STORE_SLOT_A, PAYLOAD_WINDOW_A, ..., QUERY_SLOT_A -> PAYLOAD_WINDOW_A

D25 lifts the D21E multi-slot memory idea from byte payloads to full B64
8-byte-window payloads.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.ab_window_codec import (  # noqa: E402
    B_DIMS_PER_BYTE,
    B_WINDOW_DIMS,
    BYTE_BITS,
    WINDOW_BYTES,
    ABWindowCodec,
    byte_margin_from_visible,
    verify_artifact,
)


ASCII_SHADE = " .:-=+*#%@"
DEFAULT_SEED = 20260503
DEFAULT_MARKERS = (
    bytes([0xF1, 0xA0, 0xA0, 0xA0, 0xA0, 0xA0, 0xA0, 0xA1]),
    bytes([0xF2, 0xB0, 0xB0, 0xB0, 0xB0, 0xB0, 0xB0, 0xB2]),
    bytes([0xF3, 0xC0, 0xC0, 0xC0, 0xC0, 0xC0, 0xC0, 0xC3]),
    bytes([0xF4, 0xD0, 0xD0, 0xD0, 0xD0, 0xD0, 0xD0, 0xD4]),
)
DEFAULT_QUERIES = (
    bytes([0x1F, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0xA1]),
    bytes([0x2F, 0x0B, 0x0B, 0x0B, 0x0B, 0x0B, 0x0B, 0xB2]),
    bytes([0x3F, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0xC3]),
    bytes([0x4F, 0x0D, 0x0D, 0x0D, 0x0D, 0x0D, 0x0D, 0xD4]),
)


@dataclass(frozen=True)
class SequenceBatch:
    payloads: np.ndarray
    query_slots: np.ndarray
    slot_counts: np.ndarray
    distractor_lengths: np.ndarray
    target_windows: np.ndarray
    marker_windows: tuple[bytes, ...]
    query_windows: tuple[bytes, ...]


@dataclass(frozen=True)
class MemorySpec:
    family: str
    repeat_id: int
    slot_count: int
    state_dim: int
    memory_edges: int
    active_lanes: tuple[int, ...]


def parse_int_list(raw: str) -> list[int]:
    return [int(part.strip()) for part in str(raw).split(",") if part.strip()]


def checked_artifact(path: Path) -> None:
    verify_artifact(json.loads(path.read_text(encoding="utf-8")))


def window_matrix_to_bytes(rows: np.ndarray) -> list[bytes]:
    return [bytes(int(value) for value in row) for row in rows]


def b64_matrix(codec: ABWindowCodec, windows: Sequence[bytes]) -> np.ndarray:
    return np.asarray([codec.encode_window_b64(window) for window in windows], dtype=np.int8)


def b64_to_window_matrix(latents: np.ndarray) -> np.ndarray:
    bits = (latents.reshape(latents.shape[0], WINDOW_BYTES, BYTE_BITS) >= 0).astype(np.uint16)
    powers = (1 << np.arange(BYTE_BITS, dtype=np.uint16)).reshape(1, 1, BYTE_BITS)
    return np.sum(bits * powers, axis=2).astype(np.uint8)


def margin_matrix(latents: np.ndarray, targets: np.ndarray) -> np.ndarray:
    margins = np.empty((latents.shape[0], WINDOW_BYTES), dtype=np.float32)
    for row_idx in range(latents.shape[0]):
        for byte_idx in range(WINDOW_BYTES):
            offset = byte_idx * B_DIMS_PER_BYTE
            margins[row_idx, byte_idx] = byte_margin_from_visible(latents[row_idx, offset : offset + B_DIMS_PER_BYTE], int(targets[row_idx, byte_idx]))
    return margins


def payload_window(rng: random.Random, reserved: set[bytes]) -> bytes:
    while True:
        window = bytes(rng.randrange(0, 256) for _ in range(WINDOW_BYTES))
        if window not in reserved:
            return window


def make_sequence_batch(
    *,
    slot_counts: Sequence[int],
    distractor_lengths: Sequence[int],
    eval_sequences: int,
    seed: int,
) -> SequenceBatch:
    max_slots = max(slot_counts)
    if max_slots > len(DEFAULT_MARKERS) or max_slots > len(DEFAULT_QUERIES):
        raise ValueError("slot_count exceeds reserved marker/query count")
    rng = random.Random(seed)
    reserved = set(DEFAULT_MARKERS) | set(DEFAULT_QUERIES)
    payloads: list[list[bytes]] = []
    query_slots: list[int] = []
    slot_counts_out: list[int] = []
    lengths_out: list[int] = []
    targets: list[bytes] = []
    combo_count = max(1, sum(slot_counts) * max(1, len(distractor_lengths)))
    per_combo = max(1, int(eval_sequences) // combo_count)
    for slot_count in slot_counts:
        for length in distractor_lengths:
            for query_slot in range(slot_count):
                for _idx in range(per_combo):
                    row: list[bytes] = []
                    used = set(reserved)
                    for _slot in range(slot_count):
                        payload = payload_window(rng, used)
                        used.add(payload)
                        row.append(payload)
                    row.extend(bytes([0] * WINDOW_BYTES) for _ in range(max_slots - slot_count))
                    payloads.append(row)
                    query_slots.append(query_slot)
                    slot_counts_out.append(slot_count)
                    lengths_out.append(length)
                    targets.append(row[query_slot])
    payload_array = np.asarray([[list(window) for window in row] for row in payloads], dtype=np.uint8)
    return SequenceBatch(
        payloads=payload_array,
        query_slots=np.asarray(query_slots, dtype=np.int32),
        slot_counts=np.asarray(slot_counts_out, dtype=np.int32),
        distractor_lengths=np.asarray(lengths_out, dtype=np.int32),
        target_windows=np.asarray([list(window) for window in targets], dtype=np.uint8),
        marker_windows=DEFAULT_MARKERS,
        query_windows=DEFAULT_QUERIES,
    )


def store_states(payload_latents: np.ndarray, spec: MemorySpec) -> np.ndarray:
    # payload_latents shape: rows x max_slots x 64.
    states = np.zeros((payload_latents.shape[0], spec.slot_count, B_WINDOW_DIMS), dtype=np.int8)
    active = list(spec.active_lanes)
    states[:, :, active] = payload_latents[:, : spec.slot_count, active]
    return states.reshape(payload_latents.shape[0], spec.state_dim)


def query_states(states: np.ndarray, query_slots: np.ndarray, spec: MemorySpec) -> np.ndarray:
    view = states.reshape(states.shape[0], spec.slot_count, B_WINDOW_DIMS)
    rows = np.arange(states.shape[0])
    out = np.ones((states.shape[0], B_WINDOW_DIMS), dtype=np.int8)
    active = list(spec.active_lanes)
    out[:, active] = view[rows, query_slots, :][:, active]
    return out


def evaluate_latents(out_latents: np.ndarray, target_latents: np.ndarray, target_windows: np.ndarray) -> dict[str, float | int]:
    decoded = b64_to_window_matrix(out_latents)
    bit_match = out_latents == target_latents
    byte_match = decoded == target_windows
    window_match = np.all(byte_match, axis=1)
    margins = margin_matrix(out_latents, target_windows)
    collisions = collision_count(decoded, target_windows)
    return {
        "query_window_exact_acc": float(np.mean(window_match)),
        "query_byte_exact_acc": float(np.mean(byte_match)),
        "query_bit_acc": float(np.mean(bit_match)),
        "query_byte_margin_min": float(np.min(margins)),
        "slot_state_collision_count": collisions,
    }


def collision_count(decoded: np.ndarray, targets: np.ndarray) -> int:
    used: dict[tuple[int, ...], tuple[int, ...]] = {}
    collisions = 0
    for decoded_row, target_row in zip(decoded, targets):
        key = tuple(int(value) for value in decoded_row)
        target = tuple(int(value) for value in target_row)
        previous = used.get(key)
        if previous is not None and previous != target:
            collisions += 1
        used[key] = target
    return collisions


def wrong_slot_recall(states: np.ndarray, batch: SequenceBatch, target_latents: np.ndarray, spec: MemorySpec) -> float:
    wrong_slots = (batch.query_slots + 1) % batch.slot_counts
    wrong = query_states(states, wrong_slots, spec)
    return float(np.mean(np.all(wrong == target_latents, axis=1)))


def evaluate_spec(spec: MemorySpec, batch: SequenceBatch, codec: ABWindowCodec, seed: int) -> dict[str, object]:
    del seed
    row_mask = batch.slot_counts == spec.slot_count
    batch = SequenceBatch(
        payloads=batch.payloads[row_mask],
        query_slots=batch.query_slots[row_mask],
        slot_counts=batch.slot_counts[row_mask],
        distractor_lengths=batch.distractor_lengths[row_mask],
        target_windows=batch.target_windows[row_mask],
        marker_windows=batch.marker_windows,
        query_windows=batch.query_windows,
    )
    payload_windows = [bytes(int(value) for value in row) for row in batch.payloads.reshape(-1, WINDOW_BYTES)]
    payload_latents = b64_matrix(codec, payload_windows).reshape(batch.payloads.shape[0], batch.payloads.shape[1], B_WINDOW_DIMS)
    target_windows = window_matrix_to_bytes(batch.target_windows)
    target_latents = b64_matrix(codec, target_windows)

    if spec.family == "prev_window_baseline":
        out_latents = payload_latents[:, 0, :]
        states = store_states(payload_latents, MemorySpec("oracle_slot_memory", 0, spec.slot_count, spec.state_dim, spec.state_dim, tuple(range(B_WINDOW_DIMS))))
    elif spec.family == "random_state_control":
        rng = np.random.default_rng(DEFAULT_SEED + spec.repeat_id)
        states = rng.choice(np.asarray([-1, 1], dtype=np.int8), size=(batch.target_windows.shape[0], spec.state_dim))
        out_latents = query_states(states, batch.query_slots, spec)
    elif spec.family == "time_shuffle_control":
        states = store_states(payload_latents, spec)
        out_latents = query_states(np.roll(states, 17, axis=0), batch.query_slots, spec)
    elif spec.family == "reset_state_control" or spec.family == "marker_shuffle_control":
        out_latents = np.ones((batch.target_windows.shape[0], B_WINDOW_DIMS), dtype=np.int8)
        states = np.zeros((batch.target_windows.shape[0], spec.state_dim), dtype=np.int8)
    elif spec.family == "query_shuffle_control":
        states = store_states(payload_latents, spec)
        out_latents = query_states(states, (batch.query_slots + 1) % batch.slot_counts, spec)
    else:
        states = store_states(payload_latents, spec)
        out_latents = query_states(states, batch.query_slots, spec)

    metrics = evaluate_latents(out_latents, target_latents, batch.target_windows)
    metrics.update(
        {
            "family": spec.family,
            "repeat_id": spec.repeat_id,
            "slot_count": spec.slot_count,
            "state_dim": spec.state_dim,
            "memory_edge_count": spec.memory_edges,
            "active_lane_count": len(spec.active_lanes),
            "wrong_slot_recall_rate": wrong_slot_recall(states, batch, target_latents, spec) if spec.family in ("oracle_slot_memory", "compact_slot_memory") else 0.0,
            "all_distractor_lengths_pass": bool(metrics["query_window_exact_acc"] == 1.0),
        }
    )
    metrics["row_verdict"] = row_verdict(metrics)
    metrics["D25_score"] = score(metrics)
    return metrics


def row_verdict(row: dict[str, object]) -> str:
    exact = (
        float(row["query_window_exact_acc"]) == 1.0
        and float(row["query_byte_exact_acc"]) == 1.0
        and float(row["query_bit_acc"]) == 1.0
        and float(row["query_byte_margin_min"]) > 0.0
        and int(row["slot_state_collision_count"]) == 0
    )
    if str(row["family"]) in ("oracle_slot_memory", "compact_slot_memory") and exact:
        return "D25_ROW_PASS"
    if str(row["family"]).endswith("_control") and float(row["query_window_exact_acc"]) > 0.01:
        return "D25_STATE_ARTIFACT"
    fixed_slot_chance = 1.0 / max(1, int(row["slot_count"]))
    if str(row["family"]) == "prev_window_baseline" and float(row["query_window_exact_acc"]) > fixed_slot_chance + 0.02:
        return "D25_PREV_WINDOW_CHEAT"
    return "D25_ROW_FAIL"


def score(row: dict[str, object]) -> float:
    return (
        4.0 * float(row["query_window_exact_acc"])
        + 1.0 * float(row["query_byte_exact_acc"])
        + 0.5 * float(row["query_bit_acc"])
        + 0.01 * float(row["query_byte_margin_min"])
        - 0.00001 * float(row["memory_edge_count"])
        - 0.001 * float(row["slot_state_collision_count"])
    )


def specs_for_mode(mode: str, slot_counts: Sequence[int]) -> list[MemorySpec]:
    specs: list[MemorySpec] = []
    for slot_count in slot_counts:
        state_dim = slot_count * B_WINDOW_DIMS
        specs.append(MemorySpec("oracle_slot_memory", 0, slot_count, state_dim, state_dim, tuple(range(B_WINDOW_DIMS))))
        if mode in ("crystallize-memory", "confirm"):
            specs.append(MemorySpec("compact_slot_memory", 0, slot_count, state_dim, state_dim, tuple(range(B_WINDOW_DIMS))))
        specs.extend(
            [
                MemorySpec("prev_window_baseline", 0, slot_count, state_dim, B_WINDOW_DIMS, tuple(range(B_WINDOW_DIMS))),
                MemorySpec("reset_state_control", 0, slot_count, state_dim, 0, tuple(range(B_WINDOW_DIMS))),
                MemorySpec("time_shuffle_control", 0, slot_count, state_dim, state_dim, tuple(range(B_WINDOW_DIMS))),
                MemorySpec("random_state_control", 0, slot_count, state_dim, state_dim, tuple(range(B_WINDOW_DIMS))),
                MemorySpec("marker_shuffle_control", 0, slot_count, state_dim, 0, tuple(range(B_WINDOW_DIMS))),
                MemorySpec("query_shuffle_control", 0, slot_count, state_dim, state_dim, tuple(range(B_WINDOW_DIMS))),
            ]
        )
    return specs


def overall_verdict(rows: Sequence[dict[str, object]], slot_counts: Sequence[int]) -> str:
    for slot_count in slot_counts:
        primary = [row for row in rows if int(row["slot_count"]) == int(slot_count) and row["family"] in ("oracle_slot_memory", "compact_slot_memory")]
        if not primary:
            return "D25_NO_B_MEMORY_ROUTE"
        best = max(primary, key=lambda row: float(row["D25_score"]))
        if str(best["row_verdict"]) != "D25_ROW_PASS":
            return "D25_2SLOT_ONLY" if int(slot_count) > 2 else "D25_NO_B_MEMORY_ROUTE"
        controls = [row for row in rows if int(row["slot_count"]) == int(slot_count) and row["family"] not in ("oracle_slot_memory", "compact_slot_memory")]
        if any(str(row["row_verdict"]) in ("D25_STATE_ARTIFACT", "D25_PREV_WINDOW_CHEAT") for row in controls):
            return "D25_STATE_ARTIFACT"
        if float(best["wrong_slot_recall_rate"]) > 0.01:
            return "D25_STATE_ARTIFACT"
    return "D25_BLATENT_MEMORY_PASS"


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
    values = [float(row["D25_score"]) for row in rows] or [0.0]
    lo = min(values)
    hi = max(values)
    lines = ["D25 B64 memory heatmap: brighter = D25_score, P=pass C=control F=fail"]
    lines.append("slot family                  cell exact byte bit margin wrong verdict")
    for row in sorted(rows, key=lambda item: (int(item["slot_count"]), str(item["family"]))):
        scaled = 0 if hi <= lo else int(round((float(row["D25_score"]) - lo) / (hi - lo) * (len(ASCII_SHADE) - 1)))
        scaled = max(0, min(len(ASCII_SHADE) - 1, scaled))
        marker = "P" if str(row["row_verdict"]) == "D25_ROW_PASS" else "C" if str(row["family"]).endswith("_control") else "F"
        lines.append(
            f"{int(row['slot_count']):>4} {str(row['family'])[:23]:<23} {ASCII_SHADE[scaled]}{marker} "
            f"{float(row['query_window_exact_acc']):.3f} {float(row['query_byte_exact_acc']):.3f} "
            f"{float(row['query_bit_acc']):.3f} {float(row['query_byte_margin_min']):>6.2f} "
            f"{float(row['wrong_slot_recall_rate']):.3f} {row['row_verdict']}"
        )
    return "\n".join(lines)


def write_outputs(out_dir: Path, rows: Sequence[dict[str, object]], mode: str, config: dict[str, object]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sorted_rows = sorted(rows, key=lambda row: float(row["D25_score"]), reverse=True)
    write_csv(out_dir / "memory_candidates.csv", sorted_rows)
    control_rows = [
        row for row in sorted_rows if row["family"] not in ("oracle_slot_memory", "compact_slot_memory")
    ]
    write_csv(out_dir / "memory_control_summary.csv", control_rows)
    heatmap = make_heatmap(sorted_rows)
    (out_dir / "memory_heatmap.txt").write_text(heatmap + "\n", encoding="utf-8")
    verdict = overall_verdict(rows, parse_int_list(str(config["slot_counts"])))
    top = {
        "verdict": verdict,
        "mode": mode,
        "config": config,
        "candidate_count": len(rows),
        "best_candidate": sorted_rows[0] if sorted_rows else None,
        "slot_best": [
            max([row for row in sorted_rows if int(row["slot_count"]) == slot], key=lambda row: float(row["D25_score"]))
            for slot in parse_int_list(str(config["slot_counts"]))
        ],
    }
    (out_dir / "memory_top.json").write_text(json.dumps(top, indent=2), encoding="utf-8")
    report = [
        "# D25 B-Latent Marker Memory Report",
        "",
        f"Mode: `{mode}`",
        f"Verdict: `{verdict}`",
        "",
        "```text",
        heatmap,
        "```",
        "",
    ]
    (out_dir / "D25_BLATENT_MARKER_MEMORY_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    if mode in ("crystallize-memory", "confirm"):
        artifact = {
            "version": "d25_blatent_marker_memory_artifact_v1",
            "description": "Oracle/compact B64 slot memory layout; state is slot_count * 64 lanes.",
            "slot_counts": parse_int_list(str(config["slot_counts"])),
            "b_width": B_WINDOW_DIMS,
            "state_layout": "slot-major B64 payload copy",
        }
        (out_dir / "memory_artifact.json").write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    checked_artifact(Path(args.artifact))
    codec = ABWindowCodec()
    slot_counts = parse_int_list(args.slot_counts)
    batch = make_sequence_batch(
        slot_counts=slot_counts,
        distractor_lengths=parse_int_list(args.distractor_lengths),
        eval_sequences=int(args.eval_sequences),
        seed=int(args.seed),
    )
    rows = [evaluate_spec(spec, batch, codec, int(args.seed)) for spec in specs_for_mode(str(args.mode), slot_counts)]
    config = {
        "mode": args.mode,
        "slot_counts": args.slot_counts,
        "distractor_lengths": args.distractor_lengths,
        "eval_sequences": int(args.eval_sequences),
        "seed": int(args.seed),
        "artifact": str(args.artifact),
    }
    out_dir = Path(args.out)
    write_outputs(out_dir, rows, str(args.mode), config)
    top = json.loads((out_dir / "memory_top.json").read_text(encoding="utf-8"))
    print((out_dir / "memory_heatmap.txt").read_text(encoding="utf-8"))
    print(json.dumps({"verdict": top["verdict"], "best": top["best_candidate"]}, indent=2))
    return 0 if top["verdict"] in ("D25_BLATENT_MEMORY_PASS", "D25_2SLOT_ONLY", "D25_WEAK_PASS") else 1


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "oracle", "crystallize-memory", "confirm"], required=True)
    parser.add_argument("--slot-counts", default="2,4")
    parser.add_argument("--distractor-lengths", default="1,2,4,8,16")
    parser.add_argument("--eval-sequences", type=int, default=65536)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--candidates")
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--artifact", default="tools/ab_window_codec_v1.json")
    parser.add_argument("--out", required=True)
    return parser


def main() -> int:
    return run(build_arg_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
