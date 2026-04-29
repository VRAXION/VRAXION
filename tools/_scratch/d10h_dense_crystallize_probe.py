#!/usr/bin/env python3
"""D10h dense -> crystallize microprobe.

Scratch/prototype only. The script intentionally reuses the D10g Torch
evaluator and keeps promotion/confirmation outside this path.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import struct
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

import d10g_gpu_eval_probe as gpu_eval


METRICS = ["smooth", "accuracy", "echo", "unigram"]


@dataclass
class EvalResult:
    smooth: float
    accuracy: float
    echo: float
    unigram: float

    def as_dict(self) -> dict[str, float]:
        return {
            "smooth": self.smooth,
            "accuracy": self.accuracy,
            "echo": self.echo,
            "unigram": self.unigram,
        }


def pack_usize(value: int) -> bytes:
    return struct.pack("<Q", int(value))


def pack_usize_vec(values: np.ndarray) -> bytes:
    arr = np.asarray(values, dtype="<u8")
    return pack_usize(len(arr)) + arr.tobytes()


def pack_u32_vec(values: np.ndarray) -> bytes:
    arr = np.asarray(values, dtype="<u4")
    return pack_usize(len(arr)) + arr.tobytes()


def pack_i32_vec(values: np.ndarray) -> bytes:
    arr = np.asarray(values, dtype="<i4")
    return pack_usize(len(arr)) + arr.tobytes()


def pack_i8_vec(values: np.ndarray) -> bytes:
    arr = np.asarray(values, dtype=np.int8).reshape(-1)
    return pack_usize(len(arr)) + arr.tobytes()


def pack_u8_vec(raw: bytes) -> bytes:
    return pack_usize(len(raw)) + raw


def pack_string(value: str) -> bytes:
    raw = value.encode("utf-8")
    return pack_usize(len(raw)) + raw


def serialize_network(net: gpu_eval.NetworkArrays) -> bytes:
    out = bytearray()
    out.append(1)
    out += pack_usize(net.h)
    out += pack_usize_vec(net.sources)
    out += pack_usize_vec(net.targets)
    out += pack_u32_vec(np.asarray(net.threshold, dtype=np.uint32))
    out += pack_u8_vec(np.asarray(net.channel, dtype=np.uint8).tobytes())
    out += pack_i32_vec(np.asarray(net.polarity, dtype=np.int32))
    return bytes(out)


def serialize_projection(proj: gpu_eval.ProjectionArrays) -> bytes:
    out = bytearray()
    out += pack_i8_vec(proj.weights)
    out += pack_usize(proj.input_dim)
    out += pack_usize(proj.output_classes)
    return bytes(out)


def write_checkpoint(path: Path, ckpt: gpu_eval.CheckpointArrays, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    network_bytes = serialize_network(ckpt.network)
    projection_bytes = serialize_projection(ckpt.projection)
    out = bytearray()
    out.append(1)
    out += pack_u8_vec(network_bytes)
    out += pack_u8_vec(projection_bytes)
    out += pack_usize(int(ckpt.meta.get("step", 0)))
    out += struct.pack("<d", float(ckpt.meta.get("accuracy", 0.0)))
    out += pack_string(label)
    path.write_bytes(bytes(out))


def clone_checkpoint(
    source: gpu_eval.CheckpointArrays,
    network: gpu_eval.NetworkArrays,
    label: str,
) -> gpu_eval.CheckpointArrays:
    return gpu_eval.CheckpointArrays(
        path=label,
        network=network,
        projection=source.projection,
        meta={"step": source.meta.get("step", 0), "accuracy": source.meta.get("accuracy", 0.0), "label": label},
    )


def clone_network(net: gpu_eval.NetworkArrays) -> gpu_eval.NetworkArrays:
    return gpu_eval.NetworkArrays(
        h=net.h,
        sources=net.sources.copy(),
        targets=net.targets.copy(),
        threshold=net.threshold.copy(),
        channel=net.channel.copy(),
        polarity=net.polarity.copy(),
    )


def edge_keys(net: gpu_eval.NetworkArrays) -> set[tuple[int, int]]:
    return set(zip(net.sources.astype(int).tolist(), net.targets.astype(int).tolist()))


def add_dense_fill(
    base: gpu_eval.NetworkArrays,
    density: float,
    rng: random.Random,
) -> tuple[gpu_eval.NetworkArrays, set[tuple[int, int]]]:
    h = int(base.h)
    existing = edge_keys(base)
    max_edges = h * (h - 1)
    target_edges = max(len(existing), int(round(max_edges * density)))
    missing_count = max(0, target_edges - len(existing))
    if missing_count == 0:
        return clone_network(base), set()

    added: set[tuple[int, int]] = set()
    attempts = 0
    max_attempts = missing_count * 50 + 10000
    while len(added) < missing_count and attempts < max_attempts:
        attempts += 1
        src = rng.randrange(h)
        dst = rng.randrange(h)
        if src == dst:
            continue
        key = (src, dst)
        if key in existing or key in added:
            continue
        added.add(key)

    if len(added) < missing_count:
        for src in range(h):
            for dst in range(h):
                if src == dst:
                    continue
                key = (src, dst)
                if key not in existing and key not in added:
                    added.add(key)
                    if len(added) >= missing_count:
                        break
            if len(added) >= missing_count:
                break

    added_sources = np.fromiter((s for s, _ in added), dtype=np.int64)
    added_targets = np.fromiter((t for _, t in added), dtype=np.int64)
    return (
        gpu_eval.NetworkArrays(
            h=base.h,
            sources=np.concatenate([base.sources, added_sources]),
            targets=np.concatenate([base.targets, added_targets]),
            threshold=base.threshold.copy(),
            channel=base.channel.copy(),
            polarity=base.polarity.copy(),
        ),
        added,
    )


def remove_edge_indices(net: gpu_eval.NetworkArrays, remove: set[int]) -> gpu_eval.NetworkArrays:
    if not remove:
        return clone_network(net)
    keep = np.ones(len(net.sources), dtype=bool)
    keep[list(remove)] = False
    return gpu_eval.NetworkArrays(
        h=net.h,
        sources=net.sources[keep].copy(),
        targets=net.targets[keep].copy(),
        threshold=net.threshold.copy(),
        channel=net.channel.copy(),
        polarity=net.polarity.copy(),
    )


def pruneable_indices(
    net: gpu_eval.NetworkArrays,
    protected: set[tuple[int, int]],
) -> list[int]:
    out = []
    for idx, key in enumerate(zip(net.sources.astype(int), net.targets.astype(int))):
        if key not in protected:
            out.append(idx)
    return out


def split_buckets(indices: list[int], bucket_count: int, rng: random.Random) -> list[list[int]]:
    shuffled = indices[:]
    rng.shuffle(shuffled)
    buckets = [[] for _ in range(bucket_count)]
    for pos, idx in enumerate(shuffled):
        buckets[pos % bucket_count].append(idx)
    return [b for b in buckets if b]


def load_inputs(args):
    reference = gpu_eval.load_checkpoint(Path(args.target))
    baseline = gpu_eval.load_checkpoint(Path(args.baseline))
    table = gpu_eval.load_vcbp(Path(args.packed))
    pair_ids, hot_to_idx, n_classes = gpu_eval.build_corpus_pairs(Path(args.corpus), table, gpu_eval.MAX_CLASSES)
    bigram, unigram = gpu_eval.build_bigram_unigram(pair_ids, hot_to_idx, n_classes)
    if reference.projection.output_classes != n_classes:
        raise ValueError(f"target projection classes {reference.projection.output_classes} != corpus classes {n_classes}")
    if baseline.projection.output_classes != n_classes:
        raise ValueError(f"baseline projection classes {baseline.projection.output_classes} != corpus classes {n_classes}")
    return reference, baseline, table, pair_ids, hot_to_idx, bigram, unigram


def evaluate_batch(
    checkpoints: list[gpu_eval.CheckpointArrays],
    table: gpu_eval.VcbpTablePy,
    pair_ids: np.ndarray,
    hot_to_idx: np.ndarray,
    bigram: np.ndarray,
    unigram: np.ndarray,
    eval_len: int,
    eval_seeds: list[int],
    device: torch.device,
) -> tuple[list[EvalResult], float]:
    metrics, elapsed = gpu_eval.evaluate_metrics(
        checkpoints,
        table,
        pair_ids,
        hot_to_idx,
        bigram,
        unigram,
        eval_len,
        eval_seeds,
        device,
    )
    rows = []
    for idx in range(len(checkpoints)):
        rows.append(
            EvalResult(
                smooth=float(metrics["smooth"][idx]),
                accuracy=float(metrics["accuracy"][idx]),
                echo=float(metrics["echo"][idx]),
                unigram=float(metrics["unigram"][idx]),
            )
        )
    return rows, elapsed


def delta(candidate: EvalResult, reference: EvalResult) -> EvalResult:
    return EvalResult(
        smooth=candidate.smooth - reference.smooth,
        accuracy=candidate.accuracy - reference.accuracy,
        echo=candidate.echo - reference.echo,
        unigram=candidate.unigram - reference.unigram,
    )


def mo_score(d: EvalResult) -> float:
    return d.smooth + 0.50 * d.accuracy + 1.50 * max(d.unigram, -0.0120) - 0.25 * abs(d.echo)


def classify_delta(d: EvalResult, args) -> str:
    if d.smooth < args.cliff_smooth or d.unigram < args.cliff_unigram or abs(d.echo) > args.echo_cliff:
        return "CLIFF"
    if d.smooth >= 0.0 and d.accuracy >= -0.001 and abs(d.echo) <= args.echo_safe and d.unigram >= -0.0025:
        return "RETAINED"
    if d.smooth >= -0.002 and d.accuracy >= -0.002 and d.unigram >= -0.004 and abs(d.echo) <= args.echo_cliff:
        return "NEAR_RETAINED"
    return "WEAK_OR_BAD"


def nearly_zero_delta(d: EvalResult, eps: float = 1e-9) -> bool:
    return (
        abs(d.smooth) <= eps
        and abs(d.accuracy) <= eps
        and abs(d.echo) <= eps
        and abs(d.unigram) <= eps
    )


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_csv(path: Path, fieldnames: list[str], row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def metric_row(prefix: dict, score: EvalResult, d: EvalResult, edge_count: int, extra: dict | None = None) -> dict:
    row = {
        **prefix,
        "edge_count": edge_count,
        "smooth_score": score.smooth,
        "accuracy_score": score.accuracy,
        "echo_score": score.echo,
        "unigram_score": score.unigram,
        "smooth_delta": d.smooth,
        "accuracy_delta": d.accuracy,
        "echo_delta": d.echo,
        "unigram_delta": d.unigram,
        "mo_score": mo_score(d),
    }
    if extra:
        row.update(extra)
    return row


def noise_floor(
    reference: gpu_eval.CheckpointArrays,
    table,
    pair_ids,
    hot_to_idx,
    bigram,
    unigram,
    args,
    device,
) -> dict:
    values = []
    for seed in parse_int_csv(args.noise_seeds):
        scores, _ = evaluate_batch([reference], table, pair_ids, hot_to_idx, bigram, unigram, args.eval_len, [seed], device)
        values.append(scores[0].as_dict())
    summary = {}
    for metric in METRICS:
        arr = np.asarray([v[metric] for v in values], dtype=np.float64)
        summary[metric] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            "sem95": float(1.96 * arr.std(ddof=1) / math.sqrt(arr.size)) if arr.size > 1 else 0.0,
        }
    return summary


def run_density(
    density: float,
    reference: gpu_eval.CheckpointArrays,
    table,
    pair_ids,
    hot_to_idx,
    bigram,
    unigram,
    args,
    device,
    out: Path,
) -> dict:
    rng = random.Random(args.seed + int(density * 10000))
    protected = edge_keys(reference.network) if args.protect_original else set()
    dense_net, added = add_dense_fill(reference.network, density, rng)
    current_net = dense_net
    random_net = clone_network(dense_net)
    seed_list = parse_int_csv(args.eval_seeds)

    round_fields = [
        "density",
        "track",
        "round",
        "edge_count",
        "pruneable_edges",
        "removed_edges",
        "class",
        "smooth_score",
        "accuracy_score",
        "echo_score",
        "unigram_score",
        "smooth_delta",
        "accuracy_delta",
        "echo_delta",
        "unigram_delta",
        "mo_score",
        "elapsed_s",
        "note",
    ]
    bucket_fields = [
        "density",
        "round",
        "bucket_id",
        "bucket_edges",
        "selected",
        "class",
        "smooth_delta",
        "accuracy_delta",
        "echo_delta",
        "unigram_delta",
        "mo_score",
        "delta_mo_vs_current",
    ]
    candidate_fields = [
        "density",
        "rank",
        "checkpoint",
        "round",
        "edge_count",
        "smooth_delta",
        "accuracy_delta",
        "echo_delta",
        "unigram_delta",
        "mo_score",
        "class",
    ]

    reference_eval, _ = evaluate_batch([reference], table, pair_ids, hot_to_idx, bigram, unigram, args.eval_len, seed_list, device)
    ref_score = reference_eval[0]
    start_ckpt = clone_checkpoint(reference, dense_net, f"dense_{density:.2f}")
    start_scores, start_elapsed = evaluate_batch([reference, start_ckpt], table, pair_ids, hot_to_idx, bigram, unigram, args.eval_len, seed_list, device)
    start_delta = delta(start_scores[1], start_scores[0])
    start_class = classify_delta(start_delta, args)
    append_csv(
        out / "round_progress.csv",
        round_fields,
        metric_row(
            {
                "density": density,
                "track": "ranked",
                "round": 0,
                "pruneable_edges": len(pruneable_indices(dense_net, protected)),
                "removed_edges": 0,
                "class": start_class,
                "elapsed_s": start_elapsed,
                "note": f"dense_start added_edges={len(added)}",
            },
            start_scores[1],
            start_delta,
            len(dense_net.sources),
        ),
    )
    print(
        f"D10h density={density:.2f} start class={start_class} "
        f"edges={len(dense_net.sources)} d=[{start_delta.smooth:.5f},{start_delta.accuracy:.5f},{start_delta.echo:.5f},{start_delta.unigram:.5f}]",
        flush=True,
    )
    if start_class == "CLIFF" and not args.continue_after_cliff_start:
        return {
            "density": density,
            "verdict": "DENSE_START_CLIFF",
            "added_edges": len(added),
            "final_edges": len(dense_net.sources),
            "best_mo_score": mo_score(start_delta),
            "best_class": start_class,
        }

    candidates = []
    best = {"round": 0, "net": clone_network(current_net), "delta": start_delta, "score": start_scores[1], "class": start_class}
    flat_or_bad = 0
    for round_idx in range(1, args.rounds + 1):
        current_ckpt = clone_checkpoint(reference, current_net, f"dense_{density:.2f}_round_{round_idx}_current")
        current_scores, _ = evaluate_batch([reference, current_ckpt], table, pair_ids, hot_to_idx, bigram, unigram, args.eval_len, seed_list, device)
        current_delta = delta(current_scores[1], current_scores[0])
        current_mo = mo_score(current_delta)
        pidx = pruneable_indices(current_net, protected)
        if len(pidx) < args.buckets:
            break
        buckets = split_buckets(pidx, args.buckets, random.Random(args.seed + round_idx + int(density * 10000)))
        proposal_ckpts = [reference]
        proposal_meta = []
        for bucket_id, bucket in enumerate(buckets):
            proposal_net = remove_edge_indices(current_net, set(bucket))
            proposal_ckpts.append(clone_checkpoint(reference, proposal_net, f"dense_{density:.2f}_r{round_idx}_b{bucket_id}"))
            proposal_meta.append((bucket_id, bucket))
        proposal_scores, elapsed = evaluate_batch(
            proposal_ckpts,
            table,
            pair_ids,
            hot_to_idx,
            bigram,
            unigram,
            args.eval_len,
            seed_list,
            device,
        )
        ranked = []
        for idx, (bucket_id, bucket) in enumerate(proposal_meta, start=1):
            d = delta(proposal_scores[idx], proposal_scores[0])
            klass = classify_delta(d, args)
            ranked.append(
                {
                    "bucket_id": bucket_id,
                    "bucket": bucket,
                    "delta": d,
                    "class": klass,
                    "score": proposal_scores[idx],
                    "mo": mo_score(d),
                    "delta_mo_vs_current": mo_score(d) - current_mo,
                }
            )
        ranked.sort(key=lambda r: (r["class"] in ("RETAINED", "NEAR_RETAINED"), r["delta_mo_vs_current"], r["mo"]), reverse=True)
        target_remove = max(1, int(round(len(pidx) * args.prune_fraction)))
        selected = []
        selected_edges = 0
        for row in ranked:
            if args.accept_relative_current:
                acceptable = (
                    row["delta_mo_vs_current"] >= args.min_current_mo_improve
                    and row["delta"].smooth >= current_delta.smooth - args.max_smooth_backtrack
                    and row["delta"].accuracy >= current_delta.accuracy - args.max_accuracy_backtrack
                    and row["delta"].unigram >= current_delta.unigram - args.max_unigram_backtrack
                    and abs(row["delta"].echo) <= max(args.echo_cliff, abs(current_delta.echo) + args.max_echo_backtrack)
                )
            else:
                acceptable = (
                    row["class"] in ("RETAINED", "NEAR_RETAINED")
                    and row["delta"].smooth >= current_delta.smooth - args.max_smooth_backtrack
                    and row["delta"].accuracy >= current_delta.accuracy - 0.0010
                    and row["delta"].unigram >= current_delta.unigram - 0.0025
                )
            if acceptable:
                selected.append(row)
                selected_edges += len(row["bucket"])
            if selected_edges >= target_remove:
                break
        for row in ranked:
            append_csv(
                out / "bucket_scores.csv",
                bucket_fields,
                {
                    "density": density,
                    "round": round_idx,
                    "bucket_id": row["bucket_id"],
                    "bucket_edges": len(row["bucket"]),
                    "selected": row in selected,
                    "class": row["class"],
                    "smooth_delta": row["delta"].smooth,
                    "accuracy_delta": row["delta"].accuracy,
                    "echo_delta": row["delta"].echo,
                    "unigram_delta": row["delta"].unigram,
                    "mo_score": row["mo"],
                    "delta_mo_vs_current": row["delta_mo_vs_current"],
                },
            )
        removed = set()
        for row in selected:
            removed.update(row["bucket"])
        if removed:
            current_net = remove_edge_indices(current_net, removed)
        current_ckpt = clone_checkpoint(reference, current_net, f"dense_{density:.2f}_round_{round_idx}")
        final_scores, final_elapsed = evaluate_batch([reference, current_ckpt], table, pair_ids, hot_to_idx, bigram, unigram, args.eval_len, seed_list, device)
        final_delta = delta(final_scores[1], final_scores[0])
        final_class = classify_delta(final_delta, args)
        append_csv(
            out / "round_progress.csv",
            round_fields,
            metric_row(
                {
                    "density": density,
                    "track": "ranked",
                    "round": round_idx,
                    "pruneable_edges": len(pruneable_indices(current_net, protected)),
                    "removed_edges": len(removed),
                    "class": final_class,
                    "elapsed_s": elapsed + final_elapsed,
                    "note": "ranked_prune",
                },
                final_scores[1],
                final_delta,
                len(current_net.sources),
            ),
        )

        random_pidx = pruneable_indices(random_net, protected)
        random_remove_count = min(len(random_pidx), max(1, len(removed)))
        random_removed = set(rng.sample(random_pidx, random_remove_count)) if random_remove_count else set()
        random_net = remove_edge_indices(random_net, random_removed)
        random_ckpt = clone_checkpoint(reference, random_net, f"dense_{density:.2f}_random_round_{round_idx}")
        random_scores, random_elapsed = evaluate_batch([reference, random_ckpt], table, pair_ids, hot_to_idx, bigram, unigram, args.eval_len, seed_list, device)
        random_delta = delta(random_scores[1], random_scores[0])
        append_csv(
            out / "round_progress.csv",
            round_fields,
            metric_row(
                {
                    "density": density,
                    "track": "random_control",
                    "round": round_idx,
                    "pruneable_edges": len(pruneable_indices(random_net, protected)),
                    "removed_edges": len(random_removed),
                    "class": classify_delta(random_delta, args),
                    "elapsed_s": random_elapsed,
                    "note": "same_remove_count_random",
                },
                random_scores[1],
                random_delta,
                len(random_net.sources),
            ),
        )

        print(
            f"D10h density={density:.2f} round={round_idx} removed={len(removed)} "
            f"edges={len(current_net.sources)} class={final_class} "
            f"d=[{final_delta.smooth:.5f},{final_delta.accuracy:.5f},{final_delta.echo:.5f},{final_delta.unigram:.5f}]",
            flush=True,
        )
        if mo_score(final_delta) >= mo_score(best["delta"]):
            best = {"round": round_idx, "net": clone_network(current_net), "delta": final_delta, "score": final_scores[1], "class": final_class}
            ckpt_path = out / "candidates" / f"density_{int(density * 100):02d}_round_{round_idx:02d}.ckpt"
            write_checkpoint(ckpt_path, clone_checkpoint(reference, current_net, str(ckpt_path)), f"d10h_density_{density:.2f}_round_{round_idx}")
            candidates.append(
                {
                    "density": density,
                    "rank": 0,
                    "checkpoint": str(ckpt_path),
                    "round": round_idx,
                    "edge_count": len(current_net.sources),
                    "smooth_delta": final_delta.smooth,
                    "accuracy_delta": final_delta.accuracy,
                    "echo_delta": final_delta.echo,
                    "unigram_delta": final_delta.unigram,
                    "mo_score": mo_score(final_delta),
                    "class": final_class,
                }
            )
            flat_or_bad = 0
        else:
            flat_or_bad += 1
        if not removed:
            flat_or_bad += 1
        if flat_or_bad >= args.early_stop_rounds:
            break

    candidates.sort(key=lambda r: (r["class"] == "RETAINED", r["mo_score"]), reverse=True)
    for idx, row in enumerate(candidates, start=1):
        row["rank"] = idx
    if candidates:
        write_csv(out / "crystallize_candidates.csv", candidate_fields, candidates)
    edge_reduction = 1.0 - (len(best["net"].sources) / max(1, len(dense_net.sources)))
    returned_to_reference = len(best["net"].sources) <= len(reference.network.sources) and nearly_zero_delta(best["delta"])
    if returned_to_reference:
        verdict = "D10H_RETURNED_TO_REFERENCE_ONLY"
    elif best["class"] == "RETAINED" and best["delta"].smooth > args.min_claim_smooth_gain and best["delta"].unigram >= -0.0025:
        verdict = "D10H_CRYSTALLIZE_SIGNAL"
    elif edge_reduction >= 0.30 and best["class"] in ("RETAINED", "NEAR_RETAINED"):
        verdict = "D10H_EDGE_EFFICIENCY_WIN"
    elif flat_or_bad >= args.early_stop_rounds:
        verdict = "D10H_NO_SHORT_CRYSTALLIZE_SIGNAL"
    else:
        verdict = "D10H_PRUNE_NOISE_DOMINATED"
    return {
        "density": density,
        "verdict": verdict,
        "added_edges": len(added),
        "start_edges": len(dense_net.sources),
        "best_round": best["round"],
        "best_edges": len(best["net"].sources),
        "edge_reduction": edge_reduction,
        "best_class": best["class"],
        "best_delta": best["delta"].as_dict(),
        "best_mo_score": mo_score(best["delta"]),
        "candidate_count": len(candidates),
    }


def parse_int_csv(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_float_csv(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def write_report(out: Path, summary: dict) -> None:
    lines = [
        "# D10h Dense Crystallize Progress",
        "",
        f"Verdict: `{summary['verdict']}`",
        "",
        "## Summary",
        "",
        f"- Device: `{summary['device']}`",
        f"- Eval length: `{summary['eval_len']}`",
        f"- Eval seeds: `{summary['eval_seeds']}`",
        f"- Runtime seconds: `{summary['elapsed_s']:.1f}`",
        "",
        "## Density Results",
        "",
        "| density | verdict | start edges | best edges | edge reduction | best class | best smooth | best accuracy | best echo | best unigram |",
        "|---:|---|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in summary["density_results"]:
        d = row.get("best_delta", {})
        lines.append(
            f"| {row['density']:.2f} | `{row['verdict']}` | {row.get('start_edges', 0)} | {row.get('best_edges', 0)} | "
            f"{row.get('edge_reduction', 0.0):.3f} | `{row.get('best_class', '')}` | "
            f"{d.get('smooth', 0.0):.6f} | {d.get('accuracy', 0.0):.6f} | {d.get('echo', 0.0):.6f} | {d.get('unigram', 0.0):.6f} |"
        )
    lines += [
        "",
        "## Progress Map",
        "",
        "```text",
        "GLOBAL AI PLAN MAP",
        "",
        "[1] beta.8 generalist found",
        "    DONE",
        "",
        "[2] mechanism explained",
        "    DONE: edge + threshold co-adaptation",
        "",
        "[3] seed replication",
        "    RUNNING: D10b CPU main",
        "",
        "[3.5] GPU evaluator",
        "    DONE: useful for batched scout",
        "",
        "[4] dense -> crystallize/prune",
        "    CURRENT: D10h short microprobe",
        "",
        "[5] H512/H1024 scaling",
        "    blocked until D10b or D10h gives repeatable signal",
        "```",
    ]
    (out / "D10H_DENSE_CRYSTALLIZE_PROGRESS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def choose_verdict(results: list[dict]) -> str:
    verdicts = {r["verdict"] for r in results}
    if "D10H_CRYSTALLIZE_SIGNAL" in verdicts:
        return "D10H_CRYSTALLIZE_SIGNAL"
    if "D10H_EDGE_EFFICIENCY_WIN" in verdicts:
        return "D10H_EDGE_EFFICIENCY_WIN"
    if "D10H_RETURNED_TO_REFERENCE_ONLY" in verdicts:
        return "D10H_RETURNED_TO_REFERENCE_ONLY"
    if verdicts == {"DENSE_START_CLIFF"}:
        return "D10H_DENSE_START_TOO_CLIFFY"
    if "D10H_PRUNE_NOISE_DOMINATED" in verdicts:
        return "D10H_PRUNE_NOISE_DOMINATED"
    return "D10H_NO_SHORT_CRYSTALLIZE_SIGNAL"


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", default="output/phase_d7_operator_bandit_20260427/H_384/D7_BASELINE/seed_2042/final.ckpt")
    p.add_argument("--target", default="output/releases/v5.0.0-beta.8/seed2042_improved_generalist_v1.ckpt")
    p.add_argument("--packed", default="output/block_c_bytepair_champion/packed.bin")
    p.add_argument("--corpus", default="instnct-core/tests/fixtures/alice_corpus.txt")
    p.add_argument("--out", default="output/phase_d10h_dense_crystallize_probe_20260429")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--eval-len", type=int, default=1000)
    p.add_argument("--eval-seeds", default="981001,981002,981003,981004")
    p.add_argument("--noise-seeds", default="981001,981002,981003,981004,981005,981006,981007,981008")
    p.add_argument("--densities", default="0.10,0.25")
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--buckets", type=int, default=64)
    p.add_argument("--prune-fraction", type=float, default=0.075)
    p.add_argument("--early-stop-rounds", type=int, default=3)
    p.add_argument("--seed", type=int, default=101010)
    p.add_argument("--protect-original", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--cliff-smooth", type=float, default=-0.020)
    p.add_argument("--cliff-unigram", type=float, default=-0.020)
    p.add_argument("--echo-cliff", type=float, default=0.010)
    p.add_argument("--echo-safe", type=float, default=0.0015)
    p.add_argument("--max-smooth-backtrack", type=float, default=0.0010)
    p.add_argument("--continue-after-cliff-start", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--accept-relative-current", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--min-current-mo-improve", type=float, default=0.00005)
    p.add_argument("--min-claim-smooth-gain", type=float, default=0.00025)
    p.add_argument("--max-accuracy-backtrack", type=float, default=0.0010)
    p.add_argument("--max-unigram-backtrack", type=float, default=0.0025)
    p.add_argument("--max-echo-backtrack", type=float, default=0.0025)
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    device = torch.device(args.device)
    reference, baseline, table, pair_ids, hot_to_idx, bigram, unigram = load_inputs(args)
    noise = noise_floor(reference, table, pair_ids, hot_to_idx, bigram, unigram, args, device)
    (out / "noise_floor.json").write_text(json.dumps(noise, indent=2), encoding="utf-8")
    print(f"D10h noise smooth sem95={noise['smooth']['sem95']:.6f} std={noise['smooth']['std']:.6f}", flush=True)
    results = []
    for density in parse_float_csv(args.densities):
        results.append(run_density(density, reference, table, pair_ids, hot_to_idx, bigram, unigram, args, device, out))
    summary = {
        "verdict": choose_verdict(results),
        "device": args.device,
        "eval_len": args.eval_len,
        "eval_seeds": parse_int_csv(args.eval_seeds),
        "noise_floor": noise,
        "density_results": results,
        "elapsed_s": time.perf_counter() - started,
        "note": "GPU/Torch scout only; no promotion without Rust CPU confirm.",
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(out, summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
