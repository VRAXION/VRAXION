#!/usr/bin/env python3
"""D10j sparse high-H GPU feasibility probe.

Scratch/prototype only. This answers whether a low-usage edge-list evaluator
can support H512/H1024 scout workloads while D10b runs on CPU.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

import d10g_gpu_eval_probe as gpu_eval


METRICS = ["smooth", "accuracy", "echo", "unigram"]


@dataclass
class SparseState:
    h: int
    output_start: int
    threshold: torch.Tensor
    channel: torch.Tensor
    polarity: torch.Tensor
    weights: torch.Tensor
    sources: torch.Tensor
    targets: torch.Tensor


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_int_csv(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def build_sparse_state(checkpoints: list[gpu_eval.CheckpointArrays], device: torch.device) -> SparseState:
    h = checkpoints[0].network.h
    output_start = h - gpu_eval.phi_dim(h)
    thresholds = torch.as_tensor(np.stack([c.network.threshold for c in checkpoints]), dtype=torch.int16, device=device)
    channels = torch.as_tensor(np.stack([c.network.channel for c in checkpoints]), dtype=torch.int16, device=device)
    polarity = torch.as_tensor(np.stack([c.network.polarity for c in checkpoints]), dtype=torch.int16, device=device)
    weights = torch.as_tensor(np.stack([c.projection.weights for c in checkpoints]), dtype=torch.int16, device=device)
    max_edges = max(len(c.network.sources) for c in checkpoints)
    sources = torch.full((len(checkpoints), max_edges), -1, dtype=torch.long, device=device)
    targets = torch.full((len(checkpoints), max_edges), 0, dtype=torch.long, device=device)
    for idx, c in enumerate(checkpoints):
        n = len(c.network.sources)
        sources[idx, :n] = torch.as_tensor(c.network.sources, dtype=torch.long, device=device)
        targets[idx, :n] = torch.as_tensor(c.network.targets, dtype=torch.long, device=device)
    return SparseState(h, output_start, thresholds, channels, polarity, weights, sources, targets)


def propagate_sequence_sparse(
    state: SparseState,
    input_rows: torch.Tensor,
    ticks_per_token: int = 6,
    input_duration_ticks: int = 2,
    decay_interval_ticks: int = 6,
) -> torch.Tensor:
    b = state.threshold.shape[0]
    h = state.h
    charge = torch.zeros((b, h), dtype=torch.int16, device=input_rows.device)
    activation = torch.zeros((b, h), dtype=torch.int16, device=input_rows.device)
    phase_base = gpu_eval.PHASE_BASE.to(input_rows.device)
    output_charges = []
    edge_valid = state.sources >= 0
    safe_sources = torch.clamp(state.sources, min=0)
    batch_offsets = (torch.arange(b, device=input_rows.device, dtype=torch.long) * h).view(b, 1)
    flat_targets = (state.targets + batch_offsets).reshape(-1)
    flat_valid = edge_valid.reshape(-1)
    flat_targets_valid = flat_targets[flat_valid]
    for token_idx in range(input_rows.shape[0]):
        inp = input_rows[token_idx].unsqueeze(0).expand(b, -1)
        for tick in range(ticks_per_token):
            if decay_interval_ticks > 0 and tick % decay_interval_ticks == 0:
                charge = torch.clamp(charge - 1, min=0)
            if tick < input_duration_ticks:
                activation = torch.clamp(activation + inp, min=-128, max=127)
            edge_values = torch.gather(activation, 1, safe_sources)
            flat_values = edge_values.reshape(-1)[flat_valid].to(torch.float32)
            incoming_flat = torch.zeros(b * h, dtype=torch.float32, device=input_rows.device)
            incoming_flat.scatter_add_(0, flat_targets_valid, flat_values)
            incoming = incoming_flat.view(b, h).round().to(torch.int16)
            charge = torch.clamp(charge + incoming, min=0, max=gpu_eval.MAX_CHARGE)
            activation = torch.zeros_like(activation)
            phase_indices = (tick % 8 + 9 - state.channel) & 7
            phase_mult = phase_base[phase_indices.to(torch.long)]
            should_fire = (charge * 10) >= ((state.threshold + 1) * phase_mult)
            activation = torch.where(should_fire, state.polarity, activation)
            charge = torch.where(should_fire, torch.zeros_like(charge), charge)
        output_charges.append(charge[:, state.output_start :])
    return torch.stack(output_charges, dim=1)


def evaluate_metrics_sparse(
    checkpoints: list[gpu_eval.CheckpointArrays],
    table: gpu_eval.VcbpTablePy,
    pair_ids: np.ndarray,
    hot_to_idx: np.ndarray,
    bigram: np.ndarray,
    unigram: np.ndarray,
    eval_len: int,
    eval_seeds: list[int],
    device: torch.device,
) -> tuple[dict[str, list[float]], float, float]:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    state = build_sparse_state(checkpoints, device)
    b = len(checkpoints)
    acc_total = torch.zeros(b, dtype=torch.float64, device=device)
    echo_total = torch.zeros(b, dtype=torch.float64, device=device)
    smooth_total = torch.zeros(b, dtype=torch.float64, device=device)
    unigram_total = torch.zeros(b, dtype=torch.float64, device=device)
    for seed in eval_seeds:
        off = gpu_eval.deterministic_offset(seed, len(pair_ids), eval_len)
        cur_ids = pair_ids[off : off + eval_len]
        next_ids = pair_ids[off + 1 : off + eval_len + 1]
        inputs = torch.as_tensor(
            gpu_eval.quantized_inputs(cur_ids, table, state.h, input_scatter=False),
            dtype=torch.int16,
            device=device,
        )
        outputs = propagate_sequence_sparse(state, inputs)
        scores = gpu_eval.projection_scores(outputs, state.weights)
        probs = torch.softmax(scores, dim=-1)
        target_idx = torch.as_tensor(hot_to_idx[next_ids], dtype=torch.long, device=device)
        cur_idx = torch.as_tensor(hot_to_idx[cur_ids], dtype=torch.long, device=device)
        preds = scores.argmax(dim=-1)
        valid_target = target_idx >= 0
        acc_total += ((preds == target_idx.unsqueeze(0)) & valid_target.unsqueeze(0)).sum(dim=1).to(torch.float64) / eval_len
        valid_cur = cur_idx >= 0
        if valid_cur.any():
            idxs = cur_idx[valid_cur]
            one_hot = torch.nn.functional.one_hot(idxs, num_classes=probs.shape[-1]).to(torch.float32).to(device)
            echo_total += gpu_eval.cosine(probs[:, valid_cur, :], one_hot.unsqueeze(0)).sum(dim=1).to(torch.float64) / int(valid_cur.sum())
            bigram_t = torch.as_tensor(bigram[idxs.cpu().numpy()], dtype=torch.float32, device=device)
            smooth_total += gpu_eval.cosine(probs[:, valid_cur, :], bigram_t.unsqueeze(0)).sum(dim=1).to(torch.float64) / int(valid_cur.sum())
        unigram_t = torch.as_tensor(unigram, dtype=torch.float32, device=device)
        unigram_total += gpu_eval.cosine(probs, unigram_t.view(1, 1, -1)).sum(dim=1).to(torch.float64) / eval_len
    denom = float(len(eval_seeds))
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if device.type == "cuda" else 0.0
    return (
        {
            "accuracy": (acc_total / denom).detach().cpu().tolist(),
            "echo": (echo_total / denom).detach().cpu().tolist(),
            "smooth": (smooth_total / denom).detach().cpu().tolist(),
            "unigram": (unigram_total / denom).detach().cpu().tolist(),
        },
        elapsed,
        peak_mb,
    )


def load_real_inputs(args):
    table = gpu_eval.load_vcbp(Path(args.packed))
    pair_ids, hot_to_idx, n_classes = gpu_eval.build_corpus_pairs(Path(args.corpus), table, gpu_eval.MAX_CLASSES)
    bigram, unigram = gpu_eval.build_bigram_unigram(pair_ids, hot_to_idx, n_classes)
    return table, pair_ids, hot_to_idx, bigram, unigram, n_classes


def synthetic_checkpoint(
    h: int,
    edge_count: int,
    output_classes: int,
    seed: int,
    label: str,
    projection_source: gpu_eval.ProjectionArrays | None = None,
) -> gpu_eval.CheckpointArrays:
    rng = np.random.default_rng(seed)
    max_edges = h * (h - 1)
    if edge_count > max_edges:
        raise ValueError(f"edge_count {edge_count} > max possible {max_edges}")
    edges = set()
    while len(edges) < edge_count:
        src = int(rng.integers(0, h))
        dst = int(rng.integers(0, h))
        if src != dst:
            edges.add((src, dst))
    sources = np.fromiter((s for s, _ in edges), dtype=np.int64)
    targets = np.fromiter((t for _, t in edges), dtype=np.int64)
    threshold = rng.integers(0, 4, size=h, dtype=np.int16)
    channel = rng.integers(1, 9, size=h, dtype=np.int16)
    polarity = rng.choice(np.array([-1, 1], dtype=np.int16), size=h)
    output_dim = gpu_eval.phi_dim(h)
    if projection_source is not None:
        weights = np.zeros((output_dim, output_classes), dtype=np.int16)
        src_weights = projection_source.weights
        copy_rows = min(output_dim, src_weights.shape[0])
        copy_cols = min(output_classes, src_weights.shape[1])
        weights[:copy_rows, :copy_cols] = src_weights[:copy_rows, :copy_cols]
    else:
        weights = rng.integers(-3, 4, size=(output_dim, output_classes), dtype=np.int16)
    return gpu_eval.CheckpointArrays(
        path=label,
        network=gpu_eval.NetworkArrays(h, sources, targets, threshold, channel, polarity),
        projection=gpu_eval.ProjectionArrays(weights, output_dim, output_classes),
        meta={"step": 0, "accuracy": 0.0, "label": label},
    )


def lift_checkpoint(
    source: gpu_eval.CheckpointArrays,
    h: int,
    edge_count: int,
    seed: int,
    label: str,
) -> gpu_eval.CheckpointArrays:
    ckpt = synthetic_checkpoint(h, edge_count, source.projection.output_classes, seed, label, source.projection)
    copy_h = min(source.network.h, h)
    ckpt.network.threshold[:copy_h] = source.network.threshold[:copy_h]
    ckpt.network.channel[:copy_h] = source.network.channel[:copy_h]
    ckpt.network.polarity[:copy_h] = source.network.polarity[:copy_h]
    existing = set(zip(ckpt.network.sources.astype(int), ckpt.network.targets.astype(int)))
    edges = list(existing)
    for s, t in zip(source.network.sources.astype(int), source.network.targets.astype(int)):
        if s < h and t < h and s != t and (s, t) not in existing:
            existing.add((s, t))
            edges.append((s, t))
    if len(edges) > edge_count:
        edges = edges[:edge_count]
    ckpt.network.sources = np.asarray([s for s, _ in edges], dtype=np.int64)
    ckpt.network.targets = np.asarray([t for _, t in edges], dtype=np.int64)
    return ckpt


def edge_usage(h: int, edge_count: int) -> float:
    return edge_count / max(1, h * (h - 1))


def metric_deltas(metrics_a: dict[str, list[float]], metrics_b: dict[str, list[float]], idx: int = 0) -> dict[str, float]:
    return {m: float(metrics_b[m][idx] - metrics_a[m][idx]) for m in METRICS}


def mo_score(d: dict[str, float]) -> float:
    return d["smooth"] + 0.5 * d["accuracy"] + 1.5 * max(d["unigram"], -0.012) - 0.25 * abs(d["echo"])


def classify_delta(d: dict[str, float]) -> str:
    if abs(d["echo"]) > 0.0015:
        return "ECHO_TRAP"
    if d["smooth"] > 0 and d["accuracy"] >= -0.0005 and d["unigram"] >= -0.001 and mo_score(d) > 0:
        return "POSITIVE_SAFE"
    if mo_score(d) > 0:
        return "POSITIVE_UNSAFE"
    return "NO_SIGNAL"


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def run_correctness(args) -> dict:
    out = Path(args.out) / "correctness"
    out.mkdir(parents=True, exist_ok=True)
    target = gpu_eval.load_checkpoint(Path(args.target))
    table, pair_ids, hot_to_idx, bigram, unigram, _ = load_real_inputs(args)
    device = torch.device(args.device)
    rng = random.Random(args.seed)
    variants = [target]
    existing = set(zip(target.network.sources.astype(int), target.network.targets.astype(int)))
    for i in range(args.correctness_variants):
        edges = list(existing)
        while len(edges) < len(existing) + 4:
            s = rng.randrange(target.network.h)
            t = rng.randrange(target.network.h)
            if s != t and (s, t) not in set(edges):
                edges.append((s, t))
        net = gpu_eval.NetworkArrays(
            target.network.h,
            np.asarray([s for s, _ in edges], dtype=np.int64),
            np.asarray([t for _, t in edges], dtype=np.int64),
            target.network.threshold.copy(),
            target.network.channel.copy(),
            target.network.polarity.copy(),
        )
        variants.append(gpu_eval.CheckpointArrays(f"variant_{i}", net, target.projection, target.meta))
    dense_metrics, dense_elapsed = gpu_eval.evaluate_metrics(
        variants, table, pair_ids, hot_to_idx, bigram, unigram, args.eval_len, parse_int_csv(args.eval_seeds), device
    )
    sparse_metrics, sparse_elapsed, peak_mb = evaluate_metrics_sparse(
        variants, table, pair_ids, hot_to_idx, bigram, unigram, args.eval_len, parse_int_csv(args.eval_seeds), device
    )
    rows = []
    max_abs = 0.0
    verdict_flip = False
    for idx, ckpt in enumerate(variants):
        for metric in METRICS:
            diff = float(sparse_metrics[metric][idx] - dense_metrics[metric][idx])
            max_abs = max(max_abs, abs(diff))
            rows.append({
                "checkpoint": ckpt.path,
                "metric": metric,
                "dense": dense_metrics[metric][idx],
                "sparse": sparse_metrics[metric][idx],
                "diff": diff,
            })
    dense_delta = metric_deltas({m: [dense_metrics[m][0]] for m in METRICS}, {m: [dense_metrics[m][1]] for m in METRICS})
    sparse_delta = metric_deltas({m: [sparse_metrics[m][0]] for m in METRICS}, {m: [sparse_metrics[m][1]] for m in METRICS})
    verdict_flip = classify_delta(dense_delta) != classify_delta(sparse_delta)
    write_csv(out / "metric_match.csv", rows, ["checkpoint", "metric", "dense", "sparse", "diff"])
    summary = {
        "verdict": "D10J_CORRECTNESS_PASS" if max_abs <= args.max_metric_diff and not verdict_flip else "D10J_SPARSE_NUMERIC_DRIFT_FAIL",
        "max_abs_diff": max_abs,
        "verdict_flip": verdict_flip,
        "dense_elapsed_s": dense_elapsed,
        "sparse_elapsed_s": sparse_elapsed,
        "sparse_peak_mb": peak_mb,
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_throughput(args) -> dict:
    out = Path(args.out) / "throughput"
    out.mkdir(parents=True, exist_ok=True)
    target = gpu_eval.load_checkpoint(Path(args.target))
    table, pair_ids, hot_to_idx, bigram, unigram, n_classes = load_real_inputs(args)
    device = torch.device(args.device)
    rows = []
    for h in parse_int_csv(args.h_sizes):
        for edge_count in parse_int_csv(args.edge_counts):
            if edge_count >= h * (h - 1):
                continue
            for batch_size in parse_int_csv(args.batch_sizes):
                ckpts = [
                    synthetic_checkpoint(h, edge_count, n_classes, args.seed + h * 100000 + edge_count + i, f"H{h}_E{edge_count}_B{i}", target.projection)
                    for i in range(batch_size)
                ]
                _, elapsed, peak_mb = evaluate_metrics_sparse(
                    ckpts, table, pair_ids, hot_to_idx, bigram, unigram, args.eval_len, parse_int_csv(args.eval_seeds), device
                )
                rows.append({
                    "h": h,
                    "edge_count": edge_count,
                    "usage": edge_usage(h, edge_count),
                    "batch_size": batch_size,
                    "elapsed_s": elapsed,
                    "candidates_per_s": batch_size / elapsed if elapsed > 0 else 0.0,
                    "peak_mb": peak_mb,
                })
                print(
                    f"D10j throughput H={h} E={edge_count} batch={batch_size} cps={rows[-1]['candidates_per_s']:.2f} peak_mb={peak_mb:.1f}",
                    flush=True,
                )
    write_csv(out / "throughput.csv", rows, ["h", "edge_count", "usage", "batch_size", "elapsed_s", "candidates_per_s", "peak_mb"])
    knee = {
        "h1024_25k_batch32": any(r["h"] == 1024 and r["edge_count"] >= 25000 and r["batch_size"] >= 32 for r in rows),
        "best_candidates_per_s": max((r["candidates_per_s"] for r in rows), default=0.0),
        "max_peak_mb": max((r["peak_mb"] for r in rows), default=0.0),
    }
    (out / "vram_knee.json").write_text(json.dumps(knee, indent=2), encoding="utf-8")
    return {"rows": rows, "knee": knee}


def mutate_edges(ckpt: gpu_eval.CheckpointArrays, rng: random.Random, swaps: int) -> gpu_eval.CheckpointArrays:
    h = ckpt.network.h
    edges = list(zip(ckpt.network.sources.astype(int), ckpt.network.targets.astype(int)))
    edge_set = set(edges)
    for _ in range(swaps):
        if edges and rng.random() < 0.5:
            idx = rng.randrange(len(edges))
            edge_set.discard(edges[idx])
            edges.pop(idx)
        for _attempt in range(1000):
            s = rng.randrange(h)
            t = rng.randrange(h)
            if s != t and (s, t) not in edge_set:
                edge_set.add((s, t))
                edges.append((s, t))
                break
    net = gpu_eval.NetworkArrays(
        h,
        np.asarray([s for s, _ in edges], dtype=np.int64),
        np.asarray([t for _, t in edges], dtype=np.int64),
        ckpt.network.threshold.copy(),
        ckpt.network.channel.copy(),
        ckpt.network.polarity.copy(),
    )
    return gpu_eval.CheckpointArrays("mutated", net, ckpt.projection, ckpt.meta)


def run_sensitivity(args) -> dict:
    out = Path(args.out) / "sensitivity"
    out.mkdir(parents=True, exist_ok=True)
    target = gpu_eval.load_checkpoint(Path(args.target))
    table, pair_ids, hot_to_idx, bigram, unigram, n_classes = load_real_inputs(args)
    device = torch.device(args.device)
    rows = []
    summary_rows = []
    for h in parse_int_csv(args.h_sizes):
        for edge_count in parse_int_csv(args.edge_counts):
            base = lift_checkpoint(target, h, edge_count, args.seed + h + edge_count, f"H{h}_E{edge_count}_base")
            candidates = [base]
            rng = random.Random(args.seed + h * 17 + edge_count)
            for i in range(args.sensitivity_candidates):
                candidates.append(mutate_edges(base, rng, args.sensitivity_swaps))
            metrics, elapsed, peak_mb = evaluate_metrics_sparse(
                candidates, table, pair_ids, hot_to_idx, bigram, unigram, args.eval_len, parse_int_csv(args.eval_seeds), device
            )
            class_counts: dict[str, int] = {}
            for idx in range(1, len(candidates)):
                d = {m: float(metrics[m][idx] - metrics[m][0]) for m in METRICS}
                klass = classify_delta(d)
                class_counts[klass] = class_counts.get(klass, 0) + 1
                rows.append({
                    "h": h,
                    "edge_count": edge_count,
                    "usage": edge_usage(h, edge_count),
                    "candidate_idx": idx,
                    "class": klass,
                    "smooth_delta": d["smooth"],
                    "accuracy_delta": d["accuracy"],
                    "echo_delta": d["echo"],
                    "unigram_delta": d["unigram"],
                    "mo_score": mo_score(d),
                })
            summary_rows.append({
                "h": h,
                "edge_count": edge_count,
                "usage": edge_usage(h, edge_count),
                "elapsed_s": elapsed,
                "peak_mb": peak_mb,
                "positive_safe": class_counts.get("POSITIVE_SAFE", 0),
                "positive_unsafe": class_counts.get("POSITIVE_UNSAFE", 0),
                "echo_trap": class_counts.get("ECHO_TRAP", 0),
                "no_signal": class_counts.get("NO_SIGNAL", 0),
                "candidate_count": args.sensitivity_candidates,
            })
            print(
                f"D10j sensitivity H={h} E={edge_count} safe={class_counts.get('POSITIVE_SAFE',0)} "
                f"unsafe={class_counts.get('POSITIVE_UNSAFE',0)} echo={class_counts.get('ECHO_TRAP',0)}",
                flush=True,
            )
    write_csv(out / "candidate_sensitivity.csv", rows, ["h", "edge_count", "usage", "candidate_idx", "class", "smooth_delta", "accuracy_delta", "echo_delta", "unigram_delta", "mo_score"])
    write_csv(out / "usage_summary.csv", summary_rows, ["h", "edge_count", "usage", "elapsed_s", "peak_mb", "positive_safe", "positive_unsafe", "echo_trap", "no_signal", "candidate_count"])
    summary = {"rows": summary_rows}
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def write_report(out: Path, correctness: dict | None, throughput: dict | None, sensitivity: dict | None) -> dict:
    verdict = "D10J_INFRA_ONLY"
    if correctness and correctness.get("verdict") == "D10J_SPARSE_NUMERIC_DRIFT_FAIL":
        verdict = "D10J_NUMERIC_DRIFT_FAIL"
    elif throughput:
        rows = throughput.get("rows", [])
        ready = any(r["h"] == 1024 and r["edge_count"] >= 25000 and r["batch_size"] >= 32 for r in rows)
        if ready and sensitivity:
            any_safe = any(row.get("positive_safe", 0) > 0 for row in sensitivity.get("rows", []))
            verdict = "D10J_HIGH_H_PROMISING" if any_safe else "D10J_SPARSE_GPU_READY"
        elif ready:
            verdict = "D10J_SPARSE_GPU_READY"
        else:
            verdict = "D10J_NOT_WORTH_GPU_YET"
    lines = [
        "# Phase D10j Sparse High-H GPU Feasibility",
        "",
        f"Verdict: `{verdict}`",
        "",
        "## Summary",
        "",
        "D10j tests the brain-like large-H / low-active-edge direction while D10b runs on CPU.",
        "",
        "## Progress Map",
        "",
        "```text",
        "[1] H384 beta.8 generalist: DONE",
        "[2] causal mechanism: DONE",
        "[3] H384 seed replication: RUNNING D10b",
        "[4] dense/add local probes: DONE",
        "[5] sparse high-H GPU: D10j current",
        "[6] fair H512 science run: after D10j + checkpoint gate",
        "```",
    ]
    (Path(out) / "D10J_SPARSE_HIGH_H_GPU_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"verdict": verdict}


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["all", "correctness", "throughput", "sensitivity"], default="all")
    p.add_argument("--target", default="output/releases/v5.0.0-beta.8/seed2042_improved_generalist_v1.ckpt")
    p.add_argument("--packed", default="output/block_c_bytepair_champion/packed.bin")
    p.add_argument("--corpus", default="instnct-core/tests/fixtures/alice_corpus.txt")
    p.add_argument("--out", default="output/phase_d10j_sparse_high_h_gpu_probe_20260430")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--eval-len", type=int, default=128)
    p.add_argument("--eval-seeds", default="983001,983002")
    p.add_argument("--h-sizes", default="512,1024")
    p.add_argument("--edge-counts", default="5000,10000,25000,50000,100000")
    p.add_argument("--batch-sizes", default="16,32,64,128")
    p.add_argument("--correctness-variants", type=int, default=4)
    p.add_argument("--max-metric-diff", type=float, default=1e-6)
    p.add_argument("--sensitivity-candidates", type=int, default=32)
    p.add_argument("--sensitivity-swaps", type=int, default=16)
    p.add_argument("--seed", type=int, default=20260430)
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    correctness = throughput = sensitivity = None
    if args.mode in ("all", "correctness"):
        correctness = run_correctness(args)
        if correctness["verdict"] == "D10J_SPARSE_NUMERIC_DRIFT_FAIL":
            write_report(args.out, correctness, None, None)
            print(json.dumps(correctness, indent=2), flush=True)
            return 2
    if args.mode in ("all", "throughput"):
        throughput = run_throughput(args)
    if args.mode in ("all", "sensitivity"):
        sensitivity = run_sensitivity(args)
    final = write_report(args.out, correctness, throughput, sensitivity)
    print(json.dumps(final, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
