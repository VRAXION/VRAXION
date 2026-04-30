#!/usr/bin/env python3
"""D10o high-H projection/start gate.

Scratch/prototype only. D10n showed that H8192 can be partially revived, while
H16384 is still control-noisy or echo-trap dominated. D10o tests whether better
projection/start variants improve that high-H structure gate.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch

import d10g_gpu_eval_probe as gpu_eval
import d10j_sparse_high_h_gpu_probe as d10j
import d10k_h1024_sparse_guided_scout as d10k


MAX_THRESHOLD = 15
METRICS = d10j.METRICS


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_int_csv(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def stable_arm_seed(arm: str) -> int:
    return sum((idx + 1) * ord(ch) for idx, ch in enumerate(arm))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def tiled_projection(source: gpu_eval.ProjectionArrays, h: int, output_classes: int, mode: str, seed: int) -> gpu_eval.ProjectionArrays:
    output_dim = gpu_eval.phi_dim(h)
    src = source.weights.astype(np.int16)
    weights = np.zeros((output_dim, output_classes), dtype=np.int16)
    copy_cols = min(output_classes, src.shape[1])
    if mode == "copy_zero":
        copy_rows = min(output_dim, src.shape[0])
        weights[:copy_rows, :copy_cols] = src[:copy_rows, :copy_cols]
    elif mode == "tiled":
        for row in range(output_dim):
            weights[row, :copy_cols] = src[row % src.shape[0], :copy_cols]
    elif mode == "tiled_signed":
        rng = np.random.default_rng(seed)
        for row in range(output_dim):
            sign = -1 if rng.random() < 0.5 else 1
            weights[row, :copy_cols] = np.clip(sign * src[row % src.shape[0], :copy_cols], -128, 127)
    elif mode == "random_tiny":
        rng = np.random.default_rng(seed)
        weights[:, :copy_cols] = rng.integers(-2, 3, size=(output_dim, copy_cols), dtype=np.int16)
    else:
        raise ValueError(f"unknown projection mode: {mode}")
    return gpu_eval.ProjectionArrays(weights, output_dim, output_classes)


def threshold_array(source: gpu_eval.CheckpointArrays, h: int, mode: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    copy_h = min(source.network.h, h)
    if mode == "low":
        threshold = rng.integers(0, 4, size=h, dtype=np.int16)
    elif mode == "mid":
        threshold = rng.integers(4, 9, size=h, dtype=np.int16)
    elif mode == "high":
        threshold = rng.integers(9, 16, size=h, dtype=np.int16)
    elif mode == "tiled":
        threshold = np.resize(source.network.threshold, h).astype(np.int16)
    else:
        raise ValueError(f"unknown threshold mode: {mode}")
    threshold[:copy_h] = source.network.threshold[:copy_h]
    return threshold


def node_params(source: gpu_eval.CheckpointArrays, h: int, threshold_mode: str, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    copy_h = min(source.network.h, h)
    threshold = threshold_array(source, h, threshold_mode, seed + 11)
    channel = rng.integers(1, 9, size=h, dtype=np.int16)
    polarity = rng.choice(np.array([-1, 1], dtype=np.int16), size=h)
    channel[:copy_h] = source.network.channel[:copy_h]
    polarity[:copy_h] = source.network.polarity[:copy_h]
    if threshold_mode == "tiled":
        channel = np.resize(source.network.channel, h).astype(np.int16)
        polarity = np.resize(source.network.polarity, h).astype(np.int16)
    return threshold, channel, polarity


def source_edges(source: gpu_eval.CheckpointArrays) -> list[tuple[int, int]]:
    return [(int(s), int(t)) for s, t in zip(source.network.sources, source.network.targets) if int(s) != int(t)]


def band_lift_edges(src_edges: list[tuple[int, int]], h: int, band_size: int, band_count: int) -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    for band in range(band_count):
        offset = band * band_size
        if offset >= h:
            break
        for s, t in src_edges:
            ss = (s + offset) % h
            tt = (t + offset) % h
            if ss != tt:
                edges.append((ss, tt))
    return edges


def shortcut_edges_no_echo(src_edges: list[tuple[int, int]], h: int, limit_per_mid: int = 2) -> list[tuple[int, int]]:
    outgoing: dict[int, list[int]] = {}
    reverse = {(t, s) for s, t in src_edges}
    for s, t in src_edges:
        if s < h and t < h:
            outgoing.setdefault(s, []).append(t)
    edges: list[tuple[int, int]] = []
    for s, mids in list(outgoing.items()):
        for mid in mids[:limit_per_mid]:
            for t in outgoing.get(mid, [])[:limit_per_mid]:
                if s != t and (s, t) not in reverse:
                    edges.append((s, t))
    return edges


def build_start(
    source: gpu_eval.CheckpointArrays,
    arm: str,
    h: int,
    edge_count: int,
    seed: int,
) -> gpu_eval.CheckpointArrays:
    rng = random.Random(seed)
    src_edges = source_edges(source)
    band_size = source.network.h
    projection_mode = "copy_zero"
    threshold_mode = "low"
    local_prob = 0.2
    edges: list[tuple[int, int]]

    if arm == "beta8_lifted_v2":
        projection_mode = "tiled"
        threshold_mode = "tiled"
        edges = band_lift_edges(src_edges, h, band_size, band_count=4)
        local_prob = 0.35
    elif arm == "motif_no_echo":
        projection_mode = "tiled"
        threshold_mode = "tiled"
        edges = band_lift_edges(src_edges, h, band_size, band_count=3)
        edges.extend(shortcut_edges_no_echo(src_edges, h, limit_per_mid=2))
        local_prob = 0.55
    elif arm == "projection_tiled":
        projection_mode = "tiled"
        threshold_mode = "low"
        edges = band_lift_edges(src_edges, h, band_size, band_count=2)
        local_prob = 0.25
    elif arm == "projection_signed":
        projection_mode = "tiled_signed"
        threshold_mode = "low"
        edges = band_lift_edges(src_edges, h, band_size, band_count=2)
        local_prob = 0.25
    elif arm == "threshold_mid":
        projection_mode = "tiled"
        threshold_mode = "mid"
        edges = band_lift_edges(src_edges, h, band_size, band_count=3)
        local_prob = 0.35
    elif arm == "threshold_high":
        projection_mode = "tiled"
        threshold_mode = "high"
        edges = band_lift_edges(src_edges, h, band_size, band_count=3)
        local_prob = 0.35
    elif arm == "random_tiny_projection":
        projection_mode = "random_tiny"
        threshold_mode = "low"
        edges = []
        local_prob = 0.0
    else:
        raise ValueError(f"unknown D10o arm: {arm}")

    local_nodes = sorted({n for e in edges for n in e if n < h})
    edges = d10k.fill_edges(edges, h, edge_count, rng, local_nodes, local_prob=local_prob)
    threshold, channel, polarity = node_params(source, h, threshold_mode, seed + 101)
    projection = tiled_projection(source.projection, h, source.projection.output_classes, projection_mode, seed + 202)
    return gpu_eval.CheckpointArrays(
        path=f"{arm}_base",
        network=gpu_eval.NetworkArrays(
            h,
            np.asarray([s for s, _ in edges], dtype=np.int64),
            np.asarray([t for _, t in edges], dtype=np.int64),
            threshold,
            channel,
            polarity,
        ),
        projection=projection,
        meta={"step": 0, "accuracy": 0.0, "label": f"{arm}_base"},
    )


def proposal_style(arm: str) -> str:
    if arm in {"beta8_lifted_v2", "projection_tiled", "projection_signed", "threshold_mid", "threshold_high"}:
        return "beta8_lifted"
    if arm == "motif_no_echo":
        return "motif_guided"
    return "random_label_control"


def propose(base: gpu_eval.CheckpointArrays, arm: str, idx: int, rng: random.Random, edge_swaps: int, threshold_mutations: int):
    candidate, meta = d10k.propose_from_base(base, proposal_style(arm), idx, rng, edge_swaps, threshold_mutations)
    candidate.path = f"{arm}_candidate_{idx:04d}"
    meta.arm = arm
    meta.label = candidate.path
    return candidate, meta


def evaluate_arm(args, arm: str, target, table, pair_ids, hot_to_idx, bigram, unigram, device) -> tuple[list[dict], dict, gpu_eval.CheckpointArrays]:
    arm_seed = args.seed + stable_arm_seed(arm)
    base = build_start(target, arm, args.h, args.edge_count, arm_seed)
    candidates = [base]
    metas: list[d10k.ProposalMeta] = []
    noop = d10k.clone_checkpoint(base, f"{arm}_noop_control")
    duplicate_edges, self_loops = d10k.edge_sanity(list(zip(noop.network.sources.astype(int), noop.network.targets.astype(int))))
    candidates.append(noop)
    metas.append(d10k.ProposalMeta(arm, 0, noop.path, len(noop.network.sources), 0, 0, duplicate_edges, self_loops))
    rng = random.Random(arm_seed + 17)
    for idx in range(1, args.proposals_per_arm + 1):
        candidate, meta = propose(base, arm, idx, rng, args.edge_swaps, args.threshold_mutations)
        candidates.append(candidate)
        metas.append(meta)
    metrics, elapsed_s, peak_mb = d10j.evaluate_metrics_sparse(
        candidates,
        table,
        pair_ids,
        hot_to_idx,
        bigram,
        unigram,
        args.eval_len,
        parse_int_csv(args.eval_seeds),
        device,
    )
    rows = [d10k.row_from_delta(meta, metrics, 0, idx) for idx, meta in enumerate(metas, start=1)]
    summary = d10k.summarize_arm(rows, arm, elapsed_s, peak_mb)
    print(f"D10o arm={arm} safe={summary['positive_safe']} echo={summary['echo_trap']} best_mo={summary['best_mo_score']:.6f}", flush=True)
    return rows, summary, base


def run_random_label_control(args, target, table, pair_ids, hot_to_idx, bigram, unigram, device, base: gpu_eval.CheckpointArrays) -> dict:
    rng = random.Random(args.seed + 900037)
    candidates = [base]
    metas: list[d10k.ProposalMeta] = []
    for idx in range(1, args.control_candidates + 1):
        candidate, meta = d10k.propose_from_base(base, "random_label_control", idx, rng, args.edge_swaps, args.threshold_mutations)
        candidate.path = f"random_label_control_candidate_{idx:04d}"
        meta.arm = "random_label_control"
        meta.label = candidate.path
        candidates.append(candidate)
        metas.append(meta)
    shuffled_hot, shuffled_bigram, shuffled_unigram = d10k.shuffle_metric_targets(hot_to_idx, bigram, unigram, args.seed + 900071)
    metrics, elapsed_s, peak_mb = d10j.evaluate_metrics_sparse(
        candidates,
        table,
        pair_ids,
        shuffled_hot,
        shuffled_bigram,
        shuffled_unigram,
        args.eval_len,
        parse_int_csv(args.eval_seeds),
        device,
    )
    rows = [d10k.row_from_delta(meta, metrics, 0, idx) for idx, meta in enumerate(metas, start=1)]
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["class"]] = counts.get(row["class"], 0) + 1
    return {
        "arm": "random_label_control",
        "source_arm": args.control_source_arm,
        "candidate_count": len(rows),
        "positive_safe": counts.get("POSITIVE_SAFE", 0),
        "positive_unsafe": counts.get("POSITIVE_UNSAFE", 0),
        "echo_trap": counts.get("ECHO_TRAP", 0),
        "no_signal": counts.get("NO_SIGNAL", 0),
        "elapsed_s": elapsed_s,
        "peak_mb": peak_mb,
        "rows": rows,
    }


def verdict_from(arm_summaries: list[dict], control_summary: dict) -> dict:
    control_rate = control_summary["positive_safe"] / max(1, control_summary["candidate_count"])
    if control_rate > 0.25:
        verdict = "D10O_CONTROL_FAIL"
    else:
        best = max(arm_summaries, key=lambda r: (r["positive_safe"], r["best_mo_score"]), default=None)
        safe_total = sum(r["positive_safe"] for r in arm_summaries)
        if best and best["positive_safe"] >= 4 and control_rate <= 0.10:
            verdict = "D10O_STRUCTURE_GATE_PASS"
        elif safe_total > 0 and control_rate <= 0.20:
            verdict = "D10O_WEAK_STRUCTURE_SIGNAL"
        elif any(r["echo_trap"] > r["positive_safe"] for r in arm_summaries):
            verdict = "D10O_ECHO_TRAP_DOMINATED"
        else:
            verdict = "D10O_NO_STRUCTURE_SIGNAL"
    return {
        "verdict": verdict,
        "random_label_positive_safe_rate": control_rate,
        "arm_summaries": arm_summaries,
        "control_summary": {k: v for k, v in control_summary.items() if k != "rows"},
    }


def write_outputs(out: Path, rows: list[dict], arm_summaries: list[dict], control_summary: dict, args) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    fields = [
        "arm",
        "candidate_idx",
        "label",
        "class",
        "smooth_delta",
        "accuracy_delta",
        "echo_delta",
        "unigram_delta",
        "mo_score",
        "edge_count",
        "edge_swaps",
        "threshold_mutations",
        "duplicate_edges",
        "self_loops",
        "smooth_base",
        "smooth_candidate",
        "accuracy_base",
        "accuracy_candidate",
        "echo_base",
        "echo_candidate",
        "unigram_base",
        "unigram_candidate",
    ]
    write_csv(out / "start_candidates.csv", rows, fields)
    write_csv(out / "random_label_control.csv", control_summary["rows"], fields)
    write_csv(
        out / "arm_summary.csv",
        arm_summaries,
        [
            "arm",
            "candidate_count",
            "positive_safe",
            "positive_unsafe",
            "echo_trap",
            "no_signal",
            "best_class",
            "best_mo_score",
            "best_smooth_delta",
            "best_accuracy_delta",
            "best_echo_delta",
            "best_unigram_delta",
            "elapsed_s",
            "peak_mb",
        ],
    )
    summary = verdict_from(arm_summaries, control_summary)
    summary["setup"] = {
        "h": args.h,
        "edge_count": args.edge_count,
        "usage": d10j.edge_usage(args.h, args.edge_count),
        "eval_len": args.eval_len,
        "eval_seeds": parse_int_csv(args.eval_seeds),
        "proposals_per_arm": args.proposals_per_arm,
        "arms": parse_csv(args.arms),
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(out, summary)
    return summary


def write_report(out: Path, summary: dict) -> None:
    setup = summary["setup"]
    lines = [
        "# D10o High-H Projection/Start Gate",
        "",
        f"Verdict: `{summary['verdict']}`",
        "",
        "## Setup",
        "",
        f"- H: `{setup['h']}`",
        f"- active edges: `{setup['edge_count']}`",
        f"- usage: `{setup['usage'] * 100:.4f}%`",
        f"- eval_len: `{setup['eval_len']}`",
        f"- eval_seeds: `{','.join(str(s) for s in setup['eval_seeds'])}`",
        "",
        "## Arm Summary",
        "",
        "| arm | safe | unsafe | echo trap | no signal | best MO |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["arm_summaries"]:
        lines.append(
            f"| {row['arm']} | {row['positive_safe']} | {row['positive_unsafe']} | {row['echo_trap']} | "
            f"{row['no_signal']} | {row['best_mo_score']:.6f} |"
        )
    c = summary["control_summary"]
    lines.extend(
        [
            "",
            "## Adversarial Control",
            "",
            f"Random-label safe-positive rate: `{summary['random_label_positive_safe_rate']:.3f}`",
            f"Random-label counts: safe={c['positive_safe']}, unsafe={c['positive_unsafe']}, echo={c['echo_trap']}, no_signal={c['no_signal']}",
            "",
            "GPU-only D10o output is scout evidence only. Any high-H candidate still needs dense-reference or CPU-compatible confirmation.",
        ]
    )
    (out / "D10O_HIGH_H_PROJECTION_START_GATE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_gate(args) -> dict:
    out = Path(args.out)
    target = gpu_eval.load_checkpoint(Path(args.target))
    table, pair_ids, hot_to_idx, bigram, unigram, _ = d10j.load_real_inputs(args)
    device = torch.device(args.device)
    all_rows: list[dict] = []
    summaries: list[dict] = []
    bases: dict[str, gpu_eval.CheckpointArrays] = {}
    for arm in parse_csv(args.arms):
        rows, summary, base = evaluate_arm(args, arm, target, table, pair_ids, hot_to_idx, bigram, unigram, device)
        all_rows.extend(rows)
        summaries.append(summary)
        bases[arm] = base
    control_arm = args.control_source_arm
    control_base = bases.get(control_arm) or next(iter(bases.values()))
    control_summary = run_random_label_control(args, target, table, pair_ids, hot_to_idx, bigram, unigram, device, control_base)
    return write_outputs(out, all_rows, summaries, control_summary, args)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--target", default="output/releases/v5.0.0-beta.8/seed2042_improved_generalist_v1.ckpt")
    p.add_argument("--packed", default="output/block_c_bytepair_champion/packed.bin")
    p.add_argument("--corpus", default="instnct-core/tests/fixtures/alice_corpus.txt")
    p.add_argument("--out", default="output/phase_d10o_high_h_projection_start_gate_20260430")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--h", type=int, default=8192)
    p.add_argument("--edge-count", type=int, default=100000)
    p.add_argument("--eval-len", type=int, default=128)
    p.add_argument("--eval-seeds", default="987001,987002")
    p.add_argument("--arms", default="beta8_lifted_v2,motif_no_echo,projection_tiled,projection_signed,threshold_mid,threshold_high")
    p.add_argument("--proposals-per-arm", type=int, default=16)
    p.add_argument("--edge-swaps", type=int, default=16)
    p.add_argument("--threshold-mutations", type=int, default=16)
    p.add_argument("--control-source-arm", default="motif_no_echo")
    p.add_argument("--control-candidates", type=int, default=16)
    p.add_argument("--seed", type=int, default=20260430)
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    summary = run_gate(args)
    print(json.dumps({k: v for k, v in summary.items() if k != "arm_summaries"}, indent=2), flush=True)
    return 0 if summary["verdict"] != "D10O_CONTROL_FAIL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
