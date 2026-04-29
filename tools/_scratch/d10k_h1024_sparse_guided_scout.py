#!/usr/bin/env python3
"""D10k H1024 sparse guided scout.

Scratch/prototype only. This builds on the D10j sparse GPU evaluator and asks
whether the H1024 / 25k low-usage regime is merely evaluable, or also more
searchable than the local H384 add/fill probes.
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
import d10j_sparse_high_h_gpu_probe as d10j


METRICS = d10j.METRICS
MAX_THRESHOLD = 15


@dataclass
class ProposalMeta:
    arm: str
    candidate_idx: int
    label: str
    edge_count: int
    edge_swaps: int
    threshold_mutations: int
    duplicate_edges: int
    self_loops: int


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_int_csv(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def clone_checkpoint(ckpt: gpu_eval.CheckpointArrays, label: str) -> gpu_eval.CheckpointArrays:
    return gpu_eval.CheckpointArrays(
        path=label,
        network=gpu_eval.NetworkArrays(
            ckpt.network.h,
            ckpt.network.sources.copy(),
            ckpt.network.targets.copy(),
            ckpt.network.threshold.copy(),
            ckpt.network.channel.copy(),
            ckpt.network.polarity.copy(),
        ),
        projection=gpu_eval.ProjectionArrays(
            ckpt.projection.weights.copy(),
            ckpt.projection.input_dim,
            ckpt.projection.output_classes,
        ),
        meta=dict(ckpt.meta),
    )


def resize_projection(source: gpu_eval.ProjectionArrays, h: int, output_classes: int) -> gpu_eval.ProjectionArrays:
    output_dim = gpu_eval.phi_dim(h)
    weights = np.zeros((output_dim, output_classes), dtype=np.int16)
    copy_rows = min(output_dim, source.weights.shape[0])
    copy_cols = min(output_classes, source.weights.shape[1])
    weights[:copy_rows, :copy_cols] = source.weights[:copy_rows, :copy_cols]
    return gpu_eval.ProjectionArrays(weights, output_dim, output_classes)


def edge_sanity(edges: list[tuple[int, int]]) -> tuple[int, int]:
    self_loops = sum(1 for s, t in edges if s == t)
    duplicate_edges = len(edges) - len(set(edges))
    return duplicate_edges, self_loops


def fill_edges(
    edges: list[tuple[int, int]],
    h: int,
    edge_count: int,
    rng: random.Random,
    local_nodes: list[int] | None = None,
    local_prob: float = 0.0,
) -> list[tuple[int, int]]:
    edge_set = set((int(s), int(t)) for s, t in edges if 0 <= s < h and 0 <= t < h and s != t)
    local_nodes = local_nodes or []
    attempts = 0
    max_attempts = edge_count * 200
    while len(edge_set) < edge_count and attempts < max_attempts:
        attempts += 1
        if local_nodes and rng.random() < local_prob:
            s = rng.choice(local_nodes)
            t = rng.randrange(h) if rng.random() < 0.5 else rng.choice(local_nodes)
        else:
            s = rng.randrange(h)
            t = rng.randrange(h)
        if s != t:
            edge_set.add((s, t))
    if len(edge_set) < edge_count:
        raise RuntimeError(f"could not fill {edge_count} unique edges for H={h}")
    return list(edge_set)[:edge_count]


def beta8_lift_checkpoint(
    source: gpu_eval.CheckpointArrays,
    h: int,
    edge_count: int,
    seed: int,
    label: str,
    motif_guided: bool = False,
) -> gpu_eval.CheckpointArrays:
    rng = random.Random(seed)
    copy_h = min(source.network.h, h)
    threshold = np.asarray(rng.choices(range(4), k=h), dtype=np.int16)
    channel = np.asarray(rng.choices(range(1, 9), k=h), dtype=np.int16)
    polarity = np.asarray(rng.choices([-1, 1], k=h), dtype=np.int16)
    threshold[:copy_h] = source.network.threshold[:copy_h]
    channel[:copy_h] = source.network.channel[:copy_h]
    polarity[:copy_h] = source.network.polarity[:copy_h]

    source_edges = [(int(s), int(t)) for s, t in zip(source.network.sources, source.network.targets) if int(s) != int(t)]
    edges = [(s, t) for s, t in source_edges if s < h and t < h]
    if motif_guided:
        edges.extend((t, s) for s, t in source_edges if s < h and t < h and s != t)
        outgoing: dict[int, list[int]] = {}
        for s, t in source_edges:
            if s < h and t < h:
                outgoing.setdefault(s, []).append(t)
        for s, mids in list(outgoing.items()):
            for mid in mids[:4]:
                for t in outgoing.get(mid, [])[:4]:
                    if s != t:
                        edges.append((s, t))
        # Lift the same local motif into two higher-H bands so the scout is not
        # only testing the original low-index H384 subspace.
        for offset in (copy_h, min(h - copy_h, copy_h) + copy_h):
            if offset >= h:
                continue
            for s, t in source_edges[: len(source_edges) // 2]:
                ss = (s + offset) % h
                tt = (t + offset) % h
                if ss != tt:
                    edges.append((ss, tt))

    local_nodes = sorted({n for e in source_edges for n in e if n < h})
    edges = fill_edges(edges, h, edge_count, rng, local_nodes, local_prob=0.6 if motif_guided else 0.2)
    return gpu_eval.CheckpointArrays(
        path=label,
        network=gpu_eval.NetworkArrays(
            h,
            np.asarray([s for s, _ in edges], dtype=np.int64),
            np.asarray([t for _, t in edges], dtype=np.int64),
            threshold,
            channel,
            polarity,
        ),
        projection=resize_projection(source.projection, h, source.projection.output_classes),
        meta={"step": 0, "accuracy": 0.0, "label": label},
    )


def random_sparse_checkpoint(
    source: gpu_eval.CheckpointArrays,
    h: int,
    edge_count: int,
    seed: int,
    label: str,
) -> gpu_eval.CheckpointArrays:
    ckpt = d10j.synthetic_checkpoint(h, edge_count, source.projection.output_classes, seed, label, source.projection)
    ckpt.projection = resize_projection(source.projection, h, source.projection.output_classes)
    return ckpt


def propose_from_base(
    base: gpu_eval.CheckpointArrays,
    arm: str,
    candidate_idx: int,
    rng: random.Random,
    edge_swaps: int,
    threshold_mutations: int,
) -> tuple[gpu_eval.CheckpointArrays, ProposalMeta]:
    ckpt = clone_checkpoint(base, f"{arm}_candidate_{candidate_idx:04d}")
    h = ckpt.network.h
    edges = list(zip(ckpt.network.sources.astype(int), ckpt.network.targets.astype(int)))
    edge_set = set(edges)
    active_nodes = sorted({n for s, t in edges for n in (s, t)})

    for _ in range(edge_swaps):
        if edges:
            remove_idx = rng.randrange(len(edges))
            edge_set.discard(edges[remove_idx])
            edges.pop(remove_idx)
        for _attempt in range(1000):
            if arm == "motif_guided" and active_nodes and rng.random() < 0.75:
                s = rng.choice(active_nodes)
                t = rng.choice(active_nodes) if rng.random() < 0.5 else rng.randrange(h)
            elif arm == "beta8_lifted" and active_nodes and rng.random() < 0.45:
                s = rng.choice(active_nodes)
                t = rng.randrange(h)
            else:
                s = rng.randrange(h)
                t = rng.randrange(h)
            if s != t and (s, t) not in edge_set:
                edge_set.add((s, t))
                edges.append((s, t))
                break

    touched = set()
    for _ in range(threshold_mutations):
        if arm in ("beta8_lifted", "motif_guided") and active_nodes and rng.random() < 0.7:
            node = rng.choice(active_nodes)
        else:
            node = rng.randrange(h)
        delta = rng.choice([-2, -1, 1, 2])
        ckpt.network.threshold[node] = np.int16(max(0, min(MAX_THRESHOLD, int(ckpt.network.threshold[node]) + delta)))
        touched.add(node)

    ckpt.network.sources = np.asarray([s for s, _ in edges], dtype=np.int64)
    ckpt.network.targets = np.asarray([t for _, t in edges], dtype=np.int64)
    duplicate_edges, self_loops = edge_sanity(edges)
    meta = ProposalMeta(
        arm=arm,
        candidate_idx=candidate_idx,
        label=ckpt.path,
        edge_count=len(edges),
        edge_swaps=edge_swaps,
        threshold_mutations=len(touched),
        duplicate_edges=duplicate_edges,
        self_loops=self_loops,
    )
    return ckpt, meta


def shuffle_metric_targets(
    hot_to_idx: np.ndarray,
    bigram: np.ndarray,
    unigram: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = bigram.shape[0]
    perm = rng.permutation(n)
    shuffled_hot = hot_to_idx.copy()
    mask = shuffled_hot >= 0
    shuffled_hot[mask] = perm[shuffled_hot[mask]]
    # Keep the control internally consistent but semantically wrong.
    shuffled_bigram = bigram[perm][:, perm]
    shuffled_unigram = unigram[perm]
    return shuffled_hot, shuffled_bigram, shuffled_unigram


def row_from_delta(meta: ProposalMeta, metrics: dict[str, list[float]], base_idx: int, candidate_idx: int) -> dict:
    d = {m: float(metrics[m][candidate_idx] - metrics[m][base_idx]) for m in METRICS}
    klass = d10j.classify_delta(d)
    row = {
        "arm": meta.arm,
        "candidate_idx": meta.candidate_idx,
        "label": meta.label,
        "class": klass,
        "smooth_delta": d["smooth"],
        "accuracy_delta": d["accuracy"],
        "echo_delta": d["echo"],
        "unigram_delta": d["unigram"],
        "mo_score": d10j.mo_score(d),
        "edge_count": meta.edge_count,
        "edge_swaps": meta.edge_swaps,
        "threshold_mutations": meta.threshold_mutations,
        "duplicate_edges": meta.duplicate_edges,
        "self_loops": meta.self_loops,
    }
    for metric in METRICS:
        row[f"{metric}_base"] = metrics[metric][base_idx]
        row[f"{metric}_candidate"] = metrics[metric][candidate_idx]
    return row


def summarize_arm(rows: list[dict], arm: str, elapsed_s: float, peak_mb: float) -> dict:
    arm_rows = [r for r in rows if r["arm"] == arm]
    counts: dict[str, int] = {}
    for row in arm_rows:
        counts[row["class"]] = counts.get(row["class"], 0) + 1
    best = max(arm_rows, key=lambda r: (r["class"] == "POSITIVE_SAFE", r["mo_score"]), default=None)
    return {
        "arm": arm,
        "candidate_count": len(arm_rows),
        "positive_safe": counts.get("POSITIVE_SAFE", 0),
        "positive_unsafe": counts.get("POSITIVE_UNSAFE", 0),
        "echo_trap": counts.get("ECHO_TRAP", 0),
        "no_signal": counts.get("NO_SIGNAL", 0),
        "best_class": best["class"] if best else "",
        "best_mo_score": best["mo_score"] if best else 0.0,
        "best_smooth_delta": best["smooth_delta"] if best else 0.0,
        "best_accuracy_delta": best["accuracy_delta"] if best else 0.0,
        "best_echo_delta": best["echo_delta"] if best else 0.0,
        "best_unigram_delta": best["unigram_delta"] if best else 0.0,
        "elapsed_s": elapsed_s,
        "peak_mb": peak_mb,
    }


def build_base_for_arm(
    arm: str,
    target: gpu_eval.CheckpointArrays,
    h: int,
    edge_count: int,
    seed: int,
) -> gpu_eval.CheckpointArrays:
    if arm == "random_sparse":
        return random_sparse_checkpoint(target, h, edge_count, seed, f"{arm}_base")
    if arm == "beta8_lifted":
        return beta8_lift_checkpoint(target, h, edge_count, seed, f"{arm}_base", motif_guided=False)
    if arm == "motif_guided":
        return beta8_lift_checkpoint(target, h, edge_count, seed, f"{arm}_base", motif_guided=True)
    raise ValueError(f"unknown arm: {arm}")


def run_scout(args) -> dict:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    target = gpu_eval.load_checkpoint(Path(args.target))
    table, pair_ids, hot_to_idx, bigram, unigram, _ = d10j.load_real_inputs(args)
    device = torch.device(args.device)
    eval_seeds = parse_int_csv(args.eval_seeds)
    arms = parse_csv(args.arms)
    all_rows: list[dict] = []
    arm_summaries: list[dict] = []
    base_cache: dict[str, gpu_eval.CheckpointArrays] = {}

    for arm_idx, arm in enumerate(arms):
        seed = args.seed + arm_idx * 100003
        base = build_base_for_arm(arm, target, args.h, args.edge_count, seed)
        base_cache[arm] = base
        candidates = [base]
        metas: list[ProposalMeta] = []
        rng = random.Random(seed + 17)
        # The no-op copy is an adversarial zero-delta control.
        noop = clone_checkpoint(base, f"{arm}_noop_control")
        duplicate_edges, self_loops = edge_sanity(list(zip(noop.network.sources.astype(int), noop.network.targets.astype(int))))
        candidates.append(noop)
        metas.append(ProposalMeta(arm, 0, noop.path, len(noop.network.sources), 0, 0, duplicate_edges, self_loops))
        for candidate_idx in range(1, args.proposals_per_arm + 1):
            candidate, meta = propose_from_base(
                base,
                arm,
                candidate_idx,
                rng,
                args.edge_swaps,
                args.threshold_mutations,
            )
            candidates.append(candidate)
            metas.append(meta)
        t0 = time.perf_counter()
        metrics, elapsed_s, peak_mb = d10j.evaluate_metrics_sparse(
            candidates,
            table,
            pair_ids,
            hot_to_idx,
            bigram,
            unigram,
            args.eval_len,
            eval_seeds,
            device,
        )
        elapsed_s = time.perf_counter() - t0 if elapsed_s <= 0 else elapsed_s
        for idx, meta in enumerate(metas, start=1):
            all_rows.append(row_from_delta(meta, metrics, 0, idx))
        arm_summaries.append(summarize_arm(all_rows, arm, elapsed_s, peak_mb))
        print(
            f"D10k arm={arm} safe={arm_summaries[-1]['positive_safe']} "
            f"echo={arm_summaries[-1]['echo_trap']} best_mo={arm_summaries[-1]['best_mo_score']:.6f}",
            flush=True,
        )

    control_summary = run_random_label_control(args, target, table, pair_ids, hot_to_idx, bigram, unigram, device, base_cache)
    write_outputs(out, all_rows, arm_summaries, control_summary, args)
    return verdict_from_summaries(arm_summaries, control_summary)


def run_random_label_control(
    args,
    target: gpu_eval.CheckpointArrays,
    table: gpu_eval.VcbpTablePy,
    pair_ids: np.ndarray,
    hot_to_idx: np.ndarray,
    bigram: np.ndarray,
    unigram: np.ndarray,
    device: torch.device,
    base_cache: dict[str, gpu_eval.CheckpointArrays],
) -> dict:
    control_arm = args.control_source_arm
    base = base_cache.get(control_arm)
    if base is None:
        base = build_base_for_arm(control_arm, target, args.h, args.edge_count, args.seed + 900001)
    rng = random.Random(args.seed + 900037)
    candidates = [base]
    metas = []
    for candidate_idx in range(1, args.control_candidates + 1):
        candidate, meta = propose_from_base(
            base,
            "random_label_control",
            candidate_idx,
            rng,
            args.edge_swaps,
            args.threshold_mutations,
        )
        candidates.append(candidate)
        metas.append(meta)
    shuffled_hot, shuffled_bigram, shuffled_unigram = shuffle_metric_targets(
        hot_to_idx, bigram, unigram, args.seed + 900071
    )
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
    rows = [row_from_delta(meta, metrics, 0, idx) for idx, meta in enumerate(metas, start=1)]
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["class"]] = counts.get(row["class"], 0) + 1
    return {
        "arm": "random_label_control",
        "source_arm": control_arm,
        "candidate_count": len(rows),
        "positive_safe": counts.get("POSITIVE_SAFE", 0),
        "positive_unsafe": counts.get("POSITIVE_UNSAFE", 0),
        "echo_trap": counts.get("ECHO_TRAP", 0),
        "no_signal": counts.get("NO_SIGNAL", 0),
        "elapsed_s": elapsed_s,
        "peak_mb": peak_mb,
        "rows": rows,
    }


def verdict_from_summaries(arm_summaries: list[dict], control_summary: dict) -> dict:
    control_rate = control_summary["positive_safe"] / max(1, control_summary["candidate_count"])
    if control_rate > 0.25:
        verdict = "D10K_CONTROL_FAIL"
    else:
        guided = {row["arm"]: row for row in arm_summaries}
        random_safe = guided.get("random_sparse", {}).get("positive_safe", 0)
        beta_safe = guided.get("beta8_lifted", {}).get("positive_safe", 0)
        motif_safe = guided.get("motif_guided", {}).get("positive_safe", 0)
        best_safe = max((row.get("positive_safe", 0) for row in arm_summaries), default=0)
        random_mo = guided.get("random_sparse", {}).get("best_mo_score", 0.0)
        beta_mo = guided.get("beta8_lifted", {}).get("best_mo_score", 0.0)
        motif_mo = guided.get("motif_guided", {}).get("best_mo_score", 0.0)
        guided_best_beats_random = max(beta_mo, motif_mo) > max(random_mo * 1.5, random_mo + 0.002)
        if motif_safe > random_safe or beta_safe > random_safe or guided_best_beats_random:
            verdict = "D10K_BETA8_PATTERN_SCALES"
        elif best_safe > 0:
            verdict = "D10K_HIGH_H_SCOUT_SIGNAL"
        else:
            verdict = "D10K_NO_HIGH_H_SIGNAL"
    return {
        "verdict": verdict,
        "random_label_positive_safe_rate": control_rate,
        "arm_summaries": arm_summaries,
        "control_summary": {k: v for k, v in control_summary.items() if k != "rows"},
    }


def write_outputs(out: Path, rows: list[dict], arm_summaries: list[dict], control_summary: dict, args) -> None:
    candidate_fields = [
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
    write_csv(out / "guided_candidates.csv", rows, candidate_fields)
    write_csv(out / "random_label_control.csv", control_summary["rows"], candidate_fields)
    summary_fields = [
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
    ]
    write_csv(out / "arm_summary.csv", arm_summaries, summary_fields)
    usage = {
        "h": args.h,
        "edge_count": args.edge_count,
        "usage": d10j.edge_usage(args.h, args.edge_count),
        "eval_len": args.eval_len,
        "eval_seeds": parse_int_csv(args.eval_seeds),
        "proposals_per_arm": args.proposals_per_arm,
        "edge_swaps": args.edge_swaps,
        "threshold_mutations": args.threshold_mutations,
    }
    (out / "usage_summary.json").write_text(json.dumps(usage, indent=2), encoding="utf-8")
    verdict = verdict_from_summaries(arm_summaries, control_summary)
    (out / "run_summary.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    write_report(out, verdict, usage)


def write_report(out: Path, verdict: dict, usage: dict) -> None:
    lines = [
        "# D10k H1024 Sparse Guided Scout",
        "",
        f"Verdict: `{verdict['verdict']}`",
        "",
        "## Setup",
        "",
        f"- H: `{usage['h']}`",
        f"- active edges: `{usage['edge_count']}`",
        f"- usage: `{usage['usage'] * 100:.2f}%`",
        f"- eval_len: `{usage['eval_len']}`",
        f"- eval_seeds: `{','.join(str(s) for s in usage['eval_seeds'])}`",
        "",
        "## Arm Summary",
        "",
        "| arm | safe | unsafe | echo trap | no signal | best MO |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in verdict["arm_summaries"]:
        lines.append(
            f"| {row['arm']} | {row['positive_safe']} | {row['positive_unsafe']} | "
            f"{row['echo_trap']} | {row['no_signal']} | {row['best_mo_score']:.6f} |"
        )
    c = verdict["control_summary"]
    lines.extend(
        [
            "",
            "## Adversarial Control",
            "",
            f"Random-label safe-positive rate: `{verdict['random_label_positive_safe_rate']:.3f}`",
            f"Random-label counts: safe={c['positive_safe']}, unsafe={c['positive_unsafe']}, "
            f"echo={c['echo_trap']}, no_signal={c['no_signal']}",
            "",
            "GPU-only scout output is not a promotion gate. Any candidate path from D10k needs",
            "a later fair checkpoint/projection setup and CPU or dense-reference confirmation.",
            "",
            "## Progress Map",
            "",
            "```text",
            "[1] H384 beta.8 generalist: DONE",
            "[2] causal mechanism: DONE",
            "[3] H384 seed replication: RUNNING D10b CPU",
            "[4] D10j sparse high-H evaluator: DONE",
            "[5] D10k H1024 sparse guided scout: CURRENT RESULT",
            "[6] fair H512/H1024 science run: after signal + checkpoint/projection gate",
            "```",
        ]
    )
    (out / "D10K_H1024_SPARSE_GUIDED_SCOUT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--target", default="output/releases/v5.0.0-beta.8/seed2042_improved_generalist_v1.ckpt")
    p.add_argument("--packed", default="output/block_c_bytepair_champion/packed.bin")
    p.add_argument("--corpus", default="instnct-core/tests/fixtures/alice_corpus.txt")
    p.add_argument("--out", default="output/phase_d10k_h1024_sparse_guided_scout_20260430")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--h", type=int, default=1024)
    p.add_argument("--edge-count", type=int, default=25000)
    p.add_argument("--eval-len", type=int, default=128)
    p.add_argument("--eval-seeds", default="984001,984002")
    p.add_argument("--arms", default="random_sparse,beta8_lifted,motif_guided")
    p.add_argument("--proposals-per-arm", type=int, default=32)
    p.add_argument("--edge-swaps", type=int, default=16)
    p.add_argument("--threshold-mutations", type=int, default=16)
    p.add_argument("--control-source-arm", default="motif_guided")
    p.add_argument("--control-candidates", type=int, default=32)
    p.add_argument("--seed", type=int, default=20260430)
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    summary = run_scout(args)
    print(json.dumps({k: v for k, v in summary.items() if k != "arm_summaries"}, indent=2), flush=True)
    return 0 if summary["verdict"] != "D10K_CONTROL_FAIL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
