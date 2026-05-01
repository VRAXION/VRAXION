#!/usr/bin/env python3
"""D10p semantic projection hardening.

Scratch/prototype only. D10o showed raw high-H signal, but random-label
controls were too positive. D10p evaluates the same proposal families under
multiple semantic decoys before allowing any high-H signal to pass.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

import d10g_gpu_eval_probe as gpu_eval
import d10j_sparse_high_h_gpu_probe as d10j
import d10k_h1024_sparse_guided_scout as d10k
import d10o_high_h_projection_start_gate as d10o


METRICS = d10j.METRICS
CONTROL_TYPES = ["random_label", "random_bigram", "unigram_decoy", "projection_shuffle"]


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_int_csv(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def clone_with_projection(ckpt: gpu_eval.CheckpointArrays, label: str, projection: gpu_eval.ProjectionArrays) -> gpu_eval.CheckpointArrays:
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
            projection.weights.copy(),
            projection.input_dim,
            projection.output_classes,
        ),
        meta=dict(ckpt.meta),
    )


def frozen_beta8_rows_start(source: gpu_eval.CheckpointArrays, h: int, edge_count: int, seed: int) -> gpu_eval.CheckpointArrays:
    ckpt = d10o.build_start(source, "beta8_lifted_v2", h, edge_count, seed)
    ckpt.path = "frozen_beta8_rows_base"
    ckpt.projection = d10o.tiled_projection(source.projection, h, source.projection.output_classes, "copy_zero", seed + 707)
    return ckpt


def block_local_projection(source: gpu_eval.ProjectionArrays, h: int) -> gpu_eval.ProjectionArrays:
    output_dim = gpu_eval.phi_dim(h)
    src = source.weights.astype(np.int16)
    weights = np.zeros((output_dim, source.output_classes), dtype=np.int16)
    block = src.shape[0]
    # Copy beta.8 rows into separated local blocks with zero gaps. This avoids
    # full tiling while still allowing high-H output bands to participate.
    for offset in range(0, output_dim, block * 2):
        rows = min(block, output_dim - offset)
        if rows <= 0:
            break
        weights[offset : offset + rows, : src.shape[1]] = src[:rows]
    return gpu_eval.ProjectionArrays(weights, output_dim, source.output_classes)


def block_local_projection_start(source: gpu_eval.CheckpointArrays, h: int, edge_count: int, seed: int) -> gpu_eval.CheckpointArrays:
    ckpt = d10o.build_start(source, "beta8_lifted_v2", h, edge_count, seed)
    ckpt.path = "block_local_projection_base"
    ckpt.projection = block_local_projection(source.projection, h)
    return ckpt


def copy_zero_threshold_mid_start(source: gpu_eval.CheckpointArrays, h: int, edge_count: int, seed: int) -> gpu_eval.CheckpointArrays:
    ckpt = d10o.build_start(source, "threshold_mid", h, edge_count, seed)
    ckpt.path = "copy_zero_threshold_mid_base"
    ckpt.projection = d10o.tiled_projection(source.projection, h, source.projection.output_classes, "copy_zero", seed + 807)
    return ckpt


def block_local_threshold_mid_start(source: gpu_eval.CheckpointArrays, h: int, edge_count: int, seed: int) -> gpu_eval.CheckpointArrays:
    ckpt = d10o.build_start(source, "threshold_mid", h, edge_count, seed)
    ckpt.path = "block_local_threshold_mid_base"
    ckpt.projection = block_local_projection(source.projection, h)
    return ckpt


def signed_threshold_mid_start(source: gpu_eval.CheckpointArrays, h: int, edge_count: int, seed: int) -> gpu_eval.CheckpointArrays:
    ckpt = d10o.build_start(source, "threshold_mid", h, edge_count, seed)
    ckpt.path = "signed_threshold_mid_base"
    ckpt.projection = d10o.tiled_projection(source.projection, h, source.projection.output_classes, "tiled_signed", seed + 907)
    return ckpt


def build_start(source: gpu_eval.CheckpointArrays, arm: str, h: int, edge_count: int, seed: int) -> gpu_eval.CheckpointArrays:
    if arm == "frozen_beta8_rows":
        return frozen_beta8_rows_start(source, h, edge_count, seed)
    if arm == "block_local_projection":
        return block_local_projection_start(source, h, edge_count, seed)
    if arm == "copy_zero_threshold_mid":
        return copy_zero_threshold_mid_start(source, h, edge_count, seed)
    if arm == "block_local_threshold_mid":
        return block_local_threshold_mid_start(source, h, edge_count, seed)
    if arm == "signed_threshold_mid":
        return signed_threshold_mid_start(source, h, edge_count, seed)
    return d10o.build_start(source, arm, h, edge_count, seed)


def proposal_style(arm: str) -> str:
    if arm in {"motif_no_echo"}:
        return "motif_guided"
    if arm in {
        "beta8_lifted_v2",
        "projection_tiled",
        "threshold_mid",
        "threshold_high",
        "block_local_projection",
        "frozen_beta8_rows",
        "copy_zero_threshold_mid",
        "block_local_threshold_mid",
        "signed_threshold_mid",
    }:
        return "beta8_lifted"
    return "random_label_control"


def propose(base: gpu_eval.CheckpointArrays, arm: str, idx: int, rng: random.Random, edge_swaps: int, threshold_mutations: int):
    candidate, meta = d10k.propose_from_base(base, proposal_style(arm), idx, rng, edge_swaps, threshold_mutations)
    candidate.path = f"{arm}_candidate_{idx:04d}"
    meta.arm = arm
    meta.label = candidate.path
    return candidate, meta


def make_candidates(args, source: gpu_eval.CheckpointArrays, arm: str) -> tuple[list[gpu_eval.CheckpointArrays], list[d10k.ProposalMeta]]:
    arm_seed = args.seed + d10o.stable_arm_seed(arm)
    base = build_start(source, arm, args.h, args.edge_count, arm_seed)
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
    return candidates, metas


def control_targets(control_type: str, hot_to_idx: np.ndarray, bigram: np.ndarray, unigram: np.ndarray, seed: int):
    if control_type == "real":
        return hot_to_idx, bigram, unigram
    if control_type == "random_label":
        return d10k.shuffle_metric_targets(hot_to_idx, bigram, unigram, seed)
    rng = np.random.default_rng(seed)
    if control_type == "random_bigram":
        perm = rng.permutation(bigram.shape[0])
        return hot_to_idx, bigram[perm][:, perm], unigram
    if control_type == "unigram_decoy":
        return hot_to_idx, np.tile(unigram.reshape(1, -1), (bigram.shape[0], 1)).astype(np.float32), unigram
    if control_type == "projection_shuffle":
        return hot_to_idx, bigram, unigram
    raise ValueError(f"unknown control_type: {control_type}")


def shuffle_projection(checkpoints: list[gpu_eval.CheckpointArrays], seed: int) -> list[gpu_eval.CheckpointArrays]:
    rng = np.random.default_rng(seed)
    row_perm = rng.permutation(checkpoints[0].projection.weights.shape[0])
    col_perm = rng.permutation(checkpoints[0].projection.weights.shape[1])
    shuffled: list[gpu_eval.CheckpointArrays] = []
    for ckpt in checkpoints:
        weights = ckpt.projection.weights[row_perm][:, col_perm].copy()
        projection = gpu_eval.ProjectionArrays(weights, ckpt.projection.input_dim, ckpt.projection.output_classes)
        shuffled.append(clone_with_projection(ckpt, ckpt.path, projection))
    return shuffled


def eval_candidate_set(
    args,
    candidates: list[gpu_eval.CheckpointArrays],
    metas: list[d10k.ProposalMeta],
    table,
    pair_ids,
    hot_to_idx,
    bigram,
    unigram,
    device,
    control_type: str,
) -> tuple[list[dict], dict]:
    seed = args.seed + d10o.stable_arm_seed(control_type) + 900071
    eval_hot, eval_bigram, eval_unigram = control_targets(control_type, hot_to_idx, bigram, unigram, seed)
    eval_candidates = shuffle_projection(candidates, seed) if control_type == "projection_shuffle" else candidates
    metrics, elapsed_s, peak_mb = d10j.evaluate_metrics_sparse(
        eval_candidates,
        table,
        pair_ids,
        eval_hot,
        eval_bigram,
        eval_unigram,
        args.eval_len,
        parse_int_csv(args.eval_seeds),
        device,
    )
    rows: list[dict] = []
    counts: dict[str, int] = {}
    max_noop_abs = 0.0
    for idx, meta in enumerate(metas, start=1):
        row = d10k.row_from_delta(meta, metrics, 0, idx)
        row["control_type"] = control_type
        row["safe"] = row["class"] == "POSITIVE_SAFE"
        rows.append(row)
        counts[row["class"]] = counts.get(row["class"], 0) + 1
        if meta.candidate_idx == 0:
            max_noop_abs = max(abs(float(row[f"{metric}_delta"])) for metric in METRICS)
    summary = {
        "control_type": control_type,
        "candidate_count": len(rows),
        "positive_safe": counts.get("POSITIVE_SAFE", 0),
        "positive_unsafe": counts.get("POSITIVE_UNSAFE", 0),
        "echo_trap": counts.get("ECHO_TRAP", 0),
        "no_signal": counts.get("NO_SIGNAL", 0),
        "safe_rate": counts.get("POSITIVE_SAFE", 0) / max(1, len(rows) - 1),
        "max_noop_abs_delta": max_noop_abs,
        "elapsed_s": elapsed_s,
        "peak_mb": peak_mb,
    }
    return rows, summary


def arm_verdict(real_summary: dict, control_summaries: list[dict], args) -> dict:
    real_rate = real_summary["safe_rate"]
    max_control = max((s["safe_rate"] for s in control_summaries), default=0.0)
    adjusted = real_rate - max_control
    if any(s["max_noop_abs_delta"] > args.noop_tolerance for s in [real_summary, *control_summaries]):
        verdict = "NOOP_FAIL"
    elif max_control > 0.25:
        verdict = "SEMANTIC_FAIL"
    elif real_rate >= 0.25 and max_control <= 0.10 and adjusted >= 0.20:
        verdict = "SEMANTIC_PASS"
    elif real_rate >= 0.20 and adjusted > 0.0:
        verdict = "WEAK_PASS"
    else:
        verdict = "NO_SIGNAL"
    return {
        "arm_verdict": verdict,
        "real_safe_rate": real_rate,
        "max_control_safe_rate": max_control,
        "control_adjusted_safe_rate": adjusted,
    }


def write_phase_report(out: Path, run_summary: dict) -> None:
    lines = [
        "# D10p Semantic Projection Hardening",
        "",
        f"Verdict: `{run_summary['verdict']}`",
        "",
        "## Setup",
        "",
        f"- H: `{run_summary['setup']['h']}`",
        f"- active edges: `{run_summary['setup']['edge_count']}`",
        f"- eval_len: `{run_summary['setup']['eval_len']}`",
        f"- eval_seeds: `{','.join(str(s) for s in run_summary['setup']['eval_seeds'])}`",
        "",
        "## Arm Summary",
        "",
        "| arm | verdict | real safe | max control safe | adjusted |",
        "|---|---|---:|---:|---:|",
    ]
    for arm in run_summary["arms"]:
        lines.append(
            f"| {arm['arm']} | `{arm['arm_verdict']}` | {arm['real_safe_rate']:.3f} | "
            f"{arm['max_control_safe_rate']:.3f} | {arm['control_adjusted_safe_rate']:.3f} |"
        )
    lines.extend(
        [
            "",
            "GPU-only D10p output is not promotion evidence. A later D10q confirm is required for any pass.",
            "",
            "## Progress Map",
            "",
            "```text",
            "[1] H384 beta.8 generalist: DONE",
            "[2] causal mechanism: DONE",
            "[3] D10b seed replication: RUNNING",
            "[4] high-H raw structure: DONE",
            "[5] semantic projection hardening: D10p CURRENT",
            "[6] controlled high-H proof: D10q only if D10p passes",
            "```",
        ]
    )
    (out / "D10P_SEMANTIC_PROJECTION_HARDENING_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_hardening(args) -> dict:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    target = gpu_eval.load_checkpoint(Path(args.target))
    table, pair_ids, hot_to_idx, bigram, unigram, _ = d10j.load_real_inputs(args)
    device = torch.device(args.device)
    all_rows: list[dict] = []
    arm_rows: list[dict] = []
    control_types = ["real", *CONTROL_TYPES]
    for arm in parse_csv(args.arms):
        candidates, metas = make_candidates(args, target, arm)
        summaries: dict[str, dict] = {}
        for control_type in control_types:
            rows, summary = eval_candidate_set(
                args,
                candidates,
                metas,
                table,
                pair_ids,
                hot_to_idx,
                bigram,
                unigram,
                device,
                control_type,
            )
            for row in rows:
                row["arm"] = arm
            all_rows.extend(rows)
            summaries[control_type] = summary
        verdict = arm_verdict(summaries["real"], [summaries[c] for c in CONTROL_TYPES], args)
        arm_row = {
            "arm": arm,
            **verdict,
            "real_positive_safe": summaries["real"]["positive_safe"],
            "random_label_safe_rate": summaries["random_label"]["safe_rate"],
            "random_bigram_safe_rate": summaries["random_bigram"]["safe_rate"],
            "unigram_decoy_safe_rate": summaries["unigram_decoy"]["safe_rate"],
            "projection_shuffle_safe_rate": summaries["projection_shuffle"]["safe_rate"],
            "max_noop_abs_delta": max(s["max_noop_abs_delta"] for s in summaries.values()),
        }
        arm_rows.append(arm_row)
        print(
            f"D10p arm={arm} verdict={arm_row['arm_verdict']} real={arm_row['real_safe_rate']:.3f} "
            f"max_control={arm_row['max_control_safe_rate']:.3f}",
            flush=True,
        )
    run_verdict = "D10P_SEMANTIC_FAIL"
    if any(row["arm_verdict"] == "SEMANTIC_PASS" for row in arm_rows):
        run_verdict = "D10P_SEMANTIC_PASS"
    elif any(row["arm_verdict"] == "WEAK_PASS" for row in arm_rows):
        run_verdict = "D10P_WEAK_PASS"
    fieldnames = [
        "arm",
        "control_type",
        "candidate_idx",
        "label",
        "class",
        "safe",
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
    write_csv(out / "semantic_candidates.csv", all_rows, fieldnames)
    write_csv(
        out / "semantic_arm_summary.csv",
        arm_rows,
        [
            "arm",
            "arm_verdict",
            "real_safe_rate",
            "max_control_safe_rate",
            "control_adjusted_safe_rate",
            "real_positive_safe",
            "random_label_safe_rate",
            "random_bigram_safe_rate",
            "unigram_decoy_safe_rate",
            "projection_shuffle_safe_rate",
            "max_noop_abs_delta",
        ],
    )
    run_summary = {
        "verdict": run_verdict,
        "setup": {
            "h": args.h,
            "edge_count": args.edge_count,
            "usage": d10j.edge_usage(args.h, args.edge_count),
            "eval_len": args.eval_len,
            "eval_seeds": parse_int_csv(args.eval_seeds),
            "proposals_per_arm": args.proposals_per_arm,
            "arms": parse_csv(args.arms),
            "controls": CONTROL_TYPES,
        },
        "arms": arm_rows,
    }
    (out / "run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    write_phase_report(out, run_summary)
    return run_summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="output/releases/v5.0.0-beta.8/seed2042_improved_generalist_v1.ckpt")
    parser.add_argument("--packed", default="output/block_c_bytepair_champion/packed.bin")
    parser.add_argument("--corpus", default="instnct-core/tests/fixtures/alice_corpus.txt")
    parser.add_argument("--out", default="output/phase_d10p_semantic_projection_hardening_20260430")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--h", type=int, default=8192)
    parser.add_argument("--edge-count", type=int, default=100000)
    parser.add_argument("--eval-len", type=int, default=128)
    parser.add_argument("--eval-seeds", default="988001,988002")
    parser.add_argument("--arms", default="beta8_lifted_v2,motif_no_echo,projection_tiled,threshold_mid,threshold_high,block_local_projection,frozen_beta8_rows")
    parser.add_argument("--proposals-per-arm", type=int, default=16)
    parser.add_argument("--edge-swaps", type=int, default=16)
    parser.add_argument("--threshold-mutations", type=int, default=16)
    parser.add_argument("--noop-tolerance", type=float, default=1e-9)
    parser.add_argument("--seed", type=int, default=20260430)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    summary = run_hardening(args)
    print(json.dumps({"verdict": summary["verdict"], "setup": summary["setup"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
