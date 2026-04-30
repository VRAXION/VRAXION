#!/usr/bin/env python3
"""D10s H384 wiring-prior smoke/confirm.

Scratch/prototype only. This tests whether simple edge+threshold priors make
the beta.8-style signal less seed-lottery dependent. It does not promote or
release checkpoints.
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
import d10o_high_h_projection_start_gate as d10o
import d10r_hardened_eval as d10r


ARMS = [
    "random_sparse_baseline",
    "beta8_degree_matched",
    "beta8_threshold_histogram",
    "edge_threshold_coadapted",
    "motif_biased",
    "block_local_sparse",
    "supermask_score_init",
    "rigl_lite_rewire",
]
STRICT_GATES = {
    "smooth": 0.0120,
    "accuracy": 0.0020,
    "echo_abs": 0.0010,
    "unigram": 0.0,
}


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_int_csv(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


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


def edge_pairs(ckpt: gpu_eval.CheckpointArrays) -> list[tuple[int, int]]:
    return [(int(s), int(t)) for s, t in zip(ckpt.network.sources, ckpt.network.targets) if int(s) != int(t)]


def replace_edge_arrays(ckpt: gpu_eval.CheckpointArrays, edges: list[tuple[int, int]], label: str) -> gpu_eval.CheckpointArrays:
    out = clone_checkpoint(ckpt, label)
    out.network.sources = np.asarray([s for s, _ in edges], dtype=np.int64)
    out.network.targets = np.asarray([t for _, t in edges], dtype=np.int64)
    out.path = label
    return out


def random_edge(h: int, rng: random.Random, block: tuple[int, int] | None = None) -> tuple[int, int]:
    if block is None:
        s = rng.randrange(h)
        t = rng.randrange(h - 1)
        if t >= s:
            t += 1
        return s, t
    lo, hi = block
    s = rng.randrange(lo, hi)
    t = rng.randrange(lo, hi - 1)
    if t >= s:
        t += 1
    return s, t


def beta8_weighted_edge(reference: gpu_eval.CheckpointArrays, rng: random.Random) -> tuple[int, int]:
    h = reference.network.h
    srcs = reference.network.sources.astype(int).tolist()
    tgts = reference.network.targets.astype(int).tolist()
    for _ in range(20):
        s = rng.choice(srcs)
        t = rng.choice(tgts)
        if 0 <= s < h and 0 <= t < h and s != t:
            return int(s), int(t)
    return random_edge(h, rng)


def motif_edge(base_edges: list[tuple[int, int]], h: int, rng: random.Random) -> tuple[int, int]:
    outgoing: dict[int, list[int]] = {}
    for s, t in base_edges:
        outgoing.setdefault(s, []).append(t)
    mids = [node for node in outgoing if outgoing.get(node)]
    for _ in range(30):
        s = rng.choice(mids) if mids else rng.randrange(h)
        mid = rng.choice(outgoing.get(s, [rng.randrange(h)]))
        tail = outgoing.get(mid)
        if tail:
            t = rng.choice(tail)
            if s != t:
                return s, t
    return random_edge(h, rng)


def supermask_edge(h: int, rng: random.Random, salt: int) -> tuple[int, int]:
    best = None
    best_score = -1
    for _ in range(256):
        s, t = random_edge(h, rng)
        score = ((s + 1) * 1103515245 + (t + 1) * 12345 + salt) & 0x7FFFFFFF
        if score > best_score:
            best_score = score
            best = (s, t)
    return best or random_edge(h, rng)


def mutate_edges(
    source: gpu_eval.CheckpointArrays,
    reference: gpu_eval.CheckpointArrays,
    arm: str,
    proposal_idx: int,
    rng: random.Random,
    swaps: int,
) -> list[tuple[int, int]]:
    h = source.network.h
    edges = edge_pairs(source)
    existing = set(edges)
    out_degree: dict[int, int] = {}
    for s, _ in edges:
        out_degree[s] = out_degree.get(s, 0) + 1
    high_out = sorted(out_degree, key=out_degree.get, reverse=True)[: max(1, min(32, len(out_degree)))]
    block_count = 8
    block_size = max(2, h // block_count)
    for _ in range(swaps):
        if not edges:
            break
        drop_idx = rng.randrange(len(edges))
        old = edges[drop_idx]
        existing.discard(old)
        for _attempt in range(100):
            if arm == "beta8_degree_matched":
                new = beta8_weighted_edge(reference, rng)
            elif arm == "motif_biased":
                new = motif_edge(edges, h, rng)
            elif arm == "block_local_sparse":
                block_idx = rng.randrange(block_count)
                lo = block_idx * block_size
                hi = min(h, lo + block_size)
                new = random_edge(h, rng, (lo, hi))
            elif arm == "supermask_score_init":
                new = supermask_edge(h, rng, proposal_idx + _attempt)
            elif arm == "rigl_lite_rewire":
                s = rng.choice(high_out) if high_out else rng.randrange(h)
                t = rng.randrange(h - 1)
                if t >= s:
                    t += 1
                new = (s, t)
            elif arm == "edge_threshold_coadapted":
                new = beta8_weighted_edge(reference, rng) if rng.random() < 0.7 else motif_edge(edges, h, rng)
            else:
                new = random_edge(h, rng)
            if new[0] != new[1] and new not in existing:
                edges[drop_idx] = new
                existing.add(new)
                break
        else:
            existing.add(old)
    return edges


def mutate_thresholds(
    source: gpu_eval.CheckpointArrays,
    reference: gpu_eval.CheckpointArrays,
    arm: str,
    rng: random.Random,
    count: int,
) -> np.ndarray:
    threshold = source.network.threshold.copy()
    h = len(threshold)
    ref_hist = reference.network.threshold.astype(np.int16)
    for _ in range(count):
        idx = rng.randrange(h)
        if arm in {"beta8_threshold_histogram", "edge_threshold_coadapted"}:
            threshold[idx] = int(rng.choice(ref_hist.tolist()))
        elif arm == "rigl_lite_rewire":
            threshold[idx] = max(0, min(15, int(threshold[idx]) + rng.choice([-2, -1, 1, 2])))
        else:
            threshold[idx] = rng.randrange(0, 16)
    return threshold.astype(np.int16)


def make_candidate(
    seed_ckpt: gpu_eval.CheckpointArrays,
    reference: gpu_eval.CheckpointArrays,
    seed_label: str,
    arm: str,
    proposal_idx: int,
    args,
) -> gpu_eval.CheckpointArrays:
    rng = random.Random(args.seed + d10o.stable_arm_seed(seed_label + arm) + proposal_idx * 1009)
    label = f"{seed_label}_{arm}_{proposal_idx:04d}"
    candidate = clone_checkpoint(seed_ckpt, label)
    edge_swaps = args.edge_swaps
    threshold_mutations = args.threshold_mutations
    if arm in {"beta8_degree_matched", "edge_threshold_coadapted", "motif_biased", "block_local_sparse", "supermask_score_init", "rigl_lite_rewire", "random_sparse_baseline"}:
        edges = mutate_edges(candidate, reference, arm, proposal_idx, rng, edge_swaps)
        candidate = replace_edge_arrays(candidate, edges, label)
    if arm in {"beta8_threshold_histogram", "edge_threshold_coadapted", "rigl_lite_rewire", "random_sparse_baseline"}:
        candidate.network.threshold = mutate_thresholds(candidate, reference, arm, rng, threshold_mutations)
    return candidate


def classify_candidate(row: dict) -> tuple[str, int]:
    misses = 0
    near = 0
    if row["smooth_delta"] < STRICT_GATES["smooth"]:
        misses += 1
        near += int(row["smooth_delta"] >= STRICT_GATES["smooth"] * 0.75)
    if row["accuracy_delta"] < STRICT_GATES["accuracy"]:
        misses += 1
        near += int(row["accuracy_delta"] >= STRICT_GATES["accuracy"] * 0.75)
    if abs(row["echo_delta"]) > STRICT_GATES["echo_abs"]:
        misses += 1
        near += int(abs(row["echo_delta"]) <= STRICT_GATES["echo_abs"] * 1.25)
    if row["unigram_delta"] < STRICT_GATES["unigram"]:
        misses += 1
        near += int(row["unigram_delta"] >= -0.001)
    if row["hardened_selectivity"] <= 0:
        misses += 1
    if misses == 0:
        return "STRICT_TRUSTED", misses
    if misses == 1 and near == 1 and row["hardened_selectivity"] > 0:
        return "NEAR_TRUSTED", misses
    if row["mo_delta"] > 0 and row["hardened_selectivity"] > 0:
        return "WEAK_TRUSTED", misses
    return "REJECT", misses


def evaluate_candidates(seed_label: str, baseline, candidates, args, table, pair_ids, hot_to_idx, bigram, unigram, device):
    controls = parse_csv(args.controls)
    expanded_controls = d10r.expand_controls(controls, args.control_repeats)
    eval_seeds = parse_int_csv(args.eval_seeds)
    all_ckpts = [baseline, *candidates]
    rows = []
    for control in [{"label": "real", "base": "real", "repeat": 0}, *expanded_controls]:
        control_seed = args.seed + d10o.stable_arm_seed(seed_label + control["label"]) + 1103
        for eval_seed in eval_seeds:
            metrics = d10r.evaluate_seed_control(
                all_ckpts,
                table,
                pair_ids,
                hot_to_idx,
                bigram,
                unigram,
                args.eval_len,
                eval_seed,
                control["base"],
                device,
                control_seed,
            )
            base_scores = {metric: float(metrics[metric][0]) for metric in d10r.METRICS}
            for idx, candidate in enumerate(candidates, start=1):
                scores = {metric: float(metrics[metric][idx]) for metric in d10r.METRICS}
                delta = {metric: scores[metric] - base_scores[metric] for metric in d10r.METRICS}
                rows.append(
                    {
                        "seed_label": seed_label,
                        "candidate_label": candidate.path,
                        "control_type": control["label"],
                        "eval_seed": eval_seed,
                        "smooth_delta": delta["smooth"],
                        "accuracy_delta": delta["accuracy"],
                        "echo_delta": delta["echo"],
                        "unigram_delta": delta["unigram"],
                        "mo_delta": d10r.mo_score_from_delta(delta),
                    }
                )
    return rows


def summarize_candidates(eval_rows: list[dict], candidate_specs: dict[str, dict], bootstrap_samples: int, alpha: float, seed: int) -> list[dict]:
    out = []
    labels = sorted({row["candidate_label"] for row in eval_rows})
    for label in labels:
        rows = [row for row in eval_rows if row["candidate_label"] == label]
        real_rows = [row for row in rows if row["control_type"] == "real"]
        if not real_rows:
            continue
        controls_by_seed: dict[int, list[float]] = {}
        for row in rows:
            if row["control_type"] == "real":
                continue
            controls_by_seed.setdefault(int(row["eval_seed"]), []).append(float(row["mo_delta"]))
        real_by_seed = {int(row["eval_seed"]): float(row["mo_delta"]) for row in real_rows}
        worst_margins = [
            real_by_seed[s] - max(vals)
            for s, vals in controls_by_seed.items()
            if s in real_by_seed and vals
        ]
        summary = {
            **candidate_specs[label],
            "smooth_delta": float(np.mean([float(r["smooth_delta"]) for r in real_rows])),
            "accuracy_delta": float(np.mean([float(r["accuracy_delta"]) for r in real_rows])),
            "echo_delta": float(np.mean([float(r["echo_delta"]) for r in real_rows])),
            "unigram_delta": float(np.mean([float(r["unigram_delta"]) for r in real_rows])),
            "mo_delta": float(np.mean([float(r["mo_delta"]) for r in real_rows])),
            "hardened_selectivity": float(np.mean(worst_margins)) if worst_margins else 0.0,
        }
        ci = d10r.bootstrap_ci(
            worst_margins,
            bootstrap_samples,
            alpha,
            seed + d10o.stable_arm_seed(label + "selectivity"),
        )
        summary["hardened_selectivity_ci_low"] = ci[0]
        summary["hardened_selectivity_ci_high"] = ci[1]
        candidate_class, misses = classify_candidate(summary)
        summary["candidate_class"] = candidate_class
        summary["gate_misses"] = misses
        out.append(summary)
    out.sort(
        key=lambda row: (
            {"STRICT_TRUSTED": 0, "NEAR_TRUSTED": 1, "WEAK_TRUSTED": 2, "REJECT": 3}[row["candidate_class"]],
            -row["hardened_selectivity"],
            -row["mo_delta"],
        )
    )
    return out


def run_sweep(args) -> dict:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    gpu_eval.MAX_CHARGE = args.max_charge
    table, pair_ids, hot_to_idx, bigram, unigram, _ = d10j.load_real_inputs(args)
    reference = gpu_eval.load_checkpoint(Path(args.reference))
    checkpoint_paths = parse_csv(args.checkpoints)
    arms = parse_csv(args.arms)
    eval_rows = []
    candidate_specs = {}
    started = time.perf_counter()
    for checkpoint_path in checkpoint_paths:
        baseline = gpu_eval.load_checkpoint(Path(checkpoint_path))
        seed_label = d10r.checkpoint_label(Path(checkpoint_path))
        for arm in arms:
            proposals = [
                make_candidate(baseline, reference, seed_label, arm, idx, args)
                for idx in range(1, args.proposals_per_arm + 1)
            ]
            for idx, candidate in enumerate(proposals, start=1):
                candidate_specs[candidate.path] = {
                    "seed_label": seed_label,
                    "arm": arm,
                    "proposal_idx": idx,
                    "candidate_label": candidate.path,
                }
            eval_rows.extend(evaluate_candidates(seed_label, baseline, proposals, args, table, pair_ids, hot_to_idx, bigram, unigram, device))
            print(f"D10s seed={seed_label} arm={arm} proposals={len(proposals)} done", flush=True)
    candidate_rows = summarize_candidates(eval_rows, candidate_specs, args.bootstrap_samples, args.alpha, args.seed)
    arm_rows = []
    for arm in arms:
        arm_candidates = [row for row in candidate_rows if row["arm"] == arm]
        arm_rows.append(
            {
                "arm": arm,
                "count": len(arm_candidates),
                "strict_count": sum(row["candidate_class"] == "STRICT_TRUSTED" for row in arm_candidates),
                "near_count": sum(row["candidate_class"] == "NEAR_TRUSTED" for row in arm_candidates),
                "best_class": arm_candidates[0]["candidate_class"] if arm_candidates else "NONE",
                "best_selectivity": arm_candidates[0]["hardened_selectivity"] if arm_candidates else 0.0,
                "best_mo_delta": arm_candidates[0]["mo_delta"] if arm_candidates else 0.0,
            }
        )
    non_seed2042_signal = [
        row for row in candidate_rows
        if row["seed_label"] != "seed_2042" and row["candidate_class"] in {"STRICT_TRUSTED", "NEAR_TRUSTED"}
    ]
    seed2042_signal = [
        row for row in candidate_rows
        if row["seed_label"] == "seed_2042" and row["candidate_class"] in {"STRICT_TRUSTED", "NEAR_TRUSTED"}
    ]
    if non_seed2042_signal:
        verdict = "D10S_REPLICABLE_WIRING_PRIOR_SIGNAL"
    elif seed2042_signal:
        verdict = "D10S_SEED2042_ONLY"
    else:
        verdict = "D10S_NO_TRUSTED_SIGNAL"
    run_summary = {
        "verdict": verdict,
        "elapsed_s": time.perf_counter() - started,
        "setup": {
            "mode": args.mode,
            "eval_len": args.eval_len,
            "eval_seeds": parse_int_csv(args.eval_seeds),
            "control_repeats": args.control_repeats,
            "proposals_per_arm": args.proposals_per_arm,
            "arms": arms,
            "checkpoints": checkpoint_paths,
            "max_charge": args.max_charge,
        },
        "top_candidates": candidate_rows[:10],
        "arm_summary": arm_rows,
    }
    write_csv(out / "d10s_eval_rows.csv", eval_rows, ["seed_label", "candidate_label", "control_type", "eval_seed", "smooth_delta", "accuracy_delta", "echo_delta", "unigram_delta", "mo_delta"])
    write_csv(out / "candidate_summary.csv", candidate_rows, ["seed_label", "arm", "proposal_idx", "candidate_label", "candidate_class", "gate_misses", "smooth_delta", "accuracy_delta", "echo_delta", "unigram_delta", "mo_delta", "hardened_selectivity", "hardened_selectivity_ci_low", "hardened_selectivity_ci_high"])
    write_csv(out / "arm_summary.csv", arm_rows, ["arm", "count", "strict_count", "near_count", "best_class", "best_selectivity", "best_mo_delta"])
    (out / "run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    write_report(out, run_summary)
    return run_summary


def write_report(out: Path, summary: dict) -> None:
    lines = [
        "# D10s Wiring Prior Sweep Report",
        "",
        f"Verdict: `{summary['verdict']}`",
        "",
        "## Top Candidates",
        "",
        "| seed | arm | proposal | class | selectivity | mo_delta |",
        "|---|---|---:|---|---:|---:|",
    ]
    for row in summary["top_candidates"][:10]:
        lines.append(
            f"| {row['seed_label']} | {row['arm']} | {row['proposal_idx']} | {row['candidate_class']} | "
            f"{row['hardened_selectivity']:.6f} | {row['mo_delta']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Progress Map",
            "",
            "```text",
            "[4] D10r-v2 evaluator trust: required before D10s evidence is actionable",
            "[5] D10s wiring-prior sweep: CURRENT",
            "    |-- non-seed2042 trusted signal -> H512 pilot can be planned",
            "    '-- no trusted signal -> redesign wiring priors, no H512 yet",
            "```",
        ]
    )
    (out / "D10S_WIRING_PRIOR_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "confirm"], default="smoke")
    parser.add_argument("--reference", default="output/releases/v5.0.0-beta.8/seed2042_improved_generalist_v1.ckpt")
    parser.add_argument(
        "--checkpoints",
        default=",".join(
            [
                "output/phase_d7_operator_bandit_20260427/H_384/D7_BASELINE/seed_42/final.ckpt",
                "output/phase_d7_operator_bandit_20260427/H_384/D7_BASELINE/seed_1042/final.ckpt",
                "output/phase_d7_operator_bandit_20260427/H_384/D7_BASELINE/seed_2042/final.ckpt",
                "output/phase_d7_operator_bandit_20260427/H_384/D7_BASELINE/seed_3042/final.ckpt",
                "output/phase_d7_operator_bandit_20260427/H_384/D7_BASELINE/seed_4042/final.ckpt",
            ]
        ),
    )
    parser.add_argument("--packed", default="output/block_c_bytepair_champion/packed.bin")
    parser.add_argument("--corpus", default="instnct-core/tests/fixtures/alice_corpus.txt")
    parser.add_argument("--out", default="output/phase_d10s_wiring_prior_sweep_20260430")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--eval-len", type=int, default=1000)
    parser.add_argument("--eval-seeds", default="970001,970002,970003,970004,970005,970006,970007,970008")
    parser.add_argument("--controls", default="random_label,unigram_decoy,state_shuffle,no_network_random_state")
    parser.add_argument("--control-repeats", type=int, default=2)
    parser.add_argument("--proposals-per-arm", type=int, default=32)
    parser.add_argument("--edge-swaps", type=int, default=24)
    parser.add_argument("--threshold-mutations", type=int, default=24)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--max-charge", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260430)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    summary = run_sweep(args)
    print(json.dumps({"verdict": summary["verdict"], "elapsed_s": summary["elapsed_s"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
