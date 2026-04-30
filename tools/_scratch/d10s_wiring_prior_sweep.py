#!/usr/bin/env python3
"""D10s/D10u H384 state-anchored wiring-prior smoke/confirm.

Scratch/prototype only. This tests whether simple edge+threshold priors can
produce candidates that improve the real task while beating D10r-v8 artifact
controls, especially state_shuffle_shared. It does not promote or release
checkpoints.
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
import d10h_dense_crystallize_probe as d10h
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
DEFAULT_STATE_ANCHOR_CONTROLS = [
    "random_projection_null",
    "state_shuffle_shared",
    "state_shuffle_projection_consistent",
    "no_network_random_state",
]


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


def classify_candidate(row: dict, selectivity_gate: str) -> tuple[str, int]:
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
    if selectivity_gate == "ci":
        selectivity_pass = row["hardened_selectivity_ci_low"] > 0.0
        selectivity_near = row["hardened_selectivity"] > 0.0
    else:
        selectivity_pass = row["hardened_selectivity"] > 0.0
        selectivity_near = selectivity_pass
    if not selectivity_pass:
        misses += 1
        near += int(selectivity_near)
    if misses == 0:
        return "STRICT_TRUSTED", misses
    if misses == 1 and near >= 1 and row["hardened_selectivity"] > 0:
        return "NEAR_TRUSTED", misses
    if row["mo_delta"] > 0 and selectivity_pass:
        return "WEAK_STATE_ANCHORED", misses
    if row["mo_delta"] > 0 and row["hardened_selectivity"] > 0:
        return "WEAK_SIGNAL", misses
    return "REJECT", misses


def candidate_class_rank(candidate_class: str) -> int:
    return {
        "STRICT_TRUSTED": 0,
        "NEAR_TRUSTED": 1,
        "WEAK_STATE_ANCHORED": 2,
        "WEAK_SIGNAL": 3,
        "REJECT": 4,
    }[candidate_class]


def gate_progress_score(row: dict) -> float:
    smooth = max(0.0, min(1.0, float(row["smooth_delta"]) / STRICT_GATES["smooth"]))
    accuracy = max(0.0, min(1.0, float(row["accuracy_delta"]) / STRICT_GATES["accuracy"]))
    echo = max(0.0, 1.0 - abs(float(row["echo_delta"])) / STRICT_GATES["echo_abs"])
    unigram = max(0.0, min(1.0, (float(row["unigram_delta"]) + 0.001) / 0.001))
    selectivity = max(0.0, min(1.0, float(row.get("hardened_selectivity_ci_low", 0.0)) / 0.002))
    return smooth + accuracy + echo + unigram + selectivity + float(row["mo_delta"])


def candidate_sort_key(row: dict) -> tuple[int, float, float, float]:
    return (
        candidate_class_rank(row["candidate_class"]),
        -float(row.get("ladder_score", gate_progress_score(row))),
        -float(row["hardened_selectivity"]),
        -float(row["mo_delta"]),
    )


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


def summarize_candidates(
    eval_rows: list[dict],
    candidate_specs: dict[str, dict],
    bootstrap_samples: int,
    alpha: float,
    seed: int,
    selectivity_gate: str,
) -> list[dict]:
    out = []
    labels = sorted({row["candidate_label"] for row in eval_rows})
    for label in labels:
        rows = [row for row in eval_rows if row["candidate_label"] == label]
        real_rows = [row for row in rows if row["control_type"] == "real"]
        if not real_rows:
            continue
        controls_by_seed: dict[int, list[tuple[float, str]]] = {}
        diagnostic_control_count = 0
        for row in rows:
            if row["control_type"] == "real":
                continue
            if d10r.is_diagnostic_control(str(row["control_type"])):
                diagnostic_control_count += 1
                continue
            controls_by_seed.setdefault(int(row["eval_seed"]), []).append(
                (float(row["mo_delta"]), str(row["control_type"]))
            )
        real_by_seed = {int(row["eval_seed"]): float(row["mo_delta"]) for row in real_rows}
        worst_records = []
        for eval_seed, vals in controls_by_seed.items():
            if eval_seed not in real_by_seed or not vals:
                continue
            worst_control_value, worst_control_type = max(vals, key=lambda item: item[0])
            worst_records.append(
                {
                    "eval_seed": eval_seed,
                    "control_type": worst_control_type,
                    "margin": real_by_seed[eval_seed] - worst_control_value,
                }
            )
        worst_margins = [float(record["margin"]) for record in worst_records]
        worst_record = min(worst_records, key=lambda record: float(record["margin"])) if worst_records else None
        summary = {
            **candidate_specs[label],
            "smooth_delta": float(np.mean([float(r["smooth_delta"]) for r in real_rows])),
            "accuracy_delta": float(np.mean([float(r["accuracy_delta"]) for r in real_rows])),
            "echo_delta": float(np.mean([float(r["echo_delta"]) for r in real_rows])),
            "unigram_delta": float(np.mean([float(r["unigram_delta"]) for r in real_rows])),
            "mo_delta": float(np.mean([float(r["mo_delta"]) for r in real_rows])),
            "hardened_selectivity": float(np.mean(worst_margins)) if worst_margins else 0.0,
            "artifact_control_count": sum(len(vals) for vals in controls_by_seed.values()),
            "diagnostic_control_count": diagnostic_control_count,
            "worst_artifact_control": str(worst_record["control_type"]) if worst_record else "",
            "worst_artifact_eval_seed": int(worst_record["eval_seed"]) if worst_record else 0,
            "worst_artifact_margin": float(worst_record["margin"]) if worst_record else 0.0,
        }
        ci = d10r.bootstrap_ci(
            worst_margins,
            bootstrap_samples,
            alpha,
            seed + d10o.stable_arm_seed(label + "selectivity"),
        )
        summary["hardened_selectivity_ci_low"] = ci[0]
        summary["hardened_selectivity_ci_high"] = ci[1]
        summary["selectivity_gate"] = selectivity_gate
        summary["state_anchor_pass"] = bool(
            summary["hardened_selectivity_ci_low"] > 0.0
            if selectivity_gate == "ci"
            else summary["hardened_selectivity"] > 0.0
        )
        candidate_class, misses = classify_candidate(summary, selectivity_gate)
        summary["candidate_class"] = candidate_class
        summary["gate_misses"] = misses
        summary["ladder_score"] = gate_progress_score(summary)
        out.append(summary)
    out.sort(key=candidate_sort_key)
    return out


def write_candidate_summary(path: Path, candidate_rows: list[dict]) -> None:
    write_csv(
        path,
        candidate_rows,
        [
            "seed_label",
            "arm",
            "ladder_round",
            "proposal_idx",
            "candidate_label",
            "checkpoint_path",
            "candidate_class",
            "gate_misses",
            "accepted",
            "accept_reason",
            "state_anchor_pass",
            "selectivity_gate",
            "ladder_score",
            "smooth_delta",
            "accuracy_delta",
            "echo_delta",
            "unigram_delta",
            "mo_delta",
            "hardened_selectivity",
            "hardened_selectivity_ci_low",
            "hardened_selectivity_ci_high",
            "worst_artifact_control",
            "worst_artifact_eval_seed",
            "worst_artifact_margin",
            "artifact_control_count",
            "diagnostic_control_count",
        ],
    )


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
    candidate_rows = summarize_candidates(
        eval_rows,
        candidate_specs,
        args.bootstrap_samples,
        args.alpha,
        args.seed,
        args.selectivity_gate,
    )
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
            "controls": parse_csv(args.controls),
            "selectivity_gate": args.selectivity_gate,
        },
        "top_candidates": candidate_rows[:10],
        "arm_summary": arm_rows,
    }
    write_csv(out / "d10s_eval_rows.csv", eval_rows, ["seed_label", "candidate_label", "control_type", "eval_seed", "smooth_delta", "accuracy_delta", "echo_delta", "unigram_delta", "mo_delta"])
    write_candidate_summary(out / "candidate_summary.csv", candidate_rows)
    write_csv(out / "arm_summary.csv", arm_rows, ["arm", "count", "strict_count", "near_count", "best_class", "best_selectivity", "best_mo_delta"])
    (out / "run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    write_report(out, run_summary)
    return run_summary


def run_ladder(args) -> dict:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "candidates").mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    gpu_eval.MAX_CHARGE = args.max_charge
    table, pair_ids, hot_to_idx, bigram, unigram, _ = d10j.load_real_inputs(args)
    reference = gpu_eval.load_checkpoint(Path(args.reference))
    checkpoint_paths = parse_csv(args.checkpoints)
    arms = parse_csv(args.arms)
    started = time.perf_counter()
    all_eval_rows: list[dict] = []
    all_candidate_rows: list[dict] = []
    ladder_rows: list[dict] = []
    candidate_by_label: dict[str, gpu_eval.CheckpointArrays] = {}

    for checkpoint_path in checkpoint_paths:
        baseline = gpu_eval.load_checkpoint(Path(checkpoint_path))
        seed_label = d10r.checkpoint_label(Path(checkpoint_path))
        for arm in arms:
            current = clone_checkpoint(baseline, f"{seed_label}_{arm}_start")
            current_score = -1e9
            no_accept_rounds = 0
            for round_idx in range(1, args.ladder_rounds + 1):
                candidate_specs = {}
                proposals = []
                for prop_idx in range(1, args.proposals_per_arm + 1):
                    proposal_key = round_idx * 10000 + prop_idx
                    candidate = make_candidate(current, reference, seed_label, arm, proposal_key, args)
                    candidate.path = f"{seed_label}_{arm}_r{round_idx:02d}_p{prop_idx:04d}"
                    candidate.meta["label"] = candidate.path
                    proposals.append(candidate)
                    candidate_by_label[candidate.path] = candidate
                    candidate_specs[candidate.path] = {
                        "seed_label": seed_label,
                        "arm": arm,
                        "ladder_round": round_idx,
                        "proposal_idx": prop_idx,
                        "candidate_label": candidate.path,
                    }
                eval_rows = evaluate_candidates(seed_label, baseline, proposals, args, table, pair_ids, hot_to_idx, bigram, unigram, device)
                all_eval_rows.extend(eval_rows)
                round_rows = summarize_candidates(
                    eval_rows,
                    candidate_specs,
                    args.bootstrap_samples,
                    args.alpha,
                    args.seed + d10o.stable_arm_seed(f"{seed_label}{arm}{round_idx}"),
                    args.selectivity_gate,
                )
                if not round_rows:
                    continue
                best = round_rows[0]
                best_score = float(best["ladder_score"])
                acceptable_class = best["candidate_class"] in {"STRICT_TRUSTED", "NEAR_TRUSTED", "WEAK_STATE_ANCHORED"}
                improves = best_score >= current_score + args.ladder_min_improvement
                accepted = acceptable_class and improves
                if accepted:
                    current = candidate_by_label[best["candidate_label"]]
                    current_score = best_score
                    no_accept_rounds = 0
                    accept_reason = "state_anchor_and_ladder_score_improved"
                else:
                    no_accept_rounds += 1
                    accept_reason = "reject_no_state_anchor_or_no_score_gain"
                for row in round_rows:
                    row["accepted"] = bool(row["candidate_label"] == best["candidate_label"] and accepted)
                    row["accept_reason"] = accept_reason if row["candidate_label"] == best["candidate_label"] else ""
                all_candidate_rows.extend(round_rows)
                ladder_rows.append(
                    {
                        "seed_label": seed_label,
                        "arm": arm,
                        "ladder_round": round_idx,
                        "best_candidate": best["candidate_label"],
                        "best_class": best["candidate_class"],
                        "best_ladder_score": best_score,
                        "best_mo_delta": best["mo_delta"],
                        "best_selectivity_ci_low": best["hardened_selectivity_ci_low"],
                        "best_smooth_delta": best["smooth_delta"],
                        "best_accuracy_delta": best["accuracy_delta"],
                        "best_unigram_delta": best["unigram_delta"],
                        "best_worst_artifact_control": best["worst_artifact_control"],
                        "accepted": accepted,
                        "accept_reason": accept_reason,
                    }
                )
                print(
                    f"D10u seed={seed_label} arm={arm} round={round_idx} "
                    f"best={best['candidate_class']} score={best_score:.6f} accepted={accepted}",
                    flush=True,
                )
                if no_accept_rounds >= args.ladder_patience:
                    break

    all_candidate_rows.sort(key=candidate_sort_key)
    for rank, row in enumerate(all_candidate_rows[: args.export_top], start=1):
        ckpt = candidate_by_label.get(row["candidate_label"])
        if ckpt is None:
            continue
        ckpt_path = out / "candidates" / f"top_{rank:02d}_{row['seed_label']}_{row['arm']}.ckpt"
        d10h.write_checkpoint(ckpt_path, ckpt, row["candidate_label"])
        row["checkpoint_path"] = str(ckpt_path)

    non_seed_near_or_strict = [
        row for row in all_candidate_rows
        if row["seed_label"] != "seed_2042" and row["candidate_class"] in {"STRICT_TRUSTED", "NEAR_TRUSTED"}
    ]
    seed_near_or_strict = [
        row for row in all_candidate_rows
        if row["seed_label"] == "seed_2042" and row["candidate_class"] in {"STRICT_TRUSTED", "NEAR_TRUSTED"}
    ]
    non_seed_weak = [
        row for row in all_candidate_rows
        if row["seed_label"] != "seed_2042" and row["candidate_class"] == "WEAK_STATE_ANCHORED"
    ]
    seed_weak = [
        row for row in all_candidate_rows
        if row["seed_label"] == "seed_2042" and row["candidate_class"] == "WEAK_STATE_ANCHORED"
    ]
    if non_seed_near_or_strict:
        verdict = "D10U_NON_SEED_NEAR_OR_STRICT_SIGNAL"
    elif seed_near_or_strict:
        verdict = "D10U_SEED2042_NEAR_OR_STRICT_SIGNAL"
    elif non_seed_weak:
        verdict = "D10U_WEAK_NON_SEED_STATE_ANCHORED"
    elif seed_weak:
        verdict = "D10U_WEAK_SEED2042_ONLY"
    else:
        verdict = "D10U_NO_STATE_ANCHORED_SIGNAL"

    run_summary = {
        "verdict": verdict,
        "elapsed_s": time.perf_counter() - started,
        "setup": {
            "mode": args.mode,
            "eval_len": args.eval_len,
            "eval_seeds": parse_int_csv(args.eval_seeds),
            "control_repeats": args.control_repeats,
            "proposals_per_round": args.proposals_per_arm,
            "ladder_rounds": args.ladder_rounds,
            "arms": arms,
            "checkpoints": checkpoint_paths,
            "controls": parse_csv(args.controls),
            "selectivity_gate": args.selectivity_gate,
        },
        "top_candidates": all_candidate_rows[:10],
    }
    write_csv(out / "d10u_eval_rows.csv", all_eval_rows, ["seed_label", "candidate_label", "control_type", "eval_seed", "smooth_delta", "accuracy_delta", "echo_delta", "unigram_delta", "mo_delta"])
    write_candidate_summary(out / "candidate_summary.csv", all_candidate_rows)
    write_csv(
        out / "ladder_paths.csv",
        ladder_rows,
        [
            "seed_label",
            "arm",
            "ladder_round",
            "best_candidate",
            "best_class",
            "best_ladder_score",
            "best_mo_delta",
            "best_selectivity_ci_low",
            "best_smooth_delta",
            "best_accuracy_delta",
            "best_unigram_delta",
            "best_worst_artifact_control",
            "accepted",
            "accept_reason",
        ],
    )
    (out / "run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    write_ladder_report(out, run_summary)
    return run_summary


def write_report(out: Path, summary: dict) -> None:
    lines = [
        "# D10s/D10u State-Anchored Wiring Prior Report",
        "",
        f"Verdict: `{summary['verdict']}`",
        "",
        "This run treats D10r-v8 artifact controls as part of candidate quality.",
        "A candidate is not trusted unless it beats the worst non-diagnostic",
        "artifact control, including `state_shuffle_shared`.",
        "",
        "## Top Candidates",
        "",
        "| seed | arm | proposal | class | anchor_pass | selectivity | ci_low | worst_control | mo_delta |",
        "|---|---|---:|---|---|---:|---:|---|---:|",
    ]
    for row in summary["top_candidates"][:10]:
        lines.append(
            f"| {row['seed_label']} | {row['arm']} | {row['proposal_idx']} | {row['candidate_class']} | "
            f"{row['state_anchor_pass']} | {row['hardened_selectivity']:.6f} | "
            f"{row['hardened_selectivity_ci_low']:.6f} | {row['worst_artifact_control']} | "
            f"{row['mo_delta']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Progress Map",
            "",
            "```text",
            "[4] D10r-v8 state identity gate: beta.8 failed",
            "[5] State-anchored wiring-prior search: CURRENT",
            "    |-- trusted non-seed2042 signal -> D10r confirm, then H512 can be planned",
            "    '-- no trusted signal -> redesign projection/readout or training objective",
            "```",
        ]
    )
    (out / "D10S_WIRING_PRIOR_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_ladder_report(out: Path, summary: dict) -> None:
    lines = [
        "# D10u Focused State-Anchored Ladder Report",
        "",
        f"Verdict: `{summary['verdict']}`",
        "",
        "The ladder accepts only candidates that pass the state-anchor selectivity gate.",
        "Near/strict candidates are required before any D10r-v8 confirm or H512 planning.",
        "",
        "## Top Candidates",
        "",
        "| seed | arm | round | class | anchor_pass | ladder_score | smooth | accuracy | unigram | ci_low | worst_control |",
        "|---|---|---:|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary["top_candidates"][:10]:
        lines.append(
            f"| {row['seed_label']} | {row['arm']} | {row.get('ladder_round', '')} | "
            f"{row['candidate_class']} | {row['state_anchor_pass']} | {row['ladder_score']:.6f} | "
            f"{row['smooth_delta']:.6f} | {row['accuracy_delta']:.6f} | {row['unigram_delta']:.6f} | "
            f"{row['hardened_selectivity_ci_low']:.6f} | {row['worst_artifact_control']} |"
        )
    lines.extend(
        [
            "",
            "## Progress Map",
            "",
            "```text",
            "[1] HDS basin map: DONE",
            "[2] beta.8 release path: BLOCKED by state identity",
            "[3] D10u state-anchored ladder: CURRENT",
            "    |-- near/strict signal -> D10r-v8 longer confirm",
            "    '-- weak/no signal -> redesign objective or proposal policy",
            "```",
        ]
    )
    (out / "D10U_FOCUSED_LADDER_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "confirm", "ladder"], default="smoke")
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
    parser.add_argument("--controls", default=",".join(DEFAULT_STATE_ANCHOR_CONTROLS))
    parser.add_argument("--control-repeats", type=int, default=2)
    parser.add_argument("--selectivity-gate", choices=["mean", "ci"], default="ci")
    parser.add_argument("--proposals-per-arm", type=int, default=32)
    parser.add_argument("--ladder-rounds", type=int, default=4)
    parser.add_argument("--ladder-patience", type=int, default=2)
    parser.add_argument("--ladder-min-improvement", type=float, default=0.0001)
    parser.add_argument("--export-top", type=int, default=8)
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
    summary = run_ladder(args) if args.mode == "ladder" else run_sweep(args)
    print(json.dumps({"verdict": summary["verdict"], "elapsed_s": summary["elapsed_s"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
