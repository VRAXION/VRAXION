#!/usr/bin/env python3
"""D10r hardened eval/projection trust gate.

Scratch/prototype only. This does not train or mutate networks. It evaluates
existing checkpoints against a baseline under semantic/projection controls and
reports paired control margins. The goal is to decide whether later D10s/H512
work would be interpretable, not to promote a checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
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
DEFAULT_ALTERNATE_BASELINES = [
    "output/phase_d7_operator_bandit_20260427/H_384/D7_BASELINE/seed_42/final.ckpt",
    "output/phase_d7_operator_bandit_20260427/H_384/D7_BASELINE/seed_1042/final.ckpt",
    "output/phase_d7_operator_bandit_20260427/H_384/D7_BASELINE/seed_3042/final.ckpt",
    "output/phase_d7_operator_bandit_20260427/H_384/D7_BASELINE/seed_4042/final.ckpt",
]
PRIMARY_CONTROLS = [
    "random_label",
    "random_bigram",
    "unigram_decoy",
    "projection_shuffle",
    "projection_reinit",
    "random_projection_null",
    "state_shuffle_shared",
    "no_network_random_state",
    "time_shuffle",
]
STATE_SHUFFLE_CONTROLS = {
    "state_shuffle",
    "state_shuffle_shared",
    "state_shuffle_projection_consistent",
    "state_shuffle_active_rows_shared",
    "state_shuffle_high_norm_shared",
    "state_shuffle_low_norm_shared",
    "state_shuffle_duplicate_projection_shared",
    "state_shuffle_similar_projection_shared",
}
STATE_ZONE_CONTROLS = {
    "state_shuffle_active_rows_shared",
    "state_shuffle_high_norm_shared",
    "state_shuffle_low_norm_shared",
    "state_shuffle_duplicate_projection_shared",
    "state_shuffle_similar_projection_shared",
}
DIAGNOSTIC_CONTROLS = {
    "state_shuffle_projection_consistent",
    "random_projection_null_independent",
    *STATE_ZONE_CONTROLS,
}
REPEATED_CONTROLS = {
    *STATE_SHUFFLE_CONTROLS,
    "no_network_random_state",
    "random_projection_null",
    "random_projection_null_independent",
}
STATE_ZONE_DIAGNOSTIC_CONTROLS = [
    "state_shuffle_active_rows_shared",
    "state_shuffle_high_norm_shared",
    "state_shuffle_low_norm_shared",
    "state_shuffle_duplicate_projection_shared",
    "state_shuffle_similar_projection_shared",
]
SIMILAR_PROJECTION_COSINE = 0.995
STATE_ZONE_INDEX_CACHE: dict[tuple[str, str], np.ndarray | None] = {}


def control_base_name(control: str) -> str:
    parts = control.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return control


def is_diagnostic_control(control: str) -> bool:
    return control_base_name(control) in DIAGNOSTIC_CONTROLS


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_int_csv(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def expand_controls(controls: list[str], control_repeats: int) -> list[dict]:
    expanded = []
    for control in controls:
        repeat_count = max(1, control_repeats) if control in REPEATED_CONTROLS else 1
        for repeat in range(repeat_count):
            label = f"{control}_{repeat:02d}" if repeat_count > 1 else control
            expanded.append({"label": label, "base": control, "repeat": repeat})
    return expanded


def augment_artifact_controls(controls: list[str], state_zone_diagnostics: bool) -> list[str]:
    out = list(controls)
    if state_zone_diagnostics and "state_shuffle_shared" in out:
        for control in STATE_ZONE_DIAGNOSTIC_CONTROLS:
            if control not in out:
                out.append(control)
    return out


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


def clone_with_projection(
    ckpt: gpu_eval.CheckpointArrays,
    label: str,
    projection: gpu_eval.ProjectionArrays,
) -> gpu_eval.CheckpointArrays:
    out = clone_checkpoint(ckpt, label)
    out.projection = projection
    return out


def checkpoint_label(path: Path) -> str:
    parts = [p.lower() for p in path.parts]
    for part in reversed(parts):
        if part.startswith("seed_"):
            return part
    if "beta.8" in str(path).lower() or "generalist" in path.name.lower():
        return "beta8_generalist"
    return path.stem


def control_targets(
    control_type: str,
    hot_to_idx: np.ndarray,
    bigram: np.ndarray,
    unigram: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if control_type in {
        "real",
        "projection_shuffle",
        "projection_reinit",
        "random_projection_null",
        "random_projection_null_independent",
        "state_shuffle",
        "state_shuffle_shared",
        "state_shuffle_projection_consistent",
        "state_shuffle_active_rows_shared",
        "state_shuffle_high_norm_shared",
        "state_shuffle_low_norm_shared",
        "state_shuffle_duplicate_projection_shared",
        "state_shuffle_similar_projection_shared",
        "no_network_random_state",
        "time_shuffle",
    }:
        return hot_to_idx, bigram, unigram
    if control_type == "random_label":
        return d10k.shuffle_metric_targets(hot_to_idx, bigram, unigram, seed)
    rng = np.random.default_rng(seed)
    if control_type == "random_bigram":
        perm = rng.permutation(bigram.shape[0])
        return hot_to_idx, bigram[perm][:, perm], unigram
    if control_type == "unigram_decoy":
        return hot_to_idx, np.tile(unigram.reshape(1, -1), (bigram.shape[0], 1)).astype(np.float32), unigram
    raise ValueError(f"unknown control type: {control_type}")


def transform_checkpoints(
    checkpoints: list[gpu_eval.CheckpointArrays],
    control_type: str,
    seed: int,
) -> list[gpu_eval.CheckpointArrays]:
    if control_type == "projection_shuffle":
        rng = np.random.default_rng(seed)
        row_perm = rng.permutation(checkpoints[0].projection.weights.shape[0])
        col_perm = rng.permutation(checkpoints[0].projection.weights.shape[1])
        out = []
        for ckpt in checkpoints:
            weights = ckpt.projection.weights[row_perm][:, col_perm].copy()
            projection = gpu_eval.ProjectionArrays(weights, ckpt.projection.input_dim, ckpt.projection.output_classes)
            out.append(clone_with_projection(ckpt, ckpt.path, projection))
        return out
    if control_type == "projection_reinit":
        rng = np.random.default_rng(seed)
        out = []
        for ckpt in checkpoints:
            flat = ckpt.projection.weights.reshape(-1)
            if flat.size == 0:
                weights = ckpt.projection.weights.copy()
            else:
                weights = rng.choice(flat, size=ckpt.projection.weights.shape, replace=True).astype(np.int16)
            projection = gpu_eval.ProjectionArrays(weights, ckpt.projection.input_dim, ckpt.projection.output_classes)
            out.append(clone_with_projection(ckpt, ckpt.path, projection))
        return out
    if control_type == "random_projection_null":
        rng = np.random.default_rng(seed)
        weights = rng.integers(
            -3,
            4,
            size=checkpoints[0].projection.weights.shape,
            dtype=np.int16,
        )
        projection = gpu_eval.ProjectionArrays(
            weights,
            checkpoints[0].projection.input_dim,
            checkpoints[0].projection.output_classes,
        )
        return [clone_with_projection(ckpt, ckpt.path, projection) for ckpt in checkpoints]
    if control_type == "random_projection_null_independent":
        rng = np.random.default_rng(seed)
        out = []
        for ckpt in checkpoints:
            weights = rng.integers(
                -3,
                4,
                size=ckpt.projection.weights.shape,
                dtype=np.int16,
            )
            projection = gpu_eval.ProjectionArrays(weights, ckpt.projection.input_dim, ckpt.projection.output_classes)
            out.append(clone_with_projection(ckpt, ckpt.path, projection))
        return out
    return checkpoints


def mo_score_from_delta(delta: dict[str, float]) -> float:
    return (
        delta["smooth"]
        + 0.5 * delta["accuracy"]
        + 1.5 * max(delta["unigram"], -0.012)
        - 0.25 * abs(delta["echo"])
    )


def projection_row_groups(weights: np.ndarray) -> tuple[list[dict], dict[str, list[int]]]:
    rows = np.asarray(weights, dtype=np.int16)
    hashes = [hashlib.sha1(np.ascontiguousarray(row).tobytes()).hexdigest()[:16] for row in rows]
    by_hash: dict[str, list[int]] = {}
    for idx, digest in enumerate(hashes):
        by_hash.setdefault(digest, []).append(idx)
    duplicate_group_ids: dict[str, str] = {}
    for group_idx, digest in enumerate(sorted(by_hash)):
        duplicate_group_ids[digest] = f"dup_{group_idx:04d}" if len(by_hash[digest]) > 1 else ""
    rows_f = rows.astype(np.float32)
    norms = np.linalg.norm(rows_f, axis=1)
    active = np.flatnonzero(norms > 0.0).astype(int).tolist()
    high_norm: list[int] = []
    low_norm: list[int] = []
    if active:
        active_norms = norms[active]
        lo = float(np.quantile(active_norms, 0.25))
        hi = float(np.quantile(active_norms, 0.75))
        high_norm = [idx for idx in active if norms[idx] >= hi]
        low_norm = [idx for idx in active if norms[idx] <= lo]
    duplicate_projection = [idx for idx, digest in enumerate(hashes) if len(by_hash[digest]) > 1]
    nearest_rows = [-1] * len(rows)
    nearest_cosines = [0.0] * len(rows)
    similar_projection: list[int] = []
    if len(rows) > 1:
        denom = np.maximum(norms, 1e-12)
        normalized = rows_f / denom[:, None]
        cosine = normalized @ normalized.T
        np.fill_diagonal(cosine, -2.0)
        nearest = np.argmax(cosine, axis=1)
        nearest_vals = cosine[np.arange(len(rows)), nearest]
        nearest_rows = nearest.astype(int).tolist()
        nearest_cosines = nearest_vals.astype(float).tolist()
        similar_projection = [
            int(idx)
            for idx, value in enumerate(nearest_vals)
            if norms[idx] > 0.0 and float(value) >= SIMILAR_PROJECTION_COSINE
        ]
    high_norm_set = set(high_norm)
    low_norm_set = set(low_norm)
    duplicate_projection_set = set(duplicate_projection)
    similar_projection_set = set(similar_projection)
    row_info = []
    for idx, row in enumerate(rows):
        digest = hashes[idx]
        row_info.append(
            {
                "row_idx": idx,
                "row_hash": digest,
                "row_l1": int(np.abs(row.astype(np.int32)).sum()),
                "row_l2": float(norms[idx]),
                "row_nonzero": int(np.count_nonzero(row)),
                "duplicate_group": duplicate_group_ids[digest],
                "duplicate_count": len(by_hash[digest]),
                "nearest_row": nearest_rows[idx],
                "nearest_cosine": float(nearest_cosines[idx]),
                "is_active": bool(norms[idx] > 0.0),
                "is_high_norm": idx in high_norm_set,
                "is_low_norm": idx in low_norm_set,
                "is_duplicate_projection": idx in duplicate_projection_set,
                "is_similar_projection": idx in similar_projection_set,
            }
        )
    zones = {
        "all_rows": list(range(len(rows))),
        "active_rows": active,
        "high_norm": high_norm,
        "low_norm": low_norm,
        "duplicate_projection": duplicate_projection,
        "similar_projection": similar_projection,
    }
    return row_info, zones


def projection_row_uniqueness_rows(label: str, ckpt: gpu_eval.CheckpointArrays) -> tuple[list[dict], dict[str, list[int]]]:
    row_info, zones = projection_row_groups(ckpt.projection.weights)
    rows = []
    total_rows = max(1, len(row_info))
    for row in row_info:
        rows.append(
            {
                "checkpoint_label": label,
                "projection_input_dim": ckpt.projection.input_dim,
                "projection_output_classes": ckpt.projection.output_classes,
                **row,
                "active_fraction": len(zones["active_rows"]) / total_rows,
                "duplicate_fraction": len(zones["duplicate_projection"]) / total_rows,
                "similar_fraction": len(zones["similar_projection"]) / total_rows,
            }
        )
    return rows, zones


def state_shuffle_zone_name(control_type: str) -> str:
    mapping = {
        "state_shuffle_shared": "all_rows",
        "state_shuffle_projection_consistent": "all_rows_projection_consistent",
        "state_shuffle_active_rows_shared": "active_rows",
        "state_shuffle_high_norm_shared": "high_norm",
        "state_shuffle_low_norm_shared": "low_norm",
        "state_shuffle_duplicate_projection_shared": "duplicate_projection",
        "state_shuffle_similar_projection_shared": "similar_projection",
    }
    return mapping.get(control_base_name(control_type), control_base_name(control_type))


def state_shuffle_zone_indices(control_type: str, reference_weights: np.ndarray) -> np.ndarray | None:
    digest = hashlib.sha1(np.ascontiguousarray(reference_weights).tobytes()).hexdigest()[:16]
    key = (control_type, digest)
    if key in STATE_ZONE_INDEX_CACHE:
        return STATE_ZONE_INDEX_CACHE[key]
    _, zones = projection_row_groups(reference_weights)
    base = control_base_name(control_type)
    if base in {"state_shuffle", "state_shuffle_shared", "state_shuffle_projection_consistent"}:
        STATE_ZONE_INDEX_CACHE[key] = None
        return None
    zone = state_shuffle_zone_name(base)
    indices = np.asarray(zones.get(zone, []), dtype=np.int64)
    STATE_ZONE_INDEX_CACHE[key] = indices
    return indices


def permute_outputs_in_zone(
    outputs: torch.Tensor,
    control_type: str,
    rng: np.random.Generator,
    reference_weights: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    zone_indices = state_shuffle_zone_indices(control_type, reference_weights)
    if zone_indices is not None and len(zone_indices) < 2:
        return outputs, None
    shuffled = outputs.clone()
    if zone_indices is None:
        perm_np = rng.permutation(outputs.shape[-1])
        perm = torch.as_tensor(perm_np, dtype=torch.long, device=outputs.device)
        shuffled = outputs[:, :, perm]
        return shuffled, perm
    zone = torch.as_tensor(zone_indices, dtype=torch.long, device=outputs.device)
    permuted_zone = torch.as_tensor(rng.permutation(zone_indices), dtype=torch.long, device=outputs.device)
    shuffled[:, :, zone] = outputs[:, :, permuted_zone]
    return shuffled, None


def evaluate_seed_control(
    checkpoints: list[gpu_eval.CheckpointArrays],
    table: gpu_eval.VcbpTablePy,
    pair_ids: np.ndarray,
    hot_to_idx: np.ndarray,
    bigram: np.ndarray,
    unigram: np.ndarray,
    eval_len: int,
    eval_seed: int,
    control_type: str,
    device: torch.device,
    control_seed: int,
) -> dict[str, list[float]]:
    eval_hot, eval_bigram, eval_unigram = control_targets(control_type, hot_to_idx, bigram, unigram, control_seed)
    eval_checkpoints = transform_checkpoints(checkpoints, control_type, control_seed)
    state = d10j.build_sparse_state(eval_checkpoints, device)
    off = gpu_eval.deterministic_offset(eval_seed, len(pair_ids), eval_len)
    cur_ids = pair_ids[off : off + eval_len]
    next_ids = pair_ids[off + 1 : off + eval_len + 1]
    input_ids = cur_ids
    if control_type == "time_shuffle":
        rng = np.random.default_rng(control_seed + eval_seed)
        input_ids = cur_ids[rng.permutation(len(cur_ids))]
    inputs = torch.as_tensor(
        gpu_eval.quantized_inputs(input_ids, table, state.h, input_scatter=False),
        dtype=torch.int16,
        device=device,
    )
    outputs = d10j.propagate_sequence_sparse(state, inputs)
    if control_type == "no_network_random_state":
        rng = np.random.default_rng(control_seed + eval_seed)
        output_dim = state.weights.shape[1]
        random_outputs = rng.integers(
            0,
            gpu_eval.MAX_CHARGE + 1,
            size=(len(eval_checkpoints), eval_len, output_dim),
            dtype=np.int16,
        )
        outputs = torch.as_tensor(random_outputs, dtype=torch.int16, device=device)
    score_weights = state.weights
    if control_type in STATE_SHUFFLE_CONTROLS:
        rng = np.random.default_rng(control_seed + eval_seed)
        adjusted_weights = state.weights.clone() if control_type == "state_shuffle_projection_consistent" else state.weights
        if control_type in {"state_shuffle", "state_shuffle_projection_consistent"}:
            shuffled = torch.empty_like(outputs)
            for idx in range(outputs.shape[0]):
                perm = torch.as_tensor(rng.permutation(outputs.shape[-1]), dtype=torch.long, device=device)
                shuffled[idx] = outputs[idx, :, perm]
                if control_type == "state_shuffle_projection_consistent":
                    adjusted_weights[idx] = state.weights[idx, perm, :]
            outputs = shuffled
        else:
            reference_idx = 1 if outputs.shape[0] > 1 else 0
            reference_weights = state.weights[reference_idx].detach().cpu().numpy()
            outputs, _ = permute_outputs_in_zone(outputs, control_type, rng, reference_weights)
        score_weights = adjusted_weights
    scores = gpu_eval.projection_scores(outputs, score_weights)
    probs = torch.softmax(scores, dim=-1)
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1) / math.log(probs.shape[-1])
    max_prob = probs.max(dim=-1).values
    target_idx = torch.as_tensor(eval_hot[next_ids], dtype=torch.long, device=device)
    cur_idx = torch.as_tensor(eval_hot[cur_ids], dtype=torch.long, device=device)
    preds = scores.argmax(dim=-1)
    valid_target = target_idx >= 0
    valid_cur = cur_idx >= 0
    acc = ((preds == target_idx.unsqueeze(0)) & valid_target.unsqueeze(0)).sum(dim=1).to(torch.float64) / eval_len
    echo = torch.zeros(outputs.shape[0], dtype=torch.float64, device=device)
    smooth = torch.zeros(outputs.shape[0], dtype=torch.float64, device=device)
    if valid_cur.any():
        idxs = cur_idx[valid_cur]
        one_hot = torch.nn.functional.one_hot(idxs, num_classes=probs.shape[-1]).to(torch.float32).to(device)
        echo = gpu_eval.cosine(probs[:, valid_cur, :], one_hot.unsqueeze(0)).sum(dim=1).to(torch.float64) / int(valid_cur.sum())
        bigram_t = torch.as_tensor(eval_bigram[idxs.cpu().numpy()], dtype=torch.float32, device=device)
        smooth = gpu_eval.cosine(probs[:, valid_cur, :], bigram_t.unsqueeze(0)).sum(dim=1).to(torch.float64) / int(valid_cur.sum())
    unigram_t = torch.as_tensor(eval_unigram, dtype=torch.float32, device=device)
    unigram_score = gpu_eval.cosine(probs, unigram_t.view(1, 1, -1)).sum(dim=1).to(torch.float64) / eval_len
    return {
        "accuracy": acc.detach().cpu().tolist(),
        "echo": echo.detach().cpu().tolist(),
        "smooth": smooth.detach().cpu().tolist(),
        "unigram": unigram_score.detach().cpu().tolist(),
        "entropy": entropy.mean(dim=1).detach().cpu().tolist(),
        "max_prob": max_prob.mean(dim=1).detach().cpu().tolist(),
    }


def bootstrap_ci(values: list[float], n_boot: int, alpha: float, seed: int) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0
    if arr.size == 1 or n_boot <= 0:
        return float(arr.mean()), float(arr.mean())
    rng = np.random.default_rng(seed)
    draws = rng.choice(arr, size=(n_boot, arr.size), replace=True).mean(axis=1)
    lo, hi = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


def paired_sign_flip_pvalue(values: list[float], n_perm: int, seed: int) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 1.0
    observed = float(arr.mean())
    if observed <= 0:
        return 1.0
    if arr.size <= 16:
        count = 0
        total = 1 << int(arr.size)
        for mask in range(total):
            signs = np.ones(arr.size, dtype=np.float64)
            for idx in range(arr.size):
                if (mask >> idx) & 1:
                    signs[idx] = -1.0
            if float((signs * arr).mean()) >= observed:
                count += 1
        return float(count / total)
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(max(1, n_perm), arr.size), replace=True)
    perm_means = (signs * arr.reshape(1, -1)).mean(axis=1)
    return float((np.count_nonzero(perm_means >= observed) + 1) / (len(perm_means) + 1))


def holm_bonferroni(pairs: list[tuple[str, float]], alpha: float) -> dict[str, dict]:
    ordered = sorted(pairs, key=lambda item: item[1])
    m = len(ordered)
    out: dict[str, dict] = {}
    blocked = False
    for rank, (name, p_value) in enumerate(ordered, start=1):
        threshold = alpha / max(1, m - rank + 1)
        rejected = (not blocked) and p_value <= threshold
        if not rejected:
            blocked = True
        out[name] = {
            "p_value": p_value,
            "holm_threshold": threshold,
            "holm_reject": rejected,
        }
    return out


def summarize_checkpoint(
    label: str,
    rows: list[dict],
    controls: list[str],
    bootstrap_samples: int,
    permutation_samples: int,
    alpha: float,
    seed: int,
    min_real_mo: float,
    min_trusted_mo: float,
) -> tuple[list[dict], dict]:
    real_by_seed = {int(r["eval_seed"]): float(r["mo_delta"]) for r in rows if r["checkpoint_label"] == label and r["control_type"] == "real"}
    margin_rows: list[dict] = []
    p_pairs = []
    for control in controls:
        control_values = {
            int(r["eval_seed"]): float(r["mo_delta"])
            for r in rows
            if r["checkpoint_label"] == label and r["control_type"] == control
        }
        margins = [
            real_by_seed[s] - control_values[s]
            for s in sorted(real_by_seed)
            if s in control_values
        ]
        mean = float(np.mean(margins)) if margins else 0.0
        ci_lo, ci_hi = bootstrap_ci(margins, bootstrap_samples, alpha, seed + d10o.stable_arm_seed(label + control))
        p_value = paired_sign_flip_pvalue(margins, permutation_samples, seed + 31 + d10o.stable_arm_seed(label + control))
        if not is_diagnostic_control(control):
            p_pairs.append((control, p_value))
        margin_rows.append(
            {
                "checkpoint_label": label,
                "control_type": control,
                "n": len(margins),
                "margin_mean": mean,
                "margin_ci_low": ci_lo,
                "margin_ci_high": ci_hi,
                "p_value": p_value,
                "holm_p_value": p_value,
                "holm_reject": False,
                "pass": False,
            }
        )
    holm = holm_bonferroni(p_pairs, alpha)
    for row in margin_rows:
        if is_diagnostic_control(row["control_type"]):
            row["holm_threshold"] = 0.0
            continue
        info = holm[row["control_type"]]
        row["holm_p_value"] = info["p_value"]
        row["holm_threshold"] = info["holm_threshold"]
        row["holm_reject"] = info["holm_reject"]
        row["pass"] = row["margin_ci_low"] > 0.0 and info["holm_reject"]
    max_controls = []
    median_controls = []
    for eval_seed in sorted(real_by_seed):
        vals = [
            float(r["mo_delta"])
            for r in rows
            if (
                r["checkpoint_label"] == label
                and int(r["eval_seed"]) == eval_seed
                and r["control_type"] in controls
                and not is_diagnostic_control(r["control_type"])
            )
        ]
        if vals:
            max_controls.append(real_by_seed[eval_seed] - max(vals))
            median_controls.append(real_by_seed[eval_seed] - float(np.median(vals)))
    selectivity_mean = float(np.mean(max_controls)) if max_controls else 0.0
    selectivity_ci = bootstrap_ci(max_controls, bootstrap_samples, alpha, seed + d10o.stable_arm_seed(label + "selectivity"))
    median_selectivity_mean = float(np.mean(median_controls)) if median_controls else 0.0
    median_selectivity_ci = bootstrap_ci(
        median_controls,
        bootstrap_samples,
        alpha,
        seed + d10o.stable_arm_seed(label + "median_selectivity"),
    )
    real_values = list(real_by_seed.values())
    real_mean = float(np.mean(real_values)) if real_values else 0.0
    real_ci = bootstrap_ci(real_values, bootstrap_samples, alpha, seed + d10o.stable_arm_seed(label + "real"))
    trust_margin_rows = [row for row in margin_rows if not is_diagnostic_control(row["control_type"])]
    diagnostic_margin_rows = [row for row in margin_rows if is_diagnostic_control(row["control_type"])]
    blocking_controls = [
        row["control_type"]
        for row in trust_margin_rows
        if float(row["margin_ci_low"]) <= min_trusted_mo
    ]
    summary = {
        "checkpoint_label": label,
        "real_mo_delta_mean": real_mean,
        "real_mo_delta_ci_low": real_ci[0],
        "real_mo_delta_ci_high": real_ci[1],
        "trusted_mo_mean": selectivity_mean,
        "trusted_mo_ci_low": selectivity_ci[0],
        "trusted_mo_ci_high": selectivity_ci[1],
        "selectivity_mean": selectivity_mean,
        "selectivity_ci_low": selectivity_ci[0],
        "selectivity_ci_high": selectivity_ci[1],
        "median_selectivity_mean": median_selectivity_mean,
        "median_selectivity_ci_low": median_selectivity_ci[0],
        "median_selectivity_ci_high": median_selectivity_ci[1],
        "all_controls_pass": all(bool(r["pass"]) for r in trust_margin_rows),
        "failed_controls": [r["control_type"] for r in trust_margin_rows if not bool(r["pass"])],
        "blocking_controls": blocking_controls,
        "diagnostic_controls": [r["control_type"] for r in diagnostic_margin_rows],
    }
    summary["trusted_mo_pass"] = (
        summary["real_mo_delta_ci_low"] > min_real_mo
        and summary["trusted_mo_ci_low"] > min_trusted_mo
    )
    return margin_rows, summary


def summarize_metric_margins(
    label: str,
    rows: list[dict],
    controls: list[str],
    bootstrap_samples: int,
    alpha: float,
    seed: int,
) -> list[dict]:
    out: list[dict] = []
    metric_fields = [(metric, f"{metric}_delta") for metric in METRICS]
    metric_fields.append(("mo", "mo_delta"))
    for control in controls:
        for metric, field in metric_fields:
            real_by_seed = {
                int(r["eval_seed"]): float(r[field])
                for r in rows
                if r["checkpoint_label"] == label and r["control_type"] == "real"
            }
            control_by_seed = {
                int(r["eval_seed"]): float(r[field])
                for r in rows
                if r["checkpoint_label"] == label and r["control_type"] == control
            }
            margins = [
                real_by_seed[s] - control_by_seed[s]
                for s in sorted(real_by_seed)
                if s in control_by_seed
            ]
            if margins:
                seed_order = [s for s in sorted(real_by_seed) if s in control_by_seed]
                worst_idx = int(np.argmin(np.asarray(margins, dtype=np.float64)))
                worst_seed = seed_order[worst_idx]
                worst_margin = float(margins[worst_idx])
                mean = float(np.mean(margins))
            else:
                worst_seed = 0
                worst_margin = 0.0
                mean = 0.0
            ci_lo, ci_hi = bootstrap_ci(
                margins,
                bootstrap_samples,
                alpha,
                seed + d10o.stable_arm_seed(label + control + metric + "metric_margin"),
            )
            out.append(
                {
                    "checkpoint_label": label,
                    "control_type": control,
                    "control_base": control_base_name(control),
                    "metric": metric,
                    "n": len(margins),
                    "margin_mean": mean,
                    "margin_ci_low": ci_lo,
                    "margin_ci_high": ci_hi,
                    "worst_seed": worst_seed,
                    "worst_margin": worst_margin,
                }
            )
    return out


def summarize_control_family_bounds(
    label: str,
    rows: list[dict],
    controls: list[str],
    bootstrap_samples: int,
    alpha: float,
    seed: int,
    min_trusted_mo: float,
) -> list[dict]:
    real_by_seed = {
        int(r["eval_seed"]): float(r["mo_delta"])
        for r in rows
        if r["checkpoint_label"] == label and r["control_type"] == "real"
    }
    controls_by_base: dict[str, list[str]] = {}
    for control in controls:
        if is_diagnostic_control(control):
            continue
        controls_by_base.setdefault(control_base_name(control), []).append(control)
    out: list[dict] = []
    for base, names in sorted(controls_by_base.items()):
        margins = []
        worst_control_by_seed: dict[int, str] = {}
        for eval_seed in sorted(real_by_seed):
            vals = [
                (
                    control,
                    float(r["mo_delta"]),
                )
                for control in names
                for r in rows
                if (
                    r["checkpoint_label"] == label
                    and r["control_type"] == control
                    and int(r["eval_seed"]) == eval_seed
                )
            ]
            if not vals:
                continue
            worst_control, worst_value = max(vals, key=lambda item: item[1])
            worst_control_by_seed[eval_seed] = worst_control
            margins.append(real_by_seed[eval_seed] - worst_value)
        if margins:
            seed_order = [s for s in sorted(real_by_seed) if s in worst_control_by_seed]
            worst_idx = int(np.argmin(np.asarray(margins, dtype=np.float64)))
            worst_seed = seed_order[worst_idx]
            worst_margin = float(margins[worst_idx])
            worst_control = worst_control_by_seed[worst_seed]
            mean = float(np.mean(margins))
        else:
            worst_seed = 0
            worst_margin = 0.0
            worst_control = ""
            mean = 0.0
        ci_lo, ci_hi = bootstrap_ci(
            margins,
            bootstrap_samples,
            alpha,
            seed + d10o.stable_arm_seed(label + base + "family_bound"),
        )
        out.append(
            {
                "checkpoint_label": label,
                "control_base": base,
                "control_count": len(names),
                "n": len(margins),
                "bound_mean": mean,
                "bound_ci_low": ci_lo,
                "bound_ci_high": ci_hi,
                "worst_seed": worst_seed,
                "worst_control": worst_control,
                "worst_margin": worst_margin,
                "pass": ci_lo > min_trusted_mo,
            }
        )
    return out


def summarize_state_shuffle_zone_bounds(
    label: str,
    rows: list[dict],
    controls: list[str],
    zone_counts: dict[str, int],
    total_rows: int,
    bootstrap_samples: int,
    alpha: float,
    seed: int,
    min_trusted_mo: float,
) -> list[dict]:
    real_by_seed = {
        int(r["eval_seed"]): float(r["mo_delta"])
        for r in rows
        if r["checkpoint_label"] == label and r["control_type"] == "real"
    }
    controls_by_base: dict[str, list[str]] = {}
    for control in controls:
        base = control_base_name(control)
        if base.startswith("state_shuffle"):
            controls_by_base.setdefault(base, []).append(control)
    out: list[dict] = []
    for base, names in sorted(controls_by_base.items()):
        margins = []
        worst_control_by_seed: dict[int, str] = {}
        for eval_seed in sorted(real_by_seed):
            vals = [
                (
                    control,
                    float(r["mo_delta"]),
                )
                for control in names
                for r in rows
                if (
                    r["checkpoint_label"] == label
                    and r["control_type"] == control
                    and int(r["eval_seed"]) == eval_seed
                )
            ]
            if not vals:
                continue
            worst_control, worst_value = max(vals, key=lambda item: item[1])
            worst_control_by_seed[eval_seed] = worst_control
            margins.append(real_by_seed[eval_seed] - worst_value)
        if margins:
            seed_order = [s for s in sorted(real_by_seed) if s in worst_control_by_seed]
            worst_idx = int(np.argmin(np.asarray(margins, dtype=np.float64)))
            worst_seed = seed_order[worst_idx]
            worst_margin = float(margins[worst_idx])
            worst_control = worst_control_by_seed[worst_seed]
            mean = float(np.mean(margins))
        else:
            worst_seed = 0
            worst_margin = 0.0
            worst_control = ""
            mean = 0.0
        ci_lo, ci_hi = bootstrap_ci(
            margins,
            bootstrap_samples,
            alpha,
            seed + d10o.stable_arm_seed(label + base + "zone_bound"),
        )
        zone_name = state_shuffle_zone_name(base)
        row_count = int(zone_counts.get(zone_name, total_rows if zone_name.startswith("all_rows") else 0))
        out.append(
            {
                "checkpoint_label": label,
                "control_type": ",".join(names),
                "control_base": base,
                "control_count": len(names),
                "zone_name": zone_name,
                "row_count": row_count,
                "row_fraction": row_count / max(1, total_rows),
                "n": len(margins),
                "bound_mean": mean,
                "bound_ci_low": ci_lo,
                "bound_ci_high": ci_hi,
                "worst_seed": worst_seed,
                "worst_control": worst_control,
                "worst_margin": worst_margin,
                "diagnostic": is_diagnostic_control(base),
                "pass": ci_lo > min_trusted_mo,
            }
        )
    return out


def summarize_state_shuffle_score_drift(label: str, rows: list[dict], controls: list[str]) -> list[dict]:
    out: list[dict] = []
    for control in controls:
        base = control_base_name(control)
        if not base.startswith("state_shuffle"):
            continue
        real_rows = {
            int(r["eval_seed"]): r
            for r in rows
            if r["checkpoint_label"] == label and r["control_type"] == "real"
        }
        control_rows = {
            int(r["eval_seed"]): r
            for r in rows
            if r["checkpoint_label"] == label and r["control_type"] == control
        }
        for eval_seed in sorted(real_rows):
            if eval_seed not in control_rows:
                continue
            real_row = real_rows[eval_seed]
            control_row = control_rows[eval_seed]
            row = {
                "checkpoint_label": label,
                "control_type": control,
                "control_base": base,
                "zone_name": state_shuffle_zone_name(base),
                "eval_seed": eval_seed,
                "real_mo_delta": float(real_row["mo_delta"]),
                "control_mo_delta": float(control_row["mo_delta"]),
                "mo_margin": float(real_row["mo_delta"]) - float(control_row["mo_delta"]),
            }
            for metric in METRICS:
                real_value = float(real_row[f"{metric}_delta"])
                control_value = float(control_row[f"{metric}_delta"])
                row[f"real_{metric}_delta"] = real_value
                row[f"control_{metric}_delta"] = control_value
                row[f"{metric}_margin"] = real_value - control_value
            out.append(row)
    return out


def v8_projection_symmetry_supported(zone_bound_rows: list[dict], positive_label: str) -> bool:
    by_base = {
        row["control_base"]: row
        for row in zone_bound_rows
        if row["checkpoint_label"] == positive_label
    }
    identity_bases = {
        "state_shuffle_active_rows_shared",
        "state_shuffle_high_norm_shared",
        "state_shuffle_low_norm_shared",
    }
    symmetry_bases = {
        "state_shuffle_duplicate_projection_shared",
        "state_shuffle_similar_projection_shared",
    }
    if not all(bool(by_base.get(base, {}).get("pass")) for base in identity_bases if base in by_base):
        return False
    for base in symmetry_bases:
        row = by_base.get(base)
        if row and int(row.get("row_count", 0)) >= 2 and not bool(row.get("pass")):
            return True
    return False


def write_report(
    out: Path,
    run_summary: dict,
    checkpoint_summaries: list[dict],
    margin_rows: list[dict],
    metric_margin_rows: list[dict],
    family_bound_rows: list[dict],
    zone_bound_rows: list[dict],
) -> None:
    roles = run_summary["roles"]
    positive_label = run_summary["setup"]["positive_label"]
    positive_summary = next(row for row in checkpoint_summaries if row["checkpoint_label"] == positive_label)
    positive_margins = [row for row in margin_rows if row["checkpoint_label"] == positive_label]
    positive_metric_margins = [row for row in metric_margin_rows if row["checkpoint_label"] == positive_label]
    positive_family_bounds = [row for row in family_bound_rows if row["checkpoint_label"] == positive_label]
    positive_zone_bounds = [row for row in zone_bound_rows if row["checkpoint_label"] == positive_label]
    alternate_summaries = [row for row in checkpoint_summaries if roles.get(row["checkpoint_label"]) == "alternate_baseline"]
    lines = [
        "# D10r Hardened Eval / Projection Report",
        "",
        f"Verdict: `{run_summary['verdict']}`",
        f"Alternate baseline verdict: `{run_summary['alternate_baseline_verdict']}`",
        "",
        "## Setup",
        "",
        f"- baseline: `{run_summary['setup']['baseline']}`",
        f"- positive: `{run_summary['setup']['positive']}`",
        f"- eval_len: `{run_summary['setup']['eval_len']}`",
        f"- eval_seeds: `{','.join(str(s) for s in run_summary['setup']['eval_seeds'])}`",
        f"- artifact_controls: `{','.join(run_summary['setup']['artifact_controls'])}`",
        f"- state_zone_diagnostics: `{run_summary['setup']['state_zone_diagnostics']}`",
        f"- alternate_baseline_mode: `{run_summary['setup']['alternate_baseline_mode']}`",
        f"- max_charge: `{run_summary['setup']['max_charge']}`",
        "",
        "## Artifact Null Margins",
        "",
        "| control | margin mean | margin CI | pass |",
        "|---|---:|---:|---|",
    ]
    for row in positive_margins:
        lines.append(
            f"| {row['control_type']} | {row['margin_mean']:.6f} | "
            f"[{row['margin_ci_low']:.6f},{row['margin_ci_high']:.6f}] | {row['pass']} |"
        )
    lines.extend(
        [
            "",
            "## Artifact Family Bounds",
            "",
            "| control family | repeats | bound mean | bound CI | worst seed | worst control | pass |",
            "|---|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in positive_family_bounds:
        lines.append(
            f"| {row['control_base']} | {row['control_count']} | {row['bound_mean']:.6f} | "
            f"[{row['bound_ci_low']:.6f},{row['bound_ci_high']:.6f}] | {row['worst_seed']} | "
            f"{row['worst_control']} | {row['pass']} |"
        )
    if positive_zone_bounds:
        lines.extend(
            [
                "",
                "## State Shuffle Zone Bounds",
                "",
                "| control family | repeats | zone | rows | bound mean | bound CI | worst seed | worst control | pass | diagnostic |",
                "|---|---:|---|---:|---:|---:|---:|---|---|---|",
            ]
        )
        for row in positive_zone_bounds:
            lines.append(
                f"| {row['control_base']} | {row['control_count']} | {row['zone_name']} | {row['row_count']} | "
                f"{row['bound_mean']:.6f} | [{row['bound_ci_low']:.6f},{row['bound_ci_high']:.6f}] | "
                f"{row['worst_seed']} | {row['worst_control']} | {row['pass']} | {row['diagnostic']} |"
            )
    state_metric_rows = [
        row
        for row in positive_metric_margins
        if row["control_base"] in {"state_shuffle_shared", "state_shuffle_projection_consistent"}
    ]
    if state_metric_rows:
        lines.extend(
            [
                "",
                "## State Shuffle Metric Diagnostics",
                "",
                "| control | metric | margin mean | margin CI | worst seed | worst margin |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for row in state_metric_rows:
            lines.append(
                f"| {row['control_type']} | {row['metric']} | {row['margin_mean']:.6f} | "
                f"[{row['margin_ci_low']:.6f},{row['margin_ci_high']:.6f}] | "
                f"{row['worst_seed']} | {row['worst_margin']:.6f} |"
            )
    lines.extend(
        [
            "",
            "## Alternate Baseline Trusted Scores",
            "",
            "| checkpoint | real MO delta | trusted MO | trusted pass |",
            "|---|---:|---:|---|",
        ]
    )
    for row in alternate_summaries:
        lines.append(
            f"| {row['checkpoint_label']} | {row['real_mo_delta_mean']:.6f} "
            f"[{row['real_mo_delta_ci_low']:.6f},{row['real_mo_delta_ci_high']:.6f}] | "
            f"{row['trusted_mo_mean']:.6f} [{row['trusted_mo_ci_low']:.6f},{row['trusted_mo_ci_high']:.6f}] | "
            f"{row['trusted_mo_pass']} |"
        )
    lines.extend(
        [
            "",
            "## Final Release Gate Decision",
            "",
            "| field | value |",
            "|---|---|",
            f"| artifact verdict | `{run_summary['verdict']}` |",
            f"| artifact gate pass | `{run_summary['artifact_gate_pass']}` |",
            f"| D10s unlocked | `{run_summary['d10s_unlocked']}` |",
            f"| alternate baseline verdict | `{run_summary['alternate_baseline_verdict']}` |",
            f"| alternate baseline signals | `{','.join(run_summary['alternate_baseline_signals'])}` |",
            f"| blocking controls | `{','.join(positive_summary['blocking_controls'])}` |",
            f"| blocking control families | `{','.join(run_summary['blocking_control_families'])}` |",
            "",
            "## Checkpoint Summary",
            "",
            "| checkpoint | role | real MO delta | trusted MO | median selectivity | trusted pass | failed controls |",
            "|---|---|---:|---:|---:|---|---|",
        ]
    )
    for row in checkpoint_summaries:
        label = row["checkpoint_label"]
        role = roles.get(label, "unknown")
        failed = ",".join(row["failed_controls"])
        lines.append(
            f"| {label} | {role} | {row['real_mo_delta_mean']:.6f} "
            f"[{row['real_mo_delta_ci_low']:.6f},{row['real_mo_delta_ci_high']:.6f}] | "
            f"{row['trusted_mo_mean']:.6f} [{row['trusted_mo_ci_low']:.6f},{row['trusted_mo_ci_high']:.6f}] | "
            f"{row['median_selectivity_mean']:.6f} [{row['median_selectivity_ci_low']:.6f},{row['median_selectivity_ci_high']:.6f}] | "
            f"{row['trusted_mo_pass']} | {failed} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `D10R_V5_ARTIFACT_GATE_PASS` means the known positive beats artifact/null controls by CI.",
            "- `D10R_V5_ARTIFACT_READOUT_BLOCKED` means raw real signal remains positive but artifact-adjusted trusted MO does not.",
            "- `D10R_V7_STATE_SHUFFLE_BOUND_BLOCKED` means the only remaining artifact-family blocker is shared state shuffling.",
            "- `D10R_V8_STATE_IDENTITY_PASS` means beta.8 beats the state-shuffle family bound.",
            "- `D10R_V8_PROJECTION_SYMMETRY_CONFIRMED` means the blocker is localized to projection-similar rows.",
            "- `D10R_V8_WEAK_STATE_IDENTITY_FAIL` means active or high-norm state shuffles still beat the real signal.",
            "- `D10R_V8_UNDERPOWERED` means the state-identity gate is ambiguous under this eval budget.",
            "- `D10R_V6_STATE_SHUFFLE_BLOCKED` means fair projection null passed but shared state shuffling can still match or beat the positive.",
            "- `D10R_V6_RANDOM_STATE_BLOCKED` means no-network random-state controls can still match or beat the positive.",
            "- `D10R_V6_STATE_RANDOM_CONTROL_BLOCKED` means state/no-network controls remain the only artifact blockers.",
            "- `D10R_V5_POSITIVE_CONTROL_FAIL` means beta.8/seed2042 has no positive real signal under this gate.",
            "- Alternate baselines are reported separately and do not create artifact gate failure in `report_only` mode.",
            "",
            "## Progress Map",
            "",
            "```text",
            "[1] beta.8 H384 generalist: existing positive control",
            "[2] D10b seed replication: failed",
            "[3] D10r evaluator trust: CURRENT",
            "    |-- pass -> D10s H384 wiring-prior smoke",
            "    '-- fail -> projection/eval redesign, no H scaling",
            "[4] H512/H8192: blocked until D10r + D10s gates pass",
            "```",
        ]
    )
    (out / "D10R_HARDENED_EVAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_d10r(args) -> dict:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    # D10r is a trust gate for the Rust D9 evaluator, so its charge clamp must
    # match instnct-core/examples/d9_direct_landscape.rs unless overridden.
    gpu_eval.MAX_CHARGE = args.max_charge
    baseline_path = Path(args.baseline)
    baseline = gpu_eval.load_checkpoint(baseline_path)
    checkpoints: list[tuple[str, str, gpu_eval.CheckpointArrays]] = []
    positive_path = Path(args.positive)
    checkpoints.append(("positive", args.positive_label, gpu_eval.load_checkpoint(positive_path)))
    alternate_paths = parse_csv(args.alternate_baseline_checkpoints)
    if not alternate_paths:
        # Backward-compatible alias for older D10r commands.
        alternate_paths = parse_csv(args.negative_checkpoints)
    for raw in alternate_paths:
        path = Path(raw)
        checkpoints.append(("alternate_baseline", checkpoint_label(path), gpu_eval.load_checkpoint(path)))
    for _, label, ckpt in checkpoints:
        if ckpt.network.h != baseline.network.h:
            raise ValueError(f"{label} H={ckpt.network.h} does not match baseline H={baseline.network.h}")
    table, pair_ids, hot_to_idx, bigram, unigram, _ = d10j.load_real_inputs(args)
    eval_seeds = parse_int_csv(args.eval_seeds)
    artifact_controls = parse_csv(args.artifact_controls) or parse_csv(args.controls)
    artifact_controls = augment_artifact_controls(artifact_controls, args.state_zone_diagnostics)
    expanded_controls = expand_controls(artifact_controls, args.control_repeats)
    all_eval_checkpoints = [baseline] + [ckpt for _, _, ckpt in checkpoints]
    labels = ["baseline"] + [label for _, label, _ in checkpoints]
    roles = {label: role for role, label, _ in checkpoints}
    roles["baseline"] = "baseline"
    rows: list[dict] = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for control in [{"label": "real", "base": "real", "repeat": 0}, *expanded_controls]:
        control_type = control["label"]
        control_base = control["base"]
        control_seed = args.seed + d10o.stable_arm_seed(control_type) + 701
        for eval_seed in eval_seeds:
            metrics = evaluate_seed_control(
                all_eval_checkpoints,
                table,
                pair_ids,
                hot_to_idx,
                bigram,
                unigram,
                args.eval_len,
                eval_seed,
                control_base,
                device,
                control_seed,
            )
            base_scores = {metric: float(metrics[metric][0]) for metric in METRICS}
            for idx, label in enumerate(labels):
                scores = {metric: float(metrics[metric][idx]) for metric in METRICS}
                delta = {metric: scores[metric] - base_scores[metric] for metric in METRICS}
                row = {
                    "checkpoint_label": label,
                    "role": roles[label],
                    "control_type": control_type,
                    "control_base": control_base,
                    "control_repeat": control["repeat"],
                    "eval_seed": eval_seed,
                    "entropy_score": float(metrics["entropy"][idx]),
                    "max_prob_score": float(metrics["max_prob"][idx]),
                    "mo_delta": mo_score_from_delta(delta),
                }
                for metric in METRICS:
                    row[f"{metric}_score"] = scores[metric]
                    row[f"{metric}_delta"] = delta[metric]
                rows.append(row)
            print(f"D10r control={control_type} seed={eval_seed} done", flush=True)
    margin_rows: list[dict] = []
    metric_margin_rows: list[dict] = []
    family_bound_rows: list[dict] = []
    zone_bound_rows: list[dict] = []
    score_drift_rows: list[dict] = []
    checkpoint_summaries: list[dict] = []
    expanded_control_labels = [c["label"] for c in expanded_controls]
    projection_rows: list[dict] = []
    projection_zones: dict[str, dict[str, list[int]]] = {}
    for label, ckpt in zip(labels, all_eval_checkpoints):
        p_rows, zones = projection_row_uniqueness_rows(label, ckpt)
        projection_rows.extend(p_rows)
        projection_zones[label] = zones
    for _, label, _ in checkpoints:
        m_rows, summary = summarize_checkpoint(
            label,
            rows,
            expanded_control_labels,
            args.bootstrap_samples,
            args.permutation_samples,
            args.alpha,
            args.seed,
            args.min_real_mo,
            args.min_trusted_mo,
        )
        margin_rows.extend(m_rows)
        metric_margin_rows.extend(
            summarize_metric_margins(
                label,
                rows,
                expanded_control_labels,
                args.bootstrap_samples,
                args.alpha,
                args.seed,
            )
        )
        family_bound_rows.extend(
            summarize_control_family_bounds(
                label,
                rows,
                expanded_control_labels,
                args.bootstrap_samples,
                args.alpha,
                args.seed,
                args.min_trusted_mo,
            )
        )
        zone_counts = {name: len(indices) for name, indices in projection_zones[label].items()}
        zone_bound_rows.extend(
            summarize_state_shuffle_zone_bounds(
                label,
                rows,
                expanded_control_labels,
                zone_counts,
                len(projection_zones[label].get("all_rows", [])),
                args.bootstrap_samples,
                args.alpha,
                args.seed,
                args.min_trusted_mo,
            )
        )
        score_drift_rows.extend(summarize_state_shuffle_score_drift(label, rows, expanded_control_labels))
        checkpoint_summaries.append(summary)
    positive_summary = next(row for row in checkpoint_summaries if row["checkpoint_label"] == args.positive_label)
    positive_family_bounds = [row for row in family_bound_rows if row["checkpoint_label"] == args.positive_label]
    blocking_control_families = [
        row["control_base"]
        for row in positive_family_bounds
        if float(row["bound_ci_low"]) <= args.min_trusted_mo
    ]
    alternate_baseline_signals = [
        row["checkpoint_label"]
        for row in checkpoint_summaries
        if roles.get(row["checkpoint_label"]) == "alternate_baseline" and row["trusted_mo_pass"]
    ]
    artifact_gate_pass = bool(positive_summary["trusted_mo_pass"])
    state_bound = next(
        (
            row
            for row in positive_family_bounds
            if row["control_base"] == "state_shuffle_shared"
        ),
        None,
    )
    if artifact_gate_pass:
        verdict = "D10R_V8_STATE_IDENTITY_PASS"
    elif positive_summary["real_mo_delta_ci_low"] > args.min_real_mo:
        blocking_bases = set(blocking_control_families)
        if blocking_bases == {"state_shuffle_shared"}:
            if v8_projection_symmetry_supported(zone_bound_rows, args.positive_label):
                verdict = "D10R_V8_PROJECTION_SYMMETRY_CONFIRMED"
            elif state_bound and float(state_bound["bound_mean"]) > args.min_trusted_mo and float(state_bound["bound_ci_high"]) > args.min_trusted_mo:
                verdict = "D10R_V8_UNDERPOWERED"
            else:
                verdict = "D10R_V8_WEAK_STATE_IDENTITY_FAIL"
        elif blocking_bases == {"no_network_random_state"}:
            verdict = "D10R_V6_RANDOM_STATE_BLOCKED"
        elif blocking_bases and blocking_bases <= {"state_shuffle_shared", "no_network_random_state"}:
            verdict = "D10R_V6_STATE_RANDOM_CONTROL_BLOCKED"
        elif blocking_bases:
            verdict = "D10R_V5_ARTIFACT_READOUT_BLOCKED"
        else:
            # Keep the pre-v7 per-control diagnosis as a fallback when family
            # bounds pass but an individual Holm/per-control row remains weak.
            blocking_bases = {control_base_name(name) for name in positive_summary["blocking_controls"]}
            if blocking_bases == {"state_shuffle_shared"}:
                verdict = "D10R_V6_STATE_SHUFFLE_BLOCKED"
            elif blocking_bases == {"no_network_random_state"}:
                verdict = "D10R_V6_RANDOM_STATE_BLOCKED"
            elif blocking_bases and blocking_bases <= {"state_shuffle_shared", "no_network_random_state"}:
                verdict = "D10R_V6_STATE_RANDOM_CONTROL_BLOCKED"
            elif blocking_bases:
                verdict = "D10R_V5_ARTIFACT_READOUT_BLOCKED"
            else:
                verdict = "D10R_UNDERPOWERED_NEEDS_LONGER_EVAL"
    elif positive_summary["trusted_mo_mean"] > args.min_trusted_mo:
        verdict = "D10R_UNDERPOWERED_NEEDS_LONGER_EVAL"
    else:
        verdict = "D10R_V5_POSITIVE_CONTROL_FAIL"
    alternate_baseline_verdict = (
        "D10R_V5_ALTERNATE_BASELINE_SIGNAL"
        if alternate_baseline_signals
        else "D10R_V5_NO_ALTERNATE_BASELINE_SIGNAL"
    )
    elapsed_s = time.perf_counter() - started
    peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if device.type == "cuda" else 0.0
    run_summary = {
        "verdict": verdict,
        "setup": {
            "baseline": str(baseline_path),
            "positive": str(positive_path),
            "positive_label": args.positive_label,
            "artifact_controls": artifact_controls,
            "alternate_baseline_checkpoints": alternate_paths,
            "alternate_baseline_mode": args.alternate_baseline_mode,
            "negative_checkpoints": parse_csv(args.negative_checkpoints),
            "eval_len": args.eval_len,
            "eval_seeds": eval_seeds,
            "controls": artifact_controls,
            "expanded_controls": [c["label"] for c in expanded_controls],
            "control_repeats": args.control_repeats,
            "state_zone_diagnostics": args.state_zone_diagnostics,
            "max_charge": args.max_charge,
            "bootstrap_samples": args.bootstrap_samples,
            "permutation_samples": args.permutation_samples,
            "alpha": args.alpha,
            "device": args.device,
        },
        "roles": roles,
        "checkpoint_summaries": checkpoint_summaries,
        "blocking_control_families": blocking_control_families,
        "state_zone_diagnostics": args.state_zone_diagnostics,
        "artifact_gate_pass": artifact_gate_pass,
        "d10s_unlocked": artifact_gate_pass,
        "alternate_baseline_verdict": alternate_baseline_verdict,
        "alternate_baseline_signals": alternate_baseline_signals,
        "negative_passes": [],
        "elapsed_s": elapsed_s,
        "peak_mb": peak_mb,
    }
    write_csv(
        out / "d10r_eval_results.csv",
        rows,
        [
            "checkpoint_label",
            "role",
            "control_type",
            "control_base",
            "control_repeat",
            "eval_seed",
            "entropy_score",
            "max_prob_score",
            "smooth_score",
            "smooth_delta",
            "accuracy_score",
            "accuracy_delta",
            "echo_score",
            "echo_delta",
            "unigram_score",
            "unigram_delta",
            "mo_delta",
        ],
    )
    write_csv(
        out / "d10r_control_margins.csv",
        margin_rows,
        [
            "checkpoint_label",
            "control_type",
            "n",
            "margin_mean",
            "margin_ci_low",
            "margin_ci_high",
            "p_value",
            "holm_p_value",
            "holm_threshold",
            "holm_reject",
            "pass",
        ],
    )
    write_csv(
        out / "d10r_metric_margins.csv",
        metric_margin_rows,
        [
            "checkpoint_label",
            "control_type",
            "control_base",
            "metric",
            "n",
            "margin_mean",
            "margin_ci_low",
            "margin_ci_high",
            "worst_seed",
            "worst_margin",
        ],
    )
    write_csv(
        out / "d10r_control_family_bounds.csv",
        family_bound_rows,
        [
            "checkpoint_label",
            "control_base",
            "control_count",
            "n",
            "bound_mean",
            "bound_ci_low",
            "bound_ci_high",
            "worst_seed",
            "worst_control",
            "worst_margin",
            "pass",
        ],
    )
    write_csv(
        out / "projection_row_uniqueness.csv",
        projection_rows,
        [
            "checkpoint_label",
            "projection_input_dim",
            "projection_output_classes",
            "row_idx",
            "row_hash",
            "row_l1",
            "row_l2",
            "row_nonzero",
            "duplicate_group",
            "duplicate_count",
            "nearest_row",
            "nearest_cosine",
            "is_active",
            "is_high_norm",
            "is_low_norm",
            "is_duplicate_projection",
            "is_similar_projection",
            "active_fraction",
            "duplicate_fraction",
            "similar_fraction",
        ],
    )
    write_csv(
        out / "state_shuffle_zone_bounds.csv",
        zone_bound_rows,
        [
            "checkpoint_label",
            "control_type",
            "control_base",
            "control_count",
            "zone_name",
            "row_count",
            "row_fraction",
            "n",
            "bound_mean",
            "bound_ci_low",
            "bound_ci_high",
            "worst_seed",
            "worst_control",
            "worst_margin",
            "diagnostic",
            "pass",
        ],
    )
    write_csv(
        out / "state_shuffle_score_drift.csv",
        score_drift_rows,
        [
            "checkpoint_label",
            "control_type",
            "control_base",
            "zone_name",
            "eval_seed",
            "real_smooth_delta",
            "control_smooth_delta",
            "smooth_margin",
            "real_accuracy_delta",
            "control_accuracy_delta",
            "accuracy_margin",
            "real_echo_delta",
            "control_echo_delta",
            "echo_margin",
            "real_unigram_delta",
            "control_unigram_delta",
            "unigram_margin",
            "real_mo_delta",
            "control_mo_delta",
            "mo_margin",
        ],
    )
    (out / "d10r_run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    write_report(out, run_summary, checkpoint_summaries, margin_rows, metric_margin_rows, family_bound_rows, zone_bound_rows)
    return run_summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="output/phase_d7_operator_bandit_20260427/H_384/D7_BASELINE/seed_2042/final.ckpt")
    parser.add_argument("--positive", default="output/releases/v5.0.0-beta.8/seed2042_improved_generalist_v1.ckpt")
    parser.add_argument("--positive-label", default="beta8_generalist")
    parser.add_argument(
        "--negative-checkpoints",
        default="",
        help="Deprecated alias; D10r-v5 treats these as alternate baseline checkpoints.",
    )
    parser.add_argument("--artifact-controls", default=",".join(PRIMARY_CONTROLS))
    parser.add_argument("--alternate-baseline-checkpoints", default=",".join(DEFAULT_ALTERNATE_BASELINES))
    parser.add_argument("--alternate-baseline-mode", choices=["report_only"], default="report_only")
    parser.add_argument("--packed", default="output/block_c_bytepair_champion/packed.bin")
    parser.add_argument("--corpus", default="instnct-core/tests/fixtures/alice_corpus.txt")
    parser.add_argument("--out", default="output/phase_d10r_hardened_eval_20260430")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--eval-len", type=int, default=1000)
    parser.add_argument("--eval-seeds", default="991001,991002,991003,991004")
    # Legacy alias retained so older command lines still work. D10r-v5 uses
    # --artifact-controls when provided.
    parser.add_argument("--controls", default=",".join(PRIMARY_CONTROLS))
    parser.add_argument("--control-repeats", type=int, default=1)
    parser.add_argument("--state-zone-diagnostics", dest="state_zone_diagnostics", action="store_true", default=True)
    parser.add_argument("--no-state-zone-diagnostics", dest="state_zone_diagnostics", action="store_false")
    parser.add_argument("--max-charge", type=int, default=7)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--permutation-samples", type=int, default=5000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--min-real-mo", type=float, default=0.0)
    parser.add_argument("--min-trusted-mo", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260430)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    summary = run_d10r(args)
    print(json.dumps({"verdict": summary["verdict"], "elapsed_s": summary["elapsed_s"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
