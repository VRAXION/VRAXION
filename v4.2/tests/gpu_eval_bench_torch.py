"""Deterministic full-eval benchmark for v4.2 dense CPU vs sparse CPU vs CUDA.

This measures an evolutionary "attempt" as:
  one candidate mask/leak snapshot -> one full batched evaluation -> one score

Mutation remains on CPU; the GPU path therefore includes per-attempt mask upload.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

from graph_baseline_loader import build_paired_nets, clone_common_state


@dataclass
class BenchConfig:
    name: str
    vocab: int
    neurons: int
    density: float


CONFIGS = [
    BenchConfig("V64_N192", 64, 192, 0.06),
    BenchConfig("V64_dense", 64, 192, 0.15),
    BenchConfig("V128_N384", 128, 384, 0.06),
]

SEED = 42
TICKS = 8
N_CANDIDATES = 64
GPU_WARMUP = 20


def torch_eval_candidate(mask_i8: torch.Tensor, leak: float, vocab: int, neurons: int,
                         threshold: float, clip_factor: float, self_conn: float,
                         charge_rate: float, gain: float, out_start: int,
                         targets: torch.Tensor) -> tuple[torch.Tensor, float]:
    weff = mask_i8.to(torch.float32) * gain
    charges = torch.zeros((vocab, neurons), dtype=torch.float32, device=mask_i8.device)
    acts = torch.zeros((vocab, neurons), dtype=torch.float32, device=mask_i8.device)
    eye = torch.eye(vocab, dtype=torch.float32, device=mask_i8.device)
    clip_bound = threshold * clip_factor

    for t in range(TICKS):
        if t == 0:
            acts[:, :vocab] = eye
        raw = acts @ weff + acts * self_conn
        raw = torch.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        charges = charges + raw * charge_rate
        charges = charges * leak
        acts = torch.clamp(charges - threshold, min=0.0)
        charges = torch.clamp(charges, -clip_bound, clip_bound)

    logits = charges[:, out_start : out_start + vocab]
    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)
    acc = (preds == targets).to(torch.float32).mean()
    tp = probs[torch.arange(vocab, device=mask_i8.device), targets].mean()
    score = float((0.5 * acc + 0.5 * tp).item())
    return logits, score


def generate_candidates(cfg: BenchConfig):
    dense_mod, dense_net, sparse_mod, sparse_net = build_paired_nets(
        cfg.vocab, cfg.neurons, cfg.density, SEED
    )
    perm = dense_mod.np.random.permutation(cfg.vocab)
    targets = perm.astype(np.int64)

    candidates = []
    for _ in range(N_CANDIDATES):
        dense_net.mutate_with_mood()
        candidates.append((dense_net.mask.copy(), float(dense_net.leak)))
    clone_common_state(dense_net, sparse_net)

    return dense_mod, dense_net, sparse_mod, sparse_net, targets, candidates


def bench_cpu_eval(net, targets: np.ndarray, candidates) -> tuple[float, list[np.ndarray], list[float]]:
    logits_out = []
    scores = []
    t0 = time.perf_counter()
    for mask, leak in candidates:
        net.mask[:] = mask
        net.leak = leak
        if hasattr(net, "_weff_dirty"):
            net._weff_dirty = True
        logits = net.forward_batch(TICKS)
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        preds = np.argmax(probs, axis=1)
        acc = (preds == targets).mean()
        tp = probs[np.arange(net.V), targets].mean()
        logits_out.append(logits.copy())
        scores.append(float(0.5 * acc + 0.5 * tp))
    dt = time.perf_counter() - t0
    return dt, logits_out, scores


def bench_gpu_eval(cfg: BenchConfig, dense_net, targets: np.ndarray, candidates) -> tuple[float, list[np.ndarray], list[float]]:
    device = torch.device("cuda")
    targets_t = torch.from_numpy(targets).to(device=device, dtype=torch.long)

    for i in range(min(GPU_WARMUP, len(candidates))):
        mask_i8 = torch.from_numpy(candidates[i][0]).to(device=device, dtype=torch.int8)
        torch_eval_candidate(
            mask_i8,
            candidates[i][1],
            dense_net.V,
            dense_net.N,
            dense_net.threshold,
            dense_net.clip_factor,
            dense_net.self_conn,
            dense_net.charge_rate,
            dense_net.gain,
            dense_net.out_start,
            targets_t,
        )
    torch.cuda.synchronize()

    logits_out = []
    scores = []
    t0 = time.perf_counter()
    for mask, leak in candidates:
        mask_i8 = torch.from_numpy(mask).to(device=device, dtype=torch.int8)
        logits_t, score = torch_eval_candidate(
            mask_i8,
            leak,
            dense_net.V,
            dense_net.N,
            dense_net.threshold,
            dense_net.clip_factor,
            dense_net.self_conn,
            dense_net.charge_rate,
            dense_net.gain,
            dense_net.out_start,
            targets_t,
        )
        logits_out.append(logits_t.detach().cpu().numpy())
        scores.append(score)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return dt, logits_out, scores


def summarize_parity(ref_logits, other_logits, ref_scores, other_scores):
    max_abs = 0.0
    pred_agree = []
    score_abs = 0.0
    for ref, other, s_ref, s_other in zip(ref_logits, other_logits, ref_scores, other_scores):
        max_abs = max(max_abs, float(np.max(np.abs(ref - other))))
        pred_agree.append(float((np.argmax(ref, axis=1) == np.argmax(other, axis=1)).mean()))
        score_abs = max(score_abs, abs(float(s_ref) - float(s_other)))
    return max_abs, float(np.mean(pred_agree)), float(score_abs)


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA not available.")
        return 1

    print(
        f"FULL EVAL GPU BENCH: device={torch.cuda.get_device_name(0)} "
        f"ticks={TICKS} candidates={N_CANDIDATES}"
    )
    print("=" * 128)
    print(
        f"{'config':12s} {'dense_ms':>10s} {'sparse_ms':>10s} {'gpu_ms':>10s} "
        f"{'dense_att/s':>11s} {'sparse_att/s':>12s} {'gpu_att/s':>10s} "
        f"{'gpu_vs_sparse':>14s} {'diff_sparse':>12s} {'diff_gpu':>12s} "
        f"{'agree_sparse':>13s} {'agree_gpu':>10s}"
    )

    v128_gpu_vs_sparse = None
    for cfg in CONFIGS:
        dense_mod, dense_net, sparse_mod, sparse_net, targets, candidates = generate_candidates(cfg)
        dense_dt, dense_logits, dense_scores = bench_cpu_eval(dense_net, targets, candidates)
        sparse_dt, sparse_logits, sparse_scores = bench_cpu_eval(sparse_net, targets, candidates)
        gpu_dt, gpu_logits, gpu_scores = bench_gpu_eval(cfg, dense_net, targets, candidates)

        dense_ms = dense_dt * 1000.0 / len(candidates)
        sparse_ms = sparse_dt * 1000.0 / len(candidates)
        gpu_ms = gpu_dt * 1000.0 / len(candidates)
        dense_att_s = len(candidates) / dense_dt if dense_dt > 0 else float("inf")
        sparse_att_s = len(candidates) / sparse_dt if sparse_dt > 0 else float("inf")
        gpu_att_s = len(candidates) / gpu_dt if gpu_dt > 0 else float("inf")
        sparse_diff, sparse_agree, sparse_score_diff = summarize_parity(dense_logits, sparse_logits, dense_scores, sparse_scores)
        gpu_diff, gpu_agree, gpu_score_diff = summarize_parity(dense_logits, gpu_logits, dense_scores, gpu_scores)
        gpu_vs_sparse = sparse_ms / gpu_ms if gpu_ms > 0 else float("inf")
        if cfg.name == "V128_N384":
            v128_gpu_vs_sparse = gpu_vs_sparse

        print(
            f"{cfg.name:12s} {dense_ms:10.3f} {sparse_ms:10.3f} {gpu_ms:10.3f} "
            f"{dense_att_s:11.1f} {sparse_att_s:12.1f} {gpu_att_s:10.1f} "
            f"{gpu_vs_sparse:14.2f} {sparse_diff:12.3e} {gpu_diff:12.3e} "
            f"{sparse_agree:13.3f} {gpu_agree:10.3f}"
        )

    if v128_gpu_vs_sparse is not None:
        if v128_gpu_vs_sparse < 1.2:
            verdict = "STOP_GPU_BELOW_THRESHOLD"
        elif v128_gpu_vs_sparse >= 1.5:
            verdict = "PROMOTE_GPU_FULL_EVAL"
        else:
            verdict = "MIXED_KEEP_CPU_DEFAULT"
        print(f"\nDECISION V128_N384 gpu_vs_sparse={v128_gpu_vs_sparse:.2f} verdict={verdict}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
