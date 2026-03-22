"""GPU A/B: eval dtype simplification under the current random-first proposal path.

Question:
  How far can we simplify the float-heavy eval side on GPU before quality breaks?

This keeps proposal generation fixed and only changes eval buffers/math dtype:
  - fp32 baseline
  - bf16 candidate
  - fp16 candidate
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch._dynamo

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.log import live_log, log_msg
from tests.gpu_int_mood_ab import (
    CLIP_BOUND,
    CONFIGS,
    GAIN,
    CHARGE_RATE,
    SELF_CONN,
    THRESHOLD,
    TICKS,
    add_connection,
    flip_connection,
    gpu_init,
    mutate_int,
    parse_csv_ints,
    remove_connection,
    rewire_connection,
    rollback,
)


DTYPES = ("fp32", "bf16", "fp16")
DEFAULT_CONFIGS = "V64_N192,V128_N384"
DEFAULT_ATTEMPTS = 4000
DEFAULT_SEEDS = "42,77,123"
CHECKPOINT_EVERY = 2000

torch.set_float32_matmul_precision("high")
torch._dynamo.config.suppress_errors = True


@dataclass
class DtypeEvalBuffers:
    eye: torch.Tensor
    charges: torch.Tensor
    acts: torch.Tensor
    weff: torch.Tensor
    row_idx: torch.Tensor
    scalar_gain: torch.Tensor
    scalar_self: torch.Tensor
    scalar_charge_rate: torch.Tensor
    scalar_threshold: torch.Tensor
    scalar_clip: torch.Tensor


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=DEFAULT_CONFIGS)
    ap.add_argument("--attempts", type=int, default=DEFAULT_ATTEMPTS)
    ap.add_argument("--seeds", default=DEFAULT_SEEDS)
    ap.add_argument("--dtypes", default="fp32,bf16,fp16")
    ap.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY)
    ap.add_argument("--log-name", default="gpu_eval_dtype_ab")
    return ap.parse_args()


def parse_csv(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def torch_dtype_of(name: str) -> torch.dtype:
    if name == "fp32":
        return torch.float32
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    raise ValueError(f"unknown dtype mode: {name}")


def leak_for_dtype(leak: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return leak.to(dtype=dtype)


def make_eval_buffers(vocab: int, neurons: int, device: torch.device, dtype_mode: str) -> DtypeEvalBuffers:
    dt = torch_dtype_of(dtype_mode)
    return DtypeEvalBuffers(
        eye=torch.eye(vocab, dtype=dt, device=device),
        charges=torch.empty((vocab, neurons), dtype=dt, device=device),
        acts=torch.empty((vocab, neurons), dtype=dt, device=device),
        weff=torch.empty((neurons, neurons), dtype=dt, device=device),
        row_idx=torch.arange(vocab, device=device, dtype=torch.long),
        scalar_gain=torch.tensor(GAIN, dtype=dt, device=device),
        scalar_self=torch.tensor(SELF_CONN, dtype=dt, device=device),
        scalar_charge_rate=torch.tensor(CHARGE_RATE, dtype=dt, device=device),
        scalar_threshold=torch.tensor(THRESHOLD, dtype=dt, device=device),
        scalar_clip=torch.tensor(CLIP_BOUND, dtype=dt, device=device),
    )


def gpu_eval_dtype(
    mask: torch.Tensor,
    leak: torch.Tensor,
    targets: torch.Tensor,
    out_start: int,
    buffers: DtypeEvalBuffers,
):
    eye = buffers.eye
    charges = buffers.charges
    acts = buffers.acts
    weff = buffers.weff
    row_idx = buffers.row_idx
    scalar_gain = buffers.scalar_gain
    scalar_self = buffers.scalar_self
    scalar_charge_rate = buffers.scalar_charge_rate
    scalar_threshold = buffers.scalar_threshold
    scalar_clip = buffers.scalar_clip
    leak_t = leak_for_dtype(leak, charges.dtype)

    vocab = eye.shape[0]
    charges.zero_()
    acts.zero_()
    weff.copy_(mask)
    weff.mul_(scalar_gain)

    for t in range(TICKS):
        if t == 0:
            acts[:, :vocab] = eye
        raw = acts @ weff + acts * scalar_self
        raw = torch.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        charges += raw * scalar_charge_rate
        charges *= leak_t
        acts = torch.clamp(charges - scalar_threshold, min=0.0)
        charges = torch.clamp(charges, -scalar_clip, scalar_clip)

    logits = charges[:, out_start : out_start + vocab].to(torch.float32)
    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)
    acc = (preds == targets).to(torch.float32).mean()
    tp = probs[row_idx, targets].mean()
    score = 0.5 * acc + 0.5 * tp
    return score, acc


def make_eval_runner(
    vocab: int,
    neurons: int,
    targets: torch.Tensor,
    out_start: int,
    device: torch.device,
    dtype_mode: str,
    compile_eval: bool = True,
):
    buffers = make_eval_buffers(vocab, neurons, device, dtype_mode)

    def eval_runner(mask: torch.Tensor, leak: torch.Tensor):
        return gpu_eval_dtype(mask, leak, targets, out_start, buffers)

    if compile_eval:
        return torch.compile(eval_runner, mode="reduce-overhead", fullgraph=False)
    return eval_runner


def run_one(config_name: str, seed: int, attempts: int, dtype_mode: str, checkpoint_every: int, log_q=None):
    vocab, neurons, density = CONFIGS[config_name]
    device = torch.device("cuda")
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    mask, leak, targets, out_start = gpu_init(vocab, neurons, density, seed, device)
    diag_mask = ~torch.eye(neurons, dtype=torch.bool, device=device)
    eval_runner = make_eval_runner(vocab, neurons, targets, out_start, device, dtype_mode)
    controller = {"kind": "int", "mood": 2, "intensity": 7}

    score, acc = eval_runner(mask, leak)
    best_score = score.clone()
    best_acc = acc.clone()
    kept = 0

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for att in range(1, attempts + 1):
        prev, changes = mutate_int(mask, leak, controller, gen, diag_mask)
        new_score, new_acc = eval_runner(mask, leak)
        if bool((new_score > score).item()):
            score = new_score
            kept += 1
            if bool((new_score > best_score).item()):
                best_score = new_score
                best_acc = new_acc
        else:
            rollback(mask, leak, controller, prev, changes)

        if att % checkpoint_every == 0:
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            aps = att / dt if dt > 0 else float("inf")
            log_msg(
                log_q,
                f"{config_name:10s} {dtype_mode:4s} seed={seed:3d} att={att:5d} "
                f"best_acc={float(best_acc.item())*100:5.1f}% score={float(best_score.item()):.4f} aps={aps:.1f}",
            )

    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    row = {
        "config": config_name,
        "dtype_mode": dtype_mode,
        "seed": seed,
        "attempts": attempts,
        "best_acc": float(best_acc.item()),
        "best_score": float(best_score.item()),
        "attempts_per_sec": attempts / dt if dt > 0 else float("inf"),
        "kept": kept,
        "final_leak": float(leak.item()),
    }
    log_msg(log_q, "RESULT_JSON " + json.dumps(row, sort_keys=True))
    return row


def summarize(rows: list[dict], log_q) -> None:
    log_msg(log_q, "")
    log_msg(log_q, "SUMMARY")
    configs = sorted({r["config"] for r in rows})
    for config in configs:
        for dtype_mode in DTYPES:
            mode_rows = [r for r in rows if r["config"] == config and r["dtype_mode"] == dtype_mode]
            if not mode_rows:
                continue
            payload = {
                "config": config,
                "dtype_mode": dtype_mode,
                "mean_acc": float(np.mean([r["best_acc"] for r in mode_rows])),
                "std_acc": float(np.std([r["best_acc"] for r in mode_rows])),
                "mean_score": float(np.mean([r["best_score"] for r in mode_rows])),
                "mean_aps": float(np.mean([r["attempts_per_sec"] for r in mode_rows])),
            }
            log_msg(log_q, "SUMMARY_JSON " + json.dumps(payload, sort_keys=True))


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")

    args = parse_args()
    configs = parse_csv(args.configs)
    seeds = parse_csv_ints(args.seeds)
    dtypes = parse_csv(args.dtypes)
    for config in configs:
        if config not in CONFIGS:
            raise SystemExit(f"Unknown config: {config}")
    for dtype_mode in dtypes:
        if dtype_mode not in DTYPES:
            raise SystemExit(f"Unknown dtype mode: {dtype_mode}")
        if dtype_mode == "bf16" and not torch.cuda.is_bf16_supported():
            raise SystemExit("bf16 requested but CUDA bf16 is not supported on this device")

    rows = []
    with live_log(args.log_name) as (log_q, log_path):
        log_msg(
            log_q,
            f"GPU EVAL DTYPE AB configs={configs} dtypes={dtypes} seeds={seeds} attempts={args.attempts}",
        )
        log_msg(log_q, "=" * 120)
        for config in configs:
            for dtype_mode in dtypes:
                for seed in seeds:
                    rows.append(run_one(config, seed, args.attempts, dtype_mode, args.checkpoint_every, log_q=log_q))
        summarize(rows, log_q)
        log_msg(log_q, f"LOG_PATH {log_path}")


if __name__ == "__main__":
    raise SystemExit(main())
