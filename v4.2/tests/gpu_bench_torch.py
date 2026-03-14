"""Torch CUDA parity + speed benchmark for the current v4.2 graph core.

This is a narrow benchmark for the real hot path:
  - build Weff from int8 ternary mask + fixed gain
  - run the 8-tick batch forward pass
  - compare three paths:
      1. dense NumPy reference
      2. current CPU implementation from graph.py
      3. dense Torch CUDA

The goal is to answer one question quickly:
  Is a straight dense CUDA port already enough to justify GPU work?
"""

import os
import sys
import time
import random
from dataclasses import dataclass

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.graph import SelfWiringGraph


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
CPU_ITERS = 80
GPU_ITERS = 400
WARMUP = 50


def build_net(cfg: BenchConfig) -> SelfWiringGraph:
    np.random.seed(SEED)
    random.seed(SEED)
    return SelfWiringGraph(cfg.neurons, cfg.vocab, density=cfg.density)


def torch_forward_batch(
    mask_i8: torch.Tensor,
    vocab: int,
    out_start: int,
    threshold: float,
    clip_factor: float,
    self_conn: float,
    charge_rate: float,
    leak: float,
    gain: float,
    ticks: int,
) -> torch.Tensor:
    weff = mask_i8.to(torch.float32) * gain
    n = mask_i8.shape[0]
    clip_bound = threshold * clip_factor
    charges = torch.zeros((vocab, n), dtype=torch.float32, device=mask_i8.device)
    acts = torch.zeros((vocab, n), dtype=torch.float32, device=mask_i8.device)
    eye = torch.eye(vocab, dtype=torch.float32, device=mask_i8.device)

    for t in range(ticks):
        if t == 0:
            acts[:, :vocab] = eye
        raw = acts @ weff + acts * self_conn
        raw = torch.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        charges = charges + raw * charge_rate
        charges = charges * leak
        acts = torch.clamp(charges - threshold, min=0.0)
        charges = torch.clamp(charges, -clip_bound, clip_bound)

    return charges[:, out_start : out_start + vocab]


def numpy_dense_forward_batch(net: SelfWiringGraph, ticks: int) -> np.ndarray:
    v, n = net.V, net.N
    weff = net.mask.astype(np.float32) * net.gain
    clip_bound = net.threshold * net.clip_factor
    charges = np.zeros((v, n), dtype=np.float32)
    acts = np.zeros((v, n), dtype=np.float32)

    for t in range(ticks):
        if t == 0:
            acts[:, :v] = np.eye(v, dtype=np.float32)
        raw = acts @ weff + acts * net.self_conn
        np.nan_to_num(raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        charges += raw * net.charge_rate
        charges *= net.leak
        acts = np.maximum(charges - net.threshold, 0.0)
        charges = np.clip(charges, -clip_bound, clip_bound)

    return charges[:, net.out_start : net.out_start + v]


def bench_dense_numpy(net: SelfWiringGraph, iters: int) -> tuple[np.ndarray, float]:
    out = numpy_dense_forward_batch(net, TICKS)
    t0 = time.perf_counter()
    for _ in range(iters):
        out = numpy_dense_forward_batch(net, TICKS)
    dt = time.perf_counter() - t0
    return out, dt


def bench_cpu_impl(net: SelfWiringGraph, iters: int) -> tuple[np.ndarray, float]:
    out = net.forward_batch(TICKS)
    t0 = time.perf_counter()
    for _ in range(iters):
        out = net.forward_batch(TICKS)
    dt = time.perf_counter() - t0
    return out, dt


def bench_gpu(net: SelfWiringGraph, iters: int) -> tuple[np.ndarray, float]:
    device = torch.device("cuda")
    mask_i8 = torch.from_numpy(net.mask).to(device=device, dtype=torch.int8)

    for _ in range(WARMUP):
        out = torch_forward_batch(
            mask_i8,
            net.V,
            net.out_start,
            net.threshold,
            net.clip_factor,
            net.self_conn,
            net.charge_rate,
            net.leak,
            net.gain,
            TICKS,
        )
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        out = torch_forward_batch(
            mask_i8,
            net.V,
            net.out_start,
            net.threshold,
            net.clip_factor,
            net.self_conn,
            net.charge_rate,
            net.leak,
            net.gain,
            TICKS,
        )
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return out.detach().cpu().numpy(), dt


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA not available.")
        return 1

    torch.set_float32_matmul_precision("high")
    print(
        f"GPU BENCH: torch={torch.__version__} "
        f"device={torch.cuda.get_device_name(0)} "
        f"ticks={TICKS}"
    )
    print("=" * 88)
    print(
        f"{'config':12s} {'dense_ms':>10s} {'cpu_ms':>10s} {'gpu_ms':>10s} "
        f"{'cpu_x':>8s} {'gpu_x':>8s} {'diff_cpu':>11s} {'diff_gpu':>11s}"
    )

    for cfg in CONFIGS:
        net = build_net(cfg)
        dense_out, dense_dt = bench_dense_numpy(net, CPU_ITERS)
        cpu_out, cpu_dt = bench_cpu_impl(net, CPU_ITERS)
        gpu_out, gpu_dt = bench_gpu(net, GPU_ITERS)

        dense_ms = (dense_dt / CPU_ITERS) * 1000.0
        cpu_ms = (cpu_dt / CPU_ITERS) * 1000.0
        gpu_ms = (gpu_dt / GPU_ITERS) * 1000.0
        diff_cpu = float(np.max(np.abs(dense_out - cpu_out)))
        diff_gpu = float(np.max(np.abs(dense_out - gpu_out)))
        cpu_speedup = dense_ms / cpu_ms if cpu_ms > 0 else float("inf")
        gpu_speedup = dense_ms / gpu_ms if gpu_ms > 0 else float("inf")

        print(
            f"{cfg.name:12s} {dense_ms:10.3f} {cpu_ms:10.3f} {gpu_ms:10.3f} "
            f"{cpu_speedup:8.2f} {gpu_speedup:8.2f} {diff_cpu:11.3e} {diff_gpu:11.3e}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
