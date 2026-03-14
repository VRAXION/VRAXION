"""Torch CUDA parity + speed benchmark for v4.2 baselines.

This forward-only probe freezes two CPU baselines explicitly:
  - CPU_DENSE_COMMITTED: exact origin/v4.2 graph.py
  - CPU_SPARSE_LOCAL: current local graph.py
  - CUDA_DENSE: dense Torch implementation on the current device
"""

import time
from dataclasses import dataclass

import numpy as np
import torch

from graph_baseline_loader import build_paired_nets


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


def numpy_dense_forward_batch(net, ticks: int) -> np.ndarray:
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


def bench_dense_numpy(net, iters: int) -> tuple[np.ndarray, float]:
    out = numpy_dense_forward_batch(net, TICKS)
    t0 = time.perf_counter()
    for _ in range(iters):
        out = numpy_dense_forward_batch(net, TICKS)
    dt = time.perf_counter() - t0
    return out, dt


def bench_cpu_impl(net, iters: int) -> tuple[np.ndarray, float]:
    out = net.forward_batch(TICKS)
    t0 = time.perf_counter()
    for _ in range(iters):
        out = net.forward_batch(TICKS)
    dt = time.perf_counter() - t0
    return out, dt


def bench_gpu(net, iters: int) -> tuple[np.ndarray, float]:
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
        f"{'config':12s} {'dense_ms':>10s} {'sparse_ms':>10s} {'gpu_ms':>10s} "
        f"{'sparse_x':>9s} {'gpu_x':>8s} {'diff_sparse':>12s} {'diff_gpu':>11s} "
        f"{'agree_sparse':>13s} {'agree_gpu':>10s}"
    )

    for cfg in CONFIGS:
        _, dense_net, _, sparse_net = build_paired_nets(cfg.vocab, cfg.neurons, cfg.density, SEED)
        dense_out, dense_dt = bench_dense_numpy(dense_net, CPU_ITERS)
        sparse_out, sparse_dt = bench_cpu_impl(sparse_net, CPU_ITERS)
        gpu_out, gpu_dt = bench_gpu(dense_net, GPU_ITERS)

        dense_ms = (dense_dt / CPU_ITERS) * 1000.0
        sparse_ms = (sparse_dt / CPU_ITERS) * 1000.0
        gpu_ms = (gpu_dt / GPU_ITERS) * 1000.0
        diff_sparse = float(np.max(np.abs(dense_out - sparse_out)))
        diff_gpu = float(np.max(np.abs(dense_out - gpu_out)))
        agree_sparse = float(
            (np.argmax(dense_out, axis=1) == np.argmax(sparse_out, axis=1)).mean()
        )
        agree_gpu = float(
            (np.argmax(dense_out, axis=1) == np.argmax(gpu_out, axis=1)).mean()
        )
        sparse_speedup = dense_ms / sparse_ms if sparse_ms > 0 else float("inf")
        gpu_speedup = dense_ms / gpu_ms if gpu_ms > 0 else float("inf")

        print(
            f"{cfg.name:12s} {dense_ms:10.3f} {sparse_ms:10.3f} {gpu_ms:10.3f} "
            f"{sparse_speedup:9.2f} {gpu_speedup:8.2f} {diff_sparse:12.3e} {diff_gpu:11.3e} "
            f"{agree_sparse:13.3f} {agree_gpu:10.3f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
