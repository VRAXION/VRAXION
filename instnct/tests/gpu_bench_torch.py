"""Torch CUDA parity + speed benchmark for the projection-based SWG forward path."""

from __future__ import annotations

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
    mask: torch.Tensor,
    input_projection: torch.Tensor,
    output_projection: torch.Tensor,
    theta: torch.Tensor,
    decay: torch.Tensor,
    ticks: int,
) -> torch.Tensor:
    vocab, hidden = input_projection.shape
    charges = torch.zeros((vocab, hidden), dtype=torch.float32, device=mask.device)
    acts = torch.zeros((vocab, hidden), dtype=torch.float32, device=mask.device)
    projected = input_projection
    retention = 1.0 - decay

    for t in range(ticks):
        if t == 0:
            acts = acts + projected
        raw = acts @ mask
        raw = torch.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        charges = charges + raw
        charges = charges * retention
        acts = torch.clamp(charges - theta, min=0.0)
        charges = torch.clamp(charges, min=0.0)

    return charges @ output_projection


def bench_cpu_impl(net, iters: int) -> tuple[np.ndarray, float]:
    out = net.forward_batch(TICKS)
    t0 = time.perf_counter()
    for _ in range(iters):
        out = net.forward_batch(TICKS)
    dt = time.perf_counter() - t0
    return out, dt


def bench_gpu(net, iters: int) -> tuple[np.ndarray, float]:
    device = torch.device("cuda")
    mask_t = torch.from_numpy(net.mask).to(device=device, dtype=torch.float32)
    input_projection_t = torch.from_numpy(net.input_projection).to(device=device, dtype=torch.float32)
    output_projection_t = torch.from_numpy(net.output_projection).to(device=device, dtype=torch.float32)
    theta_t = torch.from_numpy(net.theta).to(device=device, dtype=torch.float32)
    decay_t = torch.from_numpy(net.decay).to(device=device, dtype=torch.float32)

    for _ in range(WARMUP):
        out = torch_forward_batch(
            mask_t,
            input_projection_t,
            output_projection_t,
            theta_t,
            decay_t,
            TICKS,
        )
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        out = torch_forward_batch(
            mask_t,
            input_projection_t,
            output_projection_t,
            theta_t,
            decay_t,
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
        dense_out, dense_dt = bench_cpu_impl(dense_net, CPU_ITERS)
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
