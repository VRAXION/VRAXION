"""Full-GPU evolutionary loop prototype for the current v4.2 graph core.

Goal:
  - keep mask, mood, leak, rollout, score, and accept/revert state on CUDA
  - avoid per-attempt host/device transfers for the hot path
  - benchmark end-to-end attempts/sec for a single-candidate evolutionary loop

This is a prototype, not yet a drop-in replacement for model/graph.py.
It mirrors the current local v4.2 semantics:
  - ternary mask in {-1, 0, +1}
  - gain = 2.0, charge_rate = 0.3, self_conn = 0.1, threshold = 0.5
  - 2D mood + learnable leak
  - 4 expert zones on mood_x: scout/rewirer/refiner/pruner
  - batch eval over all V one-hot inputs
"""

from __future__ import annotations

import argparse
import hashlib
import time
from dataclasses import dataclass

import numpy as np
import torch

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.graph import SelfWiringGraph


@dataclass
class BenchConfig:
    name: str
    vocab: int
    neurons: int
    density: float
    threshold: float = 0.5


@dataclass
class GpuEvalBuffers:
    eye: torch.Tensor
    charges: torch.Tensor
    acts: torch.Tensor
    weff: torch.Tensor
    row_idx: torch.Tensor


@dataclass
class ReferenceMutationDelta:
    prev_mood_x: float
    prev_mood_z: float
    prev_leak: float
    changes: list[tuple[int, int, int]]


@dataclass
class TensorMutationDelta:
    prev_scalars: torch.Tensor
    rows: torch.Tensor
    cols: torch.Tensor
    old_vals: torch.Tensor
    flat_idx: torch.Tensor
    count: int = 0


CONFIGS = {
    "V64_N192": BenchConfig("V64_N192", 64, 192, 0.06),
    "V64_dense": BenchConfig("V64_dense", 64, 192, 0.15),
    "V128_N384": BenchConfig("V128_N384", 128, 384, 0.06),
    "V128_dense": BenchConfig("V128_dense", 128, 384, 0.15),
    "V256_N768": BenchConfig("V256_N768", 256, 768, 0.06),
}

GAIN = 2.0
CHARGE_RATE = 0.3
SELF_CONN = 0.1
CLIP_FACTOR = 2.0
TICKS = 8


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="V128_N384", choices=sorted(CONFIGS))
    ap.add_argument("--attempts", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verbose-every", type=int, default=200)
    ap.add_argument("--engine", default="reference", choices=["tensorized", "reference"])
    ap.add_argument("--compare-cpu", action="store_true")
    ap.add_argument("--compare-reference", action="store_true")
    ap.add_argument("--compile-eval", action="store_true")
    ap.add_argument("--determinism-check", action="store_true")
    return ap.parse_args()


def make_cpu_reference(cfg: BenchConfig, seed: int):
    np.random.seed(seed)
    import random

    random.seed(seed)
    net = SelfWiringGraph(cfg.neurons, cfg.vocab, density=cfg.density)
    perm = np.random.permutation(cfg.vocab).astype(np.int64)
    return net, perm


def gpu_init_from_cpu(cfg: BenchConfig, seed: int, device: torch.device):
    cpu_net, targets = make_cpu_reference(cfg, seed)
    mask = torch.from_numpy(cpu_net.mask.copy()).to(device=device, dtype=torch.int8)
    mood_x = torch.tensor(cpu_net.mood_x, device=device, dtype=torch.float32)
    mood_z = torch.tensor(cpu_net.mood_z, device=device, dtype=torch.float32)
    leak = torch.tensor(cpu_net.leak, device=device, dtype=torch.float32)
    targets_t = torch.from_numpy(targets).to(device=device, dtype=torch.long)
    out_start = cpu_net.out_start
    return mask, mood_x, mood_z, leak, targets_t, out_start


def make_eval_buffers(cfg: BenchConfig, device: torch.device) -> GpuEvalBuffers:
    return GpuEvalBuffers(
        eye=torch.eye(cfg.vocab, dtype=torch.float32, device=device),
        charges=torch.empty((cfg.vocab, cfg.neurons), dtype=torch.float32, device=device),
        acts=torch.empty((cfg.vocab, cfg.neurons), dtype=torch.float32, device=device),
        weff=torch.empty((cfg.neurons, cfg.neurons), dtype=torch.float32, device=device),
        row_idx=torch.arange(cfg.vocab, device=device, dtype=torch.long),
    )


def gpu_eval(
    mask: torch.Tensor,
    leak: torch.Tensor,
    targets: torch.Tensor,
    out_start: int,
    eye: torch.Tensor | None = None,
    buffers: GpuEvalBuffers | None = None,
):
    device = mask.device
    if buffers is None:
        if eye is None:
            raise ValueError("gpu_eval requires either eye or buffers")
        vocab = eye.shape[0]
        neurons = mask.shape[0]
        charges = torch.zeros((vocab, neurons), dtype=torch.float32, device=device)
        acts = torch.zeros((vocab, neurons), dtype=torch.float32, device=device)
        weff = mask.to(torch.float32) * GAIN
        row_idx = torch.arange(vocab, device=device, dtype=torch.long)
    else:
        eye = buffers.eye
        vocab = eye.shape[0]
        charges = buffers.charges
        acts = buffers.acts
        weff = buffers.weff
        row_idx = buffers.row_idx
        charges.zero_()
        acts.zero_()
        weff.copy_(mask)
        weff.mul_(GAIN)

    clip_bound = 0.5 * CLIP_FACTOR

    for t in range(TICKS):
        if t == 0:
            acts[:, :vocab] = eye
        raw = acts @ weff + acts * SELF_CONN
        raw = torch.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        charges = charges + raw * CHARGE_RATE
        charges = charges * leak
        acts = torch.clamp(charges - 0.5, min=0.0)
        charges = torch.clamp(charges, -clip_bound, clip_bound)

    logits = charges[:, out_start : out_start + vocab]
    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)
    acc = (preds == targets).to(torch.float32).mean()
    tp = probs[row_idx, targets].mean()
    score = 0.5 * acc + 0.5 * tp
    return logits, score, acc


def make_eval_runner(
    cfg: BenchConfig,
    targets: torch.Tensor,
    out_start: int,
    device: torch.device,
    compile_eval: bool = False,
):
    buffers = make_eval_buffers(cfg, device)

    def eval_runner(mask: torch.Tensor, leak: torch.Tensor):
        return gpu_eval(mask, leak, targets, out_start, buffers=buffers)

    if compile_eval:
        return torch.compile(eval_runner, mode="reduce-overhead", fullgraph=False)
    return eval_runner


def rand_uniform(gen: torch.Generator, device: torch.device) -> float:
    return float(torch.rand((), generator=gen, device=device).item())


def randn_scaled(gen: torch.Generator, device: torch.device, sigma: float) -> float:
    return float((torch.randn((), generator=gen, device=device) * sigma).item())


def record_mask_change_reference(
    mask: torch.Tensor, changes: list[tuple[int, int, int]], row: int, col: int, new_val: int
):
    old_val = int(mask[row, col].item())
    if old_val == new_val:
        return
    changes.append((row, col, old_val))
    mask[row, col] = new_val


def mutate_gpu_reference(
    mask: torch.Tensor,
    mood_x: torch.Tensor,
    mood_z: torch.Tensor,
    leak: torch.Tensor,
    gen: torch.Generator,
    diag_mask: torch.Tensor,
):
    device = mask.device
    delta = ReferenceMutationDelta(
        prev_mood_x=float(mood_x.item()),
        prev_mood_z=float(mood_z.item()),
        prev_leak=float(leak.item()),
        changes=[],
    )

    if rand_uniform(gen, device) < 0.2:
        mood_x.add_(randn_scaled(gen, device, 0.15)).clamp_(0.0, 1.0)
    if rand_uniform(gen, device) < 0.2:
        mood_z.add_(randn_scaled(gen, device, 0.15)).clamp_(0.0, 1.0)
    if rand_uniform(gen, device) < 0.2:
        leak.add_(randn_scaled(gen, device, 0.03)).clamp_(0.5, 0.99)

    n_changes = max(1, int(1 + float(mood_z.item()) * 14))
    for _ in range(n_changes):
        mx = float(mood_x.item())
        if mx < 0.25:
            if rand_uniform(gen, device) < 0.7:
                add_connection_gpu_reference(mask, gen, diag_mask, delta.changes)
            else:
                flip_connection_gpu_reference(mask, gen, delta.changes)
        elif mx < 0.50:
            r = rand_uniform(gen, device)
            if r < 0.6:
                rewire_connection_gpu_reference(mask, gen, delta.changes)
            elif r < 0.8:
                flip_connection_gpu_reference(mask, gen, delta.changes)
            else:
                add_connection_gpu_reference(mask, gen, diag_mask, delta.changes)
        elif mx < 0.75:
            if rand_uniform(gen, device) < 0.8:
                flip_connection_gpu_reference(mask, gen, delta.changes)
            else:
                rewire_connection_gpu_reference(mask, gen, delta.changes)
        else:
            r = rand_uniform(gen, device)
            if r < 0.7:
                remove_connection_gpu_reference(mask, gen, delta.changes)
            elif r < 0.9:
                flip_connection_gpu_reference(mask, gen, delta.changes)
            else:
                rewire_connection_gpu_reference(mask, gen, delta.changes)
    return delta


def add_connection_gpu_reference(
    mask: torch.Tensor,
    gen: torch.Generator,
    diag_mask: torch.Tensor,
    changes: list[tuple[int, int, int]],
):
    dead = torch.nonzero((mask == 0) & diag_mask, as_tuple=False)
    if dead.numel() == 0:
        return
    idx = int(torch.randint(dead.shape[0], (1,), generator=gen, device=mask.device).item())
    rc = dead[idx]
    sign = 1 if rand_uniform(gen, mask.device) > 0.5 else -1
    record_mask_change_reference(mask, changes, int(rc[0].item()), int(rc[1].item()), sign)


def flip_connection_gpu_reference(mask: torch.Tensor, gen: torch.Generator, changes: list[tuple[int, int, int]]):
    alive = torch.nonzero(mask != 0, as_tuple=False)
    if alive.numel() == 0:
        return
    idx = int(torch.randint(alive.shape[0], (1,), generator=gen, device=mask.device).item())
    rc = alive[idx]
    row = int(rc[0].item())
    col = int(rc[1].item())
    record_mask_change_reference(mask, changes, row, col, -int(mask[row, col].item()))


def remove_connection_gpu_reference(
    mask: torch.Tensor,
    gen: torch.Generator,
    changes: list[tuple[int, int, int]],
):
    alive = torch.nonzero(mask != 0, as_tuple=False)
    if alive.numel() == 0:
        return
    idx = int(torch.randint(alive.shape[0], (1,), generator=gen, device=mask.device).item())
    rc = alive[idx]
    row = int(rc[0].item())
    col = int(rc[1].item())
    record_mask_change_reference(mask, changes, row, col, 0)


def rewire_connection_gpu_reference(mask: torch.Tensor, gen: torch.Generator, changes: list[tuple[int, int, int]]):
    alive = torch.nonzero(mask != 0, as_tuple=False)
    if alive.numel() == 0:
        return
    idx = int(torch.randint(alive.shape[0], (1,), generator=gen, device=mask.device).item())
    rc = alive[idx]
    src = int(rc[0].item())
    dst = int(rc[1].item())
    old = int(mask[src, dst].item())
    record_mask_change_reference(mask, changes, src, dst, 0)
    n = mask.shape[0]
    new_dst = int(torch.randint(n, (1,), generator=gen, device=mask.device).item())
    while new_dst == src:
        new_dst = int(torch.randint(n, (1,), generator=gen, device=mask.device).item())
    record_mask_change_reference(mask, changes, src, new_dst, old)


def rollback_gpu_reference(
    mask: torch.Tensor,
    mood_x: torch.Tensor,
    mood_z: torch.Tensor,
    leak: torch.Tensor,
    delta: ReferenceMutationDelta,
):
    for row, col, old_val in reversed(delta.changes):
        mask[row, col] = old_val
    mood_x.fill_(delta.prev_mood_x)
    mood_z.fill_(delta.prev_mood_z)
    leak.fill_(delta.prev_leak)


def make_mutation_delta(device: torch.device, capacity: int = 32) -> TensorMutationDelta:
    return TensorMutationDelta(
        prev_scalars=torch.empty(3, dtype=torch.float32, device=device),
        rows=torch.empty(capacity, dtype=torch.long, device=device),
        cols=torch.empty(capacity, dtype=torch.long, device=device),
        old_vals=torch.empty(capacity, dtype=torch.int8, device=device),
        flat_idx=torch.full((capacity,), -1, dtype=torch.long, device=device),
        count=0,
    )


def reset_mutation_delta(delta: TensorMutationDelta, mood_x: torch.Tensor, mood_z: torch.Tensor, leak: torch.Tensor):
    delta.prev_scalars[0].copy_(mood_x)
    delta.prev_scalars[1].copy_(mood_z)
    delta.prev_scalars[2].copy_(leak)
    delta.count = 0
    delta.flat_idx.fill_(-1)


def record_mask_change_tensor(
    mask: torch.Tensor,
    delta: TensorMutationDelta,
    row: int,
    col: int,
    new_val: int,
):
    old_val = int(mask[row, col].item())
    if old_val == new_val:
        return

    flat = row * mask.shape[1] + col
    if delta.count:
        exists = bool((delta.flat_idx[: delta.count] == flat).any().item())
    else:
        exists = False
    if not exists:
        idx = delta.count
        delta.rows[idx] = row
        delta.cols[idx] = col
        delta.old_vals[idx] = old_val
        delta.flat_idx[idx] = flat
        delta.count += 1
    mask[row, col] = new_val


def add_connection_gpu_tensorized(
    mask: torch.Tensor,
    gen: torch.Generator,
    diag_mask: torch.Tensor,
    delta: TensorMutationDelta,
):
    dead = torch.nonzero((mask == 0) & diag_mask, as_tuple=False)
    if dead.numel() == 0:
        return
    idx = int(torch.randint(dead.shape[0], (1,), generator=gen, device=mask.device).item())
    rc = dead[idx]
    sign = 1 if rand_uniform(gen, mask.device) > 0.5 else -1
    record_mask_change_tensor(mask, delta, int(rc[0].item()), int(rc[1].item()), sign)


def flip_connection_gpu_tensorized(
    mask: torch.Tensor,
    gen: torch.Generator,
    delta: TensorMutationDelta,
):
    alive = torch.nonzero(mask != 0, as_tuple=False)
    if alive.numel() == 0:
        return
    idx = int(torch.randint(alive.shape[0], (1,), generator=gen, device=mask.device).item())
    rc = alive[idx]
    row = int(rc[0].item())
    col = int(rc[1].item())
    record_mask_change_tensor(mask, delta, row, col, -int(mask[row, col].item()))


def remove_connection_gpu_tensorized(
    mask: torch.Tensor,
    gen: torch.Generator,
    delta: TensorMutationDelta,
):
    alive = torch.nonzero(mask != 0, as_tuple=False)
    if alive.numel() == 0:
        return
    idx = int(torch.randint(alive.shape[0], (1,), generator=gen, device=mask.device).item())
    rc = alive[idx]
    row = int(rc[0].item())
    col = int(rc[1].item())
    record_mask_change_tensor(mask, delta, row, col, 0)


def rewire_connection_gpu_tensorized(
    mask: torch.Tensor,
    gen: torch.Generator,
    delta: TensorMutationDelta,
):
    alive = torch.nonzero(mask != 0, as_tuple=False)
    if alive.numel() == 0:
        return
    idx = int(torch.randint(alive.shape[0], (1,), generator=gen, device=mask.device).item())
    rc = alive[idx]
    src = int(rc[0].item())
    dst = int(rc[1].item())
    old = int(mask[src, dst].item())
    record_mask_change_tensor(mask, delta, src, dst, 0)
    n = mask.shape[0]
    new_dst = int(torch.randint(n, (1,), generator=gen, device=mask.device).item())
    while new_dst == src:
        new_dst = int(torch.randint(n, (1,), generator=gen, device=mask.device).item())
    record_mask_change_tensor(mask, delta, src, new_dst, old)


def mutate_gpu_tensorized(
    mask: torch.Tensor,
    mood_x: torch.Tensor,
    mood_z: torch.Tensor,
    leak: torch.Tensor,
    gen: torch.Generator,
    diag_mask: torch.Tensor,
    delta: TensorMutationDelta,
):
    device = mask.device
    reset_mutation_delta(delta, mood_x, mood_z, leak)

    gate_mx = torch.rand((), generator=gen, device=device)
    if bool((gate_mx < 0.2).item()):
        mood_x.add_(torch.randn((), generator=gen, device=device) * 0.15).clamp_(0.0, 1.0)
    gate_mz = torch.rand((), generator=gen, device=device)
    if bool((gate_mz < 0.2).item()):
        mood_z.add_(torch.randn((), generator=gen, device=device) * 0.15).clamp_(0.0, 1.0)
    gate_leak = torch.rand((), generator=gen, device=device)
    if bool((gate_leak < 0.2).item()):
        leak.add_(torch.randn((), generator=gen, device=device) * 0.03).clamp_(0.5, 0.99)

    mx_band = (
        int((mood_x >= 0.25).to(torch.int64).item())
        + int((mood_x >= 0.50).to(torch.int64).item())
        + int((mood_x >= 0.75).to(torch.int64).item())
    )
    n_changes = int(torch.clamp((1 + mood_z * 14).to(torch.int64), min=1, max=15).item())

    for _ in range(n_changes):
        if mx_band == 0:
            op_u = torch.rand((), generator=gen, device=device)
            op_code = 0 if bool((op_u < 0.7).item()) else 1
        elif mx_band == 1:
            op_u = torch.rand((), generator=gen, device=device)
            op_code = 2 if bool((op_u < 0.6).item()) else (1 if bool((op_u < 0.8).item()) else 0)
        elif mx_band == 2:
            op_u = torch.rand((), generator=gen, device=device)
            op_code = 1 if bool((op_u < 0.8).item()) else 2
        else:
            op_u = torch.rand((), generator=gen, device=device)
            op_code = 3 if bool((op_u < 0.7).item()) else (1 if bool((op_u < 0.9).item()) else 2)

        if op_code == 0:
            add_connection_gpu_tensorized(mask, gen, diag_mask, delta)
        elif op_code == 1:
            flip_connection_gpu_tensorized(mask, gen, delta)
        elif op_code == 2:
            rewire_connection_gpu_tensorized(mask, gen, delta)
        else:
            remove_connection_gpu_tensorized(mask, gen, delta)


def finalize_attempt_tensorized(
    mask: torch.Tensor,
    mood_x: torch.Tensor,
    mood_z: torch.Tensor,
    leak: torch.Tensor,
    score: torch.Tensor,
    best_score: torch.Tensor,
    best_acc: torch.Tensor,
    kept: torch.Tensor,
    new_score: torch.Tensor,
    new_acc: torch.Tensor,
    delta: TensorMutationDelta,
):
    accept = new_score > score
    improve = accept & (new_score > best_score)

    if delta.count:
        rows = delta.rows[: delta.count]
        cols = delta.cols[: delta.count]
        old_vals = delta.old_vals[: delta.count]
        curr_vals = mask[rows, cols]
        accept_vals = accept.to(torch.int8).expand_as(old_vals)
        mask[rows, cols] = torch.where(accept_vals.bool(), curr_vals, old_vals)

    current_scalars = torch.stack((mood_x, mood_z, leak))
    next_scalars = torch.where(accept.expand_as(current_scalars), current_scalars, delta.prev_scalars)
    mood_x.copy_(next_scalars[0])
    mood_z.copy_(next_scalars[1])
    leak.copy_(next_scalars[2])

    score.copy_(torch.where(accept, new_score, score))
    best_score.copy_(torch.where(improve, new_score, best_score))
    best_acc.copy_(torch.where(improve, new_acc, best_acc))
    kept.add_(accept.to(torch.int64))


def gpu_train_reference(
    cfg: BenchConfig,
    attempts: int,
    seed: int,
    verbose_every: int = 200,
    compile_eval: bool = False,
):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    device = torch.device("cuda")
    torch.manual_seed(seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    mask, mood_x, mood_z, leak, targets, out_start = gpu_init_from_cpu(cfg, seed, device)
    diag_mask = ~torch.eye(cfg.neurons, dtype=torch.bool, device=device)
    eval_runner = make_eval_runner(cfg, targets, out_start, device, compile_eval=compile_eval)

    _, score, acc = eval_runner(mask, leak)
    best_score = score.clone()
    best_acc = acc.clone()
    kept = 0

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for att in range(attempts):
        delta = mutate_gpu_reference(mask, mood_x, mood_z, leak, gen, diag_mask)
        _, new_score, new_acc = eval_runner(mask, leak)

        if bool((new_score > score).item()):
            score = new_score
            kept += 1
            if bool((new_score > best_score).item()):
                best_score = new_score
                best_acc = new_acc
        else:
            rollback_gpu_reference(mask, mood_x, mood_z, leak, delta)

        if verbose_every and (att + 1) % verbose_every == 0:
            print(
                f"[GPU-REF {att+1:5d}] acc={float(best_acc.item())*100:5.1f}% "
                f"score={float(best_score.item()):.4f} kept={kept:4d} leak={float(leak.item()):.4f}"
            )

    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    final_mask = mask.detach().cpu().numpy().tobytes()
    mask_hash = hashlib.sha256(final_mask).hexdigest()[:16]
    return {
        "best_acc": float(best_acc.item()),
        "best_score": float(best_score.item()),
        "kept": kept,
        "seconds": dt,
        "attempts_per_sec": attempts / dt if dt > 0 else float("inf"),
        "final_leak": float(leak.item()),
        "mask_hash": mask_hash,
    }


def gpu_train(
    cfg: BenchConfig,
    attempts: int,
    seed: int,
    verbose_every: int = 200,
    engine: str = "reference",
    compile_eval: bool = False,
):
    if engine == "reference":
        return gpu_train_reference(cfg, attempts, seed, verbose_every=verbose_every, compile_eval=compile_eval)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    device = torch.device("cuda")
    torch.manual_seed(seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    mask, mood_x, mood_z, leak, targets, out_start = gpu_init_from_cpu(cfg, seed, device)
    diag_mask = ~torch.eye(cfg.neurons, dtype=torch.bool, device=device)
    eval_runner = make_eval_runner(cfg, targets, out_start, device, compile_eval=compile_eval)
    delta = make_mutation_delta(device)

    _, score, acc = eval_runner(mask, leak)
    score = score.clone()
    best_score = score.clone()
    best_acc = acc.clone()
    kept = torch.zeros((), dtype=torch.int64, device=device)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for att in range(attempts):
        mutate_gpu_tensorized(mask, mood_x, mood_z, leak, gen, diag_mask, delta)
        _, new_score, new_acc = eval_runner(mask, leak)
        finalize_attempt_tensorized(
            mask,
            mood_x,
            mood_z,
            leak,
            score,
            best_score,
            best_acc,
            kept,
            new_score,
            new_acc,
            delta,
        )

        if verbose_every and (att + 1) % verbose_every == 0:
            print(
                f"[GPU {att+1:5d}] acc={float(best_acc.item())*100:5.1f}% "
                f"score={float(best_score.item()):.4f} kept={int(kept.item()):4d} leak={float(leak.item()):.4f}"
            )

    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    final_mask = mask.detach().cpu().numpy().tobytes()
    mask_hash = hashlib.sha256(final_mask).hexdigest()[:16]
    return {
        "best_acc": float(best_acc.item()),
        "best_score": float(best_score.item()),
        "kept": int(kept.item()),
        "seconds": dt,
        "attempts_per_sec": attempts / dt if dt > 0 else float("inf"),
        "final_leak": float(leak.item()),
        "mask_hash": mask_hash,
    }


def cpu_train(cfg: BenchConfig, attempts: int, seed: int):
    net, targets = make_cpu_reference(cfg, seed)

    def evaluate():
        logits = net.forward_batch(TICKS)
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
        preds = np.argmax(probs, axis=1)
        acc = (preds == targets).mean()
        tp = probs[np.arange(cfg.vocab), targets].mean()
        return 0.5 * acc + 0.5 * tp, acc

    score, acc = evaluate()
    best_score = score
    best_acc = acc
    kept = 0

    t0 = time.perf_counter()
    for _ in range(attempts):
        state = net.save_state()
        net.mutate_with_mood()
        new_score, new_acc = evaluate()
        if new_score > score:
            score = new_score
            kept += 1
            if new_score > best_score:
                best_score = new_score
                best_acc = new_acc
        else:
            net.restore_state(state)
    dt = time.perf_counter() - t0
    mask_hash = hashlib.sha256(net.mask.tobytes()).hexdigest()[:16]
    return {
        "best_acc": float(best_acc),
        "best_score": float(best_score),
        "kept": kept,
        "seconds": dt,
        "attempts_per_sec": attempts / dt if dt > 0 else float("inf"),
        "final_leak": float(net.leak),
        "mask_hash": mask_hash,
    }


def determinism_check(
    cfg: BenchConfig,
    attempts: int,
    seed: int,
    engine: str = "reference",
    compile_eval: bool = False,
):
    a = gpu_train(cfg, attempts, seed, verbose_every=0, engine=engine, compile_eval=compile_eval)
    b = gpu_train(cfg, attempts, seed, verbose_every=0, engine=engine, compile_eval=compile_eval)
    ok = (
        a["mask_hash"] == b["mask_hash"]
        and abs(a["best_acc"] - b["best_acc"]) == 0.0
        and abs(a["best_score"] - b["best_score"]) == 0.0
        and abs(a["final_leak"] - b["final_leak"]) == 0.0
    )
    print({"deterministic": ok, "run_a": a, "run_b": b})
    return 0 if ok else 1


def main() -> int:
    args = parse_args()
    cfg = CONFIGS[args.config]

    if args.determinism_check:
        return determinism_check(cfg, args.attempts, args.seed, engine=args.engine, compile_eval=args.compile_eval)

    gpu_res = gpu_train(
        cfg,
        args.attempts,
        args.seed,
        verbose_every=args.verbose_every,
        engine=args.engine,
        compile_eval=args.compile_eval,
    )
    print(
        f"GPU_FULL {cfg.name} engine={args.engine} attempts={args.attempts} "
        f"acc={gpu_res['best_acc']*100:.1f}% score={gpu_res['best_score']:.4f} "
        f"kept={gpu_res['kept']} aps={gpu_res['attempts_per_sec']:.1f} "
        f"leak={gpu_res['final_leak']:.4f} mask={gpu_res['mask_hash']}"
    )

    if args.compare_reference and args.engine != "reference":
        ref_res = gpu_train_reference(
            cfg,
            args.attempts,
            args.seed,
            verbose_every=0,
            compile_eval=args.compile_eval,
        )
        print(
            f"GPU_REF  {cfg.name} attempts={args.attempts} "
            f"acc={ref_res['best_acc']*100:.1f}% score={ref_res['best_score']:.4f} "
            f"kept={ref_res['kept']} aps={ref_res['attempts_per_sec']:.1f} "
            f"leak={ref_res['final_leak']:.4f} mask={ref_res['mask_hash']}"
        )

    if args.compare_cpu:
        cpu_res = cpu_train(cfg, args.attempts, args.seed)
        print(
            f"CPU_BASE {cfg.name} attempts={args.attempts} "
            f"acc={cpu_res['best_acc']*100:.1f}% score={cpu_res['best_score']:.4f} "
            f"kept={cpu_res['kept']} aps={cpu_res['attempts_per_sec']:.1f} "
            f"leak={cpu_res['final_leak']:.4f} mask={cpu_res['mask_hash']}"
        )
        if cpu_res["attempts_per_sec"] > 0:
            print(
                f"SPEEDUP gpu_vs_cpu={gpu_res['attempts_per_sec'] / cpu_res['attempts_per_sec']:.2f}x"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
