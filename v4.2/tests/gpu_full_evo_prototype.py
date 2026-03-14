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
class MutationDelta:
    prev_mood_x: float
    prev_mood_z: float
    prev_leak: float
    changes: list[tuple[int, int, int]]


CONFIGS = {
    "V64_N192": BenchConfig("V64_N192", 64, 192, 0.06),
    "V64_dense": BenchConfig("V64_dense", 64, 192, 0.15),
    "V128_N384": BenchConfig("V128_N384", 128, 384, 0.06),
    "V128_dense": BenchConfig("V128_dense", 128, 384, 0.15),
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
    ap.add_argument("--compare-cpu", action="store_true")
    ap.add_argument("--determinism-check", action="store_true")
    return ap.parse_args()


def make_cpu_reference(cfg: BenchConfig, seed: int):
    np.random.seed(seed)
    import random

    random.seed(seed)
    net = SelfWiringGraph(cfg.neurons, cfg.vocab, density=cfg.density, threshold=cfg.threshold)
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


def rand_uniform(gen: torch.Generator, device: torch.device) -> float:
    return float(torch.rand((), generator=gen, device=device).item())


def randn_scaled(gen: torch.Generator, device: torch.device, sigma: float) -> float:
    return float((torch.randn((), generator=gen, device=device) * sigma).item())


def record_mask_change(mask: torch.Tensor, changes: list[tuple[int, int, int]], row: int, col: int, new_val: int):
    old_val = int(mask[row, col].item())
    if old_val == new_val:
        return
    changes.append((row, col, old_val))
    mask[row, col] = new_val


def mutate_gpu(
    mask: torch.Tensor,
    mood_x: torch.Tensor,
    mood_z: torch.Tensor,
    leak: torch.Tensor,
    gen: torch.Generator,
    diag_mask: torch.Tensor,
):
    device = mask.device
    delta = MutationDelta(
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
        if mx < 0.33:
            if rand_uniform(gen, device) < 0.7:
                add_connection_gpu(mask, gen, diag_mask, delta.changes)
            else:
                flip_connection_gpu(mask, gen, delta.changes)
        elif mx < 0.66:
            r = rand_uniform(gen, device)
            if r < 0.6:
                rewire_connection_gpu(mask, gen, delta.changes)
            elif r < 0.8:
                flip_connection_gpu(mask, gen, delta.changes)
            else:
                add_connection_gpu(mask, gen, diag_mask, delta.changes)
        else:
            flip_connection_gpu(mask, gen, delta.changes)
    return delta


def add_connection_gpu(
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
    record_mask_change(mask, changes, int(rc[0].item()), int(rc[1].item()), sign)


def flip_connection_gpu(mask: torch.Tensor, gen: torch.Generator, changes: list[tuple[int, int, int]]):
    alive = torch.nonzero(mask != 0, as_tuple=False)
    if alive.numel() == 0:
        return
    idx = int(torch.randint(alive.shape[0], (1,), generator=gen, device=mask.device).item())
    rc = alive[idx]
    row = int(rc[0].item())
    col = int(rc[1].item())
    record_mask_change(mask, changes, row, col, -int(mask[row, col].item()))


def rewire_connection_gpu(mask: torch.Tensor, gen: torch.Generator, changes: list[tuple[int, int, int]]):
    alive = torch.nonzero(mask != 0, as_tuple=False)
    if alive.numel() == 0:
        return
    idx = int(torch.randint(alive.shape[0], (1,), generator=gen, device=mask.device).item())
    rc = alive[idx]
    src = int(rc[0].item())
    dst = int(rc[1].item())
    old = int(mask[src, dst].item())
    record_mask_change(mask, changes, src, dst, 0)
    n = mask.shape[0]
    new_dst = int(torch.randint(n, (1,), generator=gen, device=mask.device).item())
    while new_dst == src:
        new_dst = int(torch.randint(n, (1,), generator=gen, device=mask.device).item())
    record_mask_change(mask, changes, src, new_dst, old)


def rollback_gpu(
    mask: torch.Tensor,
    mood_x: torch.Tensor,
    mood_z: torch.Tensor,
    leak: torch.Tensor,
    delta: MutationDelta,
):
    for row, col, old_val in reversed(delta.changes):
        mask[row, col] = old_val
    mood_x.fill_(delta.prev_mood_x)
    mood_z.fill_(delta.prev_mood_z)
    leak.fill_(delta.prev_leak)


def gpu_train(cfg: BenchConfig, attempts: int, seed: int, verbose_every: int = 200):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    device = torch.device("cuda")
    torch.manual_seed(seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    mask, mood_x, mood_z, leak, targets, out_start = gpu_init_from_cpu(cfg, seed, device)
    diag_mask = ~torch.eye(cfg.neurons, dtype=torch.bool, device=device)
    buffers = make_eval_buffers(cfg, device)

    _, score, acc = gpu_eval(mask, leak, targets, out_start, buffers=buffers)
    best_score = score.clone()
    best_acc = acc.clone()
    kept = 0

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for att in range(attempts):
        delta = mutate_gpu(mask, mood_x, mood_z, leak, gen, diag_mask)
        _, new_score, new_acc = gpu_eval(mask, leak, targets, out_start, buffers=buffers)

        if bool((new_score > score).item()):
            score = new_score
            kept += 1
            if bool((new_score > best_score).item()):
                best_score = new_score
                best_acc = new_acc
        else:
            rollback_gpu(mask, mood_x, mood_z, leak, delta)

        if verbose_every and (att + 1) % verbose_every == 0:
            print(
                f"[GPU {att+1:5d}] acc={float(best_acc.item())*100:5.1f}% "
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


def determinism_check(cfg: BenchConfig, attempts: int, seed: int):
    a = gpu_train(cfg, attempts, seed, verbose_every=0)
    b = gpu_train(cfg, attempts, seed, verbose_every=0)
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
        return determinism_check(cfg, args.attempts, args.seed)

    gpu_res = gpu_train(cfg, args.attempts, args.seed, verbose_every=args.verbose_every)
    print(
        f"GPU_FULL {cfg.name} attempts={args.attempts} "
        f"acc={gpu_res['best_acc']*100:.1f}% score={gpu_res['best_score']:.4f} "
        f"kept={gpu_res['kept']} aps={gpu_res['attempts_per_sec']:.1f} "
        f"leak={gpu_res['final_leak']:.4f} mask={gpu_res['mask_hash']}"
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
