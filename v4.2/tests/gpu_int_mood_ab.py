"""GPU A/B: old float mood controller vs new int mood controller.

Purpose:
  - keep the same GPU evaluation kernel
  - compare only the mutation-controller representation
  - report accuracy and throughput on GPU-relevant configs
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.graph import SelfWiringGraph


GAIN = 2.0
CHARGE_RATE = 0.3
SELF_CONN = 0.05
THRESHOLD = 0.5
CLIP_BOUND = 1.0
TICKS = 8

SEEDS = [42, 77, 123]
CONFIGS = {
    "V64_N192": (64, 192, 0.06),
    "V128_N384": (128, 384, 0.06),
    "V256_N768": (256, 768, 0.06),
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="V128_N384", help="Comma-separated config names")
    ap.add_argument("--attempts", type=int, default=32000)
    ap.add_argument("--seeds", default="42,77,123")
    return ap.parse_args()


def parse_csv_ints(s: str):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def make_cpu_reference(vocab: int, neurons: int, density: float, seed: int):
    np.random.seed(seed)
    random.seed(seed)
    net = SelfWiringGraph(neurons, vocab, density=density)
    perm = np.random.permutation(vocab).astype(np.int64)
    return net, perm


def gpu_init(vocab: int, neurons: int, density: float, seed: int, device: torch.device):
    cpu_net, targets = make_cpu_reference(vocab, neurons, density, seed)
    mask = torch.from_numpy(cpu_net.mask.copy()).to(device=device, dtype=torch.int8)
    leak = torch.tensor(cpu_net.leak, device=device, dtype=torch.float32)
    targets_t = torch.from_numpy(targets).to(device=device, dtype=torch.long)
    out_start = cpu_net.out_start
    return mask, leak, targets_t, out_start


def gpu_eval(mask: torch.Tensor, leak: torch.Tensor, targets: torch.Tensor, out_start: int, eye: torch.Tensor):
    vocab, neurons = eye.shape[0], mask.shape[0]
    charges = torch.zeros((vocab, neurons), dtype=torch.float32, device=mask.device)
    acts = torch.zeros((vocab, neurons), dtype=torch.float32, device=mask.device)
    weff = mask.to(torch.float32) * GAIN

    for t in range(TICKS):
        if t == 0:
            acts[:, :vocab] = eye
        raw = acts @ weff + acts * SELF_CONN
        raw = torch.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        charges += raw * CHARGE_RATE
        charges *= leak
        acts = torch.clamp(charges - THRESHOLD, min=0.0)
        charges = torch.clamp(charges, -CLIP_BOUND, CLIP_BOUND)

    logits = charges[:, out_start : out_start + vocab]
    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)
    acc = (preds == targets).to(torch.float32).mean()
    tp = probs[torch.arange(vocab, device=mask.device), targets].mean()
    score = 0.5 * acc + 0.5 * tp
    return score, acc


def add_connection(mask, gen, diag_mask, changes):
    dead = torch.nonzero((mask == 0) & diag_mask, as_tuple=False)
    if dead.numel() == 0:
        return
    idx = int(torch.randint(dead.shape[0], (1,), generator=gen, device=mask.device).item())
    rc = dead[idx]
    row = int(rc[0].item()); col = int(rc[1].item())
    old = int(mask[row, col].item())
    new = 1 if float(torch.rand((), generator=gen, device=mask.device).item()) > 0.5 else -1
    if old != new:
        changes.append((row, col, old))
        mask[row, col] = new


def flip_connection(mask, gen, changes):
    alive = torch.nonzero(mask != 0, as_tuple=False)
    if alive.numel() == 0:
        return
    idx = int(torch.randint(alive.shape[0], (1,), generator=gen, device=mask.device).item())
    rc = alive[idx]
    row = int(rc[0].item()); col = int(rc[1].item())
    old = int(mask[row, col].item())
    changes.append((row, col, old))
    mask[row, col] = -old


def remove_connection(mask, gen, changes):
    alive = torch.nonzero(mask != 0, as_tuple=False)
    if alive.numel() == 0:
        return
    idx = int(torch.randint(alive.shape[0], (1,), generator=gen, device=mask.device).item())
    rc = alive[idx]
    row = int(rc[0].item()); col = int(rc[1].item())
    old = int(mask[row, col].item())
    changes.append((row, col, old))
    mask[row, col] = 0


def rewire_connection(mask, gen, changes):
    alive = torch.nonzero(mask != 0, as_tuple=False)
    if alive.numel() == 0:
        return
    idx = int(torch.randint(alive.shape[0], (1,), generator=gen, device=mask.device).item())
    rc = alive[idx]
    src = int(rc[0].item()); dst = int(rc[1].item())
    old = int(mask[src, dst].item())
    changes.append((src, dst, old))
    mask[src, dst] = 0
    n = mask.shape[0]
    new_dst = int(torch.randint(n, (1,), generator=gen, device=mask.device).item())
    while new_dst == src:
        new_dst = int(torch.randint(n, (1,), generator=gen, device=mask.device).item())
    old2 = int(mask[src, new_dst].item())
    if old2 != old:
        changes.append((src, new_dst, old2))
        mask[src, new_dst] = old


def rollback(mask, leak, controller, prev_state, changes):
    for row, col, old in reversed(changes):
        mask[row, col] = old
    leak.fill_(prev_state["leak"])
    if controller["kind"] == "float":
        controller["mood_x"].fill_(prev_state["mood_x"])
        controller["mood_z"].fill_(prev_state["mood_z"])
    else:
        controller["mood"] = prev_state["mood"]
        controller["intensity"] = prev_state["intensity"]


def mutate_float(mask, leak, controller, gen, diag_mask):
    changes = []
    prev = {
        "leak": float(leak.item()),
        "mood_x": float(controller["mood_x"].item()),
        "mood_z": float(controller["mood_z"].item()),
    }
    if float(torch.rand((), generator=gen, device=mask.device).item()) < 0.35:
        controller["mood_x"].add_(float((torch.randn((), generator=gen, device=mask.device) * 0.10).item())).clamp_(0.0, 1.0)
    if float(torch.rand((), generator=gen, device=mask.device).item()) < 0.35:
        controller["mood_z"].add_(float((torch.randn((), generator=gen, device=mask.device) * 0.10).item())).clamp_(0.0, 1.0)
    if float(torch.rand((), generator=gen, device=mask.device).item()) < 0.2:
        leak.add_(float((torch.randn((), generator=gen, device=mask.device) * 0.03).item())).clamp_(0.5, 0.99)

    n_changes = max(1, int(1 + float(controller["mood_z"].item()) * 14))
    mx = float(controller["mood_x"].item())
    for _ in range(n_changes):
        if mx < 0.25:
            if float(torch.rand((), generator=gen, device=mask.device).item()) < 0.7:
                add_connection(mask, gen, diag_mask, changes)
            else:
                flip_connection(mask, gen, changes)
        elif mx < 0.50:
            r = float(torch.rand((), generator=gen, device=mask.device).item())
            if r < 0.6:
                rewire_connection(mask, gen, changes)
            elif r < 0.8:
                flip_connection(mask, gen, changes)
            else:
                add_connection(mask, gen, diag_mask, changes)
        elif mx < 0.75:
            if float(torch.rand((), generator=gen, device=mask.device).item()) < 0.8:
                flip_connection(mask, gen, changes)
            else:
                rewire_connection(mask, gen, changes)
        else:
            r = float(torch.rand((), generator=gen, device=mask.device).item())
            if r < 0.7:
                remove_connection(mask, gen, changes)
            elif r < 0.9:
                flip_connection(mask, gen, changes)
            else:
                rewire_connection(mask, gen, changes)
    return prev, changes


def mutate_int(mask, leak, controller, gen, diag_mask):
    changes = []
    prev = {
        "leak": float(leak.item()),
        "mood": controller["mood"],
        "intensity": controller["intensity"],
    }
    if float(torch.rand((), generator=gen, device=mask.device).item()) < 0.35:
        controller["mood"] = max(0, min(3, controller["mood"] + random.choice([-1, 1])))
    if float(torch.rand((), generator=gen, device=mask.device).item()) < 0.35:
        controller["intensity"] = max(1, min(15, controller["intensity"] + random.choice([-1, 1])))
    if float(torch.rand((), generator=gen, device=mask.device).item()) < 0.2:
        leak.add_(float((torch.randn((), generator=gen, device=mask.device) * 0.03).item())).clamp_(0.5, 0.99)

    for _ in range(controller["intensity"]):
        if controller["mood"] == 0:
            if float(torch.rand((), generator=gen, device=mask.device).item()) < 0.7:
                add_connection(mask, gen, diag_mask, changes)
            else:
                flip_connection(mask, gen, changes)
        elif controller["mood"] == 1:
            r = float(torch.rand((), generator=gen, device=mask.device).item())
            if r < 0.6:
                rewire_connection(mask, gen, changes)
            elif r < 0.8:
                flip_connection(mask, gen, changes)
            else:
                add_connection(mask, gen, diag_mask, changes)
        elif controller["mood"] == 2:
            if float(torch.rand((), generator=gen, device=mask.device).item()) < 0.8:
                flip_connection(mask, gen, changes)
            else:
                rewire_connection(mask, gen, changes)
        else:
            r = float(torch.rand((), generator=gen, device=mask.device).item())
            if r < 0.7:
                remove_connection(mask, gen, changes)
            elif r < 0.9:
                flip_connection(mask, gen, changes)
            else:
                rewire_connection(mask, gen, changes)
    return prev, changes


def run_one(config_name: str, seed: int, attempts: int, kind: str):
    vocab, neurons, density = CONFIGS[config_name]
    device = torch.device("cuda")
    torch.manual_seed(seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    random.seed(seed)

    mask, leak, targets, out_start = gpu_init(vocab, neurons, density, seed, device)
    diag_mask = ~torch.eye(neurons, dtype=torch.bool, device=device)
    eye = torch.eye(vocab, dtype=torch.float32, device=device)

    if kind == "float":
        controller = {
            "kind": "float",
            "mood_x": torch.tensor(0.5, device=device, dtype=torch.float32),
            "mood_z": torch.tensor(0.5, device=device, dtype=torch.float32),
        }
        mutate = mutate_float
    else:
        controller = {
            "kind": "int",
            "mood": 2,
            "intensity": 7,
        }
        mutate = mutate_int

    score, acc = gpu_eval(mask, leak, targets, out_start, eye)
    best_score = score.clone()
    best_acc = acc.clone()
    kept = 0

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(attempts):
        prev, changes = mutate(mask, leak, controller, gen, diag_mask)
        new_score, new_acc = gpu_eval(mask, leak, targets, out_start, eye)
        if bool((new_score > score).item()):
            score = new_score
            kept += 1
            if bool((new_score > best_score).item()):
                best_score = new_score
                best_acc = new_acc
        else:
            rollback(mask, leak, controller, prev, changes)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    return {
        "config": config_name,
        "seed": seed,
        "kind": kind,
        "acc": float(best_acc.item()),
        "score": float(best_score.item()),
        "aps": attempts / dt if dt > 0 else float("inf"),
        "leak": float(leak.item()),
    }


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")
    args = parse_args()
    configs = [x.strip() for x in args.configs.split(",") if x.strip()]
    seeds = parse_csv_ints(args.seeds)

    print(f"GPU INT-vs-FLOAT MOOD A/B attempts={args.attempts} configs={configs} seeds={seeds}", flush=True)
    print("=" * 100, flush=True)
    results = []
    for config in configs:
        for kind in ("float", "int"):
            for seed in seeds:
                r = run_one(config, seed, args.attempts, kind)
                results.append(r)
                print(
                    f"{config:10s} {kind:5s} seed={seed:3d} "
                    f"acc={r['acc']*100:5.1f}% score={r['score']:.4f} aps={r['aps']:.1f} leak={r['leak']:.3f}",
                    flush=True,
                )

    print("\nSUMMARY", flush=True)
    for config in configs:
        float_rows = [r for r in results if r["config"] == config and r["kind"] == "float"]
        int_rows = [r for r in results if r["config"] == config and r["kind"] == "int"]
        f_acc = np.mean([r["acc"] for r in float_rows]) * 100.0
        i_acc = np.mean([r["acc"] for r in int_rows]) * 100.0
        f_score = np.mean([r["score"] for r in float_rows])
        i_score = np.mean([r["score"] for r in int_rows])
        f_aps = np.mean([r["aps"] for r in float_rows])
        i_aps = np.mean([r["aps"] for r in int_rows])
        print(
            f"{config:10s} float_acc={f_acc:5.1f}% int_acc={i_acc:5.1f}% diff={i_acc-f_acc:+.1f}pp | "
            f"float_score={f_score:.4f} int_score={i_score:.4f} | "
            f"float_aps={f_aps:.1f} int_aps={i_aps:.1f} speedup={i_aps/f_aps:.2f}x",
            flush=True,
        )


if __name__ == "__main__":
    raise SystemExit(main())
