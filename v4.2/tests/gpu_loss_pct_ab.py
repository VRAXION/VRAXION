"""GPU A/B: legacy leak-int vs new loss_pct-int parameterization.

Both variants use the same int mood/intensity controller and the same GPU eval
kernel. Only the scalar representation differs:

- leak_int: 50..99, interpreted as retention = leak_int / 100
- loss_pct: 1..50, interpreted as retention = 1 - loss_pct / 100
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

from tests.gpu_int_mood_ab import (
    CONFIGS,
    add_connection,
    flip_connection,
    gpu_eval,
    gpu_init,
    remove_connection,
    rewire_connection,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="V128_N384", help="Comma-separated config names")
    ap.add_argument("--attempts", type=int, default=16000)
    ap.add_argument("--seeds", default="42,77,123")
    return ap.parse_args()


def parse_csv_ints(raw: str):
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def mutate_common_ops(mask, controller, gen, diag_mask, changes):
    if float(torch.rand((), generator=gen, device=mask.device).item()) < 0.35:
        controller["mood"] = max(0, min(3, controller["mood"] + random.choice([-1, 1])))
    if float(torch.rand((), generator=gen, device=mask.device).item()) < 0.35:
        controller["intensity"] = max(1, min(15, controller["intensity"] + random.choice([-1, 1])))

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


def mutate_leak_int(mask, leak_int, controller, gen, diag_mask):
    changes = []
    prev = {
        "kind": "leak_int",
        "value": int(leak_int.item()),
        "mood": controller["mood"],
        "intensity": controller["intensity"],
    }
    if float(torch.rand((), generator=gen, device=mask.device).item()) < 0.2:
        leak_int.fill_(max(50, min(99, int(leak_int.item()) + random.randint(-3, 3))))
    mutate_common_ops(mask, controller, gen, diag_mask, changes)
    return prev, changes


def mutate_loss_pct(mask, loss_pct, controller, gen, diag_mask):
    changes = []
    prev = {
        "kind": "loss_pct",
        "value": int(loss_pct.item()),
        "mood": controller["mood"],
        "intensity": controller["intensity"],
    }
    if float(torch.rand((), generator=gen, device=mask.device).item()) < 0.2:
        loss_pct.fill_(max(1, min(50, int(loss_pct.item()) + random.randint(-3, 3))))
    mutate_common_ops(mask, controller, gen, diag_mask, changes)
    return prev, changes


def rollback_variant(mask, scalar_t, controller, prev_state, changes):
    for row, col, old in reversed(changes):
        mask[row, col] = old
    scalar_t.fill_(prev_state["value"])
    controller["mood"] = prev_state["mood"]
    controller["intensity"] = prev_state["intensity"]


def run_one(config_name: str, seed: int, attempts: int, variant: str):
    vocab, neurons, density = CONFIGS[config_name]
    device = torch.device("cuda")
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    mask, _base_leak, targets, out_start = gpu_init(vocab, neurons, density, seed, device)
    diag_mask = ~torch.eye(neurons, dtype=torch.bool, device=device)
    eye = torch.eye(vocab, dtype=torch.float32, device=device)
    controller = {"mood": 2, "intensity": 7}

    if variant == "leak_int":
        scalar_t = torch.tensor(85, device=device, dtype=torch.int16)
        mutate = mutate_leak_int
        retention = lambda: scalar_t.to(torch.float32) * 0.01
    else:
        scalar_t = torch.tensor(15, device=device, dtype=torch.int16)
        mutate = mutate_loss_pct
        retention = lambda: 1.0 - scalar_t.to(torch.float32) * 0.01

    score, acc = gpu_eval(mask, retention(), targets, out_start, eye)
    best_score = score.clone()
    best_acc = acc.clone()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(attempts):
        prev, changes = mutate(mask, scalar_t, controller, gen, diag_mask)
        new_score, new_acc = gpu_eval(mask, retention(), targets, out_start, eye)
        if bool((new_score > score).item()):
            score = new_score
            if bool((new_score > best_score).item()):
                best_score = new_score
                best_acc = new_acc
        else:
            rollback_variant(mask, scalar_t, controller, prev, changes)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    if variant == "leak_int":
        final_ret = float(int(scalar_t.item()) * 0.01)
        scalar_label = int(scalar_t.item())
    else:
        final_ret = float(1.0 - int(scalar_t.item()) * 0.01)
        scalar_label = int(scalar_t.item())

    return {
        "config": config_name,
        "seed": seed,
        "variant": variant,
        "acc": float(best_acc.item()),
        "score": float(best_score.item()),
        "aps": attempts / dt if dt > 0 else float("inf"),
        "scalar": scalar_label,
        "retention": final_ret,
    }


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")
    args = parse_args()
    configs = [x.strip() for x in args.configs.split(",") if x.strip()]
    seeds = parse_csv_ints(args.seeds)

    print(f"GPU LEAK-INT vs LOSS_PCT A/B attempts={args.attempts} configs={configs} seeds={seeds}", flush=True)
    print("=" * 100, flush=True)
    results = []
    for config in configs:
        for variant in ("leak_int", "loss_pct"):
            for seed in seeds:
                r = run_one(config, seed, args.attempts, variant)
                results.append(r)
                scalar_name = "leak" if variant == "leak_int" else "loss_pct"
                print(
                    f"{config:10s} {variant:8s} seed={seed:3d} "
                    f"acc={r['acc']*100:5.1f}% score={r['score']:.4f} aps={r['aps']:.1f} "
                    f"{scalar_name}={r['scalar']:2d} ret={r['retention']:.2f}",
                    flush=True,
                )

    print("\nSUMMARY", flush=True)
    for config in configs:
        leak_rows = [r for r in results if r["config"] == config and r["variant"] == "leak_int"]
        loss_rows = [r for r in results if r["config"] == config and r["variant"] == "loss_pct"]
        l_acc = np.mean([r["acc"] for r in leak_rows]) * 100.0
        p_acc = np.mean([r["acc"] for r in loss_rows]) * 100.0
        l_score = np.mean([r["score"] for r in leak_rows])
        p_score = np.mean([r["score"] for r in loss_rows])
        l_aps = np.mean([r["aps"] for r in leak_rows])
        p_aps = np.mean([r["aps"] for r in loss_rows])
        print(
            f"{config:10s} leak_acc={l_acc:5.1f}% loss_acc={p_acc:5.1f}% diff={p_acc-l_acc:+.1f}pp | "
            f"leak_score={l_score:.4f} loss_score={p_score:.4f} | "
            f"leak_aps={l_aps:.1f} loss_aps={p_aps:.1f} speedup={p_aps/max(1e-9,l_aps):.2f}x",
            flush=True,
        )


if __name__ == "__main__":
    main()
