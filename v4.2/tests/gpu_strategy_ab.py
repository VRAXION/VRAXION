"""GPU A/B: 4-zone coupled mood vs 2-bit decoupled strategy.

Both variants share the same dense CUDA eval kernel and the same fixed scalar
physics. Only the mutation controller differs:

- mood4_coupled:
    int mood {0..3} + intensity, both mutated before the attempt and fully
    reverted on reject together with mask/loss_pct.
- two_bit_decoupled:
    signal/grow/intensity/loss_pct as in the current graph.py logic.
    On reject only mask/loss_pct revert; strategy bits survive and signal/grow
    may flip to search a different regime.
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
    ap.add_argument("--configs", default="V128_N384,V256_N768", help="Comma-separated config names")
    ap.add_argument("--attempts", type=int, default=16000)
    ap.add_argument("--seeds", default="42,77,123")
    return ap.parse_args()


def parse_csv_ints(raw: str):
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def retention_from_loss(loss_pct_t: torch.Tensor) -> torch.Tensor:
    return 1.0 - loss_pct_t.to(torch.float32) * 0.01


def mutate_mood4(mask, loss_pct, controller, gen, diag_mask):
    changes = []
    prev = {
        "loss_pct": int(loss_pct.item()),
        "mood": controller["mood"],
        "intensity": controller["intensity"],
    }
    if float(torch.rand((), generator=gen, device=mask.device).item()) < 0.35:
        controller["mood"] = max(0, min(3, controller["mood"] + random.choice([-1, 1])))
    if float(torch.rand((), generator=gen, device=mask.device).item()) < 0.35:
        controller["intensity"] = max(1, min(15, controller["intensity"] + random.choice([-1, 1])))
    if float(torch.rand((), generator=gen, device=mask.device).item()) < 0.2:
        loss_pct.fill_(max(1, min(50, int(loss_pct.item()) + random.randint(-3, 3))))

    for _ in range(controller["intensity"]):
        mood = controller["mood"]
        if mood == 0:
            if float(torch.rand((), generator=gen, device=mask.device).item()) < 0.7:
                add_connection(mask, gen, diag_mask, changes)
            else:
                flip_connection(mask, gen, changes)
        elif mood == 1:
            r = float(torch.rand((), generator=gen, device=mask.device).item())
            if r < 0.6:
                rewire_connection(mask, gen, changes)
            elif r < 0.8:
                flip_connection(mask, gen, changes)
            else:
                add_connection(mask, gen, diag_mask, changes)
        elif mood == 2:
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


def rollback_mood4(mask, loss_pct, controller, prev_state, changes):
    for row, col, old in reversed(changes):
        mask[row, col] = old
    loss_pct.fill_(prev_state["loss_pct"])
    controller["mood"] = prev_state["mood"]
    controller["intensity"] = prev_state["intensity"]


def mutate_two_bit(mask, loss_pct, controller, gen, diag_mask):
    changes = []
    prev = {
        "loss_pct": int(loss_pct.item()),
    }
    if float(torch.rand((), generator=gen, device=mask.device).item()) < 0.35:
        controller["intensity"] = max(1, min(15, controller["intensity"] + random.choice([-1, 1])))
    if float(torch.rand((), generator=gen, device=mask.device).item()) < 0.2:
        loss_pct.fill_(max(1, min(50, int(loss_pct.item()) + random.randint(-3, 3))))

    for _ in range(controller["intensity"]):
        if controller["signal"]:
            flip_connection(mask, gen, changes)
        else:
            if controller["grow"]:
                add_connection(mask, gen, diag_mask, changes)
            else:
                if float(torch.rand((), generator=gen, device=mask.device).item()) < 0.7:
                    remove_connection(mask, gen, changes)
                else:
                    rewire_connection(mask, gen, changes)
    return prev, changes


def rollback_two_bit(mask, loss_pct, prev_state, changes):
    for row, col, old in reversed(changes):
        mask[row, col] = old
    loss_pct.fill_(prev_state["loss_pct"])


def run_one(config_name: str, seed: int, attempts: int, variant: str):
    vocab, neurons, density = CONFIGS[config_name]
    device = torch.device("cuda")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    mask, _leak, targets, out_start = gpu_init(vocab, neurons, density, seed, device)
    diag_mask = ~torch.eye(neurons, dtype=torch.bool, device=device)
    eye = torch.eye(vocab, dtype=torch.float32, device=device)
    loss_pct = torch.tensor(15, device=device, dtype=torch.int16)

    if variant == "mood4_coupled":
        controller = {"mood": 2, "intensity": 7}
        mutate = mutate_mood4
    else:
        controller = {"signal": 0, "grow": 1, "intensity": 7}
        mutate = mutate_two_bit

    score, acc = gpu_eval(mask, retention_from_loss(loss_pct), targets, out_start, eye)
    best_score = score.clone()
    best_acc = acc.clone()
    kept = 0

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(attempts):
        prev, changes = mutate(mask, loss_pct, controller, gen, diag_mask)
        new_score, new_acc = gpu_eval(mask, retention_from_loss(loss_pct), targets, out_start, eye)
        if bool((new_score > score).item()):
            score = new_score
            kept += 1
            if bool((new_score > best_score).item()):
                best_score = new_score
                best_acc = new_acc
        else:
            if variant == "mood4_coupled":
                rollback_mood4(mask, loss_pct, controller, prev, changes)
            else:
                rollback_two_bit(mask, loss_pct, prev, changes)
                if float(torch.rand((), generator=gen, device=mask.device).item()) < 0.35:
                    controller["signal"] = 1 - controller["signal"]
                if float(torch.rand((), generator=gen, device=mask.device).item()) < 0.35:
                    controller["grow"] = 1 - controller["grow"]
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    result = {
        "config": config_name,
        "seed": seed,
        "variant": variant,
        "acc": float(best_acc.item()),
        "score": float(best_score.item()),
        "aps": attempts / dt if dt > 0 else float("inf"),
        "loss_pct": int(loss_pct.item()),
        "retention": float(retention_from_loss(loss_pct).item()),
        "connections": int((mask != 0).sum().item()),
    }
    if variant == "mood4_coupled":
        result["mood"] = controller["mood"]
        result["intensity"] = controller["intensity"]
    else:
        result["signal"] = controller["signal"]
        result["grow"] = controller["grow"]
        result["intensity"] = controller["intensity"]
    return result


def print_row(r):
    if r["variant"] == "mood4_coupled":
        extra = f"mood={r['mood']} int={r['intensity']}"
    else:
        mode = "SIGNAL" if r["signal"] else ("GROW" if r["grow"] else "SHRINK")
        extra = f"{mode} int={r['intensity']}"
    print(
        f"{r['config']:10s} {r['variant']:15s} seed={r['seed']:3d} "
        f"acc={r['acc']*100:5.1f}% score={r['score']:.4f} aps={r['aps']:.1f} "
        f"{extra} loss={r['loss_pct']:2d}% conns={r['connections']}",
        flush=True,
    )


def summarize(results, configs):
    print("\nSUMMARY", flush=True)
    for config in configs:
        rows_a = [r for r in results if r["config"] == config and r["variant"] == "mood4_coupled"]
        rows_b = [r for r in results if r["config"] == config and r["variant"] == "two_bit_decoupled"]
        a_acc = np.mean([r["acc"] for r in rows_a]) * 100.0
        b_acc = np.mean([r["acc"] for r in rows_b]) * 100.0
        a_score = np.mean([r["score"] for r in rows_a])
        b_score = np.mean([r["score"] for r in rows_b])
        a_aps = np.mean([r["aps"] for r in rows_a])
        b_aps = np.mean([r["aps"] for r in rows_b])
        print(
            f"{config:10s} "
            f"mood4_acc={a_acc:5.1f}% two_bit_acc={b_acc:5.1f}% diff={b_acc-a_acc:+.1f}pp | "
            f"mood4_score={a_score:.4f} two_bit_score={b_score:.4f} | "
            f"mood4_aps={a_aps:.1f} two_bit_aps={b_aps:.1f} speedup={b_aps/max(1e-9,a_aps):.2f}x",
            flush=True,
        )


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")
    args = parse_args()
    configs = [x.strip() for x in args.configs.split(",") if x.strip()]
    seeds = parse_csv_ints(args.seeds)

    print(
        f"GPU STRATEGY A/B attempts={args.attempts} configs={configs} seeds={seeds}",
        flush=True,
    )
    print("=" * 100, flush=True)
    results = []
    for config in configs:
        for variant in ("mood4_coupled", "two_bit_decoupled"):
            for seed in seeds:
                r = run_one(config, seed, args.attempts, variant)
                results.append(r)
                print_row(r)
    summarize(results, configs)


if __name__ == "__main__":
    main()
