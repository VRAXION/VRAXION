"""Deterministic GPU plateau A/B for strategy-controller variants.

Run one config/variant/seed per invocation. The run stops only when:
  1. score, density, and accepted-count drift stay within tiny tolerances
  2. the controller state tuple is unchanged across the same checkpoint window
or when a config-specific safety cap is hit.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.log import live_log, log_msg
from tests.gpu_int_mood_ab import (
    CONFIGS,
    add_connection,
    flip_connection,
    gpu_eval,
    gpu_init,
    remove_connection,
    rewire_connection,
)


CHECKPOINT_EVERY = 1000
MIN_ATTEMPTS = 32000
PLATEAU_WINDOW = 4
PLATEAU_PATIENCE = 3
DENSITY_EPS = 0.0015
SCORE_EPS = 0.0010
ACCEPT_EPS = 48

SAFETY_CAPS = {
    "V128_N384": 220000,
    "V256_N768": 300000,
}

VARIANTS = {"mood4_coupled", "two_bit_decoupled"}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=sorted(SAFETY_CAPS))
    ap.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY)
    ap.add_argument("--min-attempts", type=int, default=MIN_ATTEMPTS)
    ap.add_argument("--plateau-window", type=int, default=PLATEAU_WINDOW)
    ap.add_argument("--plateau-patience", type=int, default=PLATEAU_PATIENCE)
    ap.add_argument("--density-eps", type=float, default=DENSITY_EPS)
    ap.add_argument("--score-eps", type=float, default=SCORE_EPS)
    ap.add_argument("--accept-eps", type=int, default=ACCEPT_EPS)
    ap.add_argument("--safety-cap", type=int, default=0, help="0 => config default")
    ap.add_argument("--log-name", default="")
    return ap.parse_args()


def mask_density(mask: torch.Tensor) -> float:
    n = mask.shape[0]
    total = n * n - n
    return float((mask != 0).sum().item()) / float(total)


def retention_from_loss(loss_pct_t: torch.Tensor) -> torch.Tensor:
    return 1.0 - loss_pct_t.to(torch.float32) * 0.01


def controller_state(variant: str, controller: dict, loss_pct: torch.Tensor) -> tuple[int, ...]:
    loss = int(loss_pct.item())
    if variant == "mood4_coupled":
        return (int(controller["mood"]), int(controller["intensity"]), loss)
    return (int(controller["signal"]), int(controller["grow"]), int(controller["intensity"]), loss)


def window_frozen(history, window: int, density_eps: float, score_eps: float, accept_eps: int) -> bool:
    if len(history) < window:
        return False
    chunk = history[-window:]
    density_delta = abs(chunk[-1]["density"] - chunk[0]["density"])
    score_delta = abs(chunk[-1]["best_score"] - chunk[0]["best_score"])
    accept_delta = chunk[-1]["accepted"] - chunk[0]["accepted"]
    state_frozen = all(item["controller_state"] == chunk[0]["controller_state"] for item in chunk[1:])
    return (
        density_delta <= density_eps
        and score_delta <= score_eps
        and accept_delta <= accept_eps
        and state_frozen
    )


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


def run_one(args, log_q=None):
    vocab, neurons, density = CONFIGS[args.config]
    variant = args.variant
    device = torch.device("cuda")
    seed = args.seed

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
    accepted = 0
    init_density = mask_density(mask)

    checkpoint_history = [
        {
            "att": 0,
            "density": init_density,
            "best_score": float(best_score.item()),
            "accepted": 0,
            "loss_pct": int(loss_pct.item()),
            "controller_state": controller_state(variant, controller, loss_pct),
        }
    ]
    frozen_windows = 0
    safety_cap = args.safety_cap or SAFETY_CAPS[args.config]
    stop_reason = "safety_cap"
    stop_att = safety_cap

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for att in range(1, safety_cap + 1):
        prev, changes = mutate(mask, loss_pct, controller, gen, diag_mask)
        new_score, new_acc = gpu_eval(mask, retention_from_loss(loss_pct), targets, out_start, eye)

        if bool((new_score > score).item()):
            score = new_score
            accepted += 1
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

        if att % args.checkpoint_every == 0:
            point = {
                "att": att,
                "density": mask_density(mask),
                "best_score": float(best_score.item()),
                "accepted": accepted,
                "loss_pct": int(loss_pct.item()),
                "controller_state": controller_state(variant, controller, loss_pct),
            }
            checkpoint_history.append(point)
            frozen = window_frozen(
                checkpoint_history,
                args.plateau_window,
                args.density_eps,
                args.score_eps,
                args.accept_eps,
            )
            frozen_windows = frozen_windows + 1 if frozen else 0
            log_msg(
                log_q,
                f"{args.config:10s} {variant:18s} seed={seed:3d} att={att:6d} "
                f"best_acc={float(best_acc.item())*100:5.1f}% score={float(best_score.item()):.4f} "
                f"density={point['density']:.4f} accepted={accepted:5d} loss={point['loss_pct']:2d}% "
                f"ctrl={point['controller_state']} frozen_win={frozen_windows:2d}",
            )
            if att >= args.min_attempts and frozen_windows >= args.plateau_patience:
                stop_reason = "plateau"
                stop_att = att
                break

    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    result = {
        "config": args.config,
        "variant": variant,
        "seed": seed,
        "best_acc": float(best_acc.item()),
        "best_score": float(best_score.item()),
        "final_density": mask_density(mask),
        "final_loss_pct": int(loss_pct.item()),
        "final_controller_state": controller_state(variant, controller, loss_pct),
        "accepted": accepted,
        "stop_reason": stop_reason,
        "stop_attempt": stop_att,
        "attempts_per_sec": stop_att / dt if dt > 0 else float("inf"),
        "checkpoint_history": checkpoint_history,
    }
    log_msg(
        log_q,
        f"{args.config:10s} {variant:18s} seed={seed:3d} FINAL "
        f"acc={result['best_acc']*100:5.1f}% score={result['best_score']:.4f} "
        f"density={init_density:.4f}->{result['final_density']:.4f} "
        f"loss={result['final_loss_pct']:2d}% ctrl={result['final_controller_state']} "
        f"accepted={accepted:5d} stop={stop_reason}@{stop_att} aps={result['attempts_per_sec']:.1f}",
    )
    return result


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")
    args = parse_args()
    log_name = args.log_name or f"gpu_strategy_plateau_{args.config}_{args.variant}_seed{args.seed}"
    log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")

    print(
        f"GPU STRATEGY PLATEAU config={args.config} variant={args.variant} seed={args.seed} "
        f"safety_cap={args.safety_cap or SAFETY_CAPS[args.config]}",
        flush=True,
    )
    print(
        f"plateau: min_attempts={args.min_attempts} checkpoint={args.checkpoint_every} "
        f"window={args.plateau_window} patience={args.plateau_patience} "
        f"density_eps={args.density_eps} score_eps={args.score_eps} accept_eps={args.accept_eps}",
        flush=True,
    )
    print("=" * 120, flush=True)

    with live_log(log_name, log_dir=log_dir) as (log_q, log_path):
        result = run_one(args, log_q=log_q)
        log_msg(log_q, "RESULT_JSON " + json.dumps(result, sort_keys=True))
    print(f"LOG_PATH={log_path}", flush=True)
    print("RESULT_JSON " + json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
