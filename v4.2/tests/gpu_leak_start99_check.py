"""Quick GPU check: if leak starts at 99, does accepted evolution move it down?

Uses the current int controller semantics:
  - mood: int zone
  - intensity: int step count
  - leak: int bucket 50..99, mutated by +/-1..3
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tests.gpu_int_mood_ab import (
    add_connection,
    flip_connection,
    gpu_eval,
    gpu_init,
    remove_connection,
    rewire_connection,
    rollback,
)


CONFIGS = {
    "V64_N192": (64, 192, 0.06),
    "V128_N384": (128, 384, 0.06),
    "V256_N768": (256, 768, 0.06),
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="V128_N384")
    ap.add_argument("--seeds", default="42,77,123")
    ap.add_argument("--attempts", type=int, default=4000)
    ap.add_argument("--start-leak", type=int, default=99)
    return ap.parse_args()


def parse_csv_ints(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def mutate_int_leak(mask, leak_int, controller, gen, diag_mask):
    changes = []
    prev = {
        "leak": int(leak_int.item()),
        "mood": controller["mood"],
        "intensity": controller["intensity"],
    }
    if float(torch.rand((), generator=gen, device=mask.device).item()) < 0.35:
        controller["mood"] = max(0, min(3, controller["mood"] + random.choice([-1, 1])))
    if float(torch.rand((), generator=gen, device=mask.device).item()) < 0.35:
        controller["intensity"] = max(1, min(15, controller["intensity"] + random.choice([-1, 1])))
    if float(torch.rand((), generator=gen, device=mask.device).item()) < 0.2:
        leak_int.fill_(max(50, min(99, int(leak_int.item()) + random.randint(-3, 3))))

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


def run_one(config_name: str, seed: int, attempts: int, start_leak: int):
    vocab, neurons, density = CONFIGS[config_name]
    device = torch.device("cuda")
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    mask, _leak_base, targets, out_start = gpu_init(vocab, neurons, density, seed, device)
    leak_int = torch.tensor(start_leak, device=device, dtype=torch.int16)
    diag_mask = ~torch.eye(neurons, dtype=torch.bool, device=device)
    eye = torch.eye(vocab, dtype=torch.float32, device=device)
    controller = {"kind": "int", "mood": 2, "intensity": 7}

    leak_float = leak_int.to(torch.float32) * 0.01
    score, acc = gpu_eval(mask, leak_float, targets, out_start, eye)
    best_acc = float(acc.item())
    min_accepted_leak = int(leak_int.item())
    first_drop = None
    accepted_below_start = 0
    trajectory = [(0, int(leak_int.item()))]

    for att in range(1, attempts + 1):
        prev, changes = mutate_int_leak(mask, leak_int, controller, gen, diag_mask)
        leak_float = leak_int.to(torch.float32) * 0.01
        new_score, new_acc = gpu_eval(mask, leak_float, targets, out_start, eye)
        if bool((new_score > score).item()):
            score = new_score
            best_acc = max(best_acc, float(new_acc.item()))
            lk = int(leak_int.item())
            if lk < start_leak:
                accepted_below_start += 1
            if lk < min_accepted_leak:
                min_accepted_leak = lk
                if first_drop is None:
                    first_drop = (att, lk)
        else:
            rollback(
                mask,
                leak_float,
                controller,
                {
                    "leak": float(prev["leak"]) * 0.01,
                    "mood": prev["mood"],
                    "intensity": prev["intensity"],
                },
                changes,
            )
            leak_int.fill_(prev["leak"])

        if att % 1000 == 0:
            trajectory.append((att, int(leak_int.item())))

    return {
        "best_acc": best_acc,
        "final_leak": int(leak_int.item()),
        "min_accepted_leak": min_accepted_leak,
        "first_drop": first_drop,
        "accepted_below_start": accepted_below_start,
        "trajectory": trajectory,
    }


def main() -> int:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")

    args = parse_args()
    configs = [x.strip() for x in args.configs.split(",") if x.strip()]
    seeds = parse_csv_ints(args.seeds)
    print(
        f"GPU LEAK START99 CHECK configs={configs} attempts={args.attempts} "
        f"start_leak={args.start_leak} seeds={seeds}",
        flush=True,
    )
    print("=" * 80, flush=True)
    for config_name in configs:
        print(f"\n--- {config_name} ---", flush=True)
        for seed in seeds:
            res = run_one(config_name, seed, args.attempts, args.start_leak)
            print(
                f"seed={seed}: best_acc={res['best_acc']*100:.1f}% "
                f"final_leak={res['final_leak']} min_accepted_leak={res['min_accepted_leak']} "
                f"accepted_below_start={res['accepted_below_start']} first_drop={res['first_drop']}",
                flush=True,
            )
            print(f"  trajectory: {res['trajectory']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
