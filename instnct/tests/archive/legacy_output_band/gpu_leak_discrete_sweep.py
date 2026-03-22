"""GPU leak discrete value sweep on larger configs.

Runs the current full-GPU reference mutation/eval path with float leak,
but records every accepted leak value quantized to hundredths. This tests
whether larger models actually want a small set of discrete leak bins, or
whether they use a broader range where float leak is doing real work.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tests.gpu_int_mood_ab import gpu_eval, gpu_init, mutate_int, rollback


CONFIGS = {
    "V128_N384": (128, 384, 0.06),
    "V256_N768": (256, 768, 0.06),
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="V128_N384,V256_N768")
    ap.add_argument("--budget", type=int, default=32000)
    ap.add_argument("--seeds", default="42,77,123")
    return ap.parse_args()


def parse_int_csv(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def quantize_leak_100(leak: float) -> int:
    """Map float leak to a 0..99 integer bucket."""
    q = int(round(leak * 100.0))
    return max(0, min(99, q))


def print_histogram(all_accepts: Counter[int]) -> None:
    print("GLOBAL LEAK HISTOGRAM (all seeds combined):", flush=True)
    print(f"  {'leak':>5s} {'accepts':>8s} {'bar'}", flush=True)
    max_count = max(all_accepts.values()) if all_accepts else 1
    for leak_int in range(50, 100):
        cnt = all_accepts.get(leak_int, 0)
        if cnt <= 0:
            continue
        bar = "#" * max(1, int(cnt / max_count * 40))
        print(f"  {leak_int/100:.2f} {cnt:8d} {bar}", flush=True)

    visited = sorted(all_accepts.keys())
    dead = [v for v in range(50, 100) if v not in all_accepts]
    print(f"\nActive values: {len(visited)}/50", flush=True)
    print(f"Dead values: {len(dead)}/50", flush=True)
    if visited:
        print(f"Visited range: {visited[0]/100:.2f} - {visited[-1]/100:.2f}", flush=True)
    if dead:
        dead_ranges: list[tuple[int, int]] = []
        start = dead[0]
        prev = dead[0]
        for val in dead[1:]:
            if val != prev + 1:
                dead_ranges.append((start, prev))
                start = val
            prev = val
        dead_ranges.append((start, prev))
        formatted = [f"{a/100:.2f}-{b/100:.2f}" for a, b in dead_ranges]
        print(f"Dead zones: {formatted}", flush=True)


def run_one(config_name: str, budget: int, seed: int):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    vocab, neurons, density = CONFIGS[config_name]
    device = torch.device("cuda")
    torch.manual_seed(seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    mask, leak, targets, out_start = gpu_init(vocab, neurons, density, seed, device)
    diag_mask = ~torch.eye(neurons, dtype=torch.bool, device=device)
    eye = torch.eye(vocab, dtype=torch.float32, device=device)
    controller = {"kind": "int", "mood": 2, "intensity": 7}

    score, acc = gpu_eval(mask, leak, targets, out_start, eye)
    best_score = score.clone()
    best_acc = acc.clone()
    kept = 0
    leak_accepts: Counter[int] = Counter()
    leak_trajectory: list[tuple[int, float]] = [(0, round(float(leak.item()), 3))]

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for att in range(budget):
        prev, changes = mutate_int(mask, leak, controller, gen, diag_mask)
        new_score, new_acc = gpu_eval(mask, leak, targets, out_start, eye)
        if bool((new_score > score).item()):
            score = new_score
            kept += 1
            if bool((new_score > best_score).item()):
                best_score = new_score
                best_acc = new_acc
            leak_accepts[quantize_leak_100(float(leak.item()))] += 1
        else:
            rollback(mask, leak, controller, prev, changes)

        if (att + 1) % 4000 == 0:
            leak_trajectory.append((att + 1, round(float(leak.item()), 3)))

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    leak_trajectory.append((budget, round(float(leak.item()), 3)))
    return {
        "best_acc": float(best_acc.item()),
        "best_score": float(best_score.item()),
        "final_leak": float(leak.item()),
        "kept": kept,
        "attempts_per_sec": budget / elapsed if elapsed > 0 else float("inf"),
        "accepts": leak_accepts,
        "trajectory": leak_trajectory,
        "elapsed": elapsed,
    }


def main() -> int:
    args = parse_args()
    config_names = [x.strip() for x in args.configs.split(",") if x.strip()]
    seeds = parse_int_csv(args.seeds)

    print("GPU LEAK DISCRETE VALUE SWEEP", flush=True)
    print(f"configs={config_names} budget={args.budget} seeds={seeds}", flush=True)
    print("=" * 80, flush=True)

    for config_name in config_names:
        if config_name not in CONFIGS:
            raise SystemExit(f"Unknown config: {config_name}")
        all_accepts: Counter[int] = Counter()
        vocab, neurons, density = CONFIGS[config_name]

        print(
            f"\n--- {config_name} (V={vocab} N={neurons} density={density}) ---",
            flush=True,
        )
        for seed in seeds:
            res = run_one(config_name, args.budget, seed)
            all_accepts += res["accepts"]
            print(
                f"\nseed={seed}: acc={res['best_acc']*100:.1f}% "
                f"score={res['best_score']:.4f} final_leak={res['final_leak']:.3f} "
                f"kept={res['kept']} aps={res['attempts_per_sec']:.1f} "
                f"({res['elapsed']:.0f}s)",
                flush=True,
            )
            print(f"  trajectory: {res['trajectory']}", flush=True)
            print("  top accepted leak values:", flush=True)
            for leak_int, cnt in res["accepts"].most_common(10):
                print(f"    {leak_int/100:.2f}: {cnt} accepts", flush=True)

        print(f"\n{'-'*80}", flush=True)
        print_histogram(all_accepts)
        print(f"\n{'='*80}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
