"""GPU tick-budget probe for grow -> pass-crystal plateaus.

Question:
  Does changing forward tick budget materially change the crystal plateau
  edge count and score for the same seeded add-only growth pipeline?
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tests.gpu_experimental.gpu_crystal_pass_ab import (
    build_start_state,
    clone_state,
    crystal_pass_based,
    make_score_runner,
    parse_csv_ints,
)


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_float32_matmul_precision("high")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="64", help="Comma-separated V sizes")
    ap.add_argument("--seeds", default="42,77", help="Comma-separated integer seeds")
    ap.add_argument("--ticks-list", default="6,10,15", help="Comma-separated tick counts")
    ap.add_argument("--grow-attempts", type=int, default=2048, help="Add-only growth attempts before crystal")
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--compile-eval", action="store_true")
    ap.add_argument("--output", default="", help="Optional explicit JSON output path")
    return ap.parse_args()


def run_case(
    V: int,
    seed: int,
    ticks: int,
    grow_attempts: int,
    eps: float,
    device: torch.device,
    compile_eval: bool,
) -> dict:
    case, start_state, grow_meta = build_start_state(
        V=V,
        seed=seed,
        grow_attempts=grow_attempts,
        ticks=ticks,
        eps=eps,
        device=device,
        compile_eval=compile_eval,
    )
    score_runner = make_score_runner(case, ticks, device, compile_eval)
    crystal = crystal_pass_based(
        state=clone_state(start_state),
        score_runner=score_runner,
        rng=random.Random(seed + 900_000 + V + ticks),
        eps=eps,
        floor_score=float(start_state.score) - eps,
        max_passes=0,
        max_wall_ms=0,
    )
    return {
        "V": V,
        "seed": seed,
        "ticks": ticks,
        "grow_attempts": grow_attempts,
        "score_before": float(start_state.score),
        "score_after": float(crystal["score_after"]),
        "edges_after_grow": int(grow_meta["edges_after_grow"]),
        "edges_after_crystal": int(crystal["edges_after"]),
        "removed_count": int(crystal["removed_count"]),
        "removed_pct": float(crystal["removed_pct"]),
        "passes": int(crystal["passes"]),
        "attempted_removes": int(crystal["attempted_removes"]),
        "wall_ms": float(crystal["wall_ms"]),
    }


def summarize(results: list[dict]) -> dict:
    summary: dict[str, dict[str, float]] = {}
    for ticks in sorted({r["ticks"] for r in results}):
        rows = [r for r in results if r["ticks"] == ticks]
        summary[str(ticks)] = {
            "n_cases": len(rows),
            "score_before_mean": float(np.mean([r["score_before"] for r in rows])),
            "score_after_mean": float(np.mean([r["score_after"] for r in rows])),
            "edges_after_grow_mean": float(np.mean([r["edges_after_grow"] for r in rows])),
            "edges_after_crystal_mean": float(np.mean([r["edges_after_crystal"] for r in rows])),
            "edges_after_crystal_median": float(np.median([r["edges_after_crystal"] for r in rows])),
            "removed_pct_mean": float(np.mean([r["removed_pct"] for r in rows])),
            "passes_mean": float(np.mean([r["passes"] for r in rows])),
        }
    return summary


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        print("CUDA not available.")
        return 1

    device = torch.device("cuda")
    configs = parse_csv_ints(args.configs)
    seeds = parse_csv_ints(args.seeds)
    ticks_list = parse_csv_ints(args.ticks_list)

    results = []
    for V in configs:
        for seed in seeds:
            for ticks in ticks_list:
                row = run_case(
                    V=V,
                    seed=seed,
                    ticks=ticks,
                    grow_attempts=args.grow_attempts,
                    eps=args.eps,
                    device=device,
                    compile_eval=args.compile_eval,
                )
                results.append(row)
                print(
                    f"V={V} seed={seed} ticks={ticks} "
                    f"score={row['score_after']*100:6.2f}% "
                    f"grow_edges={row['edges_after_grow']:4d} "
                    f"crystal_edges={row['edges_after_crystal']:4d} "
                    f"removed={row['removed_pct']*100:5.1f}%"
                )

    summary = summarize(results)
    payload = {
        "device": torch.cuda.get_device_name(0),
        "branch_intent": "gpu tick-budget crystal plateau probe",
        "args": vars(args),
        "results": results,
        "summary": summary,
    }

    if args.output:
        out_path = Path(args.output)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path("S:/AI/work/VRAXION_DEV/logs") / f"gpu_tick_plateau_probe_{stamp}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved report -> {out_path}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
