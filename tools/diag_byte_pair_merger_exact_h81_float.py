"""Train an exact H=81 pure-float merger checkpoint.

This is the exact-first canonical stage for the current L1 merger pipeline:

  32 -> 81 -> 32
  mirror-tied
  C19 activation in the hidden layer

Protocol:
  1. Start from scratch or an optional exact/warm source artifact.
  2. Adam warmup.
  3. LBFGS plateau.
  4. Optional Adam rescue + LBFGS if still not exact.
  5. Stop only when bad_pairs == 0 and lossless == 100.0000%.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from diag_byte_pair_merger_exact_utils import (
    DEVICE,
    PureFloatMerger,
    eval_stats,
    export_pure_float_json,
    load_byte_pairs,
    load_effective_pure_float,
    seed_all,
    train_adam,
    train_lbfgs_plateau,
)


def clone_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def format_stats(stats: dict[str, float | int | bool]) -> str:
    return (
        f"ll={stats['lossless']:.4f}% "
        f"bad_pairs={stats['bad_pairs']} "
        f"bad_dims={stats['bad_dims']} "
        f"mse={stats['mse']:.6e}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--restarts", type=int, default=1)
    parser.add_argument("--hidden", type=int, default=81)
    parser.add_argument("--init-source", type=str, default="")
    parser.add_argument("--adam-epochs", type=int, default=2500)
    parser.add_argument("--adam-lr", type=float, default=1e-3)
    parser.add_argument("--adam-print-every", type=int, default=250)
    parser.add_argument("--lbfgs-outer", type=int, default=250)
    parser.add_argument("--lbfgs-patience", type=int, default=35)
    parser.add_argument("--lbfgs-print-every", type=int, default=10)
    parser.add_argument("--rescue-adam-epochs", type=int, default=750)
    parser.add_argument("--rescue-adam-lr", type=float, default=3e-4)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    runlog_path = out_dir / "run.log"
    metrics_path = out_dir / "metrics.json"
    checkpoint_path = out_dir / "best_checkpoint.pt"

    def log(msg: str) -> None:
        print(msg, flush=True)
        with open(runlog_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    data = load_byte_pairs().to(DEVICE)
    t0 = time.time()
    best_model = None
    best_stats = None
    best_restart = None

    log("=== EXACT H81 PURE FLOAT TRAINER ===")
    log(f"Device: {DEVICE}")
    log(f"Init source: {args.init_source or '<scratch>'}")
    log(f"Out: {out_dir}")

    for restart in range(args.restarts):
        seed = args.seed + restart
        seed_all(seed)
        log(f"\n--- Restart {restart+1}/{args.restarts} (seed={seed}) ---")

        if args.init_source:
            model = load_effective_pure_float(args.init_source)
        else:
            model = PureFloatMerger(hidden=args.hidden, in_dim=32, out_dim=32)
        model = model.to(DEVICE)

        stats0 = eval_stats(model, data)
        log(f"Baseline: {format_stats(stats0)}")

        stats = stats0
        if not stats["exact"] and args.adam_epochs > 0:
            log("\n[Phase 1] Adam warmup")
            stats = train_adam(
                model,
                data,
                n_epochs=args.adam_epochs,
                lr=args.adam_lr,
                print_every=args.adam_print_every,
                tag=f"adam-r{restart+1}",
                log=log,
            )
            log(f"After Adam: {format_stats(stats)}")

        if not stats["exact"]:
            log("\n[Phase 2] LBFGS plateau")
            stats = train_lbfgs_plateau(
                model,
                data,
                max_outer=args.lbfgs_outer,
                patience=args.lbfgs_patience,
                print_every=args.lbfgs_print_every,
                tag=f"lbfgs-r{restart+1}",
                log=log,
            )
            log(f"After LBFGS: {format_stats(stats)}")

        if not stats["exact"] and args.rescue_adam_epochs > 0:
            log("\n[Phase 3] Adam rescue")
            stats = train_adam(
                model,
                data,
                n_epochs=args.rescue_adam_epochs,
                lr=args.rescue_adam_lr,
                print_every=max(100, args.adam_print_every),
                tag=f"rescue-r{restart+1}",
                log=log,
            )
            log(f"After rescue Adam: {format_stats(stats)}")

        if not stats["exact"]:
            log("\n[Phase 4] Final LBFGS plateau")
            stats = train_lbfgs_plateau(
                model,
                data,
                max_outer=max(60, args.lbfgs_outer // 2),
                patience=max(15, args.lbfgs_patience // 2),
                print_every=args.lbfgs_print_every,
                tag=f"lbfgs-final-r{restart+1}",
                log=log,
            )
            log(f"After final LBFGS: {format_stats(stats)}")

        if best_stats is None or (
            int(stats["bad_pairs"]),
            float(stats["mse"]),
            int(stats["bad_dims"]),
        ) < (
            int(best_stats["bad_pairs"]),
            float(best_stats["mse"]),
            int(best_stats["bad_dims"]),
        ):
            best_model = clone_state_dict(model)
            best_stats = stats
            best_restart = restart + 1
            torch.save(
                {
                    "state_dict": best_model,
                    "stats": best_stats,
                    "restart": best_restart,
                },
                checkpoint_path,
            )
            log(f"New best @ restart {best_restart}: {format_stats(best_stats)}")

        if stats["exact"]:
            log(f"Exact winner found on restart {restart+1}.")
            break

    assert best_model is not None and best_stats is not None
    final_model = PureFloatMerger(hidden=args.hidden, in_dim=32, out_dim=32).to(DEVICE)
    final_model.load_state_dict(best_model, strict=True)
    final_stats = eval_stats(final_model, data)
    elapsed = time.time() - t0

    export_pure_float_json(
        out_dir / "final_model.json",
        final_model,
        final_stats,
        meta={
            "source": args.init_source or None,
            "seed": args.seed,
            "restarts": args.restarts,
            "best_restart": best_restart,
            "time_s": elapsed,
        },
    )
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "exact": bool(final_stats["exact"]),
                "lossless": float(final_stats["lossless"]),
                "per_dim": float(final_stats["per_dim"]),
                "mse": float(final_stats["mse"]),
                "bad_pairs": int(final_stats["bad_pairs"]),
                "bad_dims": int(final_stats["bad_dims"]),
                "best_restart": best_restart,
                "time_s": elapsed,
            },
            f,
            indent=2,
        )

    log("\n=== DONE ===")
    log(f"Best restart: {best_restart}")
    log(f"Final: {format_stats(final_stats)}")
    log(f"Time: {elapsed:.1f}s")

    if not final_stats["exact"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
