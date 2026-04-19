"""Minimum-hidden activation sweep for the pure 2-bit byte unit.

Goal:
  For each activation, find the smallest hidden width H in the tested list
  where the 8 -> H -> 16 tied-mirror byte unit reaches 100% exact roundtrip
  under the pure 2-bit codebook.

This reuses the bounded warmup + static alpha + fixed-alpha QAT recipe from
diag_byte_unit_widen_sweep.py, but searches H in ascending order and stops at
the first exact hit per activation.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


THIS = Path(__file__).resolve()
BASE = THIS.with_name("diag_byte_unit_widen_sweep.py")
spec = importlib.util.spec_from_file_location("byte_widen", BASE)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hiddens", default="8,12,16,20,24,28,32,40,48,64")
    parser.add_argument("--activations", default="relu,leaky_relu,silu,softplus,tanh,identity,c19")
    parser.add_argument("--codebook", default="2bit_sym13")
    parser.add_argument("--float-epochs", type=int, default=150)
    parser.add_argument("--qat-epochs", type=int, default=150)
    parser.add_argument("--float-lr", type=float, default=2e-3)
    parser.add_argument("--qat-lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out", default="output/byte_unit_activation_min_sweep")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    hiddens = [int(s) for s in args.hiddens.split(",") if s.strip()]
    activations = [s.strip() for s in args.activations.split(",") if s.strip()]
    x = mod.build_dataset()
    x_np = x.detach().cpu().numpy()
    blob = mod.load_winner_blob()

    print("=" * 78)
    print("BYTE UNIT ACTIVATION MIN SWEEP — PURE 2-BIT")
    print("=" * 78)
    print(f"hiddens={hiddens}")
    print(f"activations={activations}")
    print(f"codebook={args.codebook}")

    all_results: dict[str, list[dict]] = {}
    first_exact: dict[str, dict | None] = {}

    for act in activations:
        print("\n" + "#" * 78)
        print(f"ACTIVATION {act}")
        print("#" * 78)
        rows: list[dict] = []
        winner = None
        for h in hiddens:
            ns = argparse.Namespace(
                seed=args.seed,
                float_epochs=args.float_epochs,
                float_lr=args.float_lr,
                qat_epochs=args.qat_epochs,
                qat_lr=args.qat_lr,
            )
            res = mod.run_one(h, act, args.codebook, x, x_np, ns, blob)
            row = res.__dict__
            rows.append(row)
            if res.final_lossless >= 100.0:
                winner = row
                break
        all_results[act] = rows
        first_exact[act] = winner

    ranked = []
    for act in activations:
        winner = first_exact[act]
        if winner is None:
            best = max(all_results[act], key=lambda r: (r["final_lossless"], r["final_per_bit"], -r["final_bad"]))
            ranked.append({
                "activation": act,
                "exact": False,
                "best_hidden": None,
                "best_final_lossless": best["final_lossless"],
                "best_final_bad": best["final_bad"],
                "best_final_per_bit": best["final_per_bit"],
            })
        else:
            ranked.append({
                "activation": act,
                "exact": True,
                "best_hidden": winner["hidden"],
                "best_final_lossless": winner["final_lossless"],
                "best_final_bad": winner["final_bad"],
                "best_final_per_bit": winner["final_per_bit"],
            })

    ranked.sort(key=lambda r: (0 if r["exact"] else 1, r["best_hidden"] if r["best_hidden"] is not None else 10**9, -r["best_final_lossless"]))

    summary = {
        "hiddens": hiddens,
        "activations": activations,
        "codebook": args.codebook,
        "float_epochs": args.float_epochs,
        "qat_epochs": args.qat_epochs,
        "first_exact": first_exact,
        "all_results": all_results,
        "ranked": ranked,
    }
    save_path = out_dir / "summary.json"
    save_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print("ACTIVATION RANKING")
    print("=" * 78)
    for i, row in enumerate(ranked, start=1):
        if row["exact"]:
            print(
                f"{i:2d}. act={row['activation']:<12s} exact=yes "
                f"min_H={row['best_hidden']:<3d} "
                f"bit={row['best_final_per_bit']:6.2f}%"
            )
        else:
            print(
                f"{i:2d}. act={row['activation']:<12s} exact=no  "
                f"best_ll={row['best_final_lossless']:6.2f}% "
                f"bad={row['best_final_bad']:3d} bit={row['best_final_per_bit']:6.2f}%"
            )
    print(f"\nSaved: {save_path}")


if __name__ == "__main__":
    main()
