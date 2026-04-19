"""L1 merger minimizer: search the smallest native exact block.

Wraps diag_byte_pair_merger_widen_sweep.py's training recipe, but ranks
configs by estimated deploy bytes instead of raw weight count.

Important distinctions from the base sweep:
  - aggregates across multiple seeds
  - includes bias / alpha / activation-meta bytes in the estimate
  - reports both "any-seed exact" and "all-seed exact" rankings

This is still a screening tool: it estimates storage from the trained model
shape and the chosen codebook, but does not emit a packed deploy artifact.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import diag_byte_pair_merger_widen_sweep as base


def parse_csv_ints(spec: str) -> list[int]:
    return [int(s.strip()) for s in spec.split(",") if s.strip()]


def parse_csv_strs(spec: str) -> list[str]:
    return [s.strip() for s in spec.split(",") if s.strip()]


def codebook_bits(name: str) -> int:
    levels = base.CODEBOOKS[name]
    if levels is None:
        return 32
    return max(1, math.ceil(math.log2(len(levels))))


def bytes_per_scalar(fmt: str) -> int:
    if fmt == "fp16":
        return 2
    if fmt == "fp32":
        return 4
    raise ValueError(f"unsupported scalar format: {fmt}")


def estimate_bytes(result: dict, arch: str, activation: str, codebook_name: str, in_dim: int, bias_format: str, meta_format: str) -> dict:
    w_bits = result["weights_count"] * codebook_bits(codebook_name)
    bias_count = result["hidden"] + in_dim
    alpha_count = 0 if codebook_name == "float" else (2 if arch == "dual" else 1)
    act_param_count = (2 * result["hidden"]) if activation == "c19" else 0
    bias_bytes = bias_count * bytes_per_scalar(bias_format)
    alpha_bytes = alpha_count * bytes_per_scalar(meta_format)
    act_param_bytes = act_param_count * bytes_per_scalar(meta_format)
    total_bytes = (w_bits / 8.0) + bias_bytes + alpha_bytes + act_param_bytes
    return {
        "weight_bits": w_bits,
        "weight_bytes": w_bits / 8.0,
        "bias_bytes": bias_bytes,
        "alpha_bytes": alpha_bytes,
        "act_param_bytes": act_param_bytes,
        "total_bytes": total_bytes,
        "total_kb": total_bytes / 1024.0,
    }


def make_base_args(cli_args: argparse.Namespace, arch: str, seed: int) -> SimpleNamespace:
    return SimpleNamespace(
        arch=arch,
        seed=seed,
        float_epochs=cli_args.float_epochs,
        float_lr=cli_args.float_lr,
        qat_epochs=cli_args.qat_epochs,
        qat_lr=cli_args.qat_lr,
        alpha_steps=cli_args.alpha_steps,
        lbfgs_outer=cli_args.lbfgs_outer,
        lbfgs_patience=cli_args.lbfgs_patience,
    )


def config_key(row: dict) -> tuple:
    return (row["arch"], row["hidden"], row["activation"], row["codebook_name"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dim", type=int, default=32)
    parser.add_argument("--arches", default="single")
    parser.add_argument("--hiddens", default="81,120")
    parser.add_argument("--activations", default="identity,relu,leaky_relu,tanh,c19")
    parser.add_argument("--codebooks", default="binary,ternary,2bit_sym13,3bit_sym1248,4bit_int,5bit_int,6bit_int,7bit_int")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--float-epochs", type=int, default=1500)
    parser.add_argument("--float-lr", type=float, default=2e-3)
    parser.add_argument("--qat-epochs", type=int, default=800)
    parser.add_argument("--qat-lr", type=float, default=5e-4)
    parser.add_argument("--alpha-steps", type=int, default=50)
    parser.add_argument("--lbfgs-outer", type=int, default=150)
    parser.add_argument("--lbfgs-patience", type=int, default=25)
    parser.add_argument("--bias-format", choices=["fp16", "fp32"], default="fp32")
    parser.add_argument("--meta-format", choices=["fp16", "fp32"], default="fp32")
    parser.add_argument("--out", default="output/merger_minimize")
    args = parser.parse_args()

    arches = parse_csv_strs(args.arches)
    hiddens = parse_csv_ints(args.hiddens)
    activations = parse_csv_strs(args.activations)
    codebooks = parse_csv_strs(args.codebooks)
    seeds = parse_csv_ints(args.seeds)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78, flush=True)
    print("L1 MERGER MINIMIZER", flush=True)
    print("=" * 78, flush=True)
    print(f"arches={arches}", flush=True)
    print(f"hiddens={hiddens}", flush=True)
    print(f"activations={activations}", flush=True)
    print(f"codebooks={codebooks}", flush=True)
    print(f"seeds={seeds}", flush=True)
    print(f"bias_format={args.bias_format} meta_format={args.meta_format}", flush=True)

    data = base.load_byte_pairs()
    print(f"Loaded {data.shape[0]} byte-pair vectors, in_dim={data.shape[1]}", flush=True)

    runs: list[dict] = []
    grouped: dict[tuple, list[dict]] = defaultdict(list)

    for arch in arches:
        for seed in seeds:
            base_args = make_base_args(args, arch, seed)
            for H in hiddens:
                for act in activations:
                    for cb in codebooks:
                        row = base.run_one(args.in_dim, H, act, cb, data, base_args)
                        row["arch"] = arch
                        row["seed"] = seed
                        row["estimate"] = estimate_bytes(
                            row,
                            arch=arch,
                            activation=act,
                            codebook_name=cb,
                            in_dim=args.in_dim,
                            bias_format=args.bias_format,
                            meta_format=args.meta_format,
                        )
                        runs.append(row)
                        grouped[config_key(row)].append(row)

    configs: list[dict] = []
    for key, rows in grouped.items():
        sample = rows[0]
        estimates = sample["estimate"]
        best = max(rows, key=lambda r: (r["final_lossless"], -r["final_bad"], r["final_per_dim"]))
        worst = min(rows, key=lambda r: (r["final_lossless"], -r["final_per_dim"], r["final_bad"]))
        exact_hits = sum(1 for r in rows if r["final_bad"] == 0)
        cfg = {
            "arch": sample["arch"],
            "hidden": sample["hidden"],
            "activation": sample["activation"],
            "codebook_name": sample["codebook_name"],
            "runs": rows,
            "seed_count": len(rows),
            "exact_hits": exact_hits,
            "exact_hit_rate": exact_hits / len(rows),
            "best_lossless": best["final_lossless"],
            "best_bad": best["final_bad"],
            "worst_lossless": worst["final_lossless"],
            "worst_bad": worst["final_bad"],
            "estimate": estimates,
        }
        configs.append(cfg)

    ranked_any_seed_exact = sorted(
        [c for c in configs if c["exact_hits"] > 0],
        key=lambda c: (
            c["estimate"]["total_bytes"],
            -c["exact_hit_rate"],
            c["hidden"],
            c["activation"],
            c["codebook_name"],
        ),
    )
    ranked_all_seed_exact = sorted(
        [c for c in configs if c["exact_hits"] == c["seed_count"]],
        key=lambda c: (
            c["estimate"]["total_bytes"],
            c["hidden"],
            c["activation"],
            c["codebook_name"],
        ),
    )
    ranked_best = sorted(
        configs,
        key=lambda c: (
            -c["best_lossless"],
            c["best_bad"],
            c["estimate"]["total_bytes"],
            -c["exact_hit_rate"],
        ),
    )

    summary = {
        "in_dim": args.in_dim,
        "arches": arches,
        "hiddens": hiddens,
        "activations": activations,
        "codebooks": codebooks,
        "seeds": seeds,
        "float_epochs": args.float_epochs,
        "qat_epochs": args.qat_epochs,
        "bias_format": args.bias_format,
        "meta_format": args.meta_format,
        "runs": runs,
        "configs": configs,
        "ranked_any_seed_exact": ranked_any_seed_exact,
        "ranked_all_seed_exact": ranked_all_seed_exact,
        "ranked_best": ranked_best,
    }
    save = out_dir / "summary.json"
    save.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 78, flush=True)
    print("ANY-SEED EXACT (ranked by estimated bytes)", flush=True)
    print("=" * 78, flush=True)
    if not ranked_any_seed_exact:
        print("No exact configurations found.", flush=True)
    for i, cfg in enumerate(ranked_any_seed_exact[:20], start=1):
        est = cfg["estimate"]
        print(
            f"{i:2d}. arch={cfg['arch']:<6s} H={cfg['hidden']:3d} "
            f"act={cfg['activation']:<10s} cb={cfg['codebook_name']:<12s} "
            f"bytes={est['total_bytes']:.1f} ({est['total_kb']:.3f} KB) "
            f"hits={cfg['exact_hits']}/{cfg['seed_count']} "
            f"best={cfg['best_lossless']:.6f}% bad={cfg['best_bad']}",
            flush=True,
        )

    print("\n" + "=" * 78, flush=True)
    print("ALL-SEED EXACT (ranked by estimated bytes)", flush=True)
    print("=" * 78, flush=True)
    if not ranked_all_seed_exact:
        print("No all-seed exact configurations found.", flush=True)
    for i, cfg in enumerate(ranked_all_seed_exact[:20], start=1):
        est = cfg["estimate"]
        print(
            f"{i:2d}. arch={cfg['arch']:<6s} H={cfg['hidden']:3d} "
            f"act={cfg['activation']:<10s} cb={cfg['codebook_name']:<12s} "
            f"bytes={est['total_bytes']:.1f} ({est['total_kb']:.3f} KB) "
            f"hits={cfg['exact_hits']}/{cfg['seed_count']}",
            flush=True,
        )

    print(f"\nSaved: {save}", flush=True)


if __name__ == "__main__":
    main()
