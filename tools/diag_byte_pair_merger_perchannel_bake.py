"""L1 Merger per-channel bake probe.

Unlike the global-alpha bake (where one alpha scales all of W), here EACH column
of W gets its own optimal alpha. The quantized cell becomes sign(W[i,j]) * alpha[j].

If per-channel binary gives >50% bake, then QAT+LBFGS can likely push to 100% —
giving a merger at ~1 KB footprint with binary weights + 120 float alpha scales.

Usage:
  python tools/diag_byte_pair_merger_perchannel_bake.py --H 120 --codebook binary
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch

import importlib.util, sys
THIS = Path(__file__).resolve()
BASE = THIS.with_name("diag_byte_pair_merger_widen_sweep.py")
spec = importlib.util.spec_from_file_location("widen_sweep", BASE)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def bake_perchannel(model, data, codebook, n_alpha_per_col: int = 40):
    """For each column of W, greedy-search the best alpha (and best alpha2 in dual).
    Quantize to sign(w) * alpha_col for binary, or nearest codebook level * alpha_col."""
    levels = torch.tensor(codebook, dtype=torch.float32, device=mod.DEVICE)

    with torch.no_grad():
        if hasattr(model, "W1"):
            W1_orig = model.W1.detach().clone()
            W2_orig = model.W2.detach().clone()
            Ws = [model.W1, model.W2]
            W_origs = [W1_orig, W2_orig]
        else:
            W_orig = model.W.detach().clone()
            Ws = [model.W]
            W_origs = [W_orig]

        # Per-column alpha search: for each column, try a grid of alphas and
        # keep the one that minimizes (W_col - quant(W_col, alpha))^2.
        for W_param, W_o in zip(Ws, W_origs):
            n_cols = W_o.shape[1]
            for j in range(n_cols):
                col = W_o[:, j]
                col_abs = col.abs()
                lo = float(col_abs.min().item()) + 1e-8
                hi = float(col_abs.max().item()) + 1e-4
                alphas = torch.linspace(lo, hi, n_alpha_per_col, device=mod.DEVICE)
                best_err = float("inf")
                best_q = None
                for a in alphas:
                    q = mod.ste_codebook(col, levels * a)
                    err = float(((col - q) ** 2).sum().item())
                    if err < best_err:
                        best_err = err
                        best_q = q
                with torch.no_grad():
                    W_param[:, j] = best_q

        m = mod.metrics(model, data)

        # Restore
        for W_param, W_o in zip(Ws, W_origs):
            W_param.copy_(W_o)
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=["single", "dual"], default="single")
    parser.add_argument("--H", type=int, default=120)
    parser.add_argument("--in-dim", type=int, default=32)
    parser.add_argument("--activation", default="identity")
    parser.add_argument("--codebook", default="binary")
    parser.add_argument("--n-alpha-per-col", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--float-epochs", type=int, default=1500)
    parser.add_argument("--lbfgs-outer", type=int, default=120)
    parser.add_argument("--out", default="output/byte_pair_merger_perchannel_bake")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    codebook_tuple = mod.CODEBOOKS.get(args.codebook)
    if codebook_tuple is None:
        raise SystemExit(f"unknown codebook: {args.codebook}")

    print(f"=== L1 MERGER PER-CHANNEL BAKE ===")
    print(f"arch={args.arch} H={args.H} act={args.activation} cb={args.codebook}")
    print(f"seed={args.seed} n_alpha_per_col={args.n_alpha_per_col}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data = mod.load_byte_pairs()

    if args.arch == "dual":
        model = mod.MergerDualW(args.in_dim, args.H, args.activation, None).to(mod.DEVICE)
    else:
        model = mod.MergerSingleW(args.in_dim, args.H, args.activation, None).to(mod.DEVICE)

    print("\n[phase 1] train to float 100%")
    mod.train_adam(model, data, args.float_epochs, 2e-3, tag="adam")
    m = mod.metrics(model, data)
    if m["lossless"] < 100.0:
        mod.train_lbfgs(model, data, args.lbfgs_outer, patience=25, tag="lbfgs")
        m = mod.metrics(model, data)
    print(f"  float: ll={m['lossless']:.2f}% bad={m['bad_pairs']} pd={m['per_dim']:.2f}%")

    print(f"\n[phase 2] per-channel bake (greedy per-column alpha search)")
    bm = bake_perchannel(model, data, codebook_tuple, args.n_alpha_per_col)
    print(f"  [result] per-channel bake: ll={bm['lossless']:.2f}% bad={bm['bad_pairs']} pd={bm['per_dim']:.2f}%")

    if bm["lossless"] >= 95:
        print("  [verdict] EXPRESSIVE ENOUGH — QAT should push to 100%")
    elif bm["lossless"] >= 50:
        print("  [verdict] MARGINAL — QAT may close the gap")
    elif bm["lossless"] >= 10:
        print("  [verdict] WEAK but better than global alpha — per-column helps")
    else:
        print("  [verdict] STILL TOO COARSE — per-channel does not help binary")

    summary = {
        "arch": args.arch, "H": args.H, "in_dim": args.in_dim,
        "activation": args.activation, "codebook": args.codebook, "seed": args.seed,
        "float_lossless": m["lossless"],
        "perchannel_bake_lossless": bm["lossless"],
        "perchannel_bake_bad": bm["bad_pairs"],
        "perchannel_bake_per_dim": bm["per_dim"],
    }
    save = out_dir / "summary.json"
    save.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved: {save}")


if __name__ == "__main__":
    main()
