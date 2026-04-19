"""L1 Merger bake-and-measure probe.

Given a trained float-100% merger (from widen_sweep), quantize its W matrices
to a codebook at varying alpha scales and measure the resulting lossless%.

Purpose: decide whether the problem is REPRESENTABLE in the target codebook
at all. If the best alpha gives <50%, no amount of QAT or LBFGS polish can
rescue it — the codebook is too coarse. If it gives >80%, then finetuning
can probably close the gap.

Usage:
  python tools/diag_byte_pair_merger_bake_probe.py --arch single --H 81 \\
    --activation identity --codebook binary --alpha-scan 200
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

# Reuse model classes + utilities from the widen sweep
import importlib.util, sys
THIS = Path(__file__).resolve()
BASE = THIS.with_name("diag_byte_pair_merger_widen_sweep.py")
spec = importlib.util.spec_from_file_location("widen_sweep", BASE)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def train_to_100(model, data, max_adam=1000, max_lbfgs=100):
    """Train the model until lossless >= 100 or plateau."""
    mod.train_adam(model, data, max_adam, 2e-3, tag="adam")
    m = mod.metrics(model, data)
    if m["lossless"] < 100.0:
        mod.train_lbfgs(model, data, max_lbfgs, patience=20, tag="lbfgs")
    return mod.metrics(model, data)


def bake_measure(model, data, codebook, alpha):
    """Temporarily snap W (and W2 if dual) at the given alpha, measure lossless."""
    levels = torch.tensor(codebook, dtype=torch.float32, device=mod.DEVICE) * alpha
    with torch.no_grad():
        if hasattr(model, "W1"):
            W1_orig = model.W1.detach().clone()
            W2_orig = model.W2.detach().clone()
            W1_q = mod.ste_codebook(model.W1.detach(), levels)
            W2_q = mod.ste_codebook(model.W2.detach(), levels)
            model.W1.copy_(W1_q)
            model.W2.copy_(W2_q)
            m = mod.metrics(model, data)
            model.W1.copy_(W1_orig)
            model.W2.copy_(W2_orig)
        else:
            W_orig = model.W.detach().clone()
            W_q = mod.ste_codebook(model.W.detach(), levels)
            model.W.copy_(W_q)
            m = mod.metrics(model, data)
            model.W.copy_(W_orig)
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=["single", "dual"], default="single")
    parser.add_argument("--H", type=int, default=81)
    parser.add_argument("--in-dim", type=int, default=32)
    parser.add_argument("--activation", default="identity")
    parser.add_argument("--codebook", default="binary")
    parser.add_argument("--alpha-scan", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--float-epochs", type=int, default=1000)
    parser.add_argument("--lbfgs-outer", type=int, default=80)
    parser.add_argument("--out", default="output/byte_pair_merger_bake_probe")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    codebook_tuple = mod.CODEBOOKS.get(args.codebook)
    if codebook_tuple is None:
        raise SystemExit(f"unknown codebook: {args.codebook}")

    print(f"=== L1 MERGER BAKE PROBE ===")
    print(f"arch={args.arch} H={args.H} act={args.activation} cb={args.codebook}")
    print(f"seed={args.seed} alpha_scan={args.alpha_scan}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data = mod.load_byte_pairs()

    if args.arch == "dual":
        model = mod.MergerDualW(args.in_dim, args.H, args.activation, None).to(mod.DEVICE)
    else:
        model = mod.MergerSingleW(args.in_dim, args.H, args.activation, None).to(mod.DEVICE)

    print("\n[phase 1] train to float 100%")
    m = train_to_100(model, data, max_adam=args.float_epochs, max_lbfgs=args.lbfgs_outer)
    print(f"  float: ll={m['lossless']:.2f}% bad={m['bad_pairs']} pd={m['per_dim']:.2f}%")
    if m["lossless"] < 95.0:
        print("  [warn] float did not reach 95%+ — bake will be meaningless")

    with torch.no_grad():
        if hasattr(model, "W1"):
            W_abs = torch.cat([model.W1.abs().flatten(), model.W2.abs().flatten()])
        else:
            W_abs = model.W.abs().flatten()
        lo = float(W_abs.min().item()) + 1e-6
        hi = float(W_abs.max().item()) + 1e-3

    print(f"\n[phase 2] alpha scan over [{lo:.5f}, {hi:.5f}]")
    alphas = np.linspace(lo, hi, args.alpha_scan)
    rows = []
    best_ll = -1.0
    best_alpha = None
    for i, a in enumerate(alphas):
        bm = bake_measure(model, data, codebook_tuple, float(a))
        rows.append({"alpha": float(a), "lossless": bm["lossless"], "bad": bm["bad_pairs"], "per_dim": bm["per_dim"]})
        if bm["lossless"] > best_ll:
            best_ll = bm["lossless"]
            best_alpha = float(a)
        if (i + 1) % 25 == 0 or i == 0:
            print(f"  [{i+1:4d}/{args.alpha_scan}] alpha={a:.5f} ll={bm['lossless']:.2f}% bad={bm['bad_pairs']:5d}")

    print(f"\n[result] best bake: alpha={best_alpha:.5f} ll={best_ll:.2f}%")
    if best_ll >= 95:
        print("  [verdict] codebook is EXPRESSIVE ENOUGH; QAT should work with this alpha init.")
    elif best_ll >= 50:
        print("  [verdict] codebook is MARGINAL; QAT may close the gap with more polish.")
    else:
        print("  [verdict] codebook is TOO COARSE; finetuning alone won't rescue it.")

    summary = {
        "arch": args.arch,
        "H": args.H,
        "in_dim": args.in_dim,
        "activation": args.activation,
        "codebook": args.codebook,
        "seed": args.seed,
        "float_lossless": m["lossless"],
        "best_bake_alpha": best_alpha,
        "best_bake_lossless": best_ll,
        "alpha_rows": rows,
    }
    save = out_dir / "summary.json"
    save.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved: {save}")


if __name__ == "__main__":
    main()
