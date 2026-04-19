"""Probe how far non-weight params of the 7-bit identity merger can be quantized.

Target model:
  - arch: single-W
  - activation: identity
  - hidden: 120
  - codebook: 7bit_int

We retrain the exact model for one or more seeds, then quantize only:
  - b1
  - b2
  - both biases
  - alpha (fp16 only)
  - all meta together (b1+b2+alpha)

Bias quantization is symmetric per-tensor signed integer quantization with zero
included. This is deliberately simple and deployment-realistic.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diag_byte_pair_merger_widen_sweep as base


def signed_levels(bits: int) -> int:
    return (1 << (bits - 1)) - 1


def quantize_bias_symmetric(x: torch.Tensor, bits: int) -> tuple[torch.Tensor, float]:
    max_abs = float(x.abs().max().item())
    if max_abs <= 0.0:
        return torch.zeros_like(x), 1.0
    if bits <= 1:
        scale = max_abs
        q = torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))
        return q * scale, scale
    qmax = signed_levels(bits)
    scale = max_abs / qmax
    q = torch.round(x / scale).clamp(-qmax, qmax)
    return q * scale, scale


def quantize_fp16(x: torch.Tensor) -> torch.Tensor:
    return x.detach().half().float()


def train_exact(seed: int) -> tuple[base.MergerSingleW, dict]:
    data = base.load_byte_pairs()
    torch.manual_seed(seed)
    model = base.MergerSingleW(32, 120, "identity", base.CODEBOOKS["7bit_int"]).to(base.DEVICE)
    base.train_adam(model, data, 1500, 2e-3, tag="float", print_every=200)
    model.use_codebook = False
    base.train_lbfgs(model, data, 150, patience=25, tag="float-lbfgs", print_every=10)
    model.use_codebook = True
    base.static_alpha_search(model, data, steps=50)
    base.train_adam(model, data, 800, 5e-4, tag="qat", print_every=200)
    final_m = base.train_lbfgs(model, data, 150, patience=25, tag="qat-lbfgs", print_every=10)
    return model, final_m


def metrics_for_probe(model: base.MergerSingleW) -> dict:
    data = base.load_byte_pairs()
    return base.metrics(model, data)


def run_seed(seed: int, bias_bits: list[int]) -> dict:
    model, baseline = train_exact(seed)
    out: dict[str, object] = {
        "seed": seed,
        "baseline": baseline,
        "alpha": float(model.alpha.item()),
        "bias_count": int(model.b1.numel() + model.b2.numel()),
        "probes": [],
    }

    # fp16 probes first
    for name in ("b1_fp16", "b2_fp16", "both_fp16", "alpha_fp16", "all_meta_fp16"):
        probe = copy.deepcopy(model)
        with torch.no_grad():
            if name in ("b1_fp16", "both_fp16", "all_meta_fp16"):
                probe.b1.copy_(quantize_fp16(probe.b1))
            if name in ("b2_fp16", "both_fp16", "all_meta_fp16"):
                probe.b2.copy_(quantize_fp16(probe.b2))
            if name in ("alpha_fp16", "all_meta_fp16"):
                probe.alpha_raw.copy_(quantize_fp16(probe.alpha_raw))
        m = metrics_for_probe(probe)
        out["probes"].append({
            "name": name,
            "kind": "fp16",
            "metrics": m,
        })

    # integer bias probes
    for bits in bias_bits:
        for target in ("b1", "b2", "both"):
            probe = copy.deepcopy(model)
            scales: dict[str, float] = {}
            with torch.no_grad():
                if target in ("b1", "both"):
                    q, s = quantize_bias_symmetric(probe.b1, bits)
                    probe.b1.copy_(q)
                    scales["b1"] = s
                if target in ("b2", "both"):
                    q, s = quantize_bias_symmetric(probe.b2, bits)
                    probe.b2.copy_(q)
                    scales["b2"] = s
            m = metrics_for_probe(probe)
            out["probes"].append({
                "name": f"{target}_int{bits}",
                "kind": "bias_int",
                "bits": bits,
                "target": target,
                "scales": scales,
                "metrics": m,
            })
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="42,7")
    parser.add_argument("--bias-bits", default="8,7,6,5,4")
    parser.add_argument("--out", default="output/merger_aux_quant_probe")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    bias_bits = [int(s.strip()) for s in args.bias_bits.split(",") if s.strip()]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for seed in seeds:
        print("=" * 78, flush=True)
        print(f"SEED {seed}", flush=True)
        print("=" * 78, flush=True)
        results.append(run_seed(seed, bias_bits))

    summary = {
        "target": {
            "arch": "single",
            "activation": "identity",
            "hidden": 120,
            "codebook": "7bit_int",
        },
        "seeds": seeds,
        "bias_bits": bias_bits,
        "results": results,
    }
    save = out_dir / "summary.json"
    save.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved: {save}", flush=True)


if __name__ == "__main__":
    main()
