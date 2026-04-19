"""Probe quantization sensitivity of non-W params on the exact float C19 merger.

Loads the exact single-W mirror C19 champion from final_model.json and measures
how much exactness is lost when quantizing:
  - b1 / b2
  - c19_c / c19_rho
  - grouped combinations of the above

This isolates auxiliary/meta parameter sensitivity without retraining.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = Path(__file__).resolve().parent
LUT_PATH = ROOT / "byte_embedder_lut_int8_nozero.json"


def load_byte_pairs() -> torch.Tensor:
    blob = json.loads(LUT_PATH.read_text(encoding="utf-8"))
    scale = blob["scale"]
    lut = torch.tensor(blob["lut"], dtype=torch.float32, device=DEVICE) * scale
    idx_a = torch.arange(256, device=DEVICE).unsqueeze(1).expand(256, 256).reshape(-1)
    idx_b = torch.arange(256, device=DEVICE).unsqueeze(0).expand(256, 256).reshape(-1)
    return torch.cat([lut[idx_a], lut[idx_b]], dim=1)


def c19(x: torch.Tensor, c: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
    c = c.clamp(min=0.1)
    rho = rho.clamp(min=0.0)
    L = 6.0 * c
    scaled = x / c
    n = scaled.floor()
    t = scaled - n
    h = t * (1.0 - t)
    sgn = torch.where(n.long() % 2 == 0, torch.ones_like(n), -torch.ones_like(n))
    interior = c * (sgn * h + rho * h * h)
    return torch.where(x >= L, x - L, torch.where(x <= -L, x + L, interior))


def forward(params: dict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    h = c19(x @ params["W"] + params["b1"], params["c19_c"], params["c19_rho"])
    return h @ params["W"].t() + params["b2"]


@torch.no_grad()
def metrics(params: dict[str, torch.Tensor], x: torch.Tensor) -> dict:
    y = forward(params, x)
    sign_match = torch.sign(y) == torch.sign(x)
    return {
        "lossless": float(sign_match.all(dim=1).float().mean().item() * 100.0),
        "per_dim": float(sign_match.float().mean().item() * 100.0),
        "bad_pairs": int((~sign_match.all(dim=1)).sum().item()),
    }


def quant_fp16(x: torch.Tensor) -> torch.Tensor:
    return x.detach().half().float()


def quant_signed(x: torch.Tensor, bits: int) -> tuple[torch.Tensor, float]:
    max_abs = float(x.abs().max().item())
    if max_abs <= 0.0:
        return torch.zeros_like(x), 1.0
    if bits <= 1:
        scale = max_abs
        q = torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))
        return q * scale, scale
    qmax = (1 << (bits - 1)) - 1
    scale = max_abs / qmax
    q = torch.round(x / scale).clamp(-qmax, qmax)
    return q * scale, scale


def quant_positive(x: torch.Tensor, bits: int) -> tuple[torch.Tensor, float]:
    qmax = (1 << bits) - 1
    xmax = float(x.max().item())
    if xmax <= 0.0:
        return torch.zeros_like(x), 1.0
    scale = xmax / qmax
    q = torch.round(x / scale).clamp(0, qmax)
    return q * scale, scale


def load_params(model_path: Path) -> dict[str, torch.Tensor]:
    blob = json.loads(model_path.read_text(encoding="utf-8"))
    return {
        "W": torch.tensor(blob["W"], dtype=torch.float32, device=DEVICE),
        "b1": torch.tensor(blob["b1"], dtype=torch.float32, device=DEVICE),
        "b2": torch.tensor(blob["b2"], dtype=torch.float32, device=DEVICE),
        "c19_c": torch.tensor(blob["c19_c"], dtype=torch.float32, device=DEVICE),
        "c19_rho": torch.tensor(blob["c19_rho"], dtype=torch.float32, device=DEVICE),
    }


def clone_params(params: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.clone() for k, v in params.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="S:/Git/VRAXION/output/merger_single_w_exhaustive_fix/final_model.json",
    )
    parser.add_argument("--bits", default="8,7,6,5,4")
    parser.add_argument("--out", default="S:/Git/VRAXION/output/merger_float_aux_quant_probe")
    args = parser.parse_args()

    bits_list = [int(s.strip()) for s in args.bits.split(",") if s.strip()]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    params = load_params(Path(args.model))
    data = load_byte_pairs()
    baseline = metrics(params, data)

    probes: list[dict] = []

    # fp16 probes
    fp16_groups = {
        "b1_fp16": ("b1",),
        "b2_fp16": ("b2",),
        "biases_fp16": ("b1", "b2"),
        "c_fp16": ("c19_c",),
        "rho_fp16": ("c19_rho",),
        "c19_fp16": ("c19_c", "c19_rho"),
        "all_aux_fp16": ("b1", "b2", "c19_c", "c19_rho"),
    }
    for name, keys in fp16_groups.items():
        probe = clone_params(params)
        for key in keys:
            probe[key] = quant_fp16(probe[key])
        probes.append({"name": name, "kind": "fp16", "metrics": metrics(probe, data)})

    # fixed-bit probes
    int_groups = {
        "b1": ("b1",),
        "b2": ("b2",),
        "biases": ("b1", "b2"),
        "c": ("c19_c",),
        "rho": ("c19_rho",),
        "c19": ("c19_c", "c19_rho"),
        "all_aux": ("b1", "b2", "c19_c", "c19_rho"),
    }
    for bits in bits_list:
        for name, keys in int_groups.items():
            probe = clone_params(params)
            scales: dict[str, float] = {}
            for key in keys:
                if key in ("b1", "b2"):
                    probe[key], scales[key] = quant_signed(probe[key], bits)
                else:
                    probe[key], scales[key] = quant_positive(probe[key], bits)
            probes.append({
                "name": f"{name}_int{bits}",
                "kind": "int",
                "bits": bits,
                "scales": scales,
                "metrics": metrics(probe, data),
            })

    summary = {
        "model": str(Path(args.model)),
        "baseline": baseline,
        "bits": bits_list,
        "probes": probes,
    }
    save = out_dir / "summary.json"
    save.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved: {save}", flush=True)


if __name__ == "__main__":
    main()
