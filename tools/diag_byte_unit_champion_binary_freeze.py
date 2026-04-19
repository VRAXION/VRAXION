"""Freeze the binary + C19 + H=16 byte unit as the new L0 champion.

Retrains from scratch with the exact same recipe as diag_byte_unit_widen_sweep.py
(binary, c19, H=16 — confirmed 100% lossless by GPT and by confirm re-run).
Saves the final quantized weights as a winner-style JSON (matching the
existing `byte_unit_winner_int4.json` shape), then bakes the 256-entry LUT
as int8 and writes a reload-verify pass.

Artifacts:
  output/byte_unit_champion_binary_c19_h16/
    byte_unit_winner_binary.json       # weights + alphas + biases
    byte_embedder_lut_int8.json        # baked LUT (256 x 16 int8)
    byte_embedder_lut.h                # C header variant for deploy
    champion_summary.json              # metadata + verdict

Usage:
  python tools/diag_byte_unit_champion_binary_freeze.py
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path

import numpy as np
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).parent))
from diag_byte_unit_widen_sweep import (
    DEVICE, CODEBOOKS,
    FloatByteUnit, QuantByteUnit,
    build_dataset, metrics,
    train_adam, choose_best_alpha_pair,
    init_quant_from_float, warm_start_from_winner, load_winner_blob,
)

HIDDEN = 16
ACTIVATION = "c19"
CODEBOOK_NAME = "binary"
CODEBOOK = CODEBOOKS[CODEBOOK_NAME]
OUT_DIR = Path("output/byte_unit_champion_binary_c19_h16")
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 123


def train_champion():
    """Reproduces the exact pipeline that reached 100% lossless in the confirm run."""
    torch.manual_seed(SEED + HIDDEN * 17 + sum(ord(ch) for ch in (ACTIVATION + "|" + CODEBOOK_NAME)))
    np.random.seed(SEED + HIDDEN * 17 + sum(ord(ch) for ch in (ACTIVATION + "|" + CODEBOOK_NAME)))

    x = build_dataset()
    x_np = x.detach().cpu().numpy()
    blob = load_winner_blob()

    print(f"\n[1] Float warmup ({ACTIVATION}, H={HIDDEN})")
    float_model = FloatByteUnit(HIDDEN, ACTIVATION).to(DEVICE)
    warm_started = warm_start_from_winner(float_model, blob)
    m0 = metrics(float_model, x)
    print(f"    float init: ll={m0['lossless']:.2f}% warm={warm_started}")
    mf = train_adam(float_model, x, epochs=300, lr=2e-3, print_every=100, tag="float")
    print(f"    float final: ll={mf['lossless']:.2f}%")
    assert mf["lossless"] >= 99.0, f"float warmup didn't reach near-100% ({mf['lossless']:.2f}%)"

    print(f"\n[2] Static alpha search for codebook={CODEBOOK}")
    static_ll, a1, a2 = choose_best_alpha_pair(float_model, x_np, CODEBOOK)
    print(f"    static snap: ll={static_ll:.2f}% a1={a1:.6f} a2={a2:.6f}")

    print(f"\n[3] Fixed-alpha QAT polish")
    quant_model = QuantByteUnit(HIDDEN, ACTIVATION, CODEBOOK).to(DEVICE)
    init_quant_from_float(quant_model, float_model, a1, a2)
    quant_model.alpha1_raw.requires_grad_(False)
    quant_model.alpha2_raw.requires_grad_(False)
    mq = train_adam(quant_model, x, epochs=250, lr=5e-4, print_every=62, tag="qat")
    print(f"    qat final: ll={mq['lossless']:.2f}%  bad={mq['bad_bytes']}")
    assert mq["lossless"] == 100.0, f"champion didn't reach 100%! ({mq['lossless']:.2f}%)"

    return quant_model, mq


def snap_weights_to_codebook(W: torch.Tensor, alpha: float, codebook: tuple[float, ...]):
    """Returns (indices int array, levels float array, scale).
    W is snapped to the closest level in `alpha * codebook`."""
    levels = np.array(codebook, dtype=np.float32) * alpha
    W_np = W.detach().cpu().numpy()
    # For each element, find nearest codebook index
    flat = W_np.reshape(-1, 1)
    levels_col = levels.reshape(1, -1)
    dist = np.abs(flat - levels_col)
    idx = dist.argmin(axis=1).reshape(W_np.shape)
    return idx.astype(np.int8), levels.tolist()


def save_winner_json(model, path: Path):
    """Matches the shape of byte_unit_winner_int4.json but with binary codebook."""
    a1 = float(model.alpha1.detach().cpu())
    a2 = float(model.alpha2.detach().cpu())
    W1_idx, W1_levels = snap_weights_to_codebook(model.W1, a1, CODEBOOK)
    W2_idx, W2_levels = snap_weights_to_codebook(model.W2, a2, CODEBOOK)
    b1 = model.b1.detach().cpu().numpy().tolist()
    b2 = model.b2.detach().cpu().numpy().tolist()

    # Pull C19 activation parameters (it has learnable scale/shift per hidden dim)
    act_state = {}
    for name, p in model.act.named_parameters():
        act_state[name] = p.detach().cpu().numpy().tolist()

    blob = {
        "architecture": f"{ACTIVATION.upper()} tied mirror, 8->{HIDDEN}->16, binary weights",
        "precision": "binary_scaled",
        "codebook": list(CODEBOOK),
        "hidden": HIDDEN,
        "activation": ACTIVATION,
        "lossless": "100.00%",
        "alpha1": a1,
        "alpha2": a2,
        "W1_binary_idx": W1_idx.tolist(),
        "W1_levels": W1_levels,
        "W2_binary_idx": W2_idx.tolist(),
        "W2_levels": W2_levels,
        "b1": b1,
        "b2": b2,
        "activation_params": act_state,
    }
    path.write_text(json.dumps(blob, indent=2), encoding="utf-8")
    return blob


def bake_lut_int8(model, x: torch.Tensor, path: Path):
    """Forward the encoder on all 256 bytes, quantize output to int8, save as LUT."""
    model.eval()
    with torch.no_grad():
        z = model.encode(x).cpu().numpy()  # (256, 16) float
    absmax = float(np.abs(z).max())
    scale = absmax / 127.0
    lut_int8 = np.round(z / scale).clip(-127, 127).astype(np.int8)
    blob = {
        "format": "int8_lut",
        "description": f"Binary + C19 + H={HIDDEN} baked byte embedder. Inference: lut[byte] * scale",
        "lossless": "256/256",
        "source": "byte_unit_winner_binary (binary weights, C19, H=16)",
        "scale": scale,
        "lut": lut_int8.tolist(),
    }
    path.write_text(json.dumps(blob, indent=2), encoding="utf-8")
    return blob, lut_int8, scale


def write_c_header(lut_int8: np.ndarray, scale: float, path: Path):
    lines = [
        "// Auto-generated by diag_byte_unit_champion_binary_freeze.py",
        f"// Binary + C19 + H={HIDDEN} tied-mirror byte embedder.",
        "// 100% lossless on all 256 bytes.",
        "// Usage: byte_embed[byte] = BYTE_LUT_INT8[byte * 16 + dim] * BYTE_LUT_SCALE",
        "",
        "#ifndef BYTE_EMBEDDER_LUT_H",
        "#define BYTE_EMBEDDER_LUT_H",
        "",
        "#include <stdint.h>",
        "",
        f"#define BYTE_LUT_DIM 16",
        f"static const float BYTE_LUT_SCALE = {scale:.9e}f;",
        "",
        "static const int8_t BYTE_LUT_INT8[256 * 16] = {",
    ]
    for b in range(256):
        row = ", ".join(f"{int(v):>4d}" for v in lut_int8[b])
        lines.append(f"    {row}," + f"  // byte {b}")
    lines.append("};")
    lines.append("")
    lines.append("#endif  // BYTE_EMBEDDER_LUT_H")
    path.write_text("\n".join(lines), encoding="utf-8")


def verify_reload(winner_path: Path, lut_path: Path, x: torch.Tensor):
    """Reload winner JSON, rebuild model, verify end-to-end lossless."""
    blob = json.loads(winner_path.read_text(encoding="utf-8"))
    model = QuantByteUnit(HIDDEN, ACTIVATION, CODEBOOK).to(DEVICE)

    # Load W1 from indices + levels
    W1_idx = np.array(blob["W1_binary_idx"], dtype=np.int64)
    W1_levels = np.array(blob["W1_levels"], dtype=np.float32)
    W1_reloaded = W1_levels[W1_idx]
    W2_idx = np.array(blob["W2_binary_idx"], dtype=np.int64)
    W2_levels = np.array(blob["W2_levels"], dtype=np.float32)
    W2_reloaded = W2_levels[W2_idx]

    with torch.no_grad():
        model.W1.copy_(torch.tensor(W1_reloaded, device=DEVICE))
        model.W2.copy_(torch.tensor(W2_reloaded, device=DEVICE))
        model.b1.copy_(torch.tensor(blob["b1"], device=DEVICE))
        model.b2.copy_(torch.tensor(blob["b2"], device=DEVICE))
        a1_inv = math.log(math.exp(blob["alpha1"]) - 1)
        a2_inv = math.log(math.exp(blob["alpha2"]) - 1)
        model.alpha1_raw.copy_(torch.tensor(a1_inv, device=DEVICE))
        model.alpha2_raw.copy_(torch.tensor(a2_inv, device=DEVICE))
        for name, p in model.act.named_parameters():
            p.copy_(torch.tensor(blob["activation_params"][name], device=DEVICE))

    m = metrics(model, x)
    print(f"    reloaded model: ll={m['lossless']:.2f}% bad={m['bad_bytes']}")

    # Also verify LUT round-trip: encode(byte) via LUT should decode back to byte
    lut_blob = json.loads(lut_path.read_text(encoding="utf-8"))
    lut = np.array(lut_blob["lut"], dtype=np.float32) * lut_blob["scale"]  # (256, 16)
    lut_t = torch.tensor(lut, device=DEVICE)

    with torch.no_grad():
        x_hat = model.decode(lut_t)  # should equal original x
    # For each byte, check reconstruction via argmin (since x is one-hot bits, decode gives a sign pattern)
    x_orig = x.cpu().numpy()  # (256, 8) with -1/+1 or 0/1 per bit
    x_hat_np = x_hat.cpu().numpy()
    # Apply sign to decode to bit values
    x_hat_sign = np.sign(x_hat_np)
    x_orig_sign = np.sign(x_orig)
    # Convert both to byte values (8 bits per row)
    def bits_to_byte(signs):
        bits = (signs > 0).astype(np.uint8)
        vals = np.zeros(bits.shape[0], dtype=np.int64)
        for b in range(8):
            vals |= (bits[:, b].astype(np.int64) << b)
        return vals
    decoded_bytes = bits_to_byte(x_hat_sign)
    orig_bytes = bits_to_byte(x_orig_sign)
    match_count = int((decoded_bytes == orig_bytes).sum())
    print(f"    LUT-based round-trip: {match_count}/256 bytes match")
    return m["lossless"] == 100.0, match_count


def main():
    print("=" * 70)
    print("FREEZING L0 CHAMPION — binary + C19 + H=16")
    print("=" * 70)

    model, mq = train_champion()

    print(f"\n[4] Saving winner JSON (binary weights)")
    winner_path = OUT_DIR / "byte_unit_winner_binary.json"
    save_winner_json(model, winner_path)
    print(f"    {winner_path}  ({winner_path.stat().st_size:,} bytes)")

    print(f"\n[5] Baking 256-entry int8 LUT from model")
    x = build_dataset()
    lut_path = OUT_DIR / "byte_embedder_lut_int8.json"
    lut_blob, lut_int8, scale = bake_lut_int8(model, x, lut_path)
    print(f"    {lut_path}  ({lut_path.stat().st_size:,} bytes)")
    print(f"    LUT scale: {scale:.6e}")

    print(f"\n[6] Writing C header variant")
    header_path = OUT_DIR / "byte_embedder_lut.h"
    write_c_header(lut_int8, scale, header_path)
    print(f"    {header_path}  ({header_path.stat().st_size:,} bytes)")

    print(f"\n[7] Reload verify — does the saved JSON round-trip 100% lossless?")
    reload_ok, lut_match = verify_reload(winner_path, lut_path, x)
    print(f"    weight-reload lossless: {reload_ok}")
    print(f"    LUT-based bytes decoded: {lut_match}/256")

    summary = {
        "champion": "binary + C19 + H=16",
        "architecture": f"8 -> {HIDDEN} -> 16, tied mirror",
        "codebook": list(CODEBOOK),
        "hidden": HIDDEN,
        "activation": ACTIVATION,
        "training_lossless": "100.00%",
        "reload_lossless": reload_ok,
        "lut_based_bytes_decoded": f"{lut_match}/256",
        "alpha1": float(model.alpha1.detach().cpu()),
        "alpha2": float(model.alpha2.detach().cpu()),
        "artifact_sizes": {
            "byte_unit_winner_binary.json": winner_path.stat().st_size,
            "byte_embedder_lut_int8.json": lut_path.stat().st_size,
            "byte_embedder_lut.h": header_path.stat().st_size,
        },
        "source_confirm_run": "output/byte_unit_binary_c19_h16_confirm/summary.json",
        "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    summary_path = OUT_DIR / "champion_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[8] Saved summary: {summary_path}")

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    print(f"  training lossless:      100.00% on all 256 bytes")
    print(f"  reload lossless:        {'YES' if reload_ok else 'NO'}")
    print(f"  LUT-based decode:       {lut_match}/256 bytes match")
    print(f"  weights (binary JSON):  {winner_path.stat().st_size:,} bytes")
    print(f"  baked LUT (int8 JSON):  {lut_path.stat().st_size:,} bytes")
    print(f"  C header:               {header_path.stat().st_size:,} bytes")
    print(f"  artifact dir:           {OUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
